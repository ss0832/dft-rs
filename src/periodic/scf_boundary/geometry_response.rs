// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Geometry derivatives of the dimensional collinear stationary reference.

use super::{
    run_cached_dimensional_scf, DimensionalScfBackend, DimensionalScfOptions, DimensionalScfReport,
};
use crate::basis::MolBasis;
use crate::math::Vec3;
use crate::periodic::magnetic_scf::StaticPeriodicField;
use crate::periodic::spinor_gradient::stationary::{
    add_exchange_geometry, exchange_geometry_terms, one_body_gradients, FieldFreeFrame,
};
use crate::periodic::spinor_gradient::stress::{assemble_stress, one_body_affine, StressQuality};
use crate::periodic::spinor_gradient::{DimensionalNuclearGradient, DimensionalStress};
use crate::periodic::stationary_density::StationaryDensity;
use crate::periodic::xc_gradient::{collinear_xc_geometry, XcGeometry};
use crate::periodic::PeriodicGeometry;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub(crate) struct GeometryDerivatives {
    pub gradient: Option<DimensionalNuclearGradient>,
    pub stress: Option<DimensionalStress>,
}

pub(crate) fn run_dimensional_scf_with_geometry(
    molecule: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    functional: &ResolvedFunctional,
    options: &DimensionalScfOptions,
    field: Option<&StaticPeriodicField>,
    [forces, stress]: [bool; 2],
) -> Result<(DimensionalScfReport, GeometryDerivatives)> {
    if options.scf.occupation.temperature_kelvin() != 0.0
        || options.scf.skala.is_some()
        || options.scf.pair_dispersion_from_caller && field.is_some()
        || options.scf.vv10.is_some() && field.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "dimensional collinear geometry derivative with thermal or additional nonlocal energy".into(),
            missing: "the corresponding free-energy or nonlocal geometry response".into(),
        });
    }
    crate::linalg::ensure_parallelism();
    let backend = DimensionalScfBackend::new(molecule, basis, geometry, functional, options)?;
    let backend = if let Some(field) = field {
        backend.with_static_field(field)?
    } else {
        backend
    };
    let frame = backend.stationary_geometry_frame();
    let report = run_cached_dimensional_scf(&backend, true)?;
    let reference = &report.reference;
    if !report.converged || report.reference_diagnostics.particle_number_error > options.scf.p_tol {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: report
                .reference_diagnostics
                .orbital_residual
                .max(report.reference_diagnostics.particle_number_error)
                .max(report.occupation_consistency)
                .max(report.coulomb_operator_defect)
                .max(report.grid_number_defect),
        });
    }
    let stationary = StationaryDensity::from_scf(reference);
    let mut derivatives = dimensional_geometry(
        frame,
        &stationary,
        reference.e_xc,
        StressQuality {
            orbital: report.reference_diagnostics.orbital_residual,
            consistency: report.occupation_consistency,
            number_error: report.reference_diagnostics.particle_number_error,
            energy_defect: report.component_energy_defect,
            fock_defect: report.final_fock_change,
            xc_defect: 0.0,
        },
        forces,
        stress,
    )?;
    if let Some(field) = field {
        let term = field.geometry_derivative(
            basis,
            geometry,
            stationary.k_points,
            &stationary.total_density(),
            options.basis_cutoff,
            molecule.len(),
        )?;
        let defect = (term.energy - report.external_field_energy).abs();
        if !defect.is_finite() || defect > options.operator_tolerance {
            return Err(DftError::InvalidInput(format!(
                "Fourier-field geometry contraction does not reproduce the SCF field energy. Absolute defect {defect:e} Hartree"
            )));
        }
        if let Some(stress) = &mut derivatives.stress {
            stress.include_external_field(geometry, term.strain, defect)?;
        }
        if let Some(gradient) = &mut derivatives.gradient {
            for (total, &field) in gradient.total.iter_mut().zip(&term.gradient) {
                *total += field;
            }
            gradient.external_field = Some(term.gradient);
            gradient.external_field_energy_defect = Some(defect);
        }
    }
    Ok((report, derivatives))
}

pub(crate) fn dimensional_geometry(
    frame: FieldFreeFrame<'_>,
    stationary: &StationaryDensity<'_>,
    reference_xc: f64,
    mut quality: StressQuality,
    forces: bool,
    stress: bool,
) -> Result<GeometryDerivatives> {
    let FieldFreeFrame {
        molecule,
        basis,
        geometry,
        functional,
        options,
        ..
    } = frame;
    let charge = stationary.total_density();
    let weighted: Vec<_> = stationary.energy_weighted_density[0]
        .iter()
        .zip(&stationary.energy_weighted_density[1])
        .map(|(a, b)| {
            let mut total = a.clone();
            total.add_scaled(b, 1.0);
            total
        })
        .collect();
    let points = stationary.k_points;
    let ecp = frame.ecp_geometry(points, &charge)?;
    let mut exchange = exchange_geometry_terms(molecule.len());
    for channel in stationary.density {
        add_exchange_geometry(&frame, points, channel, -0.5, &mut exchange)?;
    }
    let (xc_gradient, xc_strain, xc_energy) = if let Some(grid) = frame.grid {
        collinear_xc_geometry(
            XcGeometry::Dimensional {
                molecule,
                basis,
                geometry,
                grid_cutoff: options.grid_cutoff,
            },
            grid,
            functional,
            stationary.k_points,
            stationary.density,
            options.scf.xc_ao_cutoff.unwrap_or(options.basis_cutoff),
            stress,
        )?
    } else {
        (
            vec![Vec3::zero(); molecule.len()],
            stress.then_some([[0.0; 3]; 3]),
            0.0,
        )
    };
    let xc_defect = (xc_energy - reference_xc).abs();
    if !xc_defect.is_finite() || xc_defect > options.operator_tolerance {
        return Err(DftError::InvalidInput(
            "dimensional collinear geometry grid does not reproduce the stationary XC energy"
                .into(),
        ));
    }
    quality.xc_defect = xc_defect;
    let affine = if stress {
        Some(assemble_stress(
            geometry,
            one_body_affine(&frame, points, &charge, &weighted)?,
            frame.coulomb.affine_strain(&charge)?,
            ecp.strain,
            exchange.each_ref().map(|term| term.strain_derivative),
            xc_strain.expect("strain requested"),
            quality,
        )?)
    } else {
        None
    };
    let gradient = if forces {
        let [kinetic, pulay] = one_body_gradients(&frame, points, &charge, &weighted)?;
        let coulomb = frame.coulomb.evaluate(Some(&charge))?;
        if !coulomb.converged {
            return Err(DftError::InvalidInput(
                "dimensional collinear nuclear derivative requires converged Coulomb quadrature"
                    .into(),
            ));
        }
        let mut electrostatic = coulomb
            .energy_gradients
            .filter(|values| values.len() >= molecule.len())
            .ok_or_else(|| {
                DftError::InvalidInput("missing dimensional Coulomb nuclear derivatives".into())
            })?;
        let environment_gradient = electrostatic.split_off(molecule.len());
        let [full_exchange, long_range_exchange] = exchange.map(|term| term.gradient);
        let mut total = vec![Vec3::zero(); molecule.len()];
        for term in [
            &kinetic,
            &electrostatic,
            &ecp.gradient,
            &full_exchange,
            &long_range_exchange,
            &xc_gradient,
            &pulay,
        ] {
            for (target, &value) in total.iter_mut().zip(term) {
                *target += value;
            }
        }
        Some(DimensionalNuclearGradient {
            total,
            kinetic,
            electrostatic,
            ecp: ecp.gradient,
            full_exchange,
            long_range_exchange,
            semilocal_xc: xc_gradient,
            nonlocal_correlation: None,
            pulay,
            external_field: None,
            external_field_energy_defect: None,
            environment_gradient,
            fresh_orbital_residual: quality.orbital,
            fresh_occupation_consistency: quality.consistency,
            fresh_number_error: quality.number_error,
            reference_energy_defect: quality.energy_defect,
            reference_fock_defect: quality.fock_defect,
            xc_reference_energy_defect: quality.xc_defect,
        })
    } else {
        None
    };
    Ok(GeometryDerivatives {
        gradient,
        stress: affine,
    })
}
