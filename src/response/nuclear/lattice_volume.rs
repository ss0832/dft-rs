// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic isotropic volume derivatives, including internal relaxation.

use super::*;
use crate::periodic::gruneisen::{
    isotropic_log_volume_jets, mode_volume_derivatives, ModeVolumeDerivatives,
};

/// First and second log-volume response of primitive-cell Gamma phonons.
#[derive(Clone, Debug)]
pub struct LatticeVolumeGruneisen {
    pub spectrum: ModeVolumeDerivatives,
    /// Physical mass-weighted D, dD/dlnV, d2D/dlnV2.
    pub dynamical_jets: [Matrix; 3],
    pub acoustic: Vec<bool>,
    pub force_constants: Matrix,
    /// First and second material-coordinate internal relaxation along eta=sI.
    pub internal_first: Vec<f64>,
    pub internal_second: Vec<f64>,
    pub relaxed_ion: bool,
    /// Largest reference energy gradient with respect to material displacements.
    pub ionic_stationarity_residual: f64,
    pub pressure: f64,
    pub paths: usize,
    pub response_residuals: Vec<f64>,
    /// Geometry-only dispersion energy at the reference, Hartree per primitive cell.
    pub dispersion_energy_per_primitive_cell: Option<f64>,
    pub reference: SeriesReference,
}

/// Analytic third/fourth energy derivatives followed by spectral perturbation.
///
/// x=ln(V/V0); gamma=-dln(omega)/dx and curvature=d2ln(omega)/dx2.
/// Clamped ions retain material fractional coordinates. Relaxed ions require
/// an internally stationary and dynamically stable reference and include
/// both du/ds and d2u/ds2, for isotropic Lagrangian eta=sI. Gamma sampling
/// and finite-k supercell restrictions are inherited from the lattice series.
/// The reference must be neutral, all-electron and insulating; charged-cell
/// background derivatives and moving ECP projectors retain their refusal.
#[allow(clippy::too_many_arguments)]
pub fn lattice_volume_gruneisen(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    relaxed_ion: bool,
    ionic_tolerance: f64,
    config: &ResponseSolverConfig,
) -> Result<LatticeVolumeGruneisen> {
    lattice_volume_gruneisen_with_dispersion(
        cell,
        functional,
        result,
        options,
        replicas,
        relaxed_ion,
        ionic_tolerance,
        config,
        None,
    )
}

/// Log-volume mode derivatives including a geometry-only dispersion energy.
#[allow(clippy::too_many_arguments)]
pub fn lattice_volume_gruneisen_with_dispersion(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    replicas: usize,
    relaxed_ion: bool,
    ionic_tolerance: f64,
    config: &ResponseSolverConfig,
    dispersion: Option<&PeriodicDispersion>,
) -> Result<LatticeVolumeGruneisen> {
    config.validate()?;
    if !ionic_tolerance.is_finite() || ionic_tolerance <= 0.0 {
        return Err(DftError::InvalidInput(
            "volume Gruneisen ionic stationarity tolerance must be finite and positive".into(),
        ));
    }
    let primitive = primitive_atoms(cell, replicas)?;
    let setup = Setup::new_with_dispersion(cell, functional, result, options, dispersion)?;
    let reference = reference(&setup, result, options)?;
    let dispersion = dispersion
        .map(|model| PeriodicDispersionPath::new(cell, model))
        .transpose()?;
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: dispersion.as_ref(),
        primitive,
        replicas,
        config,
        order: 3,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let n = 3 * primitive;
    let dim = paths.dimension();
    let mut residual = 0.0_f64;
    for i in 0..n {
        residual = residual.max(paths.mixed(&[unit(dim, i)])?.abs());
    }
    if relaxed_ion && residual > ionic_tolerance {
        return Err(DftError::InvalidInput(format!(
            "relaxed volume Gruneisen response needs internal stationarity: energy gradient {residual:e} exceeds {ionic_tolerance:e} Hartree/Bohr"
        )));
    }
    let mut probe = vec![0.0; dim];
    probe[n..n + 3].fill(1.0);
    let volume = cell.lattice().volume().abs() / replicas as f64;
    let pressure = -paths.mixed(&[probe.clone()])? / (3.0 * volume);
    let mut internal_first = vec![0.0; n];
    let mut internal_second = vec![0.0; n];
    let phi = if relaxed_ion {
        let harmonic = harmonic(&mut paths, 3)?;
        if harmonic
            .smallest_optical_eigenvalue
            .is_some_and(|v| v <= 0.0)
        {
            return Err(DftError::InvalidInput(
                "relaxed volume Gruneisen response requires positive optical force constants"
                    .into(),
            ));
        }
        for i in 0..n {
            internal_first[i] = (0..3).map(|a| harmonic.relaxation[(i, a)]).sum();
            probe[i] = internal_first[i];
        }
        let mut g = vec![0.0; n];
        for (i, value) in g.iter_mut().enumerate() {
            *value = paths.mixed(&[probe.clone(), probe.clone(), unit(dim, i)])?;
        }
        for (i, value) in internal_second.iter_mut().enumerate() {
            *value = -(0..n).map(|j| harmonic.inverse[(i, j)] * g[j]).sum::<f64>();
        }
        harmonic.force_constants
    } else {
        let mut phi = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let value = paths.mixed(&[unit(dim, i), unit(dim, j)])?;
                phi[(i, j)] = value;
                phi[(j, i)] = value;
            }
        }
        phi
    };
    let mut first = Matrix::zeros(n, n);
    let mut curvature_term = Matrix::zeros(n, n);
    let mut curvature_direction = vec![0.0; dim];
    curvature_direction[..n].copy_from_slice(&internal_second);
    for i in 0..n {
        for j in i..n {
            let value = paths.mixed(&[probe.clone(), unit(dim, i), unit(dim, j)])?;
            first[(i, j)] = value;
            first[(j, i)] = value;
            if relaxed_ion && internal_second.iter().any(|v| *v != 0.0) {
                let value =
                    paths.mixed(&[curvature_direction.clone(), unit(dim, i), unit(dim, j)])?;
                curvature_term[(i, j)] = value;
                curvature_term[(j, i)] = value;
            }
        }
    }
    let first_count = paths.count;
    let mut response_residuals = std::mem::take(&mut paths.residuals);
    // Drop the third-order geometry and energy caches before building
    // fourth-order paths with the second-order orbital response.
    drop(paths);
    let mut paths = Paths {
        setup: &setup,
        reference: &reference,
        dispersion: dispersion.as_ref(),
        primitive,
        replicas,
        config,
        order: 4,
        cache: Vec::new(),
        geometries: Vec::new(),
        residuals: Vec::new(),
        field_geometry: None,
        count: 0,
    };
    let mut second = curvature_term;
    for i in 0..n {
        for j in i..n {
            let value = paths.mixed(&[probe.clone(), probe.clone(), unit(dim, i), unit(dim, j)])?;
            second[(i, j)] += value;
            second[(j, i)] = second[(i, j)];
        }
    }
    let inverse_roots: Vec<f64> = cell.molecule().atoms[..primitive]
        .iter()
        .map(|a| 1.0 / crate::constants::atomic_mass_au(a.z).sqrt())
        .collect();
    let jets = isotropic_log_volume_jets(&phi, &first, &second, &inverse_roots)?;
    let mut spectrum = mode_volume_derivatives(&jets[0], &jets[1], &jets[2])?;
    let mass: f64 = inverse_roots.iter().map(|r| 1.0 / (r * r)).sum();
    let acoustic: Vec<bool> = (0..n)
        .map(|mode| {
            (0..3)
                .map(|axis| {
                    let overlap: f64 = (0..primitive)
                        .map(|a| spectrum.modes[(3 * a + axis, mode)] / inverse_roots[a])
                        .sum();
                    overlap * overlap / mass
                })
                .sum::<f64>()
                > 0.5
        })
        .collect();
    for (branch, translation) in spectrum.branches.iter_mut().zip(&acoustic) {
        if *translation {
            branch.first_gruneisen = None;
            branch.second_log_frequency = None;
        }
    }
    response_residuals.append(&mut paths.residuals);
    let (reference, dispersion_energy_per_primitive_cell) = paths.report_reference(result)?;
    Ok(LatticeVolumeGruneisen {
        spectrum,
        dynamical_jets: jets,
        acoustic,
        force_constants: phi,
        internal_first,
        internal_second,
        relaxed_ion,
        ionic_stationarity_residual: residual,
        pressure,
        paths: first_count + paths.count,
        response_residuals,
        dispersion_energy_per_primitive_cell,
        reference,
    })
}
