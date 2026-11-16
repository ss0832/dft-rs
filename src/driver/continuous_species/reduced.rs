// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Candidate-only production evaluation of one generated frozen methyl core.
//! This consumes fixed coefficients, never a full-fragment Fock or density.

use super::local::{LocalCenter, LocalGaussianSpec};
use super::reduction_linear::ElectronLedger;
use super::static_core::{static_core_correction, StaticCoreParameters};
use super::{ChargeChart, ChargeParameter, ContinuousSpeciesModel, SpeciesFunctional};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::overlap_matrix;
use crate::math::Vec3;
use crate::pseudo_sub::reduction::ReducedFragment;
use crate::pseudo_sub::reduction_application::BoundFragment;
use crate::pseudo_sub::reduction_basis::build_reduction_basis;
use crate::pseudo_sub::reduction_geometry::RigidExpansion;
use crate::response::ResponseSolverConfig;
use crate::scf::{EriMode, Guess, ScfOptions};
use crate::system::{Atom, Molecule};
use crate::{DftError, Result};

pub struct ReducedEvaluation {
    pub energy: f64,
    pub electronic_energy: f64,
    pub classical_energy: f64,
    /// Physical scalar before adding the state-common internal reference.
    pub energy_without_internal_reference: f64,
    pub classical_energy_without_internal_reference: f64,
    /// Already included once in the reported total and classical energies.
    pub fragment_internal_energy: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    /// Derivative of this static model energy; internal polarization absent.
    pub model_dipole: Vec3,
    pub permanent_ionic_dipole: Vec3,
    pub active_electronic_dipole: Vec3,
    pub physical_total_charge: f64,
    pub active_electrons: usize,
    pub active_density_trace: f64,
    pub frozen_electrons: usize,
    pub full_electrons: usize,
    pub ao_count: usize,
    pub grid_points: usize,
    pub occupied_spatial_orbitals: usize,
    pub response_variables: usize,
    pub scf_iterations: usize,
    pub scf_orbital_residual: f64,
    pub frontier_gap: Option<f64>,
    pub candidate_status: &'static str,
    pub shape_status: &'static str,
    pub response_status: &'static str,
    /// Response of this static model; no frozen-internal modes are added.
    pub active_model_polarizability: Option<[[f64; 3]; 3]>,
    pub response_residuals: Option<Vec<f64>>,
}

/// Evaluate a fixed candidate at rigid reduced coordinates. There is no model
/// regeneration, observable fit, frozen-density Hartree/XC addition, internal
/// response addition or empirical shape fit in this path.
pub fn evaluate_candidate(
    candidate: &ReducedFragment,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
) -> Result<ReducedEvaluation> {
    candidate.validate()?;
    evaluate_parts(
        candidate,
        &candidate.geometry_map,
        &candidate.source,
        &candidate.electron_ledger,
        coordinates,
        field,
        origin,
        None,
    )
}

/// Evaluate unchanged fragment parameters in a separately accounted rigid host.
/// The binding retains the generated internal coordinates and model provenance.
pub fn evaluate_application(
    application: &BoundFragment,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
) -> Result<ReducedEvaluation> {
    application.validate()?;
    evaluate_parts(
        application.candidate(),
        application.geometry_map(),
        application.application_source(),
        application.electron_ledger(),
        coordinates,
        field,
        origin,
        None,
    )
}

pub fn evaluate_candidate_with_response(
    candidate: &ReducedFragment,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<ReducedEvaluation> {
    candidate.validate()?;
    evaluate_parts(
        candidate,
        &candidate.geometry_map,
        &candidate.source,
        &candidate.electron_ledger,
        coordinates,
        field,
        origin,
        Some(config),
    )
}

pub fn evaluate_application_with_response(
    application: &BoundFragment,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<ReducedEvaluation> {
    application.validate()?;
    evaluate_parts(
        application.candidate(),
        application.geometry_map(),
        application.application_source(),
        application.electron_ledger(),
        coordinates,
        field,
        origin,
        Some(config),
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_parts(
    candidate: &ReducedFragment,
    map: &RigidExpansion,
    source: &Molecule,
    ledger: &ElectronLedger,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
    response_config: Option<&ResponseSolverConfig>,
) -> Result<ReducedEvaluation> {
    let frame = map.body_frame_kinematics(coordinates)?.ok_or_else(|| {
        DftError::InvalidInput("generated methyl evaluation requires its saved body frame".into())
    })?;
    let own_ion = map.electronic_center_index().ok_or_else(|| {
        DftError::InvalidInput(
            "generated methyl evaluation requires exactly one electronic center".into(),
        )
    })?;
    let mut atoms: Vec<_> = map
        .retained_atoms()
        .iter()
        .enumerate()
        .map(|(i, &full)| Atom {
            z: source.atoms[full].z,
            position: coordinates[i],
        })
        .collect();
    // The +1 ionic role label is independent of the saved fixed AO template.
    // The original carbon and three H nuclei remain only in the expansion map.
    atoms.push(Atom {
        z: 1,
        position: coordinates[own_ion],
    });
    let molecule = Molecule::new(atoms)
        .with_charge(source.charge)
        .with_multiplicity(1);
    let basis = build_reduction_basis(&molecule, own_ion, candidate.options.basis_template)?;
    let ao_count = basis.nao_sph;
    let overlap = overlap_matrix(&basis);
    let ionic: Vec<_> = molecule.atoms.iter().map(|a| f64::from(a.z)).collect();
    let parameters = StaticCoreParameters {
        charge_exponents: candidate.options.charge_exponents,
        charge_coefficients: candidate.gaussian_charges,
        body_dipoles: std::array::from_fn(|i| {
            let p = candidate.gaussian_body_dipoles[i];
            Vec3::new(p[0], p[1], p[2])
        }),
        projector_s_exponents: candidate.options.s_exponents,
        projector_p_exponents: candidate.options.p_exponents,
        nonlocal_coefficients: candidate.nonlocal_coefficients,
    };
    let correction = static_core_correction(
        &basis,
        coordinates,
        &ionic,
        own_ion,
        &frame,
        &parameters,
        origin,
    )?;
    let mut permanent_ionic_dipole = correction.permanent_dipole;
    for (&z, &r) in ionic.iter().zip(coordinates) {
        permanent_ionic_dipole += (r - origin) * z;
    }
    let active = ledger.active_electron_count;
    if active == 0 || active % 2 != 0 || active / 2 > ao_count {
        return Err(DftError::InvalidInput(
            "candidate active closed-shell sector does not fit the reduced AO template".into(),
        ));
    }
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: candidate.options.grid_level,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )?;
    let grid_points = grid.len();
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let functional = match candidate.options.functional.as_str() {
        "lda" => SpeciesFunctional::Lda,
        "pbe" => SpeciesFunctional::Pbe,
        _ => {
            return Err(DftError::InvalidInput(
                "generated candidate functional is not LDA or PBE".into(),
            ))
        }
    };
    let chart = ChargeChart {
        reference_charges: ionic,
        conserve_total_charge: true,
        parameters: vec![ChargeParameter {
            name: "fixed-generated-model".into(),
            unit: "dimensionless".into(),
            scale: 1.0,
            bounds: [-1.0, 1.0],
            charge_direction: vec![0.0; molecule.len()],
        }],
    };
    let local: Vec<_> = (0..3)
        .map(|i| LocalGaussianSpec {
            center: LocalCenter::Atom(own_ion),
            exponent: candidate.options.local_exponents[i],
            reference_amplitude: candidate.local_amplitudes[i],
            amplitude_direction: vec![0.0],
            reference_steric_strength: 0.0,
            steric_direction: vec![0.0],
            environment_exponents: vec![1.0; molecule.len()],
            environment_hardness: vec![0.0; molecule.len()],
            excluded_atoms: vec![true; molecule.len()],
        })
        .collect();
    let model = ContinuousSpeciesModel::new(
        molecule, basis, grid, active, chart, functional, options, origin,
    )?
    .with_local_potentials(local)?
    .with_static_core_correction(correction)?
    .with_fragment_internal_energy(candidate.internal_energy_reference)?;
    let state = model.evaluate(&[0.0], field, None)?;
    let trace = ledger.check_active_trace(&state.total_density(), &overlap, 1e-9)?;
    if (state.total_charge() - ledger.reduced_total_charge as f64).abs() > 1e-10 {
        return Err(DftError::InvalidInput(
            "candidate charge distribution and electron ledger disagree".into(),
        ));
    }
    let gradient = model.nuclear_gradient(&state)?;
    let forces = gradient.iter().map(|&g| g * -1.0).collect();
    let response = response_config
        .map(|config| model.sensitivity(&state, config))
        .transpose()?;
    Ok(ReducedEvaluation {
        energy: state.energy(),
        electronic_energy: state.electronic_energy(),
        classical_energy: state.classical_energy(),
        energy_without_internal_reference: state.energy_without_internal_reference(),
        classical_energy_without_internal_reference: state
            .classical_energy_without_internal_reference(),
        fragment_internal_energy: state.fragment_internal_energy(),
        gradient,
        forces,
        model_dipole: state.dipole,
        permanent_ionic_dipole,
        active_electronic_dipole: state.dipole - permanent_ionic_dipole,
        physical_total_charge: state.total_charge(),
        active_electrons: active,
        active_density_trace: trace,
        frozen_electrons: ledger.frozen_electron_count,
        full_electrons: ledger.full_electron_count,
        ao_count,
        grid_points,
        occupied_spatial_orbitals: active / 2,
        response_variables: active / 2 * (ao_count - active / 2),
        scf_iterations: state.diagnostics.iterations,
        scf_orbital_residual: state.diagnostics.orbital_residual,
        frontier_gap: state.frontier_gap(),
        candidate_status: candidate.candidate_status(),
        shape_status: candidate.shape_status(),
        response_status: candidate.response_status(),
        active_model_polarizability: response.as_ref().map(|value| value.polarizability),
        response_residuals: response.map(|value| value.response_residuals),
    })
}
