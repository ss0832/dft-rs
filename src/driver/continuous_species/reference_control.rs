// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-reference full-AO controls separating static projection from AO loss.
//! No operator is installed in extra_hcore and no nuclear force API exists.

use super::gaussian::gaussian_potential;
use crate::basis::{data, MolBasis};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::nuclear_matrix;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::pseudo_sub::reduction::ReducedFragment;
use crate::pseudo_sub::reduction_kernels::{
    charge_operator, core_interactions, projector_overlaps, symmetric_projector_coefficients,
    ProjectorTemplate,
};
use crate::pseudo_sub::reduction_reference::{
    extract_methyl_frozen_space, ExtractionOptions, FrozenDensityEvaluator, FrozenReferenceModel,
    FrozenSelection, MethylBoundary,
};
use crate::scf::{
    run_scf, run_scf_in_potential, EriMode, FrozenPotential, Guess, JkEngine, Reference, ScfOptions,
};
use crate::system::Molecule;
use crate::xc::{resolve, FunctionalSpec, ResolvedFunctional};
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceControlKind {
    ExactConstrainedFrozen,
    UnprojectedStatic,
    ProjectedStatic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceControlSeed {
    ActiveReference,
    IndependentCore,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceControlStatus {
    Converged,
    Nonconverged,
    Failed,
}

#[derive(Clone, Debug)]
pub struct ReferenceControlValue {
    pub energy: f64,
    pub energy_difference_from_full: f64,
    pub iterations: usize,
    /// For the constrained control this is the physical Y-space residual.
    pub physical_orbital_residual: f64,
    pub residual_space: &'static str,
    pub active_trace: f64,
    /// ||S^(1/2) (PA-PAref) S^(1/2)||F, invariant to AO/MO rotations.
    pub metric_density_difference: f64,
    /// Tr(CB^T S PA S CB): raw electron weight in the excluded frozen space.
    pub frozen_electron_contamination: f64,
    /// Independent fresh scalar/Fock recomputation for the static controls.
    pub stored_energy_defect: Option<f64>,
    pub stored_fock_defect: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ReferenceControlOutcome {
    pub kind: ReferenceControlKind,
    pub seed: ReferenceControlSeed,
    pub status: ReferenceControlStatus,
    pub value: Option<ReferenceControlValue>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ReferenceControlReport {
    pub construction_fingerprint: String,
    pub template_id: String,
    pub full_reference_energy: f64,
    pub full_ao_count: usize,
    pub full_electrons: usize,
    pub frozen_electrons: usize,
    pub active_electrons: usize,
    /// Integer-label SCF adapter only; not the physical molecular charge.
    pub integer_adapter_charge: i32,
    pub physical_total_charge: i64,
    pub exact_static_energy_reference: f64,
    pub projected_static_energy_reference: f64,
    pub exact_reference_reconstruction_error: f64,
    pub projected_reference_reconstruction_error: f64,
    pub outcomes: Vec<ReferenceControlOutcome>,
    pub scope: &'static str,
}

fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
fn scaled(a: &Matrix, scale: f64) -> Matrix {
    let mut b = a.clone();
    for x in b.as_mut_slice() {
        *x *= scale;
    }
    b
}
fn norm(a: &Matrix) -> f64 {
    a.as_slice().iter().fold(0.0_f64, |a, x| a.hypot(*x))
}
fn largest(a: &Matrix) -> f64 {
    a.as_slice().iter().map(|x| x.abs()).fold(0.0, f64::max)
}
fn difference(a: &Matrix, b: &Matrix) -> Matrix {
    let mut d = a.clone();
    add(&mut d, b, -1.0);
    d
}

fn metric_powers(s: &Matrix) -> Result<(Matrix, Matrix)> {
    let (values, vectors) = symmetric_eigen(s)?;
    if values.iter().any(|v| *v <= 1e-10) {
        return Err(DftError::InvalidInput(
            "reference controls require a full-rank AO metric".into(),
        ));
    }
    let mut sqrt = Matrix::zeros(s.rows, s.cols);
    let mut inverse = Matrix::zeros(s.rows, s.cols);
    for k in 0..s.rows {
        for i in 0..s.rows {
            for j in 0..s.cols {
                sqrt[(i, j)] += vectors[(i, k)] * vectors[(j, k)] * values[k].sqrt();
                inverse[(i, j)] += vectors[(i, k)] * vectors[(j, k)] / values[k].sqrt();
            }
        }
    }
    Ok((sqrt, inverse))
}
fn density_metrics(
    p: &Matrix,
    selection: &FrozenSelection,
    s: &Matrix,
    sqrt: &Matrix,
) -> (f64, f64, f64) {
    let delta = difference(p, &selection.active_reference_density);
    let distance = norm(&sqrt.matmul_seq(&delta).matmul_seq(sqrt));
    let sb = s.matmul_seq(&selection.frozen_orbitals);
    let frozen = sb.transpose_matmul_seq(&p.matmul_seq(&sb));
    let contamination = (0..frozen.rows).map(|i| frozen[(i, i)]).sum();
    (p.frobenius_dot(s), distance, contamination)
}
fn failure(
    kind: ReferenceControlKind,
    seed: ReferenceControlSeed,
    error: DftError,
) -> ReferenceControlOutcome {
    ReferenceControlOutcome {
        kind,
        seed,
        status: ReferenceControlStatus::Failed,
        value: None,
        error: Some(error.to_string()),
    }
}

struct StaticContext<'a> {
    adapter: &'a Molecule,
    basis: &'a MolBasis,
    grid: &'a MolecularGrid,
    functional: &'a ResolvedFunctional,
    options: &'a ScfOptions,
    selection: &'a FrozenSelection,
    sqrt: &'a Matrix,
    inverse_sqrt: &'a Matrix,
    evaluator: &'a FrozenDensityEvaluator<'a>,
    engine: &'a JkEngine,
}

impl StaticContext<'_> {
    fn run(
        &self,
        kind: ReferenceControlKind,
        seed: ReferenceControlSeed,
        delta: &Matrix,
        classical: f64,
    ) -> ReferenceControlOutcome {
        let seeded = [
            scaled(&self.selection.active_reference_density, 0.5),
            scaled(&self.selection.active_reference_density, 0.5),
        ];
        let result = run_scf_in_potential(
            self.adapter,
            self.basis,
            self.options,
            self.functional,
            Some(self.grid),
            if seed == ReferenceControlSeed::ActiveReference {
                Some(&seeded)
            } else {
                None
            },
            None,
            Some(FrozenPotential {
                operator: delta,
                coulomb: self.engine,
                long_range: None,
                classical_energy: Some(classical),
            }),
        );
        let result = match result {
            Ok(v) => v,
            Err(e) => return failure(kind, seed, e),
        };
        let p = result.p_total();
        let fresh = match self.evaluator.evaluate(&p) {
            Ok(v) => v,
            Err(e) => return failure(kind, seed, e),
        };
        let fresh_energy = fresh.electronic_energy + p.frobenius_dot(delta) + classical;
        let mut fresh_fock = fresh.fock;
        add(&mut fresh_fock, delta, 1.0);
        let s = self.evaluator.overlap();
        let half = scaled(&p, 0.5);
        let mut comm = fresh_fock.matmul_seq(&half).matmul_seq(s);
        add(
            &mut comm,
            &s.matmul_seq(&half).matmul_seq(&fresh_fock),
            -1.0,
        );
        let residual = largest(
            &self
                .inverse_sqrt
                .matmul_seq(&comm)
                .matmul_seq(self.inverse_sqrt),
        );
        let (trace, distance, contamination) = density_metrics(&p, self.selection, s, self.sqrt);
        let stored_energy = (fresh_energy - result.energy).abs();
        let stored_fock = largest(&difference(&fresh_fock, &result.fock[0]));
        let finite = [
            fresh_energy,
            residual,
            trace,
            distance,
            contamination,
            stored_energy,
            stored_fock,
        ]
        .iter()
        .all(|x| x.is_finite());
        let converged = finite
            && result.converged
            && residual <= self.options.conv_tol_grad
            && (trace - self.selection.active_electrons as f64).abs() < 1e-9
            && stored_energy < 1e-10
            && stored_fock < 1e-9;
        ReferenceControlOutcome {
            kind,
            seed,
            status: if converged {
                ReferenceControlStatus::Converged
            } else {
                ReferenceControlStatus::Nonconverged
            },
            error: if converged {
                None
            } else {
                Some("static control did not meet fresh physical residual, trace or scalar/Fock consistency conditions".into())
            },
            value: Some(ReferenceControlValue {
                energy: fresh_energy,
                energy_difference_from_full: fresh_energy - self.selection.full_reference_energy,
                iterations: result.iterations,
                physical_orbital_residual: residual,
                residual_space: "full-AO",
                active_trace: trace,
                metric_density_difference: distance,
                frozen_electron_contamination: contamination,
                stored_energy_defect: Some(stored_energy),
                stored_fock_defect: Some(stored_fock),
            }),
        }
    }
}

/// Regenerates the explicit reference solely for a fixed-geometry diagnostic.
/// The candidate's coefficients, E0 and construction data are never refitted.
pub fn diagnose_candidate_reference(candidate: &ReducedFragment) -> Result<ReferenceControlReport> {
    candidate.validate()?;
    let source = &candidate.source;
    let basis = MolBasis::build(source, &data::load("sto-3g")?)?;
    let grid = MolecularGrid::build(
        source,
        &GridOptions {
            level: candidate.options.grid_level,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )?;
    let functional = resolve(&FunctionalSpec::Combined(
        candidate.options.functional.clone(),
    ))?;
    let options = ScfOptions {
        reference: Reference::Rhf,
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let full = run_scf(source, &basis, &options, &functional, Some(&grid))?;
    let hydrogens: Vec<_> = candidate
        .spec
        .explicit_atoms
        .iter()
        .copied()
        .filter(|&i| i != candidate.spec.port.fragment_atom)
        .collect();
    let hydrogens = hydrogens.try_into().map_err(|_| {
        DftError::InvalidInput("control requires exactly three internal methyl hydrogens".into())
    })?;
    let selection = extract_methyl_frozen_space(
        source,
        &basis,
        &full,
        MethylBoundary {
            carbon: candidate.spec.port.fragment_atom,
            hydrogens,
            port_atom: candidate.spec.port.host_atom,
        },
        &ExtractionOptions::default(),
    )?;
    let frozen =
        FrozenReferenceModel::new(source, &basis, &grid, &functional, &options, &selection)?;
    let target = frozen.static_residual_target(candidate.options.pauli_level)?;
    let s = frozen.evaluator().overlap();
    let (sqrt, inverse_sqrt) = metric_powers(s)?;
    let mut outcomes = Vec::new();
    for seed in [
        ReferenceControlSeed::ActiveReference,
        ReferenceControlSeed::IndependentCore,
    ] {
        let kind = ReferenceControlKind::ExactConstrainedFrozen;
        match frozen.solve(if seed == ReferenceControlSeed::ActiveReference {
            Some(&selection.active_reference_density)
        } else {
            None
        }) {
            Err(e) => outcomes.push(failure(kind, seed, e)),
            Ok(state) => {
                let (trace, distance, contamination) =
                    density_metrics(&state.active_density, &selection, s, &sqrt);
                outcomes.push(ReferenceControlOutcome {
                    kind,
                    seed,
                    status: ReferenceControlStatus::Converged,
                    error: None,
                    value: Some(ReferenceControlValue {
                        energy: state.evaluation.energy,
                        energy_difference_from_full: state.evaluation.energy - full.energy,
                        iterations: state.history.len(),
                        physical_orbital_residual: state.projected_orbital_residual,
                        residual_space: "orthogonal-complement",
                        active_trace: trace,
                        metric_density_difference: distance,
                        frozen_electron_contamination: contamination,
                        stored_energy_defect: None,
                        stored_fock_defect: None,
                    }),
                });
            }
        }
    }
    let coordinates = candidate.geometry_map.reference_coordinates();
    let frame = candidate
        .geometry_map
        .body_frame_kinematics(&coordinates)?
        .ok_or_else(|| DftError::InvalidInput("control requires a saved methyl frame".into()))?;
    let sites = candidate.gaussian_sites(frame.center, frame.orientation)?;
    let mut zpoint = vec![0.0; source.len()];
    zpoint[candidate.spec.port.fragment_atom] = 1.0;
    let mut projected = nuclear_matrix(&basis, source, Some(&zpoint));
    add(
        &mut projected,
        &charge_operator(&basis, source.len(), &sites)?.operator,
        1.0,
    );
    let template = ProjectorTemplate {
        s_exponents: candidate.options.s_exponents,
        p_exponents: candidate.options.p_exponents,
        orientation: frame.orientation,
    };
    let overlaps = projector_overlaps(&basis, source.len(), frame.center, &template)?;
    add(
        &mut projected,
        &overlaps
            .operator(&symmetric_projector_coefficients(
                &candidate.nonlocal_coefficients,
            )?)?
            .operator,
        1.0,
    );
    for i in 0..3 {
        add(
            &mut projected,
            &gaussian_potential(
                &basis,
                source.len(),
                frame.center,
                candidate.options.local_exponents[i],
            )?
            .operator,
            candidate.local_amplitudes[i],
        );
    }
    let cores: Vec<_> = candidate
        .spec
        .retained_atoms
        .iter()
        .map(|&i| (source.atoms[i].position, f64::from(source.atoms[i].z)))
        .collect();
    let mut core = core_interactions(&sites, &cores)?.energy;
    for (i, &(ri, zi)) in cores.iter().enumerate() {
        core += zi / (ri - frame.center).norm();
        for &(rj, zj) in &cores[..i] {
            core += zi * zj / (ri - rj).norm();
        }
    }
    let exact_delta = difference(&target.total, &target.removed_nuclear);
    let projected_delta = difference(&projected, &target.removed_nuclear);
    let exact_classical = target.core_reference_energy + target.internal_reference_energy;
    let projected_classical = core + candidate.internal_energy_reference;
    let zero = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    let evaluator =
        FrozenDensityEvaluator::new(source, &basis, &grid, &functional, &options, &zero)?;
    let adapter_charge =
        i32::try_from(i64::from(source.total_nuclear_charge()) - selection.active_electrons as i64)
            .map_err(|_| {
                DftError::InvalidInput("control integer electron adapter charge overflow".into())
            })?;
    let adapter = source
        .clone()
        .with_charge(adapter_charge)
        .with_multiplicity(1);
    let engine = JkEngine::from_options(&adapter, &basis, &options)?;
    let context = StaticContext {
        adapter: &adapter,
        basis: &basis,
        grid: &grid,
        functional: &functional,
        options: &options,
        selection: &selection,
        sqrt: &sqrt,
        inverse_sqrt: &inverse_sqrt,
        evaluator: &evaluator,
        engine: &engine,
    };
    let reference = evaluator.evaluate(&selection.active_reference_density)?;
    let exact_reconstruction = (reference.electronic_energy
        + selection
            .active_reference_density
            .frobenius_dot(&exact_delta)
        + exact_classical
        - full.energy)
        .abs();
    let projected_reconstruction = (reference.electronic_energy
        + selection
            .active_reference_density
            .frobenius_dot(&projected_delta)
        + projected_classical
        - full.energy)
        .abs();
    for (kind, delta, classical) in [
        (
            ReferenceControlKind::UnprojectedStatic,
            &exact_delta,
            exact_classical,
        ),
        (
            ReferenceControlKind::ProjectedStatic,
            &projected_delta,
            projected_classical,
        ),
    ] {
        for seed in [
            ReferenceControlSeed::ActiveReference,
            ReferenceControlSeed::IndependentCore,
        ] {
            outcomes.push(context.run(kind, seed, delta, classical));
        }
    }
    Ok(ReferenceControlReport {
        construction_fingerprint: candidate.construction_fingerprint.clone(),
        template_id: candidate.template_id.clone(),
        full_reference_energy: full.energy,
        full_ao_count: basis.nao_sph,
        full_electrons: selection.full_electrons,
        frozen_electrons: selection.frozen_electrons,
        active_electrons: selection.active_electrons,
        integer_adapter_charge: adapter_charge,
        physical_total_charge: candidate.electron_ledger.reduced_total_charge,
        exact_static_energy_reference: target.internal_reference_energy,
        projected_static_energy_reference: candidate.internal_energy_reference,
        exact_reference_reconstruction_error: exact_reconstruction,
        projected_reference_reconstruction_error: projected_reconstruction,
        outcomes,
        scope: "fixed-reference-full-AO-diagnostic; no-force-or-compressed-model-acceptance",
    })
}
