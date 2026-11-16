// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic geometry optimization driven by analytic finite-k derivatives.
//!
//! [`optimize_fixed_cell`] moves the atoms inside an immutable lattice with
//! analytic forces. [`optimize_cell`] also relaxes the lattice vectors with
//! the analytic electronic and nuclear stress of [`crate::periodic::stress`],
//! minimizing the enthalpy under an external hydrostatic pressure.

use crate::basis::BasisSet;
use crate::error::{DftError, Result};
use crate::math::Vec3;
use crate::periodic::forces::analytic_forces;
use crate::periodic::scf::{run_periodic_scf, PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::{Cell, Lattice};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

#[derive(Clone, Debug)]
pub struct PeriodicOptOptions {
    pub max_steps: usize,
    /// Convergence threshold on max |dF/dR| in Hartree/Bohr.
    pub gtol: f64,
    /// Maximum Cartesian displacement in one accepted step, in Bohr.
    pub step_max: f64,
    pub history: usize,
    /// Optional multi-frame XYZ trajectory, flushed after every accepted step.
    pub trajectory: Option<std::path::PathBuf>,
    /// Periodic D3 dispersion added to the objective and the forces.
    pub dispersion: Option<crate::periodic::dispersion::PeriodicDispersion>,
}

impl Default for PeriodicOptOptions {
    fn default() -> Self {
        Self {
            max_steps: 200,
            gtol: 3e-4,
            step_max: 0.3,
            history: 8,
            trajectory: None,
            dispersion: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PeriodicOptFrame {
    pub step: usize,
    pub molecule: Molecule,
    /// Total energy at zero temperature, Mermin free energy under smearing.
    pub objective: f64,
    pub grad_max: f64,
}

impl PeriodicOptFrame {
    pub fn to_xyz_string(&self) -> String {
        self.molecule.to_xyz_string(&format!(
            "periodic fixed-cell step {} objective = {:.12} Hartree |g|max = {:.6e}",
            self.step, self.objective, self.grad_max
        ))
    }
}

pub struct PeriodicOptResult<Reference = PeriodicScfResult> {
    pub molecule: Molecule,
    /// Immutable cache bundle used for the final SCF result.
    pub cell: Cell,
    pub scf: Reference,
    /// Total energy at zero temperature, Mermin free energy under smearing.
    pub objective: f64,
    pub converged: bool,
    pub iterations: usize,
    pub trajectory: Vec<PeriodicOptFrame>,
}

/// Optimize atomic positions in a fixed lattice with analytic periodic forces.
///
/// The cell, AO/ECP basis caches, Ewald parameters, grid and SCF are rebuilt at
/// every trial geometry. Lattice vectors are deliberately immutable here.
pub fn optimize_fixed_cell(
    molecule: &Molecule,
    lattice: &Lattice,
    basis_set: &BasisSet,
    overlap_cutoff: f64,
    functional: &ResolvedFunctional,
    scf_options: &PeriodicScfOptions,
    options: &PeriodicOptOptions,
) -> Result<PeriodicOptResult> {
    validate_options(options)?;
    let evaluate = |geometry: &Molecule| -> Result<(f64, Vec<Vec3>, PeriodicScfResult, Cell)> {
        let cell = Cell::new(geometry.clone(), lattice.clone(), basis_set, overlap_cutoff)?;
        let scf = run_periodic_scf(&cell, functional, scf_options)?;
        let forces = analytic_forces(&cell, functional, &scf, scf_options)?;
        let mut gradient: Vec<Vec3> = forces.total.iter().map(|force| -*force).collect();
        let mut objective = if matches!(
            scf_options.occupation,
            crate::periodic::occupations::OccupationScheme::FermiDirac { .. }
        ) {
            scf.free_energy
        } else {
            scf.energy
        };
        if let Some(dispersion) =
            crate::periodic::dispersion::cell_dispersion(&cell, &options.dispersion)?
        {
            objective += dispersion.energy;
            dispersion.add_to_gradient(&mut gradient);
        }
        if !objective.is_finite()
            || gradient
                .iter()
                .any(|value| !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite())
        {
            return Err(DftError::LinearAlgebra(
                "periodic optimizer received a non-finite objective or gradient".to_string(),
            ));
        }
        Ok((objective, gradient, scf, cell))
    };

    optimize_fixed_cell_with_evaluator(molecule, options, evaluate)
}

/// Reuse the same line search and L-BFGS history with an actual stationary reference.
fn optimize_fixed_cell_with_evaluator<Reference>(
    molecule: &Molecule,
    options: &PeriodicOptOptions,
    mut evaluate: impl FnMut(&Molecule) -> Result<(f64, Vec<Vec3>, Reference, Cell)>,
) -> Result<PeriodicOptResult<Reference>> {
    validate_options(options)?;
    let natoms = molecule.len();
    let mut geometry = molecule.clone();
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();

    let (mut objective, mut gradient, mut scf, mut cell) = evaluate(&geometry)?;
    let flatten = |vectors: &[Vec3]| -> Vec<f64> {
        vectors
            .iter()
            .flat_map(|value| [value.x, value.y, value.z])
            .collect()
    };
    let mut g = flatten(&gradient);
    let mut converged = max_abs(&g) < options.gtol;
    let mut iterations = 0usize;
    let mut trajectory = Vec::new();
    if let Some(path) = &options.trajectory {
        std::fs::write(path, "")?;
    }
    record(
        &mut trajectory,
        options,
        0,
        &geometry,
        objective,
        max_abs(&g),
    )?;

    while !converged && iterations < options.max_steps {
        iterations += 1;
        let mut q = g.clone();
        let count = s_history.len();
        let mut alpha = vec![0.0; count];
        for k in (0..count).rev() {
            let sy = dot(&s_history[k], &y_history[k]);
            if !(sy.is_finite() && sy > 0.0) {
                return Err(DftError::LinearAlgebra(
                    "periodic optimizer encountered invalid L-BFGS curvature".to_string(),
                ));
            }
            let rho = 1.0 / sy;
            alpha[k] = rho * dot(&s_history[k], &q);
            axpy(&mut q, -alpha[k], &y_history[k]);
        }
        let gamma = if count == 0 {
            1.0
        } else {
            let k = count - 1;
            dot(&s_history[k], &y_history[k]) / dot(&y_history[k], &y_history[k]).max(1e-12)
        };
        for value in &mut q {
            *value *= gamma;
        }
        for k in 0..count {
            let rho = 1.0 / dot(&s_history[k], &y_history[k]);
            let beta = rho * dot(&y_history[k], &q);
            axpy(&mut q, alpha[k] - beta, &s_history[k]);
        }
        let mut step: Vec<f64> = q.iter().map(|value| -value).collect();
        let largest = max_abs(&step);
        if largest > options.step_max {
            let scale = options.step_max / largest;
            for value in &mut step {
                *value *= scale;
            }
        }

        let mut accepted = None;
        let mut trial_error = None;
        let mut evaluated_trial = false;
        let mut scale = 1.0;
        for _ in 0..8 {
            let mut trial = geometry.clone();
            for atom in 0..natoms {
                trial.atoms[atom].position.x += scale * step[3 * atom];
                trial.atoms[atom].position.y += scale * step[3 * atom + 1];
                trial.atoms[atom].position.z += scale * step[3 * atom + 2];
            }
            match evaluate(&trial) {
                Ok((next_objective, next_gradient, next_scf, next_cell)) => {
                    evaluated_trial = true;
                    let displacement: Vec<f64> = step.iter().map(|value| scale * value).collect();
                    let armijo = objective + 1e-4 * dot(&g, &displacement);
                    if next_objective <= armijo {
                        accepted = Some((
                            trial,
                            next_objective,
                            next_gradient,
                            next_scf,
                            next_cell,
                            displacement,
                        ));
                        break;
                    }
                }
                Err(error) => trial_error = Some(error),
            }
            scale *= 0.5;
        }

        if let Some((
            next_geometry,
            next_objective,
            next_gradient,
            next_scf,
            next_cell,
            displacement,
        )) = accepted
        {
            let next_g = flatten(&next_gradient);
            let y: Vec<f64> = next_g.iter().zip(&g).map(|(new, old)| new - old).collect();
            if dot(&displacement, &y) > 1e-10 {
                s_history.push(displacement);
                y_history.push(y);
                if s_history.len() > options.history {
                    s_history.remove(0);
                    y_history.remove(0);
                }
            }
            geometry = next_geometry;
            objective = next_objective;
            gradient = next_gradient;
            g = next_g;
            scf = next_scf;
            cell = next_cell;
            record(
                &mut trajectory,
                options,
                iterations,
                &geometry,
                objective,
                max_abs(&g),
            )?;
        } else {
            if !evaluated_trial {
                return Err(trial_error.unwrap_or_else(|| {
                    DftError::LinearAlgebra(
                        "periodic optimizer could not evaluate any line-search trial".to_string(),
                    )
                }));
            }
            if s_history.is_empty() {
                return Err(DftError::LinearAlgebra(
                    "periodic optimizer line search failed from the steepest-descent direction"
                        .to_string(),
                ));
            }
            s_history.clear();
            y_history.clear();
            iterations -= 1;
            continue;
        }
        converged = max_abs(&g) < options.gtol;
    }
    let _ = gradient;
    Ok(PeriodicOptResult {
        molecule: geometry,
        cell,
        scf,
        objective,
        converged,
        iterations,
        trajectory,
    })
}

/// Fixed-cell relaxation on the integer Gamma common-orbital manifold.
///
/// Each trial uses a fresh constrained SCF and its analytic Pulay gradient.
/// The optimizer, curvature rules and Armijo tests are shared with RKS/UKS.
pub fn optimize_fixed_cell_roks(
    molecule: &Molecule,
    lattice: &Lattice,
    basis_set: &BasisSet,
    overlap_cutoff: f64,
    functional: &ResolvedFunctional,
    scf_options: &PeriodicScfOptions,
    options: &PeriodicOptOptions,
) -> Result<PeriodicOptResult<crate::periodic::roks::GammaRoksResult>> {
    validate_options(options)?;
    if let Some(term) = functional.required_nonlocal_kind().filter(|term| {
        term.pair_dispersion().is_some()
            && !options
                .dispersion
                .as_ref()
                .is_some_and(|model| model.supplies(*term))
    }) {
        return Err(DftError::InvalidInput(format!(
            "common-orbital optimization requires the functional's {} correction",
            term.name()
        )));
    }
    let evaluate = |geometry: &Molecule| {
        let cell = Cell::new(geometry.clone(), lattice.clone(), basis_set, overlap_cutoff)?;
        let problem = crate::periodic::roks::GammaRoksProblem::new(&cell, functional, scf_options)?;
        if problem.max_nuclear_response_order() == 0 {
            return Err(DftError::Unsupported {
                feature: "common-orbital fixed-cell optimization with truncated AO rank".into(),
                missing: "the geometry-dependent discarded-space projector derivative".into(),
            });
        }
        let reference = problem.run()?;
        let mut gradient = problem.nuclear_gradient(&reference)?;
        let mut objective = reference.state.energy;
        if let Some(dispersion) =
            crate::periodic::dispersion::cell_dispersion(&cell, &options.dispersion)?
        {
            objective += dispersion.energy;
            dispersion.add_to_gradient(&mut gradient);
        }
        if !objective.is_finite()
            || gradient
                .iter()
                .any(|g| !g.x.is_finite() || !g.y.is_finite() || !g.z.is_finite())
        {
            return Err(DftError::LinearAlgebra(
                "periodic optimizer received a non-finite objective or gradient".into(),
            ));
        }
        Ok((objective, gradient, reference, cell))
    };
    optimize_fixed_cell_with_evaluator(molecule, options, evaluate)
}

fn validate_options(options: &PeriodicOptOptions) -> Result<()> {
    if options.max_steps == 0
        || options.history == 0
        || !(options.gtol.is_finite() && options.gtol > 0.0)
        || !(options.step_max.is_finite() && options.step_max > 0.0)
    {
        return Err(DftError::InvalidInput(
            "periodic optimization needs max_steps/history > 0 and finite positive gtol/step_max"
                .to_string(),
        ));
    }
    Ok(())
}

fn record(
    frames: &mut Vec<PeriodicOptFrame>,
    options: &PeriodicOptOptions,
    step: usize,
    molecule: &Molecule,
    objective: f64,
    grad_max: f64,
) -> Result<()> {
    let frame = PeriodicOptFrame {
        step,
        molecule: molecule.clone(),
        objective,
        grad_max,
    };
    if let Some(path) = &options.trajectory {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
        file.write_all(frame.to_xyz_string().as_bytes())?;
        file.flush()?;
    }
    frames.push(frame);
    Ok(())
}

/// Controls of [`optimize_cell`].
#[derive(Clone, Debug)]
pub struct CellOptOptions {
    pub max_steps: usize,
    /// Convergence threshold on max |dE/dR| (Hartree/Bohr).
    pub gtol: f64,
    /// Convergence threshold on max |sigma_ab + p delta_ab| (Hartree/Bohr^3).
    pub stress_tol: f64,
    /// External hydrostatic pressure `p` (Hartree/Bohr^3); the objective is
    /// the enthalpy `E + p V` (the Mermin free energy replaces `E` under
    /// Fermi-Dirac smearing).
    pub pressure: f64,
    /// Largest change of one optimization variable per step (Bohr). Cell
    /// variables are strains times the reference length `V0^(1/3)`.
    pub step_max: f64,
    pub history: usize,
    /// Periodic D3 dispersion added to the energy, forces and stress.
    pub dispersion: Option<crate::periodic::dispersion::PeriodicDispersion>,
}

impl Default for CellOptOptions {
    fn default() -> Self {
        Self {
            max_steps: 100,
            gtol: 3e-4,
            stress_tol: 5e-6,
            pressure: 0.0,
            step_max: 0.2,
            history: 8,
            dispersion: None,
        }
    }
}

/// One accepted frame of a variable-cell optimization.
#[derive(Clone, Debug)]
pub struct CellOptFrame {
    pub step: usize,
    pub lattice: Lattice,
    pub molecule: Molecule,
    /// `E + p V` (Hartree per cell).
    pub enthalpy: f64,
    /// max |dE/dR| (Hartree/Bohr).
    pub grad_max: f64,
    /// max |sigma + p I| (Hartree/Bohr^3).
    pub stress_max: f64,
}

/// Result of [`optimize_cell`].
pub struct CellOptResult {
    pub molecule: Molecule,
    pub lattice: Lattice,
    /// Immutable cache bundle of the final geometry and lattice.
    pub cell: Cell,
    pub scf: PeriodicScfResult,
    pub stress: crate::periodic::stress::PeriodicStress,
    /// Final analytic forces (Hartree/Bohr).
    pub forces: Vec<Vec3>,
    /// `E` (Mermin free energy under smearing), Hartree per cell.
    pub energy: f64,
    /// `E + p V`, Hartree per cell.
    pub enthalpy: f64,
    pub converged: bool,
    pub iterations: usize,
    pub trajectory: Vec<CellOptFrame>,
}

struct CellPoint {
    molecule: Molecule,
    lattice: Lattice,
    cell: Cell,
    scf: PeriodicScfResult,
    stress: crate::periodic::stress::PeriodicStress,
    forces: Vec<Vec3>,
    energy: f64,
    grad_max: f64,
    stress_max: f64,
}

/// Symmetric deformation `F = 1 + e` from Voigt components
/// `(xx, yy, zz, yz, xz, xy)`.
fn deformation(e: &[f64]) -> [[f64; 3]; 3] {
    [
        [1.0 + e[0], e[5], e[4]],
        [e[5], 1.0 + e[1], e[3]],
        [e[4], e[3], 1.0 + e[2]],
    ]
}

fn apply(f: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        f[0][0] * v.x + f[0][1] * v.y + f[0][2] * v.z,
        f[1][0] * v.x + f[1][1] * v.y + f[1][2] * v.z,
        f[2][0] * v.x + f[2][1] * v.y + f[2][2] * v.z,
    )
}

fn inverse(f: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
        - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
        + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
    if !(det.is_finite() && det.abs() > 1e-8) {
        return Err(DftError::LinearAlgebra(
            "cell optimizer reached a singular deformation".to_string(),
        ));
    }
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, d) = ((i + 1) % 3, (i + 2) % 3);
            inv[i][j] = (f[a][c] * f[b][d] - f[a][d] * f[b][c]) / det;
        }
    }
    Ok(inv)
}

/// Optimize atomic positions and lattice vectors together with analytic
/// finite-k forces and stress, minimizing the enthalpy `E + p V`.
///
/// The variables are the atomic positions in the reference frame and a
/// symmetric deformation `F` of the reference cell (no rigid rotation),
/// with `R = F R_ref` and `a_i = F a_i,ref`. Their gradients are
/// `F^T dE/dR` and `(dE/d(epsilon) + p V I) F^-T` contracted with
/// `dF/de`. The k mesh is the same Monkhorst-Pack mesh of each trial cell.
pub fn optimize_cell(
    molecule: &Molecule,
    lattice: &Lattice,
    basis_set: &BasisSet,
    overlap_cutoff: f64,
    functional: &ResolvedFunctional,
    scf_options: &PeriodicScfOptions,
    options: &CellOptOptions,
) -> Result<CellOptResult> {
    if options.max_steps == 0
        || options.history == 0
        || !(options.gtol.is_finite() && options.gtol > 0.0)
        || !(options.stress_tol.is_finite() && options.stress_tol > 0.0)
        || !(options.step_max.is_finite() && options.step_max > 0.0)
        || !options.pressure.is_finite()
    {
        return Err(DftError::InvalidInput(
            "cell optimization needs max_steps/history > 0, finite positive gtol/stress_tol/step_max and a finite pressure"
                .to_string(),
        ));
    }
    let natoms = molecule.len();
    let length = lattice.volume().abs().cbrt();
    let reference_rows = lattice.a;
    let reference_positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let pressure = options.pressure;
    let smeared = matches!(
        scf_options.occupation,
        crate::periodic::occupations::OccupationScheme::FermiDirac { .. }
    );

    // x = [reference-frame positions (3N, Bohr), length * e (6, Bohr)].
    let geometry = |x: &[f64]| -> (Molecule, Lattice, [[f64; 3]; 3]) {
        let e: Vec<f64> = x[3 * natoms..].iter().map(|v| v / length).collect();
        let f = deformation(&e);
        let mut trial = molecule.clone();
        for atom in 0..natoms {
            trial.atoms[atom].position =
                apply(&f, Vec3::new(x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]));
        }
        let rows = reference_rows.map(|row| apply(&f, row));
        (trial, Lattice::new(rows[0], rows[1], rows[2]), f)
    };
    let evaluate = |x: &[f64]| -> Result<(f64, Vec<f64>, CellPoint)> {
        let (trial, trial_lattice, f) = geometry(x);
        let cell = Cell::new(
            trial.clone(),
            trial_lattice.clone(),
            basis_set,
            overlap_cutoff,
        )?;
        let scf = run_periodic_scf(&cell, functional, scf_options)?;
        let mut forces = analytic_forces(&cell, functional, &scf, scf_options)?.total;
        let mut stress =
            crate::periodic::stress::analytic_stress(&cell, functional, &scf, scf_options)?;
        let mut energy = if smeared { scf.free_energy } else { scf.energy };
        if let Some(model) = &options.dispersion {
            let numbers: Vec<u8> = trial.atoms.iter().map(|a| a.z).collect();
            let positions: Vec<Vec3> = trial.atoms.iter().map(|a| a.position).collect();
            let dispersion = crate::periodic::dispersion::periodic_d3(
                &numbers,
                &positions,
                &trial_lattice,
                model,
            )?;
            energy += dispersion.energy;
            for (force, extra) in forces.iter_mut().zip(&dispersion.forces) {
                *force += *extra;
            }
            crate::periodic::dispersion::add_to_stress(&mut stress, &dispersion.strain_derivative);
        }
        let volume = trial_lattice.volume().abs();
        let enthalpy = energy + pressure * volume;
        let mut gradient = vec![0.0; 3 * natoms + 6];
        let mut grad_max = 0.0_f64;
        for (atom, force) in forces.iter().enumerate() {
            let g = [-force.x, -force.y, -force.z];
            grad_max = grad_max.max(g.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
            // d/dR_ref = F^T dE/dR.
            for a in 0..3 {
                gradient[3 * atom + a] = (0..3).map(|b| f[b][a] * g[b]).sum();
            }
        }
        // dH/dF = (dE/d(eps) + p V I) F^-T.
        let inv = inverse(&f)?;
        let mut m = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let d = stress.strain_derivative[a][c]
                        + if a == c { pressure * volume } else { 0.0 };
                    m[a][b] += d * inv[b][c];
                }
            }
        }
        let de = [
            m[0][0],
            m[1][1],
            m[2][2],
            m[1][2] + m[2][1],
            m[0][2] + m[2][0],
            m[0][1] + m[1][0],
        ];
        for (slot, value) in de.iter().enumerate() {
            gradient[3 * natoms + slot] = value / length;
        }
        let mut stress_max = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let target = if a == b { -pressure } else { 0.0 };
                stress_max = stress_max.max((stress.stress[a][b] - target).abs());
            }
        }
        if !enthalpy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "cell optimizer received a non-finite enthalpy or gradient".to_string(),
            ));
        }
        Ok((
            enthalpy,
            gradient,
            CellPoint {
                molecule: trial,
                lattice: trial_lattice,
                cell,
                scf,
                stress,
                forces,
                energy,
                grad_max,
                stress_max,
            },
        ))
    };

    let mut x: Vec<f64> = reference_positions
        .iter()
        .flat_map(|p| [p.x, p.y, p.z])
        .collect();
    x.extend([0.0; 6]);
    let (mut enthalpy, mut g, mut point) = evaluate(&x)?;
    let done = |p: &CellPoint| p.grad_max < options.gtol && p.stress_max < options.stress_tol;
    let frame = |step: usize, h: f64, p: &CellPoint| CellOptFrame {
        step,
        lattice: p.lattice.clone(),
        molecule: p.molecule.clone(),
        enthalpy: h,
        grad_max: p.grad_max,
        stress_max: p.stress_max,
    };
    let mut trajectory = vec![frame(0, enthalpy, &point)];
    let mut converged = done(&point);
    let mut iterations = 0usize;
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    while !converged && iterations < options.max_steps {
        iterations += 1;
        let mut q = g.clone();
        let count = s_history.len();
        let mut alpha = vec![0.0; count];
        for k in (0..count).rev() {
            let rho = 1.0 / dot(&s_history[k], &y_history[k]);
            alpha[k] = rho * dot(&s_history[k], &q);
            axpy(&mut q, -alpha[k], &y_history[k]);
        }
        let gamma = if count == 0 {
            1.0
        } else {
            let k = count - 1;
            dot(&s_history[k], &y_history[k]) / dot(&y_history[k], &y_history[k]).max(1e-12)
        };
        for value in &mut q {
            *value *= gamma;
        }
        for k in 0..count {
            let rho = 1.0 / dot(&s_history[k], &y_history[k]);
            let beta = rho * dot(&y_history[k], &q);
            axpy(&mut q, alpha[k] - beta, &s_history[k]);
        }
        let mut step: Vec<f64> = q.iter().map(|value| -value).collect();
        if dot(&step, &g) >= 0.0 {
            // Not a descent direction: restart from steepest descent.
            s_history.clear();
            y_history.clear();
            step = g.iter().map(|value| -value).collect();
        }
        let largest = max_abs(&step);
        if largest > options.step_max {
            let scale = options.step_max / largest;
            for value in &mut step {
                *value *= scale;
            }
        }
        let mut accepted = None;
        let mut trial_error = None;
        let mut scale = 1.0;
        for _ in 0..8 {
            let trial: Vec<f64> = x.iter().zip(&step).map(|(v, s)| v + scale * s).collect();
            match evaluate(&trial) {
                Ok((next_enthalpy, next_g, next_point)) => {
                    let displacement: Vec<f64> = step.iter().map(|value| scale * value).collect();
                    if next_enthalpy <= enthalpy + 1e-4 * dot(&g, &displacement) {
                        accepted = Some((trial, next_enthalpy, next_g, next_point, displacement));
                        break;
                    }
                }
                Err(error) => trial_error = Some(error),
            }
            scale *= 0.5;
        }
        let Some((next_x, next_enthalpy, next_g, next_point, displacement)) = accepted else {
            if s_history.is_empty() {
                return Err(trial_error.unwrap_or_else(|| {
                    DftError::LinearAlgebra(
                        "cell optimizer line search failed from the steepest-descent direction"
                            .to_string(),
                    )
                }));
            }
            s_history.clear();
            y_history.clear();
            iterations -= 1;
            continue;
        };
        let y: Vec<f64> = next_g.iter().zip(&g).map(|(new, old)| new - old).collect();
        if dot(&displacement, &y) > 1e-10 {
            s_history.push(displacement);
            y_history.push(y);
            if s_history.len() > options.history {
                s_history.remove(0);
                y_history.remove(0);
            }
        }
        x = next_x;
        enthalpy = next_enthalpy;
        g = next_g;
        point = next_point;
        trajectory.push(frame(iterations, enthalpy, &point));
        converged = done(&point);
    }
    Ok(CellOptResult {
        molecule: point.molecule,
        lattice: point.lattice,
        cell: point.cell,
        scf: point.scf,
        stress: point.stress,
        forces: point.forces,
        energy: point.energy,
        enthalpy,
        converged,
        iterations,
        trajectory,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (target, value) in y.iter_mut().zip(x) {
        *target += alpha * value;
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::periodic::forces::analytic_forces;
    use crate::system::Atom;

    #[test]
    fn invalid_options_are_rejected_before_electronic_work() {
        let mut options = PeriodicOptOptions {
            max_steps: 0,
            ..PeriodicOptOptions::default()
        };
        assert!(matches!(
            validate_options(&options),
            Err(DftError::InvalidInput(_))
        ));
        options.max_steps = 1;
        options.gtol = f64::NAN;
        assert!(matches!(
            validate_options(&options),
            Err(DftError::InvalidInput(_))
        ));
    }

    /// Classification: exact/sanity. A one-atom neutral cell is invariant to
    /// translation, so its analytic force vanishes and optimization must stop
    /// at the initial frame. The returned SCF must remain paired with the
    /// exact final Cell identity accepted by the derivative API.
    #[test]
    fn one_atom_cell_converges_without_moving_and_preserves_cell_identity() {
        let molecule = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::new(0.31, -0.27, 0.19),
        }]);
        let set = data::load("sto-3g").unwrap();
        let lattice = Lattice::cubic(18.0);
        let functional = ResolvedFunctional::hartree_fock();
        let scf_options = PeriodicScfOptions {
            max_iter: 80,
            e_tol: 1e-10,
            p_tol: 1e-9,
            conv_tol_grad: 1e-6,
            ..PeriodicScfOptions::default()
        };
        let result = optimize_fixed_cell(
            &molecule,
            &lattice,
            &set,
            6.1,
            &functional,
            &scf_options,
            &PeriodicOptOptions {
                max_steps: 4,
                gtol: 1e-7,
                ..PeriodicOptOptions::default()
            },
        )
        .unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.trajectory.len(), 1);
        for (got, want) in result.molecule.atoms.iter().zip(&molecule.atoms) {
            assert_eq!(got.position, want.position);
        }
        let final_forces =
            analytic_forces(&result.cell, &functional, &result.scf, &scf_options).unwrap();
        assert!(final_forces.total.iter().all(|force| force.norm() < 1e-7));
    }

    /// Classification: sanity. Exercise one real accepted L-BFGS step rather
    /// than only the zero-force early return. With a deliberately tiny gtol
    /// and one-step cap, the result is an explicit non-converged checkpoint;
    /// the accepted objective must nevertheless decrease and its Cell/SCF
    /// pair must remain usable by the analytic-force API.
    #[test]
    #[ignore = "multiple full periodic SCF/force evaluations; run explicitly with --ignored"]
    fn displaced_diatomic_accepts_an_energy_lowering_step() {
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -1.2),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 1.2),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let lattice = Lattice::cubic(18.0);
        let functional = ResolvedFunctional::hartree_fock();
        let scf_options = PeriodicScfOptions {
            max_iter: 80,
            e_tol: 1e-10,
            p_tol: 1e-9,
            conv_tol_grad: 1e-6,
            ..PeriodicScfOptions::default()
        };
        let result = optimize_fixed_cell(
            &molecule,
            &lattice,
            &set,
            6.1,
            &functional,
            &scf_options,
            &PeriodicOptOptions {
                max_steps: 1,
                gtol: 1e-12,
                step_max: 0.03,
                ..PeriodicOptOptions::default()
            },
        )
        .unwrap();

        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.trajectory.len(), 2);
        assert!(result.objective < result.trajectory[0].objective);
        assert!(result
            .molecule
            .atoms
            .iter()
            .zip(&molecule.atoms)
            .any(|(got, before)| (got.position - before.position).norm() > 1e-8));
        analytic_forces(&result.cell, &functional, &result.scf, &scf_options).unwrap();
    }
}
