// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! L-BFGS geometry optimization (two-loop recursion, step clipping,
//! energy-backtracking) on the analytic gradient (+ dispersion).

use crate::basis::{BasisSet, MolBasis};
use crate::dispersion::{dispersion_energy, dispersion_gradient, Dispersion};
use crate::error::{DftError, Result};
use crate::gradient::assemble_gradient;
use crate::grid::{GridOptions, MolecularGrid};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::scf::{run_scf, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

#[derive(Clone, Debug)]
pub struct OptOptions {
    pub max_steps: usize,
    /// Convergence on max |g| (Hartree/Bohr).
    pub gtol: f64,
    /// Trust-style cap on any Cartesian displacement (Bohr).
    pub step_max: f64,
    pub history: usize,
    /// Multi-frame XYZ trajectory file, written and flushed as the walk
    /// proceeds rather than at the end.
    ///
    /// Streaming matters here: a geometry optimization is the one command
    /// long enough that the user will want to watch it, and long enough
    /// that it may be interrupted. Buffering to the end would mean an
    /// interrupted run leaves nothing behind.
    pub trajectory: Option<std::path::PathBuf>,
}

impl Default for OptOptions {
    fn default() -> Self {
        Self {
            max_steps: 200,
            gtol: 3e-4,
            step_max: 0.3,
            history: 8,
            trajectory: None,
        }
    }
}

/// One accepted point on the optimization path.
#[derive(Clone, Debug)]
pub struct OptFrame {
    /// 0 for the input geometry, then one per accepted step.
    pub step: usize,
    pub molecule: Molecule,
    pub energy: f64,
    /// max |g| at this geometry (Hartree/Bohr).
    pub grad_max: f64,
}

impl OptFrame {
    /// XYZ block whose comment line carries the step, energy, and gradient
    /// norm, so the trajectory is self-describing in any viewer.
    pub fn to_xyz_string(&self) -> String {
        self.molecule.to_xyz_string(&format!(
            "step {} E = {:.12} Hartree |g|max = {:.6e}",
            self.step, self.energy, self.grad_max
        ))
    }
}

/// Accepted geometry and the evaluator's electronic state. The default
/// state preserves the existing molecular SCF API.
pub struct OptResult<State = ScfResult> {
    pub molecule: Molecule,
    pub scf: State,
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    /// Every accepted geometry including the starting one, in order.
    pub trajectory: Vec<OptFrame>,
}

/// The name used for dispersion parameter lookup comes from the resolved
/// functional (first word of its description is not reliable — pass it in).
#[allow(clippy::too_many_arguments)]
pub fn optimize(
    mol: &Molecule,
    basis_set: &BasisSet,
    scf_opts: &ScfOptions,
    functional: &ResolvedFunctional,
    functional_name: &str,
    grid_opts: Option<&GridOptions>,
    dispersion: &Dispersion,
    opts: &OptOptions,
) -> Result<OptResult> {
    let eval = |m: &Molecule| -> Result<(f64, Vec<Vec3>, ScfResult)> {
        let mb = MolBasis::build(m, basis_set)?;
        let grid = match grid_opts {
            Some(go) => Some(MolecularGrid::build(m, go)?),
            None => None,
        };
        let scf = run_scf(m, &mb, scf_opts, functional, grid.as_ref())?;
        if !scf.converged {
            return Err(DftError::ScfNotConverged {
                iterations: scf.iterations,
                error: crate::response::diagnostics::reference_diagnostics(&scf)?.orbital_residual,
            });
        }
        let mut grad = if scf_opts.density_fit.is_some() {
            crate::gradient::numerical_gradient(
                m, basis_set, scf_opts, functional, grid_opts, 1e-4,
            )?
        } else {
            assemble_gradient(m, &mb, functional, grid.as_ref(), &scf, scf_opts)?
        };
        let mut energy = scf.energy;
        if *dispersion != Dispersion::None {
            energy += dispersion_energy(m, dispersion, functional_name)?;
            let gd = dispersion_gradient(m, dispersion, functional_name)?;
            for (g, d) in grad.iter_mut().zip(gd) {
                *g += d;
            }
        }
        Ok((energy, grad, scf))
    };

    optimize_with_evaluator(mol, opts, eval)
}

/// The shared molecular L-BFGS walk for an energy and its analytic Cartesian
/// gradient. The callback also returns its converged electronic state.
/// Gradients are energy derivatives (Hartree/Bohr), not forces; coordinates
/// are Bohr. Model-specific response/transport terms belong in the evaluator.
/// Only a trial SCF convergence failure is recoverable by shortening a step;
/// invalid inputs, missing physics and resource failures remain visible.
pub fn optimize_with_evaluator<State>(
    mol: &Molecule,
    opts: &OptOptions,
    evaluate: impl FnMut(&Molecule) -> Result<(f64, Vec<Vec3>, State)>,
) -> Result<OptResult<State>> {
    optimize_with_metric(mol, opts, None, evaluate)
}

/// The same Cartesian L-BFGS walk with a fixed positive inverse-Hessian
/// preconditioner (Bohr^2/Hartree). An analytic Hessian may supply this
/// matrix after its negative/near-zero curvature is regularized for the
/// search direction only. This does not modify energies, gradients, the
/// Cartesian step cap, stopping criteria or the final physical Hessian.
pub fn optimize_with_inverse_hessian<State>(
    mol: &Molecule,
    opts: &OptOptions,
    inverse_hessian: &Matrix,
    evaluate: impl FnMut(&Molecule) -> Result<(f64, Vec<Vec3>, State)>,
) -> Result<OptResult<State>> {
    optimize_with_metric(mol, opts, Some(inverse_hessian), evaluate)
}

fn optimize_with_metric<State>(
    mol: &Molecule,
    opts: &OptOptions,
    inverse_hessian: Option<&Matrix>,
    mut evaluate: impl FnMut(&Molecule) -> Result<(f64, Vec<Vec3>, State)>,
) -> Result<OptResult<State>> {
    if mol.is_empty()
        || opts.max_steps == 0
        || opts.history == 0
        || !opts.gtol.is_finite()
        || opts.gtol <= 0.0
        || !opts.step_max.is_finite()
        || opts.step_max <= 0.0
        || mol
            .atoms
            .iter()
            .flat_map(|a| a.position.to_array())
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "geometry optimization requires finite coordinates and positive stopping, step and history controls".into(),
        ));
    }
    let natoms = mol.len();
    if let Some(metric) = inverse_hessian {
        let scale = metric
            .as_slice()
            .iter()
            .fold(0.0_f64, |s, v| s.max(v.abs()));
        if metric.rows != 3 * natoms
            || metric.cols != 3 * natoms
            || metric.as_slice().iter().any(|v| !v.is_finite())
            || metric.max_abs_diff(&metric.transpose()) > 1e-12 * scale
            || symmetric_eigen(metric)?
                .0
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(DftError::InvalidInput(
                "geometry inverse-Hessian preconditioner must be finite, symmetric, positive definite and 3N by 3N".into(),
            ));
        }
    }
    let metric_action = |vector: &[f64]| match inverse_hessian {
        Some(metric) => (0..vector.len())
            .map(|i| {
                vector
                    .iter()
                    .enumerate()
                    .map(|(j, v)| metric[(i, j)] * v)
                    .sum()
            })
            .collect::<Vec<_>>(),
        None => vector.to_vec(),
    };
    let mut geom = mol.clone();
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut eval = |m: &Molecule| {
        let (energy, gradient, state) = evaluate(m)?;
        if !energy.is_finite()
            || gradient.len() != natoms
            || gradient
                .iter()
                .flat_map(|g| g.to_array())
                .any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "geometry evaluator must return a finite energy and one finite analytic gradient per atom".into(),
            ));
        }
        Ok((energy, gradient, state))
    };
    let (mut energy, grad, mut scf) = eval(&geom)?;
    let flat = |g: &[Vec3]| -> Vec<f64> { g.iter().flat_map(|v| [v.x, v.y, v.z]).collect() };
    let mut g_flat = flat(&grad);
    let mut converged = max_abs(&g_flat) < opts.gtol;
    let mut iterations = 0;

    // Trajectory: truncate any previous file up front, then append one
    // frame per accepted step and flush, so `tail -f` and a viewer polling
    // the file both see the walk as it happens.
    let mut trajectory: Vec<OptFrame> = Vec::new();
    if let Some(path) = &opts.trajectory {
        std::fs::write(path, "")?;
    }
    let record = |frames: &mut Vec<OptFrame>, step, mol: &Molecule, e, gmax| -> Result<()> {
        let frame = OptFrame {
            step,
            molecule: mol.clone(),
            energy: e,
            grad_max: gmax,
        };
        if let Some(path) = &opts.trajectory {
            use std::io::Write;
            let mut fh = std::fs::OpenOptions::new().append(true).open(path)?;
            fh.write_all(frame.to_xyz_string().as_bytes())?;
            fh.flush()?;
        }
        frames.push(frame);
        Ok(())
    };
    record(&mut trajectory, 0, &geom, energy, max_abs(&g_flat))?;

    while !converged && iterations < opts.max_steps {
        iterations += 1;
        // L-BFGS two-loop recursion.
        let mut q = g_flat.clone();
        let m_hist = s_hist.len();
        let mut alpha = vec![0.0; m_hist];
        for k in (0..m_hist).rev() {
            let rho = 1.0 / dot(&y_hist[k], &s_hist[k]);
            alpha[k] = rho * dot(&s_hist[k], &q);
            axpy(&mut q, -alpha[k], &y_hist[k]);
        }
        // Initial Hessian scaling.
        let gamma = if m_hist > 0 {
            let k = m_hist - 1;
            dot(&s_hist[k], &y_hist[k]) / dot(&y_hist[k], &metric_action(&y_hist[k])).max(1e-12)
        } else {
            1.0
        };
        // Apply the positive metric between the two recursion sweeps.
        // Scaling by s.y/(y.M.y) preserves the secant scale in that metric.
        q = metric_action(&q);
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for k in 0..m_hist {
            let rho = 1.0 / dot(&y_hist[k], &s_hist[k]);
            let beta = rho * dot(&y_hist[k], &q);
            axpy(&mut q, alpha[k] - beta, &s_hist[k]);
        }
        // Step = −q, clipped.
        let mut step: Vec<f64> = q.iter().map(|v| -v).collect();
        let mx = max_abs(&step);
        if mx > opts.step_max {
            let sc = opts.step_max / mx;
            for v in step.iter_mut() {
                *v *= sc;
            }
        }
        // Backtracking on the total energy.
        let mut accepted = false;
        let mut scale = 1.0;
        for _ in 0..5 {
            let mut trial = geom.clone();
            for a in 0..natoms {
                trial.atoms[a].position.x += scale * step[3 * a];
                trial.atoms[a].position.y += scale * step[3 * a + 1];
                trial.atoms[a].position.z += scale * step[3 * a + 2];
            }
            match eval(&trial) {
                Ok((e_new, g_new, scf_new)) if e_new < energy + 1e-12 => {
                    // Curvature update.
                    let g_new_flat = flat(&g_new);
                    let s: Vec<f64> = step.iter().map(|v| v * scale).collect();
                    let y: Vec<f64> = g_new_flat.iter().zip(&g_flat).map(|(a, b)| a - b).collect();
                    if dot(&s, &y) > 1e-10 {
                        s_hist.push(s);
                        y_hist.push(y);
                        if s_hist.len() > opts.history {
                            s_hist.remove(0);
                            y_hist.remove(0);
                        }
                    }
                    geom = trial;
                    energy = e_new;
                    g_flat = g_new_flat;
                    scf = scf_new;
                    accepted = true;
                    break;
                }
                Err(DftError::ScfNotConverged { iterations, error }) => {
                    crate::logger::log(&format!(
                        "[geometry] rejected unconverged electronic trial: iterations={iterations} residual={error:.6e}; reducing step"
                    ));
                    scale *= 0.4;
                }
                Err(error) => return Err(error),
                Ok(_) => scale *= 0.4,
            }
        }
        if accepted {
            record(&mut trajectory, iterations, &geom, energy, max_abs(&g_flat))?;
        } else {
            // Reset the curvature history and take a tiny steepest-descent
            // step next round.
            s_hist.clear();
            y_hist.clear();
        }
        converged = max_abs(&g_flat) < opts.gtol;
    }
    Ok(OptResult {
        molecule: geom,
        scf,
        energy,
        converged,
        iterations,
        trajectory,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += a * xv;
    }
}
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}
