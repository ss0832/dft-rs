// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Hessian-based transition-state searches. P-RFO maximizes in the lowest-
//! mode partition and minimizes in its complement. RS-I-RFO instead applies
//! a Householder image transformation to that mode and minimizes the coupled
//! image function with a restricted-step RFO equation.
//!
//! Besalu and Bofill, Theor. Chem. Acc. 100, 265-274 (1998),
//! DOI 10.1007/s002140050387, is the algorithmic reference for RS-I-RFO.

use crate::basis::BasisSet;
use crate::dispersion::{dispersion_energy, dispersion_gradient, Dispersion};
use crate::error::{DftError, Result};
use crate::gradient::{assemble_gradient, numerical_gradient};
use crate::grid::{GridOptions, MolecularGrid};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::scf::{run_scf, ScfOptions};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Hessian-based algorithm used for a first-order saddle search.
pub enum SaddleMethod {
    /// Partitioned rational-function optimization: maximize along the lowest
    /// Hessian mode and minimize in its orthogonal complement.
    #[default]
    Prfo,
    /// Restricted-step image RFO: reflect the lowest mode with a Householder
    /// transformation, then minimize the coupled image function.
    RsIrfo,
}

impl SaddleMethod {
    /// Stable lowercase spelling used by CLI and Python result dictionaries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prfo => "prfo",
            Self::RsIrfo => "rsirfo",
        }
    }
}

#[derive(Clone, Debug)]
/// Convergence, trust-region, and output controls for a saddle search.
pub struct SaddleOptions {
    /// Step construction algorithm. The compatibility default is P-RFO.
    pub method: SaddleMethod,
    /// Maximum number of geometry steps.
    pub max_steps: usize,
    /// Convergence threshold for the largest Cartesian gradient component,
    /// in Hartree/Bohr.
    pub gtol: f64,
    /// Maximum Euclidean Cartesian step norm, in Bohr.
    pub step_max: f64,
    /// Central-difference displacement, in Bohr, when the complete Hessian
    /// must use its numerical fallback.
    pub hessian_step: f64,
    /// Multi-frame XYZ trajectory, appended and flushed per step. A saddle
    /// search rebuilds a complete Hessian every step, so it is the slowest
    /// walk in the program and the one most worth watching.
    pub trajectory: Option<std::path::PathBuf>,
}

impl Default for SaddleOptions {
    fn default() -> Self {
        Self {
            method: SaddleMethod::Prfo,
            max_steps: 50,
            gtol: 3e-4,
            step_max: 0.2,
            hessian_step: 5e-3,
            trajectory: None,
        }
    }
}

/// Final state and complete accepted path of a saddle search.
pub struct SaddleResult {
    /// Algorithm that generated the steps.
    pub method: SaddleMethod,
    /// Final molecular geometry in Bohr.
    pub molecule: Molecule,
    /// Electronic plus requested empirical-dispersion energy in Hartree.
    pub energy: f64,
    /// Whether the maximum Cartesian gradient met `SaddleOptions::gtol`.
    pub converged: bool,
    /// Number of geometry steps attempted.
    pub iterations: usize,
    /// Number of negative eigenvalues of the final Cartesian Hessian.
    pub n_negative: usize,
    /// Every geometry visited, in order.
    pub trajectory: Vec<crate::optimizer::OptFrame>,
}

fn validate_step_inputs(gradient: &[f64], hessian: &Matrix, step_max: f64) -> Result<()> {
    if hessian.rows != gradient.len() || hessian.cols != gradient.len() {
        return Err(DftError::InvalidInput(format!(
            "saddle step needs a square {}x{} Hessian for a {}-component gradient",
            gradient.len(),
            gradient.len(),
            gradient.len()
        )));
    }
    if !(step_max.is_finite() && step_max > 0.0) {
        return Err(DftError::InvalidInput(
            "saddle step limit must be finite and positive".into(),
        ));
    }
    if gradient.iter().any(|x| !x.is_finite()) || hessian.as_slice().iter().any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "saddle step needs finite gradient and Hessian entries".into(),
        ));
    }
    Ok(())
}

/// Compute the P-RFO step from a gradient and Hessian (both Cartesian).
/// Public for unit testing on synthetic surfaces.
pub fn prfo_step(gradient: &[f64], hessian: &Matrix, step_max: f64) -> Result<Vec<f64>> {
    validate_step_inputs(gradient, hessian, step_max)?;
    let dim = gradient.len();
    let (vals, vecs) = symmetric_eigen(hessian)?;
    // Gradient in the eigenbasis.
    let mut g = vec![0.0; dim];
    for k in 0..dim {
        for i in 0..dim {
            g[k] += vecs[(i, k)] * gradient[i];
        }
    }
    // Skip near-zero (TR) modes.
    let active: Vec<usize> = (0..dim).filter(|&k| vals[k].abs() > 1e-6).collect();
    if active.is_empty() {
        return Ok(vec![0.0; dim]);
    }
    let ts_mode = active[0];
    // Uphill shift for the TS mode: λ_p = ½λ + ½√(λ² + 4g²).
    let (l1, g1) = (vals[ts_mode], g[ts_mode]);
    let lam_p = 0.5 * l1 + 0.5 * (l1 * l1 + 4.0 * g1 * g1).sqrt();
    // Downhill shift for the rest: lowest eigenvalue of the augmented
    // [[diag λ, g], [gᵀ, 0]].
    let rest: Vec<usize> = active[1..].to_vec();
    let m = rest.len();
    let lam_n = if m > 0 {
        let mut aug = Matrix::zeros(m + 1, m + 1);
        for (a, &k) in rest.iter().enumerate() {
            aug[(a, a)] = vals[k];
            aug[(a, m)] = g[k];
            aug[(m, a)] = g[k];
        }
        symmetric_eigen(&aug)?.0[0]
    } else {
        0.0
    };
    // Step components.
    let mut step = vec![0.0; dim];
    let add_mode = |step: &mut Vec<f64>, k: usize, c: f64| {
        for i in 0..dim {
            step[i] += c * vecs[(i, k)];
        }
    };
    let denom_ts = l1 - lam_p;
    if denom_ts.abs() > 1e-12 {
        add_mode(&mut step, ts_mode, -g1 / denom_ts);
    }
    for &k in &rest {
        let denom = vals[k] - lam_n;
        if denom.abs() > 1e-12 {
            add_mode(&mut step, k, -g[k] / denom);
        }
    }
    let mx = step.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if mx > step_max {
        let sc = step_max / mx;
        for v in step.iter_mut() {
            *v *= sc;
        }
    }
    Ok(step)
}

/// Apply the image-function Householder operator for one transition vector.
///
/// With normalized `t`, `P = I - 2 t t^T`, this returns `g* = P g` and
/// `H* = P H`. When `t` is a Hessian eigenvector, `P H = H P` is symmetric
/// and flips exactly that eigenvalue. The explicit symmetrization only
/// removes eigensolver roundoff from the supplied transition vector.
pub fn householder_image(
    gradient: &[f64],
    hessian: &Matrix,
    transition: &[f64],
) -> Result<(Vec<f64>, Matrix)> {
    let dim = gradient.len();
    if hessian.rows != dim || hessian.cols != dim || transition.len() != dim {
        return Err(DftError::InvalidInput(
            "Householder image dimensions do not match".into(),
        ));
    }
    if gradient.iter().chain(transition).any(|x| !x.is_finite())
        || hessian.as_slice().iter().any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "Householder image needs finite inputs".into(),
        ));
    }
    let norm = transition.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 1e-14 {
        return Err(DftError::InvalidInput(
            "Householder transition vector must be nonzero".into(),
        ));
    }
    let t: Vec<f64> = transition.iter().map(|x| x / norm).collect();
    let tg = t.iter().zip(gradient).map(|(a, b)| a * b).sum::<f64>();
    let image_gradient: Vec<f64> = gradient
        .iter()
        .zip(&t)
        .map(|(g, ti)| g - 2.0 * ti * tg)
        .collect();

    // P H = H - 2 t (t^T H). It is symmetric when t is an eigenvector.
    let mut th = vec![0.0; dim];
    for j in 0..dim {
        for i in 0..dim {
            th[j] += t[i] * hessian[(i, j)];
        }
    }
    let mut image_hessian = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            image_hessian[(i, j)] = hessian[(i, j)] - 2.0 * t[i] * th[j];
        }
    }
    for i in 0..dim {
        for j in 0..i {
            let value = 0.5 * (image_hessian[(i, j)] + image_hessian[(j, i)]);
            image_hessian[(i, j)] = value;
            image_hessian[(j, i)] = value;
        }
    }
    Ok((image_gradient, image_hessian))
}

fn rs_rfo_step_for_metric(
    image_gradient: &[f64],
    image_hessian: &Matrix,
    metric: f64,
) -> Result<Vec<f64>> {
    let dim = image_gradient.len();
    let root_metric = metric.sqrt();
    let mut augmented = Matrix::zeros(dim + 1, dim + 1);
    for i in 0..dim {
        for j in 0..dim {
            augmented[(i, j)] = image_hessian[(i, j)] / metric;
        }
        let coupling = image_gradient[i] / root_metric;
        augmented[(i, dim)] = coupling;
        augmented[(dim, i)] = coupling;
    }
    let (_, vectors) = symmetric_eigen(&augmented)?;
    let scalar = vectors[(dim, 0)];
    if scalar.abs() < 1e-14 {
        return Err(DftError::LinearAlgebra(
            "RS-I-RFO augmented-Hessian eigenvector has a vanishing scalar component".into(),
        ));
    }
    Ok((0..dim)
        .map(|i| vectors[(i, 0)] / (root_metric * scalar))
        .collect())
}

/// Restricted-step image-function RFO step.
///
/// The lowest non-zero Hessian mode defines the Householder image. The image
/// function is minimized without partitioning, so coupling between the
/// reaction mode and its complement remains in the augmented Hessian. If the
/// unrestricted RFO solution exceeds `trust_radius` in Euclidean norm, the
/// generalized-eigenproblem metric `S = alpha I` is increased by bisection
/// until the RFO solution itself lies on the trust sphere; the result is not
/// a scaled copy of an overlong step.
pub fn rsirfo_step(gradient: &[f64], hessian: &Matrix, trust_radius: f64) -> Result<Vec<f64>> {
    validate_step_inputs(gradient, hessian, trust_radius)?;
    let dim = gradient.len();
    let (values, vectors) = symmetric_eigen(hessian)?;
    let active: Vec<usize> = (0..dim).filter(|&mode| values[mode].abs() > 1e-6).collect();
    if active.is_empty() {
        return Ok(vec![0.0; dim]);
    }
    let transition: Vec<f64> = (0..dim).map(|i| vectors[(i, active[0])]).collect();
    let (image_gradient, image_hessian) = householder_image(gradient, hessian, &transition)?;

    // Work only in the original Hessian's active subspace so the exact
    // translational/rotational null modes cannot acquire an RFO component.
    let n_active = active.len();
    let mut projected_gradient = vec![0.0; n_active];
    let mut projected_hessian = Matrix::zeros(n_active, n_active);
    for (a, &ka) in active.iter().enumerate() {
        for i in 0..dim {
            projected_gradient[a] += vectors[(i, ka)] * image_gradient[i];
        }
        for (b, &kb) in active.iter().enumerate() {
            for i in 0..dim {
                for j in 0..dim {
                    projected_hessian[(a, b)] +=
                        vectors[(i, ka)] * image_hessian[(i, j)] * vectors[(j, kb)];
                }
            }
        }
    }
    let norm = |step: &[f64]| step.iter().map(|x| x * x).sum::<f64>().sqrt();
    let unrestricted = rs_rfo_step_for_metric(&projected_gradient, &projected_hessian, 1.0)?;
    let active_step = if norm(&unrestricted) <= trust_radius {
        unrestricted
    } else {
        let mut lower = 1.0;
        let mut upper = 2.0;
        let mut upper_step =
            rs_rfo_step_for_metric(&projected_gradient, &projected_hessian, upper)?;
        while norm(&upper_step) > trust_radius && upper < 1e16 {
            lower = upper;
            upper *= 2.0;
            upper_step = rs_rfo_step_for_metric(&projected_gradient, &projected_hessian, upper)?;
        }
        if norm(&upper_step) > trust_radius {
            return Err(DftError::LinearAlgebra(
                "RS-I-RFO could not bracket the restricted-step metric".into(),
            ));
        }
        for _ in 0..100 {
            let middle = 0.5 * (lower + upper);
            let trial = rs_rfo_step_for_metric(&projected_gradient, &projected_hessian, middle)?;
            if norm(&trial) > trust_radius {
                lower = middle;
            } else {
                upper = middle;
                upper_step = trial;
            }
            if (norm(&upper_step) - trust_radius).abs() <= 1e-10 * trust_radius.max(1.0) {
                break;
            }
        }
        upper_step
    };

    let mut step = vec![0.0; dim];
    for (a, &mode) in active.iter().enumerate() {
        for i in 0..dim {
            step[i] += vectors[(i, mode)] * active_step[a];
        }
    }
    Ok(step)
}

/// Full P-RFO or RS-I-RFO search with the best complete Hessian available at
/// each step.
#[allow(clippy::too_many_arguments)]
pub fn optimize_saddle(
    mol: &Molecule,
    basis_set: &BasisSet,
    scf_opts: &ScfOptions,
    functional: &ResolvedFunctional,
    functional_name: &str,
    grid_opts: Option<&GridOptions>,
    dispersion: &Dispersion,
    opts: &SaddleOptions,
) -> Result<SaddleResult> {
    let mut geom = mol.clone();
    let mut iterations = 0;
    let mut converged = false;
    let mut energy = 0.0;
    let mut n_negative = 0;
    let mut trajectory: Vec<crate::optimizer::OptFrame> = Vec::new();
    if let Some(path) = &opts.trajectory {
        std::fs::write(path, "")?;
    }
    while iterations < opts.max_steps {
        iterations += 1;
        let mb = crate::basis::MolBasis::build(&geom, basis_set)?;
        let grid = match grid_opts {
            Some(go) => Some(MolecularGrid::build(&geom, go)?),
            None => None,
        };
        let scf = run_scf(&geom, &mb, scf_opts, functional, grid.as_ref())?;
        let mut gradient = if scf_opts.density_fit.is_some() {
            numerical_gradient(&geom, basis_set, scf_opts, functional, grid_opts, 1e-4)?
        } else {
            assemble_gradient(&geom, &mb, functional, grid.as_ref(), &scf, scf_opts)?
        };
        energy = scf.energy;
        if *dispersion != Dispersion::None {
            energy += dispersion_energy(&geom, dispersion, functional_name)?;
            let correction = dispersion_gradient(&geom, dispersion, functional_name)?;
            for (total, delta) in gradient.iter_mut().zip(correction) {
                *total += delta;
            }
        }
        let g_flat: Vec<f64> = gradient.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
        let gmax = g_flat.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let (hess, _hessian_method) = crate::hessian::hessian_with_fallback(
            &geom,
            basis_set,
            &mb,
            scf_opts,
            functional,
            functional_name,
            grid_opts,
            grid.as_ref(),
            &scf,
            dispersion,
            opts.hessian_step,
        )?;
        let (vals, _) = symmetric_eigen(&hess.hessian)?;
        n_negative = vals.iter().filter(|&&v| v < -1e-5).count();
        {
            let frame = crate::optimizer::OptFrame {
                step: iterations - 1,
                molecule: geom.clone(),
                energy,
                grad_max: gmax,
            };
            if let Some(path) = &opts.trajectory {
                use std::io::Write;
                let mut fh = std::fs::OpenOptions::new().append(true).open(path)?;
                fh.write_all(frame.to_xyz_string().as_bytes())?;
                fh.flush()?;
            }
            trajectory.push(frame);
        }
        if gmax < opts.gtol {
            converged = true;
            break;
        }
        let step = match opts.method {
            SaddleMethod::Prfo => prfo_step(&g_flat, &hess.hessian, opts.step_max)?,
            SaddleMethod::RsIrfo => rsirfo_step(&g_flat, &hess.hessian, opts.step_max)?,
        };
        for a in 0..geom.len() {
            geom.atoms[a].position.x += step[3 * a];
            geom.atoms[a].position.y += step[3 * a + 1];
            geom.atoms[a].position.z += step[3 * a + 2];
        }
    }
    Ok(SaddleResult {
        method: opts.method,
        molecule: geom,
        energy,
        converged,
        iterations,
        n_negative,
        trajectory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On an exact quadratic surface with one negative eigenvalue, the
    /// P-RFO step from any nearby point lands on the saddle (x = 0).
    #[test]
    fn prfo_finds_quadratic_saddle() {
        // H = diag(−0.5, 1.0, 2.0), E = ½xᵀHx, saddle at origin.
        let h = Matrix::from_row_major(3, 3, vec![-0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
        let mut x = [0.3, -0.4, 0.2];
        for _ in 0..40 {
            let g: Vec<f64> = (0..3).map(|i| h[(i, i)] * x[i]).collect();
            let s = prfo_step(&g, &h, 0.5).unwrap();
            for i in 0..3 {
                x[i] += s[i];
            }
        }
        for v in x {
            assert!(v.abs() < 1e-8, "did not reach the saddle: {x:?}");
        }
    }

    #[test]
    fn householder_image_flips_only_the_followed_mode() {
        let h = Matrix::from_row_major(3, 3, vec![-0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
        let g = vec![-0.15, -0.4, 0.4];
        let t = vec![1.0, 0.0, 0.0];
        let (gi, hi) = householder_image(&g, &h, &t).unwrap();
        assert_eq!(gi, vec![0.15, -0.4, 0.4]);
        assert_eq!(
            hi.as_slice(),
            &[0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]
        );
        let (g2, h2) = householder_image(&gi, &hi, &t).unwrap();
        assert_eq!(g2, g);
        assert_eq!(h2.as_slice(), h.as_slice());
    }

    #[test]
    fn rsirfo_finds_quadratic_saddle_and_obeys_trust_radius() {
        let h = Matrix::from_row_major(3, 3, vec![-0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
        let mut x = [0.3, -0.4, 0.2];
        let initial_gradient: Vec<f64> = (0..3).map(|i| h[(i, i)] * x[i]).collect();
        let restricted = rsirfo_step(&initial_gradient, &h, 0.1).unwrap();
        let length = restricted.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(length <= 0.1 + 1e-10, "restricted length = {length}");
        assert!((length - 0.1).abs() < 1e-8, "restricted length = {length}");

        for _ in 0..60 {
            let g: Vec<f64> = (0..3).map(|i| h[(i, i)] * x[i]).collect();
            let s = rsirfo_step(&g, &h, 0.2).unwrap();
            for i in 0..3 {
                x[i] += s[i];
            }
        }
        for v in x {
            assert!(v.abs() < 1e-8, "did not reach the saddle: {x:?}");
        }
    }
}
