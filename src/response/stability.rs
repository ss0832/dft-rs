// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Analytic collinear orbital stability, including real-to-complex variations.
use super::tddft::{TdaSolver, TdaSpin};
use super::{ResponseReport, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilityChannel {
    InternalReal,
    ExternalSpin,
    ExternalComplex,
    UnrestrictedInternal,
    ExternalGeneralized,
    ExternalComplexSpin,
    UnrestrictedComplex,
}
impl StabilityChannel {
    pub fn id(self) -> &'static str {
        match self {
            Self::InternalReal => "internal-real",
            Self::ExternalSpin => "external-spin",
            Self::ExternalComplex => "external-complex",
            Self::UnrestrictedInternal => "unrestricted-internal",
            Self::ExternalGeneralized => "external-generalized",
            Self::ExternalComplexSpin => "external-complex-spin",
            Self::UnrestrictedComplex => "unrestricted-complex",
        }
    }
    pub fn is_internal(self) -> bool {
        matches!(self, Self::InternalReal | Self::UnrestrictedInternal)
    }
    pub fn from_id(id: &str) -> Result<Self> {
        [
            Self::InternalReal,
            Self::ExternalSpin,
            Self::ExternalComplex,
            Self::UnrestrictedInternal,
            Self::ExternalGeneralized,
            Self::ExternalComplexSpin,
            Self::UnrestrictedComplex,
        ]
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| DftError::InvalidInput(format!("unknown stability channel: {id}")))
    }
}
#[derive(Clone, Debug)]
pub struct StabilityOptions {
    pub n_roots: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_subspace: usize,
    pub block_buffer: usize,
    pub resolution: Option<f64>,
    pub channels: Vec<StabilityChannel>,
    pub verify_lowest: bool,
    pub verify_steps: usize,
}
impl Default for StabilityOptions {
    fn default() -> Self {
        Self {
            n_roots: 3,
            tolerance: 1e-7,
            max_iterations: 200,
            max_subspace: 60,
            block_buffer: 3,
            resolution: None,
            channels: Vec::new(),
            verify_lowest: true,
            verify_steps: 24,
        }
    }
}
impl StabilityOptions {
    pub fn resolved_resolution(&self) -> f64 {
        self.resolution.unwrap_or(10.0 * self.tolerance)
    }
    pub fn validate(&self) -> Result<()> {
        let r = self.resolved_resolution();
        if self.n_roots == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || self.tolerance >= 1.0
            || self.max_iterations == 0
            || self.max_subspace
                < self
                    .n_roots
                    .saturating_add(self.block_buffer)
                    .saturating_add(1)
            || !r.is_finite()
            || r < self.tolerance
            || (self.verify_lowest && self.verify_steps == 0)
            || self
                .channels
                .iter()
                .enumerate()
                .any(|(i, c)| self.channels[..i].contains(c))
        {
            return Err(DftError::InvalidInput(
                "invalid stability controls; resolution must be at least the positive \
                eigensolver tolerance"
                    .into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct StabilityExcitation {
    pub from: usize,
    pub to: usize,
    pub amplitude: f64,
    pub spin: &'static str,
}

#[derive(Clone, Debug)]
pub struct StabilityMode {
    /// Eigenvalue of A+B (real rotations) or A-B (imaginary rotations), Hartree.
    pub eigenvalue: f64,
    /// d^2 E/d theta^2 along the normalized reported orbital generator.
    pub energy_curvature: f64,
    /// Virtual-major, occupied-minor; alpha followed by beta for UKS/UHF.
    pub amplitudes: Vec<f64>,
    pub absolute_residual: f64,
    pub relative_residual: f64,
    pub iterations: usize,
}
#[derive(Clone, Debug)]
pub struct StabilityBlock {
    pub channel: StabilityChannel,
    pub name: &'static str,
    pub dimension: usize,
    pub classification: &'static str,
    pub modes: Vec<StabilityMode>,
    pub hermiticity_defect: f64,
    pub converged: bool,
    pub resolution: f64,
    pub matvec_count: usize,
    pub dominant: Vec<StabilityExcitation>,
    /// Occupied-major ordering, matching the public excitation indices.
    pub eigenvector: Vec<f64>,
    pub n_occupied: usize,
    pub n_virtual: usize,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct MolecularStability {
    pub blocks: Vec<StabilityBlock>,
    pub classification: &'static str,
    pub zero_tolerance: f64,
    pub curvature_factor: f64,
    pub stable: bool,
    pub unrestricted: bool,
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |r, x| r.hypot(*x))
}
fn append(mut v: Vec<f64>, basis: &mut Vec<Vec<f64>>) -> bool {
    let n = norm(&v);
    if n == 0.0 {
        return false;
    }
    for x in &mut v {
        *x /= n;
    }
    for _ in 0..2 {
        for b in basis.iter() {
            let c = dot(b, &v);
            for (x, y) in v.iter_mut().zip(b) {
                *x -= c * y;
            }
        }
    }
    let n = norm(&v);
    if n < 1e-12 {
        return false;
    }
    for x in &mut v {
        *x /= n;
    }
    basis.push(v);
    true
}
/// Block Davidson; only small projected matrices are constructed. Denominator
/// regularization is confined to preconditioning, never the physical operator.
fn lowest_modes(
    diagonal: &[f64],
    roots: usize,
    factor: f64,
    config: &ResponseSolverConfig,
    options: &StabilityOptions,
    apply: impl Fn(&[f64]) -> Result<Vec<f64>>,
) -> Result<(Vec<StabilityMode>, f64, bool, usize)> {
    let n = diagonal.len();
    let roots = roots.min(n);
    if roots == 0 {
        return Ok((Vec::new(), 0.0, true, 0));
    }
    let count = std::cell::Cell::new(0usize);
    let image = |x: &[f64]| -> Result<Vec<f64>> {
        count.set(count.get() + 1);
        let y = apply(x)?;
        if y.len() != n || y.iter().any(|x| !x.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "invalid stability operator image".into(),
            ));
        }
        Ok(y)
    };
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| diagonal[i].total_cmp(&diagonal[j]));
    let mut basis = Vec::new();
    let max_space = options.max_subspace.min(n);
    for &i in order
        .iter()
        .take((roots + options.block_buffer).min(max_space).min(n))
    {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        append(e, &mut basis);
    }
    // Full-support seeds sample blocks that do not couple to the lowest-gap
    // coordinate seeds (including an uncoupled negative-curvature block).
    for seed in 1..=2 {
        if basis.len() >= max_space {
            break;
        }
        append(
            (0..n)
                .map(|i| ((i + 1) as f64 * seed as f64 * 1.61803398875).sin())
                .collect(),
            &mut basis,
        );
    }
    let mut sigma = Vec::<Vec<f64>>::new();
    let mut defect = 0.0_f64;
    let mut last_modes = Vec::new();
    for iteration in 1..=config.max_iterations {
        while sigma.len() < basis.len() {
            sigma.push(image(&basis[sigma.len()])?);
        }
        let m = basis.len();
        let mut h = Matrix::zeros(m, m);
        for p in 0..m {
            for q in 0..m {
                h[(p, q)] = dot(&basis[p], &sigma[q]);
            }
        }
        let scale = h.as_slice().iter().fold(1.0_f64, |r, x| r.max(x.abs()));
        for p in 0..m {
            for q in p + 1..m {
                defect = defect.max((h[(p, q)] - h[(q, p)]).abs());
                let a = 0.5 * (h[(p, q)] + h[(q, p)]);
                h[(p, q)] = a;
                h[(q, p)] = a;
            }
        }
        if defect > 100.0 * config.tolerance * scale {
            return Err(DftError::LinearAlgebra(format!(
                "stability operator is not Hermitian: defect {defect:e}"
            )));
        }
        let (values, vectors) = symmetric_eigen(&h)?;
        let ritz = |r: usize| -> Vec<f64> {
            (0..n)
                .map(|k| (0..m).map(|p| vectors[(p, r)] * basis[p][k]).sum())
                .collect()
        };
        let mut modes = Vec::new();
        let mut directions = Vec::new();
        for r in 0..roots {
            let x = ritz(r);
            // Fresh full operator application is the acceptance criterion.
            let mut residual = image(&x)?;
            for (y, x) in residual.iter_mut().zip(&x) {
                *y -= values[r] * x;
            }
            let absolute = norm(&residual);
            let relative = absolute / values[r].abs().max(1.0);
            modes.push(StabilityMode {
                eigenvalue: values[r],
                energy_curvature: factor * values[r],
                amplitudes: x,
                absolute_residual: absolute,
                relative_residual: relative,
                iterations: iteration,
            });
            if absolute > config.tolerance {
                for k in 0..n {
                    let d = diagonal[k] - values[r];
                    residual[k] /= d.abs().max(config.preconditioner_floor).copysign(d);
                }
                directions.push(residual);
            }
        }
        if directions.is_empty() {
            let mut accepted = true;
            if options.verify_lowest && m < n {
                let mut best = modes[0].clone();
                // Independent steepest Rayleigh descent, with exact two-vector
                // minimizations. It does not use the Davidson basis or diagonal.
                // The multiplier is an arbitrary fixed number for reproducible
                // start vectors; it is deliberately not `SQRT_2`.
                #[allow(clippy::approx_constant)]
                let probes: Vec<Vec<f64>> = (1..=4)
                    .map(|seed| {
                        (0..n)
                            .map(|i| ((i + 1) as f64 * (seed as f64 + 0.3) * 1.41421356237).cos())
                            .collect()
                    })
                    .collect();
                for mut x in probes {
                    let q = norm(&x);
                    for v in &mut x {
                        *v /= q;
                    }
                    for _ in 0..options.verify_steps {
                        let ax = image(&x)?;
                        let value = dot(&x, &ax);
                        let mut g: Vec<_> = ax.iter().zip(&x).map(|(a, x)| a - value * x).collect();
                        let ng = norm(&g);
                        if value < best.eigenvalue {
                            best = StabilityMode {
                                eigenvalue: value,
                                energy_curvature: factor * value,
                                amplitudes: x.clone(),
                                absolute_residual: ng,
                                relative_residual: ng / value.abs().max(1.0),
                                iterations: iteration,
                            };
                        }
                        if ng <= config.tolerance {
                            break;
                        }
                        for v in &mut g {
                            *v /= ng;
                        }
                        let ag = image(&g)?;
                        let small = Matrix::from_row_major(2, 2, vec![value, ng, ng, dot(&g, &ag)]);
                        let (_, v) = symmetric_eigen(&small)?;
                        for k in 0..n {
                            x[k] = v[(0, 0)] * x[k] + v[(1, 0)] * g[k];
                        }
                        let q = norm(&x);
                        for v in &mut x {
                            *v /= q;
                        }
                    }
                }
                if best.eigenvalue < modes[0].eigenvalue - options.resolved_resolution() {
                    modes[0] = best;
                    accepted = false;
                }
            }
            return Ok((modes, defect, accepted, count.get()));
        }
        last_modes = modes;
        let retained: Vec<_> = (0..(roots + options.block_buffer).min(m))
            .map(ritz)
            .collect();
        if basis.len() + directions.len() > max_space && basis.len() < n {
            basis.clear();
            sigma.clear();
            for x in retained {
                append(x, &mut basis);
            }
        }
        let mut added = false;
        for x in directions {
            if basis.len() < max_space {
                added |= append(x, &mut basis);
            }
        }
        if !added {
            for &i in &order {
                let mut e = vec![0.0; n];
                e[i] = 1.0;
                if append(e, &mut basis) {
                    added = true;
                    break;
                }
            }
        }
        if !added {
            break;
        }
    }
    Ok((last_modes, defect, false, count.get()))
}

#[allow(clippy::too_many_arguments)]
pub fn molecular_stability(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    roots: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<MolecularStability>> {
    config.validate()?;
    let options = StabilityOptions {
        n_roots: roots,
        tolerance: config.tolerance,
        max_iterations: config.max_iterations,
        ..Default::default()
    };
    stability_from_state(mol, mb, functional, grid, scf, opts, &options)
}

#[allow(clippy::too_many_arguments)]
pub fn stability_from_state(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    options: &StabilityOptions,
) -> Result<ResponseReport<MolecularStability>> {
    options.validate()?;
    let reference = super::diagnostics::reference_diagnostics(scf)?;
    if !reference.converged {
        return Err(DftError::InvalidInput(
            "stability needs a converged SCF reference".into(),
        ));
    }
    let channels = if options.channels.is_empty() {
        if scf.unrestricted {
            vec![StabilityChannel::UnrestrictedInternal]
        } else {
            vec![
                StabilityChannel::InternalReal,
                StabilityChannel::ExternalSpin,
                StabilityChannel::ExternalComplex,
            ]
        }
    } else {
        options.channels.clone()
    };
    let factor = if scf.unrestricted { 2.0 } else { 4.0 };
    let zero = options.resolved_resolution();
    let config = ResponseSolverConfig {
        tolerance: options.tolerance,
        max_iterations: options.max_iterations,
        ..Default::default()
    };
    let mut blocks = Vec::new();
    for channel in channels {
        let (spin, imag) = match channel {
            StabilityChannel::InternalReal => (TdaSpin::Singlet, false),
            StabilityChannel::ExternalSpin => (TdaSpin::Triplet, false),
            StabilityChannel::ExternalComplex => (TdaSpin::Singlet, true),
            StabilityChannel::ExternalComplexSpin => (TdaSpin::Triplet, true),
            StabilityChannel::UnrestrictedInternal => (TdaSpin::SpinResolved, false),
            StabilityChannel::UnrestrictedComplex => (TdaSpin::SpinResolved, true),
            StabilityChannel::ExternalGeneralized => {
                return Err(DftError::InvalidInput(
                    "external-generalized stability needs a noncollinear spin-flip reference \
                    and kernel; this collinear driver does not compute that space"
                        .into(),
                ))
            }
        };
        if scf.unrestricted != matches!(spin, TdaSpin::SpinResolved) {
            return Err(DftError::InvalidInput(format!(
                "stability channel {} does not fit the chosen restricted/unrestricted reference",
                channel.id()
            )));
        }
        let solver = TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?;
        let diagonal = solver.diag();
        let (modes, defect, converged, matvec_count) =
            lowest_modes(&diagonal, options.n_roots, factor, &config, options, |x| {
                if imag {
                    solver.amb_vec(x)
                } else {
                    solver.apb_vec(x)
                }
            })?;
        let classification = match modes.first() {
            None => "empty",
            Some(m) if m.eigenvalue < -zero => "unstable",
            Some(m) if m.eigenvalue <= zero => "indefinite",
            Some(_) => "stable",
        };
        let mut warnings = Vec::new();
        if !converged {
            warnings.push(format!(
                "WARNING [stability] {}: Davidson did NOT converge or its lowest root \
                was overturned by independent Rayleigh descent; residual {:.6e}, \
                iterations {}",
                channel.id(),
                modes
                    .iter()
                    .map(|m| m.absolute_residual)
                    .fold(0.0_f64, f64::max),
                modes.iter().map(|m| m.iterations).max().unwrap_or(0)
            ));
        }
        if classification == "unstable" {
            let action = match channel {
                StabilityChannel::ExternalSpin => {
                    "restricted solution is a saddle with respect to spin polarization; \
                    re-run unrestricted"
                }
                StabilityChannel::ExternalComplex
                | StabilityChannel::ExternalComplexSpin
                | StabilityChannel::UnrestrictedComplex => {
                    "real-orbital solution is a saddle; choose a complex-orbital reference"
                }
                _ => {
                    "reference is an internal saddle; re-converge from a rotated guess with \
                    stab-following"
                }
            };
            warnings.push(format!(
                "WARNING [stability] {}: lowest {:.6e} hartree; {action}",
                channel.id(),
                modes[0].eigenvalue
            ));
        } else if classification == "indefinite" {
            warnings.push(format!(
                "WARNING [stability] {}: sign unresolved at resolution {zero:.6e} hartree",
                channel.id()
            ));
        }
        let mut eigenvector = Vec::new();
        let mut dominant = Vec::new();
        let mut offset = 0;
        let spins: &[usize] = if scf.unrestricted { &[0, 1] } else { &[0] };
        for &sp in spins {
            let no = if sp == 0 { scf.n_alpha } else { scf.n_beta };
            let nv = scf.mo_coeff[sp].cols - no;
            if let Some(mode) = modes.first() {
                for i in 0..no {
                    for a in 0..nv {
                        let x = mode.amplitudes[offset + a * no + i];
                        eigenvector.push(x);
                        if x.abs() >= 0.1 {
                            dominant.push(StabilityExcitation {
                                from: i,
                                to: no + a,
                                amplitude: x,
                                spin: if scf.unrestricted {
                                    if sp == 0 {
                                        "alpha"
                                    } else {
                                        "beta"
                                    }
                                } else if spin == TdaSpin::Triplet {
                                    "triplet"
                                } else {
                                    "singlet"
                                },
                            });
                        }
                    }
                }
            }
            offset += no * nv;
        }
        dominant.sort_by(|a, b| b.amplitude.abs().total_cmp(&a.amplitude.abs()));
        dominant.truncate(6);
        blocks.push(StabilityBlock {
            channel,
            name: channel.id(),
            dimension: diagonal.len(),
            classification,
            modes,
            hermiticity_defect: defect,
            converged,
            resolution: zero,
            matvec_count,
            dominant,
            eigenvector,
            n_occupied: scf.n_alpha,
            n_virtual: scf.mo_coeff[0].cols - scf.n_alpha,
            warnings,
        });
    }
    let classification = if blocks.iter().any(|b| b.classification == "unstable") {
        "unstable"
    } else if blocks
        .iter()
        .any(|b| b.classification == "indefinite" || !b.converged)
    {
        "indefinite"
    } else {
        "stable"
    };
    let stable = classification == "stable";
    let residuals = blocks
        .iter()
        .flat_map(|b| b.modes.iter().map(|m| m.relative_residual))
        .collect();
    let iterations = blocks
        .iter()
        .flat_map(|b| b.modes.iter().map(|m| m.iterations))
        .collect();
    let mut warnings: Vec<_> = blocks
        .iter()
        .flat_map(|b| b.warnings.iter().cloned())
        .collect();
    let converged = blocks.iter().all(|b| b.converged)
        && reference.orbital_residual <= options.tolerance.max(opts.conv_tol_grad) * 10.0
        && reference.projector_defect <= options.tolerance * 100.0
        && reference.orthonormality_defect <= options.tolerance * 100.0;
    if !converged {
        warnings.push(
            "WARNING [stability] analysis or reference residuals exceed acceptance criteria".into(),
        );
    }
    Ok(ResponseReport {
        shape: vec![blocks.len()],
        value: MolecularStability {
            blocks,
            classification,
            zero_tolerance: zero,
            curvature_factor: factor,
            stable,
            unrestricted: scf.unrestricted,
        },
        functional: functional.description.clone(),
        reference,
        units: "hartree",
        model: "adiabatic collinear orbital energy Hessian; real and imaginary \
            occupied-virtual rotations",
        axis_convention: "channel then ascending mode; eigenvector occupied-major virtual-minor; \
            mode amplitudes virtual-major occupied-minor; unrestricted alpha then \
            beta",
        converged,
        residuals,
        iterations,
        warnings,
    })
}

#[derive(Clone, Debug)]
pub struct StabilizeOptions {
    pub analysis: StabilityOptions,
    pub max_descents: usize,
    pub step_angle: f64,
    pub min_energy_drop: f64,
}
impl Default for StabilizeOptions {
    fn default() -> Self {
        Self {
            analysis: StabilityOptions::default(),
            max_descents: 3,
            step_angle: 0.3,
            min_energy_drop: 1e-8,
        }
    }
}
impl StabilizeOptions {
    pub fn validate(&self) -> Result<()> {
        self.analysis.validate()?;
        if self.max_descents == 0
            || !self.step_angle.is_finite()
            || self.step_angle <= 0.0
            || !self.min_energy_drop.is_finite()
            || self.min_energy_drop <= 0.0
        {
            return Err(DftError::InvalidInput(
                "stability following needs positive descent count, angle and minimum energy drop"
                    .into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct DescentStep {
    pub channel: StabilityChannel,
    pub eigenvalue: f64,
    pub theta: f64,
    pub energy_before: f64,
    pub energy_after: f64,
}
#[derive(Clone, Debug)]
pub struct StabilizeReport {
    pub initial_energy: f64,
    pub initial: ResponseReport<MolecularStability>,
    pub descents: Vec<DescentStep>,
    pub final_analysis: ResponseReport<MolecularStability>,
    pub final_state: ScfResult,
    pub stabilized: bool,
    pub converged: bool,
    pub warnings: Vec<String>,
}

/// Exact orthogonal exponential of a general occupied-virtual generator.
/// The SVD is avoided: diagonalize -G^2 and evaluate exp(G)=cos(sqrt(-G^2))
/// +G sinc(sqrt(-G^2)). Zero singular values use their analytic sinc limit.
pub fn rotated_density(
    scf: &ScfResult,
    channel: StabilityChannel,
    amplitudes: &[f64],
    theta: f64,
) -> Result<[Matrix; 2]> {
    if !channel.is_internal() || !theta.is_finite() {
        return Err(DftError::InvalidInput(
            "orbital following is defined for internal real channels and finite angles".into(),
        ));
    }
    if scf.unrestricted != (channel == StabilityChannel::UnrestrictedInternal) {
        return Err(DftError::InvalidInput(
            "orbital following channel disagrees with the reference class".into(),
        ));
    }
    let spins = if scf.unrestricted { 2 } else { 1 };
    let mut offset = 0;
    let expected: usize = (0..spins)
        .map(|sp| {
            let no = if sp == 0 { scf.n_alpha } else { scf.n_beta };
            no * (scf.mo_coeff[sp].cols - no)
        })
        .sum();
    if amplitudes.len() != expected || amplitudes.iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "orbital following has invalid amplitude dimensions or values".into(),
        ));
    }
    let nao = scf.overlap.rows;
    let mut density = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    for sp in 0..spins {
        let no = if sp == 0 { scf.n_alpha } else { scf.n_beta };
        let n = scf.mo_coeff[sp].cols;
        let nv = n - no;
        let mut g = Matrix::zeros(n, n);
        for a in 0..nv {
            for i in 0..no {
                let v = theta * amplitudes[offset + a * no + i];
                g[(no + a, i)] = v;
                g[(i, no + a)] = -v;
            }
        }
        let mut square = g.matmul_seq(&g);
        for x in square.as_mut_slice() {
            *x = -*x;
        }
        let (vals, v) = symmetric_eigen(&square)?;
        let mut cos = Matrix::zeros(n, n);
        let mut sinc = Matrix::zeros(n, n);
        for k in 0..n {
            if vals[k] < -1e-10 {
                return Err(DftError::LinearAlgebra(
                    "orbital rotation generator has invalid squared spectrum".into(),
                ));
            }
            let w = vals[k].max(0.0).sqrt();
            let a = w.cos();
            let b = if w == 0.0 { 1.0 } else { w.sin() / w };
            for i in 0..n {
                for j in 0..n {
                    let p = v[(i, k)] * v[(j, k)];
                    cos[(i, j)] += a * p;
                    sinc[(i, j)] += b * p;
                }
            }
        }
        let gs = g.matmul_seq(&sinc);
        for (x, y) in cos.as_mut_slice().iter_mut().zip(gs.as_slice()) {
            *x += y;
        }
        let c = scf.mo_coeff[sp].matmul_seq(&cos);
        density[sp] = c.leading_columns_gram(no, 1.0);
        offset += no * nv;
    }
    if !scf.unrestricted {
        density[1] = density[0].clone();
    }
    Ok(density)
}

#[allow(clippy::too_many_arguments)]
pub fn stabilize(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    input: &ScfResult,
    opts: &ScfOptions,
    options: &StabilizeOptions,
) -> Result<StabilizeReport> {
    options.validate()?;
    let initial = stability_from_state(mol, mb, functional, grid, input, opts, &options.analysis)?;
    let mut analysis = initial.clone();
    let mut state = input.clone();
    let mut descents = Vec::new();
    let mut warnings = initial.warnings.clone();
    let mut converged = initial.converged;
    let mut trial_options = opts.clone();
    trial_options.reference = if input.unrestricted {
        crate::scf::Reference::Uhf
    } else {
        crate::scf::Reference::Rhf
    };
    for _ in 0..options.max_descents {
        if !analysis.converged {
            break;
        }
        let block = analysis
            .value
            .blocks
            .iter()
            .filter(|b| b.channel.is_internal() && b.classification == "unstable")
            .min_by(|a, b| a.modes[0].eigenvalue.total_cmp(&b.modes[0].eigenvalue));
        let Some(block) = block else {
            break;
        };
        let mode = &block.modes[0];
        let mut best: Option<(ScfResult, f64)> = None;
        for theta in [options.step_angle, -options.step_angle] {
            let density = rotated_density(&state, block.channel, &mode.amplitudes, theta)?;
            let trial = crate::scf::run_scf_with_density(
                mol,
                mb,
                &trial_options,
                functional,
                grid,
                Some(&density),
            )?;
            if !trial.converged {
                warnings.push(format!(
                    "WARNING [stability] rotated SCF at theta {theta:.6e} did NOT converge"
                ));
                continue;
            }
            if state.energy - trial.energy >= options.min_energy_drop
                && best
                    .as_ref()
                    .is_none_or(|(old, _)| trial.energy < old.energy)
            {
                best = Some((trial, theta));
            }
        }
        let Some((trial, theta)) = best else {
            warnings.push(
                "WARNING [stability] following stopped: neither signed rotation lowered \
                the reconverged energy by the required amount"
                    .into(),
            );
            break;
        };
        descents.push(DescentStep {
            channel: block.channel,
            eigenvalue: mode.eigenvalue,
            theta,
            energy_before: state.energy,
            energy_after: trial.energy,
        });
        state = trial;
        analysis =
            stability_from_state(mol, mb, functional, grid, &state, opts, &options.analysis)?;
        converged &= analysis.converged;
        warnings.extend(analysis.warnings.iter().cloned());
    }
    let stabilized = analysis.value.stable && analysis.converged;
    if !stabilized {
        warnings.push(
            "WARNING [stability] final reference is not proven stable in every \
            requested channel; external channels require an explicit change of \
            reference class"
                .into(),
        );
    }
    Ok(StabilizeReport {
        initial_energy: input.energy,
        initial,
        descents,
        final_analysis: analysis,
        final_state: state,
        stabilized,
        converged,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn iteration_limit_returns_fresh_residuals_without_claiming_convergence() {
        let n = 40;
        let diagonal = vec![1.0; n];
        let options = StabilityOptions {
            n_roots: 1,
            max_iterations: 1,
            block_buffer: 0,
            max_subspace: 8,
            ..Default::default()
        };
        let config = ResponseSolverConfig {
            max_iterations: 1,
            ..Default::default()
        };
        let (modes, _, converged, count) =
            lowest_modes(&diagonal, 1, 2.0, &config, &options, |x| {
                Ok((0..n)
                    .map(|i| {
                        (i as f64 + 1.0) * x[i]
                            + if i > 0 { 0.2 * x[i - 1] } else { 0.0 }
                            + if i + 1 < n { 0.2 * x[i + 1] } else { 0.0 }
                    })
                    .collect())
            })
            .unwrap();
        assert!(!converged);
        assert_eq!(modes.len(), 1);
        assert!(modes[0].absolute_residual > config.tolerance);
        assert!(count >= 4);
    }
    #[test]
    fn davidson_finds_hidden_negative_block_and_degenerate_roots() {
        let n = 40;
        let diagonal = vec![2.0; n];
        let mut h = Matrix::zeros(n, n);
        for i in 0..n {
            h[(i, i)] = 2.0 + i as f64 * 0.1;
        }
        h[(n - 1, n - 1)] = -1.5;
        h[(n - 2, n - 2)] = -1.5;
        for i in 2..n - 2 {
            h[(i, i - 1)] = 0.03;
            h[(i - 1, i)] = 0.03;
        }
        let (oracle, _) = symmetric_eigen(&h).unwrap();
        let (modes, defect, converged, _) = lowest_modes(
            &diagonal,
            3,
            2.0,
            &ResponseSolverConfig::default(),
            &StabilityOptions::default(),
            |x| {
                Ok((0..n)
                    .map(|i| (0..n).map(|j| h[(i, j)] * x[j]).sum())
                    .collect())
            },
        )
        .unwrap();
        assert!(converged);
        for (i, m) in modes.iter().enumerate() {
            assert!((m.eigenvalue - oracle[i]).abs() < 1e-9);
            assert!(m.relative_residual < 1e-10);
        }
        assert!(defect < 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (dot(&modes[i].amplitudes, &modes[j].amplitudes)
                        - if i == j { 1.0 } else { 0.0 })
                    .abs()
                        < 1e-10
                );
            }
        }
    }
}
