// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear Taylor response of a restricted open-shell reference.
//!
//! One set of orbitals holds closed, open and virtual classes. In the frame
//! transported with the overlap, `B(t)^T S(t) B(t) = 1`, the state is a pair
//! of orthogonal projectors: `C(t)` on the closed and `O(t)` on the open
//! orbitals, with the spin densities `D_beta = C` and `D_alpha = C + O`. The
//! conditions
//!
//! ```text
//! C C = C,   O O = O,   C O = 0,
//! sum_s [F_s, D_s] = 0
//! ```
//!
//! hold order by order in `t`. The first three fix every block of an
//! order-`n` coefficient except the closed-open and closed-virtual blocks of
//! `C` and the open-virtual block of `O`; the last is a linear equation for
//! those blocks with the orbital Hessian of the reference on its left-hand
//! side. The energy through order `2k + 1` needs the blocks through order
//! `k`.
//!
//! No displaced calculation enters: the derivatives of the integrals and of
//! the quadrature are the coefficients of the nuclear Taylor driver.

use super::geometry::GeometryTaylor;
use super::{add, matrix_product, max_abs, metric_transport, MolecularDirection};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::response::complex::Complex;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::ResponseSolver;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod dynamic;
pub(crate) use dynamic::open_shell_cross_operator_response;
mod excited;
mod excited_path;
pub use excited::{open_shell_tda_states, OpenShellExcitations, OpenShellTdaState};
pub use excited_path::{
    open_shell_excitation_path, open_shell_excited_state_hessian, OpenShellExcitationPath,
};
mod projectors;
pub use dynamic::{
    open_shell_dynamic_polarizability, open_shell_operator_response,
    OpenShellDynamicPolarizability, OpenShellOperatorResponse,
};

/// Orbital classes of the common orbitals: closed, then open, then virtual.
#[derive(Clone, Copy)]
struct Classes {
    closed: usize,
    open: usize,
    total: usize,
}

impl Classes {
    fn class(&self, p: usize) -> usize {
        if p < self.closed {
            0
        } else if p < self.closed + self.open {
            1
        } else {
            2
        }
    }

    fn virtuals(&self) -> usize {
        self.total - self.closed - self.open
    }

    fn dimension(&self) -> usize {
        let (nc, no, nv) = (self.closed, self.open, self.virtuals());
        nc * no + nc * nv + no * nv
    }

    /// Changes of the closed and of the open projector by the amplitudes
    /// `x`: closed-open, closed-virtual and open-virtual blocks in that
    /// order.
    fn rotation(&self, x: &[f64]) -> [Matrix; 2] {
        let (nc, no, nv) = (self.closed, self.open, self.virtuals());
        let mut closed = Matrix::zeros(self.total, self.total);
        let mut open = closed.clone();
        let mut k = 0;
        for i in 0..nc {
            for q in 0..no {
                let col = nc + q;
                closed[(i, col)] = x[k];
                closed[(col, i)] = x[k];
                open[(i, col)] = -x[k];
                open[(col, i)] = -x[k];
                k += 1;
            }
        }
        for i in 0..nc {
            for a in 0..nv {
                let col = nc + no + a;
                closed[(i, col)] = x[k];
                closed[(col, i)] = x[k];
                k += 1;
            }
        }
        for q in 0..no {
            for a in 0..nv {
                let (row, col) = (nc + q, nc + no + a);
                open[(row, col)] = x[k];
                open[(col, row)] = x[k];
                k += 1;
            }
        }
        [closed, open]
    }

    /// The blocks of the stationarity matrix that pair with the amplitudes.
    fn blocks(&self, g: &Matrix) -> Vec<f64> {
        let (nc, no, nv) = (self.closed, self.open, self.virtuals());
        let mut out = Vec::with_capacity(self.dimension());
        for i in 0..nc {
            for q in 0..no {
                out.push(g[(nc + q, i)]);
            }
        }
        for i in 0..nc {
            for a in 0..nv {
                out.push(g[(nc + no + a, i)]);
            }
        }
        for q in 0..no {
            for a in 0..nv {
                out.push(g[(nc + no + a, nc + q)]);
            }
        }
        out
    }
}

/// `[f, d]` added to `g`.
fn commutator(g: &mut Matrix, f: &Matrix, d: &Matrix) {
    add(g, &f.matmul_seq(d), 1.0);
    add(g, &d.matmul_seq(f), -1.0);
}

/// Half the orbital Hessian of the reference on the rotation amplitudes.
/// The Coulomb, exchange and exchange-correlation kernel of the reference
/// is applied through the cached response engines; it is the order-zero
/// term of the moving-geometry kernel series, without the per-point series
/// arithmetic in every Krylov iteration.
struct RotationOperator<'a> {
    kernel: &'a ResponseSolver<'a>,
    orbitals: &'a Matrix,
    fock: &'a [Matrix; 2],
    classes: Classes,
}

impl RotationOperator<'_> {
    /// Frame spin densities (alpha, beta) of the amplitudes.
    fn densities(&self, x: &[f64], imaginary: bool) -> [Matrix; 2] {
        let [closed, open] = self.classes.rotation(x);
        let mut alpha = closed.clone();
        add(&mut alpha, &open, 1.0);
        let mut densities = [alpha, closed];
        if imaginary {
            for d in &mut densities {
                for i in 0..d.rows {
                    for j in i + 1..d.cols {
                        d[(i, j)] = -d[(i, j)];
                    }
                }
            }
        }
        densities
    }

    /// Frame Kohn-Sham matrix changes caused by the amplitudes.
    fn fock_response(&self, x: &[f64], imaginary: bool) -> Result<[Matrix; 2]> {
        let transposed = self.orbitals.transpose();
        let q: [Matrix; 2] = self
            .densities(x, imaginary)
            .map(|d| self.orbitals.matmul_seq(&d).matmul_seq(&transposed));
        let g = if imaginary {
            self.kernel.imaginary_fock_response(&q)?
        } else {
            self.kernel.fock_response(&q)?
        };
        Ok(std::array::from_fn(|s| {
            transposed.matmul_seq(&g[s]).matmul_seq(self.orbitals)
        }))
    }

    fn image(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.image_parity(x, false)
    }

    fn image_parity(&self, x: &[f64], imaginary: bool) -> Result<Vec<f64>> {
        let densities = self.densities(x, imaginary);
        let response = self.fock_response(x, imaginary)?;
        let size = self.classes.total;
        let mut g = Matrix::zeros(size, size);
        for s in 0..2 {
            commutator(&mut g, &self.fock[s], &densities[s]);
            let occupied = if s == 0 {
                self.classes.closed + self.classes.open
            } else {
                self.classes.closed
            };
            for i in 0..size {
                for j in 0..size {
                    if j < occupied && i >= occupied {
                        g[(i, j)] += response[s][(i, j)];
                    }
                    if i < occupied && j >= occupied {
                        g[(i, j)] -= response[s][(i, j)];
                    }
                }
            }
        }
        Ok(self.classes.blocks(&g))
    }

    fn solve(&self, rhs: &[f64], config: &ResponseSolverConfig) -> Result<(Vec<f64>, f64)> {
        if rhs.iter().all(|v| *v == 0.0) {
            return Ok((vec![0.0; rhs.len()], 0.0));
        }
        let rhs: Vec<Complex> = rhs.iter().map(|&v| Complex::new(v, 0.0)).collect();
        // The public nuclear diagnostics require both a relative Krylov
        // residual and an absolute Hartree stationarity defect. A large
        // source needs a tighter relative solve to satisfy the latter.
        // No floor, physical operator shift or relaxed gate is involved.
        let source_norm = rhs.iter().fold(0.0_f64, |n, z| n.hypot(z.re).hypot(z.im));
        let solve_config = config.for_absolute_residual(source_norm)?;
        let solution = krylov::solve(self, &rhs, None, &solve_config)?;
        Ok((
            solution.amplitudes.iter().map(|z| z.re).collect(),
            solution.relative_residual,
        ))
    }
}

impl LinearOperator for RotationOperator<'_> {
    fn dimension(&self) -> usize {
        self.classes.dimension()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.image(v))
    }
}

/// Energy coefficients of a restricted open-shell reference along a path.
#[derive(Clone, Debug)]
pub struct OpenShellTaylor {
    /// `E(t) = sum_n energy[n] t^n`, exact through `n = 2 solved + 1`,
    /// Hartree.
    pub energy: Vec<f64>,
    /// Highest order of the rotation equations that was solved.
    pub solved: usize,
    /// Largest coefficient of `C C - C`, `O O - O` and `C O` per order.
    pub flag_defects: Vec<f64>,
    /// Largest paired block of `sum_s [F_s, D_s]` per solved order.
    pub stationarity_defects: Vec<f64>,
    pub residuals: Vec<f64>,
    /// The same block of the reference itself.
    pub reference_stationarity: f64,
    pub reference_energy_error: f64,
    pub converged: bool,
}

/// The reference solution along one nuclear path.
pub(super) struct OpenShellPath {
    pub geometry: GeometryTaylor,
    /// Retained scalar remainders for subsequent mixed-energy polarization.
    pub energy_remainders: Vec<f64>,
    /// Transport coefficients `B(t)` (AO by orbital).
    pub b: Vec<Matrix>,
    /// Closed and open projector coefficients in the transported frame.
    pub closed: Vec<Matrix>,
    pub open: Vec<Matrix>,
    /// Kohn-Sham matrix coefficients in the transported frame.
    pub f: [Vec<Matrix>; 2],
    /// Spin density coefficients in the AO basis.
    pub p: [Vec<Matrix>; 2],
    /// Numbers of closed and of open orbitals.
    pub counts: [usize; 2],
}

/// Taylor series of the restricted open-shell energy along
/// `R(t) = R(0) + t direction` through `order`, with the rotation equations
/// solved through `solved`. `orbitals` are the common orbitals of `scf`:
/// closed, then open, then virtual.
#[allow(clippy::too_many_arguments)]
pub(super) fn open_shell_path(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    solved: usize,
    config: &ResponseSolverConfig,
) -> Result<(OpenShellTaylor, OpenShellPath)> {
    let mut value = krylov::with_physical_acceptance(config, |inner| {
        let value = open_shell_path_once(
            mol, mb, functional, grid, scf, orbitals, opts, direction, order, solved, inner,
        )?;
        let defect = value
            .0
            .flag_defects
            .iter()
            .chain(&value.0.stationarity_defects)
            .copied()
            .fold(0.0_f64, |a, b| {
                if b.is_finite() {
                    a.max(b)
                } else {
                    f64::INFINITY
                }
            });
        Ok((value, defect))
    })?;
    let bound = config.acceptance();
    value.0.converged = value.0.reference_stationarity <= bound
        && value.0.reference_energy_error.abs() <= bound
        && value
            .0
            .flag_defects
            .iter()
            .chain(&value.0.stationarity_defects)
            .all(|value| value.is_finite() && *value <= bound);
    Ok(value)
}

fn open_shell_path_once(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    solved: usize,
    config: &ResponseSolverConfig,
) -> Result<(OpenShellTaylor, OpenShellPath)> {
    config.validate()?;
    let reference_field = opts.external_field.map_or([0.0; 3], |f| [f.x, f.y, f.z]);
    let nmo = orbitals.cols;
    if order == usize::MAX
        || direction.nuclear.len() != mol.len()
        || direction
            .nuclear
            .iter()
            .flatten()
            .chain(direction.electric.iter())
            .chain(reference_field.iter())
            .any(|v| !v.is_finite())
        || mb.nao_sph != scf.overlap.rows
        || orbitals.rows != mb.nao_sph
        || !scf.converged
        || scf.n_alpha < scf.n_beta
        || scf.n_alpha > nmo
    {
        return Err(DftError::InvalidInput(
            "the open-shell nuclear Taylor response requires finite per-atom directions and a \
             converged restricted open-shell reference with its common orbitals"
                .into(),
        ));
    }
    if opts.pcm.is_some()
        || opts.vv10.is_some()
        || opts.skala.is_some()
        || functional.required_nonlocal_kind().is_some()
        || opts.density_fit.is_some()
        || opts.extra_hcore.is_some()
        || !matches!(opts.kernel, crate::integrals::CoulombKernel::Coulomb)
    {
        return Err(DftError::Unsupported {
            feature: "open-shell nuclear Taylor response with additional Hamiltonian terms".into(),
            missing: "moving-geometry Taylor kernels for solvent, nonlocal correlation, a \
                      learned functional, RI-J and additional one-electron operators"
                .into(),
        });
    }
    if !functional.parts.is_empty() && grid.is_none() {
        return Err(DftError::InvalidInput(
            "nuclear semilocal response requires the reference molecular grid".into(),
        ));
    }
    let classes = Classes {
        closed: scf.n_beta,
        open: scf.n_alpha - scf.n_beta,
        total: nmo,
    };
    let transposed = orbitals.transpose();
    let unit = |from: usize, count: usize| {
        let mut m = Matrix::zeros(nmo, nmo);
        for i in from..from + count {
            m[(i, i)] = 1.0;
        }
        m
    };
    let closed = [unit(0, classes.closed)];
    let open = [unit(classes.closed, classes.open)];
    let mut metric = transposed.matmul_seq(&scf.overlap).matmul_seq(orbitals);
    for i in 0..nmo {
        metric[(i, i)] -= 1.0;
    }
    let mut defect = max_abs(&metric);
    for (s, density) in scf.density.iter().enumerate() {
        let mut frame = closed[0].clone();
        if s == 0 {
            add(&mut frame, &open[0], 1.0);
        }
        let mut ao = orbitals.matmul_seq(&frame).matmul_seq(&transposed);
        add(&mut ao, density, -1.0);
        defect = defect.max(max_abs(&ao));
    }
    if defect > 1e-8 {
        return Err(DftError::InvalidInput(
            "the common orbitals are not the orthonormal closed, open and virtual orbitals of \
             the reference densities"
                .into(),
        ));
    }
    let geometry = GeometryTaylor::new(
        mol,
        mb,
        grid,
        functional,
        None,
        &direction.nuclear,
        reference_field,
        direction.electric,
        order,
        opts.incore_mb,
    )?;
    let mut b = super::precision::transport(orbitals, &geometry.overlap, order);
    let frame0: [Matrix; 2] =
        std::array::from_fn(|s| transposed.matmul_seq(&scf.fock[s]).matmul_seq(orbitals));
    let kernel = ResponseSolver::kernel_only(mol, mb, functional, grid, scf, opts)?;
    let operator = RotationOperator {
        kernel: &kernel,
        orbitals,
        fock: &frame0,
        classes,
    };
    let largest = |g: &super::precision::MatrixParts| {
        classes
            .blocks(&g.high)
            .iter()
            .zip(classes.blocks(&g.low))
            .fold(0.0_f64, |m, (h, l)| m.max((h + l).abs()))
    };
    let mut flag = projectors::Flag::new(closed[0].clone(), open[0].clone(), classes);
    let reference_stationarity = largest(&flag.stationarity(
        &std::array::from_fn(|s| vec![super::precision::MatrixParts::new(frame0[s].clone())]),
        0,
    ));
    // Retain the small common-orbital reference rotation. An accepted SCF
    // residual can otherwise be amplified by soft fourth-order response.
    let mut reference_history = super::precision::RefinementHistory::new();
    for refinement in 0..=12 {
        let p = flag.densities(&b, 0);
        let (_, ao) = geometry.evaluate_retained(functional, &p, 0)?;
        let f = std::array::from_fn(|s| super::precision::congruence(&b[..1], &ao[s], 0));
        let g = flag.stationarity(&f, 0);
        let rhs: Vec<_> = classes.blocks(&g.high).iter().map(|v| -v).collect();
        let norm = rhs.iter().fold(0.0_f64, |a, &x| a.hypot(x));
        let stalled = reference_history.observe(norm, &b)?;
        if norm <= config.tolerance * f64::EPSILON || stalled || refinement == 12 {
            break;
        }
        let (x, _) = operator.solve(&rhs, config)?;
        let [closed_change, open_change] = classes.rotation(&x);
        let mut generator = super::precision::MatrixParts::zeros(nmo, nmo);
        for i in 0..nmo {
            for j in i + 1..nmo {
                let amplitude = if classes.class(i) == 0 {
                    closed_change[(i, j)]
                } else {
                    open_change[(i, j)]
                };
                generator.high[(j, i)] = amplitude;
                generator.high[(i, j)] = -amplitude;
            }
        }
        let mut rotation = super::precision::MatrixParts::new(Matrix::identity(nmo));
        let mut power = vec![rotation.clone()];
        let mut factor = 1.0;
        for k in 1..=4 {
            power = super::precision::product(&power, std::slice::from_ref(&generator), 0);
            factor /= k as f64;
            rotation.add_scaled(&power[0], factor);
        }
        for coefficient in &mut b {
            *coefficient = super::precision::coefficient(
                std::slice::from_ref(coefficient),
                std::slice::from_ref(&rotation),
                0,
            );
        }
    }
    b = reference_history.finish();
    let mut residuals = Vec::new();
    for n in 1..=order {
        flag.advance(n);
        if n <= solved {
            let mut history = super::precision::RefinementHistory::new();
            let mut initial = None;
            for refinement in 0..=8 {
                let p = flag.densities(&b, n);
                let (_, ao) = geometry.evaluate_retained(functional, &p, n)?;
                let f = std::array::from_fn(|s| super::precision::congruence(&b, &ao[s], n));
                let g = flag.stationarity(&f, n);
                let rhs: Vec<_> = classes.blocks(&g.high).iter().map(|v| -v).collect();
                let norm = rhs.iter().fold(0.0_f64, |a, &x| a.hypot(x));
                initial.get_or_insert(norm);
                let stalled = history.observe(norm, &flag)?;
                if norm <= config.tolerance * f64::EPSILON || stalled || refinement == 8 {
                    break;
                }
                // Replace the physical common-orbital residual, keeping the
                // sub-ulp rotation in the flag coefficients and AO density.
                // The three constrained blocks and the original Hessian are
                // unchanged; no projector denominator or gate is modified.
                let (x, _) = operator.solve(&rhs, config)?;
                flag.rotate(n, classes.rotation(&x));
            }
            let initial = initial.expect("the common-orbital physical source is evaluated");
            residuals.push(if initial == 0.0 {
                history.norm
            } else {
                history.norm / initial
            });
            flag = history.finish();
        }
    }
    // Rebuild from the actual moving integrals, grid and full density path.
    let p = flag.densities(&b, order);
    let (evaluation, ao) = geometry.evaluate_retained(functional, &p, order)?;
    let actual = std::array::from_fn(|s| super::precision::congruence(&b, &ao[s], order));
    let mut flag_defects = Vec::with_capacity(order);
    let mut stationarity_defects = Vec::new();
    for n in 1..=order {
        flag_defects.push(flag.defect(n));
        if n <= solved {
            stationarity_defects.push(largest(&flag.stationarity(&actual, n)));
        }
    }
    let energy = evaluation.total_energy();
    let reference_energy_error = energy[0] - scf.energy;
    let acceptance = config.acceptance();
    let converged = flag_defects
        .iter()
        .chain(&stationarity_defects)
        .all(|x| *x <= acceptance)
        && reference_energy_error.abs() <= acceptance;
    let taylor = OpenShellTaylor {
        energy,
        solved: solved.min(order),
        flag_defects,
        stationarity_defects,
        residuals,
        reference_stationarity,
        reference_energy_error,
        converged,
    };
    let (closed, open) = flag.rounded();
    let path = OpenShellPath {
        geometry,
        energy_remainders: evaluation.energy_remainders(),
        b: super::precision::high(&b),
        closed,
        open,
        f: std::array::from_fn(|s| super::precision::high(&actual[s])),
        p: std::array::from_fn(|s| super::precision::high(&p[s])),
        counts: [classes.closed, classes.open],
    };
    Ok((taylor, path))
}

/// Stationary nuclear energy through order four on the common-orbital flag.
/// Coefficients are factorial normalized. The 2n+1 identity requires only
/// floor(order/2) rotation equations; all geometry and metric orders remain.
/// The empty open-shell class gives the restricted closed-shell limit.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_nuclear_energy_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<OpenShellTaylor> {
    open_shell_molecular_energy_taylor(
        mol,
        mb,
        functional,
        grid,
        scf,
        orbitals,
        opts,
        &MolecularDirection::nuclear(direction.to_vec()),
        order,
        config,
    )
}

/// Stationary common-orbital energy along a combined nuclear/electric path.
/// The closed/open/virtual constraints are the same as for nuclear response;
/// coefficients are factorial normalized and the energy uses the 2n+1 rule.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_molecular_energy_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<OpenShellTaylor> {
    Ok(open_shell_molecular_energy_taylor_retained(
        mol, mb, functional, grid, scf, orbitals, opts, direction, order, config,
    )?
    .0)
}

/// Preserve the scalar remainder through the driver's mixed contraction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_shell_molecular_energy_taylor_retained(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(OpenShellTaylor, Vec<f64>)> {
    if !(1..=4).contains(&order) {
        return Err(DftError::InvalidInput(
            "common-orbital nuclear energy order must be one through four".into(),
        ));
    }
    let (value, path) = open_shell_path(
        mol,
        mb,
        functional,
        grid,
        scf,
        orbitals,
        opts,
        direction,
        order,
        order / 2,
        config,
    )?;
    Ok((value, path.energy_remainders))
}

/// Cartesian gradient and Hessian of a restricted open-shell energy.
#[derive(Clone, Debug)]
pub struct OpenShellHessian {
    /// Gradient from the first-order path coefficients, Hartree/Bohr, index
    /// `3 * atom + axis`.
    pub gradient: Vec<f64>,
    /// Hessian, Hartree/Bohr^2.
    pub hessian: Matrix,
    /// Second-order path coefficient of every Cartesian direction, the
    /// reference of mixed nuclear-electric paths.
    pub second: Vec<f64>,
    /// Number of nuclear paths evaluated.
    pub paths: usize,
    pub residual: f64,
    pub flag_defect: f64,
    pub stationarity_defect: f64,
    pub reference_stationarity: f64,
    pub reference_energy_error: f64,
    pub converged: bool,
}

/// Second derivatives of the restricted open-shell energy. The second-order
/// path coefficients `c(u)` of every Cartesian direction and of every pair
/// sum give `H_ii = 2 c(e_i)` and `H_ij = c(e_i + e_j) - c(e_i) - c(e_j)`;
/// each needs the rotation equations at first order only.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_hessian(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    config: &ResponseSolverConfig,
) -> Result<OpenShellHessian> {
    let n = 3 * mol.len();
    let unit = |index: usize| -> Vec<[f64; 3]> {
        let mut direction = vec![[0.0; 3]; mol.len()];
        direction[index / 3][index % 3] = 1.0;
        direction
    };
    let mut paths = 0_usize;
    let mut converged = true;
    let mut residual = 0.0_f64;
    let mut flag_defect = 0.0_f64;
    let mut stationarity_defect = 0.0_f64;
    let mut reference_stationarity = 0.0_f64;
    let mut reference_energy_error = 0.0_f64;
    let mut run = |direction: Vec<[f64; 3]>| -> Result<OpenShellTaylor> {
        let (taylor, _) = open_shell_path(
            mol,
            mb,
            functional,
            grid,
            scf,
            orbitals,
            opts,
            &MolecularDirection::nuclear(direction),
            2,
            1,
            config,
        )?;
        paths += 1;
        converged &= taylor.converged;
        residual = taylor.residuals.iter().copied().fold(residual, f64::max);
        flag_defect = taylor
            .flag_defects
            .iter()
            .copied()
            .fold(flag_defect, f64::max);
        stationarity_defect = taylor
            .stationarity_defects
            .iter()
            .copied()
            .fold(stationarity_defect, f64::max);
        reference_stationarity = reference_stationarity.max(taylor.reference_stationarity);
        reference_energy_error = reference_energy_error.max(taylor.reference_energy_error.abs());
        Ok(taylor)
    };
    let mut gradient = vec![0.0; n];
    let mut second = vec![0.0; n];
    let mut hessian = Matrix::zeros(n, n);
    for i in 0..n {
        let taylor = run(unit(i))?;
        gradient[i] = taylor.energy[1];
        second[i] = taylor.energy[2];
        hessian[(i, i)] = 2.0 * taylor.energy[2];
    }
    for i in 0..n {
        for j in 0..i {
            let mut direction = unit(i);
            direction[j / 3][j % 3] += 1.0;
            let taylor = run(direction)?;
            let value = taylor.energy[2] - second[i] - second[j];
            hessian[(i, j)] = value;
            hessian[(j, i)] = value;
        }
    }
    Ok(OpenShellHessian {
        second,
        gradient,
        hessian,
        paths,
        residual,
        flag_defect,
        stationarity_defect,
        reference_stationarity,
        reference_energy_error,
        converged,
    })
}

/// Static electric response of a restricted open-shell reference.
#[derive(Clone, Debug)]
pub struct OpenShellElectric {
    /// Dipole moment `-dE/dF`, e Bohr.
    pub dipole: [f64; 3],
    /// Polarizability `alpha_ab = -d2E/(dF_a dF_b)`.
    pub polarizability: [[f64; 3]; 3],
    /// `d mu_a / d R_i`: rows are dipole axes, columns `3 * atom + axis`.
    pub dipole_derivatives: Option<Matrix>,
    /// Number of paths evaluated.
    pub paths: usize,
    pub residual: f64,
    pub flag_defect: f64,
    pub stationarity_defect: f64,
    pub converged: bool,
}

/// Dipole moment and polarizability from the first- and second-order energy
/// coefficients `c` of field paths: `mu_a = -c1(F_a)`,
/// `alpha_aa = -2 c2(F_a)`, `alpha_ab = -[c2(F_a + F_b) - c2(F_a) - c2(F_b)]`.
/// With `nuclear_second` (the second-order coefficients of the Cartesian
/// directions, [`OpenShellHessian::second`]) the dipole derivatives follow
/// from paths that move one coordinate and one field component together,
/// `d mu_a / d R_i = -[c2(R_i + F_a) - c2(R_i) - c2(F_a)]`.
#[allow(clippy::too_many_arguments)]
pub fn open_shell_electric(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    nuclear_second: Option<&[f64]>,
    config: &ResponseSolverConfig,
) -> Result<OpenShellElectric> {
    let natoms = mol.len();
    if nuclear_second.is_some_and(|values| values.len() != 3 * natoms) {
        return Err(DftError::InvalidInput(
            "the nuclear path coefficients hold one value per Cartesian coordinate".into(),
        ));
    }
    let mut paths = 0_usize;
    let mut converged = true;
    let mut residual = 0.0_f64;
    let mut flag_defect = 0.0_f64;
    let mut stationarity_defect = 0.0_f64;
    let mut run = |nuclear: Vec<[f64; 3]>, electric: [f64; 3]| -> Result<OpenShellTaylor> {
        let (taylor, _) = open_shell_path(
            mol,
            mb,
            functional,
            grid,
            scf,
            orbitals,
            opts,
            &MolecularDirection { nuclear, electric },
            2,
            1,
            config,
        )?;
        paths += 1;
        converged &= taylor.converged;
        residual = taylor.residuals.iter().copied().fold(residual, f64::max);
        flag_defect = taylor
            .flag_defects
            .iter()
            .copied()
            .fold(flag_defect, f64::max);
        stationarity_defect = taylor
            .stationarity_defects
            .iter()
            .copied()
            .fold(stationarity_defect, f64::max);
        Ok(taylor)
    };
    let still = || vec![[0.0; 3]; natoms];
    let axis = |a: usize| {
        let mut field = [0.0; 3];
        field[a] = 1.0;
        field
    };
    let mut dipole = [0.0; 3];
    let mut second = [0.0; 3];
    let mut polarizability = [[0.0; 3]; 3];
    for a in 0..3 {
        let taylor = run(still(), axis(a))?;
        dipole[a] = -taylor.energy[1];
        second[a] = taylor.energy[2];
        polarizability[a][a] = -2.0 * taylor.energy[2];
    }
    for a in 0..3 {
        for b in 0..a {
            let mut field = axis(a);
            field[b] = 1.0;
            let taylor = run(still(), field)?;
            let value = -(taylor.energy[2] - second[a] - second[b]);
            polarizability[a][b] = value;
            polarizability[b][a] = value;
        }
    }
    let dipole_derivatives = match nuclear_second {
        Some(values) => {
            let mut derivatives = Matrix::zeros(3, 3 * natoms);
            for (i, moved) in values.iter().enumerate() {
                for (a, field) in second.iter().enumerate() {
                    let mut direction = still();
                    direction[i / 3][i % 3] = 1.0;
                    let taylor = run(direction, axis(a))?;
                    derivatives[(a, i)] = -(taylor.energy[2] - moved - field);
                }
            }
            Some(derivatives)
        }
        None => None,
    };
    Ok(OpenShellElectric {
        dipole,
        polarizability,
        dipole_derivatives,
        paths,
        residual,
        flag_defect,
        stationarity_defect,
        converged,
    })
}
