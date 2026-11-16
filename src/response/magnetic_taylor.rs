// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static response to simultaneous uniform electric and magnetic fields by
//! order-by-order density-projector equations with a complex Hermitian
//! density.
//!
//! With the vector potential `A = B x (r - O) / 2` the one-electron operator
//! along the path `E(t) = t e`, `B(t) = t b` is
//!
//! ```text
//! h(t) = h + t [e . r - (i/2) b . ((r - O) x grad)]
//!          + t^2 [b^2 (r - O)^2 - (b . (r - O))^2] / 8.
//! ```
//!
//! In the orthonormal reference orbitals the density coefficients
//! `D_n = R_n + i I_n` have a symmetric real part and an antisymmetric
//! imaginary part. `D^2 = D` fixes their occupied and virtual diagonal
//! blocks, and `[F, D] = 0` gives the occupied-virtual block: its real part
//! from the orbital Hessian of the electric response, its imaginary part
//! from the magnetic Hessian, in which only exact exchange couples. The
//! semilocal functional sees the real part of the density only; the
//! current-density dependence of a meta-GGA is not included. All basis
//! functions share the gauge origin `O`, so the result depends on it in a
//! finite basis.

use super::complex::Complex;
use super::krylov::{self, LinearOperator, OperatorStructure};
use super::nonlinear::semilocal_fock_coefficient;
use super::{ResponseReport, ResponseSolver, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

fn add(target: &mut Matrix, source: &Matrix, weight: f64) {
    for (a, b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += weight * b;
    }
}
fn to_ao(c: &Matrix, d: &Matrix) -> Matrix {
    c.matmul_seq(d).matmul_seq(&c.transpose())
}
fn to_mo(c: &Matrix, f: &Matrix) -> Matrix {
    c.transpose().matmul_seq(f).matmul_seq(c)
}
fn largest(m: &Matrix) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// A complex matrix as its real and imaginary parts.
#[derive(Clone)]
struct Pair {
    re: Matrix,
    im: Matrix,
}

impl Pair {
    fn zeros(n: usize) -> Self {
        Self {
            re: Matrix::zeros(n, n),
            im: Matrix::zeros(n, n),
        }
    }
    fn product(&self, other: &Self) -> Self {
        let mut re = self.re.matmul_seq(&other.re);
        add(&mut re, &self.im.matmul_seq(&other.im), -1.0);
        let mut im = self.re.matmul_seq(&other.im);
        add(&mut im, &self.im.matmul_seq(&other.re), 1.0);
        Self { re, im }
    }
    fn add(&mut self, other: &Self, weight: f64) {
        add(&mut self.re, &other.re, weight);
        add(&mut self.im, &other.im, weight);
    }
    fn largest(&self) -> f64 {
        largest(&self.re).max(largest(&self.im))
    }
}

/// Imaginary part of the Kohn-Sham matrix caused by the antisymmetric
/// imaginary part of the density: `-c_x K[I]` and its long-range part.
pub(super) fn exchange(solver: &ResponseSolver, density: &[Matrix; 2]) -> Result<[Matrix; 2]> {
    let nao = solver.mb.nao_sph;
    let mut out = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    let cx = solver.functional.hybrid.cx;
    if cx != 0.0 {
        let k = solver
            .jk
            .build_k_only(solver.mb, [&density[0], &density[1]])?;
        for s in 0..2 {
            add(&mut out[s], &k[s], -cx);
        }
    }
    if let Some(engine) = &solver.jk_lr {
        let k = engine.build_k_only(solver.mb, [&density[0], &density[1]])?;
        for s in 0..2 {
            add(&mut out[s], &k[s], -solver.functional.hybrid.clr);
        }
    }
    Ok(out)
}

/// AO imaginary density `C (Y - Y^T) C^T` of the occupied-virtual
/// amplitudes `y`.
pub(super) fn imaginary_density(solver: &ResponseSolver, y: &[f64]) -> [Matrix; 2] {
    let nao = solver.mb.nao_sph;
    let mut out = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    let mut off = 0;
    for s in 0..solver.spins() {
        let (no, nv) = (solver.dims.n_occ[s], solver.dims.n_vir[s]);
        let c = &solver.scf.mo_coeff[s];
        let mut mo = Matrix::zeros(c.cols, c.cols);
        for a in 0..nv {
            for i in 0..no {
                mo[(no + a, i)] = y[off + a * no + i];
                mo[(i, no + a)] = -y[off + a * no + i];
            }
        }
        out[s] = to_ao(c, &mo);
        off += no * nv;
    }
    if solver.spins() == 1 {
        out[1] = out[0].clone();
    }
    out
}

/// The magnetic Hessian on the imaginary occupied-virtual amplitudes:
/// orbital-energy differences plus the exchange response.
struct MagneticOperator<'a, 'b> {
    solver: &'a ResponseSolver<'b>,
    diagonal: &'a [f64],
    current: Option<&'a crate::properties::giao_xc::MetaCurrentKernel>,
}

impl MagneticOperator<'_, '_> {
    fn image(&self, y: &[f64]) -> Result<Vec<f64>> {
        let density = imaginary_density(self.solver, y);
        let mut response = exchange(self.solver, &density)?;
        if let Some(kernel) = self.current {
            let grid = self
                .solver
                .grid
                .expect("a current kernel requires its checked grid");
            for s in 0..2 {
                add(
                    &mut response[s],
                    &kernel.response_fock(self.solver.mb, grid, &density[s], s)?,
                    1.0,
                );
            }
        }
        let mut out = vec![0.0; y.len()];
        self.solver
            .project_ov([&response[0], &response[1]], &mut out);
        for ((value, d), y) in out.iter_mut().zip(self.diagonal).zip(y) {
            *value += d * y;
        }
        Ok(out)
    }
}

impl LinearOperator for MagneticOperator<'_, '_> {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn structure(&self) -> OperatorStructure {
        // An aufbau gap alone does not prove stability of the interacting
        // time-odd Hessian, especially with a current-dependent kernel.
        OperatorStructure::Hermitian
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| self.image(v))
    }
}

/// Imaginary occupied-virtual amplitudes `y` with
/// `(e_a - e_i) y + G[y] = -source`, and the relative residual.
pub(super) fn solve_imaginary(
    solver: &ResponseSolver,
    source: &[f64],
    config: &ResponseSolverConfig,
) -> Result<(Vec<f64>, f64)> {
    solve_imaginary_with_current(solver, source, config, None)
}

/// The same imaginary response with the current-invariant kinetic-density
/// Hessian, for ordinary and correlated nuclear-moment response limits.
pub(super) fn solve_imaginary_with_current(
    solver: &ResponseSolver,
    source: &[f64],
    config: &ResponseSolverConfig,
    current: Option<&crate::properties::giao_xc::MetaCurrentKernel>,
) -> Result<(Vec<f64>, f64)> {
    let mut diagonal = Vec::with_capacity(source.len());
    for s in 0..solver.spins() {
        let (no, nv) = (solver.dims.n_occ[s], solver.dims.n_vir[s]);
        let eps = &solver.scf.mo_energy[s];
        for a in 0..nv {
            for i in 0..no {
                diagonal.push(eps[no + a] - eps[i]);
            }
        }
    }
    if diagonal.iter().any(|d| !(*d > 0.0)) {
        return Err(DftError::InvalidInput(
            "the magnetic response needs an aufbau reference with a positive orbital gap".into(),
        ));
    }
    if solver.functional.hybrid.cx == 0.0 && solver.jk_lr.is_none() && current.is_none() {
        return Ok((
            source.iter().zip(&diagonal).map(|(b, d)| -b / d).collect(),
            0.0,
        ));
    }
    let rhs: Vec<Complex> = source.iter().map(|&b| Complex::new(-b, 0.0)).collect();
    let operator = MagneticOperator {
        solver,
        diagonal: &diagonal,
        current,
    };
    let solution = krylov::solve(&operator, &rhs, Some(&diagonal), config)?;
    Ok((
        solution.amplitudes.iter().map(|z| z.re).collect(),
        solution.relative_residual,
    ))
}

/// Taylor coefficients of the induced moments along a combined field path.
#[derive(Clone, Debug)]
pub struct MagneticFieldTaylor {
    /// `E(t) = t electric`, atomic units.
    pub electric: [f64; 3],
    /// `B(t) = t magnetic`, atomic units.
    pub magnetic: [f64; 3],
    /// Gauge origin, Bohr.
    pub origin: [f64; 3],
    /// `mu(t) = sum_n dipole_coefficients[n] t^n`.
    pub dipole_coefficients: Vec<[f64; 3]>,
    /// `m(t) = -dE/dB = sum_n magnetic_coefficients[n] t^n`.
    pub magnetic_coefficients: Vec<[f64; 3]>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
}

/// Second moment `(r - O)_c (r - O)_d` among the rank-two moment matrices.
fn second_moment(moments: &[((i32, i32, i32), Matrix)], c: usize, d: usize) -> Result<&Matrix> {
    let mut power = [0_i32; 3];
    power[c] += 1;
    power[d] += 1;
    moments
        .iter()
        .find(|(p, _)| *p == (power[0], power[1], power[2]))
        .map(|(_, m)| m)
        .ok_or_else(|| DftError::LinearAlgebra("missing second moment integral".into()))
}

/// Induced electric dipole and magnetic moment of an integer-occupation
/// reference along `E(t) = t electric`, `B(t) = t magnetic` through `order`.
#[allow(clippy::too_many_arguments)]
pub fn magnetic_field_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    electric: [f64; 3],
    magnetic: [f64; 3],
    origin: Vec3,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<MagneticFieldTaylor>> {
    config.validate()?;
    if order == usize::MAX
        || electric
            .iter()
            .chain(magnetic.iter())
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid electric and magnetic field Taylor request".into(),
        ));
    }
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "field response requires a converged SCF reference".into(),
        ));
    }
    if functional.required_nonlocal_kind() == Some(crate::xc::RequiredNonlocalTerm::Vv10)
        && opts.vv10.is_none()
    {
        return Err(DftError::InvalidInput(
            "the functional requires self-consistent VV10 options for field response".into(),
        ));
    }
    if order > 1 && opts.skala.is_some() {
        return Err(DftError::Unsupported {
            feature: "nonlinear field response of the learned functional".into(),
            missing: "the higher density derivatives of the network on the fixed grid".into(),
        });
    }
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    let nao = mb.nao_sph;
    let dips = dipole_matrices(mb, Vec3::zero());
    let angular = crate::properties::nmr::angular_momentum_operator(mb, origin);
    let moments = crate::integrals::one_electron::moment_matrices(mb, origin, 2);
    // quadratic[3 c + d] = r^2 delta_cd - r_c r_d about the origin.
    let mut square = Matrix::zeros(nao, nao);
    for c in 0..3 {
        add(&mut square, second_moment(&moments, c, c)?, 1.0);
    }
    let mut quadratic = Vec::with_capacity(9);
    for c in 0..3 {
        for d in 0..3 {
            let mut q = Matrix::zeros(nao, nao);
            if c == d {
                add(&mut q, &square, 1.0);
            }
            add(&mut q, second_moment(&moments, c, d)?, -1.0);
            quadratic.push(q);
        }
    }
    let mut field_re = Matrix::zeros(nao, nao);
    let mut field_im = Matrix::zeros(nao, nao);
    let mut diamagnetic = Matrix::zeros(nao, nao);
    for c in 0..3 {
        add(&mut field_re, &dips[c], electric[c]);
        add(&mut field_im, &angular[c], -0.5 * magnetic[c]);
        for d in 0..3 {
            add(
                &mut diamagnetic,
                &quadratic[3 * c + d],
                magnetic[c] * magnetic[d] / 8.0,
            );
        }
    }
    let n_occ = solver.dims.n_occ;
    let n_vir = solver.dims.n_vir;
    let nmo = [scf.mo_coeff[0].cols, scf.mo_coeff[1].cols];
    let mu0 = crate::properties::dipole_moment(mol, mb, scf);
    let mut dipole_coefficients = vec![[mu0.x, mu0.y, mu0.z]];
    let mut magnetic_coefficients = vec![[0.0; 3]];
    // Real and imaginary AO density coefficients per spin.
    let mut densities = vec![scf.density.clone()];
    let mut currents: Vec<[Matrix; 2]> = vec![[Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)]];
    let mut ds: Vec<[Pair; 2]> = vec![std::array::from_fn(|s| {
        let mut d = Pair::zeros(nmo[s]);
        for i in 0..n_occ[s] {
            d.re[(i, i)] = 1.0;
        }
        d
    })];
    let mut fs: Vec<[Pair; 2]> = vec![std::array::from_fn(|s| {
        let mut f = Pair::zeros(nmo[s]);
        for (i, &e) in scf.mo_energy[s].iter().enumerate() {
            f.re[(i, i)] = e;
        }
        f
    })];
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut projector_defects = Vec::new();
    let mut stationarity_defects = Vec::new();
    for n in 1..=order {
        // Idempotency fixes D_n,oo = -R_oo and D_n,vv = R_vv.
        let mut dn: [Pair; 2] = std::array::from_fn(|s| Pair::zeros(nmo[s]));
        let mut products = dn.clone();
        for s in 0..2 {
            for m in 1..n {
                products[s].add(&ds[m][s].product(&ds[n - m][s]), 1.0);
            }
            let no = n_occ[s];
            for i in 0..nmo[s] {
                for j in 0..nmo[s] {
                    let sign = if i < no && j < no {
                        -1.0
                    } else if i >= no && j >= no {
                        1.0
                    } else {
                        0.0
                    };
                    dn[s].re[(i, j)] = sign * products[s].re[(i, j)];
                    dn[s].im[(i, j)] = sign * products[s].im[(i, j)];
                }
            }
        }
        let diag_re: [Matrix; 2] = std::array::from_fn(|s| to_ao(&scf.mo_coeff[s], &dn[s].re));
        let diag_im: [Matrix; 2] = std::array::from_fn(|s| to_ao(&scf.mo_coeff[s], &dn[s].im));
        densities.push(diag_re.clone());
        let mut fn_re = solver.fock_response(&diag_re)?;
        if n > 1 {
            if let Some((params, nlc_grid, p0)) = &solver.nlc {
                let mut delta_total = diag_re[0].clone();
                add(&mut delta_total, &diag_re[1], 1.0);
                let linear =
                    crate::nlc::vv10_fxc_contraction(mb, nlc_grid, p0, &delta_total, *params)?;
                let totals: Vec<_> = densities
                    .iter()
                    .map(|p| {
                        let mut total = p[0].clone();
                        add(&mut total, &p[1], 1.0);
                        total
                    })
                    .collect();
                let nonlinear = super::nonlocal::vv10_taylor(mb, nlc_grid, &totals, *params)?;
                for s in 0..2 {
                    add(&mut fn_re[s], &linear, -1.0);
                    add(&mut fn_re[s], &nonlinear.fock_coefficients[n], 1.0);
                }
            }
        }
        if n > 1 && !functional.parts.is_empty() {
            let linear = solver.fxc_contraction_semilocal(&diag_re)?;
            let nonlinear = semilocal_fock_coefficient(
                mb,
                grid.expect("response grid checked"),
                functional,
                &densities,
                n,
            )?;
            for s in 0..2 {
                add(&mut fn_re[s], &linear[s], -1.0);
                add(&mut fn_re[s], &nonlinear[s], 1.0);
            }
        }
        let mut fn_im = exchange(&solver, &diag_im)?;
        if n == 1 {
            for s in 0..2 {
                add(&mut fn_re[s], &field_re, 1.0);
                add(&mut fn_im[s], &field_im, 1.0);
            }
        }
        if n == 2 {
            for s in 0..2 {
                add(&mut fn_re[s], &diamagnetic, 1.0);
            }
        }
        let mut fn_mo: [Pair; 2] = std::array::from_fn(|s| Pair {
            re: to_mo(&scf.mo_coeff[s], &fn_re[s]),
            im: to_mo(&scf.mo_coeff[s], &fn_im[s]),
        });
        let mut commutators: [Pair; 2] = std::array::from_fn(|s| Pair::zeros(nmo[s]));
        for s in 0..2 {
            for m in 1..n {
                commutators[s].add(&fs[m][s].product(&ds[n - m][s]), 1.0);
                commutators[s].add(&ds[n - m][s].product(&fs[m][s]), -1.0);
            }
        }
        let mut source_re = Vec::new();
        let mut source_im = Vec::new();
        for s in 0..solver.spins() {
            let no = n_occ[s];
            for a in 0..n_vir[s] {
                for i in 0..no {
                    source_re.push(fn_mo[s].re[(no + a, i)] + commutators[s].re[(no + a, i)]);
                    source_im.push(fn_mo[s].im[(no + a, i)] + commutators[s].im[(no + a, i)]);
                }
            }
        }
        let report = solver.solve_ov_report(&source_re, config)?;
        let x: Vec<f64> = report.amplitudes.iter().map(|u| u.re).collect();
        let (y, imaginary_residual) = solve_imaginary(&solver, &source_im, config)?;
        let off_re = solver.delta_density(&x);
        let off_im = imaginary_density(&solver, &y);
        let df_re = solver.fock_response(&off_re)?;
        let df_im = exchange(&solver, &off_im)?;
        let mut offset = 0;
        for s in 0..solver.spins() {
            let no = n_occ[s];
            for a in 0..n_vir[s] {
                for i in 0..no {
                    dn[s].re[(no + a, i)] = x[offset];
                    dn[s].re[(i, no + a)] = x[offset];
                    dn[s].im[(no + a, i)] = y[offset];
                    dn[s].im[(i, no + a)] = -y[offset];
                    offset += 1;
                }
            }
        }
        if solver.spins() == 1 {
            dn[1] = dn[0].clone();
        }
        let mut current: [Matrix; 2] = diag_im;
        for s in 0..2 {
            add(&mut fn_mo[s].re, &to_mo(&scf.mo_coeff[s], &df_re[s]), 1.0);
            add(&mut fn_mo[s].im, &to_mo(&scf.mo_coeff[s], &df_im[s]), 1.0);
            add(&mut densities[n][s], &off_re[s], 1.0);
            add(&mut current[s], &off_im[s], 1.0);
        }
        currents.push(current);
        let mut defect = 0.0_f64;
        for s in 0..2 {
            let mut r = products[s].clone();
            r.add(&ds[0][s].product(&dn[s]), 1.0);
            r.add(&dn[s].product(&ds[0][s]), 1.0);
            r.add(&dn[s], -1.0);
            defect = defect.max(r.largest());
        }
        dipole_coefficients.push(std::array::from_fn(|a| {
            -densities[n][0].frobenius_dot(&dips[a]) - densities[n][1].frobenius_dot(&dips[a])
        }));
        // m_c = Tr[I L_c] / 2 - sum_d B_d Tr[R (r^2 delta_cd - r_c r_d)] / 4.
        magnetic_coefficients.push(std::array::from_fn(|c| {
            let mut value = 0.5
                * (currents[n][0].frobenius_dot(&angular[c])
                    + currents[n][1].frobenius_dot(&angular[c]));
            for d in 0..3 {
                value -= 0.25
                    * magnetic[d]
                    * (densities[n - 1][0].frobenius_dot(&quadratic[3 * c + d])
                        + densities[n - 1][1].frobenius_dot(&quadratic[3 * c + d]));
            }
            value
        }));
        ds.push(dn);
        fs.push(fn_mo);
        let mut stationary = 0.0_f64;
        for s in 0..2 {
            let mut comm = commutators[s].clone();
            comm.add(&fs[0][s].product(&ds[n][s]), 1.0);
            comm.add(&fs[n][s].product(&ds[0][s]), 1.0);
            comm.add(&ds[n][s].product(&fs[0][s]), -1.0);
            comm.add(&ds[0][s].product(&fs[n][s]), -1.0);
            stationary = stationary.max(comm.largest());
        }
        stationarity_defects.push(stationary);
        projector_defects.push(defect);
        residuals.push(report.relative_residual.max(imaginary_residual));
        iterations.push(report.iterations);
    }
    Ok(ResponseReport {
        value: MagneticFieldTaylor {
            electric,
            magnetic,
            origin: origin.to_array(),
            dipole_coefficients,
            magnetic_coefficients,
            projector_defects,
            stationarity_defects,
        },
        shape: vec![order + 1, 3],
        functional: functional.description.clone(),
        reference: super::diagnostics::reference_diagnostics(scf)?,
        units: "atomic units; coefficient n = derivative n / n!",
        model: "static integer-occupation KS/HF density-projector response with a complex \
                Hermitian density; common gauge origin; adiabatic semilocal XC of the real density",
        axis_convention:
            "[Taylor order][Cartesian component]; E(t) = t electric, B(t) = t magnetic",
        converged: true,
        residuals,
        iterations,
        warnings: Vec::new(),
    })
}

/// Static polarizability, magnetizability and hypermagnetizability with a
/// common gauge origin.
#[derive(Clone, Debug)]
pub struct Hypermagnetizability {
    /// Gauge origin, Bohr.
    pub origin: [f64; 3],
    /// `alpha[a][b] = d mu_a / d E_b`.
    pub polarizability: [[f64; 3]; 3],
    /// `xi[c][d] = d m_c / d B_d = -d2E / (dB_c dB_d)`.
    pub magnetizability: [[f64; 3]; 3],
    /// `eta[a][b][c][d] = d3 mu_a / (dE_b dB_c dB_d)`: the second magnetic
    /// derivative of the polarizability.
    pub tensor: [[[[f64; 3]; 3]; 3]; 3],
    pub accumulation_roundoff: f64,
    pub directional_evaluations: usize,
    /// Largest relative residual of the response solves.
    pub residual: f64,
}

/// `eta_ab,cd = d2 alpha_ab / (dB_c dB_d)` at zero frequency from the mixed
/// third derivative of the induced dipole, by the polarization identity of
/// combined field directions. No finite field is involved.
#[allow(clippy::too_many_arguments)]
pub fn hypermagnetizability(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<Hypermagnetizability> {
    let mut residual = 0.0_f64;
    let mut evaluations = 0_usize;
    let mut polarizability = [[0.0; 3]; 3];
    let mut magnetizability = [[0.0; 3]; 3];
    for b in 0..3 {
        let mut unit = [0.0; 3];
        unit[b] = 1.0;
        let electric = magnetic_field_taylor(
            mol, mb, functional, grid, scf, opts, unit, [0.0; 3], origin, 1, config,
        )?;
        let magnetic = magnetic_field_taylor(
            mol, mb, functional, grid, scf, opts, [0.0; 3], unit, origin, 1, config,
        )?;
        evaluations += 2;
        for report in [&electric, &magnetic] {
            residual = report.residuals.iter().copied().fold(residual, f64::max);
        }
        for a in 0..3 {
            polarizability[a][b] = electric.value.dipole_coefficients[1][a];
            magnetizability[a][b] = magnetic.value.magnetic_coefficients[1][a];
        }
    }
    // Third-order dipole coefficients of the combined directions met so far.
    let mut cache: Vec<(Vec<f64>, [f64; 3])> = Vec::new();
    let mut tensor = [[[[0.0; 3]; 3]; 3]; 3];
    let mut roundoff = 0.0_f64;
    for b in 0..3 {
        for c in 0..3 {
            for d in c..3 {
                let mut vectors = vec![vec![0.0; 6]; 3];
                vectors[0][b] = 1.0;
                vectors[1][3 + c] = 1.0;
                vectors[2][3 + d] = 1.0;
                let mixed = super::mixed::polarized_mixed(6, &vectors, |direction, n| {
                    if let Some((_, value)) =
                        cache.iter().find(|(key, _)| key.as_slice() == direction)
                    {
                        return Ok(value.to_vec());
                    }
                    let report = magnetic_field_taylor(
                        mol,
                        mb,
                        functional,
                        grid,
                        scf,
                        opts,
                        [direction[0], direction[1], direction[2]],
                        [direction[3], direction[4], direction[5]],
                        origin,
                        n,
                        config,
                    )?;
                    residual = report.residuals.iter().copied().fold(residual, f64::max);
                    evaluations += 1;
                    let value = report.value.dipole_coefficients[n];
                    cache.push((direction.to_vec(), value));
                    Ok(value.to_vec())
                })?;
                for a in 0..3 {
                    tensor[a][b][c][d] = mixed.value[a];
                    tensor[a][b][d][c] = mixed.value[a];
                    roundoff = roundoff.max(mixed.accumulation_roundoff[a]);
                }
            }
        }
    }
    Ok(Hypermagnetizability {
        origin: origin.to_array(),
        polarizability,
        magnetizability,
        tensor,
        accumulation_roundoff: roundoff,
        directional_evaluations: evaluations,
        residual,
    })
}
