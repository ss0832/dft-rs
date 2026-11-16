// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Experimental one-electron Pauli SOC susceptibility nuclear jets for an
//! integer all-electron collinear HF reference with nominal S >= 1. No public driver is implied.
//! Differentiate the constrained orbital-response equations, including the
//! moving metric, transported occupied projector, ERIs and SOC operators.
//! Two-electron SOC, general spin projection, semilocal functionals and other
//! references require separate implementations. The spin Hamiltonian is
//! `H_D = S.D.S`; its orientation-independent second-order energy is retained
//! separately from the traceless anisotropy tensor.

use crate::basis::{n_sph, MolBasis};
use crate::constants::FINE_STRUCTURE;
use crate::dual::{Scalar, Taylor};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self, AnalyticNucleus, AnalyticShell};
use crate::linalg::{solve_linear_matrix, Matrix};
use crate::math::Vec3;
use crate::response::nuclear::spin_orbit_operators::spin_orbit_operator_path;
use crate::response::nuclear::{nuclear_taylor, NuclearTaylor};
use crate::response::ResponseSolverConfig;
use crate::scf::{Reference, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

type J = Taylor<3>;
fn j(x: f64) -> J {
    J::from_f64(x)
}

#[derive(Clone)]
struct M {
    r: usize,
    c: usize,
    v: Vec<J>,
}
impl M {
    fn zero(r: usize, c: usize) -> Self {
        Self {
            r,
            c,
            v: vec![j(0.0); r * c],
        }
    }
    fn eye(n: usize) -> Self {
        let mut a = Self::zero(n, n);
        for i in 0..n {
            a.put(i, i, j(1.0));
        }
        a
    }
    fn at(&self, i: usize, k: usize) -> J {
        self.v[i * self.c + k]
    }
    fn put(&mut self, i: usize, k: usize, x: J) {
        self.v[i * self.c + k] = x;
    }
    fn from_real(a: &Matrix) -> Self {
        Self {
            r: a.rows,
            c: a.cols,
            v: a.as_slice().iter().map(|&x| j(x)).collect(),
        }
    }
    fn t(&self) -> Self {
        let mut a = Self::zero(self.c, self.r);
        for i in 0..self.r {
            for k in 0..self.c {
                a.put(k, i, self.at(i, k));
            }
        }
        a
    }
    fn mul(&self, b: &Self) -> Self {
        assert_eq!(self.c, b.r);
        let mut a = Self::zero(self.r, b.c);
        for i in 0..a.r {
            for k in 0..a.c {
                let mut x = j(0.0);
                for l in 0..self.c {
                    x = x + self.at(i, l) * b.at(l, k);
                }
                a.put(i, k, x);
            }
        }
        a
    }
    fn add(&self, b: &Self, f: f64) -> Self {
        assert_eq!((self.r, self.c), (b.r, b.c));
        let mut a = self.clone();
        for (x, y) in a.v.iter_mut().zip(&b.v) {
            *x = *x + j(f) * *y;
        }
        a
    }
    fn scale(&self, f: f64) -> Self {
        let mut a = self.clone();
        for x in &mut a.v {
            *x = *x * j(f);
        }
        a
    }
    fn block(&self, r: usize, c: usize, nr: usize, nc: usize) -> Self {
        let mut a = Self::zero(nr, nc);
        for i in 0..nr {
            for k in 0..nc {
                a.put(i, k, self.at(r + i, c + k));
            }
        }
        a
    }
    fn coefficient(&self, k: usize) -> Matrix {
        let mut a = Matrix::zeros(self.r, self.c);
        for i in 0..self.r {
            for l in 0..self.c {
                a[(i, l)] = self.at(i, l).c[k];
            }
        }
        a
    }
}

#[derive(Clone)]
struct Z {
    re: M,
    im: M,
}
impl Z {
    fn zero(n: usize, m: usize) -> Self {
        Self {
            re: M::zero(n, m),
            im: M::zero(n, m),
        }
    }
    fn real(a: M) -> Self {
        let im = M::zero(a.r, a.c);
        Self { re: a, im }
    }
    fn adj(&self) -> Self {
        Self {
            re: self.re.t(),
            im: self.im.t().scale(-1.0),
        }
    }
    fn mul(&self, b: &Self) -> Self {
        Self {
            re: self.re.mul(&b.re).add(&self.im.mul(&b.im), -1.0),
            im: self.re.mul(&b.im).add(&self.im.mul(&b.re), 1.0),
        }
    }
    fn add(&self, b: &Self, f: f64) -> Self {
        Self {
            re: self.re.add(&b.re, f),
            im: self.im.add(&b.im, f),
        }
    }
    fn block(&self, r: usize, c: usize, nr: usize, nc: usize) -> Self {
        Self {
            re: self.re.block(r, c, nr, nc),
            im: self.im.block(r, c, nr, nc),
        }
    }
}

fn unsupported(message: &str) -> DftError {
    DftError::Unsupported {
        feature: message.into(),
        missing: "dedicated constrained SOC nuclear response acceptance".into(),
    }
}

struct Integrals {
    s: M,
    h: M,
    eri: Vec<J>,
    n: usize,
}
fn moving_integrals(mol: &Molecule, basis: &MolBasis, direction: &[[f64; 3]]) -> Result<Integrals> {
    let n = basis.nao_sph;
    let shells = basis
        .shells
        .iter()
        .map(|s| {
            AnalyticShell::from_placed(
                s,
                std::array::from_fn(|a| {
                    J::from_coefficients(&[s.center.get(a), direction[s.atom][a]])
                }),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let nuclei = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| AnalyticNucleus {
            center: std::array::from_fn(|k| {
                J::from_coefficients(&[a.position.get(k), direction[i][k]])
            }),
            charge: j(a.z as f64),
        })
        .collect::<Vec<_>>();
    let mut s = M::zero(n, n);
    let mut h = M::zero(n, n);
    let mut eri = vec![j(0.0); n * n * n * n];
    for (ia, a) in shells.iter().enumerate() {
        for (ib, b) in shells.iter().enumerate() {
            let overlap = analytic_basis::overlap(a, b)?;
            let kinetic = analytic_basis::kinetic(a, b)?;
            let attraction = analytic_basis::attraction(a, b, &nuclei, GaussianKernel::Coulomb, 2)?;
            for i in 0..n_sph(a.l) {
                for k in 0..n_sph(b.l) {
                    let q = i * n_sph(b.l) + k;
                    let u = basis.shells[ia].sph_off + i;
                    let v = basis.shells[ib].sph_off + k;
                    s.put(u, v, overlap.values[q]);
                    h.put(u, v, kinetic.values[q] + attraction.values[q]);
                }
            }
            for (ic, c) in shells.iter().enumerate() {
                for (id, d) in shells.iter().enumerate() {
                    let block = analytic_basis::repulsion(a, b, c, d, GaussianKernel::Coulomb, 2)?;
                    for i in 0..n_sph(a.l) {
                        for k in 0..n_sph(b.l) {
                            for l in 0..n_sph(c.l) {
                                for m in 0..n_sph(d.l) {
                                    let q =
                                        ((i * n_sph(b.l) + k) * n_sph(c.l) + l) * n_sph(d.l) + m;
                                    let u = basis.shells[ia].sph_off + i;
                                    let v = basis.shells[ib].sph_off + k;
                                    let w = basis.shells[ic].sph_off + l;
                                    let x = basis.shells[id].sph_off + m;
                                    eri[((u * n + v) * n + w) * n + x] = block.values[q];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Integrals { s, h, eri, n })
}

fn interaction(d: &Z, g: &Integrals) -> Z {
    let n = g.n;
    let mut out = Z::zero(2 * n, 2 * n);
    for a in 0..2 {
        for b in 0..2 {
            for i in 0..n {
                for k in 0..n {
                    let mut re = j(0.0);
                    let mut im = j(0.0);
                    for l in 0..n {
                        for m in 0..n {
                            let exchange = g.eri[((i * n + l) * n + k) * n + m];
                            re = re - exchange * d.re.at(a * n + l, b * n + m);
                            im = im - exchange * d.im.at(a * n + l, b * n + m);
                            if a == b {
                                let coulomb = g.eri[((i * n + k) * n + l) * n + m];
                                for spin in 0..2 {
                                    re = re + coulomb * d.re.at(spin * n + m, spin * n + l);
                                    im = im + coulomb * d.im.at(spin * n + m, spin * n + l);
                                }
                            }
                        }
                    }
                    out.re.put(a * n + i, b * n + k, re);
                    out.im.put(a * n + i, b * n + k, im);
                }
            }
        }
    }
    out
}

fn transported_frame(
    c0: &Matrix,
    s: &M,
    p: &M,
    occupied_count: usize,
    order: usize,
) -> (M, f64, f64) {
    let c = M::from_real(c0);
    let n = c.c;
    let metric = c.t().mul(s).mul(&c);
    let mut g1 = M::zero(n, n);
    let mut g2 = M::zero(n, n);
    for i in 0..n {
        for k in 0..n {
            g1.put(i, k, j(metric.at(i, k).c[1]));
            g2.put(i, k, j(metric.at(i, k).c[2]));
        }
    }
    let w1 = g1.scale(-0.5);
    let w2 = g2.scale(-0.5).add(&g1.mul(&g1), 0.375);
    let mut w = M::eye(n);
    for i in 0..n {
        for k in 0..n {
            w.put(
                i,
                k,
                J::from_coefficients(&[f64::from(i == k), w1.at(i, k).c[0], w2.at(i, k).c[0]]),
            );
        }
    }
    let b = c.mul(&w);
    let projector = b.t().mul(s).mul(p).mul(s).mul(&b);
    let mut d1 = M::zero(n, n);
    let mut d2 = M::zero(n, n);
    let mut sign = M::eye(n).scale(-1.0);
    for i in 0..occupied_count {
        sign.put(i, i, j(1.0));
    }
    for i in 0..n {
        for k in 0..n {
            d1.put(i, k, j(projector.at(i, k).c[1]));
            d2.put(i, k, j(projector.at(i, k).c[2]));
        }
    }
    let u1 = d1.mul(&sign);
    let u2 = d2.mul(&sign).add(&d1.mul(&d1), 0.5);
    let mut u = M::eye(n);
    for i in 0..n {
        for k in 0..n {
            u.put(
                i,
                k,
                J::from_coefficients(&[f64::from(i == k), u1.at(i, k).c[0], u2.at(i, k).c[0]]),
            );
        }
    }
    let frame = b.mul(&u);
    let check = frame.t().mul(s).mul(&frame).add(&M::eye(n), -1.0);
    let mut metric_error = 0.0_f64;
    for x in check.v {
        for v in x.c.iter().take(order + 1) {
            metric_error = metric_error.max(v.abs());
        }
    }
    let occupied = frame.block(0, 0, n, occupied_count);
    let density = occupied.mul(&occupied.t()).add(p, -1.0);
    let mut density_error = 0.0_f64;
    for x in density.v {
        for v in x.c.iter().take(order + 1) {
            density_error = density_error.max(v.abs());
        }
    }
    (frame, metric_error, density_error)
}

pub struct SocNuclearPath {
    /// Nominal effective spin; a contaminated unrestricted determinant is not spin purified.
    pub nominal_spin: f64,
    pub reference_s_squared: f64,
    pub spin_normalization: f64,
    /// Coefficient k is the kth nuclear derivative divided by k!, in
    /// Hartree/bohr^k. The spin Hamiltonian convention is H_D=S.D.S.
    pub coefficients: Vec<[[f64; 3]; 3]>,
    /// Orientation-independent second-order SOC energy, Hartree/bohr^k,
    /// with the same factorial normalization as the anisotropy tensor.
    pub orientation_independent_energy_coefficients: Vec<f64>,
    /// Raw unscaled nine-source susceptibility coefficients; source order
    /// is orbital axis first, Pauli spin axis second. No SOC prefactor.
    pub susceptibility_coefficients: Vec<Matrix>,
    /// Absolute residuals of the mixed orbital/constraint KKT equations.
    pub kkt_residuals: Vec<f64>,
    pub metric_error: f64,
    pub density_error: f64,
    pub response: crate::response::ResponseReport<NuclearTaylor>,
}

/// Return analytic directional SOC anisotropy coefficients through order two.
/// A full-rank AO basis with 3..=12 functions bounds the initial dense tangent
/// implementation. Finite differences are used only in independent tests.
#[allow(clippy::too_many_arguments)]
pub fn soc_nuclear_path(
    mol: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    reference: &ScfResult,
    options: &ScfOptions,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<SocNuclearPath> {
    soc_nuclear_path_in_frame(
        mol, basis, functional, reference, options, direction, order, config, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn soc_nuclear_path_in_frame(
    mol: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    reference: &ScfResult,
    options: &ScfOptions,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
    frame_rotation: Option<&Matrix>,
) -> Result<SocNuclearPath> {
    if order > 2 {
        return Err(unsupported("SOC ZFS nuclear order above two"));
    }
    if !reference.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: f64::NAN,
        });
    }
    if !reference.has_integer_occupations() {
        return Err(unsupported("ensemble SOC reference"));
    }
    if mol.n_alpha_beta()? != (reference.n_alpha, reference.n_beta) {
        return Err(DftError::InvalidInput(
            "SOC molecular electron counts disagree with the reference".into(),
        ));
    }
    if options.reference != Reference::Uhf
        || !reference.unrestricted
        || reference.n_alpha < reference.n_beta.saturating_add(2)
        || !basis.ecps.is_empty()
        || !functional.parts.is_empty()
        || functional.hybrid.cx != 1.0
        || functional.hybrid.clr != 0.0
        || functional.hybrid.omega != 0.0
        || options.electronic_temperature != 0.0
        || reference.electronic_temperature != 0.0
    {
        return Err(unsupported(
            "SOC ZFS requires an integer all-electron collinear HF reference with nominal S >= 1",
        ));
    }
    if options.external_field.is_some()
        || options.extra_hcore.is_some()
        || options.asymptotic_correction.is_some()
        || options.pcm.is_some()
        || options.vv10.is_some()
        || options.skala.is_some()
        || options.density_fit.is_some()
        || options.electron_offset != [0.0; 2]
        || !matches!(options.kernel, crate::integrals::CoulombKernel::Coulomb)
    {
        return Err(unsupported(
            "SOC ZFS environment, non-Coulomb or modified-core response",
        ));
    }
    let n = basis.nao_sph;
    if direction.len() != mol.len()
        || direction.iter().flatten().any(|x| !x.is_finite())
        || reference
            .mo_coeff
            .iter()
            .any(|c| c.rows != n || c.cols != n)
        || basis.shells.iter().any(|s| {
            s.atom >= mol.len() || s.sph_off.checked_add(n_sph(s.l)).is_none_or(|end| end > n)
        })
    {
        return Err(DftError::InvalidInput(
            "SOC nuclear response requires finite per-atom directions and a complete AO layout"
                .into(),
        ));
    }
    if !(3..=12).contains(&n) || reference.mo_coeff[0].cols != n {
        return Err(unsupported(
            "SOC ZFS requires a finite full-rank bounded AO path",
        ));
    }
    if reference.n_alpha > n || reference.n_beta > n {
        return Err(DftError::InvalidInput(
            "SOC occupations exceed the orbital dimensions".into(),
        ));
    }
    let occupied_count = reference.n_alpha + reference.n_beta;
    let nominal_spin = 0.5 * (reference.n_alpha - reference.n_beta) as f64;
    let spin_normalization = nominal_spin * (2.0 * nominal_spin - 1.0);
    if frame_rotation.is_some() && reference.n_beta != 0 {
        return Err(unsupported(
            "paired-spin orbital-frame rotations require separate spin gauges",
        ));
    }
    let tangent_dimension = 2 * occupied_count * (2 * n - occupied_count) + 2;
    let eri_bytes = crate::resources::checked_bytes(
        &[n, n, n, n, std::mem::size_of::<J>()],
        "SOC nuclear ERI workspace",
    )?;
    let tangent_bytes = crate::resources::checked_bytes(
        &[
            tangent_dimension,
            tangent_dimension,
            std::mem::size_of::<J>(),
            12,
        ],
        "SOC nuclear tangent workspace",
    )?;
    let bytes = eri_bytes
        .checked_add(tangent_bytes)
        .ok_or_else(|| DftError::InvalidInput("SOC nuclear workspace size overflow".into()))?;
    let _workspace = crate::resources::reserve(bytes, "SOC nuclear ERI and tangent workspace")?;
    let response = nuclear_taylor(
        mol, basis, functional, None, reference, options, direction, order, config,
    )?;
    if !response.converged {
        return Err(response_failure(&response.iterations, &response.residuals));
    }
    let integrals = moving_integrals(mol, basis, direction)?;
    let densities: [M; 2] = std::array::from_fn(|spin| {
        let mut density = M::zero(n, n);
        for i in 0..n {
            for k in 0..n {
                let mut coefficients = [0.0; 3];
                for (degree, value) in coefficients.iter_mut().enumerate().take(order + 1) {
                    *value = response.value.density_coefficients[degree][spin][(i, k)]
                        + response.value.density_corrections[degree][spin][(i, k)];
                }
                density.put(i, k, J::from_coefficients(&coefficients));
            }
        }
        density
    });
    let (mut frame, mut metric_error, mut density_error) = transported_frame(
        &reference.mo_coeff[0],
        &integrals.s,
        &densities[0],
        reference.n_alpha,
        order,
    );
    let beta_frame = if reference.n_beta == 0 {
        None
    } else {
        let (frame, metric, density) = transported_frame(
            &reference.mo_coeff[1],
            &integrals.s,
            &densities[1],
            reference.n_beta,
            order,
        );
        metric_error = metric_error.max(metric);
        density_error = density_error.max(density);
        Some(frame)
    };
    if let Some(rotation) = frame_rotation {
        if rotation.rows != n || rotation.cols != n {
            return Err(DftError::InvalidInput(
                "SOC response frame rotation has wrong dimensions".into(),
            ));
        }
        for i in 0..n {
            for k in 0..n {
                let dot = (0..n)
                    .map(|l| rotation[(l, i)] * rotation[(l, k)])
                    .sum::<f64>();
                if !dot.is_finite()
                    || (dot - f64::from(i == k)).abs() > 1e-12
                    || ((i < reference.n_alpha) != (k < reference.n_alpha)
                        && rotation[(i, k)] != 0.0)
                {
                    return Err(DftError::InvalidInput(
                        "SOC response frame must preserve metric and occupied subspace".into(),
                    ));
                }
            }
        }
        frame = frame.mul(&M::from_real(rotation));
    }
    let frames = [frame.clone(), beta_frame.unwrap_or(frame)];
    let counts = [reference.n_alpha, reference.n_beta];
    let labels: Vec<_> = (0..2)
        .flat_map(|spin| (0..counts[spin]).map(move |i| (spin, i)))
        .chain((0..2).flat_map(|spin| (counts[spin]..n).map(move |i| (spin, i))))
        .collect();
    let mut full_frame = M::zero(2 * n, 2 * n);
    for (column, (spin, k)) in labels.into_iter().enumerate() {
        for i in 0..n {
            full_frame.put(spin * n + i, column, frames[spin].at(i, k));
        }
    }
    let c = Z::real(full_frame);
    let occupied = c.block(0, 0, 2 * n, occupied_count);
    let virtuals = c.block(0, occupied_count, 2 * n, 2 * n - occupied_count);
    let mut p = Z::zero(2 * n, 2 * n);
    for spin in 0..2 {
        for i in 0..n {
            for k in 0..n {
                p.re.put(spin * n + i, spin * n + k, densities[spin].at(i, k));
            }
        }
    }
    let mut core = Z::zero(2 * n, 2 * n);
    for spin in 0..2 {
        for i in 0..n {
            for k in 0..n {
                core.re
                    .put(spin * n + i, spin * n + k, integrals.h.at(i, k));
            }
        }
    }
    let fock = core.add(&interaction(&p, &integrals), 1.0);
    let fm = c.adj().mul(&fock).mul(&c);
    let nv = 2 * n - occupied_count;
    let nt = nv * occupied_count;
    let dim = 2 * nt + 2;
    let fvv = fm.block(occupied_count, occupied_count, nv, nv);
    let occupied_fock = fm.block(0, 0, occupied_count, occupied_count);
    let mut operators = Vec::new();
    let mut orbital = [M::zero(n, n), M::zero(n, n), M::zero(n, n)];
    for atom in 0..mol.len() {
        let path = spin_orbit_operator_path(
            mol,
            basis,
            atom,
            direction,
            Vec3::new(0.0, 0.0, 0.0),
            [0.0; 3],
            false,
            order,
        )?;
        for (axis, matrix) in orbital.iter_mut().enumerate() {
            for i in 0..n {
                for k in 0..n {
                    let mut coeff = [0.0; 3];
                    for (degree, x) in coeff.iter_mut().enumerate().take(order + 1) {
                        *x = path.paramagnetic[axis][degree][(i, k)] * f64::from(mol.atoms[atom].z);
                    }
                    matrix.put(i, k, matrix.at(i, k) + J::from_coefficients(&coeff));
                }
            }
        }
    }
    for matrix in &orbital {
        for spin in 0..3 {
            let mut op = Z::zero(2 * n, 2 * n);
            for i in 0..n {
                for k in 0..n {
                    let v = matrix.at(i, k);
                    match spin {
                        0 => {
                            op.im.put(i, n + k, -v);
                            op.im.put(n + i, k, -v);
                        }
                        1 => {
                            op.re.put(i, n + k, -v);
                            op.re.put(n + i, k, v);
                        }
                        _ => {
                            op.im.put(i, k, -v);
                            op.im.put(n + i, n + k, v);
                        }
                    }
                }
            }
            operators.push(op);
        }
    }
    let mut constraints = vec![Z::zero(2 * n, 2 * n); 2];
    for i in 0..n {
        for k in 0..n {
            let v = integrals.s.at(i, k) * j(0.5);
            constraints[0].re.put(i, n + k, v);
            constraints[0].re.put(n + i, k, v);
            constraints[1].im.put(i, n + k, -v);
            constraints[1].im.put(n + i, k, v);
        }
    }
    let flatten = |matrix: &Z| {
        let mut v = matrix.re.v.clone();
        v.extend_from_slice(&matrix.im.v);
        v.into_iter().map(|x| x * j(2.0)).collect::<Vec<_>>()
    };
    let sources = operators
        .iter()
        .map(|op| flatten(&virtuals.adj().mul(op).mul(&occupied)))
        .collect::<Vec<_>>();
    let constraint_rows = constraints
        .iter()
        .map(|op| flatten(&virtuals.adj().mul(op).mul(&occupied)))
        .collect::<Vec<_>>();
    let mut kkt = M::zero(dim, dim);
    for column in 0..2 * nt {
        let mut rotation = Z::zero(nv, occupied_count);
        if column < nt {
            rotation.re.v[column] = j(1.0);
        } else {
            rotation.im.v[column - nt] = j(1.0);
        }
        let tangent = virtuals.mul(&rotation).mul(&occupied.adj());
        let dp = tangent.add(&tangent.adj(), 1.0);
        let action = fvv
            .mul(&rotation)
            .add(&rotation.mul(&occupied_fock), -1.0)
            .add(
                &virtuals
                    .adj()
                    .mul(&interaction(&dp, &integrals))
                    .mul(&occupied),
                1.0,
            );
        for (row, value) in flatten(&action).iter().enumerate() {
            kkt.put(row, column, *value);
        }
    }
    for (constraint, row) in constraint_rows.iter().enumerate() {
        for (i, value) in row.iter().enumerate() {
            kkt.put(i, 2 * nt + constraint, *value);
            kkt.put(2 * nt + constraint, i, *value);
        }
    }
    let mut rhs = M::zero(dim, 9);
    for (source, values) in sources.iter().enumerate() {
        for (i, value) in values.iter().enumerate() {
            rhs.put(i, source, -*value);
        }
    }
    let base = kkt.coefficient(0);
    let mut solutions: Vec<Matrix> = Vec::new();
    let mut kkt_residuals = Vec::new();
    for degree in 0..=order {
        let mut target = rhs.coefficient(degree);
        for k in 1..=degree {
            let product = kkt.coefficient(k).matmul(&solutions[degree - k]);
            for i in 0..dim {
                for source in 0..9 {
                    target[(i, source)] -= product[(i, source)];
                }
            }
        }
        let solution = solve_linear_matrix(&base, &target)?;
        let check = base.matmul(&solution);
        let residual = check.max_abs_diff(&target);
        kkt_residuals.push(residual);
        solutions.push(solution);
    }
    let mut chi = M::zero(9, 9);
    for (a, source) in sources.iter().enumerate() {
        for b in 0..9 {
            let mut value = j(0.0);
            for (i, source_value) in source.iter().enumerate() {
                let coeff = (0..=order)
                    .map(|degree| solutions[degree][(i, b)])
                    .collect::<Vec<_>>();
                value = value + *source_value * J::from_coefficients(&coeff);
            }
            chi.put(a, b, value);
        }
    }
    let scale = (FINE_STRUCTURE.powi(2) / 4.0).powi(2);
    let mut coefficients = Vec::new();
    let mut orientation_independent_energy_coefficients = Vec::new();
    for degree in 0..=order {
        let raw: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                chi.at(3 * a + 2, 3 * b + 2).c[degree] - chi.at(3 * a, 3 * b).c[degree]
            })
        });
        let isotropic = (0..3).map(|a| raw[a][a]).sum::<f64>() / 3.0;
        let transverse_trace = (0..3).map(|a| chi.at(3 * a, 3 * a).c[degree]).sum::<f64>();
        orientation_independent_energy_coefficients
            .push(0.5 * scale * (transverse_trace + isotropic));
        coefficients.push(std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                scale / spin_normalization * (raw[a][b] - if a == b { isotropic } else { 0.0 })
            })
        }));
    }
    Ok(SocNuclearPath {
        nominal_spin,
        reference_s_squared: reference.s_squared,
        spin_normalization,
        coefficients,
        orientation_independent_energy_coefficients,
        susceptibility_coefficients: (0..=order).map(|degree| chi.coefficient(degree)).collect(),
        kkt_residuals,
        metric_error,
        density_error,
        response,
    })
}

/// Full Cartesian derivatives of the limited one-electron Pauli SOC model.
/// All requested nuclear response paths and their raw diagnostics are kept.
pub struct SocNuclearDerivatives {
    pub order: usize,
    pub value: [[f64; 3]; 3],
    /// Hartree/bohr, [3*atom+axis][spin row][spin column]. None if unrequested.
    pub gradient: Option<Vec<[[f64; 3]; 3]>>,
    /// Hartree/bohr^2, [coordinate][coordinate][spin row][spin column].
    pub hessian: Option<Vec<Vec<[[f64; 3]; 3]>>>,
    pub orientation_independent_energy: f64,
    pub orientation_independent_gradient: Option<Vec<f64>>,
    pub orientation_independent_hessian: Option<Vec<Vec<f64>>>,
    pub paths: Vec<SocNuclearPath>,
    /// This lease covers tensor/scalar outputs and path structs. The Matrix
    /// payloads use their own reservations; nested diagnostic containers and
    /// integral scratch are not represented as a hard process RSS bound.
    _output_memory: crate::resources::MemoryReservation,
}

#[allow(clippy::too_many_arguments)]
pub fn soc_nuclear_derivatives(
    mol: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    reference: &ScfResult,
    options: &ScfOptions,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<SocNuclearDerivatives> {
    if order > 2 {
        return Err(unsupported("Cartesian SOC nuclear order above two"));
    }
    if mol.is_empty() {
        return Err(DftError::InvalidInput(
            "Cartesian SOC requires a nonempty molecule".into(),
        ));
    }
    if mol.len() > 12 {
        return Err(unsupported(
            "Cartesian SOC currently requires one through twelve nuclei",
        ));
    }
    let n = 3 * mol.len();
    let npaths = if order == 0 {
        1
    } else if order == 1 {
        n
    } else {
        n * (n + 1) / 2
    };
    let elements = 1 + if order > 0 { n } else { 0 } + if order > 1 { n * n } else { 0 };
    let tensors = crate::resources::checked_bytes(
        &[elements, 10, 8],
        "Cartesian SOC tensor and scalar outputs",
    )?;
    let path_bytes = crate::resources::checked_bytes(
        &[npaths, std::mem::size_of::<SocNuclearPath>()],
        "Cartesian SOC path structs",
    )?;
    let row_bytes = if order > 1 {
        crate::resources::checked_bytes(
            &[
                n,
                std::mem::size_of::<Vec<[[f64; 3]; 3]>>() + std::mem::size_of::<Vec<f64>>(),
            ],
            "Cartesian SOC row vectors",
        )?
    } else {
        0
    };
    let coefficient_bytes = crate::resources::checked_bytes(
        &[
            npaths,
            3,
            std::mem::size_of::<[[f64; 3]; 3]>()
                + 2 * std::mem::size_of::<f64>()
                + std::mem::size_of::<Matrix>(),
        ],
        "Cartesian SOC retained coefficient containers",
    )?;
    let bytes = tensors
        .checked_add(path_bytes)
        .and_then(|v| v.checked_add(row_bytes))
        .and_then(|v| v.checked_add(coefficient_bytes))
        .ok_or_else(|| DftError::InvalidInput("Cartesian SOC size overflow".into()))?;
    let output_memory = crate::resources::reserve(bytes, "Cartesian SOC outputs")?;
    let mut paths = Vec::with_capacity(npaths);
    let mut gradient = (order > 0).then(|| vec![[[0.0; 3]; 3]; n]);
    let mut hessian = (order > 1).then(|| vec![vec![[[0.0; 3]; 3]; n]; n]);
    let mut scalar_gradient = (order > 0).then(|| vec![0.0; n]);
    let mut scalar_hessian = (order > 1).then(|| vec![vec![0.0; n]; n]);
    if order == 0 {
        paths.push(soc_nuclear_path(
            mol,
            basis,
            functional,
            reference,
            options,
            &vec![[0.0; 3]; mol.len()],
            0,
            config,
        )?);
    } else {
        for coordinate in 0..n {
            let mut direction = vec![[0.0; 3]; mol.len()];
            direction[coordinate / 3][coordinate % 3] = 1.0;
            let path = soc_nuclear_path(
                mol, basis, functional, reference, options, &direction, order, config,
            )?;
            gradient.as_mut().expect("requested gradient")[coordinate] = path.coefficients[1];
            scalar_gradient.as_mut().expect("requested gradient")[coordinate] =
                path.orientation_independent_energy_coefficients[1];
            if order == 2 {
                for a in 0..3 {
                    for b in 0..3 {
                        hessian.as_mut().expect("requested Hessian")[coordinate][coordinate][a]
                            [b] = 2.0 * path.coefficients[2][a][b];
                    }
                }
                scalar_hessian.as_mut().expect("requested Hessian")[coordinate][coordinate] =
                    2.0 * path.orientation_independent_energy_coefficients[2];
            }
            paths.push(path);
        }
        if order == 2 {
            for i in 0..n {
                for k in i + 1..n {
                    let mut direction = vec![[0.0; 3]; mol.len()];
                    direction[i / 3][i % 3] = std::f64::consts::FRAC_1_SQRT_2;
                    direction[k / 3][k % 3] = std::f64::consts::FRAC_1_SQRT_2;
                    let path = soc_nuclear_path(
                        mol, basis, functional, reference, options, &direction, 2, config,
                    )?;
                    let h = hessian.as_mut().expect("requested Hessian");
                    for a in 0..3 {
                        for b in 0..3 {
                            let value = 2.0 * path.coefficients[2][a][b]
                                - 0.5 * (h[i][i][a][b] + h[k][k][a][b]);
                            h[i][k][a][b] = value;
                            h[k][i][a][b] = value;
                        }
                    }
                    let h = scalar_hessian.as_mut().expect("requested Hessian");
                    let value = 2.0 * path.orientation_independent_energy_coefficients[2]
                        - 0.5 * (h[i][i] + h[k][k]);
                    h[i][k] = value;
                    h[k][i] = value;
                    paths.push(path);
                }
            }
        }
    }
    assert_eq!(paths.len(), npaths);
    Ok(SocNuclearDerivatives {
        order,
        value: paths[0].coefficients[0],
        gradient,
        hessian,
        orientation_independent_energy: paths[0].orientation_independent_energy_coefficients[0],
        orientation_independent_gradient: scalar_gradient,
        orientation_independent_hessian: scalar_hessian,
        paths,
        _output_memory: output_memory,
    })
}

fn response_failure(iterations: &[usize], residuals: &[f64]) -> DftError {
    let residual = residuals
        .iter()
        .copied()
        .find(|r| !r.is_finite())
        .unwrap_or_else(|| {
            if residuals.is_empty() {
                f64::NAN
            } else {
                residuals.iter().copied().fold(0.0_f64, f64::max)
            }
        });
    DftError::ResponseNotConverged {
        iterations: iterations.iter().copied().max().unwrap_or(0),
        residual,
    }
}

#[cfg(test)]
mod error_contract_tests {
    use super::{response_failure, DftError};
    #[test]
    fn failed_response_is_solver_failure_not_missing_capability() {
        assert!(
            matches!(response_failure(&[3,7],&[0.1,0.01]),DftError::ResponseNotConverged{iterations:7,residual} if residual==0.1)
        );
        assert!(
            matches!(response_failure(&[],&[]),DftError::ResponseNotConverged{iterations:0,residual} if residual.is_nan())
        );
    }
}
