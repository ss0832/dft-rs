// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static charge and separable-projector kernels for generated fragments.
//! No projector is an orbital AO or carries an electron. Electrostatic
//! corrections are independent of short-range electronic operators.

use crate::basis::{aos, MolBasis};
use crate::dual::{DualN, Scalar};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self, AnalyticNucleus, AnalyticShell};
use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::{DftError, Result};
use std::f64::consts::PI;

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn finite(v: Vec3) -> bool {
    (0..3).all(|axis| v.get(axis).is_finite())
}
fn coordinates<S: Scalar>(v: Vec3) -> [S; 3] {
    [S::from_f64(v.x), S::from_f64(v.y), S::from_f64(v.z)]
}

/// q*g_eta(r-C) - p.grad(g_eta(r-C)), with g_eta normalized to one.
/// q may be signed. The integral is q and the intrinsic dipole is p.
#[derive(Clone, Debug)]
pub struct GaussianCharge {
    pub center: Vec3,
    pub exponent: f64,
    pub charge: f64,
    pub dipole: Vec3,
}

impl GaussianCharge {
    pub fn validate(&self) -> Result<()> {
        if !finite(self.center)
            || !finite(self.dipole)
            || !self.charge.is_finite()
            || !self.exponent.is_finite()
            || self.exponent <= 0.0
        {
            return invalid(
                "Gaussian charges require finite centers/moments and positive exponents",
            );
        }
        Ok(())
    }
}

/// Raw moments; no renormalization or charge repair is performed.
pub fn charge_moments(sites: &[GaussianCharge], origin: Vec3) -> Result<(f64, Vec3)> {
    if !finite(origin) {
        return invalid("charge moment origin must be finite");
    }
    let mut q = 0.0;
    let mut p = Vec3::zero();
    for site in sites {
        site.validate()?;
        q += site.charge;
        p += (site.center - origin) * site.charge + site.dipole;
    }
    if !q.is_finite() || !finite(p) {
        return invalid("Gaussian charge moments overflowed");
    }
    Ok((q, p))
}

// Entire analytic function of r^2. The convergent origin series avoids
// differentiating sqrt(r^2) at zero; its omitted tail is below double precision.
fn radial<S: Scalar>(r: [S; 3], exponent: f64) -> S {
    let r2 = r.into_iter().fold(S::from_f64(0.0), |sum, x| sum + x * x);
    let t = r2.scale(exponent);
    if t.value() < 1e-6 {
        let mut sum = S::from_f64(1.0);
        let mut term = S::from_f64(1.0);
        for k in 1..=12 {
            term = term * (-t).scale(1.0 / k as f64);
            sum = sum + term.scale(1.0 / (2 * k + 1) as f64);
        }
        sum.scale(2.0 * (exponent / PI).sqrt())
    } else {
        let distance = r2.sqrt();
        (distance.scale(exponent.sqrt())).erf() / distance
    }
}

fn potential<S: Scalar>(site: &GaussianCharge, center: [S; 3], point: [S; 3]) -> (S, [S; 3]) {
    type Inner<T> = DualN<T, 3>;
    let shifted = std::array::from_fn(|axis| {
        Inner::constant(point[axis]) - Inner::variable(center[axis], axis)
    });
    let value = radial(shifted, site.exponent);
    let mut phi = value.re.scale(site.charge);
    for axis in 0..3 {
        phi = phi + value.im[axis].scale(site.dipole.get(axis));
    }
    (phi, value.im)
}

/// Physical potential at probes, excluding any separately handled point ions.
pub fn electrostatic_potential(sites: &[GaussianCharge], points: &[Vec3]) -> Result<Vec<f64>> {
    if points.iter().any(|&p| !finite(p)) {
        return invalid("charge probes must be finite");
    }
    for site in sites {
        site.validate()?;
    }
    Ok(points
        .iter()
        .map(|&point| {
            sites
                .iter()
                .map(|site| potential::<f64>(site, coordinates(site.center), coordinates(point)).0)
                .sum()
        })
        .collect())
}

pub struct ChargeOperator {
    pub operator: Matrix,
    pub atom_derivatives: Vec<[Matrix; 3]>,
    pub center_derivatives: Vec<[Matrix; 3]>,
    pub charge_operators: Vec<Matrix>,
    pub dipole_operators: Vec<[Matrix; 3]>,
}

fn lift<S: Scalar>(shell: &AnalyticShell<S>) -> AnalyticShell<DualN<S, 3>> {
    AnalyticShell {
        l: shell.l,
        center: shell.center.map(DualN::constant),
        exponents: shell
            .exponents
            .iter()
            .copied()
            .map(DualN::constant)
            .collect(),
        coefficients: shell
            .coefficients
            .iter()
            .copied()
            .map(DualN::constant)
            .collect(),
    }
}

/// Electron attraction of the SAME normalized charge distribution used by
/// `core_interactions` and `field_interaction`. Dipoles are center derivatives
/// of the normalized monopole, differentiated again for analytic forces.
pub fn charge_operator(
    basis: &MolBasis,
    atoms: usize,
    sites: &[GaussianCharge],
) -> Result<ChargeOperator> {
    if basis.shells.iter().any(|shell| shell.atom >= atoms) {
        return invalid("charge operator AO atom index exceeds atom count");
    }
    for site in sites {
        site.validate()?;
    }
    let n = basis.nao_sph;
    let matrix = || Matrix::zeros(n, n);
    let mut result = ChargeOperator {
        operator: matrix(),
        atom_derivatives: (0..atoms)
            .map(|_| std::array::from_fn(|_| matrix()))
            .collect(),
        center_derivatives: sites
            .iter()
            .map(|_| std::array::from_fn(|_| matrix()))
            .collect(),
        charge_operators: sites.iter().map(|_| matrix()).collect(),
        dipole_operators: sites
            .iter()
            .map(|_| std::array::from_fn(|_| matrix()))
            .collect(),
    };
    type Outer = DualN<f64, 9>;
    type Inner = DualN<Outer, 3>;
    for (site_index, site) in sites.iter().enumerate() {
        let mut site_value = matrix();
        let mut site_atoms: Vec<[Matrix; 3]> = (0..atoms)
            .map(|_| std::array::from_fn(|_| matrix()))
            .collect();
        for sa in 0..basis.shells.len() {
            for sb in sa..basis.shells.len() {
                let a = &basis.shells[sa];
                let b = &basis.shells[sb];
                let a_center =
                    std::array::from_fn(|axis| Outer::variable(a.center.get(axis), axis));
                let b_center =
                    std::array::from_fn(|axis| Outer::variable(b.center.get(axis), 3 + axis));
                let center = std::array::from_fn(|axis| {
                    Inner::variable(Outer::variable(site.center.get(axis), 6 + axis), axis)
                });
                let aa = lift(&AnalyticShell::from_placed(a, a_center)?);
                let bb = lift(&AnalyticShell::from_placed(b, b_center)?);
                let block = analytic_basis::attraction(
                    &aa,
                    &bb,
                    &[AnalyticNucleus {
                        center,
                        charge: Inner::constant(Outer::constant(1.0)),
                    }],
                    GaussianKernel::LongRange {
                        omega: Inner::constant(Outer::constant(site.exponent.sqrt())),
                    },
                    2,
                )?;
                for i in 0..block.shape[0] {
                    for j in 0..block.shape[1] {
                        let raw = block.values[i * block.shape[1] + j];
                        let mut value = raw.re.scale(site.charge);
                        for axis in 0..3 {
                            value = value + raw.im[axis].scale(site.dipole.get(axis));
                        }
                        let row = a.sph_off + i;
                        let col = b.sph_off + j;
                        for transpose in 0..if row == col { 1 } else { 2 } {
                            let (r, c) = if transpose == 0 {
                                (row, col)
                            } else {
                                (col, row)
                            };
                            site_value[(r, c)] = value.re;
                            result.charge_operators[site_index][(r, c)] = raw.re.re;
                            for axis in 0..3 {
                                result.center_derivatives[site_index][axis][(r, c)] =
                                    value.im[6 + axis];
                                for (atom, derivative) in site_atoms.iter_mut().enumerate() {
                                    derivative[axis][(r, c)] =
                                        if atom == a.atom { value.im[axis] } else { 0. }
                                            + if atom == b.atom {
                                                value.im[3 + axis]
                                            } else {
                                                0.
                                            };
                                }
                                result.dipole_operators[site_index][axis][(r, c)] = raw.im[axis].re;
                            }
                        }
                    }
                }
            }
        }
        for (v, &w) in result
            .operator
            .as_mut_slice()
            .iter_mut()
            .zip(site_value.as_slice())
        {
            *v += w;
        }
        for (target, source) in result.atom_derivatives.iter_mut().zip(site_atoms) {
            for axis in 0..3 {
                for (v, &w) in target[axis]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(source[axis].as_slice())
                {
                    *v += w;
                }
            }
        }
    }
    Ok(result)
}

pub struct CoreInteraction {
    pub energy: f64,
    pub core_gradients: Vec<Vec3>,
    pub center_gradients: Vec<Vec3>,
    pub charge_gradients: Vec<f64>,
    pub dipole_gradients: Vec<Vec3>,
}

/// Pair each listed retained core with each Gaussian correction once. The
/// caller excludes removed nuclei and the distribution's own point-ion self
/// term. Foreign point ions are included once; their baseline point-point
/// interaction remains in the ordinary core energy.
pub fn core_interactions(
    sites: &[GaussianCharge],
    cores: &[(Vec3, f64)],
) -> Result<CoreInteraction> {
    for site in sites {
        site.validate()?;
    }
    if cores.iter().any(|&(r, q)| !finite(r) || !q.is_finite()) {
        return invalid("core interactions require finite positions and charges");
    }
    let mut out = CoreInteraction {
        energy: 0.,
        core_gradients: vec![Vec3::zero(); cores.len()],
        center_gradients: vec![Vec3::zero(); sites.len()],
        charge_gradients: vec![0.; sites.len()],
        dipole_gradients: vec![Vec3::zero(); sites.len()],
    };
    type D = DualN<f64, 6>;
    for (s, site) in sites.iter().enumerate() {
        for (c, &(position, q)) in cores.iter().enumerate() {
            let center = std::array::from_fn(|axis| D::variable(site.center.get(axis), axis));
            let point = std::array::from_fn(|axis| D::variable(position.get(axis), 3 + axis));
            let (value, dp) = potential(site, center, point);
            out.energy += q * value.re;
            out.center_gradients[s] += Vec3::new(value.im[0], value.im[1], value.im[2]) * q;
            out.core_gradients[c] += Vec3::new(value.im[3], value.im[4], value.im[5]) * q;
            let unit = GaussianCharge {
                charge: 1.,
                dipole: Vec3::zero(),
                ..site.clone()
            };
            out.charge_gradients[s] +=
                q * potential::<f64>(&unit, coordinates(site.center), coordinates(position)).0;
            out.dipole_gradients[s] += Vec3::new(dp[0].re, dp[1].re, dp[2].re) * q;
        }
    }
    Ok(out)
}

/// Uniform field coupling, including permanent intrinsic dipoles.
pub fn field_interaction(
    sites: &[GaussianCharge],
    origin: Vec3,
    field: Vec3,
) -> Result<CoreInteraction> {
    if !finite(field) || !finite(origin) {
        return invalid("field and origin must be finite");
    }
    let (_, dipole) = charge_moments(sites, origin)?;
    Ok(CoreInteraction {
        energy: -dipole.dot(field),
        core_gradients: Vec::new(),
        center_gradients: sites.iter().map(|s| field * (-s.charge)).collect(),
        charge_gradients: sites
            .iter()
            .map(|s| -(s.center - origin).dot(field))
            .collect(),
        dipole_gradients: vec![field * (-1.); sites.len()],
    })
}

fn pair_terms<S: Scalar>(
    a: &GaussianCharge,
    b: &GaussianCharge,
    ca: [S; 3],
    cb: [S; 3],
) -> (S, [S; 3], [S; 3]) {
    type A<T> = DualN<T, 3>;
    type B<T> = DualN<A<T>, 3>;
    let r = std::array::from_fn(|axis| {
        B::variable(A::constant(cb[axis]), axis) - B::constant(A::variable(ca[axis], axis))
    });
    let value = radial(r, a.exponent * b.exponent / (a.exponent + b.exponent));
    let mut energy = value.re.re.scale(a.charge * b.charge);
    let dpa = std::array::from_fn(|i| {
        let mut v = value.re.im[i].scale(b.charge);
        for j in 0..3 {
            v = v + value.im[j].im[i].scale(b.dipole.get(j));
        }
        v
    });
    let dpb = std::array::from_fn(|j| {
        let mut v = value.im[j].re.scale(a.charge);
        for i in 0..3 {
            v = v + value.im[j].im[i].scale(a.dipole.get(i));
        }
        v
    });
    for i in 0..3 {
        energy = energy
            + value.re.im[i].scale(a.dipole.get(i) * b.charge)
            + value.im[i].re.scale(b.dipole.get(i) * a.charge);
        for j in 0..3 {
            energy = energy + value.im[j].im[i].scale(a.dipole.get(i) * b.dipole.get(j));
        }
    }
    (energy, dpa, dpb)
}

pub struct ChargePairInteraction {
    pub energy: f64,
    pub first_center_gradient: Vec3,
    pub second_center_gradient: Vec3,
    pub first_dipole_gradient: Vec3,
    pub second_dipole_gradient: Vec3,
}

/// One distribution pair. No self energy or half factor is added implicitly.
pub fn charge_pair(a: &GaussianCharge, b: &GaussianCharge) -> Result<ChargePairInteraction> {
    a.validate()?;
    b.validate()?;
    type D = DualN<f64, 6>;
    let ca = std::array::from_fn(|axis| D::variable(a.center.get(axis), axis));
    let cb = std::array::from_fn(|axis| D::variable(b.center.get(axis), 3 + axis));
    let (value, dpa, dpb) = pair_terms(a, b, ca, cb);
    Ok(ChargePairInteraction {
        energy: value.re,
        first_center_gradient: Vec3::new(value.im[0], value.im[1], value.im[2]),
        second_center_gradient: Vec3::new(value.im[3], value.im[4], value.im[5]),
        first_dipole_gradient: Vec3::new(dpa[0].re, dpa[1].re, dpa[2].re),
        second_dipole_gradient: Vec3::new(dpb[0].re, dpb[1].re, dpb[2].re),
    })
}

/// Four normalized shells, with two s and two p radial functions: eight
/// projector functions in order s0,s1,p0(m=0,+1,-1),p1(m=0,+1,-1).
#[derive(Clone, Debug)]
pub struct ProjectorTemplate {
    pub s_exponents: [f64; 2],
    pub p_exponents: [f64; 2],
    /// Local Cartesian axes as columns. This is an orientation, not an AO map.
    pub orientation: Mat3,
}

impl ProjectorTemplate {
    pub const COUNT: usize = 8;
    pub fn validate(&self) -> Result<()> {
        if self
            .s_exponents
            .iter()
            .chain(&self.p_exponents)
            .any(|x| !x.is_finite() || *x <= 0.)
            || self.orientation.col.iter().any(|&c| !finite(c))
        {
            return invalid("projector widths and orientation must be finite");
        }
        for i in 0..3 {
            for j in 0..3 {
                if (self.orientation.col[i].dot(self.orientation.col[j])
                    - if i == j { 1. } else { 0. })
                .abs()
                    > 1e-10
                {
                    return invalid("projector orientation must be orthonormal");
                }
            }
        }
        if (self.orientation.col[0]
            .cross(self.orientation.col[1])
            .dot(self.orientation.col[2])
            - 1.)
            .abs()
            > 1e-10
        {
            return invalid("projector orientation must be right handed");
        }
        Ok(())
    }
    fn shells<S: Scalar>(&self, center: [S; 3]) -> Result<Vec<AnalyticShell<S>>> {
        [
            (0, self.s_exponents[0]),
            (0, self.s_exponents[1]),
            (1, self.p_exponents[0]),
            (1, self.p_exponents[1]),
        ]
        .into_iter()
        .map(|(l, a)| {
            AnalyticShell::normalized(l, center, vec![S::from_f64(a)], vec![S::from_f64(1.)])
        })
        .collect()
    }
}

pub struct ProjectorOverlaps {
    pub overlap: Matrix,
    pub atom_derivatives: Vec<[Matrix; 3]>,
    pub center_derivatives: [Matrix; 3],
    /// Ambient dB/dU_(Cartesian row,local axis), row-major nine entries.
    /// Contract with the existing frame Jacobian for physical rotations.
    pub orientation_derivatives: [Matrix; 9],
    pub projector_gram: Matrix,
}

fn rotate_projector_columns(matrix: &mut Matrix, orientation: Mat3) {
    let t = aos::cart_to_sph(1);
    for row in 0..matrix.rows {
        for start in [2, 5] {
            let old = [
                matrix[(row, start)],
                matrix[(row, start + 1)],
                matrix[(row, start + 2)],
            ];
            for m in 0..3 {
                matrix[(row, start + m)] = (0..3)
                    .map(|n| {
                        old[n]
                            * (0..3)
                                .map(|i| {
                                    (0..3)
                                        .map(|j| t[(i, n)] * orientation.col[j].get(i) * t[(j, m)])
                                        .sum::<f64>()
                                })
                                .sum::<f64>()
                    })
                    .sum();
            }
        }
    }
}

pub fn projector_overlaps(
    basis: &MolBasis,
    atoms: usize,
    center: Vec3,
    template: &ProjectorTemplate,
) -> Result<ProjectorOverlaps> {
    template.validate()?;
    if !finite(center) || basis.shells.iter().any(|s| s.atom >= atoms) {
        return invalid("invalid projector center or AO atom index");
    }
    let n = basis.nao_sph;
    let matrix = || Matrix::zeros(n, ProjectorTemplate::COUNT);
    let mut result = ProjectorOverlaps {
        overlap: matrix(),
        atom_derivatives: (0..atoms)
            .map(|_| std::array::from_fn(|_| matrix()))
            .collect(),
        center_derivatives: std::array::from_fn(|_| matrix()),
        orientation_derivatives: std::array::from_fn(|_| matrix()),
        projector_gram: Matrix::zeros(ProjectorTemplate::COUNT, ProjectorTemplate::COUNT),
    };
    type D = DualN<f64, 6>;
    let beta = template.shells(std::array::from_fn(|axis| {
        D::variable(center.get(axis), 3 + axis)
    }))?;
    for shell in &basis.shells {
        let ao = AnalyticShell::from_placed(
            shell,
            std::array::from_fn(|axis| D::variable(shell.center.get(axis), axis)),
        )?;
        for (beta_index, projector) in beta.iter().enumerate() {
            let start = [0, 1, 2, 5][beta_index];
            let block = analytic_basis::overlap(&ao, projector)?;
            for i in 0..block.shape[0] {
                for j in 0..block.shape[1] {
                    let value = block.values[i * block.shape[1] + j];
                    let row = shell.sph_off + i;
                    let col = start + j;
                    result.overlap[(row, col)] = value.re;
                    for axis in 0..3 {
                        result.atom_derivatives[shell.atom][axis][(row, col)] = value.im[axis];
                        result.center_derivatives[axis][(row, col)] = value.im[3 + axis];
                    }
                }
            }
        }
    }
    let t = aos::cart_to_sph(1);
    for i in 0..3 {
        for j in 0..3 {
            for row in 0..n {
                for start in [2, 5] {
                    for m in 0..3 {
                        result.orientation_derivatives[3 * i + j][(row, start + m)] = (0..3)
                            .map(|k| result.overlap[(row, start + k)] * t[(i, k)] * t[(j, m)])
                            .sum();
                    }
                }
            }
        }
    }
    rotate_projector_columns(&mut result.overlap, template.orientation);
    for derivative in result
        .atom_derivatives
        .iter_mut()
        .flatten()
        .chain(result.center_derivatives.iter_mut())
    {
        rotate_projector_columns(derivative, template.orientation);
    }
    let plain = template.shells(coordinates::<f64>(center))?;
    for a in 0..4 {
        for b in 0..4 {
            let block = analytic_basis::overlap(&plain[a], &plain[b])?;
            for i in 0..block.shape[0] {
                for j in 0..block.shape[1] {
                    result.projector_gram[([0, 1, 2, 5][a] + i, [0, 1, 2, 5][b] + j)] =
                        block.values[i * block.shape[1] + j];
                }
            }
        }
    }
    Ok(result)
}

pub struct NonlocalOperator {
    pub operator: Matrix,
    pub atom_derivatives: Vec<[Matrix; 3]>,
    pub center_derivatives: [Matrix; 3],
    pub orientation_derivatives: [Matrix; 9],
}

fn derivative_operator(b: &Matrix, d: &Matrix, db: &Matrix) -> Matrix {
    let mut result = db.matmul(d).matmul(&b.transpose());
    let other = b.matmul(d).matmul(&db.transpose());
    for (v, &w) in result.as_mut_slice().iter_mut().zip(other.as_slice()) {
        *v += w;
    }
    result
}

impl ProjectorOverlaps {
    /// All 36 independent coefficient operators; off-diagonal coefficients
    /// multiply both symmetric entries, not half of their sum.
    pub fn coefficient_operators(&self) -> Vec<Matrix> {
        let n = self.overlap.rows;
        (0..ProjectorTemplate::COUNT)
            .flat_map(|u| {
                (u..ProjectorTemplate::COUNT).map(move |v| {
                    let mut matrix = Matrix::zeros(n, n);
                    for i in 0..n {
                        for j in 0..n {
                            matrix[(i, j)] = self.overlap[(i, u)] * self.overlap[(j, v)]
                                + if u == v {
                                    0.
                                } else {
                                    self.overlap[(i, v)] * self.overlap[(j, u)]
                                };
                        }
                    }
                    matrix
                })
            })
            .collect()
    }

    pub fn operator(&self, d: &Matrix) -> Result<NonlocalOperator> {
        if d.rows != ProjectorTemplate::COUNT
            || d.cols != ProjectorTemplate::COUNT
            || d.as_slice().iter().any(|x| !x.is_finite())
        {
            return invalid("nonlocal coefficients require a finite eight-by-eight matrix");
        }
        for i in 0..d.rows {
            for j in 0..i {
                if d[(i, j)] != d[(j, i)] {
                    return invalid("nonlocal coefficient matrix must be exactly symmetric; no PSD constraint is imposed");
                }
            }
        }
        Ok(NonlocalOperator {
            operator: self.overlap.matmul(d).matmul(&self.overlap.transpose()),
            atom_derivatives: self
                .atom_derivatives
                .iter()
                .map(|g| std::array::from_fn(|a| derivative_operator(&self.overlap, d, &g[a])))
                .collect(),
            center_derivatives: std::array::from_fn(|a| {
                derivative_operator(&self.overlap, d, &self.center_derivatives[a])
            }),
            orientation_derivatives: std::array::from_fn(|a| {
                derivative_operator(&self.overlap, d, &self.orientation_derivatives[a])
            }),
        })
    }
}

pub fn symmetric_projector_coefficients(coefficients: &[f64]) -> Result<Matrix> {
    if coefficients.len() != 36 || coefficients.iter().any(|x| !x.is_finite()) {
        return invalid("nonlocal dictionary requires exactly36 finite independent coefficients");
    }
    let mut matrix = Matrix::zeros(8, 8);
    let mut k = 0;
    for i in 0..8 {
        for j in i..8 {
            matrix[(i, j)] = coefficients[k];
            matrix[(j, i)] = coefficients[k];
            k += 1;
        }
    }
    Ok(matrix)
}
