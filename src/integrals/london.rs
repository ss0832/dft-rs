// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic-field Taylor coefficients of integrals over London orbitals.
//!
//! A London orbital carries the phase of the gauge transformation that moves
//! the origin of the vector potential `A_O = B x (r - O) / 2` to its own
//! centre, `w_m = exp(-i (B x (R_m - O)) . r / 2) chi_m`. For the field
//! `B = t b` a product `conj(w_m) w_n` carries `exp(i t k . r)` with
//! `k = b x (R_m - R_n) / 2`, which does not contain `O`, and the
//! one-electron Hamiltonian acts on `w_n` as
//!
//! ```text
//! exp(-i A_n . r) [ p^2 / 2 + V + t b . L_n / 2 + t^2 (b x r_n)^2 / 8 ] chi_n,
//! ```
//!
//! with `r_n = r - R_n` and `L_n = -i r_n x grad`. Every Taylor coefficient
//! in `t` is therefore a finite sum of integrals of polynomials times
//! Gaussians. They are evaluated by rewriting "polynomial or derivative of a
//! primitive" as a combination of primitives on the same centre. With
//! `london = false` every function keeps the common gauge origin instead and
//! only the operator depends on the field.

use super::analytic::{self, GaussianKernel, Primitive};
use crate::basis::aos::cart_to_sph;
use crate::basis::{cart_components, n_sph, MolBasis, MolShell};
use crate::dual::Scalar;
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::Result;
use rayon::prelude::*;

#[path = "london_phase.rs"]
mod phase;

/// A complex matrix as its real and imaginary parts.
#[derive(Clone, Debug)]
pub struct ComplexMatrix {
    pub re: Matrix,
    pub im: Matrix,
}

impl ComplexMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            re: Matrix::zeros(rows, cols),
            im: Matrix::zeros(rows, cols),
        }
    }

    /// Common complex matrix product for London references and their paths.
    pub(crate) fn matmul_seq(&self, rhs: &Self) -> Self {
        let mut re = self.re.matmul_seq(&rhs.re);
        for (value, other) in re
            .as_mut_slice()
            .iter_mut()
            .zip(self.im.matmul_seq(&rhs.im).as_slice())
        {
            *value -= other;
        }
        let mut im = self.re.matmul_seq(&rhs.im);
        for (value, other) in im
            .as_mut_slice()
            .iter_mut()
            .zip(self.im.matmul_seq(&rhs.re).as_slice())
        {
            *value += other;
        }
        Self { re, im }
    }

    /// Real trace of the product, without requiring either factor Hermitian.
    pub(crate) fn real_trace_product(&self, rhs: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.re.rows {
            for j in 0..self.re.cols {
                sum += self.re[(i, j)] * rhs.re[(j, i)] - self.im[(i, j)] * rhs.im[(j, i)];
            }
        }
        sum
    }
}

/// The path `E(t) = t electric`, `B(t) = t magnetic` of the integral series.
#[derive(Clone, Copy, Debug)]
pub struct FieldPath {
    pub electric: [f64; 3],
    pub magnetic: [f64; 3],
    /// Gauge origin, used by the common-origin operator only.
    pub origin: [f64; 3],
    /// Whether the basis functions carry London phases.
    pub london: bool,
    pub order: usize,
}

/// Scalar-generic finite-field controls for moving-centre integral jets.
/// Coefficients are summed through the same total field degree as FieldPath.
/// Exponents, centres, nuclei, fields and origin retain all scalar directions.
#[derive(Clone, Copy, Debug)]
pub struct ScalarFieldPath<S: Scalar> {
    pub electric: [S; 3],
    pub magnetic: [S; 3],
    pub origin: [S; 3],
    pub london: bool,
    pub phase_order: usize,
    /// Total nilpotent scalar order needed in the Boys function, including
    /// every nested/mixed direction. It is independent of phase truncation.
    pub derivative_order: usize,
}

fn scalar_half_cross<S: Scalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        (a[1] * b[2] - a[2] * b[1]).scale(0.5),
        (a[2] * b[0] - a[0] * b[2]).scale(0.5),
        (a[0] * b[1] - a[1] * b[0]).scale(0.5),
    ]
}

fn scalar_phase<S: Scalar>(
    a: &Primitive<S>,
    b: &Primitive<S>,
    field: &ScalarFieldPath<S>,
) -> [S; 3] {
    if field.london {
        scalar_half_cross(
            field.magnetic,
            std::array::from_fn(|i| a.center[i] - b.center[i]),
        )
    } else {
        [S::from_f64(0.0); 3]
    }
}

fn validate_scalar_field<S: Scalar>(field: &ScalarFieldPath<S>) -> Result<()> {
    if field.phase_order > 24
        || field
            .electric
            .iter()
            .chain(&field.magnetic)
            .chain(&field.origin)
            .any(|x| !x.value().is_finite())
    {
        return Err(crate::DftError::InvalidInput("scalar London integrals need finite field/origin values and total field degree zero through 24".into()));
    }
    Ok(())
}

/// Resummed primitive overlap, core and three position integrals, each as
/// (real, imaginary). Core includes kinetic, supplied point nuclei and uniform
/// electric/paramagnetic/diamagnetic terms. Arbitrary scalar nuclear and field
/// jets are retained. `exact_zero` may return true only when every coefficient
/// vanishes; checking the scalar's undifferentiated value is insufficient.
/// Phase truncation is explicit and must be converged for the finite field.
pub fn primitive_field_pair<S: Scalar>(
    bra: &Primitive<S>,
    ket: &Primitive<S>,
    nuclei: &[([S; 3], S)],
    field: &ScalarFieldPath<S>,
    exact_zero: fn(S) -> bool,
) -> Result<[(S, S); 5]> {
    validate_scalar_field(field)?;
    let k = scalar_phase(bra, ket, field);
    let terms = Terms::single_scalar(*ket, exact_zero);
    let gauge = if field.london {
        ket.center
    } else {
        field.origin
    };
    scalar_field_pair_terms(bra, &terms, k, gauge, nuclei, field, true)
}

fn scalar_field_pair_terms<S: Scalar>(
    bra: &Primitive<S>,
    terms: &Terms<S>,
    k: [S; 3],
    gauge: [S; 3],
    nuclei: &[([S; 3], S)],
    field: &ScalarFieldPath<S>,
    positions: bool,
) -> Result<[(S, S); 5]> {
    let order = field.phase_order;
    let plain = phased(terms, k, order);
    let kinetic = phased(&terms.kinetic(), k, order);
    let electric = phased(&terms.dotted(field.electric), k, order);
    let angular = phased(&terms.angular(field.magnetic, gauge), k, order);
    let diamagnetic = phased(&terms.diamagnetic(field.magnetic, gauge), k, order);
    let position: Vec<_> = (0..if positions { 3 } else { 0 })
        .map(|a| phased(&terms.shifted(a, S::from_f64(0.0)), k, order))
        .collect();
    let z = S::from_f64(0.0);
    let mut result = [(z, z); 5];
    for p in 0..=order {
        let (re, im) = phase_factor(p);
        let overlap = overlap_with(bra, &plain[p])?;
        result[0].0 = result[0].0 + overlap.scale(re);
        result[0].1 = result[0].1 + overlap.scale(im);
        let core = overlap_with(bra, &kinetic[p])?
            + attraction_with_order(bra, &plain[p], nuclei, field.derivative_order)?;
        result[1].0 = result[1].0 + core.scale(re);
        result[1].1 = result[1].1 + core.scale(im);
        if p < order {
            let e = overlap_with(bra, &electric[p])?;
            let l = overlap_with(bra, &angular[p])?;
            result[1].0 = result[1].0 + e.scale(re) + l.scale(0.5 * im);
            result[1].1 = result[1].1 + e.scale(im) - l.scale(0.5 * re);
        }
        if p + 2 <= order {
            let d = overlap_with(bra, &diamagnetic[p])?;
            result[1].0 = result[1].0 + d.scale(re);
            result[1].1 = result[1].1 + d.scale(im);
        }
        for axis in 0..position.len() {
            let d = overlap_with(bra, &position[axis][p])?;
            result[2 + axis].0 = result[2 + axis].0 + d.scale(re);
            result[2 + axis].1 = result[2 + axis].1 + d.scale(im);
        }
    }
    Ok(result)
}

/// Complex Coulomb/range-separated primitive quartet in a finite London field,
/// with full scalar centre/exponent/kernel/field jets. The two phase endpoints
/// are independent; no same-q or Hermitian projection is performed.
pub fn primitive_field_repulsion<S: Scalar>(
    primitives: [&Primitive<S>; 4],
    kernel: GaussianKernel<S>,
    field: &ScalarFieldPath<S>,
    exact_zero: fn(S) -> bool,
) -> Result<(S, S)> {
    validate_scalar_field(field)?;
    let [a, b, c, d] = primitives;
    let coefficients = primitive_repulsion_coefficients(
        primitives,
        kernel,
        scalar_phase(a, b, field),
        scalar_phase(c, d, field),
        0,
        field.phase_order,
        field.derivative_order,
        exact_zero,
    )?;
    let mut result = (S::from_f64(0.0), S::from_f64(0.0));
    for (re, im) in coefficients {
        result.0 = result.0 + re;
        result.1 = result.1 + im;
    }
    Ok(result)
}

fn primitive_repulsion_coefficients<S: Scalar>(
    primitives: [&Primitive<S>; 4],
    kernel: GaussianKernel<S>,
    k_bra: [S; 3],
    k_ket: [S; 3],
    first: usize,
    last: usize,
    derivative_order: usize,
    exact_zero: fn(S) -> bool,
) -> Result<Vec<(S, S)>> {
    let [a, b, c, d] = primitives;
    let second = phased(&Terms::single_scalar(*b, exact_zero), k_bra, last);
    let fourth = phased(&Terms::single_scalar(*d, exact_zero), k_ket, last);
    let engine = analytic::HermiteQuartet::new(
        primitives,
        kernel,
        derivative_order,
        [0, last, 0, last],
        last,
    )?;
    let bras: Vec<_> = second
        .iter()
        .map(|terms| engine.bra(terms.items.iter().map(|(w, t)| (*w, a.angular, t.angular))))
        .collect();
    let kets: Vec<_> = fourth
        .iter()
        .map(|terms| engine.ket(terms.items.iter().map(|(w, t)| (*w, c.angular, t.angular))))
        .collect();
    Ok(series_from_pairs(first, last, |p, q| {
        engine.contract(&bras[p], &kets[q])
    }))
}

/// Field coefficients `first..=last` from the pair values `(p, q)` of the
/// bra and ket phase powers: `i^n / (p! q!)` summed over `p + q = n`.
fn series_from_pairs<S: Scalar>(
    first: usize,
    last: usize,
    mut pair: impl FnMut(usize, usize) -> S,
) -> Vec<(S, S)> {
    let mut result = vec![(S::from_f64(0.0), S::from_f64(0.0)); last + 1 - first];
    for n in first..=last {
        let (re, im) = phase_factor(n);
        let factorial = (1..=n).fold(1.0, |a, b| a * b as f64);
        for p in 0..=n {
            let q = n - p;
            let split = (1..=p).chain(1..=q).fold(1.0, |a, b| a * b as f64);
            let value = pair(p, q);
            result[n - first].0 = result[n - first].0 + value.scale(re * factorial / split);
            result[n - first].1 = result[n - first].1 + value.scale(im * factorial / split);
        }
    }
    result
}

/// A linear combination of Cartesian Gaussian primitives that share one
/// exponent and one centre.
#[derive(Clone)]
pub(crate) struct Terms<S: crate::dual::Scalar = f64> {
    pub(crate) items: Vec<(S, Primitive<S>)>,
    exact_zero: fn(S) -> bool,
}

impl Terms<f64> {
    pub(crate) fn empty() -> Self {
        Self::empty_scalar(|x| x == 0.0)
    }
    pub(crate) fn single(p: Primitive<f64>) -> Self {
        Self::single_scalar(p, |x| x == 0.0)
    }
}

impl<S: crate::dual::Scalar> Terms<S> {
    fn empty_scalar(exact_zero: fn(S) -> bool) -> Self {
        Self {
            items: Vec::new(),
            exact_zero,
        }
    }
    fn single_scalar(p: Primitive<S>, exact_zero: fn(S) -> bool) -> Self {
        Self {
            items: vec![(S::from_f64(1.0), p)],
            exact_zero,
        }
    }
    fn with_items(&self, items: Vec<(S, Primitive<S>)>) -> Self {
        Self {
            items,
            exact_zero: self.exact_zero,
        }
    }
    fn local(&self, axis: usize) -> Self {
        self.with_items(
            self.items
                .iter()
                .map(|(c, p)| {
                    let mut q = *p;
                    q.angular[axis] += 1;
                    (*c, q)
                })
                .collect(),
        )
    }
    pub(crate) fn shifted(&self, axis: usize, shift: S) -> Self {
        let mut items = self.local(axis).items;
        for (c, p) in &self.items {
            let offset = p.center[axis] - shift;
            if !(self.exact_zero)(offset) {
                items.push((*c * offset, *p));
            }
        }
        self.with_items(items).merged()
    }
    pub(crate) fn derivative(&self, axis: usize) -> Self {
        let mut items = Vec::with_capacity(2 * self.items.len());
        for (c, p) in &self.items {
            let mut up = *p;
            up.angular[axis] += 1;
            items.push((p.exponent.scale(-2.0) * *c, up));
            if p.angular[axis] > 0 {
                let mut down = *p;
                down.angular[axis] -= 1;
                items.push((c.scale(p.angular[axis] as f64), down));
            }
        }
        self.with_items(items).merged()
    }
    pub(crate) fn scaled(&self, scale: f64) -> Self {
        self.with_items(
            self.items
                .iter()
                .map(|(c, p)| (c.scale(scale), *p))
                .collect(),
        )
    }
    fn multiplied(&self, scale: S) -> Self {
        self.with_items(self.items.iter().map(|(c, p)| (*c * scale, *p)).collect())
    }
    pub(crate) fn plus(mut self, other: &Self) -> Self {
        self.items.extend(other.items.iter().copied());
        self.merged()
    }
    fn merged(self) -> Self {
        let mut items: Vec<(S, Primitive<S>)> = Vec::with_capacity(self.items.len());
        for (c, p) in self.items {
            if let Some(found) = items.iter_mut().find(|(_, q)| q.angular == p.angular) {
                found.0 = found.0 + c;
            } else {
                items.push((c, p));
            }
        }
        items.retain(|(c, _)| !(self.exact_zero)(*c));
        Self {
            items,
            exact_zero: self.exact_zero,
        }
    }
    fn dotted(&self, k: [S; 3]) -> Self {
        let mut out = Self::empty_scalar(self.exact_zero);
        for axis in 0..3 {
            if !(self.exact_zero)(k[axis]) {
                out = out.plus(&self.shifted(axis, S::from_f64(0.0)).multiplied(k[axis]));
            }
        }
        out
    }
    fn kinetic(&self) -> Self {
        let mut out = Self::empty_scalar(self.exact_zero);
        for axis in 0..3 {
            out = out.plus(&self.derivative(axis).derivative(axis).scaled(-0.5));
        }
        out
    }
    fn angular(&self, b: [S; 3], centre: [S; 3]) -> Self {
        let mut out = Self::empty_scalar(self.exact_zero);
        for c in 0..3 {
            if (self.exact_zero)(b[c]) {
                continue;
            }
            let (j, k) = ((c + 1) % 3, (c + 2) % 3);
            out = out.plus(&self.derivative(k).shifted(j, centre[j]).multiplied(b[c]));
            out = out.plus(&self.derivative(j).shifted(k, centre[k]).multiplied(-b[c]));
        }
        out
    }
    fn diamagnetic(&self, b: [S; 3], centre: [S; 3]) -> Self {
        let square = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        let mut out = Self::empty_scalar(self.exact_zero);
        for c in 0..3 {
            for d in 0..3 {
                let factor =
                    ((if c == d { square } else { S::from_f64(0.0) }) - b[c] * b[d]).scale(0.125);
                if !(self.exact_zero)(factor) {
                    out = out.plus(
                        &self
                            .shifted(c, centre[c])
                            .shifted(d, centre[d])
                            .multiplied(factor),
                    );
                }
            }
        }
        out
    }
}

/// `i^k / k!` as real and imaginary parts.
fn phase_factor(k: usize) -> (f64, f64) {
    let factorial = (1..=k).fold(1.0, |a, b| a * b as f64);
    match k % 4 {
        0 => (1.0 / factorial, 0.0),
        1 => (0.0, 1.0 / factorial),
        2 => (-1.0 / factorial, 0.0),
        _ => (0.0, -1.0 / factorial),
    }
}

fn overlap_with<S: Scalar>(bra: &Primitive<S>, ket: &Terms<S>) -> Result<S> {
    let mut value = S::from_f64(0.0);
    for (c, p) in &ket.items {
        value = value + *c * analytic::overlap(bra, p)?;
    }
    Ok(value)
}
fn attraction_with<S: Scalar>(
    bra: &Primitive<S>,
    ket: &Terms<S>,
    nuclei: &[([S; 3], S)],
) -> Result<S> {
    attraction_with_order(bra, ket, nuclei, 0)
}

fn attraction_with_order<S: Scalar>(
    bra: &Primitive<S>,
    ket: &Terms<S>,
    nuclei: &[([S; 3], S)],
    derivative_order: usize,
) -> Result<S> {
    let mut value = S::from_f64(0.0);
    for (c, p) in &ket.items {
        for (centre, charge) in nuclei {
            value = value
                + *c * analytic::attraction(
                    bra,
                    p,
                    *centre,
                    *charge,
                    GaussianKernel::Coulomb,
                    derivative_order,
                )?;
        }
    }
    Ok(value)
}

/// `(b x v) / 2`.
fn half_cross(b: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (b[1] * v[2] - b[2] * v[1]),
        0.5 * (b[2] * v[0] - b[0] * v[2]),
        0.5 * (b[0] * v[1] - b[1] * v[0]),
    ]
}

pub(crate) fn primitive(
    shell: &MolShell,
    component: (i32, i32, i32),
    index: usize,
) -> Primitive<f64> {
    Primitive {
        angular: [
            component.0 as usize,
            component.1 as usize,
            component.2 as usize,
        ],
        exponent: shell.exps[index],
        center: shell.center.to_array(),
    }
}

/// Successive powers `(k . r)^p ket` for `p = 0..=order`.
fn phased<S: Scalar>(ket: &Terms<S>, k: [S; 3], order: usize) -> Vec<Terms<S>> {
    let mut out = vec![ket.clone()];
    for p in 1..=order {
        let next = out[p - 1].dotted(k);
        out.push(next);
    }
    out
}

/// Transform one index of a row-major tensor from Cartesian to spherical
/// components of a shell of angular momentum `l`.
fn transform_axis(values: &[f64], shape: &mut [usize], axis: usize, l: u8) -> Vec<f64> {
    let transform = cart_to_sph(l);
    let old = shape[axis];
    let stride: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let new = transform.cols;
    let mut out = vec![0.0; outer * new * stride];
    for prefix in 0..outer {
        for source in 0..old {
            for target in 0..new {
                let weight = transform[(source, target)];
                if weight == 0.0 {
                    continue;
                }
                for tail in 0..stride {
                    out[(prefix * new + target) * stride + tail] +=
                        weight * values[(prefix * old + source) * stride + tail];
                }
            }
        }
    }
    shape[axis] = new;
    out
}

/// Taylor coefficients of the one-electron matrices along a field path.
#[derive(Clone, Debug)]
pub struct OneElectronSeries {
    /// Overlap coefficients; only the zeroth one is nonzero without London
    /// phases.
    pub overlap: Vec<ComplexMatrix>,
    /// Kinetic energy, nuclear attraction and the field terms.
    pub core: Vec<ComplexMatrix>,
    /// Position operator `r_c`, for the induced dipole.
    pub dipole: [Vec<ComplexMatrix>; 3],
}

/// One-electron series of `mb` for the nuclei of `mol`. Matrix element
/// `(m, n)` has the bra function `m` complex conjugated.
pub fn one_electron_series(
    mol: &Molecule,
    mb: &MolBasis,
    path: &FieldPath,
) -> Result<OneElectronSeries> {
    let nao = mb.nao_sph;
    let width = path.order + 1;
    let charges = mb.effective_charges(mol);
    let nuclei: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .zip(&charges)
        .map(|(atom, &z)| (atom.position.to_array(), z))
        .collect();
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    // Per ordered shell pair: [quantity][order] -> (re, im) spherical blocks,
    // quantities overlap, core, dipole x, y, z.
    let blocks: Result<Vec<Vec<Vec<(Vec<f64>, Vec<f64>)>>>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let sa = &mb.shells[a];
            let sb = &mb.shells[b];
            let ca = cart_components(sa.l);
            let cb = cart_components(sb.l);
            let size = ca.len() * cb.len();
            let mut cart = vec![vec![(vec![0.0; size], vec![0.0; size]); width]; 5];
            let centre_b = sb.center.to_array();
            let separation = [
                sa.center.x - sb.center.x,
                sa.center.y - sb.center.y,
                sa.center.z - sb.center.z,
            ];
            let k = if path.london {
                half_cross(path.magnetic, separation)
            } else {
                [0.0; 3]
            };
            // Origin of the vector potential seen by the ket function.
            let gauge = if path.london { centre_b } else { path.origin };
            for (ia, &component_a) in ca.iter().enumerate() {
                for (ib, &component_b) in cb.iter().enumerate() {
                    let slot = ia * cb.len() + ib;
                    for pa in 0..sa.exps.len() {
                        let bra = primitive(sa, component_a, pa);
                        for pb in 0..sb.exps.len() {
                            let weight = sa.coefs[pa] * sb.coefs[pb];
                            let primitive_b = primitive(sb, component_b, pb);
                            if let Some(values) = phase::pair_coefficients(
                                &bra,
                                &primitive_b,
                                k,
                                &nuclei,
                                path.electric,
                                path.magnetic,
                                gauge,
                                path.order,
                            )? {
                                for q in 0..5 {
                                    for n in 0..width {
                                        cart[q][n].0[slot] += weight * values[q][n].0;
                                        cart[q][n].1[slot] += weight * values[q][n].1;
                                    }
                                }
                                continue;
                            }
                            let ket = Terms::single(primitive_b);
                            let plain = phased(&ket, k, path.order);
                            let kinetic = phased(&ket.kinetic(), k, path.order);
                            let angular = phased(&ket.angular(path.magnetic, gauge), k, path.order);
                            let field = phased(&ket.dotted(path.electric), k, path.order);
                            let diamagnetic =
                                phased(&ket.diamagnetic(path.magnetic, gauge), k, path.order);
                            let position: Vec<Vec<Terms>> = (0..3)
                                .map(|axis| phased(&ket.shifted(axis, 0.0), k, path.order))
                                .collect();
                            for n in 0..width {
                                let (re, im) = phase_factor(n);
                                let s = weight * overlap_with(&bra, &plain[n])?;
                                cart[0][n].0[slot] += re * s;
                                cart[0][n].1[slot] += im * s;
                                for axis in 0..3 {
                                    let d = weight * overlap_with(&bra, &position[axis][n])?;
                                    cart[2 + axis][n].0[slot] += re * d;
                                    cart[2 + axis][n].1[slot] += im * d;
                                }
                                for p in 0..=n {
                                    let (re, im) = phase_factor(p);
                                    match n - p {
                                        0 => {
                                            let h = weight
                                                * (overlap_with(&bra, &kinetic[p])?
                                                    + attraction_with(&bra, &plain[p], &nuclei)?);
                                            cart[1][n].0[slot] += re * h;
                                            cart[1][n].1[slot] += im * h;
                                        }
                                        1 => {
                                            // b . L / 2 = -(i / 2) b . (r x grad).
                                            let l = weight * overlap_with(&bra, &angular[p])?;
                                            cart[1][n].0[slot] += 0.5 * im * l;
                                            cart[1][n].1[slot] -= 0.5 * re * l;
                                            let e = weight * overlap_with(&bra, &field[p])?;
                                            cart[1][n].0[slot] += re * e;
                                            cart[1][n].1[slot] += im * e;
                                        }
                                        2 => {
                                            let d = weight * overlap_with(&bra, &diamagnetic[p])?;
                                            cart[1][n].0[slot] += re * d;
                                            cart[1][n].1[slot] += im * d;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let spherical = |values: &[f64]| -> Vec<f64> {
                let mut shape = [ca.len(), cb.len()];
                let first = transform_axis(values, &mut shape, 0, sa.l);
                transform_axis(&first, &mut shape, 1, sb.l)
            };
            Ok(cart
                .iter()
                .map(|orders| {
                    orders
                        .iter()
                        .map(|(re, im)| (spherical(re), spherical(im)))
                        .collect()
                })
                .collect())
        })
        .collect();
    let empty = || vec![ComplexMatrix::zeros(nao, nao); width];
    let mut overlap = empty();
    let mut core = empty();
    let mut dipole = [empty(), empty(), empty()];
    for (&(a, b), block) in pairs.iter().zip(blocks?) {
        let sa = &mb.shells[a];
        let sb = &mb.shells[b];
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for n in 0..width {
            for i in 0..na {
                for j in 0..nb {
                    let at = (sa.sph_off + i, sb.sph_off + j);
                    let slot = i * nb + j;
                    overlap[n].re[at] = block[0][n].0[slot];
                    overlap[n].im[at] = block[0][n].1[slot];
                    core[n].re[at] = block[1][n].0[slot];
                    core[n].im[at] = block[1][n].1[slot];
                    for axis in 0..3 {
                        dipole[axis][n].re[at] = block[2 + axis][n].0[slot];
                        dipole[axis][n].im[at] = block[2 + axis][n].1[slot];
                    }
                }
            }
        }
    }
    Ok(OneElectronSeries {
        overlap,
        core,
        dipole,
    })
}

/// Taylor coefficients of the electron repulsion integrals
/// `(m n | l s) = int conj(w_m) w_n (1 / r12) conj(w_l) w_s`, per order as
/// real and imaginary tensors `[((m nao + n) nao + l) nao + s]`. Without
/// London phases only the zeroth coefficient is nonzero. `omega` selects the
/// long-range kernel `erf(omega r12) / r12`.
pub fn repulsion_series(
    mb: &MolBasis,
    path: &FieldPath,
    omega: Option<f64>,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
    let nao = mb.nao_sph;
    let width = if path.london { path.order + 1 } else { 1 };
    let ns = mb.shells.len();
    let kernel = omega.map_or(GaussianKernel::Coulomb, |omega| GaussianKernel::LongRange {
        omega,
    });
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let pair_phase = |a: usize, b: usize| -> [f64; 3] {
        if !path.london {
            return [0.0; 3];
        }
        let (sa, sb) = (&mb.shells[a], &mb.shells[b]);
        half_cross(
            path.magnetic,
            [
                sa.center.x - sb.center.x,
                sa.center.y - sb.center.y,
                sa.center.z - sb.center.z,
            ],
        )
    };
    // Per ordered bra shell pair: the blocks of every ordered ket shell pair.
    type Block = (usize, usize, Vec<(Vec<f64>, Vec<f64>)>);
    let computed: Result<Vec<Vec<Block>>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let k_bra = pair_phase(a, b);
            let mut out = Vec::with_capacity(pairs.len());
            for &(c, d) in &pairs {
                out.push((
                    c,
                    d,
                    quartet_series(
                        mb,
                        [a, b, c, d],
                        k_bra,
                        pair_phase(c, d),
                        kernel,
                        0,
                        width - 1,
                    )?,
                ));
            }
            Ok(out)
        })
        .collect();
    let size = nao * nao * nao * nao;
    let mut tensor = vec![(vec![0.0; size], vec![0.0; size]); width];
    for (&(a, b), blocks) in pairs.iter().zip(computed?) {
        let (sa, sb) = (&mb.shells[a], &mb.shells[b]);
        for (c, d, block) in blocks {
            let (sc, sd) = (&mb.shells[c], &mb.shells[d]);
            let shape = [n_sph(sa.l), n_sph(sb.l), n_sph(sc.l), n_sph(sd.l)];
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        for l in 0..shape[3] {
                            let slot = ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
                            let index =
                                (((sa.sph_off + i) * nao + sb.sph_off + j) * nao + sc.sph_off + k)
                                    * nao
                                    + sd.sph_off
                                    + l;
                            for n in 0..width {
                                tensor[n].0[index] = block[n].0[slot];
                                tensor[n].1[index] = block[n].1[slot];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(tensor)
}

fn times(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Spherical blocks of the Taylor coefficients `first..=last` of one shell
/// quartet `(a b | c d)`, per order as real and imaginary arrays
/// `[((i nb + j) nc + k) nd + l]`.
#[allow(clippy::too_many_arguments)]
fn quartet_series(
    mb: &MolBasis,
    shells: [usize; 4],
    k_bra: [f64; 3],
    k_ket: [f64; 3],
    kernel: GaussianKernel<f64>,
    first: usize,
    last: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
    let [sa, sb, sc, sd] = shells.map(|s| &mb.shells[s]);
    let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
    let (cc, cd) = (cart_components(sc.l), cart_components(sd.l));
    let size = ca.len() * cb.len() * cc.len() * cd.len();
    let mut cart = vec![(vec![0.0; size], vec![0.0; size]); last + 1 - first];
    for (ia, &xa) in ca.iter().enumerate() {
        for (ib, &xb) in cb.iter().enumerate() {
            for (ic, &xc) in cc.iter().enumerate() {
                for (id, &xd) in cd.iter().enumerate() {
                    let slot = ((ia * cb.len() + ib) * cc.len() + ic) * cd.len() + id;
                    for pa in 0..sa.exps.len() {
                        let bra = primitive(sa, xa, pa);
                        for pb in 0..sb.exps.len() {
                            let second = primitive(sb, xb, pb);
                            for pc in 0..sc.exps.len() {
                                let third = primitive(sc, xc, pc);
                                for pd in 0..sd.exps.len() {
                                    let fourth = primitive(sd, xd, pd);
                                    let weight =
                                        sa.coefs[pa] * sb.coefs[pb] * sc.coefs[pc] * sd.coefs[pd];
                                    let values = phase::repulsion_coefficients(
                                        [&bra, &second, &third, &fourth],
                                        kernel,
                                        k_bra,
                                        k_ket,
                                        first,
                                        last,
                                    )?;
                                    for (n, (re, im)) in values.into_iter().enumerate() {
                                        cart[n].0[slot] += weight * re;
                                        cart[n].1[slot] += weight * im;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let spherical = |values: &[f64]| -> Vec<f64> {
        let mut shape = [ca.len(), cb.len(), cc.len(), cd.len()];
        let mut current = values.to_vec();
        for (axis, shell) in [sa, sb, sc, sd].into_iter().enumerate() {
            current = transform_axis(&current, &mut shape, axis, shell.l);
        }
        current
    };
    Ok(cart
        .iter()
        .map(|(re, im)| (spherical(re), spherical(im)))
        .collect())
}

/// Shell quartets of the London repulsion series: the field-free Schwarz
/// bounds that screen them and the phase slope of every shell pair.
struct SeriesQuartets<'a> {
    mb: &'a MolBasis,
    magnetic: [f64; 3],
    kernel: GaussianKernel<f64>,
    bound: Matrix,
    pairs: Vec<(usize, usize)>,
}

impl<'a> SeriesQuartets<'a> {
    fn new(mb: &'a MolBasis, path: &FieldPath, omega: Option<f64>) -> Self {
        let kernel = omega.map_or(GaussianKernel::Coulomb, |omega| GaussianKernel::LongRange {
            omega,
        });
        let bound = crate::integrals::schwarz_matrix(
            mb,
            match omega {
                None => crate::integrals::CoulombKernel::Coulomb,
                Some(omega) => crate::integrals::CoulombKernel::Erf { omega },
            },
        );
        let ns = mb.shells.len();
        let pairs = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
        Self {
            mb,
            magnetic: path.magnetic,
            kernel,
            bound,
            pairs,
        }
    }

    fn pair_phase(&self, a: usize, b: usize) -> [f64; 3] {
        let (sa, sb) = (&self.mb.shells[a], &self.mb.shells[b]);
        half_cross(
            self.magnetic,
            [
                sa.center.x - sb.center.x,
                sa.center.y - sb.center.y,
                sa.center.z - sb.center.z,
            ],
        )
    }

    /// Field coefficients `1..=n` of the quartet, or `None` when the
    /// Schwarz bound screens it.
    fn blocks(&self, shells: [usize; 4], n: usize) -> Result<Option<Vec<(Vec<f64>, Vec<f64>)>>> {
        self.blocks_from(shells, 1, n)
    }

    fn blocks_from(
        &self,
        [a, b, c, d]: [usize; 4],
        first: usize,
        n: usize,
    ) -> Result<Option<Vec<(Vec<f64>, Vec<f64>)>>> {
        if first != 0 && self.bound[(a, b)] * self.bound[(c, d)] < 1e-13 {
            return Ok(None);
        }
        quartet_series(
            self.mb,
            [a, b, c, d],
            self.pair_phase(a, b),
            self.pair_phase(c, d),
            self.kernel,
            first,
            n,
        )
        .map(Some)
    }
}

/// Complete complex Coulomb/exchange coefficient with retained density and
/// contraction remainders. Shell blocks are released after contraction; no
/// four-index tensor or displaced-field calculation is introduced.
#[allow(clippy::too_many_arguments)]
pub(crate) fn retained_repulsion_fock(
    mb: &MolBasis,
    path: &FieldPath,
    omega: Option<f64>,
    high: &[Vec<ComplexMatrix>; 2],
    low: &[Vec<ComplexMatrix>; 2],
    order: usize,
    coulomb: f64,
    exchange: f64,
) -> Result<([ComplexMatrix; 2], [ComplexMatrix; 2])> {
    use crate::linalg::AccurateSum;
    let nao = mb.nao_sph;
    let mut sums: [Vec<[AccurateSum; 2]>; 2] =
        std::array::from_fn(|_| vec![[AccurateSum::default(); 2]; nao * nao]);
    let mut integral_path = *path;
    if !path.london {
        integral_path.magnetic = [0.0; 3];
    }
    let quartets = SeriesQuartets::new(mb, &integral_path, omega);
    let add = |sum: &mut [AccurateSum; 2], g: (f64, f64), d: (f64, f64), scale: f64| {
        let re = g.0 * scale;
        let im = g.1 * scale;
        // Preserve both coefficient multiplication and complex contraction.
        for (re, im) in [(re, im), (g.0.mul_add(scale, -re), g.1.mul_add(scale, -im))] {
            sum[0].add_product(re, d.0);
            sum[0].add_product(-im, d.1);
            sum[1].add_product(re, d.1);
            sum[1].add_product(im, d.0);
        }
    };
    for &(a, b) in &quartets.pairs {
        for &(c, d) in &quartets.pairs {
            let last = if path.london { order } else { 0 };
            let Some(blocks) = quartets.blocks_from([a, b, c, d], 0, last)? else {
                continue;
            };
            let shells = [a, b, c, d].map(|s| &mb.shells[s]);
            let shape = shells.map(|s| n_sph(s.l));
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        for l in 0..shape[3] {
                            let slot = ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
                            let [mu, nu, lam, sig] = [
                                shells[0].sph_off + i,
                                shells[1].sph_off + j,
                                shells[2].sph_off + k,
                                shells[3].sph_off + l,
                            ];
                            for (m, block) in blocks.iter().enumerate() {
                                let g = (block.0[slot], block.1[slot]);
                                for density in [high, low] {
                                    for spin in 0..2 {
                                        if coulomb != 0.0 {
                                            let dm = &density[spin][order - m];
                                            let value = (dm.re[(sig, lam)], dm.im[(sig, lam)]);
                                            for output in &mut sums {
                                                add(&mut output[mu * nao + nu], g, value, coulomb);
                                            }
                                        }
                                        if exchange != 0.0 {
                                            let dm = &density[spin][order - m];
                                            add(
                                                &mut sums[spin][mu * nao + sig],
                                                g,
                                                (dm.re[(nu, lam)], dm.im[(nu, lam)]),
                                                -exchange,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut high = std::array::from_fn(|_| ComplexMatrix::zeros(nao, nao));
    let mut low = std::array::from_fn(|_| ComplexMatrix::zeros(nao, nao));
    for s in 0..2 {
        for i in 0..nao {
            for j in 0..nao {
                (high[s].re[(i, j)], low[s].re[(i, j)]) = sums[s][i * nao + j][0].parts();
                (high[s].im[(i, j)], low[s].im[(i, j)]) = sums[s][i * nao + j][1].parts();
            }
        }
    }
    Ok((high, low))
}

/// Coulomb and exchange matrices of order `n` from the field coefficients
/// `1..=n` of the electron repulsion integrals over London orbitals,
///
/// ```text
/// J_mn   = sum_(k=1..n) sum_ls g_k(m n | l s) D_(n-k)[s, l],
/// K^s_mn = sum_(k=1..n) sum_ls g_k(m s | l n) D^s_(n-k)[s, l],
/// ```
///
/// contracted shell quartet by shell quartet: no integral tensor is stored.
/// The zeroth coefficient, the field-free integrals, is left to the
/// integral-direct engines of the self-consistent field. `density[spin]`
/// holds the density coefficients of orders `0..n`.
pub fn contracted_repulsion_series(
    mb: &MolBasis,
    path: &FieldPath,
    omega: Option<f64>,
    density: [&[ComplexMatrix]; 2],
    n: usize,
    want_coulomb: bool,
    want_exchange: bool,
) -> Result<(ComplexMatrix, [ComplexMatrix; 2])> {
    let nao = mb.nao_sph;
    let zero = || {
        (
            ComplexMatrix::zeros(nao, nao),
            [
                ComplexMatrix::zeros(nao, nao),
                ComplexMatrix::zeros(nao, nao),
            ],
        )
    };
    if !path.london || n == 0 || (!want_coulomb && !want_exchange) {
        return Ok(zero());
    }
    // Schwarz bounds of the field-free integrals screen the quartets.
    let quartets = SeriesQuartets::new(mb, path, omega);
    type Sum = (ComplexMatrix, [ComplexMatrix; 2]);
    quartets
        .pairs
        .par_iter()
        .try_fold(zero, |mut acc, &(a, b)| -> Result<Sum> {
            for &(c, d) in &quartets.pairs {
                let Some(blocks) = quartets.blocks([a, b, c, d], n)? else {
                    continue;
                };
                let shells = [a, b, c, d].map(|s| &mb.shells[s]);
                let shape = shells.map(|s| n_sph(s.l));
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            for l in 0..shape[3] {
                                let slot = ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
                                let mu = shells[0].sph_off + i;
                                let nu = shells[1].sph_off + j;
                                let lam = shells[2].sph_off + k;
                                let sig = shells[3].sph_off + l;
                                for m in 1..=n {
                                    let g = (blocks[m - 1].0[slot], blocks[m - 1].1[slot]);
                                    if g.0 == 0.0 && g.1 == 0.0 {
                                        continue;
                                    }
                                    let order = n - m;
                                    if want_coulomb {
                                        let (d0, d1) = (&density[0][order], &density[1][order]);
                                        let v = times(
                                            g,
                                            (
                                                d0.re[(sig, lam)] + d1.re[(sig, lam)],
                                                d0.im[(sig, lam)] + d1.im[(sig, lam)],
                                            ),
                                        );
                                        acc.0.re[(mu, nu)] += v.0;
                                        acc.0.im[(mu, nu)] += v.1;
                                    }
                                    if want_exchange {
                                        for s in 0..2 {
                                            let dm = &density[s][order];
                                            let v = times(g, (dm.re[(nu, lam)], dm.im[(nu, lam)]));
                                            acc.1[s].re[(mu, sig)] += v.0;
                                            acc.1[s].im[(mu, sig)] += v.1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(acc)
        })
        .try_reduce(zero, |mut x, y| {
            for (t, v) in x.0.re.as_mut_slice().iter_mut().zip(y.0.re.as_slice()) {
                *t += *v;
            }
            for (t, v) in x.0.im.as_mut_slice().iter_mut().zip(y.0.im.as_slice()) {
                *t += *v;
            }
            for s in 0..2 {
                for (t, v) in x.1[s]
                    .re
                    .as_mut_slice()
                    .iter_mut()
                    .zip(y.1[s].re.as_slice())
                {
                    *t += *v;
                }
                for (t, v) in x.1[s]
                    .im
                    .as_mut_slice()
                    .iter_mut()
                    .zip(y.1[s].im.as_slice())
                {
                    *t += *v;
                }
            }
            Ok(x)
        })
}

/// Dense tensor of the sum `g_1 + ... + g_n` of the field coefficients of the
/// electron repulsion integrals over London orbitals, real and imaginary
/// parts at `((mu nao + nu) nao + lam) nao + sig` for `(mu nu | lam sig)`,
/// with the screening of [`contracted_repulsion_series`]. When every order
/// carries the same density, as in the self-consistent field at a fixed
/// field, contracting it with [`contract_summed_repulsion`] gives the
/// matrices of [`contracted_repulsion_series`] without regenerating the
/// integral series in every iteration.
pub fn summed_repulsion_series(
    mb: &MolBasis,
    path: &FieldPath,
    omega: Option<f64>,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let nao = mb.nao_sph;
    let size = nao * nao * nao * nao;
    if !path.london || n == 0 {
        return Ok((vec![0.0; size], vec![0.0; size]));
    }
    let quartets = SeriesQuartets::new(mb, path, omega);
    // Each bra pair fills its own (mu, nu) rows; the rows are disjoint.
    let rows = quartets
        .pairs
        .par_iter()
        .map(|&(a, b)| -> Result<(usize, usize, Vec<(f64, f64)>)> {
            let (sa, sb) = (&mb.shells[a], &mb.shells[b]);
            let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
            let mut block = vec![(0.0, 0.0); na * nb * nao * nao];
            for &(c, d) in &quartets.pairs {
                let Some(blocks) = quartets.blocks([a, b, c, d], n)? else {
                    continue;
                };
                let (sc, sd) = (&mb.shells[c], &mb.shells[d]);
                let (nc, nd) = (n_sph(sc.l), n_sph(sd.l));
                for i in 0..na {
                    for j in 0..nb {
                        for k in 0..nc {
                            for l in 0..nd {
                                let slot = ((i * nb + j) * nc + k) * nd + l;
                                let lam = sc.sph_off + k;
                                let sig = sd.sph_off + l;
                                let target = &mut block[((i * nb + j) * nao + lam) * nao + sig];
                                for (re, im) in &blocks {
                                    target.0 += re[slot];
                                    target.1 += im[slot];
                                }
                            }
                        }
                    }
                }
            }
            Ok((a, b, block))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut re = vec![0.0; size];
    let mut im = vec![0.0; size];
    for (a, b, block) in rows {
        let (sa, sb) = (&mb.shells[a], &mb.shells[b]);
        let nb = n_sph(sb.l);
        for (index, (x, y)) in block.into_iter().enumerate() {
            let rest = index % (nao * nao);
            let pair = index / (nao * nao);
            let (i, j) = (pair / nb, pair % nb);
            let slot = ((sa.sph_off + i) * nao + sb.sph_off + j) * nao * nao + rest;
            re[slot] = x;
            im[slot] = y;
        }
    }
    Ok((re, im))
}

/// Coulomb and exchange matrices of a tensor from
/// [`summed_repulsion_series`] and one density per spin,
///
/// ```text
/// J_mn   = sum_ls G(m n | l s) (D_a + D_b)[s, l],
/// K^s_ms = sum_nl G(m n | l s) D^s[n, l],
/// ```
///
/// equal to [`contracted_repulsion_series`] with that density at every order.
pub fn contract_summed_repulsion(
    tensor: &(Vec<f64>, Vec<f64>),
    nao: usize,
    density: [&ComplexMatrix; 2],
    want_coulomb: bool,
    want_exchange: bool,
) -> (ComplexMatrix, [ComplexMatrix; 2]) {
    let (g_re, g_im) = tensor;
    assert_eq!(g_re.len(), nao * nao * nao * nao);
    // One row mu of J and of both exchange matrices per task.
    let rows: Vec<(Vec<(f64, f64)>, [Vec<(f64, f64)>; 2])> = (0..nao)
        .into_par_iter()
        .map(|mu| {
            let mut j = vec![(0.0, 0.0); nao];
            let mut k = [vec![(0.0, 0.0); nao], vec![(0.0, 0.0); nao]];
            for nu in 0..nao {
                for lam in 0..nao {
                    let base = ((mu * nao + nu) * nao + lam) * nao;
                    for sig in 0..nao {
                        let g = (g_re[base + sig], g_im[base + sig]);
                        if g.0 == 0.0 && g.1 == 0.0 {
                            continue;
                        }
                        if want_coulomb {
                            let v = times(
                                g,
                                (
                                    density[0].re[(sig, lam)] + density[1].re[(sig, lam)],
                                    density[0].im[(sig, lam)] + density[1].im[(sig, lam)],
                                ),
                            );
                            j[nu].0 += v.0;
                            j[nu].1 += v.1;
                        }
                        if want_exchange {
                            for (s, row) in k.iter_mut().enumerate() {
                                let d = &density[s];
                                let v = times(g, (d.re[(nu, lam)], d.im[(nu, lam)]));
                                row[sig].0 += v.0;
                                row[sig].1 += v.1;
                            }
                        }
                    }
                }
            }
            (j, k)
        })
        .collect();
    let mut coulomb = ComplexMatrix::zeros(nao, nao);
    let mut exchange = [
        ComplexMatrix::zeros(nao, nao),
        ComplexMatrix::zeros(nao, nao),
    ];
    for (mu, (j, k)) in rows.into_iter().enumerate() {
        for col in 0..nao {
            coulomb.re[(mu, col)] = j[col].0;
            coulomb.im[(mu, col)] = j[col].1;
            for s in 0..2 {
                exchange[s].re[(mu, col)] = k[s][col].0;
                exchange[s].im[(mu, col)] = k[s][col].1;
            }
        }
    }
    (coulomb, exchange)
}

type Dual3 = crate::dual::DualN<f64, 3>;

/// The derivative of a London orbital with respect to component `axis` of
/// its centre, as complex-weighted combinations of primitives on that
/// centre: `d w / dR = exp(-i A . r) [-d chi / dr - (i / 2) (b x e) . r chi]`.
fn centre_derivative(
    ket: &Terms,
    field: [f64; 3],
    axis: usize,
    london: bool,
) -> Vec<((f64, f64), Terms)> {
    let mut out = vec![((-1.0, 0.0), ket.derivative(axis))];
    if london {
        let mut unit = [0.0; 3];
        unit[axis] = 1.0;
        let terms = ket.dotted(half_cross(field, unit));
        if !terms.items.is_empty() {
            out.push(((0.0, -1.0), terms));
        }
    }
    out
}

/// Overlap and one-electron integral of `bra` with the combination `ket`
/// whose phase is centred at `gauge`, summed over the field series through
/// `path.order`.
fn summed_pair(
    bra: &Primitive<f64>,
    ket: &Terms,
    k: [f64; 3],
    gauge: [f64; 3],
    path: &FieldPath,
    nuclei: &[([f64; 3], f64)],
) -> Result<[(f64, f64); 2]> {
    let field = ScalarFieldPath {
        electric: path.electric,
        magnetic: path.magnetic,
        origin: path.origin,
        london: path.london,
        phase_order: path.order,
        derivative_order: 0,
    };
    let result = scalar_field_pair_terms(bra, ket, k, gauge, nuclei, &field, false)?;
    Ok([result[0], result[1]])
}

fn lifted(p: &Primitive<f64>) -> Primitive<Dual3> {
    Primitive {
        angular: p.angular,
        exponent: Dual3::constant(p.exponent),
        center: p.center.map(Dual3::constant),
    }
}

/// Real and imaginary parts of the block of `x` between two shells in
/// Cartesian components, `T_a X T_b^T`: contracting it with a Cartesian
/// integral block is the contraction of `x` with the spherical one.
fn cartesian_block(x: &ComplexMatrix, sa: &MolShell, sb: &MolShell) -> (Matrix, Matrix) {
    let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
    let mut re = Matrix::zeros(na, nb);
    let mut im = Matrix::zeros(na, nb);
    for i in 0..na {
        for j in 0..nb {
            re[(i, j)] = x.re[(sa.sph_off + i, sb.sph_off + j)];
            im[(i, j)] = x.im[(sa.sph_off + i, sb.sph_off + j)];
        }
    }
    let (ta, tb) = (cart_to_sph(sa.l), cart_to_sph(sb.l).transpose());
    (
        ta.matmul_seq(&re).matmul_seq(&tb),
        ta.matmul_seq(&im).matmul_seq(&tb),
    )
}

/// Derivative with respect to the position of every atom of
/// `Re Tr[P_h h] + Re Tr[P_s S]` over London orbitals at fixed Hermitian
/// matrices `core_density = P_h` and `overlap_density = P_s`, with the
/// one-electron matrix `h` and the overlap `S` summed over the field series
/// through `path.order`. The centres of the basis functions, with their
/// phases, and the nuclei of the attraction move with the atoms.
pub fn one_electron_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    path: &FieldPath,
    core_density: &ComplexMatrix,
    overlap_density: &ComplexMatrix,
) -> Result<Vec<[f64; 3]>> {
    let natoms = mol.len();
    let charges = mb.effective_charges(mol);
    let nuclei: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .zip(&charges)
        .map(|(atom, &z)| (atom.position.to_array(), z))
        .collect();
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let zero = || vec![[0.0; 3]; natoms];
    pairs
        .par_iter()
        .try_fold(zero, |mut acc, &(a, b)| -> Result<Vec<[f64; 3]>> {
            let (sa, sb) = (&mb.shells[a], &mb.shells[b]);
            let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
            let (core_re, core_im) = cartesian_block(core_density, sa, sb);
            let (over_re, over_im) = cartesian_block(overlap_density, sa, sb);
            let centre_b = sb.center.to_array();
            let separation = [
                sa.center.x - sb.center.x,
                sa.center.y - sb.center.y,
                sa.center.z - sb.center.z,
            ];
            let k = if path.london {
                half_cross(path.magnetic, separation)
            } else {
                [0.0; 3]
            };
            let gauge = if path.london { centre_b } else { path.origin };
            for (ia, &component_a) in ca.iter().enumerate() {
                for (ib, &component_b) in cb.iter().enumerate() {
                    // Parts of the element (m, n) of P_s and of P_h.
                    let weights = [
                        (over_re[(ia, ib)], over_im[(ia, ib)]),
                        (core_re[(ia, ib)], core_im[(ia, ib)]),
                    ];
                    if weights.iter().all(|w| w.0 == 0.0 && w.1 == 0.0) {
                        continue;
                    }
                    for pa in 0..sa.exps.len() {
                        let bra = primitive(sa, component_a, pa);
                        for pb in 0..sb.exps.len() {
                            let weight = sa.coefs[pa] * sb.coefs[pb];
                            let ket = Terms::single(primitive(sb, component_b, pb));
                            // Centre of the ket function; the bra function is
                            // the complex conjugate of the transposed pair.
                            for axis in 0..3 {
                                for (factor, terms) in
                                    centre_derivative(&ket, path.magnetic, axis, path.london)
                                {
                                    let values =
                                        summed_pair(&bra, &terms, k, gauge, path, &nuclei)?;
                                    for (value, w) in values.iter().zip(&weights) {
                                        let v = times(factor, *value);
                                        acc[sb.atom][axis] +=
                                            2.0 * weight * (v.0 * w.0 + v.1 * w.1);
                                    }
                                }
                            }
                            // Positions of the nuclei in the attraction.
                            let w = weights[1];
                            if w.0 == 0.0 && w.1 == 0.0 {
                                continue;
                            }
                            let plain = phased(&ket, k, path.order);
                            let bra_dual = lifted(&bra);
                            for (n, terms) in plain.iter().enumerate() {
                                let (re, im) = phase_factor(n);
                                for (c, p) in &terms.items {
                                    let ket_dual = lifted(p);
                                    for (atom, (centre, charge)) in nuclei.iter().enumerate() {
                                        if *charge == 0.0 {
                                            continue;
                                        }
                                        let value = analytic::attraction(
                                            &bra_dual,
                                            &ket_dual,
                                            std::array::from_fn(|x| Dual3::variable(centre[x], x)),
                                            Dual3::constant(*charge),
                                            GaussianKernel::Coulomb,
                                            1,
                                        )?;
                                        for axis in 0..3 {
                                            acc[atom][axis] +=
                                                weight * c * value.im[axis] * (re * w.0 + im * w.1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(acc)
        })
        .try_reduce(zero, |mut x, y| {
            for (t, v) in x.iter_mut().zip(&y) {
                for axis in 0..3 {
                    t[axis] += v[axis];
                }
            }
            Ok(x)
        })
}

/// Spherical blocks of `d (a b | c d) / dR_b`, the derivative with respect
/// to the three components of the centre of the second function, summed
/// over the field series through `path.order`.
fn quartet_centre_derivative(
    mb: &MolBasis,
    shells: [usize; 4],
    k_bra: [f64; 3],
    k_ket: [f64; 3],
    path: &FieldPath,
    kernel: GaussianKernel<f64>,
) -> Result<[(Vec<f64>, Vec<f64>); 3]> {
    let last = if path.london { path.order } else { 0 };
    let [sa, sb, sc, sd] = shells.map(|s| &mb.shells[s]);
    let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
    let (cc, cd) = (cart_components(sc.l), cart_components(sd.l));
    let size = ca.len() * cb.len() * cc.len() * cd.len();
    let mut cart: [(Vec<f64>, Vec<f64>); 3] =
        std::array::from_fn(|_| (vec![0.0; size], vec![0.0; size]));
    for (ia, &xa) in ca.iter().enumerate() {
        for (ib, &xb) in cb.iter().enumerate() {
            for (ic, &xc) in cc.iter().enumerate() {
                for (id, &xd) in cd.iter().enumerate() {
                    let slot = ((ia * cb.len() + ib) * cc.len() + ic) * cd.len() + id;
                    for pa in 0..sa.exps.len() {
                        let bra = primitive(sa, xa, pa);
                        for pb in 0..sb.exps.len() {
                            let ket = Terms::single(primitive(sb, xb, pb));
                            let seconds: Vec<Vec<((f64, f64), Vec<Terms>)>> = (0..3)
                                .map(|axis| {
                                    centre_derivative(&ket, path.magnetic, axis, path.london)
                                        .into_iter()
                                        .map(|(factor, terms)| {
                                            (factor, phased(&terms, k_bra, last))
                                        })
                                        .collect()
                                })
                                .collect();
                            for pc in 0..sc.exps.len() {
                                let third = primitive(sc, xc, pc);
                                for pd in 0..sd.exps.len() {
                                    let fourth =
                                        phased(&Terms::single(primitive(sd, xd, pd)), k_ket, last);
                                    let weight =
                                        sa.coefs[pa] * sb.coefs[pb] * sc.coefs[pc] * sd.coefs[pd];
                                    for (axis, list) in seconds.iter().enumerate() {
                                        for (factor, second) in list {
                                            for n in 0..=last {
                                                let (re, im) = phase_factor(n);
                                                // i^n / (p! q!) for p + q = n.
                                                let total = (1..=n).fold(1.0, |x, y| x * y as f64);
                                                for p in 0..=n {
                                                    let q = n - p;
                                                    let split = (1..=p)
                                                        .fold(1.0, |x, y| x * y as f64)
                                                        * (1..=q).fold(1.0, |x, y| x * y as f64);
                                                    let mut value = 0.0;
                                                    for (cb_coef, tb) in &second[p].items {
                                                        for (cd_coef, td) in &fourth[q].items {
                                                            value += cb_coef
                                                                * cd_coef
                                                                * analytic::repulsion(
                                                                    &bra, tb, &third, td, kernel, 0,
                                                                )?;
                                                        }
                                                    }
                                                    let scaled = weight * value * total / split;
                                                    let v =
                                                        times(*factor, (re * scaled, im * scaled));
                                                    cart[axis].0[slot] += v.0;
                                                    cart[axis].1[slot] += v.1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let spherical = |values: &[f64]| -> Vec<f64> {
        let mut shape = [ca.len(), cb.len(), cc.len(), cd.len()];
        let mut current = values.to_vec();
        for (axis, shell) in [sa, sb, sc, sd].into_iter().enumerate() {
            current = transform_axis(&current, &mut shape, axis, shell.l);
        }
        current
    };
    Ok(cart.map(|(re, im)| (spherical(&re), spherical(&im))))
}

/// Gradients of the Coulomb and exchange energies of fixed spin densities
/// over London orbitals with respect to the position of every atom, without
/// their prefactors:
///
/// ```text
/// coulomb  = d/dR  (1/2) sum (m n | l s) D_nm D_sl,
/// exchange = d/dR  (1/2) sum_spin sum (m n | l s) D_nl D_sm,
/// ```
///
/// with the integrals summed over the field series through `path.order`.
/// The four centres of an integral are equivalent in pairs and by complex
/// conjugation, so each derivative is four times the real part of the term
/// in which the second function moves.
pub fn repulsion_gradient(
    mb: &MolBasis,
    path: &FieldPath,
    omega: Option<f64>,
    density: [&ComplexMatrix; 2],
    natoms: usize,
    want_coulomb: bool,
    want_exchange: bool,
) -> Result<(Vec<[f64; 3]>, Vec<[f64; 3]>)> {
    type Sum = (Vec<[f64; 3]>, Vec<[f64; 3]>);
    let zero = || -> Sum { (vec![[0.0; 3]; natoms], vec![[0.0; 3]; natoms]) };
    if !want_coulomb && !want_exchange {
        return Ok(zero());
    }
    let kernel = omega.map_or(GaussianKernel::Coulomb, |omega| GaussianKernel::LongRange {
        omega,
    });
    let bound = crate::integrals::schwarz_matrix(
        mb,
        match omega {
            None => crate::integrals::CoulombKernel::Coulomb,
            Some(omega) => crate::integrals::CoulombKernel::Erf { omega },
        },
    );
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let pair_phase = |a: usize, b: usize| -> [f64; 3] {
        if !path.london {
            return [0.0; 3];
        }
        let (sa, sb) = (&mb.shells[a], &mb.shells[b]);
        half_cross(
            path.magnetic,
            [
                sa.center.x - sb.center.x,
                sa.center.y - sb.center.y,
                sa.center.z - sb.center.z,
            ],
        )
    };
    let element = |m: &ComplexMatrix, i: usize, j: usize| (m.re[(i, j)], m.im[(i, j)]);
    pairs
        .par_iter()
        .try_fold(zero, |mut acc, &(a, b)| -> Result<Sum> {
            let atom = mb.shells[b].atom;
            for &(c, d) in &pairs {
                if bound[(a, b)] * bound[(c, d)] < 1e-15 {
                    continue;
                }
                let blocks = quartet_centre_derivative(
                    mb,
                    [a, b, c, d],
                    pair_phase(a, b),
                    pair_phase(c, d),
                    path,
                    kernel,
                )?;
                let shells = [a, b, c, d].map(|s| &mb.shells[s]);
                let shape = shells.map(|s| n_sph(s.l));
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            for l in 0..shape[3] {
                                let slot = ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
                                let mu = shells[0].sph_off + i;
                                let nu = shells[1].sph_off + j;
                                let lam = shells[2].sph_off + k;
                                let sig = shells[3].sph_off + l;
                                for axis in 0..3 {
                                    let g = (blocks[axis].0[slot], blocks[axis].1[slot]);
                                    if g.0 == 0.0 && g.1 == 0.0 {
                                        continue;
                                    }
                                    if want_coulomb {
                                        let (d0, d1) = (density[0], density[1]);
                                        let left = element(d0, nu, mu);
                                        let other = element(d1, nu, mu);
                                        let right = element(d0, sig, lam);
                                        let more = element(d1, sig, lam);
                                        let product = times(
                                            (left.0 + other.0, left.1 + other.1),
                                            (right.0 + more.0, right.1 + more.1),
                                        );
                                        acc.0[atom][axis] += 2.0 * times(product, g).0;
                                    }
                                    if want_exchange {
                                        for spin in density {
                                            let product = times(
                                                element(spin, nu, lam),
                                                element(spin, sig, mu),
                                            );
                                            acc.1[atom][axis] += 2.0 * times(product, g).0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(acc)
        })
        .try_reduce(zero, |mut x, y| {
            for (t, v) in x.0.iter_mut().zip(&y.0) {
                for axis in 0..3 {
                    t[axis] += v[axis];
                }
            }
            for (t, v) in x.1.iter_mut().zip(&y.1) {
                for axis in 0..3 {
                    t[axis] += v[axis];
                }
            }
            Ok(x)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::math::Vec3;
    use crate::system::Atom;

    /// The shared Hermite engine gives the field coefficients of a raised
    /// primitive quartet that the term-by-term evaluation of every phase
    /// monomial gives, for both kernels and phases on two axes.
    #[test]
    fn hermite_engine_reproduces_the_term_by_term_series() {
        let a = Primitive {
            angular: [1, 0, 0],
            exponent: 0.8,
            center: [0.1, -0.2, 0.3],
        };
        let b = Primitive {
            angular: [0, 1, 1],
            exponent: 1.3,
            center: [-0.4, 0.5, 0.2],
        };
        let c = Primitive {
            angular: [0, 0, 1],
            exponent: 0.6,
            center: [0.7, 0.1, -0.5],
        };
        let d = Primitive {
            angular: [1, 0, 0],
            exponent: 1.1,
            center: [-0.3, -0.6, 0.4],
        };
        let k_bra = [0.03, -0.05, 0.0];
        let k_ket = [-0.02, 0.04, 0.01];
        let last = 6;
        for kernel in [
            GaussianKernel::Coulomb,
            GaussianKernel::LongRange { omega: 0.4 },
        ] {
            let engine = primitive_repulsion_coefficients(
                [&a, &b, &c, &d],
                kernel,
                k_bra,
                k_ket,
                0,
                last,
                0,
                |x| x == 0.0,
            )
            .unwrap();
            let second = phased(&Terms::single(b), k_bra, last);
            let fourth = phased(&Terms::single(d), k_ket, last);
            let direct = series_from_pairs(0, last, |p, q| {
                let mut value = 0.0;
                for (cb, tb) in &second[p].items {
                    for (cd, td) in &fourth[q].items {
                        value += cb * cd * analytic::repulsion(&a, tb, &c, td, kernel, 0).unwrap();
                    }
                }
                value
            });
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for ((x0, y0), (x1, y1)) in engine.iter().zip(&direct) {
                worst = worst.max((x0 - x1).abs()).max((y0 - y1).abs());
                scale = scale.max(x1.abs()).max(y1.abs());
            }
            println!("{kernel:?}: largest difference {worst:e} of {scale:e}");
            assert!(scale > 1e-6);
            assert!(worst <= 1e-13 * scale, "difference {worst:e} of {scale:e}");
        }
    }

    /// The stored sum of the field coefficients contracts to the same
    /// Coulomb and exchange matrices as the quartet-direct series with the
    /// same density at every order, for the Coulomb and long-range kernels.
    #[test]
    fn summed_series_tensor_reproduces_the_direct_contraction() {
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.2217),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.4309, -0.8867),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.13, -1.4309, -0.8867),
            },
        ]);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let nao = mb.nao_sph;
        let path = FieldPath {
            electric: [0.0; 3],
            magnetic: [0.02, -0.03, 0.05],
            origin: [0.0; 3],
            london: true,
            order: 6,
        };
        // Generic complex densities, different per spin.
        let density: [ComplexMatrix; 2] = std::array::from_fn(|s| {
            let mut d = ComplexMatrix::zeros(nao, nao);
            for i in 0..nao {
                for j in 0..nao {
                    let x = (i * nao + j) as f64 + 0.37 * s as f64;
                    d.re[(i, j)] = (0.7 * x).sin() * 0.3;
                    d.im[(i, j)] = (1.3 * x + 0.2).cos() * 0.1;
                }
            }
            d
        });
        for omega in [None, Some(0.4)] {
            let repeated: [Vec<ComplexMatrix>; 2] =
                std::array::from_fn(|s| vec![density[s].clone(); path.order]);
            let (j0, k0) = contracted_repulsion_series(
                &mb,
                &path,
                omega,
                [&repeated[0][..], &repeated[1][..]],
                path.order,
                true,
                true,
            )
            .unwrap();
            let tensor = summed_repulsion_series(&mb, &path, omega, path.order).unwrap();
            let (j1, k1) =
                contract_summed_repulsion(&tensor, nao, [&density[0], &density[1]], true, true);
            let mut worst = 0.0_f64;
            let mut scale = 0.0_f64;
            for (a, b) in [(&j0, &j1), (&k0[0], &k1[0]), (&k0[1], &k1[1])] {
                for (x, y) in
                    a.re.as_slice()
                        .iter()
                        .zip(b.re.as_slice())
                        .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
                {
                    worst = worst.max((x - y).abs());
                    scale = scale.max(x.abs());
                }
            }
            println!("omega {omega:?}: largest difference {worst:e} of {scale:e}");
            assert!(scale > 1e-4, "the series contraction must not vanish");
            assert!(worst <= 1e-13 * scale, "difference {worst:e} of {scale:e}");
        }
    }
}
