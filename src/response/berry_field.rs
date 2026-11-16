// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Density-matrix Berry electric enthalpy on a finite periodic cell.
//! Complex spinors and real collinear references use the same determinant,
//! analytic Fock, density kernel and moving-operator Taylor convolutions.
use crate::linalg::Matrix;
use crate::{DftError, Result};

fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
pub(crate) type ComplexMatrix = (Matrix, Matrix);

fn complex_product(a: &ComplexMatrix, b: &ComplexMatrix) -> ComplexMatrix {
    let mut re = a.0.matmul_seq(&b.0);
    add(&mut re, &a.1.matmul_seq(&b.1), -1.0);
    let mut im = a.0.matmul_seq(&b.1);
    add(&mut im, &a.1.matmul_seq(&b.0), 1.0);
    (re, im)
}

/// Inverse of a complex matrix and the argument of its determinant, by
/// Gauss-Jordan elimination with partial pivoting.
fn complex_inverse(m: &ComplexMatrix) -> Result<(ComplexMatrix, f64)> {
    let n = m.0.rows;
    let mut a: Vec<(f64, f64)> = (0..n * n)
        .map(|i| (m.0.as_slice()[i], m.1.as_slice()[i]))
        .collect();
    let mut inverse: Vec<(f64, f64)> = vec![(0.0, 0.0); n * n];
    for i in 0..n {
        inverse[i * n + i] = (1.0, 0.0);
    }
    let times = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let mut argument = 0.0_f64;
    for col in 0..n {
        let mut pivot = col;
        let mut largest = 0.0_f64;
        for row in col..n {
            let x = a[row * n + col];
            let size = x.0 * x.0 + x.1 * x.1;
            if size > largest {
                largest = size;
                pivot = row;
            }
        }
        if !(largest > 1e-280) {
            return Err(DftError::InvalidInput(
                "the single-point Berry phase of the cell is undefined: the determinant of \
                 exp(i b . r) between the occupied orbitals vanishes"
                    .into(),
            ));
        }
        if pivot != col {
            for k in 0..n {
                a.swap(pivot * n + k, col * n + k);
                inverse.swap(pivot * n + k, col * n + k);
            }
            argument += std::f64::consts::PI;
        }
        let p = a[col * n + col];
        argument += p.1.atan2(p.0);
        let scale = (p.0 / largest, -p.1 / largest);
        for k in 0..n {
            a[col * n + k] = times(a[col * n + k], scale);
            inverse[col * n + k] = times(inverse[col * n + k], scale);
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == (0.0, 0.0) {
                continue;
            }
            for k in 0..n {
                let x = times(factor, a[col * n + k]);
                let y = times(factor, inverse[col * n + k]);
                a[row * n + k].0 -= x.0;
                a[row * n + k].1 -= x.1;
                inverse[row * n + k].0 -= y.0;
                inverse[row * n + k].1 -= y.1;
            }
        }
    }
    let mut out = (Matrix::zeros(n, n), Matrix::zeros(n, n));
    for (i, value) in inverse.iter().enumerate() {
        out.0.as_mut_slice()[i] = value.0;
        out.1.as_mut_slice()[i] = value.1;
    }
    // Principal value.
    let turn = 2.0 * std::f64::consts::PI;
    Ok((out, argument - turn * (argument / turn).round()))
}

/// N(t) = M(t)^-1 with the physical complex inverse at order zero.
fn complex_inverse_path(m: &[ComplexMatrix]) -> Result<(Vec<ComplexMatrix>, f64)> {
    let (first, argument) = complex_inverse(&m[0])?;
    let nao = m[0].0.rows;
    let mut inverse = vec![first];
    for n in 1..m.len() {
        let mut sum = (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao));
        for k in 1..=n {
            let term = complex_product(&m[k], &inverse[n - k]);
            add(&mut sum.0, &term.0, 1.0);
            add(&mut sum.1, &term.1, 1.0);
        }
        let product = complex_product(&inverse[0], &sum);
        let mut next = (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao));
        add(&mut next.0, &product.0, -1.0);
        add(&mut next.1, &product.1, -1.0);
        inverse.push(next);
    }
    Ok((inverse, argument))
}

fn sum(a: &mut ComplexMatrix, b: &ComplexMatrix, scale: f64) {
    add(&mut a.0, &b.0, scale);
    add(&mut a.1, &b.1, scale);
}

/// Hermitian gradient of Im Tr[K dP] on a complex Hermitian density space.
fn hermitian_imaginary(k: &ComplexMatrix) -> ComplexMatrix {
    let mut real = k.1.clone();
    add(&mut real, &k.1.transpose(), 1.0);
    let mut imag = k.0.transpose();
    add(&mut imag, &k.0, -1.0);
    for x in real.as_mut_slice().iter_mut().chain(imag.as_mut_slice()) {
        *x *= 0.5;
    }
    (real, imag)
}

fn hermitian(k: &ComplexMatrix) -> ComplexMatrix {
    let mut real = k.0.clone();
    add(&mut real, &k.0.transpose(), 1.0);
    let mut imag = k.1.clone();
    add(&mut imag, &k.1.transpose(), -1.0);
    for x in real.as_mut_slice().iter_mut().chain(imag.as_mut_slice()) {
        *x *= 0.5;
    }
    (real, imag)
}

/// I+P(Q-S) has the occupied determinant of exp(i b.r) when P S P=P.
/// `polar` contains Q-S, `ionic` sum_A Z_A b.R_A and `reduced` E.a/(2pi).
/// A complex spinor density is one channel; collinear densities are separate
/// channels with occupation one. The determinant phase is continued locally
/// from its supplied geometric branch, never wrapped at higher orders.
#[derive(Clone)]
pub(crate) struct BerryField {
    pub polar: [Vec<ComplexMatrix>; 3],
    pub ionic: [Vec<f64>; 3],
    pub reduced: [Vec<f64>; 3],
    /// Electronic phase branch near an explicitly transported geometry
    /// reference. It selects an integer multiple of 2pi only; derivatives
    /// still come from the physical determinant, not from this anchor.
    pub phase_reference: Option<[f64; 3]>,
}

impl BerryField {
    fn branch_shift(&self, axis: usize, phase: f64) -> f64 {
        self.phase_reference.map_or(0.0, |anchor| {
            let turn = 2.0 * std::f64::consts::PI;
            turn * ((anchor[axis] - phase) / turn).round()
        })
    }
    fn validate(&self, p: &[Vec<ComplexMatrix>], order: usize) -> Result<usize> {
        let n = p.first().and_then(|v| v.first()).map_or(0, |m| m.0.rows);
        if n == 0
            || self
                .phase_reference
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
            || (0..3).any(|a| {
                self.polar[a].len() <= order
                    || self.ionic[a].len() <= order
                    || self.reduced[a].len() <= order
            })
            || p.iter().any(|v| v.len() <= order)
            || p.iter()
                .flatten()
                .chain(self.polar.iter().flatten())
                .any(|m| {
                    [&m.0, &m.1].iter().any(|r| {
                        r.rows != n || r.cols != n || r.as_slice().iter().any(|x| !x.is_finite())
                    })
                })
            || self
                .ionic
                .iter()
                .chain(&self.reduced)
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput("Berry field requires finite complete square density, polar and scalar coefficient paths".into()));
        }
        Ok(n)
    }

    fn overlap_path(&self, axis: usize, p: &[ComplexMatrix], order: usize) -> Vec<ComplexMatrix> {
        let n = p[0].0.rows;
        (0..=order)
            .map(|j| {
                let mut m = (
                    if j == 0 {
                        Matrix::identity(n)
                    } else {
                        Matrix::zeros(n, n)
                    },
                    Matrix::zeros(n, n),
                );
                for k in 0..=j {
                    sum(
                        &mut m,
                        &complex_product(&p[k], &self.polar[axis][j - k]),
                        1.0,
                    );
                }
                m
            })
            .collect()
    }

    fn gradient_path(
        &self,
        axis: usize,
        inverse: &[ComplexMatrix],
        order: usize,
    ) -> Vec<ComplexMatrix> {
        let n = inverse[0].0.rows;
        (0..=order)
            .map(|j| {
                let mut k = (Matrix::zeros(n, n), Matrix::zeros(n, n));
                for l in 0..=j {
                    sum(
                        &mut k,
                        &complex_product(&self.polar[axis][l], &inverse[j - l]),
                        1.0,
                    );
                }
                k
            })
            .collect()
    }

    pub fn phases(&self, p: &[ComplexMatrix]) -> Result<[f64; 3]> {
        let path: Vec<_> = p.iter().map(|m| vec![m.clone()]).collect();
        self.validate(&path, 0)?;
        let mut phases = [0.0; 3];
        for (a, value) in phases.iter_mut().enumerate() {
            *value = -self.ionic[a][0];
            for density in &path {
                *value += complex_inverse(&self.overlap_path(a, density, 0)[0])?.1;
            }
            *value += self.branch_shift(a, *value + self.ionic[a][0]);
        }
        Ok(phases)
    }

    /// Exact density kernel -HermitianImag[K deltaP K], K=(Q-S) M^-1.
    pub fn linear(
        &self,
        p: &[Vec<ComplexMatrix>],
        q: &[Vec<ComplexMatrix>],
        order: usize,
    ) -> Result<Vec<Vec<ComplexMatrix>>> {
        let n = self.validate(p, order)?;
        self.validate(q, order)?;
        if p.len() != q.len() {
            return Err(DftError::InvalidInput(
                "Berry density/kernel channel counts differ".into(),
            ));
        }
        let width = order + 1;
        let zero = || (Matrix::zeros(n, n), Matrix::zeros(n, n));
        let mut output = vec![vec![zero(); width]; p.len()];
        for a in 0..3 {
            if self.reduced[a][..width].iter().all(|v| *v == 0.0) {
                continue;
            }
            for s in 0..p.len() {
                let (inverse, _) = complex_inverse_path(&self.overlap_path(a, &p[s], order))?;
                let k = self.gradient_path(a, &inverse, order);
                let q: Vec<_> = q[s][..width].iter().map(hermitian).collect();
                for j in 0..width {
                    let mut gradient = zero();
                    for l in 0..=j {
                        for m in 0..=j - l {
                            sum(
                                &mut gradient,
                                &complex_product(&complex_product(&k[l], &q[m]), &k[j - l - m]),
                                -1.0,
                            );
                        }
                    }
                    let gradient = hermitian_imaginary(&gradient);
                    for v in j..width {
                        sum(&mut output[s][v], &gradient, self.reduced[a][v - j]);
                    }
                }
            }
        }
        Ok(output)
    }

    /// Analytic electric enthalpy and its variational complex Hermitian Fock.
    pub fn evaluate(
        &self,
        p: &[Vec<ComplexMatrix>],
        order: usize,
    ) -> Result<(Vec<f64>, Vec<Vec<ComplexMatrix>>)> {
        let n = self.validate(p, order)?;
        let width = order + 1;
        let zero = || (Matrix::zeros(n, n), Matrix::zeros(n, n));
        let mut energy = vec![0.0; width];
        let mut fock = vec![vec![zero(); width]; p.len()];
        for a in 0..3 {
            if self.reduced[a][..width].iter().all(|v| *v == 0.0) {
                continue;
            }
            for j in 0..width {
                for l in 0..=j {
                    energy[j] -= self.reduced[a][l] * self.ionic[a][j - l];
                }
            }
            let mut phase_zero = 0.0;
            for s in 0..p.len() {
                let m = self.overlap_path(a, &p[s], order);
                let (inverse, argument) = complex_inverse_path(&m)?;
                phase_zero += argument;
                let mut phases = vec![argument];
                for j in 1..width {
                    let value = (1..=j)
                        .map(|l| {
                            let term = complex_product(&inverse[j - l], &m[l]);
                            l as f64 * (0..n).map(|i| term.1[(i, i)]).sum::<f64>()
                        })
                        .sum::<f64>();
                    phases.push(value / j as f64);
                }
                let gradients = self.gradient_path(a, &inverse, order);
                for j in 0..width {
                    let gradient = hermitian_imaginary(&gradients[j]);
                    for v in j..width {
                        sum(&mut fock[s][v], &gradient, self.reduced[a][v - j]);
                    }
                    for l in 0..=j {
                        energy[j] += self.reduced[a][l] * phases[j - l];
                    }
                }
            }
            let shift = self.branch_shift(a, phase_zero);
            for (j, value) in energy.iter_mut().enumerate() {
                *value += self.reduced[a][j] * shift;
            }
        }
        Ok((energy, fock))
    }
}

#[cfg(test)]
mod tests;
