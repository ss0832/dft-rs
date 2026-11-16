// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! First-order London-basis Coulomb response in the zero-field active frame.
//!
//! The metric connection is -S'/2. All four ERI indices and the optical
//! operator carry this connection. The zero-field SOC matrix is frozen in
//! that orthonormal frame; no moving-SOC or higher-field-order claim is made.
use super::{add, annihilate, create, one_body, put, z};
use crate::basis::MolBasis;
use crate::integrals::london::{one_electron_series, repulsion_series, ComplexMatrix, FieldPath};
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::system::Molecule;
use crate::Result;
use std::collections::BTreeMap;

pub(super) struct LondonLinear {
    pub magnetic: [BlochMatrix; 3],
    /// First index magnetic field, second optical polarization.
    pub electric_derivative: [[BlochMatrix; 3]; 3],
}

fn project(a: &ComplexMatrix, c: &Matrix) -> BlochMatrix {
    BlochMatrix {
        re: c.transpose_matmul(&a.re.matmul(c)),
        im: c.transpose_matmul(&a.im.matmul(c)),
    }
}

fn connected(a: &BlochMatrix, derivative: &BlochMatrix, t: &BlochMatrix) -> BlochMatrix {
    let mut out = derivative.clone();
    for p in 0..a.rows() {
        for q in 0..a.rows() {
            let mut v = z(&out, p, q);
            for k in 0..a.rows() {
                v += z(t, p, k) * z(a, k, q) + z(a, p, k) * z(t, k, q);
            }
            put(&mut out, p, q, v);
        }
    }
    out
}

fn transform4(a: &[Complex], nao: usize, c: &Matrix) -> Vec<Complex> {
    let m = c.cols;
    let mut out = a.to_vec();
    let mut dims = [nao; 4];
    for axis in 0..4 {
        let before = dims;
        dims[axis] = m;
        let size: usize = dims.iter().product();
        let mut next = vec![Complex::ZERO; size];
        for (flat, v) in next.iter_mut().enumerate() {
            let mut index = [0; 4];
            let mut rem = flat;
            for k in (0..4).rev() {
                index[k] = rem % dims[k];
                rem /= dims[k];
            }
            let orbital = index[axis];
            for mu in 0..nao {
                index[axis] = mu;
                let old = ((index[0] * before[1] + index[1]) * before[2] + index[2]) * before[3]
                    + index[3];
                *v += out[old] * c[(mu, orbital)];
            }
        }
        out = next;
    }
    out
}

fn two_body(eri: &[Complex], n: usize, det: &[u64]) -> BlochMatrix {
    let index: BTreeMap<u64, usize> = det.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let mut out = BlochMatrix::zeros(det.len(), det.len());
    for (column, &bits) in det.iter().enumerate() {
        for q in 0..2 * n {
            if let Some((b, sq)) = annihilate(bits, q) {
                for s in 0..2 * n {
                    if let Some((b, ss)) = annihilate(b, s) {
                        for r in s / n * n..s / n * n + n {
                            if let Some((b, sr)) = create(b, r) {
                                for p in q / n * n..q / n * n + n {
                                    if let Some((b, sp)) = create(b, p) {
                                        if let Some(&row) = index.get(&b) {
                                            add(
                                                &mut out,
                                                row,
                                                column,
                                                eri[(((p % n) * n + q % n) * n + r % n) * n
                                                    + s % n]
                                                    * (0.5 * sq * ss * sr * sp),
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
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    mol: &Molecule,
    mb: &MolBasis,
    coefficients: &Matrix,
    core: usize,
    h: &Matrix,
    eri: &[f64],
    det: &[u64],
    origin: [f64; 3],
    spin: &[BlochMatrix; 3],
) -> Result<LondonLinear> {
    let m = coefficients.cols;
    let n = m - core;
    let nao = mb.nao_sph;
    let zero = || BlochMatrix::zeros(det.len(), det.len());
    let mut magnetic = std::array::from_fn(|_| zero());
    let mut electric_derivative = std::array::from_fn(|_| std::array::from_fn(|_| zero()));
    let h0 = BlochMatrix {
        re: h.clone(),
        im: Matrix::zeros(m, m),
    };
    let v = |p, q, r, s| eri[((p * m + q) * m + r) * m + s];
    for axis in 0..3 {
        let mut b = [0.; 3];
        b[axis] = 1.;
        let path = FieldPath {
            electric: [0.; 3],
            magnetic: b,
            origin,
            london: true,
            order: 1,
        };
        let one = one_electron_series(mol, mb, &path)?;
        let metric = project(&one.overlap[1], coefficients);
        let mut t = metric;
        for value in t.re.as_mut_slice().iter_mut().chain(t.im.as_mut_slice()) {
            *value *= -0.5;
        }
        let dh = connected(&h0, &project(&one.core[1], coefficients), &t);
        let raw = repulsion_series(mb, &path, None)?;
        let raw: Vec<Complex> = raw[1]
            .0
            .iter()
            .zip(&raw[1].1)
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        let mut dv = transform4(&raw, nao, coefficients);
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        let idx = ((p * m + q) * m + r) * m + s;
                        for i in 0..m {
                            dv[idx] += z(&t, i, p).conj() * v(i, q, r, s)
                                + z(&t, i, q) * v(p, i, r, s)
                                + z(&t, i, r).conj() * v(p, q, i, s)
                                + z(&t, i, s) * v(p, q, r, i);
                        }
                    }
                }
            }
        }
        let dv_at = |p, q, r, s| dv[((p * m + q) * m + r) * m + s];
        let mut effective = BlochMatrix::zeros(2 * n, 2 * n);
        let mut active = vec![Complex::ZERO; n * n * n * n];
        for p in 0..n {
            for q in 0..n {
                let mut hpq = z(&dh, core + p, core + q);
                for i in 0..core {
                    hpq += dv_at(core + p, core + q, i, i) * 2. - dv_at(core + p, i, i, core + q);
                }
                put(&mut effective, p, q, hpq);
                put(&mut effective, n + p, n + q, hpq);
                for r in 0..n {
                    for s in 0..n {
                        active[((p * n + q) * n + r) * n + s] =
                            dv_at(core + p, core + q, core + r, core + s);
                    }
                }
            }
        }
        let mut derivative = one_body(&effective, det);
        derivative.add_scaled(&two_body(&active, n, det), 1.);
        for value in derivative
            .re
            .as_mut_slice()
            .iter_mut()
            .chain(derivative.im.as_mut_slice())
        {
            *value = -*value;
        }
        // Spin Zeeman is retained explicitly; the London core already contains
        // the orbital Zeeman term and its metric/ERI completion.
        derivative.add_scaled(&spin[axis], 1.);
        magnetic[axis] = derivative;
        for optical in 0..3 {
            let d0 = project(&one.dipole[optical][0], coefficients);
            let d1 = connected(&d0, &project(&one.dipole[optical][1], coefficients), &t);
            let mut active = BlochMatrix::zeros(2 * n, 2 * n);
            for p in 0..n {
                for q in 0..n {
                    for s in 0..2 {
                        put(
                            &mut active,
                            s * n + p,
                            s * n + q,
                            -z(&d1, core + p, core + q),
                        );
                    }
                }
            }
            electric_derivative[axis][optical] = one_body(&active, det);
        }
    }
    Ok(LondonLinear {
        magnetic,
        electric_derivative,
    })
}
