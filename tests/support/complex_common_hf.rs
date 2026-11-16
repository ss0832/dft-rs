// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dense complex constrained-HF finite-field oracle. No production
//! response Hessian or JK engine is called by the finite-field optimizer.

use dft_rs::dfpt::orbitals::{adjoint, matmul};
use dft_rs::integrals::eri::EriTensor;
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;

fn z(a: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(a.re[(i, j)], a.im[(i, j)])
}
fn real(a: Matrix) -> BlochMatrix {
    let im = Matrix::zeros(a.rows, a.cols);
    BlochMatrix { re: a, im }
}
fn density(c: &BlochMatrix, occupied: usize) -> BlochMatrix {
    let n = c.rows();
    let mut out = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let value =
                (0..occupied).fold(Complex::ZERO, |sum, p| sum + z(c, i, p) * z(c, j, p).conj());
            out.re[(i, j)] = value.re;
            out.im[(i, j)] = value.im;
        }
    }
    out
}

pub struct Model {
    pub overlap: BlochMatrix,
    pub core: BlochMatrix,
    pub eri: EriTensor,
    pub nuclear: f64,
    pub occupied: [usize; 2],
    pairs: Vec<(usize, usize)>,
}
pub struct State {
    pub energy: f64,
    pub density: [BlochMatrix; 2],
    pub residual: f64,
}
impl Model {
    pub fn new(
        overlap: Matrix,
        core: Matrix,
        eri: EriTensor,
        nuclear: f64,
        occupied: [usize; 2],
    ) -> Self {
        let n = core.rows;
        let pairs = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .filter(|&(p, q)| occupied.iter().any(|&no| (p < no) != (q < no)))
            .collect();
        Self {
            overlap: real(overlap),
            core: real(core),
            eri,
            nuclear,
            occupied,
            pairs,
        }
    }
    fn rotate(&self, c: &BlochMatrix, step: &[f64]) -> BlochMatrix {
        let n = c.cols();
        let mut generator = BlochMatrix::zeros(n, n);
        for (k, &(p, q)) in self.pairs.iter().enumerate() {
            let a = step[2 * k];
            let b = step[2 * k + 1];
            generator.re[(q, p)] = a;
            generator.re[(p, q)] = -a;
            generator.im[(q, p)] = b;
            generator.im[(p, q)] = b;
        }
        let mut identity = Matrix::zeros(n, n);
        for i in 0..n {
            identity[(i, i)] = 1.0;
        }
        let mut left = real(identity.clone());
        left.add_scaled(&generator, -0.5);
        let mut right = real(identity);
        right.add_scaled(&generator, 0.5);
        let mut rotation = BlochMatrix::zeros(n, n);
        for j in 0..n {
            let rhs = (0..n)
                .map(|i| right.re[(i, j)])
                .chain((0..n).map(|i| right.im[(i, j)]))
                .collect::<Vec<_>>();
            let column = solve_linear(&left.real_embedding(), &rhs).unwrap();
            for i in 0..n {
                rotation.re[(i, j)] = column[i];
                rotation.im[(i, j)] = column[n + i];
            }
        }
        matmul(c, &rotation).unwrap()
    }
    fn build(&self, c: &BlochMatrix, source: &Matrix, field: f64) -> (State, Vec<f64>) {
        let n = self.core.rows();
        let density = self.occupied.map(|no| density(c, no));
        let mut core = self.core.clone();
        for i in 0..n {
            for j in 0..n {
                core.im[(i, j)] += field * source[(i, j)];
            }
        }
        let mut coulomb = BlochMatrix::zeros(n, n);
        let mut exchange = [BlochMatrix::zeros(n, n), BlochMatrix::zeros(n, n)];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        coulomb.re[(i, j)] += (density[0].re[(k, l)] + density[1].re[(k, l)])
                            * self.eri.get(i, j, k, l);
                        for spin in 0..2 {
                            let integral = self.eri.get(i, k, j, l);
                            exchange[spin].re[(i, j)] += density[spin].re[(k, l)] * integral;
                            exchange[spin].im[(i, j)] += density[spin].im[(k, l)] * integral;
                        }
                    }
                }
            }
        }
        let mut fock = [core.clone(), core.clone()];
        let mut energy = self.nuclear;
        for spin in 0..2 {
            energy += density[spin].trace_product(&core)
                + 0.5 * density[spin].trace_product(&coulomb)
                - 0.5 * density[spin].trace_product(&exchange[spin]);
            fock[spin].add_scaled(&coulomb, 1.0);
            fock[spin].add_scaled(&exchange[spin], -1.0);
        }
        let mo = fock.map(|f| matmul(&adjoint(c).unwrap(), &matmul(&f, c).unwrap()).unwrap());
        let mut gradient = Vec::with_capacity(2 * self.pairs.len());
        for &(p, q) in &self.pairs {
            let mut value = Complex::ZERO;
            for spin in 0..2 {
                let difference = (if p < self.occupied[spin] { 1.0 } else { 0.0 })
                    - (if q < self.occupied[spin] { 1.0 } else { 0.0 });
                value += z(&mo[spin], p, q) * Complex::new(difference, 0.0);
            }
            gradient.extend([2.0 * value.re, -2.0 * value.im]);
        }
        let residual = gradient.iter().map(|v| v.abs()).fold(0.0, f64::max);
        (
            State {
                energy,
                density,
                residual,
            },
            gradient,
        )
    }
    pub fn stationary(&self, initial: &Matrix, source: &Matrix, field: f64) -> State {
        let mut c = real(initial.clone());
        let width = 2 * self.pairs.len();
        for iteration in 0..60 {
            let (state, gradient) = self.build(&c, source, field);
            if state.residual < 1e-12 {
                let metric =
                    matmul(&adjoint(&c).unwrap(), &matmul(&self.overlap, &c).unwrap()).unwrap();
                for i in 0..metric.rows() {
                    for j in 0..metric.cols() {
                        assert!((metric.re[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-10);
                        assert!(metric.im[(i, j)].abs() < 1e-10);
                    }
                }
                return state;
            }
            let delta = 1e-4;
            let mut hessian = Matrix::zeros(width, width);
            for j in 0..width {
                let mut step = vec![0.0; width];
                step[j] = delta;
                let plus = self.build(&self.rotate(&c, &step), source, field).1;
                step[j] = -delta;
                let minus = self.build(&self.rotate(&c, &step), source, field).1;
                for i in 0..width {
                    hessian[(i, j)] = (plus[i] - minus[i]) / (2.0 * delta);
                }
            }
            for i in 0..width {
                for j in 0..i {
                    let value = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
                    hessian[(i, j)] = value;
                    hessian[(j, i)] = value;
                }
            }
            let rhs = gradient.iter().map(|v| -v).collect::<Vec<_>>();
            let mut step = solve_linear(&hessian, &rhs).unwrap();
            let norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.25 {
                for value in &mut step {
                    *value *= 0.25 / norm;
                }
            }
            c = self.rotate(&c, &step);
            assert!(
                iteration < 59,
                "independent complex common-HF optimizer residual={} field={field}",
                state.residual
            );
        }
        unreachable!()
    }
}
