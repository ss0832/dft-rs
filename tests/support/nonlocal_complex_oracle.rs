// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent occupied-orbital finite-London VV10 energy, with a direct pair sum.
use dft_rs::basis::{n_sph, MolBasis};
use dft_rs::grid::{eval::eval_aos, MolecularGrid};
use dft_rs::integrals::london::ComplexMatrix;
use dft_rs::nlc::Vv10Params;
use dft_rs::response::complex::Complex;
use dft_rs::Vec3;
use rayon::prelude::*;

pub fn energy(
    basis: &MolBasis,
    grid: &MolecularGrid,
    field: [f64; 3],
    coefficients: &[ComplexMatrix; 2],
    occupied: [usize; 2],
    parameters: Vv10Params,
) -> f64 {
    let n = basis.nao_sph;
    let b = Vec3::new(field[0], field[1], field[2]);
    let mut slope = vec![Vec3::zero(); n];
    for shell in &basis.shells {
        for i in 0..n_sph(shell.l) {
            slope[shell.sph_off + i] = b.cross(shell.center) * 0.5;
        }
    }
    let mut rho = vec![0.0; grid.len()];
    let mut sigma = rho.clone();
    for (start, points) in grid.points.chunks(256).enumerate() {
        let aos = eval_aos(basis, points, true);
        let g = aos.grad.as_ref().unwrap();
        for (local, &r) in points.iter().enumerate() {
            let mut gradient = [0.0; 3];
            for s in 0..2 {
                for i in 0..occupied[s] {
                    let mut psi = Complex::ZERO;
                    let mut grad = [Complex::ZERO; 3];
                    for (mu, sl) in slope.iter().enumerate() {
                        let theta = sl.dot(r);
                        let phase = Complex::new(theta.cos(), -theta.sin());
                        let c =
                            Complex::new(coefficients[s].re[(mu, i)], coefficients[s].im[(mu, i)]);
                        psi += c * phase * Complex::new(aos.phi[(local, mu)], 0.0);
                        for a in 0..3 {
                            grad[a] += c
                                * phase
                                * Complex::new(
                                    g[a][(local, mu)],
                                    -sl.get(a) * aos.phi[(local, mu)],
                                );
                        }
                    }
                    rho[start * 256 + local] += (psi.conj() * psi).re;
                    for a in 0..3 {
                        gradient[a] += 2.0 * (psi.conj() * grad[a]).re;
                    }
                }
            }
            sigma[start * 256 + local] = gradient.iter().map(|v| v * v).sum();
        }
    }
    // The cutoff is the stated vacuum convention of the physical model,
    // not a derivative-error floor.
    let active: Vec<_> = (0..grid.len()).filter(|&i| rho[i] >= 1e-8).collect();
    let omega: Vec<_> = active
        .iter()
        .map(|&i| {
            (parameters.c * sigma[i].powi(2) / rho[i].powi(4)
                + 4.0 * std::f64::consts::PI * rho[i] / 3.0)
                .sqrt()
        })
        .collect();
    let kappa: Vec<_> = active
        .iter()
        .map(|&i| {
            parameters.b
                * 1.5
                * std::f64::consts::PI
                * (rho[i] / (9.0 * std::f64::consts::PI)).powf(1.0 / 6.0)
        })
        .collect();
    let beta = (3.0 / parameters.b.powi(2)).powf(0.75) / 32.0;
    let rows: Vec<f64> = (0..active.len())
        .into_par_iter()
        .map(|a| {
            let i = active[a];
            let mut value = beta;
            for bb in 0..active.len() {
                let j = active[bb];
                let r2 = (grid.points[i] - grid.points[j]).norm2();
                let x = omega[a] * r2 + kappa[a];
                let y = omega[bb] * r2 + kappa[bb];
                value -= 0.75 * grid.weights[j] * rho[j] / (x * y * (x + y));
            }
            grid.weights[i] * rho[i] * value
        })
        .collect();
    rows.iter().sum()
}
