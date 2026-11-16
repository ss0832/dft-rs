// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Collinear meta-GGA descriptors assembled from occupied amplitudes.

use super::CollinearPoint;
use crate::periodic::{band_transport::hermitian_eigen, BlochMatrix};
use crate::response::complex::Complex;
use crate::Result;
use rayon::prelude::*;

fn entry(matrix: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(matrix.re[(i, j)], matrix.im[(i, j)])
}

fn square(value: Complex) -> f64 {
    value.re.mul_add(value.re, value.im * value.im)
}

fn factor(
    pauli: &[BlochMatrix; 4],
    axis: [f64; 3],
    sign: f64,
    sectors: usize,
) -> Result<Option<BlochMatrix>> {
    let mut spin_density = pauli[0].clone();
    for (component, axis_component) in pauli[1..].iter().zip(axis) {
        spin_density.add_scaled(component, sign * axis_component);
    }
    let (eigenvalues, eigenvectors) = hermitian_eigen(&spin_density)?;
    let scale = eigenvalues.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let roundoff = 64.0 * f64::EPSILON * scale;
    if eigenvalues.iter().any(|&v| v < -roundoff) {
        // Preserve the descriptor path for trial matrices outside the PSD cone.
        return Ok(None);
    }
    let occupied = eigenvalues.iter().filter(|&&value| value > 0.0).count();
    let mut result = BlochMatrix::zeros(spin_density.rows(), occupied);
    let mut column = 0;
    for (i, value) in eigenvalues.into_iter().enumerate() {
        if value > 0.0 {
            // The encoded matrix is twice the spin-channel density. Fold
            // half and the uniform k weight into each amplitude.
            let weight = (0.5 * value / sectors as f64).sqrt();
            for a in 0..result.rows() {
                result.re[(a, column)] = weight * eigenvectors.re[(a, i)];
                result.im[(a, column)] = weight * eigenvectors.im[(a, i)];
            }
            column += 1;
        }
    }
    Ok(Some(result))
}

pub(super) fn amplitude_points(
    orbitals: &[&BlochMatrix],
    momentum: &[&[BlochMatrix; 3]],
    density: &[[BlochMatrix; 4]],
    axis: [f64; 3],
) -> Result<Option<Vec<CollinearPoint>>> {
    let mut factors = Vec::with_capacity(density.len());
    for sector in density {
        let Some(up) = factor(sector, axis, 1.0, density.len())? else {
            return Ok(None);
        };
        let Some(down) = factor(sector, axis, -1.0, density.len())? else {
            return Ok(None);
        };
        factors.push([up, down]);
    }
    let mut capacity = [0; 2];
    for sector in &factors {
        for (size, spin) in capacity.iter_mut().zip(sector) {
            *size += spin.cols();
        }
    }
    let points = (0..orbitals[0].rows())
        .into_par_iter()
        .map_init(
            || capacity.map(Vec::with_capacity),
            |amplitudes, point| {
                for spin in amplitudes.iter_mut() {
                    spin.clear();
                }
                for (k, sector) in factors.iter().enumerate() {
                    for (buffer, coefficients) in amplitudes.iter_mut().zip(sector) {
                        for i in 0..coefficients.cols() {
                            let mut values = [Complex::ZERO; 4];
                            for a in 0..coefficients.rows() {
                                let c = entry(coefficients, a, i);
                                values[0] += c * entry(orbitals[k], point, a);
                                for d in 0..3 {
                                    values[d + 1] += c * entry(&momentum[k][d], point, a);
                                }
                            }
                            buffer.push(values);
                        }
                    }
                }
                invariants(amplitudes)
            },
        )
        .collect();
    Ok(Some(points))
}

fn invariants(amplitudes: &[Vec<[Complex; 4]>; 2]) -> CollinearPoint {
    let mut rho = [0.0; 2];
    let mut gradients = [[0.0; 3]; 2];
    let mut currents = [[0.0; 3]; 2];
    let mut tau = [0.0; 2];
    for (s, spin) in amplitudes.iter().enumerate() {
        for &[phi, px, py, pz] in spin {
            rho[s] += square(phi);
            for (d, p) in [px, py, pz].into_iter().enumerate() {
                let cross = p * phi.conj();
                gradients[s][d] -= 2.0 * cross.im;
                currents[s][d] += cross.re;
            }
        }
        // Complete the square at the orbital-amplitude level. Forming the
        // AO kinetic contraction and subtracting its flow separately loses
        // the small kinetic density around stationary occupied orbitals.
        let velocity = if rho[s] > 0.0 {
            currents[s].map(|j| j / rho[s])
        } else {
            [0.0; 3]
        };
        for &[phi, px, py, pz] in spin {
            for (d, p) in [px, py, pz].into_iter().enumerate() {
                tau[s] += 0.5 * square(p - phi * velocity[d]);
            }
        }
    }
    let g = |a: usize, b: usize| {
        gradients[a]
            .iter()
            .zip(gradients[b])
            .map(|(x, y)| x * y)
            .sum::<f64>()
    };
    let sigma = [g(0, 0), g(0, 1), g(1, 1)];
    for s in 0..2 {
        if amplitudes[s].len() == 1 && rho[s] > 0.0 {
            // For one amplitude, the residual is exactly the amplitude
            // gradient. Keep that identity when the local current dominates.
            tau[s] = sigma[2 * s] / (8.0 * rho[s]);
        }
    }
    CollinearPoint {
        invariants: [rho[0], rho[1], sigma[0], sigma[1], sigma[2], tau[0], tau[1]],
        gradients,
        currents,
    }
}
