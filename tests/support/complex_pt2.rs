// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent direct complex semicanonical PT2 energy oracle.
use dft_rs::integrals::london::ComplexMatrix;
use dft_rs::response::complex::Complex;

fn get(c: &ComplexMatrix, i: usize, j: usize) -> Complex {
    Complex::new(c.re[(i, j)], c.im[(i, j)])
}
// This oracle does not use the production packed four-index transform or
// amplitude Taylor recurrence. The finite SCF orbitals are semicanonical.
pub fn finite_channels(
    n: usize,
    g: &[Complex],
    c: &[ComplexMatrix; 2],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
) -> [f64; 3] {
    let amplitudes = finite_amplitudes(n, g, c, energies, occupied);
    std::array::from_fn(|channel| {
        let (s, t) = [(0, 0), (0, 1), (1, 1)][channel];
        let mut energy = 0.0;
        let (vs, vt) = (
            energies[s].len() - occupied[s],
            energies[t].len() - occupied[t],
        );
        for i in 0..occupied[s] {
            for j in 0..occupied[t] {
                for a in 0..vs {
                    for b in 0..vt {
                        let index = ((i * vs + a) * occupied[t] + j) * vt + b;
                        let amplitude = amplitudes[channel][index];
                        let delta = energies[s][i] + energies[t][j]
                            - energies[s][occupied[s] + a]
                            - energies[t][occupied[t] + b];
                        energy += (amplitude.conj() * amplitude).re
                            * delta
                            * if s == t { 0.25 } else { 1.0 };
                    }
                }
            }
        }
        energy
    })
}

/// Direct complex semicanonical amplitudes, independent of the production
/// packed transformation and noncanonical Taylor amplitude recurrence.
pub fn finite_amplitudes(
    n: usize,
    g: &[Complex],
    c: &[ComplexMatrix; 2],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
) -> [Vec<Complex>; 3] {
    let integral = |s: usize, t: usize, i: usize, a: usize, j: usize, b: usize| {
        let mut value = Complex::ZERO;
        for mu in 0..n {
            for nu in 0..n {
                for la in 0..n {
                    for si in 0..n {
                        value += get(&c[s], mu, a).conj()
                            * get(&c[s], nu, i)
                            * get(&c[t], la, b).conj()
                            * get(&c[t], si, j)
                            * g[((mu * n + nu) * n + la) * n + si];
                    }
                }
            }
        }
        value
    };
    std::array::from_fn(|channel| {
        let (s, t) = [(0, 0), (0, 1), (1, 1)][channel];
        let mut result = vec![
            Complex::ZERO;
            occupied[s]
                * (energies[s].len() - occupied[s])
                * occupied[t]
                * (energies[t].len() - occupied[t])
        ];
        for i in 0..occupied[s] {
            for j in 0..occupied[t] {
                for a in occupied[s]..energies[s].len() {
                    for b in occupied[t]..energies[t].len() {
                        let mut source = integral(s, t, i, a, j, b);
                        if s == t {
                            source -= integral(s, t, i, b, j, a);
                        }
                        let delta =
                            energies[s][i] + energies[t][j] - energies[s][a] - energies[t][b];
                        let index = ((i * (energies[s].len() - occupied[s]) + a - occupied[s])
                            * occupied[t]
                            + j)
                            * (energies[t].len() - occupied[t])
                            + b
                            - occupied[t];
                        result[index] = source / delta;
                    }
                }
            }
        }
        result
    })
}
