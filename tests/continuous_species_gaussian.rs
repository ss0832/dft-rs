// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent polynomial-moment oracle for the local Gaussian kernel.
//! The isolated kernel is checked before registration in the model driver.
pub use dft_rs::{basis, integrals, linalg, math, DftError, Result};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "../src/driver/continuous_species/gaussian.rs"]
mod gaussian;

use basis::{aos, cart_components, data, MolBasis};
use dft_rs::system::{Atom, Molecule};
use gaussian::gaussian_potential;
use linalg::Matrix;
use math::Vec3;
use std::f64::consts::PI;

fn molecule() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.13, -0.21, 0.17),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.34, 0.82, -0.23),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.74, 1.13, 0.91),
        },
    ])
}

fn basis(molecule: &Molecule) -> MolBasis {
    MolBasis::build(molecule, &data::load("sto-3g").unwrap()).unwrap()
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |v, i| v * (n - i) as f64 / (i + 1) as f64)
}

// Expand both Cartesian polynomials about the final three-Gaussian product
// center. Even central moments have a closed form; no E tables are used.
fn moment(la: usize, lb: usize, ta: f64, tb: f64, exponent: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..=la {
        for j in 0..=lb {
            let power = i + j;
            if power % 2 != 0 {
                continue;
            }
            let central = (0..power / 2).fold((PI / exponent).sqrt(), |v, k| {
                v * (2 * k + 1) as f64 / (2.0 * exponent)
            });
            sum += binomial(la, i)
                * binomial(lb, j)
                * ta.powi((la - i) as i32)
                * tb.powi((lb - j) as i32)
                * central;
        }
    }
    sum
}

fn oracle(basis: &MolBasis, center: Vec3, exponent: f64) -> Matrix {
    let mut matrix = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    for a in &basis.shells {
        for b in &basis.shells {
            let ca = cart_components(a.l);
            let cb = cart_components(b.l);
            let mut block = Matrix::zeros(ca.len(), cb.len());
            for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                    let p = ea + eb;
                    let product = (a.center * ea + b.center * eb) * (1.0 / p);
                    let final_center = (product * p + center * exponent) * (1.0 / (p + exponent));
                    let prefactor = (-ea * eb / p * (a.center - b.center).norm2()
                        - p * exponent / (p + exponent) * (product - center).norm2())
                    .exp();
                    for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                        for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                            let la = [ax, ay, az];
                            let lb = [bx, by, bz];
                            let value = (0..3).fold(wa * wb * prefactor, |v, axis| {
                                v * moment(
                                    la[axis] as usize,
                                    lb[axis] as usize,
                                    final_center.get(axis) - a.center.get(axis),
                                    final_center.get(axis) - b.center.get(axis),
                                    p + exponent,
                                )
                            });
                            block[(i, j)] += value;
                        }
                    }
                }
            }
            let sph = aos::sph_block(a.l, b.l, &block);
            for i in 0..sph.rows {
                for j in 0..sph.cols {
                    matrix[(a.sph_off + i, b.sph_off + j)] = sph[(i, j)];
                }
            }
        }
    }
    matrix
}

#[test]
fn independent_energy_oracle_symmetry_translation_and_limits() {
    let mol = molecule();
    let basis = basis(&mol);
    let center = Vec3::new(0.37, -0.29, 0.83);
    let value = gaussian_potential(&basis, mol.len(), center, 0.63).unwrap();
    assert!(value.operator.max_abs_diff(&oracle(&basis, center, 0.63)) < 1e-12);
    for i in 0..basis.nao_sph {
        for j in 0..basis.nao_sph {
            assert!((value.operator[(i, j)] - value.operator[(j, i)]).abs() < 1e-14);
            for axis in 0..3 {
                let sum = value.center_derivatives[axis][(i, j)]
                    + value
                        .atom_derivatives
                        .iter()
                        .map(|g| g[axis][(i, j)])
                        .sum::<f64>();
                assert!(sum.abs() < 1e-12);
            }
        }
    }
    assert!(gaussian_potential(&basis, 3, center, 0.0).is_err());
    assert!(gaussian_potential(&basis, 3, center, f64::NAN).is_err());
    assert!(gaussian_potential(&basis, 1, center, 0.63).is_err());
    let broad = gaussian_potential(&basis, 3, center, 1e-12).unwrap();
    assert!(
        broad
            .operator
            .max_abs_diff(&integrals::overlap_matrix(&basis))
            < 1e-10
    );
}

#[test]
fn all_independent_center_and_moving_ao_components_four_step_fits() {
    let mol = molecule();
    let reference_basis = basis(&mol);
    let center = Vec3::new(0.37, -0.29, 0.83);
    let value = gaussian_potential(&reference_basis, mol.len(), center, 0.63).unwrap();
    let mut owners = vec![0; reference_basis.nao_sph];
    for shell in &reference_basis.shells {
        owners[shell.sph_off..shell.sph_off + 2 * shell.l as usize + 1].fill(shell.atom);
    }
    let mut failures = Vec::new();
    for site in 0..=mol.len() {
        for axis in 0..3 {
            let pairs: Vec<_> = (0..reference_basis.nao_sph)
                .flat_map(|i| (i..reference_basis.nao_sph).map(move |j| (i, j)))
                .filter(|&(i, j)| site == mol.len() || owners[i] == site || owners[j] == site)
                .collect();
            let names: Vec<_> = pairs
                .iter()
                .map(|&(i, j)| format!("Gaussian center[{site},{axis}] matrix[{i},{j}]"))
                .collect();
            let exact = if site == mol.len() {
                &value.center_derivatives[axis]
            } else {
                &value.atom_derivatives[site][axis]
            };
            components::fit_components(
                &names,
                &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    let mut errors = vec![[0.0; 4]; pairs.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let displaced = |offset: f64| {
                            let mut moved = mol.clone();
                            let mut point = center;
                            if site == mol.len() {
                                let mut xyz = point.to_array();
                                xyz[axis] += offset;
                                point = Vec3::new(xyz[0], xyz[1], xyz[2]);
                            } else {
                                let mut xyz = moved.atoms[site].position.to_array();
                                xyz[axis] += offset;
                                moved.atoms[site].position = Vec3::new(xyz[0], xyz[1], xyz[2]);
                            }
                            // Independent energy kernel at every displaced center.
                            oracle(&basis(&moved), point, 0.63)
                        };
                        let positive = displaced(h);
                        let negative = displaced(-h);
                        for (c, &(i, j)) in pairs.iter().enumerate() {
                            errors[c][k] = ((positive[(i, j)] - negative[(i, j)]) / (2.0 * h)
                                - exact[(i, j)])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("Gaussian kernel every independent component", &failures);
}
