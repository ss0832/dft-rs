// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent closed Gaussian scalar/current oracles on every periodic mask.
use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
use dft_rs::periodic::integrals::{bloch, dimensional_image_blocks, Operator};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::{complex::Complex, Frequency};
use dft_rs::system::{Atom, Molecule};
fn value(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn fixture(
    mask: usize,
    charge: i32,
) -> (Molecule, MolBasis, PeriodicGeometry, PeriodicFieldOptions) {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.2, -0.1, 0.3),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.6, 0.5, -0.4),
        },
    ])
    .with_charge(charge);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let axes = std::array::from_fn(|i| mask & (1 << i) != 0);
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.2, 0.5, 0.3),
            Vec3::new(0.8, 5.6, 0.4),
            Vec3::new(0.3, 0.2, 6.0),
        ),
        PeriodicBoundary::new(axes),
    )
    .unwrap();
    let options = PeriodicFieldOptions {
        k_mesh: std::array::from_fn(|i| if axes[i] { 2 } else { 1 }),
        wave: Vec3::new(0.137, -0.084, 0.067),
        k_offset: Vec3::new(0.041, -0.023, 0.027),
        frequency: Frequency::Retarded {
            omega: 0.23,
            damping: 0.01,
        },
        scalar_potential: Complex::new(0.03, -0.02),
        vector_potential: [
            Complex::new(0.05, 0.03),
            Complex::new(-0.02, 0.01),
            Complex::new(0.01, -0.04),
        ],
        ..Default::default()
    };
    (mol, mb, geometry, options)
}
fn oracle(
    mb: &MolBasis,
    g: &PeriodicGeometry,
    k: Vec3,
    wave: Vec3,
    cutoff: f64,
) -> [BlochMatrix; 4] {
    let mut out = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    let mask = g.boundary().axes;
    for i in 0..2 {
        for j in 0..2 {
            for a in -4..=4 {
                for b in -4..=4 {
                    for c in -4..=4 {
                        if (!mask[0] && a != 0) || (!mask[1] && b != 0) || (!mask[2] && c != 0) {
                            continue;
                        }
                        let shift = g.lattice().translation([a, b, c]);
                        let left = &mb.shells[i];
                        let right = &mb.shells[j];
                        let centre = right.center + shift;
                        let d = left.center - centre;
                        if d.norm() > cutoff {
                            continue;
                        }
                        let phase = Complex::new(k.dot(shift).cos(), k.dot(shift).sin());
                        for (&alpha, &ca) in left.exps.iter().zip(&left.coefs) {
                            for (&beta, &cb) in right.exps.iter().zip(&right.coefs) {
                                let p = alpha + beta;
                                let product = (left.center * alpha + centre * beta) / p;
                                let norm = ca
                                    * cb
                                    * (std::f64::consts::PI / p).powf(1.5)
                                    * (-alpha * beta / p * d.norm2() - wave.norm2() / (4.0 * p))
                                        .exp();
                                let theta = wave.dot(product);
                                let n =
                                    Complex::new(theta.cos() * norm, theta.sin() * norm) * phase;
                                out[0].re[(i, j)] += n.re;
                                out[0].im[(i, j)] += n.im;
                                for axis in 0..3 {
                                    let coeff = Complex::new(
                                        (0.5 - beta / p) * wave.get(axis),
                                        2.0 * beta * (product - centre).get(axis),
                                    );
                                    let v = coeff * n;
                                    out[1 + axis].re[(i, j)] += v.re;
                                    out[1 + axis].im[(i, j)] += v.im;
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

#[test]
fn every_mask_and_charge_matches_independent_scalar_current_contact_oracles() {
    let mut worst = 0.0_f64;
    for mask in 1..8 {
        for charge in [-1, 0, 1] {
            let (mol, mb, g, options) = fixture(mask, charge);
            let report = periodic_field_operators(&mol, &mb, &g, &options).unwrap();
            assert_eq!(report.formal_charge, charge);
            for (k, &kv) in report.k_points.iter().enumerate() {
                let independent = oracle(&mb, &g, kv, options.wave, options.basis_cutoff);
                let contact = oracle(&mb, &g, kv, options.wave * 2.0, options.basis_cutoff);
                for i in 0..2 {
                    for j in 0..2 {
                        for axis in 0..4 {
                            let actual = if axis == 0 {
                                &report.density_vertex[k]
                            } else {
                                &report.current_vertex[axis - 1][k]
                            };
                            worst = worst
                                .max((value(actual, i, j) - value(&independent[axis], i, j)).abs());
                        }
                        worst = worst.max(
                            (value(&report.contact_vertex[k], i, j) - value(&contact[0], i, j))
                                .abs(),
                        );
                        let first = value(&independent[0], i, j) * options.scalar_potential
                            + (0..3)
                                .map(|a| {
                                    value(&independent[a + 1], i, j) * options.vector_potential[a]
                                })
                                .fold(Complex::ZERO, |a, b| a + b);
                        assert!((value(&report.first_order[k], i, j) - first).abs() < 1e-12);
                    }
                }
            }
            assert!(report.transversality_defect < 1e-17);
        }
    }
    println!(
        "dimensional field scalar/current/contact closed Gaussian maximum defect {worst:.12e}"
    );
    assert!(worst < 1e-12);
}
#[test]
fn shifted_q_conjugation_zero_q_overlap_and_electromagnetic_gauge_identity() {
    let (mol, mb, g, o) = fixture(7, 1);
    let plus = periodic_field_operators(&mol, &mb, &g, &o).unwrap();
    let reverse = periodic_field_operators(
        &mol,
        &mb,
        &g,
        &PeriodicFieldOptions {
            wave: o.wave * (-1.0),
            k_offset: o.k_offset + o.wave,
            ..o.clone()
        },
    )
    .unwrap();
    for k in 0..plus.k_points.len() {
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (value(&plus.density_vertex[k], i, j)
                        - value(&reverse.density_vertex[k], j, i).conj())
                    .abs()
                        < 1e-12
                );
                for a in 0..3 {
                    assert!(
                        (value(&plus.current_vertex[a][k], i, j)
                            - value(&reverse.current_vertex[a][k], j, i).conj())
                        .abs()
                            < 1e-12
                    );
                }
            }
        }
    }
    assert!(
        (value(&plus.density_vertex[0], 0, 1) - value(&plus.density_vertex[0], 1, 0).conj()).abs()
            > 1e-3
    );
    let zero = periodic_field_operators(
        &mol,
        &mb,
        &g,
        &PeriodicFieldOptions {
            wave: Vec3::zero(),
            ..o.clone()
        },
    )
    .unwrap();
    let images = dimensional_image_blocks(&mb, &g, o.basis_cutoff, Operator::Overlap).unwrap();
    for (k, &kv) in zero.k_points.iter().enumerate() {
        let (re, im) = bloch(&images, kv);
        assert!(zero.density_vertex[k].re.max_abs_diff(&re) < 1e-12);
        assert!(zero.density_vertex[k].im.max_abs_diff(&im) < 1e-12);
    }
    // A -> A+i*q*Lambda, V -> V-i*z*Lambda leaves both E and B invariant.
    let z = o.frequency.energy().unwrap();
    let iu = Complex::new(0.0, 1.0);
    let lambda = Complex::new(0.02, -0.03);
    let gauged = periodic_field_operators(
        &mol,
        &mb,
        &g,
        &PeriodicFieldOptions {
            scalar_potential: o.scalar_potential - iu * z * lambda,
            vector_potential: std::array::from_fn(|a| {
                o.vector_potential[a] + iu * lambda * o.wave.get(a)
            }),
            ..o
        },
    )
    .unwrap();
    for a in 0..3 {
        assert!((plus.electric_field[a] - gauged.electric_field[a]).abs() < 1e-17);
        assert!((plus.magnetic_field[a] - gauged.magnetic_field[a]).abs() < 1e-17);
    }
}
