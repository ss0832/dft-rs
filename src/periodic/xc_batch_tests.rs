// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::AoDerivativeOrder;
use crate::acceptance::{components, stencils};
use crate::basis::{g94, MolBasis};
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::{Cell, Lattice};
use crate::{Atom, Molecule, Vec3};

fn compare(a: &BlochMatrix, b: &BlochMatrix) {
    assert_eq!(a.re.rows, b.re.rows);
    assert_eq!(a.re.cols, b.re.cols);
    for (x, y) in
        a.re.as_slice()
            .iter()
            .chain(a.im.as_slice())
            .zip(b.re.as_slice().iter().chain(b.im.as_slice()))
    {
        assert!(x.is_finite() && y.is_finite());
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "exact per-point order changed: {x} vs {y}"
        );
    }
}
fn slice_compare(full: &BlochMatrix, local: &BlochMatrix, start: usize) {
    for (a, b) in [(&full.re, &local.re), (&full.im, &local.im)] {
        for i in 0..b.rows {
            for j in 0..b.cols {
                assert_eq!(a[(start + i, j)].to_bits(), b[(i, j)].to_bits());
            }
        }
    }
}
fn scalar(a: &BlochMatrix, p: usize, mu: usize, imag: bool) -> f64 {
    if imag {
        a.im[(p, mu)]
    } else {
        a.re[(p, mu)]
    }
}
fn fixture() -> Cell {
    let set=g94::parse("C1 AO batching validation","H 0\nS 2 1.0\n1.3 0.6\n0.6 0.5\nP 2 1.0\n1.3 0.6\n0.6 0.5\nD 2 1.0\n1.3 0.6\n0.6 0.5\n****\nHe 0\nS 2 1.0\n1.6 0.6\n0.8 0.5\nP 2 1.0\n1.6 0.6\n0.8 0.5\nD 2 1.0\n1.6 0.6\n0.8 0.5\n****\n").unwrap();
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.13, -0.27, 0.31),
        },
        Atom {
            z: 2,
            position: Vec3::new(1.21, 0.19, -0.23),
        },
    ]);
    let mb = MolBasis::build(&molecule, &set).unwrap();
    assert_eq!(mb.nao_sph, 18);
    Cell::new(
        molecule,
        Lattice::new(
            Vec3::new(4.1, 0.31, 0.13),
            Vec3::new(0.2, 4.8, 0.43),
            Vec3::new(0.11, 0.27, 5.3),
        ),
        &set,
        8.0,
    )
    .unwrap()
}

#[test]
fn streamed_force_hessians_retain_every_global_gradient_entry() {
    let cell = fixture();
    let reciprocal = cell.lattice().reciprocal();
    let ks = [
        Vec3::zero(),
        reciprocal[0] * 0.23 + reciprocal[1] * 0.31,
        reciprocal[1] * 0.17 + reciprocal[2] * 0.29,
    ];
    let points: Vec<_> = (0..137)
        .map(|i| {
            Vec3::new(
                (i * 17 % 113) as f64 * 0.073 - 3.1,
                (i * 23 % 127) as f64 * 0.061 - 2.7,
                (i * 31 % 131) as f64 * 0.059 - 2.3,
            )
        })
        .collect();
    let global =
        super::build_bloch_aos_on_points(&cell, &ks, &points, 4.5, AoDerivativeOrder::Gradient);
    for (batch, points) in points.chunks(64).enumerate() {
        let local =
            super::build_bloch_aos_on_points(&cell, &ks, points, 4.5, AoDerivativeOrder::Hessian);
        for (global, local) in global.iter().zip(&local) {
            slice_compare(&global.phi, &local.phi, 64 * batch);
            for axis in 0..3 {
                slice_compare(
                    &global.grad.as_ref().unwrap()[axis],
                    &local.grad.as_ref().unwrap()[axis],
                    64 * batch,
                );
            }
        }
    }
    println!("All SPD Gamma/generic-k global gradients and streamed Hessian lower orders are bitwise equal");
}

#[test]
fn all_point_batches_and_spatial_derivatives() {
    let cell = fixture();
    let reciprocal = cell.lattice().reciprocal();
    let ks = [
        Vec3::zero(),
        reciprocal[0] * 0.23 + reciprocal[1] * 0.31,
        reciprocal[1] * 0.17 + reciprocal[2] * 0.29,
    ];
    let mut points: Vec<Vec3> = (0..137)
        .map(|i| {
            Vec3::new(
                (i * 17 % 113) as f64 * 0.073 - 3.1,
                (i * 23 % 127) as f64 * 0.061 - 2.7,
                (i * 31 % 131) as f64 * 0.059 - 2.3,
            )
        })
        .collect();
    points[130] = Vec3::new(0.52317, 0.41819, 0.61837);
    assert_eq!(
        AoDerivativeOrder::for_scf(crate::xc::Family::Gga),
        AoDerivativeOrder::Gradient
    );
    assert_eq!(
        AoDerivativeOrder::for_forces(crate::xc::Family::MetaGga),
        AoDerivativeOrder::Hessian
    );
    for length in [0, 1, 64, 65, 137] {
        for order in [
            AoDerivativeOrder::Value,
            AoDerivativeOrder::Gradient,
            AoDerivativeOrder::Hessian,
        ] {
            let old = legacy_bloch_aos_on_points(&cell, &ks, &points[..length], 4.5, order);
            let new = super::build_bloch_aos_on_points(&cell, &ks, &points[..length], 4.5, order);
            for (a, b) in old.iter().zip(&new) {
                compare(&a.phi, &b.phi);
                if let (Some(x), Some(y)) = (&a.grad, &b.grad) {
                    for c in 0..3 {
                        compare(&x[c], &y[c]);
                    }
                }
                if let (Some(x), Some(y)) = (&a.hess, &b.hess) {
                    for c in 0..6 {
                        compare(&x[c], &y[c]);
                    }
                }
            }
            println!("all AO entries bitwise equal: points={length} order={order:?}");
        }
    }
    for hessian in [false, true] {
        let old = super::bloch_ao_moments(&cell, &ks, &points, 4.5, hessian);
        for (chunk, p) in points.chunks(64).enumerate() {
            let new = super::bloch_ao_moments(&cell, &ks, p, 4.5, hessian);
            for (a, b) in old.iter().zip(&new) {
                for c in 0..3 {
                    for d in 0..3 {
                        slice_compare(&a.grad_u[c][d], &b.grad_u[c][d], 64 * chunk);
                    }
                }
                if let (Some(x), Some(y)) = (&a.hess_u, &b.hess_u) {
                    for c in 0..6 {
                        for d in 0..3 {
                            slice_compare(&x[c][d], &y[c][d], 64 * chunk);
                        }
                    }
                }
            }
        }
        println!("all strain moment entries bitwise equal: Hessian={hessian}");
    }
    let analytic =
        super::build_bloch_aos_on_points(&cell, &ks, &points, 8.0, AoDerivativeOrder::Hessian);
    let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let mut names = Vec::new();
    for k in 1..3 {
        for mu in 0..cell.nao() {
            for imag in [false, true] {
                for d in 0..9 {
                    names.push(format!("AO k{k} mu{mu} imag={imag} derivative{d}"));
                }
            }
        }
    }
    let at = |delta: [f64; 3]| {
        let p = points[130] + Vec3::new(delta[0], delta[1], delta[2]);
        legacy_bloch_aos_on_points(&cell, &ks, &[p], 8.0, AoDerivativeOrder::Value)
    };
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (step, &h) in steps.iter().enumerate() {
                let centre = at([0.0; 3]);
                let plus: Vec<_> = (0..3)
                    .map(|a| {
                        let mut d = [0.0; 3];
                        d[a] = h;
                        at(d)
                    })
                    .collect();
                let minus: Vec<_> = (0..3)
                    .map(|a| {
                        let mut d = [0.0; 3];
                        d[a] = -h;
                        at(d)
                    })
                    .collect();
                let mixed: Vec<_> = pairs
                    .iter()
                    .map(|&(a, b)| {
                        let mut evaluations = Vec::new();
                        if a != b {
                            for sign_a in [1.0, -1.0] {
                                for sign_b in [1.0, -1.0] {
                                    let mut d = [0.0; 3];
                                    d[a] = sign_a * h;
                                    d[b] = sign_b * h;
                                    evaluations.push(at(d));
                                }
                            }
                        }
                        evaluations
                    })
                    .collect();
                let mut entry = 0;
                for k in 1..3 {
                    for mu in 0..cell.nao() {
                        for imag in [false, true] {
                            for axis in 0..3 {
                                let fd = stencil.first(
                                    scalar(&plus[axis][k].phi, 0, mu, imag),
                                    scalar(&minus[axis][k].phi, 0, mu, imag),
                                    0.0,
                                    0.0,
                                    h,
                                );
                                errors[entry][step] = (fd
                                    - scalar(
                                        &analytic[k].grad.as_ref().unwrap()[axis],
                                        130,
                                        mu,
                                        imag,
                                    ))
                                .abs();
                                entry += 1;
                            }
                            for (component, &(a, b)) in pairs.iter().enumerate() {
                                let fd = if a == b {
                                    (scalar(&plus[a][k].phi, 0, mu, imag)
                                        - 2.0 * scalar(&centre[k].phi, 0, mu, imag)
                                        + scalar(&minus[a][k].phi, 0, mu, imag))
                                        / (h * h)
                                } else {
                                    (scalar(&mixed[component][0][k].phi, 0, mu, imag)
                                        - scalar(&mixed[component][1][k].phi, 0, mu, imag)
                                        - scalar(&mixed[component][2][k].phi, 0, mu, imag)
                                        + scalar(&mixed[component][3][k].phi, 0, mu, imag))
                                        / (4.0 * h * h)
                                };
                                errors[entry][step] = (fd
                                    - scalar(
                                        &analytic[k].hess.as_ref().unwrap()[component],
                                        130,
                                        mu,
                                        imag,
                                    ))
                                .abs();
                                entry += 1;
                            }
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all Bloch AO gradients and Hessians", &failures);
    let h = 0.004;
    let mut worst = 0.0_f64;
    for axis in 0..3 {
        let mut d = [0.0; 3];
        d[axis] = h;
        let plus = at(d);
        d[axis] = -h;
        let minus = at(d);
        d[axis] = 2.0 * h;
        let plus2 = at(d);
        d[axis] = -2.0 * h;
        let minus2 = at(d);
        for k in 1..3 {
            for mu in 0..cell.nao() {
                for imag in [false, true] {
                    let fd = stencils::Stencil::Fourth.first(
                        scalar(&plus[k].phi, 0, mu, imag),
                        scalar(&minus[k].phi, 0, mu, imag),
                        scalar(&plus2[k].phi, 0, mu, imag),
                        scalar(&minus2[k].phi, 0, mu, imag),
                        h,
                    );
                    worst = worst.max(
                        (fd - scalar(&analytic[k].grad.as_ref().unwrap()[axis], 130, mu, imag))
                            .abs(),
                    );
                }
            }
        }
    }
    assert!(worst < 2e-8);
    println!("all fourth-order spatial-gradient cross-checks maximum error={worst:.16e}");
}

fn legacy_bloch_aos_on_points(
    cell: &Cell,
    k_points: &[Vec3],
    points: &[Vec3],
    ao_cutoff: f64,
    derivative_order: AoDerivativeOrder,
) -> Vec<super::BlochAos> {
    let images = super::xc_ao_images(&cell.basis, &cell.lattice, points, ao_cutoff);
    super::build_bloch_aos_from_images(&cell.basis, k_points, points, &images, derivative_order)
}
