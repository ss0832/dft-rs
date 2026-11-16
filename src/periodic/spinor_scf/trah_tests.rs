// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All real/imaginary orbital curvatures of a generic C1 complex determinant.
use super::*;
use crate::acceptance::{c1_geometry, components, stencils};
use crate::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use crate::driver::{Controls, System, Value};
use crate::periodic::spinor_scf::{spin_diagonal, SpinorEnergyBackend};
use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};

#[test]
fn trah_every_complex_orbital_curvature_matches_independent_rotations() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    );
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("dispersion", "none")
            .with("grid_level", 1usize)
            .with("grid_weight_cutoff", 0.0)
            .with("scf_optimizer", "legacy"),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap().scf;
    // HF needs no SCF quadrature, but the target GGA Hessian does.
    let grid = crate::grid::MolecularGrid::build(
        &setup.molecule,
        &crate::grid::GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let c = spin_diagonal(&BlochMatrix {
        re: seed.mo_coeff[0].clone(),
        im: Matrix::zeros(4, 4),
    });
    let mut frame = BlochMatrix::zeros(8, 8);
    let order = [0, 1, 4, 5, 2, 3, 6, 7];
    for i in 0..8 {
        for j in 0..8 {
            put(&mut frame, i, j, z(&c, i, order[j]));
        }
    }
    let mut k = BlochMatrix::zeros(8, 8);
    for i in 0..8 {
        for j in i + 1..8 {
            let v = Complex::new(
                0.08 * ((3 * i + 7 * j + 1) as f64).sin(),
                0.05 * ((5 * i + 2 * j + 1) as f64).cos(),
            );
            put(&mut k, i, j, v);
            put(&mut k, j, i, -v.conj());
        }
    }
    let mut a = k.clone();
    a.scale(-0.5);
    let mut b = k.clone();
    b.scale(0.5);
    for i in 0..8 {
        a.re[(i, i)] += 1.0;
        b.re[(i, i)] += 1.0;
    }
    frame = matmul(&frame, &solve(&a, &b).unwrap()).unwrap();
    let occupations = vec![vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]];
    let mut p = BlochMatrix::zeros(8, 8);
    for i in 0..4 {
        p.re[(i, i)] = 1.0;
    }
    let density = vec![matmul(&matmul(&frame, &p).unwrap(), &adjoint(&frame).unwrap()).unwrap()];
    let mut failures = Vec::new();
    for xc in ["hf", "pbe"] {
        let functional =
            crate::xc::resolve(&crate::xc::FunctionalSpec::Combined(xc.into())).unwrap();
        let backend = MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            &functional,
            Some(&grid),
            &seed,
            &setup.options,
            BlochMatrix::zeros(8, 8),
            [0.0, 0.0, 1.0],
            MolecularSpinorFunctionalModel::SpinCovariant,
            1e-10,
        )
        .unwrap();
        let optimizer = AugmentedHessianMinimizer::new(vec![frame.clone()], &occupations).unwrap();
        let fock = backend.evaluate(&density).unwrap().fock;
        let mo = vec![matmul(
            &adjoint(&frame).unwrap(),
            &matmul(&fock[0], &frame).unwrap(),
        )
        .unwrap()];
        let kernel = backend.orbital_kernel(&density).unwrap();
        let length = 32;
        let mut exact = Matrix::zeros(length, length);
        for col in 0..length {
            let mut e = vec![0.0; length];
            e[col] = 1.0;
            let image = optimizer
                .curvature(&e, std::slice::from_ref(&p), &mo, kernel.as_ref(), 1.0)
                .unwrap();
            for row in 0..length {
                exact[(row, col)] = image[row];
            }
        }
        // A same-pair real/imaginary HF cross curvature is structurally zero.
        // Axis-only rotations then have no truncation error to fit. Probe a
        // complete orthogonal set of generic directions and invert that map
        // at EACH step size, fitting every ORIGINAL Hessian entry afterwards.
        // This is a full matrix reconstruction, not a contracted tensor test.
        let v = (0..length)
            .map(|i| ((3 * i + 1) as f64).sin() + 0.31)
            .collect::<Vec<_>>();
        let norm: f64 = v.iter().map(|x| x * x).sum();
        let q = Matrix::from_row_major(
            length,
            length,
            (0..length)
                .flat_map(|i| {
                    let v = &v;
                    (0..length).map(move |j| f64::from(i == j) - 2.0 * v[i] * v[j] / norm)
                })
                .collect(),
        );
        assert!(
            q.transpose_matmul_seq(&q)
                .max_abs_diff(&Matrix::identity(length))
                < 1e-14
        );
        let at = |column: usize, t: f64| {
            let mut c = frame.clone();
            // Independent closed single-plane exponentials in a palindromic
            // product give U(-t)=U(t)^dagger and generator sum q_j K_j.
            // The production optimizer instead uses a single Cayley solve.
            for k in (0..length).chain((0..length).rev()) {
                let angle = 0.5 * t * q[(k, column)];
                let i = (k / 2) % 4;
                let a = 4 + (k / 2) / 4;
                let value = if k % 2 == 0 {
                    Complex::new(angle.sin(), 0.0)
                } else {
                    Complex::new(0.0, angle.sin())
                };
                let mut u = BlochMatrix::zeros(8, 8);
                for j in 0..8 {
                    u.re[(j, j)] = 1.0;
                }
                u.re[(i, i)] = angle.cos();
                u.re[(a, a)] = angle.cos();
                put(&mut u, a, i, value);
                put(&mut u, i, a, -value.conj());
                c = matmul(&c, &u).unwrap();
            }
            let density = vec![matmul(&matmul(&c, &p).unwrap(), &adjoint(&c).unwrap()).unwrap()];
            let fock = backend.evaluate(&density).unwrap().fock;
            AugmentedHessianMinimizer::new(vec![c], &occupations)
                .unwrap()
                .project(&fock, 1.0)
                .unwrap()
        };
        let names = (0..length)
            .flat_map(|row| {
                (0..length)
                    .map(move |col| format!("TRAH complex {xc} H[{row},{col}] reconstructed"))
            })
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; length * length];
                for (k, &h) in steps.iter().enumerate() {
                    let mut columns = Matrix::zeros(length, length);
                    for col in 0..length {
                        let plus = at(col, h);
                        let minus = at(col, -h);
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (at(col, 2.0 * h), at(col, -2.0 * h))
                        } else {
                            (vec![0.0; length], vec![0.0; length])
                        };
                        for row in 0..length {
                            columns[(row, col)] =
                                stencil.first(plus[row], minus[row], p2[row], m2[row], h);
                        }
                    }
                    let reconstructed = columns.matmul_transpose_seq(&q);
                    for row in 0..length {
                        for col in 0..length {
                            errors[row * length + col][k] =
                                (reconstructed[(row, col)] - exact[(row, col)]).abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("all complex TRAH curvature entries", &failures);
}
