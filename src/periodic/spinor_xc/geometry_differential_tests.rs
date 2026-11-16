// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete input-component checks of the grid differential, not total forces.
use super::*;
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::xc::{resolve, FunctionalSpec};

#[derive(Clone)]
struct Data {
    weights: Vec<f64>,
    phi: BlochMatrix,
    momentum: [BlochMatrix; 3],
}
fn fixture() -> (Data, Vec<[BlochMatrix; 4]>) {
    let mut data = Data {
        weights: vec![0.61, 1.13],
        phi: BlochMatrix::zeros(2, 2),
        momentum: std::array::from_fn(|_| BlochMatrix::zeros(2, 2)),
    };
    for g in 0..2 {
        for a in 0..2 {
            let x = (1 + 2 * g + a) as f64;
            data.phi.re[(g, a)] = 0.4 + 0.09 * x;
            data.phi.im[(g, a)] = 0.08 * (0.7 * x).sin();
            for d in 0..3 {
                data.momentum[d].re[(g, a)] = 0.1 * (x + d as f64).cos();
                data.momentum[d].im[(g, a)] = -0.15 * (x * (d + 1) as f64).sin();
            }
        }
    }
    let vectors = [
        [
            Complex::new(0.7, 0.1),
            Complex::new(0.2, -0.3),
            Complex::new(-0.1, 0.4),
            Complex::new(0.3, 0.2),
        ],
        [
            Complex::new(0.1, -0.2),
            Complex::new(0.6, 0.1),
            Complex::new(0.4, -0.1),
            Complex::new(-0.2, 0.5),
        ],
    ];
    let mut p = BlochMatrix::zeros(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            let z = vectors
                .iter()
                .fold(Complex::ZERO, |s, v| s + v[i] * v[j].conj());
            p.re[(i, j)] = z.re;
            p.im[(i, j)] = z.im;
        }
    }
    (
        data,
        vec![crate::periodic::spinor_scf::pauli_components(&p).unwrap()],
    )
}
fn change(data: &mut Data, coordinate: usize, amount: f64) {
    if coordinate < 2 {
        data.weights[coordinate] += amount;
        return;
    }
    let index = coordinate - 2;
    let (matrix, entry) = if index < 8 {
        (&mut data.phi, index)
    } else {
        (&mut data.momentum[(index - 8) / 8], (index - 8) % 8)
    };
    let row = (entry / 2) / 2;
    let column = (entry / 2) % 2;
    if entry % 2 == 0 {
        matrix.re[(row, column)] += amount;
    } else {
        matrix.im[(row, column)] += amount;
    }
}
fn energy_only(functional: &ResolvedFunctional, data: &Data, density: &[[BlochMatrix; 4]]) -> f64 {
    super::spinor_semilocal_grid(
        functional,
        &data.weights,
        &[&data.phi],
        &[&data.momentum],
        density,
        8,
    )
    .unwrap()
    .0
}
#[test]
fn gks_xc_all_grid_input_components_have_independent_four_step_fits() {
    let (data, density) = fixture();
    let mut failures = Vec::new();
    let mut fitted = 0;
    let mut structural = 0;
    for name in ["svwn5", "pbe", "tpss", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let baseline = super::spinor_semilocal_grid(
            &functional,
            &data.weights,
            &[&data.phi],
            &[&data.momentum],
            &density,
            8,
        )
        .unwrap();
        for coordinate in 0..34 {
            let mut tangent = Data {
                weights: vec![0.; 2],
                phi: BlochMatrix::zeros(2, 2),
                momentum: std::array::from_fn(|_| BlochMatrix::zeros(2, 2)),
            };
            change(&mut tangent, coordinate, 1.);
            let (energy, analytic) = spinor_semilocal_grid_differential(
                &functional,
                &data.weights,
                &[&data.phi],
                &[&data.momentum],
                &density,
                8,
                SpinorXcGridTangent {
                    weights: &tangent.weights,
                    orbitals: &[&tangent.phi],
                    momentum: &[&tangent.momentum],
                },
            )
            .unwrap();
            assert_eq!(energy.to_bits(), baseline.0.to_bits());
            let collect = |steps: &[f64; 4]| -> [f64; 4] {
                std::array::from_fn(|j| {
                    let h = steps[j];
                    let mut plus = data.clone();
                    let mut minus = data.clone();
                    change(&mut plus, coordinate, h.sin());
                    change(&mut minus, coordinate, -h.sin());
                    let ep = energy_only(&functional, &plus, &density);
                    let em = energy_only(&functional, &minus, &density);
                    let reference = super::stencils::Stencil::Second.first(ep, em, 0., 0., h);
                    let error = (reference - analytic).abs();
                    println!("GKS_XC_GRID_SIGNED functional={name} coordinate={coordinate} h={h:.17e} analytic={analytic:.17e} reference={reference:.17e} error={error:.17e}");
                    if functional.family == Family::Lda && coordinate >= 10 {
                        assert_eq!(analytic, 0.);
                        assert_eq!(ep.to_bits(), em.to_bits());
                    }
                    error
                })
            };
            let original = [0.04, 0.02, 0.01, 0.005];
            if functional.family == Family::Lda && coordinate >= 10 {
                collect(&original);
                structural += 1;
            } else {
                // One common finer interval is declared before the run. The
                // original failed rows stay printed; only unresolved inputs
                // evaluate this interval through the canonical staged helper.
                super::stencils::fit_components_staged(
                    &[format!("GKS XC {name} grid input[{coordinate}]")],
                    &[
                        (original, super::stencils::Stencil::Second),
                        (
                            [0.01, 0.005, 0.0025, 0.00125],
                            super::stencils::Stencil::Second,
                        ),
                    ],
                    |steps, _| vec![collect(steps)],
                    &mut failures,
                );
                fitted += 1;
            }
        }
    }
    assert_eq!(fitted, 112);
    assert_eq!(structural, 24);
    super::components::assert_no_failures("all active XC grid differentials", &failures);
}
#[test]
fn gks_xc_inactive_nonfinite_tangent_is_rejected() {
    let (data, density) = fixture();
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let mut momentum = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    momentum[1].im[(1, 1)] = f64::NAN;
    let zeros = BlochMatrix::zeros(2, 2);
    assert!(matches!(
        spinor_semilocal_grid_differential(
            &functional,
            &data.weights,
            &[&data.phi],
            &[&data.momentum],
            &density,
            8,
            SpinorXcGridTangent {
                weights: &[0., 0.],
                orbitals: &[&zeros],
                momentum: &[&momentum]
            }
        ),
        Err(DftError::InvalidInput(_))
    ));
}
