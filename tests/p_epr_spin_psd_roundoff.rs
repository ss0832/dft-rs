// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Floating-point PSD-boundary controls, without a new density cutoff.

use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{Controls, Value};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::spinor_scf::pauli_components;
use dft_rs::periodic::spinor_xc::{spinor_semilocal_grid, spinor_semilocal_point, SpinorXcPoint};
use dft_rs::periodic::BlochMatrix;
use dft_rs::xc::{resolve, Family, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};

fn scalar_reference(
    functional: &ResolvedFunctional,
    phi: &Matrix,
    gradients: &[Matrix; 3],
    weights: &[f64],
    coefficients: [f64; 2],
) -> (f64, [f64; 2]) {
    let n = weights.len();
    let rho: [Vec<f64>; 2] = std::array::from_fn(|s| {
        (0..n)
            .map(|g| coefficients[s] * phi[(g, 0)].powi(2))
            .collect()
    });
    let grad: [Vec<[f64; 3]>; 2] = std::array::from_fn(|s| {
        (0..n)
            .map(|g| {
                std::array::from_fn(|d| 2.0 * coefficients[s] * phi[(g, 0)] * gradients[d][(g, 0)])
            })
            .collect()
    });
    let sigma: [Vec<f64>; 3] = std::array::from_fn(|pair| {
        let (a, b) = [(0, 0), (0, 1), (1, 1)][pair];
        (0..n)
            .map(|g| (0..3).map(|d| grad[a][g][d] * grad[b][g][d]).sum())
            .collect()
    });
    let vars = GridVars {
        rho: [&rho[0], &rho[1]],
        sigma: (functional.family != Family::Lda).then_some([&sigma[0], &sigma[1], &sigma[2]]),
        tau: None,
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 1, &mut out).unwrap();
    let energy = out.e.iter().zip(weights).map(|(e, w)| e * w).sum();
    let mut fock = [0.0; 2];
    for g in 0..n {
        for s in 0..2 {
            let mut value = out.vrho[s][g] * phi[(g, 0)].powi(2);
            if functional.family != Family::Lda {
                for d in 0..3 {
                    value += (2.0 * out.vsigma[2 * s][g] * grad[s][g][d]
                        + out.vsigma[1][g] * grad[1 - s][g][d])
                        * (2.0 * phi[(g, 0)] * gradients[d][(g, 0)]);
                }
            }
            fock[s] += weights[g] * value;
        }
    }
    (energy, fock)
}

fn lifted(value: Matrix) -> BlochMatrix {
    BlochMatrix {
        im: Matrix::zeros(value.rows, value.cols),
        re: value,
    }
}

#[test]
fn rank_one_rotated_hydrogen_grid_keeps_the_scalar_collinear_limit() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280)).unwrap().install(|| {
        for name in ["svwn5", "pbe"] {
            let parameters = Value::object().with("xc", name).with("basis", "sto-3g")
                .with("multiplicity", 2usize).with("grid_level", 2usize)
                .with("grid_weight_cutoff", 0.0).with("dispersion", "none");
            let method = MolecularMethod::from_controls(&Controls::from_value(&parameters).unwrap()).unwrap();
            let setup = prepare_molecular(&[1], &[[0.13, -0.17, 0.23]], &method).unwrap();
            let grid = setup.grid.as_ref().unwrap();
            let values = dft_rs::grid::eval::eval_aos(&setup.basis, &grid.points, true);
            let gradients = values.grad.unwrap();
            let phi = lifted(values.phi);
            let momentum = std::array::from_fn(|d| {
                let mut im = gradients[d].clone();
                for value in im.as_mut_slice() { *value = -*value; }
                BlochMatrix { re: Matrix::zeros(im.rows, im.cols), im }
            });
            for scale in [0.5, 1.0, 2.0] {
                // Actual normalized one-electron projector from the failing
                // physical backend, before its first XC evaluation.
                let mut density = BlochMatrix::zeros(2, 2);
                density.re[(0, 0)] = 0.9000000000000004 * scale;
                density.re[(1, 1)] = 0.1 * scale;
                density.re[(0, 1)] = 0.1800000000000001 * scale;
                density.re[(1, 0)] = density.re[(0, 1)];
                density.im[(0, 1)] = 0.24000000000000005 * scale;
                density.im[(1, 0)] = -density.im[(0, 1)];
                let pauli = pauli_components(&density).unwrap();
                let trace = pauli[0].re[(0, 0)];
                let spin: [f64; 3] = std::array::from_fn(|a| pauli[a + 1].re[(0, 0)]);
                let length = spin[0].hypot(spin[1]).hypot(spin[2]);
                let axis = spin.map(|x| x / length);
                let reference = scalar_reference(&setup.functional, &phi.re, &gradients, &grid.weights, [trace, 0.0]);
                let actual = spinor_semilocal_grid(&setup.functional, &grid.weights, &[&phi], &[&momentum], &[pauli], 8).unwrap();
                let scalar = 0.5 * (reference.1[0] + reference.1[1]);
                let magnetic = 0.5 * (reference.1[0] - reference.1[1]);
                let mut expected = BlochMatrix::zeros(2, 2);
                expected.re[(0, 0)] = scalar + magnetic * axis[2];
                expected.re[(1, 1)] = scalar - magnetic * axis[2];
                expected.re[(0, 1)] = magnetic * axis[0];
                expected.re[(1, 0)] = expected.re[(0, 1)];
                expected.im[(0, 1)] = -magnetic * axis[1];
                expected.im[(1, 0)] = -expected.im[(0, 1)];
                let norm = expected.re.as_slice().iter().chain(expected.im.as_slice()).map(|x| x.abs()).fold(0.0, f64::max);
                assert!(reference.0 != 0.0 && norm > 0.0);
                let e = (actual.0 - reference.0).abs() / reference.0.abs();
                let v = actual.1[0].re.max_abs_diff(&expected.re).max(actual.1[0].im.max_abs_diff(&expected.im)) / norm;
                println!("PSD boundary {name} scale={scale}: relative energy={e:e} all-complex-Fock={v:e}");
                assert!(e < 1e-12 && v < 1e-12);
            }
        }
    }).unwrap();
}

#[test]
fn a_genuine_positive_minority_is_not_replaced_by_zero() {
    for name in ["svwn5", "pbe"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let mut phi = Matrix::zeros(1, 1);
        phi[(0, 0)] = 32.0;
        let gradients = std::array::from_fn(|_| Matrix::zeros(1, 1));
        let momentum = std::array::from_fn(|_| BlochMatrix::zeros(1, 1));
        let minority = 2.0_f64.powi(-52);
        let reference = scalar_reference(&f, &phi, &gradients, &[1.0], [1.0 - minority, minority]);
        let empty = scalar_reference(&f, &phi, &gradients, &[1.0], [1.0, 0.0]);
        assert_ne!(
            reference.1[1], empty.1[1],
            "native functional must resolve this positive minority"
        );
        let mut density = BlochMatrix::zeros(2, 2);
        density.re[(0, 0)] = 1.0 - minority;
        density.re[(1, 1)] = minority;
        let p = pauli_components(&density).unwrap();
        let phi = lifted(phi);
        let actual = spinor_semilocal_grid(&f, &[1.0], &[&phi], &[&momentum], &[p], 8).unwrap();
        let norm = reference.1[0].abs().max(reference.1[1].abs());
        assert!(norm > 0.0);
        for s in 0..2 {
            assert!((actual.1[0].re[(s, s)] - reference.1[s]).abs() / norm < 1e-12);
        }
    }
}

#[test]
fn indefinite_and_negative_charge_points_still_fail_at_every_scale() {
    for name in ["svwn5", "pbe", "tpss"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        for rho in [2.0_f64.powi(-300), 1.0, 2.0_f64.powi(300)] {
            let mut point = SpinorXcPoint {
                density: [rho, rho * (1.0 + 1024.0 * f64::EPSILON), 0.0, 0.0],
                gradient: [[0.0; 3]; 4],
                tau: [0.0; 4],
                current: [[0.0; 3]; 4],
            };
            assert!(spinor_semilocal_point(&f, &point, 8).is_err());
            point.density = [-rho, 0.0, 0.0, 0.0];
            assert!(spinor_semilocal_point(&f, &point, 8).is_err());
        }
    }
}
