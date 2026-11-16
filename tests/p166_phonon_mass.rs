// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent mass-kernel expressions, joint resonances and four-order error fits.
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/mass_oracle.rs"]
mod mass_oracle;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::properties::pairing::{Lm2005, PairingInput};
type Jet = Taylor<6>;

fn data<S: Scalar>(t: S, case: usize) -> [S; 4] {
    let f = S::from_f64;
    let w = f(0.025) / (f(1.0) - t.scale(0.25));
    let kt = f(0.004) * (f(1.0) + t.scale(0.15));
    let y = if case == 4 {
        w
    } else {
        f(0.011) / (f(1.0) - t.scale(0.07))
    };
    let x = match case {
        1 => y + w,
        2 | 4 => f(0.0),
        3 => t.scale(0.002) / (f(1.0) - t.scale(0.11)),
        _ => f(0.007) * (f(1.0) + t.scale(0.17)),
    };
    [x, y, w, kt]
}
fn crossing_oracle<const W: usize>(t: Jet) -> Jet {
    // Divide the independent exponential series at its exact removable zero.
    // Evaluating that polynomial avoids repeatedly differentiating a tiny
    // numerator/denominator at the displaced validation points. Orders 24
    // and 32 independently bound its truncation on every stated fit step.
    let [x, y, w, kt] = data(Taylor::<W>::from_coefficients(&[0.0, 1.0]), 3);
    let n = mass_oracle::numerator(x, y, w, kt);
    let one = Taylor::from_f64(1.0);
    let d = one - (one + (x / kt).exp()).recip().scale(2.0);
    assert!(
        n.c[0].abs() < 1e-9,
        "raw odd-numerator cancellation: {}",
        n.c[0]
    );
    let mut coefficients = vec![0.0; W - 1];
    for k in 0..W - 1 {
        let other: f64 = (1..=k).map(|j| d.c[j + 1] * coefficients[k - j]).sum();
        coefficients[k] = (n.c[k + 1] - other) / d.c[1];
    }
    coefficients
        .iter()
        .rev()
        .fold(Jet::from_f64(0.0), |sum, &c| sum * t + Jet::from_f64(c))
}
fn oracle(t: Jet, case: usize) -> Jet {
    if case == 3 {
        let result = crossing_oracle::<24>(t);
        let check = crossing_oracle::<32>(t);
        for k in 0..5 {
            assert!((result.c[k] - check.c[k]).abs() < 1e-10);
        }
        result
    } else {
        let [x, y, w, kt] = data(t, case);
        mass_oracle::factor(x, y, w, kt)
    }
}

#[test]
fn phonon_mass_has_independent_confluent_fermi_and_four_order_fits() {
    let rule = Lm2005::default();
    let check = Lm2005::new(16).unwrap();
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut fits_pass = true;
    for case in 0..5 {
        let t = Jet::from_coefficients(&[0.0, 1.0]);
        let [x, y, w, kt] = data(t, case);
        let result = rule.mass_factor(x, y, w, kt).unwrap();
        let exact = oracle(t, case);
        let alternative = check.mass_factor(x, y, w, kt).unwrap();
        for k in 0..5 {
            let error = (result.c[k] - exact.c[k]).abs();
            println!("phonon mass case={case} coefficient={k} value={:.15e} closed error={error:.12e} quadrature={:.12e}",result.c[k],(result.c[k]-alternative.c[k]).abs());
            assert!(error < 1e-8 && (result.c[k] - alternative.c[k]).abs() < 1e-8);
        }
        assert!(result.c[0] > 0.0);
        for order in 1..=4 {
            let target = result.c[order] * order as f64;
            let approximations: Vec<_> = steps
                .iter()
                .map(|&h| {
                    (oracle(Jet::from_coefficients(&[h, 1.0]), case).c[order - 1]
                        - oracle(Jet::from_coefficients(&[-h, 1.0]), case).c[order - 1])
                        / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = approximations.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * approximations[3] - approximations[2]) / 3.0 - target).abs();
            println!("phonon mass case={case} order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            fits_pass &= (fit.exponent - 2.0).abs() < 0.03 && fit.r_squared > 0.9999 && rich < 1e-4;
        }
    }
    assert!(
        fits_pass,
        "all raw derivative fits must satisfy their original gates"
    );
}

#[test]
fn band_weights_vertices_and_coulomb_are_separate_physical_components() {
    let t = Taylor::<5>::from_coefficients(&[0.0, 1.0]);
    let f = Taylor::from_f64;
    let input = PairingInput {
        energies: vec![f(0.007) + t.scale(0.001), f(-0.014) + t.scale(0.002)],
        chemical_potential: t.scale(0.0003),
        temperature_energy: f(0.004) + t.scale(0.0002),
        phonon_energies: vec![f(0.025) + t.scale(0.001), f(0.035) - t.scale(0.001)],
        coupling_squared: vec![
            vec![f(0.0001) + t.scale(0.00002); 4],
            vec![f(0.00007) - t.scale(0.00001); 4],
        ],
        screened_coulomb: vec![f(0.01); 4],
    };
    let weights = [f(0.6) + t.scale(0.02), f(0.4) - t.scale(0.02)];
    let rule = Lm2005::default();
    let result = rule.mass(&input, &weights).unwrap();
    for i in 0..2 {
        let mut expected = f(0.0);
        for (mode, &w) in input.phonon_energies.iter().enumerate() {
            for (j, &weight) in weights.iter().enumerate() {
                expected = expected
                    + weight
                        * input.coupling_squared[mode][i * 2 + j]
                        * mass_oracle::factor(
                            input.energies[i] - input.chemical_potential,
                            input.energies[j] - input.chemical_potential,
                            w,
                            input.temperature_energy,
                        );
            }
        }
        for k in 0..5 {
            assert!((result[i].c[k] - expected.c[k]).abs() < 1e-11);
        }
    }
    let mut without = input.clone();
    without.screened_coulomb = vec![f(0.0); 4];
    assert_eq!(rule.mass(&without, &weights).unwrap(), result);
    without
        .coupling_squared
        .iter_mut()
        .flatten()
        .for_each(|g| *g = f(0.0));
    assert!(rule
        .mass(&without, &weights)
        .unwrap()
        .iter()
        .all(|z| z.c.iter().all(|x| *x == 0.0)));
}

#[test]
fn flat_dos_mass_approaches_the_independent_zero_temperature_lambda_limit() {
    let rule = Lm2005::default();
    let (nodes, weights) = dft_rs::grid::gauss_legendre(64);
    let omega: f64 = 0.025;
    let bandwidth: f64 = 0.25;
    let vertex = 0.0001;
    let target = 2.0 * vertex * (1.0 / omega - 1.0 / (omega + bandwidth));
    // The symmetric LM mass contains an explicit linear-kT term from D_b.
    // Integrating -4*kT/(omega+abs(y))^3 gives this finite-band slope.
    let slope = 4.0 * vertex * (1.0 / omega.powi(2) - 1.0 / (omega + bandwidth).powi(2));
    let temperatures = [0.000125, 0.0000625, 0.00003125, 0.000015625];
    let mut errors = Vec::new();
    for kt in temperatures {
        let mut boundaries = vec![-bandwidth, bandwidth];
        for center in [-omega, 0.0, omega] {
            for multiple in [-6.0, 0.0, 6.0] {
                boundaries.push(center + multiple * kt);
            }
        }
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup();
        let mut value = 0.0;
        for interval in boundaries.windows(2) {
            let mid = 0.5 * (interval[0] + interval[1]);
            let half = 0.5 * (interval[1] - interval[0]);
            for (&node, &weight) in nodes.iter().zip(&weights) {
                // Both signs are evaluated independently, including their resonances.
                value += half
                    * weight
                    * vertex
                    * rule.mass_factor(0.0, mid + half * node, omega, kt).unwrap();
            }
        }
        let error = (value - (target - slope * kt)).abs();
        errors.push(error);
        println!("phonon flat DOS kT={kt:.12e} value={value:.15e} exact_zero_T={target:.15e} linear_slope={slope:.15e} raw_zero_T_error={:.12e} corrected_error={error:.12e}",(value-target).abs());
    }
    let fit = convergence::error_power_fit(&temperatures, &errors);
    println!(
        "phonon flat DOS thermal-limit steps={temperatures:?} errors={errors:?} p={:.9} R2={:.12}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999);
    // Away from the Fermi surface the independent zero-T factor is elementary.
    let expected = 1.0_f64 / (0.007 + 0.011 + omega).powi(2);
    for x in [-0.007, 0.007] {
        for y in [-0.011, 0.011] {
            let value = rule.mass_factor(x, y, omega, 0.00001).unwrap();
            assert!((value - expected).abs() < 1e-10);
        }
    }
}
