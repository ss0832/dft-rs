// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent exponential/residue oracles for the LM2005 pairing convolution.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::properties::pairing::{Lm2005OffDiagonal, PairingInput};

#[path = "support/pairing_oracle.rs"]
mod pairing_oracle;
use pairing_oracle::factor as factor_oracle;

fn fermi_surface_oracle<S: Scalar>(w: S, kt: S) -> S {
    // Exact finite-temperature residue result at xi=xi'=0 (Eq. 75).
    let one = S::from_f64(1.0);
    let u = w / kt;
    let e = u.exp();
    -(one - ((e + one) / (e - one) / u).scale(4.0) + u.powi(2).recip().scale(8.0)).scale(2.0) / w
}

fn check_fit(name: &str, order: usize, target: f64, samples: impl Fn(f64) -> f64) {
    let steps = [0.04, 0.02, 0.01, 0.005];
    let estimates: Vec<_> = steps
        .iter()
        .map(|&h| (samples(h) - samples(-h)) / (2.0 * h))
        .collect();
    let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
    println!("pairing {name} order={order} target={target:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
    assert!(
        (fit.exponent - 2.0).abs() < 0.05,
        "{name} order {order}: {}",
        fit.exponent
    );
    assert!(fit.r_squared > 0.999);
    assert!(rich < 1e-6);
}

#[test]
fn off_diagonal_kernel_has_independent_terms_limits_and_four_order_fits() {
    let kernel = Lm2005OffDiagonal::default();
    for (x, y, w, kt) in [
        (0.016, 0.007, 0.025, 0.006),
        (-0.016, 0.007, 0.025, 0.006),
        (0.0, 0.0, 0.025, 0.006),
        (0.0001, -0.0002, 0.025, 0.006),
    ] {
        let value = kernel.phonon_factor(x, y, w, kt).unwrap();
        let expected = if x == 0.0 && y == 0.0 {
            fermi_surface_oracle(w, kt)
        } else {
            factor_oracle(x, y, w, kt)
        };
        println!(
            "pairing factor ({x},{y},{w},{kt})={value:.15e} oracle={expected:.15e} error={:.12e}",
            (value - expected).abs()
        );
        assert!((value - expected).abs() < 2e-9);
        assert!(value < 0.0);
        assert!((value - kernel.phonon_factor(y, x, w, kt).unwrap()).abs() < 1e-11);
        assert!((value - kernel.phonon_factor(-x, -y, w, kt).unwrap()).abs() < 1e-11);
    }
    // Exact removable resonance in the first fraction of Eq. 55.
    let (x, y, w, kt) = (0.032_f64, 0.007_f64, 0.025_f64, 0.006_f64);
    let (ex, ey, ew) = ((x / kt).exp(), (y / kt).exp(), (w / kt).exp());
    let expected = (ex / kt - (ey - ex * ew) / (2.0 * w)) / ((1.0 + ex) * (1.0 + ey) * (ew - 1.0));
    let value = kernel.convolution(x, y, w, kt).unwrap();
    println!(
        "pairing resonance value={value:.15e} residue={expected:.15e} error={:.12e}",
        (value - expected).abs()
    );
    assert!((value - expected).abs() < 1e-11);
    // The physical zero-temperature attractive limit and phonon decoupling.
    let cold = kernel.phonon_factor(0.016, 0.007, 0.025, 1e-6).unwrap();
    assert!((cold + 2.0 / (0.016 + 0.007 + 0.025)).abs() < 1e-11);
    let input = PairingInput {
        energies: vec![-0.016, 0.007],
        chemical_potential: 0.0,
        temperature_energy: 0.006,
        phonon_energies: vec![0.025],
        coupling_squared: vec![vec![0.0; 4]],
        screened_coulomb: vec![0.01, 0.002, 0.002, 0.008],
    };
    let decoupled = kernel.evaluate(&input).unwrap();
    assert_eq!(decoupled.total, input.screened_coulomb);
    assert_eq!(decoupled.phonon, vec![0.0; 4]);
    // All arguments respond, including kT, phonon energy and squared vertex.
    let values = |t: Taylor<5>| {
        let c = Taylor::<5>::from_f64;
        (
            c(0.016) + t.scale(0.004),
            c(0.007) - t.scale(0.003),
            c(0.025) + t.scale(0.002),
            c(0.006) + t.scale(0.001),
            c(0.003) + t.scale(0.0004),
            c(0.001) + t.scale(0.0003),
        )
    };
    let (x, y, w, kt, g2, coulomb) = values(Taylor::from_coefficients(&[0.0, 1.0]));
    let production = g2 * kernel.phonon_factor(x, y, w, kt).unwrap() + coulomb;
    let oracle = g2 * factor_oracle(x, y, w, kt) + coulomb;
    for k in 0..=4 {
        assert!((production.c[k] - oracle.c[k]).abs() < 2e-12);
    }
    for order in 1..=4 {
        check_fit(
            "mixed energies/temperature/vertex",
            order,
            order as f64 * production.c[order],
            |t| {
                let (x, y, w, kt, g2, coulomb) = values(Taylor::from_coefficients(&[t, 1.0]));
                (g2 * factor_oracle(x, y, w, kt) + coulomb).c[order - 1]
            },
        );
    }
    // Fermi-surface analytic limits remain differentiable through fourth order.
    let c = Taylor::<5>::from_f64;
    let t = Taylor::from_coefficients(&[0.0, 1.0]);
    let (w, kt) = (c(0.025) + t.scale(0.002), c(0.006) + t.scale(0.001));
    let production = kernel.phonon_factor(c(0.0), c(0.0), w, kt).unwrap();
    let oracle = fermi_surface_oracle(w, kt);
    for k in 0..=4 {
        assert!((production.c[k] - oracle.c[k]).abs() < 1e-10);
    }
    for order in 1..=4 {
        check_fit(
            "Fermi surface thermal limit",
            order,
            order as f64 * production.c[order],
            |u| {
                let t = Taylor::from_coefficients(&[u, 1.0]);
                fermi_surface_oracle(c(0.025) + t.scale(0.002), c(0.006) + t.scale(0.001)).c
                    [order - 1]
            },
        );
    }
}
