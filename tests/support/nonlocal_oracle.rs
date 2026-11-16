// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

// Closed form and independent integer-cube enumeration. Neither the
// production VV10 pair function nor its neighbor enumerator is called.
pub fn energy(
    rows: [[f64; 3]; 3],
    axes: [bool; 3],
    points: &[[f64; 3]],
    rho: &[f64],
    sigma: &[f64],
    weights: &[f64],
    settings: PeriodicVv10Options,
) -> f64 {
    let b = settings.parameters.b;
    let c = settings.parameters.c;
    let beta = (3.0 / (b * b)).powf(0.75) / 32.0;
    let omega = |i: usize| {
        (c * sigma[i].powi(2) / rho[i].powi(4) + 4.0 * std::f64::consts::PI * rho[i] / 3.0).sqrt()
    };
    let kappa = |i: usize| {
        b * (1.5 * std::f64::consts::PI) * (rho[i] / (9.0 * std::f64::consts::PI)).powf(1.0 / 6.0)
    };
    let switch = |x: f64| {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            let a = (-1.0 / x).exp();
            let b = (-1.0 / (1.0 - x)).exp();
            a / (a + b)
        }
    };
    let vacuum = |r: f64| switch((r - 1e-8) / 1e-8);
    let mut energy = beta
        * rho
            .iter()
            .zip(weights)
            .map(|(r, w)| r * w * vacuum(*r))
            .sum::<f64>();
    for i in 0..rho.len() {
        for j in 0..rho.len() {
            for a in -4i32..=4 {
                for b in -4i32..=4 {
                    for d in -4i32..=4 {
                        let index = [a, b, d];
                        if (0..3).any(|k| !axes[k] && index[k] != 0) {
                            continue;
                        }
                        let r2 = (0..3)
                            .map(|k| {
                                let r = points[i][k] - points[j][k]
                                    + (0..3).map(|l| rows[l][k] * index[l] as f64).sum::<f64>();
                                r * r
                            })
                            .sum::<f64>();
                        if r2 > settings.cutoff.powi(2) {
                            continue;
                        }
                        let taper = 1.0
                            - switch(
                                (r2 - (0.8 * settings.cutoff).powi(2))
                                    / (settings.cutoff.powi(2) - (0.8 * settings.cutoff).powi(2)),
                            );
                        let gi = omega(i) * r2 + kappa(i);
                        let gj = omega(j) * r2 + kappa(j);
                        energy -= 0.75
                            * weights[i]
                            * weights[j]
                            * rho[i]
                            * rho[j]
                            * taper
                            * vacuum(rho[i])
                            * vacuum(rho[j])
                            / (gi * gj * (gi + gj));
                    }
                }
            }
        }
    }
    energy
}
