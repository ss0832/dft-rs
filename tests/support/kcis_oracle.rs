// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent scalar evaluation of the published KCIS and exchange terms.

use dft_rs::dual::Scalar;
use std::f64::consts::PI;

fn pw92<S: Scalar>(rs: S, polarized: bool) -> S {
    let [a, alpha, b1, b2, b3, b4] = if polarized {
        [0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517]
    } else {
        [0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]
    };
    let root = rs.sqrt();
    let polynomial = root.scale(b1) + rs.scale(b2) + (rs * root).scale(b3) + (rs * rs).scale(b4);
    -(S::from_f64(1.0) + rs.scale(alpha)) * polynomial.scale(2.0 * a).recip().ln_1p().scale(2.0 * a)
}

fn gas<S: Scalar>(density: S, sigma: S, polarized: bool) -> S {
    let rs = density.recip().scale(0.75 / PI).powf(1.0 / 3.0);
    let ec0 = pw92(rs, false);
    let root = rs.sqrt();
    let numerator = (rs * root).scale(0.04953);
    let slope =
        numerator / (S::from_f64(1.0) + root.scale(1.07924) + rs.scale(0.07928) + numerator);
    let coefficients = [-0.02504, 0.007026, -0.001268, 0.0001136, -0.0000038421];
    let mut curvature = S::from_f64(0.0);
    let mut power = rs * rs * rs;
    for coeff in coefficients {
        curvature = curvature + power.scale(coeff);
        power = power * rs;
    }
    let asymptote = rs
        .recip()
        .powi(2)
        .scale(-0.06483 * (9.0 * PI / 4.0).powf(2.0 / 3.0));
    // Match R(0), R'(0), R''(0), and lim G*R(G) by eliminating c1.
    let quadratic = (curvature.scale(0.5) * ec0 - slope * slope) / (-asymptote * slope - ec0 * ec0);
    let linear = (asymptote * quadratic - slope) / ec0;
    let numerator_linear = asymptote * quadratic;
    let (unif, grad_scale, n_scale, d_scale, q_scale) = if polarized {
        (pw92(rs, true), 2.0_f64.powf(-1.0 / 3.0), 0.7, 1.5, 2.59)
    } else {
        (ec0, 1.0, 1.0, 1.0, 1.0)
    };
    let gap = sigma / density.powi(2).scale(8.0);
    let tsq = sigma
        * rs.powi(7)
            .scale(PI.powf(8.0 / 3.0) * (2.0_f64 / 3.0).powf(2.0 / 3.0) / 9.0);
    let gradient_energy =
        unif / (S::from_f64(1.0) + (-tsq.scale(grad_scale) / unif).ln_1p().scale(0.066725));
    (gradient_energy + (gap * numerator_linear).scale(n_scale))
        / (S::from_f64(1.0)
            + (gap * linear).scale(d_scale)
            + (gap * gap * quadratic).scale(q_scale))
}

fn correlation<S: Scalar>(point: [S; 7]) -> S {
    let [a, b, aa, ab, bb, ta, tb] = point;
    let one = S::from_f64(1.0);
    let mut terms = S::from_f64(0.0);
    for (r, grad, kinetic) in [(a, aa, ta), (b, bb, tb)] {
        if r.value() >= 1e-13 {
            terms = terms + grad * gas(r, grad, true) / kinetic.scale(8.0);
        }
    }
    let r = a + b;
    let total_gradient = aa + bb + ab.scale(2.0);
    let ec = if a.value().min(b.value()) <= 1e-13 {
        gas(r, total_gradient, true)
    } else {
        let z = (a - b) / r;
        let f = ((one + z).powf(4.0 / 3.0) + (one - z).powf(4.0 / 3.0) - S::from_f64(2.0))
            .scale(1.0 / (2.0_f64.powf(4.0 / 3.0) - 2.0));
        (one - f) * gas(r, total_gradient, false) + f * gas(r, total_gradient, true)
    };
    r * ec - terms
}

/// The unscaled semilocal exchange and KCIS correlation terms.
pub fn terms<S: Scalar>(point: [S; 7], exchange: Option<&str>) -> [S; 2] {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let [a, b, aa, _, bb, _, _] = point;
    let mut x = zero;
    let cx = 0.75 * (6.0 / PI).cbrt();
    for (density, gradient) in [(a, aa), (b, bb)] {
        if density.value() < 1e-13 {
            continue;
        }
        let reduced_sq = gradient / density.powf(8.0 / 3.0);
        let fx = match exchange {
            None => continue,
            Some("pbe") => {
                let s2 = reduced_sq.scale(0.25 / (6.0 * PI * PI).powf(2.0 / 3.0));
                let mu = 0.06672455060314922 * PI * PI / 3.0;
                S::from_f64(1.804) - (one + s2.scale(mu / 0.804)).recip().scale(0.804)
            }
            Some("mpw91") => {
                let b = 0.00426;
                let beta = 5.0 * (36.0 * PI).powf(-5.0 / 3.0);
                let gradual = if reduced_sq.value() == 0.0 {
                    zero
                } else {
                    let reduced = reduced_sq.sqrt();
                    reduced * reduced.asinh()
                };
                let common = one + gradual.scale(6.0 * b);
                let tail = if reduced_sq.value() == 0.0 {
                    zero
                } else {
                    reduced_sq.powf(1.86).scale(1e-6 / cx)
                };
                let alpha = 25.0 / (6.0 * PI * PI).powf(2.0 / 3.0);
                let coefficient = (S::from_f64(b) - reduced_sq.scale(-alpha).exp().scale(b - beta))
                    .scale(1.0 / cx);
                (common + coefficient * reduced_sq) / (common + tail)
            }
            _ => panic!("unknown test exchange"),
        };
        x = x - density.powf(4.0 / 3.0).scale(cx) * fx;
    }

    let corr = if (a + b).value() >= 1e-13 {
        correlation(point)
    } else {
        zero
    };
    [x, corr]
}
