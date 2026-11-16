// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent scalar f + D_spin f quadrature. Uses explicit projected
//! descriptors and first functional partials, without nested duals or the
//! production spinor potential/response implementation.

use dft_rs::xc::{Family, GridVars, ResolvedFunctional, XcOut};

pub fn energy(functional: &ResolvedFunctional, x: &[f64; 32], order: usize) -> f64 {
    let (nodes, weights) = dft_rs::grid::gauss_legendre(order);
    let mut fields: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::new());
    let mut slopes: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::new());
    let mut quadrature = Vec::new();
    for (&z, &weight) in nodes.iter().zip(&weights) {
        for azimuth in 0..2 * order {
            let phi = 2.0 * std::f64::consts::PI * (azimuth as f64 + 0.5) / (2 * order) as f64;
            let radius = (1.0 - z * z).sqrt();
            let axis = [radius * phi.cos(), radius * phi.sin(), z];
            let spin = |start: usize, stride: usize| {
                (0..3)
                    .map(|a| axis[a] * x[start + (a + 1) * stride])
                    .sum::<f64>()
            };
            let mut rho = [0.0; 2];
            let mut drho = [0.0; 2];
            let mut gradient = [[0.0; 3]; 2];
            let mut dgradient = [[0.0; 3]; 2];
            let mut tau = [0.0; 2];
            let mut dtau = [0.0; 2];
            for s in 0..2 {
                let sign = if s == 0 { 1.0 } else { -1.0 };
                rho[s] = 0.5 * (x[0] + sign * spin(0, 1));
                drho[s] = 0.5 * sign * spin(0, 1);
                for d in 0..3 {
                    gradient[s][d] = 0.5 * (x[4 + d] + sign * spin(4 + d, 3));
                    dgradient[s][d] = 0.5 * sign * spin(4 + d, 3);
                }
                if functional.family == Family::MetaGga {
                    tau[s] = 0.5 * (x[16] + sign * spin(16, 1));
                    dtau[s] = 0.5 * sign * spin(16, 1);
                    let mut current_square = 0.0;
                    let mut current_slope = 0.0;
                    for d in 0..3 {
                        let j = 0.5 * (x[20 + d] + sign * spin(20 + d, 3));
                        let dj = 0.5 * sign * spin(20 + d, 3);
                        current_square += j * j;
                        current_slope += j * dj;
                    }
                    tau[s] -= current_square / (2.0 * rho[s]);
                    dtau[s] += -current_slope / rho[s]
                        + current_square * drho[s] / (2.0 * rho[s] * rho[s]);
                }
            }
            let mut y = [rho[0], rho[1], 0.0, 0.0, 0.0, tau[0], tau[1]];
            let mut dy = [drho[0], drho[1], 0.0, 0.0, 0.0, dtau[0], dtau[1]];
            for (c, (a, b)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                y[c + 2] = (0..3).map(|d| gradient[a][d] * gradient[b][d]).sum();
                dy[c + 2] = (0..3)
                    .map(|d| dgradient[a][d] * gradient[b][d] + gradient[a][d] * dgradient[b][d])
                    .sum();
            }
            for i in 0..7 {
                fields[i].push(y[i]);
                slopes[i].push(dy[i]);
            }
            quadrature.push(weight / (4 * order) as f64);
        }
    }
    let vars = GridVars {
        rho: [&fields[0], &fields[1]],
        sigma: (functional.family != Family::Lda).then_some([&fields[2], &fields[3], &fields[4]]),
        tau: (functional.family == Family::MetaGga).then_some([&fields[5], &fields[6]]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 1, &mut out).unwrap();
    quadrature
        .iter()
        .enumerate()
        .map(|(g, &weight)| {
            let mut value = out.e[g];
            for s in 0..2 {
                value += out.vrho[s][g] * slopes[s][g];
            }
            if functional.family != Family::Lda {
                for c in 0..3 {
                    value += out.vsigma[c][g] * slopes[c + 2][g];
                }
            }
            if functional.family == Family::MetaGga {
                for s in 0..2 {
                    value += out.vtau[s][g] * slopes[s + 5][g];
                }
            }
            weight * value
        })
        .sum()
}
