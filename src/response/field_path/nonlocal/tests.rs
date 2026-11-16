// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Direct finite-phase, forward density-coordinate AD of both VV10 endpoints.
//! This oracle uses no production density series or potential pullback.
use super::super::current_invariant_tests::{
    coordinates, density_path, finite_fields, spatial_fixture, D,
};
use super::*;
use crate::acceptance::components::{assert_no_failures, fit_components};
use crate::dual::Scalar;

fn fixture() -> NonlocalFieldProvider {
    let f = spatial_fixture();
    NonlocalFieldProvider {
        grid: MolecularGrid {
            points: f
                .positions
                .iter()
                .map(|r| Vec3::new(r[0], r[1], r[2]))
                .collect(),
            weights: f.weights.clone(),
            raw_weights: f.weights,
            atom_of: vec![0, 1, 2, 3, 0],
        },
        points: f.points,
        parameters: Vv10Params::ORIGINAL,
    }
}
fn finite(provider: &NonlocalFieldProvider, p: &[Vec<Cm>; 2], t: f64) -> Vec<f64> {
    let c = D::constant;
    let mut rho = Vec::new();
    let mut omega = Vec::new();
    let mut kappa = Vec::new();
    for point in &provider.points {
        let (v, g) = finite_fields(point, p, t);
        let r = v[0] + v[1];
        assert!(r.re > crate::nlc::RHO_CUT);
        let sigma = (0..3)
            .map(|a| g[0][a] + g[1][a])
            .fold(c(0.0), |s, g| s + g * g);
        let pi = std::f64::consts::PI;
        rho.push(r);
        omega.push(
            (c(provider.parameters.c) * sigma * sigma / r.powi(4) + c(4.0 * pi / 3.0) * r).sqrt(),
        );
        kappa.push(c(provider.parameters.b * 1.5 * pi) * (r / c(9.0 * pi)).powf(1.0 / 6.0));
    }
    let beta = (3.0 / provider.parameters.b.powi(2)).powf(0.75) / 32.0;
    let mut energy = c(0.0);
    for i in 0..rho.len() {
        energy = energy + c(provider.grid.weights[i] * beta) * rho[i];
        for j in 0..rho.len() {
            let r2 = (provider.grid.points[i] - provider.grid.points[j]).norm2();
            let a = omega[i] * c(r2) + kappa[i];
            let b = omega[j] * c(r2) + kappa[j];
            energy = energy
                - c(0.75 * provider.grid.weights[i] * provider.grid.weights[j]) * rho[i] * rho[j]
                    / (a * b * (a + b));
        }
    }
    std::iter::once(energy.re)
        .chain(
            coordinates()
                .iter()
                .enumerate()
                .map(|(k, &(_, i, j, _))| energy.im[k] / if i == j { 1.0 } else { 2.0 }),
        )
        .collect()
}

#[test]
fn both_nonlocal_endpoints_and_every_complex_potential_component_have_independent_fits() {
    let provider = fixture();
    let p = density_path();
    let coords = coordinates();
    let mut exact = Vec::new();
    for n in 0..=2 {
        let (e, f) = provider.evaluate(&p, n, true).unwrap();
        let f = f.unwrap();
        let factor = if n == 2 { 2.0 } else { 1.0 };
        let mut row = vec![factor * e[n]];
        for &(s, i, j, im) in &coords {
            row.push(factor * if im { f[s].im[(i, j)] } else { f[s].re[(i, j)] });
        }
        exact.push(row);
    }
    let zero = finite(&provider, &p, 0.0);
    let base = zero
        .iter()
        .zip(&exact[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("nonlocal direct two-endpoint base defect {base:.12e}");
    assert!(base < 1e-14);
    let mut failures = Vec::new();
    for order in 1..=2 {
        let names: Vec<_> = (0..19)
            .map(|i| format!("nonlocal complex order={order} component={i}"))
            .collect();
        fit_components(
            &names,
            &[
                [0.08, 0.06, 0.04, 0.03],
                [0.02, 0.015, 0.01, 0.0075],
                [0.16, 0.12, 0.08, 0.06],
            ],
            |steps| {
                let mut errors = vec![[0.0; 4]; 19];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = finite(&provider, &p, h);
                    let minus = finite(&provider, &p, -h);
                    for i in 0..19 {
                        let fd = if order == 1 {
                            (plus[i] - minus[i]) / (2.0 * h)
                        } else {
                            (plus[i] + minus[i] - 2.0 * zero[i]) / (h * h)
                        };
                        errors[i][k] = (fd - exact[order][i]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("full nonlocal complex energy and Fock", &failures);
    let mut gauged = fixture();
    let shift = [0.19, -0.23, 0.31];
    for (point, r) in gauged.points.iter_mut().zip(&gauged.grid.points) {
        for mu in 0..3 {
            point.theta[mu] += (0..3).map(|a| shift[a] * r.get(a)).sum::<f64>();
            for (a, &v) in shift.iter().enumerate() {
                point.slope[mu][a] += v;
            }
        }
    }
    let (e, f) = gauged.evaluate(&p, 2, true).unwrap();
    let f = f.unwrap();
    let mut defect = (2.0 * e[2] - exact[2][0]).abs();
    for (k, &(s, i, j, im)) in coords.iter().enumerate() {
        defect = defect.max(
            (2.0 * if im { f[s].im[(i, j)] } else { f[s].re[(i, j)] } - exact[2][k + 1]).abs(),
        );
    }
    println!("nonlocal complete U(1) coefficient defect {defect:.12e}");
    assert!(defect < 1e-14);
}
