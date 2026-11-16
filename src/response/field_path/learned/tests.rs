// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The full global graph is differentiated in independent Hermitian density
//! coordinates with forward AD at finite trigonometric phases. It shares
//! neither the production density series nor its reverse-potential pullback.
use super::super::current_invariant_tests::{coordinates, density_path, finite_fields, D};
use super::*;
use crate::acceptance::components::{assert_no_failures, fit_components};
use crate::skala::{random_tensors, SkalaSizes};

fn fixture() -> (LearnedFieldProvider, Vec<Point>) {
    let fixture = super::super::current_invariant_tests::spatial_fixture();
    let (centers, positions, weights, points) = (
        fixture.centers,
        fixture.positions,
        fixture.weights,
        fixture.points,
    );
    let model = SkalaModel::from_tensors(&random_tensors(
        7,
        SkalaSizes {
            width: 6,
            channels: 2,
            shells: 1,
            correlation: 2,
            layers: 1,
            post: 2,
        },
    ))
    .unwrap();
    (
        LearnedFieldProvider {
            model: Arc::new(model),
            atom_of: vec![0, 1, 2, 3, 1],
            points: positions,
            centers,
            weights,
            raw_weights: vec![0.7, 0.9, 0.5, 0.8, 0.6],
        },
        points,
    )
}
fn finite(provider: &LearnedFieldProvider, points: &[Point], p: &[Vec<Cm>; 2], t: f64) -> Vec<f64> {
    let fields: Vec<_> = points
        .iter()
        .map(|point| finite_fields(point, p, t))
        .collect();
    let rho: [Vec<D>; 2] = std::array::from_fn(|s| fields.iter().map(|(v, _)| v[s]).collect());
    let gradient: [Vec<[D; 3]>; 2] =
        std::array::from_fn(|s| fields.iter().map(|(_, g)| g[s]).collect());
    let tau: [Vec<D>; 2] = std::array::from_fn(|s| fields.iter().map(|(v, _)| v[5 + s]).collect());
    let weights: Vec<_> = provider.weights.iter().map(|&w| D::constant(w)).collect();
    let result = provider
        .model
        .evaluate(
            &SkalaInput {
                atom_of: &provider.atom_of,
                points: &provider.points,
                centers: &provider.centers,
                weights: &weights,
                atomic_weights: &provider.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&gradient[0], &gradient[1]],
                tau: [&tau[0], &tau[1]],
            },
            false,
        )
        .unwrap()
        .energy;
    std::iter::once(result.re)
        .chain(
            coordinates()
                .iter()
                .enumerate()
                .map(|(k, &(_, i, j, _))| result.im[k] / if i == j { 1.0 } else { 2.0 }),
        )
        .collect()
}

#[test]
fn all_global_learned_energy_and_complex_fock_components_have_independent_fits() {
    let (provider, points) = fixture();
    let p = density_path();
    let coords = coordinates();
    let mut analytic = Vec::new();
    for n in 0..=2 {
        let series: Vec<_> = points.iter().map(|p| point_series(p, n + 1)).collect();
        let (energy, fock) = provider
            .evaluate(
                &series,
                &p,
                n,
                true,
                MagneticKineticDensity::CurrentInvariant,
            )
            .unwrap();
        let fock = fock.unwrap();
        let factor = if n == 2 { 2.0 } else { 1.0 };
        let mut values = vec![factor * energy[n]];
        for &(s, i, j, im) in &coords {
            values.push(
                factor
                    * if im {
                        fock[s].im[(i, j)]
                    } else {
                        fock[s].re[(i, j)]
                    },
            );
        }
        analytic.push(values);
    }
    let zero = finite(&provider, &points, &p, 0.0);
    for (a, b) in analytic[0].iter().zip(&zero) {
        assert!((a - b).abs() < 2e-13, "global base pullback {a:e} vs {b:e}");
    }
    let mut failures = Vec::new();
    for order in 1..=2 {
        let names: Vec<_> = (0..19)
            .map(|i| format!("global learned order={order} component={i}"))
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
                    let plus = finite(&provider, &points, &p, h);
                    let minus = finite(&provider, &points, &p, -h);
                    for i in 0..19 {
                        let fd = if order == 1 {
                            (plus[i] - minus[i]) / (2.0 * h)
                        } else {
                            (plus[i] + minus[i] - 2.0 * zero[i]) / (h * h)
                        };
                        errors[i][k] = (fd - analytic[order][i]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("complete global learned energy/Fock derivatives", &failures);
    // An exact common U(1) phase and its potential/gradient transformation
    // leave every global energy and Fock coefficient unchanged.
    let mut transformed = points;
    let shift = [0.19, -0.23, 0.31];
    for (point, r) in transformed.iter_mut().zip(&provider.points) {
        for mu in 0..3 {
            point.theta[mu] += (0..3).map(|a| shift[a] * r[a]).sum::<f64>();
            for a in 0..3 {
                point.slope[mu][a] += shift[a];
            }
        }
    }
    let series: Vec<_> = transformed.iter().map(|p| point_series(p, 3)).collect();
    let (energy, fock) = provider
        .evaluate(
            &series,
            &p,
            2,
            true,
            MagneticKineticDensity::CurrentInvariant,
        )
        .unwrap();
    let fock = fock.unwrap();
    let mut values = vec![2.0 * energy[2]];
    for &(s, i, j, im) in &coords {
        values.push(
            2.0 * if im {
                fock[s].im[(i, j)]
            } else {
                fock[s].re[(i, j)]
            },
        );
    }
    let defect = values
        .iter()
        .zip(&analytic[2])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("global learned full U(1) coefficient defect {defect:.12e}");
    assert!(defect < 2e-12);
}
