// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dfpt::spectral::{
    screened_response, BandPair, BandSpace, DensityKernel, SpectralPropagator,
};
use dft_rs::dfpt::thermodynamic::{divided_difference, fermi, occupation_response, Ensemble};
use dft_rs::response::complex::Complex;
use dft_rs::response::{Frequency, ResponseSolverConfig};
use dft_rs::Result;

#[test]
fn thermal_number_temperature_response_matches_independent_distributions() {
    let energies = [-0.3, -0.1, 0.08, 0.2];
    let changes = [0.1, -0.2, 0.05, 0.07];
    let weights = [0.25; 4];
    let mu = 0.01;
    let temperature = 0.07;
    let dt = 0.013;
    for ensemble in [
        Ensemble::FixedNumber { delta_number: 0.0 },
        Ensemble::FixedChemicalPotential { delta_mu: 0.04 },
    ] {
        let response =
            occupation_response(&energies, &weights, &changes, mu, temperature, dt, ensemble)
                .unwrap();
        let delta_mu = response.delta_mu.unwrap();
        let step = 1e-5;
        for i in 0..energies.len() {
            let plus = fermi(
                energies[i] + step * changes[i],
                mu + step * delta_mu,
                temperature + step * dt,
            )
            .unwrap()
            .0;
            let minus = fermi(
                energies[i] - step * changes[i],
                mu - step * delta_mu,
                temperature - step * dt,
            )
            .unwrap()
            .0;
            assert!(((plus - minus) / (2.0 * step) - response.delta_occupations[i]).abs() < 1e-9);
        }
        if matches!(ensemble, Ensemble::FixedNumber { .. }) {
            assert!(response.delta_number.abs() < 1e-14);
        }
    }
    let shifted = occupation_response(
        &energies,
        &weights,
        &[0.4; 4],
        mu,
        temperature,
        0.0,
        Ensemble::FixedNumber { delta_number: 0.0 },
    )
    .unwrap();
    assert!((shifted.delta_mu.unwrap() - 0.4).abs() < 1e-14);
    assert!(shifted.delta_occupations.iter().all(|x| x.abs() < 1e-14));
}

#[test]
fn fermi_divided_differences_retain_degeneracies_and_small_full_band_tails() {
    for energy in [-3.0, -0.1, 0.0, 0.5] {
        let expected = fermi(energy, 0.0, 0.1).unwrap().1;
        let response = divided_difference(energy, energy + 1e-10, 0.0, 0.1).unwrap();
        assert!((response / expected - 1.0).abs() < 1e-8);
    }
    let response = divided_difference(-4.0, -3.9, 0.0, 0.1).unwrap();
    let x: f64 = 39.0;
    let y: f64 = 40.0;
    let expected = -(x.exp() - y.exp()) / ((1.0 + x.exp()) * (1.0 + y.exp()) * (-0.1));
    assert!(response < 0.0);
    assert!((response / expected - 1.0).abs() < 1e-12);
}

#[test]
fn finite_q_density_is_rectangular_and_never_symmetrized() {
    let bra = BandSpace::integer(vec![0.2, 0.6, 0.9], 0).unwrap();
    let ket = BandSpace::integer(vec![-0.3, 0.1], 1).unwrap();
    let p = SpectralPropagator::new(
        vec![BandPair {
            bra,
            ket,
            weight: 1.0,
        }],
        [0.2, 0.0, 0.0],
        Frequency::Retarded {
            omega: 0.1,
            damping: 0.04,
        },
        true,
    )
    .unwrap();
    assert_eq!(p.shapes(), [(3, 2)]);
    let source = vec![Complex::new(0.7, -0.2); 6];
    let density = p.apply(&source).unwrap();
    assert!((density[0] - source[0] / Complex::new(-0.4, 0.04)).abs() < 1e-12);
    assert_eq!(density[1], Complex::ZERO);
    assert_eq!(density[3], Complex::ZERO);
    assert_eq!(density[5], Complex::ZERO);
    let grand = SpectralPropagator::new(
        vec![BandPair {
            bra: BandSpace::integer(vec![0.2, 0.6, 0.9], 0).unwrap(),
            ket: BandSpace::integer(vec![-0.3, 0.1], 1).unwrap(),
            weight: 1.0,
        }],
        [0.2, 0.0, 0.0],
        Frequency::Retarded {
            omega: 0.1,
            damping: 0.04,
        },
        false,
    )
    .unwrap();
    assert_eq!(density, grand.apply(&source).unwrap());
}

#[test]
fn number_constraint_is_mesh_global_and_uniform_shifts_are_null() {
    let a = BandSpace::thermal(vec![-0.15, 0.07], 0.0, 0.1).unwrap();
    let b = BandSpace::thermal(vec![-0.4, 0.2], 0.0, 0.1).unwrap();
    let p = SpectralPropagator::new(
        vec![
            BandPair {
                bra: a.clone(),
                ket: a,
                weight: 0.3,
            },
            BandPair {
                bra: b.clone(),
                ket: b,
                weight: 0.7,
            },
        ],
        [0.0; 3],
        Frequency::Static,
        true,
    )
    .unwrap();
    let source: Vec<_> = [0.2, 0.1, 0.1, -0.3, 0.5, -0.1, -0.1, 0.8]
        .into_iter()
        .map(Complex::from)
        .collect();
    let d = p.apply(&source).unwrap();
    assert!(((d[0] + d[3]) * 0.3 + (d[4] + d[7]) * 0.7).abs() < 1e-14);
    assert!((d[0] + d[3]).abs() > 1e-3);
    let uniform = p
        .apply(&[
            0.4.into(),
            0.0.into(),
            0.0.into(),
            0.4.into(),
            0.4.into(),
            0.0.into(),
            0.0.into(),
            0.4.into(),
        ])
        .unwrap();
    assert!(uniform.iter().all(|x| x.abs() < 1e-14));
}

struct DiagonalKernel(f64);
impl DensityKernel for DiagonalKernel {
    fn apply(&self, density: &[Complex]) -> Result<Vec<Complex>> {
        Ok(density.iter().map(|&x| x * self.0).collect())
    }
}

#[test]
fn screened_response_has_the_closed_form_interaction_pole() {
    let a = BandSpace::integer(vec![-0.2, 0.5], 1).unwrap();
    let p = SpectralPropagator::new(
        vec![BandPair {
            bra: a.clone(),
            ket: a,
            weight: 1.0,
        }],
        [0.0; 3],
        Frequency::Retarded {
            omega: 0.3,
            damping: 0.07,
        },
        false,
    )
    .unwrap();
    let source = [
        Complex::ZERO,
        Complex::new(0.4, -0.2),
        Complex::new(0.1, 0.5),
        Complex::ZERO,
    ];
    let bare = p.apply(&source).unwrap();
    let report = screened_response(
        &p,
        &DiagonalKernel(0.2),
        &source,
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    for i in [1, 2] {
        let coefficient = bare[i] / source[i];
        let expected = bare[i] / (Complex::ONE - coefficient * 0.2);
        assert!((report.amplitudes[i] - expected).abs() < 1e-12);
    }
}
