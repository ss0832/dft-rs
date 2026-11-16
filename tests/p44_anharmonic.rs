// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::constants::{AMU_TO_AU_MASS, HARTREE_TO_CM1};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::response::anharmonic::{
    vibrational_spectrum, AnharmonicConfig, ObservablePolynomial, VibrationalPolynomial,
};

fn model(scale: f64) -> VibrationalPolynomial {
    VibrationalPolynomial {
        frequencies: vec![0.05],
        cubic: vec![scale * 0.00004],
        quartic: vec![scale * scale * 0.000005],
        dipole: ObservablePolynomial {
            components: 3,
            first: vec![0.7, 0.0, 0.0],
            second: vec![scale * 0.04, 0.0, 0.0],
            third: vec![scale * scale * 0.006, 0.0, 0.0],
        },
        polarizability: None,
        rotation: None,
    }
}
fn dense_oracle(model: &VibrationalPolynomial, size: usize) -> (Vec<f64>, Vec<f64>) {
    let omega = model.frequencies[0];
    let mut q = Matrix::zeros(size, size);
    for n in 1..size {
        q[(n, n - 1)] = (n as f64 / (2.0 * omega)).sqrt();
        q[(n - 1, n)] = q[(n, n - 1)];
    }
    let q2 = q.matmul_seq(&q);
    let q3 = q2.matmul_seq(&q);
    let q4 = q2.matmul_seq(&q2);
    let mut h = Matrix::zeros(size, size);
    let mut mu = Matrix::zeros(size, size);
    for i in 0..size {
        h[(i, i)] = (i as f64 + 0.5) * omega;
        for j in 0..size {
            h[(i, j)] += model.cubic[0] * q3[(i, j)] / 6.0 + model.quartic[0] * q4[(i, j)] / 24.0;
            mu[(i, j)] = model.dipole.first[0] * q[(i, j)]
                + model.dipole.second[0] * q2[(i, j)] / 2.0
                + model.dipole.third[0] * q3[(i, j)] / 6.0;
        }
    }
    let (energies, mut vectors) = symmetric_eigen(&h).unwrap();
    for col in 0..3 {
        if vectors[(col, col)] < 0.0 {
            for row in 0..size {
                vectors[(row, col)] = -vectors[(row, col)];
            }
        }
    }
    let transformed = vectors.transpose().matmul_seq(&mu).matmul_seq(&vectors);
    (energies, vec![transformed[(0, 1)], transformed[(0, 2)]])
}

#[test]
fn harmonic_limit_and_electrical_overtone_have_closed_form_strengths() {
    let mut input = model(0.0);
    input.dipole.second[0] = 0.03;
    input.polarizability = Some(ObservablePolynomial {
        components: 9,
        first: vec![0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2],
        second: vec![0.0; 9],
        third: vec![0.0; 9],
    });
    let result = vibrational_spectrum(&input, &Default::default()).unwrap();
    assert!(result.converged);
    assert_eq!(result.bands.len(), 2);
    assert!((result.bands[0].frequency_cm - 0.05 * HARTREE_TO_CM1).abs() < 1e-9);
    assert!((result.bands[0].transition_dipole[0] - 0.7 / (0.1_f64).sqrt()).abs() < 1e-12);
    assert!((result.bands[0].ir_intensity - 974.8801 * AMU_TO_AU_MASS * 0.7 * 0.7).abs() < 1e-8);
    let expected = 0.03 * 2.0_f64.sqrt() / (4.0 * 0.05);
    assert!((result.bands[1].transition_dipole[0] - expected).abs() < 1e-12);
    assert_eq!(result.bands[1].double_harmonic_ir_intensity, 0.0);
    assert_eq!(result.bands[1].character, "overtone");
    assert!(result.bands[1].ir_intensity > 0.0);
    let activity =
        45.0 * 0.2_f64.powi(2) * AMU_TO_AU_MASS * dft_rs::constants::BOHR_TO_ANGSTROM.powi(4);
    assert!((result.bands[0].raman_activity.unwrap() - activity).abs() < 1e-10);
    assert_eq!(result.bands[1].raman_activity, Some(0.0));
}

#[test]
fn cubic_quartic_energies_and_transformed_intensities_match_independent_oscillator_oracle() {
    let mut previous: Option<[f64; 3]> = None;
    // Measure the asymptotic regime; the unscaled polynomial is not a small perturbation.
    for scale in [0.1, 0.05, 0.025] {
        let input = model(scale);
        let report = vibrational_spectrum(&input, &Default::default()).unwrap();
        let (oracle, mu) = dense_oracle(&input, 40);
        let (larger, mu_larger) = dense_oracle(&input, 50);
        for root in 0..3 {
            assert!((oracle[root] - larger[root]).abs() < 1e-12);
        }
        for k in 0..2 {
            assert!((mu[k] - mu_larger[k]).abs() < 1e-11);
        }
        assert!(report.converged && report.added_resonant_states == 0);
        let energy_error = (report.zero_point_energy - oracle[0]).abs();
        let fundamental_error = (report.bands[0].transition_dipole[0] - mu[0]).abs();
        let overtone_error = (report.bands[1].transition_dipole[0] - mu[1]).abs();
        eprintln!(
            "scale {scale} ZPE error {energy_error:.12e}, fundamental moment error \
            {fundamental_error:.12e}, overtone moment error {overtone_error:.12e}"
        );
        if let Some(old) = previous {
            for (i, expected) in [16.0, 16.0, 8.0].iter().enumerate() {
                let current = [energy_error, fundamental_error, overtone_error][i];
                assert!(
                    (old[i] / current - expected).abs() < 1.5,
                    "order check {i}: {}",
                    old[i] / current
                );
            }
        }
        previous = Some([energy_error, fundamental_error, overtone_error]);
        let omega = 0.05_f64;
        let c = input.cubic[0];
        let q = input.quartic[0];
        let e0 = 0.5 * omega + q / (32.0 * omega.powi(2)) - 11.0 * c * c / (288.0 * omega.powi(4));
        assert!((report.zero_point_energy - e0).abs() < 5e-9);
        for band in &report.bands {
            for a in 0..3 {
                assert!(
                    (band.transition_dipole[a]
                        - band.double_harmonic_dipole[a]
                        - band.mechanical_dipole[a]
                        - band.electrical_dipole[a]
                        - band.mixed_dipole[a])
                        .abs()
                        < 1e-13
                );
            }
        }
    }
}

#[test]
fn exact_fermi_and_second_order_resonances_are_retained_without_denominator_regularization() {
    let mut cubic = vec![0.0; 8];
    for ix in [1, 2, 4] {
        cubic[ix] = 0.00001;
    }
    let input = VibrationalPolynomial {
        frequencies: vec![0.05, 0.1],
        cubic,
        quartic: vec![0.0; 16],
        dipole: ObservablePolynomial {
            components: 3,
            first: vec![0.5, 0.0, 0.0, 0.2, 0.0, 0.0],
            second: vec![0.0; 12],
            third: vec![0.0; 24],
        },
        polarizability: None,
        rotation: None,
    };
    let config = AnharmonicConfig {
        resonance_coupling_factor: 0.0,
        ..Default::default()
    };
    let result = vibrational_spectrum(&input, &config).unwrap();
    assert!(result.converged, "{:?}", result.warnings);
    assert!(result.added_resonant_states >= 3);
    assert!(result.polyad_occupations.contains(&vec![4, 0]));
    assert!(result.polyad_occupations.contains(&vec![2, 1]));
    assert!(result.minimum_external_denominator.unwrap() > 1e-3);
    let around: Vec<_> = result
        .bands
        .iter()
        .filter(|b| (b.frequency_cm - 0.1 * HARTREE_TO_CM1).abs() < 100.0)
        .collect();
    assert_eq!(around.len(), 2);
    assert!((around[0].frequency_cm - around[1].frequency_cm).abs() > 20.0);
    assert!(around.iter().any(|b| b.dominant_weight < 0.9));
    // Reversing a mode's phase changes odd derivative signs, never energies or strengths.
    let mut changed = input.clone();
    for (ix, v) in changed.cubic.iter_mut().enumerate() {
        let mut k = ix;
        let mut zeros = 0;
        for _ in 0..3 {
            zeros += usize::from(k % 2 == 0);
            k /= 2;
        }
        if zeros % 2 == 1 {
            *v = -*v;
        }
    }
    changed.dipole.first[0] = -changed.dipole.first[0];
    let other = vibrational_spectrum(&changed, &config).unwrap();
    for (a, b) in result.bands.iter().zip(&other.bands) {
        assert!((a.frequency_cm - b.frequency_cm).abs() < 1e-9);
        assert!((a.ir_intensity - b.ir_intensity).abs() < 1e-7);
    }
}

#[test]
fn imaginary_modes_asymmetric_tensors_and_resource_limits_do_not_produce_spectra() {
    let mut invalid = model(0.0);
    invalid.frequencies[0] = -0.05;
    assert!(vibrational_spectrum(&invalid, &Default::default()).is_err());
    assert!(vibrational_spectrum(
        &model(0.0),
        &AnharmonicConfig {
            max_polyad_states: 1,
            ..Default::default()
        }
    )
    .is_err());
    let config = AnharmonicConfig {
        memory_mb: 0,
        ..Default::default()
    };
    assert!(vibrational_spectrum(&model(0.0), &config).is_err());
    let asymmetric = VibrationalPolynomial {
        frequencies: vec![0.05, 0.07],
        cubic: vec![0.0, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        quartic: vec![0.0; 16],
        dipole: ObservablePolynomial {
            components: 3,
            first: vec![0.0; 6],
            second: vec![0.0; 12],
            third: vec![0.0; 24],
        },
        polarizability: None,
        rotation: None,
    };
    assert!(vibrational_spectrum(&asymmetric, &Default::default()).is_err());
}
