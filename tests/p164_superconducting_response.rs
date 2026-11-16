// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent analytic two-level Bogoliubov oracles and canonical limits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::response::complex::Complex;
use dft_rs::response::superconducting::{
    anomalous_factor, critical_temperature, select_band_phase, solve_gap_taylor, BandPhase,
    GapFunctional, GapFunctionalData, GapState, GapTaylor, Lm2005BandFunctional,
    NormalBandSpectrum, PhononBandFunctional, SuppliedBandFunctional,
};
use dft_rs::response::ResponseSolverConfig;
#[path = "support/mass_oracle.rs"]
mod mass_oracle;
#[path = "support/pairing_oracle.rs"]
mod pairing_oracle;
use dft_rs::properties::pairing::Lm2005OffDiagonal;
type Jet = Taylor<5>;

/// A finite base value must not hide a nonanalytic derivative at the Fermi level.
struct NonanalyticMass;
impl GapFunctional for NonanalyticMass {
    fn bands(&self) -> usize {
        1
    }
    fn evaluate<S: Scalar + Send + Sync>(
        &self,
        parameters: &[S],
        mu: S,
        real: &[S],
        imaginary: &[S],
    ) -> dft_rs::Result<GapFunctionalData<S>> {
        let mut data =
            SuppliedBandFunctional { bands: 1 }.evaluate(parameters, mu, real, imaginary)?;
        let xi = mu - data.energies[0];
        data.mass_renormalization[0] = data.mass_renormalization[0] + (xi * xi).sqrt();
        Ok(data)
    }
}

#[test]
fn nonfinite_provider_derivatives_are_not_accepted_as_zero_residuals() {
    let parameters = [0.0, 1.0, 0.0, 0.004, 0.0].map(Jet::from_f64);
    let seed = GapState {
        gap: vec![Complex::ZERO],
        chemical_potential: 0.0,
    };
    let result = solve_gap_taylor(
        &NonanalyticMass,
        &parameters,
        Jet::from_f64(1.0),
        &seed,
        0,
        &cfg(),
    );
    assert!(
        result.is_err(),
        "a NaN Goldstone action must not pass a convergence gate"
    );
}

fn oracle(t: Jet) -> (Vec<Jet>, Jet, [Jet; 5]) {
    let f = Jet::from_f64;
    let gap = f(0.02) / (f(1.0) - t.scale(0.2));
    let xi = f(0.007) * (f(1.0) + t.scale(0.3));
    let kt = f(0.004) * (f(1.0) + t.scale(0.15));
    let mass = f(0.2) + t.scale(0.04);
    let energy = f(0.03) / (f(1.0) - t.scale(0.12));
    let e = (xi * xi + gap * gap).sqrt();
    let exponential = (e / kt).exp();
    let q = (exponential - f(1.0)) / (exponential + f(1.0)) / e.scale(2.0);
    let number = f(1.0) - (xi * q).scale(2.0);
    let kernel = -(f(1.0) + mass) / q;
    (
        vec![energy, f(1.0), mass, kt, kernel],
        number,
        [gap, energy - xi, q * gap, number, e * e],
    )
}
fn cfg() -> ResponseSolverConfig {
    ResponseSolverConfig {
        tolerance: 1e-13,
        ..Default::default()
    }
}

#[test]
fn canonical_phase_selection_matches_the_single_band_free_energy_on_both_sides() {
    let functional = SuppliedBandFunctional { bands: 1 };
    let (w, z, kt, gap): (f64, f64, f64, f64) = (0.73, 0.2, 0.004, 0.01);
    let effective_g = 2.0 * gap / (w * (gap / (2.0 * kt)).tanh());
    let kernel = -effective_g * (1.0 + z);
    let cold = select_band_phase(&functional, &[0.031, w, z, kt, kernel], w, &cfg()).unwrap();
    let expected = gap * gap / effective_g - 2.0 * w * kt * (gap / (2.0 * kt)).cosh().ln();
    assert_eq!(cold.phase, BandPhase::Superconducting);
    assert!((cold.solution.gap_real[0][0].abs() - gap).abs() < 1e-14);
    assert!((cold.solution.chemical_potential[0] - 0.031).abs() < 1e-14);
    assert!((cold.condensation_free_energy - expected).abs() < 1e-14);
    assert!((cold.pairing_energy + gap * gap / effective_g).abs() < 1e-14);
    assert!(cold.pairing_stability_margin.abs() < 1e-12);
    assert!(cold.solution.gap_residuals[0] < 1e-14);
    assert!(cold.solution.number_residuals[0] < 1e-14);
    let kt_hot = 0.009;
    let hot = select_band_phase(&functional, &[0.031, w, z, kt_hot, kernel], w, &cfg()).unwrap();
    let hot_margin = 1.0 - w * effective_g / (4.0 * kt_hot);
    assert_eq!(hot.phase, BandPhase::Normal);
    assert!((hot.pairing_stability_margin - hot_margin).abs() < 1e-14);
    assert_eq!(hot.condensation_free_energy, 0.0);
    assert_eq!(hot.solution.gap_real[0], vec![0.0]);
    println!(
        "canonical one-band phase: gap={:.16e}, dF={:.16e}, margin(cold/hot)=({:.6e},{:.6e}), gap/number residual=({:.6e},{:.6e})",
        cold.solution.gap_real[0][0],
        cold.condensation_free_energy,
        cold.pairing_stability_margin,
        hot.pairing_stability_margin,
        cold.solution.gap_residuals[0],
        cold.solution.number_residuals[0],
    );
}

#[test]
fn canonical_phase_selection_resolves_unequal_weights_and_a_sign_changing_gap() {
    let (kt, mu, gap, z): (f64, f64, f64, f64) = (0.004, -0.002, 0.014, 0.2);
    let eps = [-0.016, 0.0032, 0.009];
    let w = [0.27, 0.52, 0.45];
    let v = [-1.0, 0.7, 1.3];
    let excitation: [f64; 3] = std::array::from_fn(|i| (eps[i] - mu).hypot(gap * v[i]));
    let tanh: [f64; 3] = excitation.map(|e| (e / (2.0 * kt)).tanh());
    let number = (0..3)
        .map(|i| w[i] * (1.0 - (eps[i] - mu) * tanh[i] / excitation[i]))
        .sum();
    let effective_g = 1.0
        / (0..3)
            .map(|i| w[i] * v[i] * v[i] * tanh[i] / (2.0 * excitation[i]))
            .sum::<f64>();
    let kernel = v
        .iter()
        .flat_map(|&vi| v.map(|vj| -effective_g * (1.0 + z) * (vi * vj)));
    let parameters: Vec<_> = eps
        .into_iter()
        .chain(w)
        .chain([z; 3])
        .chain([kt])
        .chain(kernel)
        .collect();
    let functional = SuppliedBandFunctional { bands: 3 };
    let result = select_band_phase(&functional, &parameters, number, &cfg()).unwrap();
    assert_eq!(result.phase, BandPhase::Superconducting);
    let actual = &result.solution.gap_real[0];
    let phase = actual[2].signum();
    for (&actual, &vi) in actual.iter().zip(&v) {
        assert!((phase * actual - gap * vi).abs() < 2e-14);
    }
    assert!((result.solution.chemical_potential[0] - mu).abs() < 2e-14);
    assert!(result.condensation_free_energy < 0.0);
    assert!((result.pairing_energy + gap * gap / effective_g).abs() < 2e-14);
    assert!(result.pairing_stability_margin.abs() < 1e-12);
    let mut shifted = parameters.clone();
    for energy in &mut shifted[..3] {
        *energy += 0.123;
    }
    let translated = select_band_phase(&functional, &shifted, number, &cfg()).unwrap();
    assert!((translated.condensation_free_energy - result.condensation_free_energy).abs() < 2e-14);
    assert!((translated.solution.chemical_potential[0] - 0.123 - mu).abs() < 2e-14);
    let mut unequal_mass = parameters;
    unequal_mass[6] += 0.01;
    assert!(matches!(
        select_band_phase(&functional, &unequal_mass, number, &cfg()),
        Err(dft_rs::DftError::Unsupported { .. })
    ));
    println!(
        "canonical three-band phase: gap={:?}, mu={:.16e}, dF={:.16e}, margin={:.6e}, gap/number residual=({:.6e},{:.6e})",
        result.solution.gap_real[0],
        result.solution.chemical_potential[0],
        result.condensation_free_energy,
        result.pairing_stability_margin,
        result.solution.gap_residuals[0],
        result.solution.number_residuals[0],
    );
}

#[test]
fn normal_pairing_onset_has_closed_single_band_limits_and_gauge_covariance() {
    let band = SuppliedBandFunctional { bands: 1 };
    let weight = 0.73;
    let mass = 0.29;
    let kernel = -0.031;
    for filling in [0.27_f64, 0.5, 0.81] {
        let x = ((1.0 - filling) / filling).ln();
        let prefactor = if x == 0.0 {
            0.25
        } else {
            (1.0 - 2.0 * filling) / (2.0 * x)
        };
        let expected = -kernel * weight * prefactor / (1.0 + mass);
        for energy in [0.032, 0.074] {
            let result = critical_temperature(
                &band,
                NormalBandSpectrum {
                    energies: &[energy],
                    weights: &[weight],
                    electron_number: 2.0 * weight * filling,
                },
                [0.4 * expected, 1.6 * expected],
                &cfg(),
                |kt| vec![energy, weight, mass, kt, kernel],
            )
            .unwrap();
            assert!((result.temperature_energy - expected).abs() < 2e-14);
            assert!((result.chemical_potential - (energy - expected * x)).abs() < 2e-14);
            assert!((result.occupations[0] - 2.0 * filling).abs() < 1e-13);
            assert_eq!(result.gap_mode, [1.0]);
            assert!(result.gap_residual < 1e-13);
            assert!(result.eigen_residual < 1e-13);
            assert!(result.number_residual < 1e-13);
            println!(
                "SCDFT one band filling={filling} energy={energy} kTc={:.16e} oracle={expected:.16e} mu={:.16e} gap residual={:.3e} number residual={:.3e}",
                result.temperature_energy,
                result.chemical_potential,
                result.gap_residual,
                result.number_residual,
            );
        }
    }
}

#[test]
fn weighted_pairing_onset_matches_the_original_two_band_characteristic_equation() {
    let energies = [-0.013, 0.023];
    let weights = [0.35, 0.71];
    let mass = [0.23, 0.41];
    let expected_kt = 0.006;
    let expected_mu = -0.005;
    let scale = |energy: f64, weight: f64| {
        let x = (energy - expected_mu) / expected_kt;
        let f = 1.0 / (1.0 + x.exp());
        (
            2.0 * weight * f,
            weight * (1.0 - 2.0 * f) / (2.0 * (energy - expected_mu)),
        )
    };
    let (n0, wq0) = scale(energies[0], weights[0]);
    let (n1, wq1) = scale(energies[1], weights[1]);
    let base = [-0.025, -0.021, -0.021, 0.003];
    let a = -base[0] * wq0 / (1.0 + mass[0]);
    let b = -base[1] * wq1 / (1.0 + mass[0]);
    let c = -base[2] * wq0 / (1.0 + mass[1]);
    let d = -base[3] * wq1 / (1.0 + mass[1]);
    // This solves the quadratic characteristic equation of the original
    // nonsymmetric pairing map, bypassing the weighted symmetric transform.
    let lambda = 0.5 * (a + d + ((a - d).powi(2) + 4.0 * b * c).sqrt());
    let kernel = base.map(|k| k / lambda);
    let provider = SuppliedBandFunctional { bands: 2 };
    let normal = NormalBandSpectrum {
        energies: &energies,
        weights: &weights,
        electron_number: n0 + n1,
    };
    let parameters_at = |kt| {
        energies
            .into_iter()
            .chain(weights)
            .chain(mass)
            .chain([kt])
            .chain(kernel)
            .collect()
    };
    let result =
        critical_temperature(&provider, normal, [0.003, 0.012], &cfg(), parameters_at).unwrap();
    let expected_ratio = (lambda - a) / b;
    assert!((result.temperature_energy - expected_kt).abs() < 1e-14);
    assert!((result.chemical_potential - expected_mu).abs() < 1e-14);
    assert!((result.gap_mode[1] / result.gap_mode[0] - expected_ratio).abs() < 1e-12);
    assert!(result.gap_residual < 1e-13);
    assert!(result.eigen_residual < 1e-13);
    assert!(result.number_residual < 1e-13);
    println!(
        "SCDFT weighted two band kTc={:.16e} mu={:.16e} mode={:?} lambda={:.16e} gap residual={:.3e} eigen residual={:.3e} number residual={:.3e} bracket={:?}",
        result.temperature_energy,
        result.chemical_potential,
        result.gap_mode,
        result.leading_eigenvalue,
        result.gap_residual,
        result.eigen_residual,
        result.number_residual,
        result.temperature_energy_bracket,
    );
    let error =
        critical_temperature(&provider, normal, [0.010, 0.012], &cfg(), parameters_at).unwrap_err();
    assert!(matches!(error, dft_rs::DftError::InvalidInput(message)
        if message.contains("SCDFT critical temperature is not bracketed")));
}

#[test]
fn canonical_number_keeps_the_chemical_potential_inside_a_deep_gap() {
    let energies = [-0.6, -0.3, 0.05, 0.3];
    let weights = [0.4_f64, 0.5, 0.7, 0.6];
    let expected_kt = 1e-4;
    // The nearest occupied/unoccupied levels govern the number balance.
    // Every farther level is suppressed by at least exp(-2500).
    let expected_mu =
        0.5 * (energies[1] + energies[2] + expected_kt * (weights[1] / weights[2]).ln());
    let kernel_at_onset = -2.0 * (expected_mu - energies[1]) / weights[1];
    let result = critical_temperature(
        &SuppliedBandFunctional { bands: 4 },
        NormalBandSpectrum {
            energies: &energies,
            weights: &weights,
            electron_number: 2.0 * (weights[0] + weights[1]),
        },
        [0.7e-4, 1.5e-4],
        &cfg(),
        |kt| {
            let mut kernel = [0.0; 16];
            kernel[5] = kernel_at_onset * expected_kt / kt;
            energies
                .into_iter()
                .chain(weights)
                .chain([0.0; 4])
                .chain([kt])
                .chain(kernel)
                .collect()
        },
    )
    .unwrap();
    assert!((result.temperature_energy - expected_kt).abs() < 2e-16);
    assert!((result.chemical_potential - expected_mu).abs() < 1e-14);
    assert!(result.gap_residual < 1e-13);
    assert_eq!(result.occupations, [2.0, 2.0, 0.0, 0.0]);
    println!(
        "SCDFT deep gap kTc={:.16e} mu={:.16e} oracle_mu={expected_mu:.16e} gap residual={:.3e} number residual={:.3e}",
        result.temperature_energy,
        result.chemical_potential,
        result.gap_residual,
        result.number_residual,
    );
}

fn lm_parameters(t: Jet) -> (Vec<Jet>, Jet, [Jet; 5]) {
    let f = Jet::from_f64;
    let (base, number, expected) = oracle(t);
    let energy = base[0];
    let weight = f(1.0);
    let kt = base[3];
    let xi = energy - expected[1];
    let phonon = f(0.025) / (f(1.0) - t.scale(0.12));
    let coulomb = f(0.005) * (f(1.0) + t.scale(0.03));
    let q = expected[2] / expected[0];
    let k = pairing_oracle::factor(xi, xi, phonon, kt);
    let z = mass_oracle::factor(xi, xi, phonon, kt);
    // Both generated Z and K depend on the same vertex and responding mu.
    let vertex = -(f(1.0) + coulomb * q) / (weight * (z + k * q));
    assert!(vertex.c[0] > 0.0);
    (
        vec![energy, weight, kt, phonon, vertex, coulomb],
        number,
        expected,
    )
}

#[test]
fn generated_phonon_mass_and_gap_number_phase_response_share_one_jacobian() {
    let (parameters, number, expected) = lm_parameters(Jet::from_coefficients(&[0.0, 1.0]));
    let initial = GapState {
        gap: vec![Complex::new(0.0144, 0.0192)],
        chemical_potential: 0.025,
    };
    let provider = Lm2005BandFunctional {
        bands: 1,
        modes: 1,
        quadrature: Lm2005OffDiagonal::default(),
    };
    let result = solve_gap_taylor(&provider, &parameters, number, &initial, 4, &cfg()).unwrap();
    let check = solve_gap_taylor(
        &Lm2005BandFunctional {
            quadrature: Lm2005OffDiagonal::new(16).unwrap(),
            ..provider
        },
        &parameters,
        number,
        &initial,
        4,
        &cfg(),
    )
    .unwrap();
    for k in 0..5 {
        assert!((result.gap_real[k][0] - 0.6 * expected[0].c[k]).abs() < 1e-11);
        assert!((result.gap_imaginary[k][0] - 0.8 * expected[0].c[k]).abs() < 1e-11);
        assert!((result.chemical_potential[k] - expected[1].c[k]).abs() < 1e-11);
        assert!((result.gap_real[k][0] - check.gap_real[k][0]).abs() < 1e-11);
        assert!(
            (result.mass_renormalization_coefficients[k][0]
                - check.mass_renormalization_coefficients[k][0])
                .abs()
                < 1e-10
        );
    }
    check_fits("generated LM mass/gap", &observables(&result));
    println!("SCDFT generated LM raw gap={:?} number={:?} phase={:?} multiplier={:?} Goldstone={:?} linear={:?} mass={:?}",
        result.gap_residuals,result.number_residuals,result.phase_residuals,result.phase_multipliers,result.goldstone_defects,result.linear_residuals,result.mass_renormalization_coefficients);
}

fn observables(result: &GapTaylor) -> [Vec<f64>; 5] {
    [
        (0..5)
            .map(|k| 0.6 * result.gap_real[k][0] + 0.8 * result.gap_imaginary[k][0])
            .collect::<Vec<_>>(),
        result.chemical_potential.clone(),
        (0..5)
            .map(|k| 0.6 * result.anomalous_real[k][0] + 0.8 * result.anomalous_imaginary[k][0])
            .collect(),
        (0..5).map(|k| result.occupations[k][0]).collect(),
        (0..5)
            .map(|k| result.quasiparticle_energy_squared[k][0])
            .collect(),
    ]
}

fn check_fits(name: &str, actual: &[Vec<f64>; 5]) {
    let steps = [0.08, 0.04, 0.02, 0.01];
    for (component, label) in [
        "gap",
        "chemical potential",
        "anomalous density",
        "occupation",
        "quasiparticle squared",
    ]
    .iter()
    .enumerate()
    {
        for order in 1..=4 {
            let target = actual[component][order] * order as f64;
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let plus = oracle(Jet::from_coefficients(&[h, 1.0])).2[component];
                    let minus = oracle(Jet::from_coefficients(&[-h, 1.0])).2[component];
                    (plus.c[order - 1] - minus.c[order - 1]) / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("SCDFT {name} {label} order={order} target={target:.15e} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
            assert!(rich < 1e-7);
        }
    }
}

#[test]
fn canonical_complex_gap_number_phase_and_four_order_response_match_closed_oracles() {
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let (p, number, expected) = oracle(t);
    let seed = GapState {
        gap: vec![Complex::new(0.0144, 0.0192)],
        chemical_potential: 0.025,
    };
    let result = solve_gap_taylor(
        &SuppliedBandFunctional { bands: 1 },
        &p,
        number,
        &seed,
        4,
        &cfg(),
    )
    .unwrap();
    let actual = observables(&result);
    for (component, (values, expected)) in actual.iter().zip(expected).enumerate() {
        for n in 0..5 {
            assert!(
                (values[n] - expected.c[n]).abs() < 1e-11,
                "component={component} order={n}: {} vs {}",
                values[n],
                expected.c[n]
            );
        }
    }
    println!(
        "SCDFT gap raw gap={:?} number={:?} phase={:?} multiplier={:?} Goldstone={:?} linear={:?}",
        result.gap_residuals,
        result.number_residuals,
        result.phase_residuals,
        result.phase_multipliers,
        result.goldstone_defects,
        result.linear_residuals
    );
    check_fits("supplied", &actual);
    // A global seed phase rotates Delta/chi and leaves mu/number/spectrum.
    let real_seed = GapState {
        gap: vec![Complex::new(0.024, 0.0)],
        chemical_potential: 0.025,
    };
    let real = solve_gap_taylor(
        &SuppliedBandFunctional { bands: 1 },
        &p,
        number,
        &real_seed,
        4,
        &cfg(),
    )
    .unwrap();
    for k in 0..5 {
        assert!((result.gap_real[k][0] - 0.6 * real.gap_real[k][0]).abs() < 1e-12);
        assert!((result.gap_imaginary[k][0] - 0.8 * real.gap_real[k][0]).abs() < 1e-12);
        assert!((result.chemical_potential[k] - real.chemical_potential[k]).abs() < 1e-12);
    }
}

#[test]
fn normal_fermi_surface_factor_and_particle_hole_limits_are_analytic() {
    let f = Jet::from_f64;
    let e2 = Jet::from_coefficients(&[0.0, 1.0]);
    let kt = 0.7_f64;
    let actual = anomalous_factor(e2, f(kt)).unwrap();
    let coefficients = [
        1.0 / (4.0 * kt),
        -1.0 / (48.0 * kt.powi(3)),
        1.0 / (480.0 * kt.powi(5)),
        -17.0 / (80640.0 * kt.powi(7)),
        31.0 / (1451520.0 * kt.powi(9)),
    ];
    for n in 0..5 {
        assert!((actual.c[n] - coefficients[n]).abs() < 1e-13);
    }
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let thermal = f(0.004) * (f(1.0) + t.scale(0.3));
    let factor = anomalous_factor(f(0.0), thermal).unwrap();
    let steps = [0.08, 0.04, 0.02, 0.01];
    for n in 1..=4 {
        let target = n as f64 * factor.c[n];
        let estimates: Vec<_> = steps
            .iter()
            .map(|&h| {
                let eval = |x| {
                    (f(0.25) / (f(0.004) * (f(1.0) + Jet::from_coefficients(&[x, 1.0]).scale(0.3))))
                        .c[n - 1]
                };
                (eval(h) - eval(-h)) / (2.0 * h)
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
        println!("SCDFT Fermi thermal order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-6);
    }
    let seed = GapState {
        gap: vec![Complex::ZERO],
        chemical_potential: 0.03,
    };
    let params = vec![f(0.03) + t.scale(0.002), f(1.0), f(0.2), f(0.004), f(0.0)];
    let normal = solve_gap_taylor(
        &SuppliedBandFunctional { bands: 1 },
        &params,
        f(1.0),
        &seed,
        4,
        &cfg(),
    )
    .unwrap();
    assert_eq!(normal.gap_real, vec![vec![0.0]; 5]);
    assert!((normal.chemical_potential[1] - 0.002).abs() < 1e-14);
    assert!(normal.chemical_potential[2..]
        .iter()
        .all(|x| x.abs() < 1e-14));
    assert_eq!(normal.quasiparticle_energies, vec![0.0]);
    let (p, number, _) = oracle(t);
    let positive = GapState {
        gap: vec![Complex::new(0.024, 0.0)],
        chemical_potential: 0.025,
    };
    let original = solve_gap_taylor(
        &SuppliedBandFunctional { bands: 1 },
        &p,
        number,
        &positive,
        4,
        &cfg(),
    )
    .unwrap();
    let mut reflected = p.clone();
    reflected[0] = -reflected[0];
    let reflected_seed = GapState {
        chemical_potential: -positive.chemical_potential,
        ..positive
    };
    let reflected = solve_gap_taylor(
        &SuppliedBandFunctional { bands: 1 },
        &reflected,
        f(2.0) - number,
        &reflected_seed,
        4,
        &cfg(),
    )
    .unwrap();
    for k in 0..5 {
        assert!((original.gap_real[k][0] - reflected.gap_real[k][0]).abs() < 1e-12);
        assert!((original.chemical_potential[k] + reflected.chemical_potential[k]).abs() < 1e-12);
        assert!(
            (original.occupations[k][0] + reflected.occupations[k][0]
                - if k == 0 { 2.0 } else { 0.0 })
            .abs()
                < 1e-12
        );
    }
}

fn phonon_parameters(t: Jet) -> (Vec<Jet>, Jet, [Jet; 5]) {
    let (data, number, expected) = oracle(t);
    let f = Jet::from_f64;
    let xi = data[0] - expected[1];
    let w = f(0.025) * (f(1.0) + t.scale(0.08));
    let coulomb = f(0.005) * (f(1.0) + t.scale(0.06));
    let factor = pairing_oracle::factor(xi, xi, w, data[3]);
    let coupling = (data[4] - coulomb) / factor;
    let mut parameters = data[..4].to_vec();
    parameters.extend([w, coupling, coulomb]);
    (parameters, number, expected)
}

#[test]
fn phonon_kernel_chemical_potential_and_gap_response_have_independent_four_step_fits() {
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let (parameters, number, expected) = phonon_parameters(t);
    let provider = PhononBandFunctional {
        bands: 1,
        modes: 1,
        quadrature: Lm2005OffDiagonal::default(),
    };
    let initial = GapState {
        gap: vec![Complex::new(0.0144, 0.0192)],
        chemical_potential: 0.025,
    };
    let result = solve_gap_taylor(&provider, &parameters, number, &initial, 4, &cfg()).unwrap();
    let actual = observables(&result);
    for (values, expected) in actual.iter().zip(expected) {
        for n in 0..5 {
            assert!((values[n] - expected.c[n]).abs() < 1e-11);
        }
    }
    println!("SCDFT phonon raw gap={:?} number={:?} phase={:?} multiplier={:?} Goldstone={:?} linear={:?}",
        result.gap_residuals,result.number_residuals,result.phase_residuals,result.phase_multipliers,result.goldstone_defects,result.linear_residuals);
    check_fits("phonon", &actual);
    // A second quadrature order independently checks the complete coupled
    // state/path; no physical band, gap or resonance denominator is clipped.
    let alternate = PhononBandFunctional {
        quadrature: Lm2005OffDiagonal::new(16).unwrap(),
        ..provider
    };
    let check = solve_gap_taylor(&alternate, &parameters, number, &initial, 4, &cfg()).unwrap();
    let other = observables(&check);
    let error = actual
        .iter()
        .flatten()
        .zip(other.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("SCDFT phonon coupled quadrature max error={error:.12e}");
    assert!(error < 1e-11);
}

fn two_band_oracle(t: Jet) -> (Vec<Jet>, Jet, [Jet; 2], Jet) {
    let f = Jet::from_f64;
    let mu = f(0.023) / (f(1.0) - t.scale(0.1));
    let xi = [
        f(0.007) * (f(1.0) + t.scale(0.3)),
        f(-0.011) * (f(1.0) - t.scale(0.2)),
    ];
    let kt = f(0.004) * (f(1.0) + t.scale(0.15));
    let mass = [f(0.2) + t.scale(0.04), f(0.1) - t.scale(0.03)];
    let weights = [f(0.6) + t.scale(0.02), f(0.4) - t.scale(0.02)];
    let u = [1.0, 0.7];
    let amplitude = f(0.024) / (f(1.0) - t.scale(0.2));
    let gaps = std::array::from_fn(|i| amplitude.scale(u[i]) / (f(1.0) + mass[i]));
    let q: [Jet; 2] = std::array::from_fn(|i| {
        let e = (xi[i] * xi[i] + gaps[i] * gaps[i]).sqrt();
        let exponential = (e / kt).exp();
        (exponential - f(1.0)) / (exponential + f(1.0)) / e.scale(2.0)
    });
    let strength = (0..2)
        .fold(f(0.0), |s, i| {
            s + weights[i] * q[i].scale(u[i] * u[i]) / (f(1.0) + mass[i])
        })
        .recip();
    let number = (0..2).fold(f(0.0), |s, i| {
        s + weights[i] * (f(1.0) - (xi[i] * q[i]).scale(2.0))
    });
    let mut parameters = vec![
        mu + xi[0],
        mu + xi[1],
        weights[0],
        weights[1],
        mass[0],
        mass[1],
        kt,
    ];
    for a in u {
        for b in u {
            parameters.push(-strength.scale(a * b));
        }
    }
    (parameters, number, gaps, mu)
}

#[test]
fn coupled_two_band_complex_gap_and_number_have_independent_four_step_fits() {
    let t = Jet::from_coefficients(&[0.0, 1.0]);
    let (parameters, number, expected, mu) = two_band_oracle(t);
    let initial = GapState {
        gap: expected
            .iter()
            .map(|g| Complex::new(g.value() * 0.72, g.value() * 0.96))
            .collect(),
        chemical_potential: 0.025,
    };
    let result = solve_gap_taylor(
        &SuppliedBandFunctional { bands: 2 },
        &parameters,
        number,
        &initial,
        4,
        &cfg(),
    )
    .unwrap();
    for n in 0..5 {
        assert!((result.chemical_potential[n] - mu.c[n]).abs() < 1e-11);
        for i in 0..2 {
            assert!((result.gap_real[n][i] - 0.6 * expected[i].c[n]).abs() < 1e-11);
            assert!((result.gap_imaginary[n][i] - 0.8 * expected[i].c[n]).abs() < 1e-11);
        }
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    for band in 0..2 {
        for order in 1..=4 {
            let target = (0.6 * result.gap_real[order][band]
                + 0.8 * result.gap_imaginary[order][band])
                * order as f64;
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    (two_band_oracle(Jet::from_coefficients(&[h, 1.0])).2[band].c[order - 1]
                        - two_band_oracle(Jet::from_coefficients(&[-h, 1.0])).2[band].c[order - 1])
                        / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (x - target).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - target).abs();
            println!("SCDFT two-band gap band={band} order={order} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
            assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999 && rich < 1e-7);
        }
    }
    println!(
        "SCDFT two-band raw gap={:?} number={:?} phase={:?} Goldstone={:?}",
        result.gap_residuals,
        result.number_residuals,
        result.phase_residuals,
        result.goldstone_defects
    );
}
