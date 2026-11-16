// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The Skala learned functional on the common driver, with models of
//! pseudo-random weights: self-consistency for two sets of hyperparameters,
//! the Hellmann-Feynman relation between the field derivative of the energy
//! and the dipole (which holds only if the Kohn-Sham matrix is the density
//! derivative of the energy), and the refusals.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::skala::{
    random_tensors, read_tensors, write_tensors, SkalaInput, SkalaModel, SkalaSizes,
};

fn water() -> System {
    System::molecule(
        vec![8, 1, 1],
        vec![[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]],
    )
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("basis".to_string(), Value::Str("sto-3g".into())),
        ("grid_level".to_string(), Value::Int(2)),
        ("conv_tol".to_string(), Value::Float(1e-11)),
        ("conv_tol_dp".to_string(), Value::Float(1e-9)),
        ("max_iter".to_string(), Value::Int(300)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn model_file(label: &str, seed: u64, sizes: SkalaSizes) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("dft_rs_p100_{label}_{}.model", std::process::id()));
    write_tensors(&path, &random_tensors(seed, sizes)).unwrap();
    path
}

const SMALL: SkalaSizes = SkalaSizes {
    width: 10,
    channels: 3,
    shells: 2,
    correlation: 2,
    layers: 1,
    post: 3,
};
const LARGE: SkalaSizes = SkalaSizes {
    width: 16,
    channels: 4,
    shells: 3,
    correlation: 3,
    layers: 2,
    post: 5,
};

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn learned(path: &std::path::Path, extra: &[(&str, Value)]) -> Controls {
    let mut all = vec![
        ("xc", Value::Str("skala".into())),
        ("skala_model", Value::Str(path.to_str().unwrap().into())),
    ];
    all.extend(extra.iter().cloned());
    controls(&all)
}

#[test]
fn learned_functional_is_self_consistent_for_two_architectures() {
    let mut energies = Vec::new();
    for (label, sizes) in [("small", SMALL), ("large", LARGE)] {
        let path = model_file(label, 7, sizes);
        let report = run("energy", &water(), &learned(&path, &[])).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        let energy = number(&report, "energy");
        println!(
            "{label}: energy {energy:.10} Hartree in {} iterations, functional {:?}",
            number(&report, "scf_iterations"),
            report.get("functional")
        );
        assert!(energy.is_finite() && energy < -60.0);
        energies.push(energy);
    }
    assert!((energies[0] - energies[1]).abs() > 1e-6);
}

#[test]
fn field_derivative_of_the_energy_is_the_dipole() {
    let path = model_file("field", 19, LARGE);
    let field = |f: f64| Value::Array(vec![Value::Float(0.0), Value::Float(0.0), Value::Float(f)]);
    // The sign convention of the field term is taken from Hartree-Fock, for
    // which the relation is known to hold.
    let slope = |xc: &dyn Fn(&[(&str, Value)]) -> Controls, h: f64| -> f64 {
        let plus = number(
            &run("energy", &water(), &xc(&[("field", field(h))])).unwrap(),
            "energy",
        );
        let minus = number(
            &run("energy", &water(), &xc(&[("field", field(-h))])).unwrap(),
            "energy",
        );
        (plus - minus) / (2.0 * h)
    };
    let dipole = |xc: &dyn Fn(&[(&str, Value)]) -> Controls| -> f64 {
        let report = run("dipole", &water(), &xc(&[])).unwrap();
        report.get("dipole").unwrap().numeric_array().unwrap().1[2]
    };
    let hartree_fock = |extra: &[(&str, Value)]| {
        let mut all = vec![("xc", Value::Str("hf".into()))];
        all.extend(extra.iter().cloned());
        controls(&all)
    };
    let reference = (slope(&hartree_fock, 2e-3), dipole(&hartree_fock));
    let sign = if (reference.0 + reference.1).abs() < (reference.0 - reference.1).abs() {
        -1.0
    } else {
        1.0
    };
    println!(
        "Hartree-Fock: dE/dF {:.10}, dipole {:.10}, convention dE/dF = {sign} dipole",
        reference.0, reference.1
    );
    assert!((reference.0 - sign * reference.1).abs() < 1e-5);

    let with_model = |extra: &[(&str, Value)]| learned(&path, extra);
    let analytic = sign * dipole(&with_model);
    let steps = [8e-3, 4e-3, 2e-3, 1e-3];
    let slopes: Vec<f64> = steps.iter().map(|&h| slope(&with_model, h)).collect();
    let errors: Vec<f64> = slopes.iter().map(|s| (s - analytic).abs()).collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let richardson = ((4.0 * slopes[3] - slopes[2]) / 3.0 - analytic).abs();
    let _ = std::fs::remove_file(&path);
    println!(
        "learned functional: dipole derivative {analytic:.10}, field steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson difference {richardson:e}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
    assert!(richardson < 2e-6);
}

#[test]
fn custom_pt2_preserves_the_complete_learned_parent_and_field_response() {
    let path = model_file("pt2_field", 19, LARGE);
    let system = water();
    let field = |value: f64| {
        Value::Array(vec![
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(value),
        ])
    };
    let settings = |strength: f64, coefficient: Option<f64>| {
        let mut values = vec![
            ("field", field(strength)),
            ("origin", Value::Str("zero".into())),
            ("conv_tol", Value::Float(1e-13)),
            ("conv_tol_dp", Value::Float(1e-12)),
            ("conv_tol_grad", Value::Float(1e-11)),
        ];
        if let Some(coefficient) = coefficient {
            values.extend([
                ("pt2_same_spin", Value::Float(coefficient)),
                ("pt2_opposite_spin", Value::Float(1.5 * coefficient)),
            ]);
        }
        learned(&path, &values)
    };
    // The dipole entry also includes the full energy report, so the same
    // calculation checks both zero-correlation limits without a second SCF.
    let parent = run("dipole", &system, &settings(0.0, None)).unwrap();
    let zero = run("dipole", &system, &settings(0.0, Some(0.0))).unwrap();
    let defect = (number(&zero, "energy") - number(&parent, "energy")).abs();
    println!("learned zero-PT2 energy defect={defect:.12e}");
    assert!(defect < 1e-10);
    let xc = number(
        parent.get("energy_components").unwrap(),
        "exchange_correlation",
    );
    assert!(xc.abs() > 1e-3);
    assert!((number(&zero, "nonsemilocal_exchange_correlation_energy") - xc).abs() < 1e-10);
    assert!(number(&zero, "semilocal_component_sum_residual") < 1e-10);
    let parent_mu = parent.get("dipole").unwrap().numeric_array().unwrap().1;
    let zero_mu = zero.get("dipole").unwrap().numeric_array().unwrap().1;
    for (a, b) in parent_mu.iter().zip(&zero_mu) {
        assert!((a - b).abs() < 1e-10);
    }
    let correlated = run("dipole", &system, &settings(0.0, Some(0.2))).unwrap();
    let derivative = -correlated.get("dipole").unwrap().numeric_array().unwrap().1[2];
    let steps = [8e-3, 4e-3, 2e-3, 1e-3];
    let mut estimates = Vec::new();
    for h in steps {
        // The energy entry does not consume an origin control.
        let energy = |strength| {
            let mut value = settings(strength, Some(0.2)).to_value();
            if let Value::Object(entries) = &mut value {
                entries.retain(|(key, _)| key != "origin");
            }
            number(
                &run("energy", &system, &Controls::from_value(&value).unwrap()).unwrap(),
                "energy",
            )
        };
        estimates.push((energy(h) - energy(-h)) / (2.0 * h));
    }
    let errors: Vec<_> = estimates
        .iter()
        .map(|value| (value - derivative).abs())
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - derivative).abs();
    println!("learned correlated dipole: steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}; response={:?} amplitudes={:?}", fit.exponent, fit.r_squared, correlated.get("response_residuals"), correlated.get("amplitude_response_residuals"));
    let _ = std::fs::remove_file(&path);
    assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
    assert!(richardson < 1e-6);
}

#[test]
fn derivative_features_and_inconsistent_inputs_are_refused() {
    let path = model_file("refusal", 3, SMALL);
    // The model file and the functional name are given together.
    let missing = run(
        "energy",
        &water(),
        &controls(&[("xc", Value::Str("skala".into()))]),
    )
    .unwrap_err();
    assert!(missing.to_string().contains("skala_model"), "{missing}");
    let stray = run(
        "energy",
        &water(),
        &controls(&[
            ("xc", Value::Str("pbe".into())),
            ("skala_model", Value::Str(path.to_str().unwrap().into())),
        ]),
    )
    .unwrap_err();
    assert!(stray.to_string().contains("skala"), "{stray}");
    let _ = std::fs::remove_file(&path);
    // A file that is not a model is named.
    let bogus =
        std::env::temp_dir().join(format!("dft_rs_p100_bogus_{}.model", std::process::id()));
    std::fs::write(&bogus, b"not a model").unwrap();
    let error = run("energy", &water(), &learned(&bogus, &[])).unwrap_err();
    let _ = std::fs::remove_file(&bogus);
    assert!(error.to_string().contains("Skala model file"), "{error}");
}

#[test]
fn nuclear_gradient_passes_a_four_step_fit_of_the_energy() {
    let path = model_file("gradient", 31, LARGE);
    let system = |atom: usize, axis: usize, h: f64| {
        let mut positions = vec![[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
        positions[atom][axis] += h;
        System::molecule(vec![8, 1, 1], positions)
    };
    let report = run("gradient", &system(0, 0, 0.0), &learned(&path, &[])).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (_, gradient) = report.get("gradient").unwrap().numeric_array().unwrap();
    let mut sum = [0.0_f64; 3];
    for atom in 0..3 {
        for axis in 0..3 {
            sum[axis] += gradient[3 * atom + axis];
        }
    }
    println!("gradient {gradient:?}, translational sum {sum:?}");
    assert!(sum.iter().all(|s| s.abs() < 1e-7));
    let energy = |atom: usize, axis: usize, h: f64| {
        number(
            &run("energy", &system(atom, axis, h), &learned(&path, &[])).unwrap(),
            "energy",
        )
    };
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for (atom, axis) in [(1usize, 1usize), (0, 2)] {
        let analytic = gradient[3 * atom + axis];
        let slopes: Vec<f64> = steps
            .iter()
            .map(|&h| (energy(atom, axis, h) - energy(atom, axis, -h)) / (2.0 * h))
            .collect();
        let errors: Vec<f64> = slopes.iter().map(|s| (s - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * slopes[3] - slopes[2]) / 3.0 - analytic).abs();
        println!(
            "learned functional gradient ({atom},{axis}): analytic {analytic:.10}, steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(richardson < 5e-6);
    }
    let _ = std::fs::remove_file(&path);
}

/// The polarizability from the coupled-perturbed equations with the kernel
/// of the learned functional against the field derivative of the dipole.
#[test]
fn polarizability_is_the_field_derivative_of_the_dipole() {
    let path = model_file("polarizability", 41, LARGE);
    let tight = [
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
    ];
    let report = run("polarizability", &water(), &learned(&path, &tight)).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (shape, alpha) = report
        .get("polarizability")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, vec![3, 3]);
    let dipole = |axis: usize, h: f64| -> Vec<f64> {
        let mut field = [0.0_f64; 3];
        field[axis] = h;
        let mut extra = tight.to_vec();
        extra.push((
            "field",
            Value::Array(field.iter().map(|v| Value::Float(*v)).collect()),
        ));
        let report = run("dipole", &water(), &learned(&path, &extra)).unwrap();
        report.get("dipole").unwrap().numeric_array().unwrap().1
    };
    println!("polarizability {alpha:?}");
    let steps = [8e-3, 4e-3, 2e-3, 1e-3];
    // alpha_ab = d mu_a / d F_b for the diagonal elements and one
    // off-diagonal pair of the molecular plane.
    for (a, b) in [(0usize, 0usize), (1, 1), (2, 2), (1, 2)] {
        let analytic = alpha[3 * a + b];
        let slopes: Vec<f64> = steps
            .iter()
            .map(|&h| (dipole(b, h)[a] - dipole(b, -h)[a]) / (2.0 * h))
            .collect();
        let errors: Vec<f64> = slopes.iter().map(|s| (s - analytic).abs()).collect();
        let richardson = ((4.0 * slopes[3] - slopes[2]) / 3.0 - analytic).abs();
        if analytic.abs() < 1e-8 {
            // Zero by symmetry: there is no error to fit.
            println!("polarizability ({a},{b}): analytic {analytic:e}, slopes {slopes:?}");
            assert!(slopes.iter().all(|s| s.abs() < 1e-7));
            continue;
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "polarizability ({a},{b}): analytic {analytic:.10}, field steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(richardson < 5e-6);
    }
    let _ = std::fs::remove_file(&path);
}

/// The analytic Hessian and the dipole derivatives against central
/// differences of the analytic gradient and of the dipole.
#[test]
fn hessian_is_the_derivative_of_the_analytic_gradient() {
    let path = model_file("hessian", 43, LARGE);
    let tight = [
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
    ];
    let system = |atom: usize, axis: usize, h: f64| {
        let mut positions = vec![[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
        positions[atom][axis] += h;
        System::molecule(vec![8, 1, 1], positions)
    };
    let report = run("hessian", &system(0, 0, 0.0), &learned(&path, &tight)).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (shape, hessian) = report.get("hessian").unwrap().numeric_array().unwrap();
    assert_eq!(shape, vec![9, 9]);
    let (shape, apt) = report
        .get("dipole_derivatives")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, vec![3, 9]);
    println!(
        "Hessian: raw asymmetry {:e}, translation residual {:?}",
        number(&report, "raw_asymmetry"),
        report.get("translation_residual")
    );
    assert!(number(&report, "raw_asymmetry") < 1e-6);
    let at = |atom: usize, axis: usize, h: f64| -> (Vec<f64>, Vec<f64>) {
        let gradient = run("gradient", &system(atom, axis, h), &learned(&path, &tight))
            .unwrap()
            .get("gradient")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let dipole = run("dipole", &system(atom, axis, h), &learned(&path, &tight))
            .unwrap()
            .get("dipole")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        (gradient, dipole)
    };
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for (atom, axis) in [(0usize, 2usize), (1, 1)] {
        let y = 3 * atom + axis;
        let mut hessian_errors = Vec::new();
        let mut apt_errors = Vec::new();
        let mut columns = Vec::new();
        for &h in &steps {
            let (plus, minus) = (at(atom, axis, h), at(atom, axis, -h));
            let column: Vec<f64> = (0..9)
                .map(|x| (plus.0[x] - minus.0[x]) / (2.0 * h))
                .collect();
            hessian_errors.push(
                (0..9)
                    .map(|x| (column[x] - hessian[9 * x + y]).abs())
                    .fold(0.0_f64, f64::max),
            );
            apt_errors.push(
                (0..3)
                    .map(|x| ((plus.1[x] - minus.1[x]) / (2.0 * h) - apt[9 * x + y]).abs())
                    .fold(0.0_f64, f64::max),
            );
            columns.push(column);
        }
        let hessian_fit = convergence::error_power_fit(&steps, &hessian_errors);
        let apt_fit = convergence::error_power_fit(&steps, &apt_errors);
        let richardson = (0..9)
            .map(|x| ((4.0 * columns[3][x] - columns[2][x]) / 3.0 - hessian[9 * x + y]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "coordinate ({atom},{axis}): steps {steps:?} Hessian column errors {hessian_errors:?} p={:.6} R2={:.8} Richardson difference {richardson:e}; dipole derivative errors {apt_errors:?} p={:.6} R2={:.8}",
            hessian_fit.exponent, hessian_fit.r_squared, apt_fit.exponent, apt_fit.r_squared
        );
        assert!((hessian_fit.exponent - 2.0).abs() < 0.1 && hessian_fit.r_squared > 0.999);
        assert!((apt_fit.exponent - 2.0).abs() < 0.1 && apt_fit.r_squared > 0.999);
        assert!(richardson < 2e-6);
    }
    let _ = std::fs::remove_file(&path);
}

/// All singlet excitations of the full response problem against the static
/// polarizability: `alpha_ab = 2 sum_n mu_a^n mu_b^n / omega_n`. The two
/// sides are different solutions (an eigenvalue problem and linear
/// equations) built on the same kernel of the learned functional.
#[test]
fn excitations_sum_to_the_static_polarizability() {
    let path = model_file("excitations", 47, LARGE);
    let tight = [
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
    ];
    let report = run("polarizability", &water(), &learned(&path, &tight)).unwrap();
    let (_, alpha) = report
        .get("polarizability")
        .unwrap()
        .numeric_array()
        .unwrap();
    // Water in a minimal basis: five occupied and two virtual orbitals.
    let mut extra = tight.to_vec();
    extra.push(("td_method", Value::Str("rpa".into())));
    extra.push(("nstates", Value::Int(10)));
    extra.push(("td_tolerance", Value::Float(1e-9)));
    extra.push(("td_max_iterations", Value::Int(200)));
    let report = run("excitations", &water(), &learned(&path, &extra)).unwrap();
    let states = report.get("states").unwrap().as_array().unwrap();
    assert_eq!(states.len(), 10);
    let mut sum = [[0.0_f64; 3]; 3];
    let mut energies = Vec::new();
    for state in states {
        let omega = number(state, "energy");
        let mu = state
            .get("transition_dipole")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        energies.push(omega);
        for a in 0..3 {
            for b in 0..3 {
                sum[a][b] += 2.0 * mu[a] * mu[b] / omega;
            }
        }
    }
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            worst = worst.max((sum[a][b] - alpha[3 * a + b]).abs());
        }
    }
    println!(
        "excitation energies {energies:?}; sum over states {sum:?}; polarizability {alpha:?}; largest difference {worst:e}"
    );
    assert!(energies.iter().all(|e| *e > 0.0));
    assert!(worst < 1e-6);
    let _ = std::fs::remove_file(&path);
}

/// The calculations that are built from the energy, the Kohn-Sham matrices
/// and the kernel run with the learned functional.
#[test]
fn derived_calculations_run_with_the_learned_functional() {
    let path = model_file("derived", 53, SMALL);
    let radical = [("charge", Value::Int(1)), ("multiplicity", Value::Int(2))];
    let hot = [("electronic_temperature", Value::Float(5000.0))];
    let rank = [("max_rank", Value::Int(1))];
    let broadened = [("broadening_ev", Value::Float(0.3))];
    let cases: [(&str, &[(&str, Value)]); 8] = [
        ("frequencies", &[]),
        ("multipole-gradient", &rank),
        ("response-save", &[]),
        ("ecd-spectrum", &broadened),
        ("conceptual-dft", &[]),
        ("spin-conceptual-dft", &radical),
        ("hyperfine", &radical),
        ("free-energy", &hot),
    ];
    let mut failures = Vec::new();
    for (feature, extra) in cases {
        match run(feature, &water(), &learned(&path, extra)) {
            Ok(report) => {
                let converged = report.get("converged").and_then(Value::as_bool);
                println!("{feature}: converged {converged:?}");
                if converged == Some(false) {
                    failures.push(format!("{feature}: not converged"));
                }
            }
            Err(error) => failures.push(format!("{feature}: {error}")),
        }
    }
    // The dipole row of the multipole gradient is the atomic polar tensor
    // of the Hessian calculation.
    let hessian = run("hessian", &water(), &learned(&path, &[])).unwrap();
    let (_, apt) = hessian
        .get("dipole_derivatives")
        .unwrap()
        .numeric_array()
        .unwrap();
    let gradient = run("multipole-gradient", &water(), &learned(&path, &rank)).unwrap();
    let ranks = gradient.get("ranks").unwrap().as_array().unwrap();
    let (_, dipole_gradient) = ranks[1].get("gradient").unwrap().numeric_array().unwrap();
    let worst = apt
        .iter()
        .zip(&dipole_gradient)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("dipole gradient against the atomic polar tensor: largest difference {worst:e}");
    let _ = std::fs::remove_file(&path);
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(worst < 1e-8);
}

/// The nuclear Taylor series of the learned functional along a displacement:
/// its first coefficient is the analytic gradient along the displacement and
/// its second coefficient half the analytic Hessian along it.
#[test]
fn nuclear_taylor_coefficients_match_the_gradient_and_the_hessian() {
    use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    let path = model_file("taylor", 101, SMALL);
    let settings = [
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
    ];
    let system = water();
    let method = MolecularMethod::from_controls(&learned(&path, &settings)).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    let direction = vec![[0.3, -0.2, 0.5], [0.1, 0.4, -0.3], [-0.2, 0.1, 0.6]];
    let report = dft_rs::response::nuclear::nuclear_taylor(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        &direction,
        3,
        &dft_rs::response::ResponseSolverConfig::default(),
    )
    .unwrap();
    let taylor = &report.value;
    println!(
        "learned Taylor: converged {}, energy coefficients {:?}, reference energy error {:e}, reference Fock error {:e}, projector defects {:?}, stationarity defects {:?}, metric defects {:?}, particle number coefficients {:?}, warnings {:?}",
        report.converged,
        taylor.energy_coefficients,
        taylor.reference_energy_error,
        taylor.reference_fock_error,
        taylor.projector_defects,
        taylor.stationarity_defects,
        taylor.metric_defects,
        taylor.particle_number_coefficients,
        report.warnings
    );
    let flat: Vec<f64> = direction.iter().flatten().copied().collect();
    let gradient = run("gradient", &system, &learned(&path, &settings))
        .unwrap()
        .get("gradient")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let hessian = run("hessian", &system, &learned(&path, &settings))
        .unwrap()
        .get("hessian")
        .unwrap()
        .numeric_array()
        .unwrap()
        .1;
    let first: f64 = (0..9).map(|i| gradient[i] * flat[i]).sum();
    let second: f64 = (0..9)
        .map(|i| {
            (0..9)
                .map(|j| 0.5 * flat[i] * hessian[9 * i + j] * flat[j])
                .sum::<f64>()
        })
        .sum();
    println!(
        "learned Taylor: first coefficient {:.12} against gradient {first:.12} (difference {:e}); second coefficient {:.12} against Hessian {second:.12} (difference {:e})",
        taylor.energy_coefficients[1],
        taylor.energy_coefficients[1] - first,
        taylor.energy_coefficients[2],
        taylor.energy_coefficients[2] - second
    );
    let _ = std::fs::remove_file(&path);
    assert!((taylor.energy_coefficients[1] - first).abs() < 1e-8);
    assert!((taylor.energy_coefficients[2] - second).abs() < 1e-7);
    assert!(report.converged);
}

/// Raman tensors of the learned functional: the polarizability derivatives
/// along the normal modes against central differences of the analytic
/// polarizability.
#[test]
fn raman_tensors_are_the_mode_derivatives_of_the_polarizability() {
    let path = model_file("raman", 89, SMALL);
    let settings = [
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
        // The stationarity defects of the Taylor response stay near 4e-12
        // (recorded 3.9e-12, 5.7e-12, 1.2e-12 for orders one to three), the
        // roundoff of the network, so a request of 1e-12 is reported as not
        // converged.
        ("response_tolerance", Value::Float(1e-10)),
    ];
    let report = run("ir-raman", &water(), &learned(&path, &settings)).unwrap();
    println!(
        "learned Raman: warnings {:?}, permutation defect {:?}, activities {:?}",
        report.get("warnings"),
        report.get("raman_permutation_defect"),
        report.get("raman_activities")
    );
    // The `response_converged` bit is deliberately not asserted here. It
    // compares the mixed-path stationarity defects against the requested
    // tolerance, and for the learned functional those defects sit at the
    // solver-tolerance level: 2e-11 to 1.2e-10 over fresh SCF instances at
    // the requested 1e-10, so the bit flips with thread scheduling
    // (single-threaded runs converge cleanly). Gating on it would make the
    // test machine-dependent. Accuracy is carried instead by the
    // independently evaluated four-step fit below, whose Richardson gate at
    // 1e-6 is four orders of magnitude above this noise.
    let modes = report
        .get("normal_modes_cartesian")
        .unwrap()
        .as_array()
        .unwrap();
    let derivatives = report
        .get("polarizability_derivatives")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(modes.len(), derivatives.len());
    println!(
        "learned Raman: {} modes, activities {:?}, permutation defect {:e}",
        modes.len(),
        report
            .get("raman_activities")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1,
        number(&report, "raman_permutation_defect")
    );
    let to_au = 1.0 / (dft_rs::constants::BOHR_TO_ANGSTROM * dft_rs::constants::BOHR_TO_ANGSTROM);
    let alpha = |pattern: &[f64], h: f64| -> Vec<f64> {
        let mut system = water();
        for atom in 0..3 {
            for axis in 0..3 {
                system.positions[atom][axis] += h * pattern[3 * atom + axis];
            }
        }
        run("polarizability", &system, &learned(&path, &settings))
            .unwrap()
            .get("polarizability")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1
    };
    let steps = [0.04, 0.02, 0.01, 0.005];
    for k in [0, modes.len() - 1] {
        let (_, pattern) = modes[k].numeric_array().unwrap();
        let (_, derivative) = derivatives[k].numeric_array().unwrap();
        let slopes: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (alpha(&pattern, h), alpha(&pattern, -h));
                (0..9).map(|c| (plus[c] - minus[c]) / (2.0 * h)).collect()
            })
            .collect();
        let errors: Vec<f64> = slopes
            .iter()
            .map(|slope| {
                (0..9)
                    .map(|c| (slope[c] - derivative[c] * to_au).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = (0..9)
            .map(|c| ((4.0 * slopes[3][c] - slopes[2][c]) / 3.0 - derivative[c] * to_au).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "learned Raman: d alpha / dQ_{k}: steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
            "mode {k}"
        );
        assert!(richardson < 1e-6, "mode {k}: Richardson {richardson:e}");
    }
    let _ = std::fs::remove_file(&path);
}

/// Gradient of the lowest singlet Tamm-Dancoff state with the learned
/// functional against central differences of the total energy of that
/// state.
#[test]
fn excited_state_gradient_passes_a_four_step_fit_of_the_state_energy() {
    let path = model_file("tdgrad", 97, SMALL);
    let base = vec![
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-11)),
        ("nstates", Value::Int(4)),
    ];
    let mut request = base.clone();
    request.push(("root", Value::Int(1)));
    let report = run("td-state-gradient", &water(), &learned(&path, &request)).unwrap();
    let gradient = report.get("gradient").unwrap().numeric_array().unwrap().1;
    let mut excitation = base.clone();
    excitation.push(("td_method", Value::Str("tda".into())));
    // Total energy of the lowest state and its distance from the next one.
    let state_energy = |system: &System| -> (f64, f64) {
        let r = run("excitations", system, &learned(&path, &excitation)).unwrap();
        let states = r.get("states").unwrap().as_array().unwrap();
        (
            number(&r, "energy") + number(&states[0], "energy"),
            number(&states[1], "energy") - number(&states[0], "energy"),
        )
    };
    let (reference, gap) = state_energy(&water());
    println!(
        "learned excited state: energy {reference:.10}, gap to the next state {gap:e}, translation residual {:e}",
        number(&report, "translation_residual")
    );
    assert!((number(&report, "excited_state_energy") - reference).abs() < 1e-9);
    assert!(number(&report, "translation_residual") < 1e-6);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for component in [2usize, 4] {
        let (atom, axis) = (component / 3, component % 3);
        let slopes: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let at = |sign: f64| {
                    let mut system = water();
                    system.positions[atom][axis] += sign * h;
                    state_energy(&system).0
                };
                (at(1.0) - at(-1.0)) / (2.0 * h)
            })
            .collect();
        let errors: Vec<f64> = slopes
            .iter()
            .map(|s| (s - gradient[component]).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * slopes[3] - slopes[2]) / 3.0 - gradient[component]).abs();
        println!(
            "learned excited-state gradient component {component}: analytic {:.10} steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson difference {richardson:e}",
            gradient[component], fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
            "component {component}"
        );
        assert!(
            richardson < 1e-6,
            "component {component}: Richardson {richardson:e}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// Two hydrogen atoms in a triclinic cell, optionally strained by
/// `x -> (1 + epsilon) x`.
fn hydrogen_cell(displaced: Option<(usize, usize, f64)>, strain: [[f64; 3]; 3]) -> System {
    hydrogen_cell_at(vec![[0.3, 0.2, 0.1], [1.6, 0.5, 0.4]], displaced, strain)
}

/// The same cell with the two atoms at `positions` before the displacement
/// and the strain.
fn hydrogen_cell_at(
    mut positions: Vec<[f64; 3]>,
    displaced: Option<(usize, usize, f64)>,
    strain: [[f64; 3]; 3],
) -> System {
    let rows = [[6.0, 0.0, 0.0], [0.4, 6.4, 0.0], [0.2, 0.3, 7.0]];
    if let Some((atom, axis, h)) = displaced {
        positions[atom][axis] += h;
    }
    let strained = |r: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|a| r[a] + (0..3).map(|b| strain[a][b] * r[b]).sum::<f64>())
    };
    System {
        numbers: vec![1, 1],
        positions: positions.into_iter().map(strained).collect(),
        cell: Some(rows.map(strained)),
        pbc: [true; 3],
    }
}

fn periodic_controls(path: &std::path::Path, extra: &[(&str, Value)]) -> Controls {
    let mut all = vec![
        (
            "k_mesh",
            Value::Array(vec![Value::Int(2), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff", Value::Float(10.0)),
        ("grid_level", Value::Int(1)),
        ("grid_shell", Value::Int(1)),
        // The analytic forces and stress are first order in the residual of
        // the density: the tolerances keep that below the finite-difference
        // errors of the finest steps.
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-12)),
        ("max_iter", Value::Int(300)),
        ("guess", Value::Str("core".into())),
    ];
    all.extend(extra.iter().cloned());
    learned(path, &all)
}

/// Forces and stress of the learned functional in a periodic cell against
/// central differences of the self-consistent energy.
#[test]
fn periodic_forces_and_stress_pass_a_four_step_fit_of_the_energy() {
    let path = model_file("periodic", 59, LARGE);
    let none = [[0.0_f64; 3]; 3];
    let energy = |system: &System| -> f64 {
        number(
            &run("periodic-energy", system, &periodic_controls(&path, &[])).unwrap(),
            "energy",
        )
    };
    let report = run(
        "periodic-energy",
        &hydrogen_cell(None, none),
        &periodic_controls(
            &path,
            &[("forces", Value::Bool(true)), ("stress", Value::Bool(true))],
        ),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (_, forces) = report.get("forces").unwrap().numeric_array().unwrap();
    let (_, stress) = report.get("stress").unwrap().numeric_array().unwrap();
    let volume = number(&report, "volume");
    println!(
        "periodic learned functional: energy {:.10}, forces {forces:?}, translation residual {:e}, stress {stress:?}",
        number(&report, "energy"),
        number(&report, "translation_residual")
    );
    assert!(number(&report, "translation_residual") < 1e-7);

    let steps = [0.02, 0.01, 0.005, 0.0025];
    for (atom, axis) in [(1usize, 0usize), (0, 2)] {
        let analytic = -forces[3 * atom + axis];
        let slopes: Vec<f64> = steps
            .iter()
            .map(|&h| {
                (energy(&hydrogen_cell(Some((atom, axis, h)), none))
                    - energy(&hydrogen_cell(Some((atom, axis, -h)), none)))
                    / (2.0 * h)
            })
            .collect();
        let errors: Vec<f64> = slopes.iter().map(|s| (s - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * slopes[3] - slopes[2]) / 3.0 - analytic).abs();
        println!(
            "periodic gradient ({atom},{axis}): analytic {analytic:.10}, steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(richardson < 2e-6);
    }

    // sigma_ab = (1/V) dE/d(epsilon_ab), symmetric strains. Recorded on the
    // interval 0.008..0.001: errors 8.52e-8, 2.21e-8, 6.49e-9, 2.22e-9 for
    // the component (2,2) (p = 1.756, R2 = 0.9974). The analytic periodic
    // stress of this coarse cell differs from the energy derivative by
    // 1.1e-9 (3.4e-10 for (0,0)); the difference is the same for a semilocal
    // functional and does not change with the SCF tolerances, so the two
    // finest steps of that interval sit on it. The interval below is
    // resolved for the diagonal components. The shear component, whose
    // error on that floor is 3.5e-11, leaves the quadratic regime at the
    // largest strain instead (errors 9.18e-8, 4.36e-8, 1.08e-8, 2.73e-9 on
    // 0.032..0.004, p = 1.723, R2 = 0.9832) and is fitted one octave lower.
    let diagonal = [0.032, 0.016, 0.008, 0.004];
    let shear = [0.016, 0.008, 0.004, 0.002];
    for (a, b, strain_steps) in [(0usize, 0usize, diagonal), (2, 2, diagonal), (0, 1, shear)] {
        let analytic = stress[3 * a + b];
        let slopes: Vec<f64> = strain_steps
            .iter()
            .map(|&h| {
                let strained = |sign: f64| {
                    let mut epsilon = [[0.0_f64; 3]; 3];
                    epsilon[a][b] = sign * h;
                    epsilon[b][a] = sign * h;
                    energy(&hydrogen_cell(None, epsilon))
                };
                let count = if a == b { 1.0 } else { 2.0 };
                (strained(1.0) - strained(-1.0)) / (2.0 * h * count * volume)
            })
            .collect();
        let errors: Vec<f64> = slopes.iter().map(|s| (s - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&strain_steps, &errors);
        let richardson = ((4.0 * slopes[3] - slopes[2]) / 3.0 - analytic).abs();
        println!(
            "periodic stress ({a},{b}): analytic {analytic:.10}, strain steps {strain_steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(richardson < 2e-7);
    }

    // Calculations without their periodic derivatives are refused by name.
    let refused = run(
        "periodic-field-scf",
        &hydrogen_cell(None, none),
        &periodic_controls(&path, &[]),
    )
    .unwrap_err();
    assert!(refused.to_string().contains("Skala"), "{refused}");
    let _ = std::fs::remove_file(&path);
}

/// Gamma-point force constants of the learned functional against central
/// differences of the analytic forces.
#[test]
fn periodic_force_constants_pass_a_four_step_fit_of_the_forces() {
    let path = model_file("phonons", 71, LARGE);
    let none = [[0.0_f64; 3]; 3];
    let report = run(
        "phonons",
        &hydrogen_cell(None, none),
        &periodic_controls(&path, &[]),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let (_, raw) = report
        .get("raw_force_constants")
        .unwrap()
        .numeric_array()
        .unwrap();
    let forces = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "periodic-energy",
            &hydrogen_cell(Some((atom, axis, h)), none),
            &periodic_controls(&path, &[("forces", Value::Bool(true))]),
        )
        .unwrap();
        r.get("forces").unwrap().numeric_array().unwrap().1
    };
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for (atom, axis) in [(1usize, 0usize), (0, 2)] {
        let column = 3 * atom + axis;
        let fd: Vec<Vec<f64>> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (forces(atom, axis, h), forces(atom, axis, -h));
                (0..6)
                    .map(|row| -(plus[row] - minus[row]) / (2.0 * h))
                    .collect()
            })
            .collect();
        let errors: Vec<f64> = fd
            .iter()
            .map(|col| {
                (0..6)
                    .map(|row| (col[row] - raw[6 * row + column]).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let worst = (0..6)
            .map(|row| ((4.0 * fd[3][row] - fd[2][row]) / 3.0 - raw[6 * row + column]).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "periodic learned force constants: raw asymmetry {:e}; raw column ({atom},{axis}) steps {steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {worst:e}",
            number(&report, "raw_asymmetry"),
            fit.exponent,
            fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(worst < 1e-5);
    }
    let _ = std::fs::remove_file(&path);
}

/// Elastic response of the learned functional: the strain second derivatives
/// and the internal strain against central differences of the analytic
/// stress.
#[test]
fn periodic_elastic_terms_pass_four_step_fits_of_the_stress() {
    let path = model_file("elastic", 73, LARGE);
    let zero = [[0.0_f64; 3]; 3];
    // The overlap cutoff lies in a gap of the atom-pair image distances
    // (13.25 to 13.65 Bohr at the relaxed positions), so the strained and
    // displaced cells keep the image set of the reference. Recorded at
    // 14 Bohr, where pairs lie within 0.02 Bohr of the cutoff, column [0][0]
    // on 0.01..0.00125: errors 9.11e-5, 2.18e-4, 1.35e-4, 1.38e-6
    // (p = 1.882, R2 = 0.521); the strained cells gain and lose pairs.
    // Recorded at 10 Bohr, also in a gap: errors 1.05e-4, 2.35e-5, 5.68e-6,
    // 1.30e-6 (p = 2.107, R2 = 0.99989, Richardson difference 3.2e-7). The
    // orbital tails are cut at the overlap cutoff, where the most diffuse
    // primitive is 5e-8 at 10 Bohr and 5e-14 at 13.45 Bohr, and that cut is
    // not differentiated; the Richardson difference below is 1.2e-9.
    let settings = |extra: &[(&str, Value)]| {
        let mut all = vec![("overlap_cutoff", Value::Float(13.45))];
        all.extend(extra.iter().cloned());
        periodic_controls(&path, &all)
    };
    let flat =
        |v: &Value, key: &str| -> Vec<f64> { v.get(key).unwrap().numeric_array().unwrap().1 };
    let tensor = |v: &Value, key: &str| -> [[f64; 3]; 3] {
        let data = flat(v, key);
        std::array::from_fn(|a| std::array::from_fn(|b| data[3 * a + b]))
    };
    // The relaxed-ion constants need a dynamically stable reference: the
    // atoms are first relaxed in the fixed cell.
    let relaxed = run(
        "opt-fixed-cell",
        &hydrogen_cell(None, zero),
        &settings(&[("gtol", Value::Float(1e-6))]),
    )
    .unwrap();
    assert_eq!(
        relaxed
            .get("optimization_converged")
            .and_then(Value::as_bool),
        Some(true)
    );
    let base: Vec<[f64; 3]> = flat(&relaxed, "positions")
        .chunks(3)
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    println!("periodic learned elastic: relaxed positions {base:?}");
    let report = run(
        "elastic",
        &hydrogen_cell_at(base.clone(), None, zero),
        &settings(&[]),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    let w = flat(&report, "strain_hessian");
    let lambda = flat(&report, "internal_strain");
    let s0 = tensor(&report, "strain_derivative");
    println!(
        "periodic learned elastic: raw asymmetry {:e}, internal-strain asymmetry {:e}, rotation residual {:e}, force-constant asymmetry {:e}, clamped-ion (GPa) {:?}",
        number(&report, "raw_asymmetry"),
        number(&report, "internal_strain_asymmetry"),
        number(&report, "rotation_residual"),
        number(&report, "force_constant_asymmetry"),
        flat(&report, "clamped_ion_gpa")
    );
    let first = |eps: [[f64; 3]; 3], displaced: Option<(usize, usize, f64)>| -> [[f64; 3]; 3] {
        tensor(
            &run(
                "stress",
                &hydrogen_cell_at(base.clone(), displaced, eps),
                &settings(&[]),
            )
            .unwrap(),
            "strain_derivative",
        )
    };
    let worst_of = |m: &[[f64; 3]; 3], expected: &dyn Fn(usize, usize) -> f64| -> f64 {
        let mut worst = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                worst = worst.max((m[a][b] - expected(a, b)).abs());
            }
        }
        worst
    };
    let strain_steps = [0.01, 0.005, 0.0025, 0.00125];
    for (c, d) in [(0usize, 0usize), (2, 1)] {
        let fd: Vec<[[f64; 3]; 3]> = strain_steps
            .iter()
            .map(|&h| {
                let mut ep = zero;
                ep[c][d] = h;
                let mut em = zero;
                em[c][d] = -h;
                let (plus, minus) = (first(ep, None), first(em, None));
                std::array::from_fn(|a| {
                    std::array::from_fn(|b| (plus[a][b] - minus[a][b]) / (2.0 * h))
                })
            })
            .collect();
        // W_ab,cd + delta_bc dE/d(eps_ad): the composed strain.
        let expected = |a: usize, b: usize| {
            w[(3 * a + b) * 9 + 3 * c + d] + if b == c { s0[a][d] } else { 0.0 }
        };
        let errors: Vec<f64> = fd.iter().map(|m| worst_of(m, &expected)).collect();
        let fit = convergence::error_power_fit(&strain_steps, &errors);
        let extrapolated: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| (4.0 * fd[3][a][b] - fd[2][a][b]) / 3.0)
        });
        let richardson = worst_of(&extrapolated, &expected);
        println!(
            "periodic learned strain Hessian column [{c}][{d}]: steps {strain_steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {richardson:e}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
            "[{c}][{d}]"
        );
        assert!(richardson < 2e-5, "[{c}][{d}] Richardson {richardson:e}");
    }
    let displacement_steps = [0.02, 0.01, 0.005, 0.0025];
    let (atom, axis) = (1usize, 0usize);
    let column = 3 * atom + axis;
    let fd: Vec<[[f64; 3]; 3]> = displacement_steps
        .iter()
        .map(|&h| {
            let (plus, minus) = (
                first(zero, Some((atom, axis, h))),
                first(zero, Some((atom, axis, -h))),
            );
            std::array::from_fn(|a| std::array::from_fn(|b| (plus[a][b] - minus[a][b]) / (2.0 * h)))
        })
        .collect();
    let expected = |a: usize, b: usize| lambda[(3 * a + b) * 6 + column];
    let errors: Vec<f64> = fd.iter().map(|m| worst_of(m, &expected)).collect();
    let fit = convergence::error_power_fit(&displacement_steps, &errors);
    let extrapolated: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|b| (4.0 * fd[3][a][b] - fd[2][a][b]) / 3.0));
    let richardson = worst_of(&extrapolated, &expected);
    println!(
        "periodic learned internal strain column ({atom},{axis}): steps {displacement_steps:?} errors {errors:?} p={:.6} R2={:.8}; Richardson max difference {richardson:e}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
    assert!(richardson < 2e-5, "Richardson {richardson:e}");
    assert!(number(&report, "raw_asymmetry") < 1e-6);
    assert!(number(&report, "internal_strain_asymmetry") < 1e-6);
    let _ = std::fs::remove_file(&path);
}

/// Electron-phonon vertex of the learned functional: its band-diagonal
/// elements are the displacement derivatives of the band energies.
#[test]
fn periodic_vertex_diagonal_matches_band_energy_derivatives() {
    let path = model_file("vertex", 79, LARGE);
    let none = [[0.0_f64; 3]; 3];
    let flat =
        |v: &Value, key: &str| -> Vec<f64> { v.get(key).unwrap().numeric_array().unwrap().1 };
    // The overlap cutoff lies in a gap of the atom-pair image distances
    // (12.83 to 13.25 Bohr). Recorded for (k 1, band 0, x 3) on
    // 0.02..0.0025 at 14 Bohr, where pairs lie within 0.02 Bohr of the
    // cutoff: errors 1.05e-5, 1.71e-6, 1.41e-6, 1.73e-7 (p = 1.805,
    // R2 = 0.924). Recorded at 10 Bohr, where the orbital tails are cut at
    // 5e-8: errors 1.10e-5, 2.74e-6, 6.66e-7, 1.34e-7 (p = 2.113,
    // R2 = 0.99883).
    let settings = |extra: &[(&str, Value)]| {
        let mut all = vec![
            ("electronic_temperature", Value::Float(0.0)),
            ("overlap_cutoff", Value::Float(13.0)),
        ];
        all.extend(extra.iter().cloned());
        periodic_controls(&path, &all)
    };
    let report = run(
        "electron-phonon",
        &hydrogen_cell(None, none),
        &settings(&[("electron_smearing", Value::Float(0.02))]),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    // derivatives[spin][k][band][x], 2 k points, 2 bands, 6 displacements.
    let derivatives = flat(&report, "eigenvalue_derivatives");
    let index = |k: usize, band: usize, x: usize| ((k * 2) + band) * 6 + x;
    let bands = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
        let r = run(
            "electronic-structure",
            &hydrogen_cell(Some((atom, axis, h)), none),
            &settings(&[]),
        )
        .unwrap();
        let channels = r.get("channels").unwrap().as_array().unwrap();
        flat(&channels[0], "band_energies")
    };
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for (k, band, atom, axis) in [(1usize, 0usize, 1usize, 0usize), (0, 1, 0, 2)] {
        let x = 3 * atom + axis;
        let analytic = derivatives[index(k, band, x)];
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (bands(atom, axis, h), bands(atom, axis, -h));
                ((plus[2 * k + band] - minus[2 * k + band]) / (2.0 * h) - analytic).abs()
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "periodic learned vertex: d e/d x (k {k}, band {band}, x {x}): analytic {analytic:.10e} steps {steps:?} errors {errors:?} p={:.6} R2={:.8}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999,
            "k {k} band {band} x {x}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// Born effective charges and the electronic dielectric tensor of the
/// learned functional approach the atomic polar tensor and the
/// polarizability of the isolated molecule as the cell grows.
#[test]
fn periodic_born_charges_and_dielectric_tensor_approach_the_molecular_limit() {
    let path = model_file("born", 83, LARGE);
    let flat =
        |v: &Value, key: &str| -> Vec<f64> { v.get(key).unwrap().numeric_array().unwrap().1 };
    let numbers = vec![9, 1];
    let positions = vec![[0.0, 0.0, 0.0], [1.7328, 0.0, 0.0]];
    let settings = [
        ("grid_level", Value::Int(3)),
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
    ];
    let molecule = System::molecule(numbers.clone(), positions.clone());
    let mut with_rank = settings.to_vec();
    with_rank.push(("max_rank", Value::Int(1)));
    let gradient = run("multipole-gradient", &molecule, &learned(&path, &with_rank)).unwrap();
    let ranks = gradient.get("ranks").unwrap().as_array().unwrap();
    // gradient[a][3 atom + b] = d mu_a / d R_(atom, b).
    let apt = flat(&ranks[1], "gradient");
    let alpha = flat(
        &run("polarizability", &molecule, &learned(&path, &settings)).unwrap(),
        "polarizability",
    );
    println!("periodic learned Born: molecular APT {apt:?}\nperiodic learned Born: molecular polarizability {alpha:?}");
    let edges = [14.0, 16.0, 18.0, 20.0];
    let mut z_errors = Vec::new();
    let mut alpha_errors = Vec::new();
    for &edge in &edges {
        let mut extra = settings.to_vec();
        extra.push((
            "k_mesh",
            Value::Array(vec![Value::Int(1), Value::Int(1), Value::Int(1)]),
        ));
        extra.push(("overlap_cutoff", Value::Float(24.0)));
        extra.push(("grid_shell", Value::Int(1)));
        extra.push(("max_iter", Value::Int(200)));
        let report = run(
            "born-charges",
            &System {
                numbers: numbers.clone(),
                positions: positions.clone(),
                cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
                pbc: [true; 3],
            },
            &learned(&path, &extra),
        )
        .unwrap();
        let z = flat(&report, "born_charges");
        let eps = flat(&report, "epsilon_infinity");
        let volume = edge * edge * edge;
        let mut z_error = 0.0_f64;
        for atom in 0..2 {
            for a in 0..3 {
                for b in 0..3 {
                    z_error =
                        z_error.max((z[9 * atom + 3 * a + b] - apt[6 * a + 3 * atom + b]).abs());
                }
            }
        }
        let mut a_error = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                let periodic_alpha = (eps[3 * a + b] - if a == b { 1.0 } else { 0.0 }) * volume
                    / (4.0 * std::f64::consts::PI);
                a_error = a_error.max((periodic_alpha - alpha[3 * a + b]).abs());
            }
        }
        println!(
            "periodic learned Born: L = {edge}: |Z* - APT| {z_error:e}, |alpha_periodic - alpha| {a_error:e}, sum rule {:?}",
            flat(&report, "born_sum_rule")
        );
        z_errors.push(z_error);
        alpha_errors.push(a_error);
    }
    let h: Vec<f64> = edges.iter().map(|l| 1.0 / l).collect();
    for (label, errors) in [
        ("Born charges", &z_errors),
        ("polarizability", &alpha_errors),
    ] {
        let fit = convergence::error_power_fit(&h, errors);
        println!(
            "periodic learned Born: {label}: h = 1/L {h:?} errors {errors:?} p={:.6} R2={:.8}",
            fit.exponent, fit.r_squared
        );
        assert!(
            (fit.exponent - 3.0).abs() < 0.15 && fit.r_squared > 0.999,
            "{label}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// A molecule in a large cell has the energy of the isolated molecule: the
/// atomic sums of the non-local layers are the same and only the partition
/// of the quadrature differs.
#[test]
fn periodic_energy_of_an_isolated_molecule_is_the_molecular_energy() {
    let path = model_file("limit", 67, LARGE);
    let positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]];
    let settings = [
        ("grid_level", Value::Int(3)),
        ("conv_tol", Value::Float(1e-11)),
        ("conv_tol_dp", Value::Float(1e-9)),
    ];
    let molecular = number(
        &run(
            "energy",
            &System::molecule(vec![1, 1], positions.clone()),
            &learned(&path, &settings),
        )
        .unwrap(),
        "energy",
    );
    let mut extra = settings.to_vec();
    extra.push(("overlap_cutoff", Value::Float(14.0)));
    extra.push(("guess", Value::Str("core".into())));
    let edge = 24.0;
    let periodic = number(
        &run(
            "periodic-energy",
            &System {
                numbers: vec![1, 1],
                positions,
                cell: Some([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
                pbc: [true; 3],
            },
            &learned(&path, &extra),
        )
        .unwrap(),
        "energy",
    );
    println!(
        "isolated molecule {molecular:.10}, in a cell of edge {edge} Bohr {periodic:.10}, difference {:e}",
        periodic - molecular
    );
    let _ = std::fs::remove_file(&path);
    assert!((periodic - molecular).abs() < 5e-5);
}

/// The official checkpoint against the official implementation. Both files
/// live outside the package: the model converted by
/// `tools/convert_skala_model.py` and the reference written by
/// `tools/oracle/skala_reference.py`.
#[test]
fn official_checkpoint_reproduces_the_reference_energy_density() {
    let (Ok(model_path), Ok(oracle_path)) = (
        std::env::var("DFT_RS_SKALA_MODEL"),
        std::env::var("DFT_RS_SKALA_ORACLE"),
    ) else {
        println!("skipped: set DFT_RS_SKALA_MODEL and DFT_RS_SKALA_ORACLE to run the official comparison");
        return;
    };
    let model = SkalaModel::load(std::path::Path::new(&model_path)).unwrap();
    println!("{}", model.description());
    let oracle = read_tensors(std::path::Path::new(&oracle_path)).unwrap();
    let get = |name: &str| -> &Vec<f64> { &oracle[name].data };
    let triples =
        |data: &[f64]| -> Vec<[f64; 3]> { data.chunks(3).map(|c| [c[0], c[1], c[2]]).collect() };
    let n = get("weights").len();
    let points = triples(get("points"));
    let centers = triples(get("centers"));
    let atom_of: Vec<usize> = get("atom_of").iter().map(|v| *v as usize).collect();
    let gradient = triples(get("gradient"));
    let (rho, tau) = (get("rho"), get("tau"));
    let out = model
        .evaluate(
            &SkalaInput {
                atom_of: &atom_of,
                points: &points,
                centers: &centers,
                weights: get("weights"),
                atomic_weights: get("atomic_weights"),
                rho: [&rho[..n], &rho[n..]],
                gradient: [&gradient[..n], &gradient[n..]],
                tau: [&tau[..n], &tau[n..]],
            },
            false,
        )
        .unwrap();
    let reference = get("energy_density");
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = out
        .energy_density
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let total = get("energy")[0];
    println!(
        "official energy {total:.10}, dft-rs {:.10}, difference {:e}; largest energy-density difference {worst:e} (largest value {scale:e}) on {n} points",
        out.energy,
        out.energy - total
    );
    assert!((out.energy - total).abs() < 2e-5 * total.abs());
    assert!(worst < 1e-4 * scale);
}

/// Self-consistent calculation with the official checkpoint against the
/// official self-consistent calculation of the same molecule and basis
/// (water, def2-SVP).
///
/// On the quadrature grid of the reference, read point by point from the
/// oracle file, the two programs solve the same discretized equations and
/// agree to the precision of the reference evaluation. On the grids of this
/// program the comparison measures quadrature error: the reference itself
/// changes by 1.4e-5 Hartree, 4.1e-4 Hartree/Bohr and 9.2e-5 atomic units of
/// dipole between its two finest grids (33698 and 90058 points), the second
/// of which supplies the constants below.
#[test]
fn official_checkpoint_reproduces_the_self_consistent_reference() {
    use dft_rs::basis::{data, MolBasis};
    use dft_rs::grid::MolecularGrid;
    use dft_rs::math::Vec3;
    use dft_rs::scf::{run_scf, ScfOptions};
    use dft_rs::system::{Atom, Molecule};
    use dft_rs::xc::ResolvedFunctional;
    let (Ok(model_path), Ok(oracle_path)) = (
        std::env::var("DFT_RS_SKALA_MODEL"),
        std::env::var("DFT_RS_SKALA_ORACLE"),
    ) else {
        println!("skipped: set DFT_RS_SKALA_MODEL and DFT_RS_SKALA_ORACLE to run the official comparison");
        return;
    };
    let path = std::path::Path::new(&model_path);
    let oracle = read_tensors(std::path::Path::new(&oracle_path)).unwrap();
    let get = |name: &str| -> &Vec<f64> { &oracle[name].data };

    // The same equations on the same grid.
    let molecule = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.43, 1.11),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.43, 1.11),
        },
    ]);
    let basis = MolBasis::build(&molecule, &data::load("def2-svp").unwrap()).unwrap();
    let grid = MolecularGrid {
        points: get("scf_points")
            .chunks(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect(),
        weights: get("scf_weights").clone(),
        raw_weights: get("scf_atomic_weights").clone(),
        atom_of: get("scf_atom_of").iter().map(|v| *v as usize).collect(),
    };
    let options = ScfOptions {
        skala: Some(std::sync::Arc::new(SkalaModel::load(path).unwrap())),
        use_soscf: false,
        e_tol: 1e-10,
        ..ScfOptions::default()
    };
    let functional = ResolvedFunctional::model_based("Skala learned functional (model file)");
    let scf = run_scf(&molecule, &basis, &options, &functional, Some(&grid)).unwrap();
    assert!(scf.converged);
    let dipole = dft_rs::properties::dipole_moment(&molecule, &basis, &scf);
    let reference_energy = get("scf_energy")[0];
    let reference_dipole = get("scf_dipole")[2];
    println!(
        "reference grid ({} points): energy {:.10} (official {reference_energy:.10}, difference {:e}), exchange-correlation energy {:.10} (official {:.10}), dipole z {:.9} (official {reference_dipole:.9}, difference {:e}), {} iterations",
        grid.len(),
        scf.energy,
        scf.energy - reference_energy,
        scf.e_xc,
        get("scf_exchange_correlation")[0],
        dipole.z,
        dipole.z - reference_dipole,
        scf.iterations
    );
    println!(
        "official gradient on the reference grid {:?}",
        get("scf_gradient")
    );
    assert!((scf.energy - reference_energy).abs() < 2e-6);
    assert!((dipole.z - reference_dipole).abs() < 2e-5);

    // The grids of this program against the finest grid of the reference.
    const ENERGY: f64 = -76.3218112619;
    const DIPOLE_Z: f64 = 0.776564855;
    const GRADIENT: [f64; 9] = [
        0.0,
        0.0,
        1.1391891120e-2,
        0.0,
        -1.0063855644e-3,
        -5.6959456110e-3,
        0.0,
        1.0063853334e-3,
        -5.6959455088e-3,
    ];
    let settings = |level: i64| {
        learned(
            path,
            &[
                ("basis", Value::Str("def2-svp".into())),
                ("grid_level", Value::Int(level)),
            ],
        )
    };
    let mut finest = (f64::INFINITY, f64::INFINITY);
    for level in [3, 5] {
        let report = run("gradient", &water(), &settings(level)).unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        let energy = number(&report, "energy");
        let (_, gradient) = report.get("gradient").unwrap().numeric_array().unwrap();
        let worst = gradient
            .iter()
            .zip(GRADIENT)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "grid level {level}: energy {energy:.10} (official {ENERGY:.10}, difference {:e}), gradient {gradient:?}, largest gradient difference {worst:e}",
            energy - ENERGY
        );
        finest = ((energy - ENERGY).abs(), worst);
    }
    let report = run("dipole", &water(), &settings(5)).unwrap();
    let (_, dipole) = report.get("dipole").unwrap().numeric_array().unwrap();
    println!(
        "grid level 5: dipole {dipole:?} (official z {DIPOLE_Z:.9}, difference {:e})",
        dipole[2] - DIPOLE_Z
    );
    assert!(finest.0 < 1e-4, "energy difference {:e}", finest.0);
    assert!(finest.1 < 1e-3, "gradient difference {:e}", finest.1);
    assert!((dipole[2] - DIPOLE_Z).abs() < 5e-4);
}
