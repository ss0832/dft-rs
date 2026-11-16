// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::response::complex::Complex;
use dft_rs::DftError;
use std::f64::consts::PI;

fn values(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

#[test]
fn molecular_medium_driver_preserves_dipole_response_and_lorentz_limit() {
    let molecule = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    let molecular = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("frequency", 0.12)
        .with("frequency_damping", 0.008);
    let density = 0.002;
    let slab = molecular
        .clone()
        .with("medium_number_density", density)
        .with("slab_thickness", 1400.0)
        .with("incidence_angle", 0.38)
        .with("incident_refractive_index", 1.1)
        .with("exit_refractive_index", 1.3);
    let alpha = run(
        "dynamic-polarizability",
        &molecule,
        &Controls::from_value(&molecular).unwrap(),
    )
    .unwrap();
    let result = run(
        "molecular-medium-optics",
        &molecule,
        &Controls::from_value(&slab).unwrap(),
    )
    .unwrap();
    assert_eq!(result.get("converged").and_then(Value::as_bool), Some(true));
    let alpha_real = values(&alpha, "alpha_real");
    let alpha_imag = values(&alpha, "alpha_imag");
    assert_eq!(values(&result, "alpha_real"), alpha_real);
    assert_eq!(values(&result, "alpha_imag"), alpha_imag);
    let trace: Complex = [0, 4, 8].into_iter().fold(Complex::ZERO, |sum, k| {
        sum + Complex::new(alpha_real[k], alpha_imag[k])
    });
    let x = trace * (4.0 * PI * density / 9.0);
    let epsilon = (Complex::ONE + x * 2.0) / (Complex::ONE - x);
    for i in 0..9 {
        let expected = if i % 4 == 0 { epsilon } else { Complex::ZERO };
        let measured = Complex::new(
            values(&result, "relative_permittivity_real")[i],
            values(&result, "relative_permittivity_imag")[i],
        );
        assert!((measured - expected).abs() < 1e-14);
    }
    let reflectance = values(&result, "reflectance");
    let transmittance = values(&result, "transmittance");
    let absorptance = values(&result, "absorptance");
    for i in 0..2 {
        assert!(reflectance[i] > 0.0 && transmittance[i] > 0.0 && absorptance[i] > 0.0);
        assert!((reflectance[i] + transmittance[i] + absorptance[i] - 1.0).abs() < 1e-13);
    }
    assert!(values(&result, "flux_balance_residual")
        .iter()
        .all(|x| *x < 1e-13));
    println!("Molecular medium optics {}", result.to_json());
    let invalid = slab.with(
        "medium_orientation",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    );
    assert!(matches!(
        run(
            "molecular-medium-optics",
            &molecule,
            &Controls::from_value(&invalid).unwrap()
        ),
        Err(DftError::InvalidInput(_))
    ));
}
