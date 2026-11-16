// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use dft_rs::driver::{run, Controls, System, Value};

fn real(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.get(key).unwrap().as_array().unwrap()
}

#[test]
fn water_oxygen_k_edge_spectrum_and_polarization() {
    // Approximate gas-phase water geometry, prescribed without optimizing.
    let coordinates = [[0.0, 0.0, 0.0], [1.809, 0.0, 0.0], [-0.452, 1.752, 0.0]];
    let polarization = [0.37, -0.81, 0.45];
    let system = System::molecule(vec![8, 1, 1], coordinates.to_vec());
    for (functional, basis) in [
        ("wb97x-d", "sto-3g"),
        ("wb97x-d", "6-31g"),
        ("wb97x-v", "sto-3g"),
        ("wb97m-v", "sto-3g"),
    ] {
        for model in ["cvs-tda", "relaxed-core-ion"] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("xc", functional)
                    .with("grid_level", 0usize)
                    .with("vv10_grid_level", 0usize)
                    .with("basis", basis)
                    .with("grid_weight_cutoff", 0.0)
                    .with("conv_tol_grad", 1e-9)
                    .with("absorber_atoms", vec![0usize])
                    .with("absorption_edge", "k")
                    .with("nexafs_model", model)
                    .with("line_shape", "gaussian")
                    .with("points", 10001usize)
                    .with("nexafs_broadening", 0.025)
                    .with("polarization", polarization),
            )
            .unwrap();
            let result = run("nexafs", &system, &controls).unwrap();
            let lines = list(&result, "lines");
            let energies = list(&result, "spectrum_energies");
            let density = list(&result, "spectrum_oscillator_density");
            let step = energies[1].as_f64().unwrap() - energies[0].as_f64().unwrap();
            let oscillator_total: f64 = lines.iter().map(|v| real(v, "oscillator_strength")).sum();
            let quadrature: f64 = density.iter().map(|v| v.as_f64().unwrap()).sum::<f64>() * step;
            let min_energy = real(&lines[0], "energy");
            let max_residual = lines
                .iter()
                .map(|v| real(v, "residual"))
                .fold(0.0, f64::max);
            let energy_components = result.get("energy_components").unwrap();
            println!("NEXAFS water {functional} {model} {basis} n={} E_first={min_energy:.12e} f_sum={oscillator_total:.12e} integrated={quadrature:.12e} residual={max_residual:.12e} ground orbital={:.12e} ground energy components={} holes={:?}",
                lines.len(),real(&result,"scf_orbital_residual"),energy_components.to_json(),result.get("core_holes").unwrap().to_json());
            assert!((17.0..25.0).contains(&min_energy));
            assert!(oscillator_total > 0.01);
            assert!((quadrature - oscillator_total).abs() < 2e-9);
            assert!(max_residual < 1e-8);
            for line in lines {
                let dipole: Vec<_> = list(line, "transition_dipole")
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let projected: f64 = dipole.iter().zip(polarization).map(|(d, p)| d * p).sum();
                let polarization_norm: f64 = polarization.iter().map(|x| x * x).sum();
                let expected = 2.0 * real(line, "energy") * projected.powi(2) / polarization_norm;
                assert!((real(line, "polarized_strength") - expected).abs() < 2e-13);
            }
            if model == "relaxed-core-ion" {
                assert_eq!(list(&result, "core_holes").len(), 2);
            }
            if model == "relaxed-core-ion" && functional == "wb97x-d" {
                // The determinant projection removes absolute-origin terms
                // even when separately relaxed orbitals are nonorthogonal.
                let translated = System::molecule(
                    vec![8, 1, 1],
                    coordinates
                        .map(|v| [v[0] + 1.7, v[1] - 0.8, v[2] + 0.6])
                        .to_vec(),
                );
                let shifted = run("nexafs", &translated, &controls).unwrap();
                for (a, b) in lines.iter().zip(list(&shifted, "lines")) {
                    for key in ["energy", "oscillator_strength", "polarized_strength"] {
                        assert!(
                            (real(a, key) - real(b, key)).abs() < 2e-8,
                            "{model} {basis} {key}"
                        );
                    }
                }
            }
        }
    }
}
