// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex Bloch orbital export: every band, AO pair, k point and spin.
//! Independent scalar contractions check C^dagger S C, F C - S C epsilon,
//! C f C^dagger, and Tr(P S). No derivative is introduced by this export.

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::response::complex::Complex;

fn system() -> System {
    System {
        numbers: vec![1, 1],
        positions: vec![[0.13, -0.21, 0.19], [1.57, 0.16, -0.08]],
        cell: Some([[4.3, 0.27, -0.11], [0.19, 6.7, 0.31], [-0.23, 0.37, 7.1]]),
        pbc: [true; 3],
    }
}

fn controls(xc: &str, temperature: f64, open: bool, export: bool) -> Controls {
    let value = Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("k_mesh", vec![3, 1, 1])
        .with("overlap_cutoff", 12.0)
        .with("exchange_cutoff", 12.0)
        .with("grid_level", 2)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("max_iter", 300)
        .with("guess", "core")
        .with("electronic_temperature", temperature)
        .with("charge", i32::from(open))
        .with("multiplicity", if open { 2 } else { 1 })
        .with("unrestricted", open)
        .with("coefficients", export);
    Controls::from_value(&value).unwrap()
}

fn array(v: &Value) -> &[Value] {
    v.as_array().unwrap()
}

fn scalar(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn matrix(v: &Value, real: &str, imaginary: &str, k: usize) -> Vec<Vec<Complex>> {
    array(&array(v.get(real).unwrap())[k])
        .iter()
        .zip(array(&array(v.get(imaginary).unwrap())[k]))
        .map(|(r, i)| {
            array(r)
                .iter()
                .zip(array(i))
                .map(|(r, i)| Complex::new(scalar(r), scalar(i)))
                .collect()
        })
        .collect()
}

fn validate(report: &Value, xc: &str) {
    assert_eq!(report.get("converged").unwrap().as_bool(), Some(true));
    let nk = array(report.get("k_points").unwrap()).len();
    let mut count = 0.0;
    let mut complex_part = 0.0_f64;
    let mut defects = [0.0_f64; 3];
    for channel in array(report.get("channels").unwrap()) {
        let orbitals = channel.get("orbitals").unwrap();
        let weight = scalar(channel.get("spin_weight").unwrap());
        let mut population = 0.0;
        for k in 0..nk {
            let c = matrix(orbitals, "coefficients_real", "coefficients_imaginary", k);
            let s = matrix(orbitals, "overlap_real", "overlap_imaginary", k);
            let f = matrix(orbitals, "fock_real", "fock_imaginary", k);
            let p = matrix(orbitals, "density_real", "density_imaginary", k);
            let e = array(&array(channel.get("band_energies").unwrap())[k]);
            let occ = array(&array(channel.get("occupations").unwrap())[k]);
            let n = c.len();
            let nb = e.len();
            assert_eq!(nb, occ.len());
            assert!(c.iter().all(|row| row.len() == nb));
            for a in 0..nb {
                for b in 0..nb {
                    let mut metric = Complex::ZERO;
                    for i in 0..n {
                        for j in 0..n {
                            metric += c[i][a].conj() * s[i][j] * c[j][b];
                        }
                    }
                    defects[0] = defects[0]
                        .max((metric - Complex::new(if a == b { 1.0 } else { 0.0 }, 0.0)).abs());
                }
                for i in 0..n {
                    let mut residual = Complex::ZERO;
                    for j in 0..n {
                        residual += (f[i][j] - s[i][j] * scalar(&e[a])) * c[j][a];
                    }
                    defects[1] = defects[1].max(residual.abs());
                    complex_part = complex_part.max(c[i][a].im.abs());
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let mut reconstructed = Complex::ZERO;
                    for a in 0..nb {
                        reconstructed += c[i][a] * c[j][a].conj() * scalar(&occ[a]);
                    }
                    defects[2] = defects[2].max((reconstructed - p[i][j]).abs());
                    population += (p[i][j] * s[j][i]).re / nk as f64;
                }
            }
            for key in [
                "metric_defect",
                "eigen_residual",
                "band_energy_residual",
                "density_reconstruction_residual",
            ] {
                assert!(
                    scalar(&array(orbitals.get(key).unwrap())[k]) < 1e-9,
                    "{xc} {key}"
                );
            }
        }
        let expected = if channel.get("spin").unwrap().as_str() == Some("beta") {
            "n_beta"
        } else {
            "n_alpha"
        };
        assert!((population - scalar(report.get(expected).unwrap())).abs() < 1e-10);
        count += weight * population;
    }
    assert!(
        complex_part > 1e-3,
        "the fixture must exercise complex Bloch coefficients"
    );
    let electrons = scalar(report.get("n_alpha").unwrap()) + scalar(report.get("n_beta").unwrap());
    assert!((count - electrons).abs() < 1e-10);
    println!(
        "{xc} all-entry metric/eigen/density defects={defects:?}; electron defect={:.3e}",
        count - electrons
    );
    assert!(defects.into_iter().all(|r| r < 1e-9));
}

#[test]
fn every_bloch_coefficient_matches_the_reference_for_functional_classes() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0"] {
        let report = run(
            "electronic-structure",
            &system(),
            &controls(xc, 0.0, false, true),
        )
        .unwrap();
        validate(&report, xc);
    }
}

#[test]
fn thermal_and_empty_spin_channels_reconstruct_the_same_density() {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    for (temperature, open) in [(50000.0, false), (0.0, true), (50000.0, true)] {
        let report = run(
            "electronic-structure",
            &system(),
            &controls("pbe", temperature, open, true),
        )
        .unwrap();
        validate(&report, "pbe");
        if temperature > 0.0 {
            assert!(array(report.get("channels").unwrap())
                .iter()
                .any(|ch| array(ch.get("occupations").unwrap())
                    .iter()
                    .flat_map(array)
                    .any(|f| scalar(f) > 0.01 && scalar(f) < 0.99)));
        }
    }
}

#[test]
fn export_is_optional_and_does_not_change_scf_results() {
    let plain = run(
        "electronic-structure",
        &system(),
        &controls("hf", 0.0, false, false),
    )
    .unwrap();
    let exported = run(
        "electronic-structure",
        &system(),
        &controls("hf", 0.0, false, true),
    )
    .unwrap();
    assert_eq!(plain.get("energy"), exported.get("energy"));
    for (a, b) in array(plain.get("channels").unwrap())
        .iter()
        .zip(array(exported.get("channels").unwrap()))
    {
        assert!(a.get("orbitals").is_none());
        for key in [
            "band_energies",
            "occupations",
            "chemical_potential",
            "band_edges",
        ] {
            assert_eq!(a.get(key), b.get(key));
        }
    }
}
