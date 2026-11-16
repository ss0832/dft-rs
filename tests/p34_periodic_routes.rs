// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::data;
use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
use dft_rs::math::Vec3;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::Frequency;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};
use std::process::Command;

#[test]
fn rust_and_cli_periodic_routes_preserve_values_and_physical_units() {
    let path = std::env::temp_dir().join(format!("dft_periodic_routes_{}.xyz", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    let distance = 0.7 * dft_rs::constants::BOHR_TO_ANGSTROM;
    let negative = -distance;
    std::fs::write(
        &path,
        format!("2\nperiodic response validation\nH 0 0 {negative:.16}\nH 0 0 {distance:.16}\n"),
    )
    .unwrap();
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        conv_tol_grad: 1e-10,
        max_iter: 250,
        grid_shell: 1,
        grid: dft_rs::grid::GridOptions {
            level: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    let request = PeriodicResponseOptions {
        q: Vec3::new(0.043, 0.017, -0.021),
        frequency: Frequency::Retarded {
            omega: 0.13,
            damping: 0.04,
        },
        reciprocal_cutoff: 0.75,
        ..Default::default()
    };
    let solver =
        PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &request).unwrap();
    let expected = solver.solve(&solver.plane_wave_source(0).unwrap()).unwrap();
    let edge = format!("{:.16}", 9.0 * dft_rs::constants::BOHR_TO_ANGSTROM);
    let run = |command: &str| {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .arg(command)
            .arg(&path)
            .args([
                "--cell",
                &edge,
                "--xc",
                "pbe",
                "--overlap-cutoff",
                "7",
                "--grid-level",
                "0",
                "--grid-shell",
                "1",
                "--conv-tol",
                "1e-12",
                "--conv-tol-dp",
                "1e-11",
                "--conv-tol-grad",
                "1e-10",
                "--max-iter",
                "250",
                "--response-q",
                "0.043,0.017,-0.021",
                "--response-omega",
                "0.13",
                "--response-damping",
                "0.04",
                "--reciprocal-cutoff",
                "0.75",
                "--log-mode",
                "none",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("Converged: true"));
        assert!(text.contains("bohr^-3/hartree"));
        assert!(text.contains("Reference particle-number error"));
        text
    };
    let density = run("periodic-response");
    let row = density
        .lines()
        .find(|line| {
            line.split_whitespace().count() == 5
                && line
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
                    .is_some()
        })
        .unwrap();
    let values: Vec<f64> = row.split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert!((values[3] - expected.value.fourier_density[0].re).abs() < 1e-11);
    assert!((values[4] - expected.value.fourier_density[0].im).abs() < 1e-11);
    let dielectric = run("dielectric");
    let row = dielectric
        .lines()
        .find(|line| line.starts_with("0 0 "))
        .unwrap();
    let values: Vec<f64> = row
        .split_whitespace()
        .skip(2)
        .map(|x| x.parse().unwrap())
        .collect();
    let inverse = expected.value.fourier_density[0]
        * (4.0 * std::f64::consts::PI / expected.value.reciprocal_waves[0].norm2())
        + dft_rs::response::complex::Complex::from(1.0);
    assert!((values[0] - inverse.re).abs() < 1e-10);
    assert!((values[1] - inverse.im).abs() < 1e-10);
    assert!((values[2] - expected.value.fourier_density[0].re).abs() < 1e-11);
    assert!((values[3] - expected.value.fourier_density[0].im).abs() < 1e-11);
}
