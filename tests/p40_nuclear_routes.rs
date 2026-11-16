// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::response::nuclear::nuclear_taylor;
use dft_rs::scf::{run_hf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;
use std::process::Command;

#[test]
fn rust_cli_cubic_quartic_and_mixed_values_share_units_and_diagnostics() {
    let path = std::env::temp_dir().join(format!("dft_nuclear_routes_{}.xyz", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    std::fs::write(
        &path,
        format!(
            "2\nnuclear response route validation\nHe 0 0 0\nH 0 0 {:.16}\n",
            1.5 * dft_rs::constants::BOHR_TO_ANGSTROM
        ),
    )
    .unwrap();
    let mol = Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 1.5),
        },
    ])
    .with_charge(1);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        ..Default::default()
    };
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    let report = nuclear_taylor(
        &mol,
        &mb,
        &ResolvedFunctional::hartree_fock(),
        None,
        &scf,
        &opts,
        &[[0.0; 3], [0.0, 0.0, 1.0]],
        4,
        &Default::default(),
    )
    .unwrap();
    let run = |command: &str, flags: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .arg(command)
            .arg(&path)
            .args([
                "--charge",
                "1",
                "--xc",
                "hf",
                "--log-mode",
                "none",
                "--conv-tol",
                "1e-13",
                "--conv-tol-grad",
                "1e-12",
                "--conv-tol-dp",
                "1e-12",
                "--max-iter",
                "500",
            ])
            .args(flags)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("Converged: true"));
        assert!(text.contains("Hartree"));
        assert!(text.contains("Response residuals:"));
        text
    };
    let text = run("nuclear-taylor", &["--nuclear-direction", "0,0,0;0,0,1"]);
    for n in [3, 4] {
        let prefix = format!("energy_coefficient {n} ");
        let row = text.lines().find(|l| l.starts_with(&prefix)).unwrap();
        let value: f64 = row.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert!((value - report.value.energy_coefficients[n]).abs() < 1e-10);
    }
    assert!(text.contains("particle_number_coefficients"));
    assert!(text.contains("long_range_exchange"));
    let text = run(
        "nuclear-mixed",
        &[
            "--nuclear-directions",
            "0,0,0;0,0,1|0,0,0;0,0,1|0,0,0;0,0,1|0,0,0;0,0,1",
        ],
    );
    let row = text
        .lines()
        .find(|l| l.starts_with("energy_derivative "))
        .unwrap();
    let value: f64 = row.split_whitespace().nth(1).unwrap().parse().unwrap();
    assert!((value - 24.0 * report.value.energy_coefficients[4]).abs() < 1e-9);
}

#[test]
fn cli_refuses_invalid_nuclear_controls_before_reading_geometry() {
    for flags in [
        vec![],
        vec!["--nuclear-direction", "0,0,0", "--response-omega", "1"],
        vec!["--nuclear-direction", "0,0,0", "--disp", "d3bj"],
        vec![
            "--nuclear-direction",
            "0,0,0",
            "--response-order",
            "4",
            "--response-tol",
            "0",
        ],
        vec!["--nuclear-direction", "0,0,inf"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .args(["nuclear-taylor", "missing_nuclear_input.xyz"])
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("No such file"), "{stderr}");
    }
}
