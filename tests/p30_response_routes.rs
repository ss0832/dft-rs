// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::response::{dynamic_polarizability, electric_field_taylor, Frequency};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;
use std::process::Command;

#[test]
fn rust_and_cli_share_response_values_units_and_diagnostics() {
    let path = std::env::temp_dir().join(format!("dft_response_routes_{}.xyz", std::process::id()));
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
        format!("2\nresponse route validation\nH 0 0 {negative:.16}\nH 0 0 {distance:.16}\n"),
    )
    .unwrap();
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let opts = ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-10,
        ..Default::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let taylor = electric_field_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        [0.0, 0.0, 1.0],
        3,
        &Default::default(),
    )
    .unwrap();
    let run = |command: &str, flags: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .arg(command)
            .arg(&path)
            .args([
                "--log-mode",
                "none",
                "--conv-tol",
                "1e-13",
                "--conv-tol-grad",
                "1e-10",
                "--conv-tol-dp",
                "1e-12",
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
        assert!(text.contains("converged true"));
        assert!(text.contains("residuals"));
        text
    };
    let output = run("field-taylor", &[]);
    let row = output
        .lines()
        .find(|l| l.starts_with("dipole_coefficient 3 "))
        .unwrap();
    let z: f64 = row.split_whitespace().nth(4).unwrap().parse().unwrap();
    assert!((z - taylor.value.dipole_coefficients[3][2]).abs() < 1e-9);
    assert!(output.contains("derivative n / n!"));
    let output = run("field-mixed", &["--field-directions", "0,0,1;0,0,1;0,0,1"]);
    let row = output
        .lines()
        .find(|l| l.starts_with("dipole_derivative "))
        .unwrap();
    let z: f64 = row.split_whitespace().nth(3).unwrap().parse().unwrap();
    assert!((z - 6.0 * taylor.value.dipole_coefficients[3][2]).abs() < 1e-8);
    let response = dynamic_polarizability(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        Frequency::Retarded {
            omega: 0.2,
            damping: 0.03,
        },
        &Default::default(),
    )
    .unwrap();
    let output = run(
        "dynamic-polarizability",
        &["--response-omega", "0.2", "--response-damping", "0.03"],
    );
    let row = output
        .lines()
        .find(|l| l.starts_with("alpha 2 2 "))
        .unwrap();
    let values: Vec<f64> = row
        .split_whitespace()
        .skip(3)
        .map(|x| x.parse().unwrap())
        .collect();
    assert!((values[0] - response.value.alpha[2][2].re).abs() < 1e-9);
    assert!((values[1] - response.value.alpha[2][2].im).abs() < 1e-9);
    assert!(output.contains("bohr^3"));
}
