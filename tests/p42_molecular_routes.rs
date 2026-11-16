// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::response::nuclear::{
    mixed_molecular_response, molecular_taylor, static_spectroscopic_derivative, MolecularDirection,
};
use dft_rs::scf::{run_hf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;
use std::process::Command;

#[test]
fn rust_and_cli_share_joint_jets_mixed_observables_and_all_spectroscopic_entries() {
    let path =
        std::env::temp_dir().join(format!("dft_molecular_routes_{}.xyz", std::process::id()));
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
            "2\nmixed route validation\nHe 0 0 0\nH 0 0 {:.16}\n",
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
        external_field: Some(Vec3::new(0.0, 0.0, 0.02)),
        ..Default::default()
    };
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let nuclear = vec![[0.0; 3], [0.0, 0.0, 1.0]];
    let electric = [0.1, -0.2, 0.3];
    let joint = molecular_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &MolecularDirection {
            nuclear: nuclear.clone(),
            electric,
        },
        4,
        &Default::default(),
    )
    .unwrap();
    let mixed = mixed_molecular_response(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &[
            MolecularDirection::nuclear(nuclear.clone()),
            MolecularDirection::electric(2, [0.0, 0.0, 1.0]),
        ],
        &Default::default(),
    )
    .unwrap();
    let hyper = static_spectroscopic_derivative(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &nuclear,
        3,
        &Default::default(),
    )
    .unwrap();
    assert!(joint.converged && mixed.converged && hyper.converged);
    let run = |command: &str, controls: &[&str]| {
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
                "--field",
                "0,0,0.02",
            ])
            .args(controls)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(
            text.contains("Converged: true") && text.contains("Response residuals:"),
            "{text}"
        );
        text
    };
    let text = run(
        "molecular-taylor",
        &[
            "--nuclear-direction",
            "0,0,0;0,0,1",
            "--field-direction",
            "0.1,-0.2,0.3",
        ],
    );
    for (n, e) in joint.value.energy_coefficients.iter().enumerate() {
        let prefix = format!("energy_coefficient {n} ");
        let row = text.lines().find(|l| l.starts_with(&prefix)).unwrap();
        let value: f64 = row.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert!((value - e).abs() < 1e-10);
        let prefix = format!("dipole_coefficient {n} ");
        let values: Vec<f64> = text
            .lines()
            .find(|l| l.starts_with(&prefix))
            .unwrap()
            .split_whitespace()
            .skip(2)
            .map(|s| s.parse().unwrap())
            .collect();
        for a in 0..3 {
            assert!((values[a] - joint.value.dipole_coefficients[n][a]).abs() < 1e-10);
        }
    }
    assert!(text.contains("nuclear_field"));
    let text = run(
        "molecular-mixed",
        &[
            "--nuclear-directions",
            "0,0,0;0,0,1|0,0,0;0,0,0",
            "--field-directions",
            "0,0,0;0,0,1",
        ],
    );
    let energy: f64 = text
        .lines()
        .find(|l| l.starts_with("energy_derivative "))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert!((energy - mixed.value.energy_derivative).abs() < 1e-10);
    let dipole: Vec<f64> = text
        .lines()
        .find(|l| l.starts_with("dipole_derivative "))
        .unwrap()
        .split_whitespace()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    for a in 0..3 {
        assert!((dipole[a] - mixed.value.dipole_derivative[a]).abs() < 1e-10);
    }
    let text = run(
        "static-spectroscopy",
        &[
            "--nuclear-direction",
            "0,0,0;0,0,1",
            "--electric-order",
            "3",
        ],
    );
    let values: Vec<f64> = text
        .lines()
        .filter(|l| l.starts_with("tensor "))
        .map(|l| l.split_whitespace().nth(2).unwrap().parse().unwrap())
        .collect();
    assert_eq!(values.len(), 27);
    for (a, b) in values.iter().zip(&hyper.value.tensor) {
        assert!((a - b).abs() < 1e-10);
    }
    assert!(text.contains("permutation_comparisons 36"));
}

#[test]
fn incompatible_molecular_controls_are_rejected_before_loading_geometry() {
    for (command, controls) in [
        ("molecular-taylor", vec![]),
        (
            "molecular-mixed",
            vec![
                "--nuclear-directions",
                "0,0,1",
                "--field-directions",
                "0,0,1;0,1,0",
            ],
        ),
        (
            "static-spectroscopy",
            vec!["--nuclear-direction", "0,0,1", "--electric-order", "4"],
        ),
        (
            "static-spectroscopy",
            vec!["--nuclear-direction", "0,0,1", "--response-order", "4"],
        ),
        (
            "static-spectroscopy",
            vec!["--nuclear-direction", "0,0,1", "--response-omega", "1"],
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .args([command, "missing_mixed_input.xyz"])
            .args(controls)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("No such file"));
    }
}
