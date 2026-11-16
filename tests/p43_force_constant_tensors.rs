// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::gradient::assemble_gradient;
use dft_rs::hessian::analytic_hessian_full;
use dft_rs::math::Vec3;
use dft_rs::response::force_constants::{
    analytic_force_constants, cartesian_force_constant_block, cartesian_force_constants,
};
use dft_rs::response::nuclear::nuclear_taylor;
use dft_rs::scf::{run_hf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;
use std::process::Command;

fn molecule(open: bool) -> Molecule {
    let mol = Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 1.5),
        },
    ]);
    if open {
        mol.with_multiplicity(2)
    } else {
        mol.with_charge(1)
    }
}
fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        ..Default::default()
    }
}
fn ix(tuple: &[usize], dimension: usize) -> usize {
    tuple.iter().fold(0, |ix, &i| ix * dimension + i)
}

#[test]
fn full_cubic_quartic_tensors_obey_translation_and_independent_radial_derivative_identities() {
    let mol = molecule(false);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = options();
    let functional = ResolvedFunctional::hartree_fock();
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    let gradient = assemble_gradient(&mol, &mb, &functional, None, &scf, &opts).unwrap();
    let hessian = analytic_hessian_full(&mol, &mb, &opts, &functional, &scf, None)
        .unwrap()
        .hessian;
    let radial = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &[[0.0; 3], [0.0, 0.0, 1.0]],
        4,
        &Default::default(),
    )
    .unwrap();
    for order in [3, 4] {
        let report = cartesian_force_constants(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            order,
            &Default::default(),
        )
        .unwrap();
        assert!(
            report.converged,
            "{:?}; translation {:?}; permutation {}",
            report.warnings, report.value.translation_residual, report.value.permutation_defect
        );
        assert_eq!(report.shape, vec![6; order]);
        let v = report.value;
        assert_eq!(v.independent_components, if order == 3 { 56 } else { 126 });
        assert_eq!(v.permutation_comparisons, v.independent_components - 6);
        assert_eq!(
            v.translation_comparisons,
            3 * order * 6usize.pow((order - 1) as u32)
        );
        assert!(v.translation_residual.unwrap() < 1e-10);
        assert!(v.permutation_defect < 1e-10);
        let factor = if order == 3 { 6.0 } else { 24.0 };
        assert!(
            (v.tensor[ix(&vec![5; order], 6)] - factor * radial.value.energy_coefficients[order])
                .abs()
                < 1e-10
        );
        let r = 1.5;
        let expected = if order == 3 {
            hessian[(5, 5)] / r - gradient[1].z / (r * r)
        } else {
            3.0 * hessian[(5, 5)] / (r * r) - 3.0 * gradient[1].z / (r * r * r)
        };
        let tuple = if order == 3 {
            vec![3, 3, 5]
        } else {
            vec![3, 3, 3, 3]
        };
        assert!(
            (v.tensor[ix(&tuple, 6)] - expected).abs() < 1e-9,
            "order {order}"
        );
        let block = cartesian_force_constant_block(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            &[5, 3],
            order,
            &Default::default(),
        )
        .unwrap();
        assert!(block.converged && block.value.translation_residual.is_none());
        assert_eq!(block.value.translation_comparisons, 0);
        for flat in 0..2usize.pow(order as u32) {
            let tuple: Vec<_> = (0..order)
                .map(|k| [5, 3][(flat / 2usize.pow((order - 1 - k) as u32)) % 2])
                .collect();
            assert!((block.value.tensor[flat] - v.tensor[ix(&tuple, 6)]).abs() < 1e-10);
        }
    }
}

#[test]
fn arbitrary_pattern_tensors_preserve_multilinearity_in_an_unrestricted_reference() {
    let mol = molecule(true);
    let opts = options();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = ResolvedFunctional::hartree_fock();
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    let patterns = vec![
        vec![[0.0; 3], [0.0, 0.0, 1.0]],
        vec![[0.0; 3], [0.0, 0.0, -2.0]],
    ];
    let radial = nuclear_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &patterns[0],
        4,
        &Default::default(),
    )
    .unwrap();
    for order in [3, 4] {
        let report = analytic_force_constants(
            &mol,
            &mb,
            &functional,
            None,
            &scf,
            &opts,
            &patterns,
            order,
            &Default::default(),
        )
        .unwrap();
        assert!(report.converged, "{:?}", report.warnings);
        let base = radial.value.energy_coefficients[order] * if order == 3 { 6.0 } else { 24.0 };
        for flat in 0..2usize.pow(order as u32) {
            let minus = flat.count_ones();
            assert!(
                (report.value.tensor[flat] - base * (-2.0_f64).powi(minus as i32)).abs() < 1e-9
            );
        }
        assert!(report.value.translation_residual.is_none());
    }
}

#[test]
fn force_constant_cli_matches_cartesian_blocks_and_rejects_incompatible_controls() {
    let path = std::env::temp_dir().join(format!("dft_force_constants_{}.xyz", std::process::id()));
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
            "2\nforce constant route validation\nHe 0 0 0\nH 0 0 {:.16}\n",
            1.5 * dft_rs::constants::BOHR_TO_ANGSTROM
        ),
    )
    .unwrap();
    let mol = molecule(false);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = options();
    let scf = run_hf(&mol, &mb, &opts).unwrap();
    let expected = cartesian_force_constant_block(
        &mol,
        &mb,
        &ResolvedFunctional::hartree_fock(),
        None,
        &scf,
        &opts,
        &[2, 5],
        4,
        &Default::default(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
        .arg("force-constants")
        .arg(&path)
        .args([
            "--response-order",
            "4",
            "--force-coordinates",
            "2,5",
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
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let tensor: Vec<f64> = text
        .lines()
        .filter(|l| l.starts_with("tensor "))
        .map(|l| l.split_whitespace().nth(2).unwrap().parse().unwrap())
        .collect();
    assert_eq!(tensor.len(), 16);
    assert!(text.contains("translation_residual None"));
    for (a, b) in tensor.iter().zip(expected.value.tensor) {
        assert!((a - b).abs() < 1e-10);
    }
    for flags in [
        vec!["--response-order", "2"],
        vec!["--force-coordinates", "2,2"],
        vec!["--force-coordinates", "2", "--nuclear-directions", "0,0,1"],
        vec!["--field-direction", "0,0,1"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dft-rs"))
            .args(["force-constants", "missing_force_constant_input.xyz"])
            .args(flags)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("No such file"));
    }
}
