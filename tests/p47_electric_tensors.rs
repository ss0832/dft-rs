// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::properties::dipole_moment;
use dft_rs::response::{static_beta, static_electric_tensor, static_gamma};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::process::Command;

#[path = "support/convergence.rs"]
mod convergence;

fn molecule(axis: [f64; 3], center: [f64; 3]) -> Molecule {
    let mut mol = Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::new(center[0], center[1], center[2]),
        },
        Atom {
            z: 1,
            position: Vec3::new(
                center[0] + 1.75 * axis[0],
                center[1] + 1.75 * axis[1],
                center[2] + 1.75 * axis[2],
            ),
        },
    ]);
    mol.charge = 1;
    mol
}

fn options() -> ScfOptions {
    ScfOptions {
        max_iter: 250,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        ..Default::default()
    }
}

#[test]
fn all_ordered_entries_match_independent_finite_field_energy_and_dipole_oracles() {
    let mol = molecule([0.0, 0.0, 1.0], [0.0; 3]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let beta = static_beta(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &Default::default(),
    )
    .unwrap();
    let gamma = static_gamma(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &Default::default(),
    )
    .unwrap();
    assert!(beta.converged && gamma.converged);
    assert_eq!(beta.shape, [3, 3, 3]);
    assert_eq!(gamma.shape, [3, 3, 3, 3]);
    assert_eq!(beta.value.ordered_field_evaluations, 9);
    assert_eq!(gamma.value.ordered_field_evaluations, 27);
    assert_eq!(beta.value.directional_evaluations, 24);
    assert_eq!(gamma.value.directional_evaluations, 141);
    assert_eq!(beta.value.intrinsic_permutation_comparisons, 9);
    assert_eq!(beta.value.energy_permutation_comparisons, 18);
    assert_eq!(gamma.value.intrinsic_permutation_comparisons, 81);
    assert_eq!(gamma.value.energy_permutation_comparisons, 81);
    let b = beta.value.tensor[26];
    let variational = dft_rs::response::static_beta_variational(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &Default::default(),
    )
    .unwrap();
    assert!(variational.converged);
    assert_eq!(variational.residuals.len(), 3);
    for (&a, &b) in variational.value.tensor.iter().zip(&beta.value.tensor) {
        assert!((a - b).abs() < 1e-9);
    }
    let g = gamma.value.tensor[80];
    assert!(b.abs() > 1.0 && g.abs() > 1.0);
    let at_field = |field: f64| {
        let mut shifted = opts.clone();
        shifted.external_field = Some(Vec3::new(0.0, 0.0, field));
        let result = run_scf(&mol, &mb, &shifted, &functional, None).unwrap();
        assert!(result.converged);
        (dipole_moment(&mol, &mb, &result).z, result.energy)
    };
    let mut beta_errors = Vec::new();
    let mut gamma_errors = Vec::new();
    let mut beta_values = Vec::new();
    let mut gamma_values = Vec::new();
    let steps = [0.02_f64, 0.01, 0.005, 0.0025];
    for h in steps {
        let plus = at_field(h);
        let minus = at_field(-h);
        let double_plus = at_field(2.0 * h);
        let double_minus = at_field(-2.0 * h);
        let beta_fd = (plus.0 - 2.0 * dipole_moment(&mol, &mb, &scf).z + minus.0) / h.powi(2);
        let gamma_fd =
            (double_plus.0 - 2.0 * plus.0 + 2.0 * minus.0 - double_minus.0) / (2.0 * h.powi(3));
        let energy_third =
            (double_plus.1 - 2.0 * plus.1 + 2.0 * minus.1 - double_minus.1) / (2.0 * h.powi(3));
        let energy_fourth = (double_plus.1 - 4.0 * plus.1 + 6.0 * scf.energy - 4.0 * minus.1
            + double_minus.1)
            / h.powi(4);
        eprintln!(
            "h={h:.6}; beta={b:.12e}; beta_fd={beta_fd:.12e}; gamma={g:.12e}; \
            gamma_fd={gamma_fd:.12e}; energy_D3={energy_third:.12e}; \
            energy_D4={energy_fourth:.12e}"
        );
        assert!((energy_third + b).abs() < 0.1 * b.abs());
        assert!((energy_fourth + g).abs() < 0.1 * g.abs());
        beta_errors.push((beta_fd - b).abs());
        gamma_errors.push((gamma_fd - g).abs());
        beta_values.push(beta_fd);
        gamma_values.push(gamma_fd);
    }
    for (name, errors) in [("beta", beta_errors), ("gamma", gamma_errors)] {
        let fit = convergence::error_power_fit(&steps, &errors);
        eprintln!(
            "{name} four-step error fit: p={:.12}, C={:.12e}, R_squared={:.12}, log_rmse={:.12e}",
            fit.exponent, fit.prefactor, fit.r_squared, fit.log_rmse
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.02,
            "{name}: fitted order {}",
            fit.exponent
        );
        assert!(fit.r_squared > 0.99999, "{name}: poor power fit");
        for pair in errors.windows(2) {
            let ratio = pair[0] / pair[1];
            eprintln!("{name} h/2 error ratio={ratio:.12}");
            assert!((ratio - 4.0).abs() < 0.25, "{name}: {ratio}");
        }
    }
    let richardson_b = (4.0 * beta_values[3] - beta_values[2]) / 3.0;
    let richardson_g = (4.0 * gamma_values[3] - gamma_values[2]) / 3.0;
    assert!((richardson_b - b).abs() < 2e-6 * b.abs());
    assert!((richardson_g - g).abs() < 2e-5 * g.abs());

    // With an s-only diatomic basis the induced dipole is longitudinal.
    // Rotation therefore has a closed rank-one tensor oracle for EVERY entry.
    let axis = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let rotated = molecule(axis, [0.4, -0.3, 0.2]);
    let rb = MolBasis::build(&rotated, &data::load("sto-3g").unwrap()).unwrap();
    let rs = run_scf(&rotated, &rb, &opts, &functional, None).unwrap();
    for order in [2, 3] {
        let report = static_electric_tensor(
            &rotated,
            &rb,
            &functional,
            None,
            &rs,
            &opts,
            order,
            &Default::default(),
        )
        .unwrap();
        assert!(report.converged);
        for (flat, &entry) in report.value.tensor.iter().enumerate() {
            let mut remaining = flat;
            let mut product = if order == 2 { b } else { g };
            for _ in 0..=order {
                product *= axis[remaining % 3];
                remaining /= 3;
            }
            assert!((entry - product).abs() < 2e-8 * product.abs().max(1.0));
        }
        assert!(report.value.energy_permutation_defect < 1e-8);
        assert!(report.value.intrinsic_permutation_defect < 1e-8);
        assert!(report.value.projector_defect < 1e-9);
        assert!(report.value.stationarity_defect < 1e-8);
        assert!(report.value.particle_number_defect < 1e-9);
        if order == 2 {
            let variational = dft_rs::response::static_beta_variational(
                &rotated,
                &rb,
                &functional,
                None,
                &rs,
                &opts,
                &Default::default(),
            )
            .unwrap();
            assert!(variational.converged);
            for (&a, &b) in variational.value.tensor.iter().zip(&report.value.tensor) {
                assert!((a - b).abs() < 1e-8);
            }
            let vector = report.value.beta_vector.unwrap();
            for i in 0..3 {
                assert!((vector[i] - b * axis[i]).abs() < 1e-8);
            }
            assert!((report.value.beta_vector_norm.unwrap() - b.abs()).abs() < 1e-8);
        } else {
            assert!((report.value.gamma_isotropic.unwrap() - g / 5.0).abs() < 1e-8);
        }
        assert!(
            (report.value.frobenius_norm - if order == 2 { b.abs() } else { g.abs() }).abs() < 1e-8
        );
    }
    assert!(static_electric_tensor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        4,
        &Default::default()
    )
    .is_err());
}

#[test]
fn generic_semilocal_hybrid_and_range_separated_kernels_preserve_static_energy_symmetry() {
    let mol = molecule([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], [0.0; 3]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let opts = options();
    for name in ["svwn5", "pbe", "b3lyp", "cam-b3lyp", "r2scan"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let scf = run_scf(&mol, &mb, &opts, &f, Some(&grid)).unwrap();
        for order in [2, 3] {
            let report = static_electric_tensor(
                &mol,
                &mb,
                &f,
                Some(&grid),
                &scf,
                &opts,
                order,
                &Default::default(),
            )
            .unwrap();
            assert!(report.converged, "{name}/{order}: {:?}", report.warnings);
            assert!(report.value.tensor.iter().all(|x| x.is_finite()));
            assert!(report.residuals.iter().all(|&x| x <= 1e-10));
            assert!(report.value.stationarity_defect < 1e-8, "{name}/{order}");
        }
        // Independent self-consistent finite-field quadratic response, with
        // native XC potentials rather than the Taylor XC kernel as oracle.
        let beta =
            static_beta(&mol, &mb, &f, Some(&grid), &scf, &opts, &Default::default()).unwrap();
        let at = |field: f64| {
            let mut shifted = opts.clone();
            shifted.external_field = Some(Vec3::new(0.0, 0.0, field));
            let r = run_scf(&mol, &mb, &shifted, &f, Some(&grid)).unwrap();
            assert!(r.converged);
            dipole_moment(&mol, &mb, &r).z
        };
        let fd = |h: f64| (at(h) - 2.0 * dipole_moment(&mol, &mb, &scf).z + at(-h)) / h.powi(2);
        let oracle = (4.0 * fd(0.005) - fd(0.01)) / 3.0;
        assert!(
            (oracle - beta.value.tensor[26]).abs() < 2e-5 * beta.value.tensor[26].abs().max(1.0),
            "{name}"
        );
    }
}

#[test]
fn canonical_and_alias_cli_return_every_raw_entry_and_shared_diagnostics() {
    let mol = molecule([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], [0.0; 3]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let f = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &f, None).unwrap();
    let path = std::env::temp_dir().join(format!("dft_electric_tensor_{}.xyz", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    let pos = mol.atoms[1].position * dft_rs::constants::BOHR_TO_ANGSTROM;
    std::fs::write(
        &path,
        format!(
            "2\nelectric tensor validation\nHe 0 0 0\nH {:.16} {:.16} {:.16}\n",
            pos.x, pos.y, pos.z
        ),
    )
    .unwrap();
    for bin in [
        env!("CARGO_BIN_EXE_dft-rs"),
        env!("CARGO_BIN_EXE_dft_rs_cli"),
    ] {
        for (command, order, assembly) in [
            ("beta", 2, "projector"),
            ("beta", 2, "variational"),
            ("gamma", 3, "projector"),
        ] {
            let report = if assembly == "variational" {
                dft_rs::response::static_beta_variational(
                    &mol,
                    &mb,
                    &f,
                    None,
                    &scf,
                    &opts,
                    &Default::default(),
                )
                .unwrap()
            } else {
                static_electric_tensor(&mol, &mb, &f, None, &scf, &opts, order, &Default::default())
                    .unwrap()
            };
            let mut invocation = Command::new(bin);
            invocation.args([
                command,
                path.to_str().unwrap(),
                "--charge",
                "1",
                "--log-mode",
                "none",
                "--conv-tol",
                "1e-14",
                "--conv-tol-dp",
                "1e-13",
                "--conv-tol-grad",
                "1e-12",
                "--response-tol",
                "1e-10",
            ]);
            if assembly == "variational" {
                invocation.args(["--electric-assembly", "variational"]);
            }
            let output = invocation.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8(output.stdout).unwrap();
            let values: Vec<f64> = stdout
                .lines()
                .filter(|line| line.starts_with(&format!("{command} ")))
                .map(|line| line.split_whitespace().last().unwrap().parse().unwrap())
                .collect();
            assert_eq!(values.len(), report.value.tensor.len());
            for (&a, &b) in values.iter().zip(&report.value.tensor) {
                assert!((a - b).abs() < 2e-8);
            }
            assert!(stdout.contains(report.units));
            assert!(stdout.contains("raw components preserved"));
            assert!(stdout.contains("converged true"));
            assert!(stdout.contains("energy_permutation_comparisons"));
            if assembly == "variational" {
                assert!(stdout.contains("2n+1"));
                assert_eq!(report.residuals.len(), 3);
            }
        }
        for args in [
            ["beta", "--response-omega", "0.1"],
            ["gamma", "--response-max-iter", "0"],
            ["beta", "--field-directions", "0,0,1"],
        ] {
            let output = Command::new(bin)
                .arg(args[0])
                .arg("missing.xyz")
                .args(&args[1..])
                .output()
                .unwrap();
            assert!(!output.status.success());
            assert!(!String::from_utf8_lossy(&output.stderr).contains("cannot read"));
        }
    }
}

#[test]
fn open_shell_one_electron_tensors_match_closed_dense_eigenvalue_derivatives() {
    let mut mol = molecule([0.0, 0.0, 1.0], [0.0; 3]);
    mol.charge = 2;
    mol.multiplicity = 2;
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let f = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &f, None).unwrap();
    assert_eq!((scf.n_alpha, scf.n_beta), (1, 0));
    assert!(scf.unrestricted);
    let c = &scf.mo_coeff[0];
    let h = c.transpose().matmul_seq(&scf.h_core).matmul_seq(c);
    let r = dft_rs::integrals::dipole_matrices(&mb, Vec3::zero());
    let rz = c.transpose().matmul_seq(&r[2]).matmul_seq(c);
    // HF exactly cancels the single electron's J and K in the occupied
    // variational energy. This independent 2x2 one-electron Hamiltonian has
    // E(t)=trace(H(t))/2-sqrt(q0+q1*t+q2*t^2), plus linear nuclear terms.
    let dx = (h[(0, 0)] - h[(1, 1)]) / 2.0;
    let bx = h[(0, 1)];
    let dd = (rz[(0, 0)] - rz[(1, 1)]) / 2.0;
    let bd = rz[(0, 1)];
    let q0 = dx * dx + bx * bx;
    let a = 2.0 * (dx * dd + bx * bd) / q0;
    let b = (dd * dd + bd * bd) / q0;
    let beta_exact = 6.0 * q0.sqrt() * (-a * b / 4.0 + a.powi(3) / 16.0);
    let gamma_exact =
        24.0 * q0.sqrt() * (-b * b / 8.0 + 3.0 * a * a * b / 16.0 - 5.0 * a.powi(4) / 128.0);
    let variational = dft_rs::response::static_beta_variational(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        &Default::default(),
    )
    .unwrap();
    assert!(variational.converged);
    assert!((variational.value.tensor[26] - beta_exact).abs() < 1e-9);
    for (order, exact) in [(2, beta_exact), (3, gamma_exact)] {
        let report =
            static_electric_tensor(&mol, &mb, &f, None, &scf, &opts, order, &Default::default())
                .unwrap();
        assert!(report.converged);
        assert!((report.value.tensor.last().unwrap() - exact).abs() < 1e-9 * exact.abs().max(1.0));
        for &value in &report.value.tensor[..report.value.tensor.len() - 1] {
            assert!(value.abs() < 1e-10);
        }
        assert!(report.value.particle_number_defect < 1e-10);
        eprintln!("one-electron closed eigenvalue oracle: order={order}, value={exact:.12e}");
    }
}

#[test]
fn unregistered_weighted_functional_uses_energy_consistent_generic_response() {
    let mol = molecule([0.0, 0.0, 1.0], [0.0; 3]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut f = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    f.description = "unregistered weighted local hybrid acceptance model".into();
    for (weight, _) in &mut f.parts {
        *weight *= 0.63;
    }
    f.hybrid.cx = 0.37;
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &f, Some(&grid)).unwrap();
    let report = static_beta(&mol, &mb, &f, Some(&grid), &scf, &opts, &Default::default()).unwrap();
    assert!(report.converged);
    assert_eq!(report.functional, f.description);
    let at = |field: f64| {
        let mut shifted = opts.clone();
        shifted.external_field = Some(Vec3::new(0.0, 0.0, field));
        let r = run_scf(&mol, &mb, &shifted, &f, Some(&grid)).unwrap();
        assert!(r.converged);
        dipole_moment(&mol, &mb, &r).z
    };
    let fd = |h: f64| (at(h) - 2.0 * dipole_moment(&mol, &mb, &scf).z + at(-h)) / h.powi(2);
    let oracle = (4.0 * fd(0.005) - fd(0.01)) / 3.0;
    assert!(
        (oracle - report.value.tensor[26]).abs() < 2e-5 * report.value.tensor[26].abs().max(1.0)
    );
}

#[test]
fn every_tensor_functional_class_has_an_independent_four_step_error_fit() {
    let mol = molecule([0.0, 0.0, 1.0], [0.0; 3]);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let opts = options();
    let steps = [0.02_f64, 0.01, 0.005, 0.0025];
    for name in [
        "svwn5",
        "pbe",
        "b3lyp",
        "cam-b3lyp",
        "r2scan",
        "weighted-local",
    ] {
        let mut f = resolve(&FunctionalSpec::Combined(
            if name == "weighted-local" {
                "svwn5"
            } else {
                name
            }
            .into(),
        ))
        .unwrap();
        if name == "weighted-local" {
            f.description = "unregistered weighted local hybrid fit model".into();
            for (weight, _) in &mut f.parts {
                *weight *= 0.63;
            }
            f.hybrid.cx = 0.37;
        }
        let scf = run_scf(&mol, &mb, &opts, &f, Some(&grid)).unwrap();
        let analytic = dft_rs::response::electric_field_taylor(
            &mol,
            &mb,
            &f,
            Some(&grid),
            &scf,
            &opts,
            [0.0, 0.0, 1.0],
            3,
            &Default::default(),
        )
        .unwrap();
        let mut derivatives = [
            2.0 * analytic.value.dipole_coefficients[2][2],
            6.0 * analytic.value.dipole_coefficients[3][2],
        ];
        let variational = dft_rs::response::static_beta_variational(
            &mol,
            &mb,
            &f,
            Some(&grid),
            &scf,
            &opts,
            &Default::default(),
        )
        .unwrap();
        assert!(variational.converged, "{name}: {:?}", variational.warnings);
        assert!((variational.value.tensor[26] - derivatives[0]).abs() < 1e-8);
        assert_eq!(variational.residuals.len(), 3);
        derivatives[0] = variational.value.tensor[26];
        let at = |field: f64| {
            let mut shifted = opts.clone();
            shifted.external_field = Some(Vec3::new(0.0, 0.0, field));
            let r = run_scf(&mol, &mb, &shifted, &f, Some(&grid)).unwrap();
            assert!(r.converged);
            dipole_moment(&mol, &mb, &r).z
        };
        let mu0 = dipole_moment(&mol, &mb, &scf).z;
        let mut errors = [Vec::new(), Vec::new()];
        for h in steps {
            let (plus, minus, far, back) = (at(h), at(-h), at(2.0 * h), at(-2.0 * h));
            errors[0].push(((plus - 2.0 * mu0 + minus) / h.powi(2) - derivatives[0]).abs());
            errors[1].push(
                ((far - 2.0 * plus + 2.0 * minus - back) / (2.0 * h.powi(3)) - derivatives[1])
                    .abs(),
            );
        }
        for (i, errors) in errors.iter().enumerate() {
            let fit = convergence::error_power_fit(&steps, errors);
            eprintln!("{name} derivative {} four-step fit: p={:.12}, C={:.12e}, R_squared={:.12}, log_rmse={:.12e}",
                i+2, fit.exponent, fit.prefactor, fit.r_squared, fit.log_rmse);
            assert!((fit.exponent - 2.0).abs() < 0.02, "{name}/{}", i + 2);
            assert!(fit.r_squared > 0.99999, "{name}/{}", i + 2);
        }
    }
}
