// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests for isotropic Fermi-contact hyperfine couplings on the
//! common driver: the contact density of a hydrogen atom matches an
//! independent evaluation of its published basis function and the SI form of
//! the coupling; equivalent nuclei carry equal couplings for every functional
//! class, and the couplings are invariant under translation; closed shells
//! and malformed nuclear g lists are refused by name.

use dft_rs::driver::{run, Controls, System, Value};

/// Proton g factor (magnitude), the value used for every hydrogen nucleus.
const PROTON_G: f64 = 5.5856946893;

/// Planar methyl radical (Bohr); carbon first, then three hydrogens.
const METHYL: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [2.04, 0.0, 0.0],
    [-1.02, 1.76669, 0.0],
    [-1.02, -1.76669, 0.0],
];

fn methyl() -> System {
    System::molecule(vec![6, 1, 1, 1], METHYL.to_vec())
}

fn methyl_nuclear_g() -> Value {
    Value::Array(vec![
        Value::Float(1.4048236),
        Value::Float(PROTON_G),
        Value::Float(PROTON_G),
        Value::Float(PROTON_G),
    ])
}

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

/// The common reference controls of every hyperfine request.
fn request(extra: &[(&'static str, Value)]) -> Controls {
    let mut pairs = vec![
        ("basis", Value::Str("sto-3g".into())),
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-10)),
        ("max_iter", Value::Int(300)),
    ];
    pairs.extend(extra.iter().cloned());
    controls(&pairs)
}

/// Doublet reference: open-shell runs add these two controls.
fn doublet() -> Vec<(&'static str, Value)> {
    vec![
        ("multiplicity", Value::Int(2)),
        ("unrestricted", Value::Bool(true)),
    ]
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn assert_close(actual: f64, expected: f64, rtol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= rtol * expected.abs(),
        "{what}: {actual:e} differs from {expected:e} by more than relative {rtol:e}"
    );
}

#[test]
fn hydrogen_atom_reproduces_the_contact_density_of_its_basis_function() {
    let hydrogen = System::molecule(vec![1], vec![[0.3, -0.2, 0.5]]);
    let mut pairs = doublet();
    pairs.push(("nuclear_g", Value::Array(vec![Value::Float(PROTON_G)])));
    let report = run("hyperfine", &hydrogen, &request(&pairs)).unwrap();

    // Independent oracle: the published STO-3G hydrogen 1s function, its value
    // at its own centre and its normalization over the primitive overlaps.
    let a = [3.425250914_f64, 0.6239137298, 0.1688554040];
    let c = [0.1543289673_f64, 0.5353281423, 0.4446345422];
    let phi0: f64 = a
        .iter()
        .zip(&c)
        .map(|(&ai, &ci)| ci * (2.0_f64 * ai / std::f64::consts::PI).powf(0.75))
        .sum();
    let mut norm = 0.0_f64;
    for (&ai, &ci) in a.iter().zip(&c) {
        for (&aj, &cj) in a.iter().zip(&c) {
            norm += ci * cj * (2.0_f64 * (ai * aj).sqrt() / (ai + aj)).powf(1.5);
        }
    }
    let density = phi0 * phi0 / norm;

    let total = vector(&report, "density_at_nuclei")[0];
    let spin = vector(&report, "spin_density_at_nuclei")[0];
    assert_close(total, density, 1e-8, "density at the nucleus");
    assert_close(spin, density, 1e-8, "spin density at the nucleus");
    assert!((number(&report, "two_s") - 1.0).abs() < 1e-12);

    // SI form of the Fermi-contact coupling with the CODATA constants.
    let g_e = 2.00231930436256_f64;
    let mu0 = 1.25663706212e-6_f64;
    let mu_b = 9.2740100783e-24_f64;
    let mu_n = 5.0507837461e-27_f64;
    let h = 6.62607015e-34_f64;
    let a0 = 5.29177210903e-11_f64;
    let expected_mhz =
        (2.0 * mu0 / 3.0) * g_e * mu_b * PROTON_G * mu_n * density / a0.powi(3) / h * 1.0e-6;
    let mhz = vector(&report, "isotropic_coupling_mhz")[0];
    let gauss = vector(&report, "isotropic_coupling_gauss")[0];
    let expected_gauss = expected_mhz / (g_e * 1.39962449361);
    println!(
        "hydrogen density {density:.12e} spin {spin:.12e} coupling {mhz:.9e} MHz \
         (SI {expected_mhz:.9e}) gauss {gauss:.9e} (expected {expected_gauss:.9e})"
    );
    assert_close(mhz, expected_mhz, 1e-7, "isotropic coupling in MHz");
    assert_close(gauss, expected_gauss, 1e-7, "isotropic coupling in gauss");
}

#[test]
fn equivalent_nuclei_carry_equal_couplings_for_every_functional_class() {
    for xc in ["hf", "svwn5", "pbe", "tpss", "b3lyp", "wb97xd", "m06"] {
        let mut pairs = doublet();
        pairs.push(("xc", Value::Str(xc.into())));
        pairs.push(("nuclear_g", methyl_nuclear_g()));
        if xc != "hf" {
            pairs.push(("grid_level", Value::Int(3)));
        }
        let report = run("hyperfine", &methyl(), &request(&pairs)).unwrap();
        let mhz = vector(&report, "isotropic_coupling_mhz");
        let spin = vector(&report, "spin_density_at_nuclei");
        let s_squared = number(&report, "s_squared");
        println!(
            "xc {xc}: isotropic couplings {mhz:?} MHz, spin densities {spin:?}, s_squared {s_squared:.6}"
        );
        assert_eq!(mhz.len(), 4);
        for &hydrogen in &mhz[2..] {
            assert!(
                (hydrogen - mhz[1]).abs() <= 1e-4 * mhz[1].abs(),
                "{xc}: hydrogen couplings {mhz:?} are not equal"
            );
        }
        assert!(mhz[0].abs() > 1.0, "{xc}: carbon coupling {mhz:?}");

        if xc == "hf" {
            // Translation of the whole molecule leaves every coupling unchanged.
            let shift = [1.3, -0.7, 2.1];
            let moved_positions: Vec<[f64; 3]> = METHYL
                .iter()
                .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
                .collect();
            let moved = run(
                "hyperfine",
                &System::molecule(vec![6, 1, 1, 1], moved_positions),
                &request(&pairs),
            )
            .unwrap();
            let moved_mhz = vector(&moved, "isotropic_coupling_mhz");
            println!("hf translated: isotropic couplings {moved_mhz:?} MHz");
            for (reference, shifted) in mhz.iter().zip(&moved_mhz) {
                assert_close(*shifted, *reference, 1e-6, "translated coupling");
            }

            // Without nuclear_g the per-g couplings are reported and the
            // absolute couplings are absent.
            let mut bare = doublet();
            bare.push(("xc", Value::Str("hf".into())));
            let bare_report = run("hyperfine", &methyl(), &request(&bare)).unwrap();
            assert!(bare_report.get("isotropic_coupling_mhz").is_none());
            let per_g = vector(&bare_report, "isotropic_coupling_per_g_mhz");
            assert_close(
                per_g[1] * PROTON_G,
                mhz[1],
                1e-8,
                "per-g hydrogen coupling times the proton g factor",
            );
        }
    }
}

#[test]
fn closed_shells_and_malformed_requests_are_refused() {
    let water = System::molecule(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.1173],
            [0.0, 1.4332, -0.9314],
            [0.0, -1.4332, -0.9314],
        ],
    );
    let closed = run("hyperfine", &water, &request(&[]))
        .unwrap_err()
        .to_string();
    assert!(closed.contains("open-shell"), "{closed}");

    let hydrogen = System::molecule(vec![1], vec![[0.0, 0.0, 0.0]]);
    let mut pairs = doublet();
    pairs.push((
        "nuclear_g",
        Value::Array(vec![Value::Float(PROTON_G), Value::Float(PROTON_G)]),
    ));
    let malformed = run("hyperfine", &hydrogen, &request(&pairs))
        .unwrap_err()
        .to_string();
    assert!(malformed.contains("nuclear_g"), "{malformed}");
}

#[test]
fn nuclei_with_an_effective_core_potential_report_the_pseudo_orbital_value_separately() {
    // Magnesium monohydride radical: magnesium carries an effective core
    // potential in this basis, hydrogen is all-electron.
    let radical = System::molecule(vec![12, 1], vec![[0.0, 0.0, 0.0], [0.2, 0.1, 3.27]]);
    let mut request = doublet();
    request.push(("basis", Value::Str("lanl2dz".into())));
    request.push((
        "nuclear_g",
        Value::Array(vec![Value::Float(-0.34218), Value::Float(PROTON_G)]),
    ));
    let pairs: Vec<(&str, Value)> = request
        .into_iter()
        .chain([
            ("conv_tol", Value::Float(1e-12)),
            ("conv_tol_dp", Value::Float(1e-10)),
            ("max_iter", Value::Int(300)),
        ])
        .collect();
    let report = run(
        "hyperfine",
        &radical,
        &Controls::from_value(&Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        ))
        .unwrap(),
    )
    .unwrap();
    for key in [
        "total_coupling_per_g_mhz",
        "total_coupling_mhz",
        "principal_couplings_per_g_mhz",
        "principal_couplings_mhz",
        "principal_axes_per_g",
        "principal_axes",
        "tensor_converged",
        "dipolar_trace_per_g_mhz",
        "tensor_eigen_residual_per_g_mhz",
    ] {
        let entries = report.get(key).unwrap().as_array().unwrap();
        assert!(
            matches!(entries[0], Value::Null),
            "{key}: ECP tensor must be null"
        );
        assert!(
            !matches!(entries[1], Value::Null),
            "{key}: all-electron tensor must be present"
        );
    }
    let entry = |key: &str, atom: usize| report.get(key).unwrap().as_array().unwrap()[atom].clone();
    // Magnesium: no contact density and no coupling, but the pseudo-orbital value.
    assert!(matches!(entry("spin_density_at_nuclei", 0), Value::Null));
    assert!(matches!(entry("isotropic_coupling_mhz", 0), Value::Null));
    let pseudo = entry("pseudo_valence_spin_density_at_nuclei", 0)
        .as_f64()
        .unwrap();
    // Hydrogen: all-electron values, and no pseudo-orbital entry.
    let hydrogen = entry("isotropic_coupling_mhz", 1).as_f64().unwrap();
    assert!(matches!(
        entry("pseudo_valence_spin_density_at_nuclei", 1),
        Value::Null
    ));
    let warnings = report.get("warnings").unwrap().as_array().unwrap();
    println!(
        "MgH: pseudo-orbital spin density at Mg {pseudo:.6e}, hydrogen coupling {hydrogen:.4} MHz, \
         warnings {}",
        warnings.len()
    );
    assert!(pseudo.is_finite() && hydrogen.is_finite() && hydrogen.abs() > 1.0);
    assert!(warnings.iter().any(|w| w
        .as_str()
        .is_some_and(|s| s.contains("effective core potential"))));
}

#[test]
fn epr_unconnected_references_fail_before_setup_without_consuming_the_selector() {
    let hydrogen = System::molecule(vec![1], vec![[0.0; 3]]);
    for feature in ["hyperfine", "g-tensor"] {
        for kind in ["roks", "rohf", "gks", "ghf", "RoKs", "GhF"] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("reference", kind)
                    .with("basis", "missing-epr-preflight-basis"),
            )
            .unwrap();
            let error = run(feature, &hydrogen, &controls).unwrap_err();
            assert!(matches!(error, dft_rs::DftError::Unsupported { .. }));
            let message = error.to_string();
            assert!(
                message.contains(&format!("{feature} with reference={kind}")),
                "{message}"
            );
            assert!(
                message.contains("cannot replace the requested reference"),
                "{message}"
            );
            let unused = controls
                .finish("rejected EPR request")
                .unwrap_err()
                .to_string();
            assert!(unused.contains("reference"), "{unused}");
        }
    }
}

#[test]
fn epr_invalid_reference_selectors_fail_before_basis_setup() {
    let hydrogen = System::molecule(vec![1], vec![[0.0; 3]]);
    let closed_shell = System::molecule(vec![1, 1], vec![[0.0; 3], [0.0, 0.0, 1.4]]);
    for feature in ["hyperfine", "g-tensor"] {
        for reference in [
            Value::Str("rkos".into()),
            Value::Str(String::new()),
            Value::Int(42),
            Value::Bool(true),
            Value::Array(Vec::new()),
        ] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("reference", reference)
                    .with("basis", "missing-epr-preflight-basis"),
            )
            .unwrap();
            let error = run(feature, &hydrogen, &controls).unwrap_err();
            assert!(matches!(error, dft_rs::DftError::InvalidInput(_)));
            let message = error.to_string();
            assert!(message.contains("reference"), "{message}");
            assert!(
                !message.contains("missing-epr-preflight-basis"),
                "{message}"
            );
        }
        for reference in ["auto", "rks", "rhf", "uks", "uhf", "UKS"] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("reference", reference)
                    .with("basis", "missing-epr-preflight-basis"),
            )
            .unwrap();
            let error = run(feature, &closed_shell, &controls)
                .unwrap_err()
                .to_string();
            assert!(error.contains("missing-epr-preflight-basis"), "{error}");
        }
        let null_reference = Controls::from_value(
            &Value::object()
                .with("reference", Value::Null)
                .with("basis", "missing-epr-preflight-basis"),
        )
        .unwrap();
        let error = run(feature, &closed_shell, &null_reference)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing-epr-preflight-basis"), "{error}");
    }
}

#[test]
fn epr_supported_open_shell_selectors_preserve_the_automatic_collinear_result() {
    let hydrogen = System::molecule(vec![1], vec![[0.3, -0.2, 0.5]]);
    for (feature, key) in [
        ("hyperfine", "total_coupling_per_g_mhz"),
        ("g-tensor", "g_tensor"),
    ] {
        let base = Value::object()
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("multiplicity", 2)
            .with("cpu_threads", 1);
        let expected = run(feature, &hydrogen, &Controls::from_value(&base).unwrap()).unwrap();
        let expected_tensor = expected.get(key).unwrap().numeric_array().unwrap();
        for kind in ["auto", "uks", "uhf", "UkS"] {
            let mut selected = base.clone();
            selected.set("reference", kind);
            let report = run(
                feature,
                &hydrogen,
                &Controls::from_value(&selected).unwrap(),
            )
            .unwrap();
            assert_eq!(report.get("converged"), Some(&Value::Bool(true)));
            assert!(report
                .get("reference_type")
                .and_then(Value::as_str)
                .unwrap()
                .eq_ignore_ascii_case(kind));
            assert!((number(&report, "energy") - number(&expected, "energy")).abs() < 1e-12);
            let actual = report.get(key).unwrap().numeric_array().unwrap();
            assert_eq!(actual.0, expected_tensor.0);
            for (a, b) in actual.1.iter().zip(&expected_tensor.1) {
                assert!((a - b).abs() < 1e-9, "{feature}/{kind}: {a} versus {b}");
            }
        }
    }
}
