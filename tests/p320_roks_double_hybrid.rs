// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Restricted-open-shell doubles-only PT2: independent spin-orbital
//! determinant sums, parent/target energy decomposition and limiting cases.
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::{eri_incore, CoulombKernel};
use dft_rs::scf::{roks::run_roks, ScfResult};
use dft_rs::xc::double_hybrid::DoubleHybridFunctional;

fn number(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

fn parameters() -> Value {
    Value::object()
        .with("basis", "sto-3g")
        .with("reference", "roks")
        .with("dispersion", "none")
        .with("grid_level", 1usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-10)
        .with("max_iter", 500usize)
        .with("scf_optimizer", "trah")
}

fn molecule() -> System {
    System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    )
}

/// Enumerate each unique double excitation once. The AO contraction has
/// four explicit loops and does not use production OVOV transformations or
/// ordered-pair multiplicity factors.
fn determinant_sums(
    scf: &ScfResult,
    integral: impl Fn(usize, usize, usize, usize) -> f64,
) -> [f64; 3] {
    let n = scf.overlap.rows;
    let count = [scf.n_alpha, scf.n_beta];
    let occupied: Vec<_> = (0..2)
        .flat_map(|s| (0..count[s]).map(move |i| (s, i)))
        .collect();
    let virtuals: Vec<_> = (0..2)
        .flat_map(|s| (count[s]..scf.mo_coeff[s].cols).map(move |a| (s, a)))
        .collect();
    let chemist = |p: (usize, usize), r: (usize, usize), q: (usize, usize), s: (usize, usize)| {
        if p.0 != r.0 || q.0 != s.0 {
            return 0.0;
        }
        let mut value = 0.0;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        value += scf.mo_coeff[p.0][(a, p.1)]
                            * scf.mo_coeff[r.0][(b, r.1)]
                            * scf.mo_coeff[q.0][(c, q.1)]
                            * scf.mo_coeff[s.0][(d, s.1)]
                            * integral(a, b, c, d);
                    }
                }
            }
        }
        value
    };
    let mut sums = [0.0; 3];
    for i in 0..occupied.len() {
        for j in i + 1..occupied.len() {
            let (p, q) = (occupied[i], occupied[j]);
            for a in 0..virtuals.len() {
                for b in a + 1..virtuals.len() {
                    let (r, s) = (virtuals[a], virtuals[b]);
                    let coupling = chemist(p, r, q, s) - chemist(p, s, q, r);
                    if coupling == 0.0 {
                        continue;
                    }
                    let denominator = scf.mo_energy[r.0][r.1] + scf.mo_energy[s.0][s.1]
                        - scf.mo_energy[p.0][p.1]
                        - scf.mo_energy[q.0][q.1];
                    assert!(denominator > 0.0);
                    let channel = if p.0 != q.0 { 1 } else { 2 * p.0 };
                    sums[channel] -= coupling * coupling / denominator;
                }
            }
        }
    }
    sums
}

#[test]
fn common_parent_doubles_match_independent_determinants_in_every_spin_channel() {
    let system = molecule();
    for named in [true, false] {
        let mut values = parameters().with("charge", 1).with("multiplicity", 2usize);
        values = if named {
            values.with("xc", "b2plyp")
        } else {
            values
                .with("xc", "pbe0")
                .with("double_hybrid_model", "custom")
                .with("pt2_same_spin", 0.37)
                .with("pt2_opposite_spin", 0.82)
        };
        let controls = Controls::from_value(&values).unwrap();
        let report = run("energy", &system, &controls).unwrap();
        // Build the independently evaluated common parent with the same
        // quadrature and physical controls, supplying its typed composition.
        let parent_controls = Controls::from_value(
            &values
                .clone()
                .with("xc", if named { "tpss" } else { "pbe0" }),
        )
        .unwrap();
        let method = MolecularMethod::from_controls(&parent_controls).unwrap();
        let mut setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        if named {
            setup.functional = DoubleHybridFunctional::named("b2plyp").unwrap().parent;
        }
        let scf = run_roks(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap();
        assert!(scf.converged);
        let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 256).unwrap();
        let sums = determinant_sums(&scf, |a, b, c, d| eri.get(a, b, c, d));
        for (key, expected) in [
            "pt2_same_spin_alpha",
            "pt2_opposite_spin",
            "pt2_same_spin_beta",
        ]
        .into_iter()
        .zip(sums)
        {
            let actual = number(&report, key);
            println!("ROKS DH named={named} {key} value={actual:.16e} determinant={expected:.16e} defect={:.12e}", (actual-expected).abs());
            assert!(expected.abs() > 1e-7);
            assert!((actual - expected).abs() < 1e-10);
        }
        let correction = number(&report, "pt2_same_spin_coefficient") * (sums[0] + sums[2])
            + number(&report, "pt2_opposite_spin_coefficient") * sums[1];
        assert!((number(&report, "reference_energy") - scf.energy).abs() < 1e-10);
        assert!((number(&report, "energy") - scf.energy - correction).abs() < 1e-10);
        for key in [
            "reference_common_orbital_residual",
            "reference_semicanonical_residual",
            "pt2_amplitude_residual",
            "pt2_lagrangian_residual",
            "reference_energy_sum_residual",
            "semilocal_component_sum_residual",
        ] {
            let value = number(&report, key);
            println!("ROKS DH named={named} {key}={value:.12e}");
            assert!(value < 1e-9);
        }
        println!("ROKS DH named={named} parent={:.16e} correction={correction:.16e} energy={:.16e} unconstrained_spin_residual={:.12e}",
            scf.energy,number(&report,"energy"),number(&report,"reference_orbital_residual"));
        assert_eq!(report.get("reference_type").unwrap().as_str(), Some("roks"));
    }
}

#[test]
fn common_doubles_closed_zero_correlation_and_one_electron_limits() {
    let closed = System::molecule(vec![1, 1], vec![[0.1, -0.2, 0.3], [1.51, 0.13, -0.27]]);
    let reports = ["rks", "roks"].map(|reference| {
        run(
            "energy",
            &closed,
            &Controls::from_value(
                &parameters()
                    .with("xc", "b2plyp")
                    .with("reference", reference),
            )
            .unwrap(),
        )
        .unwrap()
    });
    for key in [
        "energy",
        "reference_energy",
        "pt2_same_spin_alpha",
        "pt2_same_spin_beta",
        "pt2_opposite_spin",
    ] {
        assert!((number(&reports[0], key) - number(&reports[1], key)).abs() < 1e-10);
    }
    let zero = run(
        "energy",
        &molecule(),
        &Controls::from_value(
            &parameters()
                .with("xc", "pbe0")
                .with("charge", 1)
                .with("multiplicity", 2usize)
                .with("double_hybrid_model", "custom")
                .with("pt2_same_spin", 0.0)
                .with("pt2_opposite_spin", 0.0),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(number(&zero, "energy"), number(&zero, "reference_energy"));
    let atom = run(
        "energy",
        &System::molecule(vec![1], vec![[0.0; 3]]),
        &Controls::from_value(
            &parameters()
                .with("xc", "b2plyp")
                .with("multiplicity", 2usize),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(number(&atom, "pt2_correlation"), 0.0);
    // A total energy provider does not justify an unconstrained derivative.
    let unsupported = run(
        "gradient",
        &molecule(),
        &Controls::from_value(
            &parameters()
                .with("xc", "b2plyp")
                .with("charge", 1)
                .with("multiplicity", 2usize),
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(matches!(unsupported, dft_rs::DftError::Unsupported { .. }));
    println!(
        "ROKS DH limits: closed, zero PT2, one electron and constrained derivative refusal passed"
    );
}
