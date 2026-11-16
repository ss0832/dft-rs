// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Compare named energies with independent two-center Gaussian integrals
//! and cylindrical quadrature on symmetry-resolved H2/STO-3G orbitals.
//! Ammonia supplies an independent pair sum with nonzero same-spin energy.

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{eri::eri_incore, kernels::CoulombKernel};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::double_hybrid::DoubleHybridFunctional;

fn controls(model: &str, grid_level: usize) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("reference", "rks")
            .with("basis", "sto-3g")
            .with("xc", model)
            .with("grid_level", grid_level)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-12)
            .with("conv_tol_grad", 1e-12),
    )
    .unwrap()
}

fn calculate(feature: &str, model: &str) -> Value {
    let h2 = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    let report = run(feature, &h2, &controls(model, 9)).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    report
}

fn read(report: &Value, key: &str) -> f64 {
    report
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing {key}"))
}

fn check(model: &str, report: &Value, key: &str, expected: f64, tolerance: f64) {
    let actual = read(report, key);
    let error = (actual - expected).abs();
    println!("{model} {key}: actual={actual:.14e} oracle={expected:.14e} error={error:.6e}");
    assert!(error < tolerance);
}

#[test]
fn hsesol_total_matches_independent_hydrogen_energy() {
    let report = calculate("energy", "hsesol");
    // At 1.4 Bohr the occupied STO-3G orbital is fixed by inversion
    // symmetry. The oracle uses normalized primitive s Gaussian products,
    // the closed-form Coulomb and range-separated integrals, and separate
    // 240-by-240 Gauss-Legendre integration of the three semilocal terms.
    check("hsesol", &report, "energy", -1.140_838_811_287_166, 5e-7);
    let dipole = calculate("dipole", "hsesol");
    check(
        "hsesol dipole reference",
        &dipole,
        "energy",
        read(&report, "energy"),
        1e-12,
    );
    let residual = read(&dipole, "scf_orbital_residual");
    println!("hsesol dipole reference orbital residual = {residual:.10e}");
    assert!(residual < 1e-10);
}

#[test]
fn screened_and_xdh_double_hybrids_match_independent_hydrogen_terms() {
    // The antibonding orbital is fixed by odd parity. Its eigenvalue and
    // that of the occupied orbital use the diagonal Gaussian Fock entries
    // and independent fourth-order population derivatives of semilocal
    // energies. The unscaled OS energy is (gu|gu)^2/[2(e_g-e_u)].
    let oracles = [
        (
            "wb97x-2-lp",
            [
                -1.140_974_335_056_456,
                -1.140_974_335_056_456,
                -1.147_648_039_793_244,
                -0.554_018_382_143_552,
                -0.144_835_713_492_031,
                -0.013_961_959_063_725,
            ],
        ),
        (
            "wb97x-2-tqz",
            [
                -1.137_569_573_473_623,
                -1.137_569_573_473_623,
                -1.143_857_540_697_865,
                -0.538_014_345_669_473,
                -0.157_434_988_383_278,
                -0.014_063_737_207_685,
            ],
        ),
        (
            "xdh-pbe0",
            [
                -1.154_310_954_693_99,
                -1.138_680_911_354_01,
                -1.148_926_066_046_099,
                -0.562_274_170_476_707,
                -0.134_286_501_456_425,
                -0.018_874_640_184_394,
            ],
        ),
    ];
    let keys = [
        "orbital_reference_energy",
        "reference_energy",
        "energy",
        "exact_exchange_energy",
        "semilocal_energy",
        "pt2_opposite_spin",
    ];
    for (name, expected) in oracles {
        let report = calculate("energy", name);
        for (key, want) in keys.into_iter().zip(expected) {
            if key == "semilocal_energy" {
                let Value::Array(parts) = report.get("semilocal_components").unwrap() else {
                    panic!("missing semilocal energy components");
                };
                let actual: f64 = parts.iter().map(|part| read(part, "energy")).sum();
                println!("{name} {key}: actual={actual:.14e} oracle={want:.14e}");
                assert!((actual - want).abs() < 5e-7, "{name} {key}");
            } else {
                check(name, &report, key, want, 5e-7);
            }
        }
        check(
            name,
            &report,
            "one_electron_energy",
            -2.505_594_125_216_381,
            1e-8,
        );
        check(name, &report, "hartree_energy", 1.349_188_171_509_794, 1e-8);
        assert_eq!(read(&report, "pt2_same_spin_alpha"), 0.0);
        assert_eq!(read(&report, "pt2_same_spin_beta"), 0.0);
        println!(
            "{name} reference orbital / PT2 residuals = {:.10e} / {:.10e}",
            read(&report, "reference_orbital_residual"),
            read(&report, "pt2_amplitude_residual")
        );
    }
}

#[test]
fn ammonia_has_independent_nonzero_same_and_opposite_spin_pair_sums() {
    let positions = [
        [0.13, -0.27, 0.19],
        [1.78, 0.34, -0.08],
        [-0.49, 1.65, 0.41],
        [0.31, -0.78, 1.91],
    ];
    let numbers = [7, 1, 1, 1];
    let molecule = Molecule::new(
        numbers
            .into_iter()
            .zip(positions)
            .map(|(atomic_number, [x, y, z])| Atom {
                z: atomic_number,
                position: Vec3::new(x, y, z),
            })
            .collect(),
    );
    let system = System::molecule(numbers.to_vec(), positions.to_vec());
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let eri = eri_incore(&basis, CoulombKernel::Coulomb, 16).unwrap();
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: 0,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let scf_options = ScfOptions {
        reference: Reference::Rhf,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        ..ScfOptions::default()
    };
    for model in ["wb97x-2-lp", "wb97x-2-tqz", "xdh-pbe0"] {
        let recipe = DoubleHybridFunctional::named(model).unwrap();
        let parent = run_scf(&molecule, &basis, &scf_options, &recipe.parent, Some(&grid)).unwrap();
        assert!(parent.converged);
        let c = &parent.mo_coeff[0];
        let eps = &parent.mo_energy[0];
        let nocc = parent.n_alpha;
        let nao = eri.nao;

        // Transform each occupied/virtual quartet by an explicit AO sum.
        // This oracle shares the converged parent and bare AO integrals with
        // production. The MO transform and pair summation are independent.
        let quartet = |i, a, j, b| {
            let mut value = 0.0;
            for p in 0..nao {
                for q in 0..nao {
                    for r in 0..nao {
                        for s in 0..nao {
                            value +=
                                c[(p, i)] * c[(q, a)] * eri.get(p, q, r, s) * c[(r, j)] * c[(s, b)];
                        }
                    }
                }
            }
            value
        };
        let mut same = 0.0;
        let mut opposite = 0.0;
        for i in 0..nocc {
            for j in 0..nocc {
                for a in nocc..eps.len() {
                    for b in nocc..eps.len() {
                        let direct = quartet(i, a, j, b);
                        let denominator = eps[i] + eps[j] - eps[a] - eps[b];
                        opposite += direct * direct / denominator;
                        if i > j && a > b {
                            // Each unordered same-spin pair is counted once.
                            let antisymmetrized = direct - quartet(i, b, j, a);
                            same += antisymmetrized * antisymmetrized / denominator;
                        }
                    }
                }
            }
        }
        assert!(same < -1e-4 && opposite < -1e-3);
        let report = run("energy", &system, &controls(model, 0)).unwrap();
        check(model, &report, "pt2_same_spin_alpha", same, 2e-10);
        check(model, &report, "pt2_same_spin_beta", same, 2e-10);
        check(model, &report, "pt2_opposite_spin", opposite, 2e-10);
        check(
            model,
            &report,
            "pt2_correlation",
            2.0 * recipe.correlation.same_spin * same + recipe.correlation.opposite_spin * opposite,
            2e-10,
        );
    }
}

#[test]
fn generalized_nuclear_routes_refuse_missing_physical_providers() {
    let h2 = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    for model in ["wb97x-2-lp", "wb97x-2-tqz", "xdh-pbe0"] {
        let request = controls(model, 0).to_value().with("reference", "gks");
        let controls = Controls::from_value(&request).unwrap();
        let error = run("hessian", &h2, &controls).unwrap_err();
        assert!(
            matches!(error, dft_rs::DftError::Unsupported { feature, missing }
            if feature == "generalized double-hybrid hessian"
                && missing.contains("spinor PT2 nuclear response"))
        );
    }

    let request = controls("xdh-pbe0", 0).to_value().with("reference", "gks");
    let controls = Controls::from_value(&request).unwrap();
    let error = run("gradient", &h2, &controls).unwrap_err();
    assert!(
        matches!(error, dft_rs::DftError::Unsupported { feature, missing }
        if feature == "noncollinear double hybrid with unequal spin coefficients"
            && missing.contains("spin-covariant pair model"))
    );
}
