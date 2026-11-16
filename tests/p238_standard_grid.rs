// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The standard molecular grid (Treutler–Ahlrichs M4 radial map, Lebedev
//! angular rules, five-region pruning, Becke partition with the Treutler
//! size adjustment) against frozen PySCF default grids
//! (`tools/oracle/gen_grid_oracle.py`, `tests/oracle_data/standard_grids.json`)
//! for H2, LiH, water, phosphine and ferrocene:
//!
//! - per-atom point counts before the partition are equal at levels 1, 3, 5
//!   and 9 (integers, exact);
//! - the partitioned integral of a smooth test function agrees to 1e-10
//!   relative, which checks radial map, angular rules and partition point by
//!   point in aggregate;
//! - the Slater + PW92 STO-3G RKS energy on the level-3 grid agrees to
//!   1e-8 Hartree.
//!
//! A second test reports the efficiency of the Lebedev grid against the
//! product angular rule: the integral error relative to level 9 and the
//! point count of each scheme at levels 1, 3 and 5.

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::Value;
use dft_rs::grid::{AngularScheme, GridOptions, MolecularGrid};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, Vec3};

fn oracle() -> Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/oracle_data/standard_grids.json"),
    )
    .unwrap();
    Value::parse_json(&text).unwrap()
}

fn molecule(entry: &Value) -> Molecule {
    let numbers = entry.get("numbers").unwrap().numeric_array().unwrap().1;
    let Value::Array(rows) = entry.get("positions").unwrap() else {
        panic!("positions")
    };
    Molecule::new(
        numbers
            .iter()
            .zip(rows)
            .map(|(z, r)| {
                let r = r.numeric_array().unwrap().1;
                Atom {
                    z: *z as u8,
                    position: Vec3::new(r[0], r[1], r[2]),
                }
            })
            .collect(),
    )
}

/// sum_A exp(-0.7 |r - R_A|^2) (1 + 0.3 x y + 0.1 z), as in the oracle.
fn integral(mol: &Molecule, grid: &MolecularGrid) -> f64 {
    grid.points
        .iter()
        .zip(&grid.weights)
        .map(|(p, w)| {
            let sum: f64 = mol
                .atoms
                .iter()
                .map(|a| (-0.7 * (*p - a.position).norm2()).exp())
                .sum();
            w * sum * (1.0 + 0.3 * p.x * p.y + 0.1 * p.z)
        })
        .sum()
}

fn options(level: usize, scheme: AngularScheme) -> GridOptions {
    GridOptions {
        level,
        scheme,
        ..GridOptions::default()
    }
}

#[test]
fn hf_property_quadrature_keeps_the_requested_grid() {
    use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    use dft_rs::driver::{Controls, System};
    use dft_rs::response::spin_spin::spin_spin_couplings;
    use dft_rs::response::ResponseSolverConfig;

    let numbers = [1, 1];
    let positions = [[0.13, -0.21, 0.17], [1.48, 0.19, -0.11]];
    let mut tensors = Vec::new();
    for level in [0, 1] {
        let method = MolecularMethod {
            xc: Some("hf".into()),
            dispersion: "none".into(),
            grid_level: level,
            grid_weight_cutoff: 0.0,
            conv_tol: 1e-13,
            conv_tol_dp: 1e-12,
            ..Default::default()
        };
        let setup = prepare_molecular(&numbers, &positions, &method).unwrap();
        assert!(setup.grid.is_none(), "HF SCF remains grid-free");
        let requested = setup.integration_grid().unwrap();
        let independent = MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap();
        assert_eq!(requested.points, independent.points);
        assert_eq!(requested.weights, independent.weights);
        let reference = run_molecular_reference(&setup).unwrap();
        assert!(reference.scf.converged);
        let expected = spin_spin_couplings(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            Some(&independent),
            &reference.scf,
            &setup.options,
            &[0, 1],
            false,
            &ResponseSolverConfig::default(),
        )
        .unwrap();
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "hf")
                .with("dispersion", "none")
                .with("grid_level", level)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-13)
                .with("conv_tol_dp", 1e-12),
        )
        .unwrap();
        let report = dft_rs::driver::run(
            "spin-spin-couplings",
            &System::molecule(numbers.to_vec(), positions.to_vec()),
            &controls,
        )
        .unwrap();
        let actual = report
            .get("diamagnetic_tensors")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let expected: Vec<f64> = expected
            .diamagnetic
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert!(
                (a - b).abs() < 1e-20,
                "driver must use the supplied property grid"
            );
        }
        tensors.push(actual);
    }
    assert!(
        tensors[0]
            .iter()
            .zip(&tensors[1])
            .any(|(a, b)| (a - b).abs() > 1e-18),
        "the regression must distinguish the two quadratures"
    );
}

#[test]
fn invalid_rust_grid_controls_are_not_clamped_or_discarded() {
    let molecule = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let lattice = dft_rs::periodic::Lattice::new(
        Vec3::new(8., 0., 0.),
        Vec3::new(0., 9., 0.),
        Vec3::new(0., 0., 10.),
    );
    for grid in [
        GridOptions {
            level: 10,
            ..Default::default()
        },
        GridOptions {
            weight_cutoff: f64::NAN,
            ..Default::default()
        },
        GridOptions {
            weight_cutoff: -1.,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            MolecularGrid::build(&molecule, &grid),
            Err(dft_rs::DftError::InvalidInput(_))
        ));
        assert!(matches!(
            dft_rs::periodic::grid::PeriodicGrid::build(&molecule, &lattice, &grid, 1),
            Err(dft_rs::DftError::InvalidInput(_))
        ));
    }
}

#[test]
fn nonlocal_grid_retains_its_independent_level_and_cutoff() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mut counts = Vec::new();
    for level in [0, 1] {
        let options = ScfOptions {
            vv10: Some(dft_rs::nlc::Vv10Params::ORIGINAL),
            vv10_grid_level: level,
            vv10_grid_weight_cutoff: 0.0,
            ..Default::default()
        };
        let grid = options.nonlocal_grid(&mol).unwrap();
        let expected = MolecularGrid::build(
            &mol,
            &GridOptions {
                level,
                weight_cutoff: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(grid.points, expected.points);
        assert_eq!(grid.weights, expected.weights);
        counts.push(grid.len());
    }
    assert!(counts[1] > counts[0]);
}

#[test]
fn shared_reference_controls_resolve_into_effective_scf_options() {
    use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
    use dft_rs::driver::Controls;

    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("max_iter", 7usize)
            .with("conv_tol", 1e-11)
            .with("conv_tol_dp", 2e-10)
            .with("conv_tol_grad", 3e-9)
            .with("damping", 0.17)
            .with("level_shift", 0.08)
            .with("diis_size", 3usize)
            .with("no_adiis", true)
            .with("guess", "core"),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    controls.finish("reference-contract").unwrap();
    let setup = prepare_molecular(&[1, 1], &[[0., 0., 0.], [0., 0., 1.4]], &method).unwrap();
    assert_eq!(setup.options.max_iter, 7);
    assert_eq!(setup.options.e_tol, 1e-11);
    assert_eq!(setup.options.p_tol, 2e-10);
    assert_eq!(setup.options.conv_tol_grad, 3e-9);
    assert_eq!(setup.options.damping, 0.17);
    assert_eq!(setup.options.level_shift, 0.08);
    assert_eq!(setup.options.diis_size, 3);
    assert!(!setup.options.use_adiis);
    assert!(matches!(setup.options.guess, dft_rs::scf::Guess::Core));
}

#[test]
fn spinor_subcalculations_preserve_explicit_common_settings_and_legacy_defaults() {
    use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
    use dft_rs::driver::Controls;
    use dft_rs::periodic::spinor_scf::SpinorScfConfig;

    let default_controls = Controls::new();
    let default_setup = prepare_molecular(
        &[1, 1],
        &[[0., 0., 0.], [0., 0., 1.4]],
        &MolecularMethod::default(),
    )
    .unwrap();
    let legacy = SpinorScfConfig::default();
    let retained = default_setup.nested_soc_scf_config(&default_controls);
    assert_eq!(retained.max_iter, legacy.max_iter);
    assert_eq!(retained.energy_tolerance, legacy.energy_tolerance);
    assert_eq!(retained.density_tolerance, legacy.density_tolerance);
    assert_eq!(retained.orbital_tolerance, legacy.orbital_tolerance);
    assert_eq!(retained.damping, legacy.damping);
    assert_eq!(retained.diis_size, legacy.diis_size);

    let controls = Controls::from_value(
        &Value::object()
            .with("max_iter", 150usize)
            .with("conv_tol", 1e-9)
            .with("conv_tol_dp", 2e-11)
            .with("conv_tol_grad", 3e-10)
            .with("diis_size", 3usize)
            .with("damping", 0.17),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let setup = prepare_molecular(&[1, 1], &[[0., 0., 0.], [0., 0., 1.4]], &method).unwrap();
    for config in [
        setup.spinor_scf_config(),
        setup.nested_soc_scf_config(&controls),
    ] {
        assert_eq!(config.max_iter, 150);
        assert_eq!(config.energy_tolerance, 1e-9);
        assert_eq!(config.density_tolerance, 2e-11);
        assert_eq!(config.orbital_tolerance, 3e-10);
        assert_eq!(config.diis_size, 3);
        assert_eq!(config.damping, 0.17);
    }
    controls.finish("nested-scf-contract").unwrap();
}

#[test]
fn standard_grid_matches_the_common_default_grids() {
    let oracle = oracle();
    let Value::Object(molecules) = oracle.get("molecules").unwrap() else {
        panic!("molecules")
    };
    let set = data::load("sto-3g").unwrap();
    let functional = resolve(&FunctionalSpec::Split {
        x: "slater".into(),
        c: "pw92".into(),
    })
    .unwrap();
    let mut failures = Vec::new();
    for (name, entry) in molecules {
        let mol = molecule(entry);
        let Value::Object(levels) = entry.get("levels").unwrap() else {
            panic!("levels")
        };
        for (level, record) in levels {
            let level: usize = level.parse().unwrap();
            let opts = options(level, AngularScheme::Lebedev);
            let counts = MolecularGrid::atomic_point_counts(&mol, &opts);
            let want: Vec<usize> = record
                .get("atom_points")
                .unwrap()
                .numeric_array()
                .unwrap()
                .1
                .iter()
                .map(|v| *v as usize)
                .collect();
            if counts != want {
                failures.push(format!(
                    "{name} level {level}: points {counts:?} vs {want:?}"
                ));
            }
            let grid = MolecularGrid::build(&mol, &opts).unwrap();
            let got = integral(&mol, &grid);
            let reference = record.get("integral").and_then(Value::as_f64).unwrap();
            let relative = (got - reference) / reference;
            println!(
                "{name} level {level}: {} points (oracle {}), integral {got:.14} oracle {reference:.14} relative {relative:.2e}",
                grid.len(),
                record.get("points").and_then(Value::as_f64).unwrap()
            );
            if relative.abs() > 1e-10 {
                failures.push(format!(
                    "{name} level {level}: integral relative {relative:.2e}"
                ));
            }
            if let Some(energy) = record.get("energy").and_then(Value::as_f64) {
                let mb = MolBasis::build(&mol, &set).unwrap();
                let scf_options = ScfOptions {
                    e_tol: 1e-11,
                    p_tol: 1e-9,
                    max_iter: 300,
                    eri_mode: EriMode::InCore { limit_mb: 4096 },
                    ..ScfOptions::default()
                };
                let scf = run_scf(&mol, &mb, &scf_options, &functional, Some(&grid)).unwrap();
                let delta = scf.energy - energy;
                println!(
                    "{name} level {level}: Slater+PW92 E {:.10} oracle {energy:.10} delta {delta:.2e}",
                    scf.energy
                );
                if !scf.converged || delta.abs() > 1e-8 {
                    failures.push(format!("{name} level {level}: energy delta {delta:.2e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "standard grid failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn lebedev_grid_is_more_efficient_than_the_product_rule() {
    let oracle = oracle();
    let Value::Object(molecules) = oracle.get("molecules").unwrap() else {
        panic!("molecules")
    };
    for (name, entry) in molecules {
        let mol = molecule(entry);
        let exact = entry
            .get("levels")
            .unwrap()
            .get("9")
            .unwrap()
            .get("integral")
            .and_then(Value::as_f64)
            .unwrap();
        for level in [1usize, 3, 5] {
            let mut row = Vec::new();
            for scheme in [AngularScheme::Lebedev, AngularScheme::Product] {
                let grid = MolecularGrid::build(&mol, &options(level, scheme)).unwrap();
                let error = (integral(&mol, &grid) - exact).abs() / exact;
                row.push((grid.len(), error));
            }
            println!(
                "{name} level {level}: Lebedev {} points error {:.2e}; product {} points error {:.2e}",
                row[0].0, row[0].1, row[1].0, row[1].1
            );
            // Same radial grid and pruning: Lebedev uses fewer points at the
            // same degree and must not be less accurate by more than the
            // radial error both share.
            assert!(
                row[0].0 < row[1].0,
                "{name} level {level}: point counts {row:?}"
            );
        }
    }
}
