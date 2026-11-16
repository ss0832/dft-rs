// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::hessian::analytic_hessian_full;
use dft_rs::math::Vec3;
use dft_rs::response::nuclear::{
    mixed_molecular_response, molecular_taylor, static_spectroscopic_derivative, MolecularDirection,
};
use dft_rs::response::polarizability::polarizability;
use dft_rs::response::{electric_field_taylor, ResponseSolverConfig};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn molecule(r: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::new(0.1, -0.2, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1, -0.2, r),
        },
    ])
    .with_charge(1)
}
fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..Default::default()
    }
}
fn grid_options() -> GridOptions {
    GridOptions {
        level: 1,
        prune: false,
        weight_cutoff: 0.0,
        ..Default::default()
    }
}

#[test]
fn field_taylor_and_energy_dipole_derivatives_match_the_independent_fixed_basis_driver() {
    let mol = molecule(1.5);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = ScfOptions {
        external_field: Some(Vec3::new(0.0, 0.0, 0.02)),
        ..options()
    };
    let config = ResponseSolverConfig::default();
    for name in ["hf", "pbe", "r2scan", "cam-b3lyp"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let grid = (!functional.parts.is_empty())
            .then(|| MolecularGrid::build(&mol, &grid_options()).unwrap());
        let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).unwrap();
        let direction = MolecularDirection::electric(mol.len(), [0.0, 0.0, 0.5]);
        let moving = molecular_taylor(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            4,
            &config,
        )
        .unwrap();
        let fixed = electric_field_taylor(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            direction.electric,
            4,
            &config,
        )
        .unwrap();
        assert!(
            moving.converged,
            "{name}: {:?}, {:?}",
            moving.warnings, moving.value.stationarity_defects
        );
        for n in 0..=4 {
            for axis in 0..3 {
                assert!(
                    (moving.value.dipole_coefficients[n][axis]
                        - fixed.value.dipole_coefficients[n][axis])
                        .abs()
                        < 1e-8,
                    "{name} order {n}"
                );
            }
        }
        for n in 0..4 {
            assert!(
                ((n + 1) as f64 * moving.value.energy_coefficients[n + 1]
                    + 0.5 * moving.value.dipole_coefficients[n][2])
                    .abs()
                    < 1e-8,
                "{name} energy order {}",
                n + 1
            );
        }
        let nucleus = -(2.0 * mol.atoms[0].position.z + mol.atoms[1].position.z) * 0.02;
        assert!((moving.value.energy_components[0].nuclear_field - nucleus).abs() < 1e-13);
    }
}

#[test]
fn apt_maxwell_and_charged_translation_relations_include_explicit_operator_terms() {
    let mol = molecule(1.5);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let config = ResponseSolverConfig::default();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let direction = vec![[0.0; 3], [0.0, 0.0, 1.0]];
    let apt = static_spectroscopic_derivative(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &direction,
        1,
        &config,
    )
    .unwrap();
    let full = analytic_hessian_full(&mol, &mb, &opts, &functional, &scf, None).unwrap();
    assert!(apt.converged);
    for a in 0..3 {
        assert!((apt.value.tensor[a] - full.dipole_derivatives[(a, 5)]).abs() < 1e-10);
    }
    let mixed = mixed_molecular_response(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &[
            MolecularDirection::nuclear(direction),
            MolecularDirection::electric(2, [0.0, 0.0, 1.0]),
        ],
        &config,
    )
    .unwrap();
    assert!((mixed.value.energy_derivative + apt.value.tensor[2]).abs() < 1e-9);
    let uniform = vec![[0.3, -0.7, 1.2]; 2];
    let apt = static_spectroscopic_derivative(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &uniform,
        1,
        &config,
    )
    .unwrap();
    for a in 0..3 {
        assert!((apt.value.tensor[a] - uniform[0][a] * mol.charge as f64).abs() < 1e-10);
    }
    let opts = ScfOptions {
        external_field: Some(Vec3::new(0.1, -0.02, 0.03)),
        ..options()
    };
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let report = molecular_taylor(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &MolecularDirection::nuclear(uniform),
        3,
        &config,
    )
    .unwrap();
    let expected = -mol.charge as f64 * (0.3 * 0.1 + (-0.7) * (-0.02) + 1.2 * 0.03);
    assert!((report.value.energy_coefficients[1] - expected).abs() < 1e-10);
    for n in 2..=3 {
        assert!(report.value.energy_coefficients[n].abs() < 1e-10);
    }
}

fn spectra_at(r: f64, name: &str) -> (f64, f64) {
    let mol = molecule(r);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let grid = (!functional.parts.is_empty())
        .then(|| MolecularGrid::build(&mol, &grid_options()).unwrap());
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).unwrap();
    let alpha = polarizability(&mol, &mb, &functional, grid.as_ref(), &scf, &opts).unwrap()[2][2];
    let beta = 2.0
        * electric_field_taylor(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            [0.0, 0.0, 1.0],
            2,
            &Default::default(),
        )
        .unwrap()
        .value
        .dipole_coefficients[2][2];
    (alpha, beta)
}

#[test]
fn analytic_raman_and_hyperraman_match_displaced_cpks_and_nonlinear_field_oracles() {
    let direction = vec![[0.0; 3], [0.0, 0.0, 1.0]];
    for name in ["hf", "pbe"] {
        let mol = molecule(1.5);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let grid = (!functional.parts.is_empty())
            .then(|| MolecularGrid::build(&mol, &grid_options()).unwrap());
        let opts = options();
        let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).unwrap();
        let config = ResponseSolverConfig::default();
        let raman = static_spectroscopic_derivative(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            2,
            &config,
        )
        .unwrap();
        let hyper = static_spectroscopic_derivative(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            3,
            &config,
        )
        .unwrap();
        eprintln!(
            "{name} analytic Raman zz {}, hyper-Raman zzz {}, permutation defects {}, {}",
            raman.value.tensor[8],
            hyper.value.tensor[26],
            raman.value.permutation_defect,
            hyper.value.permutation_defect
        );
        assert!(raman.converged, "{name}: {:?}", raman.warnings);
        assert!(hyper.converged, "{name}: {:?}", hyper.warnings);
        let estimate = |h: f64| {
            let p = spectra_at(1.5 + h, name);
            let m = spectra_at(1.5 - h, name);
            [(p.0 - m.0) / (2.0 * h), (p.1 - m.1) / (2.0 * h)]
        };
        let coarse = estimate(0.01);
        let fine = estimate(0.005);
        let oracle = std::array::from_fn::<_, 2, _>(|i| (4.0 * fine[i] - coarse[i]) / 3.0);
        eprintln!(
            "{name} Richardson Raman {}, hyper-Raman {}",
            oracle[0], oracle[1]
        );
        assert!((raman.value.tensor[8] - oracle[0]).abs() < 2e-6, "{name}");
        assert!((hyper.value.tensor[26] - oracle[1]).abs() < 2e-5, "{name}");
    }
}

#[test]
fn general_cartesian_hyperraman_obeys_all_static_electric_permutations() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.1, -0.1, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.5, 0.2, 0.3),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.2, 1.7, -0.4),
        },
    ])
    .with_charge(1);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let report = static_spectroscopic_derivative(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        &[[0.0; 3], [0.4, -0.2, 0.3], [-0.1, 0.3, 0.2]],
        3,
        &Default::default(),
    )
    .unwrap();
    assert!(
        report.converged,
        "{:?}; permutation {}",
        report.warnings, report.value.permutation_defect
    );
    assert!(report.value.permutation_defect < 1e-10);
    assert_eq!(report.value.permutation_comparisons, 36);
    // Centered polarization has eight nonzero corners for each of six
    // distinct-axis pairs and six for each of three repeated-axis pairs.
    // The independent nuclear leg is +/-1/2, so no corner vanishes.
    assert_eq!(report.value.directional_evaluations, 6 * 8 + 3 * 6);
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let v = report.value.tensor[9 * a + 3 * b + c];
                assert!((v - report.value.tensor[9 * c + 3 * a + b]).abs() < 1e-10);
            }
        }
    }
    assert!(
        report
            .value
            .tensor
            .iter()
            .filter(|x| x.abs() > 1e-3)
            .count()
            > 15
    );
}
