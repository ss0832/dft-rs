// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full-basis frozen-reference extraction and variational density gates.
//! The canonical public import tests the actual compiled reference module
//! without loading a second source copy or registering a generated model.
pub use dft_rs::{
    basis, grid, integrals, ks, linalg, logger, math, properties, response, scf, system, xc,
    DftError, Result,
};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
pub use dft_rs::pseudo_sub::reduction_reference;

use basis::{data, MolBasis};
use grid::{GridOptions, MolecularGrid};
use linalg::{symmetric_eigen, Matrix, Orthogonalizer};
use math::Vec3;
use reduction_reference::{
    extract_methyl_frozen_space, ExtractionOptions, FrozenDensityEvaluator, FrozenReferenceModel,
    MethylBoundary,
};
use scf::{run_scf, EriMode, Guess, ScfOptions};
use system::{Atom, Molecule};
use xc::{resolve, FunctionalSpec};

fn options() -> ScfOptions {
    ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    }
}

fn molecular_grid(molecule: &Molecule) -> MolecularGrid {
    MolecularGrid::build(
        molecule,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap()
}

fn ethane() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::new(-1.46, -0.14, 0.23),
        },
        Atom {
            z: 1,
            position: Vec3::new(-2.13, 1.69, 0.68),
        },
        Atom {
            z: 1,
            position: Vec3::new(-2.28, -1.57, 1.31),
        },
        Atom {
            z: 1,
            position: Vec3::new(-2.19, -0.43, -1.65),
        },
        Atom {
            z: 6,
            position: Vec3::new(1.35, 0.17, -0.12),
        },
        Atom {
            z: 1,
            position: Vec3::new(2.05, 2.08, 0.16),
        },
        Atom {
            z: 1,
            position: Vec3::new(2.10, -0.92, 1.45),
        },
        Atom {
            z: 1,
            position: Vec3::new(2.01, -0.84, -1.67),
        },
    ])
}

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.13, -0.21, 0.17),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.34, 0.82, -0.23),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.74, 1.13, 0.91),
        },
    ])
}

fn select_columns(matrix: &Matrix, range: std::ops::Range<usize>) -> Matrix {
    let mut output = Matrix::zeros(matrix.rows, range.len());
    for (column, source) in range.enumerate() {
        for row in 0..matrix.rows {
            output[(row, column)] = matrix[(row, source)];
        }
    }
    output
}

fn closed_density(orbitals: &Matrix, occupied: usize) -> Matrix {
    let occupied = select_columns(orbitals, 0..occupied);
    let mut result = occupied.matmul_transpose_seq(&occupied);
    for value in result.as_mut_slice() {
        *value *= 2.0;
    }
    result
}

fn frozen_recovery(name: &str) {
    let molecule = ethane();
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = molecular_grid(&molecule);
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let opts = options();
    let full = run_scf(&molecule, &basis, &opts, &functional, Some(&grid)).unwrap();
    let selection = extract_methyl_frozen_space(
        &molecule,
        &basis,
        &full,
        MethylBoundary {
            carbon: 4,
            hydrogens: [5, 6, 7],
            port_atom: 0,
        },
        &ExtractionOptions::default(),
    )
    .unwrap();
    println!("{name} frozen extraction {:?}", selection.diagnostics);
    assert!(!selection.diagnostics.ambiguous);
    assert_eq!(
        (
            selection.full_electrons,
            selection.frozen_electrons,
            selection.active_electrons,
            selection.ionic_charge
        ),
        (18, 8, 10, 1)
    );
    assert_eq!(selection.frozen_orbitals.cols, 4);
    assert_eq!(selection.active_reference_orbitals.cols, 5);
    assert_eq!(selection.diagnostics.selected_parent_labels.len(), 4);
    assert!(selection.diagnostics.excluded_port_label.contains("C"));
    assert!(selection.diagnostics.occupied_reconstruction_error < 1e-9);
    assert!((selection.diagnostics.frozen_electron_count - 8.0).abs() < 1e-9);
    assert!((selection.diagnostics.active_electron_count - 10.0).abs() < 1e-9);
    let model = FrozenReferenceModel::new(&molecule, &basis, &grid, &functional, &opts, &selection)
        .unwrap();
    let initial_evaluation = model
        .evaluator()
        .evaluate(&selection.active_reference_density)
        .unwrap();
    assert!((initial_evaluation.energy - full.energy).abs() < 1e-10);
    assert!(initial_evaluation.fock.max_abs_diff(&full.fock[0]) < 1e-9);
    let independent = model.solve(None).unwrap();
    println!(
        "{name} full={:.15} frozen={:.15} dE={:.9e} residual={:.9e} iterations={}",
        full.energy,
        independent.evaluation.energy,
        (independent.evaluation.energy - full.energy).abs(),
        independent.projected_orbital_residual,
        independent.history.len()
    );
    assert!((independent.evaluation.energy - full.energy).abs() < 1e-10);
    assert!(
        independent
            .active_density
            .max_abs_diff(&selection.active_reference_density)
            < 1e-8
    );
    assert!(independent.projected_orbital_residual < opts.conv_tol_grad);
    assert!(independent.particle_number_error < 1e-9);
    assert!(independent.active_idempotency_error < 1e-9);
    assert!(independent.active_frozen_overlap < 1e-9);
    let y = model.complement();
    let core = y.transpose_matmul_seq(&model.evaluator().h_core().matmul_seq(y));
    let (_, mut orbitals) = symmetric_eigen(&core).unwrap();
    let (sine, cosine) = 0.17_f64.sin_cos();
    for row in 0..orbitals.rows {
        let a = orbitals[(row, 0)];
        let b = orbitals[(row, 5)];
        orbitals[(row, 0)] = cosine * a + sine * b;
        orbitals[(row, 5)] = -sine * a + cosine * b;
    }
    let seed = y
        .matmul_seq(&closed_density(&orbitals, 5))
        .matmul_transpose_seq(y);
    let second = model.solve(Some(&seed)).unwrap();
    assert!(
        second
            .active_density
            .max_abs_diff(&independent.active_density)
            < 1e-8
    );
    assert!((second.evaluation.energy - independent.evaluation.energy).abs() < 1e-10);
    let target = model.static_residual_target(1.0).unwrap();
    assert!(target.reconstruction_error < 1e-12);
    assert!(target.frozen_pauli_energy.abs() < 1e-9);
    let mut reconstructed = target.removed_nuclear.clone();
    for other in [
        &target.frozen_hartree,
        &target.nonadditive_xc,
        &target.reference_pauli,
    ] {
        for (a, b) in reconstructed
            .as_mut_slice()
            .iter_mut()
            .zip(other.as_slice())
        {
            *a += b;
        }
    }
    assert!(reconstructed.max_abs_diff(&target.total) < 1e-12);
    assert!(
        target.core_reference_energy.is_finite() && target.internal_reference_energy.is_finite()
    );
    let mut pauli_fock = full.fock[0].clone();
    for (a, b) in pauli_fock
        .as_mut_slice()
        .iter_mut()
        .zip(target.reference_pauli.as_slice())
    {
        *a += b;
    }
    let coupling = selection
        .frozen_orbitals
        .transpose_matmul_seq(&pauli_fock.matmul_seq(&selection.active_reference_orbitals));
    assert!(coupling.as_slice().iter().all(|v| v.abs() < 1e-9));
    let projected_b = selection
        .frozen_orbitals
        .transpose_matmul_seq(&pauli_fock.matmul_seq(&selection.frozen_orbitals));
    assert!(projected_b.max_abs_diff(&Matrix::identity(4)) < 1e-9);
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let shift = Vec3::new(0.31, -0.23, 0.17);
    let mu = model.frozen_core_dipole(origin).unwrap();
    let shifted = model.frozen_core_dipole(origin + shift).unwrap();
    assert!((shifted - mu + shift).norm() < 1e-9); // Q_ion=+1.
    let probes = [Vec3::new(5.3, -2.1, 1.7), Vec3::new(-3.2, 4.7, -1.9)];
    assert!(model
        .frozen_core_potential(&probes)
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(model
        .frozen_core_potential(&[molecule.atoms[4].position])
        .is_err());
    assert!(model.static_residual_target(-1.0).is_err());
    let mut wrong = ethane();
    wrong.atoms[4].position.x += 0.01;
    assert!(
        FrozenReferenceModel::new(&wrong, &basis, &grid, &functional, &opts, &selection).is_err()
    );
    let incomplete = extract_methyl_frozen_space(
        &molecule,
        &basis,
        &full,
        MethylBoundary {
            carbon: 4,
            hydrogens: [1, 6, 7],
            port_atom: 0,
        },
        &ExtractionOptions::default(),
    );
    assert!(incomplete.is_err());
}

#[test]
fn lda_same_full_ao_frozen_reference_recovers_ethane() {
    frozen_recovery("lda");
}
#[test]
fn pbe_same_full_ao_frozen_reference_recovers_ethane() {
    frozen_recovery("pbe");
}

fn density_fock_components(name: &str) {
    let molecule = water();
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = molecular_grid(&molecule);
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let overlap = integrals::overlap_matrix(&basis);
    let x = Orthogonalizer::canonical(&overlap, 1e-7).unwrap().x;
    let frozen_orbital = select_columns(&x, 0..1);
    let frozen = closed_density(&frozen_orbital, 1);
    let y = select_columns(&x, 1..x.cols);
    assert_eq!(y.cols, 6);
    // A smooth ensemble density in the active complement tests the OFF-SHELL
    // functional. Production reference SCF retains integer determinants.
    let mut d = Matrix::zeros(6, 6);
    for (i, value) in [1.15, 1.22, 1.29, 1.36, 1.43, 1.55].into_iter().enumerate() {
        d[(i, i)] = value;
    }
    for i in 0..6 {
        for j in i + 1..6 {
            let value = 0.02 * (i + 1) as f64 / (j + 1) as f64;
            d[(i, j)] = value;
            d[(j, i)] = value;
        }
    }
    let opts = options();
    let evaluator =
        FrozenDensityEvaluator::new(&molecule, &basis, &grid, &functional, &opts, &frozen).unwrap();
    let evaluate = |density: &Matrix| {
        evaluator
            .evaluate(&y.matmul_seq(density).matmul_transpose_seq(&y))
            .unwrap()
    };
    let reference = evaluate(&d);
    assert!((reference.active_electrons - 8.0).abs() < 1e-9);
    assert!((reference.frozen_electrons - 2.0).abs() < 1e-9);
    assert!(
        (reference.energy - reference.electronic_energy - reference.nuclear_energy).abs() < 1e-12
    );
    assert!(
        (reference.electronic_energy
            - reference.one_electron_energy
            - reference.hartree_energy
            - reference.xc_energy)
            .abs()
            < 1e-12
    );
    assert!(reference.grid_total_electrons > 0.0);
    let fock = y.transpose_matmul_seq(&reference.fock.matmul_seq(&y));
    let pairs: Vec<_> = (0..6).flat_map(|i| (i..6).map(move |j| (i, j))).collect();
    let names: Vec<_> = pairs
        .iter()
        .map(|&(i, j)| format!("{name} frozen scalar/Fock[{i},{j}]"))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.08, 0.04, 0.02, 0.01], [0.16, 0.08, 0.04, 0.02]],
        |steps| {
            let mut errors = vec![[0.0; 4]; pairs.len()];
            for (c, &(i, j)) in pairs.iter().enumerate() {
                for (k, &h) in steps.iter().enumerate() {
                    let mut positive = d.clone();
                    let mut negative = d.clone();
                    positive[(i, j)] += h;
                    negative[(i, j)] -= h;
                    if i != j {
                        positive[(j, i)] += h;
                        negative[(j, i)] -= h;
                    }
                    let finite =
                        (evaluate(&positive).energy - evaluate(&negative).energy) / (2.0 * h);
                    errors[c][k] = (finite
                        - if i == j {
                            fock[(i, j)]
                        } else {
                            2.0 * fock[(i, j)]
                        })
                    .abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "all independent fixed-frozen-density Fock derivatives",
        &failures,
    );
    let zero = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    let ordinary = run_scf(&molecule, &basis, &opts, &functional, Some(&grid)).unwrap();
    let ordinary_evaluator =
        FrozenDensityEvaluator::new(&molecule, &basis, &grid, &functional, &opts, &zero).unwrap();
    let normal = ordinary_evaluator.evaluate(&ordinary.p_total()).unwrap();
    assert!((normal.energy - ordinary.energy).abs() < 1e-10);
    assert!(normal.fock.max_abs_diff(&ordinary.fock[0]) < 1e-9);
    let invalid_frozen = Matrix::identity(basis.nao_sph);
    assert!(FrozenDensityEvaluator::new(
        &molecule,
        &basis,
        &grid,
        &functional,
        &opts,
        &invalid_frozen
    )
    .is_err());
}

#[test]
fn lda_every_independent_fock_entry_is_energy_derivative() {
    density_fock_components("lda");
}
#[test]
fn pbe_every_independent_fock_entry_is_energy_derivative() {
    density_fock_components("pbe");
}

#[test]
fn frozen_reference_nonconvergence_logging_contract() {
    let case = std::env::var("DFT_RS_REFERENCE_LOG_CASE").unwrap_or_else(|_| "file".into());
    assert!(matches!(case.as_str(), "file" | "stderr"));
    // Finish setup and close its resource scope before installing a new log.
    let (molecule, basis, grid, functional, forced, selection) =
        dft_rs::resources::ResourceLimits::new(Some(1), None)
            .unwrap()
            .run(|| {
                let molecule = ethane();
                let basis = MolBasis::build(&molecule, &data::load("sto-3g")?)?;
                let grid = molecular_grid(&molecule);
                let functional = resolve(&FunctionalSpec::Combined("pbe".into()))?;
                let opts = options();
                let full = run_scf(&molecule, &basis, &opts, &functional, Some(&grid))?;
                let selection = extract_methyl_frozen_space(
                    &molecule,
                    &basis,
                    &full,
                    MethylBoundary {
                        carbon: 4,
                        hydrogens: [5, 6, 7],
                        port_atom: 0,
                    },
                    &ExtractionOptions::default(),
                )?;
                assert!(!selection.diagnostics.ambiguous);
                // Existing dP/dE tests require a preceding iterate, so this
                // predeclared limit fails without changing any tolerance.
                let forced = ScfOptions {
                    max_iter: 1,
                    ..opts
                };
                Ok((molecule, basis, grid, functional, forced, selection))
            })
            .unwrap();
    let solve = || {
        let model =
            FrozenReferenceModel::new(&molecule, &basis, &grid, &functional, &forced, &selection)?;
        model.solve(None)
    };
    let (result, directory) = if case == "file" {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "dft-rs-reference-log-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let result = logger::LogOptions::to_file(directory.join("reference.log"))
            .unwrap()
            .run_with_limits(
                dft_rs::resources::ResourceLimits::new(Some(1), None).unwrap(),
                solve,
            );
        (result, Some(directory))
    } else {
        (
            dft_rs::resources::ResourceLimits::new(Some(1), None)
                .unwrap()
                .run(solve),
            None,
        )
    };
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("the predeclared one-iteration reference must fail"),
    };
    assert!(matches!(
        &error,
        DftError::ScfNotConverged { iterations: 1, .. }
    ));
    if let Some(directory) = directory {
        let path = directory.join("reference.log");
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches("[frozen-reference] iteration=").count(), 1);
        assert!(text.contains("[frozen-reference] iteration=1 E="));
        assert!(text.contains("residual=") && text.contains("dP=None dE=None"));
        assert!(text.contains(&error.to_string()));
        // Preserve the log on any preceding assertion failure.
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
    println!("reference_log_case={case} returned_error=ScfNotConverged iterations=1");
}
