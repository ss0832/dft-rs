// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Compile-and-run contracts for the documented public Rust surface.

use dft_rs::basis::{data, MolBasis};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::nlc::Vv10Params;
use dft_rs::periodic::integrals::ImageBlock;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::pseudo_sub::fit::{run_synthetic_recovery_gate, FitParameterRecord};
use dft_rs::{resolve, run_scf, Atom, FunctionalSpec, Molecule, ScfOptions};

#[test]
fn unavailable_common_routes_use_typed_unsupported_errors() {
    use dft_rs::driver::{run, Controls, System, Value};
    use dft_rs::DftError;
    let system = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    let controls =
        Controls::from_value(&Value::object().with("basis", "missing-capability-test-basis"))
            .unwrap();
    for feature in ["unavailable-calculation", "dynamic-beta", "nuclear-taylor"] {
        let error = run(feature, &system, &controls).unwrap_err();
        assert!(
            matches!(error, DftError::Unsupported { .. }),
            "{feature}: {error}"
        );
        assert!(error.to_string().contains("not implemented"));
    }
    assert!(matches!(
        run("", &system, &Controls::new()),
        Err(DftError::InvalidInput(_))
    ));
    let known = Controls::from_value(&Value::object().with("xc", "hf")).unwrap();
    assert!(matches!(
        run("feature-registry", &system, &known),
        Err(DftError::Unsupported { .. })
    ));
    let typo = Controls::from_value(&Value::object().with("grid_levl", 1)).unwrap();
    assert!(matches!(
        run("feature-registry", &system, &typo),
        Err(DftError::InvalidInput(_))
    ));
}

#[test]
fn unconnected_references_are_refused_before_molecular_preparation() {
    use dft_rs::driver::{run, Controls, System, Value};
    use dft_rs::DftError;
    let system = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    for feature in ["energy", "charges", "spin-susceptibility"] {
        for value in [Value::Str("rkos".into()), Value::Int(1), Value::Bool(true)] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("reference", value)
                    .with("basis", "missing-capability-test-basis"),
            )
            .unwrap();
            assert!(matches!(
                run(feature, &system, &controls),
                Err(DftError::InvalidInput(_))
            ));
        }
    }
    for feature in ["spin-susceptibility", "nmr-shielding", "ecd-spectrum"] {
        for reference in ["roks", "rohf", "gks", "ghf"] {
            let mut input = Value::object()
                .with("reference", reference)
                .with("basis", "missing-capability-test-basis");
            if feature == "ecd-spectrum" {
                input.set("broadening_ev", 0.1);
                input.set("td_method", "tda");
            }
            let controls = Controls::from_value(&input).unwrap();
            let error = run(feature, &system, &controls).unwrap_err();
            assert!(
                matches!(error, DftError::Unsupported { .. }),
                "{feature}/{reference}: {error}"
            );
            assert!(controls.finish("rejected request").is_err());
        }
    }
    // Real dedicated routes must still reach basis setup, rather than being
    // rejected by the new unused-selector guard. No SCF is started here.
    for feature in ["energy", "charges", "orbitals", "roks"] {
        let controls = Controls::from_value(
            &Value::object()
                .with("reference", "roks")
                .with("basis", "missing-capability-test-basis"),
        )
        .unwrap();
        assert!(matches!(
            run(feature, &system, &controls),
            Err(DftError::UnknownBasis(_))
        ));
    }
    let controls = Controls::from_value(
        &Value::object()
            .with("reference", "gks")
            .with("basis", "missing-capability-test-basis"),
    )
    .unwrap();
    assert!(matches!(
        run("energy", &system, &controls),
        Err(DftError::UnknownBasis(_))
    ));
}

#[test]
fn unavailable_orders_are_distinct_from_malformed_inputs_before_setup() {
    use dft_rs::driver::{run, Controls, System, Value};
    use dft_rs::DftError;
    let molecule = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    for (key, value) in [("quantum_lmax", 4), ("species_derivative_order", 3)] {
        let controls = Controls::from_value(
            &Value::object()
                .with("basis", "missing-capability-test-basis")
                .with(key, value),
        )
        .unwrap();
        assert!(matches!(
            run("multicomponent", &molecule, &controls),
            Err(DftError::Unsupported { .. })
        ));
        let negative = Controls::from_value(&Value::object().with(key, -1)).unwrap();
        assert!(matches!(
            run("multicomponent", &molecule, &negative),
            Err(DftError::InvalidInput(_))
        ));
    }
    let periodic = System {
        cell: Some([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
        pbc: [true; 3],
        ..molecule
    };
    for (order, shear) in [(2, true), (3, false)] {
        let controls = Controls::from_value(
            &Value::object()
                .with("basis", "missing-capability-test-basis")
                .with("gruneisen_order", order)
                .with("gruneisen_shear", shear),
        )
        .unwrap();
        assert!(matches!(
            run("mode-gruneisen", &periodic, &controls),
            Err(DftError::Unsupported { .. })
        ));
    }
    let zero = Controls::from_value(&Value::object().with("gruneisen_order", 0)).unwrap();
    assert!(matches!(
        run("mode-gruneisen", &periodic, &zero),
        Err(DftError::InvalidInput(_))
    ));
    for failure in [
        DftError::InvalidInput("invalid geometry".into()),
        DftError::ScfNotConverged {
            iterations: 1,
            error: 1.0,
        },
        DftError::ResponseNotConverged {
            iterations: 1,
            residual: 1.0,
        },
        DftError::ResourceLimit {
            operation: "workspace",
            required_mb: 2,
            limit_mb: 1,
        },
    ] {
        assert!(!failure.is_unsupported());
    }
}

#[test]
fn missing_assembly_connections_are_unsupported_without_substitution() {
    use dft_rs::driver::{run, Controls, System, Value};
    use dft_rs::DftError;
    let system = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    for (feature, xc, assembly) in [
        ("gamma", "hf", "variational"),
        ("beta", "hf", "energy"),
        ("beta", "b2plyp", "variational"),
    ] {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", xc)
                .with("tensor_assembly", assembly)
                .with("basis", "missing-capability-test-basis"),
        )
        .unwrap();
        let error = run(feature, &system, &controls).unwrap_err();
        assert!(
            matches!(error, DftError::Unsupported { .. }),
            "{feature}/{xc}/{assembly}: {error}"
        );
        assert!(error.to_string().contains(assembly));
    }
    for (xc, assembly) in [("hf", "variational"), ("b2plyp", "energy")] {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", xc)
                .with("tensor_assembly", assembly)
                .with("basis", "missing-capability-test-basis"),
        )
        .unwrap();
        assert!(matches!(
            run("beta", &system, &controls),
            Err(DftError::UnknownBasis(_))
        ));
    }
    let typo =
        Controls::from_value(&Value::object().with("tensor_assembly", "unavailable-assembly"))
            .unwrap();
    assert!(matches!(
        run("beta", &system, &typo),
        Err(DftError::InvalidInput(_))
    ));
}

#[test]
fn reusing_controls_cannot_hide_an_unsupported_reference() {
    use dft_rs::driver::{run, Controls, System, Value};
    use dft_rs::DftError;
    let system = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    for threads in [None, Some(1)] {
        let mut options = Value::object()
            .with("reference", "roks")
            .with("basis", "missing-capability-test-basis");
        if let Some(threads) = threads {
            options.set("cpu_threads", threads);
        }
        let controls = Controls::from_value(&options).unwrap();
        // A caller or earlier reference-specific request may have read it.
        controls.str("reference", "auto").unwrap();
        let error = run("spin-susceptibility", &system, &controls).unwrap_err();
        assert!(matches!(error, DftError::Unsupported { .. }), "{error}");
        assert_eq!(controls.to_value(), options);
        let mut reused = Value::object().with("xc", "hf");
        if let Some(threads) = threads {
            reused.set("cpu_threads", threads);
        }
        let reused = Controls::from_value(&reused).unwrap();
        reused.str("xc", "hf").unwrap();
        assert!(matches!(
            run("feature-registry", &system, &reused),
            Err(DftError::Unsupported { .. })
        ));
    }
    let controls = Controls::from_value(
        &Value::object()
            .with("reference", "roks")
            .with("basis", "missing-capability-test-basis"),
    )
    .unwrap();
    assert!(matches!(
        run("charges", &system, &controls),
        Err(DftError::UnknownBasis(_))
    ));
    assert!(matches!(
        run("spin-susceptibility", &system, &controls),
        Err(DftError::Unsupported { .. })
    ));
    let nullable = Controls::from_value(
        &Value::object()
            .with("reference", Value::Null)
            .with("basis", "missing-capability-test-basis"),
    )
    .unwrap();
    assert!(matches!(
        run("energy", &system, &nullable),
        Err(DftError::UnknownBasis(_))
    ));
    nullable.finish("default reference accepted").unwrap();
    let unknown = Controls::from_value(&Value::object().with("caller_only_control", true)).unwrap();
    let _ = unknown.take("caller_only_control");
    assert!(matches!(
        run("feature-registry", &system, &unknown),
        Err(DftError::InvalidInput(_))
    ));
}

#[test]
fn pseudo_substituent_candidate_fitter_is_public_but_does_not_enable_a_name() {
    let fit = run_synthetic_recovery_gate().unwrap();
    assert!(fit.optimizer.converged);
    let text = fit.record.to_text().unwrap();
    let parsed = FitParameterRecord::from_text(&text).unwrap();
    assert_eq!(parsed, fit.record);
    assert!(text.contains("status=candidate-unaccepted"));
    assert!(dft_rs::pseudo_sub::NamedSubstituent::Me.build(1).is_err());
}

#[test]
fn documented_molecular_api_produces_charge_conserving_results() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let set = data::load_or_read("sto-3g").unwrap();
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let result = run_scf(&molecule, &basis, &ScfOptions::default(), &functional, None).unwrap();
    let mulliken = dft_rs::properties::mulliken_charges(&molecule, &basis, &result);
    let lowdin = dft_rs::properties::lowdin_charges(&molecule, &basis, &result).unwrap();
    assert!(mulliken.iter().sum::<f64>().abs() < 1e-10);
    assert!(lowdin.iter().sum::<f64>().abs() < 1e-10);
    assert_eq!(result.n_alpha + result.n_beta, 2);
}

#[test]
fn builtin_nonlocal_mapping_and_periodic_validation_are_public_contracts() {
    assert_eq!(
        Vv10Params::for_functional("wB97X-V"),
        Some(Vv10Params::WB97XV)
    );
    assert_eq!(
        Vv10Params::for_functional("wb97m_v"),
        Some(Vv10Params::WB97XV)
    );
    assert_eq!(
        Vv10Params::for_functional("b97m-v"),
        Some(Vv10Params::WB97XV)
    );
    assert_eq!(Vv10Params::for_functional("b3lyp"), None);

    let helium = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let set = data::load("sto-3g").unwrap();
    let periodic_cell = Cell::new(helium, Lattice::cubic(10.0), &set, 8.0).unwrap();
    for name in ["wb97x-v", "wb97m-v", "b97m-v", "wb97x-d"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        assert!(functional.required_nonlocal_term().is_some());
        let error = run_periodic_scf(&periodic_cell, &functional, &PeriodicScfOptions::default())
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("nonlocal term is not yet periodic"));
    }
    let r2scan = resolve(&FunctionalSpec::Combined("r2scan".into())).unwrap();
    assert_eq!(r2scan.required_nonlocal_term(), None);

    let molecule = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let set = data::load("sto-3g").unwrap();
    assert!(Cell::new(molecule.clone(), Lattice::cubic(0.0), &set, 20.0).is_err());
    assert!(Cell::new(molecule, Lattice::cubic(10.0), &set, 0.0).is_err());
}

#[test]
fn periodic_cell_rebuild_keeps_caches_physical_and_inputs_immutable() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(-0.61, 0.14, -0.08),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.79, -0.09, 0.17),
        },
    ]);
    let set = data::load("sto-3g").unwrap();
    let cell = Cell::new(molecule.clone(), Lattice::cubic(12.0), &set, 8.0).unwrap();
    assert_eq!(cell.molecule().len(), 2);
    assert_eq!(cell.basis().nao_sph, 2);
    assert!((cell.lattice().volume() - 12.0_f64.powi(3)).abs() < 1e-12);
    assert_eq!(cell.overlap_cutoff(), 8.0);

    // A continuous translation of the complete basis+nuclear system is a
    // gauge change. Rebuilding must reproduce all cached Bloch operators
    // and the Ewald nuclear energy, while leaving the original cell intact.
    let shift = Vec3::new(1.17, -0.46, 0.73);
    let mut translated_molecule = molecule.clone();
    for atom in &mut translated_molecule.atoms {
        atom.position += shift;
    }
    let translated = cell
        .rebuilt_with_molecule(translated_molecule, &set)
        .unwrap();
    let k = Vec3::new(0.031, -0.047, 0.019);
    let (s0_re, s0_im) = cell.overlap(k);
    let (s1_re, s1_im) = translated.overlap(k);
    let (h0_re, h0_im) = cell.core_hamiltonian(k);
    let (h1_re, h1_im) = translated.core_hamiltonian(k);
    assert!(s0_re.max_abs_diff(&s1_re) < 2e-12);
    assert!(s0_im.max_abs_diff(&s1_im) < 2e-12);
    assert!(h0_re.max_abs_diff(&h1_re) < 2e-10);
    assert!(h0_im.max_abs_diff(&h1_im) < 2e-10);
    assert!((cell.nuclear_energy() - translated.nuclear_energy()).abs() < 2e-11);
    assert_eq!(
        cell.molecule().atoms[0].position,
        molecule.atoms[0].position
    );

    // A relative displacement is a different physical system and therefore
    // must rebuild, rather than reuse, the old overlap cache.
    let mut distorted_molecule = molecule;
    distorted_molecule.atoms[1].position.x += 0.23;
    let distorted = cell
        .rebuilt_with_molecule(distorted_molecule, &set)
        .unwrap();
    assert!(cell.overlap(k).0.max_abs_diff(&distorted.overlap(k).0) > 1e-5);
}

#[test]
fn periodic_public_api_validates_density_before_building_hartree() {
    let molecule = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let set = data::load("sto-3g").unwrap();
    let cell = Cell::new(molecule, Lattice::cubic(10.0), &set, 12.0).unwrap();
    let density = vec![ImageBlock {
        translation: Vec3::zero(),
        matrix: Matrix::from_row_major(1, 1, vec![1.0]),
    }];
    assert!((cell.electron_count(&density).unwrap() - 1.0).abs() < 1e-12);

    // A Gamma-point density is constant in real-space lattice translation,
    // not a lone D(0) block. Its electron count must therefore equal the
    // direct Bloch-space contraction Tr[P(Gamma) S(Gamma)].
    let p_gamma = Matrix::from_row_major(1, 1, vec![0.75]);
    let gamma_images = cell.gamma_density_images(&p_gamma).unwrap();
    assert_eq!(
        gamma_images.len(),
        cell.lattice()
            .translations_within(cell.overlap_cutoff())
            .len()
    );
    let (s_gamma, s_imag) = cell.overlap(Vec3::zero());
    assert!(s_imag.as_slice().iter().all(|x| x.abs() < 1e-14));
    let real_space_count = cell.electron_count(&gamma_images).unwrap();
    let bloch_count = p_gamma.frobenius_dot(&s_gamma);
    assert!((real_space_count - bloch_count).abs() < 1e-12);

    // A 3x3x3 mesh exactly resolves a home block plus the six nearest
    // neighbours. This pins both the sign and normalization of the inverse
    // Bloch transform used by the k-point SCF path.
    let k_points = cell.k_points([3, 3, 3]);
    let density_k: Vec<(Matrix, Matrix)> = k_points
        .iter()
        .map(|k| {
            let phase_sum = cell
                .lattice()
                .a
                .iter()
                .map(|a| k.dot(*a).cos())
                .sum::<f64>();
            (
                Matrix::from_row_major(1, 1, vec![0.7 + 0.1 * phase_sum]),
                Matrix::zeros(1, 1),
            )
        })
        .collect();
    let transformed = cell
        .density_images_from_k(&k_points, &density_k, 12.0)
        .unwrap();
    for block in &transformed {
        let nearest = cell.lattice().a.iter().any(|a| {
            (block.translation - *a).norm() < 1e-10 || (block.translation + *a).norm() < 1e-10
        });
        let want = if block.translation.norm() < 1e-10 {
            0.7
        } else if nearest {
            0.05
        } else {
            0.0
        };
        assert!((block.matrix[(0, 0)] - want).abs() < 1e-12);
    }

    let orphan = vec![ImageBlock {
        translation: Vec3::new(10.0, 0.0, 0.0),
        matrix: Matrix::from_row_major(1, 1, vec![0.1]),
    }];
    assert!(cell.electron_count(&orphan).is_err());
    let wrong_shape = vec![ImageBlock {
        translation: Vec3::zero(),
        matrix: Matrix::zeros(2, 2),
    }];
    assert!(cell.electron_count(&wrong_shape).is_err());
}
