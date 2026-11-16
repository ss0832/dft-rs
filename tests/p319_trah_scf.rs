// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-driver parity for the complete parent SCF and post-SCF PT2 energy.
use dft_rs::driver::{run, Controls, System, Value};

#[test]
fn trah_retains_the_stationary_non_aufbau_uks_determinant() {
    use dft_rs::constants::ANGSTROM_TO_BOHR;
    use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
    use dft_rs::linalg::Orthogonalizer;
    use dft_rs::scf::run_scf;

    // This ion has a tiny negative beta occupied/virtual gap on its stationary
    // integer branch. Reoccupying its physical Fock exchanges one beta orbital
    // and raises the energy; it is not a remaining orbital-stationarity error.
    let positions = [
        [0., 0., 0.],
        [0., 0., 0.957],
        [0.927, 0., -0.240],
        [0., 0., 2.980],
        [0., 0.757, 3.549],
        [0., -0.757, 3.549],
    ]
    .map(|r| r.map(|x| x * ANGSTROM_TO_BOHR));
    let mut energies = Vec::new();
    for mode in ["auto", "trah"] {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "pbe")
                .with("basis", "sto-3g")
                .with("reference", "uks")
                .with("charge", 1)
                .with("multiplicity", 2usize)
                .with("dispersion", "none")
                .with("grid_level", 1usize)
                .with("grid_weight_cutoff", 0.)
                .with("max_iter", 120usize)
                .with("conv_tol", 1e-12)
                .with("conv_tol_dp", 1e-10)
                .with("conv_tol_grad", 1e-10)
                .with("scf_optimizer", mode),
        )
        .unwrap();
        let method = MolecularMethod::from_controls(&controls).unwrap();
        let setup = prepare_molecular(&[8, 1, 1, 8, 1, 1], &positions, &method).unwrap();
        let scf = run_scf(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap();
        assert!(
            scf.converged,
            "{mode}: incomplete SCF after {} iterations",
            scf.iterations
        );
        let diagnostics = dft_rs::response::diagnostics::reference_diagnostics(&scf).unwrap();
        assert!(diagnostics.orbital_residual < 1e-10);
        assert!(diagnostics.eigen_residual < 1e-10);
        assert!(diagnostics.projector_defect < 1e-10);
        assert!(diagnostics.orthonormality_defect < 1e-10);
        assert!(diagnostics.particle_number_error < 1e-10);
        let gaps = scf.occupied_virtual_gaps();
        assert!(gaps[0].unwrap() > 0.);
        assert!(gaps[1].unwrap() < -1e-7);
        let orth = Orthogonalizer::canonical(&scf.overlap, setup.options.orth_threshold).unwrap();
        let sx = scf.overlap.matmul_seq(&orth.x);
        for spin in 0..2 {
            let (_, c) = orth.solve_roothaan(&scf.fock[spin]).unwrap();
            let mut delta = c.leading_columns_gram([scf.n_alpha, scf.n_beta][spin], 1.);
            for (d, p) in delta
                .as_mut_slice()
                .iter_mut()
                .zip(scf.density[spin].as_slice())
            {
                *d -= p;
            }
            let metric = sx.transpose_matmul_seq(&delta).matmul_seq(&sx);
            let norm = metric.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
            let rms = (delta.as_slice().iter().map(|x| x * x).sum::<f64>()
                / delta.as_slice().len() as f64)
                .sqrt();
            println!(
                "TRAH handoff mode={mode} spin={spin} physical_map={norm:.12e} AO_rms={rms:.12e}"
            );
            if spin == 0 {
                assert!(norm < 1e-10 && rms < 1e-10);
            } else {
                assert!((norm - 2.0_f64.sqrt()).abs() < 1e-10);
                assert!(rms > 0.09);
            }
        }
        println!(
            "TRAH handoff mode={mode} energy={:.16e} iterations={} orbital={:.12e}",
            scf.energy, scf.iterations, diagnostics.orbital_residual
        );
        energies.push(scf.energy);
        // A separate complete-kernel stability calculation characterizes the
        // stationary determinant; SCF convergence itself is not that claim.
        let stability = dft_rs::response::stability::stability_from_state(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &scf,
            &setup.options,
            &dft_rs::response::stability::StabilityOptions {
                n_roots: 1,
                tolerance: 1e-9,
                max_subspace: 100,
                verify_steps: 100,
                ..Default::default()
            },
        )
        .unwrap();
        println!("TRAH occupation gaps={gaps:?} stability={stability:?}");
    }
    assert!((energies[0] - energies[1]).abs() < 1e-9);
}

#[test]
fn trah_reference_and_functional_routes_retain_the_physical_energy() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.57, 0.18, 0.33],
            [0.31, 4.74, -0.45],
            [1.60, 5.01, -0.61],
        ],
    );
    for xc in ["hf", "svwn", "pbe", "tpss", "pbe0", "b2plyp"] {
        for reference in ["rks", "uks", "roks", "gks"] {
            let parameters = Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("dispersion", "none")
                .with("grid_level", 1usize)
                .with("grid_weight_cutoff", 0.0)
                .with("reference", reference)
                .with("max_iter", 300usize)
                .with("conv_tol", 1e-10)
                .with("conv_tol_dp", 1e-10)
                .with("conv_tol_grad", 1e-9);
            let energy = |mode| {
                let c =
                    Controls::from_value(&parameters.clone().with("scf_optimizer", mode)).unwrap();
                run("energy", &system, &c).unwrap()
            };
            let baseline = energy("legacy");
            let candidate = energy("trah");
            let base = baseline
                .get("energy")
                .and_then(Value::as_f64)
                .expect("energy");
            let value = candidate
                .get("energy")
                .and_then(Value::as_f64)
                .expect("energy");
            println!("TRAH public reference={reference} xc={xc} energy={value:.15e} baseline={base:.15e} defect={:.12e} report={candidate:?}",(base-value).abs());
            assert!((base - value).abs() < 1e-8);
        }
    }
}

#[test]
fn trah_can_leave_an_unrestricted_saddle_without_changing_the_energy_functional() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    );
    let base = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("reference", "uks")
        .with("dispersion", "none")
        .with("conv_tol", 1e-11)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-10);
    let reports = ["legacy", "trah"].map(|mode| {
        run(
            "energy",
            &system,
            &Controls::from_value(&base.clone().with("scf_optimizer", mode)).unwrap(),
        )
        .unwrap()
    });
    let energies = reports
        .each_ref()
        .map(|r| r.get("energy").unwrap().as_f64().unwrap());
    println!(
        "TRAH saddle baseline={:.15e} minimum={:.15e} lowering={:.12e}",
        energies[0],
        energies[1],
        energies[0] - energies[1]
    );
    assert!(energies[1] < energies[0] - 0.09);
    assert!((energies[1] + 1.881103483281227).abs() < 1e-9);
    assert!(reports[1].get("s_squared").unwrap().as_f64().unwrap() > 0.9);
}

#[test]
fn trah_open_common_orbitals_retain_all_functional_routes() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.57, 0.18, 0.33],
            [0.31, 4.74, -0.45],
            [1.60, 5.01, -0.61],
        ],
    );
    for xc in ["hf", "svwn", "pbe", "tpss", "pbe0", "b2plyp"] {
        let base = Value::object()
            .with("xc", xc)
            .with("basis", "sto-3g")
            .with("reference", "roks")
            .with("charge", 1)
            .with("multiplicity", 2usize)
            .with("dispersion", "none")
            .with("grid_level", 1usize)
            .with("grid_weight_cutoff", 0.0)
            .with("max_iter", 500usize)
            .with("conv_tol", 1e-10)
            .with("conv_tol_dp", 1e-10)
            .with("conv_tol_grad", 1e-9);
        let reports = ["legacy", "trah"].map(|mode| {
            run(
                "energy",
                &system,
                &Controls::from_value(&base.clone().with("scf_optimizer", mode)).unwrap(),
            )
            .unwrap()
        });
        let energies = reports
            .each_ref()
            .map(|r| r.get("energy").unwrap().as_f64().unwrap());
        println!(
            "TRAH open common xc={xc} baseline={:.15e} energy={:.15e} defect={:.12e} report={:?}",
            energies[0],
            energies[1],
            (energies[1] - energies[0]).abs(),
            reports[1]
        );
        assert!((energies[1] - energies[0]).abs() < 1e-8);
    }
}

#[test]
fn trah_pinned_complex_spinor_retains_the_full_spin_solution() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.57, 0.18, 0.33],
            [0.31, 4.74, -0.45],
            [1.60, 5.01, -0.61],
        ],
    );
    let mut real = vec![vec![0.0; 8]; 8];
    let mut imaginary = real.clone();
    for i in 0..4 {
        real[i][i] = -0.013 * (i + 1) as f64;
        real[i + 4][i + 4] = -real[i][i];
        real[i][i + 4] = 0.007 * (i + 1) as f64;
        real[i + 4][i] = real[i][i + 4];
        imaginary[i][i + 4] = 0.009 * (4 - i) as f64;
        imaginary[i + 4][i] = -imaginary[i][i + 4];
    }
    let base = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("reference", "gks")
        .with("dispersion", "none")
        .with("gks_operator_real", real)
        .with("gks_operator_imaginary", imaginary)
        .with("conv_tol", 1e-10)
        .with("conv_tol_dp", 1e-10)
        .with("conv_tol_grad", 1e-10)
        .with("max_iter", 500usize);
    let reports = ["legacy", "trah"].map(|mode| {
        run(
            "energy",
            &system,
            &Controls::from_value(&base.clone().with("scf_optimizer", mode)).unwrap(),
        )
        .unwrap()
    });
    let energy = reports
        .each_ref()
        .map(|r| r.get("energy").unwrap().as_f64().unwrap());
    assert!((energy[1] - energy[0]).abs() < 1e-9);
    for report in &reports {
        assert!(report.get("orbital_residual").unwrap().as_f64().unwrap() < 1e-10);
    }
    println!(
        "TRAH pinned spinor baseline={:.15e} energy={:.15e} defect={:.12e} report={:?}",
        energy[0],
        energy[1],
        (energy[1] - energy[0]).abs(),
        reports[1]
    );
}

#[test]
fn trah_double_hybrid_common_parent_uses_the_existing_functional_composition() {
    use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
    use dft_rs::scf::{roks::run_roks, OrbitalOptimizer};
    let positions = vec![
        [0.17, -0.11, 0.23],
        [1.57, 0.18, 0.33],
        [0.31, 4.74, -0.45],
        [1.60, 5.01, -0.61],
    ];
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "tpss")
            .with("basis", "sto-3g")
            .with("charge", 1)
            .with("multiplicity", 2usize)
            .with("grid_level", 1usize)
            .with("grid_weight_cutoff", 0.0)
            .with("max_iter", 500usize)
            .with("conv_tol", 1e-10)
            .with("conv_tol_dp", 1e-10)
            .with("conv_tol_grad", 1e-9)
            .with("dispersion", "none"),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let setup = prepare_molecular(&[1; 4], &positions, &method).unwrap();
    let functional = dft_rs::xc::double_hybrid::DoubleHybridFunctional::named("b2plyp").unwrap();
    let mut energy = Vec::new();
    for mode in [OrbitalOptimizer::Legacy, OrbitalOptimizer::Trah] {
        let mut options = setup.options.clone();
        options.orbital_optimizer = mode;
        let result = run_roks(
            &setup.molecule,
            &setup.basis,
            &options,
            &functional.parent,
            setup.grid.as_ref(),
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.common_orbital_residual.unwrap() < 1e-9);
        println!("TRAH B2PLYP common parent optimizer={mode:?} energy={:.15e} residual={:.12e} iterations={}",
            result.energy,result.common_orbital_residual.unwrap(),result.iterations);
        energy.push(result.energy);
    }
    assert!((energy[1] - energy[0]).abs() < 1e-8);
}
