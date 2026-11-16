// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use dft_rs::driver::{run, Controls, System, Value};

const WATER: [[f64; 3]; 3] = [
    [0.12, -0.17, 0.08],
    [1.68, 0.61, -0.29],
    [-0.66, 1.37, 0.67],
];
const VELOCITIES: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.0],
    [0.0015, -0.0003, 0.0007],
    [-0.0004, 0.0011, -0.0003],
];

fn number(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn water_extended_energy_and_timestep_convergence() {
    fit_water_timestep("wb97x-d");
}

#[test]
fn water_vv10_extended_energy_and_timestep_convergence() {
    fit_water_timestep("wb97x-v");
}

fn fit_water_timestep(functional: &str) {
    // Distorted water geometry and nonzero internal nuclear velocities.
    let system = System::molecule(vec![8, 1, 1], WATER.to_vec());
    let steps = [8, 12, 16, 24];
    let time = 24.0;
    let mut drifts = [0.0; 4];
    let dt = steps.map(|n| time / n as f64);
    for (index, (&steps, &dt)) in steps.iter().zip(&dt).enumerate() {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", functional)
                .with("grid_level", 0usize)
                .with("vv10_grid_level", 0usize)
                .with("reference", "rks")
                .with("md_steps", steps)
                .with("md_timestep", dt)
                .with("fictitious_mass", 200.0)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol_grad", 1e-9)
                .with("nuclear_velocities", VELOCITIES),
        )
        .unwrap();
        let report = run("car-parrinello-md", &system, &controls).unwrap();
        let frames = report.get("frames").unwrap().as_array().unwrap();
        let last = frames.last().unwrap();
        let orbital = frames
            .iter()
            .map(|f| number(f, "orbital_orthogonality_defect"))
            .fold(0.0, f64::max);
        let tangent = frames
            .iter()
            .map(|f| number(f, "orbital_tangency_defect"))
            .fold(0.0, f64::max);
        let kinetic = number(last, "electronic_kinetic_energy");
        drifts[index] = number(&report, "max_extended_energy_drift");
        println!("CPMD water {functional} dt={dt} steps={steps} drift={:.12e} T_e={kinetic:.12e} |Q^TQ-I|={orbital:.12e} |tangent|={tangent:.12e} initial orbital={:.12e} E_ext={:.12e}",
            drifts[index],number(&report,"scf_orbital_residual"),number(last,"extended_energy"));
        assert_eq!(frames.len(), steps + 1);
        assert!(kinetic > 1e-9);
        assert!(orbital < 1e-12 && tangent < 1e-12 && drifts[index] < 2e-4);
    }
    let fit = convergence::error_power_fit(&dt, &drifts);
    println!("CPMD {functional} energy drift fit: dt={dt:?} errors={drifts:?} p={:.9} R2={:.12} C={:.12e} log RMSE={:.12e}",
        fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse);
    assert!((fit.exponent - 2.0).abs() < 0.2 && fit.r_squared > 0.999);
}

#[test]
fn water_vv10_extended_energy_and_initial_force_match() {
    let system = System::molecule(vec![8, 1, 1], WATER.to_vec());
    for functional in ["wb97x-v", "wb97m-v"] {
        let common = Value::object()
            .with("xc", functional)
            .with("reference", "rks")
            .with("grid_level", 0usize)
            .with("vv10_grid_level", 0usize)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol_grad", 1e-9);
        let direct = run("gradient", &system, &Controls::from_value(&common).unwrap()).unwrap();
        let md = run(
            "car-parrinello-md",
            &system,
            &Controls::from_value(
                &common
                    .with("md_steps", 3usize)
                    .with("md_timestep", 1.0)
                    .with("fictitious_mass", 200.0)
                    .with("nuclear_velocities", VELOCITIES),
            )
            .unwrap(),
        )
        .unwrap();
        let frames = md.get("frames").unwrap().as_array().unwrap();
        let initial = &frames[0];
        let vv10 = number(
            direct.get("energy_components").unwrap(),
            "nonlocal_correlation",
        );
        let mut force_defect = 0.0_f64;
        let expected = direct.get("forces").unwrap().as_array().unwrap();
        let actual = initial.get("forces").unwrap().as_array().unwrap();
        for (expected, actual) in expected.iter().zip(actual) {
            for (a, b) in expected
                .as_array()
                .unwrap()
                .iter()
                .zip(actual.as_array().unwrap())
            {
                force_defect = force_defect.max((a.as_f64().unwrap() - b.as_f64().unwrap()).abs());
            }
        }
        let energy_defect = (number(&direct, "energy") - number(initial, "potential_energy")).abs();
        let drift = number(&md, "max_extended_energy_drift");
        println!("CPMD water {functional} VV10={vv10:.12e} E_initial={:.12e} |dE|={energy_defect:.12e} |dF|max={force_defect:.12e} max drift={drift:.12e} orbital={:.12e}",
            number(initial,"potential_energy"),number(&md,"scf_orbital_residual"));
        assert!(vv10.abs() > 1e-6);
        assert!(energy_defect < 2e-9 && force_defect < 2e-7 && drift < 2e-5);
    }
}

#[test]
fn ammonia_cation_open_shell_trajectories_use_the_requested_reference() {
    // Distorted NH3+ keeps distinct open-shell orbitals and removes symmetry.
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.13, -0.27, 0.19],
            [1.78, 0.34, -0.08],
            [-0.49, 1.65, 0.41],
            [0.31, -0.78, 1.91],
        ],
    );
    let velocities = [
        [0.0; 3],
        [0.001, 0.0, 0.0],
        [0.0, 0.001, 0.0],
        [0.0, 0.0, 0.001],
    ];
    for xc in ["wb97x-d", "wb97x-v"] {
        for reference in ["uks", "roks"] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("xc", xc)
                    .with("reference", reference)
                    .with("charge", 1usize)
                    .with("multiplicity", 2usize)
                    .with("grid_level", 0usize)
                    .with("vv10_grid_level", 0usize)
                    .with("grid_weight_cutoff", 0.0)
                    .with("md_steps", 2usize)
                    .with("md_timestep", 1.0)
                    .with("nuclear_velocities", velocities),
            )
            .unwrap();
            let report = run("car-parrinello-md", &system, &controls).unwrap();
            assert_eq!(
                report.get("reference_type").unwrap().as_str(),
                Some(reference)
            );
            let frames = report.get("frames").unwrap().as_array().unwrap();
            assert_eq!(frames.len(), 3);
            let energy = number(&frames[0], "extended_energy");
            let drift = number(&report, "max_extended_energy_drift");
            let residual = number(&report, "scf_orbital_residual");
            println!("CPMD ammonia cation {xc} {reference} E_initial={energy:.12e} drift={drift:.12e} orbital={residual:.12e}");
            assert!(energy.is_finite() && residual < 1e-7 && drift < 2e-5);
            assert!(number(&frames[2], "electronic_kinetic_energy") > 0.0);
            for frame in frames {
                assert!(number(frame, "orbital_orthogonality_defect") < 1e-12);
                assert!(number(frame, "orbital_tangency_defect") < 1e-12);
            }
        }
    }
}

#[test]
fn water_complex_spinor_trajectories_keep_hermitian_constraints() {
    let system = System::molecule(vec![8, 1, 1], WATER.to_vec());
    for xc in ["wb97x-d", "wb97x-v"] {
        let input = Value::object()
            .with("xc", xc)
            .with("grid_level", 0usize)
            .with("vv10_grid_level", 0usize)
            .with("grid_weight_cutoff", 0.0)
            .with("reference", "gks")
            .with("md_steps", 2usize)
            .with("md_timestep", 1.0)
            .with("fictitious_mass", 200.0)
            .with("nuclear_velocities", VELOCITIES);
        let report = run(
            "car-parrinello-md",
            &system,
            &Controls::from_value(&input).unwrap(),
        )
        .unwrap();
        assert_eq!(report.get("reference_type").unwrap().as_str(), Some("gks"));
        let frames = report.get("frames").unwrap().as_array().unwrap();
        let drift = number(&report, "max_extended_energy_drift");
        assert!(drift < 2e-5);
        assert!(number(&frames[2], "electronic_kinetic_energy") > 0.0);
        for frame in frames {
            assert!(number(frame, "orbital_orthogonality_defect") < 1e-12);
            assert!(number(frame, "orbital_tangency_defect") < 1e-12);
        }
        // A spin-paired initial projector embeds RKS in the full complex
        // representation. Its stationary force and energy must agree.
        let collinear = run(
            "car-parrinello-md",
            &system,
            &Controls::from_value(&input.with("reference", "rks")).unwrap(),
        )
        .unwrap();
        let rks_initial = &collinear.get("frames").unwrap().as_array().unwrap()[0];
        let energy = number(&frames[0], "potential_energy");
        assert!((energy - number(rks_initial, "potential_energy")).abs() < 1e-8);
        let (_, gks_forces) = frames[0].get("forces").unwrap().numeric_array().unwrap();
        let (_, rks_forces) = rks_initial.get("forces").unwrap().numeric_array().unwrap();
        let force_defect = gks_forces
            .iter()
            .zip(rks_forces)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("CPMD water GKS {xc} E_initial={energy:.15e} drift={drift:.15e} RKS force defect={force_defect:.15e}");
        assert!(force_defect < 2e-7);
    }
}

#[path = "support/convergence.rs"]
mod convergence;
