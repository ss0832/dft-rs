// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Prepared M3 acceptance; requires the mixed_response module registration.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::local::{LocalCenter, LocalGaussianSpec};
use dft_rs::driver::continuous_species::{
    ChargeChart, ChargeParameter, ContinuousSpeciesModel, SpeciesFunctional,
};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};

fn model(f: SpeciesFunctional, open: bool) -> ContinuousSpeciesModel {
    let mol = Molecule::new(vec![
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
    ]);
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let parameter = |name: &str, charge_direction| ChargeParameter {
        name: name.into(),
        unit: "dimensionless".into(),
        scale: 1.0,
        bounds: [-0.8, 0.8],
        charge_direction,
    };
    let chart = ChargeChart {
        reference_charges: vec![8.0, 1.0, 1.0],
        parameters: vec![
            parameter("local", vec![0.0; 3]),
            parameter("charge", vec![0.17, -0.31, 0.14]),
        ],
        conserve_total_charge: true,
    };
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let result = if open {
        ContinuousSpeciesModel::new_with_spin(mol, basis, grid, [5, 4], chart, f, options, origin)
    } else {
        ContinuousSpeciesModel::new(mol, basis, grid, 10, chart, f, options, origin)
    }
    .unwrap();
    result
        .with_local_potentials(vec![LocalGaussianSpec {
            center: LocalCenter::Fixed(Vec3::new(0.38, -0.4, 0.5)),
            exponent: 0.63,
            reference_amplitude: 0.12,
            amplitude_direction: vec![1.0, 0.03],
            reference_steric_strength: 0.0,
            steric_direction: vec![0.0; 2],
            environment_exponents: vec![0.47, 0.81, 0.59],
            environment_hardness: vec![0.31, 0.73, 0.23],
            excluded_atoms: vec![true, true, false],
        }])
        .unwrap()
}

fn acceptance(f: SpeciesFunctional, open: bool) {
    let m = model(f, open);
    let q = [0.23, -0.19];
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let config = ResponseSolverConfig::default();
    let state = m.evaluate(&q, field, None).unwrap();
    let analytic = m
        .polarizability_parameter_response(&state, &config)
        .unwrap();
    println!(
        "mixed {} UKS={open} residuals={:?} projector={:?} stationary={:?} population={:?}",
        f.name(),
        analytic.solver_relative_residuals,
        analytic.projector_defects,
        analytic.stationarity_defects,
        analytic.particle_number_derivatives
    );
    assert!(analytic.converged);
    assert!(analytic.reciprocity_defect < 1e-8);
    let indices = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let names: Vec<_> = (0..2)
        .flat_map(|a| {
            indices
                .iter()
                .map(move |&(i, j)| format!("{} UKS={open} alpha_q[{a},{i},{j}]", f.name()))
        })
        .collect();
    let mut failures = Vec::new();
    // All 12 independent entries: q finite differences of fresh first-order CPKS.
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.0; 4]; 12];
            for a in 0..2 {
                for (k, &h) in steps.iter().enumerate() {
                    let response = |delta| {
                        let mut displaced = q;
                        displaced[a] += delta;
                        let s = m.evaluate(&displaced, field, None).unwrap();
                        m.sensitivity(&s, &config).unwrap()
                    };
                    let plus = response(h);
                    let minus = response(-h);
                    for (c, &(i, j)) in indices.iter().enumerate() {
                        errors[6 * a + c][k] =
                            ((plus.polarizability[i][j] - minus.polarizability[i][j]) / (2.0 * h)
                                - analytic.alpha_q[a][i][j])
                                .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    // Independently differentiate mu_q w.r.t. all 3 field axes. All 18 raw
    // Cartesian entries are measured, including both off-diagonal orders.
    let field_names: Vec<_> = (0..2)
        .flat_map(|a| {
            (0..3).flat_map(move |i| {
                (0..3).map(move |j| format!("{} UKS={open} field(mu_q)[{a},{i},{j}]", f.name()))
            })
        })
        .collect();
    components::fit_components(
        &field_names,
        &[[0.004, 0.002, 0.001, 0.0005], [0.008, 0.004, 0.002, 0.001]],
        |steps| {
            let mut errors = vec![[0.0; 4]; 18];
            for j in 0..3 {
                for (k, &h) in steps.iter().enumerate() {
                    let response = |delta| {
                        let mut bias = field.to_array();
                        bias[j] += delta;
                        let s = m
                            .evaluate(&q, Vec3::new(bias[0], bias[1], bias[2]), None)
                            .unwrap();
                        m.sensitivity(&s, &config).unwrap()
                    };
                    let plus = response(h);
                    let minus = response(-h);
                    for a in 0..2 {
                        for i in 0..3 {
                            errors[9 * a + 3 * i + j][k] = ((plus.dipole_jacobian[i][a]
                                - minus.dipole_jacobian[i][a])
                                / (2.0 * h)
                                - analytic.alpha_q[a][i][j])
                                .abs();
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("analytic mixed parameter response", &failures);
}

#[test]
fn lda_rks_mixed_components() {
    acceptance(SpeciesFunctional::Lda, false);
}
#[test]
fn pbe_rks_mixed_components() {
    acceptance(SpeciesFunctional::Pbe, false);
}
#[test]
fn lda_uks_mixed_components() {
    acceptance(SpeciesFunctional::Lda, true);
}
#[test]
fn pbe_uks_mixed_components() {
    acceptance(SpeciesFunctional::Pbe, true);
}
