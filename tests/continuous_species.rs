// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-template continuous-charge acceptance. Four-step fits retain every
//! failure and try a separately reported resolved interval without error floors.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::{
    ChargeChart, ChargeParameter, ContinuousSpeciesModel, SpeciesFunctional, SpeciesState,
};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::nuclear_matrix;
use dft_rs::math::Vec3;
use dft_rs::response::polarizability::polarizability;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use stencils::Stencil;

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

fn grid(mol: &Molecule) -> MolecularGrid {
    MolecularGrid::build(
        mol,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap()
}

fn parameter(name: &str, direction: Vec<f64>) -> ChargeParameter {
    ChargeParameter {
        name: name.into(),
        unit: "dimensionless".into(),
        scale: 1.0,
        bounds: [-1.5, 1.5],
        charge_direction: direction,
    }
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

fn model(
    mol: Molecule,
    electrons: usize,
    chart: ChargeChart,
    f: SpeciesFunctional,
    origin: Vec3,
) -> ContinuousSpeciesModel {
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = grid(&mol);
    ContinuousSpeciesModel::new(mol, basis, grid, electrons, chart, f, options(), origin).unwrap()
}

fn water_chart() -> ChargeChart {
    ChargeChart {
        reference_charges: vec![8.0, 1.0, 1.0],
        parameters: vec![
            parameter("redistribution", vec![0.17, -0.31, 0.14]),
            parameter("charging", vec![0.23, 0.11, -0.07]),
        ],
        conserve_total_charge: false,
    }
}

/// Independent first-energy derivative oracle. Assemble the weighted charge
/// operator directly, rather than using the production unit-charge cache.
/// Only re-converged SCF densities enter: endpoint polarizabilities and CPKS
/// solutions are unnecessary for finite differences of this stationary energy.
fn first_energy_oracle(
    mol: &Molecule,
    basis: &MolBasis,
    chart: &ChargeChart,
    state: &SpeciesState,
) -> Vec<f64> {
    let density = state.total_density();
    chart
        .parameters
        .iter()
        .map(|parameter| {
            let operator = nuclear_matrix(basis, mol, Some(&parameter.charge_direction));
            let mut value = density.frobenius_dot(&operator);
            for a in 0..mol.len() {
                let b_a = parameter.charge_direction[a];
                value -= b_a * (mol.atoms[a].position - state.origin()).dot(state.field());
                for b in a + 1..mol.len() {
                    value += (b_a * state.nuclear_charges()[b]
                        + state.nuclear_charges()[a] * parameter.charge_direction[b])
                        / (mol.atoms[a].position - mol.atoms[b].position).norm();
                }
            }
            value
        })
        .collect()
}

#[test]
fn chart_validation_rejects_invalid_domains() {
    let mut chart = water_chart();
    assert!(chart.charges(&[0.2, 0.1]).is_ok());
    assert!(chart.charges(&[f64::NAN, 0.0]).is_err());
    assert!(chart.charges(&[0.0]).is_err());
    assert!(chart.charges(&[2.0, 0.0]).is_err());
    chart.conserve_total_charge = true;
    assert!(chart.validate(3).is_err());
    chart.parameters.pop();
    assert!(chart.validate(3).is_ok());
    chart.parameters[0].scale = 0.0;
    assert!(chart.validate(3).is_err());
}

#[test]
fn filled_finite_ao_space_has_a_structural_zero_response() {
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::new(0.13, -0.21, 0.17),
    }]);
    let chart = ChargeChart {
        reference_charges: vec![2.0],
        parameters: vec![parameter("charging", vec![0.2])],
        conserve_total_charge: false,
    };
    let m = model(mol, 2, chart, SpeciesFunctional::Lda, Vec3::zero());
    let state = m.evaluate(&[0.25], Vec3::zero(), None).unwrap();
    let r = m
        .sensitivity(&state, &ResponseSolverConfig::default())
        .unwrap();
    assert!(state.frontier_gap().is_none());
    assert!(r.polarizability.iter().flatten().all(|x| x.abs() < 1e-14));
    assert!(r.energy_hessian[0][0].abs() < 1e-14);
    let other = m.evaluate(&[-0.25], Vec3::zero(), None).unwrap();
    assert!(state.total_density().max_abs_diff(&other.total_density()) < 1e-12);
    // Both spin projectors fill the whole single-AO space: no orbital
    // rotations exist, so these zeros are exact model limits, not placeholders.
    assert!(((state.energy() - other.energy()) / 0.5 - r.energy_gradient[0]).abs() < 1e-10);
}

#[test]
fn ordinary_hydrogen_water_nitrogen_reduction() {
    let molecules = [
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(-0.7, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.7, 0.0, 0.0),
            },
        ]),
        water(),
        Molecule::new(vec![
            Atom {
                z: 7,
                position: Vec3::new(0.0, 0.0, -1.05),
            },
            Atom {
                z: 7,
                position: Vec3::new(0.0, 0.0, 1.05),
            },
        ]),
    ];
    for f in [SpeciesFunctional::Lda, SpeciesFunctional::Pbe] {
        for mol in &molecules {
            let basis = MolBasis::build(mol, &data::load("sto-3g").unwrap()).unwrap();
            let grid = grid(mol);
            let functional = resolve(&FunctionalSpec::Combined(f.name().into())).unwrap();
            let ordinary = run_scf(mol, &basis, &options(), &functional, Some(&grid)).unwrap();
            let ordinary_alpha =
                polarizability(mol, &basis, &functional, Some(&grid), &ordinary, &options())
                    .unwrap();
            let ordinary_mu = dft_rs::properties::dipole_moment(mol, &basis, &ordinary);
            let ordinary_gradient = dft_rs::gradient::assemble_gradient(
                mol,
                &basis,
                &functional,
                Some(&grid),
                &ordinary,
                &options(),
            )
            .unwrap();
            let mut direction = vec![0.0; mol.len()];
            direction[0] = -1.0;
            direction[1] = 1.0;
            let chart = ChargeChart {
                reference_charges: mol.atoms.iter().map(|a| f64::from(a.z)).collect(),
                parameters: vec![parameter("charge_transfer", direction)],
                conserve_total_charge: true,
            };
            let extension = ContinuousSpeciesModel::new(
                mol.clone(),
                basis,
                grid,
                mol.n_electrons().unwrap(),
                chart,
                f,
                options(),
                Vec3::zero(),
            )
            .unwrap();
            let state = extension.evaluate(&[0.0], Vec3::zero(), None).unwrap();
            let response = extension
                .sensitivity(&state, &ResponseSolverConfig::default())
                .unwrap();
            let gradient = extension.nuclear_gradient(&state).unwrap();
            for (actual, expected) in gradient.iter().zip(&ordinary_gradient) {
                assert!((*actual - *expected).norm() < 1e-8);
            }
            let energy_error = (state.energy() - ordinary.energy).abs();
            let density_error = state.total_density().max_abs_diff(&ordinary.p_total());
            let dipole_error = (state.dipole - ordinary_mu).norm();
            println!("ordinary reduction {} atoms={} energy={energy_error:.9e} density={density_error:.9e} dipole={dipole_error:.9e}",f.name(),mol.len());
            assert!(energy_error < 1e-10);
            assert!(density_error < 1e-9);
            assert!(dipole_error < 1e-9);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((response.polarizability[i][j] - ordinary_alpha[i][j]).abs() < 1e-7);
                }
            }
        }
    }
}

fn derivative_components(f: SpeciesFunctional) {
    derivative_components_in_sector(f, false);
}

fn sector_model(
    mol: Molecule,
    f: SpeciesFunctional,
    origin: Vec3,
    open: bool,
) -> ContinuousSpeciesModel {
    if !open {
        return model(mol, 10, water_chart(), f, origin);
    }
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = grid(&mol);
    ContinuousSpeciesModel::new_with_spin(
        mol,
        basis,
        grid,
        [5, 4],
        water_chart(),
        f,
        options(),
        origin,
    )
    .unwrap()
}

fn derivative_components_in_sector(f: SpeciesFunctional, open: bool) {
    let oracle_molecule = water();
    let oracle_basis = MolBasis::build(&oracle_molecule, &data::load("sto-3g").unwrap()).unwrap();
    let oracle_chart = water_chart();
    let m = sector_model(water(), f, Vec3::new(0.11, 0.07, -0.09), open);
    println!(
        "fixed spin sector: {}",
        if open { "UKS [5,4]" } else { "RKS [5,5]" }
    );
    let q = [0.25, -0.19];
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let state = m.evaluate(&q, field, None).unwrap();
    let response = m
        .sensitivity(&state, &ResponseSolverConfig::default())
        .unwrap();
    assert!(response.hessian_symmetry_defect < 1e-8);
    assert!(response.polarizability_symmetry_defect < 1e-8);
    assert!(response.mixed_derivative_defect < 1e-8);
    println!(
        "{} q={q:?} E={:.15} mu={:?} Eq={:?} Eqq={:?} muq={:?} alpha={:?} SCF={:.3e} CPKS={:?}",
        f.name(),
        state.energy(),
        state.dipole,
        response.energy_gradient,
        response.energy_hessian,
        response.dipole_jacobian,
        response.polarizability,
        state.diagnostics.orbital_residual,
        response.response_residuals
    );
    let mut failures = Vec::new();
    let stages = [
        ([0.002, 0.001, 0.0005, 0.00025], Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
        ([0.1, 0.075, 0.05, 0.025], Stencil::Fourth),
    ];
    for a in 0..2 {
        let names = vec![
            format!("{} E_q[{a}]", f.name()),
            format!("{} E_qq[0,{a}]", f.name()),
            format!("{} E_qq[1,{a}]", f.name()),
            format!("{} mu_q[x,{a}]", f.name()),
            format!("{} mu_q[y,{a}]", f.name()),
            format!("{} mu_q[z,{a}]", f.name()),
        ];
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 6];
                for (k, &h) in steps.iter().enumerate() {
                    let evaluate = |offset: f64| {
                        let mut point = q;
                        point[a] += offset;
                        let s = m.evaluate(&point, field, None).unwrap();
                        let first =
                            first_energy_oracle(&oracle_molecule, &oracle_basis, &oracle_chart, &s);
                        [
                            s.energy(),
                            first[0],
                            first[1],
                            s.dipole.x,
                            s.dipole.y,
                            s.dipole.z,
                        ]
                    };
                    let p = evaluate(h);
                    let n = evaluate(-h);
                    let (p2, n2) = if stencil == Stencil::Fourth {
                        (evaluate(2.0 * h), evaluate(-2.0 * h))
                    } else {
                        ([0.0; 6], [0.0; 6])
                    };
                    let expected = [
                        response.energy_gradient[a],
                        response.energy_hessian[0][a],
                        response.energy_hessian[1][a],
                        response.dipole_jacobian[0][a],
                        response.dipole_jacobian[1][a],
                        response.dipole_jacobian[2][a],
                    ];
                    for c in 0..6 {
                        errors[c][k] =
                            (stencil.first(p[c], n[c], p2[c], n2[c], h) - expected[c]).abs();
                    }
                }
                if steps[0] == 0.002 && stencil == Stencil::Second {
                    println!(
                        "{} q-column {a} absolute errors at h={}: {:?}",
                        f.name(),
                        steps[3],
                        errors.iter().map(|v| v[3]).collect::<Vec<_>>()
                    );
                    assert!(errors[0][3] < 1e-7, "energy-gradient absolute accuracy");
                    assert!(
                        errors[1..].iter().all(|v| v[3] < 1e-5),
                        "energy-Hessian/dipole-Jacobian absolute accuracy"
                    );
                }
                errors
            },
            &mut failures,
        );
    }
    // Every alpha component and both independent E_q/F mixed components.
    for j in 0..3 {
        let names: Vec<_> = (0..3)
            .map(|i| format!("{} alpha[{i},{j}]", f.name()))
            .chain((0..2).map(|a| format!("{} -E_qF[{a},{j}]", f.name())))
            .collect();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 5];
                for (k, &h) in steps.iter().enumerate() {
                    let evaluate = |offset: f64| {
                        let mut components = [field.x, field.y, field.z];
                        components[j] += offset;
                        let s = m
                            .evaluate(
                                &q,
                                Vec3::new(components[0], components[1], components[2]),
                                None,
                            )
                            .unwrap();
                        let first =
                            first_energy_oracle(&oracle_molecule, &oracle_basis, &oracle_chart, &s);
                        [s.dipole.x, s.dipole.y, s.dipole.z, -first[0], -first[1]]
                    };
                    let p = evaluate(h);
                    let n = evaluate(-h);
                    let (p2, n2) = if stencil == Stencil::Fourth {
                        (evaluate(2.0 * h), evaluate(-2.0 * h))
                    } else {
                        ([0.0; 5], [0.0; 5])
                    };
                    let expected = [
                        response.polarizability[0][j],
                        response.polarizability[1][j],
                        response.polarizability[2][j],
                        response.dipole_jacobian[j][0],
                        response.dipole_jacobian[j][1],
                    ];
                    for c in 0..5 {
                        errors[c][k] =
                            (stencil.first(p[c], n[c], p2[c], n2[c], h) - expected[c]).abs();
                    }
                }
                if steps[0] == 0.002 && stencil == Stencil::Second {
                    println!(
                        "{} field-column {j} absolute errors at h={}: {:?}",
                        f.name(),
                        steps[3],
                        errors.iter().map(|v| v[3]).collect::<Vec<_>>()
                    );
                    assert!(
                        errors[..3].iter().all(|v| v[3] < 1e-4),
                        "polarizability field oracle absolute accuracy"
                    );
                    assert!(
                        errors[3..].iter().all(|v| v[3] < 1e-5),
                        "mixed charge-field absolute accuracy"
                    );
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures(
        "continuous species all-component derivative fits",
        &failures,
    );
}

#[test]
fn lda_all_component_four_step_fits() {
    derivative_components(SpeciesFunctional::Lda);
}

#[test]
fn pbe_all_component_four_step_fits() {
    derivative_components(SpeciesFunctional::Pbe);
}

#[test]
fn fixed_nitrogen_template_path_and_parity() {
    let mol = Molecule::new(vec![
        Atom {
            z: 7,
            position: Vec3::new(0.0, 0.0, -1.05),
        },
        Atom {
            z: 7,
            position: Vec3::new(0.0, 0.0, 1.05),
        },
    ]);
    let chart = ChargeChart {
        reference_charges: vec![7.0, 7.0],
        parameters: vec![parameter("lambda", vec![-1.0, 1.0])],
        conserve_total_charge: true,
    };
    for f in [SpeciesFunctional::Lda, SpeciesFunctional::Pbe] {
        let m = model(mol.clone(), 14, chart.clone(), f, Vec3::zero());
        let mut previous = None;
        for lambda in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let s = m
                .evaluate(&[lambda], Vec3::zero(), previous.as_ref())
                .unwrap();
            let independent = m.evaluate(&[lambda], Vec3::zero(), None).unwrap();
            assert!((s.energy() - independent.energy()).abs() < 1e-10);
            let r = m.sensitivity(&s, &ResponseSolverConfig::default()).unwrap();
            println!("path {} lambda={lambda} Z={:?} Ne={} Q={} E={:.15} mu_z={:.12} Eq={:?} Eqq={:?} muq={:?} alpha={:?}",
                f.name(),s.nuclear_charges(),s.electron_count(),s.total_charge(),s.energy(),s.dipole.z,
                r.energy_gradient,r.energy_hessian,r.dipole_jacobian,r.polarizability);
            assert_eq!(s.nuclear_charges(), &[7.0 - lambda, 7.0 + lambda]);
            assert_eq!(s.total_charge(), 0.0);
            previous = Some(s);
        }
        for lambda in [0.75, 0.5, 0.25, 0.0] {
            let s = m
                .evaluate(&[lambda], Vec3::zero(), previous.as_ref())
                .unwrap();
            let independent = m.evaluate(&[lambda], Vec3::zero(), None).unwrap();
            println!("reverse path {} lambda={lambda} energy difference={:.6e} density difference={:.6e}",
                f.name(),(s.energy()-independent.energy()).abs(),s.total_density().max_abs_diff(&independent.total_density()));
            assert!((s.energy() - independent.energy()).abs() < 1e-10);
            assert!(s.total_density().max_abs_diff(&independent.total_density()) < 1e-8);
            previous = Some(s);
        }
        let p = m.evaluate(&[0.25], Vec3::zero(), None).unwrap();
        let n = m.evaluate(&[-0.25], Vec3::zero(), None).unwrap();
        let pr = m.sensitivity(&p, &ResponseSolverConfig::default()).unwrap();
        let nr = m.sensitivity(&n, &ResponseSolverConfig::default()).unwrap();
        assert!((p.energy() - n.energy()).abs() < 1e-10);
        assert!((p.dipole.z + n.dipole.z).abs() < 1e-9);
        for i in 0..3 {
            for j in 0..3 {
                assert!((pr.polarizability[i][j] - nr.polarizability[i][j]).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn asymmetric_nitrogen_path_derivative_fits() {
    // Tilt the 2.10-bohr bond so every dipole-Jacobian component is nonzero.
    let direction = Vec3::new(0.36, 0.48, 0.8);
    let mol = Molecule::new(vec![
        Atom {
            z: 7,
            position: direction * (-1.05),
        },
        Atom {
            z: 7,
            position: direction * 1.05,
        },
    ]);
    let chart = ChargeChart {
        reference_charges: vec![7.0, 7.0],
        parameters: vec![parameter("lambda", vec![-1.0, 1.0])],
        conserve_total_charge: true,
    };
    let oracle_basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    for f in [SpeciesFunctional::Lda, SpeciesFunctional::Pbe] {
        let m = model(mol.clone(), 14, chart.clone(), f, Vec3::zero());
        let state = m.evaluate(&[0.25], Vec3::zero(), None).unwrap();
        let response = m
            .sensitivity(&state, &ResponseSolverConfig::default())
            .unwrap();
        let expected = [
            response.energy_gradient[0],
            response.energy_hessian[0][0],
            response.dipole_jacobian[0][0],
            response.dipole_jacobian[1][0],
            response.dipole_jacobian[2][0],
        ];
        let names: Vec<_> = ["E_q", "E_qq", "mu_q_x", "mu_q_y", "mu_q_z"]
            .iter()
            .map(|name| format!("{} N-template lambda=0.25 {name}", f.name()))
            .collect();
        let mut failures = Vec::new();
        components::fit_components(
            &names,
            &[
                [0.002, 0.001, 0.0005, 0.00025],
                [0.04, 0.02, 0.01, 0.005],
                [0.08, 0.04, 0.02, 0.01],
            ],
            |steps| {
                let mut errors = vec![[0.0; 4]; 5];
                for (k, &h) in steps.iter().enumerate() {
                    let evaluate = |offset: f64| {
                        let s = m.evaluate(&[0.25 + offset], Vec3::zero(), None).unwrap();
                        let first = first_energy_oracle(&mol, &oracle_basis, &chart, &s);
                        [s.energy(), first[0], s.dipole.x, s.dipole.y, s.dipole.z]
                    };
                    let p = evaluate(h);
                    let n = evaluate(-h);
                    for c in 0..5 {
                        errors[c][k] = ((p[c] - n[c]) / (2.0 * h) - expected[c]).abs();
                    }
                }
                if steps[0] == 0.002 {
                    println!(
                        "{} N-template lambda=0.25 absolute errors h={}: {:?}",
                        f.name(),
                        steps[3],
                        errors.iter().map(|v| v[3]).collect::<Vec<_>>()
                    );
                    assert!(errors[0][3] < 1e-7);
                    assert!(errors[1..].iter().all(|v| v[3] < 1e-5));
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("asymmetric fixed-nitrogen-template charge path", &failures);
    }
}

#[test]
fn charged_translation_and_cross_chart_rejection() {
    let f = SpeciesFunctional::Pbe;
    let mol = water();
    let chart = water_chart();
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let m = model(mol.clone(), 10, chart.clone(), f, origin);
    let t = Vec3::new(0.32, -0.41, 0.23);
    let mut translated = mol;
    for atom in &mut translated.atoms {
        atom.position += t;
    }
    let shifted = model(translated, 10, chart.clone(), f, origin);
    let s = m.evaluate(&[0.25, -0.19], Vec3::zero(), None).unwrap();
    let ss = shifted
        .evaluate(&[0.25, -0.19], Vec3::zero(), None)
        .unwrap();
    assert!((s.energy() - ss.energy()).abs() < 1e-10);
    assert!((ss.dipole - s.dipole - t * s.total_charge()).norm() < 1e-8);
    let r = m.sensitivity(&s, &ResponseSolverConfig::default()).unwrap();
    let rr = shifted
        .sensitivity(&ss, &ResponseSolverConfig::default())
        .unwrap();
    for a in 0..2 {
        for i in 0..3 {
            let dq: f64 = chart.parameters[a].charge_direction.iter().sum();
            assert!(
                (rr.dipole_jacobian[i][a] - r.dipole_jacobian[i][a] - dq * t.get(i)).abs() < 1e-7
            );
        }
    }
    assert!(shifted
        .evaluate(&[0.25, -0.19], Vec3::zero(), Some(&s))
        .is_err());
    assert!(shifted
        .sensitivity(&s, &ResponseSolverConfig::default())
        .is_err());
}

#[test]
fn rigid_rotation_covariance() {
    // Generic proper orthogonal rotation, independent of molecular symmetry.
    let rotate = |v: Vec3| {
        Vec3::new(
            (v.x + 2.0 * v.y + 2.0 * v.z) / 3.0,
            (2.0 * v.x + v.y - 2.0 * v.z) / 3.0,
            (-2.0 * v.x + 2.0 * v.y - v.z) / 3.0,
        )
    };
    let rotation = [
        [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
        [-2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0],
    ];
    let mol = water();
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let fine_model = |geometry: Molecule, origin| {
        let basis = MolBasis::build(&geometry, &data::load("sto-3g").unwrap()).unwrap();
        let grid = MolecularGrid::build(
            &geometry,
            &GridOptions {
                level: 5,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap();
        ContinuousSpeciesModel::new(
            geometry,
            basis,
            grid,
            10,
            water_chart(),
            SpeciesFunctional::Pbe,
            options(),
            origin,
        )
        .unwrap()
    };
    let m = fine_model(mol.clone(), origin);
    let mut rotated = mol;
    for atom in &mut rotated.atoms {
        atom.position = rotate(atom.position);
    }
    let mm = fine_model(rotated, rotate(origin));
    let s = m.evaluate(&[0.25, -0.19], Vec3::zero(), None).unwrap();
    let ss = mm.evaluate(&[0.25, -0.19], Vec3::zero(), None).unwrap();
    let r = m.sensitivity(&s, &ResponseSolverConfig::default()).unwrap();
    let rr = mm
        .sensitivity(&ss, &ResponseSolverConfig::default())
        .unwrap();
    // Finite angular quadrature has a rotational discretization error;
    // this covariance check does not replace the component derivative fits.
    println!(
        "rotation energy error={:.6e} dipole error={:.6e}",
        (s.energy() - ss.energy()).abs(),
        (ss.dipole - rotate(s.dipole)).norm()
    );
    assert!((s.energy() - ss.energy()).abs() < 1e-7);
    assert!((ss.dipole - rotate(s.dipole)).norm() < 1e-6);
    for i in 0..3 {
        for j in 0..3 {
            let mut expected = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    expected += rotation[i][a] * r.polarizability[a][b] * rotation[j][b];
                }
            }
            println!(
                "rotation alpha[{i},{j}] error={:.6e}",
                (rr.polarizability[i][j] - expected).abs()
            );
            assert!((rr.polarizability[i][j] - expected).abs() < 1e-5);
        }
    }
}

fn nuclear_components(f: SpeciesFunctional) {
    nuclear_components_in_sector(f, false);
}

fn nuclear_components_in_sector(f: SpeciesFunctional, open: bool) {
    let mol = water();
    let q = [0.25, -0.19];
    let origin = Vec3::new(0.11, 0.07, -0.09);
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let m = sector_model(mol.clone(), f, origin, open);
    let s = m.evaluate(&q, field, None).unwrap();
    let gradient = m.nuclear_gradient(&s).unwrap();
    let names: Vec<_> = (0..3)
        .flat_map(|a| (0..3).map(move |i| format!("{} nuclear[{a},{i}]", f.name())))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.02, 0.01, 0.005, 0.0025], [0.04, 0.02, 0.01, 0.005]],
        |steps| {
            let mut errors = vec![[0.0; 4]; 9];
            for a in 0..3 {
                for i in 0..3 {
                    for (k, &h) in steps.iter().enumerate() {
                        let energy = |offset: f64| {
                            let mut displaced = mol.clone();
                            let mut p = displaced.atoms[a].position.to_array();
                            p[i] += offset;
                            displaced.atoms[a].position = Vec3::new(p[0], p[1], p[2]);
                            // Rebuild moving AO and quadrature with identical label/template
                            // definitions. The charge chart and electron count remain fixed.
                            sector_model(displaced, f, origin, open)
                                .evaluate(&q, field, None)
                                .unwrap()
                                .energy()
                        };
                        errors[3 * a + i][k] =
                            ((energy(h) - energy(-h)) / (2.0 * h) - gradient[a].get(i)).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    let sum = gradient.iter().fold(Vec3::zero(), |sum, &g| sum + g);
    assert!((sum + field * s.total_charge()).norm() < 1e-7);
    components::assert_no_failures("real-charge nuclear gradients all components", &failures);
}

#[test]
fn lda_nuclear_gradient_four_step_fits() {
    nuclear_components(SpeciesFunctional::Lda);
}

#[test]
fn pbe_nuclear_gradient_four_step_fits() {
    nuclear_components(SpeciesFunctional::Pbe);
}

#[test]
fn uks_lda_all_component_four_step_fits() {
    derivative_components_in_sector(SpeciesFunctional::Lda, true);
}

#[test]
fn uks_pbe_all_component_four_step_fits() {
    derivative_components_in_sector(SpeciesFunctional::Pbe, true);
}

#[test]
fn uks_lda_nuclear_gradient_four_step_fits() {
    nuclear_components_in_sector(SpeciesFunctional::Lda, true);
}

#[test]
fn uks_pbe_nuclear_gradient_four_step_fits() {
    nuclear_components_in_sector(SpeciesFunctional::Pbe, true);
}

#[test]
fn uks_ordinary_reduction_and_fixed_spin_contract() {
    use dft_rs::scf::Reference;
    for f in [SpeciesFunctional::Lda, SpeciesFunctional::Pbe] {
        let mol = water().with_charge(1).with_multiplicity(2);
        let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let quadrature = grid(&mol);
        let mut opts = options();
        opts.reference = Reference::Uhf;
        let functional = resolve(&FunctionalSpec::Combined(f.name().into())).unwrap();
        let reference = run_scf(&mol, &basis, &opts, &functional, Some(&quadrature)).unwrap();
        let alpha = polarizability(
            &mol,
            &basis,
            &functional,
            Some(&quadrature),
            &reference,
            &opts,
        )
        .unwrap();
        let gradient = dft_rs::gradient::assemble_gradient(
            &mol,
            &basis,
            &functional,
            Some(&quadrature),
            &reference,
            &opts,
        )
        .unwrap();
        let m = sector_model(water(), f, Vec3::zero(), true);
        let s = m.evaluate(&[0.0, 0.0], Vec3::zero(), None).unwrap();
        assert_eq!(s.spin_electron_counts(), [5, 4]);
        let repeated = m.evaluate(&[0.0, 0.0], Vec3::zero(), Some(&s)).unwrap();
        let subspace = repeated.occupied_subspace_overlap.unwrap();
        assert!((subspace[0] - 5.0).abs() < 1e-9);
        assert!((subspace[1] - 4.0).abs() < 1e-9);
        assert!((s.energy() - reference.energy).abs() < 1e-10);
        assert!(s.total_density().max_abs_diff(&reference.p_total()) < 1e-9);
        assert!((s.spin_density().frobenius_dot(&reference.overlap) - 1.0).abs() < 1e-9);
        let response = m.sensitivity(&s, &ResponseSolverConfig::default()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((response.polarizability[i][j] - alpha[i][j]).abs() < 1e-7);
            }
        }
        for (a, b) in m.nuclear_gradient(&s).unwrap().iter().zip(&gradient) {
            assert!((*a - *b).norm() < 1e-8);
        }
        let changed = m.evaluate(&[0.25, -0.19], Vec3::zero(), Some(&s)).unwrap();
        assert_eq!(changed.spin_electron_counts(), [5, 4]);
        assert!(s.spin_density().max_abs_diff(&changed.spin_density()) > 1e-6);
        assert!((changed.spin_density().frobenius_dot(&reference.overlap) - 1.0).abs() < 1e-9);
        let make = |counts| {
            ContinuousSpeciesModel::new_with_spin(
                water(),
                MolBasis::build(&water(), &data::load("sto-3g").unwrap()).unwrap(),
                grid(&water()),
                counts,
                water_chart(),
                f,
                options(),
                Vec3::zero(),
            )
        };
        assert!(make([4, 5]).is_err());
        assert!(make([8, 1]).is_err());
        assert!(make([0, 0]).is_err());
        assert!(make([5, 5])
            .unwrap()
            .evaluate(&[0.25, -0.19], Vec3::zero(), Some(&changed))
            .is_err());
    }
}

#[test]
fn unsupported_models_and_nonconvergence_are_errors() {
    let mol = water();
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let constructor = |count, options: ScfOptions, basis: MolBasis| {
        ContinuousSpeciesModel::new(
            mol.clone(),
            basis,
            grid(&mol),
            count,
            water_chart(),
            SpeciesFunctional::Pbe,
            options,
            Vec3::zero(),
        )
    };
    assert!(constructor(9, options(), basis.clone()).is_err());
    assert!(constructor(100, options(), basis.clone()).is_err());
    let mut spin = options();
    spin.reference = dft_rs::scf::Reference::Uhf;
    assert!(constructor(10, spin, basis.clone()).is_err());
    let mut thermal = options();
    thermal.electronic_temperature = 0.01;
    assert!(constructor(10, thermal, basis.clone()).is_err());
    let mut ecp = basis.clone();
    ecp.n_core_electrons = 2;
    assert!(constructor(10, options(), ecp).is_err());
    let mut one_iteration = options();
    one_iteration.max_iter = 1;
    let m = constructor(10, one_iteration, basis).unwrap();
    assert!(m.evaluate(&[0.25, -0.19], Vec3::zero(), None).is_err());
    let mut periodic = dft_rs::driver::System::molecule(
        vec![8, 1, 1],
        mol.atoms.iter().map(|a| a.position.to_array()).collect(),
    );
    periodic.cell = Some([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]);
    periodic.pbc = [true; 3];
    assert!(dft_rs::driver::continuous_species::calculate(
        &periodic,
        &dft_rs::driver::Controls::new()
    )
    .is_err());
}

#[test]
fn common_fragment_reference_cancels_before_reaction_derivative_arithmetic() {
    // Unequal N-H environments and a nonplanar fourth atom give a C1
    // geometry before applying the generic external electric field.
    let mut r = water();
    r.atoms[0].z = 7;
    r.atoms.push(Atom {
        z: 1,
        position: Vec3::new(-0.27, -0.89, -1.06),
    });
    let mut p = r.clone();
    p.atoms[1].position += Vec3::new(0.19, -0.13, 0.17);
    let chart = ChargeChart {
        reference_charges: vec![7.0, 1.0, 1.0, 1.0],
        parameters: vec![
            parameter("redistribution", vec![0.17, -0.31, 0.19, -0.05]),
            parameter("charging", vec![0.23, 0.11, -0.07, 0.09]),
        ],
        conserve_total_charge: false,
    };
    let bases: Vec<_> = [&r, &p]
        .into_iter()
        .map(|mol| MolBasis::build(mol, &data::load("sto-3g").unwrap()).unwrap())
        .collect();
    let field = Vec3::new(0.0013, -0.0009, 0.0017);
    let q = [0.25, -0.19];
    let config = ResponseSolverConfig::default();
    let make = |mol: &Molecule, reference| {
        model(
            mol.clone(),
            10,
            chart.clone(),
            SpeciesFunctional::Lda,
            Vec3::new(0.11, 0.07, -0.09),
        )
        .with_fragment_internal_energy(reference)
        .unwrap()
    };
    // A large, parameter-dependent common reference makes subtraction after
    // addition lose the entire physical reaction energy. Remove it from the
    // scalar energy before differentiating; retain real classical interactions.
    let common_reference =
        |x: [f64; 2]| 1e80 * (1.0 + x[0] + x[1] * x[1] + x[0] * x[1] + x[0].powi(3));
    let mut energies = Vec::new();
    let mut gradients = Vec::new();
    let mut hessians = Vec::new();
    for mol in [&r, &p] {
        let baseline = make(mol, 0.0);
        let shifted = make(mol, common_reference(q));
        let base = baseline.evaluate(&q, field, None).unwrap();
        let state = shifted.evaluate(&q, field, None).unwrap();
        assert_eq!(state.energy_without_internal_reference(), base.energy());
        assert_eq!(
            state.classical_energy_without_internal_reference(),
            base.classical_energy()
        );
        assert_eq!(state.energy(), base.energy() + common_reference(q));
        assert_eq!(
            state.classical_energy(),
            base.classical_energy() + common_reference(q)
        );
        assert_eq!(state.fragment_internal_energy(), common_reference(q));
        assert_eq!(state.diagnostics.iterations, base.diagnostics.iterations);
        assert_eq!(
            state.diagnostics.orbital_residual,
            base.diagnostics.orbital_residual
        );
        assert_eq!(
            state.total_density().max_abs_diff(&base.total_density()),
            0.0
        );
        let response = shifted.sensitivity(&state, &config).unwrap();
        let base_response = baseline.sensitivity(&base, &config).unwrap();
        assert_eq!(response.energy_gradient, base_response.energy_gradient);
        assert_eq!(response.energy_hessian, base_response.energy_hessian);
        for (a, b) in shifted
            .nuclear_gradient(&state)
            .unwrap()
            .iter()
            .zip(baseline.nuclear_gradient(&base).unwrap())
        {
            assert_eq!((*a - b).norm(), 0.0);
        }
        println!("common reference: E_without={:.15} E_classical_without={:.15} SCF={:.3e} iterations={}",
            state.energy_without_internal_reference(), state.classical_energy_without_internal_reference(),
            state.diagnostics.orbital_residual, state.diagnostics.iterations);
        energies.push(state.energy_without_internal_reference());
        gradients.push(response.energy_gradient);
        hessians.push(response.energy_hessian);
    }
    let expected_gradient: Vec<_> = (0..2).map(|a| gradients[1][a] - gradients[0][a]).collect();
    let expected_hessian: Vec<Vec<_>> = (0..2)
        .map(|a| {
            (0..2)
                .map(|b| hessians[1][a][b] - hessians[0][a][b])
                .collect()
        })
        .collect();
    assert!((energies[1] - energies[0]).abs() > 1e-5);
    assert_eq!(
        (energies[1] + common_reference(q)) - (energies[0] + common_reference(q)),
        0.0
    );
    println!("reference-free reaction: J={:.15} gradient={expected_gradient:?} Hessian={expected_hessian:?}", energies[1] - energies[0]);
    let stages = [
        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
        ([0.1, 0.075, 0.05, 0.025], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for a in 0..2 {
        let names = vec![
            format!("common-reference J_q[{a}]"),
            format!("common-reference J_qq[0,{a}]"),
            format!("common-reference J_qq[1,{a}]"),
        ];
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 3];
                for (k, &h) in steps.iter().enumerate() {
                    let evaluate = |offset| {
                        let mut point = q;
                        point[a] += offset;
                        let values: Vec<_> = [&r, &p]
                            .into_iter()
                            .zip(&bases)
                            .map(|(mol, basis)| {
                                let state = make(mol, common_reference(point))
                                    .evaluate(&point, field, None)
                                    .unwrap();
                                let first = first_energy_oracle(mol, basis, &chart, &state);
                                [
                                    state.energy_without_internal_reference(),
                                    first[0],
                                    first[1],
                                ]
                            })
                            .collect();
                        std::array::from_fn::<_, 3, _>(|c| values[1][c] - values[0][c])
                    };
                    let plus = evaluate(h);
                    let minus = evaluate(-h);
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (evaluate(2.0 * h), evaluate(-2.0 * h))
                    } else {
                        ([0.0; 3], [0.0; 3])
                    };
                    let expected = [
                        expected_gradient[a],
                        expected_hessian[0][a],
                        expected_hessian[1][a],
                    ];
                    for c in 0..3 {
                        errors[c][k] = (stencil.first(plus[c], minus[c], plus2[c], minus2[c], h)
                            - expected[c])
                            .abs();
                    }
                }
                if steps[0] == 0.04 {
                    assert!(
                        errors.iter().all(|v| v[3] < 1e-6),
                        "reaction gradient/Hessian absolute accuracy"
                    );
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures(
        "reference-free reaction gradient and full Hessian",
        &failures,
    );
}
