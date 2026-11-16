// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent lower-order finite-k Hessians validate every volume-jet entry.
//! The neutral triclinic H/H/He cell has C1 symmetry and a nonempty virtual
//! space. It is a compact Gaussian model, not a material fit.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::dispersion::Dispersion;
use dft_rs::linalg::Matrix;
use dft_rs::periodic::dispersion::{periodic_d3, periodic_dispersion_hessian, PeriodicDispersion};
use dft_rs::periodic::phonon::gamma_force_constants;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::krylov::ResponseSolverConfig;
use dft_rs::response::nuclear::{
    lattice_volume_gruneisen, lattice_volume_gruneisen_with_dispersion,
};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, ResolvedFunctional, Vec3};

fn fixture(x: f64, charge: i32) -> (Cell, PeriodicScfOptions) {
    let scale = (x / 3.0).exp();
    let mut basis = BasisSet::new("C1 binary volume-derivative model");
    for (z, exponent) in [(1, 1.2), (2, 1.6)] {
        basis.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.0],
            }],
        });
    }
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.11, -0.17, 0.23) * scale,
        },
        Atom {
            z: 1,
            position: Vec3::new(1.24, 0.19, 0.57) * scale,
        },
        Atom {
            z: 2,
            position: Vec3::new(-0.37, 1.59, 0.42) * scale,
        },
    ])
    .with_charge(charge);
    let cell = Cell::new(
        molecule,
        Lattice::new(
            Vec3::new(7.3, 0.21, 0.17) * scale,
            Vec3::new(0.35, 7.9, 0.41) * scale,
            Vec3::new(0.13, 0.29, 8.6) * scale,
        ),
        &basis,
        3.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        exchange_cutoff: Some(3.0),
        grid: dft_rs::grid::GridOptions {
            weight_cutoff: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    (cell, options)
}

fn independent_dynamical(
    x: f64,
    functional: &ResolvedFunctional,
    dispersion: Option<&PeriodicDispersion>,
) -> (Matrix, f64) {
    let (cell, options) = fixture(x, 0);
    let reference = run_periodic_scf(&cell, functional, &options).unwrap();
    assert!(reference.converged);
    let hessian = gamma_force_constants(
        &cell,
        functional,
        &reference,
        &options,
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    println!(
        "independent x={x:.8}: energy={:.16e}, Hessian symmetry={:.3e}, ASR={:.3e}",
        reference.energy, hessian.raw_asymmetry, hessian.acoustic_residual
    );
    let inverse_mass_roots: Vec<f64> = cell
        .molecule()
        .atoms
        .iter()
        .map(|a| 1.0 / dft_rs::constants::atomic_mass_au(a.z).sqrt())
        .collect();
    let mut dynamical = hessian.hessian;
    let mut energy = reference.energy;
    if let Some(model) = dispersion {
        // The conventional Gamma Hessian and direct image sums are independent
        // of the lattice energy Taylor series being tested.
        let numbers: Vec<_> = cell.molecule().atoms.iter().map(|a| a.z).collect();
        let positions: Vec<_> = cell.molecule().atoms.iter().map(|a| a.position).collect();
        let lattice = cell.lattice();
        let correction = periodic_dispersion_hessian(&numbers, &positions, lattice, model).unwrap();
        for (value, extra) in dynamical.as_mut_slice().iter_mut().zip(correction.as_slice()) {
            *value += extra;
        }
        energy += periodic_d3(&numbers, &positions, lattice, model)
            .unwrap()
            .energy;
    }
    for i in 0..dynamical.rows {
        for j in 0..dynamical.cols {
            dynamical[(i, j)] *= inverse_mass_roots[i / 3] * inverse_mass_roots[j / 3];
        }
    }
    (dynamical, energy)
}

#[test]
fn all_clamped_volume_jet_components_match_independent_phonon_hessians() {
    check_clamped_volume_jets("hf", None);
}

#[test]
fn all_pbe_d2_volume_jet_components_match_independent_phonon_hessians() {
    // The longest intracell pair stays below the cutoff and every cell image
    // stays above it at every stencil point, including plus/minus 2h.
    let model = PeriodicDispersion::from_model(&Dispersion::D2, "pbe", 3.5, 3.5)
        .unwrap()
        .unwrap();
    check_clamped_volume_jets("pbe", Some(&model));
}

fn check_clamped_volume_jets(name: &str, dispersion: Option<&PeriodicDispersion>) {
    let (cell, options) = fixture(0.0, 0);
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    let result = lattice_volume_gruneisen_with_dispersion(
        &cell,
        &functional,
        &reference,
        &options,
        1,
        false,
        1e-5,
        &ResponseSolverConfig::default(),
        dispersion,
    )
    .unwrap();
    assert!(!result.relaxed_ion);
    let n = 3 * cell.molecule().len();
    assert_eq!(result.internal_first, vec![0.0; n]);
    assert_eq!(result.internal_second, vec![0.0; n]);
    let (zero, energy) = independent_dynamical(0.0, &functional, dispersion);
    if dispersion.is_some() {
        let correction = result.dispersion_energy_per_primitive_cell.unwrap();
        assert!(correction < 0.0);
        assert!((energy - reference.energy - correction).abs() < 1e-13);
        assert!(
            (result.reference.energy - energy - result.reference.energy_difference).abs() < 1e-13
        );
        println!(
            "{name} dispersion volume response: energy={energy:.17e} correction={correction:.17e} pressure={:.17e} gamma={:?} curvature={:?} series_residual={:.17e} response_residuals={:?}",
            result.pressure,
            result.spectrum.branches.iter().map(|b| b.first_gruneisen).collect::<Vec<_>>(),
            result.spectrum.branches.iter().map(|b| b.second_log_frequency).collect::<Vec<_>>(),
            result.reference.residual,
            result.response_residuals
        );
    }
    println!(
        "independent centre D defect={:.16e}",
        result.dynamical_jets[0].max_abs_diff(&zero)
    );
    assert!(result.dynamical_jets[0].max_abs_diff(&zero) < 1e-9);
    let entries: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let mut names: Vec<String> = entries
        .iter()
        .flat_map(|(i, j)| {
            [
                format!("volume D_x[{i},{j}]"),
                format!("volume D_xx[{i},{j}]"),
            ]
        })
        .collect();
    if dispersion.is_some() {
        names.push("volume reference pressure".into());
    }
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (k, &h) in steps.iter().enumerate() {
                let (plus, energy_plus) = independent_dynamical(h, &functional, dispersion);
                let (minus, energy_minus) = independent_dynamical(-h, &functional, dispersion);
                let ((plus2, energy_plus2), (minus2, energy_minus2)) =
                    if stencil == stencils::Stencil::Fourth {
                        (
                            independent_dynamical(2.0 * h, &functional, dispersion),
                            independent_dynamical(-2.0 * h, &functional, dispersion),
                        )
                    } else {
                        ((zero.clone(), energy), (zero.clone(), energy))
                    };
                for (index, &(i, j)) in entries.iter().enumerate() {
                    let first = stencil.first(
                        plus[(i, j)],
                        minus[(i, j)],
                        plus2[(i, j)],
                        minus2[(i, j)],
                        h,
                    );
                    let second = match stencil {
                        stencils::Stencil::Second => {
                            (plus[(i, j)] - 2.0 * zero[(i, j)] + minus[(i, j)]) / (h * h)
                        }
                        stencils::Stencil::Fourth => {
                            (-plus2[(i, j)] + 16.0 * plus[(i, j)] - 30.0 * zero[(i, j)]
                                + 16.0 * minus[(i, j)]
                                - minus2[(i, j)])
                                / (12.0 * h * h)
                        }
                    };
                    errors[2 * index][k] = (first - result.dynamical_jets[1][(i, j)]).abs();
                    errors[2 * index + 1][k] = (second - result.dynamical_jets[2][(i, j)]).abs();
                }
                if dispersion.is_some() {
                    let pressure =
                        -stencil.first(energy_plus, energy_minus, energy_plus2, energy_minus2, h)
                            / cell.lattice().volume().abs();
                    errors[2 * entries.len()][k] = (pressure - result.pressure).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all crystal volume-jet entries", &failures);
    for (b, acoustic) in result.spectrum.branches.iter().zip(&result.acoustic) {
        if *acoustic || b.squared_frequency <= 0.0 {
            assert!(b.first_gruneisen.is_none() && b.second_log_frequency.is_none());
        }
    }
}

#[test]
fn charged_volume_response_preserves_the_missing_background_derivative_guard() {
    let (cell, options) = fixture(0.0, 2);
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    let error = lattice_volume_gruneisen(
        &cell,
        &functional,
        &reference,
        &options,
        1,
        false,
        1e-5,
        &ResponseSolverConfig::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("charged cell"), "{error}");
}
