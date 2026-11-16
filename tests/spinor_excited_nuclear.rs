// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full Cartesian spinor excitation derivatives on a C1 odd-electron
//! molecule. The scalar oracle assembles every spin-orbital TDA element
//! independently from AO ERIs; no production excitation kernel is called.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dfpt::orbitals::generalized_orbitals;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::{
    spinor_scf::{run_spinor_scf, SpinorScfConfig},
    BlochMatrix,
};
use dft_rs::response::complex::Complex as C;
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};

fn molecule() -> System {
    System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    )
}
fn controls() -> Controls {
    let operator = spinor_operator::source(4);
    Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("reference", "gks")
            .with("unrestricted", true)
            .with("charge", 1i32)
            .with("multiplicity", 2usize)
            .with("guess", "core")
            .with("max_iter", 500usize)
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("grid_weight_cutoff", 0.0)
            .with("nstates", 15usize)
            .with("td_tolerance", 1e-9)
            .with("gks_operator_real", &operator.re)
            .with("gks_operator_imaginary", &operator.im),
    )
    .unwrap()
}
fn z(matrix: &BlochMatrix, i: usize, j: usize) -> C {
    C::new(matrix.re[(i, j)], matrix.im[(i, j)])
}
fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}
fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn independent_energy(system: &System) -> (f64, Vec<f64>) {
    let method = MolecularMethod::from_controls(&controls()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let n = setup.basis.nao_sph;
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &seed.scf,
        &setup.options,
        spinor_operator::source(n),
        [0.0, 0.0, 1.0],
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-10,
    )
    .unwrap();
    let reference = run_spinor_scf(
        &backend,
        &SpinorScfConfig {
            energy_tolerance: 1e-14,
            density_tolerance: 1e-14,
            orbital_tolerance: 1e-13,
            max_iter: 500,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reference.converged);
    let occupied = reference.occupations[0]
        .iter()
        .filter(|&&f| f == 1.0)
        .count();
    assert_eq!(occupied, 3);
    let coefficients = &reference.bands[0].coefficients;
    let energies = &reference.bands[0].energies;
    let empty = coefficients.cols() - occupied;
    let dim = occupied * empty;
    let eri = dft_rs::integrals::eri::eri_incore(
        &setup.basis,
        dft_rs::integrals::CoulombKernel::Coulomb,
        256,
    )
    .unwrap();
    let mut matrix = BlochMatrix::zeros(dim, dim);
    for column in 0..dim {
        let b = occupied + column / occupied;
        let j = column % occupied;
        let density = |mu, nu| z(coefficients, mu, b) * z(coefficients, nu, j).conj();
        let mut induced = vec![C::ZERO; 4 * n * n];
        for s in 0..2 {
            for t in 0..2 {
                for mu in 0..n {
                    for nu in 0..n {
                        let mut value = C::ZERO;
                        for kappa in 0..n {
                            for lambda in 0..n {
                                value -= density(s * n + kappa, t * n + lambda)
                                    * eri.get(mu, kappa, nu, lambda);
                                if s == t {
                                    for spin in 0..2 {
                                        value += density(spin * n + kappa, spin * n + lambda)
                                            * eri.get(mu, nu, kappa, lambda);
                                    }
                                }
                            }
                        }
                        induced[(s * n + mu) * 2 * n + t * n + nu] = value;
                    }
                }
            }
        }
        for row in 0..dim {
            let a = occupied + row / occupied;
            let i = row % occupied;
            let mut value = C::ZERO;
            for mu in 0..2 * n {
                for nu in 0..2 * n {
                    value += z(coefficients, mu, a).conj()
                        * induced[mu * 2 * n + nu]
                        * z(coefficients, nu, i);
                }
            }
            if row == column {
                value.re += energies[a] - energies[i];
            }
            matrix.re[(row, column)] = value.re;
            matrix.im[(row, column)] = value.im;
        }
    }
    assert!(matrix.hermiticity_error() < 1e-11);
    let identity = BlochMatrix {
        re: Matrix::identity(dim),
        im: Matrix::zeros(dim, dim),
    };
    let spectrum = generalized_orbitals(&matrix, &identity, 1e-12).unwrap();
    assert!(
        spectrum.energies[0] > 0.0,
        "reference has a nonpositive first pole"
    );
    assert!(
        spectrum.energies[1] - spectrum.energies[0] > 1e-3,
        "first pole is not isolated"
    );
    (
        reference.fields.energy + spectrum.energies[0],
        spectrum.energies,
    )
}

#[test]
fn full_spinor_hf_spectrum_matches_independent_ao_couplings() {
    let system = molecule();
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let report = run("excitations", &system, &controls()).unwrap();
    assert_eq!(
        report.get("td_spin").and_then(Value::as_str),
        Some("spinor")
    );
    let (_, expected) = independent_energy(&system);
    let states = report.get("states").and_then(Value::as_array).unwrap();
    assert_eq!(states.len(), expected.len());
    let gap = expected[1] - expected[0];
    let mut worst = 0.0_f64;
    for (state, energy) in states.iter().zip(expected) {
        worst = worst.max((number(state, "energy") - energy).abs());
        assert!((number(state, "energy") - energy).abs() < 1e-10);
    }
    println!("independent full-spin poles: count={} max_error={worst:.12e} Hartree first_gap={gap:.12e} Hartree",states.len());
}

#[test]
fn full_spinor_c1_hf_nuclear_gradient_and_hessian_fit_every_entry() {
    let system = molecule();
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let report = run("td-hessian", &system, &controls()).unwrap();
    assert_eq!(
        report.get("provenance").and_then(Value::as_str),
        Some("analytic")
    );
    let gradient = vector(&report, "gradient");
    let hessian = vector(&report, "hessian");
    assert_eq!(gradient.len(), 12);
    assert_eq!(hessian.len(), 144);
    let (energy, _) = independent_energy(&system);
    assert!((energy - number(&report, "excited_state_energy")).abs() < 1e-10);
    let stages = [
        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
        ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
        // Retain all original measurements and resolve three components
        // whose higher terms dominate those intervals; gates are unchanged.
        ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
        ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
    ];
    let mut failures = Vec::new();
    for column in 0..12 {
        let mut names = vec![format!("spinor excited gradient {column}")];
        names.extend((0..12).map(|row| format!("spinor excited Hessian {row},{column}")));
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 13];
                for (k, &h) in steps.iter().enumerate() {
                    let at = |displacement: f64| {
                        let mut point = system.clone();
                        point.positions[column / 3][column % 3] += displacement;
                        let report = run("td-state-gradient", &point, &controls()).unwrap();
                        let energy = independent_energy(&point).0;
                        assert!((energy - number(&report, "excited_state_energy")).abs() < 1e-10);
                        (energy, vector(&report, "gradient"))
                    };
                    let (ep, gp) = at(h);
                    let (em, gm) = at(-h);
                    let (ep2, gp2, em2, gm2) = if stencil == stencils::Stencil::Fourth {
                        let (ep, gp) = at(2.0 * h);
                        let (em, gm) = at(-2.0 * h);
                        (ep, gp, em, gm)
                    } else {
                        (0.0, vec![0.0; 12], 0.0, vec![0.0; 12])
                    };
                    errors[0][k] = (stencil.first(ep, em, ep2, em2, h) - gradient[column]).abs();
                    for row in 0..12 {
                        errors[row + 1][k] = (stencil
                            .first(gp[row], gm[row], gp2[row], gm2[row], h)
                            - hessian[12 * row + column])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("full-spin C1 HF excited nuclear derivatives", &failures);
}
