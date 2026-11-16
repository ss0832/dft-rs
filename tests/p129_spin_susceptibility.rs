// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent nonlinear fixed-total-number Zeeman probes of the screened
//! static spin derivative. Finite fields and differences are validation only.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::data;
use dft_rs::dfpt::orbitals::generalized_orbitals;
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::fermi_thermodynamics::fermi_thermodynamics;
use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use dft_rs::periodic::scf::{
    fock_at_density, run_periodic_scf, PeriodicScfOptions, PeriodicScfResult,
};
use dft_rs::periodic::spin_susceptibility::longitudinal_spin_susceptibility;
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::krylov::ResponseSolverConfig;
use dft_rs::xc::{HybridParams, ResolvedFunctional};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn setup() -> (Cell, PeriodicScfOptions) {
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        occupation: OccupationScheme::FermiDirac {
            temperature_kelvin: 150000.0,
        },
        max_iter: 250,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            ..Default::default()
        },
        ..Default::default()
    };
    (cell, options)
}

/// Solve the full nonlinear spin-polarized equations at finite field with
/// a single chemical potential shared by all spin and k states. This route
/// evaluates energies/occupations and the nonlinear XC potential, never the
/// production response operator or its XC derivative.
fn finite_field_moment(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    field: f64,
    g: f64,
) -> (f64, f64, f64) {
    let mut density = reference.density.clone();
    let nk = reference.k_points.len();
    let n = cell.nao();
    let kt = options.occupation.temperature_kelvin() * BOLTZMANN_HARTREE_PER_K;
    for iteration in 0..400 {
        let mut fock = fock_at_density(
            cell,
            functional,
            &reference.k_points,
            &density,
            options,
            &reference.k_points,
            false,
        )
        .unwrap();
        for spin in 0..2 {
            for k in 0..nk {
                let shift = if spin == 0 {
                    g * field / 4.0
                } else {
                    -g * field / 4.0
                };
                for i in 0..n {
                    for j in 0..n {
                        fock[spin][k].re[(i, j)] += shift * reference.overlap[k].re[(i, j)];
                        fock[spin][k].im[(i, j)] += shift * reference.overlap[k].im[(i, j)];
                    }
                }
            }
        }
        let orbitals: [Vec<_>; 2] = std::array::from_fn(|spin| {
            (0..nk)
                .map(|k| {
                    generalized_orbitals(
                        &fock[spin][k],
                        &reference.overlap[k],
                        options.orth_threshold,
                    )
                    .unwrap()
                })
                .collect()
        });
        let energies: Vec<_> = orbitals
            .iter()
            .flatten()
            .flat_map(|o| o.energies.iter().copied())
            .collect();
        let thermodynamics = fermi_thermodynamics(
            &energies,
            1.0 / nk as f64,
            (reference.n_alpha + reference.n_beta) as f64,
            kt,
        )
        .unwrap();
        let mu = thermodynamics.chemical_potential;
        let next: [Vec<_>; 2] = std::array::from_fn(|spin| {
            orbitals[spin]
                .iter()
                .map(|o| {
                    let c = &o.coefficients;
                    let mut p = BlochMatrix::zeros(n, n);
                    for band in 0..o.energies.len() {
                        let f = 0.5 * (1.0 - ((o.energies[band] - mu) / (2.0 * kt)).tanh());
                        for i in 0..n {
                            for j in 0..n {
                                p.re[(i, j)] += f
                                    * (c.re[(i, band)] * c.re[(j, band)]
                                        + c.im[(i, band)] * c.im[(j, band)]);
                                p.im[(i, j)] += f
                                    * (c.im[(i, band)] * c.re[(j, band)]
                                        - c.re[(i, band)] * c.im[(j, band)]);
                            }
                        }
                    }
                    p
                })
                .collect()
        });
        let defect = density
            .iter()
            .flatten()
            .zip(next.iter().flatten())
            .map(|(a, b)| a.rms_diff(b))
            .fold(0.0_f64, f64::max);
        if defect < 1e-13 {
            let numbers: [f64; 2] = std::array::from_fn(|spin| {
                next[spin]
                    .iter()
                    .zip(&reference.overlap)
                    .map(|(p, s)| p.trace_product(s))
                    .sum::<f64>()
                    / nk as f64
            });
            return (
                -g * (numbers[0] - numbers[1]) / 4.0,
                (numbers[0] + numbers[1] - (reference.n_alpha + reference.n_beta) as f64).abs(),
                defect,
            );
        }
        for (p, target) in density.iter_mut().flatten().zip(next.iter().flatten()) {
            for (a, b) in p.re.as_mut_slice().iter_mut().zip(target.re.as_slice()) {
                *a = 0.5 * (*a + b);
            }
            for (a, b) in p.im.as_mut_slice().iter_mut().zip(target.im.as_slice()) {
                *a = 0.5 * (*a + b);
            }
        }
        assert!(
            iteration < 399,
            "finite-field nonlinear spin SCF did not converge: {defect:e}"
        );
    }
    unreachable!()
}

#[test]
fn screened_spin_response_matches_nonlinear_zeeman_fits_for_semilocal_classes() {
    let (cell, options) = setup();
    for name in ["svwn5", "pbe", "tpss"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        let response = longitudinal_spin_susceptibility(
            &cell,
            &functional,
            &reference,
            &options,
            &ResponseSolverConfig::default(),
            0.75,
            2.0,
        )
        .unwrap();
        assert!(response.converged, "{name}: {:?}", response.residuals);
        let chi = response.value.susceptibility;
        let steps = [0.08, 0.04, 0.02, 0.01];
        let mut estimates = Vec::new();
        let mut number_residual = 0.0_f64;
        let mut nonlinear_residual = 0.0_f64;
        for h in steps {
            let (plus, np, rp) =
                finite_field_moment(&cell, &functional, &reference, &options, h, 2.0);
            let (minus, nm, rm) =
                finite_field_moment(&cell, &functional, &reference, &options, -h, 2.0);
            estimates.push((plus - minus) / (2.0 * h));
            number_residual = number_residual.max(np).max(nm);
            nonlinear_residual = nonlinear_residual.max(rp).max(rm);
        }
        let errors: Vec<_> = estimates.iter().map(|v| (v - chi).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - chi).abs();
        let energies: Vec<_> = reference.mo_energy[0].iter().flatten().copied().collect();
        let thermodynamics = fermi_thermodynamics(
            &energies,
            2.0 / reference.k_points.len() as f64,
            2.0,
            options.occupation.temperature_kelvin() * BOLTZMANN_HARTREE_PER_K,
        )
        .unwrap();
        let bare_defect = (response.value.bare_susceptibility - 0.25 * thermodynamics.dos).abs();
        println!("{name} spin chi={chi:.16e} bare={:.16e} enhancement={:?} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e} number={number_residual:.12e} nonlinear={nonlinear_residual:.12e} bare_DOS={bare_defect:.12e} response={:?} charge={:.12e} spin_cancellation={:.12e} Hermiticity={:.12e}",
            response.value.bare_susceptibility, response.value.enhancement_factor,
            fit.exponent, fit.r_squared, response.residuals, response.value.charge_constraint_residual,
            response.value.spin_density_cancellation_residual, response.value.density_hermitian_residual);
        assert!((fit.exponent - 2.0).abs() < 0.15 && fit.r_squared > 0.999);
        assert!(richardson < 1e-7 && number_residual < 1e-11 && bare_defect < 1e-11);
        assert!(chi > response.value.bare_susceptibility);
        assert!(response.value.spin_density_cancellation_residual < 1e-10);
        assert!(response.value.density_hermitian_residual < 1e-12);
    }
}

#[test]
fn vanishing_spin_kernel_recovers_bare_dos_and_quadratic_g_scaling() {
    let (cell, options) = setup();
    let mut functional = ResolvedFunctional::hartree_fock();
    functional.hybrid = HybridParams::PURE;
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    let config = ResponseSolverConfig::default();
    let a = longitudinal_spin_susceptibility(
        &cell,
        &functional,
        &reference,
        &options,
        &config,
        0.75,
        2.0,
    )
    .unwrap();
    let b = longitudinal_spin_susceptibility(
        &cell,
        &functional,
        &reference,
        &options,
        &config,
        0.75,
        1.3,
    )
    .unwrap();
    assert!(a.converged && b.converged);
    let limit = (a.value.susceptibility - a.value.bare_susceptibility).abs();
    let scaling = (b.value.susceptibility - a.value.susceptibility * (1.3_f64 / 2.0).powi(2)).abs();
    println!("zero spin-kernel bare limit={limit:.12e}; g-squared scaling={scaling:.12e}");
    assert!(limit < 1e-12 && scaling < 1e-12);
    assert!(longitudinal_spin_susceptibility(
        &cell,
        &functional,
        &reference,
        &options,
        &config,
        0.75,
        f64::NAN
    )
    .is_err());
}
