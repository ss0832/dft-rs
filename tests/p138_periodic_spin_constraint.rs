// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nonlinear finite-field validation of a common-number multiplier in an
//! unequal-channel thermal reference. Finite fields are validation only.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::data;
use dft_rs::dfpt::orbitals::generalized_orbitals;
use dft_rs::dfpt::thermodynamic::fermi;
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
use dft_rs::scf::Reference;
use dft_rs::xc::{HybridParams, ResolvedFunctional};
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn setup() -> (Cell, PeriodicScfOptions) {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.9),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.9),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.7, 0.3, 0.2),
        },
    ])
    .with_multiplicity(2);
    let cell = Cell::new(
        molecule,
        Lattice::cubic(9.5),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        occupation: OccupationScheme::FermiDirac {
            temperature_kelvin: 150000.0,
        },
        reference: Reference::Uhf,
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

fn reciprocal_cutoff() -> f64 {
    std::env::var("DFT_RS_TEST_RECIPROCAL_CUTOFF")
        .map(|value| value.parse().expect("a finite reciprocal cutoff"))
        .unwrap_or(16.0)
}

fn density_components(density: &[Vec<BlochMatrix>; 2]) -> Vec<f64> {
    density
        .iter()
        .flatten()
        .flat_map(|p| p.re.as_slice().iter().chain(p.im.as_slice()))
        .copied()
        .collect()
}

/// Independent nonlinear SCF at fixed total number and fixed baseline spin
/// bias. The XC potential and occupations are rebuilt, without response kernels.
fn finite_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    field: f64,
    g: f64,
) -> (f64, f64, f64, f64, Vec<f64>) {
    let bias =
        0.5 * (reference.chemical_potential[0].unwrap() - reference.chemical_potential[1].unwrap());
    let kt = options.occupation.temperature_kelvin() * BOLTZMANN_HARTREE_PER_K;
    let nk = reference.k_points.len();
    let n = cell.nao();
    let mut density = reference.density.clone();
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
                    g * field / 4.0 - bias
                } else {
                    -g * field / 4.0 + bias
                };
                fock[spin][k].add_scaled(&reference.overlap[k], shift);
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
        let mu = fermi_thermodynamics(
            &energies,
            1.0 / nk as f64,
            (reference.n_alpha + reference.n_beta) as f64,
            kt,
        )
        .unwrap()
        .chemical_potential;
        let next: [Vec<_>; 2] = std::array::from_fn(|spin| {
            orbitals[spin]
                .iter()
                .map(|o| {
                    let c = &o.coefficients;
                    let mut p = BlochMatrix::zeros(n, n);
                    for band in 0..o.energies.len() {
                        let f = fermi(o.energies[band], mu, kt).unwrap().0;
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
                mu,
                (numbers.iter().sum::<f64>() - (reference.n_alpha + reference.n_beta) as f64).abs(),
                defect,
                density_components(&next),
            );
        }
        for (p, next) in density.iter_mut().flatten().zip(next.iter().flatten()) {
            p.scale(0.5);
            p.add_scaled(next, 0.5);
        }
        assert!(
            iteration < 399,
            "spin-biased finite-field SCF did not converge: {defect:e}"
        );
    }
    unreachable!()
}

#[test]
fn polarized_semilocal_response_matches_fixed_total_number_nonlinear_fits() {
    let (cell, options) = setup();
    for name in ["svwn5", "pbe", "tpss"] {
        if std::env::var("DFT_RS_TEST_MODEL").is_ok_and(|selected| selected != name) {
            continue;
        }
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        let response = longitudinal_spin_susceptibility(
            &cell,
            &functional,
            &reference,
            &options,
            &ResponseSolverConfig::default(),
            reciprocal_cutoff(),
            2.0,
        )
        .unwrap();
        assert!(reference.unrestricted && reference.n_alpha != reference.n_beta);
        assert!(response.converged, "{name}: {:?}", response.residuals);
        let value = &response.value;
        if reciprocal_cutoff() == 16.0 {
            let lower = longitudinal_spin_susceptibility(
                &cell,
                &functional,
                &reference,
                &options,
                &ResponseSolverConfig::default(),
                12.0,
                2.0,
            )
            .unwrap();
            println!(
                "{name} reciprocal cutoff 12 -> 16: chi change={:.12e}, common-mu change={:.12e}",
                (lower.value.susceptibility - value.susceptibility).abs(),
                (lower.value.chemical_potential_response.unwrap()
                    - value.chemical_potential_response.unwrap())
                .abs()
            );
        }
        let steps = [0.08, 0.04, 0.02, 0.01];
        let mut estimates = Vec::new();
        let mut mu_estimates = Vec::new();
        let mut density_estimates: Vec<Vec<f64>> = Vec::new();
        let mut number_defect = 0.0_f64;
        let mut nonlinear_defect = 0.0_f64;
        for h in steps {
            let plus = finite_field(&cell, &functional, &reference, &options, h, 2.0);
            let minus = finite_field(&cell, &functional, &reference, &options, -h, 2.0);
            estimates.push((plus.0 - minus.0) / (2.0 * h));
            mu_estimates.push((plus.1 - minus.1) / (2.0 * h));
            number_defect = number_defect.max(plus.2).max(minus.2);
            nonlinear_defect = nonlinear_defect.max(plus.3).max(minus.3);
            density_estimates.push(
                plus.4
                    .iter()
                    .zip(&minus.4)
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect(),
            );
        }
        let errors: Vec<_> = estimates
            .iter()
            .map(|x| (x - value.susceptibility).abs())
            .collect();
        let mu = value.chemical_potential_response.unwrap();
        let mu_errors: Vec<_> = mu_estimates.iter().map(|x| (x - mu).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let mu_fit = convergence::error_power_fit(&steps, &mu_errors);
        let analytic_density = density_components(&value.density);
        let density_errors: Vec<_> = density_estimates
            .iter()
            .map(|estimate| {
                estimate
                    .iter()
                    .zip(&analytic_density)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let density_fit = convergence::error_power_fit(&steps, &density_errors);
        let density_richardson = density_estimates[3]
            .iter()
            .zip(&density_estimates[2])
            .zip(&analytic_density)
            .map(|((a, b), exact)| ((4.0 * a - b) / 3.0 - exact).abs())
            .fold(0.0_f64, f64::max);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - value.susceptibility).abs();
        let mu_richardson = ((4.0 * mu_estimates[3] - mu_estimates[2]) / 3.0 - mu).abs();
        let dos: [f64; 2] = std::array::from_fn(|spin| {
            reference.mo_energy[spin]
                .iter()
                .flatten()
                .map(|e| {
                    -fermi(
                        *e,
                        reference.chemical_potential[spin].unwrap(),
                        options.occupation.temperature_kelvin() * BOLTZMANN_HARTREE_PER_K,
                    )
                    .unwrap()
                    .1
                })
                .sum::<f64>()
                / reference.k_points.len() as f64
        });
        let bare = dos[0] * dos[1] / (dos[0] + dos[1]);
        let bare_defect = (bare - value.bare_susceptibility).abs();
        println!("{name} polarized chi={:.16e} bare={:.16e} mu'={mu:.16e} bias={:?} cutoff={} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={richardson:.12e}; mu_errors={mu_errors:?} mu_p={:.9} mu_R2={:.12} mu_Richardson={mu_richardson:.12e}; bare_DOS={bare_defect:.12e} number={number_defect:.12e} nonlinear={nonlinear_defect:.12e} response={:?} charge={:.12e} raw_spin_sum={:.12e}",value.susceptibility,value.bare_susceptibility,value.reference_spin_bias,reciprocal_cutoff(),fit.exponent,fit.r_squared,mu_fit.exponent,mu_fit.r_squared,response.residuals,value.charge_constraint_residual,value.spin_density_cancellation_residual);
        println!("{name} complete AO spin-density fit: steps={steps:?} errors={density_errors:?} p={:.9} R2={:.12} Richardson={density_richardson:.12e}",density_fit.exponent,density_fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.15 && fit.r_squared > 0.999);
        assert!((mu_fit.exponent - 2.0).abs() < 0.15 && mu_fit.r_squared > 0.999);
        assert!((density_fit.exponent - 2.0).abs() < 0.15 && density_fit.r_squared > 0.999);
        assert!(density_richardson < 1e-7);
        assert!(
            richardson < 1e-7
                && mu_richardson < 1e-7
                && number_defect < 1e-11
                && bare_defect < 1e-11
        );
        assert!(
            value.charge_constraint_residual < 1e-10
                && value.bare_charge_constraint_residual < 1e-10
        );
        assert!(value.unconstrained_delta_numbers.iter().sum::<f64>().abs() > 1e-5);
        assert!(value.spin_density_cancellation_residual > 1e-6);
    }
}

#[test]
fn constrained_bare_unequal_channel_limit_and_g_scaling() {
    let (cell, options) = setup();
    let mut functional = ResolvedFunctional::hartree_fock();
    functional.hybrid = HybridParams::PURE;
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    let response = |g| {
        longitudinal_spin_susceptibility(
            &cell,
            &functional,
            &reference,
            &options,
            &ResponseSolverConfig::default(),
            reciprocal_cutoff(),
            g,
        )
        .unwrap()
    };
    let a = response(2.0);
    let b = response(1.3);
    assert!(a.converged && b.converged);
    assert!(
        (b.value.susceptibility - a.value.susceptibility * (1.3_f64 / 2.0).powi(2)).abs() < 1e-11
    );
    assert!(
        (b.value.chemical_potential_response.unwrap()
            - a.value.chemical_potential_response.unwrap() * 1.3 / 2.0)
            .abs()
            < 1e-11
    );
    assert!((a.value.bare_charge_compressibility - a.value.charge_compressibility).abs() > 1e-5);
    assert!(a.value.density_hermitian_residual < 1e-12);
}
