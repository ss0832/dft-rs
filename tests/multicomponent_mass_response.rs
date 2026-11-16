// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent normalization, translation and complete quantum-mass derivatives.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::math::Vec3;
use dft_rs::multicomponent::correlation::CorrelationModel;
use dft_rs::multicomponent::mass;
use dft_rs::multicomponent::{gaussian_particle_basis, MulticomponentOptions, QuantumParticle};
use dft_rs::response::ResponseSolverConfig;

fn fit(label: &str, steps: &[f64], values: &[f64], exact: f64) {
    let errors: Vec<_> = values.iter().map(|x| (x - exact).abs()).collect();
    assert!(errors.iter().all(|x| x.is_finite() && *x > 0.0));
    let f = convergence::error_power_fit(steps, &errors);
    println!("MASS_FIT {label} exact={exact:.16e} steps={steps:?} errors={errors:?} p={:.12} r2={:.12} C={:.12e} log_rmse={:.12e}",f.exponent,f.r_squared,f.prefactor,f.log_rmse);
    assert!(
        (f.exponent - 2.).abs() < 0.025 && f.r_squared > 0.9999,
        "{label}"
    );
}

#[test]
fn gaussian_normalization_mass_scaling_and_translation_are_independent_contracts() {
    use dft_rs::integrals::{dipole_matrices, kinetic_matrix, overlap_matrix};
    let alpha = 7.3;
    let m = 1836.15267343;
    for center in [[0.13, -0.21, 0.37], [1.81, -2.07, 0.61]] {
        let basis =
            gaussian_particle_basis(Vec3::new(center[0], center[1], center[2]), &[alpha], 0)
                .unwrap();
        let s = overlap_matrix(&basis);
        let t = kinetic_matrix(&basis);
        let d = dipole_matrices(&basis, Vec3::zero());
        assert!((s[(0, 0)] - 1.).abs() < 3e-15);
        assert!((t[(0, 0)] - 1.5 * alpha).abs() < 3e-14);
        for a in 0..3 {
            assert!((d[a][(0, 0)] - center[a]).abs() < 3e-15);
        }
        let steps = [0.08 * m, 0.04 * m, 0.02 * m, 0.01 * m];
        let values: Vec<_> = steps
            .iter()
            .map(|h| (t[(0, 0)] / (m + h) - t[(0, 0)] / (m - h)) / (2. * h))
            .collect();
        fit("Gaussian dE/dm", &steps, &values, -1.5 * alpha / m / m);
        let values: Vec<_> = steps
            .iter()
            .map(|h| (-t[(0, 0)] / (m + h).powi(2) + t[(0, 0)] / (m - h).powi(2)) / (2. * h))
            .collect();
        fit("Gaussian d2E/dm2", &steps, &values, 3. * alpha / m / m / m);
    }
}

#[test]
fn complete_two_mass_gradient_and_relaxed_hessian_fit_fresh_coupled_references() {
    let xyz = [
        [0.13, -0.21, 0.37],
        [1.61, 0.32, -0.18],
        [-0.47, 1.43, 0.61],
    ];
    let method = MolecularMethod {
        xc: Some("hf".into()),
        charge: 1,
        dispersion: "none".into(),
        conv_tol: 1e-14,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 500,
        grid_weight_cutoff: 0.,
        ..MolecularMethod::default()
    };
    let setup = prepare_molecular(&[1, 1, 1], &xyz, &method).unwrap();
    let particles: Vec<_> = (0..2)
        .map(|i| QuantumParticle {
            label: format!("isotope {i}"),
            charge: 1.,
            mass: 1836.15267343 * (i + 1) as f64,
            basis: gaussian_particle_basis(setup.molecule.atoms[i].position, &[4., 8.], 1).unwrap(),
            replaced_atom: Some(i),
            correlation_channel: None,
        })
        .collect();
    let correlation = CorrelationModel::none();
    let options = MulticomponentOptions {
        tolerance: 1e-12,
        ..MulticomponentOptions::default()
    };
    let config = ResponseSolverConfig::default();
    let solve = |ps: &[QuantumParticle]| {
        dft_rs::multicomponent::constraints::solve(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            None,
            ps,
            &correlation,
            &options,
            &xyz[..2],
            &config,
        )
        .unwrap()
    };
    let state = solve(&particles);
    let result = mass::response(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        None,
        &particles,
        &correlation,
        &state,
        &config,
    )
    .unwrap();
    assert_eq!(mass::gradient(&particles, &state).unwrap(), result.gradient);
    for invalid in [0., -1., f64::NAN, f64::INFINITY] {
        let mut other = particles.clone();
        other[0].mass = invalid;
        assert!(mass::gradient(&other, &state).is_err());
        assert!(mass::response(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            None,
            &other,
            &correlation,
            &state,
            &config
        )
        .is_err());
    }
    let mut changed = particles.clone();
    changed[0].mass *= 2.;
    assert!(mass::gradient(&changed, &state).is_err());
    changed = particles.clone();
    changed[0].basis =
        gaussian_particle_basis(setup.molecule.atoms[0].position, &[5., 9.], 1).unwrap();
    assert!(mass::gradient(&changed, &state).is_err());
    changed = particles.clone();
    changed[0].basis = gaussian_particle_basis(
        setup.molecule.atoms[0].position + Vec3::new(0.1, -0.2, 0.3),
        &[4., 8.],
        1,
    )
    .unwrap();
    assert!(mass::gradient(&changed, &state).is_err());
    let mut nonstationary = state.clone();
    nonstationary.converged = false;
    assert!(mass::gradient(&particles, &nonstationary).is_err());
    assert!(mass::gradient(&particles[..1], &state).is_err());
    println!("MASS_DIAGNOSTICS asymmetry={:.16e} number={:?} response_number={:?} kinetic={:?} residual={:?}",
        result.raw_asymmetry,result.reference_number_defects,result.response_number_defects,
        result.kinetic_energy_defects,result.responses.iter().map(|r|r.relative_residual).collect::<Vec<_>>());
    assert!(result.raw_asymmetry < 1e-12);
    assert!(result
        .reference_number_defects
        .iter()
        .chain(result.response_number_defects.iter().flatten())
        .all(|v| v.abs() < 1e-12));
    assert!(result
        .kinetic_energy_defects
        .iter()
        .all(|v| v.abs() < 1e-12));
    assert!(result
        .responses
        .iter()
        .all(|r| r.converged && r.relative_residual < 1e-12 && r.position_residual < 1e-12));
    for j in 0..2 {
        let m = particles[j].mass;
        let steps = [0.08 * m, 0.04 * m, 0.02 * m, 0.01 * m];
        let mut energies = Vec::new();
        let mut gradients = [Vec::new(), Vec::new()];
        for h in steps {
            let mut ps = particles.clone();
            ps[j].mass = m + h;
            let plus = solve(&ps);
            let gp: Vec<_> = plus
                .quantum
                .iter()
                .zip(&ps)
                .map(|(q, p)| -q.kinetic_energy / p.mass)
                .collect();
            ps[j].mass = m - h;
            let minus = solve(&ps);
            let gm: Vec<_> = minus
                .quantum
                .iter()
                .zip(&ps)
                .map(|(q, p)| -q.kinetic_energy / p.mass)
                .collect();
            energies.push((plus.energy - minus.energy) / (2. * h));
            for i in 0..2 {
                gradients[i].push((gp[i] - gm[i]) / (2. * h));
            }
        }
        fit(
            &format!("coupled gradient[{j}]"),
            &steps,
            &energies,
            result.gradient[j],
        );
        for i in 0..2 {
            fit(
                &format!("coupled Hessian[{i},{j}]"),
                &steps,
                &gradients[i],
                result.hessian[(i, j)],
            );
        }
    }
}
