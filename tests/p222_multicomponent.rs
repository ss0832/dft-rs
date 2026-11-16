// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent correlation derivatives and coupled non-BO energy identities.

#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::multicomponent::correlation::{
    CorrelationModel, Epc17, InterspeciesCorrelation, NoCorrelation,
};
use dft_rs::multicomponent::{
    gaussian_particle_basis, solve, MulticomponentOptions, QuantumParticle,
};

fn independent_energy(e: f64, p: f64) -> f64 {
    -1.0 / (2.35 / (e * p) - 2.4 / (e * p).sqrt() + 3.2)
}

#[test]
fn positive_epc_density_product_underflow_is_not_a_node() {
    for c in [3.2, 6.6] {
        let model = Epc17 { a: 2.35, b: 2.4, c };
        for (e, p) in [
            (0.25, f64::from_bits(1)),
            (f64::from_bits(1), 0.25),
            (1e-200, 1e-200),
            (1e-300, 1e-20),
        ] {
            let left = model.kernel_action(e, p, [1.0, 0.0]).unwrap();
            let right = model.kernel_action(e, p, [0.0, 1.0]).unwrap();
            assert!(left.iter().chain(&right).all(|v| v.is_finite()));
            assert_eq!(left[1].to_bits(), right[0].to_bits());
            // The closed small-product expansion has a finite diagonal
            // proportional to rho_other^(3/2)/sqrt(rho_self). All omitted
            // corrections here are below binary64 rounding resolution.
            let coefficient = -3.0 * model.b / (4.0 * model.a * model.a);
            let expected = [
                coefficient * p * (p.sqrt() / e.sqrt()),
                coefficient * e * (e.sqrt() / p.sqrt()),
            ];
            for (actual, reference) in [(left[0], expected[0]), (right[1], expected[1])] {
                if reference != 0.0 {
                    assert_ne!(actual, 0.0);
                    assert!(((actual - reference) / reference).abs() < 3e-15);
                } else {
                    assert_eq!(actual, 0.0);
                }
            }
        }
    }
}

fn independent_potentials(e: f64, p: f64) -> [f64; 2] {
    let x = e * p;
    let d = 2.35 - 2.4 * x.sqrt() + 3.2 * x;
    [
        -p / d + x * (3.2 * p - 1.2 * (p / e).sqrt()) / (d * d),
        -e / d + x * (3.2 * e - 1.2 * (e / p).sqrt()) / (d * d),
    ]
}

#[test]
fn mixed_epc_kernel_has_maxwell_symmetry_and_four_step_fits() {
    let model = Epc17::default();
    let steps = [0.04, 0.02, 0.01, 0.005];
    for (e, p) in [(0.2, 0.7), (1.3, 2.1), (0.02, 8.0)] {
        let direction = [0.2 * e, -0.3 * p];
        let analytic = model.kernel_action(e, p, direction).unwrap();
        let a = model.kernel_action(e, p, [1.0, 0.0]).unwrap();
        let b = model.kernel_action(e, p, [0.0, 1.0]).unwrap();
        assert!((a[1] - b[0]).abs() < 1e-12);
        let estimates: Vec<_> = steps
            .iter()
            .map(|h| {
                let a = independent_potentials(e + h * direction[0], p + h * direction[1]);
                let b = independent_potentials(e - h * direction[0], p - h * direction[1]);
                [(a[0] - b[0]) / (2.0 * h), (a[1] - b[1]) / (2.0 * h)]
            })
            .collect();
        for s in 0..2 {
            let errors: Vec<_> = estimates
                .iter()
                .map(|v| (v[s] - analytic[s]).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = ((4.0 * estimates[3][s] - estimates[2][s]) / 3.0 - analytic[s]).abs();
            println!("EPC17 kernel e={e} p={p} channel={s} analytic={:.14e} steps={steps:?} errors={errors:?} p={:.12} R2={:.12} C={:.12e} log_rmse={:.12e} Richardson={richardson:.12e}",analytic[s],fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse);
            assert!((fit.exponent - 2.0).abs() < 0.025 && fit.r_squared > 0.9999);
            assert!(richardson < 2e-7 + 1e-8 * analytic[s].abs());
        }
    }
    assert_eq!(
        model.kernel_action(1.0, 0.0, [0.1, 0.0]).unwrap(),
        [0.0, -0.1 / 2.35]
    );
    assert!(model.kernel_action(1.0, 0.0, [0.0, 0.1]).is_err());
}

#[test]
fn epc17_both_potentials_have_independent_four_step_error_fits() {
    let model = Epc17::default();
    for row in include_str!("oracle_data/multicomponent_epc17.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
    {
        let numbers: Vec<f64> = row.split(',').map(|v| v.parse().unwrap()).collect();
        let actual = model.evaluate(numbers[0], numbers[1]).unwrap();
        for (name, value, oracle) in [
            ("energy", actual.energy, numbers[2]),
            ("electron potential", actual.electron_potential, numbers[3]),
            ("proton potential", actual.partner_potential, numbers[4]),
        ] {
            let error = (value - oracle).abs();
            println!(
                "EPC17 high-precision {name} rho_e={} rho_p={} absolute_error={error:.12e}",
                numbers[0], numbers[1]
            );
            assert!(error < 2e-13 * (1.0 + oracle.abs()));
        }
    }
    for (e, p) in [(0.2, 0.7), (1.3, 2.1), (0.02, 8.0)] {
        let exact = model.evaluate(e, p).unwrap();
        assert!((exact.energy - independent_energy(e, p)).abs() < 2e-15);
        for (axis, analytic) in [exact.electron_potential, exact.partner_potential]
            .iter()
            .enumerate()
        {
            let scale = if axis == 0 { e } else { p };
            let steps = [scale * 0.04, scale * 0.02, scale * 0.01, scale * 0.005];
            let estimates: Vec<_> = steps
                .iter()
                .map(|h| {
                    if axis == 0 {
                        (independent_energy(e + h, p) - independent_energy(e - h, p)) / (2.0 * h)
                    } else {
                        (independent_energy(e, p + h) - independent_energy(e, p - h)) / (2.0 * h)
                    }
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|v| (v - analytic).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = (4.0 * estimates[3] - estimates[2]) / 3.0;
            println!("EPC17 e={e} p={p} axis={axis} analytic={analytic:.14e} steps={steps:?} errors={errors:?} p={:.12} R2={:.12} C={:.12e} log_rmse={:.12e} Richardson={:.12e}",fit.exponent,fit.r_squared,fit.prefactor,fit.log_rmse,(richardson-analytic).abs());
            assert!((fit.exponent - 2.0).abs() < 0.025);
            assert!(fit.r_squared > 0.9999);
            assert!((richardson - analytic).abs() < 2e-7 + 1e-8 * analytic.abs());
        }
    }
    for (e, p) in [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0)] {
        let x = model.evaluate(e, p).unwrap();
        assert_eq!(x.energy, 0.0);
        assert_eq!(x.electron_potential, -p / 2.35);
        assert_eq!(x.partner_potential, -e / 2.35);
    }
    assert!(model.evaluate(-0.1, 1.0).is_err());
    assert!(Epc17 {
        a: 1.0,
        b: 3.0,
        c: 1.0
    }
    .evaluate(1.0, 1.0)
    .is_err());
}

#[test]
fn coupled_hf_energy_matches_explicit_mixed_ao_tensor_terms() {
    let method = MolecularMethod {
        xc: Some("hf".into()),
        dispersion: "none".into(),
        conv_tol: 1e-13,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 500,
        ..MolecularMethod::default()
    };
    let setup = prepare_molecular(&[1, 1], &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], &method).unwrap();
    let q = QuantumParticle {
        label: "quantum proton".into(),
        charge: 1.0,
        mass: dft_rs::constants::PROTON_ELECTRON_MASS_RATIO,
        basis: gaussian_particle_basis(setup.molecule.atoms[1].position, &[4.0, 8.0, 16.0], 1)
            .unwrap(),
        replaced_atom: Some(1),
        correlation_channel: Some(0),
    };
    let result = solve(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        None,
        std::slice::from_ref(&q),
        &CorrelationModel::single(Box::new(NoCorrelation)),
        &MulticomponentOptions {
            tolerance: 1e-10,
            ..MulticomponentOptions::default()
        },
    )
    .unwrap();
    println!("coupled HF energy={:.14e} iterations={} residuals electron={:.12e}/{:.12e} proton={:.12e}/{:.12e} variance={:?}",result.energy,result.iterations,result.electronic_orbital_residual,result.electronic_density_residual,result.quantum[0].orbital_residual,result.quantum[0].density_residual,result.quantum[0].position_variance);
    assert!(result.converged);
    assert!((result.quantum[0].particle_number - 1.0).abs() < 1e-12);
    assert!(result.quantum[0].position_variance.iter().all(|v| *v > 0.0));
    let mut combined = setup.basis.clone();
    let ne = combined.nao_sph;
    for original in &q.basis.shells {
        let mut shell = original.clone();
        shell.sph_off += ne;
        shell.cart_off += combined.nao_cart;
        combined.shells.push(shell);
    }
    combined.nao_sph += q.basis.nao_sph;
    combined.nao_cart += q.basis.nao_cart;
    combined.max_l = combined.max_l.max(q.basis.max_l);
    let tensor = eri_incore(&combined, CoulombKernel::Coulomb, 256).unwrap();
    let p = &result.electronic_density;
    let mut total = p[0].clone();
    for (a, b) in total.as_mut_slice().iter_mut().zip(p[1].as_slice()) {
        *a += b;
    }
    let mut j = 0.0;
    let mut k = 0.0;
    let mut ep = 0.0;
    for i in 0..ne {
        for j0 in 0..ne {
            for k0 in 0..ne {
                for l in 0..ne {
                    j += 0.5 * total[(i, j0)] * total[(k0, l)] * tensor.get(i, j0, k0, l);
                    for spin in p {
                        k -= 0.5 * spin[(i, k0)] * spin[(j0, l)] * tensor.get(i, j0, k0, l);
                    }
                }
            }
            for a in 0..q.basis.nao_sph {
                for b in 0..q.basis.nao_sph {
                    ep -= total[(i, j0)]
                        * result.quantum[0].density[(a, b)]
                        * tensor.get(i, j0, ne + a, ne + b);
                }
            }
        }
    }
    let mut h = kinetic_matrix(&setup.basis);
    let v = nuclear_matrix(&setup.basis, &setup.molecule, Some(&[1.0, 0.0]));
    for (a, b) in h.as_mut_slice().iter_mut().zip(v.as_slice()) {
        *a += b;
    }
    let one = total.frobenius_dot(&h);
    let qt = result.quantum[0]
        .density
        .frobenius_dot(&kinetic_matrix(&q.basis))
        / q.mass;
    let qv = -result.quantum[0].density.frobenius_dot(&nuclear_matrix(
        &q.basis,
        &setup.molecule,
        Some(&[1.0, 0.0]),
    ));
    for (name, a, b) in [
        ("electronic one body", one, result.electronic_one_body),
        ("electronic J", j, result.electronic_coulomb),
        ("electronic K", k, result.electronic_exchange_correlation),
        ("electron proton", ep, result.electron_quantum_coulomb),
        ("proton kinetic", qt, result.quantum[0].kinetic_energy),
        ("proton classical", qv, result.quantum[0].classical_energy),
    ] {
        println!(
            "coupled term {name} analytic={b:.14e} independent={a:.14e} error={:.12e}",
            (a - b).abs()
        );
        assert!((a - b).abs() < 2e-11);
    }
    assert!((result.energy - (one + j + k + ep + qt + qv)).abs() < 2e-11);
}

#[test]
fn pbe_epc17_is_jointly_stationary_and_preserves_the_energy_partition() {
    let method = MolecularMethod {
        xc: Some("pbe".into()),
        dispersion: "none".into(),
        grid_level: 3,
        conv_tol: 1e-13,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 500,
        ..MolecularMethod::default()
    };
    let setup = prepare_molecular(
        &[8, 1, 1],
        &[[0.0, 0.0, 0.0], [0.0, -1.43, 1.10], [0.0, 1.43, 1.10]],
        &method,
    )
    .unwrap();
    let q = QuantumParticle {
        label: "quantum proton".into(),
        charge: 1.0,
        mass: dft_rs::constants::PROTON_ELECTRON_MASS_RATIO,
        basis: gaussian_particle_basis(setup.molecule.atoms[1].position, &[4.0, 8.0, 16.0], 1)
            .unwrap(),
        replaced_atom: Some(1),
        correlation_channel: Some(0),
    };
    let result = solve(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        &[q],
        &CorrelationModel::single(Box::new(Epc17::default())),
        &MulticomponentOptions::default(),
    )
    .unwrap();
    println!("water PBE/EPC17 energy={:.14e} iterations={} electron_residuals={:.12e}/{:.12e} proton_residuals={:.12e}/{:.12e} epc={:.12e}",result.energy,result.iterations,result.electronic_orbital_residual,result.electronic_density_residual,result.quantum[0].orbital_residual,result.quantum[0].density_residual,result.interspecies_correlation);
    assert!(result.converged);
    assert!(result.interspecies_correlation < 0.0);
    let components = result.electronic_one_body
        + result.electronic_coulomb
        + result.electronic_exchange_correlation
        + result.electronic_nonlocal_correlation
        + result.electron_quantum_coulomb
        + result.quantum_pair_coulomb
        + result.interspecies_correlation
        + result.classical_energy
        + result
            .quantum
            .iter()
            .map(|q| q.kinetic_energy + q.classical_energy)
            .sum::<f64>();
    assert!((components - result.energy).abs() < 1e-11);
}

#[test]
fn supplied_first_derivative_provider_does_not_require_an_optimization_kernel() {
    struct Contact;
    impl InterspeciesCorrelation for Contact {
        fn description(&self) -> &str {
            "test-only local contact functional"
        }
        fn evaluate(
            &self,
            e: f64,
            p: f64,
        ) -> dft_rs::Result<dft_rs::multicomponent::correlation::CorrelationPoint> {
            Ok(dft_rs::multicomponent::correlation::CorrelationPoint {
                energy: -1e-4 * e * p,
                electron_potential: -1e-4 * p,
                partner_potential: -1e-4 * e,
            })
        }
    }
    let method = MolecularMethod {
        xc: Some("hf".into()),
        dispersion: "none".into(),
        grid_level: 2,
        conv_tol: 1e-13,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 500,
        ..MolecularMethod::default()
    };
    let setup = prepare_molecular(&[1, 1], &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], &method).unwrap();
    let grid = dft_rs::grid::MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap();
    let q = QuantumParticle {
        label: "quantum proton".into(),
        charge: 1.0,
        mass: dft_rs::constants::PROTON_ELECTRON_MASS_RATIO,
        basis: gaussian_particle_basis(setup.molecule.atoms[1].position, &[4.0, 8.0], 1).unwrap(),
        replaced_atom: Some(1),
        correlation_channel: Some(0),
    };
    let state = solve(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        Some(&grid),
        std::slice::from_ref(&q),
        &CorrelationModel::single(Box::new(Contact)),
        &MulticomponentOptions::default(),
    )
    .unwrap();
    assert!(state.converged && state.interspecies_correlation < 0.0);
    assert!(Contact.kernel_action(0.2, 0.7, [0.1, -0.2]).is_err());
    let ephi = dft_rs::grid::eval::eval_aos(&setup.basis, &grid.points, false).phi;
    let qphi = dft_rs::grid::eval::eval_aos(&q.basis, &grid.points, false).phi;
    let density = |phi: &Matrix, p: &Matrix, g: usize| -> f64 {
        (0..p.rows)
            .flat_map(|i| (0..p.cols).map(move |j| phi[(g, i)] * p[(i, j)] * phi[(g, j)]))
            .sum()
    };
    let independent: f64 = (0..grid.weights.len())
        .map(|g| {
            -1e-4
                * grid.weights[g]
                * (density(&ephi, &state.electronic_density[0], g)
                    + density(&ephi, &state.electronic_density[1], g))
                * density(&qphi, &state.quantum[0].density, g)
        })
        .sum();
    assert!((independent - state.interspecies_correlation).abs() < 1e-13);
}

#[test]
fn epc17_pointwise_wave_node_has_only_c2_regularity() -> dft_rs::Result<()> {
    // This is a scalar regularity control, not a relaxed molecular tensor gate.
    let provider = Epc17::default();
    let gradient = |wave: f64| -> dft_rs::Result<f64> {
        Ok(2.0 * wave * provider.evaluate(1.0, wave * wave)?.partner_potential)
    };
    for (wave, steps, expected_order) in [
        (0.0, [0.002, 0.001, 0.0005, 0.00025], 1.0),
        (0.003, [0.001, 0.0005, 0.00025, 0.000125], 2.0),
    ] {
        let value = provider.evaluate(1.0, wave * wave)?;
        let kernel = provider.kernel_action(1.0, wave * wave, [0.0, 2.0 * wave])?;
        let analytic = 2.0 * value.partner_potential + 2.0 * wave * kernel[1];
        let (a, b, c) = (provider.a, provider.b, provider.c);
        let t = wave.abs();
        let independent = (-2.0 * a * a + 6.0 * a * c * t * t - 2.0 * b * c * t * t * t)
            / (a - b * t + c * t * t).powi(3);
        assert!((analytic - independent).abs() < 1e-12);
        let mut errors = [0.0; 4];
        for (slot, step) in steps.iter().enumerate() {
            let estimate = (gradient(wave + step)? - gradient(wave - step)?) / (2.0 * step);
            errors[slot] = (estimate - analytic).abs();
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "EPC17 pointwise wave={wave} expected_order={expected_order} analytic={analytic} \
             steps={steps:?} errors={errors:?} p={} R2={} C={} log_RMSE={}",
            fit.exponent, fit.r_squared, fit.prefactor, fit.log_rmse
        );
        assert!((fit.exponent - expected_order).abs() < 0.025);
        assert!(fit.r_squared > 0.9999);
        if wave == 0.0 {
            assert!((fit.exponent - 2.0).abs() > 0.9);
        }
    }
    Ok(())
}

#[test]
fn epc17_nodal_branch_identity_preserves_second_jet() -> dft_rs::Result<()> {
    // This postprocessing identity must not replace a physical energy or
    // silently rescue a failed molecular finite-difference interval.
    let provider = Epc17::default();
    let mut maximum = 0.0_f64;
    for ne in [0.3_f64, 1.0, 2.0] {
        for sign in [-1.0, 1.0] {
            for wave in [-0.02_f64, 0.013, 0.0] {
                let u = ne.sqrt() * wave;
                let (a, b, c) = (provider.a, provider.b, provider.c);
                let actual = a - b * u.abs() + c * u * u;
                let branch = a - b * sign * u + c * u * u;
                let stable = -b * (u * u) * (u.abs() - sign * u) / (actual * branch);
                let direct = provider.evaluate(ne, wave * wave)?.energy + u * u / branch;
                maximum = maximum.max((stable - direct).abs());
                assert!((stable - direct).abs() < 1e-15);
            }
        }
    }
    // The piecewise rational remainder has a cubic numerator and a strictly
    // positive even denominator: its value, gradient and Hessian vanish at zero.
    let steps: [f64; 4] = [0.002, 0.001, 0.0005, 0.00025];
    let magnitudes = steps.map(|h| {
        let u = -h;
        let (a, b, c) = (provider.a, provider.b, provider.c);
        (-b * (u * u) * (u.abs() - u) / ((a - b * u.abs() + c * u * u) * (a - b * u + c * u * u)))
            .abs()
    });
    let fit = convergence::error_power_fit(&steps, &magnitudes);
    println!(
        "EPC17 zero-second-jet steps={steps:?} magnitudes={magnitudes:?} \
         p={} R2={} C={} log_RMSE={} direct_identity_defect={maximum}",
        fit.exponent, fit.r_squared, fit.prefactor, fit.log_rmse
    );
    assert!((fit.exponent - 3.0).abs() < 0.025);
    assert!(fit.r_squared > 0.9999);
    Ok(())
}
