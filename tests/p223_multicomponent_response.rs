// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent displaced-SCF acceptance of coupled non-BO response.
//! Retained TPSS UKS directional Hessian after the shared SCF and inward
//! iso-orbital endpoint corrections: H=0.193361970012764 Hartree/Bohr^2,
//! h=[0.04,0.02,0.01,0.005] Bohr, p=1.997884081962, R2=0.999999698518,
//! Richardson defect 5.532513308593e-9. All six regression cases pass.
//! Directional fits do not certify a complete coupled Cartesian tensor.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::integrals::dipole_matrices;
use dft_rs::math::Vec3;
use dft_rs::multicomponent::correlation::{CorrelationModel, Epc17, NoCorrelation};
use dft_rs::multicomponent::derivatives;
use dft_rs::multicomponent::response::CoupledResponseSolver;
use dft_rs::multicomponent::{
    gaussian_particle_basis, solve, MulticomponentOptions, QuantumParticle,
};
use dft_rs::response::ResponseSolverConfig;

fn method() -> MolecularMethod {
    MolecularMethod {
        xc: Some("hf".into()),
        dispersion: "none".into(),
        conv_tol: 1e-14,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 500,
        grid_weight_cutoff: 0.0,
        ..MolecularMethod::default()
    }
}

fn accept(label: &str, steps: &[f64], estimates: &[f64], analytic: f64) {
    let errors: Vec<_> = estimates.iter().map(|v| (v - analytic).abs()).collect();
    let f = convergence::error_power_fit(steps, &errors);
    let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
    println!("coupled {label} analytic={analytic:.14e} steps={steps:?} errors={errors:?} p={:.12} R2={:.12} C={:.12e} log_rmse={:.12e} Richardson={richardson:.12e}",f.exponent,f.r_squared,f.prefactor,f.log_rmse);
    assert!((f.exponent - 2.0).abs() < 0.025 && f.r_squared > 0.9999);
    assert!(richardson < 2e-7 + 1e-8 * analytic.abs());
}

fn particle(position: Vec3) -> QuantumParticle {
    QuantumParticle {
        label: "quantum proton".into(),
        charge: 1.0,
        mass: dft_rs::constants::PROTON_ELECTRON_MASS_RATIO,
        basis: gaussian_particle_basis(position, &[4.0, 8.0], 1).unwrap(),
        replaced_atom: Some(1),
        correlation_channel: Some(0),
    }
}

#[test]
fn coupled_polarizability_matches_four_steps_of_jointly_reoptimized_dipoles() {
    let uncorrelated = CorrelationModel::single(Box::new(NoCorrelation));
    let mut m = method();
    m.field = Some(Vec3::new(0.0, 0.0, 0.01));
    let setup = prepare_molecular(&[1, 1], &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], &m).unwrap();
    let particles = [particle(setup.molecule.atoms[1].position)];
    let options = MulticomponentOptions {
        tolerance: 1e-12,
        ..MulticomponentOptions::default()
    };
    let state = solve(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        None,
        &particles,
        &uncorrelated,
        &options,
    )
    .unwrap();
    let config = ResponseSolverConfig::default();
    let response = CoupledResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        None,
        &particles,
        &uncorrelated,
        &state,
    )
    .unwrap();
    let alpha = response.polarizability(&config).unwrap();
    assert!(alpha.responses.iter().all(|r| r.converged));
    let n = response.dimension();
    let mut matrix = dft_rs::linalg::Matrix::zeros(n, n);
    for j in 0..n {
        let mut v = vec![0.0; n];
        v[j] = 1.0;
        let image = response.apply_real(&v).unwrap();
        for i in 0..n {
            matrix[(i, j)] = image[i];
        }
    }
    let asymmetry = matrix.max_abs_diff(&matrix.transpose());
    println!("coupled Hessian dimension={n} raw_asymmetry={asymmetry:.12e} alpha_asymmetry={:.12e} residuals={:?}",alpha.raw_asymmetry,alpha.responses.iter().map(|r|r.relative_residual).collect::<Vec<_>>());
    assert!(asymmetry < 1e-11 && alpha.raw_asymmetry < 1e-9);
    let dipole = |field: f64| {
        let mut opts = setup.options.clone();
        opts.external_field = Some(Vec3::new(0.0, 0.0, field));
        let s = solve(
            &setup.molecule,
            &setup.basis,
            &opts,
            &setup.functional,
            None,
            &particles,
            &uncorrelated,
            &options,
        )
        .unwrap();
        assert!(s.converged);
        let r = dipole_matrices(&setup.basis, Vec3::zero());
        setup.molecule.atoms[0].position.z
            - s.electronic_density
                .iter()
                .map(|p| p.frobenius_dot(&r[2]))
                .sum::<f64>()
            + s.quantum[0].position[2]
    };
    let steps = [0.004, 0.002, 0.001, 0.0005];
    let estimates: Vec<_> = steps
        .iter()
        .map(|h| (dipole(0.01 + h) - dipole(0.01 - h)) / (2.0 * h))
        .collect();
    accept(
        "HF total polarizability zz",
        &steps,
        &estimates,
        alpha.tensor[2][2],
    );
}

fn geometry_fit(xc: &str, unrestricted: bool, open_shell: bool) {
    let uncorrelated = CorrelationModel::single(Box::new(NoCorrelation));
    let mut m = method();
    m.xc = Some(xc.into());
    m.unrestricted = unrestricted;
    m.grid_level = 2;
    if open_shell {
        m.charge = 1;
        m.multiplicity = 2;
    }
    let numbers = if open_shell { [3, 1] } else { [1, 1] };
    let bond = if open_shell { 3.0 } else { 1.4 };
    let at = |displacement: f64| {
        let s = prepare_molecular(
            &numbers,
            &[
                [0.0, 0.0, -0.5 * bond + displacement],
                [0.0, 0.0, 0.5 * bond],
            ],
            &m,
        )
        .unwrap();
        let q = [particle(s.molecule.atoms[1].position)];
        let r = solve(
            &s.molecule,
            &s.basis,
            &s.options,
            &s.functional,
            s.grid.as_ref(),
            &q,
            &uncorrelated,
            &MulticomponentOptions {
                tolerance: 1e-12,
                ..MulticomponentOptions::default()
            },
        )
        .unwrap();
        assert!(r.converged);
        (s, q, r)
    };
    let (s, q, r) = at(0.0);
    let d = derivatives::directional(
        &s.molecule,
        &s.basis,
        &s.options,
        &s.functional,
        s.grid.as_ref(),
        &q,
        &uncorrelated,
        &r,
        &[[0.0, 0.0, 1.0], [0.0; 3]],
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    println!("coupled geometry gradient={:.14e} Hessian={:.14e} explicit={:.14e} relaxation={:.14e} response={:.12e} reference_error={:.12e} converged={}",d.gradient,d.hessian,d.explicit_hessian,d.orbital_relaxation,d.response_residual,d.reference_energy_error,d.converged);
    assert!(d.converged);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut g = Vec::new();
    let mut hessian = Vec::new();
    for h in steps {
        let ep = at(h).2.energy;
        let em = at(-h).2.energy;
        g.push((ep - em) / (2.0 * h));
        hessian.push((ep - 2.0 * r.energy + em) / (h * h));
    }
    accept(
        &format!("{xc} unrestricted={unrestricted} open_shell={open_shell} geometry gradient"),
        &steps,
        &g,
        d.gradient,
    );
    accept(
        &format!("{xc} unrestricted={unrestricted} open_shell={open_shell} geometry Hessian"),
        &steps,
        &hessian,
        d.hessian,
    );
}

#[test]
fn analytic_quantum_geometry_gradient_and_hessian_fit_independent_energies() {
    geometry_fit("hf", false, false);
}

#[test]
fn parent_functional_and_unrestricted_quantum_geometry_share_the_complete_kernel() {
    for (xc, unrestricted) in [
        ("hf", true),
        ("tpss", true),
        ("pbe0", false),
        ("cam-b3lyp", true),
    ] {
        geometry_fit(xc, unrestricted, false);
    }
    geometry_fit("hf", true, true);
}

#[test]
fn coupled_epc_geometry_includes_both_density_endpoints_and_grid_motion() {
    let mut m = method();
    m.xc = Some("pbe".into());
    m.grid_level = 2;
    let correlation = CorrelationModel::single(Box::new(Epc17::default()));
    let at = |displacement: f64| {
        let s = prepare_molecular(
            &[1, 1],
            &[[0.0, 0.0, -0.7 + displacement], [0.0, 0.0, 0.7]],
            &m,
        )
        .unwrap();
        let q = [particle(s.molecule.atoms[1].position)];
        let r = solve(
            &s.molecule,
            &s.basis,
            &s.options,
            &s.functional,
            s.grid.as_ref(),
            &q,
            &correlation,
            &MulticomponentOptions {
                tolerance: 1e-12,
                ..MulticomponentOptions::default()
            },
        )
        .unwrap();
        assert!(r.converged);
        (s, q, r)
    };
    let (s, q, r) = at(0.0);
    let d = derivatives::directional(
        &s.molecule,
        &s.basis,
        &s.options,
        &s.functional,
        s.grid.as_ref(),
        &q,
        &correlation,
        &r,
        &[[0.0, 0.0, 1.0], [0.0; 3]],
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    println!("coupled EPC geometry gradient={:.14e} Hessian={:.14e} response={:.12e} reference_error={:.12e} converged={}",d.gradient,d.hessian,d.response_residual,d.reference_energy_error,d.converged);
    assert!(d.converged);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut g = Vec::new();
    let mut hessian = Vec::new();
    for h in steps {
        let ep = at(h).2.energy;
        let em = at(-h).2.energy;
        g.push((ep - em) / (2.0 * h));
        hessian.push((ep - 2.0 * r.energy + em) / (h * h));
    }
    accept("PBE EPC geometry gradient", &steps, &g, d.gradient);
    accept("PBE EPC geometry Hessian", &steps, &hessian, d.hessian);
    assert!(correlation.channels()[0]
        .kernel_action(0.2, 0.7, [0.1, -0.1])
        .is_ok());
}

#[test]
fn expectation_position_gradient_and_hessian_include_constraint_response() {
    let uncorrelated = CorrelationModel::single(Box::new(NoCorrelation));
    let m = method();
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..ResponseSolverConfig::default()
    };
    let options = MulticomponentOptions {
        tolerance: 1e-12,
        ..MulticomponentOptions::default()
    };
    let at = |displacement: f64| {
        let s = prepare_molecular(
            &[1, 1],
            &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7 + displacement]],
            &m,
        )
        .unwrap();
        let q = [particle(s.molecule.atoms[1].position)];
        let targets = [s.molecule.atoms[1].position.to_array()];
        let r = dft_rs::multicomponent::constraints::solve(
            &s.molecule,
            &s.basis,
            &s.options,
            &s.functional,
            None,
            &q,
            &uncorrelated,
            &options,
            &targets,
            &config,
        )
        .unwrap();
        assert!(r.converged && r.position_constraints.as_ref().unwrap().residual < 1e-12);
        (s, q, r)
    };
    let (s, q, r) = at(0.0);
    let d = derivatives::directional(
        &s.molecule,
        &s.basis,
        &s.options,
        &s.functional,
        None,
        &q,
        &uncorrelated,
        &r,
        &[[0.0; 3], [0.0, 0.0, 1.0]],
        &config,
    )
    .unwrap();
    println!("constrained geometry gradient={:.14e} Hessian={:.14e} multiplier={:?} equation_residual={:.12e} constraint_response={:.12e} reference_error={:.12e}",d.gradient,d.hessian,r.position_constraints.as_ref().unwrap().multipliers,d.response_residual,d.position_response_residual,d.reference_energy_error);
    assert!(d.converged && d.position_response_residual < 1e-10);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut gradients = Vec::new();
    let mut hessians = Vec::new();
    for h in steps {
        let ep = at(h).2.energy;
        let em = at(-h).2.energy;
        gradients.push((ep - em) / (2.0 * h));
        hessians.push((ep - 2.0 * r.energy + em) / (h * h));
    }
    accept(
        "constrained nuclear mean-position gradient",
        &steps,
        &gradients,
        d.gradient,
    );
    accept(
        "constrained nuclear mean-position Hessian",
        &steps,
        &hessians,
        d.hessian,
    );
}

#[test]
fn all_quantum_isotope_pair_has_analytic_internal_stretch_derivatives() {
    let uncorrelated = CorrelationModel::single(Box::new(NoCorrelation));
    let m = method();
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..ResponseSolverConfig::default()
    };
    let options = MulticomponentOptions {
        tolerance: 1e-12,
        ..MulticomponentOptions::default()
    };
    // Relative separation changes at fixed nuclear center of mass. This
    // chart does not subtract center-of-mass kinetic energy from the model.
    let direction = [[0.0, 0.0, -2.0 / 3.0], [0.0, 0.0, 1.0 / 3.0]];
    let at = |t: f64| {
        let s = prepare_molecular(
            &[1, 1],
            &[[0.0, 0.0, -0.7 - 2.0 * t / 3.0], [0.0, 0.0, 0.7 + t / 3.0]],
            &m,
        )
        .unwrap();
        let mut q = [
            particle(s.molecule.atoms[0].position),
            particle(s.molecule.atoms[1].position),
        ];
        q[0].replaced_atom = Some(0);
        q[1].mass *= 2.0;
        let targets: Vec<_> = s
            .molecule
            .atoms
            .iter()
            .map(|a| a.position.to_array())
            .collect();
        let r = dft_rs::multicomponent::constraints::solve(
            &s.molecule,
            &s.basis,
            &s.options,
            &s.functional,
            None,
            &q,
            &uncorrelated,
            &options,
            &targets,
            &config,
        )
        .unwrap();
        assert!(r.converged && r.position_constraints.as_ref().unwrap().residual < 1e-12);
        assert_eq!(r.classical_energy, 0.0);
        assert!(r.quantum_pair_coulomb > 0.0);
        (s, q, r)
    };
    let (s, q, r) = at(0.0);
    let d = derivatives::directional(
        &s.molecule,
        &s.basis,
        &s.options,
        &s.functional,
        None,
        &q,
        &uncorrelated,
        &r,
        &direction,
        &config,
    )
    .unwrap();
    println!("all-quantum isotope pair E={:.14e} gradient={:.14e} Hessian={:.14e} response={:.12e} constraint_response={:.12e}",r.energy,d.gradient,d.hessian,d.response_residual,d.position_response_residual);
    assert!(d.converged);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut gradients = Vec::new();
    let mut hessians = Vec::new();
    for h in steps {
        let ep = at(h).2.energy;
        let em = at(-h).2.energy;
        gradients.push((ep - em) / (2.0 * h));
        hessians.push((ep - 2.0 * r.energy + em) / (h * h));
    }
    accept(
        "all-quantum internal stretch gradient",
        &steps,
        &gradients,
        d.gradient,
    );
    accept(
        "all-quantum internal stretch Hessian",
        &steps,
        &hessians,
        d.hessian,
    );
}
