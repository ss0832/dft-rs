// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dense complex rotation, every spin torque and exact solver-budget controls.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::periodic::spinor_scf::from_pauli;
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::epr::spinor::rotation::optimal_global_spin_rotation as optimum;

use dft_rs::dfpt::orbitals::{adjoint, matmul};
use dft_rs::response::complex::Complex;

fn rotate(p: &BlochMatrix, q: [f64; 4]) -> BlochMatrix {
    let n = p.rows() / 2;
    let [w, x, y, z] = q;
    let values = [
        [Complex::new(w, -z), Complex::new(-y, -x)],
        [Complex::new(y, -x), Complex::new(w, z)],
    ];
    let mut u = BlochMatrix::zeros(2 * n, 2 * n);
    for s in 0..2 {
        for t in 0..2 {
            for i in 0..n {
                u.re[(s * n + i, t * n + i)] = values[s][t].re;
                u.im[(s * n + i, t * n + i)] = values[s][t].im;
            }
        }
    }
    matmul(&matmul(&u, p).unwrap(), &adjoint(&u).unwrap()).unwrap()
}

#[test]
fn full_complex_spin_rotation_and_all_three_torques_have_independent_oracles() {
    let mut p = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    p[0].re[(0, 0)] = 1.5;
    p[0].re[(1, 1)] = 1.5;
    for (s, im) in [0.03, 0.06, -0.02].into_iter().enumerate() {
        p[s + 1].re[(0, 0)] = 0.04 * (s + 1) as f64;
        p[s + 1].re[(1, 1)] = -0.02 * (s + 2) as f64;
        p[s + 1].im[(0, 1)] = im;
        p[s + 1].im[(1, 0)] = -im;
    }
    let density = from_pauli(&p).unwrap();
    let mut h = std::array::from_fn(|_| BlochMatrix::zeros(2, 2));
    for (s, im) in [0.7, -0.4, 1.3].into_iter().enumerate() {
        h[s + 1].im[(0, 1)] = im;
        h[s + 1].im[(1, 0)] = -im;
    }
    let operator = from_pauli(&h).unwrap();
    let result = optimum(&density, &operator).unwrap();
    let direct = rotate(&density, result.quaternion);
    assert!(direct.re.max_abs_diff(&result.density.re) < 2e-14);
    assert!(direct.im.max_abs_diff(&result.density.im) < 2e-14);
    assert!((direct.trace_product(&direct) - density.trace_product(&density)).abs() < 2e-14);
    let mut metric = BlochMatrix::zeros(2, 2);
    metric.re[(0, 0)] = 1.1;
    metric.re[(1, 1)] = 0.94;
    metric.re[(0, 1)] = 0.13;
    metric.re[(1, 0)] = 0.13;
    let metric = dft_rs::periodic::spinor_scf::spin_diagonal(&metric);
    assert!((direct.trace_product(&metric) - density.trace_product(&metric)).abs() < 1e-13);
    assert!(result.work <= 0.0);
    assert!(optimum(&result.density, &operator)
        .unwrap()
        .torque
        .iter()
        .all(|v| v.abs() < 1e-14));
    for a in 0..3 {
        let steps = [0.08, 0.04, 0.02, 0.01];
        let errors = steps.map(|angle: f64| {
            let energy = |sign: f64| {
                let mut q = [0.0; 4];
                q[0] = (angle / 2.0).cos();
                q[a + 1] = sign * (angle / 2.0).sin();
                rotate(&density, q).trace_product(&operator)
            };
            ((energy(1.0) - energy(-1.0)) / (2.0 * angle) - result.torque[a]).abs()
        });
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "Global spin torque{a}: steps={steps:?} errors={errors:?} p={} R2={}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    }
}

#[test]
fn zero_operator_and_invalid_inputs_preserve_the_rotation_contract() {
    let density = BlochMatrix::zeros(2, 2);
    let zero = BlochMatrix::zeros(2, 2);
    let result = optimum(&density, &zero).unwrap();
    assert_eq!(result.quaternion, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(result.work, 0.0);
    assert!(optimum(&density, &BlochMatrix::zeros(4, 4)).is_err());
    let mut invalid = density;
    invalid.re[(0, 0)] = f64::NAN;
    assert!(optimum(&invalid, &zero).is_err());
}

#[test]
fn the_recovery_solver_never_exceeds_the_requested_total_iteration_budget() {
    use dft_rs::periodic::spinor_scf::{SpinorEnergyBackend, SpinorFields, SpinorScfConfig};
    use dft_rs::properties::epr::spinor::rotation::run_epr_spinor_reference;
    struct Linear {
        overlap: Vec<BlochMatrix>,
        initial: Vec<BlochMatrix>,
        physical: BlochMatrix,
    }
    impl SpinorEnergyBackend for Linear {
        fn overlap(&self) -> &[BlochMatrix] {
            &self.overlap
        }
        fn initial_fock(&self) -> &[BlochMatrix] {
            &self.initial
        }
        fn electrons(&self) -> f64 {
            1.0
        }
        fn evaluate(&self, density: &[BlochMatrix]) -> dft_rs::Result<SpinorFields> {
            let energy = density[0].trace_product(&self.physical);
            let mut components = [0.0; 10];
            components[0] = energy;
            Ok(SpinorFields {
                fock: vec![self.physical.clone()],
                energy,
                energy_components: components,
                xc_component_energies: Vec::new(),
                nuclear_soc_energy: energy,
                operator_converged: true,
                coulomb_operator_defect: 0.0,
                xc_grid_number_defect: 0.0,
                xc_diagnostics: None,
            })
        }
    }
    let mut overlap = BlochMatrix::zeros(2, 2);
    overlap.re[(0, 0)] = 1.0;
    overlap.re[(1, 1)] = 1.0;
    let mut initial = BlochMatrix::zeros(2, 2);
    initial.re[(0, 0)] = -0.4;
    initial.re[(1, 1)] = 0.2;
    let mut physical = initial.clone();
    physical.re[(0, 1)] = 0.3;
    physical.re[(1, 0)] = 0.3;
    physical.im[(0, 1)] = 0.17;
    physical.im[(1, 0)] = -0.17;
    let backend = Linear {
        overlap: vec![overlap],
        initial: vec![initial],
        physical,
    };
    let f = dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("svwn5".into())).unwrap();
    let mut config = SpinorScfConfig {
        max_iter: 2,
        damping: 0.0,
        energy_tolerance: 1e-12,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-12,
        ..Default::default()
    };
    let short = run_epr_spinor_reference(&backend, &f, &config, &backend.physical).unwrap();
    assert_eq!(short.total_iterations, 2);
    assert!(!short.reference.converged);
    config.max_iter = 3;
    let converged = run_epr_spinor_reference(&backend, &f, &config, &backend.physical).unwrap();
    assert!(converged.reference.converged);
    assert_eq!(converged.total_iterations, 3);
}
