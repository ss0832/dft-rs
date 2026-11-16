// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Hyperfine tensor oracles, unprojected Poisson identities and derivative fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::properties::hyperfine::{spin_dipole_operators, spin_hyperfine_tensor};
use dft_rs::system::{Atom, Molecule};
use std::f64::consts::PI;

fn basis(exponent: f64) -> MolBasis {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mut set = BasisSet::new("hyperfine-gaussian-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![exponent],
            coefs: vec![1.0],
        }],
    });
    MolBasis::build(&mol, &set).unwrap()
}
fn potential_gradient(exponent: f64, center: Vec3) -> [f64; 3] {
    let r = center.norm();
    let p = 2.0 * exponent;
    let radial = 2.0 * p.sqrt() / PI.sqrt() * (-p * r * r).exp() / (r * r)
        - libm::erf(p.sqrt() * r) / (r * r * r);
    center.to_array().map(|v| radial * v)
}
fn tensor_error(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (a[i][j] - b[i][j]).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn dipolar_operator_matches_closed_gaussian_and_four_step_potential_gradient_fit() {
    let center = Vec3::new(0.6, -0.4, 0.5);
    let direction = Vec3::new(0.31, 0.27, -0.19);
    for exponent in [0.4, 0.7, 1.2] {
        let mb = basis(exponent);
        let operator = spin_dipole_operators(&mb, center).unwrap();
        let r = center.norm();
        let p = 2.0 * exponent;
        let density = (p / PI).powf(1.5) * (-p * r * r).exp();
        let f = libm::erf(p.sqrt() * r);
        let e = (-p * r * r).exp();
        let transverse = 2.0 * p.sqrt() / PI.sqrt() * e / (r * r) - f / r.powi(3);
        let anisotropy = 3.0 * f / r.powi(3)
            - 6.0 * p.sqrt() / PI.sqrt() * e / (r * r)
            - 4.0 * p.powf(1.5) / PI.sqrt() * e;
        let mut closed = [[0.0; 3]; 3];
        let mut hessian = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                hessian[a][b] = operator.dipolar[3 * a + b][(0, 0)]
                    - if a == b {
                        4.0 * PI / 3.0 * operator.contact[(0, 0)]
                    } else {
                        0.0
                    };
                closed[a][b] = anisotropy * center.get(a) * center.get(b) / (r * r)
                    + if a == b {
                        transverse + 4.0 * PI / 3.0 * density
                    } else {
                        0.0
                    };
            }
        }
        let actual =
            std::array::from_fn(|a| std::array::from_fn(|b| operator.dipolar[3 * a + b][(0, 0)]));
        let oracle = tensor_error(&actual, &closed);
        assert!(
            oracle < 1e-12
                && operator.poisson_residual < 1e-12
                && operator.hermiticity_residual < 1e-12
        );
        let analytic: [f64; 3] =
            std::array::from_fn(|a| (0..3).map(|b| hessian[a][b] * direction.get(b)).sum());
        let steps = [0.08, 0.04, 0.02, 0.01];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let plus = potential_gradient(exponent, center + direction * h);
            let minus = potential_gradient(exponent, center - direction * h);
            let estimate: [f64; 3] = std::array::from_fn(|a| (plus[a] - minus[a]) / (2.0 * h));
            errors.push(
                (0..3)
                    .map(|a| (estimate[a] - analytic[a]).abs())
                    .fold(0.0, f64::max),
            );
            estimates.push(estimate);
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = (0..3)
            .map(|a| ((4.0 * estimates[3][a] - estimates[2][a]) / 3.0 - analytic[a]).abs())
            .fold(0.0, f64::max);
        println!("hyperfine Gaussian exponent={exponent}: oracle={oracle:.3e} Poisson={:.3e} steps={steps:?} errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e}",operator.poisson_residual,fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!(rich < 1e-8);
    }
}

#[test]
fn complete_hyperfine_tensor_has_physical_principal_axes_and_signed_isotope_scaling() {
    let mb = basis(0.7);
    let mut spin = Matrix::zeros(1, 1);
    spin[(0, 0)] = 1.0;
    let reference = spin_hyperfine_tensor(&mb, &spin, Vec3::new(0.6, -0.4, 0.5), 1.0).unwrap();
    assert!(reference.dipolar.iter().flatten().any(|v| v.abs() > 1.0));
    for g in [5.5856946893, -0.566377, 0.0] {
        let actual = reference.scaled(g).unwrap();
        assert!(actual.principal_values.windows(2).all(|v| v[0] <= v[1]));
        let expected = reference.total.map(|row| row.map(|v| g * v));
        assert!(tensor_error(&actual.total, &expected) < 1e-10);
        assert!(actual.eigen_residual < 1e-9 && actual.axes_orthogonality_residual < 1e-12);
        assert!(actual.symmetry_residual < 1e-10 && actual.dipolar_trace.abs() < 1e-10);
        println!("hyperfine g={g}: principal={:?} raw trace={:.3e} asymmetry={:.3e} eigen={:.3e} axes={:.3e}",actual.principal_values,actual.dipolar_trace,actual.symmetry_residual,actual.eigen_residual,actual.axes_orthogonality_residual);
    }
    spin[(0, 0)] = -1.0;
    let flipped = spin_hyperfine_tensor(&mb, &spin, Vec3::new(0.6, -0.4, 0.5), -1.0).unwrap();
    assert!(tensor_error(&reference.total, &flipped.total) < 1e-10);
    let atom = spin_hyperfine_tensor(&mb, &spin, Vec3::zero(), -1.0).unwrap();
    assert!(atom.dipolar.iter().flatten().all(|v| v.abs() < 1e-9));
    assert!(spin_hyperfine_tensor(&mb, &spin, Vec3::zero(), 0.0).is_err());
}
