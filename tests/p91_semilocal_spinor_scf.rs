// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Actual semilocal spinor SCF and independent field/free-energy fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use dft_rs::periodic::scf_boundary::DimensionalScfOptions;
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::uniform_open::UniformOpenField;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn basis(mol: &Molecule) -> MolBasis {
    let mut set = BasisSet::new("spinor-semilocal-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    MolBasis::build(mol, &set).unwrap()
}

#[test]
fn all_semilocal_rungs_have_self_consistent_open_field_four_step_fits() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol);
    for axes in [[true, false, false], [true, true, false]] {
        let geometry =
            PeriodicGeometry::new(Lattice::cubic(9.0), PeriodicBoundary::new(axes)).unwrap();
        let mut options = DimensionalScfOptions::for_boundary(geometry.boundary());
        options.basis_cutoff = 6.0;
        options.grid_cutoff = 10.0;
        options.scf.exchange_cutoff = Some(3.0);
        options.scf.grid.radial_override = Some(96);
        options.scf.grid.angular_override = Some(35);
        options.electrostatics.alpha = Some(0.55);
        options.electrostatics.real_cutoff = Some(12.0);
        options.electrostatics.reciprocal_cutoff = Some(7.0);
        let config = SpinorScfConfig {
            energy_tolerance: 1e-14,
            density_tolerance: 1e-13,
            orbital_tolerance: 1e-12,
            occupation: OccupationScheme::FermiDirac {
                temperature_kelvin: 0.4 / BOLTZMANN_HARTREE_PER_K,
            },
            max_iter: 200,
            ..SpinorScfConfig::default()
        };
        let magnetic = if geometry.dimension() == 1 {
            Vec3::new(0.21, 0.17, 0.13)
        } else {
            Vec3::new(0.21, 0.17, 0.0)
        };
        for name in ["svwn5", "pbe", "tpss", "r2scan", "m06l"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let calculate = |amplitude: f64, origin: Vec3| {
                let field = UniformOpenField {
                    magnetic_field: magnetic,
                    origin,
                    amplitude,
                };
                let backend = SpinorDftBackend::new(
                    &mol,
                    &mb,
                    &geometry,
                    &functional,
                    &options,
                    None,
                    vec![BlochMatrix::zeros(2, 2)],
                )
                .unwrap()
                .with_uniform_open_field(&field)
                .unwrap();
                let report = run_spinor_scf(&backend, &config).unwrap();
                assert!(
                    report.converged && report.fields.operator_converged,
                    "{name} {axes:?}: orbital={} number={} grid={} SCF={}",
                    report.orbital_residual,
                    report.particle_number_error,
                    report.fields.xc_grid_number_defect,
                    report.converged
                );
                assert!(
                    (report.fields.energy - report.fields.energy_components.iter().sum::<f64>())
                        .abs()
                        < 1e-12
                );
                assert!(
                    (report.fields.energy_components[7]
                        - report.fields.xc_component_energies.iter().sum::<f64>())
                    .abs()
                        < 1e-12
                );
                if let Some(d) = &report.fields.xc_diagnostics {
                    assert!(d.converged);
                    assert_eq!(d.angular_energy_defect, 0.0);
                    assert_eq!(d.angular_operator_defect, 0.0);
                }
                let operator = field
                    .operators(&mb, &geometry, backend.k_points(), options.basis_cutoff)
                    .unwrap();
                let derivative = backend
                    .amplitude_derivative(
                        &report.density,
                        &operator.first_derivative,
                        &[BlochMatrix::zeros(2, 2)],
                    )
                    .unwrap();
                (report, derivative)
            };
            let (reference, analytic) = calculate(0.7, Vec3::zero());
            let steps = [0.04, 0.02, 0.01, 0.005];
            let mut errors = Vec::new();
            let mut estimates = Vec::new();
            for h in steps {
                let plus = calculate(0.7 + h, Vec3::zero()).0.free_energy;
                let minus = calculate(0.7 - h, Vec3::zero()).0.free_energy;
                let estimate = (plus - minus) / (2.0 * h);
                estimates.push(estimate);
                errors.push((estimate - analytic).abs());
            }
            let fit = convergence::error_power_fit(&steps, &errors);
            let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
            println!(
                "semilocal spinor SCF {name} axes={axes:?}: derivative={analytic:.12e} \
                steps={steps:?} errors={errors:?} p={:.9} R2={:.10} \
                Richardson={rich:.3e} orbital={:.3e} number={:.3e} grid={:.3e}",
                fit.exponent,
                fit.r_squared,
                reference.orbital_residual,
                reference.particle_number_error,
                reference.fields.xc_grid_number_defect
            );
            assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
            assert!(rich < 1e-8);
            let moved = calculate(0.7, Vec3::new(1.1, -0.4, 0.9)).0;
            assert!((moved.free_energy - reference.free_energy).abs() < 1e-12);
        }
    }
}
