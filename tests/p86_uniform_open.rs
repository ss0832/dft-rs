// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum uniform wire/slab fields, independent oracles and derivative fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::scf_boundary::DimensionalScfOptions;
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::uniform_open::UniformOpenField;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn basis(mol: &Molecule, exponents: &[f64]) -> MolBasis {
    let mut set = BasisSet::new("uniform-continuum-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: exponents
            .iter()
            .map(|a| Shell {
                l: 0,
                exps: vec![*a],
                coefs: vec![1.0],
            })
            .collect(),
    });
    MolBasis::build(mol, &set).unwrap()
}
fn defect(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
fn geometries() -> Vec<Lattice> {
    let hex = Vec3::new(-3.0, 3.0 * 3.0_f64.sqrt(), 0.0);
    vec![
        Lattice::cubic(6.0),
        Lattice::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::new(0.0, 0.0, 7.0),
        ),
        Lattice::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 7.0, 0.0),
            Vec3::new(0.0, 0.0, 8.0),
        ),
        Lattice::new(Vec3::new(6.0, 0.0, 0.0), hex, Vec3::new(0.0, 0.0, 8.0)),
        Lattice::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(1.8, 5.723635208501673, 0.0),
            Vec3::new(1.8, 1.320845093, 5.569181258),
        ),
        Lattice::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 7.0, 0.0),
            Vec3::new(2.0, 0.0, 8.0),
        ),
        Lattice::new(
            Vec3::new(6.0, 0.4, 0.7),
            Vec3::new(0.9, 7.0, 0.5),
            Vec3::new(1.2, 0.6, 8.0),
        ),
    ]
}
#[test]
fn seven_crystal_metrics_and_all_wire_axes_preserve_continuum_curl_and_origin_gauge() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.2, -0.3, 0.4),
    }]);
    let mb = basis(&mol, &[0.7]);
    let lambda = 0.63;
    for (crystal, lattice) in geometries().into_iter().enumerate() {
        for axes in [
            [true, false, false],
            [false, true, false],
            [false, false, true],
            [true, true, false],
            [true, false, true],
            [false, true, true],
        ] {
            let geometry =
                PeriodicGeometry::new(lattice.clone(), PeriodicBoundary::new(axes)).unwrap();
            let seed = Vec3::new(0.21, -0.17, 0.13);
            let b = if geometry.dimension() == 1 {
                seed
            } else {
                geometry.parallel(seed)
            };
            let field = UniformOpenField {
                magnetic_field: b,
                origin: Vec3::zero(),
                amplitude: lambda,
            };
            let m = field.gauge_matrix(&geometry).unwrap();
            let curl = Vec3::new(m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]);
            let curl_error = (curl - b).norm();
            let div = (m[0][0] + m[1][1] + m[2][2]).abs();
            assert!(curl_error < 1e-15 && div < 1e-15);
            for a in 0..3 {
                if axes[a] {
                    let delta = (field.vector_potential(&geometry, lattice.a[a]).unwrap()
                        - field.vector_potential(&geometry, Vec3::zero()).unwrap())
                    .norm();
                    assert!(delta < 1e-15, "periodic gauge defect {delta}");
                }
            }
            let op = field
                .operators(&mb, &geometry, &[Vec3::zero()], 2.0)
                .unwrap();
            let r = mol.atoms[0].position;
            let mean = Vec3::new(
                (0..3).map(|j| m[0][j] * r.get(j)).sum(),
                (0..3).map(|j| m[1][j] * r.get(j)).sum(),
                (0..3).map(|j| m[2][j] * r.get(j)).sum(),
            );
            let contact =
                mean.norm2() + m.iter().flatten().map(|v| v * v).sum::<f64>() / (4.0 * 0.7);
            assert!((op.hamiltonian[0].re[(0, 0)] - 0.5 * lambda * lambda * contact).abs() < 1e-15);
            assert!((op.first_derivative[0].re[(0, 0)] - lambda * contact).abs() < 1e-15);
            let mut moved = field.clone();
            moved.origin = Vec3::new(3.7, -1.2, 0.8);
            let origin_op = moved
                .operators(&mb, &geometry, &[Vec3::zero()], 2.0)
                .unwrap();
            let origin_defect = defect(&op.hamiltonian[0], &origin_op.hamiltonian[0]);
            assert_eq!(origin_defect, 0.0);
            assert!(moved.origin_phase_wave(&geometry).unwrap().norm() > 0.0);
            println!(
                "uniform field metric={crystal} axes={axes:?}: curl={curl_error:.3e} \
                divergence={div:.3e} origin={origin_defect:.3e} Hermiticity={:.3e}",
                op.raw_hermitian_residual
            );
        }
    }
}

#[test]
fn wire_and_slab_scf_amplitude_derivatives_have_independent_four_step_fits() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol, &[0.9, 0.3]);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    for axes in [[true, false, false], [true, true, false]] {
        let geometry =
            PeriodicGeometry::new(Lattice::cubic(9.0), PeriodicBoundary::new(axes)).unwrap();
        let mut o = DimensionalScfOptions::for_boundary(geometry.boundary());
        o.basis_cutoff = 3.0;
        o.scf.exchange_cutoff = Some(3.0);
        o.electrostatics.alpha = Some(0.55);
        o.electrostatics.real_cutoff = Some(12.0);
        o.electrostatics.reciprocal_cutoff = Some(7.0);
        let magnetic = if geometry.dimension() == 1 {
            Vec3::new(0.21, 0.17, 0.13)
        } else {
            Vec3::new(0.21, 0.17, 0.0)
        };
        let mut field = UniformOpenField {
            magnetic_field: magnetic,
            origin: Vec3::zero(),
            amplitude: 0.7,
        };
        let config = SpinorScfConfig {
            energy_tolerance: 1e-14,
            density_tolerance: 1e-13,
            orbital_tolerance: 1e-12,
            ..SpinorScfConfig::default()
        };
        let calculate = |field: &UniformOpenField| {
            let backend = SpinorDftBackend::new(
                &mol,
                &mb,
                &geometry,
                &xc,
                &o,
                None,
                vec![BlochMatrix::zeros(4, 4)],
            )
            .unwrap()
            .with_uniform_open_field(field)
            .unwrap()
            .with_nuclear_soc(1.0)
            .unwrap();
            let report = run_spinor_scf(&backend, &config).unwrap();
            assert!(report.converged && report.fields.operator_converged);
            assert!(backend.nuclear_soc_diagnostics()[0] < o.operator_tolerance);
            let operators = field
                .operators(&mb, &geometry, backend.k_points(), o.basis_cutoff)
                .unwrap();
            let derivative = backend
                .amplitude_derivative(
                    &report.density,
                    &operators.first_derivative,
                    &[BlochMatrix::zeros(4, 4)],
                )
                .unwrap();
            (report, derivative, backend.nuclear_soc_diagnostics())
        };
        let (reference, analytic, soc) = calculate(&field);
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            field.amplitude = 0.7 + h;
            let plus = calculate(&field).0.fields.energy;
            field.amplitude = 0.7 - h;
            let minus = calculate(&field).0.fields.energy;
            let estimate = (plus - minus) / (2.0 * h);
            estimates.push(estimate);
            errors.push((estimate - analytic).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = (4.0 * estimates[3] - estimates[2]) / 3.0;
        println!(
            "uniform open SCF {axes:?}: derivative={analytic:.12e} steps={steps:?} \
            errors={errors:?} p={:.9} R2={:.10} Richardson={:.3e} SOC \
            diagnostics={soc:?} orbital={:.3e} number={:.3e}",
            fit.exponent,
            fit.r_squared,
            (rich - analytic).abs(),
            reference.orbital_residual,
            reference.particle_number_error
        );
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!((rich - analytic).abs() < 1e-8);
        field.amplitude = 0.7;
        field.origin = Vec3::new(1.1, -0.4, 0.9);
        let changed_origin = calculate(&field).0;
        assert!((changed_origin.fields.energy - reference.fields.energy).abs() < 1e-12);
        field.magnetic_field = -magnetic;
        let reversed = calculate(&field).0;
        let energy_defect = (reversed.fields.energy - reference.fields.energy).abs();
        assert!(energy_defect < 1e-12);
        println!(
            "uniform open SCF {axes:?}: field reversal energy={energy_defect:.3e}, \
            origin energy={:.3e}",
            (changed_origin.fields.energy - reference.fields.energy).abs()
        );
    }
}

#[test]
fn normal_slab_flux_and_unused_origin_cannot_be_silently_discarded() {
    let geometry = PeriodicGeometry::new(
        Lattice::cubic(6.0),
        PeriodicBoundary::new([true, true, false]),
    )
    .unwrap();
    let field = UniformOpenField {
        magnetic_field: Vec3::new(0.0, 0.0, 0.2),
        origin: Vec3::zero(),
        amplitude: 1.0,
    };
    assert!(field
        .gauge_matrix(&geometry)
        .unwrap_err()
        .to_string()
        .contains("magnetic Bloch"));
}

#[test]
fn collinear_uniform_field_derivative_has_independent_four_step_fits() {
    use dft_rs::periodic::scf_boundary::run_dimensional_uniform_scf;
    let mut mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    mol.multiplicity = 2;
    let mb = basis(&mol, &[0.9, 0.3]);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    for axes in [[true, false, false], [true, true, false]] {
        let geometry =
            PeriodicGeometry::new(Lattice::cubic(9.0), PeriodicBoundary::new(axes)).unwrap();
        let mut o = DimensionalScfOptions::for_boundary(geometry.boundary());
        o.basis_cutoff = 3.0;
        o.scf.exchange_cutoff = Some(3.0);
        o.scf.e_tol = 1e-14;
        o.scf.p_tol = 1e-13;
        o.scf.conv_tol_grad = 1e-12;
        o.electrostatics.alpha = Some(0.55);
        o.electrostatics.real_cutoff = Some(12.0);
        o.electrostatics.reciprocal_cutoff = Some(7.0);
        let magnetic = if geometry.dimension() == 1 {
            Vec3::new(0.21, 0.17, 0.13)
        } else {
            Vec3::new(0.21, 0.17, 0.0)
        };
        let mut field = UniformOpenField {
            magnetic_field: magnetic,
            origin: Vec3::zero(),
            amplitude: 0.7,
        };
        let calculate = |f: &UniformOpenField| {
            let report = run_dimensional_uniform_scf(&mol, &mb, &geometry, &xc, &o, f).unwrap();
            assert!(report.converged);
            assert!(!report.reference.orbital_conjugation_constraint);
            let op = f
                .operators(&mb, &geometry, &report.reference.k_points, o.basis_cutoff)
                .unwrap();
            let derivative =
                report.reference.total_density()[0].trace_product(&op.first_derivative[0]);
            (report.reference.energy, derivative)
        };
        let analytic = calculate(&field).1;
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            field.amplitude = 0.7 + h;
            let plus = calculate(&field).0;
            field.amplitude = 0.7 - h;
            let minus = calculate(&field).0;
            let d = (plus - minus) / (2.0 * h);
            estimates.push(d);
            errors.push((d - analytic).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = (4.0 * estimates[3] - estimates[2]) / 3.0;
        println!(
            "collinear uniform SCF {axes:?}: steps={steps:?} errors={errors:?} \
            p={:.9} R2={:.10} Richardson={:.3e}",
            fit.exponent,
            fit.r_squared,
            (rich - analytic).abs()
        );
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!((rich - analytic).abs() < 1e-8);
    }
}
