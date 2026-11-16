// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Open-normal magnetic slab Coulomb oracles and response acceptance.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_cell::*;
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::f64::consts::PI;

fn inputs() -> (Molecule, MolBasis, MagneticCell, MagneticScfOptions) {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mut set = BasisSet::new("magnetic-slab-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let basis = MolBasis::build(&mol, &set).unwrap();
    let geometry = PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(9.0, 0.0, 0.0),
            Vec3::new(0.0, 9.0, 0.0),
            Vec3::zero(),
        ),
        PeriodicBoundary::new([true, true, false]),
    )
    .unwrap();
    let b = 2.0 * PI / geometry.measure();
    let cell = MagneticCell::new(geometry, Vec3::new(0.0, 0.0, b), Vec3::zero()).unwrap();
    let mut o = MagneticScfOptions {
        basis_cutoff: 3.0,
        reciprocal_cutoff: 14.0,
        nuclear_widths: vec![0.3],
        grid_order: 48,
        grid_cutoff: 8.0,
        ..Default::default()
    };
    o.electrostatics.alpha = Some(0.55);
    o.electrostatics.real_cutoff = Some(14.0);
    o.electrostatics.reciprocal_cutoff = Some(8.0);
    (mol, basis, cell, o)
}
fn gaussian_green(geometry: &PeriodicGeometry, a: f64) -> f64 {
    let sum = geometry
        .reciprocal_within(40.0)
        .unwrap()
        .iter()
        .map(|g| libm::erfc(g.norm() / (2.0 * a.sqrt())) / g.norm())
        .sum::<f64>();
    2.0 * PI / geometry.measure() * sum - 2.0 * PI.sqrt() / (geometry.measure() * a.sqrt())
}
#[test]
fn normal_flux_slab_energies_match_independent_gaussian_zero_mode_and_soc_oracles() {
    let (mol, basis, cell, o) = inputs();
    let a = 0.7_f64;
    let width = 0.3_f64;
    let mut xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    xc.hybrid.cx = 0.4;
    xc.hybrid.clr = 0.3;
    xc.hybrid.omega = 0.5;
    let backend = MagneticDftBackend::new(&mol, &basis, cell.clone(), &xc, o, None).unwrap();
    let mut p = BlochMatrix::zeros(2, 2);
    p.re[(0, 0)] = 1.0;
    let fields = backend.evaluate(&[p]).unwrap();
    let coulomb = gaussian_green(&cell.geometry, a);
    let na = (1.0 / (2.0 * a) + width * width).recip();
    let nuclear = -gaussian_green(&cell.geometry, na);
    let source = 0.5 * gaussian_green(&cell.geometry, (2.0 * width * width).recip());
    let lr_a = (a.recip() + xc.hybrid.omega.powi(2).recip()).recip();
    let lr = gaussian_green(&cell.geometry, lr_a)
        + 2.0 * PI.sqrt() / (cell.geometry.measure() * xc.hybrid.omega);
    let b = cell.field.z;
    let soc = b * PI / (16.0 * a * cell.geometry.measure() * 137.035_999_084_f64.powi(2))
        * cell
            .geometry
            .reciprocal_within(40.0)
            .unwrap()
            .iter()
            .map(|g| g.norm() * libm::erfc(g.norm() / (2.0 * na.sqrt())))
            .sum::<f64>();
    let expected = [
        source,
        1.5 * a + b * b / (16.0 * a),
        nuclear,
        0.0,
        0.5 * coulomb,
        -0.5 * xc.hybrid.cx * coulomb,
        -0.5 * xc.hybrid.clr * lr,
        0.0,
        0.0,
        0.5 * b + soc,
    ];
    let errors: Vec<_> = fields
        .energy_components
        .iter()
        .zip(expected)
        .map(|(v, e)| (v - e).abs())
        .collect();
    println!(
        "magnetic slab term errors={errors:?} source={source:.12e} \
        J={coulomb:.12e} LR={lr:.12e} SOC={soc:.12e} raw Coulomb={:.3e} \
        image={:.3e}",
        fields.coulomb_operator_defect, backend.image_operator_defect
    );
    assert!(errors.iter().all(|e| *e < 1e-10));
    assert!((fields.nuclear_soc_energy - soc).abs() < 1e-12);
    assert!(fields.operator_converged);
    assert!((fields.energy - expected.iter().sum::<f64>()).abs() < 1e-10);
}

#[test]
fn magnetic_slab_fixed_flux_derivative_has_four_step_fit_and_origin_field_reversal() {
    let (mol, basis, cell, o) = inputs();
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mut backend =
        MagneticDftBackend::new(&mol, &basis, cell.clone(), &xc, o.clone(), None).unwrap();
    let generator = from_pauli(&std::array::from_fn(|a| {
        let mut m = BlochMatrix::zeros(1, 1);
        m.re[(0, 0)] = [0.0, 0.03, 0.04, 0.05][a];
        m
    }))
    .unwrap();
    let calculate = |backend: &mut MagneticDftBackend<'_>, lambda: f64| {
        let mut operator = generator.clone();
        operator.scale(lambda);
        backend.set_external_spin_operator(operator).unwrap();
        let report = run_spinor_scf(backend, &SpinorScfConfig::default()).unwrap();
        assert!(report.converged && report.fields.operator_converged);
        report
    };
    let reference = calculate(&mut backend, 0.4);
    let analytic = reference.density[0].trace_product(&generator);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mut errors = Vec::new();
    let mut estimates = Vec::new();
    for h in steps {
        let plus = calculate(&mut backend, 0.4 + h).fields.energy;
        let minus = calculate(&mut backend, 0.4 - h).fields.energy;
        let estimate = (plus - minus) / (2.0 * h);
        estimates.push(estimate);
        errors.push((estimate - analytic).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = (4.0 * estimates[3] - estimates[2]) / 3.0;
    println!(
        "magnetic slab fixed-flux: steps={steps:?} errors={errors:?} p={:.9} \
        R2={:.10} Richardson={:.3e} stationarity={:.3e} number={:.3e}",
        fit.exponent,
        fit.r_squared,
        (rich - analytic).abs(),
        reference.orbital_residual,
        reference.particle_number_error
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    assert!((rich - analytic).abs() < 1e-8);
    let mut moved = cell.clone();
    moved.gauge_origin = Vec3::new(1.1, -0.4, 0.9);
    let mut origin = MagneticDftBackend::new(&mol, &basis, moved, &xc, o.clone(), None).unwrap();
    let changed = calculate(&mut origin, 0.4);
    assert!((changed.fields.energy - reference.fields.energy).abs() < 1e-12);
    let reversed = MagneticCell::new(cell.geometry.clone(), -cell.field, Vec3::zero()).unwrap();
    let reverse = MagneticDftBackend::new(&mol, &basis, reversed, &xc, o, None).unwrap();
    let zero = calculate(&mut backend, 0.0);
    let opposite = run_spinor_scf(&reverse, &SpinorScfConfig::default()).unwrap();
    assert!((opposite.fields.energy - zero.fields.energy).abs() < 1e-12);
    println!(
        "magnetic slab origin defect={:.3e}, reversal defect={:.3e}",
        (changed.fields.energy - reference.fields.energy).abs(),
        (opposite.fields.energy - zero.fields.energy).abs()
    );
}

#[test]
fn tilted_slab_magnetic_translations_include_only_physical_faces() {
    let (_, basis, _, _) = inputs();
    for axes in [
        [true, true, false],
        [true, false, true],
        [false, true, true],
    ] {
        let lattice = Lattice::new(
            Vec3::new(9.0, 0.4, 0.7),
            Vec3::new(0.9, 9.5, 0.5),
            Vec3::new(1.2, 0.6, 10.0),
        );
        let geometry = PeriodicGeometry::new(lattice.clone(), PeriodicBoundary::new(axes)).unwrap();
        let active: Vec<_> = (0..3).filter(|&a| axes[a]).collect();
        let normal = lattice.a[active[0]]
            .cross(lattice.a[active[1]])
            .normalized();
        let field = normal * (2.0 * PI / geometry.measure())
            + geometry.parallel(Vec3::new(0.01, -0.02, 0.03));
        let cell = MagneticCell::new(geometry.clone(), field, Vec3::zero()).unwrap();
        let points = [Vec3::new(0.13, -0.2, 0.31), Vec3::new(0.21, 0.18, -0.1)];
        let original = cell.orbital_values(&basis, &points, 14.0).unwrap();
        for &axis in &active {
            let shifted: Vec<_> = points.iter().map(|r| *r + lattice.a[axis]).collect();
            let values = cell.orbital_values(&basis, &shifted, 14.0).unwrap();
            let mut error = 0.0_f64;
            for (g, r) in points.iter().enumerate() {
                let angle = -0.5 * field.cross(lattice.a[axis]).dot(*r);
                let re = original.re[(g, 0)] * angle.cos() - original.im[(g, 0)] * angle.sin();
                let im = original.re[(g, 0)] * angle.sin() + original.im[(g, 0)] * angle.cos();
                error = error
                    .max((values.re[(g, 0)] - re).abs())
                    .max((values.im[(g, 0)] - im).abs());
            }
            println!(
                "slab magnetic boundary {axes:?} axis={axis}: defect={error:.3e} flux={:?}",
                cell.flux_quanta
            );
            assert!(error < 1e-12);
        }
    }
}
