// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent continuous-normal intersector exchange and variational fits.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::dfpt::orbitals::adjoint;
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_cell::*;
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::f64::consts::PI;

fn inputs() -> (Molecule, MolBasis, MagneticCell, MagneticScfOptions) {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mut set = BasisSet::new("slab-bloch-oracle");
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
        Lattice::new(Vec3::new(9., 0., 0.), Vec3::new(0., 9., 0.), Vec3::zero()),
        PeriodicBoundary::new([true, true, false]),
    )
    .unwrap();
    let cell = MagneticCell::new(geometry, Vec3::new(0., 0., 2. * PI / 81.), Vec3::zero()).unwrap();
    let mut o = MagneticScfOptions {
        basis_cutoff: 3.,
        reciprocal_cutoff: 14.,
        nuclear_widths: vec![0.3],
        grid_order: 48,
        grid_cutoff: 8.,
        ..Default::default()
    };
    o.electrostatics.alpha = Some(0.55);
    o.electrostatics.real_cutoff = Some(14.);
    o.electrostatics.reciprocal_cutoff = Some(8.);
    (mol, basis, cell, o)
}
/// Exact q_normal integral of two normalized radial Gaussian densities.
/// This oracle uses erfc, rather than the production Fourier quadrature.
fn closed_transfer(g: &PeriodicGeometry, shift: Vec3, a: f64, omega: Option<f64>) -> f64 {
    if omega == Some(0.) {
        return 0.;
    }
    let effective = omega.map_or(a, |w| (a.recip() + w.powi(2).recip()).recip());
    let mut waves = g.reciprocal_within(40. + shift.norm()).unwrap();
    waves.push(Vec3::zero());
    let mut sum = 0.;
    for v in waves {
        let q = v + shift;
        if q.norm2() == 0. {
            sum += -2. * PI.sqrt() / (g.measure() * effective.sqrt());
            if let Some(w) = omega {
                sum += 2. * PI.sqrt() / (g.measure() * w);
            }
        } else {
            sum +=
                2. * PI / (g.measure() * q.norm()) * libm::erfc(q.norm() / (2. * effective.sqrt()));
        }
    }
    sum
}
fn closed_aux(g: &PeriodicGeometry, mesh: [usize; 3], eta: f64, omega: Option<f64>) -> f64 {
    if eta == 0. || omega == Some(0.) {
        return 0.;
    }
    let a = g.lattice().a;
    let fine = PeriodicGeometry::new(
        Lattice::new(a[0] * mesh[0] as f64, a[1] * mesh[1] as f64, a[2]),
        g.boundary(),
    )
    .unwrap();
    let integral = omega.map_or(PI.sqrt() * eta, |w| 2. * eta / PI.sqrt() * (w / eta).atan());
    integral
        - fine
            .reciprocal_within(40.)
            .unwrap()
            .iter()
            .map(|q| {
                2. * PI / (fine.measure() * q.norm())
                    * (-q.norm2() / (4. * eta * eta)).exp()
                    * omega.map_or(1., |w| libm::erfc(q.norm() / (2. * w)))
            })
            .sum::<f64>()
}

#[test]
fn intersector_full_and_range_exchange_match_closed_normal_integrals() {
    let (mol, basis, cell, o) = inputs();
    let mut xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    xc.hybrid.cx = 0.4;
    xc.hybrid.clr = 0.3;
    xc.hybrid.omega = 0.5;
    let mesh = [2, 2, 1];
    let backend =
        MagneticBlochBackend::new(&mol, &basis, cell.clone(), &xc, o.clone(), None, mesh, true)
            .unwrap();
    let mut p = BlochMatrix::zeros(2, 2);
    p.re[(0, 0)] = 1.;
    let fields = backend.evaluate(&vec![p.clone(); 4]).unwrap();
    let mut full = 0.;
    let mut range = 0.;
    for &left in &backend.k_points {
        for &right in &backend.k_points {
            full += closed_transfer(&cell.geometry, left - right, 0.7, None) / 16.;
            range += closed_transfer(&cell.geometry, left - right, 0.7, Some(0.5)) / 16.;
        }
    }
    full += closed_aux(&cell.geometry, mesh, o.exchange_auxiliary_width, None);
    range += closed_aux(&cell.geometry, mesh, o.exchange_auxiliary_width, Some(0.5));
    let errors = [
        (fields.energy_components[4]
            - 0.5 * closed_transfer(&cell.geometry, Vec3::zero(), 0.7, None))
        .abs(),
        (fields.energy_components[5] + 0.5 * 0.4 * full).abs(),
        (fields.energy_components[6] + 0.5 * 0.3 * range).abs(),
    ];
    println!(
        "slab Bloch closed full={full:0.12e} LR={range:0.12e} \
        term-errors={errors:?} operator={:0.3e} open-order={} history={:?}",
        fields.coulomb_operator_defect,
        backend.open_quadrature_order(),
        backend.open_refinement_history()
    );
    assert!(errors.iter().all(|e| *e < 1e-10));
    assert!(fields.operator_converged);
    let mut raw = o.clone();
    raw.exchange_auxiliary_width = 0.;
    let gamma = MagneticBlochBackend::new(
        &mol,
        &basis,
        cell.clone(),
        &xc,
        raw.clone(),
        None,
        [1; 3],
        true,
    )
    .unwrap();
    let finite =
        MagneticBlochBackend::new(&mol, &basis, cell, &xc, raw, None, [1; 3], false).unwrap();
    let a = gamma.evaluate(&[p.clone()]).unwrap();
    let b = finite.evaluate(&[p]).unwrap();
    assert!((a.energy - b.energy).abs() < 1e-12);
}

#[test]
fn slab_auxiliary_resolves_four_meshes_and_exact_long_range_limits() {
    let (_, _, cell, _) = inputs();
    let a = 0.7_f64;
    let omega = 0.5_f64;
    let exact = 2. * (a / PI).sqrt();
    let exact_lr = 2. / PI.sqrt() * (a.recip() + omega.powi(2).recip()).recip().sqrt();
    for width in [0.3, 0.5] {
        let mut errors = Vec::new();
        let mut lr_errors = Vec::new();
        for n in [1, 2, 3, 4] {
            let mesh = [n, n, 1];
            let g = &cell.geometry;
            let expanded = PeriodicGeometry::new(
                Lattice::new(
                    g.lattice().a[0] * n as f64,
                    g.lattice().a[1] * n as f64,
                    Vec3::zero(),
                ),
                g.boundary(),
            )
            .unwrap();
            let full = closed_transfer(&expanded, Vec3::zero(), a, None)
                + slab_gaussian_singularity_correction(g, mesh, width, 14., None).unwrap();
            let lr = closed_transfer(&expanded, Vec3::zero(), a, Some(omega))
                + slab_gaussian_singularity_correction(g, mesh, width, 14., Some(omega)).unwrap();
            let measured = [(full - exact).abs(), (lr - exact_lr).abs()];
            errors.push(measured[0]);
            lr_errors.push(measured[1]);
            println!("slab exchange continuum mesh={n} eta={width} full-error={:0.12e} LR-error={:0.12e}",measured[0],measured[1]);
            assert!(
                (slab_gaussian_singularity_correction(g, mesh, width, 14., None).unwrap()
                    - closed_aux(g, mesh, width, None))
                .abs()
                    < 1e-12
            );
            assert!(
                (slab_gaussian_singularity_correction(g, mesh, width, 14., Some(omega)).unwrap()
                    - closed_aux(g, mesh, width, Some(omega)))
                .abs()
                    < 1e-12
            );
        }
        assert!(errors.windows(2).all(|e| e[1] < e[0]));
        assert!(lr_errors.windows(2).all(|e| e[1] < e[0]));
        assert!(errors[3] < errors[0] / 30.);
        assert!(lr_errors[3] < lr_errors[0] / 30.);
    }
    assert_eq!(
        slab_gaussian_singularity_correction(&cell.geometry, [2, 2, 1], 0., 14., None).unwrap(),
        0.
    );
    assert_eq!(
        slab_gaussian_singularity_correction(&cell.geometry, [2, 2, 1], 0.5, 14., Some(0.))
            .unwrap(),
        0.
    );
    assert!(
        slab_gaussian_singularity_correction(&cell.geometry, [2, 2, 2], 0.5, 14., None).is_err()
    );
}

#[test]
fn full_range_and_local_density_variations_have_independent_four_step_fits() {
    let (mol, basis, cell, o) = inputs();
    for name in ["hf", "svwn5"] {
        let mut xc = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        if name == "hf" {
            xc.hybrid.cx = 0.4;
            xc.hybrid.clr = 0.3;
            xc.hybrid.omega = 0.5;
        }
        let backend = MagneticBlochBackend::new(
            &mol,
            &basis,
            cell.clone(),
            &xc,
            o.clone(),
            None,
            [2, 1, 1],
            true,
        )
        .unwrap();
        let density = |theta: f64| -> Vec<BlochMatrix> {
            (0..2)
                .map(|k| {
                    let angle = 0.2 + k as f64 * 0.3 + (k + 1) as f64 * theta;
                    let values = [1., angle.sin(), 0., angle.cos()];
                    from_pauli(&std::array::from_fn(|a| {
                        let mut m = BlochMatrix::zeros(1, 1);
                        m.re[(0, 0)] = values[a];
                        m
                    }))
                    .unwrap()
                })
                .collect()
        };
        let theta = 0.3;
        let p = density(theta);
        let fields = backend.evaluate(&p).unwrap();
        assert!(fields.operator_converged);
        let mut analytic = 0.;
        for k in 0..2 {
            let factor = (k + 1) as f64;
            let angle = 0.2 + k as f64 * 0.3 + factor * theta;
            let values = [0., factor * angle.cos(), 0., -factor * angle.sin()];
            let delta = from_pauli(&std::array::from_fn(|a| {
                let mut m = BlochMatrix::zeros(1, 1);
                m.re[(0, 0)] = values[a];
                m
            }))
            .unwrap();
            analytic += delta.trace_product(&fields.fock[k]) / 2.;
        }
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let plus = backend.evaluate(&density(theta + h)).unwrap();
            let minus = backend.evaluate(&density(theta - h)).unwrap();
            assert!(plus.operator_converged && minus.operator_converged);
            let estimate = (plus.energy - minus.energy) / (2. * h);
            estimates.push(estimate);
            errors.push((estimate - analytic).abs());
        }
        let f = convergence::error_power_fit(&steps, &errors);
        let rich = ((4. * estimates[3] - estimates[2]) / 3. - analytic).abs();
        println!(
            "slab Bloch {name} derivative={analytic:0.12e} steps={steps:?} \
            errors={errors:?} p={:0.9} R2={:0.10} Richardson={rich:0.3e} grid={:?}",
            f.exponent,
            f.r_squared,
            backend.grid_diagnostics(&p).unwrap()
        );
        assert!((f.exponent - 2.).abs() < 0.02 && f.r_squared > 0.99999);
        assert!(rich < 1e-8);
    }
}

#[test]
fn all_crystal_metrics_and_slab_planes_preserve_distinct_transfer_spaces() {
    let metrics = [
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]],
        [[5., 0., 0.], [0., 5., 0.], [0., 0., 6.]],
        [[5., 0., 0.], [0., 5.5, 0.], [0., 0., 6.]],
        [[5., 0., 0.], [-2.5, 4.330127018922193, 0.], [0., 0., 6.]],
        [
            [5., 0., 0.],
            [1., 4.898979485566356, 0.],
            [1., 0.816496580927726, 4.830_458_915_396_48],
        ],
        [[5., 0., 0.], [0., 5.5, 0.], [1.2, 0., 6.]],
        [[5., 0., 0.], [0.7, 5.5, 0.], [0.5, 0.8, 6.]],
    ];
    let (_, mut basis, _, _) = inputs();
    basis.shells[0].center = Vec3::new(0.2, 0.3, -0.4);
    let mut imaginary = 0.0_f64;
    for (family, rows) in metrics.into_iter().enumerate() {
        let vectors = rows.map(|r| Vec3::new(r[0], r[1], r[2]));
        for axes in [
            [true, true, false],
            [true, false, true],
            [false, true, true],
        ] {
            let ids: Vec<_> = (0..3).filter(|&a| axes[a]).collect();
            let normal = vectors[ids[0]].cross(vectors[ids[1]]).normalized();
            let v =
                std::array::from_fn::<_, 3, _>(|a| if axes[a] { vectors[a] } else { Vec3::zero() });
            let geometry =
                PeriodicGeometry::new(Lattice::new(v[0], v[1], v[2]), PeriodicBoundary::new(axes))
                    .unwrap();
            let field =
                normal * (2. * PI / geometry.measure()) + vectors[ids[0]].normalized() * 0.02;
            let origin = Vec3::new(0.4, -0.2, 0.6);
            let cell = MagneticCell::new(geometry, field, origin).unwrap();
            let g = cell.geometry.reciprocal();
            let left = g[ids[0]] * 0.23 + g[ids[1]] * 0.34;
            let right = g[ids[0]] * 0.37 - g[ids[1]] * 0.19;
            let q = left - right + g[ids[1]] + normal * 0.27;
            let vertex = cell.fourier_transfer(&basis, 14., left, right, q).unwrap();
            let opposite =
                adjoint(&cell.fourier_transfer(&basis, 14., right, left, -q).unwrap()).unwrap();
            let defect = vertex
                .re
                .max_abs_diff(&opposite.re)
                .max(vertex.im.max_abs_diff(&opposite.im));
            imaginary = imaginary.max(vertex.im[(0, 0)].abs());
            let point = Vec3::new(0.13, -0.24, 0.31);
            let phi = cell
                .with_bloch_wave(right)
                .unwrap()
                .orbital_values(&basis, &[point, point + v[ids[0]], point + v[ids[1]]], 18.)
                .unwrap();
            let value = |row| Complex::new(phi.re[(row, 0)], phi.im[(row, 0)]);
            let mut boundary = 0.0_f64;
            for (row, axis) in ids.into_iter().enumerate() {
                let angle = right.dot(v[axis]) - 0.5 * field.cross(v[axis]).dot(point + origin);
                boundary = boundary.max(
                    (value(row + 1) - Complex::new(angle.cos(), angle.sin()) * value(0)).abs(),
                );
            }
            println!("magnetic slab metric={family} axes={axes:?} boundary={boundary:.3e} transfer-adjoint={defect:.3e}");
            assert!(boundary < 1e-12 && defect < 1e-12);
        }
    }
    assert!(
        imaginary > 1e-3,
        "off-diagonal complex transfers must remain non-Hermitian in their distinct spaces"
    );
}
