// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent magnetic Bloch transfer, energy and derivative acceptance.
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

fn phase(x: f64) -> Complex {
    Complex::new(x.cos(), x.sin())
}
fn z(m: &BlochMatrix, a: usize, b: usize) -> Complex {
    Complex::new(m.re[(a, b)], m.im[(a, b)])
}
fn gap(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
fn inputs() -> (Molecule, MolBasis, MagneticCell, MagneticScfOptions) {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mut set = BasisSet::new("magnetic-bloch-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let basis = MolBasis::build(&mol, &set).unwrap();
    let cell = MagneticCell::new(
        PeriodicGeometry::new(Lattice::cubic(8.0), PeriodicBoundary::FULL_3D).unwrap(),
        Vec3::new(1.0, 1.0, -1.0) * (2.0 * PI / 64.0),
        Vec3::zero(),
    )
    .unwrap();
    let mut options = MagneticScfOptions {
        reciprocal_cutoff: 16.0,
        grid_order: 40,
        nuclear_widths: vec![0.3],
        ..Default::default()
    };
    options.electrostatics.alpha = Some(0.55);
    options.electrostatics.real_cutoff = Some(16.0);
    options.electrostatics.reciprocal_cutoff = Some(8.0);
    (mol, basis, cell, options)
}
fn closed_vertex(cell: &MagneticCell, right: Vec3, wave: Vec3) -> Complex {
    let mut total = Complex::ZERO;
    for i in -2..=2 {
        for j in -2..=2 {
            for k in -2..=2 {
                let n = [i, j, k];
                let t = cell.geometry.lattice().translation(n);
                let m = cell.flux_quanta;
                let parity = (m[0] * i * j + m[1] * j * k - m[2] * i * k).rem_euclid(2);
                let w = wave - cell.field.cross(t) * 0.5;
                total += phase(right.dot(t) + wave.dot(t) * 0.5)
                    * if parity == 0 { 1.0 } else { -1.0 }
                    * (-0.35 * t.norm2() - w.norm2() / 5.6).exp();
            }
        }
    }
    total
}
fn cfg() -> SpinorScfConfig {
    SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-11,
        ..SpinorScfConfig::default()
    }
}
fn fit(label: &str, steps: &[f64; 4], errors: &[f64], rich: f64) {
    let f = convergence::error_power_fit(steps, errors);
    println!(
        "{label}: steps={steps:?} raw-errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e}",
        f.exponent, f.r_squared
    );
    assert!((f.exponent - 2.0).abs() < 0.02 && f.r_squared > 0.99999);
    assert!(rich < 1e-8);
}

#[test]
fn noninteger_magnetic_transfers_have_closed_vertices_and_adjoint_spaces() {
    let (_, basis, cell, _) = inputs();
    let reciprocal = cell.geometry.reciprocal();
    let left = reciprocal[0] * 0.2 + reciprocal[1] * 0.3;
    let right = reciprocal[0] * 0.375 - reciprocal[2] * 0.1;
    let q = left - right + reciprocal[1];
    let r = cell.fourier_transfer(&basis, 12.0, left, right, q).unwrap();
    let opposite = cell
        .fourier_transfer(&basis, 12.0, right, left, -q)
        .unwrap();
    let oracle = closed_vertex(&cell, right, q);
    println!(
        "magnetic transfer: closed={:.3e} adjoint={:.3e} same-q-imaginary={:.3e}",
        (z(&r, 0, 0) - oracle).abs(),
        gap(&r, &adjoint(&opposite).unwrap()),
        r.im[(0, 0)].abs()
    );
    assert!((z(&r, 0, 0) - oracle).abs() < 1e-12);
    assert!(gap(&r, &adjoint(&opposite).unwrap()) < 1e-12);
    assert!(cell
        .fourier_transfer(&basis, 12.0, left, right, q + reciprocal[0] * 0.1)
        .is_err());
    let kcell = cell.with_bloch_wave(right).unwrap();
    let point = Vec3::new(0.2, 0.3, -0.1);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for axis in 0..3 {
        let exact = z(
            &kcell
                .orbital_momentum_values(&basis, &[point], 20.0, axis)
                .unwrap(),
            0,
            0,
        );
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let d = match axis {
                0 => Vec3::new(h, 0.0, 0.0),
                1 => Vec3::new(0.0, h, 0.0),
                _ => Vec3::new(0.0, 0.0, h),
            };
            let values = kcell
                .orbital_values(&basis, &[point + d, point - d, point], 20.0)
                .unwrap();
            let estimate = Complex::new(0.0, -1.0) * (z(&values, 0, 0) - z(&values, 1, 0))
                / (2.0 * h)
                + z(&values, 2, 0) * cell.field.cross(point - cell.gauge_origin).get(axis) * 0.5;
            errors.push((estimate - exact).abs());
            estimates.push(estimate);
        }
        fit(
            &format!("nonzero-k momentum axis {axis}"),
            &steps,
            &errors,
            ((estimates[3] * 4.0 - estimates[2]) / 3.0 - exact).abs(),
        );
    }
}

#[test]
fn magnetic_bz_exchange_matches_independent_closed_intersector_sums() {
    let (mol, basis, cell, mut options) = inputs();
    options.exchange_auxiliary_width = 0.0;
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let backend = MagneticBlochBackend::new(
        &mol,
        &basis,
        cell.clone(),
        &xc,
        options.clone(),
        None,
        [2, 1, 1],
        true,
    )
    .unwrap();
    let state = run_spinor_scf(&backend, &cfg()).unwrap();
    assert!(state.converged && state.fields.operator_converged);
    let ks = &backend.k_points;
    let nk = ks.len() as f64;
    let s = ks
        .iter()
        .map(|&k| closed_vertex(&cell, k, Vec3::zero()).re)
        .collect::<Vec<_>>();
    let mut j = 0.0;
    let mut v = 0.0;
    let mut x = 0.0;
    for g in cell.geometry.reciprocal_within(16.0).unwrap() {
        let rho = ks
            .iter()
            .enumerate()
            .map(|(i, &k)| closed_vertex(&cell, k, g) / s[i] / nk)
            .fold(Complex::ZERO, |a, b| a + b);
        let kernel = 4.0 * PI / (512.0 * g.norm2());
        j += 0.5 * kernel * rho.abs().powi(2);
        v -= kernel * rho.re * (-0.25 * g.norm2() * 0.09).exp();
    }
    for (i, &left) in ks.iter().enumerate() {
        for (j, &right) in ks.iter().enumerate() {
            let shift = left - right;
            let mut waves = cell
                .geometry
                .reciprocal_within(16.0 + shift.norm())
                .unwrap();
            waves.push(Vec3::zero());
            for q in waves
                .into_iter()
                .map(|g| g + shift)
                .filter(|q| q.norm2() > 0.0 && q.norm() <= 16.0)
            {
                x -= 0.5 * 4.0 * PI / (512.0 * q.norm2())
                    * closed_vertex(&cell, right, q).abs().powi(2)
                    / (s[i] * s[j] * nk * nk);
            }
        }
    }
    println!(
        "magnetic BZ closed terms: J={} X={} V={} defects={:?} transfer-modes={}",
        j,
        x,
        v,
        [
            state.fields.energy_components[4] - j,
            state.fields.energy_components[5] - x,
            state.fields.energy_components[2] - v
        ],
        backend.exchange_transfer_modes()
    );
    assert!((state.fields.energy_components[4] - j).abs() < 1e-11);
    assert!((state.fields.energy_components[5] - x).abs() < 1e-11);
    assert!((state.fields.energy_components[2] - v).abs() < 1e-11);
    let gamma = MagneticBlochBackend::new(
        &mol,
        &basis,
        cell.clone(),
        &xc,
        options.clone(),
        None,
        [1; 3],
        true,
    )
    .unwrap();
    let finite =
        MagneticBlochBackend::new(&mol, &basis, cell, &xc, options, None, [1; 3], false).unwrap();
    let a = run_spinor_scf(&gamma, &cfg()).unwrap();
    let b = run_spinor_scf(&finite, &cfg()).unwrap();
    assert!((a.fields.energy - b.fields.energy).abs() < 1e-12);
    assert!(gap(&a.fields.fock[0], &b.fields.fock[0]) < 1e-12);
    assert!((state.fields.energy_components[5] - a.fields.energy_components[5]).abs() > 1e-3);
}

#[test]
fn magnetic_bloch_hf_lda_have_four_step_variational_density_fits() {
    let (mol, basis, cell, options) = inputs();
    for name in ["hf", "svwn5"] {
        let xc = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let backend = MagneticBlochBackend::new(
            &mol,
            &basis,
            cell.clone(),
            &xc,
            options.clone(),
            None,
            [2, 1, 1],
            true,
        )
        .unwrap();
        let density = |theta: f64| {
            backend
                .overlap()
                .iter()
                .enumerate()
                .map(|(index, s)| {
                    let angle = theta * (1.0 + 0.25 * index as f64);
                    let c = [Complex::from(angle.cos()), phase(0.3) * angle.sin()];
                    let mut p = BlochMatrix::zeros(2, 2);
                    for a in 0..2 {
                        for b in 0..2 {
                            let z = c[a] * c[b].conj() / s.re[(0, 0)];
                            p.re[(a, b)] = z.re;
                            p.im[(a, b)] = z.im;
                        }
                    }
                    p
                })
                .collect::<Vec<_>>()
        };
        let theta = 0.37;
        let p = density(theta);
        let state = backend.evaluate(&p).unwrap();
        assert!(state.operator_converged);
        let mut exact = 0.0;
        for (i, s) in backend.overlap().iter().enumerate() {
            let factor = 1.0 + 0.25 * i as f64;
            let angle = theta * factor;
            let c = [Complex::from(angle.cos()), phase(0.3) * angle.sin()];
            let d = [
                Complex::from(-factor * angle.sin()),
                phase(0.3) * (factor * angle.cos()),
            ];
            let mut delta = BlochMatrix::zeros(2, 2);
            for a in 0..2 {
                for b in 0..2 {
                    let v = (d[a] * c[b].conj() + c[a] * d[b].conj()) / s.re[(0, 0)];
                    delta.re[(a, b)] = v.re;
                    delta.im[(a, b)] = v.im;
                }
            }
            exact += delta.trace_product(&state.fock[i]) / 2.0;
        }
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let plus = backend.evaluate(&density(theta + h)).unwrap();
            let minus = backend.evaluate(&density(theta - h)).unwrap();
            assert!(plus.operator_converged && minus.operator_converged);
            let estimate = (plus.energy - minus.energy) / (2.0 * h);
            errors.push((estimate - exact).abs());
            estimates.push(estimate);
        }
        fit(
            &format!("magnetic BZ {name} density"),
            &steps,
            &errors,
            ((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs(),
        );
        println!(
            "magnetic BZ {name} grid defects={:?} retained-history={:?}",
            backend.grid_diagnostics(&p).unwrap(),
            backend.grid_refinement_history()
        );
    }
}

#[test]
fn gaussian_exchange_integration_resolves_four_meshes_and_auxiliary_widths() {
    let (_, _, cell, _) = inputs();
    let exact = 2.0 * (0.7 / PI).sqrt();
    for eta in [0.3, 0.5, 0.7_f64.sqrt()] {
        let mut errors = Vec::new();
        for n in [1, 2, 3, 4] {
            let fine =
                PeriodicGeometry::new(Lattice::cubic(8.0 * n as f64), PeriodicBoundary::FULL_3D)
                    .unwrap();
            let raw = fine
                .reciprocal_within(12.0)
                .unwrap()
                .iter()
                .map(|q| 4.0 * PI / (fine.measure() * q.norm2()) * (-q.norm2() / 2.8).exp())
                .sum::<f64>();
            let correction =
                gaussian_singularity_correction(&cell.geometry, [n; 3], eta, 12.0).unwrap();
            let error = (raw + correction - exact).abs();
            println!(
                "Gaussian magnetic exchange continuum: mesh={n} eta={eta:.9} \
                raw-error={:.12e} corrected-error={error:.12e} \
                correction={correction:.12e}",
                (raw - exact).abs()
            );
            errors.push(error);
        }
        if eta == 0.7_f64.sqrt() {
            assert!(errors.iter().all(|e| *e < 1e-12));
        } else {
            assert!(errors.windows(2).all(|e| e[1] < e[0]));
            assert!(errors[3] < errors[0] / 40.0);
        }
    }
    assert_eq!(
        gaussian_singularity_correction(&cell.geometry, [2; 3], 0.0, 12.0).unwrap(),
        0.0
    );
    assert!(gaussian_singularity_correction(&cell.geometry, [0, 1, 1], 0.5, 12.0).is_err());
}

#[test]
fn all_seven_crystal_systems_preserve_nonzero_k_boundaries_and_transfer_adjoint() {
    let rows = [
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
    for (family, rows) in rows.into_iter().enumerate() {
        let a = rows.map(|r| Vec3::new(r[0], r[1], r[2]));
        let lattice = Lattice::new(a[0], a[1], a[2]);
        let field = (a[2] + a[0] - a[1]) * (2.0 * PI / lattice.volume());
        let origin = Vec3::new(0.4, -0.2, 0.6);
        let cell = MagneticCell::new(
            PeriodicGeometry::new(lattice, PeriodicBoundary::FULL_3D).unwrap(),
            field,
            origin,
        )
        .unwrap();
        let g = cell.geometry.reciprocal();
        let left = g[0] * 0.23 + g[1] * 0.34;
        let right = g[1] * 0.37 - g[2] * 0.19;
        let q = left - right + g[2];
        let r = cell.fourier_transfer(&basis, 12., left, right, q).unwrap();
        let other = cell.fourier_transfer(&basis, 12., right, left, -q).unwrap();
        let defect = gap(&r, &adjoint(&other).unwrap());
        let point = Vec3::new(0.13, -0.24, 0.31);
        let phi = cell
            .with_bloch_wave(right)
            .unwrap()
            .orbital_values(
                &basis,
                &[point, point + a[0], point + a[1], point + a[2]],
                16.,
            )
            .unwrap();
        let mut boundary = 0.0_f64;
        for axis in 0..3 {
            let coefficient =
                phase(right.dot(a[axis]) - 0.5 * field.cross(a[axis]).dot(point + origin));
            boundary = boundary.max((z(&phi, axis + 1, 0) - coefficient * z(&phi, 0, 0)).abs());
        }
        println!(
            "nonzero-k crystal {family}: boundary={boundary:.3e} transfer-adjoint={defect:.3e}"
        );
        assert!(boundary < 1e-12 && defect < 1e-12);
    }
}
