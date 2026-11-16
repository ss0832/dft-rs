// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dimensional Gaussian Coulomb and Cartesian derivative gates.

use dft_rs::math::Vec3;
use dft_rs::periodic::electrostatics::DimensionalCoulomb;
use dft_rs::periodic::electrostatics::{
    periodic_electrostatics, ChargeCompensation, ElectrostaticOptions, ElectrostaticSource,
};
use dft_rs::periodic::ewald::EwaldParams;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::gaussian_coulomb::{
    periodic_gaussian_integral, CartesianGaussian, GaussianIntegralOptions, GaussianNucleus,
};
use dft_rs::periodic::{Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;

fn geometry(d: usize) -> PeriodicGeometry {
    PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.2, 0.5, 0.3),
            Vec3::new(0.8, 5.6, 0.4),
            Vec3::new(0.3, 0.2, 6.0),
        ),
        PeriodicBoundary::new(std::array::from_fn(|i| i < d)),
    )
    .unwrap()
}
fn gaussians() -> Vec<CartesianGaussian> {
    vec![
        CartesianGaussian {
            center: Vec3::new(0.1, -0.2, 0.3),
            exponent: 0.6,
            angular: [1, 0, 0],
        },
        CartesianGaussian {
            center: Vec3::new(0.8, 0.4, -0.1),
            exponent: 0.4,
            angular: [0, 1, 0],
        },
        CartesianGaussian {
            center: Vec3::new(-0.4, 0.9, 0.6),
            exponent: 0.5,
            angular: [0, 0, 2],
        },
        CartesianGaussian {
            center: Vec3::new(0.6, -0.7, 1.1),
            exponent: 0.7,
            angular: [1, 0, 0],
        },
    ]
}
fn options() -> GaussianIntegralOptions {
    GaussianIntegralOptions {
        alpha: Some(0.55),
        real_cutoff: Some(16.0),
        reciprocal_cutoff: Some(8.0),
        ..Default::default()
    }
}
fn fit(label: &str, steps: &[f64; 4], estimates: &[f64; 4], exact: f64) {
    let errors = estimates.map(|v| (v - exact).abs());
    assert!(errors.iter().all(|v| v.is_finite() && *v > 0.0));
    let x = steps.map(f64::ln);
    let y = errors.map(f64::ln);
    let xm = x.iter().sum::<f64>() / 4.0;
    let ym = y.iter().sum::<f64>() / 4.0;
    let p = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - xm) * (b - ym))
        .sum::<f64>()
        / x.iter().map(|a| (a - xm).powi(2)).sum::<f64>();
    let r2 = 1.0
        - x.iter()
            .zip(y)
            .map(|(a, b)| (b - ym - p * (a - xm)).powi(2))
            .sum::<f64>()
            / y.iter().map(|b| (b - ym).powi(2)).sum::<f64>();
    println!(
        "{label} errors={errors:?}; h={}..{} Bohr p={p:.12} R2={r2:.12}",
        steps[0], steps[3]
    );
    assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
    for e in errors.windows(2) {
        assert!((e[0] / e[1] / 4.0 - 1.0).abs() < 0.03);
    }
    assert!(((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs() < 1e-8);
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |s, i| s * (n - i) as f64 / (i + 1) as f64)
}
// Independent ordinary polynomial/moment expansion, without Hermite E tables.
fn fourier_pair(a: &CartesianGaussian, b: &CartesianGaussian, g: Vec3) -> Complex {
    let p = a.exponent + b.exponent;
    let mu = a.exponent * b.exponent / p;
    let center = (a.center * a.exponent + b.center * b.exponent) / p;
    let ca = [
        center.x - a.center.x,
        center.y - a.center.y,
        center.z - a.center.z,
    ];
    let cb = [
        center.x - b.center.x,
        center.y - b.center.y,
        center.z - b.center.z,
    ];
    let wave = [g.x, g.y, g.z];
    let mut result = Complex::new(g.dot(center).cos(), g.dot(center).sin())
        * (-mu * (a.center - b.center).norm2()).exp();
    for axis in 0..3 {
        let degree = a.angular[axis] + b.angular[axis];
        let mut moments = vec![Complex::ZERO; degree + 1];
        moments[0] = Complex::from(
            (std::f64::consts::PI / p).sqrt() * (-wave[axis].powi(2) / (4.0 * p)).exp(),
        );
        for n in 0..degree {
            moments[n + 1] = Complex::new(0.0, wave[axis] / (2.0 * p)) * moments[n];
            if n > 0 {
                let add = moments[n - 1] * (n as f64 / (2.0 * p));
                moments[n + 1] += add;
            }
        }
        let mut sum = Complex::ZERO;
        for i in 0..=a.angular[axis] {
            for j in 0..=b.angular[axis] {
                sum += moments[i + j]
                    * (binomial(a.angular[axis], i)
                        * binomial(b.angular[axis], j)
                        * ca[axis].powi((a.angular[axis] - i) as i32)
                        * cb[axis].powi((b.angular[axis] - j) as i32));
            }
        }
        result = result * sum;
    }
    let norm = |a: &CartesianGaussian| {
        dft_rs::basis::primitive_norm(
            a.exponent,
            (
                a.angular[0] as i32,
                a.angular[1] as i32,
                a.angular[2] as i32,
            ),
        )
    };
    result * (norm(a) * norm(b))
}

#[test]
fn charged_3d_gaussian_uniform_background_matches_full_fourier_oracle() {
    let g = PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.8, 5.4, 0.0),
            Vec3::new(0.3, 0.2, 6.1),
        ),
        PeriodicBoundary::FULL_3D,
    )
    .unwrap();
    let sources = [
        ElectrostaticSource {
            position: Vec3::new(0.3, -0.4, 0.6),
            charge: 1.2,
            width: 1.3,
        },
        ElectrostaticSource {
            position: Vec3::new(-0.7, 0.8, -0.2),
            charge: -0.4,
            width: 0.9,
        },
    ];
    let options = ElectrostaticOptions {
        compensation: ChargeCompensation::UniformBackground,
        ..Default::default()
    };
    let report = periodic_electrostatics(&g, &sources, &options).unwrap();
    let mut oracle = 0.0;
    for wave in g.reciprocal_within(10.0).unwrap() {
        let (mut re, mut im) = (0.0, 0.0);
        for source in &sources {
            let weight = source.charge * (-source.width.powi(2) * wave.norm2() / 4.0).exp();
            re += weight * wave.dot(source.position).cos();
            im += weight * wave.dot(source.position).sin();
        }
        oracle +=
            0.5 * 4.0 * std::f64::consts::PI / g.measure() / wave.norm2() * (re * re + im * im);
    }
    println!(
        "charged Gaussian energy {} full-Fourier oracle {} difference {}",
        report.energy,
        oracle,
        report.energy - oracle
    );
    assert!((report.energy - oracle).abs() < 1e-11);
}

#[test]
fn normalized_p_d_nuclear_and_eri_match_independent_full_fourier_integrals() {
    let g = geometry(3);
    let gs = gaussians();
    let source = GaussianNucleus {
        center: Vec3::new(-0.5, 0.7, 0.9),
        charge: 2.0,
        width: 0.0,
    };
    let eri = periodic_gaussian_integral(&g, &gs, None, &options()).unwrap();
    let nuclear = periodic_gaussian_integral(&g, &gs[..2], Some(&source), &options()).unwrap();
    let (mut e, mut n) = (0.0, 0.0);
    for wave in g.reciprocal_within(14.0).unwrap() {
        let kernel = 4.0 * std::f64::consts::PI / g.measure() / wave.norm2();
        let ab = fourier_pair(&gs[0], &gs[1], wave);
        let cd = fourier_pair(&gs[2], &gs[3], wave);
        e += kernel * (ab * cd.conj()).re;
        let phase = wave.dot(source.center);
        let phi = Complex::new(phase.cos(), -phase.sin());
        n -= source.charge * kernel * (ab * phi).re;
    }
    println!(
        "p/d ERI={} Fourier={} defect={}; nuclear={} Fourier={} defect={}",
        eri.value,
        e,
        eri.value - e,
        nuclear.value,
        n,
        nuclear.value - n
    );
    assert!((eri.value - e).abs() < 1e-11 && (nuclear.value - n).abs() < 1e-11);
    assert!(eri.converged && nuclear.converged);
    for report in [eri, nuclear] {
        assert!(report.translation_defect < 1e-11);
        assert!((report.components.total() - report.value).abs() < 1e-13);
    }
}

#[test]
fn all_dimensional_gaussian_tensors_match_scalar_pairs_and_four_step_high_derivatives() {
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let displacement = Vec3::new(0.6, -0.8, 1.1);
    for d in 1..=3 {
        let g = geometry(d);
        let kernel = DimensionalCoulomb::new(
            g,
            EwaldParams {
                alpha: 0.55,
                rcut: 16.0,
                gcut: 8.0,
            },
            96,
        )
        .unwrap();
        let tensor = kernel.gaussian_derivatives(displacement, 1.2, 4).unwrap();
        let (value, gradient) = kernel.pair(displacement, 1.2, false).unwrap();
        assert!((tensor.get([0; 3]).unwrap() - value).abs() < 1e-11);
        for (axis, exact) in [gradient.x, gradient.y, gradient.z].into_iter().enumerate() {
            let mut m = [0; 3];
            m[axis] = 1;
            assert!((tensor.get(m).unwrap() - exact).abs() < 1e-11);
        }
        for m in [[1, 0, 0], [2, 0, 0], [1, 1, 1], [2, 1, 1]] {
            let mut lower = m;
            lower[0] -= 1;
            let mut estimates = [0.0; 4];
            for (index, h) in steps.iter().enumerate() {
                let p = kernel
                    .gaussian_derivatives(displacement + Vec3::new(*h, 0.0, 0.0), 1.2, 3)
                    .unwrap();
                let n = kernel
                    .gaussian_derivatives(displacement - Vec3::new(*h, 0.0, 0.0), 1.2, 3)
                    .unwrap();
                estimates[index] = (p.get(lower).unwrap() - n.get(lower).unwrap()) / (2.0 * h);
            }
            fit(
                &format!("{d}D tensor {m:?}"),
                &steps,
                &estimates,
                tensor.get(m).unwrap(),
            );
        }
    }
}

#[test]
fn nuclear_and_hartree_primitives_have_all_dimension_four_step_center_gradients() {
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let gs = gaussians();
    let source = GaussianNucleus {
        center: Vec3::new(-0.5, 0.7, 0.9),
        charge: 2.0,
        width: 0.3,
    };
    for d in 1..=3 {
        let g = geometry(d);
        for (label, gaussians, source) in
            [("nuclear", &gs[..2], Some(&source)), ("eri", &gs[..], None)]
        {
            let exact = periodic_gaussian_integral(&g, gaussians, source, &options()).unwrap();
            assert!(exact.converged && exact.translation_defect < 1e-11);
            let mut estimates = [0.0; 4];
            for (index, h) in steps.iter().enumerate() {
                let mut p = gaussians.to_vec();
                p[0].center.x += h;
                let mut n = gaussians.to_vec();
                n[0].center.x -= h;
                estimates[index] = (periodic_gaussian_integral(&g, &p, source, &options())
                    .unwrap()
                    .value
                    - periodic_gaussian_integral(&g, &n, source, &options())
                        .unwrap()
                        .value)
                    / (2.0 * h);
            }
            fit(
                &format!("{d}D primitive {label}"),
                &steps,
                &estimates,
                exact.center_gradients[0].x,
            );
        }
    }
}

#[test]
fn gaussian_integrals_preserve_split_gauge_permutations_and_rigid_translation() {
    let gs = gaussians();
    let source = GaussianNucleus {
        center: Vec3::new(-0.5, 0.7, 0.9),
        charge: 2.0,
        width: 0.3,
    };
    let shift = Vec3::new(1.3, -0.8, 0.4);
    for d in 1..=3 {
        let geometry = geometry(d);
        for (gaussians, source) in [(&gs[..2], Some(&source)), (&gs[..], None)] {
            let exact =
                periodic_gaussian_integral(&geometry, gaussians, source, &options()).unwrap();
            let mut changed = options();
            changed.alpha = Some(0.7);
            let split = periodic_gaussian_integral(&geometry, gaussians, source, &changed).unwrap();
            assert!(
                (exact.value - split.value).abs() < 1e-11,
                "{d}D splitting: {}",
                exact.value - split.value
            );
            assert!((exact.components.real_space - split.components.real_space).abs() > 1e-5);
            for (a, b) in exact.center_gradients.iter().zip(&split.center_gradients) {
                assert!((*a - *b).norm() < 1e-11);
            }
            let mut moved = gaussians.to_vec();
            for g in &mut moved {
                g.center += shift;
            }
            let shifted_source = source.map(|s| GaussianNucleus {
                center: s.center + shift,
                ..s.clone()
            });
            let translated =
                periodic_gaussian_integral(&geometry, &moved, shifted_source.as_ref(), &options())
                    .unwrap();
            assert!((exact.value - translated.value).abs() < 1e-12);
            for (a, b) in exact
                .center_gradients
                .iter()
                .zip(&translated.center_gradients)
            {
                assert!((*a - *b).norm() < 1e-12);
            }
            let mut swapped = gaussians.to_vec();
            swapped.swap(0, 1);
            let swap = periodic_gaussian_integral(&geometry, &swapped, source, &options()).unwrap();
            assert!((swap.value - exact.value).abs() < 1e-12);
            assert!((swap.center_gradients[0] - exact.center_gradients[1]).norm() < 1e-12);
            if source.is_none() {
                let exchanged = vec![gs[2].clone(), gs[3].clone(), gs[0].clone(), gs[1].clone()];
                let swap =
                    periodic_gaussian_integral(&geometry, &exchanged, None, &options()).unwrap();
                assert!((swap.value - exact.value).abs() < 1e-12);
                assert!((swap.center_gradients[0] - exact.center_gradients[2]).norm() < 1e-12);
            }
        }
    }
}

#[test]
fn coincident_gaussians_are_finite_and_match_full_fourier_without_self_removal() {
    let geometry = geometry(3);
    let gaussians = vec![
        CartesianGaussian {
            center: Vec3::zero(),
            exponent: 0.6,
            angular: [0; 3]
        };
        4
    ];
    let source = GaussianNucleus {
        center: Vec3::zero(),
        charge: 1.0,
        width: 0.0,
    };
    for (gs, source) in [(&gaussians[..2], Some(&source)), (&gaussians[..], None)] {
        let report = periodic_gaussian_integral(&geometry, gs, source, &options()).unwrap();
        let mut oracle = 0.0;
        for wave in geometry.reciprocal_within(14.0).unwrap() {
            let ab = fourier_pair(&gs[0], &gs[1], wave);
            oracle += 4.0 * std::f64::consts::PI / geometry.measure() / wave.norm2()
                * if source.is_some() {
                    -ab.re
                } else {
                    (ab * fourier_pair(&gs[2], &gs[3], wave).conj()).re
                };
        }
        assert!((report.value - oracle).abs() < 1e-11);
        assert!(report.converged && report.center_gradients.iter().all(|g| g.norm() < 1e-12));
        assert_eq!(report.components.point_self, 0.0);
    }
    let kernel = DimensionalCoulomb::new(
        geometry,
        EwaldParams {
            alpha: 0.55,
            rcut: 16.0,
            gcut: 8.0,
        },
        96,
    )
    .unwrap();
    for width in [1.2, 1.0 / 0.55] {
        let tensor = kernel.gaussian_derivatives(Vec3::zero(), width, 4).unwrap();
        assert!(tensor.get([4, 0, 0]).unwrap().is_finite());
        assert!(tensor.get([1, 0, 0]).unwrap().abs() < 1e-12);
    }
}

#[test]
fn invalid_gaussian_controls_and_unrepresentable_orders_are_rejected() {
    let geometry = geometry(1);
    let gs = gaussians();
    let source = GaussianNucleus {
        center: Vec3::zero(),
        charge: 1.0,
        width: 0.0,
    };
    for exponent in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let mut bad = gs[..2].to_vec();
        bad[0].exponent = exponent;
        assert!(periodic_gaussian_integral(&geometry, &bad, Some(&source), &options()).is_err());
    }
    let mut bad = gs.clone();
    bad[0].angular = [usize::MAX, 0, 0];
    assert!(periodic_gaussian_integral(&geometry, &bad, None, &options()).is_err());
    let source = GaussianNucleus {
        width: -0.1,
        ..source
    };
    assert!(periodic_gaussian_integral(&geometry, &gs[..2], Some(&source), &options()).is_err());
    assert!(periodic_gaussian_integral(&geometry, &gs[..2], None, &options()).is_err());
    let kernel = DimensionalCoulomb::new(
        geometry,
        EwaldParams {
            alpha: 0.55,
            rcut: 16.0,
            gcut: 8.0,
        },
        96,
    )
    .unwrap();
    assert!(kernel
        .gaussian_derivatives(Vec3::zero(), 1.0, usize::MAX)
        .is_err());
    assert!(kernel.gaussian_derivatives(Vec3::zero(), 0.0, 4).is_err());
    let tensor = kernel.gaussian_derivatives(Vec3::zero(), 1.0, 4).unwrap();
    assert!(tensor.components([usize::MAX, 1, 0]).is_err());
    assert!(tensor.get([3, 2, 0]).is_err());
}
