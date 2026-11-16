// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dimensional Coulomb and derivative acceptance gates.

use dft_rs::math::Vec3;
use dft_rs::periodic::electrostatics::{
    periodic_electrostatics, ChargeCompensation, ElectrostaticOptions, ElectrostaticSource,
};
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::{Lattice, PeriodicGeometry};

fn geometry(d: usize) -> PeriodicGeometry {
    PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.1, 5.4, 0.0),
            Vec3::new(0.3, 0.4, 6.1),
        ),
        PeriodicBoundary::new(std::array::from_fn(|i| i < d)),
    )
    .unwrap()
}
fn point(position: Vec3, charge: f64) -> ElectrostaticSource {
    ElectrostaticSource {
        position,
        charge,
        width: 0.0,
    }
}
fn fit(steps: &[f64], errors: &[f64]) -> (f64, f64) {
    assert_eq!(steps.len(), errors.len());
    assert!(steps.len() >= 4);
    assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
    let x: Vec<_> = steps.iter().map(|v| v.ln()).collect();
    let y: Vec<_> = errors.iter().map(|v| v.ln()).collect();
    let xm = x.iter().sum::<f64>() / x.len() as f64;
    let ym = y.iter().sum::<f64>() / y.len() as f64;
    let p = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - xm) * (b - ym))
        .sum::<f64>()
        / x.iter().map(|v| (v - xm).powi(2)).sum::<f64>();
    let r2 = 1.0
        - x.iter()
            .zip(&y)
            .map(|(a, b)| (b - ym - p * (a - xm)).powi(2))
            .sum::<f64>()
            / y.iter().map(|v| (v - ym).powi(2)).sum::<f64>();
    (p, r2)
}

#[test]
fn periodic_subspace_ignores_open_vectors_and_preserves_skew_images() {
    for axes in [
        [true, false, false],
        [false, true, false],
        [false, false, true],
        [true, true, false],
        [true, false, true],
        [false, true, true],
        [true, true, true],
    ] {
        let lat = Lattice::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(3.6, 1.8, 0.0),
            Vec3::new(0.5, 0.3, 4.0),
        );
        let g = PeriodicGeometry::new(lat.clone(), PeriodicBoundary::new(axes)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if axes[i] && axes[j] {
                    assert!(
                        (lat.a[i].dot(g.reciprocal()[j])
                            - if i == j {
                                2.0 * std::f64::consts::PI
                            } else {
                                0.0
                            })
                        .abs()
                            < 1e-12
                    );
                }
            }
        }
        let r = Vec3::new(8.3, -2.7, 3.4);
        assert!((g.perpendicular(g.wrap(r)) - g.perpendicular(r)).norm() < 1e-12);
        let minimum = g.minimum_image(r).unwrap();
        let mut direct = r.norm();
        let mut count = 0;
        for i in -15i64..=15 {
            for j in -15i64..=15 {
                for k in -15i64..=15 {
                    if (!axes[0] && i != 0) || (!axes[1] && j != 0) || (!axes[2] && k != 0) {
                        continue;
                    }
                    let t = lat.translation([i, j, k]);
                    direct = direct.min((r - t).norm());
                    if t.norm() <= 7.0 {
                        count += 1;
                    }
                }
            }
        }
        assert!((minimum.norm() - direct).abs() < 1e-12);
        assert_eq!(g.translations_within(7.0).unwrap().len(), count);
        let mesh = std::array::from_fn(|i| if axes[i] { 3 } else { 1 });
        let points = g.k_points(mesh).unwrap();
        assert_eq!(points.len(), 3usize.pow(g.dimension() as u32));
        for k in points {
            assert!(g.perpendicular(k).norm() < 1e-12);
        }
        if g.dimension() < 3 {
            let mut changed = lat;
            for i in 0..3 {
                if !axes[i] {
                    changed.a[i] = Vec3::zero();
                }
            }
            let alternate = PeriodicGeometry::new(changed, PeriodicBoundary::new(axes)).unwrap();
            assert!((alternate.measure() - g.measure()).abs() < 1e-12);
            assert_eq!(alternate.translations_within(7.0).unwrap().len(), count);
        }
    }
}

#[test]
fn wire_alternating_chain_and_3d_background_match_closed_existing_oracles() {
    let wire = PeriodicGeometry::new(
        Lattice::new(Vec3::new(5.0, 0.0, 0.0), Vec3::zero(), Vec3::zero()),
        PeriodicBoundary::new([true, false, false]),
    )
    .unwrap();
    let sources = [
        point(Vec3::zero(), 1.0),
        point(Vec3::new(2.5, 0.0, 0.0), -1.0),
    ];
    let result = periodic_electrostatics(&wire, &sources, &Default::default()).unwrap();
    assert!(result.converged);
    assert!((result.energy + 4.0 * 2.0_f64.ln() / 5.0).abs() < 1e-11);
    assert!(result.forces.iter().all(|f| f.norm() < 1e-12));
    let g = geometry(3);
    let original_mesh = dft_rs::periodic::integrals::monkhorst_pack(g.lattice(), [3, 2, 1]);
    let new_mesh = g.k_points([3, 2, 1]).unwrap();
    for (a, b) in original_mesh.iter().zip(new_mesh) {
        assert!((*a - b).norm() < 1e-14);
    }
    let sources = [
        point(Vec3::new(0.2, 0.3, -0.1), 1.2),
        point(Vec3::new(1.1, -0.2, 0.6), -0.3),
    ];
    let result = periodic_electrostatics(
        &g,
        &sources,
        &ElectrostaticOptions {
            compensation: ChargeCompensation::UniformBackground,
            ..Default::default()
        },
    )
    .unwrap();
    let positions: Vec<_> = sources.iter().map(|s| s.position).collect();
    let charges: Vec<_> = sources.iter().map(|s| s.charge).collect();
    let expected =
        dft_rs::periodic::ewald::ewald_energy(g.lattice(), &positions, &charges, result.params);
    let forces =
        dft_rs::periodic::ewald::ewald_forces(g.lattice(), &positions, &charges, result.params);
    assert!((result.energy - expected).abs() < 1e-12);
    for (a, b) in result.forces.iter().zip(forces) {
        assert!((*a - b).norm() < 1e-12);
    }
}

#[test]
fn charged_gaussian_compensation_matches_independent_fast_real_space_sum() {
    // Co-centered point + opposite Gaussian has no dipole and a rapidly
    // converging real-space energy. This oracle has no Ewald decomposition.
    for d in 1..=3 {
        let g = geometry(d);
        let width = 1.15;
        let q = -1.7;
        let sources = [point(Vec3::new(0.2, 0.4, 0.6), q)];
        let result = periodic_electrostatics(
            &g,
            &sources,
            &ElectrostaticOptions {
                compensation: ChargeCompensation::GaussianCountercharge {
                    center: sources[0].position,
                    width,
                },
                ..Default::default()
            },
        )
        .unwrap();
        let mut oracle =
            q * q / std::f64::consts::PI.sqrt() / width * (-2.0 + 1.0 / 2.0_f64.sqrt());
        for i in -10i64..=10 {
            for j in -10i64..=10 {
                for k in -10i64..=10 {
                    if i == 0 && j == 0 && k == 0 || d < 3 && k != 0 || d < 2 && j != 0 {
                        continue;
                    }
                    let r = g.lattice().translation([i, j, k]).norm();
                    oracle += 0.5 * q * q / r
                        * (2.0 * libm::erfc(r / width) - libm::erfc(r / (2.0_f64.sqrt() * width)));
                }
            }
        }
        eprintln!(
            "{d}D Gaussian charged energy {:.14e}, direct oracle {oracle:.14e}",
            result.energy
        );
        assert!(result.converged);
        assert!((result.energy - oracle).abs() < 1e-10);
        let potential_identity =
            0.5 * (q * result.site_potentials[0] - q * result.compensation_potential.unwrap());
        assert!((potential_identity - result.energy).abs() < 1e-12);
        assert!(result.forces[0].norm() < 1e-12);
    }
}

#[test]
fn all_dimensions_neutral_and_charged_forces_have_independent_four_step_fits() {
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for d in 1..=3 {
        for charged in [false, true] {
            let g = geometry(d);
            let sources = if charged {
                vec![point(Vec3::new(0.3, -0.4, 0.6), 1.2)]
            } else {
                vec![
                    point(Vec3::new(0.3, -0.4, 0.6), 1.2),
                    point(Vec3::new(1.1, 0.7, -0.2), -1.2),
                ]
            };
            let options = ElectrostaticOptions {
                compensation: if charged {
                    ChargeCompensation::GaussianCountercharge {
                        center: Vec3::new(-0.4, 0.8, -0.1),
                        width: 1.3,
                    }
                } else {
                    ChargeCompensation::RequireNeutral
                },
                ..Default::default()
            };
            let report = periodic_electrostatics(&g, &sources, &options).unwrap();
            assert!(report.converged);
            let exact = -report.forces[0].z;
            let estimates: Vec<_> = steps
                .iter()
                .map(|&h| {
                    let mut plus = sources.clone();
                    plus[0].position.z += h;
                    let mut minus = sources.clone();
                    minus[0].position.z -= h;
                    (periodic_electrostatics(&g, &plus, &options).unwrap().energy
                        - periodic_electrostatics(&g, &minus, &options)
                            .unwrap()
                            .energy)
                        / (2.0 * h)
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|v| (v - exact).abs()).collect();
            let (p, r2) = fit(&steps, &errors);
            eprintln!("{d}D charged={charged} force errors {errors:?}; h=0.02..0.0025 Bohr p={p:.12} R2={r2:.12}");
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
            for pair in errors.windows(2) {
                assert!((pair[0] / pair[1] / 4.0 - 1.0).abs() < 0.03);
            }
            assert!(((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs() < 1e-9);
            assert!(report.force_sum_defect < 1e-13);
            let mut moved = sources.clone();
            let shift = Vec3::new(0.6, -0.2, 1.3);
            for s in &mut moved {
                s.position += shift;
            }
            let mut moved_options = options.clone();
            if let ChargeCompensation::GaussianCountercharge { center, width } =
                options.compensation
            {
                moved_options.compensation = ChargeCompensation::GaussianCountercharge {
                    center: center + shift,
                    width,
                };
            }
            let shifted = periodic_electrostatics(&g, &moved, &moved_options).unwrap();
            assert!((shifted.energy - report.energy).abs() < 1e-12);
            // Changing the Ewald partition must not change the physical operator.
            let mut altered = options.clone();
            altered.alpha = Some(report.params.alpha * 0.73);
            let other = periodic_electrostatics(&g, &sources, &altered).unwrap();
            assert!((other.energy - report.energy).abs() < 1e-10);
            for (a, b) in report.forces.iter().zip(&other.forces) {
                assert!((*a - *b).norm() < 1e-10);
            }
        }
    }
}

#[test]
fn open_boundary_sizes_and_input_compatibility_are_physical() {
    for d in [1, 2] {
        let g = geometry(d);
        let mut lat = g.lattice().clone();
        for i in d..3 {
            lat.a[i] = lat.a[i] * 17.0;
        }
        let other = PeriodicGeometry::new(lat, g.boundary()).unwrap();
        let sources = [point(Vec3::new(0.3, -0.4, 0.6), 1.0)];
        let options = ElectrostaticOptions {
            compensation: ChargeCompensation::GaussianCountercharge {
                center: Vec3::new(0.7, 0.8, -0.1),
                width: 1.1,
            },
            ..Default::default()
        };
        let a = periodic_electrostatics(&g, &sources, &options).unwrap();
        let b = periodic_electrostatics(&other, &sources, &options).unwrap();
        assert_eq!(a.energy, b.energy);
        assert!(periodic_electrostatics(&g, &sources, &Default::default()).is_err());
        assert!(periodic_electrostatics(
            &g,
            &sources,
            &ElectrostaticOptions {
                compensation: ChargeCompensation::UniformBackground,
                ..Default::default()
            }
        )
        .is_err());
        assert!(g.k_points([2, 2, 2]).is_err());
    }
    assert!(PeriodicGeometry::new(
        Lattice::new(Vec3::zero(), Vec3::zero(), Vec3::zero()),
        PeriodicBoundary::FULL_3D
    )
    .is_err());
    let g = geometry(3);
    let duplicated = [point(Vec3::zero(), 1.0), point(Vec3::zero(), -1.0)];
    assert!(periodic_electrostatics(&g, &duplicated, &Default::default()).is_err());
}

#[test]
fn dimensional_countercharges_preserve_rotation_and_supercell_extensivity() {
    let rotation = |r: Vec3| Vec3::new(0.8 * r.x - 0.6 * r.z, r.y, 0.6 * r.x + 0.8 * r.z);
    for d in 1..=3 {
        let g = geometry(d);
        let sources = [
            point(Vec3::new(0.3, -0.4, 0.6), 1.2),
            ElectrostaticSource {
                position: Vec3::new(-0.4, 0.8, -0.1),
                charge: -1.2,
                width: 1.3,
            },
        ];
        let options = ElectrostaticOptions::default();
        let base = periodic_electrostatics(&g, &sources, &options).unwrap();
        let a = g.lattice().a;
        let rotated = PeriodicGeometry::new(
            Lattice::new(rotation(a[0]), rotation(a[1]), rotation(a[2])),
            g.boundary(),
        )
        .unwrap();
        let rotated_sources: Vec<_> = sources
            .iter()
            .map(|s| ElectrostaticSource {
                position: rotation(s.position),
                ..s.clone()
            })
            .collect();
        let other = periodic_electrostatics(&rotated, &rotated_sources, &options).unwrap();
        assert!((base.energy - other.energy).abs() < 1e-11);
        for (&x, &y) in base.forces.iter().zip(&other.forces) {
            assert!((rotation(x) - y).norm() < 1e-11);
        }
        // Replicate the actual countercharge along with the supplied charge.
        // Replacing many countercharges by one wider Gaussian changes physics.
        let super_geometry =
            PeriodicGeometry::new(g.lattice().supercell([2, 1, 1]), g.boundary()).unwrap();
        let mut repeated = sources.to_vec();
        repeated.extend(sources.iter().map(|s| ElectrostaticSource {
            position: s.position + a[0],
            ..s.clone()
        }));
        let super_result = periodic_electrostatics(&super_geometry, &repeated, &options).unwrap();
        assert!((super_result.energy / 2.0 - base.energy).abs() < 1e-10);
        for i in 0..2 {
            assert!((base.forces[i] - super_result.forces[i]).norm() < 1e-10);
        }
    }
}
