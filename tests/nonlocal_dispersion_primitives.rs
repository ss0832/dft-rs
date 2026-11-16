// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use dft_rs::basis::{data, MolBasis};
use dft_rs::dispersion::{mbd, ts::FreeAtom};
use dft_rs::grid::{angular_rule, gauss_legendre, AngularScheme, GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::nlc::vdw_df::{
    unpolarized_local_scale, InternalExchange, LocalDensity, MolecularPairGrid,
    PairKernelQuadrature, PairKernelTable, Variant, WeightedDensity,
};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::f64::consts::PI;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.16e} vs {expected:.16e} (absolute tolerance {tolerance:e})"
    );
}

#[test]
fn vd_w_df_direct_kernel_matches_unfactored_dion_formula_and_tail() {
    let cutoff = 192.0;
    let quadrature = PairKernelQuadrature::new(cutoff, 8, 0.01).unwrap();
    let (nodes, weights) = gauss_legendre(8);
    let points: Vec<_> = (0..612)
        .flat_map(|panel| {
            nodes.iter().zip(&weights).map(move |(&node, &weight)| {
                (
                    (panel as f64 + 0.5 * (1.0 + node)) * cutoff / 612.0,
                    0.5 * weight * cutoff / 612.0,
                )
            })
        })
        .collect();
    fn moment(a: f64, b: f64, d: f64, dp: f64) -> f64 {
        // Direct equations (14)–(16) before extracting sinc combinations.
        let (sa, ca) = a.sin_cos();
        let (sb, cb) = b.sin_cos();
        let angular = ((3.0 - a * a) * b * cb * sa
            + (3.0 - b * b) * a * ca * sb
            + (a * a + b * b - 3.0) * sa * sb
            - 3.0 * a * b * ca * cb)
            / (a * b);
        let nu = |y: f64, length: f64| {
            0.5 * y * y / (1.0 - (-4.0 * PI * (y / length).powi(2) / 9.0).exp())
        };
        let (w, x, y, z) = (nu(a, d), nu(b, d), nu(a, dp), nu(b, dp));
        let temporal = (1.0 / (w + x) + 1.0 / (y + z))
            * (1.0 / ((w + y) * (x + z)) + 1.0 / ((w + z) * (x + y)));
        2.0 * angular * temporal / PI.powi(2)
    }
    for (d, dp) in [(0.25, 0.8), (2.0, 3.0), (5.0, 7.0)] {
        let oracle = points
            .iter()
            .map(|&(a, wa)| {
                wa * points
                    .iter()
                    .map(|&(b, wb)| wb * moment(a, b, d, dp))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let actual = quadrature.evaluate(d, dp).unwrap();
        close(actual, oracle, 2e-9);
        close(actual, quadrature.evaluate(dp, d).unwrap(), 1e-15);
        println!(
            "vdW-DF pair ({d}, {dp}) production={actual:.15e} unfactored={oracle:.15e} Hartree"
        );
    }
    let fine = PairKernelQuadrature::new(768.0, 8, 0.01).unwrap();
    let (d, dp) = (18.0_f64, 24.0_f64);
    let asymptote = -12.0 * (4.0 * PI / 9.0).powi(3) / (d.powi(2) * dp.powi(2) * (d * d + dp * dp));
    let actual = fine.evaluate(d, dp).unwrap();
    close(actual, asymptote, 3e-12);
    println!(
        "vdW-DF far pair ({d}, {dp}) production={actual:.15e} asymptote={asymptote:.15e} Hartree"
    );
    assert!(fine.evaluate(0.0, dp).is_err());
}

#[test]
fn vd_w_df_and_df2_differ_by_the_published_internal_gradient() {
    let rho = 0.04;
    let sigma = 0.021;
    let dion = |gradient| unpolarized_local_scale(rho, gradient, InternalExchange::Dion).unwrap();
    let lee = |gradient| unpolarized_local_scale(rho, gradient, InternalExchange::Lee).unwrap();
    close(dion(0.0), lee(0.0), 0.0);
    let kf = (3.0 * PI * PI * rho).cbrt();
    let expected = (1.887 - 0.8491) * sigma / (36.0 * kf * rho * rho);
    close(lee(sigma) - dion(sigma), expected, 3e-15);
    println!(
        "vdW-DF q0={:.12e}, vdW-DF2 q0={:.12e} Bohr^-1",
        dion(sigma),
        lee(sigma)
    );
}

#[test]
fn vd_w_df_families_use_distinct_published_exchange_partners() {
    let formulas = [
        Variant::VdwDf,
        Variant::VdwDf2,
        Variant::OptPbe,
        Variant::OptB88,
        Variant::OptB86b,
        Variant::Cx,
        Variant::RevVdwDf2,
    ];
    for s in [0.0_f64, 0.2, 1.5, 3.0, 60.0] {
        let s2 = s.powi(2);
        let x = 0.175519 * s2 / 1.04804;
        let pw =
            (1.0 + 15.0 * 0.1234 * s2 + 17.33 * s.powi(4) + 0.163 * s.powi(6)).powf(1.0 / 15.0);
        let expected = [
            1.0 + 0.2195149727645171 * s2 / (1.0 + 0.2195149727645171 * s2 / 1.245),
            pw,
            1.0 + 1.04804 * (0.945268 * x / (1.0 + x) - 0.054732 * (-x).exp_m1()),
            1.0 + 0.22 * s2
                / (1.0
                    + (0.22 / 1.2)
                        * s
                        * (2.0_f64.powf(4.0 / 3.0) * (3.0 * PI * PI).cbrt() * s).asinh()),
            1.0 + 0.1234 * s2 / (1.0 + 0.1234 * s2).powf(4.0 / 5.0),
            (1.0 + (0.8491 / 9.0) * s2) / (1.0 + 0.02178 * s.powi(6))
                + pw * (1.0 - 1.15 / (1.15 + 0.02178 * s.powi(6))),
            1.0 + (10.0 / 81.0) * s2 / (1.0 + (10.0 / 81.0) * s2 / 0.7114).powf(4.0 / 5.0),
        ];
        for (formula, value) in formulas.into_iter().zip(expected) {
            close(formula.exchange_enhancement(s).unwrap(), value, 8e-15);
        }
        if s == 1.5 {
            println!("vdW-DF seven outer exchange enhancement factors at s=1.5: {expected:?}");
        }
    }
}

#[test]
fn generated_vd_w_df_table_refines_to_reference_pair_quadrature() {
    let quadrature = PairKernelQuadrature::new(72.0, 8, 0.004).unwrap();
    let points = [
        (0.017, 0.21),
        (0.45, 0.72),
        (1.32, 1.84),
        (2.3, 3.6),
        (6.7, 9.3),
    ];
    let reference: Vec<_> = points
        .iter()
        .map(|&(a, b)| quadrature.evaluate(a, b).unwrap())
        .collect();
    let mut errors = Vec::new();
    for count in [26, 43, 64] {
        let table = PairKernelTable::new(&quadrature, 0.01, 20.0, count).unwrap();
        let error = points
            .iter()
            .zip(&reference)
            .map(|(&(a, b), &expected)| {
                let actual = table.interpolate(a, b).unwrap();
                assert_eq!(actual, table.interpolate(b, a).unwrap());
                (actual - expected).abs()
            })
            .fold(0.0, f64::max);
        assert!(table.interpolate(0.005, 1.0).is_none());
        assert!(table.interpolate(1.0, 20.1).is_none());
        errors.push(error);
        println!("vdW-DF {count}-point log table: maximum pair error={error:.3e} Hartree");
    }
    assert!(errors[2] < 2e-5 && errors[2] < errors[0]);
}

#[test]
fn vd_w_df_molecular_energy_reports_all_named_family_components() {
    let molecule = Molecule::new(
        [
            (7, [0.13, -0.27, 0.19]),
            (1, [1.78, 0.34, -0.08]),
            (1, [-0.49, 1.65, 0.41]),
            (1, [0.31, -0.78, 1.91]),
        ]
        .into_iter()
        .map(|(z, [x, y, z_coord])| Atom {
            z,
            position: Vec3::new(x, y, z_coord),
        })
        .collect(),
    );
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: 0,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let plan = resolve(&FunctionalSpec::Combined("spw92".into())).unwrap();
    let scf = run_scf(
        &molecule,
        &basis,
        &ScfOptions::default(),
        &plan,
        Some(&grid),
    )
    .unwrap();
    assert!(scf.converged);
    let density = scf.p_total();
    let spatial = MolecularPairGrid::new(14, 9, 1.1, 1e-7).unwrap();
    let momentum = PairKernelQuadrature::new(48.0, 6, 0.004).unwrap();
    let table = PairKernelTable::new(&momentum, 1e-5, 1e6, 72).unwrap();
    let kernel = |d, dp| match table.interpolate(d, dp) {
        Some(value) => Ok(value),
        None => momentum.evaluate(d, dp),
    };
    let mut results = Vec::new();
    for model in [
        Variant::VdwDf,
        Variant::VdwDf2,
        Variant::OptPbe,
        Variant::OptB88,
        Variant::OptB86b,
        Variant::Cx,
        Variant::RevVdwDf2,
    ] {
        let r = spatial
            .xc_on_density(&grid, &basis, &density, model, kernel)
            .unwrap();
        assert!(
            r.exchange < 0.0 && r.local_correlation < 0.0 && r.nonlocal_correlation.is_finite()
        );
        assert!(r.total().is_finite());
        println!("NH3 SPW92/STO-3G fixed density {model:?}: Ex={:.12e} Ec(PW92)={:.12e} Enl={:.12e} EXC={:.12e} Hartree",
            r.exchange, r.local_correlation, r.nonlocal_correlation, r.total());
        results.push(r);
    }
    // The original kernel is shared by the five original-generation
    // variants. The revised internal gradient enters the other two.
    for i in [2, 3, 4, 5] {
        close(
            results[i].nonlocal_correlation,
            results[0].nonlocal_correlation,
            0.0,
        );
        assert!((results[i].exchange - results[0].exchange).abs() > 1e-4);
    }
    close(
        results[1].nonlocal_correlation,
        results[6].nonlocal_correlation,
        0.0,
    );
    assert!((results[1].nonlocal_correlation - results[0].nonlocal_correlation).abs() > 1e-5);
    for pair in results.windows(2) {
        close(pair[0].local_correlation, pair[1].local_correlation, 0.0);
    }
    // Independently refine the radial and angular integration around
    // each outer point on both generations at the same molecular density.
    let mut previous = [
        results[0].nonlocal_correlation,
        results[1].nonlocal_correlation,
    ];
    let mut largest_increment = f64::INFINITY;
    for (radial, degree) in [(24, 13), (32, 17), (44, 23)] {
        let grid_pair = MolecularPairGrid::new(radial, degree, 1.1, 1e-7).unwrap();
        let variants = [Variant::VdwDf, Variant::VdwDf2];
        let mut refined = [0.0; 2];
        for (i, model) in variants.into_iter().enumerate() {
            refined[i] = grid_pair
                .xc_on_density(&grid, &basis, &density, model, kernel)
                .unwrap()
                .nonlocal_correlation;
        }
        largest_increment = refined
            .iter()
            .zip(previous)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("NH3 vdW-DF/DF2 {radial} by degree-{degree} inner spatial samples: Enl={refined:?}, max increment={largest_increment:.3e} Hartree");
        previous = refined;
    }
    assert!(largest_increment < 5e-4);
}

#[test]
fn vd_w_df_spatial_coincidence_quadrature_matches_gaussian_convolution() {
    let model = InternalExchange::Dion;
    let q = 2.5_f64;
    let scale = 1.2_f64;
    let a = 0.15_f64;
    let b = 0.85_f64;
    let cutoff = 1e-13;
    let momentum = PairKernelQuadrature::new(72.0, 8, 0.004).unwrap();
    let density = |point: Vec3| {
        let n = a * (-b * point.norm2()).exp();
        let sigma = if n < cutoff {
            0.0
        } else {
            // Compensate the local density term in this manufactured fixture
            // so the value of q0 is constant while rho remains Gaussian.
            let local = unpolarized_local_scale(n, 0.0, model).unwrap();
            let kf = (3.0 * PI * PI * n).cbrt();
            (q - local) * 36.0 * kf * n * n / 0.8491
        };
        LocalDensity {
            density: n,
            gradient_squared: sigma,
        }
    };
    let radial = |x: f64| scale * (1.0 + x) / (1.0 - x);
    let outer_order = 30;
    let (nodes, weights) = gauss_legendre(outer_order);
    let angular = angular_rule(AngularScheme::Lebedev, 11);
    let mut outer = Vec::new();
    for (x, w) in nodes.into_iter().zip(weights) {
        let r = radial(x);
        let volume = 2.0 * scale * w * (r / (1.0 - x)).powi(2);
        outer.extend(angular.iter().map(|&(direction, weight)| {
            let position = direction * r;
            WeightedDensity {
                position,
                volume: volume * weight,
                local: density(position),
            }
        }));
    }
    // The Gaussian autocorrelation eliminates three spatial coordinates
    // exactly. An independently scaled finite radial quadrature integrates it.
    let (nodes, weights) = gauss_legendre(96);
    let integral = nodes
        .into_iter()
        .zip(weights)
        .map(|(x, w)| {
            let r = 4.0 * (1.0 + x);
            4.0 * w * r * r * (-0.5 * b * r * r).exp() * momentum.evaluate(q * r, q * r).unwrap()
        })
        .sum::<f64>();
    let expected = 2.0 * PI * a * a * (PI / (2.0 * b)).powf(1.5) * integral;
    let mut defect = f64::INFINITY;
    for order in [12, 18, 26, 36] {
        let inner = MolecularPairGrid::new(order, 17, scale, cutoff).unwrap();
        // For constant q0, evaluate each distinct radial momentum pair once.
        let mut pairs: Vec<_> = gauss_legendre(order)
            .0
            .into_iter()
            .map(|x| {
                let d = q * radial(x);
                (d, momentum.evaluate(d, d).unwrap())
            })
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cached_kernel = |d: f64, dp: f64| {
            close(d, dp, 1e-11 * d.max(dp));
            let i = pairs.partition_point(|(point, _)| *point < d * (1.0 - 1e-12));
            let (point, value) = pairs[i];
            close(d, point, 1e-11 * d);
            Ok(value)
        };
        let actual = inner
            .energy_on_samples(
                &outer,
                |points| points.iter().copied().map(density).collect(),
                model,
                cached_kernel,
            )
            .unwrap();
        let next_defect = (actual - expected).abs();
        assert!(next_defect < defect * 1.25);
        defect = next_defect;
        println!("vdW-DF Gaussian spatial integral {order} radial nodes: {actual:.12e}/{expected:.12e} Hartree, |error|={defect:.3e}");
    }
    assert!(defect < 1e-8);
}

#[test]
fn screened_mbd_dimer_matches_scalar_uniform_modes_and_casimir_polder() {
    let h = FreeAtom {
        polarizability: 4.5,
        c6: 6.5,
        radius: 3.099_178_844,
    };
    let distance = 4.2_f64;
    let beta = 0.81_f64;
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(distance, 0.0, 0.0),
        },
    ]);
    let reference_frequency = 4.0 * h.c6 / (3.0 * h.polarizability.powi(2));
    let short_range = 1.0 / (1.0 + (6.0 * (distance / (2.0 * beta * h.radius) - 1.0)).exp());
    let (nodes, weights) = gauss_legendre(96);
    let scalar_response = |omega: f64| {
        let a = h.polarizability / (1.0 + (omega / reference_frequency).powi(2));
        let spread = 2.0_f64.sqrt() * ((2.0 / PI).sqrt() * a / 3.0).cbrt();
        let x = distance / spread;
        let exponential = (-x * x).exp();
        let erf = x.min(12.0) / PI.sqrt()
            * nodes
                .iter()
                .zip(&weights)
                .map(|(&node, &weight)| {
                    weight * (-(0.5 * (1.0 + node) * x.min(12.0)).powi(2)).exp()
                })
                .sum::<f64>();
        let h = erf - 2.0 * x * exponential / PI.sqrt();
        let transverse = short_range * h / distance.powi(3);
        let longitudinal =
            short_range * (4.0 * x.powi(3) * exponential / PI.sqrt() - 2.0 * h) / distance.powi(3);
        (
            (a / (1.0 + a * longitudinal) + 2.0 * a / (1.0 + a * transverse)) / 3.0,
            1.0 - a * longitudinal.abs().max(transverse.abs()),
        )
    };
    let (screened_alpha, _) = scalar_response(0.0);
    let (angular_nodes, angular_weights) = gauss_legendre(128);
    let screened_c6 = 0.75
        * angular_nodes
            .iter()
            .zip(&angular_weights)
            .map(|(&x, &w)| {
                let theta = (1.0 + x) * PI / 4.0;
                w * (scalar_response(theta.tan()).0 / theta.cos()).powi(2)
            })
            .sum::<f64>();
    let screened_radius = h.radius * (screened_alpha / h.polarizability).cbrt();
    let screened_frequency = 4.0 * screened_c6 / (3.0 * screened_alpha.powi(2));
    let t = screened_alpha
        / (distance.powi(3)
            * (1.0 + (6.0 * (1.0 - distance / (2.0 * screened_radius * beta))).exp()));
    let frequencies = [
        1.0 - 2.0 * t,
        1.0 - t,
        1.0 - t,
        1.0 + t,
        1.0 + t,
        1.0 + 2.0 * t,
    ]
    .map(|x| screened_frequency * x.sqrt());
    let expected_energy = (0.5 * frequencies.into_iter().sum::<f64>()) - 3.0 * screened_frequency;
    for count in [24, 48, 96, 128] {
        let result =
            mbd::rsscs_energy_from_volumes(&molecule, &[1.0; 2], &[h; 2], beta, count, 0.8)
                .unwrap();
        let expected_minimum = std::iter::once(0.0)
            .chain(
                gauss_legendre(count)
                    .0
                    .into_iter()
                    .map(|x| 0.8 * (1.0 + x) / (1.0 - x)),
            )
            .map(|omega| scalar_response(omega).1)
            .fold(f64::INFINITY, f64::min);
        close(result.minimum_screening_eigenvalue, expected_minimum, 2e-14);
        assert!(result.screening_residual < 2e-14);
        for atom in &result.oscillators.atoms {
            close(atom.polarizability, screened_alpha, 2e-14);
            close(atom.c6, screened_c6, 4e-7);
            close(atom.radius, screened_radius, 2e-14);
        }
        for (&a, b) in result
            .oscillators
            .coupled_frequencies
            .iter()
            .zip(frequencies)
        {
            close(a, b, 3e-8);
        }
        close(result.oscillators.energy, expected_energy, 5e-11);
        println!("MBD@rsSCS H/H {count} nodes: E={:.15e}/{expected_energy:.15e} Hartree, alpha={screened_alpha:.12e}, C6={:.12e}/{screened_c6:.12e}, min={:.12e}, residual={:.3e}",
            result.oscillators.energy, result.oscillators.atoms[0].c6, result.minimum_screening_eigenvalue, result.screening_residual);
    }
}
