// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent acceptance of hybrid/ECP band operators, all seven crystal
//! systems and the causal intraband finite-frequency Hall derivative.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod};
use dft_rs::driver::{run, Controls, System, ToValue, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::band_transport::{
    frequency_transport, BandModel, HallIntegrals, TransportIntegrals,
};
use dft_rs::periodic::ewald::{ewald_energy, ewald_forces, EwaldParams};
use dft_rs::periodic::scf::{reference_fock_at, run_periodic_scf};
use dft_rs::periodic::Lattice;
use dft_rs::response::complex::Complex;

fn controls(xc: &str, extra: &[(&str, Value)]) -> Controls {
    let mut values = Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("k_mesh", vec![2_i64; 3])
        .with("overlap_cutoff", 10.0)
        .with("exchange_cutoff", 5.5)
        .with("grid_level", 2_i64)
        .with("grid_shell", 1_i64)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-10)
        .with("max_iter", 300_i64);
    for (key, value) in extra {
        values.set(key, value.clone());
    }
    Controls::from_value(&values).unwrap()
}

fn check_bands(label: &str, system: &System, c: &Controls, steps: [f64; 4]) {
    let method = PeriodicMethod::from_controls(c).unwrap();
    let setup = prepare_periodic(system, &method).unwrap();
    let scf = run_periodic_scf(&setup.cell, &setup.functional, &setup.options).unwrap();
    assert!(scf.converged, "{label} SCF");
    let model = BandModel::new(&setup.cell, &setup.functional, &scf, &setup.options).unwrap();
    println!(
        "{label}: Fock image residual {:.12e}",
        model.fock_image_residual
    );
    assert!(model.fock_image_residual < 1e-10, "{label} Fock images");
    let k = Vec3::new(0.17, 0.11, 0.08);
    let raw = Vec3::new(0.8, 0.5, 0.33);
    let u = raw * (1.0 / raw.norm());
    // The independent field path builds its XC potentials directly on the
    // shifted AO grid, rather than Fourier transforming XC potential blocks.
    let mut points = vec![k];
    for h in steps {
        points.extend([k + u * h, k + u * (-h)]);
    }
    let independent = reference_fock_at(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &points,
    )
    .unwrap();
    for spin in 0..model.spins {
        let mut oracle = Vec::new();
        for (ik, point) in points.iter().enumerate() {
            let metric = dft_rs::periodic::BlochMatrix::from_pair(setup.cell.overlap(*point));
            let bands =
                dft_rs::dfpt::orbitals::generalized_orbitals(&independent[spin][ik], &metric, 1e-7)
                    .unwrap();
            let computed = model.bands(spin, *point).unwrap();
            for (e, exact) in computed.energies.iter().zip(&bands.energies) {
                assert!((e - exact).abs() < 1e-10, "{label} shifted energy oracle");
            }
            oracle.push(bands.energies[0]);
        }
        let d = model.derivatives(spin, k, 0, Some(raw), 1e-8).unwrap();
        let first = d.branches[0].velocity;
        let second = d.branches[0].second_derivative;
        let mut e1 = Vec::new();
        let mut e2 = Vec::new();
        for (i, h) in steps.iter().enumerate() {
            e1.push(((oracle[2 * i + 1] - oracle[2 * i + 2]) / (2.0 * h) - first).abs());
            e2.push(
                ((oracle[2 * i + 1] - 2.0 * oracle[0] + oracle[2 * i + 2]) / (h * h) - second)
                    .abs(),
            );
        }
        for (name, errors) in [("velocity", e1), ("curvature", e2)] {
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "{label} spin {spin} {name}: steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",
                fit.exponent, fit.r_squared
            );
            assert!(
                (fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999,
                "{label} {name}"
            );
        }
        let reciprocal = setup.cell.lattice().reciprocal();
        let e0 = model.bands(spin, k).unwrap().energies;
        let shifted = model.bands(spin, k + reciprocal[0]).unwrap().energies;
        for (a, b) in e0.iter().zip(shifted) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

#[test]
fn full_and_range_separated_hybrids_and_ecp_use_the_actual_scf_operator() {
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[0.0; 3], [1.6, 0.1, 0.05]],
        cell: Some([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    };
    for xc in ["hf", "pbe0", "lc-wpbe"] {
        check_bands(
            xc,
            &system,
            &controls(xc, &[("k_mesh", vec![4_i64, 1, 1].to_value())]),
            [0.04, 0.02, 0.01, 0.005],
        );
    }
    let sodium = System {
        numbers: vec![11],
        positions: vec![[0.0; 3]],
        cell: Some([[14.0, 0.0, 0.0], [0.2, 20.0, 0.0], [0.1, 0.3, 20.0]]),
        pbc: [true; 3],
    };
    for xc in ["svwn5", "pbe0"] {
        check_bands(
            &format!("Na ECP {xc}"),
            &sodium,
            &controls(
                xc,
                &[
                    ("basis", "lanl2dz".to_value()),
                    ("multiplicity", 2_i64.to_value()),
                    ("k_mesh", vec![2_i64, 1, 1].to_value()),
                    ("overlap_cutoff", 48.0.to_value()),
                ],
            ),
            [0.01, 0.005, 0.0025, 0.00125],
        );
    }
}

#[test]
fn diffuse_ecp_bloch_metric_requires_a_resolved_image_interval() {
    use dft_rs::basis::{data, MolBasis};
    use dft_rs::periodic::integrals::{bloch, image_blocks, Operator};
    use dft_rs::system::{Atom, Molecule};
    let mol = Molecule::new(vec![Atom {
        z: 11,
        position: Vec3::zero(),
    }])
    .with_multiplicity(2);
    let basis = MolBasis::build(&mol, &data::load("lanl2dz").unwrap()).unwrap();
    let lattice = Lattice::new(
        Vec3::new(14.0, 0.0, 0.0),
        Vec3::new(0.2, 20.0, 0.0),
        Vec3::new(0.1, 0.3, 20.0),
    );
    let n = basis.nao_sph;
    let k = Vec3::new(0.17, 0.11, 0.08);
    let mut minima = Vec::new();
    for cutoff in [10.0, 18.0, 24.0, 36.0, 48.0, 60.0] {
        let blocks = image_blocks(&basis, &lattice, cutoff, Operator::Overlap);
        let (re, im) = bloch(&blocks, k);
        let mut real = dft_rs::linalg::Matrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                real[(i, j)] = re[(i, j)];
                real[(n + i, n + j)] = re[(i, j)];
                real[(i, n + j)] = -im[(i, j)];
                real[(n + i, j)] = im[(i, j)];
            }
        }
        let eigen = dft_rs::linalg::symmetric_eigen(&real).unwrap().0;
        println!(
            "Na diffuse metric cutoff={cutoff} min eigenvalue={:.12e}",
            eigen[0]
        );
        minima.push(eigen[0]);
    }
    // The 36-to-48 interval fails the original 1e-7 convergence bound.
    // Retain its raw values and verify a larger interval without changing it.
    assert!(minima[4] > 0.0 && minima[5] > 0.0);
    assert!((minima[4] - minima[5]).abs() < 1e-7);
}

#[test]
fn all_seven_crystal_systems_preserve_lattice_electrostatics_and_band_derivatives() {
    let root3 = 3.0_f64.sqrt();
    let cells = [
        ("cubic", [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
        (
            "tetragonal",
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        ),
        (
            "orthorhombic",
            [[4.8, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 6.0]],
        ),
        (
            "hexagonal",
            [[5.0, 0.0, 0.0], [-2.5, 2.5 * root3, 0.0], [0.0, 0.0, 6.0]],
        ),
        (
            "trigonal",
            [
                [5.0, 0.0, 0.0],
                [1.0, 24.0_f64.sqrt(), 0.0],
                [1.0, 4.0 / 24.0_f64.sqrt(), (24.0 - 16.0 / 24.0_f64).sqrt()],
            ],
        ),
        (
            "monoclinic",
            [[5.0, 0.0, 0.0], [0.0, 5.5, 0.0], [1.2, 0.0, 6.0]],
        ),
        (
            "triclinic",
            [[4.8, 0.0, 0.0], [0.7, 5.2, 0.0], [0.4, 0.6, 5.8]],
        ),
    ];
    for (name, rows) in cells {
        let v = |r: [f64; 3]| Vec3::new(r[0], r[1], r[2]);
        let lattice = Lattice::new(v(rows[0]), v(rows[1]), v(rows[2]));
        let reciprocal = lattice.reciprocal();
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (lattice.a[a].dot(reciprocal[b])
                        - if a == b {
                            2.0 * std::f64::consts::PI
                        } else {
                            0.0
                        })
                    .abs()
                        < 1e-13
                );
            }
        }
        let positions = [
            Vec3::zero(),
            lattice.a[0] * 0.23 + lattice.a[1] * 0.13 + lattice.a[2] * 0.21,
        ];
        for charges in [[1.0, -1.0], [1.0, -0.6]] {
            let p = EwaldParams::for_accuracy(&lattice, 2, 1e-14);
            let energy = ewald_energy(&lattice, &positions, &charges, p);
            let q = EwaldParams {
                alpha: p.alpha * 1.3,
                rcut: p.rcut / 1.3,
                gcut: p.gcut * 1.3,
            };
            let split_defect = (energy - ewald_energy(&lattice, &positions, &charges, q)).abs();
            assert!(split_defect < 1e-12, "{name} Ewald splitting");
            let mut shifted = positions;
            shifted[1] += lattice.a[1];
            assert!((energy - ewald_energy(&lattice, &shifted, &charges, p)).abs() < 1e-12);
            let gradient = -ewald_forces(&lattice, &positions, &charges, p)[1].x;
            let steps = [0.04, 0.02, 0.01, 0.005];
            let errors: Vec<_> = steps
                .iter()
                .map(|h| {
                    let mut plus = positions;
                    let mut minus = positions;
                    plus[1].x += h;
                    minus[1].x -= h;
                    ((ewald_energy(&lattice, &plus, &charges, p)
                        - ewald_energy(&lattice, &minus, &charges, p))
                        / (2.0 * h)
                        - gradient)
                        .abs()
                })
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            println!(
                "{name} charges={charges:?} Ewald splitting={split_defect:.9e}, \
                steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",
                fit.exponent, fit.r_squared
            );
            assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        }
        let system = System {
            numbers: vec![2],
            positions: vec![[0.0; 3]],
            cell: Some(rows),
            pbc: [true; 3],
        };
        check_bands(
            name,
            &system,
            &controls("svwn5", &[]),
            [0.04, 0.02, 0.01, 0.005],
        );
        let c = controls(
            "svwn5",
            &[
                ("transport_k_mesh", vec![6_i64; 3].to_value()),
                ("transport_temperatures_k", vec![2000.0].to_value()),
                ("carrier_concentrations_cm3", vec![-1e22].to_value()),
                ("delta_window", 40.0.to_value()),
                (
                    "transport_frequencies_hartree",
                    vec![0.0, 0.02, -0.02].to_value(),
                ),
                ("relaxation_time_fs", 10.0.to_value()),
            ],
        );
        let hall = run("hall-transport", &system, &c).unwrap();
        let row = &hall.get("results").unwrap().as_array().unwrap()[0];
        println!(
            "{name} Hall raw Onsager={:?} relative={:?} R={:?}",
            row.get("raw_hall_onsager_residual"),
            row.get("relative_hall_onsager_residual"),
            row.get("hall_coefficient")
        );
        assert!(hall.get("fock_image_residual").unwrap().as_f64().unwrap() < 1e-10);
        assert!(row
            .get("hall_coefficient")
            .unwrap()
            .numeric_array()
            .is_some());
        assert_eq!(
            row.get("frequency_results")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
}

#[test]
fn causal_frequency_hall_passes_complex_drude_resolvent_four_step_fits() {
    let masses = [0.7, 1.1, 1.8];
    let density = 0.002;
    let l = TransportIntegrals {
        l0: std::array::from_fn(|a| {
            std::array::from_fn(|b| if a == b { density / masses[a] } else { 0.0 })
        }),
        l1: [[0.0; 3]; 3],
        l2: [[0.0; 3]; 3],
        electron_density: density,
    };
    let mut h = [[[0.0; 3]; 3]; 3];
    for c in 0..3 {
        let a = (c + 1) % 3;
        let b = (c + 2) % 3;
        h[a][b][c] = -density / (masses[a] * masses[b]);
        h[b][a][c] = -h[a][b][c];
    }
    let hall = HallIntegrals {
        conductivity_derivative_over_tau_squared: h,
        raw_onsager_residual: 0.0,
        relative_onsager_residual: Some(0.0),
    };
    let tau = 0.7;
    let steps = [0.08, 0.04, 0.02, 0.01];
    for omega in [0.0, 0.4, -0.4, 1.2] {
        let ac = frequency_transport(&l, Some(&hall), omega, tau).unwrap();
        let negative = frequency_transport(&l, Some(&hall), -omega, tau).unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert!((ac.conductivity[a][b].conj() - negative.conductivity[a][b]).abs() < 1e-14);
            }
        }
        for c in 0..3 {
            let a = (c + 1) % 3;
            let b = (c + 2) % 3;
            let exact = ac.hall_conductivity_derivative.unwrap()[a][b][c];
            // Exact inversion of the finite-field anisotropic Drude
            // equation: (tau^-1-i omega) m v + v cross B = -E.
            let z = Complex::new(1.0 / tau, -omega);
            let estimates: Vec<_> = steps
                .iter()
                .map(|bfield| {
                    -Complex::from(density)
                        / (z * z * (masses[a] * masses[b]) + Complex::from(bfield * bfield))
                })
                .collect();
            let errors: Vec<_> = estimates.iter().map(|x| (*x - exact).abs()).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let richardson = (estimates[3] * 4.0 - estimates[2]) / 3.0;
            println!(
                "AC Hall omega={omega} B axis={c} steps={steps:?} errors={errors:?} \
                p={:.9} R2={:.10} Richardson={:.9e}",
                fit.exponent,
                fit.r_squared,
                (richardson - exact).abs()
            );
            assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
            assert!((richardson - exact).abs() / exact.abs() < 1e-6);
        }
    }
    assert!(frequency_transport(&l, None, 0.0, 0.0).is_err());
    assert!(frequency_transport(&l, None, f64::INFINITY, 1.0).is_err());
}

#[test]
fn finite_frequency_common_driver_keeps_static_results_and_complex_units() {
    let system = System {
        numbers: vec![2],
        positions: vec![[0.0; 3]],
        cell: Some([[0.3, 3.1, 2.8], [2.9, 0.2, 3.2], [3.0, 2.7, 0.1]]),
        pbc: [true; 3],
    };
    let c = controls(
        "pbe0",
        &[
            ("transport_k_mesh", vec![4_i64; 3].to_value()),
            ("transport_temperatures_k", vec![2000.0].to_value()),
            ("carrier_concentrations_cm3", vec![-1e22].to_value()),
            (
                "transport_frequencies_hartree",
                vec![0.0, 0.02, -0.02].to_value(),
            ),
            ("relaxation_time_fs", 10.0.to_value()),
        ],
    );
    let report = run("hall-transport", &system, &c).unwrap();
    let row = &report.get("results").unwrap().as_array().unwrap()[0];
    let frequency = row.get("frequency_results").unwrap().as_array().unwrap();
    let (_, dc) = row
        .get("conductivity_over_tau")
        .unwrap()
        .numeric_array()
        .unwrap();
    let (shape, zero) = frequency[0]
        .get("conductivity")
        .unwrap()
        .numeric_array()
        .unwrap();
    assert_eq!(shape, vec![3, 3, 2]);
    for i in 0..9 {
        assert!((zero[2 * i] - dc[i] * 1e-14).abs() < 1e-12 * zero[2 * i].abs());
        assert_eq!(zero[2 * i + 1], 0.0);
    }
    assert_eq!(
        frequency[1]
            .get("hall_conductivity_derivative")
            .unwrap()
            .numeric_array()
            .unwrap()
            .0,
        vec![3, 3, 3, 2]
    );
    assert!(report
        .get("units")
        .unwrap()
        .get("results.frequency_results.conductivity")
        .is_some());
}
