// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Occupation and physical AO-connection derivative fits, exact degenerate
//! multiplet covariance, causal pair equations and nonzero Chern oracles.

#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dfpt::orbitals::generalized_orbitals;
use dft_rs::driver::periodic::{prepare_periodic, PeriodicMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::math::Vec3;
use dft_rs::periodic::coherent_transport::{
    conductivity, occupation_divided_difference, quantum_conductance, quantum_hall_from_links,
    CoherentModel, CoherentPoint, CrossMetric,
};
use dft_rs::periodic::scf::run_periodic_scf;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;

fn identity(n: usize) -> BlochMatrix {
    let mut m = BlochMatrix::zeros(n, n);
    for i in 0..n {
        m.re[(i, i)] = 1.0;
    }
    m
}
fn z(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn max_difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    (0..a.rows())
        .flat_map(|i| (0..a.cols()).map(move |j| (i, j)))
        .map(|(i, j)| (z(a, i, j) - z(b, i, j)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn occupations_at_exact_degeneracy_pass_matrix_fermi_four_step_fit() {
    let energies = [0.0, 0.0, 0.27];
    let mu = 0.03;
    let kt = 0.09;
    let mut h = BlochMatrix::zeros(3, 3);
    for i in 0..3 {
        h.re[(i, i)] = energies[i];
    }
    let v = BlochMatrix::from_pair((
        dft_rs::linalg::Matrix::from_row_major(
            3,
            3,
            vec![0.12, 0.07, 0.03, 0.07, -0.08, 0.04, 0.03, 0.04, 0.02],
        ),
        dft_rs::linalg::Matrix::zeros(3, 3),
    ));
    let density = |lambda: f64| {
        let mut operator = h.clone();
        operator.add_scaled(&v, lambda);
        let bands = generalized_orbitals(&operator, &identity(3), 1e-12).unwrap();
        let mut out = BlochMatrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let value = (0..3)
                    .map(|n| {
                        z(&bands.coefficients, i, n)
                            * z(&bands.coefficients, j, n).conj()
                            * (1.0 / (1.0 + ((bands.energies[n] - mu) / kt).exp()))
                    })
                    .fold(Complex::ZERO, |a, b| a + b);
                out.re[(i, j)] = value.re;
                out.im[(i, j)] = value.im;
            }
        }
        out
    };
    let mut derivative = BlochMatrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            derivative.re[(i, j)] =
                -occupation_divided_difference(energies[i], energies[j], mu, kt).unwrap()
                    * v.re[(i, j)];
        }
    }
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut estimates = Vec::new();
    let mut errors = Vec::new();
    for h in steps {
        let mut estimate = density(h);
        estimate.add_scaled(&density(-h), -1.0);
        estimate.scale(1.0 / (2.0 * h));
        errors.push(max_difference(&estimate, &derivative));
        estimates.push(estimate);
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    let mut richardson = estimates[3].clone();
    richardson.scale(4.0 / 3.0);
    richardson.add_scaled(&estimates[2], -1.0 / 3.0);
    println!(
        "degenerate matrix Fermi steps={steps:?} errors={errors:?} p={:.9} \
        R2={:.10} Richardson={:.9e}",
        fit.exponent,
        fit.r_squared,
        max_difference(&richardson, &derivative)
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    assert!(max_difference(&richardson, &derivative) < 1e-9);
    let equal = occupation_divided_difference(0.0, 0.0, mu, kt).unwrap();
    assert!((occupation_divided_difference(0.0, 1e-14, mu, kt).unwrap() - equal).abs() < 1e-12);
}

fn rotated(m: &BlochMatrix, u: &BlochMatrix) -> BlochMatrix {
    let n = m.rows();
    let mut out = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut value = Complex::ZERO;
            for a in 0..n {
                for b in 0..n {
                    value += z(u, a, i).conj() * z(m, a, b) * z(u, b, j);
                }
            }
            out.re[(i, j)] = value.re;
            out.im[(i, j)] = value.im;
        }
    }
    out
}

#[test]
fn degenerate_coherences_are_covariant_and_retarded_with_no_false_flat_band_drude() {
    let mut velocities: [BlochMatrix; 3] = std::array::from_fn(|_| BlochMatrix::zeros(3, 3));
    for (axis, v) in velocities.iter_mut().enumerate() {
        for i in 0..3 {
            for j in i..3 {
                let re = 0.04 * (1 + axis + i + j) as f64;
                let im = if i == j {
                    0.0
                } else {
                    0.03 * (1 + axis + i + j) as f64
                };
                v.re[(i, j)] = re;
                v.re[(j, i)] = re;
                v.im[(i, j)] = im;
                v.im[(j, i)] = -im;
            }
        }
    }
    let point = CoherentPoint {
        energies: vec![0.0, 0.0, 0.4],
        velocities,
    };
    let mut u = identity(3);
    let theta = 0.71_f64;
    u.re[(0, 0)] = theta.cos();
    u.re[(1, 1)] = theta.cos();
    u.im[(0, 1)] = theta.sin();
    u.im[(1, 0)] = theta.sin();
    let transformed = CoherentPoint {
        energies: point.energies.clone(),
        velocities: std::array::from_fn(|a| rotated(&point.velocities[a], &u)),
    };
    for omega in [0.0, 0.13, -0.13] {
        let a = conductivity(
            std::slice::from_ref(&point),
            2.0,
            1,
            1.0,
            0.03,
            0.08,
            omega,
            0.02,
        )
        .unwrap();
        let b = conductivity(
            std::slice::from_ref(&transformed),
            2.0,
            1,
            1.0,
            0.03,
            0.08,
            omega,
            0.02,
        )
        .unwrap();
        let opposite = conductivity(
            std::slice::from_ref(&point),
            2.0,
            1,
            1.0,
            0.03,
            0.08,
            -omega,
            0.02,
        )
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((a.conductivity[i][j] - b.conductivity[i][j]).abs() < 1e-13);
                assert!((a.intraband[i][j] - b.intraband[i][j]).abs() < 1e-13);
                assert!((a.conductivity[i][j].conj() - opposite.conductivity[i][j]).abs() < 1e-13);
            }
        }
        assert!(a.equation_residual < 1e-15 && a.current_hermitian_residual < 1e-15);
    }
    // A flat-band model with only interband currents has no intraband Drude
    // weight. Its longitudinal DC term disappears in the clean limit.
    let mut flat = point.clone();
    for v in &mut flat.velocities {
        for i in 0..2 {
            for j in 0..2 {
                v.re[(i, j)] = 0.0;
                v.im[(i, j)] = 0.0;
            }
        }
        v.re[(2, 2)] = 0.0;
    }
    let a = conductivity(&[flat.clone()], 1.0, 1, 1.0, 0.2, 0.01, 0.0, 1e-3).unwrap();
    let b = conductivity(&[flat], 1.0, 1, 1.0, 0.2, 0.01, 0.0, 1e-6).unwrap();
    assert_eq!(a.intraband, [[Complex::ZERO; 3]; 3]);
    assert!(b.conductivity[0][0].abs() < a.conductivity[0][0].abs() * 0.00101);
}

fn lower(kx: f64, ky: f64) -> BlochMatrix {
    let mut h = BlochMatrix::zeros(2, 2);
    let d = kx.cos() + ky.cos() - 1.0;
    h.re[(0, 0)] = d;
    h.re[(1, 1)] = -d;
    h.re[(0, 1)] = kx.sin();
    h.re[(1, 0)] = kx.sin();
    h.im[(0, 1)] = -ky.sin();
    h.im[(1, 0)] = ky.sin();
    generalized_orbitals(&h, &identity(2), 1e-12)
        .unwrap()
        .coefficients
}
fn overlap(a: &BlochMatrix, b: &BlochMatrix) -> Complex {
    (0..2)
        .map(|i| z(a, i, 0).conj() * z(b, i, 0))
        .fold(Complex::ZERO, |x, y| x + y)
}

#[test]
fn nonzero_integer_quantum_hall_is_non_abelian_gauge_invariant() {
    for size in [8, 16, 32] {
        let tau = 2.0 * std::f64::consts::PI;
        let mut states = Vec::new();
        for y in 0..size {
            for x in 0..size {
                states.push(lower(
                    tau * x as f64 / size as f64,
                    tau * y as f64 / size as f64,
                ));
            }
        }
        let mut links: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        let mut doubled: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        let mut gauged: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        let gauge = |index: usize| {
            let theta = 0.37 * index as f64;
            let mut u = identity(2);
            u.re[(0, 0)] = theta.cos();
            u.re[(1, 1)] = theta.cos();
            u.im[(0, 1)] = theta.sin();
            u.im[(1, 0)] = theta.sin();
            u
        };
        for axis in 0..2 {
            for y in 0..size {
                for x in 0..size {
                    let i = y * size + x;
                    let j = if axis == 0 {
                        y * size + (x + 1) % size
                    } else {
                        ((y + 1) % size) * size + x
                    };
                    let v = overlap(&states[i], &states[j]);
                    let mut single = BlochMatrix::zeros(1, 1);
                    single.re[(0, 0)] = v.re;
                    single.im[(0, 0)] = v.im;
                    links[axis].push(single);
                    let mut double = BlochMatrix::zeros(2, 2);
                    for n in 0..2 {
                        double.re[(n, n)] = v.re;
                        double.im[(n, n)] = v.im;
                    }
                    doubled[axis].push(double);
                    let left = gauge(i);
                    let right = gauge(j);
                    let mut link = BlochMatrix::zeros(2, 2);
                    for m in 0..2 {
                        for n in 0..2 {
                            let value = (0..2)
                                .map(|a| z(&left, a, m).conj() * v * z(&right, a, n))
                                .fold(Complex::ZERO, |a, b| a + b);
                            link.re[(m, n)] = value.re;
                            link.im[(m, n)] = value.im;
                        }
                    }
                    gauged[axis].push(link);
                }
            }
        }
        let single = quantum_hall_from_links(&links, [size, size]).unwrap();
        let double = quantum_hall_from_links(&doubled, [size, size]).unwrap();
        let transformed = quantum_hall_from_links(&gauged, [size, size]).unwrap();
        println!(
            "Chern mesh={size} single={} degenerate={} rotated={} integer residual={} max phase={}",
            single.chern_number,
            double.chern_number,
            transformed.chern_number,
            transformed.integer_residual,
            transformed.max_plaquette_phase
        );
        assert!((single.chern_number + 1.0).abs() < 1e-12);
        assert!((double.chern_number + 2.0).abs() < 1e-12);
        assert!((transformed.chern_number - double.chern_number).abs() < 1e-12);
        assert!((quantum_conductance(1.0) / 3.8740458649e-5 - 1.0).abs() < 1e-9);
    }
    // Independent analytic Berry curvature of the two-level Hamiltonian.
    // A centered plaquette avoids a first-order offset of its evaluation point.
    let x = 0.31_f64;
    let y = 0.21_f64;
    let d = [x.sin(), y.sin(), x.cos() + y.cos() - 1.0];
    let cross = [x.sin() * y.cos(), x.cos() * y.sin(), x.cos() * y.cos()];
    let berry = -0.5 * (0..3).map(|a| d[a] * cross[a]).sum::<f64>()
        / d.iter().map(|v| v * v).sum::<f64>().powf(1.5);
    let steps = [0.08, 0.04, 0.02, 0.01];
    let mut errors = Vec::new();
    for h in steps {
        let states = [
            lower(x - h / 2.0, y - h / 2.0),
            lower(x + h / 2.0, y - h / 2.0),
            lower(x + h / 2.0, y + h / 2.0),
            lower(x - h / 2.0, y + h / 2.0),
        ];
        let order = [0, 1, 3, 2];
        let mut links = [Vec::new(), Vec::new()];
        for axis in 0..2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    let i = iy * 2 + ix;
                    let j = if axis == 0 {
                        iy * 2 + (ix + 1) % 2
                    } else {
                        ((iy + 1) % 2) * 2 + ix
                    };
                    let value = overlap(&states[order[i]], &states[order[j]]);
                    let mut link = BlochMatrix::zeros(1, 1);
                    link.re[(0, 0)] = value.re;
                    link.im[(0, 0)] = value.im;
                    links[axis].push(link);
                }
            }
        }
        let production = quantum_hall_from_links(&links, [2, 2]).unwrap();
        errors.push((production.plaquette_phases[0] / (h * h) - berry).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "Berry holonomy steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
}

fn reference() -> (System, Controls) {
    let s = System {
        numbers: vec![1, 1],
        positions: vec![[0.0; 3], [1.6, 0.1, 0.05]],
        cell: Some([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    };
    let c = Controls::from_value(
        &Value::object()
            .with("xc", "pbe0")
            .with("basis", "sto-3g")
            .with("k_mesh", vec![4_i64, 1, 1])
            .with("overlap_cutoff", 12.0)
            .with("exchange_cutoff", 5.5)
            .with("grid_level", 2_i64)
            .with("grid_shell", 1_i64)
            .with("conv_tol", 1e-11)
            .with("conv_tol_dp", 1e-9)
            .with("max_iter", 300_i64),
    )
    .unwrap();
    (s, c)
}

#[test]
fn physical_cross_k_metric_and_current_connection_pass_four_step_fit() {
    let (s, c) = reference();
    let method = PeriodicMethod::from_controls(&c).unwrap();
    let setup = prepare_periodic(&s, &method).unwrap();
    let scf = run_periodic_scf(&setup.cell, &setup.functional, &setup.options).unwrap();
    assert!(scf.converged);
    let model = CoherentModel::new(&setup.cell, &setup.functional, &scf, &setup.options).unwrap();
    let k = Vec3::new(0.21, 0.11, 0.08);
    let point = model.point(0, k).unwrap();
    let defect = point.hermitian_residual().unwrap();
    println!("physical current raw Hermitian defect={defect:.12e}");
    assert!(defect < 1e-10);
    for band in 0..2 {
        let d = model.bands.derivatives(0, k, band, None, 1e-8).unwrap();
        for a in 0..3 {
            assert!((point.velocities[a].re[(band, band)] - d.velocity.unwrap()[a]).abs() < 1e-12);
        }
    }
    let step = Vec3::new(0.8, 0.5, 0.33);
    let u = step * (1.0 / step.norm());
    let steps = [0.04, 0.02, 0.01, 0.005];
    // Independent derivative of <u_basis(k)|u_basis(k+q)> by real-space
    // quadrature-free Gaussian moments: S'_u-i D_u. Infer D from molecular
    // dipole blocks over exactly the selected overlap translations.
    // Four-step fit of the physical cross metric derivative against the
    // exact analytic integral derivative exposed through the AO moments.
    // The independent closed-form implementation is kept in this test below.
    let derivative = analytic_s_cross_derivative(&setup.cell, k, u);
    let mut errors = Vec::new();
    for h in steps {
        let mut plus = CrossMetric::new(&setup.cell, u * h).unwrap().at(k + u * h);
        let minus = CrossMetric::new(&setup.cell, u * (-h))
            .unwrap()
            .at(k + u * (-h));
        plus.add_scaled(&minus, -1.0);
        plus.scale(1.0 / (2.0 * h));
        errors.push(max_difference(&plus, &derivative));
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "AO cross metric steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
}

fn analytic_s_cross_derivative(cell: &dft_rs::periodic::Cell, k: Vec3, u: Vec3) -> BlochMatrix {
    let mut out = BlochMatrix::zeros(cell.nao(), cell.nao());
    let spread = cell
        .basis()
        .shells
        .iter()
        .flat_map(|a| {
            cell.basis()
                .shells
                .iter()
                .map(move |b| (a.center - b.center).norm())
        })
        .fold(0.0_f64, f64::max);
    for t in cell
        .lattice()
        .translations_within(cell.overlap_cutoff() + spread)
    {
        for left in &cell.basis().shells {
            for right in &cell.basis().shells {
                assert_eq!(left.l, 0);
                assert_eq!(right.l, 0);
                let mut moment = 0.0;
                let a = left.center;
                let b = right.center + t;
                if (a - b).norm() > cell.overlap_cutoff() {
                    continue;
                }
                for (ep, cp) in left.exps.iter().zip(&left.coefs) {
                    for (eq, cq) in right.exps.iter().zip(&right.coefs) {
                        let exponent = ep + eq;
                        let center = (a * (*ep) + b * (*eq)) * (1.0 / exponent);
                        let overlap = cp
                            * cq
                            * (std::f64::consts::PI / exponent).powf(1.5)
                            * (-ep * eq / exponent * (a - b).norm2()).exp();
                        moment += overlap * (t - center).dot(u);
                    }
                }
                let phase = k.dot(t);
                let i = left.sph_off;
                let j = right.sph_off;
                out.re[(i, j)] -= phase.sin() * moment;
                out.im[(i, j)] += phase.cos() * moment;
            }
        }
    }
    out
}

#[test]
fn coherent_and_quantum_drivers_report_actual_dft_results() {
    let (s, c) = reference();
    let mut values = c.to_value();
    values.set("transport_k_mesh", vec![4_i64, 2, 2]);
    values.set("transport_frequencies_hartree", vec![0.0, 0.13, -0.13]);
    let report = run(
        "coherent-transport",
        &s,
        &Controls::from_value(&values).unwrap(),
    )
    .unwrap();
    assert!(report.get("fock_image_residual").unwrap().as_f64().unwrap() < 1e-10);
    let rows = report.get("results").unwrap().as_array().unwrap();
    for row in rows {
        assert!(
            row.get("raw_current_hermitian_residual")
                .unwrap()
                .as_f64()
                .unwrap()
                < 1e-10
        );
        assert!(row.get("raw_equation_residual").unwrap().as_f64().unwrap() < 1e-13);
    }
    let (_, positive) = rows[1]
        .get("conductivity")
        .unwrap()
        .numeric_array()
        .unwrap();
    let (_, negative) = rows[2]
        .get("conductivity")
        .unwrap()
        .numeric_array()
        .unwrap();
    for i in 0..9 {
        assert!((positive[2 * i] - negative[2 * i]).abs() < 1e-8);
        assert!((positive[2 * i + 1] + negative[2 * i + 1]).abs() < 1e-8);
    }
    let mut values = c.to_value();
    values.set("topology_mesh", vec![4_i64, 4]);
    let q = run("quantum-hall", &s, &Controls::from_value(&values).unwrap()).unwrap();
    println!(
        "DFT quantum Hall Chern={:?}, conductance={:?}",
        q.get("chern_number"),
        q.get("slice_hall_conductance")
    );
    assert!(q.get("chern_number").unwrap().as_f64().unwrap().abs() < 1e-12);
}
