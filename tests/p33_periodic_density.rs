// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::data;
use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
use dft_rs::grid::GridOptions;
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::math::Vec3;
use dft_rs::periodic::bloch::BlochMatrix;
use dft_rs::periodic::occupations::OccupationScheme;
use dft_rs::periodic::scf::{reference_fock_at, run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::response::complex::Complex;
use dft_rs::response::krylov::LinearOperator;
use dft_rs::response::Frequency;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn setup() -> (Cell, PeriodicScfOptions) {
    let cell = Cell::new(
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]),
        Lattice::cubic(9.0),
        &data::load("sto-3g").unwrap(),
        7.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        conv_tol_grad: 1e-10,
        max_iter: 250,
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            ..Default::default()
        },
        ..Default::default()
    };
    (cell, options)
}

#[test]
fn incommensurate_periodic_response_matches_dense_complex_screening_and_dielectric_convention() {
    let (cell, scf_options) = setup();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &scf_options).unwrap();
    assert!(reference.converged);
    let rebuilt = reference_fock_at(
        &cell,
        &functional,
        &reference,
        &scf_options,
        &reference.k_points,
    )
    .unwrap();
    for s in 0..2 {
        assert!(rebuilt[s][0].rms_diff(&reference.fock[s][0]) < 1e-12);
    }
    let options = PeriodicResponseOptions {
        q: Vec3::new(0.043, 0.017, -0.021),
        frequency: Frequency::Retarded {
            omega: 0.13,
            damping: 0.04,
        },
        reciprocal_cutoff: 0.75,
        ..Default::default()
    };
    let solver =
        PeriodicDensitySolver::new(&cell, &functional, &reference, &scf_options, &options).unwrap();
    let source = solver.plane_wave_source(0).unwrap();
    let report = solver.solve(&source).unwrap();
    let rhs = solver
        .independent_response(&solver.potential_to_bands(&source).unwrap())
        .unwrap();
    let n = solver.dimension();
    let mut dense = Matrix::zeros(2 * n, 2 * n);
    for j in 0..n {
        let mut unit = vec![Complex::ZERO; n];
        unit[j] = Complex::from(1.0);
        let image = solver.apply(&unit).unwrap();
        for i in 0..n {
            dense[(i, j)] = image[i].re;
            dense[(n + i, j)] = image[i].im;
            dense[(i, n + j)] = -image[i].im;
            dense[(n + i, n + j)] = image[i].re;
        }
    }
    let vector: Vec<_> = rhs
        .iter()
        .map(|z| z.re)
        .chain(rhs.iter().map(|z| z.im))
        .collect();
    let dense_solution = solve_linear(&dense, &vector).unwrap();
    let band: Vec<_> = (0..n)
        .map(|i| Complex::new(dense_solution[i], dense_solution[n + i]))
        .collect();
    let ao = solver.density_to_aos(&band).unwrap();
    for s in 0..2 {
        assert!(ao[s][0].rms_diff(&report.value.density[s][0]) < 1e-11);
    }
    assert!(report.residuals[0] <= 1e-10);
    assert!(report.reference.particle_number_error < 1e-10);
    assert!(report.value.occupation_consistency_defect < 1e-9);
    let dielectric = solver.dielectric(&[0, 1]).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let expected = dielectric.value.screened_chi[i][j]
                * (4.0 * std::f64::consts::PI / dielectric.value.reciprocal_waves[i].norm2())
                + Complex::from(if i == j { 1.0 } else { 0.0 });
            assert!((dielectric.value.inverse_epsilon[i][j] - expected).abs() < 1e-14);
        }
    }
    assert!((dielectric.value.screened_chi[0][0] - report.value.fourier_density[0]).abs() < 1e-12);
}

#[test]
fn thermal_uniform_response_conserves_number_and_respects_constant_potential_gauge() {
    let (cell, mut options) = setup();
    options.occupation = OccupationScheme::FermiDirac {
        temperature_kelvin: 40000.0,
    };
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    let response = PeriodicResponseOptions {
        reciprocal_cutoff: 0.75,
        ..Default::default()
    };
    let solver =
        PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &response).unwrap();
    let uniform = [reference.overlap.clone(), reference.overlap.clone()];
    let report = solver.solve(&uniform).unwrap();
    for d in report.value.density.iter().flatten() {
        assert!(d
            .re
            .as_slice()
            .iter()
            .chain(d.im.as_slice())
            .all(|v| v.abs() < 1e-11));
    }
    assert!(report.value.temperature_hartree > 0.0);
    assert!(report.value.occupation_consistency_defect < 1e-9);
    let mut source: [Vec<_>; 2] = std::array::from_fn(|_| vec![BlochMatrix::zeros(2, 2)]);
    source[0][0].re[(0, 0)] = 0.4;
    source[1][0].re[(1, 1)] = -0.3;
    let report = solver.solve(&source).unwrap();
    for spin in 0..2 {
        assert!(
            report.value.density[spin][0]
                .trace_product(&reference.overlap[0])
                .abs()
                < 1e-10
        );
    }
    assert!(report.residuals[0] <= 1e-10);
}
