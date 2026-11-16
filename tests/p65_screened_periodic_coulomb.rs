// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The range-separated periodic Hartree build used by the SCF reproduces the
//! exhaustive supercell build on a converged finite-k density, and a
//! three-dimensional crystal SCF with analytic forces runs at a practical cost.

use dft_rs::basis::data;
use dft_rs::grid::GridOptions;
use dft_rs::linalg::{Matrix, Orthogonalizer};
use dft_rs::math::Vec3;
use dft_rs::periodic::integrals::{bloch, hermitian_as_real};
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn coarse_grid() -> GridOptions {
    GridOptions {
        radial_override: Some(20),
        angular_override: Some(26),
        prune: false,
        ..GridOptions::default()
    }
}

#[test]
fn screened_hartree_matches_the_reference_on_a_converged_density() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        },
    ]);
    let lattice = Lattice::new(
        Vec3::new(4.2, 0.0, 0.0),
        Vec3::new(0.0, 7.0, 0.0),
        Vec3::new(0.0, 0.0, 7.0),
    );
    let cell = Cell::new(molecule, lattice, &data::load("sto-3g").unwrap(), 12.0).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let options = PeriodicScfOptions {
        k_mesh: [3, 1, 1],
        grid: coarse_grid(),
        grid_shell: 1,
        e_tol: 1e-10,
        p_tol: 1e-9,
        ..PeriodicScfOptions::default()
    };
    let scf = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(scf.converged);
    let total: Vec<_> = scf
        .total_density()
        .iter()
        .map(|p| (p.re.clone(), p.im.clone()))
        .collect();
    let images = cell
        .density_images_from_k(
            &scf.k_points,
            &total,
            2.0 * cell.hartree_real_space_cutoff(),
        )
        .unwrap();
    let start = std::time::Instant::now();
    let screened = cell.hartree_image_blocks(&images).unwrap();
    let screened_time = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let reference = cell.reference_hartree_image_blocks(&images).unwrap();
    let reference_time = start.elapsed().as_secs_f64();
    let mut worst = 0.0_f64;
    let mut energy = [0.0_f64; 2];
    for (ik, k) in scf.k_points.iter().enumerate() {
        let a = bloch(&screened, *k);
        let b = bloch(&reference, *k);
        worst = worst
            .max(a.0.max_abs_diff(&b.0))
            .max(a.1.max_abs_diff(&b.1));
        let p = &total[ik];
        for (slot, j) in [a, b].iter().enumerate() {
            energy[slot] += 0.5 / scf.k_points.len() as f64
                * (p.0.frobenius_dot(&j.0) + p.1.frobenius_dot(&j.1));
        }
    }
    println!(
        "screened {screened_time:.2} s, reference {reference_time:.2} s, max |dJ(k)| {worst:e}, \
         E_H {:.12} vs {:.12}",
        energy[0], energy[1]
    );
    // Independent screened evaluations with a different split and a
    // tighter tolerance agree with each other far below the reference
    // difference, which is therefore the truncation of the reference's own
    // image supercell rather than an error of the screened build.
    use dft_rs::periodic::screened_coulomb::{screened_hartree_image_blocks, HartreeScreening};
    let mut spread = 0.0_f64;
    for (compact_exponent, tolerance) in [(0.5, 1e-15), (2.0, 1e-15)] {
        let other = screened_hartree_image_blocks(
            cell.basis(),
            cell.lattice(),
            &images,
            &HartreeScreening {
                compact_exponent,
                tolerance,
                ..cell.hartree_screening()
            },
        )
        .unwrap();
        for k in &scf.k_points {
            let a = bloch(&screened, *k);
            let b = bloch(&other, *k);
            spread = spread
                .max(a.0.max_abs_diff(&b.0))
                .max(a.1.max_abs_diff(&b.1));
        }
    }
    println!("spread between screened splits/tolerances {spread:e}");
    assert!(spread < 1e-10, "screened builds disagree by {spread:e}");
    assert!(worst < 1e-8, "J(k) differs from the reference by {worst:e}");
    assert!((energy[0] - energy[1]).abs() < 1e-8);
}

#[test]
fn charged_li_hartree_stress_is_split_independent_at_fixed_density() {
    use dft_rs::periodic::screened_coulomb::{screened_hartree_strain, HartreeScreening};

    let full = data::load("sto-3g").unwrap();
    let mut element = full.element(3).unwrap().clone();
    // The converged Gamma-point occupied orbital in this one-site Li cell
    // is even under inversion. Retain its two s basis functions.
    element.shells.retain(|shell| shell.l == 0);
    let mut basis = dft_rs::basis::BasisSet::new("Li occupied s space");
    basis.insert(element);
    let deform = |x, y, z| Vec3::new(1.012 * x, y, z);
    let lattice = Lattice::new(
        deform(8.4, 0.7, 0.3),
        deform(0.4, 8.9, 0.6),
        deform(0.2, -0.5, 9.2),
    );
    let molecule = Molecule::new(vec![Atom {
        z: 3,
        position: deform(0.17, 0.41, -0.19),
    }])
    .with_charge(1);
    let cell = Cell::new(molecule, lattice, &basis, 13.8).unwrap();
    let density = Matrix::from_row_major(
        2,
        2,
        vec![
            1.9633012245255201,
            0.06229104841009819,
            0.06229104841009819,
            0.001976352209002944,
        ],
    );
    let images = cell
        .density_images_from_k(
            &[Vec3::zero()],
            &[(density, Matrix::zeros(2, 2))],
            2.0 * cell.hartree_real_space_cutoff(),
        )
        .unwrap();
    let evaluate = |compact_exponent, tolerance| {
        screened_hartree_strain(
            cell.basis(),
            cell.lattice(),
            &images,
            &HartreeScreening {
                compact_exponent,
                tolerance,
                ..cell.hartree_screening()
            },
        )
        .unwrap()
    };
    let default = cell.hartree_screening().tolerance;
    let a = evaluate(1.0, default);
    let b = evaluate(1.2, default);
    let tight = evaluate(1.2, 1e-17);
    let difference = |left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]| {
        left.iter()
            .flatten()
            .zip(right.iter().flatten())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0_f64, f64::max)
    };
    let split_error = difference(&a.strain_derivative, &b.strain_derivative);
    let tail_error = difference(&a.strain_derivative, &tight.strain_derivative);
    println!("charged Li Hartree zy: default {:.15e}, alternate split {:.15e}, tightened {:.15e}; max split error={split_error:.9e}, max tail error={tail_error:.9e}",
        a.strain_derivative[2][1], b.strain_derivative[2][1], tight.strain_derivative[2][1]);
    assert!(split_error < 2e-13 && tail_error < 2e-13);
    assert!((a.energy - tight.energy).abs() < 2e-13);
}

fn lih_rocksalt() -> Cell {
    let a = 3.855;
    let lattice = Lattice::new(
        Vec3::new(0.0, a, a),
        Vec3::new(a, 0.0, a),
        Vec3::new(a, a, 0.0),
    );
    let molecule = Molecule::new(vec![
        Atom {
            z: 3,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(a, 0.0, 0.0),
        },
    ]);
    Cell::new(molecule, lattice, &data::load("sto-3g").unwrap(), 14.0).unwrap()
}

/// Closed-shell core-Hamiltonian density at every k point, built through the
/// real embedding of the Hermitian generalized eigenproblem.
fn core_guess_density(cell: &Cell, k_points: &[Vec3], occupied: usize) -> Vec<(Matrix, Matrix)> {
    let n = cell.nao();
    k_points
        .iter()
        .map(|k| {
            let (sr, si) = cell.overlap(*k);
            let (hr, hi) = cell.core_hamiltonian(*k);
            let x = Orthogonalizer::canonical(&hermitian_as_real(&sr, &si), 1e-10).unwrap();
            let (_, c) = x.solve_roothaan(&hermitian_as_real(&hr, &hi)).unwrap();
            let embedded = c.leading_columns_gram(2 * occupied, 2.0);
            let mut re = Matrix::zeros(n, n);
            let mut im = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    re[(i, j)] = embedded[(i, j)];
                    im[(i, j)] = embedded[(n + i, j)];
                }
            }
            (re, im)
        })
        .collect()
}

/// Rock-salt LiH on a 2x2x2 mesh: timing and agreement of the screened
/// Hartree build for several compact/diffuse split exponents on a
/// core-Hamiltonian density.
#[test]
#[ignore = "timing probe of the screened Hartree split; run explicitly"]
fn lih_hartree_split_benchmark() {
    let _ = dft_rs::logger::init(dft_rs::logger::LogMode::Console, None);
    use dft_rs::periodic::screened_coulomb::{screened_hartree_image_blocks, HartreeScreening};
    let cell = lih_rocksalt();
    let k_points = cell.k_points([2, 2, 2]);
    let density = core_guess_density(&cell, &k_points, 2);
    let images = cell
        .density_images_from_k(&k_points, &density, 2.0 * cell.hartree_real_space_cutoff())
        .unwrap();
    let electrons = cell.electron_count(&images).unwrap();
    println!("core-guess electrons per cell {electrons:.10}");
    let mut first: Option<Vec<(Matrix, Matrix)>> = None;
    for compact_exponent in [1.0, 2.0, 4.0, 8.0, 16.0] {
        let screening = HartreeScreening {
            compact_exponent,
            ..cell.hartree_screening()
        };
        let start = std::time::Instant::now();
        let blocks =
            screened_hartree_image_blocks(cell.basis(), cell.lattice(), &images, &screening)
                .unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let j: Vec<(Matrix, Matrix)> = k_points.iter().map(|k| bloch(&blocks, *k)).collect();
        let mut energy = 0.0;
        for (p, jk) in density.iter().zip(&j) {
            energy +=
                0.5 / k_points.len() as f64 * (p.0.frobenius_dot(&jk.0) + p.1.frobenius_dot(&jk.1));
        }
        let spread = first.as_ref().map_or(0.0, |reference| {
            reference
                .iter()
                .zip(&j)
                .map(|(a, b)| a.0.max_abs_diff(&b.0).max(a.1.max_abs_diff(&b.1)))
                .fold(0.0_f64, f64::max)
        });
        println!("split {compact_exponent}: {elapsed:.2} s, E_H {energy:.12}, max |dJ(k)| vs first {spread:e}");
        first.get_or_insert(j);
    }
}

/// Rock-salt LiH primitive cell, 2x2x2 k mesh: the case that did not finish
/// in 20 minutes with the exhaustive Hartree build. Prints the timing of the
/// SCF and of the analytic forces.
#[test]
#[ignore = "timing probe of a 3D crystal SCF and forces; run explicitly"]
fn lih_rocksalt_scf_and_forces_timing() {
    let _ = dft_rs::logger::init(dft_rs::logger::LogMode::Console, None);
    let cell = lih_rocksalt();
    let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let options = PeriodicScfOptions {
        k_mesh: [2, 2, 2],
        grid: coarse_grid(),
        grid_shell: 1,
        ..PeriodicScfOptions::default()
    };
    let start = std::time::Instant::now();
    let scf = run_periodic_scf(&cell, &functional, &options).unwrap();
    println!(
        "SCF: E = {:.10} Hartree, {} iterations, {:.1} s",
        scf.energy,
        scf.iterations,
        start.elapsed().as_secs_f64()
    );
    let start = std::time::Instant::now();
    let forces =
        dft_rs::periodic::forces::analytic_forces(&cell, &functional, &scf, &options).unwrap();
    println!(
        "forces {:?} in {:.1} s",
        forces.total,
        start.elapsed().as_secs_f64()
    );
}
