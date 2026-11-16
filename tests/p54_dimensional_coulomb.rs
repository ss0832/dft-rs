// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent contracted dimensional nuclear/Hartree identities and fits.

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::coulomb_boundary::{
    periodic_coulomb_operators, DimensionalCoulombBackend, PeriodicCoulombOptions,
    PeriodicCoulombReport,
};
use dft_rs::periodic::electrostatics::ChargeCompensation;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::planewave::plane_wave_matrix;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::system::{Atom, Molecule};
use std::f64::consts::PI;

fn molecule() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.2, -0.1, 0.3),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.6, 0.5, -0.4),
        },
    ])
    .with_charge(1)
}
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
fn basis(mol: &Molecule) -> MolBasis {
    let mut definition = BasisSet::new("contracted-reference");
    definition.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7, 0.3],
            coefs: vec![0.6, 0.4],
        }],
    });
    MolBasis::build(mol, &definition).unwrap()
}
fn options() -> PeriodicCoulombOptions {
    let mut options = PeriodicCoulombOptions {
        basis_cutoff: 4.3,
        k_mesh: [3, 1, 1],
        ..Default::default()
    };
    options.electrostatics.alpha = Some(0.55);
    options.electrostatics.real_cutoff = Some(16.0);
    options.electrostatics.reciprocal_cutoff = Some(8.0);
    options.electrostatics.compensation = ChargeCompensation::GaussianCountercharge {
        center: Vec3::new(-0.5, 0.7, 0.9),
        width: 1.1,
    };
    options
}
fn density() -> Vec<BlochMatrix> {
    (0..3)
        .map(|_| {
            let mut d = BlochMatrix::zeros(2, 2);
            d.re[(0, 0)] = 0.6;
            d.re[(1, 1)] = 0.5;
            d.re[(0, 1)] = 0.2;
            d.re[(1, 0)] = 0.2;
            d.im[(0, 1)] = 0.07;
            d.im[(1, 0)] = -0.07;
            d
        })
        .collect()
}
fn defect(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.as_slice()
        .iter()
        .zip(b.re.as_slice())
        .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}
fn fit(label: &str, steps: &[f64; 4], errors: &[f64; 4]) {
    assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
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
    for v in errors.windows(2) {
        assert!((v[0] / v[1] / 4.0 - 1.0).abs() < 0.03);
    }
}
fn raw(report: &PeriodicCoulombReport) {
    println!("raw defects quadVJ={} quadD={} quadNN={} Herm={} translation={} densityIm={} potentialIm={} charge={:?}",
        report.quadrature_operator_defect,report.quadrature_derivative_defect,report.quadrature_source_defect,
        report.hermiticity_defect,report.translation_defect,report.real_space_density_imaginary_max,
        report.hartree_potential_imaginary_defect,report.charge_residual);
    assert!(
        report.quadrature_operator_defect < 1e-11
            && report.quadrature_derivative_defect < 1e-11
            && report.quadrature_source_defect < 1e-11
            && report.hermiticity_defect < 1e-11
            && report.translation_defect < 1e-11
            && report.hartree_potential_imaginary_defect < 1e-11
    );
}

#[test]
fn all_dimensions_contract_spherical_densities_without_a_time_reversal_restriction() {
    let mol = molecule();
    let mb = basis(&mol);
    for d in 1..=3 {
        let backend = DimensionalCoulombBackend::new(&mol, &mb, &geometry(d), &options()).unwrap();
        let mut density = density();
        let initial = backend.evaluate(Some(&density)).unwrap();
        raw(&initial);
        let scale = initial.expected_electrons / initial.electron_count.unwrap();
        for density in &mut density {
            density.scale(scale);
        }
        let report = backend.evaluate(Some(&density)).unwrap();
        raw(&report);
        assert!(report.converged && report.centre_count == 3);
        assert!(report.real_space_density_imaginary_max > 1e-3);
        assert!(report.nuclear[1]
            .im
            .as_slice()
            .iter()
            .any(|v| v.abs() > 1e-4));
        let mut real = density.clone();
        for r in &mut real {
            r.im = Matrix::zeros(2, 2);
        }
        let real = backend.evaluate(Some(&real)).unwrap();
        for (a, b) in report
            .hartree
            .as_ref()
            .unwrap()
            .iter()
            .zip(real.hartree.as_ref().unwrap())
        {
            assert!(defect(a, b) < 1e-11);
        }
        let v = report
            .electron_nuclear_components
            .unwrap()
            .into_iter()
            .sum::<f64>();
        let j = report
            .hartree_energy_components
            .unwrap()
            .into_iter()
            .sum::<f64>();
        assert!(
            (report.total_coulomb_energy.unwrap() - report.source_energy - v - j).abs() < 1e-12
        );
        let sum = report
            .energy_gradients
            .unwrap()
            .into_iter()
            .fold(Vec3::zero(), |s, g| s + g);
        assert!(sum.norm() < 1e-11);
    }
}

#[test]
fn contracted_coulomb_atomic_derivatives_have_four_step_matrix_and_energy_fits() {
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mol = molecule();
    let mb = basis(&mol);
    let density = density();
    for d in 1..=3 {
        let geometry = geometry(d);
        let exact =
            periodic_coulomb_operators(&mol, &mb, &geometry, Some(&density), &options()).unwrap();
        raw(&exact);
        let mut errors_v = [0.0_f64; 4];
        let mut errors_j = [0.0_f64; 4];
        let mut errors_e = [0.0_f64; 4];
        let mut estimates_e = [0.0; 4];
        let mut last_v = Vec::new();
        let mut last_j = Vec::new();
        for (index, h) in steps.iter().enumerate() {
            let mut plus = mol.clone();
            plus.atoms[1].position.x += h;
            let mut minus = mol.clone();
            minus.atoms[1].position.x -= h;
            let plus = periodic_coulomb_operators(
                &plus,
                &basis(&plus),
                &geometry,
                Some(&density),
                &options(),
            )
            .unwrap();
            let minus = periodic_coulomb_operators(
                &minus,
                &basis(&minus),
                &geometry,
                Some(&density),
                &options(),
            )
            .unwrap();
            raw(&plus);
            raw(&minus);
            let mut ev = Vec::new();
            let mut ej = Vec::new();
            for k in 0..3 {
                let mut v = plus.nuclear[k].clone();
                v.add_scaled(&minus.nuclear[k], -1.0);
                v.scale(1.0 / (2.0 * h));
                errors_v[index] =
                    errors_v[index].max(defect(&v, &exact.nuclear_derivatives[k][1][0]));
                ev.push(v);
                let mut j = plus.hartree.as_ref().unwrap()[k].clone();
                j.add_scaled(&minus.hartree.as_ref().unwrap()[k], -1.0);
                j.scale(1.0 / (2.0 * h));
                errors_j[index] = errors_j[index].max(defect(
                    &j,
                    &exact.hartree_derivatives.as_ref().unwrap()[k][1][0],
                ));
                ej.push(j);
            }
            estimates_e[index] = (plus.total_coulomb_energy.unwrap()
                - minus.total_coulomb_energy.unwrap())
                / (2.0 * h);
            errors_e[index] =
                (estimates_e[index] - exact.energy_gradients.as_ref().unwrap()[1].x).abs();
            if index == 2 {
                last_v = ev;
                last_j = ej;
            } else if index == 3 {
                for k in 0..3 {
                    let mut v = ev[k].clone();
                    v.scale(4.0);
                    v.add_scaled(&last_v[k], -1.0);
                    v.scale(1.0 / 3.0);
                    let mut j = ej[k].clone();
                    j.scale(4.0);
                    j.add_scaled(&last_j[k], -1.0);
                    j.scale(1.0 / 3.0);
                    assert!(defect(&v, &exact.nuclear_derivatives[k][1][0]) < 1e-8);
                    assert!(
                        defect(&j, &exact.hartree_derivatives.as_ref().unwrap()[k][1][0]) < 1e-8
                    );
                }
            }
        }
        fit(&format!("{d}D contracted nuclear"), &steps, &errors_v);
        fit(&format!("{d}D contracted Hartree"), &steps, &errors_j);
        fit(
            &format!("{d}D fixed-density Coulomb energy"),
            &steps,
            &errors_e,
        );
        assert!(
            ((4.0 * estimates_e[3] - estimates_e[2]) / 3.0
                - exact.energy_gradients.as_ref().unwrap()[1].x)
                .abs()
                < 1e-8
        );
    }
}

#[test]
fn contracted_p_d_operators_match_independent_full_reciprocal_matrices() {
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::new(0.2, -0.1, 0.3),
    }]);
    let mut definition = BasisSet::new("angular-reference");
    definition.insert(ElementBasis {
        z: 2,
        shells: vec![
            Shell {
                l: 1,
                exps: vec![0.6],
                coefs: vec![1.0],
            },
            Shell {
                l: 2,
                exps: vec![0.4],
                coefs: vec![1.0],
            },
        ],
    });
    let mb = MolBasis::build(&mol, &definition).unwrap();
    let mut options = options();
    options.k_mesh = [1; 3];
    options.basis_cutoff = 1.0;
    options.electrostatics.compensation = ChargeCompensation::UniformBackground;
    let mut density = BlochMatrix::zeros(8, 8);
    density.re[(0, 0)] = 1.0;
    density.re[(3, 3)] = 1.0;
    let geometry = geometry(3);
    let report =
        periodic_coulomb_operators(&mol, &mb, &geometry, Some(&[density.clone()]), &options)
            .unwrap();
    raw(&report);
    let mut oracle_v = BlochMatrix::zeros(8, 8);
    let mut oracle_j = BlochMatrix::zeros(8, 8);
    for g in geometry.reciprocal_within(14.0).unwrap() {
        let (re, im) = plane_wave_matrix(&mb, g);
        let kernel = 4.0 * PI / geometry.measure() / g.norm2();
        let phase = g.dot(mol.atoms[0].position);
        let dr = density.re.frobenius_dot(&re);
        let di = density.re.frobenius_dot(&im);
        for i in 0..8 {
            for j in 0..8 {
                oracle_v.re[(i, j)] -=
                    2.0 * kernel * (re[(i, j)] * phase.cos() + im[(i, j)] * phase.sin());
                oracle_j.re[(i, j)] += kernel * (re[(i, j)] * dr + im[(i, j)] * di);
            }
        }
    }
    println!(
        "p/d nuclear full-reciprocal defect={} Hartree defect={}",
        defect(&report.nuclear[0], &oracle_v),
        defect(&report.hartree.as_ref().unwrap()[0], &oracle_j)
    );
    assert!(defect(&report.nuclear[0], &oracle_v) < 1e-11);
    assert!(defect(&report.hartree.as_ref().unwrap()[0], &oracle_j) < 1e-11);
}

#[test]
fn all_charge_signs_and_compensations_have_split_invariant_joint_energies() {
    for d in 1..=3 {
        for charge in [-1, 0, 1] {
            let mut mol = molecule();
            mol.charge = charge;
            let mb = basis(&mol);
            let mut opts = options();
            if charge == 0 {
                opts.electrostatics.compensation = ChargeCompensation::RequireNeutral;
            } else if d == 3 {
                opts.electrostatics.compensation = ChargeCompensation::UniformBackground;
            }
            let geometry = geometry(d);
            let mut density = density();
            let initial =
                periodic_coulomb_operators(&mol, &mb, &geometry, Some(&density), &opts).unwrap();
            let scale = initial.expected_electrons / initial.electron_count.unwrap();
            for matrix in &mut density {
                matrix.scale(scale);
            }
            let exact =
                periodic_coulomb_operators(&mol, &mb, &geometry, Some(&density), &opts).unwrap();
            assert!(exact.converged);
            raw(&exact);
            opts.electrostatics.alpha = Some(0.7);
            let split =
                periodic_coulomb_operators(&mol, &mb, &geometry, Some(&density), &opts).unwrap();
            assert!(split.converged);
            raw(&split);
            assert!(
                (exact.total_coulomb_energy.unwrap() - split.total_coulomb_energy.unwrap()).abs()
                    < 1e-11
            );
            assert_eq!(exact.formal_charge, charge);
            assert_eq!(exact.compensating_charge, -(charge as f64));
            for (a, b) in exact.nuclear.iter().zip(&split.nuclear) {
                assert!(defect(a, b) < 1e-11);
            }
            for (a, b) in exact
                .hartree
                .as_ref()
                .unwrap()
                .iter()
                .zip(split.hartree.as_ref().unwrap())
            {
                assert!(defect(a, b) < 1e-11);
            }
        }
    }
}

#[test]
fn hartree_is_linear_and_self_adjoint_for_signed_complex_density_directions() {
    let mol = molecule();
    let mb = basis(&mol);
    for d in 1..=3 {
        let backend = DimensionalCoulombBackend::new(&mol, &mb, &geometry(d), &options()).unwrap();
        let a = density();
        let mut b = a.clone();
        for (k, m) in b.iter_mut().enumerate() {
            m.scale(-0.3);
            m.re[(0, 1)] += 0.2 * (k as f64).cos();
            m.re[(1, 0)] = m.re[(0, 1)];
        }
        let mut c = a.clone();
        for (c, b) in c.iter_mut().zip(&b) {
            c.add_scaled(b, 2.0);
        }
        let ja = backend.evaluate(Some(&a)).unwrap();
        let jb = backend.evaluate(Some(&b)).unwrap();
        let jc = backend.evaluate(Some(&c)).unwrap();
        raw(&ja);
        raw(&jb);
        raw(&jc);
        let ja = ja.hartree.unwrap();
        let jb = jb.hartree.unwrap();
        let jc = jc.hartree.unwrap();
        let mut ab = 0.0;
        let mut ba = 0.0;
        for k in 0..3 {
            let mut predicted = ja[k].clone();
            predicted.add_scaled(&jb[k], 2.0);
            assert!(defect(&predicted, &jc[k]) < 1e-11);
            ab += a[k].trace_product(&jb[k]);
            ba += b[k].trace_product(&ja[k]);
        }
        assert!(
            (ab - ba).abs() < 1e-11,
            "{d}D Hartree self-adjoint defect {}",
            ab - ba
        );
    }
}

#[test]
fn malformed_density_and_incompatible_compensation_are_rejected_before_contraction() {
    let mol = molecule();
    let mb = basis(&mol);
    let backend = DimensionalCoulombBackend::new(&mol, &mb, &geometry(1), &options()).unwrap();
    let mut malformed = density();
    malformed[0].im[(0, 1)] = f64::NAN;
    assert!(backend.evaluate(Some(&malformed)).is_err());
    let mut malformed = density();
    malformed[0].re[(1, 0)] += 0.1;
    assert!(backend.evaluate(Some(&malformed)).is_err());
    assert!(backend.evaluate(Some(&[])).is_err());
    let mut opts = options();
    opts.electrostatics.compensation = ChargeCompensation::UniformBackground;
    assert!(DimensionalCoulombBackend::new(&mol, &mb, &geometry(1), &opts).is_err());
    opts.electrostatics.compensation = ChargeCompensation::RequireNeutral;
    assert!(DimensionalCoulombBackend::new(&mol, &mb, &geometry(2), &opts).is_err());
    let mut bad = mb.clone();
    bad.shells[0].sph_off = usize::MAX;
    assert!(DimensionalCoulombBackend::new(&mol, &bad, &geometry(2), &options()).is_err());
    let mut opts = options();
    opts.nuclear_widths = vec![-0.3, 0.0];
    assert!(DimensionalCoulombBackend::new(&mol, &mb, &geometry(2), &opts).is_err());
}

#[test]
fn all_seven_masks_preserve_contracted_operators_when_open_vectors_are_removed() {
    let mol = molecule();
    let mb = basis(&mol);
    for mask in 1..8 {
        let axes = std::array::from_fn(|i| mask & (1 << i) != 0);
        let lattice = geometry(3).lattice().clone();
        let original = PeriodicGeometry::new(lattice.clone(), PeriodicBoundary::new(axes)).unwrap();
        let mut sparse = lattice;
        for axis in 0..3 {
            if !axes[axis] {
                sparse.a[axis] = Vec3::zero();
            }
        }
        let sparse = PeriodicGeometry::new(sparse, PeriodicBoundary::new(axes)).unwrap();
        let first = (0..3).find(|&i| axes[i]).unwrap();
        let mut opts = options();
        opts.k_mesh = [1; 3];
        opts.k_mesh[first] = 3;
        let a = periodic_coulomb_operators(&mol, &mb, &original, Some(&density()), &opts).unwrap();
        let b = periodic_coulomb_operators(&mol, &mb, &sparse, Some(&density()), &opts).unwrap();
        raw(&a);
        raw(&b);
        for (a, b) in a.nuclear.iter().zip(&b.nuclear) {
            assert!(defect(a, b) < 1e-12);
        }
        for (a, b) in a
            .hartree
            .as_ref()
            .unwrap()
            .iter()
            .zip(b.hartree.as_ref().unwrap())
        {
            assert!(defect(a, b) < 1e-12);
        }
        assert!((a.total_coulomb_energy.unwrap() - b.total_coulomb_energy.unwrap()).abs() < 1e-12);
        for (a, b) in a
            .energy_gradients
            .unwrap()
            .iter()
            .zip(b.energy_gradients.unwrap())
        {
            assert!((*a - b).norm() < 1e-12);
        }
    }
}

#[test]
fn finite_nuclear_joint_energy_matches_independent_complete_fourier_components() {
    let mol = molecule();
    let mb = basis(&mol);
    let geometry = geometry(3);
    let mut opts = options();
    opts.nuclear_widths = vec![1.0, 1.3];
    let backend = DimensionalCoulombBackend::new(&mol, &mb, &geometry, &opts).unwrap();
    let mut p = density();
    let initial = backend.evaluate(Some(&p)).unwrap();
    for matrix in &mut p {
        matrix.scale(initial.expected_electrons / initial.electron_count.unwrap());
    }
    let report = backend.evaluate(Some(&p)).unwrap();
    raw(&report);
    assert!(report.converged);
    // Deliberately use a brute integer box and the separate Fourier AO-pair
    // implementation, without dimensional image or Green tensor routines.
    let mut pairs = Vec::new();
    for i in -3..=3 {
        for j in -3..=3 {
            for k in -3..=3 {
                let translation = geometry.lattice().translation([i, j, k]);
                let active: Vec<_> = (0..2)
                    .flat_map(|a| (0..2).map(move |b| (a, b)))
                    .filter(|&(a, b)| {
                        (mb.shells[a].center - mb.shells[b].center - translation).norm()
                            <= opts.basis_cutoff
                    })
                    .collect();
                if active.is_empty() {
                    continue;
                }
                let mut joined = mb.clone();
                for original in &mb.shells {
                    let mut shell = original.clone();
                    shell.center += translation;
                    shell.sph_off += mb.nao_sph;
                    shell.cart_off += mb.nao_cart;
                    joined.shells.push(shell);
                }
                joined.nao_sph *= 2;
                joined.nao_cart *= 2;
                let mut d = BlochMatrix::zeros(2, 2);
                for (point, matrix) in report.k_points.iter().zip(&p) {
                    let phase = point.dot(translation);
                    for a in 0..2 {
                        for b in 0..2 {
                            d.re[(a, b)] += (phase.cos() * matrix.re[(a, b)]
                                + phase.sin() * matrix.im[(a, b)])
                                / 3.0;
                            d.im[(a, b)] += (phase.cos() * matrix.im[(a, b)]
                                - phase.sin() * matrix.re[(a, b)])
                                / 3.0;
                        }
                    }
                }
                pairs.push((joined, active, d));
            }
        }
    }
    let mut source = 0.0;
    let mut attraction = 0.0;
    let mut hartree = 0.0;
    let mut joint = 0.0;
    for g in geometry.reciprocal_within(14.0).unwrap() {
        let kernel = 4.0 * PI / geometry.measure() / g.norm2();
        let mut nr = 0.0;
        let mut ni = 0.0;
        for (joined, active, d) in &pairs {
            let (re, im) = plane_wave_matrix(joined, g);
            for &(a, b) in active {
                nr += d.re[(a, b)] * re[(a, 2 + b)] - d.im[(a, b)] * im[(a, 2 + b)];
                ni += d.re[(a, b)] * im[(a, 2 + b)] + d.im[(a, b)] * re[(a, 2 + b)];
            }
        }
        let mut sr = 0.0;
        let mut si = 0.0;
        for (position, charge, width) in [
            (mol.atoms[0].position, 1.0, 1.0),
            (mol.atoms[1].position, 1.0, 1.3),
            (Vec3::new(-0.5, 0.7, 0.9), -1.0, 1.1),
        ] {
            let factor = charge * (-0.25 * width * width * g.norm2()).exp();
            let phase = g.dot(position);
            sr += factor * phase.cos();
            si += factor * phase.sin();
        }
        source += 0.5 * kernel * (sr * sr + si * si);
        attraction -= kernel * (nr * sr + ni * si);
        hartree += 0.5 * kernel * (nr * nr + ni * ni);
        joint += 0.5 * kernel * ((sr - nr).powi(2) + (si - ni).powi(2));
    }
    let measured = [
        report.source_energy,
        report
            .electron_nuclear_components
            .unwrap()
            .into_iter()
            .sum(),
        report.hartree_energy_components.unwrap().into_iter().sum(),
        report.total_coulomb_energy.unwrap(),
    ];
    let oracle = [source, attraction, hartree, joint];
    let defects: Vec<_> = measured
        .iter()
        .zip(oracle)
        .map(|(a, b)| (a - b).abs())
        .collect();
    println!(
        "finite-nuclear joint Fourier measured={measured:?} oracle={oracle:?} defects={defects:?}"
    );
    assert!(defects.iter().all(|d| *d < 1e-11));
    assert!((source + attraction + hartree - joint).abs() < 1e-13);
}
