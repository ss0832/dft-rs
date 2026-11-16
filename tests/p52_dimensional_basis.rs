// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent dimensional Gaussian operators, Bloch phases and grid gates.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::GridOptions;
use dft_rs::math::Vec3;
use dft_rs::periodic::basis_boundary::dimensional_bloch_aos;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::grid::{dimensional_weight_gradient, PeriodicGrid};
use dft_rs::periodic::integrals::{image_blocks, image_derivative_blocks, Operator};
use dft_rs::periodic::{
    periodic_basis, BlochMatrix, Lattice, PeriodicBasisOptions, PeriodicGeometry,
};
use dft_rs::system::{Atom, Molecule};

fn geometry(d: usize) -> PeriodicGeometry {
    PeriodicGeometry::new(
        Lattice::new(
            Vec3::new(5.7, 0.4, 0.1),
            Vec3::new(0.6, 6.1, 0.3),
            Vec3::new(0.2, 0.4, 6.5),
        ),
        PeriodicBoundary::new(std::array::from_fn(|i| i < d)),
    )
    .unwrap()
}
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
fn options(d: usize) -> PeriodicBasisOptions {
    PeriodicBasisOptions {
        k_mesh: std::array::from_fn(|i| {
            if i == 0 && d > 0 {
                3
            } else if i < d {
                2
            } else {
                1
            }
        }),
        basis_cutoff: 8.5,
        ..Default::default()
    }
}
fn defect(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.as_slice()
        .iter()
        .zip(b.re.as_slice())
        .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
        .map(|(x, y)| (x - y).abs())
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
    for window in errors.windows(2) {
        assert!((window[0] / window[1] / 4.0 - 1.0).abs() < 0.03);
    }
}

#[test]
fn dimensional_s_gaussians_match_independent_closed_lattice_sums() {
    let mol = molecule();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    for d in 0..=3 {
        let geometry = geometry(d);
        let options = options(d);
        let report = periodic_basis(&mol, &mb, &geometry, &options).unwrap();
        assert!(report.hermiticity_defect < 1e-12 && report.translation_defect < 1e-12);
        assert!(report.minimum_overlap_eigenvalue > 0.1);
        for (ik, &k) in report.k_points.iter().enumerate() {
            let mut s = BlochMatrix::zeros(2, 2);
            let mut t = BlochMatrix::zeros(2, 2);
            for i in -3..=3 {
                for j in -3..=3 {
                    for l in -3..=3 {
                        if (d < 1 && i != 0) || (d < 2 && j != 0) || (d < 3 && l != 0) {
                            continue;
                        }
                        let image = geometry.lattice().translation([i, j, l]);
                        let phase = k.dot(image);
                        for a in 0..2 {
                            for b in 0..2 {
                                let distance2 =
                                    (mol.atoms[b].position + image - mol.atoms[a].position).norm2();
                                if distance2 > options.basis_cutoff.powi(2) {
                                    continue;
                                }
                                let (left, right) = (&mb.shells[a], &mb.shells[b]);
                                for (&alpha, &ca) in left.exps.iter().zip(&left.coefs) {
                                    for (&beta, &cb) in right.exps.iter().zip(&right.coefs) {
                                        let p = alpha + beta;
                                        let mu = alpha * beta / p;
                                        let ov = ca
                                            * cb
                                            * (std::f64::consts::PI / p).powf(1.5)
                                            * (-mu * distance2).exp();
                                        let ke = ov * mu * (3.0 - 2.0 * mu * distance2);
                                        s.re[(a, b)] += ov * phase.cos();
                                        s.im[(a, b)] += ov * phase.sin();
                                        t.re[(a, b)] += ke * phase.cos();
                                        t.im[(a, b)] += ke * phase.sin();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(defect(&s, &report.overlap[ik]) < 1e-12);
            assert!(defect(&t, &report.kinetic[ik]) < 1e-12);
        }
    }
}

#[test]
fn all_dimensions_overlap_kinetic_derivatives_have_four_step_fits() {
    let mol = molecule();
    let basis = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let steps = [0.04, 0.02, 0.01, 0.005];
    for d in 1..=3 {
        let g = geometry(d);
        let opt = options(d);
        let exact = periodic_basis(&mol, &mb, &g, &opt).unwrap();
        let mut se = [0.0; 4];
        let mut te = [0.0; 4];
        let mut estimates = Vec::new();
        for (index, h) in steps.iter().enumerate() {
            let mut plus = mol.clone();
            plus.atoms[1].position.x += h;
            let mut minus = mol.clone();
            minus.atoms[1].position.x -= h;
            let p =
                periodic_basis(&plus, &MolBasis::build(&plus, &basis).unwrap(), &g, &opt).unwrap();
            let m = periodic_basis(&minus, &MolBasis::build(&minus, &basis).unwrap(), &g, &opt)
                .unwrap();
            let mut pair = Vec::new();
            for (p, m, reference) in [
                (&p.overlap, &m.overlap, &exact.overlap_derivatives),
                (&p.kinetic, &m.kinetic, &exact.kinetic_derivatives),
            ] {
                let mut fd = p[1].clone();
                fd.add_scaled(&m[1], -1.0);
                fd.scale(0.5 / h);
                pair.push(fd.clone());
                if pair.len() == 1 {
                    se[index] = defect(&fd, &reference[1][1][0]);
                } else {
                    te[index] = defect(&fd, &reference[1][1][0]);
                }
            }
            estimates.push(pair);
        }
        fit(&format!("{d}D overlap"), &steps, &se);
        fit(&format!("{d}D kinetic"), &steps, &te);
        for (kind, reference) in [&exact.overlap_derivatives, &exact.kinetic_derivatives]
            .into_iter()
            .enumerate()
        {
            let mut rich = estimates[3][kind].clone();
            rich.scale(4.0 / 3.0);
            rich.add_scaled(&estimates[2][kind], -1.0 / 3.0);
            assert!(defect(&rich, &reference[1][1][0]) < 2e-9);
        }
    }
}

#[test]
fn legacy_3d_and_arbitrary_angular_momenta_share_the_same_integral_engine() {
    let mol = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.4, 0.3, 0.2),
        },
    ]);
    let mb = MolBasis::build(&mol, &data::load("6-31g*").unwrap()).unwrap();
    assert!(mb.max_l >= 2);
    let g = geometry(3);
    let opt = options(3);
    let report = periodic_basis(&mol, &mb, &g, &opt).unwrap();
    for (kind, operator) in [Operator::Overlap, Operator::Kinetic]
        .into_iter()
        .enumerate()
    {
        let old = image_blocks(&mb, g.lattice(), opt.basis_cutoff, operator);
        let old_d = image_derivative_blocks(&mb, g.lattice(), opt.basis_cutoff, operator);
        for (ik, &k) in report.k_points.iter().enumerate() {
            let actual = if kind == 0 {
                &report.overlap[ik]
            } else {
                &report.kinetic[ik]
            };
            let expected = BlochMatrix::from_pair(dft_rs::periodic::integrals::bloch(&old, k));
            assert!(defect(actual, &expected) < 1e-12);
            let expected =
                dft_rs::periodic::integrals::bloch_derivative(&old_d, k, mol.len(), mb.nao_sph);
            let actual = if kind == 0 {
                &report.overlap_derivatives[ik]
            } else {
                &report.kinetic_derivatives[ik]
            };
            for atom in 0..mol.len() {
                for axis in 0..3 {
                    assert!(
                        defect(
                            &actual[atom][axis],
                            &BlochMatrix::from_pair(expected[atom][axis].clone())
                        ) < 1e-12
                    );
                }
            }
        }
    }
    for d in 1..=2 {
        let low = geometry(d);
        let report = periodic_basis(&mol, &mb, &low, &options(d)).unwrap();
        assert!(report.hermiticity_defect < 1e-11 && report.translation_defect < 1e-11);
    }
}

#[test]
fn atom_cell_relabelling_preserves_bloch_covariance_and_open_vectors() {
    let mol = molecule();
    let bs = data::load("sto-3g").unwrap();
    for d in 1..=3 {
        let g = geometry(d);
        let opt = options(d);
        let report = periodic_basis(&mol, &MolBasis::build(&mol, &bs).unwrap(), &g, &opt).unwrap();
        let label = [3, if d > 1 { -2 } else { 0 }, if d > 2 { 1 } else { 0 }];
        let translation = g.lattice().translation(label);
        let mut changed = mol.clone();
        changed.atoms[1].position += translation;
        let shifted =
            periodic_basis(&changed, &MolBasis::build(&changed, &bs).unwrap(), &g, &opt).unwrap();
        for (ik, &k) in report.k_points.iter().enumerate() {
            for (a, b) in [
                (&report.overlap[ik], &shifted.overlap[ik]),
                (&report.kinetic[ik], &shifted.kinetic[ik]),
            ] {
                for i in 0..2 {
                    for j in 0..2 {
                        let phase = k.dot(translation) * (i as f64 - j as f64);
                        let re = a.re[(i, j)] * phase.cos() - a.im[(i, j)] * phase.sin();
                        let im = a.re[(i, j)] * phase.sin() + a.im[(i, j)] * phase.cos();
                        assert!(
                            (re - b.re[(i, j)]).abs() < 1e-12 && (im - b.im[(i, j)]).abs() < 1e-12
                        );
                    }
                }
            }
        }
        let mut lattice = g.lattice().clone();
        for axis in d..3 {
            lattice.a[axis] = Vec3::zero();
        }
        let open = PeriodicGeometry::new(lattice, g.boundary()).unwrap();
        let no_box =
            periodic_basis(&mol, &MolBasis::build(&mol, &bs).unwrap(), &open, &opt).unwrap();
        for ik in 0..report.k_points.len() {
            assert_eq!(report.overlap[ik], no_box.overlap[ik]);
        }
    }
}

#[test]
fn dimensional_becke_derivatives_fit_and_bloch_grid_has_physical_measure() {
    let mol = molecule();
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let point = Vec3::new(0.6, 0.3, 0.5);
    for d in 0..=3 {
        let g = geometry(d);
        let (_, grad) = dimensional_weight_gradient(&mol, &g, point, 0, 8.3).unwrap();
        let mut errors = [0.0; 4];
        let mut values = [0.0; 4];
        for (index, h) in steps.iter().enumerate() {
            let mut p = mol.clone();
            p.atoms[1].position.x += h;
            let mut m = mol.clone();
            m.atoms[1].position.x -= h;
            let wp = dimensional_weight_gradient(&p, &g, point, 0, 8.3)
                .unwrap()
                .0;
            let wm = dimensional_weight_gradient(&m, &g, point, 0, 8.3)
                .unwrap()
                .0;
            values[index] = (wp - wm) / (2.0 * h);
            errors[index] = (values[index] - grad[1].x).abs();
        }
        fit(&format!("{d}D fixed-point Becke"), &steps, &errors);
        assert!(((4.0 * values[3] - values[2]) / 3.0 - grad[1].x).abs() < 1e-8);
        let moving = grad[0].x - grad.iter().map(|g| g.x).sum::<f64>();
        for (index, h) in steps.iter().enumerate() {
            let mut p = mol.clone();
            p.atoms[0].position.x += h;
            let mut m = mol.clone();
            m.atoms[0].position.x -= h;
            let wp = dimensional_weight_gradient(&p, &g, point + Vec3::new(*h, 0.0, 0.0), 0, 8.3)
                .unwrap()
                .0;
            let wm = dimensional_weight_gradient(&m, &g, point - Vec3::new(*h, 0.0, 0.0), 0, 8.3)
                .unwrap()
                .0;
            values[index] = (wp - wm) / (2.0 * h);
            errors[index] = (values[index] - moving).abs();
        }
        fit(&format!("{d}D moving-point Becke"), &steps, &errors);
        assert!(((4.0 * values[3] - values[2]) / 3.0 - moving).abs() < 1e-8);
    }
    let bs = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &bs).unwrap();
    for d in 0..=3 {
        let g = geometry(d);
        let grid_options = GridOptions {
            level: 3,
            prune: false,
            weight_cutoff: 0.0,
            ..Default::default()
        };
        let grid = PeriodicGrid::build_dimensional(&mol, &g, &grid_options, 12.0).unwrap();
        assert!(grid.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
        let k = g
            .k_points(options(d).k_mesh)
            .unwrap()
            .last()
            .copied()
            .unwrap();
        let ao = dimensional_bloch_aos(&mb, &g, &grid.points, k, 12.0).unwrap();
        let exact = periodic_basis(
            &mol,
            &mb,
            &g,
            &PeriodicBasisOptions {
                k_mesh: options(d).k_mesh,
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let integrated = grid
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(p, w)| {
                        w * (ao.re[(p, i)] * ao.re[(p, j)] + ao.im[(p, i)] * ao.im[(p, j)])
                    })
                    .sum::<f64>();
                let error = (integrated - exact.overlap.last().unwrap().re[(i, j)]).abs();
                println!("{d}D grid S[{i},{j}] error={error:.12e}");
                assert!(error < 5e-5);
                let imaginary = grid
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(p, w)| {
                        w * (ao.re[(p, i)] * ao.im[(p, j)] - ao.im[(p, i)] * ao.re[(p, j)])
                    })
                    .sum::<f64>();
                let error = (imaginary - exact.overlap.last().unwrap().im[(i, j)]).abs();
                println!("{d}D grid Im S[{i},{j}] error={error:.12e}");
                assert!(error < 5e-5);
            }
        }
    }
}

#[test]
fn dimensional_basis_rejects_invalid_physical_geometry_before_work() {
    let mol = molecule();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let g = geometry(1);
    for cutoff in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(periodic_basis(
            &mol,
            &mb,
            &g,
            &PeriodicBasisOptions {
                basis_cutoff: cutoff,
                ..Default::default()
            }
        )
        .is_err());
    }
    assert!(periodic_basis(
        &mol,
        &mb,
        &g,
        &PeriodicBasisOptions {
            k_mesh: [1, 2, 1],
            ..Default::default()
        }
    )
    .is_err());
    assert!(periodic_basis(
        &mol,
        &mb,
        &g,
        &PeriodicBasisOptions {
            grid_derivatives: true,
            ..Default::default()
        }
    )
    .is_err());
    let mut other = mol.clone();
    other.atoms[1].position += Vec3::new(0.1, 0.0, 0.0);
    assert!(periodic_basis(&other, &mb, &g, &Default::default()).is_err());
    assert!(g
        .nearby_translation_indices(Vec3::new(f64::NAN, 0.0, 0.0), 1.0)
        .is_err());
}

#[test]
fn all_seven_masks_have_exact_nearby_images_and_supercell_band_folding() {
    for mask in 1..8 {
        let boundary = PeriodicBoundary::new(std::array::from_fn(|i| mask & (1 << i) != 0));
        let g = PeriodicGeometry::new(geometry(3).lattice().clone(), boundary).unwrap();
        let displacement = Vec3::new(7.2, -2.3, 3.4);
        let cutoff = 8.7;
        let actual = g.nearby_translation_indices(displacement, cutoff).unwrap();
        let mut expected = Vec::new();
        for i in -4..=4 {
            for j in -4..=4 {
                for k in -4..=4 {
                    let index = [i, j, k];
                    if (0..3).any(|axis| !boundary.axes[axis] && index[axis] != 0) {
                        continue;
                    }
                    if (displacement + g.lattice().translation(index)).norm() <= cutoff {
                        expected.push(index);
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }
    let bs = data::load("sto-3g").unwrap();
    for d in 1..=3 {
        let g = geometry(d);
        let atom = Atom {
            z: 1,
            position: Vec3::new(0.2, -0.1, 0.3),
        };
        let mol = Molecule::new(vec![atom.clone()]);
        let mb = MolBasis::build(&mol, &bs).unwrap();
        let primitive = periodic_basis(
            &mol,
            &mb,
            &g,
            &PeriodicBasisOptions {
                k_mesh: [2, 1, 1],
                basis_cutoff: 8.5,
                ..Default::default()
            },
        )
        .unwrap();
        let mut super_lattice = g.lattice().clone();
        super_lattice.a[0] = super_lattice.a[0] * 2.0;
        let super_geometry = PeriodicGeometry::new(super_lattice, g.boundary()).unwrap();
        let mut image = atom;
        image.position += g.lattice().a[0];
        let super_mol = Molecule::new(vec![mol.atoms[0].clone(), image]);
        let super_mb = MolBasis::build(&super_mol, &bs).unwrap();
        let report = periodic_basis(
            &super_mol,
            &super_mb,
            &super_geometry,
            &PeriodicBasisOptions {
                basis_cutoff: 8.5,
                ..Default::default()
            },
        )
        .unwrap();
        for (super_matrix, primitive_matrices) in [
            (&report.overlap[0], &primitive.overlap),
            (&report.kinetic[0], &primitive.kinetic),
        ] {
            let (values, _) = dft_rs::linalg::symmetric_eigen(&super_matrix.re).unwrap();
            let mut expected = primitive_matrices
                .iter()
                .map(|m| m.re[(0, 0)])
                .collect::<Vec<_>>();
            expected.sort_by(f64::total_cmp);
            for (a, b) in values.iter().zip(expected) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn dimensional_bloch_values_obey_boundary_phases_and_reciprocal_periodicity() {
    let mol = molecule();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let point = Vec3::new(0.8, 0.2, -0.3);
    for d in 1..=3 {
        let g = geometry(d);
        let k = g.reciprocal()[0] / 3.0;
        let values =
            dimensional_bloch_aos(&mb, &g, &[point, point + g.lattice().a[0]], k, 8.5).unwrap();
        let phase = k.dot(g.lattice().a[0]);
        for ao in 0..mb.nao_sph {
            let re = values.re[(0, ao)] * phase.cos() - values.im[(0, ao)] * phase.sin();
            let im = values.re[(0, ao)] * phase.sin() + values.im[(0, ao)] * phase.cos();
            assert!(
                (re - values.re[(1, ao)]).abs() < 1e-12 && (im - values.im[(1, ao)]).abs() < 1e-12
            );
        }
        let periodic = dimensional_bloch_aos(
            &mb,
            &g,
            &[point, point + g.lattice().a[0]],
            k + g.reciprocal()[0],
            8.5,
        )
        .unwrap();
        assert!(defect(&values, &periodic) < 1e-12);
    }
}
