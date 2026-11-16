// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

pub use dft_rs::{basis, dual, integrals, linalg, math, DftError, Result};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "../src/pseudo_sub/reduction_kernels.rs"]
mod kernels;

use basis::{data, MolBasis};
use dft_rs::system::{Atom, Molecule};
use integrals::one_electron::nuclear_matrix_kernel;
use integrals::CoulombKernel;
use kernels::*;
use linalg::Matrix;
use math::{Mat3, Vec3};
use std::f64::consts::PI;

fn molecule() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.13, -0.21, 0.17),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.34, 0.82, -0.23),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.74, 1.13, 0.91),
        },
    ])
}
fn basis(mol: &Molecule) -> MolBasis {
    MolBasis::build(mol, &data::load("sto-3g").unwrap()).unwrap()
}
fn identity() -> Mat3 {
    Mat3::from_columns(
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    )
}
fn template() -> ProjectorTemplate {
    ProjectorTemplate {
        s_exponents: [0.37, 1.13],
        p_exponents: [0.51, 1.47],
        orientation: identity(),
    }
}
fn site() -> GaussianCharge {
    GaussianCharge {
        center: Vec3::new(0.37, 0.29, -0.43),
        exponent: 0.63,
        charge: 0.83,
        dipole: Vec3::new(0.13, -0.21, 0.17),
    }
}
fn shift(p: Vec3, axis: usize, h: f64) -> Vec3 {
    let mut v = p.to_array();
    v[axis] += h;
    Vec3::new(v[0], v[1], v[2])
}
fn pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect()
}
fn density(b: &MolBasis) -> Matrix {
    let mut p = Matrix::zeros(b.nao_sph, b.nao_sph);
    for sh in &b.shells {
        for k in 0..2 * sh.l as usize + 1 {
            p[(sh.sph_off + k, sh.sph_off + k)] = 0.31 + 0.07 * sh.atom as f64 + 0.03 * sh.l as f64;
        }
    }
    p
}
fn coefficients() -> Matrix {
    let values: Vec<_> = (0..36)
        .map(|k| 0.23 * ((k + 1) as f64 * 0.71).sin() - 0.11 * ((k + 2) as f64 * 0.39).cos())
        .collect();
    symmetric_projector_coefficients(&values).unwrap()
}

fn independent_projector_overlaps(basis: &MolBasis, center: Vec3, t: &ProjectorTemplate) -> Matrix {
    use integrals::analytic_basis::AnalyticShell;
    let mut joined = basis.clone();
    for (l, exponent) in [
        (0, t.s_exponents[0]),
        (0, t.s_exponents[1]),
        (1, t.p_exponents[0]),
        (1, t.p_exponents[1]),
    ] {
        let normalized =
            AnalyticShell::normalized(l, center.to_array(), vec![exponent], vec![1.]).unwrap();
        joined.shells.push(basis::MolShell {
            l,
            atom: 0,
            center,
            exps: vec![exponent],
            coefs: normalized.coefficients,
            sph_off: joined.nao_sph,
            cart_off: joined.nao_cart,
        });
        joined.nao_sph += 2 * l as usize + 1;
        joined.nao_cart += basis::n_cart(l);
        joined.max_l = joined.max_l.max(l);
    }
    let full = integrals::overlap_matrix(&joined);
    let mut out = Matrix::zeros(basis.nao_sph, 8);
    for i in 0..out.rows {
        for j in 0..8 {
            out[(i, j)] = full[(i, basis.nao_sph + j)];
        }
    }
    out
}

// Closed scalar formula independent of the production dual-number path.
fn phi(s: &GaussianCharge, p: Vec3) -> f64 {
    let r = p - s.center;
    let d = r.norm();
    let k = s.exponent.sqrt();
    if d == 0. {
        return s.charge * 2. * k / PI.sqrt();
    }
    let erf = libm_erf(k * d);
    let a = 2. * k / PI.sqrt() * (-s.exponent * d * d).exp();
    s.charge * erf / d + s.dipole.dot(r) * (erf / d.powi(3) - a / d.powi(2))
}
fn libm_erf(x: f64) -> f64 {
    <f64 as dual::Scalar>::erf(x)
}
fn pair_oracle(a: &GaussianCharge, b: &GaussianCharge) -> f64 {
    let eta = a.exponent * b.exponent / (a.exponent + b.exponent);
    let r = b.center - a.center;
    let d = r.norm();
    let k = eta.sqrt();
    let erf = libm_erf(k * d);
    let e = 2. * k / PI.sqrt() * (-eta * d * d).exp();
    let aa = erf / d.powi(3) - e / d.powi(2);
    let bb = 3. * erf / d.powi(5) - e * (3. / d.powi(4) + 2. * eta / d.powi(2));
    a.charge * b.charge * erf / d
        + (a.dipole * b.charge - b.dipole * a.charge).dot(r) * aa
        + a.dipole.dot(b.dipole) * aa
        - a.dipole.dot(r) * b.dipole.dot(r) * bb
}

#[test]
fn charge_normalization_sign_core_field_and_pair_identity() {
    let a = site();
    let b = GaussianCharge {
        center: Vec3::new(-0.51, 1.27, 0.68),
        exponent: 1.07,
        charge: -0.83,
        dipole: Vec3::new(-0.09, 0.14, 0.23),
    };
    let origin = Vec3::new(0.11, -0.07, 0.19);
    let (q, p) = charge_moments(&[a.clone(), b.clone()], origin).unwrap();
    assert_eq!(q, 0.);
    assert!((p - ((a.center - b.center) * a.charge + a.dipole + b.dipole)).norm() < 1e-14);
    let points = [
        Vec3::new(2.1, -1.3, 0.7),
        Vec3::new(-1.1, 0.6, 2.3),
        a.center,
    ];
    for (actual, point) in electrostatic_potential(std::slice::from_ref(&a), &points)
        .unwrap()
        .iter()
        .zip(points)
    {
        assert!((actual - phi(&a, point)).abs() < 1e-13);
    }
    let field = Vec3::new(0.013, -0.021, 0.017);
    let e = field_interaction(&[a.clone(), b.clone()], origin, field).unwrap();
    assert!((e.energy + p.dot(field)).abs() < 1e-14);
    for (index, s) in [a.clone(), b.clone()].iter().enumerate() {
        assert!((e.center_gradients[index] + field * s.charge).norm() < 1e-14);
        assert!((e.dipole_gradients[index] + field).norm() < 1e-14);
        assert!((e.charge_gradients[index] + (s.center - origin).dot(field)).abs() < 1e-14);
    }
    let translated_origin = origin + Vec3::new(0.3, -0.2, 0.4);
    let (q, p0) = charge_moments(std::slice::from_ref(&a), origin).unwrap();
    let (_, p1) = charge_moments(std::slice::from_ref(&a), translated_origin).unwrap();
    assert!((p1 - p0 + (translated_origin - origin) * q).norm() < 1e-14);
    let pair = charge_pair(&a, &b).unwrap();
    assert!((pair.energy - pair_oracle(&a, &b)).abs() < 1e-13);
    let reverse = charge_pair(&b, &a).unwrap();
    assert!((pair.energy - reverse.energy).abs() < 1e-13);
    assert!((pair.first_center_gradient + pair.second_center_gradient).norm() < 1e-13);
    // Finite self terms are available but are never implicitly added.
    assert!(charge_pair(&a, &a).unwrap().energy.is_finite());
    let malformed = GaussianCharge { exponent: 0., ..a };
    assert!(charge_moments(&[malformed], origin).is_err());
}

#[test]
fn monopole_operator_independent_integral_and_dipole_all_component_fits() {
    let mol = molecule();
    let basis = basis(&mol);
    let mut s = site();
    s.dipole = Vec3::zero();
    let got = charge_operator(&basis, 3, &[s.clone()]).unwrap();
    let sources = Molecule::new(vec![Atom {
        z: 1,
        position: s.center,
    }]);
    let direct = nuclear_matrix_kernel(
        &basis,
        &sources,
        Some(&[s.charge]),
        CoulombKernel::Erf {
            omega: s.exponent.sqrt(),
        },
    );
    assert!(got.operator.max_abs_diff(&direct) < 1e-12);
    for i in 0..basis.nao_sph {
        for j in 0..basis.nao_sph {
            assert!(
                (got.operator[(i, j)] - s.charge * got.charge_operators[0][(i, j)]).abs() < 1e-13
            );
        }
    }
    let mut failures = Vec::new();
    for axis in 0..3 {
        let entries = pairs(basis.nao_sph);
        let names: Vec<_> = entries
            .iter()
            .map(|&(i, j)| format!("charge dipole_operator[{axis},{i},{j}]"))
            .collect();
        components::fit_components(
            &names,
            &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
            |steps| {
                let mut errors = vec![[0.; 4]; entries.len()];
                for (k, &h) in steps.iter().enumerate() {
                    let evaluate = |delta| {
                        let sources = Molecule::new(vec![Atom {
                            z: 1,
                            position: shift(s.center, axis, delta),
                        }]);
                        nuclear_matrix_kernel(
                            &basis,
                            &sources,
                            Some(&[1.]),
                            CoulombKernel::Erf {
                                omega: s.exponent.sqrt(),
                            },
                        )
                    };
                    let plus = evaluate(h);
                    let minus = evaluate(-h);
                    for (c, &(i, j)) in entries.iter().enumerate() {
                        errors[c][k] = ((plus[(i, j)] - minus[(i, j)]) / (2. * h)
                            - got.dipole_operators[0][axis][(i, j)])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("every independent Gaussian dipole operator", &failures);
}

#[test]
fn charge_operator_all_center_and_ao_derivatives_four_step_fits() {
    let mol = molecule();
    let reference_basis = basis(&mol);
    let s = site();
    let n = reference_basis.nao_sph;
    let got = charge_operator(&reference_basis, 3, std::slice::from_ref(&s)).unwrap();
    for i in 0..n {
        for j in 0..n {
            let reconstructed = s.charge * got.charge_operators[0][(i, j)]
                + (0..3)
                    .map(|axis| s.dipole.get(axis) * got.dipole_operators[0][axis][(i, j)])
                    .sum::<f64>();
            assert!((reconstructed - got.operator[(i, j)]).abs() < 1e-12);
        }
    }
    let owner = reference_basis.sph_ao_atoms();
    let mut failures = Vec::new();
    for atom in 0..=3 {
        for axis in 0..3 {
            let entries: Vec<_> = pairs(n)
                .into_iter()
                .filter(|&(i, j)| atom == 3 || owner[i] == atom || owner[j] == atom)
                .collect();
            let names: Vec<_> = entries
                .iter()
                .map(|&(i, j)| format!("charge dV[{atom},{axis},{i},{j}]"))
                .collect();
            let exact = if atom == 3 {
                &got.center_derivatives[0][axis]
            } else {
                &got.atom_derivatives[atom][axis]
            };
            components::fit_components(
                &names,
                &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    let mut errors = vec![[0.; 4]; entries.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let evaluate = |delta| {
                            let mut moved = mol.clone();
                            let mut changed = s.clone();
                            if atom == 3 {
                                changed.center = shift(changed.center, axis, delta)
                            } else {
                                moved.atoms[atom].position =
                                    shift(moved.atoms[atom].position, axis, delta)
                            }
                            charge_operator(&basis(&moved), 3, &[changed])
                                .unwrap()
                                .operator
                        };
                        let plus = evaluate(h);
                        let minus = evaluate(-h);
                        for (c, &(i, j)) in entries.iter().enumerate() {
                            errors[c][k] =
                                ((plus[(i, j)] - minus[(i, j)]) / (2. * h) - exact[(i, j)]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    for i in 0..n {
        for j in 0..n {
            for axis in 0..3 {
                let sum = got.center_derivatives[0][axis][(i, j)]
                    + got
                        .atom_derivatives
                        .iter()
                        .map(|g| g[axis][(i, j)])
                        .sum::<f64>();
                assert!(sum.abs() < 1e-11);
            }
        }
    }
    components::assert_no_failures("all nonstructural charge operator derivatives", &failures);
}

#[test]
fn core_and_distribution_pair_all_gradient_components_four_step_fits() {
    let a = site();
    let b = GaussianCharge {
        center: Vec3::new(-0.51, 1.27, 0.68),
        exponent: 1.07,
        charge: -0.43,
        dipole: Vec3::new(-0.09, 0.14, 0.23),
    };
    let sites = [a.clone(), b.clone()];
    let cores = [
        (Vec3::new(1.2, -0.7, 0.9), 1.7),
        (Vec3::new(-1.3, 0.4, -0.6), 0.8),
    ];
    let got = core_interactions(&sites, &cores).unwrap();
    let mut failures = Vec::new();
    for object in 0..4 {
        let names: Vec<_> = (0..3)
            .map(|axis| format!("core distribution gradient[{object},{axis}]"))
            .collect();
        components::fit_components(
            &names,
            &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
            |steps| {
                let mut errors = vec![[0.; 4]; 3];
                for (k, &h) in steps.iter().enumerate() {
                    for axis in 0..3 {
                        let energy = |delta| {
                            let mut s = sites.clone();
                            let mut c = cores;
                            if object < 2 {
                                s[object].center = shift(s[object].center, axis, delta)
                            } else {
                                c[object - 2].0 = shift(c[object - 2].0, axis, delta)
                            }
                            c.iter()
                                .map(|&(point, q)| {
                                    q * s.iter().map(|site| phi(site, point)).sum::<f64>()
                                })
                                .sum::<f64>()
                        };
                        let exact = if object < 2 {
                            got.center_gradients[object].get(axis)
                        } else {
                            got.core_gradients[object - 2].get(axis)
                        };
                        errors[axis][k] = ((energy(h) - energy(-h)) / (2. * h) - exact).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    let pair = charge_pair(&a, &b).unwrap();
    for object in 0..2 {
        let names: Vec<_> = (0..3)
            .map(|axis| format!("distribution pair gradient[{object},{axis}]"))
            .collect();
        components::fit_components(
            &names,
            &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
            |steps| {
                let mut errors = vec![[0.; 4]; 3];
                for (k, &h) in steps.iter().enumerate() {
                    for axis in 0..3 {
                        let energy = |delta| {
                            let mut first = a.clone();
                            let mut second = b.clone();
                            if object == 0 {
                                first.center = shift(first.center, axis, delta)
                            } else {
                                second.center = shift(second.center, axis, delta)
                            }
                            pair_oracle(&first, &second)
                        };
                        let exact = if object == 0 {
                            pair.first_center_gradient.get(axis)
                        } else {
                            pair.second_center_gradient.get(axis)
                        };
                        errors[axis][k] = ((energy(h) - energy(-h)) / (2. * h) - exact).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    // Coefficient and field maps are exactly linear. Check their closed-form
    // derivatives directly, rather than fitting a floor to roundoff errors.
    for (index, s) in sites.iter().enumerate() {
        let mut dq = 0.;
        let mut dp = Vec3::zero();
        for &(point, q) in &cores {
            let r = point - s.center;
            let distance = r.norm();
            let k = s.exponent.sqrt();
            let erf = libm_erf(k * distance);
            let e = 2. * k / PI.sqrt() * (-s.exponent * distance * distance).exp();
            dq += q * erf / distance;
            dp += r * (q * (erf / distance.powi(3) - e / distance.powi(2)));
        }
        assert!((got.charge_gradients[index] - dq).abs() < 1e-13);
        assert!((got.dipole_gradients[index] - dp).norm() < 1e-13);
    }
    let r = b.center - a.center;
    let distance = r.norm();
    let eta = a.exponent * b.exponent / (a.exponent + b.exponent);
    let k = eta.sqrt();
    let erf = libm_erf(k * distance);
    let e = 2. * k / PI.sqrt() * (-eta * distance * distance).exp();
    let aa = erf / distance.powi(3) - e / distance.powi(2);
    let bb =
        3. * erf / distance.powi(5) - e * (3. / distance.powi(4) + 2. * eta / distance.powi(2));
    let dpa = r * (b.charge * aa - b.dipole.dot(r) * bb) + b.dipole * aa;
    let dpb = r * (-a.charge * aa - a.dipole.dot(r) * bb) + a.dipole * aa;
    assert!((pair.first_dipole_gradient - dpa).norm() < 1e-13);
    assert!((pair.second_dipole_gradient - dpb).norm() < 1e-13);
    components::assert_no_failures("all core/pair gradients", &failures);
}

fn rotation(axis: usize, angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    match axis {
        0 => Mat3::from_columns(
            Vec3::new(1., 0., 0.),
            Vec3::new(0., c, s),
            Vec3::new(0., -s, c),
        ),
        1 => Mat3::from_columns(
            Vec3::new(c, 0., -s),
            Vec3::new(0., 1., 0.),
            Vec3::new(s, 0., c),
        ),
        _ => Mat3::from_columns(
            Vec3::new(c, s, 0.),
            Vec3::new(-s, c, 0.),
            Vec3::new(0., 0., 1.),
        ),
    }
}
fn generator(axis: usize) -> Mat3 {
    match axis {
        0 => Mat3::from_columns(Vec3::zero(), Vec3::new(0., 0., 1.), Vec3::new(0., -1., 0.)),
        1 => Mat3::from_columns(Vec3::new(0., 0., -1.), Vec3::zero(), Vec3::new(1., 0., 0.)),
        _ => Mat3::from_columns(Vec3::new(0., 1., 0.), Vec3::new(-1., 0., 0.), Vec3::zero()),
    }
}

#[test]
fn projector_normalization_signed_operator_dictionary_and_rigid_covariance() {
    let mol = molecule();
    let basis = basis(&mol);
    let center = site().center;
    let t = template();
    let d = coefficients();
    let b = projector_overlaps(&basis, 3, center, &t).unwrap();
    assert_eq!(b.overlap.rows, basis.nao_sph);
    assert_eq!(b.overlap.cols, 8);
    assert!(
        b.overlap
            .max_abs_diff(&independent_projector_overlaps(&basis, center, &t))
            < 1e-12
    );
    for i in 0..8 {
        assert!((b.projector_gram[(i, i)] - 1.).abs() < 1e-12);
    }
    assert!(b.projector_gram[(0, 1)].abs() > 0.1); // nonorthogonal radial dictionary.
    let v = b.operator(&d).unwrap();
    let dictionary = b.coefficient_operators();
    assert_eq!(dictionary.len(), 36);
    let mut reconstructed = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    let mut k = 0;
    for u in 0..8 {
        for w in u..8 {
            for (x, &y) in reconstructed
                .as_mut_slice()
                .iter_mut()
                .zip(dictionary[k].as_slice())
            {
                *x += d[(u, w)] * y;
            }
            k += 1;
        }
    }
    assert!(v.operator.max_abs_diff(&reconstructed) < 1e-12);
    let mut negative = Matrix::zeros(8, 8);
    negative[(0, 0)] = -1.;
    assert!(b.operator(&negative).unwrap().operator[(0, 0)] < 0.);
    let r = rotation(2, 0.41);
    let mut rotated = mol.clone();
    for atom in &mut rotated.atoms {
        atom.position = r.mul_vec(atom.position);
    }
    let rt = ProjectorTemplate {
        orientation: r,
        ..template()
    };
    let rotated_basis = basis_fn(&rotated);
    let rv = projector_overlaps(&rotated_basis, 3, r.mul_vec(center), &rt)
        .unwrap()
        .operator(&d)
        .unwrap();
    assert!(
        (density(&basis).frobenius_dot(&v.operator)
            - density(&rotated_basis).frobenius_dot(&rv.operator))
        .abs()
            < 1e-12
    );
    let mut malformed = d.clone();
    malformed[(0, 1)] += 1e-6;
    assert!(b.operator(&malformed).is_err());
    assert!(symmetric_projector_coefficients(&[0.; 35]).is_err());
}
fn basis_fn(mol: &Molecule) -> MolBasis {
    basis(mol)
}

#[test]
fn nonlocal_all_center_ao_and_physical_rotation_derivatives_four_step_fits() {
    let mol = molecule();
    let reference_basis = basis(&mol);
    let center = site().center;
    let t = template();
    let d = coefficients();
    let n = reference_basis.nao_sph;
    let overlaps = projector_overlaps(&reference_basis, 3, center, &t).unwrap();
    let v = overlaps.operator(&d).unwrap();
    let owner = reference_basis.sph_ao_atoms();
    let mut failures = Vec::new();
    for object in 0..5 {
        for axis in 0..3 {
            let entries: Vec<_> = pairs(n)
                .into_iter()
                .filter(|&(i, j)| object >= 3 || owner[i] == object || owner[j] == object)
                .collect();
            let names: Vec<_> = entries
                .iter()
                .map(|&(i, j)| format!("nonlocal dV[{object},{axis},{i},{j}]"))
                .collect();
            let exact = match object {
                0..=2 => v.atom_derivatives[object][axis].clone(),
                3 => v.center_derivatives[axis].clone(),
                _ => {
                    let g = generator(axis);
                    let mut result = Matrix::zeros(n, n);
                    for i in 0..3 {
                        for j in 0..3 {
                            for (x, &y) in result
                                .as_mut_slice()
                                .iter_mut()
                                .zip(v.orientation_derivatives[3 * i + j].as_slice())
                            {
                                *x += g.col[j].get(i) * y;
                            }
                        }
                    }
                    result
                }
            };
            components::fit_components(
                &names,
                &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    let mut errors = vec![[0.; 4]; entries.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let value = |delta| {
                            let mut moved = mol.clone();
                            let mut c = center;
                            let mut template = template();
                            match object {
                                0..=2 => {
                                    moved.atoms[object].position =
                                        shift(moved.atoms[object].position, axis, delta)
                                }
                                3 => c = shift(c, axis, delta),
                                _ => template.orientation = rotation(axis, delta),
                            }
                            projector_overlaps(&basis(&moved), 3, c, &template)
                                .unwrap()
                                .operator(&d)
                                .unwrap()
                                .operator
                        };
                        let plus = value(h);
                        let minus = value(-h);
                        for (c, &(i, j)) in entries.iter().enumerate() {
                            errors[c][k] =
                                ((plus[(i, j)] - minus[(i, j)]) / (2. * h) - exact[(i, j)]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    for i in 0..n {
        for j in 0..n {
            for axis in 0..3 {
                assert!(
                    (v.center_derivatives[axis][(i, j)]
                        + v.atom_derivatives
                            .iter()
                            .map(|a| a[axis][(i, j)])
                            .sum::<f64>())
                    .abs()
                        < 1e-12
                );
            }
        }
    }
    components::assert_no_failures("all nonlocal operator derivatives", &failures);
}

#[test]
fn projector_overlap_all_nonstructural_derivatives_four_step_fits() {
    let mol = molecule();
    let reference_basis = basis(&mol);
    let center = site().center;
    let t = template();
    let n = reference_basis.nao_sph;
    let overlaps = projector_overlaps(&reference_basis, 3, center, &t).unwrap();
    let owner = reference_basis.sph_ao_atoms();
    let transform = basis::aos::cart_to_sph(1);
    let mut failures = Vec::new();
    for object in 0..5 {
        for axis in 0..3 {
            let g = generator(axis);
            let columns: Vec<_> = (0..8)
                .filter(|&j| {
                    object != 4
                        || (j >= 2 && {
                            let m = (j - 2) % 3;
                            g.mul_vec(Vec3::new(
                                transform[(0, m)],
                                transform[(1, m)],
                                transform[(2, m)],
                            ))
                            .norm()
                                > 0.
                        })
                })
                .collect();
            let entries: Vec<_> = (0..n)
                .filter(|&i| object >= 3 || owner[i] == object)
                .flat_map(|i| columns.iter().copied().map(move |j| (i, j)))
                .collect();
            let names: Vec<_> = entries
                .iter()
                .map(|&(i, j)| format!("projector dB[{object},{axis},{i},{j}]"))
                .collect();
            let exact = match object {
                0..=2 => overlaps.atom_derivatives[object][axis].clone(),
                3 => overlaps.center_derivatives[axis].clone(),
                _ => {
                    let mut result = Matrix::zeros(n, 8);
                    for i in 0..3 {
                        for j in 0..3 {
                            for (x, &y) in result
                                .as_mut_slice()
                                .iter_mut()
                                .zip(overlaps.orientation_derivatives[3 * i + j].as_slice())
                            {
                                *x += g.col[j].get(i) * y;
                            }
                        }
                    }
                    result
                }
            };
            components::fit_components(
                &names,
                &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    let mut errors = vec![[0.; 4]; entries.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let value = |delta| {
                            let mut moved = mol.clone();
                            let mut c = center;
                            let mut template = template();
                            match object {
                                0..=2 => {
                                    moved.atoms[object].position =
                                        shift(moved.atoms[object].position, axis, delta)
                                }
                                3 => c = shift(c, axis, delta),
                                _ => template.orientation = rotation(axis, delta),
                            }
                            projector_overlaps(&basis(&moved), 3, c, &template)
                                .unwrap()
                                .overlap
                        };
                        let plus = value(h);
                        let minus = value(-h);
                        for (c, &(i, j)) in entries.iter().enumerate() {
                            errors[c][k] =
                                ((plus[(i, j)] - minus[(i, j)]) / (2. * h) - exact[(i, j)]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("all physical projector overlap derivatives", &failures);
}
