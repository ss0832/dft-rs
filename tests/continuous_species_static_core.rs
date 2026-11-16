// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Fixed generated-core provider tests. No SCF or fragment accuracy claim.

pub use dft_rs::{basis, dual, integrals, linalg, math, system, DftError, Result};
#[path = "../src/pseudo_sub/reduction_geometry.rs"]
pub mod geometry_impl;
#[path = "../src/pseudo_sub/reduction_kernels.rs"]
pub mod kernels_impl;
pub mod pseudo_sub {
    pub use crate::geometry_impl as reduction_geometry;
    pub use crate::kernels_impl as reduction_kernels;
    pub use dft_rs::pseudo_sub::*;
}
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "../src/driver/continuous_species/static_core.rs"]
mod static_core;

use basis::{aos, data, MolBasis};
use geometry_impl::{AttachmentPort, BodyFrameKinematics, FragmentSpec, RigidExpansion};
use kernels_impl::{
    charge_moments, charge_operator, core_interactions, projector_overlaps,
    symmetric_projector_coefficients, GaussianCharge, ProjectorTemplate,
};
use linalg::{Matrix, Orthogonalizer};
use math::{Mat3, Vec3};
use static_core::{static_core_correction, StaticCoreCorrection, StaticCoreParameters};
use system::{Atom, Molecule};

fn molecule() -> Molecule {
    let positions = [
        [-0.4, 0.2, 0.1],
        [2.3, 0.8, -0.2],
        [-1., 1.6, 0.7],
        [-1.2, -0.7, 1.1],
        [-0.9, -0.5, -1.5],
        [2.7, 2.4, 0.9],
        [3.5, -0.2, 0.7],
        [2.9, 0.3, -2.1],
    ];
    Molecule::new(
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i < 2 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}
fn map() -> RigidExpansion {
    RigidExpansion::methyl(
        &molecule(),
        FragmentSpec {
            explicit_atoms: vec![1, 5, 6, 7],
            retained_atoms: vec![0, 2, 3, 4],
            port: AttachmentPort {
                fragment_atom: 1,
                host_atom: 0,
                reference_atom: 2,
            },
            single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]],
            electronic_centers: 1,
        },
    )
    .unwrap()
}
fn coordinates() -> Vec<Vec3> {
    let mut positions = map().reference_coordinates();
    positions[4] += Vec3::new(0.11, -0.07, 0.04);
    positions[1] += Vec3::new(0.04, 0.03, 0.02);
    positions
}
fn reduced_molecule(positions: &[Vec3]) -> Molecule {
    Molecule::new(
        positions
            .iter()
            .enumerate()
            .map(|(i, &position)| Atom {
                z: if i == 0 { 6 } else { 1 },
                position,
            })
            .collect(),
    )
}
fn basis_at(positions: &[Vec3]) -> MolBasis {
    MolBasis::build(&reduced_molecule(positions), &data::load("sto-3g").unwrap()).unwrap()
}
fn parameters() -> StaticCoreParameters {
    StaticCoreParameters {
        charge_exponents: [0.27, 0.73, 1.31],
        charge_coefficients: [1.2, -0.9, -0.3],
        body_dipoles: [
            Vec3::new(0.13, -0.21, 0.17),
            Vec3::new(-0.09, 0.11, 0.07),
            Vec3::new(0.03, -0.04, 0.08),
        ],
        projector_s_exponents: [0.37, 1.13],
        projector_p_exponents: [0.51, 1.47],
        nonlocal_coefficients: std::array::from_fn(|k| {
            0.23 * ((k + 1) as f64 * 0.71).sin() - 0.11 * ((k + 2) as f64 * 0.39).cos()
        }),
    }
}
fn origin() -> Vec3 {
    Vec3::new(0.11, -0.07, 0.19)
}
fn field() -> Vec3 {
    Vec3::new(0.013, -0.021, 0.017)
}
fn charges() -> [f64; 5] {
    [6., 1., 1., 1., 1.]
}
fn frame_at(positions: &[Vec3]) -> BodyFrameKinematics {
    map().body_frame_kinematics(positions).unwrap().unwrap()
}
fn correction(
    positions: &[Vec3],
    parameters: &StaticCoreParameters,
    origin: Vec3,
) -> StaticCoreCorrection {
    static_core_correction(
        &basis_at(positions),
        positions,
        &charges(),
        4,
        &frame_at(positions),
        parameters,
        origin,
    )
    .unwrap()
}
fn shifted(point: Vec3, axis: usize, h: f64) -> Vec3 {
    let mut a = point.to_array();
    a[axis] += h;
    Vec3::new(a[0], a[1], a[2])
}
fn density(basis: &MolBasis) -> Matrix {
    let overlap = integrals::overlap_matrix(basis);
    let x = Orthogonalizer::canonical(&overlap, 1e-10).unwrap().x;
    let mut p = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    for i in 0..p.rows {
        for j in 0..p.cols {
            p[(i, j)] = 2. * (0..5).map(|a| x[(i, a)] * x[(j, a)]).sum::<f64>();
        }
    }
    assert!((p.frobenius_dot(&overlap) - 10.).abs() < 1e-12);
    p
}

struct Raw {
    operator: Matrix,
    classical: f64,
    dipole: Vec3,
}
// Independent VALUE assembly: no aggregate derivative/Jacobian is used.
fn raw(positions: &[Vec3], parameters: &StaticCoreParameters, origin: Vec3) -> Raw {
    let basis = basis_at(positions);
    let frame = frame_at(positions);
    let sites: Vec<_> = (0..3)
        .map(|s| GaussianCharge {
            center: frame.center,
            exponent: parameters.charge_exponents[s],
            charge: parameters.charge_coefficients[s],
            dipole: frame.orientation.mul_vec(parameters.body_dipoles[s]),
        })
        .collect();
    let mut operator = charge_operator(&basis, 5, &sites).unwrap().operator;
    let template = ProjectorTemplate {
        s_exponents: parameters.projector_s_exponents,
        p_exponents: parameters.projector_p_exponents,
        orientation: frame.orientation,
    };
    let nl = projector_overlaps(&basis, 5, frame.center, &template)
        .unwrap()
        .operator(&symmetric_projector_coefficients(&parameters.nonlocal_coefficients).unwrap())
        .unwrap()
        .operator;
    for (v, &w) in operator.as_mut_slice().iter_mut().zip(nl.as_slice()) {
        *v += w;
    }
    let cores: Vec<_> = (0..4).map(|i| (positions[i], charges()[i])).collect();
    Raw {
        operator,
        classical: core_interactions(&sites, &cores).unwrap().energy,
        dipole: charge_moments(&sites, origin).unwrap().1,
    }
}
fn energy(raw: &Raw, p: &Matrix, f: Vec3) -> f64 {
    p.frobenius_dot(&raw.operator) + raw.classical - raw.dipole.dot(f)
}
fn gradient(value: &StaticCoreCorrection, p: &Matrix, f: Vec3) -> Vec<Vec3> {
    (0..value.classical_gradients.len())
        .map(|atom| {
            let a: [f64; 3] = std::array::from_fn(|axis| {
                p.frobenius_dot(&value.atom_derivative_matrices[atom][axis])
                    + value.classical_gradients[atom].get(axis)
                    - (0..3)
                        .map(|i| f.get(i) * value.permanent_dipole_derivatives[atom][i][axis])
                        .sum::<f64>()
            });
            Vec3::new(a[0], a[1], a[2])
        })
        .collect()
}

#[test]
fn all_fifteen_reduced_energy_derivatives_four_step_fits() {
    let y = coordinates();
    let par = parameters();
    let p = density(&basis_at(&y));
    let value = correction(&y, &par, origin());
    let exact = gradient(&value, &p, field());
    let names: Vec<_> = (0..5)
        .flat_map(|atom| (0..3).map(move |axis| format!("fixed core energy_R[{atom},{axis}]")))
        .collect();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
        |steps| {
            let mut errors = vec![[0.; 4]; 15];
            for atom in 0..5 {
                for axis in 0..3 {
                    for (k, &h) in steps.iter().enumerate() {
                        let evaluate = |delta| {
                            let mut moved = y.clone();
                            moved[atom] = shifted(moved[atom], axis, delta);
                            energy(&raw(&moved, &par, origin()), &p, field())
                        };
                        errors[3 * atom + axis][k] =
                            ((evaluate(h) - evaluate(-h)) / (2. * h) - exact[atom].get(axis)).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("all15 scalar generated-core derivatives", &failures);
    let direct = raw(&y, &par, origin());
    assert!(value.operator.max_abs_diff(&direct.operator) < 1e-13);
    assert!((value.classical_energy - direct.classical).abs() < 1e-13);
    assert!((value.permanent_dipole - direct.dipole).norm() < 1e-13);
    println!(
        "fixedP correction energy={} grad={exact:?}",
        energy(&direct, &p, field())
    );
}

#[test]
fn all_active_operator_classical_and_moment_chain_entries_four_step_fits() {
    let y = coordinates();
    let par = parameters();
    let basis = basis_at(&y);
    let n = basis.nao_sph;
    let owner = basis.sph_ao_atoms();
    let own = basis.shells.iter().find(|s| s.atom == 4).unwrap().sph_off;
    let mut axial_s = vec![false; n];
    for shell in &basis.shells {
        if shell.l == 0 && [0, 4].contains(&shell.atom) {
            axial_s[shell.sph_off] = true;
        }
    }
    let value = correction(&y, &par, origin());
    let mut failures = Vec::new();
    for atom in 0..5 {
        for axis in 0..3 {
            let attached = [0, 1, 4].contains(&atom);
            // Moving only the reference anchor rotates the body around the
            // unchanged port axis. Two s AOs on that axis have no azimuthal
            // dependence, even for an anisotropic fixed projector matrix.
            let invariant_pair = |i: usize, j: usize| {
                (i == own && j == own)
                    || (atom == 1 && axial_s[i] && axial_s[j])
                    || (!attached && owner[i] != atom && owner[j] != atom)
            };
            let pairs: Vec<_> = (0..n)
                .flat_map(|i| (i..n).map(move |j| (i, j)))
                .filter(|&(i, j)| !invariant_pair(i, j))
                .collect();
            let names: Vec<_> = std::iter::once(format!("fixed core classical_R[{atom},{axis}]"))
                .chain(
                    (0..if attached { 3 } else { 0 })
                        .map(|i| format!("fixed core mu_R[{atom},{i},{axis}]")),
                )
                .chain(
                    pairs
                        .iter()
                        .map(|&(i, j)| format!("fixed core operator_R[{atom},{axis},{i},{j}]")),
                )
                .collect();
            components::fit_components(
                &names,
                &[[0.04, 0.02, 0.01, 0.005], [0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    let mut errors = vec![[0.; 4]; names.len()];
                    for (k, &h) in steps.iter().enumerate() {
                        let evaluate = |delta| {
                            let mut moved = y.clone();
                            moved[atom] = shifted(moved[atom], axis, delta);
                            raw(&moved, &par, origin())
                        };
                        let plus = evaluate(h);
                        let minus = evaluate(-h);
                        errors[0][k] = ((plus.classical - minus.classical) / (2. * h)
                            - value.classical_gradients[atom].get(axis))
                        .abs();
                        let mut entry = 1;
                        if attached {
                            for i in 0..3 {
                                errors[entry][k] = ((plus.dipole.get(i) - minus.dipole.get(i))
                                    / (2. * h)
                                    - value.permanent_dipole_derivatives[atom][i][axis])
                                    .abs();
                                entry += 1;
                            }
                        }
                        for &(i, j) in &pairs {
                            errors[entry][k] = ((plus.operator[(i, j)] - minus.operator[(i, j)])
                                / (2. * h)
                                - value.atom_derivative_matrices[atom][axis][(i, j)])
                                .abs();
                            entry += 1;
                        }
                        // Structural zeros are proved by support/azimuthal
                        // invariance and checked at every step, never floored.
                        for i in 0..n {
                            for j in i..n {
                                if invariant_pair(i, j) {
                                    assert!(
                                        (plus.operator[(i, j)] - minus.operator[(i, j)]).abs()
                                            < 1e-12
                                    );
                                    assert!(
                                        value.atom_derivative_matrices[atom][axis][(i, j)].abs()
                                            < 1e-12
                                    );
                                }
                            }
                        }
                        if !attached {
                            for i in 0..3 {
                                assert_eq!(value.permanent_dipole_derivatives[atom][i][axis], 0.);
                            }
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("all active fixed-core chain-rule entries", &failures);
}

fn rotation() -> Mat3 {
    let (s, c) = 0.41_f64.sin_cos();
    Mat3::from_columns(
        Vec3::new(c, s, 0.),
        Vec3::new(-s, c, 0.),
        Vec3::new(0., 0., 1.),
    )
}
fn rotate_density(basis: &MolBasis, p: &Matrix, r: Mat3) -> Matrix {
    let mut w = Matrix::identity(basis.nao_sph);
    let t = aos::cart_to_sph(1);
    for shell in &basis.shells {
        if shell.l == 1 {
            for i in 0..3 {
                for j in 0..3 {
                    w[(shell.sph_off + i, shell.sph_off + j)] = (0..3)
                        .map(|a| {
                            (0..3)
                                .map(|b| t[(a, i)] * r.col[b].get(a) * t[(b, j)])
                                .sum::<f64>()
                        })
                        .sum();
                }
            }
        }
    }
    w.matmul(p).matmul(&w.transpose())
}

#[test]
fn rigid_covariance_origin_ledger_and_no_self_or_projector_core_energy() {
    let y = coordinates();
    let par = parameters();
    let basis = basis_at(&y);
    let p = density(&basis);
    let value = correction(&y, &par, origin());
    value.validate(basis.nao_sph, 5).unwrap();
    assert_eq!(
        value.charge_correction,
        par.charge_coefficients.iter().sum::<f64>()
    );
    assert!((charges().iter().sum::<f64>() + value.charge_correction - 10.).abs() < 1e-13);
    assert!((1. + value.charge_correction - 1.).abs() < 1e-13); // one active fragment electron.
    let allcores: Vec<_> = y
        .iter()
        .zip(charges())
        .map(|(&position, z)| (position, z))
        .collect();
    let all = core_interactions(&value.electrostatic_sites, &allcores).unwrap();
    let self_term = core_interactions(&value.electrostatic_sites, &[(y[4], 1.)]).unwrap();
    assert!(self_term.energy.abs() > 1e-3);
    assert!((all.energy - self_term.energy - value.classical_energy).abs() < 1e-13);
    let mut changed = par.clone();
    for d in &mut changed.nonlocal_coefficients {
        *d = -3. * (*d) - 0.17;
    }
    let changed = correction(&y, &changed, origin());
    assert_eq!(changed.classical_energy, value.classical_energy);
    assert_eq!(changed.permanent_dipole, value.permanent_dipole);
    assert!(changed.operator.max_abs_diff(&value.operator) > 1e-3);
    let shift = Vec3::new(0.31, -0.42, 0.23);
    let moved: Vec<_> = y.iter().map(|&x| x + shift).collect();
    let translated = correction(&moved, &par, origin());
    assert!(translated.operator.max_abs_diff(&value.operator) < 1e-12);
    assert!((translated.classical_energy - value.classical_energy).abs() < 1e-12);
    assert!(
        (translated.permanent_dipole - value.permanent_dipole - shift * value.charge_correction)
            .norm()
            < 1e-13
    );
    let grad = gradient(&value, &p, field());
    let translated_grad = gradient(&translated, &p, field());
    for (a, b) in grad.iter().zip(&translated_grad) {
        assert!((*a - *b).norm() < 1e-12);
    }
    let r = rotation();
    let rotated_y: Vec<_> = y.iter().map(|&x| r.mul_vec(x)).collect();
    let rotated = correction(&rotated_y, &par, r.mul_vec(origin()));
    let rp = rotate_density(&basis, &p, r);
    let e0 = p.frobenius_dot(&value.operator) + value.classical_energy
        - value.permanent_dipole.dot(field());
    let er = rp.frobenius_dot(&rotated.operator) + rotated.classical_energy
        - rotated.permanent_dipole.dot(r.mul_vec(field()));
    assert!((er - e0).abs() < 1e-11);
    assert!((rotated.permanent_dipole - r.mul_vec(value.permanent_dipole)).norm() < 1e-13);
    let rotated_grad = gradient(&rotated, &rp, r.mul_vec(field()));
    for (a, b) in grad.iter().zip(&rotated_grad) {
        assert!((r.mul_vec(*a) - *b).norm() < 1e-10);
    }
    let changed_origin = origin() + shift;
    let shifted_origin = correction(&y, &par, changed_origin);
    assert!(
        (shifted_origin.permanent_dipole - value.permanent_dipole
            + shift * value.charge_correction)
            .norm()
            < 1e-13
    );
    assert_eq!(shifted_origin.classical_energy, value.classical_energy);
    let full_mu = value.permanent_dipole + y[4] - origin();
    let new_full_mu = shifted_origin.permanent_dipole + y[4] - changed_origin;
    assert!((new_full_mu - full_mu + shift * (1. + value.charge_correction)).norm() < 1e-13);
    assert_eq!(value.origin, origin());
}

#[test]
fn fixed_dictionary_validation_and_zero_correction_identity() {
    let y = coordinates();
    let basis = basis_at(&y);
    let frame = frame_at(&y);
    let par = parameters();
    let mut invalid = par.clone();
    invalid.charge_coefficients[0] += 0.1;
    assert!(invalid.validate().is_err());
    let mut wrong = charges();
    wrong[4] = 2.;
    assert!(static_core_correction(&basis, &y, &wrong, 4, &frame, &par, origin()).is_err());
    let badframe = BodyFrameKinematics {
        center: frame.center,
        orientation: frame.orientation,
        center_jacobian: Matrix::zeros(2, 15),
        orientation_jacobian: frame.orientation_jacobian.clone(),
    };
    assert!(static_core_correction(&basis, &y, &charges(), 4, &badframe, &par, origin()).is_err());
    let mut badbasis = basis.clone();
    badbasis.shells[0].center += Vec3::new(0.1, 0., 0.);
    assert!(static_core_correction(&badbasis, &y, &charges(), 4, &frame, &par, origin()).is_err());
    let mut bad = correction(&y, &par, origin());
    bad.operator[(0, 0)] = f64::NAN;
    assert!(bad.validate(basis.nao_sph, 5).is_err());
    let zero = StaticCoreParameters {
        charge_coefficients: [0.; 3],
        body_dipoles: [Vec3::zero(); 3],
        nonlocal_coefficients: [0.; 36],
        ..par
    };
    let result = correction(&y, &zero, origin());
    assert!(result.operator.as_slice().iter().all(|&x| x == 0.));
    assert_eq!(result.classical_energy, 0.);
    assert_eq!(result.permanent_dipole, Vec3::zero());
    assert!(result
        .classical_gradients
        .iter()
        .all(|&x| x == Vec3::zero()));
    assert!(result
        .atom_derivative_matrices
        .iter()
        .flatten()
        .all(|m| m.as_slice().iter().all(|&x| x == 0.)));
    assert!(result
        .permanent_dipole_derivatives
        .iter()
        .flatten()
        .flatten()
        .all(|&x| x == 0.));
}
