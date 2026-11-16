// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::constants::atomic_mass_au;
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::properties::vibrational::vibrational_analysis;
use dft_rs::response::anharmonic::rotation::{molecular_rotation, RotorClass, WatsonRotation};
use dft_rs::response::anharmonic::{
    vibrational_spectrum, ObservablePolynomial, VibrationalPolynomial,
};
use dft_rs::system::{Atom, Molecule};

fn model(scale: f64) -> VibrationalPolynomial {
    // Keep the smallest energy truncation error above eigensolver roundoff.
    let b = 1e-4 * scale * scale;
    let mut coriolis = vec![[0.0; 3]; 4];
    coriolis[1][2] = 0.4;
    coriolis[2][2] = -0.4;
    VibrationalPolynomial {
        frequencies: vec![0.05, 0.08],
        cubic: vec![0.0; 8],
        quartic: vec![0.0; 16],
        dipole: ObservablePolynomial {
            components: 3,
            first: vec![0.7, 0.0, 0.0, 0.0, 0.3, 0.0],
            second: vec![0.0; 12],
            third: vec![0.0; 24],
        },
        polarizability: None,
        rotation: Some(
            WatsonRotation::from_coefficients(
                [
                    [2.0 * b, 0.0, 0.0],
                    [0.0, 3.0 * b, 0.0],
                    [0.0, 0.0, 4.0 * b],
                ],
                coriolis,
                RotorClass::Nonlinear,
            )
            .unwrap(),
        ),
    }
}
fn oracle(input: &VibrationalPolynomial, size: usize) -> (Vec<f64>, f64) {
    let dim = size * size;
    let mut q1 = Matrix::zeros(dim, dim);
    let mut q2 = q1.clone();
    let mut p1 = q1.clone();
    let mut p2 = q1.clone();
    for i in 0..size {
        for j in 0..size {
            let state = i * size + j;
            if i + 1 < size {
                let to = (i + 1) * size + j;
                let q = ((i + 1) as f64 / (2.0 * input.frequencies[0])).sqrt();
                let p = ((i + 1) as f64 * input.frequencies[0] / 2.0).sqrt();
                q1[(to, state)] = q;
                q1[(state, to)] = q;
                p1[(to, state)] = p;
                p1[(state, to)] = -p;
            }
            if j + 1 < size {
                let to = i * size + j + 1;
                let q = ((j + 1) as f64 / (2.0 * input.frequencies[1])).sqrt();
                let p = ((j + 1) as f64 * input.frequencies[1] / 2.0).sqrt();
                q2[(to, state)] = q;
                q2[(state, to)] = q;
                p2[(to, state)] = p;
                p2[(state, to)] = -p;
            }
        }
    }
    let mut angular = q1.matmul_seq(&p2);
    let other = q2.matmul_seq(&p1);
    for (a, b) in angular.as_mut_slice().iter_mut().zip(other.as_slice()) {
        *a = 0.4 * (*a - *b);
    }
    let square = angular.matmul_seq(&angular);
    let rotation = input.rotation.as_ref().unwrap();
    let mut h = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            h[(i, j)] = -0.5 * rotation.inverse_inertia[2][2] * square[(i, j)];
        }
    }
    for i in 0..size {
        for j in 0..size {
            h[(i * size + j, i * size + j)] += (i as f64 + 0.5) * input.frequencies[0]
                + (j as f64 + 0.5) * input.frequencies[1]
                + rotation.watson_pseudopotential;
        }
    }
    let (energies, mut vectors) = symmetric_eigen(&h).unwrap();
    for (root, index) in [(0, 0), (1, size)] {
        if vectors[(index, root)] < 0.0 {
            for row in 0..dim {
                vectors[(row, root)] = -vectors[(row, root)];
            }
        }
    }
    let transition = vectors.transpose().matmul_seq(&q1).matmul_seq(&vectors)[(0, 1)] * 0.7;
    (energies, transition)
}
#[test]
fn coriolis_and_watson_terms_match_closed_form_and_independent_momentum_matrices() {
    let mut previous: Option<[f64; 2]> = None;
    for scale in [1.0, 0.5, 0.25] {
        let input = model(scale);
        let result = vibrational_spectrum(&input, &Default::default()).unwrap();
        let (dense, mu) = oracle(&input, 8);
        let (larger, mu_larger) = oracle(&input, 10);
        for i in 0..6 {
            assert!((dense[i] - larger[i]).abs() < 1e-12);
        }
        assert!((mu - mu_larger).abs() < 1e-12);
        let error = [
            (result.zero_point_energy - dense[0]).abs(),
            (result.bands[0].transition_dipole[0] - mu).abs(),
        ];
        eprintln!("rotational scale {scale}: energy/moment errors {error:?}");
        if let Some(old) = previous {
            for a in 0..2 {
                assert!(
                    (old[a] / error[a] - 16.0).abs() < 0.5,
                    "{}",
                    old[a] / error[a]
                );
            }
        }
        previous = Some(error);
        let rot = input.rotation.as_ref().unwrap();
        assert_eq!(rot.eckart_defect, None);
        assert_eq!(rot.mode_metric_defect, None);
        assert!((rot.watson_pseudopotential + 9e-4 * scale * scale / 8.0).abs() < 1e-15);
        for (i, state) in result.polyad_occupations.iter().enumerate() {
            let n = state[0] as f64;
            let m = state[1] as f64;
            let angular =
                0.4_f64.powi(2) * ((0.05 / 0.08 + 0.08 / 0.05) * (n + 0.5) * (m + 0.5) - 0.5);
            let expected = (n + 0.5) * 0.05
                + (m + 0.5) * 0.08
                + 0.5 * rot.inverse_inertia[2][2] * angular
                + rot.watson_pseudopotential;
            assert!((result.effective_hamiltonian[(i, i)] - expected).abs() < 1e-13);
        }
        assert!(result.converged && result.added_resonant_states == 0);
        assert!(result.bands[0].mechanical_dipole[0].abs() > 1e-9);
    }
}

fn modes(mol: &Molecule) -> Vec<Vec<[f64; 3]>> {
    let mut h = Matrix::zeros(3 * mol.len(), 3 * mol.len());
    for (a, atom) in mol.atoms.iter().enumerate() {
        for axis in 0..3 {
            h[(3 * a + axis, 3 * a + axis)] = atomic_mass_au(atom.z) * 0.05_f64.powi(2);
        }
    }
    let modes = vibrational_analysis(mol, &h, None).unwrap();
    (0..modes.frequencies_cm.len())
        .map(|i| {
            (0..mol.len())
                .map(|a| {
                    std::array::from_fn(|axis| {
                        modes.modes[(3 * a + axis, i)] / atomic_mass_au(mol.atoms[a].z).sqrt()
                    })
                })
                .collect()
        })
        .collect()
}
#[test]
fn equilibrium_inertia_coriolis_and_watson_are_covariant_under_rigid_motion() {
    let mol = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.5, 1.2, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-1.5, 1.2, 0.0),
        },
    ]);
    let patterns = modes(&mol);
    let result = molecular_rotation(&mol, &patterns, 1e-10).unwrap();
    assert_eq!(result.class, RotorClass::Nonlinear);
    let h = atomic_mass_au(1);
    let o = atomic_mass_au(8);
    let expected = [
        2.0 * h * o / (o + 2.0 * h) * 1.2_f64.powi(2),
        2.0 * h * 1.5_f64.powi(2),
    ];
    assert!((result.principal_moments[0] - expected[0]).abs() < 1e-10);
    assert!((result.principal_moments[1] - expected[1]).abs() < 1e-10);
    assert!((result.principal_moments[2] - expected.iter().sum::<f64>()).abs() < 1e-10);
    assert!(result.coriolis.iter().flatten().any(|v| v.abs() > 0.1));
    let rotate = |p: [f64; 3]| [-p[1], p[0], p[2]];
    let changed = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| Atom {
                z: a.z,
                position: Vec3::new(-a.position.y + 3.0, a.position.x - 2.0, a.position.z + 0.3),
            })
            .collect(),
    );
    let changed_patterns: Vec<_> = patterns
        .iter()
        .map(|p| p.iter().map(|&q| rotate(q)).collect())
        .collect();
    let other = molecular_rotation(&changed, &changed_patterns, 1e-10).unwrap();
    assert!((result.watson_pseudopotential - other.watson_pseudopotential).abs() < 1e-14);
    for (a, b) in result.coriolis.iter().zip(&other.coriolis) {
        let expected = rotate(*a);
        for axis in 0..3 {
            assert!((expected[axis] - b[axis]).abs() < 1e-13);
        }
    }
    let r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    for a in 0..3 {
        for b in 0..3 {
            let expected = (0..3)
                .flat_map(|i| (0..3).map(move |j| r[a][i] * result.inverse_inertia[i][j] * r[b][j]))
                .sum::<f64>();
            assert!((expected - other.inverse_inertia[a][b]).abs() < 1e-13);
        }
    }
    assert!(other.eckart_defect.unwrap() < 1e-13 && other.mode_metric_defect.unwrap() < 1e-13);
}
#[test]
fn linear_rotor_and_diatomic_radial_limits_do_not_invent_an_axial_denominator() {
    assert!(WatsonRotation::from_coefficients(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        vec![[0.0; 3]],
        RotorClass::Linear
    )
    .is_err());
    for numbers in [vec![2, 1], vec![8, 6, 8]] {
        let mol = Molecule::new(
            numbers
                .iter()
                .enumerate()
                .map(|(i, &z)| Atom {
                    z,
                    position: Vec3::new(0.0, 0.0, 2.0 * i as f64),
                })
                .collect(),
        );
        let result = molecular_rotation(&mol, &modes(&mol), 1e-10).unwrap();
        assert_eq!(result.class, RotorClass::Linear);
        assert_eq!(result.inverse_inertia[2][2], 0.0);
        assert_eq!(result.watson_pseudopotential, 0.0);
        assert!((result.principal_moments[1] - result.principal_moments[2]).abs() < 1e-10);
        if numbers.len() == 2 {
            assert!(result.coriolis.iter().flatten().all(|&v| v == 0.0));
            let m1 = atomic_mass_au(2);
            let m2 = atomic_mass_au(1);
            assert!((result.principal_moments[1] - m1 * m2 / (m1 + m2) * 4.0).abs() < 1e-10);
        } else {
            assert!(result.coriolis.iter().flatten().any(|v| v.abs() > 0.1));
        }
    }
}
#[test]
fn coriolis_mode_phase_changes_preserve_the_corrected_strengths_and_energy() {
    let input = model(1.0);
    let a = vibrational_spectrum(&input, &Default::default()).unwrap();
    let mut changed = input.clone();
    for ix in [1, 2] {
        changed.rotation.as_mut().unwrap().coriolis[ix][2] *= -1.0;
    }
    changed.dipole.first[0] *= -1.0;
    let b = vibrational_spectrum(&changed, &Default::default()).unwrap();
    for (a, b) in a.bands.iter().zip(&b.bands) {
        assert!((a.frequency_cm - b.frequency_cm).abs() < 1e-9);
        assert!((a.ir_intensity - b.ir_intensity).abs() < 1e-8);
    }
    changed.rotation.as_mut().unwrap().coriolis[1][2] += 0.1;
    assert!(vibrational_spectrum(&changed, &Default::default()).is_err());
}
