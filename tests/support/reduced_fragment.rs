// SPDX-License-Identifier: Apache-2.0
//! Fixed construction and independently held-out scaffolds; no tests here.
use dft_rs::math::Vec3;
use dft_rs::pseudo_sub::reduction_geometry::{AttachmentPort, FragmentSpec, RigidExpansion};
use dft_rs::system::{Atom, Molecule};
pub const OFFSETS: [f64; 9] = [-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2];
pub fn construction_options() -> dft_rs::pseudo_sub::reduction::GenerationOptions {
    let (source, spec) = ethane();
    let map = RigidExpansion::methyl(&source, spec).unwrap();
    let y = map.reference_coordinates();
    let ci = map.electronic_center_index().unwrap();
    let frame = map.body_frame_kinematics(&y).unwrap().unwrap();
    let direction = (y[ci] - y[0]) / (y[ci] - y[0]).norm();
    let mut points: Vec<_> = (0..4).map(|i| source.atoms[i].position).collect();
    for atom in 0..4 {
        for axis in 0..3 {
            for sign in [-1., 1.] {
                let mut d = [0.; 3];
                d[axis] = sign * 0.05;
                points.push(source.atoms[atom].position + Vec3::new(d[0], d[1], d[2]));
            }
        }
    }
    for r in [0.6, 0.9, 1.2, 1.6, 2., 2.4, 3.] {
        points.push(frame.center - direction * r);
    }
    for axis in 0..3 {
        for sign in [-1., 1.] {
            for r in [1.1, 2.2] {
                points.push(frame.center + frame.orientation.col[axis] * (sign * r));
            }
        }
    }
    assert_eq!(points.len(), 47);
    dft_rs::pseudo_sub::reduction::GenerationOptions {
        electrostatic_points: points,
        electrostatic_weight: 1e6,
        ..Default::default()
    }
}
pub fn scan(map: &RigidExpansion, host_atom: usize) -> Vec<Vec<Vec3>> {
    let y = map.reference_coordinates();
    let center = map.electronic_center_index().unwrap();
    let host = map
        .retained_atoms()
        .iter()
        .position(|&i| i == host_atom)
        .unwrap();
    let direction = (y[center] - y[host]) / (y[center] - y[host]).norm();
    OFFSETS
        .iter()
        .map(|&d| {
            let mut q = y.clone();
            q[center] += direction * d;
            q
        })
        .collect()
}
fn molecule(labels: &[u8], positions: &[[f64; 3]]) -> Molecule {
    assert_eq!(labels.len(), positions.len());
    Molecule::new(
        labels
            .iter()
            .zip(positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

pub fn ethane() -> (Molecule, FragmentSpec) {
    let source = molecule(
        &[6, 1, 1, 1, 6, 1, 1, 1],
        &[
            [-1.46, -0.14, 0.23],
            [-2.13, 1.69, 0.68],
            [-2.28, -1.57, 1.31],
            [-2.19, -0.43, -1.65],
            [1.35, 0.17, -0.12],
            [2.05, 2.08, 0.16],
            [2.10, -0.92, 1.45],
            [2.01, -0.84, -1.67],
        ],
    );
    (
        source,
        FragmentSpec {
            explicit_atoms: vec![4, 5, 6, 7],
            retained_atoms: vec![0, 1, 2, 3],
            port: AttachmentPort {
                fragment_atom: 4,
                host_atom: 0,
                reference_atom: 1,
            },
            single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
            electronic_centers: 1,
        },
    )
}

/// C3H8 sigma-connected scaffold, kept separate from ethane construction.
pub fn propane() -> (Molecule, FragmentSpec) {
    let source = molecule(
        &[6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1],
        &[
            [2.8, 0.1, 0.2],
            [0.0, 0.0, 0.0],
            [-1.0, 2.6, 0.3],
            [3.5, 1.8, 0.9],
            [3.6, -1.0, 1.6],
            [3.4, -0.8, -1.6],
            [-0.6, -0.8, 1.7],
            [-0.7, -0.9, -1.6],
            [-0.3, 3.5, 1.9],
            [-3.0, 2.7, 0.2],
            [-0.4, 3.6, -1.3],
        ],
    );
    (
        source,
        FragmentSpec {
            explicit_atoms: vec![0, 3, 4, 5],
            retained_atoms: vec![1, 2, 6, 7, 8, 9, 10],
            port: AttachmentPort {
                fragment_atom: 0,
                host_atom: 1,
                reference_atom: 2,
            },
            single_bonds: vec![
                [0, 1],
                [0, 3],
                [0, 4],
                [0, 5],
                [1, 2],
                [1, 6],
                [1, 7],
                [2, 8],
                [2, 9],
                [2, 10],
            ],
            electronic_centers: 1,
        },
    )
}

/// Branched C4H10 sigma-connected scaffold, not construction data.
pub fn isobutane() -> (Molecule, FragmentSpec) {
    let source = molecule(
        &[6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[
            [2.8, 0.1, 0.2],
            [0.0, 0.0, 0.0],
            [-1.0, 2.6, 0.3],
            [-0.9, -1.4, 2.3],
            [3.5, 1.8, 0.9],
            [3.6, -1.0, 1.6],
            [3.4, -0.8, -1.6],
            [-0.3, 3.5, 1.9],
            [-3.0, 2.7, 0.2],
            [-0.4, 3.6, -1.3],
            [0.0, -3.2, 2.7],
            [-2.9, -1.7, 2.1],
            [-0.3, -0.2, 3.8],
            [0.0, -1.0, -1.8],
        ],
    );
    (
        source,
        FragmentSpec {
            explicit_atoms: vec![0, 4, 5, 6],
            retained_atoms: vec![1, 2, 3, 7, 8, 9, 10, 11, 12, 13],
            port: AttachmentPort {
                fragment_atom: 0,
                host_atom: 1,
                reference_atom: 2,
            },
            single_bonds: vec![
                [0, 1],
                [0, 4],
                [0, 5],
                [0, 6],
                [1, 2],
                [1, 3],
                [1, 13],
                [2, 7],
                [2, 8],
                [2, 9],
                [3, 10],
                [3, 11],
                [3, 12],
            ],
            electronic_centers: 1,
        },
    )
}
