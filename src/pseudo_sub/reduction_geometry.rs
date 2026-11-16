// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Rigid explicit-structure expansion for a declared single sigma port.
//!
//! Coordinates contain retained nuclei followed by one electronic center.
//! This module assigns no charge, electrons, basis, mass or potential to that
//! center. Those are separate model and electron-ledger responsibilities.

use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::pseudo_sub::frame::{
    build_local_frame_with_connectivity, frame_hessian, frame_jacobian, FrameConnectivity,
};
use crate::pseudo_sub::SiteSymmetry;
use crate::system::Molecule;
use crate::{DftError, Result};
use std::collections::HashSet;

/// Explicit atom indices in the original molecule, not inferred neighbors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttachmentPort {
    pub fragment_atom: usize,
    pub host_atom: usize,
    pub reference_atom: usize,
}

/// Initial one-center methyl template at a declared C(sp3)-C sigma boundary.
/// `single_bonds` must specify the complete local neighborhoods of both port
/// carbons. Remote retained connectivity is irrelevant to this rigid map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentSpec {
    pub explicit_atoms: Vec<usize>,
    pub retained_atoms: Vec<usize>,
    pub port: AttachmentPort,
    pub single_bonds: Vec<[usize; 2]>,
    pub electronic_centers: usize,
}

#[derive(Clone, Debug)]
struct MethylGeometry {
    spec: FragmentSpec,
    /// Local offsets of the three removed hydrogens, in original atom order.
    offsets: Vec<(usize, Vec3)>,
    /// Reduced indices of host, center and reference, in frame derivative order.
    anchors: [usize; 3],
}

/// Saved rigid internal geometry. The ordinary identity path has no fragment
/// or additional center and restores exactly the original atom ordering.
#[derive(Clone, Debug)]
pub struct RigidExpansion {
    reference: Molecule,
    retained: Vec<usize>,
    fragment: Option<MethylGeometry>,
}

pub struct ExpandedGeometry {
    pub molecule: Molecule,
    /// J[(3*full_atom+i),(3*reduced_coordinate+j)]. Coordinates are Bohr.
    pub jacobian: Matrix,
}

pub struct BodyFrameKinematics {
    pub center: Vec3,
    pub orientation: Mat3,
    /// Rows are column-major frame entries: 3*column+Cartesian component.
    pub orientation_jacobian: Matrix,
    /// d center_i / d y_j, without any nuclear or electronic interpretation.
    pub center_jacobian: Matrix,
}

/// Complete second-order kinematics of the saved, guarded body frame.
pub struct BodyFrameSecondKinematics {
    /// Existing value/Jacobian interface, evaluated without changing its path.
    pub first: BodyFrameKinematics,
    /// Nine Hessians in the column-major frame-entry order of the Jacobian.
    /// Each matrix has both axes in the reduced Cartesian coordinate order.
    pub orientation_hessians: Vec<Matrix>,
    /// Three center-coordinate Hessians. The declared center is a primary
    /// Cartesian coordinate, so these are analytically zero.
    pub center_hessians: Vec<Matrix>,
}

/// Rigid expansion with its full nonlinear Cartesian curvature.
pub struct ExpandedSecondGeometry {
    pub geometry: ExpandedGeometry,
    /// One Hessian per full Cartesian output, indexed `3*full_atom+axis`.
    /// Both matrix axes are reduced Cartesian coordinates, in Bohr.
    pub coordinate_hessians: Vec<Matrix>,
}

/// Cartesian forces already contain the attached-frame chain rule. Torques
/// are diagnostics for independent rigid orientation, not extra Cartesian
/// forces to add a second time to the anchors.
pub struct GeneralizedForceReport {
    pub reduced_forces: Vec<Vec3>,
    pub fragment_translation_force: Option<Vec3>,
    pub fragment_torque_global: Option<Vec3>,
    pub fragment_torque_body: Option<Vec3>,
    pub cartesian_units: &'static str,
    pub torque_units: &'static str,
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn finite(point: Vec3) -> bool {
    (0..3).all(|axis| point.get(axis).is_finite())
}

fn frame(molecule: &Molecule, port: &AttachmentPort) -> Result<Mat3> {
    build_local_frame_with_connectivity(
        molecule,
        port.fragment_atom,
        FrameConnectivity {
            x_atom: port.host_atom,
            reference_candidates: &[port.reference_atom],
        },
        SiteSymmetry::None,
    )?
    .map(|value| value.u)
    .ok_or_else(|| DftError::InvalidInput("a complete rigid fragment frame was required".into()))
}

impl RigidExpansion {
    pub fn identity(reference: &Molecule) -> Result<Self> {
        if reference.is_empty() || reference.atoms.iter().any(|atom| !finite(atom.position)) {
            return invalid("identity expansion requires a nonempty finite molecule");
        }
        Ok(Self {
            reference: reference.clone(),
            retained: (0..reference.len()).collect(),
            fragment: None,
        })
    }

    pub fn methyl(reference: &Molecule, spec: FragmentSpec) -> Result<Self> {
        if spec.electronic_centers != 1 {
            return Err(DftError::Unsupported {
                feature: "rigid methyl generation with multiple electronic centers".into(),
                missing: "a separately validated two- or three-center template".into(),
            });
        }
        if reference.is_empty() || reference.atoms.iter().any(|atom| !finite(atom.position)) {
            return invalid("rigid methyl expansion requires finite reference coordinates");
        }
        let n = reference.len();
        let fragment: HashSet<_> = spec.explicit_atoms.iter().copied().collect();
        let retained: HashSet<_> = spec.retained_atoms.iter().copied().collect();
        if fragment.len() != 4
            || spec.explicit_atoms.len() != 4
            || retained.len() != spec.retained_atoms.len()
            || !fragment.is_disjoint(&retained)
            || fragment.len() + retained.len() != n
            || fragment.iter().chain(&retained).any(|&atom| atom >= n)
        {
            return invalid("one methyl fragment and retained atoms must form an explicit disjoint complete partition");
        }
        let port = &spec.port;
        if !fragment.contains(&port.fragment_atom)
            || !retained.contains(&port.host_atom)
            || !retained.contains(&port.reference_atom)
            || port.host_atom == port.reference_atom
            || reference.atoms[port.fragment_atom].z != 6
            || reference.atoms[port.host_atom].z != 6
            || fragment
                .iter()
                .any(|&atom| atom != port.fragment_atom && reference.atoms[atom].z != 1)
        {
            return invalid("initial rigid reduction requires a methyl carbon, its three hydrogens and an explicit retained carbon/reference port");
        }
        let mut edges = HashSet::new();
        let mut neighbors = vec![Vec::new(); n];
        for &[a, b] in &spec.single_bonds {
            if a >= n || b >= n || a == b || !edges.insert([a.min(b), a.max(b)]) {
                return invalid("single-bond topology contains an invalid or repeated pair");
            }
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
        let mut expected: HashSet<_> = fragment
            .iter()
            .copied()
            .filter(|&a| a != port.fragment_atom)
            .collect();
        expected.insert(port.host_atom);
        let actual: HashSet<_> = neighbors[port.fragment_atom].iter().copied().collect();
        if actual != expected
            || neighbors[port.host_atom].len() != 4
            || !neighbors[port.host_atom].contains(&port.fragment_atom)
            || !neighbors[port.host_atom].contains(&port.reference_atom)
            || neighbors[port.host_atom]
                .iter()
                .any(|a| *a != port.fragment_atom && !retained.contains(a))
            || fragment
                .iter()
                .filter(|&&a| a != port.fragment_atom)
                .any(|&h| neighbors[h] != [port.fragment_atom])
        {
            return invalid("declared port topology must contain complete single-bonded sp3 carbon neighborhoods and three internal C-H bonds");
        }
        let u = frame(reference, port)?;
        let origin = reference.atoms[port.fragment_atom].position;
        let offsets = spec
            .explicit_atoms
            .iter()
            .copied()
            .filter(|&a| a != port.fragment_atom)
            .map(|atom| {
                let displacement = reference.atoms[atom].position - origin;
                (
                    atom,
                    Vec3::new(
                        u.col[0].dot(displacement),
                        u.col[1].dot(displacement),
                        u.col[2].dot(displacement),
                    ),
                )
            })
            .collect();
        let position_of = |atom| {
            spec.retained_atoms
                .iter()
                .position(|&a| a == atom)
                .expect("validated retained port")
        };
        let anchors = [
            position_of(port.host_atom),
            spec.retained_atoms.len(),
            position_of(port.reference_atom),
        ];
        Ok(Self {
            reference: reference.clone(),
            retained: spec.retained_atoms.clone(),
            fragment: Some(MethylGeometry {
                spec,
                offsets,
                anchors,
            }),
        })
    }

    /// Bind an existing rigid internal geometry without regenerating it from
    /// application Cartesian coordinates. The ordered H indices must match the
    /// explicit specification; offset coefficients are retained bit for bit.
    pub fn methyl_with_internal_offsets(
        reference: &Molecule,
        spec: FragmentSpec,
        offsets: &[(usize, Vec3)],
    ) -> Result<Self> {
        let mut result = Self::methyl(reference, spec)?;
        let fragment = result.fragment.as_mut().ok_or_else(|| {
            DftError::InvalidInput("methyl binding requires a fragment map".into())
        })?;
        if offsets.len() != 3
            || fragment
                .offsets
                .iter()
                .zip(offsets)
                .any(|((expected, _), (actual, point))| expected != actual || !finite(*point))
        {
            return invalid(
                "transferred methyl offsets must match the ordered three finite hydrogen offsets",
            );
        }
        fragment.offsets = offsets.to_vec();
        // Reuse the same expansion and frame path as all subsequent geometry
        // evaluations. There is no alternate rotation or derivative formula.
        let coordinates = result.reference_coordinates();
        result.reference = result.expand(&coordinates)?;
        Ok(result)
    }

    pub fn full_atom_count(&self) -> usize {
        self.reference.len()
    }

    pub fn reference_molecule(&self) -> &Molecule {
        &self.reference
    }

    pub fn fragment_spec(&self) -> Option<&FragmentSpec> {
        self.fragment.as_ref().map(|fragment| &fragment.spec)
    }

    pub fn internal_offsets(&self) -> &[(usize, Vec3)] {
        self.fragment
            .as_ref()
            .map_or(&[], |fragment| fragment.offsets.as_slice())
    }
    pub fn reduced_coordinate_count(&self) -> usize {
        self.retained.len() + usize::from(self.fragment.is_some())
    }
    pub fn retained_atoms(&self) -> &[usize] {
        &self.retained
    }
    pub fn fragment_atoms(&self) -> &[usize] {
        self.fragment
            .as_ref()
            .map_or(&[], |value| value.spec.explicit_atoms.as_slice())
    }
    pub fn electronic_center_index(&self) -> Option<usize> {
        self.fragment.as_ref().map(|_| self.retained.len())
    }

    /// Coordinate projection only; no electrons, charges or AO counts are inferred.
    pub fn reduced_coordinates(&self, full: &Molecule) -> Result<Vec<Vec3>> {
        if full.len() != self.reference.len()
            || full
                .atoms
                .iter()
                .zip(&self.reference.atoms)
                .any(|(a, b)| a.z != b.z || !finite(a.position))
        {
            return invalid("coordinate projection requires the saved full atom order and labels");
        }
        let mut result: Vec<_> = self
            .retained
            .iter()
            .map(|&a| full.atoms[a].position)
            .collect();
        if let Some(fragment) = &self.fragment {
            result.push(full.atoms[fragment.spec.port.fragment_atom].position);
        }
        Ok(result)
    }

    pub fn reference_coordinates(&self) -> Vec<Vec3> {
        self.reduced_coordinates(&self.reference)
            .expect("validated reference geometry")
    }

    /// The port frame is shared by geometry, electrostatic multipoles and
    /// nonlocal projectors. A role-only center has no independent orientation;
    /// this derivative expresses its declared anchor constraint.
    pub fn body_frame_kinematics(
        &self,
        coordinates: &[Vec3],
    ) -> Result<Option<BodyFrameKinematics>> {
        if coordinates.len() != self.reduced_coordinate_count()
            || coordinates.iter().any(|&point| !finite(point))
        {
            return invalid("frame kinematics requires the saved finite reduced coordinates");
        }
        let Some(fragment) = &self.fragment else {
            return Ok(None);
        };
        let mut skeleton = self.reference.clone();
        for (reduced, &full) in self.retained.iter().enumerate() {
            skeleton.atoms[full].position = coordinates[reduced];
        }
        let site = self.retained.len();
        let port = &fragment.spec.port;
        skeleton.atoms[port.fragment_atom].position = coordinates[site];
        let orientation = frame(&skeleton, port)?;
        let derivative = frame_jacobian(
            skeleton.atoms[port.host_atom].position,
            skeleton.atoms[port.fragment_atom].position,
            skeleton.atoms[port.reference_atom].position,
        );
        let mut orientation_jacobian = Matrix::zeros(9, 3 * coordinates.len());
        for (slot, &reduced) in fragment.anchors.iter().enumerate() {
            for row in 0..9 {
                for axis in 0..3 {
                    orientation_jacobian[(row, 3 * reduced + axis)] =
                        derivative[(row, 3 * slot + axis)];
                }
            }
        }
        let mut center_jacobian = Matrix::zeros(3, 3 * coordinates.len());
        for axis in 0..3 {
            center_jacobian[(axis, 3 * site + axis)] = 1.;
        }
        Ok(Some(BodyFrameKinematics {
            center: coordinates[site],
            orientation,
            orientation_jacobian,
            center_jacobian,
        }))
    }

    pub fn evaluate(&self, coordinates: &[Vec3]) -> Result<ExpandedGeometry> {
        if coordinates.len() != self.reduced_coordinate_count()
            || coordinates.iter().any(|&p| !finite(p))
        {
            return invalid(
                "rigid expansion requires one finite vector per saved reduced coordinate",
            );
        }
        let mut molecule = self.reference.clone();
        let mut jacobian = Matrix::zeros(3 * self.full_atom_count(), 3 * coordinates.len());
        for (reduced, &full) in self.retained.iter().enumerate() {
            molecule.atoms[full].position = coordinates[reduced];
            for axis in 0..3 {
                jacobian[(3 * full + axis, 3 * reduced + axis)] = 1.0;
            }
        }
        if let Some(fragment) = &self.fragment {
            let site = self.retained.len();
            let port = &fragment.spec.port;
            molecule.atoms[port.fragment_atom].position = coordinates[site];
            for axis in 0..3 {
                jacobian[(3 * port.fragment_atom + axis, 3 * site + axis)] = 1.0;
            }
            let frame = self.body_frame_kinematics(coordinates)?.ok_or_else(|| {
                DftError::InvalidInput("methyl expansion requires its saved frame".into())
            })?;
            for &(full, offset) in &fragment.offsets {
                molecule.atoms[full].position = frame.center + frame.orientation.mul_vec(offset);
                for reduced in 0..coordinates.len() {
                    for i in 0..3 {
                        for axis in 0..3 {
                            let derivative = (0..3)
                                .map(|column| {
                                    offset.get(column)
                                        * frame.orientation_jacobian
                                            [(3 * column + i, 3 * reduced + axis)]
                                })
                                .sum::<f64>();
                            jacobian[(3 * full + i, 3 * reduced + axis)] = derivative
                                + if reduced == site && i == axis {
                                    1.0
                                } else {
                                    0.0
                                };
                        }
                    }
                }
            }
        }
        Ok(ExpandedGeometry { molecule, jacobian })
    }

    /// Analytic second frame derivatives with the same explicit anchors and
    /// domain/refusal checks as `body_frame_kinematics`. Internal offsets and
    /// the reference-atom choice stay fixed throughout this chart.
    pub fn body_frame_second_kinematics(
        &self,
        coordinates: &[Vec3],
    ) -> Result<Option<BodyFrameSecondKinematics>> {
        let Some(first) = self.body_frame_kinematics(coordinates)? else {
            return Ok(None);
        };
        let fragment = self.fragment.as_ref().ok_or_else(|| {
            DftError::InvalidInput("second frame derivatives require a fragment map".into())
        })?;
        let local = frame_hessian(
            coordinates[fragment.anchors[0]],
            coordinates[fragment.anchors[1]],
            coordinates[fragment.anchors[2]],
        );
        let n = 3 * coordinates.len();
        let mut orientation_hessians = (0..9).map(|_| Matrix::zeros(n, n)).collect::<Vec<_>>();
        for entry in 0..9 {
            for (slot_a, &anchor_a) in fragment.anchors.iter().enumerate() {
                for (slot_b, &anchor_b) in fragment.anchors.iter().enumerate() {
                    for axis_a in 0..3 {
                        for axis_b in 0..3 {
                            orientation_hessians[entry]
                                [(3 * anchor_a + axis_a, 3 * anchor_b + axis_b)] =
                                local[entry][(3 * slot_a + axis_a, 3 * slot_b + axis_b)];
                        }
                    }
                }
            }
        }
        if orientation_hessians
            .iter()
            .flat_map(|value| value.as_slice())
            .any(|value| !value.is_finite())
        {
            return invalid("second frame derivatives are not finite");
        }
        Ok(Some(BodyFrameSecondKinematics {
            first,
            orientation_hessians,
            center_hessians: (0..3).map(|_| Matrix::zeros(n, n)).collect(),
        }))
    }

    /// Value, Jacobian and analytic Hessians of `R = L(y, xi)` at fixed saved
    /// internal coordinates `xi`. Retained atoms and the electronic center are
    /// linear coordinates; only the attached rigid offsets have curvature.
    pub fn evaluate_second(&self, coordinates: &[Vec3]) -> Result<ExpandedSecondGeometry> {
        let geometry = self.evaluate(coordinates)?;
        let n = 3 * coordinates.len();
        let mut coordinate_hessians = (0..3 * self.full_atom_count())
            .map(|_| Matrix::zeros(n, n))
            .collect::<Vec<_>>();
        if let Some(fragment) = &self.fragment {
            let frame = self
                .body_frame_second_kinematics(coordinates)?
                .ok_or_else(|| {
                    DftError::InvalidInput("second rigid derivatives require a frame".into())
                })?;
            for &(full, offset) in &fragment.offsets {
                for axis in 0..3 {
                    for a in 0..n {
                        for b in 0..n {
                            coordinate_hessians[3 * full + axis][(a, b)] = (0..3)
                                .map(|column| {
                                    offset.get(column)
                                        * frame.orientation_hessians[3 * column + axis][(a, b)]
                                })
                                .sum();
                        }
                    }
                }
            }
        }
        if coordinate_hessians
            .iter()
            .flat_map(|value| value.as_slice())
            .any(|value| !value.is_finite())
        {
            return invalid("second rigid expansion derivatives are not finite");
        }
        Ok(ExpandedSecondGeometry {
            geometry,
            coordinate_hessians,
        })
    }

    /// Pull back an energy Hessian through the nonlinear rigid map.
    /// `full_gradient` is the energy gradient, not a force. The curvature term
    /// is essential even when the reduced gradient vanishes: reduced
    /// stationarity does not imply stationarity in the eliminated directions.
    /// This performs `J^T H J + sum_i gradient_i L_i''` exactly once; callers
    /// must not add an independent frame/torque correction afterward.
    pub fn pullback_hessian(
        &self,
        coordinates: &[Vec3],
        full_gradient: &[Vec3],
        full_hessian: &Matrix,
    ) -> Result<Matrix> {
        let m = 3 * self.full_atom_count();
        if full_gradient.len() != self.full_atom_count()
            || full_gradient.iter().any(|&value| !finite(value))
            || full_hessian.rows != m
            || full_hessian.cols != m
            || full_hessian
                .as_slice()
                .iter()
                .any(|value| !value.is_finite())
        {
            return invalid(
                "Hessian pullback requires a finite full gradient and full Cartesian Hessian",
            );
        }
        let expanded = self.evaluate_second(coordinates)?;
        let j = &expanded.geometry.jacobian;
        let mut result = j.transpose().matmul_seq(full_hessian).matmul_seq(j);
        for full in 0..self.full_atom_count() {
            for axis in 0..3 {
                let coefficient = full_gradient[full].get(axis);
                let curvature = &expanded.coordinate_hessians[3 * full + axis];
                for a in 0..result.rows {
                    for b in 0..result.cols {
                        result[(a, b)] += coefficient * curvature[(a, b)];
                    }
                }
            }
        }
        if result.as_slice().iter().any(|value| !value.is_finite()) {
            return invalid("pulled-back energy Hessian is not finite");
        }
        Ok(result)
    }

    /// Pull back `d^2 E / dR dq` for parameter-independent saved geometry and
    /// fixed internal coordinates. Columns keep the caller's parameter order.
    /// Geometry that itself depends on q requires an additional model contract
    /// and is not silently treated as this fixed-map case.
    pub fn pullback_mixed_derivatives(
        &self,
        coordinates: &[Vec3],
        full_mixed: &Matrix,
    ) -> Result<Matrix> {
        if full_mixed.rows != 3 * self.full_atom_count()
            || full_mixed.as_slice().iter().any(|value| !value.is_finite())
        {
            return invalid("mixed pullback requires finite full Cartesian rows and the declared parameter columns");
        }
        let expanded = self.evaluate(coordinates)?;
        let result = expanded.jacobian.transpose().matmul_seq(full_mixed);
        if result.as_slice().iter().any(|value| !value.is_finite()) {
            return invalid("pulled-back mixed derivatives are not finite");
        }
        Ok(result)
    }

    pub fn expand(&self, coordinates: &[Vec3]) -> Result<Molecule> {
        Ok(self.evaluate(coordinates)?.molecule)
    }

    pub fn pullback_forces(&self, coordinates: &[Vec3], full_forces: &[Vec3]) -> Result<Vec<Vec3>> {
        if full_forces.len() != self.full_atom_count() || full_forces.iter().any(|&f| !finite(f)) {
            return invalid(
                "force pullback requires one finite full Cartesian force per explicit atom",
            );
        }
        let expanded = self.evaluate(coordinates)?;
        let mut result = vec![Vec3::zero(); coordinates.len()];
        for (reduced, value) in result.iter_mut().enumerate() {
            let components: [f64; 3] = std::array::from_fn(|axis| {
                full_forces
                    .iter()
                    .enumerate()
                    .map(|(full, force)| {
                        (0..3)
                            .map(|i| {
                                expanded.jacobian[(3 * full + i, 3 * reduced + axis)] * force.get(i)
                            })
                            .sum::<f64>()
                    })
                    .sum()
            });
            *value = Vec3::new(components[0], components[1], components[2]);
        }
        Ok(result)
    }

    pub fn force_report(
        &self,
        coordinates: &[Vec3],
        full_forces: &[Vec3],
    ) -> Result<GeneralizedForceReport> {
        let reduced_forces = self.pullback_forces(coordinates, full_forces)?;
        let expanded = self.expand(coordinates)?;
        let (translation, torque, body) = if let Some(fragment) = &self.fragment {
            let origin = expanded.atoms[fragment.spec.port.fragment_atom].position;
            let mut translation = Vec3::zero();
            let mut torque = Vec3::zero();
            for &atom in &fragment.spec.explicit_atoms {
                translation += full_forces[atom];
                torque += (expanded.atoms[atom].position - origin).cross(full_forces[atom]);
            }
            let u = frame(&expanded, &fragment.spec.port)?;
            let body = Vec3::new(
                u.col[0].dot(torque),
                u.col[1].dot(torque),
                u.col[2].dot(torque),
            );
            (Some(translation), Some(torque), Some(body))
        } else {
            (None, None, None)
        };
        Ok(GeneralizedForceReport {
            reduced_forces,
            fragment_translation_force: translation,
            fragment_torque_global: torque,
            fragment_torque_body: body,
            cartesian_units: "hartree/bohr",
            torque_units: "hartree/radian",
        })
    }
}
