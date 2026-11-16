// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A smooth atom-attached right-handed frame with analytic anchor Jacobians.
use crate::math::Vec3;
use crate::pseudo_sub::frame::{
    build_local_frame_with_connectivity, frame_jacobian, FrameConnectivity,
};
use crate::pseudo_sub::SiteSymmetry;
use crate::system::Molecule;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct AttachedFrame {
    /// Origin, first axis anchor, and plane anchor; all three are distinct.
    pub anchors: [usize; 3],
    /// Fixed local displacement, in Bohr, from the origin anchor.
    pub offset: Vec3,
    pub min_anchor_distance: f64,
    pub min_sine_angle: f64,
}

pub struct FramePoint {
    pub center: Vec3,
    /// d center_i / d R_(atom,j), including the moving frame.
    pub atom_jacobians: Vec<[[f64; 3]; 3]>,
}

impl AttachedFrame {
    pub fn evaluate(&self, molecule: &Molecule) -> Result<FramePoint> {
        let positions: Vec<_> = molecule.atoms.iter().map(|atom| atom.position).collect();
        let [a, b, c] = self.anchors;
        if a == b
            || b == c
            || a == c
            || self.anchors.iter().any(|&i| i >= positions.len())
            || !self.min_anchor_distance.is_finite()
            || self.min_anchor_distance <= 0.0
            || !self.min_sine_angle.is_finite()
            || self.min_sine_angle <= 0.0
            || self.min_sine_angle >= 1.0
            || positions
                .iter()
                .chain(std::iter::once(&self.offset))
                .any(|v| (0..3).any(|i| !v.get(i).is_finite()))
        {
            return Err(DftError::InvalidInput("attached frames require three distinct valid anchors, finite coordinates and positive degeneracy bounds".into()));
        }
        let u = positions[b] - positions[a];
        let v = positions[c] - positions[a];
        let length = u.norm();
        if length < self.min_anchor_distance || v.norm() < self.min_anchor_distance {
            return Err(DftError::InvalidInput(
                "attached frame has coincident or too-close anchors".into(),
            ));
        }
        let e1 = u * (1.0 / length);
        let projection = e1.dot(v);
        let perpendicular = v - e1 * projection;
        let height = perpendicular.norm();
        if height < self.min_sine_angle * v.norm() {
            return Err(DftError::InvalidInput(
                "attached frame anchors are too nearly collinear; no axis switch is permitted"
                    .into(),
            ));
        }
        // Reuse the existing connectivity frame and its stricter five-degree
        // guard. One explicit candidate prevents geometry-dependent reselection.
        // Its columns are [e2,e3,e1], so permute our fixed local offset.
        let frame = build_local_frame_with_connectivity(
            molecule,
            b,
            FrameConnectivity {
                x_atom: a,
                reference_candidates: &[c],
            },
            SiteSymmetry::None,
        )?
        .ok_or_else(|| DftError::InvalidInput("a complete attached frame was required".into()))?;
        let offset = [self.offset.y, self.offset.z, self.offset.x];
        let derivative = frame_jacobian(positions[a], positions[b], positions[c]);
        let mut jacobians = vec![[[0.0; 3]; 3]; positions.len()];
        for (slot, atom) in self.anchors.into_iter().enumerate() {
            for axis in 0..3 {
                for i in 0..3 {
                    jacobians[atom][i][axis] = if slot == 0 && i == axis { 1.0 } else { 0.0 };
                    jacobians[atom][i][axis] += (0..3)
                        .map(|column| {
                            offset[column] * derivative[(3 * column + i, 3 * slot + axis)]
                        })
                        .sum::<f64>();
                }
            }
        }
        Ok(FramePoint {
            center: positions[a] + frame.u.mul_vec(Vec3::new(offset[0], offset[1], offset[2])),
            atom_jacobians: jacobians,
        })
    }
}
