// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Equilibrium Watson kinetic terms at vibrational second perturbative order.
//! Electron masses and Bohr define inertia; Coriolis tensors use mass-weighted
//! Cartesian normal modes in the same frame as the full inverse inertia.

use crate::constants::atomic_mass_au;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::Molecule;
use crate::{DftError, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotorClass {
    Nonlinear,
    Linear,
}

#[derive(Clone, Debug)]
pub struct WatsonRotation {
    pub class: RotorClass,
    pub inverse_inertia: [[f64; 3]; 3],
    pub principal_moments: [f64; 3],
    /// Ordered mode pair first, Cartesian angular-momentum axis last.
    pub coriolis: Vec<[f64; 3]>,
    pub watson_pseudopotential: f64,
    /// Absent for supplied coefficients without measured molecular mode geometry.
    pub eckart_defect: Option<f64>,
    pub mode_metric_defect: Option<f64>,
}
impl WatsonRotation {
    /// Supplied kinetic coefficients, with a physical nonlinear or linear rotor.
    pub fn from_coefficients(
        inverse_inertia: [[f64; 3]; 3],
        coriolis: Vec<[f64; 3]>,
        class: RotorClass,
    ) -> Result<Self> {
        let mut matrix = Matrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                matrix[(i, j)] = inverse_inertia[i][j];
            }
        }
        if matrix.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput("nonfinite inverse inertia".into()));
        }
        let scale = matrix
            .as_slice()
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        for i in 0..3 {
            for j in 0..3 {
                if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-12 * scale {
                    return Err(DftError::InvalidInput(
                        "inverse inertia must be symmetric before any symmetrization".into(),
                    ));
                }
            }
        }
        let (eig, _) = symmetric_eigen(&matrix)?;
        let rank = eig.iter().filter(|&&v| v > 1e-12 * scale).count();
        if eig.iter().any(|&v| v < -1e-12 * scale)
            || class == RotorClass::Linear && (eig[2] - eig[1]).abs() > 1e-12 * scale
            || rank
                != match class {
                    RotorClass::Linear => 2,
                    RotorClass::Nonlinear => 3,
                }
        {
            return Err(DftError::InvalidInput(
                "inverse inertia rank must match the positive physical rotor space".into(),
            ));
        }
        let mut moments: Vec<_> = eig
            .iter()
            .map(|&v| if v > 1e-12 * scale { 1.0 / v } else { 0.0 })
            .collect();
        moments.sort_by(f64::total_cmp);
        let watson_pseudopotential = match class {
            RotorClass::Nonlinear => -(0..3).map(|i| inverse_inertia[i][i]).sum::<f64>() / 8.0,
            // The linear isomorphic Hamiltonian has no nonlinear-rotor U term.
            RotorClass::Linear => 0.0,
        };
        Ok(Self {
            class,
            inverse_inertia,
            principal_moments: [moments[0], moments[1], moments[2]],
            coriolis,
            watson_pseudopotential,
            eckart_defect: None,
            mode_metric_defect: None,
        })
    }
    pub(super) fn validate(&self, m: usize, tol: f64) -> Result<()> {
        if self.coriolis.len() != m.saturating_mul(m)
            || self.coriolis.iter().flatten().any(|v| !v.is_finite())
            || !self.watson_pseudopotential.is_finite()
        {
            return Err(DftError::InvalidInput(
                "invalid Coriolis tensor shape or nonfinite Watson term".into(),
            ));
        }
        for i in 0..m {
            for j in 0..m {
                for a in 0..3 {
                    if (self.coriolis[m * i + j][a] + self.coriolis[m * j + i][a]).abs() > tol {
                        return Err(DftError::InvalidInput(
                            "raw Coriolis tensor must be antisymmetric in its mode indices".into(),
                        ));
                    }
                }
            }
        }
        let checked =
            Self::from_coefficients(self.inverse_inertia, self.coriolis.clone(), self.class)?;
        if (self.watson_pseudopotential - checked.watson_pseudopotential).abs()
            > tol * checked.watson_pseudopotential.abs().max(1e-15)
        {
            return Err(DftError::InvalidInput(
                "Watson pseudopotential is inconsistent with the rotor geometry".into(),
            ));
        }
        Ok(())
    }
}

/// Build inertia and Coriolis coefficients analytically from Eckart modes.
pub fn molecular_rotation(
    mol: &Molecule,
    patterns: &[Vec<[f64; 3]>],
    tol: f64,
) -> Result<WatsonRotation> {
    if mol.atoms.iter().any(|a| {
        [a.position.x, a.position.y, a.position.z]
            .iter()
            .any(|v| !v.is_finite())
    }) || patterns.is_empty()
        || patterns
            .iter()
            .any(|p| p.len() != mol.len() || p.iter().flatten().any(|v| !v.is_finite()))
        || !tol.is_finite()
        || tol <= 0.0
    {
        return Err(DftError::InvalidInput(
            "invalid molecular rotational mode request".into(),
        ));
    }
    let masses: Vec<_> = mol.atoms.iter().map(|a| atomic_mass_au(a.z)).collect();
    if masses.iter().any(|&m| m <= 0.0) {
        return Err(DftError::InvalidInput(
            "rotational kinetic terms require positive nuclear masses".into(),
        ));
    }
    let mass: f64 = masses.iter().sum();
    let center = mol
        .atoms
        .iter()
        .zip(&masses)
        .fold(Vec3::zero(), |r, (a, &m)| r + a.position * m)
        / mass;
    let positions: Vec<_> = mol.atoms.iter().map(|a| a.position - center).collect();
    let mut inertia = Matrix::zeros(3, 3);
    for (r, &mass) in positions.iter().zip(&masses) {
        let coords = [r.x, r.y, r.z];
        let r2 = coords.iter().map(|x| x * x).sum::<f64>();
        for a in 0..3 {
            for b in 0..3 {
                inertia[(a, b)] += mass * (if a == b { r2 } else { 0.0 } - coords[a] * coords[b]);
            }
        }
    }
    let (moments, axes) = symmetric_eigen(&inertia)?;
    let maximum = moments[2];
    if maximum <= 0.0 {
        return Err(DftError::InvalidInput(
            "molecular inertia has no rotational degrees of freedom".into(),
        ));
    }
    let class = if moments[0] <= maximum * 1e-12 {
        RotorClass::Linear
    } else {
        RotorClass::Nonlinear
    };
    let mut inverse = [[0.0; 3]; 3];
    for k in 0..3 {
        if moments[k] > maximum * 1e-12 {
            for a in 0..3 {
                for b in 0..3 {
                    inverse[a][b] += axes[(a, k)] * axes[(b, k)] / moments[k];
                }
            }
        }
    }
    let m = patterns.len();
    let mut coriolis = vec![[0.0; 3]; m * m];
    let mut metric_defect = 0.0_f64;
    let mut eckart_defect = 0.0_f64;
    for i in 0..m {
        let mut rotation = Vec3::zero();
        for (a, &mass) in masses.iter().enumerate() {
            let p = patterns[i][a];
            rotation += positions[a].cross(Vec3::new(p[0], p[1], p[2])) * mass;
        }
        // Normalized translation/rotation projections are dimensionless.
        for axis in 0..3 {
            let translation_projection = masses
                .iter()
                .enumerate()
                .map(|(a, &mass)| mass * patterns[i][a][axis])
                .sum::<f64>()
                / mass.sqrt();
            eckart_defect = eckart_defect.max(translation_projection.abs());
        }
        for k in 0..3 {
            if moments[k] > maximum * 1e-12 {
                let projection = axes[(0, k)] * rotation.x
                    + axes[(1, k)] * rotation.y
                    + axes[(2, k)] * rotation.z;
                eckart_defect = eckart_defect.max(projection.abs() / moments[k].sqrt());
            }
        }
        for j in 0..m {
            let mut dot = 0.0;
            let mut zeta = Vec3::zero();
            for (a, &mass) in masses.iter().enumerate() {
                let p = patterns[i][a];
                let q = patterns[j][a];
                dot += mass * (0..3).map(|axis| p[axis] * q[axis]).sum::<f64>();
                zeta += Vec3::new(p[0], p[1], p[2]).cross(Vec3::new(q[0], q[1], q[2])) * mass;
            }
            metric_defect = metric_defect.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
            coriolis[m * i + j] = [zeta.x, zeta.y, zeta.z];
        }
    }
    if metric_defect > tol || eckart_defect > tol {
        return Err(DftError::InvalidInput(
            "rotational kinetic terms require orthonormal mass-weighted Eckart modes".into(),
        ));
    }
    let mut result = WatsonRotation::from_coefficients(inverse, coriolis, class)?;
    result.principal_moments = [moments[0], moments[1], moments[2]];
    result.eckart_defect = Some(eckart_defect);
    result.mode_metric_defect = Some(metric_defect);
    Ok(result)
}

// pi = i*A. Q_i P_j acts right to left, with real coefficients of P_j/i.
fn angular_action(
    rot: &WatsonRotation,
    axis: usize,
    state: &[usize],
    omega: &[f64],
) -> BTreeMap<Vec<usize>, f64> {
    let m = state.len();
    let mut result = BTreeMap::new();
    for i in 0..m {
        for j in 0..m {
            let zeta = rot.coriolis[m * i + j][axis];
            if zeta == 0.0 {
                continue;
            }
            for up in [false, true] {
                if !up && state[j] == 0 {
                    continue;
                }
                let mut middle = state.to_vec();
                let pc = if up {
                    middle[j] += 1;
                    ((state[j] + 1) as f64 * omega[j] / 2.0).sqrt()
                } else {
                    middle[j] -= 1;
                    -(state[j] as f64 * omega[j] / 2.0).sqrt()
                };
                for raise in [false, true] {
                    if !raise && middle[i] == 0 {
                        continue;
                    }
                    let mut to = middle.clone();
                    let qc = if raise {
                        to[i] += 1;
                        ((middle[i] + 1) as f64 / (2.0 * omega[i])).sqrt()
                    } else {
                        to[i] -= 1;
                        (middle[i] as f64 / (2.0 * omega[i])).sqrt()
                    };
                    *result.entry(to).or_insert(0.0) += zeta * pc * qc;
                }
            }
        }
    }
    result.retain(|_, v| *v != 0.0);
    result
}
pub(super) fn kinetic_action(
    rot: &WatsonRotation,
    state: &[usize],
    omega: &[f64],
) -> BTreeMap<Vec<usize>, f64> {
    let mut result = BTreeMap::from([(state.to_vec(), rot.watson_pseudopotential)]);
    for a in 0..3 {
        for b in 0..3 {
            if rot.inverse_inertia[a][b] == 0.0 {
                continue;
            }
            for (middle, right) in angular_action(rot, b, state, omega) {
                for (to, left) in angular_action(rot, a, &middle, omega) {
                    *result.entry(to).or_insert(0.0) -=
                        0.5 * rot.inverse_inertia[a][b] * left * right;
                }
            }
        }
    }
    result.retain(|_, v| *v != 0.0);
    result
}
