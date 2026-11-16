// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Harmonic vibrational analysis: mass-weighting, Eckart (translation +
//! rotation) projection, frequencies, and IR intensities from the atomic
//! polar tensor.

use crate::constants::{atomic_mass_au, AMU_TO_AU_MASS, HARTREE_TO_CM1};
use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::Molecule;

pub struct VibrationalModes {
    /// Harmonic wavenumbers (cm⁻¹), ascending; imaginary modes negative.
    pub frequencies_cm: Vec<f64>,
    /// IR intensities (km/mol), aligned with `frequencies_cm`.
    pub ir_intensities: Vec<f64>,
    /// Number of projected-out translation/rotation modes.
    pub n_projected: usize,
    /// Imaginary-frequency count (saddle order).
    pub n_imaginary: usize,
    /// Mass-weighted normal-mode vectors (columns, 3N × n_modes).
    pub modes: Matrix,
}

/// Analyze a Cartesian Hessian (Hartree/Bohr²) with the APT (3 × 3N, a.u.).
pub fn vibrational_analysis(
    mol: &Molecule,
    hessian: &Matrix,
    apt: Option<&Matrix>,
) -> Result<VibrationalModes> {
    vibrational_analysis_with_projection(mol, hessian, apt, true)
}

/// Analyze the same mass-weighted Hessian with an explicit rigid-motion model.
/// With `project_rigid_motion=false`, retain every Cartesian mode, including
/// zero and unstable modes. No small eigenvalue is replaced or discarded.
pub fn vibrational_analysis_with_projection(
    mol: &Molecule,
    hessian: &Matrix,
    apt: Option<&Matrix>,
    project_rigid_motion: bool,
) -> Result<VibrationalModes> {
    let natoms = mol.len();
    let dim = 3 * natoms;
    if natoms == 0
        || hessian.rows != dim
        || hessian.cols != dim
        || hessian.as_slice().iter().any(|v| !v.is_finite())
        || apt.is_some_and(|a| {
            a.rows != 3 || a.cols != dim || a.as_slice().iter().any(|v| !v.is_finite())
        })
    {
        return Err(DftError::InvalidInput(
            "vibrational analysis requires a finite 3N by 3N Hessian and optional 3 by 3N atomic polar tensor of a nonempty molecule".into(),
        ));
    }

    // A centre with no mass cannot be mass-weighted. Ghost centres (Z = 0,
    // used for counterpoise) are legitimate in an ENERGY calculation but not
    // here: dividing by √0 silently fills the whole Hessian with NaN and the
    // frequencies come back as NaN with no indication of why.
    for (a, atom) in mol.atoms.iter().enumerate() {
        if atomic_mass_au(atom.z) <= 0.0 {
            return Err(DftError::InvalidInput(format!(
                "vibrational analysis: centre {a} has atomic number {} and therefore no \
                 tabulated mass. Ghost centres (Z = 0) carry basis functions but no nucleus, \
                 so they have no vibrational degrees of freedom; remove them before the \
                 frequency calculation, or give the site an explicit mass.",
                atom.z
            )));
        }
    }

    // Mass-weight: H̃_ij = H_ij / √(m_i m_j) (masses in mₑ).
    let mut sqrt_m = vec![0.0; dim];
    for a in 0..natoms {
        let m = atomic_mass_au(mol.atoms[a].z).sqrt();
        for ax in 0..3 {
            sqrt_m[3 * a + ax] = m;
        }
    }
    let mut hw = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            hw[(i, j)] = hessian[(i, j)] / (sqrt_m[i] * sqrt_m[j]);
        }
    }

    // Eckart vectors: translations √m ê and rotations √m (r − r_com) × ê.
    let mut com = Vec3::zero();
    let mut mtot = 0.0;
    for a in &mol.atoms {
        let m = atomic_mass_au(a.z);
        com += a.position * m;
        mtot += m;
    }
    com = com / mtot;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for ax in 0..3 {
        let mut v = vec![0.0; dim];
        for a in 0..natoms {
            v[3 * a + ax] = sqrt_m[3 * a];
        }
        basis.push(v);
    }
    for ax in 0..3 {
        let e = match ax {
            0 => Vec3::new(1.0, 0.0, 0.0),
            1 => Vec3::new(0.0, 1.0, 0.0),
            _ => Vec3::new(0.0, 0.0, 1.0),
        };
        let mut v = vec![0.0; dim];
        for (a, atom) in mol.atoms.iter().enumerate() {
            let r = atom.position - com;
            let cx = r.cross(e);
            v[3 * a] = sqrt_m[3 * a] * cx.x;
            v[3 * a + 1] = sqrt_m[3 * a] * cx.y;
            v[3 * a + 2] = sqrt_m[3 * a] * cx.z;
        }
        basis.push(v);
    }
    if !project_rigid_motion {
        basis.clear();
    }
    // Gram–Schmidt; drop near-null vectors (linear molecules).
    let mut ortho: Vec<Vec<f64>> = Vec::new();
    for mut v in basis {
        for u in &ortho {
            let d: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            for (vi, ui) in v.iter_mut().zip(u) {
                *vi -= d * ui;
            }
        }
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-8 {
            for vi in v.iter_mut() {
                *vi /= n;
            }
            ortho.push(v);
        }
    }
    let n_projected = ortho.len();
    // P = I − Σ v vᵀ; H_proj = P H̃ P.
    let mut p = Matrix::identity(dim);
    for v in &ortho {
        for i in 0..dim {
            for j in 0..dim {
                p[(i, j)] -= v[i] * v[j];
            }
        }
    }
    let hp = p.matmul(&hw.matmul(&p));

    let (vals, vecs) = symmetric_eigen(&hp)?;
    // Separate the ~zero projected modes; convert the rest.
    let mut frequencies = Vec::new();
    let mut modes_cols = Vec::new();
    for (k, &l) in vals.iter().enumerate() {
        // Projected translations/rotations sit at |λ| ≲ 1e-12.
        let is_projected = {
            // Overlap with the projected space.
            let mut ov = 0.0;
            for v in &ortho {
                let d: f64 = (0..dim).map(|i| v[i] * vecs[(i, k)]).sum();
                ov += d * d;
            }
            ov > 0.5
        };
        if is_projected {
            continue;
        }
        let freq = if l >= 0.0 {
            l.sqrt() * HARTREE_TO_CM1
        } else {
            -(-l).sqrt() * HARTREE_TO_CM1
        };
        frequencies.push(freq);
        modes_cols.push(k);
    }
    let n_imaginary = frequencies.iter().filter(|&&f| f < -1.0).count();

    // IR intensities from the APT: dμ/dQ_i = Σ_k APT[:,k]·L_ki/√m_k with
    // masses in amu → I[km/mol] = 974.8801·Σ_ax (dμ_ax/dQ_i)².
    let mut intensities = vec![0.0; frequencies.len()];
    if let Some(apt) = apt {
        for (i, &col) in modes_cols.iter().enumerate() {
            let mut acc = 0.0;
            for ax in 0..3 {
                let mut d = 0.0;
                for k in 0..dim {
                    // L is mass-weighted; convert to amu-based normal
                    // coordinates: /√(m_k[amu]).
                    let m_amu = (sqrt_m[k] * sqrt_m[k]) / AMU_TO_AU_MASS;
                    d += apt[(ax, k)] * vecs[(k, col)] / m_amu.sqrt();
                }
                acc += d * d;
            }
            intensities[i] = 974.8801 * acc;
        }
    }

    let mut modes = Matrix::zeros(dim, modes_cols.len());
    for (i, &col) in modes_cols.iter().enumerate() {
        for r in 0..dim {
            modes[(r, i)] = vecs[(r, col)];
        }
    }
    Ok(VibrationalModes {
        frequencies_cm: frequencies,
        ir_intensities: intensities,
        n_projected,
        n_imaginary,
        modes,
    })
}
