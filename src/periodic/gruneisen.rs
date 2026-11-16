// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spectral derivatives of a real symmetric mass-weighted dynamical matrix.
//! The coordinate is `x = ln(V / V0)`. Derivative matrices are inputs; this
//! module does not obtain them by finite differences. Degenerate eigenvalues
//! are resolved first by the first derivative and then by the effective
//! second derivative in the remaining degenerate subspace.

use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};

/// One analytic eigenvalue branch through the reference volume.
#[derive(Clone, Debug)]
pub struct ModeVolumeDerivative {
    /// Squared angular frequency. Roundoff-degenerate values share their mean.
    pub squared_frequency: f64,
    pub first_squared_frequency: f64,
    pub second_squared_frequency: f64,
    /// `-d ln(omega) / dx`; undefined for zero or imaginary frequencies.
    pub first_gruneisen: Option<f64>,
    /// `d2 ln(omega) / dx2 = -d gamma / dx`, not `d ln(gamma)/dx`.
    pub second_log_frequency: Option<f64>,
}

/// Volume derivatives with orthonormal branch vectors as columns.
#[derive(Clone, Debug)]
pub struct ModeVolumeDerivatives {
    pub modes: Matrix,
    pub branches: Vec<ModeVolumeDerivative>,
    /// Original reference eigenvalues before roundoff clustering.
    pub reference_eigenvalues: Vec<f64>,
    /// Maximum absolute input antisymmetry for D, D', and D''.
    pub symmetry_defects: [f64; 3],
}

/// Convert material-coordinate force-constant derivatives along the isotropic
/// Lagrangian strain `eta = s I` to physical, mass-weighted log-volume jets.
///
/// `F = sqrt(1 + 2s) I`, `x = ln(V/V0) = 3 ln(1 + 2s)/2`, and physical
/// displacements are `F u`. Consequently `Phi_phys = Phi(s)/(1 + 2s)`.
/// Inputs `first_strain` and `second_strain` are total derivatives along the
/// chosen clamped- or relaxed-ion branch; the caller must include internal
/// first and second relaxation in them. Each inverse mass root is in the
/// same mass units, with three displacement coordinates per atom.
pub fn isotropic_log_volume_jets(
    force_constants: &Matrix,
    first_strain: &Matrix,
    second_strain: &Matrix,
    inverse_mass_roots: &[f64],
) -> Result<[Matrix; 3]> {
    let n = force_constants.rows;
    if n == 0
        || n != 3 * inverse_mass_roots.len()
        || inverse_mass_roots
            .iter()
            .any(|m| !m.is_finite() || *m <= 0.0)
        || [force_constants, first_strain, second_strain]
            .iter()
            .any(|m| m.rows != n || m.cols != n || m.as_slice().iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "isotropic volume jets require finite matching 3N force constants and positive inverse mass roots".into(),
        ));
    }
    let mut jets = std::array::from_fn(|_| Matrix::zeros(n, n));
    for i in 0..n {
        for j in 0..n {
            let mass = inverse_mass_roots[i / 3] * inverse_mass_roots[j / 3];
            let phi = force_constants[(i, j)];
            let first = first_strain[(i, j)];
            jets[0][(i, j)] = mass * phi;
            jets[1][(i, j)] = mass * (first - 2.0 * phi) / 3.0;
            jets[2][(i, j)] = mass * (second_strain[(i, j)] - 2.0 * first + 4.0 * phi) / 9.0;
        }
    }
    Ok(jets)
}

fn scale(a: &Matrix) -> f64 {
    a.as_slice().iter().fold(0.0_f64, |s, x| s.max(x.abs()))
}

fn clusters(values: &[f64], tolerance: f64) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut begin = 0;
    while begin < values.len() {
        let mut end = begin + 1;
        while end < values.len() && values[end] - values[begin] <= tolerance {
            end += 1;
        }
        groups.push(begin..end);
        begin = end;
    }
    groups
}

fn project(a: &Matrix, vectors: &Matrix) -> Matrix {
    vectors.transpose().matmul(a).matmul(vectors)
}

/// Differentiate all branches through second order in log volume.
///
/// Only antisymmetry and degeneracy at floating-point roundoff are merged.
/// This does not regularize small optical frequencies or project an acoustic
/// sum rule. A crystal caller must separately identify uniform translations
/// and report their Gruneisen coefficients as undefined. No division is made
/// by an intra-degenerate eigenvalue gap.
pub fn mode_volume_derivatives(
    dynamical: &Matrix,
    first: &Matrix,
    second: &Matrix,
) -> Result<ModeVolumeDerivatives> {
    let n = dynamical.rows;
    if n == 0
        || [dynamical, first, second]
            .iter()
            .any(|a| a.rows != n || a.cols != n || a.as_slice().iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "mode volume derivatives require finite square matrices of equal nonzero size".into(),
        ));
    }
    let roundoff = 64.0 * f64::EPSILON * n as f64;
    let mut defects = [0.0_f64; 3];
    let mut symmetric = Vec::new();
    for (k, a) in [dynamical, first, second].into_iter().enumerate() {
        let mut b = a.clone();
        for i in 0..n {
            for j in 0..n {
                defects[k] = defects[k].max((a[(i, j)] - a[(j, i)]).abs());
                b[(i, j)] = 0.5 * a[(i, j)] + 0.5 * a[(j, i)];
            }
        }
        if defects[k] > roundoff * scale(a) {
            return Err(DftError::InvalidInput(format!(
                "dynamical matrix derivative {k} is not symmetric: defect {}",
                defects[k]
            )));
        }
        symmetric.push(b);
    }
    let (values, vectors) = symmetric_eigen(&symmetric[0])?;
    let b = project(&symmetric[1], &vectors);
    let c = project(&symmetric[2], &vectors);
    let mut modes = Matrix::zeros(n, n);
    let mut branches = Vec::with_capacity(n);
    for group in clusters(&values, roundoff * scale(&symmetric[0])) {
        let m = group.len();
        let lambda = values[group.clone()].iter().sum::<f64>() / m as f64;
        let mut b_sub = Matrix::zeros(m, m);
        let mut effective = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                let (a, d) = (group.start + i, group.start + j);
                b_sub[(i, j)] = b[(a, d)];
                effective[(i, j)] = c[(a, d)];
                for k in 0..n {
                    if !group.contains(&k) {
                        effective[(i, j)] += 2.0 * b[(a, k)] * b[(k, d)] / (lambda - values[k]);
                    }
                }
            }
        }
        let (slopes, rotation) = symmetric_eigen(&b_sub)?;
        let effective = project(&effective, &rotation);
        let mut final_rotation = rotation.clone();
        let mut curvature = vec![0.0; m];
        for slopes_group in clusters(&slopes, roundoff * scale(&symmetric[1])) {
            let r = slopes_group.len();
            let mut block = Matrix::zeros(r, r);
            for i in 0..r {
                for j in 0..r {
                    block[(i, j)] = effective[(slopes_group.start + i, slopes_group.start + j)];
                }
            }
            let (seconds, turn) = symmetric_eigen(&block)?;
            for i in 0..m {
                for j in 0..r {
                    final_rotation[(i, slopes_group.start + j)] = (0..r)
                        .map(|k| rotation[(i, slopes_group.start + k)] * turn[(k, j)])
                        .sum();
                }
            }
            for j in 0..r {
                curvature[slopes_group.start + j] = seconds[j];
            }
        }
        for j in 0..m {
            let column = branches.len();
            for i in 0..n {
                modes[(i, column)] = (0..m)
                    .map(|k| vectors[(i, group.start + k)] * final_rotation[(k, j)])
                    .sum();
            }
            let positive = lambda > 0.0;
            branches.push(ModeVolumeDerivative {
                squared_frequency: lambda,
                first_squared_frequency: slopes[j],
                second_squared_frequency: curvature[j],
                first_gruneisen: positive.then(|| -slopes[j] / (2.0 * lambda)),
                second_log_frequency: positive
                    .then(|| 0.5 * (curvature[j] / lambda - (slopes[j] / lambda).powi(2))),
            });
        }
    }
    Ok(ModeVolumeDerivatives {
        modes,
        branches,
        reference_eigenvalues: values,
        symmetry_defects: defects,
    })
}
