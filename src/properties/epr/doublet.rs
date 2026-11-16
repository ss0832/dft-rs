// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit single-determinant Kramers-pair projection. This is an optional
//! model, not spin purification, an orientational generator-coordinate
//! calculation or proof of an isolated many-electron doublet. In particular,
//! spin contamination can distort transverse g values even without SOC.

use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::{DftError, Result};

fn z(matrix: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(matrix.re[(i, j)], matrix.im[(i, j)])
}
fn put(matrix: &mut BlochMatrix, i: usize, j: usize, value: Complex) {
    matrix.re[(i, j)] = value.re;
    matrix.im[(i, j)] = value.im;
}

/// Raw entry defect of Theta O Theta^-1 = parity*O in a real spatial AO
/// basis. parity=+1 tests a zero-field SOC Hamiltonian; -1 tests moments.
pub fn time_reversal_defect(operator: &BlochMatrix, parity: i32) -> Result<f64> {
    let dim = operator.rows();
    let n = dim / 2;
    if dim == 0
        || dim != 2 * n
        || operator.cols() != dim
        || operator.im.rows != dim
        || operator.im.cols != dim
        || !matches!(parity, -1 | 1)
        || operator
            .re
            .as_slice()
            .iter()
            .chain(operator.im.as_slice())
            .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "time reversal needs finite even square spin/AO matrices and parity+1 or-1".into(),
        ));
    }
    let mut defect = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            let sign_i = if i < n { -1.0 } else { 1.0 };
            let sign_j = if j < n { -1.0 } else { 1.0 };
            let transformed = z(operator, (i + n) % dim, (j + n) % dim).conj() * (sign_i * sign_j);
            defect = defect.max((transformed - z(operator, i, j) * parity as f64).abs());
        }
    }
    Ok(defect)
}

/// Occupied spinors and their time-reversed determinant. Spin-major AO
/// ordering is [up AOs, down AOs], Theta(a,b)=(-conj(b),conj(a)).
pub struct KramersPair {
    density: [BlochMatrix; 2],
    transition: BlochMatrix,
    pub occupied_metric_defect: f64,
    pub determinant_overlap: Complex,
}

fn determinant(mut a: Vec<Vec<Complex>>) -> Complex {
    let n = a.len();
    let mut value = Complex::ONE;
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .expect("nonempty pivot column");
        if a[pivot][k].abs() == 0.0 {
            return Complex::ZERO;
        }
        if pivot != k {
            a.swap(pivot, k);
            value = -value;
        }
        let diagonal = a[k][k];
        value = value * diagonal;
        for i in k + 1..n {
            let coefficient = a[i][k] / diagonal;
            for j in k + 1..n {
                let row = a[k][j];
                a[i][j] -= coefficient * row;
            }
        }
    }
    value
}

impl KramersPair {
    /// Build exact determinant transition one-body densities by cofactors,
    /// retaining singular occupied overlaps. An inverse times det(M) would
    /// lose every Kramers transition because det(M)=0 for odd electron count.
    pub fn new(occupied: &BlochMatrix, overlap: &Matrix, tolerance: f64) -> Result<Self> {
        let n = overlap.rows;
        let electrons = occupied.cols();
        let dim = 2 * n;
        if n == 0
            || overlap.cols != n
            || occupied.rows() != dim
            || occupied.im.rows != dim
            || occupied.im.cols != electrons
            || electrons == 0
            || electrons % 2 == 0
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || overlap.as_slice().iter().any(|v| !v.is_finite())
            || occupied
                .re
                .as_slice()
                .iter()
                .chain(occupied.im.as_slice())
                .any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput("single Kramers-pair projection requires finite matching real AO metric and a positive odd number of occupied spinors; it is not an arbitrary-spin multiplet construction".into()));
        }
        let _workspace = crate::resources::reserve(
            crate::resources::checked_bytes(
                &[4, electrons, electrons, std::mem::size_of::<Complex>()],
                "Kramers cofactor workspace",
            )?,
            "Kramers cofactor workspace",
        )?;
        let mut partner = BlochMatrix::zeros(dim, electrons);
        for i in 0..n {
            for j in 0..electrons {
                put(&mut partner, i, j, -z(occupied, n + i, j).conj());
                put(&mut partner, n + i, j, z(occupied, i, j).conj());
            }
        }
        let metric_element = |left: &BlochMatrix, right: &BlochMatrix, i: usize, j: usize| {
            let mut sum = Complex::ZERO;
            for spin in 0..2 {
                for a in 0..n {
                    for b in 0..n {
                        sum += z(left, spin * n + a, i).conj()
                            * z(right, spin * n + b, j)
                            * overlap[(a, b)];
                    }
                }
            }
            sum
        };
        let mut defect = 0.0_f64;
        for a in [occupied, &partner] {
            for i in 0..electrons {
                for j in 0..electrons {
                    defect = defect.max(
                        (metric_element(a, a, i, j)
                            - Complex::from(if i == j { 1.0 } else { 0.0 }))
                        .abs(),
                    );
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                defect = defect.max((overlap[(i, j)] - overlap[(j, i)]).abs());
            }
        }
        if defect > tolerance {
            return Err(DftError::InvalidInput(format!(
                "occupied Kramers metric defect {defect} exceeds {tolerance}"
            )));
        }
        let cross = (0..electrons)
            .map(|i| {
                (0..electrons)
                    .map(|j| metric_element(occupied, &partner, i, j))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let determinant_overlap = determinant(cross.clone());
        if !determinant_overlap.is_finite() || determinant_overlap.abs() > tolerance {
            return Err(DftError::InvalidInput("time-reversed odd-electron determinants are not orthogonal within the supplied tolerance".into()));
        }
        let mut cofactors = vec![vec![Complex::ZERO; electrons]; electrons];
        for i in 0..electrons {
            for j in 0..electrons {
                let minor = (0..electrons)
                    .filter(|&row| row != i)
                    .map(|row| {
                        (0..electrons)
                            .filter(|&column| column != j)
                            .map(|column| cross[row][column])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                cofactors[i][j] = determinant(minor) * if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let density = std::array::from_fn(|state| {
            let columns = if state == 0 { occupied } else { &partner };
            let mut matrix = BlochMatrix::zeros(dim, dim);
            for a in 0..dim {
                for b in 0..dim {
                    put(
                        &mut matrix,
                        a,
                        b,
                        (0..electrons).fold(Complex::ZERO, |value, i| {
                            value + z(columns, a, i) * z(columns, b, i).conj()
                        }),
                    );
                }
            }
            matrix
        });
        let mut transition = BlochMatrix::zeros(dim, dim);
        for a in 0..dim {
            for b in 0..dim {
                let mut value = Complex::ZERO;
                for i in 0..electrons {
                    for j in 0..electrons {
                        value += cofactors[i][j] * z(&partner, a, j) * z(occupied, b, i).conj();
                    }
                }
                if !value.is_finite() {
                    return Err(DftError::InvalidInput(
                        "nonfinite Kramers transition cofactor".into(),
                    ));
                }
                put(&mut transition, a, b, value);
            }
        }
        // Cofactors preserve exact-zero overlap sectors; no singular-value
        // clipping, diagonal inverse or occupied-space pseudoinverse is used.
        Ok(Self {
            density,
            transition,
            occupied_metric_defect: defect,
            determinant_overlap,
        })
    }

    /// Consume the actual integer molecular spinor reference without
    /// replacing it by collinear orbitals or reconstructing SCF metadata.
    pub fn from_reference(
        reference: &crate::periodic::spinor_scf::SpinorScfReport,
        overlap: &Matrix,
        tolerance: f64,
    ) -> Result<Self> {
        if !reference.converged
            || reference.bands.len() != 1
            || reference.density.len() != 1
            || reference.occupations.len() != 1
            || reference.chemical_potential.is_some()
            || reference.entropy_term != 0.0
        {
            return Err(DftError::InvalidInput("Kramers projection needs an actual converged integer molecular spinor reference, not finite-temperature or multi-k data".into()));
        }
        let occupations = &reference.occupations[0];
        if occupations.iter().any(|&f| f != 0.0 && f != 1.0) {
            return Err(DftError::InvalidInput(
                "Kramers determinant projection does not accept fractional spinor occupations"
                    .into(),
            ));
        }
        let columns = &reference.bands[0].coefficients;
        let indices = occupations
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f == 1.0 { Some(i) } else { None })
            .collect::<Vec<_>>();
        if occupations.len() != columns.cols() || reference.density[0].rows() != columns.rows() {
            return Err(DftError::InvalidInput(
                "spinor reference band/density dimensions disagree".into(),
            ));
        }
        let mut occupied = BlochMatrix::zeros(columns.rows(), indices.len());
        for i in 0..columns.rows() {
            for (j, &column) in indices.iter().enumerate() {
                put(&mut occupied, i, j, z(columns, i, column));
            }
        }
        let pair = Self::new(&occupied, overlap, tolerance)?;
        let difference = pair.density[0]
            .re
            .max_abs_diff(&reference.density[0].re)
            .max(pair.density[0].im.max_abs_diff(&reference.density[0].im));
        if !difference.is_finite() || difference > tolerance {
            return Err(DftError::InvalidInput(format!("Kramers occupied determinant does not reproduce actual spinor density: {difference}")));
        }
        Ok(pair)
    }

    /// Every Cartesian row is projected independently. scale=4 for
    /// dH/dB -> dimensionless g (mu_B=1/2); scale=2 for an operator
    /// already expressed per nuclear spin I in MHz -> A in MHz.
    pub fn tensor(
        &self,
        operators: &[BlochMatrix; 3],
        scale: f64,
        tolerance: f64,
    ) -> Result<[[f64; 3]; 3]> {
        if !scale.is_finite() {
            return Err(DftError::InvalidInput(
                "nonfinite effective-spin tensor scale".into(),
            ));
        }
        let mut result = [[0.0; 3]; 3];
        for a in 0..3 {
            result[a] = self
                .pauli_coefficients(&operators[a], tolerance)?
                .map(|v| v * scale);
        }
        Ok(result)
    }

    /// Project a finite Hermitian, time-odd one-electron operator. The
    /// returned coefficients c satisfy O_pair = c . sigma, in the actual
    /// ordered pair (Phi,Theta Phi). No rotation to force a symmetric tensor.
    pub fn pauli_coefficients(&self, operator: &BlochMatrix, tolerance: f64) -> Result<[f64; 3]> {
        let n = self.transition.rows();
        if operator.rows() != n
            || operator.cols() != n
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || operator.im.rows != n
            || operator.im.cols != n
            || operator.hermiticity_error() > tolerance
            || operator
                .re
                .as_slice()
                .iter()
                .chain(operator.im.as_slice())
                .any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput("Kramers magnetic operator must be finite, Hermitian and match the spin/AO dimension".into()));
        }
        let time_reversal_defect = time_reversal_defect(operator, -1)?;
        if time_reversal_defect > tolerance {
            return Err(DftError::InvalidInput(format!(
                "operator time-odd residual {time_reversal_defect} exceeds {tolerance}"
            )));
        }
        let contract = |density: &BlochMatrix| {
            (0..n).fold(Complex::ZERO, |value, i| {
                value
                    + (0..n).fold(Complex::ZERO, |row, j| {
                        row + z(density, i, j) * z(operator, j, i)
                    })
            })
        };
        let a = contract(&self.density[0]);
        let b = contract(&self.density[1]);
        let transition = contract(&self.transition);
        if a.im.abs() > tolerance || b.im.abs() > tolerance || (a + b).abs() > tolerance {
            return Err(DftError::InvalidInput("projected magnetic operator is not a traceless Hermitian Kramers observable within the supplied tolerance".into()));
        }
        Ok([transition.re, -transition.im, 0.5 * (a.re - b.re)])
    }
}
