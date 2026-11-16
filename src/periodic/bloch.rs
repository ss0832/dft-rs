// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Small, explicit complex-matrix layer for k-point calculations.
//!
//! The molecular code intentionally stores only real matrices.  Periodic
//! Bloch matrices are complex even when every AO integral in real space is
//! real, so representing them as a real/imaginary pair keeps that distinction
//! visible at the public API boundary and avoids introducing complex numbers
//! into the molecular paths.

use crate::linalg::Matrix;

/// A complex matrix represented by separate row-major real and imaginary
/// parts.  Hermitian k-point operators have symmetric `re` and
/// antisymmetric `im` parts.
#[derive(Clone, Debug, PartialEq)]
pub struct BlochMatrix {
    pub re: Matrix,
    pub im: Matrix,
}

impl BlochMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            re: Matrix::zeros(rows, cols),
            im: Matrix::zeros(rows, cols),
        }
    }

    pub fn from_pair(pair: (Matrix, Matrix)) -> Self {
        Self {
            re: pair.0,
            im: pair.1,
        }
    }

    pub fn into_pair(self) -> (Matrix, Matrix) {
        (self.re, self.im)
    }

    pub fn rows(&self) -> usize {
        self.re.rows
    }

    pub fn cols(&self) -> usize {
        self.re.cols
    }

    /// Add `scale * other` in place.
    pub fn add_scaled(&mut self, other: &Self, scale: f64) {
        debug_assert_eq!(self.re.rows, other.re.rows);
        debug_assert_eq!(self.re.cols, other.re.cols);
        for (dst, src) in self.re.as_mut_slice().iter_mut().zip(other.re.as_slice()) {
            *dst += scale * src;
        }
        for (dst, src) in self.im.as_mut_slice().iter_mut().zip(other.im.as_slice()) {
            *dst += scale * src;
        }
    }

    pub fn scale(&mut self, scale: f64) {
        for value in self
            .re
            .as_mut_slice()
            .iter_mut()
            .chain(self.im.as_mut_slice())
        {
            *value *= scale;
        }
    }

    /// Real part of `Tr(self * other)` for Hermitian matrices.
    ///
    /// The imaginary contribution has a plus sign because the second
    /// matrix is transposed inside the trace and its antisymmetric
    /// imaginary part therefore changes sign.
    pub fn trace_product(&self, other: &Self) -> f64 {
        self.re.frobenius_dot(&other.re) + self.im.frobenius_dot(&other.im)
    }

    /// Real `2n x 2n` representation `[[Re,-Im],[Im,Re]]`.
    pub fn real_embedding(&self) -> Matrix {
        crate::periodic::integrals::hermitian_as_real(&self.re, &self.im)
    }

    /// Recover a complex matrix from its doubled real representation.
    pub(crate) fn from_real_embedding(matrix: &Matrix) -> Self {
        debug_assert_eq!(matrix.rows, matrix.cols);
        debug_assert_eq!(matrix.rows % 2, 0);
        let n = matrix.rows / 2;
        let mut out = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                out.re[(i, j)] = matrix[(i, j)];
                out.im[(i, j)] = matrix[(n + i, j)];
            }
        }
        out
    }

    /// Project roundoff onto the Hermitian subspace.
    pub fn make_hermitian(&mut self) {
        let n = self.rows();
        debug_assert_eq!(n, self.cols());
        for i in 0..n {
            self.im[(i, i)] = 0.0;
            for j in 0..i {
                let re = 0.5 * (self.re[(i, j)] + self.re[(j, i)]);
                let im = 0.5 * (self.im[(i, j)] - self.im[(j, i)]);
                self.re[(i, j)] = re;
                self.re[(j, i)] = re;
                self.im[(i, j)] = im;
                self.im[(j, i)] = -im;
            }
        }
    }

    pub fn hermiticity_error(&self) -> f64 {
        let mut error = 0.0_f64;
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                error = error
                    .max((self.re[(i, j)] - self.re[(j, i)]).abs())
                    .max((self.im[(i, j)] + self.im[(j, i)]).abs());
            }
        }
        error
    }

    pub fn rms_diff(&self, other: &Self) -> f64 {
        let count = 2 * self.re.as_slice().len();
        if count == 0 {
            return 0.0;
        }
        let sum: f64 = self
            .re
            .as_slice()
            .iter()
            .zip(other.re.as_slice())
            .chain(self.im.as_slice().iter().zip(other.im.as_slice()))
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        (sum / count as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_product_matches_direct_complex_trace() {
        let p = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![0.8, 0.2, 0.2, 0.4]),
            im: Matrix::from_row_major(2, 2, vec![0.0, -0.3, 0.3, 0.0]),
        };
        let f = BlochMatrix {
            re: Matrix::from_row_major(2, 2, vec![-1.0, 0.6, 0.6, 0.1]),
            im: Matrix::from_row_major(2, 2, vec![0.0, 0.5, -0.5, 0.0]),
        };
        // Direct real part of sum_ij P_ij F_ji.
        let mut direct = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                direct += p.re[(i, j)] * f.re[(j, i)] - p.im[(i, j)] * f.im[(j, i)];
            }
        }
        assert!((p.trace_product(&f) - direct).abs() < 1e-14);
    }
}
