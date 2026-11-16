// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Real, integer-occupation Fermi–Löwdin orbitals on a fixed spin subspace.
//!
//! M. R. Pederson, A. Ruzsinszky and J. P. Perdew, J. Chem. Phys. 140,
//! 121103 (2014), DOI: 10.1063/1.4869581, Eq. (4) and symmetric
//! orthogonalization of the resulting normalized Fermi orbitals.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};

/// Localized AO coefficients and raw diagnostics for one spin channel.
pub struct FloOrbitals {
    /// AO by occupied-orbital coefficients in descriptor order.
    pub coefficients: Matrix,
    /// Spin density at each descriptor, in Bohr⁻³.
    pub descriptor_densities: Vec<f64>,
    /// Least eigenvalue of the normalized Fermi-orbital metric.
    pub min_fermi_overlap_eigenvalue: Option<f64>,
    /// Maximum absolute defect of the input occupied metric from identity.
    pub input_orthogonality_residual: f64,
    /// Maximum absolute defect of the localized AO metric from identity.
    pub orthogonality_residual: f64,
}

/// F = U H in the occupied frame. The eigenvectors and roots of the
/// positive polar factor H also solve its derivative without differentiating
/// eigenvectors, including at repeated eigenvalues.
pub(super) struct FloFrame {
    pub orbitals: FloOrbitals,
    fermi: Matrix,
    orthogonal: Matrix,
    metric_vectors: Matrix,
    metric_roots: Vec<f64>,
}

impl FloFrame {
    /// Pull back through Fermi normalization and the polar factor to the
    /// point values of the occupied orbitals.
    pub fn value_gradient(&self, occupied: &Matrix, coefficient_gradient: &Matrix) -> Matrix {
        let count = occupied.cols;
        if count == 0 {
            return Matrix::zeros(0, 0);
        }
        // If dU = U Omega and F = U H, then
        // H Omega + Omega H = U^T dF - dF^T U. The adjoint solves the same
        // Sylvester operator using the skew part of U^T dE/dU.
        let in_frame = occupied.transpose_matmul_seq(coefficient_gradient);
        let polar_gradient = self.orthogonal.transpose_matmul_seq(&in_frame);
        let vectors = &self.metric_vectors;
        let eigen_gradient = vectors.transpose_matmul_seq(&polar_gradient.matmul_seq(vectors));
        let mut sylvester = Matrix::zeros(count, count);
        for i in 0..count {
            for j in 0..i {
                let value = 0.5 * (eigen_gradient[(i, j)] - eigen_gradient[(j, i)])
                    / (self.metric_roots[i] + self.metric_roots[j]);
                sylvester[(i, j)] = value;
                sylvester[(j, i)] = -value;
            }
        }
        let adjoint = vectors.matmul_seq(&sylvester).matmul_transpose_seq(vectors);
        let gradient = self.orthogonal.matmul_seq(&adjoint);
        let mut result = Matrix::zeros(count, count);
        for descriptor in 0..count {
            let radial: f64 = (0..count)
                .map(|i| self.fermi[(i, descriptor)] * gradient[(i, descriptor)])
                .sum();
            let scale = 2.0 / self.orbitals.descriptor_densities[descriptor].sqrt();
            for i in 0..count {
                result[(descriptor, i)] =
                    scale * (gradient[(i, descriptor)] - radial * self.fermi[(i, descriptor)]);
            }
        }
        result
    }

    pub fn descriptor_gradient(
        &self,
        occupied: &Matrix,
        ao_gradient: &[Matrix; 3],
        value_gradient: &Matrix,
    ) -> Result<Vec<[f64; 3]>> {
        let occupied_derivative = ao_gradient.each_ref().map(|a| a.matmul_seq(occupied));
        let result: Vec<_> = (0..occupied.cols)
            .map(|descriptor| {
                occupied_derivative.each_ref().map(|derivative| {
                    (0..occupied.cols)
                        .map(|i| value_gradient[(descriptor, i)] * derivative[(descriptor, i)])
                        .sum::<f64>()
                })
            })
            .collect();
        if result.iter().flatten().any(|x| !x.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "Fermi-Lowdin descriptor derivative is non-finite".into(),
            ));
        }
        Ok(result)
    }

    /// The occupied coefficients enter both L=C U and the orbital values
    /// at the fixed descriptors. Both contributions enter the SIC force.
    pub fn occupied_gradient(
        &self,
        descriptor_aos: &Matrix,
        coefficient_gradient: &Matrix,
        value_gradient: &Matrix,
    ) -> Matrix {
        let mut gradient = coefficient_gradient.matmul_transpose_seq(&self.orthogonal);
        let point_gradient = descriptor_aos.transpose_matmul_seq(value_gradient);
        for (a, b) in gradient
            .as_mut_slice()
            .iter_mut()
            .zip(point_gradient.as_slice())
        {
            *a += *b;
        }
        gradient
    }
}

/// Construct Fermi–Löwdin orbitals for an AO overlap metric and point values.
///
/// The columns of `occupied` are the occupied spin orbitals, orthonormal in
/// `overlap`. The rows of `descriptor_aos` are AO values at the descriptors.
/// A zero-density point or a rank-deficient set is rejected because a full
/// orthonormal basis of the occupied space requires one independent orbital
/// per descriptor.
pub fn fermi_lowdin_orbitals(
    occupied: &Matrix,
    overlap: &Matrix,
    descriptor_aos: &Matrix,
) -> Result<FloOrbitals> {
    Ok(fermi_frame(occupied, overlap, descriptor_aos)?.orbitals)
}

pub(super) fn fermi_frame(
    occupied: &Matrix,
    overlap: &Matrix,
    descriptor_aos: &Matrix,
) -> Result<FloFrame> {
    let (nao, count) = (occupied.rows, occupied.cols);
    if count > nao
        || overlap.rows != nao
        || overlap.cols != nao
        || descriptor_aos.rows != count
        || descriptor_aos.cols != nao
        || [occupied, overlap, descriptor_aos]
            .iter()
            .flat_map(|matrix| matrix.as_slice())
            .any(|value| !value.is_finite())
    {
        return Err(DftError::InvalidInput(
            "Fermi–Löwdin orbitals require finite AO and occupied matrices with one descriptor per occupied orbital".into(),
        ));
    }
    if count == 0 {
        return Ok(FloFrame {
            orbitals: FloOrbitals {
                coefficients: Matrix::zeros(nao, 0),
                descriptor_densities: Vec::new(),
                min_fermi_overlap_eigenvalue: None,
                input_orthogonality_residual: 0.0,
                orthogonality_residual: 0.0,
            },
            fermi: Matrix::zeros(0, 0),
            orthogonal: Matrix::zeros(0, 0),
            metric_vectors: Matrix::zeros(0, 0),
            metric_roots: Vec::new(),
        });
    }
    let identity = Matrix::identity(count);
    let occupied_metric = occupied.transpose_matmul_seq(&overlap.matmul_seq(occupied));
    let input_orthogonality_residual = occupied_metric.max_abs_diff(&identity);
    if !occupied_metric.as_slice().iter().all(|v| v.is_finite())
        || input_orthogonality_residual > 1e-8
    {
        return Err(DftError::InvalidInput(
            "Fermi–Löwdin orbitals require an orthonormal occupied subspace in the AO metric"
                .into(),
        ));
    }

    // In the occupied frame, Fermi orbital i has coefficients
    // ψ_j(a_i) / sqrt(sum_j ψ_j(a_i)^2). Take a stable norm before squaring.
    let amplitudes = descriptor_aos.matmul_seq(occupied);
    let mut fermi = Matrix::zeros(count, count);
    let mut descriptor_densities = Vec::with_capacity(count);
    for descriptor in 0..count {
        let norm = (0..count)
            .map(|j| amplitudes[(descriptor, j)])
            .fold(0.0_f64, f64::hypot);
        let density = norm * norm;
        if density <= 0.0 || !density.is_finite() {
            return Err(DftError::InvalidInput(format!(
                "Fermi orbital descriptor {descriptor} has zero or unrepresentable spin density"
            )));
        }
        descriptor_densities.push(density);
        for orbital in 0..count {
            fermi[(orbital, descriptor)] = amplitudes[(descriptor, orbital)] / norm;
        }
    }

    // Symmetric Löwdin orthogonalization uses the square inverse root of the
    // Fermi-orbital metric. Keep every descriptor and reject unresolved rank.
    let metric = fermi.transpose_matmul_seq(&fermi);
    let (values, vectors) = symmetric_eigen(&metric)?;
    let rank_threshold = 128.0 * f64::EPSILON * count as f64 * values[count - 1];
    if values
        .iter()
        .any(|&lambda| !lambda.is_finite() || lambda <= rank_threshold)
    {
        return Err(DftError::LinearAlgebra(
            "Fermi orbitals do not span the occupied subspace at the supplied descriptors".into(),
        ));
    }
    let metric_roots: Vec<_> = values.iter().map(|value| value.sqrt()).collect();
    let mut scaled = vectors.clone();
    for row in 0..count {
        for column in 0..count {
            scaled[(row, column)] *= metric_roots[column].recip();
        }
    }
    let lowdin = scaled.matmul_transpose_seq(&vectors);
    let orthogonal = fermi.matmul_seq(&lowdin);
    let coefficients = occupied.matmul_seq(&orthogonal);
    let localization_metric = coefficients.transpose_matmul_seq(&overlap.matmul_seq(&coefficients));
    let orthogonality_residual = localization_metric.max_abs_diff(&identity);
    if !localization_metric.as_slice().iter().all(|v| v.is_finite())
        || orthogonality_residual > 1e-8
    {
        return Err(DftError::LinearAlgebra(
            "Fermi–Löwdin orthogonalization exceeds the occupied AO metric tolerance".into(),
        ));
    }
    Ok(FloFrame {
        orbitals: FloOrbitals {
            coefficients,
            descriptor_densities,
            min_fermi_overlap_eigenvalue: Some(values[0]),
            input_orthogonality_residual,
            orthogonality_residual,
        },
        fermi,
        orthogonal,
        metric_vectors: vectors,
        metric_roots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_nonorthogonal_fermi_orbitals_match_closed_form_and_rotate_invariantly() {
        let overlap =
            Matrix::from_row_major(3, 3, vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0]);
        let occupied = Matrix::from_row_major(3, 2, vec![0.5, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0]);
        let aos = Matrix::from_row_major(2, 3, vec![4.0, 0.0, -1.5, 6.0, 12.0, 0.2]);
        let result = fermi_lowdin_orbitals(&occupied, &overlap, &aos).unwrap();
        // In the occupied frame, F=[[1,0.6],[0,0.8]]. Its orthogonal polar
        // factor is [[cos,sin],[-sin,cos]], with sin/cos=0.6/1.8.
        let co = 3.0 / 10.0_f64.sqrt();
        let si = 1.0 / 10.0_f64.sqrt();
        let expected = Matrix::from_row_major(
            3,
            2,
            vec![0.5 * co, 0.5 * si, -si / 3.0, co / 3.0, 0.0, 0.0],
        );
        let rotation = Matrix::from_row_major(2, 2, vec![0.6, -0.8, 0.8, 0.6]);
        let rotated =
            fermi_lowdin_orbitals(&occupied.matmul_seq(&rotation), &overlap, &aos).unwrap();
        let oracle_error = result.coefficients.max_abs_diff(&expected);
        let rotation_error = result.coefficients.max_abs_diff(&rotated.coefficients);
        let density_error = result
            .coefficients
            .leading_columns_gram(2, 1.0)
            .max_abs_diff(&occupied.leading_columns_gram(2, 1.0));
        println!(
            "FLO polar oracle={oracle_error:.3e} rotation={rotation_error:.3e} density={density_error:.3e} metric={:.3e} min_eig={:.16e}",
            result.orthogonality_residual,
            result.min_fermi_overlap_eigenvalue.unwrap()
        );
        assert!(oracle_error < 3e-15 && rotation_error < 3e-15 && density_error < 3e-15);
        assert!(result.input_orthogonality_residual < 3e-15);
        assert!(result.orthogonality_residual < 3e-15);
        assert_eq!(result.descriptor_densities, [4.0, 25.0]);
        assert!((result.min_fermi_overlap_eigenvalue.unwrap() - 0.4).abs() < 2e-15);
    }

    #[test]
    fn descriptor_permutation_only_permutes_the_orbitals() {
        let occupied = Matrix::identity(2);
        let overlap = Matrix::identity(2);
        let aos = Matrix::from_row_major(2, 2, vec![1.0, 0.3, -0.4, 1.4]);
        let reverse = Matrix::from_row_major(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let forward = fermi_lowdin_orbitals(&occupied, &overlap, &aos).unwrap();
        let backward =
            fermi_lowdin_orbitals(&occupied, &overlap, &reverse.matmul_seq(&aos)).unwrap();
        assert!(
            backward
                .coefficients
                .max_abs_diff(&forward.coefficients.matmul_seq(&reverse))
                < 2e-15
        );
    }

    #[test]
    fn one_electron_and_empty_spin_channel_have_the_correct_limits() {
        let overlap = Matrix::identity(2);
        let orbital = Matrix::from_row_major(2, 1, vec![0.6, 0.8]);
        let aos = Matrix::from_row_major(1, 2, vec![-2.0, -1.0]);
        let result = fermi_lowdin_orbitals(&orbital, &overlap, &aos).unwrap();
        assert!((result.descriptor_densities[0] - 4.0).abs() < 2e-15);
        assert!(result
            .coefficients
            .as_slice()
            .iter()
            .zip(orbital.as_slice())
            .all(|(a, b)| (a + b).abs() < 2e-15));
        let empty =
            fermi_lowdin_orbitals(&Matrix::zeros(2, 0), &overlap, &Matrix::zeros(0, 2)).unwrap();
        assert_eq!(empty.coefficients.rows, 2);
        assert_eq!(empty.coefficients.cols, 0);
        assert_eq!(empty.min_fermi_overlap_eigenvalue, None);
    }

    #[test]
    fn rejects_zero_density_dependent_descriptors_and_nonorthogonal_inputs() {
        let overlap = Matrix::identity(2);
        let occupied = Matrix::identity(2);
        let zero_density = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        let dependent = Matrix::from_row_major(2, 2, vec![1.0, 0.5, 2.0, 1.0]);
        assert!(matches!(
            fermi_lowdin_orbitals(&occupied, &overlap, &zero_density),
            Err(DftError::InvalidInput(_))
        ));
        assert!(matches!(
            fermi_lowdin_orbitals(&occupied, &overlap, &dependent),
            Err(DftError::LinearAlgebra(_))
        ));
        assert!(matches!(
            fermi_lowdin_orbitals(&dependent, &overlap, &occupied),
            Err(DftError::InvalidInput(_))
        ));
    }
}
