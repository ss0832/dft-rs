// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex generalized orbitals reconstructed from the existing doubled
//! real eigensolver. Degenerate real eigenvectors are rank-revealed in the
//! complex overlap metric rather than selecting every second column.

use crate::linalg::{Matrix, Orthogonalizer};
use crate::periodic::bloch::BlochMatrix;
use crate::response::complex::{dot, Complex};
use crate::{DftError, Result};

fn validate(a: &BlochMatrix) -> Result<()> {
    if a.re.rows != a.im.rows
        || a.re.cols != a.im.cols
        || a.re
            .as_slice()
            .iter()
            .chain(a.im.as_slice())
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid complex matrix parts".into(),
        ));
    }
    Ok(())
}
pub fn adjoint(a: &BlochMatrix) -> Result<BlochMatrix> {
    validate(a)?;
    let re = a.re.transpose();
    let mut im = a.im.transpose();
    for x in im.as_mut_slice() {
        *x = -*x;
    }
    Ok(BlochMatrix { re, im })
}
pub fn matmul(a: &BlochMatrix, b: &BlochMatrix) -> Result<BlochMatrix> {
    validate(a)?;
    validate(b)?;
    if a.cols() != b.rows() {
        return Err(DftError::InvalidInput(
            "complex matrix multiplication dimension mismatch".into(),
        ));
    }
    let mut re = a.re.matmul_seq(&b.re);
    let ii = a.im.matmul_seq(&b.im);
    let mut im = a.re.matmul_seq(&b.im);
    let ir = a.im.matmul_seq(&b.re);
    for (a, &b) in re.as_mut_slice().iter_mut().zip(ii.as_slice()) {
        *a -= b;
    }
    for (a, &b) in im.as_mut_slice().iter_mut().zip(ir.as_slice()) {
        *a += b;
    }
    Ok(BlochMatrix { re, im })
}
fn z(a: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(a.re[(i, j)], a.im[(i, j)])
}

fn action(a: &BlochMatrix, v: &[Complex]) -> Vec<Complex> {
    (0..a.rows())
        .map(|i| (0..a.cols()).fold(Complex::ZERO, |s, j| s + z(a, i, j) * v[j]))
        .collect()
}

/// Eigenvalues of the doubled real problem closer than this (relative to
/// `max(1, |e|)`) are reconstructed together as one cluster.
const CLUSTER_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct BandOrbitals {
    pub energies: Vec<f64>,
    pub coefficients: BlochMatrix,
    pub eigen_residual: f64,
    pub metric_defect: f64,
}

/// Identify only structurally spin-diagonal matrices. Solving the two
/// blocks separately preserves transverse zeros even near degeneracies.
pub(crate) fn exact_spin_blocks(matrix: &BlochMatrix) -> Option<[BlochMatrix; 2]> {
    let n = matrix.rows() / 2;
    if n == 0 || matrix.rows() != 2 * n {
        return None;
    }
    let mut blocks = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
    for i in 0..n {
        for j in 0..n {
            if z(matrix, i, n + j) != Complex::ZERO || z(matrix, n + i, j) != Complex::ZERO {
                return None;
            }
            for (spin, block) in blocks.iter_mut().enumerate() {
                let value = z(matrix, spin * n + i, spin * n + j);
                block.re[(i, j)] = value.re;
                block.im[(i, j)] = value.im;
            }
        }
    }
    Some(blocks)
}

fn equal_blocks([alpha, beta]: &[BlochMatrix; 2]) -> bool {
    alpha.re.as_slice() == beta.re.as_slice() && alpha.im.as_slice() == beta.im.as_slice()
}

fn exact_spin_orbitals(
    fock: &BlochMatrix,
    overlap: &BlochMatrix,
    overlap_cutoff: f64,
) -> Result<Option<BandOrbitals>> {
    let Some(fock_blocks) = exact_spin_blocks(fock) else {
        return Ok(None);
    };
    let Some(metrics) = exact_spin_blocks(overlap).filter(equal_blocks) else {
        return Ok(None);
    };
    // A common metric retains the same spatial rank and cutoff in both
    // spin channels. Exactly repeated Fock blocks share one eigensolve.
    let spatial = generalized_orbitals(&fock_blocks[0], &metrics[0], overlap_cutoff)?;
    let other = (!equal_blocks(&fock_blocks))
        .then(|| generalized_orbitals(&fock_blocks[1], &metrics[0], overlap_cutoff))
        .transpose()?;
    let spins = [&spatial, other.as_ref().unwrap_or(&spatial)];
    let mut order: Vec<_> = (0..spatial.energies.len())
        .flat_map(|orbital| [(0, orbital), (1, orbital)])
        .collect();
    order.sort_by(|&(a, i), &(b, j)| spins[a].energies[i].total_cmp(&spins[b].energies[j]));

    let n = metrics[0].rows();
    let mut coefficients = BlochMatrix::zeros(2 * n, order.len());
    let mut energies = Vec::with_capacity(order.len());
    for (column, (spin, orbital)) in order.into_iter().enumerate() {
        let block = spins[spin];
        energies.push(block.energies[orbital]);
        for i in 0..n {
            let row = spin * n + i;
            coefficients.re[(row, column)] = block.coefficients.re[(i, orbital)];
            coefficients.im[(row, column)] = block.coefficients.im[(i, orbital)];
        }
    }
    let eigen_residual = if other.is_none() {
        spatial.eigen_residual
    } else {
        // Preserve the relative Frobenius residual of the full pencil.
        let mut residual = 0.0_f64;
        let mut norm = 0.0_f64;
        for (block, bands) in fock_blocks.iter().zip(spins) {
            let fc = matmul(block, &bands.coefficients)?;
            let sc = matmul(&metrics[0], &bands.coefficients)?;
            for (j, &e) in bands.energies.iter().enumerate() {
                for i in 0..n {
                    let f = z(&fc, i, j);
                    residual = residual.hypot((f - z(&sc, i, j) * e).abs());
                    norm = norm.hypot(f.abs());
                }
            }
        }
        if norm == 0.0 {
            residual
        } else {
            residual / norm
        }
    };
    Ok(Some(BandOrbitals {
        energies,
        coefficients,
        eigen_residual,
        metric_defect: spins[0].metric_defect.max(spins[1].metric_defect),
    }))
}

/// Keep an exactly real pencil real. Reconstructing it from the doubled
/// spectrum can otherwise give nominally real orbitals unequal phases in
/// their components and seed a spurious orbital current.
fn real_orbitals(fock: &Matrix, overlap: &Matrix, cutoff: f64) -> Result<BandOrbitals> {
    let orth = Orthogonalizer::canonical(overlap, cutoff)?;
    let (energies, coefficients) = orth.solve_roothaan(fock)?;
    let rank = energies.len();
    let fc = fock.matmul_seq(&coefficients);
    let sc = overlap.matmul_seq(&coefficients);
    let metric_defect = coefficients
        .transpose()
        .matmul_seq(&sc)
        .max_abs_diff(&Matrix::identity(rank));
    let mut residual = 0.0_f64;
    let mut norm = 0.0_f64;
    for (j, &energy) in energies.iter().enumerate() {
        for i in 0..fock.rows {
            residual = residual.hypot(fc[(i, j)] - sc[(i, j)] * energy);
            norm = norm.hypot(fc[(i, j)]);
        }
    }
    let eigen_residual = if norm == 0.0 {
        residual
    } else {
        residual / norm
    };
    if metric_defect > 1e-7 || (rank == fock.rows && eigen_residual > 1e-6) {
        return Err(DftError::LinearAlgebra(format!(
            "real orbital reconstruction failed: metric defect {metric_defect:e}, \
             eigen residual {eigen_residual:e}"
        )));
    }
    Ok(BandOrbitals {
        energies,
        coefficients: BlochMatrix {
            im: Matrix::zeros(coefficients.rows, coefficients.cols),
            re: coefficients,
        },
        eigen_residual,
        metric_defect,
    })
}

pub fn generalized_orbitals(
    fock: &BlochMatrix,
    overlap: &BlochMatrix,
    overlap_cutoff: f64,
) -> Result<BandOrbitals> {
    validate(fock)?;
    validate(overlap)?;
    let n = overlap.rows();
    if n == 0
        || overlap.cols() != n
        || fock.rows() != n
        || fock.cols() != n
        || overlap.hermiticity_error() > 1e-10
        || fock.hermiticity_error() > 1e-10
        || !overlap_cutoff.is_finite()
        || overlap_cutoff <= 0.0
    {
        return Err(DftError::InvalidInput("generalized complex orbitals require finite Hermitian F/S and a positive overlap cutoff".into()));
    }
    if let Some(orbitals) = exact_spin_orbitals(fock, overlap, overlap_cutoff)? {
        return Ok(orbitals);
    }
    if fock
        .im
        .as_slice()
        .iter()
        .chain(overlap.im.as_slice())
        .all(|&v| v == 0.0)
    {
        return real_orbitals(&fock.re, &overlap.re, overlap_cutoff);
    }
    let orth = Orthogonalizer::canonical(&overlap.real_embedding(), overlap_cutoff)?;
    if orth.n_kept % 2 != 0 {
        return Err(DftError::LinearAlgebra(
            "complex overlap cutoff retained an odd doubled-real rank".into(),
        ));
    }
    let rank = orth.n_kept / 2;
    let (energies2, coefficients2) = orth.solve_roothaan(&fock.real_embedding())?;
    // Every complex eigenvalue appears twice in the doubled real problem, and
    // its two real eigenvectors give the same complex direction. The real
    // solver separates the eigenvectors of close eigenvalues only up to
    // `eps / gap`, so the columns are grouped into clusters of close
    // eigenvalues and a cluster of `m` real columns contributes exactly
    // `m / 2` complex vectors, chosen by Gram-Schmidt with pivoting on the
    // largest remaining norm. No rank threshold is involved.
    let mut columns: Vec<Vec<Complex>> = Vec::with_capacity(rank);
    let mut metric_columns: Vec<Vec<Complex>> = Vec::with_capacity(rank);
    let mut energies: Vec<f64> = Vec::with_capacity(rank);
    let total = energies2.len();
    let mut start = 0;
    while start < total {
        let mut end = start + 1;
        while end < total
            && energies2[end] - energies2[end - 1]
                <= CLUSTER_TOLERANCE * energies2[end].abs().max(1.0)
        {
            end += 1;
        }
        let size = end - start;
        if size % 2 != 0 {
            return Err(DftError::LinearAlgebra(
                "odd cluster in the doubled real spectrum".into(),
            ));
        }
        // The complex vectors of the cluster, orthogonal to earlier clusters.
        let mut candidates: Vec<(Vec<Complex>, Vec<Complex>)> = (start..end)
            .map(|column| {
                let mut vector: Vec<Complex> = (0..n)
                    .map(|i| {
                        Complex::new(coefficients2[(i, column)], coefficients2[(n + i, column)])
                    })
                    .collect();
                let mut sv = action(overlap, &vector);
                for _ in 0..2 {
                    for (q, sq) in columns.iter().zip(&metric_columns) {
                        let amplitude = dot(q, &sv);
                        for i in 0..n {
                            vector[i] -= q[i] * amplitude;
                            sv[i] -= sq[i] * amplitude;
                        }
                    }
                }
                (vector, sv)
            })
            .collect();
        let first_new = columns.len();
        for _ in 0..size / 2 {
            let (best, length2) = candidates
                .iter()
                .enumerate()
                .map(|(index, (v, sv))| (index, dot(v, sv).re))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            if !(length2 > 0.0) {
                return Err(DftError::LinearAlgebra(
                    "non-positive complex orbital metric".into(),
                ));
            }
            let (mut vector, mut sv) = candidates.swap_remove(best);
            let length = length2.sqrt();
            for i in 0..n {
                vector[i] = vector[i] / length;
                sv[i] = sv[i] / length;
            }
            for (other, other_sv) in candidates.iter_mut() {
                for _ in 0..2 {
                    let amplitude = dot(&vector, other_sv);
                    for i in 0..n {
                        other[i] -= vector[i] * amplitude;
                        other_sv[i] -= sv[i] * amplitude;
                    }
                }
            }
            columns.push(vector);
            metric_columns.push(sv);
        }
        // Rayleigh energies, ascending inside the cluster.
        let mut cluster: Vec<(f64, Vec<Complex>, Vec<Complex>)> = columns
            .drain(first_new..)
            .zip(metric_columns.drain(first_new..))
            .map(|(v, sv)| (dot(&v, &action(fock, &v)).re, v, sv))
            .collect();
        cluster.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (energy, v, sv) in cluster {
            energies.push(energy);
            columns.push(v);
            metric_columns.push(sv);
        }
        start = end;
    }
    if columns.len() != rank {
        return Err(DftError::LinearAlgebra(
            "failed to reveal the complete complex orbital rank".into(),
        ));
    }
    let mut coefficients = BlochMatrix::zeros(n, rank);
    for j in 0..rank {
        for i in 0..n {
            coefficients.re[(i, j)] = columns[j][i].re;
            coefficients.im[(i, j)] = columns[j][i].im;
        }
    }
    let metric = matmul(&adjoint(&coefficients)?, &matmul(overlap, &coefficients)?)?;
    let mut metric_defect = metric.re.max_abs_diff(&Matrix::identity(rank));
    metric_defect = metric_defect.max(
        metric
            .im
            .as_slice()
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs())),
    );
    let mut residual_norm = 0.0_f64;
    let mut fock_norm = 0.0_f64;
    for j in 0..rank {
        let fv = action(fock, &columns[j]);
        for i in 0..n {
            residual_norm = residual_norm.hypot((fv[i] - metric_columns[j][i] * energies[j]).abs());
            fock_norm = fock_norm.hypot(fv[i].abs());
        }
    }
    let eigen_residual = if fock_norm == 0.0 {
        residual_norm
    } else {
        residual_norm / fock_norm
    };
    // Without discarded directions the vectors are exact eigenvectors; a
    // large residual means the reconstruction failed and must not be used.
    if metric_defect > 1e-7 || (rank == n && eigen_residual > 1e-6) {
        return Err(DftError::LinearAlgebra(format!(
            "complex orbital reconstruction failed: metric defect {metric_defect:e}, \
             eigen residual {eigen_residual:e}"
        )));
    }
    Ok(BandOrbitals {
        energies,
        coefficients,
        eigen_residual,
        metric_defect,
    })
}

/// Contravariant AO density from normalized complex orbitals and per-spin
/// band occupations. This does not impose time reversal or Hermitianize a
/// finite-q response; it constructs the Hermitian ground-state density.
pub fn band_density(orbitals: &BandOrbitals, occupations: &[f64]) -> Result<BlochMatrix> {
    if occupations.len() != orbitals.energies.len()
        || occupations
            .iter()
            .any(|f| !f.is_finite() || !(0.0..=1.0).contains(f))
    {
        return Err(DftError::InvalidInput("invalid band occupations".into()));
    }
    density_with_weights(orbitals, occupations)
}

/// Canonical energy multiplier from the accepted bands and occupations.
pub(crate) fn energy_weighted_band_density(
    orbitals: &BandOrbitals,
    occupations: &[f64],
) -> Result<BlochMatrix> {
    if occupations.len() != orbitals.energies.len()
        || occupations.iter().any(|f| !(0.0..=1.0).contains(f))
    {
        return Err(DftError::InvalidInput("invalid band occupations".into()));
    }
    let weights: Vec<_> = occupations
        .iter()
        .zip(&orbitals.energies)
        .map(|(f, e)| f * e)
        .collect();
    density_with_weights(orbitals, &weights)
}

fn density_with_weights(orbitals: &BandOrbitals, weights: &[f64]) -> Result<BlochMatrix> {
    let mut weighted = orbitals.coefficients.clone();
    for i in 0..weighted.rows() {
        for (j, &f) in weights.iter().enumerate() {
            weighted.re[(i, j)] *= f;
            weighted.im[(i, j)] *= f;
        }
    }
    matmul(&weighted, &adjoint(&orbitals.coefficients)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `F = U diag(values) U^+` with the unitary Householder reflector
    /// `U = 1 - 2 w w^+ / (w^+ w)` of a complex vector `w`.
    fn hermitian(values: &[f64]) -> BlochMatrix {
        let n = values.len();
        let w: Vec<Complex> = (0..n)
            .map(|i| Complex::new(0.3 + 0.17 * i as f64, 0.11 - 0.07 * (i * i) as f64))
            .collect();
        let norm2: f64 = w.iter().map(|z| z.re * z.re + z.im * z.im).sum();
        let u = |i: usize, j: usize| -> Complex {
            let delta = if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::ZERO
            };
            delta - w[i] * w[j].conj() * (2.0 / norm2)
        };
        let mut f = BlochMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::ZERO;
                for (k, value) in values.iter().enumerate() {
                    sum += u(i, k) * u(j, k).conj() * *value;
                }
                f.re[(i, j)] = sum.re;
                f.im[(i, j)] = sum.im;
            }
        }
        f.make_hermitian();
        f
    }

    /// Two levels split by far less than the spread of the spectrum (the
    /// pi levels of a diatomic molecule on a quadrature grid are split by
    /// about 2e-8 Hartree) must not produce a spurious vector.
    #[test]
    fn near_degenerate_levels_are_reconstructed_without_spurious_vectors() {
        for split in [0.0, 1e-12, 1e-10, 2e-8, 1e-6, 1e-4] {
            let values = [-23.6, -0.92, -0.27, -0.068, -0.068 + split, 0.34];
            let f = hermitian(&values);
            let mut s = BlochMatrix::zeros(6, 6);
            for i in 0..6 {
                s.re[(i, i)] = 1.0;
            }
            let orbitals = generalized_orbitals(&f, &s, 1e-12).unwrap();
            assert_eq!(orbitals.energies.len(), 6);
            let worst = orbitals
                .energies
                .iter()
                .zip(&values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "splitting {split:e}: largest level error {worst:e}, eigen residual {:e}, metric defect {:e}",
                orbitals.eigen_residual, orbitals.metric_defect
            );
            assert!(worst < 1e-10, "splitting {split:e}");
            assert!(
                orbitals.eigen_residual < 1e-9 && orbitals.metric_defect < 1e-12,
                "splitting {split:e}"
            );
        }
    }

    #[test]
    fn a_real_pencil_has_zero_current_and_a_complex_source_retains_it() {
        let mut metric = BlochMatrix::zeros(3, 3);
        metric.re = Matrix::from_row_major(
            3,
            3,
            vec![1.12, 0.06, 0.1, 0.06, 0.91, -0.04, 0.1, -0.04, 1.03],
        );
        let mut fock = BlochMatrix::zeros(3, 3);
        fock.re = Matrix::from_row_major(
            3,
            3,
            vec![-1.3, 0.22, -0.06, 0.22, -0.4, 0.18, -0.06, 0.18, 0.31],
        );
        let real = generalized_orbitals(&fock, &metric, 1e-12).unwrap();
        let occupations = [1.0, 0.0, 0.0];
        let real_density = band_density(&real, &occupations).unwrap();
        assert!(real.coefficients.im.as_slice().iter().all(|&x| x == 0.0));
        assert!(real_density.im.as_slice().iter().all(|&x| x == 0.0));
        assert!(real.metric_defect < 1e-12 && real.eigen_residual < 1e-12);
        fock.im[(0, 1)] = 0.013;
        fock.im[(1, 0)] = -0.013;
        let complex = generalized_orbitals(&fock, &metric, 1e-12).unwrap();
        let complex_density = band_density(&complex, &occupations).unwrap();
        assert!(complex.metric_defect < 1e-12 && complex.eigen_residual < 1e-12);
        assert!(complex_density.im[(0, 1)].abs() > 1e-3);
        for value in fock.im.as_mut_slice() {
            *value = -*value;
        }
        let reversed = generalized_orbitals(&fock, &metric, 1e-12).unwrap();
        let mut reversed_density = band_density(&reversed, &occupations).unwrap();
        for value in reversed_density.im.as_mut_slice() {
            *value = -*value;
        }
        assert!(complex_density.re.max_abs_diff(&reversed_density.re) < 1e-12);
        assert!(complex_density.im.max_abs_diff(&reversed_density.im) < 1e-12);
    }

    #[test]
    fn different_exact_spin_blocks_match_the_rotated_dense_eigenspace() {
        let mut l = BlochMatrix::zeros(4, 4);
        let mut rotation = BlochMatrix::zeros(4, 4);
        for start in [0, 2] {
            l.re[(start, start)] = 1.1;
            l.re[(start, start + 1)] = 0.12;
            l.re[(start + 1, start + 1)] = 0.93;
        }
        for i in 0..2 {
            rotation.re[(i, i)] = 0.8;
            rotation.re[(2 + i, 2 + i)] = 0.8;
            rotation.re[(i, 2 + i)] = -0.6;
            rotation.re[(2 + i, i)] = 0.6;
        }
        let left = adjoint(&l).unwrap();
        let metric = matmul(&left, &l).unwrap();
        for split in [0.0, 1e-10, 2e-8, 1e-3] {
            let expected = [-0.91, -0.91 + split, 0.2, 1.23];
            let mut diagonal = BlochMatrix::zeros(4, 4);
            for (spin, levels) in [[expected[0], expected[2]], [expected[1], expected[3]]]
                .into_iter()
                .enumerate()
            {
                let spatial = hermitian(&levels);
                for i in 0..2 {
                    for j in 0..2 {
                        diagonal.re[(2 * spin + i, 2 * spin + j)] = spatial.re[(i, j)];
                        diagonal.im[(2 * spin + i, 2 * spin + j)] = spatial.im[(i, j)];
                    }
                }
            }
            let fock = matmul(&left, &matmul(&diagonal, &l).unwrap()).unwrap();
            let exact = generalized_orbitals(&fock, &metric, 1e-12).unwrap();
            let rotate = |matrix| {
                matmul(
                    &matmul(&rotation, matrix).unwrap(),
                    &adjoint(&rotation).unwrap(),
                )
                .unwrap()
            };
            let dense_fock = rotate(&fock);
            assert!(exact_spin_blocks(&dense_fock).is_none());
            let dense = generalized_orbitals(&dense_fock, &metric, 1e-12).unwrap();
            for (bands, label) in [(&exact, "split"), (&dense, "dense")] {
                let error = bands
                    .energies
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(error < 1e-10, "{label} energy error {error:e}");
                assert!(bands.eigen_residual < 1e-9 && bands.metric_defect < 1e-12);
            }
            // The two nearly degenerate occupied states are compared as a
            // projector, independent of phases and choices of eigenbasis.
            let active = [1.0, 1.0, 0.0, 0.0];
            let split_density = band_density(&exact, &active).unwrap();
            let dense_density = band_density(&dense, &active).unwrap();
            let rotated = rotate(&split_density);
            let error = rotated
                .re
                .max_abs_diff(&dense_density.re)
                .max(rotated.im.max_abs_diff(&dense_density.im));
            assert!(error < 1e-11, "occupied projector error {error:e}");
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(z(&split_density, i, 2 + j), Complex::ZERO);
                }
            }
            println!("spin splitting {split:e}: occupied projector difference {error:e}");
        }
    }
}
