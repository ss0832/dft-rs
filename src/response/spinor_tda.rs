// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Tamm-Dancoff excitations in the complete complex occupied/empty spinor
//! space. The kernel must be the complex-linear extension of the same
//! reference energy Hessian, including transverse spin channels. No exact
//! total-spin or multiplicity label is assigned to these states.

use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

mod pole_vector;
pub use pole_vector::{spinor_tda_eigenpair_taylor, SpinorPoleTaylor};

#[derive(Clone, Debug)]
pub struct SpinorTdaConfig {
    pub states: usize,
    pub memory_mb: usize,
    /// Absolute metric, Hermiticity and eigen-equation acceptance bound.
    pub tolerance: f64,
    /// Isolated-root derivatives refuse closer poles; no denominator floor.
    pub minimum_root_gap: f64,
}

impl Default for SpinorTdaConfig {
    fn default() -> Self {
        Self {
            states: 5,
            memory_mb: 256,
            tolerance: 1e-9,
            minimum_root_gap: 1e-6,
        }
    }
}

/// Factorial-normalized coefficients of one isolated Hermitian TDA pole
/// through second order. Operator coefficients include the complete moving
/// orbital/kernel response. Finite differences are not used here.
pub fn spinor_tda_pole_taylor(
    operator: &[BlochMatrix],
    root: usize,
    config: &SpinorTdaConfig,
) -> Result<Vec<f64>> {
    let spectral = prepare_pole(operator, root, config)?;
    pole_energy_coefficients(operator, root, &spectral, config)
}
fn prepare_pole(
    operator: &[BlochMatrix],
    root: usize,
    config: &SpinorTdaConfig,
) -> Result<crate::dfpt::orbitals::BandOrbitals> {
    if !(2..=3).contains(&operator.len())
        || !config.minimum_root_gap.is_finite()
        || config.minimum_root_gap <= 0.0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
    {
        return Err(DftError::InvalidInput("isolated spinor pole derivatives require orders zero through one or two and positive tolerances".into()));
    }
    let dim = operator[0].rows();
    let bytes = dim
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(512))
        .ok_or_else(|| {
            DftError::InvalidInput("spinor pole derivative workspace overflow".into())
        })?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "complex spinor pole derivatives",
            required_mb,
            limit_mb: config.memory_mb,
        });
    }
    if dim == 0
        || root >= dim
        || operator.iter().any(|a| {
            a.rows() != dim
                || a.cols() != dim
                || a.im.rows != dim
                || a.im.cols != dim
                || a.re
                    .as_slice()
                    .iter()
                    .chain(a.im.as_slice())
                    .any(|v| !v.is_finite())
                || a.hermiticity_error() > config.tolerance
        })
    {
        return Err(DftError::InvalidInput(
            "spinor pole coefficients must be matching finite Hermitian matrices with a valid root"
                .into(),
        ));
    }
    let identity = BlochMatrix {
        re: Matrix::identity(dim),
        im: Matrix::zeros(dim, dim),
    };
    let spectral = generalized_orbitals(&operator[0], &identity, 1e-12)?;
    if spectral.eigen_residual > config.tolerance || spectral.metric_defect > config.tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: spectral.eigen_residual.max(spectral.metric_defect),
        });
    }
    let energy = spectral.energies[root];
    if spectral
        .energies
        .iter()
        .enumerate()
        .any(|(k, e)| k != root && (e - energy).abs() <= config.minimum_root_gap)
    {
        return Err(DftError::Unsupported {
            feature: "isolated spinor excited-state derivative at a degenerate or crossing pole"
                .into(),
            missing: "a defined transported degenerate-subspace energy model".into(),
        });
    }
    Ok(spectral)
}
fn pole_energy_coefficients(
    operator: &[BlochMatrix],
    root: usize,
    spectral: &crate::dfpt::orbitals::BandOrbitals,
    config: &SpinorTdaConfig,
) -> Result<Vec<f64>> {
    let energy = spectral.energies[root];
    let vector = columns(&spectral.coefficients, root, 1);
    let dagger = adjoint(&vector)?;
    let first_action = matmul(&operator[1], &vector)?;
    let first = matmul(&dagger, &first_action)?;
    if first.im[(0, 0)].abs() > config.tolerance {
        return Err(DftError::LinearAlgebra(
            "nonreal spinor excitation energy derivative".into(),
        ));
    }
    let mut result = vec![energy, first.re[(0, 0)]];
    if operator.len() == 3 {
        let explicit = matmul(&matmul(&dagger, &operator[2])?, &vector)?;
        if explicit.im[(0, 0)].abs() > config.tolerance {
            return Err(DftError::LinearAlgebra(
                "nonreal spinor excitation second coefficient".into(),
            ));
        }
        let mut second = explicit.re[(0, 0)];
        for (k, &other_energy) in spectral.energies.iter().enumerate() {
            if k != root {
                let other = columns(&spectral.coefficients, k, 1);
                let coupling = matmul(&adjoint(&other)?, &first_action)?;
                second += (coupling.re[(0, 0)].powi(2) + coupling.im[(0, 0)].powi(2))
                    / (energy - other_energy);
            }
        }
        result.push(second);
    }
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct SpinorTdaState {
    /// Excitation energy in Hartree. Negative poles diagnose instability;
    /// they are retained rather than clipped or relabelled.
    pub energy: f64,
    /// Unit-normalized complex amplitudes, empty spinor by occupied spinor.
    pub amplitudes: BlochMatrix,
    /// Non-Hermitian AO transition density C_v X C_o^dagger.
    pub transition_density: BlochMatrix,
    pub residual: f64,
    /// Nearest pole separation, including roots beyond the requested subset.
    /// A zero gap identifies a subspace, not a differentiable isolated root.
    pub nearest_gap: f64,
}

fn columns(matrix: &BlochMatrix, first: usize, count: usize) -> BlochMatrix {
    let mut result = BlochMatrix::zeros(matrix.rows(), count);
    for i in 0..matrix.rows() {
        for j in 0..count {
            result.re[(i, j)] = matrix.re[(i, first + j)];
            result.im[(i, j)] = matrix.im[(i, first + j)];
        }
    }
    result
}

/// Solve A X = omega X with A X = gaps*X + C_v^dagger G[C_v X C_o^dagger] C_o.
/// The occupations select the first `occupied` columns of `coefficients`.
/// Full spinor transitions include both collinear spin-conserving and
/// spin-flip blocks; no projection onto a guessed spin sector is performed.
///
/// This dense reference implementation explicitly budgets the spectral
/// workspace. A non-Hermitian operator or non-complex-linear kernel is an
/// error, never repaired by averaging its matrix with its adjoint.
pub fn spinor_tda_states_with_kernel(
    coefficients: &BlochMatrix,
    energies: &[f64],
    occupied: usize,
    overlap: &BlochMatrix,
    kernel: &dyn Fn(&BlochMatrix) -> Result<BlochMatrix>,
    config: &SpinorTdaConfig,
) -> Result<Vec<SpinorTdaState>> {
    let rows = coefficients.rows();
    let spinors = coefficients.cols();
    if rows == 0
        || spinors == 0
        || occupied == 0
        || occupied >= spinors
        || energies.len() != spinors
        || energies.iter().any(|x| !x.is_finite())
        || config.states == 0
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || overlap.rows() != rows
        || overlap.cols() != rows
    {
        return Err(DftError::InvalidInput("spinor TDA requires finite orbitals, integer occupied/empty sectors, a matching metric and positive controls".into()));
    }
    let empty = spinors - occupied;
    let dim = occupied
        .checked_mul(empty)
        .ok_or_else(|| DftError::InvalidInput("spinor excitation dimension overflow".into()))?;
    let bytes = dim
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(512))
        .and_then(|x| {
            rows.checked_mul(rows)
                .and_then(|n| n.checked_mul(16 * 12))
                .and_then(|n| x.checked_add(n))
        })
        .ok_or_else(|| DftError::InvalidInput("spinor excitation workspace overflow".into()))?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "dense complex spinor TDA",
            required_mb,
            limit_mb: config.memory_mb,
        });
    }
    // adjoint/matmul validate both real and imaginary matrix storage.
    let metric = matmul(&matmul(&adjoint(coefficients)?, overlap)?, coefficients)?;
    let metric_error = metric.re.max_abs_diff(&Matrix::identity(spinors)).max(
        metric
            .im
            .as_slice()
            .iter()
            .fold(0.0_f64, |a, x| a.max(x.abs())),
    );
    if overlap.hermiticity_error() > config.tolerance || metric_error > config.tolerance {
        return Err(DftError::InvalidInput(
            "spinor excitation orbitals are not orthonormal in the supplied Hermitian AO metric"
                .into(),
        ));
    }
    let co = columns(coefficients, 0, occupied);
    let cv = columns(coefficients, occupied, empty);
    let co_dagger = adjoint(&co)?;
    let cv_dagger = adjoint(&cv)?;
    let action = |x: &BlochMatrix| -> Result<(BlochMatrix, BlochMatrix)> {
        let density = matmul(&matmul(&cv, x)?, &co_dagger)?;
        let induced = kernel(&density)?;
        let mut projected = matmul(&matmul(&cv_dagger, &induced)?, &co)?;
        for a in 0..empty {
            for i in 0..occupied {
                let gap = energies[occupied + a] - energies[i];
                projected.re[(a, i)] += gap * x.re[(a, i)];
                projected.im[(a, i)] += gap * x.im[(a, i)];
            }
        }
        Ok((projected, induced))
    };
    let mut operator = BlochMatrix::zeros(dim, dim);
    for column in 0..dim {
        let mut unit = BlochMatrix::zeros(empty, occupied);
        unit.re[(column / occupied, column % occupied)] = 1.0;
        let (real, induced_real) = action(&unit)?;
        unit.re[(column / occupied, column % occupied)] = 0.0;
        unit.im[(column / occupied, column % occupied)] = 1.0;
        let (_, induced_imaginary) = action(&unit)?;
        for i in 0..rows {
            for j in 0..rows {
                if (induced_imaginary.re[(i, j)] + induced_real.im[(i, j)])
                    .hypot(induced_imaginary.im[(i, j)] - induced_real.re[(i, j)])
                    > config.tolerance
                {
                    return Err(DftError::InvalidInput("spinor excitation kernel must extend the physical real Hessian complex-linearly to non-Hermitian transition densities".into()));
                }
            }
        }
        for row in 0..dim {
            let a = row / occupied;
            let i = row % occupied;
            operator.re[(row, column)] = real.re[(a, i)];
            operator.im[(row, column)] = real.im[(a, i)];
        }
    }
    if operator.hermiticity_error() > config.tolerance {
        return Err(DftError::LinearAlgebra(
            "the complex spinor TDA operator is not Hermitian".into(),
        ));
    }
    let identity = BlochMatrix {
        re: Matrix::identity(dim),
        im: Matrix::zeros(dim, dim),
    };
    let spectral = generalized_orbitals(&operator, &identity, 1e-12)?;
    if spectral.eigen_residual > config.tolerance || spectral.metric_defect > config.tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: spectral.eigen_residual.max(spectral.metric_defect),
        });
    }
    let mut states = Vec::new();
    for root in 0..config.states.min(dim) {
        let mut amplitudes = BlochMatrix::zeros(empty, occupied);
        for a in 0..empty {
            for i in 0..occupied {
                amplitudes.re[(a, i)] = spectral.coefficients.re[(a * occupied + i, root)];
                amplitudes.im[(a, i)] = spectral.coefficients.im[(a * occupied + i, root)];
            }
        }
        let (product, _) = action(&amplitudes)?;
        let energy = spectral.energies[root];
        let mut residual = 0.0_f64;
        for a in 0..empty {
            for i in 0..occupied {
                residual = residual.hypot(
                    (product.re[(a, i)] - energy * amplitudes.re[(a, i)])
                        .hypot(product.im[(a, i)] - energy * amplitudes.im[(a, i)]),
                );
            }
        }
        if residual > config.tolerance {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual,
            });
        }
        let nearest_gap = spectral
            .energies
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != root)
            .map(|(_, e)| (e - energy).abs())
            .fold(f64::INFINITY, f64::min);
        let transition_density = matmul(&matmul(&cv, &amplitudes)?, &co_dagger)?;
        states.push(SpinorTdaState {
            energy,
            amplitudes,
            transition_density,
            residual,
            nearest_gap,
        });
    }
    Ok(states)
}
