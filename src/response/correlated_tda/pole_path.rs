// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic contractions of the second-order propagator on a moving parent.
//! Molecular adapters must supply complete orbital, integral and amplitude jets.
use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::response::spinor_tda::{spinor_tda_eigenpair_taylor, SpinorTdaConfig};
use crate::{DftError, Result};

/// Factorial-normalized coefficients in orthonormal single/double spaces.
/// The correction is u†(K - omega M)u + (C†u)†(omega I - D)^-1(C†u).
/// K, M, D are Hermitian; C maps positive-metric normalized doubles to singles.
/// The supplied tensors must already contain the chosen physical diagram
/// weights. An indefinite signed double-space metric needs a separate adapter.
/// This interface does not define a spin decomposition or produce tensor jets.
pub struct PropagatorTaylor<'a> {
    pub parent: &'a [BlochMatrix],
    pub singles: &'a [BlochMatrix],
    pub metric: &'a [BlochMatrix],
    pub doubles: &'a [BlochMatrix],
    pub coupling: &'a [BlochMatrix],
}

/// Analytic pole correction and the full noncanonical double-space response.
pub struct CorrelatedPoleTaylor {
    pub parent_energies: Vec<f64>,
    pub corrections: Vec<f64>,
    pub energies: Vec<f64>,
    pub double_vectors: Vec<BlochMatrix>,
    pub double_equation_residuals: Vec<f64>,
    pub parent_equation_residuals: Vec<f64>,
    pub parent_normalization_defects: Vec<f64>,
    pub smallest_double_gap: f64,
}

fn axpy(out: &mut BlochMatrix, scale: f64, x: &BlochMatrix) {
    for (a, b) in out.re.as_mut_slice().iter_mut().zip(x.re.as_slice()) {
        *a += scale * b;
    }
    for (a, b) in out.im.as_mut_slice().iter_mut().zip(x.im.as_slice()) {
        *a += scale * b;
    }
}
fn norm(x: &BlochMatrix) -> f64 {
    x.re.as_slice()
        .iter()
        .chain(x.im.as_slice())
        .fold(0.0_f64, |n, v| n.hypot(*v))
}
fn valid(x: &BlochMatrix, rows: usize, cols: usize, hermitian: bool, bound: f64) -> bool {
    x.rows() == rows
        && x.cols() == cols
        && x.re
            .as_slice()
            .iter()
            .chain(x.im.as_slice())
            .all(|v| v.is_finite())
        && (!hermitian
            || (0..rows).all(|i| {
                (0..cols).all(|j| {
                    (x.re[(i, j)] - x.re[(j, i)]).hypot(x.im[(i, j)] + x.im[(j, i)]) <= bound
                })
            }))
}

/// Differentiate the defined second-order pole through order one or two.
/// The noncanonical double resolvent is solved coefficient by coefficient;
/// finite differences, orbital-gap diagonal approximations and denominator
/// floors are not used. Resonances require an enlarged propagator model.
/// This contraction alone is not a molecular correlated nuclear derivative.
pub fn correlated_pole_taylor(
    path: &PropagatorTaylor<'_>,
    root: usize,
    config: &SpinorTdaConfig,
) -> Result<CorrelatedPoleTaylor> {
    let count = path.parent.len();
    if !(2..=3).contains(&count)
        || path.doubles.len() != count
        || path.coupling.len() != count
        || path.singles.len() != count
        || path.metric.len() != count
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
    {
        return Err(DftError::InvalidInput("propagator paths need matching orders zero through one or two and a positive tolerance".into()));
    }
    let ns = path.parent[0].rows();
    let nd = path.doubles[0].rows();
    let bytes = ns
        .checked_add(nd)
        .and_then(|n| n.checked_mul(n))
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| DftError::InvalidInput("propagator path workspace overflow".into()))?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > config.memory_mb {
        return Err(DftError::ResourceLimit {
            operation: "correlated pole path",
            required_mb,
            limit_mb: config.memory_mb,
        });
    }
    if ns == 0
        || path
            .singles
            .iter()
            .chain(path.metric)
            .any(|x| !valid(x, ns, ns, true, config.tolerance))
        || path
            .doubles
            .iter()
            .any(|x| !valid(x, nd, nd, true, config.tolerance))
        || path
            .coupling
            .iter()
            .any(|x| !valid(x, ns, nd, false, config.tolerance))
    {
        return Err(DftError::InvalidInput(
            "invalid propagator tensor dimensions, Hermiticity or finite values".into(),
        ));
    }
    let parent = spinor_tda_eigenpair_taylor(path.parent, root, config)?;
    let (sources, vectors, residuals, gap) = if nd == 0 {
        (
            vec![BlochMatrix::zeros(0, 1); count],
            vec![BlochMatrix::zeros(0, 1); count],
            vec![0.0; count],
            f64::INFINITY,
        )
    } else {
        let identity = BlochMatrix {
            re: Matrix::identity(nd),
            im: Matrix::zeros(nd, nd),
        };
        let spectral = generalized_orbitals(&path.doubles[0], &identity, 1e-12)?;
        let defect = spectral.eigen_residual.max(spectral.metric_defect);
        if !defect.is_finite() || defect > config.tolerance {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual: defect,
            });
        }
        let gap = spectral
            .energies
            .iter()
            .map(|e| (parent.energies[0] - e).abs())
            .fold(f64::INFINITY, f64::min);
        if gap <= config.minimum_root_gap {
            return Err(DftError::Unsupported {
                feature: "correlated pole derivative at a double-space resonance".into(),
                missing: "an enlarged degenerate propagator space".into(),
            });
        }
        let uh = adjoint(&spectral.coefficients)?;
        let mut sources = Vec::with_capacity(count);
        for k in 0..count {
            let mut w = BlochMatrix::zeros(nd, 1);
            for m in 0..=k {
                axpy(
                    &mut w,
                    1.0,
                    &matmul(&adjoint(&path.coupling[m])?, &parent.vectors[k - m])?,
                );
            }
            sources.push(w);
        }
        let mut vectors: Vec<BlochMatrix> = Vec::with_capacity(count);
        let mut residuals = Vec::with_capacity(count);
        for k in 0..count {
            let mut rhs = sources[k].clone();
            for m in 1..=k {
                axpy(&mut rhs, 1.0, &matmul(&path.doubles[m], &vectors[k - m])?);
                axpy(&mut rhs, -parent.energies[m], &vectors[k - m]);
            }
            let mut transformed = matmul(&uh, &rhs)?;
            for i in 0..nd {
                let denominator = parent.energies[0] - spectral.energies[i];
                transformed.re[(i, 0)] /= denominator;
                transformed.im[(i, 0)] /= denominator;
            }
            let vector = matmul(&spectral.coefficients, &transformed)?;
            vectors.push(vector);
            let mut residual = sources[k].clone();
            for m in 0..=k {
                axpy(
                    &mut residual,
                    1.0,
                    &matmul(&path.doubles[m], &vectors[k - m])?,
                );
                axpy(&mut residual, -parent.energies[m], &vectors[k - m]);
            }
            let error = norm(&residual);
            if !error.is_finite() || error > config.tolerance {
                return Err(DftError::ResponseNotConverged {
                    iterations: k,
                    residual: error,
                });
            }
            residuals.push(error);
        }
        (sources, vectors, residuals, gap)
    };
    let mut corrections = Vec::with_capacity(count);
    for k in 0..count {
        let (mut re, mut im) = (0.0, 0.0);
        for a in 0..=k {
            let left = adjoint(&parent.vectors[a])?;
            for b in 0..=k - a {
                let c = k - a - b;
                let value = matmul(&left, &matmul(&path.singles[b], &parent.vectors[c])?)?;
                re += value.re[(0, 0)];
                im += value.im[(0, 0)];
                for d in 0..=c {
                    let value = matmul(&left, &matmul(&path.metric[b], &parent.vectors[d])?)?;
                    re -= parent.energies[c - d] * value.re[(0, 0)];
                    im -= parent.energies[c - d] * value.im[(0, 0)];
                }
            }
            if nd != 0 {
                let value = matmul(&adjoint(&sources[a])?, &vectors[k - a])?;
                re += value.re[(0, 0)];
                im += value.im[(0, 0)];
            }
        }
        if !re.is_finite() || !im.is_finite() || im.abs() > config.tolerance {
            return Err(DftError::ResponseNotConverged {
                iterations: k,
                residual: if re.is_finite() && im.is_finite() {
                    im.abs()
                } else {
                    f64::INFINITY
                },
            });
        }
        corrections.push(re);
    }
    let energies: Vec<f64> = parent
        .energies
        .iter()
        .zip(&corrections)
        .map(|(a, b)| a + b)
        .collect();
    if energies.iter().any(|v| !v.is_finite()) {
        return Err(DftError::ResponseNotConverged {
            iterations: count - 1,
            residual: f64::INFINITY,
        });
    }
    Ok(CorrelatedPoleTaylor {
        parent_energies: parent.energies,
        corrections,
        energies,
        double_vectors: vectors,
        double_equation_residuals: residuals,
        parent_equation_residuals: parent.equation_residuals,
        parent_normalization_defects: parent.normalization_defects,
        smallest_double_gap: gap,
    })
}
