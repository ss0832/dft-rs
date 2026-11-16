// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Collisionless linear response and separable induced fields in a fixed
//! particle/hole metric. Full complex normal and pairing blocks share the
//! same basis and Fermi reference.

use super::nambu_matrices;
use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::coherent_transport::occupation_divided_difference;
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::response::dynamic::Frequency;
use crate::response::krylov::{solve, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::{DftError, Result};
use std::slice::from_ref;

#[derive(Clone, Copy, Debug)]
pub struct NambuReference<'a> {
    pub normal: &'a BlochMatrix,
    pub pairing: &'a BlochMatrix,
    pub overlap: &'a BlochMatrix,
    pub chemical_potential: f64,
    pub temperature_energy: f64,
    pub overlap_cutoff: f64,
}

/// One Hermitian quadratic observable on the full spin-orbital space.
/// The dimensionless pairing block must be antisymmetric.
#[derive(Clone, Debug)]
pub struct NambuProbe {
    pub normal: BlochMatrix,
    pub pairing: BlochMatrix,
}

#[derive(Clone, Debug)]
pub struct NambuCausalResponse {
    /// Bare and induced-field susceptibilities per frequency, in Hartree^-1.
    pub bare: Vec<BlochMatrix>,
    pub response: Vec<BlochMatrix>,
    /// Relative residuals of the induced-field equation, frequency by probe.
    pub coupled_residuals: Vec<Vec<f64>>,
    /// Maximum bare spectral equation defect at each frequency.
    pub bare_residuals: Vec<f64>,
    pub modes: Vec<f64>,
    pub reference_eigen_residual: f64,
    pub reference_metric_defect: f64,
    /// Minimum of unity and min eig(I+A^(1/2) U A^(1/2)), with A=-chi0(0).
    pub stability_margin: f64,
    pub static_gram_negative_defect: f64,
}

#[derive(Debug)]
struct Transition {
    gap: f64,
    occupation_slope: f64,
    vertices: Vec<Complex>,
}

fn element(matrix: &BlochMatrix, row: usize, column: usize) -> Complex {
    Complex::new(matrix.re[(row, column)], matrix.im[(row, column)])
}

struct InducedEquation(BlochMatrix);
impl LinearOperator for InducedEquation {
    fn dimension(&self) -> usize {
        self.0.rows()
    }

    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }

    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        Ok((0..self.dimension())
            .map(|i| {
                x.iter()
                    .enumerate()
                    .fold(Complex::ZERO, |s, (j, &xj)| s + element(&self.0, i, j) * xj)
            })
            .collect())
    }
}

/// Response of the quadratic observables for a perturbation
/// `+sum_b u_b(t) Q_b` with time convention `exp(-i omega t)`.
/// The trace in the doubled Nambu basis carries a factor of one half.
/// Occupations are held fixed, including in the zero-frequency limit.
///
/// An optional real symmetric `interaction` (Hartree) is the curvature of
/// a supplied separable energy in the probe expectations. The induced
/// perturbation coefficients are `U * delta<Q>`; the specified reference
/// is the expansion point. The bare and coupled results use all electronic
/// states in the supplied metric. The coupled calculation requires a
/// nonnegative isospectral stability bound within numerical tolerance.
pub fn nambu_causal_response(
    reference: NambuReference<'_>,
    probes: &[NambuProbe],
    frequencies: &[Frequency],
    interaction: Option<&Matrix>,
    config: &ResponseSolverConfig,
) -> Result<NambuCausalResponse> {
    config.validate()?;
    if probes.is_empty()
        || frequencies.is_empty()
        || !reference.temperature_energy.is_finite()
        || reference.temperature_energy <= 0.0
    {
        return Err(DftError::InvalidInput(
            "collisionless Nambu response requires probes, frequencies and positive kT".into(),
        ));
    }
    let p = probes.len();
    if let Some(u) = interaction {
        if u.rows != p
            || u.cols != p
            || u.as_slice().iter().any(|x| !x.is_finite())
            || (0..p).any(|i| (0..i).any(|j| u[(i, j)] != u[(j, i)]))
        {
            return Err(DftError::InvalidInput(
                "Nambu channel interaction must be a finite, symmetric matrix in probe order"
                    .into(),
            ));
        }
    }
    let (h, s) = nambu_matrices(
        from_ref(reference.normal),
        from_ref(reference.pairing),
        from_ref(reference.overlap),
        &[reference.chemical_potential],
        0,
    )?;
    let orbitals = generalized_orbitals(&h[0], &s[0], reference.overlap_cutoff)?;
    if orbitals.energies.len() != h[0].rows() {
        return Err(DftError::InvalidInput(
            "collisionless Nambu response requires a complete positive overlap within cutoff"
                .into(),
        ));
    }
    if !orbitals.eigen_residual.is_finite()
        || !orbitals.metric_defect.is_finite()
        || orbitals.eigen_residual.max(orbitals.metric_defect) > config.acceptance()
    {
        return Err(DftError::LinearAlgebra(format!(
            "Nambu reference eigen/metric defects {:.6e}/{:.6e}",
            orbitals.eigen_residual, orbitals.metric_defect
        )));
    }
    let columns = &orbitals.coefficients;
    let columns_h = adjoint(columns)?;
    let transformed = probes
        .iter()
        .map(|probe| {
            let (q, _) = nambu_matrices(
                from_ref(&probe.normal),
                from_ref(&probe.pairing),
                from_ref(reference.overlap),
                &[0.0],
                0,
            )?;
            matmul(&columns_h, &matmul(&q[0], columns)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut transitions = Vec::new();
    for (i, &hi) in orbitals.energies.iter().enumerate() {
        for (j, &lo) in orbitals.energies[..i].iter().enumerate() {
            let gap = hi - lo;
            if !gap.is_finite() {
                return Err(DftError::InvalidInput(
                    "non-finite Nambu transition gap".into(),
                ));
            }
            // Exactly degenerate states have equal populations and no
            // collisionless weight. Keep resolved gaps, however small.
            if gap == 0.0 {
                continue;
            }
            let occupation_slope =
                occupation_divided_difference(lo, hi, 0.0, reference.temperature_energy)?;
            let vertices: Vec<_> = transformed.iter().map(|q| element(q, i, j)).collect();
            if occupation_slope != 0.0 && vertices.iter().any(|v| *v != Complex::ZERO) {
                transitions.push(Transition {
                    gap,
                    occupation_slope,
                    vertices,
                });
            }
        }
    }
    let (stability_margin, static_gram_negative_defect) = match interaction {
        Some(u) => stability(&transitions, u, config)?,
        None => (1.0, 0.0),
    };
    let mut report = NambuCausalResponse {
        bare: Vec::with_capacity(frequencies.len()),
        response: Vec::with_capacity(frequencies.len()),
        coupled_residuals: Vec::with_capacity(frequencies.len()),
        bare_residuals: Vec::with_capacity(frequencies.len()),
        modes: orbitals.energies,
        reference_eigen_residual: orbitals.eigen_residual,
        reference_metric_defect: orbitals.metric_defect,
        stability_margin,
        static_gram_negative_defect,
    };
    for &frequency in frequencies {
        let (bare, bare_residual) = bare_response(&transitions, p, frequency, config)?;
        let (response, residuals) = match interaction {
            Some(u) => couple(&bare, u, config)?,
            None => (bare.clone(), vec![0.0; p]),
        };
        report.bare.push(bare);
        report.response.push(response);
        report.bare_residuals.push(bare_residual);
        report.coupled_residuals.push(residuals);
    }
    Ok(report)
}

fn bare_response(
    transitions: &[Transition],
    p: usize,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<(BlochMatrix, f64)> {
    let z = frequency.energy()?;
    let mut bare = BlochMatrix::zeros(p, p);
    let mut residual: f64 = 0.0;
    for transition in transitions {
        let gap = Complex::from(transition.gap);
        let forward = z - gap;
        let backward = z + gap;
        if z != Complex::ZERO && forward.abs().min(backward.abs()) <= config.min_denominator {
            return Err(DftError::LinearAlgebra(format!(
                "unresolved collisionless Nambu pole at energy gap {:.15e} Hartree",
                transition.gap
            )));
        }
        // The common gap cancels at zero frequency. Keep the finite
        // occupation slope even when the resolved gap is very small.
        let slope = 0.5 * transition.occupation_slope;
        let (plus, minus) = if z == Complex::ZERO {
            (Complex::from(-slope), Complex::from(slope))
        } else {
            ((gap / forward) * slope, (gap / backward) * slope)
        };
        let weight = gap * slope;
        residual = residual
            .max((forward * plus - weight).abs())
            .max((backward * minus - weight).abs());
        for (a, &va) in transition.vertices.iter().enumerate() {
            for (b, &vb) in transition.vertices.iter().enumerate() {
                let product = va.conj() * vb;
                let value = plus * product - minus * product.conj();
                bare.re[(a, b)] += value.re;
                bare.im[(a, b)] += value.im;
            }
        }
    }
    if !residual.is_finite()
        || residual > config.acceptance()
        || bare
            .re
            .as_slice()
            .iter()
            .chain(bare.im.as_slice())
            .any(|x| !x.is_finite())
    {
        return Err(DftError::LinearAlgebra(format!(
            "collisionless Nambu spectral response failed with defect {residual:.6e}"
        )));
    }
    Ok((bare, residual))
}

fn stability(
    transitions: &[Transition],
    u: &Matrix,
    config: &ResponseSolverConfig,
) -> Result<(f64, f64)> {
    // In the real and imaginary transition coordinates the bare static
    // response is minus a Gram matrix. The full curvature is I + J^T U J
    // with J J^T = A. All non-unit eigenvalues lie in the probe space.
    let p = u.rows;
    let mut gram = Matrix::zeros(p, p);
    for transition in transitions {
        for (a, &va) in transition.vertices.iter().enumerate() {
            for (b, &vb) in transition.vertices[..=a].iter().enumerate() {
                gram[(a, b)] += transition.occupation_slope * (va.conj() * vb).re;
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            gram[(j, i)] = gram[(i, j)];
        }
    }
    if gram.as_slice().iter().any(|x| !x.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "non-finite Nambu stability Gram matrix".into(),
        ));
    }
    let (spectrum, vectors) = symmetric_eigen(&gram)?;
    let defect = (-spectrum[0]).max(0.0);
    // Only negative roundoff of this mathematically positive Gram matrix
    // is clipped, never a physical gap or response denominator.
    if defect > 64.0 * f64::EPSILON * p as f64 * spectrum[p - 1].abs() {
        return Err(DftError::LinearAlgebra(format!(
            "Nambu stability Gram matrix has negative eigenvalue {:.6e}",
            spectrum[0]
        )));
    }
    let root: Vec<_> = spectrum.iter().map(|&x| x.max(0.0).sqrt()).collect();
    let mut reduced = vectors.transpose_matmul(&u.matmul_seq(&vectors));
    for i in 0..p {
        for j in 0..=i {
            let value = reduced[(i, j)].midpoint(reduced[(j, i)]) * root[i] * root[j];
            reduced[(i, j)] = value + if i == j { 1.0 } else { 0.0 };
            reduced[(j, i)] = reduced[(i, j)];
        }
    }
    if reduced.as_slice().iter().any(|x| !x.is_finite()) {
        return Err(DftError::LinearAlgebra(
            "non-finite Nambu stability curvature".into(),
        ));
    }
    let margin = symmetric_eigen(&reduced)?.0[0].min(1.0);
    if !margin.is_finite() || margin < -config.acceptance() {
        return Err(DftError::InvalidInput(format!(
            "unstable collisionless Nambu reference: isospectral curvature {margin:.6e}"
        )));
    }
    Ok((margin, defect))
}

fn couple(
    bare: &BlochMatrix,
    u: &Matrix,
    config: &ResponseSolverConfig,
) -> Result<(BlochMatrix, Vec<f64>)> {
    let p = u.rows;
    // X = chi0 (I + U X); solve one equation for each external probe.
    let mut re = Matrix::identity(p);
    let feedback = bare.re.matmul_seq(u);
    for (x, &y) in re.as_mut_slice().iter_mut().zip(feedback.as_slice()) {
        *x -= y;
    }
    let mut im = bare.im.matmul_seq(u);
    for x in im.as_mut_slice() {
        *x = -*x;
    }
    let equation = InducedEquation(BlochMatrix { re, im });
    let mut response = BlochMatrix::zeros(p, p);
    let mut residuals = Vec::with_capacity(p);
    for j in 0..p {
        let rhs: Vec<_> = (0..p).map(|i| element(bare, i, j)).collect();
        let solution = solve(&equation, &rhs, None, config)?;
        for (i, value) in solution.amplitudes.into_iter().enumerate() {
            response.re[(i, j)] = value.re;
            response.im[(i, j)] = value.im;
        }
        residuals.push(solution.relative_residual);
    }
    Ok((response, residuals))
}
