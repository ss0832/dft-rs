// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Shared constrained determinant iteration for restricted or collinear spins.
//! Each channel has its own metric complement and integer occupation. A single
//! scalar callback and a common DIIS history retain the coupled spin response.
use super::*;

pub(super) trait ScalarEvaluation {
    fn energy(&self) -> f64;
    fn focks(&self) -> Vec<&Matrix>;
}
impl ScalarEvaluation for FrozenDensityEvaluation {
    fn energy(&self) -> f64 {
        self.energy
    }
    fn focks(&self) -> Vec<&Matrix> {
        vec![&self.fock]
    }
}

pub(super) struct Channel<'a> {
    pub overlap: &'a Matrix,
    pub complement: &'a Matrix,
    pub frozen: &'a Matrix,
    pub core: &'a Matrix,
    pub occupied: usize,
    /// Two for a spin-summed restricted determinant, one for each UKS spin.
    pub occupation: f64,
}
pub(super) struct ChannelSolution {
    pub density: Matrix,
    pub orbitals: Matrix,
    pub energies: Vec<f64>,
    pub idempotency: f64,
    pub frozen_overlap: f64,
    pub particle_error: f64,
}
pub(super) struct Solution<E> {
    pub channels: Vec<ChannelSolution>,
    pub evaluation: E,
    pub history: Vec<FrozenIteration>,
    pub residual: f64,
}

fn projector(c: &Matrix, occupied: usize, occupation: f64) -> Matrix {
    let c = columns(c, &(0..occupied).collect::<Vec<_>>());
    scaled(&c.matmul_transpose_seq(&c), occupation)
}
fn projector_error(p: &Matrix, s: &Matrix, occupation: f64) -> f64 {
    let mut error = p.matmul_seq(&s.matmul_seq(p));
    add(&mut error, p, -occupation);
    largest(&error)
}
fn residual(f: &Matrix, p: &Matrix, occupation: f64) -> f64 {
    let mut error = f.matmul_seq(p);
    add(&mut error, &p.matmul_seq(f), -1.);
    largest(&error) / occupation
}
fn project_focks<E: ScalarEvaluation>(value: &E, channels: &[Channel<'_>]) -> Result<Vec<Matrix>> {
    let focks = value.focks();
    if !value.energy().is_finite() || focks.len() != channels.len() {
        return invalid("active scalar returned a nonfinite energy or wrong spin count");
    }
    channels
        .iter()
        .zip(focks)
        .map(|(space, f)| {
            if !finite_symmetric(f, space.overlap.rows) {
                return invalid("active scalar returned a nonfinite or incompatible spin Fock");
            }
            Ok(space
                .complement
                .transpose_matmul_seq(&f.matmul_seq(space.complement)))
        })
        .collect()
}

pub(super) fn solve<E: ScalarEvaluation>(
    channels: &[Channel<'_>],
    options: &ScfOptions,
    initial: Option<&[&Matrix]>,
    mut evaluate: impl FnMut(&[Matrix]) -> Result<E>,
) -> Result<Solution<E>> {
    if channels.is_empty()
        || options.max_iter == 0
        || [options.e_tol, options.p_tol, options.conv_tol_grad]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        || initial.is_some_and(|p| p.len() != channels.len())
    {
        return invalid("active scalar SCF requires matching spin channels and positive finite stopping controls");
    }
    let mut density = Vec::with_capacity(channels.len());
    for (spin, space) in channels.iter().enumerate() {
        let n = space.overlap.rows;
        let y = space.complement;
        if n == 0
            || !finite_symmetric(space.overlap, n)
            || !finite_symmetric(space.core, n)
            || y.rows != n
            || y.cols == 0
            || space.frozen.rows != n
            || space.occupied > y.cols
            || !matches!(space.occupation, 1. | 2.)
            || y.as_slice()
                .iter()
                .chain(space.frozen.as_slice())
                .any(|v| !v.is_finite())
        {
            return invalid(
                "active scalar spin space has invalid metric, dimensions or occupation",
            );
        }
        if metric_gram(y, space.overlap).max_abs_diff(&Matrix::identity(y.cols)) > 1e-9
            || metric_gram(space.frozen, space.overlap)
                .max_abs_diff(&Matrix::identity(space.frozen.cols))
                > 1e-9
            || largest(
                &space
                    .frozen
                    .transpose_matmul_seq(&space.overlap.matmul_seq(y)),
            ) > 1e-9
        {
            return invalid("active scalar requires metric-orthonormal mutually excluded active and frozen spaces");
        }
        let d = if let Some(seed) = initial.map(|p| p[spin]) {
            if !finite_symmetric(seed, n)
                || (seed.frobenius_dot(space.overlap) - space.occupation * space.occupied as f64)
                    .abs()
                    > 1e-8
                || projector_error(seed, space.overlap, space.occupation) > 1e-8
            {
                return invalid(
                    "active scalar initial spin density must be an integer-particle determinant",
                );
            }
            let sy = space.overlap.matmul_seq(y);
            let reduced = sy.transpose_matmul_seq(&seed.matmul_seq(&sy));
            if y.matmul_seq(&reduced)
                .matmul_transpose_seq(y)
                .max_abs_diff(seed)
                > 1e-8
            {
                return invalid(
                    "active scalar initial spin density lies outside its declared complement",
                );
            }
            reduced
        } else {
            let core = y.transpose_matmul_seq(&space.core.matmul_seq(y));
            let (_, c) = symmetric_eigen(&core)?;
            projector(&c, space.occupied, space.occupation)
        };
        density.push(d);
    }
    let lift = |d: &[Matrix]| {
        channels
            .iter()
            .zip(d)
            .map(|(space, p)| {
                space
                    .complement
                    .matmul_seq(p)
                    .matmul_transpose_seq(space.complement)
            })
            .collect::<Vec<_>>()
    };
    let max_residual = |f: &[Matrix], d: &[Matrix]| {
        channels
            .iter()
            .zip(f)
            .zip(d)
            .map(|((space, f), p)| residual(f, p, space.occupation))
            .fold(0., f64::max)
    };
    let mut diis = Diis::new(options.diis_size.max(2));
    let mut history = Vec::new();
    let mut previous_energy = None;
    let mut previous_density: Option<Vec<Matrix>> = None;
    for iteration in 1..=options.max_iter {
        let active = lift(&density);
        let value = evaluate(&active)?;
        let focks = project_focks(&value, channels)?;
        let error = max_residual(&focks, &density);
        let density_change = previous_density.as_ref().map(|old| {
            density
                .iter()
                .zip(old)
                .map(|(p, q)| p.max_abs_diff(q))
                .fold(0., f64::max)
        });
        let energy_change = previous_energy.map(|e: f64| (value.energy() - e).abs());
        crate::logger::log(&format!("[active-scalar] iteration={iteration} E={:.16e} residual={error:.12e} dP={density_change:?} dE={energy_change:?}",value.energy()));
        history.push(FrozenIteration {
            iteration,
            energy: value.energy(),
            projected_orbital_residual: error,
            density_change,
            energy_change,
        });
        if error < options.conv_tol_grad
            && density_change.is_some_and(|v| v < options.p_tol)
            && energy_change.is_some_and(|v| v < options.e_tol)
        {
            let eig = focks
                .iter()
                .map(symmetric_eigen)
                .collect::<Result<Vec<_>>>()?;
            let canonical = channels
                .iter()
                .zip(&eig)
                .map(|(s, (_, c))| projector(c, s.occupied, s.occupation))
                .collect::<Vec<_>>();
            let active = lift(&canonical);
            let final_value = evaluate(&active)?;
            let physical_focks = project_focks(&final_value, channels)?;
            let final_residual = max_residual(&physical_focks, &canonical);
            if final_residual < options.conv_tol_grad {
                let mut solved = Vec::with_capacity(channels.len());
                for ((space, p), (energies, c)) in channels.iter().zip(active).zip(eig) {
                    let coefficients = space.complement.matmul_seq(&c);
                    let occupied = columns(&coefficients, &(0..space.occupied).collect::<Vec<_>>());
                    let frozen_overlap = largest(
                        &space
                            .frozen
                            .transpose_matmul_seq(&space.overlap.matmul_seq(&occupied)),
                    );
                    let particle_error = (p.frobenius_dot(space.overlap)
                        - space.occupation * space.occupied as f64)
                        .abs();
                    let idempotency = projector_error(&p, space.overlap, space.occupation);
                    if frozen_overlap > 1e-9 || particle_error > 1e-9 || idempotency > 1e-9 {
                        return invalid("active scalar violates a spin electron count, Pauli exclusion or determinant idempotency");
                    }
                    solved.push(ChannelSolution {
                        density: p,
                        orbitals: coefficients,
                        energies,
                        idempotency,
                        frozen_overlap,
                        particle_error,
                    });
                }
                return Ok(Solution {
                    channels: solved,
                    evaluation: final_value,
                    history,
                    residual: final_residual,
                });
            }
            density = canonical;
            previous_energy = Some(final_value.energy());
            previous_density = None;
            diis.reset();
            continue;
        }
        let mut errors = Vec::new();
        for ((space, f), p) in channels.iter().zip(&focks).zip(&density) {
            let identity = Matrix::identity(p.rows);
            errors.extend_from_slice(
                crate::linalg::metric_commutator(
                    f,
                    &scaled(p, 1. / space.occupation),
                    &identity,
                    &identity,
                )
                .as_slice(),
            );
        }
        diis.push_with_errors(&focks, errors);
        let next_focks = if options.use_diis {
            diis.extrapolate().unwrap_or(focks)
        } else {
            focks
        };
        let next = channels
            .iter()
            .zip(&next_focks)
            .map(|(space, f)| {
                let (_, c) = symmetric_eigen(f)?;
                Ok(projector(&c, space.occupied, space.occupation))
            })
            .collect::<Result<Vec<_>>>()?;
        previous_energy = Some(value.energy());
        previous_density = Some(density);
        density = next;
    }
    for row in &history {
        crate::logger::warn(&format!(
            "[frozen-reference] iteration={} E={:.16e} residual={:.12e} dP={:?} dE={:?}",
            row.iteration,
            row.energy,
            row.projected_orbital_residual,
            row.density_change,
            row.energy_change
        ));
    }
    Err(DftError::ScfNotConverged {
        iterations: options.max_iter,
        error: history
            .last()
            .map_or(f64::INFINITY, |r| r.projected_orbital_residual),
    })
}
