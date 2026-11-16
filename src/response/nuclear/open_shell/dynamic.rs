// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Time-dependent constrained common-orbital response. The Berry metric is
//! the total occupation difference (1, 2, 1) of closed-open, closed-virtual
//! and open-virtual rotations; it must not be replaced by the UKS metric.

use super::*;
use crate::response::spinor::SpinorPolarizability;
use crate::response::Frequency;

struct Paired<'a> {
    rotation: RotationOperator<'a>,
    metric: Vec<f64>,
    frequency: Complex,
}
impl LinearOperator for Paired<'_> {
    fn dimension(&self) -> usize {
        2 * self.metric.len()
    }
    fn structure(&self) -> OperatorStructure {
        if self.frequency.im == 0.0 {
            OperatorStructure::Hermitian
        } else {
            OperatorStructure::General
        }
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        let n = self.metric.len();
        if x.len() != 2 * n {
            return Err(DftError::InvalidInput(
                "invalid common-orbital paired response dimension".into(),
            ));
        }
        // Static electric/magnetic sources occupy only one parity block.
        // Its exactly zero partner has exactly zero image; avoid evaluating
        // all Coulomb/exchange kernels for that absent density response.
        let image = |part: &[Complex], imaginary| {
            if part.iter().all(|z| *z == Complex::ZERO) {
                Ok(vec![Complex::ZERO; n])
            } else {
                krylov::apply_real_linear(part, |u| self.rotation.image_parity(u, imaginary))
            }
        };
        let mut even = image(&x[..n], false)?;
        let odd = image(&x[n..], true)?;
        even.extend(odd);
        for k in 0..n {
            even[k] -= self.frequency * x[n + k] * self.metric[k];
            even[n + k] -= self.frequency * x[k] * self.metric[k];
        }
        Ok(even)
    }
}

pub struct OpenShellDynamicPolarizability {
    pub response: SpinorPolarizability,
    /// Residual of sum_s [F_s,D_s] in the constrained rotation space.
    pub reference_stationarity: f64,
    pub metric_defect: f64,
}

/// Adiabatic constrained ROKS/ROHF susceptibility at static, retarded or
/// imaginary frequency, with the same generic Coulomb/exchange/XC provider
/// as the stationary energy. All nine tensor elements are contracted from
/// independent Cartesian solves, without copying by reciprocity.
pub fn open_shell_dynamic_polarizability(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<OpenShellDynamicPolarizability> {
    let sources = crate::integrals::dipole_matrices(mb, crate::math::Vec3::zero()).map(|d| {
        let c = crate::integrals::london::ComplexMatrix {
            re: d,
            im: Matrix::zeros(mb.nao_sph, mb.nao_sph),
        };
        [c.clone(), c]
    });
    let result = open_shell_operator_response(
        mol, mb, functional, grid, scf, orbitals, opts, &sources, frequency, config,
    )?;
    Ok(OpenShellDynamicPolarizability {
        response: SpinorPolarizability {
            frequency: result.frequency,
            alpha: std::array::from_fn(|a| std::array::from_fn(|b| -result.susceptibility[a][b])),
            residuals: result.residuals.try_into().expect("three electric sources"),
            iterations: result
                .iterations
                .try_into()
                .expect("three electric sources"),
        },
        reference_stationarity: result.reference_stationarity,
        metric_defect: result.metric_defect,
    })
}

pub struct OpenShellOperatorResponse {
    pub frequency: Complex,
    /// d <H_a> / d lambda_b, without explicit contact derivatives.
    pub susceptibility: Vec<Vec<Complex>>,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
    pub reference_stationarity: f64,
    pub metric_defect: f64,
}

/// General spin-diagonal complex one-body response on the common-orbital
/// manifold. Both real/even and imaginary/odd rotations are retained. This
/// supplies longitudinal spin and orbital magnetic response without replacing
/// the constrained ROKS Hessian by UKS. Transverse spin-adapted response is a
/// separate space, rather than a repeated longitudinal result.
pub fn open_shell_operator_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    sources: &[[crate::integrals::london::ComplexMatrix; 2]],
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<OpenShellOperatorResponse> {
    open_shell_cross_operator_response(
        mol, mb, functional, grid, scf, orbitals, opts, sources, sources, frequency, config,
    )
}

/// Solve only the perturbing sources and contract independently supplied
/// observables. Rectangular mixed responses do not require solving unused
/// observable-source columns or invoking reciprocal/symmetry replacements.
pub(crate) fn open_shell_cross_operator_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    orbitals: &Matrix,
    opts: &ScfOptions,
    sources: &[[crate::integrals::london::ComplexMatrix; 2]],
    observables: &[[crate::integrals::london::ComplexMatrix; 2]],
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<OpenShellOperatorResponse> {
    let z = frequency.energy()?;
    config.validate()?;
    if !scf.converged
        || scf.n_alpha < scf.n_beta
        || scf.n_alpha > orbitals.cols
        || orbitals.rows != mb.nao_sph
        || !scf.has_integer_occupations()
        || scf.electronic_temperature != 0.0
    {
        return Err(DftError::InvalidInput("common-orbital response requires a converged integer restricted open-shell reference and its common orbitals".into()));
    }
    let classes = Classes {
        closed: scf.n_beta,
        open: scf.n_alpha - scf.n_beta,
        total: orbitals.cols,
    };
    let ct = orbitals.transpose();
    let mut metric = ct.matmul_seq(&scf.overlap).matmul_seq(orbitals);
    for i in 0..metric.rows {
        metric[(i, i)] -= 1.0;
    }
    let mut defect = max_abs(&metric);
    let mut projectors = [
        Matrix::zeros(orbitals.cols, orbitals.cols),
        Matrix::zeros(orbitals.cols, orbitals.cols),
    ];
    for s in 0..2 {
        for i in 0..[scf.n_alpha, scf.n_beta][s] {
            projectors[s][(i, i)] = 1.0;
        }
        let mut density = orbitals.matmul_seq(&projectors[s]).matmul_seq(&ct);
        add(&mut density, &scf.density[s], -1.0);
        defect = defect.max(max_abs(&density));
    }
    if defect > 1e-8 {
        return Err(DftError::InvalidInput(
            "common orbitals do not reproduce the reference metric and spin projectors".into(),
        ));
    }
    let fock = std::array::from_fn(|s| ct.matmul_seq(&scf.fock[s]).matmul_seq(orbitals));
    let mut g = Matrix::zeros(orbitals.cols, orbitals.cols);
    for s in 0..2 {
        commutator(&mut g, &fock[s], &projectors[s]);
    }
    let stationarity = classes
        .blocks(&g)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let kernel = ResponseSolver::kernel_only(mol, mb, functional, grid, scf, opts)?;
    let nc = classes.closed;
    let no = classes.open;
    let nv = classes.virtuals();
    let weights = [vec![1.0; nc * no], vec![2.0; nc * nv], vec![1.0; no * nv]].concat();
    let operator = Paired {
        rotation: RotationOperator {
            kernel: &kernel,
            orbitals,
            fock: &fock,
            classes,
        },
        metric: weights,
        frequency: z,
    };
    if sources.is_empty()
        || observables.is_empty()
        || sources.iter().chain(observables).flatten().any(|h| {
            h.re.rows != mb.nao_sph
                || h.re.cols != mb.nao_sph
                || h.im.rows != mb.nao_sph
                || h.im.cols != mb.nao_sph
                || h.re
                    .as_slice()
                    .iter()
                    .chain(h.im.as_slice())
                    .any(|v| !v.is_finite())
                || (0..h.re.rows).any(|i| {
                    (0..h.re.cols).any(|j| {
                        (h.re[(i, j)] - h.re[(j, i)]).abs() > config.acceptance()
                            || (h.im[(i, j)] + h.im[(j, i)]).abs() > config.acceptance()
                    })
                })
        })
    {
        return Err(DftError::InvalidInput(
            "common-orbital sources must be finite matching spin-diagonal Hermitian AO operators"
                .into(),
        ));
    }
    let project = |operators: &[[crate::integrals::london::ComplexMatrix; 2]]| {
        operators
            .iter()
            .map(|h| {
                std::array::from_fn::<_, 2, _>(|part| {
                    let mut g = Matrix::zeros(orbitals.cols, orbitals.cols);
                    for spin in 0..2 {
                        let m = if part == 0 { &h[spin].re } else { &h[spin].im };
                        let mo = ct.matmul_seq(m).matmul_seq(orbitals);
                        commutator(&mut g, &mo, &projectors[spin]);
                    }
                    classes.blocks(&g)
                })
            })
            .collect::<Vec<_>>()
    };
    // Bare common-orbital curvature retains the spin occupation differences:
    // beta for closed-open, alpha+beta for closed-virtual, and alpha for
    // open-virtual rotations. A UKS orbital-gap diagonal would be incorrect.
    let mut curvature = Matrix::zeros(orbitals.cols, orbitals.cols);
    for j in 0..orbitals.cols {
        for i in 0..j {
            for spin in 0..2 {
                let occupied = [scf.n_alpha, scf.n_beta][spin];
                if i < occupied && j >= occupied {
                    curvature[(j, i)] += fock[spin][(j, j)] - fock[spin][(i, i)];
                }
            }
        }
    }
    let bare = classes.blocks(&curvature);
    let mut diagonal = Vec::with_capacity(2 * bare.len());
    for (gap, weight) in bare.iter().zip(&operator.metric) {
        let forward = (Complex::from(*gap) - z * *weight).abs();
        let backward = (Complex::from(*gap) + z * *weight).abs();
        if forward.min(backward) < config.min_denominator {
            return Err(DftError::Unsupported {
                feature: "common-orbital response".into(),
                missing: "a resolved response above response_min_denominator".into(),
            });
        }
        // Positive two-sided scaling preserves Hermiticity. This changes
        // only the Krylov metric, never the physical response operator.
        diagonal.push((0.5 * forward + 0.5 * backward).max(config.preconditioner_floor));
    }
    diagonal.extend_from_within(..);
    let mut solve_config = config.clone();
    // Physical bare denominators were checked before the positive floor.
    solve_config.min_denominator = 0.0;
    let vertices = project(sources);
    let distinct_observables = (!std::ptr::eq(sources, observables)).then(|| project(observables));
    let measured = distinct_observables.as_deref().unwrap_or(&vertices);
    let n = operator.metric.len();
    let mut susceptibility = vec![vec![Complex::ZERO; sources.len()]; observables.len()];
    let mut residuals = vec![0.0; sources.len()];
    let mut iterations = vec![0; sources.len()];
    for b in 0..sources.len() {
        // The paired odd coordinate is i times the physical imaginary
        // orbital rotation. Preserve its phase on both the source and the
        // observable, including mixed time-even/time-odd susceptibilities.
        let rhs = vertices[b][0]
            .iter()
            .map(|v| Complex::from(-*v))
            .chain(vertices[b][1].iter().map(|v| Complex::new(0.0, -*v)))
            .collect::<Vec<_>>();
        let solution = krylov::solve(&operator, &rhs, Some(&diagonal), &solve_config)?;
        for a in 0..observables.len() {
            for part in 0..2 {
                for k in 0..n {
                    let vertex = if part == 0 {
                        Complex::from(2.0 * measured[a][part][k])
                    } else {
                        Complex::new(0.0, -2.0 * measured[a][part][k])
                    };
                    susceptibility[a][b] += solution.amplitudes[part * n + k] * vertex;
                }
            }
        }
        residuals[b] = solution.relative_residual;
        iterations[b] = solution.iterations;
        crate::logger::log(&format!(
            "[common-orbital-response] source {}/{} method={} iterations={} residual={:.6e}",
            b + 1,
            sources.len(),
            solution.method,
            solution.iterations,
            solution.relative_residual
        ));
    }
    Ok(OpenShellOperatorResponse {
        frequency: z,
        susceptibility,
        residuals,
        iterations,
        reference_stationarity: stationarity,
        metric_defect: defect,
    })
}
