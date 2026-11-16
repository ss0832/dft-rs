// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gamma-point common-spatial-orbital open-shell SCF on the periodic operators.
//!
//! Closed, open and virtual subspaces share one metric-orthonormal orbital set.
//! Geometry derivatives use the common-orbital Lagrange multipliers.

use crate::linalg::{Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::scf::{
    build_fields, build_scf_xc_grid, validate_options, Fields, PeriodicScfOptions,
};
use crate::periodic::scf_boundary::{
    dimensional_geometry, DimensionalScfBackend, GeometryDerivatives,
};
use crate::periodic::spinor_gradient::stress::StressQuality;
use crate::periodic::xc::PeriodicXcGrid;
use crate::periodic::Cell;
use crate::scf::diis::Diis;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod mesh;
pub use mesh::{BulkRoksGeometry, MeshRoksProblem, MeshRoksResult, MeshRoksState};

fn rotation_blocks(
    fock: [&Matrix; 2],
    n_alpha: usize,
    n_beta: usize,
    transpose_sign: f64,
) -> (Matrix, Matrix) {
    let rank = fock[0].rows;
    let class = |i| {
        if i < n_beta {
            0
        } else if i < n_alpha {
            1
        } else {
            2
        }
    };
    let occupation = |i, count| f64::from(i < count);
    let mut effective = Matrix::zeros(rank, rank);
    let mut gradient = Matrix::zeros(rank, rank);
    for p in 0..rank {
        for q in p..rank {
            let (fa, fb) = (fock[0][(p, q)], fock[1][(p, q)]);
            let value = match (class(p), class(q)) {
                (0, 1) => fb,
                (1, 2) => fa,
                _ => 0.5 * (fa + fb),
            };
            effective[(p, q)] = value;
            if p < q {
                effective[(q, p)] = transpose_sign * value;
                let na = occupation(p, n_alpha) - occupation(q, n_alpha);
                let nb = occupation(p, n_beta) - occupation(q, n_beta);
                let g = 2.0 * (na * fa + nb * fb);
                gradient[(p, q)] = g;
                gradient[(q, p)] = -transpose_sign * g;
            }
        }
    }
    (effective, gradient)
}

struct IterationResult<C, S> {
    coefficients: C,
    state: S,
    iterations: usize,
    density_change: f64,
    energy_change: f64,
}

fn iterate_common_orbitals<C, S>(
    options: &PeriodicScfOptions,
    mut coefficients: C,
    evaluate: impl Fn(&C) -> Result<S>,
    energy_residual: impl Fn(&S) -> (f64, f64),
    density_difference: impl Fn(&S, &S) -> f64,
    mut next: impl FnMut(&S) -> Result<C>,
) -> Result<IterationResult<C, S>> {
    let mut previous: Option<S> = None;
    let mut last_residual = f64::INFINITY;
    for iteration in 1..=options.max_iter {
        let state = evaluate(&coefficients)?;
        let (energy, residual) = energy_residual(&state);
        let energy_change = previous
            .as_ref()
            .map_or(f64::INFINITY, |old| (energy - energy_residual(old).0).abs());
        let density_change = previous
            .as_ref()
            .map_or(f64::INFINITY, |old| density_difference(&state, old));
        last_residual = residual;
        crate::logger::log(&format!("[pbc-roks] iter {iteration:>3} E={energy:.14} dE={energy_change:.3e} dP={density_change:.3e} residual={residual:.3e}"));
        if energy_change < options.e_tol
            && density_change < options.p_tol
            && residual < options.conv_tol_grad
        {
            return Ok(IterationResult {
                coefficients,
                state,
                iterations: iteration,
                density_change,
                energy_change,
            });
        }
        coefficients = next(&state)?;
        previous = Some(state);
    }
    Err(DftError::ScfNotConverged {
        iterations: options.max_iter,
        error: last_residual,
    })
}

/// Fresh physical fields and the constrained orbital gradient at one common C.
pub struct GammaRoksState {
    pub energy: f64,
    pub electronic_energy: f64,
    pub xc_energy: f64,
    pub nonlocal_correlation_energy: f64,
    pub density: [Matrix; 2],
    pub fock: [Matrix; 2],
    /// dE/dtheta_pq for C_p'=cos(theta) C_p+sin(theta) C_q, p<q.
    pub orbital_gradient: Matrix,
    pub orbital_residual: f64,
    pub metric_defect: f64,
    effective_fock: Matrix,
}

/// A constrained reference with measured, non-extrapolated final fields.
pub struct GammaRoksResult {
    pub state: GammaRoksState,
    pub coefficients: Matrix,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub iterations: usize,
    pub density_change: f64,
    pub energy_change: f64,
}

pub(crate) struct GammaRoksDiagnostics {
    pub particle_errors: [f64; 2],
    pub idempotency: [f64; 2],
    pub closed_subspace: f64,
}

enum RoksOperators<'a> {
    Bulk {
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        grid: Option<Box<PeriodicXcGrid>>,
    },
    Dimensional(&'a DimensionalScfBackend<'a>),
}

impl<'a> RoksOperators<'a> {
    fn bulk(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        options: &PeriodicScfOptions,
        points: &[Vec3],
    ) -> Result<Self> {
        let grid = build_scf_xc_grid(cell, functional, options, points)?.map(Box::new);
        Ok(Self::Bulk {
            cell,
            functional,
            grid,
        })
    }

    fn fields(
        &self,
        options: &PeriodicScfOptions,
        points: &[Vec3],
        core: &[BlochMatrix],
        density: &[Vec<BlochMatrix>; 2],
    ) -> Result<Fields> {
        match self {
            Self::Bulk {
                cell,
                functional,
                grid,
            } => build_fields(
                cell,
                functional,
                points,
                density,
                core,
                grid.as_deref(),
                options.exchange_cutoff.unwrap_or(cell.overlap_cutoff()),
                false,
            ),
            Self::Dimensional(backend) => {
                let fields = backend.evaluate(density)?;
                if !fields.coulomb.converged || fields.xc.as_ref().is_some_and(|xc| !xc.converged) {
                    return Err(DftError::InvalidInput(
                        "dimensional ROKS requires converged Coulomb and XC operators".into(),
                    ));
                }
                Ok(Fields {
                    fock: fields.fock,
                    e_elec: fields.electronic_energy,
                    e_xc: fields.semilocal_energy,
                    e_nlc: fields.xc.as_ref().map_or(0.0, |xc| xc.nonlocal_energy),
                    grid_electron_count: fields.xc.map(|xc| xc.grid_electron_count),
                })
            }
        }
    }
}

/// Immutable Gamma grid, metric and physical periodic operator inputs.
pub struct GammaRoksProblem<'a> {
    operators: RoksOperators<'a>,
    options: &'a PeriodicScfOptions,
    pub overlap: Matrix,
    pub core: Matrix,
    pub orthogonalizer: Orthogonalizer,
    pub n_alpha: usize,
    pub n_beta: usize,
    nuclear_energy: f64,
}

impl<'a> GammaRoksProblem<'a> {
    pub fn new(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        options: &'a PeriodicScfOptions,
    ) -> Result<Self> {
        if options.boundary.axes != [true; 3] {
            return Err(DftError::InvalidInput(
                "bulk ROKS requires three periodic axes; use the dimensional Gamma problem".into(),
            ));
        }
        Self::validate_options(functional, options)?;
        validate_options(options)?;
        let gamma = Vec3::zero();
        let operators = RoksOperators::bulk(cell, functional, options, &[gamma])?;
        let (n_alpha, n_beta) = cell.molecule().n_alpha_beta_for(cell.n_electrons()?)?;
        Self::from_operators(
            operators,
            options,
            BlochMatrix::from_pair(cell.overlap(gamma)),
            BlochMatrix::from_pair(cell.core_hamiltonian(gamma)),
            [n_alpha, n_beta],
            cell.nuclear_energy(),
        )
    }

    pub(crate) fn new_dimensional(
        backend: &'a DimensionalScfBackend<'a>,
        functional: &ResolvedFunctional,
        options: &'a PeriodicScfOptions,
        occupations: [usize; 2],
    ) -> Result<Self> {
        Self::validate_options(functional, options)?;
        Self::from_operators(
            RoksOperators::Dimensional(backend),
            options,
            backend.overlap()[0].clone(),
            backend.h_core()[0].clone(),
            occupations,
            backend.source_energy(),
        )
    }

    fn validate_options(
        functional: &ResolvedFunctional,
        options: &PeriodicScfOptions,
    ) -> Result<()> {
        if options.k_mesh != [1, 1, 1] {
            return Err(DftError::Unsupported {
                feature: "Gamma ROKS with a finite k mesh".into(),
                missing: "the mesh ROKS problem for this entry point".into(),
            });
        }
        Self::validate_common_options(functional, options)
    }

    fn validate_common_options(
        functional: &ResolvedFunctional,
        options: &PeriodicScfOptions,
    ) -> Result<()> {
        let supported_nonlocal = functional.required_nonlocal_kind().is_none_or(|term| {
            term.is_supplied(options.vv10.is_some(), options.pair_dispersion_from_caller)
        });
        if options.occupation != OccupationScheme::Insulator
            || options.skala.is_some()
            || !supported_nonlocal
            || !matches!(options.guess, crate::scf::Guess::Core)
            || options.level_shift != 0.0
        {
            return Err(DftError::Unsupported {
                feature: "periodic common-orbital reference outside the supported fixed-integer functional scope".into(),
                missing: "the matching thermal, initialized or additional nonlocal constrained reference".into(),
            });
        }
        if options.max_iter == 0
            || options.diis_size == 0
            || [
                options.e_tol,
                options.p_tol,
                options.conv_tol_grad,
                options.orth_threshold,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(DftError::InvalidInput(
                "Gamma ROKS requires positive finite SCF gates and iteration/history budgets"
                    .into(),
            ));
        }
        Ok(())
    }

    fn from_operators(
        operators: RoksOperators<'a>,
        options: &'a PeriodicScfOptions,
        overlap: BlochMatrix,
        core: BlochMatrix,
        [n_alpha, n_beta]: [usize; 2],
        nuclear_energy: f64,
    ) -> Result<Self> {
        if overlap
            .im
            .as_slice()
            .iter()
            .chain(core.im.as_slice())
            .any(|x| !x.is_finite() || x.abs() > 1e-12)
        {
            return Err(DftError::InvalidInput(
                "Gamma common-real-orbital reference requires real S/H".into(),
            ));
        }
        let orthogonalizer = Orthogonalizer::canonical(&overlap.re, options.orth_threshold)?;
        if n_alpha > orthogonalizer.n_kept || n_beta > n_alpha {
            return Err(DftError::InvalidInput(
                "closed/open occupations exceed the retained Gamma AO rank".into(),
            ));
        }
        Ok(Self {
            operators,
            options,
            overlap: overlap.re,
            core: core.re,
            orthogonalizer,
            n_alpha,
            n_beta,
            nuclear_energy,
        })
    }

    pub fn initial_orbitals(&self) -> Result<Matrix> {
        Ok(self.orthogonalizer.solve_roothaan(&self.core)?.1)
    }

    pub(crate) fn diagnostics(&self, state: &GammaRoksState) -> GammaRoksDiagnostics {
        let [alpha, beta] = &state.density;
        let overlap_beta = self.overlap.matmul_seq(beta);
        GammaRoksDiagnostics {
            particle_errors: [
                alpha.frobenius_dot(&self.overlap) - self.n_alpha as f64,
                beta.frobenius_dot(&self.overlap) - self.n_beta as f64,
            ],
            idempotency: [
                alpha
                    .matmul_seq(&self.overlap.matmul_seq(alpha))
                    .max_abs_diff(alpha),
                beta.matmul_seq(&overlap_beta).max_abs_diff(beta),
            ],
            closed_subspace: alpha.matmul_seq(&overlap_beta).max_abs_diff(beta),
        }
    }

    /// Evaluate both physical spin Focks and the energy on the cached grid.
    pub fn evaluate(&self, c: &Matrix) -> Result<GammaRoksState> {
        let rank = self.orthogonalizer.n_kept;
        if c.rows != self.core.rows || c.cols != rank || c.as_slice().iter().any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "Gamma ROKS common coefficients have invalid shape or values".into(),
            ));
        }
        let metric = c.transpose().matmul_seq(&self.overlap.matmul_seq(c));
        let metric_defect = metric.max_abs_diff(&Matrix::identity(rank));
        if metric_defect > 1e-10 {
            return Err(DftError::InvalidInput(format!(
                "common orbitals are not S-orthonormal: {metric_defect:.3e}"
            )));
        }
        let density: [Matrix; 2] = std::array::from_fn(|spin| {
            let occupied = if spin == 0 { self.n_alpha } else { self.n_beta };
            let weights: Vec<f64> = (0..rank)
                .map(|i| if i < occupied { 1.0 } else { 0.0 })
                .collect();
            crate::scf::occupied_density(c, &weights)
        });
        let complex: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            vec![BlochMatrix::from_pair((
                density[spin].clone(),
                Matrix::zeros(c.rows, c.rows),
            ))]
        });
        let core = BlochMatrix::from_pair((self.core.clone(), Matrix::zeros(c.rows, c.rows)));
        let fields = self
            .operators
            .fields(self.options, &[Vec3::zero()], &[core], &complex)?;
        let (physical, xc_energy, electronic_energy) = (fields.fock, fields.e_xc, fields.e_elec);
        if physical
            .iter()
            .flat_map(|channel| channel[0].re.as_slice())
            .any(|x| !x.is_finite())
            || physical
                .iter()
                .flat_map(|channel| channel[0].im.as_slice())
                .any(|x| !x.is_finite() || x.abs() > 1e-12)
        {
            return Err(DftError::InvalidInput(
                "Gamma common-real-orbital Fock has a nonzero imaginary part".into(),
            ));
        }
        let fock = [physical[0][0].re.clone(), physical[1][0].re.clone()];
        if !electronic_energy.is_finite() || !xc_energy.is_finite() {
            return Err(DftError::InvalidInput(
                "nonfinite common-orbital periodic energy".into(),
            ));
        }
        let fm: [Matrix; 2] =
            std::array::from_fn(|spin| c.transpose().matmul_seq(&fock[spin].matmul_seq(c)));
        let (effective, orbital_gradient) =
            rotation_blocks([&fm[0], &fm[1]], self.n_alpha, self.n_beta, 1.0);
        let orbital_residual = orbital_gradient
            .as_slice()
            .iter()
            .map(|g| g.abs())
            .fold(0.0, f64::max);
        let sc = self.overlap.matmul_seq(c);
        let effective_fock = sc.matmul_seq(&effective).matmul_transpose_seq(&sc);
        Ok(GammaRoksState {
            energy: electronic_energy + self.nuclear_energy,
            electronic_energy,
            xc_energy,
            nonlocal_correlation_energy: fields.e_nlc,
            density,
            fock,
            orbital_gradient,
            orbital_residual,
            metric_defect,
            effective_fock,
        })
    }

    /// Highest implemented nuclear derivative order in the current AO domain.
    /// Moving rank truncation needs its own projector derivative; do not silently
    /// use the full-rank Lagrangian formula for a geometry-dependent subspace.
    pub fn max_nuclear_response_order(&self) -> usize {
        let available = match self.operators {
            RoksOperators::Bulk { .. } => true,
            RoksOperators::Dimensional(backend) => backend.field_free_nuclear_frame().is_ok(),
        };
        usize::from(available && self.orthogonalizer.n_kept == self.core.rows)
    }

    /// Shared physical contraction data after a fresh constrained-stationarity check.
    fn with_stationary_density<T>(
        &self,
        reference: &GammaRoksResult,
        apply: impl FnOnce(
            &GammaRoksState,
            &crate::periodic::stationary_density::StationaryDensity<'_>,
        ) -> Result<T>,
    ) -> Result<T> {
        if self.max_nuclear_response_order() == 0 {
            return Err(DftError::Unsupported {
                feature: "periodic ROKS nuclear/strain derivative with truncated overlap rank"
                    .into(),
                missing: "the geometry-dependent discarded-space projector derivative".into(),
            });
        }
        let fresh = self.evaluate(&reference.coefficients)?;
        if fresh.orbital_residual >= self.options.conv_tol_grad {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual: fresh.orbital_residual,
            });
        }
        let density: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            vec![BlochMatrix::from_pair((
                fresh.density[spin].clone(),
                Matrix::zeros(self.core.rows, self.core.cols),
            ))]
        });
        // At common-orbital stationarity, closed-virtual alpha/beta multipliers
        // cancel. The total Pulay matrix equals sum_s P_s F_s P_s; it does not
        // require each spin to be an independently stationary unrestricted state.
        let weighted: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            let w = fresh.density[spin]
                .matmul_seq(&fresh.fock[spin])
                .matmul_seq(&fresh.density[spin]);
            vec![BlochMatrix::from_pair((
                w,
                Matrix::zeros(self.core.rows, self.core.cols),
            ))]
        });
        let gamma = [Vec3::zero()];
        let data = crate::periodic::stationary_density::StationaryDensity {
            k_points: &gamma,
            density: &density,
            energy_weighted_density: &weighted,
            distinct_spin_channels: true,
        };
        apply(&fresh, &data)
    }

    /// Analytic total energy gradient, including overlap, lattice J/K and moving XC.
    pub fn nuclear_gradient(&self, reference: &GammaRoksResult) -> Result<Vec<Vec3>> {
        if matches!(self.operators, RoksOperators::Dimensional(_)) {
            return self
                .dimensional_derivatives(reference, true, false)?
                .gradient
                .map(|gradient| gradient.total)
                .ok_or_else(|| {
                    DftError::LinearAlgebra("dimensional ROKS gradient was not assembled".into())
                });
        }
        let RoksOperators::Bulk {
            cell, functional, ..
        } = self.operators
        else {
            unreachable!("dimensional reference handled above");
        };
        self.with_stationary_density(reference, |_, data| {
            let electronic = crate::periodic::gradient::electronic_gradient_from_density(
                cell,
                functional,
                data,
                self.options,
            )?;
            let nuclear_force = cell.nuclear_forces();
            Ok(electronic
                .iter()
                .zip(nuclear_force)
                .map(|(g, f)| *g - f)
                .collect())
        })
    }

    /// Analytic full affine-strain derivative at fixed fractional coordinates.
    pub fn stress(
        &self,
        reference: &GammaRoksResult,
    ) -> Result<crate::periodic::stress::PeriodicStress> {
        let RoksOperators::Bulk {
            cell, functional, ..
        } = self.operators
        else {
            return Err(DftError::Unsupported {
                feature: "dimensional Gamma ROKS stress through the bulk volume API".into(),
                missing: "the active-measure report available through the common stress driver"
                    .into(),
            });
        };
        self.with_stationary_density(reference, |_, data| {
            crate::periodic::stress::analytic_stress_from_density(
                cell,
                functional,
                data,
                self.options,
            )
        })
    }

    pub(crate) fn dimensional_derivatives(
        &self,
        reference: &GammaRoksResult,
        forces: bool,
        stress: bool,
    ) -> Result<GeometryDerivatives> {
        let RoksOperators::Dimensional(backend) = self.operators else {
            return Err(DftError::InvalidInput(
                "dimensional ROKS derivatives require a dimensional reference".into(),
            ));
        };
        let frame = backend.field_free_nuclear_frame()?;
        self.with_stationary_density(reference, |fresh, data| {
            let diagnostics = self.diagnostics(fresh);
            let number_error = diagnostics
                .particle_errors
                .into_iter()
                .map(f64::abs)
                .fold(0.0, f64::max);
            let consistency = diagnostics
                .idempotency
                .into_iter()
                .fold(diagnostics.closed_subspace, f64::max);
            let fock_defect = fresh
                .fock
                .iter()
                .zip(&reference.state.fock)
                .map(|(current, accepted)| current.max_abs_diff(accepted))
                .fold(0.0, f64::max);
            dimensional_geometry(
                frame,
                data,
                fresh.xc_energy,
                StressQuality {
                    orbital: fresh.orbital_residual,
                    consistency,
                    number_error,
                    energy_defect: (fresh.energy - reference.state.energy).abs(),
                    fock_defect,
                    xc_defect: 0.0,
                },
                forces,
                stress,
            )
        })
    }

    pub fn run(&self) -> Result<GammaRoksResult> {
        let mut history = Diis::new(self.options.diis_size);
        let result = iterate_common_orbitals(
            self.options,
            self.initial_orbitals()?,
            |c| self.evaluate(c),
            |state| (state.energy, state.orbital_residual),
            |state, old| {
                state.density[0]
                    .max_abs_diff(&old.density[0])
                    .max(state.density[1].max_abs_diff(&old.density[1]))
            },
            |state| {
                let mut total = state.density[0].clone();
                for (x, y) in total
                    .as_mut_slice()
                    .iter_mut()
                    .zip(state.density[1].as_slice())
                {
                    *x += y;
                }
                history.push(
                    std::slice::from_ref(&state.effective_fock),
                    &[total],
                    &self.overlap,
                    &self.orthogonalizer.x,
                );
                let candidate = if self.options.use_diis {
                    history.extrapolate().map(|mut values| values.remove(0))
                } else {
                    None
                };
                Ok(self
                    .orthogonalizer
                    .solve_roothaan(candidate.as_ref().unwrap_or(&state.effective_fock))?
                    .1)
            },
        )?;
        Ok(GammaRoksResult {
            state: result.state,
            coefficients: result.coefficients,
            n_alpha: self.n_alpha,
            n_beta: self.n_beta,
            iterations: result.iterations,
            density_change: result.density_change,
            energy_change: result.energy_change,
        })
    }
}
