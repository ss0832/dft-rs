// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed integer common-orbital stationarity on a complex periodic mesh.

use super::{iterate_common_orbitals, rotation_blocks, GammaRoksProblem, RoksOperators};
use crate::dfpt::orbitals::{adjoint, band_density, generalized_orbitals, matmul, BandOrbitals};
use crate::linalg::Orthogonalizer;
use crate::math::Vec3;
use crate::periodic::scf::{
    collinear_time_reversal_diagnostics, commutator_error, conjugate_mesh_index,
    time_reversal_symmetrize, validate_options, PeriodicDiis, PeriodicScfOptions,
};
use crate::periodic::scf_boundary::{
    dimensional_geometry, DimensionalScfBackend, GeometryDerivatives,
};
use crate::periodic::spinor_gradient::stress::StressQuality;
use crate::periodic::stationary_density::StationaryDensity;
use crate::periodic::{BlochMatrix, Cell};
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub struct MeshRoksState {
    pub energy: f64,
    pub electronic_energy: f64,
    pub xc_energy: f64,
    pub nonlocal_correlation_energy: f64,
    pub density: [Vec<BlochMatrix>; 2],
    pub fock: [Vec<BlochMatrix>; 2],
    /// Complex anti-Hermitian common-orbital gradient before mesh weighting.
    pub orbital_gradient: Vec<BlochMatrix>,
    pub orbital_residual: f64,
    pub metric_defect: f64,
    effective_fock: Vec<BlochMatrix>,
}

pub struct MeshRoksResult {
    pub state: MeshRoksState,
    pub orbitals: Vec<BandOrbitals>,
    pub iterations: usize,
    pub density_change: f64,
    pub energy_change: f64,
    pub particle_number_error: [f64; 2],
    pub idempotency_defect: [f64; 2],
    pub closed_subspace_defect: f64,
    pub conjugation_defect: [f64; 2],
}

pub struct MeshRoksProblem<'a> {
    operators: RoksOperators<'a>,
    options: &'a PeriodicScfOptions,
    pub points: Vec<Vec3>,
    pub overlap: Vec<BlochMatrix>,
    core: Vec<BlochMatrix>,
    orthogonalizers: Vec<Orthogonalizer>,
    pub occupations: [usize; 2],
    source_energy: f64,
}

pub type BulkRoksGeometry = (
    Option<Vec<Vec3>>,
    Option<crate::periodic::stress::PeriodicStress>,
);

fn defect(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}

impl<'a> MeshRoksProblem<'a> {
    pub fn new(
        cell: &'a Cell,
        functional: &'a ResolvedFunctional,
        options: &'a PeriodicScfOptions,
    ) -> Result<Self> {
        GammaRoksProblem::validate_common_options(functional, options)?;
        validate_options(options)?;
        let points = cell.k_points(options.k_mesh);
        let overlap = points
            .iter()
            .map(|&k| BlochMatrix::from_pair(cell.overlap(k)))
            .collect();
        let core = points
            .iter()
            .map(|&k| BlochMatrix::from_pair(cell.core_hamiltonian(k)))
            .collect();
        let (alpha, beta) = cell.molecule().n_alpha_beta_for(cell.n_electrons()?)?;
        Self::from_operators(
            RoksOperators::bulk(cell, functional, options, &points)?,
            options,
            points,
            overlap,
            core,
            [alpha, beta],
            cell.nuclear_energy(),
        )
    }

    pub(crate) fn new_dimensional(
        backend: &'a DimensionalScfBackend<'a>,
        functional: &ResolvedFunctional,
        options: &'a PeriodicScfOptions,
        occupations: [usize; 2],
    ) -> Result<Self> {
        GammaRoksProblem::validate_common_options(functional, options)?;
        Self::from_operators(
            RoksOperators::Dimensional(backend),
            options,
            backend.k_points().to_vec(),
            backend.overlap().to_vec(),
            backend.h_core().to_vec(),
            occupations,
            backend.source_energy(),
        )
    }

    fn from_operators(
        operators: RoksOperators<'a>,
        options: &'a PeriodicScfOptions,
        points: Vec<Vec3>,
        overlap: Vec<BlochMatrix>,
        core: Vec<BlochMatrix>,
        occupations: [usize; 2],
        source_energy: f64,
    ) -> Result<Self> {
        let orthogonalizers = overlap
            .iter()
            .map(|s| Orthogonalizer::canonical(&s.real_embedding(), options.orth_threshold))
            .collect::<Result<Vec<_>>>()?;
        if occupations[1] > occupations[0]
            || orthogonalizers
                .iter()
                .any(|orth| orth.n_kept % 2 != 0 || orth.n_kept / 2 < occupations[0])
        {
            return Err(DftError::InvalidInput(
                "closed/open occupations exceed a retained common-orbital mesh rank".into(),
            ));
        }
        Ok(Self {
            operators,
            options,
            points,
            overlap,
            core,
            orthogonalizers,
            occupations,
            source_energy,
        })
    }

    fn solve(&self, fock: &[BlochMatrix]) -> Result<Vec<BandOrbitals>> {
        let mut fock = fock.to_vec();
        time_reversal_symmetrize(&mut fock, self.options.k_mesh);
        let mut orbitals: Vec<BandOrbitals> = Vec::with_capacity(fock.len());
        for (k, matrix) in fock.iter().enumerate() {
            let partner = conjugate_mesh_index(k, self.options.k_mesh);
            let bands = if partner < k {
                let mut conjugate = orbitals[partner].clone();
                for value in conjugate.coefficients.im.as_mut_slice() {
                    *value = -*value;
                }
                conjugate
            } else {
                generalized_orbitals(matrix, &self.overlap[k], self.options.orth_threshold)?
            };
            orbitals.push(bands);
        }
        Ok(orbitals)
    }

    pub fn evaluate(&self, orbitals: &[BandOrbitals]) -> Result<MeshRoksState> {
        if orbitals.len() != self.points.len()
            || orbitals.iter().zip(&self.orthogonalizers).any(|(c, orth)| {
                c.coefficients.cols() != orth.n_kept / 2
                    || c.energies.len() != c.coefficients.cols()
            })
        {
            return Err(DftError::InvalidInput(
                "common orbitals must span the retained mesh ranks".into(),
            ));
        }
        let mut density: [Vec<BlochMatrix>; 2] =
            std::array::from_fn(|_| Vec::with_capacity(orbitals.len()));
        let mut metric_defect = 0.0_f64;
        for (band, s) in orbitals.iter().zip(&self.overlap) {
            let c = &band.coefficients;
            let metric = matmul(&adjoint(c)?, &matmul(s, c)?)?;
            metric_defect = metric.im.as_slice().iter().map(|x| x.abs()).fold(
                metric_defect.max(
                    metric
                        .re
                        .max_abs_diff(&crate::linalg::Matrix::identity(c.cols())),
                ),
                f64::max,
            );
            for (channel, count) in density.iter_mut().zip(self.occupations) {
                let weights: Vec<_> = (0..c.cols()).map(|i| f64::from(i < count)).collect();
                channel.push(band_density(band, &weights)?);
            }
        }
        if metric_defect > 1e-10 {
            return Err(DftError::InvalidInput(format!(
                "common mesh orbitals are not S-orthonormal: {metric_defect:.3e}"
            )));
        }
        let fields = self
            .operators
            .fields(self.options, &self.points, &self.core, &density)?;
        let mut orbital_gradient = Vec::with_capacity(orbitals.len());
        let mut effective_fock = Vec::with_capacity(orbitals.len());
        let mut orbital_residual = 0.0_f64;
        for (k, band) in orbitals.iter().enumerate() {
            let c = &band.coefficients;
            let ch = adjoint(c)?;
            let fa = matmul(&ch, &matmul(&fields.fock[0][k], c)?)?;
            let fb = matmul(&ch, &matmul(&fields.fock[1][k], c)?)?;
            let [alpha, beta] = self.occupations;
            let (re, grad_re) = rotation_blocks([&fa.re, &fb.re], alpha, beta, 1.0);
            let (im, grad_im) = rotation_blocks([&fa.im, &fb.im], alpha, beta, -1.0);
            orbital_residual = grad_re
                .as_slice()
                .iter()
                .zip(grad_im.as_slice())
                .map(|(&a, &b)| a.hypot(b))
                .fold(orbital_residual, f64::max);
            orbital_gradient.push(BlochMatrix {
                re: grad_re,
                im: grad_im,
            });
            let sc = matmul(&self.overlap[k], c)?;
            let mut effective = matmul(&sc, &matmul(&BlochMatrix { re, im }, &adjoint(&sc)?)?)?;
            effective.make_hermitian();
            effective_fock.push(effective);
        }
        let energy = self.source_energy + fields.e_elec;
        if !energy.is_finite() || !fields.e_xc.is_finite() {
            return Err(DftError::InvalidInput(
                "nonfinite common-orbital mesh energy".into(),
            ));
        }
        Ok(MeshRoksState {
            energy,
            electronic_energy: fields.e_elec,
            xc_energy: fields.e_xc,
            nonlocal_correlation_energy: fields.e_nlc,
            density,
            fock: fields.fock,
            orbital_gradient,
            orbital_residual,
            metric_defect,
            effective_fock,
        })
    }

    pub fn initial_orbitals(&self) -> Result<Vec<BandOrbitals>> {
        self.solve(&self.core)
    }

    pub fn run(&self) -> Result<MeshRoksResult> {
        let mut history = PeriodicDiis::new(self.options.diis_size);
        let result = iterate_common_orbitals(
            self.options,
            self.initial_orbitals()?,
            |c| self.evaluate(c),
            |state| (state.energy, state.orbital_residual),
            |state, old| {
                state
                    .density
                    .iter()
                    .flatten()
                    .zip(old.density.iter().flatten())
                    .map(|(a, b)| defect(a, b))
                    .fold(0.0, f64::max)
            },
            |state| {
                if self.options.use_diis {
                    let mut error = Vec::new();
                    for (k, fock) in state.effective_fock.iter().enumerate() {
                        let mut total = state.density[0][k].clone();
                        total.add_scaled(&state.density[1][k], 1.0);
                        error.extend(commutator_error(
                            fock,
                            &total,
                            &self.overlap[k],
                            &self.orthogonalizers[k],
                        ));
                    }
                    history.push(state.effective_fock.clone(), error);
                    if let Some(extrapolated) = history.extrapolate() {
                        return self.solve(&extrapolated);
                    }
                }
                self.solve(&state.effective_fock)
            },
        )?;
        let state = result.state;
        let (number, idempotency, closed_subspace) = self.diagnostics(&state)?;
        let tr =
            collinear_time_reversal_diagnostics(&state.density, &state.fock, self.options.k_mesh)?;
        let conjugation_defect = [
            tr.orbital_density_residual.into_iter().fold(0.0, f64::max),
            tr.orbital_fock_residual.into_iter().fold(0.0, f64::max),
        ];
        if number
            .into_iter()
            .chain(idempotency)
            .chain([closed_subspace])
            .chain(conjugation_defect)
            .any(|value| !value.is_finite() || value > 1e-9)
        {
            return Err(DftError::InvalidInput(
                "common mesh reference violates occupation or conjugation constraints".into(),
            ));
        }
        Ok(MeshRoksResult {
            state,
            orbitals: result.coefficients,
            iterations: result.iterations,
            density_change: result.density_change,
            energy_change: result.energy_change,
            particle_number_error: number,
            idempotency_defect: idempotency,
            closed_subspace_defect: closed_subspace,
            conjugation_defect,
        })
    }

    fn diagnostics(&self, state: &MeshRoksState) -> Result<([f64; 2], [f64; 2], f64)> {
        let mut number = [0.0_f64; 2];
        let mut idempotency = [0.0_f64; 2];
        let mut closed_subspace = 0.0_f64;
        for (k, s) in self.overlap.iter().enumerate() {
            let [pa, pb] = state.density.each_ref().map(|p| &p[k]);
            let [sa, sb] = [matmul(s, pa)?, matmul(s, pb)?];
            for (spin, (p, sp)) in [(pa, &sa), (pb, &sb)].into_iter().enumerate() {
                number[spin] =
                    number[spin].max((p.trace_product(s) - self.occupations[spin] as f64).abs());
                idempotency[spin] = idempotency[spin].max(defect(&matmul(p, sp)?, p));
            }
            closed_subspace = closed_subspace.max(defect(&matmul(pa, &sb)?, pb));
        }
        Ok((number, idempotency, closed_subspace))
    }

    pub fn supports_geometry(&self) -> bool {
        let rank = self
            .overlap
            .iter()
            .zip(&self.orthogonalizers)
            .all(|(s, o)| o.n_kept / 2 == s.rows());
        rank && match self.operators {
            RoksOperators::Bulk { .. } => true,
            RoksOperators::Dimensional(backend) => backend.field_free_nuclear_frame().is_ok(),
        }
    }

    fn stationary<T>(
        &self,
        result: &MeshRoksResult,
        apply: impl FnOnce(&StationaryDensity<'_>, &MeshRoksState) -> Result<T>,
    ) -> Result<T> {
        if !self.supports_geometry() {
            return Err(DftError::Unsupported {
                feature: "mesh ROKS geometry derivative with discarded overlap or orbital fields"
                    .into(),
                missing: "their moving projector or gauge derivative".into(),
            });
        }
        let fresh = self.evaluate(&result.orbitals)?;
        if fresh.orbital_residual >= self.options.conv_tol_grad {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual: fresh.orbital_residual,
            });
        }
        let [alpha, beta] = std::array::from_fn(|spin| {
            fresh.density[spin]
                .iter()
                .zip(&fresh.fock[spin])
                .map(|(p, f)| {
                    let mut weighted = matmul(p, &matmul(f, p)?)?;
                    weighted.make_hermitian();
                    Ok(weighted)
                })
                .collect::<Result<Vec<_>>>()
        });
        let weighted = [alpha?, beta?];
        apply(
            &StationaryDensity {
                k_points: &self.points,
                density: &fresh.density,
                energy_weighted_density: &weighted,
                distinct_spin_channels: true,
            },
            &fresh,
        )
    }

    pub fn bulk_geometry(
        &self,
        result: &MeshRoksResult,
        forces: bool,
        stress: bool,
    ) -> Result<BulkRoksGeometry> {
        let RoksOperators::Bulk {
            cell, functional, ..
        } = self.operators
        else {
            return Err(DftError::InvalidInput(
                "bulk ROKS geometry requires a bulk reference".into(),
            ));
        };
        self.stationary(result, |data, _| {
            let gradient = forces
                .then(|| {
                    let electronic = crate::periodic::gradient::electronic_gradient_from_density(
                        cell,
                        functional,
                        data,
                        self.options,
                    )?;
                    Ok::<_, DftError>(
                        electronic
                            .iter()
                            .zip(cell.nuclear_forces())
                            .map(|(g, f)| *g - f)
                            .collect(),
                    )
                })
                .transpose()?;
            let stress = stress
                .then(|| {
                    crate::periodic::stress::analytic_stress_from_density(
                        cell,
                        functional,
                        data,
                        self.options,
                    )
                })
                .transpose()?;
            Ok((gradient, stress))
        })
    }

    pub(crate) fn dimensional_derivatives(
        &self,
        result: &MeshRoksResult,
        forces: bool,
        stress: bool,
    ) -> Result<GeometryDerivatives> {
        let RoksOperators::Dimensional(backend) = self.operators else {
            return Err(DftError::InvalidInput(
                "dimensional ROKS geometry requires dimensional operators".into(),
            ));
        };
        let frame = backend.field_free_nuclear_frame()?;
        self.stationary(result, |data, fresh| {
            let (number, idempotency, closed) = self.diagnostics(fresh)?;
            let fock_defect = fresh
                .fock
                .iter()
                .flatten()
                .zip(result.state.fock.iter().flatten())
                .map(|(a, b)| defect(a, b))
                .fold(0.0, f64::max);
            dimensional_geometry(
                frame,
                data,
                fresh.xc_energy,
                StressQuality {
                    orbital: fresh.orbital_residual,
                    consistency: idempotency.into_iter().fold(closed, f64::max),
                    number_error: number.into_iter().fold(0.0, f64::max),
                    energy_defect: (fresh.energy - result.state.energy).abs(),
                    fock_defect,
                    xc_defect: 0.0,
                },
                forces,
                stress,
            )
        })
    }
}
