// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static response to covariant operators that enter the Dirac core while
//! the four-component AO basis and its interaction integrals stay fixed.

use super::super::matrix::{congruence, max_abs, product};
use super::{
    adjoint, matmul, x2c_taylor, BlochMatrix, Complex, DftError, MolecularTwoComponent,
    RelativisticSpin, ResponseSolverConfig, Result, SpinorScfReport,
};
use crate::integrals::dirac::{nuclear_magnetic_dipole_vertices, GaussianMagnetization};

/// First and second derivatives of the stationary electronic energy.
/// The order of both tensor indices is the supplied covariant source order.
pub struct DiracCoreSourceResponse {
    pub first: Vec<f64>,
    pub second: Vec<Vec<f64>>,
    /// Energy Hessian at fixed electronic density coefficients in the
    /// smooth electronic-sector chart. This includes sector motion.
    pub explicit: Vec<Vec<f64>>,
    pub orbital_relaxation: Vec<Vec<f64>>,
    pub residuals: Vec<f64>,
    pub reciprocity_defect: f64,
    pub projector_metric_defect: f64,
    pub projector_residual: f64,
    pub reference_fock_defect: f64,
}

pub(super) struct SourcePath {
    pub(super) density: Vec<BlochMatrix>,
    pub(super) embedding: Vec<BlochMatrix>,
    pub(super) metric_defect: f64,
    pub(super) projector_residual: f64,
}

impl MolecularTwoComponent<'_> {
    /// Finite-Gaussian nuclear moment sources in the reference AO basis.
    /// A source amplitude is in Hartree per atomic unit of magnetic field.
    pub fn nuclear_magnetic_sources(
        &self,
        nucleus: GaussianMagnetization<f64>,
    ) -> Result<[BlochMatrix; 3]> {
        if self.dirac_core.is_none() {
            return Err(DftError::Unsupported {
                feature: "Dirac nuclear magnetic operator".into(),
                missing: "a four-component nuclear-core reference".into(),
            });
        }
        let field = self.magnetic.map_or([0.0; 3], |m| m.field);
        let phase = self.magnetic.map_or(0, |m| m.phase_order);
        let integrals = nuclear_magnetic_dipole_vertices(
            &self.shells,
            nucleus,
            self.options.speed_of_light,
            field,
            0,
            phase,
        )?;
        Ok(integrals
            .map(|[re, im]| BlochMatrix::from_pair((super::from_block(re), super::from_block(im)))))
    }

    pub(super) fn positive_sector_path(
        &self,
        core: &[BlochMatrix],
        metric: &[BlochMatrix],
        density: &BlochMatrix,
    ) -> Result<SourcePath> {
        let c = self.options.speed_of_light;
        // X2C is a smooth positive-sector coordinate chart here.
        // It preserves and differentiates all four-component vertices.
        let mapped = x2c_taylor(
            core,
            metric,
            2,
            -c.powi(2),
            self.options.metric_cutoff,
            RelativisticSpin::SpinOrbit,
        )?;
        let mapped_basis = mapped.embedding();
        // Align the chart at t=0 to the actual SCF basis using one
        // constant unitary, so KS orbital stationarity uses the
        // unmodified converged density and projected kernel.
        let rotation = matmul(
            &adjoint(&mapped_basis[0])?,
            &matmul(&metric[0], &self.map.embedding()[0])?,
        )?;
        let embedding = mapped_basis
            .iter()
            .map(|b| matmul(b, &rotation))
            .collect::<Result<Vec<_>>>()?;
        let bt = embedding.iter().map(adjoint).collect::<Result<Vec<_>>>()?;
        let density = product(
            &product(&embedding, std::slice::from_ref(density), 2)?,
            &bt,
            2,
        )?;
        Ok(SourcePath {
            density,
            embedding,
            metric_defect: mapped.metric_defects.iter().copied().fold(0.0, f64::max),
            projector_residual: mapped
                .decoupling_defects
                .iter()
                .copied()
                .fold(0.0, f64::max),
        })
    }

    /// The sources enter the operator used to regenerate the positive
    /// sector. Differentiation includes its negative-energy complement
    /// and CPKS relaxation inside the positive sector. The Gaussian AO,
    /// grid and interaction operators are held fixed with respect to each
    /// supplied source. No finite displacement is made.
    pub fn dirac_core_source_response(
        &self,
        state: &SpinorScfReport,
        sources: &[BlochMatrix],
        config: &ResponseSolverConfig,
    ) -> Result<DiracCoreSourceResponse> {
        let Some([core, metric]) = &self.dirac_core else {
            return Err(DftError::Unsupported {
                feature: "moving Dirac nuclear-core source response".into(),
                missing: "the exact four-component nuclear-core sector".into(),
            });
        };
        if self.options.electric_field != [0.0; 3] || self.additional_electric.is_some() {
            return Err(DftError::Unsupported {
                feature: "quantum-source derivatives in a clamped external electric field".into(),
                missing: "the full four-component energy form of the additional electric operator"
                    .into(),
            });
        }
        let [density] = state.density.as_slice() else {
            return Err(DftError::InvalidInput(
                "Dirac response needs one stationary density".into(),
            ));
        };
        if sources.is_empty() || !state.converged {
            return Err(DftError::InvalidInput(
                "Dirac response needs sources and a converged reference".into(),
            ));
        }
        for source in sources {
            super::validate_hermitian(std::slice::from_ref(source), core.rows(), 0)?;
        }
        let full_density = self.map.lift_density(density)?;
        let mut fock = core.clone();
        let pair = self.pair(&full_density)?;
        fock.add_scaled(&pair.operator, 1.0);
        let angular = if let Some(grid) = &self.semilocal {
            let (_, operator, _, diagnostics) = grid.semilocal(
                self.functional,
                &full_density,
                self.options.xc_angular_tolerance,
            )?;
            if !diagnostics.converged {
                return Err(DftError::LinearAlgebra(
                    "Dirac-source XC reference angular rule is unresolved".into(),
                ));
            }
            fock.add_scaled(&operator, 1.0);
            diagnostics.angular_order
        } else {
            0
        };
        if let Some((grid, parameters)) = &self.nonlocal {
            let (_, operator) = grid.vv10(&full_density, None, *parameters)?;
            fock.add_scaled(&operator, 1.0);
        }
        let mut reference = self.project(&fock)?;
        reference.add_scaled(&state.fields.fock[0], -1.0);
        let reference_fock_defect = max_abs(&reference);
        if reference_fock_defect > 1e-9 {
            return Err(DftError::InvalidInput(format!(
                "Dirac source response requires the matching Fock, defect {reference_fock_defect:.9e} Hartree"
            )));
        }
        let path = |source: &BlochMatrix| {
            self.positive_sector_path(
                &[core.clone(), source.clone()],
                std::slice::from_ref(metric),
                density,
            )
        };
        let single = sources.iter().map(path).collect::<Result<Vec<_>>>()?;
        let mut projector_metric_defect =
            single.iter().map(|p| p.metric_defect).fold(0.0, f64::max);
        let mut projector_residual = single
            .iter()
            .map(|p| p.projector_residual)
            .fold(0.0, f64::max);
        let mut first = Vec::with_capacity(sources.len());
        let mut vertices = Vec::with_capacity(sources.len());
        let mut kernel = Vec::with_capacity(sources.len());
        for (source, d) in sources.iter().zip(&single) {
            let op = self.full_density_kernel(&full_density, &d.density[1], angular)?;
            let mut f1 = op.clone();
            f1.add_scaled(source, 1.0);
            let mut source_series = congruence(&d.embedding, &[fock.clone(), f1], 1)?;
            vertices.push(source_series.remove(1));
            first.push(full_density.trace_product(source) + d.density[1].trace_product(&fock));
            kernel.push(op);
        }
        let mut explicit = vec![vec![0.0; sources.len()]; sources.len()];
        for (i, left) in sources.iter().enumerate() {
            for (j, right) in sources.iter().enumerate().take(i + 1) {
                let mut sum = left.clone();
                sum.add_scaled(right, 1.0);
                let mixed = if i == j {
                    let mut d = single[i].density[2].clone();
                    d.scale(2.0);
                    d
                } else {
                    let both = path(&sum)?;
                    projector_metric_defect = projector_metric_defect.max(both.metric_defect);
                    projector_residual = projector_residual.max(both.projector_residual);
                    let mut d = both.density[2].clone();
                    d.add_scaled(&single[i].density[2], -1.0);
                    d.add_scaled(&single[j].density[2], -1.0);
                    d
                };
                // Each density coefficient is physically Hermitian by
                // construction. The same scalar contraction is shared by
                // both orderings of the two static sources.
                let raw = mixed.hermiticity_error();
                if raw > 1e-10 {
                    return Err(DftError::LinearAlgebra(format!(
                        "mixed Dirac source density is not Hermitian, raw defect {raw:.6e}"
                    )));
                }
                let pi = &single[i].density[1];
                let pj = &single[j].density[1];
                let core =
                    mixed.trace_product(&fock) + pi.trace_product(right) + pj.trace_product(left);
                explicit[i][j] = core + pi.trace_product(&kernel[j]);
                explicit[j][i] = core + pj.trace_product(&kernel[i]);
            }
        }
        let response = self
            .operator_response(state, &vertices, &[Complex::ZERO], config)?
            .remove(0);
        let mut second = explicit.clone();
        let mut orbital_relaxation = second.clone();
        for i in 0..sources.len() {
            for j in 0..sources.len() {
                let value = response.susceptibility[i][j];
                if value.im.abs() > 1e-10 {
                    return Err(DftError::LinearAlgebra(
                        "static Dirac source response has a nonzero imaginary coefficient".into(),
                    ));
                }
                orbital_relaxation[i][j] = value.re;
                second[i][j] += value.re;
            }
        }
        let mut reciprocity_defect = 0.0_f64;
        for (i, row) in second.iter().enumerate() {
            for (j, &value) in row.iter().enumerate().take(i) {
                reciprocity_defect = reciprocity_defect.max((value - second[j][i]).abs());
            }
        }
        Ok(DiracCoreSourceResponse {
            first,
            second,
            explicit,
            orbital_relaxation,
            residuals: response.residuals,
            reciprocity_defect,
            projector_metric_defect,
            projector_residual,
            reference_fock_defect,
        })
    }
}
