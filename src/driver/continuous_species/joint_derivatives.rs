// SPDX-License-Identifier: Apache-2.0
//! Analytic joint nuclear/parameter derivatives in independent role coordinates.
//!
//! Each response uses one analytic Taylor path, never displaced SCF. Pair-kernel
//! lanes are actual derivatives; only this adapter divides the second lane by
//! two. Hessian polarization then uses Taylor coefficients without another half.
use super::core_jet2::{
    self, BaseFieldConvention, CorePairJet, GeometryJetSpace, PairCoefficients, PairGeometry,
    PairScalar, PointIonScope, Projectors,
};
use super::local::LocalCenter;
use super::{ContinuousSpeciesModel, Fingerprint, SpeciesFingerprints, SpeciesState};
use crate::linalg::{AccurateSum, Matrix};
use crate::math::Vec3;
use crate::response::nuclear::extra_core::{
    molecular_taylor_with_extra_core, second_order_operator_coefficients, ExternalClassicalTaylor,
    ExtraCoreCoordinates, ExtraCoreFieldConvention, ExtraCoreIdentity, ExtraCoreTaylor,
};
use crate::response::nuclear::MolecularDirection;
use crate::response::ResponseSolverConfig;
use crate::{DftError, Result};

/// Complete replacement for the default physical-minus-labelled core delta.
/// The supplied pair has both legs equal to (nuclear_direction,parameter_direction).
/// Include point ions, local/nonlocal sources, permanent field/origin terms once;
/// exclude the constant fragment E0, which the response context adds once.
pub trait CompleteJointCoreSource {
    fn pair(
        &self,
        model: &ContinuousSpeciesModel,
        state: &SpeciesState,
        nuclear_direction: &[Vec3],
        parameter_direction: &[f64],
    ) -> Result<CorePairJet>;
    /// Fixed operator/dictionary/frame recipe identity, independent of path.
    fn identity(&self) -> u64;
}

/// One analytic path, E(t)=E[0]+t E[1]+t^2 E[2].
pub struct JointDirectionalReport {
    pub energy_coefficients: [f64; 3],
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
    pub warnings: Vec<String>,
    pub source_fingerprint: u64,
    /// Already included in energy_coefficients; never add these again.
    pub external_classical_coefficients: Vec<f64>,
    pub internal_energy_reference: f64,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub metric_defects: Vec<f64>,
    pub particle_number_coefficients: Vec<[f64; 2]>,
    pub reference_energy_error: f64,
    pub reference_fock_error: f64,
}

/// Clamped-reference derivatives; ordering is atom-major xyz followed by the
/// existing dimensionless chart coordinates. No rigid-expansion pullback occurs.
pub struct JointDerivativeReport {
    pub fingerprints: SpeciesFingerprints,
    pub parameter_names: Vec<String>,
    pub parameter_units: Vec<String>,
    /// Reporting scales only: derivatives are with respect to raw chart q.
    pub parameter_scales: Vec<f64>,
    pub nuclear_gradient: Vec<Vec3>,
    pub parameter_gradient: Vec<f64>,
    pub nuclear_hessian: Matrix,
    pub nuclear_parameter_hessian: Matrix,
    pub parameter_hessian: Matrix,
    pub directional_reports: Vec<JointDirectionalReport>,
}

struct AllElectronSource;
impl CompleteJointCoreSource for AllElectronSource {
    fn identity(&self) -> u64 {
        0x61655f6a6f696e32
    }
    fn pair(
        &self,
        model: &ContinuousSpeciesModel,
        state: &SpeciesState,
        dr: &[Vec3],
        dq: &[f64],
    ) -> Result<CorePairJet> {
        if model.static_core.is_some() {
            return super::unsupported(
                "a static core without its complete analytic second-order recipe",
            );
        }
        let zero = core_jet2::seed(0., 0., 0., 0.);
        let ab = vec![Vec3::zero(); dr.len()];
        let geometry = PairGeometry::laboratory(
            &model.molecule,
            0,
            dr,
            dr,
            &ab,
            GeometryJetSpace::IndependentRoles,
        )?;
        let coefficients = PairCoefficients {
            ionic_charges: state
                .nuclear_charges
                .iter()
                .enumerate()
                .map(|(i, &z)| {
                    let dz = model
                        .chart
                        .parameters
                        .iter()
                        .zip(dq)
                        .map(|(p, d)| p.charge_direction[i] * d)
                        .sum();
                    core_jet2::seed(z, dz, dz, 0.)
                })
                .collect(),
            gaussian_charges: [zero; 3],
            body_dipoles: [[zero; 3]; 3],
            local_amplitudes: [zero; 3],
            nonlocal_upper: [zero; 36],
            internal_energy: zero,
        };
        // Exponents multiply identically zero dictionary slots in this point-only
        // call. Each actual local term uses its own immutable exponent below.
        let mut pair = core_jet2::evaluate_pair(
            &model.molecule,
            &model.basis,
            &geometry,
            &coefficients,
            [1.; 3],
            [1.; 3],
            Projectors::None,
            state.origin,
            std::array::from_fn(|i| core_jet2::seed(state.field.get(i), 0., 0., 0.)),
            PointIonScope::DifferenceFromLabelledPointIons,
            BaseFieldConvention::OriginZero,
        )?;
        for term in &model.local_terms {
            if term.spec.reference_steric_strength != 0.
                || term.spec.steric_direction.iter().any(|&x| x != 0.)
            {
                return super::unsupported("joint derivatives of a nonzero steric residual");
            }
            let (value, _) = term.amplitudes(&state.coordinates)?;
            let first = term
                .spec
                .amplitude_direction
                .iter()
                .zip(dq)
                .map(|(a, b)| a * b)
                .sum();
            let center = local_center(model, &geometry, &term.spec.center, dr)?;
            let local = core_jet2::local_gaussian_pair(
                &model.basis,
                &geometry,
                center,
                term.spec.exponent,
                core_jet2::seed(value, first, first, 0.),
            )?;
            for (target, addition) in pair.operator.iter_mut().zip(local.operator) {
                for (x, y) in target.as_mut_slice().iter_mut().zip(addition.as_slice()) {
                    *x += y;
                }
            }
        }
        Ok(pair)
    }
}

fn local_center(
    model: &ContinuousSpeciesModel,
    geometry: &PairGeometry,
    rule: &LocalCenter,
    dr: &[Vec3],
) -> Result<[PairScalar; 3]> {
    match rule {
        LocalCenter::Atom(atom) => geometry
            .positions()
            .get(*atom)
            .copied()
            .ok_or_else(|| DftError::InvalidInput("local center atom is out of range".into())),
        LocalCenter::Fixed(r) => Ok(std::array::from_fn(|i| {
            core_jet2::seed(r.get(i), 0., 0., 0.)
        })),
        LocalCenter::Attached(frame) => {
            frame.evaluate(&model.molecule)?; // Preserve the actual frame guards.
            let [a, b, c] = frame.anchors;
            let directions = std::array::from_fn(|i| dr[[a, b, c][i / 3]].get(i % 3));
            let u = crate::pseudo_sub::frame::frame_directional_second(
                model.molecule.atoms[a].position,
                model.molecule.atoms[b].position,
                model.molecule.atoms[c].position,
                directions,
                directions,
            );
            let offset = [frame.offset.y, frame.offset.z, frame.offset.x];
            Ok(std::array::from_fn(|axis| {
                let mut v = core_jet2::lanes(geometry.positions()[a][axis]);
                for column in 0..3 {
                    for lane in 0..4 {
                        v[lane] += u[column][axis][lane] * offset[column];
                    }
                }
                core_jet2::seed(v[0], v[1], v[2], v[3])
            }))
        }
    }
}

impl ContinuousSpeciesModel {
    /// Full moving-AO/grid second derivative for an independent nuclear/q path.
    pub fn joint_directional_derivatives(
        &self,
        state: &SpeciesState,
        nuclear_direction: &[Vec3],
        parameter_direction: &[f64],
        config: &ResponseSolverConfig,
    ) -> Result<JointDirectionalReport> {
        self.joint_directional_derivatives_with_provider(
            state,
            nuclear_direction,
            parameter_direction,
            config,
            &AllElectronSource,
        )
    }

    /// The provider REPLACES the default delta; it is never added to it.
    pub fn joint_directional_derivatives_with_provider<S: CompleteJointCoreSource + ?Sized>(
        &self,
        state: &SpeciesState,
        dr: &[Vec3],
        dq: &[f64],
        config: &ResponseSolverConfig,
        provider: &S,
    ) -> Result<JointDirectionalReport> {
        self.check_state(state)?;
        config.validate()?;
        if dr.len() != self.molecule.len()
            || dq.len() != self.chart.parameters.len()
            || dr.iter().any(|r| (0..3).any(|i| !r.get(i).is_finite()))
            || dq.iter().any(|x| !x.is_finite())
        {
            return super::invalid(
                "joint direction must match the fixed finite role/chart coordinates",
            );
        }
        let pair = provider.pair(self, state, dr, dq)?;
        if pair.geometry_space != GeometryJetSpace::IndependentRoles
            || pair.base_field_convention != BaseFieldConvention::OriginZero
            || pair.point_ion_scope != PointIonScope::DifferenceFromLabelledPointIons
            || pair.origin != state.origin
            || pair.fixed_external_field != Some(state.field)
        {
            return super::invalid("joint provider must supply the complete independent-role origin-zero delta at the reference field");
        }
        let [value, first, other_first, second] = pair.operator;
        if first
            .as_slice()
            .iter()
            .zip(other_first.as_slice())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return super::invalid("joint pair legs must be the same analytic direction");
        }
        let operator = second_order_operator_coefficients(value, first, second)?;
        let scalar = pair.classical_energy;
        if scalar[1].to_bits() != scalar[2].to_bits() {
            return super::invalid("joint scalar pair legs must agree");
        }
        if pair
            .permanent_dipole
            .iter()
            .any(|v| v[1].to_bits() != v[2].to_bits())
        {
            return super::invalid("joint permanent-dipole pair legs must agree");
        }
        let permanent_dipole = (0..3)
            .map(|k| {
                std::array::from_fn(|i| {
                    pair.permanent_dipole[i][[0, 1, 3][k]] * if k == 2 { 0.5 } else { 1. }
                })
            })
            .collect();
        let classical = ExternalClassicalTaylor {
            interaction: vec![scalar[0], scalar[1], 0.5 * scalar[3]],
            internal_energy_reference: self.fragment_internal_energy,
            permanent_dipole,
        };
        let direction = MolecularDirection::nuclear(dr.iter().map(|v| [v.x, v.y, v.z]).collect());
        let mut options = self.options.clone();
        options.external_field = Some(state.field);
        let mut recipe = Fingerprint(provider.identity());
        recipe.number(self.basis.nao_sph);
        let identity = ExtraCoreIdentity {
            chart: self.fingerprints.chart,
            operator_layout: recipe.0 ^ self.fingerprints.basis,
            grid_recipe: self.fingerprints.grid,
            coordinate_map: self.fingerprints.geometry,
        };
        let source = ExtraCoreTaylor::new(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
            &direction,
            dq,
            identity.clone(),
            ExtraCoreCoordinates::IndependentRoles,
            [operator.clone(), operator],
            classical,
            [state.origin.x, state.origin.y, state.origin.z],
            ExtraCoreFieldConvention::ProvidedCompleteDeltaFromOrdinaryOriginZero,
            2,
        )?;
        let geometry = source.geometry(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
            &direction,
            dq,
            &identity,
        )?;
        let report = molecular_taylor_with_extra_core(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
            &direction,
            dq,
            &identity,
            &source,
            2,
            config,
            geometry,
        )?;
        if report.coordinates != ExtraCoreCoordinates::IndependentRoles
            || !report.response.converged
            || report.response.value.energy_coefficients.len() != 3
            || report
                .response
                .value
                .energy_coefficients
                .iter()
                .any(|x| !x.is_finite())
        {
            return super::invalid(
                "joint analytic response did not converge to finite second-order coefficients",
            );
        }
        Ok(JointDirectionalReport {
            energy_coefficients: report.response.value.energy_coefficients[..3]
                .try_into()
                .map_err(|_| DftError::InvalidInput("joint coefficient count".into()))?,
            residuals: report.response.residuals,
            iterations: report.response.iterations,
            warnings: report.response.warnings,
            source_fingerprint: report.source_fingerprint,
            external_classical_coefficients: report.external_classical_coefficients,
            internal_energy_reference: report.internal_energy_reference,
            projector_defects: report.response.value.projector_defects,
            stationarity_defects: report.response.value.stationarity_defects,
            metric_defects: report.response.value.metric_defects,
            particle_number_coefficients: report.response.value.particle_number_coefficients,
            reference_energy_error: report.response.value.reference_energy_error,
            reference_fock_error: report.response.value.reference_fock_error,
        })
    }

    /// All ER, Eq, ERR, ERq and Eqq in one common analytic energy convention.
    /// Cost is n(n+1)/2 directional responses for n=3*Nroles+Nparameters.
    pub fn joint_derivatives(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
    ) -> Result<JointDerivativeReport> {
        self.joint_derivatives_with_provider(state, config, &AllElectronSource)
    }

    pub fn joint_derivatives_with_provider<S: CompleteJointCoreSource + ?Sized>(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
        provider: &S,
    ) -> Result<JointDerivativeReport> {
        let nr = 3 * self.molecule.len();
        let nq = self.chart.parameters.len();
        let n = nr + nq;
        let mut h = Matrix::zeros(n, n);
        let mut reports = Vec::with_capacity(n * (n + 1) / 2);
        let mut gradient = Vec::with_capacity(n);
        let mut diagonal = Vec::with_capacity(n);
        for a in 0..n {
            let (dr, dq) = axis_direction(self.molecule.len(), nq, &[a]);
            let r = self
                .joint_directional_derivatives_with_provider(state, &dr, &dq, config, provider)?;
            gradient.push(r.energy_coefficients[1]);
            diagonal.push(r.energy_coefficients[2]);
            h[(a, a)] = 2. * r.energy_coefficients[2];
            reports.push(r);
        }
        for a in 0..n {
            for b in 0..a {
                let (dr, dq) = axis_direction(self.molecule.len(), nq, &[a, b]);
                let r = self.joint_directional_derivatives_with_provider(
                    state, &dr, &dq, config, provider,
                )?;
                let mut sum = AccurateSum::default();
                sum.add_product(r.energy_coefficients[2], 1.);
                sum.add_product(diagonal[a], -1.);
                sum.add_product(diagonal[b], -1.);
                let mixed = sum.value();
                h[(a, b)] = mixed;
                h[(b, a)] = mixed;
                reports.push(r);
            }
        }
        Ok(JointDerivativeReport {
            fingerprints: self.fingerprints.clone(),
            parameter_names: self
                .chart
                .parameters
                .iter()
                .map(|p| p.name.clone())
                .collect(),
            parameter_units: self
                .chart
                .parameters
                .iter()
                .map(|p| p.unit.clone())
                .collect(),
            parameter_scales: self.chart.parameters.iter().map(|p| p.scale).collect(),
            nuclear_gradient: gradient[..nr]
                .chunks_exact(3)
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect(),
            parameter_gradient: gradient[nr..].to_vec(),
            nuclear_hessian: block(&h, 0, 0, nr, nr),
            nuclear_parameter_hessian: block(&h, 0, nr, nr, nq),
            parameter_hessian: block(&h, nr, nr, nq, nq),
            directional_reports: reports,
        })
    }

    pub fn nuclear_hessian(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
    ) -> Result<Matrix> {
        Ok(self.joint_derivatives(state, config)?.nuclear_hessian)
    }
    pub fn nuclear_parameter_hessian(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
    ) -> Result<Matrix> {
        Ok(self
            .joint_derivatives(state, config)?
            .nuclear_parameter_hessian)
    }
    pub fn nuclear_hessian_with_provider<S: CompleteJointCoreSource + ?Sized>(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
        provider: &S,
    ) -> Result<Matrix> {
        Ok(self
            .joint_derivatives_with_provider(state, config, provider)?
            .nuclear_hessian)
    }
    pub fn nuclear_parameter_hessian_with_provider<S: CompleteJointCoreSource + ?Sized>(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
        provider: &S,
    ) -> Result<Matrix> {
        Ok(self
            .joint_derivatives_with_provider(state, config, provider)?
            .nuclear_parameter_hessian)
    }
}

fn axis_direction(atoms: usize, parameters: usize, axes: &[usize]) -> (Vec<Vec3>, Vec<f64>) {
    let mut r = vec![[0.; 3]; atoms];
    let mut q = vec![0.; parameters];
    for &a in axes {
        if a < 3 * atoms {
            r[a / 3][a % 3] = 1.;
        } else {
            q[a - 3 * atoms] = 1.;
        }
    }
    (
        r.into_iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect(),
        q,
    )
}

fn block(matrix: &Matrix, row: usize, column: usize, rows: usize, columns: usize) -> Matrix {
    let mut output = Matrix::zeros(rows, columns);
    for i in 0..rows {
        for j in 0..columns {
            output[(i, j)] = matrix[(row + i, column + j)];
        }
    }
    output
}
