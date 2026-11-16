// SPDX-License-Identifier: Apache-2.0
//! Immutable saved few-center recipes for complete clamped-field derivatives.
//!
//! The selected three-dimensional chart varies the existing local amplitudes,
//! not three additional fitted coefficients. All other amplitudes, internal
//! geometry and the energy reference remain fixed. No reference SCF or fitting
//! is performed by this factory. This module is deliberately unregistered until
//! its integration and numerical acceptance gates have been executed.

use super::contracted::{self, ContractedCoefficients};
use super::core_jet2::{
    self, BaseFieldConvention, CorePairJet, GeometryJetSpace, PairCoefficients, PairGeometry,
    PointIonScope, Projectors,
};
use super::few_center::{fixed_dictionary_id, FixedFewCenterCoefficients};
use super::joint_derivatives::{CompleteJointCoreSource, JointDerivativeReport};
use super::local::{LocalCenter, LocalGaussianSpec};
use super::reduction_linear::{ElectronLedger, FrozenFragmentSpec};
use super::static_core::{static_core_correction_with_auxiliary_roles, StaticCoreParameters};
use super::{
    ChargeChart, ChargeParameter, ContinuousSpeciesModel, Fingerprint, SpeciesFunctional,
    SpeciesState,
};
use crate::basis::MolBasis;
use crate::grid::{GridOptions, MolecularGrid};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pseudo_sub::reduction::GenerationOptions;
use crate::pseudo_sub::reduction_geometry::{BodyFrameKinematics, RigidExpansion};
use crate::pseudo_sub::reduction_roles::{layout, CenterTemplate};
use crate::response::ResponseSolverConfig;
use crate::scf::{EriMode, Guess, ScfOptions};
use crate::system::Molecule;
use crate::{DftError, Result};

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

/// Fixed, dimensionless local-amplitude coordinates, in saved dictionary order.
/// Amplitude i is c_i + scale_i q_i (Eh); bounds never depend on geometry.
#[derive(Clone, Debug)]
pub struct Local3Chart {
    pub scales: [f64; 3],
    pub bounds: [[f64; 2]; 3],
}

impl Default for Local3Chart {
    fn default() -> Self {
        Self {
            scales: [0.05; 3],
            bounds: [[-1.0, 1.0]; 3],
        }
    }
}

impl Local3Chart {
    pub fn validate(&self) -> Result<()> {
        for i in 0..3 {
            if !self.scales[i].is_finite()
                || self.scales[i] <= 0.0
                || self.bounds[i].iter().any(|x| !x.is_finite())
                || self.bounds[i][0] >= self.bounds[i][1]
                || self.bounds[i][0] > 0.0
                || self.bounds[i][1] < 0.0
            {
                return invalid(
                    "local3 chart needs positive fixed scales and finite bounds containing zero",
                );
            }
        }
        Ok(())
    }
}

/// Operator conventions are discrete chart data, never inferred from labels.
#[derive(Clone, Debug)]
pub enum SavedCoreOperator {
    Primitive8(FixedFewCenterCoefficients),
    ContractedMe8(ContractedCoefficients),
}

/// Complete owned recipe. Private fields prevent replacing coefficients or the
/// lifting map while keeping a previously prepared Hamiltonian identity.
#[derive(Clone, Debug)]
pub struct FullCoreRecipe {
    source: Molecule,
    map: RigidExpansion,
    ledger: ElectronLedger,
    dictionary: GenerationOptions,
    operator: SavedCoreOperator,
    spin_electrons: Option<[usize; 2]>,
    chart: Local3Chart,
    identity: u64,
}

impl FullCoreRecipe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Molecule,
        map: RigidExpansion,
        ledger: ElectronLedger,
        dictionary: GenerationOptions,
        operator: SavedCoreOperator,
        spin_electrons: Option<[usize; 2]>,
        chart: Local3Chart,
    ) -> Result<Self> {
        chart.validate()?;
        let spec = map.fragment_spec().ok_or_else(|| {
            DftError::InvalidInput("joint recipe requires an explicit saved methyl map".into())
        })?;
        let expected = ElectronLedger::new(
            &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
            source.n_electrons()?,
            &[FrozenFragmentSpec {
                explicit_atoms: spec.explicit_atoms.clone(),
                frozen_electrons: 8,
            }],
        )?;
        if ledger != expected
            || ledger.fragments.len() != 1
            || ledger.fragments[0].effective_ionic_charge != 1
            || ledger.active_electron_count == 0
            || source.atoms.len() != map.full_atom_count()
            || source
                .atoms
                .iter()
                .zip(&map.reference_molecule().atoms)
                .any(|(a, b)| a.z != b.z || a.position != b.position)
        {
            return invalid(
                "joint recipe source, saved geometry and frozen-eight electron ledger mismatch",
            );
        }
        let spins = spin_electrons.unwrap_or([ledger.active_electron_count / 2; 2]);
        if spins[0].checked_add(spins[1]) != Some(ledger.active_electron_count)
            || spins[0] < spins[1]
            || source.multiplicity != spins[0] - spins[1] + 1
            || (spin_electrons.is_none() && ledger.active_electron_count % 2 != 0)
        {
            return invalid("joint recipe fixed active spin sector mismatch");
        }
        let mut identity = Fingerprint::new();
        identity.string("saved-few-joint-local3-v1");
        match &operator {
            SavedCoreOperator::Primitive8(c) => {
                c.validate()?;
                if c.functional_id != dictionary.functional
                    || c.dictionary_id != fixed_dictionary_id(&dictionary)?
                {
                    return invalid("primitive recipe functional/dictionary mismatch");
                }
                identity.string(&c.operator_template_id);
                identity.string(&c.dictionary_id);
                identity.string(&c.reference_id);
            }
            SavedCoreOperator::ContractedMe8(c) => {
                c.validate()?;
                if c.functional_id != dictionary.functional
                    || c.dictionary_id
                        != contracted::dictionary_id(&source, &map, c.template, &dictionary)?
                {
                    return invalid("contracted recipe functional/dictionary mismatch");
                }
                identity.string(&c.operator_id);
                identity.string(&c.dictionary_id);
                identity.string(&c.reference_id);
            }
        }
        let (parameters, local, e0, template) = amplitudes(&dictionary, &operator);
        identity.string(template.id());
        identity.string(&dictionary.functional);
        identity.number(dictionary.grid_level);
        identity.number(spins[0]);
        identity.number(spins[1]);
        identity.number(usize::from(spin_electrons.is_some()));
        for atom in &source.atoms {
            identity.number(atom.z as usize);
            identity.vector(atom.position);
        }
        for &(atom, offset) in map.internal_offsets() {
            identity.number(atom);
            identity.vector(offset);
        }
        for &i in map.retained_atoms() {
            identity.number(i);
        }
        for i in [
            spec.port.host_atom,
            spec.port.fragment_atom,
            spec.port.reference_atom,
        ] {
            identity.number(i);
        }
        for x in parameters
            .charge_coefficients
            .into_iter()
            .chain(local)
            .chain(parameters.nonlocal_coefficients)
            .chain(std::iter::once(e0))
            .chain(chart.scales)
            .chain(chart.bounds.into_iter().flatten())
        {
            identity.float(x);
        }
        for p in parameters.body_dipoles {
            identity.vector(p);
        }
        Ok(Self {
            source,
            map,
            ledger,
            dictionary,
            operator,
            spin_electrons,
            chart,
            identity: identity.0,
        })
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn chart(&self) -> &Local3Chart {
        &self.chart
    }
    pub fn operator(&self) -> &SavedCoreOperator {
        &self.operator
    }
    pub fn electron_ledger(&self) -> &ElectronLedger {
        &self.ledger
    }

    /// Assemble the same grid, AO roles, static correction and scalar E0 as the
    /// original evaluator, with a genuine local3 chart in place of its dummy q.
    pub fn prepare(&self, coordinates: &[Vec3], origin: Vec3) -> Result<PreparedFewJoint> {
        if origin.to_array().iter().any(|x| !x.is_finite()) {
            return invalid("joint recipe origin must be finite");
        }
        let (parameters, locals, e0, template) = amplitudes(&self.dictionary, &self.operator);
        let roles = layout(&self.map, &self.source, coordinates, template)?;
        let mut basis = roles.basis()?;
        let contracted = matches!(self.operator, SavedCoreOperator::ContractedMe8(_));
        let (molecule, charges, own, anchors, role_to_full, frame) = if contracted {
            let molecule = self.map.evaluate(coordinates)?.molecule;
            for shell in &mut basis.shells {
                shell.atom = roles.role_to_full_geometry[shell.atom];
            }
            let spec = self.map.fragment_spec().unwrap();
            let mut charges = vec![0.0; molecule.len()];
            for &i in self.map.retained_atoms() {
                charges[i] = f64::from(self.source.atoms[i].z);
            }
            charges[spec.port.fragment_atom] = 1.0;
            let frame = embed_frame(&self.map, coordinates)?;
            let ids = (0..molecule.len()).collect();
            (
                molecule,
                charges,
                spec.port.fragment_atom,
                [
                    spec.port.host_atom,
                    spec.port.fragment_atom,
                    spec.port.reference_atom,
                ],
                ids,
                frame,
            )
        } else {
            (
                roles.basis_role_molecule,
                roles.ionic_charges,
                roles.main_ion,
                roles.frame_anchor_roles,
                roles.role_to_full_geometry,
                roles.frame_in_role_coordinates,
            )
        };
        let spins = self
            .spin_electrons
            .unwrap_or([self.ledger.active_electron_count / 2; 2]);
        if spins.iter().any(|&n| n > basis.nao_sph) {
            return invalid("joint recipe occupation exceeds AO space");
        }
        let positions: Vec<_> = molecule.atoms.iter().map(|a| a.position).collect();
        let auxiliary: Vec<_> = charges.iter().map(|&z| z == 0.0).collect();
        let mut electrostatic_parameters = parameters.clone();
        if contracted {
            electrostatic_parameters.nonlocal_coefficients = [0.0; 36];
        }
        let mut correction = static_core_correction_with_auxiliary_roles(
            &basis,
            &positions,
            &charges,
            own,
            &frame,
            &electrostatic_parameters,
            origin,
            &auxiliary,
        )?;
        let right = if contracted {
            let (right, sites) = contracted::right_basis(&molecule, &self.map)?;
            let overlaps =
                contracted::overlaps(&basis, molecule.len(), &right, sites, frame.orientation)?;
            let nonlocal = overlaps.operator(&parameters.nonlocal_coefficients, &frame)?;
            add(&mut correction.operator, &nonlocal.operator);
            for (a, derivatives) in correction.atom_derivative_matrices.iter_mut().enumerate() {
                for axis in 0..3 {
                    add(&mut derivatives[axis], &nonlocal.atom_derivatives[a][axis]);
                }
            }
            Some((right, sites))
        } else {
            None
        };
        let grid = MolecularGrid::build(
            &molecule,
            &GridOptions {
                level: self.dictionary.grid_level,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )?;
        let chart = ChargeChart {
            reference_charges: charges.clone(),
            conserve_total_charge: true,
            parameters: (0..3)
                .map(|i| ChargeParameter {
                    name: format!("local-amplitude-{i}"),
                    unit: "dimensionless".into(),
                    scale: self.chart.scales[i],
                    bounds: self.chart.bounds[i],
                    charge_direction: vec![0.0; molecule.len()],
                })
                .collect(),
        };
        let local = (0..3)
            .map(|i| LocalGaussianSpec {
                center: LocalCenter::Atom(own),
                exponent: self.dictionary.local_exponents[i],
                reference_amplitude: locals[i],
                amplitude_direction: (0..3)
                    .map(|j| if i == j { self.chart.scales[i] } else { 0.0 })
                    .collect(),
                reference_steric_strength: 0.0,
                steric_direction: vec![0.0; 3],
                environment_exponents: vec![1.0; molecule.len()],
                environment_hardness: vec![0.0; molecule.len()],
                excluded_atoms: vec![true; molecule.len()],
            })
            .collect();
        let functional = match self.dictionary.functional.as_str() {
            "lda" => SpeciesFunctional::Lda,
            "pbe" => SpeciesFunctional::Pbe,
            _ => return invalid("joint recipe supports only fixed LDA/PBE semilocal models"),
        };
        let options = ScfOptions {
            guess: Guess::Core,
            max_iter: 300,
            e_tol: 1e-13,
            p_tol: 1e-12,
            conv_tol_grad: 1e-11,
            eri_mode: EriMode::InCore { limit_mb: 64 },
            ..ScfOptions::default()
        };
        let model = match self.spin_electrons {
            None => ContinuousSpeciesModel::new_with_auxiliary_roles(
                molecule,
                basis,
                grid,
                self.ledger.active_electron_count,
                chart,
                functional,
                options,
                origin,
                auxiliary,
            ),
            Some(counts) => ContinuousSpeciesModel::new_with_spin_and_auxiliary_roles(
                molecule, basis, grid, counts, chart, functional, options, origin, auxiliary,
            ),
        }?;
        let mut model = model
            .with_local_potentials(local)?
            .with_static_core_correction(correction)?
            .with_fragment_internal_energy(e0)?;
        let mut identity = Fingerprint(model.fingerprints.chart);
        identity.number(self.identity as usize);
        model.fingerprints.chart = identity.0;
        let provider = FewCoreSource {
            parameters,
            local_amplitudes: locals,
            local_exponents: self.dictionary.local_exponents,
            local_scales: self.chart.scales,
            ionic_charges: charges,
            anchors,
            right,
            identity: model.fingerprints.chart,
        };
        Ok(PreparedFewJoint {
            model,
            provider,
            map: self.map.clone(),
            coordinates: coordinates.to_vec(),
            role_to_full,
            ledger: self.ledger.clone(),
            spins,
        })
    }
}

fn amplitudes(
    dictionary: &GenerationOptions,
    operator: &SavedCoreOperator,
) -> (StaticCoreParameters, [f64; 3], f64, CenterTemplate) {
    let (charges, dipoles, local, d, e0, template) = match operator {
        SavedCoreOperator::Primitive8(c) => (
            c.gaussian_charges,
            c.gaussian_body_dipoles,
            c.local_amplitudes,
            c.nonlocal_coefficients,
            c.internal_energy_reference,
            c.template,
        ),
        SavedCoreOperator::ContractedMe8(c) => (
            c.gaussian_charges,
            c.gaussian_body_dipoles,
            c.local_amplitudes,
            c.body_nonlocal_upper,
            c.internal_energy_reference,
            c.template,
        ),
    };
    (
        StaticCoreParameters {
            charge_exponents: dictionary.charge_exponents,
            charge_coefficients: charges,
            body_dipoles: dipoles,
            projector_s_exponents: dictionary.s_exponents,
            projector_p_exponents: dictionary.p_exponents,
            nonlocal_coefficients: d,
        },
        local,
        e0,
        template,
    )
}

fn add(a: &mut Matrix, b: &Matrix) {
    for (x, &y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += y;
    }
}

fn embed_frame(map: &RigidExpansion, coordinates: &[Vec3]) -> Result<BodyFrameKinematics> {
    let frame = map.body_frame_kinematics(coordinates)?.ok_or_else(|| {
        DftError::InvalidInput("joint recipe requires a complete rigid frame".into())
    })?;
    let main = map.fragment_spec().unwrap().port.fragment_atom;
    let mut orientation_jacobian = Matrix::zeros(9, 3 * map.full_atom_count());
    let mut center_jacobian = Matrix::zeros(3, 3 * map.full_atom_count());
    for (primary, full) in map
        .retained_atoms()
        .iter()
        .copied()
        .chain(std::iter::once(main))
        .enumerate()
    {
        for axis in 0..3 {
            for row in 0..9 {
                orientation_jacobian[(row, 3 * full + axis)] =
                    frame.orientation_jacobian[(row, 3 * primary + axis)];
            }
            for row in 0..3 {
                center_jacobian[(row, 3 * full + axis)] =
                    frame.center_jacobian[(row, 3 * primary + axis)];
            }
        }
    }
    Ok(BodyFrameKinematics {
        center: frame.center,
        orientation: frame.orientation,
        orientation_jacobian,
        center_jacobian,
    })
}

/// Complete additional operator relative to the ordinary labelled point-ion
/// Hamiltonian at origin zero. The shared connector consumes this replacement
/// once; model local terms and static corrections must not be added again.
pub struct FewCoreSource {
    parameters: StaticCoreParameters,
    local_amplitudes: [f64; 3],
    local_exponents: [f64; 3],
    local_scales: [f64; 3],
    ionic_charges: Vec<f64>,
    anchors: [usize; 3],
    right: Option<(MolBasis, [usize; 4])>,
    identity: u64,
}

impl CompleteJointCoreSource for FewCoreSource {
    fn identity(&self) -> u64 {
        self.identity
    }

    fn pair(
        &self,
        model: &ContinuousSpeciesModel,
        state: &SpeciesState,
        nuclear_direction: &[Vec3],
        parameter_direction: &[f64],
    ) -> Result<CorePairJet> {
        model.check_state(state)?;
        if model.fingerprints.chart != self.identity
            || state.coordinates().len() != 3
            || parameter_direction.len() != 3
            || parameter_direction.iter().any(|x| !x.is_finite())
        {
            return invalid("few core provider identity or local3 direction mismatch");
        }
        let zero = vec![Vec3::zero(); model.molecule.len()];
        let geometry = PairGeometry::new(
            &model.molecule,
            self.anchors,
            nuclear_direction,
            nuclear_direction,
            &zero,
            GeometryJetSpace::IndependentRoles,
        )?;
        // E0 is separately owned by the SCF/context, never duplicated here.
        let mut coefficients = PairCoefficients::constant(
            &self.ionic_charges,
            &self.parameters,
            self.local_amplitudes,
            0.0,
        )?;
        for i in 0..3 {
            let value = self.local_amplitudes[i] + self.local_scales[i] * state.coordinates()[i];
            let tangent = self.local_scales[i] * parameter_direction[i];
            coefficients.local_amplitudes[i] = core_jet2::seed(value, tangent, tangent, 0.0);
        }
        let projectors = match &self.right {
            None => Projectors::Primitive {
                s_exponents: self.parameters.projector_s_exponents,
                p_exponents: self.parameters.projector_p_exponents,
            },
            Some((basis, sites)) => Projectors::ContractedMe8 {
                basis,
                full_role_sites: *sites,
            },
        };
        core_jet2::evaluate_pair(
            &model.molecule,
            &model.basis,
            &geometry,
            &coefficients,
            self.parameters.charge_exponents,
            self.local_exponents,
            projectors,
            model.origin,
            state
                .field()
                .to_array()
                .map(|v| core_jet2::seed(v, 0.0, 0.0, 0.0)),
            PointIonScope::DifferenceFromLabelledPointIons,
            BaseFieldConvention::OriginZero,
        )
    }
}

/// Native model and its inseparable analytic recipe. The role coordinates
/// include auxiliary sites; public derivatives contain saved primary rows only.
pub struct PreparedFewJoint {
    model: ContinuousSpeciesModel,
    provider: FewCoreSource,
    map: RigidExpansion,
    coordinates: Vec<Vec3>,
    role_to_full: Vec<usize>,
    ledger: ElectronLedger,
    spins: [usize; 2],
}

/// Analytic clamped-field derivatives in saved primary Cartesian order.
/// Local3 coordinates are dimensionless; energy is Eh and geometry is Bohr.
/// The retained role report is diagnostic evidence, not extra force coordinates.
pub struct FewJointDerivatives {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub nuclear_hessian: Matrix,
    pub parameter_gradient: Vec<f64>,
    pub parameter_hessian: Matrix,
    pub nuclear_parameter_hessian: Matrix,
    pub role_report: JointDerivativeReport,
    pub derivative_convention: &'static str,
    pub coordinate_order: &'static str,
    pub parameter_order: [&'static str; 3],
    pub energy_unit: &'static str,
    pub gradient_unit: &'static str,
    pub parameter_gradient_unit: &'static str,
    pub nuclear_hessian_unit: &'static str,
    pub mixed_hessian_unit: &'static str,
    pub parameter_hessian_unit: &'static str,
}

impl PreparedFewJoint {
    pub fn model(&self) -> &ContinuousSpeciesModel {
        &self.model
    }
    pub fn source(&self) -> &FewCoreSource {
        &self.provider
    }
    pub fn coordinates(&self) -> &[Vec3] {
        &self.coordinates
    }

    pub fn nuclear_gradient(&self, state: &SpeciesState) -> Result<Vec<Vec3>> {
        let roles = self.model.nuclear_gradient(state)?;
        let mut full = vec![Vec3::zero(); self.map.full_atom_count()];
        for (&index, value) in self.role_to_full.iter().zip(roles) {
            full[index] = value;
        }
        self.map.pullback_forces(&self.coordinates, &full)
    }

    pub fn derivatives(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
    ) -> Result<FewJointDerivatives> {
        let role_report =
            self.model
                .joint_derivatives_with_provider(state, config, &self.provider)?;
        let (gradient, nuclear_hessian, nuclear_parameter_hessian) = self.pullback(
            &role_report.nuclear_gradient,
            &role_report.nuclear_hessian,
            &role_report.nuclear_parameter_hessian,
        )?;
        let forces = gradient.iter().map(|&g| g * -1.0).collect();
        Ok(FewJointDerivatives {
            energy: state.energy(),
            gradient,
            forces,
            nuclear_hessian,
            parameter_gradient: role_report.parameter_gradient.clone(),
            parameter_hessian: role_report.parameter_hessian.clone(),
            nuclear_parameter_hessian,
            role_report,
            derivative_convention: "analytic-clamped-field-fixed-recipe-primary-cartesian-local3",
            coordinate_order: "saved-retained-atoms-then-main-center-atom-major-xyz",
            parameter_order: [
                "local-amplitude-0",
                "local-amplitude-1",
                "local-amplitude-2",
            ],
            energy_unit: "Eh",
            gradient_unit: "Eh/bohr",
            parameter_gradient_unit: "Eh",
            nuclear_hessian_unit: "Eh/bohr^2",
            mixed_hessian_unit: "Eh/bohr",
            parameter_hessian_unit: "Eh",
        })
    }
    pub fn evaluate(
        &self,
        q: &[f64],
        field: Vec3,
        previous: Option<&SpeciesState>,
    ) -> Result<SpeciesState> {
        let state = self.model.evaluate(q, field, previous)?;
        self.ledger
            .check_active_trace(&state.total_density(), &self.model.overlap, 1e-9)?;
        let spin = state.spin_density().frobenius_dot(&self.model.overlap);
        if (spin - (self.spins[0] - self.spins[1]) as f64).abs() > 1e-9
            || (state.total_charge() - self.ledger.reduced_total_charge as f64).abs() > 1e-10
        {
            return invalid("joint SCF violates saved electron/spin/charge ledger");
        }
        Ok(state)
    }

    /// Embed independent role rows into the saved full skeleton, then apply
    /// the nonlinear lifting exactly once. This also handles primitive models
    /// whose grid excludes the eliminated H sites without inventing their force.
    pub fn pullback(
        &self,
        role_gradient: &[Vec3],
        role_hessian: &Matrix,
        role_mixed: &Matrix,
    ) -> Result<(Vec<Vec3>, Matrix, Matrix)> {
        let n = self.role_to_full.len();
        if role_gradient.len() != n
            || role_hessian.rows != 3 * n
            || role_hessian.cols != 3 * n
            || role_mixed.rows != 3 * n
            || role_mixed.cols != 3
        {
            return invalid(
                "few joint pullback requires complete independent-role RR and local3 Rq",
            );
        }
        let m = self.map.full_atom_count();
        let mut gradient = vec![Vec3::zero(); m];
        let mut hessian = Matrix::zeros(3 * m, 3 * m);
        let mut mixed = Matrix::zeros(3 * m, 3);
        for (a, &full_a) in self.role_to_full.iter().enumerate() {
            gradient[full_a] = role_gradient[a];
            for i in 0..3 {
                for q in 0..3 {
                    mixed[(3 * full_a + i, q)] = role_mixed[(3 * a + i, q)];
                }
                for (b, &full_b) in self.role_to_full.iter().enumerate() {
                    for j in 0..3 {
                        hessian[(3 * full_a + i, 3 * full_b + j)] =
                            role_hessian[(3 * a + i, 3 * b + j)];
                    }
                }
            }
        }
        Ok((
            self.map.pullback_forces(&self.coordinates, &gradient)?,
            self.map
                .pullback_hessian(&self.coordinates, &gradient, &hessian)?,
            self.map
                .pullback_mixed_derivatives(&self.coordinates, &mixed)?,
        ))
    }
}
