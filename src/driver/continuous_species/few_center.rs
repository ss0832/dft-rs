// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Experimental Rust-only variational few-center consumer. A numerical model
//! is supplied explicitly; this module neither generates nor fits coefficients.

use super::local::{LocalCenter, LocalGaussianSpec};
use super::reduction_linear::{ElectronLedger, FixedElectronicSector, FrozenFragmentSpec};
use super::static_core::{static_core_correction_with_auxiliary_roles, StaticCoreParameters};
use super::{
    ChargeChart, ChargeParameter, ContinuousSpeciesModel, Fingerprint, SpeciesFingerprints,
    SpeciesFunctional,
};
use crate::driver::Value;
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::overlap_matrix;
use crate::math::Vec3;
use crate::pseudo_sub::reduction::GenerationOptions;
use crate::pseudo_sub::reduction_basis::{fixed_site_basis, OneCenterBasisTemplate};
use crate::pseudo_sub::reduction_geometry::RigidExpansion;
use crate::pseudo_sub::reduction_roles::{layout, CenterTemplate};
use crate::response::ResponseSolverConfig;
use crate::scf::{EriMode, Guess, ScfOptions};
use crate::system::Molecule;
use crate::{DftError, Result};

/// The sole implemented operator recipe. Equal coefficient dimensions do not
/// make contracted projectors or exact frozen-density electrostatics compatible.
pub const PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID: &str =
    "primitive-projector8-fitted-gaussian3-local3-v1";

/// New-model coefficients and reference are intentionally separate from the
/// version-one single-center generated record. A fresh projection is required
/// for each discrete CenterTemplate; no old-record mutation is performed here.
#[derive(Clone, Debug)]
pub struct FixedFewCenterCoefficients {
    pub template: CenterTemplate,
    pub operator_template_id: String,
    pub functional_id: String,
    pub dictionary_id: String,
    pub gaussian_charges: [f64; 3],
    pub gaussian_body_dipoles: [Vec3; 3],
    pub local_amplitudes: [f64; 3],
    pub nonlocal_coefficients: [f64; 36],
    pub internal_energy_reference: f64,
    pub reference_id: String,
}

fn invalid<T>(s: &str) -> Result<T> {
    Err(DftError::InvalidInput(s.into()))
}

impl FixedFewCenterCoefficients {
    pub fn validate(&self) -> Result<()> {
        if self.operator_template_id != PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID {
            return Err(DftError::Unsupported {
                feature: "few-center operator template".into(),
                missing: "only primitive8 projectors with fitted Gaussian3 electrostatics and local3 potentials are implemented; contracted and exact-electrostatic oracle coefficients require their own energy/force providers".into(),
            });
        }
        canonical_functional(&self.functional_id)?;
        if self.reference_id.trim().is_empty()
            || self.dictionary_id.trim().is_empty()
            || self
                .gaussian_charges
                .iter()
                .chain(
                    self.gaussian_body_dipoles
                        .iter()
                        .flat_map(|v| [&v.x, &v.y, &v.z]),
                )
                .chain(&self.local_amplitudes)
                .chain(&self.nonlocal_coefficients)
                .chain(std::iter::once(&self.internal_energy_reference))
                .any(|x| !x.is_finite())
        {
            return invalid("few-center coefficients require finite fixed values and explicit reference provenance");
        }
        let total = self.gaussian_charges.iter().sum::<f64>();
        let scale = self
            .gaussian_charges
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            .max(1.0);
        if total.abs() > 64.0 * f64::EPSILON * scale {
            return invalid(
                "few-center Gaussian correction must be neutral without silent renormalization",
            );
        }
        Ok(())
    }

    /// Save new experimental coefficients independently of old generated models.
    /// This is a record serializer, not an activation through public controls.
    pub fn to_value(&self) -> Result<Value> {
        self.validate()?;
        let numbers = |v: &[f64]| Value::Array(v.iter().copied().map(Value::Float).collect());
        Ok(Value::object()
            .with("model_version", "fixed-few-center-candidate-v1")
            .with("template", self.template.id())
            .with("operator_template_id", self.operator_template_id.clone())
            .with(
                "functional_id",
                canonical_functional(&self.functional_id)?.name(),
            )
            .with("dictionary_id", self.dictionary_id.clone())
            .with("reference_id", self.reference_id.clone())
            .with("gaussian_charges", numbers(&self.gaussian_charges))
            .with(
                "gaussian_body_dipoles",
                Value::Array(
                    self.gaussian_body_dipoles
                        .iter()
                        .map(|v| numbers(&v.to_array()))
                        .collect(),
                ),
            )
            .with("local_amplitudes", numbers(&self.local_amplitudes))
            .with(
                "nonlocal_upper_triangle",
                numbers(&self.nonlocal_coefficients),
            )
            .with("internal_energy_reference", self.internal_energy_reference)
            .with("electronic_coefficients", 39)
            .with("electrostatic_coefficients", 12)
            .with("status", "candidate-not-chemically-validated"))
    }
}

/// One immutable monovalent sigma replacement chart. Full atom and frozen
/// electron counts belong to endpoint provenance; only the retained scaffold,
/// active spin sector and actual fixed operator dictionary must agree.
/// This compatibility contract is not a physical accuracy certificate.
pub struct FixedFewCenterChart {
    geometry_map: RigidExpansion,
    connection_chart_id: String,
    template: CenterTemplate,
    dictionary_id: String,
    functional: SpeciesFunctional,
    sector: FixedElectronicSector,
    retained_nuclear_charge: i64,
    molecular_charge: i64,
}

/// A coefficient endpoint checked against one particular chart instance.
/// Its fields are private so another chart cannot acquire the same evidence
/// by changing an AO count, spin population or operator label afterwards.
pub struct CompatibleFewCenterEndpoint<'a> {
    chart: &'a FixedFewCenterChart,
    coefficients: &'a FixedFewCenterCoefficients,
    ledger: &'a ElectronLedger,
}

impl CompatibleFewCenterEndpoint<'_> {
    /// Discrete explicit-source accounting is preserved as provenance; it is
    /// never averaged into fractional nuclei or frozen electron populations.
    pub fn realization_ledger(&self) -> &ElectronLedger {
        self.ledger
    }
}

fn sigma_ledger(ledger: &ElectronLedger) -> Result<()> {
    let Some(fragment) = ledger
        .fragments
        .first()
        .filter(|_| ledger.fragments.len() == 1)
    else {
        return invalid("a monovalent sigma chart requires exactly one eliminated fragment");
    };
    let frozen = i64::try_from(ledger.frozen_electron_count)
        .map_err(|_| DftError::InvalidInput("frozen electron count overflow".into()))?;
    let active = i64::try_from(ledger.active_electron_count)
        .map_err(|_| DftError::InvalidInput("active electron count overflow".into()))?;
    if fragment.effective_ionic_charge != 1
        || fragment.frozen_electrons != ledger.frozen_electron_count
        || fragment.frozen_electrons % 2 != 0
        || fragment.nuclear_charge.checked_sub(frozen) != Some(1)
        || ledger
            .frozen_electron_count
            .checked_add(ledger.active_electron_count)
            != Some(ledger.full_electron_count)
        || ledger
            .retained_nuclear_charge
            .checked_add(fragment.nuclear_charge)
            != Some(ledger.full_nuclear_charge)
        || ledger.retained_nuclear_charge.checked_add(1) != Some(ledger.active_ionic_charge)
        || ledger.active_ionic_charge.checked_sub(active) != Some(ledger.reduced_total_charge)
        || ledger.full_total_charge != ledger.reduced_total_charge
    {
        return invalid("monovalent sigma chart requires a consistent +1 core, closed-shell eliminated sector and fixed active charge accounting");
    }
    Ok(())
}

impl FixedFewCenterChart {
    pub fn new(
        dictionary: &GenerationOptions,
        template: CenterTemplate,
        geometry_map: &RigidExpansion,
        ledger: &ElectronLedger,
        spin_electrons: [usize; 2],
    ) -> Result<Self> {
        sigma_ledger(ledger)?;
        let source = geometry_map.reference_molecule();
        let source_ledger = ElectronLedger::new(
            &source.atoms.iter().map(|atom| atom.z).collect::<Vec<_>>(),
            source.n_electrons()?,
            &[FrozenFragmentSpec {
                explicit_atoms: geometry_map.fragment_atoms().to_vec(),
                frozen_electrons: ledger.frozen_electron_count,
            }],
        )?;
        if &source_ledger != ledger {
            return invalid(
                "replacement chart geometry and explicit source electron ledger disagree",
            );
        }
        let coordinates = geometry_map.reference_coordinates();
        let roles = layout(geometry_map, source, &coordinates, template)?;
        let total_ao_functions = roles.basis()?.nao_sph;
        let sector = FixedElectronicSector {
            active_electrons: ledger.active_electron_count,
            spin_electrons,
            total_ao_functions,
        };
        sector.validate(
            ledger.active_electron_count,
            spin_electrons,
            total_ao_functions,
        )?;
        // Bind the actual saved connection and body frame, not merely the
        // number of centers or a caller-selected chart label. Replacement
        // endpoints all use this one immutable common-space geometry map.
        let mut identity = Fingerprint::new();
        identity.string("fixed-sigma-connection-chart-v1");
        identity.string(template.id());
        identity.string(&fixed_dictionary_id(dictionary)?);
        identity.string(validate_dictionary(dictionary)?.name());
        for atom in &source.atoms {
            identity.number(usize::from(atom.z));
            for x in atom.position.to_array() {
                identity.float(x);
            }
        }
        let spec = geometry_map.fragment_spec().ok_or_else(|| {
            DftError::InvalidInput(
                "replacement chart requires an explicit sigma connection map".into(),
            )
        })?;
        for indices in [&spec.explicit_atoms, &spec.retained_atoms] {
            identity.number(indices.len());
            for &index in indices {
                identity.number(index);
            }
        }
        for index in [
            spec.port.fragment_atom,
            spec.port.host_atom,
            spec.port.reference_atom,
        ] {
            identity.number(index);
        }
        for &(atom, offset) in geometry_map.internal_offsets() {
            identity.number(atom);
            for x in offset.to_array() {
                identity.float(x);
            }
        }
        for n in spin_electrons {
            identity.number(n);
        }
        Ok(Self {
            geometry_map: geometry_map.clone(),
            connection_chart_id: format!("fixed-sigma-chart:{:016x}", identity.0),
            template,
            dictionary_id: fixed_dictionary_id(dictionary)?,
            functional: validate_dictionary(dictionary)?,
            sector,
            retained_nuclear_charge: ledger.retained_nuclear_charge,
            molecular_charge: ledger.reduced_total_charge,
        })
    }

    /// Register supplied coefficients without regenerating a model or changing
    /// its basis. Charge coordinates are explicit zero-integral basis weights:
    /// q0*(g0-g2) + q1*(g1-g2), with normalized Gaussian g_i.
    pub fn endpoint<'a>(
        &'a self,
        coefficients: &'a FixedFewCenterCoefficients,
        ledger: &'a ElectronLedger,
        spin_electrons: [usize; 2],
        total_ao_functions: usize,
    ) -> Result<CompatibleFewCenterEndpoint<'a>> {
        coefficients.validate()?;
        sigma_ledger(ledger)?;
        self.sector.validate(
            ledger.active_electron_count,
            spin_electrons,
            total_ao_functions,
        )?;
        if ledger.retained_nuclear_charge != self.retained_nuclear_charge
            || ledger.reduced_total_charge != self.molecular_charge
            || coefficients.template != self.template
            || coefficients.dictionary_id != self.dictionary_id
            || canonical_functional(&coefficients.functional_id)? != self.functional
        {
            return invalid("replacement endpoint changes the retained scaffold charge, molecular charge, basis/center template, operator dictionary or functional");
        }
        if coefficients.gaussian_charges[2]
            != -(coefficients.gaussian_charges[0] + coefficients.gaussian_charges[1])
        {
            return invalid("replacement endpoint is not expressed in the explicit two-coordinate neutral Gaussian basis; no charge renormalization is performed");
        }
        Ok(CompatibleFewCenterEndpoint {
            chart: self,
            coefficients,
            ledger,
        })
    }

    /// Interpolate fixed operator coefficients, never atom counts, densities or
    /// converged Fock matrices. The +1 baseline ionic role is not interpolated.
    /// Supplied simplex weights are used unchanged; they are not normalized.
    pub fn interpolate(
        &self,
        endpoints: &[CompatibleFewCenterEndpoint<'_>],
        weights: &[f64],
    ) -> Result<FixedFewCenterCoefficients> {
        if endpoints.is_empty()
            || endpoints.len() != weights.len()
            || endpoints
                .iter()
                .any(|endpoint| !std::ptr::eq(endpoint.chart, self))
            || weights
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 8.0 * f64::EPSILON
        {
            return invalid("local replacement interpolation requires endpoints from this chart and finite nonnegative unit-sum weights");
        }
        let mut result = FixedFewCenterCoefficients {
            template: self.template,
            operator_template_id: PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID.into(),
            functional_id: self.functional.name().into(),
            dictionary_id: self.dictionary_id.clone(),
            gaussian_charges: [0.0; 3],
            gaussian_body_dipoles: [Vec3::zero(); 3],
            local_amplitudes: [0.0; 3],
            nonlocal_coefficients: [0.0; 36],
            internal_energy_reference: 0.0,
            reference_id: String::new(),
        };
        let mut identity = Fingerprint::new();
        identity.string("fixed-sector-coefficient-simplex-v1");
        identity.string(&self.connection_chart_id);
        for (endpoint, &weight) in endpoints.iter().zip(weights) {
            let c = endpoint.coefficients;
            identity.string(&c.reference_id);
            identity.float(weight);
            for &x in c
                .gaussian_charges
                .iter()
                .chain(
                    c.gaussian_body_dipoles
                        .iter()
                        .flat_map(|v| [&v.x, &v.y, &v.z]),
                )
                .chain(&c.local_amplitudes)
                .chain(&c.nonlocal_coefficients)
                .chain(std::iter::once(&c.internal_energy_reference))
            {
                identity.float(x);
            }
            for i in 0..2 {
                result.gaussian_charges[i] += weight * c.gaussian_charges[i];
            }
            for i in 0..3 {
                result.gaussian_body_dipoles[i] += c.gaussian_body_dipoles[i] * weight;
                result.local_amplitudes[i] += weight * c.local_amplitudes[i];
            }
            for (out, &x) in result
                .nonlocal_coefficients
                .iter_mut()
                .zip(&c.nonlocal_coefficients)
            {
                *out += weight * x;
            }
            result.internal_energy_reference += weight * c.internal_energy_reference;
        }
        result.gaussian_charges[2] = -(result.gaussian_charges[0] + result.gaussian_charges[1]);
        result.reference_id = format!("coefficient-simplex:{:016x}", identity.0);
        result.validate()?;
        Ok(result)
    }

    pub fn active_electron_count(&self) -> usize {
        self.sector.active_electrons
    }
    pub fn spin_electron_counts(&self) -> [usize; 2] {
        self.sector.spin_electrons
    }
    pub fn total_ao_functions(&self) -> usize {
        self.sector.total_ao_functions
    }
    pub fn connection_chart_id(&self) -> &str {
        &self.connection_chart_id
    }
    pub fn geometry_map(&self) -> &RigidExpansion {
        &self.geometry_map
    }
}

pub struct FewCenterEvaluation {
    pub energy: f64,
    pub electronic_energy: f64,
    pub classical_energy: f64,
    /// Physical scalar before adding the state-common internal reference.
    pub energy_without_internal_reference: f64,
    pub classical_energy_without_internal_reference: f64,
    /// Already included once in the reported total and classical energies.
    pub fragment_internal_energy: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    /// Independent role derivatives, before one and only one geometry pullback.
    pub role_gradient: Vec<Vec3>,
    pub model_dipole: Vec3,
    pub permanent_ionic_dipole: Vec3,
    pub active_electronic_dipole: Vec3,
    pub physical_total_charge: f64,
    pub active_density_trace: f64,
    pub active_electrons: usize,
    pub frozen_electrons: usize,
    pub full_electrons: usize,
    pub spin_electrons: [usize; 2],
    /// The eliminated Me core is an explicitly fixed singlet, not a general
    /// open-shell frozen region. Counts are not inferred from AO labels.
    pub frozen_spin_electrons: [usize; 2],
    pub full_spin_electrons: [usize; 2],
    pub active_spin_trace: f64,
    pub unrestricted: bool,
    pub spin_status: &'static str,
    pub electronic_centers: usize,
    pub role_sites: usize,
    pub retained_nuclei: usize,
    pub grid_sites: usize,
    pub ao_count: usize,
    pub grid_points: usize,
    pub projector_functions: usize,
    pub electrostatic_sites: usize,
    pub local_potential_sites: usize,
    pub response_variables: usize,
    pub overlap_condition_number: f64,
    pub scf_iterations: usize,
    pub scf_orbital_residual: f64,
    pub frontier_gap: Option<f64>,
    pub template_id: &'static str,
    pub reference_id: String,
    pub fingerprints: SpeciesFingerprints,
    pub active_model_polarizability: Option<[[f64; 3]; 3]>,
    pub response_residuals: Option<Vec<f64>>,
    pub candidate_status: &'static str,
    pub shape_status: &'static str,
    pub response_status: &'static str,
}

fn canonical_functional(id: &str) -> Result<SpeciesFunctional> {
    match id.to_ascii_lowercase().as_str() {
        "lda" => Ok(SpeciesFunctional::Lda),
        "pbe" => Ok(SpeciesFunctional::Pbe),
        _ => invalid("few-center static evaluation supports only the scoped LDA/PBE functionals"),
    }
}

fn validate_dictionary(d: &GenerationOptions) -> Result<SpeciesFunctional> {
    let standard = GenerationOptions::default();
    if d.basis_template != OneCenterBasisTemplate::CarbonCoreValenceSp
        || d.local_exponents != standard.local_exponents
        || d.charge_exponents != standard.charge_exponents
        || d.s_exponents != standard.s_exponents
        || d.p_exponents != standard.p_exponents
        || d.grid_level > 9
    {
        return invalid("few-center evaluation requires the common fixed C5/H1s contractions, Gaussian/projector widths and grid level0..9");
    }
    canonical_functional(&d.functional)
}

/// Deterministic operator/basis dictionary identity, not a security hash.
/// Generation weights and production grid resolution do not change the fixed
/// electronic operator; reference integration conditions belong in provenance.
/// Functional identity is bound separately so canonical aliases agree.
pub fn fixed_dictionary_id(d: &GenerationOptions) -> Result<String> {
    validate_dictionary(d)?;
    let mut id = Fingerprint::new();
    id.string("fixed-few-center-operator-dictionary-v1");
    id.string(PRIMITIVE_FITTED_OPERATOR_TEMPLATE_ID);
    id.string("primitive-AnalyticShell-normalized-s0-s1-p0m0m1m2-p1m0m1m2-v1");
    id.string("GaussianCharge-integrated-q-physical-body-dipole-v1");
    id.string("D-upper-triangle-row-order-u-then-v-ge-u-v1");
    for (name, widths) in [
        ("local3", d.local_exponents.as_slice()),
        ("charge3", d.charge_exponents.as_slice()),
        ("projector-s2", d.s_exponents.as_slice()),
        ("projector-p2", d.p_exponents.as_slice()),
    ] {
        id.string(name);
        id.number(widths.len());
        for &x in widths {
            id.float(x);
        }
    }
    let t = crate::basis::aos::cart_to_sph(1);
    id.string("p-Cartesian-to-real-spherical-matrix-column-order");
    id.number(t.rows);
    id.number(t.cols);
    for &x in t.as_slice() {
        id.float(x);
    }
    for template in [
        OneCenterBasisTemplate::CarbonCoreValenceSp,
        OneCenterBasisTemplate::Hydrogen1s,
    ] {
        id.string(template.id());
        let basis = fixed_site_basis(template)?;
        id.number(basis.z as usize);
        id.number(basis.shells.len());
        for shell in basis.shells {
            id.number(shell.l as usize);
            id.number(shell.exps.len());
            id.number(shell.coefs.len());
            for x in shell.exps.into_iter().chain(shell.coefs) {
                id.float(x);
            }
        }
    }
    Ok(format!("fnv1a64:{:016x}", id.0))
}

/// Actual stationary SCF, analytic role gradients and analytic rigid-coordinate
/// pullback. Auxiliary centers are exactly the first/second saved ordered H
/// positions. All role sites own fixed-label quadrature; no full-source grid or
/// full-source Fock/density is retained or recalculated in this production path.
/// The frozen eight-electron core is a singlet [4,4]. Optional fixed active
/// [Na,Nb] therefore implies full [Na+4,Nb+4]. Open frozen cores and changes of
/// spin sector are outside this static candidate model.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    source: &Molecule,
    map: &RigidExpansion,
    ledger: &ElectronLedger,
    template: CenterTemplate,
    dictionary: &GenerationOptions,
    coefficients: &FixedFewCenterCoefficients,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
    spin_electrons: Option<[usize; 2]>,
    response_config: Option<&ResponseSolverConfig>,
) -> Result<FewCenterEvaluation> {
    coefficients.validate()?;
    if coefficients.template != template {
        return invalid("few-center coefficients belong to a different discrete AO/role template");
    }
    let functional = validate_dictionary(dictionary)?;
    if canonical_functional(&coefficients.functional_id)? != functional {
        return invalid("few-center coefficients and evaluation functional differ");
    }
    if coefficients.dictionary_id != fixed_dictionary_id(dictionary)? {
        return invalid("few-center coefficients and actual operator dictionary differ");
    }
    let spec = map.fragment_spec().ok_or_else(|| {
        DftError::InvalidInput("few-center evaluation requires an explicit methyl map".into())
    })?;
    let expected = ElectronLedger::new(
        &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        source.n_electrons()?,
        &[FrozenFragmentSpec {
            explicit_atoms: spec.explicit_atoms.clone(),
            frozen_electrons: 8,
        }],
    )?;
    if &expected != ledger
        || ledger.active_electron_count == 0
        || ledger.fragments[0].effective_ionic_charge != 1
    {
        return invalid(
            "few-center source, fixed frozen methyl ledger and active electrons are inconsistent",
        );
    }
    let active = ledger.active_electron_count;
    let spins = spin_electrons.unwrap_or([active / 2; 2]);
    if spins[0].checked_add(spins[1]) != Some(active)
        || spins[0] < spins[1]
        || source.multiplicity != spins[0] - spins[1] + 1
        || (spin_electrons.is_none() && active % 2 != 0)
    {
        return invalid("few-center spin sector must match declared active electrons and explicit full-source multiplicity");
    }
    let full_spins = [
        spins[0]
            .checked_add(4)
            .ok_or_else(|| DftError::InvalidInput("full alpha count overflow".into()))?,
        spins[1]
            .checked_add(4)
            .ok_or_else(|| DftError::InvalidInput("full beta count overflow".into()))?,
    ];
    if full_spins[0].checked_add(full_spins[1]) != Some(ledger.full_electron_count)
        || source.multiplicity != full_spins[0] - full_spins[1] + 1
    {
        return invalid(
            "few-center full spin counts do not match the declared source and frozen singlet core",
        );
    }
    if field
        .to_array()
        .iter()
        .chain(origin.to_array().iter())
        .any(|x| !x.is_finite())
    {
        return invalid("few-center field and origin must be finite");
    }
    let roles = layout(map, source, coordinates, template)?;
    let molecule = roles.basis_role_molecule.clone();
    let basis = roles.basis()?;
    let nao = basis.nao_sph;
    if spins.iter().any(|&n| n > nao) {
        return invalid("few-center spin occupations exceed the fixed AO template");
    }
    let overlap = overlap_matrix(&basis);
    let role_coordinates: Vec<_> = molecule.atoms.iter().map(|a| a.position).collect();
    let auxiliary: Vec<_> = (0..molecule.len())
        .map(|a| roles.auxiliary_roles.contains(&a))
        .collect();
    let parameters = StaticCoreParameters {
        charge_exponents: dictionary.charge_exponents,
        charge_coefficients: coefficients.gaussian_charges,
        body_dipoles: coefficients.gaussian_body_dipoles,
        projector_s_exponents: dictionary.s_exponents,
        projector_p_exponents: dictionary.p_exponents,
        nonlocal_coefficients: coefficients.nonlocal_coefficients,
    };
    let correction = static_core_correction_with_auxiliary_roles(
        &basis,
        &role_coordinates,
        &roles.ionic_charges,
        roles.main_ion,
        &roles.frame_in_role_coordinates,
        &parameters,
        origin,
        &auxiliary,
    )?;
    let mut permanent = correction.permanent_dipole;
    for (&z, &r) in roles.ionic_charges.iter().zip(&role_coordinates) {
        permanent += (r - origin) * z;
    }
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: dictionary.grid_level,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )?;
    let grid_points = grid.len();
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let chart = ChargeChart {
        reference_charges: roles.ionic_charges.clone(),
        conserve_total_charge: true,
        parameters: vec![ChargeParameter {
            name: "fixed-few-center-model".into(),
            unit: "dimensionless".into(),
            scale: 1.0,
            bounds: [-1.0, 1.0],
            charge_direction: vec![0.0; molecule.len()],
        }],
    };
    let local: Vec<_> = (0..3)
        .map(|i| LocalGaussianSpec {
            center: LocalCenter::Atom(roles.main_ion),
            exponent: dictionary.local_exponents[i],
            reference_amplitude: coefficients.local_amplitudes[i],
            amplitude_direction: vec![0.0],
            reference_steric_strength: 0.0,
            steric_direction: vec![0.0],
            environment_exponents: vec![1.0; molecule.len()],
            environment_hardness: vec![0.0; molecule.len()],
            excluded_atoms: vec![true; molecule.len()],
        })
        .collect();
    let model = match spin_electrons {
        None => ContinuousSpeciesModel::new_with_auxiliary_roles(
            molecule, basis, grid, active, chart, functional, options, origin, auxiliary,
        ),
        Some(counts) => ContinuousSpeciesModel::new_with_spin_and_auxiliary_roles(
            molecule, basis, grid, counts, chart, functional, options, origin, auxiliary,
        ),
    }?;
    let mut model = model
        .with_local_potentials(local)?
        .with_static_core_correction(correction)?
        .with_fragment_internal_energy(coefficients.internal_energy_reference)?;
    let mut identity = Fingerprint(model.fingerprints.chart);
    identity.string("fixed-few-center-candidate-v1");
    identity.string(template.id());
    identity.string(&coefficients.operator_template_id);
    identity.string(canonical_functional(&coefficients.functional_id)?.name());
    identity.string(&coefficients.dictionary_id);
    identity.string(&coefficients.reference_id);
    model.fingerprints.chart = identity.0;
    let state = model.evaluate(&[0.0], field, None)?;
    let trace = ledger.check_active_trace(&state.total_density(), &overlap, 1e-9)?;
    let active_spin_trace = state.spin_density().frobenius_dot(&overlap);
    if (active_spin_trace - (spins[0] - spins[1]) as f64).abs() > 1e-9 {
        return invalid(
            "few-center active spin density does not match its fixed integer spin sector",
        );
    }
    if (state.total_charge() - ledger.reduced_total_charge as f64).abs() > 1e-10 {
        return invalid("few-center physical charge and frozen-electron ledger disagree");
    }
    let role_gradient = model.nuclear_gradient(&state)?;
    let gradient = roles.pullback(&role_gradient)?;
    let forces = gradient.iter().map(|&g| g * -1.0).collect();
    let response = response_config
        .map(|c| model.sensitivity(&state, c))
        .transpose()?;
    Ok(FewCenterEvaluation {
        energy: state.energy(),
        electronic_energy: state.electronic_energy(),
        classical_energy: state.classical_energy(),
        energy_without_internal_reference: state.energy_without_internal_reference(),
        classical_energy_without_internal_reference: state
            .classical_energy_without_internal_reference(),
        fragment_internal_energy: state.fragment_internal_energy(),
        gradient,
        forces,
        role_gradient,
        model_dipole: state.dipole,
        permanent_ionic_dipole: permanent,
        active_electronic_dipole: state.dipole - permanent,
        physical_total_charge: state.total_charge(),
        active_density_trace: trace,
        active_electrons: active,
        frozen_electrons: ledger.frozen_electron_count,
        full_electrons: ledger.full_electron_count,
        spin_electrons: spins,
        frozen_spin_electrons: [4, 4],
        full_spin_electrons: full_spins,
        active_spin_trace,
        unrestricted: state.scf.unrestricted,
        spin_status: "fixed-active-spin-sector-with-frozen-singlet-core",
        electronic_centers: template.center_count(),
        role_sites: roles.basis_role_molecule.len(),
        retained_nuclei: roles.retained_nucleus_roles.len(),
        grid_sites: roles.grid_site_roles.len(),
        ao_count: nao,
        grid_points,
        projector_functions: 8,
        electrostatic_sites: 3,
        local_potential_sites: 3,
        response_variables: if spin_electrons.is_some() {
            spins.iter().map(|&n| n * (nao - n)).sum()
        } else {
            spins[0] * (nao - spins[0])
        },
        overlap_condition_number: model.overlap_condition_number,
        scf_iterations: state.diagnostics.iterations,
        scf_orbital_residual: state.diagnostics.orbital_residual,
        frontier_gap: state.frontier_gap(),
        template_id: template.id(),
        reference_id: coefficients.reference_id.clone(),
        fingerprints: model.fingerprints.clone(),
        active_model_polarizability: response.as_ref().map(|r| r.polarizability),
        response_residuals: response.map(|r| r.response_residuals),
        candidate_status: "candidate-not-chemically-validated",
        shape_status: "not-generated",
        response_status: "active-only-no-internal-polarization",
    })
}
