// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Experimental common-driver backend for a saved, fixed few-center
//! candidate. Serialized model coordinates and coefficients stay in native units.

use super::few_center::{
    evaluate, fixed_dictionary_id, FewCenterEvaluation, FixedFewCenterCoefficients,
};
use super::fragment_driver::{fragment_spec, object_control};
use super::reduction_linear::{ElectronLedger, FrozenFragmentSpec};
use super::SpeciesFunctional;
use crate::driver::{method::molecule_from, Controls, System, Value};
use crate::math::Vec3;
use crate::pseudo_sub::reduction::GenerationOptions;
use crate::pseudo_sub::reduction_basis::OneCenterBasisTemplate;
use crate::pseudo_sub::reduction_geometry::{FragmentSpec, RigidExpansion};
use crate::pseudo_sub::reduction_roles::{layout, CenterTemplate};
use crate::system::Molecule;
use crate::{DftError, Result};
use std::collections::BTreeSet;

pub const MODEL_RECORD_VERSION: &str = "few-center-model-v1";
pub const MODE: &str = "few-evaluate";

fn invalid<T>(s: &str) -> Result<T> {
    Err(DftError::InvalidInput(s.into()))
}
fn unsupported<T>(s: &str) -> Result<T> {
    Err(DftError::Unsupported {
        feature: s.into(),
        missing: "independent saved-model energy/force acceptance for this combination".into(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FewCenterReference {
    Rks,
    Uks,
}
impl FewCenterReference {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rks => "rks",
            Self::Uks => "uks",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "rks" => Ok(Self::Rks),
            "uks" => Ok(Self::Uks),
            _ => unsupported("saved few-center reference must be rks or fixed-spin uks"),
        }
    }
}

/// Immutable model and derived rigid map. The map is constructed once from
/// saved full-reference geometry; runtime primary coordinates never redefine xi.
#[derive(Clone, Debug)]
pub struct FewCenterModelRecord {
    source: Molecule,
    spec: FragmentSpec,
    map: RigidExpansion,
    ledger: ElectronLedger,
    template: CenterTemplate,
    functional: SpeciesFunctional,
    grid_level: usize,
    reference: FewCenterReference,
    active_spin_electrons: [usize; 2],
    dictionary: GenerationOptions,
    coefficients: FixedFewCenterCoefficients,
}

fn keys(v: &Value, expected: &[&str], context: &str) -> Result<()> {
    let Value::Object(entries) = v else {
        return invalid(&format!("{context} must be an object"));
    };
    let mut seen = BTreeSet::new();
    for (name, _) in entries {
        if !seen.insert(name.as_str()) {
            return invalid(&format!("{context} contains duplicate key {name}"));
        }
        if !expected.contains(&name.as_str()) {
            return invalid(&format!("{context} contains unknown key {name}"));
        }
    }
    if seen.len() != expected.len() {
        return invalid(&format!("{context} has missing required keys"));
    }
    Ok(())
}
fn required<'a>(v: &'a Value, name: &str) -> Result<&'a Value> {
    v.get(name)
        .ok_or_else(|| DftError::InvalidInput(format!("missing required model key {name}")))
}
fn string(v: &Value) -> Result<&str> {
    v.as_str()
        .ok_or_else(|| DftError::InvalidInput("model string required".into()))
}
fn integer(v: &Value) -> Result<i64> {
    match v {
        Value::Int(x) => Ok(*x),
        _ => invalid(
            "model integer required; floating values and booleans are not integer identifiers",
        ),
    }
}
fn index(v: &Value) -> Result<usize> {
    usize::try_from(integer(v)?)
        .map_err(|_| DftError::InvalidInput("model nonnegative integer overflow".into()))
}
fn array(v: &Value) -> Result<&[Value]> {
    v.as_array()
        .ok_or_else(|| DftError::InvalidInput("model array required".into()))
}
// The legacy fragment parser accepts integral floats via Value::as_i64.
// This versioned record requires actual integer tokens for every identifier;
// leave the ordinary and old saved-fragment routes unchanged.
fn strict_fragment_indices(v: &Value) -> Result<()> {
    for name in ["explicit_atoms", "retained_atoms", "port"] {
        for value in array(required(v, name)?)? {
            index(value)?;
        }
    }
    for pair in array(required(v, "single_bonds")?)? {
        for value in array(pair)? {
            index(value)?;
        }
    }
    index(required(v, "electronic_centers")?)?;
    Ok(())
}

fn floats<const N: usize>(v: &Value) -> Result<[f64; N]> {
    let values = array(v)?;
    if values.len() != N {
        return invalid("model numeric array has wrong fixed length");
    }
    let mut out = [0.0; N];
    for (a, b) in out.iter_mut().zip(values) {
        *a = b
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| DftError::InvalidInput("model numeric arrays must be finite".into()))?;
    }
    Ok(out)
}
fn template(s: &str) -> Result<CenterTemplate> {
    [
        CenterTemplate::CoreSp,
        CenterTemplate::CoreSpH,
        CenterTemplate::CoreSpHH,
    ]
    .into_iter()
    .find(|t| t.id() == s)
    .ok_or_else(|| DftError::InvalidInput("unknown saved few-center template".into()))
}
fn source_from_value(v: &Value) -> Result<Molecule> {
    keys(
        v,
        &["numbers", "positions", "charge", "multiplicity"],
        "source",
    )?;
    let numbers = array(required(v, "numbers")?)?
        .iter()
        .map(|n| {
            u8::try_from(integer(n)?)
                .map_err(|_| DftError::InvalidInput("saved element identifier must fit u8".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let positions = array(required(v, "positions")?)?
        .iter()
        .map(floats::<3>)
        .collect::<Result<Vec<_>>>()?;
    let charge = i32::try_from(integer(required(v, "charge")?)?)
        .map_err(|_| DftError::InvalidInput("saved source charge overflows i32".into()))?;
    // Reuse the ordinary System/molecular integer-label and finite-position
    // checks. No fractional/zero/virtual element identifier is introduced.
    molecule_from(
        &numbers,
        &positions,
        charge,
        index(required(v, "multiplicity")?)?,
    )
}
fn coefficients_from_value(v: &Value) -> Result<FixedFewCenterCoefficients> {
    keys(
        v,
        &[
            "model_version",
            "template",
            "operator_template_id",
            "functional_id",
            "dictionary_id",
            "reference_id",
            "gaussian_charges",
            "gaussian_body_dipoles",
            "local_amplitudes",
            "nonlocal_upper_triangle",
            "internal_energy_reference",
            "electronic_coefficients",
            "electrostatic_coefficients",
            "status",
        ],
        "coefficients",
    )?;
    if string(required(v, "model_version")?)? != "fixed-few-center-candidate-v1"
        || string(required(v, "status")?)? != "candidate-not-chemically-validated"
        || integer(required(v, "electronic_coefficients")?)? != 39
        || integer(required(v, "electrostatic_coefficients")?)? != 12
    {
        return invalid("saved few-center coefficient schema/version/count/status mismatch");
    }
    let dipoles = array(required(v, "gaussian_body_dipoles")?)?;
    if dipoles.len() != 3 {
        return invalid("saved few-center body dipoles require three vectors");
    }
    let mut body = [Vec3::zero(); 3];
    for (target, value) in body.iter_mut().zip(dipoles) {
        let p = floats::<3>(value)?;
        *target = Vec3::new(p[0], p[1], p[2]);
    }
    let p = FixedFewCenterCoefficients {
        template: template(string(required(v, "template")?)?)?,
        operator_template_id: string(required(v, "operator_template_id")?)?.into(),
        functional_id: string(required(v, "functional_id")?)?.into(),
        dictionary_id: string(required(v, "dictionary_id")?)?.into(),
        reference_id: string(required(v, "reference_id")?)?.into(),
        gaussian_charges: floats(required(v, "gaussian_charges")?)?,
        gaussian_body_dipoles: body,
        local_amplitudes: floats(required(v, "local_amplitudes")?)?,
        nonlocal_coefficients: floats(required(v, "nonlocal_upper_triangle")?)?,
        internal_energy_reference: required(v, "internal_energy_reference")?
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| DftError::InvalidInput("saved E0 must be finite".into()))?,
    };
    p.validate()?;
    Ok(p)
}

impl FewCenterModelRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Molecule,
        spec: FragmentSpec,
        template: CenterTemplate,
        functional: SpeciesFunctional,
        grid_level: usize,
        reference: FewCenterReference,
        active_spin_electrons: [usize; 2],
        coefficients: FixedFewCenterCoefficients,
    ) -> Result<Self> {
        if functional != SpeciesFunctional::Pbe {
            return unsupported("saved few-center energy/forces currently require the independently validated PBE subset");
        }
        let numbers: Vec<_> = source.atoms.iter().map(|a| a.z).collect();
        let positions: Vec<_> = source.atoms.iter().map(|a| a.position.to_array()).collect();
        let source = molecule_from(&numbers, &positions, source.charge, source.multiplicity)?;
        let map = RigidExpansion::methyl(&source, spec.clone())?;
        let ledger = ElectronLedger::new(
            &numbers,
            source.n_electrons()?,
            &[FrozenFragmentSpec {
                explicit_atoms: spec.explicit_atoms.clone(),
                frozen_electrons: 8,
            }],
        )?;
        let [na, nb] = active_spin_electrons;
        if ledger.active_electron_count == 0
            || na.checked_add(nb) != Some(ledger.active_electron_count)
            || na < nb
            || source.multiplicity != na - nb + 1
        {
            return invalid("saved source charge/multiplicity and explicit active spin sector disagree with frozen singlet Me8");
        }
        if reference == FewCenterReference::Rks && (na != nb || source.multiplicity != 1) {
            return invalid(
                "saved rks requires equal integer active spin populations and singlet source",
            );
        }
        if reference == FewCenterReference::Uks && template != CenterTemplate::CoreSp {
            return unsupported("saved uks energy/forces currently require the independently scoped one-center CoreSp template");
        }
        let dictionary = GenerationOptions {
            basis_template: OneCenterBasisTemplate::CarbonCoreValenceSp,
            functional: functional.name().into(),
            grid_level,
            ..GenerationOptions::default()
        };
        coefficients.validate()?;
        if coefficients.template != template
            || coefficients.functional_id.to_ascii_lowercase() != functional.name()
            || coefficients.dictionary_id != fixed_dictionary_id(&dictionary)?
        {
            return invalid("saved template/functional/dictionary and coefficients disagree");
        }
        let roles = layout(&map, &source, &map.reference_coordinates(), template)?;
        let nao = roles.basis()?.nao_sph;
        if active_spin_electrons.iter().any(|&n| n > nao) {
            return invalid("saved active spin sector exceeds its fixed AO template");
        }
        Ok(Self {
            source,
            spec,
            map,
            ledger,
            template,
            functional,
            grid_level,
            reference,
            active_spin_electrons,
            dictionary,
            coefficients,
        })
    }

    pub fn source(&self) -> &Molecule {
        &self.source
    }
    pub fn geometry_map(&self) -> &RigidExpansion {
        &self.map
    }
    pub fn electron_ledger(&self) -> &ElectronLedger {
        &self.ledger
    }
    pub fn reference(&self) -> FewCenterReference {
        self.reference
    }
    pub fn coefficients(&self) -> &FixedFewCenterCoefficients {
        &self.coefficients
    }
    pub fn primary_numbers(&self) -> Vec<u8> {
        self.spec
            .retained_atoms
            .iter()
            .map(|&i| self.source.atoms[i].z)
            .chain([1])
            .collect()
    }

    pub fn from_json(text: &str) -> Result<Self> {
        Self::from_value(&Value::parse_json(text)?)
    }
    pub fn from_value(v: &Value) -> Result<Self> {
        keys(
            v,
            &[
                "model_version",
                "position_units",
                "source",
                "fragment_spec",
                "center_template",
                "functional",
                "grid_level",
                "reference",
                "active_spin_electrons",
                "coefficients",
            ],
            "few-center model",
        )?;
        if string(required(v, "model_version")?)? != MODEL_RECORD_VERSION
            || string(required(v, "position_units")?)? != "bohr"
        {
            return invalid(
                "saved few-center model version or native Bohr coordinate convention mismatch",
            );
        }
        let spec_value = required(v, "fragment_spec")?;
        keys(
            spec_value,
            &[
                "explicit_atoms",
                "retained_atoms",
                "port",
                "single_bonds",
                "electronic_centers",
            ],
            "fragment_spec",
        )?;
        strict_fragment_indices(spec_value)?;
        let s = array(required(v, "active_spin_electrons")?)?;
        if s.len() != 2 {
            return invalid("saved active_spin_electrons requires exactly [Na,Nb]");
        }
        let functional = match string(required(v, "functional")?)?
            .to_ascii_lowercase()
            .as_str()
        {
            "lda" => SpeciesFunctional::Lda,
            "pbe" => SpeciesFunctional::Pbe,
            _ => return unsupported("saved few-center functional must be LDA or PBE"),
        };
        Self::new(
            source_from_value(required(v, "source")?)?,
            fragment_spec(spec_value)?,
            template(string(required(v, "center_template")?)?)?,
            functional,
            index(required(v, "grid_level")?)?,
            FewCenterReference::parse(string(required(v, "reference")?)?)?,
            [index(&s[0])?, index(&s[1])?],
            coefficients_from_value(required(v, "coefficients")?)?,
        )
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(self.to_value()?.to_json_pretty())
    }
    pub fn to_value(&self) -> Result<Value> {
        let source = Value::object()
            .with(
                "numbers",
                self.source
                    .atoms
                    .iter()
                    .map(|a| a.z as usize)
                    .collect::<Vec<_>>(),
            )
            .with(
                "positions",
                self.source
                    .atoms
                    .iter()
                    .map(|a| a.position)
                    .collect::<Vec<_>>(),
            )
            .with("charge", self.source.charge)
            .with("multiplicity", self.source.multiplicity);
        let spec = Value::object()
            .with("explicit_atoms", self.spec.explicit_atoms.clone())
            .with("retained_atoms", self.spec.retained_atoms.clone())
            .with(
                "port",
                vec![
                    self.spec.port.fragment_atom,
                    self.spec.port.host_atom,
                    self.spec.port.reference_atom,
                ],
            )
            .with(
                "single_bonds",
                self.spec
                    .single_bonds
                    .iter()
                    .map(|p| p.to_vec())
                    .collect::<Vec<_>>(),
            )
            .with("electronic_centers", self.spec.electronic_centers);
        Ok(Value::object()
            .with("model_version", MODEL_RECORD_VERSION)
            .with("position_units", "bohr")
            .with("source", source)
            .with("fragment_spec", spec)
            .with("center_template", self.template.id())
            .with("functional", self.functional.name())
            .with("grid_level", self.grid_level)
            .with("reference", self.reference.name())
            .with("active_spin_electrons", self.active_spin_electrons.to_vec())
            .with("coefficients", self.coefficients.to_value()?))
    }

    pub fn evaluate_primary(
        &self,
        system: &System,
        field: Vec3,
        origin: Vec3,
    ) -> Result<FewCenterEvaluation> {
        if system.cell.is_some() || system.pbc.iter().any(|&v| v) {
            return unsupported("saved few-center models are nonperiodic");
        }
        // The ordinary molecular validator handles integer labels and finite
        // positions. Only the PRIMARY labels are accepted; aux roles are hidden.
        let primary = molecule_from(
            &system.numbers,
            &system.positions,
            self.source.charge,
            self.source.multiplicity,
        )?;
        if system.numbers != self.primary_numbers() {
            return invalid("runtime System labels must equal saved retained labels followed by the main H-like label; auxiliary centers are not runtime atoms");
        }
        let coordinates: Vec<_> = primary.atoms.iter().map(|a| a.position).collect();
        let spins =
            (self.reference == FewCenterReference::Uks).then_some(self.active_spin_electrons);
        evaluate(
            &self.source,
            &self.map,
            &self.ledger,
            self.template,
            &self.dictionary,
            &self.coefficients,
            &coordinates,
            field,
            origin,
            spins,
            None,
        )
    }
}

/// Saved primitive energy/force branch of the continuous-species dispatcher.
/// Global resource and unit envelopes own those controls; no local pool exists.
pub fn run(system: &System, controls: &Controls) -> Result<Value> {
    if controls.str("fragment_mode", "")? != MODE {
        return invalid("saved few-center backend requires fragment_mode=few-evaluate");
    }
    let record =
        FewCenterModelRecord::from_value(&object_control(controls, "fragment_model_json")?)?;
    let charge = controls
        .opt_i64("charge")?
        .unwrap_or(i64::from(record.source.charge));
    let multiplicity = controls.usize("multiplicity", record.source.multiplicity)?;
    if charge != i64::from(record.source.charge) || multiplicity != record.source.multiplicity {
        return invalid(
            "runtime charge/multiplicity must match the saved full-source electron sector",
        );
    }
    if let Some(reference) = controls.opt_str("reference")? {
        if FewCenterReference::parse(&reference)? != record.reference {
            return invalid("runtime reference differs from the saved few-center model");
        }
    }
    for key in [
        "ecp",
        "ecp_basis",
        "spinor",
        "spin_orbit",
        "dispersion",
        "d3",
        "d4",
        "solvent",
        "pcm",
        "smd",
    ] {
        if controls.contains(key) {
            return unsupported("saved few-center energy/forces do not support ECP, spinor, dispersion or solvent combinations");
        }
    }
    let field = controls.opt_vec3("field")?.unwrap_or_else(Vec3::zero);
    let origin = controls
        .opt_vec3("charge_origin")?
        .unwrap_or_else(Vec3::zero);
    controls.finish("continuous-species few-evaluate")?;
    let value = record.evaluate_primary(system, field, origin)?;
    let counts = Value::object()
        .with("full_electrons", value.full_electrons)
        .with("frozen_electrons", value.frozen_electrons)
        .with("active_electrons", value.active_electrons)
        .with("primary_sites", system.numbers.len())
        .with("electronic_centers", value.electronic_centers)
        .with("role_sites", value.role_sites)
        .with("retained_nuclei", value.retained_nuclei)
        .with("ao_count", value.ao_count)
        .with("grid_sites", value.grid_sites)
        .with("grid_points", value.grid_points)
        .with("projector_functions", value.projector_functions)
        .with("electrostatic_sites", value.electrostatic_sites)
        .with("local_potential_sites", value.local_potential_sites);
    Ok(Value::object().with("feature","continuous-species").with("model","saved fixed primitive few-center methyl candidate")
        .with("fragment_mode",MODE).with("model_version",MODEL_RECORD_VERSION).with("template_id",value.template_id)
        .with("reference_type",record.reference.name()).with("reference_id",value.reference_id).with("converged",true)
        .with("energy",value.energy).with("electronic_energy",value.electronic_energy).with("classical_energy",value.classical_energy)
        .with("energy_without_internal_reference",value.energy_without_internal_reference)
        .with("classical_energy_without_internal_reference",value.classical_energy_without_internal_reference)
        .with("fragment_internal_energy",value.fragment_internal_energy)
        .with("scf_energy_convention",super::FRAGMENT_SCF_ENERGY_CONVENTION)
        .with("forces",value.forces).with("physical_total_charge",value.physical_total_charge)
        .with("active_spin_electrons",value.spin_electrons.to_vec()).with("frozen_spin_electrons",value.frozen_spin_electrons.to_vec())
        .with("full_spin_electrons",value.full_spin_electrons.to_vec()).with("unrestricted",value.unrestricted).with("spin_status",value.spin_status)
        .with("active_density_trace",value.active_density_trace).with("active_spin_trace",value.active_spin_trace)
        .with("scf_iterations",value.scf_iterations).with("scf_orbital_residual",value.scf_orbital_residual).with("frontier_gap",value.frontier_gap)
        .with("overlap_condition_number",value.overlap_condition_number).with("counts",counts).with("field",field).with("charge_origin",origin)
        .with("candidate_status",value.candidate_status).with("shape_status",value.shape_status).with("response_status",value.response_status)
        .with("derivative_status","analytic-fixed-model-rigid-primary-forces")
        .with("warnings",vec!["Candidate model: chemical transferability, shape correction and frozen internal response are not established."])
        .with("units",Value::object().with("energy","hartree").with("electronic_energy","hartree").with("classical_energy","hartree")
            .with("energy_without_internal_reference","hartree").with("classical_energy_without_internal_reference","hartree")
            .with("fragment_internal_energy","hartree")
            .with("forces","hartree/bohr").with("physical_total_charge","e")
            .with("active_density_trace","dimensionless").with("active_spin_trace","dimensionless")
            .with("overlap_condition_number","dimensionless")
            .with("scf_orbital_residual","hartree").with("frontier_gap","hartree").with("field","atomic units electric field").with("charge_origin","bohr")))
}
