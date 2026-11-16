// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit immutable primitive recipes for retained-center/pose derivatives.
//!
//! This is not an adapter for a private generated-model schema or the distinct
//! few-center-model-v1 record. Retained center positions are supplied by the
//! calculation's System; Euler angles belong to this record and stay in radians.
//! Display atoms never create additional ions, electrons or AOs.
//! Generation provenance is supplied data, not a claim of transferable chemical
//! accuracy. Application functional/basis and actual derivative diagnostics must
//! be reported separately by a calculation using the recipe.
//!
//! Each reference_orientation array contains three body-axis COLUMNS in the
//! laboratory frame. The applied rotation is Rz(gamma) Ry(beta) Rx(alpha) Uref.
//! Primitive projector order is s0, s1, then the p0 and p1 triplets, each in the
//! library's real spherical m = 0,+1,-1 order. nonlocal_upper stores D[u,v] for
//! u <= v in row-major upper-triangle order without off-diagonal scaling.
//! These eight Gaussian projectors are not contracted occupied-region orbitals.

use super::core_jet2::seed;
use super::rigid_pose::{euler_rotation_pair, RigidCoreParameters, RigidFragment};
use crate::basis::{BasisSet, MolBasis};
use crate::driver::Value;
use crate::math::{Mat3, Vec3};
use crate::pseudo_sub::reduction_basis::{fixed_site_basis, OneCenterBasisTemplate};
use crate::system::Molecule;
use crate::{DftError, Result};
use std::collections::BTreeSet;

pub const RECORD_VERSION: &str = "rigid-static-core-v1";

#[derive(Clone, Debug)]
pub struct RigidElectronLedger {
    pub full_electrons: usize,
    pub frozen_electrons: usize,
    pub active_electrons: usize,
    pub full_spin_electrons: [usize; 2],
    pub frozen_spin_electrons: [usize; 2],
    pub full_nuclear_charge: usize,
}

/// Parser-owned fields prevent mutation after validation. There is no implicit
/// activation by substituent name, element label, filename or collection size.
pub struct RigidPoseRecord {
    fragments: Vec<RigidFragment>,
    templates: Vec<OneCenterBasisTemplate>,
    ledger: RigidElectronLedger,
    generation_functional: String,
    generation_basis: String,
    generation_recipe: String,
    source_record_sha256: Option<String>,
}

fn invalid<T>(s: &str) -> Result<T> {
    Err(DftError::InvalidInput(s.into()))
}
fn unsupported<T>(s: &str) -> Result<T> {
    Err(DftError::Unsupported { feature:s.into(), missing:"an explicitly implemented primitive recipe or independently validated schema adapter for this combination".into() })
}
fn keys(value: &Value, expected: &[&str]) -> Result<()> {
    let Value::Object(entries) = value else {
        return invalid("rigid recipe requires objects at the declared record locations");
    };
    let mut seen = BTreeSet::new();
    for (name, _) in entries {
        if !expected.contains(&name.as_str()) || !seen.insert(name.as_str()) {
            return invalid("rigid recipe has an unknown or duplicate key");
        }
    }
    if seen.len() != expected.len() {
        return invalid("rigid recipe is missing a required key");
    }
    Ok(())
}
fn get<'a>(v: &'a Value, k: &str) -> Result<&'a Value> {
    v.get(k)
        .ok_or_else(|| DftError::InvalidInput(format!("missing rigid recipe key {k}")))
}
fn text(v: &Value) -> Result<&str> {
    match v {
        Value::Str(s) if !s.trim().is_empty() => Ok(s),
        _ => invalid("rigid recipe requires nonempty text"),
    }
}
fn integer(v: &Value) -> Result<usize> {
    match v {
        Value::Int(i) => usize::try_from(*i).map_err(|_| {
            DftError::InvalidInput(
                "rigid recipe counts/indices must be nonnegative integers".into(),
            )
        }),
        _ => invalid("rigid recipe counts/indices require integer values, not booleans or floats"),
    }
}
fn real(v: &Value) -> Result<f64> {
    match v {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(x) if x.is_finite() => Ok(*x),
        _ => invalid("rigid recipe physical coefficients must be finite real numbers"),
    }
}
fn array(v: &Value) -> Result<&[Value]> {
    v.as_array()
        .ok_or_else(|| DftError::InvalidInput("rigid recipe requires an array".into()))
}
fn reals<const N: usize>(v: &Value) -> Result<[f64; N]> {
    let entries = array(v)?;
    if entries.len() != N {
        return invalid("rigid recipe dictionary has an incorrect fixed dimension");
    }
    let mut out = [0.; N];
    for (i, x) in entries.iter().enumerate() {
        out[i] = real(x)?;
    }
    Ok(out)
}
fn pair(v: &Value) -> Result<[usize; 2]> {
    let entries = array(v)?;
    if entries.len() != 2 {
        return invalid("rigid recipe spin counts require two integer channels");
    }
    Ok([integer(&entries[0])?, integer(&entries[1])?])
}
fn sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| {
        DftError::InvalidInput("rigid recipe electron/count arithmetic overflow".into())
    })
}

impl RigidPoseRecord {
    pub fn from_json(input: &str) -> Result<Self> {
        Self::from_value(&Value::parse_json(input)?)
    }

    pub fn from_value(value: &Value) -> Result<Self> {
        // A known private producer record is a missing capability, not an
        // invitation to relabel its version/template and bypass its contract.
        if value.get("model_version").and_then(Value::as_str) == Some("static-single-port-v2") {
            return unsupported("private static-single-port-v2 model ingestion");
        }
        let version = text(get(value, "model_version")?)?;
        if version != RECORD_VERSION {
            return unsupported("unrecognized rigid primitive record version");
        }
        keys(
            value,
            &[
                "model_version",
                "unit_convention",
                "generation",
                "ledger",
                "fragments",
                "scope",
            ],
        )?;
        if text(get(value, "unit_convention")?)? != "atomic" {
            return unsupported("nonnative units inside opaque rigid primitive records");
        }
        let generation = get(value, "generation")?;
        keys(
            generation,
            &["functional", "basis", "recipe", "source_record_sha256"],
        )?;
        let generation_functional = text(get(generation, "functional")?)?.into();
        let generation_basis = text(get(generation, "basis")?)?.into();
        let generation_recipe = text(get(generation, "recipe")?)?.into();
        let source_record_sha256 = match get(generation, "source_record_sha256")? {
            Value::Null => None,
            Value::Str(s) if s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()) => {
                Some(s.to_ascii_lowercase())
            }
            _ => {
                return invalid(
                    "source_record_sha256 must be null or an actual supplied SHA-256 string",
                )
            }
        };
        let scope = get(value, "scope")?;
        keys(
            scope,
            &["experimental", "chemical_accuracy", "internal_polarization"],
        )?;
        if get(scope, "experimental")? != &Value::Bool(true)
            || text(get(scope, "chemical_accuracy")?)? != "not_established"
            || get(scope, "internal_polarization")? != &Value::Bool(false)
        {
            return invalid("rigid primitive recipes must declare experimental static scope and unestablished chemical accuracy");
        }
        let input = get(value, "ledger")?;
        keys(
            input,
            &[
                "full_electrons",
                "frozen_electrons",
                "active_electrons",
                "full_spin_electrons",
                "frozen_spin_electrons",
                "full_nuclear_charge",
            ],
        )?;
        let ledger = RigidElectronLedger {
            full_electrons: integer(get(input, "full_electrons")?)?,
            frozen_electrons: integer(get(input, "frozen_electrons")?)?,
            active_electrons: integer(get(input, "active_electrons")?)?,
            full_spin_electrons: pair(get(input, "full_spin_electrons")?)?,
            frozen_spin_electrons: pair(get(input, "frozen_spin_electrons")?)?,
            full_nuclear_charge: integer(get(input, "full_nuclear_charge")?)?,
        };
        if sum(ledger.active_electrons, ledger.frozen_electrons)? != ledger.full_electrons
            || sum(ledger.full_spin_electrons[0], ledger.full_spin_electrons[1])?
                != ledger.full_electrons
            || sum(
                ledger.frozen_spin_electrons[0],
                ledger.frozen_spin_electrons[1],
            )? != ledger.frozen_electrons
        {
            return invalid("rigid primitive full/frozen/active electron ledgers disagree");
        }
        let active_spin = std::array::from_fn::<_, 2, _>(|s| {
            ledger.full_spin_electrons[s].checked_sub(ledger.frozen_spin_electrons[s])
        });
        if active_spin.iter().any(Option::is_none) {
            return invalid("frozen spin populations exceed full spin populations");
        }
        if ledger.active_electrons == 0
            || active_spin[0] != active_spin[1]
            || ledger.frozen_spin_electrons[0] != ledger.frozen_spin_electrons[1]
        {
            return unsupported("open-shell or empty-active-space rigid primitive derivatives");
        }
        let entries = array(get(value, "fragments")?)?;
        if entries.is_empty() {
            return invalid("rigid primitive collection requires at least one explicit core");
        }
        let mut fragments = Vec::with_capacity(entries.len());
        let mut templates = Vec::with_capacity(entries.len());
        let mut atoms = BTreeSet::new();
        for entry in entries {
            keys(
                entry,
                &[
                    "atom",
                    "basis_template",
                    "reference_orientation",
                    "euler_angles",
                    "local_exponents",
                    "local_amplitudes",
                    "charge_exponents",
                    "gaussian_charges",
                    "body_dipoles",
                    "projector_s_exponents",
                    "projector_p_exponents",
                    "nonlocal_upper",
                    "internal_energy_reference",
                ],
            )?;
            let atom = integer(get(entry, "atom")?)?;
            if !atoms.insert(atom) {
                return invalid("rigid primitive sources must use distinct explicit center roles");
            }
            let template_id = text(get(entry, "basis_template")?)?;
            let template = OneCenterBasisTemplate::from_id(template_id).map_err(|_| {
                DftError::Unsupported {
                    feature: format!("rigid primitive AO template {template_id}"),
                    missing: "a bundled explicit fixed site template".into(),
                }
            })?;
            let axes = array(get(entry, "reference_orientation")?)?;
            let dipoles = array(get(entry, "body_dipoles")?)?;
            if axes.len() != 3 || dipoles.len() != 3 {
                return invalid(
                    "rigid primitive axes and body dipoles require three Cartesian triples",
                );
            }
            let vectors = |rows: &[Value]| -> Result<[Vec3; 3]> {
                let a = reals::<3>(&rows[0])?;
                let b = reals::<3>(&rows[1])?;
                let c = reals::<3>(&rows[2])?;
                Ok([
                    Vec3::new(a[0], a[1], a[2]),
                    Vec3::new(b[0], b[1], b[2]),
                    Vec3::new(c[0], c[1], c[2]),
                ])
            };
            let axes = vectors(axes)?;
            let fragment = RigidFragment {
                atom,
                reference_orientation: Mat3::from_columns(axes[0], axes[1], axes[2]),
                euler_angles: reals(get(entry, "euler_angles")?)?,
                parameters: RigidCoreParameters {
                    charge_exponents: reals(get(entry, "charge_exponents")?)?,
                    charge_coefficients: reals(get(entry, "gaussian_charges")?)?,
                    body_dipoles: vectors(dipoles)?,
                    projector_s_exponents: reals(get(entry, "projector_s_exponents")?)?,
                    projector_p_exponents: reals(get(entry, "projector_p_exponents")?)?,
                    nonlocal_coefficients: reals(get(entry, "nonlocal_upper")?)?,
                },
                local_exponents: reals(get(entry, "local_exponents")?)?,
                local_amplitudes: reals(get(entry, "local_amplitudes")?)?,
                internal_energy_reference: real(get(entry, "internal_energy_reference")?)?,
            };
            fragment.parameters.validate()?;
            if fragment.local_exponents.iter().any(|x| *x <= 0.) {
                return invalid("rigid primitive local Gaussian widths must be positive");
            }
            euler_rotation_pair(
                fragment.reference_orientation,
                fragment.euler_angles.map(|x| seed(x, 0., 0., 0.)),
            )?;
            fragments.push(fragment);
            templates.push(template);
        }
        Ok(Self {
            fragments,
            templates,
            ledger,
            generation_functional,
            generation_basis,
            generation_recipe,
            source_record_sha256,
        })
    }

    pub fn fragments(&self) -> &[RigidFragment] {
        &self.fragments
    }
    pub fn ledger(&self) -> &RigidElectronLedger {
        &self.ledger
    }

    /// Validate the ACTIVE ionic geometry against the separate full ledger.
    /// Neutral frozen electron/charge removal is explicit; retained display
    /// coordinates are not expanded into new ions or orbital functions.
    pub fn build_basis(&self, molecule: &Molecule, data: &BasisSet) -> Result<MolBasis> {
        if molecule.is_empty()
            || molecule.multiplicity == 0
            || molecule
                .atoms
                .iter()
                .any(|a| a.position.to_array().iter().any(|x| !x.is_finite()))
        {
            return invalid("rigid primitive ionic geometry must be nonempty and finite with positive multiplicity");
        }
        if molecule.multiplicity != 1 || molecule.atoms.iter().any(|a| a.z == 0) {
            return unsupported("nonsinglet or undeclared auxiliary-role rigid primitive geometry");
        }
        for i in 0..molecule.len() {
            for j in 0..i {
                if (molecule.atoms[i].position - molecule.atoms[j].position).norm() == 0. {
                    return invalid("distinct retained point ions must not coincide");
                }
            }
        }
        if self
            .fragments
            .iter()
            .any(|f| f.atom >= molecule.len() || molecule.atoms[f.atom].z != 1)
        {
            return invalid("each rigid core requires an explicit +1 retained ionic center role");
        }
        let points = molecule
            .atoms
            .iter()
            .try_fold(0usize, |s, a| sum(s, usize::from(a.z)))?;
        if sum(points, self.ledger.frozen_electrons)? != self.ledger.full_nuclear_charge {
            return invalid("full nuclear charge does not match retained point ions plus the neutral frozen ledger");
        }
        if points as i128 - i128::from(molecule.charge) != self.ledger.active_electrons as i128 {
            return invalid("active ionic charge/electron count does not match the explicit rigid recipe ledger");
        }
        let overrides = self
            .fragments
            .iter()
            .zip(&self.templates)
            .map(|(f, t)| Ok((f.atom, fixed_site_basis(*t)?)))
            .collect::<Result<Vec<_>>>()?;
        let basis = MolBasis::build_with_overrides(molecule, data, &overrides)?;
        if !basis.ecps.is_empty() || basis.n_core_electrons != 0 {
            return unsupported(
                "an additional ECP/frozen-electron ledger in a rigid primitive recipe",
            );
        }
        let capacity = basis.nao_sph.checked_mul(2).ok_or_else(|| {
            DftError::InvalidInput("rigid primitive orbital capacity overflow".into())
        })?;
        if self.ledger.active_electrons > capacity {
            return invalid("active rigid recipe population exceeds its actual AO capacity");
        }
        Ok(basis)
    }

    /// Canonical native-unit record. No report-unit converter may rewrite it.
    pub fn to_value(&self) -> Value {
        let fragments = self
            .fragments
            .iter()
            .zip(&self.templates)
            .map(|(f, t)| {
                Value::object()
                    .with("atom", f.atom)
                    .with("basis_template", t.id())
                    .with(
                        "reference_orientation",
                        Value::Array(
                            f.reference_orientation
                                .col
                                .iter()
                                .map(|v| {
                                    Value::Array(
                                        v.to_array().into_iter().map(Value::Float).collect(),
                                    )
                                })
                                .collect(),
                        ),
                    )
                    .with("euler_angles", f.euler_angles.to_vec())
                    .with("local_exponents", f.local_exponents.to_vec())
                    .with("local_amplitudes", f.local_amplitudes.to_vec())
                    .with("charge_exponents", f.parameters.charge_exponents.to_vec())
                    .with(
                        "gaussian_charges",
                        f.parameters.charge_coefficients.to_vec(),
                    )
                    .with(
                        "body_dipoles",
                        Value::Array(
                            f.parameters
                                .body_dipoles
                                .iter()
                                .map(|v| {
                                    Value::Array(
                                        v.to_array().into_iter().map(Value::Float).collect(),
                                    )
                                })
                                .collect(),
                        ),
                    )
                    .with(
                        "projector_s_exponents",
                        f.parameters.projector_s_exponents.to_vec(),
                    )
                    .with(
                        "projector_p_exponents",
                        f.parameters.projector_p_exponents.to_vec(),
                    )
                    .with(
                        "nonlocal_upper",
                        f.parameters.nonlocal_coefficients.to_vec(),
                    )
                    .with("internal_energy_reference", f.internal_energy_reference)
            })
            .collect();
        Value::object()
            .with("model_version", RECORD_VERSION)
            .with("unit_convention", "atomic")
            .with(
                "generation",
                Value::object()
                    .with("functional", self.generation_functional.clone())
                    .with("basis", self.generation_basis.clone())
                    .with("recipe", self.generation_recipe.clone())
                    .with(
                        "source_record_sha256",
                        self.source_record_sha256
                            .as_ref()
                            .map_or(Value::Null, |s| Value::Str(s.clone())),
                    ),
            )
            .with(
                "ledger",
                Value::object()
                    .with("full_electrons", self.ledger.full_electrons)
                    .with("frozen_electrons", self.ledger.frozen_electrons)
                    .with("active_electrons", self.ledger.active_electrons)
                    .with(
                        "full_spin_electrons",
                        self.ledger.full_spin_electrons.to_vec(),
                    )
                    .with(
                        "frozen_spin_electrons",
                        self.ledger.frozen_spin_electrons.to_vec(),
                    )
                    .with("full_nuclear_charge", self.ledger.full_nuclear_charge),
            )
            .with("fragments", Value::Array(fragments))
            .with(
                "scope",
                Value::object()
                    .with("experimental", true)
                    .with("chemical_accuracy", "not_established")
                    .with("internal_polarization", false),
            )
    }
}
