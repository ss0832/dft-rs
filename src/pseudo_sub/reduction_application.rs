// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Bind immutable generated operators to a separately declared application.
//! Binding performs geometry/accounting/basis validation, never reference SCF
//! or coefficient generation. It does not establish chemical transferability.

use crate::driver::continuous_species::reduction_linear::{ElectronLedger, FrozenFragmentSpec};
use crate::pseudo_sub::reduction::ReducedFragment;
use crate::pseudo_sub::reduction_basis::build_reduction_basis;
use crate::pseudo_sub::reduction_geometry::{FragmentSpec, RigidExpansion};
use crate::system::{Atom, Molecule};
use crate::{DftError, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicationFingerprints {
    /// Exact complete canonical generated-record serialization.
    pub model: String,
    /// Original application input, including its unmodified H coordinates.
    pub source: String,
    /// Input/spec, rigid reference geometry, electron chart and model identity.
    pub application: String,
}

/// Immutable generated parameters and provenance remain in `candidate`. The
/// application has its own full electron count and therefore its own chart.
/// Read-only accessors prevent accidental parameter adjustment during binding.
#[derive(Clone, Debug)]
pub struct BoundFragment {
    candidate: ReducedFragment,
    model_serialization: String,
    application_source: Molecule,
    application_spec: FragmentSpec,
    geometry_map: RigidExpansion,
    electron_ledger: ElectronLedger,
    reduced_ao_count: usize,
    fingerprints: ApplicationFingerprints,
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn fingerprint(bytes: &[u8]) -> String {
    let mut h = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{h:016x}")
}

fn application_fingerprints(
    model: &str,
    source: &Molecule,
    spec: &FragmentSpec,
    map: &RigidExpansion,
    ledger: &ElectronLedger,
    nao: usize,
) -> ApplicationFingerprints {
    // Deterministic bit-sensitive identities, not security hashes.
    ApplicationFingerprints {
        model: fingerprint(model.as_bytes()),
        source: fingerprint(format!("{source:?}").as_bytes()),
        application: fingerprint(
            format!("methyl-application-v1|{model}|{source:?}|{spec:?}|{map:?}|{ledger:?}|{nao}")
                .as_bytes(),
        ),
    }
}

/// The H correspondence is explicit: construction `spec.explicit_atoms` order
/// excluding carbon maps onto application `spec.explicit_atoms` order excluding
/// carbon. No nearest-neighbor search, orbital fit or torsion reoptimization is
/// performed. Original application H positions are preserved as input evidence;
/// `rigid_reference_geometry()` returns the different constructed reference.
pub fn bind_methyl(
    candidate: &ReducedFragment,
    source: &Molecule,
    spec: FragmentSpec,
) -> Result<BoundFragment> {
    candidate.validate()?;
    let model_serialization = candidate.to_json()?;
    if source.multiplicity != 1 {
        return invalid("initial methyl application requires a closed-shell singlet source");
    }
    // This validates C3H, disjoint full partition, one center and a complete
    // C(sp3)-C single port, including the existing strict frame guard.
    let _validated = RigidExpansion::methyl(source, spec.clone())?;
    let application_h: Vec<_> = spec
        .explicit_atoms
        .iter()
        .copied()
        .filter(|&i| i != spec.port.fragment_atom)
        .collect();
    let original_offsets = candidate.geometry_map.internal_offsets();
    if original_offsets.len() != 3 || application_h.len() != 3 {
        return invalid("methyl application requires one fixed ordered C3H internal template");
    }
    let offsets: Vec<_> = application_h
        .into_iter()
        .zip(original_offsets.iter().map(|(_, x)| *x))
        .collect();
    let geometry_map =
        RigidExpansion::methyl_with_internal_offsets(source, spec.clone(), &offsets)?;
    let ledger = ElectronLedger::new(
        &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        source.n_electrons()?,
        &[FrozenFragmentSpec {
            explicit_atoms: spec.explicit_atoms.clone(),
            frozen_electrons: 8,
        }],
    )?;
    let rigid = geometry_map.reference_molecule();
    let mut atoms: Vec<_> = spec
        .retained_atoms
        .iter()
        .map(|&i| rigid.atoms[i].clone())
        .collect();
    // Deliberate electronic role label for the selected fixed AO template;
    // geometry anchors acquire no nuclear or electronic roles automatically.
    atoms.push(Atom {
        z: 1,
        position: rigid.atoms[spec.port.fragment_atom].position,
    });
    let reduced_ao_count = build_reduction_basis(
        &Molecule::new(atoms),
        spec.retained_atoms.len(),
        candidate.options.basis_template,
    )?
    .nao_sph;
    if ledger.active_electron_count == 0
        || ledger.active_electron_count % 2 != 0
        || ledger.active_electron_count / 2 > reduced_ao_count
        || ledger.fragments[0].effective_ionic_charge != 1
        || ledger.full_total_charge != i64::from(source.charge)
    {
        return invalid("application requires positive even active electrons fitting the fixed AO template and conserved molecular charge");
    }
    let fingerprints = application_fingerprints(
        &model_serialization,
        source,
        &spec,
        &geometry_map,
        &ledger,
        reduced_ao_count,
    );
    let result = BoundFragment {
        candidate: candidate.clone(),
        model_serialization,
        application_source: source.clone(),
        application_spec: spec,
        geometry_map,
        electron_ledger: ledger,
        reduced_ao_count,
        fingerprints,
    };
    result.validate()?;
    Ok(result)
}

impl BoundFragment {
    pub fn candidate(&self) -> &ReducedFragment {
        &self.candidate
    }
    pub fn original_model_json(&self) -> &str {
        &self.model_serialization
    }
    pub fn application_source(&self) -> &Molecule {
        &self.application_source
    }
    pub fn application_spec(&self) -> &FragmentSpec {
        &self.application_spec
    }
    pub fn geometry_map(&self) -> &RigidExpansion {
        &self.geometry_map
    }
    pub fn rigid_reference_geometry(&self) -> &Molecule {
        self.geometry_map.reference_molecule()
    }
    pub fn electron_ledger(&self) -> &ElectronLedger {
        &self.electron_ledger
    }
    pub fn reduced_ao_count(&self) -> usize {
        self.reduced_ao_count
    }
    pub fn fingerprints(&self) -> &ApplicationFingerprints {
        &self.fingerprints
    }

    /// No SCF/regeneration. Complete canonical serialization guards all original
    /// parameters, E0 and provenance, rather than only construction inputs.
    pub fn validate(&self) -> Result<()> {
        if self.candidate.to_json()? != self.model_serialization {
            return invalid("bound application changed the original serialized model");
        }
        let spec = &self.application_spec;
        RigidExpansion::methyl(&self.application_source, spec.clone())?;
        if self.geometry_map.fragment_spec() != Some(spec) {
            return invalid("bound application map/spec mismatch");
        }
        let expected_h: Vec<_> = spec
            .explicit_atoms
            .iter()
            .copied()
            .filter(|&i| i != spec.port.fragment_atom)
            .collect();
        let offsets = self.geometry_map.internal_offsets();
        if offsets.len() != 3 || self.candidate.geometry_map.internal_offsets().len() != 3 {
            return invalid("bound application internal template length mismatch");
        }
        for ((&(id, offset), &expected), &(_, original)) in offsets
            .iter()
            .zip(&expected_h)
            .zip(self.candidate.geometry_map.internal_offsets())
        {
            if id != expected
                || (0..3).any(|axis| offset.get(axis).to_bits() != original.get(axis).to_bits())
            {
                return invalid(
                    "bound application did not preserve exact internal template coefficients",
                );
            }
        }
        let ledger = ElectronLedger::new(
            &self
                .application_source
                .atoms
                .iter()
                .map(|a| a.z)
                .collect::<Vec<_>>(),
            self.application_source.n_electrons()?,
            &[FrozenFragmentSpec {
                explicit_atoms: spec.explicit_atoms.clone(),
                frozen_electrons: 8,
            }],
        )?;
        if ledger != self.electron_ledger
            || self.fingerprints
                != application_fingerprints(
                    &self.model_serialization,
                    &self.application_source,
                    spec,
                    &self.geometry_map,
                    &ledger,
                    self.reduced_ao_count,
                )
        {
            return invalid("bound application source/ledger/fingerprint mismatch");
        }
        Ok(())
    }
}
