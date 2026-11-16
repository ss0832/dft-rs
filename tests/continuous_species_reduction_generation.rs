// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Source-local candidate-generation gates against the accepted library.
pub use dft_rs::{
    basis, dual, grid, integrals, ks, linalg, math, properties, response, scf, system, xc,
    DftError, Result,
};
#[path = "../src/driver/continuous_species/gaussian.rs"]
pub mod gaussian;
#[path = "../src/pseudo_sub/reduction_basis.rs"]
pub mod reduction_basis;
#[path = "../src/pseudo_sub/reduction_geometry.rs"]
pub mod reduction_geometry;
#[path = "../src/pseudo_sub/reduction_kernels.rs"]
pub mod reduction_kernels;
#[path = "../src/driver/continuous_species/reduction_linear.rs"]
pub mod reduction_linear;
pub use dft_rs::pseudo_sub::reduction_reference;
pub mod reduction_support {
    pub use dft_rs::pseudo_sub::reduction_support::*;
}
pub mod pseudo_sub {
    pub use crate::{
        reduction_basis, reduction_geometry, reduction_kernels, reduction_reference,
        reduction_support,
    };
    pub use dft_rs::pseudo_sub::*;
}
pub mod driver {
    pub use dft_rs::driver::*;
    pub mod continuous_species {
        pub use crate::{gaussian, reduction_linear};
    }
}
#[path = "../src/pseudo_sub/reduction.rs"]
pub mod reduction;

use math::Vec3;
use reduction::{generate_methyl, GenerationOptions, ReducedFragment};
use reduction_geometry::{AttachmentPort, FragmentSpec};
use system::{Atom, Molecule};

fn molecule() -> Molecule {
    let positions = [
        [-1.46, -0.14, 0.23],
        [-2.13, 1.69, 0.68],
        [-2.28, -1.57, 1.31],
        [-2.19, -0.43, -1.65],
        [1.35, 0.17, -0.12],
        [2.05, 2.08, 0.16],
        [2.10, -0.92, 1.45],
        [2.01, -0.84, -1.67],
    ];
    Molecule::new(
        positions
            .iter()
            .enumerate()
            .map(|(i, p)| Atom {
                z: if i == 0 || i == 4 { 6 } else { 1 },
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}
fn spec() -> FragmentSpec {
    FragmentSpec {
        explicit_atoms: vec![4, 5, 6, 7],
        retained_atoms: vec![0, 1, 2, 3],
        port: AttachmentPort {
            fragment_atom: 4,
            host_atom: 0,
            reference_atom: 1,
        },
        single_bonds: vec![[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
        electronic_centers: 1,
    }
}

#[test]
fn methyl_candidate_projection_accounting_determinism_and_roundtrip() {
    let mol = molecule();
    let options = GenerationOptions::default();
    let candidate = generate_methyl(&mol, spec(), options.clone()).unwrap();
    assert_eq!(
        candidate.construction_fingerprint,
        "fnv1a64:d5f4cc45da1c31c1"
    );
    candidate.validate().unwrap();
    let ledger = &candidate.electron_ledger;
    assert_eq!(
        (
            ledger.full_electron_count,
            ledger.frozen_electron_count,
            ledger.active_electron_count
        ),
        (18, 8, 10)
    );
    assert_eq!(ledger.fragments[0].effective_ionic_charge, 1);
    assert_eq!(ledger.reduced_total_charge, 0);
    let d = &candidate.diagnostics;
    assert_eq!((d.full_ao_count, d.reduced_ao_count), (16, 9));
    assert!(d.sigma_charge_defect < 1e-9 && d.sigma_dipole_defect < 1e-9);
    assert!(d.reference_energy_reconstruction_error < 1e-12);
    assert_eq!(d.sigma_projection.free_parameters, 8);
    assert_eq!(d.electronic_projection.free_parameters, 39);
    assert_eq!(d.sigma_projection.regularized_rank, 8);
    assert_eq!(d.electronic_projection.regularized_rank, 39);
    assert!(d.sigma_projection.weighted_residual_squared.is_finite());
    assert!(d
        .electronic_projection
        .weighted_residual_squared
        .is_finite());
    println!("generated fullAO={} reducedAO={} Ne={} frozen={} active={} Qion={} Efull={:.15} E0={:.15} reconstruction={:.9e} sigmaRank={}/{} electronicRank={}/{} sigmaResidual2={:.9e} electronicResidual2={:.9e}",
        d.full_ao_count,d.reduced_ao_count,ledger.full_electron_count,ledger.frozen_electron_count,ledger.active_electron_count,
        ledger.fragments[0].effective_ionic_charge,d.full_reference_energy,candidate.internal_energy_reference,
        d.reference_energy_reconstruction_error,d.sigma_projection.data_rank,d.sigma_projection.free_parameters,
        d.electronic_projection.data_rank,d.electronic_projection.free_parameters,
        d.sigma_projection.weighted_residual_squared,d.electronic_projection.weighted_residual_squared);
    let json = candidate.to_json().unwrap();
    let restored = ReducedFragment::from_json(&json).unwrap();
    assert_eq!(json, restored.to_json().unwrap());
    let second = generate_methyl(&mol, spec(), options).unwrap();
    assert_eq!(json, second.to_json().unwrap());
    let coordinates = restored.geometry_map.reference_coordinates();
    let expanded = restored.geometry_map.expand(&coordinates).unwrap();
    for (a, b) in expanded.atoms.iter().zip(&mol.atoms) {
        assert_eq!(a.z, b.z);
        assert!((a.position - b.position).norm() < 1e-14);
    }
    let kin = restored
        .geometry_map
        .body_frame_kinematics(&coordinates)
        .unwrap()
        .unwrap();
    let sites = restored
        .gaussian_sites(kin.center, kin.orientation)
        .unwrap();
    let (q, _) = reduction_kernels::charge_moments(&sites, kin.center).unwrap();
    assert!(q.abs() < 1e-9);
    assert_eq!(restored.shape_status(), "not-generated");
    assert!(restored.candidate_status().starts_with("candidate"));
    assert_eq!(
        restored.response_status(),
        "active-only-no-internal-polarization"
    );
    for (key, bad) in [
        ("model_version", driver::Value::Str("unknown".into())),
        ("source_version", driver::Value::Str("0.2.0".into())),
        ("nonlocal_coefficients", driver::Value::Array(Vec::new())),
        ("status", driver::Value::Str("validated".into())),
        (
            "ledger",
            driver::Value::Array(vec![
                driver::Value::Int(18),
                driver::Value::Int(8),
                driver::Value::Int(12),
            ]),
        ),
    ] {
        let mut value = candidate.to_value().unwrap();
        value.set(key, bad);
        assert!(
            ReducedFragment::from_value(&value).is_err(),
            "accepted bad {key}"
        );
    }
    let mut invalid = candidate.clone();
    invalid.gaussian_charges[0] += 0.1;
    assert!(invalid.validate().is_err());
    // Loading and production use the same coefficient-scaled neutrality gate;
    // the older absolute 1e-9 bound admitted records the provider rejected.
    let mut invalid = candidate.clone();
    let scale = invalid
        .gaussian_charges
        .iter()
        .map(|q| q.abs())
        .sum::<f64>()
        .max(1.0);
    invalid.gaussian_charges[0] += 1024.0 * f64::EPSILON * scale;
    assert!(invalid.validate().is_err());
    let mut invalid = candidate.clone();
    invalid.options.local_exponents[0] *= 2.0;
    assert!(invalid.validate().is_err());
}

#[test]
fn invalid_generation_does_not_activate_named_parser() {
    let options = GenerationOptions {
        functional: "hybrid".into(),
        ..GenerationOptions::default()
    };
    assert!(generate_methyl(&molecule(), spec(), options).is_err());
    let mut fragment = spec();
    fragment.explicit_atoms.pop();
    assert!(generate_methyl(&molecule(), fragment, GenerationOptions::default()).is_err());
}

#[test]
fn fixed_basis_and_projection_choices_roundtrip_without_extra_coefficients() {
    use reduction::{ProjectionRule, TEMPLATE_ID};
    use reduction_basis::OneCenterBasisTemplate;
    let mol = molecule();
    let original = generate_methyl(&mol, spec(), GenerationOptions::default()).unwrap();
    assert_eq!(original.template_id, TEMPLATE_ID);
    let valence = generate_methyl(
        &mol,
        spec(),
        GenerationOptions {
            basis_template: OneCenterBasisTemplate::CarbonValenceSp,
            ..GenerationOptions::default()
        },
    )
    .unwrap();
    assert_eq!(valence.diagnostics.reduced_ao_count, 12);
    assert_eq!(valence.electron_ledger, original.electron_ledger);
    assert_eq!(valence.local_amplitudes, original.local_amplitudes);
    assert_eq!(
        valence.nonlocal_coefficients,
        original.nonlocal_coefficients
    );
    assert_eq!(valence.gaussian_charges, original.gaussian_charges);
    assert_eq!(
        valence.internal_energy_reference,
        original.internal_energy_reference
    );
    assert_ne!(
        valence.construction_fingerprint,
        original.construction_fingerprint
    );
    let normalized = generate_methyl(
        &mol,
        spec(),
        GenerationOptions {
            basis_template: OneCenterBasisTemplate::CarbonValenceSp,
            projection_rule: ProjectionRule::PhysicalSupportPauliNormalized,
            ..GenerationOptions::default()
        },
    )
    .unwrap();
    assert_eq!(normalized.diagnostics.reduced_ao_count, 12);
    assert_eq!(
        normalized.local_amplitudes.len() + normalized.nonlocal_coefficients.len(),
        39
    );
    assert_eq!(
        normalized.gaussian_charges.len() + normalized.gaussian_body_dipoles.len() * 3,
        12
    );
    assert!(normalized.diagnostics.reference_energy_reconstruction_error < 1e-12);
    let core = generate_methyl(
        &mol,
        spec(),
        GenerationOptions {
            basis_template: OneCenterBasisTemplate::CarbonCoreValenceSp,
            ..GenerationOptions::default()
        },
    )
    .unwrap();
    assert_eq!(core.diagnostics.reduced_ao_count, 13);
    assert_eq!(core.electron_ledger, original.electron_ledger);
    assert_eq!(core.local_amplitudes, original.local_amplitudes);
    assert_eq!(core.nonlocal_coefficients, original.nonlocal_coefficients);
    assert_eq!(core.gaussian_charges, original.gaussian_charges);
    assert_eq!(
        core.internal_energy_reference,
        original.internal_energy_reference
    );
    assert_ne!(
        core.construction_fingerprint,
        valence.construction_fingerprint
    );
    for model in [original, valence, normalized, core] {
        let json = model.to_json().unwrap();
        let restored = ReducedFragment::from_json(&json).unwrap();
        assert_eq!(
            restored.options.basis_template,
            model.options.basis_template
        );
        assert_eq!(
            restored.options.projection_rule,
            model.options.projection_rule
        );
        assert_eq!(restored.to_json().unwrap(), json);
        let mut bad = model.to_value().unwrap();
        if let driver::Value::Object(fields) = &mut bad {
            if let Some((_, options)) = fields.iter_mut().find(|(key, _)| key == "options") {
                options.set("projection_rule", "unknown-rule");
            }
        }
        assert!(ReducedFragment::from_value(&bad).is_err());
    }
}
