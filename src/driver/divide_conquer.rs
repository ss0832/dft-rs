// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Divide-and-conquer energy and gradient routed through the common driver.

use super::controls::Controls;
use super::method::MolecularReference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::math::Vec3;
use crate::scf::divide_conquer::{run_divide_conquer, DivideConquerOptions};
use crate::{DftError, Result};

pub(super) fn divide_conquer(system: &System, controls: &Controls) -> Result<Value> {
    let config = super::vibrational::response_config(controls)?;
    let with_gradient = controls.bool("gradient", true)?;
    let fragments = controls.opt_usize_list("dc_fragments")?;
    let buffer = controls.f64("dc_buffer", 8.0)?;
    let smearing = controls.f64("dc_smearing", 1.0e-3)?;
    let setup = setup(system, controls)?;
    controls.finish("dc")?;
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "divide-and-conquer SCF with the SMD non-electrostatic term".into(),
            missing: "the continuum terms of the divide-and-conquer Kohn-Sham build".into(),
        });
    }
    let natoms = setup.molecule.len();
    // One fragment per atom unless the fragments are given.
    let fragment_of = fragments.unwrap_or_else(|| (0..natoms).collect());
    // The smearing of the method, when it has one, is the Fermi function of
    // the subsystems.
    let kt = if setup.options.electronic_temperature > 0.0 {
        setup.options.electronic_temperature
    } else {
        smearing
    };
    let options = DivideConquerOptions {
        fragment_of,
        buffer,
        kt,
    };
    let result = run_divide_conquer(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        &options,
    )?;
    if !result.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: result.scf.iterations,
            error: result.density_residual,
        });
    }
    let relaxed = if with_gradient {
        Some(crate::scf::divide_conquer::divide_conquer_gradient(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            &options,
            &result,
            &config,
        )?)
    } else {
        None
    };
    let e_dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    let internal_energy = result.internal_energy;
    let density_residual = result.density_residual;
    let sizes: Vec<[usize; 3]> = result
        .subsystems
        .iter()
        .map(|part| {
            [
                part.core_atoms.len(),
                part.buffer_atoms.len(),
                part.functions.len(),
            ]
        })
        .collect();
    let reference = MolecularReference {
        scf: result.scf,
        e_dispersion,
        e_smd_cds: 0.0,
        following: None,
    };
    let mut report = reference_report(
        &setup,
        &reference,
        "divide-and-conquer Kohn-Sham: subsystem orbitals with a common Fermi level and \
         partition-weighted density assembly",
    )
    .with("reference_type", "divide-and-conquer")
    .with("dc_buffer", buffer)
    .with("dc_smearing", kt)
    .with("fragment_of", options.fragment_of.clone())
    .with("subsystem_sizes", sizes)
    .with("internal_energy", internal_energy + e_dispersion)
    .with("electronic_entropy", reference.scf.electronic_entropy)
    .with(
        "chemical_potentials",
        reference.scf.chemical_potentials.to_vec(),
    )
    .with("density_residual", density_residual)
    .with("provenance", "analytic");
    let mut unit_pairs = vec![
        (
            "energy",
            "hartree (free energy: internal energy minus kT S)",
        ),
        ("energy_components", "hartree"),
        ("dc_buffer", "bohr"),
        ("dc_smearing", "hartree"),
        ("internal_energy", "hartree"),
        ("electronic_entropy", "dimensionless (S / k_B)"),
        ("chemical_potentials", "hartree"),
        ("density_residual", "dimensionless"),
    ];
    if let Some((mut gradient, relaxation_residual)) = relaxed {
        report.set("relaxation_residual", relaxation_residual);
        unit_pairs.push(("relaxation_residual", "dimensionless"));
        if setup.dispersion != crate::dispersion::Dispersion::None {
            let correction = crate::dispersion::dispersion_gradient(
                &setup.molecule,
                &setup.dispersion,
                &setup.method.dispersion_functional_name(),
            )?;
            for (total, delta) in gradient.iter_mut().zip(correction) {
                *total += delta;
            }
        }
        let mut sum = Vec3::zero();
        for g in &gradient {
            sum += *g;
        }
        report.set(
            "gradient",
            gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
        );
        report.set("translation_residual", sum.norm());
        unit_pairs.push(("gradient", "hartree/bohr"));
        unit_pairs.push(("translation_residual", "hartree/bohr"));
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}
