// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular ground-state observables routed through the common driver.

use super::controls::Controls;
use super::method::{
    prepare_molecular, run_collinear_property_reference, run_molecular_reference,
    CollinearPropertyReference, MolecularMethod, MolecularReference, MolecularSetup,
};
use super::value::Value;
use super::System;
use crate::constants::{atomic_mass_au, AU_DIPOLE_TO_DEBYE};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use crate::{DftError, Result};

fn check_epr_reference(id: &str, controls: &Controls) -> Result<()> {
    // Inspect without consuming the selector or preparing an SCF reference.
    match controls.peek("reference") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Str(reference)) => match reference.to_ascii_lowercase().as_str() {
            "auto" | "rks" | "rhf" | "uks" | "uhf" => Ok(()),
            "roks" | "rohf" | "gks" | "ghf" => Err(DftError::Unsupported {
                feature: format!("{id} with reference={reference}"),
                missing: "a validated reference-specific EPR property connection; an automatic collinear reference cannot replace the requested reference".into(),
            }),
            _ => Err(DftError::InvalidInput(format!(
                "unknown reference `{reference}` for {id}; supported selectors are auto, rks, rhf, uks and uhf"
            ))),
        },
        Some(_) => Err(DftError::InvalidInput(format!(
            "reference for {id} must be a string selector"
        ))),
    }
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if system.is_periodic() {
        return Err(DftError::Unsupported {
            feature: format!("periodic `{id}`"),
            missing: "a periodic implementation of this molecular calculation; remove the cell or the periodic axes to request an isolated molecule".into(),
        });
    }
    super::density_dispersion::check_scope(id, controls)?;
    if id == "spin-spin-zfs" {
        // Preserve this dedicated UHF selector before generic alias translation.
        return super::spin_spin_zfs::run(system, controls);
    }
    if id == "dft-plus-u" {
        return super::hubbard::run(system, controls);
    }
    if id == "pz-sic" {
        return super::sic::run(system, controls);
    }
    if id == "flo-sic-fixed" {
        return super::sic::run_fixed_flo(system, controls);
    }
    if id == "g-tensor" {
        let order = super::gtensor_nuclear::order(controls)?;
        if order > 0 {
            return super::gtensor_nuclear::run(system, controls, order);
        }
    }
    let reference = controls.peek_str("reference")?.map(str::to_ascii_lowercase);
    if reference.is_none() {
        // An explicit null selector requests the default, rather than an
        // unhandled reference. Absent controls remain absent.
        let _ = controls.take("reference");
    }
    // Saved few-center models own their fixed reference sector; do not rewrite
    // their controls into ordinary unrestricted/reference defaults.
    if id == "continuous-species"
        && matches!(
            controls.str("fragment_mode", "")?.as_str(),
            "few-evaluate"
                | "rigid-evaluate"
                | "scalar-evaluate"
                | "scalar-optimize"
                | "scalar-gradient"
                | "scalar-geometry-optimize"
                | "scalar-hessian"
        )
    {
        return super::continuous_species::calculate(system, controls);
    }
    if let Some(kind) = reference.as_deref() {
        if !matches!(
            kind,
            "auto" | "rks" | "rhf" | "uks" | "uhf" | "roks" | "rohf" | "gks" | "ghf"
        ) {
            return Err(DftError::InvalidInput(format!(
                "unknown reference `{kind}`"
            )));
        }
    }
    if id == "hyperfine" {
        // Validate the nuclear sector before static dispatch translates reference controls.
        let _ = super::hyperfine_nuclear::NuclearOptions::parse(controls, system.numbers.len())?;
    }
    if matches!(id, "hyperfine" | "g-tensor") {
        if reference
            .as_deref()
            .is_some_and(|kind| matches!(kind, "roks" | "rohf" | "gks" | "ghf"))
        {
            return super::epr::run(id, system, controls);
        }
        // Validate before reference consumption or SCF setup. These EPR
        // properties currently use only the ordinary collinear reference.
        check_epr_reference(id, controls)?;
    }
    if matches!(id, "frequencies" | "gibbs-free-energy")
        && (controls.contains("vibrational_projection")
            || controls.contains("electronic_degeneracy")
            || super::double_hybrid::requested(controls)?
            || reference
                .as_deref()
                .is_some_and(|reference| matches!(reference, "roks" | "rohf" | "gks" | "ghf")))
    {
        return super::reference_vibrations::run(id, system, controls);
    }
    if let Some(kind) = reference.as_deref() {
        if matches!(id, "mermin-free-energy" | "free-energy")
            && matches!(kind, "roks" | "rohf")
        {
            let translated = controls.excluding(&["reference"]);
            let mut report = super::roks::mermin(system, &translated)?;
            report.set("reference_type", "roks");
            report.set("requested_reference", kind);
            return Ok(report);
        }
        if id == "magnetizability" && matches!(kind, "gks" | "ghf") {
            return super::gks::run(id, system, controls);
        }
        if matches!(kind, "roks" | "rohf" | "gks" | "ghf")
            && matches!(
                id,
                "magnetizability"
                    | "hypermagnetizability"
                    | "cotton-mouton"
                    | "magnetoelectric"
                    | "magneto-optical"
                    | "bfield"
                    | "bfield-gradient"
            )
        {
            return Err(DftError::Unsupported {
                feature: format!("magnetic response with reference={kind}"),
                missing: "reference-specific common-orbital constraints or spinor self-consistency and magnetic Hamiltonian derivatives; an automatic collinear reference cannot replace the requested reference".into(),
            });
        }
        if matches!(kind, "rks" | "rhf" | "uks" | "uhf" | "auto")
            && !matches!(id, "charges" | "orbitals")
        {
            if kind != "auto" && matches!(id, "roks" | "roks-hessian" | "roks-polarizability") {
                return Err(DftError::Unsupported {
                    feature: format!("{id} with reference={kind}"),
                    missing: "a matching reference selector; use roks or rohf for this dedicated calculation".into(),
                });
            }
            let mut translated = controls.to_value();
            if let Value::Object(entries) = &mut translated {
                entries.retain(|(k, _)| k != "reference");
            }
            if kind != "auto" {
                let unrestricted = matches!(kind, "uks" | "uhf");
                if !unrestricted && controls.usize("multiplicity", 1)? != 1 {
                    return Err(DftError::InvalidInput(
                        "a restricted closed-shell reference requires multiplicity=1".into(),
                    ));
                }
                translated.set("unrestricted", unrestricted);
            }
            let mut report = run(id, system, &Controls::from_value(&translated)?)?;
            if kind != "auto"
                || !matches!(
                    id,
                    "constrained-dft" | "car-parrinello-md" | "magnetic-quadrupole"
                )
            {
                report.set("reference_type", kind);
            }
            if matches!(
                id,
                "constrained-dft" | "car-parrinello-md" | "magnetic-quadrupole"
            ) {
                report.set("requested_reference", kind);
            }
            return Ok(report);
        }
        if matches!(kind, "roks" | "rohf")
            && matches!(
                id,
                "energy"
                    | "gradient"
                    | "hessian"
                    | "polarizability"
                    | "force-constants"
                    | "static-spectroscopy"
                    | "beta"
                    | "gamma"
                    | "roks"
                    | "roks-hessian"
                    | "roks-polarizability"
            )
        {
            if id == "energy" && super::double_hybrid::requested(controls)? {
                // Keep the selected common-orbital parent through recipe
                // resolution and the already shared semicanonical PT2 driver.
                return super::double_hybrid::energy(system, controls);
            }
            let mut translated = controls.to_value();
            if let Value::Object(entries) = &mut translated {
                entries.retain(|(k, _)| k != "reference");
            }
            if matches!(id, "energy" | "gradient") {
                translated.set("gradient", id == "gradient");
            }
            let translated = Controls::from_value(&translated)?;
            let mut report = match id {
                "energy" | "gradient" | "roks" => super::roks::roks(system, &translated)?,
                "hessian" | "roks-hessian" => super::roks::roks_hessian(system, &translated)?,
                "force-constants" => super::roks::force_constants(system, &translated)?,
                "static-spectroscopy" | "beta" | "gamma" => {
                    super::roks::electric_tensor(id, system, &translated)?
                }
                _ => super::roks::roks_polarizability(system, &translated)?,
            };
            if id == "gradient" {
                if let Some(g) = report.get("gradient") {
                    let rows = g.as_array().expect("ROKS gradient rows");
                    let forces: Vec<Vec<f64>> = rows
                        .iter()
                        .map(|r| {
                            r.as_array()
                                .expect("three Cartesian entries")
                                .iter()
                                .map(|v| -v.as_f64().expect("gradient value"))
                                .collect()
                        })
                        .collect();
                    report.set("forces", forces);
                }
            }
            report.set("reference_type", "roks");
            report.set("requested_reference", kind);
            return Ok(report);
        }
    }
    // Validate the electric-response model before delegating reference-specific
    // kernels. Otherwise spinor selection bypasses its thermal capability gate.
    if id == "dynamic-polarizability" {
        return super::polarizability::run(system, controls);
    }
    if [
        "energy",
        "mermin-free-energy",
        "free-energy",
        "charges",
        "orbitals",
        "natural-bond",
        "igmh",
        "dipole",
        "molecular-multipole",
        "quadrupole-polarizability",
        "multipole-polarizability",
        "gradient",
        "hessian",
        "polarizability",
        "force-constants",
        "spin-spin-couplings",
        "fermi-contact-couplings",
        "nmr-shielding",
        "nmr-spectrum",
        "static-spectroscopy",
        "beta",
        "gamma",
    ]
    .contains(&id)
        && reference
            .as_deref()
            .is_some_and(|r| r.eq_ignore_ascii_case("gks") || r.eq_ignore_ascii_case("ghf"))
    {
        return super::gks::run(id, system, controls);
    }
    match id {
        "constrained-dft" => super::constrained::run(system, controls),
        "nexafs" => super::nexafs::run(system, controls),
        "magnetic-quadrupole" => super::magnetic_quadrupole::run(system, controls),
        "car-parrinello-md" => super::car_parrinello::run(system, controls),
        "force-constants" => super::force_constants::run(system, controls),
        "static-spectroscopy" => super::static_spectroscopy::run(system, controls),
        "gradient" | "hessian" if super::double_hybrid::requested(controls)? => {
            super::correlated_nuclear::run(id, system, controls)
        }
        "energy" => energy(system, controls),
        "mermin-free-energy" | "free-energy" => mermin_free_energy(system, controls),
        "gradient" => gradient(system, controls),
        "streamed-gradient" => streamed_gradient(system, controls),
        "charges" => charges(system, controls),
        "natural-bond" => super::natural_bond::natural_bond(system, controls),
        "igmh" => super::igmh::run(system, controls),
        "natural-shielding" => super::natural_bond::natural_shielding(system, controls),
        "natural-coupling" => super::natural_bond::natural_coupling(system, controls),
        "natural-energy-decomposition" => {
            super::natural_bond::natural_energy_decomposition(system, controls)
        }
        "sapt-analysis" => super::sapt::sapt_analysis(system, controls),
        "dipole" | "molecular-multipole" => electric_moments(id, system, controls),
        "quadrupole-polarizability" => super::quadrupole::run(system, controls),
        "multipole-polarizability" => super::quadrupole::run_multipole(system, controls),
        "orbitals" => orbitals(system, controls),
        "molden" => molden(system, controls),
        "orbitals-save" => orbitals_save(system, controls),
        "field-taylor" | "field-mixed" | "beta" | "gamma" => {
            super::electric::run(id, system, controls)
        }
        "conceptual-dft" | "spin-conceptual-dft" => super::conceptual::run(id, system, controls),
        "excitations"
        | "ecd-spectrum"
        | "td-state-gradient"
        | "td-hessian"
        | "td-ecd-giao"
        | "xanes"
        | "franck-condon"
        | "excited-optimization" => super::excited::run(id, system, controls),
        "hyperfine" => super::hyperfine::hyperfine(system, controls),
        "nmr-shielding" | "nmr-spectrum" => super::nmr::run(id, system, controls),
        "spin-spin-couplings" | "fermi-contact-couplings" => {
            super::nmr::spin_spin(id, system, controls)
        }
        "exafs" => super::exafs::exafs(system, controls),
        "magnetizability" => super::magnetizability::magnetizability(system, controls),
        "magneto-optical" => super::magnetizability::magneto_optical(system, controls),
        "bfield" | "bfield-gradient" => super::magnetizability::finite_field(id, system, controls),
        "magnetoelectric" => super::magnetizability::magnetoelectric(system, controls),
        "hypermagnetizability" | "cotton-mouton" => {
            super::magnetizability::hypermagnetizability(id, system, controls)
        }
        "optical-activity" => super::optical::optical_activity(system, controls),
        "molecular-medium-optics" => super::medium_optics::run(system, controls),
        "roa" => super::optical::roa(system, controls),
        "broken-symmetry-exchange" => super::exchange::broken_symmetry_exchange(system, controls),
        "vcd" => super::vcd::vcd(system, controls),
        "spin-susceptibility" => super::spin_response::spin_susceptibility(system, controls),
        "g-tensor" => super::gtensor::g_tensor(system, controls),
        "roks" => super::roks::roks(system, controls),
        "roks-hessian" => super::roks::roks_hessian(system, controls),
        "roks-polarizability" => super::roks::roks_polarizability(system, controls),
        "dc" => super::divide_conquer::divide_conquer(system, controls),
        "multicomponent" => super::multicomponent::run(system, controls),
        "continuous-species" => super::continuous_species::calculate(system, controls),
        "casscf" => super::mcpdft::casscf(system, controls),
        "mcpdft" => super::mcpdft::mcpdft(system, controls),
        "mrsf" => super::mrsf::mrsf(system, controls),
        "mrsf-state" | "mrsf-hessian" => super::mrsf::mrsf_surface(id, system, controls),
        "magnon" => super::magnon::magnon(system, controls),
        "mcd" => super::mcd::mcd(system, controls),
        "spin-orbit" | "magnetic-anisotropy" => super::spinorbit::run(id, system, controls),
        "spinor-polarizability" => super::spinorbit::spinor_polarizability(system, controls),
        "hessian"
        | "frequencies"
        | "gibbs-free-energy"
        | "ir-raman"
        | "polarizability"
        | "multipole-gradient"
        | "response-save"
        | "third-derivative-2n1"
        | "cubic-fc-along"
        | "quartic-fc-along" => super::vibrational::run(id, system, controls),
        other => Err(DftError::Unsupported {
            feature: format!("molecular calculation `{other}`"),
            missing: "a common-driver implementation".into(),
        }),
    }
}

pub(super) fn setup(system: &System, controls: &Controls) -> Result<MolecularSetup> {
    controls.require_consumed("reference", "molecular reference setup")?;
    let method = MolecularMethod::from_controls(controls)?;
    prepare_molecular(&system.numbers, &system.positions, &method)
}

/// Shared reference fields: model, convergence and the energy decomposition.
pub(super) fn reference_report(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    model: &str,
) -> Value {
    let scf = &reference.scf;
    let mut warnings: Vec<String> = Vec::new();
    if !scf.converged {
        warnings.push("WARNING [scf] the reference SCF did not converge".into());
    }
    let occupation_gaps = scf.occupied_virtual_gaps();
    if occupation_gaps.iter().flatten().any(|&gap| gap < 0.0) {
        warnings.push("WARNING [scf] negative occupied/virtual gap: stationary non-Aufbau determinant; convergence does not certify orbital stability or the electronic ground state".into());
    }
    let mut report = Value::object()
        .with("model", model)
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("converged", scf.converged)
        .with("scf_iterations", scf.iterations)
        .with("scf_orbital_residual", reference_orbital_residual(scf))
        .with("unrestricted", scf.unrestricted)
        .with("n_alpha", scf.n_alpha)
        .with("n_beta", scf.n_beta)
        .with("s_squared", scf.s_squared)
        .with("occupied_virtual_gaps", occupation_gaps.to_vec())
        .with("energy", reference.total_energy())
        .with(
            "energy_components",
            Value::object()
                .with("electronic", scf.e_elec)
                .with("nuclear_repulsion", scf.e_nuc)
                .with("exchange_correlation", scf.e_xc)
                .with("nonlocal_correlation", scf.e_nlc)
                .with("continuum_electrostatic", scf.e_pcm)
                .with("dispersion", reference.e_dispersion)
                .with("smd_cds", reference.e_smd_cds),
        );
    super::density_dispersion::report_d1(
        setup,
        reference.e_dispersion,
        || reference_orbital_residual(scf),
        &mut report,
    );
    if let Some(residual) = scf.fixed_occupation_residual {
        report.set("occupation_model", "fixed per-spin orbital populations in ascending physical Fock energy order; orbital-stationary constrained density functional; no thermal or ensemble-XC weight optimization");
        report.set("occupations", scf.occupations.to_vec());
        report.set("fixed_occupation_residual", residual);
        let mut gap: Option<f64> = None;
        for spin in 0..2 {
            let f = &scf.occupations[spin];
            let e = &scf.mo_energy[spin];
            for i in 0..f.len() {
                for j in 0..i {
                    if f[i] != f[j] {
                        let difference = (e[i] - e[j]).abs();
                        gap = Some(gap.map_or(difference, |g| g.min(difference)));
                    }
                }
            }
        }
        report.set("fixed_occupation_gap", gap);
        if gap == Some(0.0) {
            warnings.push("WARNING [fixed populations] unequal populations in an exactly degenerate Fock space do not define a unique energy-ordered density or differentiable branch".into());
        }
        report.set("spin_square_model", "independent orbital-occupation ensemble with Wick number fluctuations; not a spin-projected or canonical multiplet ensemble");
    }
    if let Some(tracking) = &scf.occupation_tracking {
        report.set("occupation_tracking", Value::object()
            .with("mode", tracking.mode.id())
            .with("model", "state-specific integer delta-SCF; squared occupied-projector overlap in the physical AO metric; occupied block first; no spin projection")
            .with("initial_energy", tracking.initial_energy)
            .with("energy_difference", scf.energy - tracking.initial_energy)
            .with("occupied_indices", tracking.occupied_indices.iter().map(|v| v.iter().map(|&i| i as i64).collect::<Vec<_>>()).collect::<Vec<_>>())
            .with("overlap_weights", tracking.overlap_weights.to_vec())
            .with("boundary_gaps", tracking.boundary_gaps.to_vec())
            .with("density_residual", tracking.density_residual));
    }
    if let Some(following) = &reference.following {
        report.set(
            "stability_following",
            Value::object()
                .with("stabilized", following.stabilized)
                .with("converged", following.converged)
                .with("initial_energy", following.initial_energy)
                .with(
                    "descents",
                    following
                        .descents
                        .iter()
                        .map(|d| {
                            Value::object()
                                .with("channel", d.channel.id())
                                .with("eigenvalue", d.eigenvalue)
                                .with("theta", d.theta)
                                .with("energy_before", d.energy_before)
                                .with("energy_after", d.energy_after)
                        })
                        .collect::<Vec<_>>(),
                ),
        );
        warnings.extend(following.warnings.iter().cloned());
    }
    report.set("warnings", warnings);
    report
}

pub(super) fn units(pairs: &[(&str, &str)]) -> Value {
    let mut units = Value::object();
    for (key, unit) in pairs {
        units.set(key, *unit);
    }
    units
}

pub(super) fn reference_orbital_residual(scf: &crate::scf::ScfResult) -> f64 {
    scf.common_orbital_residual.unwrap_or_else(|| {
        (0..2)
            .map(|spin| {
                crate::linalg::metric_commutator(
                    &scf.fock[spin],
                    &scf.density[spin],
                    &scf.overlap,
                    &scf.mo_coeff[spin],
                )
                .as_slice()
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max)
    })
}

fn energy(system: &System, controls: &Controls) -> Result<Value> {
    if super::double_hybrid::requested(controls)? {
        return super::double_hybrid::energy(system, controls);
    }
    let (setup, density_dispersion) = super::density_dispersion::prepare_energy(system, controls)?;
    controls.finish("energy")?;
    let reference = run_molecular_reference(&setup)?;
    let mut report = reference_report(
        &setup,
        &reference,
        "self-consistent molecular Kohn-Sham/Hartree-Fock total energy",
    )
    .with(
        "units",
        units(&[("energy", "hartree"), ("energy_components", "hartree")]),
    );
    if let Some(request) = density_dispersion {
        request.evaluate_collinear(&setup, &reference.scf, &mut report)?;
    }
    Ok(report)
}

/// Analytic nuclear gradient of the converged reference's variational
/// energy (the Mermin free energy at finite electronic temperature) plus the
/// empirical dispersion gradient.
fn refuse_smd_gradient(setup: &MolecularSetup) -> Result<()> {
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "analytic gradient with the SMD non-electrostatic term".into(),
            missing: "the analytic CDS surface-tension gradient; the electrostatic \
                C-PCM/IEF/COSMO/SS(V)PE gradients are available without the smd: prefix"
                .into(),
        });
    }
    Ok(())
}

pub(super) fn total_gradient(
    setup: &MolecularSetup,
    reference: &MolecularReference,
) -> Result<Vec<Vec3>> {
    refuse_smd_gradient(setup)?;
    let mut gradient = crate::gradient::assemble_gradient(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
    )?;
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
    Ok(gradient)
}

fn mermin_free_energy(system: &System, controls: &Controls) -> Result<Value> {
    let setup = setup(system, controls)?;
    if setup.method.fixed_occupations.is_some() {
        return Err(DftError::InvalidInput("Mermin free energy requires thermal equilibrium occupations; use energy for prescribed fixed populations".into()));
    }
    let with_gradient = controls.bool("gradient", false)?;
    let gibbs = if controls.bool("gibbs", false)? {
        Some(crate::properties::thermochemistry::ThermochemistryOptions {
            temperature_kelvin: controls.f64(
                "temperature",
                if setup.method.electronic_temperature > 0.0 {
                    setup.method.electronic_temperature
                } else {
                    298.15
                },
            )?,
            pressure_pascal: controls.f64("pressure", 101_325.0)?,
            symmetry_number: controls.usize("symmetry_number", 1)?,
            electronic_degeneracy: setup.method.multiplicity,
        })
    } else {
        None
    };
    controls.finish("mermin-free-energy")?;
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    let kt = scf.electronic_temperature;
    let mut report = reference_report(
        &setup,
        &reference,
        "Mermin free energy A = E - T S of a canonical (fixed per-spin electron count) Fermi-Dirac ensemble",
    )
    .with("mermin_free_energy", reference.total_energy())
    .with("free_energy", reference.total_energy())
    .with("free_energy_kind", "mermin; free_energy is a compatibility alias for mermin_free_energy")
    .with("internal_energy", reference.total_energy() + kt * scf.electronic_entropy)
    .with("entropy", scf.electronic_entropy)
    .with("entropy_term", -kt * scf.electronic_entropy)
    .with("electronic_temperature_kelvin", setup.method.electronic_temperature)
    .with("kt", kt)
    .with("chemical_potentials", scf.chemical_potentials.to_vec())
    .with("occupations", scf.occupations.to_vec())
    .with("ensemble", if kt > 0.0 { "canonical per spin channel" } else { "zero temperature (integer aufbau occupations)" });
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        ("mermin_free_energy", "hartree"),
        ("free_energy", "hartree"),
        ("internal_energy", "hartree"),
        ("entropy", "k_B (dimensionless)"),
        ("entropy_term", "hartree"),
        ("kt", "hartree"),
        ("chemical_potentials", "hartree"),
    ];
    if with_gradient {
        let gradient = total_gradient(&setup, &reference)?;
        report.set(
            "gradient",
            gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
        );
        report.set("provenance", "analytic");
        unit_pairs.push(("gradient", "hartree/bohr (of the Mermin free energy)"));
    }
    if let Some(options) = gibbs {
        let analytic = super::vibrational::reference_hessian(&setup, &reference)?;
        let modes = crate::properties::vibrational::vibrational_analysis(
            &setup.molecule,
            &analytic.hessian,
            None,
        )?;
        if modes.n_imaginary != 0 {
            return Err(DftError::InvalidInput(format!(
                "Gibbs RRHO thermochemistry requires real vibrational frequencies; {} imaginary mode(s) remain",modes.n_imaginary)));
        }
        let thermo = crate::properties::thermochemistry::thermochemistry_from_mermin(
            &setup.molecule,
            &modes.frequencies_cm,
            reference.total_energy(),
            setup.method.electronic_temperature,
            scf.electronic_entropy,
            options,
        )?;
        report.set("gibbs_free_energy", thermo.gibbs_free_energy);
        report.set(
            "thermochemistry",
            super::vibrational::thermochemistry_value(&thermo),
        );
        report.set("gibbs_model","equilibrium ideal-gas RRHO on the electronic Mermin surface; analytic fixed-electronic-temperature Hessian; electronic entropy counted once; nuclear and positive electronic temperatures equal");
        report.set("frequencies_cm1", modes.frequencies_cm);
        report.set("raw_asymmetry", analytic.raw_asymmetry);
        report.set(
            "ensemble_response_residuals",
            analytic.ensemble_response_residuals,
        );
        report.set("ensemble_number_defects", analytic.ensemble_number_defects);
        unit_pairs.push(("gibbs_free_energy", "hartree per molecule"));
        unit_pairs.push(("frequencies_cm1", "cm^-1"));
        unit_pairs.push(("raw_asymmetry", "hartree/bohr^2"));
        unit_pairs.extend(super::vibrational::thermochemistry_units());
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}

fn gradient(system: &System, controls: &Controls) -> Result<Value> {
    let setup = setup(system, controls)?;
    controls.finish("gradient")?;
    refuse_smd_gradient(&setup)?;
    let reference = run_molecular_reference(&setup)?;
    let gradient = total_gradient(&setup, &reference)?;
    // Without an external field the gradient of an isolated molecule sums to
    // zero; in a uniform field the sum is minus the total charge times E.
    let mut sum = Vec3::zero();
    for g in &gradient {
        sum += *g;
    }
    let expected = match setup.options.external_field {
        Some(e) => e * (-(setup.molecule.charge as f64)),
        None => Vec3::zero(),
    };
    Ok(reference_report(
        &setup,
        &reference,
        "analytic nuclear gradient of the total energy",
    )
    .with(
        "gradient",
        gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
    )
    .with(
        "forces",
        gradient
            .iter()
            .map(|g| (-*g).to_array())
            .collect::<Vec<_>>(),
    )
    .with("translation_residual", (sum - expected).norm())
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("gradient", "hartree/bohr"),
            ("forces", "hartree/bohr"),
            ("translation_residual", "hartree/bohr"),
        ]),
    ))
}

/// Gradient without a stored integral tensor: the SCF builds its Coulomb and
/// exchange matrices from integrals that are recomputed and discarded shell
/// quartet by shell quartet, and the gradient contracts the derivative
/// integrals with the densities in the same way. Memory stays quadratic in
/// the basis size.
fn streamed_gradient(system: &System, controls: &Controls) -> Result<Value> {
    let mut setup = setup(system, controls)?;
    controls.finish("streamed-gradient")?;
    refuse_smd_gradient(&setup)?;
    // A zero budget keeps the iteration from switching to the stored tensor.
    setup.options.incore_mb = 0;
    let reference = run_molecular_reference(&setup)?;
    let gradient = total_gradient(&setup, &reference)?;
    let mut sum = Vec3::zero();
    for g in &gradient {
        sum += *g;
    }
    let expected = match setup.options.external_field {
        Some(e) => e * (-(setup.molecule.charge as f64)),
        None => Vec3::zero(),
    };
    let nao = setup.basis.nao_sph as f64;
    Ok(reference_report(
        &setup,
        &reference,
        "analytic nuclear gradient with integral-direct Coulomb and exchange builds: no \
         integral tensor is stored",
    )
    .with(
        "gradient",
        gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
    )
    .with("translation_residual", (sum - expected).norm())
    .with("integral_storage", "none")
    .with(
        "avoided_tensor_megabytes",
        nao.powi(4) * 8.0 / (1024.0 * 1024.0),
    )
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("gradient", "hartree/bohr"),
            ("translation_residual", "hartree/bohr"),
            (
                "avoided_tensor_megabytes",
                "MiB, size of the dense integral tensor that is not stored",
            ),
        ]),
    ))
}

fn charges(system: &System, controls: &Controls) -> Result<Value> {
    let kind = CollinearPropertyReference::from_controls(controls)?;
    let setup = setup(system, controls)?;
    controls.finish("charges")?;
    let reference = run_collinear_property_reference(&setup, kind)?;
    let mulliken =
        crate::properties::mulliken_charges(&setup.molecule, &setup.basis, &reference.scf);
    let lowdin = crate::properties::lowdin_charges(&setup.molecule, &setup.basis, &reference.scf)?;
    // Mulliken spin populations from P_alpha - P_beta.
    let owner = setup.basis.sph_ao_atoms();
    let pa = reference.scf.density[0].matmul(&reference.scf.overlap);
    let pb = reference.scf.density[1].matmul(&reference.scf.overlap);
    let mut spin = vec![0.0; setup.molecule.len()];
    for (mu, &atom) in owner.iter().enumerate() {
        spin[atom] += pa[(mu, mu)] - pb[(mu, mu)];
    }
    let total: f64 = mulliken.iter().sum();
    Ok(reference_report(
        &setup,
        &reference,
        "Mulliken and Loewdin population analysis of the converged density",
    )
    .with("reference_type", kind.name())
    .with("mulliken", mulliken)
    .with("lowdin", lowdin)
    .with("mulliken_spin", spin)
    .with("charge_sum_residual", total - setup.molecule.charge as f64)
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("mulliken", "e"),
            ("lowdin", "e"),
            ("mulliken_spin", "electrons (alpha minus beta)"),
        ]),
    ))
}

/// Origin of an origin-dependent moment: explicit Bohr coordinates or one of
/// `center_of_mass` (default), `center_of_nuclear_charge`, `zero`.
pub(super) fn moment_origin(controls: &Controls, setup: &MolecularSetup) -> Result<(Vec3, String)> {
    molecular_moment_origin(controls, &setup.molecule, "center_of_mass")
}

pub(super) fn molecular_moment_origin(
    controls: &Controls,
    molecule: &Molecule,
    default: &str,
) -> Result<(Vec3, String)> {
    if let Some(v) = controls
        .take("origin")
        .filter(|v| !matches!(v, Value::Null))
    {
        if let Some(name) = v.as_str() {
            return named_origin(name, molecule);
        }
        let c = Controls::from_value(&Value::object().with("origin", v.clone()))?;
        let point = c.opt_vec3("origin")?.unwrap();
        return Ok((point, "explicit".into()));
    }
    named_origin(default, molecule)
}

fn named_origin(name: &str, molecule: &Molecule) -> Result<(Vec3, String)> {
    let atoms = &molecule.atoms;
    let weighted = |weights: Vec<f64>| {
        let total: f64 = weights.iter().sum();
        let mut point = Vec3::zero();
        for (atom, w) in atoms.iter().zip(&weights) {
            point += atom.position * (*w / total);
        }
        point
    };
    match name {
        "center_of_mass" => Ok((weighted(atoms.iter().map(|a| atomic_mass_au(a.z)).collect()), name.into())),
        "center_of_nuclear_charge" => Ok((weighted(atoms.iter().map(|a| a.z as f64).collect()), name.into())),
        "zero" => Ok((Vec3::zero(), name.into())),
        other => Err(DftError::InvalidInput(format!(
            "unknown origin `{other}` (center_of_mass, center_of_nuclear_charge, zero or [x,y,z] in Bohr)"
        ))),
    }
}

/// Raw Cartesian moments `M_efg = sum_A Z_A X^e Y^f Z^g - Tr(P <x^e y^f z^g>)`
/// of total order `order` about `origin`, in `cart_components(order)` order.
pub(super) fn raw_moments(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    origin: Vec3,
    order: u8,
) -> Vec<((i32, i32, i32), f64)> {
    moments_of_density(setup, &reference.scf.p_total(), origin, order)
}

fn moments_of_density(
    setup: &MolecularSetup,
    density: &Matrix,
    origin: Vec3,
    order: u8,
) -> Vec<((i32, i32, i32), f64)> {
    moment_operators(setup, origin, order)
        .into_iter()
        .map(|term| {
            (
                term.powers,
                term.ionic - density.frobenius_dot(&term.matrix),
            )
        })
        .collect()
}

pub(super) struct MomentOperator {
    pub powers: (i32, i32, i32),
    pub matrix: Matrix,
    pub ionic: f64,
}

pub(super) fn moment_operators(
    setup: &MolecularSetup,
    origin: Vec3,
    order: u8,
) -> Vec<MomentOperator> {
    let charges = setup.basis.effective_charges(&setup.molecule);
    crate::integrals::one_electron::moment_matrices(&setup.basis, origin, order)
        .into_iter()
        .map(|((e, f, g), m)| {
            let mut nuclear = 0.0;
            for (atom, z) in setup.molecule.atoms.iter().zip(&charges) {
                let r = atom.position - origin;
                nuclear += z * r.x.powi(e) * r.y.powi(f) * r.z.powi(g);
            }
            MomentOperator {
                powers: (e, f, g),
                matrix: m,
                ionic: nuclear,
            }
        })
        .collect()
}

fn electric_moments(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let kind = CollinearPropertyReference::from_controls(controls)?;
    if super::double_hybrid::requested(controls)? {
        if kind != CollinearPropertyReference::Roks {
            return super::double_hybrid::electric_moments(system, controls, id);
        }
        return Err(MomentRequest::missing_correlated_density(id, kind.name()));
    }
    let setup = setup(system, controls)?;
    let request = MomentRequest::new(id, &setup, controls)?;
    controls.finish(id)?;
    let reference = run_collinear_property_reference(&setup, kind)?;
    let mut report = reference_report(&setup, &reference, request.model());
    if kind == CollinearPropertyReference::Roks {
        report.set("reference_type", kind.name());
    }
    request.append(&mut report, &setup, &reference.scf.p_total());
    Ok(report)
}

pub(super) struct MomentRequest {
    origin: Vec3,
    origin_name: String,
    max_rank: Option<usize>,
}

impl MomentRequest {
    pub(super) fn missing_correlated_density(id: &str, reference: &str) -> DftError {
        DftError::Unsupported {
            feature: format!("correlated electric moments for `{id}` with reference={reference}"),
            missing: "the relaxed PT2 density and matching electric moment operators".into(),
        }
    }

    pub(super) fn new(id: &str, setup: &MolecularSetup, controls: &Controls) -> Result<Self> {
        let max_rank = if id == "molecular-multipole" {
            Some(multipole_rank(controls)?)
        } else {
            None
        };
        let (origin, origin_name) = moment_origin(controls, setup)?;
        Ok(Self {
            origin,
            origin_name,
            max_rank,
        })
    }

    fn model(&self) -> &'static str {
        if self.max_rank.is_some() {
            "molecular electric multipole moments of nuclei plus electrons"
        } else {
            "electric dipole moment of nuclei plus electrons"
        }
    }

    pub(super) fn append(self, report: &mut Value, setup: &MolecularSetup, density: &Matrix) {
        let mut labels = report
            .get("units")
            .cloned()
            .unwrap_or_else(|| units(&[("energy", "hartree"), ("energy_components", "hartree")]));
        if let Some(max_rank) = self.max_rank {
            append_multipoles(report, setup, density, self.origin, max_rank);
            for key in ["ranks.cartesian", "ranks.traceless", "ranks.spherical"] {
                labels.set(key, "e*bohr^rank");
            }
        } else {
            append_dipole(report, setup, density, self.origin);
            labels.set("dipole", "e*bohr");
            labels.set("dipole_debye", "debye");
        }
        report.set("origin", self.origin.to_array());
        report.set("origin_convention", self.origin_name);
        labels.set("origin", "bohr");
        report.set("units", labels);
    }
}

pub(super) fn multipole_rank(controls: &Controls) -> Result<usize> {
    let rank = controls.usize("max_rank", 4)?;
    if rank > 8 {
        return Err(DftError::InvalidInput(
            "max_rank must be between 0 and 8".into(),
        ));
    }
    Ok(rank)
}

fn append_dipole(report: &mut Value, setup: &MolecularSetup, density: &Matrix, origin: Vec3) {
    let mu: Vec<f64> = moments_of_density(setup, density, origin, 1)
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    append_dipole_values(report, &mu, setup.molecule.charge);
}

pub(super) fn append_dipole_values(report: &mut Value, mu: &[f64], charge: i32) {
    let norm = mu.iter().map(|value| value * value).sum::<f64>().sqrt();
    report.set(
        "dipole_debye",
        mu.iter()
            .map(|v| v * AU_DIPOLE_TO_DEBYE)
            .collect::<Vec<_>>(),
    );
    report.set("dipole", mu.to_vec());
    report.set("dipole_norm", norm);
    report.set("dipole_norm_debye", norm * AU_DIPOLE_TO_DEBYE);
    warn_charged_dipole(report, charge);
}

pub(super) fn warn_charged_dipole(report: &mut Value, charge: i32) {
    if charge != 0 {
        if let Some(Value::Array(mut warnings)) = report.get("warnings").cloned() {
            warnings.push(Value::Str("WARNING [dipole] the molecule is charged; its dipole depends on the reported origin".into()));
            report.set("warnings", Value::Array(warnings));
        }
    }
}

fn orbitals(system: &System, controls: &Controls) -> Result<Value> {
    let kind = CollinearPropertyReference::from_controls(controls)?;
    let setup = setup(system, controls)?;
    let coefficients = controls.bool("coefficients", false)?;
    let density = controls.bool("density", false)?;
    controls.finish("orbitals")?;
    let reference = run_collinear_property_reference(&setup, kind)?;
    let mut report = orbital_report(&setup, &reference, coefficients, density);
    report.set("reference_type", kind.name());
    if kind == CollinearPropertyReference::Roks {
        report.set("model", "ROKS density from common spatial orbitals; exported spin semicanonical occupied/virtual Fock blocks");
    }
    Ok(report)
}

/// Largest `|sum_i f_i C_i C_i^T - P|` over both spin channels: whether the
/// reported occupations reproduce the SCF density.
fn occupation_density_deviation(scf: &crate::scf::ScfResult) -> f64 {
    let mut worst = 0.0_f64;
    for spin in 0..2 {
        let rebuilt = crate::scf::occupied_density(&scf.mo_coeff[spin], &scf.occupations[spin]);
        worst = worst.max(rebuilt.max_abs_diff(&scf.density[spin]));
    }
    worst
}

fn export_path(controls: &Controls, feature: &str) -> Result<String> {
    controls
        .opt_str("path")?
        .ok_or_else(|| DftError::InvalidInput(format!("{feature} needs path (the file to write)")))
}

fn molden(system: &System, controls: &Controls) -> Result<Value> {
    let setup = setup(system, controls)?;
    let path = export_path(controls, "molden")?;
    controls.finish("molden")?;
    let reference = run_molecular_reference(&setup)?;
    let text = crate::properties::molden::molden_text(
        &setup.molecule,
        &setup.basis_set,
        &setup.basis,
        &reference.scf,
    );
    std::fs::write(&path, &text)?;
    Ok(reference_report(
        &setup,
        &reference,
        "Molden wavefunction export of the converged reference",
    )
    .with("path", path)
    .with("bytes", text.len())
    .with("n_mo", reference.scf.mo_coeff[0].cols)
    .with(
        "occupation_density_deviation",
        occupation_density_deviation(&reference.scf),
    )
    .with("ao_order", "Molden: spherical [5D][7F][9G], p as x, y, z")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("occupation_density_deviation", "electrons per AO element"),
        ]),
    ))
}

fn orbitals_save(system: &System, controls: &Controls) -> Result<Value> {
    let setup = setup(system, controls)?;
    let path = export_path(controls, "orbitals-save")?;
    controls.finish("orbitals-save")?;
    let reference = run_molecular_reference(&setup)?;
    let deviation = occupation_density_deviation(&reference.scf);
    if deviation > 1e-8 {
        return Err(DftError::InvalidInput(format!(
            "refusing to write orbitals whose occupations do not reproduce the SCF density              (max deviation {deviation:.3e})"
        )));
    }
    let mut report = orbital_report(&setup, &reference, true, true);
    report.set("occupation_density_deviation", deviation);
    let mut export_units = report.get("units").cloned().unwrap_or_else(Value::object);
    export_units.set("occupation_density_deviation", "electrons per AO element");
    export_units.set("chemical_potentials", "hartree");
    report.set("units", export_units);
    report.set("kt", reference.scf.electronic_temperature);
    report.set(
        "chemical_potentials",
        reference.scf.chemical_potentials.to_vec(),
    );
    report.set("path", path.clone());
    let mut written = report.clone();
    if let Value::Object(entries) = &mut written {
        entries.insert(0, ("feature".into(), Value::Str("orbitals-save".into())));
        entries.insert(
            1,
            (
                "version".into(),
                Value::Str(env!("CARGO_PKG_VERSION").into()),
            ),
        );
    }
    std::fs::write(&path, written.to_json_pretty())?;
    Ok(report)
}

fn orbital_report(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    coefficients: bool,
    density: bool,
) -> Value {
    let scf = &reference.scf;
    let spins = if scf.unrestricted { 2 } else { 1 };
    let mut channels = Vec::new();
    for spin in 0..spins {
        let n_occ = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        let energies = &scf.mo_energy[spin];
        let weight = if scf.unrestricted { 1.0 } else { 2.0 };
        let occupations: Vec<f64> = scf.occupations[spin].iter().map(|f| weight * f).collect();
        let homo_index = scf
            .has_integer_occupations()
            .then(|| n_occ.checked_sub(1))
            .flatten();
        let homo = homo_index.map(|i| energies[i]);
        let lumo = scf
            .has_integer_occupations()
            .then(|| energies.get(n_occ).copied())
            .flatten();
        let mut channel = Value::object()
            .with(
                "spin",
                if scf.unrestricted {
                    if spin == 0 {
                        "alpha"
                    } else {
                        "beta"
                    }
                } else {
                    "restricted"
                },
            )
            .with("energies", energies.clone())
            .with("occupations", occupations)
            .with("homo_index", homo_index)
            .with("homo", homo)
            .with("lumo", lumo)
            .with("gap", homo.zip(lumo).map(|(h, l)| l - h));
        if coefficients {
            channel.set("coefficients", &scf.mo_coeff[spin]);
        }
        if density {
            channel.set("density", &scf.density[spin]);
        }
        channels.push(channel);
    }
    let mut report = reference_report(
        setup,
        reference,
        "canonical Kohn-Sham/Hartree-Fock orbitals of the converged reference",
    )
    .with("n_ao", setup.basis.nao_sph)
    .with("channels", channels)
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("channels.energies", "hartree"),
            ("channels.coefficients", "dimensionless"),
            ("channels.density", "dimensionless, per spin"),
            ("channels.homo", "hartree"),
            ("channels.lumo", "hartree"),
            ("channels.gap", "hartree"),
        ]),
    );
    if coefficients || density {
        report.set(
            "ao_order",
            "spherical; per shell m = 0, +1, -1, +2, -2, ...",
        );
        report.set("overlap", &scf.overlap);
    }
    report
}

fn double_factorial(n: i64) -> f64 {
    if n <= 0 {
        1.0
    } else {
        (1..=n).rev().step_by(2).map(|k| k as f64).product()
    }
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Buckingham traceless multipole of rank `n` from the raw symmetric
/// Cartesian moment tensor, returned for every monomial in
/// `cart_components(n)` order:
/// `Theta = ((2n-1)!!/n!) * detrace(M)`, where `detrace` removes all traces
/// through the Applequist expansion.
fn traceless(n: usize, raw: &[((i32, i32, i32), f64)]) -> Vec<f64> {
    let lookup = |e: [i32; 3]| -> f64 {
        raw.iter()
            .find(|((a, b, c), _)| [*a, *b, *c] == e)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    };
    let prefactor = double_factorial(2 * n as i64 - 1) / factorial(n);
    let components = crate::basis::cart_components(n as u8);
    // Index tuple representing each monomial: x repeated e times, then y, z.
    components
        .iter()
        .map(|&(e, f, g)| {
            let mut indices = Vec::with_capacity(n);
            indices.extend(std::iter::repeat_n(0, e as usize));
            indices.extend(std::iter::repeat_n(1, f as usize));
            indices.extend(std::iter::repeat_n(2, g as usize));
            let mut value = 0.0;
            // Sum over k pairings of the free indices replaced by Kronecker
            // deltas; each delta pair contracts one trace of the raw tensor.
            for k in 0..=n / 2 {
                let coefficient = if k % 2 == 0 { 1.0 } else { -1.0 }
                    * double_factorial(2 * (n - k) as i64 - 1)
                    / double_factorial(2 * n as i64 - 1);
                let mut sum = 0.0;
                for_each_pairing(&indices, k, &mut |paired, rest| {
                    // Deltas must match pairwise.
                    if paired.iter().all(|(a, b)| a == b) {
                        // Remaining free indices plus k traced pairs.
                        let mut base = [0i32; 3];
                        for &i in rest {
                            base[i] += 1;
                        }
                        // Trace over k pairs: sum over all (a1..ak) of
                        // M[rest + 2 e_a1 + ... + 2 e_ak].
                        sum += traced(&base, k, &lookup);
                    }
                });
                value += coefficient * sum;
            }
            prefactor * value
        })
        .collect()
}

fn traced(base: &[i32; 3], k: usize, lookup: &dyn Fn([i32; 3]) -> f64) -> f64 {
    if k == 0 {
        return lookup(*base);
    }
    let mut sum = 0.0;
    for axis in 0..3 {
        let mut next = *base;
        next[axis] += 2;
        sum += traced(&next, k - 1, lookup);
    }
    sum
}

/// Visit every way of choosing `k` disjoint unordered pairs among the
/// positions of `indices`; the callback receives the paired index values
/// and the remaining free index values.
fn for_each_pairing(
    indices: &[usize],
    k: usize,
    visit: &mut dyn FnMut(&[(usize, usize)], &[usize]),
) {
    fn recurse(
        free: &[usize],
        k: usize,
        start: usize,
        pairs: &mut Vec<(usize, usize)>,
        used: &mut Vec<bool>,
        visit: &mut dyn FnMut(&[(usize, usize)], &[usize]),
    ) {
        if pairs.len() == k {
            let rest: Vec<usize> = free
                .iter()
                .zip(used.iter())
                .filter(|(_, &u)| !u)
                .map(|(&v, _)| v)
                .collect();
            visit(pairs, &rest);
            return;
        }
        for i in start..free.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            for j in i + 1..free.len() {
                if used[j] {
                    continue;
                }
                used[j] = true;
                pairs.push((free[i], free[j]));
                recurse(free, k, i + 1, pairs, used, visit);
                pairs.pop();
                used[j] = false;
            }
            used[i] = false;
        }
    }
    let mut used = vec![false; indices.len()];
    recurse(indices, k, 0, &mut Vec::new(), &mut used, visit);
}

/// Racah-normalized real spherical multipoles `Q_lm` in the order
/// m = 0, +1, -1, ..., +l, -l, from the raw Cartesian moments of order l.
fn spherical_modes(l: usize) -> Vec<(i32, f64)> {
    let li = l as i32;
    std::iter::once(0)
        .chain((1..=li).flat_map(|m| [m, -m]))
        .map(|m| {
            let ma = m.unsigned_abs() as usize;
            let delta = if m == 0 { 2.0 } else { 1.0 };
            let norm = (2.0 * factorial(l + ma) * factorial(l - ma) / delta).sqrt()
                / (2f64.powi(ma as i32) * factorial(l));
            (m, norm)
        })
        .collect()
}

/// Polynomial coefficients of the existing Racah real spherical moments.
/// Rows are m=0,+1,-1,...; columns follow cart_components(rank).
/// The same coefficients define a conjugate external potential and its
/// electronic one-body response operator, avoiding a second normalization.
pub(super) fn spherical_weights(rank: usize) -> Vec<Vec<f64>> {
    let components = crate::basis::cart_components(rank as u8);
    spherical_modes(rank)
        .into_iter()
        .map(|(m, norm)| {
            components
                .iter()
                .map(|&(e, f, g)| {
                    norm * crate::basis::aos::solid_harmonic_coef(rank as i32, m, e, f, g)
                })
                .collect()
        })
        .collect()
}

fn spherical(l: usize, raw: &[((i32, i32, i32), f64)]) -> Vec<f64> {
    spherical_modes(l)
        .into_iter()
        .map(|(m, norm)| {
            norm * raw
                .iter()
                .map(|((e, f, g), v)| {
                    crate::basis::aos::solid_harmonic_coef(l as i32, m, *e, *f, *g) * v
                })
                .sum::<f64>()
        })
        .collect()
}

fn append_multipoles(
    report: &mut Value,
    setup: &MolecularSetup,
    density: &Matrix,
    origin: Vec3,
    max_rank: usize,
) {
    let ranks = (0..=max_rank)
        .map(|n| multipole_values(n, &moments_of_density(setup, density, origin, n as u8)))
        .collect();
    set_multipole_ranks(report, ranks);
}

pub(super) fn multipole_values(rank: usize, raw: &[((i32, i32, i32), f64)]) -> Value {
    Value::object()
        .with("rank", rank)
        .with(
            "monomials",
            raw.iter()
                .map(|((e, f, g), _)| [*e, *f, *g])
                .collect::<Vec<_>>(),
        )
        .with("cartesian", raw.iter().map(|(_, v)| *v).collect::<Vec<_>>())
        .with("traceless", traceless(rank, raw))
        .with("spherical", spherical(rank, raw))
}

pub(super) fn set_multipole_ranks(report: &mut Value, ranks: Vec<Value>) {
    report.set(
            "conventions",
            Value::object()
                .with("cartesian", "raw moments sum_A Z_A X^e Y^f Z^g - Tr(P x^e y^f z^g), one per monomial")
                .with("traceless", "Buckingham: ((2n-1)!!/n!) times the traceless projection, one per monomial of the symmetric tensor")
                .with("spherical", "Racah-normalized real solid harmonics Q_lm, m = 0, +1, -1, ..., +l, -l"),
    );
    report.set("ranks", ranks);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epr_reference_preflight_never_consumes_an_unsupported_selector() {
        for id in ["hyperfine", "g-tensor"] {
            for kind in ["roks", "rohf", "gks", "ghf", "RoKs", "GhF"] {
                let value = Value::object().with("reference", kind);
                let controls = Controls::from_value(&value).unwrap();
                let error = check_epr_reference(id, &controls).unwrap_err();
                assert!(matches!(error, DftError::Unsupported { .. }));
                assert_eq!(controls.to_value(), value);
                assert!(controls.finish("preflight").is_err());
            }
        }
    }

    #[test]
    fn epr_reference_preflight_rejects_unknown_and_wrong_type_selectors() {
        for id in ["hyperfine", "g-tensor"] {
            for reference in [
                Value::Str("rkos".into()),
                Value::Str(String::new()),
                Value::Int(42),
                Value::Bool(true),
                Value::Array(Vec::new()),
            ] {
                let controls =
                    Controls::from_value(&Value::object().with("reference", reference)).unwrap();
                let error = check_epr_reference(id, &controls).unwrap_err();
                assert!(matches!(error, DftError::InvalidInput(_)));
                assert!(error.to_string().contains("reference"));
            }
        }
    }

    #[test]
    fn epr_reference_preflight_preserves_supported_collinear_selectors() {
        for id in ["hyperfine", "g-tensor"] {
            for reference in ["auto", "rks", "rhf", "uks", "uhf", "UkS"] {
                let controls =
                    Controls::from_value(&Value::object().with("reference", reference)).unwrap();
                check_epr_reference(id, &controls).unwrap();
                assert!(controls.finish("preflight").is_err());
            }
            check_epr_reference(id, &Controls::new()).unwrap();
            check_epr_reference(
                id,
                &Controls::from_value(&Value::object().with("reference", Value::Null)).unwrap(),
            )
            .unwrap();
        }
    }

    fn point_charges(charges: &[(f64, [f64; 3])], n: usize) -> Vec<((i32, i32, i32), f64)> {
        crate::basis::cart_components(n as u8)
            .into_iter()
            .map(|(e, f, g)| {
                let v = charges
                    .iter()
                    .map(|(q, r)| q * r[0].powi(e) * r[1].powi(f) * r[2].powi(g))
                    .sum();
                ((e, f, g), v)
            })
            .collect()
    }

    #[test]
    fn traceless_quadrupole_and_octupole_follow_buckingham() {
        let charges = [
            (1.0, [0.3, -0.2, 0.9]),
            (-0.4, [-1.1, 0.5, 0.2]),
            (0.7, [0.0, 1.3, -0.6]),
        ];
        let raw = point_charges(&charges, 2);
        let theta = traceless(2, &raw);
        // Closed form: Theta_ab = 1/2 sum q (3 r_a r_b - r^2 delta_ab).
        let comps = crate::basis::cart_components(2);
        for (c, &(e, f, g)) in comps.iter().enumerate() {
            let axes: Vec<usize> = std::iter::repeat_n(0, e as usize)
                .chain(std::iter::repeat_n(1, f as usize))
                .chain(std::iter::repeat_n(2, g as usize))
                .collect();
            let expected: f64 = charges
                .iter()
                .map(|(q, r)| {
                    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                    0.5 * q
                        * (3.0 * r[axes[0]] * r[axes[1]]
                            - if axes[0] == axes[1] { r2 } else { 0.0 })
                })
                .sum();
            assert!(
                (theta[c] - expected).abs() < 1e-13,
                "{c}: {} vs {expected}",
                theta[c]
            );
        }
        // Every rank is traceless: contracting any index pair gives zero.
        for n in 2..=6usize {
            let raw = point_charges(&charges, n);
            let t = traceless(n, &raw);
            let comps = crate::basis::cart_components(n as u8);
            let get = |e: [i32; 3]| {
                comps
                    .iter()
                    .position(|c| [c.0, c.1, c.2] == e)
                    .map(|i| t[i])
                    .unwrap()
            };
            for base in crate::basis::cart_components((n - 2) as u8) {
                let trace: f64 = (0..3)
                    .map(|a| {
                        let mut e = [base.0, base.1, base.2];
                        e[a] += 2;
                        get(e)
                    })
                    .sum();
                assert!(
                    trace.abs() < 1e-11 * t.iter().map(|v| v.abs()).fold(1.0, f64::max),
                    "rank {n} trace {trace}"
                );
            }
        }
    }

    #[test]
    fn spherical_moments_reduce_to_dipole_and_quadrupole() {
        let charges = [
            (1.0, [0.3, -0.2, 0.9]),
            (-0.4, [-1.1, 0.5, 0.2]),
            (0.7, [0.0, 1.3, -0.6]),
        ];
        let mu = point_charges(&charges, 1);
        let q1 = spherical(1, &mu);
        // m = 0, +1, -1 are z, x, y.
        assert!((q1[0] - mu[2].1).abs() < 1e-14);
        assert!((q1[1] - mu[0].1).abs() < 1e-14);
        assert!((q1[2] - mu[1].1).abs() < 1e-14);
        let raw = point_charges(&charges, 2);
        let q2 = spherical(2, &raw);
        let theta = traceless(2, &raw);
        // Q_20 = Theta_zz; Q_21 = (2/sqrt3) Theta_xz; Q_22 = (1/sqrt3)(Theta_xx - Theta_yy).
        let (xx, xz, yy, zz) = (theta[0], theta[2], theta[3], theta[5]);
        assert!((q2[0] - zz).abs() < 1e-13);
        assert!((q2[1] - 2.0 / 3f64.sqrt() * xz).abs() < 1e-13);
        assert!((q2[3] - (xx - yy) / 3f64.sqrt()).abs() < 1e-13);
        // Rotational invariant: sum_m Q_lm^2 equals (2/3) Theta:Theta for l=2.
        let theta_norm: f64 = {
            let comps = crate::basis::cart_components(2);
            comps
                .iter()
                .zip(&theta)
                .map(|(&(e, f, g), v)| {
                    let multiplicity = factorial(2)
                        / (factorial(e as usize) * factorial(f as usize) * factorial(g as usize));
                    multiplicity * v * v
                })
                .sum()
        };
        let q_norm: f64 = q2.iter().map(|v| v * v).sum();
        assert!(
            (q_norm - 2.0 / 3.0 * theta_norm).abs() < 1e-12,
            "{q_norm} {theta_norm}"
        );
    }
}

#[cfg(test)]
mod spherical_weight_tests {
    use super::{spherical, spherical_weights};

    fn legendre(rank: usize, x: f64) -> f64 {
        let (mut previous, mut current) = (1.0, x);
        if rank == 0 {
            return previous;
        }
        for n in 2..=rank {
            let next = ((2 * n - 1) as f64 * x * current - (n - 1) as f64 * previous) / n as f64;
            previous = current;
            current = next;
        }
        current
    }

    #[test]
    fn spherical_operator_weights_obey_addition_theorem_and_laplace_equation() {
        for rank in 1..=8 {
            let powers = crate::basis::cart_components(rank as u8);
            let weights = spherical_weights(rank);
            assert_eq!(weights.len(), 2 * rank + 1);
            let count = 2 * rank + 3;
            let points: Vec<[f64; 3]> = (0..count)
                .map(|i| {
                    let z = 2.0 * (i as f64 + 0.31) / (count as f64 + 0.47) - 1.0;
                    let angle = 2.399963229728653 * i as f64 + 0.23;
                    let radial = (1.0 - z * z).sqrt();
                    [radial * angle.cos(), radial * angle.sin(), z]
                })
                .collect();
            let values: Vec<Vec<f64>> = points
                .iter()
                .map(|point| {
                    let raw: Vec<_> = powers
                        .iter()
                        .map(|&(e, f, g)| {
                            (
                                (e, f, g),
                                point[0].powi(e) * point[1].powi(f) * point[2].powi(g),
                            )
                        })
                        .collect();
                    let observed: Vec<f64> = weights
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(&raw)
                                .map(|(weight, (_, moment))| weight * moment)
                                .sum()
                        })
                        .collect();
                    for (operator, moment) in observed.iter().zip(spherical(rank, &raw)) {
                        assert!((operator - moment).abs() < 2e-12);
                    }
                    // The m=0 mode has an independent Legendre closed form.
                    assert!((observed[0] - legendre(rank, point[2])).abs() < 2e-12);
                    observed
                })
                .collect();
            for (a, r) in points.iter().enumerate() {
                for (b, t) in points.iter().enumerate() {
                    let cosine: f64 = r.iter().zip(t).map(|(x, y)| x * y).sum();
                    let overlap: f64 = values[a].iter().zip(&values[b]).map(|(x, y)| x * y).sum();
                    assert!(
                        (overlap - legendre(rank, cosine)).abs() < 4e-12,
                        "rank {rank}, points {a},{b}"
                    );
                }
            }
            if rank >= 2 {
                for (mode, row) in weights.iter().enumerate() {
                    for base in crate::basis::cart_components((rank - 2) as u8) {
                        let laplacian: f64 = (0..3)
                            .map(|axis| {
                                let mut power = [base.0, base.1, base.2];
                                power[axis] += 2;
                                let index = powers
                                    .iter()
                                    .position(|&(e, f, g)| [e, f, g] == power)
                                    .unwrap();
                                f64::from(power[axis] * (power[axis] - 1)) * row[index]
                            })
                            .sum();
                        assert!(laplacian.abs() < 2e-11, "rank {rank}, mode {mode}");
                    }
                }
            }
        }
    }
}
