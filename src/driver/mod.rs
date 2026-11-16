// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The common calculation driver behind the Rust, CLI, Python and ASE
//! surfaces.
//!
//! A request is a feature ID from the [registry](registry), a [`System`]
//! (atomic numbers, Bohr positions and, for a crystal, Bohr lattice rows and
//! periodic axes) and named [`Controls`]. [`run`] executes it and returns one
//! self-describing [`Value`] report. The CLI prints that report as text or
//! JSON, the Python extension converts it into dictionaries and lists, and
//! the ASE calculator calls the Python function, so every surface returns the
//! same numbers from the same code.
//!
//! Every report carries at least `feature`, `model`, `units`, `converged`
//! and `warnings`. Values are in atomic units unless a key names another unit
//! (`*_debye`, `*_cm1`, `*_gpa`, ...). Raw residuals are reported as measured,
//! before any symmetrization.

mod band_properties;
mod car_parrinello;
mod conceptual;
mod constrained;
pub mod continuous_species;
pub mod controls;
mod correlated_electric;
mod correlated_high_order;
mod correlated_nuclear;
mod correlated_periodic;
mod density_dispersion;
mod divide_conquer;
mod double_hybrid;
mod electric;
mod epr;
mod exafs;
mod exchange;
mod excited;
mod force_constants;
mod geomech;
mod gks;
mod gks_excited;
mod gtensor;
mod gtensor_nuclear;
mod hubbard;
mod hyperfine;
mod hyperfine_nuclear;
mod igmh;
mod magnetic_quadrupole;
mod magnetizability;
mod magnon;
mod mcd;
mod mcpdft;
mod medium_optics;
pub mod method;
mod molecular;
mod mrsf;
mod multicomponent;
mod natural_bond;
mod nexafs;
mod nmr;
mod no_pair;
mod nto;
pub mod nuclear_tensor;
mod optical;
mod pairing;
pub mod periodic;
mod periodic_magnetic_nuclear;
mod periodic_roks;
mod polarizability;
mod quadrupole;
mod reference_electric;
mod reference_vibrations;
pub mod registry;
mod roks;
mod roks_excited;
mod sapt;
mod sic;
mod spin_response;
mod spin_spin_zfs;
mod spinorbit;
mod static_spectroscopy;
mod superconducting;
mod two_component;
pub mod typed;
pub mod units;
pub mod value;
mod vcd;
mod vdw_df_post_scf;
mod vibrational;

pub use controls::Controls;
pub use density_dispersion::handles as handles_molecular_dispersion_energy;
pub use registry::{features, find, FeatureSpec, Route};
pub use value::{ToValue, Value};

use crate::{DftError, Result};

/// The structure a calculation acts on. Positions and lattice rows are Bohr.
#[derive(Clone, Debug, Default)]
pub struct System {
    pub numbers: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    /// Lattice vectors as rows, Bohr. `None` for an isolated molecule.
    pub cell: Option<[[f64; 3]; 3]>,
    /// Periodic axes; ignored without a cell.
    pub pbc: [bool; 3],
}

impl System {
    pub fn molecule(numbers: Vec<u8>, positions: Vec<[f64; 3]>) -> Self {
        Self {
            numbers,
            positions,
            cell: None,
            pbc: [false; 3],
        }
    }

    pub fn is_periodic(&self) -> bool {
        self.cell.is_some() && self.pbc.iter().any(|&p| p)
    }
}

/// Run one registered calculation through the common driver.
pub fn run(feature: &str, system: &System, controls: &Controls) -> Result<Value> {
    let logging = crate::logger::LogOptions::from_strings(
        controls.opt_str("log_file")?,
        controls.opt_str("log_mode")?,
        controls.opt_str("log_level")?,
    )?;
    let stripped = ["log_file", "log_mode", "log_level"]
        .iter()
        .any(|key| controls.contains(key))
        .then(|| controls.excluding(&["log_file", "log_mode", "log_level"]));
    let controls = stripped.as_ref().unwrap_or(controls);
    logging.install(|| {
        crate::logger::log(&format!(
            "driver feature={feature} atoms={} periodic={}",
            system.numbers.len(),
            system.is_periodic()
        ));
        let result = run_resources(feature, system, controls);
        match &result {
            Ok(report) => {
                for key in if crate::logger::enabled(crate::logger::LogLevel::Info) {
                    &[
                        "model",
                        "basis",
                        "xc",
                        "reference_type",
                        "energy",
                        "free_energy",
                        "scf_iterations",
                        "scf_orbital_residual",
                        "orbital_residual",
                        "converged",
                        "resources",
                        "units",
                    ][..]
                } else {
                    &[]
                } {
                    if let Some(value) = report.get(key) {
                        crate::logger::log(&format!("result {key}={}", value.to_json()));
                    }
                }
                if let Some(Value::Array(warnings)) = report
                    .get("warnings")
                    .filter(|_| crate::logger::is_active())
                {
                    for warning in warnings {
                        crate::logger::warn(&warning.to_json());
                    }
                }
            }
            Err(error) => crate::logger::record_error(&error.to_string()),
        }
        result
    })?
}

fn run_resources(feature: &str, system: &System, controls: &Controls) -> Result<Value> {
    let limits = crate::resources::ResourceLimits::new(
        controls.opt_strict_usize("cpu_threads")?,
        controls.opt_strict_usize("memory_mb")?,
    )?;
    // Feature adapters can reconstruct Controls while translating references
    // or modes. Global settings belong to the envelope, not those new lists.
    let feature_controls = (controls.contains("cpu_threads") || controls.contains("memory_mb"))
        .then(|| controls.excluding(&["cpu_threads", "memory_mb"]));
    let controls = feature_controls.as_ref().unwrap_or(controls);
    if limits.is_configured() || crate::logger::is_active() {
        let controls = controls.clone();
        let system = system.clone();
        return limits.run(move || run_impl(feature, &system, &controls));
    }
    if crate::resources::current_usage().is_some() {
        crate::resources::catch_allocations(|| run_impl(feature, system, controls))
    } else {
        run_impl(feature, system, controls)
    }
}

fn run_impl(feature: &str, system: &System, controls: &Controls) -> Result<Value> {
    controls.reset_usage();
    let spec = registry::driver_feature(feature)?;
    // Energy and static electric response resolve custom weighted PT2 parents
    // through the same model preparation as their single-functional paths.
    if controls.contains("xc_terms")
        && !matches!(
            spec.id,
            "energy"
                | "quadrupole-polarizability"
                | "multipole-polarizability"
                | "polarizability"
                | "dynamic-polarizability"
                | "molecular-medium-optics"
                | "constrained-dft"
                | "dft-plus-u"
                | "pz-sic"
                | "flo-sic-fixed"
                | "nexafs"
                | "magnetic-quadrupole"
                | "car-parrinello-md"
                | "two-component-relativistic"
                | "dirac-dks"
        )
        && (!matches!(
            spec.id,
            "gradient"
                | "hessian"
                | "roks-hessian"
                | "dipole"
                | "molecular-multipole"
                | "orbitals"
                | "igmh"
                | "periodic-energy"
                | "periodic-magnetic-scf"
                | "stress"
                | "elastic"
                | "phonons"
                | "dos"
        ) || double_hybrid::requested(controls)?)
    {
        return Err(crate::DftError::Unsupported {
            feature: format!("weighted XC for `{}`", spec.id),
            missing: "the weighted functional connection for this response or PT2 model".into(),
        });
    }
    let mut report = match spec.id {
        "feature-registry" => {
            controls.finish("feature-registry")?;
            Value::object()
                .with("model", "registry of public calculations")
                .with("units", Value::object())
                .with("converged", true)
                .with("warnings", Vec::<String>::new())
                .with("features", registry::registry_value())
                .with("control_units", units::control_units_value())
        }
        "scdft-pairing-kernel" => pairing::run(controls)?,
        "superconducting-state" => superconducting::run(controls)?,
        "no-pair-projector" => no_pair::run(system, controls)?,
        "dirac-coulomb-scf" => no_pair::dirac_coulomb_scf(system, controls)?,
        "two-component-relativistic" => two_component::run(system, controls)?,
        "dirac-dks" => two_component::dirac_dks(system, controls)?,
        id if controls.peek_str("relativistic_method")? == Some("dirac") => {
            two_component::dirac_property(id, system, controls)?
        }
        id if spec.space == registry::Space::Periodic => periodic::run(id, system, controls)?,
        id if spec.space == registry::Space::Data => geomech::run(id, system, controls)?,
        id => molecular::run(id, system, controls)?,
    };
    name_thermodynamic_potentials(spec.id, &mut report);
    if let Some(usage) = crate::resources::current_usage() {
        report.set("resources", Value::object()
            .with("cpu_threads", usage.cpu_threads)
            .with("memory_mb", usage.memory_mb)
            .with("managed_peak_bytes", usage.peak_reserved_bytes)
            .with("managed_live_bytes", usage.reserved_bytes)
            .with("scope", "registered Rust matrices, ERI storage and explicit workspace reservations; not process RSS"));
    }
    let problems = units::complete(&mut report);
    if !problems.is_empty() && units::strict() {
        return Err(DftError::InvalidInput(format!(
            "the report of `{feature}` has incomplete unit labels: {}",
            problems.join("; ")
        )));
    }
    if let Value::Object(entries) = &mut report {
        entries.insert(0, ("feature".to_string(), Value::Str(spec.id.to_string())));
        entries.insert(
            1,
            (
                "version".to_string(),
                Value::Str(env!("CARGO_PKG_VERSION").to_string()),
            ),
        );
    }
    Ok(report)
}

/// Keep historical report keys while naming the actual thermodynamic potential.
/// This labels completed calculations; it does not select a physical model.
fn name_thermodynamic_potentials(feature: &str, report: &mut Value) {
    let copy = |report: &mut Value, old: &str, new: &str| {
        if let Some(value) = report.get(old).cloned() {
            report.set(new, value);
            if let Some(mut labels) = report.get("units").cloned() {
                if let Some(label) = labels.get(old).cloned() {
                    labels.set(new, label);
                    report.set("units", labels);
                }
            }
        }
    };
    if report.get("free_energy").is_some() {
        let (key, kind) = match feature {
            "phonon-thermo" => ("helmholtz_free_energy", "helmholtz; harmonic lattice vibrations at fixed cell"),
            "quasi-harmonic" => ("helmholtz_free_energy", "helmholtz; quasi-harmonic E + F_vib at the equilibrium volume; gibbs_free_energy includes pV"),
            _ => ("mermin_free_energy", "mermin; electronic A = E - T_e S_e; free_energy is a compatibility alias"),
        };
        copy(report, "free_energy", key);
        report.set("free_energy_kind", kind);
    }
    if feature == "quasi-harmonic" {
        copy(report, "gibbs_energy", "gibbs_free_energy");
    }
    copy(
        report,
        "reference_free_energy",
        "reference_mermin_free_energy",
    );
    copy(report, "delta_free_energy", "delta_mermin_free_energy");
}

/// Render a report as aligned `key value` text. Arrays are printed as
/// compact JSON on the same line so that every number stays unambiguous;
/// historical component rows are retained for electric-response text views.
pub fn report_text(report: &Value) -> String {
    let mut out = String::new();
    render_text(report, "", &mut out);
    // Keep the historical ordered component rows while obtaining every
    // value from the common calculation, including constrained references.
    if let Some(feature @ ("beta" | "gamma")) = report.get("feature").and_then(Value::as_str) {
        if let Some(entries) = report.get("tensor").and_then(Value::as_array) {
            use std::fmt::Write;
            let rank = if feature == "beta" { 3 } else { 4 };
            for (flat, entry) in entries.iter().enumerate() {
                if let Some(value) = entry.as_f64() {
                    let mut axes = vec![0; rank];
                    let mut index = flat;
                    for axis in axes.iter_mut().rev() {
                        *axis = index % 3;
                        index /= 3;
                    }
                    let _ = write!(out, "{feature}");
                    for axis in axes {
                        let _ = write!(out, " {axis}");
                    }
                    let _ = writeln!(out, " {value:.12e}");
                }
            }
        }
    }
    use std::fmt::Write;
    match report.get("feature").and_then(Value::as_str) {
        Some("static-spectroscopy") => {
            if let Some(entries) = report.get("tensor").and_then(Value::as_array) {
                out = out
                    .lines()
                    .filter(|line| !line.starts_with("tensor "))
                    .collect::<Vec<_>>()
                    .join("\n");
                out.push('\n');
                for (index, value) in entries.iter().enumerate() {
                    if let Some(value) = value.as_f64() {
                        let _ = writeln!(out, "tensor {index} {value:.15e}");
                    }
                }
            }
        }
        Some("field-taylor") => {
            if let Some((shape, values)) = report
                .get("dipole_coefficients")
                .and_then(Value::numeric_array)
            {
                if shape.len() == 2 && shape[1] == 3 {
                    let _ = writeln!(out, "# Taylor coefficients = derivative n / n!");
                    for (n, row) in values.chunks_exact(3).enumerate() {
                        let _ = writeln!(
                            out,
                            "dipole_coefficient {n} {:.15e} {:.15e} {:.15e}",
                            row[0], row[1], row[2]
                        );
                    }
                }
            }
        }
        Some("field-mixed") => {
            if let Some((_, values)) = report
                .get("dipole_derivative")
                .and_then(Value::numeric_array)
            {
                if values.len() == 3 {
                    out = out
                        .lines()
                        .filter(|line| !line.starts_with("dipole_derivative "))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let _ = writeln!(
                        out,
                        "\ndipole_derivative {:.15e} {:.15e} {:.15e}",
                        values[0], values[1], values[2]
                    );
                }
            }
        }
        Some("dynamic-polarizability") => {
            // Atomic-unit volume is the historical text convention; structured
            // labels retain the explicit charge/energy dimensions for ASE.
            let _ = writeln!(out, "# alpha atomic units: bohr^3 = e^2*bohr^2/hartree");
            if let (Some((_, real)), Some((_, imaginary))) = (
                report.get("alpha_real").and_then(Value::numeric_array),
                report.get("alpha_imag").and_then(Value::numeric_array),
            ) {
                if real.len() == 9 && imaginary.len() == 9 {
                    for a in 0..3 {
                        for b in 0..3 {
                            let _ = writeln!(
                                out,
                                "alpha {a} {b} {:.15e} {:.15e}",
                                real[3 * a + b],
                                imaginary[3 * a + b]
                            );
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out
}

fn render_text(value: &Value, prefix: &str, out: &mut String) {
    match value {
        Value::Object(entries) => {
            for (key, item) in entries {
                let name = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match item {
                    Value::Object(_) => render_text(item, &name, out),
                    _ => {
                        out.push_str(&name);
                        out.push(' ');
                        match item {
                            Value::Str(s) => out.push_str(s),
                            other => out.push_str(&other.to_json()),
                        }
                        out.push('\n');
                    }
                }
            }
        }
        other => {
            out.push_str(prefix);
            out.push(' ');
            out.push_str(&other.to_json());
            out.push('\n');
        }
    }
}
