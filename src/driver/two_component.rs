// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::method::{prepare_molecular, MolecularMethod};
use super::{Controls, System, Value};
use crate::dispersion::{dispersion_energy, Dispersion};
use crate::integrals::dirac::GaussianNucleus;
use crate::periodic::spinor_scf::{run_spinor_scf, SpinorEnergyBackend};
use crate::properties::two_component::{
    MolecularRelativisticOptions, MolecularTwoComponent, RelativisticSpin, TwoComponentMethod,
    UniformMagneticBasis,
};
use crate::response::complex::Complex;
use crate::{DftError, Result};

mod magnetic_response;
mod nuclear_response;

#[derive(Clone, Copy)]
enum Observable {
    Reference,
    Dipole,
    DipoleResponse { dynamic: bool },
    ElectricMultipole,
    SpinCoupling,
    Magnetizability,
    Shielding,
    Spectrum,
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    run_balanced_scf(
        system,
        controls,
        "two-component-relativistic",
        false,
        Observable::Reference,
    )
}

pub(super) fn dirac_dks(system: &System, controls: &Controls) -> Result<Value> {
    run_balanced_scf(system, controls, "dirac-dks", true, Observable::Reference)
}

/// Select the backend capability once for all interfaces on the common
/// driver. Each supported property uses the same balanced SCF and kernel.
pub(super) fn dirac_property(feature: &str, system: &System, controls: &Controls) -> Result<Value> {
    let _ = controls.opt_str("relativistic_method")?;
    let observable = match feature {
        "dipole" => Observable::Dipole,
        "polarizability" => Observable::DipoleResponse { dynamic: false },
        "dynamic-polarizability" => Observable::DipoleResponse { dynamic: true },
        "spin-spin-couplings" => Observable::SpinCoupling,
        "magnetizability" => Observable::Magnetizability,
        "nmr-shielding" => Observable::Shielding,
        "nmr-spectrum" => Observable::Spectrum,
        id if super::quadrupole::is_electric_multipole(id) => Observable::ElectricMultipole,
        _ => {
            return Err(DftError::Unsupported {
                feature: format!("four-component Dirac–Kohn–Sham {feature}"),
                missing: "a balanced-basis operator and response for the requested observable"
                    .into(),
            })
        }
    };
    run_balanced_scf(system, controls, feature, true, observable)
}

fn run_balanced_scf(
    system: &System,
    controls: &Controls,
    feature: &str,
    dirac_sector: bool,
    observable: Observable,
) -> Result<Value> {
    let unsupported = |context: &str, missing: &str| DftError::Unsupported {
        feature: format!("{feature} {context}"),
        missing: missing.into(),
    };
    if system.is_periodic() {
        return Err(unsupported(
            "periodic reference",
            "an isolated molecular system",
        ));
    }
    if super::double_hybrid::requested(controls)? {
        return Err(unsupported(
            "double hybrid",
            "PT2 in the balanced relativistic orbital space",
        ));
    }
    let dirac_interaction = dirac_sector
        .then(|| super::no_pair::electronic_interaction(controls))
        .transpose()?;
    let magnetic_observable = matches!(
        observable,
        Observable::Magnetizability | Observable::Shielding | Observable::Spectrum
    );
    let magnetic = if dirac_interaction.is_some() {
        let field = controls.opt_vec3("magnetic_field")?.map(|f| f.to_array());
        let mode = controls.opt_str("dirac_integrals")?;
        match (mode.as_deref(), field) {
            (None | Some("rkb"), None) if !magnetic_observable => None,
            (None | Some("london-rmb"), _) => {
                let gauge = controls.opt_vec3("gauge_origin")?.unwrap_or_default();
                if gauge.to_array().iter().any(|x| !x.is_finite()) {
                    return Err(DftError::InvalidInput("gauge_origin must be finite".into()));
                }
                let field = field.unwrap_or([0.0; 3]);
                let fallback = if field == [0.0; 3] { 0 } else { 6 };
                let phase_order = controls.usize("magnetic_phase_order", fallback)?;
                if magnetic_observable && field != [0.0; 3] && phase_order < 2 {
                    return Err(DftError::InvalidInput("Dirac magnetic response at finite field needs magnetic_phase_order >= 2".into()));
                }
                Some(UniformMagneticBasis { field, phase_order })
            }
            _ => {
                return Err(DftError::InvalidInput(
                    "four-component DKS needs dirac_integrals=rkb for zero field or london-rmb for magnetic balance".into(),
                ))
            }
        }
    } else {
        None
    };
    let method = match if dirac_sector {
        "x2c".into()
    } else {
        controls.str("relativistic_method", "x2c")?
    }
    .as_str()
    {
        "x2c" => TwoComponentMethod::X2c,
        "dkh2" => TwoComponentMethod::Dkh2,
        "zora" | "matrix-zora" => TwoComponentMethod::MatrixZora,
        _ => {
            return Err(DftError::InvalidInput(
                "relativistic_method must be x2c, dkh2 or matrix-zora".into(),
            ))
        }
    };
    let (spin, spin_name) = match controls.str("relativistic_spin", "spin-orbit")?.as_str() {
        "scalar" => (RelativisticSpin::Scalar, "scalar"),
        "spin-orbit" => (RelativisticSpin::SpinOrbit, "spin-orbit"),
        _ => {
            return Err(DftError::InvalidInput(
                "relativistic_spin must be scalar or spin-orbit".into(),
            ))
        }
    };
    if dirac_sector && spin != RelativisticSpin::SpinOrbit {
        return Err(DftError::Unsupported {
            feature: "four-component Dirac–Kohn–Sham reference".into(),
            missing: "relativistic_spin=spin-orbit".into(),
        });
    }
    let expected_reference = if spin == RelativisticSpin::Scalar {
        "rks"
    } else {
        "gks"
    };
    if let Some(reference) = controls.opt_str("reference")? {
        if reference != expected_reference {
            return Err(unsupported(
                "reference selector",
                &format!("reference={expected_reference} for the selected relativistic spin model"),
            ));
        }
    }
    let settings = MolecularMethod::from_controls(controls)?;
    if settings.multiplicity != 1
        || settings.unrestricted
        || settings.electronic_temperature != 0.0
        || settings.electron_offset != [0.0; 2]
        || settings.occupation.is_some()
        || settings.fixed_occupations.is_some()
        || settings.stabilize
        || settings.guess_mix.is_some()
    {
        return Err(unsupported(
            "reference controls",
            "an insulating, integer, singlet-seeded reference without occupation or stability constraints",
        ));
    }
    if settings.ecp.is_some()
        || settings.skala_model.is_some()
        || settings.solvent.is_some()
        || settings.eps.is_some()
        || settings.density_fit.is_some()
        || settings.level_shift.is_some()
        || settings.guess.as_ref().is_some_and(|g| g != "core")
        || controls.contains("adiis_until_error")
        || !settings.adiis
    {
        return Err(unsupported(
            "provider or algorithm controls",
            "all-electron exact pair integrals, semilocal/hybrid XC and the core guess",
        ));
    }
    let radii = controls.opt_f64_list("nuclear_rms_radii")?.ok_or_else(|| {
        DftError::InvalidInput(
            "nuclear_rms_radii in Bohr must be supplied for each finite Gaussian nucleus".into(),
        )
    })?;
    if radii.len() != system.numbers.len() || radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
        return Err(DftError::InvalidInput(
            "nuclear_rms_radii requires one positive finite Bohr radius per atom".into(),
        ));
    }
    let speed = controls.f64("speed_of_light", 137.035999084)?;
    let metric_cutoff = controls.f64("metric_eigenvalue_cutoff", 1e-12)?;
    let xc_angular_tolerance = controls.f64("relativistic_xc_angular_tolerance", 1e-9)?;
    let explicit_origin = controls
        .opt_vec3("electric_origin")?
        .unwrap_or_default()
        .to_array();
    let field_gradient: Option<[f64; 5]> = if dirac_sector {
        controls.opt_f64_list("electric_field_gradient")?.map(|values| {
            if values.len() != 5 || values.iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "electric_field_gradient needs five finite Buckingham field-gradient amplitudes".into(),
                ));
            }
            Ok(values.try_into().expect("five field-gradient amplitudes"))
        }).transpose()?
    } else {
        None
    };
    let response_order = controls.usize("response_order", 0)?;
    if response_order > 1 {
        return Err(unsupported("electric response order", "order zero or one"));
    }
    if response_order != 0 && !matches!(observable, Observable::Reference) {
        return Err(DftError::InvalidInput(
            "the selected property supplies the Dirac response order".into(),
        ));
    }
    let frequencies = if response_order == 1 {
        let damping = controls.f64("frequency_damping", 0.0)?;
        if !damping.is_finite() || damping < 0.0 {
            return Err(DftError::InvalidInput(
                "frequency_damping must be finite and nonnegative".into(),
            ));
        }
        let frequencies = controls
            .opt_f64_list("frequencies")?
            .unwrap_or_else(|| vec![0.0]);
        if frequencies.is_empty() || frequencies.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "two-component response needs at least one finite frequency".into(),
            ));
        }
        frequencies
            .into_iter()
            .map(|v| Complex::new(v, damping))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let solver = (response_order == 1)
        .then(|| super::vibrational::response_config(controls))
        .transpose()?;
    let setup = prepare_molecular(&system.numbers, &system.positions, &settings)?;
    let multipole = matches!(observable, Observable::ElectricMultipole)
        .then(|| super::quadrupole::Request::new(feature, &setup, controls))
        .transpose()?;
    let dipole_origin = matches!(observable, Observable::Dipole)
        .then(|| super::molecular::moment_origin(controls, &setup))
        .transpose()?;
    if (multipole.is_some() || dipole_origin.is_some()) && controls.contains("electric_origin") {
        return Err(DftError::InvalidInput(
            "the selected moment or multipole response uses origin to select its origin".into(),
        ));
    }
    let origin = multipole
        .as_ref()
        .map(super::quadrupole::Request::origin)
        .or_else(|| dipole_origin.as_ref().map(|(point, _)| point.to_array()))
        .unwrap_or(explicit_origin);
    let dipole_response = match observable {
        Observable::DipoleResponse { dynamic } => {
            let frequency = if dynamic {
                if controls.str("thermal_response", "integer")? != "integer" {
                    return Err(unsupported(
                        "thermal dipole response",
                        "integer occupations",
                    ));
                }
                super::polarizability::frequency(controls)?.energy()?
            } else {
                Complex::ZERO
            };
            Some((frequency, super::vibrational::response_config(controls)?))
        }
        _ => None,
    };
    let magnetic_request = magnetic_observable
        .then(|| {
            magnetic_response::Request::new(
                controls,
                setup.molecule.len(),
                matches!(observable, Observable::Shielding | Observable::Spectrum),
                matches!(observable, Observable::Spectrum),
            )
        })
        .transpose()?;
    let nuclear_request = dirac_sector
        .then(|| {
            nuclear_response::Request::new(
                controls,
                setup.molecule.len(),
                matches!(observable, Observable::SpinCoupling),
            )
        })
        .transpose()?
        .flatten();
    let supplied = super::no_pair::operator_coefficients(controls, 4 * setup.basis.nao_sph, 0)?;
    controls.finish(feature)?;
    let required_pair = setup.functional.required_pair_dispersion();
    if required_pair
        .as_ref()
        .is_some_and(|model| *model != setup.dispersion)
        || (!matches!(setup.dispersion, Dispersion::None | Dispersion::Chg)
            && required_pair.as_ref() != Some(&setup.dispersion))
        || setup.options.vv10 != setup.functional.required_vv10()
    {
        return Err(unsupported(
            "dispersion combination",
            "the resolved provider's VV10 or defining pair correction, with dispersion=auto",
        ));
    }
    // Empirical pair corrections are constant during a clamped electric solve.
    let dispersion = (setup.dispersion != Dispersion::None)
        .then(|| {
            dispersion_energy(
                &setup.molecule,
                &setup.dispersion,
                &settings.dispersion_functional_name(),
            )
        })
        .transpose()?;
    let nuclei: Vec<_> = setup
        .molecule
        .atoms
        .iter()
        .zip(&radii)
        .map(|(atom, &rms_radius)| GaussianNucleus {
            center: atom.position.to_array(),
            charge: f64::from(atom.z),
            rms_radius,
        })
        .collect();
    let vv10_grid = setup
        .functional
        .required_vv10()
        .map(|_| setup.options.nonlocal_grid(&setup.molecule))
        .transpose()?;
    let mut config = setup.nested_soc_scf_config(controls);
    config.overlap_cutoff = metric_cutoff;
    config.orbital_optimizer = setup.spinor_scf_config().orbital_optimizer;
    let options = MolecularRelativisticOptions {
        method,
        spin,
        speed_of_light: speed,
        metric_cutoff,
        xc_angular_tolerance,
        pair_cache_mb: setup.options.incore_mb,
        electric_field: settings.field.unwrap_or_default().to_array(),
        electric_origin: origin,
        dispersion_energy: dispersion,
    };
    let electrons = setup.molecule.n_electrons()?;
    let mut backend = if dirac_sector {
        MolecularTwoComponent::in_dirac_sector(
            &setup.basis,
            &nuclei,
            electrons,
            &setup.functional,
            setup.grid.as_ref(),
            vv10_grid.as_ref(),
            options,
            dirac_interaction.expect("Dirac Hamiltonian").0,
            magnetic,
            nuclear_request
                .as_ref()
                .and_then(nuclear_response::Request::fields),
        )?
    } else {
        MolecularTwoComponent::new(
            &setup.basis,
            &nuclei,
            electrons,
            &setup.functional,
            setup.grid.as_ref(),
            vv10_grid.as_ref(),
            options,
        )?
    };
    let quadrupole = (field_gradient.is_some()
        || multipole
            .as_ref()
            .is_some_and(super::quadrupole::Request::uses_buckingham_modes))
    .then(|| super::quadrupole::BalancedModes::quadrupole(&backend, &setup, origin))
    .transpose()?;
    if let Some(field_gradient) = field_gradient {
        let (operator, nuclear) = quadrupole
            .as_ref()
            .expect("finite gradient operators")
            .electric_source(field_gradient);
        backend = backend.with_electric_source(operator, nuclear)?;
    }
    let state = run_spinor_scf(&backend, &config)?;
    let map = backend.transformation();
    let terms = state.fields.energy_components;
    let mut components = Value::object()
        .with("classical_nuclear_repulsion", terms[0])
        .with("relativistic_core", terms[1])
        .with("vv10", terms[2])
        .with("dispersion", terms[3])
        .with("hartree", terms[4])
        .with("exchange", terms[5])
        .with("long_range_exchange", terms[6])
        .with("semilocal_xc", terms[7])
        .with("electric", terms[8]);
    let current = backend.stationary_current_fields(&state.density[0])?;
    if let Some(fields) = &current {
        components.set("current_direct", fields.direct_energy);
        components.set("current_exchange", fields.exchange_energy);
    }
    let mut units = Value::object()
        .with("energy", "hartree")
        .with("energy_components", "hartree")
        .with("dipole", "e*bohr")
        .with("nuclear_rms_radii", "bohr")
        .with("electric_origin", "bohr")
        .with("speed_of_light", "atomic units of velocity")
        .with("orbital_energies", "hartree")
        .with("occupations", "electrons per spinor")
        .with("scf_orbital_residual", "hartree")
        .with("energy_change", "hartree")
        .with("density_change", "dimensionless")
        .with("metric_defect", "dimensionless")
        .with("occupation_consistency", "dimensionless")
        .with("particle_number_error", "electrons")
        .with("decoupling_metric_defect", "dimensionless")
        .with("decoupling_residual", "hartree")
        .with("coulomb_hermiticity_defect", "hartree");
    let model = if let Some((_, _, hamiltonian)) = dirac_interaction {
        let balance = if magnetic.is_some() {
            "London orbitals and restricted magnetic balance and a nuclear-core positive-energy sector formed at the selected uniform magnetic field"
        } else {
            "restricted kinetic balance and a fixed nuclear-core positive-energy sector"
        };
        format!("finite-Gaussian-nucleus four-component no-pair {hamiltonian} Kohn–Sham SCF with {balance}")
    } else {
        "finite-Gaussian-nucleus two-component SCF with fixed field-free relativistic picture and picture-changed Coulomb, exchange and XC".into()
    };
    let dipole = backend.dipole(&state.density[0])?;
    let mut report = Value::object()
        .with("model", model)
        .with(
            "relativistic_method",
            if dirac_sector { "dirac" } else { method.name() },
        )
        .with("relativistic_spin", spin_name)
        .with("reference_type", expected_reference)
        .with("basis", settings.basis)
        .with("xc", setup.functional.description.clone())
        .with("n_ao", setup.basis.nao_sph)
        .with("energy", state.fields.energy)
        .with("energy_components", components)
        .with("dipole", dipole.to_vec())
        .with("electric_origin", origin.to_vec())
        .with("nuclear_rms_radii", radii)
        .with("speed_of_light", speed)
        .with("scf_iterations", state.iterations)
        .with("scf_orbital_residual", state.orbital_residual)
        .with("energy_change", state.energy_change)
        .with("density_change", state.density_change)
        .with("metric_defect", state.metric_defect)
        .with("occupation_consistency", state.occupation_consistency)
        .with("particle_number_error", state.particle_number_error)
        .with("decoupling_metric_defect", map.metric_defects[0])
        .with("decoupling_residual", map.decoupling_defects[0])
        .with(
            "coulomb_hermiticity_defect",
            state.fields.coulomb_operator_defect,
        )
        .with("orbital_energies", state.bands[0].energies.clone())
        .with("occupations", state.occupations[0].clone())
        .with("converged", state.converged)
        .with("warnings", Vec::<String>::new());
    if let Some(request) = magnetic_request {
        request.append(&mut report, &mut units, &nuclei, &backend, &state)?;
    }
    if let Some(request) = &nuclear_request {
        request.append(&mut report, &mut units, &nuclei, &backend, &state)?;
    }
    if let Some(magnetic) = magnetic {
        report.set("dirac_integrals", "london-rmb");
        report.set("magnetic_field", magnetic.field);
        report.set("magnetic_phase_order", magnetic.phase_order);
        units.set("magnetic_field", "atomic units of magnetic field");
        units.set("magnetic_phase_order", "integer");
    }
    if let Some((_, name)) = dipole_origin {
        report.set("origin", origin);
        report.set("origin_convention", name);
        super::molecular::append_dipole_values(&mut report, &dipole, setup.molecule.charge);
        units.set("origin", "bohr");
        units.set("dipole_debye", "debye");
        units.set("dipole_norm", "e*bohr");
        units.set("dipole_norm_debye", "debye");
    }
    if let Some(gap) = backend.dirac_spectral_gap() {
        let (_, name, _) = dirac_interaction.expect("Dirac sector has an interaction");
        report.set("dirac_interaction", name);
        report.set("spectral_gap", gap);
        report.set("electronic_dimension", map.embedding()[0].cols());
        units.set("spectral_gap", "hartree");
        units.set("electronic_dimension", "integer");
    }
    if let Some(fields) = current {
        report.set("current_hermiticity_defect", fields.hermiticity_defect);
        report.set(
            "current_model",
            "unscaled determinant current direct and exchange added to Coulomb XC",
        );
        units.set("current_hermiticity_defect", "hartree");
    }
    if let Some(quadrupole) = &quadrupole {
        report.set(
            "quadrupole_moment",
            quadrupole.buckingham_moment(&state.density[0]),
        );
        units.set("quadrupole_moment", "e*bohr^2");
    }
    if let Some(field_gradient) = field_gradient {
        report.set("electric_field_gradient", field_gradient);
        units.set("electric_field_gradient", "hartree/e/bohr^2");
    }
    if let Some(supplied) = supplied {
        let operator = map.transform(&supplied)?.remove(0);
        let expectation = state.density[0].trace_product(&operator);
        report.set("picture_changed_operator_real", operator.re);
        report.set("picture_changed_operator_imaginary", operator.im);
        report.set("picture_changed_operator_expectation", expectation);
        units.set("picture_changed_operator_real", "dimensionless");
        units.set("picture_changed_operator_imaginary", "dimensionless");
        units.set("picture_changed_operator_expectation", "dimensionless");
    }
    if let Some(diagnostic) = &state.fields.xc_diagnostics {
        report.set("xc_angular_order", diagnostic.angular_order);
        report.set("xc_angular_energy_defect", diagnostic.angular_energy_defect);
        report.set(
            "xc_angular_operator_defect",
            diagnostic.angular_operator_defect,
        );
        report.set("xc_hermiticity_defect", diagnostic.hermiticity_defect);
        report.set(
            "xc_component_energy_defect",
            diagnostic.component_energy_defect,
        );
        report.set(
            "xc_component_energies",
            state.fields.xc_component_energies.clone(),
        );
        for key in [
            "xc_angular_energy_defect",
            "xc_angular_operator_defect",
            "xc_hermiticity_defect",
            "xc_component_energy_defect",
            "xc_component_energies",
        ] {
            units.set(key, "hartree");
        }
    }
    if let Some(solver) = solver {
        let results = backend.electric_response(&state, &frequencies, &solver)?;
        let block = |part: fn(Complex) -> f64| {
            results
                .iter()
                .map(|r| {
                    r.susceptibility
                        .iter()
                        .map(|row| row.iter().map(|&x| -part(x)).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        report.set("polarizability_real", block(|x| x.re));
        report.set("polarizability_imaginary", block(|x| x.im));
        report.set(
            "frequencies",
            frequencies.iter().map(|z| z.re).collect::<Vec<_>>(),
        );
        report.set("frequency_damping", frequencies[0].im);
        report.set(
            "response_residuals",
            results.into_iter().map(|r| r.residuals).collect::<Vec<_>>(),
        );
        units.set("polarizability_real", "e^2*bohr^2/hartree");
        units.set("polarizability_imaginary", "e^2*bohr^2/hartree");
        units.set("frequencies", "hartree");
        units.set("frequency_damping", "hartree");
        units.set("response_residuals", "dimensionless");
    }
    report.set("units", units);
    if let Some(request) = multipole {
        request.append_balanced(&mut report, &setup, &backend, &state, quadrupole.as_ref())?;
    }
    if let Some((frequency, solver)) = dipole_response {
        let value = backend
            .electric_response(&state, &[frequency], &solver)?
            .remove(0);
        let alpha = std::array::from_fn(|row| {
            std::array::from_fn(|column| -value.susceptibility[row][column])
        });
        super::polarizability::tensor_values(
            &mut report,
            alpha,
            frequency,
            &value.residuals,
            &value.iterations,
        );
    }
    Ok(report.with("electrons", backend.electrons() as usize))
}
