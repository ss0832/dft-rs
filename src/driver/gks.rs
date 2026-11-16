// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static observables of the shared molecular complex spinor reference.
//! Energy, populations and orbital exports consume the same full density.
use super::method::run_molecular_reference;
use super::molecular::{setup, units, MomentRequest};
use super::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::periodic::spinor_scf::{pauli_components, run_spinor_scf, SpinorEnergyBackend};
use crate::periodic::BlochMatrix;
use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use crate::{DftError, Result};

fn matrix(controls: &Controls, key: &str, n: usize) -> Result<Matrix> {
    let Some(rows) = controls.opt_rows(key)? else {
        return Ok(Matrix::zeros(n, n));
    };
    if rows.len() != n
        || rows
            .iter()
            .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(format!(
            "{key} must be a finite {n} by {n} spin/AO matrix"
        )));
    }
    Ok(Matrix::from_row_major(
        n,
        n,
        rows.into_iter().flatten().collect(),
    ))
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let requested_reference = controls.str("reference", "gks")?;
    let moments_requested = matches!(id, "dipole" | "molecular-multipole");
    let correlated = super::double_hybrid::requested(controls)?;
    let free_energy_requested = matches!(id, "mermin-free-energy" | "free-energy");
    let mermin_gradient = free_energy_requested && controls.bool("gradient", false)?;
    if free_energy_requested && (correlated || controls.bool("gibbs", false)?) {
        return Err(DftError::Unsupported {
            feature: "generalized Mermin free energy with PT2 or Gibbs thermochemistry".into(),
            missing:
                "the corresponding finite-temperature spinor energy or nuclear-response provider"
                    .into(),
        });
    }
    let natural_bond = (id == "natural-bond")
        .then(|| super::natural_bond::SpinorRequest::new(controls, correlated))
        .transpose()?;
    let igmh = (id == "igmh")
        .then(|| super::igmh::Request::parse(controls))
        .transpose()?;
    let magnetizability = (id == "magnetizability")
        .then(|| super::magnetizability::spinor::Request::parse(controls, correlated))
        .transpose()?;
    if moments_requested && correlated {
        return Err(MomentRequest::missing_correlated_density(
            id,
            &requested_reference,
        ));
    }
    if correlated && matches!(id, "hessian" | "force-constants") {
        return Err(DftError::Unsupported {
            feature: format!("generalized double-hybrid {id}"),
            missing: "the higher-order moving spinor PT2 nuclear response".into(),
        });
    }
    let correlated_high_order = matches!(id, "beta" | "gamma") && correlated;
    if correlated_high_order {
        super::correlated_high_order::require_electric_assembly(id, controls)?;
    }
    let (setup, correlation, density_dispersion) = if correlated_high_order
        || (matches!(id, "energy" | "polarizability" | "gradient")
            || super::quadrupole::is_electric_multipole(id))
            && correlated
    {
        let (setup, model) = super::double_hybrid::prepare_spinor(system, controls, id)?;
        (setup, Some(model), None)
    } else if id == "energy" || free_energy_requested {
        let (setup, correction) = super::density_dispersion::prepare_energy(system, controls)?;
        (setup, None, correction)
    } else {
        (setup(system, controls)?, None, None)
    };
    if magnetizability.is_some() && !setup.basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "GKS orbital magnetizability with ECP".into(),
            missing: "magnetic derivatives of the nonlocal ECP projectors".into(),
        });
    }
    if let Some(request) = &igmh {
        request.validate(&setup)?;
    }
    let electric_tensor =
        if correlation.is_none() && matches!(id, "static-spectroscopy" | "beta" | "gamma") {
            Some(super::reference_electric::Request::new(
                id, system, controls,
            )?)
        } else {
            None
        };
    let quadrupole = super::quadrupole::is_electric_multipole(id)
        .then(|| {
            if correlation.is_some() {
                super::quadrupole::Request::new_static_correlated(id, &setup, controls)
            } else {
                super::quadrupole::Request::new(id, &setup, controls)
            }
        })
        .transpose()?;
    let moments = moments_requested
        .then(|| MomentRequest::new(id, &setup, controls))
        .transpose()?;
    if setup.method.stabilize {
        return Err(DftError::Unsupported {
            feature: "generalized-reference stability following".into(),
            missing: "stability following in the full complex spinor orbital space".into(),
        });
    }
    let axis = controls
        .opt_vec3("spin_axis")?
        .map_or([0.0, 0.0, 1.0], |v| v.to_array());
    let xc_tolerance = controls.f64("spinor_xc_tolerance", 1e-10)?;
    let n = setup.basis.nao_sph;
    let operator = BlochMatrix {
        re: matrix(controls, "gks_operator_real", 2 * n)?,
        im: matrix(controls, "gks_operator_imaginary", 2 * n)?,
    };
    let coefficients = if id == "orbitals" {
        controls.bool("coefficients", false)?
    } else {
        false
    };
    let density_export = if id == "orbitals" {
        controls.bool("density", false)?
    } else {
        false
    };
    let electric = if matches!(id, "polarizability" | "dynamic-polarizability") {
        let frequency = super::polarizability::frequency(controls)?;
        if correlation.is_some() && frequency.energy()?.abs() != 0.0 {
            return Err(DftError::Unsupported {
                feature: "frequency-dependent correlated susceptibility".into(),
                missing: "the causal PT2 amplitude/orbital action kernel".into(),
            });
        }
        Some(frequency)
    } else {
        None
    };
    let nmr = if matches!(id, "nmr-shielding" | "nmr-spectrum") {
        if super::double_hybrid::requested(controls)? {
            return Err(DftError::Unsupported {
                feature: "correlated spinor NMR".into(),
                missing: "the spinor PT2 magnetic Lagrangian; parent-only tensors cannot represent a double hybrid".into(),
            });
        }
        Some(super::nmr::spinor::Request::new(id, controls)?)
    } else {
        None
    };
    let magnetic = if matches!(id, "spin-spin-couplings" | "fermi-contact-couplings") {
        let nuclei = controls
            .opt_usize_list("nuclei")?
            .unwrap_or_else(|| (0..setup.molecule.len()).collect());
        let nuclear_g = controls.opt_f64_list("nuclear_g")?;
        if nuclei.len() < 2
            || nuclear_g
                .as_ref()
                .is_some_and(|g| g.len() != setup.molecule.len())
        {
            return Err(DftError::InvalidInput(
                "spin-spin couplings need at least two nuclei and one nuclear_g per atom".into(),
            ));
        }
        Some((
            nuclei,
            nuclear_g,
            super::nmr::spinor::constraint_mode(controls)?,
        ))
    } else {
        None
    };
    let tensor_selection = if id == "force-constants" {
        super::force_constants::require_variational_reference(controls)?;
        if setup.smd.is_some() {
            return Err(DftError::Unsupported {
                feature: "complete spinor nuclear tensors with an SMD surface term".into(),
                missing: "the full-spin solvent geometry provider".into(),
            });
        }
        let selection = super::force_constants::Selection::new(system, controls)?;
        selection.check_report_storage(3, setup.options.incore_mb)?;
        Some(selection)
    } else {
        None
    };
    let excited = if matches!(id, "excitations" | "td-state-gradient" | "td-hessian") {
        Some(super::gks_excited::Request::from_controls(id, controls)?)
    } else {
        None
    };
    let response_config = if matches!(
        id,
        "gradient"
            | "excitations"
            | "td-state-gradient"
            | "td-hessian"
            | "hessian"
            | "polarizability"
            | "dynamic-polarizability"
            | "force-constants"
            | "spin-spin-couplings"
            | "fermi-contact-couplings"
            | "nmr-shielding"
            | "nmr-spectrum"
            | "static-spectroscopy"
            | "beta"
            | "gamma"
    ) || mermin_gradient
    {
        Some(super::vibrational::response_config(controls)?)
    } else {
        None
    };
    let hubbard_sites = if id == "dft-plus-u" {
        crate::scf::hubbard::validate_occupations(&setup.options)?;
        Some((
            super::hubbard::read_sites(controls)?,
            super::hubbard::read_pairs(controls)?,
        ))
    } else {
        None
    };
    controls.finish(id)?;
    let seed = run_molecular_reference(&setup)?;
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &seed.scf,
        &setup.options,
        operator.clone(),
        axis,
        MolecularSpinorFunctionalModel::SpinCovariant,
        xc_tolerance,
    )?;
    let config = setup.spinor_scf_config();
    let hubbard = hubbard_sites
        .map(|(sites, pairs)| {
            crate::scf::hubbard::HubbardModel::new(
                &seed.scf.overlap,
                sites,
                setup.options.orth_threshold,
            )?
            .with_intersite(pairs)
        })
        .transpose()?;
    let scf = if let Some(hubbard) = &hubbard {
        let corrected = crate::scf::hubbard::HubbardSpinorBackend {
            inner: &backend,
            hubbard,
        };
        run_spinor_scf(&corrected, &config)?
    } else {
        run_spinor_scf(&backend, &config)?
    };
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: scf.orbital_residual,
        });
    }
    let p = &scf.density[0];
    let pauli = pauli_components(p)?;
    let s = &seed.scf.overlap;
    let owner = setup.basis.sph_ao_atoms();
    let ionic = setup.basis.effective_charges(&setup.molecule);
    let mut mulliken = ionic.clone();
    let mut spin = vec![[0.0; 3]; ionic.len()];
    let mut electrons = 0.0;
    for i in 0..n {
        for j in 0..n {
            let pop = pauli[0].re[(i, j)] * s[(j, i)];
            mulliken[owner[i]] -= pop;
            electrons += pop;
            for a in 0..3 {
                spin[owner[i]][a] += pauli[a + 1].re[(i, j)] * s[(j, i)];
            }
        }
    }
    let lowdin =
        crate::properties::lowdin_density_charges(&setup.molecule, &setup.basis, &pauli[0].re, s)?;
    let band = &scf.bands[0];
    let spin_diagnostics = crate::properties::spin_diagnostics::molecular_spin_diagnostics(
        &band.coefficients,
        &scf.occupations[0],
        s,
    )?;
    crate::logger::log(&format!(
        "[gks] <S^2>={:.12} (hbar^2)",
        spin_diagnostics.squared
    ));
    let mut channel = Value::object()
        .with("spin", "spinor")
        .with("energies", band.energies.clone())
        .with("occupations", scf.occupations[0].clone());
    if coefficients {
        channel.set("coefficients_real", &band.coefficients.re);
        channel.set("coefficients_imaginary", &band.coefficients.im);
    }
    if density_export {
        channel.set("density_real", &p.re);
        channel.set("density_imaginary", &p.im);
        channel.set("fock_real", &scf.fields.fock[0].re);
        channel.set("fock_imaginary", &scf.fields.fock[0].im);
    }
    let mut report = Value::object().with("reference_type","gks")
        .with("requested_reference",requested_reference)
        .with("model","self-consistent complex molecular spinor density; full Coulomb and full/range-separated exchange of all spin blocks; spin-covariant globally projected semilocal XC with kinetic/current descriptors; scalar nonlocal correlation and reaction field when selected; no correlated PT2 density or transverse response kernel inferred")
        .with("functional",setup.functional.description.as_str()).with("converged",scf.converged)
        .with("scf_iterations",scf.iterations).with("energy",scf.free_energy+seed.e_dispersion+seed.e_smd_cds)
        .with("internal_energy",scf.fields.energy+seed.e_dispersion+seed.e_smd_cds)
        .with("mermin_free_energy",scf.free_energy+seed.e_dispersion+seed.e_smd_cds)
        .with("electronic_entropy_term",scf.entropy_term)
        .with("energy_components",Value::object().with("spinor",scf.fields.energy_components.to_vec())
            .with("dispersion",seed.e_dispersion).with("smd_cds",seed.e_smd_cds))
        .with("orbital_residual",scf.orbital_residual).with("metric_defect",scf.metric_defect)
        .with("density_change",scf.density_change).with("energy_change",scf.energy_change)
        .with("particle_number_error",scf.particle_number_error)
        .with("electron_count",electrons).with("charge_sum_residual",mulliken.iter().sum::<f64>()-setup.molecule.charge as f64)
        .with("mulliken",mulliken).with("lowdin",lowdin)
        .with("mulliken_spin",spin.iter().map(|v|v[2]).collect::<Vec<_>>())
        .with("mulliken_spin_vector",spin.clone())
        .with("spin_expectation", spin_diagnostics.expectation)
        .with("s_squared", spin_diagnostics.squared)
        .with("channels",vec![channel]).with("n_ao",n)
        .with("ao_order","spin-major [up spherical AOs, down spherical AOs]; spinor columns have arbitrary phases and degenerate-subspace gauges; occupations are per spinor")
        .with("warnings",Vec::<String>::new())
        .with("units",units(&[
            ("energy","hartree"),("internal_energy","hartree"),("mermin_free_energy","hartree"),("electronic_entropy_term","hartree"),("energy_components","hartree"),
            ("orbital_residual","hartree"),("metric_defect","dimensionless"),("density_change","dimensionless"),
            ("energy_change","hartree"),("particle_number_error","electrons"),("electron_count","electrons"),
            ("charge_sum_residual","e"),("mulliken","e"),("lowdin","e"),
            ("mulliken_spin","electrons, z Pauli magnetization"),("mulliken_spin_vector","electrons, xyz Pauli magnetization"),
            ("spin_expectation","dimensionless, hbar"),("s_squared","dimensionless, hbar^2"),("channels.energies","hartree"),
            ("spin_sector_weights.spin","dimensionless, hbar"),("spin_sector_weights.weight","dimensionless"),
            ("spin_sector_normalization_defect","dimensionless"),("spin_sector_squared_moment_defect","dimensionless, hbar^2"),
            ("channels.occupations","electrons per spinor"),("channels.coefficients_real","dimensionless"),
            ("channels.coefficients_imaginary","dimensionless"),("channels.density_real","dimensionless"),
            ("channels.density_imaginary","dimensionless"),("channels.fock_real","hartree"),
            ("channels.fock_imaginary","hartree"),("overlap_real","dimensionless"),("overlap_imaginary","dimensionless"),
            ("spatial_overlap","dimensionless"),("xc_grid_number_defect","electrons"),
            ("xc_angular_history","[dimensionless, hartree, hartree]"),
        ]));
    if let Some(sectors) = spin_diagnostics.sectors {
        let mut values = Vec::with_capacity(sectors.weights.len());
        let mut summary = Vec::with_capacity(sectors.weights.len());
        for (twice_spin, weight) in sectors.weights {
            let spin = 0.5 * twice_spin as f64;
            values.push(Value::object().with("spin", spin).with("weight", weight));
            summary.push(format!("S={spin:.1} weight={weight:.10}"));
        }
        crate::logger::log(&format!(
            "[gks] total-spin sectors {}; normalization_defect={:.3e} squared_moment_defect={:.3e}",
            summary.join(", "),
            sectors.normalization_defect,
            sectors.squared_moment_defect,
        ));
        report.set("spin_sector_model", "integral-occupation determinant");
        report.set("spin_sector_weights", values);
        report.set(
            "spin_sector_normalization_defect",
            sectors.normalization_defect,
        );
        report.set(
            "spin_sector_squared_moment_defect",
            sectors.squared_moment_defect,
        );
    } else {
        crate::logger::log("[gks] total-spin sector projection requires integral occupations");
        report.set(
            "spin_sector_model",
            "fractional-occupation Wick expectation",
        );
        report.set("spin_sector_weights", Value::Null);
    }
    if let Some(hubbard) = &hubbard {
        let fields = hubbard.evaluate_spinor(p)?;
        super::hubbard::append_spinor(&mut report, fields, scf.orbital_residual);
    }
    if let Some(request) = density_dispersion {
        request.evaluate(
            &setup,
            &pauli[0].re,
            &[&pauli[1].re, &pauli[2].re, &pauli[3].re],
            &mut report,
        )?;
    }
    if free_energy_requested {
        let kt = setup.options.electronic_temperature;
        let entropy = if kt > 0.0 { scf.entropy_term / kt } else { 0.0 };
        report.set(
            "free_energy",
            scf.free_energy + seed.e_dispersion + seed.e_smd_cds,
        );
        report.set(
            "electronic_temperature_kelvin",
            setup.method.electronic_temperature,
        );
        report.set("kt", kt);
        report.set("entropy", entropy);
        // The common molecular Mermin report stores the signed contribution -T S.
        report.set("entropy_term", -scf.entropy_term);
        report.set("chemical_potentials", vec![scf.chemical_potential]);
        report.set("occupations", vec![scf.occupations[0].clone()]);
        report.set(
            "ensemble",
            if kt > 0.0 {
                "fixed total electron count with one spinor chemical potential"
            } else {
                "zero temperature (integer spinor occupations)"
            },
        );
        let mut labels = report.get("units").cloned().expect("spinor report units");
        labels.set("free_energy", "hartree");
        labels.set("electronic_temperature_kelvin", "K");
        labels.set("kt", "hartree");
        labels.set("entropy", "k_B (dimensionless)");
        labels.set("entropy_term", "hartree");
        labels.set("chemical_potentials", "hartree");
        labels.set("occupations", "electrons per spinor");
        report.set("units", labels);
    }
    if moments_requested
        || quadrupole.is_some()
        || correlation.is_some()
        || setup.method.xc_terms.is_some()
    {
        report.set("scf_orbital_residual", scf.orbital_residual);
    }
    if coefficients || density_export {
        report.set("overlap_real", &backend.overlap()[0].re);
        report.set("overlap_imaginary", &backend.overlap()[0].im);
        report.set("spatial_overlap", s);
    }
    if let Some(request) = excited {
        super::gks_excited::append(
            id,
            &mut report,
            &setup,
            &scf,
            &operator,
            &request,
            response_config
                .as_ref()
                .expect("excitation response controls"),
        )?;
    } else if let Some(request) = magnetizability {
        request.append(&mut report, &setup, &scf, &operator, &pauli[0].re, &backend)?;
    } else if let Some(request) = nmr {
        request.append(
            &mut report,
            &setup,
            &scf,
            &operator,
            response_config.as_ref().expect("NMR response controls"),
        )?;
    } else if let Some((nuclei, nuclear_g, fixed_spin)) = magnetic {
        let constraints = if fixed_spin {
            crate::response::spin_spin::spin_direction_constraints(p, s)?
        } else {
            Vec::new()
        };
        let contact_only = id == "fermi-contact-couplings";
        let coupling_grid = setup.integration_grid()?;
        let result = crate::response::spin_spin::spinor_spin_spin_couplings(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            Some(&coupling_grid),
            &scf,
            &setup.options,
            &operator,
            &constraints,
            &nuclei,
            contact_only,
            response_config
                .as_ref()
                .expect("magnetic response controls"),
        )?;
        super::nmr::spinor::append(
            &mut report,
            &result,
            nuclear_g.as_deref(),
            contact_only,
            fixed_spin,
        );
    } else if let Some(frequency) = electric {
        if let Some(model) = &correlation {
            use super::correlated_electric::spinor;
            let sources = spinor::scalar_sources(crate::integrals::dipole_matrices(
                &setup.basis,
                crate::math::Vec3::zero(),
            ));
            let value = spinor::solve(
                &setup,
                &scf,
                &operator,
                model,
                sources,
                response_config
                    .as_ref()
                    .expect("electric response controls"),
            )?;
            super::polarizability::tensor_values(
                &mut report,
                value.tensor(model.coefficient),
                frequency.energy()?,
                &value.residuals,
                &[],
            );
            value.append(&mut report, model, "polarizability", 1.0, 1);
            report.set("axis_convention", "alpha[observable Cartesian axis][electric-field Cartesian axis], from the symmetric static energy Hessian with H'=+E.r");
            report.set("model", "Static noncollinear double-hybrid electric dipole Hessian with full-spin stationary orbital and generalized PT2 response. A distinct target follows the relaxed parent density.");
        } else {
            let values = crate::response::nuclear::spinor_electric_response(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                &setup.functional,
                &scf,
                &setup.options,
                &operator,
                &[frequency.energy()?],
                response_config
                    .as_ref()
                    .expect("electric response controls"),
            )?;
            super::polarizability::tensor_report(&mut report, &values.values[0]);
            report.set(
                "response_reference_fock_defect",
                values.reference_fock_defect,
            );
            report.set("response_xc_angular_history", values.angular_history);
            report.set("response_xc_cache_bounds", values.cache_bound_history);
            let mut labels = report.get("units").cloned().expect("electric units");
            labels.set("response_reference_fock_defect", "hartree");
            labels.set(
                "response_xc_angular_history",
                "[dimensionless, hartree, hartree]",
            );
            labels.set(
                "response_xc_cache_bounds",
                "[dimensionless, hartree, hartree]",
            );
            report.set("units", labels);
            report.set("model", "adiabatic complex spin-covariant GKS/GHF susceptibility; full paired projector response of all spin/AO blocks; Coulomb, full/range exchange, and the same globally projected semilocal density/gradient/tau/current Hessian as the reference; fixed additional AO source");
        }
    } else if let Some(model) = correlation.as_ref().filter(|_| id == "gradient") {
        super::correlated_nuclear::append_spinor_gradient(
            &mut report,
            system,
            &setup,
            &scf,
            &operator,
            model,
            response_config
                .as_ref()
                .expect("nuclear response configuration"),
        )?;
    } else if let Some(model) = correlation.as_ref().filter(|_| correlated_high_order) {
        super::correlated_electric::spinor_high_order::append(
            &mut report,
            &setup,
            &scf,
            &operator,
            model,
            if id == "beta" { 3 } else { 4 },
            response_config
                .as_ref()
                .expect("electric response configuration"),
        )?;
    } else if let Some(request) = electric_tensor {
        let config = response_config
            .as_ref()
            .expect("electric response configuration");
        let context = crate::response::nuclear::SpinorNuclearContext::new(
            &setup.molecule,
            &setup.basis,
            setup.grid.as_ref(),
            &setup.functional,
            &scf,
            &setup.options,
            &operator,
            4,
            config,
        )?;
        let mut diagnostics = super::nuclear_tensor::NuclearDiagnostics::default();
        report.set(
            "reference_field",
            setup
                .options
                .external_field
                .map_or([0.0; 3], |v| v.to_array()),
        );
        let mut labels = report.get("units").cloned().expect("reference units");
        labels.set("reference_field", "hartree/e/bohr");
        labels.set("residuals", "dimensionless");
        report.set("units", labels);
        request.apply(&mut report, config, |direction, order| {
            let value = context.molecular_energy_taylor(direction, order)?;
            diagnostics.response.extend(value.residuals);
            diagnostics.projector.extend(value.projector_defects);
            if value.thermal_chemical_potentials.is_empty() {
                diagnostics.stationarity.extend(value.stationarity_defects);
            } else {
                diagnostics
                    .thermal_stationarity
                    .extend(value.stationarity_defects);
                diagnostics
                    .thermal_number
                    .extend(value.thermal_number_defects);
                diagnostics
                    .thermal_chemical
                    .push(value.thermal_chemical_potentials);
            }
            diagnostics
                .reference
                .extend([value.reference_defect, value.reference_energy_defect]);
            diagnostics
                .angular_history
                .extend(value.angular_refinement_history);
            diagnostics
                .cache_bounds
                .extend(value.angular_cache_bound_history);
            Ok(value.energy)
        })?;
        report.set("residuals", diagnostics.response.clone());
        diagnostics.append(&mut report, config);
    } else if let Some(config) = response_config {
        let order = tensor_selection
            .as_ref()
            .map_or(if id == "hessian" { 2 } else { 1 }, |s| s.order);
        let mut diagnostics = super::nuclear_tensor::NuclearDiagnostics::default();
        config.validate()?;
        let direct_gradient = if order == 1 {
            crate::gradient::spinor::stationary_gradient(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &scf,
                &setup.options,
            )?
        } else {
            None
        };
        let nuclear = direct_gradient
            .is_none()
            .then(|| {
                crate::response::nuclear::SpinorNuclearContext::new(
                    &setup.molecule,
                    &setup.basis,
                    setup.grid.as_ref(),
                    &setup.functional,
                    &scf,
                    &setup.options,
                    &operator,
                    order,
                    &config,
                )
            })
            .transpose()?;
        if direct_gradient.is_some() {
            // These are fresh reference diagnostics. A variational first
            // derivative does not require a first-order orbital solve.
            let fresh = backend.evaluate(&scf.density)?;
            diagnostics.reference.extend([
                fresh.energy - scf.fields.energy,
                fresh.fock[0]
                    .re
                    .max_abs_diff(&scf.fields.fock[0].re)
                    .max(fresh.fock[0].im.max_abs_diff(&scf.fields.fock[0].im)),
            ]);
            diagnostics
                .projector
                .extend([scf.metric_defect, scf.occupation_consistency]);
            diagnostics.stationarity.push(scf.orbital_residual);
        }
        let mut evaluate = |direction: Vec<[f64; 3]>| -> Result<Vec<f64>> {
            if let Some(gradient) = &direct_gradient {
                let coefficient = gradient
                    .iter()
                    .zip(direction.iter().flatten())
                    .map(|(g, d)| g * d)
                    .sum::<f64>();
                return Ok(vec![scf.free_energy, coefficient]);
            }

            let value = nuclear
                .as_ref()
                .expect("moving spinor geometry provider")
                .energy_taylor(&direction, order)?;
            diagnostics.response.extend(value.residuals);
            diagnostics.projector.extend(value.projector_defects);
            if value.thermal_chemical_potentials.is_empty() {
                diagnostics.stationarity.extend(value.stationarity_defects);
            } else {
                diagnostics
                    .thermal_stationarity
                    .extend(value.stationarity_defects);
                diagnostics
                    .thermal_number
                    .extend(value.thermal_number_defects);
                diagnostics
                    .thermal_chemical
                    .push(value.thermal_chemical_potentials);
            }
            diagnostics
                .reference
                .extend([value.reference_defect, value.reference_energy_defect]);
            diagnostics
                .angular_history
                .extend(value.angular_refinement_history);
            diagnostics
                .cache_bounds
                .extend(value.angular_cache_bound_history);
            Ok(value.energy)
        };
        if let Some(selection) = tensor_selection {
            let (tensors, evaluations) = super::force_constants::assemble_energy_paths(
                &selection,
                3,
                setup.options.incore_mb,
                |d, n| {
                    let direction: Vec<_> = d.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
                    let energy = evaluate(direction.clone())?;
                    let dispersion = super::nuclear_tensor::dispersion_path(
                        &setup, system, &direction, n, None,
                    )?[n];
                    Ok(vec![energy[n] + dispersion, energy[n], dispersion])
                },
            )?;
            super::force_constants::append_energy_tensors(
                &mut report,
                &selection,
                tensors,
                evaluations,
                &["", "parent_", "dispersion_"],
                &config,
            )?;
            report.set("nuclear_path_evaluations", evaluations);
        } else {
            let tensors =
                super::nuclear_tensor::assemble(setup.molecule.len(), order, 1, |direction| {
                    Ok(vec![evaluate(direction)?])
                })?;
            let mut gradient = tensors.gradients[0].clone();
            let mut hessian = tensors.hessians[0].clone();
            let name = setup.method.dispersion_functional_name();
            if setup.dispersion != crate::dispersion::Dispersion::None {
                let dg = crate::dispersion::dispersion_gradient(
                    &setup.molecule,
                    &setup.dispersion,
                    &name,
                )?;
                for (g, v) in gradient.chunks_exact_mut(3).zip(dg) {
                    for (a, b) in g.iter_mut().zip(v.to_array()) {
                        *a += b;
                    }
                }
                if order == 2 {
                    let dh = crate::dispersion::dispersion_hessian(
                        &setup.molecule,
                        &setup.dispersion,
                        &name,
                    )?;
                    for (h, d) in hessian.as_mut_slice().iter_mut().zip(dh.as_slice()) {
                        *h += d;
                    }
                }
            }
            let rows = gradient
                .chunks_exact(3)
                .map(|r| r.to_vec())
                .collect::<Vec<_>>();
            report.set("gradient", rows.clone());
            report.set(
                "forces",
                rows.iter()
                    .map(|r| r.iter().map(|v| -v).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            if order == 2 {
                report.set("hessian", hessian);
            }
            report.set("nuclear_path_evaluations", tensors.paths);
            let mut labels = report.get("units").cloned().expect("nuclear units");
            labels.set("gradient", "hartree/bohr");
            labels.set("forces", "hartree/bohr");
            labels.set("hessian", "hartree/bohr^2");
            report.set("units", labels);
        }
        let reference_fock_defects = diagnostics.reference.clone();
        diagnostics.append(&mut report, &config);
        report.set("reference_fock_defects", reference_fock_defects);
        report.set(
            "reference_field",
            setup
                .options
                .external_field
                .map_or([0.0; 3], |v| v.to_array()),
        );
        report.set("provenance", "analytic");
        report.set("nuclear_model", "complex stationary projector response with full metric/Pulay, moving Becke grid and both integral endpoints; Coulomb, full/range exchange and globally projected semilocal density/gradient/tau/current paths; stationary 2n+1 energy and explicit empirical-dispersion coefficients through fourth order; scalar local/nonlocal ECP basis and operator centres differentiated analytically; additional AO spin operator held fixed; spin-orbit ECP, nonlocal correlation and solvent nuclear providers remain separate pending connections");
        if setup.options.electronic_temperature > 0.0 {
            report.set("nuclear_model", "analytic geometry derivatives of canonical spinor Mermin free energy at fixed temperature and total electron count");
            report.set(
                "energy_derivative_potential",
                "electronic Mermin free energy at fixed temperature and total electron count",
            );
        } else if direct_gradient.is_some() {
            report.set("nuclear_model", "stationary spinor gradient with energy-weighted metric/Pulay, analytic Coulomb, full/range exchange, scalar ECP and fixed electric-source derivatives; exactly real spin-diagonal references use the native AO/moving-grid semilocal and conditional VV10 contractions; general quadratic references use all complex Pauli exchange channels; the additional AO spin operator is held fixed; projector and stationarity diagnostics describe the fresh reference");
        }
        let mut labels = report.get("units").cloned().expect("report units");
        labels.set("reference_fock_defects", "hartree");
        labels.set("reference_field", "hartree/e/bohr");
        report.set("units", labels);
    }
    report.set("xc_grid_number_defect", scf.fields.xc_grid_number_defect);
    if let Some(request) = quadrupole {
        if let Some(model) = correlation {
            request.append_correlated_spinor(&mut report, &setup, &scf, &operator, model)?;
        } else {
            request.append_spinor(&mut report, &setup, &scf, &operator)?;
        }
    } else if let Some(model) = correlation.filter(|_| id == "energy") {
        let pairs = crate::response::nuclear::stationary_spinor_pt2(
            &setup.basis,
            &scf,
            model.memory_mb.min(setup.options.incore_mb),
        )?;
        let mut target_correction = 0.0;
        if let Some(target) = &model.target {
            let mut target_options = setup.options.clone();
            target_options.vv10 = model.target_vv10.map(|(_, parameters)| parameters);
            let fields = MolecularSpinorBackend::new(
                &setup.molecule,
                &setup.basis,
                target,
                setup.grid.as_ref(),
                &seed.scf,
                &target_options,
                operator,
                axis,
                MolecularSpinorFunctionalModel::SpinCovariant,
                xc_tolerance,
            )?
            .evaluate(&scf.density)?;
            if !fields.operator_converged {
                let diagnostics = fields
                    .xc_diagnostics
                    .expect("nonconverged spinor target grid");
                return Err(DftError::ResponseNotConverged {
                    iterations: diagnostics.angular_order,
                    residual: diagnostics
                        .angular_energy_defect
                        .max(diagnostics.angular_operator_defect)
                        .max(diagnostics.hermiticity_defect)
                        .max(diagnostics.component_energy_defect),
                });
            }
            target_correction = fields.energy - scf.fields.energy;
            if let Some((scale, parameters)) = model.target_vv10 {
                if scale != 1.0 {
                    // The fixed-density target provider includes VV10 with
                    // unit weight. Apply the target recipe's residual factor.
                    let grid = target_options.nonlocal_grid(&setup.molecule)?;
                    target_correction += (scale - 1.0)
                        * crate::nlc::vv10_energy(&setup.basis, &grid, &pauli[0].re, parameters)?;
                }
            }
        }
        super::double_hybrid::append_spinor_energy(
            &mut report,
            &model,
            pairs.energy[0].re,
            target_correction,
        );
        report.set("model", "Noncollinear double-hybrid energy on an integer full-spin reference. A common coefficient scales generalized PT2. A selected distinct target is evaluated on the converged parent density.");
        report.set("generalized_pt2_amplitude_residual", pairs.residuals[0]);
        report.set(
            "generalized_pt2_relative_residual",
            pairs.relative_residuals[0],
        );
        report.set("generalized_pt2_imaginary_energy", pairs.energy[0].im);
        let mut labels = report.get("units").cloned().expect("spinor energy units");
        labels.set("generalized_pt2_amplitude_residual", "hartree");
        labels.set("generalized_pt2_relative_residual", "dimensionless");
        labels.set("generalized_pt2_imaginary_energy", "hartree");
        report.set("units", labels);
    }
    if let Some(d) = scf.fields.xc_diagnostics {
        report.set("xc_angular_history", d.angular_refinement_history);
    }
    if let Some(request) = moments {
        // Real symmetric electric AO operators contract with the real
        // scalar Pauli component of the fully relaxed spinor density.
        request.append(&mut report, &setup, &pauli[0].re);
    }
    super::density_dispersion::report_d1(
        &setup,
        seed.e_dispersion,
        || scf.orbital_residual,
        &mut report,
    );
    if let Some(request) = natural_bond {
        return request.report(report, &setup, s, p, &scf.fields.fock[0], axis);
    }
    if let Some(request) = igmh {
        report.set("scf_orbital_residual", scf.orbital_residual);
        request.append(&mut report, &setup, &pauli[0].re)?;
    }
    Ok(report)
}
