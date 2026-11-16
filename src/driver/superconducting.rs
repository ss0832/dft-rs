// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical static band-decoupled gap and analytic response on supplied bands.
use super::pairing::{append, direction, matrix_direction, required, rows};
use super::{Controls, Value};
use crate::dual::{Scalar, Taylor};
use crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
use crate::properties::pairing::Lm2005OffDiagonal;
use crate::response::complex::Complex;
use crate::response::dynamic::Frequency;
use crate::response::nambu::{nambu_causal_response, NambuProbe, NambuReference};
use crate::response::superconducting::{
    critical_temperature, select_band_phase, solve_gap_taylor, BandPhase, GapFunctional,
    GapFunctionalData, GapState, Lm2005BandFunctional, NormalBandSpectrum, PhononBandFunctional,
    SuppliedBandFunctional,
};
use crate::{DftError, Result};

enum Provider {
    Supplied(SuppliedBandFunctional),
    Phonon(PhononBandFunctional),
    Lm2005(Lm2005BandFunctional),
}
impl GapFunctional for Provider {
    fn bands(&self) -> usize {
        match self {
            Self::Supplied(p) => p.bands(),
            Self::Phonon(p) => p.bands(),
            Self::Lm2005(p) => p.bands(),
        }
    }
    fn evaluate<S: Scalar + Send + Sync>(
        &self,
        p: &[S],
        mu: S,
        re: &[S],
        im: &[S],
    ) -> Result<GapFunctionalData<S>> {
        match self {
            Self::Supplied(f) => f.evaluate(p, mu, re, im),
            Self::Phonon(f) => f.evaluate(p, mu, re, im),
            Self::Lm2005(f) => f.evaluate(p, mu, re, im),
        }
    }
}
pub(super) fn run(c: &Controls) -> Result<Value> {
    let model = c.str("pairing_model", "supplied")?;
    let select_phase = c.bool("scdft_phase_selection", false)?;
    let temperature_bracket = c
        .opt_f64_list("scdft_temperature_bracket")?
        .map(|values| {
            <[f64; 2]>::try_from(values).map_err(|_| {
                DftError::InvalidInput(
                    "scdft_temperature_bracket requires a cold and a hot temperature in Kelvin"
                        .into(),
                )
            })
        })
        .transpose()?;
    if select_phase && temperature_bracket.is_some() {
        return Err(DftError::InvalidInput(
            "scdft_phase_selection uses one electronic_temperature; omit scdft_temperature_bracket"
                .into(),
        ));
    }
    if select_phase && model != "supplied" {
        return Err(DftError::Unsupported {
            feature: format!(
                "canonical superconducting phase selection with pairing_model={model}"
            ),
            missing:
                "a variational free-energy functional consistent with the gap and number equations"
                    .into(),
        });
    }
    if model == "nambu" {
        if temperature_bracket.is_some() {
            return Err(DftError::Unsupported {
                feature: "SCDFT critical temperature with a supplied Nambu operator".into(),
                missing: "a self-consistent linearized pairing provider".into(),
            });
        }
        return nambu(c);
    }
    let energies = required(c, "band_energies")?;
    let n = energies.len();
    if n == 0 || n.checked_mul(n).is_none() {
        return Err(DftError::InvalidInput(
            "superconducting state needs a nonempty finite band dimension".into(),
        ));
    }
    let weights = required(c, "band_weights")?;
    let generated_mass = model == "lm2005";
    let mass = if generated_mass {
        if c.contains("mass_renormalization") || c.contains("mass_renormalization_direction") {
            return Err(DftError::InvalidInput("lm2005 generates its own phonon mass; explicit mass inputs require lm2005-off-diagonal or supplied".into()));
        }
        Vec::new()
    } else {
        required(c, "mass_renormalization")?
    };
    if weights.len() != n || (!generated_mass && mass.len() != n) {
        return Err(DftError::InvalidInput(
            "band_weights and mass_renormalization require one entry per band".into(),
        ));
    }
    let temperature = match temperature_bracket {
        Some(bounds) => bounds[0],
        None => c.opt_f64("electronic_temperature")?.ok_or_else(|| {
            DftError::InvalidInput("electronic_temperature is required in Kelvin".into())
        })?,
    };
    let temperature_direction = c.f64("temperature_direction", 0.0)?;
    let number = c
        .opt_f64("electron_number")?
        .ok_or_else(|| DftError::InvalidInput("electron_number is required".into()))?;
    let dnumber = c.f64("electron_number_direction", 0.0)?;
    let mut parameters = Vec::new();
    append(
        &mut parameters,
        &energies,
        &direction(c, "band_energy_direction", n)?,
    )?;
    append(
        &mut parameters,
        &weights,
        &direction(c, "band_weight_direction", n)?,
    )?;
    if !generated_mass {
        append(
            &mut parameters,
            &mass,
            &direction(c, "mass_renormalization_direction", n)?,
        )?;
    }

    let temperature_index = parameters.len();
    append(
        &mut parameters,
        &[temperature * BOLTZMANN_HARTREE_PER_K],
        &[temperature_direction * BOLTZMANN_HARTREE_PER_K],
    )?;

    let provider=match model.as_str() {
        "supplied"=>{
            append(&mut parameters,&rows(c,"pairing_kernel",n)?,&matrix_direction(c,"pairing_kernel_direction",n)?)?;
            Provider::Supplied(SuppliedBandFunctional {bands:n})
        },
        "lm2005-off-diagonal" | "lm2005"=>{
            let phonons=required(c,"phonon_energies")?;let modes=phonons.len();
            let vertices=c.opt_rows("pair_couplings_squared")?.ok_or_else(||DftError::InvalidInput("pair_couplings_squared is required".into()))?;
            let derivatives=c.opt_rows("pair_couplings_squared_direction")?.unwrap_or_else(||vec![vec![0.0;n*n];modes]);
            if vertices.len()!=modes || derivatives.len()!=modes || vertices.iter().chain(&derivatives).any(|r|r.len()!=n*n) {
                return Err(DftError::InvalidInput("squared phonon vertices and their directions need m flattened n*n rows".into()));
            }
            append(&mut parameters,&phonons,&direction(c,"phonon_energy_direction",modes)?)?;
            append(&mut parameters,&vertices.into_iter().flatten().collect::<Vec<_>>(),&derivatives.into_iter().flatten().collect::<Vec<_>>())?;
            append(&mut parameters,&rows(c,"screened_pair_coulomb",n)?,&matrix_direction(c,"screened_pair_coulomb_direction",n)?)?;
            if generated_mass {Provider::Lm2005(Lm2005BandFunctional {bands:n,modes,quadrature:Lm2005OffDiagonal::default()})}
            else {Provider::Phonon(PhononBandFunctional {bands:n,modes,quadrature:Lm2005OffDiagonal::default()})}
        },
        _=>return Err(DftError::InvalidInput("pairing_model must be supplied, lm2005-off-diagonal (explicit Z), or lm2005 (generated phonon K and Z)".into())),
    };
    let order = c.usize("response_order", 0)?;
    let config = super::vibrational::response_config(c)?;
    if let Some(bracket) = temperature_bracket {
        if order != 0 || dnumber != 0.0 || parameters.iter().any(|p| p.c[1] != 0.0) {
            return Err(DftError::Unsupported {
                feature: "Taylor derivatives of the SCDFT critical temperature".into(),
                missing: "an implicit derivative of the normal-state pairing eigenvalue crossing"
                    .into(),
            });
        }
        c.finish("superconducting-state critical temperature")?;
        let base: Vec<f64> = parameters.iter().map(|p| p.value()).collect();
        let thermal_bracket = bracket.map(|t| t * BOLTZMANN_HARTREE_PER_K);
        let result = critical_temperature(
            &provider,
            NormalBandSpectrum {
                energies: &energies,
                weights: &weights,
                electron_number: number,
            },
            thermal_bracket,
            &config,
            |kt| {
                let mut trial = base.clone();
                trial[temperature_index] = kt;
                trial
            },
        )?;
        let units = Value::object()
            .with("critical_temperature_k", "K")
            .with("critical_temperature_bracket_k", "K")
            .with("chemical_potential", "hartree")
            .with("leading_pairing_eigenvalue", "dimensionless")
            .with("linearized_gap_mode", "dimensionless")
            .with("normal_occupations", "dimensionless")
            .with("mass_renormalization", "dimensionless")
            .with("linearized_gap_residual", "dimensionless")
            .with("pairing_eigenvector_residual", "dimensionless")
            .with("number_residual", "dimensionless")
            .with("kernel_asymmetry", "hartree")
            .with("temperature_evaluations", "integer")
            .with("electron_number", "dimensionless");
        return Ok(Value::object()
            .with("model", "canonical static band-decoupled singlet SCDFT normal-state linear pairing onset on fixed bands")
            .with("pairing_model", model)
            .with("converged", true)
            .with("critical_temperature_k", result.temperature_energy / BOLTZMANN_HARTREE_PER_K)
            .with("critical_temperature_bracket_k", result.temperature_energy_bracket.map(|x| x / BOLTZMANN_HARTREE_PER_K).to_vec())
            .with("chemical_potential", result.chemical_potential)
            .with("leading_pairing_eigenvalue", result.leading_eigenvalue)
            .with("linearized_gap_mode", result.gap_mode)
            .with("normal_occupations", result.occupations)
            .with("mass_renormalization", result.mass_renormalization)
            .with("linearized_gap_residual", result.gap_residual)
            .with("pairing_eigenvector_residual", result.eigen_residual)
            .with("number_residual", result.number_residual)
            .with("kernel_asymmetry", result.kernel_asymmetry)
            .with("temperature_evaluations", result.evaluations)
            .with("electron_number", number)
            .with("units", units));
    }
    let (result, zero_seed, phase_fields) = if select_phase {
        if order != 0 || dnumber != 0.0 || parameters.iter().any(|p| p.c[1] != 0.0) {
            return Err(DftError::Unsupported {
                feature: "derivatives of superconducting thermodynamic phase selection".into(),
                missing: "phase-local free-energy response and handling of the phase boundary"
                    .into(),
            });
        }
        c.finish("superconducting-state phase selection")?;
        let Provider::Supplied(supplied) = &provider else {
            unreachable!("phase selection requires the supplied provider");
        };
        let base: Vec<_> = parameters.iter().map(|p| p.value()).collect();
        let phase = select_band_phase(supplied, &base, number, &config)?;
        let fields = Value::object()
            .with(
                "selected_phase",
                match phase.phase {
                    BandPhase::Normal => "normal",
                    BandPhase::Superconducting => "superconducting",
                },
            )
            .with("normal_chemical_potential", phase.normal_chemical_potential)
            .with("normal_number_residual", phase.normal_number_residual)
            .with("normal_pairing_eigenvalue", phase.normal_pairing_eigenvalue)
            .with("pairing_stability_margin", phase.pairing_stability_margin)
            .with("condensation_free_energy", phase.condensation_free_energy)
            .with("pairing_energy", phase.pairing_energy)
            .with("entropy_over_boltzmann", phase.entropy_over_boltzmann)
            .with(
                "normal_entropy_over_boltzmann",
                phase.normal_entropy_over_boltzmann,
            );
        (phase.solution, false, Some(fields))
    } else {
        let mu = c.opt_f64("chemical_potential")?.ok_or_else(|| {
            DftError::InvalidInput("initial chemical_potential is required".into())
        })?;
        let real = required(c, "initial_gap_real")?;
        let imaginary = c
            .opt_f64_list("initial_gap_imaginary")?
            .unwrap_or_else(|| vec![0.0; n]);
        if real.len() != n || imaginary.len() != n {
            return Err(DftError::InvalidInput(
                "initial complex gap requires one entry per band".into(),
            ));
        }
        c.finish("superconducting-state")?;
        let initial = GapState {
            gap: real
                .iter()
                .zip(&imaginary)
                .map(|(&re, &im)| Complex::new(re, im))
                .collect(),
            chemical_potential: mu,
        };
        let zero_seed = initial.gap.iter().all(|z| z.re == 0.0 && z.im == 0.0);
        (
            solve_gap_taylor(
                &provider,
                &parameters,
                Taylor::from_coefficients(&[number, dnumber]),
                &initial,
                order,
                &config,
            )?,
            zero_seed,
            None,
        )
    };
    let warnings = if zero_seed {
        vec!["the zero seed selects the normal stationary solution; thermodynamic stability against superconductivity is not established".to_string()]
    } else {
        Vec::new()
    };
    let description = if select_phase {
        "canonical variational singlet model on fixed bands; reciprocal interaction K/(1+Z) with uniform supplied Z"
    } else {
        "canonical static band-decoupled singlet SCDFT gap on supplied normal bands; physical selected diagonal Z; real reciprocal K; supplied normal orbitals and phonons with explicit optional parameter paths"
    };
    let mut report=Value::object().with("model",description)
         .with("pairing_model",model).with("mass_model",if generated_mass {
            "LM2005 phonon mass from the same vertices, weights, responding mu and temperature; particle-hole-symmetric mass approximation; no explicit mass input"
        } else {"explicit supplied diagonal normal-state mass renormalization; no implicit zero-mass approximation"})
        .with("converged",true).with("warnings",warnings).with("response_order",order)
        .with("gap_real",result.gap_real[0].clone()).with("gap_imaginary",result.gap_imaginary[0].clone())
        .with("gap_coefficients_real",result.gap_real).with("gap_coefficients_imaginary",result.gap_imaginary)
        .with("chemical_potential",result.chemical_potential[0]).with("chemical_potential_coefficients",result.chemical_potential)
        .with("anomalous_density_coefficients_real",result.anomalous_real).with("anomalous_density_coefficients_imaginary",result.anomalous_imaginary)
        .with("occupation_coefficients",result.occupations).with("quasiparticle_energies",result.quasiparticle_energies)
        .with("quasiparticle_energy_squared_coefficients",result.quasiparticle_energy_squared)
        .with("pairing_kernel_coefficients",result.kernel_coefficients).with("kernel_asymmetry",result.kernel_asymmetry)
        .with("gap_residuals",result.gap_residuals).with("number_residuals",result.number_residuals)
        .with("phase_residuals",result.phase_residuals).with("phase_multipliers",result.phase_multipliers).with("goldstone_defects",result.goldstone_defects)
        .with("linear_residuals",result.linear_residuals).with("linear_iterations",result.linear_iterations)
        .with("reference_iterations",result.reference_iterations).with("reference_scaled_residual",result.reference_scaled_residual)
        .with("energy_scale",result.energy_scale).with("electron_number",number).with("band_energies",energies).with("band_weights",weights)
        .with("mass_renormalization",result.mass_renormalization_coefficients[0].clone())
        .with("mass_renormalization_coefficients",result.mass_renormalization_coefficients).with("electronic_temperature",temperature)
        .with("phase_reference",result.phase_reference.iter().map(|z|vec![z.re,z.im]).collect::<Vec<_>>())
        .with("coefficient_convention","factorial-normalized coefficients along supplied linear parameter directions in dimensionless t; each band has two spin states; chi=Delta*tanh(E/(2kT))/(2E); global phase fixed relative to the seed; E-squared coefficients are retained at normal-state zeros")
        .with("linear_method","matrix-free GMRES with exact AD Jacobian actions and a bordered phase constraint");
    let mut units = Value::object();
    for name in [
        "gap_real",
        "gap_imaginary",
        "gap_coefficients_real",
        "gap_coefficients_imaginary",
        "chemical_potential",
        "chemical_potential_coefficients",
        "quasiparticle_energies",
        "pairing_kernel_coefficients",
        "kernel_asymmetry",
        "gap_residuals",
        "phase_residuals",
        "phase_multipliers",
        "goldstone_defects",
        "energy_scale",
        "band_energies",
    ] {
        units.set(name, "hartree");
    }
    for name in [
        "anomalous_density_coefficients_real",
        "anomalous_density_coefficients_imaginary",
        "occupation_coefficients",
        "number_residuals",
        "linear_residuals",
        "reference_scaled_residual",
        "electron_number",
        "band_weights",
        "mass_renormalization",
        "mass_renormalization_coefficients",
        "phase_reference",
    ] {
        units.set(name, "dimensionless");
    }
    units.set("quasiparticle_energy_squared_coefficients", "hartree^2");
    units.set("electronic_temperature", "K");
    if let Some(fields) = phase_fields {
        let Value::Object(fields) = fields else {
            unreachable!("thermodynamic phase fields are an object");
        };
        for (name, value) in fields {
            report.set(&name, value);
        }
        for name in [
            "normal_chemical_potential",
            "condensation_free_energy",
            "pairing_energy",
        ] {
            units.set(name, "hartree");
        }
        for name in [
            "normal_number_residual",
            "normal_pairing_eigenvalue",
            "pairing_stability_margin",
            "entropy_over_boltzmann",
            "normal_entropy_over_boltzmann",
        ] {
            units.set(name, "dimensionless");
        }
    }
    report.set("units", units);
    Ok(report)
}

/// Full interband density response on explicitly supplied Nambu operators.
/// The existing band-gap path retains its separate canonical self-consistency.
fn nambu(c: &Controls) -> Result<Value> {
    use crate::linalg::Matrix;
    use crate::periodic::BlochMatrix;
    use crate::response::nambu::nambu_density_taylor;
    let n = c
        .opt_rows("nambu_normal_real")?
        .ok_or_else(|| {
            DftError::InvalidInput("nambu_normal_real is required for pairing_model=nambu".into())
        })?
        .len();
    if n == 0 || n.checked_mul(n).is_none() {
        return Err(DftError::InvalidInput(
            "Nambu electronic dimension must be positive and addressable".into(),
        ));
    }
    let frequencies = c.opt_f64_list("frequencies")?;
    let order = c.usize("response_order", if frequencies.is_some() { 0 } else { 4 })?;
    let config = super::vibrational::response_config(c)?;
    let cutoff = c.f64("metric_eigenvalue_cutoff", 1e-12)?;
    let matrix = |key: &str, identity: bool, required_input: bool| -> Result<Matrix> {
        let values = if c.contains(key) || required_input {
            rows(c, key, n)?
        } else {
            (0..n * n)
                .map(|p| if identity && p / n == p % n { 1.0 } else { 0.0 })
                .collect()
        };
        Ok(Matrix::from_row_major(n, n, values))
    };
    let path = |prefix: &str, identity: bool, required_input: bool| -> Result<Vec<BlochMatrix>> {
        let mut result = vec![BlochMatrix {
            re: matrix(&format!("{prefix}_real"), identity, required_input)?,
            im: matrix(&format!("{prefix}_imaginary"), false, false)?,
        }];
        if frequencies.is_none() {
            result.push(BlochMatrix {
                re: matrix(&format!("{prefix}_direction_real"), false, false)?,
                im: matrix(&format!("{prefix}_direction_imaginary"), false, false)?,
            });
        }
        Ok(result)
    };
    let mut normal = path("nambu_normal", false, true)?;
    let mut pair = path("nambu_pairing", false, true)?;
    let mut overlap = path("nambu_overlap", true, false)?;
    if order == 0 {
        normal.truncate(1);
        pair.truncate(1);
        overlap.truncate(1);
    }
    let mu = c.opt_f64("chemical_potential")?.ok_or_else(|| {
        DftError::InvalidInput("chemical_potential is required for Nambu density".into())
    })?;
    let temperature = c.opt_f64("electronic_temperature")?.ok_or_else(|| {
        DftError::InvalidInput("electronic_temperature is required in Kelvin".into())
    })?;
    if let Some(frequencies) = frequencies {
        if order != 0 {
            return Err(DftError::Unsupported {
                feature: "Taylor derivatives of the external-frequency Nambu susceptibility".into(),
                missing: "response of the Liouville operator and its induced fields".into(),
            });
        }
        return nambu_frequency_report(
            c,
            &frequencies,
            NambuReference {
                normal: &normal[0],
                pairing: &pair[0],
                overlap: &overlap[0],
                chemical_potential: mu,
                temperature_energy: temperature * BOLTZMANN_HARTREE_PER_K,
                overlap_cutoff: cutoff,
            },
            &config,
        );
    }
    let dmu = c.f64("chemical_potential_direction", 0.0)?;
    let dtemperature = c.f64("temperature_direction", 0.0)?;
    c.finish("superconducting-state")?;
    let (mu, kt) = if order == 0 {
        (vec![mu], vec![temperature * BOLTZMANN_HARTREE_PER_K])
    } else {
        (
            vec![mu, dmu],
            vec![
                temperature * BOLTZMANN_HARTREE_PER_K,
                dtemperature * BOLTZMANN_HARTREE_PER_K,
            ],
        )
    };
    let result = nambu_density_taylor(&normal, &pair, &overlap, &mu, &kt, order, cutoff, &config)?;
    let mut report=Value::object()
        .with("model","equilibrium full complex grand-canonical Nambu density on supplied normal and antisymmetric pairing operators with interband pairing and moving metric")
        .with("pairing_model","nambu").with("ensemble","grand_canonical")
        .with("converged",true).with("response_order",order).with("electronic_dimension",n)
        .with("chemical_potential",mu[0]).with("electronic_temperature",temperature)
        .with("electron_number_coefficients",result.electron_number)
        .with("particle_hole_defects",result.particle_hole_defects)
        .with("pairing_antisymmetry_defects",result.pairing_antisymmetry_defects)
        .with("number_imaginary_defects",result.number_imaginary_defects)
        .with("metric_defects",result.thermal.metric_defects)
        .with("density_hermiticity_defects",result.thermal.hermiticity_defects)
        .with("relative_commutator_defects",result.thermal.relative_commutator_defects)
        .with("complex_embedding_defects",result.thermal.complex_embedding_defects)
        .with("maximum_inverse_residual",result.thermal.maximum_inverse_residual)
        .with("exponential_tail_estimate",result.thermal.exponential_tail_estimate)
        .with("exponential_terms",result.thermal.exponential_terms)
        .with("doubling_steps",result.thermal.doubling_steps)
        .with("coefficient_convention","factorial-normalized coefficients in dimensionless t; full electronic indices include spin, all electrons followed by conjugate holes; H_N=[[h-mu*S,Delta],[Delta^H,-(h-mu*S)*]], S_N=diag(S,S*); contravariant R=C f C^H, normal rho=R_ee, anomalous chi=-R_eh=<c_i c_j>, N=Tr(S*rho)")
        .with("warnings",vec!["Supplied normal/pairing operators define a kinematic grand-canonical density component. Canonical gap, number and phase self-consistency uses the separate supplied/lm2005 band providers; full normal/pairing functional backreaction remains required.".to_string()]);
    let mut units = Value::object();
    for (prefix, values, unit) in [
        ("nambu_density", &result.thermal.density, "dimensionless"),
        ("normal_density", &result.normal_density, "dimensionless"),
        (
            "anomalous_density",
            &result.anomalous_density,
            "dimensionless",
        ),
        ("nambu_hamiltonian", &result.hamiltonian, "hartree"),
        ("nambu_overlap", &result.overlap, "dimensionless"),
    ] {
        for imaginary in [false, true] {
            let key = format!(
                "{prefix}_coefficients_{}",
                if imaginary { "imaginary" } else { "real" }
            );
            report.set(
                &key,
                values
                    .iter()
                    .map(|a| {
                        if imaginary {
                            a.im.clone()
                        } else {
                            a.re.clone()
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            units.set(&key, unit);
        }
    }
    for key in [
        "particle_hole_defects",
        "pairing_antisymmetry_defects",
        "number_imaginary_defects",
        "metric_defects",
        "density_hermiticity_defects",
        "relative_commutator_defects",
        "complex_embedding_defects",
        "maximum_inverse_residual",
        "exponential_tail_estimate",
    ] {
        units.set(key, "dimensionless");
    }
    units.set("electron_number_coefficients", "electrons");
    units.set("chemical_potential", "hartree");
    units.set("electronic_temperature", "K");
    report.set("units", units);
    Ok(report)
}

fn nambu_frequency_report(
    c: &Controls,
    frequencies: &[f64],
    reference: NambuReference<'_>,
    config: &crate::response::ResponseSolverConfig,
) -> Result<Value> {
    use crate::linalg::Matrix;
    use crate::periodic::BlochMatrix;
    let n = reference.normal.rows();
    let names = [
        "nambu_probe_normal_real",
        "nambu_probe_normal_imaginary",
        "nambu_probe_pairing_real",
        "nambu_probe_pairing_imaginary",
    ];
    let input = names.map(|name| c.opt_rows(name));
    let input = input.into_iter().collect::<Result<Vec<_>>>()?;
    let count = input
        .iter()
        .find_map(|v| v.as_ref().map(Vec::len))
        .unwrap_or(0);
    if count == 0
        || input
            .iter()
            .flatten()
            .any(|rows| rows.len() != count || rows.iter().any(|row| row.len() != n * n))
    {
        return Err(DftError::InvalidInput(
            "Nambu probe blocks need matching nonempty sets of row-major n*n entries".into(),
        ));
    }
    let probes: Vec<_> = (0..count)
        .map(|i| {
            let part = |index: usize| {
                input[index].as_ref().map_or_else(
                    || Matrix::zeros(n, n),
                    |rows| Matrix::from_row_major(n, n, rows[i].clone()),
                )
            };
            NambuProbe {
                normal: BlochMatrix {
                    re: part(0),
                    im: part(1),
                },
                pairing: BlochMatrix {
                    re: part(2),
                    im: part(3),
                },
            }
        })
        .collect();
    let coupling = if c.contains("nambu_probe_interaction") {
        Some(Matrix::from_row_major(
            count,
            count,
            rows(c, "nambu_probe_interaction", count)?,
        ))
    } else {
        None
    };
    let damping = c.f64("frequency_damping", 0.0)?;
    let frequency_points: Vec<_> = frequencies
        .iter()
        .map(|&omega| Frequency::Retarded { omega, damping })
        .collect();
    c.finish("superconducting-state external-frequency Nambu response")?;
    let result = nambu_causal_response(
        reference,
        &probes,
        &frequency_points,
        coupling.as_ref(),
        config,
    )?;
    let mut report = Value::object()
        .with("model", "collisionless full Nambu susceptibility in a fixed metric at supplied chemical potential")
        .with("pairing_model", "nambu")
        .with("ensemble", "grand_canonical")
        .with("converged", true)
        .with("response_order", 1)
        .with("electronic_dimension", n)
        .with("nambu_response_channels", count)
        .with("frequencies", frequencies.to_vec())
        .with("frequency_damping", damping)
        .with("chemical_potential", reference.chemical_potential)
        .with("electronic_temperature", reference.temperature_energy / BOLTZMANN_HARTREE_PER_K)
        .with("quasiparticle_energies", result.modes)
        .with("bare_spectral_residuals", result.bare_residuals)
        .with("coupled_response_residuals", result.coupled_residuals)
        .with("reference_eigen_residual", result.reference_eigen_residual)
        .with("reference_metric_defect", result.reference_metric_defect)
        .with("isospectral_stability_margin", result.stability_margin)
        .with("static_gram_negative_defect", result.static_gram_negative_defect)
        .with("axis_convention", "frequency, observed probe, driven probe; exp(-i omega t) with +uQ in H and Tr(QR)/2")
        .with("warnings", vec!["Fixed reference occupations, including at zero frequency. Optional supplied separable coupling is the reference energy curvature in probe expectations.".to_string()]);
    let mut units = Value::object();
    for (name, matrices) in [
        ("nambu_bare_susceptibility", result.bare),
        ("nambu_susceptibility", result.response),
    ] {
        let real = format!("{name}_real");
        let imaginary = format!("{name}_imaginary");
        report.set(
            &real,
            matrices.iter().map(|m| m.re.clone()).collect::<Vec<_>>(),
        );
        report.set(
            &imaginary,
            matrices.into_iter().map(|m| m.im).collect::<Vec<_>>(),
        );
        units.set(&real, "1/hartree");
        units.set(&imaginary, "1/hartree");
    }
    for name in [
        "frequencies",
        "frequency_damping",
        "chemical_potential",
        "quasiparticle_energies",
    ] {
        units.set(name, "hartree");
    }
    for name in [
        "bare_spectral_residuals",
        "coupled_response_residuals",
        "reference_eigen_residual",
        "reference_metric_defect",
        "isospectral_stability_margin",
    ] {
        units.set(name, "dimensionless");
    }
    units.set("static_gram_negative_defect", "1/hartree");
    units.set("electronic_temperature", "K");
    report.set("units", units);
    Ok(report)
}
