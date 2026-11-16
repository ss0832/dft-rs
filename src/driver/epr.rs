// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Reference-preserving static EPR routes. The generalized route declares
//! its single-Kramers-pair approximation instead of substituting a collinear
//! reference or identifying a broken-spin determinant with a pure multiplet.

use super::method::{prepare_molecular, run_molecular_reference, MolecularMethod, MolecularSetup};
use super::molecular::units;
use super::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::spinor_scf::SpinorScfConfig;
use crate::periodic::BlochMatrix;
use crate::properties::epr::{common_orbital_g_tensor, common_orbital_hyperfine_tensor, spinor};
use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use crate::{DftError, Result};

fn unsupported(feature: &str, missing: &str) -> DftError {
    DftError::Unsupported {
        feature: feature.into(),
        missing: missing.into(),
    }
}

fn source_matrix(controls: &Controls, key: &str, n: usize) -> Result<Option<Matrix>> {
    let Some(rows) = controls.opt_rows(key)? else {
        return Ok(None);
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
    Ok(Some(Matrix::from_row_major(
        n,
        n,
        rows.into_iter().flatten().collect(),
    )))
}

fn charge_origin(setup: &MolecularSetup) -> Vec3 {
    let mut result = Vec3::zero();
    let mut charge = 0.0;
    for atom in &setup.molecule.atoms {
        result += atom.position * atom.z as f64;
        charge += atom.z as f64;
    }
    result * (1.0 / charge)
}

/// Dedicated common-driver entry for the four explicit EPR references.
/// Shared dispatch calls this before consuming the reference selector.
pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if system.is_periodic() || !matches!(id, "hyperfine" | "g-tensor") {
        return Err(unsupported(
            "reference-preserving static EPR request",
            "an isolated molecule and hyperfine or g-tensor calculation",
        ));
    }
    let reference = controls.str("reference", "roks")?.to_ascii_lowercase();
    let generalized = match reference.as_str() {
        "roks" | "rohf" => false,
        "gks" | "ghf" => true,
        _ => {
            return Err(DftError::InvalidInput(
                "dedicated EPR reference must be roks, rohf, gks or ghf".into(),
            ))
        }
    };
    if super::double_hybrid::requested(controls)? {
        return Err(unsupported(
            "double-hybrid reference-preserving EPR",
            "the post-SCF PT2 magnetic property; the parent SCF tensor is not that property",
        ));
    }
    let mut method = MolecularMethod::from_controls(controls)?;
    if matches!(reference.as_str(), "rohf" | "ghf")
        && method.xc.is_none()
        && method.x.is_none()
        && method.c.is_none()
    {
        method.xc = Some("hf".into());
    }
    if method.multiplicity < 2 {
        return Err(DftError::InvalidInput(
            "static EPR requires an open-shell state with multiplicity above one".into(),
        ));
    }
    if method.stabilize
        || method.solvent.is_some()
        || method.electronic_temperature != 0.0
        || method.occupation.is_some()
        || method.fixed_occupations.is_some()
        || method.electron_offset != [0.0; 2]
    {
        return Err(unsupported("reference-preserving static EPR with solvent, stability following or tracked/fractional occupations", "the corresponding constrained or spinor magnetic reference provider"));
    }
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    if !setup.basis.ecps.is_empty() || setup.options.skala.is_some() {
        return Err(unsupported(
            "reference-preserving EPR with ECP or a learned functional",
            "all-electron magnetic operators and the matching reference kernel",
        ));
    }
    if matches!(reference.as_str(), "rohf" | "ghf")
        && (!setup.functional.parts.is_empty()
            || setup.functional.hybrid.cx != 1.0
            || setup.functional.hybrid.clr != 0.0)
    {
        return Err(DftError::InvalidInput("rohf and ghf require the Hartree-Fock functional; choose roks or gks for a density functional".into()));
    }
    let magnetic = generalized || id == "g-tensor";
    let origin = if magnetic {
        controls
            .opt_vec3("gauge_origin")?
            .unwrap_or_else(|| charge_origin(&setup))
    } else {
        Vec3::zero()
    };
    // This provider has an explicit common-origin model. London requests
    // remain errors until their constrained/spinor phase response exists.
    let gauge = if magnetic {
        controls.str("gauge", "common-origin")?
    } else {
        "common-origin".into()
    };
    if gauge != "common-origin" {
        return Err(unsupported("reference-preserving EPR with London orbitals", "the reference-specific London overlap, phase and magnetic-response provider; this route uses gauge=common-origin"));
    }
    let charges = if magnetic {
        controls.opt_f64_list("soc_charges")?
    } else {
        None
    }
    .unwrap_or_else(|| system.numbers.iter().map(|&z| z as f64).collect());
    if charges.len() != setup.molecule.len() || charges.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "soc_charges must contain one finite value per nucleus".into(),
        ));
    }
    let nuclear_g = if id == "hyperfine" {
        controls.opt_f64_list("nuclear_g")?
    } else {
        None
    };
    if nuclear_g
        .as_ref()
        .is_some_and(|g| g.len() != setup.molecule.len() || g.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "nuclear_g must contain one finite signed nuclear g factor per nucleus".into(),
        ));
    }
    if generalized {
        generalized_route(
            id,
            controls,
            &setup,
            &reference,
            origin,
            &charges,
            nuclear_g.as_deref(),
        )
    } else {
        common_route(
            id,
            controls,
            &setup,
            &reference,
            origin,
            &charges,
            nuclear_g.as_deref(),
        )
    }
}

fn common_route(
    id: &str,
    controls: &Controls,
    setup: &MolecularSetup,
    reference: &str,
    origin: Vec3,
    charges: &[f64],
    nuclear_g: Option<&[f64]>,
) -> Result<Value> {
    let config = if id == "g-tensor" {
        Some(super::vibrational::response_config(controls)?)
    } else {
        None
    };
    controls.finish(id)?;
    let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
    )?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    let mut report = Value::object()
        .with("reference", reference)
        .with("reference_manifold", "common-spatial-orbitals")
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("energy", scf.energy + dispersion)
        .with("e_scf", scf.energy)
        .with("e_dispersion", dispersion)
        .with(
            "energy_scope",
            "SCF plus nuclear repulsion and the selected empirical dispersion",
        )
        .with("converged", scf.converged)
        .with("scf_iterations", scf.iterations)
        .with("spin", 0.5 * (scf.n_alpha - scf.n_beta) as f64)
        .with("n_alpha", scf.n_alpha)
        .with("n_beta", scf.n_beta)
        .with("provenance", "analytic");
    if let Some(config) = config {
        let result = common_orbital_g_tensor(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &scf,
            &orbitals,
            &setup.options,
            origin,
            charges,
            &config,
        )?;
        report.set("model", "common-orbital one-electron Breit-Pauli g tensor");
        report.set("g_tensor", result.tensor);
        report.set(
            "g_shift_ppm",
            std::array::from_fn::<_, 3, _>(|a| {
                std::array::from_fn::<_, 3, _>(|b| {
                    1e6 * (result.tensor[a][b]
                        - if a == b {
                            crate::constants::ELECTRON_G_FACTOR
                        } else {
                            0.0
                        })
                })
            }),
        );
        report.set(
            "g_isotropic",
            (0..3).map(|a| result.tensor[a][a]).sum::<f64>() / 3.0,
        );
        report.set("mass_correction_ppm", 1e6 * result.mass_correction);
        report.set(
            "gauge_correction_ppm",
            result.gauge_correction.map(|row| row.map(|v| 1e6 * v)),
        );
        report.set(
            "spin_orbit_ppm",
            result.spin_orbit.map(|row| row.map(|v| 1e6 * v)),
        );
        report.set("response_residuals", result.response_residuals);
        report.set("response_converged", true);
        report.set("reference_stationarity", result.reference_stationarity);
        report.set("metric_defect", result.metric_defect);
        report.set("gauge", "common-origin");
        report.set("gauge_origin", origin);
        report.set("soc_charges", charges.to_vec());
        report.set("warnings", vec!["Common-origin finite-basis tensor; two-electron SOC and London response are absent."]);
        report.set(
            "units",
            units(&[
                ("energy", "hartree"),
                ("g_tensor", "dimensionless, [field axis][spin axis]"),
                ("g_isotropic", "dimensionless"),
                ("g_shift_ppm", "ppm, g minus free-electron g"),
                ("gauge_origin", "bohr"),
                ("soc_charges", "e"),
                ("spin", "hbar"),
                ("mass_correction_ppm", "ppm"),
                ("gauge_correction_ppm", "ppm"),
                ("spin_orbit_ppm", "ppm"),
                ("response_residuals", "dimensionless"),
                ("reference_stationarity", "hartree"),
                ("metric_defect", "dimensionless"),
            ]),
        );
    } else {
        let tensors = setup
            .molecule
            .atoms
            .iter()
            .map(|atom| {
                common_orbital_hyperfine_tensor(&setup.basis, &scf, &orbitals, atom.position, 1e-8)
            })
            .collect::<Result<Vec<_>>>()?;
        report.set(
            "model",
            "common-orbital point-nucleus contact and spin-dipolar hyperfine tensor",
        );
        report.set(
            "isotropic_coupling_per_g_mhz",
            tensors.iter().map(|t| t.contact).collect::<Vec<_>>(),
        );
        report.set(
            "dipolar_coupling_per_g_mhz",
            tensors.iter().map(|t| t.dipolar).collect::<Vec<_>>(),
        );
        report.set(
            "total_coupling_per_g_mhz",
            tensors.iter().map(|t| t.total).collect::<Vec<_>>(),
        );
        report.set(
            "principal_couplings_per_g_mhz",
            tensors
                .iter()
                .map(|t| t.principal_values)
                .collect::<Vec<_>>(),
        );
        report.set(
            "principal_axes_per_g",
            tensors.iter().map(|t| t.principal_axes).collect::<Vec<_>>(),
        );
        report.set(
            "operator_residuals",
            tensors
                .iter()
                .map(|t| t.relative_residual())
                .collect::<Vec<_>>(),
        );
        if let Some(g) = nuclear_g {
            let scaled = tensors
                .iter()
                .zip(g)
                .map(|(t, &v)| t.scaled(v))
                .collect::<Result<Vec<_>>>()?;
            report.set("nuclear_g", g.to_vec());
            report.set(
                "isotropic_coupling_mhz",
                scaled.iter().map(|t| t.contact).collect::<Vec<_>>(),
            );
            report.set(
                "dipolar_coupling_mhz",
                scaled.iter().map(|t| t.dipolar).collect::<Vec<_>>(),
            );
            report.set(
                "total_coupling_mhz",
                scaled.iter().map(|t| t.total).collect::<Vec<_>>(),
            );
            report.set(
                "principal_couplings_mhz",
                scaled
                    .iter()
                    .map(|t| t.principal_values)
                    .collect::<Vec<_>>(),
            );
            report.set(
                "principal_axes",
                scaled.iter().map(|t| t.principal_axes).collect::<Vec<_>>(),
            );
        }
        report.set("warnings", vec!["Point-nucleus contact and spin-dipolar terms; no SOC-induced orbital hyperfine contribution."]);
        report.set(
            "units",
            units(&[
                ("energy", "hartree"),
                ("spin", "hbar"),
                ("isotropic_coupling_per_g_mhz", "MHz per signed nuclear g"),
                (
                    "dipolar_coupling_per_g_mhz",
                    "MHz per signed nuclear g, [nuclear axis][electronic spin axis]",
                ),
                (
                    "total_coupling_per_g_mhz",
                    "MHz per signed nuclear g, [nuclear axis][electronic spin axis]",
                ),
                ("principal_couplings_per_g_mhz", "MHz per signed nuclear g"),
                (
                    "principal_axes_per_g",
                    "dimensionless, eigenvector columns ordered by ascending eigenvalue",
                ),
                (
                    "total_coupling_mhz",
                    "MHz, multiplied once by the supplied signed nuclear g",
                ),
                ("isotropic_coupling_mhz", "MHz"),
                ("dipolar_coupling_mhz", "MHz"),
                (
                    "principal_couplings_mhz",
                    "MHz, ascending signed eigenvalues",
                ),
                ("principal_axes", "dimensionless eigenvector columns"),
                ("operator_residuals", "dimensionless"),
            ]),
        );
    }
    let mut labels = report.get("units").cloned().expect("EPR units");
    labels.set("e_scf", "hartree");
    labels.set("e_dispersion", "hartree");
    for key in ["scf_iterations", "n_alpha", "n_beta"] {
        labels.set(key, "dimensionless count");
    }
    if nuclear_g.is_some() {
        labels.set("nuclear_g", "dimensionless");
    }
    report.set("units", labels);
    Ok(report)
}

fn generalized_route(
    id: &str,
    controls: &Controls,
    setup: &MolecularSetup,
    reference: &str,
    origin: Vec3,
    charges: &[f64],
    nuclear_g: Option<&[f64]>,
) -> Result<Value> {
    if controls.opt_str("epr_projection")?.as_deref() != Some("single-kramers-pair") {
        return Err(unsupported("generalized-reference EPR without an explicit effective-spin model", "set epr_projection=single-kramers-pair for the approximate odd-electron S_eff=1/2 model; it does not spin-purify the determinant or establish an isolated doublet"));
    }
    if setup.method.multiplicity != 2 || setup.molecule.n_electrons()? % 2 != 1 {
        return Err(unsupported("single-Kramers-pair EPR outside an odd-electron doublet seed", "a validated higher-multiplet effective-spin manifold; multiplicity is only the seed label in the SOC spinor reference"));
    }
    let tolerance = controls.f64("epr_projection_tolerance", 1e-8)?;
    let xc_tolerance = controls.f64("spinor_xc_tolerance", 1e-10)?;
    if !tolerance.is_finite()
        || tolerance <= 0.0
        || !xc_tolerance.is_finite()
        || xc_tolerance <= 0.0
    {
        return Err(DftError::InvalidInput(
            "EPR projection and spinor XC tolerances must be finite and positive".into(),
        ));
    }
    let axis = controls
        .opt_vec3("spin_axis")?
        .map_or([0.0, 0.0, 1.0], |v| v.to_array());
    let n = 2 * setup.basis.nao_sph;
    let real = source_matrix(controls, "gks_operator_real", n)?;
    let imag = source_matrix(controls, "gks_operator_imaginary", n)?;
    let custom_operator = real.is_some() || imag.is_some();
    let operator = if custom_operator {
        BlochMatrix {
            re: real.unwrap_or_else(|| Matrix::zeros(n, n)),
            im: imag.unwrap_or_else(|| Matrix::zeros(n, n)),
        }
    } else {
        spinor::one_electron_soc_operator(&setup.basis, &setup.molecule, charges)?
    };
    if operator.hermiticity_error() > tolerance
        || crate::properties::epr::doublet::time_reversal_defect(&operator, 1)? > tolerance
    {
        return Err(DftError::InvalidInput(
            "generalized EPR requires a Hermitian time-even zero-field spinor operator".into(),
        ));
    }
    controls.finish(id)?;
    let seed = run_molecular_reference(setup)?;
    if !seed.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: seed.scf.iterations,
            error: f64::NAN,
        });
    }
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
    let config = SpinorScfConfig {
        max_iter: setup.options.max_iter,
        energy_tolerance: setup.options.e_tol,
        density_tolerance: setup.options.p_tol,
        orbital_tolerance: setup.options.conv_tol_grad,
        diis_size: setup.options.diis_size,
        overlap_cutoff: setup.options.orth_threshold,
        ..Default::default()
    };
    let recovery = spinor::rotation::run_epr_spinor_reference(
        &backend,
        &setup.functional,
        &config,
        &operator,
    )?;
    let scf = recovery.reference;
    if !scf.converged {
        crate::logger::log(&format!(
            "[epr-reference] SCF failed: orbital={:.3e} density={:.3e} occupation={:.3e} particles={:.3e} energy_change={:.3e}",
            scf.orbital_residual, scf.density_change, scf.occupation_consistency,
            scf.particle_number_error, scf.energy_change
        ));
        return Err(DftError::ScfNotConverged {
            iterations: recovery.total_iterations,
            error: scf.orbital_residual,
        });
    }
    let result = spinor::projected_spinor_epr(
        &setup.molecule,
        &setup.basis,
        &seed.scf.overlap,
        &scf,
        &operator,
        origin,
        charges,
        tolerance,
    )?;
    let mut report = Value::object().with("reference", reference)
        .with("reference_manifold", "full-complex-self-consistent-spinor")
        .with("model", "mean-field single-Kramers-pair EPR projection")
        .with("epr_projection", "single-kramers-pair").with("effective_spin", 0.5)
        .with("spinor_functional_model", "spin-covariant")
        .with("spinor_xc_tolerance", xc_tolerance)
        .with("zero_field_operator", if custom_operator { "user-supplied-time-even-spinor-operator" } else { "one-electron-Breit-Pauli-SOC" })
        .with("seed_multiplicity", setup.method.multiplicity)
        .with("functional", setup.functional.description.as_str()).with("basis", setup.method.basis.as_str())
        .with("energy", scf.fields.energy + seed.e_dispersion).with("e_scf", scf.fields.energy)
        .with("e_dispersion", seed.e_dispersion)
        .with("energy_scope", "spinor SCF plus nuclear repulsion and the selected empirical dispersion")
        .with("converged", scf.converged).with("scf_iterations", recovery.total_iterations)
        .with("spinor_recovery_cycles", recovery.cycles)
        .with("spinor_last_segment_iterations", scf.iterations)
        .with("spinor_rotation_energy_defect", recovery.maximum_rotation_energy_defect)
        .with("orbital_residual", scf.orbital_residual).with("density_change", scf.density_change)
        .with("occupation_consistency", scf.occupation_consistency)
        .with("particle_number_error", scf.particle_number_error)
        .with("metric_defect", scf.metric_defect).with("energy_change", scf.energy_change)
        .with("operator_converged", scf.fields.operator_converged)
        .with("xc_grid_number_defect", scf.fields.xc_grid_number_defect)
        .with("spin_expectation", result.spin_expectation).with("spin_squared", result.spin_squared)
        .with("pair_overlap", result.pair_overlap).with("occupied_metric_defect", result.occupied_metric_defect)
        .with("hamiltonian_time_reversal_defect", result.hamiltonian_time_reversal_defect)
        .with("gauge", "common-origin").with("gauge_origin", origin).with("soc_charges", charges.to_vec())
        .with("tensor_columns", "Pauli axes of the ordered determinant/time-reversed-determinant pair")
        .with("provenance", "analytic-operator-projection")
        .with("warnings", vec!["Single-pair approximation: spin contamination can substantially bias transverse couplings; no spin purification or isolated-doublet guarantee.",
            "Common-origin one-electron Breit-Pauli model; no two-electron SOC or London response. SOC mixes spin: seed multiplicity is not a conserved quantum number."]);
    if let Some(d) = &scf.fields.xc_diagnostics {
        report.set("xc_angular_order", d.angular_order);
        report.set("xc_angular_energy_defect", d.angular_energy_defect);
        report.set("xc_angular_operator_defect", d.angular_operator_defect);
        report.set("xc_angular_history", d.angular_refinement_history.clone());
    }
    if id == "g-tensor" {
        report.set("g_tensor", result.g_tensor);
        report.set("g_gram", result.g_gram);
        report.set(
            "units",
            units(&[
                ("energy", "hartree"),
                ("g_tensor", "dimensionless, [field axis][pair Pauli axis]"),
                (
                    "g_gram",
                    "dimensionless, g times transpose(g); invariant to pair basis rotations",
                ),
                ("gauge_origin", "bohr"),
                ("spin_expectation", "hbar"),
                ("spin_squared", "hbar^2"),
                ("orbital_residual", "hartree"),
                ("density_change", "density metric norm"),
                ("hamiltonian_time_reversal_defect", "hartree"),
                ("pair_overlap", "dimensionless"),
                ("occupied_metric_defect", "dimensionless"),
            ]),
        );
    } else {
        report.set("contact_coupling_per_g_mhz", result.contact_per_g_mhz);
        report.set("dipolar_coupling_per_g_mhz", result.dipolar_per_g_mhz);
        report.set("orbital_coupling_per_g_mhz", result.orbital_per_g_mhz);
        if let Some(g) = nuclear_g {
            let physical = result
                .total_per_g_mhz
                .iter()
                .zip(g)
                .map(|(t, &v)| t.map(|row| row.map(|x| x * v)))
                .collect::<Vec<_>>();
            report.set("total_coupling_mhz", physical);
            report.set("nuclear_g", g.to_vec());
        }
        report.set("total_coupling_per_g_mhz", result.total_per_g_mhz);
        report.set(
            "units",
            units(&[
                ("energy", "hartree"),
                (
                    "contact_coupling_per_g_mhz",
                    "MHz per signed nuclear g, [nuclear axis][pair Pauli axis]",
                ),
                (
                    "dipolar_coupling_per_g_mhz",
                    "MHz per signed nuclear g, [nuclear axis][pair Pauli axis]",
                ),
                (
                    "orbital_coupling_per_g_mhz",
                    "MHz per signed nuclear g, [nuclear axis][pair Pauli axis]",
                ),
                (
                    "total_coupling_per_g_mhz",
                    "MHz per signed nuclear g, [nuclear axis][pair Pauli axis]",
                ),
                (
                    "total_coupling_mhz",
                    "MHz, multiplied once by supplied signed nuclear g",
                ),
                ("gauge_origin", "bohr"),
                ("spin_expectation", "hbar"),
                ("spin_squared", "hbar^2"),
                ("orbital_residual", "hartree"),
                ("occupied_metric_defect", "dimensionless"),
            ]),
        );
    }
    let mut labels = report.get("units").cloned().expect("EPR units");
    labels.set("e_scf", "hartree");
    labels.set("e_dispersion", "hartree");
    for key in [
        "scf_iterations",
        "seed_multiplicity",
        "xc_angular_order",
        "spinor_recovery_cycles",
        "spinor_last_segment_iterations",
    ] {
        labels.set(key, "dimensionless count");
    }
    labels.set(
        "effective_spin",
        "dimensionless, effective-spin quantum number",
    );
    labels.set("spinor_xc_tolerance", "hartree");
    labels.set("spinor_rotation_energy_defect", "hartree");
    labels.set("soc_charges", "e");
    labels.set("density_change", "dimensionless");
    labels.set(
        "occupation_consistency",
        "dimensionless density coefficient",
    );
    labels.set("particle_number_error", "electrons");
    labels.set("metric_defect", "dimensionless");
    labels.set("energy_change", "hartree");
    labels.set("pair_overlap", "dimensionless");
    labels.set("hamiltonian_time_reversal_defect", "hartree");
    labels.set("xc_grid_number_defect", "electrons");
    labels.set("xc_angular_energy_defect", "hartree");
    labels.set("xc_angular_operator_defect", "hartree");
    labels.set("xc_angular_history", "[dimensionless, hartree, hartree]");
    if nuclear_g.is_some() {
        labels.set("nuclear_g", "dimensionless");
    }
    report.set("units", labels);
    Ok(report)
}
