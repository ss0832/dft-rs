// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Post-SCF double-hybrid energy with an explicit parent and spin PT2 model.

use super::method::{
    prepare_molecular, run_collinear_property_reference, CollinearPropertyReference,
    MolecularMethod,
};
use super::molecular::{reference_report, units};
use super::{Controls, System, Value};
use crate::integrals::{dipole_matrices, eri_incore, CoulombKernel};
use crate::linalg::Matrix;
use crate::properties::pt2::{from_ao, OrbitalSpace};
use crate::scf::fock::JkEngine;
use crate::xc::double_hybrid::{DoubleHybridFunctional, Pt2Coefficients};
use crate::xc::{HybridParams, ResolvedFunctional};
use crate::{DftError, Result};

#[cfg(test)]
pub(super) mod moment_tests;
use std::sync::Arc;

fn trace(a: &Matrix, b: &Matrix) -> f64 {
    (0..a.rows)
        .flat_map(|i| (0..a.cols).map(move |j| a[(i, j)] * b[(j, i)]))
        .sum()
}

pub(super) fn requested(controls: &Controls) -> Result<bool> {
    Ok(controls.contains("double_hybrid_model")
        || controls.contains("double_hybrid_target_xc")
        || controls.contains("pt2_same_spin")
        || controls.contains("pt2_opposite_spin")
        || controls
            .opt_str("xc")?
            .is_some_and(|name| DoubleHybridFunctional::named(&name).is_some()))
}

pub(super) fn energy(system: &System, controls: &Controls) -> Result<Value> {
    calculate(system, controls, "energy")
}

pub(super) fn electric_moments(system: &System, controls: &Controls, id: &str) -> Result<Value> {
    calculate(system, controls, id)
}

pub(super) fn polarizability(system: &System, controls: &Controls) -> Result<Value> {
    calculate(system, controls, "polarizability")
}

/// Parent reference, PT2 problem and amplitudes of a double-hybrid request.
pub(super) struct Prepared {
    pub(super) reference_kind: CollinearPropertyReference,
    pub(super) config: Option<crate::response::ResponseSolverConfig>,
    pub(super) model: String,
    pub(super) model_dispersion: Option<crate::dispersion::D3BjParams>,
    pub(super) origin: Option<(crate::math::Vec3, String)>,
    max_moment_rank: Option<usize>,
    pub(super) quadrupole_request: Option<super::quadrupole::Request>,
    pub(super) setup: super::method::MolecularSetup,
    pub(super) reference: super::method::MolecularReference,
    pub(super) target: Option<ResolvedFunctional>,
    pub(super) target_vv10: Option<(f64, crate::nlc::Vv10Params)>,
    pub(super) reference_vv10: Option<crate::nlc::Vv10Params>,
    pub(super) coefficients: Pt2Coefficients,
    pub(super) pt2_memory_mb: usize,
    pub(super) extra_dispersion: f64,
    pub(super) eri: crate::integrals::eri::EriTensor,
    pub(super) tensor: Vec<f64>,
    pub(super) problem: crate::properties::pt2::Pt2Problem<f64>,
    pub(super) pt2: crate::properties::pt2::Pt2Solution<f64>,
}

impl Prepared {
    /// Conservative common bound for the moving integral/amplitude tensors.
    pub(super) fn check_path_memory(&self, order: usize) -> Result<()> {
        let bytes = self
            .setup
            .basis
            .nao_sph
            .checked_pow(4)
            .and_then(|v| v.checked_mul(order.checked_add(1)?))
            .and_then(|v| v.checked_mul(16 * 8))
            .ok_or_else(|| DftError::InvalidInput("PT2 response workspace size overflow".into()))?;
        let required_mb = bytes.div_ceil(1024 * 1024);
        if required_mb > self.pt2_memory_mb {
            return Err(DftError::ResourceLimit {
                operation: "PT2 response coefficient tensors",
                required_mb,
                limit_mb: self.pt2_memory_mb,
            });
        }
        Ok(())
    }
    /// Explicit target energy composition, independent of parent auxiliaries.
    pub(super) fn field_target(
        &self,
    ) -> Result<Option<crate::response::field_path::FieldEnergyModel<'_>>> {
        if self.target.is_none() && self.target_vv10.is_none() {
            return Ok(None);
        }
        let mut model = crate::response::field_path::FieldEnergyModel::new(
            self.target.as_ref().unwrap_or(&self.setup.functional),
        );
        if let Some((coefficient, parameters)) = self.target_vv10 {
            model = model.with_vv10(coefficient, parameters)?;
        }
        Ok(Some(model))
    }
}

struct DoubleHybridSetup {
    setup: super::method::MolecularSetup,
    model: String,
    model_dispersion: Option<crate::dispersion::D3BjParams>,
    target: Option<ResolvedFunctional>,
    target_vv10: Option<(f64, crate::nlc::Vv10Params)>,
    coefficients: Pt2Coefficients,
    memory: usize,
}

fn prepare_model(system: &System, controls: &Controls) -> Result<DoubleHybridSetup> {
    let named = controls.opt_str("xc")?;
    let default_model = named
        .as_deref()
        .filter(|name| DoubleHybridFunctional::named(name).is_some())
        .unwrap_or("custom");
    let model = controls.str("double_hybrid_model", default_model)?;
    let recipe = DoubleHybridFunctional::named(&model);
    let custom_target = controls.opt_str("double_hybrid_target_xc")?;
    if recipe.is_some() && custom_target.is_some() {
        return Err(DftError::InvalidInput(
            "double_hybrid_target_xc requires double_hybrid_model=custom; named recipes fix their target functional".into(),
        ));
    }
    let ss = controls.opt_f64("pt2_same_spin")?;
    let os = controls.opt_f64("pt2_opposite_spin")?;
    let memory = controls.usize("pt2_memory_mb", 2048)?;
    let mut method = MolecularMethod::from_controls(controls)?;
    let mut model_dispersion = None;
    let mut target = None;
    let mut target_vv10 = None;
    if let Some(recipe) = &recipe {
        if method.x.is_some()
            || method.c.is_some()
            || method.xc.as_ref().is_some_and(|name| {
                DoubleHybridFunctional::named(name)
                    .is_none_or(|other| other.parent.description != recipe.parent.description)
            })
        {
            return Err(DftError::InvalidInput(
                "use double_hybrid_model=custom for an explicitly supplied parent functional"
                    .into(),
            ));
        }
        if !matches!(method.dispersion.as_str(), "auto" | "none") {
            return Err(DftError::InvalidInput("named double hybrids use their own dispersion recipe; an explicit different damping model needs a separate published parametrization".into()));
        }
        if method.dispersion == "auto" {
            model_dispersion = recipe.dispersion;
        }
        if method.omega.is_some() {
            return Err(DftError::InvalidInput(
                "named double-hybrid recipes fix omega; use a custom parent to vary it".into(),
            ));
        }
        // Build the grid once, then supply the typed composition. No damping
        // or nonlocal parameters are borrowed from the grid template.
        method.dispersion = "none".into();
        method.xc = Some("tpss".into());
    } else if model != "custom" || ss.is_none() || os.is_none() {
        return Err(DftError::InvalidInput(
            "select a known double-hybrid model or custom with pt2_same_spin and pt2_opposite_spin"
                .into(),
        ));
    }
    if method.electronic_temperature != 0.0
        || method.electron_offset != [0.0; 2]
        || method.fixed_occupations.is_some()
    {
        return Err(DftError::InvalidInput(
            "semicanonical PT2 requires integer zero-temperature occupations".into(),
        ));
    }
    let mut setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    let defaults = if let Some(functional) = recipe {
        setup.functional = functional.parent;
        target = functional.target;
        target_vv10 = functional.target_vv10;
        setup.options.vv10 = functional.reference_vv10;
        functional.correlation
    } else {
        Pt2Coefficients {
            same_spin: ss.unwrap(),
            opposite_spin: os.unwrap(),
        }
    };
    let coefficients = Pt2Coefficients {
        same_spin: ss.unwrap_or(defaults.same_spin),
        opposite_spin: os.unwrap_or(defaults.opposite_spin),
    };
    coefficients.validate()?;
    if let Some(name) = custom_target {
        let nonlocal_parameters = crate::xc::built_in_nonlocal(&name).0;
        let functional = crate::xc::resolve(&crate::xc::FunctionalSpec::Combined(name))?;
        if functional.required_nonlocal_kind().is_some_and(|kind| {
            kind != crate::xc::RequiredNonlocalTerm::Vv10 || nonlocal_parameters.is_none()
        }) {
            return Err(DftError::Unsupported {
                feature: "custom double-hybrid target with an additional nonlocal term".into(),
                missing: "an explicit target nonlocal composition and moving two-endpoint provider"
                    .into(),
            });
        }
        target_vv10 = nonlocal_parameters.map(|parameters| (1.0, parameters));
        if !functional.parts.is_empty() && setup.grid.is_none() {
            setup.grid = Some(crate::grid::MolecularGrid::build(
                &setup.molecule,
                &setup.grid_options,
            )?);
        }
        target = Some(functional);
    }
    Ok(DoubleHybridSetup {
        setup,
        model,
        model_dispersion,
        target,
        target_vv10,
        coefficients,
        memory,
    })
}

pub(super) struct SpinorModel {
    pub name: String,
    pub coefficient: f64,
    pub memory_mb: usize,
    pub dispersion: f64,
    pub dispersion_parameters: Option<crate::dispersion::D3BjParams>,
    pub target: Option<ResolvedFunctional>,
    pub target_vv10: Option<(f64, crate::nlc::Vv10Params)>,
}

pub(super) fn prepare_spinor(
    system: &System,
    controls: &Controls,
    feature: &str,
) -> Result<(super::method::MolecularSetup, SpinorModel)> {
    let model = prepare_model(system, controls)?;
    if model.coefficients.same_spin != model.coefficients.opposite_spin {
        return Err(DftError::Unsupported {
            feature: "noncollinear double hybrid with unequal spin coefficients".into(),
            missing: "a spin-covariant pair model defining distinct spin weights".into(),
        });
    }
    if model.target_vv10.is_some() && !matches!(feature, "energy" | "gradient") {
        return Err(DftError::Unsupported {
            feature: "noncollinear double hybrid with a nonlocal target".into(),
            missing: "the matching nonlocal target energy Taylor provider".into(),
        });
    }
    let dispersion = model_dispersion_energy(system, model.model_dispersion);
    Ok((
        model.setup,
        SpinorModel {
            name: model.model,
            coefficient: model.coefficients.same_spin,
            memory_mb: model.memory,
            dispersion,
            dispersion_parameters: model.model_dispersion,
            target: model.target,
            target_vv10: model.target_vv10,
        },
    ))
}

pub(super) fn append_spinor_energy(
    report: &mut Value,
    model: &SpinorModel,
    unscaled_energy: f64,
    target_energy_correction: f64,
) {
    report.set("double_hybrid_model", model.name.as_str());
    report.set("generalized_pt2_coefficient", model.coefficient);
    report.set("generalized_pt2_correlation", unscaled_energy);
    let weighted_energy = model.coefficient * unscaled_energy;
    report.set("pt2_correlation", weighted_energy);
    for key in ["energy", "internal_energy", "mermin_free_energy"] {
        let reference = report
            .get(key)
            .and_then(Value::as_f64)
            .expect("spinor reference energy");
        report.set(
            key,
            reference + target_energy_correction + weighted_energy + model.dispersion,
        );
    }
    let mut terms = report
        .get("energy_components")
        .cloned()
        .expect("spinor energy components");
    terms.set("pt2", weighted_energy);
    terms.set("model_dispersion", model.dispersion);
    if let Some(target) = &model.target {
        terms.set("target_correction", target_energy_correction);
        report.set("target_functional", target.description.as_str());
    }
    report.set("energy_components", terms);
    let mut labels = report.get("units").cloned().expect("spinor energy units");
    labels.set("generalized_pt2_coefficient", "dimensionless");
    labels.set("generalized_pt2_correlation", "hartree");
    labels.set("pt2_correlation", "hartree");
    if let Some((scale, parameters)) = model.target_vv10 {
        for (key, value) in [
            ("target_vv10_scale", scale),
            ("target_vv10_b", parameters.b),
            ("target_vv10_c", parameters.c),
        ] {
            report.set(key, value);
            labels.set(key, "dimensionless");
        }
    }
    report.set("units", labels);
}

fn model_dispersion_energy(system: &System, model: Option<crate::dispersion::D3BjParams>) -> f64 {
    model.map_or(0.0, |params| {
        crate::dispersion::d3_energy_g(&system.numbers, &system.positions, None, Some(&params))
    })
}

/// Reads the double-hybrid controls, runs the parent reference and solves
/// the PT2 amplitudes; `id` finishes the controls.
pub(super) fn prepare(system: &System, controls: &Controls, id: &str) -> Result<Prepared> {
    prepare_with_core(system, controls, id, |_| None)
}

pub(in crate::driver) fn prepare_with_core(
    system: &System,
    controls: &Controls,
    id: &str,
    additional_core: impl FnOnce(&super::method::MolecularSetup) -> Option<crate::linalg::Matrix>,
) -> Result<Prepared> {
    let reference_kind = CollinearPropertyReference::from_controls(controls)?;
    if reference_kind == CollinearPropertyReference::Roks && id != "energy" {
        return Err(DftError::Unsupported {
            feature: format!("common-orbital double-hybrid {id}"),
            missing: "the constrained common-orbital PT2 response Lagrangian; the total energy provider is available".into(),
        });
    }
    let with_moments = matches!(id, "dipole" | "molecular-multipole");
    let max_moment_rank = (id == "molecular-multipole")
        .then(|| super::molecular::multipole_rank(controls))
        .transpose()?;
    let with_response = id != "energy";
    let with_quadrupole_response = super::quadrupole::is_electric_multipole(id);
    let config = if with_response && !with_quadrupole_response {
        Some(super::vibrational::response_config(controls)?)
    } else {
        None
    };
    let DoubleHybridSetup {
        mut setup,
        model,
        model_dispersion,
        target,
        target_vv10,
        coefficients,
        memory,
    } = prepare_model(system, controls)?;
    let reference_vv10 = setup.options.vv10;
    let origin = if with_moments {
        Some(super::molecular::moment_origin(controls, &setup)?)
    } else {
        None
    };
    let quadrupole_request = with_quadrupole_response
        .then(|| super::quadrupole::Request::new(id, &setup, controls))
        .transpose()?;
    let dynamic_config = if id == "dynamic-polarizability" {
        config.as_ref()
    } else if let Some(request) = &quadrupole_request {
        request.dynamic_config()?
    } else {
        None
    };
    if let Some(config) = dynamic_config.filter(|_| setup.method.conv_tol_grad.is_none()) {
        // The time-dependent parent needs a fresh orbital residual below
        // the response gate. Tighten only the implicit SCF threshold.
        setup.options.conv_tol_grad = setup.options.conv_tol_grad.min(0.1 * config.acceptance());
    }
    controls.finish(id)?;
    let n = setup.basis.nao_sph;
    let bytes = n
        .checked_pow(4)
        .and_then(|v| v.checked_mul(if with_response { 8 * 16 } else { 8 * 8 }))
        .ok_or_else(|| {
            DftError::InvalidInput("PT2 tensor memory estimate overflows addressable memory".into())
        })?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > memory {
        return Err(DftError::ResourceLimit {
            operation: "PT2 AO and transformed tensors",
            required_mb,
            limit_mb: memory,
        });
    }
    if let Some(core) = additional_core(&setup) {
        setup.options.extra_hcore = Some(core);
    }
    let reference = run_collinear_property_reference(&setup, reference_kind)?;
    let scf = &reference.scf;
    let extra_dispersion = model_dispersion_energy(system, model_dispersion);
    let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, memory)?;
    let tensor: Vec<_> = (0..n)
        .flat_map(|i| {
            (0..n).flat_map(move |j| (0..n).flat_map(move |k| (0..n).map(move |l| [i, j, k, l])))
        })
        .map(|[i, j, k, l]| eri.get(i, j, k, l))
        .collect();
    let orbitals = std::array::from_fn(|s| OrbitalSpace {
        energies: scf.mo_energy[s].clone(),
        coefficients: scf.mo_coeff[s].as_slice().to_vec(),
        occupied: if s == 0 { scf.n_alpha } else { scf.n_beta },
    });
    let problem = from_ao(n, &tensor, &orbitals)?;
    let pt2 = problem.solve(coefficients)?;
    Ok(Prepared {
        reference_kind,
        config,
        model,
        model_dispersion,
        origin,
        max_moment_rank,
        quadrupole_request,
        setup,
        reference,
        target,
        target_vv10,
        reference_vv10,
        coefficients,
        pt2_memory_mb: memory,
        extra_dispersion,
        eri,
        tensor,
        problem,
        pt2,
    })
}

fn calculate(system: &System, controls: &Controls, id: &str) -> Result<Value> {
    calculate_prepared(prepare(system, controls, id)?, id)
}

struct CorrelatedMoment {
    powers: (i32, i32, i32),
    parent: f64,
    response: super::correlated_electric::MomentResponse,
}

impl CorrelatedMoment {
    fn total(&self) -> f64 {
        self.parent + self.response.pt2 + self.response.target
    }
}

fn correlated_moment_ranks(
    setup: &super::method::MolecularSetup,
    reference: &super::method::MolecularReference,
    target: Option<&ResolvedFunctional>,
    target_vv10: Option<(f64, crate::nlc::Vv10Params)>,
    problem: &crate::properties::pt2::Pt2Problem<f64>,
    tensor: &[f64],
    coefficients: Pt2Coefficients,
    origin: crate::math::Vec3,
    max_rank: Option<usize>,
    config: &crate::response::ResponseSolverConfig,
) -> Result<Vec<Vec<CorrelatedMoment>>> {
    let density = reference.scf.p_total();
    let mut operators = Vec::new();
    let mut ranks: Vec<Vec<CorrelatedMoment>> = Vec::new();
    for rank in max_rank.map_or(1..=1, |max| 0..=max) {
        if rank == 0 {
            // A constant potential shifts every orbital by the same amount.
            // It leaves the density and PT2 denominators invariant.
            ranks.push(vec![CorrelatedMoment {
                powers: (0, 0, 0),
                parent: f64::from(setup.molecule.charge),
                response: Default::default(),
            }]);
            continue;
        }
        let terms = super::molecular::moment_operators(setup, origin, rank as u8);
        let mut values = Vec::with_capacity(terms.len());
        for term in terms {
            values.push(CorrelatedMoment {
                powers: term.powers,
                parent: term.ionic - density.frobenius_dot(&term.matrix),
                response: Default::default(),
            });
            if rank != 1 {
                operators.push(term.matrix);
            }
        }
        if rank == 1 {
            // The origin shift adds only a multiple of the overlap.
            operators.extend(crate::integrals::dipole_matrices(
                &setup.basis,
                crate::math::Vec3::zero(),
            ));
        }
        ranks.push(values);
    }
    let responses = super::correlated_electric::moment_response(
        setup,
        reference,
        target,
        target_vv10,
        problem,
        tensor,
        coefficients,
        &operators,
        config,
    )?;
    for (moment, response) in ranks
        .iter_mut()
        .flatten()
        .filter(|moment| moment.powers != (0, 0, 0))
        .zip(responses)
    {
        moment.response = response;
    }
    Ok(ranks)
}

fn append_correlated_dipole(report: &mut Value, labels: &mut Value, values: &[CorrelatedMoment]) {
    let dipole: Vec<_> = values.iter().map(CorrelatedMoment::total).collect();
    let norm = dipole.iter().map(|v| v * v).sum::<f64>().sqrt();
    let to_debye = crate::constants::AU_DIPOLE_TO_DEBYE;
    report.set(
        "dipole_debye",
        dipole.iter().map(|v| v * to_debye).collect::<Vec<_>>(),
    );
    report.set("dipole", dipole);
    report.set("dipole_norm", norm);
    report.set("dipole_norm_debye", norm * to_debye);
    report.set(
        "parent_dipole",
        values.iter().map(|v| v.parent).collect::<Vec<_>>(),
    );
    for (key, read, unit) in [
        (
            "pt2_dipole_correction",
            (|r| r.pt2) as fn(&super::correlated_electric::MomentResponse) -> f64,
            "e*bohr",
        ),
        ("target_dipole_correction", |r| r.target, "e*bohr"),
        (
            "response_residuals",
            |r| r.orbital_residual,
            "dimensionless",
        ),
        (
            "amplitude_response_residuals",
            |r| r.amplitude_residual,
            "e*bohr",
        ),
        (
            "response_number_residuals",
            |r| r.number_residual,
            "atomic units, electrons per atomic electric field",
        ),
    ] {
        report.set(
            key,
            values.iter().map(|v| read(&v.response)).collect::<Vec<_>>(),
        );
        labels.set(key, unit);
    }
    for key in ["dipole", "dipole_norm", "parent_dipole"] {
        labels.set(key, "e*bohr");
    }
    for key in ["dipole_debye", "dipole_norm_debye"] {
        labels.set(key, "debye");
    }
}

fn append_correlated_multipoles(
    report: &mut Value,
    labels: &mut Value,
    ranks: Vec<Vec<CorrelatedMoment>>,
) {
    use super::correlated_electric::MomentResponse;
    let mut rows = Vec::with_capacity(ranks.len());
    for (rank, values) in ranks.into_iter().enumerate() {
        let raw: Vec<_> = values.iter().map(|v| (v.powers, v.total())).collect();
        let mut row = super::molecular::multipole_values(rank, &raw).with(
            "parent_cartesian",
            values.iter().map(|v| v.parent).collect::<Vec<_>>(),
        );
        for (key, read, unit) in [
            (
                "pt2_cartesian_correction",
                (|r| r.pt2) as fn(&MomentResponse) -> f64,
                "e*bohr^rank",
            ),
            ("target_cartesian_correction", |r| r.target, "e*bohr^rank"),
            ("pt2_same_spin_alpha", |r| r.same_spin_alpha, "e*bohr^rank"),
            ("pt2_same_spin_beta", |r| r.same_spin_beta, "e*bohr^rank"),
            ("pt2_opposite_spin", |r| r.opposite_spin, "e*bohr^rank"),
            (
                "response_residuals",
                |r| r.orbital_residual,
                "dimensionless",
            ),
            (
                "amplitude_response_residuals",
                |r| r.amplitude_residual,
                "e*bohr^rank",
            ),
            (
                "response_number_residuals",
                |r| r.number_residual,
                "electrons per atomic moment source",
            ),
        ] {
            row.set(
                key,
                values.iter().map(|v| read(&v.response)).collect::<Vec<_>>(),
            );
            labels.set(&format!("ranks.{key}"), unit);
        }
        rows.push(row);
    }
    for key in ["cartesian", "traceless", "spherical", "parent_cartesian"] {
        labels.set(&format!("ranks.{key}"), "e*bohr^rank");
    }
    super::molecular::set_multipole_ranks(report, rows);
}

pub(super) fn calculate_prepared(prepared: Prepared, id: &str) -> Result<Value> {
    let Prepared {
        reference_kind,
        config,
        model,
        model_dispersion,
        origin,
        max_moment_rank,
        quadrupole_request: _,
        setup,
        reference,
        target,
        target_vv10,
        reference_vv10,
        coefficients,
        pt2_memory_mb: _,
        extra_dispersion,
        eri: _,
        tensor,
        problem,
        pt2,
    } = prepared;
    let scf = &reference.scf;
    let lagrangian = problem.lagrangian(&pt2.amplitudes, coefficients)?;
    let diagnostics = crate::response::diagnostics::reference_diagnostics(scf)?;
    let target_functional = target.as_ref().unwrap_or(&setup.functional);
    let mut semilocal = Vec::new();
    let mut semilocal_energy = 0.0;
    if let Some(grid) = &setup.grid {
        for (weight, part) in &target_functional.parts {
            let single = ResolvedFunctional::custom(
                "individual semilocal component",
                vec![(*weight, Arc::clone(part))],
                HybridParams::PURE,
                part.family(),
            );
            let value = crate::ks::xc_contribution(
                &single,
                &setup.basis,
                grid,
                [&scf.density[0], &scf.density[1]],
            )?
            .exc;
            semilocal_energy += value;
            semilocal.push(
                Value::object()
                    .with("component", part.name())
                    .with("coefficient", *weight)
                    .with("energy", value),
            );
        }
    }
    let p = scf.p_total();
    let engine = JkEngine::from_options(&setup.molecule, &setup.basis, &setup.options)?;
    let (j, k) = engine.build(&setup.basis, &p, [&scf.density[0], &scf.density[1]], true)?;
    let full_exchange = -0.5 * (trace(&scf.density[0], &k[0]) + trace(&scf.density[1], &k[1]));
    let exchange_for = |hybrid: HybridParams| -> Result<f64> {
        let mut exchange = hybrid.cx * full_exchange;
        if hybrid.clr == 0.0 {
            return Ok(exchange);
        }
        let engine = JkEngine::new(
            &setup.basis,
            setup.options.eri_mode,
            CoulombKernel::Erf {
                omega: hybrid.omega,
            },
        )?;
        let k = engine.build_k_only(&setup.basis, [&scf.density[0], &scf.density[1]])?;
        exchange -=
            0.5 * hybrid.clr * (trace(&scf.density[0], &k[0]) + trace(&scf.density[1], &k[1]));
        Ok(exchange)
    };
    let orbital_exchange = exchange_for(setup.functional.hybrid)?;
    let exchange = exchange_for(target_functional.hybrid)?;
    let target_nlc = if target.is_none() {
        scf.e_nlc
    } else {
        match target_vv10 {
            Some((scale, params)) if reference_vv10 == Some(params) => scale * scf.e_nlc,
            Some((scale, params)) => {
                let grid = setup.options.nonlocal_grid(&setup.molecule)?;
                scale * crate::nlc::vv10_energy(&setup.basis, &grid, &p, params)?
            }
            None => 0.0,
        }
    };
    let target_semilocal_xc = match &setup.grid {
        Some(grid) => {
            crate::ks::xc_contribution(
                target_functional,
                &setup.basis,
                grid,
                [&scf.density[0], &scf.density[1]],
            )?
            .exc
        }
        None => 0.0,
    };
    // A custom parent may supply additional energy through a model provider.
    // Replacing its complete XC energy by the semilocal composition would
    // discard that contribution even in the zero-PT2 limit.
    let target_xc = if target.is_none() {
        scf.e_xc
    } else {
        target_semilocal_xc
    };
    let nonsemilocal_xc = target_xc - target_semilocal_xc;
    let one = trace(&p, &scf.h_core);
    let hartree = 0.5 * trace(&p, &j);
    let decomposition =
        scf.e_nuc + one + hartree + orbital_exchange + scf.e_xc + scf.e_nlc + scf.e_pcm;
    let target_electronic = one + hartree + exchange + target_xc + target_nlc + scf.e_pcm;
    let target_energy = target_electronic
        + scf.e_nuc
        + reference.e_dispersion
        + extra_dispersion
        + reference.e_smd_cds;
    let mut report = reference_report(&setup,&reference,
        "post-SCF double hybrid on semicanonical generalized Kohn-Sham orbitals; full-Coulomb unregularized spin-component PT2, all explicitly treated electrons correlated; no frozen core; no correlated density or relaxed molecular derivatives in this energy entry");
    report.set("orbital_reference_energy", reference.total_energy());
    report.set("orbital_functional", setup.functional.description.as_str());
    report.set("functional", target_functional.description.as_str());
    report.set("distinct_target_energy", target.is_some());
    report.set("target_nonlocal_correlation", target_nlc);
    report.set("orbital_nonlocal_correlation", scf.e_nlc);
    report.set("target_vv10_scale", target_vv10.map(|p| p.0));
    report.set("target_vv10_b", target_vv10.map(|p| p.1.b));
    report.set("target_vv10_c", target_vv10.map(|p| p.1.c));
    report.set("reference_energy", target_energy);
    report.set("energy", target_energy + pt2.correlation);
    report.set("model_dispersion_energy", extra_dispersion);
    report.set(
        "model_dispersion",
        model_dispersion.map(|p| {
            Value::object()
                .with("model", "D3(BJ), pairwise")
                .with("s6", p.s6)
                .with("s8", p.s8)
                .with("a1", p.a1)
                .with("a2", p.a2)
        }),
    );
    let mut components = report
        .get("energy_components")
        .cloned()
        .expect("reference energy components");
    components.set("dispersion", reference.e_dispersion + extra_dispersion);
    components.set("electronic", target_electronic);
    components.set("exchange_correlation", target_xc);
    components.set("nonlocal_correlation", target_nlc);
    components.set("perturbative_correlation", pt2.correlation);
    report.set("energy_components", components);
    report.set("double_hybrid_model", model);
    report.set("pt2_same_spin_coefficient", coefficients.same_spin);
    report.set("pt2_opposite_spin_coefficient", coefficients.opposite_spin);
    report.set("pt2_same_spin_alpha", pt2.same_spin_alpha);
    report.set("pt2_same_spin_beta", pt2.same_spin_beta);
    report.set("pt2_opposite_spin", pt2.opposite_spin);
    report.set("pt2_correlation", pt2.correlation);
    report.set("pt2_amplitude_residual", pt2.amplitude_residual);
    report.set(
        "pt2_smallest_denominator",
        pt2.smallest_excitation_denominator,
    );
    report.set(
        "pt2_lagrangian_residual",
        (lagrangian - pt2.correlation).abs(),
    );
    report.set("reference_orbital_residual", diagnostics.orbital_residual);
    report.set("reference_eigen_residual", diagnostics.eigen_residual);
    report.set("one_electron_energy", one);
    report.set("hartree_energy", hartree);
    report.set("exact_exchange_energy", exchange);
    report.set("semilocal_components", semilocal);
    report.set("nonsemilocal_exchange_correlation_energy", nonsemilocal_xc);
    report.set(
        "semilocal_component_sum_residual",
        (semilocal_energy - target_semilocal_xc).abs(),
    );
    report.set(
        "reference_energy_sum_residual",
        (decomposition - scf.energy).abs(),
    );
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("reference_energy", "hartree"),
            ("orbital_reference_energy", "hartree"),
            ("target_nonlocal_correlation", "hartree"),
            ("orbital_nonlocal_correlation", "hartree"),
            ("target_vv10_scale", "dimensionless"),
            ("target_vv10_b", "dimensionless"),
            ("target_vv10_c", "dimensionless"),
            ("model_dispersion_energy", "hartree"),
            ("model_dispersion.s6", "dimensionless"),
            ("model_dispersion.s8", "dimensionless"),
            ("model_dispersion.a1", "dimensionless"),
            ("model_dispersion.a2", "bohr"),
            ("energy_components", "hartree"),
            ("pt2_same_spin_coefficient", "dimensionless"),
            ("pt2_opposite_spin_coefficient", "dimensionless"),
            ("pt2_same_spin_alpha", "hartree"),
            ("pt2_same_spin_beta", "hartree"),
            ("pt2_opposite_spin", "hartree"),
            ("pt2_correlation", "hartree"),
            ("pt2_amplitude_residual", "hartree"),
            ("pt2_smallest_denominator", "hartree"),
            ("pt2_lagrangian_residual", "hartree"),
            ("reference_orbital_residual", "hartree"),
            ("reference_eigen_residual", "dimensionless"),
            ("one_electron_energy", "hartree"),
            ("hartree_energy", "hartree"),
            ("exact_exchange_energy", "hartree"),
            ("semilocal_components.coefficient", "dimensionless"),
            ("semilocal_components.energy", "hartree"),
            ("nonsemilocal_exchange_correlation_energy", "hartree"),
            ("semilocal_component_sum_residual", "hartree"),
            ("reference_energy_sum_residual", "hartree"),
        ]),
    );
    if reference_kind == CollinearPropertyReference::Roks {
        // Each spin's occupied and virtual Fock blocks are diagonalized by
        // the existing common-orbital SCF finalization, without changing its
        // density. The DH convention retains only double excitations; it
        // does not add the singles correction of open-shell MP2.
        let stationary = scf
            .common_orbital_residual
            .unwrap_or(diagnostics.orbital_residual);
        report.set("reference_type", "roks");
        report.set("model", "double hybrid on a constrained common-orbital parent; separate alpha/beta occupied and virtual semicanonical blocks preserve the parent density; full-Coulomb unregularized SS/OS doubles-only PT2 with the supplied recipe coefficients; no singles correction and no correlated molecular derivative in this energy entry");
        report.set("pt2_excitation_model", "semicanonical doubles only");
        report.set("reference_common_orbital_residual", stationary);
        report.set(
            "reference_semicanonical_residual",
            crate::response::diagnostics::semicanonical_residual(scf),
        );
        let mut labels = report.get("units").cloned().expect("energy units");
        labels.set("reference_common_orbital_residual", "hartree");
        labels.set("reference_semicanonical_residual", "hartree");
        labels.set("pt2_excitation_model", "enum");
        report.set("units", labels);
    }
    if let (Some(config), Some((origin, convention))) = (config.as_ref(), origin) {
        let ranks = correlated_moment_ranks(
            &setup,
            &reference,
            target.as_ref(),
            target_vv10,
            &problem,
            &tensor,
            coefficients,
            origin,
            max_moment_rank,
            config,
        )?;
        report.set("model","analytic double-hybrid electric moments with full parent orbital response, noncanonical PT2 amplitude response and distinct-target density derivative; integer zero-temperature reference; full-Coulomb PT2 without denominator regularization");
        report.set("origin", origin.to_array());
        report.set("origin_convention", convention);
        report.set("provenance", "analytic");
        let mut labels = report.get("units").cloned().expect("energy units");
        if max_moment_rank.is_some() {
            append_correlated_multipoles(&mut report, &mut labels, ranks);
        } else {
            append_correlated_dipole(&mut report, &mut labels, &ranks[0]);
            super::molecular::warn_charged_dipole(&mut report, setup.molecule.charge);
        }
        labels.set("origin", "bohr");
        report.set("units", labels);
    }
    if id == "polarizability" {
        let value = super::correlated_electric::polarizability(
            &setup,
            &reference,
            target.as_ref(),
            target_vv10,
            &problem,
            &tensor,
            coefficients,
            &dipole_matrices(&setup.basis, crate::math::Vec3::zero()),
            config.as_ref().expect("response configuration"),
        )?;
        report = super::vibrational::polarizability_report_fields(report, value.tensor);
        report.set("model", "static double-hybrid energy Hessian with second-order parent orbital response, noncanonical PT2 amplitude Taylor equations and distinct-target density/kernel derivatives; six independent static components; integer zero-temperature occupations; unregularized full-Coulomb PT2");
        report.set("raw_asymmetry", Value::Null);
        report.set("independent_components", 6usize);
        report.set("parent_polarizability", value.parent);
        report.set("pt2_polarizability_correction", value.pt2);
        report.set("target_polarizability_correction", value.target);
        report.set("pt2_same_spin_alpha_polarizability", value.channels[0]);
        report.set("pt2_opposite_spin_polarizability", value.channels[1]);
        report.set("pt2_same_spin_beta_polarizability", value.channels[2]);
        let mut labels = report.get("units").cloned().expect("polarizability units");
        append_quadratic_diagnostics(
            &mut report,
            &mut labels,
            &value,
            "e*bohr",
            "e^2*bohr^2/hartree",
        );
        for key in [
            "parent_polarizability",
            "pt2_polarizability_correction",
            "target_polarizability_correction",
            "pt2_same_spin_alpha_polarizability",
            "pt2_opposite_spin_polarizability",
            "pt2_same_spin_beta_polarizability",
        ] {
            labels.set(key, "e^2*bohr^2/hartree");
        }
        report.set("units", labels);
    }
    Ok(report)
}

pub(super) fn append_quadratic_diagnostics<const N: usize>(
    report: &mut Value,
    labels: &mut Value,
    value: &super::correlated_electric::CorrelatedPolarizability<N>,
    first_order_unit: &str,
    second_order_unit: &str,
) {
    report.set("response_order", 2usize);
    report.set("response_residuals", value.response_residuals.clone());
    report.set(
        "amplitude_response_residuals",
        value
            .amplitude_residuals
            .chunks(3)
            .map(|v| v.to_vec())
            .collect::<Vec<_>>(),
    );
    report.set("response_number_residuals", value.number_residuals.clone());
    report.set("projector_residuals", value.projector_residuals.clone());
    report.set(
        "stationarity_residuals",
        value.stationarity_residuals.clone(),
    );
    report.set("orbital_frame_residuals", value.frame_residuals.clone());
    labels.set("response_residuals", "dimensionless");
    labels.set(
        "amplitude_response_residuals",
        format!("[hartree, {first_order_unit}, {second_order_unit}] per directional path and Taylor order 0,1,2"),
    );
    for key in [
        "response_number_residuals",
        "projector_residuals",
        "stationarity_residuals",
        "orbital_frame_residuals",
    ] {
        labels.set(
            key,
            "atomic units, factorial-normalized directional Taylor coefficient diagnostics",
        );
    }
}

/// Add the common post-SCF energy decomposition to a property report using
/// its already converged parent. Property diagnostics and model take priority.
pub(super) fn append_prepared_energy(prepared: Prepared, mut report: Value) -> Result<Value> {
    // Reuse the common energy/decomposition report without a second SCF.
    let energy = calculate_prepared(prepared, "energy")?;
    if let Value::Object(entries) = energy {
        for (key, value) in entries {
            if key == "units" {
                let mut labels = value;
                if let Some(Value::Object(spectral)) = report.get("units") {
                    for (key, value) in spectral {
                        labels.set(key, value.clone());
                    }
                }
                report.set("units", labels);
            } else if matches!(key.as_str(), "energy" | "energy_components")
                || report.get(&key).is_none()
            {
                report.set(&key, value);
            }
        }
    }
    Ok(report)
}
