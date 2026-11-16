// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Post-SCF density-dependent dispersion for the shared molecular energy.

use super::method::{prepare_molecular, MolecularMethod, MolecularSetup};
use super::{Controls, System, Value};
use crate::dispersion::{d1, mbd, ts};
use crate::linalg::Matrix;
use crate::nlc::vdw_df::Variant;
use crate::{DftError, Result};

#[derive(Clone, Copy)]
enum Model {
    Ts,
    MbdTs,
    MbdRsScs,
    Vv09,
    VdwDf(Variant),
}

impl Model {
    fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "ts" => Some(Self::Ts),
            "mbd-ts" => Some(Self::MbdTs),
            "mbd" | "mbd-rsscs" => Some(Self::MbdRsScs),
            "vv09" => Some(Self::Vv09),
            other => Variant::from_name(other).map(Self::VdwDf),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ts => "TS",
            Self::MbdTs => "MBD@TS",
            Self::MbdRsScs => "MBD@rsSCS",
            Self::Vv09 => "VV09",
            Self::VdwDf(variant) => variant.name(),
        }
    }
}

/// Whether the molecular energy driver handles this explicit correction.
pub fn handles(name: &str) -> bool {
    name.trim().eq_ignore_ascii_case("d1") || Model::parse(name).is_some()
}

/// Refuse derivatives and PT2 combinations before an ordinary property
/// branch could construct a reference that omits the requested correction.
pub(super) fn check_scope(id: &str, controls: &Controls) -> Result<()> {
    let name = controls
        .peek_str("dispersion")?
        .or(controls.peek_str("disp")?);
    if name.is_some_and(|name| name.trim().eq_ignore_ascii_case("d1")) {
        if !matches!(
            id,
            "energy" | "gradient" | "hessian" | "roks" | "roks-hessian" | "force-constants"
        ) || super::double_hybrid::requested(controls)?
        {
            return Err(DftError::Unsupported {
                feature: format!("molecular D1 on {id}"),
                missing: "the D1 connection for this calculation and reference".into(),
            });
        }
        return Ok(());
    }
    let Some(model) = name.and_then(Model::parse) else {
        return Ok(());
    };
    if !matches!(id, "energy" | "roks") || (id == "roks" && controls.bool("gradient", true)?) {
        return Err(DftError::Unsupported {
            feature: format!("{} on {id}", model.name()),
            missing: "the density-dependent dispersion response; use molecular energy for a post-SCF correction".into(),
        });
    }
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: format!("{} on a double-hybrid energy", model.name()),
            missing: "a validated combination of this correction and the selected PT2 model".into(),
        });
    }
    if let Model::VdwDf(variant) = model {
        let reference = controls
            .peek_str("reference")?
            .map(|name| name.trim().to_ascii_lowercase());
        if id == "roks"
            || reference
                .as_deref()
                .is_some_and(|name| !matches!(name, "rks" | "auto"))
        {
            return Err(DftError::Unsupported {
                feature: format!("{} with the selected reference", variant.name()),
                missing:
                    "an unpolarized molecular RKS reference for the fixed-density XC functional"
                        .into(),
            });
        }
    }
    Ok(())
}

pub(super) struct AtomicInputs {
    references: Vec<ts::FreeAtom>,
    scale: f64,
}

pub(super) struct ScreeningInputs {
    quadrature_nodes: usize,
    frequency_scale: f64,
}

pub(super) enum Request {
    Ts(AtomicInputs),
    MbdTs(AtomicInputs),
    MbdRsScs(AtomicInputs, ScreeningInputs),
    Vv09,
    VdwDf(Variant, super::vdw_df_post_scf::Quadrature),
}

pub(super) fn d1_inputs(controls: &Controls) -> Result<(Vec<d1::AtomParameters>, f64)> {
    let c6 = atom_values("D1", controls, "dispersion_free_c6", None)?;
    let radii = atom_values("D1", controls, "dispersion_free_radii", None)?;
    if c6.len() != radii.len() {
        return Err(DftError::InvalidInput(
            "D1 needs the same number of coefficients and radii".into(),
        ));
    }
    let atoms = c6
        .into_iter()
        .zip(radii)
        .map(|(c6, radius)| d1::AtomParameters { c6, radius })
        .collect();
    Ok((
        atoms,
        positive_control("D1", controls, "dispersion_energy_scale")?,
    ))
}

/// Preserve the molecular energy report fields when a D1 contribution comes
/// from the ordinary stationary reference and its analytic derivatives.
pub(super) fn report_d1(
    setup: &MolecularSetup,
    energy: f64,
    residual: impl FnOnce() -> f64,
    report: &mut Value,
) {
    if let crate::dispersion::Dispersion::D1 { s6, .. } = &setup.dispersion {
        report.set("dispersion_model", "D1");
        report.set("dispersion_energy", energy);
        report.set("dispersion_energy_scale", *s6);
        report.set("scf_orbital_residual", residual());
    }
}

fn atom_values(
    model: &str,
    controls: &Controls,
    key: &str,
    count: Option<usize>,
) -> Result<Vec<f64>> {
    match controls.opt_f64_list(key)? {
        Some(values)
            if !values.is_empty()
                && count.is_none_or(|n| values.len() == n)
                && values.iter().all(|v| v.is_finite() && *v > 0.0) =>
        {
            Ok(values)
        }
        _ => {
            let count = count.map_or_else(|| "one or more".into(), |n| n.to_string());
            Err(DftError::InvalidInput(format!(
                "{model} requires {count} positive finite {key} values in atom order"
            )))
        }
    }
}

fn positive_control(model: &str, controls: &Controls, key: &str) -> Result<f64> {
    controls
        .opt_f64(key)?
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| DftError::InvalidInput(format!("{model} requires a positive finite {key}")))
}

impl Request {
    fn parse(model: Model, controls: &Controls, count: Option<usize>) -> Result<Self> {
        if matches!(model, Model::Vv09) {
            return Ok(Self::Vv09);
        }
        if let Model::VdwDf(variant) = model {
            return Ok(Self::VdwDf(
                variant,
                super::vdw_df_post_scf::Quadrature::from_controls(controls)?,
            ));
        }
        let inputs = |key| atom_values(model.name(), controls, key, count);
        let c6 = inputs("dispersion_free_c6")?;
        let radii = inputs("dispersion_free_radii")?;
        let polarizabilities = inputs("dispersion_free_polarizabilities")?;
        let scale = positive_control(model.name(), controls, "dispersion_radius_scale")?;
        let references = polarizabilities
            .into_iter()
            .zip(c6)
            .zip(radii)
            .map(|((polarizability, c6), radius)| ts::FreeAtom {
                polarizability,
                c6,
                radius,
            })
            .collect();
        let inputs = AtomicInputs { references, scale };
        Ok(match model {
            Model::Ts => Self::Ts(inputs),
            Model::MbdTs => Self::MbdTs(inputs),
            Model::MbdRsScs => {
                let quadrature_nodes = controls.usize("mbd_quadrature_nodes", 64)?;
                let frequency_scale = controls.f64("mbd_frequency_scale", 0.8)?;
                if !(4..=256).contains(&quadrature_nodes)
                    || !frequency_scale.is_finite()
                    || frequency_scale <= 0.0
                {
                    return Err(DftError::InvalidInput(
                        "MBD@rsSCS needs 4..=256 quadrature nodes and a positive finite frequency scale".into(),
                    ));
                }
                Self::MbdRsScs(
                    inputs,
                    ScreeningInputs {
                        quadrature_nodes,
                        frequency_scale,
                    },
                )
            }
            Model::Vv09 | Model::VdwDf(_) => {
                unreachable!("model handled before density parameter parsing")
            }
        })
    }

    fn model(&self) -> Model {
        match self {
            Self::Ts(_) => Model::Ts,
            Self::MbdTs(_) => Model::MbdTs,
            Self::MbdRsScs(..) => Model::MbdRsScs,
            Self::Vv09 => Model::Vv09,
            Self::VdwDf(variant, _) => Model::VdwDf(*variant),
        }
    }

    pub(super) fn evaluate_collinear(
        self,
        setup: &MolecularSetup,
        scf: &crate::scf::ScfResult,
        report: &mut Value,
    ) -> Result<()> {
        report.set(
            "scf_orbital_residual",
            super::molecular::reference_orbital_residual(scf),
        );
        let density = scf.p_total();
        if let Self::VdwDf(variant, quadrature) = self {
            return super::vdw_df_post_scf::evaluate(
                variant, quadrature, setup, scf, &density, report,
            );
        }
        if matches!(self, Self::Vv09) {
            let mut spin = scf.density[0].clone();
            for (a, b) in spin
                .as_mut_slice()
                .iter_mut()
                .zip(scf.density[1].as_slice())
            {
                *a -= b;
            }
            self.evaluate(setup, &density, &[&spin], report)
        } else {
            self.evaluate(setup, &density, &[], report)
        }
    }

    /// Evaluate the same energy correction on a collinear or spinor total
    /// density. For spinors the scalar real AO density is supplied by the
    /// caller, so the partition does not depend on the spin frame.
    pub(super) fn evaluate(
        self,
        setup: &MolecularSetup,
        density: &Matrix,
        spin: &[&Matrix],
        report: &mut Value,
    ) -> Result<()> {
        if let Some(residual) = report.get("orbital_residual").and_then(Value::as_f64) {
            report.set("scf_orbital_residual", residual);
        }
        let model = self.model();
        let (energy, atomic_terms) = match self {
            Self::Ts(inputs) => {
                let result = ts::energy_on_density(
                    &setup.molecule,
                    &setup.basis,
                    setup.integration_grid()?.as_ref(),
                    density,
                    &inputs.references,
                    inputs.scale,
                )?;
                (result.energy, Some((inputs.scale, result.atoms)))
            }
            Self::MbdTs(inputs) => {
                let result = mbd::ts_energy_on_density(
                    &setup.molecule,
                    &setup.basis,
                    setup.integration_grid()?.as_ref(),
                    density,
                    &inputs.references,
                    inputs.scale,
                )?;
                (
                    result.energy,
                    Some((inputs.scale, report_oscillators(result, report))),
                )
            }
            Self::MbdRsScs(inputs, screening) => {
                let result = mbd::rsscs_energy_on_density(
                    &setup.molecule,
                    &setup.basis,
                    setup.integration_grid()?.as_ref(),
                    density,
                    &inputs.references,
                    inputs.scale,
                    screening.quadrature_nodes,
                    screening.frequency_scale,
                )?;
                report.set(
                    "mbd_minimum_screening_eigenvalue",
                    result.minimum_screening_eigenvalue,
                );
                report.set("mbd_screening_residual", result.screening_residual);
                report.set("mbd_quadrature_nodes", screening.quadrature_nodes);
                report.set("mbd_frequency_scale", screening.frequency_scale);
                report.set(
                    "mbd_unscreened_polarizabilities",
                    result
                        .unscreened_atoms
                        .iter()
                        .map(|atom| atom.polarizability)
                        .collect::<Vec<_>>(),
                );
                report.set(
                    "mbd_unscreened_c6",
                    result
                        .unscreened_atoms
                        .iter()
                        .map(|atom| atom.c6)
                        .collect::<Vec<_>>(),
                );
                report.set(
                    "mbd_unscreened_radii",
                    result
                        .unscreened_atoms
                        .iter()
                        .map(|atom| atom.radius)
                        .collect::<Vec<_>>(),
                );
                (
                    result.oscillators.energy,
                    Some((inputs.scale, report_oscillators(result.oscillators, report))),
                )
            }
            Self::Vv09 => (
                crate::nlc::vv09::energy_on_density(
                    &setup.basis,
                    setup.integration_grid()?.as_ref(),
                    density,
                    spin,
                )?,
                None,
            ),
            Self::VdwDf(variant, _) => {
                return Err(DftError::Unsupported {
                    feature: format!("{} on a spinor density", variant.name()),
                    missing: "spin-resolved nonlocal correlation and its fixed-density XC assembly"
                        .into(),
                });
            }
        };
        for key in [
            "energy",
            "internal_energy",
            "free_energy",
            "mermin_free_energy",
        ] {
            if let Some(value) = report.get(key).and_then(Value::as_f64) {
                report.set(key, value + energy);
            }
        }
        if let Some(mut components) = report.get("energy_components").cloned() {
            components.set("dispersion", energy);
            report.set("energy_components", components);
        }
        report.set("dispersion_energy", energy);
        report.set("dispersion_model", model.name());
        if let Some((scale, atoms)) = atomic_terms {
            report.set("dispersion_density_reference", "post-SCF total density, spherical neutral all-electron SAD proatoms in the molecular basis");
            report.set("dispersion_radius_scale", scale);
            report.set(
                "dispersion_volume_ratios",
                atoms.iter().map(|a| a.volume_ratio).collect::<Vec<_>>(),
            );
            report.set(
                "dispersion_effective_polarizabilities",
                atoms.iter().map(|a| a.polarizability).collect::<Vec<_>>(),
            );
            report.set(
                "dispersion_effective_c6",
                atoms.iter().map(|a| a.c6).collect::<Vec<_>>(),
            );
            report.set(
                "dispersion_effective_radii",
                atoms.iter().map(|a| a.radius).collect::<Vec<_>>(),
            );
        } else if matches!(model, Model::Vv09) {
            report.set("dispersion_density_reference", "post-SCF VV09 nonlocal correlation of the total density, its gradient and the local spin polarization, paired with PW92 local correlation");
        }
        Ok(())
    }
}

fn report_oscillators(result: mbd::MbdEnergy, report: &mut Value) -> Vec<ts::AtomEnvironment> {
    report.set("mbd_frequencies", result.frequencies);
    report.set("mbd_coupled_frequencies", result.coupled_frequencies);
    report.set(
        "mbd_minimum_squared_frequency",
        result.minimum_squared_frequency,
    );
    result.atoms
}

pub(super) fn prepare_energy(
    system: &System,
    controls: &Controls,
) -> Result<(MolecularSetup, Option<Request>)> {
    controls.require_consumed("reference", "molecular energy setup")?;
    let mut method = MolecularMethod::from_controls(controls)?;
    let request = Model::parse(&method.dispersion)
        .map(|model| Request::parse(model, controls, Some(system.numbers.len())))
        .transpose()?;
    if request.is_some() {
        method.dispersion = "none".into();
    }
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    if let Some(request) = &request {
        if let Request::VdwDf(variant, _) = request {
            super::vdw_df_post_scf::check_parent(*variant, &setup)?;
        }
        if let Some(term) = setup.functional.required_nonlocal_term() {
            return Err(DftError::InvalidInput(format!(
                "{} cannot be combined with a functional that already requires {term}",
                request.model().name()
            )));
        }
        if setup.basis.n_core_electrons != 0 || system.numbers.contains(&0) {
            return Err(DftError::Unsupported {
                feature: format!("{} with an ECP or ghost centre", request.model().name()),
                missing: "an all-electron reconstruction of each participating atomic density"
                    .into(),
            });
        }
        if matches!(request, Request::Vv09) {
            let mut correlation = setup
                .functional
                .parts
                .iter()
                .filter(|(_, part)| part.kind() != crate::xc::Kind::Exchange);
            let pw92 = crate::xc::lda::pw92();
            let local_pw92 = correlation.next().is_some_and(|(weight, part)| {
                *weight == 1.0
                    && part.kind() == crate::xc::Kind::Correlation
                    && part.name() == pw92.name()
                    && part.params() == pw92.params()
            }) && correlation.next().is_none();
            if !local_pw92 || setup.options.skala.is_some() {
                return Err(DftError::InvalidInput("VV09 requires unscaled original PW92 local correlation with no other correlation part".into()));
            }
        }
    }
    Ok((setup, request))
}
