// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared molecular DFT+U input and results.

use super::method::finish_molecular_reference;
use super::molecular::{reference_orbital_residual, reference_report, setup, units};
use super::{Controls, System, Value};
use crate::scf::hubbard::{
    self, HubbardModel, Interaction, IntersiteEnergy, IntersiteInteraction, Site, SpinorHubbardFields,
};
use crate::scf::Reference;
use crate::{DftError, Result};

fn read_site(value: &Value) -> Result<Site> {
    let controls = Controls::from_value(value)?;
    let orbitals = controls.opt_usize_list("orbitals")?.ok_or_else(|| {
        DftError::InvalidInput("each Hubbard site requires an orbitals array".into())
    })?;
    let parameter = |name: &str| {
        controls.opt_f64(name)?.ok_or_else(|| {
            DftError::InvalidInput(format!("the Hubbard site requires {name} in Hartree"))
        })
    };
    let interaction = match controls.str("model", "dudarev")?.as_str() {
        "dudarev" => Interaction::Dudarev {
            u_eff: parameter("u_eff")?,
        },
        "liechtenstein" => Interaction::Liechtenstein {
            u: parameter("u")?,
            j: parameter("j")?,
            integrals: controls.opt_f64_list("integrals")?.ok_or_else(|| {
                DftError::InvalidInput(
                    "a Liechtenstein site requires its complete on-site Coulomb tensor".into(),
                )
            })?,
        },
        model => {
            return Err(DftError::InvalidInput(format!(
                "unknown Hubbard model `{model}`; use dudarev or liechtenstein"
            )));
        }
    };
    controls.finish("Hubbard site")?;
    Ok(Site {
        orbitals,
        interaction,
    })
}

pub(super) fn read_sites(controls: &Controls) -> Result<Vec<Site>> {
    controls
        .take("hubbard_sites")
        .and_then(Value::as_array)
        .filter(|entries| !entries.is_empty())
        .ok_or_else(|| {
            DftError::InvalidInput("hubbard_sites requires a nonempty array of site objects".into())
        })?
        .iter()
        .map(read_site)
        .collect()
}

pub(super) fn read_pairs(controls: &Controls) -> Result<Vec<IntersiteInteraction>> {
    let Some(value) = controls.take("hubbard_pairs") else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| DftError::InvalidInput("hubbard_pairs must be an array of pair objects".into()))?
        .iter()
        .map(|value| {
            let pair = Controls::from_value(value)?;
            let sites = pair
                .opt_usize_list("sites")?
                .and_then(|sites| <[usize; 2]>::try_from(sites).ok())
                .ok_or_else(|| {
                    DftError::InvalidInput(
                        "each Hubbard pair requires two zero-based site indices".into(),
                    )
                })?;
            let v = pair.opt_f64("v")?.ok_or_else(|| {
                DftError::InvalidInput("each Hubbard pair requires its interaction v".into())
            })?;
            pair.finish("intersite Hubbard pair")?;
            Ok(IntersiteInteraction { sites, v })
        })
        .collect()
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let selector = controls.str("reference", "auto")?.to_ascii_lowercase();
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "DFT+U with a double hybrid".into(),
            missing: "the correlated density dependence of the site correction".into(),
        });
    }
    let explicit = match selector.as_str() {
        "auto" | "roks" | "rohf" => None,
        "rks" | "rhf" => Some(Reference::Rhf),
        "uks" | "uhf" => Some(Reference::Uhf),
        "gks" | "ghf" => return super::gks::run("dft-plus-u", system, controls),
        _ => {
            return Err(DftError::InvalidInput(format!(
                "unknown reference `{selector}`"
            )))
        }
    };
    let mut setup = setup(system, controls)?;
    if setup.method.stabilize {
        return Err(DftError::Unsupported {
            feature: "DFT+U with orbital-stability following".into(),
            missing: "the Hubbard stability-following connection".into(),
        });
    }
    if let Some(reference) = explicit {
        if reference == Reference::Rhf && setup.molecule.multiplicity != 1 {
            return Err(DftError::InvalidInput(
                "a restricted closed-shell reference requires multiplicity=1".into(),
            ));
        }
        setup.options.reference = reference;
    }
    let sites = read_sites(controls)?;
    let pairs = read_pairs(controls)?;
    controls.finish("dft-plus-u")?;

    let common = matches!(selector.as_str(), "roks" | "rohf");
    let model = HubbardModel::new(
        &crate::integrals::overlap_matrix(&setup.basis),
        sites,
        setup.options.orth_threshold,
    )?.with_intersite(pairs)?;
    let result = hubbard::run_with_model(
        &setup.molecule,
        &setup.basis,
        &setup.options,
        &setup.functional,
        setup.grid.as_ref(),
        &model,
        common,
    )?;
    let reference = finish_molecular_reference(&setup, result.scf, None)?;
    let mut report = reference_report(&setup, &reference, "variational molecular DFT+U");
    let sites: Vec<_> = result
        .correction
        .sites
        .into_iter()
        .map(|site| {
            let [alpha, beta] = site.occupations;
            Value::object()
                .with("interaction", site.interaction)
                .with("double_counting", site.double_counting)
                .with("occupations_alpha", alpha)
                .with("occupations_beta", beta)
        })
        .collect();
    report.set("hubbard_sites", sites);
    report.set("hubbard_energy", result.correction.energy);
    report.set(
        "scf_orbital_residual",
        reference_orbital_residual(&reference.scf),
    );
    report.set("requested_reference", selector);
    report.set(
        "reference_type",
        if common {
            "roks"
        } else if reference.scf.unrestricted {
            "uks"
        } else {
            "rks"
        },
    );
    report.set(
        "units",
        units(&[
            ("hubbard_energy", "hartree"),
            ("hubbard_sites.interaction", "hartree"),
            ("hubbard_sites.double_counting", "hartree"),
            ("hubbard_sites.occupations_alpha", "electrons"),
            ("hubbard_sites.occupations_beta", "electrons"),
        ]),
    );
    if !result.correction.pairs.is_empty() {
        report.set("model", "variational molecular DFT+U+V at fixed Hubbard projectors");
        append_pairs(&mut report, result.correction.pairs);
    }
    Ok(report)
}

fn append_pairs(report: &mut Value, pairs: Vec<IntersiteEnergy>) {
    let energy = pairs.iter().map(|pair| pair.correction).sum::<f64>();
    let pairs: Vec<_> = pairs.into_iter().map(|item| {
        Value::object()
            .with("sites", item.pair.sites.to_vec())
            .with("v", item.pair.v)
            .with("interaction", item.interaction)
            .with("double_counting", item.double_counting)
            .with("correction", item.correction)
    }).collect();
    report.set("hubbard_pairs", pairs);
    report.set("hubbard_intersite_energy", energy);
    let mut labels = report.get("units").cloned().expect("Hubbard energy units");
    for key in ["hubbard_pairs.v", "hubbard_pairs.interaction", "hubbard_pairs.double_counting",
        "hubbard_pairs.correction", "hubbard_intersite_energy"] {
        labels.set(key, "hartree");
    }
    report.set("units", labels);
}

pub(super) fn append_spinor(
    report: &mut Value,
    fields: SpinorHubbardFields,
    orbital_residual: f64,
) {
    let sites: Vec<_> = fields
        .sites
        .into_iter()
        .map(|site| {
            Value::object()
                .with("interaction", site.interaction)
                .with("double_counting", site.double_counting)
                .with("occupations_real", site.occupations.re)
                .with("occupations_imaginary", site.occupations.im)
        })
        .collect();
    report.set(
        "model",
        "variational molecular spinor DFT+U with spin-covariant Hubbard FLL",
    );
    report.set("hubbard_sites", sites);
    report.set("hubbard_energy", fields.energy);
    report.set(
        "hubbard_occupation_order",
        "spin-major [up site orbitals, down site orbitals]",
    );
    report.set("scf_orbital_residual", orbital_residual);
    let mut components = report
        .get("energy_components")
        .cloned()
        .expect("spinor energy terms");
    components.set("hubbard", fields.energy);
    report.set("energy_components", components);
    let mut labels = report.get("units").cloned().expect("spinor energy units");
    for (key, unit) in [
        ("hubbard_energy", "hartree"),
        ("hubbard_sites.interaction", "hartree"),
        ("hubbard_sites.double_counting", "hartree"),
        ("hubbard_sites.occupations_real", "electrons"),
        ("hubbard_sites.occupations_imaginary", "electrons"),
        ("scf_orbital_residual", "hartree"),
    ] {
        labels.set(key, unit);
    }
    report.set("units", labels);
    if !fields.pairs.is_empty() {
        report.set("model", "variational molecular spinor DFT+U+V at fixed spin-independent Hubbard projectors");
        append_pairs(report, fields.pairs);
    }
}
