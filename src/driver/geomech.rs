// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum rock-physics calculations on supplied mineral and fluid data
//! (no electronic structure): multiphase aggregate bounds, dry frames
//! (including spheroidal pores), fluid substitution with Biot parameters and
//! thermal-conductivity bounds. Physical inputs have no defaults.

use super::{Controls, System, Value};
use crate::error::{DftError, Result};
use crate::geomech::inclusions::{
    conductivity_bounds, dem, kuster_toksoz, self_consistent, Inclusion, Moduli,
};
use crate::geomech::{
    aggregate_moduli, brown_korringa, critical_porosity_frame, fluid_substitution,
    hertz_mindlin_frame, Phase,
};

fn units(pairs: &[(&str, &str)]) -> Value {
    let mut out = Value::object();
    for (k, v) in pairs {
        out.set(k, *v);
    }
    out
}

fn required(controls: &Controls, key: &str) -> Result<f64> {
    controls.opt_f64(key)?.ok_or_else(|| {
        DftError::InvalidInput(format!(
            "`{key}` is required (no default for physical inputs)"
        ))
    })
}

fn tensor(controls: &Controls, key: &str) -> Result<Option<[[f64; 6]; 6]>> {
    match controls.opt_rows(key)? {
        None => Ok(None),
        Some(rows) if rows.len() == 6 && rows.iter().all(|r| r.len() == 6) => {
            Ok(Some(std::array::from_fn(|i| {
                std::array::from_fn(|j| rows[i][j])
            })))
        }
        Some(_) => Err(DftError::InvalidInput(format!(
            "`{key}` must be a 6x6 Voigt tensor (GPa)"
        ))),
    }
}

pub(super) fn run(id: &str, _system: &System, controls: &Controls) -> Result<Value> {
    match id {
        "aggregate-moduli" => aggregate(controls),
        "piezoelectric-strain" => piezoelectric_strain(controls),
        "rock-physics" => rock_physics(controls),
        other => Err(DftError::Unsupported {
            feature: format!("data calculation `{other}`"),
            missing: "a common-driver implementation".into(),
        }),
    }
}

/// Constitutive conversion only; no reference state or stability is inferred.
fn piezoelectric_strain(controls: &Controls) -> Result<Value> {
    let e = controls.opt_rows("piezoelectric_stress")?.ok_or_else(|| {
        DftError::InvalidInput("piezoelectric_stress is required: 3x6 in e/bohr^2".into())
    })?;
    let s = controls
        .opt_rows("piezoelectric_compliance")?
        .ok_or_else(|| {
            DftError::InvalidInput(
                "piezoelectric_compliance is required: 6x6 in bohr^3/hartree".into(),
            )
        })?;
    let ions = controls.opt_str("ion_response")?.ok_or_else(|| {
        DftError::InvalidInput(
            "ion_response must explicitly be clamped or relaxed for both tensors".into(),
        )
    })?;
    if !matches!(ions.as_str(), "clamped" | "relaxed") {
        return Err(DftError::InvalidInput(
            "ion_response must be clamped or relaxed".into(),
        ));
    }
    if e.len() != 3
        || e.iter().any(|r| r.len() != 6)
        || s.len() != 6
        || s.iter().any(|r| r.len() != 6)
    {
        return Err(DftError::InvalidInput(
            "piezoelectric stress/compliance shapes must be 3x6 and 6x6".into(),
        ));
    }
    controls.finish("piezoelectric-strain")?;
    let e: [[f64; 6]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| e[i][j]));
    let s: [[f64; 6]; 6] = std::array::from_fn(|i| std::array::from_fn(|j| s[i][j]));
    let d = crate::geomech::piezoelectric::strain_coefficients(&e, &s)?;
    let si = crate::constants::ELEMENTARY_CHARGE_C * crate::constants::BOHR_TO_METER
        / crate::constants::HARTREE_TO_JOULE;
    Ok(Value::object()
        .with(
            "model",
            "proper d=eS at fixed electric field on supplied tensors",
        )
        .with(
            "provenance",
            "algebraic tensor conversion; no electronic reference or stability calculation",
        )
        .with("ion_response", ions)
        .with("voigt_order", vec!["xx", "yy", "zz", "yz", "xz", "xy"])
        .with("engineering_shear", true)
        .with("mechanical_stability_assessed", false)
        .with("converged", true)
        .with("piezoelectric_stress", e)
        .with("piezoelectric_compliance", s)
        .with("strain_coefficients", d)
        .with("strain_coefficients_si", d.map(|r| r.map(|v| v * si)))
        .with(
            "strain_coefficients_pm_v",
            d.map(|r| r.map(|v| v * si * 1e12)),
        )
        .with(
            "units",
            units(&[
                ("piezoelectric_stress", "e/bohr^2"),
                ("piezoelectric_compliance", "bohr^3/hartree"),
                ("strain_coefficients", "e*bohr/hartree"),
                ("strain_coefficients_si", "C/N"),
                ("strain_coefficients_pm_v", "pm/V"),
            ]),
        ))
}

/// Voigt, Reuss, Hill, Hashin-Shtrikman and self-consistent moduli of an
/// isotropic multiphase aggregate.
fn aggregate(controls: &Controls) -> Result<Value> {
    let rows = controls.opt_rows("phases")?.ok_or_else(|| {
        DftError::InvalidInput(
            "`phases` is required: rows of [volume fraction, K (GPa), G (GPa)]".into(),
        )
    })?;
    controls.finish("aggregate-moduli")?;
    if rows.iter().any(|r| r.len() != 3) {
        return Err(DftError::InvalidInput(
            "every phase row is [volume fraction, K (GPa), G (GPa)]".into(),
        ));
    }
    let phases: Vec<Phase> = rows
        .iter()
        .map(|r| Phase {
            fraction: r[0],
            bulk: r[1],
            shear: r[2],
        })
        .collect();
    let m = aggregate_moduli(&phases)?;
    let pair = |p: (f64, f64)| vec![p.0, p.1];
    let mut report = Value::object()
        .with(
            "model",
            "isotropic multiphase aggregate: Voigt, Reuss and Hill averages, \
            Hashin-Shtrikman bounds and the self-consistent estimate for spherical \
            grains",
        )
        .with("converged", true)
        .with("warnings", Vec::<String>::new())
        .with("phases", rows.clone())
        .with("voigt_gpa", pair(m.voigt))
        .with("reuss_gpa", pair(m.reuss))
        .with("hill_gpa", pair(m.hill))
        .with("hashin_shtrikman_lower_gpa", pair(m.hs_lower))
        .with("hashin_shtrikman_upper_gpa", pair(m.hs_upper));
    report.set(
        "self_consistent_gpa",
        m.self_consistent.map_or(Value::Null, |p| {
            Value::Array(vec![Value::Float(p.0), Value::Float(p.1)])
        }),
    );
    report.set(
        "units",
        units(&[
            ("phases", "[volume fraction, K (GPa), G (GPa)]"),
            ("voigt_gpa", "[K, G] GPa"),
            ("reuss_gpa", "[K, G] GPa"),
            ("hill_gpa", "[K, G] GPa"),
            ("hashin_shtrikman_lower_gpa", "[K, G] GPa"),
            ("hashin_shtrikman_upper_gpa", "[K, G] GPa"),
            (
                "self_consistent_gpa",
                "[K, G] GPa (null when the iteration does not converge)",
            ),
        ]),
    );
    Ok(report)
}

/// Dry frame, Gassmann fluid substitution and Biot parameters, with the
/// anisotropic Brown-Korringa tensor when mineral and dry tensors are given.
fn rock_physics(controls: &Controls) -> Result<Value> {
    let k_s = required(controls, "mineral_bulk_gpa")?;
    let g_s = required(controls, "mineral_shear_gpa")?;
    let rho_s = required(controls, "mineral_density_kg_m3")?;
    let phi = required(controls, "porosity")?;
    let k_f = required(controls, "fluid_bulk_gpa")?;
    let rho_f = required(controls, "fluid_density_kg_m3")?;
    let frame = controls.opt_str("frame")?.ok_or_else(|| {
        DftError::InvalidInput(
            "`frame` is required: critical-porosity, hertz-mindlin, hashin-shtrikman, dem, \
             kuster-toksoz, self-consistent or given"
                .into(),
        )
    })?;
    let (k_dry, g_dry) = match frame.as_str() {
        "critical-porosity" => {
            critical_porosity_frame(k_s, g_s, phi, required(controls, "critical_porosity")?)?
        }
        "hertz-mindlin" => hertz_mindlin_frame(
            k_s,
            g_s,
            phi,
            required(controls, "coordination_number")?,
            required(controls, "effective_pressure_gpa")?,
        )?,
        // Upper Hashin-Shtrikman bound of mineral and empty pores (the
        // lower bound of an empty-pore frame is zero).
        "hashin-shtrikman" => {
            let m = aggregate_moduli(&[
                Phase {
                    fraction: 1.0 - phi,
                    bulk: k_s,
                    shear: g_s,
                },
                Phase {
                    fraction: phi,
                    bulk: 0.0,
                    shear: 0.0,
                },
            ])?;
            m.hs_upper
        }
        // Dry spheroidal pores of one aspect ratio in the mineral.
        "dem" | "kuster-toksoz" | "self-consistent" => {
            let aspect = required(controls, "pore_aspect_ratio")?;
            let mineral = Moduli {
                bulk: k_s,
                shear: g_s,
            };
            let pore = Moduli {
                bulk: 0.0,
                shear: 0.0,
            };
            let dry = match frame.as_str() {
                "dem" => dem(mineral, pore, aspect, phi)?.0,
                "kuster-toksoz" => kuster_toksoz(
                    mineral,
                    &[Inclusion {
                        fraction: phi,
                        moduli: pore,
                        aspect,
                    }],
                )?,
                _ => self_consistent(&[
                    Inclusion {
                        fraction: 1.0 - phi,
                        moduli: mineral,
                        aspect: 1.0,
                    },
                    Inclusion {
                        fraction: phi,
                        moduli: pore,
                        aspect,
                    },
                ])?
                .ok_or_else(|| {
                    DftError::InvalidInput(
                        "the self-consistent frame did not converge: the porosity is above the \
                         percolation threshold of this pore shape"
                            .into(),
                    )
                })?,
            };
            (dry.bulk, dry.shear)
        }
        "given" => (
            required(controls, "dry_bulk_gpa")?,
            required(controls, "dry_shear_gpa")?,
        ),
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown frame `{other}`: critical-porosity, hertz-mindlin, hashin-shtrikman, \
                 dem, kuster-toksoz, self-consistent or given"
            )))
        }
    };
    let viscosity = controls.opt_f64("fluid_viscosity_pa_s")?;
    let permeability = controls.opt_f64("permeability_m2")?;
    let mineral_tensor = tensor(controls, "mineral_tensor_gpa")?;
    let dry_tensor = tensor(controls, "dry_tensor_gpa")?;
    let mineral_conductivity = controls.opt_f64("mineral_conductivity_w_mk")?;
    let fluid_conductivity = controls.opt_f64("fluid_conductivity_w_mk")?;
    controls.finish("rock-physics")?;
    let biot = match (viscosity, permeability) {
        (Some(eta), Some(kappa)) => Some((eta, kappa)),
        (None, None) => None,
        _ => {
            return Err(DftError::InvalidInput(
                "the Biot frequency needs both `fluid_viscosity_pa_s` and `permeability_m2`".into(),
            ))
        }
    };
    let s = fluid_substitution(k_dry, g_dry, k_s, rho_s, k_f, rho_f, phi, biot)?;
    let mut report = Value::object()
        .with("model", "dry frame, Gassmann fluid substitution and Biot poroelastic parameters (low-frequency limit)")
        .with("converged", true)
        .with("warnings", Vec::<String>::new())
        .with("frame", frame.as_str())
        .with("dry_bulk_gpa", k_dry)
        .with("dry_shear_gpa", g_dry)
        .with("saturated_bulk_gpa", s.bulk_saturated)
        .with("saturated_shear_gpa", s.shear)
        .with("biot_coefficient", s.biot_coefficient)
        .with("biot_modulus_gpa", s.biot_modulus)
        .with("skempton_b", s.skempton)
        .with("poisson_drained", s.poisson_drained)
        .with("poisson_undrained", s.poisson_undrained)
        .with("bulk_density_kg_m3", s.density)
        .with("vp_m_s", s.vp)
        .with("vs_m_s", s.vs);
    report.set(
        "biot_frequency_hz",
        s.biot_frequency.map_or(Value::Null, Value::Float),
    );
    match (mineral_tensor, dry_tensor) {
        (Some(c_s), Some(c_d)) => report.set(
            "brown_korringa_saturated_tensor_gpa",
            brown_korringa(&c_d, &c_s, k_f, phi)?,
        ),
        (None, None) => {}
        _ => {
            return Err(DftError::InvalidInput(
                "Brown-Korringa needs both `mineral_tensor_gpa` and `dry_tensor_gpa`".into(),
            ))
        }
    }
    match (mineral_conductivity, fluid_conductivity) {
        (Some(solid), Some(fluid)) => {
            let b = conductivity_bounds(&[(1.0 - phi, solid), (phi, fluid)])?;
            report.set(
                "thermal_conductivity_w_mk",
                Value::object()
                    .with("arithmetic", b.arithmetic)
                    .with("harmonic", b.harmonic)
                    .with("geometric", b.geometric)
                    .with("hashin_shtrikman_lower", b.hashin_shtrikman_lower)
                    .with("hashin_shtrikman_upper", b.hashin_shtrikman_upper),
            );
        }
        (None, None) => {}
        _ => {
            return Err(DftError::InvalidInput(
                "the thermal conductivity needs both `mineral_conductivity_w_mk` and \
                 `fluid_conductivity_w_mk`"
                    .into(),
            ))
        }
    }
    report.set(
        "units",
        units(&[
            ("dry_bulk_gpa", "GPa"),
            ("saturated_bulk_gpa", "GPa (Gassmann)"),
            ("biot_modulus_gpa", "GPa"),
            ("bulk_density_kg_m3", "kg/m^3"),
            ("vp_m_s", "m/s"),
            ("vs_m_s", "m/s"),
            ("biot_frequency_hz", "Hz (eta phi / (2 pi rho_f kappa))"),
            ("brown_korringa_saturated_tensor_gpa", "GPa (Voigt)"),
            (
                "thermal_conductivity_w_mk",
                "W/(m K): means and Hashin-Shtrikman bounds of the saturated rock",
            ),
        ]),
    );
    Ok(report)
}
