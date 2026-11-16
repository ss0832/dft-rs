// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Symmetry-adapted perturbation analysis of two molecular fragments with
//! the optional extensions of [`crate::sapt::SaptConfig`].

use super::controls::Controls;
use super::molecular::{setup, units};
use super::value::Value;
use super::System;
use crate::sapt::{sapt_interaction_diagnostics, SaptConfig, SaptDiagnostics, SaptVariant};
use crate::system::Molecule;
use crate::{DftError, Result};

/// Intermolecular perturbation analysis of fragments 0 and 1 of the system.
pub(super) fn sapt_analysis(system: &System, controls: &Controls) -> Result<Value> {
    let variant_name = controls.str("sapt_variant", "response")?;
    let (variant, model) = match variant_name.as_str() {
        "response" => (
            SaptVariant::Response,
            "Coupled monomer response with Heitler-London exchange, its linear exchange-induction, \
             and projected exchange-dispersion",
        ),
        "sapt0" => (
            SaptVariant::Sapt0,
            "SAPT(0) on Hartree-Fock monomers with the full antisymmetrizer, coupled induction, \
             and uncoupled orbital-pair dispersion",
        ),
        "sapt0-s2" => (
            SaptVariant::Sapt0SingleExchange,
            "SAPT(0) on Hartree-Fock monomers in the single-pair exchange approximation, \
             with coupled induction and uncoupled orbital-pair dispersion",
        ),
        "ssapt0" => (
            SaptVariant::ScaledSapt0,
            "sSAPT0 with third-power scaling of second-order single-pair exchange and \
             first-power scaling in the delta-HF subtraction",
        ),
        "sapt-30" => (
            SaptVariant::Sapt30Uncoupled,
            "Uncoupled canonical Hartree-Fock monomer SAPT through third intermolecular order \
             with the full antisymmetrizer",
        ),
        "sapt2+3" => (
            SaptVariant::Sapt2Plus3,
            "SAPT2+3 on closed-shell Hartree-Fock monomers with the full antisymmetrizer, \
             correlated monomer response and the delta-HF(3) correction",
        ),
        "sapt-ks-fdds" => (
            SaptVariant::KsFdds,
            "KS SAPT with coupled adiabatic frequency-dependent density susceptibilities, \
             coupled induction, full antisymmetrization and uncoupled KS exchange-dispersion",
        ),
        _ => {
            return Err(DftError::InvalidInput(
                "sapt_variant must be response, sapt0, sapt0-s2, ssapt0, sapt-30, sapt2+3 or sapt-ks-fdds".into(),
            ));
        }
    };
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "SAPT with a double hybrid".into(),
            missing: "the monomer's perturbative correlation and its response".into(),
        });
    }
    let fragment_of = controls.opt_usize_list("fragments")?.ok_or_else(|| {
        DftError::InvalidInput(
            "sapt-analysis needs fragments, the fragment index (0 or 1) of every atom".into(),
        )
    })?;
    let charges = controls.opt_f64_list("fragment_charges")?;
    let multiplicities = controls.opt_usize_list("fragment_multiplicities")?;
    let diagnostics = SaptDiagnostics {
        regularization_eta: controls.opt_f64("sapt_regularization_eta")?,
        relaxed_induction30: controls.bool("sapt_relaxed_induction30", false)?,
        correlation_terms: controls.bool("sapt_correlation_terms", false)?,
    };
    let config = SaptConfig {
        asymptotic_shift: [
            controls.opt_f64("asymptotic_shift_a")?,
            controls.opt_f64("asymptotic_shift_b")?,
        ],
        rpa_dispersion: controls.bool(
            "rpa_dispersion",
            matches!(variant, SaptVariant::Response | SaptVariant::KsFdds),
        )?,
        casimir_polder_points: controls.usize("casimir_polder_points", 24)?,
        charge_transfer: controls
            .bool("charge_transfer", variant != SaptVariant::Sapt30Uncoupled)?,
        partition: controls.bool("sapt_partition", variant != SaptVariant::Sapt30Uncoupled)?,
    };
    let setup = setup(system, controls)?;
    if variant != SaptVariant::Response
        && !matches!(setup.dispersion, crate::dispersion::Dispersion::None)
    {
        return Err(DftError::InvalidInput(format!(
            "{variant_name} requires monomers without an empirical dispersion correction"
        )));
    }
    controls.finish("sapt-analysis")?;
    let natoms = setup.molecule.len();
    if fragment_of.len() != natoms
        || fragment_of.iter().any(|&f| f > 1)
        || !fragment_of.contains(&0)
        || !fragment_of.contains(&1)
    {
        return Err(DftError::InvalidInput(
            "fragments holds one fragment index, 0 or 1, per atom with both fragments present"
                .into(),
        ));
    }
    if config.casimir_polder_points == 0 {
        return Err(DftError::InvalidInput(
            "casimir_polder_points must be positive".into(),
        ));
    }
    let charges: [i32; 2] = match charges {
        None if setup.molecule.charge == 0 => [0, 0],
        None => {
            return Err(DftError::InvalidInput(
                "a charged system needs fragment_charges, the charge of fragments 0 and 1".into(),
            ))
        }
        Some(list) => {
            if list.len() != 2 || list.iter().any(|q| q.fract() != 0.0 || !q.is_finite()) {
                return Err(DftError::InvalidInput(
                    "fragment_charges holds two integer charges".into(),
                ));
            }
            [list[0] as i32, list[1] as i32]
        }
    };
    if charges[0] + charges[1] != setup.molecule.charge {
        return Err(DftError::InvalidInput(format!(
            "fragment_charges sum to {} but the system charge is {}",
            charges[0] + charges[1],
            setup.molecule.charge
        )));
    }
    let atoms: [Vec<usize>; 2] =
        std::array::from_fn(|f| (0..natoms).filter(|&a| fragment_of[a] == f).collect());
    let mut monomers = Vec::with_capacity(2);
    for f in 0..2 {
        let molecule = Molecule::new(
            atoms[f]
                .iter()
                .map(|&a| setup.molecule.atoms[a].clone())
                .collect(),
        )
        .with_charge(charges[f]);
        let electrons = molecule.n_electrons()?;
        let multiplicity = match &multiplicities {
            Some(list) if list.len() == 2 => list[f],
            Some(_) => {
                return Err(DftError::InvalidInput(
                    "fragment_multiplicities holds two multiplicities".into(),
                ))
            }
            None => 1 + electrons % 2,
        };
        if multiplicity == 0 || multiplicity > electrons + 1 || (electrons + multiplicity) % 2 == 0
        {
            return Err(DftError::InvalidInput(format!(
                "multiplicity {multiplicity} is impossible for the {electrons} electrons of \
                 fragment {f}"
            )));
        }
        monomers.push(molecule.with_multiplicity(multiplicity));
    }
    let result = sapt_interaction_diagnostics(
        &monomers[0],
        &monomers[1],
        &setup.basis_set,
        &setup.functional,
        &setup.options,
        Some(&setup.grid_options),
        &config,
        variant,
        diagnostics,
    )?;
    let fragments: Vec<Value> = (0..2)
        .map(|f| {
            Value::object()
                .with("atoms", atoms[f].clone())
                .with("charge", i64::from(monomers[f].charge))
                .with("multiplicity", monomers[f].multiplicity)
                .with(
                    "energy",
                    if f == 0 {
                        result.e_monomer_a
                    } else {
                        result.e_monomer_b
                    },
                )
                .with("asymptotic_shift", result.asymptotic_shift[f])
        })
        .collect();
    let mut report = Value::object()
        .with("model", model)
        .with("sapt_variant", variant_name.as_str())
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("fragments", fragments)
        .with("electrostatic", result.e_elst)
        .with("exchange", result.e_exch)
        .with("induction", [result.e_ind_a, result.e_ind_b])
        .with("exchange_induction", result.e_exch_ind)
        .with(
            "exchange_induction_directions",
            result.e_exch_ind_directions,
        )
        .with("dispersion", result.e_disp_coupled)
        .with("exchange_scale", result.exchange_scale)
        .with("delta_hf", result.delta_hf)
        .with("exchange_dispersion", result.e_exch_disp)
        .with("dispersion_uncoupled", result.e_disp_unc)
        .with(
            "dispersion_uncoupled_sum_over_states",
            result.e_disp_sos_uncoupled,
        )
        .with("interaction_energy", result.interaction_total())
        .with("interaction_energy_rpa", result.interaction_total_rpa())
        .with("supermolecular_interaction", result.e_int_supermolecular)
        .with("delta", result.delta);
    if let Some(rpa) = &result.rpa {
        report.set(
            "rpa",
            Value::object()
                .with("dispersion", rpa.e_disp)
                .with("exchange_dispersion", rpa.e_exch_disp)
                .with("casimir_polder", rpa.e_disp_casimir_polder)
                .with("casimir_polder_change", rpa.casimir_polder_change)
                .with(
                    "casimir_polder_residual",
                    (rpa.e_disp_casimir_polder - rpa.e_disp).abs(),
                )
                .with(
                    "casimir_polder_uncoupled",
                    rpa.e_disp_casimir_polder_uncoupled,
                )
                .with(
                    "casimir_polder_uncoupled_residual",
                    (rpa.e_disp_casimir_polder_uncoupled - result.e_disp_unc).abs(),
                )
                .with("quadrature_points", rpa.points),
        );
    }
    if let Some(third) = result.third_order {
        report.set(
            "third_order",
            Value::object()
                .with("induction", third.induction)
                .with("dispersion", third.dispersion)
                .with("induction_dispersion", third.induction_dispersion)
                .with("exchange", third.exchange)
                .with("exchange_induction", third.exchange_induction)
                .with("exchange_dispersion", third.exchange_dispersion)
                .with(
                    "exchange_induction_dispersion",
                    third.exchange_induction_dispersion,
                )
                .with("total", third.total()),
        );
    }
    if let Some(terms) = result.sapt2_plus3 {
        report.set(
            "sapt2_plus3",
            Value::object()
                .with(
                    "relaxed_exchange_induction30",
                    terms.relaxed_exchange_induction30,
                )
                .with("delta_hf_order3", terms.delta_hf_order3)
                .with("correlated_electrostatics", terms.correlated_electrostatics)
                .with("correlated_exchange", terms.correlated_exchange)
                .with("correlated_induction", terms.correlated_induction)
                .with("correlated_dispersion", terms.correlated_dispersion)
                .with(
                    "cubic_dispersion_and_mixed",
                    terms.cubic_dispersion_and_mixed,
                ),
        );
    }
    if let Some(regularized) = result.regularized_induction {
        report.set(
            "regularized_induction",
            Value::object()
                .with("eta", regularized.eta)
                .with("induction", regularized.induction)
                .with("exchange_induction", regularized.exchange_induction)
                .with("charge_transfer", regularized.charge_transfer)
                .with(
                    "total_charge_transfer",
                    regularized.charge_transfer.iter().sum::<f64>(),
                ),
        );
    }
    if let Some(relaxed) = result.relaxed_induction30 {
        report.set(
            "relaxed_induction30",
            Value::object()
                .with("induction", relaxed.induction)
                .with("delta_hf_order2", relaxed.delta_hf_order2),
        );
    }
    if let Some(terms) = result.correlation_terms {
        report.set(
            "correlation_terms",
            Value::object()
                .with("electrostatics", terms.electrostatics)
                .with("exchange", terms.exchange)
                .with("relaxed_electrostatics", terms.relaxed_electrostatics)
                .with("second_order_polarization", terms.second_order_polarization)
                .with("second_order_exchange", terms.second_order_exchange)
                .with("truncated_induction22", terms.truncated_induction22)
                .with(
                    "truncated_exchange_induction22",
                    terms.truncated_exchange_induction22,
                ),
        );
    }
    if let Some(ct) = &result.charge_transfer {
        report.set(
            "charge_transfer",
            Value::object()
                .with("induction_dimer_basis", ct.e_ind_dimer_basis)
                .with("energy", ct.e_ct),
        );
    }
    if let Some(partition) = &result.partition {
        report.set(
            "partition",
            Value::object()
                .with("electrostatic", &partition.electrostatic)
                .with("induction_a", &partition.induction_a)
                .with("induction_b", &partition.induction_b)
                .with("residual", partition.residual),
        );
    }
    Ok(report.with(
        "units",
        units(&[
            ("fragments.energy", "hartree"),
            ("fragments.asymptotic_shift", "hartree"),
            ("electrostatic", "hartree"),
            ("exchange", "hartree"),
            ("induction", "hartree"),
            ("exchange_induction", "hartree"),
            ("exchange_induction_directions", "hartree"),
            ("exchange_scale", "dimensionless"),
            ("delta_hf", "hartree"),
            ("dispersion", "hartree"),
            ("exchange_dispersion", "hartree"),
            ("dispersion_uncoupled", "hartree"),
            ("dispersion_uncoupled_sum_over_states", "hartree"),
            ("interaction_energy", "hartree"),
            ("third_order.induction", "hartree"),
            ("third_order.dispersion", "hartree"),
            ("third_order.induction_dispersion", "hartree"),
            ("third_order.exchange", "hartree"),
            ("third_order.exchange_induction", "hartree"),
            ("third_order.exchange_dispersion", "hartree"),
            ("third_order.exchange_induction_dispersion", "hartree"),
            ("third_order.total", "hartree"),
            ("sapt2_plus3.relaxed_exchange_induction30", "hartree"),
            ("sapt2_plus3.delta_hf_order3", "hartree"),
            ("sapt2_plus3.correlated_electrostatics", "hartree"),
            ("sapt2_plus3.correlated_exchange", "hartree"),
            ("sapt2_plus3.correlated_induction", "hartree"),
            ("sapt2_plus3.correlated_dispersion", "hartree"),
            ("sapt2_plus3.cubic_dispersion_and_mixed", "hartree"),
            ("regularized_induction.eta", "bohr^-2"),
            ("regularized_induction.induction", "hartree"),
            ("regularized_induction.exchange_induction", "hartree"),
            ("regularized_induction.charge_transfer", "hartree"),
            ("regularized_induction.total_charge_transfer", "hartree"),
            ("relaxed_induction30.induction", "hartree"),
            ("relaxed_induction30.delta_hf_order2", "hartree"),
            ("correlation_terms.electrostatics", "hartree"),
            ("correlation_terms.exchange", "hartree"),
            ("correlation_terms.relaxed_electrostatics", "hartree"),
            ("correlation_terms.second_order_polarization", "hartree"),
            ("correlation_terms.second_order_exchange", "hartree"),
            ("correlation_terms.truncated_induction22", "hartree"),
            (
                "correlation_terms.truncated_exchange_induction22",
                "hartree",
            ),
            ("interaction_energy_rpa", "hartree"),
            ("supermolecular_interaction", "hartree"),
            ("delta", "hartree"),
            ("rpa.dispersion", "hartree"),
            ("rpa.exchange_dispersion", "hartree"),
            ("rpa.casimir_polder", "hartree"),
            ("rpa.casimir_polder_change", "hartree"),
            ("rpa.casimir_polder_residual", "hartree"),
            ("rpa.casimir_polder_uncoupled", "hartree"),
            ("rpa.casimir_polder_uncoupled_residual", "hartree"),
            ("charge_transfer.induction_dimer_basis", "hartree"),
            ("charge_transfer.energy", "hartree"),
            ("partition.electrostatic", "hartree"),
            ("partition.induction_a", "hartree"),
            ("partition.induction_b", "hartree"),
            ("partition.residual", "hartree"),
        ]),
    ))
}
