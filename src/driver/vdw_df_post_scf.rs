// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Restricted molecular total energy with a fixed-density vdW-DF XC term.

use super::method::MolecularSetup;
use super::{Controls, Value};
use crate::linalg::Matrix;
use crate::nlc::vdw_df::{MolecularPairGrid, PairKernelQuadrature, PairKernelTable, Variant};
use crate::scf::ScfResult;
use crate::{DftError, Result};

#[derive(Clone, Copy)]
pub(super) struct Quadrature {
    radial_order: usize,
    angular_degree: usize,
    table_points: usize,
    momentum_cutoff: f64,
}

impl Quadrature {
    pub(super) fn from_controls(controls: &Controls) -> Result<Self> {
        let quadrature = Self {
            radial_order: controls.usize("vdw_df_radial_order", 32)?,
            angular_degree: controls.usize("vdw_df_angular_degree", 17)?,
            table_points: controls.usize("vdw_df_table_points", 96)?,
            momentum_cutoff: controls.f64("vdw_df_momentum_cutoff", 72.0)?,
        };
        if !(2..=256).contains(&quadrature.radial_order)
            || !(3..=131).contains(&quadrature.angular_degree)
            || !(4..=384).contains(&quadrature.table_points)
            || !(12.0..=2000.0).contains(&quadrature.momentum_cutoff)
        {
            return Err(DftError::InvalidInput(
                "vdW-DF needs radial order 2..=256, angular degree 3..=131, 4..=384 interpolation points and a momentum cutoff of 12..=2000".into(),
            ));
        }
        Ok(quadrature)
    }
}

pub(super) fn check_parent(variant: Variant, setup: &MolecularSetup) -> Result<()> {
    if setup.method.multiplicity != 1
        || setup.method.unrestricted
        || setup.method.electronic_temperature > 0.0
        || setup.method.occupation.is_some()
        || setup.method.fixed_occupations.is_some()
        || setup.functional.hybrid != crate::xc::HybridParams::PURE
        || setup.options.skala.is_some()
    {
        return Err(DftError::Unsupported {
            feature: format!("{} fixed-density XC energy", variant.name()),
            missing: "a ground-state unpolarized semilocal RKS density with integer occupations"
                .into(),
        });
    }
    Ok(())
}

pub(super) fn evaluate(
    variant: Variant,
    settings: Quadrature,
    setup: &MolecularSetup,
    scf: &ScfResult,
    density: &Matrix,
    report: &mut Value,
) -> Result<()> {
    if scf.unrestricted || scf.n_alpha != scf.n_beta || !scf.has_integer_occupations() {
        return Err(DftError::Unsupported {
            feature: format!("{} fixed-density XC energy", variant.name()),
            missing: "a converged closed-shell molecular RKS density".into(),
        });
    }
    let spatial =
        MolecularPairGrid::new(settings.radial_order, settings.angular_degree, 1.1, 1e-8)?;
    let direct = PairKernelQuadrature::new(settings.momentum_cutoff, 8, 0.004)?;
    let table = PairKernelTable::new(&direct, 1e-5, 1e6, settings.table_points)?;
    let kernel = |d, dp| {
        table
            .interpolate(d, dp)
            .map_or_else(|| direct.evaluate(d, dp), Ok)
    };
    let parts = spatial.xc_on_density(
        setup.integration_grid()?.as_ref(),
        &setup.basis,
        density,
        variant,
        kernel,
    )?;
    // Remove the parent XC from its stationary total and add the complete
    // new fixed-density XC. Keep the other energy components at that density.
    let delta = parts.total() - scf.e_xc;
    let parent_energy = report
        .get("energy")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            DftError::InvalidInput("vdW-DF needs an ordinary molecular reference energy".into())
        })?;
    report.set("energy", parent_energy + delta);
    if let Some(mut components) = report.get("energy_components").cloned() {
        components.set("electronic", scf.e_elec + delta);
        components.set(
            "exchange_correlation",
            parts.exchange + parts.local_correlation,
        );
        components.set("nonlocal_correlation", parts.nonlocal_correlation);
        report.set("energy_components", components);
    }
    if let Some(parent) = report.get("functional").cloned() {
        report.set("vdw_df_parent_functional", parent);
    }
    report.set("functional", variant.name());
    report.set(
        "model",
        "molecular vdW-DF total energy evaluated on a converged semilocal RKS density",
    );
    report.set("vdw_df_parent_energy", parent_energy);
    report.set("vdw_df_parent_xc_energy", scf.e_xc);
    report.set("vdw_df_exchange_energy", parts.exchange);
    report.set("vdw_df_local_correlation_energy", parts.local_correlation);
    report.set(
        "vdw_df_nonlocal_correlation_energy",
        parts.nonlocal_correlation,
    );
    report.set("vdw_df_xc_energy", parts.total());
    report.set("vdw_df_radial_order", settings.radial_order);
    report.set("vdw_df_angular_degree", settings.angular_degree);
    report.set("vdw_df_table_points", settings.table_points);
    report.set("vdw_df_momentum_cutoff", settings.momentum_cutoff);
    Ok(())
}
