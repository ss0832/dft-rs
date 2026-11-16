// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Transverse spin excitations (magnon-type spin-flip states) of an
//! unrestricted reference, routed through the common driver.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::response::spin_flip::spin_flip_tda;
use crate::{DftError, Result};

pub(super) fn magnon(system: &System, controls: &Controls) -> Result<Value> {
    let nstates = controls.usize("nstates", 5)?;
    let kernel_name = controls.str("spin_flip_kernel", "noncollinear")?;
    let setup = setup(system, controls)?;
    controls.finish("magnon")?;
    let noncollinear = match kernel_name.as_str() {
        "noncollinear" => true,
        "collinear" => false,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown spin_flip_kernel `{other}` (noncollinear or collinear)"
            )))
        }
    };
    if nstates == 0 {
        return Err(DftError::InvalidInput("nstates must be at least 1".into()));
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if scf.electronic_temperature > 0.0 {
        return Err(DftError::Unsupported {
            feature: "spin-flip excitations of a reference with fractional occupations".into(),
            missing: "the occupation response of the transverse spin channel".into(),
        });
    }
    let lowering = spin_flip_tda(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        &setup.options,
        setup.grid.as_ref(),
        scf,
        true,
        nstates,
        noncollinear,
    )?;
    let raising = spin_flip_tda(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        &setup.options,
        setup.grid.as_ref(),
        scf,
        false,
        nstates,
        noncollinear,
    )?;
    let exchange = setup.functional.hybrid.cx != 0.0
        || setup.functional.hybrid.clr != 0.0
        || (noncollinear && !setup.functional.parts.is_empty());
    let mut report = reference_report(
        &setup,
        &reference,
        "transverse spin excitations: spin-flip Tamm-Dancoff states of the collinear reference \
         with exact exchange and the non-collinear kernel (spin_flip_kernel)",
    )
    .with("spin_flip_kernel", kernel_name)
    .with("spin_lowering_energies", lowering.energies.clone())
    .with("spin_raising_energies", raising.energies.clone())
    .with(
        "goldstone_energy",
        if scf.n_alpha > scf.n_beta {
            lowering.energies.first().copied()
        } else {
            None
        },
    )
    .with(
        "spin_flip_asymmetry",
        lowering.asymmetry.max(raising.asymmetry),
    )
    .with("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            (
                "spin_lowering_energies",
                "hartree, alpha to beta flips (M_S - 1), ascending",
            ),
            (
                "spin_raising_energies",
                "hartree, beta to alpha flips (M_S + 1), ascending",
            ),
            (
                "goldstone_energy",
                "hartree, lowest spin-lowering energy of a high-spin reference (zero for exact exchange)",
            ),
            ("spin_flip_asymmetry", "hartree"),
        ]),
    );
    if !exchange {
        if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
            let mut all = existing;
            all.push(Value::Str(
                "WARNING [magnon] no exact exchange and no non-collinear kernel: nothing couples \
                 the spin flips and the energies are orbital-energy differences"
                    .into(),
            ));
            report.set("warnings", Value::Array(all));
        }
    }
    Ok(report)
}
