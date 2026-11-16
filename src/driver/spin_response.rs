// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static spin susceptibility of a molecular Fermi-Dirac (Mermin) ensemble,
//! routed through the common driver.
//!
//! Moving `delta` electrons from the beta to the alpha channel at fixed
//! total number costs `eta delta^2 / 2` with the spin stiffness
//! `eta = d(mu_alpha - mu_beta)/d(delta)` of the ensemble response, which
//! holds the full Hartree, exchange and exchange-correlation kernel. In a
//! field `B` the Zeeman energy is `g_e mu_B B delta`, so the induced moment
//! is `chi B` with
//!
//! ```text
//! chi = g_e^2 mu_B^2 / eta        (mu_B = 1/2 in atomic units)
//! ```
//!
//! The orbital (Van Vleck and diamagnetic) parts are those of the
//! magnetizability and are not included.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::vibrational::response_config;
use super::System;
use crate::constants::ELECTRON_G_FACTOR;
use crate::response::ensemble::EnsembleSolver;
use crate::{DftError, Result};

/// One atomic unit of magnetizability, `e^2 a0^2 / m_e`, as a molar
/// susceptibility in 1e-6 cm^3/mol.
const ATOMIC_UNIT_IN_CGS_PPM: f64 = 4.7520933;

pub(super) fn spin_susceptibility(system: &System, controls: &Controls) -> Result<Value> {
    let config = response_config(controls)?;
    let g = controls.f64("spin_g_factor", ELECTRON_G_FACTOR)?;
    let setup = setup(system, controls)?;
    controls.finish("spin-susceptibility")?;
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if !scf.unrestricted {
        return Err(DftError::InvalidInput(
            "spin-susceptibility needs unrestricted = true: the two spin channels exchange \
             electrons"
                .into(),
        ));
    }
    let solver = EnsembleSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
    )?;
    let response = solver.spin_susceptibility(g, &config)?;
    let susceptibility = response.susceptibility;
    // A Mermin density is not idempotent. Measure its physical stationarity
    // directly instead of applying integer-occupation projector diagnostics.
    let orbital_residual = (0..2)
        .map(|s| {
            let fps = scf.fock[s]
                .matmul_seq(&scf.density[s])
                .matmul_seq(&scf.overlap);
            let spf = scf
                .overlap
                .matmul_seq(&scf.density[s])
                .matmul_seq(&scf.fock[s]);
            fps.max_abs_diff(&spf)
        })
        .fold(0.0_f64, f64::max);
    Ok(reference_report(
        &setup,
        &reference,
        "static spin susceptibility of the Fermi-Dirac ensemble from the response to a spin \
         transfer at fixed electron number",
    )
    .with("spin_susceptibility", susceptibility)
    .with(
        "spin_susceptibility_cgs_ppm",
        susceptibility * ATOMIC_UNIT_IN_CGS_PPM,
    )
    .with("spin_g_factor", g)
    .with("spin_stiffness", response.stiffness)
    .with("bare_spin_stiffness", response.bare_stiffness)
    .with("bare_spin_susceptibility", response.bare_susceptibility)
    .with("spin_enhancement", response.enhancement)
    .with(
        "spin_density_response",
        response
            .delta_density
            .iter()
            .map(|p| {
                (0..p.rows)
                    .map(|i| (0..p.cols).map(|j| p[(i, j)]).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
    .with("reference_orbital_residual", orbital_residual)
    .with("thermal_energy", scf.electronic_temperature)
    .with("response_residual", response.response_residual)
    .with("response_iterations", response.iterations)
    .with("number_defects", response.number_defects)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("spin_susceptibility", "e^2*bohr^2/m_e, chi = dM/dB"),
            ("bare_spin_susceptibility", "e^2*bohr^2/m_e, chi = dM/dB"),
            ("bare_spin_stiffness", "hartree"),
            ("spin_enhancement", "dimensionless"),
            ("spin_g_factor", "dimensionless"),
            ("spin_density_response", "atomic units, dP_sigma/dB"),
            ("reference_orbital_residual", "hartree"),
            ("spin_susceptibility_cgs_ppm", "1e-6 cm^3/mol"),
            (
                "spin_stiffness",
                "hartree, d(mu_alpha - mu_beta)/d(delta) for N_alpha + delta, N_beta - delta",
            ),
            ("thermal_energy", "hartree, k_B T of the reference"),
            ("response_residual", "dimensionless"),
            (
                "number_defects",
                "atomic units, electrons per atomic magnetic field",
            ),
        ]),
    ))
}
