// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Full-exchange SAPT2+3 on a water dimer under a fixed resource budget.

use dft_rs::basis::data;
use dft_rs::math::Vec3;
use dft_rs::resources::{current_usage, ResourceLimits, MIB};
use dft_rs::sapt::{sapt_interaction_variant, SaptConfig, SaptVariant};
use dft_rs::scf::{EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;

fn water(points: [[f64; 3]; 3]) -> Molecule {
    Molecule::new(
        [8, 1, 1]
            .into_iter()
            .zip(points)
            .map(|(z, [x, y, z_pos])| Atom {
                z,
                position: Vec3::new(x, y, z_pos),
            })
            .collect(),
    )
}

#[test]
fn water_sapt2_plus3_fits_the_managed_budget_with_all_terms() {
    let a = water([[0.0, 0.0, 0.0], [0.04, 0.10, 1.81], [1.72, 0.10, -0.45]]);
    let b = water([[0.12, -0.24, 5.35], [1.45, 0.66, 6.20], [-1.00, 1.04, 6.01]]);
    let budget = 128;
    let opts = ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-11,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: budget },
        ..ScfOptions::default()
    };
    let basis = data::load("sto-3g").unwrap();
    let hf = ResolvedFunctional::hartree_fock();
    let (result, peak) = ResourceLimits::new(Some(1), Some(budget))
        .unwrap()
        .run(|| {
            let result = sapt_interaction_variant(
                &a,
                &b,
                &basis,
                &hf,
                &opts,
                None,
                &SaptConfig::default(),
                SaptVariant::Sapt2Plus3,
            )?;
            Ok((result, current_usage().unwrap().peak_reserved_bytes))
        })
        .unwrap();
    let full = result.sapt2_plus3.unwrap();
    let values = [
        result.interaction_total(),
        full.correlated_electrostatics,
        full.correlated_exchange,
        full.correlated_induction,
        full.correlated_dispersion,
        full.cubic_dispersion_and_mixed,
        full.relaxed_exchange_induction30,
        full.delta_hf_order3,
    ];
    // Full orientation table for both spin vertices with explicit
    // dense monomer projector products and the same exact Coulomb ERIs.
    let reference = [
        1.2164009737768004e-3,
        4.1995198985432893e-4,
        4.7231296376542225e-4,
        3.24225041393614e-5,
        4.024621711626775e-5,
        4.180102597259731e-6,
        -1.7113045801788719e-6,
        1.7770975251763328e-3,
    ];
    let defect = values
        .into_iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(peak <= budget * MIB);
    assert!(defect < 5e-13);
    println!("Water STO-3G SAPT2+3 {:+.17e} Eh, maximum term difference {defect:.3e} Eh, managed peak {peak} bytes", result.interaction_total());
}
