// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete two-site FC moment Hessian, including transverse spin response,
//! in parent/PT2/total channels on a gapped C1 complex odd-electron reference.
//! All 108 entries are fitted independently. The same reference, packed
//! transform and independent finite-SCF pair oracle are shared with p296.
//! This does not certify zero-field orientational constraints or SD/PSO/DSO.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/spinor_pair_fixture.rs"]
mod spinor_pair_fixture;

#[test]
fn complete_open_shell_contact_pair_hessian_includes_every_spin_axis() {
    spinor_pair_fixture::fit(
        "C1 open-shell PT2 two-site FC moment Hessian",
        |molecule, basis| {
            let mut sources = dft_rs::response::spin_spin::spinor_nuclear_magnetic_sources(
                molecule,
                basis,
                &[0, 1],
                true,
            )
            .unwrap();
            // lambda=alpha^2*m conditions the source. Physical nuclear-moment
            // Hessians equal alpha^4 times these source-coordinate Hessians.
            let scale = 1.0 / dft_rs::constants::FINE_STRUCTURE.powi(2);
            for source in &mut sources {
                for value in source
                    .re
                    .as_mut_slice()
                    .iter_mut()
                    .chain(source.im.as_mut_slice())
                {
                    *value *= scale;
                }
            }
            assert_eq!(sources.len(), 6);
            sources
        },
        &[
            [0.002, 0.0015, 0.001, 0.00075],
            [0.0005, 0.000375, 0.00025, 0.0001875],
            [0.008, 0.006, 0.004, 0.003],
            [0.000125, 0.00009375, 0.0000625, 0.000046875],
        ],
    );
}
