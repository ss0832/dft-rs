// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A geometric C1 certificate is distinct from a derivative acceptance run.
//! Audit the actual shared high-order nuclear fixtures and reject planar
//! water or a rotated linear molecule as C1 evidence.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/correlated_fixture.rs"]
mod correlated_fixture;

#[test]
fn shared_high_order_hydrogen_fixtures_are_nonplanar_and_have_no_atom_permutation() {
    for unrestricted in [false, true] {
        let (system, controls) = correlated_fixture::fixture("hf", unrestricted, "none");
        assert_eq!(
            controls
                .get("grid_weight_cutoff")
                .and_then(dft_rs::driver::Value::as_f64),
            Some(0.0)
        );
        c1_geometry::assert_c1(&system.numbers, &system.positions);
    }
}

#[test]
fn common_orbital_ammonia_geometry_with_p_orbitals_is_c1() {
    // The p249 common-orbital response fixture, including nitrogen valence p AOs.
    c1_geometry::assert_c1(
        &[7, 1, 1, 1],
        &[
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.20, -1.31, -0.61],
        ],
    );
}

#[test]
#[should_panic(expected = "four noncoplanar centres")]
fn three_centre_water_is_not_a_c1_fixture() {
    c1_geometry::assert_c1(
        &[8, 1, 1],
        &[
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
        ],
    );
}

#[test]
#[should_panic(expected = "coplanar centres")]
fn distorted_planar_four_centre_fixture_is_not_c1() {
    c1_geometry::assert_c1(
        &[1; 4],
        &[
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
            [0.1, 1.7, 0.0],
            [1.4, 2.3, 0.0],
        ],
    );
}
