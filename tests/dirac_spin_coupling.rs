// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All mixed Cartesian indirect spin-spin couplings against independently
//! reconverged 4c SCF energies with finite nuclear magnetic moments.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;
use stencils::Stencil;

const COORDINATES: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.77, 0.36, -0.08],
    [2.44, 2.35, 1.2],
    [3.97, 2.88, 1.67],
];
const SITES: [usize; 2] = [0, 2];

fn options(xc: &str) -> Value {
    Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("speed_of_light", 6.8)
        .with("nuclear_rms_radii", [0.09, 0.083, 0.075, 0.088])
        .with("nuclear_magnetic_rms_radii", [0.08, 0.10, 0.115, 0.094])
        .with("cpu_threads", 1_usize)
        .with("grid_level", 0_usize)
        .with("grid_weight_cutoff", 0.)
        .with("conv_tol", 2e-14)
        .with("conv_tol_grad", 5e-12)
        .with("conv_tol_dp", 5e-12)
}

#[test]
fn c1_hydrogen_dimer_has_all_nine_independent_nuclear_coupling_fits() {
    c1_geometry::assert_c1(&[1; 4], &COORDINATES);
    let molecule = System::molecule(vec![1; 4], COORDINATES.to_vec());
    let params = options("hf")
        .with("relativistic_method", "dirac")
        .with("nuclei", SITES)
        .with("response_tolerance", 1e-11);
    let report = run(
        "spin-spin-couplings",
        &molecule,
        &Controls::from_value(&params).unwrap(),
    )
    .unwrap();
    let number = |key: &str| report.get(key).and_then(Value::as_f64).unwrap();
    let values = |key: &str| report.get(key).unwrap().numeric_array().unwrap().1;
    let coupling = values("reduced_couplings");
    let explicit = values("coupling_explicit_sector_tensors");
    let orbital = values("coupling_orbital_relaxation_tensors");
    assert_eq!(coupling.len(), 9);
    assert!(number("response_residual") < 1e-10);
    assert!(number("coupling_reference_fock_defect") < 1e-11);
    assert!(number("coupling_source_reciprocity_defect") < 1e-11);
    assert!(number("coupling_sector_metric_defect") < 1e-10);
    assert!(number("coupling_sector_operator_residual") < 1e-8);
    for i in 0..9 {
        assert!((coupling[i] - explicit[i] - orbital[i]).abs() < 1e-13);
    }
    println!("C1 4c indirect coupling {coupling:?}, explicit {explicit:?}, orbital {orbital:?}; core residuals {:.6e} {:.6e}; coupled residual {:.6e}",
             number("coupling_sector_metric_defect"),number("coupling_sector_operator_residual"),
             number("response_residual"));
    let cache = RefCell::new(BTreeMap::new());
    let oracle = |a, b, h: f64, k: f64| {
        let mut cache = cache.borrow_mut();
        *cache
            .entry((a, b, h.to_bits(), k.to_bits()))
            .or_insert_with(|| {
                let mut moments = [[0.; 3]; 4];
                moments[SITES[0]][a] = h;
                moments[SITES[1]][b] = k;
                let params = options("hf").with("nuclear_magnetic_moments", moments);
                let report = run(
                    "dirac-dks",
                    &molecule,
                    &Controls::from_value(&params).unwrap(),
                )
                .unwrap();
                assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
                report.get("energy").and_then(Value::as_f64).unwrap()
            })
    };
    let central = |i: usize, h: f64| {
        let (a, b) = (i / 3, i % 3);
        let first = |u| Stencil::Second.first(oracle(a, b, u, h), oracle(a, b, u, -h), 0., 0., h);
        Stencil::Second.first(first(h), first(-h), 0., 0., h)
    };
    let estimate = |i, h, stencil| {
        let base = central(i, h);
        if stencil == Stencil::Fourth {
            (4. * base - central(i, 2. * h)) / 3.
        } else {
            base
        }
    };
    // At 0.06 the residual fourth-order truncation exceeds the absolute
    // gate. Keep the coarse errors visible and resolve this contribution
    // with the same stencil at a finer step.
    let steps = [0.06, 0.035];
    let absolute = steps.map(|h| {
        coupling
            .iter()
            .enumerate()
            .map(|(i, e)| (estimate(i, h, Stencil::Fourth) - e).abs())
            .fold(0.0, f64::max)
    });
    println!("C1 4c indirect coupling fourth-order absolute defects {absolute:?} at {steps:?}");
    assert!(absolute[1] < 2e-9);
    let names: Vec<_> = (0..9)
        .map(|i| format!("four-component pair [0,2] {} {}", i / 3, i % 3))
        .collect();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.32, 0.24, 0.16, 0.12], Stencil::Second),
            ([0.20, 0.15, 0.11, 0.075], Stencil::Second),
            ([0.44, 0.32, 0.23, 0.17], Stencil::Fourth),
        ],
        |steps, stencil| {
            coupling
                .iter()
                .enumerate()
                .map(|(i, value)| steps.map(|h| (estimate(i, h, stencil) - value).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("four-component nuclear spin-spin coupling", &failures);
}
