// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Real third-order XC evaluation on the default and scoped large stacks.

use dft_rs::resources::{current_usage, ResourceLimits};
use dft_rs::xc::{resolve, Family, FunctionalSpec, GridVars, XcOut};

#[test]
fn third_order_xc_values_are_unchanged_with_one_and_three_workers() {
    let rho_a = [0.31, 0.73];
    let rho_b = [0.27, 0.49];
    let sigma_aa = [0.043, 0.082];
    let sigma_ab = [0.011, -0.017];
    let sigma_bb = [0.032, 0.064];
    let tau_a = [0.24, 0.58];
    let tau_b = [0.19, 0.43];
    for name in ["pbe", "tpss"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: Some([&sigma_aa, &sigma_ab, &sigma_bb]),
            tau: (functional.family == Family::MetaGga).then_some([&tau_a[..], &tau_b[..]]),
        };
        let evaluate = || {
            let mut output = XcOut::default();
            functional.eval(&vars, 3, &mut output)?;
            Ok(format!("{output:?}"))
        };
        let expected: dft_rs::Result<String> = evaluate();
        let expected = expected.unwrap();
        for threads in [1, 3] {
            ResourceLimits::new(Some(threads), Some(4))
                .unwrap()
                .run(|| {
                    assert_eq!(current_usage().unwrap().cpu_threads, threads);
                    assert_eq!(rayon::current_num_threads(), threads);
                    assert_eq!(evaluate()?, expected, "{name}, threads={threads}");
                    Ok(())
                })
                .unwrap();
        }
        println!("{name}: every order-zero/one/two/three output is unchanged on scoped one/three-worker stacks");
    }
}
