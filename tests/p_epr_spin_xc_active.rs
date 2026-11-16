// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Every active descriptor of the variational spinor potential is fitted
//! against an independent scalar angular-energy oracle at fixed quadrature.

#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/spinor_xc_scalar_energy.rs"]
mod scalar_energy;

use dft_rs::periodic::spinor_xc::{spinor_semilocal_point, SpinorXcPoint};
use dft_rs::xc::{resolve, Family, FunctionalSpec};

#[test]
fn all_active_spinor_descriptor_gradients_fit_independent_scalar_energy() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let mut x = std::array::from_fn(|i| 0.013 * ((i * 11 % 17) as f64 - 8.0));
            x[0] = 1.3;
            x[1] = 0.31;
            x[2] = -0.27;
            x[3] = 0.19;
            x[16] = 1.1;
            x[17] = 0.12;
            x[18] = -0.08;
            x[19] = 0.11;
            for name in ["svwn5", "pbe", "tpss", "pbe0", "cam-b3lyp"] {
                let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
                let active = match f.family {
                    Family::Lda => 4,
                    Family::Gga => 16,
                    Family::MetaGga => 32,
                };
                let result =
                    spinor_semilocal_point(&f, &SpinorXcPoint::from_descriptors(x), 16).unwrap();
                assert!((result.energy - scalar_energy::energy(&f, &x, 16)).abs() < 1e-13);
                assert!(result.gradient[active..].iter().all(|&v| v == 0.0));
                let labels = (0..active)
                    .map(|i| format!("{name} fixed-angular16 descriptor{i}"))
                    .collect::<Vec<_>>();
                let mut failures = Vec::new();
                components::fit_components(
                    &labels,
                    &[
                        [0.08, 0.04, 0.02, 0.01],
                        [0.04, 0.02, 0.01, 0.005],
                        [0.16, 0.08, 0.04, 0.02],
                    ],
                    |steps| {
                        (0..active)
                            .map(|i| {
                                std::array::from_fn(|s| {
                                    let h = steps[s];
                                    let mut plus = x;
                                    let mut minus = x;
                                    plus[i] += h;
                                    minus[i] -= h;
                                    let fd = (scalar_energy::energy(&f, &plus, 16)
                                        - scalar_energy::energy(&f, &minus, 16))
                                        / (2.0 * h);
                                    (fd - result.gradient[i]).abs()
                                })
                            })
                            .collect()
                    },
                    &mut failures,
                );
                components::assert_no_failures("all active spinor descriptors", &failures);
            }
        })
        .unwrap();
}
