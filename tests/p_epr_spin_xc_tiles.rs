// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Bounded angular batches retain arbitrary pointwise providers and every
//! active LDA/GGA derivative against independent sphere moments.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::periodic::spinor_xc::{spinor_semilocal_point, SpinorXcPoint};
use dft_rs::xc::{Family, GridVars, Kind, XcFunctional, XcOut};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Quartic {
    coefficient: f64,
    family: Family,
    maximum_batch: Arc<AtomicUsize>,
}

impl XcFunctional for Quartic {
    fn name(&self) -> &'static str {
        "test quartic sphere moments"
    }
    fn kind(&self) -> Kind {
        Kind::ExchangeCorrelation
    }
    fn family(&self) -> Family {
        self.family
    }
    fn max_analytic_order(&self) -> usize {
        2
    }
    fn params(&self) -> Vec<f64> {
        vec![self.coefficient]
    }
    fn param_names(&self) -> &'static [&'static str] {
        &["coefficient"]
    }
    fn with_params(&self, parameters: &[f64]) -> dft_rs::Result<Box<dyn XcFunctional>> {
        if parameters.len() != 1 || !parameters[0].is_finite() {
            return Err(dft_rs::DftError::InvalidInput(
                "quartic provider requires one finite coefficient".into(),
            ));
        }
        Ok(Box::new(Self {
            coefficient: parameters[0],
            family: self.family,
            maximum_batch: self.maximum_batch.clone(),
        }))
    }
    fn eval_add(
        &self,
        variables: &GridVars,
        order: usize,
        weight: f64,
        output: &mut XcOut,
    ) -> dft_rs::Result<()> {
        assert!(order <= 2);
        self.maximum_batch
            .fetch_max(variables.len(), Ordering::Relaxed);
        let c = self.coefficient;
        for g in 0..variables.len() {
            let a = variables.rho[0][g];
            let b = variables.rho[1][g];
            output.e[g] += weight * (a.powi(4) + 2.0 * b.powi(4) + c * a * a * b * b);
            if order >= 1 {
                output.vrho[0][g] += weight * (4.0 * a.powi(3) + 2.0 * c * a * b * b);
                output.vrho[1][g] += weight * (8.0 * b.powi(3) + 2.0 * c * a * a * b);
            }
            if order >= 2 {
                output.v2rho2[0][g] += weight * (12.0 * a * a + 2.0 * c * b * b);
                output.v2rho2[1][g] += weight * 4.0 * c * a * b;
                output.v2rho2[2][g] += weight * (24.0 * b * b + 2.0 * c * a * a);
            }
            if self.family == Family::Gga {
                let sigma = variables.sigma.unwrap();
                let sum = sigma[0][g] + sigma[2][g];
                output.e[g] += weight * (sum * sum + 0.23 * sigma[1][g].powi(2));
                if order >= 1 {
                    output.vsigma[0][g] += weight * 2.0 * sum;
                    output.vsigma[1][g] += weight * 0.46 * sigma[1][g];
                    output.vsigma[2][g] += weight * 2.0 * sum;
                }
                if order >= 2 {
                    for index in [0, 2, 5] {
                        output.v2sigma2[index][g] += weight * 2.0;
                    }
                    output.v2sigma2[3][g] += weight * 0.46;
                }
            }
        }
        Ok(())
    }
    fn param_gradient(
        &self,
        variables: &GridVars,
        output: &mut Vec<Vec<f64>>,
    ) -> dft_rs::Result<()> {
        *output = vec![(0..variables.len())
            .map(|g| variables.rho[0][g].powi(2) * variables.rho[1][g].powi(2))
            .collect()];
        Ok(())
    }
}

/// Independent spherical moments <s^2>=|m|^2/3 and <s^4>=|m|^4/5,
/// after applying 1 + s d/ds to the explicit collinear quartic.
fn sphere_energy(x: &[f64; 4], c: f64) -> f64 {
    let magnetic_square = x[1..].iter().map(|v| v * v).sum::<f64>();
    let a = (3.0 + c) / 16.0;
    let b = (9.0 - c) / 8.0;
    a * (x[0].powi(4) + magnetic_square.powi(2)) + b * x[0] * x[0] * magnetic_square
}

#[test]
fn maximum_angular_rule_is_bounded_and_fits_every_active_generic_derivative() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let maximum_batch = Arc::new(AtomicUsize::new(0));
            let c = 0.37;
            let mut functional =
                dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("svwn5".into())).unwrap();
            functional.parts = vec![(
                1.0,
                Arc::new(Quartic {
                    coefficient: c,
                    family: Family::Lda,
                    maximum_batch: maximum_batch.clone(),
                }),
            )];
            let x = [1.7, 0.31, -0.22, 0.17];
            let point = SpinorXcPoint {
                density: x,
                gradient: [[0.0; 3]; 4],
                tau: [0.0; 4],
                current: [[0.0; 3]; 4],
            };
            let result = spinor_semilocal_point(&functional, &point, 128).unwrap();
            assert_eq!(result.directions, 32768);
            assert_eq!(maximum_batch.load(Ordering::Relaxed), 256);
            assert!((result.energy - sphere_energy(&x, c)).abs() < 5e-13);
            assert!((result.component_energies[0] - sphere_energy(&x, c)).abs() < 5e-13);
            assert!(result.gradient[4..].iter().all(|&v| v == 0.0));
            let square = x[1..].iter().map(|v| v * v).sum::<f64>();
            let a = (3.0 + c) / 16.0;
            let b = (9.0 - c) / 8.0;
            let exact: [f64; 4] = std::array::from_fn(|i| {
                if i == 0 {
                    4.0 * a * x[0].powi(3) + 2.0 * b * x[0] * square
                } else {
                    2.0 * b * x[0] * x[0] * x[i] + 4.0 * a * square * x[i]
                }
            });
            for (observed, expected) in result.gradient[..4].iter().zip(exact) {
                assert!((observed - expected).abs() < 5e-13);
            }
            let labels = (0..4)
                .map(|i| format!("generic quartic angular128 descriptor{i}"))
                .collect::<Vec<_>>();
            let mut failures = Vec::new();
            components::fit_components(
                &labels,
                &[[0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    (0..4)
                        .map(|i| {
                            std::array::from_fn(|s| {
                                let h = steps[s];
                                let mut plus = x;
                                let mut minus = x;
                                plus[i] += h;
                                minus[i] -= h;
                                ((sphere_energy(&plus, c) - sphere_energy(&minus, c)) / (2.0 * h)
                                    - result.gradient[i])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("all generic quartic descriptors", &failures);
        })
        .unwrap();
}

/// Fourth sphere moment: <(u^T K u)^2> = [(tr K)^2+2 tr(K^2)]/15.
/// The spin-scaling derivative multiplies this degree-four term by five.
fn sphere_gga_energy(x: &[f64; 16], c: f64) -> f64 {
    let u = x[4..7].iter().map(|v| v * v).sum::<f64>();
    let v = x[7..].iter().map(|v| v * v).sum::<f64>();
    let k: [[f64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| (0..3).map(|d| x[7 + 3 * a + d] * x[7 + 3 * b + d]).sum())
    });
    let w = k.iter().flatten().map(|v| v * v).sum::<f64>();
    sphere_energy(&x[..4].try_into().unwrap(), c)
        + (4.23 / 16.0) * u * u
        + (3.77 / 8.0) * u * v
        + (4.23 / 48.0) * (v * v + 2.0 * w)
}

#[test]
fn every_generic_gga_descriptor_fits_independent_fourth_sphere_moments() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let c = 0.37;
            let maximum_batch = Arc::new(AtomicUsize::new(0));
            let mut functional =
                dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("svwn5".into())).unwrap();
            functional.family = Family::Gga;
            functional.parts = vec![(
                1.0,
                Arc::new(Quartic {
                    coefficient: c,
                    family: Family::Gga,
                    maximum_batch: maximum_batch.clone(),
                }),
            )];
            let mut x: [f64; 16] = std::array::from_fn(|i| 0.017 * ((i * 7 % 19) as f64 - 8.3));
            x[..4].copy_from_slice(&[1.7, 0.31, -0.22, 0.17]);
            let mut descriptors = [0.0; 32];
            descriptors[..16].copy_from_slice(&x);
            let result = spinor_semilocal_point(
                &functional,
                &SpinorXcPoint::from_descriptors(descriptors),
                128,
            )
            .unwrap();
            assert_eq!(maximum_batch.load(Ordering::Relaxed), 256);
            assert!((result.energy - sphere_gga_energy(&x, c)).abs() < 5e-13);
            assert!((result.component_energies[0] - sphere_gga_energy(&x, c)).abs() < 5e-13);
            assert!(result.gradient[16..].iter().all(|&v| v == 0.0));
            let labels = (0..16)
                .map(|i| format!("generic quartic GGA angular128 descriptor{i}"))
                .collect::<Vec<_>>();
            let mut failures = Vec::new();
            components::fit_components(
                &labels,
                &[[0.08, 0.04, 0.02, 0.01]],
                |steps| {
                    (0..16)
                        .map(|i| {
                            std::array::from_fn(|s| {
                                let h = steps[s];
                                let mut plus = x;
                                let mut minus = x;
                                plus[i] += h;
                                minus[i] -= h;
                                ((sphere_gga_energy(&plus, c) - sphere_gga_energy(&minus, c))
                                    / (2.0 * h)
                                    - result.gradient[i])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("all generic GGA sphere-moment descriptors", &failures);
        })
        .unwrap();
}
