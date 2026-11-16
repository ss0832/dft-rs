// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete C1 ROHF and complex GHF Raman/hyper-Raman/gamma acceptance.
//! Every nuclear coordinate and independent electric component is fitted
//! against separately converged displaced references and lower-order purely
//! electric energy response. No tensor value is copied in the oracle.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_directions.rs"]
mod symmetric_directions;
use dft_rs::driver::method::{
    prepare_molecular, run_molecular_reference, MolecularMethod, MolecularSetup,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::{
    spinor_scf::{
        run_spinor_scf, BiasedSpinorBackend, SpinorEnergyBackend, SpinorScfConfig, SpinorScfReport,
    },
    BlochMatrix,
};
use dft_rs::response::nuclear::{
    open_shell_molecular_energy_taylor, MolecularDirection, SpinorNuclearContext,
};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::ScfResult;
use std::collections::BTreeMap;
const BIAS: [f64; 3] = [0.003, -0.004, 0.005];
fn molecule() -> System {
    System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    )
}
fn options(spinor: bool, field: [f64; 3]) -> Value {
    let mut value = Value::object()
        .with("reference", if spinor { "gks" } else { "roks" })
        .with(
            "xc",
            std::env::var("DFT_RS_TEST_XC").unwrap_or_else(|_| "hf".into()),
        )
        .with("basis", "sto-3g")
        .with("charge", 1)
        .with("multiplicity", 2)
        .with("grid_level", 0)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-13)
        .with("response_tolerance", 1e-12)
        .with("max_iter", 600)
        .with("guess", "core")
        .with("dispersion", "none")
        .with("reference_field", field);
    if spinor {
        let source = spinor_operator::source(4);
        value = value
            .with("gks_operator_real", source.re)
            .with("gks_operator_imaginary", source.im)
            .with("spinor_xc_tolerance", 1e-12);
    }
    value
}
enum Solution {
    Roks(Box<ScfResult>, Matrix),
    Spinor(Box<SpinorScfReport>, BlochMatrix),
}
struct IndependentReference {
    setup: MolecularSetup,
    state: Solution,
}
impl IndependentReference {
    fn new(system: &System, spinor: bool, field: [f64; 3]) -> Self {
        Self::continuation(system, spinor, field, None)
    }
    fn continuation(system: &System, spinor: bool, field: [f64; 3], origin: Option<&Self>) -> Self {
        let value = options(spinor, field);
        let method =
            MolecularMethod::from_controls(&Controls::from_value(&value).unwrap()).unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        let state = if spinor {
            let seed = run_molecular_reference(&setup).unwrap();
            let operator = spinor_operator::source(4);
            let backend = MolecularSpinorBackend::new(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &seed.scf,
                &setup.options,
                operator.clone(),
                [0.0, 0.0, 1.0],
                MolecularSpinorFunctionalModel::SpinCovariant,
                1e-12,
            )
            .unwrap();
            // A fresh physical SCF follows the reference stationary branch.
            // The old Fock supplies only a starting matrix, never a source,
            // energy contribution or analytic derivative in the oracle.
            let initial = origin.and_then(|o| match &o.state {
                Solution::Spinor(s, _) => Some(s.fields.fock.clone()),
                _ => None,
            });
            let seeded = BiasedSpinorBackend::new(&backend, None, initial).unwrap();
            let scf = run_spinor_scf(
                &seeded,
                &SpinorScfConfig {
                    max_iter: 600,
                    energy_tolerance: 1e-14,
                    density_tolerance: 1e-13,
                    orbital_tolerance: 1e-12,
                    diis_size: setup.options.diis_size,
                    overlap_cutoff: setup.options.orth_threshold,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                scf.converged,
                "independent spinor reference: {}",
                scf.orbital_residual
            );
            assert!(
                scf.density[0].im.as_slice().iter().any(|v| v.abs() > 1e-5),
                "complex coherence must be active"
            );
            if origin.is_some() {
                use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
                let initial = generalized_orbitals(
                    &seeded.initial_fock()[0],
                    &seeded.overlap()[0],
                    setup.options.orth_threshold,
                )
                .unwrap();
                let projected = matmul(
                    &matmul(
                        &adjoint(&initial.coefficients).unwrap(),
                        &seeded.overlap()[0],
                    )
                    .unwrap(),
                    &scf.bands[0].coefficients,
                )
                .unwrap();
                let occupied = scf.occupations[0].iter().filter(|v| **v > 0.5).count();
                let mut m = BlochMatrix::zeros(occupied, occupied);
                for i in 0..occupied {
                    for j in 0..occupied {
                        m.re[(i, j)] = projected.re[(i, j)];
                        m.im[(i, j)] = projected.im[(i, j)];
                    }
                }
                let gram = matmul(&adjoint(&m).unwrap(), &m).unwrap();
                let real = dft_rs::periodic::integrals::hermitian_as_real(&gram.re, &gram.im);
                let (values, _) = dft_rs::linalg::symmetric_eigen(&real).unwrap();
                println!("independent spinor occupied projector overlap={:.15e} energy={:.16e} residual={:.15e}",values[0],scf.fields.energy,scf.orbital_residual);
                assert!(
                    values[0] > 0.90 && values.iter().all(|v| *v <= 1.0 + 1e-10),
                    "independent SCF changed the stationary branch: {values:?}"
                );
            }
            Solution::Spinor(Box::new(scf), operator)
        } else {
            let (scf, c) = dft_rs::scf::roks::run_roks_with_orbitals(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
            )
            .unwrap();
            assert!(scf.converged);
            Solution::Roks(Box::new(scf), c)
        };
        Self { setup, state }
    }
    fn electric_coefficient(&self, direction: [f64; 3], order: usize) -> f64 {
        let d = MolecularDirection {
            nuclear: vec![[0.0; 3]; 4],
            electric: direction,
        };
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let s = &self.setup;
        let energy = match &self.state {
            Solution::Roks(scf, c) => {
                open_shell_molecular_energy_taylor(
                    &s.molecule,
                    &s.basis,
                    &s.functional,
                    s.grid.as_ref(),
                    scf,
                    c,
                    &s.options,
                    &d,
                    order,
                    &config,
                )
                .unwrap()
                .energy
            }
            Solution::Spinor(scf, operator) => {
                SpinorNuclearContext::new(
                    &s.molecule,
                    &s.basis,
                    s.grid.as_ref(),
                    &s.functional,
                    scf,
                    &s.options,
                    operator,
                    order,
                    &config,
                )
                .unwrap()
                .molecular_energy_taylor(&d, order)
                .unwrap()
                .energy
            }
        };
        -energy[order] * (1..=order).product::<usize>() as f64
    }
}
fn contract(tensor: &[f64], rank: usize, direction: [f64; 3]) -> f64 {
    assert_eq!(tensor.len(), 3usize.pow(rank as u32));
    tensor
        .iter()
        .enumerate()
        .map(|(mut flat, value)| {
            let mut weight = 1.0;
            for _ in 0..rank {
                weight *= direction[flat % 3];
                flat /= 3;
            }
            value * weight
        })
        .sum()
}
fn tensor(feature: &str, system: &System, options: &Value) -> Vec<f64> {
    let report = run(feature, system, &Controls::from_value(options).unwrap()).unwrap();
    assert_eq!(
        report.get("converged").and_then(Value::as_bool),
        Some(true),
        "{report:?}"
    );
    for key in [
        "response_residuals",
        "projector_defects",
        "nuclear_stationarity_defects",
        "reference_defects",
    ] {
        let (_, values) = report.get(key).unwrap().numeric_array().unwrap();
        let max = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        println!("{feature} {key}={max:.15e}");
        assert!(max <= 1e-12, "{key}: {max}");
    }
    let values = report.get("tensor").unwrap().numeric_array().unwrap().1;
    println!("{feature} complete ordered tensor {values:?}");
    values
}
fn fit(spinor: bool) {
    std::env::set_var("DFT_RS_STRICT_UNITS", "1");
    let system = molecule();
    let opts = options(spinor, BIAS);
    let origin = IndependentReference::new(&system, spinor, BIAS);
    let label = if spinor { "complex GKS" } else { "ROKS" };
    let selected = std::env::var("DFT_RS_TEST_COORDINATE")
        .ok()
        .map(|v| v.parse::<usize>().unwrap());
    let gamma = std::env::var("DFT_RS_TEST_GAMMA_ONLY").is_ok();
    let mut failures = Vec::new();
    let mut ordered: [Vec<Vec<f64>>; 2] = std::array::from_fn(|_| vec![Vec::new(); 12]);
    if !gamma {
        let stages = [
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
            // Strong open-shell curvature can keep the original intervals
            // outside the quadratic regime. Retain those failures and
            // resolve them with complete finer four-step intervals.
            ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
            (
                [0.005, 0.0025, 0.00125, 0.000625],
                stencils::Stencil::Second,
            ),
            (
                [0.0025, 0.00125, 0.000625, 0.0003125],
                stencils::Stencil::Second,
            ),
        ];
        let directions: Vec<_> = [2, 3]
            .into_iter()
            .map(symmetric_directions::directions)
            .collect();
        for coordinate in 0..12 {
            if selected.is_some_and(|c| c != coordinate) {
                continue;
            }
            let mut nuclear = vec![[0.0; 3]; 4];
            nuclear[coordinate / 3][coordinate % 3] = 1.0;
            let exact: Vec<_> = [2, 3]
                .into_iter()
                .enumerate()
                .flat_map(|(index, rank)| {
                    let values = tensor(
                        "static-spectroscopy",
                        &system,
                        &opts
                            .clone()
                            .with("nuclear_direction", &nuclear)
                            .with("electric_order", rank),
                    );
                    ordered[index][coordinate] = values.clone();
                    directions[index]
                        .iter()
                        .map(move |&d| contract(&values, rank, d))
                        .collect::<Vec<_>>()
                })
                .collect();
            let names: Vec<_> = [(2, 6), (3, 10)]
                .into_iter()
                .flat_map(|(rank, count)| {
                    (0..count).map(move |electric| {
                        format!("{label} rank={rank} nuclear={coordinate} electric={electric}")
                    })
                })
                .collect();
            let mut cache = BTreeMap::<u64, Vec<f64>>::new();
            stencils::fit_components_staged(
                &names,
                &stages,
                |steps, stencil| {
                    let mut sample = |offset: f64| {
                        cache
                            .entry(offset.to_bits())
                            .or_insert_with(|| {
                                let mut moved = system.clone();
                                moved.positions[coordinate / 3][coordinate % 3] += offset;
                                let state = IndependentReference::continuation(
                                    &moved,
                                    spinor,
                                    BIAS,
                                    Some(&origin),
                                );
                                [2, 3]
                                    .into_iter()
                                    .enumerate()
                                    .flat_map(|(index, rank)| {
                                        directions[index]
                                            .iter()
                                            .map(|&d| state.electric_coefficient(d, rank))
                                            .collect::<Vec<_>>()
                                    })
                                    .collect()
                            })
                            .clone()
                    };
                    let mut errors = vec![[0.0; 4]; 16];
                    for (k, &h) in steps.iter().enumerate() {
                        let plus = sample(h);
                        let minus = sample(-h);
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (sample(2.0 * h), sample(-2.0 * h))
                        } else {
                            (vec![0.0; 16], vec![0.0; 16])
                        };
                        for i in 0..16 {
                            errors[i][k] = (stencil.first(plus[i], minus[i], p2[i], m2[i], h)
                                - exact[i])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    if !gamma && selected.is_none() {
        for (index, rank) in [2, 3].into_iter().enumerate() {
            let mut largest = 0.0_f64;
            for axis in 0..3 {
                for slot in 0..3_usize.pow(rank) {
                    let sum: f64 = (0..4)
                        .map(|atom| ordered[index][3 * atom + axis][slot])
                        .sum();
                    largest = largest.max(sum.abs());
                }
            }
            println!("{label} rank={rank} complete translation defect={largest:.15e}");
            assert!(largest < 1e-9, "rank {rank} translation sum: {largest}");
        }
    }
    if selected.is_none() || gamma {
        for rank in [3, 4] {
            let feature = if rank == 3 { "beta" } else { "gamma" };
            let values = tensor(feature, &system, &opts);
            let directions = symmetric_directions::directions(rank);
            let exact: Vec<_> = directions
                .iter()
                .map(|&d| contract(&values, rank, d))
                .collect();
            let names: Vec<_> = (0..directions.len())
                .map(|i| format!("{label} {feature} electric={i}"))
                .collect();
            let stages = [
                ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
                ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Second),
                ([0.008, 0.006, 0.004, 0.003], stencils::Stencil::Second),
                ([0.008, 0.006, 0.004, 0.003], stencils::Stencil::Fourth),
                (
                    [0.001, 0.00075, 0.0005, 0.000375],
                    stencils::Stencil::Second,
                ),
                (
                    [0.0005, 0.000375, 0.00025, 0.0001875],
                    stencils::Stencil::Second,
                ),
            ];
            let mut cache = BTreeMap::new();
            stencils::fit_components_staged(
                &names,
                &stages,
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; directions.len()];
                    for (i, &d) in directions.iter().enumerate() {
                        for (k, &h) in steps.iter().enumerate() {
                            let mut sample = |offset: f64| {
                                *cache.entry((i, offset.to_bits())).or_insert_with(|| {
                                    let field = std::array::from_fn(|a| BIAS[a] + offset * d[a]);
                                    IndependentReference::continuation(
                                        &system,
                                        spinor,
                                        field,
                                        Some(&origin),
                                    )
                                    .electric_coefficient(d, rank - 1)
                                })
                            };
                            let p = sample(h);
                            let m = sample(-h);
                            let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                (sample(2.0 * h), sample(-2.0 * h))
                            } else {
                                (0.0, 0.0)
                            };
                            errors[i][k] = (stencil.first(p, m, p2, m2, h) - exact[i]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures(
        "complete C1 reference-constrained electric tensors",
        &failures,
    );
}
#[test]
fn all_roks_raman_hyper_raman_and_gamma_components() {
    fit(false);
}
#[test]
fn all_complex_gks_raman_hyper_raman_and_gamma_components() {
    fit(true);
}
