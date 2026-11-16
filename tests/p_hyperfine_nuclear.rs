// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Scalar differences verify every first derivative. Fresh displaced
//! order-one responses independently verify every Hessian projection.
//! Differences are validation only; all failed fits remain printed.

#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::properties::hyperfine::nuclear::{
    hyperfine_nuclear_derivatives, hyperfine_nuclear_path, HyperfineCoefficient,
    HyperfinePathDiagnostics,
};
use dft_rs::properties::hyperfine::spin_hyperfine_tensor;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, EriMode, Reference, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.17],
    [2.05, 0.25, 0.38],
    [-0.65, 1.76, -0.44],
    [-1.19, -1.48, 0.85],
];
const STEPS: [f64; 4] = [0.04, 0.02, 0.01, 0.005];
const HESSIAN_STEPS: [f64; 4] = [0.01, 0.005, 0.0025, 0.00125];

struct Case {
    mol: Molecule,
    basis: MolBasis,
    functional: ResolvedFunctional,
    grid: Option<MolecularGrid>,
    options: ScfOptions,
    scf: ScfResult,
}

fn case(name: &str, displacement: &[f64]) -> Case {
    let mol = Molecule::new(
        POSITIONS
            .iter()
            .enumerate()
            .map(|(atom, position)| {
                let p =
                    std::array::from_fn::<_, 3, _>(|a| position[a] + displacement[3 * atom + a]);
                Atom {
                    z: if atom == 0 { 6 } else { 1 },
                    position: Vec3::new(p[0], p[1], p[2]),
                }
            })
            .collect(),
    )
    .with_multiplicity(2);
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let grid = (!functional.parts.is_empty()).then(|| {
        MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 2,
                prune: false,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap()
    });
    let options = ScfOptions {
        reference: Reference::Uhf,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &basis, &options, &functional, grid.as_ref()).unwrap();
    assert!(
        scf.converged,
        "Independent displaced reference must converge"
    );
    println!(
        "reference {name} displacement={displacement:?} E={:.16e} iterations={}",
        scf.energy, scf.iterations
    );
    Case {
        mol,
        basis,
        functional,
        grid,
        options,
        scf,
    }
}

fn flatten(value: &HyperfineCoefficient) -> Vec<f64> {
    std::iter::once(value.contact)
        .chain(value.dipolar.iter().flatten().copied())
        .chain(value.total.iter().flatten().copied())
        .collect()
}

fn raw_tensor_symmetry(value: &HyperfineCoefficient) {
    let norm = value
        .dipolar
        .iter()
        .flatten()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let trace = value.dipolar[0][0] + value.dipolar[1][1] + value.dipolar[2][2];
    let total_norm = value
        .total
        .iter()
        .flatten()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(trace.abs() <= 1e-10 * norm);
    for a in 0..3 {
        for b in 0..3 {
            assert!((value.dipolar[a][b] - value.dipolar[b][a]).abs() <= 1e-10 * norm);
            let defect =
                value.total[a][b] - value.dipolar[a][b] - if a == b { value.contact } else { 0.0 };
            assert!(defect.abs() <= 1e-10 * total_norm);
        }
    }
}

fn observable(reference: &Case) -> Vec<f64> {
    let mut spin = reference.scf.density[0].clone();
    for (a, b) in spin
        .as_mut_slice()
        .iter_mut()
        .zip(reference.scf.density[1].as_slice())
    {
        *a -= b;
    }
    reference
        .mol
        .atoms
        .iter()
        .flat_map(|atom| {
            let value = spin_hyperfine_tensor(&reference.basis, &spin, atom.position, 1.0).unwrap();
            flatten(&HyperfineCoefficient {
                contact: value.contact,
                dipolar: value.dipolar,
                total: value.total,
            })
        })
        .collect()
}

fn fits(
    label: &str,
    expected: &[f64],
    measured: &[Vec<f64>; 4],
    steps: &[f64; 4],
    failures: &mut Vec<String>,
) {
    for (i, value) in expected.iter().enumerate() {
        let samples: [f64; 4] = std::array::from_fn(|step| measured[step][i]);
        let errors = std::array::from_fn(|step| (measured[step][i] - value).abs());
        let name = format!("{label} property-entry-{i}");
        println!("gradient_oracle_samples {name}: analytic={value:.17e} measured={samples:?}");
        let common = components::component_fit_of_order(&name, steps, &errors, 2.0);
        let strict = errors.iter().all(|v| v.is_finite() && *v > 0.0) && {
            let fit = convergence::error_power_fit(steps, &errors);
            1.9 < fit.exponent && fit.exponent < 2.1 && fit.r_squared > 0.999
        };
        if !common || !strict {
            failures.push(name);
        }
    }
}

fn directional_first_derivative(
    reference: &Case,
    direction: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Vec<f64> {
    // No second-order response is evaluated by the Hessian oracle. Scalar
    // second differences and their cancellation failures remain in the
    // validation evidence; the finer interval below resolves truncation.
    let path = hyperfine_nuclear_path(
        &reference.mol,
        &reference.basis,
        &reference.functional,
        reference.grid.as_ref(),
        &reference.scf,
        &reference.options,
        &[0, 1, 2, 3],
        direction,
        1,
        config,
    )
    .unwrap();
    assert!(path.response.converged);
    path.coefficients
        .iter()
        .flat_map(|n| flatten(&n[1]))
        .collect()
}

/// Deterministic orthonormal dense directions. Their outer products and
/// normalized pair sums span the complete symmetric nuclear Hessian space.
fn directions(dimension: usize) -> Vec<Vec<f64>> {
    let mut seed = 17_u64;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for _ in 0..dimension {
        let mut vector: Vec<_> = (0..dimension)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 11) as f64) / ((1_u64 << 53) as f64) - 0.5
            })
            .collect();
        for previous in &basis {
            let dot: f64 = vector.iter().zip(previous).map(|(a, b)| a * b).sum();
            for (a, b) in vector.iter_mut().zip(previous) {
                *a -= dot * b;
            }
        }
        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-6);
        for value in &mut vector {
            *value /= norm;
        }
        basis.push(vector);
    }
    let mut result = basis.clone();
    for i in 0..dimension {
        for j in 0..i {
            result.push(
                basis[i]
                    .iter()
                    .zip(&basis[j])
                    .map(|(a, b)| (a + b) / 2.0_f64.sqrt())
                    .collect(),
            );
        }
    }
    result
}

fn complete(name: &str) {
    let center = case(name, &[0.0; 12]);
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        max_iterations: 1000,
        ..ResponseSolverConfig::default()
    };
    let nuclei = [0, 1, 2, 3];
    let analytic = hyperfine_nuclear_derivatives(
        &center.mol,
        &center.basis,
        &center.functional,
        center.grid.as_ref(),
        &center.scf,
        &center.options,
        &nuclei,
        2,
        &config,
    )
    .unwrap();
    println!(
        "analytic paths={} diagnostics={:?}",
        analytic.paths.len(),
        analytic.paths
    );
    assert!(
        analytic.converged,
        "Keep every raw physical response defect"
    );
    let tensor_hessian = analytic.hessian.as_ref().unwrap();
    assert_eq!(analytic.paths.len(), 78);
    let first_order = hyperfine_nuclear_derivatives(
        &center.mol,
        &center.basis,
        &center.functional,
        center.grid.as_ref(),
        &center.scf,
        &center.options,
        &nuclei,
        1,
        &config,
    )
    .unwrap();
    println!(
        "first-order paths={} diagnostics={:?}",
        first_order.paths.len(),
        first_order.paths
    );
    assert!(first_order.converged && first_order.hessian.is_none());
    assert_eq!(first_order.paths.len(), 12);
    for (a, b) in first_order
        .gradient
        .iter()
        .flatten()
        .zip(analytic.gradient.iter().flatten())
    {
        for (x, y) in flatten(a).iter().zip(flatten(b)) {
            assert!((x - y).abs() < 1e-8);
        }
    }
    for value in analytic
        .value
        .iter()
        .chain(analytic.gradient.iter().flatten())
        .chain(tensor_hessian.iter().flatten().flatten())
    {
        raw_tensor_symmetry(value);
    }
    let values = observable(&center);
    let analytic_values: Vec<_> = analytic.value.iter().flat_map(flatten).collect();
    for (a, b) in values.iter().zip(&analytic_values) {
        assert!((a - b).abs() < 1e-8);
    }
    let mut failures = Vec::new();
    for coordinate in 0..12 {
        let expected: Vec<_> = analytic
            .gradient
            .iter()
            .flat_map(|n| flatten(&n[coordinate]))
            .collect();
        let measured = std::array::from_fn(|step| {
            let h = STEPS[step];
            let mut shift = vec![0.0; 12];
            shift[coordinate] = h;
            let plus = observable(&case(name, &shift));
            shift[coordinate] = -h;
            let minus = observable(&case(name, &shift));
            plus.iter()
                .zip(&minus)
                .map(|(a, b)| stencils::Stencil::Second.first(*a, *b, 0.0, 0.0, h))
                .collect()
        });
        fits(
            &format!("{name} gradient-{coordinate}"),
            &expected,
            &measured,
            &STEPS,
            &mut failures,
        );
    }
    let generic = directions(12);
    assert_eq!(generic.len(), 78);
    for (index, direction) in generic.iter().enumerate() {
        let mut expected = vec![0.0; values.len()];
        for (nucleus, tensor) in tensor_hessian.iter().enumerate() {
            for (i, row) in tensor.iter().enumerate() {
                for (j, entry) in row.iter().enumerate() {
                    for (component, value) in flatten(entry).iter().enumerate() {
                        expected[19 * nucleus + component] += direction[i] * direction[j] * value;
                    }
                }
            }
        }
        let velocity: Vec<[f64; 3]> = direction
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let measured = std::array::from_fn(|step| {
            let h = HESSIAN_STEPS[step];
            let plus = case(name, &direction.iter().map(|v| h * v).collect::<Vec<_>>());
            let minus = case(name, &direction.iter().map(|v| -h * v).collect::<Vec<_>>());
            let a = directional_first_derivative(&plus, &velocity, &config);
            let b = directional_first_derivative(&minus, &velocity, &config);
            a.iter()
                .zip(&b)
                .map(|(x, y)| stencils::Stencil::Second.first(*x, *y, 0.0, 0.0, h))
                .collect()
        });
        fits(
            &format!("{name} Hessian-direction-{index}"),
            &expected,
            &measured,
            &HESSIAN_STEPS,
            &mut failures,
        );
    }
    for axis in 0..3 {
        let mut translation = vec![[0.0; 3]; 4];
        for atom in &mut translation {
            atom[axis] = 1.0;
        }
        let path = hyperfine_nuclear_path(
            &center.mol,
            &center.basis,
            &center.functional,
            center.grid.as_ref(),
            &center.scf,
            &center.options,
            &nuclei,
            &translation,
            2,
            &config,
        )
        .unwrap();
        assert!(path.response.converged);
        for nucleus in &path.coefficients {
            for coefficient in &nucleus[1..] {
                for value in flatten(coefficient) {
                    assert!(value.abs() < 1e-8, "translation {axis}: {value}");
                }
            }
        }
    }
    assert!(hyperfine_nuclear_path(
        &center.mol,
        &center.basis,
        &center.functional,
        center.grid.as_ref(),
        &center.scf,
        &center.options,
        &nuclei,
        &[[0.0; 3]; 4],
        3,
        &config
    )
    .is_err());
    assert!(hyperfine_nuclear_path(
        &center.mol,
        &center.basis,
        &center.functional,
        center.grid.as_ref(),
        &center.scf,
        &center.options,
        &[4],
        &[[0.0; 3]; 4],
        2,
        &config
    )
    .is_err());
    assert!(
        failures.is_empty(),
        "Retained failed component fits: {failures:?}"
    );
}

#[test]
fn moving_methyl_hf_operators_match_static_provider() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| {
            let center = case("hf", &[0.0; 12]);
            let config = ResponseSolverConfig {
                tolerance: 1e-12,
                max_iterations: 1000,
                ..ResponseSolverConfig::default()
            };
            let velocity = [
                [0.21, -0.13, 0.17],
                [-0.15, 0.18, 0.09],
                [0.11, 0.07, -0.19],
                [-0.12, 0.23, 0.16],
            ];
            let path = hyperfine_nuclear_path(
                &center.mol,
                &center.basis,
                &center.functional,
                center.grid.as_ref(),
                &center.scf,
                &center.options,
                &[0, 1, 2, 3],
                &velocity,
                2,
                &config,
            )
            .unwrap();
            assert!(path.response.converged);
            let original = observable(&center);
            let constant: Vec<_> = path
                .coefficients
                .iter()
                .flat_map(|n| flatten(&n[0]))
                .collect();
            for (a, b) in original.iter().zip(&constant) {
                assert!((a - b).abs() < 1e-8);
            }
            for coefficient in path.coefficients.iter().flatten() {
                raw_tensor_symmetry(coefficient);
            }
            println!("moving operator smoke path={path:?}");
            let original_iterations = path.response.iterations.clone();
            let diagnostics = HyperfinePathDiagnostics::from(path);
            assert_eq!(diagnostics.iterations, original_iterations);
            assert_eq!(diagnostics.iterations.len(), 2);
        })
        .unwrap();
}

#[test]
fn complete_methyl_hf_hyperfine_nuclear_derivatives() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| complete("hf"))
        .unwrap();
}

#[test]
fn complete_methyl_pbe_hyperfine_nuclear_derivatives() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(512))
        .unwrap()
        .install(|| complete("pbe"))
        .unwrap();
}
