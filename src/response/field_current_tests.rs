// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent density-coordinate AD oracle for the current-invariant
//! energy and every independent complex Fock coefficient, through order two.
//! Registered-provider density tangents are fitted through field order four.
use super::*;
use crate::acceptance::components::{assert_no_failures, fit_components};
use crate::dual::{DualN, Scalar};

pub(super) type D = DualN<f64, 18>;

pub(super) fn coordinates() -> Vec<(usize, usize, usize, bool)> {
    (0..2)
        .flat_map(|s| {
            (0..3).flat_map(move |i| {
                (i..3).flat_map(move |j| {
                    [false, true]
                        .into_iter()
                        .filter(move |im| i != j || !im)
                        .map(move |im| (s, i, j, im))
                })
            })
        })
        .collect()
}

// This oracle uses exact trigonometric phases, ordinary complex AO
// contractions, and AD in density coordinates. It does not use the
// production descriptor series, potential pullback or Fock assembly.
pub(super) fn finite_fields(point: &Point, path: &[Vec<Cm>; 2], t: f64) -> ([D; 7], [[D; 3]; 2]) {
    let c = D::constant;
    let coords = coordinates();
    let mut density = [[[(c(0.0), c(0.0)); 3]; 3]; 2];
    for (s, spin) in density.iter_mut().enumerate() {
        for (i, row) in spin.iter_mut().enumerate() {
            for (j, element) in row.iter_mut().enumerate() {
                *element = (
                    c((0..3)
                        .map(|k| path[s][k].re[(i, j)] * t.powi(k as i32))
                        .sum()),
                    c((0..3)
                        .map(|k| path[s][k].im[(i, j)] * t.powi(k as i32))
                        .sum()),
                );
            }
        }
    }
    for (seed, &(s, i, j, imaginary)) in coords.iter().enumerate() {
        if imaginary {
            density[s][i][j].1.im[seed] = 1.0;
            density[s][j][i].1.im[seed] = -1.0;
        } else {
            density[s][i][j].0.im[seed] = 1.0;
            density[s][j][i].0.im[seed] = 1.0;
        }
    }
    let w: [Cx; 3] = std::array::from_fn(|i| {
        let phase = -t * point.theta[i];
        (point.phi[i] * phase.cos(), point.phi[i] * phase.sin())
    });
    let grad: [[Cx; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|a| {
            let phase = -t * point.theta[i];
            mul(
                (phase.cos(), phase.sin()),
                (point.grad[i][a], -t * point.slope[i][a] * point.phi[i]),
            )
        })
    });
    let pi: [[Cx; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|a| {
            let phase = -t * point.theta[i];
            mul(
                (phase.cos(), phase.sin()),
                (t * point.gauge[i][a] * point.phi[i], -point.grad[i][a]),
            )
        })
    });
    let mut vars = [c(0.0); 7];
    let mut gradients = [[c(0.0); 3]; 2];
    for s in 0..2 {
        let mut current = [c(0.0); 3];
        let contract = |i: usize, j: usize, z: Cx| {
            density[s][i][j].0.scale(z.0) - density[s][i][j].1.scale(z.1)
        };
        for i in 0..3 {
            for j in 0..3 {
                vars[s] = vars[s] + contract(i, j, mul(w[i], conj(w[j])));
                for a in 0..3 {
                    gradients[s][a] =
                        gradients[s][a] + contract(i, j, mul(w[i], conj(grad[j][a]))).scale(2.0);
                    vars[5 + s] =
                        vars[5 + s] + contract(i, j, mul(pi[i][a], conj(pi[j][a]))).scale(0.5);
                    current[a] = current[a] + contract(i, j, mul(pi[i][a], conj(w[j])));
                }
            }
        }
        assert!(vars[s].re > 0.0);
        for j in current {
            vars[5 + s] = vars[5 + s] - (j * j / vars[s]).scale(0.5);
        }
    }
    for (index, (s, u)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        for a in 0..3 {
            vars[2 + index] = vars[2 + index] + gradients[s][a] * gradients[u][a];
        }
    }
    (vars, gradients)
}

fn finite(point: &Point, functional: &ResolvedFunctional, path: &[Vec<Cm>; 2], t: f64) -> Vec<f64> {
    let (vars, _) = finite_fields(point, path, t);
    let coords = coordinates();
    let mut values = Vec::with_capacity(19);
    let base = functional
        .semilocal_taylor_energy(&XcTaylorInput {
            variables: vars.map(|x| vec![x.re]),
            parameters: None,
        })
        .unwrap();
    values.push(point.weight * base[0]);
    for (seed, &(_, i, j, _)) in coords.iter().enumerate() {
        let energy = functional
            .semilocal_taylor_energy(&XcTaylorInput {
                variables: vars.map(|x| vec![x.re, x.im[seed]]),
                parameters: None,
            })
            .unwrap();
        values.push(point.weight * energy[1] / if i == j { 1.0 } else { 2.0 });
    }
    values
}

pub(super) fn density_path() -> [Vec<Cm>; 2] {
    std::array::from_fn(|s| {
        (0..3)
            .map(|k| {
                let mut v = Cm::zeros(3, 3);
                for i in 0..3 {
                    v.re[(i, i)] = if k == 0 {
                        0.64 + 0.13 * (i + s) as f64
                    } else {
                        0.06 * (1 + i + s + k) as f64 / (k + 1) as f64
                    };
                    for j in i + 1..3 {
                        let re = 0.037 * (1 + i + 2 * j + s + k) as f64 / (k + 1) as f64;
                        let im = 0.019 * (2 + i + j + 2 * s + k) as f64 / (k + 1) as f64;
                        v.re[(i, j)] = re;
                        v.re[(j, i)] = re;
                        v.im[(i, j)] = im;
                        v.im[(j, i)] = -im;
                    }
                }
                v
            })
            .collect()
    })
}

fn test_point() -> Point {
    Point {
        weight: 0.73,
        phi: vec![0.71, -0.43, 0.58],
        grad: vec![
            [0.18, -0.37, 0.22],
            [-0.46, 0.29, 0.17],
            [0.31, 0.41, -0.36],
        ],
        theta: vec![0.37, -0.53, 0.22],
        slope: vec![
            [0.21, -0.17, 0.32],
            [-0.16, 0.28, 0.13],
            [0.39, 0.12, -0.25],
        ],
        gauge: vec![
            [0.14, 0.22, -0.19],
            [-0.23, 0.31, 0.24],
            [0.19, -0.27, 0.34],
        ],
    }
}

#[test]
fn every_current_invariant_energy_and_complex_fock_derivative_has_an_independent_fit() {
    let point = test_point();
    let coords = coordinates();
    let p = density_path();
    let functional =
        crate::xc::resolve(&crate::xc::FunctionalSpec::Combined("tpss".into())).unwrap();
    let analytic: Vec<Vec<f64>> = (0..3)
        .map(|n| {
            let (energy, matrix) = point_xc(
                &point,
                &functional,
                &p,
                n,
                true,
                MagneticKineticDensity::CurrentInvariant,
            )
            .unwrap();
            let matrix = matrix.unwrap();
            let mut out = vec![energy[n] * if n == 2 { 2.0 } else { 1.0 }];
            for &(s, i, j, im) in &coords {
                out.push(
                    if im {
                        matrix[s].1[i * 3 + j]
                    } else {
                        matrix[s].0[i * 3 + j]
                    } * if n == 2 { 2.0 } else { 1.0 },
                );
            }
            out
        })
        .collect();
    let zero = finite(&point, &functional, &p, 0.0);
    for (a, b) in zero.iter().zip(&analytic[0]) {
        assert!(
            (a - b).abs() < 1e-13,
            "base AD pullback mismatch: {a:.16e} {b:.16e}"
        );
    }
    let mut failures = Vec::new();
    for (order, target) in analytic.iter().enumerate().skip(1) {
        let names: Vec<_> = (0..19)
            .map(|i| format!("current-invariant order={order} component={i}"))
            .collect();
        fit_components(
            &names,
            &[
                [0.08, 0.06, 0.04, 0.03],
                [0.02, 0.015, 0.01, 0.0075],
                [0.16, 0.12, 0.08, 0.06],
            ],
            |steps| {
                let mut errors = vec![[0.0; 4]; 19];
                for (k, &h) in steps.iter().enumerate() {
                    let plus = finite(&point, &functional, &p, h);
                    let minus = finite(&point, &functional, &p, -h);
                    for i in 0..19 {
                        let fd = if order == 1 {
                            (plus[i] - minus[i]) / (2.0 * h)
                        } else {
                            (plus[i] + minus[i] - 2.0 * zero[i]) / (h * h)
                        };
                        errors[i][k] = (fd - target[i]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    assert_no_failures("all current-invariant energy/Fock derivatives", &failures);
}

pub(super) struct SpatialFixture {
    pub centers: Vec<[f64; 3]>,
    pub positions: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub points: Vec<Point>,
}
pub(super) fn spatial_fixture() -> SpatialFixture {
    let centers = vec![
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    let positions = vec![
        [0.24, 0.31, 0.52],
        [1.4, 0.15, -0.1],
        [-0.1, 1.5, 0.3],
        [0.4, 0.5, 1.8],
        [0.8, -0.25, 0.7],
    ];
    let weights = vec![0.43, 0.67, 0.39, 0.51, 0.62];
    let field = [0.21, -0.13, 0.17];
    let points = positions
        .iter()
        .zip(&weights)
        .map(|(r, &weight)| {
            let mut point = Point {
                weight,
                phi: Vec::new(),
                grad: Vec::new(),
                theta: Vec::new(),
                slope: Vec::new(),
                gauge: Vec::new(),
            };
            for mu in 0..3 {
                let exponent = [0.71, 0.93, 0.56][mu];
                let delta: [f64; 3] = std::array::from_fn(|a| r[a] - centers[mu][a]);
                let phi: f64 = (-exponent * delta.iter().map(|d| d * d).sum::<f64>()).exp();
                point.phi.push(phi);
                point.grad.push(delta.map(|d| -2.0 * exponent * d * phi));
                let slope = half_cross(field, centers[mu]);
                let potential = half_cross(field, *r);
                point.theta.push((0..3).map(|a| slope[a] * r[a]).sum());
                point.slope.push(slope);
                point
                    .gauge
                    .push(std::array::from_fn(|a| potential[a] - slope[a]));
            }
            point
        })
        .collect();
    SpatialFixture {
        centers,
        positions,
        weights,
        points,
    }
}

/// The same nonlinear pullback must work for every registered semilocal
/// provider, not just a particular functional used by a molecular fixture.
/// Each of the 18 generic density directions is a column of I + v v^T,
/// hence the directions span the complete two-spin Hermitian density space.
#[test]
fn registered_field_tangents_fit_all_independent_density_components() {
    use crate::acceptance::stencils::{fit_components_staged, Stencil};
    use crate::xc::{FunctionalSpec, WeightedFunctionalTerm};
    let mut models = Vec::new();
    for name in crate::xc::COMBINED_NAMES {
        models.push((
            name.to_string(),
            crate::xc::resolve(&FunctionalSpec::Combined((*name).into())).unwrap(),
        ));
    }
    for name in crate::xc::EXCHANGE_NAMES {
        models.push((
            format!("exchange:{name}"),
            crate::xc::resolve(&FunctionalSpec::Split {
                x: (*name).into(),
                c: "none".into(),
            })
            .unwrap(),
        ));
    }
    for name in crate::xc::CORRELATION_NAMES {
        models.push((
            format!("correlation:{name}"),
            crate::xc::resolve(&FunctionalSpec::Split {
                x: "slater".into(),
                c: (*name).into(),
            })
            .unwrap(),
        ));
    }
    for name in crate::xc::double_hybrid::DOUBLE_HYBRID_NAMES {
        let recipe = crate::xc::double_hybrid::DoubleHybridFunctional::named(name).unwrap();
        models.push((format!("{name}:parent"), recipe.parent));
        if let Some(target) = recipe.target {
            models.push((format!("{name}:target"), target));
        }
    }
    models.push((
        "weighted-cross-family".into(),
        ResolvedFunctional::weighted(&[
            WeightedFunctionalTerm {
                weight: 0.37,
                functional: FunctionalSpec::Combined("svwn5".into()),
            },
            WeightedFunctionalTerm {
                weight: 0.19,
                functional: FunctionalSpec::Combined("pbe".into()),
            },
            WeightedFunctionalTerm {
                weight: 0.73,
                functional: FunctionalSpec::Combined("tpss".into()),
            },
        ])
        .unwrap(),
    ));
    let selection = std::env::var("DFT_RS_TEST_MODEL").ok();
    if let Some(name) = &selection {
        assert!(
            models.iter().any(|(label, _)| label == name),
            "unknown provider selection {name}"
        );
    }
    let point = test_point();
    let coords = coordinates();
    let mut base = density_path();
    for spin in &mut base {
        for k in 3..=4 {
            let mut term = spin[2].clone();
            for part in [&mut term.re, &mut term.im] {
                for v in part.as_mut_slice() {
                    *v *= 0.07_f64.powi(k - 2);
                }
            }
            spin.push(term);
        }
    }
    let mut failures = Vec::new();
    let mut measured = 0;
    for (label, functional) in models {
        if selection.as_ref().is_some_and(|name| name != &label) {
            continue;
        }
        if functional.parts.is_empty() {
            continue;
        }
        // Auxiliary nonlocal and PT2 terms are deliberately outside this
        // semilocal chain-rule test; their full energies have separate gates.
        for order in 0..=4 {
            for seed in 0..coords.len() {
                let mut direction: [Vec<Cm>; 2] = std::array::from_fn(|_| vec![Cm::zeros(3, 3); 5]);
                for (column, &(spin, i, j, imaginary)) in coords.iter().enumerate() {
                    let entry = if seed == column { 1.0 } else { 0.0 }
                        + 0.0003 * (seed + 1) as f64 * (column + 1) as f64;
                    let matrix = &mut direction[spin][0];
                    if imaginary {
                        matrix.im[(i, j)] = entry;
                        matrix.im[(j, i)] = -entry;
                    } else {
                        matrix.re[(i, j)] = entry;
                        matrix.re[(j, i)] = entry;
                    }
                }
                let exact = match point_xc_linear(
                    &point,
                    &functional,
                    &base,
                    &direction,
                    order,
                    MagneticKineticDensity::CurrentInvariant,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        let failed = format!("provider={label} field_order={order} density_direction={seed}: {error}");
                        println!("{failed} FAILED (missing or invalid provider response)");
                        failures.push(failed);
                        continue;
                    }
                };
                let active: Vec<_> = coords
                    .iter()
                    .copied()
                    .filter(|&(_, _, _, im)| {
                        !(order == 0 && im && functional.family != Family::MetaGga)
                    })
                    .collect();
                if order == 0 && functional.family != Family::MetaGga {
                    for &(spin, i, j, im) in &coords {
                        if im {
                            assert_eq!(exact[spin].1[3 * i + j], 0.0);
                        }
                    }
                }
                let names: Vec<_> = active.iter().map(|coordinate|
                    format!("provider={label} field_order={order} density_direction={seed} Fock={coordinate:?}")).collect();
                let mut cache = std::collections::BTreeMap::new();
                let mut at = |h: f64| {
                    cache
                        .entry(h.to_bits())
                        .or_insert_with(|| {
                            let mut shifted = base.clone();
                            for spin in 0..2 {
                                for k in 0..=order {
                                    cadd(&mut shifted[spin][k], &direction[spin][k], h);
                                }
                            }
                            point_xc(
                                &point,
                                &functional,
                                &shifted,
                                order,
                                true,
                                MagneticKineticDensity::CurrentInvariant,
                            )
                            .unwrap()
                            .1
                            .unwrap()
                        })
                        .clone()
                };
                fit_components_staged(
                    &names,
                    &[
                        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
                        ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
                        ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
                        ([0.04, 0.02, 0.01, 0.005], Stencil::Fourth),
                        // Tiny cubic truncation terms can be lost at the
                        // finer intervals above. Retain those failures, then
                        // measure larger steps without changing any gate.
                        ([0.16, 0.12, 0.08, 0.06], Stencil::Second),
                        ([0.32, 0.24, 0.16, 0.12], Stencil::Second),
                        ([0.08, 0.06, 0.04, 0.03], Stencil::Fourth),
                        ([0.16, 0.12, 0.08, 0.06], Stencil::Fourth),
                    ],
                    |steps, stencil| {
                        let mut errors = vec![[0.0; 4]; active.len()];
                        for (k, &h) in steps.iter().enumerate() {
                            let plus = at(h);
                            let minus = at(-h);
                            let (plus2, minus2) = if stencil == Stencil::Fourth {
                                (at(2.0 * h), at(-2.0 * h))
                            } else {
                                (plus.clone(), minus.clone())
                            };
                            for (entry, &(spin, i, j, im)) in active.iter().enumerate() {
                                let read = |value: &PointMatrices| {
                                    if im {
                                        value[spin].1[3 * i + j]
                                    } else {
                                        value[spin].0[3 * i + j]
                                    }
                                };
                                errors[entry][k] = (stencil.first(
                                    read(&plus),
                                    read(&minus),
                                    read(&plus2),
                                    read(&minus2),
                                    h,
                                ) - read(&exact))
                                .abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
                measured += names.len();
            }
        }
    }
    println!("Registered semilocal field tangent component fits={measured}");
    assert_no_failures("registered field density tangents", &failures);
}
