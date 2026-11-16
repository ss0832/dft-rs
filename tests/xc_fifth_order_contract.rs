// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete symmetric fifth local tensors at an interior descriptor point,
//! from mixed potential jets versus independent dense-third finite differences.
//! This does not certify branch boundaries or every named functional recipe.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::xc::taylor::XcTaylorInput;
use dft_rs::xc::{resolve, Family, FunctionalSpec, GridVars, Kind, XcFunctional, XcOut};
use std::collections::HashMap;

fn pair(n: usize, a: usize, b: usize) -> usize {
    let mut k = 0;
    for i in 0..n {
        for j in i..n {
            if i == a && j == b {
                return k;
            }
            k += 1;
        }
    }
    panic!("invalid pair")
}
fn triple(n: usize, a: usize, b: usize, c: usize) -> usize {
    let mut k = 0;
    for i in 0..n {
        for j in i..n {
            for l in j..n {
                if (i, j, l) == (a, b, c) {
                    return k;
                }
                k += 1;
            }
        }
    }
    panic!("invalid triple")
}
fn third(out: &XcOut, mut axes: [usize; 3]) -> f64 {
    axes.sort_unstable();
    let [a, b, c] = axes;
    if c < 2 {
        out.v3rho3[a + b + c][0]
    } else if b < 2 && c < 5 {
        out.v3rho2sigma[3 * pair(2, a, b) + c - 2][0]
    } else if a < 2 && c < 5 {
        out.v3rhosigma2[6 * a + pair(3, b - 2, c - 2)][0]
    } else if c < 5 {
        out.v3sigma3[triple(3, a - 2, b - 2, c - 2)][0]
    } else if b < 2 {
        out.v3rho2tau[2 * pair(2, a, b) + c - 5][0]
    } else if a < 2 && b < 5 {
        out.v3rhosigmatau[(a * 3 + b - 2) * 2 + c - 5][0]
    } else if a < 2 {
        out.v3rhotau2[a * 3 + pair(2, b - 5, c - 5)][0]
    } else if b < 5 {
        out.v3sigma2tau[2 * pair(3, a - 2, b - 2) + c - 5][0]
    } else if a < 5 {
        out.v3sigmatau2[3 * (a - 2) + pair(2, b - 5, c - 5)][0]
    } else {
        out.v3tau3[a + b + c - 15][0]
    }
}
fn dense(part: &dyn XcFunctional, point: [f64; 7], dim: usize) -> Vec<f64> {
    let v = point.map(|x| [x]);
    let vars = GridVars {
        rho: [&v[0], &v[1]],
        sigma: Some([&v[2], &v[3], &v[4]]),
        tau: Some([&v[5], &v[6]]),
    };
    let mut out = XcOut::default();
    out.prepare(1, 3, part.family());
    part.eval_add(&vars, 3, 1.0, &mut out).unwrap();
    (0..dim * dim * dim)
        .map(|k| third(&out, [k / (dim * dim), (k / dim) % dim, k % dim]))
        .collect()
}

fn check(name: &str) {
    let spec = if name == "pw92" {
        FunctionalSpec::Split {
            x: "slater".into(),
            c: "pw92".into(),
        }
    } else {
        FunctionalSpec::Combined(name.into())
    };
    let functional = resolve(&spec).unwrap();
    // Each perturbation is dimensionless and scaled by this fixed interior
    // point. Even the coarsest stencil remains within the FHC/PSD margins.
    let point = [0.14, 0.085, 0.5, 0.1, 0.38, 1.05, 0.92];
    for (index, (_, part)) in functional.parts.iter().enumerate() {
        if name == "pw92" && part.kind() == Kind::Exchange {
            continue;
        }
        let label = format!(
            "{name} part{index} {} {:?} params={:?}",
            part.name(),
            part.kind(),
            part.params()
        );
        let dim = match part.family() {
            Family::Lda => 2,
            Family::Gga => 5,
            Family::MetaGga => 7,
        };
        assert!(part.max_analytic_order() >= 3);
        let mut jets: HashMap<(usize, usize, usize, usize), [f64; 7]> = HashMap::new();
        let mut coefficient = |a: usize, b: usize, c: usize, d: usize| {
            *jets.entry((a, b, c, d)).or_insert_with(|| {
                let mut variables = point.map(|x| vec![x, 0.0, 0.0]);
                variables[c][1] += point[c];
                if d < dim {
                    variables[d][1] += point[d];
                }
                let input = XcTaylorInput {
                    variables,
                    parameters: None,
                };
                let mut first = std::array::from_fn(|_| vec![0.0; 3]);
                let mut second = first.clone();
                first[a][0] = point[a];
                second[b][0] = point[b];
                let mixed = std::array::from_fn(|_| vec![0.0; 3]);
                let out = part
                    .taylor_bidirectional_potentials(&input, &first, &second, &mixed)
                    .unwrap();
                std::array::from_fn(|j| out.mixed.potentials[j][2] * point[j])
            })
        };
        let mut indices = Vec::new();
        let mut analytic = Vec::new();
        let mut zero_indices = Vec::new();
        for j in 0..dim {
            for a in j..dim {
                for b in a..dim {
                    for c in b..dim {
                        for d in c..dim {
                            let pure_c = coefficient(a, b, c, dim)[j];
                            let value = if c == d {
                                2.0 * pure_c
                            } else {
                                coefficient(a, b, c, d)[j] - pure_c - coefficient(a, b, d, dim)[j]
                            };
                            let axes = [j, a, b, c, d];
                            let alpha = |i| matches!(i, 0 | 2 | 5);
                            let beta = |i| matches!(i, 1 | 4 | 6);
                            // The defining LYP energy is affine in its three
                            // sigma variables. Two sigma derivatives therefore
                            // vanish identically, including modified parameters.
                            // Fitting finite-difference roundoff at those exact
                            // zeros cannot establish a convergence exponent.
                            let affine_sigma_zero = part.name() == "lyp"
                                && axes.iter().filter(|&&i| (2..5).contains(&i)).count() >= 2;
                            // B95's opposite-spin term has no tau dependence;
                            // each same-spin term is affine in its own tau and
                            // depends only on that spin's rho and sigma.
                            let b95_zero = part.name() == "b95"
                                && (axes.contains(&3)
                                    || axes.iter().filter(|&&i| i >= 5).count() >= 2
                                    || (axes.contains(&5) && !axes.iter().all(|&i| alpha(i)))
                                    || (axes.contains(&6) && !axes.iter().all(|&i| beta(i))));
                            // The N12-SX B97-form correlation depends on the
                            // two same-spin sigma entries, never sigma_ab.
                            let n12_sx_zero = matches!(
                                part.name(),
                                "n12-sxc" | "thcthc" | "thcthhybc" | "bmkc" | "b98"
                            ) && axes.contains(&3);
                            // dlDF's opposite-spin term has no tau; each
                            // same-spin term depends only on its own descriptors.
                            let dldf_zero = part.name() == "dldfc"
                                && (axes.contains(&3)
                                    || (axes.contains(&5) && !axes.iter().all(|&i| alpha(i)))
                                    || (axes.contains(&6) && !axes.iter().all(|&i| beta(i))));
                            let structural = n12_sx_zero
                                || dldf_zero
                                || b95_zero
                                || affine_sigma_zero
                                || (part.kind() == Kind::Exchange
                                    && !(axes.iter().all(|&i| alpha(i))
                                        || axes.iter().all(|&i| beta(i))));
                            if structural {
                                // dlDF's tau-containing spin-separable zeros
                                // can otherwise be obtained as A-A after
                                // polarization of two univariate seeds. Test
                                // the absent mixed spin dependency directly,
                                // so the exact-zero assertion does not measure
                                // rounding of that subtraction.
                                let value = if dldf_zero && !axes.contains(&3) {
                                    let tau = if axes.contains(&5) { 5 } else { 6 };
                                    let other = axes
                                        .iter()
                                        .copied()
                                        .find(|&i| if tau == 5 { !alpha(i) } else { !beta(i) })
                                        .unwrap();
                                    let mut remaining = axes.to_vec();
                                    remaining
                                        .remove(remaining.iter().position(|&i| i == tau).unwrap());
                                    remaining.remove(
                                        remaining.iter().position(|&i| i == other).unwrap(),
                                    );
                                    let [j, c, d] = [remaining[0], remaining[1], remaining[2]];
                                    let pure = coefficient(tau, other, c, dim)[j];
                                    if c == d {
                                        2.0 * pure
                                    } else {
                                        coefficient(tau, other, c, d)[j]
                                            - pure
                                            - coefficient(tau, other, d, dim)[j]
                                    }
                                } else {
                                    value
                                };
                                assert_eq!(
                                    value,
                                    0.0,
                                    "{} defining-equation zero {axes:?}",
                                    part.name()
                                );
                                zero_indices.push(axes);
                            } else {
                                indices.push(axes);
                                analytic.push(value);
                            }
                        }
                    }
                }
            }
        }
        let names = indices
            .iter()
            .map(|axes| format!("{label} fifth {axes:?}"))
            .collect::<Vec<_>>();
        let mut failures = Vec::new();
        let mut cache: HashMap<[u64; 7], Vec<f64>> = HashMap::new();
        let mut at = |x: [f64; 7], j: usize, a: usize, b: usize| {
            let tensor = cache
                .entry(x.map(f64::to_bits))
                .or_insert_with(|| dense(part.as_ref(), x, dim));
            tensor[(j * dim + a) * dim + b] * point[j] * point[a] * point[b]
        };
        stencils::fit_components_staged(
            &names,
            &[
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.16, 0.08, 0.04, 0.02], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; indices.len()];
                for (k, &h) in steps.iter().enumerate() {
                    for (slot, &[j, a, b, c, d]) in indices.iter().enumerate() {
                        let mut fd = |h: f64| {
                            let mut shifted = |s: f64, t: f64| {
                                let mut x = point;
                                x[c] += s * point[c];
                                x[d] += t * point[d];
                                at(x, j, a, b)
                            };
                            if c == d {
                                let centre = shifted(0.0, 0.0);
                                let forward = (shifted(h, 0.0) - centre) / h;
                                let backward = (centre - shifted(-h, 0.0)) / h;
                                stencils::Stencil::Second.first(
                                    forward,
                                    backward,
                                    0.0,
                                    0.0,
                                    0.5 * h,
                                )
                            } else {
                                let forward = stencils::Stencil::Second.first(
                                    shifted(h, h),
                                    shifted(h, -h),
                                    0.0,
                                    0.0,
                                    h,
                                );
                                let backward = stencils::Stencil::Second.first(
                                    shifted(-h, h),
                                    shifted(-h, -h),
                                    0.0,
                                    0.0,
                                    h,
                                );
                                stencils::Stencil::Second.first(forward, backward, 0.0, 0.0, h)
                            }
                        };
                        let value = if stencil == stencils::Stencil::Second {
                            fd(h)
                        } else {
                            (4.0 * fd(h) - fd(2.0 * h)) / 3.0
                        };
                        errors[slot][k] = (value - analytic[slot]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        println!(
            "{} complete fifth: fitted={} structural_zero={} dense_points={}",
            label,
            indices.len(),
            zero_indices.len(),
            cache.len()
        );
        components::assert_no_failures(part.name(), &failures);
    }
}

#[test]
fn weighted_recipes_match_independent_dense_third_contractions() {
    check_weighted_recipes(&[
        "svwn5",
        "pbe",
        "tpss",
        "pkzb",
        "pbe0",
        "cam-b3lyp",
        "edf2",
        "bb95",
        "dsd-pbeb95",
    ]);
}
fn check_weighted_recipes(names: &[&str]) {
    let point = [0.14, 0.085, 0.5, 0.1, 0.38, 1.05, 0.92];
    let a = std::array::from_fn::<_, 7, _>(|i| 0.03 * (i + 1) as f64 * point[i]);
    let b = std::array::from_fn::<_, 7, _>(|i| -0.02 * (i + 2) as f64 * point[i]);
    let c = std::array::from_fn::<_, 7, _>(|i| 0.004 * (i + 1) as f64 * point[i]);
    let input = XcTaylorInput {
        variables: point.map(|x| vec![x]),
        parameters: None,
    };
    let values = point.map(|x| [x]);
    let vars = GridVars {
        rho: [&values[0], &values[1]],
        sigma: Some([&values[2], &values[3], &values[4]]),
        tau: Some([&values[5], &values[6]]),
    };
    let near = |x: f64, y: f64| assert!((x - y).abs() < 1e-10 * (1.0 + y.abs()), "{x} != {y}");
    for &name in names {
        let functional = if name == "dsd-pbeb95" {
            dft_rs::xc::double_hybrid::DoubleHybridFunctional::named(name)
                .unwrap()
                .parent
        } else {
            resolve(&FunctionalSpec::Combined(name.into())).unwrap()
        };
        let result = functional
            .semilocal_taylor_bidirectional_potentials(
                &input,
                &a.map(|x| vec![x]),
                &b.map(|x| vec![x]),
                &c.map(|x| vec![x]),
            )
            .unwrap();
        let mut output = XcOut::default();
        functional.eval(&vars, 3, &mut output).unwrap();
        let (potential, kernel) = output.potential_and_kernel(0, functional.family).unwrap();
        let dim = match functional.family {
            Family::Lda => 2,
            Family::Gga => 5,
            Family::MetaGga => 7,
        };
        near(result.value.energy[0], output.e[0]);
        near(
            result.first.energy[0],
            (0..7).map(|j| potential[j] * a[j]).sum(),
        );
        near(
            result.second.energy[0],
            (0..7).map(|j| potential[j] * b[j]).sum(),
        );
        near(
            result.mixed.energy[0],
            (0..7)
                .map(|j| {
                    potential[j] * c[j] + (0..7).map(|k| kernel[j][k] * a[j] * b[k]).sum::<f64>()
                })
                .sum(),
        );
        for j in 0..7 {
            near(result.value.potentials[j][0], potential[j]);
            near(
                result.first.potentials[j][0],
                (0..7).map(|k| kernel[j][k] * a[k]).sum(),
            );
            near(
                result.second.potentials[j][0],
                (0..7).map(|k| kernel[j][k] * b[k]).sum(),
            );
            let expected = (0..7)
                .map(|k| {
                    kernel[j][k] * c[k]
                        + (0..7)
                            .map(|l| {
                                if j < dim && k < dim && l < dim {
                                    third(&output, [j, k, l]) * a[k] * b[l]
                                } else {
                                    0.0
                                }
                            })
                            .sum::<f64>()
                })
                .sum();
            near(result.mixed.potentials[j][0], expected);
        }
    }
}

#[test]
fn lda_provider_complete_fifth_tensors() {
    check("svwn5");
}
#[test]
fn gga_provider_complete_fifth_tensors() {
    check("pbe");
}
#[test]
fn meta_provider_complete_fifth_tensors() {
    check("tpss");
}
#[test]
fn pkzb_provider_complete_fifth_tensors() {
    check("pkzb");
}
#[test]
fn original_pw92_provider_complete_fifth_tensor() {
    check("pw92");
}
#[test]
fn task_and_mtask_provider_complete_fifth_tensors() {
    check("task");
    check("mtask");
}

#[test]
fn edf2_parameterized_parts_complete_fifth_tensors() {
    check("edf2");
}

#[test]
fn standard_b95_parent_complete_fifth_tensor() {
    check("bb95");
}

#[test]
fn n12_sx_parts_complete_fifth_tensors() {
    check("n12-sx");
}

#[test]
fn new_non_dh_batch_complete_fifth_tensors() {
    for name in ["dldf", "b98", "thcth", "thcthhyb", "bmk"] {
        check(name);
    }
}

#[test]
fn n12_sx_parts_complete_fourth_tensors() {
    let functional = resolve(&FunctionalSpec::Combined("n12-sx".into())).unwrap();
    let point = [0.14, 0.085, 0.5, 0.1, 0.38, 1.05, 0.92];
    for (index, (_, part)) in functional.parts.iter().enumerate() {
        let mut labels = Vec::new();
        let mut targets = Vec::new();
        let mut structural_zeros = 0;
        for j in 0..5 {
            for a in j..5 {
                for b in a..5 {
                    for c in b..5 {
                        let mut variables = point.map(|x| vec![x, 0.0]);
                        variables[c][1] = point[c];
                        let input = XcTaylorInput {
                            variables,
                            parameters: None,
                        };
                        let mut first = std::array::from_fn(|_| vec![0.0; 2]);
                        let mut second = first.clone();
                        first[a][0] = point[a];
                        second[b][0] = point[b];
                        let mixed = std::array::from_fn(|_| vec![0.0; 2]);
                        let out = part
                            .taylor_bidirectional_potentials(&input, &first, &second, &mixed)
                            .unwrap();
                        let actual = out.mixed.potentials[j][1] * point[j];
                        let axes = [j, a, b, c];
                        let zero = if part.kind() == Kind::Exchange {
                            !(axes.iter().all(|i| [0, 2].contains(i))
                                || axes.iter().all(|i| [1, 4].contains(i)))
                        } else {
                            axes.contains(&3)
                        };
                        if zero {
                            assert_eq!(actual, 0.0, "n12-sx part{index} fourth zero {axes:?}");
                            structural_zeros += 1;
                        } else {
                            labels.push(format!("n12-sx part{index} fourth {axes:?}"));
                            targets.push((axes, actual));
                        }
                    }
                }
            }
        }
        assert_eq!(
            targets.len(),
            if part.kind() == Kind::Exchange {
                10
            } else {
                35
            }
        );
        let mut cache = HashMap::<[u64; 7], Vec<f64>>::new();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; targets.len()];
                for (slot, ([j, a, b, c], expected)) in targets.iter().enumerate() {
                    for (step, &h) in steps.iter().enumerate() {
                        let mut at = |scale: f64| {
                            let mut p = point;
                            p[*c] += scale * point[*c];
                            let tensor = cache
                                .entry(p.map(f64::to_bits))
                                .or_insert_with(|| dense(part.as_ref(), p, 5));
                            tensor[(j * 5 + a) * 5 + b] * point[*j] * point[*a] * point[*b]
                        };
                        let (p, m) = (at(h), at(-h));
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        errors[slot][step] = (stencil.first(p, m, p2, m2, h) - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("N12-SX complete fourth local tensor", &failures);
        println!(
            "n12-sx part{index} complete fourth: fitted={} structural_zero={structural_zeros}",
            targets.len()
        );
    }
}

#[test]
fn new_non_dh_batch_weighted_compositions() {
    check_weighted_recipes(&["dldf", "b98", "thcth", "thcthhyb", "bmk"]);
}
