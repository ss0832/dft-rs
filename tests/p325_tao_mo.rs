// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Defining terms and independent local derivative fits for TM and revTM.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/tao_mo_oracle.rs"]
mod oracle;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dual::DualN;
use dft_rs::xc::{tao_mo, Family, GridVars, XcFunctional, XcOut};

const POINT: [f64; 7] = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
const POLARIZED: [f64; 7] = [0.11, 0.68, 0.027, -0.039, 0.061, 0.19, 0.42];
// Independent numerical evaluation of the two primary definitions.
const FROZEN: [[[f64; 2]; 2]; 2] = [
    [
        [-0.32083890314213814, -0.03363970000945447],
        [-0.612478445788702, -0.044053614045433044],
    ],
    [
        [-0.3209544126936627, -0.033690179610598815],
        [-0.6127806451806386, -0.044056126002305844],
    ],
];

fn evaluate(functional: &dyn XcFunctional, p: [f64; 7], order: usize) -> XcOut {
    let v = p.map(|x| [x]);
    let vars = GridVars {
        rho: [&v[0], &v[1]],
        sigma: Some([&v[2], &v[3], &v[4]]),
        tau: Some([&v[5], &v[6]]),
    };
    let mut out = XcOut::default();
    out.prepare(1, order, Family::MetaGga);
    functional.eval_add(&vars, order, 1.0, &mut out).unwrap();
    out
}

#[test]
fn independent_terms_and_uniform_spin_and_one_electron_limits() {
    for (revised, exchange, correlation) in [
        (false, tao_mo::tm_x(), tao_mo::tm_c()),
        (true, tao_mo::revtm_x(), tao_mo::revtm_c()),
    ] {
        for (index, point) in [POINT, POLARIZED].into_iter().enumerate() {
            let expected = oracle::terms(point, revised);
            let swapped = [
                point[1], point[0], point[4], point[3], point[2], point[6], point[5],
            ];
            for (term, part) in [&*exchange, &*correlation].into_iter().enumerate() {
                let reference = FROZEN[usize::from(revised)][index][term];
                let energy = evaluate(part, point, 0).e[0];
                assert!((expected[term] - reference).abs() < 2e-14);
                assert!((energy - reference).abs() < 2e-14);
                assert!((evaluate(part, swapped, 0).e[0] - energy).abs() < 2e-14);
                println!(
                    "{} {:?} {point:?} energy {energy:.16e} defect {:.3e}",
                    part.name(),
                    part.kind(),
                    (energy - reference).abs()
                );
            }
        }
        let pi = std::f64::consts::PI;
        let density: f64 = 0.21;
        let kf = (6.0 * pi * pi * density).cbrt();
        let tau = 0.3 * density * kf * kf;
        let uniform = [density, density, 0.0, 0.0, 0.0, tau, tau];
        let uniform_x = -1.5 / pi * density * kf;
        assert!((evaluate(&*exchange, uniform, 0).e[0] - uniform_x).abs() < 2e-14);
        let uniform_reference = oracle::terms(uniform, revised);
        assert!((evaluate(&*correlation, uniform, 0).e[0] - uniform_reference[1]).abs() < 2e-14);

        // Radial Simpson quadrature on the analytic one-electron H 1s density.
        let panels = 4096;
        let dr = 25.0 / f64::from(panels);
        let mut integral = 0.0;
        let mut self_interaction: f64 = 0.0;
        for index in 0..=panels {
            let r = dr * f64::from(index);
            let rho = (-2.0 * r).exp() / pi;
            let point = [rho, 0.0, 4.0 * rho * rho, 0.0, 0.0, 0.5 * rho, 0.0];
            let weight = if index == 0 || index == panels {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            integral += weight * 4.0 * pi * r * r * evaluate(&*exchange, point, 0).e[0];
            self_interaction = self_interaction.max(evaluate(&*correlation, point, 0).e[0].abs());
        }
        let e_x = integral * dr / 3.0;
        assert!((e_x + 0.3125).abs() < 2e-7);
        assert!(self_interaction < 2e-15);
        println!(
            "{} H(1s) exchange {e_x:.16e}, max |correlation density| {self_interaction:.3e}",
            exchange.name()
        );
    }
}

fn third(out: &XcOut, i: usize, j: usize, k: usize) -> f64 {
    let pair = |a, b| (0..a).map(|p| 3 - p).sum::<usize>() + b - a;
    match (i, j, k) {
        (0..=1, 0..=1, 0..=1) => out.v3rho3[i + j + k][0],
        (0..=1, 0..=1, 2..=4) => out.v3rho2sigma[3 * (i + j) + k - 2][0],
        (0..=1, 2..=4, 2..=4) => out.v3rhosigma2[6 * i + pair(j - 2, k - 2)][0],
        (2..=4, 2..=4, 2..=4) => {
            let index = (2..5)
                .flat_map(|a| (a..5).flat_map(move |b| (b..5).map(move |c| (a, b, c))))
                .position(|v| v == (i, j, k))
                .unwrap();
            out.v3sigma3[index][0]
        }
        (0..=1, 0..=1, 5..=6) => out.v3rho2tau[2 * (i + j) + k - 5][0],
        (0..=1, 2..=4, 5..=6) => out.v3rhosigmatau[6 * i + 2 * (j - 2) + k - 5][0],
        (0..=1, 5..=6, 5..=6) => out.v3rhotau2[3 * i + j + k - 10][0],
        (2..=4, 2..=4, 5..=6) => out.v3sigma2tau[2 * pair(i - 2, j - 2) + k - 5][0],
        (2..=4, 5..=6, 5..=6) => out.v3sigmatau2[3 * (i - 2) + j + k - 10][0],
        (5..=6, 5..=6, 5..=6) => out.v3tau3[i + j + k - 15][0],
        _ => panic!("invalid third-derivative block"),
    }
}

type Jet = DualN<DualN<f64, 7>, 7>;

fn oracle_jet(point: [f64; 7], revised: bool, term: usize) -> Jet {
    let vars = std::array::from_fn(|i| Jet::variable(DualN::variable(point[i], i), i));
    oracle::terms(vars, revised)[term]
}

fn fit_component(
    functional: &dyn XcFunctional,
    revised: bool,
    term: usize,
    failures: &mut Vec<String>,
) {
    let steps = [
        ([0.12, 0.06, 0.03, 0.015], stencils::Stencil::Second),
        ([0.18, 0.09, 0.045, 0.0225], stencils::Stencil::Second),
        ([0.06, 0.03, 0.015, 0.0075], stencils::Stencil::Second),
        ([0.12, 0.06, 0.03, 0.015], stencils::Stencil::Fourth),
    ];
    let scales = [0.13, 0.08, 0.09, 0.04, 0.06, 0.12, 0.09];
    let out = evaluate(functional, POINT, 3);
    let (first, second) = out.potential_and_kernel(0, Family::MetaGga).unwrap();
    let closed = oracle_jet(POINT, revised, term);
    let mut entries = Vec::new();
    let mut zeros = 0;
    let mut add = |indices: &[usize], value| {
        // Exchange adds independent up/down energies and ignores sigma_ab.
        let is_separable = [[0, 2, 5], [1, 4, 6]]
            .iter()
            .any(|active| indices.iter().all(|i| active.contains(i)));
        if term == 0 && !is_separable {
            assert_eq!(value, 0.0);
            zeros += 1;
        } else {
            let (&axis, lower) = indices.split_last().unwrap();
            entries.push((
                format!("{} {:?} D{indices:?}", functional.name(), functional.kind()),
                axis,
                lower.to_vec(),
                value,
            ));
        }
    };
    for i in 0..7 {
        assert!((first[i] - closed.im[i].re).abs() < 2e-12);
        add(&[i], first[i]);
        for j in i..7 {
            assert!((second[i][j] - closed.im[i].im[j]).abs() < 2e-11);
            add(&[i, j], second[i][j]);
            for k in j..7 {
                add(&[i, j, k], third(&out, i, j, k));
            }
        }
    }
    println!(
        "{} {:?}: {} fitted derivatives, {zeros} structural zeros",
        functional.name(),
        functional.kind(),
        entries.len()
    );
    let labels: Vec<_> = entries.iter().map(|entry| entry.0.clone()).collect();
    stencils::fit_components_staged(
        &labels,
        &steps,
        |step_sizes, stencil| {
            let samples: [[[Jet; 4]; 7]; 4] = std::array::from_fn(|h| {
                std::array::from_fn(|axis| {
                    [1.0, -1.0, 2.0, -2.0].map(|sign| {
                        let mut point = POINT;
                        point[axis] += sign * step_sizes[h] * scales[axis];
                        oracle_jet(point, revised, term)
                    })
                })
            });
            entries
                .iter()
                .map(|(_, axis, lower, value)| {
                    std::array::from_fn(|h| {
                        let [plus, minus, plus2, minus2] =
                            samples[h][*axis].map(|jet| match lower.as_slice() {
                                [] => jet.re.re,
                                [i] => jet.im[*i].re,
                                [i, j] => jet.im[*i].im[*j],
                                _ => unreachable!("invalid lower derivative"),
                            });
                        (stencil.first(plus, minus, plus2, minus2, step_sizes[h] * scales[*axis])
                            - value)
                            .abs()
                    })
                })
                .collect()
        },
        failures,
    );
}

#[test]
fn all_independent_local_derivatives_through_third_order() {
    let mut failures = Vec::new();
    for (revised, exchange, correlation) in [
        (false, tao_mo::tm_x(), tao_mo::tm_c()),
        (true, tao_mo::revtm_x(), tao_mo::revtm_c()),
    ] {
        fit_component(&*exchange, revised, 0, &mut failures);
        fit_component(&*correlation, revised, 1, &mut failures);
    }
    components::assert_no_failures("TM and revTM local derivatives", &failures);
}
