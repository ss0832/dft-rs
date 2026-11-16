// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::integrals::dirac::{
    electronic_current_exchange, restricted_kinetic_balance, DiracCurrentInteraction,
};
use crate::properties::no_pair::stationary::DiracInteraction;
use std::f64::consts::PI;

const PAULI: [[[Complex; 2]; 2]; 3] = [
    [[ZERO, ONE], [ONE, ZERO]],
    [[ZERO, NEG_IM], [IM, ZERO]],
    [[ONE, ZERO], [ZERO, NEG]],
];

type Products = [[Vec<(usize, Complex)>; 4]; 4];

fn current_products(c: f64) -> [Products; 3] {
    // Expand every AO in the four physical Dirac components, with a
    // separate s lane and three unnormalized Cartesian p lanes.
    let expansion: [[[Complex; 4]; 4]; 4] = std::array::from_fn(|basis| {
        let mut components = [[ZERO; 4]; 4];
        if basis < 2 {
            components[basis][0] = ONE;
        } else {
            for (spin, lanes) in components[2..].iter_mut().enumerate() {
                for (axis, sigma) in PAULI.iter().enumerate() {
                    lanes[1 + axis] = IM * sigma[spin][basis - 2] * (0.5 / c);
                }
            }
        }
        components
    });
    PAULI.map(|sigma| {
        let alpha: [[Complex; 4]; 4] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if (i < 2) == (j < 2) {
                    ZERO
                } else {
                    sigma[i % 2][j % 2]
                }
            })
        });
        std::array::from_fn(|p| {
            std::array::from_fn(|q| {
                let mut products = Vec::new();
                for a in 0..4 {
                    for b in 0..4 {
                        let mut z = ZERO;
                        for (i, row) in alpha.iter().enumerate() {
                            for (j, &value) in row.iter().enumerate() {
                                z += expansion[p][i][a].conj() * value * expansion[q][j][b];
                            }
                        }
                        if z != ZERO {
                            let slot = if a == 0 { 3 + b - 1 } else { a - 1 };
                            products.push((slot, z));
                        }
                    }
                }
                products
            })
        })
    })
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Clone, Copy)]
struct Variable {
    kind: usize,
    axis: usize,
    mean: f64,
}

fn primitive_current_interaction(
    shells: [&AnalyticShell<f64>; 4],
    interaction: DiracCurrentInteraction,
    quadrature: &[(f64, f64)],
) -> [[[[f64; 6]; 6]; 3]; 3] {
    let exponents = shells.map(|shell| shell.exponents[0]);
    let positions = shells.map(|shell| shell.center);
    let [a, b, c, d] = exponents;
    let p = a + b;
    let q = c + d;
    let rho = p * q / (p + q);
    let pair_centre = |offset: usize, width: f64| {
        std::array::from_fn::<_, 3, _>(|axis| {
            (exponents[offset] * positions[offset][axis]
                + exponents[offset + 1] * positions[offset + 1][axis])
                / width
        })
    };
    let centre = [pair_centre(0, p), pair_centre(2, q)];
    let delta = std::array::from_fn(|axis| centre[0][axis] - centre[1][axis]);
    let ab = std::array::from_fn(|axis| positions[0][axis] - positions[1][axis]);
    let cd = std::array::from_fn(|axis| positions[2][axis] - positions[3][axis]);
    let prefactor = 2.0
        * (rho / PI).sqrt()
        * (PI * PI / (p * q)).powf(1.5)
        * (-a * b / p * dot(ab, ab) - c * d / q * dot(cd, cd)).exp()
        * shells
            .iter()
            .map(|shell| shell.coefficients[0])
            .product::<f64>();
    let mut out = [[[[0.0; 6]; 6]; 3]; 3];
    // Laplace representation of 1/r. Changing the integration variable
    // to t=sqrt(rho)*u/sqrt(1-u^2) cancels the Gaussian determinant.
    // The Breit tensor in this quadrature is -delta/2-t^2 U_i U_j.
    for &(u, weight) in quadrature {
        let u2 = u * u;
        let relative_variance = (1.0 - u2) / (2.0 * rho);
        let variance_centre = 1.0 / (2.0 * (p + q));
        let variance = |x: Variable, y: Variable| {
            if x.axis != y.axis {
                return 0.0;
            }
            let coefficients = [q / (p + q), -p / (p + q), 1.0];
            let common = if x.kind < 2 && y.kind < 2 {
                variance_centre
            } else {
                0.0
            };
            common + coefficients[x.kind] * coefficients[y.kind] * relative_variance
        };
        let mean_product = |x: Variable, y: Variable| x.mean * y.mean + variance(x, y);
        let fourth = |[x, y, z, w]: [Variable; 4]| {
            mean_product(x, y) * mean_product(z, w)
                + variance(x, z) * mean_product(y, w)
                + variance(x, w) * mean_product(y, z)
                + variance(y, z) * x.mean * w.mean
                + variance(y, w) * x.mean * z.mean
        };
        let shifted = |slot: usize, electron: usize| {
            let index = 2 * electron + slot / 3;
            let axis = slot % 3;
            let translation = if electron == 0 { -q } else { p } / (p + q);
            let mean =
                centre[electron][axis] + translation * u2 * delta[axis] - positions[index][axis];
            (
                Variable {
                    kind: electron,
                    axis,
                    mean,
                },
                2.0 * exponents[index],
            )
        };
        let relative = |axis: usize| Variable {
            kind: 2,
            axis,
            mean: (1.0 - u2) * delta[axis],
        };
        let t2 = rho * u2 / (1.0 - u2);
        let scale = weight * prefactor * (-rho * u2 * dot(delta, delta)).exp();
        for (i, row) in out.iter_mut().enumerate() {
            for (j, block) in row.iter_mut().enumerate() {
                for (l, values) in block.iter_mut().enumerate() {
                    let (x, first) = shifted(l, 0);
                    for (r, entry) in values.iter_mut().enumerate() {
                        let (y, second) = shifted(r, 1);
                        let diagonal = if i == j { mean_product(x, y) } else { 0.0 };
                        let value = match interaction {
                            DiracCurrentInteraction::Gaunt => -diagonal,
                            DiracCurrentInteraction::Breit => {
                                -0.5 * diagonal - t2 * fourth([x, y, relative(i), relative(j)])
                            }
                        };
                        *entry += first * second * scale * value;
                    }
                }
            }
        }
    }
    out
}

fn dense_current(
    shells: &[AnalyticShell<f64>; 2],
    c: f64,
    interaction: DiracCurrentInteraction,
) -> Vec<Complex> {
    let (points, weights) = crate::grid::gauss_legendre(48);
    let quadrature: Vec<_> = points
        .into_iter()
        .zip(weights)
        .map(|(x, w)| (0.5 * (1.0 + x), 0.5 * w))
        .collect();
    let products = current_products(c);
    let mut dense = vec![ZERO; N.pow(4)];
    for quartet in 0..16 {
        let spatial = std::array::from_fn::<_, 4, _>(|axis| (quartet >> axis) & 1);
        let tensors =
            primitive_current_interaction(spatial.map(|i| &shells[i]), interaction, &quadrature);
        for p in 0..4 {
            for q in 0..4 {
                for r in 0..4 {
                    for s in 0..4 {
                        let mut value = ZERO;
                        for (i, row) in tensors.iter().enumerate() {
                            for (j, block) in row.iter().enumerate() {
                                for &(a, x) in &products[i][p][q] {
                                    for &(b, y) in &products[j][r][s] {
                                        value += x * y * block[a][b];
                                    }
                                }
                            }
                        }
                        let [p, q, r, s] =
                            std::array::from_fn(|i| 2 * [p, q, r, s][i] + spatial[i]);
                        dense[((p * N + q) * N + r) * N + s] = value;
                    }
                }
            }
        }
    }
    dense
}

#[test]
fn dirac_gaunt_and_breit_match_independent_all_spin_and_density_oracles() {
    let shells = shells();
    let density = sample_density(0.47, 2);
    let one = sample_density(0.47, 1);
    let quadratic = sample_density(0.81, 3);
    let directions = hermitian_directions();
    for model in [
        DiracCurrentInteraction::Gaunt,
        DiracCurrentInteraction::Breit,
    ] {
        let dense = dense_current(&shells, C, model);
        let fields = electronic_current_exchange(&shells, &density, C, model).unwrap();
        let (j, k) = contract(&dense, &density);
        let direct = difference(&fields.direct, &j);
        let exchange = difference(&fields.exchange, &k);
        assert!(
            direct.max(exchange) < 1e-13,
            "{model:?} J {direct:e} K {exchange:e}"
        );
        assert!(fields.hermiticity_defect < 1e-13);
        let single = electronic_current_exchange(&shells, &one, C, model).unwrap();
        let self_cancellation = (single.direct_energy + single.exchange_energy).abs();
        assert!(self_cancellation < 1e-13);
        let scaled = electronic_current_exchange(&shells, &density, 2.0 * C, model).unwrap();
        let scaling = (scaled.exchange_energy * 4.0 - fields.exchange_energy)
            .abs()
            .max((scaled.direct_energy * 4.0 - fields.direct_energy).abs());
        assert!(scaling < 1e-13);
        let mut fock = fields.direct;
        fock.add_scaled(&fields.exchange, -1.0);
        let names: Vec<_> = directions
            .iter()
            .map(|(name, _)| format!("{model:?} {name}"))
            .collect();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[([0.06, 0.04, 0.03, 0.02], Stencil::Second)],
            |steps, stencil| {
                directions
                    .iter()
                    .map(|(_, direction)| {
                        let sample = |h: f64| {
                            let mut p = density.clone();
                            p.add_scaled(direction, h);
                            p.add_scaled(&quadratic, h * h);
                            let (mut j, k) = contract(&dense, &p);
                            j.add_scaled(&k, -1.0);
                            0.5 * p.trace_product(&j)
                        };
                        steps.map(|h| {
                            (stencil.first(sample(h), sample(-h), 0.0, 0.0, h)
                                - direction.trace_product(&fock))
                            .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures(
            "all independent current-interaction density entries",
            &failures,
        );
        println!("{model:?} direct={:.17e} exchange={:.17e} direct_defect={direct:.17e} exchange_defect={exchange:.17e} hermiticity={:.17e} self_cancellation={self_cancellation:.17e} scaling_defect={scaling:.17e}",
            fields.direct_energy, fields.exchange_energy, fields.hermiticity_defect);
    }
}

#[test]
fn two_electron_fixed_sector_breit_scf_matches_dense_oracle() {
    let shells = shells();
    let nuclei = [
        GaussianNucleus {
            center: [-0.21, 0.12, -0.35],
            charge: 2.0,
            rms_radius: 0.17,
        },
        GaussianNucleus {
            center: [0.48, -0.73, 0.71],
            charge: 1.0,
            rms_radius: 0.09,
        },
    ];
    let backend = DiracCoulombBackend::new(&shells, &nuclei, C, 2)
        .unwrap()
        .with_interaction(DiracInteraction::Breit);
    let mut integrals = dense_integrals(&shells, C);
    let correction = dense_current(&shells, C, DiracCurrentInteraction::Breit);
    for (value, delta) in integrals.iter_mut().zip(&correction) {
        *value += *delta;
    }
    let classical = restricted_kinetic_balance(&shells, &nuclei, C, 0)
        .unwrap()
        .nuclear_repulsion;
    let oracle = Dense {
        core: &backend,
        integrals: &integrals,
        classical,
    };
    let config = SpinorScfConfig {
        overlap_cutoff: 1e-9,
        orbital_tolerance: 1e-11,
        energy_tolerance: 2e-13,
        density_tolerance: 1e-11,
        ..SpinorScfConfig::default()
    };
    let space = backend.electronic_space(config.overlap_cutoff).unwrap();
    let state = run_spinor_scf(&NoPairBackend::new(&space, &backend).unwrap(), &config).unwrap();
    let reference = run_spinor_scf(&NoPairBackend::new(&space, &oracle).unwrap(), &config).unwrap();
    assert!(state.converged && reference.converged);
    let energy_defect = (state.fields.energy - reference.fields.energy).abs();
    assert!(energy_defect < 2e-12);
    let density = space.lift_density(&state.density[0]).unwrap();
    let term = backend.current_fields(&density).unwrap().unwrap();
    let (mut dense_kernel, exchange) = contract(&integrals, &density);
    dense_kernel.add_scaled(&exchange, -1.0);
    let kernel_defect = difference(&backend.density_kernel(&density).unwrap(), &dense_kernel);
    assert!(kernel_defect < 1e-13);
    let (j, k) = contract(&correction, &density);
    assert!((term.direct_energy - 0.5 * density.trace_product(&j)).abs() < 1e-13);
    assert!((term.exchange_energy + 0.5 * density.trace_product(&k)).abs() < 1e-13);
    assert!(term.exchange_energy.abs() > 1e-7);
    println!(
        "Dirac Breit SCF energy={:.17e} reference={:.17e} energy_defect={energy_defect:.17e} kernel_defect={kernel_defect:.17e} residual={:.17e} current_direct={:.17e} current_exchange={:.17e} iterations={}",
        state.fields.energy, reference.fields.energy, state.orbital_residual,
        term.direct_energy, term.exchange_energy, state.iterations,
    );
}
