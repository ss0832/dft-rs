// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! KCIS model and hybrid component, molecular and local response acceptance.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/kcis_oracle.rs"]
mod oracle;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::dual::DualN;
use dft_rs::gradient::assemble_gradient;
use dft_rs::grid::eval::eval_aos;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::scf::{run_scf, EriMode, Reference, ScfOptions, ScfResult};
use dft_rs::xc::{
    kcis, resolve, Family, FunctionalSpec, GridVars, HybridParams, ResolvedFunctional, XcOut,
};
use dft_rs::{Atom, Molecule, Vec3};
use std::sync::Arc;

const POINT: [f64; 7] = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
const RECIPES: [(&str, Option<&str>, f64); 3] = [
    ("kcis", None, 0.0),
    ("pbe1kcis", Some("pbe"), 0.22),
    ("mpw1kcis", Some("mpw91"), 0.15),
];

fn recipe(name: &str) -> ResolvedFunctional {
    if name == "kcis" {
        ResolvedFunctional::custom(
            name,
            vec![(1.0, Arc::from(kcis::kcis_c()))],
            HybridParams::PURE,
            Family::MetaGga,
        )
    } else {
        resolve(&FunctionalSpec::Combined(name.into())).unwrap()
    }
}

fn evaluate(functional: &ResolvedFunctional, p: [f64; 7], order: usize) -> XcOut {
    let v = p.map(|x| [x]);
    let vars = GridVars {
        rho: [&v[0], &v[1]],
        sigma: Some([&v[2], &v[3], &v[4]]),
        tau: Some([&v[5], &v[6]]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, order, &mut out).unwrap();
    out
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
        _ => panic!("invalid derivative block"),
    }
}

type Jet = DualN<DualN<f64, 7>, 7>;

fn oracle_jet(point: [f64; 7], exchange: Option<&str>, exact: f64) -> Jet {
    let point = std::array::from_fn(|i| Jet::variable(DualN::variable(point[i], i), i));
    let [x, c] = oracle::terms(point, exchange);
    x * Jet::constant(DualN::constant(1.0 - exact)) + c
}

#[test]
fn independent_component_energies_and_physical_limits() {
    for (name, exchange, exact) in RECIPES {
        let f = recipe(name);
        assert_eq!(f.family, Family::MetaGga);
        assert_eq!(f.hybrid.cx, exact);
        assert_eq!(f.hybrid.clr, 0.0);
        assert_eq!(f.hybrid.omega, 0.0);
        for point in [POINT, [0.11, 0.68, 0.027, -0.039, 0.061, 0.19, 0.42]] {
            let computed = evaluate(&f, point, 2);
            let [x, c] = oracle::terms(point, exchange);
            assert!((computed.e[0] - (1.0 - exact) * x - c).abs() < 2e-14);
            for ((weight, part), expected) in f.parts.iter().zip(if exchange.is_some() {
                vec![(1.0 - exact) * x, c]
            } else {
                vec![c]
            }) {
                let term = ResolvedFunctional::custom(
                    "component",
                    vec![(*weight, Arc::clone(part))],
                    HybridParams::PURE,
                    Family::MetaGga,
                );
                let error = (evaluate(&term, point, 0).e[0] - expected).abs();
                assert!(
                    error < 2e-14,
                    "{name} {} term defect {error:.3e}",
                    part.name()
                );
            }
            let split = [
                point[1], point[0], point[4], point[3], point[2], point[6], point[5],
            ];
            assert!((evaluate(&f, split, 0).e[0] - computed.e[0]).abs() < 2e-14);
        }
    }

    let kcis = recipe("kcis");
    let uniform = resolve(&FunctionalSpec::Combined("spw92".into())).unwrap();
    let slater = resolve(&FunctionalSpec::Split {
        x: "slater".into(),
        c: "none".into(),
    })
    .unwrap();
    for rho in [0.001, 0.2, 4.0] {
        let point = [rho, rho, 0.0, 0.0, 0.0, 0.2, 0.2];
        let expected = evaluate(&uniform, point, 0).e[0] - evaluate(&slater, point, 0).e[0];
        assert!((evaluate(&kcis, point, 0).e[0] - expected).abs() < 2e-14);
        let sigma = 0.6 * rho * rho;
        let tau_w = sigma / (8.0 * rho);
        let one_electron = [rho, 0.0, sigma, 0.0, 0.0, tau_w, 0.0];
        assert!(evaluate(&kcis, one_electron, 2).e[0].abs() < 1e-14);
    }
    // A large gap gives a negative 1/G tail; use gradients that keep the
    // total density modulation identical while removing the SIC term.
    let gap_tail = |gap: f64| {
        let sigma = 0.08 * gap;
        let point = [
            0.05,
            0.05,
            sigma / 4.0,
            sigma / 4.0,
            sigma / 4.0,
            1e25,
            1e25,
        ];
        evaluate(&kcis, point, 0).e[0] * gap / 0.1
    };
    let asymptote = -0.06483 * (0.3 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0);
    assert!((gap_tail(1e7) - asymptote).abs() < 1e-6);
}

#[test]
fn every_independent_local_quadrature_derivative_through_third_order() {
    let stages = [
        ([0.12, 0.06, 0.03, 0.015], stencils::Stencil::Second),
        ([0.06, 0.03, 0.015, 0.0075], stencils::Stencil::Second),
        ([0.18, 0.09, 0.045, 0.0225], stencils::Stencil::Second),
        ([0.12, 0.06, 0.03, 0.015], stencils::Stencil::Fourth),
    ];
    let scales = [0.13, 0.08, 0.09, 0.04, 0.06, 0.12, 0.09];
    let mut failures = Vec::new();
    for (name, exchange, exact) in RECIPES {
        let out = evaluate(&recipe(name), POINT, 3);
        let (pot, kernel) = out.potential_and_kernel(0, Family::MetaGga).unwrap();
        let independent = oracle_jet(POINT, exchange, exact);
        let structural_zero = |coordinates: &[usize]| {
            [(5, [0, 2, 5]), (6, [1, 4, 6])]
                .into_iter()
                .any(|(tau, active)| {
                    coordinates.contains(&tau)
                        && coordinates.iter().any(|index| !active.contains(index))
                })
        };
        // Store the lower-order oracle derivative that is differentiated in
        // the selected direction to check each independent tensor component.
        let mut entries = Vec::new();
        for i in 0..7 {
            assert!((pot[i] - independent.im[i].re).abs() < 2e-12);
            entries.push((format!("{name} D[{i}]"), i, (None, None), pot[i]));
            for j in i..7 {
                assert!((kernel[i][j] - independent.im[i].im[j]).abs() < 2e-11);
                if structural_zero(&[i, j]) {
                    assert_eq!(kernel[i][j], 0.0);
                } else {
                    entries.push((
                        format!("{name} D[{i},{j}]"),
                        j,
                        (Some(i), None),
                        kernel[i][j],
                    ));
                }
                for k in j..7 {
                    let value = third(&out, i, j, k);
                    if structural_zero(&[i, j, k]) {
                        assert_eq!(value, 0.0);
                    } else {
                        entries.push((
                            format!("{name} D[{i},{j},{k}]"),
                            k,
                            (Some(i), Some(j)),
                            value,
                        ));
                    }
                }
            }
        }
        let labels = entries
            .iter()
            .map(|entry| entry.0.clone())
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &labels,
            &stages,
            |steps, stencil| {
                let sampled: [[[Jet; 4]; 7]; 4] = std::array::from_fn(|step| {
                    std::array::from_fn(|axis| {
                        [1.0, -1.0, 2.0, -2.0].map(|sign| {
                            let mut point = POINT;
                            point[axis] += sign * steps[step] * scales[axis];
                            oracle_jet(point, exchange, exact)
                        })
                    })
                });
                entries
                    .iter()
                    .map(|(_, axis, lower, expected)| {
                        std::array::from_fn(|step| {
                            let [plus, minus, plus2, minus2] =
                                sampled[step][*axis].map(|v| match lower {
                                    (None, None) => v.re.re,
                                    (Some(i), None) => v.im[*i].re,
                                    (Some(i), Some(j)) => v.im[*i].im[*j],
                                    (None, Some(_)) => unreachable!("invalid lower derivative"),
                                });
                            (stencil.first(plus, minus, plus2, minus2, steps[step] * scales[*axis])
                                - expected)
                                .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("KCIS local density derivatives", &failures);
}

fn molecular_xc(
    basis: &MolBasis,
    grid: &MolecularGrid,
    density: &[Matrix; 2],
    exchange: &str,
) -> [f64; 2] {
    let ao = eval_aos(basis, &grid.points, true);
    let gradient = ao.grad.unwrap();
    let n = basis.nao_sph;
    let mut energy = [0.0; 2];
    for (i, weight) in grid.weights.iter().enumerate() {
        let mut rho = [0.0; 2];
        let mut drho = [[0.0; 3]; 2];
        let mut tau = [0.0; 2];
        for spin in 0..2 {
            for mu in 0..n {
                for nu in 0..n {
                    let p = density[spin][(mu, nu)];
                    rho[spin] += p * ao.phi[(i, mu)] * ao.phi[(i, nu)];
                    for axis in 0..3 {
                        let a = gradient[axis][(i, mu)];
                        let b = gradient[axis][(i, nu)];
                        drho[spin][axis] += p * (a * ao.phi[(i, nu)] + ao.phi[(i, mu)] * b);
                        tau[spin] += 0.5 * p * a * b;
                    }
                }
            }
        }
        let dot = |a: usize, b: usize| {
            (0..3)
                .map(|axis| drho[a][axis] * drho[b][axis])
                .sum::<f64>()
        };
        let point = [
            rho[0],
            rho[1],
            dot(0, 0),
            dot(0, 1),
            dot(1, 1),
            tau[0].max(1e-20),
            tau[1].max(1e-20),
        ];
        for (term, value) in energy.iter_mut().zip(oracle::terms(point, Some(exchange))) {
            *term += weight * value;
        }
    }
    energy
}

const MOLECULES: [(u8, usize, Reference); 2] = [(7, 1, Reference::Rhf), (6, 2, Reference::Uhf)];

fn c1_molecule(z: u8, multiplicity: usize) -> Molecule {
    let positions = [
        [0.13, -0.21, 0.09],
        [1.61, 0.77, -0.29],
        [-1.34, 0.61, 0.33],
        [0.31, -1.74, 0.58],
    ];
    Molecule::new(
        [z, 1, 1, 1]
            .into_iter()
            .zip(positions)
            .map(|(z, [x, y, z_pos])| Atom {
                z,
                position: Vec3::new(x, y, z_pos),
            })
            .collect(),
    )
    .with_multiplicity(multiplicity)
}

fn grid_for(mol: &Molecule) -> MolecularGrid {
    MolecularGrid::build(
        mol,
        &GridOptions {
            level: 2,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap()
}

fn scf_options(reference: Reference) -> ScfOptions {
    ScfOptions {
        reference,
        e_tol: 1e-12,
        p_tol: 1e-10,
        conv_tol_grad: 1e-9,
        eri_mode: EriMode::InCore { limit_mb: 128 },
        ..ScfOptions::default()
    }
}

fn frozen_orbital_residual(state: &ScfResult) -> f64 {
    (0..2)
        .map(|spin| {
            let fmo = state.mo_coeff[spin]
                .transpose_matmul(&state.fock[spin].matmul(&state.mo_coeff[spin]));
            let occupied = [state.n_alpha, state.n_beta][spin];
            (0..occupied)
                .flat_map(|i| (occupied..fmo.cols).map(move |v| (i, v)))
                .map(|(i, v)| fmo[(i, v)].abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

#[test]
fn converged_c1_molecular_energies_reconstruct_from_all_physical_terms() {
    let set = data::load("sto-3g").unwrap();
    for (z, multiplicity, reference) in MOLECULES {
        let mol = c1_molecule(z, multiplicity);
        let basis = MolBasis::build(&mol, &set).unwrap();
        let grid = grid_for(&mol);
        let options = scf_options(reference);
        let n = basis.nao_sph;
        let h_t = kinetic_matrix(&basis);
        let h_v = nuclear_matrix(&basis, &mol, None);
        let nrep = mol.nuclear_repulsion();
        let eri = eri_incore(&basis, CoulombKernel::Coulomb, 128).unwrap();
        for (name, exchange, exact) in RECIPES.into_iter().skip(1) {
            let f = recipe(name);
            let state = run_scf(&mol, &basis, &options, &f, Some(&grid)).unwrap();
            assert!(state.converged, "{name}/{z} after {}", state.iterations);
            let mut total = state.density[0].clone();
            for (a, b) in total
                .as_mut_slice()
                .iter_mut()
                .zip(state.density[1].as_slice())
            {
                *a += b;
            }
            let kinetic = h_t.frobenius_dot(&total);
            let attraction = h_v.frobenius_dot(&total);
            let mut coulomb = 0.0;
            let mut exchange_hf = 0.0;
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        for d in 0..n {
                            let integral = eri.get(a, b, c, d);
                            coulomb += 0.5 * total[(a, b)] * total[(c, d)] * integral;
                            for p in &state.density {
                                exchange_hf -= 0.5 * exact * p[(a, c)] * p[(b, d)] * integral;
                            }
                        }
                    }
                }
            }
            let [x, c] = molecular_xc(&basis, &grid, &state.density, exchange.unwrap());
            let exchange_local = (1.0 - exact) * x;
            let reconstruction =
                nrep + kinetic + attraction + coulomb + exchange_hf + exchange_local + c;
            let defect = (state.energy - reconstruction).abs();
            assert!(
                defect < 2e-10,
                "{name}/{z} reconstruction defect {defect:.4e}"
            );
            assert!((state.e_xc - exchange_local - c).abs() < 2e-11);
            let residual = frozen_orbital_residual(&state);
            assert!(residual < 1e-9, "{name}/{z} residual {residual:.4e}");
            println!("{name} Z={z} x_local={exchange_local:.15} c={c:.15} x_exact={exchange_hf:.15} E={:.15} orbital={residual:.5e} defect={defect:.5e} iters={}", state.energy, state.iterations);
        }
    }
}

#[test]
fn every_c1_molecular_nuclear_gradient_component_has_four_step_fits() {
    let set = data::load("sto-3g").unwrap();
    let mut failures = Vec::new();
    for (z, multiplicity, reference) in MOLECULES {
        let molecule = c1_molecule(z, multiplicity);
        let basis = MolBasis::build(&molecule, &set).unwrap();
        let grid = grid_for(&molecule);
        let options = scf_options(reference);
        for (name, _, _) in RECIPES.into_iter().skip(1) {
            let f = recipe(name);
            let state = run_scf(&molecule, &basis, &options, &f, Some(&grid)).unwrap();
            assert!(state.converged);
            let gradient =
                assemble_gradient(&molecule, &basis, &f, Some(&grid), &state, &options).unwrap();
            let exact: Vec<_> = gradient.iter().flat_map(|v| v.to_array()).collect();
            let net_gradient = gradient.iter().copied().fold(Vec3::zero(), |a, b| a + b);
            assert!(net_gradient.norm() < 1e-9);
            let labels = (0..exact.len())
                .map(|i| format!("{name} Z={z} nuclear[{},{}]", i / 3, i % 3))
                .collect::<Vec<_>>();
            stencils::fit_components_staged(
                &labels,
                &[
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    (0..exact.len())
                        .map(|i| {
                            let energy = |h| {
                                let mut m = molecule.clone();
                                let mut position = m.atoms[i / 3].position.to_array();
                                position[i % 3] += h;
                                m.atoms[i / 3].position =
                                    Vec3::new(position[0], position[1], position[2]);
                                let b = MolBasis::build(&m, &set).unwrap();
                                let g = grid_for(&m);
                                let scf = run_scf(&m, &b, &options, &f, Some(&g)).unwrap();
                                assert!(
                                    scf.converged && frozen_orbital_residual(&scf) < 1e-9,
                                    "{name} Z={z} component {i} step {h}"
                                );
                                scf.energy
                            };
                            steps.map(|h| {
                                let plus2 = if stencil == stencils::Stencil::Fourth {
                                    energy(2.0 * h)
                                } else {
                                    0.0
                                };
                                let minus2 = if stencil == stencils::Stencil::Fourth {
                                    energy(-2.0 * h)
                                } else {
                                    0.0
                                };
                                (stencil.first(energy(h), energy(-h), plus2, minus2, h) - exact[i])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            println!(
                "{name} Z={z} translational defect {:.6e} Hartree/Bohr",
                net_gradient.norm()
            );
        }
    }
    components::assert_no_failures("KCIS molecular gradients", &failures);
}
