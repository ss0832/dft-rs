// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of the complete-active-space SCF and of multiconfiguration
//! pair-density functional theory.
//!
//! Oracles:
//! - a full active space reproduces an independent dense determinant CI of
//!   the same molecular-orbital integrals, built here by applying the
//!   spin-orbital Hamiltonian to occupation bit strings;
//! - CAS(0,0) orbital optimization is the restricted Hartree-Fock energy;
//!   CAS(2,2) of H2 at 20 Bohr is twice the hydrogen atom;
//! - the orbital gradient passes an independent four-step fit of central
//!   energy differences along single rotations (validation only);
//! - for a single determinant, the translated local on-top functional is
//!   the Kohn-Sham functional of the same spin densities, closed and open
//!   shell; the gradient-corrected translated and fully translated energies
//!   equal an independent evaluation of the translation on the grid;
//! - MC-PDFT of stretched H2 is twice that of the hydrogen atom;
//! - the on-top potential of the linearized method passes an independent
//!   four-step fit of central differences of the on-top energy along the
//!   difference of two state density matrices (fully translated, smooth);
//! - the hybrid mixing, the linearized Hamiltonian at its expansion point and
//!   the one-state limits of the multistate forms are exact identities.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::method::{
    prepare_molecular, run_molecular_reference, MolecularMethod, MolecularSetup,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::eval::eval_aos;
use dft_rs::grid::MolecularGrid;
use dft_rs::integrals::{eri_incore, CoulombKernel, EriTensor};
use dft_rs::mcscf::casscf::{CasOperators, CasProblem, CasSpec};
use dft_rs::mcscf::pdft::{on_top_energy, OnTopDensity, OnTopFunctional, Translation};
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, XcOut};
use dft_rs::Matrix;

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn base(extra: &[(&'static str, Value)]) -> Vec<(&'static str, Value)> {
    let mut pairs = vec![
        ("basis", Value::Str("sto-3g".into())),
        ("xc", Value::Str("hf".into())),
        ("conv_tol", Value::Float(1e-12)),
        ("conv_tol_dp", Value::Float(1e-11)),
    ];
    for (k, v) in extra {
        pairs.retain(|(key, _)| key != k);
        pairs.push((k, v.clone()));
    }
    pairs
}

fn ints(values: &[i64]) -> Value {
    Value::Array(values.iter().map(|v| Value::Int(*v)).collect())
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn water() -> System {
    System::molecule(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.23],
            [0.0, 1.4309, -0.8867],
            [0.12, -1.4809, -0.9067],
        ],
    )
}

#[test]
fn active_space_driver_rejects_inconsistent_core_and_field_models() {
    let nacl = System::molecule(vec![11, 17], vec![[0.01, -0.17, 0.23], [0.21, 0.34, 4.42]]);
    let ecp = controls(&base(&[
        ("basis", Value::Str("lanl2dz".into())),
        ("cas", ints(&[2, 2])),
    ]));
    let field = controls(&base(&[
        ("cas", ints(&[2, 2])),
        ("derivative_order", Value::Int(1)),
        (
            "field",
            Value::Array(vec![
                Value::Float(0.013),
                Value::Float(-0.015),
                Value::Float(0.02),
            ]),
        ),
    ]));
    let zero_field = controls(&base(&[
        ("cas", ints(&[2, 2])),
        ("derivative_order", Value::Int(1)),
        ("field", Value::Array(vec![Value::Float(0.0); 3])),
    ]));
    for feature in ["casscf", "mcpdft"] {
        for (system, input, scope) in [
            (&nacl, &ecp, "effective core potentials"),
            (&water(), &field, "static electric field"),
        ] {
            assert!(matches!(
                run(feature, system, input),
                Err(dft_rs::DftError::Unsupported { feature: detail, .. }) if detail.contains(scope)
            ));
        }
        let reference = run(feature, &hydrogen_pair(1.4), &zero_field).unwrap();
        assert!(reference.get("gradients").is_some());
    }
}

fn lithium_hydride() -> System {
    System::molecule(vec![3, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]])
}

fn hydrogen_pair(distance: f64) -> System {
    System::molecule(vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
}

fn reference(
    system: &System,
    extra: &[(&'static str, Value)],
) -> (MolecularSetup, dft_rs::ScfResult) {
    let method = MolecularMethod::from_controls(&controls(&base(extra))).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let scf = run_molecular_reference(&setup).unwrap().scf;
    assert!(scf.converged);
    (setup, scf)
}

/// Independent dense determinant CI: spin orbitals 2p (alpha) and 2p+1
/// (beta), determinants as bit strings, the Hamiltonian applied by explicit
/// creation and annihilation operators.
fn dense_ci(
    h: &Matrix,
    eri: &[f64],
    n: usize,
    n_alpha: usize,
    n_beta: usize,
    e_nuc: f64,
) -> Vec<f64> {
    let g = |p: usize, q: usize, r: usize, s: usize| eri[((p * n + q) * n + r) * n + s];
    let mut dets = Vec::new();
    for a in 0u64..(1 << n) {
        if a.count_ones() as usize != n_alpha {
            continue;
        }
        for b in 0u64..(1 << n) {
            if b.count_ones() as usize != n_beta {
                continue;
            }
            let mut d = 0u64;
            for p in 0..n {
                if (a >> p) & 1 == 1 {
                    d |= 1 << (2 * p);
                }
                if (b >> p) & 1 == 1 {
                    d |= 1 << (2 * p + 1);
                }
            }
            dets.push(d);
        }
    }
    let annihilate = |d: u64, i: usize| -> Option<(u64, f64)> {
        if (d >> i) & 1 == 0 {
            return None;
        }
        let sign = if (d & ((1 << i) - 1)).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        Some((d & !(1 << i), sign))
    };
    let create = |d: u64, i: usize| -> Option<(u64, f64)> {
        if (d >> i) & 1 == 1 {
            return None;
        }
        let sign = if (d & ((1 << i) - 1)).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        Some((d | (1 << i), sign))
    };
    let dim = dets.len();
    let index = |d: u64| dets.iter().position(|&x| x == d);
    let mut matrix = Matrix::zeros(dim, dim);
    let m = 2 * n;
    for (col, &d) in dets.iter().enumerate() {
        for p in 0..m {
            for q in 0..m {
                if p % 2 != q % 2 {
                    continue;
                }
                let value = h[(p / 2, q / 2)];
                if let Some((x, s1)) = annihilate(d, q) {
                    if let Some((y, s2)) = create(x, p) {
                        if let Some(row) = index(y) {
                            matrix[(row, col)] += s1 * s2 * value;
                        }
                    }
                }
            }
        }
        // 1/2 sum (pq|rs) a+_p a+_r a_s a_q over spin orbitals.
        for p in 0..m {
            for q in 0..m {
                if p % 2 != q % 2 {
                    continue;
                }
                for r in 0..m {
                    for s in 0..m {
                        if r % 2 != s % 2 {
                            continue;
                        }
                        let value = 0.5 * g(p / 2, q / 2, r / 2, s / 2);
                        let step = annihilate(d, q)
                            .and_then(|(x, a)| annihilate(x, s).map(|(y, b)| (y, a * b)))
                            .and_then(|(x, a)| create(x, r).map(|(y, b)| (y, a * b)))
                            .and_then(|(x, a)| create(x, p).map(|(y, b)| (y, a * b)));
                        if let Some((y, sign)) = step {
                            if let Some(row) = index(y) {
                                matrix[(row, col)] += sign * value;
                            }
                        }
                    }
                }
            }
        }
    }
    let (values, _) = dft_rs::symmetric_eigen(&matrix).unwrap();
    values.iter().map(|v| v + e_nuc).collect()
}

fn mo_integrals(c: &Matrix, h: &Matrix, eri: &EriTensor) -> (Matrix, Vec<f64>) {
    let n = c.cols;
    let nao = c.rows;
    let hm = c.transpose_matmul(&h.matmul(c));
    let mut out = vec![0.0; n * n * n * n];
    let mut half = vec![0.0; nao * nao * n * n];
    for m in 0..nao {
        for nu in 0..nao {
            for r in 0..n {
                for s in 0..n {
                    let mut v = 0.0;
                    for l in 0..nao {
                        for sg in 0..nao {
                            v += eri.get(m, nu, l, sg) * c[(l, r)] * c[(sg, s)];
                        }
                    }
                    half[((m * nao + nu) * n + r) * n + s] = v;
                }
            }
        }
    }
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let mut v = 0.0;
                    for m in 0..nao {
                        for nu in 0..nao {
                            v += c[(m, p)] * c[(nu, q)] * half[((m * nao + nu) * n + r) * n + s];
                        }
                    }
                    out[((p * n + q) * n + r) * n + s] = v;
                }
            }
        }
    }
    (hm, out)
}

#[test]
fn full_active_space_is_the_independent_determinant_ci() {
    let system = lithium_hydride();
    let (setup, scf) = reference(&system, &[]);
    let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 256).unwrap();
    let (h, g) = mo_integrals(&scf.mo_coeff[0], &scf.h_core, &eri);
    let independent = dense_ci(&h, &g, h.rows, 2, 2, scf.e_nuc);
    for casci in [true, false] {
        let report = run(
            "casscf",
            &system,
            &controls(&base(&[
                ("cas", ints(&[4, 6])),
                ("casci", Value::Bool(casci)),
            ])),
        )
        .unwrap();
        let energy = number(&report, "energy");
        println!(
            "LiH full CI: driver {energy:.12} (casci {casci}) independent {:.12} difference {:.3e}",
            independent[0],
            energy - independent[0]
        );
        assert!((energy - independent[0]).abs() < 1e-10);
    }
}

fn problem_parts(
    system: &System,
    extra: &[(&'static str, Value)],
) -> (MolecularSetup, dft_rs::ScfResult, EriTensor) {
    let (setup, scf) = reference(system, extra);
    let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 256).unwrap();
    (setup, scf, eri)
}

#[test]
fn orbital_gradient_passes_a_four_step_fit() {
    let (_setup, scf, eri) = problem_parts(&water(), &[]);
    let problem = CasProblem::new(
        CasOperators {
            h_core: &scf.h_core,
            eri: &eri,
            e_nuc: scf.e_nuc,
        },
        CasSpec {
            n_core: 3,
            n_active: 3,
            n_alpha: 2,
            n_beta: 2,
            weights: vec![0.6, 0.4],
            ci_tolerance: 1e-12,
            ci_max_iterations: 400,
        },
    )
    .unwrap();
    let c = &scf.mo_coeff[0];
    let point = problem.evaluate(c, None).unwrap();
    let steps = [0.02, 0.01, 0.005, 0.0025];
    // Core-active, core-virtual and active-virtual rotations (water in
    // STO-3G: three core, three active and one virtual orbital). The
    // molecule is planar, so rotations between orbitals of different
    // reflection symmetry have zero gradient and zero energy change; take
    // the largest gradient element of every class.
    let class = |p: usize| -> usize {
        if p < 3 {
            0
        } else if p < 6 {
            1
        } else {
            2
        }
    };
    let mut pairs = Vec::new();
    for (lower, upper) in [(0usize, 1usize), (0, 2), (1, 2)] {
        let mut best = (0usize, 0usize, 0.0_f64);
        for p in 0..c.cols {
            for q in 0..p {
                let value = point.gradient[(p, q)].abs();
                if class(q) == lower && class(p) == upper && value > best.2 {
                    best = (p, q, value);
                }
            }
        }
        assert!(best.2 > 1e-6, "class ({lower},{upper}) has no gradient");
        pairs.push((best.0, best.1));
    }
    for (p, q) in pairs {
        let energy = |h: f64| {
            let mut kappa = Matrix::zeros(c.cols, c.cols);
            kappa[(p, q)] = h;
            problem
                .evaluate(&CasProblem::rotate(c, &kappa).unwrap(), None)
                .unwrap()
                .average_energy
        };
        let analytic = point.gradient[(p, q)];
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| ((energy(h) - energy(-h)) / (2.0 * h) - analytic).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "orbital gradient ({p},{q}) analytic {analytic:.10} steps {steps:?} errors {errors:?} \
             p={:.6} R2={:.8} C={:.3e}",
            fit.exponent, fit.r_squared, fit.prefactor
        );
        assert!((fit.exponent - 2.0).abs() < 0.05, "p = {}", fit.exponent);
        assert!(fit.r_squared > 0.9999);
    }
}

#[test]
fn orbital_optimization_limits() {
    // CAS(0,0) orbital optimization is restricted Hartree-Fock.
    let (_, scf) = reference(&water(), &[]);
    let empty = run(
        "casscf",
        &water(),
        &controls(&base(&[("cas", ints(&[0, 0]))])),
    )
    .unwrap();
    let difference = number(&empty, "energy") - scf.energy;
    println!("CAS(0,0) against RHF {difference:.3e}");
    assert!(difference.abs() < 1e-9);
    // CAS(4,4) is converged, stationary and below Hartree-Fock.
    let cas = run(
        "casscf",
        &water(),
        &controls(&base(&[("cas", ints(&[4, 4]))])),
    )
    .unwrap();
    let gradient = number(&cas, "orbital_gradient");
    println!(
        "water CAS(4,4) {:.12}, gradient {gradient:.3e}, occupations {:?}",
        number(&cas, "energy"),
        vector(&cas, "natural_occupations")
    );
    assert_eq!(
        cas.get("orbitals_converged").and_then(Value::as_bool),
        Some(true)
    );
    assert!(gradient < 1e-6);
    assert!(number(&cas, "energy") < scf.energy);
    // Stretched H2 is twice the hydrogen atom.
    let pair = run(
        "casscf",
        &hydrogen_pair(20.0),
        &controls(&base(&[("cas", ints(&[2, 2]))])),
    )
    .unwrap();
    let atom = run(
        "energy",
        &System::molecule(vec![1], vec![[0.0; 3]]),
        &controls(&base(&[
            ("multiplicity", Value::Int(2)),
            ("unrestricted", Value::Bool(true)),
        ])),
    )
    .unwrap();
    let defect = number(&pair, "energy") - 2.0 * number(&atom, "energy");
    println!("H2 at 20 Bohr CAS(2,2) against two atoms {defect:.3e}");
    assert!(defect.abs() < 1e-8);
}

fn pdft(system: &System, extra: &[(&'static str, Value)]) -> Value {
    run("mcpdft", system, &controls(&base(extra))).unwrap()
}

/// Independent Kohn-Sham functional of AO spin densities on the grid of the
/// setup.
fn kohn_sham(setup: &MolecularSetup, functional: &str, p: [&Matrix; 2]) -> f64 {
    let grid = MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap();
    let f = resolve(&FunctionalSpec::Combined(functional.into())).unwrap();
    dft_rs::ks::xc_contribution(&f, &setup.basis, &grid, p)
        .unwrap()
        .exc
}

#[test]
fn single_determinant_on_top_energy_is_the_kohn_sham_functional() {
    // Closed shell: R = 1 and zeta = 0, for local and gradient-corrected parents.
    let (setup, scf) = reference(&water(), &[]);
    for (name, parent) in [("tsvwn5", "svwn5"), ("tpbe", "pbe")] {
        let report = pdft(
            &water(),
            &[
                ("cas", ints(&[0, 0])),
                ("casci", Value::Bool(true)),
                ("on_top_functional", Value::Str(name.into())),
            ],
        );
        let ks = kohn_sham(&setup, parent, [&scf.density[0], &scf.density[1]]);
        let on_top = vector(&report, "state_on_top")[0];
        println!(
            "closed shell {name}: on-top {on_top:.12} Kohn-Sham {ks:.12} difference {:.3e}",
            on_top - ks
        );
        assert!((on_top - ks).abs() < 1e-9);
    }
    // Open shell, one determinant: Li 1s^2 2s^1, zeta = rho_a / rho exactly;
    // the local parent sees the actual spin densities.
    let lithium = System::molecule(vec![3], vec![[0.0; 3]]);
    let extra = [
        ("multiplicity", Value::Int(2)),
        ("unrestricted", Value::Bool(true)),
    ];
    let (setup, scf) = reference(&lithium, &extra);
    let report = pdft(
        &lithium,
        &[
            ("multiplicity", Value::Int(2)),
            ("unrestricted", Value::Bool(true)),
            ("cas", ints(&[1, 1])),
            ("casci", Value::Bool(true)),
            ("on_top_functional", Value::Str("tsvwn5".into())),
        ],
    );
    let c = &scf.mo_coeff[0];
    let column = |j: usize| -> Matrix {
        let mut m = Matrix::zeros(c.rows, 1);
        for i in 0..c.rows {
            m[(i, 0)] = c[(i, j)];
        }
        m
    };
    let core = column(0).matmul_transpose_seq(&column(0));
    let active = column(1).matmul_transpose_seq(&column(1));
    let mut alpha = core.clone();
    for (a, b) in alpha.as_mut_slice().iter_mut().zip(active.as_slice()) {
        *a += b;
    }
    let ks = kohn_sham(&setup, "svwn5", [&alpha, &core]);
    let on_top = vector(&report, "state_on_top")[0];
    println!(
        "Li doublet tsvwn5: on-top {on_top:.12} Kohn-Sham {ks:.12} difference {:.3e}",
        on_top - ks
    );
    assert!((on_top - ks).abs() < 1e-9);
}

/// Independent translation formulas of the test.
fn translated_spin(ft: bool, rho: f64, pi: f64, g: [f64; 3], gp: [f64; 3]) -> [f64; 5] {
    let r = (4.0 * pi / (rho * rho)).max(0.0);
    let (zeta, dz) = if ft {
        let (r0, r1) = (0.9, 1.15);
        let (a, b, cc) = (-475.60656009, -379.47331922, -85.38149682);
        if r < r0 {
            ((1.0 - r).sqrt(), -0.5 / (1.0 - r).sqrt())
        } else if r <= r1 {
            let x = r - r1;
            (
                a * x.powi(5) + b * x.powi(4) + cc * x.powi(3),
                5.0 * a * x.powi(4) + 4.0 * b * x.powi(3) + 3.0 * cc * x.powi(2),
            )
        } else {
            (0.0, 0.0)
        }
    } else if r < 1.0 {
        ((1.0 - r).sqrt(), 0.0)
    } else {
        (0.0, 0.0)
    };
    let mut ga = [0.0; 3];
    let mut gb = [0.0; 3];
    for x in 0..3 {
        let grad_r = 4.0 * gp[x] / (rho * rho) - 8.0 * pi * g[x] / rho.powi(3);
        let shift = if ft { 0.5 * rho * dz * grad_r } else { 0.0 };
        ga[x] = 0.5 * (1.0 + zeta) * g[x] + shift;
        gb[x] = 0.5 * (1.0 - zeta) * g[x] - shift;
    }
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    [
        0.5 * rho * (1.0 + zeta),
        0.5 * rho * (1.0 - zeta),
        dot(ga, ga),
        dot(ga, gb),
        dot(gb, gb),
    ]
}

#[test]
fn gradient_corrected_translations_match_an_independent_grid_evaluation() {
    // Li 1s^2 2s^1: Pi = rho_c^2/4 + rho_c rho_a/2 with the active 2s alone.
    let lithium = System::molecule(vec![3], vec![[0.0; 3]]);
    let (setup, scf) = reference(
        &lithium,
        &[
            ("multiplicity", Value::Int(2)),
            ("unrestricted", Value::Bool(true)),
        ],
    );
    let grid = MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap();
    let pbe = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let c = &scf.mo_coeff[0];
    let batch = eval_aos(&setup.basis, &grid.points, true);
    let g = batch.grad.as_ref().unwrap();
    let npts = grid.len();
    let mut vars: Vec<[f64; 5]> = Vec::new();
    for ft in [false, true] {
        vars.clear();
        for i in 0..npts {
            let (mut v1, mut v2) = (0.0, 0.0);
            let (mut g1, mut g2) = ([0.0; 3], [0.0; 3]);
            for mu in 0..c.rows {
                v1 += c[(mu, 0)] * batch.phi[(i, mu)];
                v2 += c[(mu, 1)] * batch.phi[(i, mu)];
                for x in 0..3 {
                    g1[x] += c[(mu, 0)] * g[x][(i, mu)];
                    g2[x] += c[(mu, 1)] * g[x][(i, mu)];
                }
            }
            let (rc, ra) = (2.0 * v1 * v1, v2 * v2);
            let rho = rc + ra;
            let gc: [f64; 3] = std::array::from_fn(|x| 4.0 * v1 * g1[x]);
            let ga: [f64; 3] = std::array::from_fn(|x| 2.0 * v2 * g2[x]);
            let grad: [f64; 3] = std::array::from_fn(|x| gc[x] + ga[x]);
            let pi = 0.25 * rc * rc + 0.5 * rc * ra;
            let gp: [f64; 3] =
                std::array::from_fn(|x| 0.5 * rc * gc[x] + 0.5 * (gc[x] * ra + rc * ga[x]));
            vars.push(if rho < 1e-14 {
                [0.0; 5]
            } else {
                translated_spin(ft, rho, pi, grad, gp)
            });
        }
        let col = |k: usize| -> Vec<f64> {
            vars.iter()
                .map(|v| v[k].max(if k < 2 { 0.0 } else { f64::MIN }))
                .collect()
        };
        let (ra, rb, saa, sab, sbb) = (col(0), col(1), col(2), col(3), col(4));
        let mut out = XcOut::default();
        pbe.eval(
            &GridVars {
                rho: [&ra, &rb],
                sigma: Some([&saa, &sab, &sbb]),
                tau: None,
            },
            0,
            &mut out,
        )
        .unwrap();
        let independent: f64 = (0..npts)
            .filter(|&i| ra[i] + rb[i] >= 1e-14)
            .map(|i| grid.weights[i] * out.e[i])
            .sum();
        let name = if ft { "ftpbe" } else { "tpbe" };
        let report = pdft(
            &lithium,
            &[
                ("multiplicity", Value::Int(2)),
                ("unrestricted", Value::Bool(true)),
                ("cas", ints(&[1, 1])),
                ("casci", Value::Bool(true)),
                ("on_top_functional", Value::Str(name.into())),
            ],
        );
        let on_top = vector(&report, "state_on_top")[0];
        println!("Li doublet {name}: driver {on_top:.12} independent {independent:.12} difference {:.3e}", on_top - independent);
        assert!((on_top - independent).abs() < 1e-10);
    }
}

#[test]
fn stretched_hydrogen_on_top_energy_is_size_consistent() {
    for name in ["tpbe", "ftpbe"] {
        let pair = pdft(
            &hydrogen_pair(20.0),
            &[
                ("cas", ints(&[2, 2])),
                ("on_top_functional", Value::Str(name.into())),
            ],
        );
        let atom = pdft(
            &System::molecule(vec![1], vec![[0.0; 3]]),
            &[
                ("multiplicity", Value::Int(2)),
                ("unrestricted", Value::Bool(true)),
                ("cas", ints(&[1, 1])),
                ("casci", Value::Bool(true)),
                ("on_top_functional", Value::Str(name.into())),
            ],
        );
        let defect = number(&pair, "energy") - 2.0 * number(&atom, "energy");
        println!(
            "{name}: H2 at 20 Bohr {:.10}, two atoms {:.10}, defect {defect:.3e}",
            number(&pair, "energy"),
            2.0 * number(&atom, "energy")
        );
        assert!(defect.abs() < 1e-6);
    }
}

#[test]
fn on_top_potential_passes_a_four_step_fit() {
    let (setup, scf, eri) = problem_parts(&lithium_hydride(), &[]);
    let problem = CasProblem::new(
        CasOperators {
            h_core: &scf.h_core,
            eri: &eri,
            e_nuc: scf.e_nuc,
        },
        CasSpec {
            n_core: 1,
            n_active: 3,
            n_alpha: 1,
            n_beta: 1,
            weights: vec![0.5, 0.5],
            ci_tolerance: 1e-12,
            ci_max_iterations: 400,
        },
    )
    .unwrap();
    let point = problem.evaluate(&scf.mo_coeff[0], None).unwrap();
    let grid = MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap();
    let c = &scf.mo_coeff[0];
    let mut core = Matrix::zeros(c.rows, c.rows);
    for i in 0..c.rows {
        for j in 0..c.rows {
            core[(i, j)] = 2.0 * c[(i, 0)] * c[(j, 0)];
        }
    }
    let mut active = Matrix::zeros(c.rows, 3);
    for i in 0..c.rows {
        for t in 0..3 {
            active[(i, t)] = c[(i, 1 + t)];
        }
    }
    let (g0, gg0) = problem
        .space
        .rdm12(&point.states.vectors[0], &point.states.vectors[0])
        .unwrap();
    let (g1, gg1) = problem
        .space
        .rdm12(&point.states.vectors[1], &point.states.vectors[1])
        .unwrap();
    let functional = OnTopFunctional::parse("ftpbe", None).unwrap();
    assert_eq!(functional.translation, Translation::FullyTranslated);
    let at = |h: f64, potential: bool| {
        let mut gamma = point.gamma.clone();
        for ((x, a), b) in gamma
            .as_mut_slice()
            .iter_mut()
            .zip(g1.as_slice())
            .zip(g0.as_slice())
        {
            *x += h * (a - b);
        }
        let gamma2: Vec<f64> = point
            .gamma2
            .iter()
            .zip(gg1.iter().zip(&gg0))
            .map(|(x, (a, b))| x + h * (a - b))
            .collect();
        on_top_energy(
            &functional,
            &setup.basis,
            &grid,
            &OnTopDensity {
                core: &core,
                active: &active,
                gamma: &gamma,
                gamma2: &gamma2,
            },
            potential,
        )
        .unwrap()
    };
    let centre = at(0.0, true);
    let potential = centre.potential.unwrap();
    let mut analytic = 0.0;
    for t in 0..3 {
        for u in 0..3 {
            analytic += potential.one[(t, u)] * (g1[(t, u)] - g0[(t, u)]);
        }
    }
    for (v, (a, b)) in potential.two.iter().zip(gg1.iter().zip(&gg0)) {
        analytic += v * (a - b);
    }
    let steps = [0.04, 0.02, 0.01, 0.005];
    let errors: Vec<f64> = steps
        .iter()
        .map(|&h| ((at(h, false).energy - at(-h, false).energy) / (2.0 * h) - analytic).abs())
        .collect();
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "ftPBE on-top potential along the state difference: analytic {analytic:.12} steps {steps:?} \
         errors {errors:?} p={:.6} R2={:.8} C={:.3e}",
        fit.exponent, fit.r_squared, fit.prefactor
    );
    assert!((fit.exponent - 2.0).abs() < 0.1, "p = {}", fit.exponent);
    assert!(fit.r_squared > 0.999);
}

#[test]
fn hybrid_and_multistate_identities() {
    let system = lithium_hydride();
    let common = |method: &str, roots: i64, lambda: f64| -> Value {
        pdft(
            &system,
            &[
                ("cas", ints(&[2, 3])),
                ("cas_roots", Value::Int(roots)),
                ("on_top_functional", Value::Str("tpbe".into())),
                ("pdft_lambda", Value::Float(lambda)),
                ("pdft_method", Value::Str(method.into())),
            ],
        )
    };
    let hybrid = common("state-specific", 1, 0.25);
    let mixed =
        0.25 * vector(&hybrid, "state_reference")[0] + 0.75 * vector(&hybrid, "state_pdft")[0];
    assert!((number(&hybrid, "energy") - mixed).abs() < 1e-12);
    let single = number(&common("state-specific", 1, 0.0), "energy");
    for method in ["cms-pdft", "xms-pdft"] {
        let e = number(&common(method, 1, 0.0), "energy");
        println!("{method} with one state: {e:.12} against {single:.12}");
        assert!((e - single).abs() < 1e-10);
    }
    let linear = common("l-pdft", 1, 0.0);
    let (expansion, at) = (
        number(&linear, "expansion_pdft_energy"),
        number(&linear, "expansion_linearized_energy"),
    );
    println!(
        "l-pdft one state: expansion {expansion:.12} linearized {at:.12} energy {:.12} state-specific {single:.12}",
        number(&linear, "energy")
    );
    assert!((expansion - at).abs() < 1e-9);
    assert!((expansion - single).abs() < 1e-9);
    // In the model space of one state the linearized energy is the PDFT
    // energy of that state.
    assert!((number(&linear, "energy") - single).abs() < 1e-9);
    for method in ["l-pdft", "cms-pdft", "xms-pdft", "state-specific"] {
        let report = common(method, 2, 0.0);
        let energies = vector(&report, "pdft_energies");
        let h = vector(&report, "effective_hamiltonian");
        let trace = h[0] + h[3];
        println!(
            "{method} two states: {energies:?}, trace defect {:.3e}",
            energies.iter().sum::<f64>() - trace
        );
        assert_eq!(energies.len(), 2);
        assert!(energies[0] <= energies[1]);
        if method != "l-pdft" {
            assert!((energies.iter().sum::<f64>() - trace).abs() < 1e-10);
        }
    }
}
