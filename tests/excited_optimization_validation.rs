// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! T120 bounded HF/STO-3G singlet-TDA optimization acceptance. The scalar
//! reference is an independent dense TDA solve and states are compared through
//! AO transition densities with cross-geometry overlap metrics. All Cartesian
//! gradient entries are fitted at the C1 start and optimized endpoint. Finite
//! gradient differences test curvature only; they are not production Hessians.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::overlap_matrix;
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::response::tddft::{TdaSolver, TdaSpin, TdaState};
use dft_rs::scf::{run_hf, EriMode, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use stencils::Stencil;

fn start() -> System {
    let (cz, sz) = (0.41_f64.cos(), 0.41_f64.sin());
    let (cx, sx) = (0.63_f64.cos(), 0.63_f64.sin());
    let positions = [
        [0.02, 0.01, -0.04],
        [0.04, -0.03, 2.30],
        [0.17, 1.83, -1.06],
        [0.11, -1.68, -1.19],
    ]
    .map(|[x, y, z]| {
        let (a, b) = (cz * x - sz * y, sz * x + cz * y);
        [a + 0.17, cx * b - sx * z - 0.23, sx * b + cx * z + 0.31]
    });
    System::molecule(vec![6, 8, 1, 1], positions.to_vec())
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut values = vec![
        ("xc".into(), Value::Str("hf".into())),
        ("basis".into(), Value::Str("sto-3g".into())),
        ("grid_weight_cutoff".into(), Value::Float(0.0)),
        ("conv_tol".into(), Value::Float(1e-13)),
        ("conv_tol_dp".into(), Value::Float(1e-12)),
        ("conv_tol_grad".into(), Value::Float(1e-11)),
        ("td_tolerance".into(), Value::Float(1e-11)),
        ("nstates".into(), Value::Int(4)),
        ("root".into(), Value::Int(1)),
    ];
    values.extend(
        extra
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone())),
    );
    Controls::from_value(&Value::Object(values)).unwrap()
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}
fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}
fn molecule(system: &System) -> Molecule {
    Molecule::new(
        system
            .numbers
            .iter()
            .zip(&system.positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

fn transition_density(scf: &ScfResult, state: &TdaState) -> Matrix {
    let c = &scf.mo_coeff[0];
    let occupied = scf.n_alpha;
    let mut co = Matrix::zeros(c.rows, occupied);
    let mut cv = Matrix::zeros(c.rows, c.cols - occupied);
    for mu in 0..c.rows {
        for i in 0..occupied {
            co[(mu, i)] = c[(mu, i)];
        }
        for a in occupied..c.cols {
            cv[(mu, a - occupied)] = c[(mu, a)];
        }
    }
    cv.matmul_seq(&state.x[0]).matmul_transpose_seq(&co)
}

struct Oracle {
    system: System,
    mol: Molecule,
    basis: MolBasis,
    scf: ScfResult,
    states: Vec<TdaState>,
    density: Vec<Matrix>,
}

impl Oracle {
    fn new(system: &System, check_iterative: bool) -> Self {
        let mol = molecule(system);
        let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let opts = ScfOptions {
            e_tol: 1e-13,
            p_tol: 1e-12,
            conv_tol_grad: 1e-11,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        let scf = run_hf(&mol, &basis, &opts).unwrap();
        assert!(scf.converged);
        let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
        let mut solver = TdaSolver::new(
            &mol,
            &basis,
            &functional,
            None,
            &scf,
            &opts,
            TdaSpin::Singlet,
        )
        .unwrap();
        solver.tol = 1e-11;
        solver.max_iter = 200;
        let states = solver.solve_dense(4).unwrap();
        assert_eq!(states.len(), 4);
        assert!(states[0].energy > 0.0);
        assert!(
            states[1].energy - states[0].energy > 1e-3,
            "ambiguous lowest TDA root"
        );
        if check_iterative {
            let iterative = solver.solve_davidson(4).unwrap();
            let worst = states
                .iter()
                .zip(&iterative)
                .map(|(a, b)| (a.energy - b.energy).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-9, "dense/Davidson energy difference {worst}");
            println!(
                "dense/Davidson max={worst:.12e}; root gap={:.12e}",
                states[1].energy - states[0].energy
            );
        }
        let density = states
            .iter()
            .map(|state| transition_density(&scf, state))
            .collect();
        Self {
            system: system.clone(),
            mol,
            basis,
            scf,
            states,
            density,
        }
    }
    fn energy(&self) -> f64 {
        self.scf.energy + self.states[0].energy
    }
    fn gradient(&self) -> Vec<f64> {
        let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
        let opts = ScfOptions {
            e_tol: 1e-13,
            p_tol: 1e-12,
            conv_tol_grad: 1e-11,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        dft_rs::response::tddft_grad::tda_gradient_with_grid(
            &self.mol,
            &self.basis,
            &functional,
            None,
            &self.scf,
            &self.states[0],
            &opts,
        )
        .unwrap()
        .into_iter()
        .flat_map(Vec3::to_array)
        .collect()
    }
}

fn metric_product(left: &Matrix, right: &Matrix, cross: &Matrix) -> f64 {
    left.frobenius_dot(&cross.matmul_seq(right).matmul_transpose_seq(cross))
}

fn state_overlap(reference: &Oracle, point: &Oracle) -> [f64; 4] {
    let mut atoms = reference.mol.atoms.clone();
    atoms.extend(point.mol.atoms.iter().cloned());
    let basis = MolBasis::build(&Molecule::new(atoms), &data::load("sto-3g").unwrap()).unwrap();
    let n = reference.basis.nao_sph;
    assert_eq!(basis.nao_sph, 2 * n);
    let overlap = overlap_matrix(&basis);
    let mut cross = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            cross[(mu, nu)] = overlap[(mu, n + nu)];
        }
    }
    let norm_left = metric_product(
        &reference.density[0],
        &reference.density[0],
        &reference.scf.overlap,
    );
    std::array::from_fn(|k| {
        let norm_right = metric_product(&point.density[k], &point.density[k], &point.scf.overlap);
        metric_product(&reference.density[0], &point.density[k], &cross).abs()
            / (norm_left * norm_right).sqrt()
    })
}

fn followed(
    system: &System,
    reference: &Oracle,
    min_overlap: &mut f64,
    min_gap: &mut f64,
) -> Oracle {
    let point = Oracle::new(system, false);
    let similarities = state_overlap(reference, &point);
    let best = (0..4)
        .max_by(|&a, &b| similarities[a].total_cmp(&similarities[b]))
        .unwrap();
    assert_eq!(
        best, 0,
        "AO transition-density reference selected a different root: {similarities:?}"
    );
    assert!(similarities[0] > 0.9, "lost state: {similarities:?}");
    assert!(similarities[0] - similarities[1..].iter().copied().fold(0.0, f64::max) > 0.5);
    *min_overlap = min_overlap.min(similarities[0]);
    *min_gap = min_gap.min(point.states[1].energy - point.states[0].energy);
    point
}

fn internal_basis(system: &System) -> Matrix {
    let n = 3 * system.numbers.len();
    let center = std::array::from_fn::<_, 3, _>(|axis| {
        system.positions.iter().map(|p| p[axis]).sum::<f64>() / system.positions.len() as f64
    });
    let mut rigid: Vec<Vec<f64>> = Vec::new();
    let mut append = |mut v: Vec<f64>| {
        for _ in 0..2 {
            for q in &rigid {
                let dot = v.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
                for (a, b) in v.iter_mut().zip(q) {
                    *a -= dot * b;
                }
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(norm > 1e-10);
        for a in &mut v {
            *a /= norm;
        }
        rigid.push(v);
    };
    for axis in 0..3 {
        let mut v = vec![0.0; n];
        for a in 0..system.numbers.len() {
            v[3 * a + axis] = 1.0;
        }
        append(v);
    }
    for axis in 0..3 {
        let mut v = vec![0.0; n];
        for (a, p) in system.positions.iter().enumerate() {
            let r = Vec3::new(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
            let unit = Vec3::new(
                if axis == 0 { 1.0 } else { 0.0 },
                if axis == 1 { 1.0 } else { 0.0 },
                if axis == 2 { 1.0 } else { 0.0 },
            );
            v[3 * a..3 * a + 3].copy_from_slice(&unit.cross(r).to_array());
        }
        append(v);
    }
    let mut columns = Vec::new();
    for j in 0..n {
        let mut v = vec![0.0; n];
        v[j] = 1.0;
        for _ in 0..2 {
            for q in rigid.iter().chain(&columns) {
                let dot = v.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
                for (a, b) in v.iter_mut().zip(q) {
                    *a -= dot * b;
                }
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for a in &mut v {
                *a /= norm;
            }
            columns.push(v);
        }
    }
    assert_eq!(columns.len(), n - 6);
    let mut q = Matrix::zeros(n, n - 6);
    for (j, v) in columns.iter().enumerate() {
        for i in 0..n {
            q[(i, j)] = v[i];
        }
    }
    q
}

fn fit_all_gradient_entries(
    reference: &Oracle,
    analytic: &[f64],
    label: &str,
    check_minimum: bool,
) {
    let n = analytic.len();
    let names: Vec<_> = (0..n).map(|k| format!("{label} gradient {k}")).collect();
    let mut strict = vec![false; n];
    let mut failures = Vec::new();
    let mut min_overlap = 1.0_f64;
    let mut min_gap = f64::INFINITY;
    stencils::fit_components_staged(
        &names,
        &[
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut hessians: Vec<_> = (0..4).map(|_| Matrix::zeros(n, n)).collect();
            let mut errors = vec![[0.0; 4]; n];
            for j in 0..n {
                for (s, &h) in steps.iter().enumerate() {
                    let mut sample = |amount: f64| {
                        let mut system = reference.system.clone();
                        system.positions[j / 3][j % 3] += amount;
                        followed(&system, reference, &mut min_overlap, &mut min_gap)
                    };
                    let (plus, minus) = (sample(h), sample(-h));
                    let extra = if stencil == Stencil::Fourth {
                        Some((sample(2.0 * h), sample(-2.0 * h)))
                    } else {
                        None
                    };
                    errors[j][s] = (stencil.first(
                        plus.energy(),
                        minus.energy(),
                        extra.as_ref().map_or(0.0, |x| x.0.energy()),
                        extra.as_ref().map_or(0.0, |x| x.1.energy()),
                        h,
                    ) - analytic[j])
                        .abs();
                    if check_minimum {
                        let (gp, gm) = (plus.gradient(), minus.gradient());
                        let extra_g = extra.as_ref().map(|(a, b)| (a.gradient(), b.gradient()));
                        for i in 0..n {
                            hessians[s][(i, j)] = stencil.first(
                                gp[i],
                                gm[i],
                                extra_g.as_ref().map_or(0.0, |x| x.0[i]),
                                extra_g.as_ref().map_or(0.0, |x| x.1[i]),
                                h,
                            );
                        }
                        if s == 0 {
                            assert!(
                                (plus.energy() + minus.energy() - 2.0 * reference.energy())
                                    / (h * h)
                                    > 0.0,
                                "negative diagonal curvature {j}"
                            );
                        }
                    }
                }
            }
            if check_minimum {
                let q = internal_basis(&reference.system);
                let mut projected = Vec::new();
                for (s, hessian) in hessians.iter_mut().enumerate() {
                    let mut asym = 0.0_f64;
                    for i in 0..n {
                        for j in 0..i {
                            asym = asym.max((hessian[(i, j)] - hessian[(j, i)]).abs());
                            let value = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
                            hessian[(i, j)] = value;
                            hessian[(j, i)] = value;
                        }
                    }
                    let k = q.transpose_matmul_seq(&hessian.matmul_seq(&q));
                    let (eigenvalues, _) = symmetric_eigen(&k).unwrap();
                    println!("{label} curvature h={} asym={asym:.12e} internal eigenvalues={eigenvalues:?}",steps[s]);
                    assert!(
                        eigenvalues.iter().all(|v| *v > 0.0),
                        "not an internal minimum"
                    );
                    projected.push(k);
                }
                let difference = projected[3].max_abs_diff(&projected[2]);
                let (eigenvalues, _) = symmetric_eigen(&projected[3]).unwrap();
                assert!(
                    eigenvalues[0] > q.cols as f64 * difference,
                    "curvature sign unresolved by step refinement"
                );
            }
            for k in 0..n {
                if errors[k].iter().all(|e| e.is_finite() && *e > 0.0) {
                    let fit = convergence::error_power_fit(steps, &errors[k]);
                    strict[k] |=
                        (fit.exponent - stencil.order()).abs() < 0.05 && fit.r_squared > 0.9999;
                    println!(
                        "{} strict {:?}: p={:.8} R2={:.10} C={:.12e} log_rmse={:.12e}",
                        names[k], stencil, fit.exponent, fit.r_squared, fit.prefactor, fit.log_rmse
                    );
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("excited gradient", &failures);
    assert!(
        strict.iter().all(|v| *v),
        "original strict fit gates failed: {strict:?}"
    );
    println!("{label}: minimum AO overlap={min_overlap:.12e}, minimum root gap={min_gap:.12e}");
}

#[test]
fn c1_isolated_tda_optimization_and_all_components_are_consistent() {
    let initial = start();
    let before = Oracle::new(&initial, true);
    let report = run(
        "excited-optimization",
        &initial,
        &controls(&[("gtol", Value::Float(1e-5)), ("max_steps", Value::Int(200))]),
    )
    .unwrap();
    assert_eq!(
        report
            .get("optimization_converged")
            .and_then(Value::as_bool),
        Some(true)
    );
    let trajectory = vector(&report, "trajectory_energies");
    assert!(trajectory.len() > 1);
    assert!(trajectory.windows(2).all(|e| e[1] <= e[0] + 1e-12));
    assert!((trajectory[0] - before.energy()).abs() < 1e-9);
    let positions = vector(&report, "positions");
    let optimum = System::molecule(
        initial.numbers.clone(),
        positions
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect(),
    );
    let after = Oracle::new(&optimum, true);
    assert!((after.energy() - number(&report, "excited_state_energy")).abs() < 1e-9);
    assert!(after.energy() < before.energy());
    let gradient = vector(&report, "gradient");
    assert!(gradient.iter().all(|g| g.abs() < 1e-5));
    let direct = run("td-state-gradient", &optimum, &controls(&[])).unwrap();
    assert!((number(&direct, "excited_state_energy") - after.energy()).abs() < 1e-9);
    assert!(vector(&direct, "gradient")
        .iter()
        .zip(&gradient)
        .all(|(a, b)| (a - b).abs() < 1e-8));
    println!(
        "optimization iterations={} energy {:.12} -> {:.12} gradient_max={:.12e}",
        number(&report, "iterations"),
        before.energy(),
        after.energy(),
        number(&report, "gradient_max")
    );
    println!("optimized positions={:?}", optimum.positions);
    fit_all_gradient_entries(&before, &before.gradient(), "C1 start", false);
    fit_all_gradient_entries(&after, &gradient, "optimum", true);
}
