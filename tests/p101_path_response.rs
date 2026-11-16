// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests of the response along nuclear paths in the transported
//! frame: Taylor coefficients of excitation energies and of the optical
//! tensors of the paired response.
//!
//! Oracles. The first-order excitation coefficient is compared with the
//! independent Z-vector gradient of the Tamm-Dancoff state. The zeroth-order
//! optical tensors are compared with the fixed-geometry polarizability and
//! optical-rotation routines, which share no path code. Every derivative is
//! then fitted against central differences of displaced calculations with
//! four step sizes, `log|error| = log C + p log h`; the differences are
//! validation only.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::dynamic::{dynamic_optical_rotation, dynamic_polarizability, Frequency};
use dft_rs::response::krylov::ResponseSolverConfig;
use dft_rs::response::nuclear::{excitation_path, optical_tensor_path};
use dft_rs::response::tddft::{TdaSolver, TdaSpin, TdaState};
use dft_rs::response::tddft_grad::tda_gradient_with_grid;
use dft_rs::scf::{run_scf, EriMode, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

#[path = "support/convergence.rs"]
mod convergence;

/// Water distorted away from every symmetry element, Bohr.
fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.05, 0.2300),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.15, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1200, -1.4809, -0.9067),
        },
    ])
}

/// A displacement pattern without symmetry, Bohr per unit path parameter.
fn pattern() -> Vec<[f64; 3]> {
    vec![[0.3, -0.2, 0.5], [-0.4, 0.6, 0.1], [0.2, 0.1, -0.7]]
}

fn moved(mol: &Molecule, direction: &[[f64; 3]], t: f64) -> Molecule {
    let mut out = mol.clone();
    for (atom, d) in out.atoms.iter_mut().zip(direction) {
        atom.position.x += t * d[0];
        atom.position.y += t * d[1];
        atom.position.z += t * d[2];
    }
    out
}

fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn config() -> ResponseSolverConfig {
    // The velocity-gauge tensors carry an explicit 1/omega^2 factor (omega =
    // 0.1 here), so response residuals are amplified a hundredfold in the
    // origin-shift laws. A tolerance of 1e-10 leaves quadrupole-law defects
    // near 5e-8; 1e-12 resolves every law below 3e-10 against the unchanged
    // 1e-8 thresholds (measured hf/pbe, orders 0 and 1).
    ResponseSolverConfig {
        tolerance: 1e-12,
        ..ResponseSolverConfig::default()
    }
}

struct Point {
    mol: Molecule,
    mb: MolBasis,
    grid: Option<MolecularGrid>,
    scf: ScfResult,
    opts: ScfOptions,
}

fn reference(mol: &Molecule, func: &ResolvedFunctional) -> Point {
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(mol, &set).unwrap();
    let opts = options();
    let grid = (!func.parts.is_empty()).then(|| {
        MolecularGrid::build(
            mol,
            &GridOptions {
                level: 3,
                ..GridOptions::default()
            },
        )
        .unwrap()
    });
    let scf = run_scf(mol, &mb, &opts, func, grid.as_ref()).unwrap();
    assert!(scf.converged);
    Point {
        mol: mol.clone(),
        mb,
        grid,
        scf,
        opts,
    }
}

fn states(point: &Point, func: &ResolvedFunctional, tamm_dancoff: bool) -> Vec<TdaState> {
    let solver = TdaSolver::new(
        &point.mol,
        &point.mb,
        func,
        point.grid.as_ref(),
        &point.scf,
        &point.opts,
        TdaSpin::Singlet,
    )
    .unwrap();
    if tamm_dancoff {
        solver.solve_dense(4).unwrap()
    } else {
        solver.solve_rpa(4).unwrap()
    }
}

fn amplitude_overlap(a: &TdaState, b: &TdaState) -> f64 {
    let mut s = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.x[0].rows {
        for j in 0..a.x[0].cols {
            s += a.x[0][(i, j)] * b.x[0][(i, j)];
            na += a.x[0][(i, j)] * a.x[0][(i, j)];
            nb += b.x[0][(i, j)] * b.x[0][(i, j)];
        }
    }
    s.abs() / (na * nb).sqrt()
}

/// Total energy `E_SCF + omega` of the state that continues `target` at a
/// displaced geometry.
fn state_energy(
    mol: &Molecule,
    func: &ResolvedFunctional,
    tamm_dancoff: bool,
    target: &TdaState,
) -> f64 {
    let point = reference(mol, func);
    let all = states(&point, func, tamm_dancoff);
    let best = all
        .iter()
        .max_by(|u, v| {
            amplitude_overlap(target, u)
                .partial_cmp(&amplitude_overlap(target, v))
                .expect("finite overlaps")
        })
        .expect("at least one state");
    let overlap = amplitude_overlap(target, best);
    assert!(overlap > 0.8, "state following overlap {overlap:.4}");
    point.scf.energy + best.energy
}

#[test]
fn excitation_coefficients_match_the_state_gradient_and_pass_four_step_fits() {
    let mol = water();
    let direction = pattern();
    for (name, tamm_dancoff) in [
        ("hf", true),
        ("svwn5", true),
        ("pbe0", true),
        ("hf", false),
        ("pbe", false),
    ] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func);
        let all = states(&point, &func, tamm_dancoff);
        let state = &all[0];
        let gap = (all[1].energy - state.energy).abs();
        assert!(
            gap > 1e-3,
            "{name}: the state is nearly degenerate ({gap:e})"
        );
        let path = excitation_path(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            state,
            &direction,
            2,
            &config(),
        )
        .unwrap();
        println!(
            "{name} tda={tamm_dancoff}: omega = {:.10} path {:?} ground {:?} reference residual \
             {:.3e} residuals {:?} converged {}",
            state.energy,
            path.excitation,
            path.ground,
            path.reference_residual,
            path.residuals,
            path.converged
        );
        assert!((path.excitation[0] - state.energy).abs() < 1e-12);
        assert!((path.ground[0] - point.scf.energy).abs() < 1e-8);
        assert!(
            path.reference_residual < 1e-6,
            "{name}: reference eigenvalue residual {:.3e}",
            path.reference_residual
        );
        let first = path.ground[1] + path.excitation[1];
        let second = 2.0 * (path.ground[2] + path.excitation[2]);

        // Independent analytic oracle for the first derivative: the
        // Z-vector gradient, available for Tamm-Dancoff states of
        // Hartree-Fock and the local density approximation.
        if tamm_dancoff && (name == "hf" || name == "svwn5") {
            let gradient = tda_gradient_with_grid(
                &point.mol,
                &point.mb,
                &func,
                point.grid.as_ref(),
                &point.scf,
                state,
                &point.opts,
            )
            .unwrap();
            let projected: f64 = gradient
                .iter()
                .zip(&direction)
                .map(|(g, d)| g.x * d[0] + g.y * d[1] + g.z * d[2])
                .sum();
            println!(
                "{name}: path first derivative {first:.10} Z-vector {projected:.10} difference \
                 {:.3e}",
                first - projected
            );
            assert!(
                (first - projected).abs() < 2e-7,
                "{name}: path {first:.10} vs Z-vector gradient {projected:.10}"
            );
        }

        // Four-step fits against central differences of the state energy.
        let steps = [0.08, 0.04, 0.02, 0.01];
        let center = point.scf.energy + state.energy;
        let mut first_errors = Vec::new();
        let mut second_errors = Vec::new();
        let mut second_values = Vec::new();
        for &h in &steps {
            let plus = state_energy(&moved(&mol, &direction, h), &func, tamm_dancoff, state);
            let minus = state_energy(&moved(&mol, &direction, -h), &func, tamm_dancoff, state);
            let d1 = (plus - minus) / (2.0 * h);
            let d2 = (plus - 2.0 * center + minus) / (h * h);
            first_errors.push((d1 - first).abs());
            second_errors.push((d2 - second).abs());
            second_values.push(d2);
        }
        let first_fit = convergence::error_power_fit(&steps, &first_errors);
        let second_fit = convergence::error_power_fit(&steps, &second_errors);
        let richardson = (4.0 * second_values[3] - second_values[2]) / 3.0;
        println!(
            "{name} tda={tamm_dancoff}: first p={:.6} R2={:.6} errors {first_errors:?}; second \
             p={:.6} R2={:.6} errors {second_errors:?}; second analytic {second:.10} Richardson \
             {richardson:.10} difference {:.3e}; steps {steps:?}",
            first_fit.exponent,
            first_fit.r_squared,
            second_fit.exponent,
            second_fit.r_squared,
            second - richardson
        );
        assert!(
            (first_fit.exponent - 2.0).abs() < 0.1 && first_fit.r_squared > 0.999,
            "{name}: first-derivative fit p={} R2={}",
            first_fit.exponent,
            first_fit.r_squared
        );
        assert!(
            (second_fit.exponent - 2.0).abs() < 0.1 && second_fit.r_squared > 0.999,
            "{name}: second-derivative fit p={} R2={}",
            second_fit.exponent,
            second_fit.r_squared
        );
        assert!((second - richardson).abs() < 5e-6);
    }
}

type Tensors = ([[f64; 3]; 3], [[f64; 3]; 3], [[[f64; 3]; 3]; 3]);

/// Optical tensors at one geometry: the polarizability and the rotation
/// tensor from the fixed-geometry routines, the dipole-quadrupole tensor
/// from the zeroth path coefficient.
fn fixed_tensors(mol: &Molecule, func: &ResolvedFunctional, omega: f64, origin: Vec3) -> Tensors {
    let point = reference(mol, func);
    let frequency = Frequency::Retarded {
        omega,
        damping: 0.0,
    };
    let alpha = dynamic_polarizability(
        &point.mol,
        &point.mb,
        func,
        point.grid.as_ref(),
        &point.scf,
        &point.opts,
        frequency,
        &config(),
    )
    .unwrap();
    let rotation = dynamic_optical_rotation(
        &point.mol,
        &point.mb,
        func,
        point.grid.as_ref(),
        &point.scf,
        &point.opts,
        frequency,
        origin,
        &config(),
    )
    .unwrap();
    let still = vec![[0.0; 3]; mol.len()];
    let path = optical_tensor_path(
        &point.mol,
        &point.mb,
        func,
        point.grid.as_ref(),
        &point.scf,
        &point.opts,
        &still,
        0,
        omega,
        origin,
        false,
        &config(),
    )
    .unwrap();
    (
        alpha.value.alpha.map(|row| row.map(|z| z.re)),
        rotation.value.tensor.map(|row| row.map(|z| z.re)),
        path.quadrupole[0],
    )
}

fn largest2(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}

#[test]
fn optical_tensor_coefficients_match_fixed_geometry_routines_and_pass_four_step_fits() {
    let mol = water();
    let direction = pattern();
    let omega = 0.1;
    let origin = Vec3::new(0.1, -0.2, 0.3);
    for name in ["hf", "pbe"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func);
        let path = optical_tensor_path(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            &direction,
            1,
            omega,
            origin,
            false,
            &config(),
        )
        .unwrap();
        let (alpha, rotation, quadrupole) = fixed_tensors(&mol, &func, omega, origin);
        let alpha_defect = largest2(&path.polarizability[0], &alpha);
        let rotation_defect = largest2(&path.rotation[0], &rotation);
        println!(
            "{name}: alpha {:?} G' {:?} zeroth-order defects alpha {alpha_defect:.3e} G' \
             {rotation_defect:.3e} residuals {:?} converged {}",
            path.polarizability[0], path.rotation[0], path.residuals, path.converged
        );
        assert!(alpha_defect < 1e-7, "{name}: alpha defect {alpha_defect:e}");
        assert!(
            rotation_defect < 1e-7,
            "{name}: rotation defect {rotation_defect:e}"
        );
        // The quadrupole part is symmetric and traceless.
        for a in 0..3 {
            let q = &path.quadrupole[0][a];
            assert!((q[0][0] + q[1][1] + q[2][2]).abs() < 1e-10);
            for b in 0..3 {
                for c in 0..3 {
                    assert!((q[b][c] - q[c][b]).abs() < 1e-10);
                    assert!((q[b][c] - quadrupole[a][b][c]).abs() < 1e-9);
                }
            }
        }
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = [Vec::new(), Vec::new(), Vec::new()];
        for &h in &steps {
            let plus = fixed_tensors(&moved(&mol, &direction, h), &func, omega, origin);
            let minus = fixed_tensors(&moved(&mol, &direction, -h), &func, omega, origin);
            let mut worst = [0.0_f64; 3];
            for a in 0..3 {
                for b in 0..3 {
                    let d_alpha = (plus.0[a][b] - minus.0[a][b]) / (2.0 * h);
                    let d_rotation = (plus.1[a][b] - minus.1[a][b]) / (2.0 * h);
                    worst[0] = worst[0].max((d_alpha - path.polarizability[1][a][b]).abs());
                    worst[1] = worst[1].max((d_rotation - path.rotation[1][a][b]).abs());
                    for c in 0..3 {
                        let d_quadrupole = (plus.2[a][b][c] - minus.2[a][b][c]) / (2.0 * h);
                        worst[2] = worst[2].max((d_quadrupole - path.quadrupole[1][a][b][c]).abs());
                    }
                }
            }
            for (list, value) in errors.iter_mut().zip(worst) {
                list.push(value);
            }
        }
        for (label, list) in ["alpha", "rotation", "quadrupole"].iter().zip(&errors) {
            let fit = convergence::error_power_fit(&steps, list);
            println!(
                "{name} d{label}/dt: p={:.6} R2={:.6} errors {list:?} steps {steps:?}",
                fit.exponent, fit.r_squared
            );
            assert!(
                (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
                "{name} {label}: p={} R2={}",
                fit.exponent,
                fit.r_squared
            );
        }
    }
}

fn levi_civita(a: usize, b: usize, c: usize) -> f64 {
    match (a, b, c) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,
        _ => 0.0,
    }
}

/// In the velocity gauge the tensors change with the origin exactly as in a
/// complete basis: `alpha` not at all, `G'_ab` by
/// `(w / 2) eps_bcd d_c alpha_ad` and `A_a,bc` by
/// `-(3/2)(d_b alpha_ac + d_c alpha_ab) + delta_bc d_e alpha_ae`, at every
/// Taylor order. These laws make the rotational invariants independent of
/// the origin.
#[test]
fn velocity_gauge_tensors_obey_the_exact_origin_shift_laws() {
    let mol = water();
    let direction = pattern();
    let omega = 0.1;
    let shift = [0.7, -0.4, 1.1];
    for name in ["hf", "pbe"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func);
        let run = |origin: Vec3| {
            optical_tensor_path(
                &point.mol,
                &point.mb,
                &func,
                point.grid.as_ref(),
                &point.scf,
                &point.opts,
                &direction,
                1,
                omega,
                origin,
                true,
                &config(),
            )
            .unwrap()
        };
        let base = run(Vec3::new(0.1, -0.2, 0.3));
        let moved = run(Vec3::new(0.1 + shift[0], -0.2 + shift[1], 0.3 + shift[2]));
        for n in 0..=1 {
            let alpha = &base.polarizability[n];
            let mut worst = [0.0_f64; 4];
            for a in 0..3 {
                for b in 0..3 {
                    worst[0] = worst[0].max((moved.polarizability[n][a][b] - alpha[a][b]).abs());
                    worst[3] = worst[3].max((alpha[a][b] - alpha[b][a]).abs());
                    let mut law = 0.0;
                    for c in 0..3 {
                        for d in 0..3 {
                            law += 0.5 * omega * levi_civita(b, c, d) * shift[c] * alpha[a][d];
                        }
                    }
                    worst[1] = worst[1]
                        .max((moved.rotation[n][a][b] - base.rotation[n][a][b] - law).abs());
                    for c in 0..3 {
                        let mut law = -1.5 * (shift[b] * alpha[a][c] + shift[c] * alpha[a][b]);
                        if b == c {
                            for e in 0..3 {
                                law += shift[e] * alpha[a][e];
                            }
                        }
                        worst[2] = worst[2].max(
                            (moved.quadrupole[n][a][b][c] - base.quadrupole[n][a][b][c] - law)
                                .abs(),
                        );
                    }
                }
            }
            println!(
                "{name} order {n}: origin-law defects alpha {:.3e} rotation {:.3e} quadrupole \
                 {:.3e}; alpha asymmetry {:.3e}",
                worst[0], worst[1], worst[2], worst[3]
            );
            assert!(worst[0] < 1e-8 && worst[1] < 1e-8 && worst[2] < 1e-8);
            assert!(worst[3] < 1e-7);
        }
    }
}
