// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 13 gate: the analytic TDA (CIS) excited-state nuclear gradient.
//!
//! The reference is central differences of the TOTAL excited-state energy
//! `E_SCF + omega` — the same quantity the analytic routine returns. There
//! is no external oracle here on purpose: finite differences of this
//! program's own energy test the gradient and nothing else, so a mismatch
//! localises to the gradient assembly rather than to a basis-set or
//! convergence difference against another code.
//!
//! Displaced states are followed by AMPLITUDE OVERLAP, not by index. The
//! low-lying TDA roots of water sit within ~0.05 Hartree of each other and
//! can swap order under a 5e-4 bohr displacement; following the index
//! would then differentiate a discontinuous function and manufacture a
//! failure that is not in the gradient.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::tddft::{TdaSolver, TdaSpin, TdaState};
use dft_rs::response::tddft_grad::{
    lda_third_density_contraction, tda_gradient, tda_gradient_with_grid,
};
use dft_rs::scf::{run_scf, EriMode, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};

/// Water, distorted away from C2v so no component of the gradient is zero
/// by symmetry and no low-lying root is degenerate. Bohr.
fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.2300),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1200, -1.4809, -0.9067),
        },
    ])
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

struct Point {
    scf: ScfResult,
    mb: MolBasis,
    states: Vec<TdaState>,
    grid: Option<MolecularGrid>,
}

/// SCF plus the lowest `nstates` singlet TDA roots (dense diagonalization,
/// so the roots carry no iterative error into the differences).
fn solve(mol: &Molecule, basis: &str, func: &ResolvedFunctional, nstates: usize) -> Point {
    let set = data::load(basis).unwrap();
    let mb = MolBasis::build(mol, &set).unwrap();
    let opts = options();
    let grid = (!func.parts.is_empty()).then(|| {
        MolecularGrid::build(
            mol,
            &GridOptions {
                level: 5,
                ..GridOptions::default()
            },
        )
        .unwrap()
    });
    let scf = run_scf(mol, &mb, &opts, func, grid.as_ref()).unwrap();
    let states = TdaSolver::new(mol, &mb, func, grid.as_ref(), &scf, &opts, TdaSpin::Singlet)
        .unwrap()
        .solve_dense(nstates)
        .unwrap();
    Point {
        scf,
        mb,
        states,
        grid,
    }
}

fn displaced(m: &Molecule, atom: usize, ax: usize, h: f64) -> Molecule {
    let mut out = m.clone();
    match ax {
        0 => out.atoms[atom].position.x += h,
        1 => out.atoms[atom].position.y += h,
        _ => out.atoms[atom].position.z += h,
    }
    out
}

/// |<X_a | X_b>| over the amplitude vectors. The absolute value is
/// essential: the SCF at a displaced geometry may return an MO with the
/// opposite phase, which flips the sign of every amplitude touching it.
fn amplitude_overlap(a: &TdaState, b: &TdaState) -> f64 {
    let mut s = 0.0;
    for i in 0..a.x[0].rows {
        for j in 0..a.x[0].cols {
            s += a.x[0][(i, j)] * b.x[0][(i, j)];
        }
    }
    s.abs()
}

fn component(v: Vec3, ax: usize) -> f64 {
    match ax {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Central differences of `E_SCF + omega`, following the reference state.
fn finite_difference(
    mol: &Molecule,
    basis: &str,
    func: &ResolvedFunctional,
    reference: &TdaState,
    nstates: usize,
    atom: usize,
    ax: usize,
    h: f64,
) -> f64 {
    let mut e = [0.0; 2];
    for (slot, step) in [h, -h].iter().enumerate() {
        let point = solve(&displaced(mol, atom, ax, *step), basis, func, nstates);
        let best = point
            .states
            .iter()
            .max_by(|u, v| {
                amplitude_overlap(reference, u)
                    .partial_cmp(&amplitude_overlap(reference, v))
                    .expect("amplitude overlaps are finite")
            })
            .expect("at least one state");
        let overlap = amplitude_overlap(reference, best);
        assert!(
            overlap > 0.80,
            "root tracking overlap {overlap:.6} is too small at atom {atom}, axis {ax}"
        );
        e[slot] = point.scf.energy + best.energy;
    }
    (e[0] - e[1]) / (2.0 * h)
}

/// THE gate: the analytic gradient must reproduce central differences of
/// the total excited-state energy, component by component.
///
/// The step is 5e-4 bohr. The truncation error of a central difference is
/// O(h^2) and was measured to fall by exactly a factor of four per halving
/// from 4e-3 down to 5e-4, which is what says the residual below is the
/// difference formula's and not the gradient's.
#[test]
fn analytic_gradient_matches_finite_differences() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    const H: f64 = 5e-4;
    for basis in ["sto-3g", "6-31g"] {
        let nstates = 4;
        let point = solve(&mol, basis, &func, nstates);
        let state = &point.states[0];
        let grad = tda_gradient(&mol, &point.mb, &func, &point.scf, state, &options()).unwrap();
        for atom in 0..mol.len() {
            for ax in 0..3 {
                let fd = finite_difference(&mol, basis, &func, state, nstates, atom, ax, H);
                let an = component(grad[atom], ax);
                assert!(
                    (an - fd).abs() < 1e-5,
                    "{basis} atom {atom} axis {ax}: analytic {an:.10} vs FD {fd:.10} \
                     (delta = {:.3e})",
                    an - fd
                );
            }
        }
    }
}

/// The A4-2 DFT gate: rebuild the atom-centred grid at every displaced
/// geometry and differentiate the same total `E_SCF + omega` returned by
/// the SVWN5 SCF/TDA path.  This exercises the LDA f_xc/f_xxc terms and all
/// three pieces of the moving-grid derivative together.
#[test]
fn svwn5_analytic_gradient_matches_rebuilt_grid_finite_differences() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let nstates = 4;
    let point = solve(&mol, "sto-3g", &func, nstates);
    let state = &point.states[0];
    let nearest_gap = point
        .states
        .iter()
        .skip(1)
        .map(|other| (other.energy - state.energy).abs())
        .fold(f64::INFINITY, f64::min);
    assert!(
        nearest_gap > 1e-3,
        "the differentiated root is too close to another root: gap={nearest_gap:e}"
    );
    let grad = tda_gradient_with_grid(
        &mol,
        &point.mb,
        &func,
        point.grid.as_ref(),
        &point.scf,
        state,
        &options(),
    )
    .unwrap();
    const H: f64 = 1e-4;
    let mut failures = Vec::new();
    for atom in 0..mol.len() {
        for ax in 0..3 {
            let fd = finite_difference(&mol, "sto-3g", &func, state, nstates, atom, ax, H);
            let an = component(grad[atom], ax);
            let delta = an - fd;
            eprintln!(
                "SVWN5 atom {atom} axis {ax}: analytic {an:.10}, FD {fd:.10}, delta {delta:.3e}"
            );
            if delta.abs() >= 2e-5 {
                failures.push(format!(
                    "atom {atom} axis {ax}: analytic {an:.10} vs FD {fd:.10} \
                     (delta = {delta:.3e})"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "SVWN5 rebuilt-grid FD mismatches:\n{}",
        failures.join("\n")
    );
}

/// Independent point-local check of the LDA third-kernel contraction.  The
/// reference differentiates the second-kernel response `f_xc(rho) t` in
/// the transition-density direction and therefore does not reuse the
/// production contraction helper.
#[test]
fn lda_third_contraction_matches_kernel_directional_difference() {
    fn kernel_response(
        functional: &ResolvedFunctional,
        rho: [f64; 2],
        transition: [f64; 2],
    ) -> [f64; 2] {
        let ra = [rho[0]];
        let rb = [rho[1]];
        let vars = GridVars {
            rho: [&ra, &rb],
            sigma: None,
            tau: None,
        };
        let mut out = XcOut::default();
        functional.eval(&vars, 2, &mut out).unwrap();
        [
            out.v2rho2[0][0] * transition[0] + out.v2rho2[1][0] * transition[1],
            out.v2rho2[1][0] * transition[0] + out.v2rho2[2][0] * transition[1],
        ]
    }

    let func = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let rho = [0.071, 0.043];
    let transition = [0.019, -0.013];
    let analytic = lda_third_density_contraction(&func, rho, transition).unwrap();
    let h = 2e-5;
    let plus = kernel_response(
        &func,
        [rho[0] + h * transition[0], rho[1] + h * transition[1]],
        transition,
    );
    let minus = kernel_response(
        &func,
        [rho[0] - h * transition[0], rho[1] - h * transition[1]],
        transition,
    );
    for spin in 0..2 {
        let fd = (plus[spin] - minus[spin]) / (2.0 * h);
        let scale = 1.0_f64.max(fd.abs()).max(analytic[spin].abs());
        assert!(
            (analytic[spin] - fd).abs() < 2e-7 * scale,
            "spin {spin}: analytic {:.12e} vs kernel FD {fd:.12e}",
            analytic[spin]
        );
    }
}

/// Translational invariance: displacing the whole molecule cannot change
/// the energy, so the gradient must sum to zero over atoms. This is the
/// check that the per-centre attribution of every derivative integral is
/// right — an error in which atom a term belongs to leaves the total
/// energy derivative intact and shows up only here.
#[test]
fn gradient_obeys_the_translational_sum_rule() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let point = solve(&mol, "6-31g", &func, 3);
    let grad = tda_gradient(
        &mol,
        &point.mb,
        &func,
        &point.scf,
        &point.states[0],
        &options(),
    )
    .unwrap();
    let sum = grad.iter().fold(Vec3::zero(), |acc, g| acc + *g);
    for (ax, value) in [sum.x, sum.y, sum.z].iter().enumerate() {
        assert!(
            value.abs() < 1e-9,
            "axis {ax}: gradient sums to {value:e}, not zero"
        );
    }
}

#[test]
fn svwn5_gradient_obeys_the_translational_sum_rule() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let point = solve(&mol, "sto-3g", &func, 3);
    let grad = tda_gradient_with_grid(
        &mol,
        &point.mb,
        &func,
        point.grid.as_ref(),
        &point.scf,
        &point.states[0],
        &options(),
    )
    .unwrap();
    let sum = grad.iter().fold(Vec3::zero(), |acc, g| acc + *g);
    for (ax, value) in [sum.x, sum.y, sum.z].iter().enumerate() {
        assert!(
            value.abs() < 1e-9,
            "SVWN5 axis {ax}: gradient sums to {value:e}, not zero"
        );
    }
}

/// Zeroing the amplitudes must collapse the excited-state gradient onto
/// the ordinary RHF gradient: with no excitation there is no difference
/// density, no Z-vector, and the Lagrangian reduces to the energy-weighted
/// density. This pins the ground-state limb of the assembly independently
/// of anything the excited state contributes.
#[test]
fn zero_amplitudes_reproduce_the_ground_state_gradient() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let point = solve(&mol, "6-31g", &func, 1);
    let mut state = point.states.into_iter().next().unwrap();
    for spin in 0..2 {
        for v in state.x[spin].as_mut_slice() {
            *v = 0.0;
        }
    }
    let grad = tda_gradient(&mol, &point.mb, &func, &point.scf, &state, &options()).unwrap();
    let reference =
        dft_rs::gradient::assemble_gradient(&mol, &point.mb, &func, None, &point.scf, &options())
            .unwrap();
    for atom in 0..mol.len() {
        for ax in 0..3 {
            let (a, b) = (component(grad[atom], ax), component(reference[atom], ax));
            assert!(
                (a - b).abs() < 1e-10,
                "atom {atom} axis {ax}: {a:.12} vs RHF gradient {b:.12}"
            );
        }
    }
}

#[test]
fn svwn5_zero_amplitudes_reproduce_the_ground_state_gradient() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let point = solve(&mol, "sto-3g", &func, 1);
    let mut state = point.states.into_iter().next().unwrap();
    for spin in 0..2 {
        for value in state.x[spin].as_mut_slice() {
            *value = 0.0;
        }
    }
    let grad = tda_gradient_with_grid(
        &mol,
        &point.mb,
        &func,
        point.grid.as_ref(),
        &point.scf,
        &state,
        &options(),
    )
    .unwrap();
    let reference = dft_rs::gradient::assemble_gradient(
        &mol,
        &point.mb,
        &func,
        point.grid.as_ref(),
        &point.scf,
        &options(),
    )
    .unwrap();
    for atom in 0..mol.len() {
        for ax in 0..3 {
            let (a, b) = (component(grad[atom], ax), component(reference[atom], ax));
            assert!(
                (a - b).abs() < 1e-10,
                "SVWN5 atom {atom} axis {ax}: {a:.12} vs ground gradient {b:.12}"
            );
        }
    }
}

/// Unsupported DFT families and a missing LDA grid must be typed refusals,
/// not silently incomplete forces.
#[test]
fn unsupported_dft_reference_is_refused_rather_than_approximated() {
    let mol = water();
    let hf = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let point = solve(&mol, "sto-3g", &hf, 1);
    let b3lyp = resolve(&FunctionalSpec::Combined("b3lyp".into())).unwrap();
    let err = tda_gradient(
        &mol,
        &point.mb,
        &b3lyp,
        &point.scf,
        &point.states[0],
        &options(),
    )
    .expect_err("a hybrid-GGA reference must be refused");
    let message = err.to_string();
    assert!(
        message.contains("GGA") || message.contains("meta-GGA"),
        "the refusal must name the unsupported family, got: {message}"
    );

    let svwn5 = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let err = tda_gradient(
        &mol,
        &point.mb,
        &svwn5,
        &point.scf,
        &point.states[0],
        &options(),
    )
    .expect_err("an LDA gradient without its SCF/TDA grid must be refused")
    .to_string();
    assert!(
        err.contains("same molecular grid"),
        "the refusal must request the original grid, got: {err}"
    );
}

/// Triplet and full-RPA states carry amplitudes that look exactly like
/// singlet TDA ones — same shape, same magnitudes, no numerical tell. The
/// response densities built from them would be wrong and nothing
/// downstream could notice, so the state has to declare what it is and the
/// gradient has to check.
#[test]
fn triplet_and_rpa_states_are_refused() {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &func, None).unwrap();

    let triplet = TdaSolver::new(&mol, &mb, &func, None, &scf, &opts, TdaSpin::Triplet)
        .unwrap()
        .solve_dense(1)
        .unwrap();
    let err = tda_gradient(&mol, &mb, &func, &scf, &triplet[0], &opts)
        .expect_err("a triplet state must be refused")
        .to_string();
    assert!(
        err.contains("singlet"),
        "the refusal must name the manifold, got: {err}"
    );

    let rpa = TdaSolver::new(&mol, &mb, &func, None, &scf, &opts, TdaSpin::Singlet)
        .unwrap()
        .solve_rpa(1)
        .unwrap();
    let err = tda_gradient(&mol, &mb, &func, &scf, &rpa[0], &opts)
        .expect_err("a full-RPA state must be refused")
        .to_string();
    assert!(
        err.contains("Tamm-Dancoff"),
        "the refusal must say the amplitudes are the wrong kind, got: {err}"
    );
}
