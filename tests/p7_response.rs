// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 7 gates: external field ↔ dipole consistency, analytic
//! polarizability vs finite field (independent numerical route), Raman
//! activity sanity.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::properties::dipole_moment;
use dft_rs::response::polarizability::polarizability;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.2217),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.4309, -0.8867),
        },
    ])
}

fn tight() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn setup(name: &str) -> (Molecule, ResolvedFunctional, Option<MolecularGrid>) {
    let mol = water();
    let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let grid = if func.parts.is_empty() {
        None
    } else {
        Some(
            MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: 5,
                    ..GridOptions::default()
                },
            )
            .unwrap(),
        )
    };
    (mol, func, grid)
}

/// dE/dE_ax = −μ_ax: the field implementation is exactly the dipole's
/// conjugate.
#[test]
fn field_energy_derivative_is_minus_dipole() {
    for name in ["hf", "b3lyp"] {
        let (mol, func, grid) = setup(name);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let opts = tight();
        let scf0 = run_scf(&mol, &mb, &opts, &func, grid.as_ref()).unwrap();
        let mu = dipole_moment(&mol, &mb, &scf0);
        let h = 1e-4;
        for ax in 0..3 {
            let mut fp = opts.clone();
            let mut fm = opts.clone();
            let mut ef = Vec3::zero();
            match ax {
                0 => ef.x = h,
                1 => ef.y = h,
                _ => ef.z = h,
            }
            fp.external_field = Some(ef);
            fm.external_field = Some(ef * -1.0);
            let ep = run_scf(&mol, &mb, &fp, &func, grid.as_ref())
                .unwrap()
                .energy;
            let em = run_scf(&mol, &mb, &fm, &func, grid.as_ref())
                .unwrap()
                .energy;
            let de = (ep - em) / (2.0 * h);
            assert!(
                (de + mu.get(ax)).abs() < 1e-6,
                "{name} axis {ax}: dE/dE = {de} vs −μ = {}",
                -mu.get(ax)
            );
        }
    }
}

/// The analytic gradient in a field, against central differences of the
/// field-perturbed SCF energy.
///
/// Two pieces are added by the field and both are checked here at once: the
/// electronic `Σ_c E_c Tr[P ∂D^c/∂R_A]`, which needs the geometric
/// derivative of the dipole integrals, and the nuclear `−Z_A E`. Getting
/// one and not the other still leaves a gradient that looks reasonable —
/// they are comparable in size and opposite in sign for a typical polar
/// molecule — so the finite difference is the only thing that separates
/// them.
///
/// The field is deliberately not along a symmetry axis of the water, so no
/// component of either term can vanish for a reason unrelated to its
/// correctness. Both HF and a GGA hybrid are run, because the field enters
/// through `H_core` and must therefore leave the XC grid response alone.
#[test]
fn analytic_gradient_in_a_field_matches_finite_differences() {
    use dft_rs::gradient::assemble_gradient;
    let field = Vec3::new(3e-3, -5e-3, 2e-3);
    for name in ["hf", "b3lyp"] {
        let (mol, func, grid) = setup(name);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let mut opts = tight();
        opts.external_field = Some(field);

        let scf = run_scf(&mol, &mb, &opts, &func, grid.as_ref()).unwrap();
        let analytic = assemble_gradient(&mol, &mb, &func, grid.as_ref(), &scf, &opts).unwrap();

        // A field-free run of the same molecule must differ: otherwise the
        // test could pass with the field silently ignored on both sides.
        let mut free = opts.clone();
        free.external_field = None;
        let scf_free = run_scf(&mol, &mb, &free, &func, grid.as_ref()).unwrap();
        let field_free =
            assemble_gradient(&mol, &mb, &func, grid.as_ref(), &scf_free, &free).unwrap();
        let shift = analytic
            .iter()
            .zip(&field_free)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            shift > 1e-4,
            "{name}: the field changed the gradient by only {shift:e}; the test would \
             pass even if the field term were missing"
        );

        let h = 1e-4;
        for atom in 0..mol.len() {
            for ax in 0..3 {
                let energy_at = |delta: f64| {
                    let mut moved = mol.clone();
                    match ax {
                        0 => moved.atoms[atom].position.x += delta,
                        1 => moved.atoms[atom].position.y += delta,
                        _ => moved.atoms[atom].position.z += delta,
                    }
                    let mb = MolBasis::build(&moved, &set).unwrap();
                    let grid = grid.as_ref().map(|_| {
                        MolecularGrid::build(
                            &moved,
                            &GridOptions {
                                level: 5,
                                ..GridOptions::default()
                            },
                        )
                        .unwrap()
                    });
                    run_scf(&moved, &mb, &opts, &func, grid.as_ref())
                        .unwrap()
                        .energy
                };
                let fd = (energy_at(h) - energy_at(-h)) / (2.0 * h);
                let got = analytic[atom].get(ax);
                assert!(
                    (got - fd).abs() < 5e-6,
                    "{name} atom {atom} axis {ax}: analytic {got} vs finite difference {fd}"
                );
            }
        }

        // Translational sum rule in a field. A rigid shift changes the
        // energy by −Q_tot R·E, so the gradients sum to −Q_tot E rather
        // than to zero; for this neutral molecule that is zero again, but
        // only because the electronic and nuclear halves cancel each other
        // — which is exactly the cancellation a missing half would break.
        let total = analytic.iter().fold(Vec3::zero(), |acc, g| acc + *g);
        assert!(
            total.norm() < 5e-7,
            "{name}: gradients in a field do not sum to zero for a neutral molecule: {total:?}"
        );
    }
}

/// Raman activities end-to-end: numerical hessian → modes → dα/dQ.
/// Water: all three modes Raman-active, symmetric stretch strongest.
#[test]
fn raman_activities_water_hf() {
    use dft_rs::dispersion::Dispersion;
    use dft_rs::hessian::numerical_hessian;
    use dft_rs::properties::vibrational::vibrational_analysis;
    use dft_rs::response::polarizability::raman_activities;
    let (mol, func, _) = setup("hf");
    let set = data::load("sto-3g").unwrap();
    let opts = tight();
    let hess = numerical_hessian(
        &mol,
        &set,
        &opts,
        &func,
        "hf",
        None,
        &Dispersion::None,
        5e-3,
    )
    .unwrap();
    let vib = vibrational_analysis(&mol, &hess.hessian, Some(&hess.dipole_derivatives)).unwrap();
    assert_eq!(vib.frequencies_cm.len(), 3);
    let act = raman_activities(&mol, &set, &opts, &func, None, &vib.modes, 5e-3).unwrap();
    assert_eq!(act.len(), 3);
    for (k, &a) in act.iter().enumerate() {
        assert!(a >= 0.0 && a.is_finite(), "mode {k}: activity {a}");
    }
    // At least one strongly active mode (stretches).
    assert!(act.iter().cloned().fold(0.0_f64, f64::max) > 1.0, "{act:?}");
}

/// Analytic CPKS polarizability vs finite-field dμ/dE — validates J/K
/// response AND the XC kernel contraction (LDA + GGA + hybrid).
#[test]
fn analytic_polarizability_matches_finite_field() {
    for name in ["hf", "svwn5", "b3lyp"] {
        let (mol, func, grid) = setup(name);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let opts = tight();
        let scf0 = run_scf(&mol, &mb, &opts, &func, grid.as_ref()).unwrap();
        let alpha = polarizability(&mol, &mb, &func, grid.as_ref(), &scf0, &opts).unwrap();
        // Symmetry of the tensor.
        for x in 0..3 {
            for y in 0..x {
                assert!(
                    (alpha[x][y] - alpha[y][x]).abs() < 1e-6,
                    "{name}: α not symmetric"
                );
            }
        }
        let h = 2e-4;
        for y in 0..3 {
            let mut ef = Vec3::zero();
            match y {
                0 => ef.x = h,
                1 => ef.y = h,
                _ => ef.z = h,
            }
            let mut fp = opts.clone();
            fp.external_field = Some(ef);
            let mut fm = opts.clone();
            fm.external_field = Some(ef * -1.0);
            let sp = run_scf(&mol, &mb, &fp, &func, grid.as_ref()).unwrap();
            let sm = run_scf(&mol, &mb, &fm, &func, grid.as_ref()).unwrap();
            let mup = dipole_moment(&mol, &mb, &sp);
            let mum = dipole_moment(&mol, &mb, &sm);
            for x in 0..3 {
                let ff = (mup.get(x) - mum.get(x)) / (2.0 * h);
                assert!(
                    (alpha[x][y] - ff).abs() < 1e-4,
                    "{name} α[{x}][{y}] = {} vs finite-field {ff}",
                    alpha[x][y]
                );
            }
        }
    }
}
