// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Theory-level gates for reusable foundations added after the first PBC
//! implementation.  These test identities, not fitted reference numbers.

use dft_rs::basis::{data, MolBasis};
use dft_rs::math::Vec3;
use dft_rs::pcm::{smd_cds_energy, PcmModel, PcmOptions, PcmSolver, SmdSolvent};
use dft_rs::system::{Atom, Molecule};

fn hydrogen_solver(eps: f64) -> PcmSolver {
    let molecule = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    PcmSolver::build(
        &molecule,
        &basis,
        &[1.0],
        &PcmOptions {
            model: PcmModel::CPcm,
            eps,
            n_points: 86,
            ..PcmOptions::default()
        },
    )
    .unwrap()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn cpcm_response_is_linear_reciprocal_and_negative_semidefinite() {
    let solver = hydrogen_solver(20.0);
    let n = solver.n_points();
    let v1: Vec<f64> = (0..n)
        .map(|i| 0.2 + ((7 * i + 3) % 19) as f64 / 17.0)
        .collect();
    let v2: Vec<f64> = (0..n)
        .map(|i| -0.1 + ((11 * i + 5) % 23) as f64 / 29.0)
        .collect();
    let q1 = solver.charges(&v1).unwrap();
    let q2 = solver.charges(&v2).unwrap();
    let minus_v1: Vec<f64> = v1.iter().map(|value| -*value).collect();
    let q_minus = solver.charges(&minus_v1).unwrap();
    for (&positive, &negative) in q1.iter().zip(&q_minus) {
        assert!((positive + negative).abs() < 2e-12);
    }
    let scale = 2.7;
    let scaled_v1: Vec<f64> = v1.iter().map(|value| scale * value).collect();
    let q_scaled = solver.charges(&scaled_v1).unwrap();
    for (&base, &scaled) in q1.iter().zip(&q_scaled) {
        assert!((scaled - scale * base).abs() < 3e-12);
    }
    let energy = 0.5 * dot(&q1, &v1);
    let scaled_energy = 0.5 * dot(&q_scaled, &scaled_v1);
    assert!(
        energy < 0.0,
        "passive dielectric raised the energy: {energy}"
    );
    assert!((scaled_energy - scale * scale * energy).abs() < 2e-11);
    assert!(
        (dot(&q1, &v2) - dot(&q2, &v1)).abs() < 2e-11,
        "C-PCM response operator is not reciprocal"
    );
}

#[test]
fn cpcm_fixed_potential_scales_with_the_exact_dielectric_factor() {
    let low = hydrogen_solver(4.0);
    let high = hydrogen_solver(80.0);
    assert_eq!(low.n_points(), high.n_points());
    let potential: Vec<f64> = (0..low.n_points())
        .map(|i| 0.3 + (i % 13) as f64 / 31.0)
        .collect();
    let q_low = low.charges(&potential).unwrap();
    let q_high = high.charges(&potential).unwrap();
    let f_low = (4.0 - 1.0) / 4.0;
    let f_high = (80.0 - 1.0) / 80.0;
    let ratio = f_high / f_low;
    for (&a, &b) in q_low.iter().zip(&q_high) {
        assert!((b - ratio * a).abs() < 2e-12);
    }
    let e_low = 0.5 * dot(&q_low, &potential);
    let e_high = 0.5 * dot(&q_high, &potential);
    assert!((e_high - ratio * e_low).abs() < 2e-11);
}

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.43, 1.11),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.43, 1.11),
        },
    ])
}

#[test]
fn smd_cds_is_invariant_to_rigid_motion_and_atom_permutation() {
    let original = water();
    let reference = smd_cds_energy(&original, SmdSolvent::Water, 512).unwrap();
    let moved = Molecule::new(
        original
            .atoms
            .iter()
            .map(|atom| Atom {
                z: atom.z,
                // Proper 90-degree rotation followed by a translation.
                position: Vec3::new(
                    -atom.position.y + 4.2,
                    atom.position.x - 1.7,
                    atom.position.z + 0.6,
                ),
            })
            .collect(),
    );
    let moved_energy = smd_cds_energy(&moved, SmdSolvent::Water, 512).unwrap();
    assert!((moved_energy - reference).abs() < 2e-13);

    let permuted = Molecule::new(vec![
        original.atoms[2].clone(),
        original.atoms[0].clone(),
        original.atoms[1].clone(),
    ]);
    let permuted_energy = smd_cds_energy(&permuted, SmdSolvent::Water, 512).unwrap();
    assert!((permuted_energy - reference).abs() < 2e-13);
}

#[test]
fn smd_cds_permutation_invariance_does_not_rely_on_molecular_symmetry() {
    let molecule = Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.5, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 1.5, 0.0),
        },
        Atom {
            z: 6,
            position: Vec3::new(2.5, 0.2, -0.1),
        },
        Atom {
            z: 1,
            position: Vec3::new(3.4, 1.3, -0.6),
        },
    ]);
    let reference = smd_cds_energy(&molecule, SmdSolvent::Water, 384).unwrap();
    for permutation in [
        [2usize, 4, 0, 3, 1],
        [4usize, 1, 3, 0, 2],
        [1usize, 0, 4, 2, 3],
    ] {
        let permuted = Molecule::new(
            permutation
                .iter()
                .map(|&index| molecule.atoms[index].clone())
                .collect(),
        );
        let energy = smd_cds_energy(&permuted, SmdSolvent::Water, 384).unwrap();
        assert!(
            (energy - reference).abs() < 5e-13,
            "SMD CDS changed under atom permutation: {energy} vs {reference}"
        );
    }
}

/// The molecular quadrature grid converges, and does so regardless of how
/// the molecule is oriented relative to it.
///
/// Classification: **asymptotic**. None of these are exact at a finite grid
/// level and asserting them as equalities would be wrong. What must hold is
/// that the error SHRINKS as the grid is refined.
///
/// Three properties, none of which had any gate:
///
/// 1. `integral(rho) -> N`. The electron count is checked at one level in
///    `p3_dft_oracle`, but its convergence never was.
/// 2. Orientation. The atom-centred product grid is built in the molecular
///    frame, so rotating the molecule rotates the grid with it -- but only
///    exactly in the infinite-level limit, because the angular rule is not
///    invariant under arbitrary rotations. A finite-level energy therefore
///    moves slightly when the molecule is rotated, and that movement must
///    vanish with refinement. This is the property most likely to be
///    silently degraded by a change to the Becke partition or the atomic
///    radii, and nothing was watching it.
/// 3. Pruning. `GridOptions::prune` is set to `false` in every other test in
///    this repository, so the pruned path was effectively unexercised. A
///    pruned grid must approach the same answer as an unpruned one.
#[test]
fn grid_quadrature_converges_and_is_orientation_stable() {
    let mol = water();
    let set = dft_rs::basis::data::load("sto-3g").unwrap();
    let mb = dft_rs::basis::MolBasis::build(&mol, &set).unwrap();
    let func = dft_rs::xc::resolve(&dft_rs::xc::FunctionalSpec::Combined("svwn5".into())).unwrap();
    let opts = dft_rs::scf::ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: dft_rs::scf::EriMode::InCore { limit_mb: 256 },
        ..dft_rs::scf::ScfOptions::default()
    };

    // A generic rotation: no axis aligned with a coordinate axis, so it
    // genuinely reorients the angular rule rather than permuting it.
    let axis = Vec3::new(1.0, -2.0, 3.0);
    let u = axis * (1.0 / axis.norm());
    let angle = 0.731_f64;
    let (c, s) = (angle.cos(), angle.sin());
    let rotate = |r: Vec3| r * c + u.cross(r) * s + u * (u.dot(r) * (1.0 - c));
    let rotated = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| Atom {
                z: a.z,
                position: rotate(a.position),
            })
            .collect(),
    );
    let mb_rot = dft_rs::basis::MolBasis::build(&rotated, &set).unwrap();

    let run = |m: &Molecule, b: &dft_rs::basis::MolBasis, level: usize, prune: bool| {
        let g = dft_rs::grid::MolecularGrid::build(
            m,
            &dft_rs::grid::GridOptions {
                level,
                prune,
                ..dft_rs::grid::GridOptions::default()
            },
        )
        .unwrap();
        let scf = dft_rs::scf::run_scf(m, b, &opts, &func, Some(&g)).unwrap();
        (scf.energy, scf.grid_electron_count.unwrap())
    };

    let levels = [1usize, 3, 5];
    let mut charge_err = Vec::new();
    let mut orient_err = Vec::new();
    let mut prune_err = Vec::new();
    for &level in &levels {
        let (e, ne) = run(&mol, &mb, level, false);
        let (e_rot, _) = run(&rotated, &mb_rot, level, false);
        let (e_pruned, _) = run(&mol, &mb, level, true);
        charge_err.push((ne - 10.0).abs());
        orient_err.push((e - e_rot).abs());
        prune_err.push((e - e_pruned).abs());
    }

    // Pruning is NOT a convergence sequence, so it is asserted differently:
    // it removes points whose contribution is negligible, giving a
    // different-but-equivalent grid rather than a coarser one. Measured, the
    // difference is 1.3e-13, 3.4e-13, 1.0e-9 at levels 1, 3, 5 -- i.e. the
    // pruned and unpruned grids are literally identical at the two coarse
    // levels (the angular degree there is already at its floor, so there is
    // nothing to prune) and differ by 1e-9 at level 5. Requiring that to
    // SHRINK would be requiring noise to shrink. What matters is that it
    // stays negligible, which is what the pruning is promising.
    for (i, &level) in levels.iter().enumerate() {
        assert!(
            prune_err[i] < 1e-6,
            "pruned vs unpruned energy differs by {} at level {level}; pruning may only \
             drop points whose contribution is negligible",
            prune_err[i]
        );
    }

    // Monotone improvement, not equality.
    for (name, errs) in [
        ("integral(rho) - N", &charge_err),
        ("energy under rotation", &orient_err),
    ] {
        assert!(
            errs.iter().all(|e| e.is_finite()),
            "{name}: non-finite error {errs:?}"
        );
        assert!(
            errs[levels.len() - 1] < errs[0],
            "{name} did not improve with grid level: {errs:?} at levels {levels:?}. \
             A finite grid cannot make these exact, but refinement must reduce them -- \
             if it does not, the Becke partition or the atomic radii are inconsistent \
             rather than merely coarse."
        );
    }

    // And the refined values must actually be small, so "improving" cannot
    // be satisfied by starting absurd and ending merely bad.
    assert!(
        charge_err[levels.len() - 1] < 1e-5,
        "integral(rho) is {} from N at the finest level tested",
        charge_err[levels.len() - 1]
    );
    assert!(
        orient_err[levels.len() - 1] < 1e-6,
        "the energy still moves by {} under a rigid rotation at the finest level tested",
        orient_err[levels.len() - 1]
    );
}
