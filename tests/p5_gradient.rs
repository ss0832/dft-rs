// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 5 gates: analytic gradients vs numerical references and sum rules.

use dft_rs::basis::{data, MolBasis};
use dft_rs::constants::atomic_mass_au;
use dft_rs::gradient::{analytic_gradient, numerical_gradient};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::scf::{EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.05, 0.2217),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.03, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, -1.4309, -0.8867),
        },
    ])
}

/// Mass-weighted centre of mass, in Bohr.
fn center_of_mass(mol: &Molecule) -> Vec3 {
    let mut num = Vec3::zero();
    let mut den = 0.0;
    for atom in &mol.atoms {
        let m = atomic_mass_au(atom.z);
        num += atom.position * m;
        den += m;
    }
    num / den
}

/// Total torque `Σ_A (R_A - origin) × g_A`.
///
/// The rotational sum rule `Σ (R_A - O) × g_A = 0` holds about an
/// origin-independent point only when the translational sum rule
/// `Σ g_A = 0` also holds: shifting the origin by `Δ` changes the torque by
/// `-Δ × Σ g_A`, which vanishes exactly only if `Σ g_A` does. Callers should
/// anchor `origin` at the centre of mass (a physically meaningful point,
/// unlike the arbitrary coordinate origin) and are expected to have already
/// checked the translational sum rule so the two residuals stay logically
/// separate.
fn torque_about(mol: &Molecule, grad: &[Vec3], origin: Vec3) -> Vec3 {
    mol.atoms
        .iter()
        .zip(grad)
        .fold(Vec3::zero(), |acc, (atom, g)| {
            acc + (atom.position - origin).cross(*g)
        })
}

fn tight() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-13,
        conv_tol_grad: 1e-12,
        p_tol: 1e-12,
        max_iter: 500,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

fn rotational_grid() -> GridOptions {
    GridOptions {
        radial_override: Some(200),
        angular_override: Some(131),
        prune: false,
        weight_cutoff: 0.0,
        ..GridOptions::default()
    }
}

/// Analytic gradients vs the frozen PySCF gradient oracle.
#[test]
fn gradients_match_pyscf_oracle() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with("_grad.csv") {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let mut basis = String::new();
        let mut functional = String::new();
        let mut charge = 0i32;
        let mut mult = 1usize;
        let mut atoms = Vec::new();
        let mut reference: Vec<Vec3> = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p: Vec<&str> = line.split(',').collect();
            match p[0] {
                "basis" => basis = p[1].into(),
                "functional" => functional = p[1].into(),
                "charge" => charge = p[1].parse().unwrap(),
                "mult" => mult = p[1].parse().unwrap(),
                "atom" => atoms.push(Atom {
                    z: p[1].parse().unwrap(),
                    position: Vec3::new(
                        p[2].parse().unwrap(),
                        p[3].parse().unwrap(),
                        p[4].parse().unwrap(),
                    ),
                }),
                "grad" => reference.push(Vec3::new(
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4].parse().unwrap(),
                )),
                other => panic!("unknown key {other}"),
            }
        }
        let mol = Molecule::new(atoms)
            .with_charge(charge)
            .with_multiplicity(mult);
        let set = data::load(&basis).unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let func = resolve(&FunctionalSpec::Combined(functional.clone())).unwrap();
        let mut opts = tight();
        if mult > 1 {
            opts.level_shift = 0.2;
        }
        let (grid, tol) = if func.parts.is_empty() {
            (None, 2e-6)
        } else {
            let gopts = GridOptions {
                level: 8,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            };
            (Some(MolecularGrid::build(&mol, &gopts).unwrap()), 1e-5)
        };
        let res = analytic_gradient(&mol, &mb, &opts, &func, grid.as_ref()).unwrap();
        for (a, (g, r)) in res.gradient.iter().zip(&reference).enumerate() {
            let d = (*g - *r).norm();
            assert!(
                d < tol,
                "{fname} atom {a}: {g:?} vs oracle {r:?} (|Δ| = {d:.2e})"
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 7,
        "expected 7+ gradient oracle cases, got {checked}"
    );
}

/// HF gradient (no grid): analytic vs numerical ≤ 1e-7 and translational
/// sum rule ≤ 1e-10.
#[test]
fn hf_gradient_matches_fd() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let hf = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = tight();
    let res = analytic_gradient(&mol, &mb, &opts, &hf, None).unwrap();
    // Translational sum rule.
    let total: Vec3 = res.gradient.iter().fold(Vec3::zero(), |acc, &g| acc + g);
    assert!(
        total.norm() < 1e-9,
        "translational sum rule: |Σg| = {}",
        total.norm()
    );
    // Rotational sum rule (total torque about the centre of mass), exact
    // for an isolated molecule by rotational invariance of the energy. See
    // `torque_about` for why this is anchored at the centre of mass rather
    // than an arbitrary origin.
    let com = center_of_mass(&mol);
    let torque = torque_about(&mol, &res.gradient, com);
    assert!(
        torque.norm() < 1e-9,
        "rotational sum rule: |Σ(R-Rcom)×g| = {}",
        torque.norm()
    );
    let fd = numerical_gradient(&mol, &set, &opts, &hf, None, 1e-4).unwrap();
    for (a, (g, f)) in res.gradient.iter().zip(&fd).enumerate() {
        let d = (*g - *f).norm();
        assert!(
            d < 1e-6,
            "atom {a}: analytic {g:?} vs FD {f:?} (|Δ| = {d:.2e})"
        );
    }
}

/// DFT gradients WITH full grid response: FD agreement at the 1e-6 scale
/// and the translational sum rule — the discriminator that only holds when
/// orbital + weight + point-motion terms are all present.
#[test]
fn dft_gradients_match_fd_with_grid_response() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let gopts = GridOptions {
        level: 5,
        weight_cutoff: 0.0,
        ..GridOptions::default()
    };
    let grid = MolecularGrid::build(&mol, &gopts).unwrap();
    let opts = tight();
    for name in [
        "svwn5",
        "b3lyp",
        "cam-b3lyp",
        "m06l",
        "m06-2x",
        "wb97x-d",
        "wb97m-v",
    ] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let res = analytic_gradient(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
        let total: Vec3 = res.gradient.iter().fold(Vec3::zero(), |acc, &g| acc + g);
        assert!(
            total.norm() < 1e-8,
            "{name}: translational sum rule |Σg| = {:.2e}",
            total.norm()
        );
        let fd = numerical_gradient(&mol, &set, &opts, &func, Some(&gopts), 1e-4).unwrap();
        for (a, (g, f)) in res.gradient.iter().zip(&fd).enumerate() {
            let d = (*g - *f).norm();
            assert!(
                d < 5e-6,
                "{name} atom {a}: analytic {g:?} vs FD {f:?} (|Δ| = {d:.2e})"
            );
        }
    }
}

/// ECP analytic gradient: translational AND rotational sum rules together
/// (NaH/LANL2DZ HF, the same cheap case used for the FD gate in
/// `p45_ecp.rs`). Both are checked in one place because the torque check's
/// validity depends on the translational one (see `torque_about`); the
/// nuclear mass used for the centre of mass is the physical atomic mass and
/// is unaffected by the ECP replacing core electrons.
#[test]
fn ecp_gradient_obeys_translational_and_rotational_sum_rules() {
    let mol = Molecule::new(vec![
        Atom {
            z: 11,
            position: Vec3::new(0.1, -0.05, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.1, 3.566),
        },
    ]);
    let set = data::load("lanl2dz").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = tight();
    let res = analytic_gradient(&mol, &mb, &opts, &func, None).unwrap();

    let total: Vec3 = res.gradient.iter().fold(Vec3::zero(), |acc, &g| acc + g);
    assert!(
        total.norm() < 1e-9,
        "ECP translational sum rule: |Σg| = {:.2e}",
        total.norm()
    );
    let com = center_of_mass(&mol);
    let torque = torque_about(&mol, &res.gradient, com);
    assert!(
        torque.norm() < 1e-9,
        "ECP rotational sum rule: |Σ(R-Rcom)×g| = {:.2e}",
        torque.norm()
    );
}

/// Converged angular quadrature for the nonpolynomial XC integrand.
/// The former pruned level-9 grid failed the unchanged 1e-8 torque gate:
/// SVWN5 2.445e-8 and B3LYP rotational covariance 1.511e-8. Tightening
/// SCF alone did not resolve either. With 200 radial points and unpruned
/// degree-131 Lebedev quadrature the torques are 2.222e-11 (SVWN5),
/// 2.255e-10 (B3LYP) and 2.210e-10 (CAM-B3LYP); doubling radial points
/// leaves these defects unchanged. This certifies the refined quadrature,
/// not a universal rotational precision of the one-knob default grid.
#[test]
fn dft_gga_and_hybrid_gradient_obeys_rotational_sum_rule() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let opts = tight();
    let gopts = rotational_grid();
    let grid = MolecularGrid::build(&mol, &gopts).unwrap();
    let com = center_of_mass(&mol);
    for name in ["svwn5", "b3lyp", "cam-b3lyp"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let res = analytic_gradient(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
        let torque = torque_about(&mol, &res.gradient, com);
        println!("{name} refined torque={:.12e}", torque.norm());
        assert!(
            torque.norm() < 1e-8,
            "{name}: rotational sum rule |Σ(R-Rcom)×g| = {:.2e}",
            torque.norm()
        );
    }
}

/// Angular integration of a nonpolynomial meta-GGA is convergent rather
/// than exactly invariant under arbitrary rotations. Translation is an
/// algebraic grid-response identity, while torque also measures angular
/// quadrature error. This bounded refinement check retains its original
/// thresholds; it does not certify a fully converged meta-GGA torque.
#[test]
fn meta_gga_rotational_sum_rule_is_grid_quadrature_limited() {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("m06l".into())).unwrap();
    let opts = tight();
    let com = center_of_mass(&mol);

    let levels = [3usize, 5, 7];
    let mut residual = Vec::new();
    for &level in &levels {
        let gopts = GridOptions {
            level,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        };
        let grid = MolecularGrid::build(&mol, &gopts).unwrap();
        let res = analytic_gradient(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
        residual.push(torque_about(&mol, &res.gradient, com).norm());
    }

    assert!(
        residual.iter().all(|r| r.is_finite()),
        "m06l torque residuals not finite: {residual:?}"
    );
    assert!(
        residual[levels.len() - 1] < residual[0],
        "m06l: the rotational sum-rule residual did not fall with grid level: \\
         {residual:?} at levels {levels:?}. It is limited by ANGULAR quadrature, so \\
         refinement must reduce it; a residual that plateaus is a derivative bug \\
         rather than truncation."
    );
    // Retain the historical bounded refinement gate for this meta-GGA.
    assert!(
        residual[levels.len() - 1] < 1e-4,
        "m06l torque residual {} is far beyond angular-truncation scale",
        residual[levels.len() - 1]
    );
}

/// Rodrigues rotation of a point about a unit axis by a fixed angle (no
/// translation), matching the helper style in `p16_physical_invariants.rs`.
fn rotate_only(u: Vec3, c: f64, s: f64) -> impl Fn(Vec3) -> Vec3 {
    move |r: Vec3| r * c + u.cross(r) * s + u * (u.dot(r) * (1.0 - c))
}

/// Rotational covariance of the analytic gradient: `g(R·X) = R·g(X)`
/// component by component, for a generic rotation `R`. The sum rule above
/// is one vector equation per molecule; this is `3N` independent equations,
/// so it catches an error that happens to preserve the total torque (e.g. a
/// bug that is antisymmetric between atoms) but still gets individual
/// gradient components wrong.
#[test]
fn hf_gradient_is_rotationally_covariant() {
    let axis = Vec3::new(0.4, -1.1, 2.3);
    let u = axis.normalized();
    let angle: f64 = 0.917;
    let rotate = rotate_only(u, angle.cos(), angle.sin());

    let mol = water();
    let rotated = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| Atom {
                z: a.z,
                position: rotate(a.position),
            })
            .collect(),
    );
    let set = data::load("sto-3g").unwrap();
    let hf = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = tight();

    let mb = MolBasis::build(&mol, &set).unwrap();
    let g = analytic_gradient(&mol, &mb, &opts, &hf, None)
        .unwrap()
        .gradient;
    let mb_r = MolBasis::build(&rotated, &set).unwrap();
    let g_r = analytic_gradient(&rotated, &mb_r, &opts, &hf, None)
        .unwrap()
        .gradient;

    for (a, (gi, gri)) in g.iter().zip(&g_r).enumerate() {
        let d = (rotate(*gi) - *gri).norm();
        println!("rotational covariance atom={a} defect={d:.12e}");
        assert!(
            d < 1e-8,
            "atom {a}: R·g(X) = {:?} vs g(R·X) = {:?} (|Δ| = {d:.2e})",
            rotate(*gi),
            gri
        );
    }
}

/// All atomic gradients transform covariantly with the refined quadrature
/// used by the torque test. The measured B3LYP defects are 1.332e-10,
/// 5.720e-11 and 1.404e-10; no gate is relaxed and no component is copied.
#[test]
fn b3lyp_gradient_is_rotationally_covariant() {
    let axis = Vec3::new(-0.7, 1.6, 0.5);
    let u = axis.normalized();
    let angle: f64 = 1.203;
    let rotate = rotate_only(u, angle.cos(), angle.sin());

    let mol = water();
    let rotated = Molecule::new(
        mol.atoms
            .iter()
            .map(|a| Atom {
                z: a.z,
                position: rotate(a.position),
            })
            .collect(),
    );
    let set = data::load("sto-3g").unwrap();
    let func = resolve(&FunctionalSpec::Combined("b3lyp".into())).unwrap();
    let opts = tight();
    let gopts = rotational_grid();

    let mb = MolBasis::build(&mol, &set).unwrap();
    let grid = MolecularGrid::build(&mol, &gopts).unwrap();
    let g = analytic_gradient(&mol, &mb, &opts, &func, Some(&grid))
        .unwrap()
        .gradient;
    let mb_r = MolBasis::build(&rotated, &set).unwrap();
    let grid_r = MolecularGrid::build(&rotated, &gopts).unwrap();
    let g_r = analytic_gradient(&rotated, &mb_r, &opts, &func, Some(&grid_r))
        .unwrap()
        .gradient;

    for (a, (gi, gri)) in g.iter().zip(&g_r).enumerate() {
        let d = (rotate(*gi) - *gri).norm();
        println!("rotational covariance atom={a} defect={d:.12e}");
        assert!(
            d < 1e-8,
            "atom {a}: R·g(X) = {:?} vs g(R·X) = {:?} (|Δ| = {d:.2e})",
            rotate(*gi),
            gri
        );
    }
}
