// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 9b gates: VV10 nonlocal correlation vs the self-consistent PySCF
//! oracle (post-SCF evaluation; the SC correction is µEh-scale).

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::nlc::{vv10_energy, vv10_fxc_contraction, vv10_potential, Vv10Params};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

/// The matrix returned by `vv10_potential` must be the first functional
/// derivative of the reported VV10 energy. This is the variational identity
/// required before the potential may be placed in the Kohn-Sham Fock matrix;
/// an oracle energy alone cannot detect a missing factor in that derivative.
#[test]
fn vv10_potential_is_the_density_derivative_of_the_energy() {
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1, 0.0, 0.7),
        },
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 2,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let mut p = dft_rs::linalg::Matrix::zeros(2, 2);
    p[(0, 0)] = 1.0;
    p[(0, 1)] = 0.2;
    p[(1, 0)] = 0.2;
    p[(1, 1)] = 0.8;
    let mut dp = dft_rs::linalg::Matrix::zeros(2, 2);
    dp[(0, 0)] = 0.10;
    dp[(0, 1)] = -0.03;
    dp[(1, 0)] = -0.03;
    dp[(1, 1)] = 0.05;
    let params = Vv10Params::WB97XV;
    let (energy, potential) = vv10_potential(&mb, &grid, &p, params).unwrap();
    let direct = vv10_energy(&mb, &grid, &p, params).unwrap();
    assert!((energy - direct).abs() < 1e-13);

    let h = 1e-5;
    let mut pp = p.clone();
    let mut pm = p.clone();
    for i in 0..2 {
        for j in 0..2 {
            pp[(i, j)] += h * dp[(i, j)];
            pm[(i, j)] -= h * dp[(i, j)];
        }
    }
    let fd = (vv10_energy(&mb, &grid, &pp, params).unwrap()
        - vv10_energy(&mb, &grid, &pm, params).unwrap())
        / (2.0 * h);
    let mut analytic = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            analytic += potential[(i, j)] * dp[(i, j)];
        }
    }
    assert!(
        (analytic - fd).abs() < 2e-8,
        "Tr[V_nlc dP] = {analytic:.12e}, finite difference = {fd:.12e}"
    );
}

#[test]
fn vv10_matches_pyscf_self_consistent_oracle() {
    // Read the frozen oracle.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/oracle_data/h2o_def2svp_blyp_vv10_nlc.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut atoms = Vec::new();
    let (mut e_blyp_ref, mut e_total_ref, mut e_nlc_ref) = (f64::NAN, f64::NAN, f64::NAN);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "basis" | "functional" => {}
            "atom" => atoms.push(Atom {
                z: p[1].parse().unwrap(),
                position: Vec3::new(
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4].parse().unwrap(),
                ),
            }),
            "e_blyp" => e_blyp_ref = p[1].parse().unwrap(),
            "e_blyp_vv10" => e_total_ref = p[1].parse().unwrap(),
            "e_nlc_sc" => e_nlc_ref = p[1].parse().unwrap(),
            other => panic!("unknown key {other}"),
        }
    }
    let mol = Molecule::new(atoms);
    let set = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("blyp".into())).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 8,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    assert!(
        (scf.energy - e_blyp_ref).abs() < 5e-7,
        "BLYP base energy drifted: {} vs {}",
        scf.energy,
        e_blyp_ref
    );
    // Post-SCF VV10 on a coarse dedicated grid (PySCF used nlc level 3).
    let nlc_grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 3,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let e_nlc = vv10_energy(&mb, &nlc_grid, &scf.p_total(), Vv10Params::ORIGINAL).unwrap();
    assert!(
        (e_nlc - e_nlc_ref).abs() < 2e-5,
        "VV10 energy {e_nlc} vs SC oracle {e_nlc_ref} (Δ = {:.2e})",
        e_nlc - e_nlc_ref
    );
    let total = scf.energy + e_nlc;
    assert!(
        (total - e_total_ref).abs() < 2e-5,
        "BLYP+VV10 total {total} vs oracle {e_total_ref}"
    );
}

/// f_nlc response kernel: the directional second functional derivative
/// (vv10_fxc_contraction) against central differences of the analytic
/// V_nlc potential matrix under P → P ± hΔP.
#[test]
fn vv10_fxc_contraction_matches_fd_of_potential() {
    let mol = Molecule::new(vec![
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
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-10,
        p_tol: 1e-8,
        max_iter: 100,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, None).unwrap();
    let p = scf.p_total();
    let nao = mb.nao_sph;
    // Deterministic symmetric response direction with mixed signs.
    let mut dp = dft_rs::linalg::Matrix::zeros(nao, nao);
    for mu in 0..nao {
        for nu in 0..nao {
            let vm = if mu % 2 == 0 { 1.0 } else { -1.0 } / (1.0 + mu as f64);
            let vn = if nu % 2 == 0 { 1.0 } else { -1.0 } / (1.0 + nu as f64);
            dp[(mu, nu)] = 0.05 * vm * vn + 0.01 / (1.0 + (mu + nu) as f64);
        }
    }
    // Symmetrize.
    for mu in 0..nao {
        for nu in 0..mu {
            let s = 0.5 * (dp[(mu, nu)] + dp[(nu, mu)]);
            dp[(mu, nu)] = s;
            dp[(nu, mu)] = s;
        }
    }
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 1,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let params = Vv10Params::WB97XV;
    let analytic = vv10_fxc_contraction(&mb, &grid, &p, &dp, params).unwrap();
    let h = 1e-4;
    let mut pp = p.clone();
    let mut pm = p.clone();
    for mu in 0..nao {
        for nu in 0..nao {
            pp[(mu, nu)] += h * dp[(mu, nu)];
            pm[(mu, nu)] -= h * dp[(mu, nu)];
        }
    }
    let (_, vp) = vv10_potential(&mb, &grid, &pp, params).unwrap();
    let (_, vm) = vv10_potential(&mb, &grid, &pm, params).unwrap();
    let mut worst = 0.0f64;
    for mu in 0..nao {
        for nu in 0..nao {
            let fd = (vp[(mu, nu)] - vm[(mu, nu)]) / (2.0 * h);
            let d = (analytic[(mu, nu)] - fd).abs();
            if d > worst {
                worst = d;
            }
        }
    }
    assert!(
        worst < 5e-7,
        "f_nlc contraction vs FD of V_nlc: worst |Δ| = {worst:.3e}"
    );
}

/// VV10 analytic Hessian pieces at fixed density: the 3N×3N skeleton
/// against central differences of the analytic fixed-P VV10 gradient, and
/// the D_a V_nlc matrices against central differences of the potential —
/// both under displaced geometries with rebuilt basis/grid (the same
/// owner-riding convention the analytic terms differentiate).
#[test]
fn vv10_hessian_terms_match_fd() {
    use dft_rs::hessian::nlc_terms::nlc_hessian_terms;
    use dft_rs::nlc::vv10_gradient;
    let atoms0 = [
        (8u8, [0.0, 0.05, 0.2217]),
        (1u8, [0.03, 1.4309, -0.8867]),
        (1u8, [0.0, -1.4309, -0.8867]),
    ];
    let build_mol = |shift: Option<(usize, usize, f64)>| -> Molecule {
        let mut list = Vec::new();
        for (k, (z, p)) in atoms0.iter().enumerate() {
            let mut pos = *p;
            if let Some((at, ax, h)) = shift {
                if at == k {
                    pos[ax] += h;
                }
            }
            list.push(Atom {
                z: *z,
                position: Vec3::new(pos[0], pos[1], pos[2]),
            });
        }
        Molecule::new(list)
    };
    let mol = build_mol(None);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-10,
        p_tol: 1e-8,
        max_iter: 100,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, None).unwrap();
    let p_tot = scf.p_total();
    let nao = mb.nao_sph;
    let mut p_half = p_tot.clone();
    for mu in 0..nao {
        for nu in 0..nao {
            p_half[(mu, nu)] *= 0.5;
        }
    }
    let gopts = GridOptions {
        level: 1,
        ..GridOptions::default()
    };
    let grid = MolecularGrid::build(&mol, &gopts).unwrap();
    let params = Vv10Params::WB97XV;
    let terms = nlc_hessian_terms(&mol, &mb, &grid, [&p_half, &p_half], params).unwrap();

    let h = 1e-4;
    let mut worst_skel = 0.0f64;
    let mut worst_v = 0.0f64;
    for yat in 0..3 {
        for yax in 0..3 {
            let ys = 3 * yat + yax;
            let eval_at = |hh: f64| {
                let molp = build_mol(Some((yat, yax, hh)));
                let mbp = MolBasis::build(&molp, &set).unwrap();
                let gridp = MolecularGrid::build(&molp, &gopts).unwrap();
                let g = vv10_gradient(&molp, &mbp, &gridp, &p_tot, params).unwrap();
                let (_, v) = vv10_potential(&mbp, &gridp, &p_tot, params).unwrap();
                (g, v)
            };
            let (gp, vp) = eval_at(h);
            let (gm, vm) = eval_at(-h);
            for xat in 0..3 {
                for xax in 0..3 {
                    let xs = 3 * xat + xax;
                    let fd = (gp[xat].get(xax) - gm[xat].get(xax)) / (2.0 * h);
                    let d = (terms.skeleton[(xs, ys)] - fd).abs();
                    if d > worst_skel {
                        worst_skel = d;
                    }
                }
            }
            for mu in 0..nao {
                for nu in 0..nao {
                    let fd = (vp[(mu, nu)] - vm[(mu, nu)]) / (2.0 * h);
                    let d = (terms.v_deriv[yat][yax][(mu, nu)] - fd).abs();
                    if d > worst_v {
                        worst_v = d;
                    }
                }
            }
        }
    }
    assert!(
        worst_skel < 5e-6,
        "VV10 Hessian skeleton vs FD of analytic gradient: worst |Δ| = {worst_skel:.3e}"
    );
    assert!(
        worst_v < 5e-6,
        "D_a V_nlc vs FD of the potential matrix: worst |Δ| = {worst_v:.3e}"
    );
}

/// SELF-CONSISTENT VV10: the total energy with V_nlc in the Fock matrix
/// against PySCF's self-consistent BLYP+VV10 (both sides SC now, so the
/// gate tightens from the 2e-5 post-SCF delta to the grid level).
#[test]
fn sc_vv10_total_matches_pyscf_self_consistent() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/oracle_data/h2o_def2svp_blyp_vv10_nlc.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut atoms = Vec::new();
    let mut e_total_ref = f64::NAN;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "basis" | "functional" | "e_blyp" | "e_nlc_sc" => {}
            "atom" => atoms.push(Atom {
                z: p[1].parse().unwrap(),
                position: Vec3::new(
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4].parse().unwrap(),
                ),
            }),
            "e_blyp_vv10" => e_total_ref = p[1].parse().unwrap(),
            other => panic!("unknown key {other}"),
        }
    }
    let mol = Molecule::new(atoms);
    let set = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("blyp".into())).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 8,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        vv10: Some(Vv10Params::ORIGINAL),
        vv10_grid_level: 3,
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    assert!(scf.e_nlc > 0.0, "SC VV10 energy missing");
    assert!(
        (scf.energy - e_total_ref).abs() < 5e-6,
        "SC-VV10 total {} vs PySCF {} (delta = {:.2e})",
        scf.energy,
        e_total_ref,
        scf.energy - e_total_ref
    );
}

/// SC-VV10 analytic gradient vs central differences of the SC total
/// energy (the variational property makes fixed-density grid terms exact).
#[test]
fn sc_vv10_gradient_matches_fd() {
    use dft_rs::gradient::{analytic_gradient, numerical_gradient};
    let mol = Molecule::new(vec![
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
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("blyp".into())).unwrap();
    let gopts = GridOptions {
        level: 3,
        ..GridOptions::default()
    };
    let grid = MolecularGrid::build(&mol, &gopts).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        vv10: Some(Vv10Params::WB97XV),
        vv10_grid_level: 2,
        ..ScfOptions::default()
    };
    let res = analytic_gradient(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    let total: Vec3 = res.gradient.iter().fold(Vec3::zero(), |a, &v| a + v);
    assert!(
        total.norm() < 1e-8,
        "SC-VV10 translation sum {:.2e}",
        total.norm()
    );
    let fd = numerical_gradient(&mol, &set, &opts, &func, Some(&gopts), 1e-4).unwrap();
    for (a, (g, f)) in res.gradient.iter().zip(&fd).enumerate() {
        let d = (*g - *f).norm();
        assert!(
            d < 5e-6,
            "SC-VV10 atom {a}: analytic {g:?} vs FD {f:?} (|d| = {d:.2e})"
        );
    }
}

/// VV10 analytic gradient vs central differences of `vv10_energy` at a
/// FIXED density (geometry-rebuilt grid; points ride their atoms).
#[test]
fn vv10_gradient_matches_finite_differences() {
    use dft_rs::nlc::vv10_gradient;
    let mol = Molecule::new(vec![
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
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    // A converged density for realism (functional choice is irrelevant to
    // the fixed-density gradient identity).
    let func = resolve(&FunctionalSpec::Combined("blyp".into())).unwrap();
    let gopts = GridOptions {
        level: 2,
        ..GridOptions::default()
    };
    let grid = MolecularGrid::build(&mol, &gopts).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-10,
        p_tol: 1e-8,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    let p = scf.p_total();
    let g_an = vv10_gradient(&mol, &mb, &grid, &p, Vv10Params::WB97XV).unwrap();
    // Translational sum rule.
    let total: Vec3 = g_an.iter().fold(Vec3::zero(), |a, &v| a + v);
    assert!(
        total.norm() < 1e-8,
        "VV10 translation sum {:.2e}",
        total.norm()
    );
    let h = 1e-4;
    for a in 0..mol.len() {
        for ax in 0..3 {
            let e_at = |sgn: f64| -> f64 {
                let mut mm = mol.clone();
                match ax {
                    0 => mm.atoms[a].position.x += sgn * h,
                    1 => mm.atoms[a].position.y += sgn * h,
                    _ => mm.atoms[a].position.z += sgn * h,
                }
                let mbx = MolBasis::build(&mm, &set).unwrap();
                let gx = MolecularGrid::build(&mm, &gopts).unwrap();
                vv10_energy(&mbx, &gx, &p, Vv10Params::WB97XV).unwrap()
            };
            let fd = (e_at(1.0) - e_at(-1.0)) / (2.0 * h);
            let an = match ax {
                0 => g_an[a].x,
                1 => g_an[a].y,
                _ => g_an[a].z,
            };
            assert!(
                (an - fd).abs() < 1e-6,
                "VV10 atom {a} ax {ax}: analytic {an} vs FD {fd}"
            );
        }
    }
}

/// The complete ωB97M-V (meta-GGA RSH semilocal SCF + post-SCF VV10 with
/// the functional's own b = 6.0, C = 0.01) against the self-consistent
/// PySCF total.
#[test]
fn wb97mv_total_with_vv10_matches_pyscf() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/oracle_data/h2o_def2svp_wb97mv_vv10_nlc.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut atoms = Vec::new();
    let (mut e_sl_ref, mut e_total_ref) = (f64::NAN, f64::NAN);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "basis" | "functional" | "e_nlc_sc" => {}
            "atom" => atoms.push(Atom {
                z: p[1].parse().unwrap(),
                position: Vec3::new(
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4].parse().unwrap(),
                ),
            }),
            "e_semilocal" => e_sl_ref = p[1].parse().unwrap(),
            "e_total" => e_total_ref = p[1].parse().unwrap(),
            other => panic!("unknown key {other}"),
        }
    }
    let mol = Molecule::new(atoms);
    let set = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("wb97m-v".into())).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 9,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let opts = ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    assert!(
        (scf.energy - e_sl_ref).abs() < 2.5e-6,
        "ωB97M-V semilocal SCF {} vs oracle {e_sl_ref}",
        scf.energy
    );
    let nlc_grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 3,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let e_nlc = vv10_energy(&mb, &nlc_grid, &scf.p_total(), Vv10Params::WB97XV).unwrap();
    let total = scf.energy + e_nlc;
    assert!(
        (total - e_total_ref).abs() < 2e-5,
        "ωB97M-V total {total} vs oracle {e_total_ref} (Δ = {:.2e})",
        total - e_total_ref
    );
}
