// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 4.5 gate: LANL2DZ ECP energies (HF + DFT) against the frozen
//! PySCF oracle (basis AND ECP data shared from our bundled g94 file).

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

struct EcpCase {
    name: String,
    functional: String,
    molecule: Molecule,
    e_tot: f64,
}

fn load_cases() -> Vec<EcpCase> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with("_ecp.csv") {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let mut functional = String::new();
        let mut charge = 0i32;
        let mut mult = 1usize;
        let mut atoms = Vec::new();
        let mut e_tot = f64::NAN;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p: Vec<&str> = line.split(',').collect();
            match p[0] {
                "basis" => assert_eq!(p[1], "lanl2dz"),
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
                "e_tot" => e_tot = p[1].parse().unwrap(),
                other => panic!("unknown key {other}"),
            }
        }
        cases.push(EcpCase {
            name: fname.trim_end_matches("_ecp.csv").into(),
            functional,
            molecule: Molecule::new(atoms)
                .with_charge(charge)
                .with_multiplicity(mult),
            e_tot,
        });
    }
    assert!(
        cases.len() >= 4,
        "expected 4+ ECP cases, found {}",
        cases.len()
    );
    cases
}

/// The full ECP integral matrix for EVERY first-row transition metal
/// (Sc..Zn) plus Pd, on M-H probes, against PySCF's `ECPscalar` in the
/// dft-rs AO ordering - a state-free per-element gate over all semi-local
/// channels (lmax, radial powers, exponents).
#[test]
fn transition_metal_ecp_matrices_match_pyscf() {
    use dft_rs::integrals::ecp_matrix;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with("_ecpmat.csv") {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let mut atoms = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p: Vec<&str> = line.split(',').collect();
            match p[0] {
                "basis" => assert_eq!(p[1], "lanl2dz"),
                "atom" => atoms.push(Atom {
                    z: p[1].parse().unwrap(),
                    position: Vec3::new(
                        p[2].parse().unwrap(),
                        p[3].parse().unwrap(),
                        p[4].parse().unwrap(),
                    ),
                }),
                "row" => rows.push(p[2..].iter().map(|v| v.parse().unwrap()).collect()),
                other => panic!("unknown key {other}"),
            }
        }
        let mol = Molecule::new(atoms);
        let set = data::load("lanl2dz").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        assert_eq!(mb.nao_sph, rows.len(), "{fname}: dimension mismatch");
        let u = ecp_matrix(&mb).unwrap();
        let mut max_diff = 0.0f64;
        for (r, row) in rows.iter().enumerate() {
            for (c, want) in row.iter().enumerate() {
                max_diff = max_diff.max((u[(r, c)] - want).abs());
            }
        }
        assert!(
            max_diff < 1e-9,
            "{fname}: max |ours - PySCF| = {max_diff:.3e}"
        );
        checked += 1;
    }
    assert!(
        checked >= 11,
        "expected 11+ metal ECP matrices, got {checked}"
    );
}

/// ECP analytic gradient vs central differences of the SCF energy
/// (NaH/LANL2DZ HF: type-1 + type-2 derivative blocks + Z_eff terms).
#[test]
fn ecp_analytic_gradient_matches_fd() {
    use dft_rs::gradient::{analytic_gradient, numerical_gradient};
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
    let func = dft_rs::xc::ResolvedFunctional::hartree_fock();
    let opts = ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    };
    let res = analytic_gradient(&mol, &mb, &opts, &func, None).unwrap();
    let total: Vec3 = res.gradient.iter().fold(Vec3::zero(), |a, &v| a + v);
    assert!(
        total.norm() < 1e-9,
        "ECP translation sum {:.2e}",
        total.norm()
    );
    let fd = numerical_gradient(&mol, &set, &opts, &func, None, 1e-4).unwrap();
    for (a, (g, f)) in res.gradient.iter().zip(&fd).enumerate() {
        let d = (*g - *f).norm();
        assert!(
            d < 5e-7,
            "ECP atom {a}: analytic {g:?} vs FD {f:?} (|d| = {d:.2e})"
        );
    }
}

/// LANL2DZ ECP total energies vs PySCF: exercises type-1 + type-2
/// integrals, Z_eff nuclear repulsion, and the valence electron count.
#[test]
fn lanl2dz_energies_match_pyscf_oracle() {
    let set = data::load("lanl2dz").unwrap();
    let mut failures = Vec::new();
    for case in load_cases() {
        let mb = MolBasis::build(&case.molecule, &set).unwrap();
        assert!(!mb.ecps.is_empty(), "{}: no ECPs picked up", case.name);
        let func = resolve(&FunctionalSpec::Combined(case.functional.clone())).unwrap();
        let grid = if func.parts.is_empty() {
            None
        } else {
            Some(
                MolecularGrid::build(
                    &case.molecule,
                    &GridOptions {
                        level: 9,
                        ..GridOptions::default()
                    },
                )
                .unwrap(),
            )
        };
        let opts = ScfOptions {
            e_tol: 1e-11,
            p_tol: 1e-9,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        match run_scf(&case.molecule, &mb, &opts, &func, grid.as_ref()) {
            Ok(scf) => {
                let mulliken = dft_rs::properties::mulliken_charges(&case.molecule, &mb, &scf);
                let lowdin = dft_rs::properties::lowdin_charges(&case.molecule, &mb, &scf).unwrap();
                for (name, charges) in [("Mulliken", mulliken), ("Lowdin", lowdin)] {
                    let total: f64 = charges.iter().sum();
                    if (total - case.molecule.charge as f64).abs() > 2e-8 {
                        failures.push(format!(
                            "{}: {name} charges sum to {total}, expected {}",
                            case.name, case.molecule.charge
                        ));
                    }
                }
                let tol = if func.parts.is_empty() { 1e-7 } else { 5e-7 };
                if (scf.energy - case.e_tot).abs() > tol {
                    failures.push(format!(
                        "{}: E {} vs oracle {} (delta = {:.3e})",
                        case.name,
                        scf.energy,
                        case.e_tot,
                        scf.energy - case.e_tot
                    ));
                }
            }
            Err(e) => failures.push(format!("{}: SCF failed: {e}", case.name)),
        }
    }
    assert!(
        failures.is_empty(),
        "ECP oracle failures:\n{}",
        failures.join("\n")
    );
}

/// An ECP must replace the same core charge in all three places where it
/// matters: the electron count, electron-nuclear Hamiltonian, and
/// nucleus-nucleus repulsion. NaH/LANL2DZ is especially transparent:
/// Na's ten-electron core leaves Z_eff=1 and exactly two valence electrons.
#[test]
fn ecp_core_replacement_preserves_charge_bookkeeping() {
    let distance = 3.566;
    let mol = Molecule::new(vec![
        Atom {
            z: 11,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, distance),
        },
    ]);
    let set = data::load("lanl2dz").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    assert_eq!(mb.n_core_electrons, 10);
    assert_eq!(mb.effective_charges(&mol), vec![1.0, 1.0]);
    assert!((mb.nuclear_repulsion(&mol) - 1.0 / distance).abs() < 1e-14);
    let n_valence = mol.n_electrons().unwrap() - mb.n_core_electrons as usize;
    assert_eq!(n_valence, 2);
    assert_eq!(mol.n_alpha_beta_for(n_valence).unwrap(), (1, 1));
}
