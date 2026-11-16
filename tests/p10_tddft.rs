// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 10 gate: TDA excitation energies (Davidson on the Casida A
//! matrix) against the frozen PySCF TDA oracle - CIS (HF), LDA and hybrid
//! kernels, singlet/triplet closed-shell and spin-resolved open-shell.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::tddft::{TdaSolver, TdaSpin};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

/// Run TDA on a molecule given in Bohr and return its states.
fn excitations(atoms: Vec<Atom>, nstates: usize) -> Vec<dft_rs::response::tddft::TdaState> {
    let mol = Molecule::new(atoms);
    let set = data::load("6-31g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &mb, &opts, &func, None).unwrap();
    TdaSolver::new(&mol, &mb, &func, None, &scf, &opts, TdaSpin::Singlet)
        .unwrap()
        .solve(nstates)
        .unwrap()
}

/// An ACHIRAL molecule has no circular dichroism: every rotatory strength
/// must vanish. Water has a mirror plane, so this is exact, and it is the
/// one ECD check that needs no reference data at all — a sign error or a
/// dropped term in either gauge shows up as a non-zero R here.
#[test]
fn achiral_molecules_have_no_rotatory_strength() {
    let states = excitations(
        vec![
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
        ],
        6,
    );
    for (k, st) in states.iter().enumerate() {
        assert!(
            st.rotatory_strength.abs() < 1e-8,
            "state {}: achiral molecule has length-gauge R = {:e}",
            k + 1,
            st.rotatory_strength
        );
        assert!(
            st.rotatory_strength_velocity.abs() < 1e-8,
            "state {}: achiral molecule has velocity-gauge R = {:e}",
            k + 1,
            st.rotatory_strength_velocity
        );
    }
}

/// Mirroring a CHIRAL molecule must flip the sign of every rotatory
/// strength while leaving the excitation energies and oscillator strengths
/// untouched.
///
/// This is the defining property of ECD — it is the whole reason the
/// technique distinguishes enantiomers — and it is a property no
/// magnitude-only check can confirm. A model chiral system (four different
/// substituents around a centre) is enough: the physics under test is the
/// pseudoscalar nature of `mu . m`, not the chemistry.
#[test]
fn mirror_images_have_opposite_rotatory_strengths() {
    // A twisted H2O2-like framework: chirality from the dihedral alone.
    let build = |sign: f64| -> Vec<Atom> {
        vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 1.3900, -0.1000),
            },
            Atom {
                z: 8,
                position: Vec3::new(0.0, -1.3900, -0.1000),
            },
            Atom {
                z: 1,
                position: Vec3::new(sign * 1.6000, 1.7000, 1.0000),
            },
            Atom {
                z: 1,
                position: Vec3::new(-sign * 1.6000, -1.7000, 1.0000),
            },
        ]
    };
    let right = excitations(build(1.0), 5);
    let left = excitations(build(-1.0), 5);
    for (k, (r, l)) in right.iter().zip(&left).enumerate() {
        assert!(
            (r.energy - l.energy).abs() < 1e-9,
            "state {}: enantiomers must be isoenergetic ({} vs {})",
            k + 1,
            r.energy,
            l.energy
        );
        assert!(
            (r.oscillator_strength - l.oscillator_strength).abs() < 1e-9,
            "state {}: enantiomers must share oscillator strengths",
            k + 1
        );
        for (gauge, a, b) in [
            ("length", r.rotatory_strength, l.rotatory_strength),
            (
                "velocity",
                r.rotatory_strength_velocity,
                l.rotatory_strength_velocity,
            ),
        ] {
            assert!(
                (a + b).abs() < 1e-8 * a.abs().max(1.0),
                "state {}: {gauge}-gauge R must invert under reflection, got {a:e} and {b:e}",
                k + 1
            );
        }
    }
    // ...and the effect must actually be present, or the test above would
    // pass trivially on a pair of zeros.
    let biggest = right
        .iter()
        .fold(0.0_f64, |m, s| m.max(s.rotatory_strength.abs()));
    assert!(
        biggest > 1e-6,
        "the model system produced no circular dichroism at all ({biggest:e})"
    );
}

#[test]
fn tda_excitations_match_pyscf_oracle() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut checked = 0;
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        let rpa = fname.ends_with("_rpa.csv");
        if !fname.ends_with("_tda.csv") && !rpa {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let mut basis = String::new();
        let mut functional = String::new();
        let mut spin = String::new();
        let mut charge = 0i32;
        let mut mult = 1usize;
        let mut atoms = Vec::new();
        let mut reference: Vec<(f64, f64)> = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p: Vec<&str> = line.split(',').collect();
            match p[0] {
                "basis" => basis = p[1].into(),
                "functional" => functional = p[1].into(),
                "spin" => spin = p[1].into(),
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
                "excitation" => reference.push((p[1].parse().unwrap(), p[2].parse().unwrap())),
                other => panic!("unknown key {other}"),
            }
        }
        let mol = Molecule::new(atoms)
            .with_charge(charge)
            .with_multiplicity(mult);
        let set = data::load(&basis).unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let func = resolve(&FunctionalSpec::Combined(functional.clone())).unwrap();
        let mut opts = ScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-10,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        if mult > 1 {
            opts.level_shift = 0.2;
        }
        let grid = if func.parts.is_empty() {
            None
        } else {
            Some(
                MolecularGrid::build(
                    &mol,
                    &GridOptions {
                        level: 9,
                        ..GridOptions::default()
                    },
                )
                .unwrap(),
            )
        };
        let scf = run_scf(&mol, &mb, &opts, &func, grid.as_ref()).unwrap();
        let mode = match spin.as_str() {
            "singlet" => TdaSpin::Singlet,
            "triplet" => TdaSpin::Triplet,
            _ => TdaSpin::SpinResolved,
        };
        let solver = TdaSolver::new(&mol, &mb, &func, grid.as_ref(), &scf, &opts, mode).unwrap();
        let states = if rpa {
            solver.solve_rpa(reference.len()).unwrap()
        } else {
            solver.solve(reference.len()).unwrap()
        };
        for (k, (st, &(want_e, want_f))) in states.iter().zip(&reference).enumerate() {
            // Meta-GGA kernels (M06 w-series second derivatives) are far
            // more quadrature-sensitive than the energies; the residual vs
            // PySCF's different radial scheme sits at the few-1e-5 level.
            let tol = if func.parts.is_empty() {
                2e-6
            } else if func.family == dft_rs::xc::Family::MetaGga {
                5e-5
            } else {
                5e-6
            };
            if (st.energy - want_e).abs() > tol {
                failures.push(format!(
                    "{fname} state {k}: {} vs oracle {want_e} (delta = {:.2e})",
                    st.energy,
                    st.energy - want_e
                ));
            }
            // Oscillator strengths: degenerate states mix arbitrarily, so
            // compare only clearly non-degenerate ones.
            let degenerate = states
                .iter()
                .enumerate()
                .any(|(j, o)| j != k && (o.energy - st.energy).abs() < 1e-4);
            if !degenerate && (st.oscillator_strength - want_f).abs() > 1e-4 {
                failures.push(format!(
                    "{fname} state {k}: f = {} vs oracle {want_f}",
                    st.oscillator_strength
                ));
            }
        }
        checked += 1;
    }
    assert!(checked >= 12, "expected 12+ TDA/RPA cases, got {checked}");
    assert!(
        failures.is_empty(),
        "TDA/RPA failures:\n{}",
        failures.join("\n")
    );
}

/// Davidson vs dense diagonalization on the same A matrix (B3LYP water:
/// hybrid + GGA kernel, the most complete coupling).
#[test]
fn davidson_matches_dense_diagonalization() {
    let mol = Molecule::new(vec![
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
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let func = resolve(&FunctionalSpec::Combined("b3lyp".into())).unwrap();
    let opts = ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let grid = MolecularGrid::build(&mol, &GridOptions::default()).unwrap();
    let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
    let solver =
        TdaSolver::new(&mol, &mb, &func, Some(&grid), &scf, &opts, TdaSpin::Singlet).unwrap();
    let dense = solver.solve_dense(4).unwrap();
    let david = solver.solve_davidson(4).unwrap();
    for (k, (d, v)) in dense.iter().zip(&david).enumerate() {
        assert!(
            (d.energy - v.energy).abs() < 1e-7,
            "state {k}: davidson {} vs dense {}",
            v.energy,
            d.energy
        );
    }
}

/// Structural relations every excitation spectrum must satisfy.
///
/// Classifications differ per assertion and are stated inline, because they
/// are not all the same kind of claim.
///
/// - **Positivity** (sanity): an excitation energy of a stable ground state
///   is positive. A negative root is not a small error, it says the
///   reference is not a minimum.
/// - **RPA below TDA** (sanity, conditional): Tamm-Dancoff discards the
///   de-excitation block, and restoring it lowers the roots. This holds
///   root-by-root only while `(A-B)` is positive definite, i.e. while the
///   reference is stable -- which is why it is asserted on a well-behaved
///   closed shell and not in general.
/// - **Length vs velocity gauge** (asymptotic): the two oscillator-strength
///   gauges agree only in the complete-basis limit. In a finite basis they
///   differ, so the test asserts that the difference SHRINKS as the basis
///   improves. Asserting equality here would be wrong, and asserting nothing
///   would leave the velocity-gauge momentum integrals ungated.
#[test]
fn excitation_spectra_satisfy_their_structural_relations() {
    let mol = Molecule::new(vec![
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
    ]);
    let opts = ScfOptions {
        e_tol: 1e-11,
        p_tol: 1e-9,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let func = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();

    let mut gauge_gap = Vec::new();
    for basis in ["sto-3g", "6-31g", "6-31g*"] {
        let set = data::load(basis).unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let scf = run_scf(&mol, &mb, &opts, &func, None).unwrap();
        let solver = TdaSolver::new(&mol, &mb, &func, None, &scf, &opts, TdaSpin::Singlet).unwrap();

        let tda = solver.solve_dense(4).unwrap();
        let rpa = solver.solve_rpa(4).unwrap();

        for (k, st) in tda.iter().enumerate() {
            assert!(
                st.energy > 0.0,
                "{basis}: TDA root {k} has energy {} — a negative excitation energy \
                 means the reference is not a stable ground state",
                st.energy
            );
        }
        for (k, (t, r)) in tda.iter().zip(&rpa).enumerate() {
            assert!(
                r.energy <= t.energy + 1e-9,
                "{basis}: RPA root {k} ({}) is ABOVE the TDA root ({}); restoring the \
                 de-excitation block must lower the roots for a stable reference",
                r.energy,
                t.energy
            );
            assert!(
                r.tamm_dancoff != t.tamm_dancoff,
                "solver mislabelled the manifold"
            );
        }

        // Velocity-gauge oscillator strength: f_vel = (2/3)|p|^2/omega,
        // against the length-gauge f = (2/3) omega |d|^2 the state already
        // carries. Compare on the strongest transition, where the ratio is
        // meaningful; weak transitions have both gauges near zero and their
        // relative difference is dominated by cancellation.
        let strongest = tda
            .iter()
            .max_by(|a, b| a.oscillator_strength.total_cmp(&b.oscillator_strength))
            .unwrap();
        let p2: f64 = strongest.transition_momentum.iter().map(|c| c * c).sum();
        let f_vel = (2.0 / 3.0) * p2 / strongest.energy;
        let f_len = strongest.oscillator_strength;
        assert!(
            f_len > 1e-4,
            "{basis}: the strongest transition has f = {f_len}; the gauge comparison \
             below would be meaningless"
        );
        gauge_gap.push(((f_len - f_vel) / f_len).abs());
    }

    assert!(
        *gauge_gap.last().unwrap() < gauge_gap[0],
        "length- and velocity-gauge oscillator strengths did not converge toward each \
         other as the basis improved: {gauge_gap:?}. They are equal only in the complete \
         basis, so this is a trend and not an identity — but the trend must exist, or the \
         momentum integrals do not correspond to the dipole ones."
    );
}

/// Direct TDDFT must be the same operator as the in-core reference, not an
/// approximation selected only because the ERI tensor no longer fits memory.
/// The three cases separately exercise Coulomb/exchange (HF), the semilocal
/// kernel (PBE), and the additional erf long-range exchange engine (LC-PBE).
#[test]
fn direct_and_incore_tddft_are_operator_and_spectrum_equivalent() {
    let mol = Molecule::new(vec![
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
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();

    for name in ["hf", "pbe", "lc-pbe"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        match name {
            "hf" => assert!(functional.parts.is_empty()),
            "pbe" => {
                assert!(!functional.parts.is_empty());
                assert_eq!(functional.hybrid.cx, 0.0);
                assert_eq!(functional.hybrid.clr, 0.0);
            }
            "lc-pbe" => {
                assert!(!functional.parts.is_empty());
                assert!(functional.hybrid.clr != 0.0);
                assert!(functional.hybrid.omega > 0.0);
            }
            _ => unreachable!(),
        }
        let grid = if functional.parts.is_empty() {
            None
        } else {
            Some(
                MolecularGrid::build(
                    &mol,
                    &GridOptions {
                        level: 3,
                        ..GridOptions::default()
                    },
                )
                .unwrap(),
            )
        };
        let incore_opts = ScfOptions {
            e_tol: 1e-11,
            p_tol: 1e-9,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        let scf = run_scf(&mol, &mb, &incore_opts, &functional, grid.as_ref()).unwrap();
        let mut direct_opts = incore_opts.clone();
        direct_opts.eri_mode = EriMode::Direct {
            screen_threshold: 1e-14,
        };

        let incore = TdaSolver::new(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &incore_opts,
            TdaSpin::Singlet,
        )
        .unwrap();
        let direct = TdaSolver::new(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &direct_opts,
            TdaSpin::Singlet,
        )
        .unwrap();

        let dim = scf.n_alpha * (scf.mo_coeff[0].cols - scf.n_alpha);
        assert!(dim > 1, "{name}: transition space is vacuous");
        let u: Vec<f64> = (0..dim)
            .map(|i| ((i + 1) as f64 * 0.731).sin() / dim as f64)
            .collect();
        let a_incore = incore.a_vec(&u).unwrap();
        let a_direct = direct.a_vec(&u).unwrap();
        assert_eq!(a_incore.len(), dim, "{name}: in-core A.u lost entries");
        assert_eq!(a_direct.len(), dim, "{name}: direct A.u lost entries");
        assert!(a_incore.iter().all(|value| value.is_finite()));
        assert!(a_direct.iter().all(|value| value.is_finite()));
        let worst = a_incore
            .iter()
            .zip(&a_direct)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < 2e-10,
            "{name}: direct/in-core A.u mismatch {worst:e}"
        );

        for (method, in_states, direct_states) in [
            (
                "TDA",
                incore.solve_dense(2).unwrap(),
                direct.solve_dense(2).unwrap(),
            ),
            (
                "RPA",
                incore.solve_rpa(2).unwrap(),
                direct.solve_rpa(2).unwrap(),
            ),
        ] {
            assert_eq!(in_states.len(), 2, "{name} {method}: in-core roots lost");
            assert_eq!(direct_states.len(), 2, "{name} {method}: direct roots lost");
            for root in 0..2 {
                let a = in_states[root].energy;
                let b = direct_states[root].energy;
                assert!(a.is_finite() && b.is_finite());
                assert!(
                    (a - b).abs() < 2e-9,
                    "{name} {method} root {root}: direct {b:.12} vs in-core {a:.12}"
                );
            }
        }
    }
}
