// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Registry coverage census + reference-free gates for the entries that
//! have no frozen PySCF/libxc energy yet.
//!
//! The project rule is that every functional the registry can resolve is
//! pinned to a frozen oracle in `tests/oracle_data/` or a dedicated
//! primary-equation test suite. Two tests here make
//! that rule machine-checkable rather than a habit:
//!
//! 1. `registry_tables_have_no_unlisted_arms` reads `src/xc/mod.rs` and
//!    proves the enumerable name lists cover every match arm — a new
//!    functional cannot be added to a table and stay invisible.
//! 2. `every_registry_entry_is_oracle_gated_or_declared_reference_free`
//!    cross-references those names against the `functional,<name>` keys of
//!    the frozen CSVs and the dedicated suites. An entry with no oracle must appear in the
//!    `REFERENCE_FREE` table below, with its reason — and an entry that
//!    IS gated must NOT appear there, so the list cannot go stale.
//!
//! Both tables are currently empty — every registry entry has a frozen
//! reference energy — but the machinery stays, because an entry added
//! tomorrow has to justify itself.
//!
//! The remaining tests are exact limits and exact symmetries that need no
//! external number: spin scaling, screening-constant closed forms, the
//! `omega -> 0` limits, and the series-coefficient relations. They stand
//! independently of the oracles rather than in place of them, and they are
//! deliberately NOT comparisons against dft-rs's own stored output — a
//! self-comparison would pass however wrong the functional is.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{
    resolve, FunctionalSpec, GridVars, ResolvedFunctional, XcOut, COMBINED_NAMES,
    CORRELATION_NAMES, EXCHANGE_NAMES,
};
use std::sync::Arc;

/// Registry entries with no frozen PySCF/libxc energy, and why.
///
/// **Currently empty: every combined functional the registry resolves is
/// pinned to a frozen reference energy.** The mechanism stays because the
/// census test enforces it — adding a functional without an oracle fails
/// unless it is listed here with a reason AND
/// `tools/oracle/gen_dft_missing.py` grows a case that would close the gap.
/// An excuse cannot be written without also writing the plan to retire it.
const REFERENCE_FREE: &[(&str, &str)] = &[];

/// Dedicated primary-equation oracles. These independently reconstruct
/// the physical energy terms and retain frozen local term references.
const PRIMARY_EQUATION_ORACLES: &[(&str, &str)] = &[("wb97x-d3", include_str!("p326_wb97x_d3.rs"))];

/// Which frozen oracle exercises each split (`--x` / `--c`) component. The
/// split tables carry no oracle of their own: a component is covered when
/// a gated combined functional contains it AT ITS REGISTRY PARAMETERS.
const COMPONENT_ORACLE: &[(&str, &str)] = &[
    ("slater", "svwn5"),
    ("b88", "blyp"),
    ("pbe", "pbe"),
    ("revpbe", "revpbe"),
    ("tpss", "tpss"),
    ("r2scan", "r2scan"),
    ("vwn5", "svwn5"),
    ("vwn3", "svwn3"),
    ("pw92", "spw92"),
    ("lyp", "blyp"),
    ("p86", "bp86"),
    ("pw91", "b3pw91"),
    ("pz81", "spz81"),
    ("mpw91", "mpwlyp"),
    ("g96", "g96lyp"),
    ("op", "bop"),
    ("b95", "bb95"),
];

/// Split components that no gated combined functional reaches at their
/// registry parameters. Empty: both former entries now carry their own
/// `x,<name>` oracle, frozen at the registry's parameters.
const COMPONENT_REFERENCE_FREE: &[(&str, &str)] = &[];

fn normalize(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .replace(['_', ' ', '-'], "")
}

/// The name literals of one registry table, as written in `src/xc/mod.rs`.
/// Each returned entry is one match arm with all of its aliases.
fn table_arms(src: &str, start: &str, end: &str) -> Vec<Vec<String>> {
    let s = src
        .find(start)
        .unwrap_or_else(|| panic!("registry table `{start}` not found in src/xc/mod.rs"));
    let rest = &src[s..];
    let e = rest
        .find(end)
        .unwrap_or_else(|| panic!("end marker `{end}` not found after `{start}`"));
    let mut arms = Vec::new();
    for line in rest[..e].lines() {
        let Some((head, _)) = line.split_once("=>") else {
            continue;
        };
        let mut names = Vec::new();
        let mut chars = head.chars();
        while let Some(c) = chars.next() {
            if c != '"' {
                continue;
            }
            let lit: String = chars.by_ref().take_while(|&c| c != '"').collect();
            names.push(normalize(&lit));
        }
        if !names.is_empty() {
            arms.push(names);
        }
    }
    arms
}

/// The exported name lists are the only enumerable view of the registry
/// (the tables are `match` expressions). This proves the view is complete:
/// every arm of every table is named by exactly one canonical entry, and
/// no canonical entry names a table arm that does not exist.
///
/// The census reads source text, so it takes every `=>` arm inside a table's
/// span to be one entry of that table. A second, unrelated `match` on the
/// same names placed inside the span therefore reads as one entry naming
/// several functionals at once, and fails here. That is the intended
/// outcome: helper classifications belong outside the table, where they
/// cannot be confused with it (see `required_nonlocal_for` in
/// `src/xc/mod.rs`).
#[test]
fn registry_tables_have_no_unlisted_arms() {
    let src = include_str!("../src/xc/mod.rs");
    let tables: [(&str, &str, &[&str], &str); 3] = [
        (
            "fn exchange(name: &str)",
            "fn correlation(name: &str)",
            EXCHANGE_NAMES,
            "EXCHANGE_NAMES",
        ),
        (
            "fn correlation(name: &str)",
            "fn combined(name: &str)",
            CORRELATION_NAMES,
            "CORRELATION_NAMES",
        ),
        (
            "fn combined(name: &str)",
            "pub fn resolve(",
            COMBINED_NAMES,
            "COMBINED_NAMES",
        ),
    ];
    for (start, end, listed, list_name) in tables {
        let arms = table_arms(src, start, end);
        assert!(!arms.is_empty(), "no arms parsed out of `{start}`");
        let canonical: Vec<String> = listed.iter().map(|n| normalize(n)).collect();
        for arm in &arms {
            let hits = arm.iter().filter(|n| canonical.contains(n)).count();
            assert_eq!(
                hits, 1,
                "registry arm {arm:?} is named by {hits} entries of {list_name} \
                 (must be exactly one) — a new functional was added to the table \
                 without registering a canonical name, so nothing gates it",
            );
        }
        for name in &canonical {
            let hits = arms.iter().filter(|arm| arm.contains(name)).count();
            assert_eq!(
                hits, 1,
                "{list_name} entry `{name}` matches {hits} registry arms \
                 (must be exactly one)"
            );
        }
        assert_eq!(
            arms.len(),
            canonical.len(),
            "{list_name} has {} entries for {} registry arms",
            canonical.len(),
            arms.len()
        );
    }
}

/// Every functional the registry resolves is either pinned to a frozen
/// PySCF/libxc energy or explicitly declared reference-free with a reason.
#[test]
fn every_registry_entry_is_oracle_gated_or_declared_reference_free() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut gated: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("csv") {
            continue;
        }
        // `functional,<name>` names a combined spec and appears in energy,
        // gradient, Hessian, excitation and ECP oracles alike. The split
        // grammar `x,<name>` / `c,<name>` is read only out of the energy
        // CSVs, whose keys the p3 loader owns — a one-letter key is too
        // easy to collide with in an arbitrary data table.
        let split_keys = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_dft.csv"));
        let text = std::fs::read_to_string(&path).unwrap();
        for line in text.lines() {
            let keys: &[&str] = if split_keys {
                &["functional,", "x,", "c,"]
            } else {
                &["functional,"]
            };
            for key in keys {
                if let Some(value) = line.strip_prefix(key) {
                    let name = normalize(value.split(',').next().unwrap_or(value));
                    if name != "none" {
                        gated.push(name);
                    }
                }
            }
        }
    }
    gated.sort();
    gated.dedup();
    assert!(
        gated.len() > 20,
        "only {} functionals found in the frozen CSVs — the oracle \
         directory looks broken, not the registry",
        gated.len()
    );

    let excused: Vec<String> = REFERENCE_FREE.iter().map(|(n, _)| normalize(n)).collect();
    let independent: Vec<String> = PRIMARY_EQUATION_ORACLES
        .iter()
        .map(|(n, _)| normalize(n))
        .collect();
    for (name, suite) in PRIMARY_EQUATION_ORACLES {
        assert!(COMBINED_NAMES.contains(name), "unknown oracle case {name}");
        assert!(
            suite.contains(&format!("\"{name}\"")),
            "oracle suite omits {name}"
        );
    }
    let mut missing = Vec::new();
    for &name in COMBINED_NAMES {
        let n = normalize(name);
        if gated.contains(&n) || independent.contains(&n) || excused.contains(&n) {
            continue;
        }
        missing.push(name);
    }
    assert!(
        missing.is_empty(),
        "no frozen or primary-equation oracle and no declared reason for: {missing:?}\n\
         Either freeze a reference with tools/oracle/gen_dft.py, or add the \
         name to REFERENCE_FREE in this file together with the exact gate \
         that stands in for the missing number.",
    );

    // The excuse list must not outlive the excuse.
    for (name, _) in REFERENCE_FREE {
        let n = normalize(name);
        assert!(
            COMBINED_NAMES.iter().any(|c| normalize(c) == n),
            "REFERENCE_FREE names `{name}`, which is not a registry entry"
        );
        assert!(
            !gated.contains(&n),
            "`{name}` now HAS a frozen oracle — drop it from REFERENCE_FREE \
             so the census keeps counting it as gated"
        );
    }

    // Split components: covered through the combined functional that
    // contains them, or declared reference-free in turn.
    for &name in EXCHANGE_NAMES.iter().chain(CORRELATION_NAMES) {
        let n = normalize(name);
        if gated.contains(&n) {
            continue; // pinned directly by an `x,`/`c,` oracle
        }
        if let Some((_, via)) = COMPONENT_ORACLE.iter().find(|(c, _)| normalize(c) == n) {
            assert!(
                gated.contains(&normalize(via)),
                "component `{name}` is declared covered through `{via}`, \
                 which has no frozen oracle"
            );
            continue;
        }
        assert!(
            COMPONENT_REFERENCE_FREE
                .iter()
                .any(|(c, _)| normalize(c) == n),
            "component `{name}` is neither mapped to a gated combined \
             functional in COMPONENT_ORACLE nor declared in \
             COMPONENT_REFERENCE_FREE"
        );
    }
    for (name, _) in COMPONENT_REFERENCE_FREE {
        let n = normalize(name);
        assert!(
            EXCHANGE_NAMES
                .iter()
                .chain(CORRELATION_NAMES)
                .any(|c| normalize(c) == n),
            "COMPONENT_REFERENCE_FREE names `{name}`, not a registry component"
        );
        assert!(
            !gated.contains(&n),
            "component `{name}` now HAS a frozen oracle — drop it from \
             COMPONENT_REFERENCE_FREE"
        );
    }

    // A declared gap must come with the means to close it: the generator
    // that freezes these on a machine with a capable PySCF build has to
    // name every one of them, or the gap outlives the plan to fix it.
    let generator = include_str!("../tools/oracle/gen_dft_missing.py");
    for (name, _) in REFERENCE_FREE.iter().chain(COMPONENT_REFERENCE_FREE) {
        assert!(
            generator.contains(&format!("\"{name}\"")),
            "tools/oracle/gen_dft_missing.py has no case for `{name}`, so \
             nothing would ever freeze its reference energy"
        );
    }
}

/// Energy density at one (ρα, ρβ, σαα, σαβ, σββ, τα, τβ) point.
fn eval_e(func: &ResolvedFunctional, v: [f64; 7]) -> f64 {
    let (ra, rb) = ([v[0]], [v[1]]);
    let (saa, sab, sbb) = ([v[2]], [v[3]], [v[4]]);
    let (ta, tb) = ([v[5]], [v[6]]);
    let vars = GridVars {
        rho: [&ra, &rb],
        sigma: Some([&saa, &sab, &sbb]),
        tau: Some([&ta, &tb]),
    };
    let mut out = XcOut::default();
    func.eval(&vars, 0, &mut out).unwrap();
    out.e[0]
}

/// Exact spin scaling of exchange: Ex[ρα, ρβ] = ½(Ex[2ρα] + Ex[2ρβ]), where
/// the doubled-density arguments are the fully polarized evaluations. It
/// holds for every exchange functional at every order of the ladder and
/// needs no reference value, so it gates the entries (OPTX at its own a1,
/// HJS ωPBE at ω = 0.11) that no frozen energy reaches. It fails the
/// moment a channel is mixed up — σαβ leaking into exchange, ρ used where
/// ρσ belongs, or the two spin channels swapped.
#[test]
fn exchange_functionals_obey_spin_scaling() {
    // Chosen inside the Fermi-hole-curvature bound σσσ ≤ 8ρστσ so the
    // meta-GGA input clamp is inactive and the identity is exact.
    let p = [0.31, 0.17, 0.042, 0.026, 0.021, 0.29, 0.13];
    for &name in EXCHANGE_NAMES {
        let f = resolve(&FunctionalSpec::Split {
            x: name.into(),
            c: "none".into(),
        })
        .unwrap();
        let mixed = eval_e(&f, p);
        let alpha = eval_e(&f, [p[0], p[0], p[2], p[2], p[2], p[5], p[5]]);
        let beta = eval_e(&f, [p[1], p[1], p[4], p[4], p[4], p[6], p[6]]);
        let want = 0.5 * (alpha + beta);
        assert!(
            (mixed - want).abs() < 1e-12 * (1.0 + want.abs()),
            "{name}: Ex[ρα,ρβ] = {mixed} vs ½(Ex[2ρα] + Ex[2ρβ]) = {want}"
        );
        assert!(
            mixed < 0.0,
            "{name}: exchange energy density must be negative"
        );
    }
}

/// Exchange obeys the uniform coordinate-scaling relation exactly.
///
/// Classification: **exact**. This is the defining scaling property of the
/// exchange energy, not a limit or an approximation, so it holds pointwise
/// at every density and for every rung of the ladder.
///
/// Under `rho_lambda(r) = lambda^3 rho(lambda r)` the exchange energy scales
/// as `Ex[rho_lambda] = lambda * Ex[rho]`. Changing variables in the
/// integral turns that into a pointwise statement about the energy DENSITY,
/// which is what this evaluates. The grid variables carry their own powers:
///
/// - `rho -> lambda^3 rho`
/// - `sigma = |grad rho|^2 -> lambda^8 sigma`   (grad rho picks up lambda^4)
/// - `tau -> lambda^5 tau`                      (orbitals scale as lambda^{3/2})
///
/// and the requirement is `e_x(scaled) == lambda^4 * e_x(unscaled)`.
///
/// This is a sharper instrument than it looks. It fixes the RELATIVE powers
/// of rho, sigma and tau simultaneously, so it fails if a reduced gradient is
/// built with the wrong exponent -- `sigma/rho^{8/3}` mistyped as
/// `sigma/rho^{7/3}`, say -- even though such a slip leaves the functional
/// finite, smooth, negative and close enough to a reference value at ONE
/// density to pass an oracle pin.
///
/// Screened exchange is included rather than skipped, and that is the
/// interesting part: omega is an inverse length, so a screened hole is NOT
/// scale-invariant at fixed omega. The identity is restored only when omega
/// scales with lambda too. Testing it that way verifies that omega enters
/// with the dimensions it claims to have -- something no fixed-omega
/// reference point can see.
///
/// Every parameter carries its declared scaling power
/// (`XcFunctional::param_scaling_powers`): omega of a screened hole is an
/// inverse length (power 1), the length omega_v of the nonseparable N12/GAM
/// exchange has power -1, the KT delta added to rho^(4/3) has power 4 and
/// the FT97 beta2 compared with sigma has power 8. With every dimensionful
/// parameter scaled, the identity holds for every exchange in the registry.
#[test]
fn exchange_obeys_uniform_coordinate_scaling() {
    // Chosen inside the Fermi-hole-curvature bound so no input clamp fires.
    let base = [0.31, 0.17, 0.042, 0.026, 0.021, 0.29, 0.13];
    for &name in EXCHANGE_NAMES {
        let f = resolve(&FunctionalSpec::Split {
            x: name.into(),
            c: "none".into(),
        })
        .unwrap();
        let e0 = eval_e(&f, base);
        for lambda in [0.5_f64, 1.3, 2.7] {
            let scaled = [
                lambda.powi(3) * base[0],
                lambda.powi(3) * base[1],
                lambda.powi(8) * base[2],
                lambda.powi(8) * base[3],
                lambda.powi(8) * base[4],
                lambda.powi(5) * base[5],
                lambda.powi(5) * base[6],
            ];
            // Dimensionful parameters scale with the coordinates by their
            // declared powers; the hybrid metadata of a bare `--x` spec is
            // `PURE`, so the parts are scaled directly.
            let scaled_parts: Vec<(f64, Arc<dyn dft_rs::xc::XcFunctional>)> = f
                .parts
                .iter()
                .map(|(c, p)| {
                    let powers = p.param_scaling_powers();
                    if powers.iter().all(|k| *k == 0) {
                        return (*c, Arc::clone(p));
                    }
                    let params: Vec<f64> = p
                        .params()
                        .iter()
                        .zip(&powers)
                        .map(|(v, k)| v * lambda.powi(*k))
                        .collect();
                    (*c, Arc::from(p.with_params(&params).unwrap()))
                })
                .collect();
            let fs =
                ResolvedFunctional::custom(f.description.clone(), scaled_parts, f.hybrid, f.family);
            let got = eval_e(&fs, scaled);
            let want = lambda.powi(4) * e0;
            assert!(
                (got - want).abs() < 1e-10 * (1.0 + want.abs()),
                "{name}: e_x scaled by lambda = {lambda} gave {got}, expected {want} \
                 (ratio {}). The exchange energy density must satisfy \
                 e_x(l^3 rho, l^8 sigma, l^5 tau) = l^4 e_x(rho, sigma, tau); a mismatch \
                 means a reduced-gradient or kinetic-density exponent is wrong.",
                got / want
            );
        }
    }
}
/// HSE03's two screening parameters are the only thing that separates it
/// from the oracle-gated HSE06, so they are checked against the closed
/// forms they come from: ω_HF = 0.15/√2 for the erf kernel and
/// ω_PBE = 0.15·2^{1/3} for the semilocal hole. A mistyped digit in either
/// literal — the one failure mode a shared-machinery functional really
/// has — shows up here.
#[test]
fn hse03_screening_constants_match_their_closed_forms() {
    let f = resolve(&FunctionalSpec::Combined("hse03".into())).unwrap();
    assert_eq!(f.hybrid.cx, 0.25, "HSE03 SR-HF fraction");
    assert_eq!(
        f.hybrid.clr, -0.25,
        "the K − K_erf screening trick needs clr = −cx exactly"
    );
    let omega_hf = 0.15 / 2f64.sqrt();
    assert!(
        (f.hybrid.omega - omega_hf).abs() < 1e-15,
        "ω_HF = {} vs 0.15/√2 = {omega_hf}",
        f.hybrid.omega
    );
    // The semilocal side: a full-range hole (ω = 0) minus a quarter of the
    // screened hole at ω_PBE = 0.15·2^{1/3}.
    let omega_pbe = 0.15 * 2f64.cbrt();
    let screened: Vec<(f64, f64)> = f
        .parts
        .iter()
        .filter_map(|(coef, p)| {
            p.param_names()
                .iter()
                .position(|&n| n == "omega")
                .map(|i| (*coef, p.params()[i]))
        })
        .collect();
    assert_eq!(
        screened.len(),
        2,
        "expected two wPBEh parts, got {screened:?}"
    );
    assert_eq!(screened[0], (1.0, 0.0), "full-range hole parent");
    assert_eq!(screened[1].0, -0.25, "screened hole weight");
    assert!(
        (screened[1].1 - omega_pbe).abs() < 1e-15,
        "ω_PBE = {} vs 0.15·2^{{1/3}} = {omega_pbe}",
        screened[1].1
    );
    // ω_PBE > ω_HF is the whole point of the 2003 parameterization; HSE06
    // collapsed them onto a single ω = 0.11.
    assert!(screened[1].1 > f.hybrid.omega);
    let hse06 = resolve(&FunctionalSpec::Combined("hse06".into())).unwrap();
    assert_eq!(hse06.hybrid.omega, 0.11);
}

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

/// Converged sto-3g energy of a functional on the shared water geometry.
fn energy(func: &ResolvedFunctional, level: usize) -> f64 {
    let mol = water();
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level,
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
    run_scf(&mol, &mb, &opts, func, Some(&grid)).unwrap().energy
}

/// HSE03 and HSE06 are the same screened hybrid with different screening
/// constants: `wpbeh(0) − 0.25·wpbeh(ω_PBE) + PBE-C`, with `0.25(K − K_erf)`
/// on the Fock side. Retuning HSE03 to ω = 0.11 therefore reproduces HSE06
/// *by construction* — which means HSE06's frozen PySCF energy can be
/// carried onto HSE03's own definition, and it stops being a self-comparison
/// because the number on the right comes from the oracle CSV, not from
/// dft-rs.
///
/// Together with the closed-form check on the two ω literals, this leaves
/// nothing about HSE03 unpinned except the pair of constants themselves:
/// every part, weight, sign and Fock-side coefficient is exercised against
/// an external reference here. It fails on a flipped clr, a weight that is
/// not −cx, a swapped full-range/screened hole, or ω threaded into the
/// wrong part.
#[test]
fn hse03_retuned_to_hse06_screening_matches_the_frozen_hse06_oracle() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/oracle_data/ch3_doublet_sto3g_hse06_dft.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut atoms = Vec::new();
    let (mut charge, mut mult, mut e_ref) = (0i32, 1usize, f64::NAN);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(',').collect();
        match p[0] {
            "basis" => assert_eq!(p[1], "sto-3g"),
            "functional" => assert_eq!(p[1], "hse06"),
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
            "e_tot" => e_ref = p[1].parse().unwrap(),
            other => panic!("unknown key {other}"),
        }
    }
    let mol = Molecule::new(atoms)
        .with_charge(charge)
        .with_multiplicity(mult);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let retuned = resolve(&FunctionalSpec::Combined("hse03".into()))
        .unwrap()
        .with_omega(0.11)
        .unwrap();
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
        level_shift: 0.2,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let e = run_scf(&mol, &mb, &opts, &retuned, Some(&grid))
        .unwrap()
        .energy;

    // Two DIFFERENT claims live here, and they are now separated because
    // they have different strengths and different failure meanings.
    //
    // (1) The INTERNAL identity, which is the one this test exists for:
    // HSE03 retuned to omega = 0.11 IS HSE06. libxc carries HSE03's two
    // distinct screening constants (omega_HF = 0.15/sqrt2,
    // omega_PBE = 0.15*2^(1/3)) internally, so collapsing them onto a single
    // omega and landing exactly on HSE06 is what proves this port kept them
    // straight. Being reference-free, it holds to SCF convergence, not to a
    // quadrature tolerance.
    let hse06 = resolve(&FunctionalSpec::Combined("hse06".into())).unwrap();
    let e_hse06 = run_scf(&mol, &mb, &opts, &hse06, Some(&grid))
        .unwrap()
        .energy;
    assert!(
        (e - e_hse06).abs() < 1e-8,
        "HSE03[ω = 0.11] {e} vs HSE06 {e_hse06} (Δ = {:.2e}) — these must be the same \
         functional; a mismatch means the two screening constants were not collapsed \
         correctly",
        e - e_hse06
    );

    // (2) The EXTERNAL comparison, deliberately loosened. This project's
    // wPBEh ships a smooth s-rescaling where libxc's short-circuits are
    // discontinuous (-5.141e-4 at s = 1, +1.616e-3 at s = 15), which shifts
    // every wPBEh-based total by ~1.5e-4 Ha. See docs/FUNCTIONALS.md and the
    // matching bound in tests/p3_dft_oracle.rs. A drift much beyond that is
    // a real regression, so it is still bounded rather than dropped.
    assert!(
        (e_hse06 - e_ref).abs() < 5e-4,
        "HSE06 {e_hse06} vs the frozen libxc oracle {e_ref} (Δ = {:.2e}); the known \
         s-rescaling difference is ~1.5e-4, so this is larger than the deliberate \
         deviation and indicates a regression",
        e_hse06 - e_ref
    );
}

/// The ωB97 family splits exact exchange as `cx·K + clr·K_erf(ω)` on the
/// Fock side while the functional carries the complementary short-range
/// DFT exchange. Two exact relations tie the two sides together:
///
/// - `cx + clr = 1` — at long range the exchange is 100 % exact;
/// - `cx_0 = 1 − cx` — the leading exchange series coefficient is the
///   short-range DFT share, so that SR-HF + SR-DFT sums to the whole
///   short-range exchange (the series starts at the uniform-gas limit);
/// - the ω inside the functional is the ω of the erf kernel.
///
/// ωB97 and ωB97X are oracle-gated, so this checks the ungated ωB97X-D and
/// ωB97X-V against a relation their gated siblings satisfy — a transcription
/// slip in either the hybrid metadata or the leading coefficient breaks it,
/// and that is precisely the failure mode a shared code path still has.
#[test]
fn wb97_family_hybrid_metadata_matches_the_leading_series_coefficient() {
    for name in ["wb97", "wb97x", "wb97x-d", "wb97x-v", "wb97m-v"] {
        let f = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let (cx, clr) = (f.hybrid.cx, f.hybrid.clr);
        assert!(
            (cx + clr - 1.0).abs() < 1e-15,
            "{name}: cx + clr = {} (full LR-HF requires 1)",
            cx + clr
        );
        assert_eq!(f.parts.len(), 1, "{name} is XC-integrated");
        let (weight, part) = &f.parts[0];
        assert_eq!(*weight, 1.0);
        let names = part.param_names();
        let params = part.params();
        let iw = names.iter().position(|&n| n == "omega").unwrap();
        assert!(
            (params[iw] - f.hybrid.omega).abs() < 1e-15,
            "{name}: functional ω = {} vs kernel ω = {}",
            params[iw],
            f.hybrid.omega
        );
        // The leading exchange coefficient: `cx_0` in the 1D series
        // layouts, `cx_00` in ωB97M-V's sparse 2D one.
        let ix = names.iter().position(|n| n.starts_with("cx_")).unwrap();
        assert!(
            (params[ix] - (1.0 - cx)).abs() < 1e-12,
            "{name}: leading exchange coefficient {} vs 1 − cx = {}",
            params[ix],
            1.0 - cx
        );
    }
}

/// LC-PBE's ω → 0 limit is plain PBE: the ITYH attenuation factor F(a) → 1
/// as ω → 0, and the long-range HF kernel K_erf(ω) → 0 with it. This is
/// the only exact statement available for the entry — no libxc functional
/// uses the ITYH attenuation of PBE, so there is nothing to compare the
/// finite-ω energy against — and it still catches a wrong clr, a
/// mis-threaded ω, or an SR factor that does not normalize to 1.
#[test]
fn lc_pbe_reduces_to_pbe_when_screening_vanishes() {
    let lc_pbe = resolve(&FunctionalSpec::Combined("lc-pbe".into()))
        .unwrap()
        .with_omega(1e-7)
        .unwrap();
    let pbe = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let (e_lc, e_pbe) = (energy(&lc_pbe, 4), energy(&pbe, 4));
    assert!(
        (e_lc - e_pbe).abs() < 1e-6,
        "LC-PBE(ω→0) {e_lc} vs PBE {e_pbe} (Δ = {:.2e})",
        e_lc - e_pbe
    );
}

/// Every arm of `DoubleHybridFunctional::named` is listed exactly once in
/// `DOUBLE_HYBRID_NAMES`, and every listed name resolves to that arm, so a
/// recipe added to the match cannot stay out of the catalogue.
#[test]
fn double_hybrid_names_match_the_named_arms() {
    use dft_rs::xc::double_hybrid::{DoubleHybridFunctional, DOUBLE_HYBRID_NAMES};
    let key = |name: &str| -> String {
        name.replace(['ω', 'Ω'], "w")
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    };
    let src = include_str!("../src/xc/double_hybrid.rs");
    let arms = table_arms(src, "match normalized.as_str() {", "_ => None,");
    assert!(!arms.is_empty(), "no double-hybrid arms parsed");
    let listed: Vec<String> = DOUBLE_HYBRID_NAMES.iter().map(|n| key(n)).collect();
    for arm in &arms {
        let hits = arm.iter().filter(|n| listed.contains(n)).count();
        assert_eq!(hits, 1, "double-hybrid arm {arm:?} is listed {hits} times");
    }
    assert_eq!(arms.len(), listed.len());
    for &name in DOUBLE_HYBRID_NAMES {
        assert!(
            DoubleHybridFunctional::named(name).is_some(),
            "`{name}` does not resolve"
        );
        assert!(
            resolve(&FunctionalSpec::Combined(name.into())).is_err(),
            "`{name}` is also a combined entry, which would hide its PT2 term"
        );
    }
}
