// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Bundled basis sets (Basis Set Exchange; see the data notices) embedded via `include_str!`
//! and parsed on demand. Every data file carries a PROVENANCE header
//! (machine-checked by `tests/license_audit.rs`); citations live in
//! `third_party/bse/NOTICE.md`. Files are refreshed by
//! `tools/fetch_basis.py`; each set includes **all elements** BSE provides
//! for it. ECP-carrying sets (LANL, Stuttgart) parse fully — their
//! potentials are stored on the `BasisSet` for the ECP integral engine.

use super::{g94, BasisSet};
use crate::error::{DftError, Result};

struct Entry {
    /// Canonical user-facing name (lowercase).
    canonical: &'static str,
    aliases: &'static [&'static str],
    text: &'static str,
}

macro_rules! entry {
    ($canon:literal, $file:literal, [$($alias:literal),*]) => {
        Entry {
            canonical: $canon,
            aliases: &[$($alias),*],
            text: include_str!(concat!("data/", $file, ".g94")),
        }
    };
}

static REGISTRY: &[Entry] = &[
    // Minimal / Pople family
    entry!("sto-3g", "sto-3g", ["sto3g"]),
    entry!("sto-6g", "sto-6g", ["sto6g"]),
    entry!("3-21g", "3-21g", ["321g"]),
    entry!("4-31g", "4-31g", ["431g"]),
    entry!("6-31g", "6-31g", ["631g"]),
    entry!("6-31g*", "6-31g_d", ["6-31g(d)", "631g*", "631gd"]),
    entry!("6-31g**", "6-31g_d_p", ["6-31g(d,p)", "631g**"]),
    entry!("6-31+g*", "6-31pg_d", ["6-31+g(d)", "631+g*"]),
    entry!("6-31+g**", "6-31pg_d_p", ["6-31+g(d,p)", "631+g**"]),
    entry!("6-31++g**", "6-31ppg_d_p", ["6-31++g(d,p)", "631++g**"]),
    entry!("6-311g", "6-311g", ["6311g"]),
    entry!("6-311g**", "6-311g_d_p", ["6-311g(d,p)", "6311g**"]),
    entry!("6-311+g**", "6-311pg_d_p", ["6-311+g(d,p)", "6311+g**"]),
    entry!("6-311++g**", "6-311ppg_d_p", ["6-311++g(d,p)", "6311++g**"]),
    entry!("6-311++g(2d,2p)", "6-311ppg_2d_2p", []),
    entry!("6-311++g(3df,3pd)", "6-311ppg_3df_3pd", []),
    // Karlsruhe def2 family
    entry!("def2-sv(p)", "def2-sv_p", ["def2sv(p)"]),
    entry!("def2-svp", "def2-svp", ["def2svp"]),
    entry!("def2-svpd", "def2-svpd", ["def2svpd"]),
    entry!("def2-tzvp", "def2-tzvp", ["def2tzvp"]),
    entry!("def2-tzvpp", "def2-tzvpp", ["def2tzvpp"]),
    entry!("def2-tzvpd", "def2-tzvpd", ["def2tzvpd"]),
    entry!("def2-tzvppd", "def2-tzvppd", ["def2tzvppd"]),
    entry!("def2-qzvp", "def2-qzvp", ["def2qzvp"]),
    entry!("def2-qzvpp", "def2-qzvpp", ["def2qzvpp"]),
    entry!("def2-qzvpd", "def2-qzvpd", ["def2qzvpd"]),
    entry!("def2-qzvppd", "def2-qzvppd", ["def2qzvppd"]),
    // Dunning correlation-consistent family
    entry!("cc-pvdz", "cc-pvdz", ["ccpvdz"]),
    entry!("cc-pvtz", "cc-pvtz", ["ccpvtz"]),
    entry!("cc-pvqz", "cc-pvqz", ["ccpvqz"]),
    entry!("cc-pv5z", "cc-pv5z", ["ccpv5z"]),
    entry!("aug-cc-pvdz", "aug-cc-pvdz", ["augccpvdz"]),
    entry!("aug-cc-pvtz", "aug-cc-pvtz", ["augccpvtz"]),
    entry!("aug-cc-pvqz", "aug-cc-pvqz", ["augccpvqz"]),
    entry!("aug-cc-pv5z", "aug-cc-pv5z", ["augccpv5z"]),
    // ECP families (potentials parsed and carried to the integral engine)
    entry!("lanl2dz", "lanl2dz", []),
    entry!("lanl2dzdp", "lanl2dzdp", ["lanl2dz-dp"]),
    entry!("lanl2tz", "lanl2tz", []),
    entry!("lanl2tz(f)", "lanl2tz_f", ["lanl2tzf"]),
    entry!("lanl2tz+", "lanl2tzp", []),
    entry!("lanl08", "lanl08", []),
    entry!("lanl08(d)", "lanl08_d", ["lanl08d"]),
    entry!("lanl08(f)", "lanl08_f", ["lanl08f"]),
    entry!("lanl08+", "lanl08p", []),
    entry!("stuttgart-rlc", "stuttgart-rlc", ["stuttgartrlc"]),
    entry!(
        "stuttgart-rsc-1997",
        "stuttgart-rsc-1997",
        ["stuttgart-rsc", "stuttgartrsc1997"]
    ),
    // Auxiliary (density-fitting) bases
    entry!(
        "def2-universal-jfit",
        "def2-universal-jfit",
        ["def2-jfit", "weigend"]
    ),
    entry!(
        "def2-universal-jkfit",
        "def2-universal-jkfit",
        ["def2-jkfit"]
    ),
    // Pinned BSE expansion: orbital, shielding, core-valence and J bases.
    entry!("pcseg-0", "pcseg-0", []),
    entry!("pcseg-1", "pcseg-1", []),
    entry!("pcseg-2", "pcseg-2", []),
    entry!("pcseg-3", "pcseg-3", []),
    entry!("pcseg-4", "pcseg-4", []),
    entry!("aug-pcseg-0", "aug-pcseg-0", []),
    entry!("aug-pcseg-1", "aug-pcseg-1", []),
    entry!("aug-pcseg-2", "aug-pcseg-2", []),
    entry!("aug-pcseg-3", "aug-pcseg-3", []),
    entry!("aug-pcseg-4", "aug-pcseg-4", []),
    entry!("pcsseg-0", "pcsseg-0", []),
    entry!("pcsseg-1", "pcsseg-1", []),
    entry!("pcsseg-2", "pcsseg-2", []),
    entry!("pcsseg-3", "pcsseg-3", []),
    entry!("pcsseg-4", "pcsseg-4", []),
    entry!("cc-pcvdz", "cc-pcvdz", []),
    entry!("cc-pcvtz", "cc-pcvtz", []),
    entry!("cc-pcvqz", "cc-pcvqz", []),
    entry!("cc-pwcvdz", "cc-pwcvdz", []),
    entry!("cc-pwcvtz", "cc-pwcvtz", []),
    entry!("cc-pwcvqz", "cc-pwcvqz", []),
    entry!("aug-cc-pcvdz", "aug-cc-pcvdz", []),
    entry!("aug-cc-pcvtz", "aug-cc-pcvtz", []),
    entry!("aug-cc-pcvqz", "aug-cc-pcvqz", []),
    entry!("cc-pvdz-dk", "cc-pvdz-dk", []),
    entry!("cc-pvtz-dk", "cc-pvtz-dk", []),
    entry!("cc-pvqz-dk", "cc-pvqz-dk", []),
    entry!("pcj-0", "pcj-0", []),
    entry!("pcj-1", "pcj-1", []),
    entry!("pcj-2", "pcj-2", []),
];

/// Canonical names of all bundled sets.
pub fn bundled_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|e| e.canonical).collect()
}

/// Normalize a user-supplied basis name: lowercase, trim, `_`/space → `-`.
fn normalize(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['_', ' '], "-")
}

/// Load a bundled basis set by name (canonical or alias, case-insensitive).
pub fn load(name: &str) -> Result<BasisSet> {
    let norm = normalize(name);
    for e in REGISTRY {
        if e.canonical == norm || e.aliases.iter().any(|a| normalize(a) == norm) {
            return g94::parse(e.canonical, e.text);
        }
    }
    Err(DftError::UnknownBasis(format!(
        "{name} (bundled: {})",
        bundled_names().join(", ")
    )))
}

/// Load a basis set: a bundled name, or a path to a Gaussian94 file if the
/// argument contains a path separator or ends in `.g94`.
pub fn load_or_read(name_or_path: &str) -> Result<BasisSet> {
    let looks_like_path = name_or_path.contains('/')
        || name_or_path.contains('\\')
        || name_or_path.to_ascii_lowercase().ends_with(".g94");
    if looks_like_path {
        let text = std::fs::read_to_string(name_or_path)?;
        let stem = std::path::Path::new(name_or_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name_or_path);
        g94::parse(stem, &text)
    } else {
        load(name_or_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{n_sph, MolBasis};
    use crate::math::Vec3;
    use crate::system::{Atom, Molecule};

    /// Every bundled set must parse in full (all elements, ECP sections
    /// included) and define at least one element.
    #[test]
    fn all_bundled_sets_parse() {
        for name in bundled_names() {
            let set = load(name).unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
            let n = (1..=118u8).filter(|&z| set.has_element(z)).count();
            assert!(n > 0, "{name} defines no elements");
        }
    }

    #[test]
    fn name_normalization_variants() {
        assert!(load("STO-3G").is_ok());
        assert!(load("sto_3g").is_ok());
        assert!(load("Def2-SVP").is_ok());
        assert!(load("def2svp").is_ok());
        assert!(load("6-31G(d)").is_ok());
        assert!(load("6-31g*").is_ok());
        assert!(load("6-311++G(3df,3pd)").is_ok());
        assert!(load("aug-cc-pVTZ").is_ok());
        assert!(load("LANL2DZ").is_ok());
        assert!(load("nonexistent-basis").is_err());
    }

    #[test]
    fn full_element_ranges_are_present() {
        // STO-3G: H–Xe on BSE. def2-SVP: H–Rn (ECPs beyond Kr).
        let sto = load("sto-3g").unwrap();
        for z in 1..=54u8 {
            assert!(sto.has_element(z), "sto-3g missing Z={z}");
        }
        let svp = load("def2-svp").unwrap();
        for z in 1..=86u8 {
            assert!(svp.has_element(z), "def2-svp missing Z={z}");
        }
        // All-electron through Kr; ECP-based beyond.
        assert_eq!(svp.ecp_core(36), None);
        assert_eq!(svp.ecp_core(37), Some(28));
        assert_eq!(svp.ecp_core(79), Some(60)); // Au
    }

    #[test]
    fn sto3g_oxygen_layout() {
        let set = load("sto-3g").unwrap();
        let o = set.element(8).unwrap();
        let nao: usize = o.shells.iter().map(|s| n_sph(s.l)).sum();
        assert_eq!(nao, 5);
        assert!((o.shells[0].exps[0] - 130.7093214).abs() < 1e-6);
    }

    #[test]
    fn def2_svp_carbon_has_d_polarization() {
        let set = load("def2-svp").unwrap();
        let c = set.element(6).unwrap();
        assert!(c.shells.iter().any(|s| s.l == 2));
        let nao: usize = c.shells.iter().map(|s| n_sph(s.l)).sum();
        assert_eq!(nao, 14);
    }

    #[test]
    fn lanl2dz_records_ecps_and_build_carries_them() {
        let set = load("lanl2dz").unwrap();
        // Light elements are all-electron (D95V part).
        assert!(set.has_element(1) && set.ecp_core(1).is_none());
        assert!(set.has_element(8) && set.ecp_core(8).is_none());
        // Fe carries an ECP with a stored potential.
        let ecp = set.ecp(26).expect("LANL2DZ Fe must carry an ECP");
        assert!(ecp.n_core > 0);
        assert!(!ecp.local.is_empty());
        assert_eq!(ecp.semilocal.len(), ecp.lmax);
        // Building on an ECP element records the potential + core count and
        // reduces the effective nuclear charge.
        let mol = Molecule::new(vec![Atom {
            z: 26,
            position: Vec3::zero(),
        }]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        assert_eq!(mb.ecps.len(), 1);
        assert_eq!(mb.n_core_electrons, ecp.n_core);
        let z_eff = mb.effective_charges(&mol);
        assert_eq!(z_eff[0], 26.0 - ecp.n_core as f64);
        // An all-electron molecule in the same set carries no ECPs.
        let h2 = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.4, 0.0, 0.0),
            },
        ]);
        let mbh = MolBasis::build(&h2, &set).unwrap();
        assert!(mbh.ecps.is_empty() && mbh.n_core_electrons == 0);
    }
}
