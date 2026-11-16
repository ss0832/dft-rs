// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Per-element basis and ECP assignment.
//!
//! A single basis name covers the common case, but real calculations often
//! want a mixed set: a large basis on the reacting centre and a small one
//! everywhere else, or an all-electron basis on the light atoms and an
//! ECP set on the metal. The spec grammar below expresses that in one
//! string, so it works identically from the CLI, the library, and Python.
//!
//! ```text
//! spec     := segment (';' segment)*        (newlines separate too)
//! segment  := source                        -- the base, at most one, first
//!           | elements '=' source
//! elements := SYM (',' SYM)*                -- symbol or atomic number
//! source   := bundled-name | path/to.g94 | 'none'
//! ```
//!
//! Examples:
//!
//! ```text
//! def2-svp                          the plain single-basis case
//! def2-tzvp; H=def2-svp             light atoms cheaper
//! def2-tzvp; Ni,Fe=lanl2dz          ECP set on the metals only
//! H=sto-3g; O=./custom-o.g94        no base; every element assigned
//! @basis.spec                       read the spec from a file
//! ```
//!
//! **Basis and ECP are independent.** `--ecp` takes the SAME grammar and is
//! applied after the orbital basis, so an ECP is never forced on you just
//! because it happened to ship inside the set you wanted the orbitals from,
//! and an ECP can be taken from a different set than the orbitals:
//!
//! ```text
//! --basis "def2-tzvp; Ni=lanl2dz" --ecp none          orbitals only, no ECP
//! --basis def2-tzvp --ecp "Ni=stuttgart-rsc-1997"     def2 orbitals, Stuttgart core
//! ```
//!
//! An `--ecp` spec touches ONLY the elements it names (a base source
//! touches every element that source defines an ECP for). It is an
//! override, not a reset — except `none`, which clears.

use crate::basis::{data, BasisSet};
use crate::error::{DftError, Result};
use crate::system::symbol_to_z;
use std::collections::HashMap;

/// One parsed segment.
struct Assignment {
    /// Empty = the base (applies to every element).
    elements: Vec<u8>,
    source: String,
}

/// Split a spec into assignments, expanding `@file` includes.
fn parse(spec: &str) -> Result<Vec<Assignment>> {
    let mut out = Vec::new();
    parse_into(spec, &mut out, 0)?;
    Ok(out)
}

fn parse_into(spec: &str, out: &mut Vec<Assignment>, depth: usize) -> Result<()> {
    if depth > 4 {
        return Err(DftError::InvalidInput(
            "basis spec: @include nested more than 4 deep".to_string(),
        ));
    }
    for raw in spec.split([';', '\n', '\r']) {
        // `#` starts a comment, which only matters for @-included files but
        // costs nothing to honour everywhere.
        let seg = raw.split('#').next().unwrap_or("").trim();
        if seg.is_empty() {
            continue;
        }
        if let Some(path) = seg.strip_prefix('@') {
            let text = std::fs::read_to_string(path.trim())?;
            parse_into(&text, out, depth + 1)?;
            continue;
        }
        match seg.split_once('=') {
            None => out.push(Assignment {
                elements: Vec::new(),
                source: seg.to_string(),
            }),
            Some((lhs, rhs)) => {
                let source = rhs.trim();
                if source.is_empty() {
                    return Err(DftError::InvalidInput(format!(
                        "basis spec: '{seg}' names elements but no source"
                    )));
                }
                let mut elements = Vec::new();
                for sym in lhs.split(',') {
                    let sym = sym.trim();
                    if sym.is_empty() {
                        continue;
                    }
                    let z = symbol_to_z(sym).ok_or_else(|| {
                        DftError::InvalidInput(format!(
                            "basis spec: '{sym}' is not an element symbol or atomic number"
                        ))
                    })?;
                    elements.push(z);
                }
                if elements.is_empty() {
                    return Err(DftError::InvalidInput(format!(
                        "basis spec: '{seg}' has an empty element list"
                    )));
                }
                out.push(Assignment {
                    elements,
                    source: source.to_string(),
                });
            }
        }
    }
    Ok(())
}

/// Parses each distinct source once, however many segments mention it.
struct SourceCache {
    sets: HashMap<String, BasisSet>,
}

impl SourceCache {
    fn new() -> Self {
        Self {
            sets: HashMap::new(),
        }
    }

    fn get(&mut self, name: &str) -> Result<&BasisSet> {
        if !self.sets.contains_key(name) {
            let set = data::load_or_read(name)?;
            self.sets.insert(name.to_string(), set);
        }
        Ok(&self.sets[name])
    }
}

/// Whether a spec is a single plain source with no per-element parts —
/// the ordinary `--basis def2-svp` case.
pub fn is_plain(spec: &str) -> bool {
    !spec.contains('=') && !spec.contains(';') && !spec.trim_start().starts_with('@')
}

/// Build an orbital basis set from a spec.
///
/// A base source, if present, must come first and supplies every element
/// not named by a later segment. Per-element segments bring their source's
/// ECP for those elements along with the orbitals — override that with an
/// independent `--ecp` spec through [`apply_ecp_spec`].
pub fn load_spec(spec: &str) -> Result<BasisSet> {
    if is_plain(spec) {
        return data::load_or_read(spec);
    }
    let parts = parse(spec)?;
    let mut cache = SourceCache::new();
    let mut label = String::new();
    let mut out: Option<BasisSet> = None;

    for (k, part) in parts.iter().enumerate() {
        if part.elements.is_empty() {
            if k != 0 {
                return Err(DftError::InvalidInput(format!(
                    "basis spec: the base set '{}' must come first (segment {})",
                    part.source,
                    k + 1
                )));
            }
            if part.source.eq_ignore_ascii_case("none") {
                return Err(DftError::InvalidInput(
                    "basis spec: 'none' is not a basis; every atom needs orbitals".to_string(),
                ));
            }
            let base = cache.get(&part.source)?.clone();
            label = base.name.clone();
            out = Some(base);
            continue;
        }
        if part.source.eq_ignore_ascii_case("none") {
            return Err(DftError::InvalidInput(
                "basis spec: 'none' is not a basis; every atom needs orbitals \
                 (use the --ecp spec to drop a potential)"
                    .to_string(),
            ));
        }
        let src = cache.get(&part.source)?;
        let mut picked = Vec::new();
        for &z in &part.elements {
            let eb = src.element(z)?.clone();
            let ecp = src.ecp(z).cloned();
            picked.push((z, eb, ecp));
        }
        let set = out.get_or_insert_with(|| BasisSet::new(""));
        for (z, eb, ecp) in picked {
            set.insert(eb);
            match ecp {
                Some(e) => set.set_ecp(z, e),
                // A per-element override replaces the assignment wholesale:
                // silently keeping the base set's ECP under a different
                // element basis would be a mismatched core.
                None => set.clear_ecp(z),
            }
        }
        if !label.is_empty() {
            label.push('+');
        }
        label.push_str(&format!("{}:{}", element_list(&part.elements), part.source));
    }

    let mut set =
        out.ok_or_else(|| DftError::InvalidInput("basis spec: nothing assigned".to_string()))?;
    set.name = label;
    Ok(set)
}

/// Apply an ECP spec on top of an already-built basis set.
///
/// Only the elements the spec names are touched (a base source touches
/// every element it defines an ECP for). `none` clears instead of setting,
/// which is how you keep a set's orbitals while dropping its potential.
pub fn apply_ecp_spec(set: &mut BasisSet, spec: &str) -> Result<()> {
    let parts = parse(spec)?;
    let mut cache = SourceCache::new();
    for part in &parts {
        let clearing = part.source.eq_ignore_ascii_case("none");
        if part.elements.is_empty() {
            if clearing {
                for z in set.ecp_elements() {
                    set.clear_ecp(z);
                }
                continue;
            }
            let src = cache.get(&part.source)?;
            let picked: Vec<(u8, crate::basis::EcpPotential)> = src
                .ecp_elements()
                .into_iter()
                .filter_map(|z| src.ecp(z).map(|e| (z, e.clone())))
                .collect();
            for (z, e) in picked {
                set.set_ecp(z, e);
            }
            continue;
        }
        if clearing {
            for &z in &part.elements {
                set.clear_ecp(z);
            }
            continue;
        }
        let src = cache.get(&part.source)?;
        let mut picked = Vec::new();
        for &z in &part.elements {
            let e = src.ecp(z).ok_or_else(|| {
                DftError::InvalidInput(format!(
                    "ecp spec: '{}' defines no ECP for element {}",
                    part.source,
                    crate::system::z_to_symbol(z).unwrap_or("?")
                ))
            })?;
            picked.push((z, e.clone()));
        }
        for (z, e) in picked {
            set.set_ecp(z, e);
        }
    }
    Ok(())
}

fn element_list(zs: &[u8]) -> String {
    zs.iter()
        .map(|&z| crate::system::z_to_symbol(z).unwrap_or("?").to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plain name still means exactly what it always meant — the mixed
    /// path must not change the single-basis case.
    #[test]
    fn plain_name_is_unchanged() {
        let a = load_spec("def2-svp").unwrap();
        let b = data::load("def2-svp").unwrap();
        for z in [1u8, 6, 8, 26] {
            assert_eq!(a.has_element(z), b.has_element(z));
        }
    }

    /// The point of the feature: different elements really do get shells
    /// from different sets.
    #[test]
    fn per_element_sources_are_mixed() {
        let mixed = load_spec("def2-svp; H=sto-3g").unwrap();
        let sto = data::load("sto-3g").unwrap();
        let def2 = data::load("def2-svp").unwrap();
        let h = mixed.element(1).unwrap();
        assert_eq!(h.shells.len(), sto.element(1).unwrap().shells.len());
        assert_ne!(h.shells.len(), def2.element(1).unwrap().shells.len());
        // Carbon still comes from the base.
        assert_eq!(
            mixed.element(6).unwrap().shells.len(),
            def2.element(6).unwrap().shells.len()
        );
    }

    /// Orbitals from one set, core potential from another — the case that
    /// a bundled basis+ECP pair cannot express.
    #[test]
    fn ecp_is_assignable_independently_of_the_orbitals() {
        let mut set = load_spec("def2-svp").unwrap();
        assert!(set.ecp(28).is_none(), "def2-svp carries no Ni ECP here");
        apply_ecp_spec(&mut set, "Ni=lanl2dz").unwrap();
        let ecp = set.ecp(28).expect("Ni ECP assigned");
        assert_eq!(
            ecp.n_core,
            data::load("lanl2dz").unwrap().ecp(28).unwrap().n_core
        );
        // The orbitals did NOT come along with it.
        assert_eq!(
            set.element(28).unwrap().shells.len(),
            data::load("def2-svp")
                .unwrap()
                .element(28)
                .unwrap()
                .shells
                .len()
        );
    }

    /// `none` drops a potential the orbital set brought with it, which is
    /// the other half of "basis and ECP are independent".
    #[test]
    fn ecp_none_clears() {
        let mut set = load_spec("lanl2dz").unwrap();
        assert!(set.ecp(28).is_some());
        apply_ecp_spec(&mut set, "none").unwrap();
        assert!(set.ecp(28).is_none());
        assert!(set.has_element(28), "orbitals survive");
    }

    /// A base set placed after an override is a typo, not an instruction:
    /// applying it would silently undo the override.
    #[test]
    fn base_must_come_first() {
        assert!(load_spec("H=sto-3g; def2-svp").is_err());
    }

    #[test]
    fn unknown_symbol_is_rejected() {
        assert!(load_spec("def2-svp; Xx=sto-3g").is_err());
    }
}
