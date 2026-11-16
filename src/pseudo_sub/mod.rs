// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Anisotropic pseudo-substituent sites ("G") — v0, the plumbing layer.
//!
//! v0 ships NO chemistry. It exists so that a later modelling error (a bad
//! fit, a wrong symmetry class, a missing frame convention) can never be
//! confused with a wiring error (the wrong basis landing on the wrong
//! atom, a Z-tag falling back to defaults somewhere). That confusion is
//! only possible to rule out if the plumbing is proven correct *before*
//! any numbers are attached to it, which is what the identity gate at the
//! bottom of this file does.
//!
//! # The representation
//!
//! A pseudo-substituent site is represented as:
//!
//! ```text
//! Atom { z: 1, position }        // the nucleus: Z=+1, one electron,
//!                                 // H-like grid/radii/mass, all automatic
//! + a per-atom basis override    // G's own valence basis
//!   (MolBasis::build_with_overrides, src/basis/mod.rs)
//! + a separate PseudoSite entry  // symmetry, mass, operator (this file)
//! ```
//!
//! ## Why `z = 1`, not a pseudo-element `Z` tag
//!
//! This was measured, not guessed, and should not be revisited without new
//! measurement. `atom.z` drives `GridOptions::resolve`'s row scale,
//! `treutler_xi`, and `bragg_slater_bohr` (three separate grid quantities),
//! plus `smd_radius`, the PCM cavity radius, `atomic_mass_au`, the Molden
//! writer, and the SAD guess. A `Z >= 119` tag falls back to a default in
//! every one of those, silently — which would make the identity gate below
//! fail by construction (the "G" calculation would diverge from ordinary
//! CH4 the moment any of those subsystems saw the tag). `z = 1` sidesteps
//! all of it: G's nucleus is treated exactly like a hydrogen nucleus by
//! every subsystem that does not know "pseudo-substituent" exists, and the
//! only place G's identity as a *site* — as opposed to plain hydrogen —
//! shows up is the [`PseudoSiteTable`] and the basis override.
//!
//! # What remains unparameterized
//!
//! The local-frame Jacobian and real Wigner-D machinery are implemented and
//! independently gated in [`frame`] and [`wigner`]. Frame reference topology
//! is explicit, so an unrelated high-Z atom cannot enter the candidate set.
//! There is still no fitted
//! `V_G`, no assembly of that operator into a Hamiltonian, and no SCF,
//! gradient, or Hessian consumer of [`PseudoSiteTable`]. Named-site input can
//! be parsed with [`parse_named_sites`], but resolving it to physical sites
//! deliberately fails until a provenance-bearing parameter fit passes the
//! model's stated validation gates.

use crate::basis::ElementBasis;
use crate::error::{DftError, Result};
use std::collections::HashSet;
use std::str::FromStr;

pub mod fit;
pub mod frame;
pub mod presets;
pub mod reduction;
pub mod reduction_application;
pub mod reduction_basis;
pub mod reduction_geometry;
pub mod reduction_kernels;
pub mod reduction_reference;
pub mod reduction_roles;
pub mod reduction_support;
pub mod scalar_geometry;
pub mod wigner;

/// Local symmetry class of a pseudo-substituent site's operator.
///
/// v0 does not use this for anything (there is no operator to symmetrize —
/// see [`SiteOperator`]); it exists so the table's shape does not have to
/// change again when v1 starts reading it to constrain the fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteSymmetry {
    /// Full rotational symmetry about the site's bond axis (e.g. a halide
    /// analogue: the operator depends only on the angle to the axis, not
    /// the azimuth around it).
    Cylindrical,
    /// C2v: two orthogonal mirror planes intersecting in the bond axis.
    C2v,
    /// Cs: a single mirror plane.
    Cs,
    /// No symmetry constraint beyond the identity.
    None,
}

/// The static one-electron operator carried by a pseudo-substituent site.
///
/// v0 ships no fitted parameters at all — this is an empty placeholder.
/// [`SiteOperator::is_zero`] always returns `true`, and there is no
/// constructor other than [`SiteOperator::default`], so there is no way to
/// build a nonzero operator through this API. Do not add fields here
/// without a source for the numbers that would go in them: a capability
/// that cannot be verified is not shipped (see [`NamedSubstituent`] below
/// for what that means in practice).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SiteOperator {
    _v0_has_no_parameters: (),
}

impl SiteOperator {
    /// v0: always `true` (there are no parameters to be nonzero).
    pub const fn is_zero(&self) -> bool {
        true
    }
}

/// A pseudo-substituent site "G": which atom it sits on, its local
/// symmetry class, its own mass, and its (v0: always zero) operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PseudoSite {
    /// Index into `Molecule::atoms` of the nucleus this site sits on.
    ///
    /// That atom's `z` should normally be `1` — see the module docs for
    /// why a hydrogen-like nucleus, not a pseudo-element `Z` tag, is the
    /// representation this project uses. Nothing in v0 enforces that here
    /// (a `PseudoSiteTable` is inert plumbing with no consumer yet), but
    /// every later stage assumes it.
    pub atom: usize,
    pub symmetry: SiteSymmetry,
    /// Site mass in amu (same convention as `constants::ATOMIC_MASS_AMU`,
    /// which is the value the plain-hydrogen atom this sits on already
    /// carries automatically). v0 does not read this field anywhere; it is
    /// reserved for a later stage that lets a site's mass diverge from
    /// ordinary hydrogen (e.g. for isotope or united-atom mass reasons).
    pub mass_amu: f64,
    pub operator: SiteOperator,
}

/// The set of pseudo-substituent sites on a molecule, keyed by atom index.
///
/// v0: purely a data container. No SCF, gradient, or Hessian code path
/// reads this yet — assembling `V_G` into the Hamiltonian is explicitly
/// out of scope for v0 (see the module docs).
#[derive(Clone, Debug, Default)]
pub struct PseudoSiteTable {
    sites: Vec<PseudoSite>,
}

impl PseudoSiteTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, site: PseudoSite) {
        self.sites.push(site);
    }

    pub fn sites(&self) -> &[PseudoSite] {
        &self.sites
    }

    /// The site sitting on atom index `atom`, if any (first match; v0 does
    /// not forbid duplicates, but nothing produces them either).
    pub fn get(&self, atom: usize) -> Option<&PseudoSite> {
        self.sites.iter().find(|s| s.atom == atom)
    }
}

/// A named substituent from the chemist's vocabulary (methyl, ethyl,
/// tert-butyl, phenyl, ...).
///
/// Every variant here refuses to build through [`NamedSubstituent::build`]
/// in v0. This project's rule is absolute: a capability that cannot be
/// verified is not shipped, and the undetermined part errors by name
/// rather than returning a plausible number. `pcm::PcmSolver::require_k_is_s`
/// (`src/pcm.rs`) is the worked example this follows — `DftError::Unsupported`,
/// not a silent default or an `InvalidInput` that a caller might mistake
/// for their own mistake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedSubstituent {
    Me,
    Et,
    TBu,
    Ph,
    /// Trifluoromethyl. This name is accepted by the input layer, but the
    /// original Phase-4 model specification contains no CF3 parameter or
    /// validation programme, so [`Self::build`] refuses it explicitly.
    Cf3,
    /// Methoxy. This name is accepted by the input layer, but the original
    /// Phase-4 model specification contains no OMe parameter or validation
    /// programme, so [`Self::build`] refuses it explicitly.
    OMe,
}

impl NamedSubstituent {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Me => "Me",
            Self::Et => "Et",
            Self::TBu => "tBu",
            Self::Ph => "Ph",
            Self::Cf3 => "CF3",
            Self::OMe => "OMe",
        }
    }

    /// Which parameter-validation milestone governs this name, if any.
    ///
    /// This is metadata only; it does not assert that the corresponding
    /// parameter set exists. CF3 and OMe return `None` because they were not
    /// defined by the Phase-4 v1/v2 specification.
    pub const fn planned_stage(&self) -> Option<&'static str> {
        match self {
            Self::Me | Self::TBu => Some("v1"),
            Self::Et | Self::Ph => Some("v2"),
            Self::Cf3 | Self::OMe => None,
        }
    }

    /// Build the fitted [`PseudoSite`] (basis, symmetry, mass, operator)
    /// for this named substituent, sitting on atom index `atom`.
    ///
    /// Always fails while no substituent has a fitted `V_G`, a chosen
    /// basis, or a validated parameter record. The current release ships the
    /// plumbing — a hydrogen-like site with a provably zero operator,
    /// gated against ordinary CH4 to 1e-10 for both HF and SVWN5 (see
    /// `tests::identity_gate_hf` / `tests::identity_gate_svwn5` below) —
    /// not chemistry.
    pub fn build(&self, atom: usize) -> Result<PseudoSite> {
        let missing = match self.planned_stage() {
            Some("v1") => format!(
                "the accepted v1 parameters for `{}` have not been determined: a \
                    provenance-bearing fit of the valence basis, core density (l<=1), B \
                    operator (l<=1), C6 and core-repulsion term must pass the X-Me bond, \
                    ethane torsion, neopentane structure, and analytic-gradient gates \
                    before this name can build a physical site",
                self.name()
            ),
            Some("v2") => format!(
                "the accepted v2 parameters for `{}` have not been determined: a \
                    provenance-bearing fit of the valence/auxiliary basis and V_G operator, \
                    plus torsion, torque, G-G interaction and 1-2/1-3 exclusion data, must \
                    pass the torsion-curve, conformer, cone-angle, transferability, and \
                    analytic-gradient gates before this name can build a physical site",
                self.name()
            ),
            None => format!(
                "the parameters for `{}` have not been determined, and this substituent \
                    is not defined by the Phase-4 v1/v2 specification; a model form, \
                    parameter provenance, training/validation targets, acceptance \
                    thresholds, and fitted valence basis/V_G operator are all required \
                    before this name can build a physical site",
                self.name()
            ),
            Some(_) => unreachable!("planned_stage returns only v1, v2, or None"),
        };
        Err(DftError::Unsupported {
            feature: format!("named pseudo-substituent `{}` at atom {atom}", self.name()),
            missing,
        })
    }
}

impl FromStr for NamedSubstituent {
    type Err = DftError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "me" | "methyl" => Ok(Self::Me),
            "et" | "ethyl" => Ok(Self::Et),
            "tbu" | "tert-butyl" | "tertbutyl" => Ok(Self::TBu),
            "ph" | "phenyl" => Ok(Self::Ph),
            "cf3" | "trifluoromethyl" | "trifluoro-methyl" => Ok(Self::Cf3),
            "ome" | "methoxy" => Ok(Self::OMe),
            _ => Err(DftError::InvalidInput(format!(
                "pseudo-substituent name `{}` is unknown; expected one of Me, Et, tBu, Ph, CF3, OMe",
                value.trim()
            ))),
        }
    }
}

/// One parsed `atom-index=name` request.
///
/// Parsing a name is intentionally separate from building its physical
/// operator. This lets every public surface share the exact 0-based grammar
/// and error handling without pretending that an unvalidated parameter set
/// exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedSiteRequest {
    pub atom: usize,
    pub substituent: NamedSubstituent,
}

impl NamedSiteRequest {
    /// Check that the zero-based atom index exists and denotes the required
    /// hydrogen-like (`Z=1`) G nucleus.
    pub fn validate_for_molecule(&self, molecule: &crate::system::Molecule) -> Result<()> {
        let atom = molecule.atoms.get(self.atom).ok_or_else(|| {
            DftError::InvalidInput(format!(
                "pseudo-substituent `{}` uses atom index {}, but the molecule has only {} atoms",
                self.substituent.name(),
                self.atom,
                molecule.atoms.len()
            ))
        })?;
        if atom.z != 1 {
            return Err(DftError::InvalidInput(format!(
                "pseudo-substituent `{}` at atom {} requires Atom {{ z: 1 }}, but that atom has Z={}",
                self.substituent.name(),
                self.atom,
                atom.z
            )));
        }
        Ok(())
    }

    /// Resolve this request to a physical site. Currently returns the
    /// provenance/validation refusal from [`NamedSubstituent::build`].
    pub fn build(self) -> Result<PseudoSite> {
        self.substituent.build(self.atom)
    }

    /// Validate the G-site representation, then resolve it to a physical
    /// site. Invalid indexing/Z is reported as caller input; a valid request
    /// reaches the typed missing-parameter refusal.
    pub fn build_for_molecule(self, molecule: &crate::system::Molecule) -> Result<PseudoSite> {
        self.validate_for_molecule(molecule)?;
        self.build()
    }
}

/// Parse `atom-index=name` assignments separated by semicolons or newlines.
///
/// Atom indices are zero-based. `#` starts a comment, and a segment of the
/// form `@path` includes another spec file (nested at most four levels),
/// matching the basis-spec convention. Assigning an atom more than once is
/// rejected because selecting the first or last entry would be a silent
/// chemistry change.
pub fn parse_named_sites(spec: &str) -> Result<Vec<NamedSiteRequest>> {
    let mut out = Vec::new();
    let mut atoms = HashSet::new();
    parse_named_sites_into(spec, &mut out, &mut atoms, 0)?;
    if out.is_empty() {
        return Err(DftError::InvalidInput(
            "pseudo-substituent spec: no site assignments were provided".to_string(),
        ));
    }
    Ok(out)
}

/// Parse, validate, and build all named sites for a molecule.
///
/// In v0.1.1 every recognized name returns [`DftError::Unsupported`] before
/// this can return a table. Keeping the shared build boundary here ensures the
/// CLI and Python surfaces agree on the zero-based grammar and `Z=1` check.
/// Future callers must still wire a returned table into the Hamiltonian rather
/// than treating successful construction as a calculation no-op.
pub fn build_named_site_table(
    spec: &str,
    molecule: &crate::system::Molecule,
) -> Result<PseudoSiteTable> {
    let requests = parse_named_sites(spec)?;
    let mut table = PseudoSiteTable::new();
    for request in requests {
        table.push(request.build_for_molecule(molecule)?);
    }
    Ok(table)
}

fn parse_named_sites_into(
    spec: &str,
    out: &mut Vec<NamedSiteRequest>,
    atoms: &mut HashSet<usize>,
    depth: usize,
) -> Result<()> {
    if depth > 4 {
        return Err(DftError::InvalidInput(
            "pseudo-substituent spec: @include nested more than 4 deep".to_string(),
        ));
    }
    for raw in spec.split([';', '\n', '\r']) {
        let segment = raw.split('#').next().unwrap_or("").trim();
        if segment.is_empty() {
            continue;
        }
        if let Some(path) = segment.strip_prefix('@') {
            let path = path.trim();
            if path.is_empty() {
                return Err(DftError::InvalidInput(
                    "pseudo-substituent spec: @include has an empty path".to_string(),
                ));
            }
            let text = std::fs::read_to_string(path)?;
            parse_named_sites_into(&text, out, atoms, depth + 1)?;
            continue;
        }
        let (atom_text, name_text) = segment.split_once('=').ok_or_else(|| {
            DftError::InvalidInput(format!(
                "pseudo-substituent spec: `{segment}` must have the form atom-index=name"
            ))
        })?;
        if name_text.contains('=') {
            return Err(DftError::InvalidInput(format!(
                "pseudo-substituent spec: `{segment}` contains more than one '='"
            )));
        }
        let atom = atom_text.trim().parse::<usize>().map_err(|_| {
            DftError::InvalidInput(format!(
                "pseudo-substituent spec: `{}` is not a zero-based atom index",
                atom_text.trim()
            ))
        })?;
        let substituent = name_text.parse::<NamedSubstituent>()?;
        if !atoms.insert(atom) {
            return Err(DftError::InvalidInput(format!(
                "pseudo-substituent spec: atom {atom} is assigned more than once"
            )));
        }
        out.push(NamedSiteRequest { atom, substituent });
    }
    Ok(())
}

/// Re-exported for callers building per-atom overrides directly
/// (`MolBasis::build_with_overrides`) without going through this module.
pub type BasisOverride = (usize, ElementBasis);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{data, MolBasis};
    use crate::constants::ANGSTROM_TO_BOHR;
    use crate::gradient::analytic_gradient;
    use crate::grid::{GridOptions, MolecularGrid};
    use crate::math::Vec3;
    use crate::scf::{EriMode, ScfOptions};
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec, ResolvedFunctional};

    /// Tetrahedral CH4 (idealized geometry, 1.089 Angstrom C-H bonds).
    fn methane() -> Molecule {
        let bl = 1.089 * ANGSTROM_TO_BOHR;
        let s = bl / 3.0_f64.sqrt();
        Molecule::new(vec![
            Atom {
                z: 6,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(s, s, s),
            },
            Atom {
                z: 1,
                position: Vec3::new(s, -s, -s),
            },
            Atom {
                z: 1,
                position: Vec3::new(-s, s, -s),
            },
            Atom {
                z: 1,
                position: Vec3::new(-s, -s, s),
            },
        ])
    }

    fn tight() -> ScfOptions {
        ScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-10,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        }
    }

    /// Build the "C + 3H + 1G" `MolBasis`: identical molecule to ordinary
    /// CH4, but the fourth hydrogen's basis is placed through
    /// `build_with_overrides` with an explicit per-atom copy of the
    /// ordinary Z=1 entry, and a `PseudoSiteTable` carrying one inert
    /// site sits alongside it (constructed, never consumed — v0 has no
    /// V_G assembly).
    fn build_g_variant(
        mol: &Molecule,
        basis: &crate::basis::BasisSet,
    ) -> (MolBasis, PseudoSiteTable) {
        let h_basis = basis.element(1).unwrap().clone();
        let mb = MolBasis::build_with_overrides(mol, basis, &[(4, h_basis)]).unwrap();
        let mut sites = PseudoSiteTable::new();
        sites.push(PseudoSite {
            atom: 4,
            symmetry: SiteSymmetry::None,
            mass_amu: crate::constants::ATOMIC_MASS_AMU[1],
            operator: SiteOperator::default(),
        });
        (mb, sites)
    }

    fn max_grad_diff(a: &[Vec3], b: &[Vec3]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0_f64, f64::max)
    }

    /// v0's whole point: a "G" site configured to be a hydrogen (z=1, the
    /// ordinary hydrogen basis, zero operator, no classical terms) must
    /// reproduce ordinary CH4 exactly. Hartree-Fock half.
    #[test]
    fn identity_gate_hf() {
        let basis = data::load("sto-3g").unwrap();
        let mol = methane();
        let mb_ordinary = MolBasis::build(&mol, &basis).unwrap();
        let (mb_g, sites) = build_g_variant(&mol, &basis);
        assert!(sites.get(4).unwrap().operator.is_zero());

        let opts = tight();
        let hf = ResolvedFunctional::hartree_fock();
        let ordinary = analytic_gradient(&mol, &mb_ordinary, &opts, &hf, None).unwrap();
        let g_variant = analytic_gradient(&mol, &mb_g, &opts, &hf, None).unwrap();

        let de = (ordinary.scf.energy - g_variant.scf.energy).abs();
        let dg = max_grad_diff(&ordinary.gradient, &g_variant.gradient);
        assert!(
            de < 1e-10,
            "HF energy identity broken: ordinary={:.15}, G-variant={:.15}, |diff|={de:.3e}",
            ordinary.scf.energy,
            g_variant.scf.energy
        );
        assert!(
            dg < 1e-10,
            "HF gradient identity broken: max |diff|={dg:.3e}"
        );
    }

    /// The DFT half of the identity gate: this is the part that would fail
    /// under a Z-tag representation, because the quadrature grid is built
    /// from `atom.z` (row scale, radial map, Bragg-Slater radius for the
    /// Becke partition) — so it is not optional. SVWN5 (LDA, grid-based).
    #[test]
    fn identity_gate_svwn5() {
        let basis = data::load("sto-3g").unwrap();
        let mol = methane();
        let mb_ordinary = MolBasis::build(&mol, &basis).unwrap();
        let (mb_g, sites) = build_g_variant(&mol, &basis);
        assert!(sites.get(4).unwrap().operator.is_zero());

        let grid = MolecularGrid::build(&mol, &GridOptions::default()).unwrap();
        let opts = tight();
        let func = resolve(&FunctionalSpec::Combined("SVWN5".into())).unwrap();
        let ordinary = analytic_gradient(&mol, &mb_ordinary, &opts, &func, Some(&grid)).unwrap();
        let g_variant = analytic_gradient(&mol, &mb_g, &opts, &func, Some(&grid)).unwrap();

        let de = (ordinary.scf.energy - g_variant.scf.energy).abs();
        let dg = max_grad_diff(&ordinary.gradient, &g_variant.gradient);
        assert!(
            de < 1e-10,
            "SVWN5 energy identity broken: ordinary={:.15}, G-variant={:.15}, |diff|={de:.3e}",
            ordinary.scf.energy,
            g_variant.scf.energy
        );
        assert!(
            dg < 1e-10,
            "SVWN5 gradient identity broken: max |diff|={dg:.3e}"
        );
    }

    #[test]
    fn named_substituents_refuse() {
        for sub in [
            NamedSubstituent::Me,
            NamedSubstituent::Et,
            NamedSubstituent::TBu,
            NamedSubstituent::Ph,
            NamedSubstituent::Cf3,
            NamedSubstituent::OMe,
        ] {
            match sub.build(0) {
                Err(DftError::Unsupported { feature, missing }) => {
                    assert!(feature.contains(sub.name()));
                    assert!(missing.contains("not been determined"));
                    assert!(missing.contains("provenance") || missing.contains("not defined"));
                }
                other => panic!("{:?}: expected DftError::Unsupported, got {other:?}", sub),
            }
        }
    }

    #[test]
    fn named_substituent_parser_canonicalizes_supported_vocabulary() {
        let cases = [
            ("Me", NamedSubstituent::Me, "Me"),
            ("methyl", NamedSubstituent::Me, "Me"),
            ("ETHYL", NamedSubstituent::Et, "Et"),
            ("tert-butyl", NamedSubstituent::TBu, "tBu"),
            ("phenyl", NamedSubstituent::Ph, "Ph"),
            ("trifluoromethyl", NamedSubstituent::Cf3, "CF3"),
            ("methoxy", NamedSubstituent::OMe, "OMe"),
        ];
        for (input, expected, canonical) in cases {
            let parsed: NamedSubstituent = input.parse().unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.name(), canonical);
        }
        assert_eq!(NamedSubstituent::Me.planned_stage(), Some("v1"));
        assert_eq!(NamedSubstituent::Ph.planned_stage(), Some("v2"));
        assert_eq!(NamedSubstituent::Cf3.planned_stage(), None);
    }

    #[test]
    fn named_site_spec_is_zero_based_deterministic_and_comment_aware() {
        let parsed = parse_named_sites("4=Me; 7=Ph # phenyl site\n9=CF3\r\n 11 = methoxy").unwrap();
        assert_eq!(
            parsed,
            vec![
                NamedSiteRequest {
                    atom: 4,
                    substituent: NamedSubstituent::Me,
                },
                NamedSiteRequest {
                    atom: 7,
                    substituent: NamedSubstituent::Ph,
                },
                NamedSiteRequest {
                    atom: 9,
                    substituent: NamedSubstituent::Cf3,
                },
                NamedSiteRequest {
                    atom: 11,
                    substituent: NamedSubstituent::OMe,
                },
            ]
        );

        for bad in ["", "4", "x=Me", "4=unknown", "4=Me;4=Et", "4=Me=Et"] {
            assert!(
                matches!(parse_named_sites(bad), Err(DftError::InvalidInput(_))),
                "spec `{bad}` should be rejected"
            );
        }
    }

    #[test]
    fn named_site_spec_reads_at_file() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "dft-rs-pseudo-sites-{}-{nonce}.spec",
            std::process::id()
        ));
        std::fs::write(&path, "3=tBu # fitted name is still refused\n5=OMe\n").unwrap();
        let parsed = parse_named_sites(&format!("@{}", path.display())).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].atom, 3);
        assert_eq!(parsed[0].substituent, NamedSubstituent::TBu);
        assert_eq!(parsed[1].atom, 5);
        assert_eq!(parsed[1].substituent, NamedSubstituent::OMe);
    }

    #[test]
    fn request_build_preserves_the_atom_and_name_in_refusal() {
        let request = NamedSiteRequest {
            atom: 12,
            substituent: NamedSubstituent::OMe,
        };
        match request.build() {
            Err(DftError::Unsupported { feature, missing }) => {
                assert!(feature.contains("OMe"));
                assert!(feature.contains("12"));
                assert!(missing.contains("not defined by the Phase-4 v1/v2 specification"));
            }
            other => panic!("expected a typed parameter refusal, got {other:?}"),
        }
    }

    #[test]
    fn request_validates_index_and_hydrogen_like_site_before_parameter_lookup() {
        let mol = Molecule::new(vec![
            Atom {
                z: 6,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 2.0),
            },
        ]);
        let valid = NamedSiteRequest {
            atom: 1,
            substituent: NamedSubstituent::Me,
        };
        assert!(valid.validate_for_molecule(&mol).is_ok());
        assert!(matches!(
            valid.build_for_molecule(&mol),
            Err(DftError::Unsupported { .. })
        ));

        for invalid in [
            NamedSiteRequest {
                atom: 0,
                substituent: NamedSubstituent::Me,
            },
            NamedSiteRequest {
                atom: 2,
                substituent: NamedSubstituent::Me,
            },
        ] {
            assert!(matches!(
                invalid.build_for_molecule(&mol),
                Err(DftError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn site_operator_default_is_zero() {
        assert!(SiteOperator::default().is_zero());
    }

    #[test]
    fn pseudo_site_table_lookup() {
        let mut t = PseudoSiteTable::new();
        assert!(t.get(0).is_none());
        t.push(PseudoSite {
            atom: 2,
            symmetry: SiteSymmetry::Cylindrical,
            mass_amu: 1.008,
            operator: SiteOperator::default(),
        });
        assert!(t.get(0).is_none());
        let site = t.get(2).unwrap();
        assert_eq!(site.atom, 2);
        assert_eq!(site.symmetry, SiteSymmetry::Cylindrical);
        assert_eq!(t.sites().len(), 1);
    }
}
