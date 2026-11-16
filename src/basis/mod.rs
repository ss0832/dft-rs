// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gaussian basis sets: contracted shells, normalization, and placement of a
//! basis onto a molecule.
//!
//! Normalization convention (CCA standard): each primitive is scaled by the
//! norm of its `(l,0,0)` Cartesian component, then the contraction is scaled
//! so the contracted `(l,0,0)` self-overlap is 1. All Cartesian components of
//! a shell share this radial normalization; the real-solid-harmonic transform
//! (`aos.rs`) then yields orthonormal spherical AOs on a single shell.

pub mod aos;
pub mod data;
pub mod g94;
pub mod mixed;

use crate::error::{DftError, Result};
use crate::math::Vec3;
use crate::system::Molecule;

/// One contracted shell of a basis-set definition (element-level, unplaced).
#[derive(Clone, Debug)]
pub struct Shell {
    /// Angular momentum (0 = s, 1 = p, 2 = d, ...).
    pub l: u8,
    pub exps: Vec<f64>,
    /// Contraction coefficients as read from the file (for normalized
    /// primitives); normalized on placement.
    pub coefs: Vec<f64>,
}

/// All shells of one element in a basis set.
#[derive(Clone, Debug)]
pub struct ElementBasis {
    pub z: u8,
    pub shells: Vec<Shell>,
}

/// One radial term `c · r^{n−2} · exp(−α r²)` of an ECP channel.
#[derive(Clone, Debug)]
pub struct EcpTerm {
    /// The `n` of `r^{n−2}` as written in the Gaussian94 block.
    pub power: i32,
    pub exp: f64,
    pub coef: f64,
}

/// A semi-local effective core potential
/// `U(r) = U_L(r) + Σ_{l<L} Σ_m |lm⟩ (U_l − U_L) ⟨lm|`.
#[derive(Clone, Debug)]
pub struct EcpPotential {
    /// Core electrons replaced by the potential.
    pub n_core: u32,
    /// The local channel's angular momentum L.
    pub lmax: usize,
    /// Local channel U_L radial terms.
    pub local: Vec<EcpTerm>,
    /// Projected channels `(U_l − U_L)` for l = 0..L−1, index = l.
    pub semilocal: Vec<Vec<EcpTerm>>,
}

/// A parsed basis set (element-indexed).
#[derive(Clone, Debug)]
pub struct BasisSet {
    pub name: String,
    /// Indexed by atomic number (0 unused).
    elements: Vec<Option<ElementBasis>>,
    /// Effective core potentials per Z (LANL/Stuttgart-style sets).
    ecps: Vec<Option<EcpPotential>>,
}

impl BasisSet {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            elements: vec![None; 119],
            ecps: vec![None; 119],
        }
    }

    pub fn insert(&mut self, element: ElementBasis) {
        let z = element.z as usize;
        if z >= self.elements.len() {
            self.elements.resize(z + 1, None);
            self.ecps.resize(z + 1, None);
        }
        self.elements[z] = Some(element);
    }

    pub fn set_ecp(&mut self, z: u8, ecp: EcpPotential) {
        let z = z as usize;
        if z >= self.ecps.len() {
            self.elements.resize(z + 1, None);
            self.ecps.resize(z + 1, None);
        }
        self.ecps[z] = Some(ecp);
    }

    /// Drop the ECP on element `z`, leaving its orbitals in place. The
    /// basis and the core potential are separately assignable, so removing
    /// one must not disturb the other.
    pub fn clear_ecp(&mut self, z: u8) {
        if let Some(slot) = self.ecps.get_mut(z as usize) {
            *slot = None;
        }
    }

    /// Atomic numbers this set defines an ECP for.
    pub fn ecp_elements(&self) -> Vec<u8> {
        self.ecps
            .iter()
            .enumerate()
            .filter_map(|(z, e)| e.as_ref().map(|_| z as u8))
            .collect()
    }

    /// The ECP definition for element `z`, if the set carries one.
    pub fn ecp(&self, z: u8) -> Option<&EcpPotential> {
        self.ecps.get(z as usize).and_then(|e| e.as_ref())
    }

    /// Number of ECP core electrons for element `z`, if any.
    pub fn ecp_core(&self, z: u8) -> Option<u32> {
        self.ecp(z).map(|e| e.n_core)
    }

    pub fn element(&self, z: u8) -> Result<&ElementBasis> {
        self.elements
            .get(z as usize)
            .and_then(|e| e.as_ref())
            .ok_or_else(|| DftError::MissingElement {
                basis: self.name.clone(),
                z,
            })
    }

    pub fn has_element(&self, z: u8) -> bool {
        self.elements
            .get(z as usize)
            .map(|e| e.is_some())
            .unwrap_or(false)
    }
}

/// Number of Cartesian components of angular momentum `l`: (l+1)(l+2)/2.
#[inline]
pub fn n_cart(l: u8) -> usize {
    let l = l as usize;
    (l + 1) * (l + 2) / 2
}

/// Number of spherical components: 2l+1.
#[inline]
pub fn n_sph(l: u8) -> usize {
    2 * (l as usize) + 1
}

/// Cartesian component exponents `(lx, ly, lz)` of shell `l`, in the fixed
/// internal (CCA/lexicographic, x-major) order. For l=2:
/// xx, xy, xz, yy, yz, zz.
pub fn cart_components(l: u8) -> Vec<(i32, i32, i32)> {
    let l = l as i32;
    let mut out = Vec::with_capacity(n_cart(l as u8));
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            out.push((lx, ly, l - lx - ly));
        }
    }
    out
}

/// Norm of a primitive Cartesian Gaussian `x^lx y^ly z^lz exp(-a r²)`:
/// `N² = (2a/π)^{3/2} (4a)^{lx+ly+lz} / ((2lx−1)!!(2ly−1)!!(2lz−1)!!)`.
pub fn primitive_norm(a: f64, (lx, ly, lz): (i32, i32, i32)) -> f64 {
    let l = (lx + ly + lz) as u32;
    let n2 = (2.0 * a / std::f64::consts::PI).powf(1.5) * (4.0 * a).powi(l as i32)
        / (double_factorial(2 * lx - 1)
            * double_factorial(2 * ly - 1)
            * double_factorial(2 * lz - 1));
    n2.sqrt()
}

/// (n)!! with (−1)!! = 1 and 0!! = 1.
pub fn double_factorial(n: i32) -> f64 {
    let mut v = 1.0;
    let mut k = n;
    while k > 1 {
        v *= k as f64;
        k -= 2;
    }
    v
}

/// One shell placed on an atom, with normalized contraction coefficients.
#[derive(Clone, Debug)]
pub struct MolShell {
    pub l: u8,
    pub atom: usize,
    pub center: Vec3,
    pub exps: Vec<f64>,
    /// Coefficients including primitive norms and the contracted-shell
    /// normalization (see module docs).
    pub coefs: Vec<f64>,
    /// Offset of this shell's first function in the spherical AO ordering.
    pub sph_off: usize,
    /// Offset in the Cartesian AO ordering.
    pub cart_off: usize,
}

/// An effective core potential placed on an atom of the molecule.
#[derive(Clone, Debug)]
pub struct AtomEcp {
    pub atom: usize,
    pub center: Vec3,
    pub potential: EcpPotential,
}

/// A basis set instantiated on a molecule: the AO layout used everywhere
/// downstream (integrals, SCF, properties).
#[derive(Clone, Debug)]
pub struct MolBasis {
    pub shells: Vec<MolShell>,
    pub nao_sph: usize,
    pub nao_cart: usize,
    /// Highest angular momentum present (grid/integral sizing).
    pub max_l: u8,
    /// Effective core potentials carried by the basis (LANL/Stuttgart sets).
    pub ecps: Vec<AtomEcp>,
    /// Total core electrons replaced by ECPs (Σ n_core).
    pub n_core_electrons: u32,
}

impl MolBasis {
    /// Owning atom index of every spherical AO (Mulliken/Löwdin partitions).
    pub fn sph_ao_atoms(&self) -> Vec<usize> {
        let mut owner = vec![0usize; self.nao_sph];
        for sh in &self.shells {
            for k in 0..n_sph(sh.l) {
                owner[sh.sph_off + k] = sh.atom;
            }
        }
        owner
    }

    /// Nuclear charges with ECP cores subtracted (Z_eff per atom).
    pub fn effective_charges(&self, molecule: &Molecule) -> Vec<f64> {
        let mut z: Vec<f64> = molecule.atoms.iter().map(|a| a.z as f64).collect();
        for ecp in &self.ecps {
            z[ecp.atom] -= ecp.potential.n_core as f64;
        }
        z
    }

    /// Nuclear repulsion energy on the effective charges.
    pub fn nuclear_repulsion(&self, molecule: &Molecule) -> f64 {
        let z = self.effective_charges(molecule);
        let mut e = 0.0;
        for i in 0..molecule.atoms.len() {
            for j in (i + 1)..molecule.atoms.len() {
                let r = (molecule.atoms[i].position - molecule.atoms[j].position).norm();
                e += z[i] * z[j] / r;
            }
        }
        e
    }

    /// Place `basis` on every atom of `molecule`, normalizing contractions.
    ///
    /// Thin call into [`Self::build_with_overrides`] with no overrides, so
    /// this function's behavior — and every existing test against it — is
    /// unchanged by the override machinery below.
    pub fn build(molecule: &Molecule, basis: &BasisSet) -> Result<Self> {
        Self::build_with_overrides(molecule, basis, &[])
    }

    /// Place `basis` on every atom of `molecule`, except that atoms named
    /// in `overrides` use the paired [`ElementBasis`] instead of the
    /// element-keyed lookup `basis.element(atom.z)`.
    ///
    /// This is what lets two atoms sharing the same `z` carry different
    /// bases — e.g. a pseudo-substituent site ("G", see `pseudo_sub`) whose
    /// nucleus is an ordinary `z = 1` hydrogen (so grids, radii, mass and
    /// every other `atom.z`-keyed subsystem treat it exactly like one) but
    /// whose *valence basis* is chosen per-site rather than per-element.
    /// `GridOptions::per_atom` is the precedent for this "per-atom beats
    /// per-element" priority: an override here always wins over `basis`'s
    /// own entry for that atom's `z`, and is looked up by atom index, not
    /// by `ElementBasis::z` (which need not match `atom.z` at all).
    ///
    /// ECPs are unaffected: they are still looked up from `basis` by
    /// `atom.z`, exactly as in the unoverridden path.
    pub fn build_with_overrides(
        molecule: &Molecule,
        basis: &BasisSet,
        overrides: &[(usize, ElementBasis)],
    ) -> Result<Self> {
        for &(idx, _) in overrides {
            if idx >= molecule.atoms.len() {
                return Err(DftError::InvalidInput(format!(
                    "basis override names atom index {idx}, but the molecule has only {} atoms",
                    molecule.atoms.len()
                )));
            }
        }
        let mut shells = Vec::new();
        let mut sph_off = 0usize;
        let mut cart_off = 0usize;
        let mut max_l = 0u8;
        let mut ecps = Vec::new();
        let mut n_core_electrons = 0u32;
        for (atom_idx, atom) in molecule.atoms.iter().enumerate() {
            if atom.z == 0 {
                return Err(DftError::InvalidInput(
                    "Z=0 does not identify a ghost basis centre; ghost atoms require an explicit element identity and are not supported by Molecule"
                        .to_string(),
                ));
            }
            if let Some(pot) = basis.ecp(atom.z) {
                if pot.n_core > atom.z as u32 {
                    return Err(DftError::InvalidInput(format!(
                        "ECP on Z={} replaces {} electrons, more than the atom contains",
                        atom.z, pot.n_core
                    )));
                }
                n_core_electrons += pot.n_core;
                ecps.push(AtomEcp {
                    atom: atom_idx,
                    center: atom.position,
                    potential: pot.clone(),
                });
            }
            let eb = match overrides.iter().find(|(idx, _)| *idx == atom_idx) {
                Some((_, eb)) => eb,
                None => basis.element(atom.z)?,
            };
            for shell in &eb.shells {
                let coefs = normalized_contraction(shell)?;
                max_l = max_l.max(shell.l);
                shells.push(MolShell {
                    l: shell.l,
                    atom: atom_idx,
                    center: atom.position,
                    exps: shell.exps.clone(),
                    coefs,
                    sph_off,
                    cart_off,
                });
                sph_off += n_sph(shell.l);
                cart_off += n_cart(shell.l);
            }
        }
        Ok(Self {
            shells,
            nao_sph: sph_off,
            nao_cart: cart_off,
            max_l,
            ecps,
            n_core_electrons,
        })
    }
}

/// Fold primitive norms into the contraction coefficients and normalize the
/// contracted `(l,0,0)` component to unit self-overlap.
fn normalized_contraction(shell: &Shell) -> Result<Vec<f64>> {
    if shell.exps.len() != shell.coefs.len() || shell.exps.is_empty() {
        return Err(DftError::InvalidInput(format!(
            "shell l={} has {} exponents but {} coefficients",
            shell.l,
            shell.exps.len(),
            shell.coefs.len()
        )));
    }
    if shell.exps.iter().any(|a| !a.is_finite() || *a <= 0.0) {
        return Err(DftError::InvalidInput(format!(
            "shell l={} contains a non-finite or non-positive Gaussian exponent",
            shell.l
        )));
    }
    if shell.coefs.iter().any(|c| !c.is_finite()) {
        return Err(DftError::InvalidInput(format!(
            "shell l={} contains a non-finite contraction coefficient",
            shell.l
        )));
    }
    let l = shell.l as i32;
    let lref = (l, 0, 0);
    let mut coefs: Vec<f64> = shell
        .exps
        .iter()
        .zip(&shell.coefs)
        .map(|(&a, &c)| c * primitive_norm(a, lref))
        .collect();
    // Contracted self-overlap of the (l,0,0) component on one center:
    // S = Σ_ij c_i c_j (2l−1)!! / (2p)^l · (π/p)^{3/2}, p = a_i + a_j.
    let mut s = 0.0;
    for (&ai, &ci) in shell.exps.iter().zip(&coefs) {
        for (&aj, &cj) in shell.exps.iter().zip(&coefs) {
            let p = ai + aj;
            s += ci * cj * double_factorial(2 * l - 1) / (2.0 * p).powi(l)
                * (std::f64::consts::PI / p).powf(1.5);
        }
    }
    if !(s.is_finite() && s > 0.0) {
        return Err(DftError::InvalidInput(format!(
            "shell l={} has a non-positive contracted self-overlap {s:.3e}",
            shell.l
        )));
    }
    let inv = 1.0 / s.sqrt();
    for c in &mut coefs {
        *c *= inv;
    }
    Ok(coefs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    #[test]
    fn cart_component_counts_and_order() {
        assert_eq!(cart_components(0), vec![(0, 0, 0)]);
        assert_eq!(cart_components(1), vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]);
        assert_eq!(
            cart_components(2),
            vec![
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 2)
            ]
        );
        assert_eq!(n_cart(3), 10);
        assert_eq!(n_sph(3), 7);
    }

    #[test]
    fn primitive_norm_normalizes_self_overlap() {
        // ⟨g|g⟩ = 1 for a normalized primitive: check s, p, and d_xy cases
        // against the analytic self-overlap of the raw Gaussian.
        for &(a, lxyz) in &[
            (0.8_f64, (0, 0, 0)),
            (1.7, (1, 0, 0)),
            (2.3, (1, 1, 0)),
            (0.5, (2, 0, 0)),
        ] {
            let n = primitive_norm(a, lxyz);
            let (lx, ly, lz) = lxyz;
            let p = 2.0 * a;
            let raw = double_factorial(2 * lx - 1)
                * double_factorial(2 * ly - 1)
                * double_factorial(2 * lz - 1)
                / (2.0 * p).powi(lx + ly + lz)
                * (std::f64::consts::PI / p).powf(1.5);
            assert!(
                (n * n * raw - 1.0).abs() < 1e-13,
                "norm broken for {lxyz:?}"
            );
        }
    }

    #[test]
    fn contraction_normalization_sto3g_h() {
        // STO-3G hydrogen (the standard published exponents/coefficients).
        let shell = Shell {
            l: 0,
            exps: vec![3.425250914, 0.6239137298, 0.1688554040],
            coefs: vec![0.1543289673, 0.5353281423, 0.4446345422],
        };
        let coefs = normalized_contraction(&shell).unwrap();
        // Self-overlap of the contracted normalized function must be 1.
        let mut s = 0.0;
        for (&ai, &ci) in shell.exps.iter().zip(&coefs) {
            for (&aj, &cj) in shell.exps.iter().zip(&coefs) {
                let p = ai + aj;
                s += ci * cj * (std::f64::consts::PI / p).powf(1.5);
            }
        }
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mol_basis_layout_offsets() {
        let mut bs = BasisSet::new("test");
        bs.insert(ElementBasis {
            z: 8,
            shells: vec![
                Shell {
                    l: 0,
                    exps: vec![1.0],
                    coefs: vec![1.0],
                },
                Shell {
                    l: 1,
                    exps: vec![0.8],
                    coefs: vec![1.0],
                },
            ],
        });
        bs.insert(ElementBasis {
            z: 1,
            shells: vec![Shell {
                l: 0,
                exps: vec![1.2],
                coefs: vec![1.0],
            }],
        });
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 1.8),
            },
        ]);
        let mb = MolBasis::build(&mol, &bs).unwrap();
        assert_eq!(mb.shells.len(), 3);
        assert_eq!(mb.nao_sph, 1 + 3 + 1);
        assert_eq!(mb.nao_cart, 1 + 3 + 1);
        assert_eq!(mb.shells[1].sph_off, 1);
        assert_eq!(mb.shells[2].sph_off, 4);
        assert_eq!(mb.max_l, 1);
        // Missing element errors cleanly.
        let bad = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::zero(),
        }]);
        assert!(MolBasis::build(&bad, &bs).is_err());
    }

    #[test]
    fn build_with_overrides_empty_matches_build() {
        // An empty override list must be exactly build()'s code path: this
        // is what lets `build` stay a thin wrapper with no behavior change.
        let mut bs = BasisSet::new("test");
        bs.insert(ElementBasis {
            z: 1,
            shells: vec![Shell {
                l: 0,
                exps: vec![1.2, 0.4],
                coefs: vec![0.6, 0.5],
            }],
        });
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.7),
        }]);
        let a = MolBasis::build(&mol, &bs).unwrap();
        let b = MolBasis::build_with_overrides(&mol, &bs, &[]).unwrap();
        assert_eq!(a.nao_sph, b.nao_sph);
        assert_eq!(a.shells.len(), b.shells.len());
        for (sa, sb) in a.shells.iter().zip(&b.shells) {
            assert_eq!(sa.coefs, sb.coefs);
            assert_eq!(sa.exps, sb.exps);
        }
    }

    #[test]
    fn build_with_overrides_swaps_only_the_named_atom() {
        // Two atoms of the same element (Z=1) get different bases: the
        // override applies to atom index 1 only, atom 0 keeps the set's
        // ordinary Z=1 entry.
        let mut bs = BasisSet::new("test");
        bs.insert(ElementBasis {
            z: 1,
            shells: vec![Shell {
                l: 0,
                exps: vec![1.0],
                coefs: vec![1.0],
            }],
        });
        let big = ElementBasis {
            z: 1,
            shells: vec![
                Shell {
                    l: 0,
                    exps: vec![1.0],
                    coefs: vec![1.0],
                },
                Shell {
                    l: 1,
                    exps: vec![0.8],
                    coefs: vec![1.0],
                },
            ],
        };
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 1.4),
            },
        ]);
        let mb = MolBasis::build_with_overrides(&mol, &bs, &[(1, big)]).unwrap();
        assert_eq!(mb.shells.len(), 3); // atom 0: 1 shell, atom 1: 2 shells
        assert_eq!(mb.nao_sph, 1 + (1 + 3));
        assert_eq!(mb.shells[0].atom, 0);
        assert_eq!(mb.shells[1].atom, 1);
        assert_eq!(mb.shells[2].atom, 1);

        // An override naming an out-of-range atom index is refused.
        let bad = ElementBasis {
            z: 1,
            shells: vec![Shell {
                l: 0,
                exps: vec![1.0],
                coefs: vec![1.0],
            }],
        };
        assert!(MolBasis::build_with_overrides(&mol, &bs, &[(5, bad)]).is_err());
    }
}
