// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! SAD: the superposition-of-atomic-densities initial guess.
//!
//! The molecular density is approximated by placing a converged,
//! spherically averaged ATOM on each centre and adding the blocks. It is a
//! far better starting point than a one-electron guess because it already
//! has the right number of electrons in the right shells around every
//! nucleus; what the SCF then has to find is only the bonding
//! rearrangement.
//!
//! The atomic densities are computed ON THE FLY in the very basis the
//! molecule uses, rather than read from a table. That costs a few
//! milliseconds per distinct element and buys two things a shipped table
//! cannot: the guess is automatically consistent with any basis, including
//! a user's own `.g94` file and any per-element mixture, and there is no
//! data file whose provenance has to be tracked.
//!
//! # Spherical averaging
//!
//! A bare atomic SCF would converge to a symmetry-broken density (carbon
//! putting its two p electrons in specific p orbitals), which would make
//! the molecular guess depend on the arbitrary orientation of the axes.
//! Instead the occupations are spread EQUALLY over each degenerate shell:
//! carbon's four p slots share 2 electrons at 1/2 each, so the density is
//! spherically symmetric by construction.
//!
//! Degenerate shells are found from the spectrum rather than from a table
//! of electron configurations: for a free atom the Fock matrix commutes
//! with the angular momentum, so its eigenvalues come in exactly
//! `2l+1`-fold degenerate groups. Grouping by near-equality therefore
//! recovers the shell structure without any element data — and it stays
//! right for the ECP case, where the shells present depend on the
//! potential rather than on the neutral configuration.
//!
//! The atomic problem is solved at the HF level regardless of the
//! molecular functional. A guess does not need the functional; it needs to
//! be cheap, robust, and grid-free.

use crate::basis::{n_cart, n_sph, AtomEcp, MolBasis, MolShell};
use crate::error::Result;
use crate::linalg::{Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::scf::fock::JkEngine;
use crate::scf::EriMode;
use crate::system::{Atom, Molecule};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Lift one centre's shells out of the molecular basis into a standalone
/// single-atom basis at the origin, and report the molecular AO offsets
/// they correspond to.
///
/// Taking the shells from the molecular basis rather than re-reading a
/// `BasisSet` is what lets this work for ANY basis the molecule was built
/// with, including a per-element mixture and an independently assigned
/// ECP — there is no second lookup that could disagree with the first.
fn atom_basis(mb: &MolBasis, atom_idx: usize) -> (MolBasis, Vec<usize>) {
    let mut shells = Vec::new();
    let mut offsets = Vec::new();
    let (mut so, mut co) = (0usize, 0usize);
    let mut max_l = 0u8;
    for sh in mb.shells.iter().filter(|s| s.atom == atom_idx) {
        offsets.extend(sh.sph_off..sh.sph_off + n_sph(sh.l));
        shells.push(MolShell {
            l: sh.l,
            atom: 0,
            center: Vec3::zero(),
            exps: sh.exps.clone(),
            coefs: sh.coefs.clone(),
            sph_off: so,
            cart_off: co,
        });
        so += n_sph(sh.l);
        co += n_cart(sh.l);
        max_l = max_l.max(sh.l);
    }
    let ecps: Vec<AtomEcp> = mb
        .ecps
        .iter()
        .filter(|e| e.atom == atom_idx)
        .map(|e| AtomEcp {
            atom: 0,
            center: Vec3::zero(),
            potential: e.potential.clone(),
        })
        .collect();
    let n_core_electrons = ecps.iter().map(|e| e.potential.n_core).sum();
    (
        MolBasis {
            shells,
            nao_sph: so,
            nao_cart: co,
            max_l,
            ecps,
            n_core_electrons,
        },
        offsets,
    )
}

/// Converged spherically averaged atomic density in the given single-atom
/// basis.
fn atomic_density(z: u8, mb: &MolBasis) -> Result<Matrix> {
    let mol = Molecule::new(vec![Atom {
        z,
        position: Vec3::zero(),
    }]);
    let n = mb.nao_sph;
    let s = crate::integrals::overlap_matrix(mb);
    let mut h = crate::integrals::kinetic_matrix(mb);
    let v = crate::integrals::nuclear_matrix(mb, &mol, Some(&[z as f64]));
    for (hv, vv) in h.as_mut_slice().iter_mut().zip(v.as_slice()) {
        *hv += *vv;
    }
    if !mb.ecps.is_empty() {
        let u = crate::integrals::ecp::ecp_matrix(mb)?;
        for (hv, uv) in h.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *hv += *uv;
        }
    }
    // Electrons actually described: an ECP removes its core.
    let n_elec = (z as i64) - (mb.n_core_electrons as i64);
    if n_elec <= 0 {
        return Ok(Matrix::zeros(n, n));
    }
    let n_elec = n_elec as f64;

    let orth = Orthogonalizer::canonical(&s, 1e-7)?;
    // Plain Coulomb regardless of the molecular kernel: this is a guess,
    // and a range-separated atomic SCF would cost more without making the
    // starting density any better.
    let jk = JkEngine::new(
        mb,
        EriMode::InCore { limit_mb: 256 },
        crate::integrals::CoulombKernel::Coulomb,
    )?;

    let mut p = Matrix::zeros(n, n);
    // The DAMPED density drives the iteration; the density actually
    // returned is the last undamped one. A damped average of iterates only
    // approaches the correct electron count geometrically, so returning it
    // would leave the guess a few parts in 10^6 short of neutral — small,
    // but a guess whose electron count is not exact is a needless puzzle
    // for anything downstream that checks.
    let mut p_exact = Matrix::zeros(n, n);
    let mut f = h.clone();
    // A free atom with fractional occupations is a small, benign SCF, but
    // it is also the one place a guess must never hang: heavy damping and
    // a hard iteration cap keep it bounded, and an unconverged atom is
    // still a usable guess.
    const MAX_ITER: usize = 60;
    const MIX: f64 = 0.4;
    for _ in 0..MAX_ITER {
        let (eps, c) = orth.solve_roothaan(&f)?;
        let occ = spherical_occupations(&eps, n_elec);
        // P_total = sum_p n_p C_p C_p^T; the two spin channels are equal
        // by construction, so P_spin = P_total / 2.
        let mut p_new = Matrix::zeros(n, n);
        for (col, &nocc) in occ.iter().enumerate() {
            if nocc <= 0.0 {
                continue;
            }
            for mu in 0..n {
                let cm = c[(mu, col)];
                if cm == 0.0 {
                    continue;
                }
                for nu in 0..n {
                    p_new[(mu, nu)] += nocc * cm * c[(nu, col)];
                }
            }
        }
        let delta = p_new.max_abs_diff(&p);
        for (pv, nv) in p.as_mut_slice().iter_mut().zip(p_new.as_slice()) {
            *pv = MIX * *nv + (1.0 - MIX) * *pv;
        }
        p_exact = p_new;
        if delta < 1e-6 {
            break;
        }
        let mut p_half = p.clone();
        for v in p_half.as_mut_slice().iter_mut() {
            *v *= 0.5;
        }
        let (j, k) = jk.build(mb, &p, [&p_half, &p_half], true)?;
        f = h.clone();
        for (fv, (jv, kv)) in f
            .as_mut_slice()
            .iter_mut()
            .zip(j.as_slice().iter().zip(k[0].as_slice()))
        {
            *fv += *jv - *kv;
        }
    }
    Ok(p_exact)
}

/// Spread `n_elec` electrons over the orbitals in energy order, giving
/// every member of a degenerate shell the same occupancy.
///
/// Equal occupancy within a shell is what makes the density spherical, and
/// hence what makes the molecular guess independent of how the input
/// happens to be oriented.
fn spherical_occupations(eps: &[f64], n_elec: f64) -> Vec<f64> {
    let mut occ = vec![0.0; eps.len()];
    let mut left = n_elec;
    let mut i = 0;
    // Degeneracy tolerance: atomic shells are exactly degenerate, so any
    // small absolute window separates shells cleanly. Scaling with the
    // orbital energy keeps deep core shells (tens of Hartree) grouped too.
    while i < eps.len() && left > 1e-12 {
        let mut j = i + 1;
        while j < eps.len() && (eps[j] - eps[i]).abs() <= 1e-6 * eps[i].abs().max(1.0) {
            j += 1;
        }
        let size = (j - i) as f64;
        let take = left.min(2.0 * size);
        let per = take / size;
        for o in occ.iter_mut().take(j).skip(i) {
            *o = per;
        }
        left -= take;
        i = j;
    }
    occ
}

/// Cache: one converged atomic density per (element, exact shell layout).
///
/// The key carries the shells themselves, not a basis NAME, so a
/// per-element mixture or a user `.g94` reusing a familiar name cannot
/// collide with a bundled set — the cached density always belongs to the
/// basis it is about to be used with.
fn cache() -> &'static Mutex<HashMap<String, Matrix>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Matrix>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn fingerprint(z: u8, mb: &MolBasis) -> String {
    let mut s = format!("Z{z}|core{}|", mb.n_core_electrons);
    for sh in &mb.shells {
        s.push_str(&format!("{}:", sh.l));
        for (e, c) in sh.exps.iter().zip(&sh.coefs) {
            s.push_str(&format!("{e:.9e}/{c:.9e},"));
        }
        s.push(';');
    }
    s
}

/// Molecular SAD density: the block-diagonal superposition of converged
/// spherically averaged atomic densities.
///
/// Returns the TOTAL density; the caller halves it for the spin channels
/// (the atoms are spin-averaged, so alpha and beta are equal).
pub fn sad_density(mol: &Molecule, mb: &MolBasis) -> Result<Matrix> {
    let n = mb.nao_sph;
    let mut p = Matrix::zeros(n, n);
    for (idx, atom) in mol.atoms.iter().enumerate() {
        if atom.z == 0 {
            continue; // ghost centre: basis functions, no electrons
        }
        let (amb, offsets) = atom_basis(mb, idx);
        if amb.nao_sph == 0 {
            continue;
        }
        let key = fingerprint(atom.z, &amb);
        let hit = cache().lock().ok().and_then(|c| c.get(&key).cloned());
        let block = match hit {
            Some(m) => m,
            None => {
                let m = atomic_density(atom.z, &amb)?;
                if let Ok(mut c) = cache().lock() {
                    c.insert(key, m.clone());
                }
                m
            }
        };
        for (i, &mu) in offsets.iter().enumerate() {
            for (j, &nu) in offsets.iter().enumerate() {
                p[(mu, nu)] += block[(i, j)];
            }
        }
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;

    /// The atomic density must hold the right number of electrons: that is
    /// the whole point of a superposition guess, and it is what a
    /// one-electron guess gets wrong.
    #[test]
    fn atomic_densities_have_the_right_electron_count() {
        let set = data::load("6-31g").unwrap();
        for z in [1u8, 6, 8, 9] {
            let mol = Molecule::new(vec![Atom {
                z,
                position: Vec3::zero(),
            }]);
            let mb = MolBasis::build(&mol, &set).unwrap();
            let s = crate::integrals::overlap_matrix(&mb);
            let p = sad_density(&mol, &mb).unwrap();
            let n: f64 = p.frobenius_dot(&s);
            assert!(
                (n - z as f64).abs() < 1e-6,
                "Z = {z}: guess holds {n} electrons"
            );
        }
    }

    /// An ECP basis must place only the VALENCE electrons: the core is
    /// inside the potential, and a guess that put them back would start
    /// the SCF with the wrong electron count entirely.
    #[test]
    fn ecp_atoms_hold_only_their_valence_electrons() {
        let set = data::load("lanl2dz").unwrap();
        let mol = Molecule::new(vec![Atom {
            z: 28,
            position: Vec3::zero(),
        }]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let s = crate::integrals::overlap_matrix(&mb);
        let p = sad_density(&mol, &mb).unwrap();
        let n: f64 = p.frobenius_dot(&s);
        let want = 28.0 - mb.n_core_electrons as f64;
        assert!(
            (n - want).abs() < 1e-6,
            "Ni/LANL2DZ guess holds {n} electrons, expected {want}"
        );
    }

    /// Equal occupancy across a degenerate shell is what makes the density
    /// spherical. Check it directly: the three p-block diagonal entries of
    /// a carbon atom must be equal, so the guess cannot depend on how the
    /// molecule happens to be oriented in the input file.
    #[test]
    fn open_shell_atoms_come_out_spherical() {
        let set = data::load("6-31g").unwrap();
        let mol = Molecule::new(vec![Atom {
            z: 6,
            position: Vec3::zero(),
        }]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let p = sad_density(&mol, &mb).unwrap();
        for sh in mb.shells.iter().filter(|s| s.l == 1) {
            let o = sh.sph_off;
            let (x, y, z) = (p[(o, o)], p[(o + 1, o + 1)], p[(o + 2, o + 2)]);
            let spread = (x - y).abs().max((y - z).abs()).max((x - z).abs());
            assert!(
                spread < 1e-8,
                "carbon p-shell at offset {o} is not spherical: {x} {y} {z}"
            );
        }
    }

    /// A ghost centre contributes basis functions but no electrons — the
    /// property that makes counterpoise correction exact.
    #[test]
    fn ghost_centres_contribute_no_electrons() {
        let set = data::load("sto-3g").unwrap();
        // The basis is built from the REAL molecule and the ghosting is
        // expressed by zeroing the nuclear charges — the same arrangement
        // the counterpoise correction uses, so the AO space is identical
        // between the dimer and each monomer.
        let real = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::zero(),
            },
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 3.0),
            },
        ]);
        let mb = MolBasis::build(&real, &set).unwrap();
        let mut ghosted = real.clone();
        ghosted.atoms[1].z = 0;
        let s = crate::integrals::overlap_matrix(&mb);
        let p = sad_density(&ghosted, &mb).unwrap();
        let n: f64 = p.frobenius_dot(&s);
        assert!((n - 8.0).abs() < 1e-6, "ghost added electrons: {n}");
    }
}
