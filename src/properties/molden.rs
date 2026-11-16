// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molden file writer: geometry, basis ([GTO]) and molecular orbitals for
//! visualization tools (Molden, Avogadro, IboView, …) and for the PySCF
//! round-trip verification.
//!
//! Conventions: spherical functions ([5D]/[7F]/[9G] tags). Molden's
//! within-shell order for l ≥ 2 (m = 0, +1, −1, …) is exactly the dft-rs
//! canonical order; p shells are emitted in Molden's (px, py, pz) = our
//! (m=+1, m=−1, m=0) order.

use crate::basis::{n_sph, BasisSet, MolBasis};
use crate::error::Result;
use crate::scf::ScfResult;
use crate::system::{z_to_symbol, Molecule};
use std::fmt::Write as _;

const SHELL_LABEL: [&str; 7] = ["s", "p", "d", "f", "g", "h", "i"];

/// Serialize the SCF solution in Molden format.
pub fn molden_text(mol: &Molecule, set: &BasisSet, mb: &MolBasis, scf: &ScfResult) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "[Molden Format]");
    let _ = writeln!(out, "[Title]");
    let _ = writeln!(
        out,
        "dft-rs {} | E = {:.12} Hartree",
        env!("CARGO_PKG_VERSION"),
        scf.energy
    );
    let _ = writeln!(out, "[Atoms] (AU)");
    for (i, atom) in mol.atoms.iter().enumerate() {
        let _ = writeln!(
            out,
            "{:<3} {:>4} {:>4} {:>18.10} {:>18.10} {:>18.10}",
            z_to_symbol(atom.z).unwrap_or("X"),
            i + 1,
            atom.z,
            atom.position.x,
            atom.position.y,
            atom.position.z
        );
    }
    let _ = writeln!(out, "[GTO]");
    for (i, atom) in mol.atoms.iter().enumerate() {
        let _ = writeln!(out, "{:>4} 0", i + 1);
        if let Ok(eb) = set.element(atom.z) {
            for shell in &eb.shells {
                let _ = writeln!(
                    out,
                    " {}   {:>2} 1.00",
                    SHELL_LABEL[shell.l as usize],
                    shell.exps.len()
                );
                for (e, c) in shell.exps.iter().zip(&shell.coefs) {
                    let _ = writeln!(out, "    {:>18.10e} {:>18.10e}", e, c);
                }
            }
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out, "[5D]");
    let _ = writeln!(out, "[7F]");
    let _ = writeln!(out, "[9G]");
    let _ = writeln!(out, "[MO]");
    let spins: &[(&str, usize, f64)] = if scf.unrestricted {
        &[("Alpha", 0, 1.0), ("Beta", 1, 1.0)]
    } else {
        &[("Alpha", 0, 2.0)]
    };
    // Molden AO order: permute our canonical p blocks (0,+1,−1) → (x,y,z).
    let perm = molden_ao_order(mb);
    for &(label, spin, full_occ) in spins {
        let c = &scf.mo_coeff[spin];
        for mo in 0..c.cols {
            let _ = writeln!(out, " Sym= A");
            let _ = writeln!(out, " Ene= {:>18.10}", scf.mo_energy[spin][mo]);
            let _ = writeln!(out, " Spin= {label}");
            // Fractional (Fermi-Dirac) occupations are written as they are.
            let occ = full_occ * scf.occupations[spin].get(mo).copied().unwrap_or(0.0);
            let _ = writeln!(out, " Occup= {occ:.6}");
            for (row, &mu) in perm.iter().enumerate() {
                let _ = writeln!(out, "{:>5} {:>18.10e}", row + 1, c[(mu, mo)]);
            }
        }
    }
    out
}

/// AO index permutation: Molden row r ← dft-rs AO `perm[r]`.
fn molden_ao_order(mb: &MolBasis) -> Vec<usize> {
    let mut perm = Vec::with_capacity(mb.nao_sph);
    for shell in &mb.shells {
        let off = shell.sph_off;
        if shell.l == 1 {
            // (px, py, pz) = our (m=+1, m=−1, m=0) = offsets (1, 2, 0).
            perm.extend([off + 1, off + 2, off]);
        } else {
            perm.extend((0..n_sph(shell.l)).map(|k| off + k));
        }
    }
    perm
}

pub fn write_molden(
    path: &std::path::Path,
    mol: &Molecule,
    set: &BasisSet,
    mb: &MolBasis,
    scf: &ScfResult,
) -> Result<()> {
    std::fs::write(path, molden_text(mol, set, mb, scf))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::math::Vec3;
    use crate::scf::{run_hf, EriMode, ScfOptions};
    use crate::system::Atom;

    #[test]
    fn molden_structure_and_ao_count() {
        let set = data::load("def2-svp").unwrap();
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
        let mb = MolBasis::build(&mol, &set).unwrap();
        let opts = ScfOptions {
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..Default::default()
        };
        let scf = run_hf(&mol, &mb, &opts).unwrap();
        let text = molden_text(&mol, &set, &mb, &scf);
        assert!(text.contains("[Molden Format]"));
        assert!(text.contains("[5D]"));
        // One MO block per MO, each with nao coefficient rows.
        let n_mo = scf.mo_coeff[0].cols;
        assert_eq!(text.matches(" Sym= A").count(), n_mo);
        assert_eq!(
            text.matches(" Occup= 2.000000").count(),
            scf.n_alpha,
            "occupied count"
        );
        // The permutation is a bijection.
        let mut perm = molden_ao_order(&mb);
        perm.sort_unstable();
        assert_eq!(perm, (0..mb.nao_sph).collect::<Vec<_>>());
    }
}
