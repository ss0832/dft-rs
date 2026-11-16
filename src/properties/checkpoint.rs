// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Machine-readable wavefunction checkpoint (debug & research use).
//!
//! A deliberately simple, self-describing text format (no serde): named
//! sections, one datum per line, full f64 precision (`{:.17e}` round-trips
//! exactly). Everything needed to re-analyze the state outside dft-rs —
//! geometry, basis layout, MO coefficients/energies/occupations, spin
//! densities, S, H_core, converged Fock matrices — in the canonical dft-rs
//! AO ordering (see docs/THEORY.md).

use crate::basis::MolBasis;
use crate::linalg::Matrix;
use crate::scf::ScfResult;
use crate::system::Molecule;
use std::fmt::Write as _;

fn push_matrix(out: &mut String, name: &str, m: &Matrix) {
    let _ = writeln!(out, "[{name}] rows={} cols={}", m.rows, m.cols);
    for i in 0..m.rows {
        for j in 0..m.cols {
            let _ = writeln!(out, "{i} {j} {:.17e}", m[(i, j)]);
        }
    }
}

/// Serialize the full SCF state.
pub fn wavefunction_text(mol: &Molecule, mb: &MolBasis, scf: &ScfResult) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "dft-rs wavefunction checkpoint v1");
    let _ = writeln!(
        out,
        "[system] natoms={} charge={} multiplicity={}",
        mol.len(),
        mol.charge,
        mol.multiplicity
    );
    for (i, a) in mol.atoms.iter().enumerate() {
        let _ = writeln!(
            out,
            "atom {i} z={} x={:.17e} y={:.17e} z_coord={:.17e}",
            a.z, a.position.x, a.position.y, a.position.z
        );
    }
    let _ = writeln!(
        out,
        "[scf] method={} energy={:.17e} e_elec={:.17e} e_nuc={:.17e} n_alpha={} n_beta={} s_squared={:.17e} converged={} iterations={}",
        if scf.unrestricted { "UHF" } else { "RHF" },
        scf.energy,
        scf.e_elec,
        scf.e_nuc,
        scf.n_alpha,
        scf.n_beta,
        scf.s_squared,
        scf.converged,
        scf.iterations
    );
    let _ = writeln!(
        out,
        "[basis] nao_sph={} nshells={} ao_order=m:0,+1,-1,... (docs/THEORY.md)",
        mb.nao_sph,
        mb.shells.len()
    );
    for (i, sh) in mb.shells.iter().enumerate() {
        let _ = writeln!(
            out,
            "shell {i} atom={} l={} nprim={} sph_off={}",
            sh.atom,
            sh.l,
            sh.exps.len(),
            sh.sph_off
        );
    }
    for spin in 0..2 {
        let tag = ["alpha", "beta"][spin];
        let _ = writeln!(out, "[mo_energy_{tag}] n={}", scf.mo_energy[spin].len());
        for (i, e) in scf.mo_energy[spin].iter().enumerate() {
            let _ = writeln!(out, "{i} {e:.17e}");
        }
        push_matrix(&mut out, &format!("mo_coeff_{tag}"), &scf.mo_coeff[spin]);
        push_matrix(&mut out, &format!("density_{tag}"), &scf.density[spin]);
        push_matrix(&mut out, &format!("fock_{tag}"), &scf.fock[spin]);
    }
    push_matrix(&mut out, "overlap", &scf.overlap);
    push_matrix(&mut out, "h_core", &scf.h_core);
    out
}

/// Write the checkpoint to a file.
pub fn dump_wavefunction(
    path: &std::path::Path,
    mol: &Molecule,
    mb: &MolBasis,
    scf: &ScfResult,
) -> crate::error::Result<()> {
    std::fs::write(path, wavefunction_text(mol, mb, scf))?;
    Ok(())
}
