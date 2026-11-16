// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Ground-state properties from an SCF solution: population analyses,
//! dipole moment, the MO table printer, the wavefunction checkpoint, and
//! the GIAO magnetic layer ([`nmr`] operators, [`giao_eri`] field
//! derivatives, [`magnetic`] response and NMR shieldings).

pub mod checkpoint;
pub mod correlated_axial;
pub mod correlated_magnetic;
pub mod correlated_spin_spin;
pub mod epr;
pub mod franck_condon;
pub mod g_mass_correction;
pub mod g_nuclear;
pub mod giao_eri;
pub(crate) mod giao_xc;
pub mod hyperfine;
pub mod igmh;
pub mod magnetic;
pub mod molden;
pub mod natural_bond;
pub mod natural_transition_orbitals;
pub mod nmr;
pub mod no_pair;
pub mod nuclear_spin;
pub mod pairing;
pub mod pt2;
pub mod pt2_axial;
pub mod pt2_density;
pub mod pt2_response;
pub(crate) mod pt2_transition;
pub use pt2_transition::{collinear_transition_path, CollinearTransitionPath};
/// Experimental constrained HF two-electron-triplet Pauli anisotropy; not the public direct-spin-spin route.
pub mod soc_zero_field_splitting;
pub mod spin_diagnostics;
pub mod thermochemistry;
pub mod two_component;
pub mod vibrational;
pub mod zero_field_splitting;

use crate::basis::MolBasis;
use crate::constants::AU_DIPOLE_TO_DEBYE;
use crate::error::Result;
use crate::integrals::dipole_matrices;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::scf::ScfResult;
use crate::system::Molecule;

/// Mulliken atomic charges: `q_A = Z_A − Σ_{μ∈A} (P S)_μμ`.
pub fn mulliken_charges(mol: &Molecule, mb: &MolBasis, scf: &ScfResult) -> Vec<f64> {
    let ps = scf.p_total().matmul(&scf.overlap);
    let owner = mb.sph_ao_atoms();
    // Population analysis must use the same ionic charge as the
    // Hamiltonian. With an ECP, the explicit density contains valence
    // electrons only, so subtracting it from the bare Z would leave the
    // replaced core charge spuriously on every atom.
    let mut q = mb.effective_charges(mol);
    for (mu, &atom) in owner.iter().enumerate() {
        q[atom] -= ps[(mu, mu)];
    }
    q
}

/// Löwdin atomic charges: `q_A = Z_A − Σ_{μ∈A} (S^{1/2} P S^{1/2})_μμ`.
pub fn lowdin_charges(mol: &Molecule, mb: &MolBasis, scf: &ScfResult) -> Result<Vec<f64>> {
    lowdin_density_charges(mol, mb, &scf.p_total(), &scf.overlap)
}

/// Loewdin charges from the real total spatial density and AO metric.
/// A complex spinor reference supplies the real charge Pauli component;
/// imaginary antisymmetric entries do not contribute with real scalar AOs.
pub fn lowdin_density_charges(
    mol: &Molecule,
    mb: &MolBasis,
    density: &Matrix,
    overlap: &Matrix,
) -> Result<Vec<f64>> {
    let (vals, vecs) = symmetric_eigen(overlap)?;
    let n = vals.len();
    // S^{1/2} = V diag(√λ) Vᵀ.
    let mut half = Matrix::zeros(n, n);
    for k in 0..n {
        let sq = vals[k].max(0.0).sqrt();
        for i in 0..n {
            for j in 0..n {
                half[(i, j)] += vecs[(i, k)] * sq * vecs[(j, k)];
            }
        }
    }
    let sps = half.matmul(&density.matmul(&half));
    let owner = mb.sph_ao_atoms();
    let mut q = mb.effective_charges(mol);
    for (mu, &atom) in owner.iter().enumerate() {
        q[atom] -= sps[(mu, mu)];
    }
    Ok(q)
}

/// Total dipole moment (a.u.): `μ = Σ_A Z_A R_A − Tr(P·r)` about the origin.
pub fn dipole_moment(mol: &Molecule, mb: &MolBasis, scf: &ScfResult) -> Vec3 {
    let dips = dipole_matrices(mb, Vec3::zero());
    let p = scf.p_total();
    let mut mu = Vec3::zero();
    let z_eff = mb.effective_charges(mol);
    for (atom, z) in mol.atoms.iter().zip(&z_eff) {
        mu += atom.position * *z;
    }
    mu.x -= p.frobenius_dot(&dips[0]);
    mu.y -= p.frobenius_dot(&dips[1]);
    mu.z -= p.frobenius_dot(&dips[2]);
    mu
}

/// Dipole in Debye.
pub fn dipole_debye(mol: &Molecule, mb: &MolBasis, scf: &ScfResult) -> Vec3 {
    dipole_moment(mol, mb, scf) * AU_DIPOLE_TO_DEBYE
}

/// Print the MO table (energies, occupations; both spins when unrestricted).
pub fn print_mo_table(scf: &ScfResult) {
    let spins: &[(&str, usize)] = if scf.unrestricted {
        &[("alpha", 0), ("beta", 1)]
    } else {
        &[("", 0)]
    };
    for &(label, spin) in spins {
        let n_occ = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        if scf.unrestricted {
            println!("# {label} MOs");
        } else {
            println!("# MOs (doubly occupied up to HOMO)");
        }
        println!("# idx   occ      energy (Hartree)      energy (eV)");
        for (i, &e) in scf.mo_energy[spin].iter().enumerate() {
            let occ = if i < n_occ {
                if scf.unrestricted {
                    1.0
                } else {
                    2.0
                }
            } else {
                0.0
            };
            println!(
                "{:>5} {:>5.1} {:>20.10} {:>16.6}",
                i + 1,
                occ,
                e,
                e * crate::constants::HARTREE_TO_EV
            );
        }
    }
}
