// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Natural bond orbital analysis of a periodic reference on the
//! Born-von Karman supercell of its k mesh.
//!
//! A calculation on the unshifted `n1 x n2 x n3` mesh is the calculation of
//! the supercell of `n1 n2 n3` cells with periodic boundary conditions. Its
//! real AO matrices follow from the Bloch matrices,
//! `X[(L mu),(L' nu)] = 1/N_k sum_k X(k)_mu,nu exp(-i k.(R_L' - R_L))`, for
//! the overlap, the spin densities and the Fock matrices, and the molecular
//! analysis of these matrices is the analysis of the crystal: translations
//! by a lattice vector permute the cells, so every cell carries the same
//! natural charges and orbitals. Bonds to periodic images appear between
//! atoms of different cells; along an axis with fewer than three k points
//! an atom and its image in the next cell can coincide in the supercell.

use super::{
    ecp_core_electrons, natural_bond_from_density, shell_layout, NaturalBondAnalysis, NboOptions,
    NboSpin, ShellLayout,
};
use crate::basis::MolBasis;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::scf::PeriodicScfResult;
use crate::system::Molecule;
use crate::{DftError, Result};

/// Natural bond orbital analysis of the Born-von Karman supercell.
#[derive(Clone, Debug)]
pub struct PeriodicNaturalBond {
    pub analysis: NaturalBondAnalysis,
    /// Cell indices of the supercell; atom `a` of cell `c` is supercell atom
    /// `c * atoms_per_cell + a`.
    pub cells: Vec<[usize; 3]>,
    pub mesh: [usize; 3],
    pub atoms_per_cell: usize,
    /// Largest imaginary part of the unfolded matrices.
    pub unfolding_imaginary: f64,
    /// Largest difference of natural charges between equivalent atoms of
    /// different cells.
    pub translation_defect: f64,
}

/// Real supercell matrix of Bloch matrices on the mesh and its largest
/// imaginary part.
fn unfold(x: &[BlochMatrix], k_points: &[Vec3], translations: &[Vec3]) -> (Matrix, f64) {
    let n = x[0].rows();
    let cells = translations.len();
    let mut out = Matrix::zeros(n * cells, n * cells);
    let mut imaginary = 0.0_f64;
    let scale = 1.0 / k_points.len() as f64;
    for (l, tl) in translations.iter().enumerate() {
        for (m, tm) in translations.iter().enumerate() {
            let r = *tm - *tl;
            let mut block_re = Matrix::zeros(n, n);
            let mut block_im = Matrix::zeros(n, n);
            for (k, xk) in k_points.iter().zip(x) {
                let (s, c) = (-k.dot(r)).sin_cos();
                for i in 0..n {
                    for j in 0..n {
                        let (re, im) = (xk.re[(i, j)], xk.im[(i, j)]);
                        block_re[(i, j)] += scale * (re * c - im * s);
                        block_im[(i, j)] += scale * (re * s + im * c);
                    }
                }
            }
            for i in 0..n {
                for j in 0..n {
                    out[(l * n + i, m * n + j)] = block_re[(i, j)];
                    imaginary = imaginary.max(block_im[(i, j)].abs());
                }
            }
        }
    }
    (out, imaginary)
}

/// Natural bond orbital analysis of a periodic reference on the unshifted
/// `mesh` with lattice vectors `lattice` (Bohr; vectors of open axes are
/// not used when their mesh size is one).
pub fn periodic_natural_bond(
    molecule: &Molecule,
    basis: &MolBasis,
    lattice: [Vec3; 3],
    mesh: [usize; 3],
    scf: &PeriodicScfResult,
    spin: NboSpin,
    options: &NboOptions,
) -> Result<PeriodicNaturalBond> {
    let n_k: usize = mesh.iter().product();
    if n_k == 0 || scf.k_points.len() != n_k || scf.overlap.len() != n_k {
        return Err(DftError::InvalidInput(format!(
            "the periodic reference has {} k points for the {mesh:?} mesh",
            scf.k_points.len()
        )));
    }
    for k in &scf.k_points {
        for (axis, &size) in mesh.iter().enumerate() {
            if size > 1 {
                let phase = k.dot(lattice[axis] * size as f64);
                if (phase - std::f64::consts::TAU * (phase / std::f64::consts::TAU).round()).abs()
                    > 1e-8
                {
                    return Err(DftError::InvalidInput(
                        "the k points are not the unshifted Born-von Karman mesh of the \
                         supercell, so the supercell matrices are not real"
                            .into(),
                    ));
                }
            }
        }
    }
    let mut cells = Vec::with_capacity(n_k);
    for i in 0..mesh[0] {
        for j in 0..mesh[1] {
            for k in 0..mesh[2] {
                cells.push([i, j, k]);
            }
        }
    }
    let translations: Vec<Vec3> = cells
        .iter()
        .map(|c| lattice[0] * c[0] as f64 + lattice[1] * c[1] as f64 + lattice[2] * c[2] as f64)
        .collect();
    let (overlap, mut imaginary) = unfold(&scf.overlap, &scf.k_points, &translations);
    let mut unfold_pair = |x: &[Vec<BlochMatrix>; 2]| -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let (m, im) = unfold(&x[s], &scf.k_points, &translations);
            imaginary = imaginary.max(im);
            m
        })
    };
    let density = unfold_pair(&scf.density);
    let fock = unfold_pair(&scf.fock);
    let natoms = molecule.len();
    let nao = basis.nao_sph;
    let single = shell_layout(basis);
    let shells: Vec<ShellLayout> = (0..n_k)
        .flat_map(|c| {
            single.iter().map(move |s| ShellLayout {
                atom: s.atom + c * natoms,
                l: s.l,
                offset: s.offset + c * nao,
            })
        })
        .collect();
    let numbers: Vec<u8> = (0..n_k)
        .flat_map(|_| molecule.atoms.iter().map(|a| a.z))
        .collect();
    let ecp: Vec<u32> = (0..n_k)
        .flat_map(|_| ecp_core_electrons(basis, natoms))
        .collect();
    let charges: Vec<f64> = (0..n_k)
        .flat_map(|_| basis.effective_charges(molecule))
        .collect();
    let analysis = natural_bond_from_density(
        &overlap,
        [&density[0], &density[1]],
        Some([&fock[0], &fock[1]]),
        &shells,
        &numbers,
        &ecp,
        &charges,
        spin,
        options,
    )?;
    let q = &analysis.nao.atomic_charges;
    let translation_defect =
        (0..n_k * natoms).fold(0.0_f64, |m, g| m.max((q[g] - q[g % natoms]).abs()));
    Ok(PeriodicNaturalBond {
        analysis,
        cells,
        mesh,
        atoms_per_cell: natoms,
        unfolding_imaginary: imaginary,
        translation_defect,
    })
}
