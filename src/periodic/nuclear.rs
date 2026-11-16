// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic nuclear attraction `V(k)`, by Ewald splitting.
//!
//! # Why this cannot be a cutoff
//!
//! `S` and `T` are lattice sums of exponentially decaying integrands, so a
//! real-space cutoff converges them. The nuclear attraction is not: the
//! integrand falls off as `1/r`, and the lattice sum of `1/r` is
//! CONDITIONALLY convergent — truncating it gives an answer that depends
//! on the shape of the truncation region, so "increase the cutoff until it
//! stops moving" converges to the wrong number rather than slowly.
//!
//! The fix is the same split [`super::ewald`] uses for the point charges,
//! applied here to a matrix element:
//!
//! ```text
//! 1/r = erfc(alpha r)/r + erf(alpha r)/r
//! ```
//!
//! - the `erfc` half is short-ranged and summed in real space — and it is
//!   exactly what `CoulombKernel::Erfc` already computes, so no new
//!   integral class is needed;
//! - the `erf` half is smooth, so its lattice sum converges in reciprocal
//!   space, where it needs the density Fourier components
//!   [`super::planewave::plane_wave_contraction`] provides and the nuclear
//!   structure factor.
//!
//! # Both halves are image blocks
//!
//! The natural output is not a matrix but the set of image blocks
//! `< mu(0) | V | nu(T) >`, the same objects [`super::integrals`] produces
//! for `S` and `T`, because then one build serves every `k` point. That
//! works here even though the reciprocal half is assembled from complex
//! Fourier components: `V` is a real function and the basis is real, so
//! each block is real, and the imaginary parts cancel exactly between `G`
//! and `-G`. The complex arithmetic enters only at the Bloch sum, where it
//! does for `S` and `T` as well.
//!
//! # The G = 0 term
//!
//! The reciprocal sum's `G = 0` element diverges for a charged system.
//! What is included here is the term that pairs with the neutralising
//! background of [`super::ewald`]: a constant shift
//! `pi Z_tot / (alpha^2 V)` times the overlap. It cancels against the
//! matching pieces of the Hartree and nuclear-nuclear terms in a neutral
//! cell — which is why only the total is meaningful, and why this matrix
//! on its own is convention-dependent at that one term. The same
//! convention is what [`super::ewald::ewald_potential`] carries, which is
//! what lets that function serve as the independent reference below.

use crate::basis::{MolBasis, MolShell};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::ewald::EwaldParams;
use crate::periodic::integrals::{concat, shifted, ImageBlock, ImageDerivBlock};
use crate::periodic::planewave::contracted_pair_blocks;
use crate::periodic::Lattice;
use crate::system::{Atom, Molecule};

/// Every nuclear image whose screened potential can still reach the cell,
/// as one flat charge list.
///
/// Handing the existing molecular routine a single molecule of a few
/// hundred image charges costs one pass; calling it once per image would
/// cost one pass each, and the passes are the expensive part.
pub(crate) fn image_charges(
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    reach: f64,
) -> (Molecule, Vec<f64>) {
    let mut atoms = Vec::new();
    let mut q = Vec::new();
    for t in lattice.translations_within(reach) {
        for (a, z) in mol.atoms.iter().zip(charges) {
            atoms.push(Atom {
                z: a.z,
                position: a.position + t,
            });
            q.push(*z);
        }
    }
    (Molecule::new(atoms), q)
}

/// The largest distance between two shell centres of the cell.
pub(crate) fn shell_spread(mb: &MolBasis) -> f64 {
    mb.shells
        .iter()
        .flat_map(|a| mb.shells.iter().map(move |b| (a.center - b.center).norm()))
        .fold(0.0_f64, f64::max)
}

/// The pair criterion of the overlap images (`short_range_images`): the
/// shell pair with `left` in the home cell and `right` in the cell at `t`
/// is kept when its centres are at most `cutoff` apart. Applied per pair
/// rather than per translation, the truncation does not depend on where the
/// cell boundary is drawn, so a supercell has the operator of its primitive
/// cell. (Truncating whole translation blocks at `|T| <= cutoff` broke the
/// translation symmetry of a two-cell hydrogen chain at a 14 Bohr cutoff:
/// moving an atom or its image in the other cell changed the energy by
/// 4.5e-7 Hartree and the supercell force constants by 5.9e-5.)
pub(crate) fn pair_within(left: &MolShell, right: &MolShell, t: Vec3, cutoff: f64) -> bool {
    (right.center + t - left.center).norm() <= cutoff
}

/// Every translation that can carry a shell pair within `cutoff`.
pub(crate) fn pair_translations(mb: &MolBasis, lattice: &Lattice, cutoff: f64) -> Vec<Vec3> {
    lattice.translations_within(cutoff + shell_spread(mb))
}

/// For every spherical AO, the index of its shell.
fn ao_shells(mb: &MolBasis) -> Vec<usize> {
    let mut out = vec![0; mb.nao_sph];
    for (index, shell) in mb.shells.iter().enumerate() {
        for slot in &mut out[shell.sph_off..shell.sph_off + crate::basis::n_sph(shell.l)] {
            *slot = index;
        }
    }
    out
}

/// The largest distance between any two atoms of the cell.
pub(crate) fn atom_spread(mol: &Molecule) -> f64 {
    mol.atoms
        .iter()
        .flat_map(|a| {
            mol.atoms
                .iter()
                .map(move |b| (a.position - b.position).norm())
        })
        .fold(0.0_f64, f64::max)
}

/// `< mu(0) | V | nu(T) >` for every shell pair within `basis_cutoff`.
///
/// `basis_cutoff` bounds the KET images by the distance of the pair
/// (`pair_within`) — it is an overlap-type cutoff, set by how far the basis
/// functions reach. The nuclear images are bounded
/// separately by `p.rcut`, which is set by the screening length. Conflating
/// the two is the easiest way to get a silently truncated sum, so they are
/// kept apart.
pub fn nuclear_image_blocks(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
) -> Vec<ImageBlock> {
    let n = mb.nao_sph;
    let natoms = mb.shells.iter().map(|s| s.atom + 1).max().unwrap_or(0);
    let v = lattice.volume().abs();

    // Screened nuclear images. The cutoff applies to the distance from the
    // charge to the basis functions, not to the translation length, so the
    // enumeration has to reach `rcut` plus the span of everything else.
    let reach = p.rcut + basis_cutoff + atom_spread(mol);
    let (img_mol, img_q) = image_charges(mol, charges, lattice, reach);

    // Reciprocal mesh and the coefficient of each G, shared across all
    // ket translations: `-(4 pi / V) exp(-G^2/4 alpha^2)/G^2 * conj(S_nuc)`.
    let gs = lattice.reciprocal_within(p.gcut);
    let mut c_re = Vec::with_capacity(gs.len());
    let mut c_im = Vec::with_capacity(gs.len());
    for g in &gs {
        let g2 = g.norm2();
        let w = -(4.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * p.alpha * p.alpha)).exp() / g2;
        let (mut sc, mut ss) = (0.0, 0.0);
        for (a, z) in mol.atoms.iter().zip(charges) {
            let ph = g.dot(a.position);
            sc += z * ph.cos();
            ss += z * ph.sin();
        }
        // conj(S_nuc(G)) = sc - i ss.
        c_re.push(w * sc);
        c_im.push(-w * ss);
    }

    // The G = 0 term, which multiplies the overlap of the same image block.
    let ztot: f64 = charges.iter().sum();
    let shift = std::f64::consts::PI * ztot / (p.alpha * p.alpha * v);

    use rayon::prelude::*;
    let centers: Vec<Vec3> = img_mol.atoms.iter().map(|a| a.position).collect();
    let mut out = Vec::new();
    for t in pair_translations(mb, lattice, basis_cutoff) {
        let pairs: Vec<(usize, usize)> = (0..mb.shells.len())
            .flat_map(|a| (0..mb.shells.len()).map(move |b| (a, b)))
            .filter(|&(a, b)| pair_within(&mb.shells[a], &mb.shells[b], t, basis_cutoff))
            .collect();
        if pairs.is_empty() {
            continue;
        }
        let shifted_basis = shifted(mb, t);
        let comb = concat(mb, &shifted_basis, natoms);
        let overlap = crate::integrals::overlap_matrix(&comb);
        let blocks: Vec<(usize, usize, Matrix)> = pairs
            .par_iter()
            .map(|&(a, b)| {
                let (left, right) = (&mb.shells[a], &shifted_basis.shells[b]);
                let short_cart = crate::integrals::one_electron::nuclear_pair_block_kernel(
                    left,
                    right,
                    &centers,
                    &img_q,
                    CoulombKernel::Erfc { omega: p.alpha },
                );
                let short = crate::basis::aos::sph_block(left.l, right.l, &short_cart);
                let long_cart = contracted_pair_blocks(left, right, &gs, &[&c_re], &[&c_im]);
                let long = crate::basis::aos::sph_block(left.l, right.l, &long_cart[0].0);
                let mut block = Matrix::zeros(short.rows, short.cols);
                for i in 0..block.rows {
                    for j in 0..block.cols {
                        block[(i, j)] = short[(i, j)]
                            + long[(i, j)]
                            + shift * overlap[(left.sph_off + i, n + right.sph_off + j)];
                    }
                }
                (left.sph_off, right.sph_off, block)
            })
            .collect();
        let mut matrix = Matrix::zeros(n, n);
        let mut worst = 0.0_f64;
        for (a, b, block) in blocks {
            for i in 0..block.rows {
                for j in 0..block.cols {
                    matrix[(a + i, b + j)] = block[(i, j)];
                    worst = worst.max(block[(i, j)].abs());
                }
            }
        }
        if worst > 1e-14 {
            out.push(ImageBlock {
                translation: t,
                matrix,
            });
        }
    }
    out
}

/// Nuclear derivatives of [`nuclear_image_blocks`] — the nuclear-attraction
/// half of a periodic analytic gradient.
///
/// Three things move when an atom does, and all three are here:
///
/// 1. the basis functions centred on it, in the home cell and in the shifted
///    copy (the same double bookkeeping
///    [`super::integrals::image_derivative_blocks`] does for `S` and `T`);
/// 2. the screened charge at that atom **and every one of its lattice
///    images** — the Hellmann–Feynman half of the real-space term;
/// 3. the nuclear structure factor `S_nuc(G)`, which is where the
///    reciprocal half feels the displacement.
///
/// Missing any one of them still gives a plausible-looking gradient, which
/// is why the gate below is a finite difference of the assembled `V(k)` and
/// not of the pieces.
pub fn nuclear_derivative_image_blocks(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
) -> Vec<ImageDerivBlock> {
    use crate::integrals::deriv::nuclear_block_kernel;
    use crate::integrals::deriv_mat::{
        charge_site_derivative_matrices, moving_center_derivative_matrices,
        overlap_derivative_matrices,
    };
    let n = mb.nao_sph;
    let natoms = mol.len();
    let v = lattice.volume().abs();
    let nuc_images = lattice.translations_within(p.rcut + basis_cutoff + atom_spread(mol));

    let gs = lattice.reciprocal_within(p.gcut);
    let wg: Vec<f64> = gs
        .iter()
        .map(|g| {
            let g2 = g.norm2();
            -(4.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * p.alpha * p.alpha)).exp() / g2
        })
        .collect();

    // Coefficient of each G in the energy-like term, and its derivative with
    // respect to every nuclear coordinate. `conj(S_nuc) = sc - i ss`, so
    // `d sc/dR_ax = -Z G_ax sin(G.R)` and `d ss/dR_ax = Z G_ax cos(G.R)`.
    let mut c_re = vec![0.0; gs.len()];
    let mut c_im = vec![0.0; gs.len()];
    let mut d_re: Vec<Vec<f64>> = vec![vec![0.0; gs.len()]; 3 * natoms];
    let mut d_im: Vec<Vec<f64>> = vec![vec![0.0; gs.len()]; 3 * natoms];
    for (gi, g) in gs.iter().enumerate() {
        let (mut sc, mut ss) = (0.0, 0.0);
        for (a, (at, z)) in mol.atoms.iter().zip(charges).enumerate() {
            let ph = g.dot(at.position);
            let (cs, sn) = (ph.cos(), ph.sin());
            sc += z * cs;
            ss += z * sn;
            for ax in 0..3 {
                d_re[3 * a + ax][gi] = wg[gi] * (-z * g.get(ax) * sn);
                d_im[3 * a + ax][gi] = -wg[gi] * (z * g.get(ax) * cs);
            }
        }
        c_re[gi] = wg[gi] * sc;
        c_im[gi] = -wg[gi] * ss;
    }
    let refs_re: Vec<&[f64]> = d_re.iter().map(|s| s.as_slice()).collect();
    let refs_im: Vec<&[f64]> = d_im.iter().map(|s| s.as_slice()).collect();

    let ztot: f64 = charges.iter().sum();
    let shift = std::f64::consts::PI * ztot / (p.alpha * p.alpha * v);

    let mut out = Vec::new();
    let shell_of = ao_shells(mb);
    for t in pair_translations(mb, lattice, basis_cutoff) {
        let comb = concat(mb, &shifted(mb, t), natoms);

        // Real space: one charge SITE per physical atom, its lattice images
        // folded into the site's own block so that the operator-centre term
        // lands on the right atom in one go.
        let short = charge_site_derivative_matrices(
            &comb,
            2 * natoms,
            natoms,
            |s| s,
            |sa, sb, s| {
                let mut acc = Matrix::zeros(
                    crate::basis::cart_components(sa.l).len(),
                    crate::basis::cart_components(sb.l).len(),
                );
                for tn in &nuc_images {
                    let blk = nuclear_block_kernel(
                        sa,
                        sb,
                        mol.atoms[s].position + *tn,
                        charges[s],
                        CoulombKernel::Erfc { omega: p.alpha },
                    );
                    for (o, w) in acc.as_mut_slice().iter_mut().zip(blk.as_slice()) {
                        *o += *w;
                    }
                }
                acc
            },
        );

        // Reciprocal space, basis-centre half. The plane wave is NOT
        // translationally invariant, so bra and ket derivatives do not
        // cancel on a shared centre — hence the moving-centre driver rather
        // than the overlap one.
        let long_basis = moving_center_derivative_matrices(&comb, 2 * natoms, |sa, sb| {
            contracted_pair_blocks(sa, sb, &gs, &[&c_re], &[&c_im])
                .remove(0)
                .0
        });

        // Reciprocal space, structure-factor half: one coefficient set per
        // nuclear coordinate, all sharing a single pass over the G mesh.
        let d_struct =
            crate::periodic::planewave::plane_wave_contractions(&comb, &gs, &refs_re, &refs_im);

        // The G = 0 background shift rides on the overlap, so it
        // differentiates with it.
        let dovl = overlap_derivative_matrices(&comb, 2 * natoms);

        let mut d: Vec<[Matrix; 3]> = (0..natoms)
            .map(|_| {
                [
                    Matrix::zeros(n, n),
                    Matrix::zeros(n, n),
                    Matrix::zeros(n, n),
                ]
            })
            .collect();
        for a in 0..natoms {
            for ax in 0..3 {
                for i in 0..n {
                    for j in 0..n {
                        if !pair_within(
                            &mb.shells[shell_of[i]],
                            &mb.shells[shell_of[j]],
                            t,
                            basis_cutoff,
                        ) {
                            continue;
                        }
                        let val = short[a][ax][(i, n + j)]
                            + short[natoms + a][ax][(i, n + j)]
                            + long_basis[a][ax][(i, n + j)]
                            + long_basis[natoms + a][ax][(i, n + j)]
                            + d_struct[3 * a + ax].0[(i, n + j)]
                            + shift * (dovl[a][ax][(i, n + j)] + dovl[natoms + a][ax][(i, n + j)]);
                        d[a][ax][(i, j)] = val;
                    }
                }
            }
        }
        out.push(ImageDerivBlock { translation: t, d });
    }
    out
}

/// Full periodic nuclear attraction at `k`, as `(real, imaginary)`.
pub fn nuclear_matrix_k(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    k: Vec3,
) -> (Matrix, Matrix) {
    let blocks = nuclear_image_blocks(mb, mol, charges, lattice, basis_cutoff, p);
    crate::periodic::integrals::bloch(&blocks, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;

    fn water() -> (Molecule, MolBasis) {
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
                position: Vec3::new(0.15, -1.4009, -0.9067),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb)
    }

    /// The decisive gate: the home block against quadrature of the point
    /// charges' Ewald potential.
    ///
    /// The two routes share nothing. This module reaches the answer through
    /// screened nuclear-attraction integrals plus a plane-wave expansion of
    /// the density; the reference evaluates
    /// [`crate::periodic::ewald::ewald_potential`] on the molecular grid and
    /// integrates `chi_mu chi_nu V` numerically. They agree only if the
    /// splitting, the prefactors, the structure factor and the `G = 0`
    /// convention all match — and they are checked in a cell small enough
    /// that the periodicity is doing real work, not in a large-cell limit
    /// where any of it could be wrong and invisible.
    #[test]
    fn home_block_matches_quadrature_of_the_ewald_potential() {
        use crate::grid::{GridOptions, MolecularGrid};
        use crate::periodic::ewald::ewald_potential;
        let (mol, mb) = water();
        let lat = Lattice::cubic(9.0);
        let charges: Vec<f64> = mol.atoms.iter().map(|a| a.z as f64).collect();
        let p = EwaldParams {
            alpha: 0.45,
            rcut: 12.0,
            gcut: 5.0,
        };
        let blocks = nuclear_image_blocks(&mb, &mol, &charges, &lat, 12.0, p);
        let home = blocks
            .iter()
            .find(|b| b.translation.norm() < 1e-12)
            .expect("the T = 0 block must exist");

        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 6,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let pos: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
        let batch = crate::grid::eval::eval_aos(&mb, &grid.points, false);
        let n = mb.nao_sph;
        let mut q = Matrix::zeros(n, n);
        for (idx, r) in grid.points.iter().enumerate() {
            // The operator is minus the potential of positive nuclei.
            let vop = -ewald_potential(&lat, &pos, &charges, *r, p);
            let w = grid.weights[idx] * vop;
            for i in 0..n {
                let pi = batch.phi[(idx, i)];
                if pi == 0.0 {
                    continue;
                }
                for j in 0..n {
                    q[(i, j)] += w * pi * batch.phi[(idx, j)];
                }
            }
        }
        // The quadrature is the loose side: a level-6 grid resolves the
        // nuclear cusps of the operator to a few parts in 10^5.
        let worst = home.matrix.max_abs_diff(&q);
        assert!(
            worst < 2e-5,
            "periodic V vs Ewald-potential quadrature: {worst:e}"
        );
    }

    /// The splitting parameter divides work between the two halves and
    /// must cancel from the total. This is the check that catches a
    /// mis-scaled reciprocal kernel or a wrong `G = 0` shift, since both
    /// are alpha-dependent and neither shows up at a single alpha.
    #[test]
    fn total_is_independent_of_the_splitting_parameter() {
        let (mol, mb) = water();
        let lat = Lattice::cubic(9.0);
        let charges: Vec<f64> = mol.atoms.iter().map(|a| a.z as f64).collect();
        let k = Vec3::new(0.1, -0.05, 0.07);
        let mut mats = Vec::new();
        for alpha in [0.35_f64, 0.45, 0.55] {
            let p = EwaldParams {
                alpha,
                // erfc(4.9) and exp(-(9.6/2)^2) are both below 1e-10.
                rcut: 4.9 / alpha,
                gcut: 9.6 * alpha,
            };
            mats.push(nuclear_matrix_k(&mb, &mol, &charges, &lat, 12.0, p, k));
        }
        for w in mats.windows(2) {
            let dre = w[0].0.max_abs_diff(&w[1].0);
            let dim = w[0].1.max_abs_diff(&w[1].1);
            assert!(
                dre < 1e-7 && dim < 1e-7,
                "V(k) depends on alpha: real {dre:e}, imaginary {dim:e}"
            );
        }
    }

    /// In a cell far larger than everything in it, a NEUTRAL set of charges
    /// must reproduce the molecular nuclear attraction.
    ///
    /// Neutral is the point. With a net charge the periodic value differs
    /// from the molecular one by a Madelung-type constant that dies only as
    /// `1/L` — real physics of the boundary conditions, not an error — so
    /// the charged case has no molecular limit to test against. Zero net
    /// charge removes both the background and that `1/L` term, leaving a
    /// residue set by the dipole and higher moments, which decays fast
    /// enough to see.
    #[test]
    fn a_neutral_cell_has_the_molecular_limit() {
        let (mol, mb) = water();
        let lat = Lattice::cubic(60.0);
        // Formal charges summing to zero: the operator is then a genuine
        // molecular one in the large-cell limit.
        let charges = vec![-2.0, 1.0, 1.0];
        let p = EwaldParams {
            alpha: 0.12,
            rcut: 42.0,
            gcut: 1.2,
        };
        let blocks = nuclear_image_blocks(&mb, &mol, &charges, &lat, 20.0, p);
        let got = crate::periodic::integrals::gamma_sum(&blocks);
        let want = crate::integrals::nuclear_matrix(&mb, &mol, Some(&charges));
        let worst = got.max_abs_diff(&want);
        assert!(
            worst < 1e-4,
            "periodic V differs from the molecular one at L = 60: {worst:e}"
        );
    }

    /// The analytic nuclear derivative against a finite difference of the
    /// assembled `V(k)`, at a generic `k`.
    ///
    /// A generic `k` rather than Gamma on purpose: the Bloch phases weight
    /// the image blocks differently, so a derivative that is right in total
    /// but attributed to the wrong translation passes at Gamma and fails
    /// here. The comparison is against the assembled matrix, not the
    /// pieces, because the three ways an atom enters — its basis functions,
    /// its screened charge and all that charge's images, and the structure
    /// factor — can each be wrong in a way the others hide.
    #[test]
    fn derivative_blocks_match_finite_differences() {
        use crate::periodic::integrals::bloch_derivative;
        let mol = Molecule::new(vec![
            Atom {
                z: 3,
                position: Vec3::new(0.0, 0.0, -0.9),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.2, 0.1, 2.1),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let lat = Lattice::cubic(8.0);
        let charges: Vec<f64> = mol.atoms.iter().map(|a| a.z as f64).collect();
        let p = EwaldParams {
            alpha: 0.3,
            rcut: 16.3,
            gcut: 2.9,
        };
        let cutoff = 9.0;
        let b = lat.reciprocal();
        let k = b[0] * 0.3 + b[1] * 0.15 - b[2] * 0.1;

        let blocks = nuclear_derivative_image_blocks(&mb, &mol, &charges, &lat, cutoff, p);
        let ana = bloch_derivative(&blocks, k, mol.len(), mb.nao_sph);

        let h = 1e-4;
        let n = mb.nao_sph;
        for a in 0..mol.len() {
            for ax in 0..3 {
                let mut shifted_mats = Vec::new();
                for s in [1.0_f64, -1.0] {
                    let mut m = mol.clone();
                    let mut pos = m.atoms[a].position;
                    match ax {
                        0 => pos.x += s * h,
                        1 => pos.y += s * h,
                        _ => pos.z += s * h,
                    }
                    m.atoms[a].position = pos;
                    let mbd = MolBasis::build(&m, &set).unwrap();
                    shifted_mats.push(nuclear_matrix_k(&mbd, &m, &charges, &lat, cutoff, p, k));
                }
                for i in 0..n {
                    for j in 0..n {
                        let fd_re =
                            (shifted_mats[0].0[(i, j)] - shifted_mats[1].0[(i, j)]) / (2.0 * h);
                        let fd_im =
                            (shifted_mats[0].1[(i, j)] - shifted_mats[1].1[(i, j)]) / (2.0 * h);
                        assert!(
                            (ana[a][ax].0[(i, j)] - fd_re).abs() < 2e-6,
                            "Re dV/dR[{a}][{ax}]({i},{j}): analytic {} vs FD {fd_re}",
                            ana[a][ax].0[(i, j)]
                        );
                        assert!(
                            (ana[a][ax].1[(i, j)] - fd_im).abs() < 2e-6,
                            "Im dV/dR[{a}][{ax}]({i},{j}): analytic {} vs FD {fd_im}",
                            ana[a][ax].1[(i, j)]
                        );
                    }
                }
            }
        }
    }

    /// `V(k)` must be Hermitian: real part symmetric, imaginary part
    /// antisymmetric.
    #[test]
    fn is_hermitian() {
        let (mol, mb) = water();
        let lat = Lattice::cubic(8.0);
        let charges: Vec<f64> = mol.atoms.iter().map(|a| a.z as f64).collect();
        let p = EwaldParams {
            alpha: 0.5,
            rcut: 10.0,
            gcut: 5.0,
        };
        let b = lat.reciprocal();
        for k in [b[0] * 0.5, b[0] * 0.25 + b[1] * 0.25] {
            let (re, im) = nuclear_matrix_k(&mb, &mol, &charges, &lat, 11.0, p, k);
            let n = mb.nao_sph;
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (re[(i, j)] - re[(j, i)]).abs() < 1e-10,
                        "Re V(k) is not symmetric"
                    );
                    assert!(
                        (im[(i, j)] + im[(j, i)]).abs() < 1e-10,
                        "Im V(k) is not antisymmetric"
                    );
                }
            }
        }
    }
}
