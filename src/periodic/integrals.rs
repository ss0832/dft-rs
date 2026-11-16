// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! One-electron integrals under periodic boundary conditions: real-space
//! lattice sums and their Bloch transforms.
//!
//! # The construction
//!
//! A crystal orbital is a Bloch sum of the cell's atomic orbitals,
//!
//! ```text
//! chi_mu^k(r) = sum_T exp(i k . T) chi_mu(r - T)
//! ```
//!
//! so every matrix element becomes a lattice sum over the IMAGE blocks
//!
//! ```text
//! S^T_mn = < chi_m(r) | chi_n(r - T) > ,   S(k) = sum_T exp(i k . T) S^T
//! ```
//!
//! The image blocks are computed once and reused for every `k`, which is
//! what makes k-point sampling cheap after the first pass. They are also
//! the natural place to truncate: Gaussians overlap only out to a finite
//! range, so the sum is absolutely convergent — unlike the Coulomb sums in
//! [`super::ewald`], which need Ewald splitting.
//!
//! # No complex matrix type
//!
//! The project keeps its linear algebra real. `S(k)` is Hermitian, so it is
//! carried as a `(real, imaginary)` pair, and where an eigenvalue problem
//! is needed the equivalent real symmetric form
//!
//! ```text
//! [ Re(S)  -Im(S) ]
//! [ Im(S)   Re(S) ]
//! ```
//!
//! is used, whose eigenvalues are those of `S(k)` each appearing twice.
//! That is enough for the checks here and avoids a complex layer before
//! anything needs one.

use crate::basis::{n_sph, MolBasis, MolShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::Lattice;
use std::collections::{BTreeMap, BTreeSet};

/// Which one-electron operator a lattice sum is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operator {
    Overlap,
    Kinetic,
}

/// One image block: the translation and the matrix `< mu(0) | O | nu(T) >`.
#[derive(Clone, Debug)]
pub struct ImageBlock {
    pub translation: Vec3,
    pub matrix: Matrix,
}

/// Shift every shell of a basis by `t`.
pub(crate) fn shifted(mb: &MolBasis, t: Vec3) -> MolBasis {
    let mut out = mb.clone();
    for sh in out.shells.iter_mut() {
        sh.center += t;
    }
    for e in out.ecps.iter_mut() {
        e.center += t;
    }
    out
}

/// Concatenate two bases into one, offsetting the second's AO indices and
/// its atom labels by `natoms`.
///
/// The atom offset is load-bearing for the derivative path: a shell in the
/// shifted copy belongs to the SAME physical atom as its home-cell
/// original, so moving that atom moves both. Labelling them `natoms + A`
/// keeps the two contributions separable, and the derivative routine adds
/// them back together.
pub(crate) fn concat(a: &MolBasis, b: &MolBasis, natoms: usize) -> MolBasis {
    let mut shells: Vec<MolShell> = a.shells.clone();
    let (so, co) = (a.nao_sph, a.nao_cart);
    for sh in &b.shells {
        let mut s = sh.clone();
        s.sph_off += so;
        s.cart_off += co;
        s.atom += natoms;
        shells.push(s);
    }
    MolBasis {
        shells,
        nao_sph: a.nao_sph + b.nao_sph,
        nao_cart: a.nao_cart + b.nao_cart,
        max_l: a.max_l.max(b.max_l),
        ecps: Vec::new(),
        n_core_electrons: 0,
    }
}

/// One lattice image together with the shell pairs whose *actual* centre
/// separation is inside the short-range cutoff.
pub(crate) struct ShortRangeImage {
    pub(crate) translation: Vec3,
    pub(crate) shell_pairs: Vec<(usize, usize)>,
}

/// Integer translations `n` for which `|displacement + A n| <= cutoff`.
///
/// The reciprocal vectors give exact finite bounds even for a skewed,
/// unreduced lattice.  Splitting the fractional displacement into an integer
/// origin and a small residual is important: if an AO is relabelled into a
/// neighbouring cell, the origin changes but the local enumeration and its
/// floating-point distances do not.
pub(crate) fn nearby_translation_indices(
    lattice: &Lattice,
    displacement: Vec3,
    cutoff: f64,
) -> Vec<[i64; 3]> {
    if !(cutoff.is_finite() && cutoff >= 0.0) {
        return Vec::new();
    }
    let fractional = lattice.to_fractional(displacement);
    if fractional.iter().any(|value| !value.is_finite()) {
        return Vec::new();
    }
    let origin = [
        -(fractional[0].round() as i64),
        -(fractional[1].round() as i64),
        -(fractional[2].round() as i64),
    ];
    let residual = [
        fractional[0] + origin[0] as f64,
        fractional[1] + origin[1] as f64,
        fractional[2] + origin[2] as f64,
    ];
    let reciprocal = lattice.reciprocal();
    let two_pi = 2.0 * std::f64::consts::PI;
    let tolerance = 64.0 * f64::EPSILON * (1.0 + cutoff);
    let radius = cutoff + tolerance;
    let mut lower = [0i64; 3];
    let mut upper = [0i64; 3];
    for axis in 0..3 {
        let bound = radius * reciprocal[axis].norm() / two_pi + 64.0 * f64::EPSILON;
        lower[axis] = (-residual[axis] - bound).ceil() as i64;
        upper[axis] = (-residual[axis] + bound).floor() as i64;
    }

    let radius2 = radius * radius;
    let mut indices = Vec::new();
    for i in lower[0]..=upper[0] {
        for j in lower[1]..=upper[1] {
            for k in lower[2]..=upper[2] {
                let local = [i, j, k];
                let separation = lattice.to_cartesian([
                    residual[0] + i as f64,
                    residual[1] + j as f64,
                    residual[2] + k as f64,
                ]);
                if separation.norm2() <= radius2 {
                    indices.push([
                        origin[0] + local[0],
                        origin[1] + local[1],
                        origin[2] + local[2],
                    ]);
                }
            }
        }
    }
    indices
}

/// Build the finite union of translations needed by all shell pairs.
///
/// Enumerating a single sphere enlarged by the coordinate spread is correct
/// but can become arbitrarily expensive when an atom is represented in a
/// lattice-equivalent neighbouring cell.  Per-pair local spheres avoid that
/// gauge-dependent cost, and retaining the pair list prevents an unrelated
/// long-range AO entry from leaking into a block selected for another pair.
pub(crate) fn short_range_images(
    mb: &MolBasis,
    lattice: &Lattice,
    cutoff: f64,
) -> Vec<ShortRangeImage> {
    let mut by_translation: BTreeMap<[i64; 3], Vec<(usize, usize)>> = BTreeMap::new();
    for (left_index, left) in mb.shells.iter().enumerate() {
        for (right_index, right) in mb.shells.iter().enumerate() {
            let displacement = right.center - left.center;
            for index in nearby_translation_indices(lattice, displacement, cutoff) {
                by_translation
                    .entry(index)
                    .or_default()
                    .push((left_index, right_index));
            }
        }
    }
    let mut images: Vec<ShortRangeImage> = by_translation
        .into_iter()
        .map(|(index, shell_pairs)| ShortRangeImage {
            translation: lattice.translation(index),
            shell_pairs,
        })
        .collect();
    images.sort_by(|a, b| {
        a.translation
            .norm2()
            .total_cmp(&b.translation.norm2())
            .then_with(|| a.translation.x.total_cmp(&b.translation.x))
            .then_with(|| a.translation.y.total_cmp(&b.translation.y))
            .then_with(|| a.translation.z.total_cmp(&b.translation.z))
    });
    images
}

pub(crate) fn dimensional_short_range_images(
    mb: &MolBasis,
    geometry: &super::PeriodicGeometry,
    cutoff: f64,
) -> crate::Result<Vec<ShortRangeImage>> {
    let mut selected: BTreeMap<[i64; 3], Vec<(usize, usize)>> = BTreeMap::new();
    for (i, left) in mb.shells.iter().enumerate() {
        for (j, right) in mb.shells.iter().enumerate() {
            for index in geometry.nearby_translation_indices(right.center - left.center, cutoff)? {
                selected.entry(index).or_default().push((i, j));
            }
        }
    }
    let mut images: Vec<_> = selected
        .into_iter()
        .map(|(index, shell_pairs)| ShortRangeImage {
            translation: geometry.lattice().translation(index),
            shell_pairs,
        })
        .collect();
    images.sort_by(|a, b| {
        a.translation
            .norm2()
            .total_cmp(&b.translation.norm2())
            .then_with(|| a.translation.x.total_cmp(&b.translation.x))
            .then_with(|| a.translation.y.total_cmp(&b.translation.y))
            .then_with(|| a.translation.z.total_cmp(&b.translation.z))
    });
    Ok(images)
}

type EcpImageKey = (usize, [i64; 3]);

mod ecp_geometry;
pub(crate) use ecp_geometry::{
    contract_ecp_geometry, contract_ecp_hessian, ecp_strain_blocks, ecp_strain_second,
    gamma_ecp_derivative, gamma_ecp_path, gamma_ecp_second, EcpBoundary, EcpGeometry,
};

/// Periodic ECP centres close to either AO centre of one selected shell pair.
///
/// The key retains both the source ECP and its lattice index.  A centre found
/// from both AO sides is therefore counted once, while distinct coincident
/// pseudopotentials remain distinct physical contributions.
fn ecp_images_near_pair(
    mb: &MolBasis,
    lattice: &Lattice,
    left_center: Vec3,
    right_center: Vec3,
    cutoff: f64,
) -> Vec<EcpImageKey> {
    let mut selected = BTreeSet::new();
    for (ecp_index, ecp) in mb.ecps.iter().enumerate() {
        for ao_center in [left_center, right_center] {
            for translation in nearby_translation_indices(lattice, ecp.center - ao_center, cutoff) {
                selected.insert((ecp_index, translation));
            }
        }
    }
    selected.into_iter().collect()
}

fn dimensional_ecp_images_near_pair(
    mb: &MolBasis,
    geometry: &super::PeriodicGeometry,
    left: Vec3,
    right: Vec3,
    cutoff: f64,
) -> crate::Result<Vec<EcpImageKey>> {
    let mut selected = BTreeSet::new();
    for (ecp_index, ecp) in mb.ecps.iter().enumerate() {
        for center in [left, right] {
            for index in geometry.nearby_translation_indices(ecp.center - center, cutoff)? {
                selected.insert((ecp_index, index));
            }
        }
    }
    Ok(selected.into_iter().collect())
}

/// Short-range blocks `< mu(0) | O | nu(T) >`, retaining a shell pair when
/// `|R_nu + T - R_mu| <= rcut`.
///
/// The block for a shell pair depends only on the difference of the two
/// centres, so the translated partner is obtained by building a combined
/// basis (home cell plus shifted copy) and reading off the off-diagonal
/// quarter. That reuses the molecular integral routines exactly as they
/// are — no second code path to keep in step, which matters because a
/// periodic integral bug is far harder to see than a molecular one.
pub fn image_blocks(mb: &MolBasis, lattice: &Lattice, rcut: f64, op: Operator) -> Vec<ImageBlock> {
    selected_image_blocks(mb, short_range_images(mb, lattice, rcut), op)
}

/// Overlap or kinetic image blocks in any periodic subspace.
/// Only actual shell-pair separations are screened; open cell vectors are unused.
pub fn dimensional_image_blocks(
    mb: &MolBasis,
    geometry: &super::PeriodicGeometry,
    rcut: f64,
    op: Operator,
) -> crate::Result<Vec<ImageBlock>> {
    Ok(selected_image_blocks(
        mb,
        dimensional_short_range_images(mb, geometry, rcut)?,
        op,
    ))
}

fn selected_image_blocks(
    mb: &MolBasis,
    images: Vec<ShortRangeImage>,
    op: Operator,
) -> Vec<ImageBlock> {
    let n = mb.nao_sph;
    let mut out = Vec::new();
    let natoms = mb.shells.iter().map(|s| s.atom + 1).max().unwrap_or(0);
    for image in images {
        let t = image.translation;
        let comb = concat(mb, &shifted(mb, t), natoms);
        let full = match op {
            Operator::Overlap => crate::integrals::overlap_matrix(&comb),
            Operator::Kinetic => crate::integrals::kinetic_matrix(&comb),
        };
        let mut m = Matrix::zeros(n, n);
        let mut significant = false;
        for (left_index, right_index) in image.shell_pairs {
            let left = &mb.shells[left_index];
            let right = &mb.shells[right_index];
            for i in left.sph_off..left.sph_off + n_sph(left.l) {
                for j in right.sph_off..right.sph_off + n_sph(right.l) {
                    let v = full[(i, n + j)];
                    m[(i, j)] = v;
                    if v.abs() > 1e-14 {
                        significant = true;
                    }
                }
            }
        }
        // A translation whose block is numerically zero contributes
        // nothing to any k, so dropping it here saves work at every k.
        if significant {
            out.push(ImageBlock {
                translation: t,
                matrix: m,
            });
        }
    }
    out
}

/// Periodic semi-local ECP image blocks
/// `U^T_mn = <chi_m(0)|sum_L U_L|chi_n(T)>`.
///
/// Both the AO translation `T` and the ECP-center translation `L` are
/// explicit. Both are selected by actual centre separations for each shell
/// pair, rather than by distance to the coordinate origin; this makes the
/// finite lattice sum covariant under relabelling atoms into neighbouring
/// cells. Shell pairs with identical ECP image sets are evaluated together,
/// but an image selected for one pair never leaks into another pair.
/// The Gaussian radial factors make this an absolutely convergent cutoff,
/// unlike the ionic Coulomb operator handled by Ewald summation.
///
/// Nuclear derivatives use the same retained shell pairs and ECP images.
pub fn ecp_image_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    ao_cutoff: f64,
    ecp_cutoff: f64,
) -> crate::error::Result<Vec<ImageBlock>> {
    if mb.ecps.is_empty() {
        return Ok(Vec::new());
    }
    if !(ao_cutoff.is_finite() && ao_cutoff > 0.0 && ecp_cutoff.is_finite() && ecp_cutoff > 0.0) {
        return Err(crate::error::DftError::InvalidInput(
            "periodic AO and ECP image cutoffs must be finite and positive".to_string(),
        ));
    }
    selected_ecp_image_blocks(
        mb,
        lattice,
        short_range_images(mb, lattice, ao_cutoff),
        |left, right| Ok(ecp_images_near_pair(mb, lattice, left, right, ecp_cutoff)),
    )
}

/// Local/nonlocal ECP images in the physical 1D/2D/3D subspace. Open axes
/// have no AO images or ECP images, including when their embedding is skewed.
pub fn dimensional_ecp_image_blocks(
    mb: &MolBasis,
    geometry: &super::PeriodicGeometry,
    ao_cutoff: f64,
    ecp_cutoff: f64,
) -> crate::Result<Vec<ImageBlock>> {
    if mb.ecps.is_empty() {
        return Ok(Vec::new());
    }
    if !ao_cutoff.is_finite() || ao_cutoff <= 0.0 || !ecp_cutoff.is_finite() || ecp_cutoff <= 0.0 {
        return Err(crate::DftError::InvalidInput(
            "dimensional AO and ECP cutoffs must be finite and positive".into(),
        ));
    }
    let images = dimensional_short_range_images(mb, geometry, ao_cutoff)?;
    selected_ecp_image_blocks(mb, geometry.lattice(), images, |left, right| {
        dimensional_ecp_images_near_pair(mb, geometry, left, right, ecp_cutoff)
    })
}

fn selected_ecp_image_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    images: Vec<ShortRangeImage>,
    select: impl Fn(Vec3, Vec3) -> crate::Result<Vec<EcpImageKey>>,
) -> crate::Result<Vec<ImageBlock>> {
    let n = mb.nao_sph;
    let natoms = mb
        .shells
        .iter()
        .map(|shell| shell.atom + 1)
        .max()
        .unwrap_or(0);
    let mut out = Vec::new();
    for image in images {
        let t = image.translation;
        let base = concat(mb, &shifted(mb, t), natoms);
        let mut groups: BTreeMap<Vec<EcpImageKey>, Vec<(usize, usize)>> = BTreeMap::new();
        for (left_index, right_index) in image.shell_pairs {
            let left = &mb.shells[left_index];
            let right = &mb.shells[right_index];
            let ecp_images = select(left.center, right.center + t)?;
            if !ecp_images.is_empty() {
                groups
                    .entry(ecp_images)
                    .or_default()
                    .push((left_index, right_index));
            }
        }

        let mut matrix = Matrix::zeros(n, n);
        let mut significant = false;
        for (ecp_keys, shell_pairs) in groups {
            let mut combined = base.clone();
            combined.ecps = ecp_keys
                .into_iter()
                .map(|(ecp_index, translation)| {
                    let mut image = mb.ecps[ecp_index].clone();
                    image.center += lattice.translation(translation);
                    image
                })
                .collect();
            combined.n_core_electrons = mb.n_core_electrons;
            let full = crate::integrals::ecp_matrix(&combined)?;
            for (left_index, right_index) in shell_pairs {
                let left = &mb.shells[left_index];
                let right = &mb.shells[right_index];
                for i in left.sph_off..left.sph_off + n_sph(left.l) {
                    for j in right.sph_off..right.sph_off + n_sph(right.l) {
                        let value = full[(i, n + j)];
                        matrix[(i, j)] = value;
                        significant |= value.abs() > 1e-14;
                    }
                }
            }
        }
        if significant {
            out.push(ImageBlock {
                translation: t,
                matrix,
            });
        }
    }
    Ok(out)
}

/// One image block's derivative: `d S^T_mn / dR_A` for every atom `A` and
/// Cartesian axis, plus the translation it belongs to.
pub struct ImageDerivBlock {
    pub translation: Vec3,
    /// `[atom][axis]` matrices, `nao x nao`.
    pub d: Vec<[Matrix; 3]>,
}

/// Analytic derivatives of the retained periodic ECP image blocks.
///
/// The energy builder screens an AO shell pair and then chooses the ECP
/// images close to either AO centre. This routine repeats exactly that
/// discrete selection and differentiates only translations retained by the
/// energy operator. For one physical atom, derivatives of its home-cell AO,
/// its translated AO copy, and every translated ECP centre are summed.
pub(crate) fn ecp_derivative_image_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    ao_cutoff: f64,
    ecp_cutoff: f64,
    retained_energy_blocks: &[ImageBlock],
) -> Vec<ImageDerivBlock> {
    if mb.ecps.is_empty() || retained_energy_blocks.is_empty() {
        return Vec::new();
    }
    let n = mb.nao_sph;
    let natoms = mb
        .shells
        .iter()
        .map(|shell| shell.atom + 1)
        .max()
        .unwrap_or(0);
    let retained = |translation: Vec3| {
        retained_energy_blocks
            .iter()
            .any(|block| (block.translation - translation).norm() < 1e-10)
    };
    let mut out = Vec::new();
    for image in short_range_images(mb, lattice, ao_cutoff) {
        let t = image.translation;
        if !retained(t) {
            continue;
        }
        let base = concat(mb, &shifted(mb, t), natoms);
        let mut groups: BTreeMap<Vec<EcpImageKey>, Vec<(usize, usize)>> = BTreeMap::new();
        for (left_index, right_index) in image.shell_pairs {
            let left = &mb.shells[left_index];
            let right = &mb.shells[right_index];
            let ecp_images =
                ecp_images_near_pair(mb, lattice, left.center, right.center + t, ecp_cutoff);
            if !ecp_images.is_empty() {
                groups
                    .entry(ecp_images)
                    .or_default()
                    .push((left_index, right_index));
            }
        }

        let mut d: Vec<[Matrix; 3]> = (0..natoms)
            .map(|_| {
                [
                    Matrix::zeros(n, n),
                    Matrix::zeros(n, n),
                    Matrix::zeros(n, n),
                ]
            })
            .collect();
        let mut significant = false;
        for (ecp_keys, shell_pairs) in groups {
            let mut combined = base.clone();
            combined.ecps = ecp_keys
                .into_iter()
                .map(|(ecp_index, translation)| {
                    let mut ecp_image = mb.ecps[ecp_index].clone();
                    ecp_image.center += lattice.translation(translation);
                    ecp_image
                })
                .collect();
            let full = crate::integrals::ecp::ecp_derivative_matrices(&combined, 2 * natoms);
            for (left_index, right_index) in shell_pairs {
                let left = &mb.shells[left_index];
                let right = &mb.shells[right_index];
                for atom in 0..natoms {
                    for axis in 0..3 {
                        for i in left.sph_off..left.sph_off + n_sph(left.l) {
                            for j in right.sph_off..right.sph_off + n_sph(right.l) {
                                let value = full[atom][axis][(i, n + j)]
                                    + full[natoms + atom][axis][(i, n + j)];
                                d[atom][axis][(i, j)] = value;
                                significant |= value.abs() > 1e-14;
                            }
                        }
                    }
                }
            }
        }
        if significant {
            out.push(ImageDerivBlock { translation: t, d });
        }
    }
    out
}

/// Nuclear derivatives of the image blocks — the Pulay ingredient of a
/// periodic analytic gradient.
///
/// These sums are absolutely convergent, exactly like the blocks they
/// differentiate, so they need a cutoff and not Ewald splitting.
///
/// Each physical atom contributes twice to a given image block: once
/// through the home-cell shells centred on it, and once through the
/// shifted copy, since translating the lattice does not detach an image
/// from its parent. Both halves are collected here. The consequence worth
/// stating is that `d S^T / dR_A` summed over `A` is NOT zero for a fixed
/// `T` — only the sum over all atoms AND all translations vanishes, which
/// is what the test checks.
pub fn image_derivative_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    rcut: f64,
    op: Operator,
) -> Vec<ImageDerivBlock> {
    selected_image_derivative_blocks(mb, short_range_images(mb, lattice, rcut), op)
}

/// Analytic Cartesian atomic derivatives of dimensional overlap/kinetic blocks.
/// Physical atom motion includes both home and translated AO copies at fixed cell.
pub fn dimensional_image_derivative_blocks(
    mb: &MolBasis,
    geometry: &super::PeriodicGeometry,
    rcut: f64,
    op: Operator,
) -> crate::Result<Vec<ImageDerivBlock>> {
    Ok(selected_image_derivative_blocks(
        mb,
        dimensional_short_range_images(mb, geometry, rcut)?,
        op,
    ))
}

fn selected_image_derivative_blocks(
    mb: &MolBasis,
    images: Vec<ShortRangeImage>,
    op: Operator,
) -> Vec<ImageDerivBlock> {
    use crate::integrals::deriv_mat::{kinetic_derivative_matrices, overlap_derivative_matrices};
    let n = mb.nao_sph;
    let natoms = mb.shells.iter().map(|s| s.atom + 1).max().unwrap_or(0);
    let mut out = Vec::new();
    for image in images {
        let t = image.translation;
        let comb = concat(mb, &shifted(mb, t), natoms);
        let full = match op {
            Operator::Overlap => overlap_derivative_matrices(&comb, 2 * natoms),
            Operator::Kinetic => kinetic_derivative_matrices(&comb, 2 * natoms),
        };
        let mut d: Vec<[Matrix; 3]> = (0..natoms)
            .map(|_| {
                [
                    Matrix::zeros(n, n),
                    Matrix::zeros(n, n),
                    Matrix::zeros(n, n),
                ]
            })
            .collect();
        let mut significant = false;
        for (left_index, right_index) in image.shell_pairs {
            let left = &mb.shells[left_index];
            let right = &mb.shells[right_index];
            for a in 0..natoms {
                for ax in 0..3 {
                    for i in left.sph_off..left.sph_off + n_sph(left.l) {
                        for j in right.sph_off..right.sph_off + n_sph(right.l) {
                            // Home-cell copy and shifted copy of the same atom.
                            let v = full[a][ax][(i, n + j)] + full[natoms + a][ax][(i, n + j)];
                            d[a][ax][(i, j)] = v;
                            if v.abs() > 1e-14 {
                                significant = true;
                            }
                        }
                    }
                }
            }
        }
        if significant {
            out.push(ImageDerivBlock { translation: t, d });
        }
    }
    out
}

/// Bloch transform of the derivative blocks: `dS(k)/dR_A` as
/// `(real, imaginary)` per atom and axis.
pub fn bloch_derivative(
    blocks: &[ImageDerivBlock],
    k: Vec3,
    natoms: usize,
    nao: usize,
) -> Vec<[(Matrix, Matrix); 3]> {
    let mut out: Vec<[(Matrix, Matrix); 3]> = (0..natoms)
        .map(|_| {
            [
                (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)),
                (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)),
                (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)),
            ]
        })
        .collect();
    for b in blocks {
        let phase = k.dot(b.translation);
        let (c, s) = (phase.cos(), phase.sin());
        for a in 0..natoms.min(b.d.len()) {
            for ax in 0..3 {
                let src = &b.d[a][ax];
                let (re, im) = &mut out[a][ax];
                for (o, v) in re.as_mut_slice().iter_mut().zip(src.as_slice()) {
                    *o += c * v;
                }
                for (o, v) in im.as_mut_slice().iter_mut().zip(src.as_slice()) {
                    *o += s * v;
                }
            }
        }
    }
    out
}

/// Gamma-point sum `sum_T S^T` — real, and the k = 0 case of [`bloch`].
pub fn gamma_sum(blocks: &[ImageBlock]) -> Matrix {
    let n = blocks.first().map(|b| b.matrix.rows).unwrap_or(0);
    let mut out = Matrix::zeros(n, n);
    for b in blocks {
        for (o, v) in out.as_mut_slice().iter_mut().zip(b.matrix.as_slice()) {
            *o += *v;
        }
    }
    out
}

/// Bloch transform `S(k) = sum_T exp(i k . T) S^T`, as `(real, imaginary)`.
///
/// Hermitian: the real part is symmetric and the imaginary part
/// antisymmetric, provided the image set is closed under `T -> -T` — which
/// [`Lattice::translations_within`] guarantees, since it enumerates a
/// symmetric window.
pub fn bloch(blocks: &[ImageBlock], k: Vec3) -> (Matrix, Matrix) {
    let n = blocks.first().map(|b| b.matrix.rows).unwrap_or(0);
    let (mut re, mut im) = (Matrix::zeros(n, n), Matrix::zeros(n, n));
    for b in blocks {
        let phase = k.dot(b.translation);
        let (c, s) = (phase.cos(), phase.sin());
        for (o, v) in re.as_mut_slice().iter_mut().zip(b.matrix.as_slice()) {
            *o += c * v;
        }
        for (o, v) in im.as_mut_slice().iter_mut().zip(b.matrix.as_slice()) {
            *o += s * v;
        }
    }
    (re, im)
}

/// Real symmetric `2n x 2n` embedding of a Hermitian `(Re, Im)` pair.
///
/// Its eigenvalues are those of the Hermitian matrix, each twice over, so
/// an ordinary symmetric eigensolver answers spectral questions about
/// `S(k)` without a complex layer.
pub fn hermitian_as_real(re: &Matrix, im: &Matrix) -> Matrix {
    let n = re.rows;
    let mut m = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = re[(i, j)];
            m[(n + i, n + j)] = re[(i, j)];
            m[(i, n + j)] = -im[(i, j)];
            m[(n + i, j)] = im[(i, j)];
        }
    }
    m
}

/// Monkhorst-Pack k-point mesh (unshifted), in Cartesian reciprocal units.
pub fn monkhorst_pack(lattice: &Lattice, mesh: [usize; 3]) -> Vec<Vec3> {
    let b = lattice.reciprocal();
    let mut out = Vec::new();
    for i in 0..mesh[0] {
        for j in 0..mesh[1] {
            for k in 0..mesh[2] {
                let f = [
                    i as f64 / mesh[0] as f64,
                    j as f64 / mesh[1] as f64,
                    k as f64 / mesh[2] as f64,
                ];
                out.push(b[0] * f[0] + b[1] * f[1] + b[2] * f[2]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::linalg::symmetric_eigen;
    use crate::system::{Atom, Molecule};

    fn h2_basis(sep: f64) -> (Molecule, MolBasis) {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, sep),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb)
    }

    /// A cell far larger than the basis extent must reproduce the MOLECULAR
    /// integrals exactly: every image block but `T = 0` is negligible. This
    /// is the limit that ties the periodic path to the already-verified
    /// molecular one.
    #[test]
    fn large_cell_reproduces_the_molecular_matrices() {
        let (_, mb) = h2_basis(1.4);
        let lat = Lattice::cubic(60.0);
        for (op, want) in [
            (Operator::Overlap, crate::integrals::overlap_matrix(&mb)),
            (Operator::Kinetic, crate::integrals::kinetic_matrix(&mb)),
        ] {
            let blocks = image_blocks(&mb, &lat, 25.0, op);
            let got = gamma_sum(&blocks);
            assert!(
                got.max_abs_diff(&want) < 1e-10,
                "{op:?}: periodic Gamma sum differs from the molecular matrix"
            );
        }
    }

    /// `S(k)` must be Hermitian at every k, and reduce to the real Gamma
    /// sum at k = 0. A missing `T -> -T` partner shows up here as a
    /// non-vanishing imaginary part.
    #[test]
    fn bloch_matrices_are_hermitian_and_reduce_at_gamma() {
        let (_, mb) = h2_basis(1.4);
        let lat = Lattice::cubic(5.0);
        let blocks = image_blocks(&mb, &lat, 18.0, Operator::Overlap);
        let n = mb.nao_sph;

        let (re0, im0) = bloch(&blocks, Vec3::zero());
        assert!(
            im0.as_slice().iter().all(|v| v.abs() < 1e-13),
            "the k = 0 imaginary part must vanish"
        );
        assert!(re0.max_abs_diff(&gamma_sum(&blocks)) < 1e-13);

        let b = lat.reciprocal();
        for k in [b[0] * 0.5, b[0] * 0.3 + b[1] * 0.2, b[2] * 0.25] {
            let (re, im) = bloch(&blocks, k);
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (re[(i, j)] - re[(j, i)]).abs() < 1e-13,
                        "Re S(k) is not symmetric"
                    );
                    assert!(
                        (im[(i, j)] + im[(j, i)]).abs() < 1e-13,
                        "Im S(k) is not antisymmetric"
                    );
                }
            }
        }
    }

    fn assert_bloch_gauge_transform(
        original: &(Matrix, Matrix),
        relabelled: &(Matrix, Matrix),
        k: Vec3,
        ao_shifts: &[Vec3],
        tolerance: f64,
    ) {
        let n = original.0.rows;
        assert_eq!(ao_shifts.len(), n);
        for i in 0..n {
            for j in 0..n {
                // Moving AO mu by lattice vector q_mu changes a Bloch matrix
                // by D^dagger A D, with element phase
                // exp(i k . (q_mu - q_nu)).
                let phase = k.dot(ao_shifts[i] - ao_shifts[j]);
                let (c, s) = (phase.cos(), phase.sin());
                let expected_re = c * original.0[(i, j)] - s * original.1[(i, j)];
                let expected_im = s * original.0[(i, j)] + c * original.1[(i, j)];
                assert!(
                    (relabelled.0[(i, j)] - expected_re).abs() < tolerance,
                    "real Bloch-gauge mismatch at ({i},{j}): {} vs {expected_re}",
                    relabelled.0[(i, j)]
                );
                assert!(
                    (relabelled.1[(i, j)] - expected_im).abs() < tolerance,
                    "imaginary Bloch-gauge mismatch at ({i},{j}): {} vs {expected_im}",
                    relabelled.1[(i, j)]
                );
            }
        }
    }

    /// Two AOs close across a cell face must interact even when the required
    /// translation is much longer than the centre-distance cutoff. Relabelling
    /// one AO by a lattice vector is only a diagonal Bloch-gauge change, for
    /// both S/T and the analytic nuclear derivatives.
    #[test]
    fn short_range_integrals_are_covariant_under_cell_relabelling() {
        let a = 8.0;
        let lattice = Lattice::cubic(a);
        let set = data::load("sto-3g").unwrap();
        let original_molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.2, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(a - 0.2, 0.0, 0.0),
            },
        ]);
        let mut relabelled_molecule = original_molecule.clone();
        relabelled_molecule.atoms[1].position.x -= a;
        let original_basis = MolBasis::build(&original_molecule, &set).unwrap();
        let relabelled_basis = MolBasis::build(&relabelled_molecule, &set).unwrap();
        let cutoff = 1.0;
        let k = lattice.reciprocal()[0] * 0.37;
        let shifts = [Vec3::zero(), -lattice.a[0]];

        // The local candidate construction stays the same size even though
        // the actual translation labels move from +/-a to zero.
        assert_eq!(
            short_range_images(&original_basis, &lattice, cutoff).len(),
            3
        );
        assert_eq!(
            short_range_images(&relabelled_basis, &lattice, cutoff).len(),
            1
        );

        for op in [Operator::Overlap, Operator::Kinetic] {
            let original_blocks = image_blocks(&original_basis, &lattice, cutoff, op);
            assert!(
                original_blocks
                    .iter()
                    .any(|block| block.translation.norm() > cutoff),
                "{op:?}: the across-boundary AO pair was not selected"
            );
            let relabelled_blocks = image_blocks(&relabelled_basis, &lattice, cutoff, op);
            let original = bloch(&original_blocks, k);
            let relabelled = bloch(&relabelled_blocks, k);
            assert_bloch_gauge_transform(&original, &relabelled, k, &shifts, 3e-12);

            let original_derivatives = bloch_derivative(
                &image_derivative_blocks(&original_basis, &lattice, cutoff, op),
                k,
                2,
                2,
            );
            let relabelled_derivatives = bloch_derivative(
                &image_derivative_blocks(&relabelled_basis, &lattice, cutoff, op),
                k,
                2,
                2,
            );
            for atom in 0..2 {
                for axis in 0..3 {
                    assert_bloch_gauge_transform(
                        &original_derivatives[atom][axis],
                        &relabelled_derivatives[atom][axis],
                        k,
                        &shifts,
                        8e-11,
                    );
                }
            }
        }
    }

    /// BAND FOLDING. A Gamma-point calculation on a 2x1x1 supercell must
    /// have exactly the spectrum of the primitive cell sampled at k = 0 and
    /// k = pi/a together.
    ///
    /// This is the sharpest oracle-free check on the whole periodic
    /// construction: it ties the real-space lattice sum, the Bloch phases
    /// and the reciprocal lattice to one another. An error in any of them
    /// breaks the correspondence, and none of the cheaper identities
    /// (hermiticity, the molecular limit) would notice.
    #[test]
    fn supercell_gamma_spectrum_folds_the_primitive_k_points() {
        let a = 4.2;
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let prim = Lattice::new(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, 12.0, 0.0),
            Vec3::new(0.0, 0.0, 12.0),
        );
        let rcut = 30.0;
        let blocks = image_blocks(&mb, &prim, rcut, Operator::Overlap);

        // Primitive spectrum at the two k points the 2x1 supercell folds.
        let mut prim_eigs = Vec::new();
        for k in monkhorst_pack(&prim, [2, 1, 1]) {
            let (re, im) = bloch(&blocks, k);
            let (vals, _) = symmetric_eigen(&hermitian_as_real(&re, &im)).unwrap();
            // The real embedding doubles every eigenvalue.
            for (i, v) in vals.iter().enumerate() {
                if i % 2 == 0 {
                    prim_eigs.push(*v);
                }
            }
        }
        prim_eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());

        // Supercell: two atoms in a doubled cell, Gamma point.
        let sc_mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(a, 0.0, 0.0),
            },
        ]);
        let sc_mb = MolBasis::build(&sc_mol, &set).unwrap();
        let sc = prim.supercell([2, 1, 1]);
        let sc_blocks = image_blocks(&sc_mb, &sc, rcut, Operator::Overlap);
        let (sre, sim) = bloch(&sc_blocks, Vec3::zero());
        let (svals, _) = symmetric_eigen(&hermitian_as_real(&sre, &sim)).unwrap();
        let mut sc_eigs: Vec<f64> = svals
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, v)| *v)
            .collect();
        sc_eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());

        assert_eq!(prim_eigs.len(), sc_eigs.len(), "spectrum sizes differ");
        for (p, s) in prim_eigs.iter().zip(&sc_eigs) {
            assert!(
                (p - s).abs() < 1e-9,
                "band folding failed: primitive {p} vs supercell {s}"
            );
        }
    }

    /// The derivative image blocks against central differences of the
    /// Gamma-point matrices, and the translational sum rule.
    ///
    /// The sum rule has a subtlety worth pinning: `dS^T/dR_A` summed over
    /// atoms is NOT zero for a fixed `T`, because moving one atom changes
    /// the distance to the images of the others. Only the sum over atoms
    /// of the FULL lattice sum vanishes — displacing everything together
    /// leaves every pair distance alone. Testing the wrong one of those
    /// would look like a bug in correct code.
    #[test]
    fn derivative_blocks_match_finite_differences() {
        let a = 4.6;
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.3, 0.2, 1.5),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let lat = Lattice::new(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.4, a, 0.0),
            Vec3::new(0.0, 0.2, a),
        );
        let rcut = 16.0;
        let n = mb.nao_sph;

        for op in [Operator::Overlap, Operator::Kinetic] {
            let dblocks = image_derivative_blocks(&mb, &lat, rcut, op);
            // Gamma-point derivative = sum over translations.
            let dk = bloch_derivative(&dblocks, Vec3::zero(), 2, n);

            let h = 1e-5;
            let mut worst = 0.0_f64;
            for at in 0..2 {
                for ax in 0..3 {
                    let at_shift = |s: f64| -> Matrix {
                        let mut m = mol.clone();
                        match ax {
                            0 => m.atoms[at].position.x += s,
                            1 => m.atoms[at].position.y += s,
                            _ => m.atoms[at].position.z += s,
                        }
                        let mbx = MolBasis::build(&m, &set).unwrap();
                        gamma_sum(&image_blocks(&mbx, &lat, rcut, op))
                    };
                    let (p, q) = (at_shift(h), at_shift(-h));
                    for i in 0..n {
                        for j in 0..n {
                            let fd = (p[(i, j)] - q[(i, j)]) / (2.0 * h);
                            worst = worst.max((dk[at][ax].0[(i, j)] - fd).abs());
                        }
                    }
                }
            }
            assert!(worst < 1e-7, "{op:?}: derivative vs FD, worst {worst:e}");

            // Sum over atoms of the FULL lattice sum must vanish.
            for ax in 0..3 {
                let mut acc = Matrix::zeros(n, n);
                for at in 0..2 {
                    for (o, v) in acc.as_mut_slice().iter_mut().zip(dk[at][ax].0.as_slice()) {
                        *o += *v;
                    }
                }
                let worst_sum = acc.as_slice().iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                assert!(
                    worst_sum < 1e-10,
                    "{op:?} axis {ax}: translational sum rule violated ({worst_sum:e})"
                );
            }
        }
    }

    /// `dS(k)/dR` must be Hermitian in the same way `S(k)` is: real part
    /// symmetric, imaginary part antisymmetric, at every k.
    #[test]
    fn derivative_bloch_matrices_are_hermitian() {
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
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let lat = Lattice::cubic(5.0);
        let blocks = image_derivative_blocks(&mb, &lat, 16.0, Operator::Overlap);
        let n = mb.nao_sph;
        let b = lat.reciprocal();
        for k in [b[0] * 0.5, b[0] * 0.3 + b[2] * 0.2] {
            let dk = bloch_derivative(&blocks, k, 2, n);
            for at in 0..2 {
                for ax in 0..3 {
                    let (re, im) = &dk[at][ax];
                    for i in 0..n {
                        for j in 0..n {
                            assert!((re[(i, j)] - re[(j, i)]).abs() < 1e-12);
                            assert!((im[(i, j)] + im[(j, i)]).abs() < 1e-12);
                        }
                    }
                }
            }
        }
    }

    /// Every Monkhorst-Pack point must lie on the reciprocal lattice mesh,
    /// and the mesh must have the requested size.
    #[test]
    fn monkhorst_pack_mesh_is_on_the_reciprocal_lattice() {
        let lat = Lattice::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(1.1, 3.9, 0.0),
            Vec3::new(0.2, -0.5, 5.0),
        );
        let mesh = [2usize, 3, 2];
        let ks = monkhorst_pack(&lat, mesh);
        assert_eq!(ks.len(), mesh.iter().product::<usize>());
        // k . a_i / (2 pi) must be an integer multiple of 1/mesh_i.
        for k in &ks {
            for i in 0..3 {
                let f = k.dot(lat.a[i]) / (2.0 * std::f64::consts::PI);
                let scaled = f * mesh[i] as f64;
                assert!(
                    (scaled - scaled.round()).abs() < 1e-10,
                    "k point off the mesh along a{i}: {f}"
                );
            }
        }
    }

    /// Relabelling an ECP atom and all of its AOs into a neighbouring cell
    /// must only change the AO Bloch gauge. NaH makes that transformation
    /// non-trivial while using just one LANL2DZ pseudopotential centre.
    #[test]
    fn periodic_ecp_is_covariant_when_an_ecp_atom_changes_cell() {
        let a = 8.0;
        let lattice = Lattice::cubic(a);
        let set = data::load("lanl2dz").unwrap();
        let original_molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.3, 0.0, 0.0),
            },
            Atom {
                z: 11,
                position: Vec3::new(a - 0.3, 0.0, 0.0),
            },
        ]);
        let mut relabelled_molecule = original_molecule.clone();
        relabelled_molecule.atoms[1].position.x -= a;
        let original_basis = MolBasis::build(&original_molecule, &set).unwrap();
        let relabelled_basis = MolBasis::build(&relabelled_molecule, &set).unwrap();
        assert_eq!(original_basis.ecps.len(), 1);
        assert_eq!(relabelled_basis.ecps.len(), 1);

        let ao_cutoff = 1.2;
        let ecp_cutoff = 1.5;
        let original_blocks =
            ecp_image_blocks(&original_basis, &lattice, ao_cutoff, ecp_cutoff).unwrap();
        let relabelled_blocks =
            ecp_image_blocks(&relabelled_basis, &lattice, ao_cutoff, ecp_cutoff).unwrap();
        assert!(
            original_blocks
                .iter()
                .any(|block| block.translation.norm() > ao_cutoff),
            "the ECP-coupled AO pair across the cell face was omitted"
        );

        let k = lattice.reciprocal()[0] * 0.31;
        let original = bloch(&original_blocks, k);
        let relabelled = bloch(&relabelled_blocks, k);
        let shifts: Vec<Vec3> = original_basis
            .sph_ao_atoms()
            .into_iter()
            .map(|atom| {
                if atom == 1 {
                    -lattice.a[0]
                } else {
                    Vec3::zero()
                }
            })
            .collect();
        assert_bloch_gauge_transform(&original, &relabelled, k, &shifts, 2e-10);

        // The same block selection must retain U(-T) = U(T)^T, expressed
        // here as Hermiticity of its Bloch transform.
        for matrix in [&original, &relabelled] {
            let n = matrix.0.rows;
            for i in 0..n {
                for j in 0..n {
                    assert!((matrix.0[(i, j)] - matrix.0[(j, i)]).abs() < 2e-10);
                    assert!((matrix.1[(i, j)] + matrix.1[(j, i)]).abs() < 2e-10);
                }
            }
        }
    }

    #[test]
    fn periodic_ecp_derivative_blocks_match_finite_difference_at_finite_k() {
        let a = 8.0;
        let lattice = Lattice::cubic(a);
        let set = data::load("lanl2dz").unwrap();
        let reference = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.3, 0.0, 0.0),
            },
            Atom {
                z: 11,
                position: Vec3::new(a - 0.3, 0.0, 0.0),
            },
        ]);
        let ao_cutoff = 1.2;
        let ecp_cutoff = 1.5;
        let k = lattice.reciprocal()[0] * 0.31 + lattice.reciprocal()[2] * 0.17;
        let basis = MolBasis::build(&reference, &set).unwrap();
        let energy_blocks = ecp_image_blocks(&basis, &lattice, ao_cutoff, ecp_cutoff).unwrap();
        let derivative_blocks =
            ecp_derivative_image_blocks(&basis, &lattice, ao_cutoff, ecp_cutoff, &energy_blocks);
        let analytic = bloch_derivative(&derivative_blocks, k, 2, basis.nao_sph);
        let matrix_at = |atom: usize, axis: usize, amount: f64| {
            let mut molecule = reference.clone();
            match axis {
                0 => molecule.atoms[atom].position.x += amount,
                1 => molecule.atoms[atom].position.y += amount,
                _ => molecule.atoms[atom].position.z += amount,
            }
            let displaced_basis = MolBasis::build(&molecule, &set).unwrap();
            let blocks =
                ecp_image_blocks(&displaced_basis, &lattice, ao_cutoff, ecp_cutoff).unwrap();
            bloch(&blocks, k)
        };
        let step = 1e-4;
        let mut worst = 0.0_f64;
        for atom in 0..2 {
            for axis in 0..3 {
                let plus = matrix_at(atom, axis, step);
                let minus = matrix_at(atom, axis, -step);
                for i in 0..basis.nao_sph {
                    for j in 0..basis.nao_sph {
                        let fd_re = (plus.0[(i, j)] - minus.0[(i, j)]) / (2.0 * step);
                        let fd_im = (plus.1[(i, j)] - minus.1[(i, j)]) / (2.0 * step);
                        worst = worst
                            .max((analytic[atom][axis].0[(i, j)] - fd_re).abs())
                            .max((analytic[atom][axis].1[(i, j)] - fd_im).abs());
                    }
                }
            }
        }
        assert!(
            worst < 2e-6,
            "periodic ECP derivative matrix differs from finite differences by {worst:e}"
        );

        // A common translation moves both AO copies and every ECP image;
        // the operator cannot change.
        for axis in 0..3 {
            for i in 0..basis.nao_sph {
                for j in 0..basis.nao_sph {
                    let sum_re = analytic[0][axis].0[(i, j)] + analytic[1][axis].0[(i, j)];
                    let sum_im = analytic[0][axis].1[(i, j)] + analytic[1][axis].1[(i, j)];
                    assert!(sum_re.abs() < 2e-10);
                    assert!(sum_im.abs() < 2e-10);
                }
            }
        }
    }

    /// With a cell larger than both cutoffs, the only AO and ECP image is
    /// the home cell.  The periodic builder must then be exactly the already
    /// oracle-gated molecular semi-local ECP matrix.
    #[test]
    fn periodic_ecp_home_block_reuses_the_molecular_operator() {
        let molecule = Molecule::new(vec![Atom {
            z: 11,
            position: Vec3::zero(),
        }]);
        let set = data::load("lanl2dz").unwrap();
        let basis = MolBasis::build(&molecule, &set).unwrap();
        assert_eq!(basis.ecps.len(), 1);
        let blocks = ecp_image_blocks(&basis, &Lattice::cubic(40.0), 8.0, 8.0).unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].translation.norm() < 1e-14);
        let molecular = crate::integrals::ecp_matrix(&basis).unwrap();
        assert!(
            blocks[0].matrix.max_abs_diff(&molecular) < 2e-12,
            "periodic home ECP block differs from the molecular operator"
        );
    }
}
