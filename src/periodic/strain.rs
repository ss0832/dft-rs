// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Strain derivatives of the periodic one-electron energy terms at a fixed
//! AO density.
//!
//! A homogeneous strain `epsilon` maps every atom, every lattice vector and
//! therefore every basis-function and nuclear image centre `X` to
//! `(1 + epsilon) X`, and every reciprocal vector `G` to
//! `(1 + epsilon)^-T G`. Gaussian functions move rigidly with their centres.
//! The density images `D(T)` are labelled by integer translations and the
//! Bloch phases `k . T` are invariant, so at fixed `D(T)` the strain
//! derivative of a lattice sum is
//!
//! ```text
//! dE/d(epsilon_ab) = sum_X (dE/dX_a) X_b                 (centre virial)
//!                  - sum_G G_a dE/dG_b                    (reciprocal sums)
//!                  + explicit volume terms of 1/V prefactors.
//! ```
//!
//! For a translation-invariant term the centre virial may be referred to
//! any origin; real-space terms use relative coordinates, reciprocal terms
//! the absolute centres together with the matching `G` derivative
//! `d rho_ab(G)/dG_b = i <a| r_b exp(iG.r) |b>`.

use crate::basis::aos::cart_to_sph;
use crate::basis::{cart_components, MolBasis, MolShell};
use crate::dual::Scalar;
use crate::integrals::deriv::{
    combine_bra_derivative, kinetic_kind, lowered, nuclear_block_kernel, overlap_kind, pair_block,
    raised,
};
use crate::integrals::hermite::ETable;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::ewald::EwaldParams;
use crate::periodic::integrals::{short_range_images, ImageBlock, Operator};
use crate::periodic::planewave::{contracted_pair_blocks, ReciprocalContraction};
use crate::periodic::Lattice;
use crate::system::Molecule;
use rayon::prelude::*;

/// A 3x3 strain-derivative tensor `[a][b]`.
pub type Tensor = [[f64; 3]; 3];

pub(crate) fn add_tensor(target: &mut Tensor, source: &Tensor) {
    for a in 0..3 {
        for b in 0..3 {
            target[a][b] += source[a][b];
        }
    }
}

fn sum_tensors(parts: impl ParallelIterator<Item = Tensor>) -> Tensor {
    parts.reduce(
        || [[0.0; 3]; 3],
        |mut a, b| {
            add_tensor(&mut a, &b);
            a
        },
    )
}

pub(crate) fn find_block(density: &[ImageBlock], translation: Vec3) -> Option<&Matrix> {
    density
        .iter()
        .find(|block| (block.translation - translation).norm() < 1e-8)
        .map(|block| &block.matrix)
}

/// The spherical density block of a shell pair moved to the Cartesian
/// components of the two shells, so that `sum D_sph . sph(M) = sum D_cart . M`.
pub(crate) fn cartesian_density(d: &Matrix, sa: &MolShell, sb: &MolShell) -> Matrix {
    cartesian_density_with(sa, sb, |i, j| d[(sa.sph_off + i, sb.sph_off + j)])
}

pub(crate) fn cartesian_density_with(
    sa: &MolShell,
    sb: &MolShell,
    entry: impl Fn(usize, usize) -> f64,
) -> Matrix {
    let (na, nb) = (crate::basis::n_sph(sa.l), crate::basis::n_sph(sb.l));
    let mut block = Matrix::zeros(na, nb);
    for i in 0..na {
        for j in 0..nb {
            block[(i, j)] = entry(i, j);
        }
    }
    cart_to_sph(sa.l)
        .matmul_seq(&block)
        .matmul_seq(&cart_to_sph(sb.l).transpose())
}

pub(crate) fn contract(a: &Matrix, b: &Matrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x * y)
        .sum()
}

/// `(x - B)_b` times the shell: angular momentum raised with unchanged
/// coefficients, so that component `j + 1_b` of the raised shell is the
/// first moment of component `j`.
pub(crate) fn moment_raised(sh: &MolShell) -> MolShell {
    let mut out = sh.clone();
    out.l += 1;
    out.sph_off = 0;
    out.cart_off = 0;
    out
}

fn raised_component(l: u8, component: (i32, i32, i32), axis: usize) -> usize {
    let up = match axis {
        0 => (component.0 + 1, component.1, component.2),
        1 => (component.0, component.1 + 1, component.2),
        _ => (component.0, component.1, component.2 + 1),
    };
    cart_components(l + 1)
        .iter()
        .position(|&c| c == up)
        .expect("raised Cartesian component exists")
}

/// Cartesian derivative blocks of the real part of `sum_G c(G) rho_ab(G)`
/// with respect to the bra centre `A` (index 0) and the ket centre `B`
/// (index 1).
pub(crate) fn pair_centre_derivatives(
    sa: &MolShell,
    sb: &MolShell,
    gs: &[Vec3],
    c: (&[f64], &[f64]),
) -> [[Matrix; 3]; 2] {
    let reciprocal = ReciprocalContraction::new(gs, &[c.0], &[c.1]);
    pair_centre_derivatives_with(sa, sb, &reciprocal)
}

pub(crate) fn pair_centre_derivatives_with(
    sa: &MolShell,
    sb: &MolShell,
    reciprocal: &ReciprocalContraction<'_>,
) -> [[Matrix; 3]; 2] {
    let block = |x: &MolShell, y: &MolShell| reciprocal.blocks(x, y).remove(0).0;
    let bra = (block(&raised(sa), sb), lowered(sa).map(|lo| block(&lo, sb)));
    let ket = (block(&raised(sb), sa), lowered(sb).map(|lo| block(&lo, sa)));
    let a: [Matrix; 3] =
        std::array::from_fn(|ax| combine_bra_derivative(sa.l, sb.l, ax, &bra.0, bra.1.as_ref()));
    let b: [Matrix; 3] = std::array::from_fn(|ax| {
        combine_bra_derivative(sb.l, sa.l, ax, &ket.0, ket.1.as_ref()).transpose()
    });
    [a, b]
}

/// Cartesian blocks `[a][b]` of the centre virial `sum_X (d/dX_a) X_b` of
/// `Re sum_G c(G) rho_ab(G)` over the two absolute pair centres.
pub(crate) fn reciprocal_centre_virial_blocks(
    sa: &MolShell,
    sb: &MolShell,
    gs: &[Vec3],
    c: (&[f64], &[f64]),
) -> [[Matrix; 3]; 3] {
    let derivative = pair_centre_derivatives(sa, sb, gs, c);
    std::array::from_fn(|axis| {
        std::array::from_fn(|b| {
            let mut m = derivative[0][axis].clone();
            let (wa, wb) = (sa.center.get(b), sb.center.get(b));
            for (x, y) in m
                .as_mut_slice()
                .iter_mut()
                .zip(derivative[1][axis].as_slice())
            {
                *x = *x * wa + y * wb;
            }
            m
        })
    })
}

/// Centre virial `sum_X (d/dX_a) X_b` of `Re sum_G c(G) D . rho_ab(G)` over
/// the two absolute pair centres.
pub(crate) fn reciprocal_centre_virial(
    sa: &MolShell,
    sb: &MolShell,
    gs: &[Vec3],
    c: (&[f64], &[f64]),
    d_cart: &Matrix,
) -> Tensor {
    let blocks = reciprocal_centre_virial_blocks(sa, sb, gs, c);
    std::array::from_fn(|a| std::array::from_fn(|b| contract(&blocks[a][b], d_cart)))
}

/// Cartesian blocks `[a][b]` of the wave-vector strain derivative of
/// `Re sum_G c(G) rho_ab(G)` at fixed coefficients and centres (see
/// [`reciprocal_wavevector_strain`]).
pub(crate) fn reciprocal_wavevector_blocks(
    sa: &MolShell,
    sb: &MolShell,
    gs: &[Vec3],
    c: (&[f64], &[f64]),
) -> [[Matrix; 3]; 3] {
    let [re, im] = reciprocal_wavevector_coefficients(gs, c);
    let reciprocal = ReciprocalContraction::new(
        gs,
        &re.each_ref().map(Vec::as_slice),
        &im.each_ref().map(Vec::as_slice),
    );
    let moment = reciprocal.blocks(sa, &moment_raised(sb));
    let plain = reciprocal.blocks(sa, sb);
    let ca = cart_components(sa.l);
    let cb = cart_components(sb.l);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let mut m = Matrix::zeros(ca.len(), cb.len());
            for i in 0..ca.len() {
                for (j, &component) in cb.iter().enumerate() {
                    m[(i, j)] = plain[a].0[(i, j)] * sb.center.get(b)
                        + moment[a].0[(i, raised_component(sb.l, component, b))];
                }
            }
            m
        })
    })
}

/// Strain derivative of `Re sum_G c(G) D . rho_ab(G)` through the
/// wave vectors at fixed coefficients and centres:
/// `-sum_G G_a c(G) d rho/dG_b` with `d rho/dG_b = i[rho^(b) + B_b rho]`,
/// where `rho^(b)` carries the first moment `(x - B)_b` on the ket.
pub(crate) fn reciprocal_wavevector_strain(
    sa: &MolShell,
    sb: &MolShell,
    gs: &[Vec3],
    c: (&[f64], &[f64]),
    d_cart: &Matrix,
) -> Tensor {
    let [re, im] = reciprocal_wavevector_coefficients(gs, c);
    let reciprocal = ReciprocalContraction::new(
        gs,
        &re.each_ref().map(Vec::as_slice),
        &im.each_ref().map(Vec::as_slice),
    );
    reciprocal_wavevector_strain_with(sa, sb, &reciprocal, d_cart)
}

/// The three coefficient sets c'_a(G) = -i G_a c(G), independent of the
/// shell pair being contracted.
pub(crate) fn reciprocal_wavevector_coefficients(
    gs: &[Vec3],
    c: (&[f64], &[f64]),
) -> [[Vec<f64>; 3]; 2] {
    [
        std::array::from_fn(|a| gs.iter().zip(c.1).map(|(g, ci)| g.get(a) * ci).collect()),
        std::array::from_fn(|a| gs.iter().zip(c.0).map(|(g, cr)| -g.get(a) * cr).collect()),
    ]
}

pub(crate) fn reciprocal_wavevector_strain_with(
    sa: &MolShell,
    sb: &MolShell,
    reciprocal: &ReciprocalContraction<'_>,
    d_cart: &Matrix,
) -> Tensor {
    let moment = reciprocal.blocks(sa, &moment_raised(sb));
    let plain = reciprocal.blocks(sa, sb);
    let ca = cart_components(sa.l);
    let cb = cart_components(sb.l);
    let mut out = [[0.0; 3]; 3];
    for a in 0..3 {
        let base = contract(&plain[a].0, d_cart);
        for b in 0..3 {
            let mut value = base * sb.center.get(b);
            for i in 0..ca.len() {
                for (j, &component) in cb.iter().enumerate() {
                    let dij = d_cart[(i, j)];
                    if dij != 0.0 {
                        value += dij * moment[a].0[(i, raised_component(sb.l, component, b))];
                    }
                }
            }
            out[a][b] += value;
        }
    }
    out
}

pub(crate) type PairKind =
    fn(&ETable, &ETable, &ETable, f64, f64, (i32, i32, i32), (i32, i32, i32)) -> f64;

/// Bra-centre derivative blocks (Cartesian) of a translation-invariant
/// two-centre operator.
pub(crate) fn bra_derivative(sa: &MolShell, sb: &MolShell, kind: PairKind) -> [Matrix; 3] {
    let up = pair_block(&raised(sa), sb, kind);
    let down = lowered(sa).map(|lo| pair_block(&lo, sb, kind));
    std::array::from_fn(|ax| combine_bra_derivative(sa.l, sb.l, ax, &up, down.as_ref()))
}

pub(crate) fn shifted_shell(sh: &MolShell, t: Vec3) -> MolShell {
    let mut out = sh.clone();
    out.center += t;
    out
}

/// Strain derivative `sum_T D(T) . dO(T)/d(epsilon)` of the overlap or
/// kinetic image blocks, over exactly the shell pairs of
/// [`crate::periodic::integrals::image_blocks`]. Translation invariance
/// gives `(dO/dA)_a (A - B - T)_b` per pair.
fn two_centre_strain_on_images(
    mb: &MolBasis,
    images: &[crate::periodic::integrals::ShortRangeImage],
    op: Operator,
    density: &[ImageBlock],
) -> Tensor {
    let kind: PairKind = match op {
        Operator::Overlap => overlap_kind,
        Operator::Kinetic => kinetic_kind,
    };
    sum_tensors(images.par_iter().map(|image| {
        let mut out = [[0.0; 3]; 3];
        let Some(d) = find_block(density, image.translation) else {
            return out;
        };
        for &(i, j) in &image.shell_pairs {
            let sa = &mb.shells[i];
            let sb = shifted_shell(&mb.shells[j], image.translation);
            let d_cart = cartesian_density(d, sa, &mb.shells[j]);
            if d_cart.as_slice().iter().all(|v| *v == 0.0) {
                continue;
            }
            let derivative = bra_derivative(sa, &sb, kind);
            let r = sa.center - sb.center;
            for axis in 0..3 {
                let value = contract(&derivative[axis], &d_cart);
                for b in 0..3 {
                    out[axis][b] += value * r.get(b);
                }
            }
        }
        out
    }))
}

pub(crate) fn two_centre_strain(
    mb: &MolBasis,
    lattice: &Lattice,
    rcut: f64,
    op: Operator,
    density: &[ImageBlock],
) -> Tensor {
    two_centre_strain_on_images(mb, &short_range_images(mb, lattice, rcut), op, density)
}
pub(crate) fn dimensional_two_centre_strain(
    mb: &MolBasis,
    geometry: &crate::periodic::PeriodicGeometry,
    rcut: f64,
    op: Operator,
    density: &[ImageBlock],
) -> crate::Result<Tensor> {
    let images = crate::periodic::integrals::dimensional_short_range_images(mb, geometry, rcut)?;
    Ok(two_centre_strain_on_images(mb, &images, op, density))
}
/// Second strain derivatives and mixed strain-displacement derivatives of a
/// translation-invariant pair term at fixed density.
///
/// For `O(R)` with `R = A - B - T` strained as `R -> (1 + epsilon) R`:
/// `d2O/d(eps_ab) d(eps_cd) = O_ac R_b R_d`, and for a Gamma-point
/// displacement of atom `U` along `c`, with `s = [atom(A) = U] - [atom(B) = U]`,
/// `d2O/d(eps_ab) du_(U,c) = s (O_ac R_b + delta_bc O_a)`.
#[derive(Clone, Debug)]
pub(crate) struct StrainSecond {
    /// `[3a + b][3c + d]`, 9 x 9.
    pub(crate) strain: Vec<f64>,
    /// `[3a + b][3 atom + c]`, 9 x 3N.
    pub(crate) mixed: Vec<f64>,
}

impl StrainSecond {
    pub(crate) fn zeros(natoms: usize) -> Self {
        Self {
            strain: vec![0.0; 81],
            mixed: vec![0.0; 27 * natoms],
        }
    }

    pub(crate) fn add(&mut self, other: &Self) {
        for (x, y) in self.strain.iter_mut().zip(&other.strain) {
            *x += y;
        }
        for (x, y) in self.mixed.iter_mut().zip(&other.mixed) {
            *x += y;
        }
    }
}

/// [`StrainSecond`] of the overlap or kinetic image blocks over the shell
/// pairs of [`crate::periodic::integrals::image_blocks`].
pub(crate) fn two_centre_strain_second(
    mb: &MolBasis,
    lattice: &Lattice,
    rcut: f64,
    op: Operator,
    density: &[ImageBlock],
    natoms: usize,
) -> StrainSecond {
    let kind: PairKind = match op {
        Operator::Overlap => overlap_kind,
        Operator::Kinetic => kinetic_kind,
    };
    let images = short_range_images(mb, lattice, rcut);
    images
        .par_iter()
        .map(|image| {
            let mut out = StrainSecond::zeros(natoms);
            let Some(d) = find_block(density, image.translation) else {
                return out;
            };
            for &(i, j) in &image.shell_pairs {
                let sa = &mb.shells[i];
                let sb = shifted_shell(&mb.shells[j], image.translation);
                let d_cart = cartesian_density(d, sa, &mb.shells[j]);
                if d_cart.as_slice().iter().all(|v| *v == 0.0) {
                    continue;
                }
                let block = |x: &MolShell, y: &MolShell| pair_block(x, y, kind);
                let blocks = crate::periodic::hessian::pair_second_blocks(sa, &sb, &d_cart, &block);
                let first = bra_derivative(sa, &sb, kind);
                let r = sa.center - sb.center;
                // O_ac = d2/dA dA = -d2/dA dB for a translation-invariant pair.
                let oac = |a: usize, c: usize| -blocks[1][a][c];
                let oa: [f64; 3] = std::array::from_fn(|a| contract(&first[a], &d_cart));
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            for dd in 0..3 {
                                out.strain[(3 * a + b) * 9 + 3 * c + dd] +=
                                    oac(a, c) * r.get(b) * r.get(dd);
                            }
                        }
                    }
                }
                for (atom, s) in [(sa.atom, 1.0), (sb.atom, -1.0)] {
                    for a in 0..3 {
                        for b in 0..3 {
                            for c in 0..3 {
                                let mut v = oac(a, c) * r.get(b);
                                if b == c {
                                    v += oa[a];
                                }
                                out.mixed[(3 * a + b) * 3 * natoms + 3 * atom + c] += s * v;
                            }
                        }
                    }
                }
            }
            out
        })
        .reduce(
            || StrainSecond::zeros(natoms),
            |mut a, b| {
                a.add(&b);
                a
            },
        )
}

pub(crate) fn segment_distance(point: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let length2 = ab.norm2();
    if length2 < 1e-24 {
        return (point - a).norm();
    }
    let t = ((point - a).dot(ab) / length2).clamp(0.0, 1.0);
    (point - (a + ab * t)).norm()
}

/// Strain derivative of the Ewald-split periodic nuclear attraction
/// `sum_T D(T) . V(T)` of [`crate::periodic::nuclear::nuclear_image_blocks`]:
/// screened real-space charges (centre virial of the AO pair and of every
/// nuclear image), the reciprocal half (centres, wave vectors and the
/// `exp(-G^2/4 alpha^2)/(V G^2)` kernel) and the `pi Z / (alpha^2 V)` overlap
/// background.
pub(crate) fn nuclear_attraction_strain(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
) -> Tensor {
    let parts =
        nuclear_attraction_strain_parts(mb, mol, charges, lattice, basis_cutoff, p, density);
    let mut total = parts[0];
    add_tensor(&mut total, &parts[1]);
    add_tensor(&mut total, &parts[2]);
    total
}

/// The screened real-space, reciprocal and background parts of
/// [`nuclear_attraction_strain`].
pub(crate) fn nuclear_attraction_strain_parts(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
) -> [Tensor; 3] {
    let v = lattice.volume().abs();
    let reach = p.rcut + basis_cutoff + crate::periodic::nuclear::atom_spread(mol);
    let (img_mol, img_q) = crate::periodic::nuclear::image_charges(mol, charges, lattice, reach);
    let gs = lattice.reciprocal_within(p.gcut);
    let mut c_re = Vec::with_capacity(gs.len());
    let mut c_im = Vec::with_capacity(gs.len());
    // Coefficient strain: d w/d(eps_ab) = w [-delta_ab + 2 G_a G_b (1/(4 alpha^2) + 1/G^2)];
    // the structure factor is strain invariant because G . R is.
    let voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let mut k_re: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(gs.len())).collect();
    let mut k_im: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(gs.len())).collect();
    for g in &gs {
        let g2 = g.norm2();
        let w = -(4.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * p.alpha * p.alpha)).exp() / g2;
        let (mut sc, mut ss) = (0.0, 0.0);
        for (atom, z) in mol.atoms.iter().zip(charges) {
            let phase = g.dot(atom.position);
            sc += z * phase.cos();
            ss += z * phase.sin();
        }
        c_re.push(w * sc);
        c_im.push(-w * ss);
        let scale = 2.0 * (1.0 / (4.0 * p.alpha * p.alpha) + 1.0 / g2);
        for (slot, &(a, b)) in voigt.iter().enumerate() {
            let factor = scale * g.get(a) * g.get(b) - if a == b { 1.0 } else { 0.0 };
            k_re[slot].push(factor * w * sc);
            k_im[slot].push(-factor * w * ss);
        }
    }
    let k_re_refs: Vec<&[f64]> = k_re.iter().map(|v| v.as_slice()).collect();
    let k_im_refs: Vec<&[f64]> = k_im.iter().map(|v| v.as_slice()).collect();
    let ztot: f64 = charges.iter().sum();
    let shift = std::f64::consts::PI * ztot / (p.alpha * p.alpha * v);
    let kernel = CoulombKernel::Erfc { omega: p.alpha };
    // Shell pairs within the overlap-type cutoff, as in the energy.
    let translations = crate::periodic::nuclear::pair_translations(mb, lattice, basis_cutoff);
    let ns = mb.shells.len();
    let jobs: Vec<(Vec3, usize, usize)> = translations
        .iter()
        .flat_map(|&t| (0..ns).flat_map(move |i| (0..ns).map(move |j| (t, i, j))))
        .filter(|&(t, i, j)| {
            crate::periodic::nuclear::pair_within(&mb.shells[i], &mb.shells[j], t, basis_cutoff)
        })
        .collect();
    // (real-space, reciprocal, background electron count, background overlap virial)
    let partials: Vec<(Tensor, Tensor, f64, Tensor)> = jobs
        .par_iter()
        .filter_map(|&(t, i, j)| {
            let d = find_block(density, t)?;
            let sa = &mb.shells[i];
            let sb = shifted_shell(&mb.shells[j], t);
            let d_cart = cartesian_density(d, sa, &mb.shells[j]);
            let dmax = d_cart
                .as_slice()
                .iter()
                .fold(0.0_f64, |m, x| m.max(x.abs()));
            if dmax == 0.0 {
                return None;
            }
            let mut out = [[0.0; 3]; 3];
            // Real space: every screened nuclear image. Translation
            // invariance of each (pair, charge) integral refers the AO
            // centres to the charge.
            let min_a = sa.exps.iter().copied().fold(f64::INFINITY, f64::min);
            let min_b = sb.exps.iter().copied().fold(f64::INFINITY, f64::min);
            let extent = (40.0 / (min_a + min_b)).sqrt();
            for (atom, q) in img_mol.atoms.iter().zip(&img_q) {
                let c = atom.position;
                if segment_distance(c, sa.center, sb.center) > p.rcut + extent {
                    continue;
                }
                let block = |x: &MolShell, y: &MolShell| nuclear_block_kernel(x, y, c, *q, kernel);
                let bra_up = block(&raised(sa), &sb);
                let bra_down = lowered(sa).map(|lo| block(&lo, &sb));
                let ket_up = block(&raised(&sb), sa);
                let ket_down = lowered(&sb).map(|lo| block(&lo, sa));
                let ra = sa.center - c;
                let rb = sb.center - c;
                for axis in 0..3 {
                    let da = contract(
                        &combine_bra_derivative(sa.l, sb.l, axis, &bra_up, bra_down.as_ref()),
                        &d_cart,
                    );
                    let db = contract(
                        &combine_bra_derivative(sb.l, sa.l, axis, &ket_up, ket_down.as_ref())
                            .transpose(),
                        &d_cart,
                    );
                    for b in 0..3 {
                        out[axis][b] += da * ra.get(b) + db * rb.get(b);
                    }
                }
            }
            // Reciprocal half.
            let mut recip = reciprocal_centre_virial(sa, &sb, &gs, (&c_re, &c_im), &d_cart);
            add_tensor(
                &mut recip,
                &reciprocal_wavevector_strain(sa, &sb, &gs, (&c_re, &c_im), &d_cart),
            );
            let coefficient = contracted_pair_blocks(sa, &sb, &gs, &k_re_refs, &k_im_refs);
            for (slot, &(a, b)) in voigt.iter().enumerate() {
                let value = contract(&coefficient[slot].0, &d_cart);
                recip[a][b] += value;
                if a != b {
                    recip[b][a] += value;
                }
            }
            // Background: shift * D . S, with S translation invariant.
            let overlap = pair_block(sa, &sb, overlap_kind);
            let electrons = contract(&overlap, &d_cart);
            let derivative = bra_derivative(sa, &sb, overlap_kind);
            let r = sa.center - sb.center;
            let mut overlap_virial = [[0.0; 3]; 3];
            for axis in 0..3 {
                let value = contract(&derivative[axis], &d_cart);
                for b in 0..3 {
                    overlap_virial[axis][b] += value * r.get(b);
                }
            }
            Some((out, recip, electrons, overlap_virial))
        })
        .collect();
    let mut parts = [[[0.0; 3]; 3]; 3];
    for (real, recip, electrons, overlap_virial) in partials {
        add_tensor(&mut parts[0], &real);
        add_tensor(&mut parts[1], &recip);
        for a in 0..3 {
            for b in 0..3 {
                parts[2][a][b] += shift * overlap_virial[a][b];
            }
            parts[2][a][a] -= shift * electrons;
        }
    }
    parts
}

/// [`StrainSecond`] of the screened real-space half of the Ewald-split
/// nuclear attraction. Per shell pair and nuclear image `C` with
/// `P = A - C`, `Q = B - C`:
/// `sum_{X,Y in {P,Q}} f_XY,ac X_b Y_d`, and for a displacement of atom `U`,
/// with `s_P = [atom(A) = U] - [atom(C) = U]` and likewise `s_Q`,
/// `(f_PP s_P + f_PQ s_Q)_ac P_b + (f_QP s_P + f_QQ s_Q)_ac Q_b
///  + delta_bc (f_P s_P + f_Q s_Q)_a`.
pub(crate) fn nuclear_real_space_strain_second(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
) -> StrainSecond {
    let natoms = mol.len();
    let reach = p.rcut + basis_cutoff + crate::periodic::nuclear::atom_spread(mol);
    let (img_mol, img_q) = crate::periodic::nuclear::image_charges(mol, charges, lattice, reach);
    let kernel = CoulombKernel::Erfc { omega: p.alpha };
    // Shell pairs within the overlap-type cutoff, as in the energy.
    let translations = crate::periodic::nuclear::pair_translations(mb, lattice, basis_cutoff);
    let ns = mb.shells.len();
    let jobs: Vec<(Vec3, usize, usize)> = translations
        .iter()
        .flat_map(|&t| (0..ns).flat_map(move |i| (0..ns).map(move |j| (t, i, j))))
        .filter(|&(t, i, j)| {
            crate::periodic::nuclear::pair_within(&mb.shells[i], &mb.shells[j], t, basis_cutoff)
        })
        .collect();
    jobs.par_iter()
        .map(|&(t, i, j)| {
            let mut out = StrainSecond::zeros(natoms);
            let Some(d) = find_block(density, t) else {
                return out;
            };
            let sa = &mb.shells[i];
            let sb = shifted_shell(&mb.shells[j], t);
            let d_cart = cartesian_density(d, sa, &mb.shells[j]);
            if d_cart.as_slice().iter().all(|x| *x == 0.0) {
                return out;
            }
            let min_a = sa.exps.iter().copied().fold(f64::INFINITY, f64::min);
            let min_b = sb.exps.iter().copied().fold(f64::INFINITY, f64::min);
            let extent = (40.0 / (min_a + min_b)).sqrt();
            for (index, (atom, q)) in img_mol.atoms.iter().zip(&img_q).enumerate() {
                let c = atom.position;
                if segment_distance(c, sa.center, sb.center) > p.rcut + extent {
                    continue;
                }
                let owner = index % natoms;
                let block = |x: &MolShell, y: &MolShell| nuclear_block_kernel(x, y, c, *q, kernel);
                let [fpp, fpq, fqp, fqq] =
                    crate::periodic::hessian::pair_second_blocks(sa, &sb, &d_cart, &block);
                let bra_up = block(&raised(sa), &sb);
                let bra_down = lowered(sa).map(|lo| block(&lo, &sb));
                let ket_up = block(&raised(&sb), sa);
                let ket_down = lowered(&sb).map(|lo| block(&lo, sa));
                let fp: [f64; 3] = std::array::from_fn(|a| {
                    contract(
                        &combine_bra_derivative(sa.l, sb.l, a, &bra_up, bra_down.as_ref()),
                        &d_cart,
                    )
                });
                let fq: [f64; 3] = std::array::from_fn(|a| {
                    contract(
                        &combine_bra_derivative(sb.l, sa.l, a, &ket_up, ket_down.as_ref())
                            .transpose(),
                        &d_cart,
                    )
                });
                let pv = sa.center - c;
                let qv = sb.center - c;
                for a in 0..3 {
                    for b in 0..3 {
                        for cc in 0..3 {
                            for dd in 0..3 {
                                out.strain[(3 * a + b) * 9 + 3 * cc + dd] +=
                                    fpp[a][cc] * pv.get(b) * pv.get(dd)
                                        + fpq[a][cc] * pv.get(b) * qv.get(dd)
                                        + fqp[a][cc] * qv.get(b) * pv.get(dd)
                                        + fqq[a][cc] * qv.get(b) * qv.get(dd);
                            }
                        }
                    }
                }
                for u in 0..natoms {
                    let sp = (sa.atom == u) as i32 as f64 - (owner == u) as i32 as f64;
                    let sq = (sb.atom == u) as i32 as f64 - (owner == u) as i32 as f64;
                    if sp == 0.0 && sq == 0.0 {
                        continue;
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            for cc in 0..3 {
                                let mut value = (fpp[a][cc] * sp + fpq[a][cc] * sq) * pv.get(b)
                                    + (fqp[a][cc] * sp + fqq[a][cc] * sq) * qv.get(b);
                                if b == cc {
                                    value += fp[a] * sp + fq[a] * sq;
                                }
                                out.mixed[(3 * a + b) * 3 * natoms + 3 * u + cc] += value;
                            }
                        }
                    }
                }
            }
            out
        })
        .reduce(
            || StrainSecond::zeros(natoms),
            |mut a, b| {
                a.add(&b);
                a
            },
        )
}

/// [`StrainSecond`] of the nuclear-attraction background
/// `E_bg = shift N` with `shift = pi Z / (alpha^2 V)` and `N = sum D . S`
/// over the pairs of [`nuclear_attraction_strain_parts`]. With
/// `1/det(1 + eps)` contributing `-delta_ab` and
/// `delta_ab delta_cd + delta_ad delta_bc`,
/// `E_bg,ab,cd = shift [(delta_ab delta_cd + delta_ad delta_bc) N
/// - delta_ab N_cd - delta_cd N_ab + N_ab,cd]` and
/// `E_bg,ab,u = shift [N_ab,u - delta_ab N_u]`.
pub(crate) fn nuclear_background_strain_second(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
) -> StrainSecond {
    let natoms = mol.len();
    let dim = 3 * natoms;
    let ztot: f64 = charges.iter().sum();
    let shift = std::f64::consts::PI * ztot / (p.alpha * p.alpha * lattice.volume().abs());
    // Shell pairs within the overlap-type cutoff, as in the energy.
    let translations = crate::periodic::nuclear::pair_translations(mb, lattice, basis_cutoff);
    let ns = mb.shells.len();
    let jobs: Vec<(Vec3, usize, usize)> = translations
        .iter()
        .flat_map(|&t| (0..ns).flat_map(move |i| (0..ns).map(move |j| (t, i, j))))
        .filter(|&(t, i, j)| {
            crate::periodic::nuclear::pair_within(&mb.shells[i], &mb.shells[j], t, basis_cutoff)
        })
        .collect();
    // (N, N_ab, N_u, second derivatives of N)
    let (count, first, gradient, second) = jobs
        .par_iter()
        .map(|&(t, i, j)| {
            let mut first = [[0.0; 3]; 3];
            let mut gradient = vec![0.0; dim];
            let mut second = StrainSecond::zeros(natoms);
            let Some(d) = find_block(density, t) else {
                return (0.0, first, gradient, second);
            };
            let sa = &mb.shells[i];
            let sb = shifted_shell(&mb.shells[j], t);
            let d_cart = cartesian_density(d, sa, &mb.shells[j]);
            if d_cart.as_slice().iter().all(|x| *x == 0.0) {
                return (0.0, first, gradient, second);
            }
            let count = contract(&pair_block(sa, &sb, overlap_kind), &d_cart);
            let block = |x: &MolShell, y: &MolShell| pair_block(x, y, overlap_kind);
            let blocks = crate::periodic::hessian::pair_second_blocks(sa, &sb, &d_cart, &block);
            let derivative = bra_derivative(sa, &sb, overlap_kind);
            let r = sa.center - sb.center;
            let oac = |a: usize, c: usize| -blocks[1][a][c];
            let oa: [f64; 3] = std::array::from_fn(|a| contract(&derivative[a], &d_cart));
            for a in 0..3 {
                for b in 0..3 {
                    first[a][b] += oa[a] * r.get(b);
                    for c in 0..3 {
                        for dd in 0..3 {
                            second.strain[(3 * a + b) * 9 + 3 * c + dd] +=
                                oac(a, c) * r.get(b) * r.get(dd);
                        }
                    }
                }
            }
            for (atom, s) in [(sa.atom, 1.0), (sb.atom, -1.0)] {
                for c in 0..3 {
                    gradient[3 * atom + c] += s * oa[c];
                }
                for a in 0..3 {
                    for b in 0..3 {
                        for c in 0..3 {
                            let mut v = oac(a, c) * r.get(b);
                            if b == c {
                                v += oa[a];
                            }
                            second.mixed[(3 * a + b) * dim + 3 * atom + c] += s * v;
                        }
                    }
                }
            }
            (count, first, gradient, second)
        })
        .reduce(
            || {
                (
                    0.0,
                    [[0.0; 3]; 3],
                    vec![0.0; dim],
                    StrainSecond::zeros(natoms),
                )
            },
            |mut x, y| {
                x.0 += y.0;
                add_tensor(&mut x.1, &y.1);
                for (u, v) in x.2.iter_mut().zip(&y.2) {
                    *u += v;
                }
                x.3.add(&y.3);
                x
            },
        );
    let delta = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let mut out = StrainSecond::zeros(natoms);
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for dd in 0..3 {
                    let k = (3 * a + b) * 9 + 3 * c + dd;
                    out.strain[k] = shift
                        * ((delta(a, b) * delta(c, dd) + delta(a, dd) * delta(b, c)) * count
                            - delta(a, b) * first[c][dd]
                            - delta(c, dd) * first[a][b]
                            + second.strain[k]);
                }
            }
            for x in 0..dim {
                let k = (3 * a + b) * dim + x;
                out.mixed[k] = shift * (second.mixed[k] - delta(a, b) * gradient[x]);
            }
        }
    }
    out
}

/// Second-order strain jets: value, gradient and symmetric Hessian in the
/// nine strain components.
pub(crate) type StrainJet = crate::dual::Jet2;
type MixedJet = crate::dual::DualN<crate::dual::DualN<f64, 9>, 3>;

/// Strain components seeded for second-order jets: `eps_ab` is variable
/// `3a + b`.
pub(crate) fn strain_seed() -> [[StrainJet; 3]; 3] {
    std::array::from_fn(|a| std::array::from_fn(|b| StrainJet::variable(0.0, 3 * a + b)))
}

/// Strain components as first-order jets inside a displacement jet.
fn strain_inner() -> [[MixedJet; 3]; 3] {
    use crate::dual::DualN;
    std::array::from_fn(|a| {
        std::array::from_fn(|b| MixedJet::constant(DualN::<f64, 9>::variable(0.0, 3 * a + b)))
    })
}

fn jet_vec<S: crate::dual::Scalar>(v: Vec3) -> [S; 3] {
    [S::from_f64(v.x), S::from_f64(v.y), S::from_f64(v.z)]
}

/// Accumulate the strained density transform of every shell pair of the
/// nuclear-attraction image blocks, with centres of `moving` atoms displaced
/// by the outer jet seeds of `displacement`.
fn nuclear_density_jets<S: crate::dual::Scalar + Send + Sync>(
    mb: &MolBasis,
    lattice: &Lattice,
    basis_cutoff: f64,
    density: &[ImageBlock],
    eps: &[[S; 3]; 3],
    waves: &[crate::periodic::strain_jets::StrainedWave<S>],
    displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
) -> (Vec<S>, Vec<S>) {
    let translations = crate::periodic::nuclear::pair_translations(mb, lattice, basis_cutoff);
    let ns = mb.shells.len();
    // Shell pairs carrying density within the overlap-type cutoff, as in
    // the energy, with the ket shell at its translation.
    let pairs: Vec<(&MolShell, MolShell, Matrix)> = translations
        .iter()
        .flat_map(|&t| (0..ns).flat_map(move |i| (0..ns).map(move |j| (t, i, j))))
        .filter(|&(t, i, j)| {
            crate::periodic::nuclear::pair_within(&mb.shells[i], &mb.shells[j], t, basis_cutoff)
        })
        .filter_map(|(t, i, j)| {
            let d = find_block(density, t)?;
            let d_cart = cartesian_density(d, &mb.shells[i], &mb.shells[j]);
            (!d_cart.as_slice().iter().all(|x| *x == 0.0))
                .then(|| (&mb.shells[i], shifted_shell(&mb.shells[j], t), d_cart))
        })
        .collect();
    let mut out = crate::periodic::strain_jets::strained_transforms(
        &pairs,
        waves,
        2,
        &|(sa, sb, d_cart), block, acc| {
            let ua = displacement(sa.atom);
            let ub = displacement(sb.atom);
            let a_centre: [S; 3] = std::array::from_fn(|k| jet_vec::<S>(sa.center)[k] + ua[k]);
            let b_centre: [S; 3] = std::array::from_fn(|k| jet_vec::<S>(sb.center)[k] + ub[k]);
            let (re, im) = acc.split_at_mut(1);
            crate::periodic::strain_jets::strained_pair_density(
                sa,
                sb,
                Some([a_centre, b_centre]),
                eps,
                block,
                d_cart,
                &mut re[0],
                &mut im[0],
            );
        },
    );
    let im = out.pop().unwrap_or_default();
    let re = out.pop().unwrap_or_default();
    (re, im)
}

/// Reciprocal nuclear-attraction energy `Re sum_G w(G', V') S(G)^* rho(G')`
/// as a jet, with the structure factor phases displaced by `displacement`.
fn nuclear_reciprocal_jet<S: crate::dual::Scalar + Send + Sync>(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
    eps: &[[S; 3]; 3],
    displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
) -> S {
    use crate::periodic::strain_jets::{cos_sin, strain_det, strained_waves};
    let gs = lattice.reciprocal_within(p.gcut);
    let waves = strained_waves(&gs, eps);
    let (re, im) = nuclear_density_jets(
        mb,
        lattice,
        basis_cutoff,
        density,
        eps,
        &waves,
        displacement,
    );
    let volume = S::from_f64(lattice.volume().abs()) * strain_det(eps);
    let four_pi = S::from_f64(4.0 * std::f64::consts::PI);
    let mut energy = S::from_f64(0.0);
    for (gi, wave) in waves.iter().enumerate() {
        let w = S::from_f64(-1.0) * four_pi / volume
            * (wave.norm2 * S::from_f64(-0.25 / (p.alpha * p.alpha))).exp()
            / wave.norm2;
        let (mut sc, mut ss) = (S::from_f64(0.0), S::from_f64(0.0));
        for (atom, (at, z)) in mol.atoms.iter().zip(charges).enumerate() {
            let u = displacement(atom);
            let phase = S::from_f64(wave.g.x) * (S::from_f64(at.position.x) + u[0])
                + S::from_f64(wave.g.y) * (S::from_f64(at.position.y) + u[1])
                + S::from_f64(wave.g.z) * (S::from_f64(at.position.z) + u[2]);
            let (c, s) = cos_sin(phase);
            sc = sc + S::from_f64(*z) * c;
            ss = ss + S::from_f64(*z) * s;
        }
        // conj(S) rho, real part.
        energy = energy + w * (sc * re[gi] + ss * im[gi]);
    }
    energy
}

/// [`StrainSecond`] of the reciprocal half of the Ewald-split nuclear
/// attraction by exact forward-mode jets.
pub(crate) fn nuclear_reciprocal_strain_second(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
) -> StrainSecond {
    use crate::dual::DualN;
    let natoms = mol.len();
    let mut out = StrainSecond::zeros(natoms);
    let eps = strain_seed();
    let still = |_: usize| [StrainJet::from_f64(0.0); 3];
    let energy = nuclear_reciprocal_jet(
        mb,
        mol,
        charges,
        lattice,
        basis_cutoff,
        p,
        density,
        &eps,
        &still,
    );
    for k in 0..9 {
        for l in 0..9 {
            out.strain[k * 9 + l] = energy.second(k, l);
        }
    }
    let inner = strain_inner();
    for u in 0..natoms {
        let moving = move |atom: usize| -> [MixedJet; 3] {
            if atom == u {
                std::array::from_fn(|c| {
                    let mut x = MixedJet::constant(DualN::<f64, 9>::constant(0.0));
                    x.im[c] = DualN::<f64, 9>::constant(1.0);
                    x
                })
            } else {
                [MixedJet::from_f64(0.0); 3]
            }
        };
        let energy = nuclear_reciprocal_jet(
            mb,
            mol,
            charges,
            lattice,
            basis_cutoff,
            p,
            density,
            &inner,
            &moving,
        );
        for k in 0..9 {
            for c in 0..3 {
                out.mixed[k * 3 * natoms + 3 * u + c] = energy.im[c].im[k];
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::system::Atom;

    pub(crate) const BASIS: &str =
        "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\n****\n";
    pub(crate) const STEPS: [f64; 4] = [4e-3, 2e-3, 1e-3, 5e-4];
    pub(crate) const CUTOFF: f64 = 11.0;

    pub(crate) fn deform(v: Vec3, eps: &Tensor) -> Vec3 {
        Vec3::new(
            v.x + eps[0][0] * v.x + eps[0][1] * v.y + eps[0][2] * v.z,
            v.y + eps[1][0] * v.x + eps[1][1] * v.y + eps[1][2] * v.z,
            v.z + eps[2][0] * v.x + eps[2][1] * v.y + eps[2][2] * v.z,
        )
    }

    /// The cell strained by `eps` after displacing `atom` by `u` along `axis`.
    pub(crate) fn system(eps: &Tensor, atom: usize, axis: usize, u: f64) -> (Lattice, MolBasis) {
        let base = [
            Vec3::new(4.2, 0.0, 0.0),
            Vec3::new(0.3, 4.6, 0.0),
            Vec3::new(0.2, -0.25, 4.9),
        ];
        let lattice = Lattice::new(
            deform(base[0], eps),
            deform(base[1], eps),
            deform(base[2], eps),
        );
        let mut positions = [Vec3::new(0.1, 0.2, -0.3), Vec3::new(1.2, 0.9, 0.8)];
        match axis {
            0 => positions[atom].x += u,
            1 => positions[atom].y += u,
            _ => positions[atom].z += u,
        }
        let molecule = Molecule::new(
            positions
                .iter()
                .map(|p| Atom {
                    z: 1,
                    position: deform(*p, eps),
                })
                .collect(),
        );
        let set = crate::basis::g94::parse("strain-second-test", BASIS).unwrap();
        (lattice, MolBasis::build(&molecule, &set).unwrap())
    }

    pub(crate) fn images(lattice: &Lattice, nao: usize) -> Vec<ImageBlock> {
        let entry = |n: [i64; 3], p: usize, q: usize| -> f64 {
            let s = (n[0] * 7 + n[1] * 13 + n[2] * 29) as f64 + (p * 3 + q * 5) as f64 * 0.37;
            (s * 1.37).sin()
        };
        let mut out = Vec::new();
        for i in -2..=2i64 {
            for j in -2..=2i64 {
                for k in -2..=2i64 {
                    let r2 = (i * i + j * j + k * k) as f64;
                    if r2 > 6.0 {
                        continue;
                    }
                    let decay = (-0.7 * r2).exp();
                    let mut matrix = Matrix::zeros(nao, nao);
                    for p in 0..nao {
                        for q in 0..nao {
                            matrix[(p, q)] =
                                decay * 0.25 * (entry([i, j, k], p, q) + entry([-i, -j, -k], q, p));
                            if r2 == 0.0 && p == q {
                                matrix[(p, q)] += 0.6;
                            }
                        }
                    }
                    out.push(ImageBlock {
                        translation: lattice.translation([i, j, k]),
                        matrix,
                    });
                }
            }
        }
        out
    }

    pub(crate) fn fit(errors: &[f64]) -> (f64, f64) {
        let (x, y): (Vec<f64>, Vec<f64>) = STEPS
            .iter()
            .zip(errors)
            .map(|(h, e)| (h.ln(), e.ln()))
            .unzip();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// Fit a nuclear-attraction part's second derivatives against central
    /// differences of that part's validated strain derivative.
    fn check_nuclear_part(
        label: &str,
        part: usize,
        second: impl Fn(&MolBasis, &Molecule, &Lattice, EwaldParams, &[ImageBlock]) -> StrainSecond,
    ) {
        let zero = [[0.0; 3]; 3];
        let charges = [1.0, 1.0];
        let (lattice0, _) = system(&zero, 0, 0, 0.0);
        let p = EwaldParams::for_accuracy(&lattice0, 2, 1e-12);
        let molecule = |eps: &Tensor, atom: usize, axis: usize, u: f64| {
            let mut positions = [Vec3::new(0.1, 0.2, -0.3), Vec3::new(1.2, 0.9, 0.8)];
            match axis {
                0 => positions[atom].x += u,
                1 => positions[atom].y += u,
                _ => positions[atom].z += u,
            }
            Molecule::new(
                positions
                    .iter()
                    .map(|q| Atom {
                        z: 1,
                        position: deform(*q, eps),
                    })
                    .collect(),
            )
        };
        let (lattice, basis) = system(&zero, 0, 0, 0.0);
        let d = images(&lattice, basis.nao_sph);
        let analytic = second(&basis, &molecule(&zero, 0, 0, 0.0), &lattice, p, &d);
        let first = |eps: &Tensor, atom: usize, axis: usize, u: f64| -> Tensor {
            let (lattice, basis) = system(eps, atom, axis, u);
            nuclear_attraction_strain_parts(
                &basis,
                &molecule(eps, atom, axis, u),
                &charges,
                &lattice,
                CUTOFF,
                p,
                &images(&lattice, basis.nao_sph),
            )[part]
        };
        let s0 = first(&zero, 0, 0, 0.0);
        for (c, dd) in [(0usize, 0usize), (0, 1), (2, 1)] {
            let errors: Vec<f64> = STEPS
                .iter()
                .map(|&h| {
                    let mut ep = zero;
                    ep[c][dd] = h;
                    let mut em = zero;
                    em[c][dd] = -h;
                    let (plus, minus) = (first(&ep, 0, 0, 0.0), first(&em, 0, 0, 0.0));
                    let mut worst = 0.0_f64;
                    for a in 0..3 {
                        for b in 0..3 {
                            let composed = if b == c { s0[a][dd] } else { 0.0 };
                            let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                            worst = worst.max(
                                (fd - analytic.strain[(3 * a + b) * 9 + 3 * c + dd] - composed)
                                    .abs(),
                            );
                        }
                    }
                    worst
                })
                .collect();
            let (pw, r2) = fit(&errors);
            println!("{label} strain-strain column [{c}][{dd}]: errors {errors:?} p = {pw:.6} R2 = {r2:.8}");
            assert!((pw - 2.0).abs() < 0.05 && r2 > 0.9999);
        }
        for (atom, axis) in [(0usize, 0usize), (1, 2)] {
            let errors: Vec<f64> = STEPS
                .iter()
                .map(|&h| {
                    let (plus, minus) = (first(&zero, atom, axis, h), first(&zero, atom, axis, -h));
                    let mut worst = 0.0_f64;
                    for a in 0..3 {
                        for b in 0..3 {
                            let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                            worst = worst.max(
                                (fd - analytic.mixed[(3 * a + b) * 6 + 3 * atom + axis]).abs(),
                            );
                        }
                    }
                    worst
                })
                .collect();
            let (pw, r2) = fit(&errors);
            println!("{label} strain-displacement column ({atom},{axis}): errors {errors:?} p = {pw:.6} R2 = {r2:.8}");
            assert!((pw - 2.0).abs() < 0.05 && r2 > 0.9999);
        }
    }

    #[test]
    fn nuclear_real_space_strain_second_derivatives_pass_four_step_fits() {
        check_nuclear_part("nuclear real-space", 0, |basis, molecule, lattice, p, d| {
            nuclear_real_space_strain_second(basis, molecule, &[1.0, 1.0], lattice, CUTOFF, p, d)
        });
    }

    #[test]
    fn nuclear_reciprocal_strain_second_derivatives_pass_four_step_fits() {
        check_nuclear_part("nuclear reciprocal", 1, |basis, molecule, lattice, p, d| {
            nuclear_reciprocal_strain_second(basis, molecule, &[1.0, 1.0], lattice, CUTOFF, p, d)
        });
    }

    #[test]
    fn nuclear_background_strain_second_derivatives_pass_four_step_fits() {
        check_nuclear_part("nuclear background", 2, |basis, molecule, lattice, p, d| {
            nuclear_background_strain_second(basis, molecule, &[1.0, 1.0], lattice, CUTOFF, p, d)
        });
    }

    #[test]
    fn screened_hartree_strain_second_derivatives_pass_four_step_fits() {
        use crate::periodic::screened_coulomb::{
            screened_hartree_strain, screened_hartree_strain_second, HartreeScreening,
        };
        let zero = [[0.0; 3]; 3];
        for compact_exponent in [1.0, 3.0] {
            let screening = HartreeScreening {
                compact_exponent,
                tolerance: 1e-15,
                max_translation: None,
            };
            let (lattice, basis) = system(&zero, 0, 0, 0.0);
            let d = images(&lattice, basis.nao_sph);
            let analytic =
                screened_hartree_strain_second(&basis, &lattice, &d, &screening, 2).unwrap();
            let first = |eps: &Tensor, atom: usize, axis: usize, u: f64| -> Tensor {
                let (lattice, basis) = system(eps, atom, axis, u);
                screened_hartree_strain(
                    &basis,
                    &lattice,
                    &images(&lattice, basis.nao_sph),
                    &screening,
                )
                .unwrap()
                .strain_derivative
            };
            let s0 = first(&zero, 0, 0, 0.0);
            for (c, dd) in [(0usize, 0usize), (0, 1), (2, 1)] {
                let errors: Vec<f64> = STEPS
                    .iter()
                    .map(|&h| {
                        let mut ep = zero;
                        ep[c][dd] = h;
                        let mut em = zero;
                        em[c][dd] = -h;
                        let (plus, minus) = (first(&ep, 0, 0, 0.0), first(&em, 0, 0, 0.0));
                        let mut worst = 0.0_f64;
                        for a in 0..3 {
                            for b in 0..3 {
                                let composed = if b == c { s0[a][dd] } else { 0.0 };
                                let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                                worst = worst.max(
                                    (fd - analytic.strain[(3 * a + b) * 9 + 3 * c + dd] - composed)
                                        .abs(),
                                );
                            }
                        }
                        worst
                    })
                    .collect();
                let (pw, r2) = fit(&errors);
                println!("Hartree (split {compact_exponent}) strain-strain column [{c}][{dd}]: errors {errors:?} p = {pw:.6} R2 = {r2:.8}");
                assert!((pw - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
            for (atom, axis) in [(0usize, 0usize), (1, 2)] {
                let errors: Vec<f64> = STEPS
                    .iter()
                    .map(|&h| {
                        let (plus, minus) =
                            (first(&zero, atom, axis, h), first(&zero, atom, axis, -h));
                        let mut worst = 0.0_f64;
                        for a in 0..3 {
                            for b in 0..3 {
                                let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                                worst = worst.max(
                                    (fd - analytic.mixed[(3 * a + b) * 6 + 3 * atom + axis]).abs(),
                                );
                            }
                        }
                        worst
                    })
                    .collect();
                let (pw, r2) = fit(&errors);
                println!("Hartree (split {compact_exponent}) strain-displacement column ({atom},{axis}): errors {errors:?} p = {pw:.6} R2 = {r2:.8}");
                assert!((pw - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
        }
    }

    #[test]
    fn two_centre_strain_second_derivatives_pass_four_step_fits() {
        for (label, op) in [
            ("kinetic", Operator::Kinetic),
            ("overlap", Operator::Overlap),
        ] {
            let zero = [[0.0; 3]; 3];
            let (lattice, basis) = system(&zero, 0, 0, 0.0);
            let d = images(&lattice, basis.nao_sph);
            let second = two_centre_strain_second(&basis, &lattice, CUTOFF, op, &d, 2);
            let first = |eps: &Tensor, atom: usize, axis: usize, u: f64| -> Tensor {
                let (lattice, basis) = system(eps, atom, axis, u);
                two_centre_strain(
                    &basis,
                    &lattice,
                    CUTOFF,
                    op,
                    &images(&lattice, basis.nao_sph),
                )
            };
            // Strain-strain columns. The first strain derivative evaluated at
            // a strained cell is the derivative with respect to a further
            // strain composed on top, (1 + delta)(1 + eps); its derivative
            // with respect to eps_cd is the additive second derivative plus
            // delta_bc S_ad.
            let s0 = first(&zero, 0, 0, 0.0);
            for (c, dd) in [(0usize, 0usize), (0, 1), (2, 1)] {
                let errors: Vec<f64> = STEPS
                    .iter()
                    .map(|&h| {
                        let mut ep = zero;
                        ep[c][dd] = h;
                        let mut em = zero;
                        em[c][dd] = -h;
                        let (plus, minus) = (first(&ep, 0, 0, 0.0), first(&em, 0, 0, 0.0));
                        let mut worst = 0.0_f64;
                        for a in 0..3 {
                            for b in 0..3 {
                                let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                                let composed = if b == c { s0[a][dd] } else { 0.0 };
                                worst = worst.max(
                                    (fd - second.strain[(3 * a + b) * 9 + 3 * c + dd] - composed)
                                        .abs(),
                                );
                            }
                        }
                        worst
                    })
                    .collect();
                let (p, r2) = fit(&errors);
                println!("{label} strain-strain column [{c}][{dd}]: errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
            // Strain-displacement columns.
            for (atom, axis) in [(0usize, 0usize), (1, 2)] {
                let errors: Vec<f64> = STEPS
                    .iter()
                    .map(|&h| {
                        let (plus, minus) =
                            (first(&zero, atom, axis, h), first(&zero, atom, axis, -h));
                        let mut worst = 0.0_f64;
                        for a in 0..3 {
                            for b in 0..3 {
                                let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                                worst = worst.max(
                                    (fd - second.mixed[(3 * a + b) * 6 + 3 * atom + axis]).abs(),
                                );
                            }
                        }
                        worst
                    })
                    .collect();
                let (p, r2) = fit(&errors);
                println!("{label} strain-displacement column ({atom},{axis}): errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
        }
    }
}
