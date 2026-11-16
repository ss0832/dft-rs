// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! First-order strain matrices `dM(T)/d(epsilon_ab)` of the periodic
//! one-electron operators at fixed density images, for the strain response.
//!
//! With the convention of [`super::strain`] (centres, lattice and nuclear
//! images map as `X -> (1 + epsilon) X`, wave vectors as
//! `G -> (1 + epsilon)^-T G`, Bloch phases `k . T` invariant), the strain
//! derivative of an image block is its centre virial plus, for reciprocal
//! sums, the wave-vector and coefficient terms. Contracting these matrices
//! with a density reproduces the energy strain derivatives of
//! [`super::strain`]; their Bloch sums are the strain perturbation's
//! first-order matrices `S^eps(k)` and `h^eps(k)`.

use crate::basis::aos::sph_block;
use crate::basis::{MolBasis, MolShell};
use crate::integrals::deriv::{
    combine_bra_derivative, kinetic_kind, lowered, nuclear_block_kernel, overlap_kind, pair_block,
    raised,
};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::ewald::EwaldParams;
use crate::periodic::integrals::{short_range_images, Operator};
use crate::periodic::planewave::contracted_pair_blocks;
use crate::periodic::strain::{
    bra_derivative, reciprocal_centre_virial_blocks, reciprocal_wavevector_blocks,
    segment_distance, shifted_shell, PairKind,
};
use crate::periodic::Lattice;
use crate::system::Molecule;
use rayon::prelude::*;

/// The nine strain derivatives `[3a + b]` of one image block.
#[derive(Clone, Debug)]
pub(crate) struct StrainBlock {
    pub(crate) translation: Vec3,
    pub(crate) d: Vec<Matrix>,
}

impl StrainBlock {
    pub(crate) fn zeros(translation: Vec3, n: usize) -> Self {
        Self::with_components(translation, n, 9)
    }

    pub(crate) fn with_components(translation: Vec3, n: usize, components: usize) -> Self {
        Self {
            translation,
            d: (0..components).map(|_| Matrix::zeros(n, n)).collect(),
        }
    }

    /// Add `scale * sph(cart)` at the shell offsets of component `k`.
    pub(crate) fn add_cart(
        &mut self,
        k: usize,
        sa: &MolShell,
        sb: &MolShell,
        cart: &Matrix,
        scale: f64,
    ) {
        let sph = sph_block(sa.l, sb.l, cart);
        self.add_sph(k, sa.sph_off, sb.sph_off, &sph, scale);
    }

    pub(crate) fn add_sph(&mut self, k: usize, row: usize, col: usize, sph: &Matrix, scale: f64) {
        let target = &mut self.d[k];
        for i in 0..sph.rows {
            for j in 0..sph.cols {
                target[(row + i, col + j)] += scale * sph[(i, j)];
            }
        }
    }

    fn is_zero(&self) -> bool {
        self.d
            .iter()
            .all(|m| m.as_slice().iter().all(|v| *v == 0.0))
    }
}

/// Merge strain blocks with equal translations.
pub(crate) fn merge_strain_blocks(blocks: Vec<StrainBlock>) -> Vec<StrainBlock> {
    let mut out: Vec<StrainBlock> = Vec::new();
    for block in blocks {
        if let Some(existing) = out
            .iter_mut()
            .find(|b| (b.translation - block.translation).norm() < 1e-8)
        {
            for (x, y) in existing.d.iter_mut().zip(&block.d) {
                for (u, v) in x.as_mut_slice().iter_mut().zip(y.as_slice()) {
                    *u += v;
                }
            }
        } else {
            out.push(block);
        }
    }
    out
}

/// Bloch sums `sum_T exp(i k . T) dM(T)/d(epsilon_ab)`, indexed `[3a + b]`.
pub(crate) fn strain_bloch(blocks: &[StrainBlock], k: Vec3, n: usize) -> Vec<BlochMatrix> {
    let mut out: Vec<BlochMatrix> = (0..9).map(|_| BlochMatrix::zeros(n, n)).collect();
    for block in blocks {
        let phase = k.dot(block.translation);
        let (c, s) = (phase.cos(), phase.sin());
        for (target, source) in out.iter_mut().zip(&block.d) {
            for ((re, im), v) in target
                .re
                .as_mut_slice()
                .iter_mut()
                .zip(target.im.as_mut_slice().iter_mut())
                .zip(source.as_slice())
            {
                *re += c * v;
                *im += s * v;
            }
        }
    }
    out
}

/// `sum_T D(T) . dM(T)/d(epsilon_ab)` as a tensor.
#[cfg(test)]
pub(crate) fn contract_strain(
    blocks: &[StrainBlock],
    density: &[crate::periodic::integrals::ImageBlock],
) -> crate::periodic::strain::Tensor {
    let mut out = [[0.0; 3]; 3];
    for block in blocks {
        let Some(d) = crate::periodic::strain::find_block(density, block.translation) else {
            continue;
        };
        for (k, m) in block.d.iter().enumerate() {
            out[k / 3][k % 3] += m.frobenius_dot(d);
        }
    }
    out
}

/// Strain blocks of the overlap or kinetic image blocks over the shell
/// pairs of [`crate::periodic::integrals::image_blocks`]:
/// `(dO/dA)_a (A - B - T)_b` per shell pair.
pub(crate) fn two_centre_strain_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    rcut: f64,
    op: Operator,
) -> Vec<StrainBlock> {
    let kind: PairKind = match op {
        Operator::Overlap => overlap_kind,
        Operator::Kinetic => kinetic_kind,
    };
    let n = mb.nao_sph;
    let images = short_range_images(mb, lattice, rcut);
    let blocks: Vec<StrainBlock> = images
        .par_iter()
        .map(|image| {
            let mut block = StrainBlock::zeros(image.translation, n);
            for &(i, j) in &image.shell_pairs {
                let sa = &mb.shells[i];
                let sb = shifted_shell(&mb.shells[j], image.translation);
                let derivative = bra_derivative(sa, &sb, kind);
                let r = sa.center - sb.center;
                for a in 0..3 {
                    let sph = sph_block(sa.l, sb.l, &derivative[a]);
                    for b in 0..3 {
                        block.add_sph(3 * a + b, sa.sph_off, sb.sph_off, &sph, r.get(b));
                    }
                }
            }
            block
        })
        .filter(|b| !b.is_zero())
        .collect();
    merge_strain_blocks(blocks)
}

/// Strain blocks of the Ewald-split periodic nuclear attraction of
/// [`crate::periodic::nuclear::nuclear_image_blocks`]: the screened
/// real-space charges (centre virial of the pair and of every nuclear image
/// referred to the charge), the reciprocal half (centre virial, wave vectors
/// and the kernel `exp(-G^2/4 alpha^2)/(V G^2)`) and the
/// `pi Z / (alpha^2 V)` overlap background.
pub(crate) fn nuclear_attraction_strain_blocks(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
) -> Vec<StrainBlock> {
    let n = mb.nao_sph;
    let v = lattice.volume().abs();
    let reach = p.rcut + basis_cutoff + crate::periodic::nuclear::atom_spread(mol);
    let (img_mol, img_q) = crate::periodic::nuclear::image_charges(mol, charges, lattice, reach);
    let gs = lattice.reciprocal_within(p.gcut);
    let voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let mut c_re = Vec::with_capacity(gs.len());
    let mut c_im = Vec::with_capacity(gs.len());
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
    let k_re_refs: Vec<&[f64]> = k_re.iter().map(|x| x.as_slice()).collect();
    let k_im_refs: Vec<&[f64]> = k_im.iter().map(|x| x.as_slice()).collect();
    let ztot: f64 = charges.iter().sum();
    let shift = std::f64::consts::PI * ztot / (p.alpha * p.alpha * v);
    let kernel = CoulombKernel::Erfc { omega: p.alpha };
    // Shell pairs within the overlap-type cutoff, as in the energy.
    let translations = crate::periodic::nuclear::pair_translations(mb, lattice, basis_cutoff);
    let ns = mb.shells.len();
    let blocks: Vec<StrainBlock> = translations
        .par_iter()
        .map(|&t| {
            let mut block = StrainBlock::zeros(t, n);
            for i in 0..ns {
                for j in 0..ns {
                    if !crate::periodic::nuclear::pair_within(
                        &mb.shells[i],
                        &mb.shells[j],
                        t,
                        basis_cutoff,
                    ) {
                        continue;
                    }
                    let sa = &mb.shells[i];
                    let sb = shifted_shell(&mb.shells[j], t);
                    let (na, nb) = (crate::basis::n_cart(sa.l), crate::basis::n_cart(sb.l));
                    let mut local: Vec<Matrix> = (0..9).map(|_| Matrix::zeros(na, nb)).collect();
                    let mut add = |k: usize, m: &Matrix, scale: f64| {
                        for (x, y) in local[k].as_mut_slice().iter_mut().zip(m.as_slice()) {
                            *x += scale * y;
                        }
                    };
                    // Screened real-space charges, referred to the charge.
                    let min_a = sa.exps.iter().copied().fold(f64::INFINITY, f64::min);
                    let min_b = sb.exps.iter().copied().fold(f64::INFINITY, f64::min);
                    let extent = (40.0 / (min_a + min_b)).sqrt();
                    for (atom, q) in img_mol.atoms.iter().zip(&img_q) {
                        let c = atom.position;
                        if segment_distance(c, sa.center, sb.center) > p.rcut + extent {
                            continue;
                        }
                        let kernel_block =
                            |x: &MolShell, y: &MolShell| nuclear_block_kernel(x, y, c, *q, kernel);
                        let bra_up = kernel_block(&raised(sa), &sb);
                        let bra_down = lowered(sa).map(|lo| kernel_block(&lo, &sb));
                        let ket_up = kernel_block(&raised(&sb), sa);
                        let ket_down = lowered(&sb).map(|lo| kernel_block(&lo, sa));
                        let ra = sa.center - c;
                        let rb = sb.center - c;
                        for axis in 0..3 {
                            let da = combine_bra_derivative(
                                sa.l,
                                sb.l,
                                axis,
                                &bra_up,
                                bra_down.as_ref(),
                            );
                            let db = combine_bra_derivative(
                                sb.l,
                                sa.l,
                                axis,
                                &ket_up,
                                ket_down.as_ref(),
                            )
                            .transpose();
                            for b in 0..3 {
                                add(3 * axis + b, &da, ra.get(b));
                                add(3 * axis + b, &db, rb.get(b));
                            }
                        }
                    }
                    // Reciprocal half.
                    let virial = reciprocal_centre_virial_blocks(sa, &sb, &gs, (&c_re, &c_im));
                    let waves = reciprocal_wavevector_blocks(sa, &sb, &gs, (&c_re, &c_im));
                    for a in 0..3 {
                        for b in 0..3 {
                            add(3 * a + b, &virial[a][b], 1.0);
                            add(3 * a + b, &waves[a][b], 1.0);
                        }
                    }
                    let coefficient = contracted_pair_blocks(sa, &sb, &gs, &k_re_refs, &k_im_refs);
                    for (slot, &(a, b)) in voigt.iter().enumerate() {
                        add(3 * a + b, &coefficient[slot].0, 1.0);
                        if a != b {
                            add(3 * b + a, &coefficient[slot].0, 1.0);
                        }
                    }
                    // Background shift * S with shift proportional to 1/V.
                    let overlap = pair_block(sa, &sb, overlap_kind);
                    let derivative = bra_derivative(sa, &sb, overlap_kind);
                    let r = sa.center - sb.center;
                    for a in 0..3 {
                        for b in 0..3 {
                            add(3 * a + b, &derivative[a], shift * r.get(b));
                        }
                        add(4 * a, &overlap, -shift);
                    }
                    for (k, m) in local.iter().enumerate() {
                        block.add_cart(k, sa, &sb, m, 1.0);
                    }
                }
            }
            block
        })
        .filter(|b| !b.is_zero())
        .collect();
    merge_strain_blocks(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::integrals::ImageBlock;
    use crate::periodic::screened_coulomb::{
        screened_hartree_image_blocks, screened_hartree_strain, HartreeScreening,
    };
    use crate::periodic::strain::tests::{deform, fit, images, system, CUTOFF, STEPS};
    use crate::periodic::strain::Tensor;
    use crate::system::Atom;

    fn molecule(eps: &Tensor) -> Molecule {
        let positions = [Vec3::new(0.1, 0.2, -0.3), Vec3::new(1.2, 0.9, 0.8)];
        Molecule::new(
            positions
                .iter()
                .map(|q| Atom {
                    z: 1,
                    position: deform(*q, eps),
                })
                .collect(),
        )
    }

    fn index(lattice: &Lattice, t: Vec3) -> [i64; 3] {
        let b = lattice.reciprocal();
        std::array::from_fn(|i| (b[i].dot(t) / (2.0 * std::f64::consts::PI)).round() as i64)
    }

    /// Largest deviation of the central difference of image-block builds at
    /// `+-h` from component `k` of the analytic strain blocks, over blocks
    /// matched by their lattice index.
    fn worst(
        analytic: &[StrainBlock],
        lattice0: &Lattice,
        k: usize,
        h: f64,
        build: &dyn Fn(f64) -> (Lattice, Vec<ImageBlock>),
    ) -> f64 {
        let (lp, plus) = build(h);
        let (lm, minus) = build(-h);
        let mut keys: Vec<[i64; 3]> = analytic
            .iter()
            .map(|b| index(lattice0, b.translation))
            .collect();
        keys.extend(plus.iter().map(|b| index(&lp, b.translation)));
        keys.extend(minus.iter().map(|b| index(&lm, b.translation)));
        keys.sort_unstable();
        keys.dedup();
        let lookup = |set: &[ImageBlock], lattice: &Lattice, key: [i64; 3]| {
            set.iter()
                .find(|b| index(lattice, b.translation) == key)
                .map(|b| b.matrix.clone())
        };
        let mut out = 0.0_f64;
        for key in keys {
            let p = lookup(&plus, &lp, key);
            let m = lookup(&minus, &lm, key);
            let a = analytic
                .iter()
                .find(|b| index(lattice0, b.translation) == key)
                .map(|b| &b.d[k]);
            let n = p
                .as_ref()
                .or(m.as_ref())
                .map(|x| x.rows)
                .or(a.map(|x| x.rows))
                .unwrap();
            for i in 0..n {
                for j in 0..n {
                    let fp = p.as_ref().map_or(0.0, |x| x[(i, j)]);
                    let fm = m.as_ref().map_or(0.0, |x| x[(i, j)]);
                    let value = a.map_or(0.0, |x| x[(i, j)]);
                    out = out.max(((fp - fm) / (2.0 * h) - value).abs());
                }
            }
        }
        out
    }

    fn check(
        label: &str,
        analytic: &[StrainBlock],
        build: &dyn Fn(&Tensor) -> (Lattice, Vec<ImageBlock>),
    ) {
        let zero = [[0.0; 3]; 3];
        let (lattice0, _) = build(&zero);
        for (c, dd) in [(0usize, 0usize), (0, 1), (2, 1)] {
            let strained = |h: f64| {
                let mut eps = zero;
                eps[c][dd] = h;
                build(&eps)
            };
            let errors: Vec<f64> = STEPS
                .iter()
                .map(|&h| worst(analytic, &lattice0, 3 * c + dd, h, &strained))
                .collect();
            let (p, r2) = fit(&errors);
            println!(
                "{label} strain matrix [{c}][{dd}]: errors {errors:?} p = {p:.6} R2 = {r2:.8}"
            );
            assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
        }
    }

    #[test]
    fn one_electron_strain_matrices_pass_four_step_fits() {
        let zero = [[0.0; 3]; 3];
        let (lattice, basis) = system(&zero, 0, 0, 0.0);
        let d = images(&lattice, basis.nao_sph);
        let charges = [1.0, 1.0];
        let p = EwaldParams::for_accuracy(&lattice, 2, 1e-12);
        for op in [Operator::Overlap, Operator::Kinetic] {
            let analytic = two_centre_strain_blocks(&basis, &lattice, CUTOFF, op);
            let contracted = contract_strain(&analytic, &d);
            let reference =
                crate::periodic::strain::two_centre_strain(&basis, &lattice, CUTOFF, op, &d);
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        (contracted[a][b] - reference[a][b]).abs()
                            < 1e-10 * (1.0 + reference[a][b].abs())
                    );
                }
            }
            check(&format!("{op:?}"), &analytic, &|eps| {
                let (lattice, basis) = system(eps, 0, 0, 0.0);
                let blocks = crate::periodic::integrals::image_blocks(&basis, &lattice, CUTOFF, op);
                (lattice, blocks)
            });
        }
        let mol = molecule(&zero);
        let analytic =
            nuclear_attraction_strain_blocks(&basis, &mol, &charges, &lattice, CUTOFF, p);
        let contracted = contract_strain(&analytic, &d);
        let reference = crate::periodic::strain::nuclear_attraction_strain(
            &basis, &mol, &charges, &lattice, CUTOFF, p, &d,
        );
        for a in 0..3 {
            for b in 0..3 {
                println!(
                    "nuclear contraction [{a}][{b}]: {} against {}",
                    contracted[a][b], reference[a][b]
                );
                assert!(
                    (contracted[a][b] - reference[a][b]).abs()
                        < 1e-10 * (1.0 + reference[a][b].abs())
                );
            }
        }
        check("nuclear attraction", &analytic, &|eps| {
            let (lattice, basis) = system(eps, 0, 0, 0.0);
            let blocks = crate::periodic::nuclear::nuclear_image_blocks(
                &basis,
                &molecule(eps),
                &charges,
                &lattice,
                CUTOFF,
                p,
            );
            (lattice, blocks)
        });
    }

    #[test]
    fn hartree_strain_matrices_pass_four_step_fits() {
        let zero = [[0.0; 3]; 3];
        for compact_exponent in [1.0, 3.0] {
            let screening = HartreeScreening {
                compact_exponent,
                tolerance: 1e-15,
                max_translation: None,
            };
            let (lattice, basis) = system(&zero, 0, 0, 0.0);
            let d = images(&lattice, basis.nao_sph);
            let analytic = crate::periodic::screened_coulomb::screened_hartree_strain_fock(
                &basis, &lattice, &d, &screening,
            )
            .unwrap();
            let contracted = contract_strain(&analytic, &d);
            let reference = screened_hartree_strain(&basis, &lattice, &d, &screening)
                .unwrap()
                .strain_derivative;
            for a in 0..3 {
                for b in 0..3 {
                    println!(
                        "Hartree (split {compact_exponent}) contraction [{a}][{b}]: {} against {}",
                        contracted[a][b],
                        2.0 * reference[a][b]
                    );
                    assert!(
                        (contracted[a][b] - 2.0 * reference[a][b]).abs()
                            < 1e-9 * (1.0 + reference[a][b].abs())
                    );
                }
            }
            check(
                &format!("Hartree (split {compact_exponent})"),
                &analytic,
                &|eps| {
                    let (lattice, basis) = system(eps, 0, 0, 0.0);
                    let blocks = screened_hartree_image_blocks(
                        &basis,
                        &lattice,
                        &images(&lattice, basis.nao_sph),
                        &screening,
                    )
                    .unwrap();
                    (lattice, blocks)
                },
            );
        }
    }
}
