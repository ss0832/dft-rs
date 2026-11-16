// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gamma-point skeleton second derivatives of the periodic energy terms at a
//! fixed AO density.
//!
//! A Gamma-point displacement moves an atom and every one of its lattice
//! images together, so a basis function, a nuclear charge or a grid owner
//! contributes to the row of its parent atom. Every lattice-sum term is
//! differentiated in the same pieces as its gradient:
//!
//! * translation-invariant pairs (overlap, kinetic, `erfc` nuclear charges,
//!   `erfc` quartets) supply only mixed centre derivatives; the same-centre
//!   blocks follow from `d_i d_i = -sum_{j != i} d_i d_j`, so a pair or
//!   quartet whose centres all belong to one atom is rigid and vanishes;
//! * reciprocal pair transforms `rho_ab(G) = <a| e^{iG.r} |b>` are not
//!   translation invariant, so their nested bra, nested ket and mixed blocks
//!   are all evaluated, and the structure factor contributes mixed
//!   basis-nuclear and nuclear-nuclear terms.
//!
//! The Hessians are `3N x 3N` row-major in `(atom, axis)`, in Hartree/Bohr^2.

use crate::basis::{MolBasis, MolShell};
use crate::integrals::deriv::{
    combine_bra_derivative, kinetic_kind, lowered, overlap_kind, pair_block, raised,
};
use crate::integrals::deriv2::{combine_bra_second, combine_ket_derivative, combine_ket_second};
use crate::integrals::hermite::ETable;
use crate::integrals::one_electron::nuclear_pair_block_kernel;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::ewald::EwaldParams;
use crate::periodic::integrals::{short_range_images, ImageBlock, Operator};
use crate::periodic::planewave::{contracted_pair_blocks, pair_density_transform};
use crate::periodic::strain::{cartesian_density, find_block};
use crate::periodic::Lattice;
use crate::system::Molecule;
use rayon::prelude::*;

type PairKind = fn(&ETable, &ETable, &ETable, f64, f64, (i32, i32, i32), (i32, i32, i32)) -> f64;

fn contract(a: &Matrix, b: &Matrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x * y)
        .sum()
}

fn shifted_shell(sh: &MolShell, t: Vec3) -> MolShell {
    let mut out = sh.clone();
    out.center += t;
    out
}

fn add_matrices(parts: impl ParallelIterator<Item = Matrix>, n: usize) -> Matrix {
    parts.reduce(
        || Matrix::zeros(n, n),
        |mut a, b| {
            for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *x += *y;
            }
            a
        },
    )
}

/// Second-derivative blocks of one pair integral `block(a, b)` contracted
/// with a Cartesian density: `[AA, AB, BA, BB]`, each `[x][y]` with the
/// first index on the first named centre.
pub(crate) fn pair_second_blocks(
    sa: &MolShell,
    sb: &MolShell,
    d_cart: &Matrix,
    block: &dyn Fn(&MolShell, &MolShell) -> Matrix,
) -> [[[f64; 3]; 3]; 4] {
    let ra = raised(sa);
    let rra = raised(&ra);
    let la = lowered(sa);
    let lla = la.as_ref().and_then(lowered);
    let rb = raised(sb);
    let rrb = raised(&rb);
    let lb = lowered(sb);
    let llb = lb.as_ref().and_then(lowered);
    // Nested bra (coefficients 4a^2 c, 2a c, c).
    let a_rr = block(&rra, sb);
    let a_rl = block(&lowered(&ra).expect("l+1 is lowerable"), sb);
    let a_ll = lla.as_ref().map(|lo| block(lo, sb));
    // Nested ket.
    let b_rr = block(sa, &rrb);
    let b_rl = block(sa, &lowered(&rb).expect("l+1 is lowerable"));
    let b_ll = llb.as_ref().map(|lo| block(sa, lo));
    // Mixed.
    let m_rr = block(&ra, &rb);
    let m_rl = lb.as_ref().map(|lo| block(&ra, lo));
    let m_lr = la.as_ref().map(|lo| block(lo, &rb));
    let m_ll = la
        .as_ref()
        .and_then(|loa| lb.as_ref().map(|lob| block(loa, lob)));
    let mut out = [[[0.0; 3]; 3]; 4];
    for x in 0..3 {
        let bra_r = combine_bra_derivative(sa.l, sb.l + 1, x, &m_rr, m_lr.as_ref());
        let bra_l = m_rl
            .as_ref()
            .map(|rl| combine_bra_derivative(sa.l, sb.l - 1, x, rl, m_ll.as_ref()));
        for y in 0..3 {
            out[0][x][y] = contract(
                &combine_bra_second(sa.l, sb.l, x, y, &a_rr, &a_rl, a_ll.as_ref()),
                d_cart,
            );
            out[1][x][y] = contract(
                &combine_ket_derivative(sa.l, sb.l, y, &bra_r, bra_l.as_ref()),
                d_cart,
            );
            out[3][x][y] = contract(
                &combine_ket_second(sa.l, sb.l, x, y, &b_rr, &b_rl, b_ll.as_ref()),
                d_cart,
            );
        }
    }
    for x in 0..3 {
        for y in 0..3 {
            out[2][x][y] = out[1][y][x];
        }
    }
    out
}

/// Add a translation-invariant pair's mixed block `v = d_A d_B` (first index
/// on `A`) with its same-centre partners.
fn add_invariant_pair(h: &mut Matrix, a: usize, b: usize, v: &[[f64; 3]; 3]) {
    for x in 0..3 {
        for y in 0..3 {
            h[(3 * a + x, 3 * b + y)] += v[x][y];
            h[(3 * b + y, 3 * a + x)] += v[x][y];
            h[(3 * a + x, 3 * a + y)] -= v[x][y];
            h[(3 * b + y, 3 * b + x)] -= v[x][y];
        }
    }
}

fn add_block(h: &mut Matrix, a: usize, b: usize, v: &[[f64; 3]; 3]) {
    for x in 0..3 {
        for y in 0..3 {
            h[(3 * a + x, 3 * b + y)] += v[x][y];
        }
    }
}

/// Gamma skeleton Hessian `sum_T D(T) . d2 O(T)` of the overlap or kinetic
/// image blocks over exactly the shell pairs of
/// [`crate::periodic::integrals::image_blocks`].
pub(crate) fn two_centre_hessian(
    mb: &MolBasis,
    lattice: &Lattice,
    rcut: f64,
    op: Operator,
    density: &[ImageBlock],
    natoms: usize,
) -> Matrix {
    let kind: PairKind = match op {
        Operator::Overlap => overlap_kind,
        Operator::Kinetic => kinetic_kind,
    };
    let images = short_range_images(mb, lattice, rcut);
    add_matrices(
        images.par_iter().map(|image| {
            let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
            let Some(d) = find_block(density, image.translation) else {
                return h;
            };
            for &(i, j) in &image.shell_pairs {
                let sa = &mb.shells[i];
                let sb = shifted_shell(&mb.shells[j], image.translation);
                if sa.atom == sb.atom {
                    continue; // the pair moves rigidly at Gamma
                }
                let d_cart = cartesian_density(d, sa, &mb.shells[j]);
                if d_cart.as_slice().iter().all(|v| *v == 0.0) {
                    continue;
                }
                let block = |x: &MolShell, y: &MolShell| pair_block(x, y, kind);
                let blocks = pair_second_blocks(sa, &sb, &d_cart, &block);
                add_invariant_pair(&mut h, sa.atom, sb.atom, &blocks[1]);
            }
            h
        }),
        3 * natoms,
    )
}

/// Gamma skeleton Hessian of the Ewald-split periodic nuclear attraction
/// `sum_T D(T) . V(T)` of [`crate::periodic::nuclear::nuclear_image_blocks`]:
/// screened real-space charges (three-centre blocks with the sum rules
/// `AC = -AA - AB`, `BC = -BA - BB`, `CC = AA + AB + BA + BB`), the
/// reciprocal half (pair transforms, structure factor and their mixed
/// terms) and the overlap background.
pub(crate) fn nuclear_attraction_hessian(
    mb: &MolBasis,
    mol: &Molecule,
    charges: &[f64],
    lattice: &Lattice,
    basis_cutoff: f64,
    p: EwaldParams,
    density: &[ImageBlock],
) -> Matrix {
    let natoms = mol.len();
    let n = 3 * natoms;
    let v = lattice.volume().abs();
    let reach = p.rcut + basis_cutoff + crate::periodic::nuclear::atom_spread(mol);
    let (img_mol, img_q) = crate::periodic::nuclear::image_charges(mol, charges, lattice, reach);
    // A Gamma displacement moves all images of a nucleus together. Group
    // those charges so each raised-shell integral can sum them in one batch.
    let image_centres: Vec<Vec<Vec3>> = (0..natoms)
        .map(|atom| {
            img_mol.atoms[atom..]
                .iter()
                .step_by(natoms)
                .map(|image| image.position)
                .collect()
        })
        .collect();
    let image_charges: Vec<Vec<f64>> = (0..natoms)
        .map(|atom| img_q[atom..].iter().step_by(natoms).copied().collect())
        .collect();
    let gs = lattice.reciprocal_within(p.gcut);
    let w: Vec<f64> = gs
        .iter()
        .map(|g| {
            let g2 = g.norm2();
            -(4.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * p.alpha * p.alpha)).exp() / g2
        })
        .collect();
    let (mut c_re, mut c_im) = (vec![0.0; gs.len()], vec![0.0; gs.len()]);
    // d conj(S)/dR_{n,y} = Z_n (-i G_y) e^{-i G.R_n}.
    let mut s_re: Vec<Vec<f64>> = vec![vec![0.0; gs.len()]; n];
    let mut s_im: Vec<Vec<f64>> = vec![vec![0.0; gs.len()]; n];
    for (gi, g) in gs.iter().enumerate() {
        let (mut sc, mut ss) = (0.0, 0.0);
        for (atom, (at, z)) in mol.atoms.iter().zip(charges).enumerate() {
            let phase = g.dot(at.position);
            let (cs, sn) = (phase.cos(), phase.sin());
            sc += z * cs;
            ss += z * sn;
            for y in 0..3 {
                s_re[3 * atom + y][gi] = -w[gi] * z * g.get(y) * sn;
                s_im[3 * atom + y][gi] = -w[gi] * z * g.get(y) * cs;
            }
        }
        c_re[gi] = w[gi] * sc;
        c_im[gi] = -w[gi] * ss;
    }
    let s_re_refs: Vec<&[f64]> = s_re.iter().map(|x| x.as_slice()).collect();
    let s_im_refs: Vec<&[f64]> = s_im.iter().map(|x| x.as_slice()).collect();
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
    let ng = gs.len();
    // (Hessian, density transform rho_D(G) re/im)
    let partials: Vec<(Matrix, Vec<f64>, Vec<f64>)> = jobs
        .par_iter()
        .filter_map(|&(t, i, j)| {
            let d = find_block(density, t)?;
            let sa = &mb.shells[i];
            let sb = shifted_shell(&mb.shells[j], t);
            let d_cart = cartesian_density(d, sa, &mb.shells[j]);
            if d_cart.as_slice().iter().all(|x| *x == 0.0) {
                return None;
            }
            let mut h = Matrix::zeros(n, n);
            let (ia, ib) = (sa.atom, sb.atom);
            // Use the same screened charge images as the SCF operator and
            // its first derivative. A shell-dependent screen here would
            // differentiate a different real-space sum.
            for (owner, (centres, charges)) in image_centres.iter().zip(&image_charges).enumerate()
            {
                let block = |x: &MolShell, y: &MolShell| {
                    nuclear_pair_block_kernel(x, y, centres, charges, kernel)
                };
                let [aa, ab, ba, bb] = pair_second_blocks(sa, &sb, &d_cart, &block);
                let mut ac = [[0.0; 3]; 3];
                let mut bc = [[0.0; 3]; 3];
                let mut cc = [[0.0; 3]; 3];
                for x in 0..3 {
                    for y in 0..3 {
                        ac[x][y] = -aa[x][y] - ab[x][y];
                        bc[x][y] = -ba[x][y] - bb[x][y];
                        cc[x][y] = aa[x][y] + ab[x][y] + ba[x][y] + bb[x][y];
                    }
                }
                let transpose = |m: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
                    std::array::from_fn(|x| std::array::from_fn(|y| m[y][x]))
                };
                add_block(&mut h, ia, ia, &aa);
                add_block(&mut h, ia, ib, &ab);
                add_block(&mut h, ib, ia, &ba);
                add_block(&mut h, ib, ib, &bb);
                add_block(&mut h, ia, owner, &ac);
                add_block(&mut h, owner, ia, &transpose(&ac));
                add_block(&mut h, ib, owner, &bc);
                add_block(&mut h, owner, ib, &transpose(&bc));
                add_block(&mut h, owner, owner, &cc);
            }
            // Reciprocal half: pair transforms at fixed structure factor.
            let block = |x: &MolShell, y: &MolShell| {
                contracted_pair_blocks(x, y, &gs, &[&c_re], &[&c_im])
                    .remove(0)
                    .0
            };
            let [aa, ab, ba, bb] = pair_second_blocks(sa, &sb, &d_cart, &block);
            add_block(&mut h, ia, ia, &aa);
            add_block(&mut h, ia, ib, &ab);
            add_block(&mut h, ib, ia, &ba);
            add_block(&mut h, ib, ib, &bb);
            // Mixed pair-transform and structure-factor terms.
            let d_bra = {
                let up = contracted_pair_blocks(&raised(sa), &sb, &gs, &s_re_refs, &s_im_refs);
                let down = lowered(sa)
                    .map(|lo| contracted_pair_blocks(&lo, &sb, &gs, &s_re_refs, &s_im_refs));
                (up, down)
            };
            let d_ket = {
                let up = contracted_pair_blocks(&raised(&sb), sa, &gs, &s_re_refs, &s_im_refs);
                let down = lowered(&sb)
                    .map(|lo| contracted_pair_blocks(&lo, sa, &gs, &s_re_refs, &s_im_refs));
                (up, down)
            };
            for nucleus in 0..natoms {
                let mut va = [[0.0; 3]; 3];
                let mut vb = [[0.0; 3]; 3];
                for y in 0..3 {
                    let set = 3 * nucleus + y;
                    for x in 0..3 {
                        let bra = combine_bra_derivative(
                            sa.l,
                            sb.l,
                            x,
                            &d_bra.0[set].0,
                            d_bra.1.as_ref().map(|down| &down[set].0),
                        );
                        va[x][y] = contract(&bra, &d_cart);
                        let ket = combine_bra_derivative(
                            sb.l,
                            sa.l,
                            x,
                            &d_ket.0[set].0,
                            d_ket.1.as_ref().map(|down| &down[set].0),
                        )
                        .transpose();
                        vb[x][y] = contract(&ket, &d_cart);
                    }
                }
                let transpose = |m: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
                    std::array::from_fn(|x| std::array::from_fn(|y| m[y][x]))
                };
                add_block(&mut h, ia, nucleus, &va);
                add_block(&mut h, nucleus, ia, &transpose(&va));
                add_block(&mut h, ib, nucleus, &vb);
                add_block(&mut h, nucleus, ib, &transpose(&vb));
            }
            // Density transform for the nuclear-nuclear term.
            let (mut re, mut im) = (vec![0.0; ng], vec![0.0; ng]);
            pair_density_transform(sa, &sb, &gs, &d_cart, &mut re, &mut im);
            Some((h, re, im))
        })
        .collect();
    let mut h = Matrix::zeros(n, n);
    let (mut rho_re, mut rho_im) = (vec![0.0; ng], vec![0.0; ng]);
    for (part, re, im) in partials {
        for (x, y) in h.as_mut_slice().iter_mut().zip(part.as_slice()) {
            *x += *y;
        }
        for g in 0..ng {
            rho_re[g] += re[g];
            rho_im[g] += im[g];
        }
    }
    // Nuclear-nuclear: Re sum_G w Z_n (-G_x G_y) e^{-iG.R_n} rho_D(G).
    for (nucleus, (at, z)) in mol.atoms.iter().zip(charges).enumerate() {
        for (gi, g) in gs.iter().enumerate() {
            let phase = g.dot(at.position);
            let value = w[gi] * z * (phase.cos() * rho_re[gi] + phase.sin() * rho_im[gi]);
            for x in 0..3 {
                for y in 0..3 {
                    h[(3 * nucleus + x, 3 * nucleus + y)] -= value * g.get(x) * g.get(y);
                }
            }
        }
    }
    // The uniform background multiplies the same truncated overlap that
    // enters the energy and force, including the per-shell-pair cutoff.
    let background = two_centre_hessian(
        mb,
        lattice,
        basis_cutoff,
        Operator::Overlap,
        density,
        natoms,
    );
    for (x, y) in h.as_mut_slice().iter_mut().zip(background.as_slice()) {
        *x += shift * *y;
    }
    h
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::periodic::integrals::ImageDerivBlock;
    use crate::system::Atom;

    /// One s and one p shell per hydrogen, each contracted from a compact
    /// and a diffuse primitive.
    pub(crate) const BASIS: &str =
        "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\n****\n";
    pub(crate) const STEPS: [f64; 4] = [4e-3, 2e-3, 1e-3, 5e-4];
    pub(crate) const CUTOFF: f64 = 11.0;

    pub(crate) fn lattice() -> Lattice {
        Lattice::new(
            Vec3::new(4.2, 0.0, 0.0),
            Vec3::new(0.3, 4.6, 0.0),
            Vec3::new(0.2, -0.25, 4.9),
        )
    }

    pub(crate) fn positions() -> [Vec3; 2] {
        [Vec3::new(0.1, 0.2, -0.3), Vec3::new(1.2, 0.9, 0.8)]
    }

    /// The cell with atom `atom` displaced by `h` along `axis`.
    pub(crate) fn system(atom: usize, axis: usize, h: f64) -> (Molecule, MolBasis) {
        let mut atoms: Vec<Atom> = positions()
            .iter()
            .map(|p| Atom { z: 1, position: *p })
            .collect();
        match axis {
            0 => atoms[atom].position.x += h,
            1 => atoms[atom].position.y += h,
            _ => atoms[atom].position.z += h,
        }
        let molecule = Molecule::new(atoms);
        let set = crate::basis::g94::parse("hessian-test", BASIS).unwrap();
        let basis = MolBasis::build(&molecule, &set).unwrap();
        (molecule, basis)
    }

    /// Density images with `D(-T) = D(T)^T`, decaying with `|T|`.
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

    pub(crate) fn contract_derivatives(
        blocks: &[ImageDerivBlock],
        density: &[ImageBlock],
        natoms: usize,
    ) -> Vec<f64> {
        let mut g = vec![0.0; 3 * natoms];
        for block in blocks {
            if let Some(d) = find_block(density, block.translation) {
                for atom in 0..natoms {
                    for axis in 0..3 {
                        g[3 * atom + axis] += d.frobenius_dot(&block.d[atom][axis]);
                    }
                }
            }
        }
        g
    }

    /// Fit each sampled column of `hessian` against central differences of
    /// `gradient(atom, axis, h)` over the four steps.
    pub(crate) fn check_columns(
        label: &str,
        hessian: &Matrix,
        gradient: impl Fn(usize, usize, f64) -> Vec<f64>,
    ) {
        for (atom, axis) in [(0usize, 0usize), (1, 2), (0, 1)] {
            let column = 3 * atom + axis;
            let errors: Vec<f64> = STEPS
                .iter()
                .map(|&h| {
                    let (plus, minus) = (gradient(atom, axis, h), gradient(atom, axis, -h));
                    (0..plus.len())
                        .map(|row| {
                            ((plus[row] - minus[row]) / (2.0 * h) - hessian[(row, column)]).abs()
                        })
                        .fold(0.0_f64, f64::max)
                })
                .collect();
            let (x, y): (Vec<f64>, Vec<f64>) = STEPS
                .iter()
                .zip(&errors)
                .map(|(h, e)| (h.ln(), e.ln()))
                .unzip();
            let n = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
            let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
            let (p, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
            let largest = (0..hessian.rows)
                .map(|r| hessian[(r, column)].abs())
                .fold(0.0_f64, f64::max);
            println!("{label} column ({atom},{axis}) max |H| {largest:.6e}: errors {errors:?} p = {p:.6} R2 = {r2:.8}");
            assert!(
                (p - 2.0).abs() < 0.05 && r2 > 0.9999,
                "{label} ({atom},{axis}): p = {p}, R2 = {r2}"
            );
        }
        for row in 0..hessian.rows {
            for col in 0..hessian.cols {
                assert!(
                    (hessian[(row, col)] - hessian[(col, row)]).abs() < 1e-10,
                    "{label} symmetry"
                );
            }
        }
    }

    #[test]
    fn two_centre_gamma_hessians_pass_four_step_fits() {
        let lattice = lattice();
        let (_, basis) = system(0, 0, 0.0);
        let d = images(&lattice, basis.nao_sph);
        for (label, op) in [
            ("kinetic", Operator::Kinetic),
            ("overlap", Operator::Overlap),
        ] {
            let hessian = two_centre_hessian(&basis, &lattice, CUTOFF, op, &d, 2);
            check_columns(label, &hessian, |atom, axis, h| {
                let (_, basis) = system(atom, axis, h);
                contract_derivatives(
                    &crate::periodic::integrals::image_derivative_blocks(
                        &basis, &lattice, CUTOFF, op,
                    ),
                    &d,
                    2,
                )
            });
            // Rigid translation of the whole crystal: rows sum to zero.
            for row in 0..6 {
                for b in 0..3 {
                    let total: f64 = (0..2).map(|atom| hessian[(row, 3 * atom + b)]).sum();
                    assert!(
                        total.abs() < 1e-10,
                        "{label} translation row {row}: {total:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn screened_hartree_gamma_hessian_passes_four_step_fits() {
        use crate::periodic::screened_coulomb::{
            screened_hartree_derivatives, screened_hartree_hessian, HartreeScreening,
        };
        let lattice = lattice();
        let (_, basis) = system(0, 0, 0.0);
        let d = images(&lattice, basis.nao_sph);
        for compact_exponent in [1.0, 3.0] {
            let screening = HartreeScreening {
                compact_exponent,
                tolerance: 1e-15,
                max_translation: None,
            };
            let hessian = screened_hartree_hessian(&basis, &lattice, &d, &screening, 2).unwrap();
            check_columns(
                &format!("Hartree (split {compact_exponent})"),
                &hessian,
                |atom, axis, h| {
                    let (_, basis) = system(atom, axis, h);
                    screened_hartree_derivatives(&basis, &lattice, &d, &screening, 2)
                        .unwrap()
                        .gradient
                        .iter()
                        .flat_map(|g| [g.x, g.y, g.z])
                        .collect()
                },
            );
        }
    }

    /// Gamma-point Hartree Fock derivatives at fixed density against a
    /// four-step fit of central differences of the Bloch-summed J(k).
    #[test]
    fn screened_hartree_fock_derivatives_pass_four_step_fits() {
        use crate::periodic::integrals::bloch;
        use crate::periodic::screened_coulomb::{
            screened_hartree_fock_derivatives, screened_hartree_image_blocks, HartreeScreening,
        };
        let lattice = lattice();
        let (_, basis) = system(0, 0, 0.0);
        let d = images(&lattice, basis.nao_sph);
        let k_points = [Vec3::zero(), Vec3::new(0.31, -0.12, 0.2)];
        for compact_exponent in [1.0, 3.0] {
            let screening = HartreeScreening {
                compact_exponent,
                tolerance: 1e-15,
                max_translation: None,
            };
            let derivatives =
                screened_hartree_fock_derivatives(&basis, &lattice, &d, &screening, 2).unwrap();
            for (atom, axis) in [(0usize, 0usize), (1, 2)] {
                let column = 3 * atom + axis;
                let errors: Vec<f64> = STEPS
                    .iter()
                    .map(|&h| {
                        let j = |h: f64| {
                            let (_, basis) = system(atom, axis, h);
                            screened_hartree_image_blocks(&basis, &lattice, &d, &screening).unwrap()
                        };
                        let (plus, minus) = (j(h), j(-h));
                        let mut worst = 0.0_f64;
                        for k in &k_points {
                            let (pr, pi) = bloch(&plus, *k);
                            let (mr, mi) = bloch(&minus, *k);
                            let (ar, ai) = bloch(&derivatives[column], *k);
                            for idx in 0..pr.as_slice().len() {
                                let fr = (pr.as_slice()[idx] - mr.as_slice()[idx]) / (2.0 * h);
                                let fi = (pi.as_slice()[idx] - mi.as_slice()[idx]) / (2.0 * h);
                                worst = worst
                                    .max((fr - ar.as_slice()[idx]).abs())
                                    .max((fi - ai.as_slice()[idx]).abs());
                            }
                        }
                        worst
                    })
                    .collect();
                let (x, y): (Vec<f64>, Vec<f64>) = STEPS
                    .iter()
                    .zip(&errors)
                    .map(|(h, e)| (h.ln(), e.ln()))
                    .unzip();
                let n = 4.0;
                let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
                let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
                let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
                let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
                let (p, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
                println!("Hartree Fock derivative (split {compact_exponent}) column ({atom},{axis}): errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
        }
    }

    #[test]
    fn nuclear_attraction_gamma_hessian_passes_four_step_fits() {
        let lattice = lattice();
        let (molecule, basis) = system(0, 0, 0.0);
        let d = images(&lattice, basis.nao_sph);
        let charges = vec![1.0, 1.0];
        let p = EwaldParams::for_accuracy(&lattice, 2, 1e-12);
        let hessian =
            nuclear_attraction_hessian(&basis, &molecule, &charges, &lattice, CUTOFF, p, &d);
        check_columns("nuclear attraction", &hessian, |atom, axis, h| {
            let (molecule, basis) = system(atom, axis, h);
            contract_derivatives(
                &crate::periodic::nuclear::nuclear_derivative_image_blocks(
                    &basis, &molecule, &charges, &lattice, CUTOFF, p,
                ),
                &d,
                2,
            )
        });
        for row in 0..6 {
            for b in 0..3 {
                let total: f64 = (0..2).map(|atom| hessian[(row, 3 * atom + b)]).sum();
                assert!(total.abs() < 1e-9, "translation row {row}: {total:e}");
            }
        }
    }

    #[test]
    fn nuclear_attraction_hessian_matches_all_entries_with_partial_image_blocks() {
        fit_nuclear_attraction_with_partial_image_blocks(false);
    }

    #[test]
    fn nuclear_attraction_hessian_uses_the_same_charge_images_as_the_force() {
        fit_nuclear_attraction_with_partial_image_blocks(true);
    }

    fn fit_nuclear_attraction_with_partial_image_blocks(truncated_real_sum: bool) {
        let lattice = Lattice::new(
            Vec3::new(8.4, 0.7, 0.3),
            Vec3::new(0.4, 8.9, 0.6),
            Vec3::new(0.2, -0.5, 9.2),
        );
        let positions = [
            Vec3::new(0.13, -0.21, 0.07),
            Vec3::new(2.0, 0.45, -0.31),
            Vec3::new(-0.46, 1.29, 1.62),
        ];
        let set = crate::basis::g94::parse(
            "scalar geometry fixture",
            include_str!("../../tests/data/periodic_scalar_ecp.g94"),
        )
        .unwrap();
        let build = |column: usize, h: f64| {
            let mut positions = positions;
            let displacement =
                std::array::from_fn::<_, 3, _>(|i| if i == column % 3 { h } else { 0.0 });
            positions[column / 3] += Vec3::new(displacement[0], displacement[1], displacement[2]);
            let molecule = Molecule::new(
                [1, 11, 1]
                    .into_iter()
                    .zip(positions)
                    .map(|(z, position)| Atom { z, position })
                    .collect(),
            );
            let basis = MolBasis::build(&molecule, &set).unwrap();
            (molecule, basis)
        };
        let (molecule, basis) = build(0, 0.0);
        let charges = basis.effective_charges(&molecule);
        let cutoff = 7.1;
        let selected = short_range_images(&basis, &lattice, cutoff);
        assert!(selected.iter().any(|image| image.translation.norm() > 0.0
            && image.shell_pairs.len() < basis.shells.len().pow(2)));
        let mut p = EwaldParams::for_accuracy(&lattice, 3, 1e-12);
        if truncated_real_sum {
            // Keep the actual translated-charge list fixed over the displaced
            // geometries. Slow real-space decay exposes an extra charge screen
            // in the Hessian even when it drops only tiny terms at the default split.
            p.alpha = 0.12;
            p.rcut = 3.5;
        }
        let density = images(&lattice, basis.nao_sph);
        let hessian =
            nuclear_attraction_hessian(&basis, &molecule, &charges, &lattice, cutoff, p, &density);
        let labels: Vec<_> = (0..9)
            .flat_map(|col| {
                (0..9).map(move |row| format!("screened nuclear attraction [{row},{col}]"))
            })
            .collect();
        let mut samples = std::collections::BTreeMap::new();
        let mut failures = Vec::new();
        use stencils::Stencil;
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
                ([0.01, 0.007, 0.005, 0.0035], Stencil::Second),
                ([0.02, 0.015, 0.01, 0.0075], Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; 81];
                for col in 0..9 {
                    let mut at = |h: f64| {
                        *samples.entry((col, h.to_bits())).or_insert_with(|| {
                            let (molecule, basis) = build(col, h);
                            let blocks = crate::periodic::nuclear::nuclear_derivative_image_blocks(
                                &basis, &molecule, &charges, &lattice, cutoff, p,
                            );
                            let gradient = contract_derivatives(&blocks, &density, 3);
                            std::array::from_fn::<_, 9, _>(|i| gradient[i])
                        })
                    };
                    for (step, &h) in steps.iter().enumerate() {
                        let (plus, minus) = (at(h), at(-h));
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            ([0.0; 9], [0.0; 9])
                        };
                        for row in 0..9 {
                            let fd =
                                stencil.first(plus[row], minus[row], plus2[row], minus2[row], h);
                            errors[9 * col + row][step] = (fd - hessian[(row, col)]).abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("screened nuclear-attraction Hessian", &failures);
    }
}
