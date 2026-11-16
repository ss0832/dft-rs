// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic Hartree matrix `J(T)`, by the same Ewald split as
//! [`super::nuclear`].
//!
//! # Why the split is the same one twice
//!
//! `J` and `V` are the same integral against two different charge
//! distributions — nuclei for `V`, the electron density for `J` — so they
//! have the same conditional-convergence problem and take the same cure:
//!
//! ```text
//! 1/r = erfc(alpha r)/r + erf(alpha r)/r
//! ```
//!
//! - the `erfc` half is a genuinely short-ranged four-centre lattice sum,
//!   evaluated with the existing ERI engine and `CoulombKernel::Erfc`;
//! - the `erf` half needs only the density's Fourier components, which
//!   [`super::planewave::plane_wave_density`] supplies, times the same
//!   Gaussian-damped kernel the nuclear term uses.
//!
//! That the two share the kernel is worth more than the saved code: the
//! `G = 0` background of `J` must cancel the ones in `V` and in the
//! nuclear-nuclear energy for a neutral cell, and cancellation between
//! terms written in different conventions is the classic way a periodic
//! total energy comes out wrong by a constant that looks physical.
//!
//! # The G = 0 term
//!
//! `-pi N_e / (alpha^2 V)` times the overlap, with `N_e` the electron count
//! per cell — the exact counterpart of `V`'s `+pi Z_tot / (alpha^2 V)`.
//! Together with the nuclear-nuclear `-pi Q^2 / (2 alpha^2 V)` they cancel
//! when `Z_tot = N_e`. None of the three is meaningful alone.
//!
//! # Cost
//!
//! The real-space half builds an explicit supercell of every image within
//! the cutoff and runs shell quartets over it, which is `O(N_img^2)` in the
//! number of images. That is the foundation-layer implementation: obviously
//! correct, and slow enough that a production periodic SCF will want
//! density fitting instead. It is the reference the faster path would be
//! checked against.

use crate::basis::MolBasis;
use crate::integrals::eri::eri_quartet;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::ewald::EwaldParams;
use crate::periodic::integrals::{concat, shifted, ImageBlock};
use crate::periodic::planewave::{plane_wave_contraction, plane_wave_density};
use crate::periodic::Lattice;
use rayon::prelude::*;

/// The density block for a given translation, or `None` if the caller did
/// not supply one (which means zero).
fn density_at(density: &[ImageBlock], t: Vec3) -> Option<&Matrix> {
    density
        .iter()
        .find(|b| (b.translation - t).norm() < 1e-8)
        .map(|b| &b.matrix)
}

/// Borrow the density block for every ordered pair of cells in `images`.
///
/// The same `D(T)` generally occurs for many `(c,d)` pairs.  Keeping
/// references here is important: cloning an `nao x nao` matrix for every
/// pair gives an avoidable `O(ncell^2 * nao^2)` allocation before any ERI is
/// evaluated.  The row-major pair order is the one used by the contraction
/// below and is deliberately unchanged.
fn density_pair_blocks<'a>(density: &'a [ImageBlock], images: &[Vec3]) -> Vec<Option<&'a Matrix>> {
    let ncell = images.len();
    (0..ncell * ncell)
        .map(|index| {
            let (c, d) = (index / ncell, index % ncell);
            density_at(density, images[d] - images[c])
        })
        .collect()
}

/// `< mu(0) | J[D] | nu(T) >` for every translation within `cutoff`.
///
/// `density` is the periodic density matrix as image blocks `D(T)`, in the
/// same form this function returns; it must satisfy `D(-T) = D(T)^T`, which
/// is what makes the resulting `J` Hermitian at every k point.
///
/// One cutoff serves both roles here — it bounds the ket images of the
/// output blocks AND the supercell the real-space quartets run over — so it
/// has to cover the `erfc` screening length `p.rcut` as well as the basis
/// extent. Splitting it in two would save work; conflating it with a pure
/// overlap cutoff would silently truncate the electrostatics.
///
/// The second role is the one that is easy to under-serve. The reciprocal
/// half already contains the ket density's images out to infinity, so if
/// the real-space half stops before `erfc` has died the two halves are
/// summing different lattices and `alpha` stops cancelling. That shows up
/// as a small `alpha` dependence rather than as an obvious error, which is
/// exactly why the gate below compares across three of them.
pub fn hartree_image_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    p: EwaldParams,
) -> Vec<ImageBlock> {
    let n = mb.nao_sph;
    let nsh = mb.shells.len();
    let v = lattice.volume().abs();
    let imgs = lattice.translations_within(cutoff);
    let ncell = imgs.len();

    // The supercell: home cell first in `imgs` order, so the AO index of
    // (cell c, AO i) is `c * n + i` and the shell index is `c * nsh + s`.
    let mut sup = shifted(mb, imgs[0]);
    for t in &imgs[1..] {
        sup = concat(&sup, &shifted(mb, *t), 0);
    }

    // D between cells c and d depends only on their difference.
    let dblocks = density_pair_blocks(density, &imgs);

    // ---- Reciprocal half, shared by every output block ----
    let gs = lattice.reciprocal_within(p.gcut);
    let (mut c_re, mut c_im) = (vec![0.0; gs.len()], vec![0.0; gs.len()]);
    let mut ne = 0.0;
    for b in density {
        let comb = concat(mb, &shifted(mb, b.translation), 0);
        // Embed D(T) in the combined basis as the (bra home, ket shifted)
        // quarter: that is the block whose plane-wave elements are
        // `< lambda(0) | exp(iG.r) | sigma(T) >`.
        let mut dfull = Matrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                dfull[(i, n + j)] = b.matrix[(i, j)];
            }
        }
        let (re, im) = plane_wave_density(&comb, &gs, &dfull);
        for gi in 0..gs.len() {
            c_re[gi] += re[gi];
            c_im[gi] += im[gi];
        }
        let s = crate::integrals::overlap_matrix(&comb);
        for i in 0..n {
            for j in 0..n {
                ne += b.matrix[(i, j)] * s[(i, n + j)];
            }
        }
    }
    // `n~(G)` is the conjugate of what was just accumulated, and the kernel
    // multiplies it: c(G) = (4 pi / V) exp(-G^2/4 alpha^2)/G^2 * n~(G).
    for gi in 0..gs.len() {
        let g2 = gs[gi].norm2();
        let w = (4.0 * std::f64::consts::PI / v) * (-g2 / (4.0 * p.alpha * p.alpha)).exp() / g2;
        c_re[gi] *= w;
        c_im[gi] *= -w;
    }
    let shift = -std::f64::consts::PI * ne / (p.alpha * p.alpha * v);

    // ---- Assemble, one block per translation ----
    let blocks: Vec<(Vec3, Matrix)> = imgs
        .par_iter()
        .enumerate()
        .map(|(tb, t)| {
            let mut m = Matrix::zeros(n, n);
            // Real space: screened quartets over the supercell.
            for sa in 0..nsh {
                let (la, oa) = (&mb.shells[sa], mb.shells[sa].sph_off);
                for sb in 0..nsh {
                    let ob = mb.shells[sb].sph_off;
                    let bra_b = &sup.shells[tb * nsh + sb];
                    for cd in 0..ncell * ncell {
                        let Some(db) = dblocks[cd] else { continue };
                        let (c, d) = (cd / ncell, cd % ncell);
                        for sc in 0..nsh {
                            let oc = mb.shells[sc].sph_off;
                            for sd in 0..nsh {
                                let od = mb.shells[sd].sph_off;
                                // Skip a ket pair whose density block is
                                // numerically absent; the quartet is the
                                // expensive part, the test is free.
                                let (nc, nd) = (
                                    crate::basis::n_sph(mb.shells[sc].l),
                                    crate::basis::n_sph(mb.shells[sd].l),
                                );
                                let mut dmax = 0.0_f64;
                                for k in 0..nc {
                                    for l in 0..nd {
                                        dmax = dmax.max(db[(oc + k, od + l)].abs());
                                    }
                                }
                                if dmax < 1e-14 {
                                    continue;
                                }
                                let q = eri_quartet(
                                    la,
                                    bra_b,
                                    &sup.shells[c * nsh + sc],
                                    &sup.shells[d * nsh + sd],
                                    CoulombKernel::Erfc { omega: p.alpha },
                                );
                                for i in 0..q.na {
                                    for j in 0..q.nb {
                                        let mut acc = 0.0;
                                        for k in 0..q.nc {
                                            for l in 0..q.nd {
                                                acc += db[(oc + k, od + l)] * q.get(i, j, k, l);
                                            }
                                        }
                                        m[(oa + i, ob + j)] += acc;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Reciprocal space plus the G = 0 background, both of which ride
            // on the same combined basis the overlap uses.
            let comb = concat(mb, &shifted(mb, *t), 0);
            let (long, _) = plane_wave_contraction(&comb, &gs, &c_re, &c_im);
            let ovlp = crate::integrals::overlap_matrix(&comb);
            for i in 0..n {
                for j in 0..n {
                    m[(i, j)] += long[(i, n + j)] + shift * ovlp[(i, n + j)];
                }
            }
            (*t, m)
        })
        .collect();

    blocks
        .into_iter()
        .filter(|(_, m)| m.as_slice().iter().any(|v| v.abs() > 1e-14))
        .map(|(translation, matrix)| ImageBlock {
            translation,
            matrix,
        })
        .collect()
}

/// Full periodic Hartree matrix at `k`, as `(real, imaginary)`.
pub fn hartree_matrix_k(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    p: EwaldParams,
    k: Vec3,
) -> (Matrix, Matrix) {
    let blocks = hartree_image_blocks(mb, lattice, density, cutoff, p);
    crate::periodic::integrals::bloch(&blocks, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::system::{Atom, Molecule};

    fn h2() -> (Molecule, MolBasis) {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.0, 0.7),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb)
    }

    /// A density confined to the home cell, as a one-element image list.
    fn home_density(m: Matrix) -> Vec<ImageBlock> {
        vec![ImageBlock {
            translation: Vec3::zero(),
            matrix: m,
        }]
    }

    /// The borrowed cell-pair table must preserve the legacy row-major
    /// `(c,d)` ordering and the numerical values that the former cloned
    /// table supplied.  A deliberately missing `D(+-2a)` also pins the
    /// zero-block convention.
    #[test]
    fn density_pair_table_borrows_without_changing_the_mapping() {
        let a = Vec3::new(3.0, 0.0, 0.0);
        let images = vec![Vec3::zero(), a, a * 2.0];
        let density = [
            ImageBlock {
                translation: Vec3::zero(),
                matrix: Matrix::from_row_major(1, 1, vec![10.0]),
            },
            ImageBlock {
                translation: a,
                matrix: Matrix::from_row_major(1, 1, vec![20.0]),
            },
            ImageBlock {
                translation: a * -1.0,
                matrix: Matrix::from_row_major(1, 1, vec![30.0]),
            },
        ];

        let borrowed = density_pair_blocks(&density, &images);
        let legacy_cloned: Vec<Option<Matrix>> = (0..images.len() * images.len())
            .map(|index| {
                let (c, d) = (index / images.len(), index % images.len());
                density_at(&density, images[d] - images[c]).cloned()
            })
            .collect();
        let values: Vec<Option<f64>> = borrowed
            .iter()
            .map(|block| block.map(|matrix| matrix[(0, 0)]))
            .collect();
        assert_eq!(
            values,
            vec![
                Some(10.0),
                Some(20.0),
                None,
                Some(30.0),
                Some(10.0),
                Some(20.0),
                None,
                Some(30.0),
                Some(10.0),
            ]
        );
        for (reference, cloned) in borrowed.iter().zip(&legacy_cloned) {
            match (reference, cloned) {
                (Some(reference), Some(cloned)) => {
                    assert_eq!(reference.as_slice(), cloned.as_slice());
                    assert!(
                        density
                            .iter()
                            .any(|block| std::ptr::eq(*reference, &block.matrix)),
                        "cell-pair entry must borrow an original density matrix"
                    );
                }
                (None, None) => {}
                _ => panic!("borrowed and cloned density mappings differ"),
            }
        }
    }

    /// The splitting parameter must cancel out of `J`.
    ///
    /// This is the sharpest available check and it covers the `G = 0` term
    /// as well: the background is `-pi N_e / (alpha^2 V)`, explicitly
    /// alpha-dependent, so a wrong coefficient there cannot survive a
    /// comparison across three alphas even though it is invisible at any
    /// one of them. The density is deliberately CHARGED (a normal one),
    /// because a neutral one would make that term vanish and hide it.
    #[test]
    fn total_is_independent_of_the_splitting_parameter() {
        let (_, mb) = h2();
        let lat = Lattice::cubic(7.0);
        let n = mb.nao_sph;
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            d[(i, i)] = 0.9;
        }
        d[(0, 1)] = 0.3;
        d[(1, 0)] = 0.3;
        let density = home_density(d);
        let mut mats = Vec::new();
        for alpha in [0.75_f64, 0.9, 1.05] {
            let p = EwaldParams {
                alpha,
                // erfc(4.9) and exp(-(9.6/2)^2) are both below 1e-10. The
                // cutoff below has to clear `rcut` plus the span of both
                // charge distributions, or the two halves sum different
                // lattices.
                rcut: 4.9 / alpha,
                gcut: 9.6 * alpha,
            };
            mats.push(hartree_matrix_k(
                &mb,
                &lat,
                &density,
                12.0,
                p,
                Vec3::new(0.13, -0.07, 0.21),
            ));
        }
        for w in mats.windows(2) {
            let dre = w[0].0.max_abs_diff(&w[1].0);
            let dim = w[0].1.max_abs_diff(&w[1].1);
            assert!(
                dre < 1e-6 && dim < 1e-6,
                "J(k) depends on alpha: real {dre:e}, imaginary {dim:e}"
            );
        }
    }

    /// In a large cell, a density with NO NET CHARGE AND NO DIPOLE must
    /// reproduce the molecular Hartree matrix.
    ///
    /// Both conditions are load-bearing, and neither is a convenience:
    ///
    /// - a net charge makes the periodic value differ from the molecular
    ///   one by the background plus a Madelung-type term dying only as
    ///   `1/L` — real physics of the boundary conditions;
    /// - a dipole leaves a `1/L^3` tail, which at any cell size one can
    ///   afford to compute still swamps the tolerance a matrix comparison
    ///   deserves.
    ///
    /// Killing both leaves a quadrupolar tail, and a comparison against a
    /// `J` already gated against PySCF through the molecular SCF. The
    /// density is built symmetric under `z -> -z` so the dipole vanishes by
    /// construction rather than by cancellation of computed numbers.
    #[test]
    fn a_neutral_density_reproduces_the_molecular_hartree_matrix() {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let n = mb.nao_sph;
        let s = crate::integrals::overlap_matrix(&mb);
        // D = [[b, 1], [1, b]] with b = -S01/S00: symmetric, so no dipole,
        // and Tr[D S] = 2b S00 + 2 S01 = 0, so no net charge.
        let b = -s[(0, 1)] / s[(0, 0)];
        let mut d = Matrix::zeros(n, n);
        d[(0, 0)] = b;
        d[(1, 1)] = b;
        d[(0, 1)] = 1.0;
        d[(1, 0)] = 1.0;
        let trace: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| d[(i, j)] * s[(i, j)])
            .sum();
        assert!(trace.abs() < 1e-12, "the test density is not neutral");

        let lat = Lattice::cubic(40.0);
        let p = EwaldParams {
            alpha: 0.18,
            rcut: 27.0,
            gcut: 1.8,
        };
        let blocks = hartree_image_blocks(&mb, &lat, &home_density(d.clone()), 14.0, p);
        let got = crate::periodic::integrals::gamma_sum(&blocks);

        let zero = Matrix::zeros(n, n);
        let engine = crate::scf::fock::JkEngine::new(
            &mb,
            crate::scf::EriMode::Direct {
                screen_threshold: 1e-16,
            },
            crate::integrals::CoulombKernel::Coulomb,
        )
        .unwrap();
        let (j, _) = engine.build(&mb, &d, [&d, &zero], false).unwrap();
        let worst = got.max_abs_diff(&j);
        assert!(
            worst < 1e-5,
            "periodic J differs from the molecular one at L = 40: {worst:e}"
        );
    }

    /// The whole periodic electrostatics against one number from
    /// [`crate::periodic::ewald::ewald_energy`].
    ///
    /// This is the gate that ties the three Coulomb terms together. Replace
    /// the electron density with tight normalised Gaussians of exponent
    /// `beta`; at large `beta` they are point charges, so
    ///
    /// ```text
    /// E_nn + Tr[D V] + 1/2 Tr[D J]
    ///     = E_Ewald({Z_A - q_A})
    ///       + 1/2 sum_A q_A^2 sqrt(2 gamma / pi)    [blob with itself]
    ///       - sum_A q_A Z_A 2 sqrt(gamma / pi)      [blob with its own nucleus]
    /// ```
    ///
    /// with `gamma = 2 beta`, NOT `beta`: the basis function is
    /// `exp(-beta r^2)` but the charge density is its square, so the blob
    /// the electrostatics actually sees is twice as tight as the orbital.
    /// Using `beta` here gives a wrong answer that is off by a plausible
    /// few Hartree rather than by an obvious factor.
    ///
    /// The two correction terms are the interactions a point-charge Ewald
    /// sum excludes as self-energy but a Gaussian one does not; both are
    /// closed forms, so nothing is fitted. What makes this decisive is that
    /// the right-hand side comes from the point-charge code — already
    /// pinned to the rocksalt Madelung constant to ten digits — while the
    /// left-hand side comes from three matrix routines with three separately
    /// chosen `G = 0` conventions. Those conventions cancel only if all
    /// three are right, and a common error in any one of them shifts the
    /// total by a constant that looks perfectly physical.
    ///
    /// The cell is deliberately CHARGE-NEUTRAL overall but with non-zero
    /// site charges, so the background terms are individually large and
    /// only cancel in the sum.
    #[test]
    fn electrostatic_total_matches_the_point_charge_ewald_energy() {
        use crate::basis::{MolBasis, MolShell};
        use crate::periodic::ewald::ewald_energy;
        use crate::periodic::nuclear::nuclear_image_blocks;

        let beta = 50.0_f64;
        let pos = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 2.0)];
        let z = [1.0_f64, 1.0];
        let q = [1.5_f64, 0.5];
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: pos[0],
            },
            Atom {
                z: 1,
                position: pos[1],
            },
        ]);
        // One normalised s primitive per atom: a Gaussian blob of charge.
        let norm = (2.0 * beta / std::f64::consts::PI).powf(0.75);
        let mb = MolBasis {
            shells: (0..2)
                .map(|a| MolShell {
                    l: 0,
                    atom: a,
                    center: pos[a],
                    exps: vec![beta],
                    coefs: vec![norm],
                    sph_off: a,
                    cart_off: a,
                })
                .collect(),
            nao_sph: 2,
            nao_cart: 2,
            max_l: 0,
            ecps: Vec::new(),
            n_core_electrons: 0,
        };
        let s = crate::integrals::overlap_matrix(&mb);
        assert!(
            (s[(0, 0)] - 1.0).abs() < 1e-12 && s[(0, 1)].abs() < 1e-12,
            "the blobs are not normalised and separate"
        );
        let mut d = Matrix::zeros(2, 2);
        d[(0, 0)] = q[0];
        d[(1, 1)] = q[1];

        let lat = Lattice::cubic(8.0);
        let p = EwaldParams {
            alpha: 0.75,
            rcut: 6.6,
            gcut: 7.2,
        };
        let cutoff = 12.0;

        let vb = nuclear_image_blocks(&mb, &mol, &z, &lat, cutoff, p);
        let jb = hartree_image_blocks(&mb, &lat, &home_density(d.clone()), cutoff, p);
        let contract = |blocks: &[ImageBlock]| -> f64 {
            // Only the home density block is non-zero, so only T = 0 of the
            // operator contributes to Tr over the lattice.
            blocks
                .iter()
                .find(|b| b.translation.norm() < 1e-12)
                .map(|b| {
                    (0..2)
                        .flat_map(|i| (0..2).map(move |j| (i, j)))
                        .map(|(i, j)| d[(i, j)] * b.matrix[(i, j)])
                        .sum()
                })
                .unwrap_or(0.0)
        };
        let e_ne = contract(&vb);
        let e_ee = 0.5 * contract(&jb);
        let e_nn = ewald_energy(&lat, &pos, &z, p);

        let site: Vec<f64> = (0..2).map(|a| z[a] - q[a]).collect();
        let e_pc = ewald_energy(&lat, &pos, &site, p);
        let sqrt_pi = std::f64::consts::PI.sqrt();
        // The charge density is the SQUARE of the basis function, so its
        // Gaussian exponent is 2 beta.
        let gamma = 2.0 * beta;
        let self_ee: f64 = (0..2)
            .map(|a| 0.5 * q[a] * q[a] * (2.0 * gamma).sqrt() / sqrt_pi)
            .sum();
        let self_ne: f64 = (0..2)
            .map(|a| -q[a] * z[a] * 2.0 * gamma.sqrt() / sqrt_pi)
            .sum();

        let got = e_nn + e_ne + e_ee;
        let want = e_pc + self_ee + self_ne;
        assert!(
            (got - want).abs() < 1e-6,
            "electrostatic total {got} vs point-charge Ewald + self terms {want} \
             (E_nn {e_nn}, E_ne {e_ne}, E_ee {e_ee}, E_pc {e_pc})"
        );
    }

    /// `J` must be Hermitian at every k point, which for real image blocks
    /// means `J(-T) = J(T)^T`.
    ///
    /// The identity is exact for the infinite sums, so what it really
    /// measures is the truncation: the real-space ket lattice is centred on
    /// the home cell, and the `+T` and `-T` blocks see it from opposite
    /// sides. A cutoff that clears the screening length makes the asymmetry
    /// vanish with it, which is why this is a useful independent check on
    /// the cutoff and not merely a symmetry restatement.
    #[test]
    fn image_blocks_obey_the_hermiticity_relation() {
        let (_, mb) = h2();
        let lat = Lattice::cubic(7.0);
        let n = mb.nao_sph;
        let mut d = Matrix::zeros(n, n);
        d[(0, 0)] = 1.1;
        d[(1, 1)] = 0.7;
        d[(0, 1)] = 0.25;
        d[(1, 0)] = 0.25;
        let p = EwaldParams {
            alpha: 0.9,
            rcut: 5.5,
            gcut: 8.7,
        };
        let blocks = hartree_image_blocks(&mb, &lat, &home_density(d), 12.0, p);
        let mut worst = 0.0_f64;
        for b in &blocks {
            let Some(mirror) = blocks
                .iter()
                .find(|o| (o.translation + b.translation).norm() < 1e-8)
            else {
                panic!("no -T partner for translation {:?}", b.translation);
            };
            for i in 0..n {
                for j in 0..n {
                    worst = worst.max((b.matrix[(i, j)] - mirror.matrix[(j, i)]).abs());
                }
            }
        }
        assert!(worst < 1e-9, "J(-T) is not J(T) transposed: {worst:e}");
    }
}
