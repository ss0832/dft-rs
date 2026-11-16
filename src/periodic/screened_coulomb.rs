// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Range-separated periodic Hartree matrix.
//!
//! The reference build ([`super::coulomb::hartree_image_blocks`]) sums every
//! `erfc` shell quartet of an explicit image supercell. That is exact but its
//! cost grows with the cube of the number of images, and diffuse valence
//! primitives make the `erfc` interaction long-ranged, so no fixed real-space
//! cutoff is both cheap and converged. This module evaluates the same matrix
//!
//! ```text
//! J(T)_mn = sum_{c, D, ls} (m0 nT | l_c s_{c+D}) P_ls(D)   (lattice-summed Coulomb)
//! ```
//!
//! by splitting every contracted shell into a *compact* part (primitive
//! exponents at or above `compact_exponent`) and a *diffuse* part:
//!
//! * a pair of two diffuse parts has a Fourier transform that is negligible
//!   beyond `G = 2 sqrt(2 e_c ln(1/eps))`, so every interaction in which such
//!   a pair takes part is summed exactly in reciprocal space with the bare
//!   `4 pi / G^2` kernel;
//! * an interaction of two compact pairs uses the Ewald split with
//!   `alpha^2 = e_c`: screened `erfc` quartets in real space (short-ranged,
//!   because both distributions are compact) plus the Gaussian-damped
//!   reciprocal half and the neutralizing background of that half.
//!
//! Each piece is the same Ewald sum of one pair of charge distributions with
//! its own background, which is independent of the splitting parameter; the
//! pure reciprocal pieces are its `alpha -> infinity` limit. The pieces
//! therefore add to the reference matrix exactly, up to the tolerance of the
//! screening and of the truncated sums. Quartets are screened by the Schwarz
//! bound times the density block and by the decay of the `erfc` interaction
//! between Gaussian charge distributions.

use crate::basis::{MolBasis, MolShell};
use crate::integrals::eri::eri_quartet;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::integrals::{nearby_translation_indices, ImageBlock};
use crate::periodic::planewave::{
    contracted_pair_blocks, pair_blocks_per_g, pair_density_transform, ReciprocalContraction,
};
use crate::periodic::Lattice;
use rayon::prelude::*;

/// The reciprocal sums run over one vector of every pair `(G, -G)`: the two
/// terms are complex conjugates and only real parts are used, so each kept
/// vector carries twice the weight.
const HALF_SPACE_WEIGHT: f64 = 2.0;

/// Real product of two Fourier transforms with the compact-compact term
/// removed. Form the remaining terms directly so a small diffuse component
/// survives when adding it to the compact component would round it away.
#[inline]
fn diffuse_reciprocal_product<S: crate::dual::Scalar>(
    [cr, ci]: [S; 2],
    [dr, di]: [S; 2],
    [other_cr, other_ci]: [S; 2],
    [other_dr, other_di]: [S; 2],
) -> S {
    cr * other_dr + dr * other_cr + dr * other_dr + ci * other_di + di * other_ci + di * other_di
}

/// Whether `g` is the kept member of the pair `(g, -g)`.
pub(crate) fn in_half_space(g: Vec3) -> bool {
    g.z > 0.0 || (g.z == 0.0 && (g.y > 0.0 || (g.y == 0.0 && g.x > 0.0)))
}

/// Controls of [`screened_hartree_image_blocks`].
#[derive(Clone, Copy, Debug)]
pub struct HartreeScreening {
    /// Primitive exponent separating compact from diffuse parts (Bohr^-2).
    /// The Ewald splitting parameter of compact interactions is its square
    /// root.
    pub compact_exponent: f64,
    /// Local absolute threshold for quartet and tail screening. The
    /// accumulated error also depends on the cell and the derivative.
    pub tolerance: f64,
    /// Largest separation `|A - B - T|` of a shell pair (Bohr). `None` keeps
    /// every pair above the tolerance; a cell passes its AO image cutoff, the
    /// rule of its overlap and kinetic images, so that the Hartree energy is
    /// a quadratic form on the same truncated set of pair densities.
    pub max_translation: Option<f64>,
}

impl Default for HartreeScreening {
    fn default() -> Self {
        Self {
            compact_exponent: 3.0,
            // The local bound is amplified by strain lever arms and by the
            // sum over images. Use the same tighter support for the energy,
            // Fock matrix and response so that near-cancelling stress
            // components remain insensitive to the Ewald split.
            tolerance: 1e-16,
            max_translation: None,
        }
    }
}

impl HartreeScreening {
    pub(crate) fn alpha(&self) -> f64 {
        self.compact_exponent.sqrt()
    }

    pub(crate) fn ln(&self) -> f64 {
        (1.0 / self.tolerance).ln().max(1.0)
    }

    /// Reciprocal cutoff covering both the damped compact half and the
    /// bare diffuse sums.
    pub(crate) fn gcut(&self) -> f64 {
        let ln = self.ln();
        (2.0 * self.alpha() * ln.sqrt()).max(2.0 * (2.0 * self.compact_exponent * ln).sqrt())
    }

    fn validate(&self) -> crate::Result<()> {
        if !(self.compact_exponent.is_finite()
            && self.compact_exponent > 0.0
            && self.tolerance.is_finite()
            && self.tolerance > 0.0
            && self.tolerance < 1e-4)
        {
            return Err(crate::DftError::InvalidInput(
                "Hartree screening needs a positive compact exponent and a tolerance in (0, 1e-4)"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Split exponent (Bohr^-2) that balances the reciprocal and the real-space
/// work for a cell of the given volume (Bohr^3): the reciprocal sums grow
/// with the volume, so a larger cell moves more primitives to real space.
/// Measured optimum: 3.0 at 54 Bohr^3 and 1.0 from about 300 Bohr^3. The
/// choice changes the cost only; the matrix is the same within the
/// screening tolerance.
pub fn default_compact_exponent(volume: f64) -> f64 {
    (3.0 * (54.0 / volume.abs().max(f64::MIN_POSITIVE)).powf(2.0 / 3.0)).clamp(1.0, 3.0)
}

/// The compact or diffuse primitives of one contracted shell.
#[derive(Clone, Debug)]
pub(crate) struct Part {
    pub(crate) shell: MolShell,
    pub(crate) compact: bool,
    pub(crate) min_exponent: f64,
}

pub(crate) fn split_basis(mb: &MolBasis, compact_exponent: f64) -> Vec<Part> {
    let mut out = Vec::new();
    for shell in &mb.shells {
        for compact in [true, false] {
            let (exps, coefs): (Vec<f64>, Vec<f64>) = shell
                .exps
                .iter()
                .zip(&shell.coefs)
                .filter(|(e, _)| (**e >= compact_exponent) == compact)
                .map(|(e, c)| (*e, *c))
                .unzip();
            if exps.is_empty() {
                continue;
            }
            let min_exponent = exps.iter().copied().fold(f64::INFINITY, f64::min);
            let mut part = shell.clone();
            part.exps = exps;
            part.coefs = coefs;
            out.push(Part {
                shell: part,
                compact,
                min_exponent,
            });
        }
    }
    out
}

fn moved(shell: &MolShell, t: Vec3) -> MolShell {
    let mut out = shell.clone();
    out.center += t;
    out
}

/// A significant product of part `x` (home cell) and part `y` translated by
/// `translation`.
#[derive(Clone, Debug)]
pub(crate) struct Pair {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) translation: Vec3,
    /// Both parts diffuse: summed only in reciprocal space.
    pub(crate) diffuse: bool,
    pub(crate) centre: Vec3,
    pub(crate) exponent: f64,
    pub(crate) schwarz: f64,
}

pub(crate) fn pairs(parts: &[Part], lattice: &Lattice, screening: &HartreeScreening) -> Vec<Pair> {
    let ln = screening.ln() + 4.0;
    let mut span = 0.0_f64;
    let mut mu_min = f64::INFINITY;
    for a in parts {
        for b in parts {
            span = span.max((a.shell.center - b.shell.center).norm());
            mu_min =
                mu_min.min(a.min_exponent * b.min_exponent / (a.min_exponent + b.min_exponent));
        }
    }
    let mut reach = (ln / mu_min).sqrt() + span;
    if let Some(limit) = screening.max_translation {
        reach = reach.min(limit + span);
    }
    let limit = screening.max_translation.unwrap_or(f64::INFINITY);
    let translations = lattice.translations_within(reach);
    let kernel = CoulombKernel::Erfc {
        omega: screening.alpha(),
    };
    let candidates: Vec<(usize, usize, Vec3)> = translations
        .iter()
        .flat_map(|&t| (0..parts.len()).flat_map(move |x| (0..parts.len()).map(move |y| (x, y, t))))
        .filter(|&(x, y, t)| {
            let (a, b) = (&parts[x], &parts[y]);
            let mu = a.min_exponent * b.min_exponent / (a.min_exponent + b.min_exponent);
            let r2 = (a.shell.center - b.shell.center - t).norm2();
            mu * r2 <= ln && r2.sqrt() <= limit
        })
        .collect();
    candidates
        .par_iter()
        .map(|&(x, y, t)| {
            let (a, b) = (&parts[x], &parts[y]);
            let sb = moved(&b.shell, t);
            let exponent = a.min_exponent + b.min_exponent;
            let centre = (a.shell.center * a.min_exponent + sb.center * b.min_exponent) / exponent;
            let diffuse = !a.compact && !b.compact;
            let schwarz = if diffuse {
                0.0
            } else {
                let q = eri_quartet(&a.shell, &sb, &a.shell, &sb, kernel);
                let mut largest = 0.0_f64;
                for i in 0..q.na {
                    for j in 0..q.nb {
                        largest = largest.max(q.get(i, j, i, j).abs());
                    }
                }
                largest.sqrt()
            };
            Pair {
                x,
                y,
                translation: t,
                diffuse,
                centre,
                exponent,
                schwarz,
            }
        })
        .collect()
}

/// Spherical density block of a part pair.
fn density_block(d: &Matrix, a: &MolShell, b: &MolShell) -> Matrix {
    let (na, nb) = (crate::basis::n_sph(a.l), crate::basis::n_sph(b.l));
    let mut m = Matrix::zeros(na, nb);
    for i in 0..na {
        for j in 0..nb {
            m[(i, j)] = d[(a.sph_off + i, b.sph_off + j)];
        }
    }
    m
}

fn dot(a: &Matrix, b: &Matrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x * y)
        .sum()
}

struct Ket {
    pair: usize,
    block: Matrix,
    dmax: f64,
}

/// Everything shared by the Hartree matrix and its derivatives for one
/// density: the part pairs, the density-carrying kets, the reciprocal
/// coefficients of both bra classes and the real-space cell list.
struct Prepared {
    parts: Vec<Part>,
    pairs: Vec<Pair>,
    kets: Vec<Ket>,
    with_zero: Vec<Vec3>,
    bra_compact: [Vec<f64>; 2],
    bra_diffuse: [Vec<f64>; 2],
    shift: f64,
    compact_electrons: f64,
    /// Compact density kets with their Schwarz bound `schwarz * max|D|`,
    /// in decreasing order of the bound.
    compact_kets: Vec<(usize, f64)>,
    /// Ket density transforms `[compact re, compact im, diffuse re,
    /// diffuse im]` on `with_zero`.
    transforms: [Vec<f64>; 4],
    volume: f64,
    lattice: Lattice,
    alpha: f64,
    tolerance: f64,
}

impl Prepared {
    fn new(
        mb: &MolBasis,
        lattice: &Lattice,
        density: &[ImageBlock],
        screening: &HartreeScreening,
    ) -> crate::Result<Self> {
        screening.validate()?;
        let volume = lattice.volume().abs();
        let alpha = screening.alpha();
        let parts = split_basis(mb, screening.compact_exponent);
        let pairs = pairs(&parts, lattice, screening);
        let find = |t: Vec3| {
            density
                .iter()
                .find(|b| (b.translation - t).norm() < 1e-8)
                .map(|b| &b.matrix)
        };
        let kets: Vec<Ket> = pairs
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let d = find(p.translation)?;
                let block = density_block(d, &parts[p.x].shell, &parts[p.y].shell);
                let dmax = block.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                (dmax > 0.0).then_some(Ket {
                    pair: i,
                    block,
                    dmax,
                })
            })
            .collect();
        // Fourier components of the compact and diffuse ket densities; the
        // G = 0 entry of the compact transform is its electron count.
        // Ordered by |G| so that the per-pair transforms stop where their
        // own Gaussian damping has killed every remaining term.
        let mut gs = lattice.reciprocal_within(screening.gcut());
        gs.retain(|g| in_half_space(*g));
        gs.sort_by(|a, b| a.norm2().total_cmp(&b.norm2()));
        let mut with_zero = vec![Vec3::zero()];
        with_zero.extend_from_slice(&gs);
        let ng = with_zero.len();
        // Per-thread accumulation of [compact re, compact im, diffuse re,
        // diffuse im]; the density block is moved to Cartesian components
        // once per ket so that no per-G matrix is formed.
        let accumulated = kets
            .par_iter()
            .fold(
                || vec![vec![0.0; ng]; 4],
                |mut acc, ket| {
                    let p = &pairs[ket.pair];
                    let (a, b) = (&parts[p.x].shell, &parts[p.y].shell);
                    let d_cart = crate::basis::aos::cart_to_sph(a.l)
                        .matmul_seq(&ket.block)
                        .matmul_seq(&crate::basis::aos::cart_to_sph(b.l).transpose());
                    let slot = if p.diffuse { 2 } else { 0 };
                    let (left, right) = acc.split_at_mut(slot + 1);
                    pair_density_transform(
                        a,
                        &moved(b, p.translation),
                        &with_zero,
                        &d_cart,
                        &mut left[slot],
                        &mut right[0],
                    );
                    acc
                },
            )
            .reduce(
                || vec![vec![0.0; ng]; 4],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        for (u, v) in x.iter_mut().zip(y) {
                            *u += v;
                        }
                    }
                    a
                },
            );
        let [compact_re, compact_im, diffuse_re, diffuse_im]: [Vec<f64>; 4] =
            accumulated.try_into().expect("four transform accumulators");
        let compact_electrons = compact_re[0];
        // Coefficients c(G) = w(G) conj(rho(G)) for the two bra classes.
        let mut bra_compact = [vec![0.0; ng], vec![0.0; ng]];
        let mut bra_diffuse = [vec![0.0; ng], vec![0.0; ng]];
        for g in 1..ng {
            let g2 = with_zero[g].norm2();
            let bare = HALF_SPACE_WEIGHT * 4.0 * std::f64::consts::PI / (volume * g2);
            let damped = bare * (-g2 / (4.0 * alpha * alpha)).exp();
            bra_compact[0][g] = damped * compact_re[g] + bare * diffuse_re[g];
            bra_compact[1][g] = -(damped * compact_im[g] + bare * diffuse_im[g]);
            bra_diffuse[0][g] = bare * (compact_re[g] + diffuse_re[g]);
            bra_diffuse[1][g] = -bare * (compact_im[g] + diffuse_im[g]);
        }
        let shift = -std::f64::consts::PI * compact_electrons / (alpha * alpha * volume);
        let mut compact_kets: Vec<(usize, f64)> = kets
            .iter()
            .enumerate()
            .filter(|(_, ket)| !pairs[ket.pair].diffuse)
            .map(|(index, ket)| (index, pairs[ket.pair].schwarz * ket.dmax))
            .collect();
        compact_kets.sort_by(|a, b| b.1.total_cmp(&a.1));
        crate::logger::log(&format!(
            "screened Hartree: {} parts, {} pairs ({} compact), {} density kets ({} compact), {} G",
            parts.len(),
            pairs.len(),
            pairs.iter().filter(|p| !p.diffuse).count(),
            kets.len(),
            compact_kets.len(),
            with_zero.len()
        ));
        Ok(Self {
            parts,
            pairs,
            kets,
            with_zero,
            bra_compact,
            bra_diffuse,
            shift,
            compact_electrons,
            compact_kets,
            transforms: [compact_re, compact_im, diffuse_re, diffuse_im],
            volume,
            lattice: lattice.clone(),
            alpha,
            tolerance: screening.tolerance,
        })
    }

    fn coefficients(&self, bra: &Pair) -> (&[f64], &[f64]) {
        if bra.diffuse {
            (&self.bra_diffuse[0], &self.bra_diffuse[1])
        } else {
            (&self.bra_compact[0], &self.bra_compact[1])
        }
    }

    fn reciprocal_contractions(&self) -> [ReciprocalContraction<'_>; 2] {
        [&self.bra_compact, &self.bra_diffuse]
            .map(|[re, im]| ReciprocalContraction::new(&self.with_zero, &[re], &[im]))
    }

    /// Compact kets and cells within the `erfc` reach of a compact bra.
    ///
    /// Kets are visited in decreasing order of their Schwarz bound, so the
    /// scan stops at the first one that cannot reach the tolerance, and the
    /// cells of each ket are enumerated around the bra-ket displacement
    /// rather than tested one by one.
    fn real_space_partners(&self, bra: &Pair) -> Vec<(&Ket, Vec3)> {
        let mut out = Vec::new();
        for &(index, bound) in &self.compact_kets {
            if bra.schwarz * bound < self.tolerance {
                break;
            }
            let ket = &self.kets[index];
            let kp = &self.pairs[ket.pair];
            // The interaction is bounded by the Schwarz product times the
            // erfc decay of the two distributions, so a weak pair needs a
            // shorter reach than a strong one.
            let margin = (bra.schwarz * bound / self.tolerance).ln() + 4.0;
            let reach = margin.sqrt() / decay(self.alpha, bra.exponent, kp.exponent) + 1.0;
            // |bra - ket - L| <= reach with L = -A n.
            for n in nearby_translation_indices(&self.lattice, bra.centre - kp.centre, reach) {
                out.push((ket, -self.lattice.translation(n)));
            }
        }
        out
    }
}

/// Decay constant of the `erfc` interaction of two Gaussian distributions
/// with exponents `p` and `q`.
pub(crate) fn decay(alpha: f64, p: f64, q: f64) -> f64 {
    let rho = p * q / (p + q);
    (rho * alpha * alpha / (rho + alpha * alpha)).sqrt()
}

/// `< m(0) | J[D] | n(T) >` for every translation `T` at which a bra pair
/// is significant. `density` must satisfy `D(-T) = D(T)^T`; blocks that are
/// not supplied are zero.
pub fn screened_hartree_image_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
) -> crate::Result<Vec<ImageBlock>> {
    let start = std::time::Instant::now();
    let prep = Prepared::new(mb, lattice, density, screening)?;
    let prepared = start.elapsed().as_secs_f64();
    let quartets = std::sync::atomic::AtomicUsize::new(0);
    let n = mb.nao_sph;
    let kernel = CoulombKernel::Erfc { omega: prep.alpha };
    let mut outputs: Vec<Vec3> = Vec::new();
    for p in &prep.pairs {
        if !outputs.iter().any(|t| (*t - p.translation).norm() < 1e-8) {
            outputs.push(p.translation);
        }
    }
    let zero = [Vec3::zero()];
    let reciprocal = prep.reciprocal_contractions();
    // A few central translations carry most compact quartets. Parallelize
    // over all bra pairs, then accumulate in the original pair order so
    // thread scheduling cannot change the floating-point matrix sum.
    let local_blocks: Vec<Matrix> = prep
        .pairs
        .par_iter()
        .map(|bra| {
            let t = *outputs
                .iter()
                .find(|t| (**t - bra.translation).norm() < 1e-8)
                .expect("every bra translation has an output");
            let sa = &prep.parts[bra.x].shell;
            let sb = moved(&prep.parts[bra.y].shell, t);
            let cart = reciprocal[usize::from(bra.diffuse)]
                .blocks(sa, &sb)
                .remove(0)
                .0;
            let mut local = crate::basis::aos::sph_block(sa.l, sb.l, &cart);
            if !bra.diffuse {
                let overlap = crate::basis::aos::sph_block(
                    sa.l,
                    sb.l,
                    &pair_blocks_per_g(sa, &sb, &zero)[0].0,
                );
                for (v, s) in local.as_mut_slice().iter_mut().zip(overlap.as_slice()) {
                    *v += prep.shift * s;
                }
                let partners = prep.real_space_partners(bra);
                quartets.fetch_add(partners.len(), std::sync::atomic::Ordering::Relaxed);
                for (ket, cell) in partners {
                    let kp = &prep.pairs[ket.pair];
                    let sc = moved(&prep.parts[kp.x].shell, cell);
                    let sd = moved(&prep.parts[kp.y].shell, cell + kp.translation);
                    let q = eri_quartet(sa, &sb, &sc, &sd, kernel);
                    for i in 0..q.na {
                        for j in 0..q.nb {
                            let mut acc = 0.0;
                            for k in 0..q.nc {
                                for l in 0..q.nd {
                                    acc += ket.block[(k, l)] * q.get(i, j, k, l);
                                }
                            }
                            local[(i, j)] += acc;
                        }
                    }
                }
            }
            local
        })
        .collect();
    let blocks: Vec<ImageBlock> = outputs
        .iter()
        .map(|&t| {
            let mut m = Matrix::zeros(n, n);
            for (bra, local) in prep
                .pairs
                .iter()
                .zip(&local_blocks)
                .filter(|(p, _)| (p.translation - t).norm() < 1e-8)
            {
                let oa = prep.parts[bra.x].shell.sph_off;
                let ob = prep.parts[bra.y].shell.sph_off;
                for i in 0..local.rows {
                    for j in 0..local.cols {
                        m[(oa + i, ob + j)] += local[(i, j)];
                    }
                }
            }
            ImageBlock {
                translation: t,
                matrix: m,
            }
        })
        .collect();
    crate::logger::log(&format!(
        "screened Hartree: preparation {prepared:.3} s, {} output translations, {} real-space quartets, total {:.3} s",
        outputs.len(),
        quartets.into_inner(),
        start.elapsed().as_secs_f64()
    ));
    Ok(blocks)
}

/// Hartree energy per cell and its analytic derivatives with respect to the
/// basis centres at a fixed AO density matrix.
#[derive(Clone, Debug)]
pub struct HartreeDerivatives {
    /// `E_J = 1/2 sum_T Tr[D(T)^T J(T)]` (Hartree per cell).
    pub energy: f64,
    /// `dE_J/dR_A` per atom at fixed AO density (Hartree/Bohr).
    pub gradient: Vec<Vec3>,
    /// Centre virial `sum_c (dE/dR_c)_a (R_c)_b` over every basis centre,
    /// lattice images included. With the explicit reciprocal and volume
    /// terms it forms the Hartree strain derivative.
    pub centre_virial: [[f64; 3]; 3],
    /// Electrons in the compact part of the density.
    pub compact_electrons: f64,
}

/// Hartree energy per cell and its strain derivative at fixed density
/// images `D(T)`, with the strain applied to atoms and lattice together.
#[derive(Clone, Debug)]
pub struct HartreeStrain {
    /// `E_J` (Hartree per cell).
    pub energy: f64,
    /// `dE_J/d(epsilon_ab)` (Hartree per cell).
    pub strain_derivative: [[f64; 3]; 3],
}

fn contract_quartet(q: &crate::integrals::eri::Quartet, bra: &Matrix, ket: &Matrix) -> f64 {
    let mut acc = 0.0;
    for i in 0..q.na {
        for j in 0..q.nb {
            let d = bra[(i, j)];
            if d == 0.0 {
                continue;
            }
            for k in 0..q.nc {
                for l in 0..q.nd {
                    acc += d * ket[(k, l)] * q.get(i, j, k, l);
                }
            }
        }
    }
    acc
}

/// Analytic energy, gradient and centre virial of the screened Hartree
/// energy for a fixed AO density.
pub fn screened_hartree_derivatives(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
    natoms: usize,
) -> crate::Result<HartreeDerivatives> {
    let prep = Prepared::new(mb, lattice, density, screening)?;
    Ok(prepared_derivatives(&prep, natoms))
}

fn prepared_derivatives(prep: &Prepared, natoms: usize) -> HartreeDerivatives {
    let kernel = CoulombKernel::Erfc { omega: prep.alpha };
    let zero = [Vec3::zero()];
    let reciprocal = prep.reciprocal_contractions();
    let background = ReciprocalContraction::new(&zero, &[&[1.0]], &[&[0.0]]);
    // Only bras that carry density contribute to E = 1/2 sum D J.
    let partials: Vec<(f64, Vec<Vec3>, [[f64; 3]; 3])> = prep
        .kets
        .par_iter()
        .map(|bra_ket| {
            let bra = &prep.pairs[bra_ket.pair];
            let mut energy = 0.0;
            let mut gradient = vec![Vec3::zero(); natoms];
            let mut virial = [[0.0; 3]; 3];
            let sa = &prep.parts[bra.x].shell;
            let sb = moved(&prep.parts[bra.y].shell, bra.translation);
            let db = &bra_ket.block;
            let mut add = |atom: usize, centre: Vec3, axis: usize, value: f64| {
                match axis {
                    0 => gradient[atom].x += value,
                    1 => gradient[atom].y += value,
                    _ => gradient[atom].z += value,
                }
                for b in 0..3 {
                    virial[axis][b] += value * centre.get(b);
                }
            };
            // Reciprocal part: dE = sum_b D_b d(FT_b) c_class(b).
            let contraction = &reciprocal[usize::from(bra.diffuse)];
            let value =
                crate::basis::aos::sph_block(sa.l, sb.l, &contraction.blocks(sa, &sb).remove(0).0);
            energy += 0.5 * dot(&value, db);
            let derivative =
                crate::periodic::strain::pair_centre_derivatives_with(sa, &sb, contraction);
            for (side, shell) in [(0, sa), (1, &sb)] {
                for axis in 0..3 {
                    let v = dot(
                        &crate::basis::aos::sph_block(sa.l, sb.l, &derivative[side][axis]),
                        db,
                    );
                    add(shell.atom, shell.center, axis, v);
                }
            }
            if !bra.diffuse {
                // Background: E_bg = shift N_c / 2 and dE_bg = shift dN_c.
                let overlap = crate::basis::aos::sph_block(
                    sa.l,
                    sb.l,
                    &pair_blocks_per_g(sa, &sb, &zero)[0].0,
                );
                energy += 0.5 * prep.shift * dot(&overlap, db);
                let d_overlap =
                    crate::periodic::strain::pair_centre_derivatives_with(sa, &sb, &background);
                for (side, shell) in [(0, sa), (1, &sb)] {
                    for axis in 0..3 {
                        let v = dot(
                            &crate::basis::aos::sph_block(sa.l, sb.l, &d_overlap[side][axis]),
                            db,
                        );
                        add(shell.atom, shell.center, axis, prep.shift * v);
                    }
                }
                // Real space: E = 1/2 sum D_b (b|k) D_k, derivative on all
                // four centres.
                for (ket, cell) in prep.real_space_partners(bra) {
                    let kp = &prep.pairs[ket.pair];
                    let sc = moved(&prep.parts[kp.x].shell, cell);
                    let sd = moved(&prep.parts[kp.y].shell, cell + kp.translation);
                    let q = eri_quartet(sa, &sb, &sc, &sd, kernel);
                    energy += 0.5 * contract_quartet(&q, db, &ket.block);
                    let shells = [sa, &sb, &sc, &sd];
                    let derivatives =
                        crate::integrals::deriv::eri_quartet_center_derivatives(shells, kernel);
                    for (centre, per_axis) in derivatives.iter().enumerate() {
                        for (axis, dq) in per_axis.iter().enumerate() {
                            add(
                                shells[centre].atom,
                                shells[centre].center,
                                axis,
                                0.5 * contract_quartet(dq, db, &ket.block),
                            );
                        }
                    }
                }
            }
            (energy, gradient, virial)
        })
        .collect();
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); natoms];
    let mut centre_virial = [[0.0; 3]; 3];
    for (e, g, v) in partials {
        energy += e;
        for (a, b) in gradient.iter_mut().zip(g) {
            *a += b;
        }
        for i in 0..3 {
            for j in 0..3 {
                centre_virial[i][j] += v[i][j];
            }
        }
    }
    HartreeDerivatives {
        energy,
        gradient,
        centre_virial,
        compact_electrons: prep.compact_electrons,
    }
}

/// Analytic strain derivative of the screened Hartree energy at fixed
/// density images: the centre virial of every basis image, the wave-vector
/// dependence of each pair transform, the `1/(V G^2)` kernels and the
/// `1/V` background of the compact half.
pub fn screened_hartree_strain(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
) -> crate::Result<HartreeStrain> {
    use crate::periodic::strain::{
        add_tensor, reciprocal_wavevector_coefficients, reciprocal_wavevector_strain_with,
    };
    let prep = Prepared::new(mb, lattice, density, screening)?;
    let natoms = mb.shells.iter().map(|s| s.atom + 1).max().unwrap_or(0);
    let derivatives = prepared_derivatives(&prep, natoms);
    let mut total = derivatives.centre_virial;
    // Wave vectors of every density-carrying bra transform.
    let coefficients = [&prep.bra_compact, &prep.bra_diffuse]
        .map(|[re, im]| reciprocal_wavevector_coefficients(&prep.with_zero, (re, im)));
    let reciprocal = coefficients.each_ref().map(|[re, im]| {
        ReciprocalContraction::new(
            &prep.with_zero,
            &re.each_ref().map(Vec::as_slice),
            &im.each_ref().map(Vec::as_slice),
        )
    });
    let wavevector = prep
        .kets
        .par_iter()
        .map(|bra_ket| {
            let bra = &prep.pairs[bra_ket.pair];
            let sa = &prep.parts[bra.x].shell;
            let sb = moved(&prep.parts[bra.y].shell, bra.translation);
            let d_cart = crate::basis::aos::cart_to_sph(sa.l)
                .matmul_seq(&bra_ket.block)
                .matmul_seq(&crate::basis::aos::cart_to_sph(sb.l).transpose());
            reciprocal_wavevector_strain_with(
                sa,
                &sb,
                &reciprocal[usize::from(bra.diffuse)],
                &d_cart,
            )
        })
        .reduce(
            || [[0.0; 3]; 3],
            |mut a, b| {
                add_tensor(&mut a, &b);
                a
            },
        );
    add_tensor(&mut total, &wavevector);
    // Kernels: E = 1/2 sum_G [damped |rho_c|^2 + bare (|rho_c + rho_d|^2 - |rho_c|^2)],
    // dK/d(eps_ab) = K [2 G_a G_b (1/G^2 [+ 1/(4 alpha^2)]) - delta_ab].
    let [cr, ci, dr, di] = &prep.transforms;
    let alpha2 = prep.alpha * prep.alpha;
    for (g, gv) in prep.with_zero.iter().enumerate().skip(1) {
        let g2 = gv.norm2();
        let bare = HALF_SPACE_WEIGHT * 4.0 * std::f64::consts::PI / (prep.volume * g2);
        let damped = bare * (-g2 / (4.0 * alpha2)).exp();
        let compact = cr[g] * cr[g] + ci[g] * ci[g];
        let diffuse = diffuse_reciprocal_product(
            [cr[g], ci[g]],
            [dr[g], di[g]],
            [cr[g], ci[g]],
            [dr[g], di[g]],
        );
        let damped_energy = 0.5 * damped * compact;
        let bare_energy = 0.5 * bare * diffuse;
        for a in 0..3 {
            for b in 0..3 {
                let gg = 2.0 * gv.get(a) * gv.get(b);
                total[a][b] +=
                    damped_energy * gg * (1.0 / g2 + 1.0 / (4.0 * alpha2)) + bare_energy * gg / g2;
            }
            total[a][a] -= damped_energy + bare_energy;
        }
    }
    // Background E_bg = 1/2 shift N_c with shift proportional to 1/V; its
    // N_c dependence is part of the centre virial.
    let background = 0.5 * prep.shift * prep.compact_electrons;
    for (a, row) in total.iter_mut().enumerate() {
        row[a] -= background;
    }
    Ok(HartreeStrain {
        energy: derivatives.energy,
        strain_derivative: total,
    })
}

/// `sum_ij D_ij d rho_ij(G) / d A_x` of a shell pair with the derivative on
/// the first shell `sa` (rows of the Cartesian density), accumulated into
/// `re`/`im`: the raised shell carries `D` moved up one component along `x`,
/// the lowered shell carries `-l_x` times `D` moved down.
fn derivative_transform(
    sa: &MolShell,
    sb: &MolShell,
    gs: &[Vec3],
    d_cart: &Matrix,
    axis: usize,
    re: &mut [f64],
    im: &mut [f64],
) {
    use crate::basis::cart_components;
    let comps = cart_components(sa.l);
    let up_comps = cart_components(sa.l + 1);
    let ncb = d_cart.cols;
    let mut up = Matrix::zeros(up_comps.len(), ncb);
    let mut down = (sa.l > 0).then(|| Matrix::zeros(cart_components(sa.l - 1).len(), ncb));
    for (i, &c) in comps.iter().enumerate() {
        let mut raised_c = [c.0, c.1, c.2];
        raised_c[axis] += 1;
        let iu = up_comps
            .iter()
            .position(|&u| [u.0, u.1, u.2] == raised_c)
            .expect("raised component");
        for j in 0..ncb {
            up[(iu, j)] += d_cart[(i, j)];
        }
        let lx = [c.0, c.1, c.2][axis];
        if lx > 0 {
            let mut lowered_c = [c.0, c.1, c.2];
            lowered_c[axis] -= 1;
            let low_comps = cart_components(sa.l - 1);
            let id = low_comps
                .iter()
                .position(|&u| [u.0, u.1, u.2] == lowered_c)
                .expect("lowered component");
            let down = down.as_mut().expect("lowered block");
            for j in 0..ncb {
                down[(id, j)] -= lx as f64 * d_cart[(i, j)];
            }
        }
    }
    use crate::integrals::deriv::{lowered, raised};
    pair_density_transform(&raised(sa), sb, gs, &up, re, im);
    if let (Some(lo), Some(down)) = (lowered(sa), down.as_ref()) {
        pair_density_transform(&lo, sb, gs, down, re, im);
    }
}

/// Analytic Gamma-point skeleton Hessian of the screened Hartree energy at
/// fixed density images: the `erfc` quartet second derivatives of the
/// compact real-space part, the reciprocal part
/// `sum_G K Re[rho_x^* rho_y + rho^* rho_xy]` by class, and the compact
/// background `-pi N_c^2 / (2 alpha^2 V)`.
pub fn screened_hartree_hessian(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
    natoms: usize,
) -> crate::Result<Matrix> {
    use crate::integrals::deriv2::{quartet_mixed_second, PAIRS};
    use crate::periodic::hessian::pair_second_blocks;
    let prep = Prepared::new(mb, lattice, density, screening)?;
    let n = 3 * natoms;
    let kernel = CoulombKernel::Erfc { omega: prep.alpha };
    let ng = prep.with_zero.len();
    let cart = |block: &Matrix, a: &MolShell, b: &MolShell| {
        crate::basis::aos::cart_to_sph(a.l)
            .matmul_seq(block)
            .matmul_seq(&crate::basis::aos::cart_to_sph(b.l).transpose())
    };
    let add_invariant = |h: &mut Matrix, a: usize, b: usize, v: &[[f64; 3]; 3]| {
        for x in 0..3 {
            for y in 0..3 {
                h[(3 * a + x, 3 * b + y)] += v[x][y];
                h[(3 * b + y, 3 * a + x)] += v[x][y];
                h[(3 * a + x, 3 * a + y)] -= v[x][y];
                h[(3 * b + y, 3 * b + x)] -= v[x][y];
            }
        }
    };
    // Per bra: pair second derivatives at fixed coefficients, real-space
    // quartets, and the per-atom derivative transforms of both classes.
    // transforms[class * 2 + part][3 * atom + axis][g], class 0 compact.
    let (mut h, transforms) = prep
        .kets
        .par_iter()
        .fold(
            || (Matrix::zeros(n, n), vec![vec![vec![0.0; ng]; n]; 4]),
            |(mut h, mut transforms), bra_ket| {
                let bra = &prep.pairs[bra_ket.pair];
                let sa = &prep.parts[bra.x].shell;
                let sb = moved(&prep.parts[bra.y].shell, bra.translation);
                let db = &bra_ket.block;
                let d_cart = cart(db, sa, &sb);
                // 1/2 sum_b D_b Re sum_G c(G) rho_b(G): second derivative of
                // rho_b at fixed c carries the factor 1 (both factors vary).
                let c = prep.coefficients(bra);
                let block = |x: &MolShell, y: &MolShell| {
                    contracted_pair_blocks(x, y, &prep.with_zero, &[c.0], &[c.1])
                        .remove(0)
                        .0
                };
                let [aa, ab, ba, bb] = pair_second_blocks(sa, &sb, &d_cart, &block);
                for (a, b, v) in [
                    (sa.atom, sa.atom, aa),
                    (sa.atom, sb.atom, ab),
                    (sb.atom, sa.atom, ba),
                    (sb.atom, sb.atom, bb),
                ] {
                    for x in 0..3 {
                        for y in 0..3 {
                            h[(3 * a + x, 3 * b + y)] += v[x][y];
                        }
                    }
                }
                // Derivative transforms of this pair, both centres.
                let class = if bra.diffuse { 1 } else { 0 };
                for axis in 0..3 {
                    let (left, right) = transforms.split_at_mut(2 * class + 1);
                    let (re, im) = (&mut left[2 * class], &mut right[0]);
                    derivative_transform(
                        sa,
                        &sb,
                        &prep.with_zero,
                        &d_cart,
                        axis,
                        &mut re[3 * sa.atom + axis],
                        &mut im[3 * sa.atom + axis],
                    );
                    derivative_transform(
                        &sb,
                        sa,
                        &prep.with_zero,
                        &d_cart.transpose(),
                        axis,
                        &mut re[3 * sb.atom + axis],
                        &mut im[3 * sb.atom + axis],
                    );
                }
                if !bra.diffuse {
                    // Real space: E = 1/2 sum D_b (b|k) D_k over ordered
                    // (bra, ket, cell) quartets.
                    for (ket, cell) in prep.real_space_partners(bra) {
                        let kp = &prep.pairs[ket.pair];
                        let sc = moved(&prep.parts[kp.x].shell, cell);
                        let sd = moved(&prep.parts[kp.y].shell, cell + kp.translation);
                        let shells = [sa, &sb, &sc, &sd];
                        if shells.iter().all(|s| s.atom == sa.atom) {
                            continue;
                        }
                        let d2t = quartet_mixed_second(shells, kernel);
                        let dims = [
                            crate::basis::n_sph(sa.l),
                            crate::basis::n_sph(sb.l),
                            crate::basis::n_sph(sc.l),
                            crate::basis::n_sph(sd.l),
                        ];
                        let mut weights = Vec::with_capacity(dims.iter().product());
                        for i0 in 0..dims[0] {
                            for i1 in 0..dims[1] {
                                for i2 in 0..dims[2] {
                                    for i3 in 0..dims[3] {
                                        weights.push(0.5 * db[(i0, i1)] * ket.block[(i2, i3)]);
                                    }
                                }
                            }
                        }
                        for (slot, &(i, j)) in PAIRS.iter().enumerate() {
                            let mut v = [[0.0; 3]; 3];
                            for x in 0..3 {
                                for y in 0..3 {
                                    v[x][y] = d2t[slot][3 * x + y]
                                        .iter()
                                        .zip(&weights)
                                        .map(|(a, b)| a * b)
                                        .sum();
                                }
                            }
                            add_invariant(&mut h, shells[i].atom, shells[j].atom, &v);
                        }
                    }
                }
                (h, transforms)
            },
        )
        .reduce(
            || (Matrix::zeros(n, n), vec![vec![vec![0.0; ng]; n]; 4]),
            |(mut ha, mut ta), (hb, tb)| {
                for (x, y) in ha.as_mut_slice().iter_mut().zip(hb.as_slice()) {
                    *x += *y;
                }
                for (a, b) in ta.iter_mut().zip(tb) {
                    for (u, v) in a.iter_mut().zip(b) {
                        for (s, t) in u.iter_mut().zip(v) {
                            *s += t;
                        }
                    }
                }
                (ha, ta)
            },
        );
    // Products of derivative transforms with the kernels.
    let alpha2 = prep.alpha * prep.alpha;
    let [cr, ci, dr, di] = [
        &transforms[0],
        &transforms[1],
        &transforms[2],
        &transforms[3],
    ];
    for (g, gv) in prep.with_zero.iter().enumerate().skip(1) {
        let g2 = gv.norm2();
        let bare = HALF_SPACE_WEIGHT * 4.0 * std::f64::consts::PI / (prep.volume * g2);
        let damped = bare * (-g2 / (4.0 * alpha2)).exp();
        for x in 0..n {
            for y in 0..n {
                let compact = cr[x][g] * cr[y][g] + ci[x][g] * ci[y][g];
                let diffuse = diffuse_reciprocal_product(
                    [cr[x][g], ci[x][g]],
                    [dr[x][g], di[x][g]],
                    [cr[y][g], ci[y][g]],
                    [dr[y][g], di[y][g]],
                );
                h[(x, y)] += damped * compact + bare * diffuse;
            }
        }
    }
    // Background: E_bg = -pi N_c^2 / (2 alpha^2 V); N_c,x is the G = 0
    // derivative transform of the compact class, and shift * N_c,xy comes
    // with the overlap kind of the compact pairs.
    let prefactor = std::f64::consts::PI / (alpha2 * prep.volume);
    for x in 0..n {
        for y in 0..n {
            h[(x, y)] -= prefactor * cr[x][0] * cr[y][0];
        }
    }
    for bra_ket in &prep.kets {
        let bra = &prep.pairs[bra_ket.pair];
        if bra.diffuse {
            continue;
        }
        let sa = &prep.parts[bra.x].shell;
        let sb = moved(&prep.parts[bra.y].shell, bra.translation);
        if sa.atom == sb.atom {
            continue;
        }
        let d_cart = cart(&bra_ket.block, sa, &sb);
        let block = |x: &MolShell, y: &MolShell| {
            crate::integrals::deriv::pair_block(x, y, crate::integrals::deriv::overlap_kind)
        };
        let blocks = pair_second_blocks(sa, &sb, &d_cart, &block);
        let mut v = blocks[1];
        for row in v.iter_mut() {
            for value in row.iter_mut() {
                *value *= prep.shift;
            }
        }
        add_invariant(&mut h, sa.atom, sb.atom, &v);
    }
    Ok(h)
}

/// Analytic Gamma-point derivatives `dJ(T)/dR_(A,x)` of the screened Hartree
/// matrix at fixed density images, as image blocks indexed
/// `[3 * atom + axis]`: the bra pair centres at fixed reciprocal
/// coefficients, the coefficients through the derivative transforms of the
/// ket density, the compact background, and all four centres of the
/// real-space `erfc` quartets.
pub fn screened_hartree_fock_derivatives(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
    natoms: usize,
) -> crate::Result<Vec<Vec<ImageBlock>>> {
    use crate::integrals::deriv::eri_quartet_center_derivatives;
    use crate::periodic::strain::pair_centre_derivatives;
    let prep = Prepared::new(mb, lattice, density, screening)?;
    let n = mb.nao_sph;
    let dim = 3 * natoms;
    let kernel = CoulombKernel::Erfc { omega: prep.alpha };
    let ng = prep.with_zero.len();
    let cart = |block: &Matrix, a: &MolShell, b: &MolShell| {
        crate::basis::aos::cart_to_sph(a.l)
            .matmul_seq(block)
            .matmul_seq(&crate::basis::aos::cart_to_sph(b.l).transpose())
    };
    // Derivative transforms of the ket densities, by class.
    let transforms = prep
        .kets
        .par_iter()
        .fold(
            || vec![vec![vec![0.0; ng]; dim]; 4],
            |mut transforms, ket| {
                let pair = &prep.pairs[ket.pair];
                let sa = &prep.parts[pair.x].shell;
                let sb = moved(&prep.parts[pair.y].shell, pair.translation);
                let d_cart = cart(&ket.block, sa, &sb);
                let class = if pair.diffuse { 1 } else { 0 };
                for axis in 0..3 {
                    let (left, right) = transforms.split_at_mut(2 * class + 1);
                    let (re, im) = (&mut left[2 * class], &mut right[0]);
                    derivative_transform(
                        sa,
                        &sb,
                        &prep.with_zero,
                        &d_cart,
                        axis,
                        &mut re[3 * sa.atom + axis],
                        &mut im[3 * sa.atom + axis],
                    );
                    derivative_transform(
                        &sb,
                        sa,
                        &prep.with_zero,
                        &d_cart.transpose(),
                        axis,
                        &mut re[3 * sb.atom + axis],
                        &mut im[3 * sb.atom + axis],
                    );
                }
                transforms
            },
        )
        .reduce(
            || vec![vec![vec![0.0; ng]; dim]; 4],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    for (u, v) in x.iter_mut().zip(y) {
                        for (s, t) in u.iter_mut().zip(v) {
                            *s += t;
                        }
                    }
                }
                a
            },
        );
    // Coefficient derivatives c_x(G) of both bra classes.
    let (cr, ci, dr, di) = (
        &transforms[0],
        &transforms[1],
        &transforms[2],
        &transforms[3],
    );
    let alpha2 = prep.alpha * prep.alpha;
    let mut compact_re = vec![vec![0.0; ng]; dim];
    let mut compact_im = vec![vec![0.0; ng]; dim];
    let mut diffuse_re = vec![vec![0.0; ng]; dim];
    let mut diffuse_im = vec![vec![0.0; ng]; dim];
    for x in 0..dim {
        for g in 1..ng {
            let g2 = prep.with_zero[g].norm2();
            let bare = HALF_SPACE_WEIGHT * 4.0 * std::f64::consts::PI / (prep.volume * g2);
            let damped = bare * (-g2 / (4.0 * alpha2)).exp();
            compact_re[x][g] = damped * cr[x][g] + bare * dr[x][g];
            compact_im[x][g] = -(damped * ci[x][g] + bare * di[x][g]);
            diffuse_re[x][g] = bare * (cr[x][g] + dr[x][g]);
            diffuse_im[x][g] = -bare * (ci[x][g] + di[x][g]);
        }
    }
    fn refs(sets: &[Vec<f64>]) -> Vec<&[f64]> {
        sets.iter().map(|v| v.as_slice()).collect()
    }
    let (compact_re_refs, compact_im_refs) = (refs(&compact_re), refs(&compact_im));
    let (diffuse_re_refs, diffuse_im_refs) = (refs(&diffuse_re), refs(&diffuse_im));
    // d shift / dx = -pi N_c,x / (alpha^2 V).
    let dshift: Vec<f64> = (0..dim)
        .map(|x| -std::f64::consts::PI * cr[x][0] / (alpha2 * prep.volume))
        .collect();
    let mut outputs: Vec<Vec3> = Vec::new();
    for p in &prep.pairs {
        if !outputs.iter().any(|t| (*t - p.translation).norm() < 1e-8) {
            outputs.push(p.translation);
        }
    }
    let zero = [Vec3::zero()];
    let blocks: Vec<Vec<Matrix>> = outputs
        .par_iter()
        .map(|&t| {
            let mut m: Vec<Matrix> = (0..dim).map(|_| Matrix::zeros(n, n)).collect();
            for bra in prep
                .pairs
                .iter()
                .filter(|p| (p.translation - t).norm() < 1e-8)
            {
                let sa = &prep.parts[bra.x].shell;
                let sb = moved(&prep.parts[bra.y].shell, t);
                let (oa, ob) = (sa.sph_off, sb.sph_off);
                let mut local: Vec<Matrix> = (0..dim)
                    .map(|_| Matrix::zeros(crate::basis::n_sph(sa.l), crate::basis::n_sph(sb.l)))
                    .collect();
                let mut add_cart = |x: usize, block: &Matrix, scale: f64| {
                    let sph = crate::basis::aos::sph_block(sa.l, sb.l, block);
                    for (v, s) in local[x].as_mut_slice().iter_mut().zip(sph.as_slice()) {
                        *v += scale * s;
                    }
                };
                // Bra centres at fixed coefficients.
                let c = prep.coefficients(bra);
                let derivative = pair_centre_derivatives(sa, &sb, &prep.with_zero, c);
                for axis in 0..3 {
                    add_cart(3 * sa.atom + axis, &derivative[0][axis], 1.0);
                    add_cart(3 * sb.atom + axis, &derivative[1][axis], 1.0);
                }
                // Coefficients through the ket densities.
                let sets = if bra.diffuse {
                    contracted_pair_blocks(
                        sa,
                        &sb,
                        &prep.with_zero,
                        &diffuse_re_refs,
                        &diffuse_im_refs,
                    )
                } else {
                    contracted_pair_blocks(
                        sa,
                        &sb,
                        &prep.with_zero,
                        &compact_re_refs,
                        &compact_im_refs,
                    )
                };
                for (x, set) in sets.iter().enumerate() {
                    add_cart(x, &set.0, 1.0);
                }
                if !bra.diffuse {
                    // Background: shift S_ab.
                    let unit = (&[1.0][..], &[0.0][..]);
                    let d_overlap = pair_centre_derivatives(sa, &sb, &zero, unit);
                    let overlap = pair_blocks_per_g(sa, &sb, &zero)[0].0.clone();
                    for axis in 0..3 {
                        add_cart(3 * sa.atom + axis, &d_overlap[0][axis], prep.shift);
                        add_cart(3 * sb.atom + axis, &d_overlap[1][axis], prep.shift);
                    }
                    for (x, ds) in dshift.iter().enumerate() {
                        add_cart(x, &overlap, *ds);
                    }
                    // Real space: all four centres of every quartet.
                    for (ket, cell) in prep.real_space_partners(bra) {
                        let kp = &prep.pairs[ket.pair];
                        let sc = moved(&prep.parts[kp.x].shell, cell);
                        let sd = moved(&prep.parts[kp.y].shell, cell + kp.translation);
                        let shells = [sa, &sb, &sc, &sd];
                        let derivatives = eri_quartet_center_derivatives(shells, kernel);
                        for (centre, per_axis) in derivatives.iter().enumerate() {
                            let atom = shells[centre].atom;
                            for (axis, q) in per_axis.iter().enumerate() {
                                let target = &mut local[3 * atom + axis];
                                for i in 0..q.na {
                                    for j in 0..q.nb {
                                        let mut acc = 0.0;
                                        for k in 0..q.nc {
                                            for l in 0..q.nd {
                                                acc += ket.block[(k, l)] * q.get(i, j, k, l);
                                            }
                                        }
                                        target[(i, j)] += acc;
                                    }
                                }
                            }
                        }
                    }
                }
                for x in 0..dim {
                    for i in 0..local[x].rows {
                        for j in 0..local[x].cols {
                            m[x][(oa + i, ob + j)] += local[x][(i, j)];
                        }
                    }
                }
            }
            m
        })
        .collect();
    let mut out: Vec<Vec<ImageBlock>> = (0..dim).map(|_| Vec::new()).collect();
    for (t, per_x) in outputs.iter().zip(blocks) {
        for (x, matrix) in per_x.into_iter().enumerate() {
            out[x].push(ImageBlock {
                translation: *t,
                matrix,
            });
        }
    }
    Ok(out)
}

/// First-order strain matrices `dJ(T)/d(epsilon_ab)` of the screened Hartree
/// matrix at fixed density images, indexed `[3a + b]`: the bra pair centres
/// and wave vectors at fixed reciprocal coefficients, the coefficients
/// through the kernels, the volume and the strained ket densities (exact
/// first-order jets), the compact background, and the four-centre virial of
/// the real-space `erfc` quartets. Contracted with the density this is twice
/// the Hartree strain derivative.
pub(crate) fn screened_hartree_strain_fock(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
) -> crate::Result<Vec<crate::periodic::strain_fock::StrainBlock>> {
    use crate::dual::{DualN, Scalar};
    use crate::integrals::deriv::eri_quartet_center_derivatives;
    use crate::periodic::strain::{
        pair_centre_derivatives, reciprocal_centre_virial_blocks, reciprocal_wavevector_blocks,
    };
    use crate::periodic::strain_fock::{merge_strain_blocks, StrainBlock};
    use crate::periodic::strain_jets::{
        strain_det, strained_pair_density, strained_transforms, strained_waves,
    };
    type Jet = DualN<f64, 9>;
    let prep = Prepared::new(mb, lattice, density, screening)?;
    let n = mb.nao_sph;
    let kernel = CoulombKernel::Erfc { omega: prep.alpha };
    let eps: [[Jet; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|b| Jet::variable(0.0, 3 * a + b)));
    let waves = strained_waves(&prep.with_zero, &eps);
    let ng = waves.len();
    let kets = prepared_ket_shells(&prep);
    let mut accumulated = strained_transforms(&kets, &waves, 4, &|ket, block, acc| {
        let (left, right) = acc.split_at_mut(ket.slot + 1);
        strained_pair_density(
            &ket.a,
            &ket.b,
            None,
            &eps,
            block,
            &ket.d_cart,
            &mut left[ket.slot],
            &mut right[0],
        );
    });
    let (di, dr, ci, cr) = (
        accumulated.pop().unwrap_or_default(),
        accumulated.pop().unwrap_or_default(),
        accumulated.pop().unwrap_or_default(),
        accumulated.pop().unwrap_or_default(),
    );
    // Strain derivatives of the coefficients of both bra classes.
    let volume = Jet::from_f64(prep.volume) * strain_det(&eps);
    let alpha2 = prep.alpha * prep.alpha;
    let mut compact = [vec![vec![0.0; ng]; 9], vec![vec![0.0; ng]; 9]];
    let mut diffuse = [vec![vec![0.0; ng]; 9], vec![vec![0.0; ng]; 9]];
    for g in 1..ng {
        let x = waves[g].norm2;
        let bare = Jet::from_f64(HALF_SPACE_WEIGHT * 4.0 * std::f64::consts::PI) / (volume * x);
        let damped = bare * (x * Jet::from_f64(-0.25 / alpha2)).exp();
        let values = [
            damped * cr[g] + bare * dr[g],
            -(damped * ci[g] + bare * di[g]),
            bare * (cr[g] + dr[g]),
            -(bare * (ci[g] + di[g])),
        ];
        for k in 0..9 {
            compact[0][k][g] = values[0].im[k];
            compact[1][k][g] = values[1].im[k];
            diffuse[0][k][g] = values[2].im[k];
            diffuse[1][k][g] = values[3].im[k];
        }
    }
    let shift = Jet::from_f64(-std::f64::consts::PI / alpha2) * cr[0] / volume;
    fn refs(sets: &[Vec<f64>]) -> Vec<&[f64]> {
        sets.iter().map(|v| v.as_slice()).collect()
    }
    let compact_refs = [refs(&compact[0]), refs(&compact[1])];
    let diffuse_refs = [refs(&diffuse[0]), refs(&diffuse[1])];
    let mut outputs: Vec<Vec3> = Vec::new();
    for p in &prep.pairs {
        if !outputs.iter().any(|t| (*t - p.translation).norm() < 1e-8) {
            outputs.push(p.translation);
        }
    }
    let zero = [Vec3::zero()];
    let blocks: Vec<StrainBlock> = outputs
        .par_iter()
        .map(|&t| {
            let mut block = StrainBlock::zeros(t, n);
            for bra in prep
                .pairs
                .iter()
                .filter(|p| (p.translation - t).norm() < 1e-8)
            {
                let sa = &prep.parts[bra.x].shell;
                let sb = moved(&prep.parts[bra.y].shell, t);
                // Bra centres and wave vectors at fixed coefficients.
                let c = prep.coefficients(bra);
                let virial = reciprocal_centre_virial_blocks(sa, &sb, &prep.with_zero, c);
                let wave = reciprocal_wavevector_blocks(sa, &sb, &prep.with_zero, c);
                for a in 0..3 {
                    for b in 0..3 {
                        block.add_cart(3 * a + b, sa, &sb, &virial[a][b], 1.0);
                        block.add_cart(3 * a + b, sa, &sb, &wave[a][b], 1.0);
                    }
                }
                // Coefficients.
                let sets = if bra.diffuse {
                    contracted_pair_blocks(
                        sa,
                        &sb,
                        &prep.with_zero,
                        &diffuse_refs[0],
                        &diffuse_refs[1],
                    )
                } else {
                    contracted_pair_blocks(
                        sa,
                        &sb,
                        &prep.with_zero,
                        &compact_refs[0],
                        &compact_refs[1],
                    )
                };
                for (k, set) in sets.iter().enumerate() {
                    block.add_cart(k, sa, &sb, &set.0, 1.0);
                }
                if bra.diffuse {
                    continue;
                }
                // Background shift * S_ab.
                let unit = (&[1.0][..], &[0.0][..]);
                let d_overlap = pair_centre_derivatives(sa, &sb, &zero, unit);
                let overlap = pair_blocks_per_g(sa, &sb, &zero)[0].0.clone();
                for a in 0..3 {
                    for b in 0..3 {
                        block.add_cart(
                            3 * a + b,
                            sa,
                            &sb,
                            &d_overlap[0][a],
                            shift.re * sa.center.get(b),
                        );
                        block.add_cart(
                            3 * a + b,
                            sa,
                            &sb,
                            &d_overlap[1][a],
                            shift.re * sb.center.get(b),
                        );
                    }
                }
                for k in 0..9 {
                    block.add_cart(k, sa, &sb, &overlap, shift.im[k]);
                }
                // Real space: the four-centre virial of every quartet.
                let mut local: Vec<Matrix> = (0..9)
                    .map(|_| Matrix::zeros(crate::basis::n_sph(sa.l), crate::basis::n_sph(sb.l)))
                    .collect();
                for (ket, cell) in prep.real_space_partners(bra) {
                    let kp = &prep.pairs[ket.pair];
                    let sc = moved(&prep.parts[kp.x].shell, cell);
                    let sd = moved(&prep.parts[kp.y].shell, cell + kp.translation);
                    let shells = [sa, &sb, &sc, &sd];
                    let derivatives = eri_quartet_center_derivatives(shells, kernel);
                    for (centre, per_axis) in derivatives.iter().enumerate() {
                        let r = shells[centre].center - sa.center;
                        if centre == 0 {
                            continue;
                        }
                        for (axis, q) in per_axis.iter().enumerate() {
                            for i in 0..q.na {
                                for j in 0..q.nb {
                                    let mut acc = 0.0;
                                    for k in 0..q.nc {
                                        for l in 0..q.nd {
                                            acc += ket.block[(k, l)] * q.get(i, j, k, l);
                                        }
                                    }
                                    for b in 0..3 {
                                        local[3 * axis + b][(i, j)] += acc * r.get(b);
                                    }
                                }
                            }
                        }
                    }
                }
                for (k, m) in local.iter().enumerate() {
                    block.add_sph(k, sa.sph_off, sb.sph_off, m, 1.0);
                }
            }
            block
        })
        .collect();
    Ok(merge_strain_blocks(blocks))
}

/// A density ket with its shells (the ket shell moved to its translation),
/// Cartesian density block and transform class (0 compact, 2 diffuse).
struct KetShells {
    a: MolShell,
    b: MolShell,
    d_cart: Matrix,
    slot: usize,
}

fn prepared_ket_shells(prep: &Prepared) -> Vec<KetShells> {
    prep.kets
        .iter()
        .map(|ket| {
            let pair = &prep.pairs[ket.pair];
            let a = prep.parts[pair.x].shell.clone();
            let b = moved(&prep.parts[pair.y].shell, pair.translation);
            let d_cart = crate::basis::aos::cart_to_sph(a.l)
                .matmul_seq(&ket.block)
                .matmul_seq(&crate::basis::aos::cart_to_sph(b.l).transpose());
            KetShells {
                a,
                b,
                d_cart,
                slot: if pair.diffuse { 2 } else { 0 },
            }
        })
        .collect()
}

/// Reciprocal Hartree energy with its compact background,
/// `1/2 sum_G [damped |rho_c|^2 + bare (|rho_c + rho_d|^2 - |rho_c|^2)] -
/// pi N_c^2 / (2 alpha^2 V)`, as a jet in the strain and the seeded
/// displacements.
fn hartree_reciprocal_jet<S: crate::dual::Scalar + Send + Sync>(
    prep: &Prepared,
    eps: &[[S; 3]; 3],
    displacement: Option<&(dyn Fn(usize) -> [S; 3] + Sync)>,
) -> S {
    use crate::periodic::strain_jets::{
        strain_det, strained_pair_density, strained_transforms, strained_waves,
    };
    let waves = strained_waves(&prep.with_zero, eps);
    let ng = waves.len();
    let zero = S::from_f64(0.0);
    let kets = prepared_ket_shells(prep);
    let mut accumulated = strained_transforms(&kets, &waves, 4, &|ket, block, acc| {
        let centres = displacement.map(|displacement| {
            [&ket.a, &ket.b].map(|shell| {
                let u = displacement(shell.atom);
                std::array::from_fn(|k| S::from_f64(shell.center.get(k)) + u[k])
            })
        });
        let (left, right) = acc.split_at_mut(ket.slot + 1);
        strained_pair_density(
            &ket.a,
            &ket.b,
            centres,
            eps,
            block,
            &ket.d_cart,
            &mut left[ket.slot],
            &mut right[0],
        );
    });
    let (di, dr, ci, cr) = (
        accumulated.pop().unwrap_or_default(),
        accumulated.pop().unwrap_or_default(),
        accumulated.pop().unwrap_or_default(),
        accumulated.pop().unwrap_or_default(),
    );
    let volume = S::from_f64(prep.volume) * strain_det(eps);
    let alpha2 = prep.alpha * prep.alpha;
    let half = S::from_f64(0.5);
    let mut energy = zero;
    for g in 1..ng {
        let x = waves[g].norm2;
        let bare = S::from_f64(HALF_SPACE_WEIGHT * 4.0 * std::f64::consts::PI) / (volume * x);
        let damped = bare * (x * S::from_f64(-0.25 / alpha2)).exp();
        let compact = cr[g] * cr[g] + ci[g] * ci[g];
        let diffuse = diffuse_reciprocal_product(
            [cr[g], ci[g]],
            [dr[g], di[g]],
            [cr[g], ci[g]],
            [dr[g], di[g]],
        );
        energy = energy + half * (damped * compact + bare * diffuse);
    }
    // N_c is the compact transform at G = 0.
    energy - S::from_f64(std::f64::consts::PI / (2.0 * alpha2)) * cr[0] * cr[0] / volume
}

/// [`crate::periodic::strain::StrainSecond`] of the screened Hartree energy
/// at fixed density images: the real-space `erfc` quartets in closed form
/// (four-centre first derivatives and mixed second-derivative tensors,
/// centres referred to the bra's first centre) and the reciprocal kernels
/// with the compact background by exact jets.
pub(crate) fn screened_hartree_strain_second(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    screening: &HartreeScreening,
    natoms: usize,
) -> crate::Result<crate::periodic::strain::StrainSecond> {
    use crate::dual::DualN;
    use crate::integrals::deriv::eri_quartet_center_derivatives;
    use crate::integrals::deriv2::{quartet_mixed_second, PAIRS};
    use crate::periodic::strain::{strain_seed, StrainSecond};
    type MixedJet = DualN<DualN<f64, 9>, 3>;
    let prep = Prepared::new(mb, lattice, density, screening)?;
    let kernel = CoulombKernel::Erfc { omega: prep.alpha };
    let dim = 3 * natoms;
    // ---- real-space quartets ----
    let mut out = prep
        .kets
        .par_iter()
        .map(|bra_ket| {
            let mut out = StrainSecond::zeros(natoms);
            let bra = &prep.pairs[bra_ket.pair];
            if bra.diffuse {
                return out;
            }
            let sa = &prep.parts[bra.x].shell;
            let sb = moved(&prep.parts[bra.y].shell, bra.translation);
            let db = &bra_ket.block;
            for (ket, cell) in prep.real_space_partners(bra) {
                let kp = &prep.pairs[ket.pair];
                let sc = moved(&prep.parts[kp.x].shell, cell);
                let sd = moved(&prep.parts[kp.y].shell, cell + kp.translation);
                let shells = [sa, &sb, &sc, &sd];
                let dims = [
                    crate::basis::n_sph(sa.l),
                    crate::basis::n_sph(sb.l),
                    crate::basis::n_sph(sc.l),
                    crate::basis::n_sph(sd.l),
                ];
                let mut weights = Vec::with_capacity(dims.iter().product());
                for i0 in 0..dims[0] {
                    for i1 in 0..dims[1] {
                        for i2 in 0..dims[2] {
                            for i3 in 0..dims[3] {
                                weights.push(0.5 * db[(i0, i1)] * ket.block[(i2, i3)]);
                            }
                        }
                    }
                }
                // Mixed second derivatives; same-centre blocks by invariance.
                let d2t = quartet_mixed_second(shells, kernel);
                let mut h = [[[[0.0; 3]; 3]; 4]; 4];
                for (slot, &(i, j)) in PAIRS.iter().enumerate() {
                    for x in 0..3 {
                        for y in 0..3 {
                            let v: f64 = d2t[slot][3 * x + y]
                                .iter()
                                .zip(&weights)
                                .map(|(a, b)| a * b)
                                .sum();
                            h[i][j][x][y] = v;
                            h[j][i][y][x] = v;
                        }
                    }
                }
                for i in 0..4 {
                    for x in 0..3 {
                        for y in 0..3 {
                            let mut v = 0.0;
                            for j in 0..4 {
                                if j != i {
                                    v -= h[i][j][x][y];
                                }
                            }
                            h[i][i][x][y] = v;
                        }
                    }
                }
                let origin = sa.center;
                let r: [Vec3; 4] = std::array::from_fn(|i| shells[i].center - origin);
                for i in 0..4 {
                    for j in 0..4 {
                        for a in 0..3 {
                            for b in 0..3 {
                                for c in 0..3 {
                                    for d in 0..3 {
                                        out.strain[(3 * a + b) * 9 + 3 * c + d] +=
                                            h[i][j][a][c] * r[i].get(b) * r[j].get(d);
                                    }
                                }
                            }
                        }
                    }
                }
                // A displacement of the sole owner translates the entire
                // quartet at every strain. Its internal-strain term is zero.
                if shells.iter().all(|shell| shell.atom == sa.atom) {
                    continue;
                }
                let derivatives = eri_quartet_center_derivatives(shells, kernel);
                let mut f = [[0.0; 3]; 4];
                for (centre, per_axis) in derivatives.iter().enumerate() {
                    for (axis, q) in per_axis.iter().enumerate() {
                        f[centre][axis] = contract_quartet(q, db, &ket.block) * 0.5;
                    }
                }
                for u in 0..natoms {
                    let s: [f64; 4] = std::array::from_fn(|i| (shells[i].atom == u) as i32 as f64);
                    if s.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            for c in 0..3 {
                                let mut v = 0.0;
                                for i in 0..4 {
                                    for j in 0..4 {
                                        v += h[i][j][a][c] * s[j] * r[i].get(b);
                                    }
                                    if b == c {
                                        v += f[i][a] * s[i];
                                    }
                                }
                                out.mixed[(3 * a + b) * dim + 3 * u + c] += v;
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
        );
    // ---- reciprocal kernels and background by jets ----
    let eps = strain_seed();
    let energy = hartree_reciprocal_jet(&prep, &eps, None);
    for k in 0..9 {
        for l in 0..9 {
            out.strain[k * 9 + l] += energy.second(k, l);
        }
    }
    // The only nuclear displacement in a one-site cell is a common
    // translation. It changes each transform by the same phase and leaves
    // their products and the zero-wave-vector background invariant.
    if natoms <= 1 {
        return Ok(out);
    }
    let inner: [[MixedJet; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| MixedJet::constant(DualN::<f64, 9>::variable(0.0, 3 * a + b)))
    });
    for u in 0..natoms {
        let moving = move |atom: usize| -> [MixedJet; 3] {
            if atom == u {
                std::array::from_fn(|c| {
                    let mut x = MixedJet::constant(DualN::<f64, 9>::constant(0.0));
                    x.im[c] = DualN::<f64, 9>::constant(1.0);
                    x
                })
            } else {
                [<MixedJet as crate::dual::Scalar>::from_f64(0.0); 3]
            }
        };
        let energy = hartree_reciprocal_jet(&prep, &inner, Some(&moving));
        for k in 0..9 {
            for c in 0..3 {
                out.mixed[k * dim + 3 * u + c] += energy.im[c].im[k];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::coulomb::hartree_image_blocks;
    use crate::periodic::ewald::EwaldParams;
    use crate::periodic::integrals::bloch;
    use crate::system::{Atom, Molecule};

    #[test]
    fn diffuse_reciprocal_terms_survive_a_dominant_compact_transform() {
        use crate::dual::DualN;
        let small = 2.0_f64.powi(-60);
        let compact = [1.0, -2.0];
        let diffuse = [small, -0.5 * small];
        let other_compact = [0.5, 4.0];
        let other_diffuse = [2.0 * small, -small];
        // In exact arithmetic the mixed cross term is 2.5 * small, plus a
        // quadratic term smaller than one ulp of that result.
        let value = diffuse_reciprocal_product(compact, diffuse, other_compact, other_diffuse);
        assert_eq!(value, 2.5 * small);
        let dual = DualN::<f64, 1>::constant;
        let left = [DualN::variable(small, 0), dual(-0.5 * small)];
        let response = diffuse_reciprocal_product(
            compact.map(dual),
            left,
            other_compact.map(dual),
            other_diffuse.map(dual),
        );
        assert_eq!(response.re, value);
        assert_eq!(response.im[0], 0.5);
    }

    #[test]
    fn fixed_density_hartree_affine_hessian_matches_all_ordered_strain_slopes() {
        check_ordered_affine_hartree_slopes(2);
    }

    #[test]
    fn one_site_hartree_strain_hessian_and_translation_limit() {
        check_ordered_affine_hartree_slopes(1);
    }

    fn check_ordered_affine_hartree_slopes(natoms: usize) {
        use crate::acceptance::{components, stencils};
        use stencils::{FitErrors, Stencil};

        let set = crate::basis::g94::parse(
            "mixed-exponent triclinic cell",
            "H 0\nS 2 1.0\n2.2 0.62\n0.53 0.48\n****\nHe 0\nS 2 1.0\n1.9 0.55\n0.71 0.56\n****\n",
        )
        .unwrap();
        let lattice = [
            Vec3::new(5.2, 0.31, 0.23),
            Vec3::new(0.18, 5.7, 0.49),
            Vec3::new(0.29, -0.37, 6.3),
        ];
        let points = [
            (1, Vec3::new(0.13, -0.21, 0.31)),
            (2, Vec3::new(1.67, 0.42, -0.27)),
        ];
        let build = |column: usize, step: f64| {
            let apply = |v: Vec3| {
                let mut result = v.to_array();
                result[column / 3] += step * result[column % 3];
                Vec3::new(result[0], result[1], result[2])
            };
            let molecule = Molecule::new(
                points
                    .iter()
                    .take(natoms)
                    .map(|&(z, p)| Atom {
                        z,
                        position: apply(p),
                    })
                    .collect(),
            );
            let [a, b, c] = lattice.map(apply);
            (
                MolBasis::build(&molecule, &set).unwrap(),
                Lattice::new(a, b, c),
            )
        };
        let screening = HartreeScreening {
            compact_exponent: 1.0,
            tolerance: 1e-15,
            max_translation: Some(3.0),
        };
        let density = [ImageBlock {
            translation: Vec3::zero(),
            matrix: if natoms == 1 {
                Matrix::from_row_major(1, 1, vec![0.79])
            } else {
                Matrix::from_row_major(2, 2, vec![0.79, 0.13, 0.13, 0.57])
            },
        }];
        let stress = |column, step| {
            let (basis, lattice) = build(column, step);
            screened_hartree_strain(&basis, &lattice, &density, &screening)
                .unwrap()
                .strain_derivative
        };
        let (basis, lattice) = build(0, 0.0);
        let reference = stress(0, 0.0);
        let second =
            screened_hartree_strain_second(&basis, &lattice, &density, &screening, natoms).unwrap();
        if natoms == 1 {
            assert_eq!(second.mixed, vec![0.0; 27]);
            // A common translation at fixed lattice leaves every ordered
            // stress entry invariant. Check all Cartesian translations at
            // four signed magnitudes against the ordinary stress path.
            let mut maximum: f64 = 0.0;
            for direction in [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ] {
                for h in [0.024, 0.016, 0.011, 0.008] {
                    for sign in [1.0, -1.0] {
                        let mut displaced = basis.clone();
                        for shell in &mut displaced.shells {
                            shell.center += direction * (sign * h);
                        }
                        let moved =
                            screened_hartree_strain(&displaced, &lattice, &density, &screening)
                                .unwrap()
                                .strain_derivative;
                        for (value, exact) in moved.iter().flatten().zip(reference.iter().flatten())
                        {
                            maximum = maximum.max((value - exact).abs());
                        }
                    }
                }
            }
            println!("one-site Hartree stress maximum translation defect={maximum:.12e}");
            assert!(maximum < 3e-13);
        }
        let hessian = second.strain;
        let entries: Vec<_> = (0..9)
            .flat_map(|column| (0..9).map(move |row| (column, row)))
            .collect();
        let labels: Vec<_> = entries
            .iter()
            .map(|(column, row)| format!("triclinic Hartree [{row},{column}]"))
            .collect();
        let mut failures = Vec::new();
        stencils::fit_components_staged_with_accuracy(
            &labels,
            &[
                ([0.024, 0.016, 0.011, 0.008], Stencil::Second),
                ([0.012, 0.008, 0.0055, 0.004], Stencil::Second),
            ],
            |steps, stencil| {
                let samples: Vec<_> = (0..9)
                    .map(|column| steps.map(|h| (stress(column, h), stress(column, -h))))
                    .collect();
                entries
                    .iter()
                    .map(|&(column, row)| {
                        let (a, b, c, d) = (row / 3, row % 3, column / 3, column % 3);
                        // A local strain of the displaced cell composes on
                        // the left of the strain in the reference chart.
                        let exact =
                            hessian[9 * row + column] + if b == c { reference[a][d] } else { 0.0 };
                        let errors = std::array::from_fn::<_, 4, _>(|step| {
                            let (plus, minus) = &samples[column][step];
                            stencil.first(plus[a][b], minus[a][b], 0.0, 0.0, steps[step]) - exact
                        });
                        let ratio = (steps[2] / steps[3]).powi(2);
                        let extrapolated_error =
                            ((ratio * errors[3] - errors[2]) / (ratio - 1.0)).abs();
                        FitErrors {
                            errors: errors.map(f64::abs),
                            richardson: Some((extrapolated_error, 2e-8)),
                        }
                    })
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures(
            "ordered fixed-density Hartree strain derivatives",
            &failures,
        );
    }

    /// Two hydrogens with an s and a p shell, each contracted from one
    /// compact and one diffuse primitive at the default split, so every
    /// branch (compact real space, compact reciprocal, diffuse reciprocal)
    /// is exercised. The exponents keep the unscreened reference converged
    /// at a modest image cutoff.
    fn h2_cell() -> (MolBasis, Lattice) {
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
        let set = crate::basis::g94::parse(
            "split-test",
            "H 0
S 2 1.00
 2.4 0.6
 0.55 0.5
P 2 1.00
 1.6 0.6
 0.5 0.5
****
",
        )
        .unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let lattice = Lattice::new(
            Vec3::new(5.5, 0.0, 0.0),
            Vec3::new(0.6, 5.8, 0.0),
            Vec3::new(0.0, 0.3, 6.2),
        );
        (mb, lattice)
    }

    /// A charged, translation-dependent density with D(-T) = D(T)^T.
    fn density(mb: &MolBasis, lattice: &Lattice) -> Vec<ImageBlock> {
        let n = mb.nao_sph;
        let mut raw: Vec<(Vec3, Matrix)> = Vec::new();
        for t in lattice.translations_within(6.5) {
            let decay = (-0.12 * t.norm2()).exp();
            let mut m = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let phase = 0.37 * t.x - 0.21 * t.y + 0.13 * t.z;
                    let base = ((i * 5 + j * 3) % 7) as f64 / 7.0 - 0.3;
                    m[(i, j)] = decay * (base + 0.2 * phase.sin() * ((i + 2 * j) as f64 * 0.1));
                }
            }
            raw.push((t, m));
        }
        raw.iter()
            .map(|(t, m)| {
                let mirror = &raw.iter().find(|(u, _)| (*u + *t).norm() < 1e-8).unwrap().1;
                let mut s = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        s[(i, j)] = 0.5 * (m[(i, j)] + mirror[(j, i)]);
                    }
                }
                ImageBlock {
                    translation: *t,
                    matrix: s,
                }
            })
            .collect()
    }

    #[test]
    fn default_split_exponent_follows_the_cell_volume() {
        for volume in [20.0, 54.0] {
            assert_eq!(default_compact_exponent(volume), 3.0, "volume {volume}");
        }
        for volume in [300.0, 5000.0] {
            assert_eq!(default_compact_exponent(volume), 1.0, "volume {volume}");
        }
        let middle = default_compact_exponent(120.0);
        assert!(middle > 1.0 && middle < 3.0, "volume 120: {middle}");
        let volumes = [30.0, 60.0, 100.0, 200.0, 400.0];
        let values: Vec<f64> = volumes
            .iter()
            .map(|v| default_compact_exponent(*v))
            .collect();
        for pair in values.windows(2) {
            assert!(pair[1] <= pair[0], "{values:?}");
        }
    }

    #[test]
    fn range_separated_hartree_matches_the_reference_for_every_split() {
        let (mb, lattice) = h2_cell();
        let d = density(&mb, &lattice);
        let k = Vec3::new(0.11, -0.07, 0.19);
        let alpha = 1.4;
        let start = std::time::Instant::now();
        let reference = bloch(
            &hartree_image_blocks(
                &mb,
                &lattice,
                &d,
                11.0,
                EwaldParams {
                    alpha,
                    rcut: 6.0 / alpha,
                    gcut: 12.0 * alpha,
                },
            ),
            k,
        );
        println!("reference build {:.2} s", start.elapsed().as_secs_f64());
        let mut previous: Option<(Matrix, Matrix)> = None;
        for compact_exponent in [0.4, 1.0, 2.0] {
            let screening = HartreeScreening {
                compact_exponent,
                tolerance: 1e-12,
                max_translation: None,
            };
            let start = std::time::Instant::now();
            let blocks = screened_hartree_image_blocks(&mb, &lattice, &d, &screening).unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let screened = bloch(&blocks, k);
            let dre = screened.0.max_abs_diff(&reference.0);
            let dim = screened.1.max_abs_diff(&reference.1);
            println!(
                "compact exponent {compact_exponent}: {} blocks in {elapsed:.2} s, |J - J_ref| real {dre:e} imaginary {dim:e}",
                blocks.len()
            );
            assert!(
                dre < 1e-7 && dim < 1e-7,
                "split {compact_exponent}: {dre:e} {dim:e}"
            );
            // Different splits are independent evaluations of one matrix.
            if let Some(p) = &previous {
                let spread = screened
                    .0
                    .max_abs_diff(&p.0)
                    .max(screened.1.max_abs_diff(&p.1));
                println!("  change from the previous split {spread:e}");
                assert!(spread < 1e-9, "split dependence {spread:e}");
            }
            previous = Some(screened);
        }
    }

    /// The analytic Hartree gradient at fixed AO density agrees with an
    /// independent four-step fit of the energy, sums to zero under a rigid
    /// translation, and its energy equals 1/2 sum Tr[D^T J].
    #[test]
    fn hartree_gradient_passes_a_four_step_fit_and_translation_rule() {
        let (mb, lattice) = h2_cell();
        let d = density(&mb, &lattice);
        let screening = HartreeScreening::default();
        let derivatives = screened_hartree_derivatives(&mb, &lattice, &d, &screening, 2).unwrap();
        let blocks = screened_hartree_image_blocks(&mb, &lattice, &d, &screening).unwrap();
        let mut energy = 0.0;
        for b in &blocks {
            if let Some(db) = d
                .iter()
                .find(|x| (x.translation - b.translation).norm() < 1e-8)
            {
                energy += 0.5 * b.matrix.frobenius_dot(&db.matrix);
            }
        }
        assert!(
            (energy - derivatives.energy).abs() < 1e-10,
            "{energy} vs {}",
            derivatives.energy
        );
        let sum = derivatives.gradient[0] + derivatives.gradient[1];
        assert!(sum.norm() < 1e-9, "translation sum {sum:?}");
        let energy_at = |atom: usize, axis: usize, h: f64| {
            let mut mol = crate::system::Molecule::new(
                mb.shells
                    .iter()
                    .map(|s| s.atom)
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .map(|a| {
                        let centre = mb.shells.iter().find(|s| s.atom == a).unwrap().center;
                        Atom {
                            z: 1,
                            position: centre,
                        }
                    })
                    .collect(),
            );
            match axis {
                0 => mol.atoms[atom].position.x += h,
                1 => mol.atoms[atom].position.y += h,
                _ => mol.atoms[atom].position.z += h,
            }
            let set = crate::basis::g94::parse(
                "split-test",
                "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\n****\n",
            )
            .unwrap();
            let moved_basis = MolBasis::build(&mol, &set).unwrap();
            screened_hartree_derivatives(
                &moved_basis,
                &lattice,
                &d,
                &HartreeScreening {
                    tolerance: 1e-15,
                    ..screening
                },
                2,
            )
            .unwrap()
            .energy
        };
        let steps = [0.02, 0.01, 0.005, 0.0025];
        for (atom, axis) in [(1usize, 2usize), (0, 0)] {
            let analytic = derivatives.gradient[atom].get(axis);
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    ((energy_at(atom, axis, h) - energy_at(atom, axis, -h)) / (2.0 * h) - analytic)
                        .abs()
                })
                .collect();
            let (x, y): (Vec<f64>, Vec<f64>) = steps
                .iter()
                .zip(&errors)
                .map(|(h, e)| (h.ln(), e.ln()))
                .unzip();
            let n = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
            let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let p = sxy / sxx;
            let c = my - p * mx;
            let sse: f64 = x.iter().zip(&y).map(|(a, b)| (b - c - p * a).powi(2)).sum();
            let sst: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
            let r2 = 1.0 - sse / sst;
            println!("dE_J/dR atom {atom} axis {axis}: analytic {analytic:.12e} errors {errors:?} p={p:.6} R2={r2:.8}");
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999, "p={p} R2={r2}");
        }
    }
}
