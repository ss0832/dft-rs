// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic Gamma-point force constants of a finite-k periodic SCF by the
//! coupled-perturbed Kohn-Sham equations.
//!
//! A Gamma-point displacement `x` moves an atom and all of its lattice
//! images. With the skeleton second derivatives of [`super::hessian`],
//! [`super::screened_coulomb`] and [`super::xc_hessian`], the first-order
//! skeleton Fock and overlap matrices `F^x(k)`, `S^x(k)`, and the response
//! density `P^y(k)` and energy-weighted density `W^y(k)`,
//!
//! ```text
//! H_xy = E_skel,xy + Nk^-1 sum_{k,s} [P^y . F^x - W^y . S^x](k)
//! ```
//!
//! For every k point and spin with orbitals `C = (C_o, C_v)` and
//! `S~ = C^+ S^x C`, the virtual-occupied orbital response solves
//!
//! ```text
//! (e_a - e_i) U_ai + [C_v^+ G(P) C_o]_ai = -[C_v^+ F^x C_o]_ai + S~_ai e_i
//! P = C_v U C_o^+ + C_o U^+ C_v^+ - C_o S~_oo C_o^+
//! W = C_v U e C_o^+ + h.c. + C_o [C_o^+ F^x_tot C_o - (S~ e + e S~)_oo] C_o^+
//! ```
//!
//! with `G(P) = J[P] + f_xc[P]` from the screened Hartree build and the
//! semilocal kernel. The response operator is real-linear in `U`, so it is
//! solved in real coordinates by preconditioned conjugate gradients.

use crate::dfpt::orbitals::{generalized_orbitals, BandOrbitals};
use crate::error::{DftError, Result};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::exchange::minimum_image_blocks;
use crate::periodic::exchange_response::{
    exchange_first_order, exchange_second_derivatives, ExchangeFirstOrder,
};
use crate::periodic::integrals::{bloch, ImageBlock, Operator};
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::response::complex::Complex;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::xc::ResolvedFunctional;

/// Analytic Gamma-point force constants.
#[derive(Clone, Debug)]
pub struct GammaForceConstants {
    /// Symmetrized `d2E/dR_(A,x) dR_(B,y)`, `3N x 3N` (Hartree/Bohr^2).
    pub hessian: Matrix,
    /// The matrix before symmetrization: column `y` is the derivative of the
    /// analytic gradient with respect to displacement `y`.
    pub raw: Matrix,
    /// Largest `|H_xy - H_yx|` before symmetrization.
    pub raw_asymmetry: f64,
    /// Largest `|sum_B H_(A,x),(B,y)|`: the acoustic sum rule residual.
    pub acoustic_residual: f64,
    /// Skeleton part `E_skel,xy` (Hartree/Bohr^2).
    pub skeleton: Matrix,
    /// Krylov iterations and final relative residual of each perturbation.
    pub solves: Vec<(usize, f64)>,
}

impl GammaForceConstants {
    /// Add the Hessian of a geometry-only energy to all force-constant
    /// views and recompute the diagnostics on the complete operator.
    pub(crate) fn add_geometry_hessian(&mut self, contribution: &Matrix) {
        for target in [&mut self.skeleton, &mut self.raw] {
            for (value, extra) in target
                .as_mut_slice()
                .iter_mut()
                .zip(contribution.as_slice())
            {
                *value += extra;
            }
        }
        let (asymmetry, acoustic) = symmetrize_force_constants(&self.raw, &mut self.hessian);
        self.raw_asymmetry = asymmetry;
        self.acoustic_residual = acoustic;
    }
}

/// Common mass weighting for parent and correlated harmonic force constants.
pub fn gamma_modes(hessian: &Matrix, numbers: &[u8]) -> Result<(Vec<f64>, Vec<Vec<[f64; 3]>>)> {
    let n = 3 * numbers.len();
    if n == 0 || hessian.rows != n || hessian.cols != n {
        return Err(DftError::InvalidInput(
            "Gamma modes need a 3N by 3N Hessian and N atomic numbers".into(),
        ));
    }
    let mass: Vec<_> = numbers
        .iter()
        .map(|z| 1.0 / crate::constants::atomic_mass_au(*z).sqrt())
        .collect();
    let mut weighted = hessian.clone();
    for i in 0..n {
        for j in 0..n {
            weighted[(i, j)] *= mass[i / 3] * mass[j / 3];
        }
    }
    let (e, v) = crate::linalg::symmetric_eigen(&weighted)?;
    let frequencies = e
        .iter()
        .map(|e| e.signum() * e.abs().sqrt() * crate::constants::HARTREE_TO_CM1)
        .collect();
    let modes = (0..n)
        .map(|k| {
            (0..numbers.len())
                .map(|a| std::array::from_fn(|x| v[(3 * a + x, k)] * mass[a]))
                .collect()
        })
        .collect();
    Ok((frequencies, modes))
}

/// Complex dense helpers on `BlochMatrix` values (row-major re/im).
pub(crate) fn product(a: &BlochMatrix, b: &BlochMatrix) -> BlochMatrix {
    let (re1, im1) = (a.re.matmul(&b.re), a.im.matmul(&b.im));
    let (re2, im2) = (a.re.matmul(&b.im), a.im.matmul(&b.re));
    let mut re = re1;
    for (x, y) in re.as_mut_slice().iter_mut().zip(im1.as_slice()) {
        *x -= *y;
    }
    let mut im = re2;
    for (x, y) in im.as_mut_slice().iter_mut().zip(im2.as_slice()) {
        *x += *y;
    }
    BlochMatrix { re, im }
}

pub(crate) fn adjoint(a: &BlochMatrix) -> BlochMatrix {
    let mut im = a.im.transpose();
    for v in im.as_mut_slice() {
        *v = -*v;
    }
    BlochMatrix {
        re: a.re.transpose(),
        im,
    }
}

fn columns(a: &BlochMatrix, range: std::ops::Range<usize>) -> BlochMatrix {
    let mut out = BlochMatrix::zeros(a.rows(), range.len());
    for i in 0..a.rows() {
        for (c, j) in range.clone().enumerate() {
            out.re[(i, c)] = a.re[(i, j)];
            out.im[(i, c)] = a.im[(i, j)];
        }
    }
    out
}

/// `sum_mn Re(a_mn conj(b_mn))`.
pub(crate) fn frobenius(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.frobenius_dot(&b.re) + a.im.frobenius_dot(&b.im)
}

/// Largest anti-Hermitian part accepted in a response density, relative to
/// `max(1, largest element)`.
const HERMITICITY_TOLERANCE: f64 = 1e-9;

/// Orbitals of one spin and k point split into occupied and virtual blocks.
pub(crate) struct Channel {
    pub(crate) occupied: BlochMatrix,
    pub(crate) virtuals: BlochMatrix,
    pub(crate) e_occ: Vec<f64>,
    pub(crate) e_virt: Vec<f64>,
}

impl Channel {
    fn new(orbitals: &BandOrbitals, nocc: usize) -> Self {
        let nmo = orbitals.energies.len();
        Self {
            occupied: columns(&orbitals.coefficients, 0..nocc),
            virtuals: columns(&orbitals.coefficients, nocc..nmo),
            e_occ: orbitals.energies[..nocc].to_vec(),
            e_virt: orbitals.energies[nocc..].to_vec(),
        }
    }

    fn size(&self) -> usize {
        self.e_occ.len() * self.e_virt.len()
    }
}

/// The skeleton second derivatives, first-order matrices and response
/// kernel shared by every perturbation.
struct Setup<'a> {
    cell: &'a Cell,
    k_points: Vec<crate::math::Vec3>,
    /// Index of `-k` (modulo the reciprocal lattice) for every k point.
    partners: Vec<usize>,
    channels: [Vec<Channel>; 2],
    xc_kernel: Option<crate::periodic::xc_terms::ResponseKernel>,
    screening: crate::periodic::screened_coulomb::HartreeScreening,
    image_cutoff: f64,
    /// Exact-exchange terms `(coefficient, kernel)` of the functional.
    exchange: Vec<(f64, CoulombKernel)>,
    exchange_cutoff: f64,
}

impl Setup<'_> {
    fn nk(&self) -> usize {
        self.k_points.len()
    }

    /// Projection onto time-reversal-symmetric densities,
    /// `P(k) -> [P(k) + P(-k)^*] / 2`. A real Gamma-point displacement has a
    /// time-reversal-symmetric response, and the real-image Hartree engine
    /// acts on that subspace; Krylov vectors built from a numerically zero
    /// source would otherwise carry arbitrary phases out of it.
    fn time_reversal(&self, density: &[Vec<BlochMatrix>; 2]) -> [Vec<BlochMatrix>; 2] {
        std::array::from_fn(|spin| {
            (0..self.nk())
                .map(|ik| {
                    let partner = &density[spin][self.partners[ik]];
                    let mut out = density[spin][ik].clone();
                    out.re
                        .as_mut_slice()
                        .iter_mut()
                        .zip(partner.re.as_slice())
                        .for_each(|(a, b)| *a = 0.5 * (*a + b));
                    out.im
                        .as_mut_slice()
                        .iter_mut()
                        .zip(partner.im.as_slice())
                        .for_each(|(a, b)| *a = 0.5 * (*a - b));
                    out
                })
                .collect()
        })
    }

    /// `G(P) = J[P_alpha + P_beta] + f_xc[P]` at every k point and spin, on
    /// the time-reversal-symmetric part of `P`.
    fn kernel(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        // The kernel is linear, but the screened builds use absolute
        // thresholds; applying it to the unit-normalized density and
        // rescaling keeps it exactly homogeneous for any source size.
        let mut density = self.time_reversal(density);
        // A response density is Hermitian. A source that vanishes by symmetry
        // (the first-order overlap of a cell whose atoms are all inversion
        // centres) leaves only rounding noise, which the normalization below
        // would magnify into an arbitrary matrix: the anti-Hermitian part is
        // measured on the absolute scale and projected out.
        let mut defect = 0.0_f64;
        for m in density.iter_mut().flatten() {
            for i in 0..m.rows() {
                for j in i..m.rows() {
                    let re = 0.5 * (m.re[(i, j)] + m.re[(j, i)]);
                    let im = 0.5 * (m.im[(i, j)] - m.im[(j, i)]);
                    defect = defect
                        .max((m.re[(i, j)] - re).abs())
                        .max((m.im[(i, j)] - im).abs());
                    m.re[(i, j)] = re;
                    m.re[(j, i)] = re;
                    m.im[(i, j)] = im;
                    m.im[(j, i)] = -im;
                }
            }
        }
        let scale = density
            .iter()
            .flatten()
            .flat_map(|m| m.re.as_slice().iter().chain(m.im.as_slice()))
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        if defect > HERMITICITY_TOLERANCE * scale.max(1.0) {
            return Err(DftError::InvalidInput(format!(
                "periodic response density is not Hermitian (anti-Hermitian part {defect:e}, largest element {scale:e})"
            )));
        }
        let nk = self.nk();
        if scale == 0.0 {
            return Ok(std::array::from_fn(|_| {
                (0..nk)
                    .map(|_| BlochMatrix::zeros(self.cell.nao(), self.cell.nao()))
                    .collect()
            }));
        }
        for m in density.iter_mut().flatten() {
            m.scale(1.0 / scale);
        }
        let mut out = self.unit_kernel(&density)?;
        for m in out.iter_mut().flatten() {
            m.scale(scale);
        }
        Ok(out)
    }

    fn unit_kernel(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        let nk = self.nk();
        let total: Vec<(Matrix, Matrix)> = (0..nk)
            .map(|ik| {
                let mut t = density[0][ik].clone();
                t.add_scaled(&density[1][ik], 1.0);
                (t.re, t.im)
            })
            .collect();
        let images = self
            .cell
            .density_images_from_k(&self.k_points, &total, self.image_cutoff)?;
        let blocks = crate::periodic::screened_coulomb::screened_hartree_image_blocks(
            &self.cell.basis,
            &self.cell.lattice,
            &images,
            &self.screening,
        )?;
        let hartree: Vec<BlochMatrix> = self
            .k_points
            .iter()
            .map(|k| BlochMatrix::from_pair(bloch(&blocks, *k)))
            .collect();
        let mut out: [Vec<BlochMatrix>; 2] = [hartree.clone(), hartree];
        if let Some(kernel) = &self.xc_kernel {
            let xc = kernel.apply(density)?;
            for spin in 0..2 {
                for ik in 0..nk {
                    out[spin][ik].add_scaled(&xc[spin][ik], 1.0);
                }
            }
        }
        if !self.exchange.is_empty() {
            let nao = self.cell.nao();
            // Identical spin densities (a restricted reference) share one
            // exchange build.
            let same = density[0].iter().zip(&density[1]).all(|(a, b)| {
                a.re.as_slice() == b.re.as_slice() && a.im.as_slice() == b.im.as_slice()
            });
            let mut exchange: Vec<Vec<BlochMatrix>> = Vec::with_capacity(2);
            for (spin, spin_density) in density.iter().enumerate() {
                if spin == 1 && same {
                    let copy = exchange[0].clone();
                    exchange.push(copy);
                    continue;
                }
                let images = minimum_image_blocks(
                    self.cell,
                    &self.k_points,
                    images_of(
                        self.cell,
                        &self.k_points,
                        spin_density,
                        2.0 * self.exchange_cutoff,
                    )?,
                );
                let mut sum: Vec<BlochMatrix> =
                    (0..nk).map(|_| BlochMatrix::zeros(nao, nao)).collect();
                for (coefficient, kernel) in &self.exchange {
                    let blocks = minimum_image_blocks(
                        self.cell,
                        &self.k_points,
                        crate::periodic::exchange::exchange_image_blocks(
                            &self.cell.basis,
                            &self.cell.lattice,
                            &images,
                            self.exchange_cutoff,
                            *kernel,
                        ),
                    );
                    for (target, k) in sum.iter_mut().zip(&self.k_points) {
                        target
                            .add_scaled(&BlochMatrix::from_pair(bloch(&blocks, *k)), -*coefficient);
                    }
                }
                exchange.push(sum);
            }
            for (spin, per_spin) in exchange.iter().enumerate() {
                for (ik, matrix) in per_spin.iter().enumerate() {
                    out[spin][ik].add_scaled(matrix, 1.0);
                }
            }
        }
        Ok(out)
    }

    /// Response density of the virtual-occupied amplitudes.
    fn density(&self, u: &[Vec<BlochMatrix>; 2]) -> [Vec<BlochMatrix>; 2] {
        std::array::from_fn(|spin| {
            (0..self.nk())
                .map(|ik| {
                    let ch = &self.channels[spin][ik];
                    let p = product(&product(&ch.virtuals, &u[spin][ik]), &adjoint(&ch.occupied));
                    let mut sym = p.clone();
                    sym.add_scaled(&adjoint(&p), 1.0);
                    sym
                })
                .collect()
        })
    }

    fn unpack(&self, x: &[Complex]) -> [Vec<BlochMatrix>; 2] {
        let mut offset = 0;
        std::array::from_fn(|spin| {
            (0..self.nk())
                .map(|ik| {
                    let ch = &self.channels[spin][ik];
                    let (nv, no) = (ch.e_virt.len(), ch.e_occ.len());
                    let mut u = BlochMatrix::zeros(nv, no);
                    for a in 0..nv {
                        for i in 0..no {
                            u.re[(a, i)] = x[offset].re;
                            u.im[(a, i)] = x[offset + 1].re;
                            offset += 2;
                        }
                    }
                    u
                })
                .collect()
        })
    }

    fn pack(&self, blocks: &[Vec<BlochMatrix>; 2]) -> Vec<Complex> {
        let mut out = Vec::new();
        for spin in 0..2 {
            for ik in 0..self.nk() {
                let b = &blocks[spin][ik];
                for a in 0..b.rows() {
                    for i in 0..b.cols() {
                        out.push(Complex::new(b.re[(a, i)], 0.0));
                        out.push(Complex::new(b.im[(a, i)], 0.0));
                    }
                }
            }
        }
        out
    }

    /// `C_v^+ M C_o` of every channel.
    fn vo_block(&self, m: &[Vec<BlochMatrix>; 2]) -> [Vec<BlochMatrix>; 2] {
        std::array::from_fn(|spin| {
            (0..self.nk())
                .map(|ik| {
                    let ch = &self.channels[spin][ik];
                    product(&product(&adjoint(&ch.virtuals), &m[spin][ik]), &ch.occupied)
                })
                .collect()
        })
    }

    fn dimension(&self) -> usize {
        2 * self
            .channels
            .iter()
            .flatten()
            .map(Channel::size)
            .sum::<usize>()
    }
}

struct ResponseOperator<'a, 'b> {
    setup: &'a Setup<'b>,
}

impl LinearOperator for ResponseOperator<'_, '_> {
    fn dimension(&self) -> usize {
        self.setup.dimension()
    }

    fn structure(&self) -> OperatorStructure {
        OperatorStructure::PositiveDefinite
    }

    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |real| {
            let input: Vec<_> = real.iter().copied().map(Complex::from).collect();
            Ok(self.apply_real(&input)?.into_iter().map(|z| z.re).collect())
        })
    }
}

impl ResponseOperator<'_, '_> {
    fn apply_real(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        let u = self.setup.unpack(x);
        let induced = self.setup.kernel(&self.setup.density(&u))?;
        let mut out = self.setup.vo_block(&induced);
        for spin in 0..2 {
            for ik in 0..self.setup.nk() {
                let ch = &self.setup.channels[spin][ik];
                for (a, ea) in ch.e_virt.iter().enumerate() {
                    for (i, ei) in ch.e_occ.iter().enumerate() {
                        out[spin][ik].re[(a, i)] += (ea - ei) * u[spin][ik].re[(a, i)];
                        out[spin][ik].im[(a, i)] += (ea - ei) * u[spin][ik].im[(a, i)];
                    }
                }
            }
        }
        Ok(self.setup.pack(&out))
    }
}

fn images_of(
    cell: &Cell,
    k_points: &[crate::math::Vec3],
    m: &[BlochMatrix],
    cutoff: f64,
) -> Result<Vec<ImageBlock>> {
    cell.density_images_from_k(
        k_points,
        &m.iter()
            .map(|p| (p.re.clone(), p.im.clone()))
            .collect::<Vec<_>>(),
        cutoff,
    )
}

/// First-order skeleton matrices `F^x(k)` (per spin) and `S^x(k)` of one
/// perturbation at every k point.
pub(crate) struct FirstOrder {
    pub(crate) fock: [Vec<BlochMatrix>; 2],
    pub(crate) overlap: Vec<BlochMatrix>,
    /// Per-spin `F^x(k)` that pairs with a response density, where it
    /// differs from `fock`: the density derivative of the truncated exchange
    /// gradient is not the derivative of the exchange matrix. `None` means
    /// `fock`.
    pub(crate) pairing_fock: Option<[Vec<BlochMatrix>; 2]>,
}

/// Response density `P^y(k)` and energy-weighted density `W^y(k)` of one
/// perturbation, with the Krylov iterations and final relative residual.
pub(crate) struct Response {
    pub(crate) density: [Vec<BlochMatrix>; 2],
    pub(crate) weighted: [Vec<BlochMatrix>; 2],
    /// Virtual-occupied orbital amplitudes `U(k)` per spin.
    pub(crate) amplitudes: [Vec<BlochMatrix>; 2],
    /// Total first-order Fock matrices `F^x + G[P^x]` per spin and k point:
    /// the self-consistent perturbation of the Kohn-Sham operator.
    pub(crate) fock: [Vec<BlochMatrix>; 2],
    pub(crate) iterations: usize,
    pub(crate) residual: f64,
}

/// `Nk^-1 sum_{k,s} [P^y . F^x - W^y . S^x](k)`.
pub(crate) fn pairing(first: &FirstOrder, response: &Response) -> f64 {
    let nk = first.overlap.len();
    let mut value = 0.0;
    let fock = first.pairing_fock.as_ref().unwrap_or(&first.fock);
    for spin in 0..2 {
        for ik in 0..nk {
            value += frobenius(&response.density[spin][ik], &fock[spin][ik])
                - frobenius(&response.weighted[spin][ik], &first.overlap[ik]);
        }
    }
    value / nk as f64
}

/// Response capabilities are chosen by the physical driver, independently
/// of registered functional names. ECP strain and electric sources retain
/// their own gates when only the nuclear Hessian is connected.
pub(crate) struct ResponseCapabilities {
    pub metal: bool,
    pub exchange: bool,
    pub scalar_ecp: bool,
}

/// Reject references outside the implemented periodic response: metallic
/// occupations, exact exchange and effective core potentials.
pub(crate) fn check_response_request(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    what: &str,
    capabilities: ResponseCapabilities,
) -> Result<()> {
    result.validate_derivative_request(cell, functional, options)?;
    if !capabilities.metal && !matches!(options.occupation, OccupationScheme::Insulator) {
        return Err(DftError::Unsupported {
            feature: format!("{what} of a Fermi-Dirac (metallic) reference"),
            missing: "the occupation response of the periodic coupled-perturbed equations".into(),
        });
    }
    let exchange = functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0;
    if exchange && !capabilities.exchange {
        return Err(DftError::Unsupported {
            feature: format!("periodic {what} with exact exchange"),
            missing: "the exchange terms of this response; only the Gamma-point force constants include them".into(),
        });
    }
    if !capabilities.scalar_ecp && !cell.basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: format!("periodic {what} with effective core potentials"),
            missing: "the translated ECP operator response for this perturbation".into(),
        });
    }
    Ok(())
}

/// Density images of a converged reference and the cutoffs shared by the
/// skeleton derivatives and the response kernel.
pub(crate) struct Reference {
    pub(crate) k_points: Vec<crate::math::Vec3>,
    pub(crate) total_images: Vec<ImageBlock>,
    pub(crate) weighted_images: Vec<ImageBlock>,
    pub(crate) spin_images: Vec<Vec<ImageBlock>>,
    pub(crate) screening: crate::periodic::screened_coulomb::HartreeScreening,
    pub(crate) ao_cutoff: f64,
    pub(crate) image_cutoff: f64,
    /// Per-spin minimum-image density blocks inside twice the exchange
    /// cutoff, as the self-consistent exchange build uses them.
    pub(crate) exchange_images: Vec<Vec<ImageBlock>>,
    pub(crate) exchange_cutoff: f64,
    pub(crate) unrestricted: bool,
}

impl Reference {
    pub(crate) fn new(
        cell: &Cell,
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
    ) -> Result<Self> {
        let nk = result.k_points.len();
        let k_points = result.k_points.clone();
        let ao_cutoff = options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff);
        let image_cutoff = (2.0 * cell.hartree_real_space_cutoff()).max(2.0 * ao_cutoff + 1.0);
        let total = result.total_density();
        let total_images = images_of(cell, &k_points, &total, image_cutoff)?;
        let weighted: Vec<BlochMatrix> = (0..nk)
            .map(|ik| {
                let mut sum = result.energy_weighted_density[0][ik].clone();
                sum.add_scaled(&result.energy_weighted_density[1][ik], 1.0);
                sum
            })
            .collect();
        let weighted_images = images_of(cell, &k_points, &weighted, image_cutoff)?;
        let spin_images: Vec<Vec<ImageBlock>> = (0..2)
            .map(|spin| images_of(cell, &k_points, &result.density[spin], image_cutoff))
            .collect::<Result<_>>()?;
        let exchange_cutoff = options.exchange_cutoff.unwrap_or(cell.overlap_cutoff);
        let exchange_images: Vec<Vec<ImageBlock>> = (0..2)
            .map(|spin| {
                images_of(
                    cell,
                    &k_points,
                    &result.density[spin],
                    2.0 * exchange_cutoff,
                )
                .map(|blocks| minimum_image_blocks(cell, &k_points, blocks))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            k_points,
            total_images,
            weighted_images,
            spin_images,
            screening: cell.hartree_screening(),
            ao_cutoff,
            image_cutoff,
            exchange_images,
            exchange_cutoff,
            unrestricted: result.unrestricted,
        })
    }
}

/// Skeleton second derivatives and first-order matrices of the Gamma-point
/// atomic displacements `x = 3 atom + axis`.
pub(crate) struct DisplacementSkeleton {
    pub(crate) skeleton: Matrix,
    pub(crate) first: Vec<FirstOrder>,
}

pub(crate) fn displacement_skeleton(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    reference: &Reference,
) -> Result<DisplacementSkeleton> {
    displacement_data(cell, functional, options, reference, true)
}

/// First-order displacement sources without the unrelated second-order
/// kinetic, Pulay, nuclear, Hartree and Ewald force-constant construction.
pub(crate) fn displacement_first_order(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    reference: &Reference,
) -> Result<Vec<FirstOrder>> {
    Ok(displacement_data(cell, functional, options, reference, false)?.first)
}

fn displacement_data(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    reference: &Reference,
    build_hessian: bool,
) -> Result<DisplacementSkeleton> {
    let natoms = cell.molecule.len();
    let dim = 3 * natoms;
    let k_points = &reference.k_points;
    let total_images = &reference.total_images;
    let charges = cell.basis.effective_charges(&cell.molecule);
    // The density-functional terms provide the first Fock derivatives and
    // their skeleton together; both paths reuse them.
    let xc = crate::periodic::xc_terms::displacement_terms(
        cell,
        functional,
        options,
        crate::periodic::xc_terms::Density {
            images: [&reference.spin_images[0], &reference.spin_images[1]],
            ao_cutoff: reference.ao_cutoff,
        },
    )?;
    let mut skeleton = Matrix::zeros(dim, dim);
    if build_hessian {
        skeleton = crate::periodic::hessian::two_centre_hessian(
            &cell.basis,
            &cell.lattice,
            cell.overlap_cutoff,
            Operator::Kinetic,
            total_images,
            natoms,
        );
        let ecp = crate::periodic::integrals::contract_ecp_hessian(cell, total_images)?;
        for (value, contribution) in skeleton.as_mut_slice().iter_mut().zip(ecp.as_slice()) {
            *value += contribution;
        }
        let pulay = crate::periodic::hessian::two_centre_hessian(
            &cell.basis,
            &cell.lattice,
            cell.overlap_cutoff,
            Operator::Overlap,
            &reference.weighted_images,
            natoms,
        );
        let nuclear = crate::periodic::hessian::nuclear_attraction_hessian(
            &cell.basis,
            &cell.molecule,
            &charges,
            &cell.lattice,
            cell.overlap_cutoff,
            cell.ewald_params(),
            total_images,
        );
        let hartree = crate::periodic::screened_coulomb::screened_hartree_hessian(
            &cell.basis,
            &cell.lattice,
            total_images,
            &reference.screening,
            natoms,
        )?;
        let positions: Vec<crate::math::Vec3> =
            cell.molecule.atoms.iter().map(|a| a.position).collect();
        let ewald = crate::periodic::ewald::ewald_hessian(
            &cell.lattice,
            &positions,
            &charges,
            cell.ewald_params(),
        );
        for idx in 0..dim * dim {
            let value = -pulay.as_slice()[idx]
                + nuclear.as_slice()[idx]
                + hartree.as_slice()[idx]
                + ewald.as_slice()[idx]
                + xc.as_ref().map_or(0.0, |x| x.skeleton.as_slice()[idx]);
            skeleton.as_mut_slice()[idx] += value;
        }
    }

    // Exact exchange: second derivatives of the truncated image sum and the
    // first-order exchange matrices of every spin density.
    let hybrid = [
        (functional.hybrid.cx, CoulombKernel::Coulomb),
        (
            functional.hybrid.clr,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        ),
    ];
    let exchange_spins = if reference.unrestricted { 2 } else { 1 };
    let mut exchange: [Vec<(f64, ExchangeFirstOrder)>; 2] = [Vec::new(), Vec::new()];
    for (coefficient, kernel) in hybrid {
        if coefficient == 0.0 {
            continue;
        }
        for (spin, terms) in exchange.iter_mut().enumerate().take(exchange_spins) {
            let images = &reference.exchange_images[spin];
            if build_hessian {
                let second = exchange_second_derivatives(
                    &cell.basis,
                    &cell.lattice,
                    images,
                    reference.exchange_cutoff,
                    kernel,
                    natoms,
                );
                // One spin density of a restricted reference stands for both.
                let factor = -0.5 * coefficient * (3 - exchange_spins) as f64;
                for (target, value) in skeleton.as_mut_slice().iter_mut().zip(second.as_slice()) {
                    *target += factor * value;
                }
            }
            let mut first_order = exchange_first_order(
                &cell.basis,
                &cell.lattice,
                images,
                reference.exchange_cutoff,
                kernel,
                natoms,
            );
            // The exchange matrix and the density it pairs with live on the
            // minimum images of the k mesh.
            for list in first_order
                .source
                .iter_mut()
                .chain(first_order.pairing.iter_mut())
            {
                *list = minimum_image_blocks(cell, k_points, std::mem::take(list));
            }
            terms.push((coefficient, first_order));
        }
    }
    let has_exchange = !exchange[0].is_empty();
    // ---- first-order skeleton matrices at every k ----
    let derivatives = cell.derivative_images();
    let hartree_fock = crate::periodic::screened_coulomb::screened_hartree_fock_derivatives(
        &cell.basis,
        &cell.lattice,
        total_images,
        &reference.screening,
        natoms,
    )?;
    let mut first: Vec<FirstOrder> = (0..dim)
        .map(|_| FirstOrder {
            fock: [
                Vec::with_capacity(k_points.len()),
                Vec::with_capacity(k_points.len()),
            ],
            overlap: Vec::with_capacity(k_points.len()),
            pairing_fock: has_exchange.then(|| {
                [
                    Vec::with_capacity(k_points.len()),
                    Vec::with_capacity(k_points.len()),
                ]
            }),
        })
        .collect();
    for k in k_points {
        let s = derivatives.overlap(*k);
        let h = derivatives.core_hamiltonian(*k)?;
        for atom in 0..natoms {
            for axis in 0..3 {
                let x = 3 * atom + axis;
                first[x]
                    .overlap
                    .push(BlochMatrix::from_pair(s[atom][axis].clone()));
                let mut f = BlochMatrix::from_pair(h[atom][axis].clone());
                f.add_scaled(&BlochMatrix::from_pair(bloch(&hartree_fock[x], *k)), 1.0);
                for spin in 0..2 {
                    let mut fs = f.clone();
                    if let Some(xc) = &xc {
                        fs.add_scaled(
                            &BlochMatrix::from_pair(bloch(&xc.vxc_derivative[spin][x], *k)),
                            1.0,
                        );
                    }
                    let mut fp = fs.clone();
                    for (coefficient, terms) in &exchange[spin.min(exchange_spins - 1)] {
                        fs.add_scaled(
                            &BlochMatrix::from_pair(bloch(&terms.source[x], *k)),
                            -*coefficient,
                        );
                        fp.add_scaled(
                            &BlochMatrix::from_pair(bloch(&terms.pairing[x], *k)),
                            -*coefficient,
                        );
                    }
                    first[x].fock[spin].push(fs);
                    if let Some(pairing) = first[x].pairing_fock.as_mut() {
                        pairing[spin].push(fp);
                    }
                }
            }
        }
    }
    Ok(DisplacementSkeleton { skeleton, first })
}

/// The periodic coupled-perturbed Kohn-Sham solver of a converged
/// insulating reference: orbitals, response kernel and preconditioner.
pub(crate) struct ResponseSolver<'a> {
    setup: Setup<'a>,
    diagonal: Vec<f64>,
    /// Fermi-Dirac reference: the full orbital-pair response replaces the
    /// virtual-occupied amplitudes.
    metal: Option<MetalState>,
}

impl<'a> ResponseSolver<'a> {
    pub(crate) fn new(
        cell: &'a Cell,
        functional: &ResolvedFunctional,
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
        reference: &Reference,
    ) -> Result<Self> {
        let nk = reference.k_points.len();
        let k_points = reference.k_points.clone();
        let metal = match options.occupation {
            OccupationScheme::FermiDirac { temperature_kelvin } => {
                Some(MetalState::new(result, options, temperature_kelvin)?)
            }
            OccupationScheme::Insulator => None,
        };
        let mut channels: [Vec<Channel>; 2] = [Vec::with_capacity(nk), Vec::with_capacity(nk)];
        for (spin, list) in channels.iter_mut().enumerate() {
            if metal.is_some() {
                continue;
            }
            let nocc = if spin == 0 {
                result.n_alpha
            } else {
                result.n_beta
            };
            for ik in 0..nk {
                let orbitals = generalized_orbitals(
                    &result.fock[spin][ik],
                    &result.overlap[ik],
                    options.orth_threshold,
                )?;
                list.push(Channel::new(&orbitals, nocc));
            }
        }
        let xc_kernel = crate::periodic::xc_terms::ResponseKernel::new(
            cell,
            functional,
            result,
            options,
            &k_points,
            reference.ao_cutoff,
        )?;
        let two_pi = 2.0 * std::f64::consts::PI;
        let fractional = |k: &crate::math::Vec3| -> [f64; 3] {
            std::array::from_fn(|i| cell.lattice.a[i].dot(*k) / two_pi)
        };
        let partners: Vec<usize> = k_points
            .iter()
            .map(|k| {
                let f = fractional(k);
                k_points
                    .iter()
                    .position(|other| {
                        let g = fractional(other);
                        (0..3).all(|i| {
                            let d = f[i] + g[i];
                            (d - d.round()).abs() < 1e-8
                        })
                    })
                    .ok_or_else(|| {
                        DftError::InvalidInput(
                            "the periodic response needs a time-reversal-symmetric k mesh".into(),
                        )
                    })
            })
            .collect::<Result<_>>()?;
        let setup = Setup {
            cell,
            k_points,
            partners,
            channels,
            xc_kernel,
            screening: reference.screening,
            image_cutoff: reference.image_cutoff,
            exchange: [
                (functional.hybrid.cx, CoulombKernel::Coulomb),
                (
                    functional.hybrid.clr,
                    CoulombKernel::Erf {
                        omega: functional.hybrid.omega,
                    },
                ),
            ]
            .into_iter()
            .filter(|(coefficient, _)| *coefficient != 0.0)
            .collect(),
            exchange_cutoff: reference.exchange_cutoff,
        };
        let mut diagonal = Vec::new();
        for spin in 0..2 {
            for ch in &setup.channels[spin] {
                for ea in &ch.e_virt {
                    for ei in &ch.e_occ {
                        diagonal.push(ea - ei);
                        diagonal.push(ea - ei);
                    }
                }
            }
        }
        Ok(Self {
            setup,
            diagonal,
            metal,
        })
    }

    /// Orbitals of one spin channel at k point `ik`.
    pub(crate) fn channel(&self, spin: usize, ik: usize) -> &Channel {
        &self.setup.channels[spin][ik]
    }

    pub(crate) fn k_points(&self) -> &[crate::math::Vec3] {
        &self.setup.k_points
    }

    /// Solve `(e_a - e_i) U_ai + [C_v^+ G(P) C_o]_ai = rhs_ai` for a
    /// perturbation given by its virtual-occupied source alone (no overlap
    /// change), returning the amplitudes `U(k)` and the response density.
    pub(crate) fn solve_source(
        &self,
        rhs: &[Vec<BlochMatrix>; 2],
        solver: &ResponseSolverConfig,
    ) -> Result<([Vec<BlochMatrix>; 2], [Vec<BlochMatrix>; 2], usize, f64)> {
        let setup = &self.setup;
        let solution = krylov::solve(
            &ResponseOperator { setup },
            &setup.pack(rhs),
            Some(&self.diagonal),
            solver,
        )?;
        let u = setup.unpack(&solution.amplitudes);
        let density = setup.time_reversal(&setup.density(&u));
        Ok((u, density, solution.iterations, solution.relative_residual))
    }

    /// Solve the coupled-perturbed equations of one real, time-reversal
    /// symmetric perturbation.
    pub(crate) fn solve(
        &self,
        first: &FirstOrder,
        solver: &ResponseSolverConfig,
    ) -> Result<Response> {
        if let Some(metal) = &self.metal {
            return self.solve_metal(metal, first, solver);
        }
        let setup = &self.setup;
        let nk = setup.nk();
        // S~ = C^+ S^x C blocks and the occupied-occupied density part.
        let mut s_vo: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        let mut s_oo: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        let mut p_oo: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        for spin in 0..2 {
            for ik in 0..nk {
                let ch = &setup.channels[spin][ik];
                let s = &first.overlap[ik];
                let vo = product(&product(&adjoint(&ch.virtuals), s), &ch.occupied);
                let oo = product(&product(&adjoint(&ch.occupied), s), &ch.occupied);
                let mut poo = product(&product(&ch.occupied, &oo), &adjoint(&ch.occupied));
                poo.scale(-1.0);
                s_vo[spin].push(vo);
                s_oo[spin].push(oo);
                p_oo[spin].push(poo);
            }
        }
        let skeleton_x = &first.fock;
        let g_oo = setup.kernel(&p_oo)?;
        let f_vo = setup.vo_block(skeleton_x);
        let g_vo = setup.vo_block(&g_oo);
        let mut rhs: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        for spin in 0..2 {
            for ik in 0..nk {
                let ch = &setup.channels[spin][ik];
                let mut r = f_vo[spin][ik].clone();
                r.add_scaled(&g_vo[spin][ik], 1.0);
                r.scale(-1.0);
                for a in 0..ch.e_virt.len() {
                    for (i, ei) in ch.e_occ.iter().enumerate() {
                        r.re[(a, i)] += s_vo[spin][ik].re[(a, i)] * ei;
                        r.im[(a, i)] += s_vo[spin][ik].im[(a, i)] * ei;
                    }
                }
                rhs[spin].push(r);
            }
        }
        let solution = krylov::solve(
            &ResponseOperator { setup },
            &setup.pack(&rhs),
            Some(&self.diagonal),
            solver,
        )?;
        let u = setup.unpack(&solution.amplitudes);
        let mut density = setup.density(&u);
        for spin in 0..2 {
            for ik in 0..nk {
                density[spin][ik].add_scaled(&p_oo[spin][ik], 1.0);
            }
        }
        let density = setup.time_reversal(&density);
        // Total first-order Fock and the energy-weighted response density.
        let induced = setup.kernel(&density)?;
        let mut weighted: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        let mut total_fock: [Vec<BlochMatrix>; 2] = [Vec::new(), Vec::new()];
        for spin in 0..2 {
            for ik in 0..nk {
                let ch = &setup.channels[spin][ik];
                let mut f_tot = skeleton_x[spin][ik].clone();
                f_tot.add_scaled(&induced[spin][ik], 1.0);
                total_fock[spin].push(f_tot.clone());
                let mut oo = product(&product(&adjoint(&ch.occupied), &f_tot), &ch.occupied);
                let s = &s_oo[spin][ik];
                for i in 0..ch.e_occ.len() {
                    for j in 0..ch.e_occ.len() {
                        let e = ch.e_occ[i] + ch.e_occ[j];
                        oo.re[(i, j)] -= e * s.re[(i, j)];
                        oo.im[(i, j)] -= e * s.im[(i, j)];
                    }
                }
                let mut ue = u[spin][ik].clone();
                for (i, ei) in ch.e_occ.iter().enumerate() {
                    for a in 0..ch.e_virt.len() {
                        ue.re[(a, i)] *= ei;
                        ue.im[(a, i)] *= ei;
                    }
                }
                let mut w = product(&product(&ch.virtuals, &ue), &adjoint(&ch.occupied));
                let w_t = adjoint(&w);
                w.add_scaled(&w_t, 1.0);
                w.add_scaled(
                    &product(&product(&ch.occupied, &oo), &adjoint(&ch.occupied)),
                    1.0,
                );
                weighted[spin].push(w);
            }
        }
        Ok(Response {
            density,
            weighted,
            amplitudes: u,
            fock: total_fock,
            iterations: solution.iterations,
            residual: solution.relative_residual,
        })
    }
}

/// Symmetrize the raw second-derivative matrix `E_skel + pairing` of the
/// displacements and measure its asymmetry and acoustic sum rule.
pub(crate) fn assemble_force_constants(
    natoms: usize,
    skeleton: Matrix,
    first: &[FirstOrder],
    responses: &[Response],
) -> GammaForceConstants {
    let dim = 3 * natoms;
    let mut raw = skeleton.clone();
    for x in 0..dim {
        for y in 0..dim {
            raw[(x, y)] += pairing(&first[x], &responses[y]);
        }
    }
    let mut hessian = raw.clone();
    let (raw_asymmetry, acoustic_residual) = symmetrize_force_constants(&raw, &mut hessian);
    GammaForceConstants {
        hessian,
        raw,
        raw_asymmetry,
        acoustic_residual,
        skeleton,
        solves: responses
            .iter()
            .map(|r| (r.iterations, r.residual))
            .collect(),
    }
}

fn symmetrize_force_constants(raw: &Matrix, hessian: &mut Matrix) -> (f64, f64) {
    let dim = raw.rows;
    let natoms = dim / 3;
    let mut raw_asymmetry = 0.0_f64;
    for x in 0..dim {
        for y in 0..dim {
            raw_asymmetry = raw_asymmetry.max((raw[(x, y)] - raw[(y, x)]).abs());
            hessian[(x, y)] = 0.5 * (raw[(x, y)] + raw[(y, x)]);
        }
    }
    let mut acoustic_residual = 0.0_f64;
    for x in 0..dim {
        for b in 0..3 {
            let sum: f64 = (0..natoms).map(|atom| hessian[(x, 3 * atom + b)]).sum();
            acoustic_residual = acoustic_residual.max(sum.abs());
        }
    }
    (raw_asymmetry, acoustic_residual)
}

/// Analytic Gamma-point force constants of a converged insulating
/// finite-k SCF (semilocal functionals and exact exchange).
pub fn gamma_force_constants(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
) -> Result<GammaForceConstants> {
    check_response_request(
        cell,
        functional,
        result,
        options,
        "Gamma-point force constants",
        ResponseCapabilities {
            metal: true,
            exchange: true,
            scalar_ecp: true,
        },
    )?;
    let reference = Reference::new(cell, result, options)?;
    let DisplacementSkeleton { skeleton, first } =
        displacement_skeleton(cell, functional, options, &reference)?;
    let response = ResponseSolver::new(cell, functional, result, options, &reference)?;
    let responses: Vec<Response> = first
        .iter()
        .map(|f| response.solve(f, solver))
        .collect::<Result<_>>()?;
    Ok(assemble_force_constants(
        cell.molecule.len(),
        skeleton,
        &first,
        &responses,
    ))
}

/// Orbitals, energies and Fermi-Dirac occupations of one spin channel at one
/// k point of a finite-temperature reference.
pub(crate) struct MetalChannel {
    coefficients: BlochMatrix,
    energies: Vec<f64>,
    occupations: Vec<f64>,
    holes: Vec<f64>,
    kbt: f64,
    /// `df/de = -f (1 - f) / kT`.
    slopes: Vec<f64>,
}

/// Divided differences of the occupation-weighted response in the orbital
/// basis, with their degenerate limits.
struct Lindhard {
    /// `(f_n - f_m)/(e_n - e_m)`, limit `f'`.
    delta: Vec<f64>,
    /// `(e_n f_n - e_m f_m)/(e_n - e_m)`, limit `f + e f'`.
    gamma: Vec<f64>,
    /// `(e_n^2 f_n - e_m^2 f_m)/(e_n - e_m)`, limit `2 e f + e^2 f'`.
    xi: Vec<f64>,
}

impl MetalChannel {
    fn lindhard(&self) -> Lindhard {
        let n = self.energies.len();
        let (e, f, d) = (&self.energies, &self.occupations, &self.slopes);
        let mut out = Lindhard {
            delta: vec![0.0; n * n],
            gamma: vec![0.0; n * n],
            xi: vec![0.0; n * n],
        };
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                let gap = e[i] - e[j];
                if gap == 0.0 {
                    let ei = 0.5 * (e[i] + e[j]);
                    let fi = 0.5 * (f[i] + f[j]);
                    let di = 0.5 * (d[i] + d[j]);
                    out.delta[k] = di;
                    out.gamma[k] = fi + ei * di;
                    out.xi[k] = 2.0 * ei * fi + ei * ei * di;
                } else {
                    let (low, high) = if gap > 0.0 { (j, i) } else { (i, j) };
                    // Exact logistic divided difference, including gaps far
                    // below kT. No band gap is replaced by a cutoff value.
                    let step = gap.abs() / self.kbt;
                    let quotient = if step == 0.0 {
                        1.0 / self.kbt
                    } else {
                        -(-step).exp_m1() / gap.abs()
                    };
                    let delta = -f[low] * self.holes[high] * quotient;
                    let mean_f = 0.5 * (f[i] + f[j]);
                    out.delta[k] = delta;
                    out.gamma[k] = mean_f + 0.5 * (e[i] + e[j]) * delta;
                    out.xi[k] = (e[i] + e[j]) * mean_f + 0.5 * (e[i] * e[i] + e[j] * e[j]) * delta;
                }
            }
        }
        out
    }
}

/// Fermi-Dirac (Mermin) response of a finite-temperature reference. With the
/// orbital-basis first-order Fock `V = C^+ (F^x + G[P^x]) C` and overlap
/// `S~ = C^+ S^x C`, orthonormality and the stationarity of the free energy
/// give
///
/// ```text
/// P~_nm = D_nm V_nm - G_nm S~_nm - delta_nm f'_n mu^x
/// W~_nm = G_nm V_nm - X_nm S~_nm - delta_nm e_n f'_n mu^x
/// mu^x  = sum f'_n (V_nn - e_n S~_nn) / sum f'_n          (fixed electrons per spin)
/// ```
///
/// with the divided differences `D`, `G`, `X` of [`Lindhard`]. The
/// self-consistent `V` solves `V - C^+ G[C P~_lin(V) C^+] C = source` by
/// GMRES.
pub(crate) struct MetalState {
    channels: [Vec<MetalChannel>; 2],
    lindhard: [Vec<Lindhard>; 2],
}

impl MetalState {
    pub(crate) fn new(
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
        temperature_kelvin: f64,
    ) -> Result<Self> {
        let kt = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K * temperature_kelvin;
        let nk = result.k_points.len();
        let mut channels: [Vec<MetalChannel>; 2] = [Vec::with_capacity(nk), Vec::with_capacity(nk)];
        for (spin, list) in channels.iter_mut().enumerate() {
            let mu = result.chemical_potential[spin].ok_or_else(|| {
                DftError::InvalidInput(
                    "a Fermi-Dirac response needs the chemical potential of every spin".into(),
                )
            })?;
            for ik in 0..nk {
                let orbitals = generalized_orbitals(
                    &result.fock[spin][ik],
                    &result.overlap[ik],
                    options.orth_threshold,
                )?;
                let pairs: Vec<_> = orbitals
                    .energies
                    .iter()
                    .map(|e| crate::periodic::occupations::fermi_and_hole((e - mu) / kt))
                    .collect();
                let slopes = pairs.iter().map(|(f, hole)| -f * hole / kt).collect();
                let (occupations, holes) = pairs.into_iter().unzip();
                list.push(MetalChannel {
                    coefficients: orbitals.coefficients,
                    energies: orbitals.energies,
                    occupations,
                    holes,
                    kbt: kt,
                    slopes,
                });
            }
        }
        let lindhard =
            std::array::from_fn(|spin| channels[spin].iter().map(MetalChannel::lindhard).collect());
        Ok(Self { channels, lindhard })
    }

    fn size(&self, spin: usize, ik: usize) -> usize {
        self.channels[spin][ik].energies.len()
    }

    /// Orbital-basis matrices `C^+ M C`.
    fn to_orbitals(&self, m: &[Vec<BlochMatrix>; 2]) -> [Vec<BlochMatrix>; 2] {
        std::array::from_fn(|spin| {
            self.channels[spin]
                .iter()
                .zip(&m[spin])
                .map(|(ch, a)| product(&product(&adjoint(&ch.coefficients), a), &ch.coefficients))
                .collect()
        })
    }

    /// AO matrices `C M~ C^+`.
    fn to_aos(&self, m: &[Vec<BlochMatrix>; 2]) -> [Vec<BlochMatrix>; 2] {
        std::array::from_fn(|spin| {
            self.channels[spin]
                .iter()
                .zip(&m[spin])
                .map(|(ch, a)| product(&product(&ch.coefficients, a), &adjoint(&ch.coefficients)))
                .collect()
        })
    }

    /// Fermi-level shifts `sum f' (V_nn - e S~_nn) / sum f'` per spin.
    fn shifts(&self, v: &[Vec<BlochMatrix>; 2], s: Option<&[Vec<BlochMatrix>; 2]>) -> [f64; 2] {
        std::array::from_fn(|spin| {
            let scale = self.channels[spin]
                .iter()
                .flat_map(|ch| &ch.slopes)
                .fold(0.0_f64, |m, d| m.max(d.abs()));
            if scale == 0.0 {
                return 0.0;
            }
            let (mut num, mut den) = (0.0, 0.0);
            for (ik, ch) in self.channels[spin].iter().enumerate() {
                for (n, d) in ch.slopes.iter().enumerate() {
                    let mut value = v[spin][ik].re[(n, n)];
                    if let Some(s) = s {
                        value -= ch.energies[n] * s[spin][ik].re[(n, n)];
                    }
                    num += (d / scale) * value;
                    den += d / scale;
                }
            }
            num / den
        })
    }

    /// `P~` (or `W~` when `weighted`) from `V` and, when given, `S~`, with the
    /// Fermi-level shifts `mu`.
    fn orbital_response(
        &self,
        v: &[Vec<BlochMatrix>; 2],
        s: Option<&[Vec<BlochMatrix>; 2]>,
        mu: [f64; 2],
        weighted: bool,
    ) -> [Vec<BlochMatrix>; 2] {
        std::array::from_fn(|spin| {
            self.channels[spin]
                .iter()
                .enumerate()
                .map(|(ik, ch)| {
                    let n = ch.energies.len();
                    let l = &self.lindhard[spin][ik];
                    let mut out = BlochMatrix::zeros(n, n);
                    for i in 0..n {
                        for j in 0..n {
                            let k = i * n + j;
                            let (a, b) = if weighted {
                                (l.gamma[k], l.xi[k])
                            } else {
                                (l.delta[k], l.gamma[k])
                            };
                            out.re[(i, j)] = a * v[spin][ik].re[(i, j)];
                            out.im[(i, j)] = a * v[spin][ik].im[(i, j)];
                            if let Some(s) = s {
                                out.re[(i, j)] -= b * s[spin][ik].re[(i, j)];
                                out.im[(i, j)] -= b * s[spin][ik].im[(i, j)];
                            }
                        }
                        let w = if weighted { ch.energies[i] } else { 1.0 };
                        out.re[(i, i)] -= w * ch.slopes[i] * mu[spin];
                    }
                    out
                })
                .collect()
        })
    }
}

struct MetalOperator<'a, 'b> {
    setup: &'a Setup<'b>,
    metal: &'a MetalState,
}

/// Orthonormal real coordinates of a Hermitian matrix. This is the physical
/// space of a static Gamma perturbation, not the independent (k+q,k) space
/// used by a finite-q response. The sqrt(2) factors preserve Frobenius norms.
fn hermitian_unpack(x: &[f64], n: usize) -> BlochMatrix {
    let mut matrix = BlochMatrix::zeros(n, n);
    let mut offset = 0;
    for i in 0..n {
        matrix.re[(i, i)] = x[offset];
        offset += 1;
        for j in i + 1..n {
            let re = x[offset] * std::f64::consts::FRAC_1_SQRT_2;
            let im = x[offset + 1] * std::f64::consts::FRAC_1_SQRT_2;
            offset += 2;
            matrix.re[(i, j)] = re;
            matrix.re[(j, i)] = re;
            matrix.im[(i, j)] = im;
            matrix.im[(j, i)] = -im;
        }
    }
    matrix
}

fn hermitian_pack(matrix: &BlochMatrix, out: &mut Vec<f64>) -> Result<()> {
    let mut defect = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..matrix.rows() {
        for j in 0..matrix.rows() {
            defect = defect
                .max((matrix.re[(i, j)] - matrix.re[(j, i)]).abs())
                .max((matrix.im[(i, j)] + matrix.im[(j, i)]).abs());
            scale = scale
                .max(matrix.re[(i, j)].abs())
                .max(matrix.im[(i, j)].abs());
        }
    }
    if defect > HERMITICITY_TOLERANCE * scale.max(1.0) {
        return Err(DftError::InvalidInput(format!(
            "static Gamma response source is not Hermitian (defect {defect:e}, scale {scale:e})"
        )));
    }
    for i in 0..matrix.rows() {
        out.push(matrix.re[(i, i)]);
        for j in i + 1..matrix.rows() {
            out.push((matrix.re[(i, j)] + matrix.re[(j, i)]) * std::f64::consts::FRAC_1_SQRT_2);
            out.push((matrix.im[(i, j)] - matrix.im[(j, i)]) * std::f64::consts::FRAC_1_SQRT_2);
        }
    }
    Ok(())
}

impl MetalOperator<'_, '_> {
    fn unpack(&self, x: &[f64]) -> [Vec<BlochMatrix>; 2] {
        let mut offset = 0;
        std::array::from_fn(|spin| {
            (0..self.setup.nk())
                .map(|ik| {
                    let n = self.metal.size(spin, ik);
                    let m = hermitian_unpack(&x[offset..offset + n * n], n);
                    offset += n * n;
                    m
                })
                .collect()
        })
    }

    fn pack(&self, blocks: &[Vec<BlochMatrix>; 2]) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        for spin in 0..2 {
            for b in &blocks[spin] {
                hermitian_pack(b, &mut out)?;
            }
        }
        Ok(out)
    }

    /// `C^+ G[C P~ C^+] C` of an orbital-basis density response.
    fn induced(&self, p: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        let density = self.metal.to_aos(p);
        Ok(self.metal.to_orbitals(&self.setup.kernel(&density)?))
    }
}

impl LinearOperator for MetalOperator<'_, '_> {
    fn dimension(&self) -> usize {
        (0..2)
            .map(|spin| {
                (0..self.setup.nk())
                    .map(|ik| self.metal.size(spin, ik).pow(2))
                    .sum::<usize>()
            })
            .sum()
    }

    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }

    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |real| self.apply_real(real))
    }
}

impl MetalOperator<'_, '_> {
    fn apply_real(&self, x: &[f64]) -> Result<Vec<f64>> {
        let v = self.unpack(x);
        let mu = self.metal.shifts(&v, None);
        let p = self.metal.orbital_response(&v, None, mu, false);
        let g = self.induced(&p)?;
        let mut out = v;
        for spin in 0..2 {
            for (o, gi) in out[spin].iter_mut().zip(&g[spin]) {
                o.add_scaled(gi, -1.0);
            }
        }
        self.pack(&out)
    }
}

impl ResponseSolver<'_> {
    /// Fermi-Dirac response of one perturbation (see [`MetalState`]).
    fn solve_metal(
        &self,
        metal: &MetalState,
        first: &FirstOrder,
        solver: &ResponseSolverConfig,
    ) -> Result<Response> {
        let setup = &self.setup;
        let nk = setup.nk();
        let overlap: [Vec<BlochMatrix>; 2] = [first.overlap.clone(), first.overlap.clone()];
        let s = metal.to_orbitals(&overlap);
        let f = metal.to_orbitals(&first.fock);
        let op = MetalOperator { setup, metal };
        // Source: F~^x + C^+ G[C P~_const C^+] C with P~_const from S~ alone.
        let zero: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            (0..nk)
                .map(|ik| BlochMatrix::zeros(metal.size(spin, ik), metal.size(spin, ik)))
                .collect()
        });
        let mu_s = metal.shifts(&zero, Some(&s));
        let p_const = metal.orbital_response(&zero, Some(&s), mu_s, false);
        let g_const = op.induced(&p_const)?;
        let mut rhs = f;
        for spin in 0..2 {
            for (r, g) in rhs[spin].iter_mut().zip(&g_const[spin]) {
                r.add_scaled(g, 1.0);
            }
        }
        let source: Vec<_> = op.pack(&rhs)?.into_iter().map(Complex::from).collect();
        let solution = krylov::solve(&op, &source, None, solver)?;
        let v = op.unpack(&solution.amplitudes.iter().map(|z| z.re).collect::<Vec<_>>());
        let mu = metal.shifts(&v, Some(&s));
        let p = metal.orbital_response(&v, Some(&s), mu, false);
        let w = metal.orbital_response(&v, Some(&s), mu, true);
        let density = setup.time_reversal(&metal.to_aos(&p));
        let weighted = metal.to_aos(&w);
        let induced = setup.kernel(&density)?;
        let fock: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            (0..nk)
                .map(|ik| {
                    let mut f = first.fock[spin][ik].clone();
                    f.add_scaled(&induced[spin][ik], 1.0);
                    f
                })
                .collect()
        });
        Ok(Response {
            density,
            weighted,
            amplitudes: [Vec::new(), Vec::new()],
            fock,
            iterations: solution.iterations,
            residual: solution.relative_residual,
        })
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    mod convergence {
        include!("../../tests/support/convergence.rs");
    }

    #[test]
    fn degenerate_thermal_density_and_pulay_weight_have_independent_four_step_fits() {
        use crate::periodic::occupations::{
            fermi_and_hole, solve_occupations, BOLTZMANN_HARTREE_PER_K,
        };
        let energies = vec![-0.2, -0.2, 0.35];
        let (kbt, mu0) = (0.17, 0.03);
        let pairs: Vec<_> = energies
            .iter()
            .map(|e| fermi_and_hole((e - mu0) / kbt))
            .collect();
        let count = pairs.iter().map(|p| p.0).sum::<f64>();
        let channel = || MetalChannel {
            coefficients: BlochMatrix {
                re: Matrix::identity(3),
                im: Matrix::zeros(3, 3),
            },
            energies: energies.clone(),
            occupations: pairs.iter().map(|p| p.0).collect(),
            holes: pairs.iter().map(|p| p.1).collect(),
            kbt,
            slopes: pairs.iter().map(|p| -p.0 * p.1 / kbt).collect(),
        };
        let channels = [vec![channel()], vec![channel()]];
        let lindhard = std::array::from_fn(|s| vec![channels[s][0].lindhard()]);
        let metal = MetalState { channels, lindhard };
        let v = hermitian_unpack(
            &[0.11, 0.13, -0.07, -0.04, 0.06, -0.08, 0.03, 0.02, 0.09],
            3,
        );
        let s = hermitian_unpack(
            &[0.02, -0.01, 0.03, 0.02, -0.015, -0.025, 0.01, 0.02, 0.04],
            3,
        );
        let perturbation = [vec![v.clone()], vec![v.clone()]];
        let overlap = [vec![s.clone()], vec![s.clone()]];
        let mu = metal.shifts(&perturbation, Some(&overlap));
        let expected: Vec<_> = [false, true]
            .into_iter()
            .map(|weighted| {
                metal.orbital_response(&perturbation, Some(&overlap), mu, weighted)[0][0].clone()
            })
            .collect();
        let oracle = |t: f64| {
            let mut h = v.clone();
            h.scale(t);
            let mut metric = s.clone();
            metric.scale(t);
            for i in 0..3 {
                h.re[(i, i)] += energies[i];
                metric.re[(i, i)] += 1.0;
            }
            let orbitals = generalized_orbitals(&h, &metric, 1e-12).unwrap();
            let occupation = solve_occupations(
                std::slice::from_ref(&orbitals.energies),
                count,
                OccupationScheme::FermiDirac {
                    temperature_kelvin: kbt / BOLTZMANN_HARTREE_PER_K,
                },
            )
            .unwrap();
            [false, true].map(|weighted| {
                let mut c = orbitals.coefficients.clone();
                for n in 0..3 {
                    let weight = occupation.occupations[0][n]
                        * if weighted { orbitals.energies[n] } else { 1.0 };
                    for i in 0..3 {
                        c.re[(i, n)] *= weight;
                        c.im[(i, n)] *= weight;
                    }
                }
                product(&c, &adjoint(&orbitals.coefficients))
            })
        };
        let steps = [0.08, 0.04, 0.02, 0.01];
        let estimates: Vec<_> = steps
            .iter()
            .map(|h| {
                let (a, b) = (oracle(*h), oracle(-*h));
                std::array::from_fn::<_, 2, _>(|w| {
                    let mut m = a[w].clone();
                    m.add_scaled(&b[w], -1.0);
                    m.scale(0.5 / h);
                    m
                })
            })
            .collect();
        for w in 0..2 {
            let error = |a: &BlochMatrix| {
                a.re.max_abs_diff(&expected[w].re)
                    .max(a.im.max_abs_diff(&expected[w].im))
            };
            let errors: Vec<_> = estimates.iter().map(|e| error(&e[w])).collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            let mut rich = estimates[3][w].clone();
            rich.scale(4.0 / 3.0);
            rich.add_scaled(&estimates[2][w], -1.0 / 3.0);
            println!("degenerate metallic weighted={w} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={:e}", fit.exponent, fit.r_squared, error(&rich));
            assert!(
                (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && error(&rich) < 1e-7
            );
        }
    }

    #[test]
    fn hermitian_coordinates_preserve_norm_and_complex_off_diagonals() {
        let x: Vec<_> = (0..16).map(|i| (i as f64 - 6.2) / 7.0).collect();
        let matrix = hermitian_unpack(&x, 4);
        let mut actual = Vec::new();
        hermitian_pack(&matrix, &mut actual).unwrap();
        for (a, b) in actual.iter().zip(&x) {
            assert!((a - b).abs() < 1e-15);
        }
        assert!((frobenius(&matrix, &matrix) - x.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-14);
        let mut bad = matrix;
        bad.im[(1, 0)] += 0.01;
        assert!(hermitian_pack(&bad, &mut Vec::new()).is_err());
    }

    #[test]
    fn symmetric_gamma_operator_is_homogeneous_and_accepts_a_fresh_residual() {
        use crate::{
            basis::data,
            math::Vec3,
            periodic::Lattice,
            system::{Atom, Molecule},
        };
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]);
        let cell = Cell::new(
            molecule,
            Lattice::cubic(9.0),
            &data::load("sto-3g").unwrap(),
            4.0,
        )
        .unwrap();
        let functional = ResolvedFunctional::hartree_fock();
        let options = PeriodicScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-12,
            conv_tol_grad: 1e-12,
            ..Default::default()
        };
        let scf = crate::periodic::scf::run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(scf.converged);
        let reference = Reference::new(&cell, &scf, &options).unwrap();
        let response = ResponseSolver::new(&cell, &functional, &scf, &options, &reference).unwrap();
        let op = ResponseOperator {
            setup: &response.setup,
        };
        let vector: Vec<_> = (0..op.dimension())
            .map(|i| Complex::from((i as f64 + 0.3).sin()))
            .collect();
        let image = op.apply(&vector).unwrap();
        for scale in [1.0, 1e-10, 1e-20, 1e-100] {
            let scaled: Vec<_> = vector.iter().map(|&v| v * scale).collect();
            let actual = op.apply(&scaled).unwrap();
            let defect = actual
                .iter()
                .zip(&image)
                .map(|(a, b)| (*a / scale - *b).abs())
                .fold(0.0_f64, f64::max);
            println!("Gamma operator scale={scale:e} homogeneity={defect:e}");
            assert!(defect < 1e-11);
        }
        let first = displacement_first_order(&cell, &functional, &options, &reference).unwrap();
        for (axis, perturbation) in first.iter().enumerate() {
            let solved = response
                .solve(perturbation, &ResponseSolverConfig::default())
                .unwrap();
            println!(
                "Gamma displacement {axis}: fresh residual={:e}",
                solved.residual
            );
            assert!(solved.residual <= 1e-10);
        }
    }

    #[test]
    fn polar_gamma_nuclear_and_electric_sources_meet_the_requested_fresh_residual() {
        use crate::{
            basis::data,
            math::Vec3,
            periodic::Lattice,
            system::{Atom, Molecule},
        };
        let molecule = Molecule::new(vec![
            Atom {
                z: 3,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 3.0),
            },
        ]);
        let cell = Cell::new(
            molecule,
            Lattice::cubic(9.0),
            &data::load("sto-3g").unwrap(),
            4.0,
        )
        .unwrap();
        let functional = ResolvedFunctional::hartree_fock();
        let options = PeriodicScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-12,
            conv_tol_grad: 1e-12,
            ..Default::default()
        };
        let state = crate::periodic::scf::run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(state.converged);
        let reference = Reference::new(&cell, &state, &options).unwrap();
        let response =
            ResponseSolver::new(&cell, &functional, &state, &options, &reference).unwrap();
        let controls = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let first = displacement_first_order(&cell, &functional, &options, &reference).unwrap();
        for (axis, perturbation) in first.iter().enumerate() {
            let solved = response.solve(perturbation, &controls).unwrap();
            println!(
                "polar Gamma nuclear axis={axis} iterations={} fresh residual={:e}",
                solved.iterations, solved.residual
            );
            assert!(solved.residual <= controls.tolerance);
        }
        let dielectric = crate::periodic::dielectric::dielectric_tensors_with_force_constants(
            &cell,
            &functional,
            &state,
            &options,
            &controls,
            true,
            false,
        )
        .unwrap();
        for (axis, &(iterations, residual)) in dielectric.field_solves.iter().enumerate() {
            println!("polar Gamma electric axis={axis} iterations={iterations} fresh residual={residual:e}");
            assert!(residual <= controls.tolerance);
        }
        for &(iterations, residual) in &dielectric.displacement_solves {
            println!("polar Gamma Born iterations={iterations} fresh residual={residual:e}");
            assert!(residual <= controls.tolerance);
        }
    }
}
