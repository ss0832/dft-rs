// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static electronic dielectric tensor and Born effective charges of a
//! finite-k insulating reference by density-functional perturbation theory.
//!
//! In the Bloch AO basis `chi_mk(r) = sum_T exp(ik.T) phi_m(r - T)` the
//! position operator acts on occupied states as the covariant k derivative.
//! With `C = (C_o, C_v)` at every k and spin, the Bloch matrices `S(k)`,
//! `H(k)` (Kohn-Sham Fock at fixed density images), the dipole image blocks
//! `D_a(T) = <m(0)| r_a |n(T)>` and `d_a M(k) = sum_T i T_a exp(ik.T) M(T)`:
//!
//! ```text
//! ddk:    U^ka_ai = -[C_v^+ (d_a H - e_i d_a S) C_o]_ai / (e_a - e_i)
//! field:  R_a = C_v^+ (D_a + i d_a S) C_o + i U^ka            (position operator)
//! CPKS:   (e_a - e_i) U^Eb + [C_v^+ G(P^Eb) C_o] = -R_b       (H' = +E.r, E fixed)
//! chi_ab  = -(1/(V Nk)) sum_(k,s) 2 Re Tr[U^Eb^+ R_a],   eps_inf = 1 + 4 pi chi
//! ```
//!
//! The Berry connection of the occupied manifold is
//! `A_a = C_o^+ (S i d_a C_o + (D_a + i d_a S) C_o)` and the electronic
//! polarization `P_a = -(1/(V Nk)) sum Tr A_a`. Its derivative with respect
//! to an atomic displacement `x` (moving basis, overlap response
//! `U^x_oo = -S~^x/2`), after integrating the k derivative of the response by
//! parts over the zone, is
//!
//! ```text
//! d Tr A_a = 2 Re Tr[U^x^+ R_a] - Re Tr[S~^x C_o^+ (D_a + i d_a S) C_o]
//!          + Re Tr[i C_o^+ S^x C_v U^ka] + Re Tr[C_o^+ (D^x_a + i d_a S^x) C_o]
//! Z*_(k,ab) = Z_k delta_ab - Nk^-1 sum_(k,s) d Tr A_a / d u_(k,b)
//! ```
//!
//! where the gauge part of the occupied k derivative cancels between the
//! overlap-response and skeleton terms. The sum over the zone converges to
//! the continuum value as the k mesh is refined, like the Berry phase.

use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::{bloch, concat, shifted, ImageBlock, Operator};
use crate::periodic::phonon::{
    adjoint, assemble_force_constants, check_response_request, displacement_skeleton, product,
    DisplacementSkeleton, Reference, Response, ResponseSolver,
};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::response::krylov::ResponseSolverConfig;
use crate::xc::ResolvedFunctional;

/// Static electronic dielectric response and Born effective charges.
#[derive(Clone, Debug)]
pub struct DielectricTensors {
    /// Clamped-ion (electronic) dielectric tensor `eps_inf`.
    pub epsilon_infinity: [[f64; 3]; 3],
    /// Electronic susceptibility `chi = dP/dE` (dimensionless, Gaussian units).
    pub susceptibility: [[f64; 3]; 3],
    /// Largest `|chi_ab - chi_ba|` before symmetrization.
    pub raw_asymmetry: f64,
    /// Born effective charges `Z*_(k,ab) = V dP_a/du_(k,b)` (elementary
    /// charges), one 3x3 tensor per atom; empty when not requested.
    pub born_charges: Vec<[[f64; 3]; 3]>,
    /// `sum_k Z*_k`, the charge-neutrality (acoustic) sum rule residual.
    pub born_sum_rule: [[f64; 3]; 3],
    /// Largest imaginary part of the zone sums (should vanish).
    pub imaginary_residual: f64,
    /// Largest `|bloch(H images) - F_SCF|` over k and spin.
    pub fock_image_residual: f64,
    /// Krylov iterations and final relative residuals of the field solves.
    pub field_solves: Vec<(usize, f64)>,
    /// Iterations and residuals of the displacement solves.
    pub displacement_solves: Vec<(usize, f64)>,
    /// Gamma-point force constants from the same displacement responses
    /// (present with Born charges on the compatibility wrapper; the explicit
    /// first-order-only request does not build them).
    pub force_constants: Option<crate::periodic::phonon::GammaForceConstants>,
    /// Proper clamped-ion piezoelectric tensor `dP_d/d(eps_ab)`, `[d][3a + b]`
    /// (elementary charges per Bohr^2); `None` when not requested.
    pub piezoelectric: Option<[[f64; 9]; 3]>,
    /// Iterations and residuals of the nine strain solves.
    pub strain_solves: Vec<(usize, f64)>,
}

fn trace(m: &BlochMatrix) -> (f64, f64) {
    let n = m.rows().min(m.cols());
    (0..n).fold((0.0, 0.0), |(r, i), k| (r + m.re[(k, k)], i + m.im[(k, k)]))
}

/// `i M`.
fn times_i(m: &BlochMatrix) -> BlochMatrix {
    let mut re = m.im.clone();
    for v in re.as_mut_slice() {
        *v = -*v;
    }
    BlochMatrix {
        re,
        im: m.re.clone(),
    }
}

/// `sum_T i T_a exp(ik.T) M(T)` as a Bloch matrix.
pub(crate) fn k_derivative(blocks: &[ImageBlock], k: Vec3, axis: usize) -> BlochMatrix {
    let n = blocks.first().map_or(0, |b| b.matrix.rows);
    let mut out = BlochMatrix::zeros(n, n);
    for b in blocks {
        let t = b.translation.get(axis);
        if t == 0.0 {
            continue;
        }
        let phase = k.dot(b.translation);
        let (c, s) = (phase.cos(), phase.sin());
        // i T (c + i s) = -T s + i T c.
        for ((re, im), v) in out
            .re
            .as_mut_slice()
            .iter_mut()
            .zip(out.im.as_mut_slice().iter_mut())
            .zip(b.matrix.as_slice())
        {
            *re -= t * s * v;
            *im += t * c * v;
        }
    }
    out
}

/// Dipole image blocks `<m(0)| r_a |n(T)>` (origin at zero) over the
/// translations of the overlap images.
pub(crate) fn dipole_blocks(cell: &Cell) -> [Vec<ImageBlock>; 3] {
    let mb = &cell.basis;
    let n = mb.nao_sph;
    let natoms = cell.molecule.len();
    let mut out: [Vec<ImageBlock>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for image in
        crate::periodic::integrals::short_range_images(mb, &cell.lattice, cell.overlap_cutoff)
    {
        let t = image.translation;
        let comb = concat(mb, &shifted(mb, t), natoms);
        let d = crate::integrals::one_electron::dipole_matrices(&comb, Vec3::zero());
        for (axis, full) in d.iter().enumerate() {
            let mut m = Matrix::zeros(n, n);
            for &(left, right) in &image.shell_pairs {
                let (left, right) = (&mb.shells[left], &mb.shells[right]);
                for i in left.sph_off..left.sph_off + crate::basis::n_sph(left.l) {
                    for j in right.sph_off..right.sph_off + crate::basis::n_sph(right.l) {
                        m[(i, j)] = full[(i, n + j)];
                    }
                }
            }
            // Preserve the AO shape even for an exactly zero Cartesian moment.
            out[axis].push(ImageBlock {
                translation: t,
                matrix: m,
            });
        }
    }
    out
}

/// Displacement derivatives of the dipole image blocks, `[axis a][3 atom + b]`,
/// with both the home and the shifted copy of every atom moving.
fn dipole_derivative_blocks(cell: &Cell) -> Vec<Vec<Vec<ImageBlock>>> {
    let mb = &cell.basis;
    let n = mb.nao_sph;
    let natoms = cell.molecule.len();
    let mut out: Vec<Vec<Vec<ImageBlock>>> = (0..3)
        .map(|_| (0..3 * natoms).map(|_| Vec::new()).collect())
        .collect();
    for image in
        crate::periodic::integrals::short_range_images(mb, &cell.lattice, cell.overlap_cutoff)
    {
        let t = image.translation;
        let comb = concat(mb, &shifted(mb, t), natoms);
        let d = crate::integrals::deriv_mat::dipole_derivative_matrices(&comb, 2 * natoms);
        for (axis, per_atom) in d.iter().enumerate() {
            for atom in 0..natoms {
                for b in 0..3 {
                    let mut m = Matrix::zeros(n, n);
                    for &(left, right) in &image.shell_pairs {
                        let (left, right) = (&mb.shells[left], &mb.shells[right]);
                        for i in left.sph_off..left.sph_off + crate::basis::n_sph(left.l) {
                            for j in right.sph_off..right.sph_off + crate::basis::n_sph(right.l) {
                                m[(i, j)] = per_atom[atom][b][(i, n + j)]
                                    + per_atom[natoms + atom][b][(i, n + j)];
                            }
                        }
                    }
                    out[axis][3 * atom + b].push(ImageBlock {
                        translation: t,
                        matrix: m,
                    });
                }
            }
        }
    }
    out
}

/// Strain derivatives of the dipole image blocks in the coordinates of the
/// unstrained cell, `[axis d][3a + b]`: every centre `X` moves by `eps X`
/// and the position operator is measured in the unstrained frame,
/// `dD_d / d(eps_ab) = sum_X (dD_d / dX_a) X_b - delta_da D_b`.
fn dipole_strain_blocks(cell: &Cell) -> Vec<Vec<Vec<ImageBlock>>> {
    let mb = &cell.basis;
    let n = mb.nao_sph;
    let natoms = cell.molecule.len();
    let positions: Vec<Vec3> = cell.molecule.atoms.iter().map(|a| a.position).collect();
    let mut out: Vec<Vec<Vec<ImageBlock>>> = (0..3)
        .map(|_| (0..9).map(|_| Vec::new()).collect())
        .collect();
    for image in
        crate::periodic::integrals::short_range_images(mb, &cell.lattice, cell.overlap_cutoff)
    {
        let t = image.translation;
        let comb = concat(mb, &shifted(mb, t), natoms);
        let values = crate::integrals::one_electron::dipole_matrices(&comb, Vec3::zero());
        let d = crate::integrals::deriv_mat::dipole_derivative_matrices(&comb, 2 * natoms);
        for axis in 0..3 {
            for a in 0..3 {
                for b in 0..3 {
                    let mut m = Matrix::zeros(n, n);
                    for &(left, right) in &image.shell_pairs {
                        let (left, right) = (&mb.shells[left], &mb.shells[right]);
                        for i in left.sph_off..left.sph_off + crate::basis::n_sph(left.l) {
                            for j in right.sph_off..right.sph_off + crate::basis::n_sph(right.l) {
                                let mut value = 0.0;
                                for (atom, position) in positions.iter().enumerate() {
                                    value += d[axis][atom][a][(i, n + j)] * position.get(b)
                                        + d[axis][natoms + atom][a][(i, n + j)]
                                            * (position.get(b) + t.get(b));
                                }
                                if axis == a {
                                    value -= values[b][(i, n + j)];
                                }
                                m[(i, j)] = value;
                            }
                        }
                    }
                    out[axis][3 * a + b].push(ImageBlock {
                        translation: t,
                        matrix: m,
                    });
                }
            }
        }
    }
    out
}

/// The Kohn-Sham Fock image blocks of both spins at the reference density.
pub(crate) fn fock_blocks(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    reference: &Reference,
) -> Result<[Vec<ImageBlock>; 2]> {
    let mb = &cell.basis;
    let charges = mb.effective_charges(&cell.molecule);
    let mut core = crate::periodic::integrals::image_blocks(
        mb,
        &cell.lattice,
        cell.overlap_cutoff,
        Operator::Kinetic,
    );
    let add = |list: &mut Vec<ImageBlock>, other: Vec<ImageBlock>| {
        for b in other {
            if let Some(existing) = list
                .iter_mut()
                .find(|x| (x.translation - b.translation).norm() < 1e-8)
            {
                for (u, v) in existing
                    .matrix
                    .as_mut_slice()
                    .iter_mut()
                    .zip(b.matrix.as_slice())
                {
                    *u += v;
                }
            } else {
                list.push(b);
            }
        }
    };
    add(
        &mut core,
        crate::periodic::nuclear::nuclear_image_blocks(
            mb,
            &cell.molecule,
            &charges,
            &cell.lattice,
            cell.overlap_cutoff,
            cell.ewald_params(),
        ),
    );
    // The same retained scalar projectors as in the SCF core operator
    // contribute to its covariant k derivative.
    add(&mut core, cell.ecp_image_blocks().to_vec());
    add(
        &mut core,
        crate::periodic::screened_coulomb::screened_hartree_image_blocks(
            mb,
            &cell.lattice,
            &reference.total_images,
            &reference.screening,
        )?,
    );
    let xc = crate::periodic::xc_terms::potential_blocks(
        cell,
        functional,
        options,
        crate::periodic::xc_terms::Density {
            images: [&reference.spin_images[0], &reference.spin_images[1]],
            ao_cutoff: reference.ao_cutoff,
        },
    )?;
    let mut out = [Vec::new(), Vec::new()];
    for (spin, v) in xc.into_iter().enumerate() {
        let mut f = core.clone();
        add(&mut f, v);
        out[spin] = f;
    }
    Ok(out)
}

/// Adds the exact-exchange image blocks of the reference density to the
/// blocks of [`fock_blocks`], with the minimum-image density blocks and the
/// cutoffs of the self-consistent build, so that the k derivative of the
/// Fock matrix carries the non-local exchange operator.
pub(crate) fn add_exchange_blocks(
    fock: &mut [Vec<ImageBlock>; 2],
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<()> {
    let terms = [
        (
            functional.hybrid.cx,
            crate::integrals::CoulombKernel::Coulomb,
        ),
        (
            functional.hybrid.clr,
            crate::integrals::CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        ),
    ];
    if terms.iter().all(|(coefficient, _)| *coefficient == 0.0) {
        return Ok(());
    }
    let cutoff = options.exchange_cutoff.unwrap_or(cell.overlap_cutoff);
    let spins = if result.unrestricted { 2 } else { 1 };
    for spin in 0..spins {
        let pairs: Vec<_> = result.density[spin]
            .iter()
            .map(|p| (p.re.clone(), p.im.clone()))
            .collect();
        let density = crate::periodic::exchange::minimum_image_blocks(
            cell,
            &result.k_points,
            cell.density_images_from_k(&result.k_points, &pairs, 2.0 * cutoff)?,
        );
        for (coefficient, kernel) in terms {
            if coefficient == 0.0 {
                continue;
            }
            let blocks = crate::periodic::exchange::minimum_image_blocks(
                cell,
                &result.k_points,
                crate::periodic::exchange::exchange_image_blocks(
                    &cell.basis,
                    &cell.lattice,
                    &density,
                    cutoff,
                    kernel,
                ),
            );
            for block in blocks {
                match fock[spin]
                    .iter_mut()
                    .find(|x| (x.translation - block.translation).norm() < 1e-8)
                {
                    Some(existing) => {
                        for (u, v) in existing
                            .matrix
                            .as_mut_slice()
                            .iter_mut()
                            .zip(block.matrix.as_slice())
                        {
                            *u -= coefficient * v;
                        }
                    }
                    None => {
                        let mut matrix = block.matrix;
                        for u in matrix.as_mut_slice() {
                            *u *= -coefficient;
                        }
                        fock[spin].push(ImageBlock {
                            translation: block.translation,
                            matrix,
                        });
                    }
                }
            }
        }
    }
    if spins == 1 {
        fock[1] = fock[0].clone();
    }
    Ok(())
}

/// Static electronic dielectric tensor and, when `born` is set, Born
/// effective charges of a converged insulating finite-k SCF. Exact exchange
/// enters the k derivative of the Fock matrix through its image blocks and
/// the coupled-perturbed equations through the exchange kernel.
pub fn dielectric_tensors(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    born: bool,
) -> Result<DielectricTensors> {
    dielectric_tensors_with_force_constants(cell, functional, result, options, solver, born, born)
}

/// Explicit first-order dielectric/Born calculation, with optional force
/// constants for polar modes. Both paths use exactly the same displacement
/// sources and responses; Born-only requests avoid unrelated Hessian work.
pub fn dielectric_tensors_with_force_constants(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    born: bool,
    build_force_constants: bool,
) -> Result<DielectricTensors> {
    dielectric_impl(
        cell,
        functional,
        result,
        options,
        solver,
        born,
        build_force_constants,
        false,
    )
}

/// The dielectric calculation with its optional parts: Born charges, the
/// force constants of the same displacement responses, and the strain
/// derivative of the polarization.
#[allow(clippy::too_many_arguments)]
fn dielectric_impl(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    born: bool,
    build_force_constants: bool,
    piezo: bool,
) -> Result<DielectricTensors> {
    if build_force_constants && !born {
        return Err(DftError::InvalidInput("force constants in the dielectric request require displacement responses and Born charges".into()));
    }
    check_response_request(
        cell,
        functional,
        result,
        options,
        "dielectric tensors",
        super::phonon::ResponseCapabilities {
            metal: false,
            exchange: true,
            scalar_ecp: true,
        },
    )?;
    let natoms = cell.molecule.len();
    let reference = Reference::new(cell, result, options)?;
    let response = ResponseSolver::new(cell, functional, result, options, &reference)?;
    let k_points = response.k_points().to_vec();
    let nk = k_points.len();
    let volume = cell.lattice.volume().abs();
    let overlap = crate::periodic::integrals::image_blocks(
        &cell.basis,
        &cell.lattice,
        cell.overlap_cutoff,
        Operator::Overlap,
    );
    let mut fock = fock_blocks(cell, functional, options, &reference)?;
    add_exchange_blocks(&mut fock, cell, functional, result, options)?;
    let dipoles = dipole_blocks(cell);
    let mut fock_image_residual = 0.0_f64;
    for spin in 0..2 {
        for (ik, k) in k_points.iter().enumerate() {
            let (re, im) = bloch(&fock[spin], *k);
            fock_image_residual = fock_image_residual
                .max(re.max_abs_diff(&result.fock[spin][ik].re))
                .max(im.max_abs_diff(&result.fock[spin][ik].im));
        }
    }
    // ddk amplitudes and the position operator, [spin][k][axis].
    let mut ddk: [Vec<[BlochMatrix; 3]>; 2] = [Vec::new(), Vec::new()];
    let mut position: [Vec<[BlochMatrix; 3]>; 2] = [Vec::new(), Vec::new()];
    // C_o^+ (D_a + i d_a S) C_o, [spin][k][axis].
    let mut connection: [Vec<[BlochMatrix; 3]>; 2] = [Vec::new(), Vec::new()];
    for spin in 0..2 {
        for (ik, k) in k_points.iter().enumerate() {
            let ch = response.channel(spin, ik);
            let (cv_h, co_h) = (adjoint(&ch.virtuals), adjoint(&ch.occupied));
            let mut u3: Vec<BlochMatrix> = Vec::with_capacity(3);
            let mut r3: Vec<BlochMatrix> = Vec::with_capacity(3);
            let mut a3: Vec<BlochMatrix> = Vec::with_capacity(3);
            for axis in 0..3 {
                let dh = k_derivative(&fock[spin], *k, axis);
                let ds = k_derivative(&overlap, *k, axis);
                let h_vo = product(&product(&cv_h, &dh), &ch.occupied);
                let s_vo = product(&product(&cv_h, &ds), &ch.occupied);
                let mut u = BlochMatrix::zeros(ch.e_virt.len(), ch.e_occ.len());
                for (a, ea) in ch.e_virt.iter().enumerate() {
                    for (i, ei) in ch.e_occ.iter().enumerate() {
                        let denominator = ea - ei;
                        u.re[(a, i)] = -(h_vo.re[(a, i)] - ei * s_vo.re[(a, i)]) / denominator;
                        u.im[(a, i)] = -(h_vo.im[(a, i)] - ei * s_vo.im[(a, i)]) / denominator;
                    }
                }
                // Omega = D_a + i d_a S.
                let mut omega = BlochMatrix::from_pair(bloch(&dipoles[axis], *k));
                omega.add_scaled(&times_i(&ds), 1.0);
                let mut r = product(&product(&cv_h, &omega), &ch.occupied);
                r.add_scaled(&times_i(&u), 1.0);
                a3.push(product(&product(&co_h, &omega), &ch.occupied));
                u3.push(u);
                r3.push(r);
            }
            ddk[spin].push(std::array::from_fn(|a| u3[a].clone()));
            position[spin].push(std::array::from_fn(|a| r3[a].clone()));
            connection[spin].push(std::array::from_fn(|a| a3[a].clone()));
        }
    }
    // Field responses.
    let mut field: Vec<[Vec<BlochMatrix>; 2]> = Vec::with_capacity(3);
    let mut field_solves = Vec::with_capacity(3);
    for b in 0..3 {
        let rhs: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            position[spin]
                .iter()
                .map(|r| {
                    let mut m = r[b].clone();
                    m.scale(-1.0);
                    m
                })
                .collect()
        });
        let (u, _, iterations, residual) = response.solve_source(&rhs, solver)?;
        field_solves.push((iterations, residual));
        field.push(u);
    }
    let mut imaginary_residual = 0.0_f64;
    // chi_ab = -(1/(V Nk)) sum 2 Re Tr[U^Eb^+ R_a].
    let mut raw = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut sum = 0.0;
            for spin in 0..2 {
                for ik in 0..nk {
                    let (re, _) = trace(&product(
                        &adjoint(&field[b][spin][ik]),
                        &position[spin][ik][a],
                    ));
                    sum += 2.0 * re;
                }
            }
            raw[a][b] = -sum / (volume * nk as f64);
        }
    }
    let mut susceptibility = [[0.0; 3]; 3];
    let mut epsilon_infinity = [[0.0; 3]; 3];
    let mut raw_asymmetry = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            raw_asymmetry = raw_asymmetry.max((raw[a][b] - raw[b][a]).abs());
            susceptibility[a][b] = 0.5 * (raw[a][b] + raw[b][a]);
            epsilon_infinity[a][b] =
                if a == b { 1.0 } else { 0.0 } + 4.0 * std::f64::consts::PI * susceptibility[a][b];
        }
    }
    // Born effective charges.
    let mut born_charges = Vec::new();
    let mut born_sum_rule = [[0.0; 3]; 3];
    let mut displacement_solves = Vec::new();
    let mut force_constants = None;
    if born {
        let (skeleton, first) = if build_force_constants {
            let DisplacementSkeleton { skeleton, first } =
                displacement_skeleton(cell, functional, options, &reference)?;
            (Some(skeleton), first)
        } else {
            (
                None,
                super::phonon::displacement_first_order(cell, functional, options, &reference)?,
            )
        };
        let responses: Vec<Response> = first
            .iter()
            .map(|f| response.solve(f, solver))
            .collect::<Result<_>>()?;
        let derivative_overlap = cell.derivative_images().overlap;
        let dipole_derivatives = dipole_derivative_blocks(cell);
        let charges = cell.basis.effective_charges(&cell.molecule);
        for atom in 0..natoms {
            let mut z = [[0.0; 3]; 3];
            for b in 0..3 {
                let x = 3 * atom + b;
                let solved = &responses[x];
                displacement_solves.push((solved.iterations, solved.residual));
                // d_a S^x(k) blocks of this displacement.
                let s_blocks: Vec<ImageBlock> = derivative_overlap
                    .iter()
                    .map(|block| ImageBlock {
                        translation: block.translation,
                        matrix: block.d[atom][b].clone(),
                    })
                    .collect();
                for a in 0..3 {
                    let (mut total, mut imaginary) = (0.0, 0.0);
                    for spin in 0..2 {
                        for (ik, k) in k_points.iter().enumerate() {
                            let ch = response.channel(spin, ik);
                            let co_h = adjoint(&ch.occupied);
                            let sx = &first[x].overlap[ik];
                            let s_oo = product(&product(&co_h, sx), &ch.occupied);
                            // 2 Re Tr[U^x^+ R_a].
                            let (t1, i1) = trace(&product(
                                &adjoint(&solved.amplitudes[spin][ik]),
                                &position[spin][ik][a],
                            ));
                            // - Re Tr[S~ C_o^+ Omega_a C_o].
                            let (t2, i2) = trace(&product(&s_oo, &connection[spin][ik][a]));
                            // Re Tr[i C_o^+ S^x C_v U^ka].
                            let (t3, i3) = trace(&times_i(&product(
                                &product(&product(&co_h, sx), &ch.virtuals),
                                &ddk[spin][ik][a],
                            )));
                            // Re Tr[C_o^+ (D^x_a + i d_a S^x) C_o].
                            let mut omega_x =
                                BlochMatrix::from_pair(bloch(&dipole_derivatives[a][x], *k));
                            omega_x.add_scaled(&times_i(&k_derivative(&s_blocks, *k, a)), 1.0);
                            let (t4, i4) = trace(&product(&product(&co_h, &omega_x), &ch.occupied));
                            total += 2.0 * t1 - t2 + t3 + t4;
                            // The imaginary parts cancel with Tr[S~ U^ka_oo],
                            // U^ka_oo = -C_o^+ d_a S C_o / 2.
                            let ds = k_derivative(&overlap, *k, a);
                            let (gauge, _) = trace(&product(
                                &s_oo,
                                &product(&product(&co_h, &ds), &ch.occupied),
                            ));
                            let _ = (i1, i2);
                            imaginary += i3 + i4 - 0.5 * gauge;
                        }
                    }
                    imaginary_residual = imaginary_residual.max((imaginary / nk as f64).abs());
                    z[a][b] = if a == b { charges[atom] } else { 0.0 } - total / nk as f64;
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    born_sum_rule[a][b] += z[a][b];
                }
            }
            born_charges.push(z);
        }
        if let Some(skeleton) = skeleton {
            force_constants = Some(assemble_force_constants(
                natoms, skeleton, &first, &responses,
            ));
        }
    }
    // Proper clamped-ion piezoelectric tensor: the Born-charge expression
    // with the strain response, the strain matrices and the dipole blocks in
    // the coordinates of the unstrained cell. The reduced coordinates of the
    // ions are fixed, so they do not contribute.
    let mut piezoelectric = None;
    let mut strain_solves = Vec::new();
    if piezo {
        let strain_first =
            crate::periodic::elastic::strain_first_order(cell, functional, options, &reference)?;
        let strain_overlap = crate::periodic::strain_fock::two_centre_strain_blocks(
            &cell.basis,
            &cell.lattice,
            cell.overlap_cutoff,
            Operator::Overlap,
        );
        let strain_dipoles = dipole_strain_blocks(cell);
        let mut tensor = [[0.0_f64; 9]; 3];
        for (c, first_c) in strain_first.iter().enumerate() {
            let solved = response.solve(first_c, solver)?;
            strain_solves.push((solved.iterations, solved.residual));
            let s_blocks: Vec<ImageBlock> = strain_overlap
                .iter()
                .map(|block| ImageBlock {
                    translation: block.translation,
                    matrix: block.d[c].clone(),
                })
                .collect();
            for (a, row) in tensor.iter_mut().enumerate() {
                let mut total = 0.0;
                for spin in 0..2 {
                    for (ik, k) in k_points.iter().enumerate() {
                        let ch = response.channel(spin, ik);
                        let co_h = adjoint(&ch.occupied);
                        let sx = &first_c.overlap[ik];
                        let s_oo = product(&product(&co_h, sx), &ch.occupied);
                        let (t1, _) = trace(&product(
                            &adjoint(&solved.amplitudes[spin][ik]),
                            &position[spin][ik][a],
                        ));
                        let (t2, _) = trace(&product(&s_oo, &connection[spin][ik][a]));
                        let (t3, _) = trace(&times_i(&product(
                            &product(&product(&co_h, sx), &ch.virtuals),
                            &ddk[spin][ik][a],
                        )));
                        let mut omega_x = BlochMatrix::from_pair(bloch(&strain_dipoles[a][c], *k));
                        omega_x.add_scaled(&times_i(&k_derivative(&s_blocks, *k, a)), 1.0);
                        let (t4, _) = trace(&product(&product(&co_h, &omega_x), &ch.occupied));
                        total += 2.0 * t1 - t2 + t3 + t4;
                    }
                }
                row[c] = -total / (nk as f64 * volume);
            }
        }
        piezoelectric = Some(tensor);
    }
    if fock_image_residual > 1e-6 {
        return Err(DftError::InvalidInput(format!(
            "the Fock image blocks reproduce the SCF Fock only to {fock_image_residual:e}; the k derivative needs the same operator"
        )));
    }
    Ok(DielectricTensors {
        epsilon_infinity,
        susceptibility,
        raw_asymmetry,
        born_charges,
        born_sum_rule,
        imaginary_residual,
        fock_image_residual,
        field_solves,
        displacement_solves,
        force_constants,
        piezoelectric,
        strain_solves,
    })
}

/// Proper piezoelectric tensor `e_d,(ab) = dP_d/d(eps_ab)` of an insulating
/// reference, with the polarization measured in the coordinates of the
/// unstrained cell.
pub struct PiezoelectricTensor {
    /// Clamped-ion tensor, `[d][3a + b]` (elementary charges per Bohr^2).
    pub clamped_ion: [[f64; 9]; 3],
    /// Relaxed-ion tensor: the clamped-ion tensor plus
    /// `sum_(A,c) Z*_(A,dc) du_(A,c)/d(eps_ab) / V`; `None` when the ionic
    /// relaxation was not requested.
    pub relaxed_ion: Option<[[f64; 9]; 3]>,
    /// Born effective charges of the same reference.
    pub born_charges: Vec<[[f64; 3]; 3]>,
    /// Cell volume (Bohr^3).
    pub volume: f64,
    /// Largest `|bloch(H images) - F_SCF|` over k and spin.
    pub fock_image_residual: f64,
    /// Iterations and residuals of the nine strain solves.
    pub strain_solves: Vec<(usize, f64)>,
    /// Dispersion energy from the elastic calculation, when relaxation used it.
    pub dispersion_energy: Option<f64>,
}

/// Clamped-ion and, when `relax_ions`, relaxed-ion proper piezoelectric
/// tensors. The relaxed-ion part uses the internal relaxation of the
/// elastic calculation, which needs a dynamically stable reference.
pub fn piezoelectric_tensor(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    relax_ions: bool,
) -> Result<PiezoelectricTensor> {
    piezoelectric_tensor_with_dispersion(
        cell, functional, result, options, solver, relax_ions, None,
    )
}

/// Proper piezoelectric response with the dispersion contribution to
/// internal relaxation. A geometry-only term has no clamped-ion electric
/// vertex at fixed geometry.
pub fn piezoelectric_tensor_with_dispersion(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    relax_ions: bool,
    dispersion: Option<&super::dispersion::PeriodicDispersion>,
) -> Result<PiezoelectricTensor> {
    let d = dielectric_impl(cell, functional, result, options, solver, true, false, true)?;
    let clamped = d.piezoelectric.ok_or_else(|| {
        DftError::InvalidInput("the strain response of the polarization was not built".into())
    })?;
    let natoms = cell.molecule.len();
    let volume = cell.lattice.volume().abs();
    let mut relaxed = clamped;
    let mut dispersion_energy = None;
    if relax_ions && natoms > 1 {
        let elastic = crate::periodic::elastic::elastic_constants_with_dispersion(
            cell, functional, result, options, solver, dispersion,
        )?;
        dispersion_energy = elastic.dispersion_energy;
        for (axis, row) in relaxed.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for atom in 0..natoms {
                    for b in 0..3 {
                        sum += d.born_charges[atom][axis][b]
                            * elastic.internal_relaxation[(3 * atom + b) * 9 + c];
                    }
                }
                *value += sum / volume;
            }
        }
    }
    Ok(PiezoelectricTensor {
        clamped_ion: clamped,
        relaxed_ion: relax_ions.then_some(relaxed),
        born_charges: d.born_charges,
        volume,
        fock_image_residual: d.fock_image_residual,
        strain_solves: d.strain_solves,
        dispersion_energy,
    })
}

/// Polar Gamma-point lattice dynamics from the Born charges, the electronic
/// dielectric tensor and the force constants of one reference.
#[derive(Clone, Debug)]
pub struct PolarModes {
    /// Transverse (analytic) Gamma frequencies, cm^-1 (negative = imaginary).
    pub frequencies_cm1: Vec<f64>,
    /// Mode effective charges `Z~_m,a = sum_(k,b) Z*_(k,ab) e_m(k,b) / sqrt(m_k)`
    /// (elementary charges per square-root amu).
    pub mode_charges: Vec<[f64; 3]>,
    /// Infrared intensities `974.8801 |Z~_m|^2` (km/mol).
    pub ir_intensities_km_mol: Vec<f64>,
    /// Static (relaxed-ion) dielectric tensor
    /// `eps_0 = eps_inf + (4 pi / V) sum_m Z~_m Z~_m^T / w_m^2` over modes
    /// above the acoustic cutoff (atomic units inside the sum).
    pub epsilon_static: [[f64; 3]; 3],
    /// Longitudinal frequencies along each direction from the non-analytic
    /// term `(4 pi / V)(q.Z*_k)(q.Z*_k')/(q.eps_inf.q)/sqrt(m_k m_k')`.
    pub lo_frequencies_cm1: Vec<Vec<f64>>,
    /// Modes excluded from the static sum (acoustic or imaginary).
    pub excluded_modes: usize,
}

/// Polar mode analysis of `tensors` (with Born charges and force constants)
/// for atoms of masses `masses_au` (electron masses) in a cell of `volume`.
pub fn polar_modes(
    tensors: &DielectricTensors,
    masses_au: &[f64],
    volume: f64,
    directions: &[[f64; 3]],
    cutoff_cm1: f64,
) -> Result<PolarModes> {
    let fc = tensors.force_constants.as_ref().ok_or_else(|| {
        DftError::InvalidInput("polar modes need the Born charges and force constants".into())
    })?;
    let natoms = masses_au.len();
    let dim = 3 * natoms;
    let amu = crate::constants::AMU_TO_AU_MASS;
    let weighted = |extra: &dyn Fn(usize, usize) -> f64| -> Result<(Vec<f64>, Matrix)> {
        let mut m = Matrix::zeros(dim, dim);
        for x in 0..dim {
            for y in 0..dim {
                m[(x, y)] = (fc.hessian[(x, y)] + extra(x, y))
                    / (masses_au[x / 3] * masses_au[y / 3]).sqrt();
            }
        }
        crate::linalg::symmetric_eigen(&m)
    };
    let to_cm = |l: f64| l.signum() * l.abs().sqrt() * crate::constants::HARTREE_TO_CM1;
    let (values, vectors) = weighted(&|_, _| 0.0)?;
    let z = &tensors.born_charges;
    let mut mode_charges = Vec::with_capacity(dim);
    let mut ir = Vec::with_capacity(dim);
    let mut epsilon_static = tensors.epsilon_infinity;
    let mut excluded = 0;
    for (m, value) in values.iter().enumerate() {
        // Z~ in e per sqrt(electron mass) for the dielectric sum, and per
        // sqrt(amu) for the reported charges and intensities.
        let charge: [f64; 3] = std::array::from_fn(|a| {
            (0..natoms)
                .map(|k| {
                    (0..3)
                        .map(|b| z[k][a][b] * vectors[(3 * k + b, m)])
                        .sum::<f64>()
                        / masses_au[k].sqrt()
                })
                .sum()
        });
        let per_amu = charge.map(|c| c * amu.sqrt());
        mode_charges.push(per_amu);
        ir.push(974.8801 * per_amu.iter().map(|c| c * c).sum::<f64>());
        if to_cm(*value) <= cutoff_cm1 {
            excluded += 1;
            continue;
        }
        for a in 0..3 {
            for b in 0..3 {
                epsilon_static[a][b] +=
                    4.0 * std::f64::consts::PI / volume * charge[a] * charge[b] / value;
            }
        }
    }
    let mut lo_frequencies_cm1 = Vec::with_capacity(directions.len());
    for d in directions {
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if norm == 0.0 {
            return Err(DftError::InvalidInput(
                "LO-TO directions must be nonzero".into(),
            ));
        }
        let q = d.map(|v| v / norm);
        let denominator: f64 = (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| q[a] * tensors.epsilon_infinity[a][b] * q[b])
            .sum();
        // (q.Z*_k)_b = sum_a q_a Z*_(k,ab).
        let qz: Vec<[f64; 3]> = z
            .iter()
            .map(|zk| std::array::from_fn(|b| (0..3).map(|a| q[a] * zk[a][b]).sum()))
            .collect();
        let (lo, _) = weighted(&|x, y| {
            4.0 * std::f64::consts::PI / volume * qz[x / 3][x % 3] * qz[y / 3][y % 3] / denominator
        })?;
        lo_frequencies_cm1.push(lo.iter().map(|l| to_cm(*l)).collect());
    }
    Ok(PolarModes {
        frequencies_cm1: values.iter().map(|l| to_cm(*l)).collect(),
        mode_charges,
        ir_intensities_km_mol: ir,
        epsilon_static,
        lo_frequencies_cm1,
        excluded_modes: excluded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridOptions;
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    #[test]
    fn exactly_zero_dipoles_preserve_the_ao_operator_shape() {
        let molecule = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::zero(),
        }]);
        let set = crate::basis::data::load("sto-3g").unwrap();
        let cell = Cell::new(molecule, Lattice::cubic(14.0), &set, 12.0).unwrap();
        for images in &dipole_blocks(&cell) {
            let operator = BlochMatrix::from_pair(bloch(images, Vec3::zero()));
            assert_eq!((operator.rows(), operator.cols()), (1, 1));
            assert_eq!(operator.re[(0, 0)], 0.0);
            assert_eq!(operator.im[(0, 0)], 0.0);
        }
        for moments in &dipole_derivative_blocks(&cell) {
            for images in moments {
                let operator = BlochMatrix::from_pair(bloch(images, Vec3::zero()));
                assert_eq!((operator.rows(), operator.cols()), (1, 1));
            }
        }
    }

    fn chain() -> (Cell, ResolvedFunctional, PeriodicScfOptions) {
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.45, 0.1, 0.05),
            },
        ]);
        let lattice = Lattice::new(
            Vec3::new(3.6, 0.0, 0.0),
            Vec3::new(0.3, 7.0, 0.0),
            Vec3::new(0.0, 0.4, 7.0),
        );
        let set = crate::basis::data::load("sto-3g").unwrap();
        let cell = Cell::new(molecule, lattice, &set, 12.0).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
        let options = PeriodicScfOptions {
            k_mesh: [4, 1, 1],
            grid_shell: 1,
            grid: GridOptions {
                level: 2,
                ..GridOptions::default()
            },
            e_tol: 1e-12,
            p_tol: 1e-10,
            max_iter: 200,
            ..PeriodicScfOptions::default()
        };
        (cell, functional, options)
    }

    #[test]
    fn pair_selected_dipole_displacement_derivative_passes_four_step_fit() {
        let (cell, _, _) = chain();
        let set = crate::basis::data::load("sto-3g").unwrap();
        let analytic = dipole_derivative_blocks(&cell);
        let k = Vec3::new(0.21, 0.11, 0.08);
        let steps: [f64; 4] = [0.04, 0.02, 0.01, 0.005];
        for axis in 0..3 {
            let mut errors = Vec::new();
            for h in steps {
                let moments: Vec<_> = [h, -h]
                    .iter()
                    .map(|shift| {
                        let mut molecule = cell.molecule().clone();
                        molecule.atoms[1].position += match axis {
                            0 => Vec3::new(*shift, 0.0, 0.0),
                            1 => Vec3::new(0.0, *shift, 0.0),
                            _ => Vec3::new(0.0, 0.0, *shift),
                        };
                        let displaced = cell.rebuilt_with_molecule(molecule, &set).unwrap();
                        dipole_blocks(&displaced)
                    })
                    .collect();
                let mut error = 0.0_f64;
                for moment in 0..3 {
                    let exact = BlochMatrix::from_pair(bloch(&analytic[moment][3 + axis], k));
                    let mut estimate = BlochMatrix::from_pair(bloch(&moments[0][moment], k));
                    estimate
                        .add_scaled(&BlochMatrix::from_pair(bloch(&moments[1][moment], k)), -1.0);
                    estimate.scale(1.0 / (2.0 * h));
                    error = error
                        .max(estimate.re.max_abs_diff(&exact.re))
                        .max(estimate.im.max_abs_diff(&exact.im));
                }
                errors.push(error);
            }
            assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
            let x: Vec<_> = steps.iter().map(|h| h.ln()).collect();
            let y: Vec<_> = errors.iter().map(|e| e.ln()).collect();
            let mx = x.iter().sum::<f64>() / 4.0;
            let my = y.iter().sum::<f64>() / 4.0;
            let xx = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>();
            let xy = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - mx) * (b - my))
                .sum::<f64>();
            let p = xy / xx;
            let intercept = my - p * mx;
            let sse = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (b - intercept - p * a).powi(2))
                .sum::<f64>();
            let sst = y.iter().map(|v| (v - my).powi(2)).sum::<f64>();
            let r2 = 1.0 - sse / sst;
            println!("pair-selected dipole atom 1 axis {axis}: steps={steps:?} errors={errors:?} p={p:.9} R2={r2:.10}");
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
        }
    }

    /// The analytic k derivative of the occupied projector `C_o C_o^+`
    /// (ddk amplitudes and the parallel-transport overlap part) against
    /// central differences of projectors from the Fock and overlap image
    /// blocks rebuilt at shifted k.
    #[test]
    fn ddk_projector_derivative_passes_a_four_step_fit() {
        let (cell, functional, options) = chain();
        let result = crate::periodic::scf::run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(result.converged);
        let reference = Reference::new(&cell, &result, &options).unwrap();
        let response =
            ResponseSolver::new(&cell, &functional, &result, &options, &reference).unwrap();
        let overlap = crate::periodic::integrals::image_blocks(
            &cell.basis,
            &cell.lattice,
            cell.overlap_cutoff,
            Operator::Overlap,
        );
        let fock = fock_blocks(&cell, &functional, &options, &reference).unwrap();
        let ik = 1;
        let k0 = response.k_points()[ik];
        let mut residual = 0.0_f64;
        for spin in 0..2 {
            let (re, im) = bloch(&fock[spin], k0);
            residual = residual
                .max(re.max_abs_diff(&result.fock[spin][ik].re))
                .max(im.max_abs_diff(&result.fock[spin][ik].im));
        }
        println!("Fock image blocks against the SCF Fock: {residual:e}");
        assert!(residual < 1e-8);
        let projector = |k: Vec3| -> BlochMatrix {
            let f = BlochMatrix::from_pair(bloch(&fock[0], k));
            let s = BlochMatrix::from_pair(bloch(&overlap, k));
            let orbitals =
                crate::dfpt::orbitals::generalized_orbitals(&f, &s, options.orth_threshold)
                    .unwrap();
            let nocc = result.n_alpha;
            let mut co = BlochMatrix::zeros(f.rows(), nocc);
            for r in 0..f.rows() {
                for c in 0..nocc {
                    co.re[(r, c)] = orbitals.coefficients.re[(r, c)];
                    co.im[(r, c)] = orbitals.coefficients.im[(r, c)];
                }
            }
            product(&co, &adjoint(&co))
        };
        let ch = response.channel(0, ik);
        for axis in 0..3 {
            let dh = k_derivative(&fock[0], k0, axis);
            let ds = k_derivative(&overlap, k0, axis);
            let cv_h = adjoint(&ch.virtuals);
            let h_vo = product(&product(&cv_h, &dh), &ch.occupied);
            let s_vo = product(&product(&cv_h, &ds), &ch.occupied);
            let mut u = BlochMatrix::zeros(ch.e_virt.len(), ch.e_occ.len());
            for (a, ea) in ch.e_virt.iter().enumerate() {
                for (i, ei) in ch.e_occ.iter().enumerate() {
                    u.re[(a, i)] = -(h_vo.re[(a, i)] - ei * s_vo.re[(a, i)]) / (ea - ei);
                    u.im[(a, i)] = -(h_vo.im[(a, i)] - ei * s_vo.im[(a, i)]) / (ea - ei);
                }
            }
            let mut analytic = product(&product(&ch.virtuals, &u), &adjoint(&ch.occupied));
            let transpose = adjoint(&analytic);
            analytic.add_scaled(&transpose, 1.0);
            let oo = product(&product(&adjoint(&ch.occupied), &ds), &ch.occupied);
            analytic.add_scaled(
                &product(&product(&ch.occupied, &oo), &adjoint(&ch.occupied)),
                -1.0,
            );
            let steps = [0.04, 0.02, 0.01, 0.005];
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    let mut shift = Vec3::zero();
                    match axis {
                        0 => shift.x = h,
                        1 => shift.y = h,
                        _ => shift.z = h,
                    }
                    let (plus, minus) = (projector(k0 + shift), projector(k0 - shift));
                    let mut worst = 0.0_f64;
                    for (((p, m), a), (pi, (mi, ai))) in plus
                        .re
                        .as_slice()
                        .iter()
                        .zip(minus.re.as_slice())
                        .zip(analytic.re.as_slice())
                        .zip(
                            plus.im
                                .as_slice()
                                .iter()
                                .zip(minus.im.as_slice().iter().zip(analytic.im.as_slice())),
                        )
                    {
                        worst = worst
                            .max(((p - m) / (2.0 * h) - a).abs())
                            .max(((pi - mi) / (2.0 * h) - ai).abs());
                    }
                    worst
                })
                .collect();
            let fit = {
                let (x, y): (Vec<f64>, Vec<f64>) = steps
                    .iter()
                    .zip(&errors)
                    .map(|(h, e)| (h.ln(), e.ln()))
                    .unzip();
                let n = 4.0;
                let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
                let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
                let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
                let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
                (sxy / sxx, sxy * sxy / (sxx * syy))
            };
            println!(
                "ddk projector axis {axis}: errors {errors:?} p = {:.6} R2 = {:.8}",
                fit.0, fit.1
            );
            assert!((fit.0 - 2.0).abs() < 0.05 && fit.1 > 0.9999, "axis {axis}");
        }
    }

    /// The macroscopic dielectric function `1/eps^-1_00(q)` of the finite-q
    /// density response (an independent implementation with all local
    /// fields) converges to the DFPT `eps_inf` along `q` as `q -> 0` with an
    /// error of order `q^2`; the Born charges satisfy charge neutrality.
    ///
    /// Recorded: over q = 0.08..0.01 1/Bohr the errors were 0.7935, 0.2679,
    /// 0.07325, 0.01873 (halving ratios 2.96, 3.66, 3.91; p = 1.808552,
    /// R2 = 0.99742019): the largest q still carries the q^4 term, so the
    /// fit uses 0.02..0.0025 1/Bohr.
    #[test]
    fn dielectric_tensor_is_the_long_wavelength_limit_of_the_finite_q_response() {
        let (cell, functional, mut options) = chain();
        options.k_mesh = [8, 1, 1];
        let result = crate::periodic::scf::run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(result.converged);
        let solver = ResponseSolverConfig::default();
        let d = dielectric_tensors(&cell, &functional, &result, &options, &solver, true).unwrap();
        let first = dielectric_tensors_with_force_constants(
            &cell,
            &functional,
            &result,
            &options,
            &solver,
            true,
            false,
        )
        .unwrap();
        assert!(first.force_constants.is_none() && d.force_constants.is_some());
        let born_difference = first
            .born_charges
            .iter()
            .flatten()
            .flatten()
            .zip(d.born_charges.iter().flatten().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let dielectric_difference = first
            .epsilon_infinity
            .iter()
            .flatten()
            .zip(d.epsilon_infinity.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("first-order-only Born/dielectric versus polar-mode path: {born_difference:.3e} {dielectric_difference:.3e}");
        assert!(born_difference < 1e-12 && dielectric_difference < 1e-12);
        println!(
            "eps_inf {:?}, raw asymmetry {:e}, Fock image residual {:e}, field solves {:?}",
            d.epsilon_infinity, d.raw_asymmetry, d.fock_image_residual, d.field_solves
        );
        println!(
            "Born charges {:?}
sum rule {:?}, imaginary residual {:e}",
            d.born_charges, d.born_sum_rule, d.imaginary_residual
        );
        let largest = d
            .born_charges
            .iter()
            .flatten()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        let neutrality = d
            .born_sum_rule
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            neutrality < 1e-6 * largest.max(1.0),
            "charge neutrality {neutrality:e}"
        );
        let steps = [0.02, 0.01, 0.005, 0.0025];
        let values: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let request = crate::dfpt::density::PeriodicResponseOptions {
                    q: Vec3::new(h, 0.0, 0.0),
                    reciprocal_cutoff: 8.0,
                    ..Default::default()
                };
                let solver = crate::dfpt::density::PeriodicDensitySolver::new(
                    &cell,
                    &functional,
                    &result,
                    &options,
                    &request,
                )
                .unwrap();
                let head = solver
                    .reciprocal_waves()
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.norm2().total_cmp(&b.1.norm2()))
                    .unwrap()
                    .0;
                let report = solver.dielectric(&[head]).unwrap();
                1.0 / report.value.inverse_epsilon[0][0].re
            })
            .collect();
        let errors: Vec<f64> = values
            .iter()
            .map(|v| (v - d.epsilon_infinity[0][0]).abs())
            .collect();
        let (x, y): (Vec<f64>, Vec<f64>) = steps
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
        println!(
            "eps_M(q) {values:?} against eps_xx {}: errors {errors:?} p = {p:.6} R2 = {r2:.8}",
            d.epsilon_infinity[0][0]
        );
        assert!(
            (p - 2.0).abs() < 0.05 && r2 > 0.9999,
            "long-wavelength limit"
        );
    }

    /// A diatomic model with opposite Born charges along x and one spring:
    /// eps_0 = eps_inf + 4 pi Z^2/(V k), w_LO^2 = w_TO^2 + 4 pi Z^2/(V eps_inf mu),
    /// and the Lyddane-Sachs-Teller relation w_LO^2/w_TO^2 = eps_0/eps_inf.
    #[test]
    fn polar_modes_satisfy_the_lyddane_sachs_teller_relation() {
        let (m1, m2, k, z, e, v) = (1837.0 * 7.0, 1837.0, 0.12, 0.9, 3.5, 120.0);
        let mut h = Matrix::zeros(6, 6);
        h[(0, 0)] = k;
        h[(3, 3)] = k;
        h[(0, 3)] = -k;
        h[(3, 0)] = -k;
        let charge = |q: f64| [[q, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let tensors = DielectricTensors {
            epsilon_infinity: [[e, 0.0, 0.0], [0.0, e, 0.0], [0.0, 0.0, e]],
            susceptibility: [[0.0; 3]; 3],
            raw_asymmetry: 0.0,
            born_charges: vec![charge(z), charge(-z)],
            born_sum_rule: [[0.0; 3]; 3],
            imaginary_residual: 0.0,
            fock_image_residual: 0.0,
            field_solves: Vec::new(),
            displacement_solves: Vec::new(),
            force_constants: Some(crate::periodic::phonon::GammaForceConstants {
                hessian: h.clone(),
                raw: h.clone(),
                raw_asymmetry: 0.0,
                acoustic_residual: 0.0,
                skeleton: h,
                solves: Vec::new(),
            }),
            piezoelectric: None,
            strain_solves: Vec::new(),
        };
        let polar = polar_modes(
            &tensors,
            &[m1, m2],
            v,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            1.0,
        )
        .unwrap();
        let mu = m1 * m2 / (m1 + m2);
        let to2 = k / mu;
        let eps0 = e + 4.0 * std::f64::consts::PI * z * z / (v * k);
        assert!((polar.epsilon_static[0][0] - eps0).abs() < 1e-12 * eps0);
        assert!((polar.epsilon_static[1][1] - e).abs() < 1e-14);
        let cm = crate::constants::HARTREE_TO_CM1;
        let to = polar
            .frequencies_cm1
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lo = polar.lo_frequencies_cm1[0]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((to - to2.sqrt() * cm).abs() < 1e-9 * to);
        let lo2 = to2 + 4.0 * std::f64::consts::PI * z * z / (v * e * mu);
        assert!((lo - lo2.sqrt() * cm).abs() < 1e-9 * lo);
        assert!(((lo / to).powi(2) - eps0 / e).abs() < 1e-12);
        // A wave vector normal to the dipoles leaves the modes transverse.
        let lo_y = polar.lo_frequencies_cm1[1]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((lo_y - to).abs() < 1e-9 * to);
        let amu = crate::constants::AMU_TO_AU_MASS;
        let intensity = polar
            .ir_intensities_km_mol
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        assert!((intensity - 974.8801 * z * z / mu * amu).abs() < 1e-9 * intensity);
    }
}

#[cfg(test)]
mod scalar_ecp_tests;
