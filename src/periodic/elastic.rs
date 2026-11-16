// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic second-order elastic constants of a finite-k periodic SCF by the
//! coupled-perturbed Kohn-Sham equations of homogeneous strain.
//!
//! A strain `epsilon` maps atoms and lattice as `x -> (1 + epsilon) x` with
//! the density images `D(T)` labelled by integer translations (see
//! [`super::strain`]). For the nine strain components and the `3N`
//! Gamma-point displacements `u` (applied before the strain,
//! `E((1 + epsilon)(x + u))`), with skeleton second derivatives at fixed
//! density and the first-order matrices `F^x(k)`, `S^x(k)`,
//!
//! ```text
//! W_xy = E_skel,xy + Nk^-1 sum_{k,s} [P^y . F^x - W^y . S^x](k)
//! ```
//!
//! gives the strain Hessian `W_(ab),(cd) = d2E/d(eps_ab) d(eps_cd)`, the
//! internal-strain coupling `Lambda_(ab),u` and the force constants `Phi`.
//! Relaxing the ions, `W_rel = W - Lambda Phi^+ Lambda^T` with the uniform
//! translations projected out. With `tau = sym(dE/d(eps))` and the cell
//! volume `V`, the Lagrangian (energy) elastic constants are
//!
//! ```text
//! C_abcd = sym_(ab) sym_(cd) [W_abcd - delta_ac tau_bd] / V,
//! ```
//!
//! exactly as `W = V C(eta) + delta_ac tau_bd` for `eta = (F^T F - 1)/2`,
//! and the stress-strain (Birch) coefficients are
//! `B_abcd = C_abcd + (delta_ac s_bd + delta_bc s_ad + delta_ad s_bc
//! + delta_bd s_ac - 2 delta_cd s_ab)/2` with the Cauchy stress `s = tau/V`.

use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::Operator;
use crate::periodic::phonon::{
    assemble_force_constants, check_response_request, displacement_skeleton, pairing,
    DisplacementSkeleton, FirstOrder, GammaForceConstants, Reference, Response, ResponseSolver,
};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::strain::StrainSecond;
use crate::periodic::strain_fock::{strain_bloch, StrainBlock};
use crate::periodic::Cell;
use crate::response::krylov::ResponseSolverConfig;
use crate::xc::ResolvedFunctional;

/// Voigt order of the symmetric index pairs: xx, yy, zz, yz, xz, xy.
pub const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

/// Analytic second-order elastic constants and their ingredients.
#[derive(Clone, Debug)]
pub struct ElasticConstants {
    /// `dE/d(eps_ab)` at the reference (Hartree per cell).
    pub strain_derivative: [[f64; 3]; 3],
    /// Cauchy stress `sym(dE/d(eps))/V` (Hartree/Bohr^3).
    pub stress: [[f64; 3]; 3],
    /// Cell volume (Bohr^3).
    pub volume: f64,
    /// Symmetrized clamped-ion strain Hessian `d2E/d(eps_ab) d(eps_cd)`,
    /// `[3a+b][3c+d]` (Hartree per cell).
    pub strain_hessian: Vec<f64>,
    /// Internal-strain coupling `d2E/d(eps_ab) du_(A,c)`, `[3a+b][3A+c]`
    /// (Hartree/Bohr per cell), the average of both response routes.
    pub internal_strain: Vec<f64>,
    /// Ionic relaxation per unit strain `du_(A,c)/d(eps_ab) = -(Phi^+ Lambda^T)`,
    /// `[3A+c][3a+b]` (Bohr).
    pub internal_relaxation: Vec<f64>,
    /// Gamma-point force constants of the same reference.
    pub force_constants: GammaForceConstants,
    /// Clamped-ion Lagrangian elastic constants, Voigt 6x6 (Hartree/Bohr^3).
    pub clamped_ion: [[f64; 6]; 6],
    /// Relaxed-ion Lagrangian elastic constants, Voigt 6x6 (Hartree/Bohr^3).
    pub relaxed_ion: [[f64; 6]; 6],
    /// Clamped-ion stress-strain (Birch) coefficients (Hartree/Bohr^3).
    pub clamped_ion_birch: [[f64; 6]; 6],
    /// Relaxed-ion stress-strain (Birch) coefficients (Hartree/Bohr^3).
    pub relaxed_ion_birch: [[f64; 6]; 6],
    /// Largest `|W_xy - W_yx|` of the raw strain Hessian (Hartree).
    pub raw_asymmetry: f64,
    /// Largest difference between the two routes to the internal-strain
    /// coupling (Hartree/Bohr).
    pub internal_strain_asymmetry: f64,
    /// Largest antisymmetric part `|X_abcd - X_bacd|` of
    /// `X = W - delta_ac tau_bd`, the measured rotation non-invariance of
    /// the lab-frame quadrature (Hartree).
    pub rotation_residual: f64,
    /// Smallest force-constant eigenvalue outside the uniform translations
    /// (Hartree/Bohr^2); `None` for one atom per cell.
    pub smallest_optical_eigenvalue: Option<f64>,
    /// Krylov iterations and final relative residual of the nine strain solves.
    pub strain_solves: Vec<(usize, f64)>,
    /// Pair/three-body dispersion energy added to the electronic reference.
    pub dispersion_energy: Option<f64>,
}

/// Skeleton strain second derivatives `[3a+b][3c+d]` and
/// `[3a+b][3A+c]` of every energy term at fixed density.
fn strain_skeleton(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    reference: &Reference,
) -> Result<StrainSecond> {
    use crate::periodic::strain::{
        nuclear_background_strain_second, nuclear_real_space_strain_second,
        nuclear_reciprocal_strain_second, two_centre_strain_second,
    };
    let natoms = cell.molecule.len();
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let total = &reference.total_images;
    let charges = basis.effective_charges(&cell.molecule);
    let p = cell.ewald_params();
    let mut out = crate::logger::timed("elastic: kinetic", || {
        two_centre_strain_second(
            basis,
            lattice,
            cell.overlap_cutoff,
            Operator::Kinetic,
            total,
            natoms,
        )
    });
    let pulay = crate::logger::timed("elastic: overlap", || {
        two_centre_strain_second(
            basis,
            lattice,
            cell.overlap_cutoff,
            Operator::Overlap,
            &reference.weighted_images,
            natoms,
        )
    });
    for (x, y) in out.strain.iter_mut().zip(&pulay.strain) {
        *x -= y;
    }
    for (x, y) in out.mixed.iter_mut().zip(&pulay.mixed) {
        *x -= y;
    }
    crate::logger::timed("elastic: nuclear attraction", || {
        out.add(&nuclear_real_space_strain_second(
            basis,
            &cell.molecule,
            &charges,
            lattice,
            cell.overlap_cutoff,
            p,
            total,
        ));
        out.add(&nuclear_reciprocal_strain_second(
            basis,
            &cell.molecule,
            &charges,
            lattice,
            cell.overlap_cutoff,
            p,
            total,
        ));
        out.add(&nuclear_background_strain_second(
            basis,
            &cell.molecule,
            &charges,
            lattice,
            cell.overlap_cutoff,
            p,
            total,
        ));
    });
    out.add(&crate::logger::timed("elastic: Hartree", || {
        crate::periodic::screened_coulomb::screened_hartree_strain_second(
            basis,
            lattice,
            total,
            &reference.screening,
            natoms,
        )
    })?);
    let positions: Vec<crate::math::Vec3> =
        cell.molecule.atoms.iter().map(|a| a.position).collect();
    let (strain, mixed) =
        crate::periodic::ewald::ewald_strain_second(lattice, &positions, &charges, p);
    out.add(&StrainSecond { strain, mixed });
    if !basis.ecps.is_empty() {
        out.add(&crate::periodic::integrals::ecp_strain_second(cell, total)?);
    }
    // The charge-dependent shape correction belongs only to the affine
    // skeleton. It does not change the Fock or the strain/displacement block.
    if cell.net_charge().abs() > 1e-14 {
        let monopole = crate::periodic::ewald::monopole_strain_second(lattice, cell.net_charge());
        for (target, value) in out.strain.iter_mut().zip(monopole) {
            *target += value;
        }
    }
    if let Some(xc) = crate::logger::timed("elastic: exchange-correlation", || {
        crate::periodic::xc_terms::strain_second(
            cell,
            functional,
            options,
            crate::periodic::xc_terms::Density {
                images: [&reference.spin_images[0], &reference.spin_images[1]],
                ao_cutoff: reference.ao_cutoff,
            },
        )
    })? {
        out.add(&xc);
    }
    // Exact exchange: strain second derivatives of the truncated image sum.
    let exchange_spins = if reference.unrestricted { 2 } else { 1 };
    for (coefficient, kernel) in hybrid_terms(functional) {
        if coefficient == 0.0 {
            continue;
        }
        for spin in 0..exchange_spins {
            let second = crate::logger::timed("elastic: exact exchange", || {
                crate::periodic::exchange_response::exchange_strain_second(
                    basis,
                    lattice,
                    &reference.exchange_images[spin],
                    reference.exchange_cutoff,
                    kernel,
                    natoms,
                )
            });
            // One spin density of a restricted reference stands for both.
            let factor = -0.5 * coefficient * (3 - exchange_spins) as f64;
            for (target, value) in out.strain.iter_mut().zip(&second.strain) {
                *target += factor * value;
            }
            for (target, value) in out.mixed.iter_mut().zip(&second.mixed) {
                *target += factor * value;
            }
        }
    }
    Ok(out)
}

/// The exact-exchange terms of a functional: coefficient and kernel of the
/// full-range and of the long-range part.
fn hybrid_terms(functional: &ResolvedFunctional) -> [(f64, crate::integrals::CoulombKernel); 2] {
    [
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
    ]
}

/// First-order strain matrices `F^eps(k)` and `S^eps(k)` of the nine
/// strain components.
pub(crate) fn strain_first_order(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    reference: &Reference,
) -> Result<Vec<FirstOrder>> {
    use crate::periodic::strain_fock::{
        nuclear_attraction_strain_blocks, two_centre_strain_blocks,
    };
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let n = cell.nao();
    let charges = basis.effective_charges(&cell.molecule);
    let overlap = two_centre_strain_blocks(basis, lattice, cell.overlap_cutoff, Operator::Overlap);
    let kinetic = two_centre_strain_blocks(basis, lattice, cell.overlap_cutoff, Operator::Kinetic);
    let nuclear = crate::logger::timed("elastic: nuclear strain matrices", || {
        nuclear_attraction_strain_blocks(
            basis,
            &cell.molecule,
            &charges,
            lattice,
            cell.overlap_cutoff,
            cell.ewald_params(),
        )
    });
    let hartree = crate::logger::timed("elastic: Hartree strain matrices", || {
        crate::periodic::screened_coulomb::screened_hartree_strain_fock(
            basis,
            lattice,
            &reference.total_images,
            &reference.screening,
        )
    })?;
    let ecp = crate::periodic::integrals::ecp_strain_blocks(cell)?;
    let xc: Option<[Vec<StrainBlock>; 2]> =
        crate::logger::timed("elastic: XC strain matrices", || {
            crate::periodic::xc_terms::strain_fock(
                cell,
                functional,
                options,
                crate::periodic::xc_terms::Density {
                    images: [&reference.spin_images[0], &reference.spin_images[1]],
                    ao_cutoff: reference.ao_cutoff,
                },
            )
        })?;
    // Exact exchange: first-order exchange matrices of every spin density,
    // on the minimum images of the k mesh like the exchange matrix itself.
    let exchange_spins = if reference.unrestricted { 2 } else { 1 };
    let mut exchange: [Vec<(f64, crate::periodic::exchange_response::ExchangeFirstOrder)>; 2] =
        [Vec::new(), Vec::new()];
    for (coefficient, kernel) in hybrid_terms(functional) {
        if coefficient == 0.0 {
            continue;
        }
        for (spin, terms) in exchange.iter_mut().enumerate().take(exchange_spins) {
            let mut first_order = crate::logger::timed("elastic: exchange strain matrices", || {
                crate::periodic::exchange_response::exchange_strain_first_order(
                    basis,
                    lattice,
                    &reference.exchange_images[spin],
                    reference.exchange_cutoff,
                    kernel,
                )
            });
            for list in first_order
                .source
                .iter_mut()
                .chain(first_order.pairing.iter_mut())
            {
                *list = crate::periodic::exchange::minimum_image_blocks(
                    cell,
                    &reference.k_points,
                    std::mem::take(list),
                );
            }
            terms.push((coefficient, first_order));
        }
    }
    let has_exchange = !exchange[0].is_empty();
    let nk = reference.k_points.len();
    let mut first: Vec<FirstOrder> = (0..9)
        .map(|_| FirstOrder {
            fock: [Vec::with_capacity(nk), Vec::with_capacity(nk)],
            overlap: Vec::with_capacity(nk),
            pairing_fock: has_exchange.then(|| [Vec::with_capacity(nk), Vec::with_capacity(nk)]),
        })
        .collect();
    for k in &reference.k_points {
        let s = strain_bloch(&overlap, *k, n);
        let mut h = strain_bloch(&kinetic, *k, n);
        if !ecp.is_empty() {
            for (target, source) in h.iter_mut().zip(strain_bloch(&ecp, *k, n)) {
                target.add_scaled(&source, 1.0);
            }
        }
        for (target, others) in h.iter_mut().zip(
            strain_bloch(&nuclear, *k, n)
                .into_iter()
                .zip(strain_bloch(&hartree, *k, n)),
        ) {
            target.add_scaled(&others.0, 1.0);
            target.add_scaled(&others.1, 1.0);
        }
        let v: Option<[Vec<BlochMatrix>; 2]> = xc.as_ref().map(|blocks| {
            [
                strain_bloch(&blocks[0], *k, n),
                strain_bloch(&blocks[1], *k, n),
            ]
        });
        for (c, (s_c, h_c)) in s.into_iter().zip(h).enumerate() {
            first[c].overlap.push(s_c);
            for spin in 0..2 {
                let mut f = h_c.clone();
                if let Some(v) = &v {
                    f.add_scaled(&v[spin][c], 1.0);
                }
                let mut fp = f.clone();
                for (coefficient, terms) in &exchange[spin.min(exchange_spins - 1)] {
                    f.add_scaled(
                        &BlochMatrix::from_pair(crate::periodic::integrals::bloch(
                            &terms.source[c],
                            *k,
                        )),
                        -*coefficient,
                    );
                    fp.add_scaled(
                        &BlochMatrix::from_pair(crate::periodic::integrals::bloch(
                            &terms.pairing[c],
                            *k,
                        )),
                        -*coefficient,
                    );
                }
                first[c].fock[spin].push(f);
                if let Some(pairing) = first[c].pairing_fock.as_mut() {
                    pairing[spin].push(fp);
                }
            }
        }
    }
    Ok(first)
}

/// Pseudo-inverse of the force constants with the three uniform
/// translations projected out, and the smallest remaining eigenvalue.
pub(crate) fn translation_free_inverse(
    phi: &Matrix,
    natoms: usize,
) -> Result<(Matrix, Option<f64>)> {
    let dim = 3 * natoms;
    let mut inverse = Matrix::zeros(dim, dim);
    if natoms < 2 {
        return Ok((inverse, None));
    }
    let norm = 1.0 / (natoms as f64).sqrt();
    // P = 1 - T T^T with the normalized translations T.
    let mut projector = Matrix::identity(dim);
    for x in 0..dim {
        for y in 0..dim {
            if x % 3 == y % 3 {
                projector[(x, y)] -= norm * norm;
            }
        }
    }
    let projected = projector.matmul(phi).matmul(&projector);
    let mut symmetric = projected.clone();
    for x in 0..dim {
        for y in 0..dim {
            symmetric[(x, y)] = 0.5 * (projected[(x, y)] + projected[(y, x)]);
        }
    }
    let (values, vectors) = crate::linalg::symmetric_eigen(&symmetric)?;
    let mut smallest = f64::INFINITY;
    for (index, value) in values.iter().enumerate() {
        // Overlap with the translations.
        let mut translation = 0.0;
        for c in 0..3 {
            let t: f64 = (0..natoms)
                .map(|atom| vectors[(3 * atom + c, index)] * norm)
                .sum();
            translation += t * t;
        }
        if translation > 0.5 {
            continue;
        }
        smallest = smallest.min(*value);
        if *value <= 0.0 {
            continue;
        }
        for x in 0..dim {
            for y in 0..dim {
                inverse[(x, y)] += vectors[(x, index)] * vectors[(y, index)] / value;
            }
        }
    }
    Ok((inverse, smallest.is_finite().then_some(smallest)))
}

/// Lagrangian elastic constants and Birch coefficients (Voigt) from a strain
/// Hessian `w` (`[3a+b][3c+d]`), the first strain derivative and the volume.
/// Also returns the rotation residual of `w`.
pub(crate) fn elastic_tensors(
    w: &[f64],
    first: &[[f64; 3]; 3],
    volume: f64,
) -> ([[f64; 6]; 6], [[f64; 6]; 6], f64) {
    let delta = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let tau: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (first[a][b] + first[b][a])));
    let x = |a: usize, b: usize, c: usize, d: usize| {
        w[(3 * a + b) * 9 + 3 * c + d] - delta(a, c) * tau[b][d]
    };
    let mut rotation = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    rotation = rotation.max((x(a, b, c, d) - x(b, a, c, d)).abs());
                }
            }
        }
    }
    let c4 = |a: usize, b: usize, c: usize, d: usize| {
        0.25 * (x(a, b, c, d) + x(b, a, c, d) + x(a, b, d, c) + x(b, a, d, c)) / volume
    };
    let s = |a: usize, b: usize| tau[a][b] / volume;
    let mut lagrangian = [[0.0; 6]; 6];
    let mut birch = [[0.0; 6]; 6];
    for (i, &(a, b)) in VOIGT.iter().enumerate() {
        for (j, &(c, d)) in VOIGT.iter().enumerate() {
            let value = c4(a, b, c, d);
            lagrangian[i][j] = value;
            birch[i][j] = value
                + 0.5
                    * (delta(a, c) * s(b, d)
                        + delta(b, c) * s(a, d)
                        + delta(a, d) * s(b, c)
                        + delta(b, d) * s(a, c)
                        - 2.0 * delta(c, d) * s(a, b));
        }
    }
    (lagrangian, birch, rotation)
}

/// Analytic clamped- and relaxed-ion elastic constants of a converged
/// finite-k SCF, including the affine monopole correction for a uniformly
/// compensated charged cell.
pub fn elastic_constants(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
) -> Result<ElasticConstants> {
    elastic_constants_with_dispersion(cell, functional, result, options, solver, None)
}

/// Elastic constants including a geometry-only dispersion lattice sum.
/// The dispersion curvature enters the bare strain, mixed and nuclear
/// tensors before internal relaxation. It leaves the CPKS operator intact.
pub fn elastic_constants_with_dispersion(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    dispersion: Option<&super::dispersion::PeriodicDispersion>,
) -> Result<ElasticConstants> {
    check_response_request(
        cell,
        functional,
        result,
        options,
        "elastic constants",
        super::phonon::ResponseCapabilities {
            metal: true,
            exchange: true,
            scalar_ecp: true,
        },
    )?;
    let natoms = cell.molecule.len();
    let dim = 3 * natoms;
    let reference = Reference::new(cell, result, options)?;
    let mut stress = crate::periodic::stress::analytic_stress(cell, functional, result, options)?;
    let mut skeleton = strain_skeleton(cell, functional, options, &reference)?;
    let strain_first = strain_first_order(cell, functional, options, &reference)?;
    let DisplacementSkeleton {
        skeleton: mut displacement_skeleton_matrix,
        first: displacement_first,
    } = displacement_skeleton(cell, functional, options, &reference)?;
    let mut dispersion_energy = None;
    if let Some(model) = dispersion {
        use super::dispersion::{
            add_to_stress, dispersion_strain_second, periodic_d3, periodic_dispersion_hessian,
        };
        let atoms = &cell.molecule.atoms;
        let numbers: Vec<_> = atoms.iter().map(|atom| atom.z).collect();
        let positions: Vec<_> = atoms.iter().map(|atom| atom.position).collect();
        let lattice = &cell.lattice;
        let correction = periodic_d3(&numbers, &positions, lattice, model)?;
        add_to_stress(&mut stress, &correction.strain_derivative);
        skeleton.add(&dispersion_strain_second(
            &numbers, &positions, lattice, model,
        ));
        let nuclear = periodic_dispersion_hessian(&numbers, &positions, lattice, model)?;
        for (target, extra) in displacement_skeleton_matrix
            .as_mut_slice()
            .iter_mut()
            .zip(nuclear.as_slice())
        {
            *target += extra;
        }
        dispersion_energy = Some(correction.energy);
    }
    let response = ResponseSolver::new(cell, functional, result, options, &reference)?;
    let strain_responses: Vec<Response> = strain_first
        .iter()
        .enumerate()
        .map(|(c, f)| {
            crate::logger::timed(&format!("elastic: strain response {c}"), || {
                response.solve(f, solver)
            })
        })
        .collect::<Result<_>>()?;
    let displacement_responses: Vec<Response> = displacement_first
        .iter()
        .map(|f| response.solve(f, solver))
        .collect::<Result<_>>()?;
    let force_constants = assemble_force_constants(
        natoms,
        displacement_skeleton_matrix,
        &displacement_first,
        &displacement_responses,
    );

    // Strain Hessian.
    let mut raw = skeleton.strain.clone();
    for x in 0..9 {
        for y in 0..9 {
            raw[x * 9 + y] += pairing(&strain_first[x], &strain_responses[y]);
        }
    }
    let mut strain_hessian = vec![0.0; 81];
    let mut raw_asymmetry = 0.0_f64;
    for x in 0..9 {
        for y in 0..9 {
            raw_asymmetry = raw_asymmetry.max((raw[x * 9 + y] - raw[y * 9 + x]).abs());
            strain_hessian[x * 9 + y] = 0.5 * (raw[x * 9 + y] + raw[y * 9 + x]);
        }
    }
    // Internal-strain coupling by both routes.
    let mut internal_strain = vec![0.0; 9 * dim];
    let mut internal_strain_asymmetry = 0.0_f64;
    for x in 0..9 {
        for u in 0..dim {
            let base = skeleton.mixed[x * dim + u];
            let strain_route = base + pairing(&displacement_first[u], &strain_responses[x]);
            let displacement_route = base + pairing(&strain_first[x], &displacement_responses[u]);
            internal_strain_asymmetry =
                internal_strain_asymmetry.max((strain_route - displacement_route).abs());
            internal_strain[x * dim + u] = 0.5 * (strain_route + displacement_route);
        }
    }
    // Ionic relaxation.
    let (inverse, smallest_optical_eigenvalue) =
        translation_free_inverse(&force_constants.hessian, natoms)?;
    if let Some(value) = smallest_optical_eigenvalue {
        if value <= 0.0 {
            return Err(DftError::InvalidInput(format!(
                "relaxed-ion elastic constants need a dynamically stable reference; the smallest optical force-constant eigenvalue is {value:e} Hartree/Bohr^2"
            )));
        }
    }
    let mut internal_relaxation = vec![0.0; dim * 9];
    for u in 0..dim {
        for x in 0..9 {
            let mut v = 0.0;
            for w in 0..dim {
                v -= inverse[(u, w)] * internal_strain[x * dim + w];
            }
            internal_relaxation[u * 9 + x] = v;
        }
    }
    let mut relaxed = strain_hessian.clone();
    for x in 0..9 {
        for y in 0..9 {
            let mut v = 0.0;
            for u in 0..dim {
                v += internal_strain[x * dim + u] * internal_relaxation[u * 9 + y];
            }
            relaxed[x * 9 + y] += v;
        }
    }
    let volume = cell.lattice.volume().abs();
    let first = stress.strain_derivative;
    let (clamped_ion, clamped_ion_birch, rotation_residual) =
        elastic_tensors(&strain_hessian, &first, volume);
    let (relaxed_ion, relaxed_ion_birch, _) = elastic_tensors(&relaxed, &first, volume);
    Ok(ElasticConstants {
        strain_derivative: first,
        stress: stress.stress,
        volume,
        strain_hessian,
        internal_strain,
        internal_relaxation,
        force_constants,
        clamped_ion,
        relaxed_ion,
        clamped_ion_birch,
        relaxed_ion_birch,
        raw_asymmetry,
        internal_strain_asymmetry,
        rotation_residual,
        smallest_optical_eigenvalue,
        dispersion_energy,
        strain_solves: strain_responses
            .iter()
            .map(|r| (r.iterations, r.residual))
            .collect(),
    })
}

/// Polycrystalline (Voigt, Reuss, Hill) moduli of a 6x6 Voigt elastic
/// tensor, in the tensor's units.
#[derive(Clone, Debug)]
pub struct ElasticModuli {
    /// Symmetrized tensor the moduli were computed from.
    pub tensor: [[f64; 6]; 6],
    /// Compliance `S = C^-1` (engineering shear strains).
    pub compliance: [[f64; 6]; 6],
    /// Eigenvalues of the symmetrized tensor; all positive is Born stability.
    pub eigenvalues: Vec<f64>,
    pub voigt_bulk: f64,
    pub reuss_bulk: f64,
    pub hill_bulk: f64,
    pub voigt_shear: f64,
    pub reuss_shear: f64,
    pub hill_shear: f64,
    /// Hill Young modulus `9KG/(3K + G)`.
    pub young: f64,
    /// Hill Poisson ratio `(3K - 2G)/(2(3K + G))`.
    pub poisson: f64,
    /// Universal anisotropy index `5 G_V/G_R + K_V/K_R - 6` (zero when isotropic).
    pub universal_anisotropy: f64,
}

/// Voigt, Reuss and Hill bounds and averages of a Voigt elastic tensor. The
/// Reuss bounds need an invertible tensor; Born-unstable tensors are
/// reported with their eigenvalues, not rejected.
pub fn elastic_moduli(c: &[[f64; 6]; 6]) -> Result<ElasticModuli> {
    let mut tensor = [[0.0; 6]; 6];
    let mut m = Matrix::zeros(6, 6);
    for i in 0..6 {
        for j in 0..6 {
            tensor[i][j] = 0.5 * (c[i][j] + c[j][i]);
            m[(i, j)] = tensor[i][j];
        }
    }
    let (eigenvalues, _) = crate::linalg::symmetric_eigen(&m)?;
    let s_matrix = crate::linalg::solve_linear_matrix(&m, &Matrix::identity(6))?;
    let compliance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| s_matrix[(i, j)]));
    let (c, s) = (&tensor, &compliance);
    let voigt_bulk = ((c[0][0] + c[1][1] + c[2][2]) + 2.0 * (c[0][1] + c[1][2] + c[0][2])) / 9.0;
    let voigt_shear = ((c[0][0] + c[1][1] + c[2][2]) - (c[0][1] + c[1][2] + c[0][2])
        + 3.0 * (c[3][3] + c[4][4] + c[5][5]))
        / 15.0;
    let reuss_bulk = 1.0 / ((s[0][0] + s[1][1] + s[2][2]) + 2.0 * (s[0][1] + s[1][2] + s[0][2]));
    let reuss_shear = 15.0
        / (4.0 * (s[0][0] + s[1][1] + s[2][2]) - 4.0 * (s[0][1] + s[1][2] + s[0][2])
            + 3.0 * (s[3][3] + s[4][4] + s[5][5]));
    let hill_bulk = 0.5 * (voigt_bulk + reuss_bulk);
    let hill_shear = 0.5 * (voigt_shear + reuss_shear);
    Ok(ElasticModuli {
        tensor,
        compliance,
        eigenvalues,
        voigt_bulk,
        reuss_bulk,
        hill_bulk,
        voigt_shear,
        reuss_shear,
        hill_shear,
        young: 9.0 * hill_bulk * hill_shear / (3.0 * hill_bulk + hill_shear),
        poisson: (3.0 * hill_bulk - 2.0 * hill_shear) / (2.0 * (3.0 * hill_bulk + hill_shear)),
        universal_anisotropy: 5.0 * voigt_shear / reuss_shear + voigt_bulk / reuss_bulk - 6.0,
    })
}

/// Phase velocities of plane elastic waves along `direction` from the
/// Christoffel equation `det(C_ijkl n_j n_l - rho v^2 delta_ik) = 0`, with
/// the Voigt tensor in GPa and the density in kg/m^3: velocities in m/s,
/// ascending, with the polarization vectors as columns. An unstable
/// direction (negative eigenvalue) gives a negative velocity.
pub fn christoffel(
    c_gpa: &[[f64; 6]; 6],
    density: f64,
    direction: [f64; 3],
) -> Result<([f64; 3], [[f64; 3]; 3])> {
    let norm =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    if norm == 0.0 || density <= 0.0 {
        return Err(DftError::InvalidInput(
            "Christoffel velocities need a nonzero direction and a positive density".into(),
        ));
    }
    let n = direction.map(|v| v / norm);
    let voigt = |a: usize, b: usize| {
        VOIGT
            .iter()
            .position(|&(x, y)| (x, y) == (a, b) || (y, x) == (a, b))
            .unwrap()
    };
    let mut gamma = Matrix::zeros(3, 3);
    for i in 0..3 {
        for k in 0..3 {
            let mut v = 0.0;
            for j in 0..3 {
                for l in 0..3 {
                    v += 0.5
                        * (c_gpa[voigt(i, j)][voigt(k, l)] + c_gpa[voigt(k, l)][voigt(i, j)])
                        * n[j]
                        * n[l];
                }
            }
            gamma[(i, k)] = v * 1e9 / density;
        }
    }
    let (values, vectors) = crate::linalg::symmetric_eigen(&gamma)?;
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let velocities = order.map(|i| values[i].signum() * values[i].abs().sqrt());
    let polarizations = std::array::from_fn(|r| order.map(|i| vectors[(r, i)]));
    Ok((velocities, polarizations))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic(lambda: f64, mu: f64) -> [[f64; 6]; 6] {
        let mut c = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lambda + if i == j { 2.0 * mu } else { 0.0 };
            }
            c[3 + i][3 + i] = mu;
        }
        c
    }

    #[test]
    fn isotropic_moduli_and_velocities_match_closed_forms() {
        let (lambda, mu, rho) = (60.0, 30.0, 2650.0);
        let m = elastic_moduli(&isotropic(lambda, mu)).unwrap();
        let k = lambda + 2.0 * mu / 3.0;
        for v in [m.voigt_bulk, m.reuss_bulk, m.hill_bulk] {
            assert!((v - k).abs() < 1e-12 * k);
        }
        for v in [m.voigt_shear, m.reuss_shear, m.hill_shear] {
            assert!((v - mu).abs() < 1e-12 * mu);
        }
        assert!(m.universal_anisotropy.abs() < 1e-12);
        let young = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
        let poisson = lambda / (2.0 * (lambda + mu));
        assert!((m.young - young).abs() < 1e-12 * young && (m.poisson - poisson).abs() < 1e-12);
        let vp = ((lambda + 2.0 * mu) * 1e9 / rho).sqrt();
        let vs = (mu * 1e9 / rho).sqrt();
        for direction in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.3, -1.2, 0.7]] {
            let (v, _) = christoffel(&isotropic(lambda, mu), rho, direction).unwrap();
            assert!(
                (v[0] - vs).abs() < 1e-9 * vs
                    && (v[1] - vs).abs() < 1e-9 * vs
                    && (v[2] - vp).abs() < 1e-9 * vp
            );
        }
    }

    #[test]
    fn cubic_tensor_gives_the_textbook_bounds_and_velocities() {
        let (c11, c12, c44, rho) = (165.7, 63.9, 79.6, 2330.0);
        let mut c = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = if i == j { c11 } else { c12 };
            }
            c[3 + i][3 + i] = c44;
        }
        let m = elastic_moduli(&c).unwrap();
        let k = (c11 + 2.0 * c12) / 3.0;
        assert!((m.voigt_bulk - k).abs() < 1e-12 * k && (m.reuss_bulk - k).abs() < 1e-10 * k);
        let gv = (c11 - c12 + 3.0 * c44) / 5.0;
        let gr = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12));
        assert!((m.voigt_shear - gv).abs() < 1e-12 * gv && (m.reuss_shear - gr).abs() < 1e-10 * gr);
        let (v, _) = christoffel(&c, rho, [1.0, 0.0, 0.0]).unwrap();
        assert!((v[2] - (c11 * 1e9 / rho).sqrt()).abs() < 1e-8 * v[2]);
        assert!((v[0] - (c44 * 1e9 / rho).sqrt()).abs() < 1e-8 * v[0]);
        let (v, _) = christoffel(&c, rho, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            (v[2] - ((c11 + 2.0 * c12 + 4.0 * c44) / 3.0 * 1e9 / rho).sqrt()).abs() < 1e-8 * v[2]
        );
        assert!((v[0] - ((c11 - c12 + c44) / 3.0 * 1e9 / rho).sqrt()).abs() < 1e-8 * v[0]);
    }

    #[test]
    fn lagrangian_and_birch_coefficients_follow_the_stress_terms() {
        // W = V C + delta_ac tau_bd under a hydrostatic stress -P recovers
        // C, and B11 = C11 - P, B12 = C12 + P, B44 = C44 - P.
        let volume = 50.0;
        let pressure = 0.002;
        let c = isotropic(0.02, 0.01);
        let p = pressure * volume;
        let tau = [[-p, 0.0, 0.0], [0.0, -p, 0.0], [0.0, 0.0, -p]];
        let v = |x: usize, y: usize| {
            VOIGT
                .iter()
                .position(|&(a, b)| (a, b) == (x, y) || (b, a) == (x, y))
                .unwrap()
        };
        let mut w = vec![0.0; 81];
        for a in 0..3 {
            for b in 0..3 {
                for cc in 0..3 {
                    for d in 0..3 {
                        w[(3 * a + b) * 9 + 3 * cc + d] =
                            volume * c[v(a, b)][v(cc, d)] + if a == cc { tau[b][d] } else { 0.0 };
                    }
                }
            }
        }
        let (lagrangian, birch, rotation) = elastic_tensors(&w, &tau, volume);
        assert!(rotation < 1e-14);
        for i in 0..6 {
            for j in 0..6 {
                assert!((lagrangian[i][j] - c[i][j]).abs() < 1e-15);
            }
        }
        assert!((birch[0][0] - (c[0][0] - pressure)).abs() < 1e-15);
        assert!((birch[0][1] - (c[0][1] + pressure)).abs() < 1e-15);
        assert!((birch[3][3] - (c[3][3] - pressure)).abs() < 1e-15);
    }
}
