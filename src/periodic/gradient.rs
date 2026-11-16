// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic finite-k nuclear gradients for periodic HF and KS-DFT.

use crate::error::Result;
use crate::integrals::deriv::eri_quartet_center_derivatives;
use crate::integrals::CoulombKernel;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::exchange_quartets::ExchangeCells;
use crate::periodic::integrals::ImageBlock;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::stationary_density::StationaryDensity;
use crate::periodic::Cell;
use crate::xc::ResolvedFunctional;
use rayon::prelude::*;

fn add_component(target: &mut Vec3, axis: usize, value: f64) {
    match axis {
        0 => target.x += value,
        1 => target.y += value,
        _ => target.z += value,
    }
}

/// Preserve legacy scalar screens or retain every nonzero density product.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExchangeDerivativeScreening {
    Legacy,
    Exact,
}

/// Contract the derivative of exactly the same real-space quartet sum used
/// for the periodic exchange matrix, with the quartets selected by the
/// separations of their shell centres (`super::exchange_quartets`). No
/// molecular permutation factor is applicable: translated shells are
/// distinct ordered centres and are enumerated in full here, just as they
/// are in the matrix builder.
///
/// Returns the atomic gradient and the centre virial
/// `sum_X (dE/dX_a) X_b` over every translated shell centre, which is the
/// strain derivative of this purely real-space sum.
pub(crate) fn real_two_electron_gradient(
    cell: &Cell,
    density_out: &[ImageBlock],
    density_inner: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    coefficient: f64,
) -> (Vec<Vec3>, [[f64; 3]; 3]) {
    let cells = ExchangeCells::for_lattice(&cell.basis, &cell.lattice, cutoff);
    exchange_gradient_on_cells(
        &cell.basis,
        cell.molecule.len(),
        &cells,
        (density_out, density_inner),
        kernel,
        coefficient,
        ExchangeDerivativeScreening::Legacy,
    )
}

pub(crate) fn exchange_gradient_on_cells(
    mb: &crate::basis::MolBasis,
    natoms: usize,
    cells: &ExchangeCells,
    densities: (&[ImageBlock], &[ImageBlock]),
    kernel: CoulombKernel,
    coefficient: f64,
    screening: ExchangeDerivativeScreening,
) -> (Vec<Vec3>, [[f64; 3]; 3]) {
    let (density_out, density_inner) = densities;
    let outer_blocks = cells.densities(density_out);
    let inner_blocks = cells.densities(density_inner);
    let partials: Vec<(Vec<Vec3>, [[f64; 3]; 3])> = (0..cells.near.len())
        .into_par_iter()
        .filter_map(|target| {
            let output_density = outer_blocks[target]?;
            let mut gradient = vec![Vec3::zero(); natoms];
            let mut virial = [[0.0; 3]; 3];
            cells.visit(
                mb,
                target,
                |sa, sb| {
                    let (offset_mu, offset_nu) = (mb.shells[sa].sph_off, mb.shells[sb].sph_off);
                    let (n_mu, n_nu) = (
                        crate::basis::n_sph(mb.shells[sa].l),
                        crate::basis::n_sph(mb.shells[sb].l),
                    );
                    let mut output_max = 0.0_f64;
                    for i in 0..n_mu {
                        for j in 0..n_nu {
                            output_max = output_max
                                .max(output_density[(offset_mu + i, offset_nu + j)].abs());
                        }
                    }
                    if screening == ExchangeDerivativeScreening::Legacy {
                        output_max >= 1e-14
                    } else {
                        output_max > 0.0
                    }
                },
                |source| inner_blocks[source].is_some(),
                |q| {
                    let inner_density = inner_blocks[q.source].expect("screened source");
                    let [offset_mu, offset_lambda, offset_sigma, offset_nu] = q.offsets;
                    let [n_mu, n_lambda, n_sigma, n_nu] =
                        q.shells.map(|s| crate::basis::n_sph(s.l));
                    let mut inner_max = 0.0_f64;
                    for k in 0..n_lambda {
                        for l in 0..n_sigma {
                            inner_max = inner_max
                                .max(inner_density[(offset_lambda + k, offset_sigma + l)].abs());
                        }
                    }
                    if inner_max == 0.0
                        || (screening == ExchangeDerivativeScreening::Legacy && inner_max < 1e-14)
                    {
                        return;
                    }
                    let derivatives = eri_quartet_center_derivatives(q.shells, kernel);
                    for i in 0..n_mu {
                        for j in 0..n_nu {
                            let outer = output_density[(offset_mu + i, offset_nu + j)];
                            if outer == 0.0
                                || (screening == ExchangeDerivativeScreening::Legacy
                                    && outer.abs() < 1e-16)
                            {
                                continue;
                            }
                            for k in 0..n_lambda {
                                for l in 0..n_sigma {
                                    let weight = coefficient
                                        * outer
                                        * inner_density[(offset_lambda + k, offset_sigma + l)];
                                    if weight == 0.0
                                        || (screening == ExchangeDerivativeScreening::Legacy
                                            && weight.abs() < 1e-18)
                                    {
                                        continue;
                                    }
                                    for center in 0..4 {
                                        let atom = q.shells[center].atom;
                                        let position = q.shells[center].center;
                                        for axis in 0..3 {
                                            let value =
                                                weight * derivatives[center][axis].get(i, k, l, j);
                                            add_component(&mut gradient[atom], axis, value);
                                            for (b, entry) in virial[axis].iter_mut().enumerate() {
                                                *entry += value * position.get(b);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            );
            Some((gradient, virial))
        })
        .collect();
    let mut gradient = vec![Vec3::zero(); natoms];
    let mut virial = [[0.0; 3]; 3];
    for (partial, partial_virial) in partials {
        add_gradient(&mut gradient, &partial);
        crate::periodic::strain::add_tensor(&mut virial, &partial_virial);
    }
    (gradient, virial)
}

fn add_gradient(target: &mut [Vec3], source: &[Vec3]) {
    for (left, right) in target.iter_mut().zip(source) {
        *left += *right;
    }
}

/// Analytic electronic gradient `dE_elec/dR` of a converged periodic SCF.
pub fn electronic_gradient(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<Vec<Vec3>> {
    result.validate_derivative_plan(cell, functional, options)?;
    electronic_gradient_from_density(
        cell,
        functional,
        &StationaryDensity::from_scf(result),
        options,
    )
}

/// Shared contraction after the caller establishes its own variational stationarity.
pub(crate) fn electronic_gradient_from_density(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &StationaryDensity<'_>,
    options: &PeriodicScfOptions,
) -> Result<Vec<Vec3>> {
    let natoms = cell.molecule.len();
    let nk = result.k_points.len();
    let inv_nk = 1.0 / nk as f64;
    let derivatives = crate::logger::timed("pbc-gradient: one-electron image derivatives", || {
        cell.derivative_images()
    });
    let overlap_derivative: Vec<Vec<[BlochMatrix; 3]>> = result
        .k_points
        .iter()
        .map(|k| {
            derivatives
                .overlap(*k)
                .into_iter()
                .map(|per_atom| per_atom.map(BlochMatrix::from_pair))
                .collect()
        })
        .collect();
    let hcore_derivative: Vec<Vec<[BlochMatrix; 3]>> = result
        .k_points
        .iter()
        .map(|k| {
            derivatives.core_hamiltonian(*k).map(|blocks| {
                blocks
                    .into_iter()
                    .map(|per_atom| per_atom.map(BlochMatrix::from_pair))
                    .collect()
            })
        })
        .collect::<Result<_>>()?;
    let total_k = result.total_density();
    let mut weighted: Vec<BlochMatrix> = Vec::with_capacity(nk);
    for ik in 0..nk {
        let mut sum = BlochMatrix::zeros(cell.nao(), cell.nao());
        for spin in 0..2 {
            sum.add_scaled(&result.energy_weighted_density[spin][ik], 1.0);
        }
        weighted.push(sum);
    }
    let mut gradient = vec![Vec3::zero(); natoms];
    for ik in 0..nk {
        for atom in 0..natoms {
            for axis in 0..3 {
                let value = inv_nk
                    * (total_k[ik].trace_product(&hcore_derivative[ik][atom][axis])
                        - weighted[ik].trace_product(&overlap_derivative[ik][atom][axis]));
                add_component(&mut gradient[atom], axis, value);
            }
        }
    }

    let exchange_cutoff = options.exchange_cutoff.unwrap_or(cell.overlap_cutoff);
    let interaction_cutoff = cell.hartree_real_space_cutoff().max(exchange_cutoff);
    let total_images = cell.density_images_from_k(
        result.k_points,
        &total_k
            .iter()
            .map(|p| (p.re.clone(), p.im.clone()))
            .collect::<Vec<_>>(),
        2.0 * interaction_cutoff,
    )?;
    // Hartree integral derivatives at the SCF density, by the same
    // range-separated screened build the SCF uses for J.
    let hartree = crate::periodic::screened_coulomb::screened_hartree_derivatives(
        &cell.basis,
        &cell.lattice,
        &total_images,
        &cell.hartree_screening(),
        natoms,
    )?;
    add_gradient(&mut gradient, &hartree.gradient);

    if functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0 {
        // In RKS the two spin densities are identical.  The exchange
        // gradient is quadratic in a spin density, so evaluate its costly
        // translated quartet derivatives once and apply the exact factor
        // of two.  UKS still evaluates both independent spin channels.
        let exchange_spins = if result.distinct_spin_channels { 2 } else { 1 };
        let restricted_factor = if result.distinct_spin_channels {
            1.0
        } else {
            2.0
        };
        for spin in 0..exchange_spins {
            let spin_images = crate::periodic::exchange::minimum_image_blocks(
                cell,
                result.k_points,
                cell.density_images_from_k(
                    result.k_points,
                    &result.density[spin]
                        .iter()
                        .map(|p| (p.re.clone(), p.im.clone()))
                        .collect::<Vec<_>>(),
                    2.0 * exchange_cutoff,
                )?,
            );
            if functional.hybrid.cx != 0.0 {
                let (gx, _) = real_two_electron_gradient(
                    cell,
                    &spin_images,
                    &spin_images,
                    exchange_cutoff,
                    CoulombKernel::Coulomb,
                    -0.5 * restricted_factor * functional.hybrid.cx,
                );
                add_gradient(&mut gradient, &gx);
            }
            if functional.hybrid.clr != 0.0 {
                let (gx, _) = real_two_electron_gradient(
                    cell,
                    &spin_images,
                    &spin_images,
                    exchange_cutoff,
                    CoulombKernel::Erf {
                        omega: functional.hybrid.omega,
                    },
                    -0.5 * restricted_factor * functional.hybrid.clr,
                );
                add_gradient(&mut gradient, &gx);
            }
        }
    }

    if !functional.parts.is_empty() || options.skala.is_some() || options.vv10.is_some() {
        let xc = super::xc_gradient::xc_gradient(cell, functional, result, options)?;
        add_gradient(&mut gradient, &xc);
    }
    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::linalg::Matrix;
    use crate::periodic::exchange::exchange_image_blocks;
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};

    fn displaced(molecule: &Molecule, atom: usize, amount: f64) -> Molecule {
        let mut result = molecule.clone();
        result.atoms[atom].position.z += amount;
        result
    }

    fn home_density(matrix: Matrix) -> Vec<ImageBlock> {
        vec![ImageBlock {
            translation: Vec3::zero(),
            matrix,
        }]
    }

    #[test]
    fn parallel_exact_exchange_gradient_matches_fixed_density_energy_difference() {
        // A large cell and a sub-lattice-vector cutoff leave exactly the
        // home quartet. This is the molecular/isolated limit of the same
        // translated periodic K and derivative code, with no SCF response
        // to obscure the integral derivative.
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.55),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.95),
            },
        ]);
        let lattice = Lattice::cubic(30.0);
        let set = data::load("sto-3g").unwrap();
        let cutoff = 8.0;
        let cell = Cell::new(molecule.clone(), lattice.clone(), &set, cutoff).unwrap();
        let density = home_density(Matrix::from_row_major(
            cell.nao(),
            cell.nao(),
            vec![0.60, 0.10, 0.10, 0.40],
        ));
        let (analytic, _) = real_two_electron_gradient(
            &cell,
            &density,
            &density,
            cutoff,
            CoulombKernel::Coulomb,
            -0.5,
        );
        let energy = |geometry: Molecule| {
            let displaced_cell = Cell::new(geometry, lattice.clone(), &set, cutoff).unwrap();
            let blocks = exchange_image_blocks(
                &displaced_cell.basis,
                &lattice,
                &density,
                cutoff,
                CoulombKernel::Coulomb,
            );
            let home = blocks
                .iter()
                .find(|block| block.translation.norm() < 1e-12)
                .unwrap();
            -0.5 * density[0]
                .matrix
                .as_slice()
                .iter()
                .zip(home.matrix.as_slice())
                .map(|(p, k)| p * k)
                .sum::<f64>()
        };
        let step = 2e-4;
        let finite_difference = (energy(displaced(&molecule, 1, step))
            - energy(displaced(&molecule, 1, -step)))
            / (2.0 * step);
        let error = (analytic[1].z - finite_difference).abs();
        assert!(
            error < 2e-7,
            "periodic exact-exchange derivative: analytic={} FD={} error={error:e}",
            analytic[1].z,
            finite_difference
        );
        assert!((analytic[0] + analytic[1]).norm() < 2e-10);

        // The optimized RKS route evaluates one identical spin density with
        // twice the coefficient. Linearity requires bit-level agreement
        // with adding the two old per-spin contractions.
        let (restricted, _) = real_two_electron_gradient(
            &cell,
            &density,
            &density,
            cutoff,
            CoulombKernel::Coulomb,
            -1.0,
        );
        for atom in 0..molecule.len() {
            assert!((restricted[atom] - analytic[atom] * 2.0).norm() < 2e-13);
        }
    }
}
