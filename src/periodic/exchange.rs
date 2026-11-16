// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic exact exchange in a localized Gaussian basis.
//!
//! For real-space density blocks `D(T)`, the exchange image matrix is
//!
//! ```text
//! K_mu,nu(T) = sum_C,D sum_lambda,sigma D_lambda,sigma(D-C)
//!                (mu(0) lambda(C) | sigma(D) nu(T)).
//! ```
//!
//! Unlike the Hartree term, each charge distribution in an exchange
//! integral is an AO product. Gaussian overlap and the density matrix of an
//! insulator both decay exponentially, so this real-space sum is absolutely
//! convergent and can use the full Coulomb kernel directly. This also avoids
//! inventing a convention for the isolated `q=0` exchange singularity: the
//! result is the well-defined localized real-space limit, controlled by the
//! same explicit cutoff as the density blocks. The quartets are selected by
//! the separations of their shell centres (`super::exchange_quartets`), so
//! a supercell has the exchange of its primitive cell.

use crate::basis::MolBasis;
use crate::integrals::eri::eri_quartet;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::exchange_quartets::ExchangeCells;
use crate::periodic::integrals::ImageBlock;
use crate::periodic::Lattice;
use rayon::prelude::*;

/// Born-von Karman periods of a uniform k mesh: the smallest `N_i` for which
/// `exp(i k . N_i a_i)` is the same phase at every k point. `None` when a
/// lattice row has no such period up to the number of k points.
pub(crate) fn born_von_karman_periods(lattice: &Lattice, k_points: &[Vec3]) -> Option<[usize; 3]> {
    let nk = k_points.len();
    let mut periods = [0usize; 3];
    for (axis, period) in periods.iter_mut().enumerate() {
        *period = (1..=nk).find(|n| {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for k in k_points {
                let phase = *n as f64 * k.dot(lattice.a[axis]);
                re += phase.cos();
                im += phase.sin();
            }
            (re * re + im * im).sqrt() > (1.0 - 1e-8) * nk as f64
        })?;
    }
    Some(periods)
}

/// Weight of the translation `t` among its Born-von Karman replicas
/// `t + sum_i n_i N_i a_i`: one over the number of replicas of the same,
/// smallest length, and zero when a shorter replica exists.
pub(crate) fn minimum_image_weight(lattice: &Lattice, periods: [usize; 3], t: Vec3) -> f64 {
    let length = t.norm();
    let tolerance = 1e-9 * length.max(1.0);
    let mut equal = 1usize;
    for n0 in -2i64..=2 {
        for n1 in -2i64..=2 {
            for n2 in -2i64..=2 {
                if n0 == 0 && n1 == 0 && n2 == 0 {
                    continue;
                }
                let replica = t
                    + lattice.a[0] * (n0 * periods[0] as i64) as f64
                    + lattice.a[1] * (n1 * periods[1] as i64) as f64
                    + lattice.a[2] * (n2 * periods[2] as i64) as f64;
                let other = replica.norm();
                if other < length - tolerance {
                    return 0.0;
                }
                if other <= length + tolerance {
                    equal += 1;
                }
            }
        }
    }
    1.0 / equal as f64
}

/// Image blocks of a finite k mesh restricted to minimum images.
///
/// The inverse Bloch transform of a k mesh is periodic in the Born-von
/// Karman supercell, so the density between two centres and between their
/// replicas carries the same block. The exchange sums use every pair of
/// centres once: the element `(mu, nu)` of the block of translation `T` is
/// scaled by the [`minimum_image_weight`] of the separation
/// `R_nu + T - R_mu` of its two atoms, and blocks without a surviving
/// element are dropped. Deciding by the separation of the atoms rather than
/// by the translation makes a supercell and its primitive cell use the same
/// pairs. The same projection is applied to the exchange matrix, which
/// makes the exchange energy `sum w D . K[w D]` independent of a cutoff that
/// exceeds the supercell. Converge the k mesh until the supercell exceeds
/// the range of the density matrix.
pub(crate) fn minimum_image_blocks(
    cell: &crate::periodic::Cell,
    k_points: &[Vec3],
    blocks: Vec<ImageBlock>,
) -> Vec<ImageBlock> {
    let lattice = &cell.lattice;
    let Some(periods) = born_von_karman_periods(lattice, k_points) else {
        return blocks;
    };
    let centres: Vec<Vec3> = cell.molecule.atoms.iter().map(|a| a.position).collect();
    let natoms = centres.len();
    let mut atom_of = vec![0usize; cell.basis.nao_sph];
    for shell in &cell.basis.shells {
        for i in 0..crate::basis::n_sph(shell.l) {
            atom_of[shell.sph_off + i] = shell.atom;
        }
    }
    blocks
        .into_iter()
        .filter_map(|mut block| {
            let mut weights = vec![0.0_f64; natoms * natoms];
            for a in 0..natoms {
                for b in 0..natoms {
                    weights[a * natoms + b] = minimum_image_weight(
                        lattice,
                        periods,
                        centres[b] + block.translation - centres[a],
                    );
                }
            }
            if weights.iter().all(|w| *w == 0.0) {
                return None;
            }
            for i in 0..block.matrix.rows {
                for j in 0..block.matrix.cols {
                    block.matrix[(i, j)] *= weights[atom_of[i] * natoms + atom_of[j]];
                }
            }
            Some(block)
        })
        .collect()
}

/// Exchange image blocks for a periodic density.
pub fn exchange_image_blocks(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
) -> Vec<ImageBlock> {
    exchange_blocks_on_cells(
        mb,
        density,
        &ExchangeCells::for_lattice(mb, lattice, cutoff),
        kernel,
    )
}

/// Localized exchange with translations only in the physical periodic subspace.
pub fn dimensional_exchange_image_blocks(
    mb: &MolBasis,
    geometry: &crate::periodic::PeriodicGeometry,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
) -> crate::Result<Vec<ImageBlock>> {
    dimensional_exchange_component_images(mb, geometry, density, cutoff, kernel, false)
}

/// Real or imaginary density component of the variational localized exchange
/// map. Imaginary Hermitian densities obey D(T)=-D(-T)^transpose; projecting
/// them onto the real-density parity would erase orbital-current exchange.
pub(crate) fn dimensional_exchange_component_images(
    mb: &MolBasis,
    geometry: &crate::periodic::PeriodicGeometry,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    imaginary: bool,
) -> crate::Result<Vec<ImageBlock>> {
    if !cutoff.is_finite()
        || cutoff <= 0.0
        || mb.nao_sph == 0
        || mb.shells.is_empty()
        || density.iter().any(|block| {
            !block.translation.norm().is_finite()
                || block.matrix.rows != mb.nao_sph
                || block.matrix.cols != mb.nao_sph
                || block.matrix.as_slice().iter().any(|v| !v.is_finite())
        })
    {
        return Err(crate::DftError::InvalidInput(
            "dimensional exchange requires a positive finite cutoff, AOs and finite \
            dimensioned real image densities"
                .into(),
        ));
    }
    let cells = ExchangeCells::for_geometry(mb, geometry, cutoff)?;
    let n = mb.nao_sph;
    let ntarget = cells.near.len();
    let densities = cells.densities(density);
    // A bounded three-image energy is a quadratic density functional.
    // Differentiating its two density factors gives the direct map and its
    // adjoint; the inner image D-C of a selected quartet is in `near`.
    let contributions = (0..ntarget)
        .into_par_iter()
        .map(|target| {
            let mut result: Vec<Matrix> = (0..ntarget).map(|_| Matrix::zeros(n, n)).collect();
            let outer = densities[target];
            cells.visit(
                mb,
                target,
                |_, _| true,
                |source| outer.is_some() || densities[source].is_some(),
                |q| {
                    let [mu, lambda, sigma, nu] = q.shells;
                    let [offset_mu, offset_lambda, offset_sigma, offset_nu] = q.offsets;
                    let source = densities[q.source];
                    let quartet = eri_quartet(mu, lambda, sigma, nu, kernel);
                    for i in 0..quartet.na {
                        for j in 0..quartet.nd {
                            let p_outer = outer.map_or(0.0, |p| p[(offset_mu + i, offset_nu + j)]);
                            for k in 0..quartet.nb {
                                for l in 0..quartet.nc {
                                    let value = 0.5 * quartet.get(i, k, l, j);
                                    let p_source = source
                                        .map_or(0.0, |p| p[(offset_lambda + k, offset_sigma + l)]);
                                    result[target][(offset_mu + i, offset_nu + j)] +=
                                        value * p_source;
                                    result[q.source][(offset_lambda + k, offset_sigma + l)] +=
                                        value * p_outer;
                                }
                            }
                        }
                    }
                },
            );
            result
        })
        .collect::<Vec<_>>();
    let targets = cells.near.clone();
    let mut matrices: Vec<Matrix> = targets.iter().map(|_| Matrix::zeros(n, n)).collect();
    for contribution in contributions {
        for (target, source) in matrices.iter_mut().zip(contribution) {
            for (a, b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *a += b;
            }
        }
    }
    // Project onto the Hermitian image parity of this real/imaginary component.
    let parity = if imaginary { -1.0 } else { 1.0 };
    for left in 0..targets.len() {
        let right = targets
            .iter()
            .position(|r| (*r + targets[left]).norm() < 1e-8)
            .expect("opposite image");
        if right < left {
            continue;
        }
        let a = matrices[left].clone();
        let b = matrices[right].clone();
        for i in 0..n {
            for j in 0..n {
                let value = 0.5 * (a[(i, j)] + parity * b[(j, i)]);
                matrices[left][(i, j)] = value;
                matrices[right][(j, i)] = parity * value;
            }
        }
    }
    Ok(targets
        .into_iter()
        .zip(matrices)
        .map(|(translation, matrix)| ImageBlock {
            translation,
            matrix,
        })
        .collect())
}

fn exchange_blocks_on_cells(
    mb: &MolBasis,
    density: &[ImageBlock],
    cells: &ExchangeCells,
    kernel: CoulombKernel,
) -> Vec<ImageBlock> {
    let n = mb.nao_sph;
    let densities = cells.densities(density);
    let mut blocks: Vec<(Vec3, Matrix)> = (0..cells.near.len())
        .into_par_iter()
        .map(|target| {
            let mut exchange = Matrix::zeros(n, n);
            cells.visit(
                mb,
                target,
                |_, _| true,
                |source| densities[source].is_some(),
                |q| {
                    let [mu, lambda, sigma, nu] = q.shells;
                    let [offset_mu, offset_lambda, offset_sigma, offset_nu] = q.offsets;
                    let density_block = densities[q.source].expect("screened source");
                    let (n_lambda, n_sigma) =
                        (crate::basis::n_sph(lambda.l), crate::basis::n_sph(sigma.l));
                    let mut density_max = 0.0_f64;
                    for k in 0..n_lambda {
                        for l in 0..n_sigma {
                            density_max = density_max
                                .max(density_block[(offset_lambda + k, offset_sigma + l)].abs());
                        }
                    }
                    if density_max == 0.0 {
                        return;
                    }
                    let quartet = eri_quartet(mu, lambda, sigma, nu, kernel);
                    for i in 0..quartet.na {
                        for j in 0..quartet.nd {
                            let mut value = 0.0;
                            for k in 0..quartet.nb {
                                for l in 0..quartet.nc {
                                    value += density_block[(offset_lambda + k, offset_sigma + l)]
                                        * quartet.get(i, k, l, j);
                                }
                            }
                            exchange[(offset_mu + i, offset_nu + j)] += value;
                        }
                    }
                },
            );
            (cells.near[target], exchange)
        })
        .collect();

    // The pair selection is symmetric under T -> -T, so the truncated sum
    // is Hermitian up to rounding; the projection removes the rounding
    // before negligible blocks are dropped.
    for left in 0..blocks.len() {
        let Some(right) = blocks
            .iter()
            .position(|(translation, _)| (*translation + blocks[left].0).norm() < 1e-8)
        else {
            continue;
        };
        if right < left {
            continue;
        }
        if left == right {
            for i in 0..n {
                for j in 0..i {
                    let average = 0.5 * (blocks[left].1[(i, j)] + blocks[left].1[(j, i)]);
                    blocks[left].1[(i, j)] = average;
                    blocks[left].1[(j, i)] = average;
                }
            }
        } else {
            let (before_right, from_right) = blocks.split_at_mut(right);
            let left_matrix = &mut before_right[left].1;
            let right_matrix = &mut from_right[0].1;
            for i in 0..n {
                for j in 0..n {
                    let average = 0.5 * (left_matrix[(i, j)] + right_matrix[(j, i)]);
                    left_matrix[(i, j)] = average;
                    right_matrix[(j, i)] = average;
                }
            }
        }
    }

    let retained: Vec<ImageBlock> = blocks
        .into_iter()
        .filter(|(_, matrix)| matrix.as_slice().iter().any(|value| *value != 0.0))
        .map(|(translation, matrix)| ImageBlock {
            translation,
            matrix,
        })
        .collect();
    if retained.is_empty() {
        // The exchange operator of an empty spin channel is exactly zero,
        // but an empty block list loses the AO dimension when Bloch summed.
        // Retain a dimensioned home-cell zero so open-shell systems with
        // n_beta=0 receive an n x n zero K matrix rather than a 0 x 0 one.
        vec![ImageBlock {
            translation: Vec3::zero(),
            matrix: Matrix::zeros(n, n),
        }]
    } else {
        retained
    }
}

/// Bloch-summed exchange matrix at `k`.
pub fn exchange_matrix_k(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    k: Vec3,
) -> (Matrix, Matrix) {
    let blocks = exchange_image_blocks(mb, lattice, density, cutoff, kernel);
    crate::periodic::integrals::bloch(&blocks, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{data, MolBasis};
    use crate::scf::fock::JkEngine;
    use crate::scf::EriMode;
    use crate::system::{Atom, Molecule};

    fn h2() -> (Molecule, MolBasis) {
        let molecule = Molecule::new(vec![
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
        let basis = MolBasis::build(&molecule, &set).unwrap();
        (molecule, basis)
    }

    fn home_density(matrix: Matrix) -> Vec<ImageBlock> {
        vec![ImageBlock {
            translation: Vec3::zero(),
            matrix,
        }]
    }

    #[test]
    fn isolated_cell_exchange_reproduces_the_molecular_matrix() {
        let (_, basis) = h2();
        let n = basis.nao_sph;
        let mut density = Matrix::zeros(n, n);
        density[(0, 0)] = 0.8;
        density[(1, 1)] = 0.6;
        density[(0, 1)] = 0.2;
        density[(1, 0)] = 0.2;
        let lattice = Lattice::cubic(40.0);
        let got = exchange_matrix_k(
            &basis,
            &lattice,
            &home_density(density.clone()),
            12.0,
            CoulombKernel::Coulomb,
            Vec3::zero(),
        )
        .0;

        let zero = Matrix::zeros(n, n);
        let engine = JkEngine::new(
            &basis,
            EriMode::Direct {
                screen_threshold: 1e-16,
            },
            CoulombKernel::Coulomb,
        )
        .unwrap();
        let (_, reference) = engine
            .build(&basis, &zero, [&density, &zero], true)
            .unwrap();
        assert!(
            got.max_abs_diff(&reference[0]) < 1e-11,
            "periodic isolated-cell K differs from molecular K by {:e}",
            got.max_abs_diff(&reference[0])
        );
    }

    #[test]
    fn empty_spin_channel_returns_a_dimensioned_zero_exchange() {
        let (_, basis) = h2();
        let n = basis.nao_sph;
        let density = home_density(Matrix::zeros(n, n));
        let (real, imaginary) = exchange_matrix_k(
            &basis,
            &Lattice::cubic(20.0),
            &density,
            8.0,
            CoulombKernel::Coulomb,
            Vec3::new(0.17, -0.11, 0.08),
        );
        assert_eq!((real.rows, real.cols), (n, n));
        assert_eq!((imaginary.rows, imaginary.cols), (n, n));
        assert!(real.as_slice().iter().all(|&value| value == 0.0));
        assert!(imaginary.as_slice().iter().all(|&value| value == 0.0));
    }

    #[test]
    fn exchange_images_obey_the_hermiticity_relation() {
        let (_, basis) = h2();
        let n = basis.nao_sph;
        let mut density = Matrix::zeros(n, n);
        density[(0, 0)] = 0.9;
        density[(1, 1)] = 0.7;
        density[(0, 1)] = 0.25;
        density[(1, 0)] = 0.25;
        let blocks = exchange_image_blocks(
            &basis,
            &Lattice::cubic(7.0),
            &home_density(density),
            8.0,
            CoulombKernel::Coulomb,
        );
        for block in &blocks {
            let mirror = blocks
                .iter()
                .find(|other| (other.translation + block.translation).norm() < 1e-8)
                .expect("exchange image has no -T partner");
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (block.matrix[(i, j)] - mirror.matrix[(j, i)]).abs() < 2e-10,
                        "K(-T) != K(T)^T at {:?}",
                        block.translation
                    );
                }
            }
        }
    }

    #[test]
    fn minimum_image_weights_partition_every_replica_class() {
        let lattice = Lattice::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.4, 4.0, 0.0),
            Vec3::new(0.0, 0.3, 5.0),
        );
        // A 4 x 2 x 1 Gamma-centred mesh in reduced coordinates.
        let reciprocal = lattice.reciprocal();
        let mut k_points = Vec::new();
        for i in 0..4 {
            for j in 0..2 {
                k_points.push(reciprocal[0] * (i as f64 / 4.0) + reciprocal[1] * (j as f64 / 2.0));
            }
        }
        let periods = born_von_karman_periods(&lattice, &k_points).unwrap();
        assert_eq!(periods, [4, 2, 1]);
        let weight = |n: [i64; 3]| minimum_image_weight(&lattice, periods, lattice.translation(n));
        assert_eq!(weight([0, 0, 0]), 1.0);
        assert_eq!(weight([1, 0, 0]), 1.0);
        // Half a period: two replicas of the same length.
        assert_eq!(weight([2, 0, 0]), 0.5);
        assert_eq!(weight([-2, 0, 0]), 0.5);
        assert_eq!(weight([3, 0, 0]), 0.0);
        assert_eq!(weight([0, 0, 1]), 0.0);
        // The weights of every replica class add up to one.
        for n0 in 0..4i64 {
            for n1 in 0..2i64 {
                let mut sum = 0.0;
                for m0 in -3i64..=3 {
                    for m1 in -3i64..=3 {
                        for m2 in -3i64..=3 {
                            sum += weight([n0 + 4 * m0, n1 + 2 * m1, m2]);
                        }
                    }
                }
                assert!((sum - 1.0).abs() < 1e-14, "class ({n0}, {n1}): {sum}");
            }
        }
        // Blocks are scaled by the separation of their two atoms and the
        // aliased ones dropped. Two helium atoms half a period apart: inside
        // the home block the pair (0, 1) is one of two equal replicas.
        let set = data::load("sto-3g").unwrap();
        let molecule = crate::system::Molecule::new(vec![
            crate::system::Atom {
                z: 2,
                position: Vec3::zero(),
            },
            crate::system::Atom {
                z: 2,
                position: lattice.a[0] * 2.0,
            },
        ]);
        let cell = crate::periodic::Cell::new(molecule, lattice.clone(), &set, 8.0).unwrap();
        let ones = |n: [i64; 3]| ImageBlock {
            translation: lattice.translation(n),
            matrix: Matrix::from_row_major(2, 2, vec![1.0; 4]),
        };
        let kept = minimum_image_blocks(
            &cell,
            &k_points,
            vec![
                ones([0, 0, 0]),
                ones([1, 0, 0]),
                ones([3, 0, 0]),
                ones([0, 0, 1]),
            ],
        );
        assert_eq!(kept.len(), 3);
        // Home block: same-atom elements 1; the pair two lattice vectors
        // apart is half a period: weight 1/2 in both directions.
        assert_eq!(kept[0].matrix.as_slice(), &[1.0, 0.5, 0.5, 1.0]);
        // Translation a1: separations 1, 3 (aliased), -1 and 1 lattice vectors.
        assert_eq!(kept[1].matrix.as_slice(), &[1.0, 0.0, 1.0, 1.0]);
        // Translation 3 a1: only the pair at 3 - 2 = 1 lattice vector survives.
        assert_eq!(kept[2].matrix.as_slice(), &[0.0, 0.0, 1.0, 0.0]);
    }
}
