// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Third and fourth displacement derivatives of the periodic dispersion sum.
//! Pair and triple lists are fixed at the reference geometry. Nested forward
//! duals differentiate the same coordination numbers, C6 interpolation and
//! damping functions as the energy, including the optional ATM term.

use super::{energy_g, evaluate, image_lists, Geometry, PeriodicDispersion};
use crate::dual::{DualN, Scalar};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::{Lattice, PeriodicGeometry};
use crate::{DftError, Result};
use rayon::prelude::*;

/// A cubic or quartic force constant contracted with displacement patterns.
#[derive(Clone, Debug)]
pub struct DispersionForceConstants {
    /// Dispersion energy per primitive cell, Hartree.
    pub energy: f64,
    /// Cubic `E_[u]jk` or quartic `E_[u][v]jk`, Hartree/Bohr^order.
    pub matrix: Matrix,
    pub order: usize,
    /// Largest unprojected sum over atoms on either free Cartesian index.
    pub acoustic_residual: f64,
}

/// Complete Cartesian displacement tensor of a periodic dispersion energy.
#[derive(Clone, Debug)]
pub struct DispersionForceConstantTensor {
    pub energy: f64,
    /// Last index fastest, Hartree/Bohr^order per primitive cell.
    pub tensor: Vec<f64>,
    pub dimension: usize,
    pub order: usize,
    /// Raw differences under adjacent-index permutations, before projection.
    pub permutation_residual: f64,
    /// Largest sum over atoms on any Cartesian index, before projection.
    pub acoustic_residual: f64,
}

fn acoustic(matrix: &Matrix, atoms: usize) -> f64 {
    let mut largest = 0.0_f64;
    for free in 0..matrix.rows {
        for axis in 0..3 {
            let row: f64 = (0..atoms).map(|a| matrix[(free, 3 * a + axis)]).sum();
            let column: f64 = (0..atoms).map(|a| matrix[(3 * a + axis, free)]).sum();
            largest = largest.max(row.abs()).max(column.abs());
        }
    }
    largest
}

/// Displacement derivatives with all lattice images of an atom moving with
/// it. One probe gives the cubic derivative of the Hessian; two probes give
/// its mixed quartic derivative. Probes have `3*N` finite dimensionless
/// components in `3*atom + axis` order. No finite differences enter this path.
pub fn periodic_dispersion_force_constants_along(
    numbers: &[u8],
    positions: &[Vec3],
    lattice: &Lattice,
    settings: &PeriodicDispersion,
    probes: &[Vec<f64>],
) -> Result<DispersionForceConstants> {
    if probes.is_empty() {
        return Err(DftError::InvalidInput(
            "periodic cubic/quartic dispersion needs one or two displacement probes".into(),
        ));
    }
    force_constants_along(
        numbers,
        &Geometry::bulk(positions, lattice),
        settings,
        probes,
    )
}

fn force_constants_along(
    numbers: &[u8],
    geometry: &Geometry<'_>,
    settings: &PeriodicDispersion,
    probes: &[Vec<f64>],
) -> Result<DispersionForceConstants> {
    let n = numbers.len();
    if probes.len() > 2
        || probes
            .iter()
            .any(|p| p.len() != 3 * n || p.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "periodic dispersion force constants need one or two finite 3*N displacement probes"
                .into(),
        ));
    }
    let reference = evaluate(numbers, geometry, geometry, settings)?;
    let images = image_lists(geometry, n, settings);
    let lattice = geometry.lattice;
    type First = DualN<f64, 1>;
    type Directions = DualN<First, 1>;
    type Inner = DualN<Directions, 3>;
    type Outer = DualN<Inner, 3>;
    let rows: [[Outer; 3]; 3] =
        std::array::from_fn(|k| std::array::from_fn(|c| Outer::from_f64(lattice.a[k].get(c))));
    // Evaluate both free-index orders independently so the raw tensor
    // permutation and translation defects remain observable.
    let pairs: Vec<_> = (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).collect();
    let blocks: Vec<[[f64; 3]; 3]> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let seeded: Vec<[Outer; 3]> = geometry
                .positions
                .iter()
                .enumerate()
                .map(|(atom, point)| {
                    std::array::from_fn(|axis| {
                        let index = 3 * atom + axis;
                        let mut first = First::constant(point.get(axis));
                        if let Some(probe) = probes.first() {
                            first.im[0] = probe[index];
                        }
                        let mut direction = Directions::constant(first);
                        if probes.len() == 2 {
                            direction.im[0] = First::constant(probes[1][index]);
                        }
                        let inner = if atom == b {
                            Inner::variable(direction, axis)
                        } else {
                            Inner::constant(direction)
                        };
                        if atom == a {
                            Outer::variable(inner, axis)
                        } else {
                            Outer::constant(inner)
                        }
                    })
                })
                .collect();
            let energy = energy_g(numbers, &seeded, &rows, &images, settings);
            std::array::from_fn(|x| {
                std::array::from_fn(|y| {
                    let value = energy.im[x].im[y];
                    match probes.len() {
                        0 => value.re.re,
                        1 => value.re.im[0],
                        _ => value.im[0].im[0],
                    }
                })
            })
        })
        .collect();
    let mut matrix = Matrix::zeros(3 * n, 3 * n);
    for (&(a, b), block) in pairs.iter().zip(&blocks) {
        for x in 0..3 {
            for y in 0..3 {
                matrix[(3 * a + x, 3 * b + y)] = block[x][y];
            }
        }
    }
    if matrix.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "non-finite periodic dispersion high-order derivative".into(),
        ));
    }
    Ok(DispersionForceConstants {
        energy: reference.energy,
        acoustic_residual: acoustic(&matrix, n),
        matrix,
        order: probes.len() + 2,
    })
}

/// Complete cubic or quartic force constants of the dispersion energy.
/// Leading indices are differentiated independently of the free Hessian
/// indices. The reported permutation and acoustic defects are not projected.
pub fn periodic_dispersion_force_constant_tensor(
    numbers: &[u8],
    positions: &[Vec3],
    lattice: &Lattice,
    settings: &PeriodicDispersion,
    order: usize,
) -> Result<DispersionForceConstantTensor> {
    if !(3..=4).contains(&order) {
        return Err(DftError::InvalidInput(
            "periodic dispersion force-constant tensors have order three or four".into(),
        ));
    }
    force_constant_tensor(
        numbers,
        &Geometry::bulk(positions, lattice),
        settings,
        order,
    )
}

/// Complete Cartesian energy derivatives of lattice dispersion in orders
/// one through four. Reference image lists use the active 1D/2D/3D subspace.
pub fn dimensional_dispersion_force_constant_tensor(
    numbers: &[u8],
    positions: &[Vec3],
    geometry: &PeriodicGeometry,
    settings: &PeriodicDispersion,
    order: usize,
) -> Result<DispersionForceConstantTensor> {
    if !(1..=4).contains(&order) {
        return Err(DftError::InvalidInput(
            "dimensional dispersion nuclear derivatives require order one through four".into(),
        ));
    }
    force_constant_tensor(
        numbers,
        &Geometry::dimensional(positions, geometry),
        settings,
        order,
    )
}

fn force_constant_tensor(
    numbers: &[u8],
    geometry: &Geometry<'_>,
    settings: &PeriodicDispersion,
    order: usize,
) -> Result<DispersionForceConstantTensor> {
    let n = 3 * numbers.len();
    let size = n.checked_pow(order as u32).ok_or_else(|| {
        DftError::InvalidInput("periodic dispersion force-constant tensor size overflow".into())
    })?;
    let mut tensor = Vec::new();
    tensor.try_reserve_exact(size).map_err(|_| {
        DftError::InvalidInput("periodic dispersion force-constant tensor allocation failed".into())
    })?;
    tensor.resize(size, 0.0);
    let energy = if order == 1 {
        let result = evaluate(numbers, geometry, geometry, settings)?;
        for (target, value) in tensor
            .iter_mut()
            .zip(result.forces.iter().flat_map(|f| (-*f).to_array()))
        {
            *target = value;
        }
        result.energy
    } else if order == 2 {
        let result = force_constants_along(numbers, geometry, settings, &[])?;
        tensor.copy_from_slice(result.matrix.as_slice());
        result.energy
    } else {
        let mut energy = 0.0;
        for i in 0..n {
            let mut first = vec![0.0; n];
            first[i] = 1.0;
            let last = if order == 3 { 1 } else { n };
            for j in 0..last {
                let mut probes = vec![first.clone()];
                if order == 4 {
                    let mut second = vec![0.0; n];
                    second[j] = 1.0;
                    probes.push(second);
                }
                let result = force_constants_along(numbers, geometry, settings, &probes)?;
                energy = result.energy;
                let offset = if order == 3 {
                    i * n * n
                } else {
                    (i * n + j) * n * n
                };
                tensor[offset..offset + n * n].copy_from_slice(result.matrix.as_slice());
            }
        }
        energy
    };
    let mut permutation = 0.0_f64;
    let mut translation = 0.0_f64;
    for flat in 0..size {
        let mut index = vec![0; order];
        let mut remainder = flat;
        for slot in index.iter_mut().rev() {
            *slot = remainder % n;
            remainder /= n;
        }
        for axis in 0..order {
            if axis + 1 < order {
                let mut swapped = index.clone();
                swapped.swap(axis, axis + 1);
                let other = swapped.iter().fold(0, |a, b| a * n + b);
                permutation = permutation.max((tensor[flat] - tensor[other]).abs());
            }
            if index[axis] < 3 {
                let mut entry = index.clone();
                let sum: f64 = (0..numbers.len())
                    .map(|atom| {
                        entry[axis] = 3 * atom + index[axis];
                        tensor[entry.iter().fold(0, |a, b| a * n + b)]
                    })
                    .sum();
                translation = translation.max(sum.abs());
            }
        }
    }
    Ok(DispersionForceConstantTensor {
        energy,
        tensor,
        dimension: n,
        order,
        permutation_residual: permutation,
        acoustic_residual: translation,
    })
}
