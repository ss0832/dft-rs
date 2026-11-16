// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Berry-phase polarization of an insulating periodic reference (modern
//! theory of polarization: R. D. King-Smith, D. Vanderbilt, Phys. Rev. B 47,
//! 1651 (1993); R. Resta, Rev. Mod. Phys. 66, 899 (1994)).
//!
//! For each reciprocal axis `b_i` the occupied manifold is followed along
//! strings of `N_i` points `k_j = k_perp + j b_i / N_i`. With the
//! cell-periodic cross-k AO metric `M(k, k') = <u_basis(k)|u_basis(k')>` of
//! [`super::coherent_transport::CrossMetric`] (plane-wave AO integrals, which
//! keep the orbital positions and the reciprocal-boundary sewing) the string
//! phase is
//!
//! ```text
//! phi_i(k_perp) = -Im ln prod_j det[C_o(k_j)^+ M(k_j, k_(j+1)) C_o(k_(j+1))]
//! ```
//!
//! and its average over the perpendicular mesh gives the reduced
//! polarization along the lattice vectors `a_i`:
//!
//! ```text
//! p_i = -g sum_spin <phi_i> / (2 pi) + sum_atoms Z (r . b_i) / (2 pi)
//! P   = (1/V) sum_i p_i a_i
//! ```
//!
//! (`g` = 2 for a closed-shell reference and 1 per spin channel otherwise;
//! atomic units with `e = 1`). Only changes of `P` along an insulating path
//! are physical: `p_i` is defined modulo one (an elementary charge moved by
//! a lattice vector) and its electronic part modulo `g`.

use super::band_transport::BandModel;
use super::bloch::BlochMatrix;
use super::coherent_transport::{determinant_phase, CrossMetric};
use super::phonon::{adjoint, product};
use super::scf::PeriodicScfResult;
use super::Cell;
use crate::error::{DftError, Result};

/// Berry-phase polarization and its diagnostics.
#[derive(Clone, Debug)]
pub struct BerryPolarization {
    /// Mesh-averaged Berry phases of the occupied manifold, `[spin][axis]`
    /// (radians).
    pub phases: Vec<[f64; 3]>,
    /// Electronic reduced polarization `-g sum_spin phi_i / (2 pi)`, wrapped
    /// to `(-g/2, g/2]`.
    pub electronic_reduced: [f64; 3],
    /// Ionic reduced polarization `sum Z (r . b_i) / (2 pi)`.
    pub ionic_reduced: [f64; 3],
    /// Total reduced polarization, wrapped to `(-1/2, 1/2]`.
    pub total_reduced: [f64; 3],
    /// Cartesian polarization `(1/V) sum_i p_i a_i` (e/Bohr^2), defined
    /// modulo `a_i / V`.
    pub polarization: [f64; 3],
    /// Largest deviation of a string phase from the mesh average, per axis.
    pub string_spread: [f64; 3],
    /// Smallest `ln |det|` of an occupied link.
    pub min_link_log_abs_determinant: f64,
    /// Smallest direct gap met on the strings (Hartree); infinite when the
    /// basis has no empty band.
    pub min_gap: f64,
}

/// `x` reduced to `(-period/2, period/2]`.
fn wrap(x: f64, period: f64) -> f64 {
    let y = x - period * (x / period).round();
    if y <= -0.5 * period {
        y + period
    } else {
        y
    }
}

/// The first `count` columns of `c`.
fn leading_columns(c: &BlochMatrix, count: usize) -> BlochMatrix {
    let mut out = BlochMatrix::zeros(c.rows(), count);
    for i in 0..c.rows() {
        for j in 0..count {
            out.re[(i, j)] = c.re[(i, j)];
            out.im[(i, j)] = c.im[(i, j)];
        }
    }
    out
}

/// Berry-phase polarization of the converged insulating reference `scf` on
/// the string mesh `mesh` (points along each reciprocal axis; the other two
/// entries give the perpendicular mesh of that axis).
pub fn berry_polarization(
    cell: &Cell,
    model: &BandModel,
    scf: &PeriodicScfResult,
    mesh: [usize; 3],
) -> Result<BerryPolarization> {
    if mesh.contains(&0) {
        return Err(DftError::InvalidInput(
            "the Berry-phase mesh needs three positive entries".into(),
        ));
    }
    if scf.chemical_potential.iter().any(|mu| mu.is_some()) {
        return Err(DftError::Unsupported {
            feature: "Berry-phase polarization of a Fermi-Dirac (metallic) reference".into(),
            missing: "an insulating reference with integer occupations".into(),
        });
    }
    let reciprocal = cell.lattice.reciprocal();
    let two_pi = 2.0 * std::f64::consts::PI;
    let degeneracy = if model.spins == 1 { 2.0 } else { 1.0 };
    let occupied_counts = [scf.n_alpha, scf.n_beta];
    let mut phases = Vec::with_capacity(model.spins);
    let mut string_spread = [0.0_f64; 3];
    let mut min_log = f64::INFINITY;
    let mut min_gap = f64::INFINITY;
    for (spin, &occupied) in occupied_counts.iter().enumerate().take(model.spins) {
        let mut per_axis = [0.0; 3];
        if occupied > 0 {
            for axis in 0..3 {
                let (second, third) = ((axis + 1) % 3, (axis + 2) % 3);
                let step = reciprocal[axis] * (1.0 / mesh[axis] as f64);
                let cross = CrossMetric::new(cell, step)?;
                let mut strings = Vec::with_capacity(mesh[second] * mesh[third]);
                for y in 0..mesh[second] {
                    for z in 0..mesh[third] {
                        let base = reciprocal[second] * (y as f64 / mesh[second] as f64)
                            + reciprocal[third] * (z as f64 / mesh[third] as f64);
                        let mut orbitals = Vec::with_capacity(mesh[axis]);
                        for x in 0..mesh[axis] {
                            let bands = model.bands(spin, base + step * x as f64)?;
                            if occupied > bands.energies.len() {
                                return Err(DftError::InvalidInput(
                                    "more occupied bands than basis functions".into(),
                                ));
                            }
                            if occupied < bands.energies.len() {
                                min_gap = min_gap
                                    .min(bands.energies[occupied] - bands.energies[occupied - 1]);
                            }
                            orbitals.push(leading_columns(&bands.coefficients, occupied));
                        }
                        let mut total = 0.0;
                        for x in 0..mesh[axis] {
                            let metric = cross.at(base + step * (x + 1) as f64);
                            let right = &orbitals[(x + 1) % mesh[axis]];
                            let link = product(&product(&adjoint(&orbitals[x]), &metric), right);
                            let (unit, log_abs) = determinant_phase(&link)?;
                            min_log = min_log.min(log_abs);
                            total += unit.im.atan2(unit.re);
                        }
                        strings.push(-total);
                    }
                }
                // Every string phase is defined modulo 2 pi: take the branch
                // closest to the first string before averaging.
                let first = wrap(strings[0], two_pi);
                let aligned: Vec<f64> = strings
                    .iter()
                    .map(|s| first + wrap(s - first, two_pi))
                    .collect();
                let mean = aligned.iter().sum::<f64>() / aligned.len() as f64;
                let spread = aligned.iter().fold(0.0_f64, |m, s| m.max((s - mean).abs()));
                string_spread[axis] = string_spread[axis].max(spread);
                per_axis[axis] = mean;
            }
        }
        phases.push(per_axis);
    }
    if min_gap <= 0.0 {
        return Err(DftError::Unsupported {
            feature: "Berry-phase polarization without a gap on the string mesh".into(),
            missing: "a positive direct gap between occupied and empty bands".into(),
        });
    }
    let charges = cell.basis.effective_charges(&cell.molecule);
    let volume = cell.lattice.volume().abs();
    let mut electronic_reduced = [0.0; 3];
    let mut ionic_reduced = [0.0; 3];
    let mut total_reduced = [0.0; 3];
    let mut polarization = [0.0; 3];
    for axis in 0..3 {
        let phase: f64 = phases.iter().map(|p| p[axis]).sum();
        electronic_reduced[axis] = wrap(-degeneracy * phase / two_pi, degeneracy);
        ionic_reduced[axis] = cell
            .molecule
            .atoms
            .iter()
            .zip(&charges)
            .map(|(atom, z)| z * atom.position.dot(reciprocal[axis]) / two_pi)
            .sum();
        total_reduced[axis] = wrap(electronic_reduced[axis] + ionic_reduced[axis], 1.0);
        let a = cell.lattice.a[axis].to_array();
        for (p, component) in polarization.iter_mut().zip(a) {
            *p += total_reduced[axis] * component / volume;
        }
    }
    Ok(BerryPolarization {
        phases,
        electronic_reduced,
        ionic_reduced,
        total_reduced,
        polarization,
        string_spread,
        min_link_log_abs_determinant: min_log,
        min_gap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_maps_onto_the_half_open_interval() {
        let pi = std::f64::consts::PI;
        assert!((wrap(3.0 * pi + 0.1, 2.0 * pi) - (-pi + 0.1)).abs() < 1e-14);
        assert!((wrap(-0.5, 1.0) - 0.5).abs() < 1e-15);
        assert!((wrap(0.5, 1.0) - 0.5).abs() < 1e-15);
        assert!((wrap(1.25, 1.0) - 0.25).abs() < 1e-15);
        assert!((wrap(-2.75, 2.0) - (-0.75)).abs() < 1e-15);
    }
}
