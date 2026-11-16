// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Extended X-ray absorption fine structure of one absorbing atom in the
//! single-scattering approximation, routed through the common driver.
//!
//! ```text
//! chi(k) = S0^2 sum_j |f_j(k)| / (k R_j^2) exp(-2 k^2 sigma_j^2)
//!          exp(-2 R_j / lambda) sin(2 k R_j + phi_j(k))
//! ```
//!
//! over the neighbours `j` of the absorber inside `exafs_rmax` (lattice
//! images included for a periodic structure). The backscattering amplitude
//! `|f_j(k)|` and the total phase `phi_j(k)` of each element are tabulated
//! input, interpolated linearly in `k`; the amplitude reduction `S0^2`, the
//! mean free path `lambda` and the mean-square relative displacements
//! `sigma_j^2` are input as well.
//!
//! With angle-resolved scattering functions (`exafs_scattering_angles`) and
//! twice the phase shift of the absorber (`exafs_central_phase`) the
//! double-scattering paths `0 -> j -> l -> 0` are added in the plane-wave
//! approximation, averaged over the polarization for a K edge:
//!
//! ```text
//! chi_2(k) = -S0^2 sum_(j != l) (r_j . r_l) |f_j(t_j, k)| |f_l(t_l, k)| / (k R_0j R_jl R_l0)
//!            exp(-2 k^2 sigma_ms^2) exp(-R_tot / lambda)
//!            sin(k R_tot + phi_j(t_j, k) + phi_l(t_l, k) - 2 delta_c(k))
//! ```
//!
//! with the unit vectors `r_j`, `r_l` from the absorber to the two
//! scatterers, the scattering angles `t` (180 degrees is backscattering),
//! `R_tot = R_0j + R_jl + R_l0`, and total phases `phi` in the convention of
//! the single-scattering table (each holds `2 delta_c`, so one is taken
//! out). Paths with `R_tot / 2 <= exafs_rmax` are kept.
//!
//! No electronic-structure calculation is involved: scattering of third and
//! higher order, curved-wave corrections and the energy dependence of the
//! mean free path are outside the model.

use super::controls::Controls;
use super::molecular::units;
use super::value::Value;
use super::System;
use crate::{DftError, Result};

/// One single-scattering path: the neighbour and its distance.
struct Path {
    atom: usize,
    number: u8,
    distance: f64,
    /// Position of the neighbour relative to the absorber.
    separation: [f64; 3],
}

fn invalid(message: &str) -> DftError {
    DftError::InvalidInput(message.to_string())
}

/// Linear interpolation of `(k, amplitude, phase)` rows sorted by `k`;
/// `None` outside the tabulated range.
fn interpolate(rows: &[(f64, f64, f64)], k: f64) -> Option<(f64, f64)> {
    let last = rows.len().checked_sub(1)?;
    if k < rows[0].0 || k > rows[last].0 {
        return None;
    }
    let upper = rows.iter().position(|row| row.0 >= k)?;
    if upper == 0 {
        return Some((rows[0].1, rows[0].2));
    }
    let (a, b) = (rows[upper - 1], rows[upper]);
    let t = (k - a.0) / (b.0 - a.0);
    Some((a.1 + t * (b.1 - a.1), a.2 + t * (b.2 - a.2)))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn exafs(system: &System, controls: &Controls) -> Result<Value> {
    let natoms = system.numbers.len();
    let absorber = controls.opt_usize("exafs_absorber")?.ok_or_else(|| {
        invalid("exafs needs exafs_absorber: the index of the absorbing atom, counted from 0")
    })?;
    let s02 = controls.f64("exafs_s02", 1.0)?;
    let mean_free_path = controls
        .opt_f64("exafs_mfp")?
        .ok_or_else(|| invalid("exafs needs exafs_mfp: the photoelectron mean free path"))?;
    let rmax = controls.f64("exafs_rmax", 12.0)?;
    let table = controls.opt_rows("exafs_scattering")?.ok_or_else(|| {
        invalid(
            "exafs needs exafs_scattering: rows of atomic number, k, backscattering amplitude \
             and total phase",
        )
    })?;
    let k_range = controls
        .opt_f64_list("exafs_k_range")?
        .unwrap_or_else(|| vec![1.0, 8.0, 0.05]);
    let k_weight = controls.usize("exafs_k_weight", 2)?;
    let sigma2 = controls.opt_f64_list("exafs_sigma2")?;
    let r_points = controls.usize("exafs_r_points", 401)?;
    let angle_table = controls.opt_rows("exafs_scattering_angles")?;
    let central_table = controls.opt_rows("exafs_central_phase")?;
    let ms_sigma2 = controls.f64("exafs_ms_sigma2", 0.0)?;
    controls.finish("exafs")?;
    if absorber >= natoms {
        return Err(invalid("exafs_absorber is not an atom of the structure"));
    }
    if !(s02.is_finite()
        && s02 > 0.0
        && mean_free_path.is_finite()
        && mean_free_path > 0.0
        && rmax.is_finite()
        && rmax > 0.0)
    {
        return Err(invalid(
            "exafs_s02, exafs_mfp and exafs_rmax are finite and positive",
        ));
    }
    if k_range.len() != 3
        || k_range.iter().any(|v| !v.is_finite())
        || !(k_range[0] > 0.0 && k_range[1] > k_range[0] && k_range[2] > 0.0)
        || r_points < 2
    {
        return Err(invalid(
            "exafs_k_range is [k_min, k_max, k_step] with 0 < k_min < k_max and a positive \
             step, and exafs_r_points is at least 2",
        ));
    }

    // Tabulated scattering functions of every element.
    let mut scattering: Vec<(u8, Vec<(f64, f64, f64)>)> = Vec::new();
    for row in &table {
        if row.len() != 4
            || row.iter().any(|v| !v.is_finite())
            || row[0] < 1.0
            || row[0] > 118.0
            || row[0].fract() != 0.0
        {
            return Err(invalid(
                "every row of exafs_scattering is [atomic number, k, amplitude, phase]",
            ));
        }
        let number = row[0] as u8;
        let entry = match scattering.iter().position(|(z, _)| *z == number) {
            Some(index) => index,
            None => {
                scattering.push((number, Vec::new()));
                scattering.len() - 1
            }
        };
        scattering[entry].1.push((row[1], row[2], row[3]));
    }
    for (number, rows) in &mut scattering {
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        if rows.windows(2).any(|pair| pair[1].0 <= pair[0].0) {
            return Err(DftError::InvalidInput(format!(
                "exafs_scattering has two rows with the same k for atomic number {number}"
            )));
        }
    }

    // Neighbours of the absorber inside rmax, with the lattice images of a
    // periodic structure.
    let reach: [i64; 3] = match &system.cell {
        Some(rows) => {
            let volume = dot(rows[0], cross(rows[1], rows[2])).abs();
            if !(volume.is_finite() && volume > 0.0) {
                return Err(invalid("the cell of the structure has no volume"));
            }
            std::array::from_fn(|i| {
                if !system.pbc[i] {
                    return 0;
                }
                let face = cross(rows[(i + 1) % 3], rows[(i + 2) % 3]);
                // Distance between neighbouring lattice planes of this direction.
                let height = volume / dot(face, face).sqrt();
                (rmax / height).ceil() as i64 + 1
            })
        }
        None => [0; 3],
    };
    let origin = system.positions[absorber];
    let mut paths: Vec<Path> = Vec::new();
    for atom in 0..natoms {
        for n0 in -reach[0]..=reach[0] {
            for n1 in -reach[1]..=reach[1] {
                for n2 in -reach[2]..=reach[2] {
                    let mut position = system.positions[atom];
                    if let Some(rows) = &system.cell {
                        for (axis, value) in position.iter_mut().enumerate() {
                            *value += n0 as f64 * rows[0][axis]
                                + n1 as f64 * rows[1][axis]
                                + n2 as f64 * rows[2][axis];
                        }
                    }
                    let separation: [f64; 3] = std::array::from_fn(|a| position[a] - origin[a]);
                    let distance = dot(separation, separation).sqrt();
                    if distance > 1e-8 && distance <= rmax {
                        paths.push(Path {
                            atom,
                            number: system.numbers[atom],
                            distance,
                            separation,
                        });
                    }
                }
            }
        }
    }
    paths.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.atom.cmp(&b.atom)));
    if paths.is_empty() {
        return Err(invalid(
            "no neighbour of the absorber lies inside exafs_rmax",
        ));
    }
    let mut warnings: Vec<String> = Vec::new();
    let sigma: Vec<f64> = match &sigma2 {
        None => {
            warnings.push(
                "exafs_sigma2 was not given: the mean-square relative displacements are zero \
                 (static structure)"
                    .into(),
            );
            vec![0.0; paths.len()]
        }
        Some(values) if values.len() == 1 => vec![values[0]; paths.len()],
        Some(values) if values.len() == paths.len() => values.clone(),
        Some(values) => {
            return Err(DftError::InvalidInput(format!(
                "exafs_sigma2 has {} values for {} scattering paths; give one value or one per \
                 path in order of distance",
                values.len(),
                paths.len()
            )))
        }
    };
    if sigma.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(invalid("exafs_sigma2 values are finite and not negative"));
    }

    let steps = ((k_range[1] - k_range[0]) / k_range[2]).floor() as usize;
    let k_grid: Vec<f64> = (0..=steps)
        .map(|i| k_range[0] + i as f64 * k_range[2])
        .collect();
    let mut chi = vec![0.0_f64; k_grid.len()];
    for (path, s2) in paths.iter().zip(&sigma) {
        let rows = &scattering
            .iter()
            .find(|(z, _)| *z == path.number)
            .ok_or_else(|| {
                DftError::InvalidInput(format!(
                    "exafs_scattering has no rows for atomic number {}",
                    path.number
                ))
            })?
            .1;
        for (value, k) in chi.iter_mut().zip(&k_grid) {
            let (amplitude, phase) = interpolate(rows, *k).ok_or_else(|| {
                DftError::InvalidInput(format!(
                    "exafs_scattering of atomic number {} does not cover k = {k}",
                    path.number
                ))
            })?;
            *value += s02 * amplitude / (k * path.distance * path.distance)
                * (-2.0 * k * k * s2).exp()
                * (-2.0 * path.distance / mean_free_path).exp()
                * (2.0 * k * path.distance + phase).sin();
        }
    }
    // Double scattering in the plane-wave approximation.
    let single = chi.clone();
    let mut double = vec![0.0_f64; k_grid.len()];
    let mut double_paths = 0_usize;
    if let Some(table) = &angle_table {
        let central = central_table.as_ref().ok_or_else(|| {
            invalid(
                "exafs_scattering_angles needs exafs_central_phase: rows of k and twice the \
                 phase shift of the absorber",
            )
        })?;
        if !(ms_sigma2.is_finite() && ms_sigma2 >= 0.0) {
            return Err(invalid("exafs_ms_sigma2 is finite and not negative"));
        }
        let mut centre: Vec<(f64, f64, f64)> = Vec::with_capacity(central.len());
        for row in central {
            if row.len() != 2 || row.iter().any(|v| !v.is_finite()) {
                return Err(invalid(
                    "every row of exafs_central_phase is [k, twice the phase shift of the \
                     absorber]",
                ));
            }
            centre.push((row[0], 0.0, row[1]));
        }
        centre.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Scattering functions per element and scattering angle.
        let mut angular: Vec<(u8, Vec<(f64, Vec<(f64, f64, f64)>)>)> = Vec::new();
        for row in table {
            if row.len() != 5
                || row.iter().any(|v| !v.is_finite())
                || row[0] < 1.0
                || row[0] > 118.0
                || row[0].fract() != 0.0
                || !(0.0..=180.0).contains(&row[1])
            {
                return Err(invalid(
                    "every row of exafs_scattering_angles is [atomic number, scattering angle \
                     in degrees (180 is backscattering), k, amplitude, phase]",
                ));
            }
            let number = row[0] as u8;
            let element = match angular.iter().position(|(z, _)| *z == number) {
                Some(index) => index,
                None => {
                    angular.push((number, Vec::new()));
                    angular.len() - 1
                }
            };
            let angles = &mut angular[element].1;
            let slot = match angles.iter().position(|(angle, _)| *angle == row[1]) {
                Some(index) => index,
                None => {
                    angles.push((row[1], Vec::new()));
                    angles.len() - 1
                }
            };
            angles[slot].1.push((row[2], row[3], row[4]));
        }
        for (_, angles) in &mut angular {
            angles.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (_, rows) in angles.iter_mut() {
                rows.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }
        // Amplitude and phase of an element at a scattering angle and k,
        // linear in both.
        let scatter = |number: u8, angle: f64, k: f64| -> Result<(f64, f64)> {
            let missing = || {
                DftError::InvalidInput(format!(
                    "exafs_scattering_angles does not cover atomic number {number} at \
                     {angle:.2} degrees and k = {k}"
                ))
            };
            let angles = &angular
                .iter()
                .find(|(z, _)| *z == number)
                .ok_or_else(missing)?
                .1;
            let upper = angles
                .iter()
                .position(|(tabulated, _)| *tabulated >= angle)
                .ok_or_else(missing)?;
            let high = interpolate(&angles[upper].1, k).ok_or_else(missing)?;
            if angles[upper].0 == angle {
                return Ok(high);
            }
            if upper == 0 {
                return Err(missing());
            }
            let low = interpolate(&angles[upper - 1].1, k).ok_or_else(missing)?;
            let t = (angle - angles[upper - 1].0) / (angles[upper].0 - angles[upper - 1].0);
            Ok((low.0 + t * (high.0 - low.0), low.1 + t * (high.1 - low.1)))
        };
        let degrees = |cosine: f64| cosine.clamp(-1.0, 1.0).acos().to_degrees();
        for (a, first) in paths.iter().enumerate() {
            for (b, second) in paths.iter().enumerate() {
                if a == b {
                    continue;
                }
                let leg: [f64; 3] =
                    std::array::from_fn(|c| second.separation[c] - first.separation[c]);
                let middle = dot(leg, leg).sqrt();
                let total = first.distance + middle + second.distance;
                if middle <= 1e-8 || 0.5 * total > rmax {
                    continue;
                }
                // Angle between the incoming and the outgoing direction at
                // each scatterer: 180 degrees is backscattering.
                let at_first = degrees(dot(first.separation, leg) / (first.distance * middle));
                let at_second = degrees(-dot(leg, second.separation) / (middle * second.distance));
                let orientation =
                    dot(first.separation, second.separation) / (first.distance * second.distance);
                double_paths += 1;
                for (value, k) in double.iter_mut().zip(&k_grid) {
                    let (amplitude_first, phase_first) = scatter(first.number, at_first, *k)?;
                    let (amplitude_second, phase_second) = scatter(second.number, at_second, *k)?;
                    let (_, shift) = interpolate(&centre, *k).ok_or_else(|| {
                        DftError::InvalidInput(format!(
                            "exafs_central_phase does not cover k = {k}"
                        ))
                    })?;
                    *value -= s02 * orientation * amplitude_first * amplitude_second
                        / (k * first.distance * middle * second.distance)
                        * (-2.0 * k * k * ms_sigma2).exp()
                        * (-total / mean_free_path).exp()
                        * (k * total + phase_first + phase_second - shift).sin();
                }
            }
        }
        for (value, extra) in chi.iter_mut().zip(&double) {
            *value += extra;
        }
    } else if central_table.is_some() {
        return Err(invalid(
            "exafs_central_phase is used with exafs_scattering_angles only",
        ));
    }
    let weighted: Vec<f64> = chi
        .iter()
        .zip(&k_grid)
        .map(|(value, k)| value * k.powi(k_weight as i32))
        .collect();

    // Magnitude of the Fourier transform of the weighted signal with a Hann
    // window over the k grid.
    let (first, last) = (k_grid[0], k_grid[k_grid.len() - 1]);
    let window = |k: f64| -> f64 {
        if last > first {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * (k - first) / (last - first)).cos())
        } else {
            1.0
        }
    };
    let r_grid: Vec<f64> = (0..r_points)
        .map(|i| rmax * i as f64 / (r_points - 1) as f64)
        .collect();
    let transform: Vec<f64> = r_grid
        .iter()
        .map(|r| {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (value, k) in weighted.iter().zip(&k_grid) {
                let term = window(*k) * value * k_range[2];
                re += term * (2.0 * k * r).cos();
                im += term * (2.0 * k * r).sin();
            }
            (re * re + im * im).sqrt() / std::f64::consts::PI.sqrt()
        })
        .collect();

    Ok(Value::object()
        .with(
            "model",
            "EXAFS from tabulated scattering amplitudes and phases: single scattering and, \
             with angle-resolved tables, plane-wave double scattering; no \
             electronic-structure calculation",
        )
        .with("absorber", absorber)
        .with(
            "path_atoms",
            paths.iter().map(|p| p.atom).collect::<Vec<_>>(),
        )
        .with(
            "path_numbers",
            paths.iter().map(|p| p.number as usize).collect::<Vec<_>>(),
        )
        .with(
            "path_distances",
            paths.iter().map(|p| p.distance).collect::<Vec<_>>(),
        )
        .with("path_sigma2", sigma)
        .with("amplitude_reduction", s02)
        .with("mean_free_path", mean_free_path)
        .with("k", k_grid)
        .with("chi", chi)
        .with("chi_single", single)
        .with("chi_double", double)
        .with("double_scattering_paths", double_paths)
        .with("k_weight", k_weight)
        .with("chi_weighted", weighted)
        .with("r", r_grid)
        .with("transform_magnitude", transform)
        .with("warnings", warnings)
        .with("provenance", "analytic")
        .with(
            "units",
            units(&[
                ("path_distances", "bohr"),
                ("path_sigma2", "bohr^2"),
                ("amplitude_reduction", "dimensionless"),
                ("mean_free_path", "bohr"),
                ("k", "1/bohr"),
                ("chi", "dimensionless"),
                ("chi_single", "dimensionless"),
                ("chi_double", "dimensionless"),
                ("chi_weighted", "1/bohr^k_weight"),
                ("r", "bohr"),
                (
                    "transform_magnitude",
                    "1/bohr^(k_weight+1), Hann window over the k grid, no phase correction",
                ),
            ]),
        ))
}
