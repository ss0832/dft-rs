// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Quasi-harmonic lattice thermodynamics on a volume grid.
//!
//! At every temperature the Helmholtz free energy per cell
//! `F(V, T) = E(V) + F_vib(V, T)` on the grid is fitted by the third-order
//! Birch-Murnaghan form of [`super::eos`]; its minimum gives `V(T)` and
//! `B_T = V d2F/dV2`. The vibrational entropy and heat capacity are fitted by
//! least-squares quadratics in `V`, so that
//!
//! ```text
//! alpha = (dS/dV)_T / B_T          (dP/dT at constant V over B_T)
//! C_p   = C_v + V T alpha^2 B_T
//! gamma = alpha B_T V / C_v        (thermodynamic Grueneisen parameter)
//! ```
//!
//! These are properties of the fitted volume-grid model; the grid must
//! bracket every `V(T)`.
//!
//! At an external pressure `p` the state is the point of the fitted isotherm
//! with `-dF/dV = p`, and the Gibbs free energy `F + p V` is reported.
//! [`eigenvalue_volume_derivatives`] gives the volume derivatives of the
//! phonon eigenvalues from dynamical matrices on the same volume grid, from
//! which the mode Grueneisen parameters follow.

use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::periodic::eos::{fit_birch_murnaghan, fit_birch_murnaghan_isotherm, BirchMurnaghan};

/// Quasi-harmonic state at one temperature (atomic units, per cell).
#[derive(Clone, Debug)]
pub struct QuasiHarmonicState {
    pub temperature: f64,
    /// External pressure of the state (Hartree/Bohr^3).
    pub pressure: f64,
    /// Equilibrium volume at the pressure (Bohr^3).
    pub volume: f64,
    /// Helmholtz free energy at `V(T)` (Hartree).
    pub free_energy: f64,
    /// Gibbs free energy `F + p V` at the same volume (Hartree).
    pub gibbs_energy: f64,
    /// Isothermal bulk modulus (Hartree/Bohr^3).
    pub bulk_modulus: f64,
    /// Volumetric thermal expansion coefficient (1/K).
    pub thermal_expansion: f64,
    /// Vibrational entropy at `V(T)` (Hartree/K).
    pub entropy: f64,
    /// Heat capacities at constant volume and pressure (Hartree/K).
    pub heat_capacity_v: f64,
    pub heat_capacity_p: f64,
    /// Thermodynamic Grueneisen parameter.
    pub gruneisen: f64,
    /// Root-mean-square residual of the free-energy fit (Hartree).
    pub residual_rms: f64,
}

/// Least-squares quadratic `c0 + c1 x + c2 x^2` in `x = V/V_r - 1`.
fn quadratic(volumes: &[f64], values: &[f64], reference: f64) -> Result<[f64; 3]> {
    let mut normal = crate::linalg::Matrix::zeros(3, 3);
    let mut rhs = vec![0.0; 3];
    for (v, y) in volumes.iter().zip(values) {
        let x = v / reference - 1.0;
        let p = [1.0, x, x * x];
        for i in 0..3 {
            rhs[i] += p[i] * y;
            for j in 0..3 {
                normal[(i, j)] += p[i] * p[j];
            }
        }
    }
    let c = crate::linalg::solve_linear(&normal, &rhs)?;
    Ok([c[0], c[1], c[2]])
}

/// Volume in `[lowest, highest]` at which the fitted pressure equals
/// `pressure`, by bisection; `None` when the interval does not bracket it.
fn volume_at_pressure(
    fit: &BirchMurnaghan,
    pressure: f64,
    lowest: f64,
    highest: f64,
) -> Option<f64> {
    let excess = |v: f64| fit.pressure(v) - pressure;
    if excess(lowest) < 0.0 || excess(highest) > 0.0 {
        return None;
    }
    let (mut low, mut high) = (lowest, highest);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if excess(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    Some(0.5 * (low + high))
}

/// Quasi-harmonic states at zero pressure; see
/// [`quasi_harmonic_at_pressure`].
pub fn quasi_harmonic(
    volumes: &[f64],
    energies: &[f64],
    temperatures: &[f64],
    free_energies: &[Vec<f64>],
    entropies: &[Vec<f64>],
    heat_capacities: &[Vec<f64>],
) -> Result<Vec<QuasiHarmonicState>> {
    quasi_harmonic_at_pressure(
        volumes,
        energies,
        temperatures,
        free_energies,
        entropies,
        heat_capacities,
        0.0,
    )
}

/// Quasi-harmonic states from static energies `energies[v]` and vibrational
/// free energies, entropies and heat capacities `[v][t]` on `volumes`, at
/// the external pressure `pressure` (Hartree/Bohr^3).
pub fn quasi_harmonic_at_pressure(
    volumes: &[f64],
    energies: &[f64],
    temperatures: &[f64],
    free_energies: &[Vec<f64>],
    entropies: &[Vec<f64>],
    heat_capacities: &[Vec<f64>],
    pressure: f64,
) -> Result<Vec<QuasiHarmonicState>> {
    let n = volumes.len();
    if n < 4
        || energies.len() != n
        || [free_energies, entropies, heat_capacities]
            .iter()
            .any(|a| a.len() != n || a.iter().any(|r| r.len() != temperatures.len()))
    {
        return Err(DftError::InvalidInput(
            "the quasi-harmonic model needs at least four volumes with one energy and one value per temperature".into(),
        ));
    }
    let reference = volumes.iter().sum::<f64>() / n as f64;
    let (lowest, highest) = volumes
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(a, b), v| (a.min(*v), b.max(*v)));
    temperatures
        .iter()
        .enumerate()
        .map(|(t, &temperature)| {
            let total: Vec<f64> = (0..n).map(|v| energies[v] + free_energies[v][t]).collect();
            // Zero pressure keeps the stationary point of the fit itself;
            // otherwise the isobar `-dF/dV = p` is located by bisection on
            // an isotherm that needs no minimum of its own.
            let fit = if pressure == 0.0 {
                fit_birch_murnaghan(volumes, &total)?
            } else {
                fit_birch_murnaghan_isotherm(volumes, &total)?
            };
            let volume = if pressure == 0.0 {
                fit.v0
            } else {
                volume_at_pressure(&fit, pressure, lowest, highest).ok_or_else(|| {
                    DftError::InvalidInput(format!(
                        "the volume at {temperature} K and pressure {pressure} Hartree/Bohr^3 lies outside the volume grid [{lowest}, {highest}]: the fitted pressures at its ends are {} and {}",
                        fit.pressure(lowest),
                        fit.pressure(highest)
                    ))
                })?
            };
            if volume < lowest || volume > highest {
                return Err(DftError::InvalidInput(format!(
                    "the free-energy minimum at {temperature} K ({volume} Bohr^3) lies outside the volume grid [{lowest}, {highest}]"
                )));
            }
            let bulk_modulus = fit.bulk_modulus(volume);
            if bulk_modulus.is_nan() || bulk_modulus <= 0.0 {
                return Err(DftError::InvalidInput(format!(
                    "the fitted free energy at {temperature} K is not convex at {volume} Bohr^3"
                )));
            }
            let free_energy = fit.energy(volume);
            let s = quadratic(volumes, &entropies.iter().map(|r| r[t]).collect::<Vec<_>>(), reference)?;
            let c = quadratic(volumes, &heat_capacities.iter().map(|r| r[t]).collect::<Vec<_>>(), reference)?;
            let x = volume / reference - 1.0;
            let entropy = s[0] + s[1] * x + s[2] * x * x;
            let ds_dv = (s[1] + 2.0 * s[2] * x) / reference;
            let heat_capacity_v = c[0] + c[1] * x + c[2] * x * x;
            let thermal_expansion = ds_dv / bulk_modulus;
            let heat_capacity_p = heat_capacity_v + volume * temperature * thermal_expansion * thermal_expansion * bulk_modulus;
            let gruneisen = if heat_capacity_v > 0.0 {
                thermal_expansion * bulk_modulus * volume / heat_capacity_v
            } else {
                0.0
            };
            Ok(QuasiHarmonicState {
                temperature,
                pressure,
                volume,
                free_energy,
                gibbs_energy: free_energy + pressure * volume,
                bulk_modulus,
                thermal_expansion,
                entropy,
                heat_capacity_v,
                heat_capacity_p,
                gruneisen,
                residual_rms: fit.residual_rms,
            })
        })
        .collect()
}

/// Relative width of an eigenvalue cluster treated as degenerate.
const DEGENERACY_WINDOW: f64 = 1e-8;

/// Eigenvalues of a Hermitian matrix given on a volume grid and their
/// derivatives with respect to `ln V`, at the volume `reference`.
///
/// `matrices[v] = (re, im)` is the mass-weighted dynamical matrix of one
/// reduced wave vector at `volumes[v]`. Every element is fitted by a
/// least-squares quadratic in `ln(V / reference)`. The eigenvalues
/// `lambda = omega^2` of the fitted matrix and their derivatives
/// `d lambda / d ln V` follow from first-order perturbation theory, with the
/// derivative matrix diagonalized inside every degenerate eigenspace. The
/// pairs are returned in ascending order of `lambda`; the mode Grueneisen
/// parameter is `-(d lambda / d ln V) / (2 lambda)`. These are properties of
/// the fitted volume-grid model.
pub fn eigenvalue_volume_derivatives(
    volumes: &[f64],
    matrices: &[(Matrix, Matrix)],
    reference: f64,
) -> Result<Vec<(f64, f64)>> {
    let n = volumes.len();
    let dim = matrices.first().map_or(0, |m| m.0.rows);
    let square = |m: &Matrix| m.rows == dim && m.cols == dim;
    if n < 3
        || matrices.len() != n
        || dim == 0
        || reference.is_nan()
        || reference <= 0.0
        || volumes.iter().any(|v| v.is_nan() || *v <= 0.0)
        || matrices.iter().any(|(re, im)| !square(re) || !square(im))
    {
        return Err(DftError::InvalidInput(
            "eigenvalue volume derivatives need at least three positive volumes with one square Hermitian matrix each and a positive reference volume".into(),
        ));
    }
    let x: Vec<f64> = volumes.iter().map(|v| (v / reference).ln()).collect();
    let span = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if span == 0.0 {
        return Err(DftError::InvalidInput(
            "the volumes of an eigenvalue volume derivative must not all equal the reference volume".into(),
        ));
    }
    // Least-squares quadratic in t = x / span: sample weights of the value
    // and of the slope at the reference volume.
    let powers = |xv: f64| {
        let t = xv / span;
        [1.0, t, t * t]
    };
    let mut normal = Matrix::zeros(3, 3);
    for xv in &x {
        let p = powers(*xv);
        for i in 0..3 {
            for j in 0..3 {
                normal[(i, j)] += p[i] * p[j];
            }
        }
    }
    let row = |k: usize| -> Result<Vec<f64>> {
        let mut unit = vec![0.0; 3];
        unit[k] = 1.0;
        crate::linalg::solve_linear(&normal, &unit)
    };
    let (value_row, slope_row) = (row(0)?, row(1)?);
    let weight = |r: &[f64], xv: f64| {
        let p = powers(xv);
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2]
    };
    let size = 2 * dim;
    let mut value = Matrix::zeros(size, size);
    let mut slope = Matrix::zeros(size, size);
    for (xv, (re, im)) in x.iter().zip(matrices) {
        let (wv, ws) = (weight(&value_row, *xv), weight(&slope_row, *xv) / span);
        for r in 0..dim {
            for c in 0..dim {
                // Hermitian part as the real symmetric matrix [[A, -B], [B, A]].
                let a = 0.5 * (re[(r, c)] + re[(c, r)]);
                let b = 0.5 * (im[(r, c)] - im[(c, r)]);
                for (target, w) in [(&mut value, wv), (&mut slope, ws)] {
                    target[(r, c)] += w * a;
                    target[(dim + r, dim + c)] += w * a;
                    target[(r, dim + c)] -= w * b;
                    target[(dim + r, c)] += w * b;
                }
            }
        }
    }
    let (values, vectors) = crate::linalg::symmetric_eigen(&value)?;
    let mut order: Vec<usize> = (0..size).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let window = DEGENERACY_WINDOW * values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut out = Vec::with_capacity(dim);
    let mut start = 0;
    while start < size {
        let mut end = start + 1;
        while end < size && values[order[end]] - values[order[end - 1]] <= window {
            end += 1;
        }
        let members = &order[start..end];
        // Every eigenvalue of the Hermitian matrix appears twice.
        if members.len() % 2 != 0 {
            return Err(DftError::LinearAlgebra(
                "an eigenvalue of the real form of a Hermitian matrix is not doubled".into(),
            ));
        }
        // The derivative matrix inside the eigenspace.
        let mut inner = Matrix::zeros(members.len(), members.len());
        for (p, &cp) in members.iter().enumerate() {
            for (q, &cq) in members.iter().enumerate().skip(p) {
                let mut sum = 0.0;
                for r in 0..size {
                    for c in 0..size {
                        sum += vectors[(r, cp)] * slope[(r, c)] * vectors[(c, cq)];
                    }
                }
                inner[(p, q)] = sum;
                inner[(q, p)] = sum;
            }
        }
        let (mut derivatives, _) = crate::linalg::symmetric_eigen(&inner)?;
        derivatives.sort_by(|a, b| a.total_cmp(b));
        let centre = members.iter().map(|&m| values[m]).sum::<f64>() / members.len() as f64;
        for pair in derivatives.chunks(2) {
            out.push((centre, 0.5 * (pair[0] + pair[1])));
        }
        start = end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Debye-like model with exactly known derivatives: static
    /// Birch-Murnaghan energy plus Einstein oscillators whose frequency
    /// scales as `V^-gamma`. The thermal expansion of the model follows
    /// `alpha = gamma C_v / (B_T V)`, the Grueneisen relation.
    #[test]
    fn einstein_solid_satisfies_the_gruneisen_relation() {
        let kb = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
        let (e0, v0, b0, bp) = (-10.0, 300.0, 0.004, 4.5);
        let (gamma, w0, modes) = (1.6, 0.0012, 3.0);
        let bm3 = |v: f64| {
            let x = (v0 / v).powf(2.0 / 3.0) - 1.0;
            e0 + 9.0 * v0 * b0 / 16.0
                * (x * x * x * bp + x * x * (6.0 - 4.0 * (v0 / v).powf(2.0 / 3.0)))
        };
        let volumes: Vec<f64> = (0..9).map(|i| v0 * (0.96 + 0.01 * i as f64)).collect();
        let temperatures = [0.0, 100.0, 300.0];
        let thermo = |v: f64, t: f64| -> (f64, f64, f64) {
            let w = w0 * (v / v0).powf(-gamma);
            if t == 0.0 {
                return (modes * 0.5 * w, 0.0, 0.0);
            }
            let x = w / (kb * t);
            let em = (-x).exp();
            let f = modes * (0.5 * w + kb * t * (1.0 - em).ln());
            let s = modes * kb * (x * em / (1.0 - em) - (1.0 - em).ln());
            let c = modes * kb * x * x * em / ((1.0 - em) * (1.0 - em));
            (f, s, c)
        };
        let energies: Vec<f64> = volumes.iter().map(|v| bm3(*v)).collect();
        let table = |k: usize| -> Vec<Vec<f64>> {
            volumes
                .iter()
                .map(|v| {
                    temperatures
                        .iter()
                        .map(|t| [thermo(*v, *t).0, thermo(*v, *t).1, thermo(*v, *t).2][k])
                        .collect()
                })
                .collect()
        };
        let states = quasi_harmonic(
            &volumes,
            &energies,
            &temperatures,
            &table(0),
            &table(1),
            &table(2),
        )
        .unwrap();
        for state in &states[1..] {
            // Gruneisen relation of the model at its own V(T).
            let relation = gamma * state.heat_capacity_v / (state.bulk_modulus * state.volume);
            println!(
                "T {} K: V {} alpha {:e} gamma-relation {:e} gamma {}",
                state.temperature, state.volume, state.thermal_expansion, relation, state.gruneisen
            );
            assert!((state.thermal_expansion - relation).abs() < 2e-3 * relation);
            assert!((state.gruneisen - gamma).abs() < 2e-3 * gamma);
            assert!(state.heat_capacity_p > state.heat_capacity_v);
        }
        assert!(states[2].volume > states[1].volume && states[1].volume > states[0].volume);
    }

    /// Log-log least squares `log|e| = log C + p log h`: exponent, R^2.
    fn power_fit(h: &[f64], errors: &[f64]) -> (f64, f64) {
        let x: Vec<f64> = h.iter().map(|v| v.ln()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.abs().ln()).collect();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// The model of `einstein_solid_satisfies_the_gruneisen_relation`:
    /// `(E0, V0, B0, B0')` of the static energy and `(gamma, omega0, modes)`
    /// of the oscillators.
    const STATIC: [f64; 4] = [-10.0, 300.0, 0.004, 4.5];
    const OSCILLATOR: [f64; 3] = [1.6, 0.0012, 3.0];

    fn model_static(v: f64) -> f64 {
        let [e0, v0, b0, bp] = STATIC;
        let s = (v0 / v).powf(2.0 / 3.0);
        let x = s - 1.0;
        e0 + 9.0 * v0 * b0 / 16.0 * (x * x * x * bp + x * x * (6.0 - 4.0 * s))
    }

    /// Vibrational free energy, entropy, heat capacity and internal energy.
    fn model_vibration(v: f64, t: f64) -> [f64; 4] {
        let kb = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
        let [gamma, w0, modes] = OSCILLATOR;
        let w = w0 * (v / STATIC[1]).powf(-gamma);
        if t == 0.0 {
            return [modes * 0.5 * w, 0.0, 0.0, modes * 0.5 * w];
        }
        let x = w / (kb * t);
        let em = (-x).exp();
        [
            modes * (0.5 * w + kb * t * (1.0 - em).ln()),
            modes * kb * (x * em / (1.0 - em) - (1.0 - em).ln()),
            modes * kb * x * x * em / ((1.0 - em) * (1.0 - em)),
            modes * w * (0.5 + em / (1.0 - em)),
        ]
    }

    /// Static energies and the three `[v][t]` tables of the model.
    fn model_tables(volumes: &[f64], temperatures: &[f64]) -> (Vec<f64>, [Vec<Vec<f64>>; 3]) {
        let table = |k: usize| -> Vec<Vec<f64>> {
            volumes
                .iter()
                .map(|v| {
                    temperatures
                        .iter()
                        .map(|t| model_vibration(*v, *t)[k])
                        .collect()
                })
                .collect()
        };
        (
            volumes.iter().map(|v| model_static(*v)).collect(),
            [table(0), table(1), table(2)],
        )
    }

    #[test]
    fn pressure_moves_the_state_onto_the_model_isobar() {
        let [_, v0, b0, bp] = STATIC;
        let gamma = OSCILLATOR[0];
        let volumes: Vec<f64> = (0..9).map(|i| v0 * (0.95 + 0.01 * i as f64)).collect();
        let temperatures = [0.0, 100.0, 300.0];
        let pressure: f64 = 8.0e-5;
        let (energies, tables) = model_tables(&volumes, &temperatures);
        let at = |p: f64| {
            quasi_harmonic_at_pressure(
                &volumes,
                &energies,
                &temperatures,
                &tables[0],
                &tables[1],
                &tables[2],
                p,
            )
        };
        let (free, loaded) = (at(0.0).unwrap(), at(pressure).unwrap());
        for (zero, state) in free.iter().zip(&loaded) {
            // Closed-form pressure of the model at the returned volume.
            let ratio = (v0 / state.volume).cbrt();
            let static_pressure = 1.5
                * b0
                * (ratio.powi(7) - ratio.powi(5))
                * (1.0 + 0.75 * (bp - 4.0) * (ratio * ratio - 1.0));
            let thermal_pressure =
                gamma * model_vibration(state.volume, state.temperature)[3] / state.volume;
            let model = static_pressure + thermal_pressure;
            println!(
                "T {} K: V {} -> {} Bohr^3, model pressure {:e} (target {:e}), B_T {:e} -> {:e}, gamma {}",
                state.temperature,
                zero.volume,
                state.volume,
                model,
                pressure,
                zero.bulk_modulus,
                state.bulk_modulus,
                state.gruneisen
            );
            assert!((model - pressure).abs() < 1e-5 * pressure);
            assert_eq!(state.pressure, pressure);
            assert!(state.volume < zero.volume && state.bulk_modulus > zero.bulk_modulus);
            assert!(
                (state.gibbs_energy - state.free_energy - pressure * state.volume).abs() < 1e-14
            );
            assert_eq!(zero.gibbs_energy, zero.free_energy);
            if state.temperature > 0.0 {
                assert!((state.gruneisen - gamma).abs() < 2e-3 * gamma);
            }
        }
        // A grid on the compressed branch has no free-energy minimum and is
        // still a valid isotherm at the pressure.
        let compressed: Vec<f64> = (0..9).map(|i| v0 * (0.90 + 0.01 * i as f64)).collect();
        let (compressed_energies, compressed_tables) = model_tables(&compressed, &temperatures);
        let state = &quasi_harmonic_at_pressure(
            &compressed,
            &compressed_energies,
            &temperatures,
            &compressed_tables[0],
            &compressed_tables[1],
            &compressed_tables[2],
            3.0e-4,
        )
        .unwrap()[2];
        let ratio = (v0 / state.volume).cbrt();
        let model = 1.5
            * b0
            * (ratio.powi(7) - ratio.powi(5))
            * (1.0 + 0.75 * (bp - 4.0) * (ratio * ratio - 1.0))
            + gamma * model_vibration(state.volume, state.temperature)[3] / state.volume;
        println!(
            "compressed grid at 300 K: V {} Bohr^3, model pressure {model:e} (target 3e-4)",
            state.volume
        );
        assert!((model - 3.0e-4).abs() < 1e-5 * 3.0e-4);
        assert!(quasi_harmonic(
            &compressed,
            &compressed_energies,
            &temperatures,
            &compressed_tables[0],
            &compressed_tables[1],
            &compressed_tables[2],
        )
        .is_err());
        // A pressure that the grid does not bracket is refused.
        let error = at(1.0e-3).unwrap_err();
        assert!(error.to_string().contains("outside the volume grid"));
    }

    #[test]
    fn gruneisen_parameter_converges_with_the_volume_spacing() {
        let v0 = STATIC[1];
        let gamma = OSCILLATOR[0];
        let temperatures = [300.0];
        let state = |centre: f64, h: f64| {
            let volumes: Vec<f64> = (0..9)
                .map(|i| centre * (1.0 + h * (i as f64 - 4.0)))
                .collect();
            let (energies, tables) = model_tables(&volumes, &temperatures);
            quasi_harmonic(
                &volumes,
                &energies,
                &temperatures,
                &tables[0],
                &tables[1],
                &tables[2],
            )
            .unwrap()
            .remove(0)
        };
        // Every grid is centred on the equilibrium volume, so that the
        // spacing is the only length of the fit.
        let centre = state(v0 * 1.005, 0.00125).volume;
        let steps: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| state(centre, h).gruneisen - gamma)
            .collect();
        let (p, r2) = power_fit(&steps, &errors);
        println!(
            "thermodynamic Grueneisen parameter at 300 K, centre {centre} Bohr^3: errors {errors:?}, relative volume spacing 0.0025..0.02, p {p:.6}, R^2 {r2:.8}"
        );
        assert!((p - 2.0).abs() < 0.1 && r2 > 0.999);
    }

    /// `U K U^+` with `U = diag(exp(i phase)) Q` and `Q` a Householder
    /// reflection: a Hermitian matrix with the eigenvalues of the real
    /// symmetric `kernel`.
    fn rotated(kernel: &[[f64; 3]; 3]) -> (Matrix, Matrix) {
        let v: [f64; 3] = [1.0, 2.0, 3.0];
        let phases: [f64; 3] = [0.0, 0.7, -1.3];
        let q = |r: usize, c: usize| (if r == c { 1.0 } else { 0.0 }) - 2.0 * v[r] * v[c] / 14.0;
        let mut re = Matrix::zeros(3, 3);
        let mut im = Matrix::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                let mut m = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        m += q(r, i) * kernel[i][j] * q(c, j);
                    }
                }
                re[(r, c)] = (phases[r] - phases[c]).cos() * m;
                im[(r, c)] = (phases[r] - phases[c]).sin() * m;
            }
        }
        (re, im)
    }

    #[test]
    fn degenerate_eigenvalue_derivatives_follow_perturbation_theory() {
        // D(x) = D0 + x D1 + x^2 D2 with x = ln(V / 100): D0 has a doubly
        // degenerate eigenvalue that D1 splits.
        let base: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        let first: [[f64; 3]; 3] = [[0.4, 0.3, 0.2], [0.3, -0.5, 0.1], [0.2, 0.1, 0.8]];
        let second: [[f64; 3]; 3] = [[0.2, -0.1, 0.0], [-0.1, 0.3, 0.05], [0.0, 0.05, -0.4]];
        let offsets: [f64; 5] = [-0.02, -0.01, 0.0, 0.01, 0.02];
        let volumes: Vec<f64> = offsets.iter().map(|x| 100.0 * x.exp()).collect();
        let matrices: Vec<(Matrix, Matrix)> = offsets
            .iter()
            .map(|x| {
                let mut kernel = [[0.0_f64; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        kernel[i][j] = base[i][j] + x * first[i][j] + x * x * second[i][j];
                    }
                }
                rotated(&kernel)
            })
            .collect();
        let found = eigenvalue_volume_derivatives(&volumes, &matrices, 100.0).unwrap();
        // Inside the degenerate space the derivative is [[0.4, 0.3], [0.3, -0.5]].
        let mean: f64 = -0.05;
        let half_gap: f64 = 0.45;
        let root = (half_gap * half_gap + 0.09_f64).sqrt();
        let expected = [(1.0, mean - root), (1.0, mean + root), (3.0, 0.8)];
        println!("eigenvalues and d/dlnV {found:?}, closed form {expected:?}");
        assert_eq!(found.len(), 3);
        for (f, e) in found.iter().zip(&expected) {
            assert!((f.0 - e.0).abs() < 1e-10 && (f.1 - e.1).abs() < 1e-9);
        }
    }

    #[test]
    fn mode_gruneisen_parameters_pass_a_four_step_fit() {
        // Eigenvalues lambda_i (V / 100)^(-2 gamma_i) on fixed eigenvectors.
        let lambda: [f64; 3] = [1.0e-6, 2.5e-6, 4.0e-6];
        let gamma: [f64; 3] = [1.9, 0.7, -0.4];
        let steps: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];
        let mut errors: Vec<Vec<f64>> = vec![Vec::new(); 3];
        for &h in &steps {
            let volumes: Vec<f64> = (-2..=2).map(|k| 100.0 * (h * k as f64).exp()).collect();
            let matrices: Vec<(Matrix, Matrix)> = volumes
                .iter()
                .map(|v| {
                    let mut kernel = [[0.0_f64; 3]; 3];
                    for i in 0..3 {
                        kernel[i][i] = lambda[i] * (v / 100.0).powf(-2.0 * gamma[i]);
                    }
                    rotated(&kernel)
                })
                .collect();
            let found = eigenvalue_volume_derivatives(&volumes, &matrices, 100.0).unwrap();
            for i in 0..3 {
                errors[i].push(-found[i].1 / (2.0 * found[i].0) - gamma[i]);
            }
        }
        for i in 0..3 {
            let (p, r2) = power_fit(&steps, &errors[i]);
            println!(
                "mode Grueneisen parameter {} (exact {}): errors {:?}, log-volume spacing 0.0025..0.02, p {p:.6}, R^2 {r2:.8}",
                i, gamma[i], errors[i]
            );
            assert!((p - 2.0).abs() < 0.02 && r2 > 0.9999);
        }
    }
}
