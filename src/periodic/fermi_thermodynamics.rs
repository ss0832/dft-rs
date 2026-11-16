// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fermi-level thermodynamics of a fixed (rigid) band structure: the
//! chemical potential at a fixed electron count, the thermally averaged
//! density of states, the bare Pauli spin susceptibility, the electronic
//! heat capacity and the entropy.
//!
//! With `x = (e - mu)/kT`, `f = 1/(1 + exp x)` and the weight `w` of one
//! level (spin degeneracy over the number of k points), the moments
//! `L_n = w sum (e - mu)^n f (1 - f) / kT` give
//!
//! - the thermally averaged density of states `L_0 = int g(e) (-df/de) de`,
//! - the bare spin susceptibility `chi = mu_B^2 L_0`: a rigid Zeeman shift
//!   `-/+ mu_B B` of the two spin channels at a fixed electron count,
//! - `d mu / d kT = -L_1 / (kT L_0)` at a fixed electron count, and
//! - the heat capacity `C = dU/dT = k_B (L_2 - L_1^2 / L_0) / kT`, which
//!   tends to the Sommerfeld value `(pi^2/3) k_B^2 T L_0` at low temperature.

use crate::error::{DftError, Result};

#[derive(Clone, Copy, Debug)]
pub struct FermiThermodynamics {
    /// Chemical potential that reproduces the electron count, Hartree.
    pub chemical_potential: f64,
    /// Electron count per cell at `chemical_potential`.
    pub electrons: f64,
    /// Thermally averaged density of states `L_0`, states per Hartree per
    /// cell.
    pub dos: f64,
    /// First moment `L_1`, dimensionless.
    pub first_moment: f64,
    /// Second moment `L_2`, Hartree.
    pub second_moment: f64,
    /// `d mu / d kT` at a fixed electron count, dimensionless.
    pub dmu_dkt: f64,
    /// Heat capacity at a fixed electron count, k_B per cell.
    pub heat_capacity: f64,
    /// Entropy, k_B per cell.
    pub entropy: f64,
    /// Band energy `w sum e f`, Hartree per cell.
    pub band_energy: f64,
}

/// Fixed-total-number thermodynamics of two explicit rigid spin bands.
/// The bands share one chemical potential, with no imposed spin reservoir
/// bias. Reoccupation of a constrained orbital reference is explicit and
/// does not constitute a new self-consistent thermal reference.
#[derive(Clone, Copy, Debug)]
pub struct CollinearFermiThermodynamics {
    pub thermal: FermiThermodynamics,
    pub spin_electrons: [f64; 2],
    pub spin_dos: [f64; 2],
    /// Effective DOS for the fixed-number longitudinal spin response.
    pub spin_constrained_dos: f64,
    /// M = -g(N_alpha-N_beta)/4, in atomic magnetic moment units.
    pub magnetization: f64,
    pub susceptibility: f64,
    /// Common d mu/dB in Hartree per atomic magnetic field. None denotes
    /// an inert retained space with undetermined common multiplier.
    pub chemical_potential_response: Option<f64>,
    pub delta_spin_numbers: [f64; 2],
    pub charge_constraint_residual: f64,
}

/// Reuse the common Fermi-level solve for both rigid spin channels, then
/// impose the common-number constraint on their opposite Zeeman sources.
/// `weight` is the weight of one explicit spin level, normally 1/N_k.
pub fn collinear_fermi_thermodynamics(
    energies: [&[f64]; 2],
    weight: f64,
    electrons: f64,
    kt: f64,
    spin_g_factor: f64,
) -> Result<CollinearFermiThermodynamics> {
    if !spin_g_factor.is_finite() || spin_g_factor <= 0.0 {
        return Err(DftError::InvalidInput(
            "spin_g_factor must be positive and finite".into(),
        ));
    }
    let combined: Vec<_> = energies
        .iter()
        .flat_map(|band| band.iter().copied())
        .collect();
    let thermal = fermi_thermodynamics(&combined, weight, electrons, kt)?;
    let mut spin_electrons = [0.0; 2];
    let mut spin_dos = [0.0; 2];
    for spin in 0..2 {
        for &energy in energies[spin] {
            let (occupation, variance, _) = level((energy - thermal.chemical_potential) / kt);
            spin_electrons[spin] += weight * occupation;
            spin_dos[spin] += weight * variance / kt;
        }
    }
    let dos = spin_dos[0] + spin_dos[1];
    let zeeman = spin_g_factor / 4.0;
    let multiplier = (dos > 0.0).then(|| zeeman * ((spin_dos[0] - spin_dos[1]) / dos));
    let spin_constrained_dos = if dos > 0.0 {
        4.0 * spin_dos[0] * (spin_dos[1] / dos)
    } else {
        0.0
    };
    let delta_spin_numbers = [
        spin_dos[0] * (multiplier.unwrap_or(0.0) - zeeman),
        spin_dos[1] * (multiplier.unwrap_or(0.0) + zeeman),
    ];
    let susceptibility = zeeman * zeeman * spin_constrained_dos;
    let magnetization = -zeeman * (spin_electrons[0] - spin_electrons[1]);
    if !susceptibility.is_finite() || !magnetization.is_finite() {
        return Err(DftError::InvalidInput(
            "unrepresentable rigid-band spin response".into(),
        ));
    }
    Ok(CollinearFermiThermodynamics {
        thermal,
        spin_electrons,
        spin_dos,
        spin_constrained_dos,
        magnetization,
        susceptibility,
        chemical_potential_response: multiplier,
        charge_constraint_residual: delta_spin_numbers.iter().sum::<f64>().abs(),
        delta_spin_numbers,
    })
}

/// Occupation `f`, the product `f (1 - f)` and the entropy term
/// `-[f ln f + (1 - f) ln(1 - f)]` of one level at `x = (e - mu)/kT`,
/// evaluated without overflow or cancellation.
pub fn level(x: f64) -> (f64, f64, f64) {
    let t = (-x.abs()).exp();
    let minority = t / (1.0 + t);
    let occupation = if x > 0.0 { minority } else { 1.0 - minority };
    let variance = t / ((1.0 + t) * (1.0 + t));
    let entropy = (1.0 + t).ln() + x.abs() * minority;
    (occupation, variance, entropy)
}

fn count(energies: &[f64], weight: f64, mu: f64, kt: f64) -> f64 {
    weight * energies.iter().map(|e| level((e - mu) / kt).0).sum::<f64>()
}

/// Fermi-level thermodynamics of the levels `energies` (every band of every
/// k point of one spin-degenerate set), each with the weight `weight`, at
/// the electron count `electrons` per cell and the temperature `kt`
/// (Hartree).
pub fn fermi_thermodynamics(
    energies: &[f64],
    weight: f64,
    electrons: f64,
    kt: f64,
) -> Result<FermiThermodynamics> {
    if energies.is_empty()
        || !weight.is_finite()
        || !(weight > 0.0)
        || !kt.is_finite()
        || !(kt > 0.0)
        || energies.iter().any(|e| !e.is_finite())
    {
        return Err(DftError::InvalidInput(
            "Fermi-level thermodynamics needs finite band energies, a positive level weight \
             and a positive temperature"
                .into(),
        ));
    }
    let capacity = weight * energies.len() as f64;
    if !(electrons > 0.0 && electrons < capacity) {
        return Err(DftError::InvalidInput(format!(
            "the electron count {electrons} is outside the open interval (0, {capacity}) of \
             the bands"
        )));
    }
    let lowest = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // The count increases with the chemical potential: widen the bracket
    // until it holds the target.
    let mut lo = lowest - 10.0 * kt;
    let mut hi = highest + 10.0 * kt;
    let mut widened = 0;
    while count(energies, weight, lo, kt) > electrons || count(energies, weight, hi, kt) < electrons
    {
        lo -= 10.0 * kt;
        hi += 10.0 * kt;
        widened += 1;
        if widened > 200 {
            return Err(DftError::InvalidInput(
                "no chemical potential reproduces the electron count".into(),
            ));
        }
    }
    // Largest chemical potential that holds too few electrons and smallest
    // one that holds too many. In a metal the two coincide; in a gap at low
    // temperature the count is flat to rounding between them and the
    // chemical potential is their midpoint.
    let (mut below, mut not_below) = (lo, hi);
    let (mut not_above, mut above) = (lo, hi);
    for _ in 0..200 {
        let mid = 0.5 * (below + not_below);
        if count(energies, weight, mid, kt) < electrons {
            below = mid;
        } else {
            not_below = mid;
        }
        let mid = 0.5 * (not_above + above);
        if count(energies, weight, mid, kt) > electrons {
            above = mid;
        } else {
            not_above = mid;
        }
    }
    let mu = 0.5 * (not_below + not_above);
    let (mut n, mut l0, mut l1, mut l2, mut entropy, mut band_energy) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &e in energies {
        let d = e - mu;
        let (f, variance, s) = level(d / kt);
        n += f;
        l0 += variance / kt;
        l1 += d * variance / kt;
        l2 += d * d * variance / kt;
        entropy += s;
        band_energy += e * f;
    }
    let (n, l0, l1, l2) = (weight * n, weight * l0, weight * l1, weight * l2);
    // In a gap at low temperature every f (1 - f) underflows: no thermal
    // response.
    let (dmu_dkt, heat_capacity) = if l0 > 0.0 {
        (-l1 / (kt * l0), (l2 - l1 * l1 / l0) / kt)
    } else {
        (0.0, 0.0)
    };
    Ok(FermiThermodynamics {
        chemical_potential: mu,
        electrons: n,
        dos: l0,
        first_moment: l1,
        second_moment: l2,
        dmu_dkt,
        heat_capacity,
        entropy: weight * entropy,
        band_energy: weight * band_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent log-log least squares for error = C h^p.
    fn fit(steps: &[f64], errors: &[f64]) -> (f64, f64) {
        assert!(steps.len() >= 4 && errors.iter().all(|e| *e > 0.0));
        let x: Vec<f64> = steps.iter().map(|h| h.ln()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.ln()).collect();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let p = sxy / sxx;
        let c = my - p * mx;
        let sse: f64 = x.iter().zip(&y).map(|(a, b)| (b - c - p * a).powi(2)).sum();
        let sst: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
        (p, 1.0 - sse / sst)
    }

    #[test]
    fn uniform_levels_reproduce_the_sommerfeld_limit() {
        // A ladder of equally spaced levels is a constant density of states
        // g0 = weight / spacing. Half filling, kT = 100 spacings, band edges
        // 100 kT away: L0 = g0 and C = S = (pi^2/3) kT g0, and the
        // chemical potential stays at the band centre.
        let spacing = 1.0e-4;
        let levels: Vec<f64> = (0..20001).map(|j| j as f64 * spacing).collect();
        let weight = 2.0;
        let g0 = weight / spacing;
        let kt = 0.01;
        let r = fermi_thermodynamics(&levels, weight, weight * 20001.0 / 2.0, kt).unwrap();
        let sommerfeld = std::f64::consts::PI.powi(2) / 3.0 * kt * g0;
        println!(
            "mu {:.12} L0/g0 {:.12} C/Sommerfeld {:.12} S/Sommerfeld {:.12}",
            r.chemical_potential,
            r.dos / g0,
            r.heat_capacity / sommerfeld,
            r.entropy / sommerfeld
        );
        assert!((r.chemical_potential - 1.0).abs() < 1e-10);
        assert!((r.dos / g0 - 1.0).abs() < 1e-9);
        assert!((r.heat_capacity / sommerfeld - 1.0).abs() < 1e-9);
        assert!((r.entropy / sommerfeld - 1.0).abs() < 1e-9);
        assert!(r.dmu_dkt.abs() < 1e-8);
        // Wilson ratio (pi^2/3) kT L0 / C of non-interacting electrons.
        let wilson = std::f64::consts::PI.powi(2) / 3.0 * kt * r.dos / r.heat_capacity;
        assert!((wilson - 1.0).abs() < 1e-9);
    }

    #[test]
    fn susceptibility_and_heat_capacity_pass_four_step_fits() {
        // A cosine band away from half filling, so that the chemical
        // potential moves with temperature (L1 is not zero).
        let nk = 400;
        let band: Vec<f64> = (0..nk)
            .map(|j| -(2.0 * std::f64::consts::PI * j as f64 / nk as f64).cos())
            .collect();
        let weight = 2.0 / nk as f64;
        let electrons = 0.6;
        let kt = 0.05;
        let r = fermi_thermodynamics(&band, weight, electrons, kt).unwrap();
        assert!((r.electrons - electrons).abs() < 1e-13);
        assert!(r.dmu_dkt.abs() > 1e-3, "{}", r.dmu_dkt);

        // Magnetization (in Bohr magnetons) under a rigid Zeeman shift b of
        // the two spin channels at the same electron count; dm/db = L0.
        let magnetization = |b: f64| -> f64 {
            let mut levels: Vec<f64> = band.iter().map(|e| e - b).collect();
            levels.extend(band.iter().map(|e| e + b));
            let half = 0.5 * weight;
            let mu = fermi_thermodynamics(&levels, half, electrons, kt)
                .unwrap()
                .chemical_potential;
            half * band
                .iter()
                .map(|e| level((e - b - mu) / kt).0 - level((e + b - mu) / kt).0)
                .sum::<f64>()
        };
        let steps: Vec<f64> = [0.1, 0.05, 0.025, 0.0125].iter().map(|s| s * kt).collect();
        let errors: Vec<f64> = steps
            .iter()
            .map(|&b| ((magnetization(b) - magnetization(-b)) / (2.0 * b) - r.dos).abs())
            .collect();
        let (p, r2) = fit(&steps, &errors);
        println!(
            "susceptibility: L0 {:.10} steps {steps:?} errors {errors:?} p={p:.6} R2={r2:.8}",
            r.dos
        );
        assert!((p - 2.0).abs() < 0.02, "p = {p}");
        assert!(r2 > 0.99999, "R2 = {r2}");

        // Heat capacity: central differences of the band energy in kT.
        let energy = |t: f64| {
            fermi_thermodynamics(&band, weight, electrons, t)
                .unwrap()
                .band_energy
        };
        let steps: Vec<f64> = [0.1, 0.05, 0.025, 0.0125].iter().map(|s| s * kt).collect();
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| ((energy(kt + h) - energy(kt - h)) / (2.0 * h) - r.heat_capacity).abs())
            .collect();
        let (p, r2) = fit(&steps, &errors);
        println!(
            "heat capacity: C {:.10} steps {steps:?} errors {errors:?} p={p:.6} R2={r2:.8}",
            r.heat_capacity
        );
        assert!((p - 2.0).abs() < 0.02, "p = {p}");
        assert!(r2 > 0.99999, "R2 = {r2}");

        // d mu / d kT against a central difference of the chemical potential.
        let mu = |t: f64| {
            fermi_thermodynamics(&band, weight, electrons, t)
                .unwrap()
                .chemical_potential
        };
        let h = 1e-4 * kt;
        let slope = (mu(kt + h) - mu(kt - h)) / (2.0 * h);
        assert!((slope - r.dmu_dkt).abs() < 1e-6, "{slope} {}", r.dmu_dkt);
    }

    #[test]
    fn invalid_requests_are_refused() {
        assert!(fermi_thermodynamics(&[], 1.0, 1.0, 0.01).is_err());
        assert!(fermi_thermodynamics(&[0.0, 1.0], 2.0, 4.0, 0.01).is_err());
        assert!(fermi_thermodynamics(&[0.0, 1.0], 2.0, 0.0, 0.01).is_err());
        assert!(fermi_thermodynamics(&[0.0, 1.0], 2.0, 1.0, 0.0).is_err());
        // An insulator at low temperature has no thermal response.
        // The chemical potential is then the middle of the gap.
        let r = fermi_thermodynamics(&[0.0, 1.0], 2.0, 2.0, 1e-4).unwrap();
        assert!(
            (r.chemical_potential - 0.5).abs() < 1e-3,
            "{}",
            r.chemical_potential
        );
        assert_eq!(r.dos, 0.0);
        assert_eq!(r.heat_capacity, 0.0);
    }
}
