// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Occupation-number models for finite-k calculations.
//!
//! The electron number is a Brillouin-zone average,
//! `N = (1/Nk) sum_kn f_kn`.  Keeping this solver independent of the SCF
//! driver makes the thermodynamics testable without a crystal calculation.

use crate::error::{DftError, Result};

/// Boltzmann constant in Hartree/K (CODATA 2018 exact SI conversion).
pub const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_811_563_455_607e-6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum OccupationScheme {
    /// Integer bands at every k point (the zero-temperature insulator path).
    #[default]
    Insulator,
    /// Variational Fermi-Dirac occupations at the specified electronic
    /// temperature.  Spin populations remain separately constrained.
    FermiDirac { temperature_kelvin: f64 },
}

impl OccupationScheme {
    pub fn validate(self) -> Result<()> {
        if let Self::FermiDirac { temperature_kelvin } = self {
            if !(temperature_kelvin.is_finite() && temperature_kelvin > 0.0) {
                return Err(DftError::InvalidInput(
                    "Fermi-Dirac electronic temperature must be finite and positive".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn temperature_kelvin(self) -> f64 {
        match self {
            Self::Insulator => 0.0,
            Self::FermiDirac { temperature_kelvin } => temperature_kelvin,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OccupationResult {
    /// Occupations in `[k][band]` order, between zero and one per spin.
    pub occupations: Vec<Vec<f64>>,
    /// Chemical potential in Hartree.  It is undefined for an exactly empty
    /// or exactly full spin manifold and for integer filling.
    pub chemical_potential: Option<f64>,
    /// `T S` in Hartree per cell.  The Mermin free energy is `E - entropy_term`.
    pub entropy_term: f64,
}

fn electron_count(occupations: &[Vec<f64>]) -> f64 {
    let inv_nk = 1.0 / occupations.len() as f64;
    inv_nk * occupations.iter().flat_map(|row| row.iter()).sum::<f64>()
}

fn fermi(energy: f64, chemical_potential: f64, kbt: f64) -> f64 {
    let x = (energy - chemical_potential) / kbt;
    fermi_and_hole(x).0
}

/// Evaluate particles and holes separately so that a rounded unit occupation
/// does not erase its exponentially small, physically nonzero derivative.
pub(crate) fn fermi_and_hole(x: f64) -> (f64, f64) {
    if x >= 0.0 {
        let tail = (-x).exp();
        (tail / (1.0 + tail), 1.0 / (1.0 + tail))
    } else {
        let tail = x.exp();
        (1.0 / (1.0 + tail), tail / (1.0 + tail))
    }
}

fn solve_occupations_with_policy(
    energies: &[Vec<f64>],
    electrons: f64,
    scheme: OccupationScheme,
    require_insulating_gap: bool,
) -> Result<OccupationResult> {
    scheme.validate()?;
    if energies.is_empty() || energies.iter().any(Vec::is_empty) {
        return Err(DftError::InvalidInput(
            "occupation solver needs at least one band at every k point".to_string(),
        ));
    }
    if energies
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
        || !(electrons.is_finite() && electrons >= 0.0)
    {
        return Err(DftError::InvalidInput(
            "occupation energies and electron count must be finite, with a non-negative count"
                .to_string(),
        ));
    }
    let capacity = energies.iter().map(Vec::len).sum::<usize>() as f64 / energies.len() as f64;
    if electrons > capacity + 1e-12 {
        return Err(DftError::InvalidInput(format!(
            "{electrons} electrons requested in a spin manifold with capacity {capacity}"
        )));
    }

    if scheme == OccupationScheme::Insulator {
        let n = electrons.round();
        if (electrons - n).abs() > 1e-12 {
            return Err(DftError::InvalidInput(
                "integer occupations require an integer electron count per spin".to_string(),
            ));
        }
        let n = n as usize;
        if energies.iter().any(|row| row.len() < n) {
            return Err(DftError::InvalidInput(format!(
                "{n} occupied bands requested but a k point has fewer bands"
            )));
        }
        if require_insulating_gap && n > 0 {
            let valence_max = energies
                .iter()
                .map(|row| row[n - 1])
                .fold(f64::NEG_INFINITY, f64::max);
            let conduction_min = energies
                .iter()
                .filter_map(|row| row.get(n).copied())
                .reduce(f64::min);
            if let Some(conduction_min) = conduction_min {
                let indirect_gap = conduction_min - valence_max;
                if indirect_gap <= 1e-10 {
                    return Err(DftError::InvalidInput(format!(
                        "integer insulating occupations require a positive indirect band gap; \
                            got {indirect_gap:e} Hartree. Use Fermi-Dirac occupations for a metal \
                            or semimetal"
                    )));
                }
            }
        }
        return Ok(OccupationResult {
            occupations: energies
                .iter()
                .map(|row| {
                    (0..row.len())
                        .map(|band| if band < n { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect(),
            chemical_potential: None,
            entropy_term: 0.0,
        });
    }

    if electrons <= 1e-14 {
        return Ok(OccupationResult {
            occupations: energies.iter().map(|row| vec![0.0; row.len()]).collect(),
            chemical_potential: None,
            entropy_term: 0.0,
        });
    }
    if (capacity - electrons).abs() <= 1e-14 {
        return Ok(OccupationResult {
            occupations: energies.iter().map(|row| vec![1.0; row.len()]).collect(),
            chemical_potential: None,
            entropy_term: 0.0,
        });
    }

    let temperature = scheme.temperature_kelvin();
    let kbt = BOLTZMANN_HARTREE_PER_K * temperature;
    let min_energy = energies
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_energy = energies
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = (64.0 * kbt).max(1.0);
    let (mut low, mut high) = (min_energy - padding, max_energy + padding);
    let count_at = |mu: f64| {
        let inv_nk = 1.0 / energies.len() as f64;
        inv_nk
            * energies
                .iter()
                .flat_map(|row| row.iter())
                .map(|&energy| fermi(energy, mu, kbt))
                .sum::<f64>()
    };
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if count_at(mid) < electrons {
            low = mid;
        } else {
            high = mid;
        }
    }
    let mu = 0.5 * (low + high);
    let occupations: Vec<Vec<f64>> = energies
        .iter()
        .map(|row| row.iter().map(|&energy| fermi(energy, mu, kbt)).collect())
        .collect();
    let got = electron_count(&occupations);
    if (got - electrons).abs() > 2e-12 * (1.0 + electrons.abs()) {
        return Err(DftError::LinearAlgebra(format!(
            "Fermi-level solve did not conserve electrons: {got} vs {electrons}"
        )));
    }
    let inv_nk = 1.0 / occupations.len() as f64;
    let entropy_dimensionless = -inv_nk
        * occupations
            .iter()
            .flat_map(|row| row.iter())
            .map(|&f| {
                let a = if f > 0.0 { f * f.ln() } else { 0.0 };
                let b = if f < 1.0 {
                    (1.0 - f) * (1.0 - f).ln()
                } else {
                    0.0
                };
                a + b
            })
            .sum::<f64>();
    Ok(OccupationResult {
        occupations,
        chemical_potential: Some(mu),
        entropy_term: kbt * entropy_dimensionless,
    })
}

/// Solve one spin channel's occupations on a uniformly weighted k mesh.
///
/// The integer scheme is deliberately strict: its per-k band occupation is
/// a ground-state occupation only when the occupied manifold is separated
/// from every unoccupied state by a positive *indirect* gap.  SCF drivers may
/// temporarily cross or close that gap before reaching their fixed point;
/// they must use [`solve_occupations_during_scf`] internally and call this
/// function again for the final, user-visible state.
pub fn solve_occupations(
    energies: &[Vec<f64>],
    electrons: f64,
    scheme: OccupationScheme,
) -> Result<OccupationResult> {
    solve_occupations_with_policy(energies, electrons, scheme, true)
}

/// Internal occupation path for an unfinished SCF iterate.
///
/// Only the indirect-gap rejection is deferred. Electron conservation,
/// finite inputs, capacities, integer spin populations, and all
/// Fermi-Dirac thermodynamics remain fully validated. A converged result must
/// always be re-solved through [`solve_occupations`].
pub(crate) fn solve_occupations_during_scf(
    energies: &[Vec<f64>],
    electrons: f64,
    scheme: OccupationScheme,
) -> Result<OccupationResult> {
    solve_occupations_with_policy(energies, electrons, scheme, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fermi_dirac_conserves_electrons_and_respects_energy_shifts() {
        let energies = vec![vec![-0.4, -0.03, 0.2], vec![-0.3, 0.05, 0.4]];
        let scheme = OccupationScheme::FermiDirac {
            temperature_kelvin: 4000.0,
        };
        let a = solve_occupations(&energies, 1.5, scheme).unwrap();
        assert!((electron_count(&a.occupations) - 1.5).abs() < 2e-13);
        assert!(a.entropy_term > 0.0);
        assert!(a
            .occupations
            .iter()
            .flat_map(|row| row.iter())
            .all(|&f| f > 0.0 && f < 1.0));

        let shifted: Vec<Vec<f64>> = energies
            .iter()
            .map(|row| row.iter().map(|value| value + 2.75).collect())
            .collect();
        let b = solve_occupations(&shifted, 1.5, scheme).unwrap();
        for (left, right) in a
            .occupations
            .iter()
            .flatten()
            .zip(b.occupations.iter().flatten())
        {
            assert!((left - right).abs() < 2e-13);
        }
        assert!(
            (b.chemical_potential.unwrap() - a.chemical_potential.unwrap() - 2.75).abs() < 2e-13
        );
        assert!((a.entropy_term - b.entropy_term).abs() < 2e-14);
    }

    #[test]
    fn half_filled_particle_hole_spectrum_has_zero_chemical_potential() {
        let result = solve_occupations(
            &[vec![-0.3, -0.1, 0.1, 0.3]],
            2.0,
            OccupationScheme::FermiDirac {
                // High enough that the occupations on both sides of the
                // gap are representably non-integer; otherwise many mu
                // values inside the gap give exactly N=2 in f64.
                temperature_kelvin: 10_000.0,
            },
        )
        .unwrap();
        assert!(result.chemical_potential.unwrap().abs() < 1e-12);
        assert!((result.occupations[0][0] + result.occupations[0][3] - 1.0).abs() < 1e-13);
        assert!((result.occupations[0][1] + result.occupations[0][2] - 1.0).abs() < 1e-13);
    }

    #[test]
    fn low_temperature_limit_recovers_integer_filling() {
        let result = solve_occupations(
            &[vec![-0.8, -0.2, 0.5]],
            2.0,
            OccupationScheme::FermiDirac {
                temperature_kelvin: 1.0,
            },
        )
        .unwrap();
        assert!((result.occupations[0][0] - 1.0).abs() < 1e-14);
        assert!((result.occupations[0][1] - 1.0).abs() < 1e-14);
        assert!(result.occupations[0][2] < 1e-14);
        assert!(result.entropy_term < 1e-14);
    }

    #[test]
    fn integer_scheme_rejects_an_indirect_band_overlap() {
        let error = solve_occupations(
            &[vec![-1.0, -0.9], vec![0.2, 0.3]],
            1.0,
            OccupationScheme::Insulator,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("indirect band gap"));

        // Canonical overlap truncation can leave a different number of
        // retained bands at different k points.  A conduction state that is
        // present at only one k point still participates in the global gap.
        let unequal_ranks = solve_occupations(
            &[vec![-1.0, -0.9], vec![0.2]],
            1.0,
            OccupationScheme::Insulator,
        )
        .unwrap_err()
        .to_string();
        assert!(unequal_ranks.contains("indirect band gap"));

        solve_occupations(
            &[vec![-1.0, 0.4], vec![-0.8, 0.3]],
            1.0,
            OccupationScheme::Insulator,
        )
        .unwrap();
    }

    fn free_energy(
        energies: &[Vec<f64>],
        electrons: f64,
        temperature: f64,
    ) -> (f64, OccupationResult) {
        let result = solve_occupations(
            energies,
            electrons,
            OccupationScheme::FermiDirac {
                temperature_kelvin: temperature,
            },
        )
        .unwrap();
        let inv_nk = 1.0 / energies.len() as f64;
        let band_energy = inv_nk
            * energies
                .iter()
                .zip(&result.occupations)
                .flat_map(|(row, occupations)| row.iter().zip(occupations))
                .map(|(&energy, &occupation)| energy * occupation)
                .sum::<f64>();
        (band_energy - result.entropy_term, result)
    }

    /// These are envelope-theorem identities of the canonical Mermin
    /// minimum. They independently pin both the entropy prefactor and the
    /// Brillouin-zone weight: dF/dT = -S and dF/d epsilon_kn = f_kn/Nk.
    #[test]
    fn mermin_free_energy_has_the_exact_thermodynamic_derivatives() {
        let energies = vec![vec![-0.41, 0.06, 0.37], vec![-0.28, 0.13, 0.51]];
        let electrons = 1.35;
        let temperature = 6000.0;
        let (_, result) = free_energy(&energies, electrons, temperature);

        let dt = 0.5;
        let dfd_t = (free_energy(&energies, electrons, temperature + dt).0
            - free_energy(&energies, electrons, temperature - dt).0)
            / (2.0 * dt);
        let entropy = result.entropy_term / temperature;
        assert!(
            (dfd_t + entropy).abs() < 2e-12,
            "dF/dT={dfd_t:e}, -S={:e}",
            -entropy
        );

        let de = 1e-5;
        let mut plus = energies.clone();
        let mut minus = energies.clone();
        plus[1][0] += de;
        minus[1][0] -= de;
        let derivative = (free_energy(&plus, electrons, temperature).0
            - free_energy(&minus, electrons, temperature).0)
            / (2.0 * de);
        let expected = result.occupations[1][0] / energies.len() as f64;
        assert!(
            (derivative - expected).abs() < 2e-9,
            "dF/d epsilon={derivative:e}, f/Nk={expected:e}"
        );
    }

    /// Folding two primitive k points into one doubled-cell Gamma point
    /// doubles every extensive thermodynamic quantity while preserving the
    /// chemical potential and individual band occupations.
    #[test]
    fn mermin_thermodynamics_is_extensive_under_band_folding() {
        let primitive = vec![vec![-0.52, 0.18], vec![-0.31, 0.47]];
        let supercell = vec![primitive.iter().flatten().copied().collect::<Vec<_>>()];
        let temperature = 8000.0;
        let (f_primitive, occ_primitive) = free_energy(&primitive, 1.0, temperature);
        let (f_supercell, occ_supercell) = free_energy(&supercell, 2.0, temperature);
        assert!((f_supercell - 2.0 * f_primitive).abs() < 2e-13);
        assert!((occ_supercell.entropy_term - 2.0 * occ_primitive.entropy_term).abs() < 2e-13);
        assert!(
            (occ_supercell.chemical_potential.unwrap() - occ_primitive.chemical_potential.unwrap())
                .abs()
                < 2e-13
        );
        for (&folded, &primitive) in occ_supercell.occupations[0]
            .iter()
            .zip(occ_primitive.occupations.iter().flatten())
        {
            assert!((folded - primitive).abs() < 2e-13);
        }
    }
}
