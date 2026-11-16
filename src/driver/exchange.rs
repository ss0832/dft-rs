// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Heisenberg exchange coupling of two magnetic sites from a broken-symmetry
//! calculation, routed through the common driver.
//!
//! For `H = -2 J S_A . S_B` the high-spin determinant (all unpaired
//! electrons parallel) and the broken-symmetry determinant (the spins of
//! site B reversed) give
//!
//! ```text
//! J = -(E_HS - E_BS) / (<S^2>_HS - <S^2>_BS)
//! ```
//!
//! which interpolates between the weak- and strong-overlap limits; the two
//! limits, with `S_max = (N_A + N_B) / 2`, are reported as well:
//! `-(E_HS - E_BS) / S_max^2` and `-(E_HS - E_BS) / (S_max (S_max + 1))`.
//! The broken-symmetry determinant is converged from the high-spin density
//! with the spin density of the basis functions of site B reversed.

use super::controls::Controls;
use super::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use super::molecular::{reference_report, units};
use super::value::Value;
use super::System;
use crate::constants::HARTREE_TO_CM1;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};

fn invalid(message: &str) -> DftError {
    DftError::InvalidInput(message.to_string())
}

/// Mulliken spin population of the atoms `site`.
fn spin_population(
    density: &[Matrix; 2],
    overlap: &Matrix,
    ao_atom: &[usize],
    site: &[usize],
) -> f64 {
    let n = overlap.rows;
    let mut sum = 0.0;
    for mu in 0..n {
        if !site.contains(&ao_atom[mu]) {
            continue;
        }
        for nu in 0..n {
            sum += (density[0][(mu, nu)] - density[1][(mu, nu)]) * overlap[(nu, mu)];
        }
    }
    sum
}

/// Spin densities of the broken-symmetry determinant built from the
/// high-spin determinant. In its occupied alpha space the eigenvectors of
/// the overlap with the occupied beta space separate the doubly occupied
/// (eigenvalue near one) from the magnetic orbitals (near zero); the
/// magnetic orbitals are localized by the eigenvectors of their symmetrized
/// Mulliken population on site B, and the `n_b` most site-B orbitals change
/// spin. Both densities are projectors on orthonormal orbitals.
fn broken_symmetry_seed(
    scf: &crate::scf::ScfResult,
    on_b: &[bool],
    n_b: usize,
) -> Result<[Matrix; 2]> {
    let s = &scf.overlap;
    let n = s.rows;
    let (n_alpha, n_beta) = (scf.n_alpha, scf.n_beta);
    let c = &scf.mo_coeff[0];
    let mut occupied = Matrix::zeros(n, n_alpha);
    for i in 0..n {
        for j in 0..n_alpha {
            occupied[(i, j)] = c[(i, j)];
        }
    }
    let sc = s.matmul(&occupied);
    let with_beta = sc.transpose_matmul(&scf.density[1].matmul(&sc));
    let (_, split) = symmetric_eigen(&with_beta)?;
    let rotated = occupied.matmul(&split);
    let unpaired = n_alpha - n_beta;
    // Columns 0..unpaired: magnetic; the rest: doubly occupied.
    let column = |m: &Matrix, j: usize| -> Vec<f64> { (0..m.rows).map(|i| m[(i, j)]).collect() };
    let mut magnetic = Matrix::zeros(n, unpaired);
    for j in 0..unpaired {
        for (i, v) in column(&rotated, j).into_iter().enumerate() {
            magnetic[(i, j)] = v;
        }
    }
    let mut site = s.clone();
    for mu in 0..n {
        for nu in 0..n {
            let weight = 0.5 * (f64::from(u8::from(on_b[mu])) + f64::from(u8::from(on_b[nu])));
            site[(mu, nu)] *= weight;
        }
    }
    let population = magnetic.transpose_matmul(&site.matmul(&magnetic));
    let (_, localize) = symmetric_eigen(&population)?;
    let local = magnetic.matmul(&localize);
    let mut seed = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
    let mut add = |target: usize, v: &[f64]| {
        for mu in 0..n {
            for nu in 0..n {
                seed[target][(mu, nu)] += v[mu] * v[nu];
            }
        }
    };
    for j in unpaired..n_alpha {
        let v = column(&rotated, j);
        add(0, &v);
        add(1, &v);
    }
    // Ascending population on site B: the last n_b are the site-B orbitals.
    for j in 0..unpaired {
        add(usize::from(j >= unpaired - n_b), &column(&local, j));
    }
    Ok(seed)
}

pub(super) fn broken_symmetry_exchange(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = MolecularMethod::from_controls(controls)?;
    let missing = || {
        invalid(
            "broken-symmetry-exchange needs site_a and site_b: the atom indices (from 0) of the \
             two magnetic sites",
        )
    };
    let site_a = controls.opt_usize_list("site_a")?;
    let site_b = controls.opt_usize_list("site_b")?;
    let unpaired = controls.opt_usize_list("unpaired")?;
    let sites = controls.opt_usize_list("sites")?;
    let site_sizes = controls.opt_usize_list("site_sizes")?;
    controls.finish("broken-symmetry-exchange")?;
    if let (Some(flat), Some(sizes)) = (&sites, &site_sizes) {
        return multi_site_exchange(system, method, flat, sizes, unpaired.as_deref());
    }
    let site_a = site_a.ok_or_else(missing)?;
    let site_b = site_b.ok_or_else(missing)?;
    let natoms = system.numbers.len();
    if site_a.is_empty()
        || site_b.is_empty()
        || site_a.iter().chain(&site_b).any(|atom| *atom >= natoms)
        || site_a.iter().any(|atom| site_b.contains(atom))
    {
        return Err(invalid(
            "site_a and site_b are non-empty, disjoint lists of atoms of the structure",
        ));
    }
    // The multiplicity of the controls is that of the high-spin state.
    method.unrestricted = true;
    let total = method.multiplicity.saturating_sub(1);
    if total < 2 {
        return Err(invalid(
            "broken-symmetry-exchange needs the multiplicity of the high-spin state, with at \
             least two unpaired electrons",
        ));
    }
    let (n_a, n_b) =
        match &unpaired {
            Some(counts) if counts.len() == 2 && counts[0] > 0 && counts[1] > 0 => {
                if counts[0] + counts[1] != total {
                    return Err(DftError::InvalidInput(format!(
                        "unpaired = [{}, {}] does not add up to the {total} unpaired electrons of \
                     the high-spin multiplicity",
                        counts[0], counts[1]
                    )));
                }
                (counts[0], counts[1])
            }
            Some(_) => return Err(invalid(
                "unpaired holds the positive numbers of unpaired electrons of site_a and site_b",
            )),
            None if total % 2 == 0 => (total / 2, total / 2),
            None => {
                return Err(invalid(
                    "an odd number of unpaired electrons needs unpaired = [N_A, N_B]",
                ))
            }
        };
    let high_setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    let high = run_molecular_reference(&high_setup)?;
    if !high.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: high.scf.iterations,
            error: f64::NAN,
        });
    }

    // Broken-symmetry determinant: the spin density of site B reversed.
    let mut low_method = method.clone();
    low_method.multiplicity = n_a.abs_diff(n_b) + 1;
    let low_setup = prepare_molecular(&system.numbers, &system.positions, &low_method)?;
    let ao_atom = high_setup.basis.sph_ao_atoms();
    let on_b: Vec<bool> = ao_atom.iter().map(|atom| site_b.contains(atom)).collect();
    let mut seed = broken_symmetry_seed(&high.scf, &on_b, n_b)?;
    // The majority spin of the determinant is the first channel.
    let swapped = n_b > n_a;
    if swapped {
        seed.swap(0, 1);
    }
    let low = crate::scf::run_scf_with_density(
        &low_setup.molecule,
        &low_setup.basis,
        &low_setup.options,
        &low_setup.functional,
        low_setup.grid.as_ref(),
        Some(&seed),
    )?;
    if !low.converged {
        return Err(DftError::ScfNotConverged {
            iterations: low.iterations,
            error: f64::NAN,
        });
    }
    let difference = high.scf.energy - low.energy;
    let denominator = high.scf.s_squared - low.s_squared;
    if denominator.abs() < 1e-6 {
        return Err(invalid(
            "the broken-symmetry calculation returned to a determinant with the spin of the \
             high-spin state; there is no separate broken-symmetry solution from this start",
        ));
    }
    let s_max = 0.5 * total as f64;
    let coupling = -difference / denominator;
    let weak_overlap = -difference / (s_max * s_max);
    let strong_overlap = -difference / (s_max * (s_max + 1.0));
    let sign = if swapped { -1.0 } else { 1.0 };
    let high_density = &high.scf.density;
    let low_density = &low.density;
    let overlap = &high.scf.overlap;
    let mut report = reference_report(
        &high_setup,
        &high,
        "Heisenberg exchange coupling (H = -2 J S_A . S_B) from the high-spin and the \
         broken-symmetry determinant",
    )
    .with("site_a", site_a.clone())
    .with("site_b", site_b.clone())
    .with("unpaired", vec![n_a, n_b])
    .with("high_spin_energy", high.scf.energy)
    .with("broken_symmetry_energy", low.energy)
    .with("energy_difference", difference)
    .with("high_spin_s_squared", high.scf.s_squared)
    .with("broken_symmetry_s_squared", low.s_squared)
    .with("exchange_coupling", coupling)
    .with("exchange_coupling_cm1", coupling * HARTREE_TO_CM1)
    .with("exchange_coupling_weak_overlap", weak_overlap)
    .with("exchange_coupling_strong_overlap", strong_overlap)
    .with(
        "high_spin_site_populations",
        vec![
            spin_population(high_density, overlap, &ao_atom, &site_a),
            spin_population(high_density, overlap, &ao_atom, &site_b),
        ],
    )
    .with(
        "broken_symmetry_site_populations",
        vec![
            sign * spin_population(low_density, overlap, &ao_atom, &site_a),
            sign * spin_population(low_density, overlap, &ao_atom, &site_b),
        ],
    )
    .with("broken_symmetry_iterations", low.iterations)
    .with("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("high_spin_energy", "hartree"),
            ("broken_symmetry_energy", "hartree"),
            ("energy_difference", "hartree, E_HS - E_BS"),
            ("high_spin_s_squared", "hbar^2"),
            ("broken_symmetry_s_squared", "hbar^2"),
            ("exchange_coupling", "hartree, J of H = -2 J S_A . S_B"),
            ("exchange_coupling_cm1", "cm^-1"),
            ("exchange_coupling_weak_overlap", "hartree"),
            ("exchange_coupling_strong_overlap", "hartree"),
            (
                "high_spin_site_populations",
                "electrons, Mulliken spin population of [site_a, site_b]",
            ),
            (
                "broken_symmetry_site_populations",
                "electrons, Mulliken spin population of [site_a, site_b]",
            ),
        ]),
    );
    Ok(report)
}

/// Exchange couplings of several magnetic sites, `H = -2 sum_(i<j) J_ij
/// S_i . S_j`, from the high-spin determinant and the broken-symmetry
/// determinants with one or two sites reversed (site 0 always up). With the
/// site spins `S_i = N_i / 2` and the pattern `s_i = +-1` of a determinant,
/// the energies are mapped on `E = E_0 - 2 sum_(i<j) J_ij S_i S_j s_i s_j`:
/// as many determinants as unknowns, solved as one linear system. For two
/// sites this is the weak-overlap value of the two-site formula.
fn multi_site_exchange(
    system: &System,
    mut method: MolecularMethod,
    flat: &[usize],
    sizes: &[usize],
    unpaired: Option<&[usize]>,
) -> Result<Value> {
    let natoms = system.numbers.len();
    let count = sizes.len();
    if count < 2
        || sizes.contains(&0)
        || sizes.iter().sum::<usize>() != flat.len()
        || flat.iter().any(|atom| *atom >= natoms)
    {
        return Err(invalid(
            "sites lists the atoms of all magnetic sites and site_sizes the number of atoms of \
             each of at least two sites",
        ));
    }
    let mut seen = vec![false; natoms];
    for atom in flat {
        if seen[*atom] {
            return Err(invalid("the magnetic sites are disjoint"));
        }
        seen[*atom] = true;
    }
    let mut sites: Vec<Vec<usize>> = Vec::with_capacity(count);
    let mut start = 0;
    for &size in sizes {
        sites.push(flat[start..start + size].to_vec());
        start += size;
    }
    method.unrestricted = true;
    let total = method.multiplicity.saturating_sub(1);
    let electrons: Vec<usize> =
        match unpaired {
            Some(list)
                if list.len() == count
                    && list.iter().all(|n| *n > 0)
                    && list.iter().sum::<usize>() == total =>
            {
                list.to_vec()
            }
            _ => return Err(invalid(
                "multi-site exchange needs unpaired: the positive number of unpaired electrons \
                 of every site, adding up to the unpaired electrons of the high-spin multiplicity",
            )),
        };
    let high_setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    let high = run_molecular_reference(&high_setup)?;
    if !high.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: high.scf.iterations,
            error: f64::NAN,
        });
    }
    let ao_atom = high_setup.basis.sph_ao_atoms();
    // Spin patterns with site 0 up and at most two other sites reversed.
    let mut patterns: Vec<Vec<f64>> = vec![vec![1.0; count]];
    for i in 1..count {
        let mut pattern = vec![1.0; count];
        pattern[i] = -1.0;
        patterns.push(pattern);
    }
    for i in 1..count {
        for j in i + 1..count {
            let mut pattern = vec![1.0; count];
            pattern[i] = -1.0;
            pattern[j] = -1.0;
            patterns.push(pattern);
        }
    }
    let mut energies = vec![high.scf.energy];
    let mut spins = vec![high.scf.s_squared];
    for pattern in patterns.iter().skip(1) {
        let moment: f64 = pattern
            .iter()
            .zip(&electrons)
            .map(|(s, n)| s * *n as f64)
            .sum();
        let mut low_method = method.clone();
        low_method.multiplicity = moment.abs().round() as usize + 1;
        let low_setup = prepare_molecular(&system.numbers, &system.positions, &low_method)?;
        let reversed: Vec<bool> = ao_atom
            .iter()
            .map(|atom| {
                sites
                    .iter()
                    .zip(pattern)
                    .any(|(site, s)| *s < 0.0 && site.contains(atom))
            })
            .collect();
        let flipped: usize = pattern
            .iter()
            .zip(&electrons)
            .filter(|(s, _)| **s < 0.0)
            .map(|(_, n)| *n)
            .sum();
        let mut seed = broken_symmetry_seed(&high.scf, &reversed, flipped)?;
        // The majority spin of the determinant is the first channel.
        if moment < 0.0 {
            seed.swap(0, 1);
        }
        let low = crate::scf::run_scf_with_density(
            &low_setup.molecule,
            &low_setup.basis,
            &low_setup.options,
            &low_setup.functional,
            low_setup.grid.as_ref(),
            Some(&seed),
        )?;
        if !low.converged {
            return Err(DftError::ScfNotConverged {
                iterations: low.iterations,
                error: f64::NAN,
            });
        }
        energies.push(low.energy);
        spins.push(low.s_squared);
    }
    // E = E_0 - 2 sum_(i<j) J_ij (N_i / 2) (N_j / 2) s_i s_j.
    let unknowns = patterns.len();
    let mut design = Matrix::zeros(unknowns, unknowns);
    for (row, pattern) in patterns.iter().enumerate() {
        design[(row, 0)] = 1.0;
        let mut column = 1;
        for i in 0..count {
            for j in i + 1..count {
                design[(row, column)] =
                    -0.5 * electrons[i] as f64 * electrons[j] as f64 * pattern[i] * pattern[j];
                column += 1;
            }
        }
    }
    let solution = crate::linalg::solve_linear(&design, &energies)?;
    let mut couplings = vec![vec![0.0; count]; count];
    let mut column = 1;
    for i in 0..count {
        for j in i + 1..count {
            couplings[i][j] = solution[column];
            couplings[j][i] = solution[column];
            column += 1;
        }
    }
    let in_wavenumbers: Vec<Vec<f64>> = couplings
        .iter()
        .map(|row| row.iter().map(|j| j * HARTREE_TO_CM1).collect())
        .collect();
    let mut report = reference_report(
        &high_setup,
        &high,
        "Heisenberg exchange couplings (H = -2 sum J_ij S_i . S_j) of several sites from the \
         high-spin and the broken-symmetry determinants with one or two sites reversed",
    )
    .with("sites", sites.clone())
    .with("unpaired", electrons.clone())
    .with("spin_patterns", patterns.clone())
    .with("determinant_energies", energies.clone())
    .with("determinant_s_squared", spins)
    .with("reference_energy", solution[0])
    .with("exchange_couplings", couplings)
    .with("exchange_couplings_cm1", in_wavenumbers)
    .with("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            (
                "spin_patterns",
                "dimensionless, +1 or -1 per site and determinant",
            ),
            ("determinant_energies", "hartree"),
            ("determinant_s_squared", "hbar^2"),
            ("reference_energy", "hartree, spin-independent part"),
            (
                "exchange_couplings",
                "hartree, J_ij of H = -2 sum J_ij S_i . S_j",
            ),
            ("exchange_couplings_cm1", "cm^-1"),
        ]),
    );
    Ok(report)
}
