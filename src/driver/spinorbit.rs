// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-orbit coupling on top of a collinear Kohn-Sham reference, routed
//! through the common driver.
//!
//! The one-electron spin-orbit operator
//! `H_SO = (alpha^2 / 2) sum_A Z_A (l_A . s) / r_A^3` is added to the
//! converged Kohn-Sham operator `F_alpha (+) F_beta` whose spin axis points
//! along a chosen direction, and the two-component Hamiltonian is
//! diagonalized once in the basis of the Kohn-Sham spin orbitals (no
//! self-consistency in the spin-orbit term). `Z_A` are the nuclear charges
//! unless `soc_charges` gives effective ones; the two-electron spin-orbit
//! terms are not included.
//!
//! The band energy of a spin axis is the sum of the lowest `N` spinor
//! energies. By the magnetic force theorem the difference of the band
//! energies of two axes estimates the magnetic anisotropy energy.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::constants::FINE_STRUCTURE;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::properties::nmr::paramagnetic_operator;
use crate::response::spinor::MolecularSpinorFunctionalModel;
use crate::{DftError, Result};

/// Two unit vectors that complete `axis` to a right-handed frame
/// `(e1, e2, axis)`.
fn frame(axis: [f64; 3]) -> [[f64; 3]; 3] {
    let helper = if axis[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut e1 = cross(helper, axis);
    let length = norm(e1);
    for value in e1.iter_mut() {
        *value /= length;
    }
    let e2 = cross(axis, e1);
    [e1, e2, axis]
}

/// Spinor energies for the spin axis `axis`: `orbital[l]` is the real
/// antisymmetric matrix of `(alpha^2 / 2) sum_A Z_A (r_A x grad)_l / r_A^3`,
/// whose spin-orbit operator is `sum_l (-i orbital_l) s_l`.
fn spinor_energies(
    energies: [&[f64]; 2],
    coefficients: [&Matrix; 2],
    orbital: &[Matrix; 3],
    axis: [f64; 3],
) -> Result<Vec<f64>> {
    let triad = frame(axis);
    // Components of the operator along the spin frame.
    let along = |m: usize| -> Matrix {
        let mut out = Matrix::zeros(orbital[0].rows, orbital[0].cols);
        for (l, matrix) in orbital.iter().enumerate() {
            for (t, s) in out.as_mut_slice().iter_mut().zip(matrix.as_slice()) {
                *t += triad[m][l] * s;
            }
        }
        out
    };
    let (bx, by, bz) = (along(0), along(1), along(2));
    let (ca, cb) = (coefficients[0], coefficients[1]);
    let (na, nb) = (ca.cols, cb.cols);
    let n = na + nb;
    // H = A + i B in the basis (alpha orbitals, beta orbitals).
    let mut a = Matrix::zeros(n, n);
    let mut b = Matrix::zeros(n, n);
    for (p, e) in energies[0].iter().enumerate().take(na) {
        a[(p, p)] = *e;
    }
    for (p, e) in energies[1].iter().enumerate().take(nb) {
        a[(na + p, na + p)] = *e;
    }
    let up_up = ca.transpose_matmul(&bz.matmul(ca));
    let down_down = cb.transpose_matmul(&bz.matmul(cb));
    let mixed_x = ca.transpose_matmul(&bx.matmul(cb));
    let mixed_y = ca.transpose_matmul(&by.matmul(cb));
    for p in 0..na {
        for q in 0..na {
            b[(p, q)] = -0.5 * up_up[(p, q)];
        }
    }
    for p in 0..nb {
        for q in 0..nb {
            b[(na + p, na + q)] = 0.5 * down_down[(p, q)];
        }
    }
    for p in 0..na {
        for q in 0..nb {
            // Up-down block: -(i/2) b_x - (1/2) b_y, and its adjoint.
            a[(p, na + q)] = -0.5 * mixed_y[(p, q)];
            a[(na + q, p)] = -0.5 * mixed_y[(p, q)];
            b[(p, na + q)] = -0.5 * mixed_x[(p, q)];
            b[(na + q, p)] = 0.5 * mixed_x[(p, q)];
        }
    }
    // Real symmetric form [[A, -B], [B, A]] of the Hermitian matrix; every
    // eigenvalue appears twice.
    let mut embedded = Matrix::zeros(2 * n, 2 * n);
    for p in 0..n {
        for q in 0..n {
            embedded[(p, q)] = a[(p, q)];
            embedded[(n + p, n + q)] = a[(p, q)];
            embedded[(p, n + q)] = -b[(p, q)];
            embedded[(n + p, q)] = b[(p, q)];
        }
    }
    let (values, _) = symmetric_eigen(&embedded)?;
    Ok(values.into_iter().step_by(2).collect())
}

/// Components of the spin-orbit matrices along the spin frame of `axis`,
/// scaled.
fn spin_frame_operator(orbital: &[Matrix; 3], axis: [f64; 3], scale: f64) -> [Matrix; 3] {
    let triad = frame(axis);
    std::array::from_fn(|m| {
        let mut out = Matrix::zeros(orbital[0].rows, orbital[0].cols);
        for (l, matrix) in orbital.iter().enumerate() {
            for (t, s) in out.as_mut_slice().iter_mut().zip(matrix.as_slice()) {
                *t += scale * triad[m][l] * s;
            }
        }
        out
    })
}

/// Frequency-dependent polarizability of the self-consistent two-component
/// state with the one-electron spin-orbit operator in its Fock matrix.
pub(super) fn spinor_polarizability(system: &System, controls: &Controls) -> Result<Value> {
    use crate::response::complex::Complex;
    use crate::response::spinor::{spin_orbit_spinor_reference, spinor_dynamic_polarizability};
    let config = super::vibrational::response_config(controls)?;
    let soc_charges = controls.opt_f64_list("soc_charges")?;
    let soc_scale = controls.f64("soc_scale", 1.0)?;
    let direction = controls
        .opt_vec3("spin_axis")?
        .map_or([0.0, 0.0, 1.0], |v| v.to_array());
    let frequencies = controls.opt_f64_list("frequencies")?.ok_or_else(|| {
        DftError::InvalidInput(
            "spinor-polarizability needs frequencies: the photon energies in hartree (0 for \
             the static limit)"
                .into(),
        )
    })?;
    let damping = controls.f64("frequency_damping", 0.0)?;
    let setup = setup(system, controls)?;
    controls.finish("spinor-polarizability")?;
    let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if frequencies.is_empty()
        || frequencies.iter().any(|w| !w.is_finite())
        || !(damping.is_finite() && damping >= 0.0)
        || !soc_scale.is_finite()
        || !(length.is_finite() && length > 0.0)
    {
        return Err(DftError::InvalidInput(
            "spinor-polarizability needs finite frequencies, a non-negative \
             frequency_damping, a finite soc_scale and a non-zero spin_axis"
                .into(),
        ));
    }
    let axis = direction.map(|v| v / length);
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if scf.electronic_temperature > 0.0 {
        return Err(DftError::Unsupported {
            feature: "`spinor-polarizability` of a reference with fractional occupations".into(),
            missing: "the occupation response of the spinors at a finite electronic temperature"
                .into(),
        });
    }
    if !setup.basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "`spinor-polarizability` with effective core potentials".into(),
            missing: "the spin-orbit part of the core potential".into(),
        });
    }
    let natoms = setup.molecule.len();
    let charges: Vec<f64> = match soc_charges {
        Some(values) if values.len() == natoms && values.iter().all(|v| v.is_finite()) => values,
        Some(values) => {
            return Err(DftError::InvalidInput(format!(
                "soc_charges has {} values for {natoms} atoms; give one finite effective charge \
                 per atom",
                values.len()
            )))
        }
        None => setup.molecule.atoms.iter().map(|a| a.z as f64).collect(),
    };
    // (alpha^2 / 2) sum_A Z_A (r_A x grad) / r_A^3.
    let n = scf.overlap.rows;
    let mut orbital: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    let prefactor = 0.5 * FINE_STRUCTURE * FINE_STRUCTURE;
    for (atom, charge) in setup.molecule.atoms.iter().zip(&charges) {
        if *charge == 0.0 {
            continue;
        }
        let operator = paramagnetic_operator(&setup.basis, atom.position);
        for (target, source) in orbital.iter_mut().zip(&operator) {
            for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *t += prefactor * charge * s;
            }
        }
    }
    let spinor = spin_orbit_spinor_reference(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        &spin_frame_operator(&orbital, axis, soc_scale),
        &setup.nested_soc_scf_config(controls),
    )?;
    let points: Vec<Complex> = frequencies
        .iter()
        .map(|w| Complex::new(*w, damping))
        .collect();
    let results = spinor_dynamic_polarizability(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &setup.options,
        &spinor,
        &points,
        &config,
    )?;
    let triad = frame(axis);
    let spin: Vec<f64> = (0..3)
        .map(|l| (0..3).map(|m| spinor.spin[m] * triad[m][l]).sum::<f64>())
        .collect();
    let real: Vec<[[f64; 3]; 3]> = results
        .iter()
        .map(|r| r.alpha.map(|row| row.map(|z| z.re)))
        .collect();
    let imag: Vec<[[f64; 3]; 3]> = results
        .iter()
        .map(|r| r.alpha.map(|row| row.map(|z| z.im)))
        .collect();
    let mut axial = Vec::with_capacity(results.len());
    let mut hermiticity = Vec::with_capacity(results.len());
    let mut residuals = Vec::with_capacity(results.len());
    let mut iterations = Vec::with_capacity(results.len());
    for r in &results {
        let mut vector = [0.0; 3];
        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            vector[c] = 0.5 * (r.alpha[a][b].im - r.alpha[b][a].im);
        }
        let mut defect = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                defect = defect.max((r.alpha[a][b] - r.alpha[b][a].conj()).abs());
            }
        }
        axial.push(vector);
        hermiticity.push(defect);
        residuals.push(r.residuals.iter().copied().fold(0.0_f64, f64::max));
        iterations.push(r.iterations.to_vec());
    }
    Ok(reference_report(
        &setup,
        &reference,
        "dynamic polarizability of the self-consistent two-component state: the one-electron \
         spin-orbit operator in the Fock matrix of the spinor SCF, then the coupled linear \
         response of its spinors with complex amplitudes; Coulomb and exact exchange of the \
         whole spinor density, density-functional terms as the collinear functional of the \
         spin-diagonal blocks in the reference and in the kernel; no two-electron spin-orbit \
         terms",
    )
    .with("soc_charges", charges)
    .with("soc_scale", soc_scale)
    .with("spin_axis", axis.to_vec())
    .with("spinor_energy", spinor.energy)
    .with("spin_orbit_energy", spinor.energy - scf.energy)
    .with("spin_orbit_expectation", spinor.soc_energy)
    .with("spinor_energies", spinor.energies.clone())
    .with("spin_expectation", spin)
    .with("spinor_scf_iterations", spinor.iterations)
    .with("spinor_orbital_residual", spinor.orbital_residual)
    .with("spinor_grid_electron_defect", spinor.grid_electron_defect)
    .with("frequencies", frequencies)
    .with("frequency_damping", damping)
    .with("polarizability_real", real)
    .with("polarizability_imag", imag)
    .with("polarizability_axial", axial)
    .with("hermiticity_defect", hermiticity)
    .with("response_residual", residuals)
    .with("response_iterations", iterations)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("soc_charges", "e"),
            ("soc_scale", "dimensionless"),
            ("spin_axis", "dimensionless, unit vector"),
            ("spinor_energy", "hartree"),
            ("spin_orbit_energy", "hartree"),
            ("spin_orbit_expectation", "hartree"),
            ("spinor_energies", "hartree"),
            (
                "spin_expectation",
                "dimensionless, spin expectation value in units of hbar",
            ),
            ("spinor_orbital_residual", "hartree"),
            ("spinor_grid_electron_defect", "electrons"),
            ("frequencies", "hartree"),
            ("frequency_damping", "hartree"),
            (
                "polarizability_real",
                "e^2*bohr^2/hartree, Re alpha_ab as [frequency][a][b]",
            ),
            (
                "polarizability_imag",
                "e^2*bohr^2/hartree, Im alpha_ab as [frequency][a][b]",
            ),
            (
                "polarizability_axial",
                "e^2*bohr^2/hartree, eps_cab Im alpha_ab / 2 as [frequency][c]",
            ),
            ("hermiticity_defect", "e^2*bohr^2/hartree"),
            ("response_residual", "dimensionless"),
        ]),
    ))
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let anisotropy = match id {
        "spin-orbit" => false,
        "magnetic-anisotropy" => true,
        other => {
            return Err(DftError::Unsupported {
                feature: format!("spin-orbit calculation `{other}`"),
                missing: "a common-driver implementation".into(),
            })
        }
    };
    let soc_charges = controls.opt_f64_list("soc_charges")?;
    let requested = controls.opt_rows("spin_axes")?;
    let generalized = id == "spin-orbit"
        && controls.peek_str("reference")?.is_some_and(|reference| {
            reference.eq_ignore_ascii_case("gks") || reference.eq_ignore_ascii_case("ghf")
        });
    let self_consistent = controls.bool("spin_orbit_self_consistent", generalized)?;
    let (spin_model, xc_tolerance) = if generalized {
        let reference = controls.str("reference", "gks")?;
        if !self_consistent {
            return Err(DftError::Unsupported {
                feature: format!("one-shot spin-orbit calculation with reference={reference}"),
                missing: "a self-consistent spinor state; enable spin_orbit_self_consistent".into(),
            });
        }
        (
            MolecularSpinorFunctionalModel::SpinCovariant,
            controls.f64("spinor_xc_tolerance", 1e-10)?,
        )
    } else {
        (MolecularSpinorFunctionalModel::SpinDiagonal, 1e-10)
    };
    let setup = setup(system, controls)?;
    controls.finish(id)?;
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if scf.electronic_temperature > 0.0 {
        return Err(DftError::Unsupported {
            feature: format!("`{id}` of a reference with fractional occupations"),
            missing: "the occupation of the spinors at a finite electronic temperature".into(),
        });
    }
    if !setup.basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: format!("`{id}` with effective core potentials"),
            missing: "the spin-orbit part of the core potential".into(),
        });
    }
    let natoms = setup.molecule.len();
    let charges: Vec<f64> = match soc_charges {
        Some(values) if values.len() == natoms && values.iter().all(|v| v.is_finite()) => values,
        Some(values) => {
            return Err(DftError::InvalidInput(format!(
                "soc_charges has {} values for {natoms} atoms; give one finite effective charge \
                 per atom",
                values.len()
            )))
        }
        None => setup.molecule.atoms.iter().map(|a| a.z as f64).collect(),
    };
    let default_axes = if anisotropy {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]
    } else {
        vec![vec![0.0, 0.0, 1.0]]
    };
    let mut axes: Vec<[f64; 3]> = Vec::new();
    for row in requested.unwrap_or(default_axes) {
        let length = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if row.len() != 3 || !(length.is_finite() && length > 0.0) {
            return Err(DftError::InvalidInput(
                "every row of spin_axes is a nonzero Cartesian direction".into(),
            ));
        }
        axes.push([row[0] / length, row[1] / length, row[2] / length]);
    }
    if axes.is_empty() {
        return Err(DftError::InvalidInput(
            "spin_axes holds at least one direction".into(),
        ));
    }
    // (alpha^2 / 2) sum_A Z_A (r_A x grad) / r_A^3.
    let n = scf.overlap.rows;
    let mut orbital: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    let scale = 0.5 * FINE_STRUCTURE * FINE_STRUCTURE;
    for (atom, charge) in setup.molecule.atoms.iter().zip(&charges) {
        if *charge == 0.0 {
            continue;
        }
        let operator = paramagnetic_operator(&setup.basis, atom.position);
        for (target, source) in orbital.iter_mut().zip(&operator) {
            for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *t += scale * charge * s;
            }
        }
    }
    let electrons = scf.n_alpha + scf.n_beta;
    let collinear: f64 = scf.mo_energy[0][..scf.n_alpha].iter().sum::<f64>()
        + scf.mo_energy[1][..scf.n_beta].iter().sum::<f64>();
    let mut band_energies = Vec::with_capacity(axes.len());
    let mut first_spinors = Vec::new();
    for (index, axis) in axes.iter().enumerate() {
        let spinors = spinor_energies(
            [&scf.mo_energy[0], &scf.mo_energy[1]],
            [&scf.mo_coeff[0], &scf.mo_coeff[1]],
            &orbital,
            *axis,
        )?;
        band_energies.push(spinors.iter().take(electrons).sum::<f64>());
        if index == 0 {
            first_spinors = spinors;
        }
    }
    // The self-consistent two-component state of every spin axis, with the
    // same operator in its Fock matrix.
    let mut self_consistent_energies = Vec::new();
    let mut self_consistent_residuals = Vec::new();
    if self_consistent {
        for axis in &axes {
            let spinor = crate::response::spinor::spin_orbit_spinor_reference_with_model(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                &spin_frame_operator(&orbital, *axis, 1.0),
                &setup.nested_soc_scf_config(controls),
                spin_model,
                xc_tolerance,
            )?;
            self_consistent_energies.push(spinor.energy);
            self_consistent_residuals.push(spinor.orbital_residual);
        }
    }
    let lowest = band_energies.iter().copied().fold(f64::INFINITY, f64::min);
    let mut report = reference_report(
        &setup,
        &reference,
        "one-shot spin-orbit coupling: the one-electron spin-orbit operator added to the \
         converged collinear Kohn-Sham operator and diagonalized in the Kohn-Sham spin orbitals",
    )
    .with("soc_charges", charges)
    .with(
        "spin_axes",
        axes.iter().map(|a| a.to_vec()).collect::<Vec<_>>(),
    )
    .with("band_energies", band_energies.clone())
    .with(
        "spin_orbit_energies",
        band_energies
            .iter()
            .map(|e| e - collinear)
            .collect::<Vec<_>>(),
    )
    .with("spinor_energies", first_spinors)
    .with("provenance", "analytic");
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        ("soc_charges", "e"),
        ("spin_axes", "dimensionless, unit vectors"),
        (
            "band_energies",
            "hartree, sum of the occupied spinor energies per spin axis",
        ),
        (
            "spin_orbit_energies",
            "hartree, band energy minus the sum of the occupied Kohn-Sham energies",
        ),
        ("spinor_energies", "hartree, for the first spin axis"),
    ];
    if self_consistent {
        report.set(
            "self_consistent_xc_model",
            if generalized {
                "spin-covariant"
            } else {
                "spin-diagonal"
            },
        );
        if generalized {
            report.set("reference_type", "gks");
            report.set("model", "self-consistent molecular spin-covariant GKS with a generated one-electron spin-orbit operator, initialized from the collinear reference");
        }
        report.set(
            "self_consistent_orbital_residuals",
            self_consistent_residuals,
        );
        unit_pairs.push(("self_consistent_orbital_residuals", "hartree"));
        report.set("self_consistent_energies", self_consistent_energies.clone());
        report.set(
            "self_consistent_spin_orbit_energies",
            self_consistent_energies
                .iter()
                .map(|e| e - scf.energy)
                .collect::<Vec<_>>(),
        );
        unit_pairs.push((
            "self_consistent_energies",
            "hartree, energy of the self-consistent spinor state per spin axis",
        ));
        unit_pairs.push((
            "self_consistent_spin_orbit_energies",
            "hartree, self-consistent spinor energy minus the collinear energy",
        ));
        if anisotropy {
            let easiest = self_consistent_energies
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            report.set(
                "self_consistent_anisotropy_energies",
                self_consistent_energies
                    .iter()
                    .map(|e| e - easiest)
                    .collect::<Vec<_>>(),
            );
            unit_pairs.push((
                "self_consistent_anisotropy_energies",
                "hartree, self-consistent spinor energy relative to the easiest spin axis",
            ));
        }
    }
    if anisotropy {
        report.set(
            "anisotropy_energies",
            band_energies.iter().map(|e| e - lowest).collect::<Vec<_>>(),
        );
        unit_pairs.push((
            "anisotropy_energies",
            "hartree, band energy relative to the easiest spin axis (force theorem)",
        ));
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}
