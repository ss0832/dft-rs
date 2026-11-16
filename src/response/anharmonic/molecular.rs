// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular harmonic modes, analytic force fields and electric property jets.

use super::{
    vibrational_spectrum, AnharmonicConfig, AnharmonicMethod, AnharmonicSpectrum,
    ObservablePolynomial, VibrationalPolynomial,
};
use crate::basis::MolBasis;
use crate::constants::{atomic_mass_au, HARTREE_TO_CM1};
use crate::grid::MolecularGrid;
use crate::response::force_constants::analytic_force_constants;
use crate::response::nuclear::{mixed_molecular_response, MolecularDirection};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct MolecularAnharmonic {
    pub selected_mode_ids: Vec<usize>,
    pub harmonic_frequencies_cm: Vec<f64>,
    pub patterns: Vec<Vec<[f64; 3]>>,
    pub n_projected: usize,
    pub maximum_force: f64,
    pub force_permutation_defects: [f64; 2],
    pub force_permutation_comparisons: [usize; 2],
    pub polarizability_symmetry_defect: Option<f64>,
    pub polarizability_symmetry_comparisons: usize,
    pub polynomial: VibrationalPolynomial,
    pub spectrum: AnharmonicSpectrum,
}

/// Analytic integer-occupation molecular anharmonic frequencies and strengths.
/// Optional Raman property jets reach fifth energy order. Mode IDs select a
/// reduced Hamiltonian explicitly; omitted modes are not implicitly integrated.
#[allow(clippy::too_many_arguments)]
pub fn molecular_anharmonic(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    mode_ids: Option<&[usize]>,
    include_raman: bool,
    config: &AnharmonicConfig,
    response_config: &ResponseSolverConfig,
) -> Result<ResponseReport<MolecularAnharmonic>> {
    config.validate()?;
    response_config.validate()?;
    let gradient = crate::gradient::assemble_gradient(mol, mb, functional, grid, scf, opts)?;
    let maximum_force = gradient
        .iter()
        .flat_map(|g| [g.x, g.y, g.z])
        .map(f64::abs)
        .fold(0.0, f64::max);
    if maximum_force > config.equilibrium_tolerance {
        return Err(DftError::InvalidInput(format!(
            "anharmonic expansion requires a stationary molecular minimum: maximum \
            force {maximum_force:.6e} exceeds equilibrium_tolerance {:.6e} \
            Hartree/Bohr",
            config.equilibrium_tolerance
        )));
    }
    let full = crate::hessian::analytic_hessian_full(mol, mb, opts, functional, scf, grid)?;
    let harmonic = crate::properties::vibrational::vibrational_analysis(mol, &full.hessian, None)?;
    if harmonic.frequencies_cm.is_empty()
        || harmonic
            .frequencies_cm
            .iter()
            .any(|&w| w <= 0.0 || !w.is_finite())
    {
        return Err(DftError::InvalidInput(
            "anharmonic oscillator expansion requires a positive vibrational Hessian \
            after translation/rotation projection"
                .into(),
        ));
    }
    let selected = mode_ids.map_or_else(
        || (0..harmonic.frequencies_cm.len()).collect::<Vec<_>>(),
        |ids| ids.to_vec(),
    );
    let mut sorted = selected.clone();
    sorted.sort_unstable();
    if selected.is_empty()
        || sorted.iter().any(|&i| i >= harmonic.frequencies_cm.len())
        || sorted.windows(2).any(|w| w[0] == w[1])
    {
        return Err(DftError::InvalidInput("selected vibrational mode IDs must be unique nonempty indices of the projected harmonic modes".into()));
    }
    let frequencies: Vec<_> = selected
        .iter()
        .map(|&i| harmonic.frequencies_cm[i] / HARTREE_TO_CM1)
        .collect();
    let m = selected.len();
    let initial_states = m
        .checked_mul(m + 3)
        .and_then(|v| v.checked_div(2))
        .and_then(|v| v.checked_add(1));
    if initial_states.is_none_or(|n| n > config.max_polyad_states) {
        return Err(DftError::InvalidInput(
            "fundamental/overtone/combination states exceed max_polyad_states; \
            select modes or enlarge the explicit resource limit"
                .into(),
        ));
    }
    let patterns: Vec<Vec<[f64; 3]>> = selected
        .iter()
        .map(|&mode| {
            (0..mol.len())
                .map(|a| {
                    std::array::from_fn(|axis| {
                        harmonic.modes[(3 * a + axis, mode)] / atomic_mass_au(mol.atoms[a].z).sqrt()
                    })
                })
                .collect()
        })
        .collect();
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let third = analytic_force_constants(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &patterns,
        3,
        response_config,
    )?;
    converged &= third.converged;
    residuals.extend(third.residuals);
    iterations.extend(third.iterations);
    warnings.extend(third.warnings);
    let fourth = analytic_force_constants(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &patterns,
        4,
        response_config,
    )?;
    converged &= fourth.converged;
    residuals.extend(fourth.residuals);
    iterations.extend(fourth.iterations);
    warnings.extend(fourth.warnings);
    let force_permutation_defects = [
        third.value.permutation_defect,
        fourth.value.permutation_defect,
    ];
    let force_permutation_comparisons = [
        third.value.permutation_comparisons,
        fourth.value.permutation_comparisons,
    ];
    let mut polarizability_symmetry_defect = include_raman.then_some(0.0_f64);
    let mut polarizability_symmetry_comparisons = 0;
    let mut dipole = ObservablePolynomial {
        components: 3,
        first: Vec::new(),
        second: Vec::new(),
        third: Vec::new(),
    };
    let mut polarizability = include_raman.then(|| ObservablePolynomial {
        components: 9,
        first: Vec::new(),
        second: Vec::new(),
        third: Vec::new(),
    });
    for rank in 1..=3 {
        let mut mu_values = vec![0.0; m.pow(rank) * 3];
        let mut alpha_values = include_raman.then(|| vec![0.0; m.pow(rank) * 9]);
        for flat in 0..m.pow(rank) {
            let mut tuple = vec![0; rank as usize];
            let mut index = flat;
            for i in (0..rank as usize).rev() {
                tuple[i] = index % m;
                index /= m;
            }
            let directions: Vec<_> = tuple
                .iter()
                .map(|&i| MolecularDirection::nuclear(patterns[i].clone()))
                .collect();
            let report = mixed_molecular_response(
                mol,
                mb,
                functional,
                grid,
                scf,
                opts,
                &directions,
                response_config,
            )?;
            converged &= report.converged;
            residuals.extend(report.residuals);
            iterations.extend(report.iterations);
            warnings.extend(report.warnings);
            mu_values[3 * flat..3 * flat + 3].copy_from_slice(&report.value.dipole_derivative);
            if let Some(values) = &mut alpha_values {
                for b in 0..3 {
                    let mut legs = directions.clone();
                    let mut field = [0.0; 3];
                    field[b] = 1.0;
                    legs.push(MolecularDirection::electric(mol.len(), field));
                    let report = mixed_molecular_response(
                        mol,
                        mb,
                        functional,
                        grid,
                        scf,
                        opts,
                        &legs,
                        response_config,
                    )?;
                    converged &= report.converged;
                    residuals.extend(report.residuals);
                    iterations.extend(report.iterations);
                    warnings.extend(report.warnings);
                    for a in 0..3 {
                        values[9 * flat + 3 * a + b] = report.value.dipole_derivative[a];
                    }
                }
            }
        }
        match rank {
            1 => dipole.first = mu_values,
            2 => dipole.second = mu_values,
            _ => dipole.third = mu_values,
        }
        if let (Some(polynomial), Some(values)) = (&mut polarizability, alpha_values) {
            for entries in values.chunks_exact(9) {
                for a in 0..3 {
                    for b in 0..a {
                        let error = (entries[3 * a + b] - entries[3 * b + a]).abs();
                        let measured = polarizability_symmetry_defect.as_mut().unwrap();
                        *measured = measured.max(error);
                        polarizability_symmetry_comparisons += 1;
                        if error
                            > response_config.tolerance
                                * entries.iter().fold(1.0_f64, |s, v| s.max(v.abs()))
                        {
                            converged = false;
                        }
                    }
                }
            }
            match rank {
                1 => polynomial.first = values,
                2 => polynomial.second = values,
                _ => polynomial.third = values,
            }
        }
    }
    let polynomial = VibrationalPolynomial {
        frequencies,
        cubic: third.value.tensor,
        quartic: fourth.value.tensor,
        dipole,
        polarizability,
        rotation: Some(super::rotation::molecular_rotation(
            mol,
            &patterns,
            response_config.tolerance,
        )?),
    };
    let spectrum = vibrational_spectrum(&polynomial, config)?;
    converged &= spectrum.converged;
    warnings.extend(spectrum.warnings.iter().cloned());
    if selected.len() != harmonic.frequencies_cm.len() {
        warnings.push(
            "selected-mode Hamiltonian: couplings to unselected molecular modes are omitted".into(),
        );
    }
    warnings.sort();
    warnings.dedup();
    let count = spectrum.bands.len();
    Ok(ResponseReport {
        value: MolecularAnharmonic {
            selected_mode_ids: selected,
            harmonic_frequencies_cm: harmonic.frequencies_cm,
            patterns,
            n_projected: harmonic.n_projected,
            maximum_force,
            force_permutation_defects,
            force_permutation_comparisons,
            polarizability_symmetry_defect,
            polarizability_symmetry_comparisons,
            polynomial,
            spectrum,
        },
        shape: vec![count],
        units: "frequencies cm^-1; energy Hartree; IR km/mol; static Raman activity Angstrom^4/amu",
        axis_convention: "harmonic mode IDs after Eckart projection; derivative arrays use mode \
            indices then Cartesian observable component; mass-weighted coordinates \
            use electron masses",
        model: match config.method {
            AnharmonicMethod::StrictVpt2 => {
                "zero-temperature nonresonant strict second-order \
                Rayleigh-Schrodinger VPT2; cubic second/quartic first order and equilibrium \
                Coriolis/Watson terms; coupled near resonances rejected; transition moments \
                through second order then squared for intensities; rotational levels removed; \
                analytic integer-occupation molecular KS/HF"
            }
            AnharmonicMethod::RetainedPolyad => {
                "zero-temperature Watson vibrational contact transformation through \
            cubic second/quartic first order; equilibrium Coriolis kinetic operator \
            and rotor-specific Watson pseudopotential at VPT2 order; vibrational \
            band origins with rotational levels removed; resonant polyad \
            diagonalization; dipole/static polarizability through third nuclear \
            derivatives; analytic integer-occupation molecular KS/HF"
            }
        },
        functional: functional.description.clone(),
        reference: crate::response::diagnostics::reference_diagnostics(scf)?,
        converged,
        residuals,
        iterations,
        warnings,
    })
}
