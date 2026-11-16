// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear magnetic shielding tensors and first-order or exact NMR spectra
//! routed through the common driver.
//!
//! The spectrum of an element is the sum of one Lorentzian of unit area per
//! nucleus at its chemical shift `delta = sigma_reference - sigma_isotropic`.
//! With `nmr_couplings` every line is split into its first-order multiplet
//! by the indirect spin-spin couplings to the other spin-active nuclei that
//! are not equivalent to it. Paramagnetic shifts are not part of it.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::vibrational::response_config;
use super::System;
use crate::constants::PROTON_ELECTRON_MASS_RATIO;
use crate::math::Vec3;
use crate::properties::magnetic::{molecular_nmr_response, Gauge};
use crate::properties::nuclear_spin::nuclear_spin_spectrum;
use crate::response::spin_spin::{spin_spin_couplings, SpinSpinCouplings};
use crate::{DftError, Result};

pub(super) mod spinor;

/// Proton g factor, which relates the spectrometer frequency to the Larmor
/// frequency of another nucleus.
const PROTON_G: f64 = 5.5856946893;

/// One Hartree as a frequency, Hz.
pub(super) const HARTREE_IN_HERTZ: f64 = 6.579683920502e15;

/// Isotropic coupling constants `J = g_M g_N mu_n^2 K_iso / h` in Hz of every
/// pair, with the nuclear magneton `mu_n = m_e / (2 m_p)` in atomic units.
fn coupling_hertz(couplings: &SpinSpinCouplings, tensors: &[[[f64; 3]; 3]], g: &[f64]) -> Vec<f64> {
    coupling_hertz_pairs(&couplings.pairs, tensors, g)
}

pub(super) fn coupling_hertz_pairs(
    pairs: &[(usize, usize)],
    tensors: &[[[f64; 3]; 3]],
    g: &[f64],
) -> Vec<f64> {
    let magneton = 0.5 / PROTON_ELECTRON_MASS_RATIO;
    pairs
        .iter()
        .zip(tensors)
        .map(|(&(m, n), k)| {
            let isotropic = (k[0][0] + k[1][1] + k[2][2]) / 3.0;
            g[m] * g[n] * magneton * magneton * isotropic * HARTREE_IN_HERTZ
        })
        .collect()
}

/// Controls of the broadened spectrum.
struct Spectrum {
    /// Atomic number and isotropic shielding of its reference compound (ppm).
    references: Vec<(usize, f64)>,
    /// Full width at half maximum of the Lorentzians (ppm).
    broadening: f64,
    points: usize,
    /// Extension of the axis beyond the outermost shifts (ppm).
    margin: f64,
}

fn spectrum_controls(controls: &Controls) -> Result<Spectrum> {
    let missing = || {
        DftError::InvalidInput(
            "nmr-spectrum needs nmr_reference_elements and nmr_reference_shieldings: the atomic \
             numbers to plot and the isotropic shielding of each reference compound in ppm"
                .into(),
        )
    };
    let elements = controls
        .opt_usize_list("nmr_reference_elements")?
        .ok_or_else(missing)?;
    let shieldings = controls
        .opt_f64_list("nmr_reference_shieldings")?
        .ok_or_else(missing)?;
    if elements.is_empty() || elements.len() != shieldings.len() {
        return Err(DftError::InvalidInput(format!(
            "nmr_reference_elements has {} entries and nmr_reference_shieldings {}; give one \
             shielding per element",
            elements.len(),
            shieldings.len()
        )));
    }
    let spectrum = Spectrum {
        references: elements.into_iter().zip(shieldings).collect(),
        broadening: controls.f64("nmr_broadening", 0.02)?,
        points: controls.usize("nmr_points", 2001)?,
        margin: controls.f64("nmr_margin", 1.0)?,
    };
    if !(spectrum.broadening.is_finite()
        && spectrum.broadening > 0.0
        && spectrum.margin.is_finite()
        && spectrum.margin >= 0.0)
        || spectrum.points < 2
        || spectrum.references.iter().any(|(_, s)| !s.is_finite())
    {
        return Err(DftError::InvalidInput(
            "nmr-spectrum needs a positive nmr_broadening, a non-negative nmr_margin, at least \
             two nmr_points and finite reference shieldings"
                .into(),
        ));
    }
    Ok(spectrum)
}

/// Electronic coupling data accepted by the shared isotope Hamiltonian.
pub(super) enum SpectrumCouplings {
    Ramsey(SpinSpinCouplings),
    Dirac {
        pairs: Vec<(usize, usize)>,
        tensors: Vec<[[f64; 3]; 3]>,
        residual: f64,
    },
}

pub(in crate::driver) struct SpectrumRequest {
    spectrum: Option<Spectrum>,
    couplings: bool,
    spin_model: String,
    exact_spins: bool,
    spin_max_dimension: usize,
    nuclear_g: Option<Vec<f64>>,
    nuclear_spins: Option<Vec<usize>>,
    spectrometer: f64,
}

impl SpectrumRequest {
    pub(in crate::driver) fn new(id: &str, controls: &Controls) -> Result<Self> {
        let with_spectrum = id == "nmr-spectrum";
        let spectrum = if with_spectrum {
            Some(spectrum_controls(controls)?)
        } else {
            None
        };
        let couplings = with_spectrum && controls.bool("nmr_couplings", false)?;
        let spin_model = if with_spectrum {
            controls.str("nmr_spin_model", "first-order")?
        } else {
            "first-order".into()
        };
        let exact_spins =
            match spin_model.as_str() {
                "first-order" => false,
                "exact" if couplings => true,
                "exact" => return Err(DftError::InvalidInput(
                    "nmr_spin_model=exact needs nmr_couplings=true and explicit nuclear g factors"
                        .into(),
                )),
                _ => {
                    return Err(DftError::InvalidInput(
                        "nmr_spin_model must be first-order or exact".into(),
                    ))
                }
            };
        let spin_max_dimension = if with_spectrum {
            controls.usize("nmr_spin_max_dimension", 1024)?
        } else {
            1024
        };
        let nuclear_g = if with_spectrum {
            controls.opt_f64_list("nuclear_g")?
        } else {
            None
        };
        let nuclear_spins = if with_spectrum {
            controls.opt_usize_list("nuclear_spins")?
        } else {
            None
        };
        let spectrometer = if with_spectrum {
            controls.f64("nmr_spectrometer_mhz", 400.0)?
        } else {
            400.0
        };
        Ok(Self {
            spectrum,
            couplings,
            spin_model,
            exact_spins,
            spin_max_dimension,
            nuclear_g,
            nuclear_spins,
            spectrometer,
        })
    }

    /// Assemble chemical shifts and the nuclear-spin Hamiltonian from the
    /// reference-specific electronic tensors. Electronic response is supplied
    /// by the caller; no SCF reference is chosen by spectral post-processing.
    pub(in crate::driver) fn append(
        self,
        numbers: &[usize],
        isotropic: &[f64],
        report: &mut Value,
        unit_pairs: &mut Vec<(&'static str, &'static str)>,
        config: &crate::response::ResponseSolverConfig,
        mut evaluate_couplings: impl FnMut(&[usize]) -> Result<SpectrumCouplings>,
    ) -> Result<()> {
        let Self {
            spectrum,
            couplings,
            spin_model,
            exact_spins,
            spin_max_dimension,
            nuclear_g,
            nuclear_spins,
            spectrometer,
        } = self;
        if let Some(spectrum) = &spectrum {
            // Chemical shift of every nucleus whose element has a reference.
            let shifts: Vec<Option<f64>> = numbers
                .iter()
                .zip(isotropic)
                .map(|(z, sigma)| {
                    spectrum
                        .references
                        .iter()
                        .find(|(element, _)| element == z)
                        .map(|(_, reference)| reference - sigma)
                })
                .collect();
            // Lines of every nucleus: its chemical shift, split into the
            // first-order multiplet of its couplings when requested.
            let mut multiplets: Vec<Vec<(f64, f64)>> = shifts
                .iter()
                .map(|shift| shift.map_or(Vec::new(), |s| vec![(s, 1.0)]))
                .collect();
            let mut exact_spans = vec![None; numbers.len()];
            if couplings {
                let g = nuclear_g.as_ref().ok_or_else(|| {
                    DftError::InvalidInput(
                        "nmr_couplings needs nuclear_g: one nuclear g factor per atom, zero for a \
                     nucleus without spin"
                            .into(),
                    )
                })?;
                let twice_spin: Vec<usize> = match &nuclear_spins {
                    Some(list) => list.clone(),
                    None => g.iter().map(|x| usize::from(*x != 0.0)).collect(),
                };
                if g.len() != numbers.len()
                    || twice_spin.len() != numbers.len()
                    || g.iter().any(|v| !v.is_finite())
                    || !(spectrometer.is_finite() && spectrometer > 0.0)
                {
                    return Err(DftError::InvalidInput(
                        "nuclear_g and nuclear_spins (twice the spin) hold one value per atom and \
                     nmr_spectrometer_mhz is positive"
                            .into(),
                    ));
                }
                let active: Vec<usize> = (0..numbers.len())
                    .filter(|&a| g[a] != 0.0 && twice_spin[a] > 0)
                    .collect();
                let coupled = evaluate_couplings(&active)?;
                let (pairs, tensors, residual) = match &coupled {
                    SpectrumCouplings::Ramsey(value) => {
                        for (key, tensors) in [
                            ("fermi_contact_tensors", &value.fermi_contact),
                            ("spin_dipole_tensors", &value.spin_dipole),
                            ("paramagnetic_coupling_tensors", &value.paramagnetic),
                            ("diamagnetic_coupling_tensors", &value.diamagnetic),
                        ] {
                            report.set(key, tensors.clone());
                            unit_pairs.push((key, "atomic units, d2E/(dm_M dm_N)"));
                        }
                        (&value.pairs, &value.total, value.residual)
                    }
                    SpectrumCouplings::Dirac {
                        pairs,
                        tensors,
                        residual,
                    } => (pairs, tensors, *residual),
                };
                report.set("reduced_couplings", tensors.clone());
                unit_pairs.push(("reduced_couplings", "atomic units, d2E/(dm_M dm_N)"));
                report.set("coupling_response_residual", residual);
                unit_pairs.push(("coupling_response_residual", "dimensionless"));
                let accepted = residual <= config.tolerance;
                for key in ["converged", "response_converged"] {
                    let prior = report.get(key).and_then(Value::as_bool).unwrap_or(false);
                    report.set(key, prior && accepted);
                }
                let hertz = coupling_hertz_pairs(pairs, tensors, g);
                if exact_spins {
                    // A display window around the observed Larmor band must not
                    // depend discontinuously on a roundoff-sized dark-line weight.
                    // Weyl's eigenvalue bound supplies the scalar-coupling extent;
                    // all transitions, including remote weak bands, are still
                    // returned and contribute their Lorentzian tails on this axis.
                    let coupling_bound_hz: f64 = pairs
                        .iter()
                        .zip(&hertz)
                        .map(|(&(m, n), j)| {
                            let im = 0.5 * twice_spin[m] as f64;
                            let in_ = 0.5 * twice_spin[n] as f64;
                            2.0 * j.abs() * (im * (im + 1.0) * in_ * (in_ + 1.0)).sqrt()
                        })
                        .sum();
                    for &a in &active {
                        exact_spans[a] =
                            Some(coupling_bound_hz / (spectrometer * g[a].abs() / PROTON_G));
                    }
                    let carriers: Vec<_> = active
                        .iter()
                        .map(|&a| spectrometer * g[a] / PROTON_G * 1e6)
                        .collect();
                    // Physical shielded Larmor frequencies are independent of
                    // the user's reference compound. References enter only ppm.
                    let offsets: Vec<_> = active
                        .iter()
                        .enumerate()
                        .map(|(i, &a)| -carriers[i] * isotropic[a] * 1e-6)
                        .collect();
                    let spins: Vec<_> = active.iter().map(|&a| twice_spin[a]).collect();
                    let mut isotope_groups =
                        std::collections::BTreeMap::<(usize, u64, usize), Vec<usize>>::new();
                    for (i, &a) in active.iter().enumerate() {
                        if shifts[a].is_none() {
                            continue;
                        }
                        isotope_groups
                            .entry((numbers[a], g[a].to_bits(), twice_spin[a]))
                            .or_default()
                            .push(i);
                    }
                    let groups: Vec<_> = isotope_groups.into_values().collect();
                    let pair_values: Vec<_> = pairs
                        .iter()
                        .zip(&hertz)
                        .map(|(&(a, b), &j)| {
                            (
                                active
                                    .iter()
                                    .position(|&v| v == a)
                                    .expect("active coupling nucleus"),
                                active
                                    .iter()
                                    .position(|&v| v == b)
                                    .expect("active coupling nucleus"),
                                j,
                            )
                        })
                        .collect();
                    let solved = nuclear_spin_spectrum(
                        &carriers,
                        &offsets,
                        &spins,
                        &pair_values,
                        &groups,
                        spin_max_dimension,
                    )?;
                    let spin_converged = solved.eigen_relative_residual <= config.tolerance
                        && solved
                            .sum_rule_defects
                            .iter()
                            .zip(&groups)
                            .all(|(error, group)| {
                                error.abs() <= config.tolerance * group.len() as f64
                            });
                    for lines in &mut multiplets {
                        lines.clear();
                    }
                    for (group, lines) in groups.iter().zip(&solved.lines) {
                        let first = active[group[0]];
                        let larmor = spectrometer * g[first].abs() / PROTON_G;
                        let reference_shielding =
                            shifts[first].expect("observed isotope group") + isotropic[first];
                        multiplets[first] = lines
                            .iter()
                            .map(|line| {
                                (reference_shielding + line.offset_hz / larmor, line.weight)
                            })
                            .collect();
                    }
                    report.set("spin_hilbert_dimension", solved.hilbert_dimension);
                    report.set("spin_hamiltonian_residual", solved.eigen_residual_hz);
                    report.set(
                        "spin_hamiltonian_relative_residual",
                        solved.eigen_relative_residual,
                    );
                    report.set("spin_hamiltonian_converged", spin_converged);
                    for key in ["converged", "response_converged"] {
                        let previous = report.get(key).and_then(Value::as_bool).unwrap_or(false);
                        report.set(key, previous && spin_converged);
                    }
                    report.set("spin_transition_sum_rule_defects", solved.sum_rule_defects);
                    unit_pairs.push(("spin_hilbert_dimension", "dimensionless"));
                    unit_pairs.push(("spin_hamiltonian_residual", "Hz"));
                    unit_pairs.push(("spin_hamiltonian_relative_residual", "dimensionless"));
                    unit_pairs.push(("spin_hamiltonian_converged", "flag"));
                    unit_pairs.push(("spin_transition_sum_rule_defects", "dimensionless"));
                } else {
                    for (index, &(m, n)) in pairs.iter().enumerate() {
                        for (observed, partner) in [(m, n), (n, m)] {
                            let Some(shift) = shifts[observed] else {
                                continue;
                            };
                            // Equivalent nuclei do not split each other in first order.
                            if numbers[observed] == numbers[partner]
                                && shifts[partner].is_some_and(|other| (other - shift).abs() < 1e-3)
                            {
                                continue;
                            }
                            // Larmor frequency of the observed nucleus in MHz: the
                            // coupling in Hz over it is the splitting in ppm.
                            let larmor = spectrometer * g[observed].abs() / PROTON_G;
                            let offset = hertz[index] / larmor;
                            let levels = twice_spin[partner] + 1;
                            let mut split = Vec::with_capacity(multiplets[observed].len() * levels);
                            for &(position, weight) in &multiplets[observed] {
                                for level in 0..levels {
                                    let projection =
                                        level as f64 - 0.5 * twice_spin[partner] as f64;
                                    split.push((
                                        position + projection * offset,
                                        weight / levels as f64,
                                    ));
                                }
                            }
                            multiplets[observed] = split;
                        }
                    }
                }
                report.set(
                    "coupling_pairs",
                    pairs.iter().map(|&(m, n)| [m, n]).collect::<Vec<_>>(),
                );
                report.set("coupling_constants_hz", hertz);
                report.set("spectrometer_mhz", spectrometer);
                unit_pairs.push(("coupling_constants_hz", "Hz, isotropic J of every pair"));
                unit_pairs.push(("spectrometer_mhz", "MHz, proton frequency"));
            }
            let half = 0.5 * spectrum.broadening;
            let mut elements = Vec::new();
            let mut axes = Vec::new();
            let mut intensities = Vec::new();
            let mut line_positions = Vec::new();
            let mut line_weights = Vec::new();
            for (element, _) in &spectrum.references {
                let weighted: Vec<(f64, f64)> = numbers
                    .iter()
                    .enumerate()
                    .filter(|(_, z)| *z == element)
                    .flat_map(|(atom, _)| multiplets[atom].iter().copied())
                    .collect();
                let lines: Vec<f64> = weighted.iter().map(|(position, _)| *position).collect();
                if lines.is_empty() {
                    return Err(DftError::InvalidInput(format!(
                        "nmr_reference_elements names atomic number {element}, which the molecule \
                     does not contain"
                    )));
                }
                let (low, high) = if exact_spins {
                    let observed: Vec<_> = numbers
                        .iter()
                        .enumerate()
                        .filter(|(a, z)| *z == element && exact_spans[*a].is_some())
                        .map(|(a, _)| {
                            (
                                shifts[a].expect("observed nucleus"),
                                exact_spans[a].expect("active spin"),
                            )
                        })
                        .collect();
                    (
                        observed
                            .iter()
                            .map(|(s, w)| s - w)
                            .fold(f64::INFINITY, f64::min)
                            - spectrum.margin,
                        observed
                            .iter()
                            .map(|(s, w)| s + w)
                            .fold(f64::NEG_INFINITY, f64::max)
                            + spectrum.margin,
                    )
                } else {
                    (
                        lines.iter().copied().fold(f64::INFINITY, f64::min) - spectrum.margin,
                        lines.iter().copied().fold(f64::NEG_INFINITY, f64::max) + spectrum.margin,
                    )
                };
                let step = (high - low).max(spectrum.broadening) / (spectrum.points - 1) as f64;
                let axis: Vec<f64> = (0..spectrum.points)
                    .map(|i| low + step * i as f64)
                    .collect();
                let intensity: Vec<f64> = axis
                    .iter()
                    .map(|x| {
                        weighted
                            .iter()
                            .map(|(line, weight)| {
                                weight * half
                                    / (std::f64::consts::PI * ((x - line).powi(2) + half * half))
                            })
                            .sum()
                    })
                    .collect();
                elements.push(*element);
                axes.push(axis);
                intensities.push(intensity);
                line_positions.push(lines);
                line_weights.push(
                    weighted
                        .iter()
                        .map(|(_, weight)| *weight)
                        .collect::<Vec<_>>(),
                );
            }
            report.set("chemical_shifts", shifts);
            report.set("spectrum_elements", elements);
            report.set("spectrum_axis", axes);
            report.set("spectrum_intensity", intensities);
            report.set("spectrum_broadening", spectrum.broadening);
            report.set("spectrum_spin_model", spin_model);
            report.set("spectrum_line_positions", line_positions);
            report.set("spectrum_line_weights", line_weights);
            report.set("spectrum_axis_convention", if exact_spins {
            "observed chemical-shift range plus twice the isotropic-coupling operator-norm bound and requested margin; every raw transition retained independently of the display window"
        } else {
            "first-order multiplet range plus requested margin"
        });
            report.set(
            "spectrum_model",
            if exact_spins {
                "exact isotropic scalar-coupling spin Hamiltonian in total-Iz blocks; coherent isotope-group transverse detection; common high-temperature population factor removed; unit integrated area per observed nucleus; no quadrupolar, anisotropic or relaxation terms"
            } else if couplings {
                "unit area per nucleus at delta = sigma_reference - sigma_isotropic, split into \
                 the first-order multiplet of its indirect spin-spin couplings"
            } else {
                "one Lorentzian of unit area per nucleus at delta = sigma_reference - \
                 sigma_isotropic; no spin-spin couplings"
            },
        );
            unit_pairs.push((
                "chemical_shifts",
                "ppm, null for a nucleus without a reference",
            ));
            unit_pairs.push(("spectrum_axis", "ppm, [element][point]"));
            unit_pairs.push((
                "spectrum_intensity",
                "1/ppm, [element][point], unit area per nucleus",
            ));
            unit_pairs.push(("spectrum_broadening", "ppm, full width at half maximum"));
            unit_pairs.push(("spectrum_spin_model", "dimensionless"));
            unit_pairs.push(("spectrum_axis_convention", "dimensionless"));
            unit_pairs.push(("spectrum_line_positions", "ppm, [element][transition]"));
            unit_pairs.push((
                "spectrum_line_weights",
                "dimensionless, integrated area per transition",
            ));
        }
        Ok(())
    }
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    match id {
        "nmr-shielding" | "nmr-spectrum" => (),
        other => {
            return Err(DftError::Unsupported {
                feature: format!("magnetic-resonance calculation `{other}`"),
                missing: "a common-driver implementation".into(),
            })
        }
    };
    let selected_reference = controls.str("reference", "auto")?;
    let common_orbitals_requested = matches!(selected_reference.as_str(), "roks" | "rohf");
    if common_orbitals_requested && super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "correlated common-orbital NMR".into(),
            missing: "the constrained PT2 magnetic Lagrangian".into(),
        });
    }
    let config = response_config(controls)?;
    let gauge_name = controls.str("gauge", "giao")?;
    let gauge = Gauge::from_name(&gauge_name)?;
    let origin = controls
        .opt_vec3("gauge_origin")?
        .unwrap_or_else(Vec3::zero);
    let spectral = SpectrumRequest::new(id, controls)?;
    let fixed_spin = spinor::constraint_mode(controls)?;
    let prepared = if super::double_hybrid::requested(controls)? {
        Some(super::double_hybrid::prepare(system, controls, id)?)
    } else {
        None
    };
    let mut common_orbitals = None;
    let ordinary = if prepared.is_none() {
        let data = setup(system, controls)?;
        controls.finish(id)?;
        let reference = if common_orbitals_requested {
            if data.smd.is_some() || data.method.stabilize {
                return Err(DftError::Unsupported {
                    feature: "common-orbital NMR with solvent or stability following".into(),
                    missing: "the matching constrained reference provider".into(),
                });
            }
            let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
                &data.molecule,
                &data.basis,
                &data.options,
                &data.functional,
                data.grid.as_ref(),
            )?;
            common_orbitals = Some(orbitals);
            super::method::MolecularReference {
                scf,
                e_dispersion: crate::dispersion::dispersion_energy(
                    &data.molecule,
                    &data.dispersion,
                    &data.method.dispersion_functional_name(),
                )?,
                e_smd_cds: 0.0,
                following: None,
            }
        } else {
            run_molecular_reference(&data)?
        };
        Some((data, reference))
    } else {
        None
    };
    let (setup, reference) = match (&prepared, &ordinary) {
        (Some(data), _) => (&data.setup, &data.reference),
        (_, Some((setup, reference))) => (setup, reference),
        _ => unreachable!("one molecular reference was prepared"),
    };
    if !reference.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.scf.iterations,
            error: f64::NAN,
        });
    }
    let correlated = if let Some(data) = &prepared {
        Some(
            crate::properties::correlated_magnetic::correlated_molecular_nmr_response_for_model(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                data.field_target()?,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
                data.coefficients,
                gauge,
                origin,
                controls.usize("pt2_memory_mb", 2048)?,
                &config,
            )?,
        )
    } else {
        None
    };
    let constrained = common_orbitals
        .as_ref()
        .map(|orbitals| {
            crate::properties::magnetic::common_orbital_nmr_response(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                orbitals,
                &setup.options,
                gauge,
                origin,
                &config,
            )
        })
        .transpose()?;
    let result = if let Some(value) = &constrained {
        let mut diagnostics = crate::response::diagnostics::reference_diagnostics(&reference.scf)?;
        diagnostics.orbital_residual = value.reference_stationarity;
        crate::response::ResponseReport {
            value: value.value.clone(),
            shape: vec![setup.molecule.len(), 3, 3],
            functional: setup.functional.description.clone(),
            reference: diagnostics,
            units: "ppm",
            model: if gauge == Gauge::Giao {
                "Common-orbital constrained complex orbital NMR response; London basis and metric derivatives"
            } else {
                "Common-orbital constrained complex orbital NMR response; common origin"
            },
            axis_convention: "nucleus, nuclear moment axis, external field axis",
            converged: value.residuals.iter().all(|r| *r <= config.tolerance)
                && value.reference_stationarity <= config.acceptance(),
            residuals: value.residuals.clone(),
            iterations: value.iterations.clone(),
            warnings: if gauge == Gauge::Giao {
                vec!["Orbital response excludes spin-Zeeman and thermal shifts.".into()]
            } else {
                vec!["Common-origin finite-basis shielding is gauge-origin dependent; orbital response excludes spin-Zeeman and thermal shifts.".into()]
            },
        }
    } else if let Some(data) = &correlated {
        data.response.clone()
    } else {
        molecular_nmr_response(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            gauge,
            origin,
            &config,
        )?
    };
    let mut correlated_coupling = None;
    let shieldings = &result.value;
    let isotropic: Vec<f64> = shieldings.iter().map(|s| s.isotropic()).collect();
    let mut report = reference_report(setup, reference, result.model)
        .with("gauge", gauge_name)
        .with("gauge_origin", origin)
        .with(
            "shielding_tensors",
            shieldings.iter().map(|s| s.tensor).collect::<Vec<_>>(),
        )
        .with(
            "diamagnetic_tensors",
            shieldings.iter().map(|s| s.diamagnetic).collect::<Vec<_>>(),
        )
        .with(
            "paramagnetic_tensors",
            shieldings
                .iter()
                .map(|s| s.paramagnetic)
                .collect::<Vec<_>>(),
        )
        .with("isotropic_shieldings", isotropic.clone())
        .with(
            "diamagnetic_isotropic",
            shieldings
                .iter()
                .map(|s| s.isotropic_parts().0)
                .collect::<Vec<_>>(),
        )
        .with(
            "paramagnetic_isotropic",
            shieldings
                .iter()
                .map(|s| s.isotropic_parts().1)
                .collect::<Vec<_>>(),
        )
        .with("converged", reference.scf.converged && result.converged)
        .with("response_converged", result.converged)
        .with("response_residuals", result.residuals.clone())
        .with("provenance", "analytic");
    if common_orbitals_requested {
        report.set("reference_type", "roks");
        report.set("requested_reference", selected_reference);
        report.set("reference_manifold", if reference.scf.n_alpha == reference.scf.n_beta {
            "common spatial orbitals, empty open-shell class; exact restricted closed-shell limit"
        } else {
            "common spatial orbitals with closed, open and virtual classes"
        });
    }
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        ("gauge_origin", "bohr"),
        (
            "shielding_tensors",
            "ppm, [nucleus][i][j] with sigma_ij = d2E/(dm_i dB_j)",
        ),
        ("diamagnetic_tensors", "ppm"),
        ("paramagnetic_tensors", "ppm"),
        ("isotropic_shieldings", "ppm"),
        ("diamagnetic_isotropic", "ppm"),
        ("paramagnetic_isotropic", "ppm"),
        ("response_residuals", "dimensionless"),
    ];
    if let Some(value) = &constrained {
        report.set("reference_stationarity", value.reference_stationarity);
        report.set("metric_defect", value.metric_defect);
        unit_pairs.push(("reference_stationarity", "hartree"));
        unit_pairs.push(("metric_defect", "dimensionless"));
    }
    let numbers: Vec<usize> = setup.molecule.atoms.iter().map(|a| a.z as usize).collect();
    spectral.append(&numbers, &isotropic, &mut report, &mut unit_pairs, &config, |active| {
        let coupling_grid = setup.integration_grid()?;
        if common_orbitals_requested && reference.scf.n_alpha != reference.scf.n_beta {
            return Err(DftError::Unsupported {
                feature: "open-shell common-orbital full Ramsey coupling".into(),
                missing: "the transverse spin-adapted response manifold; a longitudinal constrained tensor is not the full Ramsey tensor".into(),
            });
        }
        let coupled = if let Some(data) = &prepared {
                let value =
                    crate::properties::correlated_spin_spin::correlated_spin_spin_couplings_for_model(
                        &setup.molecule,
                        &setup.basis,
                        &setup.functional,
                        data.field_target()?,
                        Some(&coupling_grid),
                        &reference.scf,
                        &setup.options,
                        data.coefficients,
                        active,
                        false,
                        controls.usize("pt2_memory_mb", 2048)?,
                        &config,
                    )?;
            let coupled = value.response.value.clone();
                correlated_coupling = Some(value);
                coupled
            } else if reference.scf.unrestricted || fixed_spin {
                spinor::collinear_couplings(
                    setup, &reference.scf, &coupling_grid, active, false, fixed_spin, &config,
                )?.value
            } else {
                spin_spin_couplings(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    Some(&coupling_grid),
                    &reference.scf,
                    &setup.options,
                    active,
                    false,
                    &config,
                )?
            };
        Ok(SpectrumCouplings::Ramsey(coupled))
    })?;
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(result.warnings.iter().cloned().map(Value::Str));
        report.set("warnings", Value::Array(all));
    }
    report.set("units", units(&unit_pairs));
    if let Some(data) = correlated {
        for (key, value) in [
            ("parent_shielding_tensors", data.parent_tensors),
            ("target_shielding_tensors", data.target_tensors),
            ("pt2_shielding_tensors", data.correlation_tensors),
            (
                "pt2_same_spin_alpha_shielding_tensors",
                data.same_spin_alpha,
            ),
            ("pt2_opposite_spin_shielding_tensors", data.opposite_spin),
            ("pt2_same_spin_beta_shielding_tensors", data.same_spin_beta),
            (
                "parent_nonlocal_shielding_tensors",
                data.parent_nonlocal_tensors,
            ),
            (
                "target_nonlocal_shielding_tensors",
                data.target_nonlocal_tensors,
            ),
        ] {
            report.set(key, value);
            if let Some(Value::Object(labels)) = report.get("units").cloned() {
                let mut units = Value::Object(labels);
                units.set(key, "ppm");
                report.set("units", units);
            }
        }
        for (key, value, unit) in [
            (
                "amplitude_response_residuals",
                data.amplitude_residuals,
                "hartree",
            ),
            (
                "orbital_frame_residuals",
                data.frame_defects,
                "dimensionless",
            ),
            (
                "projector_residuals",
                data.projector_defects,
                "dimensionless",
            ),
            (
                "stationarity_residuals",
                data.stationarity_defects,
                "hartree",
            ),
            (
                "response_reference_residuals",
                data.reference_defects,
                "hartree",
            ),
            (
                "imaginary_energy_residuals",
                data.imaginary_energy_defects,
                "hartree",
            ),
        ] {
            report.set(key, value);
            if let Some(Value::Object(labels)) = report.get("units").cloned() {
                let mut units = Value::Object(labels);
                units.set(key, unit);
                report.set("units", units);
            }
        }
    }
    if let Some(value) = &correlated_coupling {
        append_correlated_couplings(&mut report, value);
    }
    if let Some(prepared) = prepared {
        report = super::double_hybrid::append_prepared_energy(prepared, report)?;
    }
    Ok(report)
}

/// Indirect nuclear spin-spin couplings routed through the common driver:
/// the reduced coupling tensors `K_MN = d2E / (dm_M dm_N)` with their
/// diamagnetic and paramagnetic spin-orbit, Fermi-contact and spin-dipole
/// parts, and with `nuclear_g` the isotropic coupling constants in Hz.
/// `fermi-contact-couplings` evaluates the Fermi-contact part alone.
pub(super) fn spin_spin(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let contact_only = match id {
        "spin-spin-couplings" => false,
        "fermi-contact-couplings" => true,
        other => {
            return Err(DftError::Unsupported {
                feature: format!("spin-spin coupling calculation `{other}`"),
                missing: "a common-driver implementation".into(),
            })
        }
    };
    let config = response_config(controls)?;
    let fixed_spin = spinor::constraint_mode(controls)?;
    let selected = controls.opt_usize_list("nuclei")?;
    let nuclear_g = controls.opt_f64_list("nuclear_g")?;
    let prepared = if super::double_hybrid::requested(controls)? {
        Some(super::double_hybrid::prepare(system, controls, id)?)
    } else {
        None
    };
    let ordinary = if prepared.is_none() {
        let data = setup(system, controls)?;
        controls.finish(id)?;
        let reference = run_molecular_reference(&data)?;
        Some((data, reference))
    } else {
        None
    };
    let (molecular, reference) = match (&prepared, &ordinary) {
        (Some(data), _) => (&data.setup, &data.reference),
        (_, Some((data, reference))) => (data, reference),
        _ => unreachable!("one molecular reference was prepared"),
    };
    let natoms = molecular.molecule.len();
    let nuclei = selected.unwrap_or_else(|| (0..natoms).collect());
    if nuclei.len() < 2 || nuclear_g.as_ref().is_some_and(|g| g.len() != natoms) {
        return Err(DftError::InvalidInput(
            "spin-spin couplings need at least two nuclei, and nuclear_g holds one g factor per \
             atom"
                .into(),
        ));
    }
    if !reference.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.scf.iterations,
            error: f64::NAN,
        });
    }
    let coupling_grid = molecular.integration_grid()?;
    if prepared.is_none() && reference.scf.unrestricted {
        let result = spinor::collinear_couplings(
            molecular,
            &reference.scf,
            &coupling_grid,
            &nuclei,
            contact_only,
            fixed_spin,
            &config,
        )?;
        let mut report = reference_report(
            molecular,
            reference,
            "full-spin response of the unchanged UKS/UHF reference",
        );
        spinor::append(
            &mut report,
            &result,
            nuclear_g.as_deref(),
            contact_only,
            fixed_spin,
        );
        return Ok(report);
    }
    let correlated = if let Some(data) = &prepared {
        Some(
            crate::properties::correlated_spin_spin::correlated_spin_spin_couplings_for_model(
                &molecular.molecule,
                &molecular.basis,
                &molecular.functional,
                data.field_target()?,
                Some(&coupling_grid),
                &reference.scf,
                &molecular.options,
                data.coefficients,
                &nuclei,
                contact_only,
                controls.usize("pt2_memory_mb", 2048)?,
                &config,
            )?,
        )
    } else {
        None
    };
    let coupled = if let Some(value) = &correlated {
        value.response.value.clone()
    } else {
        spin_spin_couplings(
            &molecular.molecule,
            &molecular.basis,
            &molecular.functional,
            Some(&coupling_grid),
            &reference.scf,
            &molecular.options,
            &nuclei,
            contact_only,
            &config,
        )?
    };
    let isotropic = |tensors: &[[[f64; 3]; 3]]| -> Vec<f64> {
        tensors
            .iter()
            .map(|k| (k[0][0] + k[1][1] + k[2][2]) / 3.0)
            .collect()
    };
    let mut report = reference_report(
        molecular,
        reference,
        "indirect nuclear spin-spin coupling: reduced coupling tensors from the singlet \
         (orbital) and triplet (contact and dipolar) responses of a closed-shell reference",
    )
    .with(
        "coupling_pairs",
        coupled
            .pairs
            .iter()
            .map(|&(m, n)| [m, n])
            .collect::<Vec<_>>(),
    )
    .with("reduced_couplings", coupled.total.clone())
    .with("fermi_contact_tensors", coupled.fermi_contact.clone())
    .with("reduced_isotropic", isotropic(&coupled.total))
    .with("fermi_contact_isotropic", isotropic(&coupled.fermi_contact))
    .with("response_residual", coupled.residual)
    .with(
        "axis_convention",
        "coupling_pairs[p] = [M,N] follows the requested nucleus order; \
         tensor[p][a][b] = d2E/(dm_M,a dm_N,b); reversing a pair transposes \
         its tensor without symmetrizing its Cartesian indices",
    )
    .with("provenance", "analytic");
    let mut unit_pairs = vec![
        ("energy", "hartree"),
        (
            "reduced_couplings",
            "atomic units, K_MN[pair][a][b] = d2E/(dm_M,a dm_N,b)",
        ),
        ("fermi_contact_tensors", "atomic units"),
        ("reduced_isotropic", "atomic units"),
        ("fermi_contact_isotropic", "atomic units"),
        ("response_residual", "dimensionless"),
    ];
    if !contact_only {
        report.set("diamagnetic_tensors", coupled.diamagnetic.clone());
        report.set("paramagnetic_tensors", coupled.paramagnetic.clone());
        report.set("spin_dipole_tensors", coupled.spin_dipole.clone());
        report.set("diamagnetic_isotropic", isotropic(&coupled.diamagnetic));
        report.set("paramagnetic_isotropic", isotropic(&coupled.paramagnetic));
        report.set("spin_dipole_isotropic", isotropic(&coupled.spin_dipole));
        unit_pairs.push(("diamagnetic_tensors", "atomic units"));
        unit_pairs.push(("paramagnetic_tensors", "atomic units"));
        unit_pairs.push((
            "spin_dipole_tensors",
            "atomic units, spin dipole with spin dipole and with Fermi contact",
        ));
        unit_pairs.push(("diamagnetic_isotropic", "atomic units"));
        unit_pairs.push(("paramagnetic_isotropic", "atomic units"));
        unit_pairs.push(("spin_dipole_isotropic", "atomic units"));
    }
    if let Some(g) = &nuclear_g {
        report.set(
            "coupling_constants_hz",
            coupling_hertz(&coupled, &coupled.total, g),
        );
        report.set(
            "fermi_contact_hz",
            coupling_hertz(&coupled, &coupled.fermi_contact, g),
        );
        unit_pairs.push(("coupling_constants_hz", "Hz, isotropic J of every pair"));
        unit_pairs.push(("fermi_contact_hz", "Hz"));
    }
    report.set("units", units(&unit_pairs));
    if let Some(value) = &correlated {
        append_correlated_couplings(&mut report, value);
    }
    if let Some(prepared) = prepared {
        report = super::double_hybrid::append_prepared_energy(prepared, report)?;
    }
    Ok(report)
}

fn append_correlated_couplings(
    report: &mut Value,
    data: &crate::properties::correlated_spin_spin::CorrelatedSpinSpin,
) {
    report.set("coupling_model", data.response.model);
    report.set("coupling_response_converged", data.response.converged);
    let converged = report
        .get("response_converged")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    report.set("response_converged", converged && data.response.converged);
    let mut labels = report.get("units").cloned().unwrap_or_else(|| units(&[]));
    for (key, value) in [
        ("coupling_parent_tensors", &data.parent),
        ("coupling_target_tensors", &data.target),
        ("coupling_correlation_tensors", &data.correlation),
        ("coupling_same_spin_alpha", &data.same_spin_alpha),
        ("coupling_opposite_spin", &data.opposite_spin),
        ("coupling_same_spin_beta", &data.same_spin_beta),
        ("coupling_parent_nonlocal_tensors", &data.parent_nonlocal),
        ("coupling_target_nonlocal_tensors", &data.target_nonlocal),
    ] {
        report.set(key, value.total.clone());
        labels.set(key, "atomic units");
    }
    report.set(
        "coupling_response_residuals",
        data.response.residuals.clone(),
    );
    report.set("coupling_path_residuals", data.path_residuals.clone());
    report.set(
        "coupling_amplitude_residuals",
        data.amplitude_residuals.clone(),
    );
    report.set(
        "coupling_stationarity_defects",
        data.stationarity_defects.clone(),
    );
    labels.set("coupling_response_residuals", "dimensionless");
    labels.set("coupling_path_residuals", "dimensionless");
    labels.set("coupling_amplitude_residuals", "hartree");
    labels.set("coupling_stationarity_defects", "hartree");
    report.set("units", labels);
}
