// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular excited states routed through the common driver: vertical
//! TDA/RPA excitations, Gaussian-broadened absorption and circular-dichroism
//! spectra, and the analytic gradient of a singlet TDA state.

use super::controls::Controls;
use super::method::{
    prepare_molecular, run_molecular_reference, MolecularMethod, MolecularReference, MolecularSetup,
};
use super::molecular::{reference_report, setup, units};
use super::value::{ToValue, Value};
use super::System;
use crate::constants::{
    AVOGADRO_PER_MOL, BOHR_TO_METER, FINE_STRUCTURE, HARTREE_TO_CM1, HARTREE_TO_EV,
    HARTREE_TO_JOULE, HBAR_J_S,
};
use crate::math::Vec3;
use crate::response::tddft::{TdaSolver, TdaSpin, TdaState};
use crate::scf::ScfResult;
use crate::{DftError, Result};

mod correlated;

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    if matches!(id, "excitations" | "td-state-gradient" | "td-hessian")
        && controls.opt_str("reference")?.is_some_and(|r| {
            matches!(
                r.to_ascii_lowercase().as_str(),
                "roks" | "rohf" | "gks" | "ghf"
            )
        })
        && super::double_hybrid::requested(controls)?
    {
        return Err(DftError::Unsupported {
            feature:"correlated common-orbital or full-spin excitation surface".into(),
            missing:"the corresponding second-order propagator singles/doubles/metric operator and its nuclear jets; a semilocal parent pole cannot replace the requested correlated model".into(),
        });
    }
    if matches!(id, "excitations" | "td-state-gradient" | "td-hessian")
        && controls
            .opt_str("reference")?
            .is_some_and(|r| r.eq_ignore_ascii_case("roks") || r.eq_ignore_ascii_case("rohf"))
    {
        return super::roks_excited::run(id, system, controls);
    }
    if matches!(id, "excitations" | "td-state-gradient" | "td-hessian")
        && controls
            .opt_str("reference")?
            .is_some_and(|r| r.eq_ignore_ascii_case("gks") || r.eq_ignore_ascii_case("ghf"))
    {
        return super::gks::run(id, system, controls);
    }
    if matches!(id, "td-state-gradient" | "td-hessian")
        && super::double_hybrid::requested(controls)?
    {
        return correlated::run(id, system, controls);
    }
    match id {
        "excitations" => excitations(system, controls),
        "ecd-spectrum" => ecd_spectrum(system, controls),
        "td-state-gradient" => state_gradient(system, controls),
        "td-hessian" => state_hessian(system, controls),
        "td-ecd-giao" => london_ecd(system, controls),
        "xanes" => xanes(system, controls),
        "franck-condon" => franck_condon(system, controls),
        "excited-optimization" => excited_optimization(system, controls),
        other => Err(DftError::Unsupported {
            feature: format!("excited-state calculation `{other}`"),
            missing: "a common-driver implementation".into(),
        }),
    }
}

/// Oscillator strength per unit integrated molar absorptivity:
/// `f = K int epsilon(nu) d nu` with `epsilon` in L mol^-1 cm^-1 and `nu` in
/// cm^-1, `K = 1000 ln(10) / (pi N_A r_e)` and `r_e = alpha^2 a0` the
/// classical electron radius in cm. `K = 4.319e-9`.
fn oscillator_per_integrated_absorptivity() -> f64 {
    let a0_cm = BOHR_TO_METER * 100.0;
    let electron_radius_cm = FINE_STRUCTURE * FINE_STRUCTURE * a0_cm;
    1000.0 * std::f64::consts::LN_10
        / (std::f64::consts::PI * AVOGADRO_PER_MOL * electron_radius_cm)
}

/// One atomic unit of rotatory strength, `alpha E_h a0^3`, in 1e-40 erg cm^3.
fn rotatory_au_in_1e40_cgs() -> f64 {
    let hartree_erg = HARTREE_TO_JOULE * 1.0e7;
    let a0_cm = BOHR_TO_METER * 100.0;
    FINE_STRUCTURE * hartree_erg * a0_cm.powi(3) * 1.0e40
}

/// Rotatory strength per unit integral of the reduced circular dichroism:
/// `R = K int (Delta epsilon / nu) d nu` with
/// `K = 3 h c 1000 ln(10) / (32 pi^3 N_A)`, returned in 1e-40 erg cm^3
/// (`K = 22.965`).
fn rotatory_per_integrated_ecd() -> f64 {
    let planck_erg_s = 2.0 * std::f64::consts::PI * HBAR_J_S * 1.0e7;
    // c = a0 E_h / (hbar alpha), converted from m/s to cm/s.
    let light_cm_s = BOHR_TO_METER * HARTREE_TO_JOULE / (HBAR_J_S * FINE_STRUCTURE) * 100.0;
    3.0 * planck_erg_s * light_cm_s * 1000.0 * std::f64::consts::LN_10
        / (32.0 * std::f64::consts::PI.powi(3) * AVOGADRO_PER_MOL)
        * 1.0e40
}

/// Excitation request shared by the three calculations.
struct Request {
    nstates: usize,
    /// True for the Tamm-Dancoff approximation, false for full RPA.
    tamm_dancoff: bool,
    /// `None` selects singlets for a restricted and the spin-resolved
    /// manifold for an unrestricted reference.
    spin: Option<TdaSpin>,
    tolerance: f64,
    max_iterations: usize,
}

impl Request {
    fn from_controls(controls: &Controls, default_states: usize) -> Result<Self> {
        let nstates = controls.usize("nstates", default_states)?;
        let method = controls.str("td_method", "tda")?;
        let spin = controls.str("td_spin", "auto")?;
        let tolerance = controls.f64("td_tolerance", 1e-7)?;
        let max_iterations = controls.usize("td_max_iterations", 60)?;
        if nstates == 0 {
            return Err(DftError::InvalidInput("nstates must be at least 1".into()));
        }
        if !(tolerance > 0.0) || max_iterations == 0 {
            return Err(DftError::InvalidInput(
                "td_tolerance must be positive and td_max_iterations at least 1".into(),
            ));
        }
        let tamm_dancoff = match method.as_str() {
            "tda" => true,
            "rpa" => false,
            other => {
                return Err(DftError::InvalidInput(format!(
                    "unknown td_method `{other}` (tda or rpa)"
                )))
            }
        };
        let spin = match spin.as_str() {
            "auto" => None,
            "singlet" => Some(TdaSpin::Singlet),
            "triplet" => Some(TdaSpin::Triplet),
            "spin-resolved" | "spin_resolved" => Some(TdaSpin::SpinResolved),
            other => {
                return Err(DftError::InvalidInput(format!(
                    "unknown td_spin `{other}` (auto, singlet, triplet or spin-resolved)"
                )))
            }
        };
        Ok(Self {
            nstates,
            tamm_dancoff,
            spin,
            tolerance,
            max_iterations,
        })
    }
}

/// The excitation manifold is built on the integer occupations of a
/// zero-temperature reference.
fn refuse_fractional_reference(setup: &MolecularSetup) -> Result<()> {
    if setup.method.electronic_temperature != 0.0
        || setup.method.electron_offset != [0.0; 2]
        || setup.method.fixed_occupations.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "excitations of a fractionally occupied reference".into(),
            missing: "the occupation-weighted excitation manifold; use \
                electronic_temperature = 0 and no electron_offset"
                .into(),
        });
    }
    Ok(())
}

/// Combinations whose excited-state Lagrangian derivative is not
/// implemented. The library gradient itself refuses open shells, triplet and
/// RPA states and GGA, meta-GGA and hybrid density functionals.
fn refuse_unsupported_gradient(setup: &MolecularSetup) -> Result<()> {
    let refuse = |feature: &str, missing: &str| -> Result<()> {
        Err(DftError::Unsupported {
            feature: feature.into(),
            missing: missing.into(),
        })
    };
    if setup.smd.is_some() || setup.options.pcm.is_some() {
        return refuse(
            "analytic TDA gradient with continuum solvation",
            "the excited-state reaction-field Lagrangian and cavity response",
        );
    }
    if setup.options.external_field.is_some() {
        return refuse(
            "analytic TDA gradient in an external electric field",
            "the field term of the excited-state Lagrangian derivative",
        );
    }
    if setup.method.density_fit.is_some() {
        return refuse(
            "analytic TDA gradient with density fitting",
            "the fitted three-index derivative contraction with the excited-state densities",
        );
    }
    Ok(())
}

/// Solve for the requested states of a converged reference.
fn solve(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    request: &Request,
) -> Result<(Vec<TdaState>, TdaSpin)> {
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let spin = request.spin.unwrap_or(if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    });
    let mut solver = TdaSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        spin,
    )?;
    solver.tol = request.tolerance;
    solver.max_iter = request.max_iterations;
    let states = if request.tamm_dancoff {
        solver.solve(request.nstates)?
    } else {
        solver.solve_rpa(request.nstates)?
    };
    if states.is_empty() {
        return Err(DftError::InvalidInput(
            "the basis has no empty orbitals: there are no excitations".into(),
        ));
    }
    Ok((states, spin))
}

fn spin_name(spin: TdaSpin) -> &'static str {
    match spin {
        TdaSpin::Singlet => "singlet",
        TdaSpin::Triplet => "triplet",
        TdaSpin::SpinResolved => "spin-resolved",
    }
}

/// Largest amplitude of a state: its spin channel, zero-based orbital
/// indices and its share of the squared amplitude norm.
fn dominant_transition(state: &TdaState, scf: &ScfResult) -> Value {
    let spins = if scf.unrestricted { 2 } else { 1 };
    let mut norm = 0.0;
    let mut best = (0usize, 0usize, 0usize, 0.0_f64);
    for spin in 0..spins {
        let x = &state.x[spin];
        for a in 0..x.rows {
            for i in 0..x.cols {
                let weight = x[(a, i)] * x[(a, i)];
                norm += weight;
                if weight > best.3 {
                    best = (spin, i, a, weight);
                }
            }
        }
    }
    let n_occ = if best.0 == 0 { scf.n_alpha } else { scf.n_beta };
    let channel = if !scf.unrestricted {
        "restricted"
    } else if best.0 == 0 {
        "alpha"
    } else {
        "beta"
    };
    Value::object()
        .with("spin", channel)
        .with("occupied_orbital", best.1)
        .with("virtual_orbital", n_occ + best.2)
        .with("weight", if norm > 0.0 { best.3 / norm } else { 0.0 })
}

fn state_value(
    state: &TdaState,
    scf: &ScfResult,
    amplitudes: bool,
    orbitals: super::nto::Request,
) -> Result<Value> {
    let wavenumber = state.energy * HARTREE_TO_CM1;
    let wavelength = if wavenumber > 0.0 {
        Some(1.0e7 / wavenumber)
    } else {
        None
    };
    let mut value = Value::object()
        .with("energy", state.energy)
        .with("energy_ev", state.energy * HARTREE_TO_EV)
        .with("energy_cm1", wavenumber)
        .with("wavelength_nm", wavelength)
        .with("oscillator_strength", state.oscillator_strength)
        .with("transition_dipole", state.transition_dipole)
        .with("dominant_transition", dominant_transition(state, scf));
    // The magnetic and velocity moments are contracted with the same
    // amplitudes as the electric dipole. That is exact in the Tamm-Dancoff
    // approximation; an RPA state needs the X-Y amplitudes for these
    // anti-Hermitian operators, which the solver does not return.
    if state.tamm_dancoff {
        let cgs = rotatory_au_in_1e40_cgs();
        value.set("transition_momentum", state.transition_momentum);
        value.set(
            "transition_angular_momentum",
            state.transition_angular_momentum,
        );
        value.set("rotatory_strength", state.rotatory_strength);
        value.set(
            "rotatory_strength_velocity",
            state.rotatory_strength_velocity,
        );
        value.set("rotatory_strength_cgs", state.rotatory_strength * cgs);
        value.set(
            "rotatory_strength_velocity_cgs",
            state.rotatory_strength_velocity * cgs,
        );
    }
    if amplitudes {
        let spins = if scf.unrestricted { 2 } else { 1 };
        value.set(
            "amplitudes",
            (0..spins)
                .map(|spin| state.x[spin].to_value())
                .collect::<Vec<Value>>(),
        );
    }
    if orbitals.enabled() {
        let channels = match state.spin {
            TdaSpin::Singlet => vec![super::nto::real_excitation(
                &state.x[0], &scf.mo_coeff[0], "charge", std::f64::consts::SQRT_2, orbitals,
            )?],
            TdaSpin::Triplet => vec![super::nto::real_excitation(
                &state.x[0], &scf.mo_coeff[0], "spin-difference", std::f64::consts::SQRT_2, orbitals,
            )?],
            TdaSpin::SpinResolved => ["alpha", "beta"].iter().enumerate()
                .map(|(s, name)| super::nto::real_excitation(&state.x[s], &scf.mo_coeff[s], name, 1.0, orbitals))
                .collect::<Result<Vec<_>>>()?,
        };
        orbitals.attach(&mut value, channels);
    }
    Ok(value)
}

fn push_warning(report: &mut Value, text: &str) {
    if let Value::Object(entries) = report {
        if let Some((_, Value::Array(warnings))) = entries.iter_mut().find(|(k, _)| k == "warnings")
        {
            warnings.push(Value::Str(text.into()));
        }
    }
}

fn set_states(
    report: &mut Value,
    request: &Request,
    spin: TdaSpin,
    states: &[TdaState],
    scf: &ScfResult,
    amplitudes: bool,
    orbitals: super::nto::Request,
) -> Result<()> {
    report.set(
        "td_method",
        if request.tamm_dancoff { "tda" } else { "rpa" },
    );
    report.set("td_spin", spin_name(spin));
    report.set(
        "states",
        states
            .iter()
            .map(|state| state_value(state, scf, amplitudes, orbitals))
            .collect::<Result<Vec<Value>>>()?,
    );
    report.set(
        "oscillator_strength_sum",
        states.iter().map(|s| s.oscillator_strength).sum::<f64>(),
    );
    Ok(())
}

const STATE_UNITS: [(&str, &str); 14] = [
    ("energy", "hartree (ground state)"),
    ("states.energy", "hartree (excitation energy)"),
    ("states.energy_ev", "eV"),
    ("states.energy_cm1", "cm^-1"),
    ("states.wavelength_nm", "nm"),
    (
        "states.oscillator_strength",
        "dimensionless, length gauge: (2/3) omega |d|^2",
    ),
    ("states.transition_dipole", "e*bohr"),
    (
        "states.transition_momentum",
        "1/bohr, real coefficient of <0|grad|n>",
    ),
    (
        "states.transition_angular_momentum",
        "dimensionless, real coefficient of <0|r x grad|n> about the coordinate origin",
    ),
    (
        "states.rotatory_strength",
        "atomic units, length gauge (depends on the coordinate origin in a finite basis)",
    ),
    (
        "states.rotatory_strength_velocity",
        "atomic units, velocity gauge (origin independent)",
    ),
    ("states.rotatory_strength_cgs", "1e-40 erg cm^3"),
    ("states.rotatory_strength_velocity_cgs", "1e-40 erg cm^3"),
    (
        "states.amplitudes",
        "[spin][virtual][occupied]: X for tda, X+Y for rpa",
    ),
];

fn excitations(system: &System, controls: &Controls) -> Result<Value> {
    let kind = super::method::CollinearPropertyReference::from_controls(controls)?;
    let setup = setup(system, controls)?;
    let request = Request::from_controls(controls, 5)?;
    let amplitudes = controls.bool("amplitudes", false)?;
    let orbitals = super::nto::Request::from_controls(controls)?;
    if orbitals.enabled() && !request.tamm_dancoff {
        return Err(DftError::Unsupported {
            feature: "full RPA transition and difference density orbitals".into(),
            missing: "both directed X and Y transition-density blocks".into(),
        });
    }
    controls.finish("excitations")?;
    refuse_fractional_reference(&setup)?;
    let reference = super::method::run_collinear_property_reference(&setup, kind)?;
    let (states, spin) = solve(&setup, &reference, &request)?;
    let mut report = reference_report(
        &setup,
        &reference,
        "vertical excitation energies and transition moments from linear-response TDA/RPA",
    );
    set_states(
        &mut report,
        &request,
        spin,
        &states,
        &reference.scf,
        amplitudes,
        orbitals,
    )?;
    if !request.tamm_dancoff {
        push_warning(
            &mut report,
            "WARNING [excitations] rotatory strengths and the magnetic and velocity transition \
             moments are reported for td_method = tda only",
        );
    }
    let mut labels = units(&STATE_UNITS);
    super::nto::add_units(&mut labels, orbitals);
    report.set("units", labels);
    Ok(report)
}

struct Spectrum {
    energies_ev: Vec<f64>,
    wavenumbers: Vec<f64>,
    absorption: Vec<f64>,
    ecd_length: Vec<f64>,
    ecd_velocity: Vec<f64>,
    /// `K_abs int epsilon d nu` over the window.
    absorption_integral: f64,
    /// `K_ecd int (Delta epsilon / nu) d nu` over the window in the length
    /// and velocity gauges, 1e-40 erg cm^3.
    ecd_integral: [f64; 2],
}

/// Unit-area Gaussian bands on a uniform energy grid.
///
/// `epsilon(nu) = (1/K_abs) sum_n f_n g_n(nu)` and
/// `Delta epsilon(nu) = (nu/K_ecd) sum_n R_n g_n(nu)`, where `g_n` is the
/// Gaussian of standard deviation `sigma` centred on state `n` with unit
/// area in wavenumber. Both curves then integrate exactly to their line
/// strengths: `K_abs int epsilon d nu = sum f` and
/// `K_ecd int (Delta epsilon / nu) d nu = sum R`.
fn broadened_spectrum(
    states: &[TdaState],
    sigma_ev: f64,
    lo_ev: f64,
    hi_ev: f64,
    points: usize,
) -> Spectrum {
    let ev_to_cm1 = HARTREE_TO_CM1 / HARTREE_TO_EV;
    let k_abs = oscillator_per_integrated_absorptivity();
    let k_ecd = rotatory_per_integrated_ecd();
    let cgs = rotatory_au_in_1e40_cgs();
    // Unit area in eV, converted to unit area in cm^-1.
    let norm = 1.0 / (sigma_ev * (2.0 * std::f64::consts::PI).sqrt() * ev_to_cm1);
    let step_ev = (hi_ev - lo_ev) / (points - 1) as f64;
    let mut energies_ev = Vec::with_capacity(points);
    let mut wavenumbers = Vec::with_capacity(points);
    let mut absorption = vec![0.0; points];
    let mut ecd_length = vec![0.0; points];
    let mut ecd_velocity = vec![0.0; points];
    // Delta epsilon / nu, kept separately so that the sum rule needs no
    // division by a wavenumber that is zero at the lower window edge.
    let mut reduced = vec![[0.0_f64; 2]; points];
    for i in 0..points {
        let energy = lo_ev + step_ev * i as f64;
        let wavenumber = energy * ev_to_cm1;
        for state in states {
            let t = (energy - state.energy * HARTREE_TO_EV) / sigma_ev;
            if t.abs() >= 9.0 {
                continue;
            }
            let g = norm * (-0.5 * t * t).exp();
            absorption[i] += state.oscillator_strength * g / k_abs;
            reduced[i][0] += state.rotatory_strength * cgs * g / k_ecd;
            reduced[i][1] += state.rotatory_strength_velocity * cgs * g / k_ecd;
        }
        ecd_length[i] = wavenumber * reduced[i][0];
        ecd_velocity[i] = wavenumber * reduced[i][1];
        energies_ev.push(energy);
        wavenumbers.push(wavenumber);
    }
    // Trapezoid rule in wavenumber.
    let step = step_ev * ev_to_cm1;
    let mut absorption_integral = 0.0;
    let mut ecd_integral = [0.0_f64; 2];
    for i in 1..points {
        absorption_integral += 0.5 * step * (absorption[i] + absorption[i - 1]);
        for gauge in 0..2 {
            ecd_integral[gauge] += 0.5 * step * (reduced[i][gauge] + reduced[i - 1][gauge]);
        }
    }
    Spectrum {
        energies_ev,
        wavenumbers,
        absorption,
        ecd_length,
        ecd_velocity,
        absorption_integral: k_abs * absorption_integral,
        ecd_integral: [k_ecd * ecd_integral[0], k_ecd * ecd_integral[1]],
    }
}

fn ecd_spectrum(system: &System, controls: &Controls) -> Result<Value> {
    let request = Request::from_controls(controls, 10)?;
    let sigma = controls.opt_f64("broadening_ev")?.ok_or_else(|| {
        DftError::InvalidInput(
            "ecd-spectrum needs broadening_ev, the Gaussian standard deviation in eV".into(),
        )
    })?;
    let points = controls.usize("points", 1000)?;
    let window = controls.opt_f64_list("energy_window_ev")?;
    if !(sigma > 0.0) || points < 2 {
        return Err(DftError::InvalidInput(
            "ecd-spectrum needs broadening_ev > 0 and points >= 2".into(),
        ));
    }
    if !request.tamm_dancoff {
        return Err(DftError::Unsupported {
            feature: "circular-dichroism spectrum from RPA states".into(),
            missing: "the X-Y amplitudes for the magnetic transition moments; use td_method = tda"
                .into(),
        });
    }
    if request.spin == Some(TdaSpin::Triplet) {
        return Err(DftError::InvalidInput(
            "ecd-spectrum needs spin-allowed states: singlet-triplet transitions carry no \
             intensity without spin-orbit coupling"
                .into(),
        ));
    }
    let correlated = if super::double_hybrid::requested(controls)? {
        Some(super::double_hybrid::prepare(
            system,
            controls,
            "ecd-spectrum",
        )?)
    } else {
        None
    };
    let ordinary = if correlated.is_none() {
        let setup = setup(system, controls)?;
        let reference = run_molecular_reference(&setup)?;
        Some((setup, reference))
    } else {
        None
    };
    controls.finish("ecd-spectrum")?;
    let (setup, reference) = if let Some(data) = &correlated {
        (&data.setup, &data.reference)
    } else {
        let (setup, reference) = ordinary.as_ref().expect("ordinary spectral reference");
        (setup, reference)
    };
    refuse_fractional_reference(setup)?;
    let spin = request.spin.unwrap_or(if reference.scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    });
    let mut correlation = if let Some(data) = &correlated {
        let mut parent = TdaSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            spin,
        )?;
        parent.tol = request.tolerance;
        parent.max_iter = request.max_iterations;
        Some(crate::response::correlated_tda::correlated_tda_states(
            &parent,
            &setup.basis,
            &reference.scf,
            &data.eri,
            &data.problem,
            &data.pt2,
            data.coefficients,
            request.nstates,
            controls.usize("pt2_memory_mb", 2048)?,
        )?)
    } else {
        None
    };
    let states = if let Some(data) = &mut correlation {
        std::mem::take(&mut data.states)
    } else {
        solve(setup, reference, &request)?.0
    };
    let lowest = states
        .iter()
        .map(|s| s.energy)
        .fold(f64::INFINITY, f64::min)
        * HARTREE_TO_EV;
    let highest = states
        .iter()
        .map(|s| s.energy)
        .fold(f64::NEG_INFINITY, f64::max)
        * HARTREE_TO_EV;
    let (lo, hi) = match window.as_deref() {
        Some([a, b]) if *a >= 0.0 && a < b => (*a, *b),
        Some(_) => {
            return Err(DftError::InvalidInput(
                "energy_window_ev takes [minimum, maximum] in eV with 0 <= minimum < maximum"
                    .into(),
            ))
        }
        None => ((lowest - 5.0 * sigma).max(0.0), highest + 5.0 * sigma),
    };
    let spectrum = broadened_spectrum(&states, sigma, lo, hi, points);
    let cgs = rotatory_au_in_1e40_cgs();
    let rotatory_sum = [
        states.iter().map(|s| s.rotatory_strength).sum::<f64>() * cgs,
        states
            .iter()
            .map(|s| s.rotatory_strength_velocity)
            .sum::<f64>()
            * cgs,
    ];
    let mut report = reference_report(
        setup,
        reference,
        "Gaussian-broadened vertical absorption and electronic circular-dichroism spectra from \
         TDA states",
    );
    set_states(&mut report, &request, spin, &states, &reference.scf, false, Default::default())?;
    report.set("broadening_ev", sigma);
    report.set("energies_ev", spectrum.energies_ev);
    report.set("wavenumbers_cm1", spectrum.wavenumbers);
    report.set("molar_absorptivity", spectrum.absorption);
    report.set("ecd_length", spectrum.ecd_length);
    report.set("ecd_velocity", spectrum.ecd_velocity);
    report.set(
        "absorption_integral_oscillator_strength",
        spectrum.absorption_integral,
    );
    report.set("rotatory_strength_sum_cgs", rotatory_sum);
    report.set("ecd_integral_rotatory_strength_cgs", spectrum.ecd_integral);
    let mut pairs = STATE_UNITS.to_vec();
    pairs.extend([
        ("broadening_ev", "eV, Gaussian standard deviation"),
        ("energies_ev", "eV"),
        ("wavenumbers_cm1", "cm^-1"),
        (
            "molar_absorptivity",
            "L mol^-1 cm^-1, decadic molar absorption coefficient",
        ),
        (
            "ecd_length",
            "L mol^-1 cm^-1, Delta epsilon from length-gauge rotatory strengths",
        ),
        (
            "ecd_velocity",
            "L mol^-1 cm^-1, Delta epsilon from velocity-gauge rotatory strengths",
        ),
        (
            "absorption_integral_oscillator_strength",
            "dimensionless, 4.319e-9 times the integral of molar_absorptivity over cm^-1",
        ),
        (
            "rotatory_strength_sum_cgs",
            "1e-40 erg cm^3, [length, velocity]",
        ),
        (
            "ecd_integral_rotatory_strength_cgs",
            "1e-40 erg cm^3, [length, velocity]: 22.965 times the integral of Delta epsilon / nu",
        ),
    ]);
    if let Some(data) = &correlation {
        report.set("model", data.model);
        report.set(
            "pt2_parent_excitation_energies",
            data.parent_energies.clone(),
        );
        report.set(
            "pt2_excitation_energy_corrections",
            data.energy_corrections.clone(),
        );
        report.set(
            "pt2_transition_singles_correction_norms",
            data.singles_correction_norms.clone(),
        );
        report.set("pt2_transition_doubles_norms", data.doubles_norms.clone());
        report.set("pt2_transition_pole_residuals", data.pole_residuals.clone());
        report.set(
            "pt2_transition_normalization_defects",
            data.normalization_defects.clone(),
        );
        report.set("pt2_parent_transition_residual", data.parent_residual);
        report.set(
            "pt2_transition_smallest_double_denominators",
            Value::Array(
                data.smallest_double_denominators
                    .iter()
                    .map(|&v| {
                        if v.is_finite() {
                            Value::Float(v)
                        } else {
                            Value::Null
                        }
                    })
                    .collect(),
            ),
        );
        let converged = data.parent_residual <= request.tolerance
            && data.pole_residuals.iter().all(|r| *r <= request.tolerance)
            && data
                .normalization_defects
                .iter()
                .all(|r| *r <= request.tolerance);
        report.set("response_converged", converged);
        report.set("converged", reference.scf.converged && converged);
        pairs.extend([
            ("pt2_parent_excitation_energies", "hartree"),
            ("pt2_excitation_energy_corrections", "hartree"),
            ("pt2_transition_singles_correction_norms", "dimensionless"),
            (
                "pt2_transition_doubles_norms",
                "dimensionless, signed spin-weighted norm",
            ),
            ("pt2_transition_pole_residuals", "hartree"),
            ("pt2_transition_normalization_defects", "dimensionless"),
            ("pt2_parent_transition_residual", "dimensionless"),
            ("pt2_transition_smallest_double_denominators", "hartree"),
            ("response_converged", "dimensionless"),
        ]);
    }
    report.set("units", units(&pairs));
    if let Some(data) = correlated {
        super::double_hybrid::append_prepared_energy(data, report)
    } else {
        Ok(report)
    }
}

#[derive(Clone, Copy)]
enum GradientRoute {
    ZVector,
    Path,
}

impl GradientRoute {
    fn from_controls(controls: &Controls) -> Result<Self> {
        let default = if controls.str("td_method", "tda")? == "rpa" {
            "path"
        } else {
            "z-vector"
        };
        match controls.str("td_gradient_route", default)?.as_str() {
            "z-vector" => Ok(Self::ZVector),
            "path" => Ok(Self::Path),
            other => Err(DftError::InvalidInput(format!(
                "unknown td_gradient_route `{other}` (z-vector or path)"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::ZVector => "z-vector",
            Self::Path => "path",
        }
    }
}

/// Shared analytic gradient for a fixed state. No displaced SCF calculation
/// enters the path route; every coordinate is a first-order nuclear jet.
fn excited_gradient(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    state: &TdaState,
    route: GradientRoute,
    config: &crate::response::krylov::ResponseSolverConfig,
) -> Result<(Vec<Vec3>, Option<f64>)> {
    if matches!(route, GradientRoute::ZVector) {
        return Ok((
            crate::response::tddft_grad::tda_gradient_with_grid(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                state,
                &setup.options,
            )?,
            None,
        ));
    }
    let natoms = setup.molecule.len();
    let mut values = vec![Vec3::zero(); natoms];
    let mut residual = 0.0_f64;
    for (atom, value) in values.iter_mut().enumerate() {
        for axis in 0..3 {
            let mut direction = vec![[0.0; 3]; natoms];
            direction[atom][axis] = 1.0;
            let path = crate::response::nuclear::excitation_path(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
                state,
                &direction,
                1,
                config,
            )?;
            let path_residual = path
                .residuals
                .iter()
                .copied()
                .fold(path.reference_residual, f64::max);
            if !path.converged {
                return Err(DftError::ResponseNotConverged {
                    iterations: config.max_iterations,
                    residual: path_residual,
                });
            }
            residual = residual.max(path_residual);
            let coefficient = path.ground[1] + path.excitation[1];
            match axis {
                0 => value.x = coefficient,
                1 => value.y = coefficient,
                _ => value.z = coefficient,
            }
        }
    }
    Ok((values, Some(residual)))
}

fn state_gradient(system: &System, controls: &Controls) -> Result<Value> {
    let setup = setup(system, controls)?;
    let root = controls.usize("root", 1)?;
    if root == 0 {
        return Err(DftError::InvalidInput(
            "root counts excited states from 1".into(),
        ));
    }
    let request = Request::from_controls(controls, root)?;
    let route = GradientRoute::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    controls.finish("td-state-gradient")?;
    let by_path = matches!(route, GradientRoute::Path);
    if request.nstates < root {
        return Err(DftError::InvalidInput(format!(
            "nstates = {} does not reach root {root}",
            request.nstates
        )));
    }
    refuse_fractional_reference(&setup)?;
    if !by_path {
        refuse_unsupported_gradient(&setup)?;
    }
    let reference = run_molecular_reference(&setup)?;
    let (states, spin) = solve(&setup, &reference, &request)?;
    let state = states.get(root - 1).ok_or_else(|| {
        DftError::InvalidInput(format!(
            "root {root} exceeds the {} excitations of this basis",
            states.len()
        ))
    })?;
    let (mut gradient, path_residual) =
        excited_gradient(&setup, &reference, state, route, &config)?;
    if setup.dispersion != crate::dispersion::Dispersion::None {
        let correction = crate::dispersion::dispersion_gradient(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?;
        for (total, delta) in gradient.iter_mut().zip(correction) {
            *total += delta;
        }
    }
    let mut sum = Vec3::zero();
    for g in &gradient {
        sum += *g;
    }
    let mut report = reference_report(
        &setup,
        &reference,
        if by_path {
            "analytic nuclear gradient of the total energy of an excited state from first-order \
             Taylor coefficients of the reference and excitation energies along nuclear paths"
        } else {
            "analytic nuclear gradient of the total energy of a singlet TDA excited state (Z-vector)"
        },
    );
    report.set(
        "td_method",
        if by_path && !request.tamm_dancoff {
            "rpa"
        } else {
            "tda"
        },
    );
    report.set("td_gradient_route", route.name());
    if let Some(residual) = path_residual {
        report.set("response_residual", residual);
    }
    report.set("td_spin", spin_name(spin));
    report.set("root", root);
    report.set("excitation_energy", state.energy);
    report.set(
        "excited_state_energy",
        reference.total_energy() + state.energy,
    );
    report.set("oscillator_strength", state.oscillator_strength);
    report.set(
        "gradient",
        gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
    );
    report.set("translation_residual", sum.norm());
    report.set("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree (ground state)"),
            ("excitation_energy", "hartree"),
            (
                "excited_state_energy",
                "hartree (ground state plus excitation)",
            ),
            ("gradient", "hartree/bohr, of excited_state_energy"),
            ("translation_residual", "hartree/bohr"),
            ("response_residual", "dimensionless"),
        ]),
    );
    Ok(report)
}

/// Analytic Hessian of the total energy of an excited state with its
/// harmonic analysis. The excitation energy is expanded along nuclear paths;
/// every Cartesian direction and every pair sum gives the second derivatives
/// by the polarization identity.
/// Length-gauge rotatory strengths with London orbitals of the TDA or RPA
/// manifold, next to the common-origin and velocity values of the same
/// states.
fn london_ecd(system: &System, controls: &Controls) -> Result<Value> {
    let config = super::vibrational::response_config(controls)?;
    let setup = setup(system, controls)?;
    let request = Request::from_controls(controls, 5)?;
    let origin = controls
        .opt_vec3("gauge_origin")?
        .unwrap_or_else(Vec3::zero);
    controls.finish("td-ecd-giao")?;
    refuse_fractional_reference(&setup)?;
    let reference = run_molecular_reference(&setup)?;
    let (states, spin) = solve(&setup, &reference, &request)?;
    let result = crate::response::london_ecd::london_rotatory_strengths(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        &states,
        origin,
        &config,
    )?;
    let cgs = rotatory_au_in_1e40_cgs();
    let rows: Vec<Value> = result
        .states
        .iter()
        .map(|state| {
            Value::object()
                .with("energy", state.energy)
                .with("energy_ev", state.energy * HARTREE_TO_EV)
                .with("transition_dipole", state.transition_dipole)
                .with("magnetic_gradient_static", state.static_part)
                .with("magnetic_gradient_centre", state.centre_part)
                .with("rotatory_strength_london", state.london)
                .with("rotatory_strength_london_cgs", state.london * cgs)
                .with("rotatory_strength", state.common_origin)
                .with("rotatory_strength_velocity", state.velocity)
        })
        .collect();
    let mut report = reference_report(
        &setup,
        &reference,
        "length-gauge rotatory strengths with London orbitals: the static London property \
         gradient of the magnetic response equations on X - Y plus the ket-centre term of the \
         moving basis on X + Y, contracted with the electric transition dipole",
    );
    report.set(
        "td_method",
        if request.tamm_dancoff { "tda" } else { "rpa" },
    );
    report.set("td_spin", spin_name(spin));
    report.set("gauge_origin", origin);
    report.set("states", rows);
    report.set("magnetic_response_residual", result.magnetic_residual);
    report.set("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree (ground state)"),
            ("gauge_origin", "bohr"),
            ("states.energy", "hartree (excitation energy)"),
            ("states.energy_ev", "eV"),
            ("states.transition_dipole", "e*bohr"),
            (
                "states.magnetic_gradient_static",
                "atomic units, g . (X - Y) per field axis",
            ),
            (
                "states.magnetic_gradient_centre",
                "atomic units, omega W . (X + Y) per field axis",
            ),
            (
                "states.rotatory_strength_london",
                "atomic units, length gauge with London orbitals",
            ),
            ("states.rotatory_strength_london_cgs", "1e-40 erg cm^3"),
            (
                "states.rotatory_strength",
                "atomic units, length gauge about gauge_origin",
            ),
            (
                "states.rotatory_strength_velocity",
                "atomic units, velocity gauge",
            ),
            ("magnetic_response_residual", "dimensionless"),
        ]),
    );
    Ok(report)
}

fn state_hessian(system: &System, controls: &Controls) -> Result<Value> {
    let config = super::vibrational::response_config(controls)?;
    let setup = setup(system, controls)?;
    let root = controls.usize("root", 1)?;
    if root == 0 {
        return Err(DftError::InvalidInput(
            "root counts excited states from 1".into(),
        ));
    }
    let request = Request::from_controls(controls, root)?;
    controls.finish("td-hessian")?;
    if request.nstates < root {
        return Err(DftError::InvalidInput(format!(
            "nstates = {} does not reach root {root}",
            request.nstates
        )));
    }
    refuse_fractional_reference(&setup)?;
    if setup.smd.is_some() {
        return Err(DftError::Unsupported {
            feature: "excited-state Hessian with the SMD non-electrostatic term".into(),
            missing: "the surface-tension second derivative".into(),
        });
    }
    let reference = run_molecular_reference(&setup)?;
    let (states, spin) = solve(&setup, &reference, &request)?;
    let state = states.get(root - 1).ok_or_else(|| {
        DftError::InvalidInput(format!(
            "root {root} exceeds the {} excitations of this basis",
            states.len()
        ))
    })?;
    let result = crate::response::nuclear::excited_state_hessian(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        state,
        &config,
    )?;
    let mut hessian = result.hessian.clone();
    let mut ground_hessian = result.ground_hessian.clone();
    let mut gradient = result.gradient.clone();
    if setup.dispersion != crate::dispersion::Dispersion::None {
        let name = setup.method.dispersion_functional_name();
        let correction =
            crate::dispersion::dispersion_hessian(&setup.molecule, &setup.dispersion, &name)?;
        for (total, delta) in hessian.as_mut_slice().iter_mut().zip(correction.as_slice()) {
            *total += *delta;
        }
        for (ground, delta) in ground_hessian
            .as_mut_slice()
            .iter_mut()
            .zip(correction.as_slice())
        {
            *ground += *delta;
        }
        let forces =
            crate::dispersion::dispersion_gradient(&setup.molecule, &setup.dispersion, &name)?;
        for (atom, delta) in forces.iter().enumerate() {
            let delta = delta.to_array();
            for axis in 0..3 {
                gradient[3 * atom + axis] += delta[axis];
            }
        }
    }
    let modes =
        crate::properties::vibrational::vibrational_analysis(&setup.molecule, &hessian, None)?;
    let rows = |m: &crate::linalg::Matrix| -> Vec<Vec<f64>> {
        (0..m.rows)
            .map(|r| (0..m.cols).map(|c| m[(r, c)]).collect())
            .collect()
    };
    let mut report = reference_report(
        &setup,
        &reference,
        "analytic Hessian of the total energy of an excited state from the Taylor series of \
         the excitation energy along nuclear paths",
    );
    report.set(
        "td_method",
        if request.tamm_dancoff { "tda" } else { "rpa" },
    );
    report.set("td_spin", spin_name(spin));
    report.set("root", root);
    report.set("excitation_energy", state.energy);
    report.set(
        "excited_state_energy",
        reference.total_energy() + state.energy,
    );
    report.set(
        "gradient",
        gradient
            .chunks_exact(3)
            .map(|g| [g[0], g[1], g[2]])
            .collect::<Vec<_>>(),
    );
    report.set("hessian", rows(&hessian));
    report.set("ground_hessian", rows(&ground_hessian));
    report.set(
        "hessian_quantity",
        "excited-state total energy, reference plus excitation",
    );
    report.set(
        "excitation_hessian_quantity",
        "vertical excitation energy only",
    );
    report.set("excitation_hessian", rows(&result.excitation_hessian));
    report.set("frequencies_cm1", modes.frequencies_cm.clone());
    report.set("n_imaginary", modes.n_imaginary);
    report.set("n_projected", modes.n_projected);
    report.set("reference_residual", result.reference_residual);
    report.set("response_residual", result.residual);
    report.set("response_converged", result.converged);
    report.set("path_evaluations", result.paths);
    report.set("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree (ground state)"),
            ("excitation_energy", "hartree"),
            (
                "excited_state_energy",
                "hartree (ground state plus excitation)",
            ),
            ("gradient", "hartree/bohr, of excited_state_energy"),
            ("hessian", "hartree/bohr^2, of excited_state_energy"),
            ("ground_hessian", "hartree/bohr^2, of ground-state energy"),
            ("excitation_hessian", "hartree/bohr^2"),
            ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
            ("reference_residual", "dimensionless"),
            ("response_residual", "dimensionless"),
        ]),
    );
    for warning in &result.warnings {
        push_warning(&mut report, warning);
    }
    Ok(report)
}

/// Occupied orbitals of the K shell of the atoms with atomic number
/// `element`: the lowest-energy occupied orbitals whose Mulliken population
/// on those atoms exceeds one half, one per atom.
fn k_shell_orbitals(
    setup: &MolecularSetup,
    scf: &ScfResult,
    spin: usize,
    element: u8,
) -> Result<Vec<usize>> {
    let atoms: Vec<usize> = setup
        .molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.z == element)
        .map(|(index, _)| index)
        .collect();
    if atoms.is_empty() {
        return Err(DftError::InvalidInput(format!(
            "xanes_element names atomic number {element}, which the molecule does not contain"
        )));
    }
    if setup.basis.ecps.iter().any(|ecp| atoms.contains(&ecp.atom)) {
        return Err(DftError::Unsupported {
            feature: format!("core-level absorption of atomic number {element}"),
            missing: "its core orbitals, which an effective core potential replaces".into(),
        });
    }
    let ao_atom = setup.basis.sph_ao_atoms();
    let c = &scf.mo_coeff[spin];
    let sc = scf.overlap.matmul(c);
    let n_occ = if spin == 0 { scf.n_alpha } else { scf.n_beta };
    // The orbitals are ordered by energy.
    let mut found = Vec::new();
    for i in 0..n_occ {
        let population: f64 = (0..c.rows)
            .filter(|mu| atoms.contains(&ao_atom[*mu]))
            .map(|mu| c[(mu, i)] * sc[(mu, i)])
            .sum();
        if population > 0.5 {
            found.push(i);
            if found.len() == atoms.len() {
                break;
            }
        }
    }
    if found.len() != atoms.len() {
        return Err(DftError::InvalidInput(format!(
            "found {} occupied orbitals localized on the {} atoms of atomic number {element}; \
             give core_orbitals explicitly",
            found.len(),
            atoms.len()
        )));
    }
    Ok(found)
}

/// Core-level absorption: Tamm-Dancoff states in the core-valence-separated
/// space of the K shell of one element (or of the occupied orbitals of
/// `core_orbitals`), with oscillator strengths and an optional
/// Lorentzian-broadened spectrum. The excitation energies are those of the
/// frozen orbitals; with `core_hole = delta-scf` the state with a hole in
/// the first core orbital is converged with maximum-overlap occupations,
/// and its relaxation shifts the reported energies uniformly.
fn xanes(system: &System, controls: &Controls) -> Result<Value> {
    let setup = setup(system, controls)?;
    let request = Request::from_controls(controls, 10)?;
    let element = controls.opt_usize("xanes_element")?;
    let explicit = controls.opt_usize_list("core_orbitals")?;
    let broadening = controls.opt_f64("xanes_broadening")?;
    let points = controls.usize("points", 2001)?;
    let amplitudes = controls.bool("amplitudes", false)?;
    let orbitals = super::nto::Request::from_controls(controls)?;
    let core_hole = controls.str("core_hole", "frozen")?;
    controls.finish("xanes")?;
    let relaxed_hole = match core_hole.as_str() {
        "frozen" => false,
        "delta-scf" => true,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown core_hole `{other}` (frozen or delta-scf)"
            )))
        }
    };
    if !request.tamm_dancoff {
        return Err(DftError::Unsupported {
            feature: "core-level absorption with td_method = rpa".into(),
            missing: "the core-valence-separated form of the full response equations".into(),
        });
    }
    if broadening.is_some_and(|width| !(width.is_finite() && width > 0.0)) || points < 2 {
        return Err(DftError::InvalidInput(
            "xanes_broadening is positive and points is at least 2".into(),
        ));
    }
    refuse_fractional_reference(&setup)?;
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let spin = request.spin.unwrap_or(if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    });
    let core: [Vec<usize>; 2] = match (&explicit, element) {
        (Some(list), None) => [
            list.clone(),
            if scf.unrestricted {
                // An orbital index that only the alpha spin occupies (the
                // open shell) has no beta hole.
                list.iter().copied().filter(|&i| i < scf.n_beta).collect()
            } else {
                Vec::new()
            },
        ],
        (None, Some(number)) => {
            let number = u8::try_from(number)
                .map_err(|_| DftError::InvalidInput("xanes_element is an atomic number".into()))?;
            [
                k_shell_orbitals(&setup, scf, 0, number)?,
                if scf.unrestricted {
                    k_shell_orbitals(&setup, scf, 1, number)?
                } else {
                    Vec::new()
                },
            ]
        }
        _ => {
            return Err(DftError::InvalidInput(
                "xanes needs exactly one of xanes_element (the atomic number of the absorbing \
                 element, K edge) and core_orbitals (occupied orbital indices counted from 0)"
                    .into(),
            ))
        }
    };
    let mut solver = TdaSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        spin,
    )?;
    solver.tol = request.tolerance;
    solver.max_iter = request.max_iterations;
    let states = solver.solve_core([&core[0], &core[1]], request.nstates)?;
    if states.is_empty() {
        return Err(DftError::InvalidInput(
            "the core-valence-separated space is empty: no core orbital or no empty orbital".into(),
        ));
    }
    let mut report = reference_report(
        &setup,
        &reference,
        "core-level excitations from Tamm-Dancoff linear response in the core-valence-separated \
         space with frozen orbitals; core_hole = delta-scf adds the uniform shift of a \
         self-consistent core-hole state",
    );
    set_states(&mut report, &request, spin, &states, scf, amplitudes, orbitals)?;
    report.set("core_orbitals", core[0].clone());
    report.set(
        "core_orbital_energies",
        core[0]
            .iter()
            .map(|i| scf.mo_energy[0][*i])
            .collect::<Vec<_>>(),
    );
    let mut unit_pairs: Vec<(&str, &str)> = STATE_UNITS.to_vec();
    unit_pairs.push(("core_orbital_energies", "hartree"));
    report.set("core_hole", core_hole.clone());
    // Relaxed ionization energy of the core orbital minus its frozen-orbital
    // estimate: the uniform shift of the excitation energies.
    let mut shift = 0.0;
    if relaxed_hole {
        let orbital = *core[0].first().ok_or_else(|| {
            DftError::InvalidInput("the core-hole state needs a core orbital".into())
        })?;
        let hole = crate::scf::core_hole::run_core_hole(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            0,
            orbital,
        )?;
        if !hole.converged {
            return Err(DftError::ScfNotConverged {
                iterations: hole.iterations,
                error: f64::NAN,
            });
        }
        let ionization = hole.energy - scf.energy;
        shift = ionization + scf.mo_energy[0][orbital];
        report.set("core_ionization_energy", ionization);
        report.set("core_hole_shift", shift);
        report.set("core_hole_overlap", hole.smallest_overlap);
        report.set("core_hole_iterations", hole.iterations);
        report.set(
            "energies_relaxed",
            states.iter().map(|s| s.energy + shift).collect::<Vec<_>>(),
        );
        unit_pairs.push(("core_ionization_energy", "hartree"));
        unit_pairs.push((
            "core_hole_shift",
            "hartree, relaxed ionization energy plus the orbital energy of the hole",
        ));
        unit_pairs.push(("core_hole_overlap", "dimensionless"));
        unit_pairs.push((
            "energies_relaxed",
            "hartree, excitation energies plus core_hole_shift",
        ));
        if core[0].len() > 1 {
            push_warning(
                &mut report,
                "WARNING [xanes] several core orbitals: the hole is in the first one; a hole \
                 delocalized over equivalent atoms underestimates the relaxation",
            );
        }
    }
    if let Some(width) = broadening {
        let half = 0.5 * width;
        let low = states
            .iter()
            .map(|s| s.energy)
            .fold(f64::INFINITY, f64::min)
            - 10.0 * width;
        let high = states
            .iter()
            .map(|s| s.energy)
            .fold(f64::NEG_INFINITY, f64::max)
            + 10.0 * width;
        let step = (high - low) / (points - 1) as f64;
        let axis: Vec<f64> = (0..points).map(|i| low + step * i as f64).collect();
        let density: Vec<f64> = axis
            .iter()
            .map(|e| {
                states
                    .iter()
                    .map(|s| {
                        s.oscillator_strength * half
                            / (std::f64::consts::PI * ((e - s.energy).powi(2) + half * half))
                    })
                    .sum()
            })
            .collect();
        if relaxed_hole {
            report.set(
                "spectrum_energies_relaxed",
                axis.iter().map(|e| e + shift).collect::<Vec<_>>(),
            );
            unit_pairs.push(("spectrum_energies_relaxed", "hartree"));
        }
        report.set("spectrum_energies", axis);
        report.set("spectrum_oscillator_density", density);
        report.set("spectrum_broadening", width);
        unit_pairs.push(("spectrum_energies", "hartree"));
        unit_pairs.push((
            "spectrum_oscillator_density",
            "1/hartree, oscillator strength per energy",
        ));
        unit_pairs.push(("spectrum_broadening", "hartree, full width at half maximum"));
    }
    let mut labels = units(&unit_pairs);
    super::nto::add_units(&mut labels, orbitals);
    report.set("units", labels);
    Ok(report)
}

/// Vibrational quanta of a progression: every assignment with at most
/// `factors[mode].len() - 1` quanta per mode and `remaining` in all whose
/// Franck-Condon factor stays at or above `threshold`. `factors[mode][n]` is
/// the Poisson weight of `n` quanta, which never exceeds one, so a product
/// below the threshold cannot recover.
fn progression(
    mode: usize,
    remaining: usize,
    weight: f64,
    factors: &[Vec<f64>],
    threshold: f64,
    quanta: &mut [usize],
    lines: &mut Vec<(Vec<usize>, f64)>,
) {
    if weight < threshold {
        return;
    }
    if mode == factors.len() {
        lines.push((quanta.to_vec(), weight));
        return;
    }
    for n in 0..factors[mode].len().min(remaining + 1) {
        quanta[mode] = n;
        progression(
            mode + 1,
            remaining - n,
            weight * factors[mode][n],
            factors,
            threshold,
            quanta,
            lines,
        );
    }
    quanta[mode] = 0;
}

/// Franck-Condon progression of one singlet Tamm-Dancoff state in the
/// vertical-gradient model: the excited-state surface is the ground-state
/// harmonic surface displaced along every normal mode by the gradient of
/// the excitation energy at the ground-state geometry. With `f_k` that
/// gradient along the mass-weighted mode `k` of frequency `w_k`, the
/// Huang-Rhys factor is `S_k = f_k^2 / (2 w_k^3)`, the 0-0 energy is the
/// vertical energy minus `sum_k S_k w_k`, and the line with `n_k` quanta has
/// the factor `prod_k exp(-S_k) S_k^n_k / n_k!` (zero temperature).
fn franck_condon(system: &System, controls: &Controls) -> Result<Value> {
    let root = controls.usize("root", 1)?;
    if root == 0 {
        return Err(DftError::InvalidInput(
            "root counts excited states from 1".into(),
        ));
    }
    let request = Request::from_controls(controls, root)?;
    let max_quanta = controls.usize("fc_max_quanta", 3)?;
    let max_total = controls.usize("fc_max_total", 4)?;
    let threshold = controls.f64("fc_threshold", 1e-8)?;
    let broadening = controls.opt_f64("fc_broadening")?;
    let points = controls.usize("points", 2001)?;
    let model = controls.str("fc_model", "vertical-gradient")?;
    let temperature = controls.f64("temperature", 0.0)?;
    let max_hot = controls.usize("fc_max_hot", 2)?;
    let config = super::vibrational::response_config(controls)?;
    let data = super::vibrational::compute_hessian(system, controls)?;
    controls.finish("franck-condon")?;
    if request.nstates < root {
        return Err(DftError::InvalidInput(format!(
            "nstates = {} does not reach root {root}",
            request.nstates
        )));
    }
    if !(threshold > 0.0 && threshold < 1.0)
        || broadening.is_some_and(|width| !(width.is_finite() && width > 0.0))
        || points < 2
    {
        return Err(DftError::InvalidInput(
            "fc_threshold lies between 0 and 1, fc_broadening is positive and points is at \
             least 2"
                .into(),
        ));
    }
    let setup = &data.setup;
    let reference = &data.reference;
    let vertical_hessian = match model.as_str() {
        "vertical-gradient" => false,
        "vertical-hessian" => true,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown fc_model `{other}` (vertical-gradient or vertical-hessian)"
            )))
        }
    };
    if !(temperature.is_finite() && temperature >= 0.0) {
        return Err(DftError::InvalidInput(
            "temperature must not be negative".into(),
        ));
    }
    refuse_fractional_reference(setup)?;
    if !vertical_hessian {
        refuse_unsupported_gradient(setup)?;
    }
    let (states, spin) = solve(setup, reference, &request)?;
    let state = states.get(root - 1).ok_or_else(|| {
        DftError::InvalidInput(format!(
            "root {root} exceeds the {} excitations of this basis",
            states.len()
        ))
    })?;
    if vertical_hessian {
        // Excited-state surface from its gradient and Hessian at the
        // ground-state geometry: its own frequencies and normal modes.
        let molecule = &setup.molecule;
        let ground_modes = crate::properties::vibrational::vibrational_analysis(
            molecule,
            &data.hessian,
            Some(&data.analytic.dipole_derivatives),
        )?;
        let surface = crate::response::nuclear::excited_state_hessian(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            state,
            &config,
        )?;
        let mut hessian = surface.hessian.clone();
        let mut gradient = surface.gradient.clone();
        if setup.dispersion != crate::dispersion::Dispersion::None {
            let name = setup.method.dispersion_functional_name();
            let correction =
                crate::dispersion::dispersion_hessian(&setup.molecule, &setup.dispersion, &name)?;
            for (total, delta) in hessian.as_mut_slice().iter_mut().zip(correction.as_slice()) {
                *total += *delta;
            }
            let forces =
                crate::dispersion::dispersion_gradient(&setup.molecule, &setup.dispersion, &name)?;
            for (atom, delta) in forces.iter().enumerate() {
                let delta = delta.to_array();
                for axis in 0..3 {
                    gradient[3 * atom + axis] += delta[axis];
                }
            }
        }
        // Gradient of the excitation energy: the residual ground-state
        // gradient is removed.
        let ground = super::molecular::total_gradient(setup, reference)?;
        for (atom, g) in ground.iter().enumerate() {
            let g = g.to_array();
            for axis in 0..3 {
                gradient[3 * atom + axis] -= g[axis];
            }
        }
        let excited_modes =
            crate::properties::vibrational::vibrational_analysis(molecule, &hessian, None)?;
        let count = ground_modes.modes.cols;
        if ground_modes.n_imaginary > 0
            || excited_modes.n_imaginary > 0
            || excited_modes.modes.cols != count
        {
            return Err(DftError::InvalidInput(format!(
                "the vertical-Hessian model needs positive Hessians of both states at this \
                 geometry: {} and {} imaginary modes",
                ground_modes.n_imaginary, excited_modes.n_imaginary
            )));
        }
        let natoms = molecule.len();
        let to_hartree = 1.0 / crate::constants::HARTREE_TO_CM1;
        let initial: Vec<f64> = ground_modes
            .frequencies_cm
            .iter()
            .map(|w| w * to_hartree)
            .collect();
        let final_: Vec<f64> = excited_modes
            .frequencies_cm
            .iter()
            .map(|w| w * to_hartree)
            .collect();
        if initial.iter().chain(&final_).any(|w| !(*w > 0.0)) {
            return Err(DftError::InvalidInput(
                "a vibrational mode of the vertical-Hessian model has no positive frequency".into(),
            ));
        }
        let mut rotation = crate::linalg::Matrix::zeros(count, count);
        for i in 0..count {
            for j in 0..count {
                let mut overlap = 0.0;
                for x in 0..3 * natoms {
                    overlap += excited_modes.modes[(x, i)] * ground_modes.modes[(x, j)];
                }
                rotation[(i, j)] = overlap;
            }
        }
        let mut shift = Vec::with_capacity(count);
        let mut relaxation = 0.0;
        for i in 0..count {
            let mut slope = 0.0;
            for a in 0..natoms {
                let weight = 1.0 / crate::constants::atomic_mass_au(molecule.atoms[a].z).sqrt();
                for axis in 0..3 {
                    slope +=
                        weight * excited_modes.modes[(3 * a + axis, i)] * gradient[3 * a + axis];
                }
            }
            shift.push(slope / (final_[i] * final_[i]));
            relaxation += 0.5 * slope * slope / (final_[i] * final_[i]);
        }
        let zero_point: f64 = final_
            .iter()
            .zip(&initial)
            .map(|(a, b)| 0.5 * (a - b))
            .sum();
        let surfaces = crate::properties::franck_condon::Surfaces {
            initial,
            final_,
            rotation,
            shift,
        };
        return vibronic_report(
            setup,
            reference,
            "Franck-Condon lines in the vertical-Hessian model: the excited-state surface from \
             its gradient and Hessian at the ground-state geometry, with its own frequencies \
             and rotated normal modes",
            if request.tamm_dancoff { "tda" } else { "rpa" },
            spin,
            root,
            state,
            &surfaces,
            state.energy - relaxation + zero_point,
            temperature,
            [max_hot, max_quanta, max_total],
            threshold,
            broadening,
            points,
        );
    }
    let mut excited = crate::response::tddft_grad::tda_gradient_with_grid(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        state,
        &setup.options,
    )?;
    if setup.dispersion != crate::dispersion::Dispersion::None {
        let correction = crate::dispersion::dispersion_gradient(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?;
        for (total, delta) in excited.iter_mut().zip(correction) {
            *total += delta;
        }
    }
    let ground = super::molecular::total_gradient(setup, reference)?;
    let molecule = &setup.molecule;
    let modes = crate::properties::vibrational::vibrational_analysis(
        molecule,
        &data.hessian,
        Some(&data.analytic.dipole_derivatives),
    )?;
    if modes.n_imaginary > 0 {
        return Err(DftError::InvalidInput(format!(
            "the Franck-Condon progression needs a ground-state minimum: the Hessian has {} \
             imaginary modes",
            modes.n_imaginary
        )));
    }
    let natoms = molecule.len();
    let nmodes = modes.modes.cols;
    let mut frequencies = Vec::with_capacity(nmodes);
    let mut huang_rhys = Vec::with_capacity(nmodes);
    let mut slopes = Vec::with_capacity(nmodes);
    let mut reorganization = 0.0;
    for k in 0..nmodes {
        let omega = modes.frequencies_cm[k] / crate::constants::HARTREE_TO_CM1;
        if !(omega > 0.0) {
            return Err(DftError::InvalidInput(format!(
                "vibrational mode {k} has no positive frequency"
            )));
        }
        // Gradient of the excitation energy along the mass-weighted mode.
        let mut slope = 0.0;
        for a in 0..natoms {
            let weight = 1.0 / crate::constants::atomic_mass_au(molecule.atoms[a].z).sqrt();
            let difference = excited[a] - ground[a];
            slope += weight
                * (difference.x * modes.modes[(3 * a, k)]
                    + difference.y * modes.modes[(3 * a + 1, k)]
                    + difference.z * modes.modes[(3 * a + 2, k)]);
        }
        let factor = slope * slope / (2.0 * omega.powi(3));
        frequencies.push(omega);
        huang_rhys.push(factor);
        slopes.push(slope);
        reorganization += factor * omega;
    }
    if temperature > 0.0 {
        // Thermally populated initial states of the displaced oscillators.
        let surfaces = crate::properties::franck_condon::Surfaces {
            initial: frequencies.clone(),
            final_: frequencies.clone(),
            rotation: crate::linalg::Matrix::identity(nmodes),
            shift: slopes
                .iter()
                .zip(&frequencies)
                .map(|(f, w)| f / (w * w))
                .collect(),
        };
        return vibronic_report(
            setup,
            reference,
            "Franck-Condon lines of a Tamm-Dancoff state in the vertical-gradient model: \
             displaced harmonic oscillators with the ground-state modes and thermal populations",
            "tda",
            spin,
            root,
            state,
            &surfaces,
            state.energy - reorganization,
            temperature,
            [max_hot, max_quanta, max_total],
            threshold,
            broadening,
            points,
        );
    }
    let factors: Vec<Vec<f64>> = huang_rhys
        .iter()
        .map(|s| {
            let mut row = Vec::with_capacity(max_quanta + 1);
            let mut term = (-s).exp();
            for n in 0..=max_quanta {
                row.push(term);
                term *= s / (n + 1) as f64;
            }
            row
        })
        .collect();
    let mut lines: Vec<(Vec<usize>, f64)> = Vec::new();
    let mut quanta = vec![0_usize; nmodes];
    progression(
        0,
        max_total,
        1.0,
        &factors,
        threshold,
        &mut quanta,
        &mut lines,
    );
    let origin = state.energy - reorganization;
    let energy_of = |assignment: &[usize]| -> f64 {
        origin
            + assignment
                .iter()
                .zip(&frequencies)
                .map(|(n, omega)| *n as f64 * omega)
                .sum::<f64>()
    };
    lines.sort_by(|a, b| energy_of(&a.0).total_cmp(&energy_of(&b.0)));
    let line_energies: Vec<f64> = lines.iter().map(|line| energy_of(&line.0)).collect();
    let line_factors: Vec<f64> = lines.iter().map(|line| line.1).collect();
    let covered: f64 = line_factors.iter().sum();
    let mut report = reference_report(
        setup,
        reference,
        "Franck-Condon progression of a Tamm-Dancoff state in the vertical-gradient model: \
         displaced harmonic oscillators with the ground-state modes at zero temperature",
    );
    report.set("td_method", "tda");
    report.set("td_spin", spin_name(spin));
    report.set("root", root);
    report.set("vertical_excitation_energy", state.energy);
    report.set("oscillator_strength", state.oscillator_strength);
    report.set("reorganization_energy", reorganization);
    report.set("adiabatic_excitation_energy", origin);
    report.set("mode_frequencies", frequencies.clone());
    report.set("mode_frequencies_cm1", modes.frequencies_cm.clone());
    report.set("huang_rhys", huang_rhys);
    report.set("line_energies", line_energies.clone());
    report.set("line_factors", line_factors.clone());
    report.set(
        "line_quanta",
        lines.iter().map(|line| line.0.clone()).collect::<Vec<_>>(),
    );
    report.set("franck_condon_sum", covered);
    report.set("provenance", "analytic");
    let mut unit_pairs: Vec<(&str, &str)> = vec![
        ("energy", "hartree"),
        ("vertical_excitation_energy", "hartree"),
        ("oscillator_strength", "dimensionless"),
        ("reorganization_energy", "hartree"),
        ("adiabatic_excitation_energy", "hartree"),
        ("mode_frequencies", "hartree"),
        ("mode_frequencies_cm1", "cm^-1"),
        ("huang_rhys", "dimensionless"),
        ("line_energies", "hartree"),
        (
            "line_factors",
            "dimensionless, Franck-Condon factor of the line",
        ),
        (
            "franck_condon_sum",
            "dimensionless, sum of the listed factors (one for the complete progression)",
        ),
    ];
    if let Some(width) = broadening {
        let half = 0.5 * width;
        let low = line_energies.iter().copied().fold(f64::INFINITY, f64::min) - 10.0 * width;
        let high = line_energies
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            + 10.0 * width;
        let step = (high - low) / (points - 1) as f64;
        let axis: Vec<f64> = (0..points).map(|i| low + step * i as f64).collect();
        let shape: Vec<f64> = axis
            .iter()
            .map(|e| {
                line_energies
                    .iter()
                    .zip(&line_factors)
                    .map(|(line, factor)| {
                        factor * half / (std::f64::consts::PI * ((e - line).powi(2) + half * half))
                    })
                    .sum()
            })
            .collect();
        report.set("spectrum_energies", axis);
        report.set("spectrum_lineshape", shape);
        report.set("spectrum_broadening", width);
        unit_pairs.push(("spectrum_energies", "hartree"));
        unit_pairs.push((
            "spectrum_lineshape",
            "1/hartree, Franck-Condon factor per energy",
        ));
        unit_pairs.push(("spectrum_broadening", "hartree, full width at half maximum"));
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}

/// Report of the vibronic lines between two harmonic surfaces: thermal
/// populations of the initial vibrational states, Franck-Condon integrals
/// with frequency changes, mode rotation and displacement. `limits` holds
/// the largest total number of initial quanta, the largest number of final
/// quanta per mode and in all.
#[allow(clippy::too_many_arguments)]
fn vibronic_report(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    model: &str,
    td_method: &str,
    spin: TdaSpin,
    root: usize,
    state: &TdaState,
    surfaces: &crate::properties::franck_condon::Surfaces,
    zero_zero: f64,
    temperature: f64,
    limits: [usize; 3],
    threshold: f64,
    broadening: Option<f64>,
    points: usize,
) -> Result<Value> {
    let thermal = temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let (lines, covered) = crate::properties::franck_condon::vibronic_lines(
        surfaces, thermal, limits[0], limits[1], limits[2], threshold,
    )?;
    let line_energies: Vec<f64> = lines.iter().map(|line| zero_zero + line.energy).collect();
    let line_factors: Vec<f64> = lines.iter().map(|line| line.intensity).collect();
    let count = surfaces.initial.len();
    let rotation: Vec<Vec<f64>> = (0..count)
        .map(|i| (0..count).map(|j| surfaces.rotation[(i, j)]).collect())
        .collect();
    let mut report = reference_report(setup, reference, model);
    report.set("td_method", td_method);
    report.set("td_spin", spin_name(spin));
    report.set("root", root);
    report.set("vertical_excitation_energy", state.energy);
    report.set("oscillator_strength", state.oscillator_strength);
    report.set("adiabatic_excitation_energy", zero_zero);
    report.set("temperature", temperature);
    report.set("mode_frequencies", surfaces.initial.clone());
    report.set("final_mode_frequencies", surfaces.final_.clone());
    report.set("duschinsky_matrix", rotation);
    report.set("mode_shifts", surfaces.shift.clone());
    report.set("line_energies", line_energies.clone());
    report.set("line_factors", line_factors.clone());
    report.set(
        "line_quanta",
        lines
            .iter()
            .map(|line| line.final_.clone())
            .collect::<Vec<_>>(),
    );
    report.set(
        "line_initial_quanta",
        lines
            .iter()
            .map(|line| line.initial.clone())
            .collect::<Vec<_>>(),
    );
    report.set("franck_condon_sum", covered);
    report.set("provenance", "analytic");
    let mut unit_pairs: Vec<(&str, &str)> = vec![
        ("energy", "hartree"),
        ("vertical_excitation_energy", "hartree"),
        ("oscillator_strength", "dimensionless"),
        (
            "adiabatic_excitation_energy",
            "hartree, between the vibrational ground states",
        ),
        ("temperature", "K"),
        ("mode_frequencies", "hartree"),
        ("final_mode_frequencies", "hartree"),
        (
            "duschinsky_matrix",
            "dimensionless, rows final modes, columns initial modes",
        ),
        (
            "mode_shifts",
            "bohr*sqrt(m_e), displacement in the final normal coordinates",
        ),
        ("line_energies", "hartree"),
        (
            "line_factors",
            "dimensionless, population of the initial state times the Franck-Condon factor",
        ),
        (
            "franck_condon_sum",
            "dimensionless, sum of the listed factors",
        ),
    ];
    if let Some(width) = broadening {
        let half = 0.5 * width;
        let low = line_energies.iter().copied().fold(f64::INFINITY, f64::min) - 10.0 * width;
        let high = line_energies
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            + 10.0 * width;
        let step = (high - low) / (points - 1) as f64;
        let axis: Vec<f64> = (0..points).map(|i| low + step * i as f64).collect();
        let shape: Vec<f64> = axis
            .iter()
            .map(|e| {
                line_energies
                    .iter()
                    .zip(&line_factors)
                    .map(|(line, factor)| {
                        factor * half / (std::f64::consts::PI * ((e - line).powi(2) + half * half))
                    })
                    .sum()
            })
            .collect();
        report.set("spectrum_energies", axis);
        report.set("spectrum_lineshape", shape);
        report.set("spectrum_broadening", width);
        unit_pairs.push(("spectrum_energies", "hartree"));
        unit_pairs.push((
            "spectrum_lineshape",
            "1/hartree, Franck-Condon factor per energy",
        ));
        unit_pairs.push(("spectrum_broadening", "hartree, full width at half maximum"));
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}

/// One point of the walk on an excited-state surface.
struct ExcitedPoint {
    setup: MolecularSetup,
    reference: MolecularReference,
    state: TdaState,
    /// Total energy of the excited state (Hartree).
    energy: f64,
    /// Its gradient, `3 atom + axis` (Hartree/Bohr).
    gradient: Vec<f64>,
    /// Fresh analytic path residual when the path route is selected.
    response_residual: Option<f64>,
}

fn largest(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn inner(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Geometry relaxation on one excited-state energy surface by L-BFGS on
/// the selected analytic gradient, with backtracking on the
/// excited-state energy. The first point takes state `root`; afterwards the
/// state is the one whose squared amplitudes overlap most with those of the
/// previous point.
fn excited_optimization(system: &System, controls: &Controls) -> Result<Value> {
    let method = MolecularMethod::from_controls(controls)?;
    let root = controls.usize("root", 1)?;
    if root == 0 {
        return Err(DftError::InvalidInput(
            "root counts excited states from 1".into(),
        ));
    }
    let request = Request::from_controls(controls, root)?;
    let route = GradientRoute::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let max_steps = controls.usize("max_steps", 100)?;
    let gtol = controls.f64("gtol", 3e-4)?;
    let step_max = controls.f64("step_max", 0.3)?;
    let history = controls.usize("history", 8)?;
    controls.finish("excited-optimization")?;
    if request.nstates < root {
        return Err(DftError::InvalidInput(format!(
            "nstates = {} does not reach root {root}",
            request.nstates
        )));
    }
    if !(gtol > 0.0 && step_max > 0.0) || history == 0 {
        return Err(DftError::InvalidInput(
            "gtol and step_max are positive and history is at least 1".into(),
        ));
    }
    let evaluate = |positions: &[[f64; 3]], follow: Option<&TdaState>| -> Result<ExcitedPoint> {
        let setup = prepare_molecular(&system.numbers, positions, &method)?;
        refuse_fractional_reference(&setup)?;
        if matches!(route, GradientRoute::ZVector) {
            refuse_unsupported_gradient(&setup)?;
        }
        let reference = run_molecular_reference(&setup)?;
        let (mut states, _) = solve(&setup, &reference, &request)?;
        let index = match follow {
            None => root - 1,
            Some(previous) => {
                // Squared amplitudes are unchanged by the signs of the orbitals.
                let overlap = |candidate: &TdaState| -> f64 {
                    (0..2)
                        .map(|s| {
                            previous.x[s]
                                .as_slice()
                                .iter()
                                .zip(candidate.x[s].as_slice())
                                .map(|(a, b)| a * a * b * b)
                                .sum::<f64>()
                        })
                        .sum()
                };
                let mut best = 0;
                for candidate in 1..states.len() {
                    if overlap(&states[candidate]) > overlap(&states[best]) {
                        best = candidate;
                    }
                }
                best
            }
        };
        if index >= states.len() {
            return Err(DftError::InvalidInput(format!(
                "root {root} exceeds the {} excitations of this basis",
                states.len()
            )));
        }
        let state = states.swap_remove(index);
        let (mut gradient, response_residual) =
            excited_gradient(&setup, &reference, &state, route, &config)?;
        if setup.dispersion != crate::dispersion::Dispersion::None {
            let correction = crate::dispersion::dispersion_gradient(
                &setup.molecule,
                &setup.dispersion,
                &setup.method.dispersion_functional_name(),
            )?;
            for (total, delta) in gradient.iter_mut().zip(correction) {
                *total += delta;
            }
        }
        let energy = reference.total_energy() + state.energy;
        let gradient = gradient.iter().flat_map(|g| g.to_array()).collect();
        Ok(ExcitedPoint {
            setup,
            reference,
            state,
            energy,
            gradient,
            response_residual,
        })
    };

    let mut positions = system.positions.clone();
    let mut point = evaluate(&positions, None)?;
    let mut steps = vec![0_usize];
    let mut energies = vec![point.energy];
    let mut gradients = vec![largest(&point.gradient)];
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut iterations = 0;
    let mut converged = largest(&point.gradient) < gtol;
    while !converged && iterations < max_steps {
        iterations += 1;
        // L-BFGS two-loop recursion.
        let mut q = point.gradient.clone();
        let m = s_hist.len();
        let mut alpha = vec![0.0; m];
        for k in (0..m).rev() {
            let rho = 1.0 / inner(&y_hist[k], &s_hist[k]);
            alpha[k] = rho * inner(&s_hist[k], &q);
            for (value, y) in q.iter_mut().zip(&y_hist[k]) {
                *value -= alpha[k] * y;
            }
        }
        if m > 0 {
            let gamma = inner(&s_hist[m - 1], &y_hist[m - 1])
                / inner(&y_hist[m - 1], &y_hist[m - 1]).max(1e-12);
            for value in q.iter_mut() {
                *value *= gamma;
            }
        }
        for k in 0..m {
            let rho = 1.0 / inner(&y_hist[k], &s_hist[k]);
            let beta = rho * inner(&y_hist[k], &q);
            for (value, s) in q.iter_mut().zip(&s_hist[k]) {
                *value += (alpha[k] - beta) * s;
            }
        }
        let mut step: Vec<f64> = q.iter().map(|v| -v).collect();
        let length = largest(&step);
        if length > step_max {
            for value in step.iter_mut() {
                *value *= step_max / length;
            }
        }
        // Backtracking on the excited-state energy.
        let mut accepted = false;
        let mut scale = 1.0;
        for _ in 0..5 {
            let trial: Vec<[f64; 3]> = positions
                .iter()
                .enumerate()
                .map(|(a, r)| std::array::from_fn(|axis| r[axis] + scale * step[3 * a + axis]))
                .collect();
            match evaluate(&trial, Some(&point.state)) {
                Ok(next) if next.energy < point.energy + 1e-12 => {
                    let s: Vec<f64> = step.iter().map(|v| v * scale).collect();
                    let y: Vec<f64> = next
                        .gradient
                        .iter()
                        .zip(&point.gradient)
                        .map(|(a, b)| a - b)
                        .collect();
                    if inner(&s, &y) > 1e-10 {
                        s_hist.push(s);
                        y_hist.push(y);
                        if s_hist.len() > history {
                            s_hist.remove(0);
                            y_hist.remove(0);
                        }
                    }
                    positions = trial;
                    point = next;
                    accepted = true;
                    break;
                }
                _ => scale *= 0.4,
            }
        }
        if accepted {
            steps.push(iterations);
            energies.push(point.energy);
            gradients.push(largest(&point.gradient));
        } else if m == 0 {
            // A steepest-descent step without curvature history failed: stop.
            break;
        } else {
            s_hist.clear();
            y_hist.clear();
        }
        converged = largest(&point.gradient) < gtol;
    }
    let mut report = reference_report(
        &point.setup,
        &point.reference,
        "geometry relaxation on the selected excited-state surface (L-BFGS on the analytic gradient)",
    );
    report.set(
        "td_method",
        if request.tamm_dancoff { "tda" } else { "rpa" },
    );
    report.set("td_spin", spin_name(point.state.spin));
    report.set("td_gradient_route", route.name());
    if let Some(residual) = point.response_residual {
        report.set("response_residual", residual);
    }
    report.set("root", root);
    report.set("numbers", system.numbers.clone());
    report.set("positions", positions);
    report.set("excited_state_energy", point.energy);
    report.set("excitation_energy", point.state.energy);
    report.set("oscillator_strength", point.state.oscillator_strength);
    report.set(
        "gradient",
        point
            .gradient
            .chunks(3)
            .map(|g| [g[0], g[1], g[2]])
            .collect::<Vec<_>>(),
    );
    report.set("gradient_max", largest(&point.gradient));
    report.set("optimization_converged", converged);
    report.set("iterations", iterations);
    report.set("trajectory_steps", steps);
    report.set("trajectory_energies", energies);
    report.set("trajectory_gradient_max", gradients);
    report.set("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("positions", "bohr"),
            ("excited_state_energy", "hartree"),
            ("excitation_energy", "hartree"),
            ("oscillator_strength", "dimensionless"),
            ("gradient", "hartree/bohr"),
            ("gradient_max", "hartree/bohr"),
            ("trajectory_energies", "hartree"),
            ("trajectory_gradient_max", "hartree/bohr"),
            ("response_residual", "dimensionless"),
        ]),
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Matrix;

    fn line(energy_ev: f64, oscillator: f64, rotatory: f64) -> TdaState {
        TdaState {
            energy: energy_ev / HARTREE_TO_EV,
            x: [Matrix::zeros(0, 0), Matrix::zeros(0, 0)],
            spin: TdaSpin::Singlet,
            tamm_dancoff: true,
            transition_dipole: [0.0; 3],
            oscillator_strength: oscillator,
            transition_angular_momentum: [0.0; 3],
            transition_momentum: [0.0; 3],
            rotatory_strength: rotatory,
            rotatory_strength_velocity: -2.0 * rotatory,
        }
    }

    #[test]
    fn spectral_conversion_factors_match_their_tabulated_values() {
        // f = 4.319e-9 int epsilon d nu; R = 22.965e-40 erg cm^3 times
        // int (Delta epsilon / nu) d nu; one atomic unit of rotatory
        // strength is 471.4436e-40 erg cm^3.
        let k_abs = oscillator_per_integrated_absorptivity();
        let k_ecd = rotatory_per_integrated_ecd();
        let cgs = rotatory_au_in_1e40_cgs();
        println!("K_abs = {k_abs:.6e}, K_ecd = {k_ecd:.6}, R a.u. = {cgs:.6}");
        assert!((k_abs / 4.319e-9 - 1.0).abs() < 1e-4, "{k_abs:e}");
        assert!((k_ecd / 22.965 - 1.0).abs() < 1e-4, "{k_ecd}");
        assert!((cgs / 471.4436 - 1.0).abs() < 1e-6, "{cgs}");
    }

    #[test]
    fn broadened_bands_integrate_to_their_line_strengths() {
        let states = [line(6.0, 0.1, 0.02), line(8.0, 0.3, -0.05)];
        let sigma = 0.2;
        let spectrum = broadened_spectrum(&states, sigma, 5.0, 9.0, 4001);
        let cgs = rotatory_au_in_1e40_cgs();
        let rotatory = (0.02 - 0.05) * cgs;
        println!(
            "absorption integral {:.12}, ecd integrals {:.9} {:.9}, expected {rotatory:.9}",
            spectrum.absorption_integral, spectrum.ecd_integral[0], spectrum.ecd_integral[1]
        );
        // The window ends five standard deviations beyond the outer bands,
        // which leaves 2.9e-7 of each outer band outside.
        assert!((spectrum.absorption_integral - 0.4).abs() < 1e-6);
        assert!((spectrum.ecd_integral[0] - rotatory).abs() < 1e-6 * cgs);
        assert!((spectrum.ecd_integral[1] + 2.0 * rotatory).abs() < 1e-6 * cgs);
        // The band maximum of an isolated line: f / (K sigma sqrt(2 pi)) in
        // wavenumber units.
        let ev_to_cm1 = HARTREE_TO_CM1 / HARTREE_TO_EV;
        let peak = 0.3
            / (oscillator_per_integrated_absorptivity()
                * sigma
                * ev_to_cm1
                * (2.0 * std::f64::consts::PI).sqrt());
        let index = 3000;
        assert!((spectrum.energies_ev[index] - 8.0).abs() < 1e-12);
        assert!((spectrum.absorption[index] / peak - 1.0).abs() < 1e-9);
        // Delta epsilon is the reduced band times the wavenumber.
        assert!(
            (spectrum.ecd_velocity[index] + 2.0 * spectrum.ecd_length[index]).abs()
                < 1e-12 * spectrum.ecd_length[index].abs()
        );
    }
}
