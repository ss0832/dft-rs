// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Conceptual DFT descriptors (charge and spin channels).
//!
//! Three routes are available and every report names the one it took:
//!
//! * `grand_canonical` (analytic): a Fermi-Dirac reference at a positive
//!   electronic temperature. The per-spin chemical potentials are the exact
//!   first derivatives `dA/dN_sigma` of the Mermin free energy, and the
//!   hardness matrix `d mu_sigma / d N_sigma'` and the Fukui matrices
//!   `d rho / d N_sigma'` come from the self-consistent ensemble response
//!   (`response::ensemble`), including the occupation response.
//! * `integer`: total-energy differences between the N, N-1 and N+1 (or
//!   spin-flipped) integer-electron ground states. These are the one-sided
//!   derivatives of the piecewise-linear exact E(N), not an approximation to
//!   a smooth derivative, and they are reported as such.
//! * `koopmans`: frontier orbital energies and orbital densities of the N
//!   electron reference.
//!
//! Conventions: `N = N_alpha + N_beta`, `M = N_alpha - N_beta` (not M/2 and
//! not S_z); `mu = dE/dN`, `eta = d2E/dN2` (no factor 1/2), softness
//! `S = 1/eta`, electrophilicity `omega = mu^2 / (2 eta)`. Condensed Fukui
//! values are differences of electron populations (Mulliken or Loewdin).

use super::controls::Controls;
use super::method::{
    prepare_molecular, run_molecular_reference, MolecularMethod, MolecularReference, MolecularSetup,
};
use super::molecular::{reference_report, units};
use super::value::Value;
use super::vibrational::response_config;
use super::System;
use crate::linalg::Matrix;
use crate::{DftError, Result};

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let spin = id == "spin-conceptual-dft";
    let mut method = MolecularMethod::from_controls(controls)?;
    let route = match controls.opt_str("route")? {
        Some(r) => r,
        None if method.electronic_temperature > 0.0 => "grand_canonical".into(),
        None => "integer".into(),
    };
    let population = controls.str("population", "mulliken")?;
    if !matches!(population.as_str(), "mulliken" | "lowdin") {
        return Err(DftError::InvalidInput(
            "population must be mulliken or lowdin".into(),
        ));
    }
    let config = if route == "grand_canonical" {
        Some(response_config(controls)?)
    } else {
        None
    };
    let cation_multiplicity = controls.opt_usize("cation_multiplicity")?;
    let anion_multiplicity = controls.opt_usize("anion_multiplicity")?;
    controls.finish(id)?;
    if spin && route == "grand_canonical" {
        // The spin channel needs the spin-polarized kernel even for a
        // closed-shell reference; the unrestricted solution of a closed shell
        // equals the restricted one.
        method.unrestricted = true;
    }
    let setup = prepare_molecular(&system.numbers, &system.positions, &method)?;
    let reference = run_molecular_reference(&setup)?;
    match route.as_str() {
        "grand_canonical" => grand_canonical(
            &setup,
            &reference,
            &population,
            config.as_ref().unwrap(),
            spin,
        ),
        "integer" => {
            if spin {
                integer_spin(system, &method, &setup, &reference, &population)
            } else {
                integer_charge(
                    system,
                    &method,
                    &setup,
                    &reference,
                    &population,
                    cation_multiplicity,
                    anion_multiplicity,
                )
            }
        }
        "koopmans" if !spin => koopmans(&setup, &reference, &population),
        "koopmans" => Err(DftError::InvalidInput(
            "the spin channel has grand_canonical and integer routes".into(),
        )),
        other => Err(DftError::InvalidInput(format!(
            "unknown conceptual DFT route `{other}` (grand_canonical, integer, koopmans)"
        ))),
    }
}

/// Electron populations per atom of an AO density.
fn populations(
    setup: &MolecularSetup,
    density: &Matrix,
    overlap: &Matrix,
    scheme: &str,
) -> Result<Vec<f64>> {
    let owner = setup.basis.sph_ao_atoms();
    let mut out = vec![0.0; setup.molecule.len()];
    match scheme {
        "lowdin" => {
            let (values, vectors) = crate::linalg::symmetric_eigen(overlap)?;
            let n = overlap.rows;
            let mut half = Matrix::zeros(n, n);
            for k in 0..n {
                let s = values[k].sqrt();
                for i in 0..n {
                    for j in 0..n {
                        half[(i, j)] += vectors[(i, k)] * s * vectors[(j, k)];
                    }
                }
            }
            let sps = half.matmul(&density.matmul(&half));
            for (mu, &atom) in owner.iter().enumerate() {
                out[atom] += sps[(mu, mu)];
            }
        }
        _ => {
            let ps = density.matmul(overlap);
            for (mu, &atom) in owner.iter().enumerate() {
                out[atom] += ps[(mu, mu)];
            }
        }
    }
    Ok(out)
}

fn total(reference: &MolecularReference) -> Matrix {
    reference.scf.p_total()
}

fn spin_density(reference: &MolecularReference) -> Matrix {
    let mut m = reference.scf.density[0].clone();
    for (v, b) in m
        .as_mut_slice()
        .iter_mut()
        .zip(reference.scf.density[1].as_slice())
    {
        *v -= *b;
    }
    m
}

fn global(mu: f64, eta: Option<f64>) -> Value {
    let softness = eta.filter(|e| *e != 0.0).map(|e| 1.0 / e);
    let omega = eta.filter(|e| *e != 0.0).map(|e| mu * mu / (2.0 * e));
    Value::object()
        .with("chemical_potential", mu)
        .with("electronegativity", -mu)
        .with("hardness", eta)
        .with("softness", softness)
        .with("electrophilicity", omega)
}

fn conventions() -> Value {
    Value::object()
        .with("numbers", "N = N_alpha + N_beta, M = N_alpha - N_beta")
        .with(
            "hardness",
            "eta = d2E/dN2 (no factor 1/2); softness S = 1/eta",
        )
        .with("electrophilicity", "omega = mu^2 / (2 eta)")
        .with(
            "fukui",
            "condensed values are electron-population changes per atom",
        )
}

fn with_common(
    mut report: Value,
    route: &str,
    population: &str,
    unit_pairs: &[(&str, &str)],
) -> Value {
    report.set("route", route);
    report.set("population_scheme", population);
    report.set("conventions", conventions());
    report.set("units", units(unit_pairs));
    report
}

fn rerun(
    system: &System,
    method: &MolecularMethod,
    charge_delta: i32,
    multiplicity: usize,
) -> Result<(MolecularSetup, MolecularReference)> {
    let mut m = method.clone();
    m.charge += charge_delta;
    m.multiplicity = multiplicity;
    m.stabilize = false;
    let setup = prepare_molecular(&system.numbers, &system.positions, &m)?;
    let reference = run_molecular_reference(&setup)?;
    if !reference.scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.scf.iterations,
            error: f64::NAN,
        });
    }
    Ok((setup, reference))
}

fn lowest_multiplicity(electrons: usize) -> usize {
    if electrons % 2 == 0 {
        1
    } else {
        2
    }
}

fn integer_charge(
    system: &System,
    method: &MolecularMethod,
    setup: &MolecularSetup,
    reference: &MolecularReference,
    population: &str,
    cation_multiplicity: Option<usize>,
    anion_multiplicity: Option<usize>,
) -> Result<Value> {
    let electrons = reference.scf.n_alpha + reference.scf.n_beta;
    if electrons == 0 {
        return Err(DftError::InvalidInput(
            "the N-1 state needs at least one electron".into(),
        ));
    }
    let m_cation = cation_multiplicity.unwrap_or_else(|| lowest_multiplicity(electrons - 1));
    let m_anion = anion_multiplicity.unwrap_or_else(|| lowest_multiplicity(electrons + 1));
    let (cation_setup, cation) = rerun(system, method, 1, m_cation)?;
    let (anion_setup, anion) = rerun(system, method, -1, m_anion)?;
    let e0 = reference.total_energy();
    let ip = cation.total_energy() - e0;
    let ea = e0 - anion.total_energy();
    let mu = -(ip + ea) / 2.0;
    let eta = ip - ea;
    let p0 = populations(setup, &total(reference), &reference.scf.overlap, population)?;
    let pc = populations(
        &cation_setup,
        &total(&cation),
        &cation.scf.overlap,
        population,
    )?;
    let pa = populations(&anion_setup, &total(&anion), &anion.scf.overlap, population)?;
    let f_plus: Vec<f64> = pa.iter().zip(&p0).map(|(a, b)| a - b).collect();
    let f_minus: Vec<f64> = p0.iter().zip(&pc).map(|(a, b)| a - b).collect();
    let f_zero: Vec<f64> = f_plus
        .iter()
        .zip(&f_minus)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let dual: Vec<f64> = f_plus.iter().zip(&f_minus).map(|(a, b)| a - b).collect();
    let softness = if eta != 0.0 { Some(1.0 / eta) } else { None };
    let omega = if eta != 0.0 {
        mu * mu / (2.0 * eta)
    } else {
        f64::NAN
    };
    let mut warnings = Vec::new();
    if ea < 0.0 {
        warnings.push(Value::Str(
            "WARNING [conceptual-dft] the N+1 state lies above the N state (negative electron affinity); \
             in a finite basis the anion is not bound and f+ describes a basis-confined electron"
                .into(),
        ));
    }
    let mut report = reference_report(
        setup,
        reference,
        "conceptual DFT from integer-electron total-energy differences",
    )
    .with("ionization_energy", ip)
    .with("electron_affinity", ea)
    .with("global", global(mu, Some(eta)))
    .with(
        "states",
        Value::object()
            .with("cation_multiplicity", m_cation)
            .with("anion_multiplicity", m_anion)
            .with("cation_energy", cation.total_energy())
            .with("anion_energy", anion.total_energy()),
    )
    .with("fukui_plus", f_plus.clone())
    .with("fukui_minus", f_minus.clone())
    .with("fukui_zero", f_zero)
    .with("dual_descriptor", dual)
    .with(
        "local_softness_plus",
        softness.map(|s| f_plus.iter().map(|f| s * f).collect::<Vec<_>>()),
    )
    .with(
        "local_softness_minus",
        softness.map(|s| f_minus.iter().map(|f| s * f).collect::<Vec<_>>()),
    )
    .with(
        "local_electrophilicity",
        if omega.is_finite() {
            Some(f_plus.iter().map(|f| omega * f).collect::<Vec<_>>())
        } else {
            None
        },
    );
    if let Some(Value::Array(w)) = report.get("warnings").cloned() {
        let mut all = w;
        all.extend(warnings);
        report.set("warnings", Value::Array(all));
    }
    Ok(with_common(
        report,
        "integer",
        population,
        &[
            ("energy", "hartree"),
            ("ionization_energy", "hartree"),
            ("electron_affinity", "hartree"),
            ("global.chemical_potential", "hartree"),
            ("global.hardness", "hartree"),
            ("global.softness", "1/hartree"),
            ("global.electrophilicity", "hartree"),
            ("fukui_plus", "electrons per atom"),
            ("local_softness_plus", "electrons/hartree"),
        ],
    ))
}

fn orbital_populations(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    spin: usize,
    index: usize,
    population: &str,
) -> Result<Vec<f64>> {
    let c = &reference.scf.mo_coeff[spin];
    let n = c.rows;
    let mut density = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            density[(mu, nu)] = c[(mu, index)] * c[(nu, index)];
        }
    }
    populations(setup, &density, &reference.scf.overlap, population)
}

fn koopmans(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    population: &str,
) -> Result<Value> {
    let scf = &reference.scf;
    let mut homo: Option<(f64, usize, usize)> = None;
    let mut lumo: Option<(f64, usize, usize)> = None;
    for spin in 0..if scf.unrestricted { 2 } else { 1 } {
        let n_occ = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        let e = &scf.mo_energy[spin];
        if let Some(i) = n_occ.checked_sub(1) {
            if homo.is_none_or(|h| e[i] > h.0) {
                homo = Some((e[i], spin, i));
            }
        }
        if n_occ < e.len() && lumo.is_none_or(|l| e[n_occ] < l.0) {
            lumo = Some((e[n_occ], spin, n_occ));
        }
    }
    let (Some(homo), Some(lumo)) = (homo, lumo) else {
        return Err(DftError::InvalidInput(
            "Koopmans descriptors need an occupied and a virtual orbital".into(),
        ));
    };
    let ip = -homo.0;
    let ea = -lumo.0;
    let mu = -(ip + ea) / 2.0;
    let eta = ip - ea;
    let f_plus = orbital_populations(setup, reference, lumo.1, lumo.2, population)?;
    let f_minus = orbital_populations(setup, reference, homo.1, homo.2, population)?;
    let dual: Vec<f64> = f_plus.iter().zip(&f_minus).map(|(a, b)| a - b).collect();
    let report = reference_report(
        setup,
        reference,
        "conceptual DFT from frontier orbital energies and densities (Koopmans)",
    )
    .with("ionization_energy", ip)
    .with("electron_affinity", ea)
    .with("global", global(mu, Some(eta)))
    .with(
        "homo",
        Value::object()
            .with("energy", homo.0)
            .with("spin", homo.1)
            .with("index", homo.2),
    )
    .with(
        "lumo",
        Value::object()
            .with("energy", lumo.0)
            .with("spin", lumo.1)
            .with("index", lumo.2),
    )
    .with("fukui_plus", f_plus.clone())
    .with("fukui_minus", f_minus.clone())
    .with(
        "fukui_zero",
        f_plus
            .iter()
            .zip(&f_minus)
            .map(|(a, b)| 0.5 * (a + b))
            .collect::<Vec<_>>(),
    )
    .with("dual_descriptor", dual);
    Ok(with_common(
        report,
        "koopmans",
        population,
        &[
            ("energy", "hartree"),
            ("ionization_energy", "hartree"),
            ("electron_affinity", "hartree"),
            ("global.chemical_potential", "hartree"),
            ("global.hardness", "hartree"),
            ("fukui_plus", "electrons per atom"),
        ],
    ))
}

fn integer_spin(
    system: &System,
    method: &MolecularMethod,
    setup: &MolecularSetup,
    reference: &MolecularReference,
    population: &str,
) -> Result<Value> {
    // M = N_alpha - N_beta changes in steps of two at fixed N.
    let m0 = method.multiplicity as i64 - 1;
    let electrons = (reference.scf.n_alpha + reference.scf.n_beta) as i64;
    let e0 = reference.total_energy();
    let state = |m: i64| -> Result<Option<(MolecularSetup, MolecularReference)>> {
        if m.abs() > electrons {
            return Ok(None);
        }
        // E(M) = E(-M): a negative M is the mirror of the positive state.
        let mut mm = method.clone();
        mm.multiplicity = (m.unsigned_abs() + 1) as usize;
        mm.unrestricted = true;
        mm.stabilize = false;
        let s = prepare_molecular(&system.numbers, &system.positions, &mm)?;
        let r = run_molecular_reference(&s)?;
        Ok(Some((s, r)))
    };
    let up = state(m0 + 2)?;
    let down = state(m0 - 2)?;
    let e_up = up.as_ref().map(|(_, r)| r.total_energy());
    let e_down = down.as_ref().map(|(_, r)| r.total_energy());
    let mu_s_plus = e_up.map(|e| (e - e0) / 2.0);
    let mu_s_minus = e_down.map(|e| (e0 - e) / 2.0);
    let eta_ss = match (e_up, e_down) {
        (Some(u), Some(d)) => Some((u - 2.0 * e0 + d) / 4.0),
        _ => None,
    };
    let s0 = populations(
        setup,
        &spin_density(reference),
        &reference.scf.overlap,
        population,
    )?;
    let spin_fukui = |state: &Option<(MolecularSetup, MolecularReference)>,
                      sign: f64,
                      mirrored: bool|
     -> Result<Option<Vec<f64>>> {
        match state {
            None => Ok(None),
            Some((s, r)) => {
                let mut p = populations(s, &spin_density(r), &r.scf.overlap, population)?;
                if mirrored {
                    for v in &mut p {
                        *v = -*v;
                    }
                }
                Ok(Some(
                    p.iter()
                        .zip(&s0)
                        .map(|(a, b)| sign * (a - b) / 2.0)
                        .collect(),
                ))
            }
        }
    };
    let f_up = spin_fukui(&up, 1.0, false)?;
    let f_down = spin_fukui(&down, -1.0, m0 - 2 < 0)?;
    let report = reference_report(
        setup,
        reference,
        "spin-resolved conceptual DFT from fixed-N spin-flip total-energy differences",
    )
    .with("spin_number", m0)
    .with("spin_chemical_potential_plus", mu_s_plus)
    .with("spin_chemical_potential_minus", mu_s_minus)
    .with("spin_hardness", eta_ss)
    .with("charge_spin_coupling", Value::Null)
    .with("charge_hardness", Value::Null)
    .with("spin_fukui_plus", f_up)
    .with("spin_fukui_minus", f_down)
    .with(
        "absent",
        "charge_hardness and charge_spin_coupling need changes of N at fixed M, which integer \
             states with N and M of equal parity cannot isolate; use the grand_canonical route",
    );
    Ok(with_common(
        report,
        "integer",
        population,
        &[
            ("energy", "hartree"),
            ("spin_chemical_potential_plus", "hartree per unit M"),
            ("spin_hardness", "hartree per unit M^2"),
            ("spin_fukui_plus", "spin population change per unit M"),
        ],
    ))
}

fn grand_canonical(
    setup: &MolecularSetup,
    reference: &MolecularReference,
    population: &str,
    config: &crate::response::ResponseSolverConfig,
    spin: bool,
) -> Result<Value> {
    let scf = &reference.scf;
    if scf.electronic_temperature <= 0.0 {
        return Err(DftError::InvalidInput(
            "the grand_canonical route needs electronic_temperature > 0 (Kelvin)".into(),
        ));
    }
    let solver = crate::response::ensemble::EnsembleSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
    )?;
    let mu = [
        scf.chemical_potentials[0].unwrap_or(f64::NAN),
        scf.chemical_potentials[1].unwrap_or(f64::NAN),
    ];
    // Columns of the per-spin hardness matrix and Fukui matrices.
    let directions: Vec<[f64; 2]> = if scf.unrestricted {
        vec![[1.0, 0.0], [0.0, 1.0]]
    } else {
        vec![[0.5, 0.5]]
    };
    let mut responses = Vec::new();
    let mut residual = 0.0_f64;
    let mut number_defect = 0.0_f64;
    let mut iterations = Vec::new();
    for d in &directions {
        let r = solver.solve(None, *d, config)?;
        residual = residual.max(r.relative_residual);
        number_defect = number_defect.max(r.number_defects[0].max(r.number_defects[1]));
        iterations.push(r.iterations);
        responses.push(r);
    }
    // Derivatives with respect to N (adding half an electron per spin) and
    // M (alpha up, beta down by one half).
    let (dmu_dn, dp_dn, dmu_dm, dp_dm, hardness_matrix) = if scf.unrestricted {
        let ra = &responses[0];
        let rb = &responses[1];
        let combine = |wa: f64, wb: f64| {
            let mut dp = [ra.delta_density[0].clone(), ra.delta_density[1].clone()];
            for s in 0..2 {
                for (v, (a, b)) in dp[s].as_mut_slice().iter_mut().zip(
                    ra.delta_density[s]
                        .as_slice()
                        .iter()
                        .zip(rb.delta_density[s].as_slice()),
                ) {
                    *v = wa * a + wb * b;
                }
            }
            let dmu = [
                wa * ra.delta_chemical_potentials[0] + wb * rb.delta_chemical_potentials[0],
                wa * ra.delta_chemical_potentials[1] + wb * rb.delta_chemical_potentials[1],
            ];
            (dmu, dp)
        };
        let (dmu_n, dp_n) = combine(0.5, 0.5);
        let (dmu_m, dp_m) = combine(0.5, -0.5);
        let eta = [
            [
                ra.delta_chemical_potentials[0],
                rb.delta_chemical_potentials[0],
            ],
            [
                ra.delta_chemical_potentials[1],
                rb.delta_chemical_potentials[1],
            ],
        ];
        (dmu_n, dp_n, Some(dmu_m), Some(dp_m), Some(eta))
    } else {
        let r = &responses[0];
        (
            r.delta_chemical_potentials,
            r.delta_density.clone(),
            None,
            None,
            None,
        )
    };
    let mu_n = 0.5 * (mu[0] + mu[1]);
    let eta_nn = 0.5 * (dmu_dn[0] + dmu_dn[1]);
    let total_change = {
        let mut m = dp_dn[0].clone();
        for (v, b) in m.as_mut_slice().iter_mut().zip(dp_dn[1].as_slice()) {
            *v += *b;
        }
        m
    };
    let fukui = populations(setup, &total_change, &scf.overlap, population)?;
    let softness = 1.0 / eta_nn;
    let mut report = reference_report(
        setup,
        reference,
        "analytic grand-canonical conceptual DFT of a Fermi-Dirac (Mermin) reference with self-consistent occupation response",
    )
    .with("global", global(mu_n, Some(eta_nn)))
    .with("chemical_potentials_per_spin", mu.to_vec())
    .with("fukui", fukui.clone())
    .with("local_softness", fukui.iter().map(|f| softness * f).collect::<Vec<_>>())
    .with("hardness_matrix_per_spin", hardness_matrix.map(|m| m.to_vec()))
    .with(
        "hardness_matrix_symmetry_defect",
        hardness_matrix.map(|m| (m[0][1] - m[1][0]).abs()),
    )
    .with("response_relative_residual", residual)
    .with("response_number_defect", number_defect)
    .with("response_iterations", iterations)
    .with("electronic_temperature_kelvin", setup.method.electronic_temperature)
    .with("free_energy", reference.total_energy());
    if spin {
        let dmu_m = dmu_dm.expect("spin route uses an unrestricted reference");
        let dp_m = dp_dm.expect("spin route uses an unrestricted reference");
        let mu_s = 0.5 * (mu[0] - mu[1]);
        let eta_ss = 0.5 * (dmu_m[0] - dmu_m[1]);
        let eta_ns = 0.5 * (dmu_m[0] + dmu_m[1]);
        let eta_sn = 0.5 * (dmu_dn[0] - dmu_dn[1]);
        let spin_change = |dp: &[Matrix; 2]| {
            let mut m = dp[0].clone();
            for (v, b) in m.as_mut_slice().iter_mut().zip(dp[1].as_slice()) {
                *v -= *b;
            }
            m
        };
        let total_change_m = {
            let mut m = dp_m[0].clone();
            for (v, b) in m.as_mut_slice().iter_mut().zip(dp_m[1].as_slice()) {
                *v += *b;
            }
            m
        };
        report.set("spin_chemical_potential", mu_s);
        report.set("spin_hardness", eta_ss);
        report.set("charge_spin_coupling", eta_ns);
        report.set("charge_spin_coupling_transpose", eta_sn);
        report.set(
            "spin_fukui_ss",
            populations(setup, &spin_change(&dp_m), &scf.overlap, population)?,
        );
        report.set(
            "spin_fukui_ns",
            populations(setup, &total_change_m, &scf.overlap, population)?,
        );
        report.set(
            "spin_fukui_sn",
            populations(setup, &spin_change(&dp_dn), &scf.overlap, population)?,
        );
    }
    Ok(with_common(
        report,
        "grand_canonical",
        population,
        &[
            ("energy", "hartree"),
            ("global.chemical_potential", "hartree"),
            ("global.hardness", "hartree"),
            ("global.softness", "1/hartree"),
            ("global.electrophilicity", "hartree"),
            ("chemical_potentials_per_spin", "hartree"),
            (
                "hardness_matrix_per_spin",
                "hartree (d mu_sigma / d N_sigma')",
            ),
            ("fukui", "electrons per atom per electron"),
            ("spin_hardness", "hartree per unit M^2"),
        ],
    ))
}
