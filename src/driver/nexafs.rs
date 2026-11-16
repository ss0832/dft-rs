// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Polarization-resolved molecular core absorption in a finite Gaussian basis.

use super::method::{run_molecular_reference, MolecularSetup};
use super::molecular::{reference_report, setup, units};
use super::{Controls, System, Value};
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::tddft::{TdaSolver, TdaSpin};
use crate::scf::{core_hole::core_hole_reference, ScfResult};
use crate::{DftError, Result};

#[derive(Clone, Copy)]
struct Line {
    energy: f64,
    dipole: [f64; 3],
    spin: Option<usize>,
    core: Option<usize>,
    virtual_orbital: Option<usize>,
    residual: f64,
}

fn site_core(setup: &MolecularSetup, state: &ScfResult, atom: usize, spin: usize) -> Result<usize> {
    let first = setup
        .basis
        .shells
        .iter()
        .filter(|s| s.atom == atom && s.l == 0)
        .max_by(|a, b| {
            a.exps
                .iter()
                .copied()
                .fold(0.0, f64::max)
                .total_cmp(&b.exps.iter().copied().fold(0.0, f64::max))
        })
        .ok_or_else(|| DftError::InvalidInput("absorber has no s-type Gaussian shell".into()))?;
    let c = &state.mo_coeff[spin];
    let sc = state.overlap.matmul_seq(c);
    let occupied = [state.n_alpha, state.n_beta][spin];
    let weight = |i| c[(first.sph_off, i)] * sc[(first.sph_off, i)];
    (0..occupied).max_by(|&a,&b| weight(a).total_cmp(&weight(b)))
    .filter(|&i| weight(i)>0.5).ok_or_else(|| DftError::Unsupported {
        feature: format!("atom-resolved K-edge absorption at atom {atom} in spin {spin}"),
        missing: "a localized occupied 1s orbital; provide a core space explicitly for delocalized edges".into(),
    })
}

fn spin_columns(matrix: &Matrix, columns: &[usize]) -> Matrix {
    let mut result = Matrix::zeros(matrix.rows, columns.len());
    for i in 0..matrix.rows {
        for (j, &index) in columns.iter().enumerate() {
            result[(i, j)] = matrix[(i, index)];
        }
    }
    result
}

fn determinant(matrix: &Matrix) -> f64 {
    if matrix.rows == 0 {
        return 1.0;
    }
    faer::MatRef::from_row_major_slice(matrix.as_slice(), matrix.rows, matrix.cols).determinant()
}

// Multilinearity replaces one ket overlap column by the dipole column.
// This gives the determinant one-body matrix element even at zero overlap.
fn determinant_dipole(overlap: &Matrix, dipole: &Matrix) -> f64 {
    let mut work = overlap.clone();
    let mut value = 0.0;
    for j in 0..work.cols {
        for i in 0..work.rows {
            work[(i, j)] = dipole[(i, j)];
        }
        value += determinant(&work);
        for i in 0..work.rows {
            work[(i, j)] = overlap[(i, j)];
        }
    }
    value
}

fn final_state_lines(
    setup: &MolecularSetup,
    reference: &ScfResult,
    core: &[Vec<usize>; 2],
    minimum_overlap: f64,
) -> Result<(Vec<Line>, Vec<Value>)> {
    let mut lines = Vec::new();
    let mut holes = Vec::new();
    let dipoles = dipole_matrices(&setup.basis, Vec3::zero());
    let occupied = [reference.n_alpha, reference.n_beta];
    let initial: [Matrix; 2] = std::array::from_fn(|s| {
        spin_columns(
            &reference.mo_coeff[s],
            &(0..occupied[s]).collect::<Vec<_>>(),
        )
    });
    let initial_bra = initial.map(|c| c.transpose());
    let density = reference.p_total();
    let ground_dipoles = dipoles.each_ref().map(|m| m.frobenius_dot(&density));
    for spin in 0..2 {
        let spectator = 1 - spin;
        for &orbital in &core[spin] {
            let state = core_hole_reference(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
                reference,
                spin,
                orbital,
            )?;
            let diagnostics = state
                .occupation_tracking
                .as_ref()
                .expect("core hole tracks occupations");
            let density_residual = diagnostics.density_residual;
            let orbital_residual = super::molecular::reference_orbital_residual(&state);
            let overlap_minimum = diagnostics
                .overlap_weights
                .iter()
                .flatten()
                .copied()
                .reduce(f64::min)
                .unwrap_or(1.0);
            let spin_occupied = [state.n_alpha, state.n_beta];
            let final_overlap: [Matrix; 2] = std::array::from_fn(|s| {
                initial_bra[s]
                    .matmul_seq(&state.overlap)
                    .matmul_seq(&state.mo_coeff[s])
            });
            let vac = spin_occupied[spin]..state.mo_coeff[spin].cols;
            let refill = vac
                .clone()
                .max_by(|&a, &b| {
                    final_overlap[spin][(orbital, a)]
                        .abs()
                        .total_cmp(&final_overlap[spin][(orbital, b)].abs())
                })
                .expect("core ion has at least one vacancy");
            let hole_weight = final_overlap[spin][(orbital, refill)].powi(2);
            if hole_weight < minimum_overlap || overlap_minimum < minimum_overlap {
                return Err(DftError::Unsupported {
                    feature: format!("relaxed absorption from spin {spin} core orbital {orbital}"),
                    missing: format!("a tracked core ion (hole weight {hole_weight:.6e}, smallest occupied weight {overlap_minimum:.6e}) above minimum_core_overlap {minimum_overlap:.6e}"),
                });
            }
            let ip = state.energy - reference.energy;
            if !ip.is_finite() || ip <= 0.0 {
                return Err(DftError::InvalidInput(format!(
                    "core ionization energy must be positive, got {ip:.8e}"
                )));
            }
            holes.push(
                Value::object()
                    .with("spin", spin)
                    .with("core_orbital", orbital)
                    .with("refill_orbital", refill)
                    .with("ionization_energy", ip)
                    .with("hole_weight", hole_weight)
                    .with("occupied_overlap", overlap_minimum)
                    .with("iterations", state.iterations)
                    .with("residual", density_residual)
                    .with("orbital_residual", orbital_residual),
            );
            let spectator_cols: Vec<_> = (0..spin_occupied[spectator]).collect();
            let spectator_overlap = spin_columns(&final_overlap[spectator], &spectator_cols);
            let spectator_det = determinant(&spectator_overlap);
            let transformed: [[Matrix; 3]; 2] = std::array::from_fn(|s| {
                std::array::from_fn(|a| {
                    initial_bra[s]
                        .matmul_seq(&dipoles[a])
                        .matmul_seq(&state.mo_coeff[s])
                })
            });
            let spectator_dipole: [f64; 3] = std::array::from_fn(|a| {
                determinant_dipole(
                    &spectator_overlap,
                    &spin_columns(&transformed[spectator][a], &spectator_cols),
                )
            });
            for a in vac.filter(|&a| a != refill) {
                let energy = ip + state.mo_energy[spin][a];
                if !energy.is_finite() || energy <= 0.0 {
                    return Err(DftError::InvalidInput(format!("nonpositive relaxed absorption energy for spin {spin}, core {orbital}, final orbital {a}: {energy:.8e}")));
                }
                let active_cols: Vec<_> = (0..spin_occupied[spin]).chain([a]).collect();
                let active_overlap = spin_columns(&final_overlap[spin], &active_cols);
                let active_det = determinant(&active_overlap);
                // Orthogonalize the approximate final determinant to the
                // ground state. This removes the origin-dependent overlap term.
                let cross_overlap = active_det * spectator_det;
                let norm = (1.0 - cross_overlap.powi(2)).sqrt();
                if !norm.is_finite() || norm < 1e-6 {
                    return Err(DftError::LinearAlgebra(
                        "relaxed core-excited determinant overlaps the reference too strongly"
                            .into(),
                    ));
                }
                let dipole = std::array::from_fn(|direction| {
                    (spectator_det
                        * determinant_dipole(
                            &active_overlap,
                            &spin_columns(&transformed[spin][direction], &active_cols),
                        )
                        + active_det * spectator_dipole[direction]
                        - cross_overlap * ground_dipoles[direction])
                        / norm
                });
                lines.push(Line {
                    energy,
                    dipole,
                    spin: Some(spin),
                    core: Some(orbital),
                    virtual_orbital: Some(a),
                    residual: density_residual,
                });
            }
        }
    }
    Ok((lines, holes))
}

fn intensity(line: &Line) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| 2.0 * line.energy * line.dipole[i] * line.dipole[j])
    })
}

fn isotropic(matrix: &[[f64; 3]; 3]) -> f64 {
    (0..3).map(|i| matrix[i][i]).sum::<f64>() / 3.0
}

fn polarized(matrix: &[[f64; 3]; 3], e: [f64; 3]) -> f64 {
    (0..3)
        .map(|i| (0..3).map(|j| e[i] * matrix[i][j] * e[j]).sum::<f64>())
        .sum()
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let feature = "nexafs";
    controls.require_consumed("reference", feature)?;
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "a correlated core excitation and transition density provider".into(),
        });
    }
    let setup = setup(system, controls)?;
    if setup.options.asymptotic_correction.is_some() {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "a consistent variational core-hole or linear-response kernel for the asymptotically corrected potential".into(),
        });
    }
    let model = controls.str("nexafs_model", "cvs-tda")?;
    let edge = controls.str("absorption_edge", "k")?.to_ascii_lowercase();
    let atoms = controls.opt_usize_list("absorber_atoms")?;
    let alpha = controls.opt_usize_list("core_orbitals_alpha")?;
    let beta = controls.opt_usize_list("core_orbitals_beta")?;
    let width = controls.f64("nexafs_broadening", 0.01)?;
    let broadening = controls.str("line_shape", "lorentzian")?;
    let energy_window = controls.opt_f64_list("nexafs_energy_window")?;
    let points = controls.usize("points", 1501)?;
    let max_points =
        setup.options.incore_mb.saturating_mul(1024 * 1024) / (13 * std::mem::size_of::<f64>());
    let roots = controls.opt_usize("nstates")?;
    let minimum_overlap = controls.f64("minimum_core_overlap", 0.5)?;
    let tolerance = controls.f64("core_response_tolerance", 1e-8)?;
    let polarization = controls
        .opt_vec3("polarization")?
        .map(|p| {
            let length = p.norm();
            if !(length.is_finite() && length > 0.0) {
                return Err(DftError::InvalidInput(
                    "polarization needs a nonzero Cartesian direction".into(),
                ));
            }
            Ok((p * length.recip()).to_array())
        })
        .transpose()?;
    controls.finish(feature)?;
    if !matches!(model.as_str(), "cvs-tda" | "relaxed-core-ion")
        || !matches!(broadening.as_str(), "gaussian" | "lorentzian")
        || !matches!(edge.as_str(), "k" | "l1" | "l23" | "explicit")
        || !(width.is_finite()
            && width > 0.0
            && (0.0..=1.0).contains(&minimum_overlap)
            && tolerance.is_finite()
            && tolerance > 0.0)
        || points < 2
        || points > max_points
        || roots == Some(0)
        || energy_window
            .as_ref()
            .is_some_and(|window| window.len() != 2 || window[0] >= window[1])
        || atoms.is_some() == alpha.is_some()
        || (atoms.is_some() && beta.is_some())
    {
        return Err(DftError::InvalidInput("invalid near-edge model, broadening, edge or core-space controls; supply K-edge absorber_atoms or explicit spin-resolved core orbitals, and at least two spectrum points within incore_mb".into()));
    }
    if !setup.basis.ecps.is_empty()
        || setup.options.electronic_temperature != 0.0
        || setup.options.electron_offset != [0.0; 2]
        || setup.options.occupation.is_some()
        || setup.options.fixed_occupations.is_some()
    {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "an all-electron collinear integer aufbau reference".into(),
        });
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: super::molecular::reference_orbital_residual(scf),
        });
    }
    let nmo = [scf.mo_coeff[0].cols, scf.mo_coeff[1].cols];
    let core = if let Some(atoms) = atoms.as_ref() {
        if edge != "k" {
            return Err(DftError::Unsupported {
                feature: format!("automatic core selection for {edge} edge"),
                missing: "explicit spin orbital lists for edges other than K".into(),
            });
        }
        if atoms.is_empty()
            || atoms
                .iter()
                .enumerate()
                .any(|(i, a)| *a >= setup.molecule.len() || atoms[..i].contains(a))
        {
            return Err(DftError::InvalidInput(
                "absorber_atoms requires distinct zero-based atom indices".into(),
            ));
        }
        [
            atoms
                .iter()
                .map(|&a| site_core(&setup, scf, a, 0))
                .collect::<Result<_>>()?,
            if scf.n_beta == 0 {
                Vec::new()
            } else {
                atoms
                    .iter()
                    .map(|&a| site_core(&setup, scf, a, 1))
                    .collect::<Result<_>>()?
            },
        ]
    } else {
        let a = alpha.expect("explicit core list validated");
        let b = beta.unwrap_or_else(|| a.iter().copied().filter(|&i| i < scf.n_beta).collect());
        [a, b]
    };
    for (s, selected) in core.iter().enumerate() {
        if selected
            .iter()
            .enumerate()
            .any(|(k, i)| *i >= [scf.n_alpha, scf.n_beta][s] || selected[..k].contains(i))
        {
            return Err(DftError::InvalidInput(
                "core orbital lists need distinct occupied indices in the selected spin".into(),
            ));
        }
    }
    if core.iter().all(Vec::is_empty) {
        return Err(DftError::InvalidInput(
            "the requested core space is empty".into(),
        ));
    }
    if model == "cvs-tda" {
        let alpha = core[0].len().saturating_mul(nmo[0] - scf.n_alpha);
        let beta = core[1].len().saturating_mul(nmo[1] - scf.n_beta);
        let dimension = alpha.saturating_add(if scf.unrestricted { beta } else { 0 });
        let required = dimension
            .saturating_mul(dimension)
            .saturating_mul(5 * std::mem::size_of::<f64>());
        let mib = crate::resources::MIB;
        if required > setup.options.incore_mb.saturating_mul(mib) {
            return Err(DftError::ResourceLimit {
                operation: "dense core-valence-separated TDA matrix",
                required_mb: required.div_ceil(mib),
                limit_mb: setup.options.incore_mb,
            });
        }
    }
    let (mut lines, holes) = if model == "relaxed-core-ion" {
        final_state_lines(&setup, scf, &core, minimum_overlap)?
    } else {
        if !scf.unrestricted && core[0] != core[1] {
            return Err(DftError::Unsupported {
                feature: "spin-asymmetric CVS on an RKS reference".into(),
                missing: "a common spin-adapted core space or an unrestricted reference".into(),
            });
        }
        let spin = if scf.unrestricted {
            TdaSpin::SpinResolved
        } else {
            TdaSpin::Singlet
        };
        let solver = TdaSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            spin,
        )?;
        let (states, residuals) =
            solver.solve_core_with_residuals([&core[0], &core[1]], roots.unwrap_or(usize::MAX))?;
        let lines = states
            .into_iter()
            .zip(residuals)
            .map(|(s, residual)| Line {
                energy: s.energy,
                dipole: s.transition_dipole,
                residual,
                spin: None,
                core: None,
                virtual_orbital: None,
            })
            .collect();
        (lines, Vec::new())
    };
    if lines.is_empty() {
        return Err(DftError::InvalidInput(
            "the requested core space has no empty final states".into(),
        ));
    }
    for line in &lines {
        if !line.energy.is_finite() || line.energy <= 0.0 {
            return Err(DftError::InvalidInput(
                "near-edge spectrum contains a nonpositive transition energy".into(),
            ));
        }
        if model == "cvs-tda" && (!line.residual.is_finite() || line.residual > tolerance) {
            return Err(DftError::ResponseNotConverged {
                iterations: 0,
                residual: line.residual,
            });
        }
        if line.dipole.iter().any(|x| !x.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "near-edge transition dipole is nonfinite".into(),
            ));
        }
    }
    lines.sort_by(|a, b| a.energy.total_cmp(&b.energy));
    lines.truncate(roots.unwrap_or(usize::MAX));
    let matrix: Vec<_> = lines.iter().map(intensity).collect();
    let line_values: Vec<_> = lines
        .iter()
        .zip(&matrix)
        .map(|(line, tensor)| {
            Value::object()
                .with("energy", line.energy)
                .with("transition_dipole", line.dipole)
                .with("oscillator_tensor", tensor)
                .with("oscillator_strength", isotropic(tensor))
                .with(
                    "polarized_strength",
                    polarization.map(|p| polarized(tensor, p)),
                )
                .with("spin", line.spin)
                .with("core_orbital", line.core)
                .with("virtual_orbital", line.virtual_orbital)
                .with("residual", line.residual)
        })
        .collect();
    let (low, high) = energy_window.as_deref().map_or_else(
        || {
            (
                lines[0].energy - 12.0 * width,
                lines[lines.len() - 1].energy + 12.0 * width,
            )
        },
        |window| (window[0], window[1]),
    );
    if !low.is_finite() || !high.is_finite() || !(high - low).is_finite() {
        return Err(DftError::InvalidInput(
            "near-edge spectrum energy range overflows".into(),
        ));
    }
    let axis: Vec<_> = (0..points)
        .map(|i| low + (high - low) * i as f64 / (points - 1) as f64)
        .collect();
    let tensor: Vec<[[f64; 3]; 3]> = axis
        .iter()
        .map(|&e| {
            let mut result = [[0.0; 3]; 3];
            for (line, f) in lines.iter().zip(&matrix) {
                let x = (e - line.energy) / width;
                let density = if broadening == "lorentzian" {
                    0.5 / (std::f64::consts::PI * width * (x * x + 0.25))
                } else {
                    let k = 4.0 * std::f64::consts::LN_2;
                    (k / std::f64::consts::PI).sqrt() / width * (-k * x * x).exp()
                };
                for i in 0..3 {
                    for j in 0..3 {
                        result[i][j] += density * f[i][j];
                    }
                }
            }
            result
        })
        .collect();
    let description = if model == "cvs-tda" {
        "all-electron length-gauge core-valence-separated Tamm-Dancoff response on the neutral or charged input determinant"
    } else {
        "separate full core ion relaxation per spin and orbital; transition energy equals relaxed ionization energy plus ion virtual energy; spectator-relaxed one-body determinant moments projected orthogonal to the initial determinant"
    };
    let mut report = reference_report(&setup, &reference, description);
    report.set(
        "scf_orbital_residual",
        super::molecular::reference_orbital_residual(scf),
    );
    report.set("absorption_edge", edge);
    report.set("nexafs_model", model.clone());
    report.set("absorber_atoms", atoms);
    report.set("core_orbitals_alpha", &core[0]);
    report.set("core_orbitals_beta", &core[1]);
    report.set("lines", line_values);
    report.set("core_holes", holes);
    report.set("polarization", polarization);
    report.set("line_shape", broadening);
    report.set("spectrum_broadening", width);
    report.set("spectrum_energies", axis);
    report.set(
        "spectrum_oscillator_density",
        tensor.iter().map(isotropic).collect::<Vec<_>>(),
    );
    report.set(
        "spectrum_polarized_density",
        polarization.map(|p| tensor.iter().map(|t| polarized(t, p)).collect::<Vec<_>>()),
    );
    report.set("spectrum_oscillator_tensor_density", tensor);
    let existing = report
        .get("units")
        .and_then(|v| match v {
            Value::Object(entries) => Some(entries.clone()),
            _ => None,
        })
        .unwrap_or_default();
    let mut labels = Value::Object(existing);
    let new = units(&[
        ("lines.energy", "hartree"),
        ("scf_orbital_residual", "hartree"),
        ("lines.transition_dipole", "e*bohr"),
        ("lines.oscillator_tensor", "dimensionless"),
        ("lines.oscillator_strength", "dimensionless"),
        ("lines.polarized_strength", "dimensionless"),
        (
            "lines.residual",
            if model == "cvs-tda" {
                "hartree"
            } else {
                "dimensionless, determinant selection density defect"
            },
        ),
        ("core_holes.ionization_energy", "hartree"),
        ("core_holes.hole_weight", "dimensionless"),
        ("core_holes.occupied_overlap", "dimensionless"),
        ("core_holes.residual", "dimensionless"),
        ("core_holes.orbital_residual", "hartree"),
        ("polarization", "dimensionless"),
        ("spectrum_broadening", "hartree, full width at half maximum"),
        ("spectrum_energies", "hartree"),
        ("spectrum_oscillator_density", "1/hartree"),
        ("spectrum_oscillator_tensor_density", "1/hartree"),
        ("spectrum_polarized_density", "1/hartree"),
    ]);
    if let Value::Object(entries) = new {
        for (name, value) in entries {
            labels.set(&name, value);
        }
    }
    report.set("units", labels);
    Ok(report)
}
