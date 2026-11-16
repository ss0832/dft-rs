// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular density constraints and their stationary geometry derivatives.

use super::method::{run_molecular_reference, MolecularSetup};
use super::molecular::{setup, units};
use super::vibrational::response_config;
use super::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::spinor_scf::{pauli_components, run_spinor_scf, spin_diagonal};
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use crate::scf::constrained::{
    self, CollinearConstraint, ConstrainedResult, ConstraintOptions, Trial,
};
use crate::{DftError, Result};

fn read_operator(entry: &Controls, name: &str, n: usize) -> Result<Option<Matrix>> {
    let Some(rows) = entry.opt_rows(name)? else {
        return Ok(None);
    };
    if rows.len() != n || rows.iter().any(|row| row.len() != n) {
        return Err(DftError::InvalidInput(format!(
            "{name} needs {n} by {n} AO entries"
        )));
    }
    Ok(Some(Matrix::from_row_major(
        n,
        n,
        rows.into_iter().flatten().collect(),
    )))
}

struct FragmentProjector {
    ao_weights: Vec<f64>,
    pauli: usize,
}

impl FragmentProjector {
    fn new(setup: &MolecularSetup, atoms: &[usize], kind: &str) -> Result<Self> {
        if atoms.is_empty()
            || atoms.iter().any(|&i| i >= setup.molecule.len())
            || atoms
                .iter()
                .enumerate()
                .any(|(i, a)| atoms[..i].contains(a))
        {
            return Err(DftError::InvalidInput(
                "constraint atoms must be distinct zero-based indices".into(),
            ));
        }
        let pauli = match kind {
            "population" => 0,
            "spin-x" => 1,
            "spin-y" => 2,
            "spin" | "spin-z" => 3,
            _ => {
                return Err(DftError::InvalidInput(
                    "constraint kind must be population, spin-x, spin-y, or spin-z".into(),
                ))
            }
        };
        let ao_weights = setup
            .basis
            .sph_ao_atoms()
            .iter()
            .map(|owner| f64::from(u8::from(atoms.contains(owner))))
            .collect();
        Ok(Self { ao_weights, pauli })
    }

    fn weight(&self, i: usize, j: usize) -> f64 {
        0.5 * (self.ao_weights[i] + self.ao_weights[j])
    }

    fn operator(&self, mut overlap: Matrix) -> Matrix {
        for i in 0..overlap.rows {
            for j in 0..overlap.cols {
                overlap[(i, j)] *= self.weight(i, j);
            }
        }
        overlap
    }
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let feature = "constrained-dft";
    if controls.bool("optimize_geometry", false)? {
        return optimize_geometry(system, controls);
    }
    let with_gradient = controls.bool("gradient", false)?;
    let electric_response = controls.bool("constraint_electric_response", false)?;
    let reference = controls.str("reference", "auto")?.to_ascii_lowercase();
    let gks = matches!(reference.as_str(), "gks" | "ghf");
    let roks = matches!(reference.as_str(), "roks" | "rohf");
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "a variational correlated density and its multiplier response".into(),
        });
    }
    let setup = setup(system, controls)?;
    if setup.method.stabilize
        || setup.options.occupation.is_some()
        || setup.options.fixed_occupations.is_some()
        || setup.options.electronic_temperature != 0.0
        || setup.options.electron_offset != [0.0; 2]
        || setup.options.asymptotic_correction.is_some()
    {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "variational integer aufbau references without PT2 or unconstrained stability following".into(),
        });
    }
    if with_gradient
        && (setup.smd.is_some()
            || gks && (setup.options.pcm.is_some() || setup.options.skala.is_some()))
    {
        return Err(DftError::Unsupported {
            feature: "constrained DFT geometry derivatives with the requested solvent or learned spinor model".into(),
            missing: "the corresponding moving solvent or spin-covariant functional gradient".into(),
        });
    }
    let list = controls
        .take("density_constraints")
        .and_then(Value::as_array)
        .filter(|entries| !entries.is_empty())
        .ok_or_else(|| {
            DftError::InvalidInput(
                "density_constraints needs a nonempty array of constraint objects".into(),
            )
        })?;
    let n = setup.basis.nao_sph;
    let mut collinear = Vec::new();
    let mut spinor = Vec::new();
    let mut targets = Vec::new();
    let mut fragments = Vec::new();
    let mut overlap = None;
    for value in list {
        let entry = Controls::from_value(value)?;
        let atoms = entry.opt_usize_list("atoms")?;
        let electrons = entry.opt_f64("electrons")?;
        let target = match (entry.opt_f64("target")?, electrons) {
            (Some(target), None) => target,
            (None, Some(electrons)) if atoms.is_some() => electrons,
            _ => return Err(DftError::InvalidInput(
                "a density constraint needs exactly one target, or electrons for a fragment population".into(),
            )),
        };
        targets.push(target);
        if let Some(atoms) = atoms {
            let axis = entry.str("kind", "population")?;
            let fragment = FragmentProjector::new(&setup, &atoms, &axis)?;
            if electrons.is_some() && fragment.pauli != 0 {
                return Err(DftError::InvalidInput(
                    "electrons is a fragment population target and requires kind=population".into(),
                ));
            }
            let matrix = fragment.operator(
                overlap
                    .get_or_insert_with(|| crate::integrals::overlap_matrix(&setup.basis))
                    .clone(),
            );
            if gks {
                let mut operator = BlochMatrix::zeros(2 * n, 2 * n);
                match axis.as_str() {
                    "population" => {
                        operator = spin_diagonal(&BlochMatrix {
                            re: matrix,
                            im: Matrix::zeros(n, n),
                        })
                    }
                    "spin-x" | "spin-y" | "spin" | "spin-z" => {
                        for i in 0..n {
                            for j in 0..n {
                                let q = matrix[(i, j)];
                                match axis.as_str() {
                                    "spin-x" => {
                                        operator.re[(i, n + j)] = q;
                                        operator.re[(n + i, j)] = q;
                                    }
                                    "spin-y" => {
                                        operator.im[(i, n + j)] = -q;
                                        operator.im[(n + i, j)] = q;
                                    }
                                    _ => {
                                        operator.re[(i, j)] = q;
                                        operator.re[(n + i, n + j)] = -q;
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(DftError::InvalidInput(
                            "constraint kind must be population, spin-x, spin-y, or spin-z".into(),
                        ))
                    }
                }
                spinor.push(operator);
            } else {
                let mut beta = matrix.clone();
                match axis.as_str() {
                    "population" => {}
                    "spin" | "spin-z" => beta.as_mut_slice().iter_mut().for_each(|v| *v = -*v),
                    "spin-x" | "spin-y" => {
                        return Err(DftError::Unsupported {
                            feature: "transverse density constraint on a collinear reference"
                                .into(),
                            missing: "the full spinor reference, selected with reference=gks"
                                .into(),
                        })
                    }
                    _ => {
                        return Err(DftError::InvalidInput(
                            "constraint kind must be population or spin-z".into(),
                        ))
                    }
                }
                collinear.push(CollinearConstraint {
                    operators: [matrix, beta],
                    target,
                });
            }
            fragments.push(Some(fragment));
        } else if gks {
            let re = read_operator(&entry, "operator_real", 2 * n)?.ok_or_else(|| {
                DftError::InvalidInput(
                    "spinor constraints need operator_real or fragment atoms".into(),
                )
            })?;
            let im = read_operator(&entry, "operator_imaginary", 2 * n)?
                .unwrap_or_else(|| Matrix::zeros(2 * n, 2 * n));
            let operator = BlochMatrix { re, im };
            if operator.hermiticity_error() > 1e-12 {
                return Err(DftError::InvalidInput(
                    "density constraint spinor operator must be Hermitian".into(),
                ));
            }
            spinor.push(operator);
            fragments.push(None);
        } else {
            let alpha = read_operator(&entry, "operator_alpha", n)?.ok_or_else(|| {
                DftError::InvalidInput(
                    "collinear constraints need operator_alpha or fragment atoms".into(),
                )
            })?;
            let beta = read_operator(&entry, "operator_beta", n)?.unwrap_or_else(|| alpha.clone());
            collinear.push(CollinearConstraint {
                operators: [alpha, beta],
                target,
            });
            fragments.push(None);
        }
        entry.finish("density constraint")?;
    }
    let initial = controls
        .opt_f64_list("initial_constraint_multipliers")?
        .unwrap_or_else(|| vec![0.0; list.len()]);
    let options = ConstraintOptions {
        max_iterations: controls.usize("constraint_max_iterations", 40)?,
        tolerance: controls.f64("constraint_tolerance", 1e-9)?,
        max_multiplier_step: controls.f64("constraint_max_step", 0.5)?,
        response: response_config(controls)?,
    };
    let axis = if gks {
        controls
            .opt_vec3("spin_axis")?
            .map_or([0.0, 0.0, 1.0], |v| v.to_array())
    } else {
        [0.0, 0.0, 1.0]
    };
    let xc_tolerance = if gks {
        controls.f64("spinor_xc_tolerance", 1e-10)?
    } else {
        1e-10
    };
    controls.finish(feature)?;
    let dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    let cds = setup
        .smd
        .map(|s| crate::pcm::smd_cds_energy(&setup.molecule, s, 2048))
        .transpose()?
        .unwrap_or(0.0);
    let correction = dispersion + cds;
    let mut gradient = None;
    let mut electric = None;
    let report = if gks {
        let seed = run_molecular_reference(&setup)?;
        let spinor_config = setup.spinor_scf_config();
        let result = constrained::maximize_dual(
            &targets,
            &initial,
            &options,
            |lambda, _| {
                let mut operator = BlochMatrix::zeros(2 * n, 2 * n);
                for (q, &value) in spinor.iter().zip(lambda) {
                    for (part, source) in [(&mut operator.re, &q.re), (&mut operator.im, &q.im)] {
                        for (x, &v) in part.as_mut_slice().iter_mut().zip(source.as_slice()) {
                            *x += value * v;
                        }
                    }
                }
                let backend = MolecularSpinorBackend::new(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    setup.grid.as_ref(),
                    &seed.scf,
                    &setup.options,
                    operator.clone(),
                    axis,
                    MolecularSpinorFunctionalModel::SpinCovariant,
                    xc_tolerance,
                )?;
                let state = run_spinor_scf(&backend, &spinor_config)?;
                if !state.converged {
                    return Err(DftError::ScfNotConverged {
                        iterations: state.iterations,
                        error: state.orbital_residual,
                    });
                }
                let expectations = spinor
                    .iter()
                    .map(|q| q.trace_product(&state.density[0]))
                    .collect();
                Ok(Trial {
                    biased_energy: state.fields.energy,
                    expectations,
                    state: (state, operator),
                })
            },
            |(state, operator)| {
                let response = crate::response::nuclear::spinor_one_body_response(
                    &setup.molecule,
                    &setup.basis,
                    setup.grid.as_ref(),
                    &setup.functional,
                    state,
                    &setup.options,
                    operator,
                    &spinor,
                    &[Complex::ZERO],
                    &options.response,
                )?;
                let response = response
                    .values
                    .into_iter()
                    .next()
                    .expect("one static response");
                let size = spinor.len();
                Ok((
                    Matrix::from_row_major(
                        size,
                        size,
                        response
                            .susceptibility
                            .iter()
                            .flatten()
                            .map(|v| v.re)
                            .collect(),
                    ),
                    response.residuals,
                ))
            },
        )?;
        let state = &result.state.0;
        if electric_response {
            let sources = electric_operators(&setup).map(|re| {
                spin_diagonal(&BlochMatrix {
                    im: Matrix::zeros(n, n),
                    re,
                })
            });
            let response = crate::response::nuclear::spinor_constrained_one_body_response(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                &setup.functional,
                state,
                &setup.options,
                &result.state.1,
                &sources,
                &spinor,
                &[Complex::ZERO],
                &options.response,
            )?;
            let value = &response.values[0];
            electric = Some((
                std::array::from_fn::<_, 3, _>(|i| {
                    std::array::from_fn::<_, 3, _>(|j| -value.susceptibility[i][j].re)
                }),
                value
                    .constraint_residuals
                    .iter()
                    .flatten()
                    .map(|v| v.abs())
                    .fold(0.0, f64::max),
                value.residuals.clone(),
            ));
        }
        if with_gradient {
            let (_, _, mut nuclear) = crate::response::nuclear::spinor_off_shell_geometry(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                &setup.functional,
                &setup.options,
                &state.density[0],
                options.response.tolerance,
            )?;
            let mut coefficients = state.bands[0].coefficients.clone();
            for (i, (f, e)) in state.occupations[0]
                .iter()
                .zip(&state.bands[0].energies)
                .enumerate()
            {
                for j in 0..2 * n {
                    coefficients.re[(j, i)] *= f * e;
                    coefficients.im[(j, i)] *= f * e;
                }
            }
            // The constrained eigenvalues include the multiplier operator.
            // Its metric chain belongs to the biased stationary projector.
            let weighted = crate::periodic::phonon::product(
                &coefficients,
                &crate::periodic::phonon::adjoint(&state.bands[0].coefficients),
            );
            let trace = pauli_components(&weighted)?[0].re.clone();
            for (value, metric) in
                nuclear
                    .iter_mut()
                    .zip(crate::integrals::deriv::overlap_gradient(
                        &setup.basis,
                        &trace,
                        setup.molecule.len(),
                        -1.0,
                    ))
            {
                *value += metric;
            }
            let density = pauli_components(&state.density[0])?;
            add_fragment_gradient(
                &setup,
                &fragments,
                &result.multipliers,
                |pauli| &density[pauli].re,
                &mut nuclear,
            );
            gradient = Some(nuclear);
        }
        result_value(&result, correction, targets)
            .with("reference_type", "gks")
            .with("scf_iterations", state.iterations)
            .with("scf_orbital_residual", state.orbital_residual)
            .with("metric_defect", state.metric_defect)
            .with(
                "spinor_energy_components",
                state.fields.energy_components.to_vec(),
            )
            .with("density_real", &state.density[0].re)
            .with("density_imaginary", &state.density[0].im)
    } else {
        let result = constrained::run_collinear(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            roks,
            &collinear,
            &initial,
            &options,
        )?;
        let scf = &result.state.0;
        if electric_response {
            let mut sources = collinear.clone();
            sources.extend(electric_operators(&setup).map(|r| CollinearConstraint {
                operators: [r.clone(), r],
                target: 0.0,
            }));
            let (jacobian, residuals) = constrained::operator_susceptibility(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                result.state.1.as_ref(),
                &sources,
                &options.response,
            )?;
            let count = collinear.len();
            let mut alpha = [[0.0; 3]; 3];
            let mut defect = 0.0_f64;
            for j in 0..3 {
                let rhs: Vec<_> = (0..count).map(|i| -jacobian[(i, count + j)]).collect();
                let lambda = crate::linalg::solve_linear(&result.susceptibility, &rhs)?;
                let reduced = |i| {
                    jacobian[(i, count + j)]
                        + (0..count)
                            .map(|q| jacobian[(i, q)] * lambda[q])
                            .sum::<f64>()
                };
                for i in 0..count {
                    defect = defect.max(reduced(i).abs());
                }
                for i in 0..3 {
                    alpha[i][j] = -reduced(count + i);
                }
            }
            electric = Some((alpha, defect, residuals));
        }
        if with_gradient {
            // The biased orbital energies supply the metric multiplier of
            // the constrained Lagrangian, including on the common orbitals.
            let mut nuclear = crate::gradient::assemble_gradient(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
            )?;
            let mut spin_density = scf.density[0].clone();
            for (alpha, beta) in spin_density
                .as_mut_slice()
                .iter_mut()
                .zip(scf.density[1].as_slice())
            {
                *alpha -= beta;
            }
            let densities = [scf.p_total(), spin_density];
            add_fragment_gradient(
                &setup,
                &fragments,
                &result.multipliers,
                |pauli| &densities[usize::from(pauli != 0)],
                &mut nuclear,
            );
            gradient = Some(nuclear);
        }
        let diagnostics = crate::response::diagnostics::reference_diagnostics(scf)?;
        result_value(&result, correction, targets)
            .with(
                "reference_type",
                if roks {
                    "roks"
                } else if scf.unrestricted {
                    "uks"
                } else {
                    "rks"
                },
            )
            .with("scf_iterations", scf.iterations)
            .with(
                "scf_orbital_residual",
                scf.common_orbital_residual
                    .unwrap_or(diagnostics.orbital_residual),
            )
            .with("metric_defect", diagnostics.orthonormality_defect)
            .with("nonlocal_correlation_energy", scf.e_nlc)
            .with("density_alpha", &scf.density[0])
            .with("density_beta", &scf.density[1])
    };
    let mut report = report
        .with("requested_reference", reference)
        .with("dispersion_energy", dispersion)
        .with("smd_cds_energy", cds)
        .with("functional", setup.functional.description.as_str());
    if let Some(mut nuclear) = gradient {
        let dispersion = crate::dispersion::dispersion_gradient(
            &setup.molecule,
            &setup.dispersion,
            &setup.method.dispersion_functional_name(),
        )?;
        for (value, correction) in nuclear.iter_mut().zip(dispersion) {
            *value += correction;
        }
        let translation = nuclear.iter().fold(Vec3::zero(), |sum, g| sum + *g);
        report.set("forces", nuclear.iter().map(|g| -*g).collect::<Vec<_>>());
        report.set("gradient", nuclear);
        report.set("translation_residual", translation.norm());
        report.set("provenance", "analytic stationary constrained Lagrangian");
    }
    if let Some((alpha, defect, residuals)) = electric {
        report.set("polarizability", alpha);
        report.set("electric_constraint_defect", defect);
        report.set("electric_response_residuals", residuals);
    }
    Ok(report)
}

fn electric_operators(setup: &MolecularSetup) -> [Matrix; 3] {
    // The electric-field perturbation of the electronic Hamiltonian is +r.
    crate::integrals::dipole_matrices(&setup.basis, Vec3::zero())
}

fn add_fragment_gradient<'a>(
    setup: &MolecularSetup,
    fragments: &[Option<FragmentProjector>],
    multipliers: &[f64],
    density: impl Fn(usize) -> &'a Matrix,
    gradient: &mut [Vec3],
) {
    let n = setup.basis.nao_sph;
    let mut weighted = Matrix::zeros(n, n);
    // Q(R) = (W S(R) + S(R) W)/2. The population equation
    // eliminates the multiplier derivative, leaving lambda Tr[P dQ].
    for (fragment, &lambda) in fragments.iter().zip(multipliers) {
        if let Some(fragment) = fragment {
            let p = density(fragment.pauli);
            for i in 0..n {
                for j in 0..n {
                    weighted[(i, j)] += lambda * fragment.weight(i, j) * p[(i, j)];
                }
            }
        }
    }
    if fragments.iter().any(Option::is_some) {
        for (g, contribution) in gradient
            .iter_mut()
            .zip(crate::integrals::deriv::overlap_gradient(
                &setup.basis,
                &weighted,
                setup.molecule.len(),
                1.0,
            ))
        {
            *g += contribution;
        }
    }
}

fn optimize_geometry(system: &System, controls: &Controls) -> Result<Value> {
    if !controls.bool("gradient", true)? {
        return Err(DftError::InvalidInput(
            "constrained geometry optimization requires gradient=true".into(),
        ));
    }
    let electric_response = controls.bool("constraint_electric_response", false)?;
    let defaults = crate::optimizer::OptOptions::default();
    let options = crate::optimizer::OptOptions {
        max_steps: controls.usize("max_steps", defaults.max_steps)?,
        gtol: controls.f64("gtol", defaults.gtol)?,
        step_max: controls.f64("step_max", defaults.step_max)?,
        history: controls.usize("history", defaults.history)?,
        ..defaults
    };
    let molecule = super::method::molecule_from(
        &system.numbers,
        &system.positions,
        controls
            .opt_i64("charge")?
            .unwrap_or(0)
            .try_into()
            .map_err(|_| {
                DftError::InvalidInput("charge is outside the molecular integer range".into())
            })?,
        controls.usize("multiplicity", 1)?,
    )?;
    let mut settings = controls.to_value();
    if let Value::Object(entries) = &mut settings {
        entries.retain(|(key, _)| {
            !matches!(
                key.as_str(),
                "optimize_geometry" | "max_steps" | "gtol" | "step_max" | "history"
            )
        });
    }
    settings.set("gradient", true);
    settings.set("constraint_electric_response", false);
    let mut trial_settings = settings.clone();
    let result = crate::optimizer::optimize_with_evaluator(&molecule, &options, |mol| {
        let mut displaced = system.clone();
        displaced.positions = mol.atoms.iter().map(|a| a.position.to_array()).collect();
        let report = run(&displaced, &Controls::from_value(&trial_settings)?)?;
        trial_settings.set(
            "initial_constraint_multipliers",
            report
                .get("constraint_multipliers")
                .expect("converged multipliers")
                .clone(),
        );
        let energy = report
            .get("energy")
            .and_then(Value::as_f64)
            .expect("constrained energy");
        let (_, cartesian) = report
            .get("gradient")
            .expect("constrained gradient")
            .numeric_array()
            .expect("Cartesian constrained gradient");
        let gradient = cartesian
            .chunks_exact(3)
            .map(|x| Vec3::new(x[0], x[1], x[2]))
            .collect();
        Ok((energy, gradient, report))
    })?;
    let positions: Vec<_> = result
        .molecule
        .atoms
        .iter()
        .map(|a| a.position.to_array())
        .collect();
    let mut report = if electric_response {
        let mut final_system = system.clone();
        final_system.positions.clone_from(&positions);
        settings.set("constraint_electric_response", true);
        settings.set(
            "initial_constraint_multipliers",
            result
                .scf
                .get("constraint_multipliers")
                .expect("converged multipliers")
                .clone(),
        );
        run(&final_system, &Controls::from_value(&settings)?)?
    } else {
        result.scf
    };
    report.set("positions", positions);
    report.set("geometry_converged", result.converged);
    report.set("optimization_steps", result.iterations);
    report.set(
        "grad_max",
        result.trajectory.last().expect("initial frame").grad_max,
    );
    Ok(report)
}

fn result_value<State>(
    result: &ConstrainedResult<State>,
    correction: f64,
    targets: Vec<f64>,
) -> Value {
    Value::object()
        .with("model", "fixed-geometry variational integer-determinant density constraints; AO populations count electrons, spin projections use Pauli magnetization; L = E + sum lambda (Q-target)")
        .with("energy", result.energy + correction)
        .with("constraint_lagrangian", result.lagrangian + correction)
        .with("constraint_multipliers", result.multipliers.clone())
        .with("constraint_targets", targets)
        .with("constraint_expectations", result.expectations.clone())
        .with("constraint_residuals", result.residuals.clone())
        .with("constraint_susceptibility", &result.susceptibility)
        .with("constraint_response_residuals", result.response_residuals.clone())
        .with("constraint_iterations", result.iterations)
        .with("converged", true)
        .with("warnings", Vec::<String>::new())
        .with("units", units(&[
            ("energy","hartree"), ("constraint_lagrangian","hartree"),
            ("constraint_multipliers","hartree"), ("constraint_targets","dimensionless"),
            ("constraint_expectations","dimensionless"), ("constraint_residuals","dimensionless"),
            ("constraint_susceptibility","1/hartree"), ("constraint_response_residuals","dimensionless, solver-specific raw residual"),
            ("dispersion_energy","hartree"), ("smd_cds_energy","hartree"),
            ("nonlocal_correlation_energy","hartree"), ("spinor_energy_components","hartree"),
            ("scf_orbital_residual","hartree"), ("metric_defect","dimensionless"),
            ("polarizability","e^2*bohr^2/hartree"),
            ("electric_constraint_defect","e*bohr/hartree"),
            ("electric_response_residuals","dimensionless, solver-specific raw residual"),
            ("density_real","dimensionless"), ("density_imaginary","dimensionless"),
            ("density_alpha","dimensionless"), ("density_beta","dimensionless"),
        ]))
}
