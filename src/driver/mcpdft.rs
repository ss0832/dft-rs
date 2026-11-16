// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete-active-space SCF and multiconfiguration pair-density functional
//! theory on the common driver.
//!
//! `cas = [active electrons, active orbitals]` defines the active space; the
//! remaining electrons fill doubly occupied core orbitals. The reference
//! SCF (`xc`, normally `hf`) supplies the starting orbitals, ordered by
//! orbital energy unless `active_orbitals` names the active ones. The
//! lowest `cas_roots` states of the reference multiplicity are averaged with
//! `state_weights` (equal by default). `casci = true` keeps the starting
//! orbitals.

use super::controls::Controls;
use super::method::{run_molecular_reference, MolecularSetup};
use super::molecular::{setup, units};
use super::value::Value;
use super::System;
use crate::grid::MolecularGrid;
use crate::integrals::{eri_incore, CoulombKernel};
use crate::linalg::Matrix;
use crate::mcscf::casscf::{natural_orbitals, CasOperators, CasProblem, CasResult, CasSpec};
use crate::mcscf::path::{pdft_derivatives, PdftDerivatives};
use crate::mcscf::pdft::{multistate, MultiState, OnTopFunctional, Translation};
use crate::{DftError, Result};

struct CasControls {
    electrons: usize,
    orbitals: usize,
    roots: usize,
    weights: Option<Vec<f64>>,
    active: Option<Vec<usize>>,
    casci: bool,
    max_iterations: usize,
    gradient_tolerance: f64,
    energy_tolerance: f64,
    ci_tolerance: f64,
    ci_max_iterations: usize,
}

fn cas_controls(controls: &Controls) -> Result<CasControls> {
    let cas = controls.opt_usize_list("cas")?.ok_or_else(|| {
        DftError::InvalidInput(
            "cas = [active electrons, active orbitals] defines the active space".into(),
        )
    })?;
    if cas.len() != 2 {
        return Err(DftError::InvalidInput(
            "cas holds two integers: active electrons and active orbitals".into(),
        ));
    }
    Ok(CasControls {
        electrons: cas[0],
        orbitals: cas[1],
        roots: controls.usize("cas_roots", 1)?,
        weights: controls.opt_f64_list("state_weights")?,
        active: controls.opt_usize_list("active_orbitals")?,
        casci: controls.bool("casci", false)?,
        max_iterations: controls.usize("casscf_max_iter", 200)?,
        // The PDFT energy is not stationary in the orbitals: it follows
        // their convergence linearly, hence the tight default.
        gradient_tolerance: controls.f64("casscf_gradient_tol", 1e-8)?,
        energy_tolerance: controls.f64("casscf_energy_tol", 1e-10)?,
        ci_tolerance: controls.f64("ci_tol", 1e-10)?,
        ci_max_iterations: controls.usize("ci_max_iter", 200)?,
    })
}

/// Starting orbitals with the active space after the core: the named
/// active orbitals, the core filled in order of orbital energy.
fn ordered_orbitals(
    c: &Matrix,
    n_core: usize,
    n_active: usize,
    active: Option<&[usize]>,
) -> Result<Matrix> {
    let nmo = c.cols;
    let order: Vec<usize> = match active {
        None => (0..nmo).collect(),
        Some(list) => {
            let mut unique = list.to_vec();
            unique.sort_unstable();
            unique.dedup();
            if list.len() != n_active
                || unique.len() != n_active
                || unique.iter().any(|&i| i >= nmo)
            {
                return Err(DftError::InvalidInput(format!(
                    "active_orbitals names {n_active} distinct orbital indices below {nmo}"
                )));
            }
            let rest: Vec<usize> = (0..nmo).filter(|i| !list.contains(i)).collect();
            rest[..n_core]
                .iter()
                .chain(list)
                .chain(&rest[n_core..])
                .copied()
                .collect()
        }
    };
    let mut out = Matrix::zeros(c.rows, nmo);
    for (col, &source) in order.iter().enumerate() {
        for row in 0..c.rows {
            out[(row, col)] = c[(row, source)];
        }
    }
    Ok(out)
}

pub(super) struct Computed {
    pub(super) result: CasResult,
    pub(super) spec: CasSpec,
    scf_energy: f64,
    scf_converged: bool,
    casci: bool,
}

/// Reads the active-space controls, runs the reference and the CASSCF (or
/// CASCI) and hands setup, problem and result to `finish`; `id` finishes
/// the controls.
pub(super) fn compute(
    system: &System,
    controls: &Controls,
    id: &str,
    finish: impl FnOnce(&MolecularSetup, &CasProblem, &Computed) -> Result<Value>,
) -> Result<Value> {
    let cc = cas_controls(controls)?;
    let setup = setup(system, controls)?;
    if !setup.basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: format!("{id} with effective core potentials"),
            missing: "the matched valence active-space electron counts and derivative contract".into(),
        });
    }
    if controls.usize("derivative_order", 0)? != 0
        && setup
            .options
            .external_field
            .is_some_and(|field| field.to_array().iter().any(|&component| component != 0.0))
    {
        return Err(DftError::Unsupported {
            feature: format!("{id} nuclear derivatives in a static electric field"),
            missing: "the field-dependent one-electron and nuclear paths in the active-space derivative".into(),
        });
    }
    controls.finish(id)?;
    let electrons = setup.molecule.n_electrons()?;
    let multiplicity = setup.molecule.multiplicity;
    let unpaired = multiplicity.saturating_sub(1);
    if cc.electrons > electrons
        || (electrons - cc.electrons) % 2 != 0
        || unpaired > cc.electrons
        || (cc.electrons - unpaired) % 2 != 0
        || cc.orbitals == 0 && cc.electrons > 0
    {
        return Err(DftError::InvalidInput(format!(
            "{} active electrons do not fit the {electrons} electrons and the multiplicity \
             {multiplicity} of the molecule with a closed-shell core",
            cc.electrons
        )));
    }
    let n_core = (electrons - cc.electrons) / 2;
    let n_alpha = (cc.electrons + unpaired) / 2;
    let n_beta = cc.electrons - n_alpha;
    let roots = cc.roots.max(1);
    let weights = match cc.weights {
        Some(w) if w.len() == roots => w,
        Some(w) => {
            return Err(DftError::InvalidInput(format!(
                "state_weights has {} entries for {roots} states",
                w.len()
            )))
        }
        None => vec![1.0 / roots as f64; roots],
    };
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if n_core + cc.orbitals > scf.mo_coeff[0].cols {
        return Err(DftError::InvalidInput(format!(
            "the basis holds {} orbitals, fewer than {n_core} core and {} active ones",
            scf.mo_coeff[0].cols, cc.orbitals
        )));
    }
    let start = ordered_orbitals(&scf.mo_coeff[0], n_core, cc.orbitals, cc.active.as_deref())?;
    let eri = eri_incore(
        &setup.basis,
        CoulombKernel::Coulomb,
        setup.options.incore_mb.max(256),
    )?;
    let spec = CasSpec {
        n_core,
        n_active: cc.orbitals,
        n_alpha,
        n_beta,
        weights,
        ci_tolerance: cc.ci_tolerance,
        ci_max_iterations: cc.ci_max_iterations,
    };
    let problem = CasProblem::new(
        CasOperators {
            h_core: &scf.h_core,
            eri: &eri,
            e_nuc: scf.e_nuc,
        },
        spec.clone(),
    )?;
    let result = if cc.casci {
        let point = problem.evaluate(&start, None)?;
        CasResult {
            energy_history: vec![point.average_energy],
            point,
            iterations: 0,
            converged: true,
        }
    } else {
        problem.optimize(
            &start,
            cc.max_iterations,
            cc.gradient_tolerance,
            cc.energy_tolerance,
        )?
    };
    let computed = Computed {
        result,
        spec,
        scf_energy: reference.total_energy(),
        scf_converged: scf.converged,
        casci: cc.casci,
    };
    finish(&setup, &problem, &computed)
}

fn cas_report(setup: &MolecularSetup, c: &Computed, model: &str) -> Result<Value> {
    let point = &c.result.point;
    let (occupations, _) = natural_orbitals(&point.gamma)?;
    let mut warnings: Vec<Value> = Vec::new();
    if !c.result.converged {
        warnings.push(Value::Str(
            "WARNING [casscf] the orbital optimization did not converge".into(),
        ));
    }
    if !c.scf_converged {
        warnings.push(Value::Str(
            "WARNING [scf] the reference SCF that supplied the starting orbitals did not converge"
                .into(),
        ));
    }
    Ok(Value::object()
        .with("model", model)
        .with("basis", setup.method.basis.as_str())
        .with(
            "reference_functional",
            setup.functional.description.as_str(),
        )
        .with("scf_energy", c.scf_energy)
        .with("energy", point.energies[0])
        .with("cas_energies", point.energies.clone())
        .with("average_energy", point.average_energy)
        .with("state_weights", c.spec.weights.clone())
        .with("n_core", c.spec.n_core)
        .with("n_active", c.spec.n_active)
        .with("active_alpha", c.spec.n_alpha)
        .with("active_beta", c.spec.n_beta)
        .with("determinants", point.states.vectors[0].len())
        .with("natural_occupations", occupations)
        .with("spin_squared", point.states.spin_squared.clone())
        .with("ci_residuals", point.states.residuals.clone())
        .with("orbital_gradient", point.gradient_norm())
        .with(
            "orbital_optimization",
            if c.casci { "casci" } else { "casscf" },
        )
        .with("macro_iterations", c.result.iterations)
        .with("orbitals_converged", c.result.converged)
        .with("energy_history", c.result.energy_history.clone())
        .with("warnings", Value::Array(warnings))
        .with("provenance", "analytic"))
}

const CAS_UNITS: [(&str, &str); 7] = [
    ("scf_energy", "hartree"),
    (
        "cas_energies",
        "hartree, active-space energy of every selected state",
    ),
    (
        "average_energy",
        "hartree, state-averaged active-space energy",
    ),
    ("state_weights", "dimensionless"),
    (
        "natural_occupations",
        "dimensionless, state-averaged active natural occupations",
    ),
    ("ci_residuals", "hartree, norm of H C - E C"),
    ("orbital_gradient", "hartree, largest element of dE/dkappa"),
];

/// Gradients and Hessians of the derivative path in the report.
fn derivative_entries(
    report: &mut Value,
    labels: &mut Vec<(&'static str, &'static str)>,
    derivatives: &PdftDerivatives,
    natoms: usize,
    with_reference: bool,
) {
    let gradients = |g: &[Vec<f64>]| -> Vec<Vec<Vec<f64>>> {
        g.iter()
            .map(|state| {
                (0..natoms)
                    .map(|a| state[3 * a..3 * a + 3].to_vec())
                    .collect()
            })
            .collect()
    };
    let hessians = |h: &[Matrix]| -> Vec<Vec<Vec<f64>>> {
        h.iter()
            .map(|m| {
                (0..m.rows)
                    .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect())
                    .collect()
            })
            .collect()
    };
    report.set("gradients", gradients(&derivatives.gradient));
    labels.push(("gradients", "hartree/bohr, [state][atom][axis]"));
    if with_reference {
        report.set("cas_gradients", gradients(&derivatives.reference_gradient));
        labels.push((
            "cas_gradients",
            "hartree/bohr, active-space energy, [state][atom][axis]",
        ));
    }
    if let Some(h) = &derivatives.hessian {
        report.set("hessians", hessians(h));
        labels.push((
            "hessians",
            "hartree/bohr^2, [state][3 atom + axis][3 atom + axis]",
        ));
    }
    if let (true, Some(h)) = (with_reference, &derivatives.reference_hessian) {
        report.set("cas_hessians", hessians(h));
        labels.push(("cas_hessians", "hartree/bohr^2, active-space energy"));
    }
    report.set("derivative_paths", derivatives.paths);
    report.set("derivative_stationarity", derivatives.stationarity);
    report.set("derivative_solve_residual", derivatives.solve_residual);
    labels.push((
        "derivative_stationarity",
        "hartree, largest orbital and CI stationarity residual left along the paths",
    ));
    labels.push((
        "derivative_solve_residual",
        "dimensionless, largest relative residual of the path solves",
    ));
}

pub(super) fn casscf(system: &System, controls: &Controls) -> Result<Value> {
    let order = controls.usize("derivative_order", 0)?;
    if order > 2 {
        return Err(DftError::InvalidInput(
            "derivative_order is 0, 1 or 2".into(),
        ));
    }
    compute(system, controls, "casscf", |setup, problem, computed| {
        let mut report = cas_report(
            setup,
            computed,
            "complete-active-space SCF: string-driven determinant CI in the active space, \
             state-averaged orbital optimization by preconditioned limited-memory BFGS; \
             nonrelativistic all-electron Hamiltonian, closed-shell core",
        )?;
        let mut labels = CAS_UNITS.to_vec();
        labels.push(("energy", "hartree, lowest selected state"));
        labels.push((
            "spin_squared",
            "dimensionless, <S^2> of every selected state",
        ));
        labels.push(("energy_history", "hartree"));
        if order > 0 {
            let derivatives = pdft_derivatives(
                problem,
                &computed.result.point,
                None,
                &setup.molecule,
                &setup.basis,
                None,
                order,
                setup.options.incore_mb.max(256),
            )?;
            derivative_entries(
                &mut report,
                &mut labels,
                &derivatives,
                setup.molecule.len(),
                false,
            );
        }
        report.set("units", units(&labels));
        Ok(report)
    })
}

pub(super) fn mcpdft(system: &System, controls: &Controls) -> Result<Value> {
    let name = controls.str("on_top_functional", "tpbe")?;
    let lambda = controls.opt_f64("pdft_lambda")?;
    let method_name = controls.str("pdft_method", "state-specific")?;
    let functional = OnTopFunctional::parse(&name, lambda)?;
    let method = MultiState::parse(&method_name)?;
    let order = controls.usize("derivative_order", 0)?;
    if order > 2 {
        return Err(DftError::InvalidInput(
            "derivative_order is 0, 1 or 2".into(),
        ));
    }
    if order > 0 && method != MultiState::StateSpecific {
        return Err(DftError::Unsupported {
            feature: format!("nuclear derivatives of {method_name}"),
            missing: "the response of the multistate rotation or the linearized Hamiltonian".into(),
        });
    }
    compute(system, controls, "mcpdft", |setup, problem, computed| {
        let point = &computed.result.point;
        let grid = MolecularGrid::build(&setup.molecule, &setup.grid_options)?;
        let result = multistate(problem, point, &functional, &setup.basis, &grid, method)?;
        let mut report = cas_report(
            setup,
            computed,
            "multiconfiguration pair-density functional theory: classical Coulomb energy and a \
             translated on-top functional of the density and the on-top pair density of the \
             (state-averaged) active-space reference; hybrid fraction mixes the active-space energy",
        )?;
        let column = |f: fn(&crate::mcscf::pdft::PdftEnergy) -> f64| -> Vec<f64> {
            result.state_terms.iter().map(f).collect()
        };
        let rows = |m: &Matrix| -> Vec<Vec<f64>> {
            (0..m.rows)
                .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect())
                .collect()
        };
        report.set("energy", result.energies[0]);
        report.set("pdft_energies", result.energies.clone());
        report.set("on_top_functional", functional.name.as_str());
        report.set(
            "translation",
            match functional.translation {
                Translation::Translated => "translated",
                Translation::FullyTranslated => "fully translated",
            },
        );
        report.set("pdft_lambda", functional.lambda);
        report.set("pdft_method", method_name.to_ascii_lowercase());
        report.set("state_nuclear_repulsion", column(|t| t.nuclear_repulsion));
        report.set("state_one_electron", column(|t| t.one_electron));
        report.set("state_coulomb", column(|t| t.coulomb));
        report.set("state_on_top", column(|t| t.on_top));
        report.set("state_pdft", column(|t| t.pdft));
        report.set("state_reference", column(|t| t.reference));
        report.set("state_total", column(|t| t.total));
        report.set("grid_electrons", column(|t| t.electrons));
        report.set("grid_on_top_pairs", column(|t| t.on_top_pairs));
        report.set("intermediate_rotation", rows(&result.rotation));
        report.set("effective_hamiltonian", rows(&result.hamiltonian));
        report.set("eigenvectors", rows(&result.eigenvectors));
        if let Some((pdft, linearized)) = result.expansion_energy {
            report.set("expansion_pdft_energy", pdft);
            report.set("expansion_linearized_energy", linearized);
        }
        if let Some((objective, sweeps)) = result.objective {
            report.set("coulomb_objective", objective);
            report.set("rotation_sweeps", sweeps);
        }
        let mut labels = CAS_UNITS.to_vec();
        labels.extend([
            ("energy", "hartree, lowest PDFT state"),
            (
                "spin_squared",
                "dimensionless, <S^2> of every selected state",
            ),
            ("energy_history", "hartree"),
            ("pdft_energies", "hartree, ascending"),
            ("pdft_lambda", "dimensionless"),
            ("state_nuclear_repulsion", "hartree"),
            (
                "state_one_electron",
                "hartree, Tr[h P] of every reference or intermediate state",
            ),
            ("state_coulomb", "hartree, 1/2 Tr[P J[P]]"),
            ("state_on_top", "hartree, on-top functional energy"),
            ("state_pdft", "hartree, sum of the four terms"),
            (
                "state_reference",
                "hartree, active-space energy of the same state",
            ),
            (
                "state_total",
                "hartree, lambda reference + (1 - lambda) pdft",
            ),
            ("grid_electrons", "dimensionless, integrated density"),
            (
                "grid_on_top_pairs",
                "dimensionless, integrated on-top pair density",
            ),
            (
                "intermediate_rotation",
                "dimensionless, reference states by intermediate states",
            ),
            ("effective_hamiltonian", "hartree"),
            ("eigenvectors", "dimensionless"),
            (
                "expansion_pdft_energy",
                "hartree, hybrid PDFT energy at the state-averaged expansion density",
            ),
            (
                "expansion_linearized_energy",
                "hartree, linearized Hamiltonian at the expansion density",
            ),
            (
                "coulomb_objective",
                "hartree, sum of active Coulomb self-energies of the intermediate states",
            ),
        ]);
        if order > 0 {
            let derivatives = pdft_derivatives(
                problem,
                point,
                Some(&functional),
                &setup.molecule,
                &setup.basis,
                Some(&grid),
                order,
                setup.options.incore_mb.max(256),
            )?;
            derivative_entries(
                &mut report,
                &mut labels,
                &derivatives,
                setup.molecule.len(),
                true,
            );
        }
        report.set("units", units(&labels));
        Ok(report)
    })
}
