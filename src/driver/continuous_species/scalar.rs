// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Explicit experimental frozen scalar actions on a common active AO chart.
//!
//! Endpoint full-system integrals are retained. This route is not a compressed
//! potential and never reconstructs projectors from a reaction target SCF.

mod geometry;

use super::fragment_driver::object_control;
use crate::basis::{n_cart, n_sph, MolBasis, MolShell};
use crate::driver::{Controls, System, Value};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix};
use crate::linalg::{AccurateSum, Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::pseudo_sub::fit::{projected_lbfgs, OptimizerResult};
use crate::pseudo_sub::reduction_basis::cross_basis_overlap;
use crate::pseudo_sub::reduction_reference::{
    ActiveSpaceScf, FrozenDensityEvaluator, MappedFrozenAction, WeightedFrozenAction,
};
use crate::pseudo_sub::reduction_support::OrthogonalReducedChart;
use crate::scf::{EriMode, Guess, ScfOptions};
use crate::system::{Atom, Molecule};
use crate::xc::{resolve, ResolvedFunctional};
use crate::{DftError, Result};

pub const VERSION: &str = "frozen-scalar-action-v1";
/// Independent alpha/beta frozen spaces and active counts; never inferred from v1.
pub const COLLINEAR_VERSION: &str = "frozen-scalar-collinear-action-v1";

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn required<'a>(c: &'a Controls, key: &str) -> Result<&'a Value> {
    c.take(key)
        .ok_or_else(|| DftError::InvalidInput(format!("missing scalar model field {key}")))
}
fn list(v: &Value) -> Result<&[Value]> {
    v.as_array()
        .ok_or_else(|| DftError::InvalidInput("scalar model array required".into()))
}
fn string(v: &Value) -> Result<String> {
    v.as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| DftError::InvalidInput("nonempty scalar model identifier required".into()))
}
fn matrix(v: &Value, columns: Option<usize>) -> Result<Matrix> {
    let rows = list(v)?;
    let n = columns.unwrap_or(if rows.is_empty() {
        0
    } else {
        list(&rows[0])?.len()
    });
    let mut result = Matrix::zeros(rows.len(), n);
    for (i, row) in rows.iter().enumerate() {
        let values = list(row)?;
        if values.len() != n {
            return invalid("scalar model matrix is ragged");
        }
        for (j, value) in values.iter().enumerate() {
            result[(i, j)] = value.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                DftError::InvalidInput("scalar model matrix must be finite".into())
            })?;
        }
    }
    Ok(result)
}
fn molecule(v: &Value) -> Result<Molecule> {
    let c = Controls::from_value(v)?;
    let labels = c
        .opt_usize_list("numbers")?
        .ok_or_else(|| DftError::InvalidInput("scalar model numbers required".into()))?;
    let positions = matrix(required(&c, "positions")?, Some(3))?;
    let charge = i32::try_from(c.opt_i64("charge")?.unwrap_or(0))
        .map_err(|_| DftError::InvalidInput("scalar model charge overflow".into()))?;
    let multiplicity = c.usize("multiplicity", 1)?;
    if multiplicity == 0
        || labels.len() != positions.rows
        || labels.is_empty()
        || labels.iter().any(|z| *z == 0 || *z > 118)
    {
        return invalid(
            "scalar model requires finite real-atom molecules and a positive multiplicity",
        );
    }
    c.finish("scalar model molecule")?;
    let atoms = labels
        .iter()
        .enumerate()
        .map(|(i, z)| Atom {
            z: *z as u8,
            position: Vec3::new(positions[(i, 0)], positions[(i, 1)], positions[(i, 2)]),
        })
        .collect();
    let result = Molecule::new(atoms)
        .with_charge(charge)
        .with_multiplicity(multiplicity);
    result.n_alpha_beta()?;
    for i in 0..result.len() {
        for j in 0..i {
            if (result.atoms[i].position - result.atoms[j].position).norm() < 1e-10 {
                return invalid("scalar model nuclei coincide");
            }
        }
    }
    Ok(result)
}
fn basis(v: &Value, molecule: &Molecule) -> Result<MolBasis> {
    let mut result = MolBasis {
        shells: Vec::new(),
        nao_sph: 0,
        nao_cart: 0,
        max_l: 0,
        ecps: Vec::new(),
        n_core_electrons: 0,
    };
    for value in list(v)? {
        let c = Controls::from_value(value)?;
        let atom = c.usize("atom", usize::MAX)?;
        let l = c.usize("l", usize::MAX)?;
        let exps = c
            .opt_f64_list("exponents")?
            .ok_or_else(|| DftError::InvalidInput("shell exponents required".into()))?;
        let coefs = c.opt_f64_list("normalized_coefficients")?.ok_or_else(|| {
            DftError::InvalidInput("normalized shell coefficients required".into())
        })?;
        if atom >= molecule.len()
            || l > 1
            || exps.is_empty()
            || exps.len() != coefs.len()
            || exps.iter().any(|x| !x.is_finite() || *x <= 0.)
            || coefs.iter().any(|x| !x.is_finite())
        {
            return invalid("scalar model supports finite normalized s/p all-electron shells only");
        }
        c.finish("scalar model shell")?;
        result.shells.push(MolShell {
            atom,
            l: l as u8,
            center: molecule.atoms[atom].position,
            exps,
            coefs,
            sph_off: result.nao_sph,
            cart_off: result.nao_cart,
        });
        result.nao_sph += n_sph(l as u8);
        result.nao_cart += n_cart(l as u8);
        result.max_l = result.max_l.max(l as u8);
    }
    if result.shells.is_empty() {
        return invalid("scalar model basis is empty");
    }
    let s = overlap_matrix(&result);
    if (0..s.rows).any(|i| (s[(i, i)] - 1.).abs() > 1e-8) {
        return invalid("saved scalar AO contractions are not normalized");
    }
    Ok(result)
}

struct Endpoint {
    molecule: Molecule,
    basis: MolBasis,
    frozen: FrozenSpace,
    geometry_atom_owners: Option<Vec<usize>>,
}
enum FrozenSpace {
    Restricted {
        density: Matrix,
        orbitals: Matrix,
        chart: Box<OrthogonalReducedChart>,
    },
    Collinear {
        orbitals: [Matrix; 2],
    },
}
impl FrozenSpace {
    fn restricted(&self) -> Result<(&Matrix, &Matrix, &OrthogonalReducedChart)> {
        match self {
            Self::Restricted { density, orbitals, chart } => Ok((density, orbitals, chart)),
            Self::Collinear { .. } => Err(DftError::Unsupported {
                feature: "saved unrestricted scalar weight interpolation".into(),
                missing: "a coupled spin-resolved weighted action; use an explicit endpoint geometry calculation".into(),
            }),
        }
    }
}
struct State {
    id: String,
    coefficient: f64,
    molecule: Molecule,
    basis: MolBasis,
    active_electrons: usize,
    active_spin_electrons: [usize; 2],
    endpoints: Vec<Endpoint>,
}

struct Site {
    id: String,
    candidates: Vec<String>,
}

/// Weight optimization only; physical SCF thresholds are separate controls.
pub struct ScalarOptimization {
    pub tolerance: f64,
    pub max_iterations: usize,
    /// `joint`, `one-pass` or `cyclic`; blocks follow the saved site order.
    pub strategy: String,
}

/// A validated immutable native-unit record. It stores one ordered raw AO
/// definition per state, shared by every endpoint, and one common reference
/// constant per endpoint. Distinct reaction states retain distinct spaces.
pub struct ScalarActionRecord {
    value: Value,
    ids: Vec<String>,
    reference_energies: Vec<f64>,
    reference_residuals: Vec<f64>,
    states: Vec<State>,
    grid_level: usize,
    sites: Vec<Site>,
    choices: Vec<Vec<usize>>,
    functional: ResolvedFunctional,
    unrestricted: bool,
}
impl ScalarActionRecord {
    pub fn from_value(value: &Value) -> Result<Self> {
        let c = Controls::from_value(value)?;
        let version = c.str("model_version", "")?;
        let reference = c.str("reference", "")?;
        let unrestricted = version == COLLINEAR_VERSION && reference == "uks";
        if !(unrestricted || version == VERSION && reference == "rks")
            || c.str("provider", "")? != "builtin-functional-registry-v1"
            || c.str("shell_convention", "")? != "normalized-molecular-spherical-ao-v1"
        {
            return Err(DftError::Unsupported {
                feature: "saved frozen scalar action".into(),
                missing: "paired RKS frozen-scalar-action-v1 or explicit UKS frozen-scalar-collinear-action-v1, builtin functional registry and normalized s/p AO contract".into(),
            });
        }
        let functional_input = required(&c, "functional")?;
        let Value::Object(fields) = functional_input else {
            return invalid("saved scalar functional specification must be an object");
        };
        if fields.iter().any(|(name, _)| {
            !matches!(
                name.as_str(),
                "xc" | "x" | "c" | "xc_terms" | "omega" | "xc_parameters"
            )
        }) {
            return invalid("saved scalar functional contains unsupported settings");
        }
        let functional_controls = Controls::from_value(functional_input)?;
        let method = crate::driver::method::MolecularMethod::from_controls(&functional_controls)?;
        let mut functional = if let Some(terms) = &method.xc_terms {
            if method.xc.is_some() || method.x.is_some() || method.c.is_some() {
                return invalid("saved scalar xc_terms cannot be combined with xc/x/c");
            }
            ResolvedFunctional::weighted(terms)?
        } else {
            resolve(&method.functional_spec()?)?
        };
        if let Some(omega) = method.omega {
            functional = functional.with_omega(omega)?;
        }
        if !method.xc_parameters.is_empty() {
            functional = functional.with_parameters(&method.xc_parameters)?;
        }
        functional_controls.finish("saved scalar functional")?;
        let grid_level = c.usize("grid_level", 1)?;
        if grid_level > 9 {
            return invalid("scalar model grid level must be in 0..9");
        }
        let mut sites = Vec::new();
        for site in list(required(&c, "sites")?)? {
            let s = Controls::from_value(site)?;
            let id = string(required(&s, "id")?)?;
            let candidates = list(required(&s, "candidates")?)?
                .iter()
                .map(string)
                .collect::<Result<Vec<_>>>()?;
            if candidates.len() < 2
                || sites.iter().any(|s: &Site| s.id == id)
                || candidates
                    .iter()
                    .enumerate()
                    .any(|(i, x)| candidates[..i].contains(x))
            {
                return invalid(
                    "scalar sites require unique identities and at least two distinct candidates",
                );
            }
            s.finish("scalar site")?;
            sites.push(Site { id, candidates });
        }
        if sites.is_empty() {
            return invalid("scalar model requires at least one site");
        }
        let mut choices = Vec::new();
        let mut ids = Vec::new();
        let mut reference_energies = Vec::new();
        let mut reference_residuals = Vec::new();
        for reference in list(required(&c, "references")?)? {
            let r = Controls::from_value(reference)?;
            let id = string(required(&r, "id")?)?;
            if ids.contains(&id) {
                return invalid("duplicate scalar endpoint identity");
            }
            let tuple = r.opt_usize_list("site_choices")?.ok_or_else(|| {
                DftError::InvalidInput("scalar endpoint site tuple required".into())
            })?;
            if tuple.len() != sites.len()
                || tuple
                    .iter()
                    .zip(&sites)
                    .any(|(i, s)| *i >= s.candidates.len())
                || choices.contains(&tuple)
            {
                return invalid("scalar endpoint has an invalid or duplicate site tuple");
            }
            choices.push(tuple);
            string(required(&r, "construction_identity")?)?;
            if r.str("construction_state", "")? != "reactant-only" {
                return invalid("scalar endpoints require declared reactant-only construction");
            }
            let energy = r.f64("energy", f64::NAN)?;
            let residual = r.f64("orbital_residual", f64::NAN)?;
            if !energy.is_finite() || !residual.is_finite() || !(0.0..=1e-8).contains(&residual) {
                return invalid(
                    "scalar references need finite energies and converged raw residuals",
                );
            }
            r.finish("scalar reference")?;
            ids.push(id);
            reference_energies.push(energy);
            reference_residuals.push(residual);
        }
        if ids.len() < 2 {
            return invalid("scalar interpolation requires at least two explicit endpoints");
        }
        if sites
            .iter()
            .try_fold(1usize, |n, s| n.checked_mul(s.candidates.len()))
            != Some(ids.len())
        {
            return invalid("scalar model must provide every joint site combination; additive endpoint completion is not permitted");
        }
        let mut states = Vec::new();
        for value in list(required(&c, "states")?)? {
            let s = Controls::from_value(value)?;
            let id = string(required(&s, "id")?)?;
            let coefficient = s.f64("coefficient", f64::NAN)?;
            let active_electrons = s.usize("active_electrons", 0)?;
            let active_spin_electrons = if unrestricted {
                let counts = s.opt_usize_list("active_spin_electrons")?.ok_or_else(|| {
                    DftError::InvalidInput(
                        "UKS scalar state requires explicit [alpha,beta] active_spin_electrons"
                            .into(),
                    )
                })?;
                <[usize; 2]>::try_from(counts).map_err(|_| {
                    DftError::InvalidInput(
                        "active_spin_electrons must have exactly two entries".into(),
                    )
                })?
            } else {
                [active_electrons / 2; 2]
            };
            let raw_molecule = molecule(required(&s, "raw_molecule")?)?;
            let raw_basis = basis(required(&s, "raw_basis")?, &raw_molecule)?;
            let overlap = overlap_matrix(&raw_basis);
            if !coefficient.is_finite()
                || coefficient == 0.
                || active_electrons == 0
                || (!unrestricted && (active_electrons % 2 != 0 || raw_molecule.multiplicity != 1))
                || active_spin_electrons.iter().any(|n| *n > raw_basis.nao_sph)
                || active_spin_electrons[0].checked_add(active_spin_electrons[1])
                    != Some(active_electrons)
                || raw_molecule.n_alpha_beta()?
                    != (active_spin_electrons[0], active_spin_electrons[1])
            {
                return invalid("invalid fixed scalar state coefficient or active-electron ledger");
            }
            if states.iter().any(|x: &State| x.id == id) {
                return invalid("duplicate scalar state identity");
            }
            let mut endpoints = Vec::new();
            for value in list(required(&s, "endpoints")?)? {
                let e = Controls::from_value(value)?;
                let full_molecule = molecule(required(&e, "molecule")?)?;
                let full_basis = basis(required(&e, "basis")?, &full_molecule)?;
                let frozen = if unrestricted {
                    [
                        matrix(required(&e, "frozen_orbitals_alpha")?, None)?,
                        matrix(required(&e, "frozen_orbitals_beta")?, None)?,
                    ]
                } else {
                    let paired = matrix(required(&e, "frozen_orbitals")?, None)?;
                    [paired.clone(), paired]
                };
                let geometry_atom_owners = e.opt_usize_list("geometry_atom_owners")?;
                let geometry_transport = e.opt_str("geometry_transport")?;
                if geometry_atom_owners.is_some() != geometry_transport.is_some()
                    || geometry_transport
                        .as_deref()
                        .is_some_and(|v| v != "fixed-internal-affine-v1")
                    || geometry_atom_owners.as_ref().is_some_and(|owners| {
                        owners.len() != full_molecule.len()
                            || owners.iter().any(|i| *i >= raw_molecule.len())
                            || (0..raw_molecule.len()).any(|i| !owners.contains(&i))
                    })
                {
                    return invalid("saved affine geometry requires fixed-internal-affine-v1 and one valid owner per full atom, covering every raw center");
                }
                let full_counts = full_molecule.n_alpha_beta()?;
                if (!unrestricted && full_molecule.multiplicity != 1)
                    || frozen.iter().any(|b| b.rows != full_basis.nao_sph)
                    || active_spin_electrons[0].checked_add(frozen[0].cols) != Some(full_counts.0)
                    || active_spin_electrons[1].checked_add(frozen[1].cols) != Some(full_counts.1)
                {
                    return invalid(
                        "scalar full/frozen/active spin electron ledger is inconsistent",
                    );
                }
                e.finish("scalar endpoint")?;
                let full_overlap = overlap_matrix(&full_basis);
                let cross = cross_basis_overlap(&full_basis, &raw_basis)?;
                let alpha_chart = OrthogonalReducedChart::new(
                    &full_overlap,
                    &cross,
                    &overlap,
                    &frozen[0],
                    1e-10,
                )?;
                let frozen = if unrestricted {
                    // Both maps must have the same declared raw rank; no spin
                    // may silently drop a direction when its frozen space differs.
                    OrthogonalReducedChart::new(
                        &full_overlap,
                        &cross,
                        &overlap,
                        &frozen[1],
                        1e-10,
                    )?;
                    FrozenSpace::Collinear { orbitals: frozen }
                } else {
                    let [orbitals, _] = frozen;
                    let mut density = orbitals.matmul_transpose_seq(&orbitals);
                    for x in density.as_mut_slice() {
                        *x *= 2.;
                    }
                    FrozenSpace::Restricted {
                        density,
                        orbitals,
                        chart: Box::new(alpha_chart),
                    }
                };
                endpoints.push(Endpoint {
                    molecule: full_molecule,
                    basis: full_basis,
                    frozen,
                    geometry_atom_owners,
                });
            }
            if endpoints.len() != ids.len() {
                return invalid("scalar state endpoint count differs from reference identities");
            }
            s.finish("scalar state")?;
            states.push(State {
                id,
                coefficient,
                molecule: raw_molecule,
                basis: raw_basis,
                active_electrons,
                active_spin_electrons,
                endpoints,
            });
        }
        if states.is_empty() {
            return invalid("scalar action has no states");
        }
        // Every endpoint's reference occurs with the same stoichiometry. Thus
        // no large internal reference is subtracted after rounding a reaction.
        if states.len() > 1 && states.iter().map(|s| s.coefficient).sum::<f64>().abs() > 1e-12 {
            return invalid("scalar reaction state coefficients must cancel the common reference");
        }
        c.finish("saved scalar action")?;
        Ok(Self {
            value: value.clone(),
            ids,
            reference_energies,
            reference_residuals,
            states,
            grid_level,
            sites,
            choices,
            functional,
            unrestricted,
        })
    }
    pub fn from_json(text: &str) -> Result<Self> {
        Self::from_value(&Value::parse_json(text)?)
    }
    pub fn to_value(&self) -> Value {
        self.value.clone()
    }
    pub fn to_json(&self) -> String {
        self.value.to_json_pretty()
    }
    pub fn endpoint_ids(&self) -> &[String] {
        &self.ids
    }
    pub fn primary_molecule(&self) -> &Molecule {
        &self.states[0].molecule
    }

    /// Evaluate or optimize only endpoint weights. All state geometries,
    /// frozen orbitals and common maps remain exactly fixed in the record.
    pub fn calculate(
        &self,
        weights: &[f64],
        options: &ScfOptions,
        optimize: Option<ScalarOptimization>,
    ) -> Result<Value> {
        if self.unrestricted {
            return Err(DftError::Unsupported {
                feature: "saved unrestricted scalar weight interpolation".into(),
                missing: "a coupled spin-resolved weighted action; only fixed-endpoint geometry is connected".into(),
            });
        }
        self.joint_weights(weights)?;
        let xc = &self.functional;
        let grids = self
            .states
            .iter()
            .map(|s| {
                s.endpoints
                    .iter()
                    .map(|e| {
                        MolecularGrid::build(
                            &e.molecule,
                            &GridOptions {
                                level: self.grid_level,
                                weight_cutoff: 0.,
                                ..Default::default()
                            },
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let evaluators = self
            .states
            .iter()
            .zip(&grids)
            .map(|(s, g)| {
                s.endpoints
                    .iter()
                    .zip(g)
                    .map(|(e, g)| {
                        FrozenDensityEvaluator::new(
                            &e.molecule,
                            &e.basis,
                            g,
                            xc,
                            options,
                            e.frozen.restricted()?.0,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let actions = self
            .states
            .iter()
            .zip(&evaluators)
            .map(|(s, e)| {
                s.endpoints
                    .iter()
                    .zip(e)
                    .zip(&self.reference_energies)
                    .map(|((a, e), reference)| {
                        MappedFrozenAction::new(e, a.frozen.restricted()?.2, *reference)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let overlaps = self
            .states
            .iter()
            .map(|s| overlap_matrix(&s.basis))
            .collect::<Vec<_>>();
        let spaces = overlaps
            .iter()
            .map(|s| Orthogonalizer::canonical(s, options.orth_threshold))
            .collect::<Result<Vec<_>>>()?;
        if spaces.iter().any(|x| x.n_dropped != 0) {
            return invalid("scalar raw AO space is not full rank");
        }
        let mut previous = vec![None; self.states.len()];
        let mut evaluate = |weights: &[f64]| -> Result<(f64, Vec<f64>, Vec<Value>)> {
            let (joint_weights, chain) = self.joint_weights(weights)?;
            let mut objective = AccurateSum::default();
            let mut gradient = vec![0.; weights.len()];
            let mut reports = Vec::new();
            for (index, state) in self.states.iter().enumerate() {
                let action = WeightedFrozenAction::new(&actions[index], &joint_weights)?;
                let mut h = kinetic_matrix(&state.basis);
                let v = nuclear_matrix(&state.basis, &state.molecule, None);
                for (a, b) in h.as_mut_slice().iter_mut().zip(v.as_slice()) {
                    *a += b;
                }
                let empty = Matrix::zeros(state.basis.nao_sph, 0);
                let problem = ActiveSpaceScf {
                    overlap: &overlaps[index],
                    complement: &spaces[index].x,
                    frozen_orbitals: &empty,
                    active_electrons: state.active_electrons,
                    options,
                    initial_hcore: &h,
                };
                let mut warm_failure = None;
                let solution = match problem
                    .solve(previous[index].as_ref(), |p| action.evaluate_scalar(p))
                {
                    Err(error @ DftError::ScfNotConverged { .. }) if previous[index].is_some() => {
                        crate::logger::log(&format!("[active-scalar] warm trial failed: {error}; retrying independent core guess at unchanged stopping gates"));
                        warm_failure = Some(error.to_string());
                        problem.solve(None, |p| action.evaluate_scalar(p))?
                    }
                    other => other?,
                };
                let value = action.evaluate(&solution.active_density)?;
                objective.add_product(state.coefficient, value.evaluation.energy);
                for (g, derivatives) in gradient.iter_mut().zip(&chain) {
                    let mut contribution = AccurateSum::default();
                    for (d, e) in derivatives.iter().zip(&value.term_energies) {
                        contribution.add_product(*d, *e);
                    }
                    *g += state.coefficient * contribution.value();
                }
                reports.push(
                    Value::object()
                        .with("id", state.id.clone())
                        .with("warm_start_failure", warm_failure)
                        .with("coefficient", state.coefficient)
                        .with("active_electrons", state.active_electrons)
                        .with("ao_count", state.basis.nao_sph)
                        .with(
                            "endpoint_full_electrons",
                            state
                                .endpoints
                                .iter()
                                .map(|e| e.molecule.n_electrons())
                                .collect::<Result<Vec<_>>>()?,
                        )
                        .with(
                            "endpoint_full_ao",
                            state
                                .endpoints
                                .iter()
                                .map(|e| e.basis.nao_sph)
                                .collect::<Vec<_>>(),
                        )
                        .with("energy_without_internal_reference", value.evaluation.energy)
                        .with("reference_energy", value.evaluation.reference_energy)
                        .with("electronic_energy", value.evaluation.electronic_energy)
                        .with("nuclear_energy", value.evaluation.nuclear_energy)
                        .with("term_energies", value.term_energies)
                        .with("scf_orbital_residual", solution.projected_orbital_residual)
                        .with("particle_number_error", solution.particle_number_error)
                        .with("idempotency_error", solution.active_idempotency_error)
                        .with("iterations", solution.history.len()),
                );
                previous[index] = Some(solution.active_density);
            }
            Ok((objective.value(), gradient, reports))
        };
        let mut block_history = Vec::new();
        let strategy = optimize
            .as_ref()
            .map_or("none", |o| o.strategy.as_str())
            .to_string();
        let optimizer = if let Some(settings) = optimize {
            let tolerance = settings.tolerance;
            let iterations = settings.max_iterations;
            if !tolerance.is_finite() || tolerance <= 0. || iterations == 0 {
                return invalid("scalar weight controls must be positive");
            }
            let project = |w: &mut [f64]| self.project(w);
            if settings.strategy == "joint" {
                Some(projected_lbfgs(
                    weights,
                    8,
                    iterations,
                    tolerance,
                    1e-12,
                    |w| {
                        let (e, g, _) = evaluate(w)?;
                        Ok((e, g))
                    },
                    project,
                )?)
            } else if matches!(settings.strategy.as_str(), "one-pass" | "cyclic") {
                let mut current = weights.to_vec();
                self.joint_weights(&current)?;
                let mut evaluations = 0;
                let mut result = None;
                let cycles = if settings.strategy == "one-pass" {
                    1
                } else {
                    iterations
                };
                for cycle in 0..cycles {
                    let mut offset = 0;
                    for site in &self.sites {
                        let end = offset + site.candidates.len();
                        let base = current.clone();
                        let block = projected_lbfgs(
                            &current[offset..end],
                            8,
                            iterations,
                            tolerance,
                            1e-12,
                            |w| {
                                let mut full = base.clone();
                                full[offset..end].copy_from_slice(w);
                                let (e, g, _) = evaluate(&full)?;
                                Ok((e, g[offset..end].to_vec()))
                            },
                            project_weights,
                        )?;
                        current[offset..end].copy_from_slice(&block.parameters);
                        evaluations += block.evaluations;
                        block_history.push(
                            Value::object()
                                .with("cycle", cycle)
                                .with("site", site.id.clone())
                                .with("result", optimizer_value(block)),
                        );
                        offset = end;
                    }
                    let (e, g, _) = evaluate(&current)?;
                    evaluations += 1;
                    let mut projected = current
                        .iter()
                        .zip(&g)
                        .map(|(w, g)| w - g)
                        .collect::<Vec<_>>();
                    self.project(&mut projected)?;
                    let residual = current
                        .iter()
                        .zip(projected)
                        .map(|(w, p)| (w - p).abs())
                        .fold(0., f64::max);
                    let converged = residual <= tolerance;
                    result = Some(OptimizerResult {
                        parameters: current.clone(),
                        converged,
                        iterations: cycle + 1,
                        evaluations,
                        objective: e,
                        projected_gradient_max: residual,
                    });
                    if converged {
                        break;
                    }
                }
                result
            } else {
                return invalid("scalar weight strategy must be joint, one-pass or cyclic");
            }
        } else {
            None
        };
        let final_weights = optimizer
            .as_ref()
            .map_or(weights, |o| o.parameters.as_slice());
        let (objective, gradient, reports) = evaluate(final_weights)?;
        let mut offset = 0;
        let mut independent = Vec::new();
        let mut site_reports = Vec::new();
        for site in &self.sites {
            let end = offset + site.candidates.len();
            independent.extend_from_slice(&gradient[offset + 1..end]);
            site_reports.push(
                Value::object()
                    .with("id", site.id.clone())
                    .with("candidates", site.candidates.clone())
                    .with("weights", final_weights[offset..end].to_vec()),
            );
            offset = end;
        }
        let mut report=Value::object().with("feature","continuous-species").with("model",VERSION).with("functional",self.functional.description.clone()).with("model_assumptions","Experimental full frozen joint-endpoint scalar evaluators mapped into fixed common active AO spaces; product site weights retain intersite interactions; endpoint count grows as the product of candidate counts; semilocal RKS; current chemical acceptance is PBE-only; clamped geometry; no frozen bath polarization; not a compressed potential").with("endpoint_ids",self.ids.clone()).with("sites",site_reports).with("weights",final_weights.to_vec()).with("joint_weights",self.joint_weights(final_weights)?.0).with("energy",objective).with("weight_gradient",independent).with("reference_residuals",self.reference_residuals.clone()).with("states",reports).with("weight_strategy",strategy).with("block_history",block_history).with("weight_optimization",optimizer.map(optimizer_value));
        report.set("scf_energy_convention","Each endpoint removes its state-common reference inside compensated scalar accumulation before rounding; reaction coefficients cancel those references. Physical electronic/nuclear components are reported unshifted.");
        report.set(
            "construction_records",
            self.value.get("references").cloned().unwrap(),
        );
        report.set(
            "scf_controls",
            Value::object()
                .with("energy_tolerance", options.e_tol)
                .with("density_tolerance", options.p_tol)
                .with("orbital_tolerance", options.conv_tol_grad)
                .with("max_iter", options.max_iter),
        );
        report.set(
            "units",
            Value::object()
                .with("weights", "dimensionless")
                .with("joint_weights", "dimensionless")
                .with("coefficient", "dimensionless")
                .with("energy", "hartree")
                .with("weight_gradient", "hartree")
                .with("reference_residuals", "hartree")
                .with("orbital_residual", "hartree")
                .with("energy_without_internal_reference", "hartree")
                .with("reference_energy", "hartree")
                .with("electronic_energy", "hartree")
                .with("nuclear_energy", "hartree")
                .with("term_energies", "hartree")
                .with("scf_orbital_residual", "hartree")
                .with("particle_number_error", "dimensionless")
                .with("idempotency_error", "dimensionless")
                .with("objective", "hartree")
                .with("projected_gradient", "hartree")
                .with("energy_tolerance", "hartree")
                .with("density_tolerance", "dimensionless")
                .with("orbital_tolerance", "hartree"),
        );
        Ok(report)
    }
    fn project(&self, weights: &mut [f64]) -> Result<usize> {
        if weights.len() != self.sites.iter().map(|s| s.candidates.len()).sum::<usize>() {
            return invalid("scalar weights have wrong site layout");
        }
        let mut offset = 0;
        let mut active = 0;
        for site in &self.sites {
            let end = offset + site.candidates.len();
            active += project_weights(&mut weights[offset..end])?;
            offset = end;
        }
        Ok(active)
    }
    fn joint_weights(&self, weights: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        if weights.len() != self.sites.iter().map(|s| s.candidates.len()).sum::<usize>() {
            return invalid("scalar weights have wrong site layout");
        }
        let mut offsets = Vec::new();
        let mut offset = 0;
        for site in &self.sites {
            let end = offset + site.candidates.len();
            let block = &weights[offset..end];
            if block.iter().any(|x| !x.is_finite() || *x < 0.)
                || (block.iter().sum::<f64>() - 1.).abs() > 1e-12
            {
                return invalid("each scalar site requires finite simplex weights");
            }
            offsets.push(offset);
            offset = end;
        }
        // Each site uses q_i=w_i (i>0), w_0=1-sum(q). The zero baseline
        // gradient is a representative of the same covector on its simplex.
        // Product-rule factors from every other site retain cross interactions.
        let mut joint = Vec::new();
        let mut chain = vec![vec![0.; self.ids.len()]; weights.len()];
        for (endpoint, tuple) in self.choices.iter().enumerate() {
            joint.push(
                tuple
                    .iter()
                    .enumerate()
                    .map(|(s, c)| weights[offsets[s] + c])
                    .product(),
            );
            for (site, candidate) in tuple.iter().enumerate() {
                let others = tuple
                    .iter()
                    .enumerate()
                    .filter(|(s, _)| *s != site)
                    .map(|(s, c)| weights[offsets[s] + c])
                    .product::<f64>();
                for c in 1..self.sites[site].candidates.len() {
                    chain[offsets[site] + c][endpoint] = others
                        * ((if *candidate == c { 1. } else { 0. })
                            - (if *candidate == 0 { 1. } else { 0. }));
                }
            }
        }
        Ok((joint, chain))
    }
}

fn optimizer_value(o: OptimizerResult) -> Value {
    Value::object()
        .with("converged", o.converged)
        .with("iterations", o.iterations)
        .with("evaluations", o.evaluations)
        .with("objective", o.objective)
        .with("projected_gradient", o.projected_gradient_max)
}
fn project_weights(weights: &mut [f64]) -> Result<usize> {
    if weights.is_empty() || weights.iter().any(|x| !x.is_finite()) {
        return invalid("finite nonempty simplex weights required");
    }
    let mut sorted = weights.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.;
    let mut threshold = 0.;
    for (i, x) in sorted.iter().enumerate() {
        sum += x;
        let t = (sum - 1.) / (i + 1) as f64;
        if *x > t {
            threshold = t;
        }
    }
    for x in weights.iter_mut() {
        *x = (*x - threshold).max(0.);
    }
    let sum = weights.iter().sum::<f64>();
    let largest = (0..weights.len())
        .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
        .unwrap();
    weights[largest] += 1. - sum;
    if weights[largest] < 0. {
        return invalid("simplex projection normalization failed");
    }
    Ok(weights.iter().filter(|x| **x == 0.).count())
}

pub fn run(system: &System, controls: &Controls) -> Result<Value> {
    let mode = controls.str("fragment_mode", "")?;
    let record = ScalarActionRecord::from_value(&object_control(controls, "fragment_model_json")?)?;
    if matches!(
        mode.as_str(),
        "scalar-gradient" | "scalar-geometry-optimize" | "scalar-hessian"
    ) {
        return geometry::run(&record, system, controls, &mode);
    }
    let reference = record.primary_molecule();
    if system.is_periodic()
        || system.numbers.len() != reference.len()
        || system.positions.len() != reference.len()
        || system
            .numbers
            .iter()
            .zip(&reference.atoms)
            .any(|(z, a)| *z != a.z)
        || system.positions.iter().zip(&reference.atoms).any(|(r, a)| {
            r.iter().any(|v| !v.is_finite())
                || (Vec3::new(r[0], r[1], r[2]) - a.position).norm() > 1e-10
        })
    {
        return invalid("scalar weight modes require the saved first-state raw geometry and integer labels; moving geometry requires an explicit fixed endpoint and transport contract");
    }
    let weights = controls.opt_f64_list("fragment_weights")?.ok_or_else(|| {
        DftError::InvalidInput(
            "fragment_weights must explicitly name every saved endpoint weight".into(),
        )
    })?;
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: controls.usize("max_iter", 600)?,
        e_tol: controls.f64("conv_tol", 1e-14)?,
        p_tol: controls.f64("conv_tol_dp", 1e-13)?,
        conv_tol_grad: controls.f64("conv_tol_grad", 1e-12)?,
        eri_mode: EriMode::InCore { limit_mb: 128 },
        ..Default::default()
    };
    let optimize = match mode.as_str() {
        "scalar-evaluate" => None,
        "scalar-optimize" => Some(ScalarOptimization {
            tolerance: controls.f64("fragment_weight_tolerance", 1e-4)?,
            max_iterations: controls.usize("fragment_weight_max_iter", 160)?,
            strategy: controls.str("fragment_weight_strategy", "joint")?,
        }),
        _ => return invalid("unknown saved scalar action mode"),
    };
    controls.finish("continuous-species")?;
    record.calculate(&weights, &options, optimize)
}
