// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Fixed-endpoint geometry in the saved scalar's explicit spin sector.
use super::*;
use crate::optimizer::{optimize_with_evaluator, OptOptions};
use crate::pseudo_sub::scalar_geometry::{
    AffineFrozenGeometry, AffineFrozenHessian, CollinearAffineFrozenGeometry,
};
use crate::response::krylov::ResponseSolverConfig;
use crate::scf::Reference;

enum GeometryModel<'a> {
    Restricted(AffineFrozenGeometry<'a>),
    Collinear(CollinearAffineFrozenGeometry<'a>),
}
struct GeometryValue {
    energy: f64,
    gradient: Vec<Vec3>,
    fields: Value,
}
impl GeometryModel<'_> {
    fn evaluate(&self, r: &[Vec3]) -> Result<GeometryValue> {
        // Keep legacy scalar diagnostics for paired RKS and explicit two-spin
        // diagnostics for UKS. Both branches delegate to the same scalar core.
        match self {
            Self::Restricted(model) => {
                let v = model.evaluate(r, None)?;
                let s = &v.active_scf;
                let e = &s.evaluation;
                Ok(GeometryValue {
                    energy: v.energy,
                    gradient: v.gradient,
                    fields: Value::object()
                        .with(
                            "electronic_energy",
                            e.one_electron_energy + e.hartree_energy + e.xc_energy,
                        )
                        .with("nuclear_energy", e.nuclear_energy)
                        .with(
                            "full_positions",
                            v.full_molecule
                                .atoms
                                .iter()
                                .map(|a| a.position.to_array())
                                .collect::<Vec<_>>(),
                        )
                        .with("scf_orbital_residual", s.projected_orbital_residual)
                        .with("scf_iterations", s.history.len())
                        .with("particle_number_error", s.particle_number_error)
                        .with("idempotency_error", s.active_idempotency_error)
                        .with("chart_metric_error", v.chart_metric_error)
                        .with("frozen_overlap", v.frozen_overlap),
                })
            }
            Self::Collinear(model) => {
                let v = model.evaluate(r, None)?;
                let s = &v.active_scf;
                let e = &s.evaluation;
                Ok(GeometryValue {
                    energy: v.energy,
                    gradient: v.gradient,
                    fields: Value::object()
                        .with(
                            "electronic_energy",
                            e.one_electron_energy + e.hartree_energy + e.xc_energy,
                        )
                        .with("nuclear_energy", e.nuclear_energy)
                        .with(
                            "full_positions",
                            v.full_molecule
                                .atoms
                                .iter()
                                .map(|a| a.position.to_array())
                                .collect::<Vec<_>>(),
                        )
                        .with("scf_orbital_residual", s.projected_orbital_residual)
                        .with("scf_iterations", s.history.len())
                        .with("particle_number_error", s.particle_number_error)
                        .with("idempotency_error", s.active_idempotency_error)
                        .with("chart_metric_error", v.chart_metric_error)
                        .with("frozen_overlap", v.frozen_overlap),
                })
            }
        }
    }
    fn hessian(&self, r: &[Vec3], config: &ResponseSolverConfig) -> Result<AffineFrozenHessian> {
        match self {
            Self::Restricted(m) => m.hessian(r, config),
            Self::Collinear(m) => m.hessian(r, config),
        }
    }
}
enum Calculation<'a> {
    Evaluate,
    Optimize(&'a OptOptions),
    Hessian(&'a ResponseSolverConfig),
}

impl ScalarActionRecord {
    /// Analytic geometry for one endpoint and state, never weight optimization.
    pub fn calculate_geometry(
        &self,
        state_id: &str,
        endpoint_id: &str,
        coordinates: &[Vec3],
        options: &ScfOptions,
        optimize: Option<&OptOptions>,
    ) -> Result<Value> {
        self.geometry_calculation(
            state_id,
            endpoint_id,
            coordinates,
            options,
            optimize.map_or(Calculation::Evaluate, Calculation::Optimize),
        )
    }
    /// Model-consistent analytic curvature, including frozen metric and active
    /// spin response. No ordinary all-atom curvature or thermal model is used.
    pub fn calculate_hessian(
        &self,
        state_id: &str,
        endpoint_id: &str,
        coordinates: &[Vec3],
        options: &ScfOptions,
        response: &ResponseSolverConfig,
    ) -> Result<Value> {
        self.geometry_calculation(
            state_id,
            endpoint_id,
            coordinates,
            options,
            Calculation::Hessian(response),
        )
    }
    fn geometry_calculation(
        &self,
        state_id: &str,
        endpoint_id: &str,
        coordinates: &[Vec3],
        options: &ScfOptions,
        calculation: Calculation<'_>,
    ) -> Result<Value> {
        let state = self
            .states
            .iter()
            .find(|s| s.id == state_id)
            .ok_or_else(|| DftError::InvalidInput("unknown saved scalar geometry state".into()))?;
        let index = self
            .ids
            .iter()
            .position(|id| id == endpoint_id)
            .ok_or_else(|| {
                DftError::InvalidInput("unknown saved scalar geometry endpoint".into())
            })?;
        let endpoint = &state.endpoints[index];
        let owners = endpoint.geometry_atom_owners.as_deref().ok_or_else(|| DftError::Unsupported {
            feature: "saved scalar moving geometry".into(),
            missing: "explicit fixed-internal-affine-v1 geometry_atom_owners in the selected endpoint".into(),
        })?;
        if coordinates.len() != state.molecule.len()
            || coordinates
                .iter()
                .any(|r| r.to_array().iter().any(|v| !v.is_finite()))
        {
            return invalid("scalar geometry coordinates must match the saved state raw centers");
        }
        let grid = GridOptions {
            level: self.grid_level,
            weight_cutoff: 0.,
            ..Default::default()
        };
        let model = match &endpoint.frozen {
            FrozenSpace::Restricted { orbitals, .. } => {
                GeometryModel::Restricted(AffineFrozenGeometry {
                    full_reference: &endpoint.molecule,
                    full_basis: &endpoint.basis,
                    raw_reference: &state.molecule,
                    raw_basis: &state.basis,
                    frozen_reference: orbitals,
                    full_atom_owners: owners,
                    functional: &self.functional,
                    options,
                    grid_options: &grid,
                    active_electrons: state.active_electrons,
                    reference_energy: self.reference_energies[index],
                })
            }
            FrozenSpace::Collinear { orbitals } => {
                GeometryModel::Collinear(CollinearAffineFrozenGeometry {
                    full_reference: &endpoint.molecule,
                    full_basis: &endpoint.basis,
                    raw_reference: &state.molecule,
                    raw_basis: &state.basis,
                    frozen_reference: [&orbitals[0], &orbitals[1]],
                    full_atom_owners: owners,
                    functional: &self.functional,
                    options,
                    grid_options: &grid,
                    active_electrons: state.active_spin_electrons,
                    reference_energy: self.reference_energies[index],
                })
            }
        };
        let mut raw = state.molecule.clone();
        for (a, r) in raw.atoms.iter_mut().zip(coordinates) {
            a.position = *r;
        }
        let mut optimization = None;
        let mut converged = true;
        let evaluated = if let Calculation::Optimize(opt) = &calculation {
            let result = optimize_with_evaluator(&raw, opt, |current| {
                let r = current.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
                let v = model.evaluate(&r)?;
                Ok((v.energy, v.gradient.clone(), v))
            })?;
            converged = result.converged;
            optimization = Some(
                Value::object()
                    .with("converged", result.converged)
                    .with("iterations", result.iterations)
                    .with("gradient_tolerance", opt.gtol)
                    .with(
                        "maximum_gradient",
                        result.trajectory.last().unwrap().grad_max,
                    )
                    .with(
                        "trajectory",
                        result
                            .trajectory
                            .iter()
                            .map(|f| {
                                Value::object()
                                    .with("step", f.step)
                                    .with("energy_without_internal_reference", f.energy)
                                    .with("maximum_gradient", f.grad_max)
                            })
                            .collect::<Vec<_>>(),
                    ),
            );
            raw = result.molecule;
            result.scf
        } else {
            model.evaluate(coordinates)?
        };
        let mut units = Value::object();
        for (name, unit) in [
            ("energy", "hartree"),
            ("energy_without_internal_reference", "hartree"),
            ("reference_energy", "hartree"),
            ("electronic_energy", "hartree"),
            ("nuclear_energy", "hartree"),
            ("gradient", "hartree/bohr"),
            ("forces", "hartree/bohr"),
            ("positions", "bohr"),
            ("full_positions", "bohr"),
            ("geometry_atom_owners", "dimensionless"),
            ("active_spin_electrons", "dimensionless"),
            ("scf_orbital_residual", "hartree"),
            ("particle_number_error", "dimensionless"),
            ("idempotency_error", "dimensionless"),
            ("chart_metric_error", "dimensionless"),
            ("frozen_overlap", "dimensionless"),
            ("gradient_tolerance", "hartree/bohr"),
            ("maximum_gradient", "hartree/bohr"),
        ] {
            units.set(name, unit);
        }
        let mut report = evaluated.fields.with("feature", "continuous-species")
            .with("model", if self.unrestricted { COLLINEAR_VERSION } else { VERSION })
            .with("reference", if self.unrestricted { "uks" } else { "rks" }).with("converged", converged)
            .with("warnings", if converged { Vec::<String>::new() } else { vec!["The bounded affine geometry optimization did not meet its nuclear gradient tolerance".into()] })
            .with("geometry_transport", "fixed-internal-affine-v1").with("state", state_id).with("endpoint", endpoint_id)
            .with("functional", self.functional.description.clone())
            .with("model_assumptions", "One saved endpoint and spin sector; full endpoint integrals; frozen reactant coefficient spans normalized in the current metric; full atoms translate with declared raw owners and fixed laboratory internal offsets; no freely rotating fragment frame, internal deformation or mixed-weight geometry; constrained model curvature does not imply full molecular RRHO thermochemistry")
            .with("energy", evaluated.energy + self.reference_energies[index]).with("energy_without_internal_reference", evaluated.energy)
            .with("reference_energy", self.reference_energies[index])
            .with("gradient", evaluated.gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>())
            .with("forces", evaluated.gradient.iter().map(|g| (-*g).to_array()).collect::<Vec<_>>())
            .with("positions", raw.atoms.iter().map(|a| a.position.to_array()).collect::<Vec<_>>())
            .with("geometry_atom_owners", owners.to_vec()).with("active_electrons", state.active_electrons)
            .with("active_spin_electrons", state.active_spin_electrons).with("geometry_optimization", optimization)
            .with("scf_energy_convention", "The common reference is removed inside scalar accumulation before rounding; physical total energy restores it only for reporting");
        if let Calculation::Hessian(config) = calculation {
            let v = model.hessian(coordinates, config)?;
            report.set("hessian", v.hessian);
            report.set("raw_hessian", v.raw_hessian);
            report.set("raw_asymmetry", v.raw_asymmetry);
            report.set("reference_orbital_residual", v.reference_orbital_residual);
            report.set(
                "directional_response",
                v.directional_results
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        Value::object()
                            .with("direction", i)
                            .with("orbital_response_residual", d.orbital_response_residual)
                            .with("relative_response_residual", d.relative_response_residual)
                            .with("response_iterations", d.response_iterations)
                            .with("gradient_consistency_error", d.gradient_consistency_error)
                    })
                    .collect::<Vec<_>>(),
            );
            report.set("hessian_convention", "All ordered raw-coordinate entries are retained before symmetrization; directions list Cartesian unit vectors followed by pairs i<j. No rotational projection or molecular thermal model is applied.");
            for (name, unit) in [
                ("hessian", "hartree/bohr^2"),
                ("raw_hessian", "hartree/bohr^2"),
                ("raw_asymmetry", "hartree/bohr^2"),
                ("reference_orbital_residual", "hartree"),
                ("orbital_response_residual", "hartree/bohr"),
                ("relative_response_residual", "dimensionless"),
                ("gradient_consistency_error", "hartree/bohr"),
                ("response_iterations", "dimensionless"),
                ("direction", "dimensionless"),
            ] {
                units.set(name, unit);
            }
        }
        report.set("units", units);
        Ok(report)
    }
}

pub(super) fn run(
    record: &ScalarActionRecord,
    system: &System,
    controls: &Controls,
    mode: &str,
) -> Result<Value> {
    let state_id = controls.str("fragment_state", "")?;
    let endpoint_id = controls.str("fragment_endpoint", "")?;
    if state_id.is_empty() || endpoint_id.is_empty() {
        return invalid(
            "scalar geometry requires explicit fragment_state and fragment_endpoint identities",
        );
    }
    let state = record
        .states
        .iter()
        .find(|s| s.id == state_id)
        .ok_or_else(|| DftError::InvalidInput("unknown scalar state".into()))?;
    if system.is_periodic()
        || system.numbers.len() != state.molecule.len()
        || system
            .numbers
            .iter()
            .zip(&state.molecule.atoms)
            .any(|(z, a)| *z != a.z)
        || system.positions.len() != state.molecule.len()
    {
        return invalid("scalar geometry requires the selected state's isolated raw atom labels and coordinate count");
    }
    let options = ScfOptions {
        reference: if record.unrestricted {
            Reference::Uhf
        } else {
            Reference::Rhf
        },
        guess: Guess::Core,
        max_iter: controls.usize("max_iter", 600)?,
        e_tol: controls.f64("conv_tol", 1e-14)?,
        p_tol: controls.f64("conv_tol_dp", 1e-13)?,
        conv_tol_grad: controls.f64("conv_tol_grad", 1e-12)?,
        eri_mode: EriMode::InCore { limit_mb: 128 },
        ..Default::default()
    };
    let opt = if mode == "scalar-geometry-optimize" {
        Some(OptOptions {
            max_steps: controls.usize("max_steps", 200)?,
            gtol: controls.f64("gtol", 1e-5)?,
            step_max: controls.f64("step_max", 0.2)?,
            ..Default::default()
        })
    } else {
        None
    };
    let response = if mode == "scalar-hessian" {
        Some(crate::driver::vibrational::response_config(controls)?)
    } else {
        None
    };
    // Moving mixtures need the derivative of their coupled scalar. Weight
    // controls must not be silently consumed by a fixed-endpoint calculation.
    controls.finish("continuous-species affine geometry")?;
    let r = system
        .positions
        .iter()
        .map(|r| Vec3::new(r[0], r[1], r[2]))
        .collect::<Vec<_>>();
    match response {
        Some(c) => record.calculate_hessian(&state_id, &endpoint_id, &r, &options, &c),
        None => record.calculate_geometry(&state_id, &endpoint_id, &r, &options, opt.as_ref()),
    }
}
