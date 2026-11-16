// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common periodic double-hybrid driver. All derivatives solve the parent
//! orbitals and the noncanonical pair equations through the requested order.
use super::molecular::units;
use super::periodic::{
    prepare_periodic, reference_report, run_reference, stress_fields, supercell_system,
    PeriodicMethod, STRESS_UNITS,
};
use super::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::dispersion::{D3Damping, PeriodicDispersion};
use crate::periodic::stress::PeriodicStress;
use crate::response::mixed::{polarized_mixed, DirectionalCoefficients};
use crate::response::nuclear::{LatticeCorrelatedSolver, LatticeCorrelatedTaylor, LatticePath};
use crate::response::ResponseSolverConfig;
use crate::xc::double_hybrid::{DoubleHybridFunctional, Pt2Coefficients};
use crate::{DftError, Result};

struct Paths<'a, 'b> {
    solver: &'a LatticeCorrelatedSolver<'b>,
    config: &'a ResponseSolverConfig,
    primitive: usize,
    replicas: usize,
    cache: DirectionalCoefficients,
    orbital: Vec<f64>,
    amplitude: Vec<f64>,
    projector: Vec<f64>,
    stationarity: Vec<f64>,
    metric: Vec<f64>,
    number: Vec<f64>,
    frame: Vec<f64>,
}

impl Paths<'_, '_> {
    fn record(&mut self, value: &LatticeCorrelatedTaylor) {
        self.orbital.extend(&value.response_residuals);
        self.amplitude.extend(&value.amplitude_residuals);
        self.projector.extend(&value.projector_residuals);
        self.stationarity.extend(&value.stationarity_residuals);
        self.metric.extend(&value.metric_residuals);
        self.number.extend(value.number_residuals.iter().flatten());
        self.frame.push(value.orbital_frame_residual);
    }
    fn mixed(&mut self, directions: &[Vec<f64>]) -> Result<f64> {
        // Nuclear and field Taylor coefficients are homogeneous in the linear
        // path. Reuse the common signed-scale cache, including odd orders;
        // every recorded residual still comes from an actual normalized solve.
        // Temporarily move the cache so the callback can also record diagnostics.
        let mut cache = std::mem::take(&mut self.cache);
        let result = polarized_mixed(3 * self.primitive + 3, directions, |direction, order| {
            cache.get(direction, order, 1, |direction, order| {
                let n = 3 * self.primitive;
                let displacements: Vec<_> = (0..self.replicas)
                    .flat_map(|_| direction[..n].chunks_exact(3).map(|d| [d[0], d[1], d[2]]))
                    .collect();
                let moving = displacements.iter().flatten().any(|v| *v != 0.0);
                let field = [direction[n], direction[n + 1], direction[n + 2]];
                let path = LatticePath {
                    displacement: if moving {
                        vec![displacements]
                    } else {
                        Vec::new()
                    },
                    field: vec![field],
                    ..Default::default()
                };
                let result = self.solver.solve(&path, order, self.config)?;
                let value = result.energy[order];
                self.record(&result);
                Ok(vec![value])
            })
        });
        // Preserve completed paths even when a later response solve fails.
        self.cache = cache;
        result.map(|result| result.value[0])
    }
    fn stress(
        &mut self,
        volume: f64,
        coefficients: Pt2Coefficients,
        cartesian_field: [f64; 3],
    ) -> Result<PeriodicStress> {
        let mut terms = vec![
            ("target_density_functional", [[0.0; 3]; 3]),
            ("pt2_same_spin_alpha", [[0.0; 3]; 3]),
            ("pt2_opposite_spin", [[0.0; 3]; 3]),
            ("pt2_same_spin_beta", [[0.0; 3]; 3]),
        ];
        // Affinely deform the nuclei and every image at fixed reduced mesh.
        // The same stationary parent orbitals and PT2 pair equations used
        // for the energy supply each ordered first derivative.
        for a in 0..3 {
            for b in 0..3 {
                let mut affine = [[0.0; 3]; 3];
                affine[a][b] = 1.0;
                // E · F(t) a_i = (F(t)^T E) · a_i. Supply the first
                // coefficient in the reference frame used by the Berry path.
                let mut field_tangent = [0.0; 3];
                field_tangent[b] = cartesian_field[a];
                let result = self.solver.solve(
                    &LatticePath {
                        deformation: vec![affine],
                        field: vec![field_tangent],
                        ..Default::default()
                    },
                    1,
                    self.config,
                )?;
                for ((_, tensor), value) in terms.iter_mut().zip([
                    result.target[1],
                    coefficients.same_spin * result.same_spin_alpha[1],
                    coefficients.opposite_spin * result.opposite_spin[1],
                    coefficients.same_spin * result.same_spin_beta[1],
                ]) {
                    tensor[a][b] = value;
                }
                self.record(&result);
            }
        }
        Ok(PeriodicStress::from_terms(volume, terms))
    }

    fn report(&self, report: &mut Value, labels: &mut Value, affine: bool) {
        for (key, values, unit) in [
            ("response_residuals", &self.orbital, "dimensionless"),
            (
                "amplitude_response_residuals",
                &self.amplitude,
                "hartree along dimensionless paths",
            ),
            ("projector_residuals", &self.projector, "dimensionless"),
            (
                "stationarity_residuals",
                &self.stationarity,
                "hartree along dimensionless paths",
            ),
            ("metric_residuals", &self.metric, "dimensionless"),
            ("response_number_residuals", &self.number, "electrons"),
            ("orbital_frame_residuals", &self.frame, "dimensionless"),
        ] {
            report.set(key, values.clone());
            labels.set(key, unit);
        }
        report.set(
            "path_evaluations",
            self.cache.evaluations + 9 * usize::from(affine),
        );
    }
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    calculate(id, system, controls, None).map(|(report, _)| report)
}

pub(super) fn supercell_force_constants(
    feature: &str,
    system: &System,
    controls: &Controls,
    supercell: [usize; 3],
) -> Result<(Value, Matrix)> {
    let (report, hessian) = calculate("phonons", system, controls, Some((feature, supercell)))?;
    Ok((
        report,
        hessian.expect("phonon calculation produces a Hessian"),
    ))
}

fn calculate(
    id: &str,
    system: &System,
    controls: &Controls,
    spectral_supercell: Option<(&str, [usize; 3])>,
) -> Result<(Value, Option<Matrix>)> {
    if ![
        "periodic-energy",
        "stress",
        "periodic-efield",
        "periodic-beta",
        "periodic-gamma",
        "dielectric-tensor",
        "born-charges",
        "phonons",
    ]
    .contains(&id)
    {
        return Err(DftError::Unsupported {
            feature: format!("correlated periodic {id}"),
            missing: "the correlated observable contraction on the full periodic orbital/pair path"
                .into(),
        });
    }
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let primitive_charge = method.charge;
    // The spectral supercell defines the independent nuclear coordinates. The
    // remaining k mesh is folded into a larger electronic Born–von Karman cell
    // in exactly the same way as for the ordinary correlated Gamma Hessian.
    let spectral_system;
    let system = if let Some((_, supercell)) = spectral_supercell {
        (spectral_system, method) = supercell_system(system, &method, supercell)?;
        &spectral_system
    } else {
        system
    };
    let named = DoubleHybridFunctional::named(&method.xc);
    let default_model = if named.is_some() {
        method.xc.as_str()
    } else {
        "custom"
    };
    let model = controls.str("double_hybrid_model", default_model)?;
    let recipe = DoubleHybridFunctional::named(&model);
    let ss = controls.opt_f64("pt2_same_spin")?;
    let os = controls.opt_f64("pt2_opposite_spin")?;
    let memory = controls.usize("pt2_memory_mb", 2048)?;
    if recipe.is_none() && (model != "custom" || ss.is_none() || os.is_none()) {
        return Err(DftError::InvalidInput(
            "a custom periodic double hybrid needs both PT2 spin coefficients".into(),
        ));
    }
    let mut dispersion = None;
    if let Some(recipe) = &recipe {
        if named
            .as_ref()
            .is_some_and(|f| f.parent.description != recipe.parent.description)
            || (named.is_none() && method.xc != "pbe")
        {
            return Err(DftError::InvalidInput("a named periodic double hybrid fixes its parent; use double_hybrid_model=custom for another parent".into()));
        }
        let dispersion_input = controls
            .opt_str("dispersion")?
            .or(controls.opt_str("disp")?);
        if dispersion_input.as_deref().is_none_or(|d| d == "auto") {
            dispersion = recipe.dispersion.map(|p| PeriodicDispersion {
                damping: D3Damping::Bj(p),
                cn_cutoff: method.cn_cutoff,
                disp_cutoff: method.disp_cutoff,
                three_body: None,
            });
        } else if dispersion_input.as_deref() != Some("none") {
            return Err(DftError::InvalidInput(
                "a named periodic double hybrid fixes its dispersion recipe".into(),
            ));
        }
        method.xc = "tpss".into();
        method.dispersion = crate::dispersion::Dispersion::None;
    } else {
        dispersion = method.dispersion_model()?;
    }
    if method.electronic_temperature != 0.0 {
        return Err(DftError::InvalidInput(
            "unregularized periodic PT2 requires integer zero-temperature occupations".into(),
        ));
    }
    let field = controls
        .opt_vec3("field")?
        .unwrap_or_else(Vec3::zero)
        .to_array();
    let forces = if ["periodic-efield", "periodic-energy", "stress"].contains(&id) {
        controls.bool("forces", id == "stress")?
    } else {
        false
    };
    let with_stress = if matches!(id, "periodic-energy" | "stress") {
        controls.bool("stress", false)? || id == "stress"
    } else {
        false
    };
    let order = if id == "periodic-efield" {
        controls.usize("response_order", 0)?
    } else {
        0
    };
    let direction = if id == "periodic-efield" {
        controls
            .opt_vec3("field_direction")?
            .unwrap_or(Vec3::new(0., 0., 1.))
            .to_array()
    } else {
        [0.; 3]
    };
    let use_symmetry = if id == "periodic-gamma" {
        controls.bool("use_symmetry", true)?
    } else {
        true
    };
    let relaxed = if ["periodic-beta", "periodic-gamma"].contains(&id) {
        controls.bool("relaxed_ion", false)?
    } else {
        false
    };
    if relaxed {
        return Err(DftError::Unsupported {
            feature: "correlated relaxed-ion electric response".into(),
            missing: "stationary correlated ionic geometry and internal-coordinate response".into(),
        });
    }
    let config = super::vibrational::response_config(controls)?;
    controls.finish(spectral_supercell.map_or(id, |(feature, _)| feature))?;
    let mesh = method.k_mesh;
    let replicas = mesh.iter().product();
    let (super_system, super_method) = supercell_system(system, &method, mesh)?;
    let mut setup = prepare_periodic(&super_system, &super_method)?;
    let mut target_nonlocal = None;
    let (target, coefficients) = if let Some(recipe) = recipe {
        // Required nonlocal pieces are checked by the physical SCF/path
        // providers. Never omit a recipe's defining nonlocal contribution.
        if let Some(parameters) = recipe.reference_vv10 {
            if (controls.contains("vv10_b")
                && setup
                    .options
                    .vv10
                    .is_some_and(|v| v.parameters.b != parameters.b))
                || (controls.contains("vv10_c")
                    && setup
                        .options
                        .vv10
                        .is_some_and(|v| v.parameters.c != parameters.c))
            {
                return Err(DftError::InvalidInput(
                    "a named double hybrid fixes its orbital-reference nonlocal parameters".into(),
                ));
            }
            setup.options.vv10 = Some(crate::periodic::nonlocal::PeriodicVv10Options {
                parameters,
                cutoff: method.vv10_cutoff,
            });
        }
        target_nonlocal = recipe.target_vv10.map(|(coefficient, parameters)| {
            (
                coefficient,
                crate::periodic::nonlocal::PeriodicVv10Options {
                    parameters,
                    cutoff: method.vv10_cutoff,
                },
            )
        });
        setup.functional = recipe.parent;
        (recipe.target, recipe.correlation)
    } else {
        (
            None,
            Pt2Coefficients {
                same_spin: ss.unwrap(),
                opposite_spin: os.unwrap(),
            },
        )
    };
    let coefficients = Pt2Coefficients {
        same_spin: ss.unwrap_or(coefficients.same_spin),
        opposite_spin: os.unwrap_or(coefficients.opposite_spin),
    };
    let scf = run_reference(&setup)?;
    let solver = LatticeCorrelatedSolver::new(
        &setup.cell,
        &setup.functional,
        target.as_ref(),
        &scf,
        &setup.options,
        replicas,
        field,
        coefficients,
        memory,
    )?;
    let solver = if let Some((coefficient, settings)) = target_nonlocal {
        solver.with_target_nonlocal(coefficient, settings)?
    } else {
        solver
    };
    let centre = solver.solve(&LatticePath::default(), 0, &config)?;
    let positions: Vec<_> = system
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let dispersion_value = dispersion
        .as_ref()
        .map(|d| {
            crate::periodic::dispersion::periodic_d3(
                &system.numbers,
                &positions,
                &super::periodic::lattice_of(system)?,
                d,
            )
        })
        .transpose()?;
    let dispersion_energy = dispersion_value.as_ref().map_or(0.0, |d| d.energy);
    let mut report = reference_report(&setup,&scf,
        "static periodic post-SCF double hybrid; full orbital and noncanonical PT2 amplitude response at every requested order; periodic Ewald Coulomb tensor with zero reciprocal component removed; Gamma Born-von Karman supercell sampling; integer occupations, no frozen core or denominator regularization; dimensionless Taylor path parameter");
    report.set("energy", centre.energy[0] + dispersion_energy);
    report.set("reference_energy", centre.target[0] + dispersion_energy);
    report.set("orbital_reference_energy", centre.parent[0]);
    report.set(
        "parent_nonlocal_correlation_energy",
        centre.parent_nonlocal[0],
    );
    report.set(
        "target_nonlocal_correlation_energy",
        centre.target_nonlocal[0],
    );
    report.set(
        "energy_components",
        Value::object()
            .with("target_density_functional", centre.target[0])
            .with(
                "pt2_same_spin_alpha",
                coefficients.same_spin * centre.same_spin_alpha[0],
            )
            .with(
                "pt2_opposite_spin",
                coefficients.opposite_spin * centre.opposite_spin[0],
            )
            .with(
                "pt2_same_spin_beta",
                coefficients.same_spin * centre.same_spin_beta[0],
            )
            .with("dispersion", dispersion_energy),
    );
    if let Some((coefficient, settings)) = target_nonlocal {
        report.set("target_nonlocal_coefficient", coefficient);
        report.set("target_nonlocal_b", settings.parameters.b);
        report.set("target_nonlocal_c", settings.parameters.c);
    }
    report.set("pt2_correlation", centre.correlation[0]);
    report.set("pt2_same_spin_alpha", centre.same_spin_alpha[0]);
    report.set("pt2_opposite_spin", centre.opposite_spin[0]);
    report.set("pt2_same_spin_beta", centre.same_spin_beta[0]);
    report.set("dispersion_energy", dispersion_energy);
    report.set("pt2_same_spin_coefficient", coefficients.same_spin);
    report.set("pt2_opposite_spin_coefficient", coefficients.opposite_spin);
    report.set("double_hybrid_model", model);
    report.set("supercell", mesh.to_vec());
    report.set("reference_field", field);
    report.set("series_reference_residual", centre.reference.residual);
    report.set("series_reference_energy", centre.reference.energy);
    report.set(
        "series_reference_energy_difference",
        centre.reference.energy_difference,
    );
    report.set("provenance", "analytic");
    let mut labels = units(&[
        ("energy", "hartree per primitive cell"),
        ("reference_energy", "hartree per primitive cell"),
        ("orbital_reference_energy", "hartree per primitive cell"),
        (
            "parent_nonlocal_correlation_energy",
            "hartree per primitive cell",
        ),
        (
            "target_nonlocal_correlation_energy",
            "hartree per primitive cell",
        ),
        ("target_nonlocal_coefficient", "dimensionless"),
        ("target_nonlocal_b", "dimensionless"),
        ("target_nonlocal_c", "dimensionless"),
        ("pt2_correlation", "hartree per primitive cell"),
        ("pt2_same_spin_alpha", "hartree per primitive cell"),
        ("pt2_opposite_spin", "hartree per primitive cell"),
        ("pt2_same_spin_beta", "hartree per primitive cell"),
        ("dispersion_energy", "hartree per primitive cell"),
        ("pt2_same_spin_coefficient", "dimensionless"),
        ("pt2_opposite_spin_coefficient", "dimensionless"),
        ("reference_field", "hartree/e/bohr"),
        ("series_reference_residual", "hartree"),
        ("series_reference_energy", "hartree per primitive cell"),
        (
            "series_reference_energy_difference",
            "hartree per primitive cell",
        ),
    ]);
    let n = 3 * system.numbers.len();
    let dim = n + 3;
    let axis = |i: usize| {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    };
    let mut paths = Paths {
        solver: &solver,
        config: &config,
        primitive: system.numbers.len(),
        replicas,
        cache: DirectionalCoefficients::default(),
        orbital: Vec::new(),
        amplitude: Vec::new(),
        projector: Vec::new(),
        stationarity: Vec::new(),
        metric: Vec::new(),
        number: Vec::new(),
        frame: Vec::new(),
    };
    paths.record(&centre);
    let volume = setup.cell.lattice().volume().abs() / replicas as f64;
    if primitive_charge != 0 {
        report.set("charge_compensation", "uniform 3D Ewald background");
        report.set("primitive_cell_charge", primitive_charge);
        report.set("born_von_karman_charge", setup.cell.net_charge());
        labels.set("primitive_cell_charge", "e");
        labels.set("born_von_karman_charge", "e");
    }
    if id == "periodic-efield" {
        let value = solver.solve(
            &LatticePath {
                field: vec![direction],
                ..Default::default()
            },
            order,
            &config,
        )?;
        paths.record(&value);
        for (key, values) in [
            ("field_energy_coefficients", value.energy),
            ("parent_energy_coefficients", value.parent),
            ("target_energy_coefficients", value.target),
            ("parent_nonlocal_coefficients", value.parent_nonlocal),
            ("target_nonlocal_coefficients", value.target_nonlocal),
            ("pt2_energy_coefficients", value.correlation),
            ("pt2_same_spin_alpha_coefficients", value.same_spin_alpha),
            ("pt2_opposite_spin_coefficients", value.opposite_spin),
            ("pt2_same_spin_beta_coefficients", value.same_spin_beta),
        ] {
            let mut values = values;
            values[0] += if [
                "field_energy_coefficients",
                "parent_energy_coefficients",
                "target_energy_coefficients",
            ]
            .contains(&key)
            {
                dispersion_energy
            } else {
                0.0
            };
            report.set(key, values);
            labels.set(key, "hartree per primitive cell along dimensionless path");
        }
        report.set("response_order", order);
        report.set("field_direction", direction);
        labels.set(
            "field_direction",
            "hartree/e/bohr per dimensionless path parameter",
        );
        report.set("field_energy", centre.energy[0] + dispersion_energy);
        labels.set("field_energy", "hartree per primitive cell");
    }
    if id == "periodic-efield"
        || [
            "periodic-beta",
            "periodic-gamma",
            "dielectric-tensor",
            "born-charges",
        ]
        .contains(&id)
    {
        let dipole: Vec<_> = (0..3)
            .map(|a| paths.mixed(&[axis(n + a)]).map(|e| -e))
            .collect::<Result<_>>()?;
        report.set("dipole", dipole.clone());
        labels.set("dipole", "e*bohr per primitive cell");
        report.set(
            "polarization",
            dipole.iter().map(|d| d / volume).collect::<Vec<_>>(),
        );
        labels.set("polarization", "e/bohr^2");
        report.set("volume", volume);
        labels.set("volume", "bohr^3 per primitive cell");
    }
    if [
        "periodic-beta",
        "periodic-gamma",
        "dielectric-tensor",
        "born-charges",
    ]
    .contains(&id)
    {
        let mut susceptibility = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in a..3 {
                let v = -paths.mixed(&[axis(n + a), axis(n + b)])? / volume;
                susceptibility[a][b] = v;
                susceptibility[b][a] = v;
            }
        }
        report.set("susceptibility", susceptibility);
        labels.set("susceptibility", "dimensionless");
        let epsilon = std::array::from_fn::<_, 3, _>(|a| {
            std::array::from_fn::<_, 3, _>(|b| {
                (if a == b { 1.0 } else { 0.0 }) + 4.0 * std::f64::consts::PI * susceptibility[a][b]
            })
        });
        report.set("dielectric_tensor", epsilon);
        labels.set("dielectric_tensor", "dimensionless");
        report.set("epsilon_infinity", epsilon);
        labels.set("epsilon_infinity", "dimensionless");
        report.set("raw_asymmetry", Value::Null);
    }
    if id == "periodic-beta" || id == "periodic-gamma" {
        let rank = if id == "periodic-beta" {
            3usize
        } else {
            4usize
        };
        let mut tensor = vec![0.0; 3usize.pow(rank as u32)];
        let mut independent: Vec<(Vec<usize>, f64)> = Vec::new();
        for (index, value) in tensor.iter_mut().enumerate() {
            let mut axes = vec![0; rank];
            let mut rest = index;
            for a in axes.iter_mut().rev() {
                *a = rest % 3;
                rest /= 3;
            }
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            if use_symmetry {
                if let Some((_, v)) = independent.iter().find(|(a, _)| a == &sorted) {
                    *value = *v;
                    continue;
                }
            }
            let directions: Vec<_> = axes.iter().map(|a| axis(n + *a)).collect();
            *value = -paths.mixed(&directions)? / volume;
            independent.push((sorted, *value));
        }
        let key = if rank == 3 {
            "second_susceptibility"
        } else {
            "third_susceptibility"
        };
        let unit = if rank == 3 {
            "e*bohr/hartree"
        } else {
            "e^2*bohr^2/hartree^2"
        };
        report.set(key, tensor.clone());
        labels.set(key, unit);
        let factorial = if rank == 3 { 2.0 } else { 6.0 };
        let chi = if rank == 3 { "chi2" } else { "chi3" };
        report.set(
            chi,
            tensor.iter().map(|v| v / factorial).collect::<Vec<_>>(),
        );
        labels.set(chi, unit);
        let mut defect = 0.0_f64;
        if !use_symmetry {
            for (index, value) in tensor.iter().enumerate() {
                let mut axes = vec![0; rank];
                let mut rest = index;
                for a in axes.iter_mut().rev() {
                    *a = rest % 3;
                    rest /= 3;
                }
                for j in 0..rank - 1 {
                    let mut other = axes.clone();
                    other.swap(j, j + 1);
                    let k = other.iter().fold(0, |v, a| 3 * v + a);
                    defect = defect.max((value - tensor[k]).abs());
                }
            }
        }
        report.set("tensor", tensor);
        labels.set("tensor", unit);
        report.set("tensor_shape", vec![3usize; rank]);
        report.set("polarization_derivative_order", rank - 1);
        report.set("response_order", rank);
        report.set("tensor_convention","flat row-major polarization and ordered field axes; actual static derivatives; full correlated orbital response through the energy derivative order");
        report.set(
            "raw_permutation_residual",
            if use_symmetry {
                Value::Null
            } else {
                Value::Float(defect)
            },
        );
        labels.set("raw_permutation_residual", unit);
        report.set("independent_components", independent.len());
        report.set("use_symmetry", use_symmetry);
    }
    if id == "born-charges" {
        let mut born = vec![vec![0.0; 3]; n];
        for i in 0..n {
            for a in 0..3 {
                born[i][a] = -paths.mixed(&[axis(i), axis(n + a)])?;
            }
        }
        let sum = std::array::from_fn::<_, 3, _>(|b| {
            std::array::from_fn::<_, 3, _>(|a| {
                (0..system.numbers.len())
                    .map(|i| born[3 * i + b][a])
                    .sum::<f64>()
            })
        });
        report.set("born_sum_rule", sum);
        labels.set("born_sum_rule", "e");
        report.set(
            "born_charges",
            (0..system.numbers.len())
                .map(|i| {
                    std::array::from_fn::<_, 3, _>(|a| {
                        std::array::from_fn::<_, 3, _>(|b| born[3 * i + b][a])
                    })
                })
                .collect::<Vec<_>>(),
        );
        labels.set("born_charges", "e axes atom, polarization, displacement");
    }
    if forces {
        let mut out = vec![[0.0; 3]; system.numbers.len()];
        for i in 0..n {
            out[i / 3][i % 3] = -paths.mixed(&[axis(i)])?;
            if let Some(d) = &dispersion_value {
                out[i / 3][i % 3] += d.forces[i / 3].to_array()[i % 3];
            }
        }
        report.set("forces", out);
        labels.set("forces", "hartree/bohr");
    }
    if with_stress {
        let mut stress = paths.stress(volume, coefficients, field)?;
        if let Some(d) = &dispersion_value {
            crate::periodic::dispersion::add_to_stress(&mut stress, &d.strain_derivative);
        }
        stress_fields(&mut report, &stress);
        report.set("volume", volume);
        report.set("orbital_residual", scf.orbital_residual);
        report.set("stress_provenance", "analytic");
        if field != [0.0; 3] {
            report.set(
                "stress_field_convention",
                "fixed Cartesian Berry electric field",
            );
        }
        for (name, label) in STRESS_UNITS {
            labels.set(name, label);
        }
        labels.set("strain_derivative_terms", "hartree per cell");
        labels.set("volume", "bohr^3");
        labels.set("orbital_residual", "hartree");
    }
    let mut spectral_hessian = None;
    if id == "phonons" {
        report.set("q", [0.0; 3]);
        labels.set("q", "1/bohr");
        let mut hessian = Matrix::zeros(n, n);
        for a in 0..n {
            for b in a..n {
                let v = paths.mixed(&[axis(a), axis(b)])?;
                hessian[(a, b)] = v;
                hessian[(b, a)] = v;
            }
        }
        if let Some(d) = &dispersion {
            let h = crate::periodic::dispersion::periodic_dispersion_hessian(
                &system.numbers,
                &positions,
                &super::periodic::lattice_of(system)?,
                d,
            )?;
            for (v, d) in hessian.as_mut_slice().iter_mut().zip(h.as_slice()) {
                *v += d;
            }
        }
        if let Some((_, supercell)) = spectral_supercell {
            let born_von_karman =
                std::array::from_fn::<_, 3, _>(|axis| supercell[axis] * mesh[axis]);
            report.set("born_von_karman_supercell", born_von_karman);
            report.set("supercell", supercell);
            report.set("supercell_k_mesh", mesh);
            labels.set("born_von_karman_supercell", "dimensionless");
            if primitive_charge != 0 {
                report.set("force_constant_supercell_charge", method.charge);
                labels.set("force_constant_supercell_charge", "e");
            }
            for key in [
                "energy",
                "reference_energy",
                "orbital_reference_energy",
                "parent_nonlocal_correlation_energy",
                "target_nonlocal_correlation_energy",
                "pt2_correlation",
                "pt2_same_spin_alpha",
                "pt2_opposite_spin",
                "pt2_same_spin_beta",
                "dispersion_energy",
                "energy_components",
                "series_reference_energy",
                "series_reference_energy_difference",
            ] {
                labels.set(key, "hartree per force-constant supercell");
            }
            spectral_hessian = Some(hessian);
        } else {
            report.set(
                "force_constants",
                (0..n)
                    .map(|i| (0..n).map(|j| hessian[(i, j)]).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            labels.set("force_constants", "hartree/bohr^2");
            report.set("raw_asymmetry", Value::Null);
            let mut acoustic = 0.0_f64;
            for i in 0..n {
                for a in 0..3 {
                    acoustic = acoustic.max(
                        (0..system.numbers.len())
                            .map(|j| hessian[(i, 3 * j + a)])
                            .sum::<f64>()
                            .abs(),
                    );
                }
            }
            report.set("acoustic_residual", acoustic);
            labels.set("acoustic_residual", "hartree/bohr^2");
            let (frequencies, modes) =
                crate::periodic::phonon::gamma_modes(&hessian, &system.numbers)?;
            report.set("frequencies_cm1", frequencies);
            labels.set("frequencies_cm1", "cm^-1");
            report.set("normal_modes_cartesian", modes);
            labels.set(
                "normal_modes_cartesian",
                "atomic units electron-mass-weighted displacement pattern",
            );
        }
    }
    paths.report(&mut report, &mut labels, with_stress);
    report.set("units", labels);
    Ok((report, spectral_hessian))
}
