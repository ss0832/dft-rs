// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electric response to Buckingham gradient modes or Racah solid harmonics.

use super::correlated_electric::spinor;
use super::double_hybrid::SpinorModel;
use super::method::{
    run_collinear_property_reference, CollinearPropertyReference, MolecularReference,
    MolecularSetup,
};
use super::molecular::{
    moment_operators, moment_origin, reference_report, setup, spherical_weights,
};
use super::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::{spinor_scf::SpinorScfReport, BlochMatrix};
use crate::properties::two_component::MolecularTwoComponent;
use crate::response::complex::Complex;
use crate::response::dynamic::{real_operator_polarizability, OperatorPolarizability};
use crate::response::electric_tensor::{
    quadrupole_modes as modes, quadrupole_weights, BUCKINGHAM_FACTOR,
};
use crate::response::{Frequency, ResponseSolverConfig};
use crate::{DftError, Result};

const ID: &str = "quadrupole-polarizability";
const MULTIPOLE_ID: &str = "multipole-polarizability";

#[derive(Clone, Copy)]
enum Convention {
    Buckingham,
    Racah(usize),
}

impl Convention {
    fn rank(self) -> usize {
        match self {
            Self::Buckingham => 2,
            Self::Racah(rank) => rank,
        }
    }

    fn scale(self) -> f64 {
        match self {
            Self::Buckingham => BUCKINGHAM_FACTOR,
            Self::Racah(_) => 1.0,
        }
    }

    fn property(self) -> &'static str {
        match self {
            Self::Buckingham => "quadrupole_polarizability",
            Self::Racah(_) => "multipole_polarizability",
        }
    }

    fn unit(self) -> String {
        format!("e^2*bohr^{}/hartree", 2 * self.rank())
    }

    fn count(self) -> usize {
        2 * self.rank() + 1
    }

    fn weights(self) -> Vec<Vec<f64>> {
        match self {
            Self::Buckingham => modes()
                .into_iter()
                .map(|mode| quadrupole_weights(mode).to_vec())
                .collect(),
            Self::Racah(rank) => spherical_weights(rank),
        }
    }
}

pub(super) fn is_electric_multipole(id: &str) -> bool {
    matches!(id, ID | MULTIPOLE_ID)
}

// The static pair solvers retain fixed-size tensors. Dispatch only the
// documented ranks while the ordinary complex response works on a slice.
macro_rules! with_mode_count {
    ($count:expr, $n:ident, $solve:expr) => {
        match $count {
            3 => {
                const $n: usize = 3;
                $solve
            }
            5 => {
                const $n: usize = 5;
                $solve
            }
            7 => {
                const $n: usize = 7;
                $solve
            }
            9 => {
                const $n: usize = 9;
                $solve
            }
            11 => {
                const $n: usize = 11;
                $solve
            }
            _ => unreachable!("validated multipole rank"),
        }
    };
}

#[cfg(test)]
mod tests;

pub(super) struct Request {
    convention: Convention,
    origin: Vec3,
    origin_name: String,
    frequency: Frequency,
    config: ResponseSolverConfig,
}

/// Covariant electronic source operators and the finite nuclei's conjugate
/// moments in the same harmonic basis. An isotropic Gaussian has the value
/// of a harmonic polynomial at its centre by the mean-value property.
pub(super) struct BalancedModes {
    operators: Vec<BlochMatrix>,
    nuclear: Vec<f64>,
}

impl BalancedModes {
    pub(super) fn quadrupole(
        backend: &MolecularTwoComponent<'_>,
        setup: &MolecularSetup,
        origin: [f64; 3],
    ) -> Result<Self> {
        Self::new(backend, setup, origin, Convention::Buckingham)
    }

    fn new(
        backend: &MolecularTwoComponent<'_>,
        setup: &MolecularSetup,
        origin: [f64; 3],
        convention: Convention,
    ) -> Result<Self> {
        let weights = convention.weights();
        let powers: Vec<_> = crate::basis::cart_components(convention.rank() as u8)
            .into_iter()
            .map(|(a, b, c)| [a as usize, b as usize, c as usize])
            .collect();
        let ionic: Vec<f64> = powers
            .iter()
            .map(|powers| {
                setup
                    .molecule
                    .atoms
                    .iter()
                    .map(|atom| {
                        let position = atom.position.to_array();
                        f64::from(atom.z)
                            * (0..3)
                                .map(|a| (position[a] - origin[a]).powi(powers[a] as i32))
                                .product::<f64>()
                    })
                    .sum()
            })
            .collect();
        let nuclear = weights
            .iter()
            .map(|row| row.iter().zip(&ionic).map(|(w, m)| w * m).sum())
            .collect();
        Ok(Self {
            operators: backend.electric_modes(&powers, &weights, origin)?,
            nuclear,
        })
    }

    pub(super) fn electric_source(&self, field: [f64; 5]) -> (BlochMatrix, f64) {
        let n = self.operators[0].rows();
        let mut perturbation = BlochMatrix::zeros(n, n);
        let mut nuclear_energy = 0.0;
        for ((op, &moment), amplitude) in self.operators.iter().zip(&self.nuclear).zip(field) {
            perturbation.add_scaled(op, amplitude);
            nuclear_energy -= amplitude * moment;
        }
        (perturbation, nuclear_energy)
    }

    pub(super) fn buckingham_moment(&self, density: &BlochMatrix) -> Vec<f64> {
        self.operators
            .iter()
            .zip(&self.nuclear)
            .map(|(operator, &nuclear)| {
                BUCKINGHAM_FACTOR * (nuclear - density.trace_product(operator))
            })
            .collect()
    }
}

impl Request {
    pub(super) fn new(id: &str, setup: &MolecularSetup, controls: &Controls) -> Result<Self> {
        let convention = if id == ID {
            Convention::Buckingham
        } else {
            let rank = controls.usize("multipole_rank", 3)?;
            if !(1..=5).contains(&rank) {
                return Err(DftError::InvalidInput(
                    "multipole_rank must be between 1 and 5".into(),
                ));
            }
            Convention::Racah(rank)
        };
        let (origin, origin_name) = moment_origin(controls, setup)?;
        Ok(Self {
            convention,
            origin,
            origin_name,
            frequency: super::polarizability::frequency(controls)?,
            config: super::vibrational::response_config(controls)?,
        })
    }

    pub(super) fn new_static_correlated(
        id: &str,
        setup: &MolecularSetup,
        controls: &Controls,
    ) -> Result<Self> {
        let request = Self::new(id, setup, controls)?;
        if request.frequency.energy()? != Complex::ZERO {
            return Err(DftError::Unsupported {
                feature: format!("frequency-dependent double-hybrid {id}"),
                missing: "retarded parent, target and PT2 response to multipole fields".into(),
            });
        }
        Ok(request)
    }

    pub(super) fn dynamic_config(&self) -> Result<Option<&ResponseSolverConfig>> {
        Ok((self.frequency.energy()? != Complex::ZERO).then_some(&self.config))
    }

    pub(super) fn origin(&self) -> [f64; 3] {
        self.origin.to_array()
    }

    pub(super) fn uses_buckingham_modes(&self) -> bool {
        matches!(self.convention, Convention::Buckingham)
    }

    pub(super) fn append_balanced(
        &self,
        report: &mut Value,
        setup: &MolecularSetup,
        backend: &MolecularTwoComponent<'_>,
        state: &SpinorScfReport,
        quadrupole: Option<&BalancedModes>,
    ) -> Result<()> {
        let generated;
        let modes = if let Some(quad) = quadrupole.filter(|_| self.uses_buckingham_modes()) {
            quad
        } else {
            generated = BalancedModes::new(backend, setup, self.origin(), self.convention)?;
            &generated
        };
        let mixed = self.uses_buckingham_modes();
        let mut operators = modes.operators.clone();
        let count = operators.len();
        if mixed {
            operators.extend_from_slice(backend.dipole_operators());
        }
        let z = self.frequency.energy()?;
        let response = backend
            .operator_response(state, &operators, &[z], &self.config)?
            .remove(0);
        let tensor = negative_expectation(response.susceptibility);
        self.append(
            report,
            OperatorPolarizability {
                tensor: tensor
                    .iter()
                    .take(count)
                    .map(|row| row[..count].to_vec())
                    .collect(),
                residuals: response.residuals[..count].to_vec(),
                iterations: response.iterations[..count].to_vec(),
            },
        )?;
        if mixed {
            let mut units = report
                .get("units")
                .cloned()
                .expect("multipole response units");
            let from_gradient: Vec<_> = tensor[count..].iter().map(|row| &row[..count]).collect();
            let from_field: Vec<_> = tensor[..count].iter().map(|row| &row[count..]).collect();
            for (name, block) in [
                ("dipole_from_field_gradient", from_gradient),
                ("field_conjugate_moment_from_field", from_field),
            ] {
                let real: Vec<Vec<_>> = block
                    .iter()
                    .map(|row| row.iter().map(|z| z.re).collect())
                    .collect();
                let imag: Vec<Vec<_>> = block
                    .iter()
                    .map(|row| row.iter().map(|z| z.im).collect())
                    .collect();
                for (key, value) in [
                    (format!("{name}_real"), &real),
                    (format!("{name}_imag"), &imag),
                ] {
                    report.set(&key, value.clone());
                    units.set(&key, "e^2*bohr^3/hartree");
                }
                if z == Complex::ZERO {
                    report.set(name, real);
                    units.set(name, "e^2*bohr^3/hartree");
                }
            }
            report.set(
                "dipole_source_residuals",
                response.residuals[count..].to_vec(),
            );
            report.set(
                "dipole_source_iterations",
                response.iterations[count..].to_vec(),
            );
            units.set("dipole_source_residuals", "dimensionless");
            units.set("dipole_source_iterations", "integer");
            report.set("units", units);
        }
        Ok(())
    }

    fn operators(&self, setup: &MolecularSetup) -> Vec<Matrix> {
        let terms = moment_operators(setup, self.origin, self.convention.rank() as u8);
        let n = setup.basis.nao_sph;
        self.convention
            .weights()
            .into_iter()
            .map(|weights| {
                let mut operator = Matrix::zeros(n, n);
                // An electrostatic field is coupled as -lambda M. The charge
                // sign gives the electron a positive solid-harmonic operator.
                // Fixed ionic moments have no orbital response.
                for (weight, term) in weights.into_iter().zip(&terms).filter(|(w, _)| *w != 0.0) {
                    for (out, value) in operator
                        .as_mut_slice()
                        .iter_mut()
                        .zip(term.matrix.as_slice())
                    {
                        *out += weight * value;
                    }
                }
                operator
            })
            .collect()
    }

    fn append(&self, report: &mut Value, response: OperatorPolarizability) -> Result<()> {
        let z = self.frequency.energy()?;
        let scale = self.convention.scale();
        let property = self.convention.property();
        let real_key = format!("{property}_real");
        let imag_key = format!("{property}_imag");
        let real: Vec<Vec<f64>> = response
            .tensor
            .iter()
            .map(|row| row.iter().map(|v| scale * v.re).collect())
            .collect();
        let imag: Vec<Vec<f64>> = response
            .tensor
            .iter()
            .map(|row| row.iter().map(|v| scale * v.im).collect())
            .collect();
        let tensor = &response.tensor;
        let reciprocity = tensor
            .iter()
            .enumerate()
            .flat_map(|(a, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(b, v)| scale * (*v - tensor[b][a]).abs())
            })
            .fold(0.0_f64, f64::max);
        report.set(&real_key, real.clone());
        report.set(&imag_key, imag);
        if z == Complex::ZERO {
            report.set(property, real);
        }
        match self.convention {
            Convention::Buckingham => {
                report.set("field_gradient_modes", modes());
                report.set(
                    "mode_order",
                    ["xx-yy", "xx+yy-2zz", "xy+yx", "xz+zx", "yz+zy"],
                );
                report.set("axis_convention", "rows are projected Buckingham quadrupole components, columns are field-gradient mode amplitudes; G = sum_b lambda_b T_b; H'(G) = +G:rr/2 for electrons; exp(-i z t)");
            }
            Convention::Racah(rank) => {
                report.set(
                    "cartesian_powers",
                    crate::basis::cart_components(rank as u8)
                        .into_iter()
                        .map(|(a, b, c)| [a, b, c])
                        .collect::<Vec<_>>(),
                );
                report.set("mode_cartesian_coefficients", self.convention.weights());
                let rank = rank as i32;
                let order = std::iter::once(0).chain((1..=rank).flat_map(|m| [m, -m]));
                report.set("mode_order", order.collect::<Vec<_>>());
                report.set("multipole_rank", rank);
                report.set("axis_convention", "rows are Racah-normalized real spherical charge moments and columns are conjugate multipole-field amplitudes; E_ext = -sum_m lambda_m M_m; exp(-i z t)");
            }
        }
        report.set("shape", [self.convention.count(); 2]);
        report.set("origin", self.origin.to_array());
        report.set("origin_convention", self.origin_name.as_str());
        report.set("frequency_real", z.re);
        report.set("frequency_imag", z.im);
        report.set("reciprocity_defect", reciprocity);
        report.set("response_residuals", response.residuals);
        report.set("response_iterations", response.iterations);
        report.set("provenance", "analytic");
        let mut units = report.get("units").cloned().unwrap_or_else(Value::object);
        let unit = self.convention.unit();
        for key in [property, &real_key, &imag_key, "reciprocity_defect"] {
            units.set(key, unit.as_str());
        }
        for (key, unit) in [
            ("energy", "hartree"),
            ("origin", "bohr"),
            ("frequency_real", "hartree"),
            ("frequency_imag", "hartree"),
            ("shape", "integer"),
            ("response_residuals", "dimensionless"),
            ("response_iterations", "integer"),
        ] {
            units.set(key, unit);
        }
        match self.convention {
            Convention::Buckingham => units.set("field_gradient_modes", "dimensionless"),
            Convention::Racah(_) => {
                units.set("multipole_rank", "integer");
                units.set("mode_order", "integer");
                units.set("cartesian_powers", "integer");
                units.set("mode_cartesian_coefficients", "dimensionless");
            }
        }
        if report.get("response_reference_fock_defect").is_some() {
            units.set("response_reference_fock_defect", "hartree");
        }
        report.set("units", units);
        Ok(())
    }

    fn spinor_sources(&self, setup: &MolecularSetup) -> Vec<BlochMatrix> {
        self.operators(setup)
            .into_iter()
            .map(spinor::scalar_source)
            .collect()
    }

    pub(super) fn append_spinor(
        &self,
        report: &mut Value,
        setup: &MolecularSetup,
        scf: &SpinorScfReport,
        fixed_operator: &BlochMatrix,
    ) -> Result<()> {
        let sources = self.spinor_sources(setup);
        let value = crate::response::nuclear::spinor_one_body_response(
            &setup.molecule,
            &setup.basis,
            setup.grid.as_ref(),
            &setup.functional,
            scf,
            &setup.options,
            fixed_operator,
            &sources,
            &[self.frequency.energy()?],
            &self.config,
        )?;
        report.set(
            "response_reference_fock_defect",
            value.reference_fock_defect,
        );
        let response = value
            .values
            .into_iter()
            .next()
            .expect("one requested frequency");
        self.append(
            report,
            OperatorPolarizability {
                tensor: negative_expectation(response.susceptibility),
                residuals: response.residuals,
                iterations: response.iterations,
            },
        )
    }

    pub(super) fn append_correlated_spinor(
        &self,
        report: &mut Value,
        setup: &MolecularSetup,
        scf: &SpinorScfReport,
        fixed_operator: &BlochMatrix,
        model: SpinorModel,
    ) -> Result<()> {
        let sources = self.spinor_sources(setup);
        with_mode_count!(
            sources.len(),
            N,
            self.append_correlated_spinor_modes::<N>(
                report,
                setup,
                scf,
                fixed_operator,
                model,
                sources
            )
        )
    }

    fn append_correlated_spinor_modes<const N: usize>(
        &self,
        report: &mut Value,
        setup: &MolecularSetup,
        scf: &SpinorScfReport,
        fixed_operator: &BlochMatrix,
        model: SpinorModel,
        sources: Vec<BlochMatrix>,
    ) -> Result<()> {
        let sources: [BlochMatrix; N] = sources.try_into().expect("validated multipole mode count");
        let value = spinor::solve(setup, scf, fixed_operator, &model, sources, &self.config)?;
        self.append(
            report,
            OperatorPolarizability {
                tensor: value
                    .tensor(model.coefficient)
                    .map(|row| row.to_vec())
                    .to_vec(),
                residuals: value.residuals.clone(),
                iterations: Vec::new(),
            },
        )?;
        value.append(
            report,
            &model,
            self.convention.property(),
            self.convention.scale(),
            self.convention.rank(),
        );
        report.set("model", "Static noncollinear double-hybrid electric energy Hessian with full-spin stationary projector and generalized PT2 amplitude Taylor response. One spin-invariant PT2 coefficient multiplies unregularized full-Coulomb correlation. A selected distinct target is evaluated on the relaxed parent trajectory.");
        Ok(())
    }
}

fn negative_expectation(mut susceptibility: Vec<Vec<Complex>>) -> Vec<Vec<Complex>> {
    for value in susceptibility.iter_mut().flatten() {
        *value = -*value;
    }
    susceptibility
}

fn correlated(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let prepared = super::double_hybrid::prepare(system, controls, id)?;
    let request = prepared
        .quadrupole_request
        .as_ref()
        .expect("electric multipole controls");
    let sources = request.operators(&prepared.setup);
    let z = request.frequency.energy()?;
    if z != Complex::ZERO {
        let solved =
            super::correlated_electric::frequency::solve(&prepared, &sources, z, &request.config)?;
        let mut report = Value::object();
        solved.append_channels(
            &mut report,
            request.convention.property(),
            request.convention.scale(),
            request.convention.rank(),
            &request.convention.unit(),
        );
        request.append(&mut report, solved.response)?;
        return super::double_hybrid::append_prepared_energy(prepared, report);
    }
    with_mode_count!(sources.len(), N, correlated_modes::<N>(prepared, sources))
}

fn correlated_modes<const N: usize>(
    prepared: super::double_hybrid::Prepared,
    sources: Vec<Matrix>,
) -> Result<Value> {
    let request = prepared
        .quadrupole_request
        .as_ref()
        .expect("electric multipole controls");
    let sources: [Matrix; N] = sources.try_into().expect("validated multipole mode count");
    prepared.check_path_memory(2)?;
    let value = super::correlated_electric::polarizability(
        &prepared.setup,
        &prepared.reference,
        prepared.target.as_ref(),
        prepared.target_vv10,
        &prepared.problem,
        &prepared.tensor,
        prepared.coefficients,
        &sources,
        &request.config,
    )?;
    let mut report = Value::object();
    request.append(
        &mut report,
        OperatorPolarizability {
            tensor: value
                .tensor
                .into_iter()
                .map(|row| row.into_iter().map(|v| Complex::new(v, 0.0)).collect())
                .collect(),
            residuals: value.response_residuals.clone(),
            iterations: value.response_iterations.clone(),
        },
    )?;
    let mut labels = report
        .get("units")
        .cloned()
        .expect("quadrupole response units");
    let property = request.convention.property();
    let unit = request.convention.unit();
    let scale = request.convention.scale();
    for (key, tensor) in [
        (format!("parent_{property}"), value.parent),
        (format!("target_{property}_correction"), value.target),
        (format!("pt2_{property}_correction"), value.pt2),
        (format!("pt2_same_spin_alpha_{property}"), value.channels[0]),
        (format!("pt2_opposite_spin_{property}"), value.channels[1]),
        (format!("pt2_same_spin_beta_{property}"), value.channels[2]),
    ] {
        report.set(
            &key,
            tensor.map(|row| row.map(|component| scale * component)),
        );
        labels.set(&key, unit.as_str());
    }
    let moment_unit = format!("e*bohr^{}", request.convention.rank());
    super::double_hybrid::append_quadratic_diagnostics(
        &mut report,
        &mut labels,
        &value,
        &moment_unit,
        &unit,
    );
    report.set("units", labels);
    report.set("independent_components", N * (N + 1) / 2);
    report.set("model", "static double-hybrid electric energy Hessian with second-order parent orbital response, noncanonical PT2 amplitude Taylor equations and distinct-target density/kernel derivatives; integer zero-temperature occupations; unregularized full-Coulomb PT2");
    super::double_hybrid::append_prepared_energy(prepared, report)
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    run_electric_multipole(ID, system, controls)
}

pub(super) fn run_multipole(system: &System, controls: &Controls) -> Result<Value> {
    run_electric_multipole(MULTIPOLE_ID, system, controls)
}

fn run_electric_multipole(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let kind = CollinearPropertyReference::from_controls(controls)?;
    if super::double_hybrid::requested(controls)? {
        if kind == CollinearPropertyReference::Roks {
            return Err(DftError::Unsupported {
                feature: format!("ROKS {id} with a double hybrid"),
                missing: "reference-specific second-order orbital, target and PT2 response to multipole fields".into(),
            });
        }
        return correlated(id, system, controls);
    }
    let setup = setup(system, controls)?;
    let request = Request::new(id, &setup, controls)?;
    controls.finish(id)?;
    let sources = request.operators(&setup);
    let common_open_shell =
        kind == CollinearPropertyReference::Roks && setup.molecule.multiplicity > 1;
    let (reference, response, model) = if common_open_shell {
        if setup.method.stabilize || setup.smd.is_some() {
            return Err(DftError::Unsupported {
                feature: format!("ROKS {id} with stability following or an SMD surface"),
                missing: "the matching constrained reference and response".into(),
            });
        }
        let (scf, orbitals) = crate::scf::roks::run_roks_with_orbitals(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )?;
        let n = setup.basis.nao_sph;
        let operators: Vec<_> = sources
            .into_iter()
            .map(|re| {
                let operator = crate::integrals::london::ComplexMatrix {
                    re,
                    im: Matrix::zeros(n, n),
                };
                [operator.clone(), operator]
            })
            .collect();
        let value = crate::response::nuclear::open_shell_operator_response(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &scf,
            &orbitals,
            &setup.options,
            &operators,
            request.frequency,
            &request.config,
        )?;
        let reference = MolecularReference {
            scf,
            e_dispersion: crate::dispersion::dispersion_energy(
                &setup.molecule,
                &setup.dispersion,
                &setup.method.dispersion_functional_name(),
            )?,
            e_smd_cds: 0.0,
            following: None,
        };
        (
            reference,
            OperatorPolarizability {
                tensor: negative_expectation(value.susceptibility),
                residuals: value.residuals,
                iterations: value.iterations,
            },
            "analytic electric multipole response on the constrained common-orbital reference",
        )
    } else {
        let reference = run_collinear_property_reference(&setup, kind)?;
        let response = real_operator_polarizability(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            &sources,
            request.frequency,
            &request.config,
        )?;
        (
            reference,
            response,
            "analytic electric multipole response on the paired collinear reference",
        )
    };
    let mut report = reference_report(&setup, &reference, model);
    if report.get("scf_orbital_residual").is_none() {
        report.set(
            "scf_orbital_residual",
            super::molecular::reference_orbital_residual(&reference.scf),
        );
    }
    if kind == CollinearPropertyReference::Roks {
        report.set("reference_type", kind.name());
    }
    request.append(&mut report, response)?;
    Ok(report)
}
