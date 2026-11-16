// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::{particle_requests, particles_and_model, quantum_basis_options, species_options};
use crate::basis::MolBasis;
use crate::driver::molecular::molecular_moment_origin;
use crate::driver::no_pair::{
    electronic_interaction, electronic_options, finite_geometry, FiniteGeometry,
};
use crate::driver::{Controls, System, Value};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{attraction, AnalyticNucleus, AnalyticShell};
use crate::integrals::dirac::GaussianNucleus;
use crate::integrals::one_electron::{kinetic_matrix, overlap_matrix};
use crate::linalg::Matrix;
use crate::multicomponent::relativistic::{
    BalancedQuantumNucleus, JointElectricOperators, JointElectricResponse, JointNoPairSystem,
};
use crate::properties::no_pair::stationary::DiracCoulombBackend;
use crate::response::complex::Complex;
use crate::response::electric_tensor::{quadrupole_modes, BUCKINGHAM_FACTOR};
use crate::{DftError, Result};

fn analytic_shells(basis: &MolBasis) -> Result<Vec<AnalyticShell<f64>>> {
    basis
        .shells
        .iter()
        .map(|s| AnalyticShell::from_placed(s, s.center.to_array()))
        .collect()
}

fn nuclear_core(
    basis: &MolBasis,
    shells: &[AnalyticShell<f64>],
    classical: &[GaussianNucleus<f64>],
    charge: f64,
    radius: f64,
    mass: f64,
) -> Result<Matrix> {
    let mut core = kinetic_matrix(basis);
    for value in core.as_mut_slice() {
        *value /= mass;
    }
    for (i, a) in shells.iter().enumerate() {
        let offset_a = basis.shells[i].sph_off;
        for (j, b) in shells.iter().enumerate().take(i + 1) {
            let offset_b = basis.shells[j].sph_off;
            for nucleus in classical {
                let kernel = GaussianKernel::LongRange {
                    omega: 1.5_f64.sqrt() / nucleus.rms_radius.hypot(radius),
                };
                // The shared attraction integral uses the electron sign.
                let repulsion = attraction(
                    a,
                    b,
                    &[AnalyticNucleus {
                        center: nucleus.center,
                        charge: -charge * nucleus.charge,
                    }],
                    kernel,
                    0,
                )?;
                let [rows, columns] = repulsion.shape;
                for row in 0..rows {
                    for column in 0..columns {
                        let value = repulsion.values[row * columns + column];
                        core[(offset_a + row, offset_b + column)] += value;
                        if i != j {
                            core[(offset_b + column, offset_a + row)] += value;
                        }
                    }
                }
            }
        }
    }
    Ok(core)
}

fn dipole_response<const N: usize>(response: &JointElectricResponse<N>) -> Value {
    let tensor = std::array::from_fn::<_, 3, _>(|a| {
        std::array::from_fn::<_, 3, _>(|b| response.tensor[a][b].re)
    });
    let tensor_imag = std::array::from_fn::<_, 3, _>(|a| {
        std::array::from_fn::<_, 3, _>(|b| response.tensor[a][b].im)
    });
    let raw_asymmetry = (0..3)
        .flat_map(|a| (0..3).map(move |b| (response.tensor[a][b] - response.tensor[b][a]).abs()))
        .fold(0.0_f64, f64::max);
    Value::object()
        .with(
            "tensor",
            (response.frequency == Complex::ZERO).then_some(tensor),
        )
        .with("tensor_real", tensor)
        .with("tensor_imag", tensor_imag)
        .with("frequency_real", response.frequency.re)
        .with("frequency_imag", response.frequency.im)
        .with(
            "axis_convention",
            "dipole axis by electric-field axis; exp(-i z t)",
        )
        .with("raw_asymmetry", raw_asymmetry)
        .with("response_residuals", &response.relative_residuals[..3])
        .with("response_iterations", &response.iterations[..3])
        .with("reference_residual", response.reference_residual)
}

fn append_response_block<const ROWS: usize, const COLS: usize>(
    report: &mut Value,
    units: &mut Value,
    name: &str,
    unit: &str,
    tensor: [[Complex; COLS]; ROWS],
    static_frequency: bool,
) {
    let real = tensor.map(|row| row.map(|z| z.re));
    let real_key = format!("{name}_real");
    let imag_key = format!("{name}_imag");
    report.set(&real_key, real);
    report.set(&imag_key, tensor.map(|row| row.map(|z| z.im)));
    units.set(&real_key, unit);
    units.set(&imag_key, unit);
    if static_frequency {
        report.set(name, real);
        units.set(name, unit);
    }
}

fn quadrupole_report(
    moments: [f64; 8],
    origin: [f64; 3],
    origin_name: &str,
    response: Option<&JointElectricResponse<8>>,
) -> (Value, Value) {
    let mut report = Value::object()
        .with("moment", std::array::from_fn::<_, 5, _>(|a| BUCKINGHAM_FACTOR * moments[a + 3]))
        .with("origin", origin)
        .with("origin_convention", origin_name)
        .with("field_gradient_modes", quadrupole_modes())
        .with("mode_order", ["xx-yy", "xx+yy-2zz", "xy+yx", "xz+zx", "yz+zy"])
        .with("axis_convention", "rows are observables and columns are fields; Buckingham moment equals three times the field-conjugate moment; G = sum_b lambda_b T_b; electron H'(G) = G:rr/2; exp(-i z t)");
    let mut units = Value::object()
        .with("moment", "e*bohr^2")
        .with("origin", "bohr")
        .with("field_gradient_modes", "dimensionless");
    if let Some(response) = response {
        let stat = response.frequency == Complex::ZERO;
        let block = |a: usize, b: usize| response.tensor[a][b];
        let dipole_gradient: [[Complex; 5]; 3] =
            std::array::from_fn(|a| std::array::from_fn(|b| block(a, b + 3)));
        let gradient_dipole: [[Complex; 3]; 5] =
            std::array::from_fn(|a| std::array::from_fn(|b| block(a + 3, b)));
        let quadrupole: [[Complex; 5]; 5] = std::array::from_fn(|a| {
            std::array::from_fn(|b| block(a + 3, b + 3) * BUCKINGHAM_FACTOR)
        });
        let cross_asymmetry = (0..3)
            .flat_map(|a| {
                (0..5).map(move |b| (dipole_gradient[a][b] - gradient_dipole[b][a]).abs())
            })
            .fold(0.0_f64, f64::max);
        let quadrupole_asymmetry = (0..5)
            .flat_map(|a| (0..5).map(move |b| (quadrupole[a][b] - quadrupole[b][a]).abs()))
            .fold(0.0_f64, f64::max);
        append_response_block(
            &mut report,
            &mut units,
            "polarizability",
            "e^2*bohr^4/hartree",
            quadrupole,
            stat,
        );
        append_response_block(
            &mut report,
            &mut units,
            "dipole_from_field_gradient",
            "e^2*bohr^3/hartree",
            dipole_gradient,
            stat,
        );
        // Keep this transpose partner in the field-conjugate normalization.
        // Multiplying it by three gives the Buckingham moment response.
        append_response_block(
            &mut report,
            &mut units,
            "field_conjugate_moment_from_field",
            "e^2*bohr^3/hartree",
            gradient_dipole,
            stat,
        );
        report.set("frequency_real", response.frequency.re);
        report.set("frequency_imag", response.frequency.im);
        report.set("cross_raw_asymmetry", cross_asymmetry);
        report.set("raw_asymmetry", quadrupole_asymmetry);
        report.set("response_residuals", response.relative_residuals);
        report.set("response_iterations", response.iterations);
        report.set("reference_residual", response.reference_residual);
        for (key, unit) in [
            ("frequency_real", "hartree"),
            ("frequency_imag", "hartree"),
            ("cross_raw_asymmetry", "e^2*bohr^3/hartree"),
            ("raw_asymmetry", "e^2*bohr^4/hartree"),
            ("response_residuals", "dimensionless"),
            ("response_iterations", "integer"),
            ("reference_residual", "hartree"),
        ] {
            units.set(key, unit);
        }
    }
    (report, units)
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    if !controls.str("xc", "hf")?.eq_ignore_ascii_case("hf")
        || controls.str("dispersion", "none")? != "none"
    {
        return Err(DftError::Unsupported {
            feature: "Dirac–Coulomb multicomponent energy with XC or dispersion".into(),
            missing: "a relativistic interspecies variational XC or dispersion provider".into(),
        });
    }
    if controls
        .opt_f64("grid_weight_cutoff")?
        .is_some_and(|cutoff| cutoff != 0.0)
    {
        return Err(DftError::InvalidInput(
            "joint no-pair HF has no XC quadrature".into(),
        ));
    }
    let field = controls.opt_vec3("field")?.map(|v| v.to_array());
    let polarizability = controls.bool("species_polarizability", false)?;
    let quadrupole_polarizability = controls.bool("species_quadrupole_polarizability", false)?;
    let field_gradient = controls
        .opt_f64_list("species_field_gradient")?
        .map(|values| {
            <[f64; 5]>::try_from(values).map_err(|_| {
                DftError::InvalidInput(
                    "species_field_gradient requires five mode amplitudes".into(),
                )
            })
        })
        .transpose()?;
    let gradient_model = field_gradient.is_some() || quadrupole_polarizability;
    let response = if polarizability || quadrupole_polarizability {
        Some((
            crate::driver::polarizability::frequency(controls)?.energy()?,
            crate::driver::vibrational::response_config(controls)?,
        ))
    } else {
        None
    };
    let (charge, electronic_config) = electronic_options(controls)?;
    let (interaction, interaction_name, hamiltonian) = electronic_interaction(controls)?;
    let options = species_options(controls)?;
    let (exponents, lmax) = quantum_basis_options(controls)?;
    let proton_correlation = controls.str("interspecies_correlation", "none")?;
    let FiniteGeometry {
        molecule,
        basis,
        basis_name,
        radii,
        c,
    } = finite_geometry(system, controls, charge)?;
    let (origin, origin_name) = if gradient_model {
        molecular_moment_origin(controls, &molecule, "zero")?
    } else {
        (crate::math::Vec3::zero(), "zero".into())
    };
    let origin = origin.to_array();
    let requests = particle_requests(
        controls,
        &molecule,
        &exponents,
        lmax,
        &proton_correlation,
        "none",
    )?;
    controls.finish("multicomponent Dirac–Coulomb")?;
    if requests.is_empty() {
        return Err(DftError::InvalidInput(
            "Dirac–Coulomb multicomponent energy requires at least one quantum nucleus".into(),
        ));
    }
    let mut replaced = vec![false; molecule.atoms.len()];
    let mut atoms = Vec::with_capacity(requests.len());
    for request in &requests {
        let Some(atom) = request.replaced_atom else {
            return Err(DftError::Unsupported {
                feature: format!(
                    "Dirac–Coulomb multicomponent energy with a {} particle",
                    request.kind
                ),
                missing: "relativistic coupling for added particles".into(),
            });
        };
        if replaced[atom] || !request.mass.is_finite() || request.mass <= 0.0 {
            return Err(DftError::InvalidInput(
                "quantum nuclei must replace distinct atoms and have positive finite masses".into(),
            ));
        }
        replaced[atom] = true;
        atoms.push(atom);
    }
    let (particles, correlation) = particles_and_model(&requests)?;
    if !correlation.is_zero() || !proton_correlation.eq_ignore_ascii_case("none") {
        return Err(DftError::Unsupported {
            feature: "interspecies correlation with Dirac–Coulomb electrons".into(),
            missing:
                "a relativistic interspecies correlation functional and both of its Fock potentials"
                    .into(),
        });
    }
    let classical: Vec<_> = molecule
        .atoms
        .iter()
        .zip(&radii)
        .enumerate()
        .filter(|(i, _)| !replaced[*i])
        .map(|(_, (atom, &rms_radius))| GaussianNucleus {
            center: atom.position.to_array(),
            charge: f64::from(atom.z),
            rms_radius,
        })
        .collect();
    let shells = analytic_shells(&basis)?;
    let nuclear_shells: Vec<_> = particles
        .iter()
        .map(|p| analytic_shells(&p.basis))
        .collect::<Result<_>>()?;
    let overlaps: Vec<_> = particles.iter().map(|p| overlap_matrix(&p.basis)).collect();
    let cores: Vec<_> = particles
        .iter()
        .zip(&nuclear_shells)
        .zip(&atoms)
        .map(|((p, shells), &atom)| {
            nuclear_core(&p.basis, shells, &classical, p.charge, radii[atom], p.mass)
        })
        .collect::<Result<_>>()?;
    let nuclei: Vec<_> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| BalancedQuantumNucleus {
            shells: &nuclear_shells[i],
            overlap: &overlaps[i],
            one_body: &cores[i],
            charge: p.charge,
            intrinsic_rms_radius: Some(radii[atoms[i]]),
        })
        .collect();
    // The same electronic two-particle operator enters the coupled SCF
    // and the electronic block of its response. Couplings to the massive
    // quantum nuclei remain electrostatic finite-charge contractions.
    let provider = DiracCoulombBackend::new(&shells, &classical, c, molecule.n_electrons()?)?
        .with_interaction(interaction);
    let space = provider.electronic_space(electronic_config.overlap_cutoff)?;
    let model = JointNoPairSystem {
        electronic: &provider,
        electronic_space: &space,
        electronic_shells: &shells,
        speed_of_light: c,
        nuclei: &nuclei,
    };
    let (result, dipole, electric, quadrupole) = if gradient_model {
        let operators = JointElectricOperators::with_field_gradient(&model, &classical, origin)?;
        let mut amplitudes = [0.0; 8];
        amplitudes[..3].copy_from_slice(&field.unwrap_or_default());
        amplitudes[3..].copy_from_slice(&field_gradient.unwrap_or_default());
        let perturbation =
            (field.is_some() || field_gradient.is_some()).then_some((&operators, amplitudes));
        let result = model.solve_in_electric_field(&electronic_config, &options, perturbation)?;
        let moments = operators.field_conjugate_moments(&result)?;
        let dipole = Some(std::array::from_fn(|a| moments[a]));
        let solved = response
            .map(|(frequency, config)| {
                model.electric_response(
                    &result,
                    &operators,
                    frequency,
                    electronic_config.overlap_cutoff,
                    &config,
                )
            })
            .transpose()?;
        let electric = if polarizability {
            solved.as_ref().map_or(Value::Null, dipole_response)
        } else {
            Value::Null
        };
        let quadrupole_response = if quadrupole_polarizability {
            solved.as_ref()
        } else {
            None
        };
        let quadrupole = quadrupole_report(moments, origin, &origin_name, quadrupole_response);
        (result, dipole, electric, Some(quadrupole))
    } else if field.is_some() || polarizability {
        let classical_source = std::array::from_fn(|axis| {
            -classical
                .iter()
                .map(|n| n.charge * n.center[axis])
                .sum::<f64>()
        });
        let operators = JointElectricOperators::new(&model, classical_source)?;
        let result = model.solve_in_field(
            &electronic_config,
            &options,
            field.map(|field| (&operators, field)),
        )?;
        let dipole = Some(operators.dipole(&result)?);
        let electric = if let Some((frequency, config)) = response {
            dipole_response(&model.electric_response(
                &result,
                &operators,
                frequency,
                electronic_config.overlap_cutoff,
                &config,
            )?)
        } else {
            Value::Null
        };
        (result, dipole, electric, None)
    } else {
        (
            model.solve(&electronic_config, &options)?,
            None,
            Value::Null,
            None,
        )
    };
    let quantum = result
        .nuclei
        .iter()
        .zip(&particles)
        .zip(&atoms)
        .map(|((state, p), &atom)| {
            Value::object()
                .with("label", p.label.as_str())
                .with("atom", atom)
                .with("charge", p.charge)
                .with("mass", p.mass)
                .with("rms_radius", radii[atom])
                .with("density", state.density.clone())
                .with("fock", state.fock.clone())
                .with("one_body_energy", state.one_body_energy)
                .with("orbital_residual", state.orbital_residual)
                .with("density_residual", state.density_residual)
                .with("particle_number_error", state.particle_number_error)
        })
        .collect();
    let terms = result.electronic.fields.energy_components;
    let nuclear_one_body = result.nuclei.iter().map(|n| n.one_body_energy).sum::<f64>();
    let labels = [
        ("energy", "hartree"),
        ("energy_components", "hartree"),
        ("electronic_energy", "hartree"),
        ("field", "hartree/e/bohr"),
        ("dipole", "e*bohr"),
        ("dipole_origin", "bohr"),
        ("polarizability.tensor", "bohr^3"),
        ("polarizability.tensor_real", "bohr^3"),
        ("polarizability.tensor_imag", "bohr^3"),
        ("polarizability.frequency_real", "hartree"),
        ("polarizability.frequency_imag", "hartree"),
        ("polarizability.raw_asymmetry", "bohr^3"),
        ("polarizability.response_residuals", "dimensionless"),
        ("polarizability.response_iterations", "integer"),
        ("polarizability.reference_residual", "hartree"),
        ("electronic_density_real", "dimensionless"),
        ("electronic_density_imaginary", "dimensionless"),
        ("electronic_orbital_residual", "hartree"),
        ("electronic_density_residual", "dimensionless"),
        ("electronic_particle_number_error", "electrons"),
        ("max_electron_nuclear_reciprocity_defect", "hartree"),
        ("max_nuclear_pair_reciprocity_defect", "hartree"),
        ("projection_residual", "hartree"),
        ("spectral_gap", "hartree"),
        ("speed_of_light", "atomic units of velocity"),
        ("electronic_dimension", "integer"),
        ("classical_nucleus_count", "integer"),
        ("iterations", "integer"),
        ("quantum_atoms", "integer"),
        ("quantum.atom", "integer"),
        ("quantum.charge", "e"),
        ("quantum.mass", "atomic units of mass"),
        ("quantum.rms_radius", "bohr"),
        ("quantum.density", "dimensionless"),
        ("quantum.fock", "hartree"),
        ("quantum.one_body_energy", "hartree"),
        ("quantum.orbital_residual", "hartree"),
        ("quantum.density_residual", "dimensionless"),
        ("quantum.particle_number_error", "dimensionless"),
    ];
    let mut units = Value::object();
    for (key, label) in labels {
        units.set(key, label);
    }
    if field_gradient.is_some() {
        units.set("field_gradient", "hartree/e/bohr^2");
    }
    if let Some((_, Value::Object(labels))) = &quadrupole {
        for (key, label) in labels {
            units.set(&format!("electric_quadrupole.{key}"), label);
        }
    }
    let mut components = Value::object()
        .with("classical_nuclear_repulsion", terms[0])
        .with("electronic_kinetic_rest_subtracted", terms[1])
        .with("electron_classical_nuclear", terms[2])
        .with("electronic_hartree", terms[4])
        .with("electronic_exchange", terms[5])
        .with("nuclear_one_body", nuclear_one_body)
        .with("electron_quantum_nuclear", result.electron_nuclear_energy)
        .with("quantum_nuclear_pair", result.nuclear_pair_energy);
    if field.is_some() || field_gradient.is_some() {
        components.set("external_field", result.external_field_energy);
    }
    let current = provider.current_fields(&result.electronic_density)?;
    if let Some(current) = &current {
        components.set("electronic_hartree", terms[4] - current.direct_energy);
        components.set("electronic_exchange", terms[5] - current.exchange_energy);
        components.set("current_direct", current.direct_energy);
        components.set("current_exchange", current.exchange_energy);
        units.set("current_hermiticity_defect", "hartree");
    }
    let mut report = Value::object().with("model", format!("fixed-sector no-pair {hamiltonian} HF and distinguishable Gaussian quantum nuclei in the laboratory frame"))
        .with("species_electronic_model", "dirac-coulomb").with("basis", basis_name)
        .with("dirac_interaction", interaction_name)
        .with("projector_reference", if classical.is_empty() { "free-particle" } else { "classical-nuclear-core" })
        .with("classical_nucleus_count", classical.len()).with("quantum_atoms", atoms)
        .with("energy", result.energy)
        .with("energy_components", components)
        .with("electronic_energy", result.electronic_energy)
        .with("electronic_density_real", result.electronic_density.re)
        .with("electronic_density_imaginary", result.electronic_density.im)
        .with("electronic_dimension", space.basis().cols())
        .with("electronic_orbital_residual", result.electronic.orbital_residual)
        .with("electronic_density_residual", result.electronic.density_change)
        .with("electronic_particle_number_error", result.electronic.particle_number_error)
        .with("max_electron_nuclear_reciprocity_defect", result.max_electron_nuclear_reciprocity_defect)
        .with("max_nuclear_pair_reciprocity_defect", result.max_nuclear_pair_reciprocity_defect)
        .with("projection_residual", result.projection_residual)
        .with("speed_of_light", c).with("spectral_gap", space.spectral_gap)
        .with("iterations", result.iterations).with("quantum", Value::Array(quantum))
        .with("center_of_mass_removed", false).with("converged", result.converged)
        .with("warnings", Vec::<String>::new()).with("units", units);
    if let Some(current) = current {
        report.set("current_hermiticity_defect", current.hermiticity_defect);
    }
    if let Some(field) = field {
        report.set("field", field);
    }
    if let Some(field_gradient) = field_gradient {
        report.set("field_gradient", field_gradient);
    }
    if let Some(dipole) = dipole {
        report.set("dipole", dipole);
        report.set("dipole_origin", origin);
    }
    if polarizability {
        report.set("polarizability", electric);
    }
    if let Some((quadrupole, _)) = quadrupole {
        report.set("electric_quadrupole", quadrupole);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_joint_dirac_coulomb_with_classical_and_all_quantum_nuclei() {
        let path =
            std::env::temp_dir().join(format!("dft-rs-dirac-quantum-{}.g94", std::process::id()));
        std::fs::write(
            &path,
            "H 0\nS 1 1.0\n0.84 1.0\n****\nHe 0\nS 1 1.0\n0.93 1.0\n****\n",
        )
        .unwrap();
        let positions = vec![[0.12, -0.27, 0.09], [1.41, 0.31, -0.46]];
        let system = System::molecule(vec![2, 1], positions.clone());
        let options = Value::object()
            .with("basis", path.to_str().unwrap())
            .with("charge", 1_i64)
            .with("species_electronic_model", "dirac-coulomb")
            .with("nuclear_rms_radii", [0.12, 0.081])
            .with("speed_of_light", 7.4)
            .with("quantum_exponents", [4.0])
            .with("quantum_lmax", 0_usize);
        let run = |atoms: Vec<usize>, masses: Vec<f64>| {
            let input = options
                .clone()
                .with("quantum_atoms", atoms)
                .with("quantum_masses", masses);
            crate::driver::run(
                "multicomponent",
                &system,
                &Controls::from_value(&input).unwrap(),
            )
            .unwrap()
        };
        let mixed = run(vec![1], vec![19.0]);
        let all = run(vec![0, 1], vec![17.0, 19.0]);
        std::fs::remove_file(path).unwrap();
        let scalar = |report: &Value, key: &str| report.get(key).unwrap().as_f64().unwrap();
        let component =
            |report: &Value, key: &str| scalar(report.get("energy_components").unwrap(), key);
        let distance = (0..3)
            .map(|i| (positions[0][i] - positions[1][i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let w_mixed = (1.5 / (0.12_f64.powi(2) + 0.081_f64.powi(2) + 0.75 / 4.0)).sqrt();
        let w_pair = (1.5 / (0.12_f64.powi(2) + 0.081_f64.powi(2) + 2.0 * 0.75 / 4.0)).sqrt();
        let nuclear_mixed = 1.5 * 4.0 / 19.0 + 2.0 * libm::erf(w_mixed * distance) / distance;
        let nuclear_pair = 2.0 * libm::erf(w_pair * distance) / distance;
        assert!((component(&mixed, "nuclear_one_body") - nuclear_mixed).abs() < 1e-12);
        assert!((component(&all, "quantum_nuclear_pair") - nuclear_pair).abs() < 1e-12);
        assert_eq!(
            mixed.get("projector_reference").and_then(Value::as_str),
            Some("classical-nuclear-core")
        );
        assert_eq!(
            all.get("projector_reference").and_then(Value::as_str),
            Some("free-particle")
        );
        assert_eq!(component(&all, "electron_classical_nuclear"), 0.0);
        assert_eq!(component(&all, "classical_nuclear_repulsion"), 0.0);
        for (name, report) in [("mixed", mixed), ("all", all)] {
            assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
            assert!(scalar(&report, "projection_residual").abs() < 1e-13);
            let terms = report.get("energy_components").unwrap();
            let Value::Object(terms) = terms else {
                panic!("expected energy components");
            };
            assert!(
                (scalar(&report, "energy")
                    - terms.iter().map(|(_, v)| v.as_f64().unwrap()).sum::<f64>())
                .abs()
                    < 1e-12
            );
            assert!(crate::driver::units::complete(&mut report.clone()).is_empty());
            println!("Joint Dirac Coulomb public {name} {}", report.to_json());
        }
    }
}

#[cfg(test)]
#[path = "relativistic_quadrupole_tests.rs"]
mod quadrupole_tests;

#[cfg(test)]
mod electric_tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use stencils::Stencil;

    #[test]
    fn public_joint_dirac_coulomb_electric_response_fits_all_components() {
        check_electric_response("coulomb");
    }

    #[test]
    fn public_joint_dirac_gaunt_electric_response_fits_all_components() {
        check_electric_response("gaunt");
    }

    #[test]
    fn public_joint_dirac_breit_electric_response_fits_all_components() {
        check_electric_response("breit");
    }

    pub(super) fn fixture(interaction: &str, tag: &str) -> (System, std::path::PathBuf, Value) {
        let system = System::molecule(
            vec![2, 1, 1, 1],
            vec![
                [0.12, -0.27, 0.09],
                [1.41, 0.31, -0.46],
                [-0.47, 1.71, 0.28],
                [0.54, -0.39, 1.92],
            ],
        );
        let path = std::env::temp_dir().join(format!(
            "dft-rs-dirac-quantum-{tag}-{interaction}-{}.g94",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "H 0\nS 1 1.0\n0.84 1.0\n****\nHe 0\nS 1 1.0\n0.93 1.0\n****\n",
        )
        .unwrap();
        let options = Value::object()
            .with("basis", path.to_str().unwrap())
            .with("charge", 3_i64)
            .with("species_electronic_model", "dirac-coulomb")
            .with("dirac_interaction", interaction)
            .with("nuclear_rms_radii", [0.12, 0.081, 0.093, 0.105])
            .with("speed_of_light", 11.7)
            .with("quantum_atoms", [1, 3])
            .with("quantum_masses", [7.0, 9.0])
            .with("quantum_exponents", [1.0])
            .with("quantum_lmax", 1_usize)
            .with("grid_weight_cutoff", 0.0)
            .with("species_tolerance", 3e-12)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-12)
            .with("cpu_threads", 1_usize);
        (system, path, options)
    }

    fn check_electric_response(interaction: &str) {
        let (system, path, options) = fixture(interaction, "field");
        let bias = [0.013, -0.009, 0.007];
        let solve = |input: Value| {
            let result = crate::driver::run(
                "multicomponent",
                &system,
                &Controls::from_value(&input).unwrap(),
            )
            .unwrap_or_else(|error| panic!("joint electric response {input:?}: {error}"));
            assert_eq!(result.get("converged").and_then(Value::as_bool), Some(true));
            assert!(crate::driver::units::complete(&mut result.clone()).is_empty());
            result
        };
        let energy = |value: &Value| value.get("energy").unwrap().as_f64().unwrap();
        let dipole = |value: &Value| value.get("dipole").unwrap().numeric_array().unwrap().1;
        let base = options
            .clone()
            .with("field", bias)
            .with("species_polarizability", true);
        let analytic = solve(base.clone());
        assert_eq!(
            analytic.get("dirac_interaction").and_then(Value::as_str),
            Some(interaction)
        );
        let terms = analytic.get("energy_components").unwrap();
        let Value::Object(entries) = terms else {
            panic!("expected energy terms");
        };
        assert!(
            (energy(&analytic)
                - entries
                    .iter()
                    .map(|(_, v)| v.as_f64().unwrap())
                    .sum::<f64>())
            .abs()
                < 1e-12
        );
        if interaction != "coulomb" {
            assert!(
                terms
                    .get("current_exchange")
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    .abs()
                    > 1e-6
            );
            assert!(
                analytic
                    .get("current_hermiticity_defect")
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    < 1e-12
            );
        }
        let response = analytic.get("polarizability").unwrap();
        let tensor = |value: &Value, key| value.get(key).unwrap().numeric_array().unwrap().1;
        let alpha = tensor(response, "tensor");
        let mu = dipole(&analytic);
        let field_energy = analytic
            .get("energy_components")
            .unwrap()
            .get("external_field")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!(
            (field_energy + bias.iter().zip(&mu).map(|(e, m)| e * m).sum::<f64>()).abs() < 1e-12
        );
        let labels: Vec<_> = (0..3)
            .flat_map(|b| (0..3).map(move |a| format!("joint {interaction} alpha[{a},{b}]")))
            .chain((0..3).map(|a| format!("joint {interaction} dipole[{a}] from energy")))
            .chain(
                ["xy", "xz", "yz"].map(|axes| format!("joint {interaction} mixed energy {axes}")),
            )
            .collect();
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let mut failed = Vec::new();
        components::fit_components(
            &labels,
            &[[0.03, 0.022, 0.016, 0.011], [0.013, 0.009, 0.006, 0.004]],
            |steps| {
                let mut errors = vec![[0.0; 4]; labels.len()];
                let sample = |a: usize, b: Option<usize>, h: f64| {
                    let mut field = bias;
                    field[a] += h;
                    if let Some(b) = b {
                        field[b] += h;
                    }
                    solve(options.clone().with("field", field))
                };
                for (i, &h) in steps.iter().enumerate() {
                    let mut second = [0.0; 3];
                    for b in 0..3 {
                        let plus = sample(b, None, h);
                        let minus = sample(b, None, -h);
                        let (mp, mm) = (dipole(&plus), dipole(&minus));
                        for a in 0..3 {
                            let fd = Stencil::Second.first(mp[a], mm[a], 0.0, 0.0, h);
                            errors[3 * b + a][i] = (alpha[3 * a + b] - fd).abs();
                        }
                        let (ep, em) = (energy(&plus), energy(&minus));
                        errors[9 + b][i] =
                            (mu[b] + Stencil::Second.first(ep, em, 0.0, 0.0, h)).abs();
                        second[b] = (ep + em - 2.0 * energy(&analytic)) / h.powi(2);
                    }
                    for (p, (a, b)) in pairs.into_iter().enumerate() {
                        let ep = energy(&sample(a, Some(b), h));
                        let em = energy(&sample(a, Some(b), -h));
                        let mixed = 0.5
                            * ((ep + em - 2.0 * energy(&analytic)) / h.powi(2)
                                - second[a]
                                - second[b]);
                        errors[12 + p][i] = (alpha[3 * a + b] + mixed).abs();
                    }
                }
                errors
            },
            &mut failed,
        );
        components::assert_no_failures("joint electric dipole and response", &failed);
        let retarded = |omega, damping, imaginary| {
            solve(
                base.clone()
                    .with("frequency", omega)
                    .with("frequency_damping", damping)
                    .with("imaginary_frequency", imaginary),
            )
        };
        let plus = retarded(0.08, 0.008, false);
        let minus = retarded(-0.08, 0.008, false);
        let near_zero = retarded(1e-5, 0.0, true);
        let get = |report: &Value, key| tensor(report.get("polarizability").unwrap(), key);
        let conjugate_error = get(&plus, "tensor_real")
            .iter()
            .zip(get(&minus, "tensor_real"))
            .map(|(a, b)| (a - b).abs())
            .chain(
                get(&plus, "tensor_imag")
                    .iter()
                    .zip(get(&minus, "tensor_imag"))
                    .map(|(a, b)| (a + b).abs()),
            )
            .fold(0.0_f64, f64::max);
        let limit_error = alpha
            .iter()
            .zip(get(&near_zero, "tensor_real"))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(conjugate_error < 1e-8);
        assert!(limit_error < 1e-7);
        assert!(get(&plus, "tensor_imag")[0] > 0.0);
        assert!(get(&near_zero, "tensor_imag")
            .iter()
            .all(|v| v.abs() < 1e-8));
        std::fs::remove_file(path).unwrap();
        println!("Joint electric static {}", analytic.to_json());
        println!("Joint electric retarded {}", plus.to_json());
        println!(
            "Joint electric limits conjugation={conjugate_error:.17e} static={limit_error:.17e} interaction={interaction}"
        );
    }
}
