// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/stencils.rs"]
pub mod stencils;

use dft_rs::dfpt::orbitals::{band_density, generalized_orbitals, matmul};
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::dirac::GaussianNucleus;
use dft_rs::linalg::Orthogonalizer;
use dft_rs::periodic::occupations::solve_occupations;
use dft_rs::periodic::spinor_scf::{
    from_pauli, pauli_components, run_spinor_scf, spinful_time_reversal_defect,
    BiasedSpinorBackend, SpinorEnergyBackend, SpinorFields,
};
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::two_component::{
    MolecularRelativisticOptions, MolecularTwoComponent, RelativisticSpin, TwoComponentMethod,
};
use dft_rs::resources::ResourceLimits;
use dft_rs::Result;
use stencils::{fit_components_staged, Stencil};

fn object(key: &str, value: &Value) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

fn array(value: &Value) -> &[Value] {
    value.as_array().unwrap()
}

fn geometry(numbers: &[u8], positions: &[[f64; 3]]) -> System {
    System {
        numbers: numbers.to_vec(),
        positions: positions.to_vec(),
        ..System::default()
    }
}

fn controls(xc: &str, method: &str, spin: &str, basis: &str, radii: &[f64]) -> Value {
    let mut values = Value::object()
        .with("xc", xc)
        .with("basis", basis)
        .with("relativistic_spin", spin)
        .with("nuclear_rms_radii", radii.to_vec())
        .with("grid_level", 0_usize)
        .with("grid_weight_cutoff", 0.0)
        .with("vv10_grid_level", 0_usize)
        .with("cpu_threads", 2_usize)
        .with("conv_tol", 2e-12)
        .with("conv_tol_dp", 2e-10)
        .with("conv_tol_grad", 2e-10)
        .with("scf_optimizer", "legacy");
    if method != "dirac" {
        values.set("relativistic_method", method);
    }
    values
}

fn report(system: &System, controls: &Value) -> Value {
    report_for("two-component-relativistic", system, controls)
}

fn report_for(feature: &str, system: &System, controls: &Value) -> Value {
    let result = run(feature, system, &Controls::from_value(controls).unwrap()).unwrap();
    assert_eq!(
        result.get("converged"),
        Some(&Value::Bool(true)),
        "controls {} report {}",
        controls.to_json(),
        result.to_json()
    );
    assert!(object("scf_orbital_residual", &result) <= 2e-10);
    assert!(object("decoupling_metric_defect", &result) < 2e-10);
    assert!(object("particle_number_error", &result) < 2e-10);
    assert!(object("coulomb_hermiticity_defect", &result) < 1e-10);
    result
}

#[test]
fn dirac_dks_matches_the_exact_x2c_chart_for_real_hydrogen_fluoride() {
    let system = geometry(&[9, 1], &[[0.0; 3], [0.0, 0.0, 1.7325]]);
    let radii = [0.0000547, 0.0000159];
    for xc in ["wb97x-d", "wb97x-v"] {
        let x2c = controls(xc, "x2c", "spin-orbit", "sto-3g", &radii);
        let dirac = controls(xc, "dirac", "spin-orbit", "sto-3g", &radii);
        let exact = report_for("dirac-dks", &system, &dirac);
        let transformed = report(&system, &x2c);
        let terms = exact.get("energy_components").unwrap();
        let transformed_terms = transformed.get("energy_components").unwrap();
        assert!((object("energy", &exact) - object("energy", &transformed)).abs() < 3e-10);
        for key in [
            "relativistic_core",
            "hartree",
            "exchange",
            "long_range_exchange",
            "semilocal_xc",
            "vv10",
            "dispersion",
        ] {
            assert!(
                (object(key, terms) - object(key, transformed_terms)).abs() < 3e-10,
                "{xc} {key}"
            );
        }
        assert!(object("spectral_gap", &exact) > 10_000.0);
        assert_eq!(
            exact.get("relativistic_method").and_then(Value::as_str),
            Some("dirac")
        );
        println!("Hydrogen fluoride Dirac–Coulomb {xc} {}", exact.to_json());
    }
}

#[test]
fn real_hydrogen_fluoride_requested_functionals_and_relativistic_limits() {
    // Fixed experimental-scale bond length, with no geometry optimization.
    // Radii are explicit approximate proton and fluorine Gaussian RMS values.
    let system = geometry(&[9, 1], &[[0.0; 3], [0.0, 0.0, 1.7325]]);
    let radii = [0.000054_7, 0.000015_9];
    for xc in ["wb97x-d", "wb97x-v", "wb97m-v"] {
        let base = controls(xc, "x2c", "spin-orbit", "sto-3g", &radii);
        let relativistic = report(&system, &base);
        let limiting = report(&system, &base.clone().with("speed_of_light", 1370.35999084));
        let scalar = report(&system, &base.clone().with("relativistic_spin", "scalar"));
        let energy = object("energy", &relativistic);
        let delta = energy - object("energy", &limiting);
        let soc_shift = energy - object("energy", &scalar);
        let terms = relativistic.get("energy_components").unwrap();
        let dipole = array(relativistic.get("dipole").unwrap())[2]
            .as_f64()
            .unwrap();
        assert!(energy > -110.0 && energy < -85.0);
        assert!(delta < -1e-3 && delta > -1.0);
        assert!(soc_shift.abs() < 5e-3);
        assert!((0.1..2.0).contains(&dipole.abs()));
        assert!(object("long_range_exchange", terms) < -0.01);
        if xc == "wb97x-d" {
            assert!(object("dispersion", terms) < 0.0);
            assert_eq!(object("vv10", terms), 0.0);
        } else {
            assert!(object("vv10", terms).abs() > 1e-6);
            assert_eq!(object("dispersion", terms), 0.0);
        }
        println!("Hydrogen fluoride {xc} STO-3G physical {} scalar {} large-c {} corrections {delta:.12e} {soc_shift:.12e}",
            relativistic.to_json(), scalar.to_json(), limiting.to_json());
    }
    let original = report(
        &system,
        &controls("wb97x-d", "x2c", "spin-orbit", "sto-3g", &radii),
    );
    let expanded = report(
        &system,
        &controls("wb97x-d", "x2c", "spin-orbit", "6-31g", &radii),
    );
    assert!(object("energy", &expanded) < object("energy", &original));
    println!("Hydrogen fluoride wb97x-d 6-31G {}", expanded.to_json());
}

#[test]
fn neutral_uniform_electric_origin_and_nonrelativistic_hf_oracles() {
    let positions = [[0.18, -0.21, 0.06], [1.13, 0.48, -0.81]];
    let system = geometry(&[1, 1], &positions);
    let settings = controls("hf", "x2c", "spin-orbit", "sto-3g", &[0.0000159; 2]);
    let field = [0.004, -0.002, 0.003];
    let reference = report(&system, &settings.clone().with("field", field.to_vec()));
    let shifted = report(
        &system,
        &settings
            .clone()
            .with("field", field.to_vec())
            .with("electric_origin", vec![0.34, -0.27, 0.45]),
    );
    assert!((object("energy", &reference) - object("energy", &shifted)).abs() < 2e-12);
    for (a, b) in array(reference.get("dipole").unwrap())
        .iter()
        .zip(array(shifted.get("dipole").unwrap()))
    {
        assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-10);
    }
    let ordinary = run(
        "energy",
        &system,
        &Controls::from_value(&Value::object().with("xc", "hf").with("basis", "sto-3g")).unwrap(),
    )
    .unwrap();
    let model = |speed| {
        report(&system, &settings.clone().with("speed_of_light", speed))
            .get("energy")
            .unwrap()
            .as_f64()
            .unwrap()
    };
    let reference = object("energy", &ordinary);
    let steps = [0.04, 0.03, 0.02, 0.015];
    let errors = steps.map(|x| (model(1.0 / x) - reference).abs());
    assert!(components::component_fit_of_order(
        "H2 X2C HF nonrelativistic limit",
        &steps,
        &errors,
        2.0
    ));
    println!(
        "H2 nonrelativistic reference {reference:.15e} Ha and absolute differences {errors:?}"
    );
}

#[test]
fn real_hydrogen_chloride_relativistic_methods() {
    // Fixed gas-phase equilibrium-scale H–Cl distance. Nuclear sizes are
    // approximate isotope RMS values for chlorine-35 and hydrogen-1.
    let system = geometry(&[17, 1], &[[0.0; 3], [0.0, 0.0, 2.4086]]);
    let radii = [0.000063_6, 0.000015_9];
    for (xc, method) in [
        ("wb97x-d", "x2c"),
        ("wb97x-v", "dkh2"),
        ("wb97m-v", "matrix-zora"),
    ] {
        let controls = controls(xc, method, "spin-orbit", "sto-3g", &radii);
        let physical = report(&system, &controls);
        let scalar = report(
            &system,
            &controls.clone().with("relativistic_spin", "scalar"),
        );
        let limiting = report(&system, &controls.with("speed_of_light", 1370.35999084));
        let energy = object("energy", &physical);
        let correction = energy - object("energy", &limiting);
        let spin_orbit = energy - object("energy", &scalar);
        let dipole = array(physical.get("dipole").unwrap())[2].as_f64().unwrap();
        assert!((-500.0..-420.0).contains(&energy));
        assert!((-10.0..-0.02).contains(&correction));
        assert!(spin_orbit.abs() < 0.03);
        assert!((0.08..2.0).contains(&dipole.abs()));
        println!(
            "Hydrogen chloride {xc}/{method} STO-3G physical {} scalar {} large-c {} corrections {correction:.12e} {spin_orbit:.12e}",
            physical.to_json(),
            scalar.to_json(),
            limiting.to_json()
        );
    }
}

#[test]
fn all_electric_dipole_and_relaxed_polarizability_components() {
    check_electric_response(&[
        ("x2c", "scalar", "hf"),
        ("dkh2", "spin-orbit", "hf"),
        ("dkh2", "scalar", "hf"),
        ("matrix-zora", "spin-orbit", "hf"),
        ("matrix-zora", "scalar", "hf"),
        ("x2c", "scalar", "wb97x-d"),
        ("x2c", "scalar", "wb97x-v"),
        ("x2c", "scalar", "wb97m-v"),
    ]);
}

#[test]
fn x2c_spin_orbit_electric_response_follows_the_stationary_reference() {
    let (system, radii) = c1_electric_fixture();
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let values = controls("hf", "x2c", "spin-orbit", "sto-3g", &radii).with("speed_of_light", 13.0);
    let analytic = report(&system, &values.clone().with("response_order", 1_usize));
    ResourceLimits::new(Some(2), None)
        .unwrap()
        .install(|| {
            let settings = Controls::from_value(&values).unwrap();
            let method = MolecularMethod::from_controls(&settings).unwrap();
            let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
            let nuclei: Vec<_> = setup.molecule.atoms.iter().zip(radii).map(
                |(atom, rms_radius)| GaussianNucleus {
                    center: atom.position.to_array(),
                    charge: f64::from(atom.z),
                    rms_radius,
                },
            ).collect();
            let electrons = setup.molecule.n_electrons().unwrap();
            let options = MolecularRelativisticOptions {
                method: TwoComponentMethod::X2c,
                spin: RelativisticSpin::SpinOrbit,
                speed_of_light: 13.0,
                metric_cutoff: 1e-12,
                xc_angular_tolerance: 1e-9,
                pair_cache_mb: setup.options.incore_mb,
                electric_field: [0.0; 3],
                electric_origin: [0.0; 3],
                dispersion_energy: None,
            };
            let mut config = setup.nested_soc_scf_config(&settings);
            config.overlap_cutoff = options.metric_cutoff;
            config.orbital_optimizer = setup.spinor_scf_config().orbital_optimizer;
            let backend = MolecularTwoComponent::new(
                &setup.basis, &nuclei, electrons, &setup.functional,
                setup.grid.as_ref(), None, options,
            ).unwrap();
            let reference = run_spinor_scf(&backend, &config).unwrap();
            assert!(reference.converged);
            assert!((reference.fields.energy - object("energy", &analytic)).abs() < 2e-12);
            let metric = &backend.overlap()[0];
            let orthogonalizer =
                Orthogonalizer::canonical(&metric.real_embedding(), config.overlap_cutoff)
                    .unwrap();
            let reference_projector = matmul(metric, &reference.density[0]).unwrap();
            let initial = time_even_part(&reference.fields.fock[0]).unwrap();
            let (nuclear, vertices) = dipole_vertices(&backend);
            let evaluate = |field: [f64; 3]| {
                let mut vertex = BlochMatrix::zeros(metric.rows(), metric.cols());
                for (operator, amplitude) in vertices.iter().zip(field) {
                    vertex.add_scaled(operator, amplitude);
                }
                // A scalar electric field preserves spinful time reversal.
                // Keep that branch during iteration and check the final
                // density against the unprojected physical Hamiltonian.
                let source = BiasedSpinorBackend::new(
                    &backend, Some(vec![vertex]), Some(vec![initial.clone()]),
                ).unwrap();
                let state = run_spinor_scf(&TimeEvenBackend(&source), &config).unwrap();
                let density = &state.density[0];
                let fields = source.evaluate(&state.density).unwrap();
                let fock = &fields.fock[0];
                let bands = generalized_orbitals(fock, metric, config.overlap_cutoff).unwrap();
                let occupations =
                    solve_occupations(&[bands.energies.clone()], electrons as f64, config.occupation)
                        .unwrap();
                let canonical = band_density(&bands, &occupations.occupations[0]).unwrap();
                let occupation = density
                    .re
                    .max_abs_diff(&canonical.re)
                    .max(density.im.max_abs_diff(&canonical.im));
                let projector = matmul(metric, density).unwrap();
                let mut commutator = matmul(&matmul(fock, density).unwrap(), metric).unwrap();
                commutator.add_scaled(&matmul(&projector, fock).unwrap(), -1.0);
                let x = &orthogonalizer.x;
                let residual = x
                    .transpose_matmul(&commutator.real_embedding().matmul(x))
                    .as_slice()
                    .iter()
                    .map(|entry| entry.abs())
                    .fold(0.0, f64::max);
                let overlap = projector.trace_product(&reference_projector) / electrons as f64;
                let [x, y, z] = backend.dipole(density).unwrap();
                let energy = fields.energy
                    - nuclear.iter().zip(field).map(|(m, f)| m * f).sum::<f64>();
                println!(
                    "C1 X2C branch field={field:?} energy={energy:.15e} overlap={overlap:.12e} residual={residual:.12e} occupation={occupation:.12e} dP={:.12e}",
                    state.density_change,
                );
                assert!(state.converged && overlap > 0.99, "field {field:?}");
                assert!(spinful_time_reversal_defect(&state.density, [1; 3]).unwrap() < 2e-10);
                assert!(residual <= config.orbital_tolerance);
                assert!(occupation < config.density_tolerance);
                assert!(state.particle_number_error < 2e-10);
                assert!(fields.coulomb_operator_defect < 1e-10);
                [energy, x, y, z]
            };
            let mut failures = Vec::new();
            fit_electric_dipole(
                &analytic, &CARTESIAN_ELECTRIC_DIRECTIONS,
                "x2c/spin-orbit/hf followed branch", &mut failures, evaluate,
            );
            components::assert_no_failures("C1 X2C branch electric response", &failures);
        })
        .unwrap();
}

struct TimeEvenBackend<'a, B>(&'a B);

impl<B: SpinorEnergyBackend> SpinorEnergyBackend for TimeEvenBackend<'_, B> {
    fn overlap(&self) -> &[BlochMatrix] {
        self.0.overlap()
    }

    fn initial_fock(&self) -> &[BlochMatrix] {
        self.0.initial_fock()
    }

    fn electrons(&self) -> f64 {
        self.0.electrons()
    }

    fn quadratic_density_energy(&self) -> bool {
        self.0.quadratic_density_energy()
    }

    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let mut fields = self.0.evaluate(density)?;
        for fock in &mut fields.fock {
            *fock = time_even_part(fock)?;
        }
        Ok(fields)
    }
}

fn time_even_part(operator: &BlochMatrix) -> Result<BlochMatrix> {
    // In the Pauli representation, time reversal conjugates the charge
    // block and changes the sign of each conjugated spin block.
    let [mut charge, mut x, mut y, mut z] = pauli_components(operator)?;
    charge.im.as_mut_slice().fill(0.0);
    for spin in [&mut x, &mut y, &mut z] {
        spin.re.as_mut_slice().fill(0.0);
    }
    from_pauli(&[charge, x, y, z])
}

fn dipole_vertices(backend: &MolecularTwoComponent<'_>) -> ([f64; 3], [BlochMatrix; 3]) {
    // Recover each one-electron vertex from the affine dipole expectation
    // on an independent Hermitian basis. The zero-density term is nuclear.
    let size = backend.overlap()[0].rows();
    let zero = BlochMatrix::zeros(size, size);
    let nuclear = backend.dipole(&zero).unwrap();
    let mut vertices = std::array::from_fn(|_| zero.clone());
    for i in 0..size {
        let mut density = zero.clone();
        density.re[(i, i)] = 1.0;
        let diagonal = backend.dipole(&density).unwrap();
        for a in 0..3 {
            vertices[a].re[(i, i)] = nuclear[a] - diagonal[a];
        }
        for j in 0..i {
            let mut density = zero.clone();
            density.re[(i, j)] = 0.5;
            density.re[(j, i)] = 0.5;
            let real = backend.dipole(&density).unwrap();
            density.re[(i, j)] = 0.0;
            density.re[(j, i)] = 0.0;
            density.im[(i, j)] = 0.5;
            density.im[(j, i)] = -0.5;
            let imaginary = backend.dipole(&density).unwrap();
            for a in 0..3 {
                let re = nuclear[a] - real[a];
                let im = nuclear[a] - imaginary[a];
                vertices[a].re[(i, j)] = re;
                vertices[a].re[(j, i)] = re;
                vertices[a].im[(i, j)] = im;
                vertices[a].im[(j, i)] = -im;
            }
        }
    }
    (nuclear, vertices)
}

#[test]
fn dirac_dks_all_electric_dipole_and_relaxed_polarizability_components() {
    check_electric_response(&[("dirac", "spin-orbit", "pbe")]);
}

#[test]
fn legacy_x2c_negative_field_converges_from_the_core_guess() {
    let (system, radii) = c1_electric_fixture();
    let settings = controls("hf", "x2c", "spin-orbit", "sto-3g", &radii)
        .with("speed_of_light", 13.0)
        .with("field", [-0.0, -0.008, -0.0]);
    let result = report(&system, &settings);
    println!(
        "C1 X2C Legacy cold energy={:.15e} residual={:.12e} dP={:.12e} occupation={:.12e} iterations={}",
        object("energy", &result),
        object("scf_orbital_residual", &result),
        object("density_change", &result),
        object("occupation_consistency", &result),
        object("scf_iterations", &result),
    );
    assert!(object("density_change", &result) < 2e-10);
    assert!(object("occupation_consistency", &result) < 2e-10);
}

#[test]
fn dirac_dks_current_interactions_match_the_dirac_hf_limit() {
    let system = geometry(&[1, 1], &[[0.18, -0.21, 0.06], [1.13, 0.48, -0.81]]);
    let radii = [0.0000159; 2];
    for (interaction, xc) in [("gaunt", "wb97x-d"), ("breit", "wb97x-v")] {
        let model = controls(xc, "dirac", "spin-orbit", "sto-3g", &radii)
            .with("dirac_interaction", interaction);
        let physical = report_for("dirac-dks", &system, &model);
        let terms = physical.get("energy_components").unwrap();
        let energy = object("energy", &physical);
        let current = object("current_direct", terms) + object("current_exchange", terms);
        assert!(current.abs() > 1e-10);
        assert!(
            (energy
                - [
                    "classical_nuclear_repulsion",
                    "relativistic_core",
                    "vv10",
                    "dispersion",
                    "hartree",
                    "exchange",
                    "long_range_exchange",
                    "semilocal_xc",
                    "electric",
                    "current_direct",
                    "current_exchange",
                ]
                .into_iter()
                .map(|key| object(key, terms))
                .sum::<f64>())
            .abs()
                < 2e-14
        );
        assert!(object("current_hermiticity_defect", &physical) < 1e-10);
        println!("H2 {xc} {interaction} energy={energy:.16e} signed_current={current:.16e}");
        let inverse_speeds = [0.04, 0.03, 0.02, 0.015];
        let errors = inverse_speeds.map(|scale| {
            let result = report_for(
                "dirac-dks",
                &system,
                &model.clone().with("speed_of_light", 1.0 / scale),
            );
            let components = result.get("energy_components").unwrap();
            (object("current_direct", components) + object("current_exchange", components)).abs()
        });
        assert!(components::component_fit_of_order(
            &format!("H2 {xc} {interaction} current nonrelativistic limit"),
            &inverse_speeds,
            &errors,
            2.0,
        ));

        let shared = Value::object()
            .with("basis", "sto-3g")
            .with("dirac_interaction", interaction)
            .with("nuclear_rms_radii", radii)
            .with("speed_of_light", 25.0)
            .with("conv_tol", 2e-12)
            .with("conv_tol_dp", 2e-10)
            .with("conv_tol_grad", 2e-10);
        let dks = report_for("dirac-dks", &system, &shared.clone().with("xc", "hf"));
        let hf = run(
            "dirac-coulomb-scf",
            &system,
            &Controls::from_value(&shared).unwrap(),
        )
        .unwrap();
        assert_eq!(hf.get("converged"), Some(&Value::Bool(true)));
        let defect = (object("energy", &dks) - object("energy", &hf)).abs();
        assert!(defect < 3e-10);
        for key in ["hartree", "exchange", "current_direct", "current_exchange"] {
            let a = object(key, dks.get("energy_components").unwrap());
            let b = object(key, hf.get("energy_components").unwrap());
            assert!((a - b).abs() < 3e-10, "{interaction} {key}");
        }
        println!(
            "H2 Dirac-HF {interaction} energy={:.16e} defect={defect:.12e}",
            object("energy", &hf)
        );
    }
}

#[test]
fn dirac_dks_current_electric_response_matches_all_static_components() {
    let positions = [
        [-0.14, 0.28, -0.19],
        [1.18, 0.37, 0.02],
        [-0.43, 3.18, 0.57],
        [0.75, 3.31, 1.33],
    ];
    c1_geometry::assert_c1(&[1; 4], &positions);
    let system = geometry(&[1; 4], &positions);
    // Strict diagonal dominance makes these mixed field directions
    // invertible, so the three measured columns span all Cartesian sources.
    let directions = [[1.0, 0.11, 0.29], [0.23, 1.0, 0.17], [0.09, 0.21, 1.0]];
    let mut failures = Vec::new();
    for (current, xc) in [("gaunt", "wb97x-d"), ("breit", "wb97x-v")] {
        let options = controls(xc, "dirac", "spin-orbit", "sto-3g", &[0.0000159; 4])
            .with("dirac_interaction", current)
            .with("speed_of_light", 25.0);
        check_electric_dipole(
            "dirac-dks",
            &system,
            &options,
            &directions,
            current,
            &mut failures,
        );
    }
    components::assert_no_failures("C1 Dirac current electric response", &failures);
}

#[test]
fn dirac_dks_quadrupole_and_mixed_response_match_all_static_components() {
    // A nonsymmetric hydrogen dimer fixture keeps each electronic solve
    // small while the field-gradient sources span all five traceless modes.
    let positions = [
        [-0.14, 0.28, -0.19],
        [1.18, 0.37, 0.02],
        [-0.43, 3.18, 0.57],
        [0.75, 3.31, 1.33],
    ];
    c1_geometry::assert_c1(&[1; 4], &positions);
    let system = geometry(&[1; 4], &positions);
    let origin = vec![0.21, 0.79, -0.30];
    let controls = controls("wb97x-d", "dirac", "spin-orbit", "sto-3g", &[0.0000159; 4]);
    let calculate = |feature: &str, controls: &Value| {
        let r = run(feature, &system, &Controls::from_value(controls).unwrap()).unwrap();
        assert_eq!(
            r.get("converged").and_then(Value::as_bool),
            Some(true),
            "{}",
            r.to_json()
        );
        assert!(object("scf_orbital_residual", &r) <= 2e-10);
        r
    };
    let read = |report: &Value, key: &str| report.get(key).unwrap().numeric_array().unwrap().1;
    let request = controls
        .clone()
        .with("relativistic_method", "dirac")
        .with("origin", origin.clone());
    let stationary = calculate("quadrupole-polarizability", &request);
    let quadrupole = read(&stationary, "quadrupole_polarizability");
    let mixed_dipole = read(&stationary, "dipole_from_field_gradient");
    let mixed_moment = read(&stationary, "field_conjugate_moment_from_field");
    let dipole = read(&stationary, "dipole");
    let moment = read(&stationary, "quadrupole_moment");
    for key in ["response_residuals", "dipole_source_residuals"] {
        assert!(read(&stationary, key).iter().all(|x| *x < 1e-8));
    }
    for a in 0..3 {
        for b in 0..5 {
            assert!((mixed_dipole[5 * a + b] - mixed_moment[3 * b + a]).abs() < 2e-8);
        }
    }
    let mut targets = Vec::new();
    for source in 0..8 {
        let first = if source < 3 {
            -dipole[source]
        } else {
            -moment[source - 3] / 3.0
        };
        targets.push((source, 0, first));
        if source < 3 {
            targets.extend((0..5).map(|a| (source, a + 4, 3.0 * mixed_moment[3 * a + source])));
        } else {
            let b = source - 3;
            targets.extend((0..3).map(|a| (source, a + 1, mixed_dipole[5 * a + b])));
            targets.extend((0..5).map(|a| (source, a + 4, quadrupole[5 * a + b])));
        }
    }
    let labels: Vec<_> = targets
        .iter()
        .map(|&(s, a, _)| format!("DKS electric observable[{a}] source[{s}]"))
        .collect();
    let solve = |source, amplitude| {
        let (mut electric, mut gradient) = (vec![0.0; 3], vec![0.0; 5]);
        if source < 3 {
            electric[source] = amplitude;
        } else {
            gradient[source - 3] = amplitude;
        }
        let k = controls
            .clone()
            .with("electric_origin", origin.clone())
            .with("field", electric)
            .with("electric_field_gradient", gradient);
        let report = calculate("dirac-dks", &k);
        let mut values = vec![object("energy", &report)];
        values.extend(read(&report, "dipole"));
        values.extend(read(&report, "quadrupole_moment"));
        values
    };
    println!("C1 DKS electric responses: {}", stationary.to_json());
    let stages = [
        ([0.01, 0.007, 0.005, 0.003], Stencil::Second),
        ([0.005, 0.003, 0.002, 0.00125], Stencil::Second),
        ([0.02, 0.014, 0.01, 0.007], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    fit_components_staged(
        &labels,
        &stages,
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; targets.len()];
            for source in 0..8 {
                for (k, &h) in steps.iter().enumerate() {
                    let (plus, minus) = (solve(source, h), solve(source, -h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (solve(source, 2.0 * h), solve(source, -2.0 * h))
                    } else {
                        (vec![0.0; 9], vec![0.0; 9])
                    };
                    for (i, &(_, a, actual)) in
                        targets.iter().enumerate().filter(|(_, t)| t.0 == source)
                    {
                        errors[i][k] = (stencil.first(plus[a], minus[a], plus2[a], minus2[a], h)
                            - actual)
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("DKS quadratic and mixed electric response", &failures);
}

fn c1_electric_fixture() -> (System, [f64; 4]) {
    // Four distinct species and noncoplanar centers give a C1 fixed geometry.
    let system = geometry(
        &[1, 2, 3, 4],
        &[
            [0.11, -0.24, 0.07],
            [1.92, 0.53, -0.82],
            [-1.61, 1.38, 0.92],
            [0.81, -1.21, 2.07],
        ],
    );
    (system, [0.021, 0.018, 0.014, 0.025])
}

const CARTESIAN_ELECTRIC_DIRECTIONS: [[f64; 3]; 3] =
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn check_electric_response(models: &[(&str, &str, &str)]) {
    // A reduced speed makes the small-component effect independently resolvable.
    let (system, radii) = c1_electric_fixture();
    let mut failures = Vec::new();
    for &(method, spin, xc) in models {
        let controls = controls(xc, method, spin, "sto-3g", &radii)
            .with("speed_of_light", 13.0)
            .with("scf_optimizer", "auto");
        let feature = if method == "dirac" {
            "dirac-dks"
        } else {
            "two-component-relativistic"
        };
        check_electric_dipole(
            feature,
            &system,
            &controls,
            &CARTESIAN_ELECTRIC_DIRECTIONS,
            &format!("{method}/{spin}/{xc}"),
            &mut failures,
        );
    }
    components::assert_no_failures("C1 molecular electric response", &failures);
}

fn check_electric_dipole(
    feature: &str,
    system: &System,
    controls: &Value,
    directions: &[[f64; 3]; 3],
    label: &str,
    failures: &mut Vec<String>,
) {
    let initial = report_for(
        feature,
        system,
        &controls.clone().with("response_order", 1_usize),
    );
    let evaluate = |field| {
        let perturbed = report_for(feature, system, &controls.clone().with("field", field));
        let mu = array(perturbed.get("dipole").unwrap());
        [
            object("energy", &perturbed),
            mu[0].as_f64().unwrap(),
            mu[1].as_f64().unwrap(),
            mu[2].as_f64().unwrap(),
        ]
    };
    fit_electric_dipole(&initial, directions, label, failures, evaluate);
}

fn fit_electric_dipole(
    initial: &Value,
    directions: &[[f64; 3]; 3],
    label: &str,
    failures: &mut Vec<String>,
    evaluate: impl Fn([f64; 3]) -> [f64; 4],
) {
    let dipole: [f64; 3] =
        std::array::from_fn(|a| array(initial.get("dipole").unwrap())[a].as_f64().unwrap());
    let block = array(&array(initial.get("polarizability_real").unwrap())[0]);
    let alpha: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|b| array(&block[a])[b].as_f64().unwrap()));
    let residual = array(&array(initial.get("response_residuals").unwrap())[0]);
    assert!(residual.iter().all(|r| r.as_f64().unwrap() < 1e-8));
    for a in 0..3 {
        for b in 0..3 {
            assert!((alpha[a][b] - alpha[b][a]).abs() < 1e-7);
        }
    }
    println!(
        "C1 all-component electric response {label} {}",
        initial.to_json()
    );
    let names: Vec<String> = (0..3)
        .flat_map(|a| {
            [format!("{label} dipole[{a}]")]
                .into_iter()
                .chain((0..3).map(move |b| format!("{label} alpha[{b},{a}]")))
        })
        .collect();
    fit_components_staged(
        &names,
        &[
            ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
            ([0.01, 0.008, 0.006, 0.004], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (k, &h) in steps.iter().enumerate() {
                for (a, &direction) in directions.iter().enumerate() {
                    let signed = |amplitude| evaluate(direction.map(|v| amplitude * v));
                    let plus = signed(h);
                    let minus = signed(-h);
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (signed(2.0 * h), signed(-2.0 * h))
                    } else {
                        ([0.0; 4], [0.0; 4])
                    };
                    for b in 0..4 {
                        let estimate = stencil.first(plus[b], minus[b], plus2[b], minus2[b], h);
                        let response = if b == 0 {
                            dipole.map(|v| -v)
                        } else {
                            alpha[b - 1]
                        };
                        let expected = response
                            .iter()
                            .zip(direction)
                            .map(|(m, d)| m * d)
                            .sum::<f64>();
                        errors[a * 4 + b][k] = (estimate - expected).abs();
                    }
                }
            }
            errors
        },
        failures,
    );
}
