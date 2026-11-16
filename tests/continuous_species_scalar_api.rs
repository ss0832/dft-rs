// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Serialized common-chart actions and all independent two-site derivatives.
use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::continuous_species::scalar::{ScalarActionRecord, ScalarOptimization, VERSION};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::scf::{run_scf, EriMode, Guess, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

fn mol_value(m: &Molecule) -> Value {
    Value::object()
        .with(
            "numbers",
            m.atoms.iter().map(|a| usize::from(a.z)).collect::<Vec<_>>(),
        )
        .with(
            "positions",
            m.atoms
                .iter()
                .map(|a| a.position.to_array().to_vec())
                .collect::<Vec<_>>(),
        )
        .with("charge", i64::from(m.charge))
        .with("multiplicity", 1usize)
}
fn basis_value(b: &MolBasis) -> Value {
    Value::Array(
        b.shells
            .iter()
            .map(|s| {
                Value::object()
                    .with("atom", s.atom)
                    .with("l", usize::from(s.l))
                    .with("exponents", s.exps.clone())
                    .with("normalized_coefficients", s.coefs.clone())
            })
            .collect(),
    )
}
fn fixture() -> (Value, ScfOptions) {
    let m = Molecule::new(
        [
            [-1.31, 0.14, 0.03],
            [-0.22, 1.18, -0.12],
            [0.83, -0.27, 0.34],
            [1.56, 0.85, -0.43],
        ]
        .into_iter()
        .map(|r| Atom {
            z: 1,
            position: Vec3::new(r[0], r[1], r[2]),
        })
        .collect(),
    );
    let raw = Molecule::new(m.atoms[..3].to_vec()).with_charge(1);
    let b = MolBasis::build(&m, &data::load("sto-3g").unwrap()).unwrap();
    let rb = MolBasis::build(&raw, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &m,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let xc = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..Default::default()
    };
    let full = run_scf(&m, &b, &options, &xc, Some(&grid)).unwrap();
    let mut references = Vec::new();
    let mut endpoints = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            let angle =
                [0.0_f64, 0.23, -0.19][i] + [0.0_f64, -0.13, 0.31][j] + 0.017 * (i * j) as f64;
            references.push(
                Value::object()
                    .with("id", format!("test-{i}-{j}"))
                    .with("site_choices", vec![i, j])
                    .with(
                        "construction_identity",
                        "synthetic occupied-rotation algebra test; not a chemical endpoint",
                    )
                    .with("construction_state", "reactant-only")
                    .with("energy", full.energy)
                    .with(
                        "orbital_residual",
                        dft_rs::response::diagnostics::reference_diagnostics(&full)
                            .unwrap()
                            .orbital_residual,
                    ),
            );
            endpoints.push(
                Value::object()
                    .with("molecule", mol_value(&m))
                    .with("basis", basis_value(&b))
                    .with("geometry_transport", "fixed-internal-affine-v1")
                    .with("geometry_atom_owners", vec![0usize, 1, 2, 2])
                    .with(
                        "frozen_orbitals",
                        (0..4)
                            .map(|k| {
                                vec![
                                    angle.cos() * full.mo_coeff[0][(k, 0)]
                                        + angle.sin() * full.mo_coeff[0][(k, 1)],
                                ]
                            })
                            .collect::<Vec<_>>(),
                    ),
            );
        }
    }
    let sites = (0..2)
        .map(|i| {
            Value::object()
                .with("id", format!("site-{i}"))
                .with("candidates", vec!["a", "b", "c"])
        })
        .collect::<Vec<_>>();
    (
        Value::object()
            .with("model_version", VERSION)
            .with("provider", "builtin-functional-registry-v1")
            .with("functional", Value::object().with("xc", "pbe"))
            .with("reference", "rks")
            .with("shell_convention", "normalized-molecular-spherical-ao-v1")
            .with("grid_level", 1usize)
            .with("sites", sites)
            .with("references", references)
            .with(
                "states",
                vec![Value::object()
                    .with("id", "neutral-test")
                    .with("coefficient", 1.)
                    .with("active_electrons", 2usize)
                    .with("raw_molecule", mol_value(&raw))
                    .with("raw_basis", basis_value(&rb))
                    .with("endpoints", endpoints)],
            ),
        options,
    )
}

#[test]
fn scalar_saved_affine_geometry_uses_the_common_driver_and_optimizer() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let (value, options) = fixture();
            let record = ScalarActionRecord::from_value(&value).unwrap();
            let mut positions = record
                .primary_molecule()
                .atoms
                .iter()
                .map(|a| a.position)
                .collect::<Vec<_>>();
            positions[0] += Vec3::new(0.021, -0.013, 0.007);
            positions[1] += Vec3::new(-0.009, 0.017, -0.011);
            let direct = record
                .calculate_geometry("neutral-test", "test-0-0", &positions, &options, None)
                .unwrap();
            let system = System {
                numbers: vec![1; 3],
                positions: positions.iter().map(|r| r.to_array()).collect(),
                cell: None,
                pbc: [false; 3],
            };
            let input = Value::object()
                .with("fragment_mode", "scalar-gradient")
                .with("fragment_model_json", record.to_json())
                .with("fragment_state", "neutral-test")
                .with("fragment_endpoint", "test-0-0");
            let output = dft_rs::driver::run(
                "continuous-species",
                &system,
                &Controls::from_value(&input).unwrap(),
            )
            .unwrap();
            assert!((number(&output, "energy") - number(&direct, "energy")).abs() < 1e-11);
            assert_eq!(output.get("gradient"), direct.get("gradient"));
            assert!(
                (number(&output, "electronic_energy") + number(&output, "nuclear_energy")
                    - number(&output, "energy"))
                .abs()
                    < 1e-12
            );
            let optimized = record
                .calculate_geometry(
                    "neutral-test",
                    "test-0-0",
                    &positions,
                    &options,
                    Some(&dft_rs::optimizer::OptOptions {
                        max_steps: 2,
                        gtol: 1e-5,
                        ..Default::default()
                    }),
                )
                .unwrap();
            assert!(number(&optimized, "energy") <= number(&direct, "energy") + 1e-12);
            assert!(optimized
                .get("geometry_optimization")
                .unwrap()
                .get("converged")
                .is_some());
            let mixed = input.with("fragment_weights", vec![1., 0., 0., 1., 0., 0.]);
            assert!(dft_rs::driver::continuous_species::scalar::run(
                &system,
                &Controls::from_value(&mixed).unwrap()
            )
            .is_err());
            assert!(record
                .calculate_geometry("missing", "test-0-0", &positions, &options, None)
                .is_err());
            println!("SAVED_AFFINE_GEOMETRY {}", optimized.to_json_pretty());
            Ok(())
        })
        .unwrap();
}
fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}

fn remove_field(value: &mut Value, key: &str) {
    let Value::Object(fields) = value else {
        panic!("object required")
    };
    fields.retain(|(name, _)| name != key);
}

fn collinear_record(mut value: Value) -> Value {
    use dft_rs::driver::continuous_species::scalar::COLLINEAR_VERSION;
    value.set("model_version", COLLINEAR_VERSION);
    value.set("reference", "uks");
    let mut states = value.get("states").unwrap().as_array().unwrap().to_vec();
    for state in &mut states {
        state.set("active_spin_electrons", [1usize, 1]);
        let mut endpoints = state.get("endpoints").unwrap().as_array().unwrap().to_vec();
        for endpoint in &mut endpoints {
            let mut full = endpoint.get("molecule").unwrap().clone();
            full.set("charge", 1i64);
            full.set("multiplicity", 2usize);
            endpoint.set("molecule", full);
            let paired = endpoint.get("frozen_orbitals").unwrap().clone();
            endpoint.set("frozen_orbitals_alpha", paired);
            endpoint.set("frozen_orbitals_beta", vec![Vec::<f64>::new(); 4]);
            remove_field(endpoint, "frozen_orbitals");
        }
        state.set("endpoints", endpoints);
    }
    value.set("states", states);
    value
}

#[test]
fn saved_scalar_hessians_preserve_spin_schema_and_driver_parity() {
    use dft_rs::response::krylov::ResponseSolverConfig;
    use dft_rs::scf::Reference;
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384)).unwrap().run(|| {
        let (paired, options) = fixture();
        let collinear = collinear_record(paired.clone());
        for (value, unrestricted) in [(paired, false), (collinear, true)] {
            let record = ScalarActionRecord::from_value(&value)?;
            assert_eq!(ScalarActionRecord::from_json(&record.to_json())?.to_value(), value);
            let mut opts = options.clone();
            opts.reference = if unrestricted { Reference::Uhf } else { Reference::Rhf };
            let r = record.primary_molecule().atoms.iter().map(|a| a.position).collect::<Vec<_>>();
            let config = ResponseSolverConfig { tolerance: 1e-12, ..Default::default() };
            let direct = record.calculate_hessian("neutral-test", "test-0-0", &r, &opts, &config)?;
            let system = System { numbers: vec![1; 3], positions: r.iter().map(|r| r.to_array()).collect(), cell: None, pbc: [false;3] };
            let input = Value::object().with("fragment_mode", "scalar-hessian")
                .with("fragment_model_json", record.to_json()).with("fragment_state", "neutral-test")
                .with("fragment_endpoint", "test-0-0").with("response_tolerance", 1e-12);
            let common = dft_rs::driver::run("continuous-species", &system, &Controls::from_value(&input)?)?;
            assert_eq!(common.get("hessian"), direct.get("hessian"));
            assert_eq!(common.get("raw_hessian"), direct.get("raw_hessian"));
            let rows = direct.get("hessian").unwrap().as_array().unwrap();
            assert_eq!(rows.len(), 9);
            assert!(rows.iter().all(|r| r.as_array().unwrap().len() == 9));
            assert!(number(&direct, "reference_orbital_residual") < 1e-11);
            assert!(number(&direct, "raw_asymmetry") < 1e-9);
            assert_eq!(direct.get("units").unwrap().get("hessian").unwrap().as_str(), Some("hartree/bohr^2"));
            if unrestricted {
                assert_eq!(direct.get("particle_number_error").unwrap().as_array().unwrap().len(), 2);
                assert!(matches!(record.calculate(&[0.4,0.3,0.3,0.4,0.3,0.3], &opts, None), Err(dft_rs::DftError::Unsupported { .. })));
            }
            println!("SAVED_SCALAR_HESSIAN unrestricted={unrestricted} energy={:.16e} residual={:.12e} asymmetry={:.12e}",
                number(&direct,"energy"), number(&direct,"reference_orbital_residual"), number(&direct,"raw_asymmetry"));
        }
        Ok(())
    }).unwrap();
}

#[test]
fn saved_scalar_spin_schema_refuses_implicit_reinterpretation() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let (paired, _) = fixture();
            let changed_reference = paired.clone().with("reference", "uks");
            assert!(ScalarActionRecord::from_value(&changed_reference).is_err());
            let collinear = collinear_record(paired);
            for field in ["frozen_orbitals_alpha", "frozen_orbitals_beta"] {
                let mut invalid = collinear.clone();
                let mut states = invalid.get("states").unwrap().as_array().unwrap().to_vec();
                let mut endpoints = states[0]
                    .get("endpoints")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .to_vec();
                remove_field(&mut endpoints[0], field);
                states[0].set("endpoints", endpoints);
                invalid.set("states", states);
                assert!(ScalarActionRecord::from_value(&invalid).is_err());
            }
            let mut invalid = collinear;
            let mut states = invalid.get("states").unwrap().as_array().unwrap().to_vec();
            states[0].set("active_spin_electrons", [2usize, 0]);
            invalid.set("states", states);
            assert!(ScalarActionRecord::from_value(&invalid).is_err());
            Ok(())
        })
        .unwrap();
}
#[test]
fn scalar_record_roundtrip_and_all_four_site_derivatives() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(384))
        .unwrap()
        .run(|| {
            let (value, options) = fixture();
            if let Ok(path) = std::env::var("DFT_RS_SCALAR_FIXTURE") {
                std::fs::write(path, value.to_json_pretty()).unwrap();
            }
            let record = ScalarActionRecord::from_value(&value).unwrap();
            let roundtrip = ScalarActionRecord::from_json(&record.to_json()).unwrap();
            assert_eq!(roundtrip.endpoint_ids(), record.endpoint_ids());
            let q = [0.23, 0.31, 0.27, 0.18];
            let weights =
                |q: [f64; 4]| vec![1. - q[0] - q[1], q[0], q[1], 1. - q[2] - q[3], q[2], q[3]];
            let base = record.calculate(&weights(q), &options, None).unwrap();
            let analytic = base
                .get("weight_gradient")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>();
            let names = (0..4)
                .map(|i| format!("saved full scalar site gradient {i}"))
                .collect::<Vec<_>>();
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    (0..4)
                        .map(|i| {
                            std::array::from_fn(|k| {
                                let h = steps[k];
                                let energy = |d: f64| {
                                    let mut trial = q;
                                    trial[i] += d;
                                    number(
                                        &roundtrip
                                            .calculate(&weights(trial), &options, None)
                                            .unwrap(),
                                        "energy",
                                    )
                                };
                                let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                    (energy(2. * h), energy(-2. * h))
                                } else {
                                    (0., 0.)
                                };
                                (stencil.first(energy(h), energy(-h), p2, m2, h) - analytic[i])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("every independent two-site scalar gradient", &failures);
            let m = record.primary_molecule();
            let system = System {
                numbers: m.atoms.iter().map(|a| a.z).collect(),
                positions: m.atoms.iter().map(|a| a.position.to_array()).collect(),
                cell: None,
                pbc: [false; 3],
            };
            let controls = Controls::from_value(
                &Value::object()
                    .with("fragment_mode", "scalar-evaluate")
                    .with("fragment_model_json", record.to_json())
                    .with("fragment_weights", weights(q)),
            )
            .unwrap();
            let report =
                dft_rs::driver::continuous_species::scalar::run(&system, &controls).unwrap();
            assert!((number(&report, "energy") - number(&base, "energy")).abs() < 1e-11);
            for strategy in ["joint", "one-pass", "cyclic"] {
                let output = record
                    .calculate(
                        &weights(q),
                        &options,
                        Some(ScalarOptimization {
                            tolerance: 1e-4,
                            max_iterations: 8,
                            strategy: strategy.into(),
                        }),
                    )
                    .unwrap();
                assert!(number(&output, "energy") <= number(&base, "energy") + 1e-10);
                println!(
                    "SCALAR_OPTIMIZER {strategy} {}",
                    output.get("weight_optimization").unwrap().to_json_pretty()
                );
            }
            Ok(())
        })
        .unwrap();
}
