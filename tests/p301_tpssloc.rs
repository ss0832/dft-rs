// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Original TPSSloc component equations, complete local first/second partials,
//! spin/kinetic limits and independent numerical kernels.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::xc::{
    resolve, tpss, Family, FunctionalSpec, GridVars, HybridParams, ResolvedFunctional,
    XcFunctional, XcOut,
};
use dft_rs::{Atom, Molecule, Vec3};
use std::sync::Arc;

fn providers() -> [(&'static str, Box<dyn XcFunctional>); 2] {
    [
        ("exchange", tpss::tpss_x()),
        ("correlation", tpss::tpssloc_c()),
    ]
}

fn evaluate(f: &dyn XcFunctional, p: &[f64], order: usize) -> XcOut {
    let v: Vec<_> = p.iter().map(|&x| [x]).collect();
    let vars = GridVars {
        rho: [&v[0], &v[1]],
        sigma: Some([&v[2], &v[3], &v[4]]),
        tau: Some([&v[5], &v[6]]),
    };
    let mut output = XcOut::default();
    output.prepare(1, order, Family::MetaGga);
    f.eval_add(&vars, order, 1.0, &mut output).unwrap();
    output
}

fn array(v: &Value) -> Vec<f64> {
    v.numeric_array().unwrap().1
}

fn rows(v: &Value) -> &[Value] {
    let Value::Array(values) = v else {
        panic!("array expected")
    };
    values
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}

#[test]
fn frozen_component_energies_potentials_and_every_active_kernel_entry() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/tpssloc.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let Value::Array(points) = reference.get("points").unwrap() else {
        panic!("points")
    };
    for (name, provider) in providers() {
        let Value::Array(values) = reference.get("components").unwrap().get(name).unwrap() else {
            panic!("component values")
        };
        let mut maxima = [0.0_f64; 3];
        for (index, (point, expected)) in points.iter().zip(values).enumerate() {
            let point = array(point);
            let result = evaluate(provider.as_ref(), &point, 2);
            let energy = expected.get("energy").unwrap().as_f64().unwrap();
            maxima[0] = maxima[0].max((result.e[0] - energy).abs());
            assert!(
                (result.e[0] - energy).abs() < 1e-11 * (1.0 + energy.abs()),
                "{name}/{index} energy"
            );
            let (potential, kernel) = result.potential_and_kernel(0, Family::MetaGga).unwrap();
            let expected_potential = array(expected.get("potential").unwrap());
            for i in 0..7 {
                if point[1] == 0.0 && [1, 3, 4, 6].contains(&i) {
                    continue;
                }
                let error = (potential[i] - expected_potential[i]).abs();
                maxima[1] = maxima[1].max(error);
                assert!(
                    error < 1e-10 * (1.0 + expected_potential[i].abs()),
                    "{name}/{index} potential{i}: {error:e}"
                );
            }
            if let Some(Value::Array(rows)) = expected.get("kernel") {
                for i in 0..7 {
                    let row = array(&rows[i]);
                    for j in 0..7 {
                        let error = (kernel[i][j] - row[j]).abs();
                        maxima[2] = maxima[2].max(error);
                        assert!(
                            error < 1e-9 * (1.0 + row[j].abs()),
                            "{name}/{index} kernel{i},{j}: {error:e}"
                        );
                    }
                }
            }
        }
        println!("TPSSloc {name} maximum [energy,potential,kernel] errors {maxima:?}");
    }
}

#[test]
fn combined_recipe_matches_independent_energy_potential_and_kernel() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/tpssloc.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let f = resolve(&FunctionalSpec::Combined("tpssloc".into())).unwrap();
    let mut maxima = [0.0_f64; 3];
    for (point, expected) in rows(reference.get("points").unwrap()).iter().zip(rows(
        reference
            .get("components")
            .unwrap()
            .get("combined")
            .unwrap(),
    )) {
        let p = array(point);
        let v: Vec<_> = p.iter().map(|&x| [x]).collect();
        let vars = GridVars {
            rho: [&v[0], &v[1]],
            sigma: Some([&v[2], &v[3], &v[4]]),
            tau: Some([&v[5], &v[6]]),
        };
        let mut result = XcOut::default();
        f.eval(&vars, 2, &mut result).unwrap();
        let error = (result.e[0] - number(expected, "energy")).abs();
        maxima[0] = maxima[0].max(error);
        assert!(error < 1e-11 * (1.0 + number(expected, "energy").abs()));
        let (potential, kernel) = result.potential_and_kernel(0, Family::MetaGga).unwrap();
        let expected_potential = array(expected.get("potential").unwrap());
        for i in 0..7 {
            if p[1] == 0.0 && [1, 3, 4, 6].contains(&i) {
                continue;
            }
            let error = (potential[i] - expected_potential[i]).abs();
            maxima[1] = maxima[1].max(error);
            assert!(error < 1e-10 * (1.0 + expected_potential[i].abs()));
        }
        if let Some(Value::Array(expected_kernel)) = expected.get("kernel") {
            for i in 0..7 {
                let row = array(&expected_kernel[i]);
                for j in 0..7 {
                    let error = (kernel[i][j] - row[j]).abs();
                    maxima[2] = maxima[2].max(error);
                    assert!(error < 1e-9 * (1.0 + row[j].abs()));
                }
            }
        }
    }
    println!("TPSSloc combined maximum [energy,potential,kernel] errors {maxima:?}");
}

#[test]
fn every_nonzero_component_partial_has_four_step_error_fits() {
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let u = [0.23, -0.17, 0.11, 0.13, -0.19, 0.07, -0.09];
    let denominator = 1.0 + u.iter().sum::<f64>();
    let mut failures = Vec::new();
    for (name, provider) in providers() {
        let output = evaluate(provider.as_ref(), &point, 2);
        let (potential, kernel) = output.potential_and_kernel(0, Family::MetaGga).unwrap();
        let exchange_active = |i| [0, 1, 2, 4, 5, 6].contains(&i);
        let same_spin = |i, j| {
            ([0, 2, 5].contains(&i) && [0, 2, 5].contains(&j))
                || ([1, 4, 6].contains(&i) && [1, 4, 6].contains(&j))
        };
        let mut targets = Vec::new();
        let mut labels = Vec::new();
        for j in 0..7 {
            if name != "exchange" || exchange_active(j) {
                targets.push((0, j, potential[j]));
                labels.push(format!("TPSSloc {name} potential{j}"));
            } else {
                assert_eq!(potential[j], 0.0);
            }
            for i in 0..=j {
                if name != "exchange" || same_spin(i, j) {
                    targets.push((i + 1, j, kernel[i][j]));
                    labels.push(format!("TPSSloc {name} kernel{i},{j}"));
                } else {
                    assert_eq!(kernel[i][j], 0.0);
                    assert_eq!(kernel[j][i], 0.0);
                }
            }
        }
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                ([0.004, 0.002, 0.001, 0.0005], stencils::Stencil::Second),
                ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; targets.len()];
                for (step, &h) in steps.iter().enumerate() {
                    let mut directions = [[0.0; 7]; 8];
                    for j in 0..7 {
                        let at = |scale: f64| {
                            let mut displaced = point;
                            for k in 0..7 {
                                displaced[k] += scale * (u[k] + if j == k { 1.0 } else { 0.0 });
                            }
                            assert!(displaced[0] > 0.0 && displaced[1] > 0.0);
                            assert!(
                                displaced[2] >= 0.0
                                    && displaced[4] >= 0.0
                                    && displaced[3].powi(2) < displaced[2] * displaced[4]
                            );
                            for (r, s, t) in [(0, 2, 5), (1, 4, 6)] {
                                assert!(displaced[t] > displaced[s] / (8.0 * displaced[r]));
                            }
                            let value = evaluate(provider.as_ref(), &displaced, 2);
                            let (potential, _) =
                                value.potential_and_kernel(0, Family::MetaGga).unwrap();
                            let mut result = [0.0; 8];
                            result[0] = value.e[0];
                            result[1..].copy_from_slice(&potential);
                            result
                        };
                        let plus = at(h);
                        let minus = at(-h);
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            ([0.0; 8], [0.0; 8])
                        };
                        for i in 0..8 {
                            directions[i][j] =
                                stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                        }
                    }
                    for (k, &(i, j, expected)) in targets.iter().enumerate() {
                        let shared = u.iter().zip(directions[i]).map(|(u, d)| u * d).sum::<f64>()
                            / denominator;
                        errors[k][step] = (directions[i][j] - shared - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("TPSSloc local component derivatives", &failures);
}

#[test]
fn spin_scaling_uniform_exchange_one_electron_and_registry_contracts() {
    for (name, provider) in providers() {
        assert_eq!(provider.max_analytic_order(), 3);
        assert_eq!(provider.params().len(), provider.param_names().len());
        let point = [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21];
        let values: Vec<_> = point.iter().map(|&x| [x]).collect();
        let vars = GridVars {
            rho: [&values[0], &values[1]],
            sigma: Some([&values[2], &values[3], &values[4]]),
            tau: Some([&values[5], &values[6]]),
        };
        let mut unsupported = XcOut::default();
        unsupported.prepare(1, 3, Family::MetaGga);
        assert!(matches!(
            provider.eval_add(&vars, 4, 1.0, &mut unsupported),
            Err(dft_rs::DftError::DerivOrderUnsupported {
                requested: 4,
                max_analytic: 3,
                ..
            })
        ));
        let permutation = [1, 0, 4, 3, 2, 6, 5];
        let flipped: Vec<_> = permutation.iter().map(|&k| point[k]).collect();
        let a = evaluate(provider.as_ref(), &point, 2);
        let b = evaluate(provider.as_ref(), &flipped, 2);
        assert!((a.e[0] - b.e[0]).abs() < 1e-13, "{name}");
        let (pa, ka) = a.potential_and_kernel(0, Family::MetaGga).unwrap();
        let (pb, kb) = b.potential_and_kernel(0, Family::MetaGga).unwrap();
        for i in 0..7 {
            assert!((pa[i] - pb[permutation[i]]).abs() < 1e-12);
            for j in 0..7 {
                assert!((ka[i][j] - kb[permutation[i]][permutation[j]]).abs() < 1e-10);
            }
        }
        assert_eq!(evaluate(provider.as_ref(), &[0.0; 7], 0).e[0], 0.0);
    }
    for rho in [0.001_f64, 0.1, 1.0, 10.0] {
        let tau = 0.3 * (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0) * rho.powf(5.0 / 3.0);
        let point = [rho, rho, 0.0, 0.0, 0.0, tau, tau];
        let expected = -1.5 * (6.0 / std::f64::consts::PI).cbrt() * rho.powf(4.0 / 3.0);
        assert!(
            (evaluate(tpss::tpss_x().as_ref(), &point, 0).e[0] - expected).abs()
                < 1e-12 * (1.0 + expected.abs())
        );
        let sigma = rho * rho * 0.04;
        let one_electron = [rho, 0.0, sigma, 0.0, 0.0, sigma / (8.0 * rho), 0.0];
        assert!(evaluate(tpss::tpssloc_c().as_ref(), &one_electron, 0).e[0].abs() < 1e-13);
    }
    let combined = resolve(&FunctionalSpec::Combined("T_P-S S loc".into())).unwrap();
    assert_eq!(combined.family, Family::MetaGga);
    assert_eq!(combined.hybrid.cx, 0.0);
    assert_eq!(combined.hybrid.clr, 0.0);
    assert_eq!(combined.parts.len(), 2);
    assert!(combined.required_nonlocal_kind().is_none());
}

#[test]
fn rks_uks_scf_and_fixed_density_exchange_correlation_terms() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/tpssloc.json");
    let reference = Value::parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    let set = data::load("sto-3g").unwrap();
    for entry in rows(reference.get("scf").unwrap()) {
        let numbers = array(entry.get("numbers").unwrap());
        let positions: Vec<_> = rows(entry.get("positions").unwrap())
            .iter()
            .map(|r| {
                let r = array(r);
                [r[0], r[1], r[2]]
            })
            .collect();
        let charge = number(entry, "charge") as i32;
        let multiplicity = number(entry, "multiplicity") as usize;
        let mol = Molecule::new(
            numbers
                .iter()
                .zip(&positions)
                .map(|(z, p)| Atom {
                    z: *z as u8,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        )
        .with_charge(charge)
        .with_multiplicity(multiplicity);
        let basis = MolBasis::build(&mol, &set).unwrap();
        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 9,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let density: Vec<_> = rows(entry.get("density").unwrap())
            .iter()
            .map(|spin| {
                Matrix::from_row_major(
                    basis.nao_sph,
                    basis.nao_sph,
                    rows(spin).iter().flat_map(array).collect(),
                )
            })
            .collect();
        let mut total = density[0].clone();
        for (value, beta) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
            *value += beta;
        }
        let core = kinetic_matrix(&basis).frobenius_dot(&total)
            + nuclear_matrix(&basis, &mol, None).frobenius_dot(&total);
        let eri = eri_incore(&basis, CoulombKernel::Coulomb, 512).unwrap();
        let mut hartree = 0.0;
        for i in 0..basis.nao_sph {
            for j in 0..basis.nao_sph {
                for k in 0..basis.nao_sph {
                    for l in 0..basis.nao_sph {
                        hartree += 0.5 * total[(i, j)] * total[(k, l)] * eri.get(i, j, k, l);
                    }
                }
            }
        }
        for (name, value) in [("core", core), ("coulomb", hartree)] {
            let error = value - number(entry, name);
            println!("TPSSloc multiplicity{multiplicity} fixed-density {name} error {error:e}");
            assert!(error.abs() < 1e-9);
        }
        let mut reconstruction = mol.nuclear_repulsion() + core + hartree;
        for (name, provider) in providers() {
            let f = ResolvedFunctional::custom(
                name,
                vec![(1.0, Arc::from(provider))],
                HybridParams::PURE,
                Family::MetaGga,
            );
            let actual = dft_rs::ks::xc_contribution(&f, &basis, &grid, [&density[0], &density[1]])
                .unwrap()
                .exc;
            let error = actual - number(entry, name);
            reconstruction += actual;
            println!("TPSSloc multiplicity{multiplicity} fixed-density {name} error {error:e}");
            assert!(error.abs() < 5e-7);
        }
        assert!((reconstruction - number(entry, "total")).abs() < 5e-7);
        let system = System::molecule(numbers.iter().map(|z| *z as u8).collect(), positions);
        // Populations belong to the controls of the common public driver.
        let controls = Controls::from_value(&Value::Object(vec![
            ("xc".into(), Value::Str("tpssloc".into())),
            ("basis".into(), Value::Str("sto-3g".into())),
            ("charge".into(), Value::Int(charge as i64)),
            ("multiplicity".into(), Value::Int(multiplicity as i64)),
            ("unrestricted".into(), Value::Bool(multiplicity == 2)),
            ("grid_level".into(), Value::Int(9)),
            ("grid_weight_cutoff".into(), Value::Float(0.0)),
            ("conv_tol".into(), Value::Float(1e-12)),
            ("conv_tol_dp".into(), Value::Float(1e-10)),
            ("conv_tol_grad".into(), Value::Float(1e-10)),
        ]))
        .unwrap();
        let result = run("energy", &system, &controls).unwrap();
        let error = number(&result, "energy") - number(entry, "total");
        println!("TPSSloc multiplicity{multiplicity} SCF error {error:e}");
        assert!(error.abs() < 5e-7);
    }
}

#[test]
fn every_c1_molecular_gradient_component_has_four_step_fits() {
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.43, 0.81, -0.62],
            [-0.63, 1.59, -0.43],
            [-1.12, -1.36, -0.71],
        ],
    );
    let mut failures = Vec::new();
    {
        let name = "tpssloc";
        let controls = Controls::from_value(&Value::Object(vec![
            ("xc".into(), Value::Str(name.into())),
            ("basis".into(), Value::Str("sto-3g".into())),
            ("grid_level".into(), Value::Int(5)),
            ("grid_weight_cutoff".into(), Value::Float(0.0)),
            ("conv_tol".into(), Value::Float(1e-13)),
            ("conv_tol_grad".into(), Value::Float(1e-11)),
            ("conv_tol_dp".into(), Value::Float(1e-11)),
            ("dispersion".into(), Value::Str("none".into())),
        ]))
        .unwrap();
        let report = run("gradient", &system, &controls).unwrap();
        let gradient = array(report.get("gradient").unwrap());
        assert_eq!(gradient.len(), 12);
        for coordinate in 0..12 {
            stencils::fit_components_staged(
                &[format!("{name} C1 gradient {coordinate}")],
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let energy = |h: f64| {
                        let mut s = system.clone();
                        s.positions[coordinate / 3][coordinate % 3] += h;
                        number(&run("energy", &s, &controls).unwrap(), "energy")
                    };
                    let mut errors = [0.0; 4];
                    for (i, &h) in steps.iter().enumerate() {
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (energy(2.0 * h), energy(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        errors[i] = (stencil.first(energy(h), energy(-h), p2, m2, h)
                            - gradient[coordinate])
                            .abs();
                    }
                    vec![errors]
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("TPSSloc C1 gradients", &failures);
}
