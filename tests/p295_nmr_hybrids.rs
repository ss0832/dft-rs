// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! WC04/WP04, explicit VWN-RPA convention: independent point/term oracles,
//! every active density derivative and every C1 molecular force component.
//! Five-molecule checks also close the separate BLYP35 catalogue gap.

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
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::xc::{
    resolve, Family, FunctionalSpec, GridVars, HybridParams, ResolvedFunctional, XcOut,
};
use dft_rs::{Atom, Molecule, Vec3};

const NAMES: [&str; 2] = ["wc04", "wp04"];
fn rows(value: &Value) -> &[Value] {
    let Value::Array(v) = value else {
        panic!("array expected")
    };
    v
}
fn number(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}
fn array(value: &Value, key: &str) -> Vec<f64> {
    value.get(key).unwrap().numeric_array().unwrap().1
}
fn oracle() -> Value {
    Value::parse_json(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle_data/nmr_hybrids.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
fn functional(name: &str) -> ResolvedFunctional {
    resolve(&FunctionalSpec::Combined(name.into())).unwrap()
}
fn evaluate(f: &ResolvedFunctional, point: &[f64], order: usize) -> XcOut {
    let p: Vec<_> = point.iter().map(|v| [*v]).collect();
    let vars = GridVars {
        rho: [&p[0], &p[1]],
        sigma: Some([&p[2], &p[3], &p[4]]),
        tau: Some([&p[5], &p[6]]),
    };
    let mut out = XcOut::default();
    f.eval(&vars, order, &mut out).unwrap();
    out
}
fn vector(out: &XcOut) -> [f64; 6] {
    [
        out.e[0],
        out.vrho[0][0],
        out.vrho[1][0],
        out.vsigma[0][0],
        out.vsigma[1][0],
        out.vsigma[2][0],
    ]
}
fn molecule(entry: &Value, charge: i32, multiplicity: usize) -> Molecule {
    Molecule::new(
        array(entry, "numbers")
            .iter()
            .zip(rows(entry.get("positions").unwrap()))
            .map(|(z, r)| {
                let p = r.numeric_array().unwrap().1;
                Atom {
                    z: *z as u8,
                    position: Vec3::new(p[0], p[1], p[2]),
                }
            })
            .collect(),
    )
    .with_charge(charge)
    .with_multiplicity(multiplicity)
}
fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    }
}

#[test]
fn pointwise_components_potentials_and_full_kernel_match_frozen_oracles() {
    let oracle = oracle();
    for name in NAMES {
        let model = oracle.get("functionals").unwrap().get(name).unwrap();
        let f = functional(name);
        let mut worst = [0.0_f64; 4];
        assert_eq!(
            f.hybrid,
            HybridParams::global(number(model, "exact_exchange"))
        );
        assert_eq!(f.family, Family::Gga);
        assert!(f.required_nonlocal_kind().is_none());
        assert_eq!(f.parts.len(), 4);
        let labels = ["lda_exchange", "b88_exchange", "vwn_rpa", "lyp"];
        for (i, label) in labels.iter().enumerate() {
            assert!((f.parts[i].0 - number(model.get("weights").unwrap(), label)).abs() < 1e-15);
        }
        for (index, point) in rows(oracle.get("points").unwrap()).iter().enumerate() {
            let p = point.numeric_array().unwrap().1;
            let out = evaluate(&f, &p, if p[1] > 0.0 { 2 } else { 1 });
            let expected = &rows(model.get("values").unwrap())[index];
            worst[0] = worst[0].max((out.e[0] - number(expected, "energy")).abs());
            assert!(
                (out.e[0] - number(expected, "energy")).abs() < 1e-11,
                "{name} point {index}"
            );
            let actual = vector(&out);
            let potential = array(expected, "potential");
            for k in 0..5 {
                if p[1] == 0.0 && [1, 3, 4].contains(&k) {
                    continue;
                }
                worst[1] = worst[1].max((actual[k + 1] - potential[k]).abs());
                assert!(
                    (actual[k + 1] - potential[k]).abs() < 1e-10 * (1.0 + potential[k].abs()),
                    "{name} point {index} potential {k}"
                );
            }
            for (i, label) in labels.iter().enumerate() {
                let part = ResolvedFunctional::custom(
                    *label,
                    vec![(1.0, f.parts[i].1.clone())],
                    HybridParams::PURE,
                    f.parts[i].1.family(),
                );
                let component = evaluate(&part, &p, 1);
                let expected = &rows(oracle.get("components").unwrap().get(label).unwrap())[index];
                worst[2] = worst[2].max((component.e[0] - number(expected, "energy")).abs());
                assert!(
                    (component.e[0] - number(expected, "energy")).abs() < 1e-11,
                    "{name} {label}"
                );
            }
            if let Some(expected) = expected.get("kernel") {
                let (_, kernel) = out.potential_and_kernel(0, f.family).unwrap();
                for (i, row) in rows(expected).iter().enumerate() {
                    for (j, value) in row.numeric_array().unwrap().1.iter().enumerate() {
                        worst[3] = worst[3].max((kernel[i][j] - value).abs());
                        assert!(
                            (kernel[i][j] - value).abs() < 1e-9 * (1.0 + value.abs()),
                            "{name} point {index} kernel [{i}][{j}]"
                        );
                    }
                }
            }
        }
        println!("{name} maximum frozen-point errors [total energy density, potential, component energy density, kernel] = {worst:?}");
    }
}

#[test]
fn every_active_density_derivative_has_independent_four_step_fits() {
    // Five linearly independent dense directions D=I+u*1^T. Inverting its
    // directional differences recovers every partial separately. A generic
    // rho/sigma mixture prevents a linear sigma term's exact stencil from
    // hiding its convergence order behind roundoff. Frozen kernels above
    // independently fix even the small 1e-4-weighted LYP entries.
    let u = [0.23, -0.17, 0.11, 0.13, -0.19];
    let denominator = 1.0 + u.iter().sum::<f64>();
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let mut failures = Vec::new();
    for name in NAMES {
        let f = functional(name);
        let (potential, kernel) = evaluate(&f, &point, 2)
            .potential_and_kernel(0, f.family)
            .unwrap();
        let mut labels = Vec::new();
        let mut targets = Vec::new();
        for j in 0..5 {
            labels.push(format!("{name} potential {j}"));
            targets.push((0, j, potential[j]));
            for i in 0..5 {
                if i >= 2 && j >= 2 && !(i == j && i != 3) {
                    assert_eq!(kernel[i][j], 0.0);
                    continue;
                }
                labels.push(format!("{name} kernel [{i}][{j}]"));
                targets.push((i + 1, j, kernel[i][j]));
            }
        }
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                ([0.012, 0.006, 0.003, 0.0015], stencils::Stencil::Second),
                ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; targets.len()];
                for (step, &h) in steps.iter().enumerate() {
                    let mut directional = [[0.0; 5]; 6];
                    for j in 0..5 {
                        let at = |t: f64| {
                            let mut p = point;
                            for k in 0..5 {
                                p[k] += t * (u[k] + if k == j { 1.0 } else { 0.0 });
                            }
                            assert!(p[0] > 0.0 && p[1] > 0.0);
                            assert!(p[2] >= 0.0 && p[4] >= 0.0 && p[3] * p[3] <= p[2] * p[4]);
                            vector(&evaluate(&f, &p, 1))
                        };
                        let plus = at(h);
                        let minus = at(-h);
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            ([0.0; 6], [0.0; 6])
                        };
                        for i in 0..6 {
                            directional[i][j] =
                                stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                        }
                    }
                    for (k, &(i, j, expected)) in targets.iter().enumerate() {
                        let common = u
                            .iter()
                            .zip(directional[i])
                            .map(|(u, d)| u * d)
                            .sum::<f64>()
                            / denominator;
                        errors[k][step] = (directional[i][j] - common - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        assert_eq!(&potential[5..], &[0.0, 0.0]);
        for i in 0..7 {
            for j in 5..7 {
                assert_eq!(kernel[i][j], 0.0);
                assert_eq!(kernel[j][i], 0.0);
            }
        }
    }
    components::assert_no_failures("WC04/WP04 density derivatives", &failures);
}

#[test]
fn physical_limits_preserve_the_published_independent_coefficients() {
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let swapped = [
        point[1], point[0], point[4], point[3], point[2], point[6], point[5],
    ];
    for name in NAMES {
        let f = functional(name);
        let a = evaluate(&f, &point, 2);
        let b = evaluate(&f, &swapped, 2);
        assert!((a.e[0] - b.e[0]).abs() < 1e-14);
        let (va, ka) = a.potential_and_kernel(0, f.family).unwrap();
        let (vb, kb) = b.potential_and_kernel(0, f.family).unwrap();
        let permutation = [1, 0, 4, 3, 2, 6, 5];
        for i in 0..7 {
            assert!((va[i] - vb[permutation[i]]).abs() < 1e-13);
            for j in 0..7 {
                assert!((ka[i][j] - kb[permutation[i]][permutation[j]]).abs() < 1e-12);
            }
        }
        assert_eq!(evaluate(&f, &[0.0; 7], 0).e[0], 0.0);
        let x = ResolvedFunctional::custom(
            "uniform exchange limit",
            f.parts[..2].to_vec(),
            HybridParams::PURE,
            Family::Gga,
        );
        let scale = f.parts[0].0 + f.parts[1].0;
        assert!((scale - if name == "wc04" { 0.0001 } else { 0.9999 }).abs() < 1e-15);
        for rho in [0.001_f64, 0.1, 1.0, 10.0] {
            let expected = scale
                * (-0.75 * (6.0 / std::f64::consts::PI).cbrt())
                * (rho.powf(4.0 / 3.0) + (0.73 * rho).powf(4.0 / 3.0));
            assert!(
                (evaluate(&x, &[rho, 0.73 * rho, 0.0, 0.0, 0.0, 0.4, 0.3], 0).e[0] - expected)
                    .abs()
                    < 1e-13 * (1.0 + expected.abs())
            );
        }
    }
}

#[test]
fn restricted_unrestricted_scf_and_each_fixed_density_term_match_oracles() {
    let oracle = oracle();
    let set = data::load("sto-3g").unwrap();
    for name in NAMES {
        let f = functional(name);
        let model = oracle.get("functionals").unwrap().get(name).unwrap();
        for entry in rows(model.get("scf").unwrap()) {
            let mol = molecule(
                entry,
                number(entry, "charge") as i32,
                number(entry, "multiplicity") as usize,
            );
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
            let scf = run_scf(&mol, &basis, &options(), &f, Some(&grid)).unwrap();
            assert!(scf.converged);
            let defect = scf.energy - number(entry, "total");
            let charge = scf.grid_electron_count.unwrap() - mol.n_electrons().unwrap() as f64;
            println!(
                "{name} {} SCF energy {:.14e}, defect {defect:.6e}, electron defect {charge:.6e}",
                entry.get("molecule").unwrap().as_str().unwrap(),
                scf.energy
            );
            assert!(defect.abs() < 5e-7);
            assert!(charge.abs() < 1e-7);
            let density: Vec<_> = rows(entry.get("density").unwrap())
                .iter()
                .map(|s| {
                    Matrix::from_row_major(
                        basis.nao_sph,
                        basis.nao_sph,
                        rows(s)
                            .iter()
                            .flat_map(|r| r.numeric_array().unwrap().1)
                            .collect(),
                    )
                })
                .collect();
            let mut total = density[0].clone();
            for (a, b) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
                *a += b;
            }
            let core = kinetic_matrix(&basis).frobenius_dot(&total)
                + nuclear_matrix(&basis, &mol, None).frobenius_dot(&total);
            let eri = eri_incore(&basis, CoulombKernel::Coulomb, 512).unwrap();
            let mut hartree = 0.0;
            let mut exact = 0.0;
            for i in 0..basis.nao_sph {
                for j in 0..basis.nao_sph {
                    for k in 0..basis.nao_sph {
                        for l in 0..basis.nao_sph {
                            let integral = eri.get(i, j, k, l);
                            hartree += 0.5 * total[(i, j)] * total[(k, l)] * integral;
                            for spin in &density {
                                exact -= 0.5 * spin[(i, k)] * spin[(j, l)] * integral;
                            }
                        }
                    }
                }
            }
            let mut terms = vec![
                ("core", core, 1e-9),
                ("coulomb", hartree, 1e-9),
                ("hf_exchange", exact, 1e-9),
            ];
            let mut sum = scf.e_nuc + core + hartree + f.hybrid.cx * exact;
            for (label, (weight, part)) in ["lda_exchange", "b88_exchange", "vwn_rpa", "lyp"]
                .into_iter()
                .zip(&f.parts)
            {
                let component = ResolvedFunctional::custom(
                    label,
                    vec![(1.0, part.clone())],
                    HybridParams::PURE,
                    part.family(),
                );
                let energy = dft_rs::ks::xc_contribution(
                    &component,
                    &basis,
                    &grid,
                    [&density[0], &density[1]],
                )
                .unwrap()
                .exc;
                sum += weight * energy;
                terms.push((label, energy, 5e-7));
            }
            for (label, value, tolerance) in terms {
                let error = (value - number(entry, label)).abs();
                println!("{name} {label} fixed-density error {error:.6e}");
                assert!(error < tolerance, "{name} {label}");
            }
            assert!((sum - number(entry, "total")).abs() < 5e-7);
        }
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
    for name in NAMES {
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
        let gradient = array(&report, "gradient");
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
    components::assert_no_failures("WC04/WP04 C1 gradients", &failures);
}

fn check_five_molecules(name: &str) {
    let oracle = Value::parse_json(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle_data/catalogue_hybrid_molecules.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let set = data::load("sto-3g").unwrap();
    let f = functional(name);
    let case = oracle.get("functionals").unwrap().get(name).unwrap();
    for label in ["h2", "lih", "water", "phosphine", "ferrocene"] {
        let entry = oracle.get("molecules").unwrap().get(label).unwrap();
        let mol = molecule(entry, 0, 1);
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
        let scf = run_scf(&mol, &basis, &options(), &f, Some(&grid)).unwrap();
        assert!(scf.converged);
        let error = scf.energy - number(case.get("energies").unwrap(), label);
        let tolerance = if label == "ferrocene" { 1e-5 } else { 5e-7 };
        println!(
            "{name} {label}: energy {:.14e}, oracle error {error:.6e}, tolerance {tolerance:.1e}",
            scf.energy
        );
        assert!(error.abs() < tolerance, "{name} {label}");
    }
}

#[test]
fn catalogue_blyp35_five_molecules() {
    check_five_molecules("blyp35");
}

#[test]
fn catalogue_wc04_five_molecules() {
    check_five_molecules("wc04");
}

#[test]
fn catalogue_wp04_five_molecules() {
    check_five_molecules("wp04");
}
