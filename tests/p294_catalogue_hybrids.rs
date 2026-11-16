// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent numerical and full-component acceptance for BLYP35.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::xc::{resolve, Family, FunctionalSpec, GridVars, ResolvedFunctional, XcOut};

fn functional() -> ResolvedFunctional {
    resolve(&FunctionalSpec::Combined("blyp35".into())).unwrap()
}

fn evaluate(functional: &ResolvedFunctional, point: &[f64], order: usize) -> XcOut {
    let data: Vec<_> = point.iter().map(|v| [*v]).collect();
    let vars = GridVars {
        rho: [&data[0], &data[1]],
        sigma: Some([&data[2], &data[3], &data[4]]),
        tau: Some([&data[5], &data[6]]),
    };
    let mut out = XcOut::default();
    functional.eval(&vars, order, &mut out).unwrap();
    out
}

fn rows(value: &Value) -> &[Value] {
    let Value::Array(rows) = value else {
        panic!("array expected")
    };
    rows
}

#[test]
fn pointwise_energy_and_each_potential_match_independent_components() {
    let oracle = Value::parse_json(include_str!("oracle_data/catalogue_hybrids.json")).unwrap();
    let functional = functional();
    assert_eq!(functional.family, Family::Gga);
    assert_eq!(functional.hybrid.cx, 0.35);
    assert_eq!(functional.hybrid.clr, 0.0);
    assert_eq!(functional.hybrid.omega, 0.0);
    assert!(functional.required_nonlocal_kind().is_none());
    let components = oracle.get("components").unwrap();
    let exchange = rows(components.get("exchange").unwrap());
    let correlation = rows(components.get("correlation").unwrap());
    for (index, point) in rows(oracle.get("points").unwrap()).iter().enumerate() {
        let p = point.numeric_array().unwrap().1;
        let out = evaluate(&functional, &p, 1);
        let energy = |entry: &Value| entry.get("energy").unwrap().as_f64().unwrap();
        let expected = 0.65 * energy(&exchange[index]) + energy(&correlation[index]);
        assert!((out.e[0] - expected).abs() < 1e-11, "point {index} energy");
        let x = exchange[index]
            .get("potential")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let c = correlation[index]
            .get("potential")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let actual = [
            out.vrho[0][0],
            out.vrho[1][0],
            out.vsigma[0][0],
            out.vsigma[1][0],
            out.vsigma[2][0],
        ];
        for k in 0..5 {
            // Potentials of an empty spin channel have no physical definition.
            if (k == 0 || k == 2) && p[0] == 0.0
                || (k == 1 || k == 4) && p[1] == 0.0
                || k == 3 && p[0] * p[1] == 0.0
            {
                continue;
            }
            let expected = 0.65 * x[k] + c[k];
            assert!(
                (actual[k] - expected).abs() < 1e-10 * (1.0 + expected.abs()),
                "point {index} component {k}"
            );
        }
    }
    for alias in ["BLYP35", "blyp-35", "blyp_35", "blyp 35"] {
        assert_eq!(
            resolve(&FunctionalSpec::Combined(alias.into()))
                .unwrap()
                .hybrid,
            functional.hybrid
        );
    }
}

#[test]
fn spin_interchange_and_uniform_density_exchange_limits() {
    let f = functional();
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let swapped = [
        point[1], point[0], point[4], point[3], point[2], point[6], point[5],
    ];
    let a = evaluate(&f, &point, 1);
    let b = evaluate(&f, &swapped, 1);
    assert!((a.e[0] - b.e[0]).abs() < 1e-14);
    for spin in 0..2 {
        assert!((a.vrho[spin][0] - b.vrho[1 - spin][0]).abs() < 1e-14);
    }
    for rho in [0.001_f64, 0.1, 1.0, 10.0] {
        let uniform = [rho, 0.73 * rho, 0.0, 0.0, 0.0, 0.4, 0.3];
        let exchange = dft_rs::xc::gga::b88();
        let part = ResolvedFunctional::custom(
            "exchange limit",
            vec![(1.0, exchange.into())],
            dft_rs::xc::HybridParams::PURE,
            Family::Gga,
        );
        let actual = evaluate(&part, &uniform, 0).e[0];
        let factor = -0.75 * (6.0 / std::f64::consts::PI).cbrt();
        let dirac = factor * (rho.powf(4.0 / 3.0) + (0.73 * rho).powf(4.0 / 3.0));
        assert!((actual - dirac).abs() < 1e-13 * (1.0 + dirac.abs()));
    }
    assert_eq!(evaluate(&f, &[0.0; 7], 0).e[0], 0.0);
}

#[test]
fn all_density_potential_and_kernel_components_have_independent_error_fits() {
    let f = functional();
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let (potential, kernel) = evaluate(&f, &point, 2)
        .potential_and_kernel(0, f.family)
        .unwrap();
    let mut names = Vec::new();
    let mut targets = Vec::new();
    for j in 0..5 {
        names.push(format!("BLYP35 potential {j}"));
        targets.push((j, 0, potential[j]));
        for i in 0..5 {
            // LYP is linear in sigma and B88 separates the spin channels.
            // These seven mixed sigma entries vanish identically, rather
            // than possessing a nonzero truncation error to fit.
            if i >= 2 && j >= 2 && !(i == j && i != 3) {
                assert_eq!(kernel[i][j], 0.0);
                continue;
            }
            names.push(format!("BLYP35 kernel [{i}][{j}]"));
            targets.push((j, i + 1, kernel[i][j]));
        }
    }
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
            ([0.002, 0.001, 0.0005, 0.00025], stencils::Stencil::Second),
            ([0.006, 0.003, 0.0015, 0.00075], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (step, &h) in steps.iter().enumerate() {
                for (target, &(j, i, expected)) in targets.iter().enumerate() {
                    let sample = |delta: f64| {
                        let mut displaced = point;
                        displaced[j] += delta;
                        // LYP is linear in every sigma variable. Opposite-
                        // spin rho/sigma partials therefore have no cubic
                        // remainder along sigma alone. Subtract the paths
                        // e_sigma + e_rho_a and e_rho_a to isolate the same
                        // partial with a measurable quadratic remainder.
                        if j >= 2 {
                            displaced[0] += delta;
                        }
                        let out = evaluate(&f, &displaced, 1);
                        let mut values = [
                            out.e[0],
                            out.vrho[0][0],
                            out.vrho[1][0],
                            out.vsigma[0][0],
                            out.vsigma[1][0],
                            out.vsigma[2][0],
                        ];
                        if j >= 2 {
                            displaced[j] = point[j];
                            let other = evaluate(&f, &displaced, 1);
                            let baseline = [
                                other.e[0],
                                other.vrho[0][0],
                                other.vrho[1][0],
                                other.vsigma[0][0],
                                other.vsigma[1][0],
                                other.vsigma[2][0],
                            ];
                            for k in 0..6 {
                                values[k] -= baseline[k];
                            }
                        }
                        values
                    };
                    let plus = sample(h);
                    let minus = sample(-h);
                    let plus2 = sample(2.0 * h);
                    let minus2 = sample(-2.0 * h);
                    let estimate = stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                    errors[target][step] = (estimate - expected).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("BLYP35 density derivatives", &failures);
    // The GGA has no tau dependence, including mixed derivatives.
    assert_eq!(&potential[5..], &[0.0, 0.0]);
    for i in 0..7 {
        for j in 5..7 {
            assert_eq!(kernel[i][j], 0.0);
            assert_eq!(kernel[j][i], 0.0);
        }
    }
}

#[test]
fn restricted_and_unrestricted_scf_and_fixed_density_terms_match_oracle() {
    use dft_rs::basis::{data, MolBasis};
    use dft_rs::grid::{GridOptions, MolecularGrid};
    use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
    use dft_rs::linalg::Matrix;
    use dft_rs::scf::{run_scf, EriMode, ScfOptions};
    use dft_rs::{Atom, Molecule, Vec3};

    let oracle = Value::parse_json(include_str!("oracle_data/catalogue_hybrids.json")).unwrap();
    for entry in rows(oracle.get("scf").unwrap()) {
        let number = |key: &str| entry.get(key).unwrap().as_f64().unwrap();
        let numbers = entry.get("numbers").unwrap().numeric_array().unwrap().1;
        let atoms = numbers
            .iter()
            .zip(rows(entry.get("positions").unwrap()))
            .map(|(z, r)| {
                let r = r.numeric_array().unwrap().1;
                Atom {
                    z: *z as u8,
                    position: Vec3::new(r[0], r[1], r[2]),
                }
            })
            .collect();
        let molecule = Molecule::new(atoms)
            .with_charge(number("charge") as i32)
            .with_multiplicity(number("multiplicity") as usize);
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        let grid = MolecularGrid::build(
            &molecule,
            &GridOptions {
                level: 9,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let options = ScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-10,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        let f = functional();
        let scf = run_scf(&molecule, &basis, &options, &f, Some(&grid)).unwrap();
        assert!(scf.converged);
        assert!((scf.energy - number("total")).abs() < 5e-7);
        let charge_defect =
            scf.grid_electron_count.unwrap() - molecule.n_electrons().unwrap() as f64;
        println!(
            "BLYP35 {} SCF total {:.14e}, oracle error {:.3e}, integrated electron defect {:.3e}",
            entry.get("molecule").unwrap().as_str().unwrap(),
            scf.energy,
            scf.energy - number("total"),
            charge_defect
        );
        assert!(charge_defect.abs() < 1e-7);
        let density: Vec<_> = rows(entry.get("density").unwrap())
            .iter()
            .map(|spin| {
                Matrix::from_row_major(
                    basis.nao_sph,
                    basis.nao_sph,
                    rows(spin)
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
            + nuclear_matrix(&basis, &molecule, None).frobenius_dot(&total);
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
        let mut terms = Vec::new();
        for part in [dft_rs::xc::gga::b88(), dft_rs::xc::gga::lyp()] {
            let component = ResolvedFunctional::custom(
                "fixed-density component",
                vec![(1.0, part.into())],
                dft_rs::xc::HybridParams::PURE,
                Family::Gga,
            );
            terms.push(
                dft_rs::ks::xc_contribution(&component, &basis, &grid, [&density[0], &density[1]])
                    .unwrap()
                    .exc,
            );
        }
        for (key, value, tolerance) in [
            ("core", core, 1e-9),
            ("coulomb", hartree, 1e-9),
            ("hf_exchange", exact, 1e-9),
            ("exchange", terms[0], 5e-7),
            ("correlation", terms[1], 5e-7),
        ] {
            println!(
                "BLYP35 {} {key}: {value:.14e}, oracle {:.14e}, error {:.3e}",
                entry.get("molecule").unwrap().as_str().unwrap(),
                number(key),
                (value - number(key)).abs()
            );
            assert!((value - number(key)).abs() < tolerance, "{key}");
        }
        let sum = scf.e_nuc + core + hartree + 0.35 * exact + 0.65 * terms[0] + terms[1];
        assert!((sum - number("total")).abs() < 5e-7);
    }
}

#[test]
fn all_c1_molecular_gradient_components_have_four_step_error_fits() {
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.43, 0.81, -0.62],
            [-0.63, 1.59, -0.43],
            [-1.12, -1.36, -0.71],
        ],
    );
    let controls = Controls::from_value(&Value::Object(vec![
        ("xc".into(), Value::Str("blyp35".into())),
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
    let gradient = report.get("gradient").unwrap().numeric_array().unwrap().1;
    assert_eq!(gradient.len(), 12);
    let mut failures = Vec::new();
    for coordinate in 0..12 {
        stencils::fit_components_staged(
            &[format!("BLYP35 C1 gradient {coordinate}")],
            &[
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = [0.0; 4];
                let energy = |h: f64| {
                    let mut displaced = system.clone();
                    displaced.positions[coordinate / 3][coordinate % 3] += h;
                    run("energy", &displaced, &controls)
                        .unwrap()
                        .get("energy")
                        .unwrap()
                        .as_f64()
                        .unwrap()
                };
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
    components::assert_no_failures("BLYP35 C1 gradient", &failures);
}
