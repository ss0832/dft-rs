// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! EDF2: published hybrid coefficients, independent component/term oracles,
//! complete local partials and C1 gradient/Hessian fits.

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

const NAMES: [&str; 1] = ["edf2"];
const LABELS: [&str; 6] = [
    "lda_exchange",
    "b88_exchange",
    "modified_b88",
    "vwn5",
    "lyp",
    "modified_lyp",
];
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
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/edf2.json"),
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
    let mut failures = Vec::new();
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
        assert_eq!(f.parts.len(), 6);
        assert_eq!(f.parts[2].1.params(), &[0.0035]);
        assert_eq!(f.parts[5].1.params(), &[0.055, 0.158, 0.25, 0.3505]);
        assert_eq!(f.parts[3].1.name(), "vwn5");
        let labels = LABELS;
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
                let component = evaluate(&part, &p, if p[1] > 0.0 { 2 } else { 1 });
                let expected = &rows(oracle.get("components").unwrap().get(label).unwrap())[index];
                worst[2] = worst[2].max((component.e[0] - number(expected, "energy")).abs());
                assert!(
                    (component.e[0] - number(expected, "energy")).abs() < 1e-11,
                    "{name} {label}"
                );
                let ep = array(expected, "potential");
                let cp = [
                    component.vrho[0][0],
                    component.vrho[1][0],
                    if part.family == Family::Lda {
                        0.0
                    } else {
                        component.vsigma[0][0]
                    },
                    if part.family == Family::Lda {
                        0.0
                    } else {
                        component.vsigma[1][0]
                    },
                    if part.family == Family::Lda {
                        0.0
                    } else {
                        component.vsigma[2][0]
                    },
                ];
                for k in 0..5 {
                    if p[1] == 0.0 && [1, 3, 4].contains(&k) {
                        continue;
                    }
                    assert!(
                        (cp[k] - ep[k]).abs() < 1e-10 * (1.0 + ep[k].abs()),
                        "component {label} point{index} potential{k}"
                    );
                }
                if let Some(ek) = expected.get("kernel") {
                    let (_, ck) = component.potential_and_kernel(0, part.family).unwrap();
                    for (i, row) in rows(ek).iter().enumerate() {
                        for (j, expected) in row.numeric_array().unwrap().1.iter().enumerate() {
                            let error = (ck[i][j] - expected).abs();
                            if !error.is_finite() || error >= 1e-9 * (1.0 + expected.abs()) {
                                failures.push(format!("component {label} point{index} kernel{i},{j}: actual={} reference={expected}",ck[i][j]));
                            }
                        }
                    }
                }
            }
            if let Some(expected) = expected.get("kernel") {
                let (_, kernel) = out.potential_and_kernel(0, f.family).unwrap();
                for (i, row) in rows(expected).iter().enumerate() {
                    for (j, value) in row.numeric_array().unwrap().1.iter().enumerate() {
                        worst[3] = worst[3].max((kernel[i][j] - value).abs());
                        let error = (kernel[i][j] - value).abs();
                        if !error.is_finite() || error >= 1e-9 * (1.0 + value.abs()) {
                            failures.push(format!(
                                "{name} point{index} kernel{i},{j}: actual={} reference={value}",
                                kernel[i][j]
                            ));
                        }
                    }
                }
            }
        }
        println!("{name} maximum frozen-point errors [total energy density, potential, component energy density, kernel] = {worst:?}");
    }
    components::assert_no_failures("EDF2 frozen full-kernel oracle", &failures);
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
            for i in 0..=j {
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
    components::assert_no_failures("EDF2 independent density partials", &failures);
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
            f.parts[..3].to_vec(),
            HybridParams::PURE,
            Family::Gga,
        );
        let scale = f.parts[..3].iter().map(|(weight, _)| *weight).sum::<f64>();
        assert!((scale - 0.8487).abs() < 1e-15);
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
            for (label, (weight, part)) in LABELS.into_iter().zip(&f.parts) {
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
fn every_c1_gradient_and_independent_hessian_component_has_four_step_fits() {
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.17, -0.11, 0.23],
            [1.43, 0.81, -0.62],
            [-0.63, 1.59, -0.43],
            [-1.12, -1.36, -0.71],
        ],
    );
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "edf2")
            .with("basis", "sto-3g")
            .with("grid_level", 5usize)
            .with("grid_weight_cutoff", 0.0)
            .with("dispersion", "none")
            .with("conv_tol", 1e-14)
            .with("conv_tol_grad", 1e-12)
            .with("conv_tol_dp", 1e-12)
            .with("max_iter", 500usize),
    )
    .unwrap();
    let base = run("gradient", &system, &controls).unwrap();
    let gradient = array(&base, "gradient");
    let hessian = array(&run("hessian", &system, &controls).unwrap(), "hessian");
    assert_eq!(gradient.len(), 12);
    assert_eq!(hessian.len(), 144);
    for i in 0..12 {
        for j in 0..12 {
            assert!((hessian[12 * i + j] - hessian[12 * j + i]).abs() < 1e-9);
        }
    }
    let mut failures = Vec::new();
    for coordinate in 0..12 {
        let mut cache = std::collections::HashMap::<u64, (f64, Vec<f64>)>::new();
        let mut displaced = |h: f64| {
            cache
                .entry(h.to_bits())
                .or_insert_with(|| {
                    let mut shifted = system.clone();
                    shifted.positions[coordinate / 3][coordinate % 3] += h;
                    let report = run("gradient", &shifted, &controls).unwrap();
                    (number(&report, "energy"), array(&report, "gradient"))
                })
                .clone()
        };
        let mut labels = vec![format!("edf2 C1 gradient{coordinate}")];
        labels.extend((0..=coordinate).map(|r| format!("edf2 C1 Hessian{r},{coordinate}")));
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                (
                    [0.005, 0.0025, 0.00125, 0.000625],
                    stencils::Stencil::Second,
                ),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; coordinate + 2];
                for (i, &h) in steps.iter().enumerate() {
                    let (ep, gp) = displaced(h);
                    let (em, gm) = displaced(-h);
                    let ((ep2, gp2), (em2, gm2)) = if stencil == stencils::Stencil::Fourth {
                        (displaced(2.0 * h), displaced(-2.0 * h))
                    } else {
                        ((0.0, vec![0.0; 12]), (0.0, vec![0.0; 12]))
                    };
                    errors[0][i] =
                        (stencil.first(ep, em, ep2, em2, h) - gradient[coordinate]).abs();
                    for row in 0..=coordinate {
                        errors[row + 1][i] = (stencil
                            .first(gp[row], gm[row], gp2[row], gm2[row], h)
                            - hessian[12 * row + coordinate])
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("EDF2 C1 gradient and independent Hessian", &failures);
}

fn symmetric_third(output: &XcOut, i: usize, j: usize, k: usize) -> f64 {
    assert!(i <= j && j <= k && k < 7);
    let pair = |a, b, n| (0..a).map(|p| n - p).sum::<usize>() + b - a;
    let triple = |a, b, c, n| {
        let mut index = 0;
        for p in 0..n {
            for q in p..n {
                for r in q..n {
                    if (p, q, r) == (a, b, c) {
                        return index;
                    }
                    index += 1;
                }
            }
        }
        panic!("invalid symmetric triple")
    };
    let indices = [i, j, k];
    let rho = indices.iter().filter(|&&v| v < 2).count();
    let sigma = indices.iter().filter(|&&v| (2..5).contains(&v)).count();
    match (rho, sigma) {
        (3, 0) => output.v3rho3[i + j + k][0],
        (2, 1) => output.v3rho2sigma[3 * (i + j) + k - 2][0],
        (1, 2) => output.v3rhosigma2[6 * i + pair(j - 2, k - 2, 3)][0],
        (0, 3) => output.v3sigma3[triple(i - 2, j - 2, k - 2, 3)][0],
        (2, 0) => output.v3rho2tau[2 * (i + j) + k - 5][0],
        (1, 1) => output.v3rhosigmatau[6 * i + 2 * (j - 2) + k - 5][0],
        (1, 0) => output.v3rhotau2[3 * i + j + k - 10][0],
        (0, 2) => output.v3sigma2tau[2 * pair(i - 2, j - 2, 3) + k - 5][0],
        (0, 1) => output.v3sigmatau2[3 * (i - 2) + j + k - 10][0],
        (0, 0) => output.v3tau3[i + j + k - 15][0],
        _ => panic!("invalid derivative block"),
    }
}

#[test]
fn every_independent_semilocal_third_partial_has_four_step_fits() {
    let f = functional("edf2");
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let u = [0.23, -0.17, 0.11, 0.13, -0.19];
    let output = evaluate(&f, &point, 3);
    let mut targets = Vec::new();
    let mut labels = Vec::new();
    for i in 0..5 {
        for j in i..5 {
            for k in j..5 {
                let expected = symmetric_third(&output, i, j, k);
                let axes = [i, j, k];
                let sigma = axes.iter().filter(|&&v| v >= 2).count();
                let same_spin = axes.iter().all(|v| [0, 2].contains(v))
                    || axes.iter().all(|v| [1, 4].contains(v));
                if sigma >= 2 && !same_spin {
                    assert_eq!(expected, 0.0);
                } else {
                    targets.push((i, j, k, expected));
                    labels.push(format!("edf2 third{i},{j},{k}"));
                }
            }
        }
    }
    assert_eq!(targets.len(), 17);
    let mut failures = Vec::new();
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
                let mut directions = [[[0.0; 5]; 5]; 5];
                for axis in 0..5 {
                    let at = |scale: f64| {
                        let mut p = point;
                        for t in 0..5 {
                            p[t] += scale * (u[t] + if axis == t { 1.0 } else { 0.0 });
                        }
                        assert!(
                            p[0] > 0.0
                                && p[1] > 0.0
                                && p[2] > 0.0
                                && p[4] > 0.0
                                && p[3] * p[3] < p[2] * p[4]
                        );
                        evaluate(&f, &p, 2)
                            .potential_and_kernel(0, Family::Gga)
                            .unwrap()
                            .1
                    };
                    let plus = at(h);
                    let minus = at(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        ([[0.0; 7]; 7], [[0.0; 7]; 7])
                    };
                    for i in 0..5 {
                        for j in 0..5 {
                            directions[i][j][axis] = stencil.first(
                                plus[i][j],
                                minus[i][j],
                                plus2[i][j],
                                minus2[i][j],
                                h,
                            );
                        }
                    }
                }
                for (target, &(i, j, k, expected)) in targets.iter().enumerate() {
                    let shared = u
                        .iter()
                        .zip(directions[i][j])
                        .map(|(u, d)| u * d)
                        .sum::<f64>()
                        / (1.0 + u.iter().sum::<f64>());
                    errors[target][step] = (directions[i][j][k] - shared - expected).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("EDF2 complete semilocal third partials", &failures);
}

fn original_vwn5_density_partials(point: &[f64]) -> [f64; 15] {
    let v = [point[0], point[1], 0.0, 0.0, 0.0, 0.0, 0.0];
    let f = ResolvedFunctional::custom(
        "VWN5",
        vec![(1.0, std::sync::Arc::from(dft_rs::xc::lda::vwn5()))],
        HybridParams::PURE,
        Family::Lda,
    );
    assert_eq!(
        f.parts[0].1.params()[8],
        -1.0 / (6.0 * std::f64::consts::PI * std::f64::consts::PI)
    );
    let output = evaluate(&f, &v, 3);
    let (potential, kernel) = output.potential_and_kernel(0, Family::Lda).unwrap();
    let mut result = [0.0; 15];
    result[0] = output.e[0];
    for i in 0..2 {
        result[1 + i] = potential[i];
        for j in 0..2 {
            result[3 + 2 * i + j] = kernel[i][j];
            for k in 0..2 {
                result[7 + 4 * i + 2 * j + k] = output.v3rho3[i + j + k][0];
            }
        }
    }
    result
}

#[test]
fn zero_sigma_b88_kernels_match_independent_inward_limits() {
    // Differentiate the published B88 denominator using scalar arithmetic,
    // independently of the production small-u polynomial and AD driver.
    let density = [0.31_f64, 0.22];
    let beta = [0.0042_f64, 0.0035];
    let weights = [0.6227, -0.0551];
    let potential = |rho: f64, beta: f64, sigma: f64| {
        let u = sigma / rho.powf(8.0 / 3.0);
        if u == 0.0 {
            return -beta / rho.powf(4.0 / 3.0);
        }
        let x = u.sqrt();
        let y = x * x.asinh();
        let dy = x.asinh() / (2.0 * x) + 0.5 / (1.0 + u).sqrt();
        -beta / rho.powf(4.0 / 3.0) * (1.0 + 6.0 * beta * (y - u * dy))
            / (1.0 + 6.0 * beta * y).powi(2)
    };
    let point = [density[0], density[1], 0.0, 0.0, 0.0, 0.37, 0.21];
    let f = functional("edf2");
    let mut targets = Vec::new();
    let mut labels = Vec::new();
    for (part, &b) in beta.iter().enumerate() {
        let provider = ResolvedFunctional::custom(
            "B88 boundary",
            vec![(1.0, f.parts[part + 1].1.clone())],
            HybridParams::PURE,
            Family::Gga,
        );
        let (_, kernel) = evaluate(&provider, &point, 2)
            .potential_and_kernel(0, Family::Gga)
            .unwrap();
        for spin in 0..2 {
            let index = 2 + 2 * spin;
            let closed = 12.0 * b * b / density[spin].powi(4);
            assert!((kernel[index][index] - closed).abs() < 1e-14);
            // Positive physical sigma approaches test the neighborhood as well
            // as the exact boundary. This scalar derivative is independent of AD.
            for sigma in [0.0008, 0.0004, 0.0002, 0.0001] {
                let u = sigma / density[spin].powf(8.0 / 3.0);
                let x = u.sqrt();
                let y = x * x.asinh();
                let dy = x.asinh() / (2.0 * x) + 0.5 / (1.0 + u).sqrt();
                let ddy = 1.0 / (4.0 * u * (1.0 + u).sqrt())
                    - x.asinh() / (4.0 * u * x)
                    - 1.0 / (4.0 * (1.0 + u).powf(1.5));
                let denominator = 1.0 + 6.0 * b * y;
                let numerator = 1.0 + 6.0 * b * (y - u * dy);
                let expected = 6.0 * b * b / density[spin].powi(4)
                    * (u * ddy / denominator.powi(2) + 2.0 * numerator * dy / denominator.powi(3));
                let mut near = point;
                near[index] = sigma;
                let (_, actual) = evaluate(&provider, &near, 2)
                    .potential_and_kernel(0, Family::Gga)
                    .unwrap();
                assert!((actual[index][index] - expected).abs() < 1e-12);
                assert!(
                    (evaluate(&provider, &near, 1).vsigma[index - 2][0]
                        - potential(density[spin], b, sigma))
                    .abs()
                        < 1e-14
                );
            }
            labels.push(format!("B88 beta{b} spin{spin} inward kernel"));
            targets.push((spin, Some(part), closed));
        }
    }
    let (_, kernel) = evaluate(&f, &point, 2)
        .potential_and_kernel(0, Family::Gga)
        .unwrap();
    for spin in 0..2 {
        let closed = beta
            .iter()
            .zip(weights)
            .map(|(b, w)| w * 12.0 * b * b / density[spin].powi(4))
            .sum::<f64>();
        let index = 2 + 2 * spin;
        assert!((kernel[index][index] - closed).abs() < 1e-14);
        labels.push(format!("EDF2 spin{spin} inward kernel"));
        targets.push((spin, None, closed));
    }
    let mut failures = Vec::new();
    components::fit_components(
        &labels,
        &[
            [0.0008, 0.0004, 0.0002, 0.0001],
            [0.0004, 0.0002, 0.0001, 0.00005],
        ],
        |steps| {
            targets
                .iter()
                .map(|&(spin, part, closed)| {
                    steps.map(|h| {
                        let at = |s| match part {
                            Some(part) => potential(density[spin], beta[part], s),
                            None => beta
                                .iter()
                                .zip(weights)
                                .map(|(&b, w)| w * potential(density[spin], b, s))
                                .sum(),
                        };
                        ((-3.0 * at(0.0) + 4.0 * at(h) - at(2.0 * h)) / (2.0 * h) - closed).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(
        "Independent B88 inward limits; raw frozen comparisons remain separate",
        &failures,
    );
}

#[test]
fn every_original_vwn5_partial_through_three_has_four_step_fits() {
    let point = [0.31, 0.22];
    let u = [0.13, -0.09];
    let analytic = original_vwn5_density_partials(&point);
    let targets = [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 3),
        (1, 1, 4),
        (2, 1, 6),
        (3, 0, 7),
        (3, 1, 8),
        (4, 1, 10),
        (6, 1, 14),
    ];
    let labels: Vec<_> = (0..targets.len())
        .map(|i| format!("original VWN5 partial{i}"))
        .collect();
    let mut failures = Vec::new();
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
                let mut directions = [[0.0; 2]; 15];
                for axis in 0..2 {
                    let at = |scale: f64| {
                        let v: Vec<_> = (0..2)
                            .map(|i| point[i] + scale * (u[i] + if axis == i { 1.0 } else { 0.0 }))
                            .collect();
                        assert!(v.iter().all(|v| *v > 0.0));
                        original_vwn5_density_partials(&v)
                    };
                    let plus = at(h);
                    let minus = at(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        ([0.0; 15], [0.0; 15])
                    };
                    for i in 0..15 {
                        directions[i][axis] =
                            stencil.first(plus[i], minus[i], plus2[i], minus2[i], h);
                    }
                }
                for (target, &(lower, axis, expected)) in targets.iter().enumerate() {
                    let shared = (u[0] * directions[lower][0] + u[1] * directions[lower][1])
                        / (1.0 + u[0] + u[1]);
                    errors[target][step] =
                        (directions[lower][axis] - shared - analytic[expected]).abs();
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("original VWN5 full first/second/third partials", &failures);
}
