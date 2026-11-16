// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! N12-SX exact screened-HF recipe, independent local values and full kernels.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::Value;
use dft_rs::xc::{
    resolve, Family, FunctionalSpec, GridVars, HybridParams, ResolvedFunctional, XcOut,
};
const NAMES: [&str; 3] = ["n12-sx", "exchange", "correlation"];
fn functional(name: &str) -> ResolvedFunctional {
    let f = resolve(&FunctionalSpec::Combined("n12-sx".into())).unwrap();
    match name {
        "n12-sx" => f,
        "exchange" => ResolvedFunctional::custom(
            name,
            vec![f.parts[0].clone()],
            HybridParams::PURE,
            Family::Gga,
        ),
        "correlation" => ResolvedFunctional::custom(
            name,
            vec![f.parts[1].clone()],
            HybridParams::PURE,
            Family::Gga,
        ),
        _ => panic!("unknown component"),
    }
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

#[test]
fn frozen_components_and_recipe_limits() {
    let f = functional("n12-sx");
    assert_eq!(
        f.hybrid,
        HybridParams {
            cx: 0.25,
            clr: -0.25,
            omega: 0.11
        }
    );
    assert_eq!(f.family, Family::Gga);
    assert_eq!(f.parts.len(), 2);
    assert!(f.parts.iter().all(|p| p.0 == 1.0));
    assert!(f.required_nonlocal_kind().is_none());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let data = Value::parse_json(
        &std::fs::read_to_string(root.join("tests/oracle_data/n12_sx.json")).unwrap(),
    )
    .unwrap();
    let Value::Array(points) = data.get("points").unwrap() else {
        panic!()
    };
    for name in ["exchange", "correlation"] {
        let part = functional(name);
        let Value::Array(values) = data.get("components").unwrap().get(name).unwrap() else {
            panic!()
        };
        for (point, value) in points.iter().zip(values) {
            let p = point.numeric_array().unwrap().1;
            let out = evaluate(&part, &p, if p[1] > 0.0 { 2 } else { 1 });
            let expected = value.get("energy").unwrap().as_f64().unwrap();
            assert!((out.e[0] - expected).abs() < 1e-11, "{name} {p:?} energy");
            let expected = value.get("potential").unwrap().numeric_array().unwrap().1;
            let actual = vector(&out);
            for i in 0..5 {
                if p[1] == 0.0 && [1, 3, 4].contains(&i) {
                    continue;
                }
                assert!(
                    (actual[i + 1] - expected[i]).abs() < 1e-10 * (1.0 + expected[i].abs()),
                    "{name} {p:?} potential{i}"
                );
            }
            if let Some(Value::Array(rows)) = value.get("kernel") {
                let (_, kernel) = out.potential_and_kernel(0, Family::Gga).unwrap();
                for (i, row) in rows.iter().enumerate() {
                    for (j, expected) in row.numeric_array().unwrap().1.into_iter().enumerate() {
                        assert!(
                            (kernel[i][j] - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                            "{name} {p:?} kernel{i},{j}: {} vs {expected}",
                            kernel[i][j]
                        );
                    }
                }
            }
        }
    }
    let p = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let swapped = [p[1], p[0], p[4], p[3], p[2], p[6], p[5]];
    let (a, b) = (evaluate(&f, &p, 2), evaluate(&f, &swapped, 2));
    assert!((a.e[0] - b.e[0]).abs() < 1e-14);
    let (va, ka) = a.potential_and_kernel(0, Family::Gga).unwrap();
    let (vb, kb) = b.potential_and_kernel(0, Family::Gga).unwrap();
    let perm = [1, 0, 4, 3, 2, 6, 5];
    for i in 0..7 {
        assert!((va[i] - vb[perm[i]]).abs() < 1e-13);
        for j in 0..7 {
            assert!((ka[i][j] - kb[perm[i]][perm[j]]).abs() < 1e-12);
        }
    }
    assert_eq!(evaluate(&f, &[0.0; 7], 0).e[0], 0.0);
}
#[test]
fn every_active_density_derivative_has_independent_four_step_fits() {
    // Five linearly independent dense directions D=I+u*1^T. Inverting its
    // directional differences recovers every partial separately. A generic
    // rho/sigma mixture prevents a linear sigma term's exact stencil from
    // hiding its convergence order behind roundoff. Frozen kernels below
    // independently check every nonzero local kernel entry.
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
            if j != 3 {
                labels.push(format!("{name} potential {j}"));
                targets.push((0, j, potential[j]));
            } else {
                assert_eq!(potential[j], 0.0);
            }
            for i in 0..=j {
                if i == 3
                    || j == 3
                    || (name == "exchange"
                        && !([i, j].iter().all(|v| [0, 2].contains(v))
                            || [i, j].iter().all(|v| [1, 4].contains(v))))
                {
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
    components::assert_no_failures("N12-SX complete active density partials", &failures);
}
fn symmetric_third(output: &XcOut, i: usize, j: usize, k: usize) -> f64 {
    assert!(i <= j && j <= k && k < 7);
    if k >= 5 {
        return 0.0;
    }
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
fn every_independent_third_partial_has_four_step_kernel_fits() {
    let point = [0.31, 0.22, 0.041, 0.017, 0.029, 0.37, 0.21];
    let u = [0.23, -0.17, 0.11, 0.13, -0.19, 0.07, -0.09];
    let denominator = 1.0 + u.iter().sum::<f64>();
    let mut failures = Vec::new();
    for name in NAMES {
        let provider = functional(name);
        let output = evaluate(&provider, &point, 3);
        let mut targets = Vec::new();
        let mut labels = Vec::new();
        for i in 0..7 {
            for j in i..7 {
                for k in j..7 {
                    let expected = symmetric_third(&output, i, j, k);
                    let same_spin = [i, j, k].iter().all(|v| [0, 1, 2, 4].contains(v))
                        && (name != "exchange"
                            || [i, j, k].iter().all(|v| [0, 2].contains(v))
                            || [i, j, k].iter().all(|v| [1, 4].contains(v)));
                    if !same_spin {
                        assert_eq!(expected, 0.0, "structural zero {i},{j},{k}");
                    } else {
                        targets.push((i, j, k, expected));
                        labels.push(format!("N12-SX {name} third{i},{j},{k}"));
                    }
                }
            }
        }
        assert_eq!(targets.len(), if name == "exchange" { 8 } else { 20 });
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
                    let mut directions = [[[0.0; 7]; 7]; 7];
                    for axis in 0..7 {
                        let at = |scale: f64| {
                            let mut displaced = point;
                            for t in 0..7 {
                                displaced[t] += scale * (u[t] + if axis == t { 1.0 } else { 0.0 });
                            }
                            assert!(displaced[0] > 0.0 && displaced[1] > 0.0);
                            assert!(displaced[2] > 0.0 && displaced[4] > 0.0);
                            assert!(displaced[3].powi(2) < displaced[2] * displaced[4]);
                            for (r, s, t) in [(0, 2, 5), (1, 4, 6)] {
                                assert!(displaced[t] > displaced[s] / (8.0 * displaced[r]));
                            }
                            evaluate(&provider, &displaced, 2)
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
                        for i in 0..7 {
                            for j in 0..7 {
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
                            / denominator;
                        errors[target][step] = (directions[i][j][k] - shared - expected).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("N12-SX complete local third derivatives", &failures);
}

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::{run, Controls, System};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};

use dft_rs::{Atom, Molecule, Vec3};
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
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/n12_sx.json"),
        )
        .unwrap(),
    )
    .unwrap()
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
fn restricted_unrestricted_scf_and_each_fixed_density_term_match_oracles() {
    let oracle = oracle();
    let set = data::load("sto-3g").unwrap();
    for name in ["n12-sx"] {
        let f = functional(name);
        let model = &oracle;
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
            let mut long_range = 0.0;
            let lr = eri_incore(&basis, CoulombKernel::Erf { omega: 0.11 }, 512).unwrap();
            for i in 0..basis.nao_sph {
                for j in 0..basis.nao_sph {
                    for k in 0..basis.nao_sph {
                        for l in 0..basis.nao_sph {
                            let integral = eri.get(i, j, k, l);
                            hartree += 0.5 * total[(i, j)] * total[(k, l)] * integral;
                            for spin in &density {
                                exact -= 0.5 * spin[(i, k)] * spin[(j, l)] * integral;
                                long_range -=
                                    0.5 * spin[(i, k)] * spin[(j, l)] * lr.get(i, j, k, l);
                            }
                        }
                    }
                }
            }
            let mut terms = vec![
                ("one_electron", core, 1e-9),
                ("hartree", hartree, 1e-9),
                ("full_exchange", exact, 1e-9),
                ("long_range_exchange", long_range, 1e-9),
            ];
            let mut sum =
                scf.e_nuc + core + hartree + f.hybrid.cx * exact + f.hybrid.clr * long_range;
            for (label, (weight, part)) in
                ["local_exchange", "correlation"].into_iter().zip(&f.parts)
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
                let error = (value - number(entry.get("terms").unwrap(), label)).abs();
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
            .with("xc", "n12-sx")
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
        let mut labels = vec![format!("n12-sx C1 gradient{coordinate}")];
        labels.extend((0..=coordinate).map(|r| format!("n12-sx C1 Hessian{r},{coordinate}")));
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
    components::assert_no_failures("N12-SX C1 gradient and independent Hessian", &failures);
}
