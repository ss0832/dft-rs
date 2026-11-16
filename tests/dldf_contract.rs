// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Dispersionless hybrid meta-GGA definition and complete local derivatives.
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

fn provider(part: usize) -> ResolvedFunctional {
    let f = resolve(&FunctionalSpec::Combined("dldf".into())).unwrap();
    if part == 2 {
        f
    } else {
        ResolvedFunctional::custom(
            "dlDF component",
            vec![f.parts[part].clone()],
            HybridParams::PURE,
            Family::MetaGga,
        )
    }
}
fn evaluate(f: &ResolvedFunctional, p: &[f64], order: usize) -> XcOut {
    let values: Vec<_> = p.iter().map(|&v| [v]).collect();
    let vars = GridVars {
        rho: [&values[0], &values[1]],
        sigma: Some([&values[2], &values[3], &values[4]]),
        tau: Some([&values[5], &values[6]]),
    };
    let mut out = XcOut::default();
    f.eval(&vars, order, &mut out).unwrap();
    out
}
fn present(part: usize, indices: &[usize]) -> bool {
    let a = indices.iter().all(|i| [0, 2, 5].contains(i));
    let b = indices.iter().all(|i| [1, 4, 6].contains(i));
    a || b || (part != 0 && indices.iter().all(|i| [0, 1, 2, 4].contains(i)))
}
fn third(o: &XcOut, i: usize, j: usize, k: usize) -> f64 {
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
        panic!("invalid triple")
    };
    let v = [i, j, k];
    let rho = v.iter().filter(|&&v| v < 2).count();
    let sigma = v.iter().filter(|&&v| (2..5).contains(&v)).count();
    match (rho, sigma) {
        (3, 0) => o.v3rho3[i + j + k][0],
        (2, 1) => o.v3rho2sigma[3 * (i + j) + k - 2][0],
        (1, 2) => o.v3rhosigma2[6 * i + pair(j - 2, k - 2, 3)][0],
        (0, 3) => o.v3sigma3[triple(i - 2, j - 2, k - 2, 3)][0],
        (2, 0) => o.v3rho2tau[2 * (i + j) + k - 5][0],
        (1, 1) => o.v3rhosigmatau[6 * i + 2 * (j - 2) + k - 5][0],
        (1, 0) => o.v3rhotau2[3 * i + j + k - 10][0],
        (0, 2) => o.v3sigma2tau[2 * pair(i - 2, j - 2, 3) + k - 5][0],
        (0, 1) => o.v3sigmatau2[3 * (i - 2) + j + k - 10][0],
        (0, 0) => o.v3tau3[i + j + k - 15][0],
        _ => panic!("invalid third block"),
    }
}

#[test]
fn independent_point_kernels_recipe_and_physical_limits() {
    let f = provider(2);
    assert_eq!(f.hybrid, HybridParams::global(0.6144129));
    assert_eq!(f.family, Family::MetaGga);
    assert_eq!(f.parts.len(), 2);
    assert!(f.parts.iter().all(|p| p.0 == 1.0));
    assert!(f.required_nonlocal_kind().is_none());
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/dldf.json"),
    )
    .unwrap();
    let data = Value::parse_json(&text).unwrap();
    let Value::Array(points) = data.get("points").unwrap() else {
        panic!()
    };
    for (part, name) in ["exchange", "correlation"].iter().enumerate() {
        let Value::Array(refs) = data.get("components").unwrap().get(name).unwrap() else {
            panic!()
        };
        for (point, reference) in points.iter().zip(refs) {
            let p = point.numeric_array().unwrap().1;
            let out = evaluate(&provider(part), &p, if p[1] > 0.0 { 2 } else { 1 });
            let expected = reference.get("energy").unwrap().as_f64().unwrap();
            assert!(
                (out.e[0] - expected).abs() < 1e-11,
                "part{part} {p:?} energy"
            );
            let potential = [
                out.vrho[0][0],
                out.vrho[1][0],
                out.vsigma[0][0],
                out.vsigma[1][0],
                out.vsigma[2][0],
                out.vtau[0][0],
                out.vtau[1][0],
            ];
            for (i, e) in reference
                .get("potential")
                .unwrap()
                .numeric_array()
                .unwrap()
                .1
                .iter()
                .enumerate()
            {
                if p[1] == 0.0 && [1, 3, 4, 6].contains(&i) {
                    continue;
                }
                assert!(
                    (potential[i] - e).abs() < 1e-10 * (1.0 + e.abs()),
                    "part{part} potential{i}"
                );
            }
            if let Some(Value::Array(rows)) = reference.get("kernel") {
                let (_, kernel) = out.potential_and_kernel(0, Family::MetaGga).unwrap();
                for (i, row) in rows.iter().enumerate() {
                    for (j, e) in row.numeric_array().unwrap().1.iter().enumerate() {
                        assert!(
                            (kernel[i][j] - e).abs() < 1e-9 * (1.0 + e.abs()),
                            "part{part} kernel{i},{j}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(evaluate(&f, &[0.0; 7], 0).e[0], 0.0);
    let rho: f64 = 0.31;
    let sigma: f64 = 0.041;
    let one_orbital = [rho, 0.0, sigma, 0.0, 0.0, sigma / (8.0 * rho), 0.0];
    assert!(evaluate(&provider(1), &one_orbital, 0).e[0].abs() < 1e-16);
    let uniform_tau =
        0.3 * (6.0 * std::f64::consts::PI.powi(2)).powf(2.0 / 3.0) * rho.powf(5.0 / 3.0);
    let uniform = [rho, rho, 0.0, 0.0, 0.0, uniform_tau, uniform_tau];
    let lsda = -0.75 * (6.0 / std::f64::consts::PI).cbrt() * 2.0 * rho.powf(4.0 / 3.0);
    assert!((evaluate(&provider(0), &uniform, 0).e[0] - (1.0 - 0.6144129) * lsda).abs() < 1e-14);
    let p = [0.14, 0.085, 0.5, 0.1, 0.38, 1.05, 0.92];
    let swapped = [p[1], p[0], p[4], p[3], p[2], p[6], p[5]];
    let perm = [1, 0, 4, 3, 2, 6, 5];
    let a = evaluate(&f, &p, 2);
    let b = evaluate(&f, &swapped, 2);
    assert!((a.e[0] - b.e[0]).abs() < 1e-14);
    let (va, ka) = a.potential_and_kernel(0, Family::MetaGga).unwrap();
    let (vb, kb) = b.potential_and_kernel(0, Family::MetaGga).unwrap();
    for i in 0..7 {
        assert!((va[i] - vb[perm[i]]).abs() < 1e-13);
        for j in 0..7 {
            assert!((ka[i][j] - kb[perm[i]][perm[j]]).abs() < 1e-12);
        }
    }
}

#[test]
fn complete_first_second_third_partial_fits() {
    let p = [0.14, 0.085, 0.5, 0.1, 0.38, 1.05, 0.92];
    let mut failures = Vec::new();
    for part in 0..3 {
        let f = provider(part);
        let out = evaluate(&f, &p, 3);
        let (v, k) = out.potential_and_kernel(0, Family::MetaGga).unwrap();
        let mut labels = Vec::new();
        let mut targets = Vec::new();
        for i in 0..7 {
            if present(part, &[i]) {
                labels.push(format!("dlDF part{part} first{i}"));
                targets.push((0, i, 0, 0, v[i]));
            } else {
                assert_eq!(v[i], 0.0);
            }
            for (j, &kernel) in k[i].iter().enumerate().skip(i) {
                if present(part, &[i, j]) {
                    labels.push(format!("dlDF part{part} second{i},{j}"));
                    targets.push((1, j, i, 0, kernel));
                } else {
                    assert_eq!(kernel, 0.0);
                }
                for n in j..7 {
                    let t = third(&out, i, j, n);
                    if present(part, &[i, j, n]) {
                        labels.push(format!("dlDF part{part} third{i},{j},{n}"));
                        targets.push((2, n, i, j, t));
                    } else {
                        assert_eq!(t, 0.0);
                    }
                }
            }
        }
        assert_eq!(targets.len(), if part == 0 { 38 } else { 54 });
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; targets.len()];
                for (step, &relative) in steps.iter().enumerate() {
                    for axis in [0, 1, 2, 4, 5, 6] {
                        let h = relative * p[axis];
                        let at = |t: f64| {
                            let mut point = p;
                            point[axis] += t;
                            let o = evaluate(&f, &point, 2);
                            let (v, k) = o.potential_and_kernel(0, Family::MetaGga).unwrap();
                            (o.e[0], v, k)
                        };
                        let a = at(h);
                        let b = at(-h);
                        let sample = |s: &(f64, [f64; 7], [[f64; 7]; 7]),
                                      order: usize,
                                      i: usize,
                                      j: usize| match order {
                            0 => s.0,
                            1 => s.1[i],
                            2 => s.2[i][j],
                            _ => panic!(),
                        };
                        let extra = if stencil == stencils::Stencil::Fourth {
                            Some((at(2.0 * h), at(-2.0 * h)))
                        } else {
                            None
                        };
                        for (target, &(order, direction, i, j, expected)) in
                            targets.iter().enumerate()
                        {
                            if direction != axis {
                                continue;
                            }
                            let (a2, b2) = extra
                                .as_ref()
                                .map(|(a, b)| (sample(a, order, i, j), sample(b, order, i, j)))
                                .unwrap_or((0.0, 0.0));
                            errors[target][step] = (stencil.first(
                                sample(&a, order, i, j),
                                sample(&b, order, i, j),
                                a2,
                                b2,
                                h,
                            ) - expected)
                                .abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("dlDF complete first-third partials", &failures);
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
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data/dldf.json"),
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
    for name in ["dldf"] {
        let f = provider(2);
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
                ("one_electron", core, 1e-9),
                ("hartree", hartree, 1e-9),
                ("full_exchange", exact, 1e-9),
            ];
            let mut sum = scf.e_nuc + core + hartree + f.hybrid.cx * exact;
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
            .with("xc", "dldf")
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
        let mut labels = vec![format!("dldf C1 gradient{coordinate}")];
        labels.extend((0..=coordinate).map(|r| format!("dldf C1 Hessian{r},{coordinate}")));
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
    components::assert_no_failures("dlDF C1 gradient and independent Hessian", &failures);
}
