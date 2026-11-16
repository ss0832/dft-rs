// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full C1 nuclear tensors against fresh static propagator calculations.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::{eri_incore, overlap_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::properties::pt2::{from_ao, OrbitalSpace};
use dft_rs::response::correlated_tda::correlated_tda_states;
use dft_rs::response::tddft::{TdaSolver, TdaSpin, TdaState};
use dft_rs::system::Molecule;
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use std::collections::HashMap;

fn system() -> System {
    System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    )
}
fn values(uhf: bool) -> Value {
    Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("guess", "core")
        .with("charge", if uhf { 1i32 } else { 2i32 })
        .with("multiplicity", if uhf { 2usize } else { 1usize })
        .with("unrestricted", uhf)
        .with("reference", if uhf { "uks" } else { "rks" })
        .with("dispersion", "none")
        .with("double_hybrid_model", "custom")
        .with("pt2_same_spin", if uhf { 0.17 } else { 0.2 })
        .with("pt2_opposite_spin", if uhf { 0.31 } else { 0.2 })
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-13)
        .with("conv_tol_grad", 1e-12)
        .with("max_iter", 400usize)
        .with("grid_weight_cutoff", 0.0)
        .with("td_tolerance", 1e-10)
        .with("response_tolerance", 1e-11)
        .with("td_memory_mb", 128usize)
        .with("pt2_memory_mb", 128usize)
        .with("root", 1usize)
}
fn scalar(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}
fn array(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}
fn columns(c: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, count);
    for i in 0..c.rows {
        for j in 0..count {
            out[(i, j)] = c[(i, start + j)];
        }
    }
    out
}
fn density(scf: &dft_rs::scf::ScfResult, state: &TdaState) -> [Matrix; 2] {
    std::array::from_fn(|s| {
        let o = if s == 0 { scf.n_alpha } else { scf.n_beta };
        let c = &scf.mo_coeff[s];
        let mut x = if state.spin == TdaSpin::SpinResolved {
            state.x[s].clone()
        } else {
            state.x[0].clone()
        };
        if state.spin != TdaSpin::SpinResolved {
            for value in x.as_mut_slice() {
                *value /= std::f64::consts::SQRT_2;
                if s == 1 && state.spin == TdaSpin::Triplet {
                    *value = -*value;
                }
            }
        }
        columns(c, o, c.cols - o)
            .matmul_seq(&x)
            .matmul_transpose_seq(&columns(c, 0, o))
    })
}
struct Oracle {
    molecule: Molecule,
    overlap: Matrix,
    density: Vec<[Matrix; 2]>,
    energies: [f64; 4],
    gap: f64,
}
impl Oracle {
    fn new(system: &System, input: &Value) -> Self {
        let controls = Controls::from_value(input).unwrap();
        let method = MolecularMethod::from_controls(&controls).unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        let reference = run_molecular_reference(&setup).unwrap();
        let scf = &reference.scf;
        let n = setup.basis.nao_sph;
        let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 128).unwrap();
        let tensor: Vec<_> = (0..n)
            .flat_map(|i| {
                (0..n)
                    .flat_map(move |j| (0..n).flat_map(move |k| (0..n).map(move |l| [i, j, k, l])))
            })
            .map(|[i, j, k, l]| eri.get(i, j, k, l))
            .collect();
        let orbitals = std::array::from_fn(|s| OrbitalSpace {
            energies: scf.mo_energy[s].clone(),
            coefficients: scf.mo_coeff[s].as_slice().to_vec(),
            occupied: if s == 0 { scf.n_alpha } else { scf.n_beta },
        });
        let problem = from_ao(n, &tensor, &orbitals).unwrap();
        let coefficients = Pt2Coefficients {
            same_spin: scalar(input, "pt2_same_spin"),
            opposite_spin: scalar(input, "pt2_opposite_spin"),
        };
        let pt2 = problem.solve(coefficients).unwrap();
        let spin = if scf.unrestricted {
            TdaSpin::SpinResolved
        } else {
            TdaSpin::Singlet
        };
        let parent = TdaSolver::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            scf,
            &setup.options,
            spin,
        )
        .unwrap();
        let parents = parent.solve_dense(parent.dim()).unwrap();
        let corrected = correlated_tda_states(
            &parent,
            &setup.basis,
            scf,
            &eri,
            &problem,
            &pt2,
            coefficients,
            1,
            128,
        )
        .unwrap();
        let ground = reference.total_energy() + pt2.correlation;
        let energies = [
            ground,
            corrected.parent_energies[0],
            corrected.energy_corrections[0],
            ground + corrected.states[0].energy,
        ];
        let density = parents.iter().map(|state| density(scf, state)).collect();
        drop(parent);
        Self {
            molecule: setup.molecule,
            overlap: scf.overlap.clone(),
            density,
            energies,
            gap: parents[1].energy - parents[0].energy,
        }
    }
}
fn similarities(base: &Oracle, point: &Oracle) -> Vec<f64> {
    let n = base.overlap.rows;
    let mut atoms = base.molecule.atoms.clone();
    atoms.extend(point.molecule.atoms.iter().cloned());
    let basis = MolBasis::build(&Molecule::new(atoms), &data::load("sto-3g").unwrap()).unwrap();
    let all = overlap_matrix(&basis);
    let mut cross = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cross[(i, j)] = all[(i, n + j)];
        }
    }
    let product = |a: &[Matrix; 2], b: &[Matrix; 2], s: &Matrix| {
        (0..2)
            .map(|spin| a[spin].frobenius_dot(&s.matmul_seq(&b[spin]).matmul_transpose_seq(s)))
            .sum::<f64>()
    };
    let left = product(&base.density[0], &base.density[0], &base.overlap);
    point
        .density
        .iter()
        .map(|d| {
            product(&base.density[0], d, &cross).abs()
                / (left * product(d, d, &point.overlap)).sqrt()
        })
        .collect()
}

fn complete(uhf: bool) {
    let system = system();
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let input = values(uhf);
    let base = Oracle::new(&system, &input);
    let controls = Controls::from_value(&input).unwrap();
    let analytic = run("td-hessian", &system, &controls).unwrap();
    assert_eq!(
        analytic.get("provenance").unwrap().as_str(),
        Some("analytic")
    );
    assert!(analytic
        .get("response_converged")
        .unwrap()
        .as_bool()
        .unwrap());
    assert!(scalar(&analytic, "parent_pole_defect") < 1e-10);
    let totalg = array(&analytic, "gradient");
    let exg = array(&analytic, "excitation_gradient");
    let cg = array(&analytic, "excitation_correlation_gradient");
    let totalh = array(&analytic, "hessian");
    let exh = array(&analytic, "excitation_hessian");
    let ch = array(&analytic, "excitation_correlation_hessian");
    let gradients = [
        totalg
            .iter()
            .zip(&exg)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
        exg.iter().zip(&cg).map(|(a, b)| a - b).collect(),
        cg,
        totalg,
    ];
    let hessians = [
        totalh
            .iter()
            .zip(&exh)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
        exh.iter().zip(&ch).map(|(a, b)| a - b).collect(),
        ch,
        totalh,
    ];
    let energies = [
        scalar(&analytic, "energy"),
        scalar(&analytic, "parent_excitation_energy"),
        scalar(&analytic, "excitation_correlation_shift"),
        scalar(&analytic, "excited_state_energy"),
    ];
    for c in 0..4 {
        assert!((energies[c] - base.energies[c]).abs() < 1e-10);
    }
    let mut names = Vec::new();
    let mut target = Vec::new();
    for channel in 0..4 {
        for a in 0..12 {
            names.push(format!("correlated uhf={uhf} channel{channel} gradient{a}"));
            target.push(gradients[channel][a]);
        }
        for a in 0..12 {
            for b in 0..12 {
                names.push(format!(
                    "correlated uhf={uhf} channel{channel} Hessian{a},{b}"
                ));
                target.push(hessians[channel][12 * a + b]);
            }
        }
    }
    let mut cache: HashMap<[u64; 12], [f64; 4]> = HashMap::new();
    let mut min_overlap = 1.0_f64;
    let mut min_gap = base.gap;
    let mut at = |displacement: [f64; 12]| {
        *cache
            .entry(displacement.map(f64::to_bits))
            .or_insert_with(|| {
                if displacement.iter().all(|v| *v == 0.0) {
                    return base.energies;
                }
                let mut shifted = system.clone();
                for (i, r) in shifted.positions.iter_mut().flatten().enumerate() {
                    *r += displacement[i];
                }
                let point = Oracle::new(&shifted, &input);
                let overlap = similarities(&base, &point);
                let best = (0..overlap.len())
                    .max_by(|&i, &j| overlap[i].total_cmp(&overlap[j]))
                    .unwrap();
                assert_eq!(best, 0, "state changed {overlap:?}");
                assert!(
                    overlap[0] > 0.9,
                    "state overlap={overlap:?}, displacement={displacement:?}"
                );
                assert!(overlap[0] - overlap[1..].iter().copied().fold(0.0_f64, f64::max) > 0.5);
                min_overlap = min_overlap.min(overlap[0]);
                min_gap = min_gap.min(point.gap);
                point.energies
            })
    };
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 624];
            let zero = at([0.0; 12]);
            for (k, &h) in steps.iter().enumerate() {
                for a in 0..12 {
                    let mut plus = [0.0; 12];
                    plus[a] = h;
                    let p = at(plus);
                    plus[a] = -h;
                    let m = at(plus);
                    let extra = if stencil == stencils::Stencil::Fourth {
                        plus[a] = 2.0 * h;
                        let pp = at(plus);
                        plus[a] = -2.0 * h;
                        Some((pp, at(plus)))
                    } else {
                        None
                    };
                    for c in 0..4 {
                        let (pp, mm) = extra.map_or((0.0, 0.0), |(p, m)| (p[c], m[c]));
                        let index = 156 * c + a;
                        errors[index][k] =
                            (stencil.first(p[c], m[c], pp, mm, h) - target[index]).abs();
                    }
                    for b in 0..12 {
                        let mut fd = |h: f64| {
                            if a == b {
                                let mut x = [0.0; 12];
                                x[a] = h;
                                let p = at(x);
                                x[a] = -h;
                                let m = at(x);
                                std::array::from_fn(|c| (p[c] + m[c] - 2.0 * zero[c]) / (h * h))
                            } else {
                                let mut x = [0.0; 12];
                                x[a] = h;
                                x[b] = h;
                                let pp = at(x);
                                x[b] = -h;
                                let pm = at(x);
                                x[a] = -h;
                                let mm = at(x);
                                x[b] = h;
                                let mp = at(x);
                                std::array::from_fn(|c| {
                                    (pp[c] - pm[c] - mp[c] + mm[c]) / (4.0 * h * h)
                                })
                            }
                        };
                        let first: [f64; 4] = fd(h);
                        let value = if stencil == stencils::Stencil::Fourth {
                            let second = fd(2.0 * h);
                            std::array::from_fn(|c| (4.0 * first[c] - second[c]) / 3.0)
                        } else {
                            first
                        };
                        for c in 0..4 {
                            let index = 156 * c + 12 + 12 * a + b;
                            errors[index][k] = (value[c] - target[index]).abs();
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    if !failures.is_empty() {
        // Energy second differences lose precision for small fourth-order
        // coefficients. Every first derivative must first have passed its
        // independent static-energy fit. Fresh first-order gradients can then
        // check the remaining second-order recurrence without subtracting
        // nearly equal total energies. No second-order path enters this oracle.
        let pending: Vec<_> = failures
            .iter()
            .map(|name| names.iter().position(|n| n == name).unwrap())
            .collect();
        assert!(
            pending.iter().all(|&k| k % 156 >= 12),
            "independent first-order energy validation is incomplete: {failures:?}"
        );
        println!("retained unresolved energy-Hessian fits={}; supplement with independently energy-validated first-order gradients",pending.len());
        let pending_names: Vec<_> = pending
            .iter()
            .map(|&k| format!("{} from fresh first-order gradients", names[k]))
            .collect();
        let mut gradient_cache: HashMap<[u64; 12], [Vec<f64>; 4]> = HashMap::new();
        let mut gradient_at = |displacement: [f64; 12]| {
            at(displacement); // The same independent AO-overlap tracking gate.
            gradient_cache
                .entry(displacement.map(f64::to_bits))
                .or_insert_with(|| {
                    let mut point = system.clone();
                    for (i, r) in point.positions.iter_mut().flatten().enumerate() {
                        *r += displacement[i];
                    }
                    let options = input.clone().with("td_gradient_route", "path");
                    let report = run(
                        "td-state-gradient",
                        &point,
                        &Controls::from_value(&options).unwrap(),
                    )
                    .unwrap();
                    let total = array(&report, "gradient");
                    let excited = array(&report, "excitation_gradient");
                    let correction = array(&report, "excitation_correlation_gradient");
                    [
                        total.iter().zip(&excited).map(|(a, b)| a - b).collect(),
                        excited
                            .iter()
                            .zip(&correction)
                            .map(|(a, b)| a - b)
                            .collect(),
                        correction,
                        total,
                    ]
                })
                .clone()
        };
        let mut remaining = Vec::new();
        stencils::fit_components_staged(
            &pending_names,
            &[
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; pending.len()];
                for (k, &h) in steps.iter().enumerate() {
                    for (slot, &index) in pending.iter().enumerate() {
                        let channel = index / 156;
                        let element = index % 156 - 12;
                        let row = element / 12;
                        let column = element % 12;
                        let mut x = [0.0; 12];
                        x[column] = h;
                        let p = gradient_at(x);
                        x[column] = -h;
                        let m = gradient_at(x);
                        let (pp, mm) = if stencil == stencils::Stencil::Fourth {
                            x[column] = 2.0 * h;
                            let pp = gradient_at(x);
                            x[column] = -2.0 * h;
                            (pp[channel][row], gradient_at(x)[channel][row])
                        } else {
                            (0.0, 0.0)
                        };
                        let value = stencil.first(p[channel][row], m[channel][row], pp, mm, h);
                        errors[slot][k] = (value - target[index]).abs();
                    }
                }
                errors
            },
            &mut remaining,
        );
        components::assert_no_failures("supplemental fresh-gradient Hessian fits", &remaining);
    }
    println!("correlated uhf={uhf} accepted=624 fresh_points={} min_AO_overlap={min_overlap:.12e} min_parent_gap={min_gap:.12e}",cache.len());
}
#[test]
fn c1_restricted_correlated_tda_all_nuclear_components() {
    complete(false);
}
#[test]
fn c1_unrestricted_correlated_tda_all_nuclear_components() {
    complete(true);
}
