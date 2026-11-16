// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent complex London quadrature and nonlinear learned-current
//! response. Every independent imaginary AO matrix element is fitted.
use super::*;
use crate::acceptance::components;
use crate::basis::data;
use crate::skala::{random_tensors, SkalaModel, SkalaSizes};
use crate::system::{Atom, Molecule};

fn mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[test]
fn learned_magnetic_partials_fit_all_london_and_current_elements() {
    let centers = [
        Vec3::new(0.13, -0.21, 0.19),
        Vec3::new(1.57, 0.16, -0.08),
        Vec3::new(-0.19, 1.93, 0.37),
        Vec3::new(0.42, 0.29, 2.28),
    ];
    let mol = Molecule::new(
        centers
            .iter()
            .map(|&position| Atom { z: 1, position })
            .collect(),
    );
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let points = vec![
        Vec3::new(0.44, 0.21, 0.72),
        Vec3::new(1.36, 0.49, 0.36),
        Vec3::new(0.08, 1.55, 0.86),
        Vec3::new(1.21, -0.13, 0.33),
        Vec3::new(0.61, 0.66, 2.0),
    ];
    let grid = MolecularGrid {
        points,
        weights: vec![0.043, 0.067, 0.039, 0.051, 0.062],
        raw_weights: vec![0.043, 0.067, 0.039, 0.051, 0.062],
        atom_of: vec![0, 1, 2, 1, 3],
    };
    let model = SkalaModel::from_tensors(&random_tensors(
        19,
        SkalaSizes {
            width: 6,
            channels: 2,
            shells: 1,
            correlation: 2,
            layers: 1,
            post: 2,
        },
    ))
    .unwrap();
    let n = mb.nao_sph;
    let density: [Matrix; 2] = std::array::from_fn(|s| {
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                p[(i, j)] = (0.3 + 0.07 * (i + s) as f64) * (0.3 + 0.07 * (j + s) as f64)
                    + if i == j { 0.13 + 0.03 * s as f64 } else { 0.0 };
            }
        }
        p
    });
    let refs = [&density[0], &density[1]];
    let fields = WeightedDensityPartials::learned(&model, &mol, &mb, &grid, refs).unwrap();
    let kernel = MetaCurrentKernel::from_partials(&fields);
    let origin = Vec3::new(0.47, -0.31, 0.82);
    let plain = ResolvedFunctional::hartree_fock();
    let london =
        giao_density_xc_fock_derivative(&plain, &mb, &grid, refs, origin, Some(&fields)).unwrap();
    let ao = eval_aos(&mb, &grid.points, true);
    let grad = ao.grad.as_ref().unwrap();
    let intervals = [
        [0.08, 0.06, 0.04, 0.03],
        [0.02, 0.015, 0.01, 0.0075],
        [0.2, 0.15, 0.1, 0.075],
    ];
    let pairs: Vec<_> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    let mut failures = Vec::new();
    for s in 0..2 {
        for b in 0..3 {
            let e = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ][b];
            // Evaluate complex AO products directly. Partials of the complete
            // nonlocal reverse pass are held fixed in this independent first
            // basis derivative; their first magnetic derivative vanishes.
            let complex_fock = |h: f64, m: usize, k: usize| {
                let mut sum = 0.0;
                for g in 0..grid.len() {
                    let r = grid.points[g] - origin;
                    let qm = 0.5 * e.dot((centers[m] - origin).cross(r));
                    let qk = 0.5 * e.dot((centers[k] - origin).cross(r));
                    let tm = e.cross(centers[m] - origin) * 0.5;
                    let tk = e.cross(centers[k] - origin) * 0.5;
                    let pm = ((h * qm).cos(), (h * qm).sin());
                    let pk = ((h * qk).cos(), -(h * qk).sin());
                    let vm = (pm.0 * ao.phi[(g, m)], pm.1 * ao.phi[(g, m)]);
                    let vk = (pk.0 * ao.phi[(g, k)], pk.1 * ao.phi[(g, k)]);
                    sum += fields.rho[s][g] * mul(vm, vk).1;
                    for d in 0..3 {
                        let gm = mul(pm, (grad[d][(g, m)], h * tm.get(d) * ao.phi[(g, m)]));
                        let gk = mul(pk, (grad[d][(g, k)], -h * tk.get(d) * ao.phi[(g, k)]));
                        sum += fields.gradient[s][g][d] * (mul(gm, vk).1 + mul(vm, gk).1)
                            + 0.5 * fields.tau[s][g] * mul(gm, gk).1;
                    }
                }
                sum
            };
            let names: Vec<_> = pairs
                .iter()
                .map(|(m, k)| format!("learned London s={s} B={b} ({m},{k})"))
                .collect();
            components::fit_components(
                &names,
                &intervals,
                |steps| {
                    pairs
                        .iter()
                        .map(|&(m, k)| {
                            std::array::from_fn(|i| {
                                let h = steps[i];
                                ((complex_fock(h, m, k) - complex_fock(-h, m, k)) / (2.0 * h)
                                    - london[s][b][(m, k)])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            let base = kernel
                .basis_fock(&mb, &grid, &density[s], origin, b, s)
                .unwrap();
            let current = |h: f64| -> Vec<[f64; 3]> {
                (0..grid.len())
                    .map(|g| {
                        let mut j = [0.0; 3];
                        for m in 0..n {
                            for k in 0..n {
                                let r = grid.points[g] - origin;
                                let qm = 0.5 * e.dot((centers[m] - origin).cross(r));
                                let qk = 0.5 * e.dot((centers[k] - origin).cross(r));
                                let tm = e.cross(centers[m] - origin) * 0.5;
                                let tk = e.cross(centers[k] - origin) * 0.5;
                                let pm = ((h * qm).cos(), (h * qm).sin());
                                let pk = ((h * qk).cos(), -(h * qk).sin());
                                let vm = (pm.0 * ao.phi[(g, m)], pm.1 * ao.phi[(g, m)]);
                                let vk = (pk.0 * ao.phi[(g, k)], pk.1 * ao.phi[(g, k)]);
                                for d in 0..3 {
                                    let gm =
                                        mul(pm, (grad[d][(g, m)], h * tm.get(d) * ao.phi[(g, m)]));
                                    let gk =
                                        mul(pk, (grad[d][(g, k)], -h * tk.get(d) * ao.phi[(g, k)]));
                                    j[d] +=
                                        0.5 * density[s][(m, k)] * (mul(vm, gk).1 - mul(gm, vk).1);
                                }
                            }
                        }
                        j
                    })
                    .collect()
            };
            let names: Vec<_> = pairs
                .iter()
                .map(|(m, k)| format!("learned basis current s={s} B={b} ({m},{k})"))
                .collect();
            components::fit_components(
                &names,
                &intervals,
                |steps| {
                    let currents: Vec<_> =
                        steps.iter().map(|&h| (current(h), current(-h))).collect();
                    pairs
                        .iter()
                        .map(|&(m, k)| {
                            std::array::from_fn(|i| {
                                let mut value = 0.0;
                                for g in 0..grid.len() {
                                    for d in 0..3 {
                                        let jd = grad[d][(g, m)] * ao.phi[(g, k)]
                                            - grad[d][(g, k)] * ao.phi[(g, m)];
                                        value -= 0.5
                                            * kernel.coefficient[s][g]
                                            * (currents[i].0[g][d] - currents[i].1[g][d])
                                            * jd
                                            / (2.0 * steps[i]);
                                    }
                                }
                                (value - base[(m, k)]).abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
    }

    let semilocal =
        crate::xc::resolve(&crate::xc::FunctionalSpec::Combined("tpss".into())).unwrap();
    let combined =
        giao_density_xc_fock_derivative(&semilocal, &mb, &grid, refs, origin, Some(&fields))
            .unwrap();
    let ordinary =
        giao_density_xc_fock_derivative(&semilocal, &mb, &grid, refs, origin, None).unwrap();
    for spin in 0..2 {
        for axis in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (combined[spin][axis][(i, j)]
                            - ordinary[spin][axis][(i, j)]
                            - london[spin][axis][(i, j)])
                            .abs()
                            < 1e-15
                    );
                }
            }
        }
    }
    let mut combined_current = MetaCurrentKernel::build(&semilocal, &mb, &grid, refs)
        .unwrap()
        .unwrap();
    let ordinary_current = MetaCurrentKernel::build(&semilocal, &mb, &grid, refs)
        .unwrap()
        .unwrap();
    combined_current.add(&kernel);
    for spin in 0..2 {
        for g in 0..grid.len() {
            assert!(
                (combined_current.coefficient[spin][g]
                    - ordinary_current.coefficient[spin][g]
                    - kernel.coefficient[spin][g])
                    .abs()
                    < 1e-15
            );
        }
    }
    let ranges = crate::skala::batch_ranges(grid.len());
    let (rho, g_density, tau) = crate::skala::grid_densities(&mb, &grid, refs, &ranges);
    let (points, positions) = crate::skala::grid_geometry(&mol, &grid);
    for spin in 0..2 {
        for &(m, k) in &pairs {
            let mut direction = Matrix::zeros(n, n);
            direction[(m, k)] = 1.0;
            direction[(k, m)] = -1.0;
            let production = kernel.response_fock(&mb, &grid, &direction, spin).unwrap();
            let current: Vec<[f64; 3]> = (0..grid.len())
                .map(|g| {
                    std::array::from_fn(|d| {
                        grad[d][(g, m)] * ao.phi[(g, k)] - grad[d][(g, k)] * ao.phi[(g, m)]
                    })
                })
                .collect();
            let finite = |h: f64| {
                let mut kinetic = tau.clone();
                for g in 0..grid.len() {
                    kinetic[spin][g] -=
                        0.5 * h * h * current[g].iter().map(|j| j * j).sum::<f64>() / rho[spin][g];
                }
                let out = model
                    .evaluate(
                        &crate::skala::SkalaInput {
                            atom_of: &grid.atom_of,
                            points: &points,
                            centers: &positions,
                            weights: &grid.weights,
                            atomic_weights: &grid.raw_weights,
                            rho: [&rho[0], &rho[1]],
                            gradient: [&g_density[0], &g_density[1]],
                            tau: [&kinetic[0], &kinetic[1]],
                        },
                        true,
                    )
                    .unwrap();
                pairs
                    .iter()
                    .map(|&(i, j)| {
                        let mut value = 0.0;
                        for g in 0..grid.len() {
                            for d in 0..3 {
                                value += 0.5 * out.d_tau[spin][g] / rho[spin][g]
                                    * h
                                    * current[g][d]
                                    * (ao.phi[(g, i)] * grad[d][(g, j)]
                                        - ao.phi[(g, j)] * grad[d][(g, i)]);
                            }
                        }
                        value
                    })
                    .collect::<Vec<f64>>()
            };
            let names: Vec<_> = pairs
                .iter()
                .map(|(i, j)| format!("learned current kernel s={spin} ({i},{j})<-({m},{k})"))
                .collect();
            components::fit_components(
                &names,
                &intervals,
                |steps| {
                    let mut errors = vec![[0.0; 4]; pairs.len()];
                    for (step, &h) in steps.iter().enumerate() {
                        let (plus, minus) = (finite(h), finite(-h));
                        for (a, &(i, j)) in pairs.iter().enumerate() {
                            errors[a][step] =
                                ((plus[a] - minus[a]) / (2.0 * h) - production[(i, j)]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("learned London and basis-current matrices", &failures);
}
