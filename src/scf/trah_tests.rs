// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent finite rotations validate every orbital-Hessian entry used
//! by collinear TRAH, on a nonsymmetric molecule and an unpruned grid.
use crate::acceptance::{c1_geometry, components, stencils};
use crate::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use crate::driver::{Controls, System, Value};
use crate::linalg::Matrix;
use crate::scf::soscf::OrbitalHessian;

#[test]
fn trah_every_collinear_curvature_matches_independent_fock_rotations() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    );
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let mut failures = Vec::new();
    for xc in ["hf", "svwn", "pbe", "tpss", "pbe0"] {
        for unrestricted in [false, true] {
            let controls = Controls::from_value(
                &Value::object()
                    .with("xc", xc)
                    .with("basis", "sto-3g")
                    .with("dispersion", "none")
                    .with("grid_level", 1usize)
                    .with("grid_weight_cutoff", 0.0)
                    .with("unrestricted", unrestricted)
                    .with("conv_tol", 1e-11)
                    .with("conv_tol_dp", 1e-11)
                    .with("scf_optimizer", "legacy"),
            )
            .unwrap();
            let method = MolecularMethod::from_controls(&controls).unwrap();
            let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
            let seed_controls = Controls::from_value(
                &Value::object()
                    .with("xc", "hf")
                    .with("basis", "sto-3g")
                    .with("dispersion", "none")
                    .with("unrestricted", unrestricted)
                    .with("scf_optimizer", "legacy"),
            )
            .unwrap();
            let seed_method = MolecularMethod::from_controls(&seed_controls).unwrap();
            let seed_setup =
                prepare_molecular(&system.numbers, &system.positions, &seed_method).unwrap();
            let mut r = run_molecular_reference(&seed_setup).unwrap().scf;
            // No target-functional SCF is needed: this checks its complete
            // energy/Fock derivative at a deliberately nonstationary state.
            // Move away from stationarity so every cross curvature is tested,
            // including the complete noncanonical occupied/virtual Fock blocks.
            let n = r.mo_coeff[0].cols;
            for spin in 0..2 {
                let no = [r.n_alpha, r.n_beta][spin];
                let a = no;
                let i = 0;
                let t: f64 = 0.137
                    + if unrestricted {
                        0.061 * spin as f64
                    } else {
                        0.0
                    };
                let mut u = Matrix::identity(n);
                u[(i, i)] = t.cos();
                u[(a, a)] = t.cos();
                u[(a, i)] = t.sin();
                u[(i, a)] = -t.sin();
                r.mo_coeff[spin] = r.mo_coeff[spin].matmul_seq(&u);
                r.density[spin] = r.mo_coeff[spin].leading_columns_gram(no, 1.0);
            }
            let jk = crate::scf::fock::JkEngine::from_options(
                &setup.molecule,
                &setup.basis,
                &setup.options,
            )
            .unwrap();
            let model = super::MolecularDensityModel {
                mol: &setup.molecule,
                mb: &setup.basis,
                opts: &setup.options,
                functional: &setup.functional,
                grid: setup.grid.as_ref(),
                h_core: &r.h_core,
                extension: None,
                jk: &jk,
                jk_lr: None,
                pcm_solver: None,
                nlc_grid: None,
                restricted: !unrestricted,
            };
            r.fock = model.evaluate(&r.density).unwrap().fock;
            let kernel = crate::response::ResponseSolver::kernel_with_engines(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &r,
                &setup.options,
                &jk,
                None,
            )
            .unwrap();
            let oh = OrbitalHessian {
                c: &r.mo_coeff,
                eps: &r.mo_energy,
                n_occ: [r.n_alpha, r.n_beta],
                spins: if unrestricted { 2 } else { 1 },
            };
            let m = oh.len();
            for col in 0..m {
                let mut e = vec![0.0; m];
                e[col] = 1.0;
                let mut exact = oh.project(&kernel.fock_response(&oh.delta_density(&e)).unwrap());
                for (a, b) in exact.iter_mut().zip(oh.fock_commutator(&e, &r.fock)) {
                    *a += b;
                }
                // Exact single-plane sine/cosine rotation is independent of
                // the production exponential and its analytic density map.
                let at = |t: f64| {
                    let mut c = r.mo_coeff.clone();
                    let mut slot = col;
                    for spin in 0..oh.spins {
                        let no = [r.n_alpha, r.n_beta][spin];
                        let length = no * (n - no);
                        if slot < length {
                            let a = no + slot / no;
                            let i = slot % no;
                            let mut u = Matrix::identity(n);
                            u[(a, a)] = t.cos();
                            u[(i, i)] = t.cos();
                            u[(a, i)] = t.sin();
                            u[(i, a)] = -t.sin();
                            c[spin] = c[spin].matmul_seq(&u);
                            break;
                        }
                        slot -= length;
                    }
                    if !unrestricted {
                        c[1] = c[0].clone();
                    }
                    let density = std::array::from_fn(|s| {
                        c[s].leading_columns_gram([r.n_alpha, r.n_beta][s], 1.0)
                    });
                    let f = model.evaluate(&density).unwrap().fock;
                    // Transport the trial gradient back to the original chart.
                    // At the origin its derivative is the symmetric Hessian.
                    OrbitalHessian {
                        c: &c,
                        eps: &r.mo_energy,
                        n_occ: [r.n_alpha, r.n_beta],
                        spins: oh.spins,
                    }
                    .gradient(&f)
                };
                let names = (0..m)
                    .map(|row| format!("TRAH {xc} U={unrestricted} H[{row},{col}]"))
                    .collect::<Vec<_>>();
                stencils::fit_components_staged(
                    &names,
                    &[
                        ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                        ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                        ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
                    ],
                    |steps, stencil| {
                        let mut errors = vec![[0.0; 4]; m];
                        for (k, &h) in steps.iter().enumerate() {
                            let p = at(h);
                            let q = at(-h);
                            let (p2, q2) = if stencil == stencils::Stencil::Fourth {
                                (at(2.0 * h), at(-2.0 * h))
                            } else {
                                (vec![0.0; m], vec![0.0; m])
                            };
                            for row in 0..m {
                                errors[row][k] = (stencil
                                    .first(p[row], q[row], p2[row], q2[row], h)
                                    - exact[row])
                                    .abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
        }
    }
    components::assert_no_failures("every TRAH collinear orbital-Hessian element", &failures);
}
