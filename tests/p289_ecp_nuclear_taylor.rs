// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All AO entries of the ECP directional Taylor interface through order four.
//! Every nonstructural independent matrix entry has a four-step fit against
//! the scalar evaluator and separately implemented first-derivative matrices.
//! Central-shell/ECP co-motion gives exact structural zero derivatives and
//! is checked algebraically. Full nuclear tensor acceptance is separate in p290.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spdf_fixture.rs"]
mod spdf_fixture;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{Controls, Value};
use dft_rs::integrals::ecp::{ecp_derivative_matrices, ecp_matrix, ecp_nuclear_taylor};
use dft_rs::{Matrix, Vec3};
use std::collections::BTreeMap;

#[test]
fn every_spdf_ecp_path_matrix_entry_has_an_independent_fit() {
    let (system, controls) = spdf_fixture::fixture(true, false, "hf");
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let basis = data::load_or_read(controls.get("basis").and_then(Value::as_str).unwrap()).unwrap();
    let mut direction: Vec<[f64; 3]> = (0..4)
        .map(|atom| {
            std::array::from_fn(|axis| {
                0.3 * ((3 * atom + axis + 2) as f64).sin() + 0.07 * (atom + axis + 1) as f64
            })
        })
        .collect();
    let norm = direction.iter().flatten().fold(0.0_f64, |a, &b| a.hypot(b));
    for x in direction.iter_mut().flatten() {
        *x /= norm;
    }
    let exact = ecp_nuclear_taylor(&setup.basis, &direction, 4).unwrap();
    assert_eq!(setup.basis.max_l, 3);
    assert_eq!(setup.basis.n_core_electrons, 10);
    assert!(setup.basis.ecps[0]
        .potential
        .semilocal
        .iter()
        .all(|v| !v.is_empty()));
    assert!(exact[0].max_abs_diff(&ecp_matrix(&setup.basis).unwrap()) < 1e-12);
    let owner = setup.basis.sph_ao_atoms();
    let pairs = (0..owner.len())
        .flat_map(|i| (i..owner.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| owner[i] != 0 || owner[j] != 0)
        .collect::<Vec<_>>();
    for coefficient in &exact[1..] {
        for i in 0..owner.len() {
            for j in 0..owner.len() {
                if owner[i] == 0 && owner[j] == 0 {
                    assert_eq!(coefficient[(i, j)], 0.0);
                }
            }
        }
    }
    let mut cache = BTreeMap::new();
    let mut at = |t: f64| {
        cache
            .entry(t.to_bits())
            .or_insert_with(|| {
                let mut mol = setup.molecule.clone();
                for (atom, d) in mol.atoms.iter_mut().zip(&direction) {
                    atom.position += Vec3::new(d[0] * t, d[1] * t, d[2] * t);
                }
                let mb = MolBasis::build(&mol, &basis).unwrap();
                let mut gradient = Matrix::zeros(mb.nao_sph, mb.nao_sph);
                let derivative = ecp_derivative_matrices(&mb, 4);
                for atom in 0..4 {
                    for axis in 0..3 {
                        for (a, b) in gradient
                            .as_mut_slice()
                            .iter_mut()
                            .zip(derivative[atom][axis].as_slice())
                        {
                            *a += direction[atom][axis] * b;
                        }
                    }
                }
                (ecp_matrix(&mb).unwrap(), gradient)
            })
            .clone()
    };
    let mut failures = Vec::new();
    for order in 1..=4 {
        let factorial = (1..=order).product::<usize>() as f64;
        let names = pairs
            .iter()
            .map(|&(i, j)| format!("C1 spdf ECP path order={order} AO=({i},{j})"))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut finite = |h: f64| {
                    let (p, m) = (at(h), at(-h));
                    let z = if order == 3 { Some(at(0.0)) } else { None };
                    let far = if order == 4 {
                        Some((at(2.0 * h), at(-2.0 * h)))
                    } else {
                        None
                    };
                    pairs
                        .iter()
                        .map(|&(i, j)| match order {
                            1 => stencils::Stencil::Second.first(
                                p.0[(i, j)],
                                m.0[(i, j)],
                                0.0,
                                0.0,
                                h,
                            ),
                            2 => stencils::Stencil::Second.first(
                                p.1[(i, j)],
                                m.1[(i, j)],
                                0.0,
                                0.0,
                                h,
                            ),
                            3 => {
                                (p.1[(i, j)] - 2.0 * z.as_ref().unwrap().1[(i, j)] + m.1[(i, j)])
                                    / h.powi(2)
                            }
                            4 => {
                                let (p2, m2) = far.as_ref().unwrap();
                                (p2.1[(i, j)] - 2.0 * p.1[(i, j)] + 2.0 * m.1[(i, j)]
                                    - m2.1[(i, j)])
                                    / (2.0 * h.powi(3))
                            }
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>()
                };
                let mut errors = vec![[0.0; 4]; pairs.len()];
                for (step, &h) in steps.iter().enumerate() {
                    let near = finite(h);
                    let far = if stencil == stencils::Stencil::Fourth {
                        finite(2.0 * h)
                    } else {
                        vec![0.0; pairs.len()]
                    };
                    for (index, &(i, j)) in pairs.iter().enumerate() {
                        let value = if stencil == stencils::Stencil::Fourth {
                            (4.0 * near[index] - far[index]) / 3.0
                        } else {
                            near[index]
                        };
                        errors[index][step] = (value - factorial * exact[order][(i, j)]).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    components::assert_no_failures("all nonstructural spdf ECP path entries", &failures);
}
