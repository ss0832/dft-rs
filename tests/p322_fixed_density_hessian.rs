// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Off-shell nuclear Hessians and Fock derivatives without fabricated SCF states.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::basis::{data, BasisSet, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::hessian::fixed_density::fixed_density_hessian_terms;
use dft_rs::integrals::{kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::scf::{fock::JkEngine, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

fn fixture() -> Molecule {
    Molecule::new(
        [
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ]
        .into_iter()
        .map(|r| Atom {
            z: 1,
            position: Vec3::new(r[0], r[1], r[2]),
        })
        .collect(),
    )
}

fn densities(unrestricted: bool) -> [Matrix; 2] {
    std::array::from_fn(|spin| {
        let mut c = Matrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let bias = if unrestricted {
                    spin as f64 * 0.13
                } else {
                    0.0
                };
                c[(i, j)] = if i == j {
                    0.58 + 0.03 * i as f64 + bias
                } else {
                    0.06 * (1.0 + i as f64 + 3.0 * j as f64 + bias).sin()
                };
            }
        }
        c.matmul_transpose_seq(&c)
    })
}

// The independent displaced oracle uses the zero-order integral/XC builders
// and the separately assembled fixed-density analytic gradient. No SCF or
// Hessian response routine participates in the finite differences.
fn oracle(
    mol: &Molecule,
    basis: &BasisSet,
    functional: &ResolvedFunctional,
    density: &[Matrix; 2],
    options: &ScfOptions,
    grid_options: &GridOptions,
) -> Vec<f64> {
    let mb = MolBasis::build(mol, basis).unwrap();
    let grid = MolecularGrid::build(mol, grid_options).unwrap();
    let p = [&density[0], &density[1]];
    let gradient = dft_rs::gradient::assemble_gradient_density(
        mol,
        &mb,
        functional,
        Some(&grid),
        p,
        options,
        &Matrix::zeros(4, 4),
    )
    .unwrap();
    let mut total = density[0].clone();
    for (x, y) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
        *x += y;
    }
    let engine = JkEngine::new(
        &mb,
        EriMode::InCore { limit_mb: 64 },
        CoulombKernel::Coulomb,
    )
    .unwrap();
    let (j, k) = engine.build(&mb, &total, p, true).unwrap();
    let t = kinetic_matrix(&mb);
    let v = nuclear_matrix(&mb, mol, None);
    let xc = if functional.parts.is_empty() {
        None
    } else {
        Some(dft_rs::ks::xc_contribution(functional, &mb, &grid, p).unwrap())
    };
    let mut result = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .collect::<Vec<_>>();
    for spin in 0..2 {
        for i in 0..4 {
            for l in 0..4 {
                result.push(
                    t[(i, l)] + v[(i, l)] + j[(i, l)] - functional.hybrid.cx * k[spin][(i, l)]
                        + xc.as_ref().map_or(0.0, |q| q.vxc[spin][(i, l)]),
                );
            }
        }
    }
    result
}

fn fit(name: &str, unrestricted: bool) {
    let mol = fixture();
    let basis = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &basis).unwrap();
    let functional = if name == "hf" {
        ResolvedFunctional::hartree_fock()
    } else {
        resolve(&FunctionalSpec::Combined(name.into())).unwrap()
    };
    let grid_options = GridOptions {
        level: 0,
        weight_cutoff: 0.0,
        ..Default::default()
    };
    let grid = MolecularGrid::build(&mol, &grid_options).unwrap();
    let density = densities(unrestricted);
    let options = ScfOptions::default();
    let terms = fixed_density_hessian_terms(
        &mol,
        &mb,
        &options,
        &functional,
        [&density[0], &density[1]],
        Some(&grid),
    )
    .unwrap();
    let dim = 12;
    let width = dim + 2 * 16;
    let names = (0..dim).flat_map(|coordinate| (0..width).map(move |entry|
        format!("fixed-density {name} unequal={unrestricted} coordinate={coordinate} entry={entry}"))).collect::<Vec<_>>();
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            (
                [0.005, 0.0025, 0.00125, 0.000625],
                stencils::Stencil::Second,
            ),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for coordinate in 0..dim {
                for (s, &h) in steps.iter().enumerate() {
                    let displaced = |shift| {
                        let mut trial = mol.clone();
                        let r = &mut trial.atoms[coordinate / 3].position;
                        match coordinate % 3 {
                            0 => r.x += shift,
                            1 => r.y += shift,
                            _ => r.z += shift,
                        };
                        oracle(
                            &trial,
                            &basis,
                            &functional,
                            &density,
                            &options,
                            &grid_options,
                        )
                    };
                    let plus = displaced(h);
                    let minus = displaced(-h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (displaced(2.0 * h), displaced(-2.0 * h))
                    } else {
                        (vec![0.0; width], vec![0.0; width])
                    };
                    for entry in 0..width {
                        let analytic = if entry < dim {
                            terms.hessian[(entry, coordinate)]
                        } else {
                            let f = entry - dim;
                            terms.fock_derivatives[f / 16][coordinate / 3][coordinate % 3]
                                [((f % 16) / 4, f % 4)]
                        };
                        errors[coordinate * width + entry][s] = (stencil.first(
                            plus[entry],
                            minus[entry],
                            plus2[entry],
                            minus2[entry],
                            h,
                        ) - analytic)
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "off-shell nuclear Hessian and both spin Fock derivatives",
        &failures,
    );
}

macro_rules! cases {
    ($(($case:ident,$functional:expr,$unequal:expr)),* $(,)?) => {$(
        #[test] fn $case() {fit($functional,$unequal);}
    )*};
}
cases! {
    (equal_hf,"hf",false), (unequal_hf,"hf",true),
    (equal_lda,"svwn5",false), (unequal_lda,"svwn5",true),
    (equal_gga,"pbe",false), (unequal_gga,"pbe",true),
    (equal_meta,"tpss",false), (unequal_meta,"tpss",true),
    (equal_hybrid,"pbe0",false), (unequal_hybrid,"pbe0",true),
}
