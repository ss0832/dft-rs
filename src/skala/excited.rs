// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Third-order terms of the Skala functional for the gradient of an excited
//! state.
//!
//! The stationary Lagrangian of a Tamm-Dancoff state holds the energy of the
//! functional at the ground-state matrices `P`, its first directional
//! derivative along the relaxed difference density `R` and a quarter of its
//! second directional derivative along the transition density `T`:
//!
//! ```text
//! L = E[P] + d/dl E[P + l R] + (1/4) d2/dm2 E[P + m T]
//! ```
//!
//! The densities at the grid points are linear in the matrices, so the
//! reverse pass of the network evaluated on numbers that carry the densities
//! of `R` and `T` as nested directional derivatives gives every derivative of
//! `(dE/drho, dE/dgrad rho, dE/dtau)` and of the energy density that the
//! Lagrangian needs, exactly for the non-local layers.

use super::{
    assemble, batch_ranges, gradient_contraction, grid_densities, grid_geometry, SkalaInput,
    SkalaModel, SkalaOutput,
};
use crate::basis::MolBasis;
use crate::dual::DualN;
use crate::error::Result;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;

/// One real part of the reverse pass at every grid point.
struct Potentials {
    rho: [Vec<f64>; 2],
    gradient: [Vec<[f64; 3]>; 2],
    tau: [Vec<f64>; 2],
}

fn extract<S: Copy>(out: &SkalaOutput<S>, part: impl Fn(S) -> f64) -> Potentials {
    Potentials {
        rho: std::array::from_fn(|s| out.d_rho[s].iter().map(|v| part(*v)).collect()),
        gradient: std::array::from_fn(|s| {
            out.d_gradient[s]
                .iter()
                .map(|g| [part(g[0]), part(g[1]), part(g[2])])
                .collect()
        }),
        tau: std::array::from_fn(|s| out.d_tau[s].iter().map(|v| part(*v)).collect()),
    }
}

/// Second directional derivative of the Skala Kohn-Sham matrices along the
/// change `dp` of the symmetric spin density matrices,
/// `sum_tu d3E/(dP_s dP_t dP_u) dp_t dp_u` for each spin `s`.
pub fn skala_third_contraction(
    model: &SkalaModel,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    dp: [&Matrix; 2],
) -> Result<[Matrix; 2]> {
    type Inner = DualN<f64, 1>;
    type Outer = DualN<Inner, 1>;
    let ranges = batch_ranges(grid.len());
    let (rho, gradient, tau) = grid_densities(mb, grid, p, &ranges);
    let (d_rho, d_gradient, d_tau) = grid_densities(mb, grid, dp, &ranges);
    // The same direction in both levels: the mixed part is the second
    // derivative along it.
    let joined = |value: f64, slope: f64| Outer {
        re: Inner {
            re: value,
            im: [slope],
        },
        im: [Inner {
            re: slope,
            im: [0.0],
        }],
    };
    let scalars = |values: &[f64], slopes: &[f64]| -> Vec<Outer> {
        values
            .iter()
            .zip(slopes)
            .map(|(v, t)| joined(*v, *t))
            .collect()
    };
    let rho_j: [Vec<Outer>; 2] = std::array::from_fn(|s| scalars(&rho[s], &d_rho[s]));
    let tau_j: [Vec<Outer>; 2] = std::array::from_fn(|s| scalars(&tau[s], &d_tau[s]));
    let gradient_j: [Vec<[Outer; 3]>; 2] = std::array::from_fn(|s| {
        gradient[s]
            .iter()
            .zip(&d_gradient[s])
            .map(|(v, t)| std::array::from_fn(|axis| joined(v[axis], t[axis])))
            .collect()
    });
    let (points, centers) = grid_geometry(mol, grid);
    let weights: Vec<Outer> = grid.weights.iter().map(|w| joined(*w, 0.0)).collect();
    let out = model.evaluate(
        &SkalaInput {
            atom_of: &grid.atom_of,
            points: &points,
            centers: &centers,
            weights: &weights,
            atomic_weights: &grid.raw_weights,
            rho: [&rho_j[0], &rho_j[1]],
            gradient: [&gradient_j[0], &gradient_j[1]],
            tau: [&tau_j[0], &tau_j[1]],
        },
        true,
    )?;
    let second = extract(&out, |v: Outer| v.im[0].im[0]);
    Ok(assemble(
        mb,
        grid,
        &ranges,
        [&second.rho[0], &second.rho[1]],
        [&second.gradient[0], &second.gradient[1]],
        [&second.tau[0], &second.tau[1]],
    ))
}

/// Nuclear gradient, at fixed symmetric spin matrices, of the Skala part of
/// the Lagrangian of a Tamm-Dancoff state,
/// `E[P] + d/dl E[P + l R] + (1/4) d2/dm2 E[P + m T]` with the ground-state
/// matrices `ground`, the relaxed difference matrices `relaxed` and the
/// transition matrices `transition`. The basis functions, the grid points
/// riding their atoms and the partition weights all move.
pub fn skala_lagrangian_gradient(
    model: &SkalaModel,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    ground: [&Matrix; 2],
    relaxed: [&Matrix; 2],
    transition: [&Matrix; 2],
) -> Result<Vec<Vec3>> {
    // Inner directions: (T, R); outer direction: T.
    type Inner = DualN<f64, 2>;
    type Outer = DualN<Inner, 1>;
    let ranges = batch_ranges(grid.len());
    let (rho, gradient, tau) = grid_densities(mb, grid, ground, &ranges);
    let (r_rho, r_gradient, r_tau) = grid_densities(mb, grid, relaxed, &ranges);
    let (t_rho, t_gradient, t_tau) = grid_densities(mb, grid, transition, &ranges);
    let joined = |value: f64, t: f64, r: f64| Outer {
        re: Inner {
            re: value,
            im: [t, r],
        },
        im: [Inner {
            re: t,
            im: [0.0, 0.0],
        }],
    };
    let scalars = |values: &[f64], t: &[f64], r: &[f64]| -> Vec<Outer> {
        (0..values.len())
            .map(|i| joined(values[i], t[i], r[i]))
            .collect()
    };
    let rho_j: [Vec<Outer>; 2] = std::array::from_fn(|s| scalars(&rho[s], &t_rho[s], &r_rho[s]));
    let tau_j: [Vec<Outer>; 2] = std::array::from_fn(|s| scalars(&tau[s], &t_tau[s], &r_tau[s]));
    let gradient_j: [Vec<[Outer; 3]>; 2] = std::array::from_fn(|s| {
        (0..gradient[s].len())
            .map(|i| {
                std::array::from_fn(|axis| {
                    joined(
                        gradient[s][i][axis],
                        t_gradient[s][i][axis],
                        r_gradient[s][i][axis],
                    )
                })
            })
            .collect()
    });
    let (points, centers) = grid_geometry(mol, grid);
    let weights: Vec<Outer> = grid.weights.iter().map(|w| joined(*w, 0.0, 0.0)).collect();
    let out = model.evaluate(
        &SkalaInput {
            atom_of: &grid.atom_of,
            points: &points,
            centers: &centers,
            weights: &weights,
            atomic_weights: &grid.raw_weights,
            rho: [&rho_j[0], &rho_j[1]],
            gradient: [&gradient_j[0], &gradient_j[1]],
            tau: [&tau_j[0], &tau_j[1]],
        },
        true,
    )?;
    // With `a` standing for (dE/drho, dE/dgrad rho, dE/dtau) and `e` for the
    // energy density, and `D` for the nuclear derivative of the densities of
    // a matrix or of the weights:
    // dL = (a + a_R + a_TT / 4) . D[P] + a . D[R] + (a_T / 2) . D[T]
    //    + (e + e_R + e_TT / 4) D[w].
    let total = |v: Outer| v.re.re + v.re.im[1] + 0.25 * v.im[0].im[0];
    let on_ground = extract(&out, total);
    let on_relaxed = extract(&out, |v: Outer| v.re.re);
    let on_transition = extract(&out, |v: Outer| 0.5 * v.re.im[0]);
    let energy: Vec<f64> = out.energy_density.iter().map(|v| total(*v)).collect();
    let none = vec![0.0; grid.len()];
    let mut gradient = vec![Vec3::zero(); mol.atoms.len()];
    for (matrices, potentials, weighted) in [
        (ground, &on_ground, &energy),
        (relaxed, &on_relaxed, &none),
        (transition, &on_transition, &none),
    ] {
        let part = gradient_contraction(
            mol,
            mb,
            grid,
            &ranges,
            matrices,
            [&potentials.rho[0], &potentials.rho[1]],
            [&potentials.gradient[0], &potentials.gradient[1]],
            [&potentials.tau[0], &potentials.tau[1]],
            weighted,
        );
        for (target, value) in gradient.iter_mut().zip(part) {
            *target += value;
        }
    }
    Ok(gradient)
}
