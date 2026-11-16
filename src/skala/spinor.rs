// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Global spin projection of the complete learned grid energy.
//! The same angular direction is used at every point before the nonlocal
//! network is evaluated. Its spin-scaling derivative restores the existing
//! collinear model; averaging pointwise energy densities independently would
//! define a different nonlocal functional. Raw tau retains the learned model's
//! convention. Electromagnetic gauge extensions are not defined here.
use super::{Number, SkalaInput, SkalaModel};
use crate::dual::{DualN, Scalar};
use crate::grid::gauss_legendre;
use crate::periodic::spinor_xc::MAX_SPIN_ANGULAR_ORDER;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Whole-grid inputs with the usual Pauli density/gradient/tau/current layout.
/// Current slots are inactive because the existing learned model uses raw tau.
pub struct SpinorSkalaInput<'a, S> {
    pub atom_of: &'a [usize],
    pub points: &'a [[f64; 3]],
    pub centers: &'a [[f64; 3]],
    pub weights: &'a [S],
    pub atomic_weights: &'a [f64],
    pub descriptors: &'a [[S; 32]],
}
/// Energy and weighted descriptor potentials, including nonlocal grid terms.
pub struct SpinorSkalaOutput<S> {
    pub energy: S,
    pub gradient: Vec<[S; 32]>,
    pub angular_order: usize,
}
fn magnetic(i: usize) -> bool {
    match i {
        0..=3 => i != 0,
        4..=15 => (i - 4) / 3 != 0,
        _ => i != 16,
    }
}
type Inner<S> = DualN<S, 1>;
type Jet<S> = DualN<Inner<S>, 20>;

/// Variational global spin projection with an explicit angular rule.
/// Taylor/nested directional scalars preserve all grid cross derivatives.
/// Consecutive-rule convergence is the caller's independent responsibility.
pub fn spinor_skala_grid<S: Number>(
    model: &SkalaModel,
    input: &SpinorSkalaInput<'_, S>,
    angular_order: usize,
) -> Result<SpinorSkalaOutput<S>> {
    let n = input.descriptors.len();
    if n == 0
        || input.points.len() != n
        || input.weights.len() != n
        || input.atom_of.len() != n
        || input.atomic_weights.len() != n
        || !(4..=MAX_SPIN_ANGULAR_ORDER).contains(&angular_order)
        || angular_order % 2 != 0
        || input
            .descriptors
            .iter()
            .flatten()
            .any(|x| !x.value().is_finite())
        || input.weights.iter().any(|x| !x.value().is_finite())
        || input
            .points
            .iter()
            .chain(input.centers)
            .flatten()
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput("learned spin projection requires finite matching nonempty grid arrays and even angular order 4..128".into()));
    }
    for x in input.descriptors {
        if x[0].value() < x[1].value().hypot(x[2].value()).hypot(x[3].value()) {
            return Err(DftError::InvalidInput(
                "learned spin projection requires positive-semidefinite Pauli densities".into(),
            ));
        }
    }
    let zero = S::from_f64(0.0);
    let mut out = SpinorSkalaOutput {
        energy: zero,
        gradient: vec![[zero; 32]; n],
        angular_order,
    };
    let (nodes, weights) = gauss_legendre(angular_order);
    let azimuthal = 2 * angular_order;
    for (z, weight) in nodes.into_iter().zip(weights) {
        let transverse = (1.0 - z * z).sqrt();
        for j in 0..azimuthal {
            let angle = 2.0 * PI * (j as f64 + 0.5) / azimuthal as f64;
            let direction = [transverse * angle.cos(), transverse * angle.sin(), z];
            let q: Vec<[Jet<S>; 10]> = input
                .descriptors
                .iter()
                .map(|values| {
                    let x: [Jet<S>; 20] = std::array::from_fn(|i| {
                        let mut v = Jet::variable(
                            Inner {
                                re: values[i],
                                im: [if magnetic(i) { values[i] } else { zero }],
                            },
                            i,
                        );
                        v.im[i].im[0] = S::from_f64(if magnetic(i) { 1.0 } else { 0.0 });
                        v
                    });
                    let projected = |start: usize, stride: usize, sign: f64| {
                        (x[start]
                            + (0..3).fold(Jet::from_f64(0.0), |a, s| {
                                a + x[start + stride * (s + 1)].scale(direction[s] * sign)
                            }))
                        .scale(0.5)
                    };
                    std::array::from_fn(|a| match a {
                        0..=1 => projected(0, 1, if a == 0 { 1.0 } else { -1.0 }),
                        2..=7 => projected(4 + (a - 2) % 3, 3, if a < 5 { 1.0 } else { -1.0 }),
                        _ => projected(16, 1, if a == 8 { 1.0 } else { -1.0 }),
                    })
                })
                .collect();
            let rho: [Vec<Inner<S>>; 2] =
                std::array::from_fn(|s| q.iter().map(|p| p[s].re).collect());
            let gradient: [Vec<[Inner<S>; 3]>; 2] = std::array::from_fn(|s| {
                q.iter()
                    .map(|p| std::array::from_fn(|d| p[2 + 3 * s + d].re))
                    .collect()
            });
            let tau: [Vec<Inner<S>>; 2] =
                std::array::from_fn(|s| q.iter().map(|p| p[8 + s].re).collect());
            let quadrature: Vec<_> = input.weights.iter().map(|&v| Inner::constant(v)).collect();
            let network = model.evaluate(
                &SkalaInput {
                    atom_of: input.atom_of,
                    points: input.points,
                    centers: input.centers,
                    weights: &quadrature,
                    atomic_weights: input.atomic_weights,
                    rho: [&rho[0], &rho[1]],
                    gradient: [&gradient[0], &gradient[1]],
                    tau: [&tau[0], &tau[1]],
                },
                true,
            )?;
            let w = weight / (2.0 * azimuthal as f64);
            out.energy = out.energy + (network.energy.re + network.energy.im[0]).scale(w);
            for (p, target) in out.gradient.iter_mut().enumerate() {
                let potentials: [Inner<S>; 10] = std::array::from_fn(|a| match a {
                    0..=1 => network.d_rho[a][p],
                    2..=7 => network.d_gradient[(a - 2) / 3][p][(a - 2) % 3],
                    _ => network.d_tau[a - 8][p],
                });
                for (i, value) in target.iter_mut().enumerate().take(20) {
                    let mut v = zero;
                    for (a, potential) in potentials.iter().enumerate() {
                        let seed = q[p][a].im[i];
                        v = v + potential.re * (seed.re + seed.im[0]) + potential.im[0] * seed.re;
                    }
                    *value = *value + v.scale(w);
                }
            }
        }
    }
    Ok(out)
}
