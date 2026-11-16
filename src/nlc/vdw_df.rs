// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Dion nonlocal van der Waals pair kernel, Phys. Rev. Lett. 92, 246401
//! (2004), equations (11)–(17). The original and second-generation vdW-DF
//! use the same kernel with distinct internal exchange gradients.

use crate::grid::gauss_legendre;
use crate::{DftError, Result};
use std::f64::consts::PI;

mod spatial;
mod table;
mod variant;
pub use spatial::{LocalDensity, MolecularPairGrid, WeightedDensity};
pub use table::PairKernelTable;
pub use variant::{MolecularVdwDfEnergy, Variant};

/// Internal exchange gradient in the plasmon model determining q₀.
#[derive(Clone, Copy, Debug)]
pub enum InternalExchange {
    /// Original vdW-DF gradient coefficient Z_ab = −0.8491.
    Dion,
    /// Revised vdW-DF2 gradient coefficient Z_ab = −1.887.
    Lee,
}

/// Unpolarized local inverse length q₀ in Bohr⁻¹ at a positive density.
/// `gradient_squared` is |∇rho|² and the local correlation is original PW92.
/// This supplies the internal plasmon scale rather than the outer exchange.
pub fn unpolarized_local_scale(
    density: f64,
    gradient_squared: f64,
    model: InternalExchange,
) -> Result<f64> {
    if !density.is_finite()
        || density <= 0.0
        || !gradient_squared.is_finite()
        || gradient_squared < 0.0
    {
        return Err(DftError::InvalidInput(
            "vdW-DF q0 needs positive density and a nonnegative finite squared gradient".into(),
        ));
    }
    let exchange_gradient = match model {
        InternalExchange::Dion => -0.8491,
        InternalExchange::Lee => -1.887,
    };
    let kf = (3.0 * PI * PI * density).cbrt();
    let ec = crate::xc::lda::pw92_epsilon_original(
        0.5 * density,
        0.5 * density,
        &crate::xc::lda::PW92_PARAMS,
    );
    let ex = -3.0 * kf / (4.0 * PI);
    let s2 = gradient_squared / (2.0 * kf * density).powi(2);
    let q0 = kf * (1.0 + ec / ex - exchange_gradient * s2 / 9.0);
    if !q0.is_finite() {
        return Err(DftError::InvalidInput(
            "vdW-DF local inverse length is nonfinite".into(),
        ));
    }
    Ok(q0)
}

#[derive(Clone, Copy)]
struct Node {
    coordinate_squared: f64,
    weight: f64,
    a_sin: f64,
    sinc_minus_cos: f64,
}

impl Node {
    fn new(a: f64, weight: f64) -> Self {
        let a2 = a * a;
        let (sin, cos) = a.sin_cos();
        let sinc_minus_cos = if a < 0.15 {
            a2 * (1.0 / 3.0
                + a2 * (-1.0 / 30.0 + a2 * (1.0 / 840.0 + a2 * (-1.0 / 45360.0 + a2 / 3991680.0))))
        } else {
            sin / a - cos
        };
        Self {
            coordinate_squared: a2,
            weight,
            a_sin: a * sin,
            sinc_minus_cos,
        }
    }

    fn frequency(self, length: f64) -> f64 {
        let rate = (4.0 * PI / 9.0) / length.powi(2);
        -0.5 * self.coordinate_squared / (-rate * self.coordinate_squared).exp_m1()
    }
}

/// Direct quadrature of the dimensionless vdW-DF pair kernel. A reusable
/// finite momentum grid integrates the exact integrand over [0, cutoff]².
/// For convergence, increase both the cutoff and the panel order, and
/// resolve the near-origin momentum panels against the smaller of d and d′.
pub struct PairKernelQuadrature {
    nodes: Vec<Node>,
}

impl PairKernelQuadrature {
    /// Create a composite Gauss–Legendre grid and resolve the origin with
    /// geometrically shrinking panels down to `smallest_panel`.
    pub fn new(cutoff: f64, order: usize, smallest_panel: f64) -> Result<Self> {
        if !cutoff.is_finite()
            || cutoff <= 0.0
            || !smallest_panel.is_finite()
            || smallest_panel <= 0.0
            || smallest_panel > cutoff
            || !(2..=64).contains(&order)
        {
            return Err(DftError::InvalidInput(
                "invalid vdW-DF pair-kernel momentum quadrature".into(),
            ));
        }
        // One panel per half trigonometric period is sufficient away from
        // the origin. Near zero, geometric refinement also resolves ν(a).
        let width = PI.min(cutoff);
        let mut end = smallest_panel.min(width);
        let panels = (cutoff / width).ceil() + (width / end).log2().ceil() + 1.0;
        if !panels.is_finite() || panels > 100_000.0 {
            return Err(DftError::InvalidInput(
                "vdW-DF momentum grid exceeds 100000 panels".into(),
            ));
        }
        let (legendre_nodes, legendre_weights) = gauss_legendre(order);
        let mut nodes = Vec::with_capacity(order * panels as usize);
        let mut start = 0.0;
        while start < cutoff {
            let half = 0.5 * (end - start);
            let center = start + half;
            nodes.extend(
                legendre_nodes
                    .iter()
                    .zip(&legendre_weights)
                    .map(|(&x, &w)| Node::new(center + half * x, half * w)),
            );
            start = end;
            end = (end + end.min(width)).min(cutoff);
        }
        Ok(Self { nodes })
    }

    /// Evaluate phi(d,d′) in Hartree for strictly positive dimensionless
    /// distances. The logarithmic coincidence singularity belongs in a
    /// spatial quadrature with the radial Jacobian.
    pub fn evaluate(&self, d: f64, dp: f64) -> Result<f64> {
        if !d.is_finite() || d <= 0.0 || !dp.is_finite() || dp <= 0.0 {
            return Err(DftError::InvalidInput(
                "the vdW-DF pair kernel needs finite positive d and d'".into(),
            ));
        }
        let f: Vec<_> = self
            .nodes
            .iter()
            .map(|node| [node.frequency(d), node.frequency(dp)])
            .collect();
        if f.iter().flatten().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(DftError::InvalidInput(
                "vdW-DF pair frequencies are outside the representable range".into(),
            ));
        }
        let mut sum = 0.0;
        let mut compensation = 0.0;
        for (i, (a, &[w, y])) in self.nodes.iter().zip(&f).enumerate() {
            for (j, (b, &[x, z])) in self.nodes[..=i].iter().zip(&f).enumerate() {
                // Combine a²b²W and T to cancel their prefactors 2 and 1/2.
                let angular = a.a_sin * b.sinc_minus_cos + b.a_sin * a.sinc_minus_cos
                    - 3.0 * a.sinc_minus_cos * b.sinc_minus_cos;
                let temporal = ((w + x).recip() + (y + z).recip())
                    * ((w + y).recip() * (x + z).recip() + (w + z).recip() * (x + y).recip());
                let multiplicity = if i == j { 1.0 } else { 2.0 };
                let increment =
                    multiplicity * a.weight * b.weight * angular * temporal - compensation;
                let next = sum + increment;
                compensation = (next - sum) - increment;
                sum = next;
            }
        }
        Ok((2.0 / (PI * PI)) * sum)
    }
}
