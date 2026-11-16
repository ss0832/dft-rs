// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-centre strain derivatives of the wire and slab reciprocal gauge.

use super::super::{exponential_derivatives, slab_zero_radial_derivatives};
use super::{indices, Complex, DimensionalCoulomb, Vec3, PI};
use std::array::from_fn;

/// Differentiated geometry for one ordered affine component.
struct Tangent {
    measure_log: f64,
    open_projector: [[f64; 3]; 3],
    open_displacement: [f64; 3],
    open_square: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

/// Add the derivative of D^m exp(i G.r - t^2 r.Q.r).
/// The logarithmic differential is quadratic in r, so its Cartesian
/// derivatives need only the original Hermite entries through degree m.
fn exponential_tangent(
    scratch: &[Complex],
    index: impl Fn([usize; 3]) -> usize,
    m: [usize; 3],
    constant: Complex,
    linear: [Complex; 3],
    quadratic: &[[f64; 3]; 3],
) -> Complex {
    let mut value = scratch[index(m)] * constant;
    for i in 0..3 {
        if m[i] == 0 {
            continue;
        }
        let mut lower = m;
        lower[i] -= 1;
        value += scratch[index(lower)] * linear[i] * m[i] as f64;
        for j in 0..3 {
            if lower[j] > 0 {
                let mut twice = lower;
                twice[j] -= 1;
                value += scratch[index(twice)] * (quadratic[i][j] * (m[i] * lower[j]) as f64);
            }
        }
    }
    value
}

/// Long-range cell derivative with the supplied displacements held fixed.
/// The same reciprocal vectors, Ewald split, zero modes and quadrature
/// rule as the primal Green function are retained.
pub(in crate::periodic) fn reciprocal_response(
    kernel: &DimensionalCoulomb,
    reciprocal: &[Vec3],
    r: Vec3,
    order: usize,
    dim: usize,
    out: &mut [[[f64; 3]; 9]],
) {
    let geometry = kernel.geometry();
    let dimension = geometry.dimension();
    let measure = geometry.measure();
    let alpha = kernel.params().alpha;
    let cartesian = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let columns = cartesian.map(|v| geometry.perpendicular(v));
    let q: [[f64; 3]; 3] = from_fn(|i| from_fn(|j| columns[j].get(i)));
    let p: [[f64; 3]; 3] = from_fn(|i| from_fn(|j| f64::from(i == j) - q[i][j]));
    let position = r.to_array();
    let open = geometry.perpendicular(r);
    let tangent: [Tangent; 9] = from_fn(|k| {
        let (a, b) = (k / 3, k % 3);
        // dQ = -Q A P - P A^T Q for A_ab = 1.
        let dq: [[f64; 3]; 3] = from_fn(|i| from_fn(|j| -q[i][a] * p[b][j] - p[i][b] * q[a][j]));
        let dr = dq.map(|row| dot(row, position));
        Tangent {
            measure_log: p[b][a],
            open_projector: dq,
            open_displacement: dr,
            open_square: dot(position, dr),
        }
    });
    let ix = |m: [usize; 3]| (m[0] * dim + m[1]) * dim + m[2];
    let derivatives: Vec<_> = indices(order).map(|m| (m, ix(m))).collect();
    let mut scratch = vec![Complex::ZERO; out.len()];
    let factor = 2.0 * PI.powf((dimension as f64 - 1.0) * 0.5) / measure;
    let open_square = open.norm2();
    for &wave in reciprocal {
        let g = wave.to_array();
        let dg: [[f64; 3]; 9] = from_fn(|k| {
            let (a, b) = (k / 3, k % 3);
            // A periodic reciprocal vector stays in the tilted subspace.
            from_fn(|i| -p[i][b] * g[a] + q[i][a] * g[b])
        });
        let wave_square = wave.norm2();
        let half_wave_square_derivative = dg.map(|change| dot(g, change) * 0.5);
        let phase_derivative = dg.map(|change| dot(change, position));
        let phase = wave.dot(r);
        let trig = Complex::new(phase.cos(), phase.sin());
        for (&x, &w) in kernel.nodes.iter().zip(&kernel.weights) {
            let t = alpha * x;
            let square = t * t;
            let scale = factor
                * w
                * alpha
                * t.powi(-(dimension as i32))
                * (-wave_square / (4.0 * square) - open_square * square).exp();
            if scale == 0.0 {
                continue;
            }
            exponential_derivatives(order, dim, open, q, wave, t, &mut scratch);
            for (k, change) in tangent.iter().enumerate() {
                let constant = Complex::new(
                    -change.measure_log
                        - half_wave_square_derivative[k] / square
                        - square * change.open_square,
                    phase_derivative[k],
                );
                let linear = from_fn(|i| {
                    Complex::new(-2.0 * square * change.open_displacement[i], dg[k][i])
                });
                let quadratic = change.open_projector.map(|row| row.map(|v| -square * v));
                for &(m, index) in &derivatives {
                    let value = exponential_tangent(&scratch, ix, m, constant, linear, &quadratic);
                    out[index][k][1] += scale * (trig * value).re;
                }
            }
        }
    }
    if dimension == 2 {
        let normal = columns
            .iter()
            .max_by(|a, b| a.norm2().total_cmp(&b.norm2()))
            .expect("Cartesian axes")
            .normalized()
            .to_array();
        let radial = slab_zero_radial_derivatives(dot(normal, position), alpha, measure, order + 1);
        let direction =
            |m: [usize; 3]| (0..3).map(|i| normal[i].powi(m[i] as i32)).product::<f64>();
        for (k, change) in tangent.iter().enumerate() {
            let (a, b) = (k / 3, k % 3);
            let dn = p[b].map(|value| -normal[a] * value);
            let dz = dot(dn, position);
            for &(m, index) in &derivatives {
                let degree = m.iter().sum::<usize>();
                let direction_derivative = (0..3)
                    .filter(|&i| m[i] > 0)
                    .map(|i| {
                        let mut lower = m;
                        lower[i] -= 1;
                        m[i] as f64 * dn[i] * direction(lower)
                    })
                    .sum::<f64>();
                out[index][k][2] = (radial[degree + 1] * dz - radial[degree] * change.measure_log)
                    * direction(m)
                    + radial[degree] * direction_derivative;
            }
        }
    } else {
        for (k, change) in tangent.iter().enumerate() {
            out[0][k][2] = -factor * alpha.ln() * change.measure_log;
        }
        for (&x, &w) in kernel.nodes.iter().zip(&kernel.weights) {
            let t = alpha * x;
            let square = t * t;
            let exponent = -open_square * square;
            let scale = factor * w / x;
            exponential_derivatives(order, dim, open, q, Vec3::zero(), t, &mut scratch);
            for (k, change) in tangent.iter().enumerate() {
                let constant = -change.measure_log - square * change.open_square;
                let linear = change
                    .open_displacement
                    .map(|value| Complex::from(-2.0 * square * value));
                let quadratic = change
                    .open_projector
                    .map(|row| row.map(|value| -square * value));
                out[0][k][2] += scale
                    * (-change.measure_log * exponent.exp_m1()
                        - square * change.open_square * exponent.exp());
                for &(m, index) in derivatives.iter().skip(1) {
                    let value =
                        exponential_tangent(&scratch, ix, m, constant.into(), linear, &quadratic);
                    out[index][k][2] += scale * exponent.exp() * value.re;
                }
            }
        }
    }
}
