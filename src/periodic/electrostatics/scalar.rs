// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The shared scalar Green function for energies, analytic forces and jets.
use super::*;
use std::f64::consts::PI;

fn norm2<S: Scalar>(r: [S; 3]) -> S {
    r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
}
fn exp_m1<S: Scalar>(x: S) -> S {
    let e = x.exp();
    // Rebase only the constant coefficient; retain all analytic derivatives.
    e - S::from_f64(e.value()) + S::from_f64(x.value().exp_m1())
}
fn erfc<S: Scalar>(x: S) -> S {
    x.erfcx() * (-(x * x)).exp()
}

impl DimensionalCoulomb {
    fn perpendicular_scalar<S: Scalar>(&self, r: [S; 3]) -> [S; 3] {
        let reciprocal = self.geometry.reciprocal();
        let fractions: [S; 3] = std::array::from_fn(|a| {
            (r[0].scale(reciprocal[a].x)
                + r[1].scale(reciprocal[a].y)
                + r[2].scale(reciprocal[a].z))
            .scale(1.0 / (2.0 * PI))
        });
        std::array::from_fn(|a| {
            r[a] - (0..3).fold(S::from_f64(0.0), |p, b| {
                p + fractions[b].scale(self.geometry.lattice().a[b].get(a))
            })
        })
    }
    fn real_scalar<S: Scalar>(
        &self,
        r: [S; 3],
        width: f64,
        exclude_home: bool,
    ) -> Result<(S, [S; 3])> {
        let alpha = self.params.alpha;
        let inverse = if width == 0.0 {
            f64::INFINITY
        } else {
            width.recip()
        };
        let zero = S::from_f64(0.0);
        let mut value = zero;
        let mut gradient = [zero; 3];
        let reference = Vec3::new(r[0].value(), r[1].value(), r[2].value());
        // Fix the image partition at the reference, including every moving
        // source and all open coordinates inside each retained image term.
        for index in self
            .geometry
            .nearby_translation_indices(reference, self.params.rcut)?
        {
            let t = self.geometry.lattice().translation(index);
            if exclude_home && t.norm2() < 1e-24 {
                continue;
            }
            let delta = std::array::from_fn(|a| r[a] + S::from_f64(t.get(a)));
            let r2 = norm2(delta);
            if r2.value() > self.params.rcut * self.params.rcut {
                continue;
            }
            if r2.value() == 0.0 {
                if width == 0.0 {
                    return Err(DftError::InvalidInput(
                        "distinct point sources coincide with a periodic image".into(),
                    ));
                }
                // erf(beta*r)/r-erf(alpha*r)/r is analytic in r^2 even
                // at coincident Gaussian centres. Never differentiate sqrt(0).
                let mut radial = zero;
                for (&x, &w) in self.nodes.iter().zip(&self.weights) {
                    let t = alpha + (inverse - alpha) * x;
                    let term = (r2.scale(-t * t)).exp().scale(w * (inverse - alpha));
                    value = value + term.scale(2.0 / PI.sqrt());
                    radial = radial - term.scale(4.0 * t * t / PI.sqrt());
                }
                for a in 0..3 {
                    gradient[a] = gradient[a] + delta[a] * radial;
                }
                continue;
            }
            if r2.value() < 1e-24 && width == 0.0 {
                return Err(DftError::InvalidInput(
                    "distinct point sources coincide with a periodic image".into(),
                ));
            }
            let distance = r2.sqrt();
            let short = erfc(distance.scale(alpha))
                - if width == 0.0 {
                    zero
                } else {
                    erfc(distance.scale(inverse))
                };
            value = value + short / distance;
            let smooth = if width == 0.0 {
                zero
            } else {
                (r2.scale(-inverse * inverse)).exp().scale(inverse)
            };
            let numerator =
                (smooth - (r2.scale(-alpha * alpha)).exp().scale(alpha)).scale(2.0 / PI.sqrt());
            let radial = (numerator - short / distance) / r2;
            for a in 0..3 {
                gradient[a] = gradient[a] + delta[a] * radial;
            }
        }
        Ok((value, gradient))
    }
    fn nonzero_scalar<S: Scalar>(&self, r: [S; 3], cos_sin: impl Fn(S) -> (S, S)) -> (S, [S; 3]) {
        let d = self.geometry.dimension();
        let measure = self.geometry.measure();
        let alpha = self.params.alpha;
        let perp = self.perpendicular_scalar(r);
        let rho2 = norm2(perp);
        let factor = 2.0 * PI.powf((d as f64 - 1.0) * 0.5) / measure;
        let zero = S::from_f64(0.0);
        let mut potential = zero;
        let mut gradient = [zero; 3];
        for &g in &self.reciprocal {
            let theta = r[0].scale(g.x) + r[1].scale(g.y) + r[2].scale(g.z);
            let (cos, sin) = cos_sin(theta);
            if d == 3 {
                let value =
                    4.0 * PI / measure * (-g.norm2() / (4.0 * alpha * alpha)).exp() / g.norm2();
                potential = potential + cos.scale(value);
                for a in 0..3 {
                    gradient[a] = gradient[a] - sin.scale(value * g.get(a));
                }
            } else {
                let (mut integral, mut derivative) = (zero, zero);
                for (&x, &w) in self.nodes.iter().zip(&self.weights) {
                    let t = alpha * x;
                    let exponential =
                        (rho2.scale(-t * t) + S::from_f64(-g.norm2() / (4.0 * t * t))).exp();
                    let term = exponential.scale(w * alpha * t.powi(-(d as i32)));
                    integral = integral + term;
                    derivative = derivative + term.scale(t * t);
                }
                potential = potential + (cos * integral).scale(factor);
                for a in 0..3 {
                    gradient[a] = gradient[a]
                        - (sin * integral).scale(factor * g.get(a))
                        - (perp[a] * cos * derivative).scale(2.0 * factor);
                }
            }
        }
        (potential, gradient)
    }
    fn zero_scalar<S: Scalar>(&self, r: [S; 3], width: f64) -> (S, [S; 3]) {
        let d = self.geometry.dimension();
        let measure = self.geometry.measure();
        let alpha = self.params.alpha;
        let perp = self.perpendicular_scalar(r);
        let rho2 = norm2(perp);
        let zero = S::from_f64(0.0);
        match d {
            3 => (
                S::from_f64(-PI / (measure * alpha * alpha) + PI * width * width / measure),
                [zero; 3],
            ),
            2 => {
                if rho2.value() == 0.0 {
                    let (mut integral, mut derivative) = (zero, zero);
                    for (&x, &w) in self.nodes.iter().zip(&self.weights) {
                        let exponential = (rho2.scale(-alpha * alpha * x * x)).exp();
                        integral = integral + exponential.scale(w * (1.0 - x));
                        derivative = derivative + exponential.scale(w);
                    }
                    let value = S::from_f64(1.0 / (alpha * PI.sqrt()))
                        + (rho2 * integral).scale(2.0 * alpha / PI.sqrt());
                    (
                        value.scale(-2.0 * PI / measure),
                        perp.map(|p| (p * derivative).scale(-4.0 * PI.sqrt() * alpha / measure)),
                    )
                } else {
                    let rho = rho2.sqrt();
                    let erf = rho.scale(alpha).erf();
                    let value = (rho * erf
                        + (rho2.scale(-alpha * alpha))
                            .exp()
                            .scale(1.0 / (alpha * PI.sqrt())))
                    .scale(-2.0 * PI / measure);
                    (
                        value,
                        perp.map(|p| (p * erf / rho).scale(-2.0 * PI / measure)),
                    )
                }
            }
            _ => {
                let b = rho2.scale(alpha * alpha);
                let mut integral = zero;
                let mut radial = zero;
                for (&x, &w) in self.nodes.iter().zip(&self.weights) {
                    integral = integral + exp_m1(b.scale(-x * x)).scale(w / x);
                    if rho2.value() == 0.0 {
                        radial = radial
                            + (b.scale(-x * x))
                                .exp()
                                .scale(-4.0 * alpha * alpha * w * x / measure);
                    }
                }
                if rho2.value() != 0.0 {
                    radial = (exp_m1(-b) / rho2).scale(2.0 / measure);
                }
                (
                    (S::from_f64(alpha.ln()) + integral).scale(2.0 / measure),
                    perp.map(|p| p * radial),
                )
            }
        }
    }
    pub(super) fn scalar_pair<S: Scalar>(
        &self,
        r: [S; 3],
        width: f64,
        exclude_home: bool,
        cos_sin: impl Fn(S) -> (S, S),
    ) -> Result<(ElectrostaticComponents<S>, [S; 3])> {
        let norm = r.iter().map(|x| x.value().powi(2)).sum::<f64>().sqrt();
        if !norm.is_finite()
            || !width.is_finite()
            || width < 0.0
            || exclude_home && (width != 0.0 || norm > 1e-12)
        {
            return Err(DftError::InvalidInput("Coulomb pair needs a finite displacement and non-negative width; only a point self interaction may exclude its home image".into()));
        }
        let (a, ga) = self.real_scalar(r, width, exclude_home)?;
        let (b, gb) = self.nonzero_scalar(r, cos_sin);
        let (c, gc) = self.zero_scalar(r, width);
        Ok((
            ElectrostaticComponents {
                real_space: a,
                reciprocal_nonzero: b,
                reciprocal_zero: c,
                point_self: S::from_f64(if exclude_home {
                    -2.0 * self.params.alpha / PI.sqrt()
                } else {
                    0.0
                }),
            },
            std::array::from_fn(|a| ga[a] + gb[a] + gc[a]),
        ))
    }
    /// Fixed-cell source energy with arbitrary scalar position derivatives.
    /// Charges, widths, Ewald controls and the reference image partition stay
    /// fixed. Include a prescribed Gaussian countercharge as an explicit source
    /// with constant coordinates. The cosine/sine callback returns (cos,sin).
    /// Coefficients retain the scalar convention (Taylor jets are factorial
    /// normalized); this is the same kernel as the ordinary energy and forces.
    pub fn energy_components_scalar<S: Scalar>(
        &self,
        sources: &[ElectrostaticSource],
        positions: &[[S; 3]],
        cos_sin: impl Fn(S) -> (S, S) + Copy,
    ) -> Result<ElectrostaticComponents<S>> {
        if sources.is_empty()
            || sources.len() != positions.len()
            || sources
                .iter()
                .any(|s| !s.charge.is_finite() || !s.width.is_finite() || s.width < 0.0)
        {
            return Err(DftError::InvalidInput("scalar Coulomb energy needs matching nonempty sources/positions, finite charges and nonnegative widths".into()));
        }
        let mut out = ElectrostaticComponents::<S>::default();
        for i in 0..sources.len() {
            for j in 0..=i {
                let r = std::array::from_fn(|a| positions[i][a] - positions[j][a]);
                let width = sources[i].width.hypot(sources[j].width);
                let (pair, _) = self.scalar_pair(r, width, i == j && width == 0.0, cos_sin)?;
                let scale = sources[i].charge * sources[j].charge * if i == j { 0.5 } else { 1.0 };
                out.real_space = out.real_space + pair.real_space.scale(scale);
                out.reciprocal_nonzero =
                    out.reciprocal_nonzero + pair.reciprocal_nonzero.scale(scale);
                out.reciprocal_zero = out.reciprocal_zero + pair.reciprocal_zero.scale(scale);
                out.point_self = out.point_self + pair.point_self.scale(scale);
            }
        }
        Ok(out)
    }
}
