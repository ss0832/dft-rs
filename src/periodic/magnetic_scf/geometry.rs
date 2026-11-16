// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Frozen-density geometry derivative of the commensurate field Hamiltonian.

use super::StaticPeriodicField;
use crate::basis::{cart_components, MolBasis};
use crate::dual::{DualN, Scalar};
use crate::math::Vec3;
use crate::periodic::planewave::{current_primitive, plane_wave_primitive_g};
use crate::periodic::strain::cartesian_density_with;
use crate::periodic::strain_jets::{cos_sin, strain_apply};
use crate::periodic::{BlochMatrix, PeriodicGeometry};
use crate::response::complex::Complex;
use crate::{DftError, Result};

type Jet = DualN<f64, 15>;

pub(crate) struct FieldGeometry {
    pub gradient: Vec<Vec3>,
    pub strain: [[f64; 3]; 3],
    pub energy: f64,
}

struct Harmonic {
    wave: [Jet; 3],
    scalar: Complex,
    vector: [Complex; 3],
}

fn wave_jet(geometry: &PeriodicGeometry, wave: Vec3) -> [Jet; 3] {
    let mut components = wave.to_array().map(Jet::constant);
    let axes = [
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    ];
    // Fixed reduced wave indices imply P dq = -P eps^T q. Rotation of
    // the periodic subspace supplies the open component (I-P) eps q.
    let parallel = axes.map(|v| geometry.parallel(v));
    for a in 0..3 {
        for b in 0..3 {
            let tangent = (axes[a] - parallel[a]) * wave.get(b) - parallel[b] * wave.get(a);
            for (axis, component) in components.iter_mut().enumerate() {
                component.im[6 + 3 * a + b] = tangent.get(axis);
            }
        }
    }
    components
}

impl StaticPeriodicField {
    pub(crate) fn geometry_derivative(
        &self,
        basis: &MolBasis,
        geometry: &PeriodicGeometry,
        points: &[Vec3],
        density: &[BlochMatrix],
        cutoff: f64,
        natoms: usize,
    ) -> Result<FieldGeometry> {
        if geometry.perpendicular(self.wave).norm() > 1e-12 * (1. + self.wave.norm()) {
            return Err(DftError::Unsupported {
                feature: "affine Fourier-field derivatives of an open-axis wave component".into(),
                missing: "a prescribed transport rule for the open-axis wave".into(),
            });
        }
        let eps =
            std::array::from_fn(|a| std::array::from_fn(|b| Jet::variable(0., 6 + 3 * a + b)));
        let seed = |r: Vec3, offset| {
            strain_apply(
                &eps,
                std::array::from_fn(|a| Jet::variable(r.get(a), offset + a)),
            )
        };
        let lambda = self.amplitude;
        let (square, norm) = self.quadratic_coefficients();
        let mut modes = Vec::new();
        for sign in [1., -1.] {
            let (scalar, vector) = self.linear_coefficients(sign > 0.);
            modes.push(Harmonic {
                wave: wave_jet(geometry, self.wave * sign),
                scalar: scalar * lambda,
                vector: vector.map(|v| v * lambda),
            });
            let contact = if sign > 0. { square } else { square.conj() };
            modes.push(Harmonic {
                wave: wave_jet(geometry, self.wave * (2. * sign)),
                scalar: contact * (0.5 * lambda * lambda),
                vector: [Complex::ZERO; 3],
            });
        }
        modes.push(Harmonic {
            wave: [Jet::constant(0.); 3],
            scalar: Complex::new(norm * lambda * lambda, 0.),
            vector: [Complex::ZERO; 3],
        });
        modes.retain(|m| m.scalar != Complex::ZERO || m.vector.iter().any(|z| *z != Complex::ZERO));
        let mut result = FieldGeometry {
            gradient: vec![Vec3::zero(); natoms],
            strain: [[0.; 3]; 3],
            energy: 0.,
        };
        if modes.is_empty() {
            return Ok(result);
        }
        for a in &basis.shells {
            for b in &basis.shells {
                let ca = cart_components(a.l);
                let cb = cart_components(b.l);
                for index in geometry.nearby_translation_indices(b.center - a.center, cutoff)? {
                    let shift = geometry.lattice().translation(index);
                    let distance = (b.center + shift - a.center).norm();
                    if (distance - cutoff).abs() <= 32. * f64::EPSILON * cutoff.max(1.) {
                        return Err(DftError::InvalidInput(
                            "Fourier-field geometry derivative is undefined at a shell image cutoff".into(),
                        ));
                    }
                    // With fixed reduced k the phase k.T stays constant. Fold
                    // that phase into the conjugate density for the shell pair.
                    let phases: Vec<_> = points.iter().map(|k| k.dot(shift).sin_cos()).collect();
                    let cart = |imaginary| {
                        cartesian_density_with(a, b, |i, j| {
                            let slot = (a.sph_off + i, b.sph_off + j);
                            phases
                                .iter()
                                .zip(density)
                                .map(|(&(s, c), p)| {
                                    if imaginary {
                                        s * p.re[slot] - c * p.im[slot]
                                    } else {
                                        c * p.re[slot] + s * p.im[slot]
                                    }
                                })
                                .sum::<f64>()
                                / points.len() as f64
                        })
                    };
                    let real = cart(false);
                    let imaginary = cart(true);
                    let centres = [seed(a.center, 0), seed(b.center + shift, 3)];
                    let mut pair = Jet::constant(0.);
                    for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                        for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                            for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                                let left = [ax as usize, ay as usize, az as usize];
                                for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                                    let right = [bx as usize, by as usize, bz as usize];
                                    let weight =
                                        Complex::new(real[(i, j)], imaginary[(i, j)]) * (wa * wb);
                                    for mode in &modes {
                                        let transform = |lb| {
                                            plane_wave_primitive_g(
                                                left, lb, ea, eb, centres[0], centres[1],
                                                mode.wave, cos_sin,
                                            )
                                        };
                                        let contract = |(re, im): (Jet, Jet), coefficient| {
                                            let z: Complex = weight * coefficient;
                                            re.scale(z.re) - im.scale(z.im)
                                        };
                                        if mode.scalar != Complex::ZERO {
                                            pair = pair + contract(transform(right), mode.scalar);
                                        }
                                        for (axis, coefficient) in
                                            mode.vector.into_iter().enumerate()
                                        {
                                            if coefficient != Complex::ZERO {
                                                let current = current_primitive(
                                                    right,
                                                    eb,
                                                    mode.wave[axis],
                                                    axis,
                                                    transform,
                                                );
                                                pair = pair + contract(current, coefficient);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    result.energy += pair.re;
                    for (atom, components) in [(a.atom, &pair.im[..3]), (b.atom, &pair.im[3..6])] {
                        result.gradient[atom] +=
                            Vec3::new(components[0], components[1], components[2]);
                    }
                    for (a, row) in result.strain.iter_mut().enumerate() {
                        for (b, element) in row.iter_mut().enumerate() {
                            *element += pair.im[6 + 3 * a + b];
                        }
                    }
                }
            }
        }
        if !result.energy.is_finite()
            || result.gradient.iter().any(|v| !v.norm().is_finite())
            || result.strain.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "nonfinite Fourier-field geometry derivative".into(),
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
