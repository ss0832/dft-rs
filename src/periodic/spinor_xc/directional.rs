// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! An independent density tangent of the globally projected potential path.
use super::{project, spin_descriptor, SpinorXcPointTaylor, MAX_SPIN_ANGULAR_ORDER};
use crate::dual::{DualN, Scalar};
use crate::grid::gauss_legendre;
use crate::response::series::{evaluate_series_many_result, SeriesScalar};
use crate::xc::{Family, ResolvedFunctional, XcTaylorInput};
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Value and independent density derivative of every normalized coefficient.
#[derive(Clone, Debug)]
pub struct SpinorXcDirectionalTaylor {
    pub value: SpinorXcPointTaylor,
    pub directional: SpinorXcPointTaylor,
}

/// Analytic derivative with respect to epsilon of the point potential at
/// descriptors(t)+epsilon*tangent(t). The spin-scaling tangent is also
/// differentiated. Order N requires collinear scalar-body derivatives N+3.
/// This uses a fixed angular rule; the caller must check angular convergence.
fn evaluate_active<const N: usize>(
    functional: &ResolvedFunctional,
    descriptors: &[Vec<f64>; 32],
    tangent: &[Vec<f64>; 32],
    angular_order: usize,
) -> Result<SpinorXcDirectionalTaylor> {
    let width = descriptors[0].len();
    if width == 0
        || !(4..=MAX_SPIN_ANGULAR_ORDER).contains(&angular_order)
        || angular_order % 2 != 0
        || descriptors
            .iter()
            .chain(tangent)
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "spinor directional paths require finite matching coefficients and even angular order 4..128".into(),
        ));
    }
    let magnetic = descriptors[1][0]
        .hypot(descriptors[2][0])
        .hypot(descriptors[3][0]);
    if descriptors[0][0] < magnetic {
        return Err(DftError::InvalidInput(
            "spinor directional base density must be positive semidefinite".into(),
        ));
    }
    let zero = || SpinorXcPointTaylor {
        energy: vec![0.0; width],
        gradient: std::array::from_fn(|_| vec![0.0; width]),
        angular_order,
    };
    let mut result = SpinorXcDirectionalTaylor {
        value: zero(),
        directional: zero(),
    };
    if descriptors[0][0] < crate::xc::RHO_CUTOFF {
        return Ok(result);
    }
    type Density = DualN<SeriesScalar, 1>;
    type Spin = DualN<Density, 1>;
    type Jet<const D: usize> = DualN<Spin, D>;
    let inputs: Vec<Vec<f64>> = descriptors.iter().chain(tangent).cloned().collect();
    let (nodes, weights) = gauss_legendre(angular_order);
    let azimuthal = 2 * angular_order;
    for (z, weight) in nodes.into_iter().zip(weights) {
        let transverse = (1.0 - z * z).sqrt();
        for j in 0..azimuthal {
            let angle = 2.0 * PI * (j as f64 + 0.5) / azimuthal as f64;
            let direction = [transverse * angle.cos(), transverse * angle.sin(), z];
            let q = evaluate_series_many_result(&inputs, |values| {
                let x = std::array::from_fn(|i| {
                    let density = Density {
                        re: values[i],
                        im: [values[32 + i]],
                    };
                    let inner = Spin {
                        re: density,
                        im: [if spin_descriptor(i) {
                            density
                        } else {
                            Density::from_f64(0.0)
                        }],
                    };
                    let mut x = if i < N {
                        Jet::<N>::variable(inner, i)
                    } else {
                        Jet::<N>::constant(inner)
                    };
                    if i < N {
                        x.im[i].im[0].re =
                            SeriesScalar::from_f64(if spin_descriptor(i) { 1.0 } else { 0.0 });
                    }

                    x
                });
                let q = project(&x, direction, functional.family, false)?;
                let mut out = Vec::with_capacity(4 * 7 * (N + 1));
                // q, q_s, q_v, q_sv, followed by their descriptor Jacobians.
                out.extend(q.iter().map(|a| a.re.re.re));
                out.extend(q.iter().map(|a| a.re.im[0].re));
                out.extend(q.iter().map(|a| a.re.re.im[0]));
                out.extend(q.iter().map(|a| a.re.im[0].im[0]));
                out.extend(q.iter().flat_map(|a| a.im.iter().map(|d| d.re.re)));
                out.extend(q.iter().flat_map(|a| a.im.iter().map(|d| d.im[0].re)));
                out.extend(q.iter().flat_map(|a| a.im.iter().map(|d| d.re.im[0])));
                out.extend(q.iter().flat_map(|a| a.im.iter().map(|d| d.im[0].im[0])));
                Ok(out)
            })?;
            let input = XcTaylorInput {
                variables: std::array::from_fn(|a| q[a].clone()),
                parameters: None,
            };
            let spin = std::array::from_fn(|a| q[7 + a].clone());
            let density = std::array::from_fn(|a| q[14 + a].clone());
            let mixed = std::array::from_fn(|a| q[21 + a].clone());
            let xc = functional
                .semilocal_taylor_bidirectional_potentials(&input, &spin, &density, &mixed)?;
            let w = weight / (2.0 * azimuthal as f64);
            for k in 0..width {
                result.value.energy[k] += w * (xc.value.energy[k] + xc.first.energy[k]);
                result.directional.energy[k] += w * (xc.second.energy[k] + xc.mixed.energy[k]);
                for i in 0..N {
                    let mut value = 0.0;
                    let mut derivative = 0.0;
                    for a in 0..7 {
                        let offset = 28 + N * a + i;
                        let qi = &q[offset];
                        let qsi = &q[offset + 7 * N];
                        let qvi = &q[offset + 14 * N];
                        let qsvi = &q[offset + 21 * N];
                        for m in 0..=k {
                            let n = k - m;
                            value += xc.value.potentials[a][m] * (qi[n] + qsi[n])
                                + xc.first.potentials[a][m] * qi[n];
                            derivative += xc.second.potentials[a][m] * (qi[n] + qsi[n])
                                + xc.value.potentials[a][m] * (qvi[n] + qsvi[n])
                                + xc.mixed.potentials[a][m] * qi[n]
                                + xc.first.potentials[a][m] * qvi[n];
                        }
                    }
                    result.value.gradient[i][k] += w * value;
                    result.directional.gradient[i][k] += w * derivative;
                }
            }
        }
    }
    Ok(result)
}

/// Analytic independent density derivative of every normalized potential
/// coefficient, including the density dependence of the spin-scaling tangent.
/// Path order N consumes energy derivatives through N+3. The caller checks
/// angular convergence. Only active descriptor seeds are instantiated: four
/// for LDA, sixteen for GGA, thirty-two for current-corrected meta-GGA.
pub fn spinor_semilocal_point_directional_taylor(
    functional: &ResolvedFunctional,
    descriptors: &[Vec<f64>; 32],
    tangent: &[Vec<f64>; 32],
    angular_order: usize,
) -> Result<SpinorXcDirectionalTaylor> {
    match functional.family {
        Family::Lda => evaluate_active::<4>(functional, descriptors, tangent, angular_order),
        Family::Gga => evaluate_active::<16>(functional, descriptors, tangent, angular_order),
        Family::MetaGga => evaluate_active::<32>(functional, descriptors, tangent, angular_order),
    }
}
