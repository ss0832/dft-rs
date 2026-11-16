// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Mixed analytic paths of the same globally projected spinor functional.
//! Density and spin-scaling tangents are independent nested duals; the
//! functional providers evaluate all mixed derivatives over Taylor scalars.
use super::{project, spin_descriptor, MAX_SPIN_ANGULAR_ORDER};
use crate::dual::{DualN, Scalar};
use crate::grid::gauss_legendre;
use crate::response::series::{evaluate_series_many_result, SeriesScalar};
use crate::xc::{Family, ResolvedFunctional, XcTaylorInput};
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Factorial-normalized energy and all 32 descriptor-potential paths.
#[derive(Clone, Debug)]
pub struct SpinorXcPointTaylor {
    pub energy: Vec<f64>,
    pub gradient: [Vec<f64>; 32],
    pub angular_order: usize,
}

/// Exact LDA limit of the continuous spin projection. For collinear
/// energy e(n,u), the integrand e(n,sz)+sz*e_u(n,sz) is the derivative
/// of z*e(n,sz). Its spherical average is the mean of the two endpoints
/// u=+-|m|. Use the angular path near zero spin and the density boundary,
/// where derivatives of the norm or the minority density are ill conditioned.
pub(crate) fn lda_endpoint_taylor(
    functional: &ResolvedFunctional,
    descriptors: &[Vec<f64>; 32],
    angular_order: usize,
) -> Result<Option<SpinorXcPointTaylor>> {
    if functional.family != Family::Lda {
        return Ok(None);
    }
    let rho = descriptors[0][0];
    let spin = descriptors[1][0]
        .hypot(descriptors[2][0])
        .hypot(descriptors[3][0]);
    let width = descriptors[0].len();
    if !rho.is_finite()
        || !spin.is_finite()
        || rho < crate::xc::RHO_CUTOFF
        || spin <= rho / 64.0
        || rho - spin <= (2.0 * crate::xc::RHO_CUTOFF).max(rho * f64::EPSILON.sqrt())
    {
        return Ok(None);
    }
    let projected = evaluate_series_many_result(&descriptors[..4], |x| {
        let scaled: [SeriesScalar; 3] = std::array::from_fn(|a| x[a + 1].scale(spin.recip()));
        let norm = (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2]).sqrt();
        let magnitude = norm.scale(spin);
        Ok([
            (x[0] + magnitude).scale(0.5),
            (x[0] - magnitude).scale(0.5),
            scaled[0] / norm,
            scaled[1] / norm,
            scaled[2] / norm,
        ]
        .to_vec())
    })?;
    let endpoint = |swap: bool| {
        functional.semilocal_taylor_potentials(&XcTaylorInput {
            variables: std::array::from_fn(|i| {
                if i < 2 {
                    projected[if swap { 1 - i } else { i }].clone()
                } else {
                    vec![0.0; width]
                }
            }),
            parameters: None,
        })
    };
    let plus = endpoint(false)?;
    let minus = endpoint(true)?;
    let spin_potential: Vec<_> = (0..width)
        .map(|k| {
            0.25 * (plus.potentials[0][k] - plus.potentials[1][k] + minus.potentials[1][k]
                - minus.potentials[0][k])
        })
        .collect();
    let result = SpinorXcPointTaylor {
        energy: (0..width)
            .map(|k| 0.5 * (plus.energy[k] + minus.energy[k]))
            .collect(),
        gradient: std::array::from_fn(|a| match a {
            0 => (0..width)
                .map(|k| {
                    0.25 * (plus.potentials[0][k]
                        + plus.potentials[1][k]
                        + minus.potentials[0][k]
                        + minus.potentials[1][k])
                })
                .collect(),
            1..=3 => (0..width)
                .map(|k| {
                    (0..=k)
                        .map(|m| spin_potential[m] * projected[a + 1][k - m])
                        .sum()
                })
                .collect(),
            _ => vec![0.0; width],
        }),
        angular_order,
    };
    Ok(result
        .energy
        .iter()
        .chain(result.gradient.iter().flatten())
        .all(|v| v.is_finite())
        .then_some(result))
}

/// Energy path without constructing descriptor potentials. It is the same
/// global projection, using first scalar XC partials for the spin tangent.
pub fn spinor_semilocal_energy_taylor(
    functional: &ResolvedFunctional,
    descriptors: &[Vec<f64>; 32],
    angular_order: usize,
) -> Result<Vec<f64>> {
    let width = descriptors[0].len();
    if width == 0
        || !(4..=MAX_SPIN_ANGULAR_ORDER).contains(&angular_order)
        || angular_order % 2 != 0
        || descriptors
            .iter()
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "spinor energy paths require matching finite coefficients and even angular order 4..128"
                .into(),
        ));
    }
    let mut energy = vec![0.0; width];
    if descriptors[0][0] < crate::xc::RHO_CUTOFF {
        return Ok(energy);
    }
    let magnetic = descriptors[1][0]
        .hypot(descriptors[2][0])
        .hypot(descriptors[3][0]);
    if descriptors[0][0] < magnetic {
        return Err(DftError::InvalidInput(
            "spinor energy base density must be positive semidefinite".into(),
        ));
    }
    let (nodes, weights) = gauss_legendre(angular_order);
    let azimuthal = 2 * angular_order;
    for (z, weight) in nodes.into_iter().zip(weights) {
        let transverse = (1.0 - z * z).sqrt();
        for j in 0..azimuthal {
            let angle = 2.0 * PI * (j as f64 + 0.5) / azimuthal as f64;
            let direction = [transverse * angle.cos(), transverse * angle.sin(), z];
            let q = evaluate_series_many_result(descriptors, |values| {
                let x = std::array::from_fn(|i| DualN {
                    re: values[i],
                    im: [if spin_descriptor(i) {
                        values[i]
                    } else {
                        SeriesScalar::from_f64(0.0)
                    }],
                });
                let q = project(&x, direction, functional.family, false)?;
                Ok(q.iter()
                    .map(|v| v.re)
                    .chain(q.iter().map(|v| v.im[0]))
                    .collect())
            })?;
            let xc = functional.semilocal_taylor_potentials(&XcTaylorInput {
                variables: std::array::from_fn(|a| q[a].clone()),
                parameters: None,
            })?;
            for k in 0..width {
                let tangent: f64 = (0..7)
                    .map(|a| {
                        (0..=k)
                            .map(|m| xc.potentials[a][m] * q[a + 7][k - m])
                            .sum::<f64>()
                    })
                    .sum();
                energy[k] += weight / (2.0 * azimuthal as f64) * (xc.energy[k] + tangent);
            }
        }
    }
    Ok(energy)
}

/// Taylor extension of `spinor_semilocal_point`, with the same angular rule,
/// spin-scaling derivative and current-corrected kinetic density. The order
/// is determined by the descriptor arrays. Angular convergence is independent
/// of Taylor convergence and must be checked by the integrating caller.
fn evaluate_active<const N: usize>(
    functional: &ResolvedFunctional,
    descriptors: &[Vec<f64>; 32],
    angular_order: usize,
) -> Result<SpinorXcPointTaylor> {
    let width = descriptors[0].len();
    if width == 0
        || !(4..=MAX_SPIN_ANGULAR_ORDER).contains(&angular_order)
        || angular_order % 2 != 0
        || descriptors
            .iter()
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput("spinor XC Taylor paths require matching finite coefficients and even angular order 4..128".into()));
    }
    let magnetic = descriptors[1][0]
        .hypot(descriptors[2][0])
        .hypot(descriptors[3][0]);
    if descriptors[0][0] < magnetic {
        return Err(DftError::InvalidInput(
            "spinor XC Taylor base density must be positive semidefinite".into(),
        ));
    }
    let mut result = SpinorXcPointTaylor {
        energy: vec![0.0; width],
        gradient: std::array::from_fn(|_| vec![0.0; width]),
        angular_order,
    };
    if descriptors[0][0] < crate::xc::RHO_CUTOFF {
        return Ok(result);
    }
    type Inner = DualN<SeriesScalar, 1>;
    type Jet<const D: usize> = DualN<Inner, D>;
    let (nodes, weights) = gauss_legendre(angular_order);
    let azimuthal = 2 * angular_order;
    for (z, weight) in nodes.into_iter().zip(weights) {
        let transverse = (1.0 - z * z).sqrt();
        for j in 0..azimuthal {
            let angle = 2.0 * PI * (j as f64 + 0.5) / azimuthal as f64;
            let direction = [transverse * angle.cos(), transverse * angle.sin(), z];
            let q = evaluate_series_many_result(descriptors, |values| {
                let x = std::array::from_fn(|i| {
                    let inner = Inner {
                        re: values[i],
                        im: [if spin_descriptor(i) {
                            values[i]
                        } else {
                            SeriesScalar::from_f64(0.0)
                        }],
                    };
                    let mut x = if i < N {
                        Jet::<N>::variable(inner, i)
                    } else {
                        Jet::<N>::constant(inner)
                    };
                    if i < N {
                        x.im[i].im[0] =
                            SeriesScalar::from_f64(if spin_descriptor(i) { 1.0 } else { 0.0 });
                    }

                    x
                });
                let q = project(&x, direction, functional.family, false)?;
                Ok(q.iter()
                    .map(|q| q.re.re)
                    .chain(q.iter().map(|q| q.re.im[0]))
                    .chain(q.iter().flat_map(|q| q.im.iter().map(|d| d.re)))
                    .chain(q.iter().flat_map(|q| q.im.iter().map(|d| d.im[0])))
                    .collect())
            })?;
            let input = XcTaylorInput {
                variables: std::array::from_fn(|a| q[a].clone()),
                parameters: None,
            };
            let spin: [Vec<f64>; 7] = std::array::from_fn(|a| q[a + 7].clone());
            let xc = functional.semilocal_taylor_directional_potentials(&input, &spin)?;
            let w = weight / (2.0 * azimuthal as f64);
            for k in 0..width {
                result.energy[k] += w * (xc.value.energy[k] + xc.directional.energy[k]);
                for i in 0..N {
                    let mut v = 0.0;
                    for a in 0..7 {
                        let first = &q[14 + N * a + i];
                        let mixed = &q[14 + 7 * N + N * a + i];
                        for m in 0..=k {
                            v += xc.value.potentials[a][m] * (first[k - m] + mixed[k - m])
                                + xc.directional.potentials[a][m] * first[k - m];
                        }
                    }
                    result.gradient[i][k] += w * v;
                }
            }
        }
    }
    Ok(result)
}

/// Evaluate the same projected potential using only the active descriptor seeds.
/// Inactive descriptor potentials retain exact zeros in the public 32-slot layout.
pub fn spinor_semilocal_point_taylor(
    functional: &ResolvedFunctional,
    descriptors: &[Vec<f64>; 32],
    angular_order: usize,
) -> Result<SpinorXcPointTaylor> {
    match functional.family {
        Family::Lda => evaluate_active::<4>(functional, descriptors, angular_order),
        Family::Gga => evaluate_active::<16>(functional, descriptors, angular_order),
        Family::MetaGga => evaluate_active::<32>(functional, descriptors, angular_order),
    }
}

#[cfg(test)]
mod lda_tests {
    use super::*;
    use crate::acceptance::{components, stencils::Stencil};
    use crate::xc::{resolve, FunctionalSpec};

    /// Independent exchange of the two eigenvalues of the local spin matrix.
    fn slater(density: [f64; 4]) -> (f64, [f64; 4]) {
        let spin = density[1].hypot(density[2]).hypot(density[3]);
        let [up, down] = [0.5 * (density[0] + spin), 0.5 * (density[0] - spin)];
        let prefactor = -0.75 * (6.0 / PI).cbrt();
        let energy = prefactor * (up.powf(4.0 / 3.0) + down.powf(4.0 / 3.0));
        let [vu, vd] = [up, down].map(|r| (4.0 / 3.0) * prefactor * r.cbrt());
        let gradient = std::array::from_fn(|a| {
            if a == 0 {
                0.5 * (vu + vd)
            } else {
                0.5 * (vu - vd) * density[a] / spin
            }
        });
        (energy, gradient)
    }

    #[test]
    fn lda_endpoint_taylor_fits_all_active_potential_derivatives() {
        let functional = resolve(&FunctionalSpec::Split {
            x: "slater".into(),
            c: "none".into(),
        })
        .unwrap();
        let density = [0.91, 0.12, -0.17, 0.23];
        let path = |a, b| {
            std::array::from_fn(|i| {
                vec![
                    density.get(i).copied().unwrap_or(0.0),
                    if i == a || i == b { 1.0 } else { 0.0 },
                    0.0,
                ]
            })
        };
        let axial = (0..4)
            .map(|i| {
                lda_endpoint_taylor(&functional, &path(i, i), 8)
                    .unwrap()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let centre = slater(density);
        assert!((axial[0].energy[0] - centre.0).abs() < 2e-14);
        let mut names = Vec::new();
        for a in 0..4 {
            names.push(format!("LDA endpoint gradient {a}"));
        }
        for a in 0..4 {
            for b in 0..4 {
                names.push(format!("LDA endpoint kernel {a},{b}"));
            }
        }
        let pairs: Vec<_> = (0..4).flat_map(|a| (a..4).map(move |b| (a, b))).collect();
        let second: Vec<_> = pairs
            .iter()
            .map(|&(a, b)| {
                if a == b {
                    std::array::from_fn::<_, 4, _>(|i| 2.0 * axial[a].gradient[i][2])
                } else {
                    let mixed = lda_endpoint_taylor(&functional, &path(a, b), 8)
                        .unwrap()
                        .unwrap();
                    std::array::from_fn(|i| {
                        mixed.gradient[i][2] - axial[a].gradient[i][2] - axial[b].gradient[i][2]
                    })
                }
            })
            .collect();
        for (a, b) in &pairs {
            for row in 0..4 {
                names.push(format!(
                    "LDA endpoint potential second derivative {row},{a},{b}"
                ));
            }
        }
        let sample = |a: usize, u: f64, b: usize, v: f64| {
            let mut x = density;
            x[a] += u;
            x[b] += v;
            slater(x)
        };
        let intervals = [[0.016, 0.012, 0.008, 0.006], [0.008, 0.006, 0.004, 0.003]];
        let mut failures = Vec::new();
        components::fit_components(
            &names,
            &intervals,
            |steps| {
                let mut errors = vec![[0.0; 4]; names.len()];
                for (step, &h) in steps.iter().enumerate() {
                    for b in 0..4 {
                        let plus = sample(b, h, b, 0.0);
                        let minus = sample(b, -h, b, 0.0);
                        errors[b][step] = (Stencil::Second.first(plus.0, minus.0, 0.0, 0.0, h)
                            - axial[b].gradient[b][0])
                            .abs();
                        for a in 0..4 {
                            errors[4 + 4 * a + b][step] = (Stencil::Second
                                .first(plus.1[a], minus.1[a], 0.0, 0.0, h)
                                - axial[b].gradient[a][1])
                                .abs();
                        }
                    }
                    for (pair, &(a, b)) in pairs.iter().enumerate() {
                        let pp = sample(a, h, b, if a == b { 0.0 } else { h });
                        let mm = sample(a, -h, b, if a == b { 0.0 } else { -h });
                        let pm = sample(a, h, b, -h);
                        let mp = sample(a, -h, b, h);
                        for row in 0..4 {
                            let measured = if a == b {
                                (pp.1[row] - 2.0 * centre.1[row] + mm.1[row]) / (h * h)
                            } else {
                                let plus = Stencil::Second.first(pp.1[row], pm.1[row], 0.0, 0.0, h);
                                let minus =
                                    Stencil::Second.first(mp.1[row], mm.1[row], 0.0, 0.0, h);
                                Stencil::Second.first(plus, minus, 0.0, 0.0, h)
                            };
                            errors[20 + 4 * pair + row][step] =
                                (measured - second[pair][row]).abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("LDA spin endpoint", &failures);
    }

    #[test]
    fn lda_endpoint_agrees_with_angular_svwn5_and_preserves_limiting_dispatch() {
        let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
        let mut path = std::array::from_fn(|_| vec![0.0; 3]);
        path[..4].clone_from_slice(&[
            vec![0.91, 0.021, -0.004],
            vec![0.12, -0.008, 0.003],
            vec![-0.17, 0.013, 0.002],
            vec![0.23, 0.009, -0.005],
        ]);
        let exact = lda_endpoint_taylor(&functional, &path, 16)
            .unwrap()
            .unwrap();
        let angular = spinor_semilocal_point_taylor(&functional, &path, 32).unwrap();
        for (a, b) in exact
            .energy
            .iter()
            .chain(exact.gradient.iter().flatten())
            .zip(
                angular
                    .energy
                    .iter()
                    .chain(angular.gradient.iter().flatten()),
            )
        {
            assert!((a - b).abs() < 2e-12, "{a} {b}");
        }
        for spin in [[0.0; 3], [0.91, 0.0, 0.0]] {
            for (axis, value) in spin.into_iter().enumerate() {
                path[axis + 1][0] = value;
            }
            assert!(lda_endpoint_taylor(&functional, &path, 16)
                .unwrap()
                .is_none());
        }
    }
}
