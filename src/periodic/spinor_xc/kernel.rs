// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static descriptor Hessian of the existing global spin projection. The XC
//! body is evaluated once through dense third order per angular batch. Fixed
//! duals differentiate only the inexpensive projection map, never XC bodies.
use super::{project, spin_descriptor, SpinorXcPoint, MAX_SPIN_ANGULAR_ORDER};
use crate::dual::DualN;
use crate::grid::gauss_legendre;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Unweighted point energy, potential and row-major active descriptor kernel.
pub struct SpinorXcPointKernel {
    pub energy: f64,
    pub gradient: [f64; 32],
    pub kernel: Vec<f64>,
    pub active: usize,
    pub angular_order: usize,
}

fn pair(n: usize, a: usize, b: usize) -> usize {
    a * n - a * (a + 1) / 2 + b
}
fn triple(n: usize, a: usize, b: usize, c: usize) -> usize {
    let mut slot = 0;
    for i in 0..n {
        for j in i..n {
            for k in j..n {
                if (i, j, k) == (a, b, c) {
                    return slot;
                }
                slot += 1;
            }
        }
    }
    unreachable!("sorted active XC triple")
}

fn third(out: &XcOut, point: usize, mut indices: [usize; 3]) -> Option<f64> {
    indices.sort_unstable();
    let [a, b, c] = indices;
    let block = match (a, b, c) {
        (_, _, 0..=1) => &out.v3rho3[a + b + c],
        (0..=1, 0..=1, 2..=4) => &out.v3rho2sigma[3 * (a + b) + c - 2],
        (0..=1, 2..=4, 2..=4) => &out.v3rhosigma2[6 * a + pair(3, b - 2, c - 2)],
        (2..=4, 2..=4, 2..=4) => &out.v3sigma3[triple(3, a - 2, b - 2, c - 2)],
        (0..=1, 0..=1, 5..=6) => &out.v3rho2tau[2 * (a + b) + c - 5],
        (0..=1, 2..=4, 5..=6) => &out.v3rhosigmatau[2 * (3 * a + b - 2) + c - 5],
        (0..=1, 5..=6, 5..=6) => &out.v3rhotau2[3 * a + b + c - 10],
        (2..=4, 2..=4, 5..=6) => &out.v3sigma2tau[2 * pair(3, a - 2, b - 2) + c - 5],
        (2..=4, 5..=6, 5..=6) => &out.v3sigmatau2[3 * (a - 2) + b + c - 10],
        (5..=6, 5..=6, 5..=6) => &out.v3tau3[a + b + c - 15],
        _ => return None,
    };
    block.get(point).copied().filter(|v| v.is_finite())
}

fn evaluate<const N: usize>(
    functional: &ResolvedFunctional,
    values: &[f64; 32],
    angular_order: usize,
    azimuthal: usize,
) -> Result<Option<SpinorXcPointKernel>> {
    type Spin = DualN<f64, 1>;
    let spin = std::array::from_fn(|i| Spin {
        re: values[i],
        im: [if spin_descriptor(i) { values[i] } else { 0.0 }],
    });
    let (nodes, weights) = gauss_legendre(angular_order);
    let mut directions = Vec::with_capacity(azimuthal * angular_order);
    let mut quadrature = Vec::with_capacity(directions.capacity());
    let mut projected = Vec::with_capacity(directions.capacity());
    for (z, weight) in nodes.into_iter().zip(weights) {
        let transverse = (1.0 - z * z).sqrt();
        for j in 0..azimuthal {
            let angle = 2.0 * PI * (j as f64 + 0.5) / azimuthal as f64;
            let direction = [transverse * angle.cos(), transverse * angle.sin(), z];
            let q = project(&spin, direction, functional.family, false)?;
            // Stay away from existing density cutoffs, sigma boundaries and
            // kinetic/current FHC clipping. The original Taylor path owns them.
            if q[..2].iter().any(|v| v.re <= crate::xc::RHO_CUTOFF)
                || (functional.family != Family::Lda && (q[2].re <= 0.0 || q[4].re <= 0.0))
                || (functional.family == Family::MetaGga
                    && (q[5].re <= q[2].re / (8.0 * q[0].re)
                        || q[6].re <= q[4].re / (8.0 * q[1].re)))
            {
                return Ok(None);
            }
            directions.push(direction);
            quadrature.push(weight / (2.0 * azimuthal as f64));
            projected.push(q);
        }
    }
    let data: [Vec<f64>; 7] = std::array::from_fn(|a| projected.iter().map(|q| q[a].re).collect());
    let vars = GridVars {
        rho: [&data[0], &data[1]],
        sigma: (functional.family != Family::Lda).then_some([&data[2], &data[3], &data[4]]),
        tau: (functional.family == Family::MetaGga).then_some([&data[5], &data[6]]),
    };
    let mut out = XcOut::default();
    match functional.eval(&vars, 3, &mut out) {
        Ok(()) => {}
        Err(DftError::DerivOrderUnsupported { .. } | DftError::Unsupported { .. }) => {
            return Ok(None)
        }
        Err(error) => return Err(error),
    }
    let mut result = SpinorXcPointKernel {
        energy: 0.0,
        gradient: [0.0; 32],
        kernel: vec![0.0; N * N],
        active: N,
        angular_order,
    };
    let dimension = match functional.family {
        Family::Lda => 2,
        Family::Gga => 5,
        Family::MetaGga => 7,
    };
    for (g, (&direction, &weight)) in directions.iter().zip(&quadrature).enumerate() {
        let Ok((first, second)) = out.potential_and_kernel(g, functional.family) else {
            return Ok(None);
        };
        let Some(&energy) = out.e.get(g).filter(|v| v.is_finite()) else {
            return Ok(None);
        };
        let mut v = first;
        let mut w = second;
        for a in 0..dimension {
            for b in 0..dimension {
                v[a] += second[a][b] * projected[g][b].im[0];
                for c in 0..dimension {
                    let Some(value) = third(&out, g, [a, b, c]) else {
                        return Ok(None);
                    };
                    w[a][b] += value * projected[g][c].im[0];
                }
            }
        }
        result.energy += weight
            * (energy
                + (0..dimension)
                    .map(|a| first[a] * projected[g][a].im[0])
                    .sum::<f64>());
        // Only the cheap geometric projection is differentiated per column.
        // Keeping the outer seed width one bounds stack storage even for N=32.
        for column in 0..N {
            let x = std::array::from_fn(|i| {
                let mut middle = if i < N {
                    DualN::<Spin, N>::variable(spin[i], i)
                } else {
                    DualN::<Spin, N>::constant(spin[i])
                };
                if i < N {
                    middle.im[i].im[0] = f64::from(spin_descriptor(i));
                }
                let mut outer = DualN::<DualN<Spin, N>, 1>::constant(middle);
                if i == column {
                    outer.im[0] = DualN::constant(Spin {
                        re: 1.0,
                        im: [f64::from(spin_descriptor(i))],
                    });
                }
                outer
            });
            let q = project(&x, direction, functional.family, false)?;
            for row in 0..N {
                let mut value = 0.0;
                for a in 0..dimension {
                    if column == 0 {
                        result.gradient[row] +=
                            weight * (v[a] * q[a].re.im[row].re + first[a] * q[a].re.im[row].im[0]);
                    }
                    value += v[a] * q[a].im[0].im[row].re + first[a] * q[a].im[0].im[row].im[0];
                    for b in 0..dimension {
                        value += w[a][b] * q[a].re.im[row].re * q[b].im[0].re.re
                            + second[a][b]
                                * (q[a].re.im[row].re * q[b].im[0].re.im[0]
                                    + q[a].im[0].re.re * q[b].re.im[row].im[0]);
                    }
                }
                result.kernel[row * N + column] += weight * value;
            }
        }
    }
    if !result.energy.is_finite()
        || result
            .gradient
            .iter()
            .chain(&result.kernel)
            .any(|v| !v.is_finite())
    {
        return Ok(None);
    }
    Ok(Some(result))
}

fn azimuthal_count(point: &SpinorXcPoint, family: Family, angular_order: usize) -> usize {
    let zero = |values: &[f64]| values.iter().all(|&v| v == 0.0);
    let collinear = zero(&point.density[1..3])
        && (family == Family::Lda || point.gradient[1..3].iter().all(|v| zero(v)))
        && (family != Family::MetaGga
            || (zero(&point.tau[1..3]) && point.current[1..3].iter().all(|v| zero(v))));
    if collinear {
        // At an exactly z-collinear base, both the projected descriptors
        // and their spin-scaling direction are independent of azimuth.
        // Each of the two descriptor derivatives introduces at most one
        // transverse direction factor. Four equally spaced samples
        // integrate the resulting degree-two Fourier polynomial exactly.
        4
    } else {
        2 * angular_order
    }
}

/// Try the dense-third chain rule for a smooth interior point. `None` requests
/// the existing Taylor fallback; invalid caller inputs remain explicit errors.
/// Angular convergence is still the responsibility of the existing caller.
pub fn spinor_semilocal_point_kernel(
    functional: &ResolvedFunctional,
    point: &SpinorXcPoint,
    angular_order: usize,
) -> Result<Option<SpinorXcPointKernel>> {
    let values = point.descriptors();
    if !(4..=MAX_SPIN_ANGULAR_ORDER).contains(&angular_order)
        || angular_order % 2 != 0
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "spinor point kernel requires finite descriptors and even angular order 4..128".into(),
        ));
    }
    if point.density[0]
        < point.density[1]
            .hypot(point.density[2])
            .hypot(point.density[3])
    {
        return Err(DftError::InvalidInput(
            "spinor point kernel requires a positive-semidefinite density".into(),
        ));
    }
    if values[0] < crate::xc::RHO_CUTOFF
        || functional
            .parts
            .iter()
            .any(|(_, part)| part.max_analytic_order() < 3)
    {
        return Ok(None);
    }
    let azimuthal = azimuthal_count(point, functional.family, angular_order);
    match functional.family {
        Family::Lda => evaluate::<4>(functional, &values, angular_order, azimuthal),
        Family::Gga => evaluate::<16>(functional, &values, angular_order, azimuthal),
        Family::MetaGga => evaluate::<32>(functional, &values, angular_order, azimuthal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{resolve, FunctionalSpec};

    fn collinear_point() -> SpinorXcPoint {
        SpinorXcPoint {
            density: [0.91, 0.0, 0.0, 0.23],
            gradient: [
                [0.19, -0.13, 0.17],
                [0.0; 3],
                [0.0; 3],
                [0.057, 0.019, -0.043],
            ],
            tau: [0.83, 0.0, 0.0, 0.093],
            current: [
                [0.071, -0.053, 0.043],
                [0.0; 3],
                [0.0; 3],
                [0.041, 0.029, -0.026],
            ],
        }
    }

    #[test]
    fn reduced_azimuth_matches_every_dense_kernel_entry() {
        const ORDER: usize = 12;
        let point = collinear_point();
        let x = point.descriptors();
        for name in ["svwn5", "pbe", "m06-l", "m06", "m06-2x", "m06-sx"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let full = match functional.family {
                Family::Lda => evaluate::<4>(&functional, &x, ORDER, 2 * ORDER),
                Family::Gga => evaluate::<16>(&functional, &x, ORDER, 2 * ORDER),
                Family::MetaGga => evaluate::<32>(&functional, &x, ORDER, 2 * ORDER),
            }
            .unwrap()
            .expect("interior point");
            let reduced = spinor_semilocal_point_kernel(&functional, &point, ORDER)
                .unwrap()
                .expect("interior point");
            assert_eq!(azimuthal_count(&point, functional.family, ORDER), 4);
            assert_eq!(reduced.angular_order, full.angular_order);
            assert_eq!(reduced.active, full.active);
            let mut max_error = 0.0_f64;
            for (expected, actual) in std::iter::once(full.energy)
                .chain(full.gradient)
                .chain(full.kernel)
                .zip(
                    std::iter::once(reduced.energy)
                        .chain(reduced.gradient)
                        .chain(reduced.kernel),
                )
            {
                let error = (expected - actual).abs();
                assert!(
                    error < 5e-13 * expected.abs().max(1.0),
                    "{name}: {expected} vs {actual}"
                );
                max_error = max_error.max(error);
            }
            println!(
                "{name} angular_order={ORDER} maximum absolute descriptor defect={max_error:.16e}"
            );
        }

        // A nonzero transverse descriptor of any active family forces the
        // general rule even when its magnitude is below a typical threshold.
        for i in [1, 2, 7, 12, 17, 18, 24, 27] {
            let mut x = x;
            x[i] = 1e-30;
            let moved = SpinorXcPoint::from_descriptors(x);
            assert_eq!(azimuthal_count(&moved, Family::MetaGga, ORDER), 2 * ORDER);
        }
    }

    #[test]
    #[ignore = "Report five alternating dense versus reduced local timings"]
    fn time_collinear_m06_descriptor_kernel() {
        use std::hint::black_box;
        use std::time::Instant;

        const ORDER: usize = 20;
        let functional = resolve(&FunctionalSpec::Combined("m06".into())).unwrap();
        let point = collinear_point();
        let x = point.descriptors();
        for sample in 0..5_usize {
            let sequence = if sample.is_multiple_of(2) {
                [2 * ORDER, 4]
            } else {
                [4, 2 * ORDER]
            };
            for azimuthal in sequence {
                let start = Instant::now();
                let result = evaluate::<32>(&functional, &x, ORDER, azimuthal)
                    .unwrap()
                    .expect("interior point");
                let elapsed = start.elapsed().as_secs_f64();
                black_box(result);
                println!("sample={sample} azimuthal={azimuthal} wall_seconds={elapsed:.9}");
            }
        }
    }
}
