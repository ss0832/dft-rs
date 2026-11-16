// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spheroidal inclusions and effective thermal conductivity.
//!
//! - Geometric factors `P` and `Q` of randomly oriented spheroidal inclusions
//!   (aspect ratio `a`: oblate `a < 1`, prolate `a > 1`) from the
//!   strain-concentration tensor `T = [I + S C_m^-1 (C_i - C_m)]^-1` of
//!   Eshelby's solution (J. D. Eshelby, Proc. R. Soc. A 241, 376 (1957);
//!   T. T. Wu, Int. J. Solids Struct. 2, 1 (1966)), `P = T_iijj / 3`,
//!   `Q = (T_ijij - T_iijj/3) / 5`, in Berryman's closed form
//!   (J. G. Berryman, J. Acoust. Soc. Am. 68, 1809 (1980)). The unit tests
//!   check the closed form against an independent numerical Eshelby tensor
//!   (Mura's depolarization integrals, T. Mura, Micromechanics of Defects
//!   in Solids (1987), eq. 11.16) and against the sphere, penny-crack and
//!   needle limits.
//! - Kuster-Toksoz dilute estimate (G. T. Kuster, M. N. Toksoz, Geophysics
//!   39, 587 (1974)), the differential effective medium (DEM) for spheroidal
//!   inclusions and Berryman's self-consistent estimate.
//! - Isotropic multiphase thermal conductivity: Hashin-Shtrikman bounds
//!   (Z. Hashin, S. Shtrikman, J. Appl. Phys. 33, 3125 (1962)) and the
//!   arithmetic, harmonic and geometric means.

use crate::error::{DftError, Result};

/// Moduli of an isotropic phase (GPa).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moduli {
    pub bulk: f64,
    pub shear: f64,
}

/// `theta` and `f` of a spheroid of aspect ratio `a` (Berryman 1980).
fn spheroid_functions(a: f64) -> (f64, f64) {
    let theta = if a < 1.0 {
        let e = 1.0 - a * a;
        a / e.powf(1.5) * (a.acos() - a * e.sqrt())
    } else {
        let e = a * a - 1.0;
        a / e.powf(1.5) * (a * e.sqrt() - a.acosh())
    };
    let f = a * a / (1.0 - a * a) * (3.0 * theta - 2.0);
    (theta, f)
}

/// Berryman's `P` and `Q` of an inclusion `inclusion` with aspect ratio
/// `aspect` in the matrix `matrix`, averaged over orientations.
pub fn spheroid_pq(matrix: Moduli, inclusion: Moduli, aspect: f64) -> Result<(f64, f64)> {
    if !(aspect > 0.0 && aspect.is_finite() && matrix.bulk > 0.0 && matrix.shear > 0.0)
        || inclusion.bulk < 0.0
        || inclusion.shear < 0.0
    {
        return Err(DftError::InvalidInput(
            "spheroidal inclusions need a positive aspect ratio, positive matrix \
            moduli and non-negative inclusion moduli"
                .into(),
        ));
    }
    let (km, mm, ki, mi) = (matrix.bulk, matrix.shear, inclusion.bulk, inclusion.shear);
    if (aspect - 1.0).abs() < 1e-5 {
        let zeta = mm / 6.0 * (9.0 * km + 8.0 * mm) / (km + 2.0 * mm);
        return Ok((
            (km + 4.0 * mm / 3.0) / (ki + 4.0 * mm / 3.0),
            (mm + zeta) / (mi + zeta),
        ));
    }
    let (theta, g) = spheroid_functions(aspect);
    let a = mi / mm - 1.0;
    let b = (ki / km - mi / mm) / 3.0;
    let r = 3.0 * mm / (3.0 * km + 4.0 * mm);
    let f1 = 1.0 + a * (1.5 * (g + theta) - r * (1.5 * g + 2.5 * theta - 4.0 / 3.0));
    let f2 = 1.0
        + a * (1.0 + 1.5 * (g + theta) - 0.5 * r * (3.0 * g + 5.0 * theta))
        + b * (3.0 - 4.0 * r)
        + 0.5
            * a
            * (a + 3.0 * b)
            * (3.0 - 4.0 * r)
            * (g + theta - r * (g - theta + 2.0 * theta * theta));
    let f3 = 1.0 + a * (1.0 - (g + 1.5 * theta) + r * (g + theta));
    let f4 = 1.0 + 0.25 * a * (g + 3.0 * theta - r * (g - theta));
    let f5 = a * (-g + r * (g + theta - 4.0 / 3.0)) + b * theta * (3.0 - 4.0 * r);
    let f6 = 1.0 + a * (1.0 + g - r * (g + theta)) + b * (1.0 - theta) * (3.0 - 4.0 * r);
    let f7 = 2.0
        + 0.25 * a * (3.0 * g + 9.0 * theta - r * (3.0 * g + 5.0 * theta))
        + b * theta * (3.0 - 4.0 * r);
    let f8 = a * (1.0 - 2.0 * r + 0.5 * g * (r - 1.0) + 0.5 * theta * (5.0 * r - 3.0))
        + b * (1.0 - theta) * (3.0 - 4.0 * r);
    let f9 = a * ((r - 1.0) * g - r * theta) + b * theta * (3.0 - 4.0 * r);
    let p = f1 / f2;
    let q = (2.0 / f3 + 1.0 / f4 + (f4 * f5 + f6 * f7 - f8 * f9) / (f2 * f4)) / 5.0;
    if !(p.is_finite() && q.is_finite()) {
        return Err(DftError::InvalidInput(format!(
            "the inclusion factors are singular for aspect ratio {aspect}"
        )));
    }
    Ok((p, q))
}

/// One inclusion family: volume fraction, moduli and aspect ratio.
#[derive(Clone, Copy, Debug)]
pub struct Inclusion {
    pub fraction: f64,
    pub moduli: Moduli,
    pub aspect: f64,
}

/// Kuster-Toksoz effective moduli of dilute inclusions in `host`.
pub fn kuster_toksoz(host: Moduli, inclusions: &[Inclusion]) -> Result<Moduli> {
    let (km, mm) = (host.bulk, host.shear);
    let zeta = mm / 6.0 * (9.0 * km + 8.0 * mm) / (km + 2.0 * mm);
    let (mut sk, mut sm) = (0.0, 0.0);
    for inc in inclusions {
        let (p, q) = spheroid_pq(host, inc.moduli, inc.aspect)?;
        sk += inc.fraction * (inc.moduli.bulk - km) * p;
        sm += inc.fraction * (inc.moduli.shear - mm) * q;
    }
    // (K - Km)(Km + 4/3 Gm)/(K + 4/3 Gm) = sk, and the shear analogue.
    let c = 4.0 * mm / 3.0;
    let bulk = (km * (km + c) + c * sk) / (km + c - sk);
    let shear = (mm * (mm + zeta) + zeta * sm) / (mm + zeta - sm);
    if !(bulk >= 0.0 && shear >= 0.0) {
        return Err(DftError::InvalidInput(
            "the Kuster-Toksoz estimate is outside its dilute range (negative modulus)".into(),
        ));
    }
    Ok(Moduli { bulk, shear })
}

/// Differential effective medium: inclusions of `inclusion` moduli and
/// `aspect` added to `host` up to the volume fraction `fraction`, by RK4
/// with `steps` equal steps of
/// `dK/dy = (K_i - K) P(K, G)/(1 - y)`, `dG/dy = (G_i - G) Q(K, G)/(1 - y)`.
pub fn dem_fixed_steps(
    host: Moduli,
    inclusion: Moduli,
    aspect: f64,
    fraction: f64,
    steps: usize,
) -> Result<Moduli> {
    if !(0.0..1.0).contains(&fraction) || steps == 0 {
        return Err(DftError::InvalidInput(
            "DEM needs a fraction in [0, 1) and at least one step".into(),
        ));
    }
    let rate = |y: f64, m: Moduli| -> Result<(f64, f64)> {
        let (p, q) = spheroid_pq(m, inclusion, aspect)?;
        Ok((
            (inclusion.bulk - m.bulk) * p / (1.0 - y),
            (inclusion.shear - m.shear) * q / (1.0 - y),
        ))
    };
    let h = fraction / steps as f64;
    let mut m = host;
    for n in 0..steps {
        let y = n as f64 * h;
        let shift = |m: Moduli, d: (f64, f64), s: f64| Moduli {
            bulk: m.bulk + s * d.0,
            shear: m.shear + s * d.1,
        };
        let k1 = rate(y, m)?;
        let k2 = rate(y + 0.5 * h, shift(m, k1, 0.5 * h))?;
        let k3 = rate(y + 0.5 * h, shift(m, k2, 0.5 * h))?;
        let k4 = rate(y + h, shift(m, k3, h))?;
        m = Moduli {
            bulk: m.bulk + h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
            shear: m.shear + h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
        };
        if !(m.bulk > 0.0 && m.shear > 0.0) {
            return Err(DftError::InvalidInput(
                "the DEM integration reached a non-positive modulus".into(),
            ));
        }
    }
    Ok(m)
}

/// DEM to a relative accuracy of 1e-10 by step doubling; returns the moduli
/// and the number of RK4 steps used.
pub fn dem(host: Moduli, inclusion: Moduli, aspect: f64, fraction: f64) -> Result<(Moduli, usize)> {
    let mut steps = 16;
    let mut previous = dem_fixed_steps(host, inclusion, aspect, fraction, steps)?;
    while steps < 1 << 16 {
        steps *= 2;
        let next = dem_fixed_steps(host, inclusion, aspect, fraction, steps)?;
        let change = ((next.bulk - previous.bulk).abs() / next.bulk.abs().max(1e-30))
            .max((next.shear - previous.shear).abs() / next.shear.abs().max(1e-30));
        previous = next;
        if change < 1e-10 {
            return Ok((previous, steps));
        }
    }
    Err(DftError::InvalidInput(
        "the DEM integration did not reach 1e-10 relative accuracy".into(),
    ))
}

/// Berryman's self-consistent estimate for phases with aspect ratios
/// (`sum x_i (K_i - K) P_i = 0`, `sum x_i (G_i - G) Q_i = 0`), by damped
/// fixed-point iteration; `None` when it does not converge (for example
/// above the percolation threshold of the stiff phase).
pub fn self_consistent(phases: &[Inclusion]) -> Result<Option<Moduli>> {
    let total: f64 = phases.iter().map(|p| p.fraction).sum();
    if phases.is_empty() || (total - 1.0).abs() > 1e-9 || phases.iter().any(|p| p.fraction < 0.0) {
        return Err(DftError::InvalidInput(
            "self-consistent phases need non-negative fractions summing to one".into(),
        ));
    }
    let mut m = Moduli {
        bulk: phases.iter().map(|p| p.fraction * p.moduli.bulk).sum(),
        shear: phases.iter().map(|p| p.fraction * p.moduli.shear).sum(),
    };
    for _ in 0..2000 {
        if !(m.bulk > 0.0 && m.shear > 0.0) {
            return Ok(None);
        }
        let (mut nk, mut dk, mut ng, mut dg) = (0.0, 0.0, 0.0, 0.0);
        for p in phases {
            let (pp, qq) = spheroid_pq(m, p.moduli, p.aspect)?;
            nk += p.fraction * p.moduli.bulk * pp;
            dk += p.fraction * pp;
            ng += p.fraction * p.moduli.shear * qq;
            dg += p.fraction * qq;
        }
        let next = Moduli {
            bulk: nk / dk,
            shear: ng / dg,
        };
        let change = (next.bulk - m.bulk).abs() / m.bulk + (next.shear - m.shear).abs() / m.shear;
        m = Moduli {
            bulk: 0.5 * (m.bulk + next.bulk),
            shear: 0.5 * (m.shear + next.shear),
        };
        if change < 1e-13 {
            return Ok(Some(m));
        }
    }
    Ok(None)
}

/// Bounds and means of the isotropic thermal conductivity (W/(m K)).
#[derive(Clone, Debug)]
pub struct ConductivityBounds {
    pub arithmetic: f64,
    pub harmonic: f64,
    pub geometric: f64,
    pub hashin_shtrikman_lower: f64,
    pub hashin_shtrikman_upper: f64,
}

/// `[sum x_i / (l_i + 2 z)]^-1 - 2 z`.
fn hs_conductivity(phases: &[(f64, f64)], z: f64) -> f64 {
    1.0 / phases.iter().map(|(x, l)| x / (l + 2.0 * z)).sum::<f64>() - 2.0 * z
}

/// Effective conductivity bounds of phases `(fraction, conductivity)`.
pub fn conductivity_bounds(phases: &[(f64, f64)]) -> Result<ConductivityBounds> {
    let total: f64 = phases.iter().map(|p| p.0).sum();
    if phases.is_empty()
        || (total - 1.0).abs() > 1e-9
        || phases.iter().any(|(x, l)| *x < 0.0 || !(*l > 0.0))
    {
        return Err(DftError::InvalidInput("conductivity phases need non-negative fractions summing to one and positive conductivities".into()));
    }
    let lmax = phases.iter().fold(0.0_f64, |m, p| m.max(p.1));
    let lmin = phases.iter().fold(f64::INFINITY, |m, p| m.min(p.1));
    Ok(ConductivityBounds {
        arithmetic: phases.iter().map(|(x, l)| x * l).sum(),
        harmonic: 1.0 / phases.iter().map(|(x, l)| x / l).sum::<f64>(),
        geometric: phases.iter().map(|(x, l)| x * l.ln()).sum::<f64>().exp(),
        hashin_shtrikman_lower: hs_conductivity(phases, lmin),
        hashin_shtrikman_upper: hs_conductivity(phases, lmax),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent oracle: the Eshelby tensor of a spheroid with semi-axes
    /// (1, 1, a) from Mura's depolarization integrals evaluated by
    /// Gauss-Legendre quadrature on `s = exp(y)`, and `T = [I + S C_m^-1
    /// (C_i - C_m)]^-1` in the 9x9 representation of fourth-rank tensors.
    fn oracle_pq(matrix: Moduli, inclusion: Moduli, aspect: f64) -> (f64, f64) {
        let axes = [1.0, 1.0, aspect];
        let (nodes, weights) = crate::grid::gauss_legendre(64);
        let integral = |f: &dyn Fn(f64) -> f64| -> f64 {
            let (lo, hi, panels) = (-60.0_f64, 60.0_f64, 120);
            let width = (hi - lo) / panels as f64;
            let mut sum = 0.0;
            for p in 0..panels {
                let mid = lo + (p as f64 + 0.5) * width;
                for (x, w) in nodes.iter().zip(&weights) {
                    let y = mid + 0.5 * width * x;
                    let s = y.exp();
                    sum += 0.5 * width * w * f(s) * s;
                }
            }
            sum
        };
        let delta = |s: f64| {
            ((axes[0] * axes[0] + s) * (axes[1] * axes[1] + s) * (axes[2] * axes[2] + s)).sqrt()
        };
        let pre = 2.0 * std::f64::consts::PI * axes[0] * axes[1] * axes[2];
        let ii: Vec<f64> = (0..3)
            .map(|i| pre * integral(&|s| 1.0 / ((axes[i] * axes[i] + s) * delta(s))))
            .collect();
        let iij: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        pre * integral(&|s| {
                            1.0 / ((axes[i] * axes[i] + s) * (axes[j] * axes[j] + s) * delta(s))
                        })
                    })
                    .collect()
            })
            .collect();
        let (km, mm) = (matrix.bulk, matrix.shear);
        let nu = (3.0 * km - 2.0 * mm) / (2.0 * (3.0 * km + mm));
        let pi = std::f64::consts::PI;
        let mut s = [[0.0; 9]; 9];
        let idx = |i: usize, j: usize| 3 * i + j;
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    s[idx(i, i)][idx(i, i)] =
                        3.0 / (8.0 * pi * (1.0 - nu)) * axes[i] * axes[i] * iij[i][i]
                            + (1.0 - 2.0 * nu) / (8.0 * pi * (1.0 - nu)) * ii[i];
                } else {
                    s[idx(i, i)][idx(j, j)] =
                        1.0 / (8.0 * pi * (1.0 - nu)) * axes[j] * axes[j] * iij[i][j]
                            - (1.0 - 2.0 * nu) / (8.0 * pi * (1.0 - nu)) * ii[i];
                    let shear = (axes[i] * axes[i] + axes[j] * axes[j]) / (16.0 * pi * (1.0 - nu))
                        * iij[i][j]
                        + (1.0 - 2.0 * nu) / (16.0 * pi * (1.0 - nu)) * (ii[i] + ii[j]);
                    for (a, b) in [(i, j), (j, i)] {
                        for (c, d) in [(i, j), (j, i)] {
                            s[idx(a, b)][idx(c, d)] = shear;
                        }
                    }
                }
            }
        }
        // Isotropic stiffness and compliance in the 9x9 representation.
        let iso = |k: f64, g: f64| -> [[f64; 9]; 9] {
            let mut c = [[0.0; 9]; 9];
            for i in 0..3 {
                for j in 0..3 {
                    for a in 0..3 {
                        for b in 0..3 {
                            let d = |p: usize, q: usize| if p == q { 1.0 } else { 0.0 };
                            c[idx(i, j)][idx(a, b)] = (k - 2.0 * g / 3.0) * d(i, j) * d(a, b)
                                + g * (d(i, a) * d(j, b) + d(i, b) * d(j, a));
                        }
                    }
                }
            }
            c
        };
        let mul = |a: &[[f64; 9]; 9], b: &[[f64; 9]; 9]| -> [[f64; 9]; 9] {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..9).map(|k| a[i][k] * b[k][j]).sum())
            })
        };
        // Compliance of the matrix on symmetric tensors: s = I_sym/(2G) - (3K-2G)/(18 K G) 1 1.
        let mut compliance = [[0.0; 9]; 9];
        for i in 0..3 {
            for j in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        let d = |p: usize, q: usize| if p == q { 1.0 } else { 0.0 };
                        compliance[idx(i, j)][idx(a, b)] = (d(i, a) * d(j, b) + d(i, b) * d(j, a))
                            / (4.0 * mm)
                            - (3.0 * km - 2.0 * mm) / (18.0 * km * mm) * d(i, j) * d(a, b);
                    }
                }
            }
        }
        let (cm, ci) = (iso(km, mm), iso(inclusion.bulk, inclusion.shear));
        let diff: [[f64; 9]; 9] =
            std::array::from_fn(|i| std::array::from_fn(|j| ci[i][j] - cm[i][j]));
        let x = mul(&s, &mul(&compliance, &diff));
        let mut a = crate::linalg::Matrix::zeros(9, 9);
        for i in 0..9 {
            for j in 0..9 {
                a[(i, j)] = x[i][j] + if i == j { 1.0 } else { 0.0 };
            }
        }
        let t =
            crate::linalg::solve_linear_matrix(&a, &crate::linalg::Matrix::identity(9)).unwrap();
        let iijj: f64 = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| t[(idx(i, i), idx(j, j))])
            .sum();
        let ijij: f64 = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| t[(idx(i, j), idx(i, j))])
            .sum();
        // The 9x9 representation is the identity on the three antisymmetric
        // components, which adds 3 to the full trace; strains are symmetric.
        let ijij_symmetric = ijij - 3.0;
        (iijj / 3.0, (ijij_symmetric - iijj / 3.0) / 5.0)
    }

    const QUARTZ: Moduli = Moduli {
        bulk: 37.0,
        shear: 44.0,
    };
    const CALCITE: Moduli = Moduli {
        bulk: 70.8,
        shear: 30.3,
    };
    const WATER: Moduli = Moduli {
        bulk: 2.25,
        shear: 0.0,
    };
    const DRY: Moduli = Moduli {
        bulk: 0.0,
        shear: 0.0,
    };

    #[test]
    fn closed_form_factors_match_the_numerical_eshelby_tensor() {
        let mut worst = 0.0_f64;
        for (matrix, inclusion) in [
            (QUARTZ, DRY),
            (QUARTZ, WATER),
            (CALCITE, QUARTZ),
            (QUARTZ, CALCITE),
        ] {
            for aspect in [0.01, 0.1, 0.5, 0.9, 1.0, 1.5, 4.0, 20.0] {
                let (p, q) = spheroid_pq(matrix, inclusion, aspect).unwrap();
                let (po, qo) = oracle_pq(matrix, inclusion, aspect);
                let e = ((p - po) / po).abs().max(((q - qo) / qo).abs());
                worst = worst.max(e);
                println!("{matrix:?} {inclusion:?} a {aspect}: P {p:.12} oracle {po:.12}, Q {q:.12} oracle {qo:.12}, relative {e:e}");
                assert!(e < 1e-8, "a {aspect}");
            }
        }
        println!("largest relative difference to the numerical Eshelby tensor {worst:e}");
    }

    #[test]
    fn factors_reach_the_sphere_penny_crack_and_needle_limits() {
        let (km, mm) = (QUARTZ.bulk, QUARTZ.shear);
        for inc in [DRY, WATER] {
            let (ki, mi) = (inc.bulk, inc.shear);
            let zeta = mm / 6.0 * (9.0 * km + 8.0 * mm) / (km + 2.0 * mm);
            let sphere = (
                (km + 4.0 * mm / 3.0) / (ki + 4.0 * mm / 3.0),
                (mm + zeta) / (mi + zeta),
            );
            let near = spheroid_pq(QUARTZ, inc, 1.0 + 2e-5).unwrap();
            println!("sphere {sphere:?}, a = 1 + 2e-5 {near:?}");
            assert!(
                (near.0 - sphere.0).abs() < 1e-4 * sphere.0
                    && (near.1 - sphere.1).abs() < 1e-4 * sphere.1
            );
            // Penny crack, a -> 0 (relative corrections of order a).
            let beta = mm * (3.0 * km + mm) / (3.0 * km + 4.0 * mm);
            let a = 1e-5;
            let pi = std::f64::consts::PI;
            let penny = (
                (km + 4.0 * mi / 3.0) / (ki + 4.0 * mi / 3.0 + pi * a * beta),
                (1.0 + 8.0 * mm / (4.0 * mi + pi * a * (mm + 2.0 * beta))
                    + 2.0 * (ki + 2.0 * (mi + mm) / 3.0) / (ki + 4.0 * mi / 3.0 + pi * a * beta))
                    / 5.0,
            );
            let crack = spheroid_pq(QUARTZ, inc, a).unwrap();
            println!("penny crack {penny:?}, a = 1e-5 {crack:?}");
            assert!(
                (crack.0 / penny.0 - 1.0).abs() < 1e-3 && (crack.1 / penny.1 - 1.0).abs() < 1e-3
            );
            // Needles, a -> infinity.
            let gamma = mm * (3.0 * km + mm) / (3.0 * km + 7.0 * mm);
            let needle = (
                (km + mm + mi / 3.0) / (ki + mm + mi / 3.0),
                (4.0 * mm / (mm + mi)
                    + 2.0 * (mm + gamma) / (mi + gamma)
                    + (ki + 4.0 * mm / 3.0) / (ki + mm + mi / 3.0))
                    / 5.0,
            );
            let long = spheroid_pq(QUARTZ, inc, 1e5).unwrap();
            println!("needle {needle:?}, a = 1e5 {long:?}");
            assert!(
                (long.0 / needle.0 - 1.0).abs() < 1e-3 && (long.1 / needle.1 - 1.0).abs() < 1e-3
            );
        }
    }

    #[test]
    fn dem_integration_and_dilute_limits() {
        // RK4 global error: four step counts against a converged reference.
        let (reference, used) = dem(QUARTZ, WATER, 0.1, 0.2).unwrap();
        let counts = [4usize, 8, 16, 32];
        let steps: Vec<f64> = counts.iter().map(|n| 0.2 / *n as f64).collect();
        let errors: Vec<f64> = counts
            .iter()
            .map(|n| {
                (dem_fixed_steps(QUARTZ, WATER, 0.1, 0.2, *n).unwrap().bulk - reference.bulk).abs()
            })
            .collect();
        let (p, r2) = fit(&steps, &errors);
        println!("DEM K {:.10} G {:.10} ({used} steps); RK4 errors {errors:?} at h {steps:?}: p {p:.6}, R^2 {r2:.8}", reference.bulk, reference.shear);
        assert!((p - 4.0).abs() < 0.15 && r2 > 0.9999);
        // Dilute limit: DEM and Kuster-Toksoz agree to O(phi^2).
        let fractions = [0.04, 0.02, 0.01, 0.005];
        let differences: Vec<f64> = fractions
            .iter()
            .map(|phi| {
                let d = dem(QUARTZ, WATER, 0.3, *phi).unwrap().0.bulk;
                let k = kuster_toksoz(
                    QUARTZ,
                    &[Inclusion {
                        fraction: *phi,
                        moduli: WATER,
                        aspect: 0.3,
                    }],
                )
                .unwrap()
                .bulk;
                (d - k).abs()
            })
            .collect();
        let (p, r2) = fit(&fractions, &differences);
        println!("DEM against Kuster-Toksoz: differences {differences:?} at phi {fractions:?}: p {p:.6}, R^2 {r2:.8}");
        assert!((p - 2.0).abs() < 0.1 && r2 > 0.9999);
    }

    fn fit(h: &[f64], e: &[f64]) -> (f64, f64) {
        let x: Vec<f64> = h.iter().map(|v| v.ln()).collect();
        let y: Vec<f64> = e.iter().map(|v| v.abs().ln()).collect();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    #[test]
    fn self_consistent_spheres_reproduce_the_aggregate_estimate_and_bounds_hold() {
        let phases = [
            Inclusion {
                fraction: 0.6,
                moduli: QUARTZ,
                aspect: 1.0,
            },
            Inclusion {
                fraction: 0.4,
                moduli: CALCITE,
                aspect: 1.0,
            },
        ];
        let sc = self_consistent(&phases).unwrap().unwrap();
        let aggregate = crate::geomech::aggregate_moduli(&[
            crate::geomech::Phase {
                fraction: 0.6,
                bulk: QUARTZ.bulk,
                shear: QUARTZ.shear,
            },
            crate::geomech::Phase {
                fraction: 0.4,
                bulk: CALCITE.bulk,
                shear: CALCITE.shear,
            },
        ])
        .unwrap();
        let (k, g) = aggregate.self_consistent.unwrap();
        println!("spheroid self-consistent {sc:?}, spherical aggregate ({k}, {g})");
        assert!((sc.bulk - k).abs() < 1e-10 * k && (sc.shear - g).abs() < 1e-10 * g);
        // Spheres of dry pores: DEM and KT lie inside the Hashin-Shtrikman bounds.
        for phi in [0.05, 0.15, 0.3] {
            let bounds = crate::geomech::aggregate_moduli(&[
                crate::geomech::Phase {
                    fraction: 1.0 - phi,
                    bulk: QUARTZ.bulk,
                    shear: QUARTZ.shear,
                },
                crate::geomech::Phase {
                    fraction: phi,
                    bulk: 0.0,
                    shear: 0.0,
                },
            ])
            .unwrap();
            let d = dem(QUARTZ, DRY, 1.0, phi).unwrap().0;
            let kt = kuster_toksoz(
                QUARTZ,
                &[Inclusion {
                    fraction: phi,
                    moduli: DRY,
                    aspect: 1.0,
                }],
            )
            .unwrap();
            println!(
                "phi {phi}: DEM {d:?}, KT {kt:?}, HS upper {:?}",
                bounds.hs_upper
            );
            assert!(d.bulk <= bounds.hs_upper.0 + 1e-9 && d.shear <= bounds.hs_upper.1 + 1e-9);
            // Dry spherical pores: KT coincides with the upper HS bound.
            assert!((kt.bulk - bounds.hs_upper.0).abs() < 1e-9 * bounds.hs_upper.0);
            assert!((kt.shear - bounds.hs_upper.1).abs() < 1e-9 * bounds.hs_upper.1);
        }
    }

    #[test]
    fn conductivity_bounds_are_ordered_and_match_maxwell_garnett() {
        let phases = [(0.7, 7.7), (0.3, 0.6)];
        let b = conductivity_bounds(&phases).unwrap();
        println!("{b:?}");
        assert!(b.harmonic <= b.hashin_shtrikman_lower && b.hashin_shtrikman_lower <= b.geometric);
        assert!(
            b.geometric <= b.hashin_shtrikman_upper && b.hashin_shtrikman_upper <= b.arithmetic
        );
        // Maxwell-Garnett: dilute spheres of l_i in the host l_m.
        let (lm, li, x) = (7.7, 0.6, 0.3);
        let beta = (li - lm) / (li + 2.0 * lm);
        let mg = lm * (1.0 + 2.0 * x * beta) / (1.0 - x * beta);
        assert!((mg - b.hashin_shtrikman_upper).abs() < 1e-12 * mg);
        assert!(
            conductivity_bounds(&[(1.0, 2.0)])
                .unwrap()
                .hashin_shtrikman_lower
                - 2.0
                < 1e-14
        );
    }
}
