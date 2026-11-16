// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum rock and soil physics on top of computed mineral properties.
//!
//! Units: moduli in GPa, densities in kg/m^3, porosity as a fraction,
//! viscosity in Pa s, permeability in m^2, thermal conductivity in W/(m K).
//! Physical inputs have no defaults; every function takes them explicitly.
//!
//! - Multiphase isotropic aggregates: Voigt, Reuss and Hill averages,
//!   Hashin-Shtrikman bounds (Berryman's form) and the self-consistent
//!   estimate for spherical grains.
//! - Dry frames: Hashin-Shtrikman bounds with empty pores, the critical
//!   porosity model and Hertz-Mindlin contact theory.
//! - Fluid substitution: Gassmann, anisotropic Brown-Korringa, Biot
//!   coefficient and modulus, Skempton coefficient, undrained moduli,
//!   saturated wave speeds and the Biot characteristic frequency.
//! - Isothermal to adiabatic elastic constants and orthotropic engineering
//!   constants for material cards.

use crate::error::{DftError, Result};
use crate::linalg::Matrix;

pub mod inclusions;
pub mod piezoelectric;

/// One isotropic phase of an aggregate.
#[derive(Clone, Copy, Debug)]
pub struct Phase {
    /// Volume fraction.
    pub fraction: f64,
    pub bulk: f64,
    pub shear: f64,
}

fn check_fractions(phases: &[Phase]) -> Result<()> {
    let total: f64 = phases.iter().map(|p| p.fraction).sum();
    if phases.is_empty()
        || phases
            .iter()
            .any(|p| p.fraction < 0.0 || p.bulk < 0.0 || p.shear < 0.0)
        || (total - 1.0).abs() > 1e-9
    {
        return Err(DftError::InvalidInput(
            "phases need non-negative fractions summing to one and non-negative moduli".into(),
        ));
    }
    Ok(())
}

/// Bounds and estimates of an isotropic aggregate (GPa).
#[derive(Clone, Debug)]
pub struct AggregateModuli {
    pub voigt: (f64, f64),
    pub reuss: (f64, f64),
    pub hill: (f64, f64),
    /// Hashin-Shtrikman lower and upper bounds `(K, G)`.
    pub hs_lower: (f64, f64),
    pub hs_upper: (f64, f64),
    /// Self-consistent estimate for spherical grains, when it converges.
    pub self_consistent: Option<(f64, f64)>,
}

fn hs_lambda(phases: &[Phase], z: f64) -> f64 {
    1.0 / phases
        .iter()
        .map(|p| p.fraction / (p.bulk + 4.0 * z / 3.0))
        .sum::<f64>()
        - 4.0 * z / 3.0
}

fn hs_gamma(phases: &[Phase], z: f64) -> f64 {
    1.0 / phases
        .iter()
        .map(|p| p.fraction / (p.shear + z))
        .sum::<f64>()
        - z
}

fn hs_zeta(k: f64, g: f64) -> f64 {
    if k + 2.0 * g == 0.0 {
        0.0
    } else {
        g / 6.0 * (9.0 * k + 8.0 * g) / (k + 2.0 * g)
    }
}

/// Voigt, Reuss, Hill, Hashin-Shtrikman and self-consistent moduli of an
/// isotropic multiphase aggregate. Phases with zero bulk modulus (empty
/// pores) give zero Reuss and lower bounds.
pub fn aggregate_moduli(phases: &[Phase]) -> Result<AggregateModuli> {
    check_fractions(phases)?;
    let voigt = (
        phases.iter().map(|p| p.fraction * p.bulk).sum::<f64>(),
        phases.iter().map(|p| p.fraction * p.shear).sum::<f64>(),
    );
    let harmonic = |f: &dyn Fn(&Phase) -> f64| -> f64 {
        if phases.iter().any(|p| p.fraction > 0.0 && f(p) == 0.0) {
            0.0
        } else {
            1.0 / phases.iter().map(|p| p.fraction / f(p)).sum::<f64>()
        }
    };
    let reuss = (harmonic(&|p| p.bulk), harmonic(&|p| p.shear));
    let hill = (0.5 * (voigt.0 + reuss.0), 0.5 * (voigt.1 + reuss.1));
    let (gmax, gmin) = phases.iter().fold((0.0_f64, f64::INFINITY), |(a, b), p| {
        (a.max(p.shear), b.min(p.shear))
    });
    let (kmax, kmin) = phases.iter().fold((0.0_f64, f64::INFINITY), |(a, b), p| {
        (a.max(p.bulk), b.min(p.bulk))
    });
    let bound = |k: f64, g: f64, kz: f64| -> (f64, f64) {
        let kb = if phases
            .iter()
            .any(|p| p.fraction > 0.0 && p.bulk + 4.0 * kz / 3.0 == 0.0)
        {
            0.0
        } else {
            hs_lambda(phases, kz)
        };
        let z = hs_zeta(k, g);
        let gb = if phases
            .iter()
            .any(|p| p.fraction > 0.0 && p.shear + z == 0.0)
        {
            0.0
        } else {
            hs_gamma(phases, z)
        };
        (kb, gb)
    };
    let hs_upper = bound(kmax, gmax, gmax);
    let hs_lower = bound(kmin, gmin, gmin);
    // Self-consistent (Berryman) for spheres by fixed-point iteration on
    // sum f_i (K_i - K)/(K_i + 4G/3) = 0, sum f_i (G_i - G)/(G_i + z(K,G)) = 0.
    let mut k = hill.0;
    let mut g = hill.1;
    let mut converged = false;
    for _ in 0..500 {
        if k <= 0.0 || g <= 0.0 {
            break;
        }
        let z = hs_zeta(k, g);
        let k_new = 1.0
            / phases
                .iter()
                .map(|p| p.fraction / (p.bulk + 4.0 * g / 3.0))
                .sum::<f64>()
            - 4.0 * g / 3.0;
        let g_new = 1.0
            / phases
                .iter()
                .map(|p| p.fraction / (p.shear + z))
                .sum::<f64>()
            - z;
        let change = (k_new - k).abs() + (g_new - g).abs();
        k = 0.5 * (k + k_new);
        g = 0.5 * (g + g_new);
        if change < 1e-12 * (k.abs() + g.abs()).max(1e-30) {
            converged = true;
            break;
        }
    }
    Ok(AggregateModuli {
        voigt,
        reuss,
        hill,
        hs_lower,
        hs_upper,
        self_consistent: (converged && k > 0.0 && g > 0.0).then_some((k, g)),
    })
}

/// Critical-porosity dry frame `K_dry = K_s (1 - phi/phi_c)`,
/// `G_dry = G_s (1 - phi/phi_c)` (zero beyond `phi_c`).
pub fn critical_porosity_frame(
    k_mineral: f64,
    g_mineral: f64,
    porosity: f64,
    critical: f64,
) -> Result<(f64, f64)> {
    if !(0.0..1.0).contains(&porosity) || critical <= 0.0 || critical > 1.0 {
        return Err(DftError::InvalidInput(
            "porosity in [0, 1) and critical porosity in (0, 1] are required".into(),
        ));
    }
    let factor = (1.0 - porosity / critical).max(0.0);
    Ok((k_mineral * factor, g_mineral * factor))
}

/// Hertz-Mindlin dry pack of identical spheres at effective pressure `p`
/// (GPa) with coordination number `c`, porosity `phi` and grain moduli.
pub fn hertz_mindlin_frame(
    k_mineral: f64,
    g_mineral: f64,
    porosity: f64,
    coordination: f64,
    pressure: f64,
) -> Result<(f64, f64)> {
    if !(0.0..1.0).contains(&porosity) || coordination <= 0.0 || pressure < 0.0 || g_mineral <= 0.0
    {
        return Err(DftError::InvalidInput(
            "Hertz-Mindlin needs porosity in [0, 1), positive coordination, grain \
            shear modulus and a non-negative pressure"
                .into(),
        ));
    }
    let nu = (3.0 * k_mineral - 2.0 * g_mineral) / (2.0 * (3.0 * k_mineral + g_mineral));
    let pi2 = std::f64::consts::PI.powi(2);
    let base =
        coordination * coordination * (1.0 - porosity).powi(2) * g_mineral * g_mineral * pressure;
    let k = (base / (18.0 * pi2 * (1.0 - nu).powi(2))).cbrt();
    let g = (5.0 - 4.0 * nu) / (5.0 * (2.0 - nu))
        * (3.0 * base / (2.0 * pi2 * (1.0 - nu).powi(2))).cbrt();
    Ok((k, g))
}

/// Saturated rock from a dry frame by Gassmann and Biot theory.
#[derive(Clone, Debug)]
pub struct FluidSubstitution {
    /// Saturated (undrained) bulk modulus by Gassmann (GPa).
    pub bulk_saturated: f64,
    /// Shear modulus, unchanged by the fluid (GPa).
    pub shear: f64,
    /// Biot coefficient `alpha = 1 - K_dry/K_s`.
    pub biot_coefficient: f64,
    /// Biot modulus `M`, `1/M = phi/K_f + (alpha - phi)/K_s` (GPa).
    pub biot_modulus: f64,
    /// Skempton coefficient `B = alpha M / K_u`.
    pub skempton: f64,
    /// Drained and undrained Poisson ratios.
    pub poisson_drained: f64,
    pub poisson_undrained: f64,
    /// Bulk density (kg/m^3).
    pub density: f64,
    /// Saturated P and S wave speeds (m/s, low-frequency Gassmann limit).
    pub vp: f64,
    pub vs: f64,
    /// Biot characteristic frequency `eta phi / (2 pi rho_f kappa)` (Hz), when
    /// the viscosity and permeability are given.
    pub biot_frequency: Option<f64>,
}

/// Gassmann fluid substitution with Biot parameters.
#[allow(clippy::too_many_arguments)]
pub fn fluid_substitution(
    k_dry: f64,
    g_dry: f64,
    k_mineral: f64,
    rho_mineral: f64,
    k_fluid: f64,
    rho_fluid: f64,
    porosity: f64,
    viscosity_permeability: Option<(f64, f64)>,
) -> Result<FluidSubstitution> {
    if !(0.0..1.0).contains(&porosity)
        || k_mineral <= 0.0
        || k_dry < 0.0
        || g_dry < 0.0
        || k_fluid < 0.0
        || rho_mineral <= 0.0
        || rho_fluid < 0.0
    {
        return Err(DftError::InvalidInput(
            "fluid substitution needs porosity in [0, 1), positive mineral modulus \
            and density, non-negative frame and fluid moduli"
                .into(),
        ));
    }
    let alpha = 1.0 - k_dry / k_mineral;
    let inverse_m = porosity / k_fluid.max(f64::MIN_POSITIVE)
        * if k_fluid == 0.0 { f64::INFINITY } else { 1.0 }
        + (alpha - porosity) / k_mineral;
    let biot_modulus = if inverse_m.is_finite() && inverse_m > 0.0 {
        1.0 / inverse_m
    } else {
        0.0
    };
    let bulk_saturated = k_dry + alpha * alpha * biot_modulus;
    let skempton = if bulk_saturated > 0.0 {
        alpha * biot_modulus / bulk_saturated
    } else {
        0.0
    };
    let poisson = |k: f64, g: f64| (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g));
    let density = (1.0 - porosity) * rho_mineral + porosity * rho_fluid;
    let vp = ((bulk_saturated + 4.0 * g_dry / 3.0) * 1e9 / density).sqrt();
    let vs = (g_dry * 1e9 / density).sqrt();
    let biot_frequency = match viscosity_permeability {
        Some((eta, kappa)) if kappa > 0.0 && rho_fluid > 0.0 => {
            Some(eta * porosity / (2.0 * std::f64::consts::PI * rho_fluid * kappa))
        }
        Some(_) => {
            return Err(DftError::InvalidInput(
                "the Biot frequency needs a positive permeability and fluid density".into(),
            ))
        }
        None => None,
    };
    Ok(FluidSubstitution {
        bulk_saturated,
        shear: g_dry,
        biot_coefficient: alpha,
        biot_modulus,
        skempton,
        poisson_drained: poisson(k_dry, g_dry),
        poisson_undrained: poisson(bulk_saturated, g_dry),
        density,
        vp,
        vs,
        biot_frequency,
    })
}

/// Invert a symmetric 6x6 Voigt tensor.
pub fn compliance(c: &[[f64; 6]; 6]) -> Result<[[f64; 6]; 6]> {
    let mut m = Matrix::zeros(6, 6);
    for i in 0..6 {
        for j in 0..6 {
            m[(i, j)] = 0.5 * (c[i][j] + c[j][i]);
        }
    }
    let s = crate::linalg::solve_linear_matrix(&m, &Matrix::identity(6))?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| s[(i, j)])))
}

/// Anisotropic Brown-Korringa substitution for a single-mineral frame:
/// `S_sat = S_dry - (S_dry - S_s) : 1 1 : (S_dry - S_s) /
/// [(S_dry - S_s) :: 1 1 + phi (1/K_f - 1/K_s)]` in compliance form, where
/// `1/K_s = 1 : S_s : 1`. Returns the saturated stiffness (GPa).
pub fn brown_korringa(
    c_dry: &[[f64; 6]; 6],
    c_mineral: &[[f64; 6]; 6],
    k_fluid: f64,
    porosity: f64,
) -> Result<[[f64; 6]; 6]> {
    if !(0.0..1.0).contains(&porosity) || k_fluid <= 0.0 {
        return Err(DftError::InvalidInput(
            "Brown-Korringa needs porosity in [0, 1) and a positive fluid modulus".into(),
        ));
    }
    let s_dry = compliance(c_dry)?;
    let s_min = compliance(c_mineral)?;
    // The isotropic stress 1 in Voigt form contracts with the first three
    // rows and columns.
    let row = |s: &[[f64; 6]; 6], i: usize| -> f64 { (0..3).map(|j| s[i][j]).sum() };
    let inv_k_min: f64 = (0..3).map(|i| row(&s_min, i)).sum();
    let diff: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| s_dry[i][j] - s_min[i][j]));
    let d: [f64; 6] = std::array::from_fn(|i| row(&diff, i));
    let denominator = (0..3).map(|i| d[i]).sum::<f64>() + porosity * (1.0 / k_fluid - inv_k_min);
    let s_sat: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| s_dry[i][j] - d[i] * d[j] / denominator));
    compliance(&s_sat)
}

/// Adiabatic elastic constants `C^S = C^T + (T V / C_V) beta beta^T` with
/// the thermal stress coefficients `beta = C^T alpha` (Voigt, engineering
/// thermal strains `alpha` with the shear entries doubled), the volume per
/// cell `V` and `C_V` per cell in consistent units (GPa, m^3, J/K).
pub fn adiabatic_constants(
    c_t: &[[f64; 6]; 6],
    alpha: &[f64; 6],
    temperature: f64,
    volume: f64,
    heat_capacity: f64,
) -> Result<[[f64; 6]; 6]> {
    if heat_capacity <= 0.0 || volume <= 0.0 || temperature < 0.0 {
        return Err(DftError::InvalidInput("adiabatic constants need positive volume and heat capacity and a non-negative temperature".into()));
    }
    let beta: [f64; 6] = std::array::from_fn(|i| (0..6).map(|j| c_t[i][j] * alpha[j]).sum());
    // GPa^2 m^3 K / (J/K) = GPa^2 m^3 / J = 1e9 GPa.
    let scale = temperature * volume / heat_capacity * 1e9;
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| c_t[i][j] + scale * beta[i] * beta[j])
    }))
}

/// Orthotropic engineering constants from a compliance in the material
/// axes: Young moduli `E_i = 1/S_ii`, Poisson ratios `nu_ij = -S_ji/S_ii`
/// and shear moduli `G_23 = 1/S_44`, `G_13 = 1/S_55`, `G_12 = 1/S_66`.
pub fn engineering_constants(s: &[[f64; 6]; 6]) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
    let young = [1.0 / s[0][0], 1.0 / s[1][1], 1.0 / s[2][2]];
    let poisson = std::array::from_fn(|i| {
        std::array::from_fn(|j| if i == j { 0.0 } else { -s[j][i] / s[i][i] })
    });
    let shear = [1.0 / s[3][3], 1.0 / s[4][4], 1.0 / s[5][5]];
    (young, poisson, shear)
}

/// Rotate a Voigt stiffness tensor by the rotation matrix `r` (rows are the
/// new axes in old coordinates) with the Bond transformation.
pub fn rotate_stiffness(c: &[[f64; 6]; 6], r: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    const PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let index = |a: usize, b: usize| {
        PAIRS
            .iter()
            .position(|&(x, y)| (x, y) == (a, b) || (y, x) == (a, b))
            .unwrap()
    };
    let full = |a: usize, b: usize, c2: usize, d: usize| c[index(a, b)][index(c2, d)];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (p, q) = PAIRS[i];
            let (s, t) = PAIRS[j];
            let mut v = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    for cc in 0..3 {
                        for d in 0..3 {
                            v += r[p][a] * r[q][b] * r[s][cc] * r[t][d] * full(a, b, cc, d);
                        }
                    }
                }
            }
            v
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_bounds_are_ordered_and_collapse_for_one_phase() {
        let quartz = Phase {
            fraction: 0.6,
            bulk: 37.0,
            shear: 44.0,
        };
        let calcite = Phase {
            fraction: 0.3,
            bulk: 76.8,
            shear: 32.0,
        };
        let clay = Phase {
            fraction: 0.1,
            bulk: 21.0,
            shear: 7.0,
        };
        let m = aggregate_moduli(&[quartz, calcite, clay]).unwrap();
        assert!(
            m.reuss.0 <= m.hs_lower.0 + 1e-12
                && m.hs_lower.0 <= m.hs_upper.0 + 1e-12
                && m.hs_upper.0 <= m.voigt.0 + 1e-12
        );
        assert!(
            m.reuss.1 <= m.hs_lower.1 + 1e-12
                && m.hs_lower.1 <= m.hs_upper.1 + 1e-12
                && m.hs_upper.1 <= m.voigt.1 + 1e-12
        );
        let (k, g) = m.self_consistent.unwrap();
        assert!(m.hs_lower.0 <= k && k <= m.hs_upper.0 && m.hs_lower.1 <= g && g <= m.hs_upper.1);
        let one = aggregate_moduli(&[Phase {
            fraction: 1.0,
            bulk: 37.0,
            shear: 44.0,
        }])
        .unwrap();
        for (a, b) in [
            one.voigt,
            one.reuss,
            one.hs_lower,
            one.hs_upper,
            one.self_consistent.unwrap(),
        ] {
            assert!((a - 37.0).abs() < 1e-9 && (b - 44.0).abs() < 1e-9);
        }
    }

    #[test]
    fn gassmann_limits_and_biot_consistency() {
        let (k_s, g_s, rho_s) = (37.0, 44.0, 2650.0);
        let (k_f, rho_f) = (2.25, 1000.0);
        let phi = 0.2;
        let (k_dry, g_dry) = critical_porosity_frame(k_s, g_s, phi, 0.4).unwrap();
        let sat = fluid_substitution(
            k_dry,
            g_dry,
            k_s,
            rho_s,
            k_f,
            rho_f,
            phi,
            Some((1e-3, 1e-13)),
        )
        .unwrap();
        // Gassmann in its classical form.
        let gassmann = k_dry
            + (1.0 - k_dry / k_s).powi(2) / (phi / k_f + (1.0 - phi) / k_s - k_dry / (k_s * k_s));
        assert!((sat.bulk_saturated - gassmann).abs() < 1e-12 * gassmann);
        assert!(
            sat.skempton > 0.0 && sat.skempton < 1.0 && sat.poisson_undrained > sat.poisson_drained
        );
        // phi -> 0: the mineral; K_f -> 0: the dry frame.
        let zero = fluid_substitution(k_s, g_s, k_s, rho_s, k_f, rho_f, 0.0, None).unwrap();
        assert!((zero.bulk_saturated - k_s).abs() < 1e-12 * k_s);
        let dry = fluid_substitution(k_dry, g_dry, k_s, rho_s, 1e-12, rho_f, phi, None).unwrap();
        assert!((dry.bulk_saturated - k_dry).abs() < 1e-9 * k_dry);
        let f = sat.biot_frequency.unwrap();
        assert!((f - 1e-3 * phi / (2.0 * std::f64::consts::PI * rho_f * 1e-13)).abs() < 1e-9 * f);
    }

    fn isotropic(k: f64, g: f64) -> [[f64; 6]; 6] {
        let lambda = k - 2.0 * g / 3.0;
        let mut c = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = lambda + if i == j { 2.0 * g } else { 0.0 };
            }
            c[3 + i][3 + i] = g;
        }
        c
    }

    #[test]
    fn brown_korringa_reduces_to_gassmann_for_isotropic_media() {
        let (k_s, g_s, k_dry, g_dry, k_f, phi) = (37.0, 44.0, 18.5, 22.0, 2.25, 0.2);
        let c_sat =
            brown_korringa(&isotropic(k_dry, g_dry), &isotropic(k_s, g_s), k_f, phi).unwrap();
        let k_sat = (c_sat[0][0] + 2.0 * c_sat[0][1]) / 3.0;
        let gassmann = k_dry
            + (1.0 - k_dry / k_s).powi(2) / (phi / k_f + (1.0 - phi) / k_s - k_dry / (k_s * k_s));
        assert!((k_sat - gassmann).abs() < 1e-9 * gassmann);
        assert!((c_sat[3][3] - g_dry).abs() < 1e-9 * g_dry);
    }

    #[test]
    fn engineering_constants_and_rotation_of_an_isotropic_tensor() {
        let (k, g) = (60.0, 30.0);
        let s = compliance(&isotropic(k, g)).unwrap();
        let (e, nu, shear) = engineering_constants(&s);
        let young = 9.0 * k * g / (3.0 * k + g);
        let poisson = (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g));
        for i in 0..3 {
            assert!((e[i] - young).abs() < 1e-9 * young && (shear[i] - g).abs() < 1e-9 * g);
            for j in 0..3 {
                if i != j {
                    assert!((nu[i][j] - poisson).abs() < 1e-12);
                }
            }
        }
        let (c, s1) = (0.6_f64, 0.8_f64);
        let r = [[c, s1, 0.0], [-s1, c, 0.0], [0.0, 0.0, 1.0]];
        let rotated = rotate_stiffness(&isotropic(k, g), &r);
        let original = isotropic(k, g);
        for i in 0..6 {
            for j in 0..6 {
                assert!((rotated[i][j] - original[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn adiabatic_bulk_modulus_follows_the_gruneisen_relation() {
        // Isotropic: K_S = K_T (1 + alpha_V gamma T), gamma = alpha_V K_T V / C_V.
        let (k, g) = (60.0, 30.0);
        let alpha_v = 3e-5;
        let (t, v, cv) = (300.0, 2.0e-5, 25.0);
        let a = alpha_v / 3.0;
        let cs =
            adiabatic_constants(&isotropic(k, g), &[a, a, a, 0.0, 0.0, 0.0], t, v, cv).unwrap();
        let k_s = (cs[0][0] + 2.0 * cs[0][1]) / 3.0;
        let gamma = alpha_v * k * 1e9 * v / cv;
        assert!((k_s - k * (1.0 + alpha_v * gamma * t)).abs() < 1e-9 * k);
        assert!((cs[3][3] - g).abs() < 1e-12);
    }
}
