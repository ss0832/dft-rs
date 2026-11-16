// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! TPSS meta-GGA (Tao-Perdew-Staroverov-Scuseria, PRL 91, 146401 (2003)).
//!
//! Exchange (per spin via the scaling e_x[rho_s] = 1/2 e_x[2 rho_s]):
//!   p = |grad rho|^2 / [4 (3 pi^2)^{2/3} rho^{8/3}],
//!   z = tau^W/tau (tau^W = |grad rho|^2/(8 rho)),
//!   alpha = (tau - tau^W)/tau^unif,
//!   q_b = (9/20)(alpha-1)/sqrt(1 + b alpha(alpha-1)) + 2p/3,
//!   x = { [10/81 + c z^2/(1+z^2)^2] p + (146/2025) q_b^2
//!         - (73/405) q_b sqrt((3z/5)^2/2 + p^2/2)
//!         + (10/81)^2 p^2 / kappa + 2 sqrt(e) (10/81)(3z/5)^2
//!         + e mu p^3 } / (1 + sqrt(e) p)^2,
//!   F_x = 1 + kappa - kappa/(1 + x/kappa).
//!
//! Correlation (revPKZB + self-interaction correction):
//!   eps_revPKZB = eps^PBE [1 + C(zeta,xi) z_t^2]
//!               - (1 + C) z_t^2 sum_s (rho_s/rho) eps~_s,
//!   eps~_s = max[eps^PBE(rho_s, 0), eps^PBE(full)],
//!   C(zeta,xi) = (0.53 + 0.87 zeta^2 + 0.50 zeta^4 + 2.26 zeta^6)
//!                / [1 + xi^2 ((1+zeta)^{-4/3} + (1-zeta)^{-4/3})/2]^4,
//!   e_c = rho eps_revPKZB [1 + d eps_revPKZB z_t^3],  z_t = tau^W/tau.
//!
//! Fully spin-polarized channels route through the ferro branches
//! (pw92_epsilon_ferro / pbe_c_eps_ferro) - the zeta interpolations have
//! singular second dual derivatives at zeta = 1 (see THEORY.md).
//! Verified pointwise against libxc (mgga_x_tpss / mgga_c_tpss).

use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

use super::common::RHO_CUTOFF;
const CX_SPIN: f64 = -super::common::X_FACTOR_C;

/// Spin-scaled TPSS exchange for one spin channel: 1/2 e_x[2 rho, 4 sig,
/// 2 tau]. theta: [kappa, mu, b, c, e, n], where the z term of `x` is
/// `c z^n / (1 + z^2)^2` (n = 2 for TPSS and modTPSS, 3 for revTPSS).
fn tpss_x_spin<S: Scalar>(rho_s: S, sig_s: S, tau_s: S, th: &[S]) -> S {
    let (kappa, mu, b, c, ee) = (th[0], th[1], th[2], th[3], th[4]);
    let one = S::from_f64(1.0);
    let rho = S::from_f64(2.0) * rho_s;
    let sig = S::from_f64(4.0) * sig_s;
    let tau = S::from_f64(2.0) * tau_s;
    let tau = if tau.value() < 1e-20 {
        S::from_f64(1e-20)
    } else {
        tau
    };
    let r83 = rho.powf(8.0 / 3.0);
    // Keep the per-σ COEFFICIENTS, not just the products: `term3` below needs
    // to divide σ back out of a square root, and doing that symbolically is
    // what keeps the radicand away from zero.
    let p_coef =
        (S::from_f64(4.0 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0))
            * r83)
            .recip();
    let p = sig * p_coef;
    let tau_w = sig / (S::from_f64(8.0) * rho);
    let z = tau_w / tau;
    let tau_unif =
        S::from_f64(0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0))
            * rho.powf(5.0 / 3.0);
    let alpha = (tau - tau_w) / tau_unif;
    let qb = S::from_f64(9.0 / 20.0) * (alpha - one) / (one + b * alpha * (alpha - one)).sqrt()
        + S::from_f64(2.0 / 3.0) * p;
    let z2 = z * z;
    let t35 = S::from_f64(3.0 / 5.0) * z; // (3z/5)
    let zn = if th[5].value() == 2.0 { z2 } else { z2 * z };
    let term1 = (S::from_f64(10.0 / 81.0) + c * zn / ((one + z2) * (one + z2))) * p;
    let term2 = S::from_f64(146.0 / 2025.0) * qb * qb;
    // (½t₃₅² + ½p²) is σ²·(½(t₃₅/σ)² + ½(p/σ)²) — BOTH terms are exactly
    // proportional to σ, so the root is σ·constant: linear in σ, not a kink.
    // Rooting the sum directly evaluates √0 at σ = 0, and `DualN::sqrt` forms
    // `0.5/√0 = ∞` there, which the zero seed turns into NaN in EVERY tangent
    // slot — vrho and vtau included, not just vsigma. Factoring σ out is an
    // identity (σ ≥ 0 is guaranteed by the driver) and leaves the radicand
    // strictly positive, so the derivatives are the true ones rather than a
    // regularisation.
    //   t₃₅/σ = (3/5)·(1/(8ρτ)) = 0.075/(ρτ)
    let t35_coef = S::from_f64(0.075) / (rho * tau);
    let term3 = S::from_f64(-73.0 / 405.0)
        * qb
        * sig
        * (S::from_f64(0.5) * t35_coef * t35_coef + S::from_f64(0.5) * p_coef * p_coef).sqrt();
    let term4 = S::from_f64((10.0f64 / 81.0).powi(2)) * p * p / kappa;
    let sqrte = ee.sqrt();
    let term5 = S::from_f64(2.0) * sqrte * S::from_f64(10.0 / 81.0) * t35 * t35;
    let term6 = ee * mu * p * p * p;
    let den = (one + sqrte * p) * (one + sqrte * p);
    let x = (term1 + term2 + term3 + term4 + term5 + term6) / den;
    let fx = one + kappa - kappa / (one + x / kappa);
    // 1/2 e_x^LDA(2 rho_s) = CX_SPIN rho_s^{4/3} (spin scaling built in).
    S::from_f64(CX_SPIN) * rho_s.powf(4.0 / 3.0) * fx
}

pub(crate) fn tpss_x_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        e = e + tpss_x_spin(rho, sig, tau, th);
    }
    e
}

/// TPSS-family correlation. theta: [d, beta, gamma, pw92_mod x 18, c0, c1,
/// c2, c3, a, b] with `C(zeta, 0) = c0 + c1 zeta^2 + c2 zeta^4 + c3 zeta^6` and
/// the PBE `beta(rs) = beta (1 + a rs) / (1 + b rs)` (a = b = 0 for TPSS;
/// revTPSS takes 0.1 and 0.1778).
pub(crate) fn tpss_c_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    tpss_c_body::<S, false>(ra, rb, saa, sab, sbb, ta, tb, th)
}

fn tpssloc_c_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    tpss_c_body::<S, true>(ra, rb, saa, sab, sbb, ta, tb, th)
}

// The constant selects the correlation base at monomorphization time;
// ordinary TPSS/revTPSS retain their original parameter layout and arithmetic.
fn tpss_c_body<S: Scalar, const LOCALIZED: bool>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    let d = th[0];
    let gamma = th[2];
    let pw = &th[3..21];
    let cz = [th[21], th[22], th[23], th[24]];
    let beta_of = |rho: S| -> S {
        let rs = super::common::wigner_seitz(rho);
        th[1] * (S::from_f64(1.0) + th[25] * rs) / (S::from_f64(1.0) + th[26] * rs)
    };
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let have_a = ra.value() >= RHO_CUTOFF;
    let have_b = rb.value() >= RHO_CUTOFF;
    if !have_a && !have_b {
        return zero;
    }
    // Single-spin branch: everything collapses to the ferro forms.
    if !have_a || !have_b {
        let (rho, sig, tau) = if have_a { (ra, saa, ta) } else { (rb, sbb, tb) };
        let tau = if tau.value() < 1e-20 {
            S::from_f64(1e-20)
        } else {
            tau
        };
        let eps = if LOCALIZED {
            super::gga::pbeloc_c_eps(rho, zero, sig, zero, zero, [th[1], th[25]], gamma, pw)
        } else {
            super::gga::pbe_c_eps_ferro(rho, sig, beta_of(rho), gamma, pw)
        };
        let tau_w = sig / (S::from_f64(8.0) * rho);
        let zt = at_most_one(tau_w / tau);
        let zt2 = zt * zt;
        // C(1, 0) = c0 + c1 + c2 + c3; xi = 0 for one spin.
        let c10 = cz[0] + cz[1] + cz[2] + cz[3];
        let rev = eps * (one + c10 * zt2) - (one + c10) * zt2 * eps;
        rho * rev * (one + d * rev * zt2 * zt)
    } else {
        let rho = ra + rb;
        let tau = ta + tb;
        let tau = if tau.value() < 1e-20 {
            S::from_f64(1e-20)
        } else {
            tau
        };
        let sig_tot = saa + S::from_f64(2.0) * sab + sbb;
        let eps_full = if LOCALIZED {
            super::gga::pbeloc_c_eps(ra, rb, saa, sab, sbb, [th[1], th[25]], gamma, pw)
        } else {
            super::gga::pbe_c_eps(ra, rb, saa, sab, sbb, beta_of(rho), gamma, pw)
        };
        let tau_w = sig_tot / (S::from_f64(8.0) * rho);
        let zt = at_most_one(tau_w / tau);
        let zt2 = zt * zt;
        let zeta = (ra - rb) / rho;
        // |grad zeta|^2 = 4 [rb^2 saa - 2 ra rb sab + ra^2 sbb] / rho^4.
        let gz2 = S::from_f64(4.0)
            * (rb * rb * saa - S::from_f64(2.0) * ra * rb * sab + ra * ra * sbb)
            / (rho * rho * rho * rho);
        let kf2 =
            (S::from_f64(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).powf(2.0 / 3.0);
        let xi2 = gz2 / (S::from_f64(4.0) * kf2);
        let z2 = zeta * zeta;
        let c0 = cz[0] + cz[1] * z2 + cz[2] * z2 * z2 + cz[3] * zeta.powf(6.0);
        let opz = one + zeta;
        let omz = one - zeta;
        let cd = one + xi2 * (opz.powf(-4.0 / 3.0) + omz.powf(-4.0 / 3.0)) * S::from_f64(0.5);
        let cc = c0 / (cd * cd * cd * cd);
        // eps~_s = max[eps^PBE(rho_s only), eps_full].
        let mut sum_tilde = zero;
        for (rs, ss) in [(ra, saa), (rb, sbb)] {
            let eps_s = if LOCALIZED {
                super::gga::pbeloc_c_eps(rs, zero, ss, zero, zero, [th[1], th[25]], gamma, pw)
            } else {
                super::gga::pbe_c_eps_ferro(rs, ss, beta_of(rs), gamma, pw)
            };
            let tilde = if eps_s.value() > eps_full.value() {
                eps_s
            } else {
                eps_full
            };
            sum_tilde = sum_tilde + rs / rho * tilde;
        }
        let rev = eps_full * (one + cc * zt2) - (one + cc) * zt2 * sum_tilde;
        rho * rev * (one + d * rev * zt2 * zt)
    }
}

/// `min(z, 1)` for the correlation indicator `z = tau_W / tau` of the
/// total density: the per-spin Fermi-hole bound of the driver does not bound
/// the total-density ratio, and the reference implementation caps it. Only
/// densities with `tau < tau_W` reach the cap. At the attainable physical
/// endpoint use the inward derivative. A total-density ratio involves
/// seven additions/products/divisions; an eight-epsilon roundoff enclosure
/// prevents an exactly iso-orbital density from randomly choosing the
/// constant extension when assembled by different AO contractions. The
/// value stays capped at one; no density or physical denominator is floored.
fn at_most_one<S: Scalar>(z: S) -> S {
    if z.value() > 1.0 + 8.0 * f64::EPSILON {
        S::from_f64(1.0)
    } else if z.value() > 1.0 {
        z - S::from_f64(z.value() - 1.0)
    } else {
        z
    }
}

static TPSS_X_PARAM_NAMES: [&str; 6] = ["kappa", "mu", "b", "c", "e", "n"];
static TPSS_C_PARAM_NAMES: [&str; 27] = [
    "d", "beta", "gamma", "A_0", "a1_0", "b1_0", "b2_0", "b3_0", "b4_0", "A_1", "a1_1", "b1_1",
    "b2_1", "b3_1", "b4_1", "A_a", "a1_a", "b1_a", "b2_a", "b3_a", "b4_a", "c0", "c1", "c2", "c3",
    "beta_a", "beta_b",
];

fn tpss_x_with(name: &'static str, params: [f64; 6]) -> Box<dyn XcFunctional> {
    mgga_driver!(
        name,
        Kind::Exchange,
        params.to_vec(),
        &TPSS_X_PARAM_NAMES,
        tpss_x_e,
        true
    )
}

fn tpss_c_with(
    name: &'static str,
    d: f64,
    cz: [f64; 4],
    beta_rs: [f64; 2],
) -> Box<dyn XcFunctional> {
    let mut params = vec![d, 0.06672455060314922, 0.031090690869654895];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    params.extend_from_slice(&cz);
    params.extend_from_slice(&beta_rs);
    mgga_driver!(
        name,
        Kind::Correlation,
        params,
        &TPSS_C_PARAM_NAMES,
        tpss_c_e,
        true
    )
}

/// revTPSS exchange (Perdew et al., PRL 103, 026403 (2009)).
pub fn revtpss_x() -> Box<dyn XcFunctional> {
    tpss_x_with("revtpss", [0.804, 0.14, 0.40, 2.35203946, 2.16769874, 3.0])
}

/// revTPSS correlation: revised C(zeta, 0) and beta(rs).
pub fn revtpss_c() -> Box<dyn XcFunctional> {
    tpss_c_with(
        "revtpss",
        2.8,
        [0.59, 0.9269, 0.6225, 2.1540],
        [0.1, 0.1778],
    )
}

/// modTPSS exchange (Perdew-Ruzsinszky-Tao et al., PRA 76, 042506 (2007)).
pub fn modtpss_x() -> Box<dyn XcFunctional> {
    tpss_x_with("modtpss", [0.804, 0.252, 0.40, 1.38496, 1.37, 2.0])
}

pub fn tpss_x() -> Box<dyn XcFunctional> {
    tpss_x_with("tpss", [0.804, 0.21951, 0.40, 1.59096, 1.537, 2.0])
}

pub fn tpss_c() -> Box<dyn XcFunctional> {
    tpss_c_with("tpss", 2.8, [0.53, 0.87, 0.50, 2.26], [0.0, 0.0])
}

/// TPSSloc correlation (Constantin, Fabiano and Della Sala, PRB 86,
/// 035130 (2012)): localized PBE base, C(0,0)=0.35 and d=4.5.
/// Parameters retain the TPSS correlation layout through c3, followed by
/// the localized beta slope; beta0 occupies the ordinary beta slot.
pub fn tpssloc_c() -> Box<dyn XcFunctional> {
    let mut params = vec![4.5, 0.0375, 0.031090690869654895];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    params.extend_from_slice(&[0.35, 0.87, 0.50, 2.26, 0.08]);
    mgga_driver!(
        "tpssloc",
        Kind::Correlation,
        params,
        &[
            "d",
            "beta0",
            "gamma",
            "A_0",
            "a1_0",
            "b1_0",
            "b2_0",
            "b3_0",
            "b4_0",
            "A_1",
            "a1_1",
            "b1_1",
            "b2_1",
            "b3_1",
            "b4_1",
            "A_a",
            "a1_a",
            "b1_a",
            "b2_a",
            "b3_a",
            "b4_a",
            "c0",
            "c1",
            "c2",
            "c3",
            "beta_gradient",
        ],
        tpssloc_c_e,
        true
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{Family, GridVars, XcOut};

    #[test]
    fn iso_orbital_endpoint_uses_the_physical_inward_derivative() {
        use crate::dual::DualN;
        type D = DualN<f64, 1>;
        type H = DualN<D, 1>;
        type T = DualN<H, 1>;
        for value in [
            1.0 - f64::EPSILON,
            1.0,
            1.0 + f64::EPSILON,
            1.0 + 8.0 * f64::EPSILON,
        ] {
            let mut z = T::from_f64(value);
            z.re.re.im[0] = 1.0;
            z.re.im[0].re = 1.0;
            z.im[0].re.re = 1.0;
            let bounded = at_most_one(z);
            assert_eq!(bounded.value(), value.min(1.0));
            let cube = bounded * bounded * bounded;
            let x = bounded.value();
            assert!((cube.re.re.im[0] - 3.0 * x * x).abs() < 1e-14);
            assert!((cube.re.im[0].im[0] - 6.0 * x).abs() < 1e-14);
            assert!((cube.im[0].im[0].im[0] - 6.0).abs() < 1e-14);
        }
        let z = D {
            re: 1.0 + 1e-10,
            im: [1.0],
        };
        let cap = at_most_one(z);
        assert_eq!((cap.re, cap.im[0]), (1.0, 0.0));
    }

    fn eval(f: &dyn XcFunctional, v: [f64; 7]) -> f64 {
        let rho_a = [v[0]];
        let rho_b = [v[1]];
        let saa = [v[2]];
        let sab = [v[3]];
        let sbb = [v[4]];
        let ta = [v[5]];
        let tb = [v[6]];
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: Some([&saa, &sab, &sbb]),
            tau: Some([&ta, &tb]),
        };
        let mut out = XcOut::default();
        out.prepare(1, 0, Family::MetaGga);
        f.eval_add(&vars, 0, 1.0, &mut out).unwrap();
        out.e[0]
    }

    /// TPSS X and C pinned pointwise against libxc at three spin-polarized
    /// meta points (PySCF probe 2026-08-04).
    #[test]
    fn tpss_matches_libxc_reference_points() {
        let sig = |g: [f64; 3], h: [f64; 3]| g[0] * h[0] + g[1] * h[1] + g[2] * h[2];
        let cases: [([f64; 2], [f64; 3], [f64; 3], [f64; 2], f64, f64); 3] = [
            (
                [0.3, 0.1],
                [0.1, 0.1, 0.15],
                [0.08, 0.05, 0.12],
                [0.35, 0.12],
                -0.232443560323,
                -0.021627488389,
            ),
            (
                [1.2, 0.9],
                [0.3, -0.2, 0.4],
                [0.25, 0.1, -0.3],
                [1.9, 1.4],
                -2.010870924312,
                -0.160999823877,
            ),
            (
                [0.05, 0.4],
                [0.02, 0.0, 0.05],
                [0.3, 0.2, -0.1],
                [0.06, 0.9],
                -0.292724142324,
                -0.020810377498,
            ),
        ];
        let fx = tpss_x();
        let fc = tpss_c();
        for (r, ga, gb, t, want_x, want_c) in cases {
            let v = [
                r[0],
                r[1],
                sig(ga, ga),
                sig(ga, gb),
                sig(gb, gb),
                t[0],
                t[1],
            ];
            let ex = eval(fx.as_ref(), v);
            let ec = eval(fc.as_ref(), v);
            assert!((ex - want_x).abs() < 1e-9, "TPSS-X {ex} vs {want_x}");
            assert!((ec - want_c).abs() < 1e-9, "TPSS-C {ec} vs {want_c}");
        }
    }
}
