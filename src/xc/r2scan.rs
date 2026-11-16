// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! r²SCAN meta-GGA (Furness, Kaplan, Ning, Perdew, Sun, J. Phys. Chem.
//! Lett. 11, 8208 (2020); + erratum 11, 9248). The re-regularized SCAN:
//! SCAN's iso-orbital indicator α is replaced by
//!   ᾱ = (τ − τ_W)/(τ_unif + η·τ_W),
//! the α-interpolation f(ᾱ) becomes a degree-7 polynomial on [0, 2.5]
//! (rSCAN switching function) with exp branches outside, and the gradient
//! expansion is restored through the C_η/dp2 damping (exchange, Eq. S10)
//! and the Δy term (correlation, Eq. S34).
//!
//! Exchange per spin: F_x = [h1x(x) + f(ᾱ)(h0x − h1x(x))]·g_x(s) with
//! h1x(x) = 1 + k1(1 − k1/(k1 + x)), x = (C_η C₂ exp(−p²/dp2⁴) + μ)p,
//! g_x = 1 − exp(−a1/√s); correlation: ε = ε_c1 + f_c(ᾱ)(ε_c0 − ε_c1)
//! on the PW92(MOD) local base with the β(rs) of regTPSS.
//!
//! Fully polarized channels route through ferro branches (the ζ
//! interpolations have singular second dual derivatives at ζ = 1).
//! Verified pointwise against libxc mgga_x_r2scan / mgga_c_r2scan.
//!
//! The original SCAN (unregularized α, exponential interpolation) shares
//! the constants and the local correlation pieces and is defined below.

use super::traits::{Kind, XcFunctional};
use crate::dual::{DualN, Scalar};

use super::common::RHO_CUTOFF;
/// −(3/4)(3/π)^{1/3}·2^{1/3}: per-spin LDA exchange prefactor.
const CX_SPIN: f64 = -super::common::X_FACTOR_C;
/// X2S = 1/(2(6π²)^{1/3}) (per-spin reduced gradient), K_FACTOR_C =
/// (3/10)(6π²)^{2/3} (per-spin τ_unif prefactor) — libxc reduced units.
fn x2s() -> f64 {
    super::common::X2S
}
fn k_factor_c() -> f64 {
    super::common::K_FACTOR_C
}

const MU_GE: f64 = 10.0 / 81.0;
const ETA: f64 = 0.001;
const DP2: f64 = 0.361;

/// rSCAN switching polynomial (c0..c7) for exchange and its (c1x, c2x, dx).
const FX_POLY: [f64; 8] = [
    1.0,
    -0.667,
    -0.4445555,
    -0.663086601049,
    1.451297044490,
    -0.887998041597,
    0.234528941479,
    -0.023185843322,
];
const C1X: f64 = 0.667;
const C2X: f64 = 0.8;
const DX: f64 = 1.24;
const K1: f64 = 0.065;
const H0X: f64 = 1.174;
const A1X: f64 = 4.9479;

/// rSCAN switching polynomial for correlation and its (c1c, c2c, dc).
const FC_POLY: [f64; 8] = [
    1.0,
    -0.64,
    -0.4352,
    -1.535685604549,
    3.061560252175,
    -1.915710236206,
    0.516884468372,
    -0.051848879792,
];
const C1C: f64 = 0.64;
const C2C: f64 = 1.5;
const DC: f64 = 0.7;

/// SCAN LDA0 correlation fit (Eq. S25 constants).
const B1C: f64 = 0.0285764;
const B2C: f64 = 0.0889;
const B3C: f64 = 0.125541;
const CHI_INF: f64 = 0.12802585262625815;
const GC_CNST: f64 = 2.363;

/// PW92 with the reference-implementation "MOD" digits and the exact
/// f''(0) — the local base of the SCAN family.
use super::lda::PW92_MOD_PARAMS as PW92_MOD;

/// The α-interpolation: exp branch (ᾱ ≤ 0), degree-7 polynomial
/// (0 < ᾱ ≤ 2.5), −d·exp(c2/(1−ᾱ)) beyond.
fn f_interp<S: Scalar>(a: S, poly: &[f64; 8], c1: f64, c2: f64, d: f64) -> S {
    f_interp_bounded(a, poly, [c1, c2, d], [0.0, 2.5], u8::MAX)
}

/// The α-interpolation with the polynomial on `(lower, upper]`, the SCAN
/// branch `exp(-c1 ᾱ/(1-ᾱ))` at or below `lower` and `-d exp(c2/(1-ᾱ))`
/// above `upper`. `[0, 2.5]` is the published switch, which joins the
/// branches with continuous value and first three derivatives (the fourth
/// jumps); `[1, 1]` removes the polynomial and leaves the infinitely
/// differentiable SCAN interpolation with the same branch constants.
///
/// `site` names the interpolation (exchange of spin 0 or 1, correlation 2)
/// for the test-only branch freezing of `branch_freeze`.
fn f_interp_bounded<S: Scalar>(
    a: S,
    poly: &[f64; 8],
    c: [f64; 3],
    bounds: [f64; 2],
    site: u8,
) -> S {
    let [c1, c2, d] = c;
    let av = a.value();
    let one = S::from_f64(1.0);
    let branch: u8 = if av <= bounds[0] {
        0
    } else if av <= bounds[1] {
        1
    } else {
        2
    };
    #[cfg(test)]
    let branch = super::branch_freeze::branch(bounds[1], site, branch);
    #[cfg(not(test))]
    let _ = site;
    if branch == 0 {
        (-(S::from_f64(c1) * a) / (one - a)).exp()
    } else if branch == 1 {
        let mut acc = S::from_f64(poly[7]);
        for k in (0..7).rev() {
            acc = acc * a + S::from_f64(poly[k]);
        }
        acc
    } else {
        -S::from_f64(d) * (S::from_f64(c2) / (one - a)).exp()
    }
}

/// r²SCAN exchange for one spin channel: CX_SPIN·ρσ^{4/3}·F_x.
fn r2scan_x_spin<S: Scalar>(rho: S, sig: S, tau: S) -> S {
    let one = S::from_f64(1.0);
    let r43 = rho.powf(4.0 / 3.0);
    let r53 = rho.powf(5.0 / 3.0);
    let xs2 = sig / (r43 * r43); // xs² = σ/ρ^{8/3}
    let ts = tau / r53;
    let alpha = (ts - xs2 * S::from_f64(0.125))
        / (S::from_f64(k_factor_c()) + S::from_f64(ETA * 0.125) * xs2);
    let p = S::from_f64(x2s() * x2s()) * xs2;
    // C2 = −Σ_{k=0..7} (k+1)c_k · (1 − h0x); Cη = 20/27 + 5η/3 (Eqs. S11-S12).
    let sum: f64 = FX_POLY
        .iter()
        .enumerate()
        .map(|(k, c)| (k + 1) as f64 * c)
        .sum();
    let c2 = -sum * (1.0 - H0X);
    let cn = 20.0 / 27.0 + ETA * 5.0 / 3.0;
    let x = (S::from_f64(cn * c2) * (-(p * p) / S::from_f64(DP2.powi(4))).exp()
        + S::from_f64(MU_GE))
        * p;
    let h1x = one + S::from_f64(K1) - S::from_f64(K1 * K1) * (S::from_f64(K1) + x).recip();
    let fa = f_interp(alpha, &FX_POLY, C1X, C2X, DX);
    // g_x = 1 − exp(−a1/√s); s = X2S·xs. Cut exactly to 1 at vanishing s
    // (the exponential and all its derivatives vanish there).
    let s = S::from_f64(x2s()) * xs2.sqrt();
    let gx = if s.value() < 1e-10 {
        one
    } else {
        one - (-S::from_f64(A1X) / s.sqrt()).exp()
    };
    let fx = (h1x + fa * (S::from_f64(H0X) - h1x)) * gx;
    S::from_f64(CX_SPIN) * r43 * fx
}

/// r²SCAN exchange energy density (per-spin sum).
pub(crate) fn r2scan_x_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    _th: &[S],
) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        e = e + r2scan_x_spin(rho, sig, tau);
    }
    e
}

/// ε_c^{LDA0}(rs) (Eq. S25).
fn eclda0<S: Scalar>(rs: S) -> S {
    let one = S::from_f64(1.0);
    -S::from_f64(B1C) / (one + S::from_f64(B2C) * rs.sqrt() + S::from_f64(B3C) * rs)
}

/// PW92(MOD) ε_c(rs, ζ) with the para/ferro split handled by the caller.
fn pw92_mod_eps<S: Scalar>(rs: S, z: S) -> S {
    let one = S::from_f64(1.0);
    let g = |i: usize| -> [S; 6] {
        [
            S::from_f64(PW92_MOD[6 * i]),
            S::from_f64(PW92_MOD[6 * i + 1]),
            S::from_f64(PW92_MOD[6 * i + 2]),
            S::from_f64(PW92_MOD[6 * i + 3]),
            S::from_f64(PW92_MOD[6 * i + 4]),
            S::from_f64(PW92_MOD[6 * i + 5]),
        ]
    };
    let [a, a1, b1, b2, b3, b4] = g(0);
    let e0 = super::lda::pw92_g(rs, a, a1, b1, b2, b3, b4);
    let [a, a1, b1, b2, b3, b4] = g(1);
    let e1 = super::lda::pw92_g(rs, a, a1, b1, b2, b3, b4);
    let [a, a1, b1, b2, b3, b4] = g(2);
    let neg_ac = super::lda::pw92_g(rs, a, a1, b1, b2, b3, b4);
    let fz = super::common::f_zeta(z);
    let fpp0 = 8.0 / (9.0 * (2.0f64.powf(4.0 / 3.0) - 2.0));
    let z4 = z * z * z * z;
    e0 - neg_ac * fz * S::from_f64(1.0 / fpp0) * (one - z4) + (e1 - e0) * fz * z4
}

fn pw92_mod_eps_ferro<S: Scalar>(rs: S) -> S {
    super::lda::pw92_g(
        rs,
        S::from_f64(PW92_MOD[6]),
        S::from_f64(PW92_MOD[7]),
        S::from_f64(PW92_MOD[8]),
        S::from_f64(PW92_MOD[9]),
        S::from_f64(PW92_MOD[10]),
        S::from_f64(PW92_MOD[11]),
    )
}

/// r²SCAN correlation energy density.
pub(crate) fn r2scan_c_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    _th: &[S],
) -> S {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let have_a = ra.value() >= RHO_CUTOFF;
    let have_b = rb.value() >= RHO_CUTOFF;
    if !have_a && !have_b {
        return zero;
    }
    // ζ-dependent factors: φ, d(ζ), t_total(ζ,1,1), G_c(ζ), plus the ferro
    // closed forms at ζ = 1 (singular second duals otherwise).
    let (rho, sigt, tau, phi, dz, tt11, gc) = if have_a && have_b {
        let rho = ra + rb;
        let sigt = saa + S::from_f64(2.0) * sab + sbb;
        let tau = ta + tb;
        let z = (ra - rb) / rho;
        let opz53 = (one + z).powf(5.0 / 3.0);
        let omz53 = (one - z).powf(5.0 / 3.0);
        let phi = ((one + z).powf(2.0 / 3.0) + (one - z).powf(2.0 / 3.0)) * S::from_f64(0.5);
        let dz = (opz53 + omz53) * S::from_f64(0.5);
        let tt11 = (opz53 + omz53) / S::from_f64(2.0f64.powf(5.0 / 3.0));
        let fz = super::common::f_zeta(z);
        let z12 = z.powi(12);
        let gc = (one - S::from_f64(GC_CNST * (2.0f64.cbrt() - 1.0)) * fz) * (one - z12);
        (rho, sigt, tau, phi, dz, tt11, gc)
    } else {
        let (rho, sigt, tau) = if have_a { (ra, saa, ta) } else { (rb, sbb, tb) };
        (
            rho,
            sigt,
            tau,
            S::from_f64(2.0f64.powf(-1.0 / 3.0)),
            S::from_f64(2.0f64.powf(2.0 / 3.0)),
            one,
            zero,
        )
    };
    let ferro = !(have_a && have_b);
    let rs = super::common::wigner_seitz(rho);
    let r43 = rho.powf(4.0 / 3.0);
    let xt2 = sigt / (r43 * r43);
    // `xt2` is analytic in σ and keeps its exact derivative (it is what
    // `alpha` below depends on). `xt = √xt2` is not: at σ = 0 its derivative
    // is infinite and the zero seed turns it into NaN across every tangent
    // slot. Only `s` and `t` are built from `xt`, and both are exactly zero
    // at σ = 0, so a CONSTANT zero there is the correct value with zero
    // σ-derivatives — the absent-gradient convention, applied only to the
    // non-analytic half so `alpha` keeps its true σ-derivative.
    let xt = if super::gga::sigma_is_absent(sigt) {
        S::from_f64(0.0)
    } else {
        xt2.sqrt()
    };
    // s (total-density reduced gradient) and the PBE t variable.
    let s = S::from_f64(x2s() * 2.0f64.cbrt()) * xt;
    let t = xt / (S::from_f64(4.0 * 2.0f64.cbrt()) * phi * rs.sqrt());
    let alpha = (tau / rho.powf(5.0 / 3.0) - xt2 * S::from_f64(0.125))
        / (S::from_f64(k_factor_c()) * tt11 + S::from_f64(ETA * 0.125) * xt2);

    // LSDA pieces and their rs-derivatives via a nested single-slot dual.
    type DR<S2> = DualN<S2, 1>;
    let rsd: DR<S> = DR::variable(rs, 0);
    let ecl0 = eclda0(rsd);
    let epw = if ferro {
        pw92_mod_eps_ferro(rsd)
    } else {
        // ζ enters as an rs-independent constant here.
        pw92_mod_eps(rsd, DR::constant((ra - rb) / rho))
    };
    let (elsda0, delsda0) = (ecl0.re * gc, ecl0.im[0] * gc);
    let (elsda1, delsda1) = (epw.re, epw.im[0]);

    let gamma = (1.0 - std::f64::consts::LN_2) / std::f64::consts::PI.powi(2);
    let phi3 = phi * phi * phi;
    let w1 = (-elsda1 / (S::from_f64(gamma) * phi3)).exp() - one;
    let beta = S::from_f64(0.066_724_550_603_149_22) * (one + S::from_f64(0.1) * rs)
        / (one + S::from_f64(0.1778) * rs);
    let y = beta * t * t / (S::from_f64(gamma) * w1);
    // Δy (Eq. S34) with Δf_c2 = Σ_{k=1..7} k·c_k of the switching poly.
    let dfc2: f64 = FC_POLY
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, c)| k as f64 * c)
        .sum();
    let s2 = s * s;
    let dy = S::from_f64(dfc2 / (27.0 * gamma)) / (dz * phi3 * w1)
        * (S::from_f64(20.0) * rs * (delsda0 - delsda1)
            - S::from_f64(45.0 * ETA) * (elsda0 - elsda1))
        * s2
        * (-(s2 * s2) / S::from_f64(DP2.powi(4))).exp();
    let g = (one + S::from_f64(4.0) * (y - dy)).powf(-0.25);
    let fh = S::from_f64(gamma) * phi3 * (w1 * (one - g)).ln_1p();
    let ec1 = elsda1 + fh;

    let ginf = (one + S::from_f64(4.0 * CHI_INF) * s2).powf(-0.25);
    let h0 =
        S::from_f64(B1C) * (((-ecl0.re / S::from_f64(B1C)).exp() - one) * (one - ginf)).ln_1p();
    let ec0 = (ecl0.re + h0) * gc;

    let fa = f_interp(alpha, &FC_POLY, C1C, C2C, DC);
    rho * (ec1 + fa * (ec0 - ec1))
}

/// r²SCAN exchange (nonempirical; no adjustable parameters exposed).
pub fn r2scan_x() -> Box<dyn XcFunctional> {
    mgga_driver!("r2scanx", Kind::Exchange, vec![], &[], r2scan_x_e, true)
}

/// r²SCAN correlation.
pub fn r2scan_c() -> Box<dyn XcFunctional> {
    mgga_driver!("r2scanc", Kind::Correlation, vec![], &[], r2scan_c_e, true)
}

// ---------------------------------------------------------------------------
// SCAN (Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)): the original
// iso-orbital indicator alpha = (tau - tau_W)/tau_unif and the exponential
// interpolation f(alpha) = exp(-c1 alpha/(1 - alpha)) for alpha < 1,
// -d exp(c2/(1 - alpha)) for alpha > 1. Both branches and all their
// derivatives vanish at alpha = 1; within 1e-3 of it the interpolation is
// the constant zero (its value there is below 1e-280).
// ---------------------------------------------------------------------------

fn f_scan<S: Scalar>(a: S, c1: f64, c2: f64, d: f64) -> S {
    let one = S::from_f64(1.0);
    let gap = 1.0 - a.value();
    if gap.abs() < 1e-3 {
        S::from_f64(0.0)
    } else if gap > 0.0 {
        (-(S::from_f64(c1) * a) / (one - a)).exp()
    } else {
        -S::from_f64(d) * (S::from_f64(c2) / (one - a)).exp()
    }
}

const SCAN_A1X: f64 = 4.9479;

fn scan_x_spin<S: Scalar>(rho: S, sig: S, tau: S) -> S {
    let r43 = rho.powf(4.0 / 3.0);
    let r53 = rho.powf(5.0 / 3.0);
    let xs2 = sig / (r43 * r43);
    let ts = tau / r53;
    let alpha = (ts - xs2 * S::from_f64(0.125)) / S::from_f64(k_factor_c());
    scan_x_core(rho, xs2, alpha, f_scan(alpha, C1X, C2X, DX))
}

/// rSCAN exchange of one spin channel (Bartok-Yates, JCP 150, 161101
/// (2019)): the SCAN form with `alpha = alpha0^3 / (alpha0^2 + alpha_r)`,
/// `alpha0 = (tau - tau_W) / (K rho^(5/3) + tau_r / 2)` and the polynomial
/// interpolation of r2SCAN. `tau_r` has the dimension of a kinetic-energy
/// density, so it carries the scaling power 5 of `tau`.
fn rscan_x_spin<S: Scalar>(
    rho: S,
    sig: S,
    tau: S,
    tau_r: S,
    alpha_r: S,
    bounds: [f64; 2],
    site: u8,
) -> S {
    let r43 = rho.powf(4.0 / 3.0);
    let r53 = rho.powf(5.0 / 3.0);
    let xs2 = sig / (r43 * r43);
    let tau_w = sig / (S::from_f64(8.0) * rho);
    let alpha0 =
        nonnegative(tau - tau_w) / (S::from_f64(k_factor_c()) * r53 + S::from_f64(0.5) * tau_r);
    let alpha = alpha0 * alpha0 * alpha0 / (alpha0 * alpha0 + alpha_r);
    scan_x_core(
        rho,
        xs2,
        alpha,
        f_interp_bounded(alpha, &FX_POLY, [C1X, C2X, DX], bounds, site),
    )
}

/// `max(x, 0)`: the rSCAN indicator numerator `tau - tau_W` is floored at
/// zero, as in the reference implementation of rSCAN (the regularization
/// `alpha^3 / (alpha^2 + alpha_r)` is defined for `alpha >= 0`). Physical
/// densities never reach the floor; it acts only where `tau < tau_W`.
fn nonnegative<S: Scalar>(x: S) -> S {
    if x.value() < 0.0 {
        S::from_f64(0.0)
    } else {
        x
    }
}

/// rSCAN regularization parameters `[tau_r, alpha_r]` (Bartok-Yates), the
/// switching interval `[switch_lower, switch_upper]` of the polynomial
/// interpolation (published: `[0, 2.5]`; see `f_interp_bounded`) and their
/// uniform-scaling powers: `tau_r` scales like `tau`, the others are
/// dimensionless.
const RSCAN_PARAMS: [f64; 4] = [1e-4, 1e-3, 0.0, 2.5];
const RSCAN_NAMES: &[&str] = &["tau_r", "alpha_r", "switch_lower", "switch_upper"];
const RSCAN_SCALING: &[i32] = &[5, 0, 0, 0];

/// SCAN-type exchange of one channel for a given iso-orbital indicator and
/// interpolation value `fa`.
fn scan_x_core<S: Scalar>(rho: S, xs2: S, alpha: S, fa: S) -> S {
    let one = S::from_f64(1.0);
    let r43 = rho.powf(4.0 / 3.0);
    let p = S::from_f64(x2s() * x2s()) * xs2;
    let b2 = (5913.0_f64 / 405_000.0).sqrt();
    let b1 = 511.0 / 13_500.0 / (2.0 * b2);
    let b3 = 0.5;
    let b4 = MU_GE * MU_GE / K1 - 1606.0 / 18_225.0 - b1 * b1;
    let oma = one - alpha;
    let first = S::from_f64(MU_GE)
        * p
        * (one + S::from_f64(b4 / MU_GE) * p * (-(S::from_f64(b4.abs() / MU_GE) * p)).exp());
    let second =
        S::from_f64(b1) * p + S::from_f64(b2) * oma * (-(S::from_f64(b3) * oma * oma)).exp();
    let x = first + second * second;
    let h1x = one + S::from_f64(K1) - S::from_f64(K1) / (one + x / S::from_f64(K1));
    let s = S::from_f64(x2s()) * xs2.sqrt();
    let gx = if s.value() < 1e-10 {
        one
    } else {
        one - (-S::from_f64(SCAN_A1X) / s.sqrt()).exp()
    };
    let fx = (h1x + fa * (S::from_f64(H0X) - h1x)) * gx;
    S::from_f64(CX_SPIN) * r43 * fx
}

/// SCAN exchange energy density (per-spin sum).
pub(crate) fn scan_x_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    _th: &[S],
) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        e = e + scan_x_spin(rho, sig, tau);
    }
    e
}

/// rSCAN exchange energy density (per-spin sum).
pub(crate) fn rscan_x_e<S: Scalar>(
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
    for (site, (rho, sig, tau)) in [(ra, saa, ta), (rb, sbb, tb)].into_iter().enumerate() {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        e = e + rscan_x_spin(
            rho,
            sig,
            tau,
            th[0],
            th[1],
            [th[2].value(), th[3].value()],
            site as u8,
        );
    }
    e
}

/// SCAN correlation energy density: the r2SCAN correlation without the
/// regularization (eta = 0, no gradient-expansion restoration term) and
/// with the exponential interpolation.
pub(crate) fn scan_c_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    _th: &[S],
) -> S {
    scan_family_c(ra, rb, saa, sab, sbb, ta, tb, None)
}

/// SCAN and rSCAN correlation; `regularization = Some([tau_r, alpha_r])`
/// selects the rSCAN indicator.
#[allow(clippy::too_many_arguments)]
fn scan_family_c<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    regularization: Option<([S; 2], [f64; 2])>,
) -> S {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let have_a = ra.value() >= RHO_CUTOFF;
    let have_b = rb.value() >= RHO_CUTOFF;
    if !have_a && !have_b {
        return zero;
    }
    let (rho, sigt, tau, phi, tt11, gc) = if have_a && have_b {
        let rho = ra + rb;
        let sigt = saa + S::from_f64(2.0) * sab + sbb;
        let tau = ta + tb;
        let z = (ra - rb) / rho;
        let opz53 = (one + z).powf(5.0 / 3.0);
        let omz53 = (one - z).powf(5.0 / 3.0);
        let phi = ((one + z).powf(2.0 / 3.0) + (one - z).powf(2.0 / 3.0)) * S::from_f64(0.5);
        let tt11 = (opz53 + omz53) / S::from_f64(2.0f64.powf(5.0 / 3.0));
        let fz = super::common::f_zeta(z);
        let z12 = z.powi(12);
        let gc = (one - S::from_f64(GC_CNST * (2.0f64.cbrt() - 1.0)) * fz) * (one - z12);
        (rho, sigt, tau, phi, tt11, gc)
    } else {
        let (rho, sigt, tau) = if have_a { (ra, saa, ta) } else { (rb, sbb, tb) };
        (
            rho,
            sigt,
            tau,
            S::from_f64(2.0f64.powf(-1.0 / 3.0)),
            one,
            zero,
        )
    };
    let ferro = !(have_a && have_b);
    let rs = super::common::wigner_seitz(rho);
    let r43 = rho.powf(4.0 / 3.0);
    let xt2 = sigt / (r43 * r43);
    let xt = if super::gga::sigma_is_absent(sigt) {
        S::from_f64(0.0)
    } else {
        xt2.sqrt()
    };
    let s = S::from_f64(x2s() * 2.0f64.cbrt()) * xt;
    let t = xt / (S::from_f64(4.0 * 2.0f64.cbrt()) * phi * rs.sqrt());
    let fa = if let Some(([tau_r, alpha_r], bounds)) = regularization {
        let tau_w = sigt / (S::from_f64(8.0) * rho);
        let r53 = rho.powf(5.0 / 3.0);
        let alpha0 = nonnegative(tau - tau_w)
            / ((S::from_f64(k_factor_c()) * r53 + S::from_f64(2.0f64.powf(2.0 / 3.0)) * tau_r)
                * tt11);
        let alpha = alpha0 * alpha0 * alpha0 / (alpha0 * alpha0 + alpha_r);
        f_interp_bounded(alpha, &FC_POLY, [C1C, C2C, DC], bounds, 2)
    } else {
        let alpha = (tau / rho.powf(5.0 / 3.0) - xt2 * S::from_f64(0.125))
            / (S::from_f64(k_factor_c()) * tt11);
        f_scan(alpha, C1C, C2C, DC)
    };
    let ecl0 = eclda0(rs);
    let epw = if ferro {
        pw92_mod_eps_ferro(rs)
    } else {
        pw92_mod_eps(rs, (ra - rb) / rho)
    };
    let gamma = (1.0 - std::f64::consts::LN_2) / std::f64::consts::PI.powi(2);
    let phi3 = phi * phi * phi;
    let w1 = (-epw / (S::from_f64(gamma) * phi3)).exp() - one;
    let beta = S::from_f64(0.066_724_550_603_149_22) * (one + S::from_f64(0.1) * rs)
        / (one + S::from_f64(0.1778) * rs);
    let y = beta * t * t / (S::from_f64(gamma) * w1);
    let g = (one + S::from_f64(4.0) * y).powf(-0.25);
    let ec1 = epw + S::from_f64(gamma) * phi3 * (w1 * (one - g)).ln_1p();
    let s2 = s * s;
    let ginf = (one + S::from_f64(4.0 * CHI_INF) * s2).powf(-0.25);
    let h0 = S::from_f64(B1C) * (((-ecl0 / S::from_f64(B1C)).exp() - one) * (one - ginf)).ln_1p();
    let ec0 = (ecl0 + h0) * gc;
    rho * (ec1 + fa * (ec0 - ec1))
}

/// rSCAN correlation energy density: SCAN correlation with the
/// regularized indicator and the polynomial interpolation.
pub(crate) fn rscan_c_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    scan_family_c(
        ra,
        rb,
        saa,
        sab,
        sbb,
        ta,
        tb,
        Some(([th[0], th[1]], [th[2].value(), th[3].value()])),
    )
}

/// SCAN exchange (nonempirical; no adjustable parameters exposed).
pub fn scan_x() -> Box<dyn XcFunctional> {
    mgga_driver!("scanx", Kind::Exchange, vec![], &[], scan_x_e, true)
}

/// rSCAN exchange.
pub fn rscan_x() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "rscanx",
        Kind::Exchange,
        RSCAN_PARAMS.to_vec(),
        RSCAN_NAMES,
        rscan_x_e,
        true;
        scaling = RSCAN_SCALING
    )
}

/// rSCAN correlation.
pub fn rscan_c() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "rscanc",
        Kind::Correlation,
        RSCAN_PARAMS.to_vec(),
        RSCAN_NAMES,
        rscan_c_e,
        true;
        scaling = RSCAN_SCALING
    )
}

/// SCAN correlation.
pub fn scan_c() -> Box<dyn XcFunctional> {
    mgga_driver!("scanc", Kind::Correlation, vec![], &[], scan_c_e, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{Family, GridVars, XcOut};

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

    fn eval_ord(f: &dyn XcFunctional, v: [f64; 7], order: usize) -> XcOut {
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
        out.prepare(1, order, Family::MetaGga);
        f.eval_add(&vars, order, 1.0, &mut out).unwrap();
        out
    }

    /// First and second derivatives against central differences (X and C)
    /// at a point where the FHC bound is inactive and ᾱ is mid-polynomial.
    #[test]
    fn derivatives_match_finite_differences() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12];
        let h = 1e-6;
        for f in [r2scan_x(), r2scan_c()] {
            let out = eval_ord(f.as_ref(), p, 2);
            let firsts = [
                (0usize, out.vrho[0][0]),
                (1, out.vrho[1][0]),
                (2, out.vsigma[0][0]),
                (3, out.vsigma[1][0]),
                (4, out.vsigma[2][0]),
                (5, out.vtau[0][0]),
                (6, out.vtau[1][0]),
            ];
            for (k, an) in firsts {
                let mut pp = p;
                pp[k] += h;
                let ep = eval_ord(f.as_ref(), pp, 0).e[0];
                pp[k] -= 2.0 * h;
                let em = eval_ord(f.as_ref(), pp, 0).e[0];
                let fd = (ep - em) / (2.0 * h);
                assert!(
                    (an - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                    "{} var {k}: analytic {an} vs FD {fd}",
                    f.name()
                );
            }
            let d1 = |v: [f64; 7], slot: usize| -> f64 {
                let o = eval_ord(f.as_ref(), v, 1);
                match slot {
                    0 => o.vrho[0][0],
                    2 => o.vsigma[0][0],
                    _ => o.vtau[0][0],
                }
            };
            for (k, slot, an) in [
                (0usize, 0usize, out.v2rho2[0][0]),
                (2, 0, out.v2rhosigma[0][0]),
                (2, 2, out.v2sigma2[0][0]),
                (5, 5, out.v2tau2[0][0]),
                (0, 5, out.v2rhotau[0][0]),
            ] {
                let mut pp = p;
                pp[k] += h;
                let ep = d1(pp, slot);
                pp[k] -= 2.0 * h;
                let em = d1(pp, slot);
                let fd = (ep - em) / (2.0 * h);
                assert!(
                    (an - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                    "{} d2({k},{slot}): analytic {an} vs FD {fd}",
                    f.name()
                );
            }
        }
    }

    /// Pinned pointwise against libxc mgga_x_r2scan / mgga_c_r2scan
    /// (tools/oracle/probe_r2scan.py). Points cover the polynomial ᾱ
    /// branch, ᾱ > 2.5, ᾱ < 0, and the fully polarized channel.
    #[test]
    fn r2scan_matches_libxc_reference_points() {
        let cases: [([f64; 7], f64, f64); 5] = [
            (
                [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12],
                -0.247154924614,
                -0.016556869405,
            ),
            (
                [1.2, 0.9, 0.29, 0.055, 0.1725, 2.2, 1.6],
                -2.229086595856,
                -0.089119935945,
            ),
            (
                [0.5, 0.4, 0.1, 0.05, 0.08, 3.0, 2.5],
                -0.570591330117,
                -0.070354843256,
            ),
            (
                [0.5, 0.4, 0.1, 0.05, 0.08, 0.01, 0.008],
                -0.755499840688,
                -0.022159039454,
            ),
            (
                [0.4, 0.0, 0.09, 0.0, 0.0, 0.5, 0.0],
                -0.301209559293,
                -0.005731500836,
            ),
        ];
        let fx = r2scan_x();
        let fc = r2scan_c();
        for (v, want_x, want_c) in cases {
            let ex = eval(fx.as_ref(), v);
            let ec = eval(fc.as_ref(), v);
            assert!(
                (ex - want_x).abs() < 1e-9,
                "r2scan-x {ex} vs libxc {want_x} at {v:?}"
            );
            assert!(
                (ec - want_c).abs() < 1e-9,
                "r2scan-c {ec} vs libxc {want_c} at {v:?}"
            );
        }
    }
}
