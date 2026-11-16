// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The B97 power-series functional family (Becke, JCP 107, 8554 (1997)) and
//! its HCTH reparameterizations (Hamprecht-Cohen-Tozer-Handy, JCP 109, 6264
//! (1998); Boese et al. for /120, /147, /407).
//!
//! e = sum_s ex_LSDA(rho_s)*g_x(u_xs) + sum_s ec_ss*g_ss(u_s)
//!     + ec_ab*g_ab(u_avg),
//! with u = gamma*x^2/(1+gamma*x^2), g(u) = sum_k c_k u^k, and the
//! same-spin / opposite-spin LSDA correlation split
//! Ec_ss = Ec_PW92(rho_s, 0), Ec_ab = Ec_PW92(ra, rb) - sum_s Ec_PW92(rho_s, 0).
//!
//! Every gamma and series coefficient is an exposed, differentiable
//! parameter (layout [g_x, g_ss, g_ab, cx..., css..., cab...]; the term
//! count follows from the parameter length) — B97-style coefficient refits
//! are `with_params` calls.

use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

/// Static parameter-name pool sliced to the actual parameter count
/// (3 gammas + up to 5 coefficients per channel).
static B97_PARAM_NAMES: [&str; 18] = [
    "gamma_x", "gamma_ss", "gamma_ab", "cx_0", "cx_1", "cx_2", "cx_3", "cx_4", "css_0", "css_1",
    "css_2", "css_3", "css_4", "cab_0", "cab_1", "cab_2", "cab_3", "cab_4",
];

/// The name pool is ordered for the 5-term layout; for other counts the
/// names are positional only (parameter IDENTITY is by index).
fn boxed(name: &'static str, kind: Kind, params: Vec<f64>) -> Box<dyn XcFunctional> {
    let names = &B97_PARAM_NAMES[..params.len()];
    gga_driver!(name, kind, params, names, b97_e)
}

use super::common::RHO_CUTOFF;

/// Per-spin LSDA exchange energy density coefficient.
const CX_SPIN: f64 = -super::common::X_FACTOR_C;

fn series<S: Scalar>(u: S, coefs: &[S]) -> S {
    let mut acc = S::from_f64(0.0);
    let mut upow = S::from_f64(1.0);
    for &c in coefs {
        acc = acc + c * upow;
        upow = upow * u;
    }
    acc
}

/// The generic B97-family energy density.
pub(crate) fn b97_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    b97_with_spin_normalization::<S, false>(ra, rb, saa, sbb, th)
}

fn b97_original_pw_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    b97_with_spin_normalization::<S, true>(ra, rb, saa, sbb, th)
}

fn b97_with_spin_normalization<S: Scalar, const ORIGINAL: bool>(
    ra: S,
    rb: S,
    saa: S,
    sbb: S,
    th: &[S],
) -> S {
    let n = (th.len() - 3) / 3;
    let (gx, gss, gab) = (th[0], th[1], th[2]);
    let cx = &th[3..3 + n];
    let css = &th[3 + n..3 + 2 * n];
    let cab = &th[3 + 2 * n..3 + 3 * n];
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let mut e = zero;
    let mut x2 = [zero, zero];
    let mut have = [false, false];
    for (idx, (rho, sig)) in [(ra, saa), (rb, sbb)].into_iter().enumerate() {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        have[idx] = true;
        let r43 = rho.powf(4.0 / 3.0);
        let xs2 = sig / (r43 * r43);
        x2[idx] = xs2;
        // Exchange.
        let ux = gx * xs2 / (one + gx * xs2);
        e = e + S::from_f64(CX_SPIN) * r43 * series(ux, cx);
        // Same-spin correlation on the (rho_s, 0) PW92 branch.
        let ec_ss = rho * super::lda::pw92_epsilon_ferro(rho, &pw_mod_params::<S>());
        let uss = gss * xs2 / (one + gss * xs2);
        e = e + ec_ss * series(uss, css);
    }
    // Opposite-spin correlation.
    if have[0] && have[1] {
        let ra_s = if have[0] { ra } else { zero };
        let rb_s = if have[1] { rb } else { zero };
        let epsilon = if ORIGINAL {
            super::lda::pw92_epsilon_original(ra_s, rb_s, &pw_mod_params::<S>())
        } else {
            super::lda::pw92_epsilon(ra_s, rb_s, &pw_mod_params::<S>())
        };
        let mut ec_ab = (ra_s + rb_s) * epsilon;
        if have[0] {
            ec_ab = ec_ab - ra_s * super::lda::pw92_epsilon_ferro(ra_s, &pw_mod_params::<S>());
        }
        if have[1] {
            ec_ab = ec_ab - rb_s * super::lda::pw92_epsilon_ferro(rb_s, &pw_mod_params::<S>());
        }
        let x_avg2 = (x2[0] + x2[1]) * S::from_f64(0.5);
        let uab = gab * x_avg2 / (one + gab * x_avg2);
        e = e + ec_ab * series(uab, cab);
    }
    e
}

fn pw_mod_params<S: Scalar>() -> Vec<S> {
    // The B97 family builds on PW92 with the PAPER digits (unlike PBE-C,
    // which uses the MOD digits) — pinned pointwise against libxc.
    super::lda::PW92_PARAMS
        .iter()
        .map(|&v| S::from_f64(v))
        .collect()
}

/// The ωB97 short-range variant: the LSDA exchange factor is attenuated by
/// the SR hole factor F(ω/(2kFσ)), kFσ = (6π²ρσ)^{1/3}; the correlation
/// channels are plain B97. θ layout: [γx, γss, γab, ω, cx…, css…, cab…].
pub(crate) fn wb97_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let n = (th.len() - 4) / 3;
    let (gx, gss, gab, omega) = (th[0], th[1], th[2], th[3]);
    let cx = &th[4..4 + n];
    let css = &th[4 + n..4 + 2 * n];
    let cab = &th[4 + 2 * n..4 + 3 * n];
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let mut e = zero;
    let mut x2 = [zero, zero];
    let mut have = [false, false];
    for (idx, (rho, sig)) in [(ra, saa), (rb, sbb)].into_iter().enumerate() {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        have[idx] = true;
        let r43 = rho.powf(4.0 / 3.0);
        let xs2 = sig / (r43 * r43);
        x2[idx] = xs2;
        // SR-attenuated LSDA exchange × series.
        let kf =
            (S::from_f64(6.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).powf(1.0 / 3.0);
        let a = omega / (S::from_f64(2.0) * kf);
        let fsr = super::gga::sr_factor(a);
        let ux = gx * xs2 / (one + gx * xs2);
        e = e + S::from_f64(CX_SPIN) * r43 * fsr * series(ux, cx);
        let ec_ss = rho * super::lda::pw92_epsilon_ferro(rho, &pw_mod_params::<S>());
        let uss = gss * xs2 / (one + gss * xs2);
        e = e + ec_ss * series(uss, css);
    }
    if have[0] && have[1] {
        let ra_s = if have[0] { ra } else { zero };
        let rb_s = if have[1] { rb } else { zero };
        let mut ec_ab = (ra_s + rb_s) * super::lda::pw92_epsilon(ra_s, rb_s, &pw_mod_params::<S>());
        if have[0] {
            ec_ab = ec_ab - ra_s * super::lda::pw92_epsilon_ferro(ra_s, &pw_mod_params::<S>());
        }
        if have[1] {
            ec_ab = ec_ab - rb_s * super::lda::pw92_epsilon_ferro(rb_s, &pw_mod_params::<S>());
        }
        let x_avg2 = (x2[0] + x2[1]) * S::from_f64(0.5);
        let uab = gab * x_avg2 / (one + gab * x_avg2);
        e = e + ec_ab * series(uab, cab);
    }
    e
}

static WB97_PARAM_NAMES: [&str; 19] = [
    "gamma_x", "gamma_ss", "gamma_ab", "omega", "cx_0", "cx_1", "cx_2", "cx_3", "cx_4", "css_0",
    "css_1", "css_2", "css_3", "css_4", "cab_0", "cab_1", "cab_2", "cab_3", "cab_4",
];
static WB97_CUBIC_PARAM_NAMES: [&str; 16] = [
    "gamma_x", "gamma_ss", "gamma_ab", "omega", "cx_0", "cx_1", "cx_2", "cx_3", "css_0", "css_1",
    "css_2", "css_3", "cab_0", "cab_1", "cab_2", "cab_3",
];
static WB97_QUADRATIC_PARAM_NAMES: [&str; 13] = [
    "gamma_x", "gamma_ss", "gamma_ab", "omega", "cx_0", "cx_1", "cx_2", "css_0", "css_1", "css_2",
    "cab_0", "cab_1", "cab_2",
];

pub(super) fn wb97_boxed(name: &'static str, params: Vec<f64>) -> Box<dyn XcFunctional> {
    let names: &[_] = match params.len() {
        19 => &WB97_PARAM_NAMES,
        16 => &WB97_CUBIC_PARAM_NAMES,
        13 => &WB97_QUADRATIC_PARAM_NAMES,
        _ => unreachable!("unsupported number of B97 polynomial coefficients"),
    };
    gga_driver!(name, Kind::ExchangeCorrelation, params, names, wb97_e)
}

/// Semilocal parent of the ωB97X-2 double hybrids, with equal-degree
/// exchange, same-spin correlation, and opposite-spin correlation fits.
pub(super) fn wb97x2(
    name: &'static str,
    omega: f64,
    [x, ss, os]: [[f64; 4]; 3],
) -> Box<dyn XcFunctional> {
    wb97_boxed(name, wb97_params(omega, &x, &ss, &os))
}

pub(super) fn wb97_params(omega: f64, cx: &[f64], css: &[f64], cab: &[f64]) -> Vec<f64> {
    let mut p = vec![GAMMAS[0], GAMMAS[1], GAMMAS[2], omega];
    p.extend_from_slice(cx);
    p.extend_from_slice(css);
    p.extend_from_slice(cab);
    p
}

/// ωB97 (Chai–Head-Gordon 2008): ω = 0.4, no SR-HF (cx = 0, clr = 1).
pub fn wb97() -> Box<dyn XcFunctional> {
    wb97_boxed(
        "wb97",
        wb97_params(
            0.4,
            &[1.0, 1.13116, -2.74915, 12.09, -5.71642],
            &[1.0, -2.55352, 11.8926, -26.9452, 17.0927],
            &[1.0, 3.99051, -17.0066, 1.07292, 8.88211],
        ),
    )
}

/// ωB97X: ω = 0.3, SR-HF cx = 0.157706.
pub fn wb97x() -> Box<dyn XcFunctional> {
    wb97_boxed(
        "wb97x",
        wb97_params(
            0.3,
            &[0.842294, 0.726479, 1.04760, -5.70635, 13.2794],
            &[1.0, -4.33879, 18.2308, -31.7430, 17.2901],
            &[1.0, 2.37031, -11.3995, 6.58405, -3.78132],
        ),
    )
}

/// ωB97X-D (semilocal part; ω = 0.2, SR-HF cx = 0.222036; the CHG
/// dispersion tail is a separate `--disp` choice).
pub fn wb97x_d() -> Box<dyn XcFunctional> {
    wb97_boxed(
        "wb97x_d",
        wb97_params(
            0.2,
            &[0.777964, 0.661160, 0.574541, -5.25671, 11.6386],
            &[1.0, -6.90539, 31.3343, -51.0533, 26.4423],
            &[1.0, 1.79413, -12.0477, 14.0847, -8.50809],
        ),
    )
}

/// ωB97X-V semilocal part (ω = 0.3, SR-HF cx = 0.167; the VV10 nonlocal
/// term completes the functional).
pub fn wb97x_v() -> Box<dyn XcFunctional> {
    wb97_boxed(
        "wb97x_v",
        wb97_params(
            0.3,
            &[0.833, 0.603, 1.194],
            &[0.556, -0.257, 0.0],
            &[1.219, -1.850, 0.0],
        ),
    )
}

/// ωB97M-V semilocal part (Mardirossian–Head-Gordon, JCP 144, 214110
/// (2016)): a meta-GGA ωB97 with sparse 2D power series in (w, u):
///   e_x  = e_x^SR-LSDA(ρσ; ω) Σ_k c_k w^i u_x^j          (per spin),
///   e_ss = e_c^UEG(ρσ,0) Σ_k c_k w^i u_ss^j              (per spin),
///   e_ab = Δe_c^UEG Σ_k c_k w̄^i u_ab^j,
/// with w = (τ^LSDA − τ)/(τ^LSDA + τ), u = γx²/(1 + γx²),
/// x̄² = ½(x_α² + x_β²), t̄ = ½(t_α + t_β), w̄ = (t̄ − 1)/(t̄ + 1), and
/// the PW92 "modified"-digit UEG correlation base. The sparse (i, j)
/// term pattern and every coefficient were recovered from libxc by
/// least squares (residual ≤ 1.1e-14 over 800 random points).
/// θ layout: [γx, γss, γab, ω, cx×3, css×5, cab×6].
const WB97M_X_TERMS: [(i32, i32); 3] = [(0, 0), (0, 1), (1, 0)];
const WB97M_SS_TERMS: [(i32, i32); 5] = [(0, 0), (0, 4), (1, 0), (2, 0), (4, 3)];
const WB97M_AB_TERMS: [(i32, i32); 6] = [(0, 0), (1, 0), (2, 0), (2, 1), (6, 0), (6, 1)];

fn series_2d<S: Scalar>(w: S, u: S, terms: &[(i32, i32)], coefs: &[S]) -> S {
    let mut acc = S::from_f64(0.0);
    for (&(i, j), &c) in terms.iter().zip(coefs) {
        let mut t = c;
        for _ in 0..i {
            t = t * w;
        }
        for _ in 0..j {
            t = t * u;
        }
        acc = acc + t;
    }
    acc
}

pub(crate) fn wb97m_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    wb97m_sparse_e(
        ra,
        rb,
        saa,
        _sab,
        sbb,
        ta,
        tb,
        th,
        &WB97M_X_TERMS,
        &WB97M_SS_TERMS,
        &WB97M_AB_TERMS,
    )
}

fn wb97m_sparse_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
    x_terms: &[(i32, i32)],
    ss_terms: &[(i32, i32)],
    ab_terms: &[(i32, i32)],
) -> S {
    let (gx, gss, gab, omega) = (th[0], th[1], th[2], th[3]);
    let split = 4 + x_terms.len();
    let split2 = split + ss_terms.len();
    let cx = &th[4..split];
    let css = &th[split..split2];
    let cab = &th[split2..];
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let pw = super::m06::pw_mod_params::<S>();
    let mut e = zero;
    let mut x2s = [zero, zero];
    let mut ts = [zero, zero];
    let mut have = [false, false];
    for (idx, (rho, sig, tau)) in [(ra, saa, ta), (rb, sbb, tb)].into_iter().enumerate() {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        have[idx] = true;
        let r53 = rho.powf(5.0 / 3.0);
        let tlsda =
            S::from_f64(0.3 * (6.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0))
                * r53;
        let tau_c = if tau.value() < 1e-20 {
            S::from_f64(1e-20)
        } else {
            tau
        };
        let w = (tlsda - tau_c) / (tlsda + tau_c);
        ts[idx] = tlsda / tau_c;
        let x2 = sig / (r53 * rho);
        x2s[idx] = x2;
        // SR-attenuated LSDA exchange × sparse series.
        let kf =
            (S::from_f64(6.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).powf(1.0 / 3.0);
        let fsr = super::gga::sr_factor(omega / (S::from_f64(2.0) * kf));
        let ux = gx * x2 / (one + gx * x2);
        e = e + S::from_f64(CX_SPIN) * rho.powf(4.0 / 3.0) * fsr * series_2d(w, ux, x_terms, cx);
        // Same-spin correlation.
        let ec_ss = rho * super::lda::pw92_epsilon_ferro(rho, &pw);
        let uss = gss * x2 / (one + gss * x2);
        e = e + ec_ss * series_2d(w, uss, ss_terms, css);
    }
    // Opposite-spin correlation on the averaged variables.
    if have[0] && have[1] {
        let ra_s = if have[0] { ra } else { zero };
        let rb_s = if have[1] { rb } else { zero };
        let mut ec_ab = (ra_s + rb_s) * super::lda::pw92_epsilon(ra_s, rb_s, &pw);
        if have[0] {
            ec_ab = ec_ab - ra_s * super::lda::pw92_epsilon_ferro(ra_s, &pw);
        }
        if have[1] {
            ec_ab = ec_ab - rb_s * super::lda::pw92_epsilon_ferro(rb_s, &pw);
        }
        let half = S::from_f64(0.5);
        let xbar2 = (x2s[0] + x2s[1]) * half;
        let tbar = (ts[0] + ts[1]) * half;
        let wbar = (tbar - one) / (tbar + one);
        let uab = gab * xbar2 / (one + gab * xbar2);
        e = e + ec_ab * series_2d(wbar, uab, ab_terms, cab);
    }
    e
}

fn wb97m2_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    // Table II, JCP 148, 241736 (2018); powers are (w, u).
    wb97m_sparse_e(
        ra,
        rb,
        saa,
        sab,
        sbb,
        ta,
        tb,
        th,
        &[(0, 0), (2, 0), (3, 0), (0, 1), (4, 1)],
        &[(0, 0), (1, 0), (2, 0)],
        &[(0, 0), (2, 0), (0, 1), (0, 2), (2, 2)],
    )
}

/// Individual semilocal target terms of omegaB97M(2), evaluated on the
/// distinct omegaB97M-V orbital reference. Exact exchange, VV10 and PT2
/// are separate terms in the double-hybrid composition.
pub fn wb97m2_parts() -> Vec<(f64, std::sync::Arc<dyn XcFunctional>)> {
    let coefficients = [
        0.37806, 0.28193, -0.21886, 0.13642, 0.70767, 0.54846, -1.17724, -3.67267, 0.46152,
        2.30490, -1.94794, 3.24910, -2.26280,
    ];
    [
        ("wb97m2_x", Kind::Exchange, 0..5),
        ("wb97m2_ss", Kind::Correlation, 5..8),
        ("wb97m2_os", Kind::Correlation, 8..13),
    ]
    .into_iter()
    .map(|(name, kind, active)| {
        let mut params = vec![0.004, 0.2, 0.006, 0.3];
        params.extend(coefficients.iter().enumerate().map(|(i, c)| {
            if active.contains(&i) {
                *c
            } else {
                0.0
            }
        }));
        let boxed: Box<dyn XcFunctional> = mgga_driver!(
            name,
            kind,
            params,
            &[
                "gamma_x", "gamma_ss", "gamma_os", "omega", "cx_00", "cx_20", "cx_30", "cx_01",
                "cx_41", "css_00", "css_10", "css_20", "cos_00", "cos_20", "cos_01", "cos_02",
                "cos_22",
            ],
            wb97m2_e,
            false
        );
        let part: std::sync::Arc<dyn XcFunctional> = std::sync::Arc::from(boxed);
        (1.0, part)
    })
    .collect()
}

/// B97M-V semilocal part (Mardirossian-Head-Gordon, JCP 142, 074111
/// (2015)): the same sparse 2D (w, u) machinery as wB97M-V but with plain
/// LSDA exchange (no SR attenuation) and its own selected terms. All
/// coefficients recovered from libxc by least squares (residual 6.6e-14).
/// theta layout: [gx, gss, gab, cx x5, css x5, cab x5].
const B97M_X_TERMS: [(i32, i32); 5] = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)];
const B97M_SS_TERMS: [(i32, i32); 5] = [(0, 0), (0, 2), (1, 0), (3, 2), (4, 2)];
const B97M_AB_TERMS: [(i32, i32); 5] = [(0, 0), (0, 1), (0, 3), (1, 0), (3, 2)];

pub(crate) fn b97m_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    let (gx, gss, gab) = (th[0], th[1], th[2]);
    let cx = &th[3..8];
    let css = &th[8..13];
    let cab = &th[13..18];
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let pw = super::m06::pw_mod_params::<S>();
    let mut e = zero;
    let mut x2s = [zero, zero];
    let mut ts = [zero, zero];
    let mut have = [false, false];
    for (idx, (rho, sig, tau)) in [(ra, saa, ta), (rb, sbb, tb)].into_iter().enumerate() {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        have[idx] = true;
        let r53 = rho.powf(5.0 / 3.0);
        let tlsda =
            S::from_f64(0.3 * (6.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0))
                * r53;
        let tau_c = if tau.value() < 1e-20 {
            S::from_f64(1e-20)
        } else {
            tau
        };
        let w = (tlsda - tau_c) / (tlsda + tau_c);
        ts[idx] = tlsda / tau_c;
        let x2 = sig / (r53 * rho);
        x2s[idx] = x2;
        let ux = gx * x2 / (one + gx * x2);
        e = e + S::from_f64(CX_SPIN) * rho.powf(4.0 / 3.0) * series_2d(w, ux, &B97M_X_TERMS, cx);
        let ec_ss = rho * super::lda::pw92_epsilon_ferro(rho, &pw);
        let uss = gss * x2 / (one + gss * x2);
        e = e + ec_ss * series_2d(w, uss, &B97M_SS_TERMS, css);
    }
    if have[0] && have[1] {
        let mut ec_ab = (ra + rb) * super::lda::pw92_epsilon(ra, rb, &pw);
        ec_ab = ec_ab - ra * super::lda::pw92_epsilon_ferro(ra, &pw);
        ec_ab = ec_ab - rb * super::lda::pw92_epsilon_ferro(rb, &pw);
        let half = S::from_f64(0.5);
        let xbar2 = (x2s[0] + x2s[1]) * half;
        let tbar = (ts[0] + ts[1]) * half;
        let wbar = (tbar - one) / (tbar + one);
        let uab = gab * xbar2 / (one + gab * xbar2);
        e = e + ec_ab * series_2d(wbar, uab, &B97M_AB_TERMS, cab);
    }
    e
}

static B97M_PARAM_NAMES: [&str; 18] = [
    "gamma_x", "gamma_ss", "gamma_ab", "cx_00", "cx_01", "cx_02", "cx_10", "cx_11", "css_00",
    "css_02", "css_10", "css_32", "css_42", "cab_00", "cab_01", "cab_03", "cab_10", "cab_32",
];

/// B97M-V semilocal part (pure meta-GGA; the VV10 NLC term with b = 6.0,
/// C = 0.01 completes the functional).
pub fn b97m_v() -> Box<dyn XcFunctional> {
    let params = vec![
        0.004, 0.2, 0.006, // gammas
        1.000, 1.308, 1.901, 0.416, 3.070, // x
        1.000, -1.855, -5.668, -20.497, -20.364, // ss
        1.000, 1.573, -6.298, 2.535, -6.427, // ab
    ];
    mgga_driver!(
        "b97m_v",
        Kind::ExchangeCorrelation,
        params,
        &B97M_PARAM_NAMES,
        b97m_e,
        false
    )
}

static WB97M_PARAM_NAMES: [&str; 18] = [
    "gamma_x", "gamma_ss", "gamma_ab", "omega", "cx_00", "cx_01", "cx_10", "css_00", "css_04",
    "css_10", "css_20", "css_43", "cab_00", "cab_10", "cab_20", "cab_21", "cab_60", "cab_61",
];

/// ωB97M-V semilocal part (ω = 0.3, SR-HF cx = 0.15; the VV10 NLC term
/// with b = 6.0, C = 0.01 completes the functional).
pub fn wb97m_v() -> Box<dyn XcFunctional> {
    let params = vec![
        0.004, 0.2, 0.006, 0.3, // γx, γss, γab, ω
        0.85, 1.007, 0.259, // exchange (0,0) (0,1) (1,0)
        0.443, -1.437, -4.535, -3.390, 4.278, // ss (0,0) (0,4) (1,0) (2,0) (4,3)
        1.000, 1.358, 2.924, -8.812, -1.390, 9.142, // ab (0,0) (1,0) (2,0) (2,1) (6,0) (6,1)
    ];
    mgga_driver!(
        "wb97m_v",
        Kind::ExchangeCorrelation,
        params,
        &WB97M_PARAM_NAMES,
        wb97m_e,
        false
    )
}

fn family_params(gammas: [f64; 3], cx: &[f64], css: &[f64], cab: &[f64]) -> Vec<f64> {
    let mut p = gammas.to_vec();
    p.extend_from_slice(cx);
    p.extend_from_slice(css);
    p.extend_from_slice(cab);
    p
}

const GAMMAS: [f64; 3] = [0.004, 0.2, 0.006];

pub fn b97() -> Box<dyn XcFunctional> {
    boxed(
        "b97",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[0.8094, 0.5073, 0.7481],
            &[0.1737, 2.3487, -2.4868],
            &[0.9454, 0.7471, -4.5961],
        ),
    )
}

/// Schmider-Becke 1998 fit2c (B98); fitted local terms are not rescaled.
pub fn b98() -> Box<dyn XcFunctional> {
    let params = family_params(
        GAMMAS,
        &[0.790194, 0.400271, 0.832857, 0.0, 0.0],
        &[-0.120163, 2.82332, -2.59412, 0.0, 0.0],
        &[0.934715, 1.14105, -5.33398, 0.0, 0.0],
    );
    let names = &B97_PARAM_NAMES[..params.len()];
    gga_driver!(
        "b98",
        Kind::ExchangeCorrelation,
        params,
        names,
        b97_original_pw_e
    )
}

/// B97-1 (Hamprecht-Cohen-Tozer-Handy 1998), exact exchange 0.21.
pub fn b97_1() -> Box<dyn XcFunctional> {
    boxed(
        "b97_1",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[0.789518, 0.573805, 0.660975],
            &[0.0820011, 2.71681, -2.87103],
            &[0.955689, 0.788552, -5.47869],
        ),
    )
}

/// B97-2 (Wilson-Bradley-Tozer 2001), exact exchange 0.21.
pub fn b97_2() -> Box<dyn XcFunctional> {
    boxed(
        "b97_2",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[0.827642, 0.04784, 1.76125],
            &[0.585808, -0.691682, 0.394796],
            &[0.999849, 1.40626, -7.4406],
        ),
    )
}

/// B97-3 (Keal-Tozer 2005), exact exchange 0.269288.
pub fn b97_3() -> Box<dyn XcFunctional> {
    boxed(
        "b97_3",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[0.7334648, 0.292527, 3.338789, -10.51158, 10.60907],
            &[0.5623649, -1.32298, 6.359191, -7.464002, 1.827082],
            &[1.13383, -2.811967, 7.431302, -1.969342, -11.74423],
        ),
    )
}

/// B97-K (Boese-Martin 2004, kinetics), exact exchange 0.42.
pub fn b97_k() -> Box<dyn XcFunctional> {
    boxed(
        "b97_k",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[0.507863, 1.46873, -1.51301],
            &[0.12355, 2.65399, -3.20694],
            &[1.58613, -6.20977, 6.46106],
        ),
    )
}

/// N12 correlation (Peverati-Truhlar 2012): the B97 same-spin and
/// opposite-spin series on the PW92 split without an exchange series.
pub fn n12_c() -> Box<dyn XcFunctional> {
    boxed(
        "n12c",
        Kind::Correlation,
        family_params(
            GAMMAS,
            &[0.0; 5],
            &[1.0, -5.5317, 30.7958, -56.4196, 32.125],
            &[1.0, 3.24511, -25.2893, 14.4407, 19.687],
        ),
    )
}

/// N12-SX correlation, PCCP 14, 16187 (2012), using the same PW92 split.
pub fn n12_sx_c() -> Box<dyn XcFunctional> {
    original_pw_c(
        "n12-sxc",
        &[2.63373, -1.05450, -0.729853, 4.94024, -7.31760],
        &[0.833615, 3.24128, -10.6407, -16.0471, 25.1047],
    )
}

/// Shared correlation refits with the original rounded spin normalization.
fn original_pw_c(
    name: &'static str,
    same: &[f64; 5],
    opposite: &[f64; 5],
) -> Box<dyn XcFunctional> {
    let params = family_params(GAMMAS, &[0.0; 5], same, opposite);
    let names = &B97_PARAM_NAMES[..params.len()];
    // This fitted correlation uses the original PW92 normalization. Keep the
    // previously accepted B97 variants' precise interpolation unchanged.
    gga_driver!(name, Kind::Correlation, params, names, b97_original_pw_e)
}

pub fn thcth_c() -> Box<dyn XcFunctional> {
    original_pw_c(
        "thcthc",
        &[0.41385, -0.9086, -0.0549, 1.7480, 0.0],
        &[0.65262, 6.3638, -14.080, -3.3755, 0.0],
    )
}
pub fn thcth_hybrid_c() -> Box<dyn XcFunctional> {
    original_pw_c(
        "thcthhybc",
        &[0.18600, 3.9782, -7.0694, 3.4747, 0.0],
        &[0.80490, 3.8388, -13.547, 3.9133, 0.0],
    )
}

/// Boese-Martin correlation, JCP 121, 3405 (2004), Eqs. 9--20/Table II.
/// Uses the published PW92 spin normalization and the Stoll spin split.
pub fn bmk_c() -> Box<dyn XcFunctional> {
    original_pw_c(
        "bmkc",
        &[-2.19098, 23.8939, -44.3303, 22.5982, 0.0],
        &[1.22334, -3.4631, 10.0731, -11.1974, 0.0],
    )
}

/// GAM correlation (Yu-Zhang-Truhlar 2015): same form as N12.
pub fn gam_c() -> Box<dyn XcFunctional> {
    boxed(
        "gamc",
        Kind::Correlation,
        family_params(
            GAMMAS,
            &[0.0; 5],
            &[0.231765, 0.575592, -3.43391, -5.77281, 9.52448],
            &[0.860548, -2.94135, 15.4176, -5.99825, -23.4119],
        ),
    )
}

/// B97-D (Grimme, J. Comput. Chem. 27, 1787 (2006)): the B97 series
/// refitted without exact exchange, for use with a dispersion correction.
pub fn b97_d() -> Box<dyn XcFunctional> {
    boxed(
        "b97d",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[1.08662, -0.52127, 3.25429],
            &[0.2234, -1.56208, 1.94293],
            &[0.69041, 6.3027, -14.9712],
        ),
    )
}

pub fn hcth93() -> Box<dyn XcFunctional> {
    boxed(
        "hcth93",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[1.09320, -0.744056, 5.59920, -6.78549, 4.49357],
            &[0.222601, -0.0338622, -0.0125170, -0.802496, 1.55396],
            &[0.729974, 3.35287, -11.5430, 8.08564, -4.47857],
        ),
    )
}

pub fn hcth120() -> Box<dyn XcFunctional> {
    boxed(
        "hcth120",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[1.09163, -0.747215, 5.07833, -4.10746, 1.17173],
            &[0.489508, -0.260699, 0.432917, -1.99247, 2.48531],
            &[0.514730, 6.92982, -24.7073, 23.1098, -11.3234],
        ),
    )
}

pub fn hcth147() -> Box<dyn XcFunctional> {
    boxed(
        "hcth147",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[1.09025, -0.799194, 5.57212, -5.86760, 3.04544],
            &[0.562576, 0.0171436, -1.30636, 1.05747, 0.885429],
            &[0.542352, 7.01464, -28.3822, 35.0329, -20.4284],
        ),
    )
}

pub fn hcth407() -> Box<dyn XcFunctional> {
    boxed(
        "hcth407",
        Kind::ExchangeCorrelation,
        family_params(
            GAMMAS,
            &[1.08184, -0.518339, 3.42562, -2.62901, 2.28855],
            &[1.18777, -2.40292, 5.61741, -9.17923, 6.24798],
            &[0.589076, 4.42374, -19.2218, 42.5721, -42.0052],
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{Family, GridVars, XcOut};

    fn eval_pt(f: &dyn XcFunctional, vars5: [f64; 5]) -> f64 {
        let rho_a = [vars5[0]];
        let rho_b = [vars5[1]];
        let saa = [vars5[2]];
        let sab = [vars5[3]];
        let sbb = [vars5[4]];
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: Some([&saa, &sab, &sbb]),
            tau: None,
        };
        let mut out = XcOut::default();
        out.prepare(1, 0, Family::Gga);
        f.eval_add(&vars, 0, 1.0, &mut out).unwrap();
        out.e[0]
    }

    #[test]
    fn wb97x2_cubic_parent_matches_independent_energy_terms() {
        use crate::xc::double_hybrid::DoubleHybridFunctional;

        // Exchange from direct radial quadrature of the erfc-screened
        // uniform-gas exchange hole, correlation from the PW92 and Becke
        // spin formulas. Array order is short-range exchange, SS, OS.
        let cases = [
            (
                [0.27, 0.12, 0.78, 0.19, 0.25],
                [
                    [
                        -0.058_954_848_898_338,
                        -0.000_598_065_098_378,
                        -0.006_120_100_058_911,
                    ],
                    [
                        -0.065_451_549_345_124,
                        -0.000_232_600_873_760,
                        -0.006_036_682_258_243,
                    ],
                ],
            ),
            (
                [0.42, 0.0, 0.21, 0.0, 0.0],
                [
                    [-0.060_274_567_163_607, -0.008_616_557_698_758, 0.0],
                    [-0.073_828_862_314_379, -0.006_901_856_364_020, 0.0],
                ],
            ),
            (
                [0.44, 0.31, 0.0, 0.0, 0.0],
                [
                    [
                        -0.098_460_986_262_633,
                        -0.029_027_524_203_945,
                        -0.014_398_518_720_602,
                    ],
                    [
                        -0.123_386_847_953_939,
                        -0.022_792_394_542_962,
                        -0.013_486_010_407_346,
                    ],
                ],
            ),
        ];
        for (index, name) in ["wb97x-2-lp", "wb97x-2-tqz"].into_iter().enumerate() {
            let recipe = DoubleHybridFunctional::named(name).unwrap();
            let cx = [0.678792, 0.636158][index];
            let ss = [0.581569, 0.529319][index];
            let os = [0.477992, 0.447105][index];
            assert_eq!(recipe.parent.hybrid.cx, cx);
            assert_eq!(recipe.parent.hybrid.clr, 1.0 - cx);
            assert_eq!(recipe.parent.hybrid.omega, 0.3);
            assert_eq!(recipe.correlation.same_spin, ss);
            assert_eq!(recipe.correlation.opposite_spin, os);
            assert!(recipe.target.is_none());
            assert!(recipe.dispersion.is_none());
            let part = &recipe.parent.parts[0].1;
            let groups = [4..8, 8..12, 12..16];
            for (density, expected) in cases {
                let total = eval_pt(part.as_ref(), density);
                assert!((total - expected[index].iter().sum::<f64>()).abs() < 2e-12);
                for (range, want) in groups.iter().zip(expected[index]) {
                    let mut isolated = part.params();
                    for (j, coef) in isolated.iter_mut().enumerate().skip(4) {
                        if !range.contains(&j) {
                            *coef = 0.0;
                        }
                    }
                    let value = eval_pt(part.with_params(&isolated).unwrap().as_ref(), density);
                    assert!(
                        (value - want).abs() < 2e-12,
                        "{name} {range:?}: {value} vs {want}"
                    );
                    if want == 0.0 {
                        assert_eq!(value, 0.0);
                    }
                }
            }
        }
    }

    fn eval_gga(f: &dyn XcFunctional, v: [f64; 5], order: usize) -> XcOut {
        let rho_a = [v[0]];
        let rho_b = [v[1]];
        let saa = [v[2]];
        let sab = [v[3]];
        let sbb = [v[4]];
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: Some([&saa, &sab, &sbb]),
            tau: None,
        };
        let mut out = XcOut::default();
        out.prepare(1, order, Family::Gga);
        f.eval_add(&vars, order, 1.0, &mut out).unwrap();
        out
    }

    fn eval_mgga(f: &dyn XcFunctional, v: [f64; 7], order: usize) -> XcOut {
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

    /// Every first and second derivative of the GGA members against central
    /// differences of the energy density.
    ///
    /// The SCF stationary point is defined by vrho/vsigma while the energy
    /// is accumulated from `e`: a derivative that does not belong to `e`
    /// converges the density to the wrong place and quietly shifts the
    /// total. ωB97X-D and ωB97X-V have no frozen SCF total, so this is what
    /// stands between them and that failure.
    #[test]
    fn family_derivatives_match_finite_differences() {
        let pts = [
            [0.3, 0.1, 0.0425, 0.031, 0.0233],
            [0.9, 0.55, 0.21, 0.08, 0.14],
        ];
        let family: [Box<dyn XcFunctional>; 9] = [
            b97(),
            hcth93(),
            hcth120(),
            hcth147(),
            hcth407(),
            wb97(),
            wb97x(),
            wb97x_d(),
            wb97x_v(),
        ];
        for f in family {
            for p in pts {
                let h = 1e-6;
                let d1 = eval_gga(f.as_ref(), p, 1);
                for var in 0..5 {
                    let mut up = p;
                    up[var] += h;
                    let mut dn = p;
                    dn[var] -= h;
                    let fd = (eval_gga(f.as_ref(), up, 0).e[0] - eval_gga(f.as_ref(), dn, 0).e[0])
                        / (2.0 * h);
                    let an = if var < 2 {
                        d1.vrho[var][0]
                    } else {
                        d1.vsigma[var - 2][0]
                    };
                    assert!(
                        (an - fd).abs() < 2e-6 * (1.0 + fd.abs()),
                        "{} var {var}: analytic {an} vs FD {fd} at {p:?}",
                        f.name()
                    );
                }
                // Second derivatives vs FD of the analytic first ones:
                // one block from each of the ρρ, ρσ and σσ groups.
                let d2 = eval_gga(f.as_ref(), p, 2);
                let first = |q: [f64; 5], slot: usize| -> f64 {
                    let o = eval_gga(f.as_ref(), q, 1);
                    match slot {
                        0 => o.vrho[0][0],
                        1 => o.vrho[1][0],
                        _ => o.vsigma[0][0],
                    }
                };
                for (var, slot, an) in [
                    (0usize, 0usize, d2.v2rho2[0][0]),
                    (1, 0, d2.v2rho2[1][0]),
                    (1, 1, d2.v2rho2[2][0]),
                    (2, 0, d2.v2rhosigma[0][0]),
                    (2, 2, d2.v2sigma2[0][0]),
                ] {
                    let mut up = p;
                    up[var] += h;
                    let mut dn = p;
                    dn[var] -= h;
                    let fd = (first(up, slot) - first(dn, slot)) / (2.0 * h);
                    assert!(
                        (an - fd).abs() < 5e-5 * (1.0 + fd.abs()),
                        "{} d2(var {var}, slot {slot}): analytic {an} vs FD {fd}",
                        f.name()
                    );
                }
            }
        }
    }

    /// The same finite-difference gate for the two meta-GGA members, whose
    /// τ derivatives feed the meta-GGA Fock and CPKS paths.
    #[test]
    fn meta_family_derivatives_match_finite_differences() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12];
        for f in [wb97m_v(), b97m_v()] {
            let h = 1e-6;
            let d1 = eval_mgga(f.as_ref(), p, 1);
            for var in 0..7 {
                let mut up = p;
                up[var] += h;
                let mut dn = p;
                dn[var] -= h;
                let fd = (eval_mgga(f.as_ref(), up, 0).e[0] - eval_mgga(f.as_ref(), dn, 0).e[0])
                    / (2.0 * h);
                let an = match var {
                    0 | 1 => d1.vrho[var][0],
                    2..=4 => d1.vsigma[var - 2][0],
                    _ => d1.vtau[var - 5][0],
                };
                assert!(
                    (an - fd).abs() < 2e-6 * (1.0 + fd.abs()),
                    "{} var {var}: analytic {an} vs FD {fd}",
                    f.name()
                );
            }
            let d2 = eval_mgga(f.as_ref(), p, 2);
            let first = |q: [f64; 7], slot: usize| -> f64 {
                let o = eval_mgga(f.as_ref(), q, 1);
                match slot {
                    0 => o.vrho[0][0],
                    1 => o.vsigma[0][0],
                    _ => o.vtau[0][0],
                }
            };
            for (var, slot, an) in [
                (0usize, 0usize, d2.v2rho2[0][0]),
                (2, 0, d2.v2rhosigma[0][0]),
                (2, 1, d2.v2sigma2[0][0]),
                (5, 0, d2.v2rhotau[0][0]),
                (5, 2, d2.v2tau2[0][0]),
            ] {
                let mut up = p;
                up[var] += h;
                let mut dn = p;
                dn[var] -= h;
                let fd = (first(up, slot) - first(dn, slot)) / (2.0 * h);
                assert!(
                    (an - fd).abs() < 5e-5 * (1.0 + fd.abs()),
                    "{} d2(var {var}, slot {slot}): analytic {an} vs FD {fd}",
                    f.name()
                );
            }
        }
    }

    /// With the range separation switched off the SR-LSDA attenuation
    /// factor F(ω/2kF) is exactly 1, so every ωB97 member collapses onto
    /// the plain B97 power series carrying its own coefficients.
    ///
    /// This pins the [γx, γss, γab, ω, cx…, css…, cab…] layout of the ωB97
    /// path against the [γx, γss, γab, cx…, css…, cab…] layout of the B97
    /// path: an off-by-one slice — which for ωB97X-V, the only member with
    /// a 3-term series, would silently reinterpret every coefficient —
    /// cannot survive it. It also proves the attenuation multiplies the
    /// exchange channel only, never the two correlation channels.
    #[test]
    fn wb97_family_collapses_to_the_b97_form_at_zero_omega() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233];
        for f in [wb97(), wb97x(), wb97x_d(), wb97x_v()] {
            let mut params = f.params();
            params[3] = 0.0;
            let unscreened = f.with_params(&params).unwrap();
            let mut b97_layout = params.clone();
            b97_layout.remove(3);
            let parent = boxed("b97-form", Kind::ExchangeCorrelation, b97_layout);
            let (got, want) = (eval_pt(unscreened.as_ref(), p), eval_pt(parent.as_ref(), p));
            assert!(
                (got - want).abs() < 1e-14 * (1.0 + want.abs()),
                "{}(ω = 0) {got} vs its B97 form {want}",
                f.name()
            );
        }
    }

    /// The whole family pinned pointwise against libxc at a spin-polarized
    /// GGA point (PySCF probe 2026-08-03; σ from the probe gradients).
    #[test]
    fn family_matches_libxc_reference_point() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233];
        let cases: [(&str, Box<dyn XcFunctional>, f64); 4] = [
            ("wb97x_d", wb97x_d(), -0.164983820320),
            ("wb97x_v", wb97x_v(), -0.160893613740),
            ("wb97", wb97(), -0.173831845676),
            ("wb97x", wb97x(), -0.162504790351),
        ];
        for (name, f, want) in cases {
            let got = eval_pt(f.as_ref(), p);
            assert!((got - want).abs() < 1e-8, "{name}: {got} vs libxc {want}");
        }
    }

    /// Second-order derivatives (fxc kernels) must stay finite over the
    /// full density range hit by real grids (small-rho tails included) —
    /// the analytic-Hessian kernel quadratics multiply them by possibly
    /// zero first derivatives, and NaN * 0 = NaN.
    #[test]
    fn wb97_second_derivatives_finite_at_small_density() {
        let f = wb97x_d();
        for &r in &[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.5] {
            for &s in &[0.0, 1e-20, 1e-8, 1.0, 1e4] {
                let rho_a = [r];
                let rho_b = [r];
                let saa = [s];
                let sab = [0.5 * s];
                let sbb = [s];
                let vars = GridVars {
                    rho: [&rho_a, &rho_b],
                    sigma: Some([&saa, &sab, &sbb]),
                    tau: None,
                };
                let mut out = XcOut::default();
                out.prepare(1, 2, Family::Gga);
                f.eval_add(&vars, 2, 1.0, &mut out).unwrap();
                let mut all = vec![out.e[0]];
                for k in 0..2 {
                    all.push(out.vrho[k][0]);
                }
                for k in 0..3 {
                    all.push(out.vsigma[k][0]);
                    all.push(out.v2rho2[k][0]);
                }
                for k in 0..6 {
                    all.push(out.v2rhosigma[k][0]);
                    all.push(out.v2sigma2[k][0]);
                }
                for (k, v) in all.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "wb97x_d rho={r:.0e} sigma={s:.0e}: output {k} = {v}"
                    );
                }
            }
        }
    }

    /// ωB97M-V pinned pointwise against libxc at a spin-polarized
    /// meta-GGA point (PySCF probe 2026-08-03).
    #[test]
    fn wb97m_v_matches_libxc_reference_point() {
        let rho_a = [0.3];
        let rho_b = [0.1];
        let saa = [0.0425];
        let sab = [0.031];
        let sbb = [0.0233];
        let ta = [0.35];
        let tb = [0.12];
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: Some([&saa, &sab, &sbb]),
            tau: Some([&ta, &tb]),
        };
        let mut out = XcOut::default();
        out.prepare(1, 0, Family::MetaGga);
        wb97m_v().eval_add(&vars, 0, 1.0, &mut out).unwrap();
        let want = -0.159514603125;
        assert!(
            (out.e[0] - want).abs() < 1e-10,
            "wb97m_v: {} vs libxc {want}",
            out.e[0]
        );
        // B97M-V at the same point (PySCF probe 2026-08-04).
        let mut out2 = XcOut::default();
        out2.prepare(1, 0, Family::MetaGga);
        b97m_v().eval_add(&vars, 0, 1.0, &mut out2).unwrap();
        let want2 = -0.263765805355;
        assert!(
            (out2.e[0] - want2).abs() < 1e-10,
            "b97m_v: {} vs libxc {want2}",
            out2.e[0]
        );
    }
}
