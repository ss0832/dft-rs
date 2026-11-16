// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The M06 meta-GGA family (Zhao–Truhlar: M06-L, JCP 125, 194101 (2006);
//! M06 / M06-2X, Theor. Chem. Acc. 120, 215 (2008)).
//!
//! Exchange (per spin):
//!   e_x = e_x^PBE(ρσ, σσσ) · Σ_i a_i w^i  +  e_x^LSDA(ρσ) · h(x², z; d_x),
//! with w = (τ^LSDA − τ)/(τ^LSDA + τ), τ^LSDA = (3/10)(6π²)^{2/3} ρσ^{5/3},
//! x² = σσσ/ρσ^{8/3}, z = 2τσ/ρσ^{5/3} − C_F, C_F = (3/5)(6π²)^{2/3}, and
//! the VS98-style rational h(x², z; d) = d0/γ + (d1 x² + d2 z)/γ²
//! + (d3 x⁴ + d4 x²z + d5 z²)/γ³, γ = 1 + α(x² + z).
//!
//! Correlation splits into opposite- and same-spin channels on the PW92
//! ("modified"-digit) UEG base:
//!   e_ab = e^UEG_ab · [Σ_i c_i u^i + h(x̄², z̄; d_ab)],
//!     u = γ_ab x̄²/(1 + γ_ab x̄²), x̄² = xα² + xβ², z̄ = zα + zβ;
//!   e_ss = e^UEG_ss · D_σ · [Σ_i c_i u^i + h(x², z; d_ss)],
//!     u = γ_ss x²/(1 + γ_ss x²), D_σ = 1 − x²/(4(z + C_F)).
//!
//! Every constant of the fits is an exposed, differentiable parameter
//! (layout [α_x, α_ss, α_ab, γ_css, γ_cab, a×12, dx×6, cab×5, dab×6,
//! css×5, dss×6] = 45). All three coefficient tables are verified against
//! libxc by linear least-squares recovery over 900 random spin-polarized
//! meta-GGA points (max residual ≤ 8e-13; note M06-L a11 = −9.049762 and
//! cab4 = −12.55699 in the reference implementation).

use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

use super::common::RHO_CUTOFF;

/// Per-spin LSDA exchange energy density coefficient.
const CX_SPIN: f64 = -super::common::X_FACTOR_C;
/// C_F = (3/5)(6π²)^{2/3}.
const CF: f64 = 9.115599744691192;
/// PBE exchange enhancement constants.
const PBE_KAPPA: f64 = 0.804;
const PBE_MU: f64 = 0.2195149727645171;

/// PW92 with the "modified" reference-implementation A digits (the same
/// base PBE-C builds on) — pinned against libxc for the whole family.
pub(crate) fn pw_mod_params<S: Scalar>() -> Vec<S> {
    super::lda::PW92_MOD_PARAMS
        .iter()
        .map(|&v| S::from_f64(v))
        .collect()
}

fn series<S: Scalar>(u: S, coefs: &[S]) -> S {
    let mut acc = S::from_f64(0.0);
    let mut upow = S::from_f64(1.0);
    for &c in coefs {
        acc = acc + c * upow;
        upow = upow * u;
    }
    acc
}

/// VS98-style rational polynomial h(x², z; d) with γ = 1 + α(x² + z).
fn h_vs98<S: Scalar>(x2: S, z: S, alpha: S, d: &[S]) -> S {
    let g = S::from_f64(1.0) + alpha * (x2 + z);
    let g2 = g * g;
    d[0] / g
        + (d[1] * x2 + d[2] * z) / g2
        + (d[3] * x2 * x2 + d[4] * x2 * z + d[5] * z * z) / (g2 * g)
}

/// Spin-scaled PBE exchange energy density e_x^PBE(ρσ, σσσ) = ½e^PBE(2ρσ, 4σσσ).
fn pbe_x_spin<S: Scalar>(rho: S, sig: S) -> S {
    let two = S::from_f64(2.0);
    let r = two * rho;
    let kf2 = (S::from_f64(3.0 * std::f64::consts::PI * std::f64::consts::PI) * r).powf(2.0 / 3.0);
    let s2 = sig / (kf2 * r * r);
    let kappa = S::from_f64(PBE_KAPPA);
    let fx = S::from_f64(1.0) + kappa
        - kappa / (S::from_f64(1.0) + S::from_f64(PBE_MU / PBE_KAPPA) * s2);
    S::from_f64(0.5 * -0.75 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0)) * r.powf(4.0 / 3.0) * fx
}

/// The generic M06-family energy density (7 grid variables).
pub(crate) fn m06_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let (ax, ass, aab) = (th[0], th[1], th[2]);
    let (gss, gab) = (th[3], th[4]);
    let aw = &th[5..17];
    let dx = &th[17..23];
    let cab = &th[23..28];
    let dab = &th[28..34];
    let css = &th[34..39];
    let dss = &th[39..45];
    let zero = S::from_f64(0.0);
    let mut e = zero;
    let mut x2s = [zero, zero];
    let mut zs = [zero, zero];
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
        // w = (t−1)/(t+1), t = τ^LSDA/τ, in the τ-division-free form.
        let w = (tlsda - tau) / (tlsda + tau);
        e = e + pbe_x_spin(rho, sig) * series(w, aw);
        let x2 = sig / (r53 * rho);
        let kinetic_ratio = S::from_f64(2.0) * tau / r53;
        let z = kinetic_ratio - S::from_f64(CF);
        x2s[idx] = x2;
        zs[idx] = z;
        e = e + S::from_f64(CX_SPIN) * rho.powf(4.0 / 3.0) * h_vs98(x2, z, ax, dx);
        // Same-spin correlation with the self-interaction factor D.
        let ec_ss = rho * super::lda::pw92_epsilon_ferro(rho, &pw_mod_params::<S>());
        let zpcf = if kinetic_ratio.value() < 1e-15 {
            S::from_f64(1e-15)
        } else {
            kinetic_ratio
        };
        let d = S::from_f64(1.0) - x2 / (S::from_f64(4.0) * zpcf);
        // In the physical domain D >= 0. At the iso-orbital boundary use
        // the derivative from that side even if independently evaluated
        // density descriptors put D a few ulps below zero.
        let d_fac = if d.value() < -64.0 * f64::EPSILON {
            zero
        } else if d.value() < 0.0 {
            d - S::from_f64(d.value())
        } else {
            d
        };
        let uss = gss * x2 / (S::from_f64(1.0) + gss * x2);
        e = e + ec_ss * d_fac * (series(uss, css) + h_vs98(x2, z, ass, dss));
    }
    // Opposite-spin correlation.
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
        let xbar2 = x2s[0] + x2s[1];
        let zbar = zs[0] + zs[1];
        let uab = gab * xbar2 / (S::from_f64(1.0) + gab * xbar2);
        e = e + ec_ab * (series(uab, cab) + h_vs98(xbar2, zbar, aab, dab));
    }
    e
}

static M06_PARAM_NAMES: [&str; 45] = [
    "alpha_x",
    "alpha_ss",
    "alpha_ab",
    "gamma_css",
    "gamma_cab",
    "a_0",
    "a_1",
    "a_2",
    "a_3",
    "a_4",
    "a_5",
    "a_6",
    "a_7",
    "a_8",
    "a_9",
    "a_10",
    "a_11",
    "dx_0",
    "dx_1",
    "dx_2",
    "dx_3",
    "dx_4",
    "dx_5",
    "cab_0",
    "cab_1",
    "cab_2",
    "cab_3",
    "cab_4",
    "dab_0",
    "dab_1",
    "dab_2",
    "dab_3",
    "dab_4",
    "dab_5",
    "css_0",
    "css_1",
    "css_2",
    "css_3",
    "css_4",
    "dss_0",
    "dss_1",
    "dss_2",
    "dss_3",
    "dss_4",
    "dss_5",
];

/// VS98 attenuation α's and correlation γ's shared by the whole family.
const ALPHAS_GAMMAS: [f64; 5] = [0.00186726, 0.00515088, 0.00304966, 0.06, 0.0031];

fn m06_params(
    a: [f64; 12],
    dx: [f64; 6],
    cab: [f64; 5],
    dab: [f64; 6],
    css: [f64; 5],
    dss: [f64; 6],
) -> Vec<f64> {
    let mut p = ALPHAS_GAMMAS.to_vec();
    p.extend_from_slice(&a);
    p.extend_from_slice(&dx);
    p.extend_from_slice(&cab);
    p.extend_from_slice(&dab);
    p.extend_from_slice(&css);
    p.extend_from_slice(&dss);
    p
}

fn boxed(name: &'static str, params: Vec<f64>) -> Box<dyn XcFunctional> {
    mgga_driver!(
        name,
        Kind::ExchangeCorrelation,
        params,
        &M06_PARAM_NAMES,
        m06_e,
        false
    )
}

/// M06-L (pure meta-GGA, no HF exchange).
pub fn m06_l() -> Box<dyn XcFunctional> {
    boxed(
        "m06l",
        m06_params(
            [
                0.3987756, 0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615, 30.97273,
                -23.18489, -56.73480, 21.60364, 34.21814, -9.049762,
            ],
            [
                0.6012244,
                0.004748822,
                -0.008635108,
                -9.308062e-6,
                4.482811e-5,
                0.0,
            ],
            [0.6042374, 177.6783, -251.3252, 76.35173, -12.55699],
            [
                0.3957626,
                -0.5614546,
                0.01403963,
                0.0009831442,
                -0.003577176,
                0.0,
            ],
            [0.5349466, 0.5396620, -31.61217, 51.49592, -29.19613],
            [
                0.4650534,
                0.1617589,
                0.1833657,
                0.00046921,
                -0.004990573,
                0.0,
            ],
        ),
    )
}

/// M06 semilocal part (global hybrid, cx = 0.27).
pub fn m06() -> Box<dyn XcFunctional> {
    boxed(
        "m06",
        m06_params(
            [
                0.5877943, -0.1371776, 0.2682367, -2.515898, -2.978892, 8.710679, 16.88195,
                -4.489724, -32.99983, -14.49050, 20.43747, 12.56504,
            ],
            [0.1422057, 0.0007370319, -0.01601373, 0.0, 0.0, 0.0],
            [3.741539, 218.7098, -453.1252, 293.6479, -62.87470],
            [
                -2.741539,
                -0.6720113,
                -0.07932688,
                0.001918681,
                -0.002032902,
                0.0,
            ],
            [0.5094055, -1.491085, 17.23922, -38.59018, 28.45044],
            [
                0.4905945,
                -0.1437348,
                0.2357824,
                0.001871015,
                -0.003788963,
                0.0,
            ],
        ),
    )
}

/// Exchange w-series of an M05-type hybrid: the published coefficients
/// times the semilocal fraction `1 - cx`.
fn m05_exchange(cx: f64, published: [f64; 12]) -> [f64; 12] {
    published.map(|a| (1.0 - cx) * a)
}

/// M05 semilocal part (Zhao, Schultz and Truhlar, JCP 123, 161103 (2005);
/// global hybrid, cx = 0.28): the M06 form without the VS98 terms.
pub fn m05() -> Box<dyn XcFunctional> {
    boxed(
        "m05",
        m06_params(
            m05_exchange(
                0.28,
                [
                    1.0, 0.08151, -0.43956, -3.22422, 2.01819, 8.79431, -0.00295, 9.82029,
                    -4.82351, -48.17574, 3.64802, 34.02248,
                ],
            ),
            [0.0; 6],
            [1.0, 3.78569, -14.15261, -7.46589, 17.94491],
            [0.0; 6],
            [1.0, 3.77344, -26.04463, 30.69913, -9.22695],
            [0.0; 6],
        ),
    )
}

/// M05-2X semilocal part (Zhao, Schultz and Truhlar, JCTC 2, 364 (2006);
/// global hybrid, cx = 0.56).
pub fn m05_2x() -> Box<dyn XcFunctional> {
    boxed(
        "m052x",
        m06_params(
            m05_exchange(
                0.56,
                [
                    1.0, -0.56833, -1.30057, 5.50070, 9.06402, -32.21075, -23.73298, 70.22996,
                    29.88614, -60.25778, -13.22205, 15.23694,
                ],
            ),
            [0.0; 6],
            [1.0, 1.09297, -3.79171, 2.82810, -10.58909],
            [0.0; 6],
            [1.0, -3.05430, 7.61854, 1.47665, -11.92365],
            [0.0; 6],
        ),
    )
}

/// M06-HF semilocal part (Zhao and Truhlar, JPCA 110, 13126 (2006); full
/// exact exchange, cx = 1): the M06 form with its own VS98 terms.
pub fn m06_hf() -> Box<dyn XcFunctional> {
    boxed(
        "m06hf",
        m06_params(
            [
                0.1179732, -1.066708, -0.1462405, 7.481848, 3.776679, -44.36118, -18.30962,
                100.3903, 38.64360, -98.06018, -25.57716, 35.90404,
            ],
            [-0.1179732, -0.0025, -0.01180065, 0.0, 0.0, 0.0],
            [1.674634, 57.32017, 59.55416, -231.1007, 125.5199],
            [
                -0.6746338,
                -0.1534002,
                -0.09021521,
                -0.001292037,
                -0.0002352983,
                0.0,
            ],
            [0.1023254, -2.453783, 29.13180, -34.94358, 23.15955],
            [
                0.8976746,
                -0.2345830,
                0.2368173,
                -0.000991389,
                -0.01146165,
                0.0,
            ],
        ),
    )
}

/// VSXC (Van Voorhis and Scuseria, JCP 109, 400 (1998)): the VS98 terms
/// of the M06 form alone. The correlation coefficients are the published
/// ones; the exchange coefficients, which the publication states in a
/// different LSDA normalization, were recovered from libxc by linear least
/// squares (residual 1.9e-13).
pub fn vsxc() -> Box<dyn XcFunctional> {
    boxed(
        "vsxc",
        m06_params(
            [0.0; 12],
            [
                1.05324147585604,
                0.00382234242471457,
                -0.0067169834810035,
                2.53030938091731e-5,
                0.000137850244218862,
                -0.000384172286737358,
            ],
            [0.0; 5],
            [
                0.703501,
                0.007694574,
                0.05152765,
                3.394308e-5,
                -0.00126942,
                0.001296118,
            ],
            [0.0; 5],
            [
                0.3270912,
                -0.03228915,
                -0.02942406,
                0.002134222,
                -0.005451559,
                0.01577575,
            ],
        ),
    )
}

/// revM06-L (Wang et al., PNAS 114, 8487 (2017)): same functional form
/// and alpha/gamma constants as M06; coefficients recovered from libxc by
/// least squares (residual 6.8e-13).
pub fn revm06_l() -> Box<dyn XcFunctional> {
    boxed(
        "revm06l",
        m06_params(
            [
                1.42322725,
                0.471820438,
                -0.167555701,
                -0.250154262,
                0.062487588,
                0.73350124,
                -2.35973678,
                -1.43659437,
                0.444643793,
                1.52992505,
                2.05394172,
                -0.036536031,
            ],
            [-0.423227252, 0.0, 0.003724234, 0.0, 0.0, 0.0],
            [
                0.344360696,
                -0.557080242,
                -2.00982116,
                -1.85764189,
                -1.07663986,
            ],
            [0.4007146, 0.015796569, -0.032680984, 0.0, 0.0, 0.001260132],
            [
                1.22765975,
                0.855201283,
                -3.11334668,
                -2.23967803,
                0.354638962,
            ],
            [
                -0.538821292,
                -0.02829603,
                0.023889696,
                0.0,
                0.0,
                -0.002437902,
            ],
        ),
    )
}

/// revM06 semilocal part (Wang et al., JCP... 2018; global hybrid,
/// cx = 0.4041). Coefficients LSQ-recovered from libxc (7.1e-14).
pub fn revm06() -> Box<dyn XcFunctional> {
    boxed(
        "revm06",
        m06_params(
            [
                0.651139401,
                -0.121449776,
                -0.136704113,
                0.398721855,
                0.605674136,
                -2.37973866,
                -1.49209835,
                3.03147342,
                0.514963711,
                2.63375191,
                0.988674925,
                -4.24371413,
            ],
            [-0.0552394014, 0.0, -0.00378263123, 0.0, 0.0, 0.0],
            [
                1.2224016,
                0.661390734,
                -1.88458104,
                -2.78036057,
                -3.06857934,
            ],
            [
                -0.339066672,
                0.00379015638,
                -0.0276248597,
                0.0,
                0.0,
                0.000407628516,
            ],
            [
                0.901722457,
                0.207999183,
                -1.82374756,
                -1.38443043,
                -0.442325338,
            ],
            [
                -0.14670959,
                -0.000183218701,
                0.0848437243,
                0.0,
                0.0,
                0.000228067717,
            ],
        ),
    )
}

/// M06-SX semilocal part (Wang et al., PNAS 117, 2294 (2020); screened
/// hybrid: SR-HF = 0.335, ω = 0.10 → cx = 0.335, clr = −0.335 on the
/// composite). Same functional form as revM06.
pub fn m06_sx() -> Box<dyn XcFunctional> {
    boxed(
        "m06sx",
        m06_params(
            [
                0.996501680264007,
                0.0301264933631367,
                -0.103366758333673,
                -0.155653062500239,
                0.00795768051149902,
                0.0871986277454856,
                -0.816152625764469,
                0.672773006612420,
                0.521127186174968,
                0.399466945122217,
                0.519400018999204,
                -0.965261552636835,
            ],
            [-0.347792307472902, 0.0, -0.00270366787478266, 0.0, 0.0, 0.0],
            [
                0.163738167314691,
                -0.436481171027951,
                -1.90232628449712,
                -1.42432902881841,
                -0.905909137360893,
            ],
            [
                0.740594619832397,
                0.0123306511345974,
                -0.0188253421850249,
                0.0,
                0.0,
                4.87276242162303e-4,
            ],
            [
                1.17575011057022,
                0.658083496678423,
                -2.78913774852905,
                -1.18597601856255,
                1.16439928209688,
            ],
            [
                0.0817322574473352,
                -0.0288531085759385,
                0.0905917734868130,
                0.0,
                0.0,
                -4.86297499082106e-4,
            ],
        ),
    )
}

/// M06-2X semilocal part (global hybrid, cx = 0.54; no VS98 exchange term).
pub fn m06_2x() -> Box<dyn XcFunctional> {
    boxed(
        "m062x",
        m06_params(
            [
                0.46,
                -0.2206052,
                -0.09431788,
                2.164494,
                -2.556466,
                -14.22133,
                15.55044,
                35.98078,
                -27.22754,
                -39.24093,
                15.22808,
                15.22227,
            ],
            [0.0; 6],
            [0.8833596, 33.57972, -70.43548, 49.78271, -18.52891],
            [
                0.1166404,
                -0.09120847,
                -0.06726189,
                6.720580e-5,
                0.0008448011,
                0.0,
            ],
            [0.3097855, -5.528642, 13.47420, -32.13623, 28.46742],
            [
                0.6902145,
                0.09847204,
                0.2214797,
                -0.001968264,
                -0.006775479,
                0.0,
            ],
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{Family, GridVars, XcOut};

    fn eval_pt(f: &dyn XcFunctional, v: [f64; 7], order: usize) -> XcOut {
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

    #[test]
    fn iso_orbital_derivatives_are_stable_to_roundoff() {
        let f = m06_l();
        let mut point = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.0, 0.0];
        point[5] = point[2] / (8.0 * point[0]);
        point[6] = point[4] / (8.0 * point[1]);
        let evaluate = |scale| {
            let mut p = point;
            p[5] *= scale;
            p[6] *= scale;
            let result = eval_pt(f.as_ref(), p, 1);
            [
                result.e[0],
                result.vrho[0][0],
                result.vrho[1][0],
                result.vsigma[0][0],
                result.vsigma[1][0],
                result.vsigma[2][0],
                result.vtau[0][0],
                result.vtau[1][0],
            ]
        };
        let inside = evaluate(1.0 + 4.0 * f64::EPSILON);
        let rounded = evaluate(1.0 - 4.0 * f64::EPSILON);
        for (i, (a, b)) in inside.into_iter().zip(rounded).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "iso-orbital component {i}: {a} vs {b}"
            );
        }
    }

    /// The family pinned pointwise against libxc at a spin-polarized
    /// meta-GGA point (PySCF probe 2026-08-03).
    #[test]
    fn family_matches_libxc_reference_point() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12];
        // The revM06 coefficients are transcribed from the LSQ fit at 9
        // significant digits - the pins carry that truncation (5e-9).
        let cases: [(&str, Box<dyn XcFunctional>, f64, f64); 6] = [
            ("m06l", m06_l(), -0.261733012176, 1e-10),
            ("m06", m06(), -0.181625209346, 1e-10),
            ("m062x", m06_2x(), -0.112216849670, 1e-10),
            ("revm06l", revm06_l(), -0.265452990823, 5e-9),
            ("revm06", revm06(), -0.150870102365, 5e-9),
            ("m06sx", m06_sx(), -0.173924401834, 5e-9),
        ];
        for (name, f, want, tol) in cases {
            let got = eval_pt(f.as_ref(), p, 0).e[0];
            assert!((got - want).abs() < tol, "{name}: {got} vs libxc {want}");
        }
    }

    /// First derivatives (vrho/vsigma/vtau) against central differences.
    #[test]
    fn first_derivatives_match_finite_differences() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12];
        let f = m06_l();
        let out = eval_pt(f.as_ref(), p, 1);
        let h = 1e-6;
        for (k, slot) in [(0usize, 0usize), (1, 1), (2, 2), (4, 4), (5, 5), (6, 6)] {
            let mut pp = p;
            pp[k] += h;
            let ep = eval_pt(f.as_ref(), pp, 0).e[0];
            pp[k] -= 2.0 * h;
            let em = eval_pt(f.as_ref(), pp, 0).e[0];
            let fd = (ep - em) / (2.0 * h);
            let an = match slot {
                0 => out.vrho[0][0],
                1 => out.vrho[1][0],
                2 => out.vsigma[0][0],
                4 => out.vsigma[2][0],
                5 => out.vtau[0][0],
                _ => out.vtau[1][0],
            };
            assert!(
                (an - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                "var {k}: analytic {an} vs FD {fd}"
            );
        }
    }
}
