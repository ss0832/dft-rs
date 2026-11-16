// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The MN12/MN15 meta-nonseparable gradient approximations (Peverati and
//! Truhlar 2012; Yu, He and Truhlar 2016) and the M08 meta-GGAs (Zhao and
//! Truhlar 2008), which share the M08 correlation form.
//!
//! Exchange (per spin, on the spin density):
//!   e_x = e_x^LSDA(rho_s) sum_ijk CC_ijk v^i u^j w^k,
//! with u = gamma x^2 / (1 + gamma x^2), x^2 = sigma_ss / rho_s^(8/3),
//! gamma = 0.004; v = omega rho_s^(1/3) / (1 + omega rho_s^(1/3)),
//! omega = 2.5; and w = (tau^LSDA - tau_s) / (tau^LSDA + tau_s),
//! tau^LSDA = (3/10)(6 pi^2)^(2/3) rho_s^(5/3). The 40 terms have i, j <= 3
//! and i + j + k <= 5.
//!
//! Correlation (the M08 form, on the total density):
//!   e_c = rho [eps_c^PW92 sum_i a_i W^i + H^PBE sum_i b_i W^i],
//! W = (t - 1)/(t + 1), t = (3/10)(3 pi^2)^(2/3) rho^(5/3) / (tau_a + tau_b),
//! with the PBE gradient correction H at beta = 0.06672455060314922.
//!
//! The coefficients are the published tables. The conventions (the order of
//! the CC indices and the total-density W of the correlation) were settled
//! against libxc at random spin-polarized points to 5e-14
//! (`tools/oracle/probe_m08_form.py`).

use super::common::{RHO_CUTOFF, X_FACTOR_C};
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

/// Powers (i, j, k) of v, u and w of the 40 exchange coefficients.
const POWERS: [[usize; 3]; 40] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 2],
    [0, 0, 3],
    [0, 0, 4],
    [0, 0, 5],
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 2],
    [0, 1, 3],
    [0, 1, 4],
    [0, 2, 0],
    [0, 2, 1],
    [0, 2, 2],
    [0, 2, 3],
    [0, 3, 0],
    [0, 3, 1],
    [0, 3, 2],
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 2],
    [1, 0, 3],
    [1, 0, 4],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 3],
    [1, 2, 0],
    [1, 2, 1],
    [1, 2, 2],
    [2, 0, 0],
    [2, 0, 1],
    [2, 0, 2],
    [2, 0, 3],
    [2, 1, 0],
    [2, 1, 1],
    [2, 1, 2],
    [3, 0, 0],
    [3, 0, 1],
    [3, 0, 2],
];

static NAMES: [&str; 64] = [
    "CC000", "CC001", "CC002", "CC003", "CC004", "CC005", "CC010", "CC011", "CC012", "CC013",
    "CC014", "CC020", "CC021", "CC022", "CC023", "CC030", "CC031", "CC032", "CC100", "CC101",
    "CC102", "CC103", "CC104", "CC110", "CC111", "CC112", "CC113", "CC120", "CC121", "CC122",
    "CC200", "CC201", "CC202", "CC203", "CC210", "CC211", "CC212", "CC300", "CC301", "CC302", "a0",
    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "b0", "b1", "b2", "b3",
    "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11",
];

const TAU_SPIN: f64 = 9.115599744691192 / 2.0;
const PBE_BETA: f64 = 0.06672455060314922;

fn series<S: Scalar>(w: S, coefficients: &[S]) -> S {
    let mut acc = S::from_f64(0.0);
    for &c in coefficients.iter().rev() {
        acc = acc * w + c;
    }
    acc
}

pub(crate) fn mn12_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let one = S::from_f64(1.0);
    let zero = S::from_f64(0.0);
    let cc = &th[..40];
    let mut e = zero;
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r13 = rho.cbrt();
        let r43 = rho * r13;
        let x2 = sig / (r43 * r43);
        let u = S::from_f64(0.004) * x2 / (one + S::from_f64(0.004) * x2);
        let v = S::from_f64(2.5) * r13 / (one + S::from_f64(2.5) * r13);
        let tlsda = S::from_f64(TAU_SPIN) * r43 * r13;
        let w = (tlsda - tau) / (tlsda + tau);
        let pv = [one, v, v * v, v * v * v];
        let pu = [one, u, u * u, u * u * u];
        let pw = [one, w, w * w, w * w * w, w * w * w * w, w * w * w * w * w];
        let mut f = zero;
        for (c, [i, j, k]) in cc.iter().zip(POWERS) {
            f = f + *c * pv[i] * pu[j] * pw[k];
        }
        e = e - S::from_f64(X_FACTOR_C) * r43 * f;
    }
    e + m08_correlation(ra, rb, saa, sab, sbb, ta, tb, &th[40..52], &th[52..64])
}

/// The M08 correlation form on the total density.
#[allow(clippy::too_many_arguments)]
fn m08_correlation<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    a: &[S],
    b: &[S],
) -> S {
    let one = S::from_f64(1.0);
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let pw = super::lda::PW92_MOD_PARAMS.map(S::from_f64);
    let (eps, phi) = if super::lda::one_channel_only(ra, rb) {
        (
            super::lda::pw92_epsilon_ferro(rho, &pw),
            S::from_f64(2.0f64.powf(-1.0 / 3.0)),
        )
    } else {
        let z = (ra - rb) / rho;
        (
            super::lda::pw92_epsilon(ra, rb, &pw),
            ((one + z).powf(2.0 / 3.0) + (one - z).powf(2.0 / 3.0)) * S::from_f64(0.5),
        )
    };
    let sig = saa + S::from_f64(2.0) * sab + sbb;
    let gamma =
        S::from_f64((1.0 - std::f64::consts::LN_2) / (std::f64::consts::PI * std::f64::consts::PI));
    let h = super::gga::pbe_c_h(eps, phi, rho, sig, S::from_f64(PBE_BETA), gamma) - eps;
    let t = S::from_f64(0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0))
        * rho.powf(5.0 / 3.0)
        / (ta + tb);
    let w = (t - one) / (t + one);
    rho * (eps * series(w, a) + h * series(w, b))
}

fn boxed(name: &'static str, params: Vec<f64>) -> Box<dyn XcFunctional> {
    mgga_driver!(
        name,
        Kind::ExchangeCorrelation,
        params,
        &NAMES,
        mn12_e,
        false
    )
}

/// MN12-L (Peverati and Truhlar, PCCP 14, 13171 (2012)): pure meta-NGA.
pub fn mn12_l() -> Box<dyn XcFunctional> {
    boxed(
        "mn12l",
        vec![
            0.6735981,
            -2.270598,
            -2.613712,
            3.993609,
            4.635575,
            1.250676,
            0.844492,
            -13.01173,
            -17.7773,
            -4.627211,
            5.976605,
            1.142897,
            -20.40226,
            -23.82843,
            7.119109,
            -23.35726,
            -16.22633,
            14.82732,
            1.449285,
            10.20598,
            4.40745,
            -20.08193,
            -12.53561,
            -5.435031,
            16.56736,
            20.00229,
            -2.513105,
            9.658436,
            -3.825281,
            -25.0,
            -2.07008,
            -9.951913,
            0.8731211,
            22.10891,
            8.822633,
            24.99949,
            25.0,
            0.6851693,
            -0.07406948,
            -0.6788,
            0.884461,
            -0.2202279,
            5.701372,
            -2.562378,
            -0.9646827,
            0.1982183,
            10.19976,
            0.9789352,
            -1.512722,
            0.0,
            0.0,
            0.0,
            0.5323948,
            -5.831909,
            3.882386,
            5.878488,
            14.93228,
            -13.74636,
            -8.492327,
            -2.486548,
            -18.22346,
            0.0,
            0.0,
            0.0,
        ],
    )
}

/// MN12-SX semilocal part (Peverati and Truhlar, PCCP 14, 16187 (2012));
/// the screened exact exchange (0.25 short range, omega = 0.11) is added by
/// the combined entry.
pub fn mn12_sx() -> Box<dyn XcFunctional> {
    boxed(
        "mn12sx",
        vec![
            0.5226556, -0.2681208, -4.670705, 3.06732, 4.09537, 2.653023, 0.5165969, -20.35442,
            -9.946472, 2.938637, 11.311, 4.752452, -3.061331, -25.23173, 17.10903, -23.5748,
            -27.27754, 16.03291, 1.842503, 1.92712, 11.07987, -11.82087, -11.17768, -5.821,
            22.66545, 8.246708, -4.778364, 0.5329122, -6.666755, 1.671429, -3.311409, 0.3415913,
            -6.413076, 10.38584, 9.026277, 19.29689, 26.69232, 1.517278, -3.442503, 1.100161,
            0.7171161, -2.380914, 5.793565, -1.243624, 13.6492, -21.10812, -15.98767, 14.29208,
            6.149191, 0.0, 0.0, 0.0, 0.4663699, -9.110685, 8.705051, -1.813949, -0.4147211,
            -10.21527, 0.824027, 4.993815, -25.6393, 0.0, 0.0, 0.0,
        ],
    )
}

/// MN15-L (Yu, He and Truhlar, JCTC 12, 1280 (2016)): pure meta-NGA.
pub fn mn15_l() -> Box<dyn XcFunctional> {
    boxed(
        "mn15l",
        vec![
            0.670864162,
            -0.822003903,
            -1.022407046,
            1.689460986,
            -0.00562032,
            -0.110293849,
            0.972245178,
            -6.697641991,
            -4.322814495,
            -6.786641376,
            -5.687461462,
            9.419643818,
            11.83939406,
            5.086951311,
            4.302369948,
            -8.07344065,
            2.429988978,
            11.09485698,
            1.247333909,
            3.700485291,
            0.867791614,
            -0.591190518,
            -0.295305435,
            -5.825759145,
            2.537532196,
            3.143390933,
            2.939126332,
            0.599342114,
            2.241702738,
            2.035713838,
            -1.525344043,
            -2.325875691,
            1.141940663,
            -1.563165026,
            7.882032871,
            11.93400684,
            9.852928303,
            0.584030245,
            -0.720941131,
            -2.836037078,
            0.952058087,
            -0.756954364,
            5.677396094,
            -5.017104782,
            -5.10654071,
            -4.812053335,
            3.397640087,
            1.980041517,
            10.1231046,
            0.0,
            0.0,
            0.0,
            0.819504932,
            -7.689358913,
            -0.70532663,
            -0.600096421,
            11.03332527,
            5.861969337,
            8.913865465,
            5.74529876,
            4.254880837,
            0.0,
            0.0,
            0.0,
        ],
    )
}

/// MN15 semilocal part (Yu, He and Truhlar, Chem. Sci. 7, 5032 (2016));
/// global hybrid with cx = 0.44.
pub fn mn15() -> Box<dyn XcFunctional> {
    boxed(
        "mn15",
        vec![
            0.073852235,
            -0.839976156,
            -3.082660125,
            -1.02881285,
            -0.811697255,
            -0.063404387,
            2.54805518,
            -5.031578906,
            0.31702159,
            2.981868205,
            -0.749503735,
            0.231825661,
            1.261961411,
            1.665920815,
            7.483304941,
            -2.544245723,
            1.384720031,
            6.902569885,
            1.657399451,
            2.98526709,
            6.89391326,
            2.489813993,
            1.454724691,
            -5.054324071,
            2.35273334,
            1.299104132,
            1.203168217,
            0.121595877,
            8.048348238,
            21.91203659,
            -1.852335832,
            -3.4722735,
            -1.564591493,
            -2.29578769,
            3.666482991,
            10.87074639,
            9.696691388,
            0.630701064,
            -0.505825216,
            -3.562354535,
            1.093250748,
            -0.269735037,
            6.368997613,
            -0.245337101,
            -1.587103441,
            0.124698862,
            1.605819855,
            0.466206031,
            3.484978654,
            0.0,
            0.0,
            0.0,
            1.427424993,
            -3.57883682,
            7.398727547,
            3.927810559,
            2.789804639,
            4.988320462,
            3.079464318,
            3.521636859,
            4.769671992,
            0.0,
            0.0,
            0.0,
        ],
    )
}

static M08_NAMES: [&str; 48] = [
    "xa0", "xa1", "xa2", "xa3", "xa4", "xa5", "xa6", "xa7", "xa8", "xa9", "xa10", "xa11", "xb0",
    "xb1", "xb2", "xb3", "xb4", "xb5", "xb6", "xb7", "xb8", "xb9", "xb10", "xb11", "ca0", "ca1",
    "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cb0", "cb1", "cb2",
    "cb3", "cb4", "cb5", "cb6", "cb7", "cb8", "cb9", "cb10", "cb11",
];

/// Enhancement of the M08 exchange (Zhao and Truhlar, JCTC 4, 1849 (2008))
/// on one spin density: `F^PBE(s) sum_i a_i w^i + F^RPBE(s) sum_i b_i w^i`
/// with the published factors `F^PBE` (kappa = 0.804, mu = 0.21951) and
/// `F^RPBE = 1 + kappa (1 - exp(-mu s^2 / kappa))` (kappa = 0.552,
/// mu = 10/81), checked against libxc to 1.3e-14.
fn m08_enhancement<S: Scalar>(s2: S, w: S, a: &[S], b: &[S]) -> S {
    let one = S::from_f64(1.0);
    let (k1, m1, k2, m2) = (0.804, 0.21951, 0.552, 10.0 / 81.0);
    let f_pbe = one + S::from_f64(k1) - S::from_f64(k1) / (one + S::from_f64(m1 / k1) * s2);
    let f_rpbe = one + S::from_f64(k2) * (one - (-(S::from_f64(m2 / k2) * s2)).exp());
    f_pbe * series(w, a) + f_rpbe * series(w, b)
}

/// Per-spin ingredients of the M08-type exchange: the LSDA exchange energy,
/// s^2, w and the short-range attenuation factor of the omegaB97 family at
/// `omega` (one when `omega` is None).
fn m08_spin<S: Scalar>(rho: S, sig: S, tau: S, omega: Option<S>) -> (S, S, S, S) {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    let s2_factor = 1.0 / (4.0 * (6.0 * pi2).powf(2.0 / 3.0));
    let r13 = rho.cbrt();
    let r43 = rho * r13;
    let s2 = S::from_f64(s2_factor) * sig / (r43 * r43);
    let tlsda = S::from_f64(TAU_SPIN) * r43 * r13;
    let w = (tlsda - tau) / (tlsda + tau);
    let sr = match omega {
        Some(omega) => {
            let kf = (S::from_f64(6.0 * pi2) * rho).cbrt();
            super::gga::sr_factor(omega / (S::from_f64(2.0) * kf))
        }
        None => S::from_f64(1.0),
    };
    (-S::from_f64(X_FACTOR_C) * r43, s2, w, sr)
}

/// M08 exchange and correlation; layout `[xa (12), xb (12), ca (12), cb (12)]`.
pub(crate) fn m08_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let (ex, s2, w, _) = m08_spin(rho, sig, tau, None);
        e = e + ex * m08_enhancement(s2, w, &th[..12], &th[12..24]);
    }
    e + m08_correlation(ra, rb, saa, sab, sbb, ta, tb, &th[24..36], &th[36..48])
}

/// M11 (Peverati and Truhlar, JPCL 2, 2810 (2011)) semilocal part: the M08
/// exchange on the short-range (erfc) LSDA exchange, attenuated as in the
/// omegaB97 family, and the M08 correlation; layout
/// `[omega, xa (12), xb (12), ca (12), cb (12)]`. The exact exchange is
/// added by the combined entry.
pub(crate) fn m11_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let (ex, s2, w, sr) = m08_spin(rho, sig, tau, Some(th[0]));
        e = e + ex * sr * m08_enhancement(s2, w, &th[1..13], &th[13..25]);
    }
    e + m08_correlation(ra, rb, saa, sab, sbb, ta, tb, &th[25..37], &th[37..49])
}

/// M11-L (Peverati and Truhlar, JPCL 3, 117 (2012)): dual-range local
/// exchange, the M08 enhancement with one coefficient set on the
/// short-range and another on the long-range LSDA exchange, and the M08
/// correlation; layout `[omega, xa, xb, xc, xd, ca, cb]` (12 each).
pub(crate) fn m11_l_e<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    ta: S,
    tb: S,
    th: &[S],
) -> S {
    let one = S::from_f64(1.0);
    let mut e = S::from_f64(0.0);
    for (rho, sig, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let (ex, s2, w, sr) = m08_spin(rho, sig, tau, Some(th[0]));
        e = e + ex
            * (sr * m08_enhancement(s2, w, &th[1..13], &th[13..25])
                + (one - sr) * m08_enhancement(s2, w, &th[25..37], &th[37..49]));
    }
    e + m08_correlation(ra, rb, saa, sab, sbb, ta, tb, &th[49..61], &th[61..73])
}

/// M08-HX semilocal part (global hybrid, cx = 0.5223).
pub fn m08_hx() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "m08hx",
        Kind::ExchangeCorrelation,
        vec![
            1.3340172,
            -9.4751087,
            -12.541893,
            9.1369974,
            34.717204,
            58.831807,
            71.369574,
            23.312961,
            4.8314679,
            -6.5044167,
            -14.058265,
            12.88057,
            -0.85631823,
            9.2810354,
            12.260749,
            -5.5189665,
            -35.534989,
            -82.049996,
            -68.586558,
            36.085694,
            -9.3740983,
            -59.731688,
            16.587868,
            13.993203,
            1.0,
            -0.40661387,
            -3.323253,
            1.554098,
            44.248033,
            -84.35193,
            -119.55581,
            391.47081,
            183.63851,
            -632.68223,
            -112.97403,
            336.29312,
            1.3812334,
            -2.4683806,
            -11.901501,
            -54.112667,
            10.055846,
            148.00687,
            115.6142,
            255.91815,
            213.20772,
            -484.12067,
            -434.30813,
            56.627964,
        ],
        &M08_NAMES,
        m08_e,
        false
    )
}

/// M08-SO semilocal part (global hybrid, cx = 0.5679).
pub fn m08_so() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "m08so",
        Kind::ExchangeCorrelation,
        vec![
            -0.34888428,
            -5.8157416,
            37.55081,
            63.727406,
            -53.742313,
            -98.595529,
            16.282216,
            17.513468,
            -6.7627553,
            11.106658,
            1.5663545,
            8.760347,
            0.78098428,
            5.4538178,
            -37.853348,
            -62.29508,
            46.713254,
            87.321376,
            16.053446,
            20.12692,
            -40.343695,
            -58.577565,
            20.890272,
            10.946903,
            1.0,
            0.0,
            -3.9980886,
            12.98234,
            101.17507,
            -89.541984,
            -356.40242,
            206.98803,
            460.3778,
            -245.10559,
            -196.38425,
            118.81459,
            1.0,
            -4.4117403,
            -6.4128622,
            47.583635,
            186.30053,
            -128.00784,
            -553.85258,
            138.73727,
            416.46537,
            -266.26577,
            56.6763,
            316.73746,
        ],
        &M08_NAMES,
        m08_e,
        false
    )
}

static M11_NAMES: [&str; 49] = [
    "omega", "xa0", "xa1", "xa2", "xa3", "xa4", "xa5", "xa6", "xa7", "xa8", "xa9", "xa10", "xa11",
    "xb0", "xb1", "xb2", "xb3", "xb4", "xb5", "xb6", "xb7", "xb8", "xb9", "xb10", "xb11", "ca0",
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cb0", "cb1",
    "cb2", "cb3", "cb4", "cb5", "cb6", "cb7", "cb8", "cb9", "cb10", "cb11",
];

static M11_L_NAMES: [&str; 73] = [
    "omega", "xa0", "xa1", "xa2", "xa3", "xa4", "xa5", "xa6", "xa7", "xa8", "xa9", "xa10", "xa11",
    "xb0", "xb1", "xb2", "xb3", "xb4", "xb5", "xb6", "xb7", "xb8", "xb9", "xb10", "xb11", "xc0",
    "xc1", "xc2", "xc3", "xc4", "xc5", "xc6", "xc7", "xc8", "xc9", "xc10", "xc11", "xd0", "xd1",
    "xd2", "xd3", "xd4", "xd5", "xd6", "xd7", "xd8", "xd9", "xd10", "xd11", "ca0", "ca1", "ca2",
    "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cb0", "cb1", "cb2", "cb3",
    "cb4", "cb5", "cb6", "cb7", "cb8", "cb9", "cb10", "cb11",
];

/// M11 semilocal part (range-separated hybrid: SR exact exchange 0.428,
/// LR 1.0, omega = 0.25).
pub fn m11() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "m11",
        Kind::ExchangeCorrelation,
        vec![
            0.25,
            -0.183999,
            -13.9046703,
            11.8206837,
            31.0098465,
            -51.9625696,
            15.5750312,
            -6.9477573,
            -158.465014,
            -1.48447565,
            55.1042124,
            -13.4714184,
            0.0,
            0.755999,
            13.7137944,
            -12.7998304,
            -29.3428814,
            59.1075674,
            -22.7604866,
            -10.276934,
            164.752731,
            18.5349258,
            -55.6825639,
            7.47980859,
            0.0,
            1.0,
            0.0,
            -3.893325,
            -2.1688455,
            9.34972,
            -19.84514,
            2.3455253,
            79.246513,
            9.6042757,
            -67.856719,
            -9.1841067,
            0.0,
            0.72239798,
            0.43730564,
            -16.088809,
            -65.542437,
            32.05723,
            186.17888,
            20.483468,
            -70.853739,
            44.483915,
            -94.484747,
            -114.59868,
            0.0,
        ],
        &M11_NAMES,
        m11_e,
        false
    )
}

/// revM11 semilocal part (Verma, Wang and Truhlar, JPCL 10, 2809 (2019);
/// SR exact exchange 0.225, LR 1.0, omega = 0.40).
pub fn revm11() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "revm11",
        Kind::ExchangeCorrelation,
        vec![
            0.4,
            -0.3288860885,
            -8.3888150476,
            0.7123891057,
            3.6196212952,
            4.3941708207,
            5.0453345584,
            7.8667061191,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.1038860885,
            8.0476369587,
            -0.7353624773,
            -2.473527555,
            -4.7319060355,
            -5.8502502096,
            -7.5059975327,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -0.7860212983,
            -5.1132585425,
            -4.0716488878,
            1.5806421214,
            8.4135687567,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.9732839024,
            -2.1674450396,
            -9.3318324572,
            -12.9399606617,
            -2.212932066,
            -2.95085491,
            -1.506631936,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &M11_NAMES,
        m11_e,
        false
    )
}

/// M11-L (dual-range local meta-GGA, omega = 0.25).
pub fn m11_l() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "m11l",
        Kind::ExchangeCorrelation,
        vec![
            0.25, 0.8121131, 17.38124, 1.154007, 68.69556, 101.6864, -5.887467, 45.17409,
            -2.773149, -26.17211, 0.0, 0.0, 0.0, 0.1878869, -16.53877, 0.6755753, -75.67572,
            -104.0272, 18.31853, -55.73352, -3.52021, 37.24276, 0.0, 0.0, 0.0, -0.4386615,
            -121.4016, -139.3573, -2.046649, 28.04098, -13.12258, -6.361819, -0.8055758, 3.736551,
            0.0, 0.0, 0.0, 1.438662, 120.9465, 132.8252, 12.96355, 5.854866, -3.378162, -44.23393,
            6.844475, 19.49541, 0.0, 0.0, 0.0, 1.0, 0.0, 2.75088, -15.62287, 9.363381, 21.41024,
            -14.24975, -11.34712, 10.22365, 0.0, 0.0, 0.0, 1.0, -9.08206, 6.134682, -13.33216,
            -14.64115, 17.13143, 2.480738, -10.07036, -0.1117521, 0.0, 0.0, 0.0,
        ],
        &M11_L_NAMES,
        m11_l_e,
        false
    )
}
