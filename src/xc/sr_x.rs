// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Short-range (erfc-screened) DFT exchange holes for screened hybrids and
//! canonical RSH functionals.
//!
//! HJS exchange holes: T. M. Henderson, B. G. Janesko, G. E. Scuseria,
//! J. Chem. Phys. 128, 194105 (2008), Eq. (43) and Table II.
//!
//! Spin scaling: e_x = Σσ −Cx·2^{1/3}·ρσ^{4/3}·F(sσ, νσ) with per-spin
//! sσ = |∇ρσ|/(2(6π²)^{1/3}ρσ^{4/3}) and νσ = ω/kFσ, kFσ = (6π²ρσ)^{1/3}
//! (quantum-chemistry spin-scaling convention for the screened hole).
//!
//! ω enters as a functional parameter θ = [ω], so ∂E/∂ω (ω-tuning) comes
//! from the same dual machinery as every other parameter gradient.
//! Verified pointwise against libxc gga_x_hjs_pbe (through the
//! HYB_GGA_XC_LC_WPBE_WHS composite minus GGA_C_PBE).

use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

use super::common::RHO_CUTOFF;

/// HJS universal hole constants (paper Eq. (25) ff.).
const HJS_A: f64 = 0.757211;
const HJS_B: f64 = -0.106364;
const HJS_C: f64 = -0.118649;
const HJS_D: f64 = 0.609650;

/// PBE-variant fit: H(s) = Σ a_i s^{i+2} / (1 + Σ b_j s^{j+1}).
const HJS_PBE_A: [f64; 6] = [
    0.0159941, 0.0852995, -0.160368, 0.152645, -0.0971263, 0.0422061,
];
const HJS_PBE_B: [f64; 9] = [
    5.33319, -12.4780, 11.0988, -5.11013, 1.71468, -0.610380, 0.307555, -0.0770547, 0.0334840,
];
const HJS_PBESOL_A: [f64; 6] = [
    0.0047333, 0.0403304, -0.0574615, 0.0435395, -0.0216251, 0.0063721,
];
const HJS_PBESOL_B: [f64; 9] = [
    8.52056, -13.9885, 9.28583, -3.27287, 0.843499, -0.235543, 0.0847074, -0.0171561, 0.0050552,
];

/// Positivity floor for reduced quantities feeding √ / ln (1e-10, matching
/// the libxc 7.0 release convention our oracle pins against); the floored
/// constant carries zero derivative.
fn floor_at<S: Scalar>(x: S) -> S {
    const FLOOR: f64 = 1e-10;
    if x.value() < FLOOR {
        S::from_f64(FLOOR)
    } else {
        x
    }
}

/// The HJS SR enhancement factor F(s, ν) for one spin channel.
fn hjs_enhancement<S: Scalar>(s: S, nu: S, numerator: &[f64; 6], denominator: &[f64; 9]) -> S {
    let one = S::from_f64(1.0);
    let (aa, bb, cc, dd) = (
        S::from_f64(HJS_A),
        S::from_f64(HJS_B),
        S::from_f64(HJS_C),
        S::from_f64(HJS_D),
    );
    // H(s) rational fit.
    let mut num = S::from_f64(0.0);
    let mut sp = s * s;
    for &ai in numerator {
        num = num + S::from_f64(ai) * sp;
        sp = sp * s;
    }
    let mut den = one;
    let mut sq = s;
    for &bi in denominator {
        den = den + S::from_f64(bi) * sq;
        sq = sq * s;
    }
    let h = num / den;

    let zeta = floor_at(s * s * h);
    let eta = floor_at(aa + zeta);
    let lambda = dd + zeta;
    let nu2 = nu * nu;
    let sqrt_lnu = (lambda + nu2).sqrt();
    let chi = nu / sqrt_lnu;
    // 1 − χ rationalized: λ/((√(λ+ν²)+ν)·√(λ+ν²)) — cancellation-free
    // at large ν where χ → 1.
    let omchi = lambda / ((sqrt_lnu + nu) * sqrt_lnu);

    let s2 = s * s;
    let ff = one
        - s2 / (S::from_f64(27.0) * cc * (one + s2 * S::from_f64(0.25)))
        - zeta / (S::from_f64(2.0) * cc);
    let l2 = lambda * lambda;
    let l3 = l2 * lambda;
    // √ζ − √η written as (ζ−η)/(√ζ+√η); ζ − η = −A exactly.
    let gg = -S::from_f64(0.4) * cc * ff * lambda
        - S::from_f64(4.0 / 15.0) * bb * l2
        - S::from_f64(1.2) * aa * l3
        - l3 * lambda.sqrt()
            * (S::from_f64(0.8) * S::from_f64(std::f64::consts::PI).sqrt()
                + S::from_f64(2.4) * (zeta - eta) / (zeta.sqrt() + eta.sqrt()));

    let sqrt_znu = (zeta + nu2).sqrt();
    let sqrt_enu = (eta + nu2).sqrt();
    aa - S::from_f64(4.0 / 9.0) * bb * omchi / lambda
        - S::from_f64(2.0 / 9.0) * cc * ff * omchi * omchi * (S::from_f64(2.0) + chi) / l2
        - S::from_f64(1.0 / 9.0)
            * gg
            * omchi
            * omchi
            * omchi
            * (S::from_f64(8.0) + S::from_f64(9.0) * chi + S::from_f64(3.0) * chi * chi)
            / l3
        + S::from_f64(2.0) * nu * (zeta - eta) / (sqrt_znu + sqrt_enu)
        + S::from_f64(2.0)
            * zeta
            * ((zeta - lambda) / ((sqrt_znu + sqrt_lnu) * (nu + sqrt_lnu))).ln_1p()
        - S::from_f64(2.0)
            * eta
            * ((eta - lambda) / ((sqrt_enu + sqrt_lnu) * (nu + sqrt_lnu))).ln_1p()
}

/// Spin-scaled HJS screened exchange for the given fit to H(s).
fn hjs_energy<S: Scalar>(
    spins: [(S, S); 2],
    omega: S,
    numerator: &[f64; 6],
    denominator: &[f64; 9],
) -> S {
    let x2s = 0.5 / (6.0 * std::f64::consts::PI * std::f64::consts::PI).cbrt();
    let kf_c = (6.0 * std::f64::consts::PI * std::f64::consts::PI).cbrt();
    let cx_spin = 0.75 * (3.0 / std::f64::consts::PI).cbrt() * 2.0_f64.cbrt();
    let mut e = S::from_f64(0.0);
    for (rho, sig) in spins {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r43 = rho.powf(4.0 / 3.0);
        let nu = omega / (S::from_f64(kf_c) * rho.cbrt());
        // s = x2s·√σ/ρ^{4/3}. HJS's H(s) rational fit carries ODD powers of
        // s (see `hjs_enhancement`), so the enhancement is genuinely
        // non-analytic in σ at zero and no rewriting removes it. At σ = 0 the
        // reduced gradient is exactly zero, so pass a CONSTANT zero: the
        // internal `floor_at` keeps ζ away from zero, nothing takes √0, and
        // every σ-derivative comes out zero — the absent-gradient convention.
        let s = if super::gga::sigma_is_absent(sig) {
            S::from_f64(0.0)
        } else {
            S::from_f64(x2s) * sig.sqrt() / r43
        };
        e = e - S::from_f64(cx_spin) * r43 * hjs_enhancement(s, nu, numerator, denominator);
    }
    e
}

fn hjs_pbe_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    hjs_energy([(ra, saa), (rb, sbb)], th[0], &HJS_PBE_A, &HJS_PBE_B)
}

fn hjs_pbesol_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    hjs_energy([(ra, saa), (rb, sbb)], th[0], &HJS_PBESOL_A, &HJS_PBESOL_B)
}

/// HJS ωPBE SR exchange with range-separation parameter ω.
pub fn hjs_pbe_x(omega: f64) -> Box<dyn XcFunctional> {
    gga_driver!(
        "hjs-pbe-x",
        Kind::Exchange,
        vec![omega],
        &["omega"],
        hjs_pbe_e
    )
}

/// HJS short-range PBEsol exchange with screening in inverse Bohr.
pub(crate) fn hjs_pbesol_x(omega: f64) -> Box<dyn XcFunctional> {
    gga_driver!(
        "hjs-pbesol-x",
        Kind::Exchange,
        vec![omega],
        &["omega"],
        hjs_pbesol_e
    )
}

// ---------------------------------------------------------------------------
// wPBEh: the HSE screened exchange hole (Heyd-Scuseria-Ernzerhof,
// J. Chem. Phys. 118, 8207 (2003); Henderson-Izmaylov-Scalmani-Scuseria,
// J. Chem. Phys. 131, 044108 (2009) for the s-rescaling). ω = 0 gives the
// full-range PBE-hole exchange used as HSE's parent. All constants are the
// published hole-model fit values. Verified pointwise against libxc
// gga_x_wpbeh (through HSE06 / LC_WPBE composites).
// ---------------------------------------------------------------------------

const WA: f64 = 1.0161144;
const WB: f64 = -0.37170836;
const WC: f64 = -0.077215461;
const WD: f64 = 0.57786348;
const WE: f64 = -0.051955731;
/// H(s) = (a1 s² + a2 s⁴)/(1 + a3 s⁴ + a4 s⁵ + a5 s⁶).
const WHA: [f64; 5] = [0.00979681, 0.0410834, 0.187440, 0.00120824, 0.0347188];
/// F(s) = Fc1·H(s) + Fc2 (published rounding of 4A²/(9C)+(B−AD)/C, −1/(27C)).
const WFC1: f64 = 6.4753871;
const WFC2: f64 = 0.47965830;
/// Small-s expansion of EG(s) below s = 0.08.
const WEGA: [f64; 3] = [-0.02628417880, -0.07117647788, 0.08534541323];
/// Expansion coefficients of the screened kernel (exp fit of the hole).
const WEA: [f64; 8] = [
    -1.128223946706117,
    1.452736265762971,
    -1.243162299390327,
    0.971824836115601,
    -0.568861079687373,
    0.246880514820192,
    -0.065032363850763,
    0.008401793031216,
];
const WEB1: f64 = 1.455915450052607;
const WSMAX: f64 = 8.572844;
const WEGSCUT: f64 = 0.08;
const WCUT: f64 = 14.0;

/// Lower bound on the reduced gradient `s` inside the wPBEh hole.
///
/// Not cosmetic, and not about `√σ`. When ω = 0 — which HSE06 and HSE03 both
/// carry, as `wpbeh(0) − 0.25·wpbeh(ω)` — the screening variable `w` is
/// exactly zero, so `aux4 = s²h + eb1·w²` collapses to `s²h ≈ 0.0098·s⁴`.
/// The series below raises `aux4` to inverse powers as far as `aux4⁻³`, and
/// a SECOND-order dual needs two derivatives of those, i.e. `aux4⁻⁵` and
/// `aux4⁻⁷`. At the previous floor of 1e-15 that is `aux4 ≈ 1e-62`, whose
/// seventh inverse power overflows to `+∞` — and the enclosing `f9·w⁸`
/// then evaluates `∞·0 = NaN`. The first derivative survived (only `aux4⁻⁴`
/// is reached), which is exactly why this hid until the order-2 sweep.
///
/// 1e-8 keeps `aux4⁻⁷ ≈ 1e238`, comfortably inside f64, while perturbing the
/// enhancement factor by O(s²) = 1e-16 relative — at the rounding floor of
/// the arithmetic itself, and far below the 2.5e-6 gate the wPBEh-based
/// functionals are held to. Both libxc pin points (s ≈ 0.05–0.15 and 9.6)
/// are orders of magnitude above it.
const WPBEH_S_FLOOR: f64 = 1e-8;

/// Henderson 2009 Eq. (8) s-rescaling (saturates s at smax).
///
/// # Why this is branchless, and where it departs from libxc
///
/// The saturation Henderson Eq. (8) defines is the softplus
/// `s − ln(1 + e^{s−smax})`, which is already C^∞ and already `≈ s` for small
/// s. libxc 7.0.0 (`maple/scaling.mpl:26-29`, labelled there as the version
/// "apparently used by Gaussian") nonetheless short-circuits it as
/// `s < 1 → s` and `s > 15 → smax`, and BOTH shortcuts are discontinuous:
/// measured, the function jumps by −5.140957e-04 at s = 1 and by
/// +1.615737e-03 at s = 15. This implementation reproduced libxc's structure
/// and therefore its jumps to the last quoted digit.
///
/// s ≈ 1 is the valence-density region that carries most of the exchange
/// integral, so those are not tail artefacts. This project implements from
/// the primary literature and treats a mathematical discontinuity as a defect
/// even where the reference implementation shares it, so the shortcuts are
/// removed.
///
/// # Why the bare softplus is not the answer either
///
/// Removing the shortcuts and using `s − ln(1 + e^{s−smax})` alone is WRONG,
/// and measurably so: that function is not `≈ s` at small s, it is
/// `s − 1.892e-4`. It therefore returns a NEGATIVE rescaled gradient for
/// `s < 1.892e-4` and is 19 % low at s = 0.001. A rescaling must satisfy
/// `s → 0 ⟹ s_scaled → 0`, because s = 0 is the uniform electron gas and the
/// enhancement factor has to reduce to its UEG limit exactly there. libxc's
/// `s < 1 → s` shortcut is enforcing precisely that constraint — it is not
/// sloppiness — at the cost of the discontinuity.
///
/// So the offset is subtracted back out:
///
/// ```text
/// f(s) = smax − ln(1 + e^{smax−s}) + ln(1 + e^{−smax})
/// ```
///
/// which is C^∞, branchless, exactly `f(0) = 0`, monotone, never negative,
/// and saturates at `smax + 1.892e-4` (a 2e-5 relative shift of a fitted
/// saturation constant). It is also CLOSER to libxc than the bare softplus
/// everywhere it matters: 1.9e-7 at s = 0.001 and 2.0e-5 at s = 0.1, against
/// 1.9e-4 and 2.1e-4 for the bare form.
///
/// The `smax − ln(1 + e^{smax−s})` ordering is used rather than the
/// algebraically identical `s − ln(1 + e^{s−smax})` because `s ≥ 0` bounds
/// its exponent argument by `smax = 8.57`, so overflow is impossible — the
/// only thing the `s > 15` branch ever protected against.
///
/// Consequence, recorded in docs/FUNCTIONALS.md: HSE06/HSE03/LC-ωPBE totals
/// no longer coincide with libxc's to the frozen-oracle tolerance, so those
/// entries are gated on internal identities (the s → 0 limit, continuity,
/// the ω → 0 and ω → ∞ screening limits) rather than on numerical agreement
/// with libxc.
fn wpbeh_scale_s<S: Scalar>(s: S) -> S {
    // Written as `s − softplus(s−smax) + softplus(−smax)` rather than the
    // algebraically identical `smax − softplus(smax−s) + softplus(−smax)`.
    // Both are the same function, but this ordering subtracts and adds the
    // SAME quantity at s = 0, so f(0) is exactly zero instead of the
    // −7.5e-16 of cancellation the other ordering leaves behind. A rescaled
    // gradient must never come out negative: downstream the enhancement
    // factor takes odd powers of s.
    let saturation = softplus(s - S::from_f64(WSMAX));
    let at_zero = softplus(S::from_f64(-WSMAX));
    s - saturation + at_zero
}

/// `ln(1 + e^x)`, without overflowing `exp` for large `x`.
///
/// For `x > 30` the identity `ln(1+e^x) = x + ln(1+e^{−x})` moves the
/// exponential to the safe side; the correction there is below 1e-13, so the
/// switch is continuous at working precision.
fn softplus<S: Scalar>(x: S) -> S {
    if x.value() > 30.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// The wPBEh enhancement factor F(w, s); w = ω/kFσ, s already rescaled
/// and floored at 1e-15.
fn wpbeh_enhancement<S: Scalar>(w: S, s: S) -> S {
    let one = S::from_f64(1.0);
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let s2 = s * s;
    let s4 = s2 * s2;
    let h = (S::from_f64(WHA[0]) * s2 + S::from_f64(WHA[1]) * s4)
        / (one
            + S::from_f64(WHA[2]) * s4
            + S::from_f64(WHA[3]) * s4 * s
            + S::from_f64(WHA[4]) * s4 * s2);
    let ff = S::from_f64(WFC1) * h + S::from_f64(WFC2);
    let s2h = s2 * h;
    let aux1 = S::from_f64(WD) + s2h;
    let aux2 = S::from_f64(9.0 / (4.0 * WA)) * s2h;
    let wv = w.value();
    let eb1 = if wv < WCUT { WEB1 } else { 2.0 };
    let w2 = w * w;
    let aux3 = aux1 + w2;
    let aux4 = s2h + S::from_f64(eb1) * w2;
    let aux5 = S::from_f64(9.0 / (4.0 * WA)) * aux4;
    let aux6 = S::from_f64(WD) + aux4;

    let eg = if s.value() > WEGSCUT {
        let a1_72 = aux1.powi(3) * aux1.sqrt();
        let ga = S::from_f64(sqrt_pi)
            * (S::from_f64(15.0 * WE)
                + S::from_f64(6.0 * WC) * (one + ff * s2) * aux1
                + S::from_f64(4.0 * WB) * aux1 * aux1
                + S::from_f64(8.0 * WA) * aux1.powi(3))
            / (S::from_f64(16.0) * a1_72)
            - S::from_f64(0.75 * std::f64::consts::PI * WA.sqrt()) * aux2.sqrt().erfcx();
        let gb = S::from_f64(15.0 * sqrt_pi) * s2 / (S::from_f64(16.0) * a1_72);
        -(S::from_f64(0.75 * std::f64::consts::PI) + ga) / gb
    } else {
        S::from_f64(WEGA[0]) + S::from_f64(WEGA[1]) * s2 + S::from_f64(WEGA[2]) * s4
    };

    let term2 = (aux1 * aux1 * S::from_f64(WB)
        + aux1 * S::from_f64(WC)
        + S::from_f64(2.0 * WE)
        + aux1 * s2 * S::from_f64(WC) * ff
        + S::from_f64(2.0) * s2 * eg)
        / (S::from_f64(2.0) * aux1.powi(3));
    let a3_52 = aux3 * aux3 * aux3.sqrt();
    let term3 = -w
        * (S::from_f64(4.0 * WB) * aux3 * aux3
            + S::from_f64(6.0 * WC) * aux3
            + S::from_f64(15.0 * WE)
            + S::from_f64(6.0 * WC) * aux3 * s2 * ff
            + S::from_f64(15.0) * s2 * eg)
        / (S::from_f64(8.0) * aux1 * a3_52);
    let term4 = -w
        * w2
        * (aux3 * S::from_f64(WC)
            + S::from_f64(5.0 * WE)
            + aux3 * s2 * S::from_f64(WC) * ff
            + S::from_f64(5.0) * s2 * eg)
        / (S::from_f64(2.0) * aux1 * aux1 * a3_52);
    let term5 = -w * w2 * w2 * (S::from_f64(WE) + s2 * eg) / (aux1.powi(3) * a3_52);

    let term1 = if wv > WCUT {
        S::from_f64(-0.5 * WA) * (-(aux5.e1_scaled()) + aux6.ln() - aux4.ln())
    } else {
        let sqrt_a = WA.sqrt();
        let np1 = S::from_f64(-1.5 * WEA[0] * sqrt_a) * w
            + S::from_f64(27.0 * WEA[2] / (8.0 * sqrt_a)) * w * w2
            - S::from_f64(243.0 * WEA[4] / (32.0 * sqrt_a.powi(3))) * w * w2 * w2
            + S::from_f64(2187.0 * WEA[6] / (128.0 * sqrt_a.powi(5))) * w * w2 * w2 * w2;
        let np2 = S::from_f64(-WA) + S::from_f64(9.0 * WEA[1] / 4.0) * w2
            - S::from_f64(81.0 * WEA[3] / (16.0 * WA)) * w2 * w2
            + S::from_f64(729.0 * WEA[5] / (64.0 * WA * WA)) * w2 * w2 * w2
            - S::from_f64(6561.0 * WEA[7] / (256.0 * WA.powi(3))) * w2 * w2 * w2 * w2;
        let t1 = S::from_f64(0.5)
            * (np1 * S::from_f64(std::f64::consts::PI) * aux5.sqrt().erfcx()
                - np2 * aux5.e1_scaled());
        let f2 = S::from_f64(0.5 * WEA[0] * sqrt_pi * WA) / aux6.sqrt();
        let f3 = S::from_f64(0.5 * WEA[1] * WA) / aux6;
        let f4 = S::from_f64(WEA[2] * sqrt_pi)
            * (S::from_f64(-9.0 / 8.0) / aux4.sqrt()
                + S::from_f64(0.25 * WA) / (aux6 * aux6.sqrt()));
        let f5 = S::from_f64(WEA[3] / 128.0)
            * (S::from_f64(-144.0) / aux4 + S::from_f64(64.0 * WA) / (aux6 * aux6));
        let a6_52 = aux6 * aux6 * aux6.sqrt();
        let a4_32 = aux4 * aux4.sqrt();
        let f6 = S::from_f64(WEA[4])
            * (S::from_f64(3.0 * sqrt_pi)
                * (S::from_f64(3.0) * a6_52 * (S::from_f64(9.0) * aux4 - S::from_f64(2.0 * WA))
                    + S::from_f64(4.0 * WA * WA) * a4_32))
            / (S::from_f64(32.0 * WA) * a6_52 * a4_32);
        let f7 = S::from_f64(WEA[5] / 32.0)
            * (S::from_f64(32.0 * WA) / aux6.powi(3)
                + (S::from_f64(-36.0) + S::from_f64(81.0 / WA) * s2h) / (aux4 * aux4));
        let a6_72 = aux6.powi(3) * aux6.sqrt();
        let a4_52 = aux4 * aux4 * aux4.sqrt();
        let f8 = S::from_f64(WEA[6])
            * (S::from_f64(-3.0 * sqrt_pi)
                * (S::from_f64(-40.0 * WA.powi(3)) * a4_52
                    + S::from_f64(9.0)
                        * a6_72
                        * (S::from_f64(27.0) * aux4 * aux4 - S::from_f64(6.0 * WA) * aux4
                            + S::from_f64(4.0 * WA * WA))))
            / (S::from_f64(128.0 * WA * WA) * a6_72 * a4_52);
        let a6_4 = aux6.powi(4);
        let f9 = (S::from_f64(324.0 * WEA[5] * eb1 * WA) * a6_4 * aux4
            + S::from_f64(WEA[7])
                * (S::from_f64(384.0 * WA.powi(3)) * aux4.powi(3)
                    + a6_4
                        * (S::from_f64(-729.0) * aux4 * aux4 + S::from_f64(324.0 * WA) * aux4
                            - S::from_f64(288.0 * WA * WA))))
            / (S::from_f64(128.0 * WA * WA) * a6_4 * aux4.powi(3));
        let t2t9 = f2 * w
            + f3 * w2
            + f4 * w * w2
            + f5 * w2 * w2
            + f6 * w * w2 * w2
            + f7 * w2 * w2 * w2
            + f8 * w * w2 * w2 * w2
            + f9 * w2 * w2 * w2 * w2;
        let t10 = S::from_f64(0.5 * WA) * (aux4 / aux6).ln();
        t1 + t2t9 + t10
    };

    S::from_f64(-8.0 / 9.0) * (term1 + term2 + term3 + term4 + term5)
}

/// wPBEh exchange energy density. theta: [omega].
pub(crate) fn wpbeh_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    wpbeh_e_scaled(ra, rb, saa, sab, sbb, th, wpbeh_scale_s, WPBEH_S_FLOOR)
}

/// `wpbeh_e` with the s-rescaling and its floor injected.
///
/// The rescaling is the ONLY place this implementation deliberately departs
/// from libxc (see `wpbeh_scale_s`), and the hole model downstream of it —
/// the H(s) fit, the EG expansion, the nine-term screened kernel, the spin
/// scaling — is unchanged. Parameterising it lets the frozen libxc reference
/// points keep gating all of that machinery, by evaluating them through
/// libxc's own rescaling, instead of being deleted along with the
/// convention. The project already keeps `#[cfg(test)]` reference
/// implementations for the same purpose in the ERI derivative path.
fn wpbeh_e_scaled<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    _sab: S,
    sbb: S,
    th: &[S],
    scale: fn(S) -> S,
    floor: f64,
) -> S {
    let omega = th[0];
    let x2s = 0.5 / (6.0 * std::f64::consts::PI * std::f64::consts::PI).cbrt();
    let kf_c = (6.0 * std::f64::consts::PI * std::f64::consts::PI).cbrt();
    let cx_spin = 0.75 * (3.0 / std::f64::consts::PI).cbrt() * 2.0_f64.cbrt();
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r43 = rho.powf(4.0 / 3.0);
        // Do not form √σ at all when σ is absent. The 1e-15 floor below
        // REPLACES the poisoned value with a clean constant, which is why the
        // first derivative survived — but replacing a value cannot un-poison
        // the intermediates the second-order dual already built from it.
        // Skipping the root entirely is what makes order 2 finite too.
        let s = if super::gga::sigma_is_absent(sig) {
            S::from_f64(floor)
        } else {
            let scaled = scale(S::from_f64(x2s) * sig.sqrt() / r43);
            if scaled.value() < floor {
                S::from_f64(floor)
            } else {
                scaled
            }
        };
        let w = omega / (S::from_f64(kf_c) * rho.cbrt());
        e = e - S::from_f64(cx_spin) * r43 * wpbeh_enhancement(w, s);
    }
    e
}

/// wPBEh SR exchange with range-separation parameter ω (ω = 0: full-range
/// PBE-hole exchange).
pub fn wpbeh_x(omega: f64) -> Box<dyn XcFunctional> {
    gga_driver!("wpbeh-x", Kind::Exchange, vec![omega], &["omega"], wpbeh_e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{Family, GridVars, XcOut};

    /// The s-rescaling must be smooth AND leave the uniform-gas limit alone.
    ///
    /// Both halves matter and they pull against each other. libxc short-
    /// circuits `s < 1 -> s`, which gets s -> 0 exactly right but is
    /// discontinuous by -5.141e-4 at s = 1 and by +1.616e-3 at s = 15. The
    /// bare Henderson softplus is smooth but maps s = 0 to -1.892e-4 --
    /// negative, and 19% low at s = 0.001 -- which breaks the UEG limit the
    /// enhancement factor has to reduce to. Only the offset-corrected form
    /// satisfies both, so both are asserted here.
    #[test]
    fn s_rescaling_is_smooth_and_preserves_the_uniform_gas_limit() {
        // Exact at the uniform electron gas.
        assert!(
            wpbeh_scale_s(0.0_f64).abs() < 1e-15,
            "s = 0 must map to 0 (uniform-gas limit), got {}",
            wpbeh_scale_s(0.0_f64)
        );
        // Never negative, and monotone increasing.
        let mut prev = f64::NEG_INFINITY;
        let mut s = 0.0_f64;
        while s < 40.0 {
            let v = wpbeh_scale_s(s);
            assert!(v >= 0.0, "s = {s} rescaled to a negative {v}");
            // Non-decreasing, not strictly increasing: the function
            // saturates, so consecutive samples MUST tie in f64 at large s.
            assert!(v >= prev, "not monotone at s = {s}");
            prev = v;
            s += 0.001;
        }
        // Continuous everywhere, checked hardest at the two points the old
        // short-circuits jumped: s = 1 and s = 15.
        for probe in [1.0_f64, 15.0, 0.5, 8.572844, 20.0] {
            let h = 1e-9;
            let lo = wpbeh_scale_s(probe - h);
            let hi = wpbeh_scale_s(probe + h);
            assert!(
                (hi - lo).abs() < 1e-8,
                "discontinuity at s = {probe}: {lo} vs {hi} (jump {})",
                hi - lo
            );
        }
        // Small s is the identity to first order: f(s)/s -> 1 as s -> 0.
        for probe in [1e-3_f64, 1e-2, 1e-1] {
            let ratio = wpbeh_scale_s(probe) / probe;
            assert!(
                (ratio - 1.0).abs() < 3e-4,
                "f({probe})/{probe} = {ratio}, but the rescaling must not disturb small s"
            );
        }
        // Saturates: bounded just above the fitted smax.
        assert!(wpbeh_scale_s(1e3_f64) < WSMAX + 1e-3);
        assert!(wpbeh_scale_s(1e3_f64) > WSMAX - 1e-9);
    }

    /// Raising the reduced-gradient floor must not move the energy.
    ///
    /// `WPBEH_S_FLOOR` was raised from libxc's 1e-15 to 1e-8 to stop a
    /// second-order dual overflowing `aux4^-7` when ω = 0. The justification
    /// was that the enhancement depends on the floored s only at O(s²) —
    /// which deserves a gate rather than an assertion, because at ω = 0 the
    /// enhancement ALSO contains `e1_scaled(aux5)` and `ln(aux4/aux6)`, both
    /// logarithmic in `aux4 ∝ s⁴`. Those two logarithms cancel analytically,
    /// leaving only the O(s²) dependence; this test is what confirms the
    /// cancellation actually happens in floating point.
    ///
    /// ω = 0 is the case that matters: it is the full-range wPBEh term that
    /// HSE06 and HSE03 carry, and the only one where `aux4` collapses to
    /// `s²h` with nothing else propping it up.
    #[test]
    fn the_reduced_gradient_floor_does_not_move_the_energy() {
        // Points with a genuinely tiny gradient, where the floor engages.
        let cases = [
            [0.5, 0.5, 1e-12, 1e-12, 1e-12],
            [0.31, 0.17, 1e-20, 0.0, 1e-20],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.02, 0.05, 1e-30, 0.0, 1e-30],
        ];
        for omega in [0.0_f64, 0.11, 0.4] {
            for v in &cases {
                let tight =
                    wpbeh_e_scaled(v[0], v[1], v[2], v[3], v[4], &[omega], wpbeh_scale_s, 1e-15);
                let loose =
                    wpbeh_e_scaled(v[0], v[1], v[2], v[3], v[4], &[omega], wpbeh_scale_s, 1e-8);
                assert!(
                    tight.is_finite() && loose.is_finite(),
                    "ω={omega} {v:?}: non-finite ({tight}, {loose})"
                );
                let scale = 1.0 + tight.abs();
                assert!(
                    (tight - loose).abs() < 1e-10 * scale,
                    "ω={omega} {v:?}: the s floor moved the energy density, {tight} (1e-15) vs \
                     {loose} (1e-8). The ln(aux4) terms in t1 and t10 are supposed to cancel, \
                     leaving only O(s²) — if this fires, they do not."
                );
            }
        }
    }
    fn eval(f: &dyn XcFunctional, v: [f64; 5]) -> f64 {
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
        out.prepare(1, 0, Family::Gga);
        f.eval_add(&vars, 0, 1.0, &mut out).unwrap();
        out.e[0]
    }

    /// Pinned against libxc HYB_GGA_XC_LC_WPBE_WHS − GGA_C_PBE (ω = 0.4),
    /// i.e. gga_x_hjs_pbe at ω = 0.4 (tools/oracle/probe_hjs.py).
    #[test]
    fn hjs_pbe_matches_libxc_reference_points() {
        let cases: [([f64; 5], f64); 4] = [
            ([0.3, 0.1, 0.0525, 0.0301, 0.0233], -0.154262651246),
            ([1.2, 0.9, 0.29, 0.055, 0.1725], -1.562755119818),
            ([0.02, 0.05, 0.0005, 0.0006, 0.0009], -0.010569638375),
            ([0.5, 0.5, 1e-12, 1e-12, 1e-12], -0.537562126874),
        ];
        let f = hjs_pbe_x(0.4);
        for (v, want) in cases {
            let e = eval(f.as_ref(), v);
            assert!((e - want).abs() < 1e-9, "HJS ωPBE {e} vs libxc {want}");
        }
    }

    #[test]
    fn hjs_pbesol_matches_exchange_hole_quadrature() {
        // Direct composite Simpson integration of the spherically averaged
        // hole in Eq. (35), with the erfc kernel from HSEsol Eq. (5).
        // Refined 16384 to 131072 panels over y=0..320 before freezing values.
        // The shorter y=0..48 interval missed the weak-screening tail at s=0.2.
        let cases = [
            (0.20, 0.045, 0.902_174_383_944_926),
            (0.80, 0.070, 0.910_342_185_822_401),
            (1.40, 0.045, 1.080_082_306_112_336),
            (2.70, 0.125, 1.133_871_500_916_527),
            (4.80, 0.490, 0.641_064_044_171_742),
        ];
        let mut error: f64 = 0.0;
        for (s, nu, expected) in cases {
            let actual = hjs_enhancement(s, nu, &HJS_PBESOL_A, &HJS_PBESOL_B);
            error = error.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() < 3e-11,
                "s={s}, nu={nu}: {actual} vs {expected}"
            );
        }
        println!("HJS PBEsol maximum enhancement error vs hole quadrature = {error:.6e}");
    }

    #[test]
    fn hjs_pbesol_screening_limits_and_spin_scaling() {
        // The unscreened hole fit reproduces the parent enhancement within
        // the precision of the published rational fit, not exactly.
        let parent = [
            (0.8, 1.071_942_285_107_097),
            (1.4, 1.185_996_883_999_811),
            (2.7, 1.424_647_887_323_944),
        ];
        for (s, full_pbesol) in parent {
            let enhancement = |nu| hjs_enhancement(s, nu, &HJS_PBESOL_A, &HJS_PBESOL_B);
            assert!((enhancement(0.0) - full_pbesol).abs() < 3e-5);
            assert!(enhancement(0.1) < enhancement(0.0));
            assert!(enhancement(1.0) < enhancement(0.1));
            assert!((0.0..1e-6).contains(&enhancement(1000.0)));
        }
        let srx = hjs_pbesol_x(0.11);
        let unpolarized = eval(srx.as_ref(), [0.42, 0.42, 0.21, 0.21, 0.21]);
        let fully_polarized = eval(srx.as_ref(), [0.42, 0.0, 0.21, 0.0, 0.0]);
        assert!((unpolarized - 2.0 * fully_polarized).abs() < 2e-16);
    }

    #[test]
    fn hsesol_terms_match_the_published_energy_expression() {
        use crate::xc::{resolve, FunctionalSpec};

        let hsesol = resolve(&FunctionalSpec::Combined("hsesol".into())).unwrap();
        assert_eq!(hsesol.hybrid.cx, 0.25);
        assert_eq!(hsesol.hybrid.clr, -0.25);
        assert_eq!(hsesol.hybrid.omega, 0.11);
        assert_eq!(hsesol.parts.len(), 3);

        let pbe_x = crate::xc::gga::pbesol_x();
        let pbe_c = crate::xc::gga::pbesol_c();
        let (s, nu, sr_enhancement) = (1.4_f64, 0.045_f64, 1.080_082_306_112_336_f64);
        let kf = 0.11 / nu;
        let rho = kf.powi(3) / (6.0 * std::f64::consts::PI.powi(2));
        let sigma = (2.0 * s * kf * rho).powi(2);
        let input = [rho, 0.0, sigma, 0.0, 0.0];
        let sr_x =
            -0.75 * (6.0 / std::f64::consts::PI).cbrt() * rho.powf(4.0 / 3.0) * sr_enhancement;
        let expected = [
            eval(pbe_x.as_ref(), input),
            -0.25 * sr_x,
            eval(pbe_c.as_ref(), input),
        ];
        let actual: Vec<_> = hsesol
            .parts
            .iter()
            .map(|(w, part)| w * eval(part.as_ref(), input))
            .collect();
        for (value, want) in actual.iter().zip(expected) {
            assert!((value - want).abs() < 2e-12, "{actual:?} vs {expected:?}");
        }
        println!("HSEsol semilocal energy density terms = {actual:?}");
    }

    /// Pinned against libxc gga_x_wpbeh at ω = 0 (direct), ω = 0.11
    /// (extracted from HSE06) and ω = 0.4 (extracted from LC_WPBE);
    /// tools/oracle/probe_wpbeh.py. Points cover the EG polynomial branch
    /// (s < 0.08), the direct branch, the s-rescaling regime (s ≈ 9.6),
    /// and the large-w branch (w > 14, last ω = 0.4 row).
    #[test]
    fn wpbeh_matches_libxc_reference_points() {
        #[allow(clippy::type_complexity)]
        let cases: [([f64; 5], [f64; 3]); 6] = [
            (
                [0.3, 0.1, 0.0525, 0.0301, 0.0233],
                [-0.232267373348, -0.208218527158, -0.153928468815],
            ),
            (
                [1.2, 0.9, 0.29, 0.055, 0.1725],
                [-1.996283547512, -1.868931699230, -1.562128532965],
            ),
            (
                [0.02, 0.05, 0.0005, 0.0006, 0.0009],
                [-0.022608996836, -0.018503606321, -0.010518025540],
            ),
            (
                [0.5, 0.5, 1e-12, 1e-12, 1e-12],
                [-0.738558762838, -0.678357536505, -0.537298512122],
            ),
            (
                [3.0, 3.0, 2.8561, 0.0, 2.8561],
                [-8.055754252692, -7.689392815968, -6.782068978254],
            ),
            (
                [1e-4, 2e-4, 1.2e-7, 0.5e-7, 4.1e-7],
                [-2.6884161e-5, -1.0986386e-5, -8.80991e-7],
            ),
        ];
        for (i, omega) in [0.0, 0.11, 0.4].into_iter().enumerate() {
            for (v, want) in &cases {
                // Evaluated through libxc's OWN s-rescaling, not ours.
                //
                // `wpbeh_scale_s` deliberately departs from libxc: libxc's
                // short-circuits are discontinuous by -5.141e-4 at s = 1 and
                // +1.616e-3 at s = 15, and this project treats a
                // discontinuity as a defect even where the reference
                // implementation shares it. That single choice shifts
                // HSE06/HSE03/LC-wPBE totals by ~1.5e-4 Ha.
                //
                // Everything else in the hole model is unchanged, so these
                // frozen points still gate it: the H(s) rational fit, the EG
                // small-s quartic, the nine-term screened kernel, the erfcx
                // and E1 branches, and the spin scaling. Only the rescaling
                // is exempt, and it has its own gate in
                // `s_rescaling_is_smooth_and_preserves_the_uniform_gas_limit`.
                let e = wpbeh_e_scaled(
                    v[0],
                    v[1],
                    v[2],
                    v[3],
                    v[4],
                    &[omega],
                    wpbeh_scale_s_libxc,
                    1e-15,
                );
                // The last-row references come through subtractive
                // extraction of tiny values — gate absolutely.
                let tol = if want[i].abs() < 1e-4 { 5e-11 } else { 1e-9 };
                assert!(
                    (e - want[i]).abs() < tol,
                    "wPBEh(ω={omega}) {e} vs libxc {} (case {v:?})",
                    want[i]
                );
            }
        }
    }

    /// libxc 7.0.0's `s_scaling_2` (`maple/scaling.mpl:26-29`), kept ONLY as
    /// a test reference so the frozen wPBEh points still gate the hole model.
    ///
    /// Not used in production: it is discontinuous at s = 1 and s = 15. See
    /// `wpbeh_scale_s` for what ships instead and why.
    fn wpbeh_scale_s_libxc<S: Scalar>(s: S) -> S {
        let sv = s.value();
        if sv < 1.0 {
            s
        } else if sv > 15.0 {
            S::from_f64(WSMAX)
        } else {
            s - (s - S::from_f64(WSMAX)).exp().ln_1p()
        }
    }

    fn eval_ord(f: &dyn XcFunctional, v: [f64; 5], order: usize) -> XcOut {
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

    /// First and second derivatives against central differences (both
    /// holes, screened and unscreened).
    #[test]
    fn derivatives_match_finite_differences() {
        let p = [0.3, 0.1, 0.0525, 0.0301, 0.0233];
        let h = 1e-6;
        for f in [hjs_pbe_x(0.4), wpbeh_x(0.11), wpbeh_x(0.0)] {
            let out = eval_ord(f.as_ref(), p, 2);
            // exchange: no σαβ dependence — check ρα, σαα, ρβ, σββ.
            for (k, an) in [
                (0usize, out.vrho[0][0]),
                (1, out.vrho[1][0]),
                (2, out.vsigma[0][0]),
                (4, out.vsigma[2][0]),
            ] {
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
            // second derivatives vs FD of the analytic first derivatives.
            let d1 = |v: [f64; 5], slot: usize| -> f64 {
                let o = eval_ord(f.as_ref(), v, 1);
                match slot {
                    0 => o.vrho[0][0],
                    _ => o.vsigma[0][0],
                }
            };
            for (k, slot, an) in [
                (0usize, 0usize, out.v2rho2[0][0]),
                (2, 0, out.v2rhosigma[0][0]),
                (2, 2, out.v2sigma2[0][0]),
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

    /// Exact spin scaling of the two SR holes at the ω values the registry
    /// actually instantiates. HSE03's screened hole (ω = 0.15·2^{1/3}) and
    /// the bare `--x hjs-pbe-x` (ω = 0.11) are the two that no frozen
    /// energy reaches; the relation holds channel by channel, so it catches
    /// a spin index reused across the two loop iterations of the hole
    /// evaluation — which is exactly how a screened exchange goes wrong
    /// only in the open-shell case.
    #[test]
    fn sr_holes_obey_spin_scaling() {
        let p = [0.3, 0.1, 0.0525, 0.0301, 0.0233];
        let holes = [
            wpbeh_x(0.0),
            wpbeh_x(0.11),
            wpbeh_x(0.15 * 2f64.cbrt()),
            wpbeh_x(0.4),
            hjs_pbe_x(0.11),
            hjs_pbe_x(0.4),
        ];
        for f in holes {
            let mixed = eval(f.as_ref(), p);
            let a = eval(f.as_ref(), [p[0], p[0], p[2], p[2], p[2]]);
            let b = eval(f.as_ref(), [p[1], p[1], p[4], p[4], p[4]]);
            let want = 0.5 * (a + b);
            assert!(
                (mixed - want).abs() < 1e-13 * (1.0 + want.abs()),
                "{} (ω = {}): {mixed} vs ½(Ex[2ρα] + Ex[2ρβ]) = {want}",
                f.name(),
                f.params()[0]
            );
        }
    }

    /// The screening is monotone in ω: more screening removes more
    /// short-range exchange, and at ω → ∞ nothing is left. HSE03 sits
    /// between the pinned ω = 0.11 and ω = 0.4 wPBEh references, so its
    /// hole value is bracketed by two libxc-pinned numbers even though the
    /// functional itself has no frozen energy.
    #[test]
    fn wpbeh_screening_is_monotone_in_omega() {
        let p = [0.3, 0.1, 0.0525, 0.0301, 0.0233];
        // Ascending: HSE03's ω_HF (0.15/√2) sits just below HSE06's 0.11,
        // and its ω_PBE (0.15·2^{1/3}) between that and LC-ωPBE's 0.4.
        let omegas = [0.0, 0.15 / 2f64.sqrt(), 0.11, 0.15 * 2f64.cbrt(), 0.4, 4.0];
        let mut previous = f64::NEG_INFINITY;
        for w in omegas {
            let e = eval(wpbeh_x(w).as_ref(), p);
            assert!(e < 0.0, "wPBEh(ω = {w}) must be negative, got {e}");
            assert!(
                e > previous,
                "wPBEh is not monotone in ω: e({w}) = {e} is below the \
                 previous, less screened value {previous}"
            );
            previous = e;
        }
        assert!(
            eval(wpbeh_x(1e3).as_ref(), p).abs() < 1e-6,
            "wPBEh does not vanish under complete screening"
        );
    }

    /// erfcx / e1_scaled: the two independent branch algorithms must agree
    /// at their junctions.
    #[test]
    fn scaled_special_functions_branch_continuity() {
        // erfcx: series-with-libm branch vs continued fraction at x = 5.
        let series = f64::exp(25.0) * libm::erfc(5.0);
        let mut c = 0.0;
        for k in (1..=60).rev() {
            c = (k as f64 * 0.5) / (5.0 + c);
        }
        let cf = 1.0 / (std::f64::consts::PI.sqrt() * (5.0 + c));
        assert!(
            ((series - cf) / cf).abs() < 1e-13,
            "erfcx branches {series} vs {cf}"
        );
        // e1_scaled: power series vs continued fraction at x = 1.5.
        const EULER_GAMMA: f64 = 0.5772156649015329;
        let x = 1.5;
        let mut sum = -EULER_GAMMA - f64::ln(x);
        let mut term = 1.0;
        for k in 1..=40 {
            term *= -x / k as f64;
            sum -= term / k as f64;
        }
        let series = sum * f64::exp(x);
        let mut t = 0.0;
        for k in (1..=100).rev() {
            let kf = k as f64;
            t = kf / (1.0 + kf / (x + t));
        }
        let cf = 1.0 / (x + t);
        assert!(
            ((series - cf) / cf).abs() < 1e-12,
            "e1_scaled branches {series} vs {cf}"
        );
    }
}
