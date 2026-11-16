// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Legacy GGA correlation functionals: P86 (Perdew, PRB 33, 8822 (1986))
//! on the PZ81 local parameterization, and PW91 correlation
//! (Perdew-Wang 1991) on PW92 - the pieces needed by BP86 and B3PW91.
//!
//! Both share the density-dependent coefficient
//!   C(rs) = 0.001667 + (0.002568 + a rs + b rs^2)
//!           / (1 + g rs + d rs^2 + 1e4 b rs^3),
//! a = 0.023266, b = 7.389e-6, g = 8.723, d = 0.472.
//!
//! Fully polarized channels (one spin below cutoff) route through ferro
//! branches: the zeta interpolations and the P86 d(zeta) factor have
//! singular second dual derivatives at zeta = 1 (THEORY.md).
//! Verified pointwise against libxc gga_c_p86 / gga_c_pw91.

use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

use super::common::RHO_CUTOFF;

fn c_rs<S: Scalar>(rs: S) -> S {
    let one = S::from_f64(1.0);
    S::from_f64(0.001667)
        + (S::from_f64(0.002568) + S::from_f64(0.023266) * rs + S::from_f64(7.389e-6) * rs * rs)
            / (one
                + S::from_f64(8.723) * rs
                + S::from_f64(0.472) * rs * rs
                + S::from_f64(1.0e4 * 7.389e-6) * rs * rs * rs)
}

/// PZ81 epsilon for one branch (para: idx 0, ferro: idx 1).
pub(crate) fn pz81_branch<S: Scalar>(rs: S, ferro: bool) -> S {
    let (a, b, c, d, gam, b1, b2) = if ferro {
        (0.01555, -0.0269, 0.0007, -0.0048, -0.0843, 1.3981, 0.2611)
    } else {
        (0.0311, -0.048, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334)
    };
    if rs.value() < 1.0 {
        let ln = rs.ln();
        S::from_f64(a) * ln + S::from_f64(b) + S::from_f64(c) * rs * ln + S::from_f64(d) * rs
    } else {
        S::from_f64(gam) / (S::from_f64(1.0) + S::from_f64(b1) * rs.sqrt() + S::from_f64(b2) * rs)
    }
}

use super::common::f_zeta;

/// P86 correlation energy density. theta: [] (constants fixed).
pub(crate) fn p86_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, _th: &[S]) -> S {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let have_a = ra.value() >= RHO_CUTOFF;
    let have_b = rb.value() >= RHO_CUTOFF;
    if !have_a && !have_b {
        return zero;
    }
    let (rho, sig, eps, dfac) = if have_a && have_b {
        let rho = ra + rb;
        let sig = saa + S::from_f64(2.0) * sab + sbb;
        let rs = super::common::wigner_seitz(rho);
        let zeta = (ra - rb) / rho;
        let ep = pz81_branch(rs, false);
        let ef = pz81_branch(rs, true);
        let eps = ep + f_zeta(zeta) * (ef - ep);
        // d = 2^{1/3} sqrt[((1+z)/2)^{5/3} + ((1-z)/2)^{5/3}].
        let half = S::from_f64(0.5);
        let d = S::from_f64(2.0f64.powf(1.0 / 3.0))
            * (((one + zeta) * half).powf(5.0 / 3.0) + ((one - zeta) * half).powf(5.0 / 3.0))
                .sqrt();
        (rho, sig, eps, d)
    } else {
        let (rho, sig) = if have_a { (ra, saa) } else { (rb, sbb) };
        let rs = super::common::wigner_seitz(rho);
        let eps = pz81_branch(rs, true);
        // d(zeta = 1) = 2^{1/3} sqrt[1^{5/3} + 0^{5/3}] = 2^{1/3}, the limit of
        // the two-channel expression above. (An earlier version used
        // 2^{-1/2}, which made the gradient term 2^{5/6} times too large
        // wherever only one spin channel carries density.)
        (rho, sig, eps, S::from_f64(2.0f64.powf(1.0 / 3.0)))
    };
    let rs = super::common::wigner_seitz(rho);
    let cc = c_rs(rs);
    if super::gga::sigma_is_absent(sig) {
        // The gradient correction is σ·exp(−c√σ), which vanishes at σ = 0.
        // Returning the local part in closed form avoids evaluating √σ at
        // zero — where `DualN::sqrt` builds an infinite derivative that the
        // zero seed turns into NaN across every tangent slot, vrho included.
        // P86 cannot be rewritten to be analytic here: in t = √σ the term is
        // t² − c t³ + …, so its second σ-derivative genuinely diverges. This
        // is the absent-gradient convention, not an exact limit.
        return rho * eps;
    }
    let grad = sig.sqrt();
    let phi = S::from_f64(1.745 * 0.11 * 0.004235) / cc * grad / rho.powf(7.0 / 6.0);
    rho * eps + (-phi).exp() * cc * sig / (dfac * rho.powf(4.0 / 3.0))
}

/// PW91's LOCAL base: LDA_C_PW with the PAPER digits AND the truncated
/// f''(0) = 1.709921 of the original paper (libxc convention for
/// GGA_C_PW91 - identified by probing the sigma -> 0 limit).
fn pw91_local_eps<S: Scalar>(ra: S, rb: S, pw: &[S]) -> S {
    let rho = ra + rb;
    let rs = super::common::wigner_seitz(rho);
    let z = (ra - rb) / rho;
    let e0 = super::lda::pw92_g(rs, pw[0], pw[1], pw[2], pw[3], pw[4], pw[5]);
    let e1 = super::lda::pw92_g(rs, pw[6], pw[7], pw[8], pw[9], pw[10], pw[11]);
    let neg_ac = super::lda::pw92_g(rs, pw[12], pw[13], pw[14], pw[15], pw[16], pw[17]);
    let one = S::from_f64(1.0);
    let fz = ((one + z).powf(4.0 / 3.0) + (one - z).powf(4.0 / 3.0) - S::from_f64(2.0))
        / S::from_f64(2.0f64.powf(4.0 / 3.0) - 2.0);
    let z4 = z * z * z * z;
    e0 - neg_ac * fz * S::from_f64(1.0 / 1.709921) * (one - z4) + (e1 - e0) * fz * z4
}

fn pw91_local_eps_ferro<S: Scalar>(rho: S, pw: &[S]) -> S {
    let rs = super::common::wigner_seitz(rho);
    super::lda::pw92_g(rs, pw[6], pw[7], pw[8], pw[9], pw[10], pw[11])
}

/// PW91 correlation energy density. theta: [pw92_paper x 18].
pub(crate) fn pw91_c_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    let zero = S::from_f64(0.0);
    let one = S::from_f64(1.0);
    let pw = &th[0..18];
    let have_a = ra.value() >= RHO_CUTOFF;
    let have_b = rb.value() >= RHO_CUTOFF;
    if !have_a && !have_b {
        return zero;
    }
    let (rho, sig, eps, g) = if have_a && have_b {
        let rho = ra + rb;
        let sig = saa + S::from_f64(2.0) * sab + sbb;
        let eps = pw91_local_eps(ra, rb, pw);
        let zeta = (ra - rb) / rho;
        let g = ((one + zeta).powf(2.0 / 3.0) + (one - zeta).powf(2.0 / 3.0)) * S::from_f64(0.5);
        (rho, sig, eps, g)
    } else {
        let (rho, sig) = if have_a { (ra, saa) } else { (rb, sbb) };
        let eps = pw91_local_eps_ferro(rho, pw);
        (rho, sig, eps, S::from_f64(2.0f64.powf(2.0 / 3.0) * 0.5))
    };
    let kf = (S::from_f64(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).powf(1.0 / 3.0);
    let ks = (S::from_f64(4.0 / std::f64::consts::PI) * kf).sqrt();
    let t2 = sig / (S::from_f64(4.0) * g * g * ks * ks * rho * rho);
    let rs = super::common::wigner_seitz(rho);
    let alpha = 0.09;
    let cc0 = 0.004235;
    let cx = -0.001667;
    let nu = (16.0 / std::f64::consts::PI) * (3.0 * std::f64::consts::PI.powi(2)).powf(1.0 / 3.0);
    let beta = nu * cc0;
    let g3 = g * g * g;
    let two_a_b = S::from_f64(2.0 * alpha / beta);
    let expo = (-(S::from_f64(2.0 * alpha) * eps) / (g3 * S::from_f64(beta * beta))).exp();
    let aa = two_a_b / (expo - one);
    let num = one + aa * t2;
    let den = one + aa * t2 + aa * aa * t2 * t2;
    let h0 = g3 * S::from_f64(beta * beta / (2.0 * alpha)) * (one + two_a_b * t2 * num / den).ln();
    let ksq = ks * ks / (kf * kf);
    // PW91 uses the Rasolt-Geldart '86 Pade for C_xc (quadratic/quadratic;
    // NO cubic denominator term - this differs from P86's C(rs)), and the
    // H1 offset is C_xc(0) = 0.002568.
    let c_xc =
        (S::from_f64(0.002568) + S::from_f64(0.023266) * rs + S::from_f64(7.389e-6) * rs * rs)
            / (one + S::from_f64(8.723) * rs + S::from_f64(0.472) * rs * rs);
    let h1 = S::from_f64(nu)
        * (c_xc - S::from_f64(0.002568) - S::from_f64(3.0 * cx / 7.0))
        * g3
        * t2
        * (-(S::from_f64(100.0) * g3 * g * ksq * t2)).exp();
    rho * (eps + h0 + h1)
}

pub fn p86() -> Box<dyn XcFunctional> {
    gga_driver!("p86", Kind::Correlation, vec![], &[], p86_e)
}

pub fn pw91_c() -> Box<dyn XcFunctional> {
    let pw_paper = super::lda::PW92_PARAMS;
    gga_driver!(
        "pw91",
        Kind::Correlation,
        pw_paper.to_vec(),
        &[
            "A_0", "a1_0", "b1_0", "b2_0", "b3_0", "b4_0", "A_1", "a1_1", "b1_1", "b2_1", "b3_1",
            "b4_1", "A_a", "a1_a", "b1_a", "b2_a", "b3_a", "b4_a"
        ],
        pw91_c_e
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xc::{Family, GridVars, XcOut};

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

    /// P86 and PW91-C pinned pointwise against libxc at three points
    /// (PySCF probe 2026-08-04).
    #[test]
    fn p86_pw91_match_libxc_reference_points() {
        let sig = |g: [f64; 3], h: [f64; 3]| g[0] * h[0] + g[1] * h[1] + g[2] * h[2];
        let cases: [([f64; 2], [f64; 3], [f64; 3], f64, f64); 3] = [
            (
                [0.3, 0.1],
                [0.1, 0.1, 0.15],
                [0.08, 0.05, 0.12],
                -0.021362903844,
                -0.021666204262,
            ),
            (
                [1.2, 0.9],
                [0.3, -0.2, 0.4],
                [0.25, 0.1, -0.3],
                -0.159599968297,
                -0.160923198206,
            ),
            (
                [0.02, 0.05],
                [0.01, 0.0, 0.02],
                [0.02, 0.01, -0.01],
                -0.003170438825,
                -0.003166745362,
            ),
        ];
        let fp86 = p86();
        let fpw91 = pw91_c();
        for (r, ga, gb, want_p86, want_pw91) in cases {
            let v = [r[0], r[1], sig(ga, ga), sig(ga, gb), sig(gb, gb)];
            let e1 = eval(fp86.as_ref(), v);
            let e2 = eval(fpw91.as_ref(), v);
            assert!((e1 - want_p86).abs() < 1e-9, "P86 {e1} vs {want_p86}");
            assert!((e2 - want_pw91).abs() < 1e-9, "PW91 {e2} vs {want_pw91}");
        }
    }
}
