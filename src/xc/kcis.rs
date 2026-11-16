// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! KCIS correlation for an electron gas with a local gap, including the
//! spin-resolved one-electron self-correlation subtraction.
//! See Kurth, Perdew and Blaha (1999), Appendix A, and Toulouse, Savin and
//! Adamo (2002), Eqs. (2), (A3)–(A12), DOI 10.1063/1.1521432.

use super::common::{f_zeta, wigner_seitz, RHO_CUTOFF};
use super::lda::{one_channel_only, pw92_g, PW92_PARAMS};
use super::traits::{Kind, XcFunctional};
use crate::dual::Scalar;

const PARAMETER_NAMES: &[&str] = &[
    "beta",
    "large_gap_prefactor",
    "gap_a1",
    "gap_a2",
    "gap_a3",
    "gap_b3",
    "gap_b4",
    "gap_b5",
    "gap_b6",
    "gap_b7",
    "polarized_c1_scale",
    "polarized_c2_scale",
    "polarized_c3_scale",
];

/// Gap and reduced density gradient for a fixed density and gradient.
struct GapGas<S> {
    rs: S,
    t2: S,
    e0: S,
    c1_gap: S,
    c2_gap: S,
    c3_gap2: S,
}

impl<S: Scalar> GapGas<S> {
    fn new(rho: S, sigma: S, p: &[S]) -> Self {
        let one = S::from_f64(1.0);
        let rs = wigner_seitz(rho);
        let kf = (S::from_f64(3.0 * std::f64::consts::PI.powi(2)) * rho).cbrt();
        let gap = sigma / (S::from_f64(8.0) * rho * rho);
        let t2 = S::from_f64(std::f64::consts::FRAC_PI_2) * gap / kf;
        let e0 = uniform_correlation(rs, 0);

        let root = rs.sqrt();
        let leading = p[2] * rs * root;
        let slope = leading / (one + p[3] * root + p[4] * rs + leading);
        let curvature = rs * rs * rs * (p[5] + rs * (p[6] + rs * (p[7] + rs * (p[8] + rs * p[9]))));
        let large_gap = p[1] * kf * kf;
        let two = S::from_f64(2.0);
        let denominator = two * (large_gap * slope - e0 * e0);
        // The rational fit matches the zero-gap slope and curvature and
        // tends to -large_gap/G. This fixes c1 = -large_gap*c3 and the
        // sign of c3 (Fabiano et al., DOI 10.1021/ct500073b, Eq. A4).
        let c3 = (e0 * curvature - two * slope * slope) / denominator;
        let c2 = (two * e0 * slope - large_gap * curvature) / denominator;
        Self {
            rs,
            t2,
            e0,
            c1_gap: -large_gap * c3 * gap,
            c2_gap: c2 * gap,
            c3_gap2: c3 * gap * gap,
        }
    }

    fn endpoint(&self, polarized: bool, p: &[S]) -> S {
        let one = S::from_f64(1.0);
        let (uniform, gradient_scale, c1, c2, c3) = if polarized {
            (
                uniform_correlation(self.rs, 6),
                S::from_f64(2.0_f64.powf(-1.0 / 3.0)),
                p[10],
                p[11],
                p[12],
            )
        } else {
            (self.e0, one, one, one, one)
        };
        let gga = uniform / (one + p[0] * (-gradient_scale * self.t2 / uniform).ln_1p());
        (gga + c1 * self.c1_gap) / (one + c2 * self.c2_gap + c3 * self.c3_gap2)
    }
}

fn uniform_correlation<S: Scalar>(rs: S, offset: usize) -> S {
    let [a, a1, b1, b2, b3, b4]: [S; 6] =
        std::array::from_fn(|i| S::from_f64(PW92_PARAMS[offset + i]));
    pw92_g(rs, a, a1, b1, b2, b3, b4)
}

fn correlation<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, ta: S, tb: S, p: &[S]) -> S {
    let one = S::from_f64(1.0);
    let ratio = |rho, sigma, tau| sigma / (S::from_f64(8.0) * rho * tau);
    if one_channel_only(ra, rb) {
        let (rho, sigma, tau) = if ra.value() > rb.value() {
            (ra, saa, ta)
        } else {
            (rb, sbb, tb)
        };
        return rho * GapGas::new(rho, sigma, p).endpoint(true, p) * (one - ratio(rho, sigma, tau));
    }

    let rho = ra + rb;
    let gas = GapGas::new(rho, saa + S::from_f64(2.0) * sab + sbb, p);
    let unpolarized = gas.endpoint(false, p);
    let zeta = (ra - rb) / rho;
    let mut energy = rho * (unpolarized + f_zeta(zeta) * (gas.endpoint(true, p) - unpolarized));
    // Same-spin subtraction uses that spin's density, gradient and gap.
    for (rho, sigma, tau) in [(ra, saa, ta), (rb, sbb, tb)] {
        if rho.value() >= RHO_CUTOFF {
            energy = energy
                - ratio(rho, sigma, tau) * rho * GapGas::new(rho, sigma, p).endpoint(true, p);
        }
    }
    energy
}

pub fn kcis_c() -> Box<dyn XcFunctional> {
    mgga_driver!(
        "kcis",
        Kind::Correlation,
        vec![
            0.066725,
            0.06483,
            0.04953,
            1.07924,
            0.07928,
            -0.02504,
            0.007026,
            -0.001268,
            0.0001136,
            -0.0000038421,
            0.7,
            1.5,
            2.59,
        ],
        PARAMETER_NAMES,
        correlation,
        true
    )
}
