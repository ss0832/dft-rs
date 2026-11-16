// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent TM/revTM reference in doubled-density atomic units.
//! Equations (7), (9), (10)--(12) of doi:10.1103/PhysRevLett.117.073001
//! and Section II of doi:10.1021/acs.jpca.9b02921 define the models.

use dft_rs::dual::Scalar;
use std::f64::consts::PI;

pub fn terms<S: Scalar>(point: [S; 7], revised: bool) -> [S; 2] {
    let [a, b, aa, ab, bb, ta, tb] = point;
    let n = a + b;
    let value = S::from_f64;
    let one = value(1.0);
    let (mut x, mut c) = (value(0.0), value(0.0));
    for (spin, sigma, tau) in [(a, aa, ta), (b, bb, tb)] {
        if spin.value() < 1e-13 {
            continue;
        }
        let rho = value(2.0) * spin;
        let grad2 = value(4.0) * sigma;
        let tau = value(2.0) * tau;
        let k = (value(3.0 * PI * PI) * rho).cbrt();
        let uniform = value(0.3) * rho * k * k;
        let von_weizsacker = grad2 / (value(8.0) * rho);
        let p = grad2 / (value(2.0) * rho * k).powi(2);
        let z = if tau.value() == 0.0 {
            value(0.0)
        } else {
            von_weizsacker / tau
        };
        let overlap = (tau - von_weizsacker) / uniform;
        let b = value(if revised { 0.4 } else { 0.0 });
        let q = value(0.45) * (overlap - one) / (one + b * overlap * (overlap - one)).sqrt()
            + value(2.0 / 3.0) * p;
        let lambda: f64 = 0.6866;
        let y = value((2.0 * lambda - 1.0).powi(2)) * p;
        let f = (one + value(700.0 / 27.0) * y + value(79.873) * y.powi(2)).powf(0.1);
        let laplacian = value(6.0) * (tau - uniform - grad2 / (value(72.0) * rho));
        let r = (uniform - tau + value(0.5 * (lambda * lambda - lambda + 0.5)) * laplacian)
            / uniform
            + value(595.0 / 54.0) * y;
        let compact = f.powi(-2) + value(7.0 / 9.0) * r * f.powi(-4);
        let extended = (one
            + value(100.0 / 81.0) * p
            + value(500.0 / 729.0) * p.powi(2)
            + value(1460.0 / 2025.0) * q.powi(2)
            - value(146.0 / 135.0) * q * z * (one - z))
            .powf(0.1);
        let weight = z.powi(2) * (one + value(3.0) * z) / (one + z.powi(3)).powi(2);
        x = x - value(3.0 / (8.0 * PI)) * rho * k * (extended + weight * (compact - extended));
    }
    if a.value() < 1e-13 && b.value() < 1e-13 {
        return [x, c];
    }
    let sigma = aa + value(2.0) * ab + bb;
    let zeta = (a - b) / n;
    let tau = ta + tb;
    let unbounded_z = sigma / (value(8.0) * n * tau);
    let z = if unbounded_z.value() > 1.0 + 8.0 * f64::EPSILON {
        one
    } else {
        unbounded_z
    };
    let pbe = reference_pbe(n, zeta, sigma, revised);
    let c0 = value(0.1) * zeta.powi(2) + value(0.32) * zeta.powi(4);
    let (c_factor, self_energy) = if zeta.value().abs() == 1.0 {
        (c0, pbe)
    } else {
        // xi^2 comes from |grad(n_up/n)|^2 and the local Fermi wavevector.
        let mixture_gradient =
            (b.powi(2) * aa - value(2.0) * a * b * ab + a.powi(2) * bb) / n.powi(4);
        let xi2 = mixture_gradient / (value(3.0 * PI * PI) * n).powf(2.0 / 3.0);
        let spin_gradient =
            value(0.5) * ((one + zeta).powf(-4.0 / 3.0) + (one - zeta).powf(-4.0 / 3.0));
        let c_factor = c0 / (one + xi2 * spin_gradient).powi(4);
        let select = |rho, sig| {
            let ferro = reference_pbe(rho, one, sig, revised);
            if ferro.value() >= pbe.value() {
                ferro
            } else {
                pbe
            }
        };
        (c_factor, (a * select(a, aa) + b * select(b, bb)) / n)
    };
    let reduced = pbe + z.powi(2) * (c_factor * pbe - (one + c_factor) * self_energy);
    c = n * reduced * (one + value(2.8) * reduced * z.powi(3));
    [x, c]
}

fn reference_pbe<S: Scalar>(rho: S, zeta: S, sigma: S, revised: bool) -> S {
    let value = S::from_f64;
    let one = value(1.0);
    let rs = (value(3.0 / (4.0 * PI)) / rho).cbrt();
    let length = rs.sqrt();
    let pw92 = |[a, alpha, b1, b2, b3, b4]: [f64; 6]| {
        let denominator = value(2.0 * a)
            * length
            * (value(b1) + length * (value(b2) + length * (value(b3) + length * value(b4))));
        -value(2.0 * a) * (one + value(alpha) * rs) * denominator.recip().ln_1p()
    };
    let ferro = pw92([0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517]);
    let (lda, phi) = if zeta.value().abs() == 1.0 {
        (ferro, value(2.0f64.powf(-1.0 / 3.0)))
    } else {
        let u = (one + zeta).powf(2.0 / 3.0);
        let v = (one - zeta).powf(2.0 / 3.0);
        let spin_mixing = u * u + v * v - value(2.0);
        let una = pw92([0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294]);
        let stiffness = pw92([0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671]);
        let z4 = zeta.powi(4);
        let lda = una
            + spin_mixing
                * ((ferro - una) * z4 / value(2.0f64.powf(4.0 / 3.0) - 2.0)
                    - value(9.0 / 8.0) * stiffness * (one - z4));
        (lda, (u + v) * value(0.5))
    };
    let beta = value(0.06672455060314922);
    let beta = if revised {
        beta * (one + value(0.1) * rs) / (one + value(0.1778) * rs)
    } else {
        beta
    };
    let gamma = value(0.031090690869654895);
    let screening2 = value(4.0 / PI) * (value(3.0 * PI * PI) * rho).cbrt();
    let qt = sigma / (value(4.0) * phi.powi(2) * rho.powi(2) * screening2);
    let coefficient = beta / gamma;
    let c = gamma * phi.powi(3);
    let a = coefficient / ((-lda / c).exp() - one);
    let rational = one + a * qt;
    lda + c * (coefficient * qt * rational / (rational + (a * qt).powi(2))).ln_1p()
}
