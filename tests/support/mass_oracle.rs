// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Independent exponential LM mass formulas, with closed confluent limits.
use dft_rs::dual::Scalar;

fn f<S: Scalar>(x: S, t: S) -> S {
    (S::from_f64(1.0) + (x / t).exp()).recip()
}
fn derivatives<S: Scalar>(x: S, t: S) -> [S; 3] {
    let one = S::from_f64(1.0);
    let p = f(x, t);
    let q = p * (one - p);
    [
        -q / t,
        q * (one - p.scale(2.0)) / t.powi(2),
        -q * (one - p.scale(6.0) + p.powi(2).scale(6.0)) / t.powi(3),
    ]
}
fn dd<S: Scalar>(b: S, y: S, t: S) -> S {
    let d = b - y;
    // Test paths selecting this branch are exactly confluent at every order.
    if (d.value() / t.value()).abs() < 1e-12 {
        derivatives(y, t)[1].scale(0.5)
    } else {
        (derivatives(b, t)[0] - (f(b, t) - f(y, t)) / d) / d
    }
}
fn dd_prime<S: Scalar>(b: S, y: S, t: S) -> S {
    let d = b - y;
    if (d.value() / t.value()).abs() < 1e-12 {
        derivatives(y, t)[2].scale(1.0 / 3.0)
    } else {
        derivatives(b, t)[1] / d - derivatives(b, t)[0].scale(2.0) / d.powi(2)
            + (f(b, t) - f(y, t)).scale(2.0) / d.powi(3)
    }
}
pub fn numerator<S: Scalar>(x: S, y: S, w: S, t: S) -> S {
    let one = S::from_f64(1.0);
    let n = ((w / t).exp() - one).recip();
    let term = |y| (n + f(x, t)) * dd(x - w, y, t) + (n + f(-x, t)) * dd(x + w, y, t);
    term(y) + term(-y)
}
/// At x=0 this formula assumes a path that keeps x identically zero.
/// A crossing path instead divides numerator and tanh Taylor series explicitly.
pub fn factor<S: Scalar>(x: S, y: S, w: S, t: S) -> S {
    let one = S::from_f64(1.0);
    if x.value() == 0.0 {
        let n = ((w / t).exp() - one).recip();
        let fp0 = -one / t.scale(4.0);
        let derivative = |y| {
            fp0 * (dd(-w, y, t) - dd(w, y, t))
                + (n + one.scale(0.5)) * (dd_prime(-w, y, t) + dd_prime(w, y, t))
        };
        t.scale(2.0) * (derivative(y) + derivative(-y))
    } else {
        numerator(x, y, w, t) / (one - f(x, t).scale(2.0))
    }
}
