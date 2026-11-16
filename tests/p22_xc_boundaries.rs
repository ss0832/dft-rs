// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Behaviour of every registry functional on the BOUNDARY of its domain.
//!
//! The interior of the density domain is already gated hard (frozen libxc
//! reference points, per-order finite differences). The boundary is not, and
//! it is where a functional written with the wrong algebra fails silently:
//!
//! - an empty spin channel (`ρβ = 0`) — a hydrogen atom, H₂⁺, any
//!   high-multiplicity state;
//! - a vanishing density gradient (`σ = 0`) — the centre of a closed-shell
//!   atom, any point of exact symmetry;
//! - the iso-orbital limit `τ = τ_W`, where meta-GGA switching functions sit
//!   on their branch point.
//!
//! Forward-mode AD makes these sharp. A term like `(1−ζ)^{2/3}` or `√σ` is
//! perfectly finite at the boundary but has an INFINITE derivative there, and
//! `DualN` multiplies that infinity by a zero seed: `0·∞ = NaN`. The NaN then
//! contaminates EVERY tangent slot of that dual, not just the one whose
//! variable is at the boundary — so a σ-derivative singularity silently
//! destroys `vrho`, i.e. the Kohn–Sham potential itself.
//!
//! # What "correct" means here, and why it is not simply "finite"
//!
//! Two different statements get confused at a boundary, so this file keeps
//! them apart ([`Expect`]):
//!
//! - the value is finite AND equals the true one-sided limit ([`Expect::Finite`]);
//! - the true one-sided limit DIVERGES, and we return a finite number under a
//!   stated convention instead ([`Expect::FiniteByConvention`]).
//!
//! The second case is real and unavoidable. `∂²/∂ρβ²` of `ρβ^{4/3}` is
//! `(4/9)ρβ^{−2/3} → ∞` as `ρβ → 0`: Slater exchange genuinely has a divergent
//! same-channel curvature at an empty channel. The convention this project
//! adopts — and which `src/xc/b97.rs` already used for the same-spin
//! correlation channels before this file existed — is that **below
//! `RHO_CUTOFF` a spin channel is ABSENT**: the functional collapses to its
//! one-channel (ferromagnetic) form and every derivative with respect to the
//! absent channel is zero. That is the only choice that keeps the Fock build
//! and the response kernel well posed, and it is exact for the energy, since
//! `E(ρ, 0) ≡ E_ferro(ρ)`.
//!
//! What is NEVER acceptable is `NaN` or `±∞`, in any block, at any order.

use dft_rs::xc::{
    resolve, FunctionalSpec, GridVars, ResolvedFunctional, XcOut, COMBINED_NAMES,
    CORRELATION_NAMES, EXCHANGE_NAMES,
};

/// One point of the seven-dimensional grid-variable domain.
#[derive(Clone, Copy)]
struct Point {
    name: &'static str,
    /// (ρα, ρβ, σαα, σαβ, σββ, τα, τβ)
    v: [f64; 7],
}

/// The boundary points every functional is swept over.
///
/// τ is kept at or above the von Weizsäcker bound `τ_W = σ/(8ρ)` so that the
/// meta-GGA iso-orbital variable stays inside its physical range; a point
/// that violated it would be testing the input clamp rather than the
/// functional.
fn boundary_points() -> Vec<Point> {
    vec![
        // The interior control. If this one fails, nothing else means anything.
        Point {
            name: "interior",
            v: [0.31, 0.17, 0.042, 0.026, 0.021, 0.29, 0.13],
        },
        // A hydrogen atom: one spin channel is exactly, bit-for-bit empty.
        Point {
            name: "beta-empty",
            v: [0.31, 0.0, 0.042, 0.0, 0.0, 0.29, 0.0],
        },
        // The centre of a closed-shell atom: the density is flat.
        Point {
            name: "sigma-zero",
            v: [0.31, 0.17, 0.0, 0.0, 0.0, 0.29, 0.13],
        },
        // Both at once — a one-electron s density at its own nucleus.
        Point {
            name: "beta-empty+sigma-zero",
            v: [0.31, 0.0, 0.0, 0.0, 0.0, 0.29, 0.0],
        },
        // Iso-orbital limit τ = τ_W exactly: meta-GGA α = 0.
        Point {
            name: "tau-weizsaecker",
            v: [0.20, 0.20, 0.04, 0.04, 0.04, 0.025, 0.025],
        },
        // Small but above RHO_CUTOFF (1e-13): the density tail.
        Point {
            name: "small-rho",
            v: [1e-11, 1e-11, 1e-24, 1e-24, 1e-24, 1e-13, 1e-13],
        },
        // Fully polarised the other way, to catch a channel swap.
        Point {
            name: "alpha-empty",
            v: [0.0, 0.31, 0.0, 0.0, 0.042, 0.0, 0.29],
        },
    ]
}

/// Every functional the registry can build, as (label, resolved).
fn all_functionals() -> Vec<(String, ResolvedFunctional)> {
    let mut out = Vec::new();
    for &name in EXCHANGE_NAMES {
        let f = resolve(&FunctionalSpec::Split {
            x: name.into(),
            c: "none".into(),
        })
        .unwrap_or_else(|e| panic!("exchange `{name}` failed to resolve: {e}"));
        out.push((format!("x:{name}"), f));
    }
    // There is no `--x none`: the registry offers a null correlation but not
    // a null exchange, so a correlation functional cannot be resolved on its
    // own. Pairing it with Slater is harmless for this sweep — Slater is
    // swept independently as `x:slater`, so a non-finite value appearing here
    // and NOT there belongs to the correlation part.
    for &name in CORRELATION_NAMES {
        let f = resolve(&FunctionalSpec::Split {
            x: "slater".into(),
            c: name.into(),
        })
        .unwrap_or_else(|e| panic!("correlation `{name}` failed to resolve: {e}"));
        out.push((format!("c:{name}+slater"), f));
    }
    for &name in COMBINED_NAMES {
        let f = resolve(&FunctionalSpec::Combined(name.into()))
            .unwrap_or_else(|e| panic!("combined `{name}` failed to resolve: {e}"));
        out.push((format!("xc:{name}"), f));
    }
    out
}

/// Names and contents of every output block that `order` populates.
fn blocks(out: &XcOut) -> Vec<(&'static str, &[f64])> {
    let mut b: Vec<(&'static str, &[f64])> = vec![("e", &out.e)];
    let named = [
        ("vrho", &out.vrho[..]),
        ("vsigma", &out.vsigma[..]),
        ("vtau", &out.vtau[..]),
        ("v2rho2", &out.v2rho2[..]),
        ("v2rhosigma", &out.v2rhosigma[..]),
        ("v2sigma2", &out.v2sigma2[..]),
        ("v2rhotau", &out.v2rhotau[..]),
        ("v2sigmatau", &out.v2sigmatau[..]),
        ("v2tau2", &out.v2tau2[..]),
        ("v3rho3", &out.v3rho3[..]),
        ("v3rho2sigma", &out.v3rho2sigma[..]),
        ("v3rhosigma2", &out.v3rhosigma2[..]),
        ("v3sigma3", &out.v3sigma3[..]),
        ("v3rho2tau", &out.v3rho2tau[..]),
        ("v3rhosigmatau", &out.v3rhosigmatau[..]),
        ("v3rhotau2", &out.v3rhotau2[..]),
        ("v3sigma2tau", &out.v3sigma2tau[..]),
        ("v3sigmatau2", &out.v3sigmatau2[..]),
        ("v3tau3", &out.v3tau3[..]),
    ];
    for (label, group) in named {
        for slot in group {
            if !slot.is_empty() {
                b.push((label, slot.as_slice()));
            }
        }
    }
    b
}

/// Evaluate one functional at one point and report every non-finite output.
///
/// A panic is caught and reported like any other defect: it is a worse
/// outcome than a NaN, not a reason to abandon the sweep. The whole point of
/// this test is the FULL matrix of which functionals break where — stopping
/// at the first casualty would hide the systemic pattern.
fn nonfinite(func: &ResolvedFunctional, p: &Point, order: usize) -> Vec<String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (ra, rb) = ([p.v[0]], [p.v[1]]);
        let (saa, sab, sbb) = ([p.v[2]], [p.v[3]], [p.v[4]]);
        let (ta, tb) = ([p.v[5]], [p.v[6]]);
        let vars = GridVars {
            rho: [&ra, &rb],
            sigma: Some([&saa, &sab, &sbb]),
            tau: Some([&ta, &tb]),
        };
        let mut out = XcOut::default();
        if func.eval(&vars, order, &mut out).is_err() {
            // An explicit refusal is a legitimate answer; a NaN is not.
            return Vec::new();
        }
        let mut bad = Vec::new();
        for (label, slot) in blocks(&out) {
            for &value in slot {
                if !value.is_finite() {
                    bad.push(format!("{label}={value}"));
                }
            }
        }
        bad.sort();
        bad.dedup();
        bad
    }));
    match result {
        Ok(bad) => bad,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic payload".to_string());
            vec![format!("PANIC: {msg}")]
        }
    }
}

/// No functional, at any implemented derivative order, may return a NaN or an
/// infinity anywhere on the boundary of its domain.
///
/// This is the gate the ζ = ±1 and σ = 0 branches exist to satisfy. It is
/// deliberately a single test over the whole registry rather than one test per
/// functional: the failure mode is systemic (a shared driver, a shared helper
/// like `f_zeta`, a shared `√σ` idiom), so the useful output is the full
/// matrix of which functionals break where, not the first one that trips.
#[test]
fn no_functional_returns_nan_or_infinity_on_its_domain_boundary() {
    let funcs = all_functionals();
    let points = boundary_points();
    let mut failures: Vec<String> = Vec::new();

    for (label, func) in &funcs {
        for p in &points {
            for order in 0..=3usize {
                let bad = nonfinite(func, p, order);
                if !bad.is_empty() {
                    failures.push(format!(
                        "  {label:<18} {:<22} order {order}: {}",
                        p.name,
                        bad.join(", ")
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} functional/point/order combinations returned a non-finite value.\n\
         A NaN here is not a boundary subtlety: forward-mode AD spreads it from \n\
         the one singular variable into EVERY tangent slot, so a σ-derivative \n\
         blow-up silently destroys vrho — the Kohn-Sham potential itself.\n\n\
         {}\n\n\
         swept {} functionals x {} points x 4 orders",
        failures.len(),
        failures.join("\n"),
        funcs.len(),
        points.len()
    );
}

fn eval_point(func: &ResolvedFunctional, point: [f64; 7], order: usize) -> XcOut {
    let (ra, rb) = ([point[0]], [point[1]]);
    let (saa, sab, sbb) = ([point[2]], [point[3]], [point[4]]);
    let (ta, tb) = ([point[5]], [point[6]]);
    let vars = GridVars {
        rho: [&ra, &rb],
        sigma: Some([&saa, &sab, &sbb]),
        tau: Some([&ta, &tb]),
    };
    let mut out = XcOut::default();
    func.eval(&vars, order, &mut out).unwrap();
    out
}

fn symmetric_pair_index(a: usize, b: usize, width: usize) -> usize {
    debug_assert!(a <= b && b < width);
    a * width - a * (a.saturating_sub(1)) / 2 + (b - a)
}

fn symmetric_triple3_index(a: usize, b: usize, c: usize) -> usize {
    debug_assert!(a <= b && b <= c && c < 3);
    match (a, b, c) {
        (0, 0, 0) => 0,
        (0, 0, 1) => 1,
        (0, 0, 2) => 2,
        (0, 1, 1) => 3,
        (0, 1, 2) => 4,
        (0, 2, 2) => 5,
        (1, 1, 1) => 6,
        (1, 1, 2) => 7,
        (1, 2, 2) => 8,
        (2, 2, 2) => 9,
        _ => unreachable!(),
    }
}

fn hessian_entry(out: &XcOut, p: usize, q: usize) -> f64 {
    let (p, q) = if p <= q { (p, q) } else { (q, p) };
    match (p, q) {
        (0..=1, 0..=1) => out.v2rho2[symmetric_pair_index(p, q, 2)][0],
        (0..=1, 2..=4) => out.v2rhosigma[3 * p + (q - 2)][0],
        (2..=4, 2..=4) => out.v2sigma2[symmetric_pair_index(p - 2, q - 2, 3)][0],
        (0..=1, 5..=6) => out.v2rhotau[2 * p + (q - 5)][0],
        (2..=4, 5..=6) => out.v2sigmatau[2 * (p - 2) + (q - 5)][0],
        (5..=6, 5..=6) => out.v2tau2[symmetric_pair_index(p - 5, q - 5, 2)][0],
        _ => unreachable!(),
    }
}

fn third_entry(out: &XcOut, p: usize, q: usize, r: usize) -> f64 {
    let mut index = [p, q, r];
    index.sort_unstable();
    let [p, q, r] = index;
    match (p, q, r) {
        (0..=1, 0..=1, 0..=1) => out.v3rho3[p + q + r][0],
        (0..=1, 0..=1, 2..=4) => out.v3rho2sigma[3 * symmetric_pair_index(p, q, 2) + (r - 2)][0],
        (0..=1, 2..=4, 2..=4) => out.v3rhosigma2[6 * p + symmetric_pair_index(q - 2, r - 2, 3)][0],
        (2..=4, 2..=4, 2..=4) => out.v3sigma3[symmetric_triple3_index(p - 2, q - 2, r - 2)][0],
        (0..=1, 0..=1, 5..=6) => out.v3rho2tau[2 * symmetric_pair_index(p, q, 2) + (r - 5)][0],
        (0..=1, 2..=4, 5..=6) => out.v3rhosigmatau[6 * p + 2 * (q - 2) + (r - 5)][0],
        (0..=1, 5..=6, 5..=6) => out.v3rhotau2[3 * p + symmetric_pair_index(q - 5, r - 5, 2)][0],
        (2..=4, 2..=4, 5..=6) => {
            out.v3sigma2tau[2 * symmetric_pair_index(p - 2, q - 2, 3) + (r - 5)][0]
        }
        (2..=4, 5..=6, 5..=6) => {
            out.v3sigmatau2[3 * (p - 2) + symmetric_pair_index(q - 5, r - 5, 2)][0]
        }
        (5..=6, 5..=6, 5..=6) => out.v3tau3[(p - 5) + (q - 5) + (r - 5)][0],
        _ => unreachable!(),
    }
}

/// Every analytic third-derivative block is the derivative of the matching
/// analytic Hessian block. This checks all 4 LDA, 35 GGA, and 84 meta-GGA
/// symmetric tensor components for every registered functional at a smooth,
/// interior point. Richardson extrapolation keeps the numerical reference
/// independent of the nested-dual implementation without weakening the gate.
#[test]
fn third_derivatives_match_finite_differences_of_second_derivatives() {
    let base = [0.43, 0.27, 0.07, 0.018, 0.04, 0.35, 0.22];
    let h = 2.0e-4;
    let mut failures = Vec::new();
    for (label, func) in all_functionals() {
        let nvar = match func.family {
            dft_rs::xc::Family::Lda => 2,
            dft_rs::xc::Family::Gga => 5,
            dft_rs::xc::Family::MetaGga => 7,
        };
        let analytic = eval_point(&func, base, 3);
        for r in 0..nvar {
            let derivative = |step: f64, p: usize, q: usize| {
                let mut plus = base;
                let mut minus = base;
                plus[r] += step;
                minus[r] -= step;
                (hessian_entry(&eval_point(&func, plus, 2), p, q)
                    - hessian_entry(&eval_point(&func, minus, 2), p, q))
                    / (2.0 * step)
            };
            for p in 0..nvar {
                for q in p..nvar {
                    let coarse = derivative(h, p, q);
                    let fine = derivative(0.5 * h, p, q);
                    let fd = (4.0 * fine - coarse) / 3.0;
                    let got = third_entry(&analytic, p, q, r);
                    let scale = got.abs().max(fd.abs()).max(1.0);
                    let err = (got - fd).abs();
                    if !got.is_finite() || !fd.is_finite() || err > 2.0e-6 * scale {
                        failures.push(format!(
                            "  {label:<18} d3[{p},{q},{r}]={got:.12e}, fd={fd:.12e}, err={err:.3e}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "third-XC derivative finite-difference failures:\n{}",
        failures.join("\n")
    );
}

/// The public third-order entry points must not inherit the caller's small
/// platform stack. GGA/meta-GGA nested dual towers are evaluated through the
/// stack-sized scoped worker both through `ResolvedFunctional::eval` and a
/// direct public trait-object `eval_add` call.
#[test]
fn third_order_public_api_is_safe_from_a_one_mib_stack() {
    let worker = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            let point = [0.43, 0.27, 0.07, 0.018, 0.04, 0.35, 0.22];
            for name in ["pbe", "tpss", "r2scan", "wb97m-v", "m06-2x"] {
                let functional = resolve(&FunctionalSpec::Combined(name.to_string())).unwrap();
                let combined = eval_point(&functional, point, 3);
                assert!(combined.e[0].is_finite(), "{name}: combined energy");
                assert!(
                    combined
                        .v3rho3
                        .iter()
                        .flatten()
                        .all(|value| value.is_finite()),
                    "{name}: combined third derivative"
                );

                let (ra, rb) = ([point[0]], [point[1]]);
                let (saa, sab, sbb) = ([point[2]], [point[3]], [point[4]]);
                let (ta, tb) = ([point[5]], [point[6]]);
                let vars = GridVars {
                    rho: [&ra, &rb],
                    sigma: Some([&saa, &sab, &sbb]),
                    tau: Some([&ta, &tb]),
                };
                for (_, part) in &functional.parts {
                    let mut direct = XcOut::default();
                    direct.prepare(1, 3, part.family());
                    part.eval_add(&vars, 3, 1.0, &mut direct).unwrap();
                    assert!(direct.e[0].is_finite(), "{name}: direct energy");
                    assert!(
                        direct
                            .v3rho3
                            .iter()
                            .flatten()
                            .all(|value| value.is_finite()),
                        "{name}: direct third derivative"
                    );
                }
            }
        })
        .unwrap();
    worker.join().unwrap();
}

/// Functionals whose σ-derivative at σ = 0 is a stated CONVENTION rather than
/// the true one-sided limit.
///
/// These three depend on σ through combinations that are genuinely
/// non-analytic at zero — odd powers of `√σ` — so `∂e/∂σ` really does have a
/// nonzero (or divergent) one-sided limit that this implementation does not
/// return. It reports zero instead, under the absent-gradient convention in
/// `xc::gga::sigma_is_absent`. That is a defensible choice and it is what
/// keeps the Fock build finite, but it is NOT the limit, and the difference
/// should never be discovered by accident.
///
/// These component functionals are affected, and every combined entry built
/// on one of them inherits it:
///
/// | component | why σ = 0 is not analytic | inherited by |
/// |---|---|---|
/// | P86 | gradient term is `σ·exp(−c√σ)`, i.e. `t² − c t³ + …` in `t = √σ` | bp86, b3p86 |
/// | HJS ωPBE / ωPBEsol | the H(s) rational fit carries ODD powers of `s ∝ √σ` | lc-wpbe-hjs, hsesol |
/// | r²SCAN-C | `s` and `t` are built from `√(σ/ρ^{8/3})` | r2scan, r2scan0, r2scanh, r2scan50 |
/// | SCAN-C, rSCAN-C | the same `√σ` construction as r²SCAN-C | scan, scan0, rscan |
/// | wPBEh | `s` is floored at `WPBEH_S_FLOOR` to stop an order-2 overflow | hse03, hse06, lc-wpbe, lrc-wpbeh |
/// | G96 | the enhancement grows like `x^{3/2}` with `x ∝ √σ` | g96lyp |
///
/// B88 is deliberately absent: it LOOKS like it belongs — it was written
/// with `√σ` and produced NaN — but it is genuinely analytic in σ, so
/// rewriting it in `u = x²` recovered the TRUE derivative rather than a
/// convention. Likewise PBE and revPBE, which use `s²` directly and never
/// needed touching. If any of those three appears here, an analytic rewrite
/// has silently regressed into a convention, which is exactly the
/// substitution this test exists to prevent.
const SIGMA_DERIVATIVE_IS_A_CONVENTION: &[&str] = &[
    // components
    "c:p86+slater",
    "c:r2scan+slater",
    "c:rscan+slater",
    "c:scan+slater",
    "x:g96",
    "x:hjs-pbe-x",
    // composites that inherit it
    "xc:b3p86",
    "xc:bp86",
    "xc:g96lyp",
    "xc:hse03",
    "xc:hse06",
    "xc:hsesol",
    "xc:lc-wpbe",
    "xc:lc-wpbe-hjs",
    "xc:lrc-wpbeh",
    "xc:r2scan",
    "xc:r2scan0",
    "xc:r2scan50",
    "xc:r2scanh",
    "xc:rscan",
    "xc:scan",
    "xc:scan0",
];

/// Pin exactly where the absent-gradient convention is in force.
///
/// The finiteness sweep above proves nothing is NaN; it cannot tell a true
/// one-sided limit from a convention that returns zero. This does, by
/// comparing the reported `∂e/∂σ` at σ = 0 against a one-sided finite
/// difference taken from σ > 0.
///
/// Two failure directions, both worth catching:
/// - a functional appears here that was not declared — someone added a
///   convention site, or an analytic rewrite regressed into one;
/// - a declared functional stops appearing — the convention became
///   unnecessary (good), and the declaration is now stale.
#[test]
fn the_absent_gradient_convention_is_in_force_exactly_where_declared() {
    let ra = 0.31;
    let rb = 0.17;
    let ta = 0.29;
    let tb = 0.13;

    let e_at = |func: &ResolvedFunctional, sigma: f64| -> f64 {
        let (aa, bb) = ([ra], [rb]);
        let (s0, s1, s2) = ([sigma], [0.0], [sigma]);
        let (t0, t1) = ([ta], [tb]);
        let vars = GridVars {
            rho: [&aa, &bb],
            sigma: Some([&s0, &s1, &s2]),
            tau: Some([&t0, &t1]),
        };
        let mut out = XcOut::default();
        func.eval(&vars, 0, &mut out).unwrap();
        out.e[0]
    };
    let vsigma_at_zero = |func: &ResolvedFunctional| -> f64 {
        let (aa, bb) = ([ra], [rb]);
        let (s0, s1, s2) = ([0.0], [0.0], [0.0]);
        let (t0, t1) = ([ta], [tb]);
        let vars = GridVars {
            rho: [&aa, &bb],
            sigma: Some([&s0, &s1, &s2]),
            tau: Some([&t0, &t1]),
        };
        let mut out = XcOut::default();
        func.eval(&vars, 1, &mut out).unwrap();
        if out.vsigma[0].is_empty() {
            0.0
        } else {
            out.vsigma[0][0]
        }
    };

    let mut observed: Vec<String> = Vec::new();
    for (label, func) in all_functionals() {
        if func.family == dft_rs::xc::Family::Lda {
            continue; // no σ dependence at all
        }
        let reported = vsigma_at_zero(&func);
        // One-sided slope in σαα. Both αα and ββ move together above, so
        // the finite difference carries both channels; compare like with
        // like by summing the two reported same-spin σ derivatives.
        let reported_pair = {
            let (aa, bb) = ([ra], [rb]);
            let (s0, s1, s2) = ([0.0], [0.0], [0.0]);
            let (t0, t1) = ([ta], [tb]);
            let vars = GridVars {
                rho: [&aa, &bb],
                sigma: Some([&s0, &s1, &s2]),
                tau: Some([&t0, &t1]),
            };
            let mut out = XcOut::default();
            func.eval(&vars, 1, &mut out).unwrap();
            if out.vsigma[0].is_empty() {
                0.0
            } else {
                out.vsigma[0][0] + out.vsigma[2][0]
            }
        };
        let _ = reported;
        let h = 1e-9;
        let slope = (e_at(&func, h) - e_at(&func, 0.0)) / h;
        let scale = 1.0 + slope.abs() + reported_pair.abs();
        let is_convention = (slope - reported_pair).abs() > 1e-4 * scale;
        if is_convention {
            observed.push(label);
        }
    }
    observed.sort();
    let mut declared: Vec<String> = SIGMA_DERIVATIVE_IS_A_CONVENTION
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    declared.sort();

    let undeclared: Vec<&String> = observed.iter().filter(|o| !declared.contains(o)).collect();
    let stale: Vec<&String> = declared.iter().filter(|d| !observed.contains(d)).collect();
    assert!(
        undeclared.is_empty() && stale.is_empty(),
        "the absent-gradient convention is not in force where declared.\n\
         undeclared (returning a convention, not the limit, without saying so): {undeclared:?}\n\
         stale (declared but now returning the true limit): {stale:?}\n\
         observed: {observed:?}\n declared: {declared:?}"
    );
}

/// The energy and the OCCUPIED channel's potential must be continuous as the
/// other channel empties.
///
/// This is the test that pins the branch condition itself. Switching to the
/// ferromagnetic form on `ζ == 1.0` exactly would pass the finiteness gate
/// above and still be wrong: an SCF routinely produces `ρβ = 1e-300`, which is
/// not `ζ == 1` and would take the general branch straight back into the
/// singularity. Branching on `ρβ <= RHO_CUTOFF` is what makes the limit and
/// the boundary agree.
#[test]
fn the_empty_channel_limit_is_continuous() {
    let funcs = all_functionals();
    // Well inside the cutoff (1e-13), spanning the range an SCF actually
    // reaches, including a denormal-adjacent value.
    let approaches = [1e-14, 1e-16, 1e-20, 1e-100, 1e-300];
    let ra = 0.31;
    let saa = 0.042;
    let ta = 0.29;

    for (label, func) in &funcs {
        let at_zero = |rb: f64| -> (f64, f64) {
            let (aa, bb) = ([ra], [rb]);
            let (s0, s1, s2) = ([saa], [0.0], [0.0]);
            let (t0, t1) = ([ta], [0.0]);
            let vars = GridVars {
                rho: [&aa, &bb],
                sigma: Some([&s0, &s1, &s2]),
                tau: Some([&t0, &t1]),
            };
            let mut out = XcOut::default();
            func.eval(&vars, 1, &mut out).unwrap();
            (out.e[0], out.vrho[0][0])
        };

        let (e0, v0) = at_zero(0.0);
        assert!(
            e0.is_finite() && v0.is_finite(),
            "{label}: exactly-empty beta channel is already non-finite \
             (e={e0}, vrho_alpha={v0})"
        );
        for &rb in &approaches {
            let (e, v) = at_zero(rb);
            assert!(
                e.is_finite() && v.is_finite(),
                "{label}: rho_beta={rb:e} gave e={e}, vrho_alpha={v}"
            );
            let scale = 1.0 + e0.abs();
            assert!(
                (e - e0).abs() < 1e-9 * scale,
                "{label}: energy density jumps between rho_beta=0 ({e0}) and \
                 rho_beta={rb:e} ({e}) — the empty-channel branch does not \
                 agree with the limit it is supposed to represent"
            );
            let vscale = 1.0 + v0.abs();
            assert!(
                (v - v0).abs() < 1e-9 * vscale,
                "{label}: the OCCUPIED channel's potential jumps between \
                 rho_beta=0 ({v0}) and rho_beta={rb:e} ({v})"
            );
        }
    }
}

/// The single-spin branch must be the limit of the two-spin expression: the
/// energy density with a small but representable second channel (above the
/// density cutoff, so the two-spin code runs) approaches the value with an
/// empty channel. The test above only approaches from below the cutoff, where
/// both evaluations take the single-spin branch, so it cannot see a wrong
/// constant in that branch. (It missed P86, whose single-spin spin factor
/// d(1) was 2^(-1/2) instead of 2^(1/3).) The second channel carries its own
/// small gradient and kinetic energy density so that the meta-GGA indicators
/// stay physical. The remaining difference is the analytic approach to the
/// limit: of order (rho_beta / rho)^(2/3) for the PBE-type phi(zeta) and
/// rho_beta^(1/3) for OP correlation (e ~ rho_a rho_b / beta_ab^2 with
/// beta_ab ~ rho_b^(1/3)), and mixtures of such terms. Over rho_beta =
/// 1e-6 ... 1e-12 the relative deviation must reach round-off (below 1e-12)
/// or decrease monotonically by at least a factor of 20 (OP, the slowest,
/// decreases by 99); a wrong constant in the single-spin branch leaves a
/// plateau with a ratio near one (P86 before the fix).
#[test]
fn the_single_spin_branch_is_the_limit_of_the_two_spin_expression() {
    let funcs = all_functionals();
    let approaches = [1e-6, 1e-8, 1e-10, 1e-12];
    let ra = 0.31;
    let saa = 0.042;
    let ta = 0.29;
    let mut failures = Vec::new();
    for (label, func) in &funcs {
        let energy = |rb: f64| -> f64 {
            // A second channel with the same reduced gradient and indicator
            // as the first: sigma and tau scale as rho^(8/3) and rho^(5/3).
            let ratio = rb / ra;
            let (aa, bb) = ([ra], [rb]);
            let sbb = saa * ratio.powf(8.0 / 3.0);
            let sab = (saa * sbb).sqrt();
            let (s0, s1, s2) = ([saa], [sab], [sbb]);
            let (t0, t1) = ([ta], [ta * ratio.powf(5.0 / 3.0)]);
            let vars = GridVars {
                rho: [&aa, &bb],
                sigma: Some([&s0, &s1, &s2]),
                tau: Some([&t0, &t1]),
            };
            let mut out = XcOut::default();
            func.eval(&vars, 0, &mut out).unwrap();
            out.e[0]
        };
        let limit = energy(0.0);
        let deviations: Vec<f64> = approaches
            .iter()
            .map(|&rb| (energy(rb) - limit).abs() / (1.0 + limit.abs()))
            .collect();
        let last = deviations[approaches.len() - 1];
        if last < 1e-12 {
            continue;
        }
        let monotone = deviations.windows(2).all(|w| w[1] <= w[0]);
        let ratio = last / deviations[0];
        println!("{label}: relative deviations {deviations:?}, last/first {ratio:.3e}");
        if !monotone || ratio > 0.05 {
            failures.push(format!(
                "  {label:<18} relative deviations {deviations:?} (last/first {ratio:.3e})"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "single-spin branch is not the limit of the two-spin expression:\n{}",
        failures.join("\n")
    );
}
