// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! LDA functionals: Slater exchange, VWN5 / VWN3(RPA) correlation
//! (S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)), and
//! PW92 correlation (J. P. Perdew, Y. Wang, Phys. Rev. B 45, 13244 (1992)).
//! Parameters transcribed from the papers (Hartree units); correctness is
//! gated by structural tests here plus the frozen-PySCF energy oracle.
//!
//! Each functional is ONE `Scalar`-generic energy-density expression; the
//! shared driver instantiates it with f64 / nested duals for derivatives
//! and parameter gradients.

use super::traits::{Family, GridVars, Kind, XcFunctional, XcOut};
use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};

/// Densities below this are treated as vacuum (avoids ρ^{-2/3} noise).
use super::common::RHO_CUTOFF;

/// The generic body: energy density per volume e(ρα, ρβ; θ).
type EnergyFn<S> = fn(S, S, &[S]) -> S;

/// Shared driver: evaluates an `EnergyFn` at orders 0..=3 via dual nesting,
/// and parameter gradients by seeding θ.
struct LdaDriver {
    name: &'static str,
    kind: Kind,
    params: Vec<f64>,
    param_names: &'static [&'static str],
    f0: EnergyFn<f64>,
    fj: EnergyFn<crate::response::series::SeriesScalar>,
    fjt: EnergyFn<DualN<crate::response::series::SeriesScalar, 2>>,
    fjv: EnergyFn<super::taylor::ActiveDirectionalTaylorScalar<2>>,
    fjb: EnergyFn<super::bidirectional::BidirectionalTaylorScalar>,
    f1: EnergyFn<DualN<f64, 2>>,
    f2: EnergyFn<DualN<DualN<f64, 2>, 2>>,
    f3: EnergyFn<DualN<DualN<DualN<f64, 2>, 2>, 2>>,
    // Parameter seeding uses a single-slot dual over the parameter.
    fp: EnergyFn<DualN<f64, 1>>,
}

macro_rules! lda_driver {
    ($name:expr, $kind:expr, $params:expr, $names:expr, $body:ident) => {
        Box::new(LdaDriver {
            name: $name,
            kind: $kind,
            params: $params,
            param_names: $names,
            f0: $body::<f64>,
            fj: $body::<crate::response::series::SeriesScalar>,
            fjt: $body::<crate::dual::DualN<crate::response::series::SeriesScalar, 2>>,
            fjv: $body::<crate::xc::taylor::ActiveDirectionalTaylorScalar<2>>,
            fjb: $body::<crate::xc::bidirectional::BidirectionalTaylorScalar>,
            f1: $body::<DualN<f64, 2>>,
            f2: $body::<DualN<DualN<f64, 2>, 2>>,
            f3: $body::<DualN<DualN<DualN<f64, 2>, 2>, 2>>,
            fp: $body::<DualN<f64, 1>>,
        })
    };
}

impl XcFunctional for LdaDriver {
    fn name(&self) -> &'static str {
        self.name
    }
    fn kind(&self) -> Kind {
        self.kind
    }
    fn family(&self) -> Family {
        Family::Lda
    }
    fn max_analytic_order(&self) -> usize {
        3
    }
    fn taylor_energy(&self, input: &super::taylor::XcTaylorInput) -> Result<Vec<f64>> {
        input.evaluate(&self.params, |v, p| {
            if v[0].value() + v[1].value() < RHO_CUTOFF {
                crate::response::series::SeriesScalar::from_f64(0.0)
            } else {
                (self.fj)(v[0], v[1], p)
            }
        })
    }
    fn params(&self) -> Vec<f64> {
        self.params.clone()
    }
    fn taylor_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
    ) -> Result<super::taylor::XcTaylorOutput> {
        input.evaluate_potentials_for::<2>(&self.params, |v, p| (self.fjt)(v[0], v[1], p))
    }
    fn taylor_directional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        direction: &[Vec<f64>; 7],
    ) -> Result<super::taylor::XcTaylorDirectionalOutput> {
        input.evaluate_directional_potentials_for::<2>(direction, &self.params, |v, p| {
            (self.fjv)(v[0], v[1], p)
        })
    }
    fn taylor_bidirectional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        first: &[Vec<f64>; 7],
        second: &[Vec<f64>; 7],
        mixed: &[Vec<f64>; 7],
    ) -> Result<super::bidirectional::XcTaylorBidirectionalOutput> {
        input.evaluate_bidirectional_potentials(first, second, mixed, &self.params, |v, p| {
            (self.fjb)(v[0], v[1], p)
        })
    }
    fn param_names(&self) -> &'static [&'static str] {
        self.param_names
    }
    fn with_params(&self, params: &[f64]) -> Result<Box<dyn XcFunctional>> {
        if params.len() != self.params.len() {
            return Err(DftError::InvalidInput(format!(
                "{} takes {} parameter(s), got {}",
                self.name,
                self.params.len(),
                params.len()
            )));
        }
        Ok(Box::new(Self {
            name: self.name,
            kind: self.kind,
            params: params.to_vec(),
            param_names: self.param_names,
            f0: self.f0,
            fj: self.fj,
            fjt: self.fjt,
            fjv: self.fjv,
            fjb: self.fjb,
            f1: self.f1,
            f2: self.f2,
            f3: self.f3,
            fp: self.fp,
        }))
    }

    fn eval_add(&self, vars: &GridVars, order: usize, coef: f64, out: &mut XcOut) -> Result<()> {
        super::traits::eval_builtin_with_stack(self.family(), order, |token| {
            self.eval_add_impl(token, vars, order, coef, out)
        })
    }

    fn eval_add_impl(
        &self,
        _token: super::traits::XcEvalDispatchToken,
        vars: &GridVars,
        order: usize,
        coef: f64,
        out: &mut XcOut,
    ) -> Result<()> {
        if order > self.max_analytic_order() {
            return Err(DftError::DerivOrderUnsupported {
                functional: self.name.to_string(),
                requested: order,
                max_analytic: self.max_analytic_order(),
            });
        }
        let n = vars.len();
        match order {
            0 => {
                let th: Vec<f64> = self.params.clone();
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i], vars.rho[1][i]);
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    out.e[i] += coef * (self.f0)(ra.max(0.0), rb.max(0.0), &th);
                }
            }
            1 => {
                type D = DualN<f64, 2>;
                let th: Vec<D> = self.params.iter().map(|&p| D::constant(p)).collect();
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i], vars.rho[1][i]);
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let v = (self.f1)(
                        D::variable(ra.max(0.0), 0),
                        D::variable(rb.max(0.0), 1),
                        &th,
                    );
                    out.e[i] += coef * v.re;
                    out.vrho[0][i] += coef * v.im[0];
                    out.vrho[1][i] += coef * v.im[1];
                }
            }
            2 => {
                type D = DualN<f64, 2>;
                type DD = DualN<DualN<f64, 2>, 2>;
                let th: Vec<DD> = self.params.iter().map(|&p| DD::from_f64(p)).collect();
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i], vars.rho[1][i]);
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let xa = DD {
                        re: D::variable(ra.max(0.0), 0),
                        im: [D::constant(1.0), D::constant(0.0)],
                    };
                    let xb = DD {
                        re: D::variable(rb.max(0.0), 1),
                        im: [D::constant(0.0), D::constant(1.0)],
                    };
                    let v = (self.f2)(xa, xb, &th);
                    out.e[i] += coef * v.re.re;
                    out.vrho[0][i] += coef * v.re.im[0];
                    out.vrho[1][i] += coef * v.re.im[1];
                    out.v2rho2[0][i] += coef * v.im[0].im[0];
                    out.v2rho2[1][i] += coef * v.im[0].im[1];
                    out.v2rho2[2][i] += coef * v.im[1].im[1];
                }
            }
            3 => {
                type D1 = DualN<f64, 2>;
                type D2 = DualN<D1, 2>;
                type D3 = DualN<D2, 2>;
                let th: Vec<D3> = self.params.iter().map(|&p| D3::from_f64(p)).collect();
                let seed = |value: f64, k: usize| {
                    let mut d2 = D2 {
                        re: D1::variable(value, k),
                        im: [D1::constant(0.0); 2],
                    };
                    d2.im[k] = D1::constant(1.0);
                    let mut d3 = D3 {
                        re: d2,
                        im: [D2::from_f64(0.0); 2],
                    };
                    d3.im[k] = D2::from_f64(1.0);
                    d3
                };
                const RRR: [(usize, usize, usize); 4] =
                    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)];
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i], vars.rho[1][i]);
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let v = (self.f3)(seed(ra.max(0.0), 0), seed(rb.max(0.0), 1), &th);
                    out.e[i] += coef * v.re.re.re;
                    out.vrho[0][i] += coef * v.re.re.im[0];
                    out.vrho[1][i] += coef * v.re.re.im[1];
                    out.v2rho2[0][i] += coef * v.re.im[0].im[0];
                    out.v2rho2[1][i] += coef * v.re.im[0].im[1];
                    out.v2rho2[2][i] += coef * v.re.im[1].im[1];
                    for (block, &(p, q, r)) in RRR.iter().enumerate() {
                        out.v3rho3[block][i] += coef * v.im[p].im[q].im[r];
                    }
                }
            }
            _ => unreachable!("derivative order checked above"),
        }
        Ok(())
    }

    fn param_gradient(&self, vars: &GridVars, out: &mut Vec<Vec<f64>>) -> Result<()> {
        type P = DualN<f64, 1>;
        let n = vars.len();
        out.clear();
        out.resize(self.params.len(), vec![0.0; n]);
        for k in 0..self.params.len() {
            let th: Vec<P> = self
                .params
                .iter()
                .enumerate()
                .map(|(j, &p)| {
                    if j == k {
                        P::variable(p, 0)
                    } else {
                        P::constant(p)
                    }
                })
                .collect();
            for i in 0..n {
                let (ra, rb) = (vars.rho[0][i], vars.rho[1][i]);
                if ra + rb < RHO_CUTOFF {
                    continue;
                }
                let v = (self.fp)(P::constant(ra.max(0.0)), P::constant(rb.max(0.0)), &th);
                out[k][i] = v.im[0];
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Slater exchange. Parameter: overall scale (default 1.0).
// e_x = −scale · (3/4)(3/π)^{1/3} · 2^{1/3} · (ρα^{4/3} + ρβ^{4/3}).
// ---------------------------------------------------------------------------

fn slater_e<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    let cx = 0.75 * (3.0 / std::f64::consts::PI).cbrt() * 2.0_f64.cbrt();
    // Sum only over OCCUPIED channels. `ρ^{4/3}` is perfectly finite at
    // ρ = 0 and so is its first derivative, but its second is
    // (4/9)ρ^{−2/3} → ∞, and `DualN::powf` forms that infinity and then
    // multiplies it by the zero tangent seed: the block comes back as
    // `-inf` AND `NaN` together. Below RHO_CUTOFF the channel is treated as
    // ABSENT — every derivative with respect to it is zero — which is the
    // only convention that leaves the Fock build and the response kernel
    // well posed. It is exact for the energy: E(ρ, 0) ≡ E_ferro(ρ).
    let mut acc = S::from_f64(0.0);
    for r in [ra, rb] {
        if r.value() <= RHO_CUTOFF {
            continue;
        }
        acc = acc + r.powf(4.0 / 3.0);
    }
    -th[0] * S::from_f64(cx) * acc
}

pub fn slater() -> Box<dyn XcFunctional> {
    lda_driver!("slater", Kind::Exchange, vec![1.0], &["scale"], slater_e)
}

// ---------------------------------------------------------------------------
// VWN correlation (form shared by VWN5 and VWN3/RPA; parameters differ).
// θ layout: [A_P, x0_P, b_P, c_P,  A_F, x0_F, b_F, c_F,  A_α, x0_α, b_α, c_α]
// ---------------------------------------------------------------------------

/// The VWN fit F(rs; A, x0, b, c) in Hartree, x = √rs.
fn vwn_fit<S: Scalar>(x: S, a: S, x0: S, b: S, c: S) -> S {
    let xx = x * x + b * x + c;
    let x0x0 = x0 * x0 + b * x0 + c;
    let q = (S::from_f64(4.0) * c - b * b).sqrt();
    let at = (q / (S::from_f64(2.0) * x + b)).atan();
    a * ((x * x / xx).ln() + S::from_f64(2.0) * b / q * at
        - b * x0 / x0x0
            * (((x - x0) * (x - x0) / xx).ln()
                + S::from_f64(2.0) * (b + S::from_f64(2.0) * x0) / q * at))
}

use super::common::f_zeta;

/// Is exactly one spin channel occupied? Then ζ = ±1 and the ζ-interpolation
/// must not be evaluated.
///
/// `f_zeta` contains `(1∓ζ)^{4/3}`. At ζ = ±1 that is finite and so is its
/// first derivative — `(4/3)·0^{1/3} = 0` — which is why this defect is
/// invisible at order 1 and only surfaces in the second derivative, where the
/// inner dual takes `powf(1/3)` at zero and produces an infinity that the
/// zero seed turns into NaN. Everything downstream of `v2rho2` is then
/// poisoned: CPKS, the analytic Hessian, polarizabilities, TDDFT.
///
/// At ζ = ±1 the interpolation collapses ANALYTICALLY, not approximately:
/// `f_zeta(±1) = 1` and `ζ⁴ = 1`, so the `(1 − ζ⁴)` spin-stiffness term
/// vanishes and the paramagnetic fit cancels, leaving exactly the
/// ferromagnetic fit. Taking that branch is therefore not a regularisation —
/// it is the same number, computed without the singular intermediate.
#[inline]
pub(crate) fn one_channel_only<S: Scalar>(ra: S, rb: S) -> bool {
    (ra.value() <= RHO_CUTOFF) != (rb.value() <= RHO_CUTOFF)
}

/// VWN5 correlation energy density (per volume) for a parameter slice.
pub(crate) fn vwn5_energy<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    vwn_e(ra, rb, th)
}

fn vwn_e<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    let rho = ra + rb;
    let rs = super::common::wigner_seitz(rho);
    let x = rs.sqrt();
    if one_channel_only(ra, rb) {
        // ζ = ±1: ε_c = ε_F exactly.
        return rho * vwn_fit(x, th[4], th[5], th[6], th[7]);
    }
    let z = (ra - rb) / rho;
    let ec_p = vwn_fit(x, th[0], th[1], th[2], th[3]);
    let ec_f = vwn_fit(x, th[4], th[5], th[6], th[7]);
    let alpha = vwn_fit(x, th[8], th[9], th[10], th[11]);
    let fz = f_zeta(z);
    let fpp0 = 4.0 / (9.0 * (2.0_f64.cbrt() - 1.0));
    let z4 = z * z * z * z;
    let one = S::from_f64(1.0);
    let ec = ec_p + alpha * fz * S::from_f64(1.0 / fpp0) * (one - z4) + (ec_f - ec_p) * fz * z4;
    rho * ec
}

/// VWN with the SIMPLE von Barth–Hedin-style spin interpolation
/// ε = ε_P + (ε_F − ε_P) f(ζ) — the convention libxc calls `VWN_RPA` and
/// the one entering Gaussian-style B3LYP ("VWN3").
fn vwn_simple_e<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    let rho = ra + rb;
    if one_channel_only(ra, rb) {
        return vwn_simple_ferro(rho, th);
    }
    let rs = super::common::wigner_seitz(rho);
    let x = rs.sqrt();
    let z = (ra - rb) / rho;
    let ec_p = vwn_fit(x, th[0], th[1], th[2], th[3]);
    let ec_f = vwn_fit(x, th[4], th[5], th[6], th[7]);
    let fz = f_zeta(z);
    rho * (ec_p + (ec_f - ec_p) * fz)
}

/// `vwn_simple_e` with the ζ = ±1 branch taken (see [`one_channel_only`]):
/// `f(±1) = 1`, so ε_c = ε_P + (ε_F − ε_P) = ε_F.
fn vwn_simple_ferro<S: Scalar>(rho: S, th: &[S]) -> S {
    let rs = super::common::wigner_seitz(rho);
    let x = rs.sqrt();
    rho * vwn_fit(x, th[4], th[5], th[6], th[7])
}

pub(crate) const VWN5_PARAMS: [f64; 12] = [
    0.0310907,
    -0.10498,
    3.72744,
    12.9352, // paramagnetic
    0.01554535,
    -0.32500,
    7.06042,
    18.0578, // ferromagnetic
    // Exact spin-stiffness coefficient.
    -1.0 / (6.0 * std::f64::consts::PI * std::f64::consts::PI),
    -0.0047584,
    1.13107,
    13.0045, // spin stiffness
];

const VWN3_PARAMS: [f64; 8] = [
    0.0310907, -0.409286, 13.0720, 42.7198, // paramagnetic (RPA fit)
    0.01554535, -0.743294, 20.1231, 101.578, // ferromagnetic (RPA fit)
];

const VWN_PARAM_NAMES: &[&str] = &[
    "A_P", "x0_P", "b_P", "c_P", "A_F", "x0_F", "b_F", "c_F", "A_a", "x0_a", "b_a", "c_a",
];

const VWN_SIMPLE_PARAM_NAMES: &[&str] = &["A_P", "x0_P", "b_P", "c_P", "A_F", "x0_F", "b_F", "c_F"];

pub fn vwn5() -> Box<dyn XcFunctional> {
    lda_driver!(
        "vwn5",
        Kind::Correlation,
        VWN5_PARAMS.to_vec(),
        VWN_PARAM_NAMES,
        vwn_e
    )
}

pub fn vwn3() -> Box<dyn XcFunctional> {
    lda_driver!(
        "vwn3",
        Kind::Correlation,
        VWN3_PARAMS.to_vec(),
        VWN_SIMPLE_PARAM_NAMES,
        vwn_simple_e
    )
}

// ---------------------------------------------------------------------------
// PW92 correlation.
// θ layout: three G-fits × [A, α1, β1, β2, β3, β4]:
// ec(rs,0), ec(rs,1), −αc(rs).
// ---------------------------------------------------------------------------

pub(crate) fn pw92_g<S: Scalar>(rs: S, a: S, a1: S, b1: S, b2: S, b3: S, b4: S) -> S {
    let one = S::from_f64(1.0);
    let two_a = S::from_f64(2.0) * a;
    let srs = rs.sqrt();
    let den = two_a * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs);
    -(two_a) * (one + a1 * rs) * den.recip().ln_1p()
}

/// PW92 correlation energy **per particle** ε_c(rs, ζ) — shared with the
/// PBE correlation functional.
pub(crate) fn pw92_epsilon<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    let fpp0 = 8.0 / (9.0 * (2.0_f64.powf(4.0 / 3.0) - 2.0));
    pw92_epsilon_with_spin_stiffness(ra, rb, th, fpp0)
}

/// Keep the original PW92 rounded spin-stiffness normalization separate
/// from the precise normalization used by the PW92-MOD local base.
fn pw92_epsilon_with_spin_stiffness<S: Scalar>(ra: S, rb: S, th: &[S], fpp0: f64) -> S {
    let rho = ra + rb;
    if one_channel_only(ra, rb) {
        // ζ = ±1: the (1 − ζ⁴) spin-stiffness term vanishes and ec0 cancels,
        // leaving exactly ec(rs, 1). Identical value, no singular
        // intermediate — see `pw92_epsilon_ferro`, which this reuses.
        return pw92_epsilon_ferro(rho, th);
    }
    let rs = super::common::wigner_seitz(rho);
    let z = (ra - rb) / rho;
    let ec0 = pw92_g(rs, th[0], th[1], th[2], th[3], th[4], th[5]);
    let ec1 = pw92_g(rs, th[6], th[7], th[8], th[9], th[10], th[11]);
    let neg_ac = pw92_g(rs, th[12], th[13], th[14], th[15], th[16], th[17]);
    let fz = f_zeta(z);
    let z4 = z * z * z * z;
    let one = S::from_f64(1.0);
    ec0 - neg_ac * fz * S::from_f64(1.0 / fpp0) * (one - z4) + (ec1 - ec0) * fz * z4
}

/// PW92 ε_c in the fully polarized (ζ = 1) limit: exactly ec(rs, 1).
/// The general ζ-interpolation evaluates (1−ζ)^{4/3} at ζ = 1, whose
/// SECOND dual derivative is 0^{-2/3}·0 = NaN — the same-spin correlation
/// channels of the B97/M06 families must use this closed branch instead.
pub(crate) fn pw92_epsilon_ferro<S: Scalar>(rho: S, th: &[S]) -> S {
    let rs = super::common::wigner_seitz(rho);
    pw92_g(rs, th[6], th[7], th[8], th[9], th[10], th[11])
}

fn pw92_e<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    // Original PW92 uses its published rounded normalization. The shared
    // epsilon helper above retains the precise PW92-MOD convention.
    (ra + rb) * pw92_epsilon_original(ra, rb, th)
}

/// Published PW92 spin interpolation, including its rounded normalization.
pub(crate) fn pw92_epsilon_original<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    pw92_epsilon_with_spin_stiffness(ra, rb, th, 1.709921)
}

pub(crate) const PW92_PARAMS: [f64; 18] = [
    0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, // ec(rs, 0)
    0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, // ec(rs, 1)
    0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, // −αc(rs)
];

/// PW92 with the more precise A digits of the reference implementations
/// ("MOD"), the local base of PBE, TPSS, M06 and the SCAN family.
pub(crate) const PW92_MOD_PARAMS: [f64; 18] = [
    0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, // ec(rs, 0)
    0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, // ec(rs, 1)
    0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, // -alpha_c(rs)
];

const PW92_PARAM_NAMES: &[&str] = &[
    "A_0", "a1_0", "b1_0", "b2_0", "b3_0", "b4_0", "A_1", "a1_1", "b1_1", "b2_1", "b3_1", "b4_1",
    "A_a", "a1_a", "b1_a", "b2_a", "b3_a", "b4_a",
];

pub fn pw92() -> Box<dyn XcFunctional> {
    lda_driver!(
        "pw92",
        Kind::Correlation,
        PW92_PARAMS.to_vec(),
        PW92_PARAM_NAMES,
        pw92_e
    )
}

/// X-alpha exchange (Slater 1951): Slater exchange scaled by `3 alpha / 2`;
/// the historical alpha = 0.7.
pub fn xalpha() -> Box<dyn XcFunctional> {
    slater().with_params(&[1.05]).expect("x-alpha scale")
}

/// PZ81 correlation (Perdew-Zunger, PRB 23, 5048 (1981)): the
/// Ceperley-Alder fits for rs < 1 and rs >= 1 in the paramagnetic and
/// ferromagnetic limits and the von Barth-Hedin spin interpolation.
fn pz81_e<S: Scalar>(ra: S, rb: S, _th: &[S]) -> S {
    let rho = ra + rb;
    let rs = super::common::wigner_seitz(rho);
    if one_channel_only(ra, rb) {
        return rho * super::legacy_c::pz81_branch(rs, true);
    }
    let z = (ra - rb) / rho;
    let ep = super::legacy_c::pz81_branch(rs, false);
    let ef = super::legacy_c::pz81_branch(rs, true);
    rho * (ep + f_zeta(z) * (ef - ep))
}

pub fn pz81() -> Box<dyn XcFunctional> {
    lda_driver!("pz81", Kind::Correlation, vec![], &[], pz81_e)
}

/// Wigner correlation (Wigner 1938, in the spin-scaled form
/// `(1 - zeta^2) a / (b + rs)`). theta = [a, b].
fn wigner_e<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    let rho = ra + rb;
    if one_channel_only(ra, rb) {
        return S::from_f64(0.0);
    }
    let rs = super::common::wigner_seitz(rho);
    let z = (ra - rb) / rho;
    rho * (S::from_f64(1.0) - z * z) * th[0] / (th[1] + rs)
}

pub fn wigner() -> Box<dyn XcFunctional> {
    lda_driver!(
        "wigner",
        Kind::Correlation,
        vec![-0.44, 7.8],
        &["a", "b"],
        wigner_e
    )
}

/// Hedin-Lundqvist form `-c [(1 + x^3) ln(1 + 1/x) + x/2 - x^2 - 1/3]`,
/// x = rs / r, for one spin limit.
fn hl_branch<S: Scalar>(rs: S, r: S, c: S) -> S {
    let one = S::from_f64(1.0);
    let x = rs / r;
    -c * ((one + x * x * x) * (one + x.recip()).ln() + x * S::from_f64(0.5)
        - x * x
        - S::from_f64(1.0 / 3.0))
}

/// Hedin-Lundqvist-type correlation with the von Barth-Hedin spin
/// interpolation. theta = [r_P, r_F, c_P, c_F].
fn hl_e<S: Scalar>(ra: S, rb: S, th: &[S]) -> S {
    let rho = ra + rb;
    let rs = super::common::wigner_seitz(rho);
    if one_channel_only(ra, rb) {
        return rho * hl_branch(rs, th[1], th[3]);
    }
    let z = (ra - rb) / rho;
    let ep = hl_branch(rs, th[0], th[2]);
    let ef = hl_branch(rs, th[1], th[3]);
    rho * (ep + f_zeta(z) * (ef - ep))
}

/// Hedin-Lundqvist correlation (J. Phys. C 4, 2064 (1971)).
pub fn hl() -> Box<dyn XcFunctional> {
    lda_driver!(
        "hl",
        Kind::Correlation,
        vec![21.0, 21.0, 0.0225, 0.0225],
        &["r_P", "r_F", "c_P", "c_F"],
        hl_e
    )
}

/// Gunnarsson-Lundqvist correlation (PRB 13, 4274 (1976)).
pub fn gl() -> Box<dyn XcFunctional> {
    lda_driver!(
        "gl",
        Kind::Correlation,
        vec![11.4, 15.9, 0.0333, 0.0203],
        &["r_P", "r_F", "c_P", "c_F"],
        hl_e
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval1(f: &dyn XcFunctional, ra: f64, rb: f64) -> (f64, f64, f64) {
        let rho_a = [ra];
        let rho_b = [rb];
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: None,
            tau: None,
        };
        let mut out = XcOut::default();
        out.prepare(1, 1, Family::Lda);
        f.eval_add(&vars, 1, 1.0, &mut out).unwrap();
        (out.e[0], out.vrho[0][0], out.vrho[1][0])
    }

    fn energy0(f: &dyn XcFunctional, ra: f64, rb: f64) -> f64 {
        let rho_a = [ra];
        let rho_b = [rb];
        let vars = GridVars {
            rho: [&rho_a, &rho_b],
            sigma: None,
            tau: None,
        };
        let mut out = XcOut::default();
        out.prepare(1, 0, Family::Lda);
        f.eval_add(&vars, 0, 1.0, &mut out).unwrap();
        out.e[0]
    }

    /// Slater exchange closed form at ζ = 0:
    /// e_x = −(3/4)(3/π)^{1/3} ρ^{4/3}.
    #[test]
    fn slater_ueg_closed_form() {
        let f = slater();
        let rho: f64 = 0.7;
        let want = -0.75 * (3.0 / std::f64::consts::PI).cbrt() * rho.powf(4.0 / 3.0);
        let got = energy0(f.as_ref(), rho / 2.0, rho / 2.0);
        assert!((got - want).abs() < 1e-14, "{got} vs {want}");
    }

    /// Exchange spin scaling: Ex[ρa,ρb] = ½(Ex[2ρa] + Ex[2ρb]) where Ex[ρ]
    /// is the unpolarized functional of total density ρ, i.e. e(ρ/2, ρ/2).
    #[test]
    fn slater_spin_scaling() {
        let f = slater();
        let (ra, rb) = (0.31, 0.87);
        let lhs = energy0(f.as_ref(), ra, rb);
        let rhs = 0.5 * (energy0(f.as_ref(), ra, ra) + energy0(f.as_ref(), rb, rb));
        assert!((lhs - rhs).abs() < 1e-13, "{lhs} vs {rhs}");
    }

    /// All first derivatives against central finite differences.
    #[test]
    fn first_derivatives_match_fd() {
        for f in [slater(), vwn5(), vwn3(), pw92()] {
            for &(ra, rb) in &[(0.3, 0.3), (0.9, 0.2), (0.05, 0.4), (1.7, 1.1)] {
                let (_, va, vb) = eval1(f.as_ref(), ra, rb);
                let h = 1e-6;
                let fda =
                    (energy0(f.as_ref(), ra + h, rb) - energy0(f.as_ref(), ra - h, rb)) / (2.0 * h);
                let fdb =
                    (energy0(f.as_ref(), ra, rb + h) - energy0(f.as_ref(), ra, rb - h)) / (2.0 * h);
                assert!(
                    (va - fda).abs() < 1e-7 * va.abs().max(1.0),
                    "{} vrho_a {va} vs {fda} at ({ra},{rb})",
                    f.name()
                );
                assert!(
                    (vb - fdb).abs() < 1e-7 * vb.abs().max(1.0),
                    "{} vrho_b {vb} vs {fdb}",
                    f.name()
                );
            }
        }
    }

    /// Second derivatives against finite differences of the analytic first
    /// derivatives.
    #[test]
    fn second_derivatives_match_fd() {
        for f in [slater(), vwn5(), pw92()] {
            let (ra, rb) = (0.6, 0.35);
            let rho_a = [ra];
            let rho_b = [rb];
            let vars = GridVars {
                rho: [&rho_a, &rho_b],
                sigma: None,
                tau: None,
            };
            let mut out = XcOut::default();
            out.prepare(1, 2, Family::Lda);
            f.eval_add(&vars, 2, 1.0, &mut out).unwrap();
            let h = 1e-6;
            let g = |ra: f64, rb: f64| {
                let (_, va, vb) = eval1(f.as_ref(), ra, rb);
                (va, vb)
            };
            let (vap, _) = g(ra + h, rb);
            let (vam, _) = g(ra - h, rb);
            let (vbp_a, vbp) = g(ra, rb + h);
            let (vbm_a, vbm) = g(ra, rb - h);
            let faa = (vap - vam) / (2.0 * h);
            let fab = (vbp_a - vbm_a) / (2.0 * h);
            let fbb = (vbp - vbm) / (2.0 * h);
            assert!(
                (out.v2rho2[0][0] - faa).abs() < 1e-6 * faa.abs().max(1.0),
                "{}",
                f.name()
            );
            assert!(
                (out.v2rho2[1][0] - fab).abs() < 1e-6 * fab.abs().max(1.0),
                "{}",
                f.name()
            );
            assert!(
                (out.v2rho2[2][0] - fbb).abs() < 1e-6 * fbb.abs().max(1.0),
                "{}",
                f.name()
            );
        }
    }

    /// Parameter gradients against finite differences through with_params.
    #[test]
    fn param_gradient_matches_fd() {
        for f in [slater(), vwn5(), pw92()] {
            let (ra, rb) = (0.45, 0.3);
            let rho_a = [ra];
            let rho_b = [rb];
            let vars = GridVars {
                rho: [&rho_a, &rho_b],
                sigma: None,
                tau: None,
            };
            let mut grads = Vec::new();
            f.param_gradient(&vars, &mut grads).unwrap();
            let base = f.params();
            for k in 0..base.len() {
                let h = 1e-6 * base[k].abs().max(1e-3);
                let mut up = base.clone();
                up[k] += h;
                let mut dn = base.clone();
                dn[k] -= h;
                let fu = f.with_params(&up).unwrap();
                let fd_ = f.with_params(&dn).unwrap();
                let fd = (energy0(fu.as_ref(), ra, rb) - energy0(fd_.as_ref(), ra, rb)) / (2.0 * h);
                assert!(
                    (grads[k][0] - fd).abs() < 1e-6 * fd.abs().max(1.0),
                    "{} ∂e/∂{} = {} vs FD {fd}",
                    f.name(),
                    f.param_names()[k],
                    grads[k][0]
                );
            }
        }
    }

    /// Correlation at ζ = ±1 consistency: vwn/pw92 must be symmetric in
    /// (ρa, ρb) and reduce to the ferromagnetic branch.
    #[test]
    fn correlation_symmetry() {
        for f in [vwn5(), vwn3(), pw92()] {
            let e_ab = energy0(f.as_ref(), 0.8, 0.2);
            let e_ba = energy0(f.as_ref(), 0.2, 0.8);
            assert!((e_ab - e_ba).abs() < 1e-14, "{}", f.name());
        }
    }
}
