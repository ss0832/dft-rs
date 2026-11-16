// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! GGA functionals: B88 exchange (Becke, PRA 38, 3098 (1988)), PBE exchange
//! and correlation (Perdew–Burke–Ernzerhof, PRL 77, 3865 (1996)), OPTX
//! exchange (Handy–Cohen, Mol. Phys. 99, 403 (2001)), and LYP correlation
//! (Lee–Yang–Parr 1988 in the Miehlich et al. form, CPL 157, 200 (1989)).
//! Parameters transcribed from the papers; verified against the frozen
//! PySCF oracle and per-order finite-difference gates.
//!
//! Variable order everywhere: (ρα, ρβ, σαα, σαβ, σββ).

use super::traits::{Family, GridVars, Kind, XcFunctional, XcOut};
use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};

use super::common::RHO_CUTOFF;

pub(crate) type GgaFn<S> = fn(S, S, S, S, S, &[S]) -> S;

/// Shared GGA driver: one generic energy density, dual-instantiated per
/// order; parameter gradients by θ seeding. Also reused by the B97 family.
pub(crate) struct GgaDriver {
    /// Declared scaling powers of the parameters (empty: the default).
    pub(crate) param_scaling: &'static [i32],
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    pub(crate) params: Vec<f64>,
    pub(crate) param_names: &'static [&'static str],
    pub(crate) f0: GgaFn<f64>,
    pub(crate) fj: GgaFn<crate::response::series::SeriesScalar>,
    pub(crate) fjt: GgaFn<DualN<crate::response::series::SeriesScalar, 5>>,
    pub(crate) fjv: GgaFn<super::taylor::ActiveDirectionalTaylorScalar<5>>,
    pub(crate) fjb: GgaFn<super::bidirectional::BidirectionalTaylorScalar>,
    pub(crate) f1: GgaFn<DualN<f64, 5>>,
    pub(crate) f2: GgaFn<DualN<DualN<f64, 5>, 5>>,
    pub(crate) f3: GgaFn<DualN<DualN<DualN<f64, 5>, 5>, 5>>,
    pub(crate) fp: GgaFn<DualN<f64, 1>>,
}

impl XcFunctional for GgaDriver {
    fn name(&self) -> &'static str {
        self.name
    }
    fn kind(&self) -> Kind {
        self.kind
    }
    fn family(&self) -> Family {
        Family::Gga
    }
    fn max_analytic_order(&self) -> usize {
        3
    }
    fn taylor_energy(&self, input: &super::taylor::XcTaylorInput) -> Result<Vec<f64>> {
        input.evaluate(&self.params, |v, p| {
            if v[0].value() + v[1].value() < RHO_CUTOFF {
                crate::response::series::SeriesScalar::from_f64(0.0)
            } else {
                (self.fj)(v[0], v[1], v[2], v[3], v[4], p)
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
        input.evaluate_potentials_for::<5>(&self.params, |v, p| {
            (self.fjt)(v[0], v[1], v[2], v[3], v[4], p)
        })
    }
    fn taylor_directional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        direction: &[Vec<f64>; 7],
    ) -> Result<super::taylor::XcTaylorDirectionalOutput> {
        input.evaluate_directional_potentials_for::<5>(direction, &self.params, |v, p| {
            (self.fjv)(v[0], v[1], v[2], v[3], v[4], p)
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
            (self.fjb)(v[0], v[1], v[2], v[3], v[4], p)
        })
    }
    fn param_names(&self) -> &'static [&'static str] {
        self.param_names
    }
    fn param_scaling_powers(&self) -> Vec<i32> {
        if self.param_scaling.is_empty() {
            super::traits::default_param_scaling(self.param_names)
        } else {
            self.param_scaling.to_vec()
        }
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
            param_scaling: self.param_scaling,
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
        let sigma = vars.sigma.ok_or_else(|| {
            DftError::InvalidInput(format!("{} needs σ = ∇ρ·∇ρ grid variables", self.name))
        })?;
        let n = vars.len();
        match order {
            0 => {
                let th = self.params.clone();
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    out.e[i] += coef
                        * (self.f0)(
                            ra,
                            rb,
                            sigma[0][i].max(0.0),
                            sigma[1][i],
                            sigma[2][i].max(0.0),
                            &th,
                        );
                }
            }
            1 => {
                type D = DualN<f64, 5>;
                let th: Vec<D> = self.params.iter().map(|&p| D::constant(p)).collect();
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let v = (self.f1)(
                        D::variable(ra, 0),
                        D::variable(rb, 1),
                        D::variable(sigma[0][i].max(0.0), 2),
                        D::variable(sigma[1][i], 3),
                        D::variable(sigma[2][i].max(0.0), 4),
                        &th,
                    );
                    out.e[i] += coef * v.re;
                    for k in 0..2 {
                        out.vrho[k][i] += coef * v.im[k];
                    }
                    for k in 0..3 {
                        out.vsigma[k][i] += coef * v.im[2 + k];
                    }
                }
            }
            2 => {
                type D = DualN<f64, 5>;
                type DD = DualN<DualN<f64, 5>, 5>;
                let th: Vec<DD> = self.params.iter().map(|&p| DD::from_f64(p)).collect();
                let seed = |v: f64, k: usize| {
                    let mut x = DD {
                        re: D::variable(v, k),
                        im: [D::constant(0.0); 5],
                    };
                    x.im[k] = D::constant(1.0);
                    x
                };
                // Second-derivative block orders (see traits.rs).
                const RR: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];
                const RS: [(usize, usize); 6] = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
                const SS: [(usize, usize); 6] = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)];
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let v = (self.f2)(
                        seed(ra, 0),
                        seed(rb, 1),
                        seed(sigma[0][i].max(0.0), 2),
                        seed(sigma[1][i], 3),
                        seed(sigma[2][i].max(0.0), 4),
                        &th,
                    );
                    out.e[i] += coef * v.re.re;
                    for k in 0..2 {
                        out.vrho[k][i] += coef * v.re.im[k];
                    }
                    for k in 0..3 {
                        out.vsigma[k][i] += coef * v.re.im[2 + k];
                    }
                    for (b, &(p, q)) in RR.iter().enumerate() {
                        out.v2rho2[b][i] += coef * v.im[p].im[q];
                    }
                    for (b, &(p, q)) in RS.iter().enumerate() {
                        out.v2rhosigma[b][i] += coef * v.im[p].im[q];
                    }
                    for (b, &(p, q)) in SS.iter().enumerate() {
                        out.v2sigma2[b][i] += coef * v.im[p].im[q];
                    }
                }
            }
            3 => {
                type D1 = DualN<f64, 5>;
                type D2 = DualN<D1, 5>;
                type D3 = DualN<D2, 5>;
                let th: Vec<D3> = self.params.iter().map(|&p| D3::from_f64(p)).collect();
                let seed = |value: f64, k: usize| {
                    let mut d2 = D2 {
                        re: D1::variable(value, k),
                        im: [D1::constant(0.0); 5],
                    };
                    d2.im[k] = D1::constant(1.0);
                    let mut d3 = D3 {
                        re: d2,
                        im: [D2::from_f64(0.0); 5],
                    };
                    d3.im[k] = D2::from_f64(1.0);
                    d3
                };
                const RR: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];
                const RS: [(usize, usize); 6] = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
                const SS: [(usize, usize); 6] = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)];
                const RRR: [(usize, usize, usize); 4] =
                    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)];
                const SSS: [(usize, usize, usize); 10] = [
                    (2, 2, 2),
                    (2, 2, 3),
                    (2, 2, 4),
                    (2, 3, 3),
                    (2, 3, 4),
                    (2, 4, 4),
                    (3, 3, 3),
                    (3, 3, 4),
                    (3, 4, 4),
                    (4, 4, 4),
                ];
                for i in 0..n {
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let v = (self.f3)(
                        seed(ra, 0),
                        seed(rb, 1),
                        seed(sigma[0][i].max(0.0), 2),
                        seed(sigma[1][i], 3),
                        seed(sigma[2][i].max(0.0), 4),
                        &th,
                    );
                    out.e[i] += coef * v.re.re.re;
                    for k in 0..2 {
                        out.vrho[k][i] += coef * v.re.re.im[k];
                    }
                    for k in 0..3 {
                        out.vsigma[k][i] += coef * v.re.re.im[2 + k];
                    }
                    for (block, &(p, q)) in RR.iter().enumerate() {
                        out.v2rho2[block][i] += coef * v.re.im[p].im[q];
                    }
                    for (block, &(p, q)) in RS.iter().enumerate() {
                        out.v2rhosigma[block][i] += coef * v.re.im[p].im[q];
                    }
                    for (block, &(p, q)) in SS.iter().enumerate() {
                        out.v2sigma2[block][i] += coef * v.re.im[p].im[q];
                    }
                    let d3 = |p: usize, q: usize, r: usize| v.im[p].im[q].im[r];
                    for (block, &(p, q, r)) in RRR.iter().enumerate() {
                        out.v3rho3[block][i] += coef * d3(p, q, r);
                    }
                    for (rp, &(p, q)) in RR.iter().enumerate() {
                        for s in 0..3 {
                            out.v3rho2sigma[3 * rp + s][i] += coef * d3(p, q, 2 + s);
                        }
                    }
                    for rho in 0..2 {
                        for (sp, &(p, q)) in SS.iter().enumerate() {
                            out.v3rhosigma2[6 * rho + sp][i] += coef * d3(rho, p, q);
                        }
                    }
                    for (block, &(p, q, r)) in SSS.iter().enumerate() {
                        out.v3sigma3[block][i] += coef * d3(p, q, r);
                    }
                }
            }
            _ => unreachable!("derivative order checked above"),
        }
        Ok(())
    }

    fn param_gradient(&self, vars: &GridVars, out: &mut Vec<Vec<f64>>) -> Result<()> {
        type P = DualN<f64, 1>;
        let sigma = vars.sigma.ok_or_else(|| {
            DftError::InvalidInput(format!("{} needs σ grid variables", self.name))
        })?;
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
                let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                if ra + rb < RHO_CUTOFF {
                    continue;
                }
                let v = (self.fp)(
                    P::constant(ra),
                    P::constant(rb),
                    P::constant(sigma[0][i].max(0.0)),
                    P::constant(sigma[1][i]),
                    P::constant(sigma[2][i].max(0.0)),
                    &th,
                );
                out[k][i] = v.im[0];
            }
        }
        Ok(())
    }
}

/// −(3/2)(3/(4π))^{1/3}: the per-spin LDA exchange coefficient of
/// e = Σσ C ρσ^{4/3}.
const CX_SPIN: f64 = -super::common::X_FACTOR_C;

// ---------------------------------------------------------------------------
// B88 exchange (LDA part included). θ = [beta].
// ---------------------------------------------------------------------------

/// Threshold below which the density GRADIENT is treated as absent.
///
/// Paired with `RHO_CUTOFF = 1e-13` by dimension: σ = |∇ρ|², so a density at
/// the cutoff carries a gradient of order 1e-13 and a σ of order 1e-26.
pub(crate) const SIGMA_CUTOFF: f64 = 1e-26;

/// Is the density gradient absent at this point?
///
/// The companion of the empty-spin-channel rule in `lda::one_channel_only`,
/// and it exists for the cases that rule CANNOT cover. B88 turned out to be
/// analytic in σ once written in `u = x²` (see [`x_asinh_x`]), but three
/// functionals are genuinely not:
///
/// - HJS ωPBE's rational fit carries ODD powers of `s ∝ √σ`;
/// - P86's gradient term is `σ·exp(−c√σ)`, i.e. `t² − c t³ + …` in `t = √σ`;
/// - r²SCAN's correlation takes the root of a quantity linear in σ.
///
/// Their σ-derivatives really do diverge as σ → 0 — that is a property of the
/// published functionals, not of this implementation, and no rewriting
/// removes it. Below this cutoff the gradient is therefore treated as ABSENT:
/// the functional takes its σ = 0 value in closed form and every
/// σ-derivative is zero. `tests/p22_xc_boundaries.rs` records these as
/// `finite_by_convention` rather than `finite`, precisely so the distinction
/// from an exact one-sided limit stays visible.
#[inline]
pub(crate) fn sigma_is_absent<S: Scalar>(sig: S) -> bool {
    sig.value() <= SIGMA_CUTOFF
}

/// `x·asinh(x)` as an ANALYTIC function of `u = x²`.
///
/// B88 depends on the reduced gradient only through `x² = σ/ρ^{8/3}` and this
/// one combination, and both are analytic in σ — including at σ = 0. Writing
/// them via `x = √σ/ρ^{4/3}` is what breaks: `√·` has an infinite derivative
/// at zero, and `DualN` multiplies that infinity by the zero seed, so `0·∞`
/// poisons EVERY tangent slot of the dual. The σ-derivative singularity thus
/// destroys `vrho` as well — a functional whose energy is perfectly finite
/// returns a NaN potential at the centre of any closed-shell atom.
///
/// Evaluated as `√u·asinh(√u) = Σ_{n≥0} (−1)ⁿ (2n)!/(4ⁿ(n!)²(2n+1)) u^{n+1}`
/// below the switch and in closed form above it. The two agree to 1e-14
/// relative at u = 1e-2 (measured), so the switch is invisible at working
/// precision while removing the singular intermediate entirely.
pub(crate) fn x_asinh_x<S: Scalar>(u: S) -> S {
    const C: [f64; 7] = [
        1.0,
        -1.0 / 6.0,
        3.0 / 40.0,
        -5.0 / 112.0,
        35.0 / 1152.0,
        -63.0 / 2816.0,
        231.0 / 13312.0,
    ];
    if u.value() < 1.0e-2 {
        let mut acc = S::from_f64(0.0);
        let mut p = u;
        for &c in C.iter() {
            acc = acc + S::from_f64(c) * p;
            p = p * u;
        }
        return acc;
    }
    let x = u.sqrt();
    x * x.asinh()
}

fn b88_spin<S: Scalar>(rho: S, sig: S, beta: S) -> S {
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let r43 = rho.powf(4.0 / 3.0);
    // u = x² = σ/ρ^{8/3}. Never form `x` itself: see `x_asinh_x`.
    let u = sig / (r43 * r43);
    let denom = S::from_f64(1.0) + S::from_f64(6.0) * beta * x_asinh_x(u);
    S::from_f64(CX_SPIN) * r43 - beta * r43 * u / denom
}

fn b88_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    b88_spin(ra, saa, th[0]) + b88_spin(rb, sbb, th[0])
}

pub fn b88() -> Box<dyn XcFunctional> {
    gga_driver!("b88", Kind::Exchange, vec![0.0042], &["beta"], b88_e)
}

// ---------------------------------------------------------------------------
// PBE exchange. θ = [kappa, mu].
// ---------------------------------------------------------------------------

/// Spin-scaled GGA exchange of one channel, `1/2 e_x^unif(2 rho) F(s^2)`,
/// with the reduced gradient `s^2 = |grad 2 rho|^2 / (4 kF^2 (2 rho)^2)`,
/// `kF = (3 pi^2 2 rho)^(1/3)`, of the spin-scaled density. Every
/// enhancement-factor exchange (PBE, RPBE, PW91, ...) is this with its own
/// `F`.
pub(crate) fn spin_scaled_exchange<S: Scalar>(rho: S, sig: S, enhancement: impl Fn(S) -> S) -> S {
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let r = S::from_f64(2.0) * rho;
    let s4 = S::from_f64(4.0) * sig;
    let kf2 = (S::from_f64(3.0 * std::f64::consts::PI * std::f64::consts::PI) * r).powf(2.0 / 3.0);
    let s2 = s4 / (S::from_f64(4.0) * kf2 * r * r);
    let e_lda_unpol = S::from_f64(-0.75 * 0.984_745_021_842_696_5) * r.powf(4.0 / 3.0);
    S::from_f64(0.5) * e_lda_unpol * enhancement(s2)
}

fn pbe_x_spin<S: Scalar>(rho: S, sig: S, kappa: S, mu: S) -> S {
    spin_scaled_exchange(rho, sig, |s2| pbe_enhancement(s2, kappa, mu))
}

pub(crate) fn pbe_enhancement<S: Scalar>(s2: S, kappa: S, mu: S) -> S {
    S::from_f64(1.0) + kappa - kappa / (S::from_f64(1.0) + mu * s2 / kappa)
}

fn pbe_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    pbe_x_spin(ra, saa, th[0], th[1]) + pbe_x_spin(rb, sbb, th[0], th[1])
}

pub fn pbe_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "pbe",
        Kind::Exchange,
        vec![0.804, 0.2195149727645171],
        &["kappa", "mu"],
        pbe_x_e
    )
}

// ---------------------------------------------------------------------------
// OPTX exchange (Handy–Cohen). θ = [a1, a2, gamma].
// Full form: e = Σσ ρσ^{4/3} [ a1·CX_SPIN − a2 (γ xσ²/(1+γ xσ²))² ].
// The pure gradient correction (for O3LYP) is obtained with a1 = 0.
// ---------------------------------------------------------------------------

fn optx_spin<S: Scalar>(rho: S, sig: S, a1: S, a2: S, gamma: S) -> S {
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let r43 = rho.powf(4.0 / 3.0);
    let x2 = sig / (r43 * r43);
    let u = gamma * x2 / (S::from_f64(1.0) + gamma * x2);
    a1 * S::from_f64(CX_SPIN) * r43 - a2 * u * u * r43
}

fn optx_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    optx_spin(ra, saa, th[0], th[1], th[2]) + optx_spin(rb, sbb, th[0], th[1], th[2])
}

/// revPBE exchange (Zhang-Yang, PRL 80, 890 (1998)): PBE-X with
/// kappa = 1.245 (mu unchanged).
pub fn revpbe_x() -> Box<dyn XcFunctional> {
    pbe_x()
        .with_params(&[1.245, 0.2195149727645171])
        .expect("revpbe params")
}

pub fn optx() -> Box<dyn XcFunctional> {
    gga_driver!(
        "optx",
        Kind::Exchange,
        vec![1.05151, 1.43169, 0.006],
        &["a1", "a2", "gamma"],
        optx_e
    )
}

// ---------------------------------------------------------------------------
// ITYH short-range attenuation (Iikura–Tsuneda–Yanai–Hirao, JCP 115, 3540
// (2001)): the short-range (erfc) part of any GGA exchange via its
// effective exchange hole. Writing e_xσ^GGA = −½ρσ^{4/3}Kσ:
//   aσ = ω√Kσ / (6√π ρσ^{1/3}),
//   F(a) = 1 − (8/3)a[√π erf(1/(2a)) − 3a + 4a³ + (2a − 4a³)e^{−1/(4a²)}],
//   e_xσ^SR = e_xσ^GGA · F(aσ).
// Used for CAM-B3LYP and LC functionals; the ω parameter is the same one
// entering the long-range HF exchange kernel.
// ---------------------------------------------------------------------------

/// The universal short-range attenuation factor F(a) of a Gaussian-model
/// exchange hole (Gill/Adamson; used by ITYH and the ωB97 family).
pub(crate) fn sr_factor<S: Scalar>(a: S) -> S {
    if a.value() < 1e-10 {
        return S::from_f64(1.0);
    }
    let one = S::from_f64(1.0);
    if a.value() > 5.0 {
        // Large-a asymptotics (the direct form suffers 4a³-scale
        // cancellation): F = u²/9 − u⁴/60 + u⁶/420 − O(u⁸), u = 1/(2a).
        //
        // The u⁶ term is not decoration. Truncating after u⁴ leaves a
        // relative error of 2.15e-6 AT the switch point, so `sr_factor` was
        // discontinuous there by that much — in the ρ ≲ 1e-6 tail of every
        // ITYH-attenuated functional (cam-b3lyp, lc-blyp, lc-pbe, the ωB97
        // family). Including it drops the truncation to ~3e-12 relative.
        //
        // The switch stays at a = 5 deliberately. Moving it up would shrink
        // the series error further but hands MORE of the range to the direct
        // form, whose cancellation error grows like a³ — measured, a = 5 is
        // close to where the two meet, so the residual jump (~1e-9, set by
        // the direct form, not by this series) is as small as it gets
        // without rewriting the direct branch itself.
        let u = (S::from_f64(2.0) * a).recip();
        let u2 = u * u;
        let u4 = u2 * u2;
        return u2 * S::from_f64(1.0 / 9.0) - u4 * S::from_f64(1.0 / 60.0)
            + u4 * u2 * S::from_f64(1.0 / 420.0);
    }
    let a2 = a * a;
    let a3 = a2 * a;
    let inv2a = (S::from_f64(2.0) * a).recip();
    let bracket = S::from_f64(std::f64::consts::PI.sqrt()) * inv2a.erf() - S::from_f64(3.0) * a
        + S::from_f64(4.0) * a3
        + (S::from_f64(2.0) * a - S::from_f64(4.0) * a3) * (-(inv2a * inv2a)).exp();
    one - S::from_f64(8.0 / 3.0) * a * bracket
}

fn ityh_factor<S: Scalar>(rho: S, e_spin: S, omega: S) -> S {
    // Kσ = −2 e_xσ / ρσ^{4/3} (positive).
    let k = -(S::from_f64(2.0) * e_spin) * rho.powf(-4.0 / 3.0);
    if k.value() <= 0.0 {
        return S::from_f64(1.0);
    }
    let a = omega * k.sqrt() / (S::from_f64(6.0 * std::f64::consts::PI.sqrt()) * rho.cbrt());
    sr_factor(a)
}

/// Short-range B88 (ITYH). θ = [beta, omega].
fn b88_sr_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let e_spin = b88_spin(rho, sig, th[0]);
        e = e + e_spin * ityh_factor(rho, e_spin, th[1]);
    }
    e
}

pub fn b88_sr(omega: f64) -> Box<dyn XcFunctional> {
    gga_driver!(
        "b88_sr",
        Kind::Exchange,
        vec![0.0042, omega],
        &["beta", "omega"],
        b88_sr_e
    )
}

/// Short-range PBE exchange (ITYH). θ = [kappa, mu, omega].
///
/// NOTE: this is the ITYH attenuation of PBE, not the wPBE/HJS exchange
/// hole — canonical LC-ωPBE uses the latter; the difference is documented
/// in docs/FUNCTIONALS.md.
fn pbe_x_sr_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let e_spin = pbe_x_spin(rho, sig, th[0], th[1]);
        e = e + e_spin * ityh_factor(rho, e_spin, th[2]);
    }
    e
}

pub fn pbe_x_sr(omega: f64) -> Box<dyn XcFunctional> {
    gga_driver!(
        "pbe_sr",
        Kind::Exchange,
        vec![0.804, 0.2195149727645171, omega],
        &["kappa", "mu", "omega"],
        pbe_x_sr_e
    )
}

// ---------------------------------------------------------------------------
// LYP correlation (Miehlich form). θ = [a, b, c, d].
// ---------------------------------------------------------------------------

fn lyp_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    let (a, b, c, d) = (th[0], th[1], th[2], th[3]);
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    // LYP is an opposite-spin correlation functional: on the physical
    // one-channel boundary (the absent channel also has zero sigma/tau) its
    // energy is exactly zero. Evaluating the general algebra there leaves
    // products such as `ra*rb*rb^(8/3)` split across dual operations; at
    // third order the vanishing prefactor meets a divergent derivative and
    // produces `0*inf = NaN`. Collapse to the exact polarized limit, using
    // the same absent-channel derivative convention as the LDA and B97
    // drivers.
    if super::lda::one_channel_only(ra, rb) {
        return S::from_f64(0.0);
    }
    let one = S::from_f64(1.0);
    let sig = saa + S::from_f64(2.0) * sab + sbb;
    let rm13 = rho.powf(-1.0 / 3.0);
    let denom = one + d * rm13;
    let omega = (-(c * rm13)).exp() / denom * rho.powf(-11.0 / 3.0);
    let delta = c * rm13 + d * rm13 / denom;
    let cf = 0.3 * (3.0 * std::f64::consts::PI * std::f64::consts::PI).powf(2.0 / 3.0);
    let rab = ra * rb;
    let term1 = -a * S::from_f64(4.0) / denom * rab / rho;
    let eleven = S::from_f64(11.0);
    let bracket = rab
        * (S::from_f64(2.0_f64.powf(11.0 / 3.0) * cf) * (ra.powf(8.0 / 3.0) + rb.powf(8.0 / 3.0))
            + (S::from_f64(47.0 / 18.0) - S::from_f64(7.0 / 18.0) * delta) * sig
            - (S::from_f64(2.5) - delta * S::from_f64(1.0 / 18.0)) * (saa + sbb)
            - (delta - eleven) * S::from_f64(1.0 / 9.0) * (ra * saa + rb * sbb) / rho)
        - S::from_f64(2.0 / 3.0) * rho * rho * sig
        + (S::from_f64(2.0 / 3.0) * rho * rho - ra * ra) * sbb
        + (S::from_f64(2.0 / 3.0) * rho * rho - rb * rb) * saa;
    term1 - a * b * omega * bracket
}

pub fn lyp() -> Box<dyn XcFunctional> {
    gga_driver!(
        "lyp",
        Kind::Correlation,
        vec![0.04918, 0.132, 0.2533, 0.349],
        &["a", "b", "c", "d"],
        lyp_e
    )
}

// ---------------------------------------------------------------------------
// PBE correlation on the PW92 local base.
// θ = [beta, gamma] + the 18 PW92 parameters appended.
// ---------------------------------------------------------------------------

/// PBE-H gradient enhancement shared by the general and ferro branches:
/// eps + H(eps, phi, t2).
pub(crate) fn pbe_c_h<S: Scalar>(eps: S, phi: S, rho: S, sig: S, beta: S, gamma: S) -> S {
    let one = S::from_f64(1.0);
    let kf = (S::from_f64(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).powf(1.0 / 3.0);
    let ks = (S::from_f64(4.0 / std::f64::consts::PI) * kf).sqrt();
    let t2 = sig / (S::from_f64(4.0) * phi * phi * ks * ks * rho * rho);
    let phi3 = phi * phi * phi;
    let x = -eps / (gamma * phi3);
    let x0 = x.value();
    let exp0 = x0.exp();
    // Centering leaves every AD/Taylor coefficient exact while exp_m1
    // supplies a cancellation-free zeroth coefficient. The expansion is
    // the exact exponential identity, not a truncated power series.
    let r = if exp0.is_finite() {
        S::from_f64(x0.exp_m1()) + S::from_f64(exp0) * ((x - S::from_f64(x0)).exp() - one)
    } else {
        x.exp() - one
    };
    let q = beta / gamma / r * t2;
    if q.value() >= 1.0 {
        // eps+H = -c*log1p(r/(1+(1+r)*q*(1+q))). Divide numerator
        // and denominator by q^2 to avoid unnecessary tail overflow.
        let inverse = q.recip();
        -gamma
            * phi3
            * (r * inverse * inverse / (inverse * inverse + (one + r) * (one + inverse))).ln_1p()
    } else {
        eps + gamma * phi3 * (beta / gamma * t2 * (one + q) / (one + q + q * q)).ln_1p()
    }
}

/// PBE correlation energy PER PARTICLE ε_c^PBE (general spin densities;
/// meta-GGA correlation functionals compose on top of this).
pub(crate) fn pbe_c_eps<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    beta: S,
    gamma: S,
    pw: &[S],
) -> S {
    let rho = ra + rb;
    let sig = saa + S::from_f64(2.0) * sab + sbb;
    // ζ = ±1 must not reach `phi`. Unlike the `(1∓ζ)^{4/3}` of the LDA
    // ζ-interpolation — which is harmless at first order because
    // `(4/3)·0^{1/3} = 0` — this exponent is 2/3, so `DualN::powf` forms
    // `(2/3)·0^{−1/3} = ∞` at the FIRST derivative and `0·∞ = NaN` lands
    // directly in `vrho`. That is the Kohn–Sham potential itself: a hydrogen
    // atom, H₂⁺, or any N_β = 0 state with pbe/pbe0/revpbe/hse06/hse03/
    // lc-wpbe/lc-pbe returned NaN rather than an energy.
    //
    // The branch is exact, not a regularisation: at ζ = ±1 the spin factor
    // is φ = 2^{−1/3} in closed form and PW92 reduces to ec(rs, 1).
    if super::lda::one_channel_only(ra, rb) {
        return pbe_c_eps_ferro(rho, sig, beta, gamma, pw);
    }
    let one = S::from_f64(1.0);
    let eps = super::lda::pw92_epsilon(ra, rb, pw);
    let zeta = (ra - rb) / rho;
    let phi = ((one + zeta).powf(2.0 / 3.0) + (one - zeta).powf(2.0 / 3.0)) * S::from_f64(0.5);
    pbe_c_h(eps, phi, rho, sig, beta, gamma)
}

/// ε_c^PBE in the fully polarized (ζ = 1) branch: φ = 2^{-1/3} exactly and
/// the PW92 ferro fit — avoids the (1−ζ)^{2/3} dual singularity at ζ = 1
/// (same reasoning as `lda::pw92_epsilon_ferro`).
pub(crate) fn pbe_c_eps_ferro<S: Scalar>(rho: S, sig: S, beta: S, gamma: S, pw: &[S]) -> S {
    let eps = super::lda::pw92_epsilon_ferro(rho, pw);
    // φ(ζ=1) = (2^{2/3} + 0)/2 = 2^{−1/3}.
    let phi = S::from_f64(2.0f64.powf(2.0 / 3.0) * 0.5);
    pbe_c_h(eps, phi, rho, sig, beta, gamma)
}

fn pbe_c_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    let (beta, gamma) = (th[0], th[1]);
    let pw = &th[2..20];
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    rho * pbe_c_eps(ra, rb, saa, sab, sbb, beta, gamma, pw)
}

pub fn pbe_c() -> Box<dyn XcFunctional> {
    let mut params = vec![0.06672455060314922, 0.031090690869654895];
    // PBE's local base is the "modified" PW92 parameter set (A values to
    // more digits than the PW92 paper tables) — the reference-implementation
    // convention, required to match other codes to sub-µEh.
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    gga_driver!(
        "pbe",
        Kind::Correlation,
        params,
        &[
            "beta", "gamma", "A_0", "a1_0", "b1_0", "b2_0", "b3_0", "b4_0", "A_1", "a1_1", "b1_1",
            "b2_1", "b3_1", "b4_1", "A_a", "a1_a", "b1_a", "b2_a", "b3_a", "b4_a"
        ],
        pbe_c_e
    )
}

/// PBEsol exchange (Perdew et al., PRL 100, 136406 (2008)): PBE exchange
/// with the gradient-expansion coefficient mu = 10/81.
pub fn pbesol_x() -> Box<dyn XcFunctional> {
    pbe_x()
        .with_params(&[0.804, 10.0 / 81.0])
        .expect("pbesol exchange params")
}

/// PBE-form correlation with another beta and gamma.
fn pbe_c_with(beta: f64, gamma: f64) -> Box<dyn XcFunctional> {
    let base = pbe_c();
    let mut params = base.params();
    params[0] = beta;
    params[1] = gamma;
    base.with_params(&params).expect("PBE correlation params")
}

/// xPBE correlation (Xu-Goddard 2004): beta = 0.089809, gamma = 0.0204335576602504.
pub fn xpbe_c() -> Box<dyn XcFunctional> {
    pbe_c_with(0.089809, 0.0204335576602504)
}

/// APBE correlation: beta = 3 mu_APBE / pi^2 with mu = 0.26.
pub fn apbe_c() -> Box<dyn XcFunctional> {
    pbe_c_with(
        3.0 * 0.26 / (std::f64::consts::PI * std::f64::consts::PI),
        0.031090690869654895,
    )
}

/// PBEint correlation: beta = 0.052.
pub fn pbeint_c() -> Box<dyn XcFunctional> {
    pbe_c_with(0.052, 0.031090690869654895)
}

/// PBEsol correlation: PBE correlation with beta = 0.046.
pub fn pbesol_c() -> Box<dyn XcFunctional> {
    let base = pbe_c();
    let mut params = base.params();
    params[0] = 0.046;
    base.with_params(&params)
        .expect("pbesol correlation params")
}

// ---------------------------------------------------------------------------
// RPBE exchange (Hammer-Hansen-Norskov, PRB 59, 7413 (1999)).
// F_x = 1 + kappa (1 - exp(-mu s^2 / kappa)). theta = [kappa, mu].
// ---------------------------------------------------------------------------

fn rpbe_x_spin<S: Scalar>(rho: S, sig: S, kappa: S, mu: S) -> S {
    spin_scaled_exchange(rho, sig, |s2| rpbe_enhancement(s2, kappa, mu))
}

pub(crate) fn rpbe_enhancement<S: Scalar>(s2: S, kappa: S, mu: S) -> S {
    S::from_f64(1.0) + kappa * (S::from_f64(1.0) - (-(mu * s2) / kappa).exp())
}

fn rpbe_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    rpbe_x_spin(ra, saa, th[0], th[1]) + rpbe_x_spin(rb, sbb, th[0], th[1])
}

pub fn rpbe_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "rpbe",
        Kind::Exchange,
        vec![0.804, 0.2195149727645171],
        &["kappa", "mu"],
        rpbe_x_e
    )
}

// ---------------------------------------------------------------------------
// PW91 exchange (Perdew-Wang 1991; Perdew et al., PRB 46, 6671 (1992)).
// F_x = 1 + [(c + d exp(-alpha s^2)) s^2 - f s^4] / [1 + a s asinh(b s) + f s^4],
// theta = [a, b, c, d, f, alpha]. `s asinh(b s)` is formed as an analytic
// function of s^2 (see `x_asinh_x`).
// ---------------------------------------------------------------------------

fn pw91_x_spin<S: Scalar>(rho: S, sig: S, th: &[S]) -> S {
    let (a, b, c, d, f, alpha) = (th[0], th[1], th[2], th[3], th[4], th[5]);
    spin_scaled_exchange(rho, sig, |s2| {
        let one = S::from_f64(1.0);
        // s asinh(b s) = (b s) asinh(b s) / b.
        let sas = x_asinh_x(b * b * s2) / b;
        let numerator = (c + d * (-(alpha * s2)).exp()) * s2 - f * s2 * s2;
        let denominator = one + a * sas + f * s2 * s2;
        one + numerator / denominator
    })
}

fn pw91_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    pw91_x_spin(ra, saa, th) + pw91_x_spin(rb, sbb, th)
}

pub fn pw91_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "pw91",
        Kind::Exchange,
        vec![0.19645, 7.7956, 0.2743, -0.1508, 0.004, 100.0],
        &["a", "b", "c", "d", "f", "alpha"],
        pw91_x_e
    )
}

// ---------------------------------------------------------------------------
// Enhancement-factor exchanges on `spin_scaled_exchange`. `s2` is the
// squared reduced gradient of the spin-scaled density; `X2S` converts the
// per-spin `x = |grad rho_s| / rho_s^(4/3)` to `s`; `X_FACTOR_C` is minus
// the per-spin LDA prefactor.
// ---------------------------------------------------------------------------

use super::common::{X2S, X_FACTOR_C};

/// PW86 (Perdew-Wang, PRB 33, 8800 (1986)):
/// `F = (1 + a s^2 + b s^4 + c s^6)^(1/15)`. theta = [a, b, c].
fn pw86_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let f = |s2: S| pw86_enhancement(s2, [th[0], th[1], th[2]]);
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub(crate) fn pw86_enhancement<S: Scalar>(s2: S, [a, b, c]: [S; 3]) -> S {
    (S::from_f64(1.0) + a * s2 + b * s2 * s2 + c * s2 * s2 * s2).powf(1.0 / 15.0)
}

pub fn pw86_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "pw86",
        Kind::Exchange,
        vec![1.296, 14.0, 0.2],
        &["a", "b", "c"],
        pw86_x_e
    )
}

/// mPW91 exchange (Adamo-Barone, JCP 108, 664 (1998)): the generalized
/// PW91 form of [`super::dh_components::pw_exchange`] with bt = 0.00426,
/// the Gaussian exponent 100 s^2 = 1.6455 x^2 and the power 3.72.
pub fn mpw91_x() -> Box<dyn XcFunctional> {
    super::dh_components::pw_form_exchange("mpw91", [0.00426, 100.0 * X2S * X2S, 3.72])
}

/// G96 exchange (Gill, Mol. Phys. 89, 433 (1996)): `F = 1 + x^(3/2) / (137 X_FACTOR_C)`.
fn g96_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        let absent = sigma_is_absent(sig);
        e = e + spin_scaled_exchange(rho, sig, |s2| {
            if absent {
                S::from_f64(1.0)
            } else {
                // x^(3/2) = (s^2)^(3/4) / X2S^(3/2).
                S::from_f64(1.0)
                    + th[0] / S::from_f64(X_FACTOR_C * X2S.powf(1.5))
                        * (S::from_f64(0.75) * s2.ln()).exp()
            }
        });
    }
    e
}

pub fn g96_x() -> Box<dyn XcFunctional> {
    gga_driver!("g96", Kind::Exchange, vec![1.0 / 137.0], &["c1"], g96_x_e)
}

/// LG93 exchange (Lacks-Gordon, PRA 47, 4681 (1993)):
/// `F = (1 + a2 s^2 + ... + a12 s^12)^b / (1 + ad s^2)`.
/// theta = [ad, b, a2, a4, a6, a8, a10, a12].
fn lg93_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let f = |s2: S| {
        let mut poly = S::from_f64(1.0);
        let mut power = s2;
        for k in 2..8 {
            poly = poly + th[k] * power;
            power = power * s2;
        }
        (th[1] * poly.ln()).exp() / (S::from_f64(1.0) + th[0] * s2)
    };
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub fn lg93_x() -> Box<dyn XcFunctional> {
    let (ad, b) = (1e-8, 0.024974);
    gga_driver!(
        "lg93",
        Kind::Exchange,
        vec![
            ad,
            b,
            (ad + 0.1234) / b,
            29.790,
            22.417,
            12.119,
            1570.1,
            55.944
        ],
        &["ad", "b", "a2", "a4", "a6", "a8", "a10", "a12"],
        lg93_x_e
    )
}

/// FT97-B exchange (Filatov-Thiel, Mol. Phys. 91, 847 (1997)):
/// `F = 1 + beta x^2 / (X_FACTOR_C sqrt(1 + 9 x^2 beta^2 asinh(x^2)^2))`,
/// `beta = beta0 + beta1 sigma_s / (beta2 + sigma_s)`. theta = [beta0, beta1, beta2].
fn ft97_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r43 = rho.powf(4.0 / 3.0);
        let x2 = sig / (r43 * r43);
        let beta = th[0] + th[1] * sig / (th[2] + sig);
        let a = x2.asinh();
        let f = S::from_f64(1.0)
            + beta * x2
                / (S::from_f64(X_FACTOR_C)
                    * (S::from_f64(1.0) + S::from_f64(9.0) * x2 * beta * beta * a * a).sqrt());
        e = e - S::from_f64(X_FACTOR_C) * r43 * f;
    }
    e
}

pub fn ft97_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "ft97",
        Kind::Exchange,
        vec![0.002913644, 0.0009474169, 6255746.320201],
        &["beta0", "beta1", "beta2"],
        ft97_x_e; scaling = &[0, 0, 8]
    )
}

/// xPBE exchange (Xu-Goddard, JCP 121, 4068 (2004)): PBE form.
pub fn xpbe_x() -> Box<dyn XcFunctional> {
    pbe_x()
        .with_params(&[0.91954, 0.23214])
        .expect("xpbe params")
}

/// APBE exchange (Constantin et al., PRL 106, 186406 (2011)): PBE form, mu = 0.26.
pub fn apbe_x() -> Box<dyn XcFunctional> {
    pbe_x().with_params(&[0.804, 0.26]).expect("apbe params")
}

/// SOGGA exchange (Zhao-Truhlar, JCP 128, 184109 (2008)): the mean of the
/// PBE and RPBE forms with kappa = 0.552 and mu = 10/81.
fn sogga_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let (kappa, mu) = (th[0], th[1]);
    let f = |s2: S| {
        let y = mu * s2 / kappa;
        S::from_f64(1.0)
            + kappa
                * (S::from_f64(1.0)
                    - S::from_f64(0.5) / (S::from_f64(1.0) + y)
                    - S::from_f64(0.5) * (-y).exp())
    };
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub fn sogga_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "sogga",
        Kind::Exchange,
        vec![0.552, 10.0 / 81.0],
        &["kappa", "mu"],
        sogga_x_e
    )
}

/// mPBE exchange (Adamo-Barone, JCP 116, 5933 (2002)):
/// `F = 1 + c1 y/(1 + a y) + c2 (y/(1 + a y))^2`, `y = s^2`. theta = [a, c1, c2].
fn mpbe_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let f = |s2: S| {
        let q = s2 / (S::from_f64(1.0) + th[0] * s2);
        S::from_f64(1.0) + th[1] * q + th[2] * q * q
    };
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub fn mpbe_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "mpbe",
        Kind::Exchange,
        vec![0.157, 0.21951, -0.015],
        &["a", "c1", "c2"],
        mpbe_x_e
    )
}

/// PBEint exchange (Fabiano-Constantin-Della Sala, PRB 82, 113104 (2010)):
/// PBE form with `mu(s) = muGE + (muPBE - muGE) alpha s^2 / (1 + alpha s^2)`.
/// theta = [kappa, alpha, muPBE, muGE].
fn pbeint_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let f = |s2: S| {
        let mu = th[3] + (th[2] - th[3]) * th[1] * s2 / (S::from_f64(1.0) + th[1] * s2);
        S::from_f64(1.0) + th[0] - th[0] / (S::from_f64(1.0) + mu * s2 / th[0])
    };
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub fn pbeint_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "pbeint",
        Kind::Exchange,
        vec![0.804, 0.197, 0.2195149727645171, 10.0 / 81.0],
        &["kappa", "alpha", "muPBE", "muGE"],
        pbeint_x_e
    )
}

/// Wu-Cohen exchange (PRB 73, 235116 (2006)): PBE form with
/// `x = muGE s^2 + (mu - muGE) s^2 e^(-s^2) + ln(1 + c s^4)`. theta = [kappa, mu, c].
fn wc_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let f = |s2: S| {
        let x = S::from_f64(10.0 / 81.0) * s2
            + (th[1] - S::from_f64(10.0 / 81.0)) * s2 * (-s2).exp()
            + (S::from_f64(1.0) + th[2] * s2 * s2).ln();
        S::from_f64(1.0) + th[0] - th[0] / (S::from_f64(1.0) + x / th[0])
    };
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub fn wc_x() -> Box<dyn XcFunctional> {
    let mu = 0.2195149727645171;
    let c = 146.0 / 2025.0 * 4.0 / 9.0 - 73.0 / 405.0 * 2.0 / 3.0 + (mu - 10.0 / 81.0);
    gga_driver!(
        "wc",
        Kind::Exchange,
        vec![0.804, mu, c],
        &["kappa", "mu", "c"],
        wc_x_e
    )
}

/// Keal-Tozer exchange gradient term (JCP 119, 3015 (2003)):
/// `e = -X_FACTOR_C rho_s^(4/3) + gamma sigma_s / (rho_s^(4/3) + delta)` per spin.
/// theta = [gamma, delta].
fn kt1_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r43 = rho.powf(4.0 / 3.0);
        e = e - S::from_f64(X_FACTOR_C) * r43 + th[0] * sig / (r43 + th[1]);
    }
    e
}

pub fn kt1_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "kt1",
        Kind::Exchange,
        vec![-0.006, 0.1],
        &["gamma", "delta"],
        kt1_x_e; scaling = &[0, 4]
    )
}

/// The Keal-Tozer gradient term of KT3 (JCP 121, 5654 (2004)): gamma = -0.004.
pub fn kt3_x() -> Box<dyn XcFunctional> {
    kt1_x().with_params(&[-0.004, 0.1]).expect("kt3 params")
}

// ---------------------------------------------------------------------------
// Catalogue batch 2b.
// ---------------------------------------------------------------------------

/// AM05 exchange (Armiento-Mattsson, PRB 72, 085108 (2005)):
/// `F = 1 + alpha s^2/(1 + alpha s^2) * c s^2 (fb - 1)/(fb + c s^2)`, with the
/// Airy-gas factor `fb - 1 = exp(2 W / 3) / (1 + xi^2 / d)^(1/4) - 1`,
/// `W = W(s^(3/2) / (2 sqrt 6))`, `xi^2 = (3 W / 2)^(4/3)`. theta = [alpha, c].
fn am05_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let (alpha, c) = (th[0], th[1]);
    let d = 28.237_057_402_489_32;
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        let absent = sigma_is_absent(sig);
        e = e + spin_scaled_exchange(rho, sig, |p| {
            if absent {
                return S::from_f64(1.0);
            }
            let y = (S::from_f64(0.75) * p.ln()).exp() / S::from_f64(2.0 * 6.0_f64.sqrt());
            let w = super::common::lambert_w(y);
            let xi2 = (S::from_f64(1.5) * w).powf(4.0 / 3.0);
            let fb_m1 = (S::from_f64(2.0 / 3.0) * w
                - S::from_f64(0.25) * (xi2 / S::from_f64(d)).ln_1p())
            .exp()
                - S::from_f64(1.0);
            let flaa_m1 = c * p * fb_m1 / (S::from_f64(1.0) + fb_m1 + c * p);
            S::from_f64(1.0) + alpha * p / (S::from_f64(1.0) + alpha * p) * flaa_m1
        });
    }
    e
}

pub fn am05_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "am05",
        Kind::Exchange,
        vec![2.804, 0.7168],
        &["alpha", "c"],
        am05_x_e
    )
}

/// AM05 correlation: `eps_PW (rho_a X_a + rho_b X_b) / rho` with
/// `X_s = (1 + gamma alpha s_s^2) / (1 + alpha s_s^2)` of the spin-scaled
/// reduced gradient. theta = [alpha, gamma, PW92 parameters].
fn am05_c_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let (alpha, gamma) = (th[0], th[1]);
    let eps = super::lda::pw92_epsilon(ra, rb, &th[2..20]);
    let mut weighted = S::from_f64(0.0);
    for (r, sig) in [(ra, saa), (rb, sbb)] {
        if r.value() < RHO_CUTOFF {
            continue;
        }
        let r43 = r.powf(4.0 / 3.0);
        let s2 = S::from_f64(X2S * X2S) * sig / (r43 * r43);
        weighted = weighted
            + r * (S::from_f64(1.0) + gamma * alpha * s2) / (S::from_f64(1.0) + alpha * s2);
    }
    eps * weighted
}

pub fn am05_c() -> Box<dyn XcFunctional> {
    let mut params = vec![2.804, 0.8098];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    gga_driver!("am05", Kind::Correlation, params, &[], am05_c_e)
}

/// N12 / GAM exchange (Peverati-Truhlar, JCTC 8, 2310 (2012); PCCP 17,
/// 12146 (2015)): `F = sum_ij CC_ij v^i u^j`, `u = gamma x^2/(1 + gamma x^2)`,
/// `v = omega rho_s^(1/3)/(1 + omega rho_s^(1/3))`. theta = [CC (16), omega, gamma].
fn n12_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let (omega, gamma) = (th[16], th[17]);
    let mut e = S::from_f64(0.0);
    for (rho, sig) in [(ra, saa), (rb, sbb)] {
        if rho.value() < RHO_CUTOFF {
            continue;
        }
        let r13 = rho.cbrt();
        let r43 = rho * r13;
        let x2 = sig / (r43 * r43);
        let u = gamma * x2 / (S::from_f64(1.0) + gamma * x2);
        let v = omega * r13 / (S::from_f64(1.0) + omega * r13);
        let mut f = S::from_f64(0.0);
        let mut vi = S::from_f64(1.0);
        for i in 0..4 {
            let mut row = S::from_f64(0.0);
            let mut uj = S::from_f64(1.0);
            for j in 0..4 {
                row = row + th[4 * i + j] * uj;
                uj = uj * u;
            }
            f = f + row * vi;
            vi = vi * v;
        }
        e = e - S::from_f64(X_FACTOR_C) * r43 * f;
    }
    e
}

const N12_NAMES: &[&str] = &[
    "CC00", "CC01", "CC02", "CC03", "CC10", "CC11", "CC12", "CC13", "CC20", "CC21", "CC22", "CC23",
    "CC30", "CC31", "CC32", "CC33", "omega_v", "gamma",
];

/// `omega_v` multiplies `rho_s^(1/3)`: a length, scaling power -1.
const N12_SCALING: &[i32] = &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0];

fn n12_family(name: &'static str, cc: [f64; 16]) -> Box<dyn XcFunctional> {
    let mut params = cc.to_vec();
    params.extend_from_slice(&[2.5, 0.004]);
    gga_driver!(name, Kind::Exchange, params, N12_NAMES, n12_x_e; scaling = N12_SCALING)
}

pub fn n12_x() -> Box<dyn XcFunctional> {
    n12_family(
        "n12",
        [
            1.0, 0.50788, 0.168233, 0.128887, 0.0860211, -17.1008, 65.0814, -70.1726, -0.390755,
            51.3392, -166.22, 142.738, 0.403611, -34.4631, 76.1661, -2.41834,
        ],
    )
}

pub fn gam_x() -> Box<dyn XcFunctional> {
    n12_family(
        "gam",
        [
            1.3273, 0.886102, -5.73833, 8.60197, -0.786018, -4.78787, 3.90989, -2.11611, 0.802575,
            14.4363, 8.42735, -6.21552, -0.142331, -13.4598, 1.52355, -10.053,
        ],
    )
}

/// N12-SX nonseparable local exchange, PCCP 14, 16187 (2012).
/// Coefficients use rows of density powers and columns of gradient powers;
/// this resolves the interchanged first two indices in the printed table.
/// The fitted local part is used whole, alongside separate screened HF.
pub fn n12_sx_x() -> Box<dyn XcFunctional> {
    n12_family(
        "n12-sx",
        [
            0.681116, 1.88858, 1.78590, 0.879456, -0.0812270, -1.08723, -4.18682, -30.0, 0.536236,
            -5.45678, 30.0, 55.1105, -0.709913, 13.0001, -72.4877, 29.8363,
        ],
    )
}

/// SOGGA11 exchange (Peverati-Zhao-Truhlar, JPCL 2, 1991 (2011)):
/// `F = sum_i a_i f0^i + sum_i b_i f1^i`, `f0 = y/(1 + y)`, `f1 = 1 - e^(-y)`,
/// `y = mu s^2 / kappa`. theta = [kappa, mu, a0..a5, b0..b5].
fn sogga11_x_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    let f = |s2: S| {
        let y = th[1] * s2 / th[0];
        let f0 = y / (S::from_f64(1.0) + y);
        let f1 = S::from_f64(1.0) - (-y).exp();
        let mut total = S::from_f64(0.0);
        let (mut p0, mut p1) = (S::from_f64(1.0), S::from_f64(1.0));
        for i in 0..6 {
            total = total + th[2 + i] * p0 + th[8 + i] * p1;
            p0 = p0 * f0;
            p1 = p1 * f1;
        }
        total
    };
    spin_scaled_exchange(ra, saa, f) + spin_scaled_exchange(rb, sbb, f)
}

pub fn sogga11_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "sogga11",
        Kind::Exchange,
        vec![
            0.552,
            10.0 / 81.0,
            0.5,
            -2.95535,
            15.7974,
            -91.1804,
            96.203,
            0.18683,
            0.5,
            3.50743,
            -12.9523,
            49.787,
            -33.2545,
            -11.1396,
        ],
        &["kappa", "mu", "a0", "a1", "a2", "a3", "a4", "a5", "b0", "b1", "b2", "b3", "b4", "b5"],
        sogga11_x_e
    )
}

/// SOGGA11 correlation: `eps_PW (sum_i a_i f0^i + sum_i b_i f1^i)` with
/// `y = -beta phi^3 t^2 / eps_PW`, beta = 15.75592 * 0.004235.
/// theta = [a0..a5, b0..b5, PW92 parameters].
fn sogga11_c_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let pw = &th[12..30];
    let one = S::from_f64(1.0);
    let (eps, phi) = if super::lda::one_channel_only(ra, rb) {
        (
            super::lda::pw92_epsilon_ferro(rho, pw),
            S::from_f64(2.0f64.powf(-1.0 / 3.0)),
        )
    } else {
        let z = (ra - rb) / rho;
        (
            super::lda::pw92_epsilon(ra, rb, pw),
            ((one + z).powf(2.0 / 3.0) + (one - z).powf(2.0 / 3.0)) * S::from_f64(0.5),
        )
    };
    let sig = saa + S::from_f64(2.0) * sab + sbb;
    let rs = super::common::wigner_seitz(rho);
    let r43 = rho.powf(4.0 / 3.0);
    let xt2 = sig / (r43 * r43);
    let alpha = 15.75592 * 0.004235 / (16.0 * 2.0f64.powf(2.0 / 3.0));
    let y = -(S::from_f64(alpha) * phi * xt2) / (rs * eps);
    let f0 = y / (one + y);
    let f1 = one - (-y).exp();
    let mut total = S::from_f64(0.0);
    let (mut p0, mut p1) = (one, one);
    for i in 0..6 {
        total = total + th[i] * p0 + th[6 + i] * p1;
        p0 = p0 * f0;
        p1 = p1 * f1;
    }
    rho * eps * total
}

pub fn sogga11_c() -> Box<dyn XcFunctional> {
    let mut params = vec![
        0.5, -4.62334, 8.0041, -130.226, 38.2685, 69.5599, 0.5, 3.62334, 9.36393, 34.5114,
        -18.5684, -0.16519,
    ];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    gga_driver!("sogga11", Kind::Correlation, params, &[], sogga11_c_e)
}

/// SOGGA11-X exchange (Peverati and Truhlar, JCP 135, 191102 (2011)): the
/// SOGGA11 form with the hybrid's coefficients, which already carry the
/// `1 - 0.4015` semilocal fraction.
pub fn sogga11x_x() -> Box<dyn XcFunctional> {
    gga_driver!(
        "sogga11x",
        Kind::Exchange,
        vec![
            0.552,
            10.0 / 81.0,
            0.29925,
            3.21638,
            -3.55605,
            7.65852,
            -11.283,
            5.25813,
            0.29925,
            -2.88595,
            3.23617,
            -2.45393,
            -3.75495,
            3.96613,
        ],
        &["kappa", "mu", "a0", "a1", "a2", "a3", "a4", "a5", "b0", "b1", "b2", "b3", "b4", "b5"],
        sogga11_x_e
    )
}

/// SOGGA11-X correlation: the SOGGA11 form with the hybrid's coefficients.
pub fn sogga11x_c() -> Box<dyn XcFunctional> {
    let mut params = vec![
        0.5, 78.2439, 25.7211, -13.883, -9.87375, -14.1357, 0.5, -79.2439, 16.3725, 2.08129,
        7.50769, -10.1861,
    ];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    gga_driver!("sogga11x", Kind::Correlation, params, &[], sogga11_c_e)
}

/// PBEloc correlation (Constantin et al., PRB 86, 035130 (2012)): PBE with
/// `beta = 0.0375 + 0.08 t^2 (1 - exp(-rs^2))`.
pub(crate) fn pbeloc_c_eps<S: Scalar>(
    ra: S,
    rb: S,
    saa: S,
    sab: S,
    sbb: S,
    beta_parameters: [S; 2],
    gamma: S,
    pw: &[S],
) -> S {
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let one = S::from_f64(1.0);
    let sig = saa + S::from_f64(2.0) * sab + sbb;
    let (eps, phi) = if super::lda::one_channel_only(ra, rb) {
        (
            super::lda::pw92_epsilon_ferro(rho, pw),
            S::from_f64(2.0f64.powf(-1.0 / 3.0)),
        )
    } else {
        let z = (ra - rb) / rho;
        (
            super::lda::pw92_epsilon(ra, rb, pw),
            ((one + z).powf(2.0 / 3.0) + (one - z).powf(2.0 / 3.0)) * S::from_f64(0.5),
        )
    };
    let rs = super::common::wigner_seitz(rho);
    let kf = (S::from_f64(3.0 * std::f64::consts::PI * std::f64::consts::PI) * rho).cbrt();
    let ks = (S::from_f64(4.0 / std::f64::consts::PI) * kf).sqrt();
    let t2 = sig / (S::from_f64(4.0) * phi * phi * ks * ks * rho * rho);
    let beta = beta_parameters[0] + beta_parameters[1] * t2 * (one - (-(rs * rs)).exp());
    pbe_c_h(eps, phi, rho, sig, beta, gamma)
}

fn pbeloc_c_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    rho * pbeloc_c_eps(
        ra,
        rb,
        saa,
        sab,
        sbb,
        [S::from_f64(0.0375), S::from_f64(0.08)],
        th[0],
        &th[1..19],
    )
}

pub fn pbeloc_c() -> Box<dyn XcFunctional> {
    let mut params = vec![0.031090690869654895];
    params.extend_from_slice(&super::lda::PW92_MOD_PARAMS);
    gga_driver!("pbeloc", Kind::Correlation, params, &[], pbeloc_c_e)
}

/// vPBE correlation: the PBE gradient correction on the VWN5 local
/// correlation. theta = [beta, gamma, VWN5 parameters].
fn pbe_vwn_c_e<S: Scalar>(ra: S, rb: S, saa: S, sab: S, sbb: S, th: &[S]) -> S {
    let rho = ra + rb;
    if rho.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let one = S::from_f64(1.0);
    let sig = saa + S::from_f64(2.0) * sab + sbb;
    let eps = super::lda::vwn5_energy(ra, rb, &th[2..14]) / rho;
    let phi = if super::lda::one_channel_only(ra, rb) {
        S::from_f64(2.0f64.powf(-1.0 / 3.0))
    } else {
        let z = (ra - rb) / rho;
        ((one + z).powf(2.0 / 3.0) + (one - z).powf(2.0 / 3.0)) * S::from_f64(0.5)
    };
    rho * pbe_c_h(eps, phi, rho, sig, th[0], th[1])
}

pub fn pbe_vwn_c() -> Box<dyn XcFunctional> {
    let mut params = vec![0.06672455060314922, 0.031090690869654895];
    params.extend_from_slice(&super::lda::VWN5_PARAMS);
    gga_driver!("vpbe", Kind::Correlation, params, &[], pbe_vwn_c_e)
}

/// OP correlation with B88 exchange (Tsuneda-Suzumura-Hirao, JCP 110, 10664
/// (1999)): `e = -rho_a rho_b (a1 b + a2)/(b^4 + b1 b^3 + b2 b^2)` per volume,
/// `b = q rho_a^(1/3) rho_b^(1/3) K_a K_b / (rho_a^(1/3) K_a + rho_b^(1/3) K_b)`,
/// `K_s = -2 e_x,s / rho_s^(4/3)` of B88. theta = [q, a1, a2, b1, b2, beta_B88].
fn op_b88_c_e<S: Scalar>(ra: S, rb: S, saa: S, _sab: S, sbb: S, th: &[S]) -> S {
    if ra.value() < RHO_CUTOFF || rb.value() < RHO_CUTOFF {
        return S::from_f64(0.0);
    }
    let k = |rho: S, sig: S| -> S {
        S::from_f64(-2.0) * b88_spin(rho, sig, th[5]) / rho.powf(4.0 / 3.0)
    };
    let (ka, kb) = (k(ra, saa), k(rb, sbb));
    let (a13, b13) = (ra.cbrt(), rb.cbrt());
    let beta = th[0] * a13 * b13 * ka * kb / (a13 * ka + b13 * kb);
    let b2 = beta * beta;
    -(ra * rb) * (th[1] * beta + th[2]) / (b2 * b2 + th[3] * b2 * beta + th[4] * b2)
}

pub fn op_b88_c() -> Box<dyn XcFunctional> {
    gga_driver!(
        "opb88",
        Kind::Correlation,
        // The published four-digit constants (b1 and b2 are not 2/sqrt(pi)
        // and 1/pi, which they resemble).
        vec![
            2.3670,
            1.5214,
            0.5764,
            11_284.0 / 10_000.0,
            3_183.0 / 10_000.0,
            0.0042
        ],
        &["q", "a1", "a2", "b1", "b2", "beta"],
        op_b88_c_e
    )
}

/// EDF1 (Adamson-Gill-Pople, CPL 284, 6 (1998)): -0.922818 Slater +
/// 10.4017 B88(beta = 0.0035) - 8.44793 B88(beta = 0.0042) + LYP with
/// a = 0.055, b = 0.158, c = 0.25, d = 0.3505.
pub fn edf1_parts() -> Vec<(f64, Box<dyn XcFunctional>)> {
    vec![
        (-0.922818, super::lda::slater()),
        (10.4017, b88().with_params(&[0.0035]).expect("edf1 b88")),
        (-8.44793, b88()),
        (
            1.0,
            lyp()
                .with_params(&[0.055, 0.158, 0.25, 0.3505])
                .expect("edf1 lyp"),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_pt(f: &dyn XcFunctional, order: usize, vars5: [f64; 5]) -> XcOut {
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
        out.prepare(1, order, Family::Gga);
        f.eval_add(&vars, order, 1.0, &mut out).unwrap();
        out
    }

    /// σ → 0 limits: B88/PBE-X collapse to Slater, OPTX to a1·Slater,
    /// PBE-C to PW92.
    #[test]
    fn zero_gradient_lda_limits() {
        let (ra, rb) = (0.53, 0.29);
        let z = [ra, rb, 0.0, 0.0, 0.0];
        let slater = crate::xc::lda::slater();
        let e_slater = {
            let rho_a = [ra];
            let rho_b = [rb];
            let vars = GridVars {
                rho: [&rho_a, &rho_b],
                sigma: None,
                tau: None,
            };
            let mut out = XcOut::default();
            out.prepare(1, 0, Family::Lda);
            slater.eval_add(&vars, 0, 1.0, &mut out).unwrap();
            out.e[0]
        };
        assert!((eval_pt(b88().as_ref(), 0, z).e[0] - e_slater).abs() < 1e-13);
        assert!((eval_pt(pbe_x().as_ref(), 0, z).e[0] - e_slater).abs() < 1e-12);
        let optx_e0 = eval_pt(optx().as_ref(), 0, z).e[0];
        assert!((optx_e0 - 1.05151 * e_slater).abs() < 1e-12);
        // PBE-C → PW92 with the "modified" (reference-implementation)
        // parameter digits that PBE builds on.
        let pw92 = {
            let base = crate::xc::lda::pw92();
            base.with_params(&crate::xc::lda::PW92_MOD_PARAMS).unwrap()
        };
        let e_pw = {
            let rho_a = [ra];
            let rho_b = [rb];
            let vars = GridVars {
                rho: [&rho_a, &rho_b],
                sigma: None,
                tau: None,
            };
            let mut out = XcOut::default();
            out.prepare(1, 0, Family::Lda);
            pw92.eval_add(&vars, 0, 1.0, &mut out).unwrap();
            out.e[0]
        };
        assert!(
            (eval_pt(pbe_c().as_ref(), 0, z).e[0] - e_pw).abs() < 1e-12,
            "{} vs {}",
            eval_pt(pbe_c().as_ref(), 0, z).e[0],
            e_pw
        );
    }

    /// All first derivatives (ρ and σ) against central differences.
    #[test]
    fn gga_first_derivatives_match_fd() {
        let pts = [
            [0.4, 0.3, 0.05, 0.02, 0.04],
            [0.9, 0.1, 0.3, 0.1, 0.2],
            [0.2, 0.2, 0.01, 0.005, 0.01],
        ];
        for f in [
            b88(),
            pbe_x(),
            optx(),
            lyp(),
            pbe_c(),
            b88_sr(0.33),
            pbe_x_sr(0.47),
        ] {
            for &p in &pts {
                let out = eval_pt(f.as_ref(), 1, p);
                let h = 1e-7;
                for var in 0..5 {
                    let mut up = p;
                    up[var] += h;
                    let mut dn = p;
                    dn[var] -= h;
                    let fd = (eval_pt(f.as_ref(), 0, up).e[0] - eval_pt(f.as_ref(), 0, dn).e[0])
                        / (2.0 * h);
                    let an = if var < 2 {
                        out.vrho[var][0]
                    } else {
                        out.vsigma[var - 2][0]
                    };
                    assert!(
                        (an - fd).abs() < 2e-6 * fd.abs().max(1.0),
                        "{} var {var}: {an} vs {fd} at {p:?}",
                        f.name()
                    );
                }
            }
        }
    }

    /// Second derivatives against FD of analytic first derivatives.
    #[test]
    fn gga_second_derivatives_match_fd() {
        let p = [0.5, 0.35, 0.08, 0.03, 0.06];
        for f in [b88(), pbe_x(), lyp(), pbe_c()] {
            let out2 = eval_pt(f.as_ref(), 2, p);
            let h = 1e-6;
            let first = |q: [f64; 5]| -> [f64; 5] {
                let o = eval_pt(f.as_ref(), 1, q);
                [
                    o.vrho[0][0],
                    o.vrho[1][0],
                    o.vsigma[0][0],
                    o.vsigma[1][0],
                    o.vsigma[2][0],
                ]
            };
            let pairs_rr = [(0usize, 0usize, 0usize), (0, 1, 1), (1, 1, 2)];
            for &(p1, p2, blk) in &pairs_rr {
                let mut up = p;
                up[p2] += h;
                let mut dn = p;
                dn[p2] -= h;
                let fd = (first(up)[p1] - first(dn)[p1]) / (2.0 * h);
                assert!(
                    (out2.v2rho2[blk][0] - fd).abs() < 5e-5 * fd.abs().max(1.0),
                    "{} v2rho2[{blk}]: {} vs {fd}",
                    f.name(),
                    out2.v2rho2[blk][0]
                );
            }
            // Spot-check a ρσ cross block and a σσ block.
            let mut up = p;
            up[2] += h;
            let mut dn = p;
            dn[2] -= h;
            let fd_rs = (first(up)[0] - first(dn)[0]) / (2.0 * h);
            assert!(
                (out2.v2rhosigma[0][0] - fd_rs).abs() < 5e-5 * fd_rs.abs().max(1.0),
                "{} v2rhosigma[0]",
                f.name()
            );
            let fd_ss = (first(up)[2] - first(dn)[2]) / (2.0 * h);
            assert!(
                (out2.v2sigma2[0][0] - fd_ss).abs() < 5e-5 * fd_ss.abs().max(1.0),
                "{} v2sigma2[0]",
                f.name()
            );
        }
    }

    /// ITYH short-range limits: ω → 0 recovers the full GGA; large ω kills
    /// the short-range part. Both attenuated parents are covered — SR-B88
    /// backs CAM-B3LYP and LC-BLYP, SR-PBE backs LC-PBE, and LC-PBE has no
    /// libxc counterpart to compare a finite-ω energy against, so its ω → 0
    /// limit is the statement that pins it.
    #[test]
    fn ityh_range_limits() {
        let p = [0.5, 0.3, 0.1, 0.04, 0.08];
        for (parent, sr) in [
            (b88(), (b88_sr(1e-8), b88_sr(1e4), b88_sr(0.33))),
            (pbe_x(), (pbe_x_sr(1e-8), pbe_x_sr(1e4), pbe_x_sr(0.47))),
        ] {
            let name = parent.name();
            let full = eval_pt(parent.as_ref(), 0, p).e[0];
            // F(a) → 1 linearly in ω, so the deviation at ω = 1e-8 is O(1e-8).
            let sr_w0 = eval_pt(sr.0.as_ref(), 0, p).e[0];
            assert!(
                (sr_w0 - full).abs() < 1e-6 * full.abs(),
                "{name}: SR(ω→0) {sr_w0} vs {full}"
            );
            let sr_big = eval_pt(sr.1.as_ref(), 0, p).e[0];
            assert!(
                sr_big.abs() < 1e-5 * full.abs(),
                "{name}: SR not vanishing at large ω: {sr_big}"
            );
            // Monotone attenuation: 0 < e_SR/e_full < 1 at a physical ω.
            let sr_phys = eval_pt(sr.2.as_ref(), 0, p).e[0];
            assert!(
                sr_phys / full > 0.0 && sr_phys / full < 1.0,
                "{name}: attenuation ratio {} outside (0, 1)",
                sr_phys / full
            );
            // Exchange spin scaling survives the attenuation (the SR factor
            // is built per spin channel, so a shared-channel slip here would
            // not show up in the ω limits).
            let mixed = eval_pt(sr.2.as_ref(), 0, p).e[0];
            let a = eval_pt(sr.2.as_ref(), 0, [p[0], p[0], p[2], p[2], p[2]]).e[0];
            let b = eval_pt(sr.2.as_ref(), 0, [p[1], p[1], p[4], p[4], p[4]]).e[0];
            assert!(
                (mixed - 0.5 * (a + b)).abs() < 1e-13 * (1.0 + mixed.abs()),
                "{name}: SR exchange breaks spin scaling"
            );
        }
    }

    /// SR-B88 (ITYH) pinned against a libxc `gga_x_ityh` reference value
    /// (ω = 0.33, spin-polarized point; PySCF probe 2026-08-03).
    #[test]
    fn ityh_matches_libxc_reference_point() {
        let p = [0.3, 0.1, 0.0425, 0.031, 0.0233];
        let got = eval_pt(b88_sr(0.33).as_ref(), 0, p).e[0];
        let want = -0.165792242940;
        assert!(
            (got - want).abs() < 1e-9,
            "SR-B88 vs libxc ITYH: {got} vs {want}"
        );
    }

    /// Parameter gradients against FD (exercises the θ-seeding path for
    /// GGA, including PBE-C's composite parameter vector).
    #[test]
    fn gga_param_gradients_match_fd() {
        let p = [0.45, 0.3, 0.06, 0.02, 0.05];
        for f in [b88(), pbe_x(), lyp()] {
            let rho_a = [p[0]];
            let rho_b = [p[1]];
            let saa = [p[2]];
            let sab = [p[3]];
            let sbb = [p[4]];
            let vars = GridVars {
                rho: [&rho_a, &rho_b],
                sigma: Some([&saa, &sab, &sbb]),
                tau: None,
            };
            let mut grads = Vec::new();
            f.param_gradient(&vars, &mut grads).unwrap();
            let base = f.params();
            for k in 0..base.len() {
                let h = 1e-6 * base[k].abs().max(1e-4);
                let mut up = base.clone();
                up[k] += h;
                let mut dn = base.clone();
                dn[k] -= h;
                let eu = eval_pt(f.with_params(&up).unwrap().as_ref(), 0, p).e[0];
                let ed = eval_pt(f.with_params(&dn).unwrap().as_ref(), 0, p).e[0];
                let fd = (eu - ed) / (2.0 * h);
                assert!(
                    (grads[k][0] - fd).abs() < 1e-5 * fd.abs().max(1.0),
                    "{} ∂e/∂{}: {} vs {fd}",
                    f.name(),
                    f.param_names()[k],
                    grads[k][0]
                );
            }
        }
    }
}
