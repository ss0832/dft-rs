// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Meta-GGA driver: 7 grid variables (ρα, ρβ, σαα, σαβ, σββ, τα, τβ)
//! through the same dual-tower machinery as the LDA/GGA drivers.

use super::traits::{Family, GridVars, Kind, XcFunctional, XcOut};
use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};

use super::common::RHO_CUTOFF;

pub(crate) type MggaFn<S> = fn(S, S, S, S, S, S, S, &[S]) -> S;

pub(crate) struct MggaDriver {
    /// Declared scaling powers of the parameters (empty: the default).
    pub(crate) param_scaling: &'static [i32],
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    pub(crate) params: Vec<f64>,
    pub(crate) param_names: &'static [&'static str],
    pub(crate) f0: MggaFn<f64>,
    pub(crate) fj: MggaFn<crate::response::series::SeriesScalar>,
    pub(crate) fjt: MggaFn<DualN<crate::response::series::SeriesScalar, 7>>,
    pub(crate) fjv: MggaFn<super::taylor::DirectionalTaylorScalar>,
    pub(crate) fjb: MggaFn<super::bidirectional::BidirectionalTaylorScalar>,
    pub(crate) f1: MggaFn<DualN<f64, 7>>,
    pub(crate) f2: MggaFn<DualN<DualN<f64, 7>, 7>>,
    pub(crate) f3: MggaFn<DualN<DualN<DualN<f64, 7>, 7>, 7>>,
    pub(crate) fp: MggaFn<DualN<f64, 1>>,
    /// Enforce the Fermi-hole-curvature bound σσσ ≤ 8ρστσ (and
    /// |σαβ| ≤ (σαα+σββ)/2) on the INPUT VALUES before evaluation, as the
    /// libxc 7 meta-GGA driver does for ENFORCE_FHC functionals (SCAN, TPSS
    /// and B95 families). Only the values are clamped — derivative seeds stay on the
    /// original variables, matching the reference derivative convention.
    pub(crate) clamp_fhc: bool,
}

/// Clamped (σαα, σαβ, σββ, τα, τβ) input values under the FHC bound.
fn fhc_values(
    ra: f64,
    rb: f64,
    saa: f64,
    sab: f64,
    sbb: f64,
    ta: f64,
    tb: f64,
) -> (f64, f64, f64, f64, f64) {
    let ta = ta.max(1e-20);
    let tb = tb.max(1e-20);
    let saa = saa.min(8.0 * ra * ta);
    let sbb = sbb.min(8.0 * rb * tb);
    let s_ave = 0.5 * (saa + sbb);
    let sab = sab.clamp(-s_ave, s_ave);
    (saa, sab, sbb, ta, tb)
}

impl MggaDriver {
    /// Per-point (σαα, σαβ, σββ, τα, τβ) input values, FHC-clamped when
    /// this functional requests it.
    fn point_vals(
        &self,
        ra: f64,
        rb: f64,
        saa: f64,
        sab: f64,
        sbb: f64,
        ta: f64,
        tb: f64,
    ) -> (f64, f64, f64, f64, f64) {
        if self.clamp_fhc {
            fhc_values(ra, rb, saa, sab, sbb, ta, tb)
        } else {
            (saa, sab, sbb, ta, tb)
        }
    }
}

/// Clears the test-only point marker when an evaluation returns.
#[cfg(test)]
struct PointClear;

#[cfg(test)]
impl Drop for PointClear {
    fn drop(&mut self) {
        super::branch_freeze::set_point(None);
    }
}

impl XcFunctional for MggaDriver {
    fn name(&self) -> &'static str {
        self.name
    }
    fn kind(&self) -> Kind {
        self.kind
    }
    fn family(&self) -> Family {
        Family::MetaGga
    }
    fn max_analytic_order(&self) -> usize {
        3
    }
    fn taylor_energy(&self, input: &super::taylor::XcTaylorInput) -> Result<Vec<f64>> {
        input.validate()?;
        let mut point = input.clone();
        let v: [f64; 7] = std::array::from_fn(|i| point.variables[i][0]);
        let (saa, sab, sbb, ta, tb) = self.point_vals(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        for (i, value) in [saa, sab, sbb, ta, tb].into_iter().enumerate() {
            point.variables[i + 2][0] = value;
        }
        point.evaluate(&self.params, |v, p| {
            if v[0].value() + v[1].value() < RHO_CUTOFF {
                crate::response::series::SeriesScalar::from_f64(0.0)
            } else {
                (self.fj)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], p)
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
        input.validate()?;
        let mut point = input.clone();
        let v: [f64; 7] = std::array::from_fn(|i| point.variables[i][0]);
        let (saa, sab, sbb, ta, tb) = self.point_vals(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        for (i, value) in [saa, sab, sbb, ta, tb].into_iter().enumerate() {
            point.variables[i + 2][0] = value;
        }
        point.evaluate_potentials(&self.params, |v, p| {
            (self.fjt)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], p)
        })
    }
    fn taylor_directional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        direction: &[Vec<f64>; 7],
    ) -> Result<super::taylor::XcTaylorDirectionalOutput> {
        input.validate()?;
        let mut point = input.clone();
        let v: [f64; 7] = std::array::from_fn(|i| point.variables[i][0]);
        let (saa, sab, sbb, ta, tb) = self.point_vals(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        for (i, value) in [saa, sab, sbb, ta, tb].into_iter().enumerate() {
            point.variables[i + 2][0] = value;
        }
        point.evaluate_directional_potentials(direction, &self.params, |v, p| {
            (self.fjv)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], p)
        })
    }
    fn taylor_bidirectional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        first: &[Vec<f64>; 7],
        second: &[Vec<f64>; 7],
        mixed: &[Vec<f64>; 7],
    ) -> Result<super::bidirectional::XcTaylorBidirectionalOutput> {
        input.validate()?;
        let mut point = input.clone();
        let v: [f64; 7] = std::array::from_fn(|i| point.variables[i][0]);
        let (saa, sab, sbb, ta, tb) = self.point_vals(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        for (i, value) in [saa, sab, sbb, ta, tb].into_iter().enumerate() {
            point.variables[i + 2][0] = value;
        }
        point.evaluate_bidirectional_potentials(first, second, mixed, &self.params, |v, p| {
            (self.fjb)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], p)
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
            clamp_fhc: self.clamp_fhc,
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
            DftError::InvalidInput(format!("{} needs σ grid variables", self.name))
        })?;
        let tau = vars.tau.ok_or_else(|| {
            DftError::InvalidInput(format!("{} needs τ grid variables", self.name))
        })?;
        let n = vars.len();
        #[cfg(test)]
        let _clear = PointClear;
        match order {
            0 => {
                let th = self.params.clone();
                for i in 0..n {
                    #[cfg(test)]
                    super::branch_freeze::set_point(Some(i));
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let (saa, sab, sbb, ta, tb) = self.point_vals(
                        ra,
                        rb,
                        sigma[0][i].max(0.0),
                        sigma[1][i],
                        sigma[2][i].max(0.0),
                        tau[0][i].max(0.0),
                        tau[1][i].max(0.0),
                    );
                    out.e[i] += coef * (self.f0)(ra, rb, saa, sab, sbb, ta, tb, &th);
                }
            }
            1 => {
                type D = DualN<f64, 7>;
                let th: Vec<D> = self.params.iter().map(|&p| D::constant(p)).collect();
                for i in 0..n {
                    #[cfg(test)]
                    super::branch_freeze::set_point(Some(i));
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let (saa, sab, sbb, ta, tb) = self.point_vals(
                        ra,
                        rb,
                        sigma[0][i].max(0.0),
                        sigma[1][i],
                        sigma[2][i].max(0.0),
                        tau[0][i].max(0.0),
                        tau[1][i].max(0.0),
                    );
                    let v = (self.f1)(
                        D::variable(ra, 0),
                        D::variable(rb, 1),
                        D::variable(saa, 2),
                        D::variable(sab, 3),
                        D::variable(sbb, 4),
                        D::variable(ta, 5),
                        D::variable(tb, 6),
                        &th,
                    );
                    out.e[i] += coef * v.re;
                    for k in 0..2 {
                        out.vrho[k][i] += coef * v.im[k];
                    }
                    for k in 0..3 {
                        out.vsigma[k][i] += coef * v.im[2 + k];
                    }
                    for k in 0..2 {
                        out.vtau[k][i] += coef * v.im[5 + k];
                    }
                }
            }
            2 => {
                // Order 2: nested 7-variable duals; second derivatives are
                // v.im[p].im[q]. Variable order: ra rb saa sab sbb ta tb.
                type D1 = DualN<f64, 7>;
                type D2 = DualN<D1, 7>;
                let th: Vec<D2> = self
                    .params
                    .iter()
                    .map(|&p| D2::constant(D1::constant(p)))
                    .collect();
                let seed = |val: f64, k: usize| -> D2 {
                    let mut d = D2 {
                        re: D1::variable(val, k),
                        im: [D1::constant(0.0); 7],
                    };
                    d.im[k] = D1::constant(1.0);
                    d
                };
                for i in 0..n {
                    #[cfg(test)]
                    super::branch_freeze::set_point(Some(i));
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let (saa, sab, sbb, ta, tb) = self.point_vals(
                        ra,
                        rb,
                        sigma[0][i].max(0.0),
                        sigma[1][i],
                        sigma[2][i].max(0.0),
                        tau[0][i].max(0.0),
                        tau[1][i].max(0.0),
                    );
                    let v = (self.f2)(
                        seed(ra, 0),
                        seed(rb, 1),
                        seed(saa, 2),
                        seed(sab, 3),
                        seed(sbb, 4),
                        seed(ta, 5),
                        seed(tb, 6),
                        &th,
                    );
                    out.e[i] += coef * v.re.re;
                    for k in 0..2 {
                        out.vrho[k][i] += coef * v.re.im[k];
                    }
                    for k in 0..3 {
                        out.vsigma[k][i] += coef * v.re.im[2 + k];
                    }
                    for k in 0..2 {
                        out.vtau[k][i] += coef * v.re.im[5 + k];
                    }
                    let d2 = |p: usize, q: usize| v.im[p].im[q];
                    out.v2rho2[0][i] += coef * d2(0, 0);
                    out.v2rho2[1][i] += coef * d2(0, 1);
                    out.v2rho2[2][i] += coef * d2(1, 1);
                    // (rho_s, sigma_p): rows rho_a then rho_b.
                    for p in 0..3 {
                        out.v2rhosigma[p][i] += coef * d2(0, 2 + p);
                        out.v2rhosigma[3 + p][i] += coef * d2(1, 2 + p);
                    }
                    let ss = |p: usize, q: usize| -> usize {
                        let (a, b) = if p <= q { (p, q) } else { (q, p) };
                        match (a, b) {
                            (0, 0) => 0,
                            (0, 1) => 1,
                            (0, 2) => 2,
                            (1, 1) => 3,
                            (1, 2) => 4,
                            _ => 5,
                        }
                    };
                    for p in 0..3 {
                        for q in p..3 {
                            out.v2sigma2[ss(p, q)][i] += coef * d2(2 + p, 2 + q);
                        }
                    }
                    // tau blocks.
                    out.v2rhotau[0][i] += coef * d2(0, 5);
                    out.v2rhotau[1][i] += coef * d2(0, 6);
                    out.v2rhotau[2][i] += coef * d2(1, 5);
                    out.v2rhotau[3][i] += coef * d2(1, 6);
                    for p in 0..3 {
                        out.v2sigmatau[2 * p][i] += coef * d2(2 + p, 5);
                        out.v2sigmatau[2 * p + 1][i] += coef * d2(2 + p, 6);
                    }
                    out.v2tau2[0][i] += coef * d2(5, 5);
                    out.v2tau2[1][i] += coef * d2(5, 6);
                    out.v2tau2[2][i] += coef * d2(6, 6);
                }
            }
            3 => {
                type D1 = DualN<f64, 7>;
                type D2 = DualN<D1, 7>;
                type D3 = DualN<D2, 7>;
                let th: Vec<D3> = self.params.iter().map(|&p| D3::from_f64(p)).collect();
                let seed = |value: f64, k: usize| {
                    let mut d2 = D2 {
                        re: D1::variable(value, k),
                        im: [D1::constant(0.0); 7],
                    };
                    d2.im[k] = D1::constant(1.0);
                    let mut d3 = D3 {
                        re: d2,
                        im: [D2::from_f64(0.0); 7],
                    };
                    d3.im[k] = D2::from_f64(1.0);
                    d3
                };
                const RR: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];
                const SS: [(usize, usize); 6] = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)];
                const TT: [(usize, usize); 3] = [(5, 5), (5, 6), (6, 6)];
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
                const TTT: [(usize, usize, usize); 4] =
                    [(5, 5, 5), (5, 5, 6), (5, 6, 6), (6, 6, 6)];
                for i in 0..n {
                    #[cfg(test)]
                    super::branch_freeze::set_point(Some(i));
                    let (ra, rb) = (vars.rho[0][i].max(0.0), vars.rho[1][i].max(0.0));
                    if ra + rb < RHO_CUTOFF {
                        continue;
                    }
                    let (saa, sab, sbb, ta, tb) = self.point_vals(
                        ra,
                        rb,
                        sigma[0][i].max(0.0),
                        sigma[1][i],
                        sigma[2][i].max(0.0),
                        tau[0][i].max(0.0),
                        tau[1][i].max(0.0),
                    );
                    let v = (self.f3)(
                        seed(ra, 0),
                        seed(rb, 1),
                        seed(saa, 2),
                        seed(sab, 3),
                        seed(sbb, 4),
                        seed(ta, 5),
                        seed(tb, 6),
                        &th,
                    );
                    out.e[i] += coef * v.re.re.re;
                    for k in 0..2 {
                        out.vrho[k][i] += coef * v.re.re.im[k];
                    }
                    for k in 0..3 {
                        out.vsigma[k][i] += coef * v.re.re.im[2 + k];
                    }
                    for k in 0..2 {
                        out.vtau[k][i] += coef * v.re.re.im[5 + k];
                    }
                    let d2 = |p: usize, q: usize| v.re.im[p].im[q];
                    for (block, &(p, q)) in RR.iter().enumerate() {
                        out.v2rho2[block][i] += coef * d2(p, q);
                    }
                    for rho in 0..2 {
                        for s in 0..3 {
                            out.v2rhosigma[3 * rho + s][i] += coef * d2(rho, 2 + s);
                        }
                    }
                    for (block, &(p, q)) in SS.iter().enumerate() {
                        out.v2sigma2[block][i] += coef * d2(p, q);
                    }
                    for rho in 0..2 {
                        for t in 0..2 {
                            out.v2rhotau[2 * rho + t][i] += coef * d2(rho, 5 + t);
                        }
                    }
                    for s in 0..3 {
                        for t in 0..2 {
                            out.v2sigmatau[2 * s + t][i] += coef * d2(2 + s, 5 + t);
                        }
                    }
                    for (block, &(p, q)) in TT.iter().enumerate() {
                        out.v2tau2[block][i] += coef * d2(p, q);
                    }
                    let d3 = |p: usize, q: usize, r: usize| v.im[p].im[q].im[r];
                    for (block, &(p, q, r)) in RRR.iter().enumerate() {
                        out.v3rho3[block][i] += coef * d3(p, q, r);
                    }
                    for (rp, &(p, q)) in RR.iter().enumerate() {
                        for s in 0..3 {
                            out.v3rho2sigma[3 * rp + s][i] += coef * d3(p, q, 2 + s);
                        }
                        for t in 0..2 {
                            out.v3rho2tau[2 * rp + t][i] += coef * d3(p, q, 5 + t);
                        }
                    }
                    for rho in 0..2 {
                        for (sp, &(p, q)) in SS.iter().enumerate() {
                            out.v3rhosigma2[6 * rho + sp][i] += coef * d3(rho, p, q);
                        }
                        for s in 0..3 {
                            for t in 0..2 {
                                out.v3rhosigmatau[6 * rho + 2 * s + t][i] +=
                                    coef * d3(rho, 2 + s, 5 + t);
                            }
                        }
                        for (tp, &(p, q)) in TT.iter().enumerate() {
                            out.v3rhotau2[3 * rho + tp][i] += coef * d3(rho, p, q);
                        }
                    }
                    for (block, &(p, q, r)) in SSS.iter().enumerate() {
                        out.v3sigma3[block][i] += coef * d3(p, q, r);
                    }
                    for (sp, &(p, q)) in SS.iter().enumerate() {
                        for t in 0..2 {
                            out.v3sigma2tau[2 * sp + t][i] += coef * d3(p, q, 5 + t);
                        }
                    }
                    for s in 0..3 {
                        for (tp, &(p, q)) in TT.iter().enumerate() {
                            out.v3sigmatau2[3 * s + tp][i] += coef * d3(2 + s, p, q);
                        }
                    }
                    for (block, &(p, q, r)) in TTT.iter().enumerate() {
                        out.v3tau3[block][i] += coef * d3(p, q, r);
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
        let tau = vars.tau.ok_or_else(|| {
            DftError::InvalidInput(format!("{} needs τ grid variables", self.name))
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
                let (saa, sab, sbb, ta, tb) = self.point_vals(
                    ra,
                    rb,
                    sigma[0][i].max(0.0),
                    sigma[1][i],
                    sigma[2][i].max(0.0),
                    tau[0][i].max(0.0),
                    tau[1][i].max(0.0),
                );
                let v = (self.fp)(
                    P::constant(ra),
                    P::constant(rb),
                    P::constant(saa),
                    P::constant(sab),
                    P::constant(sbb),
                    P::constant(ta),
                    P::constant(tb),
                    &th,
                );
                out[k][i] = v.im[0];
            }
        }
        Ok(())
    }
}
