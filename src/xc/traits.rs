// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The exchange–correlation functional interface: spin-resolved grid
//! variables in, energy density and derivatives (to a requested order) out.
//!
//! Conventions:
//! - `e` is the energy **density per volume**: `Exc = Σ_i w_i e_i`. The
//!   first density derivatives `vrho = ∂e/∂ρ_σ` are therefore the KS
//!   potential values directly.
//! - Derivatives are produced by instantiating one `Scalar`-generic energy
//!   expression with nested duals — any implemented order is exact, and the
//!   same mechanism differentiates with respect to functional parameters.

use crate::error::{DftError, Result};
use std::cell::Cell;
use std::sync::Mutex;

static ORDER3_WORKER_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static ON_ORDER3_WORKER: Cell<bool> = const { Cell::new(false) };
}

struct Order3WorkerReset<'a> {
    flag: &'a Cell<bool>,
    previous: bool,
}

impl Drop for Order3WorkerReset<'_> {
    fn drop(&mut self) {
        self.flag.set(self.previous);
    }
}

/// Unforgeable safe-code token for the raw driver dispatch.
///
/// This type is public only because it appears in the public extension trait;
/// its private field prevents callers from bypassing [`XcFunctional::eval_add`].
#[doc(hidden)]
pub struct XcEvalDispatchToken {
    _private: (),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Lda,
    Gga,
    MetaGga,
}

impl Family {
    /// Rung on Jacob's ladder: LDA < GGA < meta-GGA.
    pub fn rung(self) -> u8 {
        match self {
            Self::Lda => 0,
            Self::Gga => 1,
            Self::MetaGga => 2,
        }
    }

    /// The higher of two rungs.
    ///
    /// A composite's family must be the MAXIMUM over its parts, never the
    /// last one seen or a pairwise special case: the family is what
    /// [`XcOut::prepare`] uses to decide which output blocks exist, so a
    /// composite that reports a lower rung than one of its parts hands that
    /// part an unallocated block to write into. Getting this wrong is not a
    /// numerical inaccuracy — it is an index-out-of-bounds panic on the
    /// first grid point.
    pub fn max_rung(self, other: Self) -> Self {
        if other.rung() > self.rung() {
            other
        } else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Exchange,
    Correlation,
    ExchangeCorrelation,
}

/// Spin-resolved density variables on a batch of grid points (SoA slices).
pub struct GridVars<'a> {
    pub rho: [&'a [f64]; 2],
    /// σ contractions ∇ρ·∇ρ as (αα, αβ, ββ); `None` for LDA-only callers.
    pub sigma: Option<[&'a [f64]; 3]>,
    /// Kinetic-energy densities τ_σ = ½Σ|∇ψ|² (meta-GGA).
    pub tau: Option<[&'a [f64]; 2]>,
}

impl GridVars<'_> {
    pub fn len(&self) -> usize {
        self.rho[0].len()
    }
    pub fn is_empty(&self) -> bool {
        self.rho[0].is_empty()
    }
}

/// Output blocks; filled up to the requested order (unused blocks are left
/// empty). Index layout mirrors libxc: pair derivatives ordered
/// (αα, αβ, ββ).
#[derive(Clone, Debug, Default)]
pub struct XcOut {
    /// Energy density per volume.
    pub e: Vec<f64>,
    /// ∂e/∂ρ_σ (order ≥ 1).
    pub vrho: [Vec<f64>; 2],
    /// ∂e/∂σ_p, p ∈ {αα, αβ, ββ} (order ≥ 1, GGA).
    pub vsigma: [Vec<f64>; 3],
    /// ∂e/∂τ_σ (order ≥ 1, meta-GGA).
    pub vtau: [Vec<f64>; 2],
    /// ∂²e/∂ρ∂ρ as (αα, αβ, ββ) (order ≥ 2).
    pub v2rho2: [Vec<f64>; 3],
    /// ∂²e/∂ρ_σ∂σ_p, 6 combinations (order ≥ 2, GGA).
    pub v2rhosigma: [Vec<f64>; 6],
    /// ∂²e/∂σ∂σ, 6 combinations (order ≥ 2, GGA).
    pub v2sigma2: [Vec<f64>; 6],
    /// ∂²e/∂ρ_σ∂τ_τ as (ρα τα, ρα τβ, ρβ τα, ρβ τβ) (order ≥ 2, meta).
    pub v2rhotau: [Vec<f64>; 4],
    /// ∂²e/∂σ_p∂τ_σ, 6 combinations (σ_p × τ_σ; order ≥ 2, meta).
    pub v2sigmatau: [Vec<f64>; 6],
    /// ∂²e/∂τ∂τ as (αα, αβ, ββ) (order ≥ 2, meta).
    pub v2tau2: [Vec<f64>; 3],
    /// ∂³e/∂ρ³ as (ααα, ααβ, αββ, βββ) (order ≥ 3).
    pub v3rho3: [Vec<f64>; 4],
    /// ∂³e/∂ρ²∂σ, indexed by symmetric ρ pair then σ
    /// (`(αα,αβ,ββ) × (αα,αβ,ββ)`; order ≥ 3, GGA).
    pub v3rho2sigma: [Vec<f64>; 9],
    /// ∂³e/∂ρ∂σ², indexed by ρ then symmetric σ pair
    /// (`(α,β) × (αα·αα,αα·αβ,…,ββ·ββ)`; order ≥ 3, GGA).
    pub v3rhosigma2: [Vec<f64>; 12],
    /// ∂³e/∂σ³ in lexicographic symmetric-triple order (10 blocks;
    /// order ≥ 3, GGA).
    pub v3sigma3: [Vec<f64>; 10],
    /// ∂³e/∂ρ²∂τ, indexed by symmetric ρ pair then τ
    /// (6 blocks; order ≥ 3, meta-GGA).
    pub v3rho2tau: [Vec<f64>; 6],
    /// ∂³e/∂ρ∂σ∂τ, indexed by ρ, then σ, then τ
    /// (12 blocks; order ≥ 3, meta-GGA).
    pub v3rhosigmatau: [Vec<f64>; 12],
    /// ∂³e/∂ρ∂τ², indexed by ρ then symmetric τ pair
    /// (6 blocks; order ≥ 3, meta-GGA).
    pub v3rhotau2: [Vec<f64>; 6],
    /// ∂³e/∂σ²∂τ, indexed by symmetric σ pair then τ
    /// (12 blocks; order ≥ 3, meta-GGA).
    pub v3sigma2tau: [Vec<f64>; 12],
    /// ∂³e/∂σ∂τ², indexed by σ then symmetric τ pair
    /// (9 blocks; order ≥ 3, meta-GGA).
    pub v3sigmatau2: [Vec<f64>; 9],
    /// ∂³e/∂τ³ as (ααα, ααβ, αββ, βββ) (order ≥ 3,
    /// meta-GGA).
    pub v3tau3: [Vec<f64>; 4],
}

impl XcOut {
    /// Dense first and second partial derivatives in the common order
    /// `(rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b)`.
    /// Requires output evaluated through second order. Inactive family
    /// channels are zero; missing active derivative blocks are an error.
    pub fn potential_and_kernel(
        &self,
        point: usize,
        family: Family,
    ) -> Result<([f64; 7], [[f64; 7]; 7])> {
        let get = |v: &Vec<f64>| -> Result<f64> {
            v.get(point)
                .copied()
                .filter(|x| x.is_finite())
                .ok_or_else(|| {
                    DftError::InvalidInput("missing or non-finite second-order XC output".into())
                })
        };
        let mut potential = [0.0; 7];
        let mut kernel = [[0.0; 7]; 7];
        for spin in 0..2 {
            potential[spin] = get(&self.vrho[spin])?;
        }
        kernel[0][0] = get(&self.v2rho2[0])?;
        kernel[0][1] = get(&self.v2rho2[1])?;
        kernel[1][1] = get(&self.v2rho2[2])?;
        if family != Family::Lda {
            let mut pair = 0;
            for p in 0..3 {
                potential[2 + p] = get(&self.vsigma[p])?;
                kernel[0][2 + p] = get(&self.v2rhosigma[p])?;
                kernel[1][2 + p] = get(&self.v2rhosigma[3 + p])?;
                for q in p..3 {
                    kernel[2 + p][2 + q] = get(&self.v2sigma2[pair])?;
                    pair += 1;
                }
            }
        }
        if family == Family::MetaGga {
            for spin in 0..2 {
                potential[5 + spin] = get(&self.vtau[spin])?;
                for rho in 0..2 {
                    kernel[rho][5 + spin] = get(&self.v2rhotau[2 * rho + spin])?;
                }
                for p in 0..3 {
                    kernel[2 + p][5 + spin] = get(&self.v2sigmatau[2 * p + spin])?;
                }
            }
            kernel[5][5] = get(&self.v2tau2[0])?;
            kernel[5][6] = get(&self.v2tau2[1])?;
            kernel[6][6] = get(&self.v2tau2[2])?;
        }
        for i in 0..7 {
            for j in 0..i {
                kernel[i][j] = kernel[j][i];
            }
        }
        Ok((potential, kernel))
    }

    /// Resize every block used at `order` for `n` points and zero them.
    pub fn prepare(&mut self, n: usize, order: usize, family: Family) {
        let gga = family != Family::Lda;
        self.e.clear();
        self.e.resize(n, 0.0);
        let fill = |v: &mut Vec<f64>, active: bool| {
            v.clear();
            if active {
                v.resize(n, 0.0);
            }
        };
        for b in self.vrho.iter_mut() {
            fill(b, order >= 1);
        }
        for b in self.vsigma.iter_mut() {
            fill(b, order >= 1 && gga);
        }
        for b in self.vtau.iter_mut() {
            fill(b, order >= 1 && family == Family::MetaGga);
        }
        for b in self.v2rho2.iter_mut() {
            fill(b, order >= 2);
        }
        for b in self.v2rhosigma.iter_mut() {
            fill(b, order >= 2 && gga);
        }
        for b in self.v2sigma2.iter_mut() {
            fill(b, order >= 2 && gga);
        }
        let meta = family == Family::MetaGga;
        for b in self.v2rhotau.iter_mut() {
            fill(b, order >= 2 && meta);
        }
        for b in self.v2sigmatau.iter_mut() {
            fill(b, order >= 2 && meta);
        }
        for b in self.v2tau2.iter_mut() {
            fill(b, order >= 2 && meta);
        }
        for b in self.v3rho3.iter_mut() {
            fill(b, order >= 3);
        }
        for b in self.v3rho2sigma.iter_mut() {
            fill(b, order >= 3 && gga);
        }
        for b in self.v3rhosigma2.iter_mut() {
            fill(b, order >= 3 && gga);
        }
        for b in self.v3sigma3.iter_mut() {
            fill(b, order >= 3 && gga);
        }
        for b in self.v3rho2tau.iter_mut() {
            fill(b, order >= 3 && meta);
        }
        for b in self.v3rhosigmatau.iter_mut() {
            fill(b, order >= 3 && meta);
        }
        for b in self.v3rhotau2.iter_mut() {
            fill(b, order >= 3 && meta);
        }
        for b in self.v3sigma2tau.iter_mut() {
            fill(b, order >= 3 && meta);
        }
        for b in self.v3sigmatau2.iter_mut() {
            fill(b, order >= 3 && meta);
        }
        for b in self.v3tau3.iter_mut() {
            fill(b, order >= 3 && meta);
        }
    }
}

/// A named functional (exchange, correlation, or combined XC part).
pub(crate) fn eval_builtin_with_stack<F>(family: Family, order: usize, job: F) -> Result<()>
where
    F: FnOnce(XcEvalDispatchToken) -> Result<()> + Send,
{
    // Resource-scoped Rayon workers already have the required 8 MiB stack.
    // A separate native helper would lose the shared budget and CPU ceiling.
    let direct = order != 3
        || family == Family::Lda
        || ON_ORDER3_WORKER.with(Cell::get)
        || crate::resources::current_usage().is_some();
    if direct {
        return job(XcEvalDispatchToken { _private: () });
    }
    let _worker_slot = ORDER3_WORKER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("dft-rs-xc-third".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn_scoped(scope, || {
                ON_ORDER3_WORKER.with(|flag| {
                    let reset = Order3WorkerReset {
                        flag,
                        previous: flag.replace(true),
                    };
                    let result = job(XcEvalDispatchToken { _private: () });
                    drop(reset);
                    result
                })
            })
            .map_err(DftError::Io)?;
        match worker.join() {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

/// A named functional (exchange, correlation, or combined XC part).
/// Scaling powers of parameters without a declaration: the range-separation
/// parameter `omega` (an inverse length) carries 1, every other parameter is
/// dimensionless.
pub fn default_param_scaling(names: &[&str]) -> Vec<i32> {
    names.iter().map(|n| i32::from(*n == "omega")).collect()
}

pub trait XcFunctional: Send + Sync {
    fn name(&self) -> &'static str;
    fn kind(&self) -> Kind;
    fn family(&self) -> Family;
    /// Highest analytic derivative order implemented.
    fn max_analytic_order(&self) -> usize;
    /// Taylor-normalized semilocal energy-density expansion. Unlike the
    /// dense XcOut blocks, this interface need not have a fixed order cap.
    /// External functionals may implement this in addition to eval_add.
    fn taylor_energy(&self, input: &super::taylor::XcTaylorInput) -> Result<Vec<f64>> {
        Err(DftError::DerivOrderUnsupported {
            functional: self.name().into(),
            requested: input.order(),
            max_analytic: self.max_analytic_order(),
        })
    }
    /// Analytic Taylor series of the energy and its density-variable
    /// gradient; required for nonlinear self-consistent response sources.
    fn taylor_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
    ) -> Result<super::taylor::XcTaylorOutput> {
        Err(DftError::DerivOrderUnsupported {
            functional: self.name().into(),
            requested: input.order() + 1,
            max_analytic: self.max_analytic_order(),
        })
    }
    /// Potential Taylor path and its independent mixed directional derivative.
    fn taylor_directional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        _direction: &[Vec<f64>; 7],
    ) -> Result<super::taylor::XcTaylorDirectionalOutput> {
        Err(DftError::DerivOrderUnsupported {
            functional: self.name().into(),
            requested: input.order() + 2,
            max_analytic: self.max_analytic_order(),
        })
    }
    /// Two independent mixed tangents over an analytic nuclear path.
    /// External providers must implement this explicitly; dense derivatives
    /// or a single-direction API do not imply the required higher jets.
    fn taylor_bidirectional_potentials(
        &self,
        input: &super::taylor::XcTaylorInput,
        _first: &[Vec<f64>; 7],
        _second: &[Vec<f64>; 7],
        _mixed: &[Vec<f64>; 7],
    ) -> Result<super::bidirectional::XcTaylorBidirectionalOutput> {
        Err(DftError::DerivOrderUnsupported {
            functional: self.name().into(),
            requested: input.order() + 3,
            max_analytic: self.max_analytic_order(),
        })
    }
    /// Current parameter values (fit coefficients, scalings, ω, …).
    fn params(&self) -> Vec<f64>;
    fn param_names(&self) -> &'static [&'static str];
    /// A copy of this functional with replaced parameters (ω-tuning,
    /// functional fitting).
    fn with_params(&self, params: &[f64]) -> Result<Box<dyn XcFunctional>>;
    /// Power of the uniform-coordinate-scaling factor carried by each
    /// parameter: under `r -> r / lambda` (`rho -> lambda^3 rho`,
    /// `sigma -> lambda^8 sigma`, `tau -> lambda^5 tau`) an exchange energy
    /// density obeys `e(lambda^3 rho, ...; lambda^k theta) = lambda^4 e(rho, ...; theta)`.
    /// Default: [`default_param_scaling`].
    fn param_scaling_powers(&self) -> Vec<i32> {
        default_param_scaling(self.param_names())
    }
    /// Evaluate energy density and derivatives up to `order`, **adding**
    /// `coef ×` the results into `out` (parts of a composite accumulate).
    fn eval_add(&self, vars: &GridVars, order: usize, coef: f64, out: &mut XcOut) -> Result<()>;
    /// Driver implementation behind the stack-safe public dispatch.
    #[doc(hidden)]
    fn eval_add_impl(
        &self,
        token: XcEvalDispatchToken,
        vars: &GridVars,
        order: usize,
        coef: f64,
        out: &mut XcOut,
    ) -> Result<()> {
        let _ = (token, vars, order, coef, out);
        Err(DftError::Unsupported {
            feature: "raw XC driver dispatch".to_string(),
            missing: "an in-crate driver implementation; external functionals implement eval_add directly"
                .to_string(),
        })
    }
    /// Parameter gradient: `out[k][i] = ∂e_i/∂θ_k` at fixed densities.
    fn param_gradient(&self, vars: &GridVars, out: &mut Vec<Vec<f64>>) -> Result<()>;
}

#[cfg(test)]
mod resource_dispatch_tests {
    use super::*;
    use crate::resources::{current_usage, reserve, ResourceLimits};
    use rayon::prelude::*;

    #[test]
    fn third_order_dispatch_keeps_the_scoped_pool_and_budget() {
        for threads in [1, 3] {
            ResourceLimits::new(Some(threads), Some(1))
                .unwrap()
                .run(|| {
                    eval_builtin_with_stack(Family::MetaGga, 3, |_| {
                        assert_eq!(rayon::current_num_threads(), threads);
                        assert!(rayon::current_thread_index().is_some());
                        assert_eq!(current_usage().unwrap().cpu_threads, threads);
                        let _lease = reserve(12345, "third-order XC dispatch fixture")?;
                        (0..8).into_par_iter().for_each(|_| {
                            assert_eq!(current_usage().unwrap().reserved_bytes, 12345);
                            assert_eq!(rayon::current_num_threads(), threads);
                        });
                        Ok(())
                    })
                })
                .unwrap();
        }
    }
}
