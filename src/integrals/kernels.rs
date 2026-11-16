// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Interchangeable interelectronic operator kernels.
//!
//! Every Coulomb-type integral enters the McMurchie–Davidson scheme only
//! through the Boys values at the R-tensor base case, so swapping the
//! operator `1/r₁₂` for its range-separated partners is a pure change of
//! those values:
//!
//! - `Coulomb`: F_n(T) unchanged.
//! - `Erf{ω}` (long range, `erf(ωr)/r`): with composite exponent ρ, define
//!   `f = ω²/(ρ + ω²)`; then F_n → f^{n+1/2} · F_n(f·T).
//! - `Erfc{ω}` (short range, `erfc(ωr)/r`): linearity, Coulomb − Erf.
//!
//! This is the single switch point that makes range-separated hybrids
//! (LC-ωPBE, CAM-B3LYP, ω-tuning) an entry in the functional table rather
//! than an integral-engine rewrite.

use super::boys::boys_into;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoulombKernel {
    /// Plain 1/r₁₂.
    Coulomb,
    /// Long-range erf(ω r₁₂)/r₁₂.
    Erf { omega: f64 },
    /// Short-range erfc(ω r₁₂)/r₁₂.
    Erfc { omega: f64 },
}

impl CoulombKernel {
    /// Fill `out[0..=nmax]` with the kernel-adjusted Boys values for
    /// composite exponent `rho` and argument `t = rho·|PQ|²`.
    pub fn boys_into(self, nmax: usize, rho: f64, t: f64, out: &mut [f64]) {
        match self {
            Self::Coulomb => boys_into(nmax, t, out),
            Self::Erf { omega } => {
                let f = omega * omega / (rho + omega * omega);
                boys_into(nmax, f * t, out);
                let mut scale = f.sqrt();
                for o in out.iter_mut().take(nmax + 1) {
                    *o *= scale;
                    scale *= f;
                }
            }
            Self::Erfc { omega } => {
                let mut lr = vec![0.0; nmax + 1];
                Self::Erf { omega }.boys_into(nmax, rho, t, &mut lr);
                boys_into(nmax, t, out);
                for (o, l) in out.iter_mut().zip(&lr).take(nmax + 1) {
                    *o -= l;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// erf + erfc must reassemble the plain Coulomb kernel exactly.
    #[test]
    fn erf_plus_erfc_is_coulomb() {
        let (rho, t) = (0.73, 4.2);
        let omega = 0.4;
        let mut full = vec![0.0; 9];
        let mut lr = vec![0.0; 9];
        let mut sr = vec![0.0; 9];
        CoulombKernel::Coulomb.boys_into(8, rho, t, &mut full);
        CoulombKernel::Erf { omega }.boys_into(8, rho, t, &mut lr);
        CoulombKernel::Erfc { omega }.boys_into(8, rho, t, &mut sr);
        for n in 0..=8 {
            assert!((lr[n] + sr[n] - full[n]).abs() < 1e-15);
        }
    }

    /// ω → ∞ turns erf(ωr)/r into 1/r.
    #[test]
    fn large_omega_erf_approaches_coulomb() {
        let (rho, t) = (1.1, 2.0);
        let mut full = vec![0.0; 5];
        let mut lr = vec![0.0; 5];
        CoulombKernel::Coulomb.boys_into(4, rho, t, &mut full);
        CoulombKernel::Erf { omega: 1e6 }.boys_into(4, rho, t, &mut lr);
        for n in 0..=4 {
            assert!((lr[n] - full[n]).abs() < 1e-9);
        }
    }

    /// The attenuated F_0 has a closed form for s functions:
    /// with f = ω²/(ρ+ω²), F₀^erf(T) = √f · F₀(fT) — cross-check against a
    /// direct quadrature of ∫₀¹ √f… by construction, and additionally pin
    /// the T = 0 value F₀^erf(0) = √f.
    #[test]
    fn erf_zero_argument_closed_form() {
        let rho = 0.9;
        let omega = 0.33;
        let f = omega * omega / (rho + omega * omega);
        let mut lr = vec![0.0; 1];
        CoulombKernel::Erf { omega }.boys_into(0, rho, 0.0, &mut lr);
        assert!((lr[0] - f.sqrt()).abs() < 1e-15);
    }
}
