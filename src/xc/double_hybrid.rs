// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Explicit orbital-dependent double-hybrid composition. The parent functional
//! defines the generalized Kohn-Sham orbitals; PT2 is a separate energy term.
//! Keeping that term typed prevents treating a parent's response as the full
//! correlated response. No response code dispatches on a functional name.

use super::{gga, Family, HybridParams, ResolvedFunctional};
use crate::{DftError, Result};
use std::sync::Arc;

/// Exact exchange fraction `3^(-1/3)` of the quadratic-integrand
/// double-hybrid model (QIDH).
const QIDH_EXCHANGE: f64 = 0.693_361_274_350_634_8;

#[derive(Clone, Copy, Debug)]
pub struct Pt2Coefficients {
    pub same_spin: f64,
    pub opposite_spin: f64,
}
impl Pt2Coefficients {
    pub fn validate(self) -> Result<()> {
        if !self.same_spin.is_finite() || !self.opposite_spin.is_finite() {
            return Err(DftError::InvalidInput(
                "PT2 spin coefficients must be finite".into(),
            ));
        }
        Ok(())
    }
}

/// Canonical names of the named double hybrids, one per arm of
/// [`DoubleHybridFunctional::named`] (aliases are accepted but not listed).
pub const DOUBLE_HYBRID_NAMES: &[&str] = &[
    "b2plyp",
    "wb2plyp",
    "wb2gp-plyp",
    "rsx-qidh",
    "rsx-0dh",
    "dsd-pbep86",
    "dsd-blyp",
    "dsd-pbeb95",
    "wb97m(2)",
    "wb97x-2-lp",
    "wb97x-2-tqz",
    "pwpb95",
    "b2gp-plyp",
    "b2k-plyp",
    "b2t-plyp",
    "mpw2plyp",
    "mpw2k-plyp",
    "pbe0-dh",
    "pbe-qidh",
    "pbe0-2",
    "tpss-qidh",
    "scan0-dh",
    "scan-qidh",
    "r2scan0-dh",
    "r2scan-qidh",
    "revdsd-pbep86",
    "xyg3",
    "xdh-pbe0",
];

pub struct DoubleHybridFunctional {
    pub parent: ResolvedFunctional,
    /// A distinct xDH target, evaluated on the parent's stationary orbitals.
    /// None means that the target and orbital-generating energy coincide.
    pub target: Option<ResolvedFunctional>,
    pub reference_vv10: Option<crate::nlc::Vv10Params>,
    /// Scale and parameters of the target's nonlocal correlation energy.
    pub target_vv10: Option<(f64, crate::nlc::Vv10Params)>,
    pub correlation: Pt2Coefficients,
    /// Named recipes may require a separate pairwise dispersion energy.
    pub dispersion: Option<crate::dispersion::D3BjParams>,
}
impl DoubleHybridFunctional {
    /// Resolve a named orbital-dependent functional separately from the
    /// semilocal/hybrid parent registry, so its PT2 term cannot be lost.
    pub fn named(name: &str) -> Option<Self> {
        let ascii_name = name.replace(['ω', 'Ω'], "w");
        let normalized: String = ascii_name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        match normalized.as_str() {
            "b2plyp" => Some(Self::b2plyp()),
            "wb2plyp" | "omegab2plyp" => Some(Self::wb2plyp()),
            "wb2gpplyp" | "omegab2gpplyp" => Some(Self::wb2gp_plyp()),
            "rsxqidh" => Some(Self::rsx_qidh()),
            "rsx0dh" => Some(Self::rsx_0dh()),
            "dsdpbep86" | "dsdpbep86d3bj" | "dsdpbep862013" => Some(Self::dsd_pbep86()),
            "dsdblyp" | "dsdblypd3bj" | "dsdblyp2013" => Some(Self::dsd_blyp()),
            "dsdpbeb95" | "dsdpbeb95d3bj" | "dsdpbeb952013" => Some(Self::dsd_pbeb95()),
            "wb97m2" | "omegab97m2" => Some(Self::wb97m2()),
            "wb97x2lp" | "omegab97x2lp" => Some(Self::wb97x2_lp()),
            "wb97x2tqz" | "omegab97x2tqz" => Some(Self::wb97x2_tqz()),
            "pwpb95" | "pwpb95d3bj" => Some(Self::pwpb95()),
            "b2gpplyp" => Some(Self::b2gp_plyp()),
            "b2kplyp" => Some(Self::b2k_plyp()),
            "b2tplyp" => Some(Self::b2t_plyp()),
            "mpw2plyp" => Some(Self::mpw2_plyp()),
            "mpw2kplyp" => Some(Self::mpw2k_plyp()),
            "pbe0dh" => Some(Self::pbe0_dh()),
            "pbeqidh" => Some(Self::pbe_qidh()),
            "pbe02" => Some(Self::pbe0_2()),
            "tpssqidh" => Some(Self::tpss_qidh()),
            "scan0dh" => Some(Self::scan0_dh()),
            "scanqidh" => Some(Self::scan_qidh()),
            "r2scan0dh" => Some(Self::r2scan0_dh()),
            "r2scanqidh" => Some(Self::r2scan_qidh()),
            "revdsdpbep86" | "revdsdpbep86d3bj" => Some(Self::revdsd_pbep86()),
            "xyg3" => Some(Self::xyg3()),
            "xdhpbe0" => Some(Self::xdh_pbe0()),
            _ => None,
        }
    }
    /// Arbitrary semilocal, global-hybrid or range-separated parent and
    /// independent same/opposite-spin full-Coulomb PT2 coefficients.
    pub fn new(parent: ResolvedFunctional, correlation: Pt2Coefficients) -> Result<Self> {
        correlation.validate()?;
        Ok(Self {
            parent,
            target: None,
            reference_vv10: None,
            target_vv10: None,
            correlation,
            dispersion: None,
        })
    }
    /// B2PLYP: 0.53 E_x^HF + 0.47 E_x^B88 + 0.73 E_c^LYP
    /// plus 0.27 times both spin components of PT2 on those orbitals.
    pub fn b2plyp() -> Self {
        Self {
            target: None,
            reference_vv10: None,
            target_vv10: None,
            parent: ResolvedFunctional::custom(
                "B2PLYP generalized Kohn-Sham parent",
                vec![(0.47, Arc::from(gga::b88())), (0.73, Arc::from(gga::lyp()))],
                HybridParams {
                    cx: 0.53,
                    clr: 0.0,
                    omega: 0.0,
                },
                Family::Gga,
            ),
            correlation: Pt2Coefficients {
                same_spin: 0.27,
                opposite_spin: 0.27,
            },
            dispersion: None,
        }
    }
    /// Range-separated B2PLYP: 53% short-range and full long-range exact
    /// exchange, omega=0.30, unchanged 73% LYP and 27% full-Coulomb PT2.
    pub fn wb2plyp() -> Self {
        Self::range_b2_plyp_type(
            "omegaB2PLYP generalized Kohn-Sham parent",
            0.53,
            0.30,
            0.73,
            0.27,
        )
    }

    /// Range-separated B2GP-PLYP: 65% short-range and full long-range
    /// exact exchange, omega=0.27, 64% LYP and 36% full-Coulomb PT2.
    pub fn wb2gp_plyp() -> Self {
        Self::range_b2_plyp_type(
            "omegaB2GP-PLYP generalized Kohn-Sham parent",
            0.65,
            0.27,
            0.64,
            0.36,
        )
    }

    fn range_b2_plyp_type(description: &str, a_x: f64, omega: f64, c: f64, pt2: f64) -> Self {
        Self {
            parent: ResolvedFunctional::custom(
                description,
                vec![
                    (1.0 - a_x, Arc::from(gga::b88_sr(omega))),
                    (c, Arc::from(gga::lyp())),
                ],
                HybridParams {
                    cx: a_x,
                    clr: 1.0 - a_x,
                    omega,
                },
                Family::Gga,
            ),
            target: None,
            reference_vv10: None,
            target_vv10: None,
            correlation: Pt2Coefficients {
                same_spin: pt2,
                opposite_spin: pt2,
            },
            dispersion: None,
        }
    }

    /// RSX-QIDH: exact QIDH exchange fraction, full long-range exchange,
    /// omega=0.27 and full-Coulomb PT2/3 (DOI10.1021/acs.jctc.8b00261).
    pub fn rsx_qidh() -> Self {
        Self::range_pbe_adiabatic(
            "RSX-QIDH generalized Kohn-Sham parent",
            QIDH_EXCHANGE,
            0.27,
            1.0 / 3.0,
        )
    }

    /// RSX-0DH: short-range HF=0.5, full long-range HF, omega=0.33 and
    /// full-Coulomb PT2/8 (DOI10.1063/1.5097164); no implicit dispersion.
    pub fn rsx_0dh() -> Self {
        Self::range_pbe_adiabatic("RSX-0DH generalized Kohn-Sham parent", 0.5, 0.33, 0.125)
    }

    fn range_pbe_adiabatic(description: &str, a_x: f64, omega: f64, pt2: f64) -> Self {
        Self {
            parent: ResolvedFunctional::custom(
                description,
                vec![
                    (1.0 - a_x, Arc::from(gga::pbe_x_sr(omega))),
                    (1.0 - pt2, Arc::from(gga::pbe_c())),
                ],
                HybridParams {
                    cx: a_x,
                    clr: 1.0 - a_x,
                    omega,
                },
                Family::Gga,
            ),
            target: None,
            reference_vv10: None,
            target_vv10: None,
            correlation: Pt2Coefficients {
                same_spin: pt2,
                opposite_spin: pt2,
            },
            dispersion: None,
        }
    }

    /// Original 2013 DSD-PBEP86-D3(BJ) parametrization, not a revised D4 fit.
    pub fn dsd_pbep86() -> Self {
        Self {
            target: None,
            reference_vv10: None,
            target_vv10: None,
            parent: ResolvedFunctional::custom(
                "DSD-PBEP86 (2013 BJ) generalized Kohn-Sham parent",
                vec![
                    (0.31, Arc::from(gga::pbe_x())),
                    (0.44, Arc::from(super::legacy_c::p86())),
                ],
                HybridParams {
                    cx: 0.69,
                    clr: 0.0,
                    omega: 0.0,
                },
                Family::Gga,
            ),
            correlation: Pt2Coefficients {
                same_spin: 0.22,
                opposite_spin: 0.52,
            },
            dispersion: Some(crate::dispersion::D3BjParams {
                s6: 0.48,
                s8: 0.0,
                a1: 0.0,
                a2: 5.6,
            }),
        }
    }
    /// Published DSD-PBEB95-D3(BJ) parametrization: PBE exchange, standard
    /// B95 correlation and independently scaled same/opposite-spin PT2.
    /// The separately optimized dispersion-free fit is not substituted.
    pub fn dsd_pbeb95() -> Self {
        Self::global(
            "DSD-PBEB95 (BJ) generalized Kohn-Sham parent",
            vec![(0.34, gga::pbe_x()), (0.55, super::dh_components::b95_c())],
            Family::MetaGga,
            0.66,
            Pt2Coefficients {
                same_spin: 0.09,
                opposite_spin: 0.46,
            },
            Some(crate::dispersion::D3BjParams {
                s6: 0.61,
                s8: 0.0,
                a1: 0.0,
                a2: 6.2,
            }),
        )
    }

    /// Original 2013 DSD-BLYP-D3(BJ) parametrization. The earlier 2010
    /// D2 fit has different coefficients and is not silently substituted.
    pub fn dsd_blyp() -> Self {
        Self {
            target: None,
            reference_vv10: None,
            target_vv10: None,
            parent: ResolvedFunctional::custom(
                "DSD-BLYP (2013 BJ) generalized Kohn-Sham parent",
                vec![(0.29, Arc::from(gga::b88())), (0.54, Arc::from(gga::lyp()))],
                HybridParams {
                    cx: 0.71,
                    clr: 0.0,
                    omega: 0.0,
                },
                Family::Gga,
            ),
            correlation: Pt2Coefficients {
                same_spin: 0.40,
                opposite_spin: 0.47,
            },
            dispersion: Some(crate::dispersion::D3BjParams {
                s6: 0.57,
                s8: 0.0,
                a1: 0.0,
                a2: 5.4,
            }),
        }
    }

    /// PWPB95-D3(BJ), including the reoptimized PW and B95 ingredients.
    /// The optional pair correction is omitted by dispersion=none.
    pub fn pwpb95() -> Self {
        Self {
            parent: ResolvedFunctional::custom(
                "PWPB95 generalized Kohn-Sham parent",
                vec![
                    (0.5, Arc::from(super::dh_components::pwpb95_x())),
                    (0.731, Arc::from(super::dh_components::pwpb95_c())),
                ],
                HybridParams {
                    cx: 0.5,
                    clr: 0.0,
                    omega: 0.0,
                },
                Family::MetaGga,
            ),
            target: None,
            reference_vv10: None,
            target_vv10: None,
            correlation: Pt2Coefficients {
                same_spin: 0.0,
                opposite_spin: 0.269,
            },
            dispersion: Some(crate::dispersion::D3BjParams {
                s6: 0.82,
                s8: 0.2904,
                a1: 0.0,
                a2: 7.3141,
            }),
        }
    }

    /// A global double hybrid: weighted semilocal parts and the exact
    /// exchange fraction define the self-consistent parent, and
    /// same/opposite-spin PT2 on its orbitals completes the energy.
    fn global(
        description: &str,
        parts: Vec<(f64, Box<dyn super::XcFunctional>)>,
        family: Family,
        exact_exchange: f64,
        correlation: Pt2Coefficients,
        dispersion: Option<crate::dispersion::D3BjParams>,
    ) -> Self {
        Self {
            parent: ResolvedFunctional::custom(
                description,
                parts.into_iter().map(|(w, f)| (w, Arc::from(f))).collect(),
                HybridParams {
                    cx: exact_exchange,
                    clr: 0.0,
                    omega: 0.0,
                },
                family,
            ),
            target: None,
            reference_vv10: None,
            target_vv10: None,
            correlation,
            dispersion,
        }
    }

    /// PT2 with one coefficient for both spin components.
    fn both_spins(scale: f64) -> Pt2Coefficients {
        Pt2Coefficients {
            same_spin: scale,
            opposite_spin: scale,
        }
    }

    /// B2-PLYP-type recipe: `a_x` exact exchange, `1 - a_x` B88 exchange,
    /// `c` LYP correlation and `pt2` of both PT2 spin components.
    fn b2_plyp_type(description: &str, a_x: f64, c: f64, pt2: f64) -> Self {
        Self::global(
            description,
            vec![(1.0 - a_x, gga::b88()), (c, gga::lyp())],
            Family::Gga,
            a_x,
            Self::both_spins(pt2),
            None,
        )
    }

    /// B2GP-PLYP (general purpose): 0.65 exact exchange, 0.64 LYP, 0.36 PT2.
    pub fn b2gp_plyp() -> Self {
        Self::b2_plyp_type("B2GP-PLYP generalized Kohn-Sham parent", 0.65, 0.64, 0.36)
    }
    /// B2K-PLYP (kinetics): 0.72 exact exchange, 0.58 LYP, 0.42 PT2.
    pub fn b2k_plyp() -> Self {
        Self::b2_plyp_type("B2K-PLYP generalized Kohn-Sham parent", 0.72, 0.58, 0.42)
    }
    /// B2T-PLYP (thermochemistry): 0.60 exact exchange, 0.69 LYP, 0.31 PT2.
    pub fn b2t_plyp() -> Self {
        Self::b2_plyp_type("B2T-PLYP generalized Kohn-Sham parent", 0.60, 0.69, 0.31)
    }

    /// mPW2-PLYP-type recipe with mPW91 instead of B88 exchange.
    fn mpw2_plyp_type(description: &str, a_x: f64, c: f64, pt2: f64) -> Self {
        Self::global(
            description,
            vec![(1.0 - a_x, gga::mpw91_x()), (c, gga::lyp())],
            Family::Gga,
            a_x,
            Self::both_spins(pt2),
            None,
        )
    }
    /// mPW2-PLYP: 0.55 exact exchange, 0.45 mPW91, 0.75 LYP, 0.25 PT2.
    pub fn mpw2_plyp() -> Self {
        Self::mpw2_plyp_type("mPW2-PLYP generalized Kohn-Sham parent", 0.55, 0.75, 0.25)
    }
    /// mPW2K-PLYP: 0.72 exact exchange, 0.28 mPW91, 0.58 LYP, 0.42 PT2.
    pub fn mpw2k_plyp() -> Self {
        Self::mpw2_plyp_type("mPW2K-PLYP generalized Kohn-Sham parent", 0.72, 0.58, 0.42)
    }

    /// Adiabatic-connection double hybrid on one exchange-correlation
    /// pair: `a_x` exact exchange, `1 - pt2` of the semilocal correlation
    /// and `pt2` of both PT2 spin components.
    fn adiabatic(
        description: &str,
        exchange: Box<dyn super::XcFunctional>,
        correlation: Box<dyn super::XcFunctional>,
        family: Family,
        a_x: f64,
        pt2: f64,
    ) -> Self {
        Self::global(
            description,
            vec![(1.0 - a_x, exchange), (1.0 - pt2, correlation)],
            family,
            a_x,
            Self::both_spins(pt2),
            None,
        )
    }

    /// PBE0-DH: the 0DH model with `a_x = 1/2` and `pt2 = a_x^3 = 1/8`.
    pub fn pbe0_dh() -> Self {
        Self::adiabatic(
            "PBE0-DH generalized Kohn-Sham parent",
            gga::pbe_x(),
            gga::pbe_c(),
            Family::Gga,
            0.5,
            0.125,
        )
    }
    /// PBE-QIDH: the quadratic-integrand model with `a_x = 3^(-1/3)` and
    /// `pt2 = 1/3`.
    pub fn pbe_qidh() -> Self {
        Self::adiabatic(
            "PBE-QIDH generalized Kohn-Sham parent",
            gga::pbe_x(),
            gga::pbe_c(),
            Family::Gga,
            QIDH_EXCHANGE,
            1.0 / 3.0,
        )
    }
    /// PBE0-2: `a_x = 2^(-1/3)` and `pt2 = 1/2`.
    pub fn pbe0_2() -> Self {
        Self::adiabatic(
            "PBE0-2 generalized Kohn-Sham parent",
            gga::pbe_x(),
            gga::pbe_c(),
            Family::Gga,
            0.793_700_525_984_099_8,
            0.5,
        )
    }
    /// TPSS-QIDH: the QIDH model on TPSS exchange and correlation.
    pub fn tpss_qidh() -> Self {
        Self::adiabatic(
            "TPSS-QIDH generalized Kohn-Sham parent",
            super::tpss::tpss_x(),
            super::tpss::tpss_c(),
            Family::MetaGga,
            QIDH_EXCHANGE,
            1.0 / 3.0,
        )
    }
    /// SCAN0-DH: the 0DH model on SCAN exchange and correlation.
    pub fn scan0_dh() -> Self {
        Self::adiabatic(
            "SCAN0-DH generalized Kohn-Sham parent",
            super::r2scan::scan_x(),
            super::r2scan::scan_c(),
            Family::MetaGga,
            0.5,
            0.125,
        )
    }
    /// SCAN-QIDH: the QIDH model on SCAN exchange and correlation.
    pub fn scan_qidh() -> Self {
        Self::adiabatic(
            "SCAN-QIDH generalized Kohn-Sham parent",
            super::r2scan::scan_x(),
            super::r2scan::scan_c(),
            Family::MetaGga,
            QIDH_EXCHANGE,
            1.0 / 3.0,
        )
    }
    /// r2SCAN0-DH: the 0DH model on r2SCAN exchange and correlation.
    pub fn r2scan0_dh() -> Self {
        Self::adiabatic(
            "r2SCAN0-DH generalized Kohn-Sham parent",
            super::r2scan::r2scan_x(),
            super::r2scan::r2scan_c(),
            Family::MetaGga,
            0.5,
            0.125,
        )
    }
    /// r2SCAN-QIDH: the QIDH model on r2SCAN exchange and correlation.
    pub fn r2scan_qidh() -> Self {
        Self::adiabatic(
            "r2SCAN-QIDH generalized Kohn-Sham parent",
            super::r2scan::r2scan_x(),
            super::r2scan::r2scan_c(),
            Family::MetaGga,
            QIDH_EXCHANGE,
            1.0 / 3.0,
        )
    }

    /// revDSD-PBEP86-D3(BJ) (2019 revision): 0.69 exact exchange, 0.31 PBE
    /// exchange, 0.4210 P86 correlation, 0.5922 opposite-spin and 0.0636
    /// same-spin PT2, and its D3(BJ) term.
    pub fn revdsd_pbep86() -> Self {
        Self::global(
            "revDSD-PBEP86 (2019 BJ) generalized Kohn-Sham parent",
            vec![(0.31, gga::pbe_x()), (0.4210, super::legacy_c::p86())],
            Family::Gga,
            0.69,
            Pt2Coefficients {
                same_spin: 0.0636,
                opposite_spin: 0.5922,
            },
            Some(crate::dispersion::D3BjParams {
                s6: 0.4377,
                s8: 0.0,
                a1: 0.0,
                a2: 5.5,
            }),
        )
    }

    /// XYG3: B3LYP orbitals and the target
    /// `0.8033 E_x^HF - 0.0140 E_x^S + 0.2107 E_x^B88 + 0.6789 E_c^LYP`
    /// plus 0.3211 of both PT2 spin components on those orbitals.
    pub fn xyg3() -> Self {
        Self {
            parent: super::resolve(&super::FunctionalSpec::Combined("b3lyp".into()))
                .expect("built-in orbital reference"),
            target: Some(ResolvedFunctional::custom(
                "XYG3 xDH target",
                vec![
                    (-0.0140, Arc::from(super::lda::slater())),
                    (0.2107, Arc::from(gga::b88())),
                    (0.6789, Arc::from(gga::lyp())),
                ],
                HybridParams {
                    cx: 0.8033,
                    clr: 0.0,
                    omega: 0.0,
                },
                Family::Gga,
            )),
            reference_vv10: None,
            target_vv10: None,
            correlation: Self::both_spins(0.3211),
            dispersion: None,
        }
    }

    /// xDH-PBE0 (Su and Xu, J. Chem. Phys. 136, 174103, 2012) evaluates
    /// a separate hybrid target and opposite-spin PT2 on PBE0 orbitals.
    pub fn xdh_pbe0() -> Self {
        Self {
            parent: super::resolve(&super::FunctionalSpec::Combined("pbe0".into()))
                .expect("built-in orbital reference"),
            target: Some(ResolvedFunctional::custom(
                "xDH-PBE0 target",
                vec![
                    (0.1665, Arc::from(gga::pbe_x())),
                    (0.5292, Arc::from(gga::pbe_c())),
                ],
                HybridParams {
                    cx: 0.8335,
                    clr: 0.0,
                    omega: 0.0,
                },
                Family::Gga,
            )),
            reference_vv10: None,
            target_vv10: None,
            correlation: Pt2Coefficients {
                same_spin: 0.0,
                opposite_spin: 0.5428,
            },
            dispersion: None,
        }
    }

    /// Chai and Head-Gordon's LP fit for 6-311++G(3df,3pd).
    pub fn wb97x2_lp() -> Self {
        Self::wb97x2(
            "ωB97X-2(LP)",
            0.678792,
            [
                [0.251767, 1.57375, -5.26624, 6.74313],
                [1.15698, -3.31669, 6.27265, -4.51464],
                [0.553261, -1.16626, 6.84409, -8.90640],
            ],
            Pt2Coefficients {
                same_spin: 0.581569,
                opposite_spin: 0.477992,
            },
        )
    }

    /// Chai and Head-Gordon's TQZ fit at a supplied basis. Applications
    /// following the paper's two-basis protocol extrapolate the PT2 terms.
    pub fn wb97x2_tqz() -> Self {
        Self::wb97x2(
            "ωB97X-2(TQZ)",
            0.636158,
            [
                [0.315503, 1.04772, -2.33506, 3.19909],
                [0.908460, -2.80936, 6.02676, -4.56981],
                [0.518198, -0.585956, 4.27080, -6.48897],
            ],
            Pt2Coefficients {
                same_spin: 0.529319,
                opposite_spin: 0.447105,
            },
        )
    }

    /// Equation (1) and Table II of J. Chem. Phys. 131, 174105 (2009).
    /// Full-range PT2 is evaluated on the fitted parent's stationary orbitals.
    fn wb97x2(
        name: &'static str,
        short_range: f64,
        coeff: [[f64; 4]; 3],
        correlation: Pt2Coefficients,
    ) -> Self {
        let omega = 0.3;
        Self {
            parent: ResolvedFunctional::custom(
                name,
                vec![(1.0, Arc::from(super::b97::wb97x2(name, omega, coeff)))],
                HybridParams {
                    cx: short_range,
                    clr: 1.0 - short_range,
                    omega,
                },
                Family::Gga,
            ),
            target: None,
            reference_vv10: None,
            target_vv10: None,
            correlation,
            dispersion: None,
        }
    }

    /// omegaB97M(2): omegaB97M-V stationary orbitals, a distinct RSH
    /// meta-GGA target, 0.65904 VV10 and 0.34096 full-Coulomb PT2.
    pub fn wb97m2() -> Self {
        Self {
            parent: super::resolve(&super::FunctionalSpec::Combined("wb97m-v".into()))
                .expect("built-in orbital reference"),
            target: Some(ResolvedFunctional::custom(
                "omegaB97M(2) xDH target",
                super::b97::wb97m2_parts(),
                HybridParams {
                    cx: 0.62194,
                    clr: 0.37806,
                    omega: 0.3,
                },
                Family::MetaGga,
            )),
            reference_vv10: Some(crate::nlc::Vv10Params::WB97XV),
            target_vv10: Some((0.65904, crate::nlc::Vv10Params { b: 10.0, c: 0.01 })),
            correlation: Pt2Coefficients {
                same_spin: 0.34096,
                opposite_spin: 0.34096,
            },
            dispersion: None,
        }
    }
}
