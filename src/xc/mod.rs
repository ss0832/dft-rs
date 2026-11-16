// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Functional registry and name resolution.
//!
//! Users specify either a combined name (`--xc svwn5`) or split exchange +
//! correlation (`--x slater --c vwn5`). Resolution happens **before** any
//! SCF work:
//! 1. combined names hit the COMBINED table;
//! 2. split names hit the EXCHANGE / CORRELATION tables;
//! 3. a combined-only (XC-integrated) name given component-wise is accepted
//!    iff X == C (then treated as combined), otherwise it is a hard error;
//! 4. `hf` resolves to zero functional parts with 100 % exact exchange.
//!
//! Hybrid metadata (`cx`, `clr`, `ω`) lives on the resolved composite —
//! the Fock builder always accepts `cx·K + clr·K_erf(ω)`, so range-separated
//! hybrids are new table entries, not new machinery.

#[macro_use]
mod common;
pub(crate) use common::RHO_CUTOFF;
pub mod b97;
#[cfg(test)]
pub(crate) mod branch_freeze;
pub mod catalogue;
pub mod dh_components;
pub mod dldf;
pub mod double_hybrid;
pub mod gga;
pub mod kcis;
pub mod kinetic;
pub mod lda;
pub use kinetic::MagneticKineticDensity;
pub mod bidirectional;
pub mod legacy_c;
pub mod m06;
pub mod mgga;
pub mod mn12;
pub mod pkzb;
pub mod r2scan;
pub mod sr_x;
pub mod tao_mo;
pub mod task;
pub mod taylor;
pub mod thcth;
pub mod tpss;
pub mod traits;
pub(crate) mod wb97x_d3;
pub use taylor::{XcTaylorDirectionalOutput, XcTaylorInput, XcTaylorOutput};

use crate::error::{DftError, Result};
pub use traits::{Family, GridVars, Kind, XcFunctional, XcOut};

use std::sync::Arc;

/// Exact-exchange metadata of a resolved functional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridParams {
    /// Global HF-exchange fraction.
    pub cx: f64,
    /// Additional long-range HF exchange (range-separated hybrids).
    pub clr: f64,
    /// Range-separation parameter ω (Bohr⁻¹); 0 when unused.
    pub omega: f64,
}

impl HybridParams {
    pub const PURE: Self = Self {
        cx: 0.0,
        clr: 0.0,
        omega: 0.0,
    };
    pub const HF: Self = Self {
        cx: 1.0,
        clr: 0.0,
        omega: 0.0,
    };

    /// Global hybrid with exact-exchange fraction `cx`.
    pub const fn global(cx: f64) -> Self {
        Self {
            cx,
            clr: 0.0,
            omega: 0.0,
        }
    }
}

/// User-level functional specification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionalSpec {
    Combined(String),
    Split { x: String, c: String },
}

/// A functional term in a weighted variational XC specification.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedFunctionalTerm {
    pub weight: f64,
    pub functional: FunctionalSpec,
}

/// A ready-to-evaluate functional: weighted parts + hybrid metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequiredNonlocalTerm {
    Vv10,
    Chg,
    D3Zero,
}

impl RequiredNonlocalTerm {
    pub fn name(self) -> &'static str {
        match self {
            Self::Vv10 => "VV10",
            Self::Chg => "CHG",
            Self::D3Zero => "D3(0)",
        }
    }

    /// Geometry-only model that the calculation must add to the electronic energy.
    pub fn pair_dispersion(self) -> Option<crate::dispersion::Dispersion> {
        match self {
            Self::Vv10 => None,
            Self::Chg => Some(crate::dispersion::Dispersion::Chg),
            Self::D3Zero => Some(crate::dispersion::Dispersion::D3Zero),
        }
    }

    /// Whether the caller has supplied this model's density or geometry term.
    pub(crate) fn is_supplied(self, vv10: bool, pair_dispersion: bool) -> bool {
        match self {
            Self::Vv10 => vv10,
            Self::Chg | Self::D3Zero => pair_dispersion,
        }
    }
}

/// Nonlocal term that a combined-table name carries by definition.
///
/// Deliberately a free function rather than a `match` inside [`combined`]:
/// the coverage census in `tests/p14_functional_coverage.rs` reads the
/// registry tables as source text, and every `=>` arm between `fn combined`
/// and `pub fn resolve` is taken to be one table entry. A second, unrelated
/// match sitting inside that span reads as an entry naming three functionals
/// at once and breaks the census. Keeping the classification out here keeps
/// the table a table.
fn required_nonlocal_for(name: &str) -> Option<RequiredNonlocalTerm> {
    match name {
        "wb97xv" | "wb97mv" | "b97mv" => Some(RequiredNonlocalTerm::Vv10),
        "wb97xd" => Some(RequiredNonlocalTerm::Chg),
        "wb97xd3" => Some(RequiredNonlocalTerm::D3Zero),
        _ => None,
    }
}

/// VV10 parameters of the nonlocal correlation that a combined-table name
/// carries by definition (kept next to [`required_nonlocal_for`], outside
/// the registry table, for the same reason).
fn required_vv10_for(name: &str) -> Option<crate::nlc::Vv10Params> {
    match required_nonlocal_for(name) {
        Some(RequiredNonlocalTerm::Vv10) => Some(crate::nlc::Vv10Params::WB97XV),
        _ => None,
    }
}

/// Nonlocal terms that the named combined functional carries by
/// definition: its VV10 parameters and whether it carries the CHG pair
/// term. A name that is not a combined functional carries neither.
pub fn built_in_nonlocal(name: &str) -> (Option<crate::nlc::Vv10Params>, bool) {
    let (vv10, model) = built_in_dispersion(name);
    (vv10, model == Some(RequiredNonlocalTerm::Chg))
}

/// Density and geometry models carried by a named functional.
pub fn built_in_dispersion(
    name: &str,
) -> (Option<crate::nlc::Vv10Params>, Option<RequiredNonlocalTerm>) {
    let name = normalize(name);
    (required_vv10_for(&name), required_nonlocal_for(&name))
}

pub struct ResolvedFunctional {
    pub description: String,
    pub parts: Vec<(f64, Arc<dyn XcFunctional>)>,
    pub hybrid: HybridParams,
    pub family: Family,
    required_nonlocal: Option<RequiredNonlocalTerm>,
    vv10_parameters: Option<crate::nlc::Vv10Params>,
}

impl ResolvedFunctional {
    /// Resolve a weighted XC energy on one common set of orbitals.
    /// This operator supports local parts and one long-range exchange kernel.
    pub fn weighted(terms: &[WeightedFunctionalTerm]) -> Result<Self> {
        let mut result = Self::hartree_fock();
        result.hybrid = HybridParams::PURE;
        let mut labels = Vec::new();
        for term in terms {
            let weight = term.weight;
            if !weight.is_finite() {
                return Err(DftError::InvalidInput(
                    "XC term weights must be finite".into(),
                ));
            }
            if let FunctionalSpec::Combined(name) = &term.functional {
                if name.eq_ignore_ascii_case("skala")
                    || double_hybrid::DoubleHybridFunctional::named(name).is_some()
                {
                    return Err(DftError::Unsupported {
                        feature: format!("weighted XC term `{name}`"),
                        missing: "weighted learned or post-SCF functional operators".into(),
                    });
                }
            }
            let functional = resolve(&term.functional)?;
            if weight == 0.0 {
                continue;
            }
            if let Some(nonlocal) = functional.required_nonlocal_term() {
                return Err(DftError::Unsupported {
                    feature: "weighted XC composition".into(),
                    missing: format!("a weighted {nonlocal} operator and its derivatives"),
                });
            }
            let hybrid = functional.hybrid;
            if hybrid.clr != 0.0 {
                if result.hybrid.omega != 0.0 && result.hybrid.omega != hybrid.omega {
                    return Err(DftError::Unsupported {
                        feature: "weighted XC composition".into(),
                        missing: "multiple distinct long-range exchange kernels".into(),
                    });
                }
                result.hybrid.omega = hybrid.omega;
            }
            result.hybrid.cx += weight * hybrid.cx;
            result.hybrid.clr += weight * hybrid.clr;
            result.family = result.family.max_rung(functional.family);
            result.parts.extend(
                functional
                    .parts
                    .into_iter()
                    .map(|(coef, part)| (weight * coef, part)),
            );
            labels.push(format!("{weight} * ({})", functional.description));
        }
        if labels.is_empty() {
            return Err(DftError::InvalidInput(
                "xc_terms must contain at least one nonzero term".into(),
            ));
        }
        if !result.hybrid.cx.is_finite()
            || !result.hybrid.clr.is_finite()
            || result
                .parts
                .iter()
                .any(|(coefficient, _)| !coefficient.is_finite())
        {
            return Err(DftError::InvalidInput(
                "XC term coefficients overflow".into(),
            ));
        }
        if result.hybrid.clr == 0.0 {
            result.hybrid.omega = 0.0;
        }
        result.description = labels.join(" + ");
        Ok(result)
    }

    /// Potential and mixed directional-potential paths for arbitrary providers.
    /// This supplies transverse spin kernels without a named-functional branch.
    pub fn semilocal_taylor_directional_potentials(
        &self,
        input: &XcTaylorInput,
        direction: &[Vec<f64>; 7],
    ) -> Result<XcTaylorDirectionalOutput> {
        input.validate()?;
        if input.parameters.is_some() {
            return Err(DftError::InvalidInput(
                "composite parameter derivatives require per-part Taylor inputs".into(),
            ));
        }
        let mut result = XcTaylorDirectionalOutput {
            value: XcTaylorOutput::zeros(input.order() + 1),
            directional: XcTaylorOutput::zeros(input.order() + 1),
        };
        for (weight, part) in &self.parts {
            let contribution = part.taylor_directional_potentials(input, direction)?;
            for (out, value) in [
                (&mut result.value, contribution.value),
                (&mut result.directional, contribution.directional),
            ] {
                if value.energy.len() != out.energy.len()
                    || value.potentials.iter().any(|p| p.len() != out.energy.len())
                {
                    return Err(DftError::InvalidInput(
                        "functional mixed Taylor output has an incompatible order".into(),
                    ));
                }
                for (a, b) in out.energy.iter_mut().zip(value.energy) {
                    *a += weight * b;
                }
                for (a, b) in out.potentials.iter_mut().zip(value.potentials) {
                    for (a, b) in a.iter_mut().zip(b) {
                        *a += weight * b;
                    }
                }
            }
        }
        Ok(result)
    }
    /// Semilocal energy and density-potential series. This does not include
    /// Hartree, exact exchange, nonlocal correlation or solvent terms.
    pub fn semilocal_taylor_potentials(&self, input: &XcTaylorInput) -> Result<XcTaylorOutput> {
        input.validate()?;
        if input.parameters.is_some() {
            return Err(DftError::InvalidInput(
                "composite parameter derivatives require per-part Taylor inputs".into(),
            ));
        }
        let mut result = XcTaylorOutput::zeros(input.order() + 1);
        for (weight, part) in &self.parts {
            let coefficients = part.taylor_potentials(input)?;
            if coefficients.energy.len() != result.energy.len()
                || coefficients
                    .potentials
                    .iter()
                    .any(|v| v.len() != result.energy.len())
            {
                return Err(DftError::InvalidInput(
                    "functional Taylor output has an incompatible order".into(),
                ));
            }
            for (a, b) in result.energy.iter_mut().zip(coefficients.energy) {
                *a += weight * b;
            }
            for (a, b) in result.potentials.iter_mut().zip(coefficients.potentials) {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += weight * b;
                }
            }
        }
        Ok(result)
    }
    /// Factorial-normalized directional coefficients of the semilocal
    /// energy density. Exact exchange and required nonlocal tails are
    /// separate contributions and are not included in this result.
    /// Composite parameters are provided to each part through its own input.
    pub fn semilocal_taylor_energy(&self, input: &XcTaylorInput) -> Result<Vec<f64>> {
        input.validate()?;
        if input.parameters.is_some() {
            return Err(DftError::InvalidInput(
                "composite parameter derivatives require per-part Taylor inputs".into(),
            ));
        }
        let mut result = vec![0.0; input.order() + 1];
        for (weight, part) in &self.parts {
            let coefficients = part.taylor_energy(input)?;
            if coefficients.len() != result.len() {
                return Err(DftError::InvalidInput(
                    "functional Taylor output has an incompatible order".into(),
                ));
            }
            for (value, coefficient) in result.iter_mut().zip(coefficients) {
                *value += weight * coefficient;
            }
        }
        Ok(result)
    }
    /// Nonlocal term that is part of the named functional's definition.
    ///
    /// The semilocal and exact-exchange pieces live in this object, while
    /// the defining density or geometry term is assembled by the SCF driver. Consumers that do
    /// not implement that nonlocal operator must reject the functional
    /// rather than silently report its semilocal part under the full name.
    pub fn required_nonlocal_term(&self) -> Option<&'static str> {
        self.required_nonlocal.map(RequiredNonlocalTerm::name)
    }

    /// Typed nonlocal contribution required by the named functional.
    pub fn required_nonlocal_kind(&self) -> Option<RequiredNonlocalTerm> {
        self.required_nonlocal
    }

    /// Geometry-only dispersion model carried by the resolved functional.
    pub fn required_pair_dispersion(&self) -> Option<crate::dispersion::Dispersion> {
        self.required_nonlocal
            .and_then(RequiredNonlocalTerm::pair_dispersion)
    }

    /// VV10 parameters of the nonlocal correlation the named functional
    /// carries by definition.
    pub fn required_vv10(&self) -> Option<crate::nlc::Vv10Params> {
        self.vv10_parameters
    }

    /// Construct a custom semilocal/hybrid functional with no implicit
    /// nonlocal tail. Named built-ins should be created through [`resolve`],
    /// which attaches their VV10/CHG requirement as typed metadata.
    pub fn custom(
        description: impl Into<String>,
        parts: Vec<(f64, Arc<dyn XcFunctional>)>,
        hybrid: HybridParams,
        family: Family,
    ) -> Self {
        Self {
            description: description.into(),
            parts,
            hybrid,
            family,
            required_nonlocal: None,
            vv10_parameters: None,
        }
    }

    /// Evaluate all parts (weighted) into `out`.
    pub fn eval(&self, vars: &GridVars, order: usize, out: &mut XcOut) -> Result<()> {
        out.prepare(vars.len(), order, self.family);
        for (coef, part) in &self.parts {
            part.eval_add(vars, order, *coef, out)?;
        }
        Ok(())
    }

    /// Pure Hartree–Fock (no functional parts, full exact exchange).
    pub fn hartree_fock() -> Self {
        Self {
            description: "HF (exact exchange, no XC functional)".to_string(),
            parts: Vec::new(),
            hybrid: HybridParams::HF,
            family: Family::Lda,
            required_nonlocal: None,
            vv10_parameters: None,
        }
    }

    /// A functional whose exchange-correlation energy is evaluated outside
    /// the semilocal parts (a learned model on the molecular grid): no parts
    /// and no exact exchange.
    pub fn model_based(description: &str) -> Self {
        Self {
            description: description.to_string(),
            parts: Vec::new(),
            hybrid: HybridParams::PURE,
            family: Family::MetaGga,
            required_nonlocal: None,
            vv10_parameters: None,
        }
    }

    /// Override named parameters of the functional parts: every part that
    /// has a parameter of a given name takes the new value (for example
    /// `tau_r` or `switch_upper` of rSCAN). A name that no part has is an
    /// error, so a misspelt override cannot pass silently. Range separation
    /// is changed with [`Self::with_omega`], which also updates the exact
    /// exchange kernel.
    pub fn with_parameters(&self, overrides: &[(String, f64)]) -> Result<Self> {
        let mut parts = self.parts.clone();
        let mut applied = Vec::new();
        for (_, part) in parts.iter_mut() {
            let names = part.param_names();
            let mut params = part.params();
            let mut changed = false;
            for (name, value) in overrides {
                if name == "omega" {
                    return Err(DftError::InvalidInput(
                        "set the range separation with the omega control".into(),
                    ));
                }
                if let Some(index) = names.iter().position(|n| n == name) {
                    params[index] = *value;
                    changed = true;
                    applied.push(name.clone());
                }
            }
            if changed {
                *part = Arc::from(part.with_params(&params)?);
            }
        }
        if let Some((name, _)) = overrides.iter().find(|(name, _)| !applied.contains(name)) {
            return Err(DftError::InvalidInput(format!(
                "`{}` has no functional parameter `{name}`",
                self.description
            )));
        }
        let listed: Vec<String> = overrides
            .iter()
            .map(|(n, v)| format!("{n} = {v}"))
            .collect();
        Ok(Self {
            description: format!("{} [{}]", self.description, listed.join(", ")),
            parts,
            hybrid: self.hybrid,
            family: self.family,
            required_nonlocal: self.required_nonlocal,
            vv10_parameters: self.vv10_parameters,
        })
    }

    /// Replace the range-separation parameter ω everywhere it appears: in
    /// the hybrid metadata (long-range HF kernel) and in every SCREENED
    /// functional part carrying an "omega" parameter (ω-tuning workflows).
    /// Parts whose current ω is exactly 0 are full-range by construction
    /// (e.g. the wPBEh(0) parent inside HSE06) and are left untouched.
    pub fn with_omega(&self, omega: f64) -> Result<Self> {
        if self.hybrid.clr == 0.0 {
            return Err(DftError::InvalidInput(format!(
                "`{}` is not range-separated; ω has no effect",
                self.description
            )));
        }
        let mut parts: Vec<(f64, Arc<dyn XcFunctional>)> = Vec::with_capacity(self.parts.len());
        for (coef, part) in &self.parts {
            let idx = part
                .param_names()
                .iter()
                .position(|&n| n == "omega")
                .filter(|&i| part.params()[i] != 0.0);
            if let Some(idx) = idx {
                let mut params = part.params();
                params[idx] = omega;
                parts.push((*coef, Arc::from(part.with_params(&params)?)));
            } else {
                parts.push((*coef, Arc::clone(part)));
            }
        }
        Ok(Self {
            description: format!("{} [ω = {omega}]", self.description),
            parts,
            hybrid: HybridParams {
                omega,
                ..self.hybrid
            },
            family: self.family,
            required_nonlocal: self.required_nonlocal,
            vv10_parameters: self.vv10_parameters,
        })
    }
}

pub(crate) fn normalize(name: &str) -> String {
    name.trim()
        .to_ascii_lowercase()
        .replace(['_', ' ', '-'], "")
}

/// Exchange table: canonical name + aliases → constructor.
fn exchange(name: &str) -> Option<Box<dyn XcFunctional>> {
    match name {
        "slater" | "s" | "ldax" | "xalpha" => Some(lda::slater()),
        "xalpha07" | "xa07" => Some(lda::xalpha()),
        "b88" | "b" | "becke88" => Some(gga::b88()),
        "pbe" | "pbex" => Some(gga::pbe_x()),
        "revpbe" | "revpbex" => Some(gga::revpbe_x()),
        "pbesol" | "pbesolx" => Some(gga::pbesol_x()),
        "rpbe" | "rpbex" => Some(gga::rpbe_x()),
        "pw91" | "pw91x" => Some(gga::pw91_x()),
        "pw86" | "pw86x" => Some(gga::pw86_x()),
        "mpw91" | "mpw" | "mpw91x" => Some(gga::mpw91_x()),
        "g96" | "g96x" => Some(gga::g96_x()),
        "lg93" => Some(gga::lg93_x()),
        "ft97" | "ft97b" => Some(gga::ft97_x()),
        "xpbe" | "xpbex" => Some(gga::xpbe_x()),
        "sogga" | "soggax" => Some(gga::sogga_x()),
        "mpbe" | "mpbex" => Some(gga::mpbe_x()),
        "apbe" | "apbex" => Some(gga::apbe_x()),
        "pbeint" | "pbeintx" => Some(gga::pbeint_x()),
        "wc" | "wcx" => Some(gga::wc_x()),
        "kt1" | "kt1x" => Some(gga::kt1_x()),
        "am05" | "am05x" => Some(gga::am05_x()),
        "n12" | "n12x" => Some(gga::n12_x()),
        "n12sx" | "n12sxx" => Some(gga::n12_sx_x()),
        "dldf" | "dldfx" => Some(dldf::dldf_x()),
        "thcth" | "tauhcth" => Some(thcth::thcth_x()),
        "thcthhyb" | "hybtauhcth" => Some(thcth::thcth_hybrid_x()),
        "bmk" | "bmkx" => Some(thcth::bmk_x()),
        "gam" | "gamx" => Some(gga::gam_x()),
        "sogga11" | "sogga11x" => Some(gga::sogga11_x()),
        "scan" | "scanx" => Some(r2scan::scan_x()),
        "rscan" | "rscanx" => Some(r2scan::rscan_x()),
        "revtpss" | "revtpssx" => Some(tpss::revtpss_x()),
        "modtpss" | "modtpssx" => Some(tpss::modtpss_x()),
        "tpss" | "tpssx" => Some(tpss::tpss_x()),
        "tm" | "tmx" => Some(tao_mo::tm_x()),
        "revtm" | "revtmx" => Some(tao_mo::revtm_x()),
        "pkzb" | "pkzbx" => Some(pkzb::pkzb_x()),
        "task" | "taskx" => Some(task::task_x()),
        "mtask" | "mtaskx" => Some(task::mtask_x()),
        "r2scan" | "r2scanx" => Some(r2scan::r2scan_x()),
        "optx" => Some(gga::optx()),
        // SR (erfc-screened) ωPBE exchange, HJS hole; default ω = 0.11.
        "hjspbex" | "hjspbe" => Some(sr_x::hjs_pbe_x(0.11)),
        _ => None,
    }
}

/// Correlation table.
fn correlation(name: &str) -> Option<Box<dyn XcFunctional>> {
    match name {
        "vwn5" | "vwn" => Some(lda::vwn5()),
        "vwn3" | "vwnrpa" => Some(lda::vwn3()),
        "pw92" | "pw" | "pwlda" => Some(lda::pw92()),
        "pz81" | "pz" => Some(lda::pz81()),
        "wigner" => Some(lda::wigner()),
        "hl" | "hedinlundqvist" => Some(lda::hl()),
        "gl" | "gunnarssonlundqvist" => Some(lda::gl()),
        "lyp" => Some(gga::lyp()),
        "pbe" | "pbec" => Some(gga::pbe_c()),
        "tpss" | "tpssc" => Some(tpss::tpss_c()),
        "tm" | "tmc" => Some(tao_mo::tm_c()),
        "revtm" | "revtmc" => Some(tao_mo::revtm_c()),
        "tpssloc" | "tpsslocc" => Some(tpss::tpssloc_c()),
        "pkzb" | "pkzbc" => Some(pkzb::pkzb_c()),
        "r2scan" | "r2scanc" => Some(r2scan::r2scan_c()),
        "p86" => Some(legacy_c::p86()),
        "pw91" | "pw91c" => Some(legacy_c::pw91_c()),
        "pbesol" | "pbesolc" => Some(gga::pbesol_c()),
        "xpbe" | "xpbec" => Some(gga::xpbe_c()),
        "apbe" | "apbec" => Some(gga::apbe_c()),
        "pbeint" | "pbeintc" => Some(gga::pbeint_c()),
        "am05" | "am05c" => Some(gga::am05_c()),
        "n12" | "n12c" => Some(b97::n12_c()),
        "n12sx" | "n12sxc" => Some(b97::n12_sx_c()),
        "dldf" | "dldfc" => Some(dldf::dldf_c()),
        "kcis" | "kcisc" => Some(kcis::kcis_c()),
        "thcth" | "tauhcth" => Some(b97::thcth_c()),
        "thcthhyb" | "hybtauhcth" => Some(b97::thcth_hybrid_c()),
        "bmk" | "bmkc" => Some(b97::bmk_c()),
        "gam" | "gamc" => Some(b97::gam_c()),
        "sogga11" | "sogga11c" => Some(gga::sogga11_c()),
        "pbeloc" | "pbelocc" => Some(gga::pbeloc_c()),
        "vpbe" | "pbevwn" => Some(gga::pbe_vwn_c()),
        "op" | "opb88" => Some(gga::op_b88_c()),
        "b95" | "bc95" => Some(dh_components::b95_c()),
        "scan" | "scanc" => Some(r2scan::scan_c()),
        "rscan" | "rscanc" => Some(r2scan::rscan_c()),
        "revtpss" | "revtpssc" => Some(tpss::revtpss_c()),
        _ => None,
    }
}

/// Combined table: full XC definitions (and XC-integrated functionals).
fn combined(name: &str) -> Option<ResolvedFunctional> {
    let make = |desc: &str, hybrid: HybridParams, parts: Vec<(f64, Box<dyn XcFunctional>)>| {
        ResolvedFunctional {
            description: desc.to_string(),
            family: parts
                .iter()
                .map(|(_, p)| p.family())
                .fold(Family::Lda, Family::max_rung),
            parts: parts.into_iter().map(|(c, p)| (c, Arc::from(p))).collect(),
            hybrid,
            required_nonlocal: required_nonlocal_for(name),
            vv10_parameters: required_vv10_for(name),
        }
    };
    let pure = HybridParams::PURE;
    match name {
        "svwn5" | "svwn" | "lda" | "ldavwn5" => Some(make(
            "SVWN5 (Slater exchange + VWN5 correlation)",
            pure,
            vec![(1.0, lda::slater()), (1.0, lda::vwn5())],
        )),
        "svwn3" | "ldavwn3" => Some(make(
            "SVWN3 (Slater exchange + VWN3/RPA correlation)",
            pure,
            vec![(1.0, lda::slater()), (1.0, lda::vwn3())],
        )),
        "spz81" | "ldapz81" => Some(make(
            "SPZ81 (Slater exchange + Perdew-Zunger 1981 correlation)",
            pure,
            vec![(1.0, lda::slater()), (1.0, lda::pz81())],
        )),
        "xalpha07" => Some(make(
            "X-alpha (alpha = 0.7, no correlation)",
            pure,
            vec![(1.0, lda::xalpha())],
        )),
        "spw92" | "ldapw92" | "pw92lda" | "spwlda" => Some(make(
            "SPW92 (Slater exchange + PW92 correlation)",
            pure,
            vec![(1.0, lda::slater()), (1.0, lda::pw92())],
        )),
        "pbe" => Some(make(
            "PBE (PBE exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::pbe_x()), (1.0, gga::pbe_c())],
        )),
        "revpbe" => Some(make(
            "revPBE (Zhang-Yang exchange, kappa = 1.245 + PBE correlation)",
            pure,
            vec![(1.0, gga::revpbe_x()), (1.0, gga::pbe_c())],
        )),
        "pbesol" => Some(make(
            "PBEsol (PBE with mu = 10/81 and beta = 0.046)",
            pure,
            vec![(1.0, gga::pbesol_x()), (1.0, gga::pbesol_c())],
        )),
        "bpw91" => Some(make(
            "BPW91 (B88 exchange + PW91 correlation)",
            pure,
            vec![(1.0, gga::b88()), (1.0, legacy_c::pw91_c())],
        )),
        "olyp" => Some(make(
            "OLYP (OPTX exchange + LYP correlation)",
            pure,
            vec![(1.0, gga::optx()), (1.0, gga::lyp())],
        )),
        "opbe" => Some(make(
            "OPBE (OPTX exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::optx()), (1.0, gga::pbe_c())],
        )),
        "opw91" => Some(make(
            "OPW91 (OPTX exchange + PW91 correlation)",
            pure,
            vec![(1.0, gga::optx()), (1.0, legacy_c::pw91_c())],
        )),
        "mpwlyp" => Some(make(
            "mPWLYP (mPW91 exchange + LYP correlation)",
            pure,
            vec![(1.0, gga::mpw91_x()), (1.0, gga::lyp())],
        )),
        "mpwpw91" | "mpwpw" => Some(make(
            "mPWPW91 (mPW91 exchange + PW91 correlation)",
            pure,
            vec![(1.0, gga::mpw91_x()), (1.0, legacy_c::pw91_c())],
        )),
        "mpwpbe" => Some(make(
            "mPWPBE (mPW91 exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::mpw91_x()), (1.0, gga::pbe_c())],
        )),
        "g96lyp" => Some(make(
            "G96LYP (Gill 96 exchange + LYP correlation)",
            pure,
            vec![(1.0, gga::g96_x()), (1.0, gga::lyp())],
        )),
        "xlyp" => Some(make(
            "XLYP (Slater, B88 and PW91 exchange + LYP correlation)",
            pure,
            vec![
                (-0.069, lda::slater()),
                (0.722, gga::b88()),
                (0.347, gga::pw91_x()),
                (1.0, gga::lyp()),
            ],
        )),
        "kt1" => Some(make(
            "KT1 (Keal-Tozer exchange + VWN5 correlation)",
            pure,
            vec![(1.0, gga::kt1_x()), (1.0, lda::vwn5())],
        )),
        "kt2" => Some(make(
            "KT2 (1.07173 Slater + Keal-Tozer gradient term + 0.576727 VWN5)",
            pure,
            vec![
                (0.07173, lda::slater()),
                (1.0, gga::kt1_x()),
                (0.576727, lda::vwn5()),
            ],
        )),
        "kt3" => Some(make(
            "KT3 (Slater, OPTX and Keal-Tozer exchange + 0.864409 LYP)",
            pure,
            vec![
                (-0.5877016340967667, lda::slater()),
                (1.0, gga::kt3_x()),
                (0.6464052972361336, gga::optx()),
                (0.864409, gga::lyp()),
            ],
        )),
        "sogga" => Some(make(
            "SOGGA (second-order GGA exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::sogga_x()), (1.0, gga::pbe_c())],
        )),
        "xpbe" => Some(make(
            "xPBE (Xu-Goddard exchange and correlation)",
            pure,
            vec![(1.0, gga::xpbe_x()), (1.0, gga::xpbe_c())],
        )),
        "apbe" => Some(make(
            "APBE (mu = 0.26 exchange and correlation)",
            pure,
            vec![(1.0, gga::apbe_x()), (1.0, gga::apbe_c())],
        )),
        "pbeint" => Some(make(
            "PBEint (interface PBE exchange and correlation)",
            pure,
            vec![(1.0, gga::pbeint_x()), (1.0, gga::pbeint_c())],
        )),
        "wc" | "wcpbe" => Some(make(
            "WC (Wu-Cohen exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::wc_x()), (1.0, gga::pbe_c())],
        )),
        "mpbe" | "mpbepbe" => Some(make(
            "mPBE (Adamo-Barone exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::mpbe_x()), (1.0, gga::pbe_c())],
        )),
        "am05" => Some(make(
            "AM05 (Armiento-Mattsson exchange and correlation)",
            pure,
            vec![(1.0, gga::am05_x()), (1.0, gga::am05_c())],
        )),
        "n12" => Some(make(
            "N12 (nonseparable gradient approximation)",
            pure,
            vec![(1.0, gga::n12_x()), (1.0, b97::n12_c())],
        )),
        "n12sx" => Some(make(
            "N12-SX (screened hybrid NGA; SR-HF = 0.25, LR-HF = 0, omega = 0.11)",
            HybridParams {
                cx: 0.25,
                clr: -0.25,
                omega: 0.11,
            },
            vec![(1.0, gga::n12_sx_x()), (1.0, b97::n12_sx_c())],
        )),
        "dldf" => Some(make(
            "dlDF (dispersionless hybrid meta-GGA; HF = 0.6144129)",
            HybridParams::global(0.6144129),
            vec![(1.0, dldf::dldf_x()), (1.0, dldf::dldf_c())],
        )),
        "b98" => Some(make(
            "B98 (Schmider-Becke fit2c; HF = 0.219847)",
            HybridParams::global(0.219847),
            vec![(1.0, b97::b98())],
        )),
        "bmk" => Some(make(
            "BMK (Boese-Martin published fit; HF = 0.42)",
            HybridParams::global(0.42),
            vec![(1.0, thcth::bmk_x()), (1.0, b97::bmk_c())],
        )),
        "thcth" | "tauhcth" => Some(make(
            "tau-HCTH (pure meta-GGA)",
            pure,
            vec![(1.0, thcth::thcth_x()), (1.0, b97::thcth_c())],
        )),
        "thcthhyb" | "hybtauhcth" => Some(make(
            "Hybrid tau-HCTH (HF = 0.15)",
            HybridParams::global(0.15),
            vec![(1.0, thcth::thcth_hybrid_x()), (1.0, b97::thcth_hybrid_c())],
        )),
        "gam" => Some(make(
            "GAM (gradient approximation for molecules)",
            pure,
            vec![(1.0, gga::gam_x()), (1.0, b97::gam_c())],
        )),
        "sogga11" => Some(make(
            "SOGGA11 (second-order GGA 2011 exchange and correlation)",
            pure,
            vec![(1.0, gga::sogga11_x()), (1.0, gga::sogga11_c())],
        )),
        "sogga11x" => Some(make(
            "SOGGA11-X (global hybrid; cx = 0.4015)",
            HybridParams {
                cx: 0.4015,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, gga::sogga11x_x()), (1.0, gga::sogga11x_c())],
        )),
        "pbeloc" => Some(make(
            "PBEloc (PBE exchange + PBEloc correlation)",
            pure,
            vec![(1.0, gga::pbe_x()), (1.0, gga::pbeloc_c())],
        )),
        "bop" => Some(make(
            "BOP (B88 exchange + one-parameter progressive correlation)",
            pure,
            vec![(1.0, gga::b88()), (1.0, gga::op_b88_c())],
        )),
        "edf1" => Some(make(
            "EDF1 (empirical density functional 1)",
            pure,
            gga::edf1_parts(),
        )),
        "edf2" => Some(make(
            "EDF2 (cx = 0.1695; published independent component weights)",
            HybridParams::global(0.1695),
            vec![
                (0.2811, lda::slater()),
                (0.6227, gga::b88()),
                (
                    -0.0551,
                    gga::b88().with_params(&[0.0035]).expect("EDF2 B88"),
                ),
                (0.3029, lda::vwn5()),
                (0.5998, gga::lyp()),
                (
                    -0.0053,
                    gga::lyp()
                        .with_params(&[0.055, 0.158, 0.25, 0.3505])
                        .expect("EDF2 LYP"),
                ),
            ],
        )),
        "bb95" => Some(make(
            "BB95 (B88 exchange + B95 correlation)",
            pure,
            vec![(1.0, gga::b88()), (1.0, dh_components::b95_c())],
        )),
        "b1b95" | "b88b95" => Some(make(
            "B1B95 (cx = 0.28; B88 exchange + B95 correlation)",
            HybridParams {
                cx: 0.28,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(0.72, gga::b88()), (1.0, dh_components::b95_c())],
        )),
        "bb1k" => Some(make(
            "BB1K (cx = 0.42; B88 exchange + B95 correlation)",
            HybridParams {
                cx: 0.42,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(0.58, gga::b88()), (1.0, dh_components::b95_c())],
        )),
        "mpw1b95" => Some(make(
            "mPW1B95 (cx = 0.31; mPW91 exchange + B95 correlation)",
            HybridParams {
                cx: 0.31,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(0.69, gga::mpw91_x()), (1.0, dh_components::b95_c())],
        )),
        "mpw1kcis" => Some(make(
            "mPW1KCIS (cx = 0.15; mPW91 exchange + KCIS correlation)",
            HybridParams { cx: 0.15, ..pure },
            vec![(0.85, gga::mpw91_x()), (1.0, kcis::kcis_c())],
        )),
        "pbe1kcis" => Some(make(
            "PBE1KCIS (cx = 0.22; PBE exchange + KCIS correlation)",
            HybridParams { cx: 0.22, ..pure },
            vec![(0.78, gga::pbe_x()), (1.0, kcis::kcis_c())],
        )),
        "pw6b95" => {
            let (x, c) = dh_components::pw6b95_parts();
            Some(make(
                "PW6B95 (cx = 0.28; refitted PW exchange and B95 correlation)",
                HybridParams {
                    cx: 0.28,
                    clr: 0.0,
                    omega: 0.0,
                },
                vec![(0.72, x), (1.0, c)],
            ))
        }
        "pwb6k" => {
            let (x, c) = dh_components::pwb6k_parts();
            Some(make(
                "PWB6K (cx = 0.46; refitted PW exchange and B95 correlation)",
                HybridParams {
                    cx: 0.46,
                    clr: 0.0,
                    omega: 0.0,
                },
                vec![(0.54, x), (1.0, c)],
            ))
        }
        "b97d" => Some(make(
            "B97-D (XC-integrated power series without exact exchange)",
            pure,
            vec![(1.0, b97::b97_d())],
        )),
        "rpbe" => Some(make(
            "RPBE (Hammer-Hansen-Norskov exchange + PBE correlation)",
            pure,
            vec![(1.0, gga::rpbe_x()), (1.0, gga::pbe_c())],
        )),
        "pw91" => Some(make(
            "PW91 (Perdew-Wang 1991 exchange and correlation)",
            pure,
            vec![(1.0, gga::pw91_x()), (1.0, legacy_c::pw91_c())],
        )),
        "scan" => Some(make(
            "SCAN (strongly constrained and appropriately normed meta-GGA)",
            pure,
            vec![(1.0, r2scan::scan_x()), (1.0, r2scan::scan_c())],
        )),
        // LRC-wPBEh (Rohrdanz-Martins-Herbert 2009): 0.2 SR-HF + full
        // LR-HF at omega = 0.2, 0.8 SR-wPBE (HJS hole) + PBE correlation.
        "lrcwpbeh" => Some(make(
            "LRC-ωPBEh (ω = 0.2; SR-HF 0.2, LR-HF 1.0; HJS hole)",
            HybridParams {
                cx: 0.2,
                clr: 0.8,
                omega: 0.2,
            },
            vec![(0.8, sr_x::hjs_pbe_x(0.2)), (1.0, gga::pbe_c())],
        )),
        "rscan" => Some(make(
            "rSCAN (regularized SCAN)",
            pure,
            vec![(1.0, r2scan::rscan_x()), (1.0, r2scan::rscan_c())],
        )),
        "revtpss" => Some(make(
            "revTPSS (revised TPSS exchange and correlation)",
            pure,
            vec![(1.0, tpss::revtpss_x()), (1.0, tpss::revtpss_c())],
        )),
        "modtpss" => Some(make(
            "modTPSS (modified TPSS exchange + TPSS correlation)",
            pure,
            vec![(1.0, tpss::modtpss_x()), (1.0, tpss::tpss_c())],
        )),
        "tpss" => Some(make(
            "TPSS (nonempirical meta-GGA)",
            pure,
            vec![(1.0, tpss::tpss_x()), (1.0, tpss::tpss_c())],
        )),
        "tm" => Some(make(
            "Tao-Mo (TM exchange and correlation)",
            pure,
            vec![(1.0, tao_mo::tm_x()), (1.0, tao_mo::tm_c())],
        )),
        "revtm" => Some(make(
            "revTM (revised Tao-Mo exchange and correlation)",
            pure,
            vec![(1.0, tao_mo::revtm_x()), (1.0, tao_mo::revtm_c())],
        )),
        "tpssloc" => Some(make(
            "TPSSloc (TPSS exchange + localized TPSS correlation)",
            pure,
            vec![(1.0, tpss::tpss_x()), (1.0, tpss::tpssloc_c())],
        )),
        "task" => Some(make(
            "TASK exchange + PW92 correlation",
            pure,
            vec![(1.0, task::task_x()), (1.0, lda::pw92())],
        )),
        "mtask" => Some(make(
            "mTASK exchange + PW92 correlation",
            pure,
            vec![(1.0, task::mtask_x()), (1.0, lda::pw92())],
        )),
        "pkzb" => Some(make(
            "PKZB (original nonempirical meta-GGA; FHC input convention)",
            pure,
            vec![(1.0, pkzb::pkzb_x()), (1.0, pkzb::pkzb_c())],
        )),
        "tpssh" => Some(make(
            "TPSSh (TPSS hybrid; cx = 0.10)",
            HybridParams {
                cx: 0.10,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(0.90, tpss::tpss_x()), (1.0, tpss::tpss_c())],
        )),
        "r2scan" => Some(make(
            "r²SCAN (re-regularized SCAN meta-GGA)",
            pure,
            vec![(1.0, r2scan::r2scan_x()), (1.0, r2scan::r2scan_c())],
        )),
        "bp86" => Some(make(
            "BP86 (B88 exchange + P86 correlation)",
            pure,
            vec![(1.0, gga::b88()), (1.0, legacy_c::p86())],
        )),
        // LC-ωPBE with the HJS 2008 SR hole (libxc LC_WPBE_WHS): full
        // LR-HF + SR ωPBE exchange at ω = 0.4 + PBE correlation.
        "lcwpbehjs" | "lcwpbewhs" => Some(make(
            "LC-ωPBE (HJS hole; ω = 0.4; full LR-HF)",
            HybridParams {
                cx: 0.0,
                clr: 1.0,
                omega: 0.4,
            },
            vec![(1.0, sr_x::hjs_pbe_x(0.4)), (1.0, gga::pbe_c())],
        )),
        // HSE06 (Heyd-Scuseria-Ernzerhof, ω-tuned 2006 form): 0.25 SR-HF
        // (K − K_erf) + full-range PBE-hole exchange − 0.25 SR-wPBEh(0.11)
        // + PBE correlation.
        "hse06" | "hse" => Some(make(
            "HSE06 (screened hybrid; ω = 0.11, SR-HF = 0.25)",
            HybridParams {
                cx: 0.25,
                clr: -0.25,
                omega: 0.11,
            },
            vec![
                (1.0, sr_x::wpbeh_x(0.0)),
                (-0.25, sr_x::wpbeh_x(0.11)),
                (1.0, gga::pbe_c()),
            ],
        )),
        // HSEsol replaces one quarter of the HJS short-range PBEsol exchange
        // with exchange from the identically screened Fock operator.
        "hsesol" => Some(make(
            "HSEsol (screened PBEsol hybrid; ω = 0.11, SR-HF = 0.25)",
            HybridParams {
                cx: 0.25,
                clr: -0.25,
                omega: 0.11,
            },
            vec![
                (1.0, gga::pbesol_x()),
                (-0.25, sr_x::hjs_pbesol_x(0.11)),
                (1.0, gga::pbesol_c()),
            ],
        )),
        // HSE03 (original 2003 parameters): ω_HF = 0.15/√2, ω_PBE = 0.15·2^{1/3}.
        "hse03" => Some(make(
            "HSE03 (screened hybrid; ω_HF = 0.1061, ω_PBE = 0.1890, SR-HF = 0.25)",
            HybridParams {
                cx: 0.25,
                clr: -0.25,
                omega: 0.10606601717798213,
            },
            vec![
                (1.0, sr_x::wpbeh_x(0.0)),
                (-0.25, sr_x::wpbeh_x(0.18898815748423097)),
                (1.0, gga::pbe_c()),
            ],
        )),
        // Canonical LC-ωPBE (Vydrov-Scuseria 2006): SR-wPBEh(0.4) + full LR-HF.
        "lcwpbe" => Some(make(
            "LC-ωPBE (HSE hole; ω = 0.4; full LR-HF)",
            HybridParams {
                cx: 0.0,
                clr: 1.0,
                omega: 0.4,
            },
            vec![(1.0, sr_x::wpbeh_x(0.4)), (1.0, gga::pbe_c())],
        )),
        // B3PW91 (Becke 1993): 0.20 Ex^HF + 0.80 Ex^Slater + 0.72 dEx^B88
        // + Ec^PW92 + 0.81 (Ec^PW91 - Ec^PW92).
        "b3pw91" => Some(make(
            "B3PW91 (cx = 0.20; PW91 gradient correlation)",
            HybridParams {
                cx: 0.20,
                clr: 0.0,
                omega: 0.0,
            },
            vec![
                (0.08, lda::slater()),
                (0.72, gga::b88()),
                (0.19, lda::pw92()),
                (0.81, legacy_c::pw91_c()),
            ],
        )),
        "blyp" => Some(make(
            "BLYP (B88 exchange + LYP correlation)",
            pure,
            vec![(1.0, gga::b88()), (1.0, gga::lyp())],
        )),
        // B3LYP (Gaussian convention: VWN3 local correlation):
        // Exc = 0.20 Ex^HF + 0.80 Ex^Slater + 0.72 ΔEx^B88
        //     + 0.19 Ec^VWN3 + 0.81 Ec^LYP,  with ΔB88 = B88 − Slater.
        "b3lyp" | "b3lypg" => Some(make(
            "B3LYP (VWN3/Gaussian convention; cx = 0.20)",
            HybridParams {
                cx: 0.20,
                clr: 0.0,
                omega: 0.0,
            },
            vec![
                (0.08, lda::slater()),
                (0.72, gga::b88()),
                (0.19, lda::vwn3()),
                (0.81, gga::lyp()),
            ],
        )),
        "b3lyp5" => Some(make(
            "B3LYP5 (VWN5 convention; cx = 0.20)",
            HybridParams {
                cx: 0.20,
                clr: 0.0,
                omega: 0.0,
            },
            vec![
                (0.08, lda::slater()),
                (0.72, gga::b88()),
                (0.19, lda::vwn5()),
                (0.81, gga::lyp()),
            ],
        )),
        // O3LYP (Handy): Exc = 0.1161 Ex^HF + 0.9262 Ex^Slater
        //   + 0.8133 ΔEx^OPTX + 0.19 Ec^VWN5 + 0.81 Ec^LYP,
        // where ΔOPTX is OPTX's gradient term (a1 = 0).
        "o3lyp" => {
            let optx_delta = gga::optx()
                .with_params(&[0.0, 1.43169, 0.006])
                .expect("optx params");
            Some(make(
                "O3LYP (cx = 0.1161)",
                HybridParams {
                    cx: 0.1161,
                    clr: 0.0,
                    omega: 0.0,
                },
                vec![
                    (0.9262, lda::slater()),
                    (0.8133, optx_delta),
                    (0.19, lda::vwn5()),
                    (0.81, gga::lyp()),
                ],
            ))
        }
        // APF (Austin et al., JCTC 8, 4989 (2012)): 0.411 B3PW91 + 0.589
        // PBE0, without the APFD dispersion model.
        "apf" => {
            let (b3, pbe0) = (0.411, 0.589);
            Some(make(
                "APF (0.411 B3PW91 + 0.589 PBE0; cx = 0.22945)",
                HybridParams {
                    cx: b3 * 0.20 + pbe0 * 0.25,
                    clr: 0.0,
                    omega: 0.0,
                },
                vec![
                    (b3 * 0.08, lda::slater()),
                    (b3 * 0.72, gga::b88()),
                    (b3 * 0.19, lda::pw92()),
                    (b3 * 0.81, legacy_c::pw91_c()),
                    (pbe0 * 0.75, gga::pbe_x()),
                    (pbe0, gga::pbe_c()),
                ],
            ))
        }
        "pbe0" | "pbe1pbe" => Some(make(
            "PBE0 (cx = 0.25)",
            HybridParams {
                cx: 0.25,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(0.75, gga::pbe_x()), (1.0, gga::pbe_c())],
        )),
        // CAM-B3LYP (Yanai/Tew/Handy 2004): 1/r = [α + β·erf(ωr)]/r (HF)
        //   + [1−α−β·erf(ωr)]/r (DFT) with α = 0.19, β = 0.46, ω = 0.33;
        // DFT exchange = (1−α−β)·B88 + β·SR-B88(ITYH);
        // correlation = 0.19 VWN5 + 0.81 LYP.
        "camb3lyp" => {
            let (alpha, beta, omega) = (0.19, 0.46, 0.33);
            Some(make(
                "CAM-B3LYP (α = 0.19, β = 0.46, ω = 0.33)",
                HybridParams {
                    cx: alpha,
                    clr: beta,
                    omega,
                },
                vec![
                    (1.0 - alpha - beta, gga::b88()),
                    (beta, gga::b88_sr(omega)),
                    (0.19, lda::vwn5()),
                    (0.81, gga::lyp()),
                ],
            ))
        }
        // RCAM-B3LYP (Cohen, Mori-Sanchez and Yang, JCP 126, 191109 (2007)):
        // exact exchange alpha + beta erfc(omega r) with alpha = 1.13331,
        // beta = -0.94979, omega = 0.33; B88 fraction 0.95238 and LYP.
        "rcamb3lyp" => {
            let (alpha, beta, omega, b88) = (1.13331, -0.94979, 0.33, 0.95238);
            Some(make(
                "RCAM-B3LYP (alpha = 1.13331, beta = -0.94979, omega = 0.33)",
                HybridParams {
                    cx: alpha + beta,
                    clr: -beta,
                    omega,
                },
                vec![
                    (1.0 - alpha - b88 - beta, lda::slater()),
                    (b88 + beta, gga::b88()),
                    (-beta, gga::b88_sr(omega)),
                    (1.0, gga::lyp()),
                ],
            ))
        }
        // LC-BOP (Song et al., JCP 126, 154105 (2007)): full long-range
        // exact exchange, ITYH SR-B88 and OP(B88) correlation, omega = 0.47.
        "lcbop" => {
            let omega = 0.47;
            Some(make(
                "LC-BOP (ITYH SR-B88 + OP correlation; omega = 0.47)",
                HybridParams {
                    cx: 0.0,
                    clr: 1.0,
                    omega,
                },
                vec![(1.0, gga::b88_sr(omega)), (1.0, gga::op_b88_c())],
            ))
        }
        // LC-BLYP: full long-range HF exchange (clr = 1), SR-B88 + LYP.
        "lcblyp" => {
            let omega = 0.33;
            Some(make(
                "LC-BLYP (ITYH SR-B88 + LYP; ω = 0.33)",
                HybridParams {
                    cx: 0.0,
                    clr: 1.0,
                    omega,
                },
                vec![(1.0, gga::b88_sr(omega)), (1.0, gga::lyp())],
            ))
        }
        // LC-PBE via ITYH attenuation (NOT the HJS/ωPBE hole of canonical
        // LC-ωPBE — see docs/FUNCTIONALS.md).
        "lcpbe" => {
            let omega = 0.47;
            Some(make(
                "LC-PBE (ITYH SR-PBE + PBE correlation; ω = 0.47)",
                HybridParams {
                    cx: 0.0,
                    clr: 1.0,
                    omega,
                },
                vec![(1.0, gga::pbe_x_sr(omega)), (1.0, gga::pbe_c())],
            ))
        }
        // B97 family: exchange-correlation-INTEGRATED functionals (the
        // component-wise resolution rule applies).
        // B3-type hybrids: (1 - a0 - ax) Slater + ax B88-type + a0 HF,
        // (1 - ac) VWN(RPA) + ac gradient correlation.
        "b3lyp*" | "b3lyps" => Some(make(
            "B3LYP* (cx = 0.15; ax 0.72, ac 0.81, VWN RPA)",
            HybridParams::global(0.15),
            vec![
                (0.13, lda::slater()),
                (0.72, gga::b88()),
                (0.19, lda::vwn3()),
                (0.81, gga::lyp()),
            ],
        )),
        "x3lyp" => Some(make(
            "X3LYP (cx = 0.218; 0.709 of 0.765 B88 + 0.235 PW91, ac 0.871, VWN RPA)",
            HybridParams::global(0.218),
            vec![
                (0.073, lda::slater()),
                (0.542385, gga::b88()),
                (0.166615, gga::pw91_x()),
                (0.129, lda::vwn3()),
                (0.871, gga::lyp()),
            ],
        )),
        "b3p86" => Some(make(
            "B3P86 (cx = 0.20; ax 0.72, ac 0.81, VWN RPA + P86)",
            HybridParams::global(0.20),
            vec![
                (0.08, lda::slater()),
                (0.72, gga::b88()),
                (0.19, lda::vwn3()),
                (0.81, legacy_c::p86()),
            ],
        )),
        "mpw3pw" | "mpw3pw91" => Some(make(
            "mPW3PW91 (cx = 0.20; ax 0.72 mPW91, ac 0.81, VWN RPA + PW91)",
            HybridParams::global(0.20),
            vec![
                (0.08, lda::slater()),
                (0.72, gga::mpw91_x()),
                (0.19, lda::vwn3()),
                (0.81, legacy_c::pw91_c()),
            ],
        )),
        "b1lyp" => Some(make(
            "B1LYP (cx = 0.25; B88 + LYP)",
            HybridParams::global(0.25),
            vec![(0.75, gga::b88()), (1.0, gga::lyp())],
        )),
        // Renz et al., JACS 131, 16292 (2009): one-parameter BLYP hybrid.
        "blyp35" => Some(make(
            "BLYP35 (cx = 0.35; B88 exchange + full LYP correlation)",
            HybridParams::global(0.35),
            vec![(0.65, gga::b88()), (1.0, gga::lyp())],
        )),
        // Wiitala, Hoye and Cramer, JCTC 2, 1085 (2006). Use the explicit
        // VWN-RPA convention of the numerical reference, not VWN5: the paper
        // does not specify that local fit. The five fitted coefficients are
        // independent; neither exchange nor correlation is renormalized.
        "wc04" => Some(make(
            "WC04 (cx = 0.7400; fitted carbon-shift hybrid, VWN-RPA convention)",
            HybridParams::global(0.7400),
            vec![
                (0.0001 - 0.9999, lda::slater()),
                (0.9999, gga::b88()),
                (0.9999 - 0.0001, lda::vwn3()),
                (0.0001, gga::lyp()),
            ],
        )),
        "wp04" => Some(make(
            "WP04 (cx = 0.1189; fitted proton-shift hybrid, VWN-RPA convention)",
            HybridParams::global(0.1189),
            vec![
                (0.9999 - 0.9614, lda::slater()),
                (0.9614, gga::b88()),
                (0.9999 - 0.0001, lda::vwn3()),
                (0.0001, gga::lyp()),
            ],
        )),
        "b1pw91" => Some(make(
            "B1PW91 (cx = 0.25; B88 + PW91 correlation)",
            HybridParams::global(0.25),
            vec![(0.75, gga::b88()), (1.0, legacy_c::pw91_c())],
        )),
        "bhandh" => Some(make(
            "BHandH (cx = 0.5; Slater + LYP)",
            HybridParams::global(0.5),
            vec![(0.5, lda::slater()), (1.0, gga::lyp())],
        )),
        "bhandhlyp" | "bhhlyp" => Some(make(
            "BHandHLYP (cx = 0.5; B88 + LYP)",
            HybridParams::global(0.5),
            vec![(0.5, gga::b88()), (1.0, gga::lyp())],
        )),
        "pbe01/3" | "pbe013" => Some(make(
            "PBE0-1/3 (cx = 1/3; PBE exchange and correlation)",
            HybridParams::global(1.0 / 3.0),
            vec![(2.0 / 3.0, gga::pbe_x()), (1.0, gga::pbe_c())],
        )),
        "mpw1pw91" | "mpw1pw" => Some(make(
            "mPW1PW91 (cx = 0.25; mPW91 + PW91 correlation)",
            HybridParams::global(0.25),
            vec![(0.75, gga::mpw91_x()), (1.0, legacy_c::pw91_c())],
        )),
        "mpw1lyp" => Some(make(
            "mPW1LYP (cx = 0.25; mPW91 + LYP)",
            HybridParams::global(0.25),
            vec![(0.75, gga::mpw91_x()), (1.0, gga::lyp())],
        )),
        "mpw1pbe" => Some(make(
            "mPW1PBE (cx = 0.25; mPW91 + PBE correlation)",
            HybridParams::global(0.25),
            vec![(0.75, gga::mpw91_x()), (1.0, gga::pbe_c())],
        )),
        "mpw1k" => Some(make(
            "mPW1K (cx = 0.428; mPW91 + PW91 correlation)",
            HybridParams::global(0.428),
            vec![(0.572, gga::mpw91_x()), (1.0, legacy_c::pw91_c())],
        )),
        "b971" => Some(make(
            "B97-1 (XC-integrated power series; cx = 0.21)",
            HybridParams::global(0.21),
            vec![(1.0, b97::b97_1())],
        )),
        "b972" => Some(make(
            "B97-2 (XC-integrated power series; cx = 0.21)",
            HybridParams::global(0.21),
            vec![(1.0, b97::b97_2())],
        )),
        "b973" => Some(make(
            "B97-3 (XC-integrated power series; cx = 0.269288)",
            HybridParams::global(0.269288),
            vec![(1.0, b97::b97_3())],
        )),
        "b97k" => Some(make(
            "B97-K (XC-integrated power series; cx = 0.42)",
            HybridParams::global(0.42),
            vec![(1.0, b97::b97_k())],
        )),
        "tpss0" => Some(make(
            "TPSS0 (cx = 0.25; TPSS exchange and correlation)",
            HybridParams::global(0.25),
            vec![(0.75, tpss::tpss_x()), (1.0, tpss::tpss_c())],
        )),
        "revtpssh" => Some(make(
            "revTPSSh (cx = 0.10; revTPSS exchange and correlation)",
            HybridParams::global(0.10),
            vec![(0.90, tpss::revtpss_x()), (1.0, tpss::revtpss_c())],
        )),
        "scan0" => Some(make(
            "SCAN0 (cx = 0.25; SCAN exchange and correlation)",
            HybridParams::global(0.25),
            vec![(0.75, r2scan::scan_x()), (1.0, r2scan::scan_c())],
        )),
        "r2scan0" => Some(make(
            "r2SCAN0 (cx = 0.25; r2SCAN exchange and correlation)",
            HybridParams::global(0.25),
            vec![(0.75, r2scan::r2scan_x()), (1.0, r2scan::r2scan_c())],
        )),
        "r2scanh" => Some(make(
            "r2SCANh (cx = 0.10; r2SCAN exchange and correlation)",
            HybridParams::global(0.10),
            vec![(0.90, r2scan::r2scan_x()), (1.0, r2scan::r2scan_c())],
        )),
        "r2scan50" => Some(make(
            "r2SCAN50 (cx = 0.50; r2SCAN exchange and correlation)",
            HybridParams::global(0.50),
            vec![(0.50, r2scan::r2scan_x()), (1.0, r2scan::r2scan_c())],
        )),
        "b97" => Some(make(
            "B97 (XC-integrated power series; cx = 0.1943)",
            HybridParams {
                cx: 0.1943,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, b97::b97())],
        )),
        "hcth93" | "hcth" => Some(make(
            "HCTH/93 (XC-integrated power series)",
            pure,
            vec![(1.0, b97::hcth93())],
        )),
        "hcth120" => Some(make(
            "HCTH/120 (XC-integrated power series)",
            pure,
            vec![(1.0, b97::hcth120())],
        )),
        "hcth147" => Some(make(
            "HCTH/147 (XC-integrated power series)",
            pure,
            vec![(1.0, b97::hcth147())],
        )),
        "hcth407" => Some(make(
            "HCTH/407 (XC-integrated power series)",
            pure,
            vec![(1.0, b97::hcth407())],
        )),
        // ωB97 family (coefficients verified against libxc by linear
        // least-squares recovery; SR-HF/LR-HF split via cx/clr).
        "wb97" => Some(make(
            "ωB97 (ω = 0.4; full LR-HF)",
            HybridParams {
                cx: 0.0,
                clr: 1.0,
                omega: 0.4,
            },
            vec![(1.0, b97::wb97())],
        )),
        "wb97x" => Some(make(
            "ωB97X (ω = 0.3; SR-HF cx = 0.157706)",
            HybridParams {
                cx: 0.157706,
                clr: 0.842294,
                omega: 0.3,
            },
            vec![(1.0, b97::wb97x())],
        )),
        "wb97xd" => Some(make(
            "ωB97X-D semilocal part (ω = 0.2; SR-HF cx = 0.222036)",
            HybridParams {
                cx: 0.222036,
                clr: 0.777964,
                omega: 0.2,
            },
            vec![(1.0, b97::wb97x_d())],
        )),
        "wb97xd3" => Some(make(
            "ωB97X-D3 (original zero-damped pair correction)",
            wb97x_d3::HYBRID,
            vec![(1.0, wb97x_d3::semilocal())],
        )),
        "wb97xv" => Some(make(
            "ωB97X-V (ω = 0.3; SR-HF cx = 0.167; requires the VV10 NLC term)",
            HybridParams {
                cx: 0.167,
                clr: 0.833,
                omega: 0.3,
            },
            vec![(1.0, b97::wb97x_v())],
        )),
        "wb97mv" => Some(make(
            "ωB97M-V (meta-GGA RSH; ω = 0.3, SR-HF cx = 0.15; requires the VV10 NLC term)",
            HybridParams {
                cx: 0.15,
                clr: 0.85,
                omega: 0.3,
            },
            vec![(1.0, b97::wb97m_v())],
        )),
        "b97mv" => Some(make(
            "B97M-V (pure meta-GGA; requires the VV10 NLC term)",
            pure,
            vec![(1.0, b97::b97m_v())],
        )),
        // M06 family (meta-GGA; coefficient tables verified against libxc
        // by least-squares recovery, ≤8e-13).
        "m06l" => Some(make(
            "M06-L (pure meta-GGA)",
            pure,
            vec![(1.0, m06::m06_l())],
        )),
        "m06" => Some(make(
            "M06 (meta-GGA hybrid; cx = 0.27)",
            HybridParams {
                cx: 0.27,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, m06::m06())],
        )),
        "m062x" => Some(make(
            "M06-2X (meta-GGA hybrid; cx = 0.54)",
            HybridParams {
                cx: 0.54,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, m06::m06_2x())],
        )),
        "revm06l" => Some(make(
            "revM06-L (pure meta-GGA)",
            pure,
            vec![(1.0, m06::revm06_l())],
        )),
        "m08hx" => Some(make(
            "M08-HX (meta-GGA hybrid; cx = 0.5223)",
            HybridParams {
                cx: 0.5223,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, mn12::m08_hx())],
        )),
        "m08so" => Some(make(
            "M08-SO (meta-GGA hybrid; cx = 0.5679)",
            HybridParams {
                cx: 0.5679,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, mn12::m08_so())],
        )),
        "m11" => Some(make(
            "M11 (range-separated meta-GGA hybrid; SR-HF = 0.428, LR-HF = 1, omega = 0.25)",
            HybridParams {
                cx: 0.428,
                clr: 0.572,
                omega: 0.25,
            },
            vec![(1.0, mn12::m11())],
        )),
        "revm11" => Some(make(
            "revM11 (range-separated meta-GGA hybrid; SR-HF = 0.225, LR-HF = 1, omega = 0.40)",
            HybridParams {
                cx: 0.225,
                clr: 0.775,
                omega: 0.40,
            },
            vec![(1.0, mn12::revm11())],
        )),
        "m11l" => Some(make(
            "M11-L (dual-range local meta-GGA)",
            pure,
            vec![(1.0, mn12::m11_l())],
        )),
        "mn12l" => Some(make(
            "MN12-L (meta-nonseparable gradient approximation)",
            pure,
            vec![(1.0, mn12::mn12_l())],
        )),
        "mn12sx" => Some(make(
            "MN12-SX (screened exchange: SR-HF = 0.25, omega = 0.11)",
            HybridParams {
                cx: 0.25,
                clr: -0.25,
                omega: 0.11,
            },
            vec![(1.0, mn12::mn12_sx())],
        )),
        "mn15l" => Some(make(
            "MN15-L (meta-nonseparable gradient approximation)",
            pure,
            vec![(1.0, mn12::mn15_l())],
        )),
        "mn15" => Some(make(
            "MN15 (meta-NGA hybrid; cx = 0.44)",
            HybridParams {
                cx: 0.44,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, mn12::mn15())],
        )),
        "vsxc" => Some(make(
            "VSXC (Van Voorhis-Scuseria meta-GGA)",
            pure,
            vec![(1.0, m06::vsxc())],
        )),
        "m05" => Some(make(
            "M05 (meta-GGA hybrid; cx = 0.28)",
            HybridParams {
                cx: 0.28,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, m06::m05())],
        )),
        "m052x" => Some(make(
            "M05-2X (meta-GGA hybrid; cx = 0.56)",
            HybridParams {
                cx: 0.56,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, m06::m05_2x())],
        )),
        "m06hf" => Some(make(
            "M06-HF (meta-GGA hybrid; cx = 1.0)",
            HybridParams {
                cx: 1.0,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, m06::m06_hf())],
        )),
        "revm06" => Some(make(
            "revM06 (meta-GGA hybrid; cx = 0.4041)",
            HybridParams {
                cx: 0.4041,
                clr: 0.0,
                omega: 0.0,
            },
            vec![(1.0, m06::revm06())],
        )),
        // M06-SX (Wang 2020): screened Minnesota hybrid, K − K_erf trick.
        "m06sx" => Some(make(
            "M06-SX (screened meta-GGA hybrid; ω = 0.10, SR-HF = 0.335)",
            HybridParams {
                cx: 0.335,
                clr: -0.335,
                omega: 0.10,
            },
            vec![(1.0, m06::m06_sx())],
        )),
        "hf" => Some(ResolvedFunctional::hartree_fock()),
        _ => None,
    }
}

/// Canonical `--x` names — one per EXCHANGE table arm, aliases omitted.
/// The registry is a `match`, so it cannot be enumerated by reflection;
/// these lists are the enumerable mirror used by `--list-functionals` and
/// by the coverage census (which reads this file and fails if a table arm
/// has no canonical name here).
pub const EXCHANGE_NAMES: &[&str] = &[
    "tm",
    "revtm",
    "thcth",
    "thcthhyb",
    "bmk",
    "dldf",
    "task",
    "mtask",
    "pkzb",
    "slater",
    "xalpha07",
    "b88",
    "pbe",
    "revpbe",
    "tpss",
    "r2scan",
    "optx",
    "hjs-pbe-x",
    "pbesol",
    "rpbe",
    "pw91",
    "scan",
    "pw86",
    "mpw91",
    "g96",
    "lg93",
    "ft97",
    "xpbe",
    "sogga",
    "mpbe",
    "apbe",
    "pbeint",
    "wc",
    "kt1",
    "am05",
    "n12",
    "n12-sx",
    "gam",
    "sogga11",
    "rscan",
    "revtpss",
    "modtpss",
];

/// Canonical `--c` names — one per CORRELATION table arm.
pub const CORRELATION_NAMES: &[&str] = &[
    "thcth", "thcthhyb", "bmk", "dldf", "tpssloc", "pkzb", "vwn5", "vwn3", "pw92", "pz81",
    "wigner", "hl", "gl", "lyp", "pbe", "tpss", "r2scan", "p86", "pw91", "pbesol", "scan", "xpbe",
    "apbe", "pbeint", "am05", "n12", "n12-sx", "gam", "sogga11", "pbeloc", "vpbe", "op", "rscan",
    "revtpss", "b95", "kcis", "tm", "revtm",
];

/// Canonical `--xc` names — one per COMBINED table arm.
pub const COMBINED_NAMES: &[&str] = &[
    "tm",
    "revtm",
    "b98",
    "thcth",
    "thcthhyb",
    "bmk",
    "dldf",
    "edf2",
    "task",
    "mtask",
    "tpssloc",
    "pkzb",
    "svwn5",
    "svwn3",
    "spw92",
    "spz81",
    "xalpha07",
    "pbe",
    "revpbe",
    "pbesol",
    "rpbe",
    "pw91",
    "scan",
    "lrc-wpbeh",
    "bpw91",
    "olyp",
    "opbe",
    "opw91",
    "mpwlyp",
    "mpwpw91",
    "mpwpbe",
    "g96lyp",
    "xlyp",
    "kt1",
    "kt2",
    "kt3",
    "sogga",
    "xpbe",
    "apbe",
    "pbeint",
    "wc",
    "mpbe",
    "b97-d",
    "am05",
    "n12",
    "n12-sx",
    "gam",
    "sogga11",
    "sogga11-x",
    "pbeloc",
    "bop",
    "edf1",
    "rscan",
    "revtpss",
    "modtpss",
    "bb95",
    "b1b95",
    "bb1k",
    "mpw1b95",
    "mpw1kcis",
    "pbe1kcis",
    "pw6b95",
    "pwb6k",
    "tpss",
    "tpssh",
    "r2scan",
    "bp86",
    "lc-wpbe-hjs",
    "hse06",
    "hse03",
    "hsesol",
    "lc-wpbe",
    "b3pw91",
    "blyp",
    "b3lyp",
    "b3lyp5",
    "o3lyp",
    "pbe0",
    "cam-b3lyp",
    "lc-blyp",
    "lc-pbe",
    "b3lyp*",
    "x3lyp",
    "b3p86",
    "mpw3pw91",
    "b1lyp",
    "blyp35",
    "wc04",
    "wp04",
    "b1pw91",
    "bhandh",
    "bhandhlyp",
    "pbe0-1/3",
    "mpw1pw91",
    "mpw1lyp",
    "mpw1pbe",
    "mpw1k",
    "b97-1",
    "b97-2",
    "b97-3",
    "b97-k",
    "tpss0",
    "revtpssh",
    "scan0",
    "r2scan0",
    "r2scanh",
    "r2scan50",
    "b97",
    "hcth93",
    "hcth120",
    "hcth147",
    "hcth407",
    "wb97",
    "wb97x",
    "wb97x-d",
    "wb97x-d3",
    "wb97x-v",
    "wb97m-v",
    "b97m-v",
    "m06l",
    "m06",
    "m06-2x",
    "vsxc",
    "m08-hx",
    "m08-so",
    "m11",
    "revm11",
    "m11-l",
    "mn12-l",
    "mn12-sx",
    "mn15-l",
    "mn15",
    "m05",
    "m05-2x",
    "m06-hf",
    "apf",
    "rcam-b3lyp",
    "lc-bop",
    "revm06l",
    "revm06",
    "m06-sx",
    "hf",
];

/// Resolve a user specification (fails fast, before any SCF work).
pub fn resolve(spec: &FunctionalSpec) -> Result<ResolvedFunctional> {
    match spec {
        FunctionalSpec::Combined(name) => {
            let norm = normalize(name);
            combined(&norm).ok_or_else(|| {
                // Named from the registry table, not a literal list: the
                // literal had drifted to 10 of the 40 combined entries, so
                // the message actively misled about what is available.
                DftError::FunctionalResolution(format!(
                    "unknown combined functional `{name}`\n  known: {}, hf\n\
                     (`--list-functionals` prints the whole registry with descriptions)",
                    COMBINED_NAMES.join(", ")
                ))
            })
        }
        FunctionalSpec::Split { x, c } => {
            let (nx, nc) = (normalize(x), normalize(c));
            // `hf` as exchange with no correlation.
            if nx == "hf" && (nc == "none" || nc.is_empty()) {
                return Ok(ResolvedFunctional::hartree_fock());
            }
            let ex = exchange(&nx);
            let co = if nc == "none" { None } else { correlation(&nc) };
            match (ex, co) {
                (Some(ex), Some(co)) => {
                    // The MAXIMUM rung, not a two-case test. The previous
                    // form collapsed every non-(LDA,LDA) pair onto GGA, so a
                    // split spec that named a meta-GGA on ONE side only —
                    // `--x slater --c tpss`, `--x pbe --c r2scan`,
                    // `--x tpss --c vwn5`, all of them legal, since `tpss`
                    // and `r2scan` appear in both EXCHANGE_NAMES and
                    // CORRELATION_NAMES — resolved to GGA. `XcOut::prepare`
                    // then left `vtau` empty and the meta-GGA driver panicked
                    // indexing it, on ANY density, not just a boundary one.
                    let family = ex.family().max_rung(co.family());
                    Ok(ResolvedFunctional {
                        description: format!("{} exchange + {} correlation", ex.name(), co.name()),
                        parts: vec![(1.0, Arc::from(ex)), (1.0, Arc::from(co))],
                        hybrid: HybridParams::PURE,
                        family,
                        required_nonlocal: None,
                        vv10_parameters: None,
                    })
                }
                (Some(ex), None) if nc == "none" => Ok(ResolvedFunctional {
                    description: format!("{} exchange only", ex.name()),
                    family: ex.family(),
                    parts: vec![(1.0, Arc::from(ex))],
                    hybrid: HybridParams::PURE,
                    required_nonlocal: None,
                    vv10_parameters: None,
                }),
                _ => {
                    // XC-integrated functional specified component-wise:
                    // accept iff both names are the same combined entry.
                    if combined(&nx).is_some() || combined(&nc).is_some() {
                        if nx == nc {
                            return combined(&nx).ok_or_else(|| {
                                DftError::FunctionalResolution(format!(
                                    "internal: combined lookup lost `{x}`"
                                ))
                            });
                        }
                        return Err(DftError::FunctionalResolution(format!(
                            "`{x}` / `{c}`: an exchange-correlation-integrated functional \
                             cannot be mixed component-wise; specify it as --xc, or use \
                             identical --x and --c"
                        )));
                    }
                    // Built from the registry tables rather than typed out:
                    // the hand-written list had gone stale (it omitted
                    // revpbe/tpss/r2scan/hjs-pbe-x and p86/pw91), which is
                    // the failure mode a literal list always eventually has.
                    Err(DftError::FunctionalResolution(format!(
                        "unknown exchange `{x}` or correlation `{c}`\n  exchange: {}, hf\n  \
                         correlation: {}, none\n(`--list-functionals` prints the whole registry)",
                        EXCHANGE_NAMES.join(", "),
                        CORRELATION_NAMES.join(", ")
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_and_split_resolution() {
        assert!(resolve(&FunctionalSpec::Combined("SVWN5".into())).is_ok());
        assert!(resolve(&FunctionalSpec::Combined("lda".into())).is_ok());
        let split = resolve(&FunctionalSpec::Split {
            x: "slater".into(),
            c: "vwn5".into(),
        })
        .unwrap();
        assert_eq!(split.parts.len(), 2);
        assert_eq!(split.hybrid, HybridParams::PURE);
        assert!(resolve(&FunctionalSpec::Combined("nonsense".into())).is_err());
    }

    #[test]
    fn hf_special_cases() {
        let hf = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
        assert!(hf.parts.is_empty());
        assert_eq!(hf.hybrid, HybridParams::HF);
        let hf2 = resolve(&FunctionalSpec::Split {
            x: "hf".into(),
            c: "none".into(),
        })
        .unwrap();
        assert_eq!(hf2.hybrid, HybridParams::HF);
    }

    #[test]
    fn nonlocal_requirements_are_typed_and_independent_of_descriptions() {
        for (name, expected) in [
            ("wb97x-v", RequiredNonlocalTerm::Vv10),
            ("wb97m-v", RequiredNonlocalTerm::Vv10),
            ("b97m-v", RequiredNonlocalTerm::Vv10),
            ("wb97x-d", RequiredNonlocalTerm::Chg),
            ("wb97x-d3", RequiredNonlocalTerm::D3Zero),
        ] {
            let mut functional = resolve(&FunctionalSpec::Combined(name.to_string())).unwrap();
            assert_eq!(functional.required_nonlocal_kind(), Some(expected));
            assert_eq!(functional.required_nonlocal_term(), Some(expected.name()));
            functional.description = "description deliberately replaced".to_string();
            assert_eq!(functional.required_nonlocal_kind(), Some(expected));
            if functional.hybrid.clr != 0.0 {
                assert_eq!(
                    functional
                        .with_omega(functional.hybrid.omega * 1.01)
                        .unwrap()
                        .required_nonlocal_kind(),
                    Some(expected)
                );
            }
        }

        let misleading = ResolvedFunctional::custom(
            "requires the VV10 NLC term",
            Vec::new(),
            HybridParams::PURE,
            Family::Lda,
        );
        assert_eq!(misleading.required_nonlocal_kind(), None);
    }

    /// Every canonical name in the enumerable mirror must resolve, and the
    /// three lists must be free of duplicates (a duplicate would silently
    /// hide a table arm from the coverage census).
    #[test]
    fn every_canonical_name_resolves() {
        for &name in COMBINED_NAMES {
            let f = resolve(&FunctionalSpec::Combined(name.into()))
                .unwrap_or_else(|e| panic!("combined `{name}` does not resolve: {e}"));
            assert!(!f.description.is_empty(), "{name} has no description");
        }
        for &name in EXCHANGE_NAMES {
            resolve(&FunctionalSpec::Split {
                x: name.into(),
                c: "none".into(),
            })
            .unwrap_or_else(|e| panic!("exchange `{name}` does not resolve: {e}"));
        }
        // Correlation entries have no correlation-only spec, so pair each
        // with Slater exchange.
        for &name in CORRELATION_NAMES {
            resolve(&FunctionalSpec::Split {
                x: "slater".into(),
                c: name.into(),
            })
            .unwrap_or_else(|e| panic!("correlation `{name}` does not resolve: {e}"));
        }
        for list in [EXCHANGE_NAMES, CORRELATION_NAMES, COMBINED_NAMES] {
            let mut seen: Vec<String> = list.iter().map(|n| normalize(n)).collect();
            let before = seen.len();
            seen.sort();
            seen.dedup();
            assert_eq!(before, seen.len(), "duplicate canonical name in {list:?}");
        }
    }

    /// The XC-integrated rule: identical X=C continues as combined; a
    /// mismatch is a hard, early error.
    #[test]
    fn xc_only_component_rule() {
        let same = resolve(&FunctionalSpec::Split {
            x: "svwn5".into(),
            c: "svwn5".into(),
        })
        .unwrap();
        assert_eq!(same.parts.len(), 2);
        let err = resolve(&FunctionalSpec::Split {
            x: "svwn5".into(),
            c: "vwn5".into(),
        });
        assert!(err.is_err());
        let msg = format!("{}", err.err().unwrap());
        assert!(msg.contains("component-wise"), "{msg}");
        // The real XC-integrated family: identical split continues …
        let same = resolve(&FunctionalSpec::Split {
            x: "hcth407".into(),
            c: "HCTH407".into(),
        })
        .unwrap();
        assert_eq!(same.parts.len(), 1);
        // … mismatched split is a hard pre-SCF error.
        assert!(resolve(&FunctionalSpec::Split {
            x: "hcth407".into(),
            c: "lyp".into(),
        })
        .is_err());
    }
}
