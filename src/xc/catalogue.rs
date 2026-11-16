// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Discovery of the functional registry: one record per canonical name of
//! the exchange, correlation, combined and double-hybrid tables, with its
//! family, exact-exchange split and, for double hybrids, the PT2
//! coefficients and the recipe's own terms. Every record is built by
//! resolving its name, so the catalogue cannot drift from the registry.

use super::double_hybrid::{DoubleHybridFunctional, DOUBLE_HYBRID_NAMES};
use super::{
    resolve, Family, FunctionalSpec, RequiredNonlocalTerm, ResolvedFunctional, COMBINED_NAMES,
    CORRELATION_NAMES, EXCHANGE_NAMES,
};
use crate::{DftError, Result};

/// Registry table a catalogue name belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionalClass {
    /// Exchange component (`--x`, `x=`).
    Exchange,
    /// Correlation component (`--c`, `c=`).
    Correlation,
    /// Complete semilocal, hybrid or range-separated functional (`--xc`).
    Combined,
    /// Named double hybrid: a parent functional plus PT2 on its orbitals.
    DoubleHybrid,
}

impl FunctionalClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exchange => "exchange",
            Self::Correlation => "correlation",
            Self::Combined => "combined",
            Self::DoubleHybrid => "double-hybrid",
        }
    }
}

/// One catalogue record. Fractions are of the full exchange energy;
/// `omega` is the range-separation parameter in Bohr^-1 (0 when unused).
#[derive(Clone, Debug)]
pub struct FunctionalInfo {
    pub name: &'static str,
    pub class: FunctionalClass,
    pub family: Family,
    pub description: String,
    /// Global exact-exchange fraction.
    pub exact_exchange: f64,
    /// Additional long-range exact-exchange fraction.
    pub long_range_exchange: f64,
    pub omega: f64,
    /// Density or pair correction the functional requires (`vv10`, `chg`, `d3zero`).
    pub nonlocal: Option<&'static str>,
    /// Same-spin and opposite-spin PT2 coefficients of a double hybrid.
    pub pt2: Option<(f64, f64)>,
    /// Whether a double hybrid evaluates a distinct target functional on
    /// the parent orbitals (xDH).
    pub distinct_target: bool,
    /// Whether a double hybrid adds its own D3(BJ) term by default.
    pub dispersion: bool,
}

fn nonlocal_name(functional: &ResolvedFunctional) -> Option<&'static str> {
    functional.required_nonlocal_kind().map(|kind| match kind {
        RequiredNonlocalTerm::Vv10 => "vv10",
        RequiredNonlocalTerm::Chg => "chg",
        RequiredNonlocalTerm::D3Zero => "d3zero",
    })
}

fn record(name: &'static str, class: FunctionalClass, f: &ResolvedFunctional) -> FunctionalInfo {
    FunctionalInfo {
        name,
        class,
        family: f.family,
        description: f.description.clone(),
        exact_exchange: f.hybrid.cx,
        long_range_exchange: f.hybrid.clr,
        omega: f.hybrid.omega,
        nonlocal: nonlocal_name(f),
        pt2: None,
        distinct_target: false,
        dispersion: false,
    }
}

fn component_record(
    name: &'static str,
    class: FunctionalClass,
    part: &dyn super::XcFunctional,
) -> FunctionalInfo {
    FunctionalInfo {
        name,
        class,
        family: part.family(),
        description: part.name().to_string(),
        exact_exchange: 0.0,
        long_range_exchange: 0.0,
        omega: 0.0,
        nonlocal: None,
        pt2: None,
        distinct_target: false,
        dispersion: false,
    }
}

/// Every canonical registry name with its resolved metadata, in the order
/// exchange, correlation, combined, double hybrids.
pub fn catalogue() -> Result<Vec<FunctionalInfo>> {
    let mut out = Vec::new();
    let tables: [(&[&'static str], FunctionalClass); 2] = [
        (EXCHANGE_NAMES, FunctionalClass::Exchange),
        (CORRELATION_NAMES, FunctionalClass::Correlation),
    ];
    for (names, class) in tables {
        for &name in names {
            let key = super::normalize(name);
            let part = match class {
                FunctionalClass::Exchange => super::exchange(&key),
                _ => super::correlation(&key),
            }
            .ok_or_else(|| {
                DftError::InvalidInput(format!("{} `{name}` is not resolvable", class.as_str()))
            })?;
            out.push(component_record(name, class, part.as_ref()));
        }
    }
    for &name in COMBINED_NAMES {
        let f = resolve(&FunctionalSpec::Combined(name.into()))?;
        out.push(record(name, FunctionalClass::Combined, &f));
    }
    for &name in DOUBLE_HYBRID_NAMES {
        let recipe = DoubleHybridFunctional::named(name).ok_or_else(|| {
            DftError::InvalidInput(format!("double hybrid `{name}` is not resolvable"))
        })?;
        let energy = recipe.target.as_ref().unwrap_or(&recipe.parent);
        let mut info = record(name, FunctionalClass::DoubleHybrid, energy);
        info.description = match &recipe.target {
            Some(target) => format!(
                "{}; orbitals from {}",
                target.description, recipe.parent.description
            ),
            None => recipe.parent.description.clone(),
        };
        if info.nonlocal.is_none()
            && (recipe.reference_vv10.is_some() || recipe.target_vv10.is_some())
        {
            info.nonlocal = Some("vv10");
        }
        info.pt2 = Some((
            recipe.correlation.same_spin,
            recipe.correlation.opposite_spin,
        ));
        info.distinct_target = recipe.target.is_some();
        info.dispersion = recipe.dispersion.is_some();
        out.push(info);
    }
    Ok(out)
}

/// Family name as the catalogue reports it.
pub fn family_name(family: Family) -> &'static str {
    match family {
        Family::Lda => "lda",
        Family::Gga => "gga",
        Family::MetaGga => "meta-gga",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registry_name_has_one_catalogue_record() {
        let records = catalogue().unwrap();
        let total = EXCHANGE_NAMES.len()
            + CORRELATION_NAMES.len()
            + COMBINED_NAMES.len()
            + DOUBLE_HYBRID_NAMES.len();
        assert_eq!(records.len(), total);
        let b2plyp = records.iter().find(|r| r.name == "b2plyp").unwrap();
        assert_eq!(b2plyp.class, FunctionalClass::DoubleHybrid);
        assert_eq!(b2plyp.pt2, Some((0.27, 0.27)));
        assert_eq!(b2plyp.exact_exchange, 0.53);
        let xyg3 = records.iter().find(|r| r.name == "xyg3").unwrap();
        assert!(xyg3.distinct_target);
        assert_eq!(xyg3.exact_exchange, 0.8033);
        let pbe0 = records
            .iter()
            .find(|r| r.name == "pbe0" && r.class == FunctionalClass::Combined)
            .unwrap();
        assert_eq!((pbe0.exact_exchange, pbe0.pt2), (0.25, None));
    }
}
