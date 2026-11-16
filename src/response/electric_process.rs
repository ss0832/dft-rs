// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Physical nonlinear optical frequency presets; no independent numerics.
use super::{
    dynamic_electric_tensor, DynamicElectricTensor, Frequency, ResponseReport, ResponseSolverConfig,
};
use crate::{
    basis::MolBasis,
    grid::MolecularGrid,
    scf::{ScfOptions, ScfResult},
    system::Molecule,
    xc::ResolvedFunctional,
    DftError, Result,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElectricProcess {
    SecondHarmonic,
    SumFrequency,
    DifferenceFrequency,
    OpticalRectification,
    ElectroOptic,
    ThirdHarmonic,
    FieldInducedSecondHarmonic,
    FieldInducedOpticalRectification,
    TwoPhotonAbsorption,
    OpticalKerr,
    DcKerr,
    FourWaveMixing,
}
impl ElectricProcess {
    pub fn name(self) -> &'static str {
        match self {
            Self::SecondHarmonic => "shg",
            Self::SumFrequency => "sfg",
            Self::DifferenceFrequency => "dfg",
            Self::OpticalRectification => "optical-rectification",
            Self::ElectroOptic => "electro-optic",
            Self::ThirdHarmonic => "thg",
            Self::FieldInducedSecondHarmonic => "efish",
            Self::FieldInducedOpticalRectification => "field-induced-rectification",
            Self::TwoPhotonAbsorption => "two-photon-absorption",
            Self::OpticalKerr => "optical-kerr",
            Self::DcKerr => "dc-kerr",
            Self::FourWaveMixing => "four-wave-mixing",
        }
    }
    pub fn order(self) -> usize {
        match self {
            Self::SecondHarmonic
            | Self::SumFrequency
            | Self::DifferenceFrequency
            | Self::OpticalRectification
            | Self::ElectroOptic => 2,
            _ => 3,
        }
    }
    pub fn frequency_count(self) -> usize {
        match self {
            Self::SumFrequency | Self::DifferenceFrequency => 2,
            Self::FourWaveMixing => 3,
            _ => 1,
        }
    }
    /// Nonnegative optical magnitudes; negative-frequency legs retain positive
    /// incoming damping. A prescribed DC field leg has exactly zero damping.
    pub fn inputs(self, frequencies: &[f64], damping: f64) -> Result<Vec<Frequency>> {
        if frequencies.len() != self.frequency_count()
            || frequencies.iter().any(|w| !w.is_finite() || *w < 0.0)
            || !damping.is_finite()
            || damping < 0.0
        {
            return Err(DftError::InvalidInput(format!(
                "{} requires {} finite nonnegative optical frequencies and nonnegative damping",
                self.name(),
                self.frequency_count()
            )));
        }
        let optical = |omega| Frequency::Retarded { omega, damping };
        let w = frequencies[0];
        Ok(match self {
            Self::SecondHarmonic => vec![optical(w), optical(w)],
            Self::SumFrequency => vec![optical(w), optical(frequencies[1])],
            Self::DifferenceFrequency => vec![optical(w), optical(-frequencies[1])],
            Self::OpticalRectification => vec![optical(w), optical(-w)],
            Self::ElectroOptic => vec![optical(w), Frequency::Static],
            Self::ThirdHarmonic => vec![optical(w), optical(w), optical(w)],
            Self::FieldInducedSecondHarmonic => vec![optical(w), optical(w), Frequency::Static],
            Self::FieldInducedOpticalRectification => {
                vec![optical(w), optical(-w), Frequency::Static]
            }
            Self::TwoPhotonAbsorption => vec![optical(w), optical(w), optical(-w)],
            Self::OpticalKerr => vec![optical(w), optical(-w), optical(w)],
            Self::DcKerr => vec![optical(w), Frequency::Static, Frequency::Static],
            Self::FourWaveMixing => vec![
                optical(w),
                optical(frequencies[1]),
                optical(-frequencies[2]),
            ],
        })
    }
}
impl std::str::FromStr for ElectricProcess {
    type Err = DftError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "shg" | "second-harmonic" => Ok(Self::SecondHarmonic),
            "sfg" | "sum-frequency" => Ok(Self::SumFrequency),
            "dfg" | "difference-frequency" => Ok(Self::DifferenceFrequency),
            "optical-rectification" | "rectification" => Ok(Self::OpticalRectification),
            "electro-optic" | "pockels" => Ok(Self::ElectroOptic),
            "thg" | "third-harmonic" => Ok(Self::ThirdHarmonic),
            "efish" | "field-induced-second-harmonic" => Ok(Self::FieldInducedSecondHarmonic),
            "field-induced-rectification" | "efior" => Ok(Self::FieldInducedOpticalRectification),
            "two-photon-absorption" | "tpa" => Ok(Self::TwoPhotonAbsorption),
            "optical-kerr" | "kerr" | "intensity-dependent-refraction" => Ok(Self::OpticalKerr),
            "dc-kerr" | "quadratic-electro-optic" => Ok(Self::DcKerr),
            "four-wave-mixing" | "fwm" => Ok(Self::FourWaveMixing),
            _ => Err(DftError::InvalidInput(format!(
                "unknown electric process: {s}"
            ))),
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn dynamic_electric_process(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    process: ElectricProcess,
    frequencies: &[f64],
    damping: f64,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<DynamicElectricTensor>> {
    let inputs = process.inputs(frequencies, damping)?;
    let mut report =
        dynamic_electric_tensor(mol, mb, functional, grid, scf, opts, &inputs, config)?;
    report.value.process = Some(process);
    Ok(report)
}
