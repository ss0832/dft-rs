// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::nuclear_response;
use super::{Controls, Value};
use crate::driver::nmr::{SpectrumCouplings, SpectrumRequest};
use crate::integrals::dirac::{GaussianMagnetization, GaussianNucleus};
use crate::math::Vec3;
use crate::periodic::spinor_scf::SpinorScfReport;
use crate::properties::two_component::MolecularTwoComponent;
use crate::response::krylov::ResponseSolverConfig;
use crate::{DftError, Result};

pub(super) struct Request {
    shielding: Vec<(usize, f64)>,
    solver: ResponseSolverConfig,
    origin: Vec3,
    spectrum: Option<SpectrumRequest>,
}

impl Request {
    pub(super) fn new(
        controls: &Controls,
        count: usize,
        shielding: bool,
        spectrum: bool,
    ) -> Result<Self> {
        if !matches!(controls.str("gauge", "london")?.as_str(), "london" | "giao") {
            return Err(DftError::Unsupported {
                feature: "four-component magnetic response gauge".into(),
                missing: "the covariant London-RMB basis selected with gauge=london".into(),
            });
        }
        if controls.contains("nuclear_magnetic_moments") {
            return Err(DftError::Unsupported {
                feature: "four-component magnetic response with nonzero reference nuclear moments"
                    .into(),
                missing: "the uniform-field derivatives of the finite reference nuclear sources"
                    .into(),
            });
        }
        let selected = if shielding {
            let radii = nuclear_response::magnetization_radii(controls, count)?;
            let nuclei = controls
                .opt_usize_list("nuclei")?
                .unwrap_or_else(|| (0..count).collect());
            if nuclei.is_empty()
                || nuclei
                    .iter()
                    .enumerate()
                    .any(|(i, a)| *a >= count || nuclei[..i].contains(a))
            {
                return Err(DftError::InvalidInput(
                    "Dirac shielding needs distinct selected nuclei".into(),
                ));
            }
            if spectrum && !nuclei.iter().copied().eq(0..count) {
                return Err(DftError::InvalidInput(
                    "Dirac nmr-spectrum needs the shieldings of all atoms in input order".into(),
                ));
            }
            nuclei.into_iter().map(|a| (a, radii[a])).collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            shielding: selected,
            solver: crate::driver::vibrational::response_config(controls)?,
            origin: controls.opt_vec3("gauge_origin")?.unwrap_or_default(),
            spectrum: spectrum
                .then(|| SpectrumRequest::new("nmr-spectrum", controls))
                .transpose()?,
        })
    }

    pub(super) fn append(
        self,
        report: &mut Value,
        units: &mut Value,
        nuclei: &[GaussianNucleus<f64>],
        backend: &MolecularTwoComponent<'_>,
        state: &SpinorScfReport,
    ) -> Result<()> {
        let sources = self
            .shielding
            .iter()
            .map(|&(index, rms_radius)| GaussianMagnetization {
                center: nuclei[index].center,
                rms_radius,
            })
            .collect::<Vec<_>>();
        let result = backend.dirac_magnetic_response(state, nuclei, &sources, &self.solver)?;
        let isotropic = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        report.set("gauge", "london");
        report.set("gauge_origin", self.origin.to_array());
        units.set("gauge_origin", "bohr");
        report.set("magnetic_dipole", result.moment);
        units.set("magnetic_dipole", "atomic units of magnetic dipole moment");
        let isotropic_shieldings = if self.shielding.is_empty() {
            let tensor = result.tensors.total[0];
            report.set("magnetizability", tensor);
            report.set("isotropic", isotropic(&tensor));
            report.set(
                "isotropic_cgs_ppm",
                isotropic(&tensor) * crate::driver::magnetizability::ATOMIC_UNIT_IN_CGS_PPM,
            );
            report.set(
                "magnetizability_explicit_sector",
                result.tensors.explicit[0],
            );
            report.set(
                "magnetizability_orbital_relaxation",
                result.tensors.relaxed[0],
            );
            for key in [
                "magnetizability",
                "isotropic",
                "magnetizability_explicit_sector",
                "magnetizability_orbital_relaxation",
            ] {
                units.set(key, "e^2*bohr^2/m_e");
            }
            units.set("isotropic_cgs_ppm", "1e-6 cm^3/mol");
            report.set("axis_convention", "xi[i][j] = -d2E/(dB_i dB_j)");
            Vec::new()
        } else {
            let ppm = |t: Vec<[[f64; 3]; 3]>| {
                t.into_iter()
                    .map(|t| t.map(|row| row.map(|x| x * 1e6)))
                    .collect::<Vec<_>>()
            };
            let tensor = ppm(result.tensors.total);
            report.set(
                "shielding_nuclei",
                self.shielding.iter().map(|x| x.0).collect::<Vec<_>>(),
            );
            units.set("shielding_nuclei", "atom index");
            report.set(
                "nuclear_magnetic_rms_radii",
                self.shielding.iter().map(|x| x.1).collect::<Vec<_>>(),
            );
            units.set("nuclear_magnetic_rms_radii", "bohr");
            let isotropic_shieldings = tensor.iter().map(isotropic).collect::<Vec<_>>();
            report.set("isotropic_shieldings", isotropic_shieldings.clone());
            report.set("shielding_tensors", tensor);
            report.set(
                "shielding_explicit_sector_tensors",
                ppm(result.tensors.explicit),
            );
            report.set(
                "shielding_orbital_relaxation_tensors",
                ppm(result.tensors.relaxed),
            );
            for key in [
                "isotropic_shieldings",
                "shielding_tensors",
                "shielding_explicit_sector_tensors",
                "shielding_orbital_relaxation_tensors",
            ] {
                units.set(key, "ppm");
            }
            report.set(
                "axis_convention",
                "sigma[nucleus][i][j] = d2E/(dmu_i dB_j); electronic shielding",
            );
            isotropic_shieldings
        };
        let residual = result.residuals.iter().copied().fold(0.0, f64::max);
        report.set("response_converged", true);
        report.set("response_residual", residual);
        report.set("response_residuals", result.residuals);
        report.set(
            "magnetic_response_reciprocity_defect",
            result.reciprocity_defect,
        );
        report.set(
            "magnetic_response_sector_metric_defect",
            result.metric_defect,
        );
        report.set(
            "magnetic_response_sector_operator_residual",
            result.sector_residual,
        );
        report.set(
            "magnetic_response_reference_energy_defect",
            result.reference_energy_defect,
        );
        report.set(
            "magnetic_response_reference_fock_defect",
            result.reference_fock_defect,
        );
        report.set("provenance", "analytic");
        report.set("magnetic_response_model", "Stationary molecular Dirac derivatives in the London-RMB basis with the selected two-electron interaction. Explicit terms include the moving electronic sector, basis and operator derivatives. Orbital-relaxation terms come from the positive-sector CPKS solve.");
        for key in [
            "response_residual",
            "response_residuals",
            "magnetic_response_sector_metric_defect",
        ] {
            units.set(key, "dimensionless");
        }
        for key in [
            "magnetic_response_sector_operator_residual",
            "magnetic_response_reference_energy_defect",
            "magnetic_response_reference_fock_defect",
        ] {
            units.set(key, "hartree");
        }
        units.set("magnetic_response_reciprocity_defect", "atomic units");
        if let Some(spectrum) = self.spectrum {
            let numbers: Vec<_> = nuclei.iter().map(|n| n.charge as usize).collect();
            let radii: Vec<_> = self.shielding.iter().map(|&(_, r)| r).collect();
            let mut spectral_units = Vec::new();
            let mut coupling_diagnostics = None;
            spectrum.append(
                &numbers,
                &isotropic_shieldings,
                report,
                &mut spectral_units,
                &self.solver,
                |active| {
                    if active.len() < 2 {
                        return Ok(SpectrumCouplings::Dirac {
                            pairs: Vec::new(),
                            tensors: Vec::new(),
                            residual: 0.0,
                        });
                    }
                    let value = nuclear_response::evaluate(
                        nuclei,
                        &radii,
                        backend,
                        state,
                        active,
                        &self.solver,
                    )?;
                    let residual = value.response.residuals.iter().copied().fold(0.0, f64::max);
                    coupling_diagnostics = Some(value.response);
                    Ok(SpectrumCouplings::Dirac {
                        pairs: value.pairs,
                        tensors: value.tensor,
                        residual,
                    })
                },
            )?;
            for (key, unit) in spectral_units {
                units.set(key, unit);
            }
            if let Some(response) = coupling_diagnostics {
                report.set("coupling_response_residuals", response.residuals);
                units.set("coupling_response_residuals", "dimensionless");
                report.set(
                    "coupling_source_reciprocity_defect",
                    response.reciprocity_defect,
                );
                units.set("coupling_source_reciprocity_defect", "atomic units");
                report.set(
                    "coupling_sector_operator_residual",
                    response.projector_residual,
                );
                units.set("coupling_sector_operator_residual", "hartree");
            }
        }
        Ok(())
    }
}
