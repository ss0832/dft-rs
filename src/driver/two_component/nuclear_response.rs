// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use crate::driver::nmr::coupling_hertz_pairs;
use crate::driver::{Controls, Value};
use crate::integrals::dirac::{GaussianMagnetization, GaussianNucleus};
use crate::periodic::spinor_scf::SpinorScfReport;
use crate::properties::two_component::{MolecularTwoComponent, NuclearMagneticFields};
use crate::response::krylov::ResponseSolverConfig;
use crate::{DftError, Result};

pub(super) struct Request {
    radii: Vec<f64>,
    moments: Option<Vec<[f64; 3]>>,
    calculation: Option<Coupling>,
}

struct Coupling {
    nuclei: Vec<usize>,
    g_factors: Option<Vec<f64>>,
    solver: ResponseSolverConfig,
}

impl Request {
    pub(super) fn new(controls: &Controls, count: usize, coupling: bool) -> Result<Option<Self>> {
        let moments = controls
            .opt_rows("nuclear_magnetic_moments")?
            .map(|rows| {
                rows.into_iter()
                    .map(|row| {
                        if row.len() == 3 && row.iter().all(|x| x.is_finite()) {
                            Ok([row[0], row[1], row[2]])
                        } else {
                            Err(DftError::InvalidInput(
                                "nuclear_magnetic_moments needs finite Cartesian rows".into(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if moments.is_none() && !coupling {
            return Ok(None);
        }
        if moments.as_ref().is_some_and(|rows| rows.len() != count) {
            return Err(DftError::InvalidInput(
                "nuclear_magnetic_moments needs one row per atom".into(),
            ));
        }
        let radii = magnetization_radii(controls, count)?;
        let calculation = if coupling {
            let nuclei = controls
                .opt_usize_list("nuclei")?
                .unwrap_or_else(|| (0..count).collect());
            let g_factors = controls.opt_f64_list("nuclear_g")?;
            if nuclei.len() < 2
                || nuclei.iter().any(|&a| a >= count)
                || nuclei
                    .iter()
                    .enumerate()
                    .any(|(i, a)| nuclei[..i].contains(a))
                || g_factors
                    .as_ref()
                    .is_some_and(|g| g.len() != count || g.iter().any(|x| !x.is_finite()))
            {
                return Err(DftError::InvalidInput(
                    "four-component coupling needs distinct selected nuclei and one finite nuclear_g per atom when requested".into(),
                ));
            }
            Some(Coupling {
                nuclei,
                g_factors,
                solver: crate::driver::vibrational::response_config(controls)?,
            })
        } else {
            None
        };
        Ok(Some(Self {
            radii,
            moments,
            calculation,
        }))
    }

    pub(super) fn fields(&self) -> Option<NuclearMagneticFields<'_>> {
        self.moments
            .as_deref()
            .map(|moments| NuclearMagneticFields {
                moments,
                radii: &self.radii,
            })
    }

    pub(super) fn append(
        &self,
        report: &mut Value,
        units: &mut Value,
        geometry: &[GaussianNucleus<f64>],
        backend: &MolecularTwoComponent<'_>,
        state: &SpinorScfReport,
    ) -> Result<()> {
        report.set("nuclear_magnetic_rms_radii", self.radii.clone());
        units.set("nuclear_magnetic_rms_radii", "bohr");
        if let Some(moments) = &self.moments {
            report.set("nuclear_magnetic_moments", moments.clone());
            units.set(
                "nuclear_magnetic_moments",
                "atomic units of magnetic dipole moment",
            );
        }
        let Some(coupling) = &self.calculation else {
            return Ok(());
        };
        let values = evaluate(
            geometry,
            &self.radii,
            backend,
            state,
            &coupling.nuclei,
            &coupling.solver,
        )?;
        let CoupledNuclei {
            pairs,
            tensor,
            explicit,
            relaxed,
            response,
        } = values;
        let isotropic: Vec<_> = tensor
            .iter()
            .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0)
            .collect();
        if let Some(g) = &coupling.g_factors {
            report.set(
                "coupling_constants_hz",
                coupling_hertz_pairs(&pairs, &tensor, g),
            );
            units.set("coupling_constants_hz", "Hz");
        }
        let residual = response.residuals.iter().copied().fold(0.0, f64::max);
        report.set(
            "coupling_pairs",
            pairs.into_iter().map(|(a, b)| [a, b]).collect::<Vec<_>>(),
        );
        report.set("reduced_couplings", tensor);
        report.set("reduced_isotropic", isotropic);
        report.set("coupling_explicit_sector_tensors", explicit);
        report.set("coupling_orbital_relaxation_tensors", relaxed);
        report.set("coupling_response_residuals", response.residuals);
        report.set("response_residual", residual);
        report.set(
            "coupling_source_reciprocity_defect",
            response.reciprocity_defect,
        );
        report.set(
            "coupling_sector_metric_defect",
            response.projector_metric_defect,
        );
        report.set(
            "coupling_sector_operator_residual",
            response.projector_residual,
        );
        report.set(
            "coupling_reference_fock_defect",
            response.reference_fock_defect,
        );
        report.set("coupling_model", "indirect nuclear spin coupling of the Dirac electronic energy. Finite Gaussian magnetization sources vary the nuclear-core positive-energy sector and the variational electronic density. The discrete four-component current and charge operators use the same reference. The direct nuclear dipole interaction is excluded.");
        report.set(
            "axis_convention",
            "pair [M,N], tensor[a][b] = d2E/(dmu_M,a dmu_N,b) in atomic units",
        );
        report.set("provenance", "analytic");
        for key in [
            "reduced_couplings",
            "reduced_isotropic",
            "coupling_explicit_sector_tensors",
            "coupling_orbital_relaxation_tensors",
            "coupling_source_reciprocity_defect",
        ] {
            units.set(key, "atomic units");
        }
        for key in [
            "coupling_response_residuals",
            "response_residual",
            "coupling_sector_metric_defect",
        ] {
            units.set(key, "dimensionless");
        }
        for key in [
            "coupling_sector_operator_residual",
            "coupling_reference_fock_defect",
        ] {
            units.set(key, "hartree");
        }
        Ok(())
    }
}

pub(super) fn magnetization_radii(controls: &Controls, count: usize) -> Result<Vec<f64>> {
    let radii = controls.opt_f64_list("nuclear_magnetic_rms_radii")?.ok_or_else(|| {
        DftError::InvalidInput(
            "nuclear_magnetic_rms_radii must give the Gaussian magnetization radius of each atom in Bohr".into(),
        )
    })?;
    if radii.len() != count || radii.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(DftError::InvalidInput(
            "nuclear_magnetic_rms_radii needs one positive finite value per atom".into(),
        ));
    }
    Ok(radii)
}

pub(super) struct CoupledNuclei {
    pub(super) pairs: Vec<(usize, usize)>,
    pub(super) tensor: Vec<[[f64; 3]; 3]>,
    explicit: Vec<[[f64; 3]; 3]>,
    relaxed: Vec<[[f64; 3]; 3]>,
    pub(super) response: crate::properties::two_component::DiracCoreSourceResponse,
}

/// Contract the same relativistic electronic Hessian for both standalone
/// coupling reports and nuclear-spin Hamiltonians.
pub(super) fn evaluate(
    geometry: &[GaussianNucleus<f64>],
    radii: &[f64],
    backend: &MolecularTwoComponent<'_>,
    state: &SpinorScfReport,
    nuclei: &[usize],
    config: &ResponseSolverConfig,
) -> Result<CoupledNuclei> {
    let sources = nuclei
        .iter()
        .map(|&i| {
            backend.nuclear_magnetic_sources(GaussianMagnetization {
                center: geometry[i].center,
                rms_radius: radii[i],
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let response = backend.dirac_core_source_response(state, &sources, config)?;
    let mut pairs = Vec::new();
    let mut tensor = Vec::new();
    let mut explicit = Vec::new();
    let mut relaxed = Vec::new();
    let submatrix = |values: &[Vec<f64>], i: usize, j: usize| {
        std::array::from_fn::<_, 3, _>(|a| {
            std::array::from_fn::<_, 3, _>(|b| values[3 * i + a][3 * j + b])
        })
    };
    for (i, &atom) in nuclei.iter().enumerate() {
        for (j, &other) in nuclei.iter().enumerate().skip(i + 1) {
            pairs.push((atom, other));
            tensor.push(submatrix(&response.second, i, j));
            explicit.push(submatrix(&response.explicit, i, j));
            relaxed.push(submatrix(&response.orbital_relaxation, i, j));
        }
    }
    Ok(CoupledNuclei {
        pairs,
        tensor,
        explicit,
        relaxed,
        response,
    })
}
