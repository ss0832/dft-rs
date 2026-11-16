// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared full-spin Ramsey report, used by collinear and complex drivers.

use super::*;
use crate::response::spin_spin::SpinorSpinSpinCouplings;

pub(in crate::driver) fn constraint_mode(controls: &Controls) -> Result<bool> {
    match controls.str("spin_response_constraint", "free")?.as_str() {
        "fixed-spin-direction" => Ok(true),
        "free" => Ok(false),
        _ => Err(DftError::InvalidInput(
            "spin_response_constraint must be fixed-spin-direction or free".into(),
        )),
    }
}

pub(in crate::driver) fn append(
    report: &mut Value,
    result: &SpinorSpinSpinCouplings,
    nuclear_g: Option<&[f64]>,
    contact_only: bool,
    constrained: bool,
) {
    let value = &result.value;
    let raw = &result.response.values[0];
    let isotropic = |tensors: &[[[f64; 3]; 3]]| {
        tensors
            .iter()
            .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0)
            .collect::<Vec<_>>()
    };
    report.set(
        "coupling_pairs",
        value.pairs.iter().map(|&(m, n)| [m, n]).collect::<Vec<_>>(),
    );
    report.set("reduced_couplings", value.total.clone());
    report.set("reduced_isotropic", isotropic(&value.total));
    report.set("fermi_contact_tensors", value.fermi_contact.clone());
    report.set("fermi_contact_isotropic", isotropic(&value.fermi_contact));
    report.set("response_residual", value.residual);
    report.set(
        "spin_response_constraint",
        if constrained {
            "fixed-spin-direction"
        } else {
            "free"
        },
    );
    report.set("spin_response_constraint_rank", raw.constraint_rank);
    report.set(
        "spin_response_constraint_residuals",
        raw.constraint_residuals
            .iter()
            .map(|v| v.iter().map(|z| z.abs()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    report.set(
        "spin_response_discarded_projection_norms",
        raw.constraint_discarded_norms.clone(),
    );
    report.set(
        "response_reference_fock_defect",
        result.response.reference_fock_defect,
    );
    report.set(
        "response_xc_angular_history",
        result.response.angular_history.clone(),
    );
    report.set(
        "response_xc_cache_bounds",
        result.response.cache_bound_history.clone(),
    );
    report.set("axis_convention","K[pair][a][b] = d2E/(dm_M,a dm_N,b); requested nucleus order; all Cartesian/spin components solved independently; total includes both FC/SD-PSO source orders");
    report.set("model","static full complex Pauli Ramsey response of the supplied stationary reference; FC, SD, PSO and explicit DSO, including spin-orbit mixed terms; same spin-covariant semilocal/full-range-exchange Hessian as the reference; common-origin all-electron model; fixed-spin-direction allows longitudinal magnitude relaxation and excludes free-spin orientational Curie enhancement; correlated PT2 and common-orbital ROKS response are separate providers");
    report.set("provenance", "analytic");
    let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
    for (key, unit) in [
        (
            "reduced_couplings",
            "atomic units, K_MN[pair][a][b] = d2E/(dm_M,a dm_N,b)",
        ),
        ("reduced_isotropic", "atomic units"),
        ("fermi_contact_tensors", "atomic units"),
        ("fermi_contact_isotropic", "atomic units"),
        ("response_residual", "dimensionless"),
        ("spin_response_constraint_rank", "dimensionless"),
        (
            "spin_response_constraint_residuals",
            "atomic units, d<S>/dm_nuclear",
        ),
        ("spin_response_discarded_projection_norms", "dimensionless"),
        ("response_reference_fock_defect", "hartree"),
        (
            "response_xc_angular_history",
            "[dimensionless, hartree, hartree]",
        ),
        (
            "response_xc_cache_bounds",
            "[dimensionless, hartree, hartree]",
        ),
    ] {
        labels.set(key, unit);
    }
    if !contact_only {
        for (key, tensors) in [
            ("diamagnetic", &value.diamagnetic),
            ("paramagnetic", &value.paramagnetic),
            ("spin_dipole", &value.spin_dipole),
            ("spin_orbit_mixed", &result.spin_orbit_mixed),
        ] {
            let tensor = format!("{key}_tensors");
            let scalar = format!("{key}_isotropic");
            report.set(&tensor, tensors.clone());
            report.set(&scalar, isotropic(tensors));
            labels.set(&tensor, "atomic units");
            labels.set(&scalar, "atomic units");
        }
    }
    if let Some(g) = nuclear_g {
        report.set(
            "coupling_constants_hz",
            coupling_hertz(value, &value.total, g),
        );
        report.set(
            "fermi_contact_hz",
            coupling_hertz(value, &value.fermi_contact, g),
        );
        labels.set("coupling_constants_hz", "Hz, isotropic J of every pair");
        labels.set("fermi_contact_hz", "Hz");
    }
    report.set("units", labels);
}

/// Reference-independent spectrum settings with a spinor magnetic request.
pub(in crate::driver) struct Request {
    spectrum: SpectrumRequest,
    gauge: Gauge,
    origin: Vec3,
    fixed_spin: bool,
}

impl Request {
    pub(in crate::driver) fn new(id: &str, controls: &Controls) -> Result<Self> {
        Ok(Self {
            spectrum: SpectrumRequest::new(id, controls)?,
            gauge: Gauge::from_name(&controls.str("gauge", "common-origin")?)?,
            origin: controls
                .opt_vec3("gauge_origin")?
                .unwrap_or_else(Vec3::zero),
            fixed_spin: constraint_mode(controls)?,
        })
    }

    pub(in crate::driver) fn append(
        self,
        report: &mut Value,
        setup: &super::super::method::MolecularSetup,
        reference: &crate::periodic::spinor_scf::SpinorScfReport,
        operator: &crate::periodic::BlochMatrix,
        config: &crate::response::ResponseSolverConfig,
    ) -> Result<()> {
        let overlap = crate::integrals::overlap_matrix(&setup.basis);
        let constraints = if self.fixed_spin {
            crate::response::spin_spin::spin_direction_constraints(&reference.density[0], &overlap)?
        } else {
            Vec::new()
        };
        let result = crate::properties::magnetic::spinor_nmr_response(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            reference,
            &setup.options,
            operator,
            &constraints,
            self.gauge,
            self.origin,
            config,
        )?;
        let isotropic: Vec<_> = result.value.iter().map(|s| s.isotropic()).collect();
        report.set(
            "shielding_tensors",
            result.value.iter().map(|s| s.tensor).collect::<Vec<_>>(),
        );
        report.set(
            "diamagnetic_tensors",
            result
                .value
                .iter()
                .map(|s| s.diamagnetic)
                .collect::<Vec<_>>(),
        );
        report.set(
            "paramagnetic_tensors",
            result
                .value
                .iter()
                .map(|s| s.paramagnetic)
                .collect::<Vec<_>>(),
        );
        report.set("isotropic_shieldings", isotropic.clone());
        report.set(
            "diamagnetic_isotropic",
            result
                .value
                .iter()
                .map(|s| s.isotropic_parts().0)
                .collect::<Vec<_>>(),
        );
        report.set(
            "paramagnetic_isotropic",
            result
                .value
                .iter()
                .map(|s| s.isotropic_parts().1)
                .collect::<Vec<_>>(),
        );
        report.set("gauge", "common-origin");
        report.set("gauge_origin", self.origin);
        report.set("response_residuals", result.residuals.clone());
        let converged = reference.converged
            && result.residuals.iter().all(|r| *r <= config.tolerance)
            && result.reference_stationarity <= config.acceptance();
        report.set("converged", converged);
        report.set("response_converged", converged);
        report.set("model", "Full complex spinor orbital NMR shielding; common-origin minimal coupling; independent full-spin Hessian solves");
        report.set("provenance", "analytic");
        report.set(
            "warnings",
            vec![
                "Common-origin finite-basis shielding is gauge-origin dependent.",
                "Orbital shielding excludes spin-Zeeman, Curie and thermal population shifts.",
            ],
        );
        let response = result
            .spinor_response
            .as_ref()
            .expect("spinor shielding response");
        report.set(
            "response_reference_fock_defect",
            response.reference_fock_defect,
        );
        report.set(
            "response_xc_angular_history",
            response.angular_history.clone(),
        );
        let mut unit_pairs = vec![
            ("gauge_origin", "bohr"),
            ("shielding_tensors", "ppm"),
            ("diamagnetic_tensors", "ppm"),
            ("paramagnetic_tensors", "ppm"),
            ("isotropic_shieldings", "ppm"),
            ("diamagnetic_isotropic", "ppm"),
            ("paramagnetic_isotropic", "ppm"),
            ("response_residuals", "dimensionless"),
            ("response_reference_fock_defect", "hartree"),
            (
                "response_xc_angular_history",
                "[dimensionless, hartree, hartree]",
            ),
        ];
        let numbers: Vec<_> = setup.molecule.atoms.iter().map(|a| a.z as usize).collect();
        self.spectrum.append(
            &numbers,
            &isotropic,
            report,
            &mut unit_pairs,
            config,
            |active| {
                let grid = setup.integration_grid()?;
                let coupling = crate::response::spin_spin::spinor_spin_spin_couplings(
                    &setup.molecule,
                    &setup.basis,
                    &setup.functional,
                    Some(&grid),
                    reference,
                    &setup.options,
                    operator,
                    &constraints,
                    active,
                    false,
                    config,
                )?;
                Ok(super::SpectrumCouplings::Ramsey(coupling.value))
            },
        )?;
        let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
        for (key, unit) in unit_pairs {
            labels.set(key, unit);
        }
        report.set("units", labels);
        Ok(())
    }
}

/// Preserve the converged UKS state while reusing the existing full-spin
/// Ramsey provider. Shared by standalone couplings and spectrum generation.
pub(in crate::driver) fn collinear_couplings(
    setup: &super::super::method::MolecularSetup,
    reference: &crate::scf::ScfResult,
    grid: &crate::grid::MolecularGrid,
    nuclei: &[usize],
    contact_only: bool,
    fixed_spin: bool,
    config: &crate::response::ResponseSolverConfig,
) -> Result<SpinorSpinSpinCouplings> {
    let state = crate::response::spinor::embedded_collinear_spinor_reference(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        Some(grid),
        reference,
        &setup.options,
        config.acceptance(),
    )?;
    let constraints = if fixed_spin {
        crate::response::spin_spin::spin_direction_constraints(
            &state.density[0],
            &reference.overlap,
        )?
    } else {
        Vec::new()
    };
    crate::response::spin_spin::spinor_spin_spin_couplings(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        Some(grid),
        &state,
        &setup.options,
        &crate::periodic::BlochMatrix::zeros(2 * setup.basis.nao_sph, 2 * setup.basis.nao_sph),
        &constraints,
        nuclei,
        contact_only,
        config,
    )
}
