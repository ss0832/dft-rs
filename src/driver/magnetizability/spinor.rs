// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Static orbital magnetizability of the complex spinor reference at a
//! specified common gauge origin.

use super::{diamagnetic_tensor, nuclear_charge_origin, ATOMIC_UNIT_IN_CGS_PPM};
use crate::dfpt::orbitals::exact_spin_blocks;
use crate::driver::{method::MolecularSetup, vibrational::response_config, Controls, Value};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::{
    spinor_scf::{SpinorEnergyBackend, SpinorScfReport},
    BlochMatrix,
};
use crate::properties::magnetic::spinor_orbital_field_sources;
use crate::response::spinor::{spinor_operator_response_with_kernel, Interaction};
use crate::response::{complex::Complex, nuclear::spinor_one_body_response, ResponseSolverConfig};
use crate::{DftError, Result};

pub(in crate::driver) struct Request {
    origin: Option<Vec3>,
    config: ResponseSolverConfig,
}

impl Request {
    pub(in crate::driver) fn parse(controls: &Controls, correlated: bool) -> Result<Self> {
        let gauge = controls.opt_str("gauge")?.ok_or_else(|| {
            DftError::InvalidInput(
                "GKS orbital magnetizability requires the explicit gauge=common-origin selector"
                    .into(),
            )
        })?;
        if gauge != "common-origin" {
            return Err(DftError::Unsupported {
                feature: format!("GKS magnetizability with gauge={gauge}"),
                missing: "London overlap transport, spinor integral derivatives and the phase-dependent XC response".into(),
            });
        }
        if correlated {
            return Err(DftError::Unsupported {
                feature: "correlated GKS orbital magnetizability".into(),
                missing: "the magnetic spinor PT2 Lagrangian".into(),
            });
        }
        if controls.contains("gks_operator_real") || controls.contains("gks_operator_imaginary") {
            return Err(DftError::Unsupported {
                feature: "GKS orbital magnetizability with an additional spinor operator".into(),
                missing: "the magnetic field derivatives of the additional operator".into(),
            });
        }
        Ok(Self {
            origin: controls.opt_vec3("gauge_origin")?,
            config: response_config(controls)?,
        })
    }

    pub(in crate::driver) fn append(
        self,
        report: &mut Value,
        setup: &MolecularSetup,
        reference: &SpinorScfReport,
        operator: &BlochMatrix,
        charge_density: &Matrix,
        backend: &impl SpinorEnergyBackend,
    ) -> Result<()> {
        let origin = match self.origin {
            Some(origin) => origin,
            None => nuclear_charge_origin(&setup.molecule)?,
        };
        let sources = spinor_orbital_field_sources(&setup.basis, origin);
        let (values, reference_fock_defect) = if exact_orbital_sector(setup, reference) {
            // With real, spin-diagonal orbitals, the linear magnetic source
            // preserves the same-spin antisymmetric density sector. Those
            // densities give zero rho and sigma derivatives; only exchange
            // contributes. Project the kernel action in and out so dense
            // and Krylov methods use the same invariant sector.
            let pairs = Interaction::at_spinor_reference(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &setup.options,
                &reference.density[0],
            )?;
            let kernel = |density: &BlochMatrix| {
                let induced = pairs.pair_response(&odd_spin_diagonal(density))?;
                Ok(odd_spin_diagonal(&induced))
            };
            let bands = &reference.bands[0];
            let occupied = reference.occupations[0]
                .iter()
                .take_while(|&&f| f == 1.0)
                .count();
            let verified = backend.evaluate(&reference.density)?;
            let expected = &reference.fields.fock[0];
            let actual = &verified.fock[0];
            let defect = actual
                .re
                .max_abs_diff(&expected.re)
                .max(actual.im.max_abs_diff(&expected.im));
            if defect > self.config.acceptance() {
                return Err(DftError::ResponseNotConverged {
                    iterations: 0,
                    residual: defect,
                });
            }
            let values = spinor_operator_response_with_kernel(
                &bands.coefficients,
                &bands.energies,
                occupied,
                &kernel,
                &sources,
                &[Complex::ZERO],
                &self.config,
            )?;
            (values, defect)
        } else {
            let response = spinor_one_body_response(
                &setup.molecule,
                &setup.basis,
                setup.grid.as_ref(),
                &setup.functional,
                reference,
                &setup.options,
                operator,
                &sources,
                &[Complex::ZERO],
                &self.config,
            )?;
            (response.values, response.reference_fock_defect)
        };
        let static_response = &values[0];
        let imaginary_defect = static_response
            .susceptibility
            .iter()
            .flatten()
            .map(|v| v.im.abs())
            .fold(0.0_f64, f64::max);
        if imaginary_defect > self.config.acceptance() {
            return Err(DftError::ResponseNotConverged {
                iterations: static_response
                    .iterations
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                residual: imaginary_defect,
            });
        }
        let diamagnetic = diamagnetic_tensor(&setup.basis, origin, charge_density);
        let paramagnetic: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| -static_response.susceptibility[a][b].re)
        });
        let tensor: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| diamagnetic[a][b] + paramagnetic[a][b])
        });
        let isotropic = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
        let asymmetry = [(0, 1), (0, 2), (1, 2)]
            .map(|(a, b)| (tensor[a][b] - tensor[b][a]).abs())
            .into_iter()
            .fold(0.0_f64, f64::max);

        report.set("model", "Static orbital magnetizability of the fully coupled complex spinor reference at a common gauge origin.");
        report.set("gauge", "common-origin");
        report.set("gauge_origin", origin);
        report.set("magnetizability", tensor);
        report.set("diamagnetic", diamagnetic);
        report.set("paramagnetic", paramagnetic);
        report.set("isotropic", isotropic);
        report.set("isotropic_cgs_ppm", isotropic * ATOMIC_UNIT_IN_CGS_PPM);
        report.set("raw_asymmetry", asymmetry);
        report.set("response_converged", true);
        report.set("response_residuals", static_response.residuals.clone());
        report.set("response_iterations", static_response.iterations.clone());
        report.set("reference_fock_defect", reference_fock_defect);
        report.set("magnetizability_imaginary_defect", imaginary_defect);
        report.set("provenance", "analytic");
        report.set("warnings", vec!["Common gauge origin: the orbital tensor depends on gauge_origin in a finite basis."]);
        let mut labels = report.get("units").cloned().expect("GKS energy units");
        for (key, unit) in [
            ("gauge_origin", "bohr"),
            (
                "magnetizability",
                "e^2*bohr^2/m_e, xi_ab = -d2E/(dB_a dB_b)",
            ),
            ("diamagnetic", "e^2*bohr^2/m_e"),
            ("paramagnetic", "e^2*bohr^2/m_e"),
            ("isotropic", "e^2*bohr^2/m_e"),
            ("isotropic_cgs_ppm", "1e-6 cm^3/mol"),
            ("raw_asymmetry", "e^2*bohr^2/m_e"),
            ("response_residuals", "dimensionless"),
            ("response_iterations", "integer"),
            ("reference_fock_defect", "hartree"),
            ("magnetizability_imaginary_defect", "e^2*bohr^2/m_e"),
        ] {
            labels.set(key, unit);
        }
        report.set("units", labels);
        Ok(())
    }
}

fn exact_orbital_sector(setup: &MolecularSetup, reference: &SpinorScfReport) -> bool {
    let options = &setup.options;
    setup.functional.family != crate::xc::Family::MetaGga
        && options.vv10.is_none()
        && options.skala.is_none()
        && options.pcm.is_none()
        && options.density_fit.is_none()
        && matches!(options.kernel, crate::integrals::CoulombKernel::Coulomb)
        && options.electronic_temperature == 0.0
        && reference.occupations[0]
            .iter()
            .all(|f| *f == 0.0 || *f == 1.0)
        && reference.bands[0]
            .coefficients
            .im
            .as_slice()
            .iter()
            .all(|&v| v == 0.0)
        && [&reference.density[0], &reference.fields.fock[0]]
            .into_iter()
            .all(|m| m.im.as_slice().iter().all(|&v| v == 0.0) && exact_spin_blocks(m).is_some())
}

/// Project complex orbital-response amplitudes onto the two same-spin
/// antisymmetric spatial blocks. Remove both longitudinal even and transverse
/// components before using a kernel valid only on the odd sector.
fn odd_spin_diagonal(density: &BlochMatrix) -> BlochMatrix {
    let mut out = BlochMatrix::zeros(density.rows(), density.cols());
    let n = density.rows() / 2;
    for offset in [0, n] {
        for i in offset..offset + n {
            for j in offset..i {
                out.re[(i, j)] = 0.5 * (density.re[(i, j)] - density.re[(j, i)]);
                out.re[(j, i)] = -out.re[(i, j)];
                out.im[(i, j)] = 0.5 * (density.im[(i, j)] - density.im[(j, i)]);
                out.im[(j, i)] = -out.im[(i, j)];
            }
        }
    }
    out
}
