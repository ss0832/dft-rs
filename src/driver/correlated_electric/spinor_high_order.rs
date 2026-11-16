// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static electric energy derivatives on a stationary complex spinor reference.

use super::spinor::scalar_sources;
use crate::driver::correlated_high_order::{electric_coefficients_tensor, Diagnostics};
use crate::driver::double_hybrid::{append_spinor_energy, SpinorModel};
use crate::driver::{method::MolecularSetup, reference_electric::append_energy_tensor, Value};
use crate::integrals::london::ComplexMatrix;
use crate::math::Vec3;
use crate::periodic::{spinor_scf::SpinorScfReport, BlochMatrix};
use crate::response::electric_tensor::tensor_contractions;
use crate::response::{nuclear::SpinorCorrelatedSolver, ResponseSolverConfig};
use crate::{DftError, Result};

pub(in crate::driver) fn append(
    report: &mut Value,
    setup: &MolecularSetup,
    scf: &SpinorScfReport,
    fixed_operator: &BlochMatrix,
    model: &SpinorModel,
    rank: usize,
    config: &ResponseSolverConfig,
) -> Result<()> {
    let solver = SpinorCorrelatedSolver::new(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        model.target.as_ref(),
        scf,
        &setup.options,
        fixed_operator,
        rank,
        model.coefficient,
        model.memory_mb,
        config,
    )?;
    let sources = scalar_sources(crate::integrals::dipole_matrices(
        &setup.basis,
        Vec3::zero(),
    ));
    let dimension = sources[0].rows();
    let mut core = vec![ComplexMatrix::zeros(dimension, dimension); rank + 1];
    let mut diagnostics = Diagnostics::default();
    let mut base_energy = None;
    let mut residuals = Vec::new();
    let mut reference_defect = 0.0_f64;
    let assembled = electric_coefficients_tensor(rank, 6, |direction, order| {
        let mut source = BlochMatrix::zeros(dimension, dimension);
        for (operator, &component) in sources.iter().zip(direction) {
            source.add_scaled(operator, component);
        }
        core[1] = ComplexMatrix {
            re: source.re,
            im: source.im,
        };
        // The electric perturbation is +E.r for the electron. The nuclear
        // dipole term is linear in E and contributes no rank-3 or rank-4 terms.
        let value = solver.solve(&core)?;
        let target = solver.target_correction(&core, &value.parent)?;
        let target_coefficient = target.as_ref().map_or(0.0, |t| t[order]);
        base_energy.get_or_insert_with(|| {
            (
                value.pairs.energy[0].re,
                target.as_ref().map_or(0.0, |t| t[0]),
            )
        });
        diagnostics.spinor_field(&value);
        let path = &value.parent.parent;
        residuals.extend_from_slice(&path.residuals);
        reference_defect = reference_defect.max(path.reference_defect);
        let parent = path.energy[order];
        let pair = value.pairs.energy[order].re;
        // The response tensor is minus the electric derivative of the energy.
        // Generalized PT2 scales the full spin sum by one common coefficient.
        Ok(vec![
            -value.energy[order] - target_coefficient,
            -parent,
            -parent - target_coefficient,
            -target_coefficient,
            -pair,
            -model.coefficient * pair,
        ])
    })?;
    let (pair_energy, target_energy) = base_energy.expect("at least one electric path");
    append_spinor_energy(report, model, pair_energy, target_energy);

    let mut relative_error = 0.0_f64;
    for (tensor, roundoff) in assembled.tensors.iter().zip(&assembled.roundoff) {
        let c = tensor_contractions(rank - 1, tensor);
        let scale = tensor.iter().fold(1.0_f64, |bound, &v| bound.max(v.abs()));
        relative_error = relative_error.max(
            c.intrinsic_permutation_defect
                .max(c.energy_permutation_defect)
                .max(roundoff.iter().copied().fold(0.0, f64::max))
                / scale,
        );
    }
    if !relative_error.is_finite() || relative_error > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: assembled.evaluations,
            residual: relative_error,
        });
    }

    report.set("shape", vec![3usize; rank]);
    report.set("assembly", "energy-polarization");
    report.set("provenance", "analytic");
    report.set("axis_convention", "row-major Cartesian dipole and field axes; clamped nuclei; minus the actual static energy derivatives");
    report.set("interchange_diagnostics", "scalar energy polarization reuses equivalent directional coefficients; observed-leg interchange is not measured");
    report.set(
        "reference_field",
        setup
            .options
            .external_field
            .map_or([0.0; 3], |v| v.to_array()),
    );
    report.set("response_reference_fock_defect", reference_defect);
    report.set("residuals", residuals);
    report.set("model", "Static noncollinear double-hybrid hyperpolarizability from the relaxed parent orbitals and generalized PT2 pair amplitudes. A distinct target is evaluated on the full parent density series.");
    let mut labels = report
        .get("units")
        .cloned()
        .expect("electric response units");
    labels.set("reference_field", "hartree/e/bohr");
    labels.set("response_reference_fock_defect", "hartree");
    labels.set("residuals", "dimensionless");
    labels.set("interchange_diagnostics", "text");
    let mut tensors = assembled.tensors.into_iter();
    let total = tensors.next().expect("total electric tensor");
    let contractions = tensor_contractions(rank - 1, &total);
    let unit = append_energy_tensor(
        report,
        rank,
        total,
        assembled
            .roundoff
            .into_iter()
            .next()
            .expect("total tensor roundoff"),
        assembled.evaluations,
        contractions,
        labels,
    );
    let mut labels = report.get("units").cloned().expect("electric tensor units");
    for (key, tensor) in [
        "parent_tensor",
        "target_tensor",
        "target_tensor_correction",
        "generalized_pt2_tensor",
        "pt2_tensor",
    ]
    .into_iter()
    .zip(tensors)
    {
        report.set(key, tensor);
        labels.set(key, unit);
    }
    report.set("units", labels);
    diagnostics.append(report, config);
    Ok(())
}
