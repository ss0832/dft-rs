// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Stationary full-spin PT2 energy Hessians for scalar one-electron sources.

use super::set_component;
use crate::driver::{
    double_hybrid::{append_spinor_energy, SpinorModel},
    method::MolecularSetup,
    Value,
};
use crate::integrals::london::ComplexMatrix;
use crate::linalg::Matrix;
use crate::periodic::{
    spinor_scf::{spin_diagonal, SpinorScfReport},
    BlochMatrix,
};
use crate::response::{complex::Complex, nuclear::SpinorCorrelatedSolver, ResponseSolverConfig};
use crate::Result;

pub(in crate::driver) fn scalar_sources<const N: usize>(
    operators: [Matrix; N],
) -> [BlochMatrix; N] {
    operators.map(scalar_source)
}

pub(in crate::driver) fn scalar_source(re: Matrix) -> BlochMatrix {
    let im = Matrix::zeros(re.rows, re.cols);
    spin_diagonal(&BlochMatrix { re, im })
}

pub(in crate::driver) struct SpinorHessian<const N: usize> {
    parent: [[f64; N]; N],
    pair: [[f64; N]; N],
    target: [[f64; N]; N],
    pub residuals: Vec<f64>,
    diagnostics: Vec<Value>,
    unscaled_energy: f64,
    target_energy_correction: f64,
    reference_defect: f64,
}

pub(in crate::driver) fn solve<const N: usize>(
    setup: &MolecularSetup,
    scf: &SpinorScfReport,
    fixed_operator: &BlochMatrix,
    model: &SpinorModel,
    sources: [BlochMatrix; N],
    config: &ResponseSolverConfig,
) -> Result<SpinorHessian<N>> {
    let solver = SpinorCorrelatedSolver::new(
        &setup.molecule,
        &setup.basis,
        setup.grid.as_ref(),
        &setup.functional,
        model.target.as_ref(),
        scf,
        &setup.options,
        fixed_operator,
        2,
        model.coefficient,
        model.memory_mb,
        config,
    )?;
    let n = 2 * setup.basis.nao_sph;
    let mut core = vec![ComplexMatrix::zeros(n, n); 3];
    let mut parent = [[0.0; N]; N];
    let mut pair = parent;
    let mut target = parent;
    let mut residuals = Vec::new();
    let mut diagnostics = Vec::new();
    let mut unscaled_energy = 0.0;
    let mut target_energy_correction = 0.0;
    let mut reference_defect = 0.0_f64;
    let directions = (0..N)
        .map(|a| (a, a))
        .chain((0..N).flat_map(|a| (a + 1..N).map(move |b| (a, b))));
    for (a, b) in directions {
        let mut source = sources[a].clone();
        if a != b {
            source.add_scaled(&sources[b], 1.0);
        }
        core[1] = ComplexMatrix {
            re: source.re,
            im: source.im,
        };
        let value = solver.solve(&core)?;
        let path = &value.parent.parent;
        set_component(&mut parent, a, b, path.energy[2]);
        set_component(&mut pair, a, b, value.pairs.energy[2].re);
        if let Some(correction) = solver.target_correction(&core, &value.parent)? {
            set_component(&mut target, a, b, correction[2]);
            if a == 0 && b == 0 {
                target_energy_correction = correction[0];
            }
        }
        if a == 0 && b == 0 {
            unscaled_energy = value.pairs.energy[0].re;
        }
        reference_defect = reference_defect.max(path.reference_defect);
        residuals.extend_from_slice(&path.residuals);
        diagnostics.push(
            Value::object()
                .with("modes", [a, b])
                .with("projector_defects", &path.projector_defects)
                .with("stationarity_defects", &path.stationarity_defects)
                .with(
                    "particle_number_defects",
                    value.parent.particle_number_defects,
                )
                .with("orbital_frame_defect", value.parent.frame_defect)
                .with("pair_residuals", value.pairs.residuals)
                .with("pair_relative_residuals", value.pairs.relative_residuals)
                .with(
                    "pair_imaginary_energy",
                    value
                        .pairs
                        .energy
                        .iter()
                        .map(|coefficient| coefficient.im)
                        .collect::<Vec<_>>(),
                ),
        );
    }
    Ok(SpinorHessian {
        parent,
        pair,
        target,
        residuals,
        diagnostics,
        unscaled_energy,
        target_energy_correction,
        reference_defect,
    })
}

impl<const N: usize> SpinorHessian<N> {
    pub(in crate::driver) fn tensor(&self, coefficient: f64) -> [[Complex; N]; N] {
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                Complex::from(self.parent[a][b] + self.target[a][b] + coefficient * self.pair[a][b])
            })
        })
    }

    pub(in crate::driver) fn append(
        self,
        report: &mut Value,
        model: &SpinorModel,
        property: &str,
        scale: f64,
        source_length_power: usize,
    ) {
        let scaled =
            |tensor: [[f64; N]; N], factor: f64| tensor.map(|row| row.map(|value| factor * value));
        report.set("response_reference_fock_defect", self.reference_defect);
        report.set("independent_components", N * (N + 1) / 2);
        report.set("response_order", 2usize);
        report.set("spinor_quadratic_diagnostics", self.diagnostics);
        append_spinor_energy(
            report,
            model,
            self.unscaled_energy,
            self.target_energy_correction,
        );

        let mut labels = report
            .get("units")
            .cloned()
            .expect("electric response units");
        let length = if source_length_power == 1 {
            "bohr".to_string()
        } else {
            format!("bohr^{source_length_power}")
        };
        let squared_length = format!("bohr^{}", 2 * source_length_power);
        let unit = format!("e^2*{squared_length}/hartree");
        for (key, matrix) in [
            (format!("parent_{property}"), scaled(self.parent, scale)),
            (
                format!("generalized_pt2_{property}"),
                scaled(self.pair, scale),
            ),
            (
                format!("pt2_{property}_correction"),
                scaled(self.pair, scale * model.coefficient),
            ),
        ] {
            report.set(&key, matrix);
            labels.set(&key, unit.as_str());
        }
        if model.target.is_some() {
            let key = format!("target_{property}_correction");
            report.set(&key, scaled(self.target, scale));
            labels.set(&key, unit.as_str());
        }
        let energy = format!("[hartree, e*{length}, {unit}]");
        let dimensionless =
            format!("[dimensionless, e*{length}/hartree, e^2*{squared_length}/hartree^2]");
        for (key, unit) in [
            ("orbital_frame_defect", "atomic units by Taylor order"),
            ("pair_relative_residuals", "dimensionless"),
            ("pair_residuals", energy.as_str()),
            ("pair_imaginary_energy", energy.as_str()),
            ("projector_defects", dimensionless.as_str()),
            ("particle_number_defects", dimensionless.as_str()),
            ("stationarity_defects", energy.as_str()),
        ] {
            labels.set(&format!("spinor_quadratic_diagnostics.{key}"), unit);
        }
        labels.set("response_reference_fock_defect", "hartree");
        report.set("units", labels);
    }
}
