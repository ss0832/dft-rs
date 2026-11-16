// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static magnetic derivatives of the same variational Dirac energy used for SCF.

use super::super::grid::FieldXcCache;
use super::super::matrix::{congruence, max_abs};
use super::{BlochMatrix, Complex, MolecularTwoComponent, SpinorScfReport};
use crate::dual::{Scalar, Taylor};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::AnalyticShell;
use crate::integrals::dirac::{
    london_magnetic_balance, matrix_coefficients, nuclear_magnetic_dipole_vertices,
    GaussianMagnetization, GaussianNucleus, LondonCoulombTaylor, LondonCurrentTaylor,
};
use crate::math::Vec3;
use crate::response::krylov::ResponseSolverConfig;
use crate::{DftError, Result};

type Jet = Taylor<3>;
type Tensor = [[f64; 3]; 3];

const AXES: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub(crate) struct MagneticTensors {
    pub total: Vec<Tensor>,
    pub explicit: Vec<Tensor>,
    pub relaxed: Vec<Tensor>,
}

pub(crate) struct DiracMagneticResponse {
    pub moment: [f64; 3],
    pub tensors: MagneticTensors,
    pub reciprocity_defect: f64,
    pub reference_energy_defect: f64,
    pub reference_fock_defect: f64,
    pub metric_defect: f64,
    pub sector_residual: f64,
    pub residuals: Vec<f64>,
}

/// Maximum raw errors across all analytic directions.
#[derive(Default)]
struct Diagnostics {
    energy: f64,
    fock: f64,
    metric: f64,
    sector: f64,
}

struct UniformPath {
    direction: Vec3,
    core: Vec<BlochMatrix>,
    metric: Vec<BlochMatrix>,
    coulomb: Option<LondonCoulombTaylor>,
    range: Option<LondonCoulombTaylor>,
    current: Option<LondonCurrentTaylor>,
}

struct PathResult {
    energy: [f64; 3],
    vertex: BlochMatrix,
    diagnostics: Diagnostics,
}

impl Diagnostics {
    fn include(&mut self, other: &Self) {
        self.energy = self.energy.max(other.energy);
        self.fock = self.fock.max(other.fock);
        self.metric = self.metric.max(other.metric);
        self.sector = self.sector.max(other.sector);
    }
}

struct MagneticResponse<'a, 'model> {
    backend: &'a MolecularTwoComponent<'model>,
    state: &'a SpinorScfReport,
    shells: Vec<AnalyticShell<Jet>>,
    nuclei: Vec<GaussianNucleus<Jet>>,
    phase_order: usize,
    field: [f64; 3],
    xc_cache: Option<FieldXcCache>,
}

impl MagneticResponse<'_, '_> {
    fn field_jet(&self, direction: [f64; 3]) -> [Jet; 3] {
        std::array::from_fn(|axis| Jet::from_coefficients(&[self.field[axis], direction[axis]]))
    }

    fn uniform(&self, direction: [f64; 3]) -> Result<UniformPath> {
        let [x, y, z] = direction;
        if direction == [0.0; 3] {
            let [core, metric] = self.backend.dirac_core.as_ref().expect("Dirac reference");
            return Ok(UniformPath {
                direction: Vec3::zero(),
                core: vec![
                    core.clone(),
                    BlochMatrix::zeros(core.rows(), core.cols()),
                    BlochMatrix::zeros(core.rows(), core.cols()),
                ],
                metric: vec![metric.clone()],
                coulomb: None,
                range: None,
                current: None,
            });
        }
        let speed = self.backend.options.speed_of_light;
        let integrals = london_magnetic_balance(
            &self.shells,
            &self.nuclei,
            Jet::from_f64(speed),
            self.field_jet(direction),
            2,
            self.phase_order,
        )?;
        let weights = self.backend.functional.hybrid;
        // At most the three Cartesian uniform paths and one mixed path
        // coexist. Divide the cache budget across them and both kernels.
        let current = self.backend.dirac_interaction.current();
        let cache_count = 1 + usize::from(weights.clr != 0.0) + usize::from(current.is_some());
        let memory_mb = self.backend.options.pair_cache_mb / (4 * cache_count);
        let create = |kernel| {
            LondonCoulombTaylor::new(
                &self.backend.shells,
                speed,
                kernel,
                self.field,
                direction,
                2,
                self.phase_order,
                memory_mb,
            )
        };
        Ok(UniformPath {
            direction: Vec3::new(x, y, z),
            core: matrix_coefficients(
                &integrals.hamiltonian_real,
                Some(&integrals.hamiltonian_imaginary),
                2,
            )?,
            metric: matrix_coefficients(&integrals.overlap, Some(&integrals.overlap_imaginary), 2)?,
            coulomb: Some(create(GaussianKernel::Coulomb)?),
            range: (weights.clr != 0.0)
                .then(|| {
                    create(GaussianKernel::LongRange {
                        omega: weights.omega,
                    })
                })
                .transpose()?,
            current: current
                .map(|interaction| {
                    LondonCurrentTaylor::new(
                        &self.backend.shells,
                        speed,
                        interaction,
                        self.field,
                        direction,
                        2,
                        self.phase_order,
                        memory_mb,
                    )
                })
                .transpose()?,
        })
    }

    fn evaluate(
        &self,
        uniform: &UniformPath,
        source: Option<(GaussianMagnetization<f64>, usize)>,
    ) -> Result<PathResult> {
        let backend = self.backend;
        let mut core = uniform.core.clone();
        if let Some((source, axis)) = source {
            let order = usize::from(uniform.coulomb.is_some());
            let all = nuclear_magnetic_dipole_vertices(
                &self.shells,
                GaussianMagnetization {
                    center: source.center.map(Jet::from_f64),
                    rms_radius: Jet::from_f64(source.rms_radius),
                },
                Jet::from_f64(backend.options.speed_of_light),
                self.field_jet(uniform.direction.to_array()),
                order,
                self.phase_order,
            )?;
            let [re, im] = &all[axis];
            for (coefficient, op) in
                core.iter_mut()
                    .skip(1)
                    .zip(matrix_coefficients(re, Some(im), order)?)
            {
                coefficient.add_scaled(&op, 1.0);
            }
        }
        let path = backend.positive_sector_path(&core, &uniform.metric, &self.state.density[0])?;
        let dim = core[0].rows();
        let mut pair = std::array::from_fn::<_, 3, _>(|_| BlochMatrix::zeros(dim, dim));
        let weights = backend.functional.hybrid;
        for (n, potential) in pair.iter_mut().enumerate() {
            if let Some(cache) = &uniform.coulomb {
                for (k, d) in path.density.iter().enumerate().take(n + 1) {
                    let fields = cache.fields(n - k, d)?;
                    potential.add_scaled(&fields.direct, 1.0);
                    potential.add_scaled(&fields.exchange, -weights.cx);
                    if let Some(cache) = &uniform.range {
                        potential.add_scaled(&cache.fields(n - k, d)?.exchange, -weights.clr);
                    }
                    if let Some(cache) = &uniform.current {
                        let fields = cache.fields(n - k, d)?;
                        potential.add_scaled(&fields.direct, 1.0);
                        potential.add_scaled(&fields.exchange, -1.0);
                    }
                }
            } else {
                *potential = backend.pair(&path.density[n])?.operator;
            }
        }
        let mut energy = std::array::from_fn::<_, 3, _>(|n| {
            (0..=n)
                .map(|k| {
                    path.density[k].trace_product(&core[n - k])
                        + 0.5 * path.density[k].trace_product(&pair[n - k])
                })
                .sum::<f64>()
        });
        let mut fock = [core[0].clone(), core[1].clone()];
        for (matrix, interaction) in fock.iter_mut().zip(pair) {
            matrix.add_scaled(&interaction, 1.0);
        }
        let accumulate = |energy: &mut [f64; 3],
                          fock: &mut [BlochMatrix; 2],
                          term: ([f64; 3], [BlochMatrix; 2])| {
            for (total, local) in energy.iter_mut().zip(term.0) {
                *total += local;
            }
            for (total, local) in fock.iter_mut().zip(term.1) {
                total.add_scaled(&local, 1.0);
            }
        };
        if let Some(grid) = &backend.semilocal {
            let term = grid.semilocal_field_path(
                backend.functional,
                &path.density,
                uniform.direction,
                self.xc_cache.as_ref().expect("active balanced grid"),
            )?;
            accumulate(&mut energy, &mut fock, term);
        }
        if let Some((grid, parameters)) = &backend.nonlocal {
            let term = grid.vv10_field_path(&path.density, uniform.direction, *parameters)?;
            accumulate(&mut energy, &mut fock, term);
        }
        energy[0] += backend.nuclear_repulsion + backend.options.dispersion_energy.unwrap_or(0.0);
        let mut projected = congruence(&path.embedding, &fock, 1)?;
        let vertex = projected.remove(1);
        projected[0].add_scaled(&self.state.fields.fock[0], -1.0);
        Ok(PathResult {
            energy,
            vertex,
            diagnostics: Diagnostics {
                energy: (energy[0] - self.state.fields.energy).abs(),
                fock: max_abs(&projected[0]),
                metric: path.metric_defect,
                sector: path.projector_residual,
            },
        })
    }
}

impl MolecularTwoComponent<'_> {
    /// Return static magnetizability with no nuclear sources, or electronic
    /// shielding for each supplied nucleus. The common KS kernel is evaluated
    /// at the actual stationary reference with the same balanced interaction.
    pub(crate) fn dirac_magnetic_response(
        &self,
        state: &SpinorScfReport,
        nuclei: &[GaussianNucleus<f64>],
        shielding: &[GaussianMagnetization<f64>],
        config: &ResponseSolverConfig,
    ) -> Result<DiracMagneticResponse> {
        let Some(model) = self.magnetic else {
            return Err(DftError::Unsupported {
                feature: "four-component magnetic response".into(),
                missing: "a London-RMB stationary reference".into(),
            });
        };
        if self.dirac_core.is_none()
            || self.options.electric_field != [0.0; 3]
            || self.additional_electric.is_some()
        {
            return Err(DftError::Unsupported {
                feature: "four-component magnetic response with additional external operators"
                    .into(),
                missing: "London-RMB derivatives of the additional electric operators".into(),
            });
        }
        if !state.converged || state.density.len() != 1 {
            return Err(DftError::InvalidInput(
                "Dirac magnetic response needs a converged molecular reference".into(),
            ));
        }
        let xc_cache = self
            .semilocal
            .as_ref()
            .map(|grid| {
                let diagnostics = state
                    .fields
                    .xc_diagnostics
                    .as_ref()
                    .expect("balanced XC state");
                let density = self.map.lift_density(&state.density[0])?;
                grid.prepare_field_xc(
                    self.functional,
                    &density,
                    diagnostics.angular_order,
                    self.options.pair_cache_mb / 4,
                )
            })
            .transpose()?;
        let calculation = MagneticResponse {
            backend: self,
            state,
            shells: self
                .shells
                .iter()
                .map(|s| AnalyticShell {
                    l: s.l,
                    center: s.center.map(Jet::from_f64),
                    exponents: s.exponents.iter().copied().map(Jet::from_f64).collect(),
                    coefficients: s.coefficients.iter().copied().map(Jet::from_f64).collect(),
                })
                .collect(),
            nuclei: nuclei
                .iter()
                .map(|n| GaussianNucleus {
                    center: n.center.map(Jet::from_f64),
                    charge: Jet::from_f64(n.charge),
                    rms_radius: Jet::from_f64(n.rms_radius),
                })
                .collect(),
            phase_order: model.phase_order.max(2),
            field: model.field,
            xc_cache,
        };
        let mut diagnostics = Diagnostics::default();
        let mut evaluate = |path: &UniformPath, source| {
            let result = calculation.evaluate(path, source)?;
            diagnostics.include(&result.diagnostics);
            Ok::<_, DftError>(result)
        };
        let uniform = AXES
            .map(|axis| calculation.uniform(axis))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let base = uniform
            .iter()
            .map(|path| evaluate(path, None))
            .collect::<Result<Vec<_>>>()?;
        let moment = std::array::from_fn(|i| -base[i].energy[1]);
        let mut vertices = base
            .iter()
            .map(|path| path.vertex.clone())
            .collect::<Vec<_>>();
        let mut explicit = Vec::new();
        if shielding.is_empty() {
            let mut tensor = [[0.0; 3]; 3];
            for (i, row) in tensor.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate().take(i + 1) {
                    *value = if i == j {
                        -2.0 * base[i].energy[2]
                    } else {
                        let combined = std::array::from_fn(|a| AXES[i][a] + AXES[j][a]);
                        let both = evaluate(&calculation.uniform(combined)?, None)?;
                        -(both.energy[2] - base[i].energy[2] - base[j].energy[2])
                    };
                }
            }
            for i in 0..3 {
                for j in 0..i {
                    tensor[j][i] = tensor[i][j];
                }
            }
            explicit.push(tensor);
        } else {
            let fixed = calculation.uniform([0.0; 3])?;
            for &nucleus in shielding {
                let mut tensor = [[0.0; 3]; 3];
                for (i, row) in tensor.iter_mut().enumerate() {
                    let source = Some((nucleus, i));
                    let pure = evaluate(&fixed, source)?;
                    for (j, (path, magnetic)) in uniform.iter().zip(&base).enumerate() {
                        let both = evaluate(path, source)?;
                        row[j] = both.energy[2] - pure.energy[2] - magnetic.energy[2];
                    }
                    vertices.push(pure.vertex);
                }
                explicit.push(tensor);
            }
        }
        if diagnostics.fock > 1e-9 || diagnostics.energy > 1e-9 {
            return Err(DftError::LinearAlgebra(format!(
                "Dirac magnetic path mismatches stationary energy {:.6e} or Fock {:.6e} Hartree",
                diagnostics.energy, diagnostics.fock,
            )));
        }
        let response = self
            .operator_response(state, &vertices, &[Complex::ZERO], config)?
            .remove(0);
        let mut reciprocity = 0.0_f64;
        let mut relaxed = vec![[[0.0; 3]; 3]; explicit.len()];
        let mut total = explicit.clone();
        for (atom, (total, relaxed)) in total.iter_mut().zip(&mut relaxed).enumerate() {
            for (i, (total, relaxed)) in total.iter_mut().zip(relaxed).enumerate() {
                for (j, (value, change)) in total.iter_mut().zip(relaxed).enumerate() {
                    let k = if shielding.is_empty() {
                        i
                    } else {
                        3 * (atom + 1) + i
                    };
                    let forward = response.susceptibility[k][j];
                    let reverse = response.susceptibility[j][k];
                    if forward.im.abs().max(reverse.im.abs()) > 1e-10 {
                        return Err(DftError::LinearAlgebra(
                            "static Dirac magnetic response has an imaginary coefficient".into(),
                        ));
                    }
                    reciprocity = reciprocity.max((forward.re - reverse.re).abs());
                    *change = if shielding.is_empty() {
                        -forward.re
                    } else {
                        forward.re
                    };
                    *value += *change;
                }
            }
        }
        Ok(DiracMagneticResponse {
            moment,
            tensors: MagneticTensors {
                total,
                explicit,
                relaxed,
            },
            reciprocity_defect: reciprocity,
            reference_energy_defect: diagnostics.energy,
            reference_fock_defect: diagnostics.fock,
            metric_defect: diagnostics.metric,
            sector_residual: diagnostics.sector,
            residuals: response.residuals,
        })
    }
}
