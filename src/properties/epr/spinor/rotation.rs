// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Safeguarded global SU(2) relaxation for insulating molecular LDA references.

use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::spinor_scf::{from_pauli, pauli_components};
use crate::periodic::spinor_scf::{
    run_spinor_scf, SpinorEnergyBackend, SpinorFields, SpinorScfConfig, SpinorScfReport,
};
use crate::periodic::BlochMatrix;
use crate::xc::{Family, ResolvedFunctional};

pub struct GlobalSpinRotation {
    pub density: BlochMatrix,
    pub quaternion: [f64; 4],
    pub work: f64,
    pub torque: [f64; 3],
}

/// Actual solver outcome; recovery never changes a convergence decision.
pub struct SpinorRecovery {
    pub reference: SpinorScfReport,
    pub cycles: usize,
    pub total_iterations: usize,
    pub maximum_rotation_energy_defect: f64,
}

struct Seeded<'a, B> {
    physical: &'a B,
    initial: Option<Vec<BlochMatrix>>,
}

impl<B: SpinorEnergyBackend> SpinorEnergyBackend for Seeded<'_, B> {
    fn overlap(&self) -> &[BlochMatrix] {
        self.physical.overlap()
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        self.initial
            .as_deref()
            .unwrap_or_else(|| self.physical.initial_fock())
    }
    fn electrons(&self) -> f64 {
        self.physical.electrons()
    }
    fn validate_occupation(&self, occupation: OccupationScheme) -> crate::Result<()> {
        self.physical.validate_occupation(occupation)
    }
    fn quadratic_density_energy(&self) -> bool {
        self.physical.quadratic_density_energy()
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> crate::Result<SpinorFields> {
        self.physical.evaluate(density)
    }
}

/// Relax a full complex molecular LDA reference, including its global spin
/// orientation. Short physical SCF segments use unitary global spin steps.
/// Bounded orientation-secant seeds are accepted only after a physical SCF
/// trial improves stationarity without increasing energy beyond arithmetic
/// resolution. Every trial counts against the configured iteration budget.
/// Once orbital/occupation stationarity holds, physical SCF continues without
/// another global spin step until its remaining convergence gates pass.
/// HF, GGA, meta-GGA and statistical occupations retain the ordinary solver;
/// finite angular rules are not assumed rotation invariant.
pub fn run_epr_spinor_reference(
    backend: &impl SpinorEnergyBackend,
    functional: &ResolvedFunctional,
    config: &SpinorScfConfig,
    operator: &BlochMatrix,
) -> crate::Result<SpinorRecovery> {
    if functional.family != Family::Lda
        || functional.parts.is_empty()
        || !matches!(config.occupation, OccupationScheme::Insulator)
        || config.max_iter == 0
    {
        let reference = run_spinor_scf(backend, config)?;
        return Ok(SpinorRecovery {
            total_iterations: reference.iterations,
            reference,
            cycles: 0,
            maximum_rotation_energy_defect: 0.0,
        });
    }
    if backend.overlap().len() != 1 {
        return Err(crate::DftError::Unsupported {
            feature: "molecular EPR global-spin recovery on multiple k sectors".into(),
            missing: "a weighted periodic rotation and reference validation".into(),
        });
    }
    let metric = pauli_components(&backend.overlap()[0])?;
    let metric_scale = metric[0]
        .re
        .as_slice()
        .iter()
        .chain(metric[0].im.as_slice())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    if metric[1..].iter().any(|m| {
        m.re.as_slice()
            .iter()
            .chain(m.im.as_slice())
            .any(|v| v.abs() > 64.0 * f64::EPSILON * metric_scale)
    }) {
        return Err(crate::DftError::Unsupported {
            feature: "EPR global-spin recovery with a spin-dependent overlap metric".into(),
            missing: "the spin-covariant metric transformation for that basis".into(),
        });
    }
    let mut seeded = Seeded {
        physical: backend,
        initial: None,
    };
    let mut total_iterations = 0;
    let mut cycles = 0;
    let mut maximum_rotation_energy_defect = 0.0_f64;
    let mut previous: Option<[f64; 3]> = None;
    let mut saved: Option<SpinorScfReport> = None;
    loop {
        let mut segment = config.clone();
        segment.max_iter = (config.max_iter - total_iterations).min(8);
        let mut reference = run_spinor_scf(&seeded, &segment)?;
        total_iterations += reference.iterations;
        let mut allow_acceleration = true;
        if let Some(old) = saved.take() {
            let score = |r: &SpinorScfReport| {
                (r.orbital_residual / config.orbital_tolerance)
                    .max(r.occupation_consistency / config.density_tolerance)
            };
            let roundoff = 64.0
                * f64::EPSILON
                * (old.fields.energy.abs() + reference.fields.energy.abs()).max(1.0);
            let accepted = reference.fields.operator_converged
                && reference.fields.energy <= old.fields.energy + roundoff
                && score(&reference) < score(&old);
            crate::logger::log(&format!("[epr-spin-secant] trial accepted={accepted} iterations={total_iterations} old_score={} new_score={} energy_change={}",
                score(&old), score(&reference), reference.fields.energy-old.fields.energy));
            if !accepted {
                reference = old;
                previous = None;
                allow_acceleration = false;
            }
        }
        if reference.converged || total_iterations >= config.max_iter {
            return Ok(SpinorRecovery {
                reference,
                cycles,
                total_iterations,
                maximum_rotation_energy_defect,
            });
        }
        if reference.fields.operator_converged
            && reference.orbital_residual < config.orbital_tolerance
            && reference.occupation_consistency < config.density_tolerance
        {
            // The physical stationarity gates already hold. Continue the
            // same fully noncollinear SCF state while its energy/density
            // iteration gates settle, rather than applying another rotation.
            let overlap = &backend.overlap()[0];
            let product = crate::dfpt::orbitals::matmul(
                overlap,
                &crate::dfpt::orbitals::matmul(&reference.density[0], overlap)?,
            )?;
            let mut initial = overlap.clone();
            initial.add_scaled(&product, -2.0);
            seeded.initial = Some(vec![initial]);
            previous = None;
            crate::logger::log(&format!(
                "[epr-spin-settle] iter={total_iterations} orbital={} occupation={} dP={} dE={}",
                reference.orbital_residual,
                reference.occupation_consistency,
                reference.density_change,
                reference.energy_change
            ));
            continue;
        }
        let rotation = optimal_global_spin_rotation(&reference.density[0], operator)?;
        let mut q = rotation.quaternion;
        if q[0] < 0.0 {
            q = q.map(|x| -x);
        }
        let sine = q[1].hypot(q[2]).hypot(q[3]);
        let angle = 2.0 * sine.atan2(q[0]);
        let vector = if sine == 0.0 {
            [0.0; 3]
        } else {
            std::array::from_fn(|a| angle * q[a + 1] / sine)
        };
        let mut alpha = 1.0_f64;
        if allow_acceleration && angle > 0.0 && angle < 0.01 {
            if let Some(old) = previous {
                let denominator: f64 = old.iter().map(|x| x * x).sum();
                let product: f64 = old.iter().zip(vector).map(|(a, b)| a * b).sum();
                let lambda = product / denominator;
                let mismatch = old
                    .iter()
                    .zip(vector)
                    .map(|(a, b)| (b - lambda * a).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if lambda > 0.5 && lambda < 1.0 && mismatch < 0.05 * angle {
                    alpha = (1.0 / (1.0 - lambda)).clamp(1.0, 128.0).min(0.1 / angle);
                }
            }
        }
        let trial = if alpha > 1.0 {
            let half = 0.5 * alpha * angle;
            let mut power = [half.cos(), 0.0, 0.0, 0.0];
            for a in 0..3 {
                power[a + 1] = vector[a] / angle * half.sin();
            }
            rotate_quaternion(&reference.density[0], operator, power)?
        } else {
            rotation
        };
        let fields = backend.evaluate(std::slice::from_ref(&trial.density))?;
        let defect = (fields.energy - reference.fields.energy - trial.work).abs();
        let roundoff =
            64.0 * f64::EPSILON * (fields.energy.abs() + reference.fields.energy.abs()).max(1.0);
        if !defect.is_finite() || defect > roundoff || alpha == 1.0 && trial.work > roundoff {
            return Err(crate::DftError::Unsupported {
                feature: "global-spin EPR recovery for a noninvariant energy model".into(),
                missing: "a scalar-spin-invariant backend with consistent one-body rotation work"
                    .into(),
            });
        }
        maximum_rotation_energy_defect = maximum_rotation_energy_defect.max(defect);
        crate::logger::log(&format!("[epr-spin-secant] proposal cycle={cycles} iter={total_iterations} alpha={alpha} angle={angle} occupation={} orbital={} defect={defect}",
            reference.occupation_consistency, reference.orbital_residual));
        let overlap = &backend.overlap()[0];
        let product = crate::dfpt::orbitals::matmul(
            overlap,
            &crate::dfpt::orbitals::matmul(&trial.density, overlap)?,
        )?;
        let mut initial = overlap.clone();
        initial.add_scaled(&product, -2.0);
        seeded.initial = Some(vec![initial]);
        previous = if alpha > 1.0 { None } else { Some(vector) };
        if alpha > 1.0 {
            saved = Some(reference);
        }
        cycles += 1;
    }
}

pub(crate) fn rotate_quaternion(
    density: &BlochMatrix,
    operator: &BlochMatrix,
    q: [f64; 4],
) -> crate::Result<GlobalSpinRotation> {
    let p = pauli_components(density)?;
    let h = pauli_components(operator)?;
    let [w, x, y, z] = q;
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut rotated = p.clone();
    for i in 0..3 {
        rotated[i + 1].scale(0.0);
        for j in 0..3 {
            rotated[i + 1].add_scaled(&p[j + 1], r[i][j]);
        }
    }
    let work = (1..4)
        .map(|a| 0.5 * (rotated[a].trace_product(&h[a]) - p[a].trace_product(&h[a])))
        .sum();
    Ok(GlobalSpinRotation {
        density: from_pauli(&rotated)?,
        quaternion: q,
        work,
        torque: [0.0; 3],
    })
}

/// An exact proper global spin rotation minimizes the complete one-body
/// spin-dependent energy. Spin-scalar and rotationally invariant electronic
/// terms are unchanged; no spin direction is imposed on the final SCF state.
pub fn optimal_global_spin_rotation(
    density: &BlochMatrix,
    operator: &BlochMatrix,
) -> crate::Result<GlobalSpinRotation> {
    if density.rows() != operator.rows() || density.cols() != operator.cols() {
        return Err(crate::DftError::InvalidInput(
            "global spin rotation requires matching density and operator dimensions".into(),
        ));
    }
    let p = pauli_components(density)?;
    let h = pauli_components(operator)?;
    let m: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * p[j + 1].trace_product(&h[i + 1])));
    let torque = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]];
    let scale = m.iter().flatten().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if !scale.is_finite() || m.iter().flatten().any(|value| !value.is_finite()) {
        return Err(crate::DftError::InvalidInput(
            "non-finite global spin-rotation contraction".into(),
        ));
    }
    let mut q = [1.0, 0.0, 0.0, 0.0];
    if scale != 0.0 {
        let trace = (0..3).map(|i| m[i][i]).sum::<f64>();
        let mut k = Matrix::zeros(4, 4);
        k[(0, 0)] = trace / scale;
        for a in 0..3 {
            k[(0, a + 1)] = torque[a] / scale;
            k[(a + 1, 0)] = torque[a] / scale;
            for b in 0..3 {
                k[(a + 1, b + 1)] = (m[a][b] + m[b][a] - if a == b { trace } else { 0.0 }) / scale;
            }
        }
        let (values, vectors) = symmetric_eigen(&k)?;
        let norm = values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        // Project the identity quaternion into a numerically degenerate
        // minimum eigenspace. This selects its closest gauge representative.
        q = [0.0; 4];
        for a in 0..4 {
            if values[a] - values[0] <= 32.0 * f64::EPSILON * norm {
                for i in 0..4 {
                    q[i] += vectors[(i, a)] * vectors[(0, a)];
                }
            }
        }
        let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        if length == 0.0 {
            q = std::array::from_fn(|i| vectors[(i, 0)]);
        } else {
            q = q.map(|v| v / length);
        }
    }
    let [w, x, y, z] = q;
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut rotated = p.clone();
    for i in 0..3 {
        rotated[i + 1].scale(0.0);
        for j in 0..3 {
            rotated[i + 1].add_scaled(&p[j + 1], r[i][j]);
        }
    }
    let result = from_pauli(&rotated)?;
    // Exclude the invariant scalar block before subtracting work.
    let work = (1..4)
        .map(|a| 0.5 * (rotated[a].trace_product(&h[a]) - p[a].trace_product(&h[a])))
        .sum();
    Ok(GlobalSpinRotation {
        density: result,
        quaternion: q,
        work,
        torque,
    })
}
