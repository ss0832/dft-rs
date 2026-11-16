// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic periodic XC Fock differential using the SCF descriptor map.

use super::{
    density_is_psd_to_roundoff, integrate_spinor_grid_gradient, project, spin_descriptor,
    spinor_grid_descriptors, spinor_semilocal_point_kernel, spinor_semilocal_point_taylor,
    SpinorXcPoint, MAX_SPIN_ANGULAR_ORDER,
};
use crate::dual::DualN;
use crate::periodic::BlochMatrix;
use crate::resources::{checked_bytes, reserve, MemoryReservation};
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut, RHO_CUTOFF};
use crate::{DftError, Result};
use rayon::prelude::*;

pub(in crate::periodic) struct GridKernel {
    active: usize,
    points: Vec<Vec<f64>>,
    _memory: MemoryReservation,
}

impl GridKernel {
    pub(in crate::periodic) fn new(
        functional: &ResolvedFunctional,
        weights: &[f64],
        orbitals: &[&BlochMatrix],
        momentum: &[&[BlochMatrix; 3]],
        density: &[[BlochMatrix; 4]],
        tolerance: f64,
    ) -> Result<Self> {
        let active = if functional.family == Family::MetaGga {
            32
        } else {
            16
        };
        let bytes = checked_bytes(
            &[weights.len(), active, active, size_of::<f64>()],
            "projected spinor XC kernel",
        )?;
        let memory = reserve(bytes, "projected spinor XC kernel")?;
        let descriptors = spinor_grid_descriptors(orbitals, momentum, density);
        let point_tolerance = tolerance / weights.len().max(1) as f64;
        let points = descriptors
            .into_par_iter()
            .enumerate()
            .map(|(g, mut point)| {
                if point.density[0] < RHO_CUTOFF || weights[g] == 0.0 {
                    return Ok((Vec::new(), 0.0));
                }
                let values = point.descriptors();
                let scale = values[0]
                    .abs()
                    .max(values[16].abs())
                    .max(values[0].abs().sqrt() * values[16].abs().sqrt());
                if (0..active)
                    .filter(|&i| spin_descriptor(i))
                    .all(|i| values[i].abs() <= 64.0 * f64::EPSILON * scale)
                {
                    return Ok((unpolarized_kernel(functional, values, active)?, 0.0));
                }
                if !density_is_psd_to_roundoff(point.density) {
                    return Err(DftError::InvalidInput(
                        "periodic spinor XC response requires a positive-semidefinite grid density"
                            .into(),
                    ));
                }
                // Repair cancellation in the charge-minus-spin eigenvalue only
                // within the same roundoff bound used by the SCF projection.
                let spin = point.density[1]
                    .hypot(point.density[2])
                    .hypot(point.density[3]);
                point.density[0] = point.density[0].max(spin);
                let bound = ao_descriptor_bound(g, orbitals, momentum);
                let mut previous = point_kernel(functional, &point, active, 4)?;
                let mut defect = f64::INFINITY;
                for rule in [8, 16, 32, 64, MAX_SPIN_ANGULAR_ORDER] {
                    let current = point_kernel(functional, &point, active, rule)?;
                    defect = current
                        .iter()
                        .zip(&previous)
                        .enumerate()
                        .map(|(i, (a, b))| (a - b).abs() * bound[i / active] * bound[i % active])
                        .sum::<f64>()
                        * weights[g].abs();
                    previous = current;
                    if defect <= point_tolerance {
                        break;
                    }
                }
                Ok((previous, defect))
            })
            .collect::<Result<Vec<_>>>()?;
        let defect: f64 = points.iter().map(|(_, d)| d).sum();
        if defect > tolerance || !defect.is_finite() {
            return Err(DftError::LinearAlgebra(format!(
                "projected spinor XC density kernel did not converge: weighted AO defect {defect:e}, tolerance {tolerance:e}"
            )));
        }
        crate::logger::debug(&format!(
            "Periodic spinor XC kernel weighted angular defect={defect:.6e}"
        ));
        Ok(Self {
            active,
            points: points.into_iter().map(|(values, _)| values).collect(),
            _memory: memory,
        })
    }

    pub(in crate::periodic) fn apply(
        &self,
        weights: &[f64],
        orbitals: &[&BlochMatrix],
        momentum: &[&[BlochMatrix; 3]],
        direction: &[[BlochMatrix; 4]],
    ) -> Result<Vec<BlochMatrix>> {
        let tangent = spinor_grid_descriptors(orbitals, momentum, direction);
        let gradient: Vec<[f64; 32]> = self
            .points
            .par_iter()
            .zip(&tangent)
            .map(|(matrix, direction)| {
                let data = direction.descriptors();
                let mut response = [0.0; 32];
                for (row, out) in matrix.chunks_exact(self.active).zip(&mut response) {
                    *out = row.iter().zip(data).map(|(a, b)| a * b).sum();
                }
                response
            })
            .collect();
        integrate_spinor_grid_gradient(weights, orbitals, momentum, |g| gradient[g])
    }
}

fn ao_descriptor_bound(
    g: usize,
    orbitals: &[&BlochMatrix],
    momentum: &[&[BlochMatrix; 3]],
) -> [f64; 32] {
    let row_max = |m: &BlochMatrix| {
        (0..m.cols())
            .map(|i| m.re[(g, i)].hypot(m.im[(g, i)]))
            .fold(0.0_f64, f64::max)
    };
    let value = orbitals.iter().map(|m| row_max(m)).fold(0.0_f64, f64::max);
    let p: [f64; 3] = std::array::from_fn(|axis| {
        momentum
            .iter()
            .map(|m| row_max(&m[axis]))
            .fold(0.0_f64, f64::max)
    });
    std::array::from_fn(|i| match i {
        0..=3 => 4.0 * value * value,
        4..=15 => 8.0 * value * p[(i - 4) % 3],
        16..=19 => 2.0 * p.iter().map(|v| v * v).sum::<f64>(),
        _ => 4.0 * value * p[(i - 20) % 3],
    })
}

fn point_kernel(
    functional: &ResolvedFunctional,
    point: &SpinorXcPoint,
    active: usize,
    rule: usize,
) -> Result<Vec<f64>> {
    if let Some(result) = spinor_semilocal_point_kernel(functional, point, rule)? {
        return Ok(result.kernel);
    }
    // The analytic Taylor path owns density and one-orbital boundaries.
    let input = point.descriptors();
    let mut matrix = vec![0.0; active * active];
    for column in 0..active {
        let coefficients = std::array::from_fn(|a| vec![input[a], f64::from(a == column)]);
        let taylor = spinor_semilocal_point_taylor(functional, &coefficients, rule)?;
        for row in 0..active {
            matrix[row * active + column] = taylor.gradient[row][1];
        }
    }
    Ok(matrix)
}

fn unpolarized_kernel(
    functional: &ResolvedFunctional,
    mut values: [f64; 32],
    active: usize,
) -> Result<Vec<f64>> {
    // At zero spin descriptors, the spin-scaling contribution multiplies
    // the projected spin curvature by three. Angular averaging of n_i n_j
    // gives delta_ij/3, so a single collinear projection gives all three
    // identical spin blocks. Mixed charge/spin blocks average to zero.
    const CHARGE: [usize; 8] = [0, 4, 5, 6, 16, 20, 21, 22];
    type Tangent = DualN<f64, 16>;
    let count = if active == 32 { 8 } else { 4 };
    let charge = &CHARGE[..count];
    let magnetic = |axis: usize, i: usize| match i {
        0 | 16 => i + axis + 1,
        _ => i + 3 * (axis + 1),
    };
    for (i, v) in values.iter_mut().enumerate() {
        if spin_descriptor(i) {
            *v = 0.0;
        }
    }
    let mut descriptors = values.map(|v| DualN::<Tangent, 16>::constant(Tangent::constant(v)));
    for (j, &i) in charge.iter().enumerate() {
        descriptors[i] = DualN::variable(Tangent::variable(values[i], j), j);
        let i = magnetic(2, i);
        descriptors[i] = DualN::variable(Tangent::variable(0.0, count + j), count + j);
    }
    let projection = project(&descriptors, [0.0, 0.0, 1.0], functional.family, false)?;
    let data = projection.map(|q| [q.re.re]);
    let input = GridVars {
        rho: [&data[0], &data[1]],
        sigma: Some([&data[2], &data[3], &data[4]]),
        tau: (active == 32).then_some([&data[5], &data[6]]),
    };
    let mut output = XcOut::default();
    functional.eval(&input, 2, &mut output)?;
    let (v, f) = output.potential_and_kernel(0, functional.family)?;
    let derivative = |i: usize, j: usize| {
        (0..7)
            .map(|a| {
                v[a] * projection[a].im[i].im[j]
                    + (0..7)
                        .map(|b| f[a][b] * projection[a].re.im[i] * projection[b].re.im[j])
                        .sum::<f64>()
            })
            .sum::<f64>()
    };
    let mut hessian = vec![0.0; active * active];
    for (i, &a) in charge.iter().enumerate() {
        for (j, &b) in charge.iter().enumerate() {
            hessian[a * active + b] = derivative(i, j);
            let spin = derivative(count + i, count + j);
            for axis in 0..3 {
                hessian[magnetic(axis, a) * active + magnetic(axis, b)] = spin;
            }
        }
    }
    Ok(hessian)
}
