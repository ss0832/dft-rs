// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Variational optimization of the real occupied subspaces and Fermi
//! descriptors in the same Fermi-Lowdin energy.

use super::{
    evaluate_descriptors, occupied_blocks, optimize_coordinates, DescriptorGradient,
    FixedDescriptorSic, FloDescriptorRun,
};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{fields::MolecularDensityModel, sic::rotate, ScfResult};
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug)]
pub struct StationaryFloOptions {
    pub orbital_tolerance: f64,
    pub descriptor_tolerance: f64,
    pub max_iterations: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct FloStationarity {
    pub converged: bool,
    pub iterations: usize,
    pub evaluations: usize,
    pub orbital_residual: f64,
    pub descriptor_residual: f64,
}

struct Point {
    state: FixedDescriptorSic,
    descriptor_gradient: DescriptorGradient,
    orbital_gradient: [Matrix; 2],
    orbital_direction: [Matrix; 2],
}

impl Point {
    fn residuals(&self) -> (f64, f64) {
        let orbital = self
            .orbital_gradient
            .iter()
            .flat_map(Matrix::as_slice)
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        let descriptor = self
            .descriptor_gradient
            .iter()
            .flatten()
            .flatten()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        (orbital, descriptor)
    }
}

fn point(
    model: &MolecularDensityModel<'_>,
    seed: &ScfResult,
    orbitals: &[Matrix; 2],
    descriptors: [&[Vec3]; 2],
) -> Result<Point> {
    let counts = [seed.n_alpha, seed.n_beta];
    let occupied = occupied_blocks(orbitals, counts);
    let density = occupied
        .each_ref()
        .map(|c| c.leading_columns_gram(c.cols, 1.0));
    let fields = model.evaluate(&density)?;
    let (mut state, derivatives) =
        evaluate_descriptors(model, seed, &occupied, descriptors, true, 0.0)?;
    let derivatives = derivatives.expect("requested FLO derivatives");
    state.parent_energy = fields.electronic_energy + seed.e_nuc;
    state.energy = state.parent_energy + state.correction;
    let mut gradient = orbitals.each_ref().map(|c| Matrix::zeros(c.cols, c.cols));
    let mut direction = gradient.clone();
    for s in 0..2 {
        let c = &orbitals[s];
        let h_mo = c.transpose_matmul_seq(&fields.fock[s].matmul_seq(c));
        let sic_gradient = c.transpose_matmul_seq(&derivatives.occupied[s]);
        for i in 0..counts[s] {
            for a in counts[s]..c.cols {
                let parent_gradient = h_mo[(a, i)];
                state.parent_orbital_residual =
                    state.parent_orbital_residual.max(parent_gradient.abs());
                gradient[s][(a, i)] = 2.0 * parent_gradient + sic_gradient[(a, i)];
                // The diagonal parent gap conditions the search only.
                let curvature = (2.0 * (h_mo[(a, a)] - h_mo[(i, i)]).abs()).max(0.2);
                direction[s][(a, i)] = -gradient[s][(a, i)] / curvature;
            }
        }
    }
    if !state.energy.is_finite()
        || gradient
            .iter()
            .flat_map(Matrix::as_slice)
            .any(|x| !x.is_finite())
    {
        return Err(DftError::LinearAlgebra(
            "non-finite self-consistent FLO-SIC orbital residual".into(),
        ));
    }
    Ok(Point {
        state,
        descriptor_gradient: derivatives.descriptor,
        orbital_gradient: gradient,
        orbital_direction: direction,
    })
}

fn scaled_residual(residuals: (f64, f64), config: StationaryFloOptions) -> f64 {
    (residuals.0 / config.orbital_tolerance).max(residuals.1 / config.descriptor_tolerance)
}

pub(super) fn optimize(
    model: &MolecularDensityModel<'_>,
    seed: &ScfResult,
    descriptors: [&[Vec3]; 2],
    config: StationaryFloOptions,
) -> Result<FloDescriptorRun> {
    if config.max_iterations == 0
        || [config.orbital_tolerance, config.descriptor_tolerance]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(DftError::InvalidInput(
            "self-consistent FLO-SIC needs positive gradient tolerances and iteration count".into(),
        ));
    }
    let mut orbitals = seed.mo_coeff.clone();
    let mut descriptors = descriptors.map(<[Vec3]>::to_vec);
    let mut current = point(model, seed, &orbitals, [&descriptors[0], &descriptors[1]])?;
    let mut evaluations = 1;
    let mut completed = 0;
    let mut radius = 0.25_f64;
    for iteration in 0..=config.max_iterations {
        // FOD and occupied-virtual rotations have very different curvature
        // scales. Relax descriptors in their own coordinates before an
        // orbital step, reusing the frozen-subspace objective and gradient.
        if current.residuals().1 > config.descriptor_tolerance {
            let occupied = occupied_blocks(&orbitals, [seed.n_alpha, seed.n_beta]);
            let (relaxed, optimization) = optimize_coordinates(
                model,
                seed,
                &occupied,
                [&descriptors[0], &descriptors[1]],
                config.descriptor_tolerance,
                config.max_iterations.min(32),
            )?;
            evaluations += optimization.evaluations;
            if optimization.iterations != 0 {
                descriptors = relaxed;
                current = point(model, seed, &orbitals, [&descriptors[0], &descriptors[1]])?;
                evaluations += 1;
            }
        }
        let residuals = current.residuals();
        let residual = scaled_residual(residuals, config);
        completed = iteration;
        crate::logger::log(&format!(
            "[flo-sic] iter {iteration:>3} E={:.12} orbital={:.3e} FOD={:.3e}",
            current.state.energy, residuals.0, residuals.1
        ));
        if residuals.0 <= config.orbital_tolerance || iteration == config.max_iterations {
            break;
        }
        let orbital_slope: f64 = current
            .orbital_gradient
            .iter()
            .zip(&current.orbital_direction)
            .map(|(g, d)| g.frobenius_dot(d))
            .sum();
        let max_rotation = current
            .orbital_direction
            .iter()
            .flat_map(Matrix::as_slice)
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        let mut step = (radius * 1.5).min(0.2 / max_rotation);
        let mut accepted = None;
        for _ in 0..32 {
            let candidate_orbitals = rotate(&orbitals, &current.orbital_direction, step);
            let trial = point(
                model,
                seed,
                &candidate_orbitals,
                [&descriptors[0], &descriptors[1]],
            )?;
            evaluations += 1;
            let difference = (trial.state.parent_energy - current.state.parent_energy)
                + (trial.state.correction - current.state.correction);
            if difference <= 1e-4 * step * orbital_slope
                || difference.abs() <= 8.0 * f64::EPSILON * current.state.energy.abs()
                    && scaled_residual(trial.residuals(), config) < 0.95 * residual
            {
                accepted = Some((candidate_orbitals, trial));
                break;
            }
            step *= 0.5;
        }
        let Some((new_orbitals, trial)) = accepted else {
            break;
        };
        orbitals = new_orbitals;
        current = trial;
        radius = step;
    }
    let residuals = current.residuals();
    Ok(FloDescriptorRun {
        state: current.state,
        descriptors,
        descriptor_gradient: Some(current.descriptor_gradient),
        optimization: None,
        stationarity: Some(FloStationarity {
            converged: scaled_residual(residuals, config) <= 1.0,
            iterations: completed,
            evaluations,
            orbital_residual: residuals.0,
            descriptor_residual: residuals.1,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::c1_geometry::assert_c1;
    use crate::acceptance::stencils::{fit_components_staged, Stencil};
    use crate::driver::method::{prepare_molecular, MolecularMethod};
    use crate::scf::sic::flo_sic::with_uks_sic_reference;

    #[test]
    fn distorted_hydrogen_dimers_fit_every_fod_and_occupied_virtual_derivative() {
        let positions = [
            [0.13, -0.27, 0.19],
            [1.77, 0.36, -0.08],
            [2.44, 2.35, 1.20],
            [3.97, 2.88, 1.67],
        ];
        assert_c1(&[1; 4], &positions);
        let setup = prepare_molecular(
            &[1; 4],
            &positions,
            &MolecularMethod {
                xc: Some("pbe".into()),
                unrestricted: true,
                grid_level: 0,
                grid_weight_cutoff: 0.0,
                dispersion: "none".into(),
                ..MolecularMethod::default()
            },
        )
        .unwrap();
        let descriptors = [
            [Vec3::new(0.68, 0.07, -0.09), Vec3::new(3.41, 2.31, 1.29)],
            [Vec3::new(0.79, 0.03, 0.11), Vec3::new(3.29, 2.55, 1.44)],
        ];
        with_uks_sic_reference(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            "FLO-SIC derivative acceptance",
            |model, seed| {
                let orbitals = &seed.mo_coeff;
                let base = point(model, seed, orbitals, [&descriptors[0], &descriptors[1]])?;
                // Each energy is independently evaluated after moving a FOD
                // or rotating one occupied-virtual pair in a single spin.
                let relative_energy = |orbitals: &[Matrix; 2], d: &[[Vec3; 2]; 2]| -> Result<f64> {
                    let occupied = occupied_blocks(orbitals, [2, 2]);
                    let density = occupied.each_ref().map(|c| c.leading_columns_gram(2, 1.0));
                    let parent = model.evaluate(&density)?.electronic_energy;
                    let (state, _) = evaluate_descriptors(model, seed, &occupied, [&d[0], &d[1]], false, 0.0)?;
                    Ok((parent - (seed.energy - seed.e_nuc)) + state.correction)
                };
                let mut labels = Vec::new();
                let mut derivatives = Vec::new();
                let mut oracle: Vec<Box<dyn Fn(f64) -> f64 + '_>> = Vec::new();
                for spin in 0..2 {
                    for descriptor in 0..2 {
                        for axis in 0..3 {
                            labels.push(format!("FOD spin={spin} descriptor={descriptor} axis={axis}"));
                            derivatives.push(base.descriptor_gradient[spin][descriptor][axis]);
                            oracle.push(Box::new(move |h| {
                                let mut moved = descriptors;
                                let mut position = moved[spin][descriptor].to_array();
                                position[axis] += h;
                                moved[spin][descriptor] = Vec3::new(position[0], position[1], position[2]);
                                relative_energy(orbitals, &moved).unwrap()
                            }));
                        }
                    }
                    for i in 0..2 {
                        for a in 2..orbitals[spin].cols {
                            labels.push(format!("occupied-virtual spin={spin} a={a} i={i}"));
                            derivatives.push(base.orbital_gradient[spin][(a, i)]);
                            oracle.push(Box::new(move |h| {
                                let mut direction = orbitals.each_ref().map(|c| Matrix::zeros(c.cols, c.cols));
                                direction[spin][(a, i)] = 1.0;
                                relative_energy(&rotate(orbitals, &direction, h), &descriptors).unwrap()
                            }));
                        }
                    }
                }
                fn estimate(path: &dyn Fn(f64) -> f64, h: f64, stencil: Stencil) -> f64 {
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (path(2.0 * h), path(-2.0 * h))
                    } else { (0.0, 0.0) };
                    stencil.first(path(h), path(-h), plus2, minus2, h)
                }
                let mut failures = Vec::new();
                fit_components_staged(
                    &labels,
                    &[
                        ([0.012, 0.008, 0.005, 0.003], Stencil::Second),
                        ([0.006, 0.004, 0.0025, 0.0015], Stencil::Second),
                        ([0.032, 0.024, 0.016, 0.010], Stencil::Fourth),
                    ],
                    |steps, stencil| oracle.iter().zip(&derivatives).map(|(path, &target)| {
                        steps.map(|h| (estimate(path.as_ref(), h, stencil) - target).abs())
                    }).collect(),
                    &mut failures,
                );
                assert!(failures.is_empty(), "{failures:?}");
                for ((name, path), target) in labels.iter().zip(oracle).zip(derivatives) {
                    let estimate = estimate(path.as_ref(), 0.002, Stencil::Fourth);
                    assert!((estimate - target).abs() < 2e-8, "{name} {estimate:.12e} {target:.12e}");
                }
                let relaxed = optimize(
                    model, seed, [&descriptors[0], &descriptors[1]],
                    StationaryFloOptions { orbital_tolerance: 1e-6, descriptor_tolerance: 1e-6, max_iterations: 12 },
                )?;
                let diag = relaxed.stationarity.unwrap();
                assert!(relaxed.state.energy < base.state.energy - 1e-6);
                assert!(relaxed.state.density_residual < 1e-12);
                println!("C1 H4 joint FLO-SIC E={:.12e} initial={:.12e} orbital={:.4e} FOD={:.4e} cycles={}",
                    relaxed.state.energy, base.state.energy, diag.orbital_residual, diag.descriptor_residual, diag.iterations);
                Ok(())
            },
        ).unwrap();
    }
}
