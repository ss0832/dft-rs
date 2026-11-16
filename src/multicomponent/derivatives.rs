// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic coupled-species geometry derivatives through the shared moving
//! Gaussian/quadrature provider and the variational coupled orbital Hessian.

use super::correlation::CorrelationModel;
use super::response::CoupledResponseSolver;
use super::{add, append_basis, block, MulticomponentResult, QuantumParticle};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::response::nuclear::{
    matrix_product, metric_transport, product, GeometrySources, GeometryTaylor, ParticleSource,
};
use crate::response::ResponseSolverConfig;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Directional derivatives with respect to nuclear sites and independent
/// added-particle Gaussian sites. The path also moves any expectation-position
/// target tied to the displaced basis site.
#[derive(Clone, Debug)]
pub struct CoupledGeometryDerivative {
    pub gradient: f64,
    pub hessian: f64,
    pub explicit_hessian: f64,
    pub orbital_relaxation: f64,
    pub reference_energy_error: f64,
    pub response_residual: f64,
    pub reference_residual: f64,
    pub iterations: usize,
    pub converged: bool,
    pub position_response_residual: f64,
}

#[derive(Clone, Debug)]
pub struct CoupledGeometryHessian {
    pub gradient: Vec<[f64; 3]>,
    pub hessian: Option<Matrix>,
    pub directional_evaluations: usize,
    pub response_residuals: Vec<f64>,
    pub position_response_residuals: Vec<f64>,
    pub reference_energy_errors: Vec<f64>,
    pub converged: bool,
}

fn site_count(molecule: &Molecule, particles: &[QuantumParticle]) -> usize {
    molecule.len()
        + particles
            .iter()
            .filter(|p| p.replaced_atom.is_none())
            .count()
}

// Complete fixed-laboratory-point AO jets. Gaussian-centre motion is the
// negative spatial derivative; second Taylor coefficients include 1/2.
// Public AO derivatives avoid building a large per-point symbolic tape.
fn fixed_epc_grid_jets(
    basis: &MolBasis,
    points: &[crate::math::Vec3],
    weights: &[f64],
    direction: &[[f64; 3]],
) -> Vec<crate::response::nuclear::geometry::GridPointTaylor> {
    use crate::grid::eval::{eval_aos_deriv3, third_index};
    let values = eval_aos_deriv3(basis, points, true, true, true);
    let grad = values.grad.as_ref().unwrap();
    let hess = values.hess.as_ref().unwrap();
    let third = values.third.as_ref().unwrap();
    let second = |i: usize, j: usize| match (i.min(j), i.max(j)) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        _ => 5,
    };
    let mut directions = vec![[0.0; 3]; basis.nao_sph];
    for shell in &basis.shells {
        for d in directions
            .iter_mut()
            .skip(shell.sph_off)
            .take(crate::basis::n_sph(shell.l))
        {
            *d = direction[shell.atom];
        }
    }
    (0..points.len())
        .map(|g| {
            let aos = (0..basis.nao_sph)
                .map(|mu| {
                    let d = directions[mu];
                    let mut phi = vec![values.phi[(g, mu)], 0.0, 0.0];
                    for i in 0..3 {
                        phi[1] -= d[i] * grad[i][(g, mu)];
                        for j in 0..3 {
                            phi[2] += 0.5 * d[i] * d[j] * hess[second(i, j)][(g, mu)];
                        }
                    }
                    let mut out = std::array::from_fn(|axis| {
                        if axis == 0 {
                            phi.clone()
                        } else {
                            let a = axis - 1;
                            let mut v = vec![grad[a][(g, mu)], 0.0, 0.0];
                            for i in 0..3 {
                                v[1] -= d[i] * hess[second(a, i)][(g, mu)];
                                for j in 0..3 {
                                    v[2] +=
                                        0.5 * d[i] * d[j] * third[third_index(a, i, j)][(g, mu)];
                                }
                            }
                            v
                        }
                    });
                    // Avoid retaining a second temporary value vector per AO.
                    out[0] = phi;
                    out
                })
                .collect();
            crate::response::nuclear::geometry::GridPointTaylor {
                weight: vec![weights[g], 0.0, 0.0],
                aos,
            }
        })
        .collect()
}

fn pad(p: &Matrix, offset: usize, n: usize) -> Matrix {
    let mut out = Matrix::zeros(n, n);
    for i in 0..p.rows {
        for j in 0..p.cols {
            out[(offset + i, offset + j)] = p[(i, j)];
        }
    }
    out
}

/// First and second total energy derivatives in one geometric direction.
/// Directions list the nuclear sites followed by the independently added
/// particles in particle order. A nuclear-only direction holds the added
/// centers fixed. For a position-constrained reference, each target follows
/// its basis site, and both the multiplier and constraint are differentiated.
/// All density, overlap, cross-Coulomb, electronic XC and interspecies XC
/// terms use analytic coefficients. No displaced SCF or finite difference is
/// used. Scalar nuclear/electronic masses and particle charges are held fixed.
#[allow(clippy::too_many_arguments)]
pub fn directional(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    state: &MulticomponentResult,
    direction: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Result<CoupledGeometryDerivative> {
    directional_impl(
        molecule,
        electronic_basis,
        options,
        functional,
        grid,
        particles,
        correlation,
        state,
        direction,
        config,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn directional_fixed_grid(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: &MolecularGrid,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    state: &MulticomponentResult,
    direction: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Result<CoupledGeometryDerivative> {
    directional_impl(
        molecule,
        electronic_basis,
        options,
        functional,
        Some(grid),
        particles,
        correlation,
        state,
        direction,
        config,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn directional_impl(
    molecule: &Molecule,
    electronic_basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    state: &MulticomponentResult,
    direction: &[[f64; 3]],
    config: &ResponseSolverConfig,
    fixed_grid: bool,
) -> Result<CoupledGeometryDerivative> {
    config.validate()?;
    let atoms = molecule.len();
    let sites = site_count(molecule, particles);
    if ![atoms, sites].contains(&direction.len())
        || direction.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            format!("coupled geometry directions need {atoms} nuclear displacements or {sites} nuclear and added-particle displacements"),
        ));
    }
    let mut displacements = direction.to_vec();
    displacements.resize(sites, [0.0; 3]);
    if !electronic_basis.ecps.is_empty()
        || options.vv10.is_some()
        || options.density_fit.is_some()
        || functional.required_nonlocal_term().is_some()
    {
        return Err(DftError::Unsupported {
            feature:
                "coupled quantum-particle geometry response with ECP, RI or nonlocal correlation"
                    .into(),
            missing: "the corresponding moving-species integral/nonlocal derivative provider"
                .into(),
        });
    }
    let solver = CoupledResponseSolver::new(
        molecule,
        electronic_basis,
        options,
        functional,
        grid,
        particles,
        correlation,
        state,
    )?;
    let mut combined = electronic_basis.clone();
    let mut offsets = vec![0];
    let mut sources = vec![
        ParticleSource {
            block: 0,
            mass: 1.0,
            charge: -1.0
        };
        electronic_basis.nao_sph
    ];
    let mut charges: Vec<_> = molecule.atoms.iter().map(|a| f64::from(a.z)).collect();
    let mut additional_centers = Vec::with_capacity(sites - atoms);
    let mut particle_sites = Vec::with_capacity(particles.len());
    for (i, particle) in particles.iter().enumerate() {
        let site = if let Some(atom) = particle.replaced_atom {
            *charges.get_mut(atom).ok_or_else(|| {
                DftError::InvalidInput(
                    "quantum-nucleus geometry owner is outside the molecule".into(),
                )
            })? = 0.0;
            atom
        } else {
            let center = particle
                .basis
                .shells
                .first()
                .ok_or_else(|| {
                    DftError::InvalidInput(
                        "an independent particle basis needs a Gaussian center".into(),
                    )
                })?
                .center;
            let site = atoms + additional_centers.len();
            additional_centers.push((center, displacements[site]));
            site
        };
        particle_sites.push(site);
        offsets.push(combined.nao_sph);
        let mut basis = particle.basis.clone();
        for shell in &mut basis.shells {
            shell.atom = site;
        }
        append_basis(&mut combined, &basis);
        sources.extend(vec![
            ParticleSource {
                block: i + 1,
                mass: particle.mass,
                charge: particle.charge
            };
            basis.nao_sph
        ]);
    }
    let source = GeometrySources {
        charges: &charges,
        particles: &sources,
        force_grid: !correlation.is_zero() && !fixed_grid,
        additional_centers: &additional_centers,
    };
    let geometry = GeometryTaylor::with_sources(
        molecule,
        &combined,
        if fixed_grid { None } else { grid },
        functional,
        options.skala.as_ref(),
        &displacements[..atoms],
        options.external_field.map_or([0.0; 3], |f| f.to_array()),
        [0.0; 3],
        2,
        options.incore_mb,
        Some(&source),
    )?;
    let ne = electronic_basis.nao_sph;
    let n = combined.nao_sph;
    // Two spin frames are retained in the physical density even for RKS.
    let mut b = Vec::new();
    let mut densities = Vec::new();
    for spin in 0..2 {
        let c = &state.electronic_reference.mo_coeff[spin];
        let overlap: Vec<_> = geometry.overlap.iter().map(|s| block(s, 0, ne)).collect();
        let frame = metric_transport(c, &overlap, 2);
        let mut d = vec![Matrix::zeros(c.cols, c.cols); 3];
        for i in 0..[
            state.electronic_reference.n_alpha,
            state.electronic_reference.n_beta,
        ][spin]
        {
            d[0][(i, i)] = 1.0;
        }
        let bt: Vec<_> = frame.iter().map(Matrix::transpose).collect();
        densities.push(matrix_product(&matrix_product(&frame, &d, 2), &bt, 2));
        b.push(frame);
    }
    for i in 0..particles.len() {
        let f = &solver.frames[solver.electronic_blocks + i];
        let dim = particles[i].basis.nao_sph;
        let overlap: Vec<_> = geometry
            .overlap
            .iter()
            .map(|s| block(s, offsets[i + 1], dim))
            .collect();
        let frame = metric_transport(&f.coefficients, &overlap, 2);
        let mut d = vec![Matrix::zeros(f.coefficients.cols, f.coefficients.cols); 3];
        d[0][(0, 0)] = 1.0;
        let bt: Vec<_> = frame.iter().map(Matrix::transpose).collect();
        densities.push(matrix_product(&matrix_product(&frame, &d, 2), &bt, 2));
        b.push(frame);
    }
    let electronic: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| densities[s].iter().map(|p| pad(p, 0, n)).collect());
    let evaluated = geometry.evaluate(functional, &electronic, 2)?;
    let mut energy: Vec<_> = evaluated.energy.iter().map(|e| e.total()).collect();
    let mut position_source = vec![0.0; 3 * particles.len()];
    let mut fock: Vec<Vec<Matrix>> = (0..2)
        .map(|s| {
            evaluated.fock[s][..2]
                .iter()
                .map(|f| block(f, 0, ne))
                .collect()
        })
        .collect();
    let mut electron_total = densities[0].clone();
    for k in 0..3 {
        add(&mut electron_total[k], &densities[1][k], 1.0);
    }
    let mut physical = vec![electron_total];
    physical.extend(densities[2..].iter().cloned());
    let eri = geometry.coulomb_coefficients();
    let coulomb: Vec<Vec<Matrix>> = physical
        .iter()
        .enumerate()
        .map(|(s, density)| {
            let offset = offsets[s];
            (0..3)
                .map(|k| {
                    let mut j = Matrix::zeros(n, n);
                    for mu in 0..n {
                        for nu in 0..n {
                            for a in 0..density[0].rows {
                                for d in 0..density[0].cols {
                                    for t in 0..=k {
                                        let index =
                                            ((mu * n + nu) * n + offset + a) * n + offset + d;
                                        j[(mu, nu)] += eri[t][index] * density[k - t][(a, d)];
                                    }
                                }
                            }
                        }
                    }
                    j
                })
                .collect()
        })
        .collect();
    for i in 0..particles.len() {
        let offset = offsets[i + 1];
        let dim = particles[i].basis.nao_sph;
        let charge = particles[i].charge;
        let core: Vec<_> = geometry
            .core_coefficients()
            .iter()
            .map(|h| block(h, offset, dim))
            .collect();
        let mut qf = core[..2].to_vec();
        for k in 0..3 {
            for t in 0..=k {
                energy[k] += densities[i + 2][t].frobenius_dot(&core[k - t]);
                energy[k] -=
                    charge * physical[0][t].frobenius_dot(&block(&coulomb[i + 1][k - t], 0, ne));
                for j in 0..i {
                    energy[k] += charge
                        * particles[j].charge
                        * densities[i + 2][t].frobenius_dot(&block(
                            &coulomb[j + 1][k - t],
                            offset,
                            dim,
                        ));
                }
            }
            if k < 2 {
                for e in &mut fock[..2] {
                    add(&mut e[k], &block(&coulomb[i + 1][k], 0, ne), -charge);
                }
                add(&mut qf[k], &block(&coulomb[0][k], offset, dim), -charge);
                for j in 0..particles.len() {
                    if i != j {
                        add(
                            &mut qf[k],
                            &block(&coulomb[j + 1][k], offset, dim),
                            charge * particles[j].charge,
                        );
                    }
                }
            }
        }
        fock.push(qf);
    }
    if let Some(constraints) = &state.position_constraints {
        for (i, &site) in particle_sites.iter().enumerate() {
            let offset = offsets[i + 1];
            let dim = particles[i].basis.nao_sph;
            for axis in 0..3 {
                let target = constraints.targets[i][axis];
                let velocity = displacements[site][axis];
                let mut operators = Vec::new();
                for k in 0..3 {
                    let mut operator = block(&geometry.dipoles[axis][k], offset, dim);
                    add(
                        &mut operator,
                        &block(&geometry.overlap[k], offset, dim),
                        -target,
                    );
                    if k > 0 {
                        add(
                            &mut operator,
                            &block(&geometry.overlap[k - 1], offset, dim),
                            -velocity,
                        );
                    }
                    operators.push(operator);
                }
                for k in 0..3 {
                    let c: f64 = (0..=k)
                        .map(|t| densities[i + 2][t].frobenius_dot(&operators[k - t]))
                        .sum();
                    energy[k] += constraints.multipliers[i][axis] * c;
                    if k == 1 {
                        position_source[3 * i + axis] = c;
                    }
                    if k < 2 {
                        add(
                            &mut fock[i + 2][k],
                            &operators[k],
                            constraints.multipliers[i][axis],
                        );
                    }
                }
            }
        }
    }
    if !correlation.is_zero() {
        let channels = correlation.len();
        let mut epc_sum = [0.0_f64; 3];
        let mut epc_remainder = [0.0_f64; 3];
        let mut integrate =
            |points: &[crate::response::nuclear::geometry::GridPointTaylor]| -> Result<()> {
                for point in points {
                    // Path coefficients of the electron density (index 0) and of the
                    // partner density of every correlation channel.
                    let mut rho = vec![[0.0; 3]; 1 + channels];
                    for s in 0..b.len() {
                        let channel = if s < 2 {
                            0
                        } else if let Some(c) = particles[s - 2].correlation_channel {
                            1 + c
                        } else {
                            continue;
                        };
                        let offset = offsets[s.saturating_sub(1)];
                        let occupied = if s < 2 {
                            [
                                state.electronic_reference.n_alpha,
                                state.electronic_reference.n_beta,
                            ][s]
                        } else {
                            1
                        };
                        // Orbital products preserve positivity and exact nodal
                        // tangents without flooring a physical density or kernel.
                        for i in 0..occupied {
                            let mut wave = [0.0; 3];
                            for a in 0..b[s][0].rows {
                                for k in 0..3 {
                                    for t in 0..=k {
                                        wave[k] +=
                                            point.aos[offset + a][0][t] * b[s][k - t][(a, i)];
                                    }
                                }
                            }
                            for k in 0..3 {
                                for t in 0..=k {
                                    rho[channel][k] += wave[t] * wave[k - t];
                                }
                            }
                        }
                    }
                    // Path coefficients of a density that underflowed to zero are zero
                    // (see `correlation::representable_tangent`).
                    let mut tangent = rho.clone();
                    for value in tangent.iter_mut() {
                        if value[0] == 0.0 {
                            value[1] = 0.0;
                            value[2] = 0.0;
                        }
                    }
                    // Potentials `[first, kernel action]` of the electrons and of
                    // every channel.
                    let mut potentials = vec![[0.0; 2]; 1 + channels];
                    let mut e = [0.0; 3];
                    for (c, provider) in correlation.channels().iter().enumerate() {
                        if provider.is_zero() {
                            continue;
                        }
                        let (electron, partner) = (&tangent[0], &tangent[1 + c]);
                        let value = provider.evaluate(rho[0][0], rho[1 + c][0])?;
                        let delta = provider.kernel_action(
                            rho[0][0],
                            rho[1 + c][0],
                            [electron[1], partner[1]],
                        )?;
                        e[0] += value.energy;
                        e[1] += value.electron_potential * electron[1]
                            + value.partner_potential * partner[1];
                        e[2] += value.electron_potential * electron[2]
                            + value.partner_potential * partner[2]
                            + 0.5 * (electron[1] * delta[0] + partner[1] * delta[1]);
                        potentials[0][0] += value.electron_potential;
                        potentials[0][1] += delta[0];
                        potentials[1 + c] = [value.partner_potential, delta[1]];
                    }
                    for k in 0..3 {
                        for t in 0..=k {
                            let contribution = point.weight[t] * e[k - t];
                            if fixed_grid {
                                let next = epc_sum[k] + contribution;
                                epc_remainder[k] += if epc_sum[k].abs() >= contribution.abs() {
                                    (epc_sum[k] - next) + contribution
                                } else {
                                    (contribution - next) + epc_sum[k]
                                };
                                epc_sum[k] = next;
                            } else {
                                energy[k] += contribution;
                            }
                        }
                    }
                    for s in 0..fock.len() {
                        let potential = if s < 2 {
                            potentials[0]
                        } else if let Some(c) = particles[s - 2].correlation_channel {
                            potentials[1 + c]
                        } else {
                            continue;
                        };
                        let block_id = s.saturating_sub(1);
                        let offset = offsets[block_id];
                        for a in 0..fock[s][0].rows {
                            for d in 0..fock[s][0].cols {
                                let ao = product(
                                    &point.aos[offset + a][0],
                                    &point.aos[offset + d][0],
                                    1,
                                );
                                let wa = product(&point.weight, &ao, 1);
                                for k in 0..2 {
                                    for t in 0..=k {
                                        fock[s][k][(a, d)] += wa[t] * potential[k - t];
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            };
        if fixed_grid {
            let grid =
                grid.ok_or_else(|| DftError::InvalidInput("missing frozen EPC grid".into()))?;
            let _working_memory = crate::resources::reserve(
                256 * combined.nao_sph * 512,
                "fixed EPC geometry batch",
            )?;
            for start in (0..grid.len()).step_by(256) {
                let end = (start + 256).min(grid.len());
                let jets = fixed_epc_grid_jets(
                    &combined,
                    &grid.points[start..end],
                    &grid.weights[start..end],
                    &displacements,
                );
                integrate(&jets)?;
            }
        } else {
            integrate(geometry.grid_points())?;
        }
        if fixed_grid {
            for k in 0..3 {
                energy[k] += epc_sum[k] + epc_remainder[k];
            }
        }
    }
    let effective_source: Vec<_> = fock
        .iter()
        .enumerate()
        .map(|(s, f)| {
            let bt: Vec<_> = b[s].iter().map(Matrix::transpose).collect();
            let transported = matrix_product(&matrix_product(&bt[..2], f, 1), &b[s][..2], 1);
            let block_id = s.saturating_sub(1);
            let offset = offsets[block_id];
            let overlap = block(&geometry.overlap[0], offset, f[0].rows);
            let sc = overlap.matmul_seq(&b[s][0]);
            sc.matmul_seq(&transported[1]).matmul_transpose_seq(&sc)
        })
        .collect();
    let e_source = [effective_source[0].clone(), effective_source[1].clone()];
    let q_source = effective_source[2..].to_vec();
    let source = solver.project(&e_source, &q_source);
    let response = solver.solve_position_sources(&e_source, &q_source, &position_source, config)?;
    let mut relaxation = 2.0
        * source
            .iter()
            .zip(&response.amplitudes)
            .map(|(b, u)| b * u)
            .sum::<f64>();
    relaxation += response
        .position_multipliers
        .iter()
        .flatten()
        .zip(&position_source)
        .map(|(l, c)| l * c)
        .sum::<f64>();
    let error = energy[0] - state.energy;
    Ok(CoupledGeometryDerivative {
        gradient: energy[1],
        hessian: 2.0 * energy[2] + relaxation,
        explicit_hessian: 2.0 * energy[2],
        orbital_relaxation: relaxation,
        reference_energy_error: error,
        response_residual: response.relative_residual,
        reference_residual: response.reference_residual,
        iterations: response.iterations,
        converged: response.converged && error.abs() <= config.acceptance(),
        position_response_residual: response.position_residual,
    })
}

/// Full analytic tensor in laboratory Cartesian coordinates. Off-diagonal
/// entries use polarization of independently solved directional derivatives;
/// finite differences or copied symmetry values do not generate a derivative.
#[allow(clippy::too_many_arguments)]
pub fn hessian(
    molecule: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    state: &MulticomponentResult,
    config: &ResponseSolverConfig,
) -> Result<CoupledGeometryHessian> {
    cartesian(
        molecule,
        basis,
        options,
        functional,
        grid,
        particles,
        correlation,
        state,
        2,
        config,
    )
}

/// Cartesian analytic gradient (order 1), with the full relaxed Hessian for
/// order 2. Coordinates list the nuclear sites then added Gaussian sites.
#[allow(clippy::too_many_arguments)]
pub fn cartesian(
    molecule: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    particles: &[QuantumParticle],
    correlation: &CorrelationModel,
    state: &MulticomponentResult,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<CoupledGeometryHessian> {
    if order != 1 && order != 2 {
        return Err(DftError::InvalidInput(
            "coupled Cartesian derivative order must be 1 or 2".into(),
        ));
    }
    let sites = site_count(molecule, particles);
    let n = 3 * sites;
    let mut hessian = Matrix::zeros(n, n);
    let mut gradient = vec![[0.0; 3]; sites];
    let mut directions = Vec::new();
    let mut residuals = Vec::new();
    let mut position_residuals = Vec::new();
    let mut errors = Vec::new();
    let mut converged = true;
    for i in 0..n {
        let mut d = vec![[0.0; 3]; sites];
        d[i / 3][i % 3] = 1.0;
        let r = directional(
            molecule,
            basis,
            options,
            functional,
            grid,
            particles,
            correlation,
            state,
            &d,
            config,
        )?;
        gradient[i / 3][i % 3] = r.gradient;
        hessian[(i, i)] = r.hessian;
        residuals.push(r.response_residual);
        position_residuals.push(r.position_response_residual);
        errors.push(r.reference_energy_error);
        converged &= r.converged;
        directions.push(d);
    }
    let mut evaluations = n;
    for i in 0..if order == 2 { n } else { 0 } {
        for j in 0..i {
            let mut d = directions[i].clone();
            d[j / 3][j % 3] += 1.0;
            let r = directional(
                molecule,
                basis,
                options,
                functional,
                grid,
                particles,
                correlation,
                state,
                &d,
                config,
            )?;
            let value = 0.5 * (r.hessian - hessian[(i, i)] - hessian[(j, j)]);
            hessian[(i, j)] = value;
            hessian[(j, i)] = value;
            evaluations += 1;
            residuals.push(r.response_residual);
            position_residuals.push(r.position_response_residual);
            errors.push(r.reference_energy_error);
            converged &= r.converged;
        }
    }
    Ok(CoupledGeometryHessian {
        gradient,
        hessian: if order == 2 { Some(hessian) } else { None },
        directional_evaluations: evaluations,
        response_residuals: residuals,
        position_response_residuals: position_residuals,
        reference_energy_errors: errors,
        converged,
    })
}
