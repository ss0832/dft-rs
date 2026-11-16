// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Restricted open-shell Kohn-Sham / Hartree-Fock self-consistent field.
//!
//! One set of spatial orbitals holds `N_beta` doubly occupied (closed) and
//! `N_alpha - N_beta` singly occupied (open) orbitals. The energy is the
//! spin-polarized functional of `P_alpha` (closed plus open) and `P_beta`
//! (closed), stationary with respect to the closed-open, open-virtual and
//! closed-virtual rotations. In the basis of the current orbitals the
//! effective Fock matrix is
//!
//! ```text
//!            closed          open            virtual
//! closed   (Fa + Fb)/2       Fb            (Fa + Fb)/2
//! open        Fb         (Fa + Fb)/2          Fa
//! virtual  (Fa + Fb)/2       Fa            (Fa + Fb)/2
//! ```
//!
//! whose off-diagonal blocks are the orbital gradient. The iteration starts
//! from the alpha orbitals of the unrestricted solution and extrapolates the
//! effective Fock matrix with the off-diagonal blocks as the error vector.
//! Near convergence or after stagnation, a constrained common-orbital Newton
//! proposal uses the complete density-response kernel and an actual-energy
//! line search. It reuses the SCF rotation and truncated-Newton machinery.
//!
//! The result is returned with the occupied and the virtual space of each
//! spin separately diagonalized in its own Fock matrix. The two densities
//! are unchanged by that, and because the closed-virtual blocks of the two
//! spins cancel and the other mixed blocks vanish at convergence, the
//! energy-weighted density `sum_s P_s F_s P_s` of those orbitals is the one
//! of the analytic gradient.

use super::fock::JkEngine;
use super::{occupied_density, run_scf, Reference, ScfOptions, ScfResult};
use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::MolecularGrid;
use crate::integrals::CoulombKernel;
use crate::linalg::{solve_linear, symmetric_eigen, Matrix, Orthogonalizer};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

pub(crate) mod thermal;

/// Fock matrices and energies of one orbital set.
struct Build {
    fock: [Matrix; 2],
    density: [Matrix; 2],
    electronic: f64,
    exc: f64,
    nlc: f64,
    pcm: f64,
    grid_electrons: Option<f64>,
}

fn add_scaled(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *t += scale * s;
    }
}

/// Eigenvectors of the block `rows` of `m`, as a rotation of those orbitals.
fn block_eigen(m: &Matrix, start: usize, count: usize) -> Result<(Vec<f64>, Matrix)> {
    let mut block = Matrix::zeros(count, count);
    for i in 0..count {
        for j in 0..count {
            block[(i, j)] = 0.5 * (m[(start + i, start + j)] + m[(start + j, start + i)]);
        }
    }
    symmetric_eigen(&block)
}

/// Restricted open-shell self-consistent field of the state of `mol`
/// (its multiplicity gives `N_alpha - N_beta`).
pub fn run_roks(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
) -> Result<ScfResult> {
    Ok(run_roks_with_orbitals(mol, mb, opts, functional, grid)?.0)
}

/// [`run_roks`] together with the common spatial orbitals: the closed
/// orbitals first, then the open ones, then the virtual ones.
pub fn run_roks_with_orbitals(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
) -> Result<(ScfResult, Matrix)> {
    if opts.electronic_temperature > 0.0 {
        return Err(DftError::Unsupported {
            feature: "integer common-orbital response at finite temperature".into(),
            missing: "the coupled shared-orbital and occupation response".into(),
        });
    }
    run_roks_with_extension(mol, mb, opts, functional, grid, None)
}

/// Optimize a molecular Mermin ensemble with one shared spatial frame and
/// fixed alpha and beta electron counts. Spin populations are returned in
/// that frame, without semicanonicalizing the two spins independently.
pub fn run_thermal_roks(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
) -> Result<ScfResult> {
    if !(opts.electronic_temperature.is_finite() && opts.electronic_temperature > 0.0) {
        return Err(DftError::InvalidInput(
            "thermal common-orbital SCF requires a positive electronic temperature".into(),
        ));
    }
    Ok(run_roks_with_extension(mol, mb, opts, functional, grid, None)?.0)
}

pub(super) fn run_roks_in_spin_potential(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    spin_potential: Option<&[Matrix; 2]>,
) -> Result<(ScfResult, Matrix)> {
    run_roks_with_extension(
        mol,
        mb,
        opts,
        functional,
        grid,
        spin_potential.map(super::fields::DensityExtension::SpinPotential),
    )
}

pub(super) fn run_roks_with_extension(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    extension: Option<super::fields::DensityExtension<'_>>,
) -> Result<(ScfResult, Matrix)> {
    if opts.occupation.is_some() || opts.fixed_occupations.is_some() {
        return Err(DftError::Unsupported { feature: "restricted-open-shell determinant tracking".into(),
            missing: "the common closed/open/virtual occupation tracking constraint in the restricted-open-shell optimizer".into() });
    }

    if opts.electronic_temperature > 0.0 && extension.is_some() {
        return Err(DftError::Unsupported {
            feature: "extended restricted open-shell SCF with fractional occupations".into(),
            missing: "the external constraint or Hubbard contribution to the common-orbital thermal kernel".into(),
        });
    }
    let has_xc = !functional.parts.is_empty();
    if has_xc && grid.is_none() {
        return Err(DftError::InvalidInput(
            "restricted open-shell Kohn-Sham needs the molecular grid".into(),
        ));
    }
    // The unrestricted solution of the same state gives the core
    // Hamiltonian, the overlap and the starting orbitals.
    let mut start_options = opts.clone();
    start_options.reference = Reference::Uhf;
    let initialize = |options: &ScfOptions| {
        match extension {
            Some(super::fields::DensityExtension::SpinPotential(operators)) => {
                super::constrained::run_biased_collinear(
                    mol, mb, options, functional, grid, operators, None,
                )
            }
            Some(super::fields::DensityExtension::Hubbard(_)) => {
                super::run_scf_in_potential_with_completion(
                    mol, mb, options, functional, grid, None, None, None, extension, true,
                )
            }
            None => run_scf(mol, mb, options, functional, grid),
        }
    };
    let start = initialize(&start_options)?;
    let (n_alpha, n_beta) = (start.n_alpha, start.n_beta);
    if n_alpha == n_beta {
        // An empty open-shell class is the restricted closed-shell manifold.
        // The unrestricted initializer can be broken-symmetry even here.
        let mut closed_options = opts.clone();
        closed_options.reference = Reference::Rhf;
        let closed = initialize(&closed_options)?;
        let orbitals = closed.mo_coeff[0].clone();
        return Ok((closed, orbitals));
    }
    if n_alpha < n_beta {
        return Err(DftError::InvalidInput(
            "restricted common-orbital SCF requires N_alpha >= N_beta".into(),
        ));
    }
    let overlap = &start.overlap;
    let h_core = &start.h_core;
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let orth = Orthogonalizer::canonical(overlap, opts.orth_threshold)?;
    let jk = JkEngine::from_options(mol, mb, opts)?;
    let jk_lr = if clr != 0.0 {
        Some(JkEngine::new(
            mb,
            opts.eri_mode,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        )?)
    } else {
        None
    };
    // The cavity of the continuum and the grid of the nonlocal correlation
    // depend on the geometry only.
    let pcm_solver = match &opts.pcm {
        Some(p) => Some(crate::pcm::PcmSolver::build(
            mol,
            mb,
            &mb.effective_charges(mol),
            p,
        )?),
        None => None,
    };
    let nlc_grid = match opts.vv10 {
        Some(_) => Some(opts.nonlocal_grid(mol)?),
        None => None,
    };
    let density_model = super::fields::MolecularDensityModel {
        mol,
        mb,
        opts,
        functional,
        grid,
        h_core,
        extension,
        jk: &jk,
        jk_lr: jk_lr.as_ref(),
        pcm_solver: pcm_solver.as_ref(),
        nlc_grid: nlc_grid.as_ref(),
        restricted: false,
    };
    if opts.electronic_temperature > 0.0 {
        return thermal::solve(&density_model, start.clone());
    }
    let nmo = start.mo_coeff[0].cols;
    let (closed, open) = (n_beta, n_alpha - n_beta);
    let occupations = |count: usize| -> Vec<f64> {
        (0..nmo)
            .map(|i| if i < count { 1.0 } else { 0.0 })
            .collect()
    };
    let build = |c: &Matrix| -> Result<Build> {
        let density = [
            occupied_density(c, &occupations(n_alpha)),
            occupied_density(c, &occupations(n_beta)),
        ];
        if extension.is_some() {
            let fields = density_model.evaluate(&density)?;
            return Ok(Build {
                fock: fields.fock,
                density,
                electronic: fields.electronic_energy,
                exc: fields.xc_energy,
                nlc: fields.nonlocal_energy,
                pcm: fields.solvent_energy,
                grid_electrons: fields.grid_electron_count,
            });
        }
        let mut total = density[0].clone();
        add_scaled(&mut total, &density[1], 1.0);
        let (j, k) = jk.build(mb, &total, [&density[0], &density[1]], cx != 0.0)?;
        let k_lr = match &jk_lr {
            Some(engine) => Some(engine.build_k_only(mb, [&density[0], &density[1]])?),
            None => None,
        };
        let xc = match (has_xc, grid) {
            (true, Some(grid)) => Some(crate::ks::xc_contribution(
                functional,
                mb,
                grid,
                [&density[0], &density[1]],
            )?),
            _ => None,
        };
        let mut electronic = total.frobenius_dot(h_core) + 0.5 * total.frobenius_dot(&j);
        let mut fock = [h_core.clone(), h_core.clone()];
        for spin in 0..2 {
            add_scaled(&mut fock[spin], &j, 1.0);
            if cx != 0.0 {
                add_scaled(&mut fock[spin], &k[spin], -cx);
                electronic -= 0.5 * cx * density[spin].frobenius_dot(&k[spin]);
            }
            if let Some(k_lr) = &k_lr {
                add_scaled(&mut fock[spin], &k_lr[spin], -clr);
                electronic -= 0.5 * clr * density[spin].frobenius_dot(&k_lr[spin]);
            }
            if let Some(xc) = &xc {
                add_scaled(&mut fock[spin], &xc.vxc[spin], 1.0);
            }
        }
        let mut exc = xc.as_ref().map_or(0.0, |x| x.exc);
        if let (Some(model), Some(grid)) = (&opts.skala, grid) {
            let (e, v) =
                crate::skala::skala_fock(model, mol, mb, grid, [&density[0], &density[1]])?;
            exc += e;
            for spin in 0..2 {
                add_scaled(&mut fock[spin], &v[spin], 1.0);
            }
        }
        electronic += exc;
        let mut nlc = 0.0;
        if let (Some(params), Some(grid)) = (&opts.vv10, &nlc_grid) {
            let (e, v) = crate::nlc::vv10_potential(mb, grid, &total, *params)?;
            nlc = e;
            for matrix in fock.iter_mut() {
                add_scaled(matrix, &v, 1.0);
            }
        }
        let mut pcm = 0.0;
        if let Some(solver) = &pcm_solver {
            let (e, v) = solver.energy_and_fock(&total)?;
            pcm = e;
            for matrix in fock.iter_mut() {
                add_scaled(matrix, &v, 1.0);
            }
        }
        electronic += nlc + pcm;
        Ok(Build {
            fock,
            density,
            electronic,
            exc,
            nlc,
            pcm,
            grid_electrons: xc.as_ref().map(|x| x.n_elec),
        })
    };

    let class = |p: usize| -> usize {
        if p < closed {
            0
        } else if p < closed + open {
            1
        } else {
            2
        }
    };
    const HISTORY: usize = 8;
    let mut c = start.mo_coeff[0].clone();
    let mut history: Vec<(Matrix, Matrix)> = Vec::new();
    let mut last_energy = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut current = build(&c)?;
    let mut best_error = f64::INFINITY;
    let mut stagnant = 0;
    let mut orbital_active = false;
    let mut newton_available = opts.use_soscf;
    let mut trust_region = super::trah::TrustRegion::new(opts.soscf_trust)?;
    let mut common_orbital_residual = None;
    for iteration in 1..=opts.max_iter {
        iterations = iteration;
        let fa = c.matmul_accurate(&current.fock[0].matmul_accurate(&c, false), true);
        let fb = c.matmul_accurate(&current.fock[1].matmul_accurate(&c, false), true);
        let mut effective = Matrix::zeros(nmo, nmo);
        let mut gradient = Matrix::zeros(nmo, nmo);
        let mut error = 0.0_f64;
        for p in 0..nmo {
            for q in 0..nmo {
                let average = 0.5 * (fa[(p, q)] + fb[(p, q)]);
                let value = match (class(p), class(q)) {
                    (0, 1) | (1, 0) => 0.5 * (fb[(p, q)] + fb[(q, p)]),
                    (1, 2) | (2, 1) => 0.5 * (fa[(p, q)] + fa[(q, p)]),
                    _ => 0.5 * (average + 0.5 * (fa[(q, p)] + fb[(q, p)])),
                };
                effective[(p, q)] = value;
                if class(p) != class(q) {
                    gradient[(p, q)] = value;
                    error = error.max(value.abs());
                }
            }
        }
        let change = (current.electronic - last_energy).abs();
        crate::logger::log(&format!(
            "[roks-scf] iter {iteration:>3} E={:.12} dE={change:.3e} residual={error:.3e}",
            current.electronic
        ));
        if error < opts.conv_tol_grad && change < opts.e_tol {
            converged = true;
            common_orbital_residual = Some(error);
            break;
        }
        last_energy = current.electronic;
        if error < 0.7 * best_error {
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        best_error = best_error.min(error);
        if newton_available
            && iteration >= 4
            && (error < opts.soscf_from_error
                || stagnant >= 6
                || orbital_active
                || opts.orbital_optimizer == super::OrbitalOptimizer::Trah)
        {
            // Reuse the complete density kernel; the common orbital frame
            // retains the closed/open/virtual restriction throughout a step.
            let proposal = (|| -> Result<Option<(Matrix, Build)>> {
                let mut reference = start.clone();
                reference.density = current.density.clone();
                reference.fock = current.fock.clone();
                reference.mo_coeff = [c.clone(), c.clone()];
                let kernel = crate::response::ResponseSolver::kernel_with_engines(
                    mol,
                    mb,
                    functional,
                    grid,
                    &reference,
                    opts,
                    &jk,
                    jk_lr.as_ref(),
                )?;
                let fock_response = |variation: &[Matrix; 2]| -> Result<[Matrix; 2]> {
                    let mut response = kernel.fock_response(variation)?;
                    if let Some(extension) = extension {
                        extension.add_response(variation, &mut response)?;
                    }
                    Ok(response)
                };
                let hessian =
                    super::soscf::CommonOrbitalHessian::new(&c, &current.fock, [n_alpha, n_beta]);
                let gradient = hessian.gradient();
                if gradient.is_empty() {
                    return Ok(None);
                }
                let diagonal = hessian.diagonal();
                if opts.orbital_optimizer != super::OrbitalOptimizer::Legacy {
                    for rejection in 0..12 {
                        let step = super::trah::direction(
                            &diagonal,
                            &gradient,
                            |x| hessian.image(x, fock_response),
                            trust_region.radius,
                            opts.soscf_max_cg.max(2),
                            1e-3,
                        )?;
                        if step.predicted_work >= 0.0 {
                            return Ok(None);
                        }
                        let next = hessian.rotate(&step.rotation);
                        let candidate = build(&next)?;
                        let quadratic = functional.parts.is_empty()
                            && opts.skala.is_none()
                            && opts.vv10.is_none()
                            && opts.pcm.is_none();
                        let work = if quadratic {
                            let delta = hessian.finite_density_change(&step.rotation);
                            let induced = fock_response(&delta)?;
                            hessian.quadratic_energy_work(&step.rotation, &delta, &induced)
                        } else {
                            let delta = hessian.finite_density_change(&step.rotation);
                            let zero = std::array::from_fn(|s| {
                                Matrix::zeros(delta[s].rows, delta[s].cols)
                            });
                            let Some(value) = super::trah::energy_work(
                                current.electronic,
                                candidate.electronic,
                                |t| {
                                    let intermediate: [Matrix; 2] = std::array::from_fn(|s| {
                                        let mut p = current.density[s].clone();
                                        for (x, d) in
                                            p.as_mut_slice().iter_mut().zip(delta[s].as_slice())
                                        {
                                            *x += t * d;
                                        }
                                        p
                                    });
                                    let fields = density_model.evaluate(&intermediate)?;
                                    let view = super::soscf::CommonOrbitalHessian::new(
                                        &c,
                                        &fields.fock,
                                        [n_alpha, n_beta],
                                    );
                                    Ok(view.quadratic_energy_work(&step.rotation, &delta, &zero))
                                },
                            )?
                            else {
                                return Ok(None);
                            };
                            value
                        };
                        let length = step.rotation.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let radius = trust_region.radius;
                        let (accepted, ratio) =
                            trust_region.assess(step.predicted_work, work, length);
                        crate::logger::log(&format!(
                            "[roks-scf] TRAH accepted={accepted} rejection={rejection} radius={radius:.12e} step={length:.12e} predicted={:.12e} actual={work:.12e} ratio={ratio:.12e} shift={:.12e} inner_residual={:.12e} products={}",
                            step.predicted_work,step.shift,step.residual,step.products));
                        if !accepted && length < 0.5 * trust_region.radius {
                            return Ok(None);
                        }
                        if accepted {
                            return Ok(Some((next, candidate)));
                        }
                    }
                    return Ok(None);
                }
                let mut step = super::soscf::newton_direction_diagonal(
                    &diagonal,
                    &gradient,
                    |x| hessian.image(x, fock_response),
                    opts.soscf_max_cg,
                    1e-3,
                )?;
                let mut slope: f64 = gradient.iter().zip(&step).map(|(g, x)| g * x).sum();
                if !slope.is_finite() || slope >= 0.0 {
                    step = gradient
                        .iter()
                        .zip(&diagonal)
                        .map(|(g, d)| -g / d)
                        .collect();
                }
                let norm = step.iter().map(|x| x * x).sum::<f64>().sqrt();
                if !norm.is_finite() || norm == 0.0 {
                    return Ok(None);
                }
                let scale = (opts.soscf_trust / norm).min(1.0);
                for x in &mut step {
                    *x *= scale;
                }
                slope = gradient.iter().zip(&step).map(|(g, x)| g * x).sum();
                if !slope.is_finite() || slope >= 0.0 {
                    return Ok(None);
                }
                let mut fraction = 1.0;
                for _ in 0..12 {
                    let trial: Vec<f64> = step.iter().map(|x| fraction * x).collect();
                    let next = hessian.rotate(&trial);
                    let candidate = build(&next)?;
                    let quadratic =
                        functional.parts.is_empty() && opts.skala.is_none() && opts.vv10.is_none();
                    let work = if quadratic {
                        let delta = hessian.finite_density_change(&trial);
                        let induced = fock_response(&delta)?;
                        hessian.quadratic_energy_work(&trial, &delta, &induced)
                    } else {
                        candidate.electronic - current.electronic
                    };
                    if work <= 1e-4 * fraction * slope {
                        crate::logger::log(&format!(
                            "[roks-scf] common-orbital Newton fraction={fraction:.6e} energy_work={:.6e}",
                            work
                        ));
                        return Ok(Some((next, candidate)));
                    }
                    fraction *= 0.5;
                }
                Ok(None)
            })();
            match proposal {
                Ok(Some((next, candidate))) => {
                    c = next;
                    current = candidate;
                    history.clear();
                    orbital_active = true;
                    continue;
                }
                Ok(None) => {
                    trust_region = super::trah::TrustRegion::new(opts.soscf_trust)?;
                }
                // A missing optional optimization kernel must not remove a
                // previously available energy-only SCF calculation.
                Err(DftError::Unsupported { .. })
                    if opts.orbital_optimizer != super::OrbitalOptimizer::Trah =>
                {
                    newton_available = false
                }
                Err(error) => return Err(error),
            }
        }
        // Effective Fock matrix and error in the atomic-orbital basis.
        let sc = overlap.matmul(&c);
        let effective_ao = sc.matmul(&effective).matmul_transpose_seq(&sc);
        let error_ao = sc.matmul(&gradient).matmul_transpose_seq(&sc);
        history.push((effective_ao, error_ao));
        if history.len() > HISTORY {
            history.remove(0);
        }
        // Pulay extrapolation: minimize the combined error with unit sum.
        let m = history.len();
        let mut extrapolated = history[m - 1].0.clone();
        if m > 1 {
            let mut b = Matrix::zeros(m + 1, m + 1);
            let mut rhs = vec![0.0; m + 1];
            for i in 0..m {
                for j in 0..m {
                    b[(i, j)] = history[i].1.frobenius_dot(&history[j].1);
                }
                b[(i, m)] = 1.0;
                b[(m, i)] = 1.0;
            }
            rhs[m] = 1.0;
            if let Ok(weights) = solve_linear(&b, &rhs) {
                if weights.iter().all(|w| w.is_finite()) {
                    extrapolated = Matrix::zeros(extrapolated.rows, extrapolated.cols);
                    for (weight, (matrix, _)) in weights.iter().zip(&history) {
                        add_scaled(&mut extrapolated, matrix, *weight);
                    }
                }
            }
        }
        let (_, next) = orth.solve_roothaan(&extrapolated)?;
        c = next;
        current = build(&c)?;
    }

    // Occupied and virtual spaces of each spin diagonalized in its own Fock
    // matrix; the densities do not change.
    let mut mo_coeff = [c.clone(), c.clone()];
    let mut mo_energy = [vec![0.0; nmo], vec![0.0; nmo]];
    for spin in 0..2 {
        let occupied = if spin == 0 { n_alpha } else { n_beta };
        let f_mo = c.matmul_accurate(&current.fock[spin].matmul_accurate(&c, false), true);
        for (start_index, count) in [(0, occupied), (occupied, nmo - occupied)] {
            if count == 0 {
                continue;
            }
            let (values, vectors) = block_eigen(&f_mo, start_index, count)?;
            for col in 0..count {
                mo_energy[spin][start_index + col] = values[col];
                for row in 0..c.rows {
                    let mut value = 0.0;
                    for k in 0..count {
                        value += c[(row, start_index + k)] * vectors[(k, col)];
                    }
                    mo_coeff[spin][(row, start_index + col)] = value;
                }
            }
        }
    }
    let sz = 0.5 * (n_alpha - n_beta) as f64;
    let mut result = start;
    result.energy = current.electronic + result.e_nuc;
    result.e_elec = current.electronic;
    result.mo_energy = mo_energy;
    result.mo_coeff = mo_coeff;
    result.density = current.density;
    result.fock = current.fock;
    result.s_squared = sz * (sz + 1.0);
    crate::logger::log(&format!(
        "[roks] <S^2>={:.12} (hbar^2) S={sz:.6}",
        result.s_squared
    ));
    result.e_xc = current.exc;
    result.e_nlc = current.nlc;
    result.e_pcm = current.pcm;
    result.grid_electron_count = current.grid_electrons;
    result.converged = converged;
    result.common_orbital_residual = common_orbital_residual;
    result.iterations = iterations;
    result.occupations = [occupations(n_alpha), occupations(n_beta)];
    Ok((result, c))
}
