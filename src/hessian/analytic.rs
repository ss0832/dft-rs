// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic canonical molecular HF/KS Hessian and atomic polar tensor.
//! The shared moving-basis/grid skeleton includes all connected energy
//! terms. Integer occupations use occupied/virtual CPKS. Prescribed nonthermal
//! populations and finite electronic temperature use the full spectral
//! response and moving-metric energy-weighted density. The former holds every
//! MO population fixed; the latter includes the Fermi and chemical-potential
//! derivatives and differentiates the electronic Mermin free energy.
//!
//! With P(1)/W(1) from CPKS the Hessian is
//!   H_ab = E2_skel(a,b) + sum_s Tr[P(b,s) F(a,skel,s)] - Tr[W(b) S(a)],
//! where F(a,skel,s) = h(a) + J(a)[Pt] - cx K(a)[Ps] - clr Klr(a)[Ps].
//!
//! Gauge-invariant response quantities (only U_ai and the MO overlap
//! derivative Stilde enter):
//!   P(1,s) = sum_ai U_ai (Ca Ci^T + Ci Ca^T) - sum_ij Stilde_ij Ci Cj^T,
//!   W(1,s) = sum_ij Ci Cj^T [Ftot_ij - (eps_i + eps_j) Stilde_ij]
//!          + sum_ai eps_i U_ai (Ca Ci^T + Ci Ca^T),
//! with Ftot = MO transform of F(skel) + G(P(1)).
//! CPKS: (eps_a - eps_i) U_ai + [G(dP_U)]_ai
//!         = -[Fskel_ai + G(dP_oo)_ai - eps_i Stilde_ai].

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::integrals::deriv2::overlap_second_contraction;
use crate::integrals::deriv_mat::overlap_derivative_matrices;
use crate::linalg::Matrix;
use crate::response::ResponseSolver;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Analytic Hessian plus the analytic atomic polar tensor.
pub struct AnalyticHessianResult {
    /// 3N x 3N Cartesian Hessian (Hartree/Bohr^2), symmetrized; Mermin at finite T.
    pub hessian: Matrix,
    /// APT `d mu_x / dR_(a,ax)` as a 3 x 3N matrix (a.u.).
    pub dipole_derivatives: Matrix,
    /// Largest `|H_rc - H_cr|` measured before symmetrization.
    pub raw_asymmetry: f64,
    /// Relaxed CPKS density derivatives `[dP_alpha/dx, dP_beta/dx]` for each
    /// Cartesian coordinate `x = 3*atom + axis`, including the metric term.
    pub density_derivatives: Vec<[Matrix; 2]>,
    /// Derivative of sum_s C_s f_s epsilon_s C_s^T for each Cartesian coordinate.
    pub energy_weighted_density_derivatives: Vec<Matrix>,
    /// Original matrix-free ensemble equation residual per Cartesian perturbation.
    /// Empty for the integer occupied/virtual solver; populated for both
    /// prescribed populations and thermal ensembles.
    pub ensemble_response_residuals: Vec<f64>,
    /// Fresh maximum absolute `A u + b` for every Cartesian perturbation,
    /// Hartree/Bohr, for integer-occupation RKS/UKS. Empty for ensembles,
    /// whose separate dimensionless residuals are reported above.
    pub orbital_response_residuals: Vec<f64>,
    /// Measured Tr(S dP) + Tr(dS P), per perturbation and spin.
    pub ensemble_number_defects: Vec<[f64; 2]>,
}

/// Analytic Hessian only (see `analytic_hessian_full` for the APT too).
pub fn analytic_hessian(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    scf: &ScfResult,
    grid: Option<&crate::grid::MolecularGrid>,
) -> Result<Matrix> {
    Ok(analytic_hessian_full(mol, mb, opts, functional, scf, grid)?.hessian)
}

/// Analytic Hessian: HF/hybrid exact exchange plus LDA, GGA and meta-GGA
/// XC grid terms, the ECP second derivative, and the SC-VV10 nonlocal
/// skeleton with its f_nlc CPKS coupling. The analytic APT comes from the
/// same CPKS solutions at no extra cost. A uniform external field is included
/// through fixed-origin dipole-integral first and second derivatives. RI-J
/// and an unavailable continuum reaction-field response are refused with a
/// typed DftError::Unsupported; no finite-difference fallback is selected.
pub fn analytic_hessian_full(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    scf: &ScfResult,
    grid: Option<&crate::grid::MolecularGrid>,
) -> Result<AnalyticHessianResult> {
    let natoms = mol.len();
    let n = mb.nao_sph;
    let p_tot = scf.p_total();
    // Energy-weighted density W = sum_s C_occ eps C_occ^T.
    let mut w = Matrix::zeros(n, n);
    for spin in 0..2 {
        let c = &scf.mo_coeff[spin];
        for i in 0..c.cols {
            let e = scf.occupations[spin][i] * scf.mo_energy[spin][i];
            for mu in 0..n {
                let cmu = c[(mu, i)];
                if cmu == 0.0 {
                    continue;
                }
                for nu in 0..n {
                    w[(mu, nu)] += e * cmu * c[(nu, i)];
                }
            }
        }
    }

    let fixed = super::fixed_density::fixed_density_hessian_terms(
        mol,
        mb,
        opts,
        functional,
        [&scf.density[0], &scf.density[1]],
        grid,
    )?;
    let mut h = fixed.hessian;
    // The ordinary stationary reference contributes its metric covector.
    // The shared off-shell skeleton deliberately contains no Pulay term.
    let hs = overlap_second_contraction(mb, &w, natoms);
    for (total, metric) in h.as_mut_slice().iter_mut().zip(hs.as_slice()) {
        *total -= *metric;
    }
    let ds = overlap_derivative_matrices(mb, natoms);
    let fskel =
        |atom: usize, axis: usize, spin: usize| fixed.fock_derivatives[spin][atom][axis].clone();

    // ---------------- CPKS per perturbation ----------------
    // The CPKS drives J/K once per PCG iteration per perturbation, i.e.
    // O(3N * n_iter) Fock builds. In direct mode every one of them
    // recomputes the whole ERI list, which dominates the analytic Hessian
    // (measured: the skeleton integrals and the XC grid are noise beside
    // it). Caching the integrals once pays for itself immediately whenever
    // they fit the caller's budget, using the shared SCF/response policy.
    let mut cpks_opts = opts.clone();
    cpks_opts.eri_mode = opts.effective_eri_mode(mb.nao_sph);
    let spins = if scf.unrestricted { 2 } else { 1 };
    let mut p1_store: Vec<[Matrix; 2]> = Vec::with_capacity(3 * natoms);
    let mut w1_store: Vec<Matrix> = Vec::with_capacity(3 * natoms);
    let mut ensemble_response_residuals = Vec::new();
    let mut orbital_response_residuals = Vec::new();
    let mut ensemble_number_defects = Vec::new();
    if scf.electronic_temperature > 0.0 || scf.fixed_occupation_residual.is_some() {
        let solver = crate::response::ensemble::EnsembleSolver::new(
            mol, mb, functional, grid, scf, &cpks_opts,
        )?;
        let config = crate::response::ResponseSolverConfig {
            tolerance: cpks_opts.p_tol.min(1e-10),
            ..Default::default()
        };
        for b in 0..natoms {
            for by in 0..3 {
                let response = solver.solve_metric(
                    &[fskel(b, by, 0), fskel(b, by, 1)],
                    &ds[b][by],
                    &config,
                )?;
                ensemble_response_residuals.push(response.response.relative_residual);
                ensemble_number_defects.push(response.response.number_defects);
                p1_store.push(response.response.delta_density);
                w1_store.push(response.delta_energy_weighted_density);
            }
        }
    } else {
        let mut solver = ResponseSolver::new(mol, mb, functional, grid, scf, &cpks_opts)?;
        // The analytic APT is built directly from these first-order densities.
        // ResponseSolver's general-purpose 1e-8 residual is adequate for most
        // properties, but it leaves an O(1e-8) violation of the exact APT
        // translational sum rule for the more strongly coupled meta hybrids
        // (M06-2X is the regression case). Tighten the actual CPKS solve rather
        // than projecting the returned tensor onto the sum rule, which would hide
        // a wrong response path behind a post-processing identity.
        solver.tol = solver.tol.min(cpks_opts.p_tol).min(1e-10);
        solver.max_iter = solver.max_iter.max(200);
        let (n_occs, _) = solver.ov_dims();
        // Reuse the exact CPKS operator, including its cached full-range/LR J/K
        // engines and density kernels. Keeping a second pair of in-core engines
        // here can otherwise retain four O(n^4) tensors for an RSH Hessian.
        let g_of = |dp: &[Matrix; 2]| solver.fock_response(dp);

        // Per perturbation b: response P(1)/W(1); then contract against all a.
        // The CPKS right-hand sides are collected FIRST and solved in one
        // block, so the shared response machinery is driven once per iteration
        // instead of once per perturbation (the iterates are unchanged; only
        // the scheduling differs).
        let mut rhs_store: Vec<Vec<f64>> = Vec::with_capacity(3 * natoms);
        let mut dp_oo_store: Vec<[Matrix; 2]> = Vec::with_capacity(3 * natoms);
        let mut st_oo_store: Vec<[Matrix; 2]> = Vec::with_capacity(3 * natoms);
        for b in 0..natoms {
            for by in 0..3 {
                // MO overlap derivative blocks and dP_oo.
                let sb = &ds[b][by];
                let mut dp_oo = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
                // Stilde_occ-occ per spin, kept for W(1).
                let mut st_oo = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
                for spin in 0..spins {
                    let no = n_occs[spin];
                    let c = &scf.mo_coeff[spin];
                    let mut st = Matrix::zeros(no, no);
                    for i in 0..no {
                        for j in 0..no {
                            let mut acc = 0.0;
                            for mu in 0..n {
                                let cmi = c[(mu, i)];
                                if cmi == 0.0 {
                                    continue;
                                }
                                for nu in 0..n {
                                    acc += cmi * sb[(mu, nu)] * c[(nu, j)];
                                }
                            }
                            st[(i, j)] = acc;
                        }
                    }
                    for i in 0..no {
                        for j in 0..no {
                            let sij = st[(i, j)];
                            if sij == 0.0 {
                                continue;
                            }
                            for mu in 0..n {
                                let cmi = c[(mu, i)];
                                if cmi == 0.0 {
                                    continue;
                                }
                                for nu in 0..n {
                                    dp_oo[spin][(mu, nu)] -= sij * cmi * c[(nu, j)];
                                }
                            }
                        }
                    }
                    st_oo[spin] = st;
                }
                if !scf.unrestricted {
                    dp_oo[1] = dp_oo[0].clone();
                    st_oo[1] = st_oo[0].clone();
                }
                let g_oo = g_of(&dp_oo)?;
                // RHS_ai = [Fskel + G(dP_oo)]_ai - eps_i Stilde_ai.
                let f_b: Vec<Matrix> = (0..spins).map(|s| fskel(b, by, s)).collect();
                let mut rhs_m = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
                for spin in 0..spins {
                    let mut m = f_b[spin].clone();
                    for (mv, gv) in m.as_mut_slice().iter_mut().zip(g_oo[spin].as_slice()) {
                        *mv += *gv;
                    }
                    rhs_m[spin] = m;
                }
                if spins == 1 {
                    rhs_m[1] = rhs_m[0].clone();
                }
                let mut bvec = vec![
                    0.0;
                    (0..spins)
                        .map(|s| {
                            let no = n_occs[s];
                            let nv = scf.mo_coeff[s].cols - no;
                            no * nv
                        })
                        .sum()
                ];
                solver.project_ov([&rhs_m[0], &rhs_m[1]], &mut bvec);
                // Subtract eps_i Stilde_ai.
                {
                    let mut off = 0;
                    for spin in 0..spins {
                        let no = n_occs[spin];
                        let c = &scf.mo_coeff[spin];
                        let nv = c.cols - no;
                        let eps = &scf.mo_energy[spin];
                        for a in 0..nv {
                            for i in 0..no {
                                let mut sai = 0.0;
                                for mu in 0..n {
                                    let cva = c[(mu, no + a)];
                                    if cva == 0.0 {
                                        continue;
                                    }
                                    for nu in 0..n {
                                        sai += cva * sb[(mu, nu)] * c[(nu, i)];
                                    }
                                }
                                bvec[off + a * no + i] -= eps[i] * sai;
                            }
                        }
                        off += no * nv;
                    }
                }
                rhs_store.push(bvec);
                dp_oo_store.push(dp_oo);
                st_oo_store.push(st_oo);
            }
        }
        let u_all = crate::logger::timed("hess: CPKS block solve", || {
            solver.solve_ov_block(&rhs_store)
        })?;
        for (bcol, u) in u_all.iter().enumerate() {
            let residual = solver.ov_equation_residual(u, &rhs_store[bcol])?;
            orbital_response_residuals.push(residual);
            if !residual.is_finite() || residual > solver.tol {
                return Err(DftError::ResponseNotConverged {
                    iterations: solver.max_iter,
                    residual,
                });
            }
            let (b, by) = (bcol / 3, bcol % 3);
            let dp_oo = &dp_oo_store[bcol];
            let st_oo = &st_oo_store[bcol];
            // F^(a),skel is cheap to rebuild and keeping 3N copies alive would
            // cost more memory than it saves.
            let f_b: Vec<Matrix> = (0..spins).map(|s| fskel(b, by, s)).collect();
            {
                let dp_u = solver.delta_density(u);
                // P(1) = dP_U + dP_oo (restricted: the beta response equals the
                // alpha response; delta_density only fills spin 0 then).
                let mut p1 = [dp_u[0].clone(), dp_u[1].clone()];
                for spin in 0..2 {
                    for (t, v) in p1[spin]
                        .as_mut_slice()
                        .iter_mut()
                        .zip(dp_oo[spin].as_slice())
                    {
                        *t += *v;
                    }
                }
                if !scf.unrestricted {
                    p1[1] = p1[0].clone();
                }
                // W(1) = sum_s { C_o [Ftot_oo - (eps_i+eps_j) Stilde_oo] C_o^T
                //              + sum_ai eps_i U_ai (Ca Ci^T + Ci Ca^T) }.
                let g_p1 = g_of(&p1)?;
                let mut w1 = Matrix::zeros(n, n);
                let mut off = 0;
                for spin in 0..spins {
                    let no = n_occs[spin];
                    let c = &scf.mo_coeff[spin];
                    let nv = c.cols - no;
                    let eps = &scf.mo_energy[spin];
                    let spin_scale = if scf.unrestricted { 1.0 } else { 2.0 };
                    // Ftot = Fskel + G(P1), occ-occ MO block.
                    let mut ftot = f_b[spin].clone();
                    for (fv, gv) in ftot.as_mut_slice().iter_mut().zip(g_p1[spin].as_slice()) {
                        *fv += *gv;
                    }
                    for i in 0..no {
                        for j in 0..no {
                            let mut fij = 0.0;
                            for mu in 0..n {
                                let cmi = c[(mu, i)];
                                if cmi == 0.0 {
                                    continue;
                                }
                                for nu in 0..n {
                                    fij += cmi * ftot[(mu, nu)] * c[(nu, j)];
                                }
                            }
                            let coefw = fij - (eps[i] + eps[j]) * st_oo[spin][(i, j)];
                            if coefw == 0.0 {
                                continue;
                            }
                            for mu in 0..n {
                                let cmi = c[(mu, i)];
                                if cmi == 0.0 {
                                    continue;
                                }
                                for nu in 0..n {
                                    w1[(mu, nu)] += spin_scale * coefw * cmi * c[(nu, j)];
                                }
                            }
                        }
                    }
                    let nvno = nv * no;
                    for a in 0..nv {
                        for i in 0..no {
                            let uai = u[off + a * no + i];
                            if uai == 0.0 {
                                continue;
                            }
                            let f = spin_scale * eps[i] * uai;
                            for mu in 0..n {
                                let ca_mu = c[(mu, no + a)];
                                let ci_mu = c[(mu, i)];
                                for nu in 0..n {
                                    w1[(mu, nu)] +=
                                        f * (ca_mu * c[(nu, i)] + ci_mu * c[(nu, no + a)]);
                                }
                            }
                        }
                    }
                    off += nvno;
                }
                p1_store.push(p1);
                w1_store.push(w1);
            }
        }
    }

    crate::logger::log("hess: CPKS post-processing done");
    // ---------------- Response contraction ----------------
    for a in 0..natoms {
        for ax in 0..3 {
            let f_a: Vec<Matrix> = (0..spins).map(|s| fskel(a, ax, s)).collect();
            let sa = &ds[a][ax];
            for bcol in 0..3 * natoms {
                let p1 = &p1_store[bcol];
                let w1 = &w1_store[bcol];
                let mut v = 0.0;
                for spin in 0..spins {
                    // Restricted: p1[0] is the alpha response, total = 2x.
                    let pspin = &p1[spin];
                    let fs = &f_a[spin];
                    let mut tr = 0.0;
                    for (pv, fv) in pspin.as_slice().iter().zip(fs.as_slice()) {
                        tr += pv * fv;
                    }
                    v += if scf.unrestricted { tr } else { 2.0 * tr };
                }
                let mut trw = 0.0;
                for (wv, sv) in w1.as_slice().iter().zip(sa.as_slice()) {
                    trw += wv * sv;
                }
                v -= trw;
                h[(3 * a + ax, bcol)] += v;
            }
        }
    }
    // Enforce the exact symmetry (response asymmetry is solver noise), after
    // recording the raw defect for the caller.
    let dim = 3 * natoms;
    let mut raw_asymmetry = 0.0_f64;
    for r in 0..dim {
        for cidx in (r + 1)..dim {
            raw_asymmetry = raw_asymmetry.max((h[(r, cidx)] - h[(cidx, r)]).abs());
        }
    }
    for r in 0..dim {
        for cidx in (r + 1)..dim {
            let avg = 0.5 * (h[(r, cidx)] + h[(cidx, r)]);
            h[(r, cidx)] = avg;
            h[(cidx, r)] = avg;
        }
    }

    crate::logger::log("hess: response contraction done");
    // ---------------- Analytic APT ----------------
    // d mu_x / d(b) = Z_eff_b delta_{x,by} - Tr[P(b) D_x] - Tr[P D_x^(b)].
    let dips = crate::integrals::dipole_matrices(mb, crate::math::Vec3::zero());
    let ddip = crate::integrals::deriv_mat::dipole_derivative_matrices(mb, natoms);
    let z_eff = mb.effective_charges(mol);
    let mut apt = Matrix::zeros(3, dim);
    for b in 0..natoms {
        for by in 0..3 {
            let bcol = 3 * b + by;
            let p1 = &p1_store[bcol];
            let mut p1_tot = p1[0].clone();
            for (t, v) in p1_tot.as_mut_slice().iter_mut().zip(p1[1].as_slice()) {
                *t += *v;
            }
            for x in 0..3 {
                let mut v = if x == by { z_eff[b] } else { 0.0 };
                for (pv, dv) in p1_tot.as_slice().iter().zip(dips[x].as_slice()) {
                    v -= pv * dv;
                }
                for (pv, dv) in p_tot.as_slice().iter().zip(ddip[x][b][by].as_slice()) {
                    v -= pv * dv;
                }
                apt[(x, bcol)] = v;
            }
        }
    }
    Ok(AnalyticHessianResult {
        hessian: h,
        dipole_derivatives: apt,
        raw_asymmetry,
        density_derivatives: p1_store,
        energy_weighted_density_derivatives: w1_store,
        ensemble_response_residuals,
        orbital_response_residuals,
        ensemble_number_defects,
    })
}
