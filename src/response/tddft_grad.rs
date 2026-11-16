// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic nuclear gradient of a TDA (CIS) excited state.
//!
//! # What is differentiated
//!
//! The TOTAL excited-state energy `E_SCF + omega` of a closed-shell
//! singlet TDA state, including the nuclear repulsion — i.e. the force on
//! the excited state, not `d omega / dR` on its own.
//!
//! # Why a Z-vector
//!
//! Differentiating `omega` directly leaves terms in the orbital-rotation
//! derivatives `U^x_ai`, which would mean one CPHF solve per nuclear
//! coordinate — `3N` of them. The Z-vector trick (Handy–Schaefer) reorders
//! the contraction so a SINGLE solve of the same CPHF operator, driven by
//! a right-hand side built from the excited-state densities, serves every
//! coordinate. That is the whole reason to have this rather than finite
//! differences.
//!
//! # Amplitude normalisation
//!
//! The eigenvector of the singlet TDA matrix `A_ai,bj = dEps + 2(ai|bj) -
//! cx(ab|ij)` is normalised to `sum X^2 = 1` over the SPATIAL orbital
//! pairs. The response densities below are the ones of the spin-adapted
//! singlet, whose amplitudes carry an extra `1/sqrt(2)` per spin channel;
//! `xs = X / sqrt(2)` is that scale, and every density built from it is
//! then a TOTAL (both-spin) density on the same footing as `2 C_o C_o^T`.
//! Trace check: `Tr dvv = -Tr doo = 1`, one electron promoted.
//!
//! # Densities
//!
//! ```text
//! dvv_ab =  2 sum_i xs_ai xs_bi      (virtual-virtual)
//! doo_ij = -2 sum_a xs_ai xs_aj      (occupied-occupied)
//! T      = C_v xs C_o^T              (AO transition density, NOT symmetric)
//! D_zoo  = C_o doo C_o^T + C_v dvv C_v^T
//! ```
//!
//! `D_zoo` is the change in density on excitation with the orbitals held
//! fixed; the Z-vector `z1` adds the part that comes from the orbitals
//! relaxing. `T` enters quadratically, through the same kernel that built
//! the TDA matrix `A`. The factor 2 in `dvv`/`doo` is the sum of the
//! `(X+Y)` and `(X-Y)` contributions, which in TDA both reduce to `X`; for
//! the same reason `T` contributes through BOTH its symmetric part (full
//! singlet kernel) and its antisymmetric part (exchange only, since `J` of
//! an antisymmetric density vanishes).
//!
//! # Derivative-integral bookkeeping
//!
//! `overlap/kinetic/nuclear_derivative_matrices` and `eri_derivative_fock`
//! return the FULL derivative of the assembled matrix — every centre of
//! every index, with the density held fixed. So a single Frobenius
//! contraction `X . d(V[P])/dR` already equals the complete derivative of
//! the bilinear form `sum_mnls (mn|ls) X_mn P_ls`, covering the bra AND
//! ket sides of both densities at once. Formulations that differentiate
//! only the first index need two such contractions (roles of `X` and `P`
//! swapped) to say the same thing; here they collapse into one, which is
//! why the ground-state two-electron term below carries a coefficient of
//! one rather than two and the quadratic transition-density terms carry
//! one half.
//!
//! # Scope
//!
//! Closed-shell singlet TDA. [`tda_gradient`] preserves the original HF/CIS
//! entry point; [`tda_gradient_with_grid`] additionally covers gas-phase
//! pure-LDA references. GGA/meta-GGA, nonlocal correlation, PCM, and DFT
//! hybrids remain typed refusals until their complete excited-state
//! Lagrangian derivatives are present.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::eval::eval_aos_full;
use crate::grid::MolecularGrid;
use crate::integrals::deriv_mat::{
    eri_derivative_fock, kinetic_derivative_matrices, nuclear_derivative_matrices,
    overlap_derivative_matrices,
};
use crate::integrals::{schwarz_matrix, CoulombKernel};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::tddft::{TdaSpin, TdaState};
use crate::response::ResponseSolver;
use crate::scf::{JkEngine, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;

/// Analytic gradient of `E_SCF + omega` for one singlet TDA state, in
/// Hartree/Bohr per atom.
pub fn tda_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    scf: &ScfResult,
    state: &TdaState,
    opts: &ScfOptions,
) -> Result<Vec<Vec3>> {
    tda_gradient_impl(mol, mb, functional, None, scf, state, opts)
}

/// Analytic gradient of `E_SCF + omega` with the molecular grid needed by
/// a DFT reference. The currently complete DFT scope is gas-phase,
/// closed-shell singlet pure LDA.
pub fn tda_gradient_with_grid(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    state: &TdaState,
    opts: &ScfOptions,
) -> Result<Vec<Vec3>> {
    tda_gradient_impl(mol, mb, functional, grid, scf, state, opts)
}

fn tda_gradient_impl(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    state: &TdaState,
    opts: &ScfOptions,
) -> Result<Vec<Vec3>> {
    if scf.unrestricted {
        return Err(DftError::Unsupported {
            feature: "analytic TDA gradient for an open-shell reference".to_string(),
            missing: "spin-resolved excited-state and Z-vector density matrices".to_string(),
        });
    }
    // The amplitudes of a triplet or an RPA state are numerically
    // indistinguishable from a singlet TDA one, so nothing downstream would
    // notice the wrong response densities being built from them. Refuse.
    if state.spin != TdaSpin::Singlet {
        return Err(DftError::Unsupported {
            feature: format!("analytic {:?} TDA gradient", state.spin),
            missing: "the response densities below are spin-adapted for a singlet state"
                .to_string(),
        });
    }
    if !state.tamm_dancoff {
        return Err(DftError::Unsupported {
            feature: "analytic full-RPA excited-state gradient".to_string(),
            missing: "the Tamm-Dancoff response densities used here are not valid for RPA; \
                      RPA densities must be built separately from X+Y and X-Y"
                .to_string(),
        });
    }
    let has_xc = !functional.parts.is_empty();
    if has_xc {
        if functional.family != Family::Lda {
            return Err(DftError::Unsupported {
                feature: "analytic TDA gradient for a GGA or meta-GGA reference".to_string(),
                missing: "the complete rho/sigma/tau third-kernel contraction and moving-grid excited-state Lagrangian".to_string(),
            });
        }
        if functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0 {
            return Err(DftError::Unsupported {
                feature: "analytic TDA gradient for an LDA hybrid reference".to_string(),
                missing: "the hybrid/RSH excited-state exchange Lagrangian has not been combined with the LDA third-kernel path".to_string(),
            });
        }
        if let Some(term) = functional.required_nonlocal_term() {
            return Err(DftError::Unsupported {
                feature: format!("analytic TDA gradient with {term}"),
                missing: "the nonlocal excited-state third-response kernel and its moving-grid derivative".to_string(),
            });
        }
        if opts.vv10.is_some() {
            return Err(DftError::Unsupported {
                feature: "analytic TDA gradient with self-consistent VV10".to_string(),
                missing: "the VV10 excited-state third-response kernel and its grid derivative"
                    .to_string(),
            });
        }
        if opts.pcm.is_some() {
            return Err(DftError::Unsupported {
                feature: "analytic TDA gradient with continuum solvation".to_string(),
                missing: "the excited-state reaction-field Lagrangian and cavity response"
                    .to_string(),
            });
        }
        if grid.is_none() {
            return Err(DftError::InvalidInput(
                "an LDA TDA gradient needs the same molecular grid used by SCF and TDA".to_string(),
            ));
        }
    }
    // The learned functional enters through the same three places as the
    // semilocal one: the kernel, its third-order contraction with the
    // transition density and the moving-grid derivative of the Lagrangian.
    let learned = opts.skala.as_ref();
    if learned.is_some() && grid.is_none() {
        return Err(DftError::InvalidInput(
            "a TDA gradient with the learned functional needs the molecular grid used by SCF and TDA"
                .to_string(),
        ));
    }
    let has_kernel = has_xc || learned.is_some();
    let cx = functional.hybrid.cx;
    let natoms = mol.len();
    let c = &scf.mo_coeff[0];
    let eps = &scf.mo_energy[0];
    let no = scf.n_alpha;
    let nmo = c.cols;
    let nv = nmo - no;

    // Amplitudes on the spin-adapted scale (see the module header).
    let mut xs = Matrix::zeros(nv, no);
    for a in 0..nv {
        for i in 0..no {
            xs[(a, i)] = state.x[0][(a, i)] * std::f64::consts::FRAC_1_SQRT_2;
        }
    }

    // Occupied / virtual coefficient blocks.
    let orbo = block(c, 0, no);
    let orbv = block(c, no, nv);

    // ---- Excited-state densities ----
    let mut dvv = Matrix::zeros(nv, nv);
    for a in 0..nv {
        for b in 0..nv {
            let mut v = 0.0;
            for i in 0..no {
                v += xs[(a, i)] * xs[(b, i)];
            }
            dvv[(a, b)] = 2.0 * v;
        }
    }
    let mut doo = Matrix::zeros(no, no);
    for i in 0..no {
        for j in 0..no {
            let mut v = 0.0;
            for a in 0..nv {
                v += xs[(a, i)] * xs[(a, j)];
            }
            doo[(i, j)] = -2.0 * v;
        }
    }
    // AO transition density and its symmetric / antisymmetric halves.
    let t_ao = orbv.matmul(&xs).matmul_transpose_seq(&orbo);
    let t_sym = add(&t_ao, &t_ao.transpose());
    let t_anti = sub(&t_ao, &t_ao.transpose());
    // Unrelaxed difference density in AO.
    let dm_zoo = add(
        &orbo.matmul(&doo).matmul_transpose_seq(&orbo),
        &orbv.matmul(&dvv).matmul_transpose_seq(&orbv),
    );

    let jk = JkEngine::from_options(mol, mb, opts)?;
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    // Closed-shell J/K on an arbitrary (possibly non-symmetric) density.
    // `build` returns J of the TOTAL density and K of the per-SPIN one, so
    // getting `J[D]` and `K[D]` for the same `D` means passing `D` in both
    // slots. Passing `D/2` for the spin blocks — the shape the SCF uses,
    // where the spin densities really are half the total — silently halves
    // every exchange term.
    let jk_of = |d: &Matrix| -> Result<(Matrix, Matrix)> {
        let (j, k) = jk.build(mb, d, [d, d], true)?;
        Ok((j, k[0].clone()))
    };

    // veff0doo = 2 J[D_zoo] - cx K[D_zoo]
    let (j0, k0) = jk_of(&dm_zoo)?;
    let mut veff0doo = axpy(&scale(&j0, 2.0), -cx, &k0);
    if has_kernel {
        // The restricted f1oo contraction uses twice the total difference
        // density against the total-density LDA kernel.  In spin-resolved
        // form this is the same as handing dm_zoo to both channels.
        let d = [dm_zoo.clone(), dm_zoo.clone()];
        let response = solver.fxc_contraction(&d)?;
        veff0doo = add(&veff0doo, &response[0]);
    }
    // The same operator on the symmetric transition density, and exchange
    // alone on the antisymmetric one (J of an antisymmetric density
    // vanishes). `veff0mop` is symmetric; `veff0mom` is ANTIsymmetric, so
    // every index order below is load-bearing.
    let (jt, kt) = jk_of(&t_sym)?;
    let mut veff0mop_ao = axpy(&scale(&jt, 2.0), -cx, &kt);
    if has_kernel {
        let t = [t_sym.clone(), t_sym.clone()];
        let response = solver.fxc_contraction(&t)?;
        veff0mop_ao = add(&veff0mop_ao, &response[0]);
    }
    if let (Some(model), Some(grid)) = (learned, grid) {
        let ktt = crate::skala::skala_third_contraction(
            model,
            mol,
            mb,
            grid,
            [&scf.density[0], &scf.density[1]],
            [&t_sym, &t_sym],
        )?;
        veff0doo = axpy(&veff0doo, 0.5, &ktt[0]);
    }
    if has_xc {
        let ktt = lda_third_transition_potential(
            functional,
            mb,
            grid.expect("LDA grid checked above"),
            [&scf.density[0], &scf.density[1]],
            [&t_sym, &t_sym],
        )?;
        // The spin-adapted TDA Lagrangian contains 1/4 T f_xc T.  This is
        // the derivative with respect to one spin density; the explicit
        // factor two for a shared closed-shell orbital rotation is already
        // applied when `wvo` is assembled below.
        veff0doo = axpy(&veff0doo, 0.5, &ktt[0]);
    }
    let veff0mop = mo_transform(c, &veff0mop_ao);
    let (_, kta) = jk_of(&t_anti)?;
    let veff0mom = mo_transform(c, &scale(&kta, -cx));
    let doo_v = mo_transform(c, &veff0doo);

    // ---- Z-vector right-hand side ----
    let mut wvo = Matrix::zeros(nv, no);
    for a in 0..nv {
        for i in 0..no {
            let mut v = 2.0 * doo_v[(no + a, i)];
            for k in 0..no {
                v -= 2.0 * (veff0mop[(i, k)] + veff0mom[(i, k)]) * xs[(a, k)];
            }
            for e in 0..nv {
                v += 2.0 * (veff0mop[(no + e, no + a)] + veff0mom[(no + e, no + a)]) * xs[(e, i)];
            }
            wvo[(a, i)] = v;
        }
    }

    // ---- Solve the Z-vector equation ----
    // `solve_ov` solves A z = -b with the SAME operator A that the CPHF /
    // CPKS Hessian uses, which is exactly what the Z-vector equation needs.
    let mut b = vec![0.0; nv * no];
    for a in 0..nv {
        for i in 0..no {
            b[a * no + i] = wvo[(a, i)];
        }
    }
    let z1v = solver.solve_ov(&b)?;
    let mut z1 = Matrix::zeros(nv, no);
    for a in 0..nv {
        for i in 0..no {
            z1[(a, i)] = z1v[a * no + i];
        }
    }
    let z1ao = orbv.matmul(&z1).matmul_transpose_seq(&orbo);
    // The closed-shell response operator `J - cx K/2` applied to the
    // relaxation density — the plain one, NOT the `2J - cx K` combination
    // used for the difference density. It is the same operator the CPHF
    // solve above applies, so the Lagrangian stays consistent with `z1`.
    let zsym = add(&z1ao, &z1ao.transpose());
    let (jz, kz) = jk_of(&zsym)?;
    let mut veff_z = axpy(&jz, -0.5 * cx, &kz);
    if has_kernel {
        let zhalf = scale(&zsym, 0.5);
        let response = solver.fxc_contraction(&[zhalf.clone(), zhalf])?;
        veff_z = add(&veff_z, &response[0]);
    }

    // ---- Energy-weighted (Lagrangian) matrix ----
    let mut im0 = Matrix::zeros(nmo, nmo);
    let doo_plus_z = mo_transform(c, &add(&veff0doo, &veff_z));
    for i in 0..no {
        for j in 0..no {
            let mut v = doo_plus_z[(i, j)];
            for a in 0..nv {
                v += (veff0mop[(no + a, i)] + veff0mom[(no + a, i)]) * xs[(a, j)];
            }
            im0[(i, j)] = v;
        }
    }
    for a in 0..nv {
        for e in 0..nv {
            let mut v = 0.0;
            for i in 0..no {
                v += (veff0mop[(no + e, i)] + veff0mom[(no + e, i)]) * xs[(a, i)];
            }
            im0[(no + a, no + e)] = v;
        }
    }
    for a in 0..nv {
        for k in 0..no {
            let mut v = 0.0;
            for i in 0..no {
                v += (veff0mop[(k, i)] + veff0mom[(k, i)]) * xs[(a, i)];
            }
            im0[(no + a, k)] = 2.0 * v;
        }
    }
    // zeta[p,q] = (eps_p + eps_q)/2, except the ov/vo blocks which take the
    // energy of whichever index is occupied.
    let mut dm1 = Matrix::zeros(nmo, nmo);
    for i in 0..no {
        for j in 0..no {
            dm1[(i, j)] = doo[(i, j)];
        }
        dm1[(i, i)] += 2.0; // ground state
    }
    for a in 0..nv {
        for e in 0..nv {
            dm1[(no + a, no + e)] = dvv[(a, e)];
        }
        for i in 0..no {
            dm1[(no + a, i)] = z1[(a, i)];
        }
    }
    for p in 0..nmo {
        for q in 0..nmo {
            let zeta = if (p >= no) != (q >= no) {
                eps[q]
            } else {
                0.5 * (eps[p] + eps[q])
            };
            im0[(p, q)] += zeta * dm1[(p, q)];
        }
    }
    let im0_ao = c.matmul(&im0).matmul_transpose_seq(c);

    // ---- Contract with the derivative integrals ----
    let ds = overlap_derivative_matrices(mb, natoms);
    let dt = kinetic_derivative_matrices(mb, natoms);
    let charges: Vec<f64> = mb.effective_charges(mol);
    let dv = nuclear_derivative_matrices(mb, mol, &charges);
    // Effective core potentials are part of the one-electron operator; the
    // matrices are zero for an all-electron basis.
    let du = crate::integrals::ecp::ecp_derivative_matrices(mb, natoms);
    let schwarz = schwarz_matrix(mb, CoulombKernel::Coulomb);
    // Ground-state density of the doubly occupied orbitals, per spin.
    let oo0 = orbo.matmul_transpose_seq(&orbo);
    let dmz1doo = add(&z1ao, &dm_zoo);

    let deriv_jk = |d: &Matrix| {
        eri_derivative_fock(
            mb,
            natoms,
            d,
            [d, d],
            CoulombKernel::Coulomb,
            &schwarz,
            1e-14,
        )
    };
    let (dj_oo0, dk_oo0) = deriv_jk(&oo0);
    let (dj_t, dk_t) = deriv_jk(&t_sym);
    let (_, dk_ta) = deriv_jk(&t_anti);

    let mut grad = vec![Vec3::zero(); natoms];
    for at in 0..natoms {
        let mut g = [0.0; 3];
        for ax in 0..3 {
            let h1 = add(&add(&dt[at][ax], &dv[at][ax]), &du[at][ax]);
            // Ground state (doubly occupied, hence the 2) plus the change
            // on excitation, against the one-electron derivative.
            let mut v = 2.0 * oo0.frobenius_dot(&h1) + dmz1doo.frobenius_dot(&h1);
            // vhf0 is the derivative of the ground-state two-electron
            // operator 2J - cx K at fixed density. Contracting it with
            // `oo0` gives the whole ground-state two-electron gradient
            // (both densities of the quadratic form are differentiated by
            // the full derivative, so no second factor); contracting it
            // with the excited-state difference density covers BOTH cross
            // terms of that bilinear form at once.
            let vhf0 = axpy(&scale(&dj_oo0[at][ax], 2.0), -cx, &dk_oo0[0][at][ax]);
            v += add(&oo0, &dmz1doo).frobenius_dot(&vhf0);
            // Transition-density terms. These are quadratic in ONE density
            // and the full derivative differentiates both of its
            // occurrences, so each needs a half.
            let vhf_t = axpy(&scale(&dj_t[at][ax], 2.0), -cx, &dk_t[0][at][ax]);
            v += 0.5 * t_sym.frobenius_dot(&vhf_t);
            v -= 0.5 * cx * t_anti.frobenius_dot(&dk_ta[0][at][ax]);
            // Pulay.
            v -= im0_ao.frobenius_dot(&ds[at][ax]);
            g[ax] = v;
        }
        grad[at] = Vec3::new(g[0], g[1], g[2]);
    }
    if has_xc {
        // The local scalar represented by the nonsymmetric `z1ao` is the
        // density of zsym/2.  Split the total relaxed density equally over
        // the two spin channels: dm_zoo/2 + zsym/4.
        let relaxed = axpy(&scale(&dm_zoo, 0.5), 0.25, &zsym);
        let gxc = lda_tda_lagrangian_gradient(
            functional,
            mol,
            mb,
            grid.expect("LDA grid checked above"),
            [&oo0, &oo0],
            [&relaxed, &relaxed],
            [&t_sym, &t_sym],
        )?;
        for (g, x) in grad.iter_mut().zip(gxc) {
            *g += x;
        }
    }
    if let (Some(model), Some(grid)) = (learned, grid) {
        // The same split of the relaxed density over the spin channels as
        // for the semilocal Lagrangian.
        let relaxed = axpy(&scale(&dm_zoo, 0.5), 0.25, &zsym);
        let gxc = crate::skala::skala_lagrangian_gradient(
            model,
            mol,
            mb,
            grid,
            [&oo0, &oo0],
            [&relaxed, &relaxed],
            [&t_sym, &t_sym],
        )?;
        for (g, x) in grad.iter_mut().zip(gxc) {
            *g += x;
        }
    }
    // Nuclear repulsion, with Z_eff so an ECP basis is consistent.
    for a in 0..natoms {
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let d = mol.atoms[a].position - mol.atoms[b].position;
            let r = d.norm();
            grad[a] -= d * (charges[a] * charges[b] / (r * r * r));
        }
    }
    Ok(grad)
}

/// Point-local LDA third-kernel contraction
/// `g_s = sum_tu f_xc^(3)[s,t,u] t_t t_u`.
///
/// This small public diagnostic is useful for independently checking the
/// third-order contraction without involving AO or nuclear derivatives.
#[doc(hidden)]
pub fn lda_third_density_contraction(
    functional: &ResolvedFunctional,
    rho: [f64; 2],
    transition: [f64; 2],
) -> Result<[f64; 2]> {
    if functional.parts.is_empty() || functional.family != Family::Lda {
        return Err(DftError::InvalidInput(
            "lda_third_density_contraction requires a semilocal LDA functional".to_string(),
        ));
    }
    let ra = [rho[0]];
    let rb = [rho[1]];
    let vars = GridVars {
        rho: [&ra, &rb],
        sigma: None,
        tau: None,
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 3, &mut out)?;
    let (ta, tb) = (transition[0], transition[1]);
    Ok([
        out.v3rho3[0][0] * ta * ta + 2.0 * out.v3rho3[1][0] * ta * tb + out.v3rho3[2][0] * tb * tb,
        out.v3rho3[1][0] * ta * ta + 2.0 * out.v3rho3[2][0] * ta * tb + out.v3rho3[3][0] * tb * tb,
    ])
}

fn lda_third_transition_potential(
    functional: &ResolvedFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p0: [&Matrix; 2],
    transition: [&Matrix; 2],
) -> Result<[Matrix; 2]> {
    let nao = mb.nao_sph;
    const BATCH: usize = 256;
    let ranges: Vec<(usize, usize)> = (0..grid.len())
        .step_by(BATCH)
        .map(|start| (start, (start + BATCH).min(grid.len())))
        .collect();
    let partials: Result<Vec<[Matrix; 2]>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let points = &grid.points[start..end];
            let weights = &grid.weights[start..end];
            let aos = eval_aos_full(mb, points, false, false);
            let npts = points.len();
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut trho = [vec![0.0; npts], vec![0.0; npts]];
            for spin in 0..2 {
                let x0 = aos.phi.matmul_seq(p0[spin]);
                let xt = aos.phi.matmul_seq(transition[spin]);
                for i in 0..npts {
                    for mu in 0..nao {
                        rho[spin][i] += x0[(i, mu)] * aos.phi[(i, mu)];
                        trho[spin][i] += xt[(i, mu)] * aos.phi[(i, mu)];
                    }
                    rho[spin][i] = rho[spin][i].max(0.0);
                }
            }
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: None,
                tau: None,
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 3, &mut out)?;
            let mut potentials = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
            for i in 0..npts {
                let (ta, tb) = (trho[0][i], trho[1][i]);
                let values = [
                    out.v3rho3[0][i] * ta * ta
                        + 2.0 * out.v3rho3[1][i] * ta * tb
                        + out.v3rho3[2][i] * tb * tb,
                    out.v3rho3[1][i] * ta * ta
                        + 2.0 * out.v3rho3[2][i] * ta * tb
                        + out.v3rho3[3][i] * tb * tb,
                ];
                for spin in 0..2 {
                    let factor = weights[i] * values[spin];
                    for mu in 0..nao {
                        let left = factor * aos.phi[(i, mu)];
                        for nu in 0..nao {
                            potentials[spin][(mu, nu)] += left * aos.phi[(i, nu)];
                        }
                    }
                }
            }
            Ok(potentials)
        })
        .collect();
    let mut total = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    for part in partials? {
        for spin in 0..2 {
            for (t, p) in total[spin]
                .as_mut_slice()
                .iter_mut()
                .zip(part[spin].as_slice())
            {
                *t += p;
            }
        }
    }
    Ok(total)
}

/// Full derivative of the LDA part of the stationary TDA Lagrangian,
/// including AO-centre motion, Becke-weight response, and owner-point motion.
fn lda_tda_lagrangian_gradient(
    functional: &ResolvedFunctional,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    ground: [&Matrix; 2],
    relaxed: [&Matrix; 2],
    transition: [&Matrix; 2],
) -> Result<Vec<Vec3>> {
    let natoms = mol.len();
    let nao = mb.nao_sph;
    let ao_atom = mb.sph_ao_atoms();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|atom| atom.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    const BATCH: usize = 256;
    let ranges: Vec<(usize, usize)> = (0..grid.len())
        .step_by(BATCH)
        .map(|start| (start, (start + BATCH).min(grid.len())))
        .collect();
    let partials: Result<Vec<Vec<Vec3>>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let points = &grid.points[start..end];
            let weights = &grid.weights[start..end];
            let raw_weights = &grid.raw_weights[start..end];
            let owners = &grid.atom_of[start..end];
            let aos = eval_aos_full(mb, points, true, false);
            let grad_ao = aos
                .grad
                .as_ref()
                .expect("LDA gradient requested AO gradients");
            let npts = points.len();
            let matrices = [ground, relaxed, transition];
            let mut x: [[Matrix; 2]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(npts, nao)));
            let mut fields: [[Vec<f64>; 2]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; npts]));
            let mut gradients: [[[Vec<f64>; 3]; 2]; 3] = std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; npts]))
            });
            for field in 0..3 {
                for spin in 0..2 {
                    x[field][spin] = aos.phi.matmul_seq(matrices[field][spin]);
                    for i in 0..npts {
                        for mu in 0..nao {
                            fields[field][spin][i] += x[field][spin][(i, mu)] * aos.phi[(i, mu)];
                        }
                    }
                    for axis in 0..3 {
                        for i in 0..npts {
                            let mut value = 0.0;
                            for mu in 0..nao {
                                value += x[field][spin][(i, mu)] * grad_ao[axis][(i, mu)];
                            }
                            gradients[field][spin][axis][i] = 2.0 * value;
                        }
                    }
                }
            }
            for spin in 0..2 {
                for value in fields[0][spin].iter_mut() {
                    *value = value.max(0.0);
                }
            }
            let vars = GridVars {
                rho: [&fields[0][0], &fields[0][1]],
                sigma: None,
                tau: None,
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 3, &mut out)?;
            let mut acc = vec![Vec3::zero(); natoms];
            for i in 0..npts {
                let (ra, rb) = (fields[1][0][i], fields[1][1][i]);
                let (ta, tb) = (fields[2][0][i], fields[2][1][i]);
                let (faa, fab, fbb) = (out.v2rho2[0][i], out.v2rho2[1][i], out.v2rho2[2][i]);
                let ktt = [
                    out.v3rho3[0][i] * ta * ta
                        + 2.0 * out.v3rho3[1][i] * ta * tb
                        + out.v3rho3[2][i] * tb * tb,
                    out.v3rho3[1][i] * ta * ta
                        + 2.0 * out.v3rho3[2][i] * ta * tb
                        + out.v3rho3[3][i] * tb * tb,
                ];
                let coefficient = [
                    [
                        out.vrho[0][i] + faa * ra + fab * rb + 0.25 * ktt[0],
                        out.vrho[1][i] + fab * ra + fbb * rb + 0.25 * ktt[1],
                    ],
                    [out.vrho[0][i], out.vrho[1][i]],
                    [0.5 * (faa * ta + fab * tb), 0.5 * (fab * ta + fbb * tb)],
                ];
                let ell = out.e[i]
                    + out.vrho[0][i] * ra
                    + out.vrho[1][i] * rb
                    + 0.25 * (faa * ta * ta + 2.0 * fab * ta * tb + fbb * tb * tb);

                // AO-centre motion at a fixed global grid point.
                for field in 0..3 {
                    for spin in 0..2 {
                        let c = coefficient[field][spin];
                        if c == 0.0 {
                            continue;
                        }
                        for mu in 0..nao {
                            let atom = ao_atom[mu];
                            let common = -2.0 * weights[i] * c * x[field][spin][(i, mu)];
                            acc[atom].x += common * grad_ao[0][(i, mu)];
                            acc[atom].y += common * grad_ao[1][(i, mu)];
                            acc[atom].z += common * grad_ao[2][(i, mu)];
                        }
                    }
                }

                // Becke weights and the translation-related motion of the
                // partition itself.
                let owner = owners[i];
                let (_, dwb) =
                    crate::grid::becke_weight_gradient(&centers, &aij, natoms, owner, points[i]);
                let mut dwb_sum = Vec3::zero();
                for atom in 0..natoms {
                    acc[atom] += dwb[atom] * (raw_weights[i] * ell);
                    dwb_sum += dwb[atom];
                }
                acc[owner] -= dwb_sum * (raw_weights[i] * ell);

                // The atom-centred quadrature point rides its owner through
                // every density field in the Lagrangian.
                let mut grad_ell = [0.0; 3];
                for axis in 0..3 {
                    for field in 0..3 {
                        for spin in 0..2 {
                            grad_ell[axis] +=
                                coefficient[field][spin] * gradients[field][spin][axis][i];
                        }
                    }
                }
                acc[owner] += Vec3::new(grad_ell[0], grad_ell[1], grad_ell[2]) * weights[i];
            }
            Ok(acc)
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for part in partials? {
        for (t, p) in total.iter_mut().zip(part) {
            *t += p;
        }
    }
    Ok(total)
}

fn block(c: &Matrix, start: usize, count: usize) -> Matrix {
    let mut m = Matrix::zeros(c.rows, count);
    for mu in 0..c.rows {
        for k in 0..count {
            m[(mu, k)] = c[(mu, start + k)];
        }
    }
    m
}

fn scale(m: &Matrix, f: f64) -> Matrix {
    let mut out = m.clone();
    for v in out.as_mut_slice() {
        *v *= f;
    }
    out
}

fn add(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    for (v, w) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *v += *w;
    }
    out
}

fn sub(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    for (v, w) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *v -= *w;
    }
    out
}

fn axpy(a: &Matrix, f: f64, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    for (v, w) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *v += f * *w;
    }
    out
}

/// `C^T M C`.
fn mo_transform(c: &Matrix, m: &Matrix) -> Matrix {
    c.transpose_matmul(&m.matmul(c))
}
