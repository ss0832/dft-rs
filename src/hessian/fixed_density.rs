// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared explicit moving-AO/grid Hessian at arbitrary fixed spin density.
//! This is an off-shell scalar derivative, not a converged-state Hessian.
//! Density transport, the overlap metric covector and orbital relaxation
//! must be supplied by the owning stationary or constrained model.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::MolecularGrid;
use crate::integrals::deriv2::{
    dipole_second_contraction, eri_second_contraction, kinetic_second_contraction,
    nuclear_repulsion_hessian, nuclear_second_contraction,
};
use crate::integrals::deriv_mat::{
    eri_derivative_fock, kinetic_derivative_matrices, nuclear_derivative_matrices, AtomDerivs,
};
use crate::integrals::{schwarz_matrix, CoulombKernel};
use crate::linalg::Matrix;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

pub struct FixedDensityHessianTerms {
    /// Explicit Cartesian Hessian, Hartree/Bohr^2, including the nuclei.
    /// No overlap-energy-weighted, density-response or CPKS term is included.
    pub hessian: Matrix,
    /// Explicit nuclear AO Fock derivatives [spin][atom][axis], Hartree/Bohr.
    pub fock_derivatives: [Vec<AtomDerivs>; 2],
}

/// Differentiate the shared physical scalar with Pa/Pb held fixed as AO
/// coefficient matrices. A constrained chart must add its own density chain;
/// a stationary SCF caller adds its metric and relaxed response exactly once.
/// This accepts no ScfResult and does not certify electronic convergence.
pub fn fixed_density_hessian_terms(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    p_spin: [&Matrix; 2],
    grid: Option<&MolecularGrid>,
) -> Result<FixedDensityHessianTerms> {
    if let Some(aux) = &opts.density_fit {
        return Err(DftError::DerivOrderUnsupported {
            functional: format!("RI-J with auxiliary basis `{aux}`"),
            requested: 2,
            max_analytic: 0,
        });
    }
    let has_xc = !functional.parts.is_empty();
    if has_xc && grid.is_none() {
        return Err(DftError::InvalidInput(
            "the analytic Hessian needs the SCF quadrature grid for XC functionals".to_string(),
        ));
    }

    let natoms = mol.len();
    let n = mb.nao_sph;
    if p_spin.iter().any(|p| {
        p.rows != n
            || p.cols != n
            || p.as_slice().iter().any(|v| !v.is_finite())
            || p.max_abs_diff(&p.transpose()) > 1e-10
    }) {
        return Err(DftError::InvalidInput(
            "fixed-density Hessian requires finite symmetric AO spin-density matrices".into(),
        ));
    }
    let mut p_tot = p_spin[0].clone();
    for (a, b) in p_tot.as_mut_slice().iter_mut().zip(p_spin[1].as_slice()) {
        *a += *b;
    }
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let omega = functional.hybrid.omega;
    // ECP nuclear repulsion uses the same effective charges as its
    // attraction and nonlocal-projector derivatives.
    let charges = mb.effective_charges(mol);
    // Explicit nuclear derivatives with both AO spin densities held fixed.
    let schwarz = schwarz_matrix(mb, CoulombKernel::Coulomb);
    let mut h = kinetic_second_contraction(mb, &p_tot, natoms);
    let hv = nuclear_second_contraction(mb, mol, &charges, &p_tot);
    let h2e = crate::logger::timed("hess: ERI 2nd derivatives", || {
        eri_second_contraction(
            mb,
            natoms,
            &p_tot,
            p_spin,
            CoulombKernel::Coulomb,
            1.0,
            cx,
            &schwarz,
            1e-14,
        )
    });
    let hnn = nuclear_repulsion_hessian(mol, &charges);
    for (t, v) in h.as_mut_slice().iter_mut().zip(hv.as_slice()) {
        *t += *v;
    }
    for (t, v) in h.as_mut_slice().iter_mut().zip(h2e.as_slice()) {
        *t += *v;
    }
    for (t, v) in h.as_mut_slice().iter_mut().zip(hnn.as_slice()) {
        *t += *v;
    }
    // ECP semi-local potential: part of H_core, so it contributes a
    // skeleton second derivative here and a perturbed-Fock term below.
    // Both are zero (and cheap) when the basis carries no potential.
    let hecp = crate::integrals::ecp::ecp_second_contraction(mb, natoms, &p_tot);
    for (t, v) in h.as_mut_slice().iter_mut().zip(hecp.as_slice()) {
        *t += *v;
    }
    // Uniform field: H_core contains +E.c D_c, where D_c is anchored at a
    // fixed laboratory origin. Its nuclear term -sum_A Z_A E.R_A is linear
    // in the coordinates and has no second derivative.
    if let Some(field) = opts.external_field {
        for component in 0..3 {
            let scale = field.get(component);
            if scale == 0.0 {
                continue;
            }
            let hd = dipole_second_contraction(mb, &p_tot, natoms, component);
            for (target, value) in h.as_mut_slice().iter_mut().zip(hd.as_slice()) {
                *target += scale * value;
            }
        }
    }
    if clr != 0.0 {
        let schwarz_lr = schwarz_matrix(mb, CoulombKernel::Erf { omega });
        let h2e_lr = eri_second_contraction(
            mb,
            natoms,
            &p_tot,
            p_spin,
            CoulombKernel::Erf { omega },
            0.0,
            clr,
            &schwarz_lr,
            1e-14,
        );
        for (t, v) in h.as_mut_slice().iter_mut().zip(h2e_lr.as_slice()) {
            *t += *v;
        }
    }
    // XC skeleton + derivative potential matrices (LDA/GGA in xc_terms,
    // the tau-channel meta version in its own module).
    let xc = if has_xc {
        let g = grid.expect("grid checked above");
        let terms = if functional.family == crate::xc::Family::MetaGga {
            crate::hessian::mgga_terms::xc_hessian_terms_mgga(functional, mol, mb, g, p_spin)?
        } else {
            crate::hessian::xc_terms::xc_hessian_terms(functional, mol, mb, g, p_spin)?
        };
        for (t, v) in h.as_mut_slice().iter_mut().zip(terms.skeleton.as_slice()) {
            *t += *v;
        }
        Some(terms)
    } else {
        None
    };
    // Learned functional: skeleton and derivative potential matrices (the
    // CPKS response side rides the kernel of the solver).
    let learned = match (&opts.skala, grid) {
        (Some(model), Some(g)) => {
            let terms =
                crate::hessian::skala_terms::skala_hessian_terms(model, mol, mb, g, p_spin)?;
            for (t, v) in h.as_mut_slice().iter_mut().zip(terms.skeleton.as_slice()) {
                *t += *v;
            }
            Some(terms)
        }
        (Some(_), None) => {
            return Err(DftError::InvalidInput(
                "the analytic Hessian needs the SCF quadrature grid for the learned functional"
                    .to_string(),
            ));
        }
        (None, _) => None,
    };
    // SC-VV10: nonlocal skeleton + D_a V_nlc on the dedicated coarse grid
    // (the CPKS response side rides the solver's f_nlc kernel).
    let nlc = match opts.vv10 {
        Some(params) => {
            let nlc_grid = opts.nonlocal_grid(mol)?;
            let terms =
                crate::hessian::nlc_terms::nlc_hessian_terms(mol, mb, &nlc_grid, p_spin, params)?;
            for (t, v) in h.as_mut_slice().iter_mut().zip(terms.skeleton.as_slice()) {
                *t += *v;
            }
            Some(terms)
        }
        None => None,
    };

    crate::logger::log("hess: skeleton done");
    // ---------------- Derivative matrices ----------------
    let dt = kinetic_derivative_matrices(mb, natoms);
    let dv = nuclear_derivative_matrices(mb, mol, &charges);
    let du_ecp = crate::integrals::ecp::ecp_derivative_matrices(mb, natoms);
    let ddip_field = opts
        .external_field
        .map(|_| crate::integrals::deriv_mat::dipole_derivative_matrices(mb, natoms));
    let (dj, dk) = crate::logger::timed("hess: ERI 1st-deriv Fock", || {
        eri_derivative_fock(
            mb,
            natoms,
            &p_tot,
            p_spin,
            CoulombKernel::Coulomb,
            &schwarz,
            1e-14,
        )
    });
    let dk_lr = if clr != 0.0 {
        let schwarz_lr = schwarz_matrix(mb, CoulombKernel::Erf { omega });
        Some(
            eri_derivative_fock(
                mb,
                natoms,
                &p_tot,
                p_spin,
                CoulombKernel::Erf { omega },
                &schwarz_lr,
                1e-14,
            )
            .1,
        )
    } else {
        None
    };
    // Continuum solvation: the reaction field's skeleton second derivative
    // and its own entry in the Fock derivative. The CPKS coupling (the
    // solvent relaxing with the density) rides the response solver's
    // kernel, which already carries `pcm::fock_response`.
    let pcm = match &opts.pcm {
        Some(po) => {
            let solver = crate::pcm::PcmSolver::build(mol, mb, &charges, po)?;
            let hs = solver.hessian_skeleton(mol, mb, &charges, &p_tot)?;
            let fd = solver.fock_derivative(mol, mb, &charges, &p_tot)?;
            for (t, v) in h.as_mut_slice().iter_mut().zip(hs.as_slice()) {
                *t += *v;
            }
            Some(fd)
        }
        None => None,
    };
    // F(skel)[atom][ax][spin] and S(1)[atom][ax].
    let fskel = |a: usize, ax: usize, spin: usize| -> Matrix {
        let mut f = dt[a][ax].clone();
        for (fv, vv) in f.as_mut_slice().iter_mut().zip(dv[a][ax].as_slice()) {
            *fv += *vv;
        }
        for (fv, uv) in f.as_mut_slice().iter_mut().zip(du_ecp[a][ax].as_slice()) {
            *fv += *uv;
        }
        if let (Some(field), Some(ddip)) = (opts.external_field, ddip_field.as_ref()) {
            for component in 0..3 {
                let scale = field.get(component);
                if scale == 0.0 {
                    continue;
                }
                for (fv, dv) in f
                    .as_mut_slice()
                    .iter_mut()
                    .zip(ddip[component][a][ax].as_slice())
                {
                    *fv += scale * dv;
                }
            }
        }
        for (fv, jv) in f.as_mut_slice().iter_mut().zip(dj[a][ax].as_slice()) {
            *fv += *jv;
        }
        if cx != 0.0 {
            for (fv, kv) in f.as_mut_slice().iter_mut().zip(dk[spin][a][ax].as_slice()) {
                *fv -= cx * kv;
            }
        }
        if let Some(dk_lr) = &dk_lr {
            for (fv, kv) in f
                .as_mut_slice()
                .iter_mut()
                .zip(dk_lr[spin][a][ax].as_slice())
            {
                *fv -= clr * kv;
            }
        }
        if let Some(xc) = &xc {
            for (fv, xv) in f
                .as_mut_slice()
                .iter_mut()
                .zip(xc.vxc_deriv[spin][a][ax].as_slice())
            {
                *fv += *xv;
            }
        }
        if let Some(learned) = &learned {
            for (fv, xv) in f
                .as_mut_slice()
                .iter_mut()
                .zip(learned.vxc_deriv[spin][a][ax].as_slice())
            {
                *fv += *xv;
            }
        }
        if let Some(nlc) = &nlc {
            // V_nlc is a total-density potential: the same derivative
            // matrix enters both spin Fock derivatives.
            for (fv, xv) in f
                .as_mut_slice()
                .iter_mut()
                .zip(nlc.v_deriv[a][ax].as_slice())
            {
                *fv += *xv;
            }
        }
        if let Some(pcm) = &pcm {
            // The reaction field is a total-density operator too.
            for (fv, xv) in f.as_mut_slice().iter_mut().zip(pcm[a][ax].as_slice()) {
                *fv += *xv;
            }
        }
        f
    };

    let fock_derivatives = std::array::from_fn(|spin| {
        (0..natoms)
            .map(|atom| std::array::from_fn(|axis| fskel(atom, axis, spin)))
            .collect()
    });
    Ok(FixedDensityHessianTerms {
        hessian: h,
        fock_derivatives,
    })
}
