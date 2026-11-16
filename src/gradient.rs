// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic nuclear gradients for HF / DFT / hybrids / RSH:
//! `dE/dR_A = Tr[P ∂H_core] − Tr[W ∂S] + 2e-derivative terms + XC grid
//! term + ∂E_nn`, with the energy-weighted density
//! `W = Σ_σ Σ_i^occ ε_σi c_σi c_σiᵀ`.
//!
//! The XC grid term carries the FULL grid response: Becke weight
//! derivatives and the motion of the quadrature points with their host
//! atoms, not just the basis-function derivatives at fixed points. That
//! is what the translational sum rule gates to 1e-9; an earlier staged
//! version neglected it and was documented here as a ~1e-4 error.

use crate::basis::MolBasis;
use crate::error::Result;
use crate::grid::eval::eval_aos_full;
use crate::grid::MolecularGrid;
use crate::integrals::{deriv, schwarz_matrix, CoulombKernel};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{run_scf, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;

pub(crate) mod spinor;

pub struct GradientResult {
    pub scf: ScfResult,
    /// dE/dR_A in Hartree/Bohr.
    pub gradient: Vec<Vec3>,
}

/// Run the SCF and assemble the analytic gradient.
pub fn analytic_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
) -> Result<GradientResult> {
    let scf = run_scf(mol, mb, opts, functional, grid)?;
    let gradient = assemble_gradient(mol, mb, functional, grid, &scf, opts)?;
    Ok(GradientResult { scf, gradient })
}

/// Assemble the gradient for a converged SCF solution.
pub fn assemble_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
) -> Result<Vec<Vec3>> {
    if let Some(aux) = &opts.density_fit {
        return Err(crate::error::DftError::DerivOrderUnsupported {
            functional: format!("RI-J with auxiliary basis `{aux}`"),
            requested: 1,
            max_analytic: 0,
        });
    }
    // Energy-weighted density W = Σσ Σi fσi εσi Cσi Cσiᵀ. With Fermi-Dirac
    // occupations this is the derivative of the Mermin free energy: the
    // occupations are stationary, so no occupation-response term appears.
    // Prescribed nonthermal populations have df/dR=0 by their constraint,
    // and the same weighted Pulay term differentiates their internal energy.
    let mut w = Matrix::zeros(mb.nao_sph, mb.nao_sph);
    for spin in 0..2 {
        let c = &scf.mo_coeff[spin];
        for i in 0..c.cols {
            let f = scf.occupations[spin].get(i).copied().unwrap_or(0.0);
            if f == 0.0 {
                continue;
            }
            let e = f * scf.mo_energy[spin][i];
            for mu in 0..mb.nao_sph {
                let cmu = c[(mu, i)];
                if cmu == 0.0 {
                    continue;
                }
                for nu in 0..mb.nao_sph {
                    w[(mu, nu)] += e * cmu * c[(nu, i)];
                }
            }
        }
    }
    assemble_gradient_weighted(mol, mb, functional, grid, scf, opts, &w)
}

/// Assemble the gradient for the densities of `scf` and an explicit
/// energy-weighted density `w`, the matrix contracted with the overlap
/// derivative.
pub fn assemble_gradient_weighted(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    w: &Matrix,
) -> Result<Vec<Vec3>> {
    assemble_gradient_density(
        mol,
        mb,
        functional,
        grid,
        [&scf.density[0], &scf.density[1]],
        opts,
        w,
    )
}

/// Explicit moving-AO/grid energy derivative for supplied spin densities,
/// with a separately specified metric covector. This accepts an off-shell
/// density without inventing an SCF solution or a convergence certificate.
/// For constrained frozen actions, their chart/density chain is added by the
/// caller. `w=0` gives the derivative at fixed AO density coefficients.
pub fn assemble_gradient_density(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    p_spin: [&Matrix; 2],
    opts: &ScfOptions,
    w: &Matrix,
) -> Result<Vec<Vec3>> {
    if let Some(aux) = &opts.density_fit {
        return Err(crate::error::DftError::DerivOrderUnsupported {
            functional: format!("RI-J with auxiliary basis `{aux}`"),
            requested: 1,
            max_analytic: 0,
        });
    }
    let natoms = mol.len();
    if p_spin.iter().copied().chain(std::iter::once(w)).any(|p| {
        p.rows != mb.nao_sph
            || p.cols != mb.nao_sph
            || p.as_slice().iter().any(|x| !x.is_finite())
            || p.max_abs_diff(&p.transpose()) > 1e-10
    }) {
        return Err(crate::DftError::InvalidInput(
            "density gradient requires finite symmetric AO densities and metric covector".into(),
        ));
    }
    let mut p_tot = p_spin[0].clone();
    for (a, b) in p_tot.as_mut_slice().iter_mut().zip(p_spin[1].as_slice()) {
        *a += b;
    }

    // Z_eff with ECP cores subtracted; the semi-local potential itself
    // contributes its own derivative term below.
    let charges: Vec<f64> = mb.effective_charges(mol);

    // One-electron terms: Tr[P ∂T] + Tr[P ∂V] − Tr[W ∂S].
    let mut grad = crate::logger::timed("grad: one-electron", || {
        one_particle_gradient(mol, mb, &charges, &p_tot, w)
    });

    // Two-electron terms: full-range J/K, plus the long-range K for RSH.
    let schwarz = schwarz_matrix(mb, CoulombKernel::Coulomb);
    let g2 = crate::logger::timed("grad: ERI derivative", || {
        deriv::eri_gradient(
            mb,
            natoms,
            &p_tot,
            p_spin,
            CoulombKernel::Coulomb,
            1.0,
            functional.hybrid.cx,
            &schwarz,
            1e-12,
        )
    });
    for a in 0..natoms {
        grad[a] += g2[a];
    }
    if functional.hybrid.clr != 0.0 {
        let kernel = CoulombKernel::Erf {
            omega: functional.hybrid.omega,
        };
        let schwarz_lr = schwarz_matrix(mb, kernel);
        let glr = deriv::eri_gradient(
            mb,
            natoms,
            &p_tot,
            p_spin,
            kernel,
            0.0,
            functional.hybrid.clr,
            &schwarz_lr,
            1e-12,
        );
        for a in 0..natoms {
            grad[a] += glr[a];
        }
    }

    // XC grid term with full grid response (orbital + weight + point
    // motion) — the translational sum rule holds only with all three.
    if !functional.parts.is_empty() {
        let grid = grid.expect("grid required for XC gradient");
        let gxc = crate::logger::timed("grad: XC grid", || {
            xc_gradient_full(functional, mol, mb, grid, p_spin, natoms)
        })?;
        for a in 0..natoms {
            grad[a] += gxc[a];
        }
    }

    // Learned functional on the molecular grid: its fixed-density grid
    // terms; the Kohn-Sham matrices of the SCF make the energy stationary.
    if let Some(model) = &opts.skala {
        let grid = grid.expect("grid required for the learned functional");
        let g_learned = crate::logger::timed("grad: learned functional", || {
            crate::skala::skala_gradient(model, mol, mb, grid, p_spin)
        })?;
        for a in 0..natoms {
            grad[a] += g_learned[a];
        }
    }

    // ECP semi-local potential derivative.
    if !mb.ecps.is_empty() {
        let g_ecp = crate::integrals::ecp::ecp_gradient(mb, natoms, &p_tot);
        for (g, e) in grad.iter_mut().zip(g_ecp) {
            *g += e;
        }
    }

    // Self-consistent VV10: the variational energy makes the fixed-density
    // grid-response terms the exact derivative.
    if let Some(params) = opts.vv10 {
        let nlc_grid = opts.nonlocal_grid(mol)?;
        let g_nlc = crate::nlc::vv10_gradient(mol, mb, &nlc_grid, &p_tot, params)?;
        for (g, v) in grad.iter_mut().zip(g_nlc) {
            *g += v;
        }
    }

    // Continuum solvation: the polarization energy is variational in the
    // apparent charges, so only dv/dR and dS/dR survive (pcm.rs).
    if let Some(pcm_opts) = &opts.pcm {
        let solver = crate::pcm::PcmSolver::build(mol, mb, &charges, pcm_opts)?;
        let g_pcm = solver.gradient(mol, mb, &charges, &p_tot)?;
        for (g, v) in grad.iter_mut().zip(g_pcm) {
            *g += v;
        }
    }

    add_source_gradient(mol, mb, &charges, &p_tot, opts.external_field, &mut grad);

    Ok(grad)
}

/// Fixed-density one-electron work and the metric Pulay contraction.
fn one_particle_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    charges: &[f64],
    total: &Matrix,
    weighted: &Matrix,
) -> Vec<Vec3> {
    let natoms = mol.len();
    let mut gradient = deriv::kinetic_gradient(mb, total, natoms, 1.0);
    let nuclear = deriv::nuclear_gradient(mb, mol, charges, total, 1.0);
    let overlap = deriv::overlap_gradient(mb, weighted, natoms, -1.0);
    for a in 0..natoms {
        gradient[a] += nuclear[a] + overlap[a];
    }
    gradient
}

fn add_source_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    charges: &[f64],
    p_tot: &Matrix,
    field: Option<Vec3>,
    grad: &mut [Vec3],
) {
    let natoms = mol.len();
    // Uniform external field. The SCF adds `+E·r` to H_core and
    // `−Z_A R_A·E` to the nuclear energy; both move with the nuclei, so both
    // differentiate.
    //
    // The electronic half is `Σ_c E_c Tr[P ∂D^c/∂R_A]`. There is no
    // operator-center term: the dipole operator is anchored at a fixed
    // origin rather than on a nucleus, so unlike `∂V/∂R` only the two AO
    // centers move — which is also why same-atom pairs do not cancel and
    // the derivative matrices are not antisymmetric in the way `∂S/∂R` is.
    //
    // The nuclear half is simply `−Z_A E`, and it is the piece that makes
    // the translational sum rule non-trivial here: `Σ_A ∂E/∂R_A = −Q_tot E`
    // rather than zero, because a uniform field breaks translational
    // invariance for a charged system. For a neutral one it is zero again.
    if let Some(field) = field {
        let ddip = crate::integrals::deriv_mat::dipole_derivative_matrices(mb, natoms);
        for (comp, blocks) in ddip.iter().enumerate() {
            let f = field.get(comp);
            if f == 0.0 {
                continue;
            }
            for (a, per_atom) in blocks.iter().enumerate() {
                grad[a].x += f * p_tot.frobenius_dot(&per_atom[0]);
                grad[a].y += f * p_tot.frobenius_dot(&per_atom[1]);
                grad[a].z += f * p_tot.frobenius_dot(&per_atom[2]);
            }
        }
        for (a, g) in grad.iter_mut().enumerate() {
            *g += field * (-charges[a]);
        }
    }

    // Nuclear repulsion (Z_eff).
    for a in 0..natoms {
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let rab = mol.atoms[a].position - mol.atoms[b].position;
            let r = rab.norm();
            grad[a] += rab * (-(charges[a] * charges[b]) / (r * r * r));
        }
    }
}

/// Gradient of an energy that is not stationary in its density: the terms of
/// [`assemble_gradient_weighted`] for the densities of `scf` and the matrix
/// `w`, plus the derivative at fixed density of the Kohn-Sham matrices of
/// those densities contracted with the relaxation matrices `z`,
/// `sum_s Tr[z_s dF_s/dR]`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_gradient_relaxed(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    z: [&Matrix; 2],
    w: &Matrix,
) -> Result<Vec<Vec3>> {
    if opts.skala.is_some() || opts.vv10.is_some() || opts.pcm.is_some() {
        return Err(crate::error::DftError::Unsupported {
            feature: "relaxed-density gradient with a learned functional, nonlocal correlation \
                      or continuum solvation"
                .into(),
            missing: "the derivative of those Kohn-Sham terms at fixed density".into(),
        });
    }
    let mut grad = assemble_gradient_weighted(mol, mb, functional, grid, scf, opts, w)?;
    let natoms = mol.len();
    let mut z_tot = z[0].clone();
    for (t, b) in z_tot.as_mut_slice().iter_mut().zip(z[1].as_slice()) {
        *t += *b;
    }
    let charges: Vec<f64> = mb.effective_charges(mol);
    let accumulate = |grad: &mut Vec<Vec3>, part: Vec<Vec3>, scale: f64| {
        for (g, p) in grad.iter_mut().zip(part) {
            *g += p * scale;
        }
    };
    // One-electron terms are linear in the density.
    accumulate(
        &mut grad,
        deriv::kinetic_gradient(mb, &z_tot, natoms, 1.0),
        1.0,
    );
    accumulate(
        &mut grad,
        deriv::nuclear_gradient(mb, mol, &charges, &z_tot, 1.0),
        1.0,
    );
    // Two-electron cross term: P g z = [e2(P + z) - e2(P - z)] / 2 for the
    // quadratic form e2(P) = P g P / 2.
    let p_tot = scf.p_total();
    let combined = |sign: f64| -> (Matrix, [Matrix; 2]) {
        let mut total = p_tot.clone();
        for (t, b) in total.as_mut_slice().iter_mut().zip(z_tot.as_slice()) {
            *t += sign * b;
        }
        let mut spin = [scf.density[0].clone(), scf.density[1].clone()];
        for s in 0..2 {
            for (t, b) in spin[s].as_mut_slice().iter_mut().zip(z[s].as_slice()) {
                *t += sign * b;
            }
        }
        (total, spin)
    };
    let mut kernels = vec![(CoulombKernel::Coulomb, 1.0, functional.hybrid.cx)];
    if functional.hybrid.clr != 0.0 {
        kernels.push((
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
            0.0,
            functional.hybrid.clr,
        ));
    }
    for (kernel, coulomb, exchange) in kernels {
        let schwarz = schwarz_matrix(mb, kernel);
        for sign in [1.0, -1.0] {
            let (total, spin) = combined(sign);
            let part = deriv::eri_gradient(
                mb,
                natoms,
                &total,
                [&spin[0], &spin[1]],
                kernel,
                coulomb,
                exchange,
                &schwarz,
                1e-12,
            );
            accumulate(&mut grad, part, 0.5 * sign);
        }
    }
    // Derivative of the exchange-correlation matrices at fixed density.
    if !functional.parts.is_empty() {
        let grid = grid.expect("grid required for XC gradient");
        let p_spin = [&scf.density[0], &scf.density[1]];
        let terms = if functional.family == Family::MetaGga {
            crate::hessian::mgga_terms::xc_hessian_terms_mgga(functional, mol, mb, grid, p_spin)?
        } else {
            crate::hessian::xc_terms::xc_hessian_terms(functional, mol, mb, grid, p_spin)?
        };
        for spin in 0..2 {
            for (atom, derivatives) in terms.vxc_deriv[spin].iter().enumerate() {
                grad[atom].x += z[spin].frobenius_dot(&derivatives[0]);
                grad[atom].y += z[spin].frobenius_dot(&derivatives[1]);
                grad[atom].z += z[spin].frobenius_dot(&derivatives[2]);
            }
        }
    }
    if !mb.ecps.is_empty() {
        let part = crate::integrals::ecp::ecp_gradient(mb, natoms, &z_tot);
        accumulate(&mut grad, part, 1.0);
    }
    if let Some(field) = opts.external_field {
        let ddip = crate::integrals::deriv_mat::dipole_derivative_matrices(mb, natoms);
        for (comp, blocks) in ddip.iter().enumerate() {
            let f = field.get(comp);
            if f == 0.0 {
                continue;
            }
            for (a, per_atom) in blocks.iter().enumerate() {
                grad[a].x += f * z_tot.frobenius_dot(&per_atom[0]);
                grad[a].y += f * z_tot.frobenius_dot(&per_atom[1]);
                grad[a].z += f * z_tot.frobenius_dot(&per_atom[2]);
            }
        }
    }
    Ok(grad)
}

/// XC gradient with full grid response:
/// - orbital term `−2 Σ_i w_i Σ_s Σ_{μ∈A,ν} P^s_μν [vρ_s ∂φμ φν
///   + b_s·(∂∇φμ φν + ∂φμ ∇φν)]`, `b_s = 2vσ_ss ∇ρ_s + vσ_ab ∇ρ_s̄`;
/// - weight response `Σ_i raw_i (∂w^Becke_i/∂R_G) e_i`;
/// - point motion `Σ_{i∈A} w_i ∇_r e_i` (points ride their parent atom).
fn xc_gradient_full(
    functional: &ResolvedFunctional,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    natoms: usize,
) -> Result<Vec<Vec3>> {
    let meta = functional.family == Family::MetaGga;
    let gga = functional.family != Family::Lda;
    let nao = mb.nao_sph;
    let ao_atom = mb.sph_ao_atoms();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    const BATCH: usize = 256;
    let n = grid.len();
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(n)))
        .collect();
    let partials: Result<Vec<Vec<Vec3>>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let pts = &grid.points[start..end];
            let wts = &grid.weights[start..end];
            let raws = &grid.raw_weights[start..end];
            let owners = &grid.atom_of[start..end];
            let npts = pts.len();
            // Hessians are needed for the GGA orbital term AND for ∇σ in
            // the point-motion term.
            let batch = eval_aos_full(mb, pts, true, gga);
            let g = batch.grad.as_ref().expect("gradients");
            // Densities and gradients on the batch.
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut drho: [[Vec<f64>; 3]; 2] = Default::default();
            let mut x = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
            let mut y: [[Matrix; 3]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(0, 0)));
            for spin in 0..2 {
                x[spin] = batch.phi.matmul_seq(p[spin]);
                for i in 0..npts {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        acc += x[spin][(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = acc.max(0.0);
                }
                for ax in 0..3 {
                    if gga {
                        y[spin][ax] = g[ax].matmul_seq(p[spin]);
                    }
                    let mut v = vec![0.0; npts];
                    for i in 0..npts {
                        let mut acc = 0.0;
                        for mu in 0..nao {
                            acc += x[spin][(i, mu)] * g[ax][(i, mu)];
                        }
                        v[i] = 2.0 * acc;
                    }
                    drho[spin][ax] = v;
                }
            }
            // Kinetic-energy density tau_s = 1/2 sum_u (g_u P) . g_u rows
            // (the y matrices already hold g_u P for the GGA path).
            let tau_store;
            let tau_ref = if meta {
                let mut t = [vec![0.0; npts], vec![0.0; npts]];
                for spin in 0..2 {
                    for i in 0..npts {
                        let mut acc = 0.0;
                        for u in 0..3 {
                            for mu in 0..nao {
                                acc += y[spin][u][(i, mu)] * g[u][(i, mu)];
                            }
                        }
                        t[spin][i] = 0.5 * acc;
                    }
                }
                tau_store = t;
                Some([&tau_store[0][..], &tau_store[1][..]])
            } else {
                None
            };
            let sigma_store;
            let vars = if gga {
                let mut s3 = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
                for i in 0..npts {
                    let ga = [drho[0][0][i], drho[0][1][i], drho[0][2][i]];
                    let gb = [drho[1][0][i], drho[1][1][i], drho[1][2][i]];
                    s3[0][i] = ga.iter().map(|v| v * v).sum();
                    s3[1][i] = ga.iter().zip(&gb).map(|(a, b)| a * b).sum();
                    s3[2][i] = gb.iter().map(|v| v * v).sum();
                }
                sigma_store = s3;
                GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
                    tau: tau_ref,
                }
            } else {
                GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: None,
                    tau: None,
                }
            };
            let mut out = XcOut::default();
            #[cfg(test)]
            crate::xc::branch_freeze::set_batch(Some(start));
            let evaluated = functional.eval(&vars, 1, &mut out);
            #[cfg(test)]
            crate::xc::branch_freeze::set_batch(None);
            evaluated?;
            let hess = batch.hess.as_ref();
            let mut acc = vec![Vec3::zero(); natoms];
            // Hessian component lookup (ax, g) → packed index.
            let hidx = |u: usize, v: usize| -> usize {
                let (a, b) = if u <= v { (u, v) } else { (v, u) };
                match (a, b) {
                    (0, 0) => 0,
                    (0, 1) => 1,
                    (0, 2) => 2,
                    (1, 1) => 3,
                    (1, 2) => 4,
                    _ => 5,
                }
            };
            for i in 0..npts {
                let wi = wts[i];
                for spin in 0..2 {
                    let vr = out.vrho[spin][i];
                    let b_vec = if gga {
                        let (own, cross) = if spin == 0 {
                            (out.vsigma[0][i], out.vsigma[1][i])
                        } else {
                            (out.vsigma[2][i], out.vsigma[1][i])
                        };
                        let other = 1 - spin;
                        [
                            2.0 * own * drho[spin][0][i] + cross * drho[other][0][i],
                            2.0 * own * drho[spin][1][i] + cross * drho[other][1][i],
                            2.0 * own * drho[spin][2][i] + cross * drho[other][2][i],
                        ]
                    } else {
                        [0.0; 3]
                    };
                    for mu in 0..nao {
                        let xm = x[spin][(i, mu)];
                        if xm == 0.0 && !gga {
                            continue;
                        }
                        let atom = ao_atom[mu];
                        let mut gvec = [0.0; 3];
                        for ax in 0..3 {
                            let mut term = vr * g[ax][(i, mu)] * xm;
                            if gga {
                                let h = hess.expect("hessians for GGA");
                                for gx in 0..3 {
                                    term += b_vec[gx]
                                        * (h[hidx(ax, gx)][(i, mu)] * xm
                                            + g[ax][(i, mu)] * y[spin][gx][(i, mu)]);
                                }
                                // Meta-GGA: d tau/dA contracts the AO
                                // second derivatives with the g P rows
                                // (1/2 vtau; the -2 loop prefactor makes
                                // -w vtau sum_u h_{xu} y_u).
                                if meta {
                                    let vt = out.vtau[spin][i];
                                    if vt != 0.0 {
                                        let mut tterm = 0.0;
                                        for u in 0..3 {
                                            tterm += h[hidx(ax, u)][(i, mu)] * y[spin][u][(i, mu)];
                                        }
                                        term += 0.5 * vt * tterm;
                                    }
                                }
                            }
                            gvec[ax] = term;
                        }
                        acc[atom] += Vec3::new(
                            -2.0 * wi * gvec[0],
                            -2.0 * wi * gvec[1],
                            -2.0 * wi * gvec[2],
                        );
                    }
                }
            }
            // Grid response: weight derivatives + point motion.
            for i in 0..npts {
                let e_i = out.e[i];
                let owner = owners[i];
                // Weight response (all atoms).
                let (_, dwb) =
                    crate::grid::becke_weight_gradient(&centers, &aij, natoms, owner, pts[i]);
                let mut dwb_sum = Vec3::zero();
                for gat in 0..natoms {
                    acc[gat] += dwb[gat] * (raws[i] * e_i);
                    dwb_sum += dwb[gat];
                }
                // The moving point also carries the Becke weight through
                // space: ∇_r w = −Σ_G ∂w/∂R_G (translation invariance).
                acc[owner] -= dwb_sum * (raws[i] * e_i);
                // Point motion: the point rides its parent atom through the
                // integrand: + w_i ∇_r e_i.
                let mut grad_e = Vec3::new(
                    out.vrho[0][i] * drho[0][0][i] + out.vrho[1][i] * drho[1][0][i],
                    out.vrho[0][i] * drho[0][1][i] + out.vrho[1][i] * drho[1][1][i],
                    out.vrho[0][i] * drho[0][2][i] + out.vrho[1][i] * drho[1][2][i],
                );
                if gga {
                    let h = hess.expect("hessians for GGA");
                    // ρ-hessian per spin: 2(Σ ∂uφ P ∂vφ + Σ φP ∂u∂vφ).
                    let mut hr = [[[0.0; 3]; 3]; 2];
                    for spin in 0..2 {
                        for u in 0..3 {
                            for v in u..3 {
                                let mut t1 = 0.0;
                                let mut t2 = 0.0;
                                for mu in 0..nao {
                                    t1 += y[spin][u][(i, mu)] * g[v][(i, mu)];
                                    t2 += x[spin][(i, mu)] * h[hidx(u, v)][(i, mu)];
                                }
                                let val = 2.0 * (t1 + t2);
                                hr[spin][u][v] = val;
                                hr[spin][v][u] = val;
                            }
                        }
                    }
                    let dr = |s: usize, v: usize| drho[s][v][i];
                    for u in 0..3 {
                        let mut dsig = [0.0; 3]; // aa, ab, bb
                        for v in 0..3 {
                            dsig[0] += 2.0 * hr[0][u][v] * dr(0, v);
                            dsig[1] += hr[0][u][v] * dr(1, v) + dr(0, v) * hr[1][u][v];
                            dsig[2] += 2.0 * hr[1][u][v] * dr(1, v);
                        }
                        let mut ge = out.vsigma[0][i] * dsig[0]
                            + out.vsigma[1][i] * dsig[1]
                            + out.vsigma[2][i] * dsig[2];
                        // Meta-GGA point motion: (grad_r tau_s)_u =
                        // sum_v sum_mu h_{uv,mu} y_v(i,mu).
                        if meta {
                            let h = hess.expect("hessians for meta");
                            for spin in 0..2 {
                                let vt = out.vtau[spin][i];
                                if vt == 0.0 {
                                    continue;
                                }
                                let mut dtau = 0.0;
                                for v in 0..3 {
                                    for mu in 0..nao {
                                        dtau += h[hidx(u, v)][(i, mu)] * y[spin][v][(i, mu)];
                                    }
                                }
                                ge += vt * dtau;
                            }
                        }
                        match u {
                            0 => grad_e.x += ge,
                            1 => grad_e.y += ge,
                            _ => grad_e.z += ge,
                        }
                    }
                }
                acc[owner] += grad_e * wts[i];
            }
            Ok(acc)
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for p in partials? {
        for (t, v) in total.iter_mut().zip(p) {
            *t += v;
        }
    }
    Ok(total)
}

/// Reference numerical gradient: central differences of the total SCF
/// energy with geometry-rebuilt basis and grid (includes ALL grid
/// response). Slow — test use only.
pub fn numerical_gradient(
    mol: &Molecule,
    basis_set: &crate::basis::BasisSet,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid_opts: Option<&crate::grid::GridOptions>,
    step: f64,
) -> Result<Vec<Vec3>> {
    let natoms = mol.len();
    let energy_at = |geom: &Molecule| -> Result<f64> {
        let mb = MolBasis::build(geom, basis_set)?;
        let grid = match grid_opts {
            Some(go) => Some(MolecularGrid::build(geom, go)?),
            None => None,
        };
        Ok(run_scf(geom, &mb, opts, functional, grid.as_ref())?.energy)
    };
    let mut grad = vec![Vec3::zero(); natoms];
    for a in 0..natoms {
        for ax in 0..3 {
            let mut mp = mol.clone();
            let mut mm = mol.clone();
            match ax {
                0 => {
                    mp.atoms[a].position.x += step;
                    mm.atoms[a].position.x -= step;
                }
                1 => {
                    mp.atoms[a].position.y += step;
                    mm.atoms[a].position.y -= step;
                }
                _ => {
                    mp.atoms[a].position.z += step;
                    mm.atoms[a].position.z -= step;
                }
            }
            let d = (energy_at(&mp)? - energy_at(&mm)?) / (2.0 * step);
            match ax {
                0 => grad[a].x = d,
                1 => grad[a].y = d,
                _ => grad[a].z = d,
            }
        }
    }
    Ok(grad)
}
