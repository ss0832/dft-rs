// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! TDDFT/TDHF excitation energies in the Tamm-Dancoff approximation
//! (Casida A-matrix diagonalization by block Davidson).
//!
//! Spin-resolved A-matrix product:
//! `(A u)_{ai,s} = (eps_a - eps_i) u_{ai,s}
//!   + [C_v^T (J[rho_X] - cx K[X_s] - clr K_lr[X_s] + fxc^s[rho_X]) C_o]_ai`
//! with the UNSYMMETRIZED transition density X_s = C_v U_s C_o^T. Note the
//! ground-state response operator (`hess_vec`) applies J/K/fxc to the
//!
//! symmetrized X + X^T - for real orbitals that is the (A+B) product used
//! by static CPKS. TDA needs A alone, hence the dedicated product here with
//! a non-symmetric J/K contraction. The contraction uses the same
//! direct/in-core engine as SCF, so excited states do not impose a second,
//! hard-coded ERI memory cliff.
//!
//! Closed-shell references reduce to the spin-resolved formula with
//! X_beta = +X_alpha (singlet) or X_beta = -X_alpha (triplet); the
//! spin-adapted couplings 2J + (f_aa + f_ab) resp. 0J + (f_aa - f_ab)
//! then emerge automatically.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::MolecularGrid;
use crate::integrals::CoulombKernel;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::response::ResponseSolver;
use crate::scf::{JkEngine, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Excitation manifold for closed-shell references. Open-shell (UKS/UHF)
/// references use `SpinResolved` (both spin blocks vary independently).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TdaSpin {
    Singlet,
    Triplet,
    SpinResolved,
}

/// One TDA excited state.
pub struct TdaState {
    /// Excitation energy (Hartree).
    pub energy: f64,
    /// Amplitudes per spin: X[s] is n_vir x n_occ (row a, col i).
    pub x: [Matrix; 2],
    /// Which excitation manifold this state was solved in. Anything that
    /// re-derives a spin-adapted quantity from `x` (the excited-state
    /// gradient does) must CHECK this rather than assume singlet: the
    /// amplitudes of the three manifolds are numerically indistinguishable
    /// but mean different things.
    pub spin: TdaSpin,
    /// True when this came from a Tamm-Dancoff solve (`Y = 0`), false for
    /// full RPA. The response densities of the two differ — TDA is the
    /// limit where `X+Y` and `X-Y` both collapse to `X` — so a consumer
    /// holding only `X` needs to know which it has.
    pub tamm_dancoff: bool,
    /// Transition dipole (a.u.).
    pub transition_dipole: [f64; 3],
    /// Length-gauge oscillator strength f = (2/3) omega |d|^2.
    pub oscillator_strength: f64,
    /// Transition MAGNETIC dipole `<0| L |n>` about the coordinate origin
    /// (a.u.). Real here; the physical operator carries a factor `-i/2`,
    /// which the rotatory strengths below apply.
    pub transition_angular_momentum: [f64; 3],
    /// Transition momentum `<0| grad |n>` (a.u.), the velocity-gauge
    /// counterpart of the electric transition dipole.
    pub transition_momentum: [f64; 3],
    /// Length-gauge rotatory strength `R = Im[<0|mu|n> . <n|m|0>]`, in
    /// atomic units. Multiply by 471.44 for the customary 1e-40 cgs.
    ///
    /// ECD spectra are the imaginary part of a product of an electric and
    /// a magnetic transition moment, so unlike the oscillator strength the
    /// SIGN is the observable: it is what distinguishes enantiomers.
    ///
    /// This gauge is ORIGIN-DEPENDENT for an incomplete basis. Compare it
    /// with `rotatory_strength_velocity`, which is not: the gap between
    /// them measures how far the basis is from the exact-state identity
    /// that makes them equal, and is the error bar on the result.
    pub rotatory_strength: f64,
    /// Velocity-gauge rotatory strength,
    /// `R = -Im[<0|grad|n> . <n|L|0>] / omega`. ORIGIN-INDEPENDENT.
    pub rotatory_strength_velocity: f64,
}

impl TdaState {
    /// Fix the real state phase by its largest electric transition component,
    /// falling back to momentum and angular momentum for dark electric lines.
    /// A perturbative caller supplies parent moments, keeping the phase fixed
    /// as correlation coefficients vary. All amplitudes/moments transform
    /// together; intensities and energies are unchanged.
    pub(crate) fn canonicalize_phase(&mut self, anchor: Option<[[f64; 3]; 3]>) {
        let anchors = anchor.unwrap_or([
            self.transition_dipole,
            self.transition_momentum,
            self.transition_angular_momentum,
        ]);
        let mut sign = 1.0;
        for vector in anchors {
            let index = (0..3)
                .max_by(|&a, &b| vector[a].abs().total_cmp(&vector[b].abs()))
                .unwrap();
            if vector[index] != 0.0 {
                sign = vector[index].signum();
                break;
            }
        }
        if sign < 0.0 {
            for matrix in &mut self.x {
                for value in matrix.as_mut_slice() {
                    *value = -*value;
                }
            }
            for vector in [
                &mut self.transition_dipole,
                &mut self.transition_momentum,
                &mut self.transition_angular_momentum,
            ] {
                for value in vector {
                    *value = -*value;
                }
            }
        }
    }
}

pub struct TdaSolver<'a> {
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    scf: &'a ScfResult,
    /// fxc contraction machinery (shares the SCF grid).
    fxc: Option<ResponseSolver<'a>>,
    jk: JkEngine,
    jk_lr: Option<JkEngine>,
    spin: TdaSpin,
    n_occ: [usize; 2],
    n_vir: [usize; 2],
    pub max_iter: usize,
    pub tol: f64,
}

impl<'a> TdaSolver<'a> {
    pub fn new(
        mol: &'a Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        opts: &'a ScfOptions,
        spin: TdaSpin,
    ) -> Result<Self> {
        // Every entry, including pure HF (without an XC ResponseSolver), uses
        // an occupied/virtual integer projector. Thermal population-weighted
        // dynamics require a different operator, not just the same residual.
        super::diagnostics::reference_diagnostics(scf)?;
        if scf.unrestricted && spin != TdaSpin::SpinResolved {
            return Err(DftError::InvalidInput(
                "singlet/triplet TDA needs a restricted (closed-shell) reference; \
                 use SpinResolved for UKS/UHF"
                    .to_string(),
            ));
        }
        if !scf.unrestricted && spin == TdaSpin::SpinResolved {
            return Err(DftError::InvalidInput(
                "SpinResolved TDA expects an unrestricted reference".to_string(),
            ));
        }
        // ResponseSolver owns no reference to its options, so this local copy
        // safely keeps its auxiliary response engines on the same explicit
        // direct/in-core path as the TDA-specific non-symmetric contraction.
        let mut response_opts = opts.clone();
        response_opts.eri_mode = opts.eri_mode;
        let has_kernel = !functional.parts.is_empty()
            || opts.vv10.is_some()
            || opts.pcm.is_some()
            || opts.skala.is_some();
        let fxc = if has_kernel {
            Some(ResponseSolver::new(
                mol,
                mb,
                functional,
                grid,
                scf,
                &response_opts,
            )?)
        } else {
            None
        };
        let reference_total = scf.p_total();
        let reference_spin = [&scf.density[0], &scf.density[1]];
        let jk = JkEngine::from_options(mol, mb, opts)?
            .with_reference_screening(&reference_total, reference_spin);
        let jk_lr = if functional.hybrid.clr != 0.0 {
            Some(
                JkEngine::new(
                    mb,
                    opts.eri_mode,
                    CoulombKernel::Erf {
                        omega: functional.hybrid.omega,
                    },
                )?
                .with_reference_screening(&reference_total, reference_spin),
            )
        } else {
            None
        };
        let nmo = [scf.mo_coeff[0].cols, scf.mo_coeff[1].cols];
        Ok(Self {
            mb,
            functional,
            scf,
            fxc,
            jk,
            jk_lr,
            spin,
            n_occ: [scf.n_alpha, scf.n_beta],
            n_vir: [nmo[0] - scf.n_alpha, nmo[1] - scf.n_beta],
            max_iter: 60,
            tol: 1e-7,
        })
    }

    /// Number of independent amplitude components.
    pub fn dim(&self) -> usize {
        match self.spin {
            TdaSpin::SpinResolved => self.n_occ[0] * self.n_vir[0] + self.n_occ[1] * self.n_vir[1],
            _ => self.n_occ[0] * self.n_vir[0],
        }
    }

    /// Orbital-energy differences on the amplitude vector (preconditioner
    /// and the diagonal part of A).
    pub(crate) fn diag(&self) -> Vec<f64> {
        let mut d = Vec::with_capacity(self.dim());
        let spins: &[usize] = match self.spin {
            TdaSpin::SpinResolved => &[0, 1],
            _ => &[0],
        };
        for &s in spins {
            let (no, nv) = (self.n_occ[s], self.n_vir[s]);
            let eps = &self.scf.mo_energy[s];
            for a in 0..nv {
                for i in 0..no {
                    d.push(eps[no + a] - eps[i]);
                }
            }
        }
        d
    }

    /// Per-spin AO transition densities X_s = C_v U_s C_o^T (unsymmetrized).
    fn x_ao(&self, u: &[f64]) -> [Matrix; 2] {
        let nao = self.mb.nao_sph;
        let mut out = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
        let mut off = 0;
        let spins: &[usize] = match self.spin {
            TdaSpin::SpinResolved => &[0, 1],
            _ => &[0],
        };
        for &s in spins {
            let (no, nv) = (self.n_occ[s], self.n_vir[s]);
            let c = &self.scf.mo_coeff[s];
            for a in 0..nv {
                for i in 0..no {
                    let uai = u[off + a * no + i];
                    if uai == 0.0 {
                        continue;
                    }
                    for mu in 0..nao {
                        let cva = c[(mu, no + a)];
                        if cva == 0.0 {
                            continue;
                        }
                        for nu in 0..nao {
                            out[s][(mu, nu)] += uai * cva * c[(nu, i)];
                        }
                    }
                }
            }
            off += no * nv;
        }
        match self.spin {
            TdaSpin::Singlet => out[1] = out[0].clone(),
            TdaSpin::Triplet => {
                let mut neg = out[0].clone();
                for v in neg.as_mut_slice() {
                    *v = -*v;
                }
                out[1] = neg;
            }
            TdaSpin::SpinResolved => {}
        }
        out
    }

    /// The TDA matrix-vector product A.u.
    pub fn a_vec(&self, u: &[f64]) -> Result<Vec<f64>> {
        let cx = self.functional.hybrid.cx;
        let clr = self.functional.hybrid.clr;
        let nao = self.mb.nao_sph;
        let x = self.x_ao(u);
        // Coulomb on the total transition density (symmetric part suffices
        // for J, but the tensor contraction handles X as-is).
        let mut x_tot = x[0].clone();
        for (t, v) in x_tot.as_mut_slice().iter_mut().zip(x[1].as_slice()) {
            *t += *v;
        }
        let (j_tot, k) = self.jk.build(self.mb, &x_tot, [&x[0], &x[1]], cx != 0.0)?;
        let k_lr = match &self.jk_lr {
            Some(engine) => Some(engine.build_k_only(self.mb, [&x[0], &x[1]])?),
            None => None,
        };
        let mut f = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
        for s in 0..2 {
            let mut m = j_tot.clone();
            if cx != 0.0 {
                for (mv, kv) in m.as_mut_slice().iter_mut().zip(k[s].as_slice()) {
                    *mv -= cx * kv;
                }
            }
            if let Some(k_lr) = &k_lr {
                for (mv, kv) in m.as_mut_slice().iter_mut().zip(k_lr[s].as_slice()) {
                    *mv -= clr * kv;
                }
            }
            f[s] = m;
        }
        if let Some(fxc) = &self.fxc {
            // fxc is linear in dp and depends only on its SYMMETRIC part
            // (rho- and grad-rho contractions are mu-nu symmetric); the
            // contraction machinery assumes a symmetric input, so hand it
            // (X + X^T)/2 - exactly equivalent for the A matrix.
            let mut x_sym = [x[0].clone(), x[1].clone()];
            for s in 0..2 {
                let n = x_sym[s].rows;
                for r in 0..n {
                    for c in (r + 1)..n {
                        let avg = 0.5 * (x[s][(r, c)] + x[s][(c, r)]);
                        x_sym[s][(r, c)] = avg;
                        x_sym[s][(c, r)] = avg;
                    }
                }
            }
            let vxc = fxc.fxc_contraction(&x_sym)?;
            for s in 0..2 {
                for (mv, xv) in f[s].as_mut_slice().iter_mut().zip(vxc[s].as_slice()) {
                    *mv += *xv;
                }
            }
        }
        // Project per spin and add the orbital-energy diagonal.
        let mut out = vec![0.0; self.dim()];
        let mut off = 0;
        let spins: &[usize] = match self.spin {
            TdaSpin::SpinResolved => &[0, 1],
            _ => &[0],
        };
        for &s in spins {
            let (no, nv) = (self.n_occ[s], self.n_vir[s]);
            let c = &self.scf.mo_coeff[s];
            let eps = &self.scf.mo_energy[s];
            for a in 0..nv {
                for i in 0..no {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        let cva = c[(mu, no + a)];
                        if cva == 0.0 {
                            continue;
                        }
                        for nu in 0..nao {
                            acc += cva * f[s][(mu, nu)] * c[(nu, i)];
                        }
                    }
                    let idx = off + a * no + i;
                    out[idx] = acc + (eps[no + a] - eps[i]) * u[idx];
                }
            }
            off += no * nv;
        }
        Ok(out)
    }

    /// Lowest `nroots` TDA excitations: exact diagonalization for small
    /// amplitude spaces, block Davidson above.
    pub fn solve(&self, nroots: usize) -> Result<Vec<TdaState>> {
        if self.dim() <= 512 {
            self.solve_dense(nroots)
        } else {
            self.solve_davidson(nroots)
        }
    }

    /// The (A+B) matrix-vector product: same as `a_vec` but with the
    /// SYMMETRIZED transition density in the exchange (J and fxc already
    /// see only the symmetric part; for real orbitals B adds the X^T
    /// exchange image).
    pub fn apb_vec(&self, u: &[f64]) -> Result<Vec<f64>> {
        let cx = self.functional.hybrid.cx;
        let clr = self.functional.hybrid.clr;
        let nao = self.mb.nao_sph;
        let x = self.x_ao(u);
        let mut x_tot = x[0].clone();
        for (t, v) in x_tot.as_mut_slice().iter_mut().zip(x[1].as_slice()) {
            *t += *v;
        }
        let mut x_sym = [x[0].clone(), x[1].clone()];
        for s in 0..2 {
            for r in 0..nao {
                for c in 0..nao {
                    x_sym[s][(r, c)] = x[s][(r, c)] + x[s][(c, r)];
                }
            }
        }
        let (j_tot, k) = self
            .jk
            .build(self.mb, &x_tot, [&x_sym[0], &x_sym[1]], cx != 0.0)?;
        let k_lr = match &self.jk_lr {
            Some(engine) => Some(engine.build_k_only(self.mb, [&x_sym[0], &x_sym[1]])?),
            None => None,
        };
        let mut f = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
        for s in 0..2 {
            // 2 J[X_sym] = J[X] + J[X^T] = 2 J[X] for the projected value;
            // exchange needs K[X + X^T].
            let mut m = Matrix::zeros(nao, nao);
            for (mv, jv) in m.as_mut_slice().iter_mut().zip(j_tot.as_slice()) {
                *mv = 2.0 * jv;
            }
            if cx != 0.0 {
                for (mv, kv) in m.as_mut_slice().iter_mut().zip(k[s].as_slice()) {
                    *mv -= cx * kv;
                }
            }
            if let Some(k_lr) = &k_lr {
                for (mv, kv) in m.as_mut_slice().iter_mut().zip(k_lr[s].as_slice()) {
                    *mv -= clr * kv;
                }
            }
            f[s] = m;
        }
        if let Some(fxc) = &self.fxc {
            // fxc term doubles in A+B; pass the full symmetrized density.
            let vxc = fxc.fxc_contraction(&x_sym)?;
            for s in 0..2 {
                for (mv, xv) in f[s].as_mut_slice().iter_mut().zip(vxc[s].as_slice()) {
                    *mv += *xv;
                }
            }
        }
        self.project_with_diag(u, &f)
    }

    /// The (A-B) matrix-vector product: J and (adiabatic) fxc cancel;
    /// only the antisymmetric exchange image survives.
    pub fn amb_vec(&self, u: &[f64]) -> Result<Vec<f64>> {
        let cx = self.functional.hybrid.cx;
        let clr = self.functional.hybrid.clr;
        let nao = self.mb.nao_sph;
        let x = self.x_ao(u);
        let mut f = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
        if cx != 0.0 || clr != 0.0 {
            let mut x_anti = [x[0].clone(), x[1].clone()];
            for s in 0..2 {
                for r in 0..nao {
                    for c in 0..nao {
                        x_anti[s][(r, c)] = x[s][(r, c)] - x[s][(c, r)];
                    }
                }
            }
            let k = if cx != 0.0 {
                Some(self.jk.build_k_only(self.mb, [&x_anti[0], &x_anti[1]])?)
            } else {
                None
            };
            let k_lr = match &self.jk_lr {
                Some(engine) => Some(engine.build_k_only(self.mb, [&x_anti[0], &x_anti[1]])?),
                None => None,
            };
            for s in 0..2 {
                let mut m = Matrix::zeros(nao, nao);
                if let Some(k) = &k {
                    for (mv, kv) in m.as_mut_slice().iter_mut().zip(k[s].as_slice()) {
                        *mv -= cx * kv;
                    }
                }
                if let Some(k_lr) = &k_lr {
                    for (mv, kv) in m.as_mut_slice().iter_mut().zip(k_lr[s].as_slice()) {
                        *mv -= clr * kv;
                    }
                }
                f[s] = m;
            }
        }
        self.project_with_diag(u, &f)
    }

    /// Shared projection: out_ai = (eps_a - eps_i) u_ai + (C_v^T F C_o)_ai.
    fn project_with_diag(&self, u: &[f64], f: &[Matrix; 2]) -> Result<Vec<f64>> {
        let nao = self.mb.nao_sph;
        let mut out = vec![0.0; self.dim()];
        let mut off = 0;
        let spins: &[usize] = match self.spin {
            TdaSpin::SpinResolved => &[0, 1],
            _ => &[0],
        };
        for &s in spins {
            let (no, nv) = (self.n_occ[s], self.n_vir[s]);
            let c = &self.scf.mo_coeff[s];
            let eps = &self.scf.mo_energy[s];
            for a in 0..nv {
                for i in 0..no {
                    let mut acc = 0.0;
                    for mu in 0..nao {
                        let cva = c[(mu, no + a)];
                        if cva == 0.0 {
                            continue;
                        }
                        for nu in 0..nao {
                            acc += cva * f[s][(mu, nu)] * c[(nu, i)];
                        }
                    }
                    let idx = off + a * no + i;
                    out[idx] = acc + (eps[no + a] - eps[i]) * u[idx];
                }
            }
            off += no * nv;
        }
        Ok(out)
    }

    /// Full TDDFT/TDHF (RPA) excitations via the Casida reduced form:
    /// with M = (A-B)^{1/2} (A+B) (A-B)^{1/2}, M Z = omega^2 Z. Dense
    /// build (the amplitude spaces of interest here are small); (A-B)
    /// must be positive definite (ground-state stability).
    pub fn solve_rpa(&self, nroots: usize) -> Result<Vec<TdaState>> {
        if nroots.min(self.dim()) == 0 {
            return Ok(Vec::new());
        }
        let (apb, amb) = self.orbital_hessians()?;
        self.rpa_from_hessians(&apb, &amb, nroots)
    }

    /// Dense symmetric `A + B` and `A - B` in the amplitude basis, one
    /// product per unit vector.
    pub fn orbital_hessians(&self) -> Result<(Matrix, Matrix)> {
        let dim = self.dim();
        let mut apb = Matrix::zeros(dim, dim);
        let mut amb = Matrix::zeros(dim, dim);
        for col in 0..dim {
            let mut u = vec![0.0; dim];
            u[col] = 1.0;
            let p = self.apb_vec(&u)?;
            let m = self.amb_vec(&u)?;
            for row in 0..dim {
                apb[(row, col)] = p[row];
                amb[(row, col)] = m[row];
            }
        }
        for r in 0..dim {
            for c in (r + 1)..dim {
                let s1 = 0.5 * (apb[(r, c)] + apb[(c, r)]);
                apb[(r, c)] = s1;
                apb[(c, r)] = s1;
                let s2 = 0.5 * (amb[(r, c)] + amb[(c, r)]);
                amb[(r, c)] = s2;
                amb[(c, r)] = s2;
            }
        }
        Ok((apb, amb))
    }

    /// RPA states from the dense `A + B` and `A - B` of
    /// [`Self::orbital_hessians`].
    pub fn rpa_from_hessians(
        &self,
        apb: &Matrix,
        amb: &Matrix,
        nroots: usize,
    ) -> Result<Vec<TdaState>> {
        let dim = self.dim();
        let nroots = nroots.min(dim);
        if nroots == 0 {
            return Ok(Vec::new());
        }
        // (A-B)^{1/2} by eigendecomposition.
        let (mvals, mvecs) = symmetric_eigen(amb)?;
        for (k, &v) in mvals.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(DftError::InvalidInput(format!(
                    "(A-B) eigenvalue {k} = {v:.3e} is non-positive: reference-state \
                     instability; RPA excitations are undefined"
                )));
            }
        }
        let mut half = Matrix::zeros(dim, dim);
        for r in 0..dim {
            for c in 0..dim {
                let mut acc = 0.0;
                for k in 0..dim {
                    acc += mvecs[(r, k)] * mvals[k].sqrt() * mvecs[(c, k)];
                }
                half[(r, c)] = acc;
            }
        }
        let m_mat = half.matmul_seq(&apb.matmul_seq(&half));
        let mut m_sym = m_mat;
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (m_sym[(r, c)] + m_sym[(c, r)]);
                m_sym[(r, c)] = avg;
                m_sym[(c, r)] = avg;
            }
        }
        let (w2, zvecs) = symmetric_eigen(&m_sym)?;
        let mut states = Vec::with_capacity(nroots);
        for r in 0..nroots {
            if !w2[r].is_finite() || w2[r] <= 0.0 {
                return Err(DftError::InvalidInput(format!(
                    "RPA squared frequency {r} = {:.3e} is not positive and finite: \
                     reference-state instability or a zero-metric mode; no normalizable \
                     positive-frequency excitation is defined",
                    w2[r]
                )));
            }
            let omega = w2[r].sqrt();
            // |X+Y> = (A-B)^{1/2} Z / sqrt(omega) (normalized transition
            // vector for properties).
            let mut xpy = vec![0.0; dim];
            for k in 0..dim {
                let mut acc = 0.0;
                for p in 0..dim {
                    acc += half[(k, p)] * zvecs[(p, r)];
                }
                xpy[k] = acc;
            }
            let scale = 1.0 / omega.sqrt();
            for v in xpy.iter_mut() {
                *v *= scale;
            }
            states.push(self.pack_state(omega, &xpy, false));
        }
        Ok(states)
    }

    /// Build the full A matrix column by column and diagonalize (small
    /// systems and Davidson cross-checks).
    pub fn solve_dense(&self, nroots: usize) -> Result<Vec<TdaState>> {
        let dim = self.dim();
        let nroots = nroots.min(dim);
        if nroots == 0 {
            return Ok(Vec::new());
        }
        let mut a = Matrix::zeros(dim, dim);
        for col in 0..dim {
            let mut u = vec![0.0; dim];
            u[col] = 1.0;
            let av = self.a_vec(&u)?;
            for row in 0..dim {
                a[(row, col)] = av[row];
            }
        }
        // Symmetrize (quadrature/tensor noise only).
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (a[(r, c)] + a[(c, r)]);
                a[(r, c)] = avg;
                a[(c, r)] = avg;
            }
        }
        let (vals, vecs) = symmetric_eigen(&a)?;
        let mut states = Vec::with_capacity(nroots);
        for r in 0..nroots {
            let xr: Vec<f64> = (0..dim).map(|k| vecs[(k, r)]).collect();
            states.push(self.pack_state(vals[r], &xr, true));
        }
        Ok(states)
    }

    /// Tamm-Dancoff states in the core-valence-separated space: only the
    /// excitations out of the occupied orbitals `core[spin]` are kept, and
    /// the amplitude of every other occupied orbital is zero. The restricted
    /// matrix is built column by column and diagonalized. A spin-adapted
    /// manifold reads `core[0]` only.
    pub fn solve_core(&self, core: [&[usize]; 2], nroots: usize) -> Result<Vec<TdaState>> {
        Ok(self.solve_core_impl(core, nroots, false)?.0)
    }

    /// CVS-TDA roots and their absolute residuals in the original
    /// projected matrix, before its numerical Hermitian averaging.
    pub fn solve_core_with_residuals(
        &self,
        core: [&[usize]; 2],
        nroots: usize,
    ) -> Result<(Vec<TdaState>, Vec<f64>)> {
        self.solve_core_impl(core, nroots, true)
    }

    fn solve_core_impl(
        &self,
        core: [&[usize]; 2],
        nroots: usize,
        residuals_requested: bool,
    ) -> Result<(Vec<TdaState>, Vec<f64>)> {
        let spins: &[usize] = match self.spin {
            TdaSpin::SpinResolved => &[0, 1],
            _ => &[0],
        };
        // Amplitude indices of the kept occupied-virtual pairs.
        let mut kept = Vec::new();
        let mut off = 0;
        for &s in spins {
            let (no, nv) = (self.n_occ[s], self.n_vir[s]);
            for &i in core[s] {
                if i >= no {
                    return Err(DftError::InvalidInput(format!(
                        "core orbital {i} is not an occupied orbital (there are {no})"
                    )));
                }
                for a in 0..nv {
                    kept.push(off + a * no + i);
                }
            }
            off += no * nv;
        }
        kept.sort_unstable();
        kept.dedup();
        let dim = kept.len();
        let nroots = nroots.min(dim);
        if nroots == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let full = self.dim();
        let mut a = Matrix::zeros(dim, dim);
        for (col, &index) in kept.iter().enumerate() {
            let mut u = vec![0.0; full];
            u[index] = 1.0;
            let av = self.a_vec(&u)?;
            for (row, &target) in kept.iter().enumerate() {
                a[(row, col)] = av[target];
            }
        }
        let raw = residuals_requested.then(|| a.clone());
        // Symmetrize (quadrature/tensor noise only).
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (a[(r, c)] + a[(c, r)]);
                a[(r, c)] = avg;
                a[(c, r)] = avg;
            }
        }
        let (vals, vecs) = symmetric_eigen(&a)?;
        let mut states = Vec::with_capacity(nroots);
        let mut residuals = Vec::new();
        for r in 0..nroots {
            let mut u = vec![0.0; full];
            for (k, &index) in kept.iter().enumerate() {
                u[index] = vecs[(k, r)];
            }
            if let Some(raw) = &raw {
                let residual = (0..dim)
                    .map(|i| {
                        (0..dim).map(|j| raw[(i, j)] * vecs[(j, r)]).sum::<f64>()
                            - vals[r] * vecs[(i, r)]
                    })
                    .fold(0.0_f64, |norm, value| norm.hypot(value));
                residuals.push(residual);
            }
            states.push(self.pack_state(vals[r], &u, true));
        }
        Ok((states, residuals))
    }

    /// Block Davidson for large amplitude spaces.
    pub fn solve_davidson(&self, nroots: usize) -> Result<Vec<TdaState>> {
        let dim = self.dim();
        let nroots = nroots.min(dim);
        if nroots == 0 {
            return Ok(Vec::new());
        }
        let diag = self.diag();
        // Seed: unit vectors on the lowest orbital-energy gaps.
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| diag[a].partial_cmp(&diag[b]).unwrap());
        let nseed = (2 * nroots).min(dim);
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for &idx in order.iter().take(nseed) {
            let mut v = vec![0.0; dim];
            v[idx] = 1.0;
            basis.push(v);
        }
        let mut sigma: Vec<Vec<f64>> = Vec::new();
        let max_space = (8 * nroots + 12).min(dim);
        for _ in 0..self.max_iter {
            while sigma.len() < basis.len() {
                sigma.push(self.a_vec(&basis[sigma.len()])?);
            }
            let m = basis.len();
            let mut h = Matrix::zeros(m, m);
            for p in 0..m {
                for q in 0..m {
                    let mut acc = 0.0;
                    for k in 0..dim {
                        acc += basis[p][k] * sigma[q][k];
                    }
                    h[(p, q)] = acc;
                }
            }
            // Symmetrize subspace matrix (A is symmetric).
            for p in 0..m {
                for q in (p + 1)..m {
                    let avg = 0.5 * (h[(p, q)] + h[(q, p)]);
                    h[(p, q)] = avg;
                    h[(q, p)] = avg;
                }
            }
            let (vals, vecs) = symmetric_eigen(&h)?;
            // Ritz vectors + residuals for the lowest nroots.
            let mut converged = true;
            let mut new_dirs: Vec<Vec<f64>> = Vec::new();
            for r in 0..nroots {
                let theta = vals[r];
                let mut xr = vec![0.0; dim];
                let mut res = vec![0.0; dim];
                for p in 0..m {
                    let c = vecs[(p, r)];
                    if c == 0.0 {
                        continue;
                    }
                    for k in 0..dim {
                        xr[k] += c * basis[p][k];
                        res[k] += c * sigma[p][k];
                    }
                }
                for k in 0..dim {
                    res[k] -= theta * xr[k];
                }
                let rnorm = res.iter().map(|v| v * v).sum::<f64>().sqrt();
                if rnorm > self.tol {
                    converged = false;
                    // Davidson preconditioner.
                    for k in 0..dim {
                        let denom = diag[k] - theta;
                        res[k] /= if denom.abs() > 1e-4 {
                            denom
                        } else {
                            1e-4_f64.copysign(denom)
                        };
                    }
                    new_dirs.push(res);
                }
            }
            if converged {
                // Extract states.
                let mut states = Vec::with_capacity(nroots);
                for r in 0..nroots {
                    let mut xr = vec![0.0; dim];
                    for p in 0..m {
                        let c = vecs[(p, r)];
                        for k in 0..dim {
                            xr[k] += c * basis[p][k];
                        }
                    }
                    states.push(self.pack_state(vals[r], &xr, true));
                }
                return Ok(states);
            }
            // Orthonormalize new directions against the basis and append
            // (two Gram-Schmidt passes for numerical robustness).
            let mut added = 0;
            for mut dir in new_dirs {
                let norm0: f64 = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm0 == 0.0 {
                    continue;
                }
                for v in dir.iter_mut() {
                    *v /= norm0;
                }
                for _pass in 0..2 {
                    for b in &basis {
                        let ip: f64 = b.iter().zip(&dir).map(|(x, y)| x * y).sum();
                        if ip != 0.0 {
                            for k in 0..dim {
                                dir[k] -= ip * b[k];
                            }
                        }
                    }
                }
                let n2: f64 = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                if n2 > 1e-8 {
                    for v in dir.iter_mut() {
                        *v /= n2;
                    }
                    basis.push(dir);
                    added += 1;
                }
            }
            if added == 0 && basis.len() < dim {
                // Stagnation: expand with the lowest-gap unit vector not
                // yet represented.
                for &idx in &order {
                    let mut e = vec![0.0; dim];
                    e[idx] = 1.0;
                    for _pass in 0..2 {
                        for b in &basis {
                            let ip: f64 = b.iter().zip(&e).map(|(x, y)| x * y).sum();
                            if ip != 0.0 {
                                for k in 0..dim {
                                    e[k] -= ip * b[k];
                                }
                            }
                        }
                    }
                    let n2: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if n2 > 1e-6 {
                        for v in e.iter_mut() {
                            *v /= n2;
                        }
                        basis.push(e);
                        break;
                    }
                }
            }
            // Subspace collapse.
            if basis.len() > max_space {
                let mut collapsed: Vec<Vec<f64>> = Vec::new();
                for r in 0..(2 * nroots).min(m) {
                    let mut xr = vec![0.0; dim];
                    for p in 0..m {
                        let c = vecs[(p, r)];
                        for k in 0..dim {
                            xr[k] += c * basis[p][k];
                        }
                    }
                    // Orthonormalize against collapsed set.
                    for b in &collapsed {
                        let ip: f64 = b.iter().zip(&xr).map(|(x, y)| x * y).sum();
                        for k in 0..dim {
                            xr[k] -= ip * b[k];
                        }
                    }
                    let n2: f64 = xr.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if n2 > 1e-8 {
                        for v in xr.iter_mut() {
                            *v /= n2;
                        }
                        collapsed.push(xr);
                    }
                }
                basis = collapsed;
                sigma.clear();
            }
        }
        Err(DftError::ResponseNotConverged {
            iterations: self.max_iter,
            residual: f64::NAN,
        })
    }

    fn pack_state(&self, energy: f64, u: &[f64], tamm_dancoff: bool) -> TdaState {
        let mut x = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        let mut off = 0;
        let spins: &[usize] = match self.spin {
            TdaSpin::SpinResolved => &[0, 1],
            _ => &[0],
        };
        for &s in spins {
            let (no, nv) = (self.n_occ[s], self.n_vir[s]);
            let mut m = Matrix::zeros(nv, no);
            for a in 0..nv {
                for i in 0..no {
                    m[(a, i)] = u[off + a * no + i];
                }
            }
            x[s] = m;
            off += no * nv;
        }
        if !self.scf.unrestricted {
            x[1] = x[0].clone();
        }
        // Transition dipole d_u = sum_spin-orbitals X <a|r_u|i>. For a
        // restricted reference the alpha-block amplitudes (norm 1) carry
        // the spin factor sqrt(2) (singlet) or exact cancellation
        // (triplet); spin-resolved amplitudes contribute directly.
        let nao = self.mb.nao_sph;
        let dips = crate::integrals::dipole_matrices(self.mb, crate::math::Vec3::zero());
        let mut d = [0.0; 3];
        if self.spin != TdaSpin::Triplet {
            let contrib: &[(usize, f64)] = match self.spin {
                TdaSpin::Singlet => &[(0, std::f64::consts::SQRT_2)],
                _ => &[(0, 1.0), (1, 1.0)],
            };
            for &(s, factor) in contrib {
                let (no, nv) = (self.n_occ[s], self.n_vir[s]);
                let c = &self.scf.mo_coeff[s];
                for (u_ax, dm) in dips.iter().enumerate() {
                    let mut acc = 0.0;
                    for a in 0..nv {
                        for i in 0..no {
                            let xai = x[s][(a, i)];
                            if xai == 0.0 {
                                continue;
                            }
                            let mut dai = 0.0;
                            for mu in 0..nao {
                                let cva = c[(mu, no + a)];
                                if cva == 0.0 {
                                    continue;
                                }
                                for nu in 0..nao {
                                    dai += cva * dm[(mu, nu)] * c[(nu, i)];
                                }
                            }
                            acc += xai * dai;
                        }
                    }
                    d[u_ax] += factor * acc;
                }
            }
        }
        // Magnetic transition moment and its velocity-gauge partner, from
        // the SAME amplitude contraction as the electric one above. Both
        // operators are ANTI-Hermitian (the physical ones carry a factor
        // i), which is why a rotatory strength is real: the `i`s of the
        // magnetic moment and of these matrix elements cancel.
        let angmom =
            crate::properties::nmr::angular_momentum_operator(self.mb, crate::math::Vec3::zero());
        let grad = crate::properties::nmr::gradient_operator(self.mb);
        let contract = |ops: &[Matrix; 3]| -> [f64; 3] {
            let mut out = [0.0; 3];
            if self.spin == TdaSpin::Triplet {
                return out;
            }
            let contrib: &[(usize, f64)] = match self.spin {
                TdaSpin::Singlet => &[(0, std::f64::consts::SQRT_2)],
                _ => &[(0, 1.0), (1, 1.0)],
            };
            for &(s, factor) in contrib {
                let (no, nv) = (self.n_occ[s], self.n_vir[s]);
                let c = &self.scf.mo_coeff[s];
                for (u_ax, m) in ops.iter().enumerate() {
                    let mut acc = 0.0;
                    for a in 0..nv {
                        for i in 0..no {
                            let xai = x[s][(a, i)];
                            if xai == 0.0 {
                                continue;
                            }
                            let mut mai = 0.0;
                            for mu in 0..nao {
                                let cva = c[(mu, no + a)];
                                if cva == 0.0 {
                                    continue;
                                }
                                for nu in 0..nao {
                                    mai += cva * m[(mu, nu)] * c[(nu, i)];
                                }
                            }
                            acc += xai * mai;
                        }
                    }
                    out[u_ax] += factor * acc;
                }
            }
            out
        };
        let l = contract(&angmom);
        let p = contract(&grad);
        // m = -(1/2) L in atomic units, so R_length = -(1/2) d . L.
        let r_len = -0.5 * (d[0] * l[0] + d[1] * l[1] + d[2] * l[2]);
        // Velocity gauge replaces <0|mu|n> by <0|grad|n>/omega, with the
        // sign that makes the two agree in the complete-basis limit.
        let r_vel = if energy.abs() > 1e-12 {
            0.5 * (p[0] * l[0] + p[1] * l[1] + p[2] * l[2]) / energy
        } else {
            0.0
        };
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut state = TdaState {
            energy,
            x,
            spin: self.spin,
            tamm_dancoff,
            transition_dipole: d,
            oscillator_strength: 2.0 / 3.0 * energy * d2,
            transition_angular_momentum: l,
            transition_momentum: p,
            rotatory_strength: r_len,
            rotatory_strength_velocity: r_vel,
        };
        state.canonicalize_phase(None);
        state
    }
}
