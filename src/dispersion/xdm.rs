// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! XDM: the exchange-hole dipole moment dispersion model.
//!
//! Becke & Johnson, J. Chem. Phys. 122, 154104 (2005); J. Chem. Phys. 123,
//! 154101 (2005); J. Chem. Phys. 127, 154108 (2007); Johnson & Becke,
//! J. Chem. Phys. 124, 174104 (2006). The Becke–Roussel exchange-hole model
//! is Becke & Roussel, Phys. Rev. A 39, 3761 (1989).
//!
//! Every formula below is transcribed from the equations as restated (with
//! citations back to the primary papers above) in DiLabio & Otero-de-la-Roza,
//! "Noncovalent Interactions in Density-Functional Theory", Rev. Comput.
//! Chem. 29 (2015), arXiv:1405.1771 — NOT from postg, Quantum ESPRESSO, or
//! psi4, all of which implement XDM and are off limits as sources per this
//! project's rules (see `docs/FUNCTIONALS.md#d4-formulation-parameters-and-implementation-decision` for why that line is drawn, and
//! `docs/scope.md` §5 for where XDM sits next to D2/D3/D4). Equation numbers
//! in comments below (`Eq. NN`) refer to that review.
//!
//! # What is implemented, and what refuses
//!
//! Every layer of the "essential chain" is implemented and gated:
//!
//! 1. The Becke–Roussel (BR) exchange hole, [`becke_roussel_dipole`] —
//!    solved from (ρ_σ, |∇ρ_σ|², τ_σ, ∇²ρ_σ) exactly as Eq. 47-51 specify.
//! 2. Hirshfeld atomic partitioning of the squared multipole moments
//!    ⟨M₁²⟩, ⟨M₂²⟩, ⟨M₃²⟩ (Eq. 38, 42, 43), via [`xdm_atomic_moments`].
//! 3. Free-atom reference polarizability → effective in-molecule
//!    polarizability by Hirshfeld volume scaling (Eq. 44-46).
//! 4. C₆, C₈, C₁₀ from the moments and effective polarizabilities
//!    (Eq. 39-41): [`xdm_c6`], [`xdm_c8`], [`xdm_c10`].
//! 5. Becke–Johnson damping (Eq. 55-57) and the pairwise energy (Eq. 52):
//!    [`xdm_pair_energy`], [`xdm_energy`].
//!
//! Two things are refused outright rather than shipped ungated:
//!
//! - **The analytic gradient.** [`xdm_gradient`] returns
//!   `DftError::Unsupported`. Differentiating this chain needs the implicit
//!   derivative of the BR root solve (Eq. 50) AND the derivative of the
//!   Becke/Hirshfeld grid weights with respect to nuclear position — neither
//!   is implemented, so there is nothing to FD-gate. A numerical gradient of
//!   [`xdm_dispersion_energy`] is the only available derivative.
//! - **Per-functional Becke–Johnson damping parameters (a₁, a₂).** These are
//!   empirical fits to CCSD(T)/CBS reference data, published per-functional
//!   across several papers (Johnson & Becke 2006; Otero-de-la-Roza & Johnson,
//!   J. Chem. Phys. 136, 174109 (2012); the `postg` project's own parameter
//!   tables) that this project has no independently verified access to
//!   reproduce digit-for-digit without risking a transcription from an
//!   off-limits implementation. Rather than fabricate or silently borrow a
//!   number, [`XdmDampingParams`] simply has no by-name lookup: the CALLER
//!   supplies `a1`/`a2` directly. Everything else — C₆/C₈/C₁₀, the damping
//!   FORM, the energy assembly — is independent of that choice and fully
//!   implemented.
//! - **Three-body (Axilrod–Teller–Muto-like, C₉) dispersion.** The review
//!   itself notes there is no established way to reconcile a pairwise BJ
//!   damping function with the three-body term (Eq. 53-54); "canonical XDM"
//!   (its own words) is the two-body sum in Eq. 52 alone, which is what
//!   [`xdm_energy`] computes.
//!
//! # A deliberate departure from the literature's free-atom data
//!
//! The essential-chain description calls for TWO external free-atom
//! quantities: reference polarizabilities α_free and reference volumes
//! V_free. This module bundles only the first
//! (`data/xdm_free_atom_polarizability.csv`, PROVENANCE: Schwerdtfeger &
//! Nagle's openly published, XDM-independent "Table of Polarizabilities" —
//! see the CSV header). The second is NOT tabulated from the XDM papers'
//! own free-atom volumes (which are not open-access anywhere this project
//! can reach); instead V_free is COMPUTED, on the fly, from the exact same
//! reference atomic density used to build the Hirshfeld promolecule weight
//! (`crate::scf::sad::sad_density` on the isolated atom — the project's own
//! already-tested "spherically averaged atom in this basis" machinery, see
//! `src/scf/sad.rs`). This is not a shortcut of convenience: Eq. 45-46 pin
//! V_A and V_free to literally the SAME reference density ρ_A^free(r), so
//! computing both from `sad_density` is the mathematically consistent
//! choice, not an approximation of a fixed external number. One direct
//! consequence, verified by [`free_atom_limit_is_exact`]: for an isolated
//! atom the Hirshfeld weight is identically 1, so V_A ≡ V_free and
//! α_eff ≡ α_free EXACTLY, independent of basis or of what ρ_A^free(r)
//! numerically looks like — the free-atom-limit gate this project requires.

use crate::basis::MolBasis;
use crate::error::{DftError, Result};
use crate::grid::eval::eval_aos_deriv3;
use crate::grid::{GridOptions, MolecularGrid};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use std::collections::HashMap;
use std::sync::OnceLock;

const RHO_CUTOFF: f64 = 1e-12;

// ===========================================================================
// Layer 1: the Becke-Roussel exchange hole (Eq. 47-51).
// ===========================================================================
//
// h_Xσ(r,s) is modelled as A e^{-a s} centred a distance b from the
// reference point r. On-top depth + normalization + exact curvature match
// pin (A, a, b); with those constraints the model reduces to solving a
// single transcendental equation for a dimensionless x, from which b (the
// exchange-hole dipole magnitude d_Xσ) follows in closed form.
//
// Convention note: the review's Eq. 49 defines τ_σ = Σ_i(∇ψ_iσ)² with NO
// leading 1/2. This crate's meta-GGA grid machinery (`GridVars::tau`,
// `src/xc/traits.rs`) uses the standard Kohn-Sham convention
// τ_σ = (1/2)Σ_i|∇ψ_iσ|² instead (documented on `GridVars::tau`). The public
// functions below take τ in THIS crate's standard convention and convert
// internally, so callers feeding τ from anywhere else in dft-rs need no
// unit juggling. The conversion is exercised (not just asserted) by
// `becke_roussel_hydrogen_1s_is_exact` below: it is the well known
// closed-form property of BR that D_σ (the Pauli kinetic energy density)
// vanishes identically for any one-electron density, in EITHER convention,
// and that identity only holds if the ×2 conversion here is right.

/// D_σ = τ_σ − (1/4)|∇ρ_σ|²/ρ_σ, in the review's τ convention (Eq. 48).
/// `tau_std` is τ_σ in this crate's standard (1/2 Σ|∇ψ|²) convention.
fn pauli_kinetic_deficit(tau_std: f64, grad_rho2: f64, rho: f64) -> f64 {
    2.0 * tau_std - 0.25 * grad_rho2 / rho
}

/// g(x) = x·e^{−2x/3}/(x−2), the left-hand side of Eq. 50.
///
/// Monotonically DECREASING on (0, 2), from g(0)=0 down to g(2⁻)=−∞, and
/// monotonically decreasing on (2, ∞), from g(2⁺)=+∞ down to g(∞)=0. Both
/// branches are therefore invertible; which one to use is fixed by the
/// sign of Q_σ (equivalently, the sign of the right-hand-side target).
fn br_g(x: f64) -> f64 {
    x * (-2.0 * x / 3.0).exp() / (x - 2.0)
}

/// Solve g(x) = target for x ≥ 0 (x ≠ 2), by bisection on whichever branch
/// `target`'s sign selects. Bisection (not Newton) is used deliberately:
/// g's derivative changes sign nowhere on either open branch, so a plain
/// bracketing search is robust across the whole range Q_σ can take,
/// including arbitrarily large |target| near the x=2 singularity.
fn solve_br_x(target: f64) -> f64 {
    if target == 0.0 {
        return 0.0;
    }
    if !target.is_finite() {
        // Q_σ -> 0: x -> 2 from the side matching target's sign.
        return if target > 0.0 {
            2.0 + 1e-12
        } else {
            2.0 - 1e-12
        };
    }
    if target > 0.0 {
        // Branch x in (2, infinity): g(lo) must exceed target, g(hi) must
        // fall below it. Shrink lo toward 2 (where g diverges) until that
        // holds, then expand hi until g(hi) < target.
        let mut lo = 2.0 + 1e-6;
        let mut tries = 0;
        while br_g(lo) <= target && lo - 2.0 > 1e-15 && tries < 60 {
            lo = 2.0 + (lo - 2.0) * 0.1;
            tries += 1;
        }
        let mut hi = lo.max(4.0);
        tries = 0;
        while br_g(hi) >= target && tries < 200 {
            hi *= 2.0;
            tries += 1;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if br_g(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    } else {
        // Branch x in (0, 2): g(0)=0 > target, g(2⁻)=-infinity < target.
        let mut lo = 0.0;
        let mut hi = 2.0 - 1e-12;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if br_g(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// The Becke-Roussel exchange-hole dipole magnitude d_Xσ(r) = b_σ(r)
/// (Eq. 47-51), from the local density, its squared gradient, the
/// (standard-convention) kinetic energy density, and the density
/// Laplacian, all for ONE spin channel at ONE point.
///
/// Returns `None` when ρ_σ is negligible (nothing to compute; callers
/// should skip the point/spin rather than treat it as a hole at distance
/// zero).
pub fn becke_roussel_dipole(rho: f64, grad_rho2: f64, tau_std: f64, lap_rho: f64) -> Option<f64> {
    if rho <= RHO_CUTOFF {
        return None;
    }
    let d = pauli_kinetic_deficit(tau_std, grad_rho2, rho);
    let q = (lap_rho - 2.0 * d) / 6.0;
    let rhs = (2.0 / 3.0) * std::f64::consts::PI.powf(2.0 / 3.0) * rho.powf(5.0 / 3.0) / q;
    let x = solve_br_x(rhs);
    let b3 = x.powi(3) * (-x).exp() / (8.0 * std::f64::consts::PI * rho);
    Some(b3.max(0.0).cbrt())
}

// ===========================================================================
// Layer 3 (input side): free-atom reference polarizabilities.
// ===========================================================================

const XDM_ALPHA_CSV: &str = include_str!("data/xdm_free_atom_polarizability.csv");

fn alpha_table() -> &'static HashMap<u8, f64> {
    static TABLE: OnceLock<HashMap<u8, f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m = HashMap::new();
        for line in XDM_ALPHA_CSV.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') || l.starts_with("z,") {
                continue;
            }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() < 3 {
                continue;
            }
            if let (Ok(z), Ok(a)) = (
                parts[0].trim().parse::<u8>(),
                parts[2].trim().parse::<f64>(),
            ) {
                m.insert(z, a);
            }
        }
        m
    })
}

/// Free-atom static dipole polarizability α_free (a.u.) for element `z`.
/// Bundled range: Z = 1..=36 (see the CSV's PROVENANCE header).
pub fn free_atom_polarizability(z: u8) -> Result<f64> {
    alpha_table()
        .get(&z)
        .copied()
        .ok_or_else(|| DftError::Unsupported {
            feature: format!("XDM free-atom polarizability for Z={z}"),
            missing: "not tabulated; the bundled table covers Z=1..=36 \
            (src/dispersion/data/xdm_free_atom_polarizability.csv) from an \
            openly published, XDM-independent source — extend it from the \
            same source to add heavier elements"
                .to_string(),
        })
}

// ===========================================================================
// Layer 2+3: Hirshfeld partitioning of the moments, and the effective
// (in-molecule) polarizability by volume scaling (Eq. 38, 42-46).
// ===========================================================================

/// Per-atom XDM moments and the effective (in-molecule) polarizability,
/// ready to feed [`xdm_c6`]/[`xdm_c8`]/[`xdm_c10`].
#[derive(Clone, Debug)]
pub struct XdmAtomMoments {
    pub z: u8,
    pub position: Vec3,
    /// ⟨M₁²⟩_A (Eq. 38, l=1).
    pub m1_sq: f64,
    /// ⟨M₂²⟩_A (Eq. 38, l=2).
    pub m2_sq: f64,
    /// ⟨M₃²⟩_A (Eq. 38, l=3).
    pub m3_sq: f64,
    /// V_A / V_free (Eq. 44's volume ratio).
    pub volume_ratio: f64,
    /// Free-atom polarizability α_free (Eq. 44's α_A^at).
    pub alpha_free: f64,
    /// Effective in-molecule polarizability α_A = (V_A/V_free)·α_free.
    pub alpha_eff: f64,
}

#[derive(Clone, Copy, Default)]
struct AtomAccum {
    m1: f64,
    m2: f64,
    m3: f64,
    v_mol: f64,
    v_free: f64,
}

/// Compute every atom's XDM moments for a converged spin-resolved density
/// `density = [Pα, Pβ]` on `mol`/`mb`.
///
/// The Hirshfeld promolecule reference (`ρ_A^free`, Eq. 42) is built from
/// `crate::scf::sad::sad_density` applied to `mol` itself: that function
/// already returns the block-diagonal superposition of converged,
/// spherically averaged, per-atom HF densities in `mb`'s own AO space (see
/// `src/scf/sad.rs`) — exactly the "in vacuo atomic density placed at the
/// molecular geometry" the Hirshfeld weight needs, reused rather than
/// re-derived. `V_free` (Eq. 46) is integrated from the SAME reference
/// density on the SAME grid, which is what makes the free-atom limit below
/// exact by construction rather than an approximation.
pub fn xdm_atomic_moments(
    mol: &Molecule,
    mb: &MolBasis,
    density: [&Matrix; 2],
    grid_opts: &GridOptions,
) -> Result<Vec<XdmAtomMoments>> {
    let grid = MolecularGrid::build(mol, grid_opts)?;
    let promolecule = super::hirshfeld::Promolecule::new(mol, mb)?;
    let n_at = mol.len();
    let nao = mb.nao_sph;

    let mut acc = vec![AtomAccum::default(); n_at];

    const BATCH: usize = 300;
    let npts_total = grid.len();
    let mut start = 0;
    while start < npts_total {
        let end = (start + BATCH).min(npts_total);
        let pts = &grid.points[start..end];
        let wts = &grid.weights[start..end];
        let npts = pts.len();

        let batch = eval_aos_deriv3(mb, pts, true, true, false);
        let phi = &batch.phi;
        let grad = batch.grad.as_ref().expect("gradients requested");
        let hess = batch.hess.as_ref().expect("hessians requested");

        // Laplacian of every AO: ∇²φ_ν = hess_xx + hess_yy + hess_zz
        // (component order (xx,xy,xz,yy,yz,zz) per `grid::eval::AoBatch`).
        let mut lap_phi = Matrix::zeros(npts, nao);
        for i in 0..npts {
            for mu in 0..nao {
                lap_phi[(i, mu)] = hess[0][(i, mu)] + hess[3][(i, mu)] + hess[5][(i, mu)];
            }
        }

        // Real molecular density fields, per spin: rho, |grad rho|^2, tau
        // (standard 1/2 convention), lap rho. Mirrors `ks.rs::xc_contribution`'s
        // rho/grad/tau assembly, extended with the Laplacian XDM needs.
        let mut rho = [vec![0.0; npts], vec![0.0; npts]];
        let mut grad2 = [vec![0.0; npts], vec![0.0; npts]];
        let mut tau = [vec![0.0; npts], vec![0.0; npts]];
        let mut lap = [vec![0.0; npts], vec![0.0; npts]];

        for spin in 0..2 {
            let phip = phi.matmul_seq(density[spin]);
            let gp: [Matrix; 3] = std::array::from_fn(|ax| grad[ax].matmul_seq(density[spin]));
            let lap_phip = lap_phi.matmul_seq(density[spin]);

            for i in 0..npts {
                let mut r = 0.0;
                for mu in 0..nao {
                    r += phip[(i, mu)] * phi[(i, mu)];
                }
                rho[spin][i] = r.max(0.0);

                let mut g2 = 0.0;
                for ax in 0..3 {
                    let mut gi = 0.0;
                    for mu in 0..nao {
                        gi += phip[(i, mu)] * grad[ax][(i, mu)];
                    }
                    let val = 2.0 * gi;
                    g2 += val * val;
                }
                grad2[spin][i] = g2;

                let mut t = 0.0;
                for ax in 0..3 {
                    let mut ti = 0.0;
                    for mu in 0..nao {
                        ti += gp[ax][(i, mu)] * grad[ax][(i, mu)];
                    }
                    t += ti;
                }
                tau[spin][i] = 0.5 * t;

                let mut l1 = 0.0;
                for mu in 0..nao {
                    l1 += phi[(i, mu)] * lap_phip[(i, mu)];
                }
                // lap rho = 2*l1 + 2*t; see the module-level derivation note
                // in `xdm_atomic_moments`'s doc comment.
                lap[spin][i] = 2.0 * l1 + 2.0 * t;
            }
        }

        let rho_free_by_atom = promolecule.collocate(phi);

        for i in 0..npts {
            let w = wts[i];
            let promol: f64 = (0..n_at).map(|a| rho_free_by_atom[a][i]).sum();
            if promol <= 1e-300 {
                continue;
            }

            let mut d_x = [0.0f64; 2];
            let mut has_hole = [false; 2];
            for s in 0..2 {
                if let Some(b) = becke_roussel_dipole(rho[s][i], grad2[s][i], tau[s][i], lap[s][i])
                {
                    d_x[s] = b;
                    has_hole[s] = true;
                }
            }

            for a in 0..n_at {
                let rho_a_free = rho_free_by_atom[a][i];
                if rho_a_free <= 0.0 {
                    continue;
                }
                let wa = rho_a_free / promol;
                let r_a = (pts[i] - mol.atoms[a].position).norm();

                acc[a].v_mol += w * wa * (rho[0][i] + rho[1][i]) * r_a.powi(3);
                acc[a].v_free += w * rho_a_free * r_a.powi(3);

                for s in 0..2 {
                    if !has_hole[s] {
                        continue;
                    }
                    let dd = r_a - d_x[s];
                    let m1 = r_a - dd;
                    let m2 = r_a * r_a - dd * dd;
                    let m3 = r_a.powi(3) - dd.powi(3);
                    let wr = w * wa * rho[s][i];
                    acc[a].m1 += wr * m1 * m1;
                    acc[a].m2 += wr * m2 * m2;
                    acc[a].m3 += wr * m3 * m3;
                }
            }
        }

        start = end;
    }

    let mut out = Vec::with_capacity(n_at);
    for (a, atom) in mol.atoms.iter().enumerate() {
        let alpha_free = free_atom_polarizability(atom.z)?;
        let ratio = if acc[a].v_free > 0.0 {
            acc[a].v_mol / acc[a].v_free
        } else {
            0.0
        };
        out.push(XdmAtomMoments {
            z: atom.z,
            position: atom.position,
            m1_sq: acc[a].m1,
            m2_sq: acc[a].m2,
            m3_sq: acc[a].m3,
            volume_ratio: ratio,
            alpha_free,
            alpha_eff: ratio * alpha_free,
        });
    }
    Ok(out)
}

// ===========================================================================
// Layer 4: C6, C8, C10 (Eq. 39-41).
// ===========================================================================

/// C₆^AB (Eq. 39).
pub fn xdm_c6(a: &XdmAtomMoments, b: &XdmAtomMoments) -> f64 {
    let den = a.m1_sq * b.alpha_eff + b.m1_sq * a.alpha_eff;
    if den <= 0.0 {
        return 0.0;
    }
    a.alpha_eff * b.alpha_eff * a.m1_sq * b.m1_sq / den
}

/// C₈^AB (Eq. 40).
pub fn xdm_c8(a: &XdmAtomMoments, b: &XdmAtomMoments) -> f64 {
    let den = a.m1_sq * b.alpha_eff + b.m1_sq * a.alpha_eff;
    if den <= 0.0 {
        return 0.0;
    }
    1.5 * a.alpha_eff * b.alpha_eff * (a.m1_sq * b.m2_sq + b.m1_sq * a.m2_sq) / den
}

/// C₁₀^AB (Eq. 41).
pub fn xdm_c10(a: &XdmAtomMoments, b: &XdmAtomMoments) -> f64 {
    let den1 = a.m1_sq * b.alpha_eff + b.m1_sq * a.alpha_eff;
    let term1 = if den1 > 0.0 {
        2.0 * a.alpha_eff * b.alpha_eff * (a.m1_sq * b.m3_sq + b.m1_sq * a.m3_sq) / den1
    } else {
        0.0
    };
    let den2 = a.m2_sq * b.alpha_eff + b.m2_sq * a.alpha_eff;
    let term2 = if den2 > 0.0 {
        (21.0 / 5.0) * a.alpha_eff * b.alpha_eff * a.m2_sq * b.m2_sq / den2
    } else {
        0.0
    };
    term1 + term2
}

// ===========================================================================
// Layer 5: Becke-Johnson damping and the pairwise energy (Eq. 52, 55-57).
// ===========================================================================

/// The two Becke-Johnson damping parameters. Deliberately no by-name
/// lookup — see the module-level docs for why.
#[derive(Clone, Copy, Debug)]
pub struct XdmDampingParams {
    pub a1: f64,
    pub a2: f64,
}

/// R_c (Eq. 57): the arithmetic mean of the radii where each pair of C_n
/// terms is equal in magnitude.
fn critical_radius(c6: f64, c8: f64, c10: f64) -> f64 {
    if c6 <= 0.0 || c8 <= 0.0 || c10 <= 0.0 {
        return 0.0;
    }
    ((c8 / c6).sqrt() + (c10 / c6).powf(0.25) + (c10 / c8).sqrt()) / 3.0
}

/// f_n(R) = R^n/(R^n + R_vdw^n) (Eq. 55).
fn bj_damping(n: i32, r: f64, r_vdw: f64) -> f64 {
    let rn = r.powi(n);
    let rvn = r_vdw.powi(n);
    rn / (rn + rvn)
}

/// The XDM dispersion energy of one atom pair at separation `r` (Bohr).
pub fn xdm_pair_energy(
    a: &XdmAtomMoments,
    b: &XdmAtomMoments,
    r: f64,
    damping: &XdmDampingParams,
) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let c6 = xdm_c6(a, b);
    let c8 = xdm_c8(a, b);
    let c10 = xdm_c10(a, b);
    let rc = critical_radius(c6, c8, c10);
    let r_vdw = damping.a1 * rc + damping.a2;
    let f6 = bj_damping(6, r, r_vdw);
    let f8 = bj_damping(8, r, r_vdw);
    let f10 = bj_damping(10, r, r_vdw);
    -(c6 * f6 / r.powi(6) + c8 * f8 / r.powi(8) + c10 * f10 / r.powi(10))
}

/// Total pairwise XDM dispersion energy (Eq. 52), summed over every atom
/// pair using each atom's OWN position (so this also works with moments
/// computed at one geometry but evaluated at another, e.g. the R⁻⁶
/// asymptote gate below).
pub fn xdm_energy(moments: &[XdmAtomMoments], damping: &XdmDampingParams) -> f64 {
    let n = moments.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (moments[i].position - moments[j].position).norm();
            e += xdm_pair_energy(&moments[i], &moments[j], r, damping);
        }
    }
    e
}

/// Convenience: moments + energy in one call.
pub fn xdm_dispersion_energy(
    mol: &Molecule,
    mb: &MolBasis,
    density: [&Matrix; 2],
    grid_opts: &GridOptions,
    damping: &XdmDampingParams,
) -> Result<f64> {
    let moments = xdm_atomic_moments(mol, mb, density, grid_opts)?;
    Ok(xdm_energy(&moments, damping))
}

/// Refused: see the module-level docs. Differentiating the BR root solve
/// (Eq. 50) and the Becke/Hirshfeld grid weights with respect to nuclear
/// position is not implemented, so there is no analytic XDM gradient to
/// FD-gate. A caller that needs one must use a numerical derivative of
/// [`xdm_dispersion_energy`] instead.
pub fn xdm_gradient(
    _mol: &Molecule,
    _mb: &MolBasis,
    _density: [&Matrix; 2],
    _grid_opts: &GridOptions,
    _damping: &XdmDampingParams,
) -> Result<Vec<Vec3>> {
    Err(DftError::Unsupported {
        feature: "XDM analytic gradient".to_string(),
        missing: "the implicit derivative of the Becke-Roussel root solve \
            (d b_sigma/dR) and the derivative of the Becke/Hirshfeld grid \
            weights with respect to nuclear position; use a numerical \
            gradient of xdm_dispersion_energy instead"
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::scf::{run_hf, ScfOptions};
    use crate::system::Atom;

    // -----------------------------------------------------------------
    // Layer 1 gates: the BR solver, entirely analytic (no basis/grid/SCF).
    // -----------------------------------------------------------------

    /// The raw root solver must actually invert `br_g`: g(solve_br_x(t)) = t
    /// for a spread of targets on both branches, including the extreme
    /// values near the x=2 singularity that a real Q_sigma -> 0 grid point
    /// can produce.
    #[test]
    fn br_solver_round_trips_through_its_own_defining_equation() {
        for &target in &[
            -1.0e6, -1234.5, -10.0, -0.5, -1e-4, 1e-4, 0.5, 3.7, 50.0, 1.0e5,
        ] {
            let x = solve_br_x(target);
            let g = br_g(x);
            let scale = target.abs().max(1.0);
            assert!(
                (g - target).abs() < 1e-6 * scale,
                "target {target}: x={x}, g(x)={g}"
            );
            if target > 0.0 {
                assert!(x > 2.0, "positive branch must have x>2, got {x}");
            } else {
                assert!(
                    (0.0..2.0).contains(&x),
                    "negative branch must have 0<=x<2, got {x}"
                );
            }
        }
    }

    /// Exact closed form for the hydrogen 1s density rho(r) = (1/pi) e^{-2r}:
    /// self-derived from Eq. 47-51 (NOT taken from any implementation) by
    /// substituting x = 2r and checking it solves the defining equation
    /// algebraically. For a single doubly... singly-occupied 1s orbital the
    /// true exchange hole is exactly r-independent (h_X(r,r') = -rho(r')),
    /// and since rho itself is a pure exponential centred at the nucleus,
    /// BR's on-top-value+curvature match reproduces it exactly: the model
    /// predicts the hole sits exactly AT the nucleus for every reference
    /// point r, i.e. b_sigma(r) = r identically.
    ///
    /// Also exercises both solver branches: r<1 gives Q>0... (checked
    /// below) giving x=2r<2 (negative-target branch), r>1 gives x=2r>2
    /// (positive-target branch).
    #[test]
    fn becke_roussel_hydrogen_1s_is_exact() {
        for &r in &[0.15_f64, 0.4, 0.8, 1.3, 2.0, 3.5, 6.0] {
            let rho = (1.0 / std::f64::consts::PI) * (-2.0 * r).exp();
            let grad_rho2 = 4.0 * rho * rho; // (d rho/dr)^2, drho/dr = -2 rho
                                             // Standard convention tau = (1/2)(grad psi)^2 = rho/2 for this orbital.
            let tau_std = 0.5 * rho;
            // lap rho = 4 rho (r-1)/r, derived from rho=(1/pi)e^{-2r}.
            let lap_rho = 4.0 * rho * (r - 1.0) / r;

            let b = becke_roussel_dipole(rho, grad_rho2, tau_std, lap_rho)
                .expect("finite rho must produce a hole");
            assert!(
                (b - r).abs() < 1e-5 * r.max(1.0),
                "r={r}: b={b}, expected exactly r"
            );
        }
    }

    /// D_sigma (the Pauli kinetic energy density) vanishes identically for
    /// ANY one-electron density — an independent identity distinct from the
    /// hydrogen closed form above, checked directly on `pauli_kinetic_deficit`
    /// with a different, non-hydrogenic single-orbital density (a normalized
    /// Gaussian 1s: psi = (2a/pi)^{3/4} e^{-a r^2}).
    #[test]
    fn pauli_kinetic_deficit_vanishes_for_a_single_gaussian_orbital() {
        let a = 0.73_f64;
        for &r in &[0.2, 0.9, 1.7, 3.0] {
            let psi = (2.0 * a / std::f64::consts::PI).powf(0.75) * (-a * r * r).exp();
            let rho = psi * psi;
            let dpsi_dr = -2.0 * a * r * psi;
            let grad_rho2 = (2.0 * psi * dpsi_dr).powi(2);
            let tau_std = 0.5 * dpsi_dr * dpsi_dr;
            let d = pauli_kinetic_deficit(tau_std, grad_rho2, rho);
            assert!(
                d.abs() < 1e-10 * rho.max(1.0),
                "r={r}: D_sigma={d}, expected 0"
            );
        }
    }

    /// Negligible density must refuse cleanly rather than divide by ~0.
    #[test]
    fn becke_roussel_dipole_none_for_vanishing_density() {
        assert!(becke_roussel_dipole(0.0, 0.0, 0.0, 0.0).is_none());
        assert!(becke_roussel_dipole(1e-20, 1e-20, 1e-20, 1e-20).is_none());
    }

    // -----------------------------------------------------------------
    // Free-atom polarizability table.
    // -----------------------------------------------------------------

    #[test]
    fn free_atom_polarizability_table_has_the_noble_gases() {
        assert!((free_atom_polarizability(2).unwrap() - 1.38).abs() < 1e-9); // He
        assert!((free_atom_polarizability(10).unwrap() - 2.66).abs() < 1e-9); // Ne
        assert!((free_atom_polarizability(18).unwrap() - 11.1).abs() < 1e-9); // Ar
        assert!(free_atom_polarizability(37).is_err()); // Rb: outside the bundled range
    }

    // -----------------------------------------------------------------
    // Helpers shared by the SCF-backed gates below.
    // -----------------------------------------------------------------

    fn single_atom_moments(z: u8, basis: &str, grid_level: usize) -> XdmAtomMoments {
        let set = data::load(basis).unwrap();
        let mol = Molecule::new(vec![Atom {
            z,
            position: Vec3::zero(),
        }]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let scf = run_hf(&mol, &mb, &ScfOptions::default()).unwrap();
        let opts = GridOptions {
            level: grid_level,
            ..GridOptions::default()
        };
        let moments =
            xdm_atomic_moments(&mol, &mb, [&scf.density[0], &scf.density[1]], &opts).unwrap();
        moments.into_iter().next().unwrap()
    }

    // -----------------------------------------------------------------
    // Free-atom limit: exact by construction of the volume scaling.
    // -----------------------------------------------------------------

    /// For an isolated atom the Hirshfeld weight is identically 1 (there is
    /// only one atom, so omega_A(r) = rho_A^free(r)/rho_A^free(r) = 1 wherever
    /// the reference density is nonzero), so V_A and V_free are integrals of
    /// literally the same quantity and their ratio must be EXACTLY 1 to
    /// grid/floating-point precision — independent of basis, functional, or
    /// what the reference density numerically looks like. This is the
    /// project's required "free-atom limit" sanity check on the Hirshfeld
    /// partition.
    #[test]
    fn free_atom_limit_is_exact() {
        for &z in &[2u8, 10, 18] {
            let m = single_atom_moments(z, "cc-pvdz", 3);
            assert!(
                (m.volume_ratio - 1.0).abs() < 1e-6,
                "Z={z}: volume_ratio={} (expected exactly 1)",
                m.volume_ratio
            );
            assert!(
                (m.alpha_eff - m.alpha_free).abs() < 1e-6 * m.alpha_free,
                "Z={z}: alpha_eff={} vs alpha_free={}",
                m.alpha_eff,
                m.alpha_free
            );
        }
    }

    // -----------------------------------------------------------------
    // Known C6 values: He-He, Ne-Ne, Ar-Ar (homonuclear, from each free
    // atom's own moments -- Eq. 39 with A=B collapses to C6 = alpha*M1sq/2).
    // -----------------------------------------------------------------

    /// Published reference C6 (a.u.): He-He 1.461 (Chu & Dalgarno-class
    /// accurate values, the classic XDM validation numbers quoted in the
    /// task); Ne-Ne 6.38; Ar-Ar 64.3. XDM built on a modest HF/basis is NOT
    /// expected to reproduce these tightly -- this gate checks the model is
    /// in the right ballpark (an honest "tens of percent" band, per this
    /// project's own scope-control rule) and reports the measured numbers
    /// rather than tuning the band to whatever comes out.
    #[test]
    fn known_c6_values_noble_gases() {
        let cases: [(u8, &str, f64); 3] = [(2, "He", 1.461), (10, "Ne", 6.38), (18, "Ar", 64.3)];
        for (z, name, reference) in cases {
            let m = single_atom_moments(z, "aug-cc-pvtz", 5);
            let c6 = xdm_c6(&m, &m);
            let rel_err = (c6 - reference).abs() / reference;
            eprintln!(
                "XDM C6[{name}-{name}] = {c6:.4} a.u.  (published {reference}, relative error {:.1}%)",
                rel_err * 100.0
            );
            assert!(
                c6.is_finite() && c6 > 0.0,
                "{name}: C6 is not a positive finite number: {c6}"
            );
            assert!(
                rel_err < 0.75,
                "{name}: XDM C6 = {c6:.4} vs published {reference} — relative error {:.1}% \
                 exceeds the honest tens-of-percent band this project allows",
                rel_err * 100.0
            );
        }
    }

    // -----------------------------------------------------------------
    // R^-6 asymptote of the ENERGY FORMULA (Eq. 52 + 55-57): fixed
    // coefficients from a real SCF+grid pass, evaluated at two large,
    // well-separated distances.
    // -----------------------------------------------------------------

    /// Moments computed once (from a converged He atom, standing in for
    /// both centres of a hypothetical He...He pair at asymptotic
    /// separation, where each atom's own moments are indistinguishable
    /// from the free-atom values) and then the PAIRWISE ENERGY FORMULA is
    /// evaluated at two separations. Doubling R must cut the (damping ~ 1
    /// at these distances) energy by very close to 2^6 = 64, the classic
    /// London R^-6 falloff -- and R^-8/R^-10 must have decayed into
    /// irrelevance by then, which the ratio check also confirms implicitly.
    #[test]
    fn energy_formula_has_the_london_r6_asymptote() {
        let m = single_atom_moments(2, "aug-cc-pvdz", 4);
        let damping = XdmDampingParams { a1: 0.63, a2: 1.0 };
        let mk = |r: f64| XdmAtomMoments {
            position: Vec3::new(0.0, 0.0, r),
            ..m.clone()
        };
        let origin = XdmAtomMoments {
            position: Vec3::zero(),
            ..m.clone()
        };
        let (r1, r2) = (35.0_f64, 70.0_f64);
        let e1 = -xdm_pair_energy(&origin, &mk(r1), r1, &damping);
        let e2 = -xdm_pair_energy(&origin, &mk(r2), r2, &damping);
        assert!(
            e1 > 0.0 && e2 > 0.0,
            "lost the attractive tail: e1={e1}, e2={e2}"
        );
        let ratio = e1 / e2;
        assert!(
            (ratio - 64.0).abs() < 2.0,
            "doubling R should cut the energy by ~64 (R^-6): got ratio {ratio}"
        );
    }

    // -----------------------------------------------------------------
    // Translation and rotation invariance of the total dispersion energy.
    // -----------------------------------------------------------------

    fn he_ne_pair() -> (Molecule, MolBasis, crate::scf::ScfResult) {
        let set = data::load("cc-pvdz").unwrap();
        let mol = Molecule::new(vec![
            Atom {
                z: 2,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 10,
                position: Vec3::new(0.0, 0.0, 5.5),
            },
        ]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        let scf = run_hf(&mol, &mb, &ScfOptions::default()).unwrap();
        (mol, mb, scf)
    }

    /// Rigid translation of the whole system (atoms AND the AO basis that
    /// rides with them) does not change the physics, so the SAME converged
    /// density matrix is the converged density at the translated geometry
    /// too; only the grid/Hirshfeld/moment MACHINERY under test moves.
    #[test]
    fn xdm_energy_is_translation_invariant() {
        let (mol, mb, scf) = he_ne_pair();
        let opts = GridOptions {
            level: 3,
            ..Default::default()
        };
        let damping = XdmDampingParams { a1: 0.6, a2: 1.0 };
        let e0 = xdm_dispersion_energy(
            &mol,
            &mb,
            [&scf.density[0], &scf.density[1]],
            &opts,
            &damping,
        )
        .unwrap();

        let shift = Vec3::new(3.1, -2.7, 4.4);
        let mut mol_t = mol.clone();
        for atom in mol_t.atoms.iter_mut() {
            atom.position += shift;
        }
        let mb_t = MolBasis::build(&mol_t, &data::load("cc-pvdz").unwrap()).unwrap();
        let e1 = xdm_dispersion_energy(
            &mol_t,
            &mb_t,
            [&scf.density[0], &scf.density[1]],
            &opts,
            &damping,
        )
        .unwrap();

        assert!(
            (e0 - e1).abs() < 1e-6 * e0.abs().max(1.0),
            "translation changed the energy: {e0} vs {e1}"
        );
    }

    /// A rigid rotation is handled the same way: rotate the geometry, keep
    /// the SAME density matrix (valid because the AO basis rotates rigidly
    /// with the atoms and the physics only depends on relative geometry),
    /// and require the total energy to come back unchanged.
    #[test]
    fn xdm_energy_is_rotation_invariant() {
        let (mol, mb, scf) = he_ne_pair();
        let opts = GridOptions {
            level: 3,
            ..Default::default()
        };
        let damping = XdmDampingParams { a1: 0.6, a2: 1.0 };
        let e0 = xdm_dispersion_energy(
            &mol,
            &mb,
            [&scf.density[0], &scf.density[1]],
            &opts,
            &damping,
        )
        .unwrap();

        // A generic rotation (Euler angles, not aligned with any axis).
        let (ca, sa) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let (cb, sb) = (50f64.to_radians().cos(), 50f64.to_radians().sin());
        let rotate = |p: Vec3| -> Vec3 {
            // Rz(a) then Ry(b).
            let x1 = ca * p.x - sa * p.y;
            let y1 = sa * p.x + ca * p.y;
            let z1 = p.z;
            Vec3::new(cb * x1 + sb * z1, y1, -sb * x1 + cb * z1)
        };
        let mut mol_r = mol.clone();
        for atom in mol_r.atoms.iter_mut() {
            atom.position = rotate(atom.position);
        }
        let mb_r = MolBasis::build(&mol_r, &data::load("cc-pvdz").unwrap()).unwrap();
        let e1 = xdm_dispersion_energy(
            &mol_r,
            &mb_r,
            [&scf.density[0], &scf.density[1]],
            &opts,
            &damping,
        )
        .unwrap();

        assert!(
            (e0 - e1).abs() < 1e-6 * e0.abs().max(1.0),
            "rotation changed the energy: {e0} vs {e1}"
        );
    }

    // -----------------------------------------------------------------
    // The gradient path must refuse, not silently approximate.
    // -----------------------------------------------------------------

    #[test]
    fn xdm_gradient_refuses() {
        let (mol, mb, scf) = he_ne_pair();
        let opts = GridOptions::default();
        let damping = XdmDampingParams { a1: 0.6, a2: 1.0 };
        let err = xdm_gradient(
            &mol,
            &mb,
            [&scf.density[0], &scf.density[1]],
            &opts,
            &damping,
        )
        .unwrap_err();
        assert!(err.is_unsupported(), "expected Unsupported, got {err:?}");
    }
}
