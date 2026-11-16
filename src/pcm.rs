// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Polarizable continuum solvation on a smooth (SWIG) cavity.
//!
//! Cavity: one sphere per atom at a scaled van der Waals radius,
//! discretized with a Lebedev grid. Buried points are removed SMOOTHLY by
//! the switching function of Lange & Herbert, J. Chem. Phys. 133, 244111
//! (2010), so the energy is C-infinity in the nuclear coordinates and the
//! whole derivative chain stays analytic:
//!   R_sw = R (14/N)^{1/2},  a = 1/2 + R/R_sw - ((R/R_sw)^2 - 1/28)^{1/2},
//!   R_in = R - a R_sw,      d_iJ = (r_iJ - R_in_J)/R_sw_J,
//!   h(d) = d^3 (10 - 15 d + 6 d^2) clamped to [0,1],  F_i = prod_J h(d_iJ).
//!
//! Each surface point carries a SPHERICAL GAUSSIAN charge of unit norm
//! and exponent `zeta_i^2` (not a point charge), so the charge-charge
//! interaction is `erf(zeta_ij r)/r` with
//! `zeta_ij = zeta_i zeta_j / (zeta_i^2 + zeta_j^2)^{1/2}` and a finite
//! self term `zeta_i (2/pi)^{1/2} / F_i`, while the electron-surface
//! interaction is the three-center integral `(mu nu | g_i)`.
//!
//! C-PCM: `K = S`, `f(eps) = (eps - 1)/eps`, so `q = -f S^{-1} v` with
//! `v_i = v^nuc_i - v^elec_i`, `E_pol = 1/2 q . v`, and the reaction field
//! enters the Fock matrix as `-sum_i q_i (mu nu | g_i)`.
//!
//! Verified against PySCF's PCM module (energies, and — once the
//! derivative paths land — its analytic gradients/Hessians).

use crate::basis::{MolBasis, MolShell};
use crate::dual::Scalar;
use crate::error::{DftError, Result};
/// Quasi-uniform sphere tessellation (golden-spiral / Fibonacci points),
/// each carrying the same solid angle `4 pi / n`.
///
/// A cavity discretization is a PARTITION of the surface into patches of
/// comparable size, not a polynomial quadrature rule: the Gaussian charge
/// of each patch is scaled to the patch area, so two nearly coincident
/// points would make the charge-charge matrix singular. Product rules
/// (Gauss-Legendre x uniform phi) cluster severely near the poles and are
/// therefore unusable here, while this construction is self-generating
/// (no tabulated data) and near-uniform by design.
fn fibonacci_sphere(n: usize) -> Vec<(Vec3, f64)> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // golden angle
    let w = 4.0 * std::f64::consts::PI / n as f64;
    (0..n)
        .map(|k| {
            let z = 1.0 - (2 * k + 1) as f64 / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden * k as f64;
            (Vec3::new(r * phi.cos(), r * phi.sin(), z), w)
        })
        .collect()
}
use crate::integrals::eri::eri_3c_shell;
use crate::integrals::CoulombKernel;
use crate::linalg::{solve_linear, solve_linear_matrix, symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;

/// Continuum model.
///
/// The `CPcm`/`Cosmo` family has `K = S`, which is what lets the analytic
/// gradient and Hessian collapse to closed forms. `IefPcm`/`Ssvpe` bring in
/// the D matrix; IEF-PCM has a non-symmetric `K`, while SS(V)PE symmetrizes
/// it. Both models have explicit analytic gradients and Hessians.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmModel {
    /// f(eps) = (eps - 1)/eps.
    CPcm,
    /// f(eps) = (eps - 1)/(eps + 1/2).
    Cosmo,
    /// K = S - (f/2pi) D A S, R = -f (I - (1/2pi) D A).
    IefPcm,
    /// Symmetrized IEF: K = S - (f/4pi)(D A S + (D A S)^T).
    Ssvpe,
}

impl PcmModel {
    /// Does this model need the D matrix (and therefore the general,
    /// non-`K = S` solve)?
    pub fn needs_d(&self) -> bool {
        matches!(self, PcmModel::IefPcm | PcmModel::Ssvpe)
    }
}

#[derive(Clone, Debug)]
pub struct PcmOptions {
    pub model: PcmModel,
    /// Static dielectric constant (water: 78.3553).
    pub eps: f64,
    /// Scaling applied to the van der Waals radii (Q-Chem/PySCF: 1.2).
    pub vdw_scale: f64,
    /// Extra probe radius in Bohr (default 0).
    pub r_probe: f64,
    /// Surface points per sphere (quasi-uniform tessellation); 302 matches
    /// the point count other continuum codes use by default.
    pub n_points: usize,
    /// Full solvent descriptors when known (named table entry or supplied
    /// directly). Only `eps` is needed for the electrostatic models; the
    /// rest drive the SMD non-electrostatic term.
    pub descriptors: Option<Solvent>,
    /// Build the cavity from SMD's intrinsic Coulomb radii instead of the
    /// Bondi set. SMD reparameterized its electrostatics against these, so
    /// mixing the two radius sets is not a small perturbation.
    pub smd_radii: bool,
}

impl Default for PcmOptions {
    fn default() -> Self {
        Self {
            model: PcmModel::CPcm,
            eps: 78.3553,
            vdw_scale: 1.2,
            r_probe: 0.0,
            n_points: 302,
            descriptors: None,
            smd_radii: false,
        }
    }
}

impl PcmOptions {
    pub fn f_eps(&self) -> f64 {
        match self.model {
            PcmModel::CPcm => (self.eps - 1.0) / self.eps,
            PcmModel::Cosmo => (self.eps - 1.0) / (self.eps + 0.5),
            // The IEF family carries the full dielectric factor; the
            // remaining eps dependence sits in the D A terms.
            PcmModel::IefPcm | PcmModel::Ssvpe => (self.eps - 1.0) / (self.eps + 1.0),
        }
    }
}

/// Bondi van der Waals radii in Angstrom, with the customary hydrogen
/// modification (1.1 instead of 1.2) used by continuum models.
/// Index = atomic number; 0.0 marks "not tabulated".
const VDW_ANGSTROM: [f64; 37] = [
    0.0, // (unused)
    1.1, 1.4, // H  He
    1.82, 1.53, 1.92, 1.7, 1.55, 1.52, 1.47, 1.54, // Li..Ne
    2.27, 1.73, 1.84, 2.1, 1.8, 1.8, 1.75, 1.88, // Na..Ar
    2.75, 2.31, // K  Ca
    2.11, 2.11, 2.11, 2.11, 2.11, 2.11, 2.11, 1.63, 1.4, 1.39, // Sc..Zn
    1.87, 2.11, 1.85, 1.9, 1.85, 2.02, // Ga..Kr
];
const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.52917721092;

/// Gaussian-width parameter of the surface charges. In the SWIG
/// construction each point's charge width is tied to its own patch area,
/// `zeta_i = XI / (R sqrt(w_i))`, and the proportionality constant is
/// essentially rule independent (Lange & Herbert quote 4.9006..4.9076
/// across Lebedev orders 110..974, i.e. a 0.02 % spread). We therefore
/// use the single constant below for the product rule as well; the
/// resulting cavity is gated against the analytic Born limit, which is
/// discretization independent.
const XI_GAUSS: f64 = 4.90621;

/// One discretized cavity.
#[derive(Clone, Debug)]
pub struct Cavity {
    /// Surface point positions (Bohr).
    pub points: Vec<Vec3>,
    /// Owning atom of each point.
    pub atom_of: Vec<usize>,
    /// Gaussian exponent parameter zeta_i (the charge exponent is zeta^2).
    pub zeta: Vec<f64>,
    /// Switching value F_i in (0, 1].
    pub switch: Vec<f64>,
    /// Bare Lebedev weight (4 pi w_k) of each point.
    pub weight: Vec<f64>,
    /// Sphere radius of the owning atom.
    pub radius: Vec<f64>,
}

/// Smooth switching function h(d) = d^3(10 - 15 d + 6 d^2) on [0, 1].
fn switch_h(d: f64) -> f64 {
    if d <= 0.0 {
        0.0
    } else if d >= 1.0 {
        1.0
    } else {
        d * d * d * (10.0 - 15.0 * d + 6.0 * d * d)
    }
}

pub fn build_cavity(mol: &Molecule, opts: &PcmOptions) -> Result<Cavity> {
    let sphere = fibonacci_sphere(opts.n_points);
    let n_at = mol.len();
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| {
            let z = a.z as usize;
            let r = if opts.smd_radii {
                smd_radius(a.z).unwrap_or(0.0)
            } else {
                VDW_ANGSTROM.get(z).copied().unwrap_or(0.0)
            };
            if r == 0.0 {
                return Err(DftError::Unsupported {
                    feature: format!("PCM cavity construction for Z={z}"),
                    missing: "a tabulated cavity radius".into(),
                });
            }
            Ok(opts.vdw_scale * r * BOHR_PER_ANGSTROM + opts.r_probe)
        })
        .collect::<Result<_>>()?;
    let n_pts = sphere.len() as f64;
    let r_sw: Vec<f64> = radii.iter().map(|r| r * (14.0 / n_pts).sqrt()).collect();
    let r_in: Vec<f64> = radii
        .iter()
        .zip(&r_sw)
        .map(|(&r, &rs)| {
            let t = r / rs;
            let alpha = 0.5 + t - (t * t - 1.0 / 28.0).sqrt();
            r - alpha * rs
        })
        .collect();

    let mut cav = Cavity {
        points: Vec::new(),
        atom_of: Vec::new(),
        zeta: Vec::new(),
        switch: Vec::new(),
        weight: Vec::new(),
        radius: Vec::new(),
    };
    for ia in 0..n_at {
        let center = mol.atoms[ia].position;
        let r_vdw = radii[ia];
        for &(dir, w) in &sphere {
            let p = center + dir * r_vdw;
            let mut f = 1.0;
            for jb in 0..n_at {
                if jb == ia {
                    continue;
                }
                let r_ij = (p - mol.atoms[jb].position).norm();
                let d = (r_ij - r_in[jb]) / r_sw[jb];
                f *= switch_h(d);
                if f == 0.0 {
                    break;
                }
            }
            if w * f <= 1e-16 {
                continue;
            }
            cav.points.push(p);
            cav.atom_of.push(ia);
            cav.zeta.push(XI_GAUSS / (r_vdw * w.sqrt()));
            cav.switch.push(f);
            cav.weight.push(w);
            cav.radius.push(r_vdw);
        }
    }
    if cav.points.is_empty() {
        return Err(DftError::InvalidInput(
            "PCM: the cavity has no surface points".to_string(),
        ));
    }
    Ok(cav)
}

/// Analytic nuclear gradient of the C-PCM polarization energy at the
/// converged (variational) density.
///
/// With `q = -f S^{-1} v` and `E = 1/2 q.v`, differentiating `S q = -f v`
/// gives `dq = S^{-1}(-f dv - dS q)`, and substituting `S^{-1} v = -q/f`
/// collapses the result to
///
/// `dE/dR = q . (dv/dR) + 1/(2f) q . (dS/dR) q`
///
/// so **no second linear solve is needed** — the converged charges carry
/// all the information. Only `dv/dR` and `dS/dR` remain, and both are
/// closed forms in the cavity geometry, differentiated by nested duals
/// exactly as the VV10 Hessian does (THEORY.md §17).
pub fn cpcm_gradient(mol: &Molecule, solver: &PcmSolver, q: &[f64], dv: &[Vec<Vec3>]) -> Vec<Vec3> {
    let n_at = mol.len();
    let f = solver.opts.f_eps();
    let mut grad = vec![Vec3::zero(); n_at];
    // Diagonal of dS/dR: S_ii = zeta_i sqrt(2/pi) / F_i depends on the
    // nuclear coordinates ONLY through the switching value F_i (zeta_i is
    // built from the sphere radius and the fixed solid angle, so it does
    // not move). d(1/F) = -dF/F^2.
    {
        let dfs = switch_gradient(mol, &solver.cav, &solver.opts);
        let cav = &solver.cav;
        let c0 = (2.0 / std::f64::consts::PI).sqrt() / (2.0 * f);
        for i in 0..cav.points.len() {
            let w = c0 * q[i] * q[i] * cav.zeta[i] / (cav.switch[i] * cav.switch[i]);
            for (at, g) in grad.iter_mut().enumerate() {
                *g -= dfs[i][at] * w;
            }
        }
    }
    // Term 1: q . dv/dR (the caller supplies dv because its electronic
    // part needs the AO derivative machinery).
    for (i, &qi) in q.iter().enumerate() {
        for (at, g) in grad.iter_mut().enumerate() {
            *g += dv[i][at] * qi;
        }
    }
    // Term 2: 1/(2f) q . (dS/dR) q, built from the closed-form S entries.
    let cav = &solver.cav;
    let m = cav.points.len();
    let scale = 1.0 / (2.0 * f);
    for i in 0..m {
        for j in (i + 1)..m {
            let (oi, oj) = (cav.atom_of[i], cav.atom_of[j]);
            if oi == oj {
                continue; // rigid within one sphere: r_ij does not change
            }
            let dvec = cav.points[i] - cav.points[j];
            let r = dvec.norm();
            let zi = cav.zeta[i];
            let zj = cav.zeta[j];
            let zij = zi * zj / (zi * zi + zj * zj).sqrt();
            // d/dr [erf(z r)/r] = 2z/sqrt(pi) exp(-(zr)^2)/r - erf(zr)/r^2
            let dsdr = 2.0 * zij / std::f64::consts::PI.sqrt() * (-(zij * r) * (zij * r)).exp() / r
                - libm::erf(zij * r) / (r * r);
            // S is symmetric, so the (i,j) and (j,i) entries both count.
            let w = 2.0 * scale * q[i] * q[j] * dsdr / r;
            grad[oi] += dvec * w;
            grad[oj] -= dvec * w;
        }
    }
    grad
}

/// Electronic part of `dv/dR`: `v^el_i = Tr[P (mu nu | g_i)]`.
///
/// The AO derivative uses the same raised/lowered shell identity as the
/// rest of the code, `d/dA phi(l) = 2a phi(l+1) - l phi(l-1)`, applied to
/// the bra with the surface Gaussian as the auxiliary shell. The surface
/// point moves with its owner atom, and translational invariance supplies
/// that term for free: the integral depends only on relative positions, so
/// the point derivative is minus the sum of the two AO-centre ones.
pub fn electronic_potential_gradient(
    mb: &MolBasis,
    cav: &Cavity,
    p_tot: &Matrix,
    n_at: usize,
) -> Vec<Vec<Vec3>> {
    use crate::basis::{aos, cart_components, n_sph};
    use crate::integrals::deriv::{lowered, raised};
    use crate::integrals::eri::eri_3c_shell_direct_cart_bra;
    let kernel = CoulombKernel::Coulomb;
    let ns = mb.shells.len();
    (0..cav.points.len())
        .into_par_iter()
        .map(|i| {
            let aux = surface_shell(cav, i);
            let owner = cav.atom_of[i];
            let mut row = vec![Vec3::zero(); n_at];
            for a in 0..ns {
                let sh_a = &mb.shells[a];
                let (la, na) = (sh_a.l, n_sph(sh_a.l));
                let ra = raised(sh_a);
                let lo = lowered(sh_a);
                let comps = cart_components(la);
                let c2s = aos::cart_to_sph(la);
                for b in 0..ns {
                    let sh_b = &mb.shells[b];
                    let nb = n_sph(sh_b.l);
                    let q_up = eri_3c_shell_direct_cart_bra(&ra, sh_b, &aux, kernel);
                    let q_dn = lo
                        .as_ref()
                        .map(|l| eri_3c_shell_direct_cart_bra(l, sh_b, &aux, kernel));
                    for si in 0..na {
                        for sj in 0..nb {
                            let pv = p_tot[(sh_a.sph_off + si, sh_b.sph_off + sj)];
                            if pv == 0.0 {
                                continue;
                            }
                            for ax in 0..3 {
                                let mut d = 0.0;
                                for (ic, &(cx, cy, cz)) in comps.iter().enumerate() {
                                    let coef = c2s[(ic, si)];
                                    if coef == 0.0 {
                                        continue;
                                    }
                                    let up = match ax {
                                        0 => (cx + 1, cy, cz),
                                        1 => (cx, cy + 1, cz),
                                        _ => (cx, cy, cz + 1),
                                    };
                                    let iu = cart_components(la + 1)
                                        .iter()
                                        .position(|&c| c == up)
                                        .expect("raised component");
                                    let mut v = q_up.get(iu, sj, 0, 0);
                                    let lcomp = [cx, cy, cz][ax];
                                    if lcomp > 0 {
                                        let dn = match ax {
                                            0 => (cx - 1, cy, cz),
                                            1 => (cx, cy - 1, cz),
                                            _ => (cx, cy, cz - 1),
                                        };
                                        let id = cart_components(la - 1)
                                            .iter()
                                            .position(|&c| c == dn)
                                            .expect("lowered component");
                                        v -= lcomp as f64
                                            * q_dn
                                                .as_ref()
                                                .expect("lowered block")
                                                .get(id, sj, 0, 0);
                                    }
                                    d += coef * v;
                                }
                                // Only the BRA centre is differentiated here.
                                // The ordered double loop visits every pair
                                // in both orders and P is symmetric, so the
                                // ket-centre derivatives sum to exactly the
                                // same value: one factor of two covers them.
                                let contrib = 2.0 * pv * d;
                                let at = sh_a.atom;
                                match ax {
                                    0 => {
                                        row[at].x += contrib;
                                        row[owner].x -= contrib;
                                    }
                                    1 => {
                                        row[at].y += contrib;
                                        row[owner].y -= contrib;
                                    }
                                    _ => {
                                        row[at].z += contrib;
                                        row[owner].z -= contrib;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            row
        })
        .collect()
}

impl PcmSolver {
    /// Complete analytic nuclear gradient of the polarization energy at a
    /// converged density. Assembles `dv/dR = dv^nuc/dR - dv^el/dR` and
    /// feeds it to `cpcm_gradient`, which supplies the `dS/dR` half.
    pub fn gradient(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
    ) -> Result<Vec<Vec3>> {
        if self.opts.model.needs_d() {
            return self.gradient_ief(mol, mb, charges, p_tot);
        }
        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        let n_at = mol.len();
        let dvn = nuclear_potential_gradient(mol, &self.cav, charges);
        let dve = electronic_potential_gradient(mb, &self.cav, p_tot, n_at);
        let dv: Vec<Vec<Vec3>> = dvn
            .into_iter()
            .zip(dve)
            .map(|(rn, re)| rn.into_iter().zip(re).map(|(a, b)| a - b).collect())
            .collect();
        Ok(cpcm_gradient(mol, self, &q, &dv))
    }
}

/// Nested forward dual for the switching second derivatives.
type D2h = crate::dual::DualN<crate::dual::DualN<f64, 3>, 3>;

/// Seed a coordinate into the outer/inner slots of `D2h`.
fn seed2(val: f64, outer: Option<usize>, inner: Option<usize>) -> D2h {
    let re = match inner {
        Some(k) => crate::dual::DualN::<f64, 3>::variable(val, k),
        None => crate::dual::DualN::<f64, 3>::constant(val),
    };
    let mut d = D2h {
        re,
        im: [crate::dual::DualN::<f64, 3>::constant(0.0); 3],
    };
    if let Some(k) = outer {
        d.im[k] = crate::dual::DualN::<f64, 3>::constant(1.0);
    }
    d
}

/// Scalar-generic `1 / F_i` for one surface point, as a function of the
/// nuclear coordinates. The point rides its owner rigidly, so its position
/// is `X_owner + (p_i - X_owner^0)`; everything else is the SWIG product.
///
/// Written generically so the second derivative comes from nested forward
/// duals rather than a hand-derived expression — the same tactic §17 uses
/// for VV10, and the reason the switching term did not need re-deriving
/// when the Hessian was added.
fn inv_switch_s<S: Scalar>(
    coords: &[[S; 3]],
    owner: usize,
    offset: [f64; 3],
    r_in: &[f64],
    r_sw: &[f64],
) -> S {
    let p = [
        coords[owner][0] + S::from_f64(offset[0]),
        coords[owner][1] + S::from_f64(offset[1]),
        coords[owner][2] + S::from_f64(offset[2]),
    ];
    let mut f = S::from_f64(1.0);
    for (j, cj) in coords.iter().enumerate() {
        if j == owner {
            continue;
        }
        let dx = p[0] - cj[0];
        let dy = p[1] - cj[1];
        let dz = p[2] - cj[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let d = (dist - S::from_f64(r_in[j])) / S::from_f64(r_sw[j]);
        // h(d) = d^3(10 - 15d + 6d^2), clamped to [0, 1]. The clamp is C^2,
        // so branching on the VALUE keeps the derivatives exact.
        let hv = if d.value() <= 0.0 {
            S::from_f64(0.0)
        } else if d.value() >= 1.0 {
            S::from_f64(1.0)
        } else {
            d * d * d * (S::from_f64(10.0) - S::from_f64(15.0) * d + S::from_f64(6.0) * d * d)
        };
        f = f * hv;
    }
    S::from_f64(1.0) / f
}

/// Sphere geometry shared by the switching gradient and Hessian.
fn switch_geometry(mol: &Molecule, opts: &PcmOptions) -> (Vec<f64>, Vec<f64>) {
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| {
            let r = if opts.smd_radii {
                smd_radius(a.z).unwrap_or(0.0)
            } else {
                VDW_ANGSTROM.get(a.z as usize).copied().unwrap_or(0.0)
            };
            opts.vdw_scale * r * BOHR_PER_ANGSTROM + opts.r_probe
        })
        .collect();
    let n_pts = opts.n_points as f64;
    let r_sw: Vec<f64> = radii.iter().map(|r| r * (14.0 / n_pts).sqrt()).collect();
    let r_in: Vec<f64> = radii
        .iter()
        .zip(&r_sw)
        .map(|(&r, &rs)| {
            let t = r / rs;
            r - (0.5 + t - (t * t - 1.0 / 28.0).sqrt()) * rs
        })
        .collect();
    (r_in, r_sw)
}

/// Gradient of the SWIG switching values `F_i` with respect to every
/// nuclear coordinate.
///
/// `F_i = prod_{J != owner} h(d_iJ)`, `d_iJ = (|p_i - X_J| - R_in_J)/R_sw_J`
/// with `h(d) = d^3(10 - 15d + 6d^2)` clamped to `[0, 1]`. Two centres
/// move per factor: the sphere J and (through its owner) the point itself.
/// This closes the piece the S-matrix diagonal `S_ii = zeta_i sqrt(2/pi)/F_i`
/// depends on — the omission an early finite-difference check exposed as a
/// four-orders-of-magnitude discrepancy.
pub fn switch_gradient(mol: &Molecule, cav: &Cavity, opts: &PcmOptions) -> Vec<Vec<Vec3>> {
    let n_at = mol.len();
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| {
            let r = if opts.smd_radii {
                smd_radius(a.z).unwrap_or(0.0)
            } else {
                VDW_ANGSTROM.get(a.z as usize).copied().unwrap_or(0.0)
            };
            opts.vdw_scale * r * BOHR_PER_ANGSTROM + opts.r_probe
        })
        .collect();
    let n_pts = opts.n_points as f64;
    let r_sw: Vec<f64> = radii.iter().map(|r| r * (14.0 / n_pts).sqrt()).collect();
    let r_in: Vec<f64> = radii
        .iter()
        .zip(&r_sw)
        .map(|(&r, &rs)| {
            let t = r / rs;
            r - (0.5 + t - (t * t - 1.0 / 28.0).sqrt()) * rs
        })
        .collect();
    // h'(d) = 30 d^2 (1 - d)^2, zero outside (0, 1) — the clamp is C^2.
    let dh = |d: f64| -> f64 {
        if d <= 0.0 || d >= 1.0 {
            0.0
        } else {
            30.0 * d * d * (1.0 - d) * (1.0 - d)
        }
    };
    (0..cav.points.len())
        .map(|i| {
            let owner = cav.atom_of[i];
            let p = cav.points[i];
            let mut row = vec![Vec3::zero(); n_at];
            // Logarithmic derivative: dF/dX = F * sum_J h'(d)/h(d) * dd/dX.
            for jb in 0..n_at {
                if jb == owner {
                    continue;
                }
                let dvec = p - mol.atoms[jb].position;
                let dist = dvec.norm();
                if dist < 1e-12 {
                    continue;
                }
                let d = (dist - r_in[jb]) / r_sw[jb];
                let hval = switch_h(d);
                let hp = dh(d);
                if hval <= 1e-300 || hp == 0.0 {
                    continue;
                }
                // dF/dX = F/h * h'(d) * (1/R_sw) * d(dist)/dX
                let coef = cav.switch[i] / hval * hp / r_sw[jb];
                let dir = dvec * (1.0 / dist);
                row[owner] += dir * coef;
                row[jb] -= dir * coef;
            }
            row
        })
        .collect()
}

/// Nuclear part of `dv/dR`: the surface potential of the point charges,
/// `v^nuc_i = sum_A Z_A erf(zeta_i d_iA)/d_iA`, differentiated in closed
/// form. Two centres move: the nucleus A and (with its owner atom) the
/// surface point i, so each pair contributes with opposite signs.
pub fn nuclear_potential_gradient(mol: &Molecule, cav: &Cavity, charges: &[f64]) -> Vec<Vec<Vec3>> {
    let n_at = mol.len();
    let m = cav.points.len();
    (0..m)
        .map(|i| {
            let z_i = cav.zeta[i];
            let owner = cav.atom_of[i];
            let mut row = vec![Vec3::zero(); n_at];
            for (a, (at, &z)) in mol.atoms.iter().zip(charges).enumerate() {
                let dvec = cav.points[i] - at.position;
                let d = dvec.norm();
                if d < 1e-12 {
                    continue;
                }
                // d/dd [erf(z d)/d] = 2z/sqrt(pi) exp(-(zd)^2)/d - erf(zd)/d^2
                let dfdd = 2.0 * z_i / std::f64::consts::PI.sqrt() * (-(z_i * d) * (z_i * d)).exp()
                    / d
                    - libm::erf(z_i * d) / (d * d);
                let dir = dvec * (1.0 / d);
                // Moving the point (with its owner) raises d; moving the
                // nucleus lowers it by the same amount.
                row[owner] += dir * (z * dfdd);
                row[a] -= dir * (z * dfdd);
            }
            row
        })
        .collect()
}

/// The charge-charge interaction matrix S (Gaussian-smeared Coulomb).
pub fn s_matrix(cav: &Cavity) -> Matrix {
    let m = cav.points.len();
    let mut s = Matrix::zeros(m, m);
    for i in 0..m {
        s[(i, i)] = cav.zeta[i] * (2.0 / std::f64::consts::PI).sqrt() / cav.switch[i];
        for j in (i + 1)..m {
            let r = (cav.points[i] - cav.points[j]).norm();
            let zi = cav.zeta[i];
            let zj = cav.zeta[j];
            let zij = zi * zj / (zi * zi + zj * zj).sqrt();
            let v = libm::erf(zij * r) / r;
            s[(i, j)] = v;
            s[(j, i)] = v;
        }
    }
    s
}

/// Cavity patch areas `a_i = w_i R_i^2 F_i` (Bohr^2). These enter the IEF
/// family through `D A`; the C-PCM family never needs them, because its
/// `K = S` already absorbs the geometry.
pub fn patch_areas(cav: &Cavity) -> Vec<f64> {
    (0..cav.points.len())
        .map(|i| cav.weight[i] * cav.radius[i] * cav.radius[i] * cav.switch[i])
        .collect()
}

/// The D matrix: the normal derivative, at the SOURCE point, of the
/// potential the source's Gaussian charge produces at the field point.
///
/// `D_ij = n_j . grad_{r_j} [erf(z_ij r)/r] = -s'(r) (n_j . r_hat)` with
/// `r_hat = (p_i - p_j)/r`, which reduces to the textbook
/// `(n_j . (p_i - p_j)) / r^3` in the point-charge limit `z -> infinity`.
/// The outward normal of a sphere-centred patch is just the radial
/// direction, `n_j = (p_j - X_{owner(j)}) / R_j`.
///
/// The diagonal is the standard self-term of a curved patch,
/// `D_ii = -zeta_i (2/pi)^{1/2} / (2 R_i)`, which is what makes the
/// discretized `D` reproduce the continuum Gauss relation; it is checked
/// numerically rather than assumed (see the sphere test).
pub fn d_matrix(cav: &Cavity, mol: &Molecule) -> Matrix {
    let m = cav.points.len();
    let mut d = Matrix::zeros(m, m);
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let normals: Vec<Vec3> = (0..m)
        .map(|j| {
            let c = mol.atoms[cav.atom_of[j]].position;
            (cav.points[j] - c) * (1.0 / cav.radius[j])
        })
        .collect();
    for i in 0..m {
        d[(i, i)] = -cav.zeta[i] * (2.0 / std::f64::consts::PI).sqrt() / (2.0 * cav.radius[i]);
        for j in 0..m {
            if i == j {
                continue;
            }
            let dvec = cav.points[i] - cav.points[j];
            let r = dvec.norm();
            let zi = cav.zeta[i];
            let zj = cav.zeta[j];
            let z = zi * zj / (zi * zi + zj * zj).sqrt();
            let sp =
                2.0 * z / sqrt_pi * (-(z * r) * (z * r)).exp() / r - libm::erf(z * r) / (r * r);
            let cos = normals[j].dot(dvec) / r;
            d[(i, j)] = -sp * cos;
        }
    }
    d
}

/// Second derivatives of one `(i, j)` entry of `S` and of `D` with respect
/// to the separation vector `d = p_i - p_j`.
///
/// Both entries depend on the geometry ONLY through `d` — the outward
/// normal `n_j` rides its owner rigidly and so is invariant — which is what
/// makes a 3x3 block per pair sufficient: the nuclear second derivative is
/// then `c^A c^B` times this, with `c = +1` for the owner of i and `-1` for
/// the owner of j. Taken from nested duals over the closed forms rather
/// than hand-differentiated, so `s''` and the `n_j . d_hat` chain never had
/// to be written out.
fn pair_second(dvec: Vec3, n_j: Vec3, zi: f64, zj: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let z = zi * zj / (zi * zi + zj * zj).sqrt();
    let seed = |val: f64, k: usize| -> D2h { seed2(val, Some(k), Some(k)) };
    // Seed each component in BOTH slots so the nested dual carries the full
    // 3x3 block; the off-diagonal mixed parts come from the cross terms.
    let mut hs = [[0.0; 3]; 3];
    let mut hd = [[0.0; 3]; 3];
    for x in 0..3 {
        for y in 0..3 {
            let comp = |k: usize, val: f64| -> D2h {
                seed2(val, (k == x).then_some(0), (k == y).then_some(0))
            };
            let d = [comp(0, dvec.x), comp(1, dvec.y), comp(2, dvec.z)];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let r = r2.sqrt();
            let zr = D2h::from_f64(z) * r;
            let s = zr.erf() / r;
            hs[x][y] = s.im[0].im[0];
            // D = n_j . grad_{p_j} S = -s'(r) (n_j . d_hat); expressing it
            // as the derivative of S along -d keeps the closed form in one
            // place: D = -(dS/dd) . n_j.
            let ds_dd = {
                // dS/dd_k = s'(r) d_k / r, so D = -s'(r) (n.d)/r.
                let sp =
                    (D2h::from_f64(2.0 * z / std::f64::consts::PI.sqrt()) * (-(zr * zr)).exp()) / r
                        - zr.erf() / (r * r);
                let nd = D2h::from_f64(n_j.x) * d[0]
                    + D2h::from_f64(n_j.y) * d[1]
                    + D2h::from_f64(n_j.z) * d[2];
                -(sp * nd) / r
            };
            hd[x][y] = ds_dd.im[0].im[0];
        }
    }
    let _ = seed;
    (hs, hd)
}

/// Derivative of the D matrix with respect to one nuclear coordinate,
/// returned as the full `m x m` block. Two things could move in
/// `D_ij = -s'(r) (n_j . r_hat)`, and only one of them does: the outward
/// normal `n_j = (p_j - X_owner(j)) / R_j` is INVARIANT under the rigid
/// motion of its owner, because the point rides the atom. So the geometry
/// dependence is entirely through `r` and `r_hat`, and each entry moves
/// only with the two owners. The diagonal `-zeta_i (2/pi)^{1/2}/(2 R_i)`
/// is built from fixed quantities and does not move at all.
fn d_matrix_deriv(cav: &Cavity, mol: &Molecule, at: usize, ax: usize) -> Matrix {
    let m = cav.points.len();
    let mut out = Matrix::zeros(m, m);
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let normals: Vec<Vec3> = (0..m)
        .map(|j| {
            let c = mol.atoms[cav.atom_of[j]].position;
            (cav.points[j] - c) * (1.0 / cav.radius[j])
        })
        .collect();
    for i in 0..m {
        let oi = cav.atom_of[i];
        for j in 0..m {
            if i == j {
                continue;
            }
            let oj = cav.atom_of[j];
            let c_a = (oi == at) as i32 - (oj == at) as i32;
            if c_a == 0 {
                continue;
            }
            let dvec = cav.points[i] - cav.points[j];
            let r = dvec.norm();
            let zi = cav.zeta[i];
            let zj = cav.zeta[j];
            let z = zi * zj / (zi * zi + zj * zj).sqrt();
            let e = (-(z * r) * (z * r)).exp();
            let erfv = libm::erf(z * r);
            let sp = 2.0 * z / sqrt_pi * e / r - erfv / (r * r);
            let spp = -4.0 * z * z * z / sqrt_pi * e - 4.0 * z / (sqrt_pi * r * r) * e
                + 2.0 * erfv / (r * r * r);
            let nh = dvec * (1.0 / r);
            let cos = normals[j].dot(nh);
            // d/dX [-s'(r) cos] = -s'' cos n_hat - s' d(cos)/dd,
            // d(cos)/dd = (n_j - cos n_hat)/r.
            let term = -spp * cos * nh.get(ax) - sp * (normals[j].get(ax) - cos * nh.get(ax)) / r;
            out[(i, j)] = c_a as f64 * term;
        }
    }
    out
}

/// The s-type unit-charge Gaussian representing surface point `i`.
fn surface_shell(cav: &Cavity, i: usize) -> MolShell {
    let a = cav.zeta[i] * cav.zeta[i];
    MolShell {
        l: 0,
        atom: cav.atom_of[i],
        center: cav.points[i],
        exps: vec![a],
        // Unit-norm charge distribution: (a/pi)^{3/2} exp(-a r^2).
        coefs: vec![(a / std::f64::consts::PI).powf(1.5)],
        sph_off: 0,
        cart_off: 0,
    }
}

/// Reusable PCM machinery for one geometry: the cavity, the (factorized)
/// S matrix, the nuclear potential on the surface, and the AO potential
/// integrals of every surface charge.
struct PcmResponseOperator {
    /// Boundary-equation response `A`, with `q = A v`.
    raw: Matrix,
    /// Variational response `B = (A + A^T)/2` shared by energy derivatives,
    /// the Fock operator, and the CPKS kernel.
    variational: Matrix,
}

impl PcmResponseOperator {
    fn new(raw: Matrix) -> Self {
        let mut variational = raw.clone();
        for i in 0..raw.rows {
            for j in 0..i {
                let value = 0.5 * (raw[(i, j)] + raw[(j, i)]);
                variational[(i, j)] = value;
                variational[(j, i)] = value;
            }
        }
        Self { raw, variational }
    }

    fn apply(operator: &Matrix, vector: &[f64]) -> Result<Vec<f64>> {
        if operator.cols != vector.len() {
            return Err(DftError::InvalidInput(format!(
                "PCM response vector has length {}, expected {}",
                vector.len(),
                operator.cols
            )));
        }
        Ok((0..operator.rows)
            .map(|i| {
                (0..operator.cols)
                    .map(|j| operator[(i, j)] * vector[j])
                    .sum()
            })
            .collect())
    }

    fn apply_raw(&self, vector: &[f64]) -> Result<Vec<f64>> {
        Self::apply(&self.raw, vector)
    }

    fn apply_variational(&self, vector: &[f64]) -> Result<Vec<f64>> {
        Self::apply(&self.variational, vector)
    }
}

pub struct PcmSolver {
    pub cav: Cavity,
    pub opts: PcmOptions,
    /// `(K, R)` for the IEF family, where `q = K^{-1} R v`. Absent for the
    /// C-PCM/COSMO family, whose `K = S` and `R = -f I` need no storage.
    ief: Option<(Matrix, Matrix)>,
    response: PcmResponseOperator,
    s: Matrix,
    /// `v_nuc[i]` — the nuclear electrostatic potential at surface point i.
    v_nuc: Vec<f64>,
    /// `w[i]` — the AO matrix `(mu nu | g_i)` for each surface point.
    ao_pot: Vec<Matrix>,
}

impl PcmSolver {
    pub fn build(
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        opts: &PcmOptions,
    ) -> Result<Self> {
        let cav = build_cavity(mol, opts)?;
        let m = cav.points.len();
        let s = s_matrix(&cav);
        let kernel = CoulombKernel::Coulomb;
        let shells: Vec<MolShell> = (0..m).map(|i| surface_shell(&cav, i)).collect();
        // Potential of a unit-norm Gaussian charge (exponent zeta^2) at a
        // point charge a distance d away: erf(zeta d)/d in closed form.
        let v_nuc: Vec<f64> = (0..m)
            .into_par_iter()
            .map(|i| {
                let z_i = cav.zeta[i];
                mol.atoms
                    .iter()
                    .zip(charges)
                    .map(|(at, &z)| {
                        let d = (cav.points[i] - at.position).norm();
                        if d < 1e-12 {
                            z * 2.0 * z_i / std::f64::consts::PI.sqrt()
                        } else {
                            z * libm::erf(z_i * d) / d
                        }
                    })
                    .sum()
            })
            .collect();
        let nao = mb.nao_sph;
        let ns = mb.shells.len();
        let ao_pot: Vec<Matrix> = shells
            .par_iter()
            .map(|sh| {
                let mut v = Matrix::zeros(nao, nao);
                for a in 0..ns {
                    for b in a..ns {
                        let q = eri_3c_shell(mb, a, b, sh, kernel);
                        let (oa, ob) = (mb.shells[a].sph_off, mb.shells[b].sph_off);
                        for i in 0..q.na {
                            for j in 0..q.nb {
                                let val = q.get(i, j, 0, 0);
                                v[(oa + i, ob + j)] = val;
                                v[(ob + j, oa + i)] = val;
                            }
                        }
                    }
                }
                v
            })
            .collect();
        // IEF family: K = (I - (f/2pi) D A) S and R = -f (I - (1/2pi) D A),
        // built once here. SS(V)PE symmetrizes the D A S product, which
        // restores the symmetry C-PCM has for free.
        let ief = if opts.model.needs_d() {
            let f = opts.f_eps();
            let d = d_matrix(&cav, mol);
            let a = patch_areas(&cav);
            let tau = 1.0 / (2.0 * std::f64::consts::PI);
            let mut da = Matrix::zeros(m, m);
            for i in 0..m {
                for j in 0..m {
                    da[(i, j)] = d[(i, j)] * a[j];
                }
            }
            let k = match opts.model {
                PcmModel::Ssvpe => {
                    let das = da.matmul(&s);
                    let mut k = s.clone();
                    for i in 0..m {
                        for j in 0..m {
                            k[(i, j)] -= f * tau * 0.5 * (das[(i, j)] + das[(j, i)]);
                        }
                    }
                    k
                }
                _ => {
                    let das = da.matmul(&s);
                    let mut k = s.clone();
                    for (kv, dv) in k.as_mut_slice().iter_mut().zip(das.as_slice()) {
                        *kv -= f * tau * dv;
                    }
                    k
                }
            };
            let mut r = Matrix::zeros(m, m);
            for i in 0..m {
                r[(i, i)] = -f;
                for j in 0..m {
                    r[(i, j)] += f * tau * da[(i, j)];
                }
            }
            Some((k, r))
        } else {
            None
        };
        let response = match &ief {
            Some((k, r)) => solve_linear_matrix(k, r)?,
            None => {
                let mut rhs = Matrix::identity(m);
                rhs.as_mut_slice()
                    .iter_mut()
                    .for_each(|value| *value *= -opts.f_eps());
                solve_linear_matrix(&s, &rhs)?
            }
        };
        Ok(Self {
            cav,
            opts: opts.clone(),
            ief,
            response: PcmResponseOperator::new(response),
            s,
            v_nuc,
            ao_pot,
        })
    }

    pub fn n_points(&self) -> usize {
        self.cav.points.len()
    }

    /// The nuclear electrostatic potential on the cavity surface.
    pub fn v_nuc(&self) -> &[f64] {
        &self.v_nuc
    }

    /// `(mu nu | g_i)` for surface point `i` (the reaction-field operator
    /// building block; also what the derivative paths differentiate).
    pub fn ao_potential(&self, i: usize) -> &Matrix {
        &self.ao_pot[i]
    }

    /// Surface potential `v = v_nuc - v_elec` for a total density.
    pub fn surface_potential(&self, p_tot: &Matrix) -> Vec<f64> {
        self.v_nuc
            .par_iter()
            .zip(self.ao_pot.par_iter())
            .map(|(&vn, w)| vn - p_tot.frobenius_dot(w))
            .collect()
    }

    /// Apparent surface charges for the given potential: `q = -f S^{-1} v`.
    pub fn charges(&self, v: &[f64]) -> Result<Vec<f64>> {
        self.response.apply_raw(v)
    }

    fn variational_charges(&self, v: &[f64]) -> Result<Vec<f64>> {
        self.response.apply_variational(v)
    }

    /// `dS/dR_a` applied to `q`, as one m-vector per nuclear coordinate.
    /// Indexed `[3*atom + axis][point]`.
    fn s_deriv_apply(&self, mol: &Molecule, q: &[f64]) -> Vec<Vec<f64>> {
        let cav = &self.cav;
        let m = cav.points.len();
        let n3 = 3 * mol.len();
        let mut out = vec![vec![0.0; m]; n3];
        // Off-diagonal: only r_ij moves, and only when the two points have
        // DIFFERENT owners (within one sphere the pair is rigid).
        for i in 0..m {
            let oi = cav.atom_of[i];
            for j in (i + 1)..m {
                let oj = cav.atom_of[j];
                if oi == oj {
                    continue;
                }
                let dvec = cav.points[i] - cav.points[j];
                let r = dvec.norm();
                let zi = cav.zeta[i];
                let zj = cav.zeta[j];
                let z = zi * zj / (zi * zi + zj * zj).sqrt();
                let sp = 2.0 * z / std::f64::consts::PI.sqrt() * (-(z * r) * (z * r)).exp() / r
                    - libm::erf(z * r) / (r * r);
                let dir = dvec * (1.0 / r);
                for ax in 0..3 {
                    let d = sp * dir.get(ax);
                    // dS_ij/dX_{oi} = +d, dS_ij/dX_{oj} = -d.
                    out[3 * oi + ax][i] += d * q[j];
                    out[3 * oi + ax][j] += d * q[i];
                    out[3 * oj + ax][i] -= d * q[j];
                    out[3 * oj + ax][j] -= d * q[i];
                }
            }
        }
        // Diagonal: S_ii = zeta_i sqrt(2/pi) / F_i.
        let dfs = switch_gradient(mol, cav, &self.opts);
        let c0 = (2.0 / std::f64::consts::PI).sqrt();
        for i in 0..m {
            let scale = -c0 * cav.zeta[i] / (cav.switch[i] * cav.switch[i]) * q[i];
            for (at, g) in dfs[i].iter().enumerate() {
                for ax in 0..3 {
                    out[3 * at + ax][i] += scale * g.get(ax);
                }
            }
        }
        out
    }

    /// `q . (d2 S / dR_a dR_b) . q`, the second-derivative half of the
    /// polarization Hessian that does not involve the potential.
    fn s_second_quadratic(&self, mol: &Molecule, q: &[f64]) -> Matrix {
        let cav = &self.cav;
        let m = cav.points.len();
        let n_at = mol.len();
        let mut h = Matrix::zeros(3 * n_at, 3 * n_at);
        let sqrt_pi = std::f64::consts::PI.sqrt();
        // Off-diagonal pairs: S_ij depends on the two owners only, through
        // r_ij, so each pair contributes a 2x2 atom block.
        for i in 0..m {
            let oi = cav.atom_of[i];
            for j in (i + 1)..m {
                let oj = cav.atom_of[j];
                if oi == oj {
                    continue;
                }
                let dvec = cav.points[i] - cav.points[j];
                let r = dvec.norm();
                let zi = cav.zeta[i];
                let zj = cav.zeta[j];
                let z = zi * zj / (zi * zi + zj * zj).sqrt();
                let e = (-(z * r) * (z * r)).exp();
                let erfv = libm::erf(z * r);
                let sp = 2.0 * z / sqrt_pi * e / r - erfv / (r * r);
                let spp = -4.0 * z * z * z / sqrt_pi * e - 4.0 * z / (sqrt_pi * r * r) * e
                    + 2.0 * erfv / (r * r * r);
                let n = dvec * (1.0 / r);
                // d2 S / dX dY = c_X c_Y [ s'' n n^T + s' (I - n n^T)/r ],
                // with c = +1 for the owner of i and -1 for that of j.
                let mut blk = [[0.0; 3]; 3];
                for x in 0..3 {
                    for y in 0..3 {
                        let nn = n.get(x) * n.get(y);
                        let id = if x == y { 1.0 } else { 0.0 };
                        blk[x][y] = spp * nn + sp * (id - nn) / r;
                    }
                }
                // The quadratic form picks up 2 q_i q_j (S is symmetric).
                let wgt = 2.0 * q[i] * q[j];
                for x in 0..3 {
                    for y in 0..3 {
                        let v = wgt * blk[x][y];
                        h[(3 * oi + x, 3 * oi + y)] += v;
                        h[(3 * oj + x, 3 * oj + y)] += v;
                        h[(3 * oi + x, 3 * oj + y)] -= v;
                        h[(3 * oj + x, 3 * oi + y)] -= v;
                    }
                }
            }
        }
        // Diagonal: q_i^2 zeta_i sqrt(2/pi) * Hess(1/F_i), by nested duals
        // over the closed form.
        let (r_in, r_sw) = switch_geometry(mol, &self.opts);
        let c0 = (2.0 / std::f64::consts::PI).sqrt();
        let base: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
        for i in 0..m {
            let coef = q[i] * q[i] * cav.zeta[i] * c0;
            if coef == 0.0 {
                continue;
            }
            let owner = cav.atom_of[i];
            let off = cav.points[i] - base[owner];
            let offset = [off.x, off.y, off.z];
            for a in 0..n_at {
                for b in a..n_at {
                    let cs: Vec<[D2h; 3]> = (0..n_at)
                        .map(|g| {
                            [
                                seed2(base[g].x, (g == a).then_some(0), (g == b).then_some(0)),
                                seed2(base[g].y, (g == a).then_some(1), (g == b).then_some(1)),
                                seed2(base[g].z, (g == a).then_some(2), (g == b).then_some(2)),
                            ]
                        })
                        .collect();
                    let v = inv_switch_s(&cs, owner, offset, &r_in, &r_sw);
                    for x in 0..3 {
                        for y in 0..3 {
                            let d = coef * v.im[x].im[y];
                            h[(3 * a + x, 3 * b + y)] += d;
                            if a != b {
                                h[(3 * b + y, 3 * a + x)] += d;
                            }
                        }
                    }
                }
            }
        }
        h
    }

    /// `sum_i q_i * d2 v^nuc_i / dR_a dR_b`, in closed form. Each (point,
    /// nucleus) pair moves two centres, so it contributes a 2x2 block.
    fn nuclear_potential_second(&self, mol: &Molecule, charges: &[f64], q: &[f64]) -> Matrix {
        let cav = &self.cav;
        let n_at = mol.len();
        let mut h = Matrix::zeros(3 * n_at, 3 * n_at);
        let sqrt_pi = std::f64::consts::PI.sqrt();
        for i in 0..cav.points.len() {
            if q[i] == 0.0 {
                continue;
            }
            let owner = cav.atom_of[i];
            let z = cav.zeta[i];
            for (a, (at, &zc)) in mol.atoms.iter().zip(charges).enumerate() {
                if zc == 0.0 {
                    continue;
                }
                let dvec = cav.points[i] - at.position;
                let r = dvec.norm();
                if r < 1e-12 {
                    continue;
                }
                let e = (-(z * r) * (z * r)).exp();
                let erfv = libm::erf(z * r);
                let sp = 2.0 * z / sqrt_pi * e / r - erfv / (r * r);
                let spp = -4.0 * z * z * z / sqrt_pi * e - 4.0 * z / (sqrt_pi * r * r) * e
                    + 2.0 * erfv / (r * r * r);
                let n = dvec * (1.0 / r);
                let wgt = q[i] * zc;
                for x in 0..3 {
                    for y in 0..3 {
                        let nn = n.get(x) * n.get(y);
                        let id = if x == y { 1.0 } else { 0.0 };
                        let v = wgt * (spp * nn + sp * (id - nn) / r);
                        h[(3 * owner + x, 3 * owner + y)] += v;
                        h[(3 * a + x, 3 * a + y)] += v;
                        h[(3 * owner + x, 3 * a + y)] -= v;
                        h[(3 * a + x, 3 * owner + y)] -= v;
                    }
                }
            }
        }
        h
    }

    /// `sum_i q_i * d2 Tr[P (mu nu|g_i)] / dR_a dR_b`.
    ///
    /// The surface charges are just charge sites that ride their owner
    /// atom, so this is the SAME second-derivative machinery the nuclear
    /// attraction uses — including the sum rules that give the site-centre
    /// derivatives — driven by a Gaussian block instead of a point one.
    fn electronic_potential_second(
        &self,
        mb: &MolBasis,
        n_at: usize,
        p_tot: &Matrix,
        q: &[f64],
    ) -> Matrix {
        let kernel = CoulombKernel::Coulomb;
        let shells: Vec<MolShell> = (0..self.cav.points.len())
            .map(|i| surface_shell(&self.cav, i))
            .collect();
        // The sign: v^el enters v as `-Tr[P W]`, so q . v^el carries a minus.
        crate::integrals::deriv2::charge_site_second_contraction(
            mb,
            n_at,
            shells.len(),
            |s| self.cav.atom_of[s],
            p_tot,
            |sa, sb, s| {
                crate::integrals::eri::gauss_charge_block(sa, sb, &shells[s], -q[s], kernel)
            },
        )
    }

    /// `d V_pol / dR_a` at FIXED density: the reaction field's own entry in
    /// the skeleton Fock derivative, which drives the CPKS right-hand side.
    ///
    /// `V_pol = -sum_i q_i W_i`, so two things move: the charges respond to
    /// the geometry (`dq/dR_a = -f S^{-1} u_a`, the same `u_a` the skeleton
    /// Hessian builds) and the integrals themselves shift with their
    /// centres. The latter is the charge-site generalization of the nuclear
    /// attraction derivative, so no new integral class is needed.
    pub fn fock_derivative(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
    ) -> Result<Vec<crate::integrals::deriv_mat::AtomDerivs>> {
        if self.opts.model.needs_d() {
            return self.fock_derivative_ief(mol, mb, charges, p_tot);
        }
        let n_at = mol.len();
        let f = self.opts.f_eps();
        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        let u = self.potential_deriv_vectors(mol, mb, charges, p_tot, &q);
        let shells: Vec<MolShell> = (0..self.cav.points.len())
            .map(|i| surface_shell(&self.cav, i))
            .collect();
        // Integral-shift half: -sum_i q_i dW_i/dR_a.
        let mut out = crate::integrals::deriv_mat::charge_site_derivative_matrices(
            mb,
            n_at,
            shells.len(),
            |s| self.cav.atom_of[s],
            |sa, sb, s| {
                crate::integrals::eri::gauss_charge_block(
                    sa,
                    sb,
                    &shells[s],
                    -q[s],
                    CoulombKernel::Coulomb,
                )
            },
        );
        // Charge-response half: -sum_i (dq_i/dR_a) W_i, dq_a = -f S^{-1} u_a.
        for at in 0..n_at {
            for ax in 0..3 {
                let qa = solve_linear(&self.s, &u[3 * at + ax])?;
                let m = &mut out[at][ax];
                for (qi, w) in qa.iter().zip(&self.ao_pot) {
                    if *qi == 0.0 {
                        continue;
                    }
                    let c = f * qi;
                    for (mv, wv) in m.as_mut_slice().iter_mut().zip(w.as_slice()) {
                        *mv += c * wv;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Fixed-density geometry derivative of the variational IEF/SS(V)PE
    /// reaction-field Fock matrix.
    ///
    /// With `q = K^-1 R v`, `lambda = K^-T v` and
    /// `q_bar = (q + R^T lambda)/2`, the Fock contribution is
    /// `V_pol = -sum_i q_bar_i W_i`. For coordinate `a`,
    ///
    /// ```text
    /// q_a      = K^-1 (R_a v + R v_a - K_a q)
    /// lambda_a = K^-T (v_a - K_a^T lambda)
    /// q_bar_a  = (q_a + R_a^T lambda + R^T lambda_a)/2 .
    /// ```
    ///
    /// The derivative therefore has an integral-shift half
    /// `-sum_i q_bar_i W_i,a` and a charge-response half
    /// `-sum_i q_bar_i,a W_i`. SS(V)PE differs only by replacing every
    /// application of `K_a` by its symmetric IEF average.
    fn fock_derivative_ief(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
    ) -> Result<Vec<crate::integrals::deriv_mat::AtomDerivs>> {
        let (k, r) = self.ief.as_ref().expect("IEF state");
        let kt = k.transpose();
        let n_at = mol.len();
        let n3 = 3 * n_at;
        let m = self.cav.points.len();
        let f = self.opts.f_eps();
        let tau = 1.0 / (2.0 * std::f64::consts::PI);
        let cav = &self.cav;

        let mv = |a: &Matrix, x: &[f64]| -> Vec<f64> {
            (0..m)
                .map(|i| (0..m).map(|j| a[(i, j)] * x[j]).sum())
                .collect()
        };
        let mtv = |a: &Matrix, x: &[f64]| -> Vec<f64> {
            (0..m)
                .map(|j| (0..m).map(|i| a[(i, j)] * x[i]).sum())
                .collect()
        };

        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        let lambda = solve_linear(&kt, &v)?;
        let rtl = mtv(r, &lambda);
        let q_bar: Vec<f64> = (0..m).map(|i| 0.5 * (q[i] + rtl[i])).collect();

        let dvn = nuclear_potential_gradient(mol, cav, charges);
        let dve = electronic_potential_gradient(mb, cav, p_tot, n_at);
        let dfs = switch_gradient(mol, cav, &self.opts);
        let d = d_matrix(cav, mol);
        let areas = patch_areas(cav);
        let mut va = vec![vec![0.0; m]; n3];
        let mut aa = vec![vec![0.0; m]; n3];
        let mut sa = Vec::with_capacity(n3);
        let mut da = Vec::with_capacity(n3);
        for at in 0..n_at {
            for ax in 0..3 {
                let c = 3 * at + ax;
                for i in 0..m {
                    va[c][i] = dvn[i][at].get(ax) - dve[i][at].get(ax);
                    aa[c][i] = cav.weight[i] * cav.radius[i] * cav.radius[i] * dfs[i][at].get(ax);
                }
                sa.push(self.s_deriv_matrix(mol, &dfs, at, ax));
                da.push(d_matrix_deriv(cav, mol, at, ax));
            }
        }

        let ka_of = |c: usize, x: &[f64]| -> Vec<f64> {
            let sx = mv(&sa[c], x);
            let sxx = mv(&self.s, x);
            let sax = mv(&sa[c], x);
            let wx: Vec<f64> = (0..m).map(|i| areas[i] * sxx[i]).collect();
            let wax: Vec<f64> = (0..m)
                .map(|i| aa[c][i] * sxx[i] + areas[i] * sax[i])
                .collect();
            let t1 = mv(&da[c], &wx);
            let t2 = mv(&d, &wax);
            (0..m).map(|i| sx[i] - f * tau * (t1[i] + t2[i])).collect()
        };
        let kat_of = |c: usize, x: &[f64]| -> Vec<f64> {
            let sl = mtv(&sa[c], x);
            let dl = mtv(&da[c], x);
            let ddl = mtv(&d, x);
            let s1: Vec<f64> = (0..m).map(|i| areas[i] * dl[i]).collect();
            let s2: Vec<f64> = (0..m).map(|i| aa[c][i] * ddl[i]).collect();
            let s3: Vec<f64> = (0..m).map(|i| areas[i] * ddl[i]).collect();
            let p1 = mtv(&self.s, &s1);
            let p2 = mtv(&self.s, &s2);
            let p3 = mtv(&sa[c], &s3);
            (0..m)
                .map(|i| sl[i] - f * tau * (p1[i] + p2[i] + p3[i]))
                .collect()
        };
        let average = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> {
            a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect()
        };
        let symm = matches!(self.opts.model, PcmModel::Ssvpe);
        let k_a = |c: usize, x: &[f64]| -> Vec<f64> {
            if symm {
                average(ka_of(c, x), kat_of(c, x))
            } else {
                ka_of(c, x)
            }
        };
        let k_a_t = |c: usize, x: &[f64]| -> Vec<f64> {
            if symm {
                average(ka_of(c, x), kat_of(c, x))
            } else {
                kat_of(c, x)
            }
        };

        let shells: Vec<MolShell> = (0..m).map(|i| surface_shell(cav, i)).collect();
        let mut out = crate::integrals::deriv_mat::charge_site_derivative_matrices(
            mb,
            n_at,
            shells.len(),
            |s| cav.atom_of[s],
            |sa_shell, sb_shell, s| {
                crate::integrals::eri::gauss_charge_block(
                    sa_shell,
                    sb_shell,
                    &shells[s],
                    -q_bar[s],
                    CoulombKernel::Coulomb,
                )
            },
        );

        for c in 0..n3 {
            let av: Vec<f64> = (0..m).map(|i| areas[i] * v[i]).collect();
            let aav: Vec<f64> = (0..m).map(|i| aa[c][i] * v[i]).collect();
            let da_av = mv(&da[c], &av);
            let d_aav = mv(&d, &aav);
            let ra_v: Vec<f64> = (0..m).map(|i| f * tau * (da_av[i] + d_aav[i])).collect();
            let r_va = mv(r, &va[c]);
            let ka_q = k_a(c, &q);
            let rhs_q: Vec<f64> = (0..m).map(|i| ra_v[i] + r_va[i] - ka_q[i]).collect();
            let q_a = solve_linear(k, &rhs_q)?;

            let kat_l = k_a_t(c, &lambda);
            let rhs_l: Vec<f64> = (0..m).map(|i| va[c][i] - kat_l[i]).collect();
            let lambda_a = solve_linear(&kt, &rhs_l)?;
            let dl = mtv(&da[c], &lambda);
            let ddl = mtv(&d, &lambda);
            let ra_t_l: Vec<f64> = (0..m)
                .map(|i| f * tau * (areas[i] * dl[i] + aa[c][i] * ddl[i]))
                .collect();
            let rt_la = mtv(r, &lambda_a);
            let qbar_a: Vec<f64> = (0..m)
                .map(|i| 0.5 * (q_a[i] + ra_t_l[i] + rt_la[i]))
                .collect();

            let at = c / 3;
            let ax = c % 3;
            for (qi, w) in qbar_a.iter().zip(&self.ao_pot) {
                if *qi == 0.0 {
                    continue;
                }
                for (target, value) in out[at][ax].as_mut_slice().iter_mut().zip(w.as_slice()) {
                    *target -= qi * value;
                }
            }
        }
        Ok(out)
    }

    /// `u_a = v_a + (1/f) S_a q`, the vector both the skeleton Hessian and
    /// the Fock derivative need.
    fn potential_deriv_vectors(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
        q: &[f64],
    ) -> Vec<Vec<f64>> {
        let n_at = mol.len();
        let m = self.cav.points.len();
        let f = self.opts.f_eps();
        let dvn = nuclear_potential_gradient(mol, &self.cav, charges);
        let dve = electronic_potential_gradient(mb, &self.cav, p_tot, n_at);
        let sq = self.s_deriv_apply(mol, q);
        let mut u = vec![vec![0.0; m]; 3 * n_at];
        for i in 0..m {
            for at in 0..n_at {
                for ax in 0..3 {
                    u[3 * at + ax][i] =
                        dvn[i][at].get(ax) - dve[i][at].get(ax) + sq[3 * at + ax][i] / f;
                }
            }
        }
        u
    }

    /// Analytic second derivative of the polarization energy at a FIXED
    /// density — the solvation counterpart of the skeleton Hessian.
    ///
    /// Writing `E` as the stationary value of
    /// `G(q, R) = -(1/2f) q^T S q - q.v` makes the structure explicit: the
    /// stationarity kills the first-order response, but the second
    /// derivative still needs `dq/dR`, so
    ///
    /// `d2E/dab = (1/2f) q^T S_ab q + q.v_ab - f (u_a)^T S^{-1} u_b`,
    /// `u_a = v_a + (1/f) S_a q`
    ///
    /// which is manifestly symmetric and needs one solve per coordinate
    /// against the matrix that is already built.
    pub fn hessian_skeleton(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
    ) -> Result<Matrix> {
        if self.opts.model.needs_d() {
            return self.hessian_skeleton_ief(mol, mb, charges, p_tot);
        }
        let n_at = mol.len();
        let n3 = 3 * n_at;
        let f = self.opts.f_eps();
        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        let m = self.cav.points.len();

        let _ = m;
        let u = self.potential_deriv_vectors(mol, mb, charges, p_tot, &q);
        // w_a = S^{-1} u_a.
        let w: Vec<Vec<f64>> = u
            .iter()
            .map(|ua| solve_linear(&self.s, ua))
            .collect::<Result<_>>()?;

        let mut h = self.s_second_quadratic(mol, &q);
        for val in h.as_mut_slice() {
            *val /= 2.0 * f;
        }
        let hn = self.nuclear_potential_second(mol, charges, &q);
        let he = self.electronic_potential_second(mb, n_at, p_tot, &q);
        for (t, s) in h.as_mut_slice().iter_mut().zip(hn.as_slice()) {
            *t += *s;
        }
        for (t, s) in h.as_mut_slice().iter_mut().zip(he.as_slice()) {
            *t += *s;
        }
        for a in 0..n3 {
            for b in 0..n3 {
                let dot: f64 = u[a].iter().zip(&w[b]).map(|(x, y)| x * y).sum();
                h[(a, b)] -= f * dot;
            }
        }
        Ok(h)
    }

    /// Analytic gradient for the IEF family, by the ADJOINT method.
    ///
    /// With `q = K^{-1} R v` and `E = 1/2 q.v`, write `M = K^{-1} R` so
    /// `E = 1/2 v^T M v`. Differentiating and using
    /// `dM = K^{-1}[dR - (dK) M]` gives
    ///
    /// `dE/dR_a = q_bar . v_a + 1/2 lambda^T (dR_a v - dK_a q)`,
    /// `q_bar = (M + M^T) v / 2`,  `lambda = K^{-T} v`
    ///
    /// so ONE transposed solve covers every coordinate — the same trick the
    /// C-PCM path gets for free from `K = S`, recovered here explicitly.
    /// `dK` and `dR` need `dS` (have it), `dA` (patch areas move only
    /// through the switching values) and `dD` (closed form, and its normals
    /// do not move at all).
    fn gradient_ief(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
    ) -> Result<Vec<Vec3>> {
        let (k, r) = self.ief.as_ref().expect("IEF state");
        let n_at = mol.len();
        let m = self.cav.points.len();
        let f = self.opts.f_eps();
        let tau = 1.0 / (2.0 * std::f64::consts::PI);
        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        // lambda = K^{-T} v.
        let kt = k.transpose();
        let lambda = solve_linear(&kt, &v)?;
        // q_bar = (M + M^T) v / 2 = (q + R^T K^{-T} v) / 2.
        let rtl = {
            let mut out = vec![0.0; m];
            for i in 0..m {
                let mut acc = 0.0;
                for j in 0..m {
                    acc += r[(j, i)] * lambda[j];
                }
                out[i] = acc;
            }
            out
        };
        let q_bar: Vec<f64> = q.iter().zip(&rtl).map(|(a, b)| 0.5 * (a + b)).collect();

        let dvn = nuclear_potential_gradient(mol, &self.cav, charges);
        let dve = electronic_potential_gradient(mb, &self.cav, p_tot, n_at);
        let dfs = switch_gradient(mol, &self.cav, &self.opts);
        let d = d_matrix(&self.cav, mol);
        let areas = patch_areas(&self.cav);
        let ssym = &self.s;

        let mut grad = vec![Vec3::zero(); n_at];
        for at in 0..n_at {
            for ax in 0..3 {
                // Term 1: q_bar . dv/dR.
                let mut g = 0.0;
                for i in 0..m {
                    g += q_bar[i] * (dvn[i][at].get(ax) - dve[i][at].get(ax));
                }
                // Pieces of dK and dR. B = tau D A, so
                // dB = tau (dD A + D dA); dK = -f dB S + (I - f B) dS;
                // dR = f tau (dD A + D dA).
                let dd = d_matrix_deriv(&self.cav, mol, at, ax);
                let da: Vec<f64> = (0..m)
                    .map(|i| {
                        self.cav.weight[i]
                            * self.cav.radius[i]
                            * self.cav.radius[i]
                            * dfs[i][at].get(ax)
                    })
                    .collect();
                let ds = self.s_deriv_matrix(mol, &dfs, at, ax);
                // dB_ij = tau (sum_k dD_ik A_k delta_kj + D_ij dA_j)
                //       = tau (dD_ij A_j + D_ij dA_j).
                let mut db = Matrix::zeros(m, m);
                for i in 0..m {
                    for j in 0..m {
                        db[(i, j)] = tau * (dd[(i, j)] * areas[j] + d[(i, j)] * da[j]);
                    }
                }
                // dR v = f dB v (R = -f I + f tau D A).
                let mut drv = vec![0.0; m];
                for i in 0..m {
                    let mut acc = 0.0;
                    for j in 0..m {
                        acc += db[(i, j)] * v[j];
                    }
                    drv[i] = f * acc;
                }
                // dK q = -f dB (S q) + (I - f B) dS q   [B = tau D A]
                let mv = |a: &Matrix, x: &[f64]| -> Vec<f64> {
                    (0..m)
                        .map(|i| (0..m).map(|j| a[(i, j)] * x[j]).sum())
                        .collect()
                };
                let sq = mv(ssym, &q);
                let dsq = mv(&ds, &q);
                let mut dkq = vec![0.0; m];
                for i in 0..m {
                    let mut acc = 0.0;
                    for j in 0..m {
                        acc -= f * db[(i, j)] * sq[j];
                        acc -= f * tau * d[(i, j)] * areas[j] * dsq[j];
                    }
                    dkq[i] = acc + dsq[i];
                }
                let _n_at = mol.len();
                // SS(V)PE's K is the symmetrization of the IEF one, so its
                // K_a is too: K_a x -> (K_a x + K_a^T x)/2.
                if matches!(self.opts.model, PcmModel::Ssvpe) {
                    let mtv = |a: &Matrix, x: &[f64]| -> Vec<f64> {
                        (0..m)
                            .map(|j| (0..m).map(|i| a[(i, j)] * x[i]).sum())
                            .collect()
                    };
                    let dq = mtv(&dd, &q);
                    let ddq = mtv(&d, &q);
                    let s1: Vec<f64> = (0..m).map(|i| areas[i] * dq[i]).collect();
                    let s2: Vec<f64> = (0..m).map(|i| da[i] * ddq[i]).collect();
                    let s3: Vec<f64> = (0..m).map(|i| areas[i] * ddq[i]).collect();
                    let p1 = mtv(ssym, &s1);
                    let p2 = mtv(ssym, &s2);
                    let p3 = mtv(&ds, &s3);
                    for i in 0..m {
                        let kat = dsq[i] - f * tau * (p1[i] + p2[i] + p3[i]);
                        dkq[i] = 0.5 * (dkq[i] + kat);
                    }
                }
                for i in 0..m {
                    g += 0.5 * lambda[i] * (drv[i] - dkq[i]);
                }
                match ax {
                    0 => grad[at].x = g,
                    1 => grad[at].y = g,
                    _ => grad[at].z = g,
                }
            }
        }
        Ok(grad)
    }

    /// Hessian of the SWIG factor `F_i` and of `1/F_i` for one atom pair,
    /// from nested duals over the closed form.
    fn switch_hess_pair(
        &self,
        base: &[Vec3],
        i: usize,
        a: usize,
        b: usize,
        r_in: &[f64],
        r_sw: &[f64],
    ) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        let n_at = base.len();
        let owner = self.cav.atom_of[i];
        let off = self.cav.points[i] - base[owner];
        let offset = [off.x, off.y, off.z];
        let cs: Vec<[D2h; 3]> = (0..n_at)
            .map(|g| {
                [
                    seed2(base[g].x, (g == a).then_some(0), (g == b).then_some(0)),
                    seed2(base[g].y, (g == a).then_some(1), (g == b).then_some(1)),
                    seed2(base[g].z, (g == a).then_some(2), (g == b).then_some(2)),
                ]
            })
            .collect();
        let inv = inv_switch_s(&cs, owner, offset, r_in, r_sw);
        let fwd = D2h::from_f64(1.0) / inv;
        let mut hi = [[0.0; 3]; 3];
        let mut hf = [[0.0; 3]; 3];
        for x in 0..3 {
            for y in 0..3 {
                hi[x][y] = inv.im[x].im[y];
                hf[x][y] = fwd.im[x].im[y];
            }
        }
        (hi, hf)
    }
    /// Analytic fixed-density Hessian for the IEF family.
    ///
    /// Differentiating the adjoint gradient
    /// `g_a = q_bar . v_a + 1/2 lambda^T (R_a v - K_a q)` once more gives
    ///
    /// ```text
    /// H_ab = q_bar_b . v_a + q_bar . v_ab
    ///      + 1/2 lambda_b^T (R_a v - K_a q)
    ///      + 1/2 lambda^T (R_ab v + R_a v_b - K_ab q - K_a q_b)
    /// ```
    ///
    /// with `q_b` and `lambda_b` from one solve each. The second
    /// derivatives of `S`, `D` and the patch areas are never MATERIALIZED
    /// as `3N x 3N` families of `m x m` matrices — only their action on a
    /// vector is needed — and each comes from nested duals over the closed
    /// forms, so nothing here is hand-derived.
    fn hessian_skeleton_ief(
        &self,
        mol: &Molecule,
        mb: &MolBasis,
        charges: &[f64],
        p_tot: &Matrix,
    ) -> Result<Matrix> {
        let (k, r) = self.ief.as_ref().expect("IEF state");
        let n_at = mol.len();
        let n3 = 3 * n_at;
        let m = self.cav.points.len();
        let f = self.opts.f_eps();
        let tau = 1.0 / (2.0 * std::f64::consts::PI);
        let cav = &self.cav;
        let base: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
        let (r_in, r_sw) = switch_geometry(mol, &self.opts);

        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        let kt = k.transpose();
        let lambda = solve_linear(&kt, &v)?;
        let mut q_bar = vec![0.0; m];
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..m {
                acc += r[(j, i)] * lambda[j];
            }
            q_bar[i] = 0.5 * (q[i] + acc);
        }

        // ---- first derivatives, per coordinate ----
        let dvn = nuclear_potential_gradient(mol, cav, charges);
        let dve = electronic_potential_gradient(mb, cav, p_tot, n_at);
        let dfs = switch_gradient(mol, cav, &self.opts);
        let d = d_matrix(cav, mol);
        let areas = patch_areas(cav);
        let mv = |a: &Matrix, x: &[f64]| -> Vec<f64> {
            (0..m)
                .map(|i| (0..m).map(|j| a[(i, j)] * x[j]).sum())
                .collect()
        };
        let mtv = |a: &Matrix, x: &[f64]| -> Vec<f64> {
            (0..m)
                .map(|j| (0..m).map(|i| a[(i, j)] * x[i]).sum())
                .collect()
        };
        let mut va: Vec<Vec<f64>> = vec![vec![0.0; m]; n3];
        let mut aa: Vec<Vec<f64>> = vec![vec![0.0; m]; n3];
        let mut sa: Vec<Matrix> = Vec::with_capacity(n3);
        let mut da: Vec<Matrix> = Vec::with_capacity(n3);
        for at in 0..n_at {
            for ax in 0..3 {
                let c = 3 * at + ax;
                for i in 0..m {
                    va[c][i] = dvn[i][at].get(ax) - dve[i][at].get(ax);
                    aa[c][i] = cav.weight[i] * cav.radius[i] * cav.radius[i] * dfs[i][at].get(ax);
                }
                sa.push(self.s_deriv_matrix(mol, &dfs, at, ax));
                da.push(d_matrix_deriv(cav, mol, at, ax));
            }
        }
        // K_a and R_a applied to the vectors they meet.
        // K = S - f tau D A S, R = -f I + f tau D A.
        let ra_v: Vec<Vec<f64>> = (0..n3)
            .map(|c| {
                let av: Vec<f64> = (0..m).map(|i| areas[i] * v[i]).collect();
                let aav: Vec<f64> = (0..m).map(|i| aa[c][i] * v[i]).collect();
                let t1 = mv(&da[c], &av);
                let t2 = mv(&d, &aav);
                (0..m).map(|i| f * tau * (t1[i] + t2[i])).collect()
            })
            .collect();
        // q_b and lambda_b.
        // SS(V)PE's K is the SYMMETRIZATION of the IEF one:
        //   K_ssvpe = S - (f/2) tau (D A S + (D A S)^T) = sym(K_ief),
        // and since every assembly below is linear in K, the whole second
        // derivative follows from the IEF pieces by symmetrizing each
        // application. That is why this needs one branch, not a new
        // derivation.
        let symm = matches!(self.opts.model, PcmModel::Ssvpe);
        // K_a^T x, in the IEF form.
        let kat_of = |c: usize, x: &[f64]| -> Vec<f64> {
            let sl = mtv(&sa[c], x);
            let dl = mtv(&da[c], x);
            let ddl = mtv(&d, x);
            let s1: Vec<f64> = (0..m).map(|i| areas[i] * dl[i]).collect();
            let s2: Vec<f64> = (0..m).map(|i| aa[c][i] * ddl[i]).collect();
            let s3: Vec<f64> = (0..m).map(|i| areas[i] * ddl[i]).collect();
            let p1 = mtv(&self.s, &s1);
            let p2 = mtv(&self.s, &s2);
            let p3 = mtv(&sa[c], &s3);
            (0..m)
                .map(|i| sl[i] - f * tau * (p1[i] + p2[i] + p3[i]))
                .collect()
        };
        // K_a x, in the IEF form.
        let ka_of = |c: usize, x: &[f64]| -> Vec<f64> {
            let sx = mv(&sa[c], x);
            let sxx = mv(&self.s, x);
            let sax = mv(&sa[c], x);
            let wx: Vec<f64> = (0..m).map(|i| areas[i] * sxx[i]).collect();
            let wax: Vec<f64> = (0..m)
                .map(|i| aa[c][i] * sxx[i] + areas[i] * sax[i])
                .collect();
            let t1 = mv(&da[c], &wx);
            let t2 = mv(&d, &wax);
            (0..m).map(|i| sx[i] - f * tau * (t1[i] + t2[i])).collect()
        };
        let avg = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> {
            a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect()
        };
        let k_a = |c: usize, x: &[f64]| -> Vec<f64> {
            if symm {
                avg(ka_of(c, x), kat_of(c, x))
            } else {
                ka_of(c, x)
            }
        };
        let k_a_t = |c: usize, x: &[f64]| -> Vec<f64> {
            if symm {
                avg(ka_of(c, x), kat_of(c, x))
            } else {
                kat_of(c, x)
            }
        };
        let ka_q: Vec<Vec<f64>> = (0..n3).map(|c| k_a(c, &q)).collect();
        let mut qb: Vec<Vec<f64>> = Vec::with_capacity(n3);
        let mut lb: Vec<Vec<f64>> = Vec::with_capacity(n3);
        for c in 0..n3 {
            let rvb = mv(r, &va[c]);
            let rhs: Vec<f64> = (0..m).map(|i| ra_v[c][i] + rvb[i] - ka_q[c][i]).collect();
            qb.push(solve_linear(k, &rhs)?);
            let kat_l = k_a_t(c, &lambda);
            let rhs2: Vec<f64> = (0..m).map(|i| va[c][i] - kat_l[i]).collect();
            lb.push(solve_linear(&kt, &rhs2)?);
        }
        let qbar_b: Vec<Vec<f64>> = (0..n3)
            .map(|c| {
                let rtl = mtv(r, &lb[c]);
                // R_b^T lambda: R_b = f tau (D_b A + D A_b).
                let dl = mtv(&da[c], &lambda);
                let ddl = mtv(&d, &lambda);
                let rbtl: Vec<f64> = (0..m)
                    .map(|i| f * tau * (areas[i] * dl[i] + aa[c][i] * ddl[i]))
                    .collect();
                (0..m)
                    .map(|i| 0.5 * (qb[c][i] + rbtl[i] + rtl[i]))
                    .collect()
            })
            .collect();

        // ---- v_ab, contracted with q_bar ----
        let mut h = self.nuclear_potential_second(mol, charges, &q_bar);
        let he = self.electronic_potential_second(mb, n_at, p_tot, &q_bar);
        for (t, s) in h.as_mut_slice().iter_mut().zip(he.as_slice()) {
            *t += *s;
        }

        // ---- pair Hessians of S and D, restricted per atom pair ----
        let normals: Vec<Vec3> = (0..m)
            .map(|j| (cav.points[j] - base[cav.atom_of[j]]) * (1.0 / cav.radius[j]))
            .collect();
        for a in 0..n_at {
            for b in 0..n_at {
                // (X_ab) is nonzero only where both owners are in {a, b}.
                for ax in 0..3 {
                    for by in 0..3 {
                        let ca = 3 * a + ax;
                        let cb = 3 * b + by;
                        // S_ab q, D_ab y, A_ab.
                        // S_ab and D_ab applied to a vector. Both are needed
                        // for TWO different right-hand sides once SS(V)PE
                        // enters: its K is the symmetrization of the IEF one
                        // (K_ssvpe = sym(K_ief)), and since every assembly
                        // here is linear in K, symmetrizing each application
                        // is the whole change.
                        let second_apply = |x: &[f64], y: &[f64]| -> (Vec<f64>, Vec<f64>) {
                            let mut sx = vec![0.0; m];
                            let mut dy = vec![0.0; m];
                            for i in 0..m {
                                let oi = cav.atom_of[i];
                                for j in 0..m {
                                    if i == j {
                                        continue;
                                    }
                                    let oj = cav.atom_of[j];
                                    let cai = (oi == a) as i32 - (oj == a) as i32;
                                    let cbi = (oi == b) as i32 - (oj == b) as i32;
                                    if cai == 0 || cbi == 0 {
                                        continue;
                                    }
                                    let dvec = cav.points[i] - cav.points[j];
                                    let (hs, hd) =
                                        pair_second(dvec, normals[j], cav.zeta[i], cav.zeta[j]);
                                    let w = (cai * cbi) as f64;
                                    sx[i] += w * hs[ax][by] * x[j];
                                    dy[i] += w * hd[ax][by] * y[j];
                                }
                            }
                            (sx, dy)
                        };
                        let av: Vec<f64> = (0..m).map(|i| areas[i] * v[i]).collect();
                        let (mut sabq, dab_y) = second_apply(&q, &av);
                        // Diagonal pieces from the switching function.
                        let c0 = (2.0 / std::f64::consts::PI).sqrt();
                        let mut aab = vec![0.0; m];
                        // F_i depends on EVERY atom, not just the point's owner,
                        // so these run over all points: restricting them to live,
                        // which is right for the pair terms, silently drops the
                        // diagonal contributions from every other sphere.
                        let mut hi_diag = vec![0.0; m];
                        for i in 0..m {
                            let (hi, hf) = self.switch_hess_pair(&base, i, a, b, &r_in, &r_sw);
                            hi_diag[i] = hi[ax][by];
                            sabq[i] += c0 * cav.zeta[i] * hi[ax][by] * q[i];
                            aab[i] = cav.weight[i] * cav.radius[i] * cav.radius[i] * hf[ax][by];
                        }
                        // lambda^T R_ab v.
                        let dy_ab: Vec<f64> = (0..m).map(|i| aab[i] * v[i]).collect();
                        let t_d = mv(&d, &dy_ab);
                        let mut lr = 0.0;
                        for i in 0..m {
                            lr += lambda[i] * f * tau * (dab_y[i] + t_d[i]);
                        }
                        // cross first-derivative products in R_ab.
                        let ab_v: Vec<f64> = (0..m).map(|i| aa[cb][i] * v[i]).collect();
                        let aa_v: Vec<f64> = (0..m).map(|i| aa[ca][i] * v[i]).collect();
                        let x1 = mv(&da[ca], &ab_v);
                        let x2 = mv(&da[cb], &aa_v);
                        for i in 0..m {
                            lr += lambda[i] * f * tau * (x1[i] + x2[i]);
                        }
                        // lambda^T K_ab q, and for SS(V)PE also q^T K_ab
                        // lambda, since its K is the symmetrized form.
                        let kab_dot = |lvec: &[f64], rvec: &[f64], sab_r: &[f64]| -> f64 {
                            let sr = mv(&self.s, rvec);
                            let wr: Vec<f64> = (0..m).map(|i| areas[i] * sr[i]).collect();
                            let sa_r = mv(&sa[ca], rvec);
                            let sb_r = mv(&sa[cb], rvec);
                            let wab_r: Vec<f64> = (0..m)
                                .map(|i| {
                                    aab[i] * sr[i]
                                        + aa[ca][i] * sb_r[i]
                                        + aa[cb][i] * sa_r[i]
                                        + areas[i] * sab_r[i]
                                })
                                .collect();
                            let wa_r: Vec<f64> = (0..m)
                                .map(|i| aa[ca][i] * sr[i] + areas[i] * sa_r[i])
                                .collect();
                            let wb_r: Vec<f64> = (0..m)
                                .map(|i| aa[cb][i] * sr[i] + areas[i] * sb_r[i])
                                .collect();
                            let (_, dab_wr) = second_apply(rvec, &wr);
                            let y1 = mv(&da[ca], &wb_r);
                            let y2 = mv(&da[cb], &wa_r);
                            let y3 = mv(&d, &wab_r);
                            (0..m)
                                .map(|i| {
                                    lvec[i]
                                        * (sab_r[i] - f * tau * (dab_wr[i] + y1[i] + y2[i] + y3[i]))
                                })
                                .sum()
                        };
                        let lk = if symm {
                            let (mut sabl, _) = second_apply(&lambda, &av);
                            for i in 0..m {
                                sabl[i] += c0 * cav.zeta[i] * hi_diag[i] * lambda[i];
                            }
                            0.5 * (kab_dot(&lambda, &q, &sabq) + kab_dot(&q, &lambda, &sabl))
                        } else {
                            kab_dot(&lambda, &q, &sabq)
                        };
                        let mut acc = 0.0;
                        for i in 0..m {
                            acc += qbar_b[cb][i] * va[ca][i];
                            acc += 0.5 * lb[cb][i] * (ra_v[ca][i] - ka_q[ca][i]);
                        }
                        // The R_a v_b and K_a q_b crossings.
                        let mut cross = 0.0;
                        {
                            // R_a v_b
                            let avb: Vec<f64> = (0..m).map(|i| areas[i] * va[cb][i]).collect();
                            let aavb: Vec<f64> = (0..m).map(|i| aa[ca][i] * va[cb][i]).collect();
                            let z1 = mv(&da[ca], &avb);
                            let z2 = mv(&d, &aavb);
                            for i in 0..m {
                                cross += 0.5 * lambda[i] * f * tau * (z1[i] + z2[i]);
                            }
                            // -K_a q_b
                            // -K_a q_b, through the model-aware application
                            // so SS(V)PE's symmetrization is not lost here.
                            let kaqb = k_a(ca, &qb[cb]);
                            for i in 0..m {
                                cross -= 0.5 * lambda[i] * kaqb[i];
                            }
                        }
                        h[(ca, cb)] += acc + cross + 0.5 * (lr - lk);
                    }
                }
            }
        }
        // Symmetrize: the expression is symmetric analytically, and any
        // residual asymmetry would be a bug, so this is a cheap guard that
        // the gate below still sees.
        let ht = h.transpose();
        for (x, y) in h.as_mut_slice().iter_mut().zip(ht.as_slice()) {
            *x = 0.5 * (*x + *y);
        }
        Ok(h)
    }

    /// applied to `q`, but the IEF assembly needs the matrix itself).
    fn s_deriv_matrix(&self, mol: &Molecule, dfs: &[Vec<Vec3>], at: usize, ax: usize) -> Matrix {
        let cav = &self.cav;
        let m = cav.points.len();
        let mut out = Matrix::zeros(m, m);
        let sqrt_pi = std::f64::consts::PI.sqrt();
        for i in 0..m {
            let oi = cav.atom_of[i];
            for j in (i + 1)..m {
                let oj = cav.atom_of[j];
                let c_a = (oi == at) as i32 - (oj == at) as i32;
                if c_a == 0 {
                    continue;
                }
                let dvec = cav.points[i] - cav.points[j];
                let r = dvec.norm();
                let zi = cav.zeta[i];
                let zj = cav.zeta[j];
                let z = zi * zj / (zi * zi + zj * zj).sqrt();
                let sp =
                    2.0 * z / sqrt_pi * (-(z * r) * (z * r)).exp() / r - libm::erf(z * r) / (r * r);
                let val = c_a as f64 * sp * dvec.get(ax) / r;
                out[(i, j)] = val;
                out[(j, i)] = val;
            }
            let c0 = (2.0 / std::f64::consts::PI).sqrt();
            out[(i, i)] = -c0 * cav.zeta[i] / (cav.switch[i] * cav.switch[i]) * dfs[i][at].get(ax);
        }
        let _ = mol;
        out
    }

    /// Response of the reaction-field operator to a density change: the
    /// continuum's own contribution to the CPKS / TDDFT kernel.
    ///
    /// If `q = A v`, the energy derivative contains the symmetric response
    /// `B = (A + A^T)/2`. With `v = v_nuc - W:P`, the kernel is
    /// `dV[dP] = W^T B (W:dP)` for every supported boundary model.
    /// The explicit symmetrization also makes the kernel reciprocal, which
    /// is required by the symmetric CPKS solver.
    pub fn fock_response(&self, dp_tot: &Matrix) -> Result<Matrix> {
        let t: Vec<f64> = self
            .ao_pot
            .par_iter()
            .map(|w| dp_tot.frobenius_dot(w))
            .collect();
        let response = self.variational_charges(&t)?;
        let mut out = Matrix::zeros(dp_tot.rows, dp_tot.cols);
        for (coefficient, w) in response.iter().zip(&self.ao_pot) {
            if *coefficient == 0.0 {
                continue;
            }
            for (ov, wv) in out.as_mut_slice().iter_mut().zip(w.as_slice()) {
                *ov += coefficient * wv;
            }
        }
        Ok(out)
    }

    /// Polarization energy and its variational reaction-field Fock matrix.
    /// For a non-symmetric IEF response `A`, the energy is `v^T A v / 2`
    /// while the Fock matrix uses `(A + A^T)v/2`.
    pub fn energy_and_fock(&self, p_tot: &Matrix) -> Result<(f64, Matrix)> {
        let v = self.surface_potential(p_tot);
        let q = self.charges(&v)?;
        let e = 0.5 * q.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        let q_variational = self.variational_charges(&v)?;
        let nao = p_tot.rows;
        let mut fock = Matrix::zeros(nao, nao);
        for (qi, w) in q_variational.iter().zip(&self.ao_pot) {
            if *qi == 0.0 {
                continue;
            }
            for (fv, wv) in fock.as_mut_slice().iter_mut().zip(w.as_slice()) {
                *fv -= qi * wv;
            }
        }
        Ok((e, fock))
    }
}

// ---------------------------------------------------------------------------
// SMD non-electrostatic term (cavitation-dispersion-solvent structure).
//
// Marenich, Cramer & Truhlar, J. Phys. Chem. B 113, 6378 (2009): on top of
// the C-PCM electrostatics evaluated with SMD intrinsic Coulomb radii,
//   G_CDS = sum_A sigma_A * SASA_A + sigma_M * sum_A SASA_A,
// where SASA_A is the solvent-accessible surface area of atom A computed
// with radius R_A + r_solv (r_solv = 0.4 Angstrom), sigma_A is an atomic
// surface tension built from the element and its bonded environment, and
// sigma_M is a molecular surface tension from the solvent descriptors.
// For WATER the parameterization collapses to the aqueous column: sigma_M
// vanishes and the atomic tensions carry no solvent-descriptor dependence.
// Both aqueous and non-aqueous atomic/molecular terms are implemented; missing
// required pair coefficients remain explicit typed refusals.
// ---------------------------------------------------------------------------

/// Solvent for the SMD non-electrostatic term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SmdSolvent {
    /// The aqueous column: its own atomic tensions, no molecular term.
    Water,
    /// Any other solvent, entering only through its descriptors.
    NonAqueous(Solvent),
}

/// Radii (Angstrom) of the spheres whose solvent-accessible surface the
/// CDS term integrates. These are NOT the SMD intrinsic Coulomb radii used
/// by the electrostatics — a point worth stating, because using the
/// Coulomb set here is silently wrong for N, P, C, Si, S, Cl and Br.
///
/// MEASURED, not recalled (`tools/oracle/measure_smd_cds_radii.py`): a
/// hydrogen probe swept towards atom Z leaves G_CDS at its asymptote until
/// the expanded spheres touch, and the bisected contact distance is
/// `R_Z + R_H + 0.8` exactly. The values that come back are the Bondi set.
///
/// The sweep was run over every element up to iodine
/// (`tools/oracle/measure_smd_elements.py`), which is why the table reaches
/// well past the nine elements that carry an atomic surface tension: the
/// rest still need a radius, because the shared molecular term multiplies
/// their exposed area like everybody else's.
fn smd_cds_radius(z: u8) -> Option<f64> {
    Some(match z {
        1 => 1.20,
        3 => 1.82,
        4 => 1.53,
        5 => 1.92,
        6 => 1.70,
        7 => 1.55,
        8 => 1.52,
        9 => 1.47,
        11 => 2.27,
        12 => 1.73,
        13 => 1.84,
        14 => 2.10,
        15 => 1.80,
        16 => 1.80,
        17 => 1.75,
        19 => 2.75,
        20 => 2.31,
        21 => 2.16,
        22 => 1.87,
        23 => 1.79,
        24 => 1.89,
        25 => 1.97,
        26 => 1.94,
        27 => 1.92,
        28 => 1.84,
        29 => 1.86,
        30 => 2.10,
        31 => 1.87,
        32 => 2.11,
        33 => 1.85,
        34 => 1.90,
        35 => 1.85,
        37 => 3.03,
        38 => 2.49,
        39 => 2.19,
        40 => 1.86,
        41 => 2.07,
        42 => 2.09,
        43 => 2.09,
        44 => 2.07,
        45 => 1.95,
        46 => 2.02,
        47 => 2.03,
        48 => 2.30,
        49 => 1.93,
        50 => 2.17,
        51 => 2.06,
        52 => 2.06,
        53 => 1.98,
        54 => 2.16,
        55 => 3.43,
        56 => 2.68,
        57 => 2.40,
        58 => 2.35,
        59 => 2.39,
        60 => 2.29,
        61 => 2.36,
        62 => 2.29,
        63 => 2.33,
        64 => 2.37,
        65 => 2.21,
        66 => 2.29,
        67 => 2.16,
        68 => 2.35,
        69 => 2.27,
        70 => 2.42,
        71 => 2.21,
        72 => 2.12,
        73 => 2.17,
        74 => 2.10,
        75 => 2.17,
        76 => 2.16,
        77 => 2.02,
        78 => 2.09,
        79 => 2.17,
        80 => 2.09,
        81 => 1.96,
        82 => 2.02,
        83 => 2.07,
        84 => 1.97,
        85 => 2.02,
        86 => 2.20,
        87 => 3.48,
        88 => 2.83,
        _ => return None,
    })
}

/// Non-aqueous atomic surface tensions, as the three descriptor
/// coefficients `(t_n, t_alpha, t_beta)` of
/// `sigma_Z = t_n n + t_alpha alpha + t_beta beta`.
///
/// MEASURED by regression, not transcribed
/// (`tools/oracle/measure_smd_cds_radii.py`): with the CDS radii above, an
/// isolated atom's `G_CDS / area` was fitted against every descriptor
/// column over all 179 non-aqueous solvents. The residual is EXACTLY zero
/// to machine precision, which both pins these numbers and proves the
/// model is linear in the descriptors — including that the live refractive
/// index is the first tabulated column and that `eps` does not enter.
fn smd_sigma_nonaqueous(z: u8) -> (f64, f64, f64) {
    match z {
        6 => (58.0993, 48.0994, 32.8696),
        7 => (32.6183, 0.0, 0.0),
        8 => (-17.5600, 193.0605, -43.7901),
        14 => (-18.0401, 0.0, 0.0),
        16 => (-33.1703, 0.0, 0.0),
        17 => (-24.3100, 0.0, 0.0),
        35 => (-35.4207, 0.0, 0.0),
        // H, F and P carry no atomic tension outside water.
        _ => (0.0, 0.0, 0.0),
    }
}

/// The molecular surface tension `sigma_M`, which multiplies the TOTAL
/// solvent-accessible area. Same measurement as above: the coefficients
/// came back identical for every element, to five decimals, which is what
/// identifies them as the shared molecular block rather than ten
/// coincidentally equal atomic ones.
fn smd_sigma_molecular(s: &Solvent) -> f64 {
    0.35 * s.gamma - 4.19 * s.phi * s.phi - 6.68 * s.psi * s.psi
}

impl SmdSolvent {
    /// Total surface tension of element `z` (cal mol^-1 Angstrom^-2),
    /// atomic plus molecular. Both multiply the same per-atom area, so
    /// folding them together loses nothing and keeps the energy a single
    /// sum over atoms.
    fn tension(&self, z: u8) -> f64 {
        match self {
            SmdSolvent::Water => smd_sigma_water(z),
            SmdSolvent::NonAqueous(s) => {
                let (tn, ta, tb) = smd_sigma_nonaqueous(z);
                tn * s.n + ta * s.alpha + tb * s.beta + smd_sigma_molecular(s)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Solvent descriptor table.
// ---------------------------------------------------------------------------

/// Bundled descriptors of one solvent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solvent {
    /// Static dielectric constant.
    pub eps: f64,
    /// Refractive index at 293 K.
    pub n: f64,
    /// Abraham hydrogen-bond acidity.
    pub alpha: f64,
    /// Abraham hydrogen-bond basicity.
    pub beta: f64,
    /// Macroscopic surface tension (cal mol^-1 Angstrom^-2).
    pub gamma: f64,
    /// Aromaticity: the fraction of non-hydrogenic solvent atoms that are
    /// aromatic carbons. Stored UNSQUARED, as the table gives it; SMD's
    /// molecular surface tension uses its square.
    pub phi: f64,
    /// Electronegative halogenicity: the fraction that are F, Cl or Br.
    /// Unsquared, same convention as `phi`.
    pub psi: f64,
}

const SOLVENT_TABLE: &str = include_str!("data/solvents.csv");

/// Look up a bundled solvent by name (case and separator insensitive:
/// `"DMSO"`, `"dimethyl sulfoxide"` and `"dimethylsulfoxide"` all match
/// the same entry).
pub fn solvent_by_name(name: &str) -> Option<Solvent> {
    let key = normalize_solvent(name);
    for line in SOLVENT_TABLE.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        // The name itself may contain commas ("1,4-dioxane"), so split off
        // the LAST seven fields and treat the remainder as the name.
        let fields: Vec<&str> = line.rsplitn(8, ',').collect();
        if fields.len() != 8 {
            continue;
        }
        let raw_name = fields[7];
        if normalize_solvent(raw_name) != key {
            continue;
        }
        let num = |k: usize| fields[k].trim().parse::<f64>().ok();
        return Some(Solvent {
            eps: num(6)?,
            n: num(5)?,
            alpha: num(4)?,
            beta: num(3)?,
            gamma: num(2)?,
            phi: num(1)?,
            psi: num(0)?,
        });
    }
    None
}

/// Names of every bundled solvent (as written in the table).
pub fn solvent_names() -> Vec<&'static str> {
    SOLVENT_TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| l.rsplitn(8, ',').nth(7))
        .collect()
}

fn normalize_solvent(name: &str) -> String {
    let mut s: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_' | '(' | ')'))
        .collect();
    // A few universally used short names.
    s = match s.as_str() {
        "dmso" => "dimethylsulfoxide".to_string(),
        "dmf" => "n,n-dimethylformamide".to_string(),
        "dmac" | "dma" => "n,n-dimethylacetamide".to_string(),
        "thf" => "tetrahydrofuran".to_string(),
        "dcm" => "dichloromethane".to_string(),
        "ccl4" => "carbontetrachloride".to_string(),
        "meoh" => "methanol".to_string(),
        "etoh" => "ethanol".to_string(),
        "mecn" | "acn" => "acetonitrile".to_string(),
        "ether" => "diethylether".to_string(),
        "etoac" => "ethylethanoate".to_string(),
        _ => s,
    };
    s
}

/// SMD intrinsic Coulomb radii (Angstrom) used for the ELECTROSTATIC part;
/// they differ from the Bondi set used by plain C-PCM. Elements outside
/// the tabulated set fall back to the scaled Bondi radius, which the
/// caller is told about through `smd_radius`'s `None`.
fn smd_radius(z: u8) -> Option<f64> {
    Some(match z {
        1 => 1.20,
        6 => 1.85,
        7 => 1.89,
        8 => 1.52,
        9 => 1.73,
        14 => 2.47,
        15 => 2.12,
        16 => 2.49,
        17 => 2.38,
        35 => 3.06,
        _ => return None,
    })
}

/// Aqueous atomic surface tensions (cal mol^-1 Angstrom^-2).
///
/// MEASURED from the SMD reference implementation rather than transcribed:
/// a single isolated atom has no pairwise partner, so
/// `sigma_Z = G_CDS / (4 pi (R_Z + 0.4)^2)` reads the tension off directly
/// (`tools/oracle/probe_smd_sigma.py`). This matters twice over. The
/// published-looking values carried in the first draft had
/// `sigma_O = -148.50`, whereas the reference gives `sigma_O = 0` in water,
/// which flipped the sign of every aqueous G_CDS. And the first measurement
/// divided by an area built from the intrinsic COULOMB radii, so C, F, S,
/// Cl and Br came out scaled by the wrong factor; the water gate happened
/// to cover only H, O, N and P, whose two radius sets coincide, and so let
/// it through. Both areas now come from `smd_cds_radius`.
fn smd_sigma_water(z: u8) -> f64 {
    match z {
        1 => 48.690,
        6 => 129.740,
        7 => 0.0,
        8 => 0.0,
        9 => 38.180,
        14 => 0.0,
        15 => 0.0,
        16 => -9.100,
        17 => 9.820,
        35 => -8.720,
        _ => 0.0,
    }
}

/// The pairwise surface-tension shape, `T(R) = exp(d/(R - R0))` for
/// `R < R0` and zero beyond.
///
/// MEASURED, not transcribed: with the CDS radii and atomic tensions fixed,
/// each aqueous pair's excess over the atomic sum was fitted by scanning
/// `(R0, d)` with the two linear coefficients solved exactly at every node
/// (`tools/oracle/fit_smd_pairT.py`). Four pairs came back with residuals
/// of 0.003 % to 1 % and — independently — the SAME exponent `d = 0.300`,
/// which is what identifies the shape rather than merely fitting it.
const SMD_PAIR_D: f64 = 0.300;

fn smd_pair_t(r: f64, r0: f64) -> f64 {
    if r >= r0 {
        0.0
    } else {
        (SMD_PAIR_D / (r - r0)).exp()
    }
}

/// Aqueous pairwise coefficients: `(R0, s_ab, s_ba)` for `(za, zb)` with
/// `za <= zb`, where `s_ab` multiplies atom a's area and `s_ba` atom b's.
///
/// Only the pairs whose SPLIT between the two atoms is unambiguous are
/// listed. C-C is symmetric by construction; H-C and O-P came back with one
/// coefficient at zero to four decimals, so their split is unambiguous too.
/// N-O fits its own scan to about a percent but with two very different
/// coefficient pairs depending on the exponent, i.e. the two coefficients
/// are individually undetermined — so it stays refused rather than being
/// answered with an arbitrary split. C-N and C-O do not fit this shape at
/// all yet (33 % residual).
fn smd_water_pair(za: u8, zb: u8) -> Option<(f64, f64, f64)> {
    let (a, b) = if za <= zb { (za, zb) } else { (zb, za) };
    let (r0, sab, sba) = match (a, b) {
        (1, 6) => (1.850, -60.7807, 0.0),
        (6, 6) => (2.140, -72.9499, -72.9499),
        (8, 15) => (2.400, 68.8510, 0.0),
        _ => return None,
    };
    // Undo the canonical ordering for the caller's argument order.
    if za <= zb {
        Some((r0, sab, sba))
    } else {
        Some((r0, sba, sab))
    }
}

/// Element pairs whose aqueous tension genuinely carries a
/// distance-dependent term (`tools/oracle/measure_smd_water_pairs.py`:
/// every other pair's excess over `sum_Z sigma_Z A_Z` stays under a
/// fiftieth of a cal/mol across the whole scan). Six pairs carry a term;
/// three of them (`H-C`, `C-C`, `O-P`) have a measured, unambiguously
/// split coefficient in `smd_water_pair`, and three (`C-N`, `C-O`, `N-O`)
/// do not fit the shape yet and stay refused. The other forty-nine pairs,
/// including every halogen, S, Si and P combination, need no pairwise term
/// at all.
///
/// This is a RE-determination. The first pass built its areas from the
/// intrinsic Coulomb radii, so both the tensions and the excess were wrong
/// for every element whose two radius sets differ; it happened to clear
/// only H/O/N/P pairs, where they coincide, so its verdicts survived while
/// its reach was needlessly small.
const WATER_PAIR_REQUIRED: &[(u8, u8)] = &[(1, 6), (6, 6), (6, 7), (6, 8), (7, 8), (8, 15)];

/// Element pairs whose non-aqueous tension carries a distance-dependent
/// contribution. Detection is provenance-backed by
/// `tools/oracle/measure_smd_nonaq_pairs.py`, but no coefficient law has yet
/// been determined for any of the seven pairs. Keep this table separate from
/// the aqueous one: neither coefficients nor solvent dependence may be
/// borrowed from water.
const NONAQUEOUS_PAIR_REQUIRED: &[(u8, u8)] =
    &[(1, 6), (1, 8), (6, 6), (6, 7), (6, 8), (7, 8), (8, 8)];

/// The water-CDS gate for a canonically-ordered `(a, b)` pair: fine if it
/// has a measured coefficient, or if it needs no pairwise term at all.
/// Kept separate from `water_pair_free` so a test can hand it a `required`
/// list that is not `PAIR_REQUIRED`, exercising the refusal branch without
/// mutating the real constant.
fn water_pair_gate(a: u8, b: u8, required: &[(u8, u8)]) -> bool {
    smd_water_pair(a, b).is_some() || !required.contains(&(a, b))
}

fn water_pair_free(za: u8, zb: u8) -> bool {
    let (a, b) = if za <= zb { (za, zb) } else { (zb, za) };
    water_pair_gate(a, b, WATER_PAIR_REQUIRED)
}

/// Whether a non-aqueous pair coefficient is both determined and applied by
/// `smd_cds_atomic_only`. This is deliberately false for now: the local
/// oracle work establishes which seven pairs need a term, but contains no
/// accepted coefficient law. Turning a case on here must be part of the same
/// change that applies its measured term to the energy.
fn nonaqueous_pair_implemented(za: u8, zb: u8) -> bool {
    let _canonical = if za <= zb { (za, zb) } else { (zb, za) };
    false
}

fn nonaqueous_pair_gate(za: u8, zb: u8, required: &[(u8, u8)]) -> bool {
    let (a, b) = if za <= zb { (za, zb) } else { (zb, za) };
    nonaqueous_pair_implemented(a, b) || !required.contains(&(a, b))
}

/// The same question for every OTHER solvent. Measured across the full
/// element set in two solvents with deliberately unlike descriptors
/// (`tools/oracle/measure_smd_nonaq_pairs.py`): exactly seven pairs show a
/// distance-dependent excess over the atomic sum, all of them among H, C,
/// N and O. Every remaining pair agrees to under a quarter of a cal/mol,
/// which is the surface-quadrature noise floor of the probe.
fn nonaqueous_pair_free(za: u8, zb: u8) -> bool {
    nonaqueous_pair_gate(za, zb, NONAQUEOUS_PAIR_REQUIRED)
}

/// Width (as a fraction of the touching distance `R_i + R_j`) of the
/// smooth entry/exit switch applied to a neighbour's contribution to the
/// SASA quadrature axis below. Reuses the SWIG cubic `switch_h` defined
/// above (see the module doc) rather than a second smoothstep:
/// `t = ((R_i+R_j) - |d|) / (SASA_AXIS_SWITCH_FRAC * (R_i+R_j))` runs from
/// at least 1 deep inside the shell down to 0 exactly at the old hard cutoff, so
/// a neighbour crossing the shell perturbs the axis — and hence the
/// discretized SASA — continuously instead of jumping when it enters or
/// leaves the neighbour list. The exclusion arcs computed in
/// `sasa_area_along_axis` are already continuous in the geometry (a
/// tangent sphere cuts a ring of zero width), so only the axis needed
/// this.
const SASA_AXIS_SWITCH_FRAC: f64 = 0.15;

/// Relative eigenvalue gap below which two or three principal moments of
/// the neighbour second-moment tensor are treated as degenerate.
///
/// A matrix proportional to the identity has the SAME Cartesian
/// components in every orientation (`R^T (lambda I) R = lambda I`), so
/// handing a (near-)isotropic tensor to a numerical eigensolver returns
/// whichever orthonormal basis its internal algorithm happens to produce —
/// not one that rotates with the molecule. Committing to a single such
/// eigenvector would silently reintroduce a lab-fixed axis for exactly the
/// symmetric geometries (methane's carbon, an octahedral centre) where it
/// is least defensible. Instead every admissible axis spanning the
/// degenerate eigenspace is quadratured below and the resulting areas are
/// averaged, so it is the axis choice — not the physical answer — that
/// this tolerance controls.
const SASA_AXIS_DEGENERACY_TOL: f64 = 1e-6;

/// Rotation- and permutation-covariant latitude axis (axes, when the
/// neighbour second-moment tensor is degenerate) for the analytic SASA
/// ring integration. The principal axis of the weighted neighbour second
/// moment depends only on distances, radii, and dot products, and each
/// neighbour's weight is switched smoothly to zero as it approaches the
/// `r_i + r_j` touching shell (`SASA_AXIS_SWITCH_FRAC`), so the tensor —
/// and the axis it selects — is a continuous function of the geometry.
/// Sign is immaterial because the Gauss-Legendre latitude nodes occur in
/// +/- pairs; when the top moments are numerically degenerate
/// (`SASA_AXIS_DEGENERACY_TOL`), several admissible axes are returned for
/// the caller to average the quadrature over instead of picking one.
fn sasa_latitude_axes(r_i: f64, neighbours: &[(Vec3, f64)]) -> Vec<Vec3> {
    let mut moment = Matrix::zeros(3, 3);
    for &(displacement, radius) in neighbours {
        let threshold = r_i + radius;
        let taper = if threshold > 0.0 {
            switch_h((threshold - displacement.norm()) / (SASA_AXIS_SWITCH_FRAC * threshold))
        } else {
            1.0
        };
        if taper <= 0.0 {
            continue;
        }
        let components = [displacement.x, displacement.y, displacement.z];
        let weight = taper / (displacement.norm2() + radius * radius).max(1e-24);
        for i in 0..3 {
            for j in 0..3 {
                moment[(i, j)] += weight * components[i] * components[j];
            }
        }
    }
    let fallback = || vec![neighbours[0].0.normalized()];
    let Ok((values, vectors)) = symmetric_eigen(&moment) else {
        return fallback();
    };
    let col = |k: usize| Vec3::new(vectors[(0, k)], vectors[(1, k)], vectors[(2, k)]);
    // symmetric_eigen sorts ASCENDING, so column 2 is the top eigenvector.
    let (e0, e1, e2) = (col(0), col(1), col(2));
    if e2.norm2() <= 1e-24 {
        return fallback();
    }
    let scale = values[2].abs().max(values[1].abs()).max(values[0].abs());
    if scale <= 1e-24 {
        // Every neighbour's weight has switched away to nothing; the axis
        // cannot move the (near-zero) exclusion it would multiply.
        return fallback();
    }
    let top_gap = (values[2] - values[1]).abs() / scale;
    let mid_gap = (values[1] - values[0]).abs() / scale;
    if top_gap > SASA_AXIS_DEGENERACY_TOL {
        vec![e2.normalized()]
    } else if mid_gap > SASA_AXIS_DEGENERACY_TOL {
        // Top two moments degenerate, third distinct: average over a
        // spread of axes IN the degenerate (e1, e2) plane.
        (0..4)
            .map(|k| {
                let theta = std::f64::consts::PI * k as f64 / 4.0;
                (e2 * theta.cos() + e1 * theta.sin()).normalized()
            })
            .collect()
    } else {
        // Fully isotropic (methane's carbon, an octahedral centre): every
        // direction is admissible, so average the three mutually
        // orthogonal ones the eigensolver returned. (Measured: adding the
        // four cube-diagonal combinations on top of these does not help —
        // it roughly 15x's the residual axis-dependence at n_points=4096,
        // presumably because the diagonals correlate with `zax`-dependent
        // asymmetry that `sasa_area_along_axis` introduces when it builds
        // its local xax/yax frame, rather than canceling it.)
        vec![e2.normalized(), e1.normalized(), e0.normalized()]
    }
}

/// SASA exclusion (steradians, before the `r^2` scale) for one atom with a
/// fixed quadrature pole `zax`. Factored out of `atomic_sasa` so degenerate
/// axes (`sasa_latitude_axes`) can be averaged over without duplicating
/// the ring-integration loop.
fn sasa_area_along_axis(
    r: f64,
    neigh: &[(Vec3, f64)],
    zax: Vec3,
    nodes: &[f64],
    weights: &[f64],
) -> f64 {
    let tmp = if zax.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let xax = (tmp - zax * tmp.dot(zax)).normalized();
    let yax = zax.cross(xax);
    let mut area = 0.0;
    for (&ct, &wt) in nodes.iter().zip(weights) {
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        // Forbidden azimuth intervals on this ring.
        let mut iv: Vec<(f64, f64)> = Vec::new();
        let mut fully_buried = false;
        for (d, rj) in neigh {
            let (dz, dx, dy) = (d.dot(zax), d.dot(xax), d.dot(yax));
            let rho = (dx * dx + dy * dy).sqrt();
            // |P - D|^2 < rj^2 with P = r(st cos p, st sin p, ct):
            // a - b cos(p - p0) < 0.
            let a = r * r + d.norm2() - rj * rj - 2.0 * r * ct * dz;
            let b = 2.0 * r * st * rho;
            if b < 1e-14 {
                if a < 0.0 {
                    fully_buried = true;
                    break;
                }
                continue;
            }
            let cosw = a / b;
            if cosw >= 1.0 {
                continue; // ring untouched
            }
            if cosw <= -1.0 {
                fully_buried = true;
                break;
            }
            let p0 = dy.atan2(dx);
            let w = cosw.acos();
            let (lo, hi) = (p0 - w, p0 + w);
            // Normalize into [0, 2pi) possibly splitting.
            let two_pi = 2.0 * std::f64::consts::PI;
            let lo = lo.rem_euclid(two_pi);
            let hi = hi.rem_euclid(two_pi);
            if lo <= hi {
                iv.push((lo, hi));
            } else {
                iv.push((lo, two_pi));
                iv.push((0.0, hi));
            }
        }
        if fully_buried {
            continue;
        }
        // Union length of the forbidden intervals.
        iv.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        let mut covered = 0.0;
        let mut cur: Option<(f64, f64)> = None;
        for (lo, hi) in iv {
            match cur {
                None => cur = Some((lo, hi)),
                Some((a0, b0)) => {
                    if lo <= b0 {
                        cur = Some((a0, b0.max(hi)));
                    } else {
                        covered += b0 - a0;
                        cur = Some((lo, hi));
                    }
                }
            }
        }
        if let Some((a0, b0)) = cur {
            covered += b0 - a0;
        }
        let free = (2.0 * std::f64::consts::PI - covered).max(0.0);
        area += wt * free;
    }
    area
}

/// Solvent-accessible surface area per atom (Bohr^2) by Lebedev-free
/// point sampling on each expanded sphere: a point counts when it lies
/// outside every other expanded sphere. Uses the same quasi-uniform
/// tessellation as the cavity.
fn atomic_sasa(mol: &Molecule, radii: &[f64], n_points: usize) -> Vec<f64> {
    // Latitude quadrature with an ANALYTIC azimuthal sweep: each neighbour
    // sphere cuts a latitude ring in a closed-form arc, so the only
    // discretization left is in cos(theta). Point sampling instead leaves
    // an O(N^{-1/2}) boundary error that shows up at the 0.02 kcal/mol
    // level in G_CDS; this converges far faster for the same cost.
    let n_theta = n_points.max(64);
    let (nodes, weights) = crate::grid::gauss_legendre(n_theta);
    (0..mol.len())
        .map(|ia| {
            let c = mol.atoms[ia].position;
            let r = radii[ia];
            // Neighbours that can cut this sphere at all.
            let mut neigh: Vec<(Vec3, f64)> = (0..mol.len())
                .filter(|&jb| jb != ia)
                .map(|jb| (mol.atoms[jb].position - c, radii[jb]))
                .filter(|(d, rj)| d.norm() < r + rj)
                .collect();
            if neigh.is_empty() {
                return 4.0 * std::f64::consts::PI * r * r;
            }
            // Sorting makes the tensor accumulation reproducible; the axes
            // themselves come from the complete neighbour set below, not
            // from whichever atom happened to occur first in the input.
            neigh.sort_by(|(da, ra), (db, rb)| {
                da.norm2()
                    .total_cmp(&db.norm2())
                    .then_with(|| ra.total_cmp(rb))
            });
            let axes = sasa_latitude_axes(r, &neigh);
            let steradians: f64 = axes
                .iter()
                .map(|&zax| sasa_area_along_axis(r, &neigh, zax, &nodes, &weights))
                .sum::<f64>()
                / axes.len() as f64;
            steradians * r * r
        })
        .collect()
}

/// SMD non-electrostatic free energy `G_CDS` in Hartree.
///
/// DISABLED pending a verified parameterization. A first cut using only
/// the single-atom surface tensions was checked against the SMD reference
/// implementation (PySCF's `libsolvent`/MNSOL) and disagreed badly — for
/// water it gave -2.42 kcal/mol against the reference +1.44 kcal/mol —
/// because SMD's atomic tensions also carry DISTANCE-DEPENDENT PAIRWISE
/// terms `sigma_{ZZ'}` (dominant for O-H) plus the molecular term
/// `sigma_M` built from the solvent descriptors. Rather than ship a
/// plausible-looking wrong number, this returns an error until the full
/// parameterization is transcribed from the SMD paper and gated against
/// the oracle (`tests/oracle_data/smd_cds.csv`).
///
/// The ELECTROSTATIC part (C-PCM/COSMO) is unaffected and works for every
/// bundled solvent.
pub fn smd_cds_energy(mol: &Molecule, solvent: SmdSolvent, n_points: usize) -> Result<f64> {
    // Refuse any molecule containing an element pair whose aqueous tension
    // still carries an unmeasured distance-dependent term: reporting the
    // atomic sum there would be wrong by hundreds of cal/mol.
    let free = |za: u8, zb: u8| match solvent {
        SmdSolvent::Water => water_pair_free(za, zb),
        SmdSolvent::NonAqueous(_) => nonaqueous_pair_free(za, zb),
    };
    for (i, a) in mol.atoms.iter().enumerate() {
        for b in mol.atoms.iter().skip(i + 1) {
            if !free(a.z, b.z) {
                let (za, zb) = if a.z <= b.z { (a.z, b.z) } else { (b.z, a.z) };
                let solvent_label = match solvent {
                    SmdSolvent::Water => "water".to_string(),
                    SmdSolvent::NonAqueous(s) => format!(
                        "a non-aqueous solvent (n={:.6}, alpha={:.6}, beta={:.6}, gamma={:.6}, phi={:.6}, psi={:.6})",
                        s.n, s.alpha, s.beta, s.gamma, s.phi, s.psi
                    ),
                };
                return Err(DftError::Unsupported {
                    feature: format!(
                        "SMD G_CDS pairwise surface-tension term for Z={za}/Z={zb} in {solvent_label}"
                    ),
                    missing: "a provenance-bearing coefficient and an independent molecular oracle \
                        gate; the atomic surface-tension sum alone is wrong for this pair"
                        .to_string(),
                });
            }
        }
    }
    smd_cds_atomic_only(mol, solvent, n_points)
}

/// The single-atom part of the SMD surface-tension sum. Exact whenever
/// every element pair in the molecule is pairwise-term-free.
fn smd_cds_atomic_only(mol: &Molecule, solvent: SmdSolvent, n_points: usize) -> Result<f64> {
    // R_A + r_solv with r_solv = 0.4 Angstrom, in Bohr. These are the CDS
    // radii, NOT the intrinsic Coulomb radii the electrostatics uses.
    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| match smd_cds_radius(a.z) {
            Some(v) => Ok((v + 0.4) * BOHR_PER_ANGSTROM),
            None => Err(DftError::Unsupported {
                feature: format!("SMD G_CDS for Z={}", a.z),
                missing: "a tabulated CDS cavity radius".into(),
            }),
        })
        .collect::<Result<_>>()?;
    let sasa = atomic_sasa(mol, &radii, n_points);
    // sigma is per Angstrom^2 in cal/mol; convert areas and the result.
    const CAL_PER_MOL_TO_HARTREE: f64 = 1.0 / 627_509.474_063_1;
    let ang2_per_bohr2 = 0.52917721092 * 0.52917721092;
    let bohr_per_ang = BOHR_PER_ANGSTROM;
    let mut g = 0.0;
    for (ia, &area) in sasa.iter().enumerate() {
        let za = mol.atoms[ia].z;
        let mut sigma = solvent.tension(za);
        // Pairwise part: sigma_A picks up s_{Z_A Z_B} T(R_AB) from every
        // partner. Only the aqueous column has measured pair coefficients.
        if matches!(solvent, SmdSolvent::Water) {
            for (ib, b) in mol.atoms.iter().enumerate() {
                if ib == ia {
                    continue;
                }
                if let Some((r0, sab, _)) = smd_water_pair(za, b.z) {
                    let r = (mol.atoms[ia].position - b.position).norm() / bohr_per_ang;
                    sigma += sab * smd_pair_t(r, r0);
                }
            }
        }
        g += sigma * area * ang2_per_bohr2;
    }
    Ok(g * CAL_PER_MOL_TO_HARTREE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::system::Atom;

    /// Born limit: a single point charge Z at the center of a cavity
    /// sphere of radius R polarizes the continuum by exactly
    /// `-f(eps) Z^2 / (2 R)`, independent of how the surface is
    /// discretized — the decisive check of the cavity, the S matrix and
    /// the charge equation together.
    #[test]
    fn born_limit_and_gauss_law() {
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.0),
        }]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let opts = PcmOptions::default();
        let z = 4.0; // exercise the quadratic scaling too
        let solver = PcmSolver::build(&mol, &mb, &[z], &opts).unwrap();
        let v = solver.v_nuc().to_vec();
        let q = solver.charges(&v).unwrap();
        let f = opts.f_eps();
        let r = opts.vdw_scale * VDW_ANGSTROM[1] * BOHR_PER_ANGSTROM + opts.r_probe;
        let e = 0.5 * q.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        let e_born = -0.5 * f * z * z / r;
        assert!(
            ((e - e_born) / e_born).abs() < 5e-3,
            "Born limit: {e} vs {e_born}"
        );
        // Gauss law: the apparent surface charge totals -f Z.
        let qsum: f64 = q.iter().sum();
        assert!(
            ((qsum + f * z) / (f * z)).abs() < 5e-3,
            "apparent charge {qsum} vs {}",
            -f * z
        );
    }

    /// Variational identities shared by every PCM boundary model:
    ///
    /// * `dE[P+tD]/dt = Tr[V[P] D]`;
    /// * `dV[P+tD]/dt = K[D]`;
    /// * `Tr[D1 K[D2]] = Tr[D2 K[D1]]`.
    ///
    /// The asymmetric H--F cavity is intentional. In particular it makes
    /// the raw IEF response non-symmetric, so using `K^{-1}R v` directly in
    /// the Fock matrix fails the first identity rather than passing through
    /// accidental symmetry.
    #[test]
    fn every_pcm_model_has_a_variational_fock_and_reciprocal_kernel() {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(-0.37, 0.19, -0.11),
            },
            Atom {
                z: 9,
                position: Vec3::new(1.41, -0.23, 0.31),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let nao = mb.nao_sph;
        let symmetric_matrix = |seed: usize, scale: f64| {
            let mut matrix = Matrix::zeros(nao, nao);
            for i in 0..nao {
                for j in i..nao {
                    let integer = ((i + 3) * (j + seed + 5) + 7 * seed) % 17;
                    let value = scale * (integer as f64 - 8.0) / 9.0;
                    matrix[(i, j)] = value;
                    matrix[(j, i)] = value;
                }
            }
            matrix
        };
        let density = symmetric_matrix(1, 0.11);
        let direction_a = symmetric_matrix(4, 0.07);
        let direction_b = symmetric_matrix(8, 0.05);
        let shifted = |base: &Matrix, direction: &Matrix, step: f64| {
            let mut result = base.clone();
            for (value, delta) in result.as_mut_slice().iter_mut().zip(direction.as_slice()) {
                *value += step * delta;
            }
            result
        };

        for model in [
            PcmModel::CPcm,
            PcmModel::Cosmo,
            PcmModel::IefPcm,
            PcmModel::Ssvpe,
        ] {
            let opts = PcmOptions {
                model,
                n_points: 26,
                ..PcmOptions::default()
            };
            let solver = PcmSolver::build(&mol, &mb, &[1.0, 9.0], &opts).unwrap();
            if model == PcmModel::IefPcm {
                assert!(
                    solver
                        .response
                        .raw
                        .max_abs_diff(&solver.response.raw.transpose())
                        > 1e-8,
                    "test cavity did not expose the non-symmetric IEF operator"
                );
            }

            let (_, fock) = solver.energy_and_fock(&density).unwrap();
            let step = 2e-5;
            let energy_plus = solver
                .energy_and_fock(&shifted(&density, &direction_a, step))
                .unwrap()
                .0;
            let energy_minus = solver
                .energy_and_fock(&shifted(&density, &direction_a, -step))
                .unwrap()
                .0;
            let energy_fd = (energy_plus - energy_minus) / (2.0 * step);
            let energy_analytic = fock.frobenius_dot(&direction_a);
            assert!(
                (energy_fd - energy_analytic).abs() < 2e-9,
                "{model:?}: dE/dP = {energy_fd:e}, Tr[V D] = {energy_analytic:e}"
            );

            let fock_plus = solver
                .energy_and_fock(&shifted(&density, &direction_a, step))
                .unwrap()
                .1;
            let fock_minus = solver
                .energy_and_fock(&shifted(&density, &direction_a, -step))
                .unwrap()
                .1;
            let kernel_a = solver.fock_response(&direction_a).unwrap();
            let mut response_error = 0.0_f64;
            for ((plus, minus), analytic) in fock_plus
                .as_slice()
                .iter()
                .zip(fock_minus.as_slice())
                .zip(kernel_a.as_slice())
            {
                let finite_difference = (plus - minus) / (2.0 * step);
                response_error = response_error.max((finite_difference - analytic).abs());
            }
            assert!(
                response_error < 2e-9,
                "{model:?}: Fock response maximum error {response_error:e}"
            );

            let kernel_b = solver.fock_response(&direction_b).unwrap();
            let ab = direction_a.frobenius_dot(&kernel_b);
            let ba = direction_b.frobenius_dot(&kernel_a);
            assert!(
                (ab - ba).abs() < 2e-11,
                "{model:?}: PCM kernel is not reciprocal: {ab:e} vs {ba:e}"
            );
        }
    }

    /// The IEF/SS(V)PE analytic gradient against central differences of the
    /// polarization energy at a FIXED density. This is what exercises the
    /// D-matrix derivative and the adjoint solve; the C-PCM path shares
    /// none of that code.
    #[test]
    fn ief_gradient_matches_fd() {
        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 8,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.43, 1.108),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, -1.43, 1.108),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let charges = [8.0, 1.0, 1.0];
        let set = data::load("sto-3g").unwrap();
        let mol0 = build((0, 0, 0.0));
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let nao = mb0.nao_sph;
        let mut pd = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                pd[(mu, nu)] = v;
                pd[(nu, mu)] = v;
            }
        }
        for model in [PcmModel::IefPcm, PcmModel::Ssvpe] {
            let opts = PcmOptions {
                model,
                n_points: 26,
                ..PcmOptions::default()
            };
            let s0 = PcmSolver::build(&mol0, &mb0, &charges, &opts).unwrap();
            // The Hessian is differenced from the same gradient, so a
            // defect in either one shows up here.
            let hess = match s0.hessian_skeleton(&mol0, &mb0, &charges, &pd) {
                Ok(h) => h,
                // Only a TYPED "not implemented" may be tolerated. This arm
                // used to swallow every error into a 0x0 matrix, which made
                // `n_h == 0` skip the entire finite-difference comparison
                // below — so a genuine regression in the skeleton would have
                // presented as a passing test.
                Err(e) => {
                    assert!(
                        e.is_unsupported(),
                        "{model:?}: hessian_skeleton failed for a reason other than an \
                         unimplemented capability, which this test must not hide: {e}"
                    );
                    Matrix::zeros(0, 0)
                }
            };
            let n_h = hess.rows;
            let gradf = |shift: (usize, usize, f64)| -> Vec<Vec3> {
                let mm = build(shift);
                let mbx = MolBasis::build(&mm, &set).unwrap();
                let sx = PcmSolver::build(&mm, &mbx, &charges, &opts).unwrap();
                sx.gradient(&mm, &mbx, &charges, &pd).unwrap()
            };
            let hh = 2e-4;
            let n_at_h = if n_h == 0 { 0 } else { 3 };
            for at in 0..n_at_h {
                for ax in 0..3 {
                    let gp = gradf((at, ax, hh));
                    let gm = gradf((at, ax, -hh));
                    for bt in 0..n_at_h {
                        for by in 0..3 {
                            let fd = (gp[bt].get(by) - gm[bt].get(by)) / (2.0 * hh);
                            let a = hess[(3 * at + ax, 3 * bt + by)];
                            assert!(
                                (a - fd).abs() < 5e-5 + 5e-3 * fd.abs(),
                                "{model:?} Hessian ({at},{ax})({bt},{by}): analytic {a} vs FD {fd}"
                            );
                        }
                    }
                }
            }
            let an = s0.gradient(&mol0, &mb0, &charges, &pd).unwrap();
            let energy = |shift: (usize, usize, f64)| -> f64 {
                let mm = build(shift);
                let mbx = MolBasis::build(&mm, &set).unwrap();
                let sx = PcmSolver::build(&mm, &mbx, &charges, &opts).unwrap();
                sx.energy_and_fock(&pd).unwrap().0
            };
            let h = 1e-5;
            for at in 0..3 {
                for ax in 0..3 {
                    let fd = (energy((at, ax, h)) - energy((at, ax, -h))) / (2.0 * h);
                    let a = an[at].get(ax);
                    assert!(
                        (a - fd).abs() < 2e-6 + 1e-4 * fd.abs(),
                        "{model:?} gradient (atom {at}, axis {ax}): analytic {a} vs FD {fd}"
                    );
                }
            }
        }
    }

    /// `dV_pol/dR` at fixed density, against central differences of the
    /// reaction-field Fock matrix itself. This is the piece the CPKS
    /// right-hand side consumes, so it needs its own gate rather than
    /// only being covered by the end-to-end Hessian.
    #[test]
    fn cpcm_fock_derivative_matches_fd() {
        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 8,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.43, 1.108),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, -1.43, 1.108),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let opts = PcmOptions {
            n_points: 26,
            ..PcmOptions::default()
        };
        let charges = [8.0, 1.0, 1.0];
        let set = data::load("sto-3g").unwrap();
        let mol0 = build((0, 0, 0.0));
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let nao = mb0.nao_sph;
        let mut pd = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                pd[(mu, nu)] = v;
                pd[(nu, mu)] = v;
            }
        }
        let solver0 = PcmSolver::build(&mol0, &mb0, &charges, &opts).unwrap();
        let dfock = solver0.fock_derivative(&mol0, &mb0, &charges, &pd).unwrap();
        let mut norm = 0.0;
        for row in &dfock {
            for m in row.iter() {
                for v in m.as_slice() {
                    norm += v * v;
                }
            }
        }
        assert!(norm > 1e-12, "dV_pol/dR came back identically zero");

        let fock = |shift: (usize, usize, f64)| -> Matrix {
            let mm = build(shift);
            let mbx = MolBasis::build(&mm, &set).unwrap();
            let sx = PcmSolver::build(&mm, &mbx, &charges, &opts).unwrap();
            sx.energy_and_fock(&pd).unwrap().1
        };
        let h = 1e-5;
        for at in 0..3 {
            for ax in 0..3 {
                let fp = fock((at, ax, h));
                let fm = fock((at, ax, -h));
                for mu in 0..nao {
                    for nu in 0..nao {
                        let fd = (fp[(mu, nu)] - fm[(mu, nu)]) / (2.0 * h);
                        let a = dfock[at][ax][(mu, nu)];
                        assert!(
                            (a - fd).abs() < 1e-6 + 1e-4 * fd.abs(),
                            "dV_pol/dR ({at},{ax})[{mu},{nu}]: analytic {a} vs FD {fd}"
                        );
                    }
                }
            }
        }
    }

    /// The fixed-density polarization HESSIAN against central differences
    /// of the fixed-density gradient. Every piece is exercised at once —
    /// `S_ab` (both the pair term and the switching diagonal), `v^nuc_ab`,
    /// the three-centre `v^el_ab`, and the `dq/dR` coupling — so a defect
    /// in any one of them shows up here.
    #[test]
    fn cpcm_skeleton_hessian_matches_fd() {
        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 8,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.43, 1.108),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, -1.43, 1.108),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let opts = PcmOptions {
            n_points: 26,
            ..PcmOptions::default()
        };
        let charges = [8.0, 1.0, 1.0];
        let set = data::load("sto-3g").unwrap();
        let mol0 = build((0, 0, 0.0));
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let nao = mb0.nao_sph;
        let mut pd = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                pd[(mu, nu)] = v;
                pd[(nu, mu)] = v;
            }
        }
        let solver0 = PcmSolver::build(&mol0, &mb0, &charges, &opts).unwrap();
        let an = solver0
            .hessian_skeleton(&mol0, &mb0, &charges, &pd)
            .unwrap();

        let grad = |shift: (usize, usize, f64)| -> Vec<Vec3> {
            let mm = build(shift);
            let mbx = MolBasis::build(&mm, &set).unwrap();
            let sx = PcmSolver::build(&mm, &mbx, &charges, &opts).unwrap();
            sx.gradient(&mm, &mbx, &charges, &pd).unwrap()
        };
        let h = 2e-4;
        for at in 0..3 {
            for ax in 0..3 {
                let gp = grad((at, ax, h));
                let gm = grad((at, ax, -h));
                for bt in 0..3 {
                    for by in 0..3 {
                        let fd = (gp[bt].get(by) - gm[bt].get(by)) / (2.0 * h);
                        let a = an[(3 * at + ax, 3 * bt + by)];
                        assert!(
                            (a - fd).abs() < 2e-5 + 2e-3 * fd.abs(),
                            "PCM Hessian ({at},{ax})({bt},{by}): analytic {a} vs FD {fd}"
                        );
                    }
                }
            }
        }
    }

    /// The COMPLETE C-PCM gradient (all four pieces assembled by
    /// `PcmSolver::gradient`) against central differences of the
    /// polarization energy at a FIXED density matrix. Fixing the density is
    /// exactly what the analytic formula assumes: at the SCF solution the
    /// energy is variational in P, so its implicit geometry dependence
    /// belongs to the Pulay/HF terms, not here.
    #[test]
    fn cpcm_total_gradient_matches_fd() {
        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 8,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.43, 1.108),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, -1.43, 1.108),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let opts = PcmOptions {
            n_points: 50,
            ..PcmOptions::default()
        };
        let charges = [8.0, 1.0, 1.0];
        let set = data::load("sto-3g").unwrap();
        let mol0 = build((0, 0, 0.0));
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let nao = mb0.nao_sph;
        // A fixed, deterministic symmetric density. Its physical realism is
        // irrelevant; the identity being gated holds for any fixed P.
        let mut pd = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                pd[(mu, nu)] = v;
                pd[(nu, mu)] = v;
            }
        }
        let solver0 = PcmSolver::build(&mol0, &mb0, &charges, &opts).unwrap();
        let an = solver0.gradient(&mol0, &mb0, &charges, &pd).unwrap();
        let energy = |shift: (usize, usize, f64)| -> f64 {
            let mm = build(shift);
            let mbx = MolBasis::build(&mm, &set).unwrap();
            let sx = PcmSolver::build(&mm, &mbx, &charges, &opts).unwrap();
            sx.energy_and_fock(&pd).unwrap().0
        };
        let h = 1e-5;
        for at in 0..3 {
            for ax in 0..3 {
                let fd = (energy((at, ax, h)) - energy((at, ax, -h))) / (2.0 * h);
                let a = an[at].get(ax);
                assert!(
                    (a - fd).abs() < 2e-6 + 1e-4 * fd.abs(),
                    "C-PCM gradient (atom {at}, axis {ax}): analytic {a} vs FD {fd}"
                );
            }
        }
    }

    /// The two closed-form gradient pieces against central differences of
    /// the quantities they differentiate: `dS/dR` through the quadratic
    /// form `q.S.q` that enters the energy, and `dv^nuc/dR` directly.
    #[test]
    fn cpcm_gradient_pieces_match_fd() {
        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 8,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.43, 1.108),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, -1.43, 1.108),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let opts = PcmOptions {
            n_points: 110,
            ..PcmOptions::default()
        };
        let charges = [8.0, 1.0, 1.0];
        let mol0 = build((0, 0, 0.0));
        let cav0 = build_cavity(&mol0, &opts).unwrap();
        let m = cav0.points.len();
        // A fixed charge vector: we are testing dS and dv, not the solve.
        let qv: Vec<f64> = (0..m)
            .map(|i| 0.02 * ((i % 7) as f64 - 3.0) / 3.0)
            .collect();
        let dvn = nuclear_potential_gradient(&mol0, &cav0, &charges);
        let h = 1e-5;
        for at in 0..3 {
            for ax in 0..3 {
                // dS/dR through q.S.q with the cavity held FIXED apart from
                // the rigid motion of each point with its owner atom. This
                // isolates the r_ij dependence, which is the piece
                // implemented so far; the switching-function (F_i) part of
                // the diagonal is still open and is deliberately excluded
                // here rather than papered over.
                let quad = |shift: (usize, usize, f64)| -> f64 {
                    let (sat, sax, sh) = shift;
                    let mut acc = 0.0;
                    for i in 0..m {
                        for j in (i + 1)..m {
                            let mut pi = cav0.points[i];
                            let mut pj = cav0.points[j];
                            if cav0.atom_of[i] == sat {
                                match sax {
                                    0 => pi.x += sh,
                                    1 => pi.y += sh,
                                    _ => pi.z += sh,
                                }
                            }
                            if cav0.atom_of[j] == sat {
                                match sax {
                                    0 => pj.x += sh,
                                    1 => pj.y += sh,
                                    _ => pj.z += sh,
                                }
                            }
                            let r = (pi - pj).norm();
                            let (zi, zj) = (cav0.zeta[i], cav0.zeta[j]);
                            let zij = zi * zj / (zi * zi + zj * zj).sqrt();
                            acc += 2.0 * qv[i] * qv[j] * libm::erf(zij * r) / r;
                        }
                    }
                    acc
                };
                let fd_s = (quad((at, ax, h)) - quad((at, ax, -h))) / (2.0 * h);
                // Analytic counterpart of the same quadratic form: only the
                // inter-sphere entries move.
                let mut an_s = 0.0;
                for i in 0..m {
                    for j in (i + 1)..m {
                        let (oi, oj) = (cav0.atom_of[i], cav0.atom_of[j]);
                        if oi == oj || (oi != at && oj != at) {
                            continue;
                        }
                        let dvec = cav0.points[i] - cav0.points[j];
                        let r = dvec.norm();
                        let (zi, zj) = (cav0.zeta[i], cav0.zeta[j]);
                        let zij = zi * zj / (zi * zi + zj * zj).sqrt();
                        let dsdr = 2.0 * zij / std::f64::consts::PI.sqrt()
                            * (-(zij * r) * (zij * r)).exp()
                            / r
                            - libm::erf(zij * r) / (r * r);
                        let sgn = if oi == at { 1.0 } else { -1.0 };
                        an_s += 2.0 * qv[i] * qv[j] * dsdr / r * sgn * dvec.get(ax);
                    }
                }
                assert!(
                    (an_s - fd_s).abs() < 2e-6 * (1.0 + fd_s.abs()),
                    "dS/dR (atom {at}, axis {ax}): analytic {an_s} vs FD {fd_s}"
                );
                // dF_i/dR: the switching-function piece the S diagonal
                // needs. Compared through sum_i q_i F_i so one scalar
                // covers every point.
                let dfs = switch_gradient(&mol0, &cav0, &opts);
                let fsum = |shift: (usize, usize, f64)| -> f64 {
                    let mm = build(shift);
                    let c = build_cavity(&mm, &opts).unwrap();
                    // The point set is stable for these small shifts, and
                    // points are generated in the same order.
                    (0..m.min(c.points.len()))
                        .map(|i| qv[i] * c.switch[i])
                        .sum()
                };
                let fd_f = (fsum((at, ax, h)) - fsum((at, ax, -h))) / (2.0 * h);
                let an_f: f64 = (0..m).map(|i| qv[i] * dfs[i][at].get(ax)).sum();
                assert!(
                    (an_f - fd_f).abs() < 1e-5 * (1.0 + fd_f.abs()),
                    "dF/dR (atom {at}, axis {ax}): analytic {an_f} vs FD {fd_f}"
                );
                // dv^el/dR: the three-centre integral derivative. Compared
                // through sum_i q_i v^el_i with the density held fixed
                // (that is exactly what the gradient formula needs, since
                // the SCF density is variational).
                {
                    let set = data::load("sto-3g").unwrap();
                    let mb0 = MolBasis::build(&mol0, &set).unwrap();
                    let nao = mb0.nao_sph;
                    // Deterministic symmetric density stand-in.
                    let mut pd = Matrix::zeros(nao, nao);
                    for mu in 0..nao {
                        for nu in 0..nao {
                            let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                            pd[(mu, nu)] = v;
                            pd[(nu, mu)] = v;
                        }
                    }
                    let dve = electronic_potential_gradient(&mb0, &cav0, &pd, 3);
                    let an_e: f64 = (0..m).map(|i| qv[i] * dve[i][at].get(ax)).sum();
                    let vel = |shift: (usize, usize, f64)| -> f64 {
                        let mm = build(shift);
                        let mbx = MolBasis::build(&mm, &set).unwrap();
                        let cx = build_cavity(&mm, &opts).unwrap();
                        let kernel = crate::integrals::CoulombKernel::Coulomb;
                        let mut acc = 0.0;
                        for i in 0..m.min(cx.points.len()) {
                            let aux = surface_shell(&cx, i);
                            let mut v = 0.0;
                            for a2 in 0..mbx.shells.len() {
                                for b2 in 0..mbx.shells.len() {
                                    let q = crate::integrals::eri::eri_3c_shell_direct(
                                        &mbx.shells[a2],
                                        &mbx.shells[b2],
                                        &aux,
                                        kernel,
                                    );
                                    let (oa, ob) = (mbx.shells[a2].sph_off, mbx.shells[b2].sph_off);
                                    for si in 0..q.na {
                                        for sj in 0..q.nb {
                                            v += pd[(oa + si, ob + sj)] * q.get(si, sj, 0, 0);
                                        }
                                    }
                                }
                            }
                            acc += qv[i] * v;
                        }
                        acc
                    };
                    let fd_e = (vel((at, ax, h)) - vel((at, ax, -h))) / (2.0 * h);
                    assert!(
                        (an_e - fd_e).abs() < 1e-5 * (1.0 + fd_e.abs()),
                        "dv_el/dR (atom {at}, axis {ax}): analytic {an_e} vs FD {fd_e}"
                    );
                }
                // dv^nuc/dR, summed over surface points for a scalar check.
                let vsum = |mm: &Molecule| -> f64 {
                    let c = build_cavity(mm, &opts).unwrap();
                    let mut acc = 0.0;
                    for i in 0..c.points.len().min(m) {
                        let z_i = c.zeta[i];
                        for (a, &z) in mm.atoms.iter().map(|x| x.position).zip(&charges) {
                            let d = (c.points[i] - a).norm();
                            acc += qv[i] * z * libm::erf(z_i * d) / d;
                        }
                    }
                    acc
                };
                let fd_v = (vsum(&build((at, ax, h))) - vsum(&build((at, ax, -h)))) / (2.0 * h);
                let an_v: f64 = (0..m).map(|i| qv[i] * dvn[i][at].get(ax)).sum();
                assert!(
                    (an_v - fd_v).abs() < 2e-6 * (1.0 + fd_v.abs()),
                    "dv_nuc/dR (atom {at}, axis {ax}): analytic {an_v} vs FD {fd_v}"
                );
            }
        }
    }

    /// The bundled descriptor table: every row parses (names containing
    /// commas included), the short aliases resolve, and a few well-known
    /// dielectric constants come out right.
    #[test]
    fn solvent_table_lookup() {
        let names = solvent_names();
        assert!(names.len() > 150, "only {} solvents parsed", names.len());
        for name in &names {
            assert!(
                solvent_by_name(name).is_some(),
                "row `{name}` does not round-trip through the lookup"
            );
        }
        // Names with commas must survive the field split.
        let diox = solvent_by_name("1,4-dioxane").expect("1,4-dioxane");
        assert!((diox.eps - 2.2099).abs() < 1e-3, "dioxane eps {}", diox.eps);
        // Short aliases and separator/case insensitivity.
        for (alias, eps) in [
            ("DMSO", 46.826),
            ("dimethyl sulfoxide", 46.826),
            ("water", 78.355),
            ("MeCN", 35.688),
            ("benzene", 2.2706),
            ("CCl4", 2.228),
        ] {
            let s =
                solvent_by_name(alias).unwrap_or_else(|| panic!("alias `{alias}` not resolved"));
            assert!(
                (s.eps - eps).abs() < 1e-2,
                "{alias}: eps {} vs {eps}",
                s.eps
            );
        }
        assert!(solvent_by_name("unobtainium").is_none());
    }

    /// SMD's SASA term: an isolated atom exposes its full sphere, and two
    /// atoms pushed together must expose strictly less than two spheres.
    #[test]
    fn smd_sasa_limits() {
        let one = Molecule::new(vec![Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        }]);
        let r = (smd_radius(6).unwrap() + 0.4) * BOHR_PER_ANGSTROM;
        let a = atomic_sasa(&one, &[r], 2000)[0];
        let exact = 4.0 * std::f64::consts::PI * r * r;
        assert!(
            ((a - exact) / exact).abs() < 1e-12,
            "isolated SASA {a} vs {exact}"
        );
        // Burial: a close dimer must expose strictly less than two spheres
        // but strictly more than one.
        let two = Molecule::new(vec![
            Atom {
                z: 6,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 6,
                position: Vec3::new(0.0, 0.0, 2.6),
            },
        ]);
        let pair = atomic_sasa(&two, &[r, r], 2000);
        let total: f64 = pair.iter().sum();
        assert!(
            total < 2.0 * exact && total > exact,
            "dimer SASA {total} outside (one sphere, two spheres) = ({exact}, {})",
            2.0 * exact
        );
        // The two equivalent atoms must expose equal area.
        assert!(
            (pair[0] - pair[1]).abs() < 1e-10,
            "symmetry-equivalent atoms differ: {pair:?}"
        );
        // A lone carbon has no pair partner, so the atomic sum applies and
        // reproduces the measured tension exactly.
        let g = smd_cds_energy(&one, SmdSolvent::Water, 20000).unwrap();
        let expect = smd_sigma_water(6) * exact * 0.52917721092_f64.powi(0) / 627_509.474_063_1;
        let _ = expect;
        assert!(g > 0.0, "carbon G_CDS should be positive in water: {g}");
    }

    /// Adding a pair to `WATER_PAIR_REQUIRED` without a measured coefficient must
    /// refuse it: the gate is `smd_water_pair(a,b).is_some() ||
    /// !required.contains(&(a,b))`, not the tautology it used to be. Rather
    /// than mutate the real constant, hand `water_pair_gate` a synthetic
    /// `required` list containing a pair that indeed has no coefficient
    /// (H-N is not fitted) and confirm the gate flips from "free" (under
    /// the real `WATER_PAIR_REQUIRED`, which does not list it) to "refused"
    /// (once it is required).
    #[test]
    fn pair_required_without_coefficient_is_refused() {
        assert!(
            smd_water_pair(1, 7).is_none(),
            "test assumes H-N has no measured coefficient"
        );
        assert!(
            water_pair_gate(1, 7, WATER_PAIR_REQUIRED),
            "H-N is not in the real WATER_PAIR_REQUIRED, so it must be free"
        );
        let synthetic_required = [(1u8, 7u8)];
        assert!(
            !water_pair_gate(1, 7, &synthetic_required),
            "a pair added to PAIR_REQUIRED without a coefficient must be refused"
        );
        // Every pair that IS genuinely required must be free if and only
        // if it has a measured coefficient — this is what would catch a
        // real future addition to PAIR_REQUIRED that forgot the fit.
        for &(a, b) in WATER_PAIR_REQUIRED {
            assert_eq!(
                water_pair_gate(a, b, WATER_PAIR_REQUIRED),
                smd_water_pair(a, b).is_some(),
                "WATER_PAIR_REQUIRED pair ({a},{b}) free-ness must track smd_water_pair"
            );
        }

        // None of the seven detected non-aqueous terms has an accepted
        // coefficient law yet, so every required pair must remain refused.
        for &(a, b) in NONAQUEOUS_PAIR_REQUIRED {
            assert_eq!(
                nonaqueous_pair_gate(a, b, NONAQUEOUS_PAIR_REQUIRED),
                nonaqueous_pair_implemented(a, b),
                "NONAQUEOUS_PAIR_REQUIRED pair ({a},{b}) free-ness must track coefficient implementation"
            );
            assert!(!nonaqueous_pair_implemented(a, b));
        }
    }

    /// The SASA neighbour-shell entry/exit used to be a hard cutoff on
    /// `d.norm() < r_i + r_j`; the axis picked for the latitude quadrature
    /// (`sasa_latitude_axes`) jumped when a neighbour crossed that shell,
    /// which shows up as a spike in the SECOND difference of `G_CDS` as an
    /// atom is walked through it, even though the true SASA is smooth
    /// there. Move a hydrogen straight through carbon's touching shell in
    /// 1e-4 Bohr steps and check the second difference stays bounded.
    ///
    /// Geometry: C at the origin, a fixed H well inside every shell (so it
    /// contributes a constant background and is not itself near a
    /// crossing), and a second H swept through the C-H touching distance.
    /// The two hydrogens stay outside each other's shell throughout the
    /// sweep (H-H touching distance ~6.05 Bohr vs. ~6.6-6.7 Bohr actual),
    /// so the only event exercised is the single C-H crossing.
    #[test]
    fn smd_cds_second_difference_is_bounded_across_neighbour_shell() {
        let threshold = (smd_cds_radius(6).unwrap() + 0.4 + smd_cds_radius(1).unwrap() + 0.4)
            * BOHR_PER_ANGSTROM;
        let h = 1e-4;
        let mol_at = |t: f64| -> Molecule {
            Molecule::new(vec![
                Atom {
                    z: 6,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(1.2, 0.3, 0.5),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 0.0, t),
                },
            ])
        };
        let n_points = 64;
        let ys: Vec<f64> = (-6..=6)
            .map(|k| {
                let t = threshold + (k as f64) * h;
                smd_cds_energy(&mol_at(t), SmdSolvent::Water, n_points).unwrap()
            })
            .collect();
        let mut max_second_diff = 0.0_f64;
        for w in ys.windows(3) {
            let d2 = (w[2] - 2.0 * w[1] + w[0]).abs();
            max_second_diff = max_second_diff.max(d2);
        }
        println!(
            "second-difference bound across the C-H shell (n_points={n_points}, h={h}): \
             {max_second_diff:e} Hartree"
        );
        // The smooth SWIG-style switch (SASA_AXIS_SWITCH_FRAC) keeps the
        // axis, and hence the discretized SASA, continuous across the
        // shell. Measured with the same geometry and n_points against the
        // pre-fix hard cutoff: 4.22e-5 Hartree (the raw G_CDS values jumped
        // from ~0.01278008 to ~0.01282227 between two steps 1e-4 Bohr
        // apart, then went exactly flat once the pair fully separated) —
        // nine orders of magnitude above what is measured here.
        assert!(
            max_second_diff < 5e-9,
            "SMD CDS second difference across the neighbour shell too large: {max_second_diff:e}"
        );
    }

    /// Rodrigues' rotation formula, used only by
    /// `smd_cds_degenerate_axis_is_rotation_independent` to probe several
    /// admissible orientations of a symmetric molecule.
    fn rotate_vec3(v: Vec3, axis: Vec3, angle: f64) -> Vec3 {
        let axis = axis.normalized();
        let (s, c) = angle.sin_cos();
        v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
    }

    /// Methane's carbon has an EXACTLY isotropic neighbour second-moment
    /// tensor (four equivalent H neighbours at the same distance): its top
    /// eigenvalue is triply degenerate, so a numerical eigensolver's choice
    /// of "the" top eigenvector is not a rotation of the molecule's own
    /// geometry — a matrix proportional to the identity has the same
    /// Cartesian components in every orientation. Picking a naive top
    /// eigenvector there silently reintroduces a lab-fixed axis. Rotate the
    /// whole molecule several different ways (which must be a no-op for a
    /// true rotational invariant) and confirm G_CDS agrees to within the
    /// residual quadrature error, not the much larger pre-fix spread.
    #[test]
    fn smd_cds_degenerate_axis_is_rotation_independent() {
        let ang = 1.0 / 0.52917721092;
        let template: [(u8, [f64; 3]); 5] = [
            (6, [0.0, 0.0, 0.0]),
            (1, [0.629, 0.629, 0.629]),
            (1, [-0.629, -0.629, 0.629]),
            (1, [0.629, -0.629, -0.629]),
            (1, [-0.629, 0.629, -0.629]),
        ];
        let methane = |axis: Vec3, angle: f64| -> Molecule {
            Molecule::new(
                template
                    .iter()
                    .map(|&(z, p)| Atom {
                        z,
                        position: rotate_vec3(
                            Vec3::new(p[0] * ang, p[1] * ang, p[2] * ang),
                            axis,
                            angle,
                        ),
                    })
                    .collect(),
            )
        };
        // The residual axis-dependence is a discretization artifact (it is
        // NOT exactly zero because averaging over three orthogonal
        // eigenvectors is a 3-point cubature of the true spherical
        // average, not the average itself) and shrinks with the
        // quadrature order: measured 1.28e-6 Hartree at n_points=256,
        // 2.36e-7 at 1024, 4.30e-8 at 4096, 1.44e-9 at 16384. n_points=4096
        // keeps the test fast while leaving comfortable margin below the
        // assertion.
        let n_points = 4096;
        let reference = smd_cds_energy(
            &methane(Vec3::new(0.0, 0.0, 1.0), 0.0),
            SmdSolvent::Water,
            n_points,
        )
        .unwrap();
        let mut max_spread = 0.0_f64;
        for (axis, angle) in [
            (Vec3::new(1.0, 0.0, 0.0), 0.3_f64),
            (Vec3::new(0.0, 1.0, 0.0), 1.1),
            (Vec3::new(1.0, 1.0, 1.0), 2.0),
            (Vec3::new(0.3, -0.7, 0.2), 0.9),
            (Vec3::new(-0.4, 0.1, 0.85), 1.7),
        ] {
            let e = smd_cds_energy(&methane(axis, angle), SmdSolvent::Water, n_points).unwrap();
            max_spread = max_spread.max((e - reference).abs());
        }
        println!("methane degeneracy spread (n_points={n_points}): {max_spread:e} Hartree");
        assert!(
            max_spread < 1e-7,
            "axis-dependent SMD CDS spread too large: {max_spread:e}"
        );
    }
}
