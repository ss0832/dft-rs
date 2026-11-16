// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic band velocities, effective-mass tensors and constant-relaxation-
//! time Boltzmann transport of the converged Kohn-Sham operator.
//!
//! The converged real-space Fock and overlap image blocks `F_T`, `S_T` give
//! the Bloch matrices at any wave vector, `H(k) = sum_T exp(ik.T) F_T` and
//! `S(k)`, and their k derivatives `d_a M = sum_T i T_a exp(ik.T) M_T`,
//! `d_a d_b M = -sum_T T_a T_b exp(ik.T) M_T`. Off the SCF mesh this is the
//! exact non-self-consistent band structure of the converged density.
//!
//! With `H c_n = e_n S c_n`, `c_m^+ S c_n = delta_mn`, the first-order
//! amplitudes `d_b c_n = sum_m c_m U^b_mn` follow from the k derivative of
//! the eigenvalue equation and the normalization:
//!
//! ```text
//! V^a_mn  = c_m^+ (d_a H - e_n d_a S) c_n,    S~^a_mn = c_m^+ d_a S c_n
//! U^b_mn  = V^b_mn / (e_n - e_m)  (m != n),   U^b_nn = -S~^b_nn / 2
//! v_n,a   = d e_n / d k_a = V^a_nn
//! d2 e_n / dk_a dk_b = c_n^+ (d_a d_b H - e_n d_a d_b S) c_n
//!                      - v_a S~^b_nn - v_b S~^a_nn
//!                      + sum_(m != n) 2 Re[V^a_mn* V^b_mn] / (e_n - e_m)
//! ```
//!
//! (differentiate `v_a = c_n^+ (d_a H - e_n d_a S) c_n` along `k_b`; the
//! `m = n` amplitude gives the overlap terms). The sum runs over every band
//! of the basis, so the expression is exact within the basis.
//!
//! A cluster `D` of degenerate bands (within `tolerance`) is treated by
//! degenerate perturbation theory along a direction `u`: with the block
//! matrices
//!
//! ```text
//! W_ij = c_i^+ (H_u - e S_u) c_j,   S~_ij = c_i^+ S_u c_j      (i, j in D)
//! M_ij = c_i^+ (H_uu - e S_uu) c_j - (W S~ + S~ W)_ij
//!        + 2 sum_(m not in D) V_im V_mj / (e - e_m)
//! ```
//!
//! the velocities are the eigenvalues of `W`, and the second derivatives of
//! the branches are the eigenvalues of `M` restricted to each eigenspace of
//! `W` (Rellich: branches split at first order keep the diagonal of `M` in
//! the eigenbasis of `W`). For one band this is the formula above.
//!
//! Transport uses the rigid band and a constant relaxation time `tau`:
//!
//! ```text
//! L^(alpha)_ab = g/(V Nk) sum_(n,k) v_a v_b (e - mu)^alpha (-df/de)
//! sigma/tau = e^2 L0,  S = -(1/(e T)) L0^-1 L1,  kappa_e/tau = (1/T)(L2 - L1 L0^-1 L1)
//! ```
//!
//! For a degenerate cluster the basis-independent `sum_ij Re[W^a_ij W^b_ji]`
//! replaces `v_a v_b`.

use crate::dfpt::orbitals::{generalized_orbitals, BandOrbitals};
use crate::error::{DftError, Result};
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::dielectric::{add_exchange_blocks, fock_blocks, k_derivative};
use crate::periodic::integrals::{bloch, ImageBlock, Operator};
use crate::periodic::phonon::{adjoint, product, Reference};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::xc::ResolvedFunctional;

/// Canonical-orthogonalization threshold of the band eigenproblems.
const ORTHOGONALIZATION_CUTOFF: f64 = 1e-7;

/// The converged Kohn-Sham operator as image blocks.
pub struct BandModel {
    fock: [Vec<ImageBlock>; 2],
    overlap: Vec<ImageBlock>,
    /// 1 for a closed-shell reference, 2 for an unrestricted one.
    pub spins: usize,
    /// Largest `|bloch(F images) - F_SCF|` over the SCF mesh and spins.
    pub fock_image_residual: f64,
}

/// `-sum_T T_a T_b exp(ik.T) M(T)`.
fn k_second_derivative(blocks: &[ImageBlock], k: Vec3, a: usize, b: usize) -> BlochMatrix {
    let n = blocks.first().map_or(0, |x| x.matrix.rows);
    let mut out = BlochMatrix::zeros(n, n);
    for block in blocks {
        let weight = block.translation.get(a) * block.translation.get(b);
        if weight == 0.0 {
            continue;
        }
        let phase = k.dot(block.translation);
        let (c, s) = (phase.cos(), phase.sin());
        for ((re, im), v) in out
            .re
            .as_mut_slice()
            .iter_mut()
            .zip(out.im.as_mut_slice().iter_mut())
            .zip(block.matrix.as_slice())
        {
            *re -= weight * c * v;
            *im -= weight * s * v;
        }
    }
    out
}

/// `c_i^+ M c_j` for the columns `rows` and `cols` of `c`.
fn sandwich(c: &BlochMatrix, m: &BlochMatrix) -> BlochMatrix {
    product(&product(&adjoint(c), m), c)
}

fn complex(m: &BlochMatrix, i: usize, j: usize) -> (f64, f64) {
    (m.re[(i, j)], m.im[(i, j)])
}

/// Eigenvalues and eigenvectors (columns) of a small Hermitian matrix.
pub(crate) fn hermitian_eigen(m: &BlochMatrix) -> Result<(Vec<f64>, BlochMatrix)> {
    let mut h = m.clone();
    h.make_hermitian();
    let n = h.rows();
    let mut identity = BlochMatrix::zeros(n, n);
    for i in 0..n {
        identity.re[(i, i)] = 1.0;
    }
    let orbitals = generalized_orbitals(&h, &identity, 1e-12)?;
    Ok((orbitals.energies, orbitals.coefficients))
}

/// Clusters of consecutive bands whose energies differ by less than
/// `tolerance` (transitively).
fn clusters(energies: &[f64], tolerance: f64) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    for (n, e) in energies.iter().enumerate() {
        match out.last_mut() {
            Some(last) if (e - energies[*last.last().unwrap()]).abs() < tolerance => last.push(n),
            _ => out.push(vec![n]),
        }
    }
    out
}

/// One branch of a band cluster along a direction.
#[derive(Clone, Debug)]
pub struct Branch {
    /// `d e / dt` along the unit direction (Hartree Bohr).
    pub velocity: f64,
    /// `d2 e / dt2` along the unit direction (inverse electron masses).
    pub second_derivative: f64,
}

/// Band derivatives at one k point and spin.
#[derive(Clone, Debug)]
pub struct BandDerivatives {
    /// Band energies (Hartree).
    pub energies: Vec<f64>,
    /// Bands of the cluster that contains the requested band.
    pub cluster: Vec<usize>,
    /// Velocity `d e_n/dk` (Hartree Bohr), for a non-degenerate band.
    pub velocity: Option<[f64; 3]>,
    /// Inverse effective-mass tensor `d2 e_n/dk_a dk_b` (1/m_e), for a
    /// non-degenerate band.
    pub inverse_mass: Option<[[f64; 3]; 3]>,
    /// Branches of the cluster along the requested direction.
    pub branches: Vec<Branch>,
}

impl BandModel {
    /// Image blocks of a converged semilocal, full/range-separated hybrid or
    /// Hartree-Fock reference, including translated local/nonlocal ECPs.
    /// The density is frozen; no nuclear response kernel is required.
    pub fn new(
        cell: &Cell,
        functional: &ResolvedFunctional,
        result: &PeriodicScfResult,
        options: &PeriodicScfOptions,
    ) -> Result<Self> {
        result.validate_derivative_request(cell, functional, options)?;
        let reference = Reference::new(cell, result, options)?;
        let mut fock = fock_blocks(cell, functional, options, &reference)?;
        let overlap = crate::periodic::integrals::image_blocks(
            &cell.basis,
            &cell.lattice,
            cell.overlap_cutoff,
            Operator::Overlap,
        );
        let spins = if result.unrestricted { 2 } else { 1 };
        add_exchange_blocks(&mut fock, cell, functional, result, options)?;
        let mut fock_image_residual = 0.0_f64;
        for spin in 0..spins {
            for (ik, k) in result.k_points.iter().enumerate() {
                let (re, im) = bloch(&fock[spin], *k);
                fock_image_residual = fock_image_residual
                    .max(re.max_abs_diff(&result.fock[spin][ik].re))
                    .max(im.max_abs_diff(&result.fock[spin][ik].im));
            }
        }
        Ok(Self {
            fock,
            overlap,
            spins,
            fock_image_residual,
        })
    }

    fn matrices(&self, spin: usize, k: Vec3) -> (BlochMatrix, BlochMatrix) {
        let mut h = BlochMatrix::from_pair(bloch(&self.fock[spin], k));
        let mut s = BlochMatrix::from_pair(bloch(&self.overlap, k));
        h.make_hermitian();
        s.make_hermitian();
        (h, s)
    }

    /// Orbitals and band energies at a Cartesian `k` (1/Bohr).
    pub fn bands(&self, spin: usize, k: Vec3) -> Result<BandOrbitals> {
        if spin >= self.spins {
            return Err(DftError::InvalidInput(format!(
                "spin channel {spin} is not present"
            )));
        }
        let (h, s) = self.matrices(spin, k);
        generalized_orbitals(&h, &s, ORTHOGONALIZATION_CUTOFF)
    }

    /// Physical intra/interband currents with the AO position connection.
    /// v_mn=H'_mn-e_m S'_mn+i(e_m-e_n) D_mn. This division-free
    /// expression remains regular at exact degeneracy and is Hermitian by
    /// D^+=D+i S'. Dipoles must use this model's original AO image cutoff.
    pub(crate) fn coherent_point(
        &self,
        spin: usize,
        k: Vec3,
        dipoles: &[Vec<ImageBlock>; 3],
    ) -> Result<crate::periodic::coherent_transport::CoherentPoint> {
        let orbitals = self.bands(spin, k)?;
        let n = orbitals.energies.len();
        let velocities = std::array::from_fn(|axis| {
            let dh = sandwich(
                &orbitals.coefficients,
                &k_derivative(&self.fock[spin], k, axis),
            );
            let ds = sandwich(
                &orbitals.coefficients,
                &k_derivative(&self.overlap, k, axis),
            );
            let d = sandwich(
                &orbitals.coefficients,
                &BlochMatrix::from_pair(bloch(&dipoles[axis], k)),
            );
            let mut v = BlochMatrix::zeros(n, n);
            for m in 0..n {
                for j in 0..n {
                    let em = orbitals.energies[m];
                    let de = em - orbitals.energies[j];
                    v.re[(m, j)] = dh.re[(m, j)] - em * ds.re[(m, j)] - de * d.im[(m, j)];
                    v.im[(m, j)] = dh.im[(m, j)] - em * ds.im[(m, j)] + de * d.re[(m, j)];
                }
            }
            v
        });
        Ok(crate::periodic::coherent_transport::CoherentPoint {
            energies: orbitals.energies,
            velocities,
        })
    }

    /// First derivatives `(d_a H, d_a S)` along the unit direction `u`.
    fn directional(&self, spin: usize, k: Vec3, u: Vec3) -> (BlochMatrix, BlochMatrix) {
        let mut h = BlochMatrix::zeros(0, 0);
        let mut s = BlochMatrix::zeros(0, 0);
        for a in 0..3 {
            let w = u.get(a);
            if w == 0.0 {
                continue;
            }
            let (dh, ds) = (
                k_derivative(&self.fock[spin], k, a),
                k_derivative(&self.overlap, k, a),
            );
            if h.rows() == 0 {
                h = BlochMatrix::zeros(dh.rows(), dh.cols());
                s = BlochMatrix::zeros(ds.rows(), ds.cols());
            }
            h.add_scaled(&dh, w);
            s.add_scaled(&ds, w);
        }
        (h, s)
    }

    fn directional_second(&self, spin: usize, k: Vec3, u: Vec3) -> (BlochMatrix, BlochMatrix) {
        let mut h = BlochMatrix::zeros(0, 0);
        let mut s = BlochMatrix::zeros(0, 0);
        for a in 0..3 {
            for b in 0..3 {
                let w = u.get(a) * u.get(b);
                if w == 0.0 {
                    continue;
                }
                let (dh, ds) = (
                    k_second_derivative(&self.fock[spin], k, a, b),
                    k_second_derivative(&self.overlap, k, a, b),
                );
                if h.rows() == 0 {
                    h = BlochMatrix::zeros(dh.rows(), dh.cols());
                    s = BlochMatrix::zeros(ds.rows(), ds.cols());
                }
                h.add_scaled(&dh, w);
                s.add_scaled(&ds, w);
            }
        }
        (h, s)
    }

    /// Velocity, inverse effective mass (non-degenerate band) and the
    /// branches of the band's cluster along `direction` (default: the
    /// Cartesian axes are used only for the tensor).
    pub fn derivatives(
        &self,
        spin: usize,
        k: Vec3,
        band: usize,
        direction: Option<Vec3>,
        tolerance: f64,
    ) -> Result<BandDerivatives> {
        let orbitals = self.bands(spin, k)?;
        let energies = orbitals.energies.clone();
        if band >= energies.len() {
            return Err(DftError::InvalidInput(format!(
                "band {band} is out of range (0..{})",
                energies.len()
            )));
        }
        let cluster = clusters(&energies, tolerance)
            .into_iter()
            .find(|c| c.contains(&band))
            .unwrap();
        let c = &orbitals.coefficients;
        let mut velocity = None;
        let mut inverse_mass = None;
        if cluster.len() == 1 {
            let e = energies[band];
            let mut v_mats = Vec::with_capacity(3);
            let mut s_mats = Vec::with_capacity(3);
            for a in 0..3 {
                let (dh, ds) = (
                    k_derivative(&self.fock[spin], k, a),
                    k_derivative(&self.overlap, k, a),
                );
                let mut g = dh.clone();
                g.add_scaled(&ds, -e);
                v_mats.push(sandwich(c, &g));
                s_mats.push(sandwich(c, &ds));
            }
            let v: [f64; 3] = std::array::from_fn(|a| v_mats[a].re[(band, band)]);
            let mut tensor = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in a..3 {
                    let mut g = k_second_derivative(&self.fock[spin], k, a, b);
                    g.add_scaled(&k_second_derivative(&self.overlap, k, a, b), -e);
                    let mut value = sandwich(c, &g).re[(band, band)]
                        - v[a] * s_mats[b].re[(band, band)]
                        - v[b] * s_mats[a].re[(band, band)];
                    for (m, em) in energies.iter().enumerate() {
                        if m == band {
                            continue;
                        }
                        let (ar, ai) = complex(&v_mats[a], m, band);
                        let (br, bi) = complex(&v_mats[b], m, band);
                        value += 2.0 * (ar * br + ai * bi) / (e - em);
                    }
                    tensor[a][b] = value;
                    tensor[b][a] = value;
                }
            }
            velocity = Some(v);
            inverse_mass = Some(tensor);
        }
        let branches = match direction {
            None => Vec::new(),
            Some(d) => {
                let norm = d.norm();
                if !(norm > 0.0) {
                    return Err(DftError::InvalidInput(
                        "direction must be a non-zero vector".into(),
                    ));
                }
                self.cluster_branches(spin, k, &orbitals, &cluster, d * (1.0 / norm))?
            }
        };
        Ok(BandDerivatives {
            energies,
            cluster,
            velocity,
            inverse_mass,
            branches,
        })
    }

    /// Degenerate perturbation theory of a cluster along the unit direction.
    fn cluster_branches(
        &self,
        spin: usize,
        k: Vec3,
        orbitals: &BandOrbitals,
        cluster: &[usize],
        u: Vec3,
    ) -> Result<Vec<Branch>> {
        let energies = &orbitals.energies;
        let c = &orbitals.coefficients;
        let e = cluster.iter().map(|n| energies[*n]).sum::<f64>() / cluster.len() as f64;
        let (hu, su) = self.directional(spin, k, u);
        let (huu, suu) = self.directional_second(spin, k, u);
        let mut g1 = hu.clone();
        g1.add_scaled(&su, -e);
        let mut g2 = huu.clone();
        g2.add_scaled(&suu, -e);
        let v = sandwich(c, &g1);
        let st = sandwich(c, &su);
        let second = sandwich(c, &g2);
        let d = cluster.len();
        let pick = |m: &BlochMatrix| -> BlochMatrix {
            let mut out = BlochMatrix::zeros(d, d);
            for (i, &p) in cluster.iter().enumerate() {
                for (j, &q) in cluster.iter().enumerate() {
                    out.re[(i, j)] = m.re[(p, q)];
                    out.im[(i, j)] = m.im[(p, q)];
                }
            }
            out
        };
        let w = pick(&v);
        let s_block = pick(&st);
        let mut m = pick(&second);
        let ws = product(&w, &s_block);
        let sw = product(&s_block, &w);
        m.add_scaled(&ws, -1.0);
        m.add_scaled(&sw, -1.0);
        for (i, &p) in cluster.iter().enumerate() {
            for (j, &q) in cluster.iter().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (n, en) in energies.iter().enumerate() {
                    if cluster.contains(&n) {
                        continue;
                    }
                    let (ar, ai) = complex(&v, p, n);
                    let (br, bi) = complex(&v, n, q);
                    let scale = 2.0 / (e - en);
                    re += scale * (ar * br - ai * bi);
                    im += scale * (ar * bi + ai * br);
                }
                m.re[(i, j)] += re;
                m.im[(i, j)] += im;
            }
        }
        let (velocities, q) = hermitian_eigen(&w)?;
        let scale = velocities.iter().fold(1.0_f64, |s, x| s.max(x.abs()));
        let mut branches = Vec::with_capacity(d);
        for group in clusters(&velocities, 1e-9 * scale) {
            let mut basis = BlochMatrix::zeros(d, group.len());
            for (col, &g) in group.iter().enumerate() {
                for row in 0..d {
                    basis.re[(row, col)] = q.re[(row, g)];
                    basis.im[(row, col)] = q.im[(row, g)];
                }
            }
            let (seconds, _) = hermitian_eigen(&sandwich(&basis, &m))?;
            for (index, s) in seconds.into_iter().enumerate() {
                branches.push(Branch {
                    velocity: velocities[group[index]],
                    second_derivative: s,
                });
            }
        }
        Ok(branches)
    }

    /// Transport states of one k point and spin: per degenerate cluster the
    /// mean energy, `sum_ij Re[W^a_ij W^b_ji]` and the cluster size.
    /// Velocities are evaluated only for clusters with `include(e)`.
    pub fn transport_states(
        &self,
        spin: usize,
        k: Vec3,
        tolerance: f64,
        include: impl Fn(f64) -> bool,
    ) -> Result<Vec<TransportState>> {
        let orbitals = self.bands(spin, k)?;
        let groups = clusters(&orbitals.energies, tolerance);
        let mut out = Vec::with_capacity(groups.len());
        let mut blocks: Option<[BlochMatrix; 6]> = None;
        for group in groups {
            let e = group.iter().map(|n| orbitals.energies[*n]).sum::<f64>() / group.len() as f64;
            let mut velocity2 = [[0.0; 3]; 3];
            if include(e) {
                if blocks.is_none() {
                    blocks = Some(std::array::from_fn(|i| {
                        if i < 3 {
                            sandwich(
                                &orbitals.coefficients,
                                &k_derivative(&self.fock[spin], k, i),
                            )
                        } else {
                            sandwich(
                                &orbitals.coefficients,
                                &k_derivative(&self.overlap, k, i - 3),
                            )
                        }
                    }));
                }
                let b = blocks.as_ref().unwrap();
                // W^a = H~^a - e S~^a on the cluster.
                let w = |a: usize, i: usize, j: usize| -> (f64, f64) {
                    (
                        b[a].re[(i, j)] - e * b[a + 3].re[(i, j)],
                        b[a].im[(i, j)] - e * b[a + 3].im[(i, j)],
                    )
                };
                for x in 0..3 {
                    for y in 0..3 {
                        let mut sum = 0.0;
                        for &i in &group {
                            for &j in &group {
                                let (ar, ai) = w(x, i, j);
                                let (br, bi) = w(y, j, i);
                                sum += ar * br - ai * bi;
                            }
                        }
                        velocity2[x][y] = sum;
                    }
                }
            }
            out.push(TransportState {
                energy: e,
                velocity2,
                count: group.len() as f64,
            });
        }
        Ok(out)
    }

    /// Isolated bands in the transport window, with analytic velocities and
    /// curvatures for the ordinary, weak-field Hall response. A contributing
    /// degenerate cluster requires a multiband kinetic equation and is refused;
    /// it is never split by an artificial denominator or silently omitted.
    pub fn hall_states(
        &self,
        spin: usize,
        k: Vec3,
        tolerance: f64,
        include: impl Fn(f64) -> bool,
    ) -> Result<Vec<HallState>> {
        if !tolerance.is_finite() || tolerance <= 0.0 || !(0..3).all(|a| k.get(a).is_finite()) {
            return Err(DftError::InvalidInput(
                "Hall states need finite k and a positive finite degeneracy tolerance".into(),
            ));
        }
        let orbitals = self.bands(spin, k)?;
        let mut states = Vec::new();
        for group in clusters(&orbitals.energies, tolerance) {
            if !group.iter().any(|&n| include(orbitals.energies[n])) {
                continue;
            }
            if group.len() != 1 {
                return Err(DftError::Unsupported {
                    feature: format!("Hall transport of bands {group:?} at k={k:?}, spin={spin}"),
                    missing: "a covariant multiband kinetic equation for a degenerate cluster in the transport window".into(),
                });
            }
            let n = group[0];
            let derivatives = self.derivatives(spin, k, n, None, tolerance)?;
            states.push(HallState {
                energy: orbitals.energies[n],
                velocity: derivatives.velocity.unwrap(),
                inverse_mass: derivatives.inverse_mass.unwrap(),
            });
        }
        Ok(states)
    }
}

/// One state (or degenerate cluster) entering the transport sums.
#[derive(Clone, Debug)]
pub struct TransportState {
    /// Energy (Hartree).
    pub energy: f64,
    /// `sum v_a v_b` over the cluster (atomic units).
    pub velocity2: [[f64; 3]; 3],
    /// Number of bands in the cluster.
    pub count: f64,
}

/// Transport integrals in atomic units at one chemical potential and
/// temperature.
#[derive(Clone, Debug)]
pub struct TransportIntegrals {
    pub l0: [[f64; 3]; 3],
    pub l1: [[f64; 3]; 3],
    pub l2: [[f64; 3]; 3],
    /// Electrons per Bohr^3 in the bands.
    pub electron_density: f64,
}

/// Fermi function and `-df/de` without overflow.
fn fermi(x: f64) -> (f64, f64) {
    if x > 0.0 {
        let t = (-x).exp();
        (t / (1.0 + t), t / ((1.0 + t) * (1.0 + t)))
    } else {
        let t = x.exp();
        (1.0 / (1.0 + t), t / ((1.0 + t) * (1.0 + t)))
    }
}

/// `L^(alpha)` and the electron density of `states` (all k points of a mesh
/// of `nk` points in a cell of `volume`, spin degeneracy `g`) at `mu` and
/// `kt` (Hartree). States with `|e - mu| > window kt` add to the electron
/// count but not to `L` (their `-df/de < e^-window / kt`).
pub fn transport_integrals(
    states: &[TransportState],
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
) -> Result<TransportIntegrals> {
    let (l, electron_density) =
        weighted_transport_integrals(states, volume, nk, g, mu, kt, window, &|_| Ok(1.0))?;
    Ok(TransportIntegrals {
        l0: l[0],
        l1: l[1],
        l2: l[2],
        electron_density,
    })
}

/// Scalars used by the shared real and retarded transport quadrature.
pub(crate) trait TransportWeight:
    Copy + Default + std::ops::AddAssign + std::ops::Mul<f64, Output = Self> + Send
{
    fn finite(self) -> bool;
}

impl TransportWeight for f64 {
    fn finite(self) -> bool {
        self.is_finite()
    }
}

impl TransportWeight for crate::response::complex::Complex {
    fn finite(self) -> bool {
        self.is_finite()
    }
}

fn validate_transport(volume: f64, nk: usize, g: f64, mu: f64, kt: f64, window: f64) -> Result<()> {
    if ![volume, g, kt, window]
        .iter()
        .all(|x| x.is_finite() && *x > 0.0)
        || !mu.is_finite()
        || nk == 0
    {
        return Err(DftError::InvalidInput(
            "transport needs finite positive temperature, volume, spin weight, k count and window, and finite chemical potential".into(),
        ));
    }
    Ok(())
}

/// Evaluate scalar lifetimes or causal propagators independently, then sum
/// in state order. Parallel evaluation leaves the quadrature deterministic.
fn transport_weights<T: TransportWeight>(
    energies: &[f64],
    mu: f64,
    kt: f64,
    window: f64,
    weight: &(impl Fn(f64) -> Result<T> + Sync),
) -> Result<Vec<T>> {
    use rayon::prelude::*;
    let evaluate = |e: &f64| {
        if !e.is_finite() {
            return Err(DftError::InvalidInput(
                "transport energies must be finite".into(),
            ));
        }
        if ((*e - mu) / kt).abs() > window {
            return Ok(T::default());
        }
        let value = weight(*e)?;
        if !value.finite() {
            return Err(DftError::InvalidInput(
                "transport weight exceeds the finite floating-point range".into(),
            ));
        }
        Ok(value)
    };
    if energies.len() >= 1024 {
        energies.par_iter().map(evaluate).collect()
    } else {
        energies.iter().map(evaluate).collect()
    }
}

pub(crate) fn weighted_transport_integrals<T: TransportWeight>(
    states: &[TransportState],
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    weight: &(impl Fn(f64) -> Result<T> + Sync),
) -> Result<([[[T; 3]; 3]; 3], f64)> {
    validate_transport(volume, nk, g, mu, kt, window)?;
    let weights = transport_weights(
        &states.iter().map(|s| s.energy).collect::<Vec<_>>(),
        mu,
        kt,
        window,
        weight,
    )?;
    let scale = g / (volume * nk as f64);
    let mut l = [[[T::default(); 3]; 3]; 3];
    let mut electrons = 0.0;
    for (s, lifetime) in states.iter().zip(weights) {
        if !s.count.is_finite()
            || s.count <= 0.0
            || s.velocity2.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "transport cluster counts must be positive finite and velocity products finite"
                    .into(),
            ));
        }
        let x = (s.energy - mu) / kt;
        let (f, df) = fermi(x);
        electrons += s.count * f;
        if x.abs() > window {
            continue;
        }
        let weight = df / kt;
        let de = s.energy - mu;
        let powers = [1.0, de, de * de];
        for (alpha, p) in powers.iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    l[alpha][a][b] += lifetime * (scale * weight * p * s.velocity2[a][b]);
                }
            }
        }
    }
    if !electrons.is_finite()
        || !(scale * electrons).is_finite()
        || l.iter().flatten().flatten().any(|v| !v.finite())
    {
        return Err(DftError::InvalidInput(
            "transport quadrature exceeds the finite floating-point range".into(),
        ));
    }
    Ok((l, scale * electrons))
}

fn inverse3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let scale = m.iter().flatten().fold(0.0_f64, |s, v| s.max(v.abs()));
    if !(det.abs() > 1e-13 * scale.powi(3)) {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, d) = ((i + 1) % 3, (i + 2) % 3);
            out[i][j] = (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / det;
        }
    }
    Some(out)
}

fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// Transport coefficients in atomic units (`e = hbar = m_e = k_B = 1`):
/// `sigma/tau = L0`, Seebeck `S = -(L0^-1 L1)/kT` (units of `k_B/e`) and
/// `kappa_e/tau = (L2 - L1 L0^-1 L1)/kT`. `S` and `kappa_e` are `None` when
/// `L0` is singular (no carriers in some direction); along a periodic
/// direction of an insulator the conductivity then vanishes exponentially.
#[derive(Clone, Debug)]
pub struct TransportCoefficients {
    pub conductivity_over_tau: [[f64; 3]; 3],
    pub seebeck: Option<[[f64; 3]; 3]>,
    pub thermal_conductivity_over_tau: Option<[[f64; 3]; 3]>,
}

pub fn transport_coefficients(l: &TransportIntegrals, kt: f64) -> TransportCoefficients {
    let inverse = inverse3(&l.l0);
    let seebeck = inverse.map(|inv| {
        let s = mul3(&inv, &l.l1);
        s.map(|row| row.map(|v| -v / kt))
    });
    let thermal = inverse.map(|inv| {
        let correction = mul3(&l.l1, &mul3(&inv, &l.l1));
        std::array::from_fn(|i| std::array::from_fn(|j| (l.l2[i][j] - correction[i][j]) / kt))
    });
    TransportCoefficients {
        conductivity_over_tau: l.l0,
        seebeck,
        thermal_conductivity_over_tau: thermal,
    }
}

/// SI factor of `sigma/tau`: `e^2 E_h/(hbar^2 a0)` in S/(m s).
pub fn conductivity_factor() -> f64 {
    use crate::constants::*;
    ELEMENTARY_CHARGE_C * ELEMENTARY_CHARGE_C * HARTREE_TO_JOULE
        / (HBAR_J_S * HBAR_J_S * BOHR_TO_METER)
}

/// SI factor of the Seebeck coefficient: `k_B/e` in V/K.
pub fn seebeck_factor() -> f64 {
    crate::constants::BOLTZMANN_J_PER_K / crate::constants::ELEMENTARY_CHARGE_C
}

/// SI factor of `kappa_e/tau`: `E_h^2 k_B/(hbar^2 a0)` in W/(m K s).
pub fn thermal_conductivity_factor() -> f64 {
    use crate::constants::*;
    HARTREE_TO_JOULE * HARTREE_TO_JOULE * BOLTZMANN_J_PER_K / (HBAR_J_S * HBAR_J_S * BOHR_TO_METER)
}

/// Cartesian axes `[current, electric field, magnetic field]`, all x,y,z.
pub type HallTensor = [[[f64; 3]; 3]; 3];

/// One isolated band entering the ordinary Hall sum (atomic units).
#[derive(Clone, Debug)]
pub struct HallState {
    pub energy: f64,
    pub velocity: [f64; 3],
    /// Analytic `d2 energy / dk_a dk_b`; no mass inversion is needed.
    pub inverse_mass: [[f64; 3]; 3],
}

/// Electron Lorentz vertex `v_a (v x e_c).grad_k v_b`. The electron charge
/// is -1; this convention gives `H_xyz = -n/m^2` for a parabolic band.
/// No antisymmetric projection is applied to this single-state quantity.
pub fn hall_vertex(velocity: &[f64; 3], inverse_mass: &[[f64; 3]; 3]) -> HallTensor {
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            std::array::from_fn(|c| {
                let (d, e) = ((c + 1) % 3, (c + 2) % 3);
                velocity[a] * (inverse_mass[b][d] * velocity[e] - inverse_mass[b][e] * velocity[d])
            })
        })
    })
}

/// `sigma_ab(B) = tau L0_ab + tau^2 sum_c B_c H_abc + O(B^2)`.
/// The raw Onsager defect measures the k integration, without symmetrizing H.
#[derive(Clone, Debug)]
pub struct HallIntegrals {
    pub conductivity_derivative_over_tau_squared: HallTensor,
    pub raw_onsager_residual: f64,
    /// `max|H_abc + H_bac| / max|H_abc|`; None for an identically zero H.
    pub relative_onsager_residual: Option<f64>,
}

/// Ordinary Hall conductivity derivative for constant relaxation time,
/// rigid isolated bands, finite positive temperature and electron charge -1.
/// The k mesh, volume, spin weight, chemical potential and window must match
/// those used in [`transport_integrals`]. Only the Fermi-window states need
/// to be supplied. Finite differences are not used in this calculation.
pub fn hall_integrals(
    states: &[HallState],
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
) -> Result<HallIntegrals> {
    Ok(hall_diagnostics(weighted_hall_integrals(
        states,
        volume,
        nk,
        g,
        mu,
        kt,
        window,
        &|_| Ok(1.0),
    )?))
}

pub(crate) fn weighted_hall_integrals<T: TransportWeight>(
    states: &[HallState],
    volume: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    window: f64,
    weight: &(impl Fn(f64) -> Result<T> + Sync),
) -> Result<[[[T; 3]; 3]; 3]> {
    validate_transport(volume, nk, g, mu, kt, window)?;
    let weights = transport_weights(
        &states.iter().map(|s| s.energy).collect::<Vec<_>>(),
        mu,
        kt,
        window,
        weight,
    )?;
    let mut h = [[[T::default(); 3]; 3]; 3];
    for (s, lifetime_squared) in states.iter().zip(weights) {
        if !s.energy.is_finite()
            || !s
                .velocity
                .iter()
                .chain(s.inverse_mass.iter().flatten())
                .all(|x| x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "Hall band energies, velocities and curvatures must be finite".into(),
            ));
        }
        let x = (s.energy - mu) / kt;
        if x.abs() > window {
            continue;
        }
        let weight = g / (volume * nk as f64) * fermi(x).1 / kt;
        let vertex = hall_vertex(&s.velocity, &s.inverse_mass);
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    h[a][b][c] += lifetime_squared * (weight * vertex[a][b][c]);
                }
            }
        }
    }
    if h.iter().flatten().flatten().any(|v| !v.finite()) {
        return Err(DftError::InvalidInput(
            "Hall quadrature exceeds the finite floating-point range".into(),
        ));
    }
    Ok(h)
}

pub(crate) fn hall_diagnostics(h: HallTensor) -> HallIntegrals {
    let mut residual = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                residual = residual.max((h[a][b][c] + h[b][a][c]).abs());
                scale = scale.max(h[a][b][c].abs());
            }
        }
    }
    HallIntegrals {
        conductivity_derivative_over_tau_squared: h,
        raw_onsager_residual: residual,
        relative_onsager_residual: if scale > 0.0 {
            Some(residual / scale)
        } else {
            None
        },
    }
}

/// Hall coefficient `R_abc = d rho_ab / d B_c = -L0^-1 H_c L0^-1`.
/// Thus `E_a = rho_ab j_b` and the usual electron Hall coefficient is
/// `R_yxz = -1/n` for a parabolic band. Tau cancels. A singular zero-field
/// conductivity returns None; no inverse floor or pseudoinverse is applied.
pub fn hall_coefficient(l: &TransportIntegrals, hall: &HallIntegrals) -> Option<HallTensor> {
    inverse3(&l.l0).map(|inverse| {
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                std::array::from_fn(|c| {
                    -(0..3)
                        .map(|u| {
                            (0..3)
                                .map(|v| {
                                    inverse[a][u]
                                        * hall.conductivity_derivative_over_tau_squared[u][v][c]
                                        * inverse[v][b]
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
            })
        })
    })
}

/// SI factor of `d sigma/dB / tau^2`: `e^3 E_h^2 a0 / hbar^4`,
/// in S/(m T s^2).
pub fn hall_conductivity_factor() -> f64 {
    use crate::constants::*;
    ELEMENTARY_CHARGE_C.powi(3) * HARTREE_TO_JOULE.powi(2) * BOHR_TO_METER / HBAR_J_S.powi(4)
}

/// SI factor of `d rho/dB`: `a0^3/e`, in m^3/C (equivalently ohm m/T).
pub fn hall_coefficient_factor() -> f64 {
    crate::constants::BOHR_TO_METER.powi(3) / crate::constants::ELEMENTARY_CHARGE_C
}

/// Retarded intraband conductivity and ordinary weak-field Hall response.
/// Values are in atomic units, with actual relaxation time included.
#[derive(Clone, Debug)]
pub struct FrequencyTransport {
    pub conductivity: [[crate::response::complex::Complex; 3]; 3],
    pub hall_conductivity_derivative: Option<[[[crate::response::complex::Complex; 3]; 3]; 3]>,
}

/// Solve the constant-tau kinetic equation with the convention exp(-i omega t).
/// `omega` is a signed angular frequency in Hartree/hbar and `tau` is in
/// hbar/Hartree. The poles are at -i/tau; interband coherence is a separate
/// response model. No static or frequency-dependent denominator is floored.
pub fn frequency_transport(
    l: &TransportIntegrals,
    hall: Option<&HallIntegrals>,
    omega: f64,
    tau: f64,
) -> Result<FrequencyTransport> {
    use crate::response::complex::Complex;
    if !omega.is_finite()
        || !tau.is_finite()
        || tau <= 0.0
        || !(omega * tau).is_finite()
        || l.l0.iter().flatten().any(|x| !x.is_finite())
        || hall.is_some_and(|h| {
            h.conductivity_derivative_over_tau_squared
                .iter()
                .flatten()
                .flatten()
                .any(|x| !x.is_finite())
        })
    {
        return Err(DftError::InvalidInput("frequency transport requires finite angular frequency, positive finite tau and finite transport tensors".into()));
    }
    let propagator = Complex::from(tau) / Complex::new(1.0, -omega * tau);
    let conductivity = std::array::from_fn(|a| std::array::from_fn(|b| propagator * l.l0[a][b]));
    let hall_conductivity_derivative = hall.map(|h| {
        std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                std::array::from_fn(|c| {
                    propagator * (propagator * h.conductivity_derivative_over_tau_squared[a][b][c])
                })
            })
        })
    });
    let out = FrequencyTransport {
        conductivity,
        hall_conductivity_derivative,
    };
    if out.conductivity.iter().flatten().any(|x| !x.is_finite())
        || out
            .hall_conductivity_derivative
            .as_ref()
            .is_some_and(|h| h.iter().flatten().flatten().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "frequency transport tensor exceeds the floating-point range".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hall_degenerate_clusters_are_refused_only_inside_the_requested_window() {
        let identity = crate::linalg::Matrix::identity(2);
        let block = ImageBlock {
            translation: Vec3::zero(),
            matrix: identity,
        };
        let model = BandModel {
            fock: [vec![block.clone()], vec![block.clone()]],
            overlap: vec![block],
            spins: 1,
            fock_image_residual: 0.0,
        };
        assert!(matches!(
            model.hall_states(0, Vec3::zero(), 1e-6, |_| true),
            Err(DftError::Unsupported { .. })
        ));
        assert!(model
            .hall_states(0, Vec3::zero(), 1e-6, |e| e < 0.5)
            .unwrap()
            .is_empty());
        // Zero-field transport still retains the whole velocity block.
        assert_eq!(
            model
                .transport_states(0, Vec3::zero(), 1e-6, |_| true)
                .unwrap()[0]
                .count,
            2.0
        );
        assert!(model.hall_states(0, Vec3::zero(), 0.0, |_| false).is_err());
        assert!(model.hall_states(1, Vec3::zero(), 1e-6, |_| false).is_err());
    }

    #[test]
    fn hall_onsager_diagnostics_do_not_project_a_single_state() {
        let state = HallState {
            energy: 0.0,
            velocity: [1.0, 2.0, 3.0],
            inverse_mass: [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        };
        let hall = hall_integrals(&[state], 1.0, 1, 2.0, 0.0, 1.0, 20.0).unwrap();
        assert_eq!(hall.conductivity_derivative_over_tau_squared[0][0][2], 1.0);
        assert!(hall.raw_onsager_residual > 0.0);
        assert!(hall.relative_onsager_residual.unwrap() > 0.0);
    }

    /// Parabolic band `e = k^2/(2 m)` sampled on a cubic grid of spacing `dk`
    /// inside the sphere `e < e_max` (a continuum model; `volume nk` is the
    /// real-space volume per sampled state, `(2 pi)^3 / dk^3`).
    fn parabolic(m: f64, dk: f64, e_max: f64) -> (Vec<TransportState>, f64) {
        let k_max = (2.0 * m * e_max).sqrt();
        let n = (k_max / dk).ceil() as i64;
        let mut states = Vec::new();
        for i in -n..=n {
            for j in -n..=n {
                for l in -n..=n {
                    let k = [
                        (i as f64 + 0.5) * dk,
                        (j as f64 + 0.5) * dk,
                        (l as f64 + 0.5) * dk,
                    ];
                    let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                    let e = k2 / (2.0 * m);
                    if e > e_max {
                        continue;
                    }
                    let v = k.map(|x| x / m);
                    states.push(TransportState {
                        energy: e,
                        velocity2: std::array::from_fn(|a| std::array::from_fn(|b| v[a] * v[b])),
                        count: 1.0,
                    });
                }
            }
        }
        let volume_per_state = (2.0 * std::f64::consts::PI / dk).powi(3);
        (states, volume_per_state)
    }

    #[test]
    fn parabolic_band_reproduces_the_closed_form_transport_limits() {
        let m = 0.7;
        // Degenerate metal: sigma/tau = n/m, Wiedemann-Franz pi^2/3.
        let (mu, kt) = (0.02, 0.0005);
        let (states, volume) = parabolic(m, 0.004, mu + 40.0 * kt);
        let l = transport_integrals(&states, volume, 1, 2.0, mu, kt, 40.0).unwrap();
        let coefficients = transport_coefficients(&l, kt);
        let drude = l.electron_density / m;
        let sigma = coefficients.conductivity_over_tau[0][0];
        let lorenz = coefficients.thermal_conductivity_over_tau.unwrap()[0][0] / (sigma * kt);
        let lorenz_exact = std::f64::consts::PI.powi(2) / 3.0;
        println!(
            "degenerate: {} states, sigma/tau {sigma:.6e}, n/m {drude:.6e}, ratio \
                {:.6}; Lorenz {lorenz:.6} against {lorenz_exact:.6} (ratio {:.6}); \
                off-diagonal {:e}",
            states.len(),
            sigma / drude,
            lorenz / lorenz_exact,
            coefficients.conductivity_over_tau[0][1]
        );
        assert!((sigma / drude - 1.0).abs() < 0.01);
        assert!((lorenz / lorenz_exact - 1.0).abs() < 0.02);
        // Non-degenerate semiconductor: S = -(k_B/e)[(e_c - mu)/kT + 5/2].
        let (mu, kt) = (-0.01, 0.002);
        let (states, volume) = parabolic(m, 0.002, 30.0 * kt);
        let l = transport_integrals(&states, volume, 1, 2.0, mu, kt, 60.0).unwrap();
        let s = transport_coefficients(&l, kt).seebeck.unwrap()[1][1];
        let exact = -((0.0 - mu) / kt + 2.5);
        println!(
            "non-degenerate: Seebeck {s:.6} k_B/e against {exact:.6} (ratio {:.6})",
            s / exact
        );
        assert!((s / exact - 1.0).abs() < 0.02);
    }

    #[test]
    fn degenerate_clusters_are_grouped_transitively() {
        assert_eq!(
            clusters(&[0.0, 1e-8, 2e-8, 1.0], 1.5e-8),
            vec![vec![0, 1, 2], vec![3]]
        );
        let (f, df) = fermi(800.0);
        assert!(f == 0.0 && df == 0.0);
        let (f, df) = fermi(-800.0);
        assert!(f == 1.0 && df == 0.0);
    }

    #[test]
    fn si_factors_match_the_atomic_units() {
        // sigma a.u. = e^2/(hbar a0) per atomic time unit hbar/E_h.
        println!(
            "sigma/tau {:e} S/(m s), S {:e} V/K, kappa/tau {:e} W/(m K s)",
            conductivity_factor(),
            seebeck_factor(),
            thermal_conductivity_factor()
        );
        assert!((conductivity_factor() / 1.90163e23 - 1.0).abs() < 1e-4);
        assert!((seebeck_factor() / 8.617333262e-5 - 1.0).abs() < 1e-9);
    }
}
