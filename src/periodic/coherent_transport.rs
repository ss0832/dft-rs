// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Covariant independent-particle optical conductivity, intrinsic Hall and
//! non-Abelian integer quantum Hall response. These kernels accept physical
//! current vertices and occupied-subspace links from any functional provider.
//! A current includes the AO position connection; a derivative of the AO
//! Hamiltonian alone is insufficient in a moving, nonorthogonal Bloch basis.

use super::band_transport::BandModel;
use super::bloch::BlochMatrix;
use super::integrals::ImageBlock;
use super::scf::{PeriodicScfOptions, PeriodicScfResult};
use super::Cell;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Energies in Hartree, physical velocity matrices in Hartree Bohr, in the
/// orthonormal eigenstate basis. Exact degeneracies may have arbitrary bases.
#[derive(Clone, Debug)]
pub struct CoherentPoint {
    pub energies: Vec<f64>,
    pub velocities: [BlochMatrix; 3],
}

/// Converged physical Bloch currents, retaining AO connection terms.
pub struct CoherentModel {
    pub bands: BandModel,
    dipoles: [Vec<ImageBlock>; 3],
}
impl CoherentModel {
    pub fn new(
        cell: &Cell,
        functional: &ResolvedFunctional,
        scf: &PeriodicScfResult,
        options: &PeriodicScfOptions,
    ) -> Result<Self> {
        Ok(Self {
            bands: BandModel::new(cell, functional, scf, options)?,
            dipoles: super::dielectric::dipole_blocks(cell),
        })
    }
    pub fn point(&self, spin: usize, k: Vec3) -> Result<CoherentPoint> {
        self.bands.coherent_point(spin, k, &self.dipoles)
    }
}

/// Cross-k cell-periodic AO metric <u_basis(k)|u_basis(k+step)>.
/// Plane-wave images retain the orbital positions and reciprocal-boundary
/// sewing. A coefficient-only overlap would erase the physical AO connection.
pub struct CrossMetric {
    images: Vec<(Vec3, BlochMatrix)>,
}
impl CrossMetric {
    pub fn new(cell: &Cell, step: Vec3) -> Result<Self> {
        if !(0..3).all(|a| step.get(a).is_finite()) {
            return Err(DftError::InvalidInput(
                "cross-k metric step must be finite".into(),
            ));
        }
        let n = cell.basis.nao_sph;
        let images =
            super::integrals::short_range_images(&cell.basis, &cell.lattice, cell.overlap_cutoff)
                .into_iter()
                .map(|image| {
                    let t = image.translation;
                    let shifted = super::integrals::shifted(&cell.basis, t);
                    let mut out = BlochMatrix::zeros(n, n);
                    for (left, right) in image.shell_pairs {
                        let (left, right) = (&cell.basis.shells[left], &shifted.shells[right]);
                        let (re, im) =
                            super::planewave::plane_wave_shell_pair(left, right, step * (-1.0));
                        for i in 0..re.rows {
                            for j in 0..re.cols {
                                out.re[(left.sph_off + i, right.sph_off + j)] = re[(i, j)];
                                out.im[(left.sph_off + i, right.sph_off + j)] = im[(i, j)];
                            }
                        }
                    }
                    (t, out)
                })
                .collect();
        Ok(Self { images })
    }
    /// `right_k` must be the unwrapped k+step, including boundary sewing.
    pub fn at(&self, right_k: Vec3) -> BlochMatrix {
        let n = self.images.first().map_or(0, |(_, m)| m.rows());
        let mut out = BlochMatrix::zeros(n, n);
        for (t, m) in &self.images {
            let theta = right_k.dot(*t);
            let (s, c) = theta.sin_cos();
            for i in 0..n {
                for j in 0..n {
                    out.re[(i, j)] += c * m.re[(i, j)] - s * m.im[(i, j)];
                    out.im[(i, j)] += s * m.re[(i, j)] + c * m.im[(i, j)];
                }
            }
        }
        out
    }
}

/// Positive-gap slice of a three-dimensional Bloch reference. A slice
/// conductance is a 2D topological invariant, not a 3D bulk conductivity.
pub fn quantum_hall_slice(
    cell: &Cell,
    model: &BandModel,
    spin: usize,
    occupied: usize,
    mesh: [usize; 2],
    axes: [usize; 2],
    fixed_k: f64,
) -> Result<(QuantumHall, f64)> {
    if axes[0] >= 3
        || axes[1] >= 3
        || axes[0] == axes[1]
        || mesh.iter().any(|n| *n < 2)
        || !fixed_k.is_finite()
        || occupied == 0
    {
        return Err(DftError::InvalidInput(
            "quantum Hall slice requires distinct lattice axes 0..2, a mesh >=2, \
            finite fixed k and a non-empty occupied multiplet"
                .into(),
        ));
    }
    let reciprocal = cell.lattice.reciprocal();
    let fixed_axis = (0..3).find(|a| !axes.contains(a)).unwrap();
    let point = |x: usize, y: usize| {
        reciprocal[axes[0]] * (x as f64 / mesh[0] as f64)
            + reciprocal[axes[1]] * (y as f64 / mesh[1] as f64)
            + reciprocal[fixed_axis] * fixed_k
    };
    let mut orbitals = Vec::new();
    let mut gap = f64::INFINITY;
    let mut occupied_max = f64::NEG_INFINITY;
    let mut unoccupied_min = f64::INFINITY;
    for y in 0..mesh[1] {
        for x in 0..mesh[0] {
            let bands = model.bands(spin, point(x, y))?;
            if occupied >= bands.energies.len() {
                return Err(DftError::InvalidInput(
                    "quantum Hall slice requires an unoccupied complement in the retained basis"
                        .into(),
                ));
            }
            gap = gap.min(bands.energies[occupied] - bands.energies[occupied - 1]);
            occupied_max = occupied_max.max(bands.energies[occupied - 1]);
            unoccupied_min = unoccupied_min.min(bands.energies[occupied]);
            orbitals.push(bands.coefficients);
        }
    }
    gap = gap.min(unoccupied_min - occupied_max);
    if gap <= 0.0 {
        return Err(DftError::InvalidInput(
            "occupied multiplet has no common insulating chemical-potential gap: a \
            quantized Hall conductance requires a positive direct and indirect gap \
            over the slice"
                .into(),
        ));
    }
    let mut links = [Vec::new(), Vec::new()];
    for a in 0..2 {
        let step = reciprocal[axes[a]] * (1.0 / mesh[a] as f64);
        let cross = CrossMetric::new(cell, step)?;
        for y in 0..mesh[1] {
            for x in 0..mesh[0] {
                let i = y * mesh[0] + x;
                let neighbor = if a == 0 {
                    y * mesh[0] + (x + 1) % mesh[0]
                } else {
                    ((y + 1) % mesh[1]) * mesh[0] + x
                };
                let metric = cross.at(point(x, y) + step);
                let left = &orbitals[i];
                let right = &orbitals[neighbor];
                let mut link = BlochMatrix::zeros(occupied, occupied);
                for m in 0..occupied {
                    for n in 0..occupied {
                        let mut z = Complex::ZERO;
                        for p in 0..left.rows() {
                            for q in 0..right.rows() {
                                z += entry(left, p, m).conj()
                                    * entry(&metric, p, q)
                                    * entry(right, q, n);
                            }
                        }
                        link.re[(m, n)] = z.re;
                        link.im[(m, n)] = z.im;
                    }
                }
                links[a].push(link);
            }
        }
    }
    Ok((quantum_hall_from_links(&links, mesh)?, gap))
}

/// Hybrid Wannier charge centres along one reciprocal axis as a function of
/// a transverse wave vector.
#[derive(Clone, Debug)]
pub struct WannierFlow {
    /// Reduced coordinate of the transverse wave vector of every flow point.
    pub flow: Vec<f64>,
    /// Centres `[flow point][band]` in units of the lattice vector of the
    /// loop axis, in `(-1/2, 1/2]`, ascending.
    pub centres: Vec<Vec<f64>>,
    /// Sum of the centres of every flow point, continued along the flow
    /// without jumps.
    pub total: Vec<f64>,
    /// Winding of the total over the closed flow, without integer rounding:
    /// the Chern number of the slice.
    pub winding: f64,
    pub min_link_log_abs_determinant: f64,
    /// Smallest direct gap of the occupied multiplet on the sampled points.
    pub sampled_min_gap: f64,
}

/// Phases of the eigenvalues of the unitary part of `w`. The Hermitian
/// matrices `(U + U^H) / 2` and `(U - U^H) / (2 i)` of a unitary `U` commute,
/// so the eigenvectors of a generic real combination of the two are their
/// common eigenvectors, and the phase is the angle of the two expectation
/// values.
fn eigenphases(w: &BlochMatrix) -> Result<Vec<f64>> {
    use super::band_transport::hermitian_eigen;
    use super::phonon::{adjoint, product};
    let n = w.rows();
    // Polar decomposition: U = W (W^H W)^(-1/2).
    let mut gram = product(&adjoint(w), w);
    gram.make_hermitian();
    let (values, vectors) = hermitian_eigen(&gram)?;
    if values.iter().any(|v| !(*v > 0.0)) {
        return Err(DftError::InvalidInput(
            "singular Wilson loop: refine the loop mesh and verify the gap".into(),
        ));
    }
    let mut scaled = vectors.clone();
    for j in 0..n {
        let factor = 1.0 / values[j].sqrt();
        for i in 0..n {
            scaled.re[(i, j)] *= factor;
            scaled.im[(i, j)] *= factor;
        }
    }
    let unitary = product(w, &product(&scaled, &adjoint(&vectors)));
    let conjugate = adjoint(&unitary);
    let mut cosine = unitary.clone();
    cosine.add_scaled(&conjugate, 1.0);
    cosine.scale(0.5);
    // (U - U^H) / (2 i): (a + i b) / (2 i) = (b - i a) / 2.
    let mut difference = unitary.clone();
    difference.add_scaled(&conjugate, -1.0);
    let mut sine = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sine.re[(i, j)] = 0.5 * difference.im[(i, j)];
            sine.im[(i, j)] = -0.5 * difference.re[(i, j)];
        }
    }
    let mut mixed = cosine.clone();
    mixed.add_scaled(&sine, 0.618_033_988_749_895);
    mixed.make_hermitian();
    let (_, common) = hermitian_eigen(&mixed)?;
    let expectation = |m: &BlochMatrix, column: usize| -> f64 {
        let mut value = Complex::ZERO;
        for i in 0..n {
            for j in 0..n {
                value +=
                    entry(&common, i, column).conj() * entry(m, i, j) * entry(&common, j, column);
            }
        }
        value.re
    };
    Ok((0..n)
        .map(|column| expectation(&sine, column).atan2(expectation(&cosine, column)))
        .collect())
}

/// Hybrid Wannier charge centres of the occupied multiplet: minus the
/// eigenphases over `2 pi` of the product of the occupied-subspace links
/// `<u_k|u_(k+b)>` around the closed loop along the lattice axis `axes[0]`
/// (the Berry phase of a band is `-Im ln` of its loop product, and its
/// centre is the Berry phase over `2 pi`), for `flow_points`
/// transverse wave vectors along `axes[1]`. The winding of their sum over
/// the closed flow is the Chern number of the slice.
#[allow(clippy::too_many_arguments)]
pub fn wannier_flow(
    cell: &Cell,
    model: &BandModel,
    spin: usize,
    occupied: usize,
    loop_divisions: usize,
    flow_points: usize,
    axes: [usize; 2],
    fixed_k: f64,
) -> Result<WannierFlow> {
    use super::phonon::product;
    if axes[0] >= 3
        || axes[1] >= 3
        || axes[0] == axes[1]
        || loop_divisions < 2
        || flow_points < 2
        || !fixed_k.is_finite()
        || occupied == 0
    {
        return Err(DftError::InvalidInput(
            "the Wannier flow requires distinct lattice axes 0..2, at least two loop divisions \
             and flow points, finite fixed k and a non-empty occupied multiplet"
                .into(),
        ));
    }
    let reciprocal = cell.lattice.reciprocal();
    let fixed_axis = (0..3).find(|a| !axes.contains(a)).unwrap();
    let step = reciprocal[axes[0]] * (1.0 / loop_divisions as f64);
    let cross = CrossMetric::new(cell, step)?;
    let mut flow = Vec::with_capacity(flow_points);
    let mut centres = Vec::with_capacity(flow_points);
    let mut min_log = f64::INFINITY;
    let mut gap = f64::INFINITY;
    for y in 0..flow_points {
        let transverse = y as f64 / flow_points as f64;
        let point = |x: usize| {
            reciprocal[axes[0]] * (x as f64 / loop_divisions as f64)
                + reciprocal[axes[1]] * transverse
                + reciprocal[fixed_axis] * fixed_k
        };
        let mut orbitals = Vec::with_capacity(loop_divisions);
        for x in 0..loop_divisions {
            let bands = model.bands(spin, point(x))?;
            if occupied >= bands.energies.len() {
                return Err(DftError::InvalidInput(
                    "the Wannier flow requires an unoccupied complement in the retained basis"
                        .into(),
                ));
            }
            gap = gap.min(bands.energies[occupied] - bands.energies[occupied - 1]);
            orbitals.push(bands.coefficients);
        }
        let mut wilson: Option<BlochMatrix> = None;
        for x in 0..loop_divisions {
            let metric = cross.at(point(x) + step);
            let left = &orbitals[x];
            let right = &orbitals[(x + 1) % loop_divisions];
            let mut link = BlochMatrix::zeros(occupied, occupied);
            for m in 0..occupied {
                for n in 0..occupied {
                    let mut z = Complex::ZERO;
                    for p in 0..left.rows() {
                        for q in 0..right.rows() {
                            z += entry(left, p, m).conj()
                                * entry(&metric, p, q)
                                * entry(right, q, n);
                        }
                    }
                    link.re[(m, n)] = z.re;
                    link.im[(m, n)] = z.im;
                }
            }
            let (_, log) = determinant_phase(&link)?;
            min_log = min_log.min(log);
            wilson = Some(match wilson {
                None => link,
                Some(w) => product(&w, &link),
            });
        }
        let wilson = wilson.expect("at least two loop divisions");
        let mut row: Vec<f64> = eigenphases(&wilson)?
            .into_iter()
            .map(|phase| -phase / std::f64::consts::TAU)
            .collect();
        row.sort_by(|a, b| a.total_cmp(b));
        centres.push(row);
        flow.push(transverse);
    }
    if gap <= 0.0 {
        return Err(DftError::InvalidInput(
            "the occupied multiplet has no positive gap on the sampled wave vectors".into(),
        ));
    }
    // Continue the sum of the centres along the flow: every step is brought
    // into (-1/2, 1/2], and the closing step gives the winding.
    let sums: Vec<f64> = centres.iter().map(|row| row.iter().sum()).collect();
    let wrapped = |x: f64| x - x.round();
    let mut total = Vec::with_capacity(flow_points);
    total.push(sums[0]);
    for y in 1..flow_points {
        let previous = total[y - 1];
        total.push(previous + wrapped(sums[y] - sums[y - 1]));
    }
    let winding = total[flow_points - 1] + wrapped(sums[0] - sums[flow_points - 1]) - total[0];
    Ok(WannierFlow {
        flow,
        centres,
        total,
        winding,
        min_link_log_abs_determinant: min_log,
        sampled_min_gap: gap,
    })
}

fn entry(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}

impl CoherentPoint {
    /// Raw Hermitian defect, before any projection or basis change.
    pub fn hermitian_residual(&self) -> Result<f64> {
        let n = self.energies.len();
        if n == 0
            || self.energies.iter().any(|e| !e.is_finite())
            || self
                .velocities
                .iter()
                .any(|v| v.rows() != n || v.cols() != n)
        {
            return Err(DftError::InvalidInput(
                "coherent point requires non-empty finite energies and three square \
                current matrices of the same dimension"
                    .into(),
            ));
        }
        let mut residual = 0.0_f64;
        for v in &self.velocities {
            for i in 0..n {
                for j in 0..n {
                    let z = entry(v, i, j);
                    if !z.is_finite() {
                        return Err(DftError::InvalidInput(
                            "coherent current vertices must be finite".into(),
                        ));
                    }
                    residual = residual.max((z - entry(v, j, i).conj()).abs());
                }
            }
        }
        Ok(residual)
    }
}

fn fermi(x: f64) -> f64 {
    if x >= 0.0 {
        let t = (-x).exp();
        t / (1.0 + t)
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Positive occupation divided difference (f(e1)-f(e2))/(e2-e1).
/// `expm1` resolves close levels without subtracting nearly equal occupations.
/// Only exactly equal energies use the analytic -f' limit: close physical
/// splittings and the retarded denominator are never replaced by a tolerance.
pub fn occupation_divided_difference(e1: f64, e2: f64, mu: f64, kt: f64) -> Result<f64> {
    if [e1, e2, mu, kt].iter().any(|x| !x.is_finite()) || kt <= 0.0 {
        return Err(DftError::InvalidInput(
            "occupation divided differences require finite energies and positive finite kT".into(),
        ));
    }
    let (lo, hi) = (e1.min(e2), e1.max(e2));
    let x = (lo - mu) / kt;
    let y = (hi - mu) / kt;
    let de = hi - lo;
    if de == 0.0 {
        let u = (-x.abs()).exp();
        let value = u / ((1.0 + u) * (1.0 + u) * kt);
        if !value.is_finite() {
            return Err(DftError::InvalidInput(
                "degenerate occupation derivative exceeds finite floating-point range; \
                use a resolved positive temperature"
                    .into(),
            ));
        }
        return Ok(value);
    }
    let d = de / kt;
    let difference = if y <= 0.0 {
        let u = y.exp();
        u * (-(-d).exp_m1()) / ((1.0 + u) * (1.0 + u * (-d).exp()))
    } else if x >= 0.0 || x > -350.0 {
        let u = (-x).exp();
        u * (-(-d).exp_m1()) / ((1.0 + u) * (1.0 + u * (-d).exp()))
    } else {
        fermi(x) - fermi(y)
    };
    let value = difference / de;
    if !value.is_finite() {
        return Err(DftError::InvalidInput(
            "occupation divided difference exceeds the floating-point range".into(),
        ));
    }
    Ok(value)
}

/// Atomic-unit retarded conductivity, including every intra/interband pair.
/// `sigma_ab=i g/(measure Nk) sum_nm w_nm v_a,nm v_b,mn /
/// (omega+e_n-e_m+i eta)`, exp(-i omega t). The dimension of the physical
/// periodic measure determines the units; eta is a physical scattering rate.
#[derive(Clone, Debug)]
pub struct CoherentConductivity {
    pub conductivity: [[Complex; 3]; 3],
    pub intraband: [[Complex; 3]; 3],
    pub interband: [[Complex; 3]; 3],
    pub antisymmetric: [[Complex; 3]; 3],
    pub current_hermitian_residual: f64,
    pub equation_residual: f64,
}

pub fn conductivity(
    points: &[CoherentPoint],
    measure: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
    omega: f64,
    eta: f64,
) -> Result<CoherentConductivity> {
    if points.is_empty()
        || nk == 0
        || [measure, g, mu, kt, omega, eta]
            .iter()
            .any(|x| !x.is_finite())
        || measure <= 0.0
        || g <= 0.0
        || kt <= 0.0
        || eta <= 0.0
    {
        return Err(DftError::InvalidInput(
            "coherent conductivity requires points, a positive physical measure, k \
            count, spin weight, kT and retarded scattering rate"
                .into(),
        ));
    }
    let mut intra = [[Complex::ZERO; 3]; 3];
    let mut inter = intra;
    let mut current_residual = 0.0_f64;
    let mut equation_residual = 0.0_f64;
    let scale = g / (measure * nk as f64);
    for point in points {
        let residual = point.hermitian_residual()?;
        current_residual = current_residual.max(residual);
        let norm = point
            .velocities
            .iter()
            .flat_map(|v| v.re.as_slice().iter().chain(v.im.as_slice()))
            .fold(0.0_f64, |a, x| a.max(x.abs()));
        if residual > 1e-8 * norm {
            return Err(DftError::InvalidInput(format!(
                "physical current vertex is not Hermitian: raw defect {residual:e}, scale {norm:e}"
            )));
        }
        for (n, en) in point.energies.iter().enumerate() {
            for (m, em) in point.energies.iter().enumerate() {
                let w = occupation_divided_difference(*en, *em, mu, kt)?;
                let denominator = Complex::new(omega + en - em, eta);
                for a in 0..3 {
                    for b in 0..3 {
                        let source = Complex::new(0.0, w)
                            * entry(&point.velocities[a], n, m)
                            * entry(&point.velocities[b], m, n);
                        let solved = source / denominator;
                        equation_residual =
                            equation_residual.max((denominator * solved - source).abs());
                        // Intraband means the entire exactly degenerate block;
                        // splitting it by diagonal indices would violate covariance.
                        if en == em {
                            intra[a][b] += solved * scale;
                        } else {
                            inter[a][b] += solved * scale;
                        }
                    }
                }
            }
        }
    }
    let total = std::array::from_fn(|a| std::array::from_fn(|b| intra[a][b] + inter[a][b]));
    let antisymmetric =
        std::array::from_fn(|a| std::array::from_fn(|b| (total[a][b] - total[b][a]) * 0.5));
    if total.iter().flatten().any(|z| !z.is_finite()) {
        return Err(DftError::InvalidInput(
            "coherent conductivity exceeds the floating-point range".into(),
        ));
    }
    Ok(CoherentConductivity {
        conductivity: total,
        intraband: intra,
        interband: inter,
        antisymmetric,
        current_hermitian_residual: current_residual,
        equation_residual,
    })
}

/// Gauge-invariant Chern sum of a constant-rank occupied multiplet.
/// Links are physical cross-k overlaps, not coefficient-only AO overlaps.
#[derive(Clone, Debug)]
pub struct QuantumHall {
    /// Sum of principal plaquette phases/(2 pi), without integer rounding.
    pub chern_number: f64,
    pub integer_residual: f64,
    pub max_plaquette_phase: f64,
    /// A vanishing determinant invalidates the mesh; logarithms avoid floors.
    pub min_link_log_abs_determinant: f64,
    pub plaquette_phases: Vec<f64>,
}

pub(crate) fn determinant_phase(m: &BlochMatrix) -> Result<(Complex, f64)> {
    let n = m.rows();
    if n == 0 || m.cols() != n {
        return Err(DftError::InvalidInput(
            "occupied links must be non-empty square matrices of constant rank".into(),
        ));
    }
    let mut a: Vec<Vec<Complex>> = (0..n)
        .map(|i| (0..n).map(|j| entry(m, i, j)).collect())
        .collect();
    if a.iter().flatten().any(|z| !z.is_finite()) {
        return Err(DftError::InvalidInput(
            "occupied links must be finite".into(),
        ));
    }
    let mut phase = Complex::ONE;
    let mut log_abs = 0.0;
    for i in 0..n {
        let pivot = (i..n)
            .max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))
            .unwrap();
        let magnitude = a[pivot][i].abs();
        if magnitude == 0.0 {
            return Err(DftError::InvalidInput(
                "singular occupied link: refine the mesh and verify the \
                occupied/unoccupied gap; no determinant floor is applied"
                    .into(),
            ));
        }
        if pivot != i {
            a.swap(i, pivot);
            phase = -phase;
        }
        let diagonal = a[i][i];
        phase = phase * (diagonal / diagonal.abs());
        log_abs += diagonal.abs().ln();
        for row in i + 1..n {
            let factor = a[row][i] / diagonal;
            for col in i + 1..n {
                a[row][col] = a[row][col] - factor * a[i][col];
            }
        }
    }
    if !phase.is_finite() || !log_abs.is_finite() {
        return Err(DftError::InvalidInput(
            "occupied determinant exceeds the floating-point range".into(),
        ));
    }
    Ok((phase / phase.abs(), log_abs))
}

/// Non-Abelian lattice-gauge Chern number (x fastest, periodic mesh).
/// Degeneracies inside the occupied multiplet are allowed. The provider must
/// supply a positive spectral gap to its complement at every mesh point.
pub fn quantum_hall_from_links(
    links: &[Vec<BlochMatrix>; 2],
    mesh: [usize; 2],
) -> Result<QuantumHall> {
    let count = mesh[0]
        .checked_mul(mesh[1])
        .ok_or_else(|| DftError::InvalidInput("Chern mesh size overflow".into()))?;
    if mesh.iter().any(|n| *n < 2) || links.iter().any(|v| v.len() != count) {
        return Err(DftError::InvalidInput(
            "Chern mesh requires two dimensions >= 2 and a link in each direction at every point"
                .into(),
        ));
    }
    let rank = links[0][0].rows();
    if links
        .iter()
        .flatten()
        .any(|l| l.rows() != rank || l.cols() != rank)
    {
        return Err(DftError::InvalidInput(
            "occupied links must have constant rank over the entire mesh".into(),
        ));
    }
    let mut phases = [Vec::new(), Vec::new()];
    let mut min_log = f64::INFINITY;
    for axis in 0..2 {
        for link in &links[axis] {
            let (z, l) = determinant_phase(link)?;
            phases[axis].push(z);
            min_log = min_log.min(l);
        }
    }
    let mut flux = Vec::with_capacity(count);
    for y in 0..mesh[1] {
        for x in 0..mesh[0] {
            let i = y * mesh[0] + x;
            let ix = y * mesh[0] + (x + 1) % mesh[0];
            let iy = ((y + 1) % mesh[1]) * mesh[0] + x;
            let z = phases[0][i] * phases[1][ix] * phases[0][iy].conj() * phases[1][i].conj();
            flux.push(z.im.atan2(z.re));
        }
    }
    let chern = flux.iter().sum::<f64>() / (2.0 * std::f64::consts::PI);
    Ok(QuantumHall {
        chern_number: chern,
        integer_residual: (chern - chern.round()).abs(),
        max_plaquette_phase: flux.iter().fold(0.0_f64, |a, x| a.max(x.abs())),
        min_link_log_abs_determinant: min_log,
        plaquette_phases: flux,
    })
}

/// SI conversion of atomic-unit conductivity per d-dimensional cell measure:
/// e^2/(hbar a0^(d-2)), giving S*m (1D), S (2D) or S/m (3D).
pub fn conductivity_factor(dimension: usize) -> Result<f64> {
    if !(1..=3).contains(&dimension) {
        return Err(DftError::InvalidInput(
            "conductivity dimension must be 1, 2 or 3".into(),
        ));
    }
    use crate::constants::*;
    Ok(ELEMENTARY_CHARGE_C.powi(2) / HBAR_J_S * BOHR_TO_METER.powi(2 - dimension as i32))
}

/// Conductance of a filled two-dimensional multiplet: C e^2/h, per spin.
pub fn quantum_conductance(chern: f64) -> f64 {
    use crate::constants::*;
    chern * ELEMENTARY_CHARGE_C.powi(2) / (2.0 * std::f64::consts::PI * HBAR_J_S)
}

/// Berry curvature dipole of the Fermi surface.
#[derive(Clone, Debug)]
pub struct BerryDipole {
    /// `D_ab = int d3k/(2 pi)^3 sum_n (-df/de) v_a Omega_b`, the
    /// Fermi-surface form of `int f d Omega_b / dk_a`; dimensionless in
    /// three dimensions.
    pub dipole: [[f64; 3]; 3],
    /// `int d3k/(2 pi)^3 sum_n (-df/de)`: density of states at the chemical
    /// potential per Hartree and unit of the periodic measure.
    pub fermi_weight: f64,
    /// Electrons per cell at the chemical potential.
    pub electrons: f64,
    pub current_hermitian_residual: f64,
}

/// Berry curvature dipole from the physical velocity matrices. Every
/// exactly degenerate level is one multiplet: its curvature is the matrix
/// `Omega_b = i eps_bcd sum_m v_c |m><m| v_d / (e - e_m)^2` over the other
/// levels, and the level contributes `-f'(e) Tr[v_a Omega_b]`, which does
/// not depend on the basis chosen inside the multiplet. `points` holds the
/// k points of every spin channel, `nk` the number of k points, `measure`
/// the cell volume and `g` the spin weight.
pub fn berry_curvature_dipole(
    points: &[CoherentPoint],
    measure: f64,
    nk: usize,
    g: f64,
    mu: f64,
    kt: f64,
) -> Result<BerryDipole> {
    if points.is_empty()
        || nk == 0
        || [measure, g, mu, kt].iter().any(|x| !x.is_finite())
        || measure <= 0.0
        || g <= 0.0
        || kt <= 0.0
    {
        return Err(DftError::InvalidInput(
            "the Berry curvature dipole requires points, a positive physical measure, k count, \
             spin weight and temperature"
                .into(),
        ));
    }
    let scale = g / (measure * nk as f64);
    let mut dipole = [[0.0; 3]; 3];
    let mut fermi_weight = 0.0;
    let mut electrons = 0.0;
    let mut current_residual = 0.0_f64;
    for point in points {
        current_residual = current_residual.max(point.hermitian_residual()?);
        let v = &point.velocities;
        let n = point.energies.len();
        for i in 0..n {
            let e = point.energies[i];
            let f = fermi((e - mu) / kt);
            electrons += g * f / nk as f64;
            let slope = f * (1.0 - f) / kt;
            fermi_weight += scale * slope;
            if slope < 1e-300 {
                continue;
            }
            for j in 0..n {
                if point.energies[j] != e {
                    continue;
                }
                for b in 0..3 {
                    let (c, d) = ((b + 1) % 3, (b + 2) % 3);
                    let mut sum = Complex::ZERO;
                    for m in 0..n {
                        let gap = e - point.energies[m];
                        if gap == 0.0 {
                            continue;
                        }
                        sum += (entry(&v[c], i, m) * entry(&v[d], m, j)
                            - entry(&v[d], i, m) * entry(&v[c], m, j))
                            * (1.0 / (gap * gap));
                    }
                    let omega = Complex::new(0.0, 1.0) * sum;
                    for (a, row) in dipole.iter_mut().enumerate() {
                        row[b] += scale * slope * (entry(&v[a], j, i) * omega).re;
                    }
                }
            }
        }
    }
    if dipole.iter().flatten().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "the Berry curvature dipole exceeds the floating-point range".into(),
        ));
    }
    Ok(BerryDipole {
        dipole,
        fermi_weight,
        electrons,
        current_hermitian_residual: current_residual,
    })
}

/// Chemical potential at which the Fermi-Dirac occupations of `points` hold
/// `electrons` electrons per cell.
pub fn doped_chemical_potential(
    points: &[CoherentPoint],
    nk: usize,
    g: f64,
    electrons: f64,
    kt: f64,
) -> Result<f64> {
    if points.is_empty() || nk == 0 || !(kt > 0.0) || !electrons.is_finite() || electrons < 0.0 {
        return Err(DftError::InvalidInput(
            "the doped chemical potential requires points, a positive temperature and a \
             non-negative electron count"
                .into(),
        ));
    }
    let count = |mu: f64| -> f64 {
        points
            .iter()
            .flat_map(|p| p.energies.iter())
            .map(|e| g * fermi((e - mu) / kt) / nk as f64)
            .sum()
    };
    let lowest = points
        .iter()
        .flat_map(|p| p.energies.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let highest = points
        .iter()
        .flat_map(|p| p.energies.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let (mut lo, mut hi) = (lowest - 60.0 * kt - 1.0, highest + 60.0 * kt + 1.0);
    if count(hi) < electrons {
        return Err(DftError::InvalidInput(
            "the bands cannot hold the requested number of electrons".into(),
        ));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count(mid) < electrons {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 4.0 * f64::EPSILON * mid.abs().max(1.0) {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}
