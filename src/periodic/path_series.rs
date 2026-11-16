// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Taylor coefficients of the integrals of a Gamma-point cell along a path
//! of its atoms and of its lattice: every centre, with all its lattice
//! images, moves as `F(t) (R + T + u(t))`.
//!
//! With the Bloch sums at Gamma, `chi_m(r) = sum_T phi_m(r - T)`, the cell is
//! a closed problem in a real basis: the density is
//! `sum_mn D_mn chi_m chi_n`, overlap and kinetic matrices are lattice sums
//! of two-centre integrals, and two pair densities interact through their
//! Ewald sum. Every shell is split into its compact and its diffuse
//! primitives as in [`super::screened_coulomb`]: an interaction with a pair
//! of two diffuse parts is summed in reciprocal space with the bare kernel,
//! and two pairs with a compact part, or such a pair and a nucleus, use the
//! Ewald split with its neutralizing background. The partition of the
//! lattice sums (which pairs, cells and reciprocal vectors are kept) is the
//! one of the reference geometry; the kept terms are differentiated exactly.
//! The exchange tensor is the minimum-image sum of [`super::exchange`] at
//! Gamma.
//!
//! Gaussian products and Boys functions are carried by the runtime Taylor
//! scalar. A reciprocal vector follows the lattice as `G'(t) = F(t)^-T G`,
//! so the phase of a plane wave at a moving centre,
//! `G'(t) . F(t) (P + u(t)) = G . (P + u(t))`, does not depend on the
//! deformation; its exponential, the Gaussian envelopes
//! `exp(-|G'(t)|^2 / 4p)`, the powers of `G'(t)` and the Coulomb weights
//! `4 pi / (V(t) |G'(t)|^2)` are series with closed recursions.

#![allow(clippy::needless_range_loop)]

use crate::basis::aos::cart_to_sph;
use crate::basis::{cart_components, n_sph, MolShell};
use crate::dual::Scalar;
use crate::error::{DftError, Result};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self as jets, AnalyticNucleus, AnalyticShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::exchange::minimum_image_weight;
use crate::periodic::grid::{local_images_of, PeriodicGrid, Switch};
use crate::periodic::integrals::nearby_translation_indices;
use crate::periodic::screened_coulomb::{decay, in_half_space, pairs, split_basis};
use crate::periodic::strain_jets::{hermite, inverse_transpose, strain_apply, strain_det};
use crate::periodic::xc_strain::{becke_weight_jet, shell_values};
use crate::periodic::{Cell, Lattice};
use crate::response::series::reciprocal as reciprocal_series;
use crate::response::series::{evaluate_series_many_result, SeriesScalar};
use rayon::prelude::*;
use std::f64::consts::PI;

type Ts = SeriesScalar;
/// Complex series as real and imaginary coefficients.
type Cs = (Vec<f64>, Vec<f64>);
/// Quadrature weight series and `[AO][value, gradient]` series per point.
type GridSeries = (Vec<Vec<f64>>, Vec<Vec<[Vec<f64>; 4]>>);

/// Two primitives overlap while `mu |A - B|^2` stays below this value.
const PAIR_LOG: f64 = 40.0;

/// A cell-periodic scalar potential `amplitude(t) cos(G . r)` (or `sin`) in
/// the one-electron operator.
#[derive(Clone, Debug)]
pub struct ScalarProbe {
    /// The reciprocal lattice vector `G` of the cell in units of its
    /// reciprocal basis vectors; it follows a deformation of the lattice.
    pub wave: [i64; 3],
    pub sine: bool,
    /// Coefficients of `t^1, t^2, ...` of the amplitude.
    pub amplitude: Vec<f64>,
}

/// A path of a cell: every centre moves as `F(t) (R + T + u(t))`.
#[derive(Clone, Debug, Default)]
pub struct LatticePath {
    /// Coefficients of `t^1, t^2, ...` of the displacement `u(t)`, one
    /// vector per atom of the cell.
    pub displacement: Vec<Vec<[f64; 3]>>,
    /// Coefficients of `t^1, t^2, ...` of the deformation `F(t) - 1`.
    pub deformation: Vec<[[f64; 3]; 3]>,
    /// Coefficients of `t^1, t^2, ...` of a uniform electric field, given in
    /// the frame of the reference lattice: the reduced field `E . a_i` is
    /// this function of `t` whatever the deformation.
    pub field: Vec<[f64; 3]>,
    /// Cell-periodic scalar potentials along the path.
    pub potential: Vec<ScalarProbe>,
}

impl LatticePath {
    /// The straight displacement `u_A(t) = t direction[A]` at a fixed lattice.
    pub fn displacement(direction: &[[f64; 3]]) -> Self {
        Self {
            displacement: vec![direction.to_vec()],
            deformation: Vec::new(),
            field: Vec::new(),
            potential: Vec::new(),
        }
    }
}

/// What [`lattice_series`] evaluates.
pub(crate) struct LatticeRequest<'a> {
    pub(crate) cell: &'a Cell,
    pub(crate) path: &'a LatticePath,
    pub(crate) order: usize,
    /// Quadrature grid, partition shell and AO cutoff of the semilocal part.
    pub(crate) grid: Option<(&'a PeriodicGrid, i64, f64)>,
    /// Image cutoff of the full-range exact exchange.
    pub(crate) exchange: Option<f64>,
    /// Range-separation parameter and image cutoff of the long-range
    /// exact exchange.
    pub(crate) long_range: Option<(f64, f64)>,
}

/// Series of every integral of the cell, `[order]` first.
pub(crate) struct LatticeSeries {
    pub(crate) overlap: Vec<Matrix>,
    /// Kinetic energy and nuclear attraction.
    pub(crate) core: Vec<Matrix>,
    /// Ewald sums `(m n | l s)` as `[((m nao + n) nao + l) nao + s]`.
    pub(crate) eri: Vec<Vec<f64>>,
    /// Minimum-image exchange sums `(m a | n b)` in the same layout.
    pub(crate) exchange: Option<Vec<Vec<f64>>>,
    pub(crate) exchange_lr: Option<Vec<Vec<f64>>>,
    /// Ewald energy of the nuclei.
    pub(crate) nuclear: Vec<f64>,
    /// `[re, im]` series of `<m| exp(i b_a . r) - 1 |n>` between the Bloch
    /// sums, for the three reciprocal basis vectors `b_a` of the cell.
    pub(crate) polar: [[Vec<Matrix>; 2]; 3],
    /// Series of the phases of the nuclei, `sum_A Z_A b_a . (R_A + u_A)`.
    pub(crate) ionic: [Vec<f64>; 3],
    /// Quadrature weights per grid point.
    pub(crate) weights: Vec<Vec<f64>>,
    /// `[point][AO][value, gradient]` of the Gamma Bloch sums.
    pub(crate) aos: Vec<Vec<[Vec<f64>; 4]>>,
}

/// Inputs of a series scope: the displacement of every Cartesian coordinate
/// `3 atom + axis`, then the nine components of `F(t) - 1`.
fn inputs(path: &LatticePath, natoms: usize, order: usize) -> Vec<Vec<f64>> {
    let width = order + 1;
    let mut out = Vec::with_capacity(3 * natoms + 9);
    for atom in 0..natoms {
        for axis in 0..3 {
            let mut v = vec![0.0; width];
            for (n, step) in path.displacement.iter().enumerate().take(order) {
                v[n + 1] = step[atom][axis];
            }
            out.push(v);
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            let mut v = vec![0.0; width];
            for (n, step) in path.deformation.iter().enumerate().take(order) {
                v[n + 1] = step[a][b];
            }
            out.push(v);
        }
    }
    out
}

/// Expand a geometry-only energy along the same affine and internal motion
/// as the electronic integrals, with the displacement applied before strain.
pub(super) fn geometry_energy_series(
    positions: &[Vec3],
    lattice: &Lattice,
    path: &LatticePath,
    order: usize,
    energy: impl FnOnce(&[[Ts; 3]], &[[Ts; 3]; 3]) -> Ts,
) -> Result<Vec<f64>> {
    let mut values = evaluate_series_many_result(&inputs(path, positions.len(), order), |v| {
        let positions: Vec<_> = positions
            .iter()
            .enumerate()
            .map(|(atom, &position)| moved(v, atom, position))
            .collect();
        let affine = deformation(v);
        let rows = lattice
            .a
            .map(|row| strain_apply(&affine, row.to_array().map(Ts::from_f64)));
        Ok(vec![energy(&positions, &rows)])
    })?;
    Ok(values.remove(0))
}

/// The physical quadrature coordinates follow exactly the moving atom
/// anchors used by `grid_series`; the radial offsets remain in Cartesian
/// coordinates. Lattice images follow the deformation independently.
pub(crate) fn nonlocal_geometry_series(
    cell: &Cell,
    grid: &PeriodicGrid,
    path: &LatticePath,
    order: usize,
) -> Result<(Vec<[Vec<f64>; 3]>, [[Vec<f64>; 3]; 3])> {
    let values = evaluate_series_many_result(&inputs(path, cell.molecule.len(), order), |v| {
        let mut out = Vec::with_capacity(3 * grid.points.len() + 9);
        for (i, point) in grid.points.iter().enumerate() {
            let owner = grid.atom_of[i];
            let home = cell.molecule.atoms[owner].position;
            let anchor = moved(v, owner, home);
            let offset = *point - home;
            out.extend((0..3).map(|k| anchor[k] + Ts::from_f64(offset.get(k))));
        }
        for vector in cell.lattice.a {
            out.extend(strain_apply(
                &deformation(v),
                std::array::from_fn(|k| Ts::from_f64(vector.get(k))),
            ));
        }
        Ok(out)
    })?;
    let positions = (0..grid.points.len())
        .map(|i| std::array::from_fn(|a| values[3 * i + a].clone()))
        .collect();
    let offset = 3 * grid.points.len();
    let lattice =
        std::array::from_fn(|a| std::array::from_fn(|b| values[offset + 3 * a + b].clone()));
    Ok((positions, lattice))
}

/// The deformation `F(t) - 1` inside a series scope.
fn deformation(v: &[Ts]) -> [[Ts; 3]; 3] {
    let start = v.len() - 9;
    std::array::from_fn(|a| std::array::from_fn(|b| v[start + 3 * a + b]))
}

/// `F(t) (position + u_atom(t))` of a point that rides `atom`.
fn moved(v: &[Ts], atom: usize, position: Vec3) -> [Ts; 3] {
    strain_apply(
        &deformation(v),
        std::array::from_fn(|k| Ts::from_f64(position.get(k)) + v[3 * atom + k]),
    )
}

/// A shell at `centre + translation` riding its atom.
fn placed(shell: &MolShell, translation: Vec3, v: &[Ts]) -> Result<AnalyticShell<Ts>> {
    AnalyticShell::from_placed(shell, moved(v, shell.atom, shell.center + translation))
}

fn least(shell: &MolShell) -> f64 {
    shell.exps.iter().copied().fold(f64::INFINITY, f64::min)
}

fn reduced(a: f64, b: f64) -> f64 {
    a * b / (a + b)
}

fn cauchy(a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..a.len())
        .map(|n| (0..=n).map(|k| a[k] * b[n - k]).sum())
        .collect()
}

fn scaled(a: &[f64], factor: f64) -> Vec<f64> {
    a.iter().map(|x| factor * x).collect()
}

fn exp_series(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    y[0] = x[0].exp();
    for n in 1..x.len() {
        y[n] = (1..=n).map(|k| k as f64 * x[k] * y[n - k]).sum::<f64>() / n as f64;
    }
    y
}

fn cmul(a: &Cs, b: &Cs) -> Cs {
    let (rr, ii) = (cauchy(&a.0, &b.0), cauchy(&a.1, &b.1));
    let (ri, ir) = (cauchy(&a.0, &b.1), cauchy(&a.1, &b.0));
    (
        rr.iter().zip(&ii).map(|(x, y)| x - y).collect(),
        ri.iter().zip(&ir).map(|(x, y)| x + y).collect(),
    )
}

/// `exp(i phase(t))` of a polynomial phase as a series in `t`.
fn phase_series(phase: &[f64]) -> Cs {
    let width = phase.len();
    let (mut re, mut im) = (vec![0.0; width], vec![0.0; width]);
    re[0] = phase[0].cos();
    im[0] = phase[0].sin();
    for n in 1..width {
        let (mut a, mut b) = (0.0, 0.0);
        for k in 1..=n {
            // k i phase_k y_(n-k).
            a -= k as f64 * phase[k] * im[n - k];
            b += k as f64 * phase[k] * re[n - k];
        }
        re[n] = a / n as f64;
        im[n] = b / n as f64;
    }
    (re, im)
}

fn add_matrix(target: &mut Matrix, source: &Matrix) {
    for (x, y) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *x += *y;
    }
}

/// Lattice sums of the overlap and kinetic integrals.
fn two_centre(cell: &Cell, input: &[Vec<f64>], order: usize) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    let width = order + 1;
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let nao = basis.nao_sph;
    let ns = basis.shells.len();
    let shell_pairs: Vec<(usize, usize)> =
        (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let blocks = shell_pairs
        .par_iter()
        .map(|&(a, b)| -> Result<Vec<Vec<f64>>> {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            let count = n_sph(sa.l) * n_sph(sb.l);
            // The one-electron lattice sums of the cell stop at its cutoff.
            let reach = (PAIR_LOG / reduced(least(sa), least(sb)))
                .sqrt()
                .min(cell.overlap_cutoff);
            let mut total = vec![vec![0.0; width]; 2 * count];
            for index in nearby_translation_indices(lattice, sb.center - sa.center, reach) {
                let t = lattice.translation(index);
                let block = evaluate_series_many_result(input, |v| {
                    let (pa, pb) = (placed(sa, Vec3::zero(), v)?, placed(sb, t, v)?);
                    let mut out = jets::overlap(&pa, &pb)?.values;
                    out.extend(jets::kinetic(&pa, &pb)?.values);
                    Ok(out)
                })?;
                for (sum, part) in total.iter_mut().zip(&block) {
                    for (x, y) in sum.iter_mut().zip(part) {
                        *x += *y;
                    }
                }
            }
            Ok(total)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut overlap = vec![Matrix::zeros(nao, nao); width];
    let mut kinetic = overlap.clone();
    for (&(a, b), block) in shell_pairs.iter().zip(&blocks) {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for i in 0..na {
            for j in 0..nb {
                for n in 0..width {
                    overlap[n][(sa.sph_off + i, sb.sph_off + j)] = block[i * nb + j][n];
                    kinetic[n][(sa.sph_off + i, sb.sph_off + j)] = block[na * nb + i * nb + j][n];
                }
            }
        }
    }
    Ok((overlap, kinetic))
}

/// The path as the reciprocal sums see it.
struct Reciprocal {
    width: usize,
    /// Displacement series `[atom][axis]`.
    shift: Vec<[Vec<f64>; 3]>,
    /// Reciprocal vectors of the reference lattice: zero first, then one of
    /// every pair `(G, -G)` in increasing length.
    waves: Vec<Vec3>,
    /// Series of the components and of the squared length of `F(t)^-T G`.
    component: Vec<[Vec<f64>; 3]>,
    norm2: Vec<Vec<f64>>,
    /// Series of the reciprocal cell volume.
    inverse_volume: Vec<f64>,
}

impl Reciprocal {
    fn new(cell: &Cell, gcut: f64, input: &[Vec<f64>], width: usize) -> Result<Self> {
        let lattice = &cell.lattice;
        let natoms = cell.molecule.len();
        let mut waves = lattice.reciprocal_within(gcut);
        waves.retain(|g| in_half_space(*g));
        waves.sort_by(|a, b| a.norm2().total_cmp(&b.norm2()));
        waves.insert(0, Vec3::zero());
        let volume = lattice.volume().abs();
        let series = evaluate_series_many_result(input, |v| {
            let eps = deformation(v);
            let m = inverse_transpose(&eps);
            let mut out = Vec::with_capacity(4 * waves.len() + 1);
            for g in &waves {
                let s: [Ts; 3] = std::array::from_fn(|e| {
                    m[e][0] * Ts::from_f64(g.x)
                        + m[e][1] * Ts::from_f64(g.y)
                        + m[e][2] * Ts::from_f64(g.z)
                });
                out.push(s[0]);
                out.push(s[1]);
                out.push(s[2]);
                out.push(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
            }
            out.push((Ts::from_f64(volume) * strain_det(&eps)).recip());
            Ok(out)
        })?;
        let component = (0..waves.len())
            .map(|g| std::array::from_fn(|k| series[4 * g + k].clone()))
            .collect();
        let norm2 = (0..waves.len())
            .map(|g| series[4 * g + 3].clone())
            .collect();
        let shift = (0..natoms)
            .map(|atom| std::array::from_fn(|k| input[3 * atom + k].clone()))
            .collect();
        let inverse_volume = series[4 * waves.len()].clone();
        Ok(Self {
            width,
            shift,
            waves,
            component,
            norm2,
            inverse_volume,
        })
    }

    /// Series of `exp(i G'(t) . F(t) (position + u(t)))` for wave `g`, where
    /// `u = sum_i weight_i u_(atom_i)`: the phase is `G . (position + u(t))`.
    fn phase(&self, g: usize, position: Vec3, riders: &[(usize, f64)]) -> Cs {
        let wave = self.waves[g];
        let mut phase = vec![0.0; self.width];
        phase[0] = wave.dot(position);
        for n in 1..self.width {
            for &(atom, weight) in riders {
                for k in 0..3 {
                    phase[n] += weight * wave.get(k) * self.shift[atom][k][n];
                }
            }
        }
        phase_series(&phase)
    }
}

/// Shape-dependent monopole correction on the same deformed lattice as the
/// Coulomb series. The unit charge at the origin has a constant structure
/// factor, so only the real distances, reciprocal lengths and volume move.
fn monopole_path(cell: &Cell, input: &[Vec<f64>], order: usize) -> Result<Vec<f64>> {
    let lattice = &cell.lattice;
    let charge = cell.net_charge();
    let mut correction = vec![0.0; order + 1];
    correction[0] = cell.charged_cell_correction();
    if order == 0 || charge.abs() < 1e-14 {
        return Ok(correction);
    }
    let parameters = super::ewald::EwaldParams::for_accuracy(lattice, 1, 1e-13);
    let reciprocal = Reciprocal::new(cell, parameters.gcut, input, order + 1)?;
    let alpha = parameters.alpha;
    let mut energy = scaled(&reciprocal.inverse_volume, -0.5 * PI / (alpha * alpha));
    for norm2 in &reciprocal.norm2[1..] {
        let radial = cauchy(
            &reciprocal_series(norm2),
            &exp_series(&scaled(norm2, -0.25 / (alpha * alpha))),
        );
        let term = cauchy(&reciprocal.inverse_volume, &radial);
        for (total, coefficient) in energy.iter_mut().zip(term) {
            *total += 4.0 * PI * coefficient;
        }
    }
    let translations = lattice.translations_within(parameters.rcut);
    let real = evaluate_series_many_result(input, |v| {
        let eps = deformation(v);
        let mut energy = Ts::from_f64(0.0);
        for t in &translations {
            if t.norm2() < 1e-20 || t.norm() > parameters.rcut {
                continue;
            }
            let r = strain_apply(&eps, t.to_array().map(Ts::from_f64));
            let length = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            energy = energy + (Ts::from_f64(1.0) - length.scale(alpha).erf()).scale(0.5) / length;
        }
        Ok(vec![energy])
    })?;
    // The Ewald self term is constant for the fixed splitting parameter.
    // Preserve the SCF's exact reference value from its own Ewald call.
    for n in 1..=order {
        correction[n] = -charge * charge * (energy[n] + real[0][n]);
    }
    Ok(correction)
}

/// Add the series of `int a_i(r) b_j(r - t) exp(i G' . r)` for every wave
/// into `re`/`im`, laid out `[((i nb + j) ng + g) width + order]`.
fn pair_transform(
    sa: &MolShell,
    sb: &MolShell,
    t: Vec3,
    path: &Reciprocal,
    input: &[Vec<f64>],
    re: &mut [f64],
    im: &mut [f64],
) -> Result<()> {
    let width = path.width;
    let (la, lb) = (sa.l as usize, sb.l as usize);
    let nt = la + lb + 1;
    let size = (la + 1) * (lb + 1) * nt;
    let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
    let (ta, tb) = (cart_to_sph(sa.l), cart_to_sph(sb.l));
    let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
    let ng = path.waves.len();
    let b_centre = sb.center + t;
    // Hermite coefficients of every primitive pair and Cartesian direction.
    let tables = evaluate_series_many_result(input, |v| {
        let a = moved(v, sa.atom, sa.center);
        let b = moved(v, sb.atom, b_centre);
        let mut out = Vec::with_capacity(sa.exps.len() * sb.exps.len() * 3 * size);
        for &ea in &sa.exps {
            for &eb in &sb.exps {
                for k in 0..3 {
                    out.extend(hermite(la, lb, a[k] - b[k], ea, eb));
                }
            }
        }
        Ok(out)
    })?;
    for g in 0..ng {
        // Powers of the components of the moving wave vector.
        let powers: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|k| {
                let mut list: Vec<Vec<f64>> = Vec::with_capacity(nt);
                let mut unit = vec![0.0; width];
                unit[0] = 1.0;
                list.push(unit);
                for tau in 1..nt {
                    let next = cauchy(&list[tau - 1], &path.component[g][k]);
                    list.push(next);
                }
                list
            })
            .collect();
        let mut cart: Vec<Cs> = vec![(vec![0.0; width], vec![0.0; width]); ca.len() * cb.len()];
        let mut any = false;
        let mut block = 0;
        for (&ea, &coa) in sa.exps.iter().zip(&sa.coefs) {
            for (&eb, &cob) in sb.exps.iter().zip(&sb.coefs) {
                let base = block * 3 * size;
                block += 1;
                let p = ea + eb;
                let envelope = scaled(
                    &exp_series(&scaled(&path.norm2[g], -0.25 / p)),
                    (PI / p).powf(1.5) * coa * cob,
                );
                if envelope[0].abs() < 1e-18 {
                    continue;
                }
                any = true;
                let centre = (sa.center * ea + b_centre * eb) / p;
                let phase = path.phase(g, centre, &[(sa.atom, ea / p), (sb.atom, eb / p)]);
                let factor = cmul(&phase, &(envelope, vec![0.0; width]));
                // sum_tau E^{ij}_tau (i G'_k)^tau for every direction k.
                let axis: Vec<Vec<Cs>> = (0..3)
                    .map(|k| {
                        (0..(la + 1) * (lb + 1))
                            .map(|ij| {
                                let (mut real, mut imag) = (vec![0.0; width], vec![0.0; width]);
                                for tau in 0..nt {
                                    let term = cauchy(
                                        &tables[base + k * size + ij * nt + tau],
                                        &powers[k][tau],
                                    );
                                    let sign = if tau % 4 < 2 { 1.0 } else { -1.0 };
                                    let target = if tau % 2 == 0 { &mut real } else { &mut imag };
                                    for (x, y) in target.iter_mut().zip(&term) {
                                        *x += sign * y;
                                    }
                                }
                                (real, imag)
                            })
                            .collect()
                    })
                    .collect();
                for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                    for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                        let pick =
                            |k: usize, a: i32, b: i32| &axis[k][a as usize * (lb + 1) + b as usize];
                        let product = cmul(
                            &cmul(pick(0, ax, bx), pick(1, ay, by)),
                            &cmul(pick(2, az, bz), &factor),
                        );
                        let slot = &mut cart[i * cb.len() + j];
                        for n in 0..width {
                            slot.0[n] += product.0[n];
                            slot.1[n] += product.1[n];
                        }
                    }
                }
            }
        }
        if !any {
            // Every later wave is longer and its envelopes are smaller.
            break;
        }
        for i in 0..na {
            for j in 0..nb {
                let start = ((i * nb + j) * ng + g) * width;
                for ci in 0..ca.len() {
                    for cj in 0..cb.len() {
                        let w = ta[(ci, i)] * tb[(cj, j)];
                        if w == 0.0 {
                            continue;
                        }
                        let c = &cart[ci * cb.len() + cj];
                        for n in 0..width {
                            re[start + n] += w * c.0[n];
                            im[start + n] += w * c.1[n];
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

struct Coulomb {
    eri: Vec<Vec<f64>>,
    attraction: Vec<Matrix>,
    nuclear: Vec<f64>,
    polar: [[Vec<Matrix>; 2]; 3],
    ionic: [Vec<f64>; 3],
    /// Series of the matrices of the scalar probes.
    probes: Vec<Vec<Matrix>>,
}

/// Ewald sums of the electron repulsion, the nuclear attraction and the
/// nuclear repulsion.
fn coulomb(
    cell: &Cell,
    input: &[Vec<f64>],
    order: usize,
    probes: &[ScalarProbe],
    metric: &[Matrix],
) -> Result<Coulomb> {
    let width = order + 1;
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let molecule = &cell.molecule;
    let nao = basis.nao_sph;
    let n2 = nao * nao;
    let ns = basis.shells.len();
    let natoms = molecule.len();
    let screening = cell.hartree_screening();
    let alpha = screening.alpha();
    let tolerance = screening.tolerance;
    let charges = basis.effective_charges(molecule);
    let total_charge: f64 = charges.iter().sum();
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let parts = split_basis(basis, screening.compact_exponent);
    let pair_list = pairs(&parts, lattice, &screening);
    let shell_of: Vec<usize> = parts
        .iter()
        .map(|part| {
            basis
                .shells
                .iter()
                .position(|s| s.sph_off == part.shell.sph_off)
                .unwrap_or(0)
        })
        .collect();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ns * ns];
    for (index, pair) in pair_list.iter().enumerate() {
        groups[shell_of[pair.x] * ns + shell_of[pair.y]].push(index);
    }
    // ---- reciprocal vectors and Coulomb weights along the path ----
    let path = Reciprocal::new(cell, screening.gcut(), input, width)?;
    let ng = path.waves.len();
    let mut bare = vec![vec![0.0; width]; ng];
    let mut damped = bare.clone();
    for g in 1..ng {
        // One of every pair (G, -G) carries twice the weight.
        let inverse = reciprocal_series(&path.norm2[g]);
        bare[g] = scaled(&cauchy(&path.inverse_volume, &inverse), 8.0 * PI);
        damped[g] = cauchy(
            &bare[g],
            &exp_series(&scaled(&path.norm2[g], -0.25 / (alpha * alpha))),
        );
    }
    // ---- pair transforms: [class][re, im][order] as (nao^2, ng) ----
    let blocks = groups
        .par_iter()
        .enumerate()
        .map(|(index, members)| -> Result<[Vec<f64>; 4]> {
            let (sa, sb) = (&basis.shells[index / ns], &basis.shells[index % ns]);
            let size = n_sph(sa.l) * n_sph(sb.l) * ng * width;
            // Real parts of the two classes, then their imaginary parts.
            let mut out: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; size]);
            for &member in members {
                let pair = &pair_list[member];
                let class = usize::from(pair.diffuse);
                let (real, imag) = out.split_at_mut(2);
                pair_transform(
                    &parts[pair.x].shell,
                    &parts[pair.y].shell,
                    pair.translation,
                    &path,
                    input,
                    &mut real[class],
                    &mut imag[class],
                )?;
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut transform: [[Vec<Matrix>; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![Matrix::zeros(n2, ng); width]));
    for (index, block) in blocks.iter().enumerate() {
        let (sa, sb) = (&basis.shells[index / ns], &basis.shells[index % ns]);
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for i in 0..na {
            for j in 0..nb {
                let row = (sa.sph_off + i) * nao + sb.sph_off + j;
                for g in 0..ng {
                    let start = ((i * nb + j) * ng + g) * width;
                    for n in 0..width {
                        for class in 0..2 {
                            transform[class][0][n][(row, g)] = block[class][start + n];
                            transform[class][1][n][(row, g)] = block[2 + class][start + n];
                        }
                    }
                }
            }
        }
    }
    drop(blocks);
    // What a pair with a compact part (class 0) and a pair of two diffuse
    // parts (class 1) see of the electrons, with the weights of the path.
    let seen = |class: usize, part: usize, n: usize| -> Matrix {
        let mut out = Matrix::zeros(n2, ng);
        for k in 0..=n {
            let j = n - k;
            let (c, d) = (&transform[0][part][k], &transform[1][part][k]);
            for row in 0..n2 {
                for g in 1..ng {
                    out[(row, g)] += if class == 0 {
                        damped[g][j] * c[(row, g)] + bare[g][j] * d[(row, g)]
                    } else {
                        bare[g][j] * (c[(row, g)] + d[(row, g)])
                    };
                }
            }
        }
        out
    };
    let mut eri = vec![vec![0.0; n2 * n2]; width];
    for k in 0..width {
        for class in 0..2 {
            for part in 0..2 {
                let ket = seen(class, part, k).transpose();
                for m in 0..width - k {
                    let product = transform[class][part][m].matmul(&ket);
                    for (x, y) in eri[m + k].iter_mut().zip(product.as_slice()) {
                        *x += *y;
                    }
                }
            }
        }
    }
    // Background of the Ewald split between two compact distributions:
    // the compact electron counts weighted by `pi / (alpha^2 V(t))`.
    let background = scaled(&path.inverse_volume, PI / (alpha * alpha));
    let weighted: Vec<Vec<f64>> = (0..width)
        .map(|k| {
            (0..n2)
                .map(|q| {
                    (0..=k)
                        .map(|j| background[j] * transform[0][0][k - j][(q, 0)])
                        .sum()
                })
                .collect()
        })
        .collect();
    for n in 0..width {
        for m in 0..=n {
            for p in 0..n2 {
                let a = transform[0][0][m][(p, 0)];
                if a == 0.0 {
                    continue;
                }
                for q in 0..n2 {
                    eri[n][p * n2 + q] -= a * weighted[n - m][q];
                }
            }
        }
    }
    // ---- matrices of exp(i b_a . r) - 1 and phases of the nuclei ----
    let reciprocal = lattice.reciprocal();
    let mut polar: [[Vec<Matrix>; 2]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![Matrix::zeros(nao, nao); width]));
    let mut ionic: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; width]);
    for a in 0..3 {
        // The kept member of the pair (b_a, -b_a) and its sign.
        let found = (1..ng).find_map(|g| {
            if (path.waves[g] - reciprocal[a]).norm() < 1e-8 {
                Some((g, 1.0))
            } else if (path.waves[g] + reciprocal[a]).norm() < 1e-8 {
                Some((g, -1.0))
            } else {
                None
            }
        });
        let (g, sign) = found.ok_or_else(|| {
            DftError::InvalidInput(
                "the reciprocal basis vectors of the cell lie outside the reciprocal cutoff of \
                 its Coulomb sums"
                    .into(),
            )
        })?;
        for n in 0..width {
            for row in 0..n2 {
                polar[a][0][n].as_mut_slice()[row] = transform[0][0][n][(row, g)]
                    + transform[1][0][n][(row, g)]
                    // P S P=P uses the actual one-electron metric. The
                    // Coulomb pair partition can have a different image
                    // cutoff; its G=0 charge is not a replacement for S.
                    - metric[n].as_slice()[row];
                polar[a][1][n].as_mut_slice()[row] =
                    sign * (transform[0][1][n][(row, g)] + transform[1][1][n][(row, g)]);
            }
        }
        for atom in 0..natoms {
            ionic[a][0] += charges[atom] * reciprocal[a].dot(positions[atom]);
            for n in 1..width {
                for k in 0..3 {
                    ionic[a][n] += charges[atom] * reciprocal[a].get(k) * path.shift[atom][k][n];
                }
            }
        }
    }
    // ---- matrices of the scalar probes cos(G . r) and sin(G . r) ----
    let mut probe_matrices: Vec<Vec<Matrix>> = Vec::with_capacity(probes.len());
    for probe in probes {
        let vector = reciprocal[0] * probe.wave[0] as f64
            + reciprocal[1] * probe.wave[1] as f64
            + reciprocal[2] * probe.wave[2] as f64;
        let found = (1..ng).find_map(|g| {
            if (path.waves[g] - vector).norm() < 1e-8 {
                Some((g, 1.0))
            } else if (path.waves[g] + vector).norm() < 1e-8 {
                Some((g, -1.0))
            } else {
                None
            }
        });
        let (g, sign) = found.ok_or_else(|| {
            DftError::InvalidInput(
                "the wave vector of a scalar probe is zero or lies outside the reciprocal \
                 cutoff of the Coulomb sums of the cell"
                    .into(),
            )
        })?;
        // cos is the real part of exp(i G . r), sin its imaginary part.
        let part = usize::from(probe.sine);
        let factor = if probe.sine { sign } else { 1.0 };
        let mut series = vec![Matrix::zeros(nao, nao); width];
        for n in 0..width {
            for row in 0..n2 {
                series[n].as_mut_slice()[row] =
                    factor * (transform[0][part][n][(row, g)] + transform[1][part][n][(row, g)]);
            }
        }
        probe_matrices.push(series);
    }
    // ---- structure factor of the nuclei: [re, im][order][g] ----
    let mut structure = [vec![vec![0.0; ng]; width], vec![vec![0.0; ng]; width]];
    for g in 0..ng {
        for atom in 0..natoms {
            if charges[atom] == 0.0 {
                continue;
            }
            let (real, imag) = path.phase(g, positions[atom], &[(atom, 1.0)]);
            for n in 0..width {
                structure[0][n][g] += charges[atom] * real[n];
                structure[1][n][g] += charges[atom] * imag[n];
            }
        }
    }
    let mut attraction = vec![Matrix::zeros(nao, nao); width];
    for m in 0..width {
        let fields = [seen(0, 0, m), seen(0, 1, m)];
        for k in 0..width - m {
            for row in 0..n2 {
                let mut value = 0.0;
                for g in 1..ng {
                    value += fields[0][(row, g)] * structure[0][k][g]
                        + fields[1][(row, g)] * structure[1][k][g];
                }
                attraction[m + k].as_mut_slice()[row] -= value;
            }
        }
        for row in 0..n2 {
            attraction[m].as_mut_slice()[row] += total_charge * weighted[m][row];
        }
    }
    let mut nuclear = vec![0.0; width];
    for g in 1..ng {
        // |S(G)|^2 as a series.
        let square: Vec<f64> = (0..width)
            .map(|l| {
                (0..=l)
                    .map(|m| {
                        structure[0][m][g] * structure[0][l - m][g]
                            + structure[1][m][g] * structure[1][l - m][g]
                    })
                    .sum()
            })
            .collect();
        let term = cauchy(&damped[g], &square);
        for n in 0..width {
            nuclear[n] += 0.5 * term[n];
        }
    }
    nuclear[0] -= alpha / PI.sqrt() * charges.iter().map(|z| z * z).sum::<f64>();
    for n in 0..width {
        nuclear[n] -= 0.5 * background[n] * total_charge * total_charge;
    }
    // ---- screened nuclear repulsion in real space ----
    let reach = ((1.0 / tolerance).ln() + 4.0).sqrt() / alpha;
    let mut images: Vec<(usize, usize, Vec3)> = Vec::new();
    for a in 0..natoms {
        for b in 0..natoms {
            if charges[a] == 0.0 || charges[b] == 0.0 {
                continue;
            }
            for index in nearby_translation_indices(lattice, positions[b] - positions[a], reach) {
                let t = lattice.translation(index);
                if a == b && t.norm() < 1e-10 {
                    continue;
                }
                images.push((a, b, t));
            }
        }
    }
    if !images.is_empty() {
        let real = evaluate_series_many_result(input, |v| {
            let mut energy = Ts::from_f64(0.0);
            for &(a, b, t) in &images {
                let (ra, rb) = (moved(v, a, positions[a]), moved(v, b, positions[b] + t));
                let d: [Ts; 3] = std::array::from_fn(|k| rb[k] - ra[k]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                energy = energy
                    + Ts::from_f64(0.5 * charges[a] * charges[b])
                        * (Ts::from_f64(1.0) - (r * Ts::from_f64(alpha)).erf())
                        / r;
            }
            Ok(vec![energy])
        })?;
        for n in 0..width {
            nuclear[n] += real[0][n];
        }
    }
    // ---- screened real-space interactions of the compact pairs ----
    let compact: Vec<usize> = (0..pair_list.len())
        .filter(|&i| !pair_list[i].diffuse)
        .collect();
    let slabs = groups
        .par_iter()
        .enumerate()
        .map(|(index, members)| -> Result<(Vec<f64>, Vec<f64>)> {
            let (sa, sb) = (&basis.shells[index / ns], &basis.shells[index % ns]);
            let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
            let mut two = vec![0.0; na * nb * n2 * width];
            let mut one = vec![0.0; na * nb * width];
            for &member in members {
                let bra = &pair_list[member];
                if bra.diffuse {
                    continue;
                }
                let (bx, by) = (&parts[bra.x].shell, &parts[bra.y].shell);
                // Nuclei within the reach of the screened attraction.
                let rate = (bra.exponent * alpha * alpha / (bra.exponent + alpha * alpha)).sqrt();
                let reach = ((1.0 / tolerance).ln() + 4.0).sqrt() / rate + 1.0;
                let mut nuclei: Vec<(usize, Vec3)> = Vec::new();
                for atom in 0..natoms {
                    if charges[atom] == 0.0 {
                        continue;
                    }
                    for cell_index in
                        nearby_translation_indices(lattice, positions[atom] - bra.centre, reach)
                    {
                        nuclei.push((atom, positions[atom] + lattice.translation(cell_index)));
                    }
                }
                if !nuclei.is_empty() {
                    let block = evaluate_series_many_result(input, |v| {
                        let list: Vec<AnalyticNucleus<Ts>> = nuclei
                            .iter()
                            .map(|(atom, position)| AnalyticNucleus {
                                center: moved(v, *atom, *position),
                                charge: Ts::from_f64(charges[*atom]),
                            })
                            .collect();
                        Ok(jets::attraction(
                            &placed(bx, Vec3::zero(), v)?,
                            &placed(by, bra.translation, v)?,
                            &list,
                            GaussianKernel::ShortRange {
                                omega: Ts::from_f64(alpha),
                            },
                            order,
                        )?
                        .values)
                    })?;
                    for (slot, series) in block.iter().enumerate() {
                        for n in 0..width {
                            one[slot * width + n] += series[n];
                        }
                    }
                }
                for &other in &compact {
                    let ket = &pair_list[other];
                    let bound = bra.schwarz * ket.schwarz;
                    if bound < tolerance {
                        continue;
                    }
                    let (kx, ky) = (&parts[ket.x].shell, &parts[ket.y].shell);
                    let (nc, nd) = (n_sph(kx.l), n_sph(ky.l));
                    let margin = (bound / tolerance).ln() + 4.0;
                    let reach = margin.sqrt() / decay(alpha, bra.exponent, ket.exponent) + 1.0;
                    for cell_index in
                        nearby_translation_indices(lattice, bra.centre - ket.centre, reach)
                    {
                        let shift = Vec3::zero() - lattice.translation(cell_index);
                        let block = evaluate_series_many_result(input, |v| {
                            Ok(jets::repulsion(
                                &placed(bx, Vec3::zero(), v)?,
                                &placed(by, bra.translation, v)?,
                                &placed(kx, shift, v)?,
                                &placed(ky, shift + ket.translation, v)?,
                                GaussianKernel::ShortRange {
                                    omega: Ts::from_f64(alpha),
                                },
                                order,
                            )?
                            .values)
                        })?;
                        for i in 0..na {
                            for j in 0..nb {
                                for k in 0..nc {
                                    for l in 0..nd {
                                        let series = &block[((i * nb + j) * nc + k) * nd + l];
                                        let start = (((i * nb + j) * nao + kx.sph_off + k) * nao
                                            + ky.sph_off
                                            + l)
                                            * width;
                                        for n in 0..width {
                                            two[start + n] += series[n];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok((two, one))
        })
        .collect::<Result<Vec<_>>>()?;
    for (index, (two, one)) in slabs.iter().enumerate() {
        let (sa, sb) = (&basis.shells[index / ns], &basis.shells[index % ns]);
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for i in 0..na {
            for j in 0..nb {
                let row = (sa.sph_off + i) * nao + sb.sph_off + j;
                for n in 0..width {
                    attraction[n].as_mut_slice()[row] += one[(i * nb + j) * width + n];
                    for q in 0..n2 {
                        eri[n][row * n2 + q] += two[((i * nb + j) * n2 + q) * width + n];
                    }
                }
            }
        }
    }
    Ok(Coulomb {
        eri,
        attraction,
        nuclear,
        polar,
        ionic,
        probes: probe_matrices,
    })
}

/// Minimum-image exchange sums
/// `X(m a | n b) = sum_(T, C, D) w_mn(T) w_ab(D - C) (m_0 a_C | n_T b_D)`
/// over the translations inside `cutoff`, with the weights of
/// [`minimum_image_weight`] on the separations of the atoms; the exchange
/// matrix of a density is `K_mn = sum_ab X(m a | n b) D_ab`.
fn exchange_tensor(
    cell: &Cell,
    cutoff: f64,
    omega: Option<f64>,
    input: &[Vec<f64>],
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    let width = order + 1;
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let nao = basis.nao_sph;
    let n2 = nao * nao;
    let ns = basis.shells.len();
    let positions: Vec<Vec3> = cell.molecule.atoms.iter().map(|a| a.position).collect();
    let natoms = positions.len();
    let images = lattice.translations_within(cutoff);
    let wide = lattice.translations_within(2.0 * cutoff);
    let weighted = |list: &[Vec3]| -> Vec<Vec<(Vec3, f64)>> {
        (0..natoms * natoms)
            .map(|ab| {
                list.iter()
                    .filter_map(|t| {
                        let w = minimum_image_weight(
                            lattice,
                            [1, 1, 1],
                            positions[ab % natoms] + *t - positions[ab / natoms],
                        );
                        (w != 0.0).then_some((*t, w))
                    })
                    .collect()
            })
            .collect()
    };
    let outer = weighted(&images);
    let inner = weighted(&wide);
    let shell_pairs: Vec<(usize, usize)> =
        (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let slabs = shell_pairs
        .par_iter()
        .map(|&(sm, sa)| -> Result<Vec<f64>> {
            let (shell_m, shell_a) = (&basis.shells[sm], &basis.shells[sa]);
            let (nm, na) = (n_sph(shell_m.l), n_sph(shell_a.l));
            let mut slab = vec![0.0; nm * na * n2 * width];
            let reach_ma = (PAIR_LOG / reduced(least(shell_m), least(shell_a))).sqrt();
            for c in &images {
                if (shell_a.center + *c - shell_m.center).norm() > reach_ma {
                    continue;
                }
                for sn in 0..ns {
                    for sb in 0..ns {
                        let (shell_n, shell_b) = (&basis.shells[sn], &basis.shells[sb]);
                        let (nn, nb) = (n_sph(shell_n.l), n_sph(shell_b.l));
                        let reach_nb = (PAIR_LOG / reduced(least(shell_n), least(shell_b))).sqrt();
                        for (t, wt) in &outer[shell_m.atom * natoms + shell_n.atom] {
                            for (step, ws) in &inner[shell_a.atom * natoms + shell_b.atom] {
                                let d = *c + *step;
                                if d.norm() > cutoff * (1.0 + 1e-12)
                                    || (shell_b.center + d - shell_n.center - *t).norm() > reach_nb
                                {
                                    continue;
                                }
                                let block = evaluate_series_many_result(input, |v| {
                                    let kernel = omega.map_or(GaussianKernel::Coulomb, |w| {
                                        GaussianKernel::LongRange {
                                            omega: Ts::from_f64(w),
                                        }
                                    });
                                    Ok(jets::repulsion(
                                        &placed(shell_m, Vec3::zero(), v)?,
                                        &placed(shell_a, *c, v)?,
                                        &placed(shell_n, *t, v)?,
                                        &placed(shell_b, d, v)?,
                                        kernel,
                                        order,
                                    )?
                                    .values)
                                })?;
                                let weight = wt * ws;
                                for i in 0..nm {
                                    for j in 0..na {
                                        for k in 0..nn {
                                            for l in 0..nb {
                                                let series =
                                                    &block[((i * na + j) * nn + k) * nb + l];
                                                let start =
                                                    (((i * na + j) * nao + shell_n.sph_off + k)
                                                        * nao
                                                        + shell_b.sph_off
                                                        + l)
                                                        * width;
                                                for n in 0..width {
                                                    slab[start + n] += weight * series[n];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(slab)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut tensor = vec![vec![0.0; n2 * n2]; width];
    for (&(sm, sa), slab) in shell_pairs.iter().zip(&slabs) {
        let (shell_m, shell_a) = (&basis.shells[sm], &basis.shells[sa]);
        let (nm, na) = (n_sph(shell_m.l), n_sph(shell_a.l));
        for i in 0..nm {
            for j in 0..na {
                let row = (shell_m.sph_off + i) * nao + shell_a.sph_off + j;
                for q in 0..n2 {
                    for n in 0..width {
                        tensor[n][row * n2 + q] = slab[((i * na + j) * n2 + q) * width + n];
                    }
                }
            }
        }
    }
    // The truncated sum is symmetrized in its two pairs, so that the
    // exchange energy is a quadratic form with this tensor as its Hessian.
    for coefficient in &mut tensor {
        for p in 0..n2 {
            for q in 0..p {
                let mean = 0.5 * (coefficient[p * n2 + q] + coefficient[q * n2 + p]);
                coefficient[p * n2 + q] = mean;
                coefficient[q * n2 + p] = mean;
            }
        }
    }
    Ok(tensor)
}

/// Quadrature weights and Gamma Bloch sums of the AO values and gradients
/// at grid points that ride their owning atoms rigidly:
/// `r(t) = F(t) (R_O + u_O(t)) + offset`.
fn grid_series(
    cell: &Cell,
    grid: &PeriodicGrid,
    grid_shell: i64,
    ao_cutoff: f64,
    input: &[Vec<f64>],
) -> Result<GridSeries> {
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let molecule = &cell.molecule;
    let nao = basis.nao_sph;
    let sw = Switch::periodic(molecule, lattice, grid_shell);
    let points = (0..grid.points.len())
        .into_par_iter()
        .map(|i| -> Result<(Vec<f64>, Vec<[Vec<f64>; 4]>)> {
            let point = grid.points[i];
            let owner = grid.atom_of[i];
            let home = molecule.atoms[owner].position;
            let offset = point - home;
            let mut local: Vec<(usize, Vec3)> = Vec::new();
            for (s, shell) in basis.shells.iter().enumerate() {
                for index in nearby_translation_indices(lattice, shell.center - point, ao_cutoff) {
                    local.push((s, lattice.translation(index)));
                }
            }
            let raw = grid.raw_weights[i];
            // A point whose partition weight is one keeps its raw weight.
            let fixed = (grid.weights[i] - raw).abs() <= 1e-14 * raw.abs();
            let centres = if fixed {
                Vec::new()
            } else {
                local_images_of(molecule, lattice, point, sw.cut)
            };
            let target = centres
                .iter()
                .position(|(c, _, atom)| *atom == owner && (*c - home).norm() < 1e-10);
            let coefficients = evaluate_series_many_result(input, |v| {
                let anchor = moved(v, owner, home);
                let r: [Ts; 3] = std::array::from_fn(|k| anchor[k] + Ts::from_f64(offset.get(k)));
                let weight = if fixed {
                    Ts::from_f64(grid.weights[i])
                } else {
                    match target {
                        Some(target) => {
                            let centre_jets: Vec<[Ts; 3]> = centres
                                .iter()
                                .map(|(c, _, atom)| moved(v, *atom, *c))
                                .collect();
                            // The switching radii scale with the length
                            // per atom of the deformed cell.
                            let length_scale = strain_det(&deformation(v)).cbrt();
                            Ts::from_f64(raw)
                                * becke_weight_jet(
                                    &centres,
                                    &centre_jets,
                                    target,
                                    r,
                                    &sw,
                                    length_scale,
                                )
                        }
                        // Outside the partition radius of its owner.
                        None => Ts::from_f64(0.0),
                    }
                };
                let mut values = vec![Ts::from_f64(0.0); 1 + 4 * nao];
                values[0] = weight;
                for (s, t) in &local {
                    let shell = &basis.shells[*s];
                    let centre = moved(v, shell.atom, shell.center + *t);
                    let d: [Ts; 3] = std::array::from_fn(|k| r[k] - centre[k]);
                    let (value, gradient) = shell_values(shell, d);
                    for (m, (x, g)) in value.into_iter().zip(gradient).enumerate() {
                        let ix = 1 + 4 * (shell.sph_off + m);
                        values[ix] = values[ix] + x;
                        for k in 0..3 {
                            values[ix + 1 + k] = values[ix + 1 + k] + g[k];
                        }
                    }
                }
                Ok(values)
            })?;
            let aos = (0..nao)
                .map(|mu| std::array::from_fn(|a| coefficients[1 + 4 * mu + a].clone()))
                .collect();
            Ok((coefficients[0].clone(), aos))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(points.into_iter().unzip())
}

/// The conventional three-dimensional Ewald source uses a uniform
/// compensating background. Its compact-pair and nuclear terms are already
/// in the integral series. The monopole term follows affine lattice motion.
pub(crate) fn validate_charged_path(cell: &Cell, path: &LatticePath) -> Result<()> {
    let electric = path.field.iter().flatten().any(|&x| x != 0.0) || !path.potential.is_empty();
    if cell.net_charge().abs() > 1e-10 && electric {
        return Err(DftError::Unsupported {
            feature: "electric response of a charged periodic lattice path".into(),
            missing: "a charged-cell polarization and compensation convention".into(),
        });
    }
    Ok(())
}

/// Integral series of a Gamma-point cell along a path of its atoms and lattice.
/// The scalar ECP operator follows the same analytic nuclear and affine path.
pub(crate) fn lattice_series(request: &LatticeRequest) -> Result<LatticeSeries> {
    let cell = request.cell;
    let natoms = cell.molecule.len();
    let order = request.order;
    if natoms == 0
        || request
            .path
            .displacement
            .iter()
            .any(|step| step.len() != natoms || step.iter().flatten().any(|x| !x.is_finite()))
        || request
            .path
            .deformation
            .iter()
            .flatten()
            .flatten()
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "a lattice path holds one finite displacement for every atom and a finite \
             deformation at every order"
                .into(),
        ));
    }
    let moving_ecp = !cell.ecp_image_blocks().is_empty()
        && request
            .path
            .displacement
            .iter()
            .take(order)
            .flatten()
            .flatten()
            .chain(
                request
                    .path
                    .deformation
                    .iter()
                    .take(order)
                    .flatten()
                    .flatten(),
            )
            .any(|&value| value != 0.0);
    validate_charged_path(cell, request.path)?;
    let input = inputs(request.path, natoms, order);
    let (overlap, mut core) = two_centre(cell, &input, order)?;
    if !cell.ecp_image_blocks().is_empty() {
        let (ecp, _) = super::integrals::bloch(cell.ecp_image_blocks(), Vec3::zero());
        add_matrix(&mut core[0], &ecp);
        if moving_ecp && order > 2 {
            let coefficients = super::integrals::gamma_ecp_path(cell, request.path, order)?;
            for (target, coefficient) in core.iter_mut().zip(&coefficients).skip(1) {
                add_matrix(target, coefficient);
            }
        } else if moving_ecp {
            let derivative = super::integrals::gamma_ecp_derivative(
                cell,
                request.path.displacement.first().map(Vec::as_slice),
                request.path.deformation.first(),
            )?;
            add_matrix(&mut core[1], &derivative);
            if order >= 2 {
                let second = super::integrals::gamma_ecp_second(cell, request.path)?;
                add_matrix(&mut core[2], &second);
            }
        }
    }
    let mut coulomb = coulomb(cell, &input, order, &request.path.potential, &overlap)?;
    let moving_cell = request
        .path
        .deformation
        .iter()
        .take(order)
        .flatten()
        .flatten()
        .any(|&x| x != 0.0);
    if moving_cell {
        for (nuclear, correction) in coulomb
            .nuclear
            .iter_mut()
            .zip(monopole_path(cell, &input, order)?)
        {
            *nuclear += correction;
        }
    } else {
        coulomb.nuclear[0] += cell.charged_cell_correction();
    }
    for (target, source) in core.iter_mut().zip(&coulomb.attraction) {
        add_matrix(target, source);
    }
    for (probe, matrices) in request.path.potential.iter().zip(&coulomb.probes) {
        for n in 1..=order {
            // amplitude[k] multiplies t^(k + 1).
            for (k, amplitude) in probe.amplitude.iter().enumerate().take(n) {
                let source = &matrices[n - k - 1];
                for (x, y) in core[n].as_mut_slice().iter_mut().zip(source.as_slice()) {
                    *x += amplitude * y;
                }
            }
        }
    }
    let exchange = request
        .exchange
        .map(|cutoff| exchange_tensor(cell, cutoff, None, &input, order))
        .transpose()?;
    let exchange_lr = request
        .long_range
        .map(|(omega, cutoff)| exchange_tensor(cell, cutoff, Some(omega), &input, order))
        .transpose()?;
    let (weights, aos) = match request.grid {
        Some((grid, shell, cutoff)) => grid_series(cell, grid, shell, cutoff, &input)?,
        None => (Vec::new(), Vec::new()),
    };
    Ok(LatticeSeries {
        overlap,
        core,
        eri: coulomb.eri,
        exchange,
        exchange_lr,
        nuclear: coulomb.nuclear,
        polar: coulomb.polar,
        ionic: coulomb.ionic,
        weights,
        aos,
    })
}

/// Lattice sums `<chi_m| exp(i G . r) d/dr_k |chi_n>` between the Gamma Bloch
/// sums at the reference geometry for the three Cartesian directions `k`,
/// as real and imaginary parts. `wave` is a reciprocal lattice vector of the
/// cell, or zero for the momentum matrix itself. The derivative of a
/// Cartesian Gaussian lowers and raises its power, so the ket functions are
/// expanded with one more power.
pub(crate) fn momentum_matrices(cell: &Cell, wave: Vec3) -> [(Matrix, Matrix); 3] {
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let nao = basis.nao_sph;
    let ns = basis.shells.len();
    let g2 = wave.norm2();
    let times = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let shell_pairs: Vec<(usize, usize)> =
        (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let blocks: Vec<[(Vec<f64>, Vec<f64>); 3]> = shell_pairs
        .par_iter()
        .map(|&(a, b)| {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            let (la, lb) = (sa.l as usize, sb.l as usize);
            let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
            let (ta, tb) = (cart_to_sph(sa.l), cart_to_sph(sb.l));
            let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
            let nt = la + lb + 2;
            let size = ca.len() * cb.len();
            let mut cart: [(Vec<f64>, Vec<f64>); 3] =
                std::array::from_fn(|_| (vec![0.0; size], vec![0.0; size]));
            let reach = (PAIR_LOG / reduced(least(sa), least(sb)))
                .sqrt()
                .min(cell.overlap_cutoff);
            for cell_index in nearby_translation_indices(lattice, sb.center - sa.center, reach) {
                let centre_b = sb.center + lattice.translation(cell_index);
                for (&ea, &coa) in sa.exps.iter().zip(&sa.coefs) {
                    for (&eb, &cob) in sb.exps.iter().zip(&sb.coefs) {
                        let p = ea + eb;
                        let envelope = (PI / p).powf(1.5) * coa * cob * (-0.25 * g2 / p).exp();
                        let centre = (sa.center * ea + centre_b * eb) / p;
                        let angle = wave.dot(centre);
                        let phase = (envelope * angle.cos(), envelope * angle.sin());
                        // sum_tau E^{ij}_tau (i G_k)^tau for every direction.
                        let axis: Vec<Vec<(f64, f64)>> = (0..3)
                            .map(|k| {
                                let table =
                                    hermite(la, lb + 1, sa.center.get(k) - centre_b.get(k), ea, eb);
                                let gk = wave.get(k);
                                (0..(la + 1) * (lb + 2))
                                    .map(|ij| {
                                        let (mut re, mut im) = (0.0, 0.0);
                                        let mut power = 1.0;
                                        for tau in 0..nt {
                                            let value = table[ij * nt + tau] * power;
                                            match tau % 4 {
                                                0 => re += value,
                                                1 => im += value,
                                                2 => re -= value,
                                                _ => im -= value,
                                            }
                                            power *= gk;
                                        }
                                        (re, im)
                                    })
                                    .collect()
                            })
                            .collect();
                        for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                            let bra = [ax as usize, ay as usize, az as usize];
                            for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                                let ket = [bx as usize, by as usize, bz as usize];
                                let product = |powers: [usize; 3]| {
                                    times(
                                        times(
                                            axis[0][bra[0] * (lb + 2) + powers[0]],
                                            axis[1][bra[1] * (lb + 2) + powers[1]],
                                        ),
                                        axis[2][bra[2] * (lb + 2) + powers[2]],
                                    )
                                };
                                for k in 0..3 {
                                    let mut up = ket;
                                    up[k] += 1;
                                    let raised = product(up);
                                    let mut value = (-2.0 * eb * raised.0, -2.0 * eb * raised.1);
                                    if ket[k] > 0 {
                                        let mut down = ket;
                                        down[k] -= 1;
                                        let lowered = product(down);
                                        value.0 += ket[k] as f64 * lowered.0;
                                        value.1 += ket[k] as f64 * lowered.1;
                                    }
                                    let v = times(value, phase);
                                    cart[k].0[i * cb.len() + j] += v.0;
                                    cart[k].1[i * cb.len() + j] += v.1;
                                }
                            }
                        }
                    }
                }
            }
            std::array::from_fn(|k| {
                let mut re = vec![0.0; na * nb];
                let mut im = vec![0.0; na * nb];
                for i in 0..na {
                    for j in 0..nb {
                        for ci in 0..ca.len() {
                            for cj in 0..cb.len() {
                                let w = ta[(ci, i)] * tb[(cj, j)];
                                if w == 0.0 {
                                    continue;
                                }
                                re[i * nb + j] += w * cart[k].0[ci * cb.len() + cj];
                                im[i * nb + j] += w * cart[k].1[ci * cb.len() + cj];
                            }
                        }
                    }
                }
                (re, im)
            })
        })
        .collect();
    let mut out: [(Matrix, Matrix); 3] =
        std::array::from_fn(|_| (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)));
    for (&(a, b), block) in shell_pairs.iter().zip(&blocks) {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for k in 0..3 {
            for i in 0..na {
                for j in 0..nb {
                    out[k].0[(sa.sph_off + i, sb.sph_off + j)] = block[k].0[i * nb + j];
                    out[k].1[(sa.sph_off + i, sb.sph_off + j)] = block[k].1[i * nb + j];
                }
            }
        }
    }
    out
}

/// Lattice sums `<chi_m| exp(i G . r) |chi_n>` between the Gamma Bloch sums
/// at the reference geometry, as real and imaginary parts.
pub(crate) fn pair_matrices(cell: &Cell, wave: Vec3) -> (Matrix, Matrix) {
    let basis = &cell.basis;
    let lattice = &cell.lattice;
    let nao = basis.nao_sph;
    let ns = basis.shells.len();
    let g2 = wave.norm2();
    let times = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let shell_pairs: Vec<(usize, usize)> =
        (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let blocks: Vec<(Vec<f64>, Vec<f64>)> = shell_pairs
        .par_iter()
        .map(|&(a, b)| {
            let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
            let (la, lb) = (sa.l as usize, sb.l as usize);
            let (ca, cb) = (cart_components(sa.l), cart_components(sb.l));
            let (ta, tb) = (cart_to_sph(sa.l), cart_to_sph(sb.l));
            let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
            let nt = la + lb + 1;
            let size = ca.len() * cb.len();
            let mut cart = (vec![0.0; size], vec![0.0; size]);
            let reach = (PAIR_LOG / reduced(least(sa), least(sb)))
                .sqrt()
                .min(cell.overlap_cutoff);
            for cell_index in nearby_translation_indices(lattice, sb.center - sa.center, reach) {
                let centre_b = sb.center + lattice.translation(cell_index);
                for (&ea, &coa) in sa.exps.iter().zip(&sa.coefs) {
                    for (&eb, &cob) in sb.exps.iter().zip(&sb.coefs) {
                        let p = ea + eb;
                        let envelope = (PI / p).powf(1.5) * coa * cob * (-0.25 * g2 / p).exp();
                        let centre = (sa.center * ea + centre_b * eb) / p;
                        let angle = wave.dot(centre);
                        let phase = (envelope * angle.cos(), envelope * angle.sin());
                        let axis: Vec<Vec<(f64, f64)>> = (0..3)
                            .map(|k| {
                                let table =
                                    hermite(la, lb, sa.center.get(k) - centre_b.get(k), ea, eb);
                                let gk = wave.get(k);
                                (0..(la + 1) * (lb + 1))
                                    .map(|ij| {
                                        let (mut re, mut im) = (0.0, 0.0);
                                        let mut power = 1.0;
                                        for tau in 0..nt {
                                            let value = table[ij * nt + tau] * power;
                                            match tau % 4 {
                                                0 => re += value,
                                                1 => im += value,
                                                2 => re -= value,
                                                _ => im -= value,
                                            }
                                            power *= gk;
                                        }
                                        (re, im)
                                    })
                                    .collect()
                            })
                            .collect();
                        for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                            for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                                let value = times(
                                    times(
                                        axis[0][ax as usize * (lb + 1) + bx as usize],
                                        axis[1][ay as usize * (lb + 1) + by as usize],
                                    ),
                                    axis[2][az as usize * (lb + 1) + bz as usize],
                                );
                                let v = times(value, phase);
                                cart.0[i * cb.len() + j] += v.0;
                                cart.1[i * cb.len() + j] += v.1;
                            }
                        }
                    }
                }
            }
            let mut re = vec![0.0; na * nb];
            let mut im = vec![0.0; na * nb];
            for i in 0..na {
                for j in 0..nb {
                    for ci in 0..ca.len() {
                        for cj in 0..cb.len() {
                            let w = ta[(ci, i)] * tb[(cj, j)];
                            if w == 0.0 {
                                continue;
                            }
                            re[i * nb + j] += w * cart.0[ci * cb.len() + cj];
                            im[i * nb + j] += w * cart.1[ci * cb.len() + cj];
                        }
                    }
                }
            }
            (re, im)
        })
        .collect();
    let mut out = (Matrix::zeros(nao, nao), Matrix::zeros(nao, nao));
    for (&(a, b), block) in shell_pairs.iter().zip(&blocks) {
        let (sa, sb) = (&basis.shells[a], &basis.shells[b]);
        let (na, nb) = (n_sph(sa.l), n_sph(sb.l));
        for i in 0..na {
            for j in 0..nb {
                out.0[(sa.sph_off + i, sb.sph_off + j)] = block.0[i * nb + j];
                out.1[(sa.sph_off + i, sb.sph_off + j)] = block.1[i * nb + j];
            }
        }
    }
    out
}

#[cfg(test)]
mod charged_tests {
    use super::*;
    use crate::basis::data;
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};

    #[test]
    fn charged_monopole_shape_derivatives_fit_every_symmetric_component() {
        use crate::acceptance::{components, stencils, symmetric_cartesian_directions};
        let ion = Molecule::new(vec![Atom {
            z: 2,
            position: Vec3::new(0.17, 0.41, -0.19),
        }])
        .with_charge(1);
        let lattice = Lattice::new(
            Vec3::new(8.4, 0.7, 0.3),
            Vec3::new(0.4, 8.9, 0.6),
            Vec3::new(0.2, -0.5, 9.2),
        );
        let basis = data::load("sto-3g").unwrap();
        let cell = Cell::new(ion, lattice.clone(), &basis, 5.5).unwrap();
        let zero = cell.charged_cell_correction();
        let mut failures = Vec::new();
        for order in 1..=4 {
            let directions = symmetric_cartesian_directions::directions(6, order);
            let labels: Vec<_> = (0..directions.len())
                .map(|i| format!("charged monopole order {order} direction {i}"))
                .collect();
            let matrices: Vec<_> = directions
                .iter()
                .map(|v| {
                    [
                        [v[0], v[5] * 0.5, v[4] * 0.5],
                        [v[5] * 0.5, v[1], v[3] * 0.5],
                        [v[4] * 0.5, v[3] * 0.5, v[2]],
                    ]
                })
                .collect();
            let values: Vec<_> = matrices
                .iter()
                .map(|&deformation| {
                    let path = LatticePath {
                        deformation: vec![deformation],
                        ..Default::default()
                    };
                    let coefficients = monopole_path(&cell, &inputs(&path, 1, 4), 4).unwrap();
                    let first = super::super::ewald::monopole_strain(&lattice, cell.net_charge());
                    let closed = first
                        .iter()
                        .flatten()
                        .zip(deformation.iter().flatten())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    assert!((coefficients[1] - closed).abs() < 1e-11);
                    coefficients[order] * (1..=order).product::<usize>() as f64
                })
                .collect();
            stencils::fit_components_staged(
                &labels,
                &[
                    ([0.04, 0.03, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                    ([0.04, 0.03, 0.02, 0.01], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    matrices
                        .iter()
                        .zip(&values)
                        .map(|(strain, &exact)| {
                            let energy = |h: f64| {
                                let deformed = lattice.a.map(|v| {
                                    let source = v.to_array();
                                    let xyz: [f64; 3] = std::array::from_fn(|a| {
                                        source[a]
                                            + h * (0..3)
                                                .map(|b| strain[a][b] * source[b])
                                                .sum::<f64>()
                                    });
                                    Vec3::new(xyz[0], xyz[1], xyz[2])
                                });
                                super::super::ewald::makov_payne_correction(
                                    &Lattice::new(deformed[0], deformed[1], deformed[2]),
                                    cell.net_charge(),
                                )
                            };
                            let derivative = |h: f64| match order {
                                1 => stencils::Stencil::Second.first(
                                    energy(h),
                                    energy(-h),
                                    0.0,
                                    0.0,
                                    h,
                                ),
                                2 => (energy(h) + energy(-h) - 2.0 * zero) / h.powi(2),
                                3 => {
                                    (energy(2.0 * h)
                                        - energy(-2.0 * h)
                                        - 2.0 * (energy(h) - energy(-h)))
                                        / (2.0 * h.powi(3))
                                }
                                _ => {
                                    (energy(2.0 * h) + energy(-2.0 * h)
                                        - 4.0 * (energy(h) + energy(-h))
                                        + 6.0 * zero)
                                        / h.powi(4)
                                }
                            };
                            steps.map(|h| {
                                let near = derivative(h);
                                let measured = if stencil == stencils::Stencil::Fourth {
                                    (4.0 * near - derivative(2.0 * h)) / 3.0
                                } else {
                                    near
                                };
                                (measured - exact).abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
        components::assert_no_failures("charged monopole shape derivatives", &failures);
    }

    #[test]
    fn fixed_lattice_series_includes_the_charged_nuclear_reference_energy() {
        let ion = Molecule::new(vec![
            Atom {
                z: 2,
                position: Vec3::new(0.17, 0.41, -0.19),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.49, -0.12, 1.19),
            },
        ])
        .with_charge(1);
        let lattice = Lattice::new(
            Vec3::new(8.4, 0.7, 0.3),
            Vec3::new(0.4, 8.9, 0.6),
            Vec3::new(0.2, -0.5, 9.2),
        );
        let basis = data::load("sto-3g").unwrap();
        let cell = Cell::new(ion, lattice, &basis, 5.5).unwrap();
        let series = lattice_series(&LatticeRequest {
            cell: &cell,
            path: &LatticePath::default(),
            order: 0,
            grid: None,
            exchange: None,
            long_range: None,
        })
        .unwrap();
        // The ordinary Ewald SCF computes this term with a separate summation.
        let expected = cell.nuclear_energy();
        let defect = series.nuclear[0] - expected;
        println!(
            "charged nuclear energy={:.17e} Ewald SCF={expected:.17e} monopole={:.17e} defect={defect:.3e}",
            series.nuclear[0],
            cell.charged_cell_correction()
        );
        assert!(defect.abs() < 2e-9);
    }
}
