// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular quadrature grid for the XC integration, in the form that is
//! the common standard of quantum-chemistry programs: Treutler–Ahlrichs M4
//! radial maps (Gauss–Chebyshev second kind) × Lebedev–Laikov angular rules
//! (octahedral, exact to degree 3–131), five-region angular pruning near
//! and far from each nucleus, and Becke fuzzy-cell partitioning with the
//! Treutler size adjustment.
//!
//! The Lebedev rules are the published orbit generators
//! (`lebedev_data.rs`), expanded and checked against closed-form monomial
//! integrals; the self-generated product rule (Gauss–Legendre in cosθ ×
//! uniform φ) stays selectable as [`AngularScheme::Product`].
//!
//! Resolution follows a one-knob `level` (0–9, default 3) whose radial
//! counts and angular degrees depend on the period of the element, or a
//! named [`GridPreset`] with fixed (radial, angular) counts; both are
//! refinable per element and per atom. Becke weights form a partition of
//! unity, so radii/ξ tables affect only the convergence *rate*, never the
//! converged integral.

pub mod eval;
mod lebedev;
mod lebedev_data;

use crate::error::Result;
use crate::math::Vec3;
use crate::system::Molecule;
use rayon::prelude::*;
use std::borrow::Cow;

/// Angular quadrature backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AngularScheme {
    /// Lebedev–Laikov rules; a requested degree is rounded up to the next
    /// published rule (degrees above 131 use the product rule).
    #[default]
    Lebedev,
    /// Gauss–Legendre in cosθ × uniform φ, exact to the requested degree.
    Product,
}

impl AngularScheme {
    /// Parse a control value (`lebedev` or `product`).
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "lebedev" => Ok(Self::Lebedev),
            "product" => Ok(Self::Product),
            other => Err(crate::DftError::InvalidInput(format!(
                "unknown grid_angular_scheme {other:?} (lebedev or product)"
            ))),
        }
    }
}

/// Named grids with fixed per-atom (radial points, Lebedev points), the
/// sizes conventionally called Fine (75, 302), UltraFine (99, 590) and
/// SuperFine (150 for H–Ne and 225 beyond, 974). They use this module's
/// radial map, pruning and partition, so energies are not those of another
/// program's grid of the same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridPreset {
    Fine,
    UltraFine,
    SuperFine,
}

impl GridPreset {
    /// Parse a control value; `level` (or an empty value) means no preset.
    pub fn parse(name: &str) -> Result<Option<Self>> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "" | "level" => Ok(None),
            "fine" => Ok(Some(Self::Fine)),
            "ultrafine" => Ok(Some(Self::UltraFine)),
            "superfine" => Ok(Some(Self::SuperFine)),
            other => Err(crate::DftError::InvalidInput(format!(
                "unknown grid_preset {other:?} (level, fine, ultrafine or superfine)"
            ))),
        }
    }

    /// (radial points, angular degree) for an element.
    fn counts(self, z: u8) -> (usize, usize) {
        match self {
            Self::Fine => (75, 29),
            Self::UltraFine => (99, 41),
            Self::SuperFine => (if z <= 10 { 150 } else { 225 }, 53),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GridOptions {
    /// One-knob resolution 0..=9 (default 3).
    pub level: usize,
    /// Reduce angular degree near nuclei.
    pub prune: bool,
    pub scheme: AngularScheme,
    /// Named grid replacing the `level` table.
    pub preset: Option<GridPreset>,
    /// Global overrides (radial points / angular degree).
    pub radial_override: Option<usize>,
    pub angular_override: Option<usize>,
    /// Per-element overrides `(Z, n_radial, angular_degree)`.
    pub per_element: Vec<(u8, usize, usize)>,
    /// Per-atom overrides `(atom index, n_radial, angular_degree)` —
    /// strongest priority ("this atom finer/coarser").
    pub per_atom: Vec<(usize, usize, usize)>,
    /// Points with partition-weighted |w| below this are dropped.
    pub weight_cutoff: f64,
}

impl Default for GridOptions {
    fn default() -> Self {
        Self {
            level: 3,
            prune: true,
            scheme: AngularScheme::Lebedev,
            preset: None,
            radial_override: None,
            angular_override: None,
            per_element: Vec::new(),
            per_atom: Vec::new(),
            weight_cutoff: 1e-14,
        }
    }
}

impl GridOptions {
    /// Validate controls before allocating quadrature points.
    pub fn validate(&self) -> Result<()> {
        if self.level > 9 {
            return Err(crate::DftError::InvalidInput(
                "grid_level must be between 0 and 9".into(),
            ));
        }
        if !self.weight_cutoff.is_finite() || self.weight_cutoff < 0.0 {
            return Err(crate::DftError::InvalidInput(
                "grid_weight_cutoff must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }

    /// Options from the `grid_level`, `grid_preset`, `grid_angular_scheme`
    /// and `grid_weight_cutoff` controls.
    pub fn from_controls(
        level: usize,
        preset: Option<&str>,
        scheme: Option<&str>,
        weight_cutoff: Option<f64>,
    ) -> Result<Self> {
        if level > 9 {
            return Err(crate::DftError::InvalidInput(
                "grid_level must be between 0 and 9".into(),
            ));
        }
        let weight_cutoff = weight_cutoff.unwrap_or(1e-14);
        if !weight_cutoff.is_finite() || weight_cutoff < 0.0 {
            return Err(crate::DftError::InvalidInput(
                "grid_weight_cutoff must be finite and non-negative".into(),
            ));
        }
        Ok(Self {
            weight_cutoff,
            level,
            preset: preset.map(GridPreset::parse).transpose()?.flatten(),
            scheme: scheme
                .map(AngularScheme::parse)
                .transpose()?
                .unwrap_or_default(),
            ..Self::default()
        })
    }

    /// Resolve `(n_radial, angular_degree)` for one atom.
    ///
    /// The level table gives radial points and angular degree by period
    /// (H–He, Li–Ne, Na–Ar, K–Kr, Rb–Xe, Cs–Rn, beyond), the values of the
    /// standard Treutler–Ahlrichs/Lebedev grids of common practice
    /// (level 3: 50/75/80/90 radial points and degree 29/29/35/35 from H–He
    /// to K–Kr).
    pub(crate) fn resolve(&self, atom_idx: usize, z: u8) -> (usize, usize) {
        #[rustfmt::skip]
        const NRAD: [[usize; 7]; 10] = [
            [10, 15, 20, 30, 35, 40, 50],
            [30, 40, 50, 60, 65, 70, 75],
            [40, 60, 65, 75, 80, 85, 90],
            [50, 75, 80, 90, 95, 100, 105],
            [60, 90, 95, 105, 110, 115, 120],
            [70, 105, 110, 120, 125, 130, 135],
            [80, 120, 125, 135, 140, 145, 150],
            [90, 135, 140, 150, 155, 160, 165],
            [100, 150, 155, 165, 170, 175, 180],
            [200, 200, 200, 200, 200, 200, 200],
        ];
        #[rustfmt::skip]
        const DEGREE: [[usize; 7]; 10] = [
            [11, 15, 17, 17, 17, 17, 17],
            [17, 23, 23, 23, 23, 23, 23],
            [23, 29, 29, 29, 29, 29, 29],
            [29, 29, 35, 35, 35, 35, 35],
            [35, 41, 41, 41, 41, 41, 41],
            [41, 47, 47, 47, 47, 47, 47],
            [47, 53, 53, 53, 53, 53, 53],
            [53, 59, 59, 59, 59, 59, 59],
            [59, 59, 59, 59, 59, 59, 59],
            [65, 65, 65, 65, 65, 65, 65],
        ];
        const PERIOD_ENDS: [u8; 6] = [2, 10, 18, 36, 54, 86];
        let level = self.level.min(9);
        let period = PERIOD_ENDS.iter().filter(|&&end| z > end).count();
        let (mut n_rad, mut degree) = match self.preset {
            Some(preset) => preset.counts(z),
            None => (NRAD[level][period], DEGREE[level][period]),
        };
        if let Some(r) = self.radial_override {
            n_rad = r;
        }
        if let Some(a) = self.angular_override {
            degree = a;
        }
        for &(ze, r, d) in &self.per_element {
            if ze == z {
                n_rad = r;
                degree = d;
            }
        }
        for &(ai, r, d) in &self.per_atom {
            if ai == atom_idx {
                n_rad = r;
                degree = d;
            }
        }
        (n_rad.max(5), degree.max(3))
    }
}

/// The assembled molecular grid.
#[derive(Clone, Debug)]
pub struct MolecularGrid {
    pub points: Vec<Vec3>,
    /// Radial × angular × Becke partition weights.
    pub weights: Vec<f64>,
    /// Radial × angular weights WITHOUT the Becke partition factor
    /// (needed by the gradient's weight-response term).
    pub raw_weights: Vec<f64>,
    /// Owning atom of each point.
    pub atom_of: Vec<usize>,
}

impl MolecularGrid {
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn build(mol: &Molecule, opts: &GridOptions) -> Result<Self> {
        let radii: Vec<f64> = mol.atoms.iter().map(|a| bragg_slater_bohr(a.z)).collect();
        Self::build_with_radii(mol, opts, &radii)
    }

    /// Number of points per atom before the partition-weight cutoff.
    pub fn atomic_point_counts(mol: &Molecule, opts: &GridOptions) -> Vec<usize> {
        mol.atoms
            .iter()
            .enumerate()
            .map(|(ia, atom)| {
                let (n_rad, degree) = opts.resolve(ia, atom.z);
                let (r_pts, _) = radial_treutler_ahlrichs(n_rad, treutler_xi(atom.z));
                let radius = bragg_slater_bohr(atom.z);
                r_pts
                    .iter()
                    .map(|r| {
                        let deg = pruned_degree(atom.z, *r, radius, degree, opts.prune);
                        angular_rule(opts.scheme, deg).len()
                    })
                    .sum()
            })
            .collect()
    }

    /// [`Self::build`] with explicit per-atom Becke partition radii (Bohr);
    /// the radial map and the pruning regions keep the element defaults.
    pub(crate) fn build_with_radii(
        mol: &Molecule,
        opts: &GridOptions,
        radii: &[f64],
    ) -> Result<Self> {
        opts.validate()?;
        let centers: Vec<Vec3> = mol.atoms.iter().map(|a| a.position).collect();
        let n_at = mol.len();
        let aij = partition_aij(radii);

        let per_atom: Vec<(Vec<Vec3>, Vec<f64>, Vec<f64>, usize)> = (0..n_at)
            .into_par_iter()
            .map(|ia| {
                let z = mol.atoms[ia].z;
                let (n_rad, degree) = opts.resolve(ia, z);
                let (r_pts, r_wts) = radial_treutler_ahlrichs(n_rad, treutler_xi(z));
                let prune_radius = bragg_slater_bohr(z);
                let mut pts = Vec::new();
                let mut wts = Vec::new();
                let mut raws = Vec::new();
                for (r, wr) in r_pts.iter().zip(&r_wts) {
                    // Pruning regions follow the element's Bragg radius, not
                    // the partition radii a caller may vary.
                    let deg = pruned_degree(z, *r, prune_radius, degree, opts.prune);
                    for &(dir, wa) in angular_rule(opts.scheme, deg).iter() {
                        let p = centers[ia] + dir * *r;
                        // Becke partition weight.
                        let wb = becke_weight(&centers, &aij, n_at, ia, p);
                        let raw = wr * wa;
                        let w = raw * wb;
                        if w.abs() >= opts.weight_cutoff {
                            pts.push(p);
                            wts.push(w);
                            raws.push(raw);
                        }
                    }
                }
                (pts, wts, raws, ia)
            })
            .collect();

        let mut points = Vec::new();
        let mut weights = Vec::new();
        let mut raw_weights = Vec::new();
        let mut atom_of = Vec::new();
        for (pts, wts, raws, ia) in per_atom {
            atom_of.extend(std::iter::repeat_n(ia, pts.len()));
            points.extend(pts);
            weights.extend(wts);
            raw_weights.extend(raws);
        }
        Ok(Self {
            points,
            weights,
            raw_weights,
            atom_of,
        })
    }

    /// Becke size-adjustment table for external weight-derivative callers.
    pub fn becke_aij(mol: &Molecule) -> Vec<f64> {
        let radii: Vec<f64> = mol.atoms.iter().map(|a| bragg_slater_bohr(a.z)).collect();
        partition_aij(&radii)
    }
}

/// Pairwise Becke size adjustment `a_ij = (1/χ − χ)/4`, clamped to ±1/2,
/// with Treutler's ratio χ = √(R_i/R_j) of the Bragg radii (JCP 102, 346
/// (1995)); Becke's original form uses χ = R_i/R_j.
pub(crate) fn partition_aij(radii: &[f64]) -> Vec<f64> {
    let n_at = radii.len();
    let mut aij = vec![0.0; n_at * n_at];
    for i in 0..n_at {
        for j in 0..n_at {
            if i != j {
                let chi = (radii[i] / radii[j]).sqrt();
                aij[i * n_at + j] = (0.25 * (1.0 / chi - chi)).clamp(-0.5, 0.5);
            }
        }
    }
    aij
}

/// Angular rule of at least `degree` for the scheme, as unit directions and
/// weights summing to 4π.
pub fn angular_rule(scheme: AngularScheme, degree: usize) -> Cow<'static, [(Vec3, f64)]> {
    match scheme {
        AngularScheme::Lebedev => match lebedev::rule(degree) {
            Some(rule) => Cow::Borrowed(rule),
            None => Cow::Owned(angular_product_grid(degree)),
        },
        AngularScheme::Product => Cow::Owned(angular_product_grid(degree)),
    }
}

/// Becke fuzzy-cell weight AND its analytic derivatives ∂w/∂R_G (point
/// fixed in space) for every atom G. `s'(ν)/s(ν)` chains through the three
/// smoothing iterations; near-vanishing cells are skipped (their P_B ≈ 0).
pub fn becke_weight_gradient(
    centers: &[Vec3],
    aij: &[f64],
    n_at: usize,
    ia: usize,
    p: Vec3,
) -> (f64, Vec<Vec3>) {
    let mut grads = vec![Vec3::zero(); n_at];
    if n_at == 1 {
        return (1.0, grads);
    }
    let dist: Vec<f64> = centers.iter().map(|c| (p - *c).norm()).collect();
    // Cell values and t_BC = (ds/dν)/s per pair.
    let mut cell = vec![1.0; n_at];
    // ∂P_B/∂R_G accumulated as P_B · Σ t·∂ν/∂R_G.
    let mut dcell = vec![Vec3::zero(); n_at * n_at]; // [B][G]
    for b in 0..n_at {
        for c in 0..n_at {
            if c == b {
                continue;
            }
            let rbc_vec = centers[b] - centers[c];
            let rbc = rbc_vec.norm();
            let mu = (dist[b] - dist[c]) / rbc;
            let a = aij[b * n_at + c];
            let nu = mu + a * (1.0 - mu * mu);
            // s(ν) and ds/dν through p∘p∘p.
            let mut f = nu;
            let mut dp_chain = 1.0;
            for _ in 0..3 {
                dp_chain *= 1.5 - 1.5 * f * f;
                f = 1.5 * f - 0.5 * f * f * f;
            }
            let s = 0.5 * (1.0 - f);
            let ds = -0.5 * dp_chain;
            cell[b] *= s;
            if s.abs() < 1e-14 {
                continue;
            }
            let t = ds / s * (1.0 - 2.0 * a * mu); // includes dν/dμ
                                                   // ∂μ/∂R_B and ∂μ/∂R_C (point fixed).
            let db = dist[b].max(1e-300);
            let dc = dist[c].max(1e-300);
            let dmu_db =
                (centers[b] - p) / (db * rbc) - rbc_vec * ((dist[b] - dist[c]) / (rbc * rbc * rbc));
            let dmu_dc =
                (p - centers[c]) / (dc * rbc) + rbc_vec * ((dist[b] - dist[c]) / (rbc * rbc * rbc));
            dcell[b * n_at + b] += dmu_db * t;
            dcell[b * n_at + c] += dmu_dc * t;
        }
    }
    let z: f64 = cell.iter().sum();
    if z <= 0.0 {
        return (0.0, grads);
    }
    let w = cell[ia] / z;
    // ∂w/∂R_G = [P_ia·t_ia,G·Z − P_ia Σ_B P_B t_B,G]/Z²
    // with ∂P_B/∂R_G = P_B·dcell[B][G].
    for g in 0..n_at {
        let dp_ia = dcell[ia * n_at + g] * cell[ia];
        let mut dz = Vec3::zero();
        for b in 0..n_at {
            dz += dcell[b * n_at + g] * cell[b];
        }
        grads[g] = (dp_ia * z - dz * cell[ia]) / (z * z);
    }
    (w, grads)
}

/// Becke fuzzy-cell weight of atom `ia` at point `p` (k = 3 iterations,
/// size-adjusted). Partition of unity by construction.
pub(crate) fn becke_weight(centers: &[Vec3], aij: &[f64], n_at: usize, ia: usize, p: Vec3) -> f64 {
    if n_at == 1 {
        return 1.0;
    }
    let dist: Vec<f64> = centers.iter().map(|c| (p - *c).norm()).collect();
    let mut num = 0.0;
    let mut den = 0.0;
    for b in 0..n_at {
        let mut cell = 1.0;
        for c in 0..n_at {
            if c == b {
                continue;
            }
            let rbc = (centers[b] - centers[c]).norm();
            let mu = (dist[b] - dist[c]) / rbc;
            let nu = mu + aij[b * n_at + c] * (1.0 - mu * mu);
            // s(ν) = ½(1 − p(p(p(ν)))) with p(x) = 1.5x − 0.5x³.
            let mut f = nu;
            for _ in 0..3 {
                f = 1.5 * f - 0.5 * f * f * f;
            }
            cell *= 0.5 * (1.0 - f);
            if cell == 0.0 {
                break;
            }
        }
        if b == ia {
            num = cell;
        }
        den += cell;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Angular degree at radius `r` around an atom of charge `z` with Bragg
/// radius `radius` (Bohr): the five-region pruning of Treutler-Ahlrichs and
/// NWChem (the scheme of PySCF's default grids). The region boundaries are
/// `alpha * radius` with `alpha` = (0.25, 0.5, 1.0, 4.5) for H and He,
/// (0.1667, 0.5, 0.9, 3.5) for Li-Ne and (0.1, 0.4, 0.8, 2.5) beyond, and the
/// degrees from the nucleus outwards are 11, 15, the next lower rule, the
/// full degree and the next lower rule again, along the ladder of Lebedev
/// degrees (a degree between two rules is first rounded up). Grids of
/// degree 11 or less are not pruned.
pub(crate) fn pruned_degree(z: u8, r: f64, radius: f64, degree: usize, prune: bool) -> usize {
    if !prune || degree <= 11 {
        return degree;
    }
    let degree = lebedev::degrees().find(|&d| d >= degree).unwrap_or(degree);
    let alpha: [f64; 4] = match z {
        0..=2 => [0.25, 0.5, 1.0, 4.5],
        3..=10 => [0.1667, 0.5, 0.9, 3.5],
        _ => [0.1, 0.4, 0.8, 2.5],
    };
    let lower = lebedev::degrees()
        .take_while(|&d| d < degree)
        .last()
        .unwrap_or(degree);
    let region = alpha.iter().filter(|&&a| r > a * radius).count();
    [11, 15.min(degree), lower, degree, lower][region]
}

/// Treutler–Ahlrichs M4 radial quadrature: Gauss–Chebyshev (2nd kind)
/// abscissae with the mapping r = (ξ/ln2)·(1+x)^0.6·ln(2/(1−x)).
/// Returns points and weights INCLUDING the r² volume factor.
pub fn radial_treutler_ahlrichs(n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    const ALPHA: f64 = 0.6;
    let ln2 = std::f64::consts::LN_2;
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    for i in 1..=n {
        let theta = i as f64 * std::f64::consts::PI / (n + 1) as f64;
        let x = theta.cos();
        let w_gc = std::f64::consts::PI / (n + 1) as f64 * theta.sin();
        let one_px = 1.0 + x;
        let one_mx = 1.0 - x;
        if one_mx <= 0.0 || one_px <= 0.0 {
            continue;
        }
        let lg = (2.0 / one_mx).ln();
        let r = xi / ln2 * one_px.powf(ALPHA) * lg;
        let drdx = xi / ln2 * (ALPHA * one_px.powf(ALPHA - 1.0) * lg + one_px.powf(ALPHA) / one_mx);
        pts.push(r);
        wts.push(w_gc * drdx * r * r);
    }
    (pts, wts)
}

/// Resolve the small-radius part with Gauss intervals and translate the
/// ordinary radial rule onto the remaining half-line. Both parts integrate
/// the radial volume factor.
pub(crate) fn radial_treutler_split(n: usize, xi: f64, split: f64) -> (Vec<f64>, Vec<f64>) {
    let (mut radii, mut weights) = radial_treutler_ahlrichs(n, xi);
    for (r, w) in radii.iter_mut().zip(&mut weights) {
        let shifted = *r + split;
        *w *= (shifted / *r).powi(2);
        *r = shifted;
    }
    let (gauss, gauss_weights) = gauss_legendre(n.div_ceil(16));
    let bounds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0];
    for panel in bounds.windows(2) {
        let center = 0.5 * split * (panel[0] + panel[1]);
        let half = 0.5 * split * (panel[1] - panel[0]);
        for (x, w) in gauss.iter().zip(&gauss_weights) {
            let r = center + half * x;
            radii.push(r);
            weights.push(half * w * r * r);
        }
    }
    (radii, weights)
}

/// Element scale ξ for the M4 map: the Treutler–Ahlrichs values for H–Kr
/// (JCP 102, 346 (1995)) and the values of the common grids of practice
/// beyond (1.5 where no value is assigned); convergence-rate parameter
/// only.
pub fn treutler_xi(z: u8) -> f64 {
    #[rustfmt::skip]
    const XI: [f64; 104] = [
        1.0, 0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9, 1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0,
        1.0, 1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9,
        0.9, 2.0, 1.7, 1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5, 1.3, 1.2, 1.2, 1.15,
        1.15, 1.15, 2.5, 2.2, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5,
        2.1, 3.685, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    ];
    XI.get(z as usize).copied().unwrap_or(1.5)
}

/// Bragg–Slater radius (Bohr), Z = 0..=36 (dummy, H..Kr).
///
/// Source: J. C. Slater, "Atomic Radii in Crystals", J. Chem. Phys. **41**,
/// 3199 (1964), DOI:10.1063/1.1725697 — the empirical/bonding radii table,
/// with H raised from Slater's 0.25 Å to 0.35 Å per Becke's 1988
/// multicenter-integration convention (A. D. Becke, J. Chem. Phys. **88**,
/// 2547 (1988)). This is the same radii set used by PySCF's
/// `dft.radi.BRAGG_RADII` / `pyscf.data.radii.BRAGG`, verified entry-by-entry
/// against that table for Z = 1..=36.
///
/// Previously the noble-gas slots (He/Ne/Ar/Kr) held Clementi *calculated*
/// radii (0.31/0.38/0.71/1.10 Å) instead of the Slater/Becke bonding radii
/// (1.40/1.50/1.80/1.90 Å) — Slater's original table uses the same
/// extrapolated-trend values as the other rows for these closed-shell atoms,
/// not the smaller calculated ones. Mixing the two conventions flips the
/// sign of the Becke `a_ij` size adjustment for He/Ne/Ar/Kr centres (see
/// `chi = R_i/R_j` above), degrading fuzzy-cell partitioning around noble
/// gases. All non-noble-gas entries (Z = 1, 3–9, 11–17, 19–35) already
/// matched Slater/PySCF and are unchanged. Z = 0 is an unused dummy slot
/// (ghost atoms are rejected elsewhere, see `basis::Basis::build`); set to
/// PySCF's convention value of 1.00 Å.
///
/// Defaults to 1.35 Å beyond Kr (Z > 36); affects only partition sharpness,
/// never the converged (infinite-grid) integral, since Becke weights are a
/// partition of unity regardless of the radii used.
pub fn bragg_slater_bohr(z: u8) -> f64 {
    #[rustfmt::skip]
    const ANGSTROM: [f64; 37] = [
        1.00, // 0  dummy (unused; Z=0 ghost atoms are rejected in Basis::build)
        0.35, // 1  H  (Becke 1988 convention; Slater's raw value is 0.25)
        1.40, // 2  He (was 0.31 Clementi-calculated; fixed to Slater/PySCF)
        1.45, // 3  Li
        1.05, // 4  Be
        0.85, // 5  B
        0.70, // 6  C
        0.65, // 7  N
        0.60, // 8  O
        0.50, // 9  F
        1.50, // 10 Ne (was 0.38 Clementi-calculated; fixed to Slater/PySCF)
        1.80, // 11 Na
        1.50, // 12 Mg
        1.25, // 13 Al
        1.10, // 14 Si
        1.00, // 15 P
        1.00, // 16 S
        1.00, // 17 Cl
        1.80, // 18 Ar (was 0.71 Clementi-calculated; fixed to Slater/PySCF)
        2.20, // 19 K
        1.80, // 20 Ca
        1.60, // 21 Sc
        1.40, // 22 Ti
        1.35, // 23 V
        1.40, // 24 Cr
        1.40, // 25 Mn
        1.40, // 26 Fe
        1.35, // 27 Co
        1.35, // 28 Ni
        1.35, // 29 Cu
        1.35, // 30 Zn
        1.30, // 31 Ga
        1.25, // 32 Ge
        1.15, // 33 As
        1.15, // 34 Se
        1.15, // 35 Br
        1.90, // 36 Kr (was 1.10 Clementi-calculated; fixed to Slater/PySCF)
    ];
    let a = ANGSTROM.get(z as usize).copied().unwrap_or(1.35);
    a * crate::constants::ANGSTROM_TO_BOHR
}

/// Product angular rule exact for spherical polynomials up to `degree`:
/// Gauss–Legendre in cosθ × uniform φ. Returns unit directions and weights
/// summing to 4π.
pub fn angular_product_grid(degree: usize) -> Vec<(Vec3, f64)> {
    let n_theta = degree / 2 + 1;
    let n_phi = degree + 1;
    let (nodes, weights) = gauss_legendre(n_theta);
    let wphi = 2.0 * std::f64::consts::PI / n_phi as f64;
    let mut out = Vec::with_capacity(n_theta * n_phi);
    for (ct, wt) in nodes.iter().zip(&weights) {
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        for k in 0..n_phi {
            let phi = 2.0 * std::f64::consts::PI * k as f64 / n_phi as f64;
            let dir = Vec3::new(st * phi.cos(), st * phi.sin(), *ct);
            out.push((dir, wt * wphi));
        }
    }
    out
}

/// Gauss–Legendre nodes/weights on [−1, 1] by Newton iteration on P_n.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre_p_dp(n, x);
            let dx = -p / dp;
            x += dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (_, dp) = legendre_p_dp(n, x);
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

fn legendre_p_dp(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 2..=n {
        let pk = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = pk;
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::double_factorial;
    use crate::system::Atom;

    #[test]
    fn invalid_grid_controls_are_rejected_before_construction() {
        let molecule = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }]);
        for options in [
            GridOptions {
                level: 10,
                ..Default::default()
            },
            GridOptions {
                level: usize::MAX,
                ..Default::default()
            },
            GridOptions {
                weight_cutoff: f64::NAN,
                ..Default::default()
            },
            GridOptions {
                weight_cutoff: f64::INFINITY,
                ..Default::default()
            },
            GridOptions {
                weight_cutoff: -1.0,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                MolecularGrid::build(&molecule, &options),
                Err(crate::DftError::InvalidInput(_))
            ));
        }
    }

    /// Self-verification: the angular rule must integrate every monomial
    /// x^a y^b z^c with a+b+c ≤ degree exactly:
    /// ∫ dΩ = 4π (a−1)!!(b−1)!!(c−1)!!/(a+b+c+1)!! (all even), else 0.
    #[test]
    fn angular_rules_are_exact_to_their_degree() {
        for degree in [3, 9, 15, 23, 35, 47, 59] {
            let rule = angular_product_grid(degree);
            let wsum: f64 = rule.iter().map(|(_, w)| w).sum();
            assert!((wsum - 4.0 * std::f64::consts::PI).abs() < 1e-11);
            for a in 0..=degree {
                for b in 0..=(degree - a) {
                    for c in 0..=(degree - a - b) {
                        let got: f64 = rule
                            .iter()
                            .map(|(d, w)| {
                                w * d.x.powi(a as i32) * d.y.powi(b as i32) * d.z.powi(c as i32)
                            })
                            .sum();
                        let want = if a % 2 == 0 && b % 2 == 0 && c % 2 == 0 {
                            4.0 * std::f64::consts::PI
                                * double_factorial(a as i32 - 1)
                                * double_factorial(b as i32 - 1)
                                * double_factorial(c as i32 - 1)
                                / double_factorial((a + b + c) as i32 + 1)
                        } else {
                            0.0
                        };
                        assert!(
                            (got - want).abs() < 1e-11,
                            "degree {degree}: monomial ({a},{b},{c}) = {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    /// Radial rule: ∫₀^∞ e^{−r²} r² dr = √π/4 (and with r⁴: 3√π/8).
    #[test]
    fn radial_rule_gaussian_moments() {
        let (r, w) = radial_treutler_ahlrichs(60, 1.0);
        let i2: f64 = r.iter().zip(&w).map(|(r, w)| w * (-r * r).exp()).sum();
        assert!(
            (i2 - std::f64::consts::PI.sqrt() / 4.0).abs() < 1e-10,
            "{i2}"
        );
        let i4: f64 = r
            .iter()
            .zip(&w)
            .map(|(r, w)| w * r * r * (-r * r).exp())
            .sum();
        assert!(
            (i4 - 3.0 * std::f64::consts::PI.sqrt() / 8.0).abs() < 1e-10,
            "{i4}"
        );
    }

    /// Becke weights are a partition of unity at arbitrary points.
    #[test]
    fn becke_partition_of_unity() {
        let centers = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.1),
            Vec3::new(1.5, -0.4, 0.9),
        ];
        let radii = [1.0, 0.7, 1.3];
        let n = 3;
        let mut aij = vec![0.0; 9];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let chi: f64 = radii[i] / radii[j];
                    let u = (chi - 1.0) / (chi + 1.0);
                    aij[i * n + j] = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
                }
            }
        }
        for &p in &[
            Vec3::new(0.3, 0.2, 0.5),
            Vec3::new(-1.0, 0.8, 1.7),
            Vec3::new(0.7, -0.2, 2.5),
        ] {
            let total: f64 = (0..n)
                .map(|ia| becke_weight(&centers, &aij, n, ia, p))
                .sum();
            assert!((total - 1.0).abs() < 1e-12, "partition sum {total}");
        }
    }

    /// Becke weight derivatives vs central FD (moving each atom, point
    /// fixed in space).
    #[test]
    fn becke_weight_gradient_matches_fd() {
        let mk = |shift: Option<(usize, usize, f64)>| -> Vec<Vec3> {
            let mut c = vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.2, 2.1),
                Vec3::new(1.5, -0.4, 0.9),
            ];
            if let Some((atom, ax, h)) = shift {
                match ax {
                    0 => c[atom].x += h,
                    1 => c[atom].y += h,
                    _ => c[atom].z += h,
                }
            }
            c
        };
        let radii = [1.0, 0.7, 1.3];
        let n = 3;
        let mut aij = vec![0.0; 9];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let chi: f64 = radii[i] / radii[j];
                    let u = (chi - 1.0) / (chi + 1.0);
                    aij[i * n + j] = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
                }
            }
        }
        let p = Vec3::new(0.4, 0.3, 0.8);
        let h = 1e-6;
        for ia in 0..n {
            let (w0, grads) = becke_weight_gradient(&mk(None), &aij, n, ia, p);
            assert!(w0.is_finite());
            for g in 0..n {
                for ax in 0..3 {
                    let wp = becke_weight(&mk(Some((g, ax, h))), &aij, n, ia, p);
                    let wm = becke_weight(&mk(Some((g, ax, -h))), &aij, n, ia, p);
                    let fd = (wp - wm) / (2.0 * h);
                    let an = grads[g].get(ax);
                    assert!(
                        (an - fd).abs() < 1e-7,
                        "w_{ia} ∂/∂R_{g}[{ax}]: {an} vs {fd}"
                    );
                }
            }
        }
    }

    /// Grid-level machinery: a full molecular grid integrates a known
    /// Gaussian density to its exact number of electrons.
    #[test]
    fn molecular_grid_integrates_gaussian() {
        // A single "atom" carrying a normalized s Gaussian density
        // ρ(r) = N² e^{−2αr²} with ∫ρ = 1.
        let mol = crate::system::Molecule::new(vec![Atom {
            z: 8,
            position: Vec3::new(0.4, -0.2, 0.1),
        }]);
        let grid = MolecularGrid::build(&mol, &GridOptions::default()).unwrap();
        let alpha = 1.3;
        let norm2 = (2.0 * alpha / std::f64::consts::PI).powf(1.5);
        let integral: f64 = grid
            .points
            .iter()
            .zip(&grid.weights)
            .map(|(p, w)| {
                let r2 = (*p - mol.atoms[0].position).norm2();
                w * norm2 * (-2.0 * alpha * r2).exp()
            })
            .sum();
        assert!((integral - 1.0).abs() < 1e-8, "∫ρ = {integral}");
        // Per-atom override machinery: coarse override must still be sane.
        let mut coarse = GridOptions::default();
        coarse.per_atom.push((0, 30, 11));
        let grid2 = MolecularGrid::build(&mol, &coarse).unwrap();
        assert!(grid2.len() < grid.len());
    }

    /// Build a molecular grid exactly like [`MolecularGrid::build`] but with
    /// an explicit per-atom radii vector instead of always calling
    /// [`bragg_slater_bohr`] — lets a test compare two radii tables against
    /// the same radial map / angular rule / Becke partitioning, with
    /// everything else held fixed. Sequential (fine for tiny test systems).
    fn build_grid_with_radii(
        mol: &Molecule,
        level: usize,
        prune: bool,
        radii: &[f64],
    ) -> MolecularGrid {
        let opts = GridOptions {
            level,
            prune,
            ..GridOptions::default()
        };
        MolecularGrid::build_with_radii(mol, &opts, radii).unwrap()
    }

    /// Regression for the He/Ne/Ar/Kr Bragg–Slater radii fix: the table
    /// used to hold Clementi *calculated* radii for the noble gases
    /// (He 0.31, Ne 0.38, Ar 0.71, Kr 1.10 Å) instead of the Slater/PySCF
    /// bonding radii (1.40/1.50/1.80/1.90 Å) used everywhere else in the
    /// table, which changes the Becke `a_ij` size adjustment for any pair
    /// involving a noble-gas centre and a lighter atom.
    ///
    /// Becke weights are a partition of unity regardless of the radii used,
    /// so the near-converged (high grid level) energy is essentially the
    /// same either way — checked first, tightly, on the unpruned level-9
    /// grid. With the former product angular grid NeH⁺ also converged
    /// 6-14× faster at level 3 with the corrected radii; on the standard
    /// Lebedev grid with Bragg-radius pruning the order is reversed
    /// (level-3 error 1.882e-5 Ha with the corrected radii, 1.538e-6 Ha with
    /// Ne = 0.38 Å, both against the unpruned level-9 reference), because
    /// the pruning regions follow the corrected radii while the 0.38 Å
    /// partition happens to put the cell boundary where Ne's grid is full.
    /// Only the absolute level-3 bound is asserted.
    ///
    /// This is NOT claimed to hold for every noble-gas/light-atom geometry:
    /// a broader sweep (He···H2, HeH⁺, ArH⁺, KrH⁺, He···HF, at two basis
    /// sets) found the corrected radii give *worse* level-3 error than the
    /// old table for most He-containing cases and are roughly tied for
    /// Kr — the `a_ij` clamp (±0.5) saturates for both tables once the
    /// radius ratio is large, and which saturated boundary integrates
    /// better at a coarse level is geometry-sensitive. What is robust,
    /// and what this test asserts, is the partition-of-unity invariant:
    /// the converged answer barely moves. See the task report for the
    /// full set of measured numbers.
    #[test]
    fn noble_gas_radii_partition_of_unity_and_neh_convergence() {
        let ang = crate::constants::ANGSTROM_TO_BOHR;
        // NeH+: a real, closed-shell, bonded heteronuclear cation (10
        // electrons) — a natural "rare-gas-containing system" test case.
        let mol = crate::system::Molecule::new(vec![
            Atom {
                z: 10,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.99 * ang),
            },
        ])
        .with_charge(1);

        // New (fixed) radii: whatever `bragg_slater_bohr` returns today.
        let new_radii: Vec<f64> = mol.atoms.iter().map(|a| bragg_slater_bohr(a.z)).collect();
        // Old (buggy) radii: identical except Ne uses the pre-fix Clementi
        // *calculated* value (0.38 Å) instead of the Slater/PySCF bonding
        // radius (1.50 Å); H is untouched by the fix either way.
        let old_radii: Vec<f64> = mol
            .atoms
            .iter()
            .map(|a| {
                if a.z == 10 {
                    0.38 * ang
                } else {
                    bragg_slater_bohr(a.z)
                }
            })
            .collect();

        let set = crate::basis::data::load("sto-3g").unwrap();
        let mb = crate::basis::MolBasis::build(&mol, &set).unwrap();
        let func =
            crate::xc::resolve(&crate::xc::FunctionalSpec::Combined("svwn5".into())).unwrap();
        let sopts = crate::scf::ScfOptions {
            e_tol: 1e-11,
            p_tol: 1e-9,
            max_iter: 200,
            ..crate::scf::ScfOptions::default()
        };
        let run = |level: usize, prune: bool, radii: &[f64]| -> f64 {
            let g = build_grid_with_radii(&mol, level, prune, radii);
            crate::scf::run_scf(&mol, &mb, &sopts, &func, Some(&g))
                .unwrap()
                .energy
        };

        // Near-converged reference: Becke weights are a partition of unity
        // at any radii, so the unpruned level-9 grid with either table
        // should agree tightly; the new-radii value is the reference. The
        // pruned grid is not a converged reference for a foreign partition:
        // its angular degree stays at 11-15 out to 0.5 R_Bragg (1.42 Bohr
        // for Ne), and the Ne/H cell boundary of the 0.38 Angstrom table
        // (about 0.94 Bohr from Ne) lies inside that region. Measured on
        // the pruned level-9 grid: new -126.380299140639, old
        // -126.380319383632 (2.0e-5 apart).
        let e_ref_new = run(9, false, &new_radii);
        let e_ref_old = run(9, false, &old_radii);
        eprintln!(
            "NeH+ unpruned level 9: new={e_ref_new:.12} old={e_ref_old:.12} difference {:.3e}",
            e_ref_new - e_ref_old
        );
        assert!(
            (e_ref_new - e_ref_old).abs() < 1e-6,
            "high-level (converged) energy must not depend strongly on the \
             radii table (partition of unity): new={e_ref_new:.12} old={e_ref_old:.12}"
        );

        let e_low_new = run(3, true, &new_radii);
        let e_low_old = run(3, true, &old_radii);
        let err_new = (e_low_new - e_ref_new).abs();
        let err_old = (e_low_old - e_ref_new).abs();
        eprintln!(
            "NeH+ SVWN5/STO-3G level-3 vs level-9 error: \
             corrected radii = {err_new:.3e} Ha, buggy (Clementi Ne) radii = {err_old:.3e} Ha"
        );

        assert!(
            err_new < 1e-4,
            "level-3 error with corrected radii should already be small: {err_new:.3e}"
        );
    }
}
