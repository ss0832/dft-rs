// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Real-space quadrature over one cell, and the periodic density on it.
//!
//! # Why a molecular grid almost works, and where it stops
//!
//! Exchange-correlation is not an integral of a product of basis functions;
//! it is an integral of a non-linear function of the density, so it has to
//! be done by quadrature. The molecular machinery — Treutler–Ahlrichs
//! radial shells, a product angular grid, Becke partitioning — is reused
//! wholesale here. One thing changes, and it is the thing that makes the
//! result an integral over a CELL rather than over a molecule.
//!
//! Becke's construction works because the atomic weights sum to one
//! everywhere:
//!
//! ```text
//! sum_A w_A(r) = 1   for every r
//! ```
//!
//! Under periodic boundary conditions the sum that must equal one runs over
//! the atoms of the cell AND all their images:
//!
//! ```text
//! sum_{A in cell} sum_T w_A(r - T) = 1
//! ```
//!
//! so each weight's denominator has to include the image atoms. Get that
//! wrong and the weights still look reasonable — they are smooth, positive,
//! and of order one — but they no longer tile space, and every integrated
//! quantity is off by a few percent in a way no symmetry check notices.
//! With it right, `sum_{A in cell} Int w_A f` over all space is exactly
//! `Int_cell f` for any lattice-periodic `f`, and the electron count below
//! is the direct test of that statement.
//!
//! # The density
//!
//! ```text
//! rho(r) = sum_{T1} sum_{T} sum_{mu nu} D_mu-nu(T) chi_mu(r - T1) chi_nu(r - T1 - T)
//! ```
//!
//! — every pair of basis functions from any two cells whose product reaches
//! `r`, weighted by the density block for their separation. That is the
//! same "supercell of images, indexed by cell pair" contraction the
//! periodic Hartree term uses, so the two agree by construction on what the
//! periodic density means.

use crate::basis::MolBasis;
use crate::dual::{DualN, Scalar};
use crate::error::Result;
use crate::grid::{
    angular_rule, bragg_slater_bohr, radial_treutler_ahlrichs, treutler_xi, GridOptions,
};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::integrals::{concat, shifted, ImageBlock};
use crate::periodic::Lattice;
use crate::system::Molecule;
use rayon::prelude::*;

/// Quadrature points and weights covering one cell.
pub struct PeriodicGrid {
    pub points: Vec<Vec3>,
    /// Radial x angular x periodic Becke partition weight.
    pub weights: Vec<f64>,
    /// Radial x angular weight before the periodic Becke factor.  Needed
    /// by the analytic weight response in periodic XC forces.
    pub raw_weights: Vec<f64>,
    /// Owning atom, indexed into the cell's own atom list.
    pub atom_of: Vec<usize>,
}

/// Proper orthonormal frame tied to the first two bulk lattice vectors.
/// Its Cartesian strains rotate the angular offsets without stretching them.
#[derive(Clone, Debug)]
pub(crate) struct AngularGridFrame {
    axes: [Vec3; 3],
    axis_strains: [[[Vec3; 3]; 3]; 3],
}

impl AngularGridFrame {
    pub(crate) fn bulk(geometry: &super::PeriodicGeometry) -> Option<Self> {
        if geometry.dimension() != 3 {
            return None;
        }
        let [a, b, _] = geometry.lattice().a;
        let normal = a.cross(b);
        let a_norm = a.norm();
        let normal_norm = normal.norm();
        let x = a / a_norm;
        let z = normal / normal_norm;
        let y = z.cross(x);
        let axis_strains = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let axis = Vec3::new(f64::from(i == 0), f64::from(i == 1), f64::from(i == 2));
                let da = axis * a.get(j);
                let db = axis * b.get(j);
                let dn = da.cross(b) + a.cross(db);
                let dx = (da - x * x.dot(da)) / a_norm;
                let dz = (dn - z * z.dot(dn)) / normal_norm;
                [dx, dz.cross(x) + z.cross(dx), dz]
            })
        });
        Some(Self {
            axes: [x, y, z],
            axis_strains,
        })
    }

    fn to_cartesian(&self, local: Vec3) -> Vec3 {
        self.axes[0] * local.x + self.axes[1] * local.y + self.axes[2] * local.z
    }

    /// Extra Cartesian point motion beyond the affine motion of its owner.
    pub(crate) fn offset_strain(&self, offset: Vec3) -> [[Vec3; 3]; 3] {
        let local = self.axes.map(|axis| axis.dot(offset));
        self.axis_strains
            .map(|row| row.map(|ds| ds[0] * local[0] + ds[1] * local[1] + ds[2] * local[2]))
    }
}

/// Physical cutoff used for the local periodic Becke denominator.
///
/// The length per atom, `(V/N)^(1/3)`, is unchanged when a primitive cell is
/// replaced by an integer supercell. A fixed number of lattice-index shells
/// is not: doubling one lattice vector doubles the physical reach of the
/// same shell. That made equivalent cells use different denominators and
/// broke size extensivity.
pub(crate) fn local_cutoff(mol: &Molecule, lattice: &Lattice, shell: i64) -> f64 {
    let volume_per_atom = lattice.volume().abs() / mol.len().max(1) as f64;
    (shell.max(1) as f64 + 1.5) * volume_per_atom.cbrt()
}

/// Smooth switching of the local Becke neighbourhood.
///
/// A hard cut of the image list at `cut` makes the partition of unity, and
/// with it the XC energy, jump whenever an image centre crosses the sphere as
/// atoms or the cell move. Every centre `c` therefore carries the factor
///
/// ```text
/// g(r) = 1 - s(t),  t = (r - on)/(cut - on),  s(t) = e^(-1/t) / (e^(-1/t) + e^(-1/(1-t)))
/// ```
///
/// with `g = 1` for `r <= on`, `g = 0` for `r >= cut` and a C-infinity step
/// in between, `r = |p - c|`. For the periodic grid `cut` is the physical
/// radius of [`local_cutoff`] (so the partition stays size extensive) and the
/// switching zone is one length per atom wide, `on = cut - (V/N)^(1/3)`.
/// Both radii are proportional to the length per atom `L = (V/N)^(1/3)`, so
/// `g` is a function of `r/L` alone and a strain that changes the volume moves
/// the switching zone smoothly: `g(r; L) = G(r/L - (shell + 1/2))`.
/// (`on >= cut` is the old hard sphere, `g = 1` inside and `0` outside; it
/// is kept for the smoothness comparison tests.)
#[derive(Clone, Copy, Debug)]
pub(crate) struct Switch {
    /// Distance up to which a centre takes part with full weight.
    pub(crate) on: f64,
    /// Distance from which a centre is left out.
    pub(crate) cut: f64,
}

impl Switch {
    pub(crate) fn periodic(mol: &Molecule, lattice: &Lattice, shell: i64) -> Self {
        let length = (lattice.volume().abs() / mol.len().max(1) as f64).cbrt();
        let cut = local_cutoff(mol, lattice, shell);
        Self {
            on: cut - length,
            cut,
        }
    }

    /// Dimensional grids have no volume per atom; the zone is the outer
    /// 30 % of the cutoff.
    pub(crate) fn dimensional(cut: f64) -> Self {
        Self { on: 0.7 * cut, cut }
    }

    /// `g(r)` on any scalar (jets included).
    pub(crate) fn value<S: Scalar>(&self, r: S) -> S {
        let one = S::from_f64(1.0);
        if self.cut <= self.on {
            return if r.value() < self.cut {
                one
            } else {
                S::from_f64(0.0)
            };
        }
        let t = (r - S::from_f64(self.on)) / S::from_f64(self.cut - self.on);
        let tv = t.value();
        if tv <= 0.0 {
            return one;
        }
        if tv >= 1.0 {
            return S::from_f64(0.0);
        }
        // g = 1 / (1 + exp(x)), x = 1/(1-t) - 1/t. Beyond |x| = 60 the value
        // is below 1e-26 from 0 or 1 (a jump far under roundoff), and the
        // nested jets stay free of overflow.
        let x = (one - t).recip() - t.recip();
        if x.value() > 60.0 {
            return S::from_f64(0.0);
        }
        if x.value() < -60.0 {
            return one;
        }
        (x.exp() + one).recip()
    }

    /// `g(r)` and `dg/dr` in closed form.
    pub(crate) fn value_slope(&self, r: f64) -> (f64, f64) {
        if self.cut <= self.on {
            return (if r < self.cut { 1.0 } else { 0.0 }, 0.0);
        }
        let width = self.cut - self.on;
        let t = (r - self.on) / width;
        if t <= 0.0 {
            return (1.0, 0.0);
        }
        if t >= 1.0 {
            return (0.0, 0.0);
        }
        let x = 1.0 / (1.0 - t) - 1.0 / t;
        if x > 60.0 {
            return (0.0, 0.0);
        }
        if x < -60.0 {
            return (1.0, 0.0);
        }
        let g = 1.0 / (1.0 + x.exp());
        let dx = 1.0 / ((1.0 - t) * (1.0 - t)) + 1.0 / (t * t);
        (g, -g * (1.0 - g) * dx / width)
    }
}

/// Becke's three-fold iterated polynomial `s(nu) = (1 - p(p(p(nu))))/2` on a
/// scalar.
fn becke_s<S: Scalar>(nu: S) -> S {
    let mut f = nu;
    for _ in 0..3 {
        f = S::from_f64(1.5) * f - S::from_f64(0.5) * f * f * f;
    }
    S::from_f64(0.5) * (S::from_f64(1.0) - f)
}

/// `s(nu)` and `ds/dnu` in closed form.
fn becke_s_slope(nu: f64) -> (f64, f64) {
    let mut f = nu;
    let mut chain = 1.0;
    for _ in 0..3 {
        chain *= 1.5 - 1.5 * f * f;
        f = 1.5 * f - 0.5 * f * f * f;
    }
    (0.5 * (1.0 - f), -0.5 * chain)
}

/// Becke size-adjustment parameter of a pair of radii.
pub(crate) fn size_adjustment(radius_b: f64, radius_c: f64) -> f64 {
    let chi = radius_b / radius_c;
    let u = (chi - 1.0) / (chi + 1.0);
    (u / (u * u - 1.0)).clamp(-0.5, 0.5)
}

/// The factor of competitor `c` in the cell function of `b`,
/// `q = 1 - g_c (1 - s(nu_bc)) = s + (1 - g_c)(1 - s)` (the second form has no
/// cancellation for `g_c = 1`, where a centre in the shadow of another has
/// `s` far below `1e-16`), from the distances `db = |p - b|`,
/// `dc = |p - c|`, `r = |b - c|`, the size adjustment `a` and `gc = g(dc)`.
///
/// A competitor that is switched out (`g_c = 0`) leaves the cell function of
/// every other centre untouched, so entering or leaving the list is
/// continuous in all orders; with `g_c = 1` it is Becke's factor.
pub(crate) fn pair_factor<S: Scalar>(db: S, dc: S, r: S, a: f64, gc: S) -> S {
    let one = S::from_f64(1.0);
    let mu = (db - dc) / r;
    let nu = mu + S::from_f64(a) * (one - mu * mu);
    let s = becke_s(nu);
    s + (one - gc) * (one - s)
}

/// Distances of a point to the local centres and their switching values.
struct Neighbourhood {
    d: Vec<f64>,
    g: Vec<f64>,
    /// `dg/dr` at the distances.
    dg: Vec<f64>,
}

impl Neighbourhood {
    fn new(centers: &[(Vec3, f64, usize)], p: Vec3, sw: &Switch) -> Self {
        let d: Vec<f64> = centers.iter().map(|(c, _, _)| (p - *c).norm()).collect();
        let (g, dg) = d.iter().map(|r| sw.value_slope(*r)).unzip();
        Self { d, g, dg }
    }
}

/// Switched Becke cell function `P_b = g_b prod_c q_bc`, with the exact-zero
/// exit of the partition sum.
fn switched_cell(centers: &[(Vec3, f64, usize)], nb: &Neighbourhood, b: usize) -> f64 {
    if nb.g[b] == 0.0 {
        return 0.0;
    }
    let mut cell = nb.g[b];
    for (c, other) in centers.iter().enumerate() {
        if c == b || nb.g[c] == 0.0 {
            continue;
        }
        let rbc = (centers[b].0 - other.0).norm();
        if rbc < 1e-12 {
            continue;
        }
        let a = size_adjustment(centers[b].1, other.1);
        cell *= pair_factor(nb.d[b], nb.d[c], rbc, a, nb.g[c]);
        if cell == 0.0 {
            break;
        }
    }
    cell
}

/// The image atoms that matter at one point: every physical image within a
/// spherical cutoff of `p`.
///
/// The sphere is centred on the grid point, not on the cell origin, and its
/// radius is a physical length. Therefore a primitive cell and an equivalent
/// supercell select the same physical centres. Every target at the point is
/// evaluated against this shared set, preserving the partition of unity.
fn local_images(
    mol: &Molecule,
    lattice: &Lattice,
    p: Vec3,
    cutoff: f64,
) -> Vec<(Vec3, f64, usize)> {
    let reciprocal = lattice.reciprocal();
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut nmax = [0i64; 3];
    for i in 0..3 {
        let perpendicular_width = two_pi / reciprocal[i].norm();
        nmax[i] = (cutoff / perpendicular_width).ceil() as i64 + 1;
    }

    let mut out = Vec::new();
    for (b, atom) in mol.atoms.iter().enumerate() {
        let f = lattice.to_fractional(p - atom.position);
        let base = [
            f[0].round() as i64,
            f[1].round() as i64,
            f[2].round() as i64,
        ];
        let r = bragg_slater_bohr(atom.z);
        for i in -nmax[0]..=nmax[0] {
            for j in -nmax[1]..=nmax[1] {
                for k in -nmax[2]..=nmax[2] {
                    let t = lattice.translation([base[0] + i, base[1] + j, base[2] + k]);
                    let center = atom.position + t;
                    if (center - p).norm() <= cutoff {
                        out.push((center, r, b));
                    }
                }
            }
        }
    }
    // `local_becke_weight` usually proves that a distant candidate has zero
    // weight as soon as it is compared with the nearest centre. Keeping the
    // list in distance order makes that exact-zero early exit effective;
    // it changes neither the selected physical set nor the Becke formula.
    out.sort_by(|a, b| {
        (a.0 - p)
            .norm2()
            .total_cmp(&(b.0 - p).norm2())
            .then_with(|| a.0.x.total_cmp(&b.0.x))
            .then_with(|| a.0.y.total_cmp(&b.0.y))
            .then_with(|| a.0.z.total_cmp(&b.0.z))
            .then_with(|| a.2.cmp(&b.2))
    });
    out
}

/// The local image atoms of [`local_images`] for a caller in the crate.
pub(crate) fn local_images_of(
    mol: &Molecule,
    lattice: &Lattice,
    p: Vec3,
    cutoff: f64,
) -> Vec<(Vec3, f64, usize)> {
    local_images(mol, lattice, p, cutoff)
}

fn dimensional_local_images(
    mol: &Molecule,
    geometry: &super::PeriodicGeometry,
    p: Vec3,
    cutoff: f64,
) -> Result<Vec<(Vec3, f64, usize)>> {
    let mut out = Vec::new();
    for (atom_index, atom) in mol.atoms.iter().enumerate() {
        for index in geometry.nearby_translation_indices(atom.position - p, cutoff)? {
            out.push((
                atom.position + geometry.lattice().translation(index),
                bragg_slater_bohr(atom.z),
                atom_index,
            ));
        }
    }
    out.sort_by(|a, b| {
        (a.0 - p)
            .norm2()
            .total_cmp(&(b.0 - p).norm2())
            .then_with(|| a.0.x.total_cmp(&b.0.x))
            .then_with(|| a.0.y.total_cmp(&b.0.y))
            .then_with(|| a.0.z.total_cmp(&b.0.z))
            .then_with(|| a.2.cmp(&b.2))
    });
    Ok(out)
}

/// Switched Becke weight `P_target / sum_b P_b` of one image atom against the
/// local set; the sum runs over every centre of the list (those switched out
/// contribute zero), so the weights of the list sum to one.
fn local_becke_weight(centers: &[(Vec3, f64, usize)], target: usize, p: Vec3, sw: &Switch) -> f64 {
    let nb = Neighbourhood::new(centers, p, sw);
    let mut total = 0.0;
    let mut mine = 0.0;
    for b in 0..centers.len() {
        let cell = switched_cell(centers, &nb, b);
        if b == target {
            mine = cell;
        }
        total += cell;
    }
    if total <= 0.0 {
        0.0
    } else {
        mine / total
    }
}

impl PeriodicGrid {
    /// Build the grid for one cell.
    ///
    /// `shell` controls the physical radius of the local Becke denominator.
    /// The radius is scaled by volume per atom, so an equivalent supercell
    /// uses exactly the same physical neighbourhood.
    ///
    /// Points beyond the same physical radius are dropped. This is a
    /// controlled real-space truncation: the home atom's Becke weight tends
    /// rapidly to zero there, and increasing `shell` converges the result.
    pub fn build(
        mol: &Molecule,
        lattice: &Lattice,
        opts: &GridOptions,
        shell: i64,
    ) -> Result<Self> {
        let sw = Switch::periodic(mol, lattice, shell);
        Self::build_with_images(
            mol,
            opts,
            sw,
            true,
            None,
            |point| Ok(local_images(mol, lattice, point, sw.cut)),
            |_, z, count| radial_treutler_ahlrichs(count, treutler_xi(z)),
        )
    }

    /// Atom-centred quadrature over one periodic cell and all open directions.
    /// The physical cutoff in Bohr bounds radial quadrature and the local image
    /// denominator. Increase it independently of the grid level to resolve tails.
    pub fn build_dimensional(
        mol: &Molecule,
        geometry: &super::PeriodicGeometry,
        opts: &GridOptions,
        cutoff: f64,
    ) -> Result<Self> {
        Self::build_dimensional_oriented(mol, geometry, opts, cutoff, None)
    }

    pub(crate) fn build_dimensional_oriented(
        mol: &Molecule,
        geometry: &super::PeriodicGeometry,
        opts: &GridOptions,
        cutoff: f64,
        frame: Option<&AngularGridFrame>,
    ) -> Result<Self> {
        Self::build_dimensional_with_radial(
            mol,
            geometry,
            opts,
            cutoff,
            true,
            frame,
            |_, z, count| radial_treutler_ahlrichs(count, treutler_xi(z)),
        )
    }

    /// Value-only grid with the same image partition and a caller-supplied
    /// radial rule. The caller supplies weights including the radial Jacobian.
    pub(crate) fn dimensional_values_with_radial(
        mol: &Molecule,
        geometry: &super::PeriodicGeometry,
        opts: &GridOptions,
        cutoff: f64,
        radial: impl Fn(usize, u8, usize) -> (Vec<f64>, Vec<f64>) + Sync,
    ) -> Result<(Vec<Vec3>, Vec<f64>)> {
        let grid =
            Self::build_dimensional_with_radial(mol, geometry, opts, cutoff, false, None, radial)?;
        Ok((grid.points, grid.weights))
    }

    fn build_dimensional_with_radial(
        mol: &Molecule,
        geometry: &super::PeriodicGeometry,
        opts: &GridOptions,
        cutoff: f64,
        retain_derivative_data: bool,
        frame: Option<&AngularGridFrame>,
        radial: impl Fn(usize, u8, usize) -> (Vec<f64>, Vec<f64>) + Sync,
    ) -> Result<Self> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(crate::DftError::InvalidInput(
                "dimensional grid cutoff must be finite and positive".into(),
            ));
        }
        Self::build_with_images(
            mol,
            opts,
            Switch::dimensional(cutoff),
            retain_derivative_data,
            frame,
            |point| dimensional_local_images(mol, geometry, point, cutoff),
            radial,
        )
    }

    fn build_with_images<F>(
        mol: &Molecule,
        opts: &GridOptions,
        sw: Switch,
        retain_derivative_data: bool,
        frame: Option<&AngularGridFrame>,
        images: F,
        radial: impl Fn(usize, u8, usize) -> (Vec<f64>, Vec<f64>) + Sync,
    ) -> Result<Self>
    where
        F: Fn(Vec3) -> Result<Vec<(Vec3, f64, usize)>> + Sync,
    {
        opts.validate()?;
        let n_cell = mol.len();
        // The switching function is exactly zero from `sw.cut` on, so the
        // radial points beyond it carry nothing.
        let keep = sw.cut;

        let per_atom: Vec<(Vec<Vec3>, Vec<f64>, Vec<f64>, usize)> = (0..n_cell)
            .into_par_iter()
            .map(|ia| -> Result<_> {
                let z = mol.atoms[ia].z;
                let (n_rad, degree) = opts.resolve(ia, z);
                let (r_pts, r_wts) = radial(ia, z, n_rad);
                let radius = bragg_slater_bohr(z);
                let full_angular = angular_rule(opts.scheme, degree);
                let mut pts = Vec::new();
                let mut wts = Vec::new();
                let mut raws = Vec::new();
                for (r, wr) in r_pts.iter().zip(&r_wts) {
                    if *r > keep {
                        continue;
                    }
                    let deg = crate::grid::pruned_degree(z, *r, radius, degree, opts.prune);
                    let angular = if deg == degree {
                        std::borrow::Cow::Borrowed(full_angular.as_ref())
                    } else {
                        angular_rule(opts.scheme, deg)
                    };
                    for &(dir, wa) in angular.iter() {
                        let dir = frame.map_or(dir, |frame| frame.to_cartesian(dir));
                        let p = mol.atoms[ia].position + dir * *r;
                        let centers = images(p)?;
                        // The home image of this atom is the numerator. It
                        // is in the set exactly when the point still belongs
                        // to this atom's region; when it is not, the weight
                        // is zero and the point carries nothing.
                        let Some(target) = centers.iter().position(|(c, _, bidx)| {
                            *bidx == ia && (*c - mol.atoms[ia].position).norm() < 1e-10
                        }) else {
                            continue;
                        };
                        let raw = wr * wa;
                        let w = raw * local_becke_weight(&centers, target, p, &sw);
                        if w.abs() >= opts.weight_cutoff {
                            pts.push(p);
                            wts.push(w);
                            if retain_derivative_data {
                                raws.push(raw);
                            }
                        }
                    }
                }
                Ok((pts, wts, raws, ia))
            })
            .collect::<Result<_>>()?;

        let len = per_atom.iter().map(|(pts, ..)| pts.len()).sum();
        let derivative_len = if retain_derivative_data { len } else { 0 };
        let mut points = Vec::with_capacity(len);
        let mut weights = Vec::with_capacity(len);
        let mut raw_weights = Vec::with_capacity(derivative_len);
        let mut atom_of = Vec::with_capacity(derivative_len);
        for (pts, wts, raws, ia) in per_atom {
            if retain_derivative_data {
                atom_of.extend(std::iter::repeat_n(ia, pts.len()));
            }
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

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Periodic Becke weight and its fixed-point derivatives with respect to
/// the atoms in the primitive cell.  Derivatives of all images belonging to
/// the same atom are folded together.
pub(crate) fn periodic_weight_gradient(
    mol: &Molecule,
    lattice: &Lattice,
    point: Vec3,
    owner: usize,
    shell: i64,
) -> (f64, Vec<Vec3>) {
    let sw = Switch::periodic(mol, lattice, shell);
    let centers = local_images(mol, lattice, point, sw.cut);
    folded_weight_gradient(mol, centers, point, owner, &sw)
}

/// Fixed-Cartesian-point Becke derivatives folded over physical atom images.
/// The grid point derivative is minus the sum of the returned atomic vectors.
/// A point moving with its owner includes that point derivative on the owner.
pub fn dimensional_weight_gradient(
    mol: &Molecule,
    geometry: &super::PeriodicGeometry,
    point: Vec3,
    owner: usize,
    cutoff: f64,
) -> Result<(f64, Vec<Vec3>)> {
    if owner >= mol.len() || !point.norm().is_finite() || !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(crate::DftError::InvalidInput(
            "dimensional Becke point, owner and cutoff must be valid".into(),
        ));
    }
    let centers = dimensional_local_images(mol, geometry, point, cutoff)?;
    Ok(folded_weight_gradient(
        mol,
        centers,
        point,
        owner,
        &Switch::dimensional(cutoff),
    ))
}

/// Switched Becke weight at `point` and its fixed-point derivatives with
/// respect to every image centre, with the image positions; `None` when the
/// owner is not among the local images.
///
/// With `P_b = g_b prod_c q_bc` and `q = 1 - g_c (1 - s(nu))`, every
/// logarithmic derivative `d ln P_b / d C_k` is a sum of pair terms in the
/// three distances `(|p - C_b|, |p - C_c|, |C_b - C_c|)`, so the partial sums
/// of `d Z / d C_k` and `d P_t / d C_k` are accumulated cell by cell.
/// Relative size below which a switched Becke cell enters the normalization
/// of the periodic weight but not its derivatives: such a cell times the
/// derivatives of its logarithm (far below 1e14) is below the double
/// precision resolution of the result.
const NEGLIGIBLE_CELL: f64 = 1e-30;

fn image_weight_gradient(
    mol: &Molecule,
    centers: &[(Vec3, f64, usize)],
    point: Vec3,
    owner: usize,
    sw: &Switch,
) -> Option<(f64, Vec<Vec3>, f64)> {
    let target = centers.iter().position(|(center, _, atom)| {
        *atom == owner && (*center - mol.atoms[owner].position).norm() < 1e-10
    })?;
    let n = centers.len();
    let nb = Neighbourhood::new(centers, point, sw);
    let cells: Vec<f64> = (0..n).map(|b| switched_cell(centers, &nb, b)).collect();
    let total: f64 = cells.iter().sum();
    if total <= 0.0 {
        return Some((0.0, vec![Vec3::zero(); n], 0.0));
    }
    let mut dz = vec![Vec3::zero(); n];
    let mut dp = vec![Vec3::zero(); n];
    // d ln P_b / d ln L at fixed distances: `dg/d ln L = -r g'(r)` of every
    // switching function in the cell function.
    let (mut dz_scale, mut dp_scale) = (0.0, 0.0);
    // A cell below `NEGLIGIBLE_CELL` of the largest one stays in the
    // normalization, but its derivatives cannot change the result in double
    // precision and are not formed.
    let largest = cells.iter().cloned().fold(0.0_f64, f64::max);
    for b in 0..n {
        if cells[b] == 0.0 || (b != target && cells[b] < NEGLIGIBLE_CELL * largest) {
            continue;
        }
        let mut scale_log = -nb.d[b] * nb.dg[b] / nb.g[b];
        let eb = (centers[b].0 - point) / nb.d[b].max(1e-300);
        // d ln g_b / d C_b.
        let mut own = eb * (nb.dg[b] / nb.g[b]);
        for c in 0..n {
            if c == b || nb.g[c] == 0.0 {
                continue;
            }
            let rvec = centers[b].0 - centers[c].0;
            let r = rvec.norm();
            if r < 1e-12 {
                continue;
            }
            let a = size_adjustment(centers[b].1, centers[c].1);
            let mu = (nb.d[b] - nb.d[c]) / r;
            let nu = mu + a * (1.0 - mu * mu);
            let (s, ds) = becke_s_slope(nu);
            let q = s + (1.0 - nb.g[c]) * (1.0 - s);
            if q == 1.0 {
                continue; // a flat factor of a distant image
            }
            let dnu = ds * (1.0 - 2.0 * a * mu);
            let l_db = nb.g[c] * dnu / (r * q);
            let l_dc = (-nb.g[c] * dnu / r + nb.dg[c] * (s - 1.0)) / q;
            let l_r = -nb.g[c] * dnu * mu / (r * q);
            let ec = (centers[c].0 - point) / nb.d[c].max(1e-300);
            let er = rvec / r;
            own += eb * l_db + er * l_r;
            scale_log += (1.0 - s) * nb.d[c] * nb.dg[c] / q;
            let to_c = ec * l_dc - er * l_r;
            dz[c] += to_c * cells[b];
            if b == target {
                dp[c] += to_c * cells[b];
            }
        }
        dz[b] += own * cells[b];
        dz_scale += scale_log * cells[b];
        if b == target {
            dp[b] += own * cells[b];
            dp_scale = scale_log * cells[b];
        }
    }
    let weight = cells[target] / total;
    let gradients = (0..n)
        .map(|k| (dp[k] * total - dz[k] * cells[target]) / (total * total))
        .collect();
    let d_ln_length = (dp_scale * total - dz_scale * cells[target]) / (total * total);
    Some((weight, gradients, d_ln_length))
}

fn folded_weight_gradient(
    mol: &Molecule,
    centers: Vec<(Vec3, f64, usize)>,
    point: Vec3,
    owner: usize,
    sw: &Switch,
) -> (f64, Vec<Vec3>) {
    let Some((weight, image_gradients, _)) = image_weight_gradient(mol, &centers, point, owner, sw)
    else {
        return (0.0, vec![Vec3::zero(); mol.len()]);
    };
    let mut folded = vec![Vec3::zero(); mol.len()];
    for (gradient, (_, _, atom)) in image_gradients.into_iter().zip(centers) {
        folded[atom] += gradient;
    }
    (weight, folded)
}

/// Strain virial of the periodic Becke weight of a point that rides its
/// owner atom: `sum_C (dP/dC)_a (C - R_owner)_b` over every image centre `C`
/// at fixed point, which together with the owner-carried point motion is
/// the full centre virial of the partition weight, plus the dependence of the
/// switching radii on the volume, `delta_ab/3 dw/d ln L`.
pub(crate) fn periodic_weight_virial(
    mol: &Molecule,
    lattice: &Lattice,
    point: Vec3,
    owner: usize,
    shell: i64,
) -> [[f64; 3]; 3] {
    let sw = Switch::periodic(mol, lattice, shell);
    let centers = local_images(mol, lattice, point, sw.cut);
    let mut virial = [[0.0; 3]; 3];
    let Some((_, image_gradients, d_ln_length)) =
        image_weight_gradient(mol, &centers, point, owner, &sw)
    else {
        return virial;
    };
    // The switching radii scale with L = (V/N)^(1/3), `d ln L / d eps_ab = delta_ab / 3`.
    for a in 0..3 {
        virial[a][a] += d_ln_length / 3.0;
    }
    let origin = mol.atoms[owner].position;
    for (gradient, (position, _, _)) in image_gradients.iter().zip(&centers) {
        let relative = *position - origin;
        for a in 0..3 {
            for b in 0..3 {
                virial[a][b] += gradient.get(a) * relative.get(b);
            }
        }
    }
    virial
}

/// Periodic Becke weight of a point riding its owner atom and its total
/// Gamma-point derivatives: every image of an atom moves with it, and the
/// point moves with its owner.
#[derive(Clone, Debug)]
pub(crate) struct WeightDerivatives {
    #[cfg(test)]
    pub(crate) weight: f64,
    /// `D_A w`, `[3 * atom + axis]`.
    pub(crate) first: Vec<f64>,
    /// `D_A D_B w`, `3N x 3N` row-major.
    pub(crate) second: Vec<f64>,
}

/// First and second derivatives of `ln q` with respect to `(|p - C_b|,
/// |p - C_c|, |C_b - C_c|)` for the competitor factor
/// `q = 1 - g_c (1 - s(nu))`, from nested forward-mode jets.
fn pair_log_jets(db: f64, dc: f64, r: f64, a: f64, sw: &Switch) -> ([f64; 3], [[f64; 3]; 3]) {
    type Inner = DualN<f64, 3>;
    type Outer = DualN<Inner, 3>;
    let vars = [db, dc, r];
    let x: [Outer; 3] = std::array::from_fn(|k| Outer::variable(Inner::variable(vars[k], k), k));
    let gc = sw.value(x[1]);
    let l = pair_factor(x[0], x[1], x[2], a, gc).ln();
    let first = std::array::from_fn(|k| l.re.im[k]);
    let second = std::array::from_fn(|k| std::array::from_fn(|m| l.im[k].im[m]));
    (first, second)
}

/// First and second derivatives of `ln g(d)` with respect to the distance.
fn own_log_jets(d: f64, sw: &Switch) -> (f64, f64) {
    type Inner = DualN<f64, 1>;
    type Outer = DualN<Inner, 1>;
    let l = sw.value(Outer::variable(Inner::variable(d, 0), 0)).ln();
    (l.re.im[0], l.im[0].im[0])
}

/// Analytic total first and second derivatives of the periodic Becke weight
/// at `point`, owned by `owner`, over the same local image set as the
/// quadrature. Each switched cell function `P_b = g_b prod_c q_bc` is
/// differentiated in the nine coordinates of its point and two centres (the
/// own factor in six), and the slots are folded onto parent atoms (the point
/// onto its owner).
pub(crate) fn periodic_weight_hessian(
    mol: &Molecule,
    lattice: &Lattice,
    point: Vec3,
    owner: usize,
    shell: i64,
) -> WeightDerivatives {
    let natoms = mol.len();
    let dim = 3 * natoms;
    let sw = Switch::periodic(mol, lattice, shell);
    let centers = local_images(mol, lattice, point, sw.cut);
    let zero = WeightDerivatives {
        #[cfg(test)]
        weight: 0.0,
        first: vec![0.0; dim],
        second: vec![0.0; dim * dim],
    };
    let Some(target) = centers.iter().position(|(center, _, atom)| {
        *atom == owner && (*center - mol.atoms[owner].position).norm() < 1e-10
    }) else {
        return zero;
    };
    let n = centers.len();
    let nb = Neighbourhood::new(&centers, point, &sw);
    // Cell values and folded derivatives of P_b. The values come first:
    // a cell below `NEGLIGIBLE_CELL` times the largest one enters the
    // normalization Z, but its derivatives (P_b times derivatives of ln P_b,
    // which stay far below 1e14) cannot change the result in double
    // precision and are not formed. The cell of the owner always is.
    let mut cells: Vec<f64> = (0..n).map(|b| switched_cell(&centers, &nb, b)).collect();
    let largest = cells.iter().cloned().fold(0.0_f64, f64::max);
    let mut d_cells: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut dd_cells: Vec<Vec<f64>> = vec![Vec::new(); n];
    for b in 0..n {
        // Exact-zero early exit of the quadrature; only living cells carry
        // derivatives.
        let value = cells[b];
        if value == 0.0 || (b != target && value < NEGLIGIBLE_CELL * largest) {
            continue;
        }
        let mut dlog = vec![0.0; dim];
        let mut ddlog = vec![0.0; dim * dim];
        let mut dead = false;
        // Own switching factor g_b: slots point (0..3) and C_b (3..6).
        {
            let cb = centers[b].0;
            let ub_vec = point - cb;
            let db = nb.d[b];
            let ub = [ub_vec.x / db, ub_vec.y / db, ub_vec.z / db];
            let (l1, l2) = own_log_jets(db, &sw);
            if !l1.is_finite() || !l2.is_finite() {
                cells[b] = 0.0;
                continue;
            }
            let fold = |slot: usize| -> usize {
                match slot / 3 {
                    0 => 3 * owner + slot % 3,
                    _ => 3 * centers[b].2 + slot % 3,
                }
            };
            let mut g = [0.0; 6];
            for k in 0..3 {
                g[k] = ub[k];
                g[3 + k] = -ub[k];
            }
            for x in 0..3 {
                for y in 0..3 {
                    let delta = if x == y { 1.0 } else { 0.0 };
                    let kb = (delta - ub[x] * ub[y]) / db;
                    for (i, si) in [(0usize, 1.0), (3, -1.0)] {
                        for (j, sj) in [(0usize, 1.0), (3, -1.0)] {
                            ddlog[fold(i + x) * dim + fold(j + y)] += l1 * si * sj * kb;
                        }
                    }
                }
            }
            for i in 0..6 {
                dlog[fold(i)] += l1 * g[i];
                for j in 0..6 {
                    ddlog[fold(i) * dim + fold(j)] += l2 * g[i] * g[j];
                }
            }
        }
        for c in 0..n {
            if c == b || nb.g[c] == 0.0 {
                continue;
            }
            let cb = centers[b].0;
            let cc = centers[c].0;
            let rvec = cb - cc;
            let r = rvec.norm();
            if r < 1e-12 {
                continue;
            }
            let (db, dc) = (nb.d[b], nb.d[c]);
            let a = size_adjustment(centers[b].1, centers[c].1);
            if pair_factor(db, dc, r, a, nb.g[c]) == 1.0 {
                continue; // a flat factor of a distant image
            }
            let (l1, l2) = pair_log_jets(db, dc, r, a, &sw);
            if l1.iter().chain(l2.iter().flatten()).any(|v| !v.is_finite()) {
                // The factor underflows: the cell is empty to roundoff.
                dead = true;
                break;
            }
            let flat =
                l1.iter().all(|v| v.abs() < 1e-15) && l2.iter().flatten().all(|v| v.abs() < 1e-13);
            if flat {
                continue;
            }
            let ub_vec = point - cb;
            let uc_vec = point - cc;
            // Nine coordinates: point (0..3), C_b (3..6), C_c (6..9).
            let ub = [ub_vec.x / db, ub_vec.y / db, ub_vec.z / db];
            let uc = [uc_vec.x / dc, uc_vec.y / dc, uc_vec.z / dc];
            let e = [rvec.x / r, rvec.y / r, rvec.z / r];
            // Gradients of the three distances in the nine coordinates.
            let mut gk = [[0.0; 9]; 3];
            for k in 0..3 {
                gk[0][k] = ub[k];
                gk[0][3 + k] = -ub[k];
                gk[1][k] = uc[k];
                gk[1][6 + k] = -uc[k];
                gk[2][3 + k] = e[k];
                gk[2][6 + k] = -e[k];
            }
            // local = sum_k l1_k Hess(d_k) + sum_km l2_km grad(d_k) grad(d_m)^T.
            // Each distance Hessian has four 3x3 blocks +-K on the slot pair
            // of its two ends; they are added directly instead of building
            // the 3x9x9 tensor.
            let mut local = [[0.0; 9]; 9];
            for x in 0..3 {
                for y in 0..3 {
                    let delta = if x == y { 1.0 } else { 0.0 };
                    let blocks = [
                        (l1[0] * (delta - ub[x] * ub[y]) / db, 0usize, 3usize),
                        (l1[1] * (delta - uc[x] * uc[y]) / dc, 0, 6),
                        (l1[2] * (delta - e[x] * e[y]) / r, 3, 6),
                    ];
                    for (value, first_slot, second_slot) in blocks {
                        local[first_slot + x][first_slot + y] += value;
                        local[second_slot + x][second_slot + y] += value;
                        local[first_slot + x][second_slot + y] -= value;
                        local[second_slot + x][first_slot + y] -= value;
                    }
                }
            }
            // t_kj = sum_m l2_km grad_j(d_m).
            let mut tk = [[0.0; 9]; 3];
            for k in 0..3 {
                for j in 0..9 {
                    tk[k][j] = l2[k][0] * gk[0][j] + l2[k][1] * gk[1][j] + l2[k][2] * gk[2][j];
                }
            }
            for i in 0..9 {
                for j in 0..9 {
                    local[i][j] += gk[0][i] * tk[0][j] + gk[1][i] * tk[1][j] + gk[2][i] * tk[2][j];
                }
            }
            let slots: [usize; 9] = std::array::from_fn(|slot| match slot / 3 {
                0 => 3 * owner + slot % 3,
                1 => 3 * centers[b].2 + slot % 3,
                _ => 3 * centers[c].2 + slot % 3,
            });
            for i in 0..9 {
                dlog[slots[i]] += l1[0] * gk[0][i] + l1[1] * gk[1][i] + l1[2] * gk[2][i];
                let row = slots[i] * dim;
                for j in 0..9 {
                    ddlog[row + slots[j]] += local[i][j];
                }
            }
        }
        if dead {
            cells[b] = 0.0;
            continue;
        }
        let mut first = vec![0.0; dim];
        let mut second = vec![0.0; dim * dim];
        for i in 0..dim {
            first[i] = value * dlog[i];
            for j in 0..dim {
                second[i * dim + j] = value * (ddlog[i * dim + j] + dlog[i] * dlog[j]);
            }
        }
        d_cells[b] = first;
        dd_cells[b] = second;
    }
    let z: f64 = cells.iter().sum();
    if z <= 0.0 || cells[target] == 0.0 {
        return zero;
    }
    let mut dz = vec![0.0; dim];
    let mut ddz = vec![0.0; dim * dim];
    for b in 0..n {
        if d_cells[b].is_empty() {
            continue;
        }
        for i in 0..dim {
            dz[i] += d_cells[b][i];
        }
        for k in 0..dim * dim {
            ddz[k] += dd_cells[b][k];
        }
    }
    let pt = cells[target];
    let dpt = &d_cells[target];
    let ddpt = &dd_cells[target];
    #[cfg(test)]
    let weight = pt / z;
    let mut first = vec![0.0; dim];
    let mut second = vec![0.0; dim * dim];
    for i in 0..dim {
        first[i] = dpt[i] / z - pt * dz[i] / (z * z);
        for j in 0..dim {
            second[i * dim + j] = ddpt[i * dim + j] / z
                - (dpt[i] * dz[j] + dz[i] * dpt[j]) / (z * z)
                - pt * ddz[i * dim + j] / (z * z)
                + 2.0 * pt * dz[i] * dz[j] / (z * z * z);
        }
    }
    WeightDerivatives {
        #[cfg(test)]
        weight,
        first,
        second,
    }
}

/// The image-expanded basis and the density matrix laid out on it.
///
/// Built once and reused across grid batches, because assembling it costs
/// more than evaluating a few thousand points against it.
pub struct SupercellDensity {
    pub basis: MolBasis,
    /// `D` between every pair of image cells, `(n_cell * nao)^2`.
    pub density: Matrix,
}

/// Expand a periodic density into an explicit supercell of images.
pub fn supercell_density(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
) -> SupercellDensity {
    let n = mb.nao_sph;
    let imgs = lattice.translations_within(cutoff);
    let ncell = imgs.len();
    let mut basis = shifted(mb, imgs[0]);
    for t in &imgs[1..] {
        basis = concat(&basis, &shifted(mb, *t), 0);
    }
    let mut d = Matrix::zeros(ncell * n, ncell * n);
    for c in 0..ncell {
        for e in 0..ncell {
            let sep = imgs[e] - imgs[c];
            let Some(block) = density.iter().find(|b| (b.translation - sep).norm() < 1e-8) else {
                continue;
            };
            for i in 0..n {
                for j in 0..n {
                    d[(c * n + i, e * n + j)] = block.matrix[(i, j)];
                }
            }
        }
    }
    SupercellDensity { basis, density: d }
}

/// `rho(r)` at each point, and optionally `grad rho`.
///
/// Returned as `(rho, grad)` with `grad[k]` the k-th Cartesian component,
/// matching the layout the molecular XC assembly expects.
pub fn periodic_density(
    sup: &SupercellDensity,
    points: &[Vec3],
    want_grad: bool,
) -> (Vec<f64>, Option<[Vec<f64>; 3]>) {
    let nao = sup.basis.nao_sph;
    let batch = crate::grid::eval::eval_aos(&sup.basis, points, want_grad);
    // The explicit supercell matrix is block sparse: only cell separations
    // carried by `density` are populated. Contracting its many zero entries
    // made converging the image cutoff prohibitively expensive (quadratic in
    // the number of image AOs). Keeping the exact non-zero list makes the
    // work linear in the number of retained density-block entries.
    let nonzero: Vec<(usize, usize, f64)> = sup
        .density
        .as_slice()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(flat, value)| (value != 0.0).then_some((flat / nao, flat % nao, value)))
        .collect();
    let npts = points.len();
    if let Some(gb) = batch.grad.as_ref() {
        let values: Vec<(f64, [f64; 3])> = (0..npts)
            .into_par_iter()
            .map(|idx| {
                let mut value = 0.0;
                let mut g = [0.0; 3];
                for &(i, j, dij) in &nonzero {
                    let (pi, pj) = (batch.phi[(idx, i)], batch.phi[(idx, j)]);
                    value += dij * pi * pj;
                    for k in 0..3 {
                        g[k] += dij * (gb[k][(idx, i)] * pj + pi * gb[k][(idx, j)]);
                    }
                }
                (value, g)
            })
            .collect();
        let mut rho = Vec::with_capacity(npts);
        let mut grad = [
            Vec::with_capacity(npts),
            Vec::with_capacity(npts),
            Vec::with_capacity(npts),
        ];
        for (value, g) in values {
            rho.push(value);
            for k in 0..3 {
                grad[k].push(g[k]);
            }
        }
        (rho, Some(grad))
    } else {
        let mut rho = vec![0.0; npts];
        rho.par_iter_mut().enumerate().for_each(|(idx, value)| {
            for &(i, j, dij) in &nonzero {
                *value += dij * batch.phi[(idx, i)] * batch.phi[(idx, j)];
            }
        });
        (rho, None)
    }
}

/// Electrons per cell from the density matrix: `sum_T Tr[D(T) S(T)^T]`.
///
/// The Gamma-point electron count, computed without any quadrature — which
/// is what makes it a reference for the integrated density.
pub fn electron_count(mb: &MolBasis, density: &[ImageBlock]) -> f64 {
    let n = mb.nao_sph;
    let mut ne = 0.0;
    for b in density {
        let comb = concat(mb, &shifted(mb, b.translation), 0);
        let s = crate::integrals::overlap_matrix(&comb);
        for i in 0..n {
            for j in 0..n {
                ne += b.matrix[(i, j)] * s[(i, n + j)];
            }
        }
    }
    ne
}

/// Fixed-cutoff dimensional Becke virial for an owner-carried point.
/// The radial/angular offset and the physical switch radius remain fixed.
pub(crate) fn dimensional_weight_virial(
    mol: &Molecule,
    geometry: &super::PeriodicGeometry,
    point: Vec3,
    owner: usize,
    cutoff: f64,
    offset_strain: &[[Vec3; 3]; 3],
) -> Result<[[f64; 3]; 3]> {
    if owner >= mol.len() || !point.norm().is_finite() || !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(crate::DftError::InvalidInput(
            "invalid dimensional Becke strain point/owner/cutoff".into(),
        ));
    }
    let centers = dimensional_local_images(mol, geometry, point, cutoff)?;
    let Some((_, gradients, _)) =
        image_weight_gradient(mol, &centers, point, owner, &Switch::dimensional(cutoff))
    else {
        return Ok([[0.0; 3]; 3]);
    };
    let origin = mol.atoms[owner].position;
    let mut result = [[0.0; 3]; 3];
    let mut centre_gradient = Vec3::zero();
    for (gradient, (position, _, _)) in gradients.iter().zip(&centers) {
        let relative = *position - origin;
        centre_gradient += *gradient;
        for a in 0..3 {
            for b in 0..3 {
                result[a][b] += gradient.get(a) * relative.get(b);
            }
        }
    }
    // Translation invariance gives the point gradient as minus the sum of
    // the fixed-point centre gradients. The angular offset rotates in addition
    // to the affine motion of the owner.
    for a in 0..3 {
        for b in 0..3 {
            result[a][b] -= centre_gradient.dot(offset_strain[a][b]);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {

    #[test]
    fn angular_grid_frame_strain_matches_all_ordered_displacements() {
        use crate::acceptance::{components, stencils};
        use crate::periodic::{features::PeriodicBoundary, PeriodicGeometry};

        let lattice = [
            Vec3::new(5.2, 0.9, 0.4),
            Vec3::new(0.7, 4.8, 1.1),
            Vec3::new(0.2, -0.5, 5.6),
        ];
        let local_offset = Vec3::new(0.73, -0.58, 0.81);
        let frame = |a: [Vec3; 3]| {
            let geometry =
                PeriodicGeometry::new(Lattice::new(a[0], a[1], a[2]), PeriodicBoundary::FULL_3D)
                    .unwrap();
            AngularGridFrame::bulk(&geometry).unwrap()
        };
        let base = frame(lattice);
        let analytic = base.offset_strain(base.to_cartesian(local_offset));
        let components: Vec<_> = (0..3)
            .flat_map(|a| (0..3).flat_map(move |b| (0..3).map(move |d| (a, b, d))))
            .collect();
        let names: Vec<_> = components
            .iter()
            .map(|(a, b, d)| format!("angular offset {d}, strain [{a},{b}]"))
            .collect();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &[([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second)],
            |steps, stencil| {
                components
                    .iter()
                    .map(|&(a, b, d)| {
                        let value = |h: f64| {
                            let lattice = lattice.map(|v| {
                                let mut x = v.to_array();
                                x[a] += h * x[b];
                                Vec3::new(x[0], x[1], x[2])
                            });
                            frame(lattice).to_cartesian(local_offset).get(d)
                        };
                        std::array::from_fn(|i| {
                            let h = steps[i];
                            (stencil.first(value(h), value(-h), 0.0, 0.0, h)
                                - analytic[a][b].get(d))
                            .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
        components::assert_no_failures("all angular-frame strain components", &failures);
    }

    /// Analytic total (owner-riding) Becke weight derivatives: first order
    /// against the folded fixed-point gradient, second order against a
    /// four-step fit of central differences of that independent gradient.
    #[test]
    fn periodic_weight_hessian_matches_the_gradient_and_a_four_step_fit() {
        let lattice = Lattice::new(
            Vec3::new(4.2, 0.0, 0.0),
            Vec3::new(0.3, 4.6, 0.0),
            Vec3::new(0.2, -0.25, 4.9),
        );
        let base = Molecule::new(vec![
            crate::system::Atom {
                z: 1,
                position: Vec3::new(0.1, 0.2, -0.3),
            },
            crate::system::Atom {
                z: 8,
                position: Vec3::new(1.2, 0.9, 0.8),
            },
        ]);
        let shell = 1;
        let offsets = [
            Vec3::new(0.9, -0.4, 0.5),
            Vec3::new(-1.3, 0.7, 0.2),
            Vec3::new(0.3, 1.1, -0.9),
        ];
        for owner in 0..2 {
            for offset in offsets {
                let total_first = |mol: &Molecule| -> Vec<f64> {
                    let point = mol.atoms[owner].position + offset;
                    let (_, folded) = periodic_weight_gradient(mol, &lattice, point, owner, shell);
                    let mut sum = Vec3::zero();
                    for g in &folded {
                        sum += *g;
                    }
                    let mut out = vec![0.0; 6];
                    for atom in 0..2 {
                        let mut g = folded[atom];
                        if atom == owner {
                            g -= sum;
                        }
                        out[3 * atom] = g.x;
                        out[3 * atom + 1] = g.y;
                        out[3 * atom + 2] = g.z;
                    }
                    out
                };
                let point = base.atoms[owner].position + offset;
                let analytic = periodic_weight_hessian(&base, &lattice, point, owner, shell);
                let (weight, _) = periodic_weight_gradient(&base, &lattice, point, owner, shell);
                assert!(
                    (analytic.weight - weight).abs() < 1e-14,
                    "weight {} against {weight}",
                    analytic.weight
                );
                assert!(
                    analytic
                        .second
                        .iter()
                        .chain(&analytic.first)
                        .all(|v| v.is_finite()),
                    "non-finite derivative"
                );
                let reference = total_first(&base);
                for i in 0..6 {
                    assert!(
                        (analytic.first[i] - reference[i]).abs() < 1e-12,
                        "first derivative {i}"
                    );
                }
                if analytic.second.iter().all(|v| v.abs() < 1e-10) {
                    continue; // a point deep inside its own cell
                }
                let steps = [4e-3, 2e-3, 1e-3, 5e-4];
                for column in [0usize, 4] {
                    let (atom, axis) = (column / 3, column % 3);
                    let moved = |h: f64| {
                        let mut mol = base.clone();
                        match axis {
                            0 => mol.atoms[atom].position.x += h,
                            1 => mol.atoms[atom].position.y += h,
                            _ => mol.atoms[atom].position.z += h,
                        }
                        total_first(&mol)
                    };
                    let errors: Vec<f64> = steps
                        .iter()
                        .map(|&h| {
                            let (plus, minus) = (moved(h), moved(-h));
                            (0..6)
                                .map(|row| {
                                    ((plus[row] - minus[row]) / (2.0 * h)
                                        - analytic.second[row * 6 + column])
                                        .abs()
                                })
                                .fold(0.0_f64, |m, e| {
                                    if e.is_finite() {
                                        m.max(e)
                                    } else {
                                        f64::INFINITY
                                    }
                                })
                        })
                        .collect();
                    if errors.iter().all(|e| *e < 1e-11) {
                        continue;
                    }
                    let (x, y): (Vec<f64>, Vec<f64>) = steps
                        .iter()
                        .zip(&errors)
                        .map(|(h, e)| (h.ln(), e.ln()))
                        .unzip();
                    let n = 4.0;
                    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
                    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
                    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
                    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
                    let (slope, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
                    println!("Becke weight owner {owner} offset {offset:?} column {column}: errors {errors:?} p = {slope:.6} R2 = {r2:.8}");
                    assert!((slope - 2.0).abs() < 0.05 && r2 > 0.9999);
                }
                for i in 0..6 {
                    for j in 0..6 {
                        assert!(
                            (analytic.second[i * 6 + j] - analytic.second[j * 6 + i]).abs() < 1e-10,
                            "owner {owner} offset {offset:?} element ({i},{j}): {} vs {}",
                            analytic.second[i * 6 + j],
                            analytic.second[j * 6 + i]
                        );
                    }
                }
            }
        }
    }
    use super::*;
    use crate::basis::data;
    use crate::system::Atom;

    fn h2_cell(l: f64) -> (Molecule, MolBasis, Lattice) {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.0, 0.7),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        (mol, mb, Lattice::cubic(l))
    }

    /// The periodic Becke weights must tile space: for any point, the
    /// weights of the cell's atoms and all their images sum to one.
    ///
    /// This is the property the whole construction rests on, and it is
    /// checkable at a point without any integration. It is also where the
    /// shared denominator earns its keep — the sum telescopes only because
    /// every image is weighed against the same neighbour set.
    ///
    /// The probes include a point far outside the cell, which is the case a
    /// list of images centred on the origin gets wrong.
    #[test]
    fn periodic_becke_weights_are_a_partition_of_unity() {
        let (mol, _, lat) = h2_cell(6.0);
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.5, 1.3, -0.4),
            Vec3::new(-1.0, 2.9, 2.2),
            Vec3::new(3.0, 3.0, 3.0),
            Vec3::new(17.0, -11.0, 23.0),
        ] {
            let sw = Switch::periodic(&mol, &lat, 1);
            let centers = local_images(&mol, &lat, p, sw.cut);
            let total: f64 = (0..centers.len())
                .map(|i| local_becke_weight(&centers, i, p, &sw))
                .sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "periodic Becke weights sum to {total} at {p:?}, not 1"
            );
        }
    }

    /// A primitive cell and an equivalent supercell must select the same
    /// physical neighbours and assign the same weight to the same image.
    /// A fixed lattice-index shell fails this identity because its physical
    /// size changes when a lattice vector is doubled.
    #[test]
    fn local_neighbourhood_is_supercell_invariant() {
        let (mol, _, lat) = h2_cell(6.0);
        let sc_lat = lat.supercell([2, 1, 1]);
        let sc_mol = Molecule::new(
            mol.atoms
                .iter()
                .cloned()
                .chain(mol.atoms.iter().map(|a| Atom {
                    z: a.z,
                    position: a.position + lat.a[0],
                }))
                .collect(),
        );
        let r1 = local_cutoff(&mol, &lat, 1);
        let r2 = local_cutoff(&sc_mol, &sc_lat, 1);
        assert!((r1 - r2).abs() < 1e-14);

        for p in [
            Vec3::new(0.4, -0.2, 0.9),
            Vec3::new(2.9, 2.9, 2.9),
            Vec3::new(-4.1, 1.0, 5.5),
        ] {
            let c1 = local_images(&mol, &lat, p, r1);
            let c2 = local_images(&sc_mol, &sc_lat, p, r2);
            assert_eq!(c1.len(), c2.len(), "different image counts at {p:?}");
            for (center, radius, _) in &c1 {
                assert!(
                    c2.iter().any(|(other, other_radius, _)| {
                        (*center - *other).norm() < 1e-10 && (*radius - *other_radius).abs() < 1e-14
                    }),
                    "physical centre {center:?} is missing in the supercell"
                );
            }

            let t1 = c1
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (a.0 - p).norm2().total_cmp(&(b.0 - p).norm2()))
                .map(|(i, _)| i)
                .expect("the local set is non-empty");
            let target = c1[t1].0;
            let t2 = c2
                .iter()
                .position(|(c, _, _)| (*c - target).norm() < 1e-10)
                .expect("the same image is present in the wider set");
            let w1 = local_becke_weight(&c1, t1, p, &Switch::periodic(&mol, &lat, 1));
            let w2 = local_becke_weight(&c2, t2, p, &Switch::periodic(&sc_mol, &sc_lat, 1));
            assert!(
                (w1 - w2).abs() < 1e-12,
                "equivalent cells assign different weights: {w1} vs {w2}"
            );
        }
    }

    /// The integrated density must be the electron count per cell.
    ///
    /// Two independent routes to the same number: quadrature of `rho` on
    /// the periodic grid, against `sum_T Tr[D(T) S(T)]` from the analytic
    /// overlap. It fails if the partition does not tile, if the density's
    /// image expansion is truncated too early, or if the grid is too
    /// coarse — and those are the three ways a periodic XC term goes wrong.
    #[test]
    fn integrated_density_is_the_electron_count() {
        let (mol, mb, lat) = h2_cell(6.0);
        let n = mb.nao_sph;
        // A plausible density: two electrons shared over the bond.
        let mut d = Matrix::zeros(n, n);
        d[(0, 0)] = 0.6;
        d[(1, 1)] = 0.6;
        d[(0, 1)] = 0.4;
        d[(1, 0)] = 0.4;
        let density = vec![ImageBlock {
            translation: Vec3::zero(),
            matrix: d,
        }];

        let want = electron_count(&mb, &density);
        let integrate = |level: usize| -> f64 {
            let opts = GridOptions {
                level,
                ..GridOptions::default()
            };
            let grid = PeriodicGrid::build(&mol, &lat, &opts, 1).unwrap();
            let sup = supercell_density(&mb, &lat, &density, 24.0);
            let (rho, _) = periodic_density(&sup, &grid.points, false);
            grid.weights.iter().zip(&rho).map(|(w, r)| w * r).sum()
        };
        let coarse = integrate(4);
        let fine = integrate(9);
        assert!(
            (fine - want).abs() < 2e-6,
            "grid electron count {fine} vs analytic {want}"
        );
        // The residual is quadrature error, not a defect of the partition,
        // so it has to fall when the grid is refined. Asserting the value
        // alone would pass just as well on a construction that happens to
        // be wrong by a constant.
        assert!(
            (fine - want).abs() < 0.15 * (coarse - want).abs(),
            "refining the grid barely helped: level 4 off by {}, level 9 by {}",
            (coarse - want).abs(),
            (fine - want).abs()
        );
    }

    /// The result must not depend on how the cell is chosen: a supercell
    /// holding two copies of the same content must integrate to twice the
    /// electron count.
    ///
    /// Extensivity is the periodic analogue of size consistency, and it
    /// catches the errors a single-cell check cannot — a partition that
    /// tiles but is mis-centred, or an image sum that double counts.
    #[test]
    fn electron_count_is_extensive_in_a_supercell() {
        let (mol, mb, lat) = h2_cell(6.0);
        let n = mb.nao_sph;
        let mut d = Matrix::zeros(n, n);
        d[(0, 0)] = 0.6;
        d[(1, 1)] = 0.6;
        d[(0, 1)] = 0.4;
        d[(1, 0)] = 0.4;
        let opts = GridOptions {
            level: 7,
            ..GridOptions::default()
        };

        let single = {
            let density = vec![ImageBlock {
                translation: Vec3::zero(),
                matrix: d.clone(),
            }];
            let grid = PeriodicGrid::build(&mol, &lat, &opts, 1).unwrap();
            let sup = supercell_density(&mb, &lat, &density, 24.0);
            let (rho, _) = periodic_density(&sup, &grid.points, false);
            grid.weights
                .iter()
                .zip(&rho)
                .map(|(w, r)| w * r)
                .sum::<f64>()
        };

        // The 2x1x1 supercell: same physical system, twice the cell.
        let sc_lat = lat.supercell([2, 1, 1]);
        let sc_mol = Molecule::new(
            mol.atoms
                .iter()
                .cloned()
                .chain(mol.atoms.iter().map(|a| Atom {
                    z: a.z,
                    position: a.position + lat.a[0],
                }))
                .collect(),
        );
        let set = data::load("sto-3g").unwrap();
        let sc_mb = MolBasis::build(&sc_mol, &set).unwrap();
        let mut sc_d = Matrix::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                sc_d[(i, j)] = d[(i, j)];
                sc_d[(n + i, n + j)] = d[(i, j)];
            }
        }
        let sc_density = vec![ImageBlock {
            translation: Vec3::zero(),
            matrix: sc_d,
        }];
        // Isolate the two things that could disagree: the density's own
        // bookkeeping, and the quadrature. If the analytic counts already
        // differ, the supercell is not the same physical system and no grid
        // could rescue it.
        let n_single = electron_count(
            &mb,
            &[ImageBlock {
                translation: Vec3::zero(),
                matrix: d.clone(),
            }],
        );
        let n_double = electron_count(&sc_mb, &sc_density);
        assert!(
            (n_double - 2.0 * n_single).abs() < 1e-10,
            "analytic counts already differ: {n_double} vs twice {n_single}"
        );

        let sc_grid = PeriodicGrid::build(&sc_mol, &sc_lat, &opts, 1).unwrap();
        let sc_sup = supercell_density(&sc_mb, &sc_lat, &sc_density, 24.0);
        let (sc_rho, _) = periodic_density(&sc_sup, &sc_grid.points, false);
        let doubled: f64 = sc_grid
            .weights
            .iter()
            .zip(&sc_rho)
            .map(|(w, r)| w * r)
            .sum();
        assert!(
            (doubled - 2.0 * single).abs() < 1e-5,
            "supercell integrates to {doubled}, not twice {single}"
        );
    }

    /// Quadrature of a smooth periodic function on the periodic grid:
    /// `F(r) = sum_(A,T) N exp(-alpha |r - A - T|^2)` with normalized
    /// Gaussians (`N = (alpha/pi)^(3/2)`), so the exact integral per atom and
    /// cell is one for any geometry. Returns the quadrature of one cell.
    fn gaussian_quadrature(mol: &Molecule, lat: &Lattice, shell: i64, hard: bool) -> f64 {
        const ALPHA: f64 = 0.5;
        let opts = GridOptions {
            level: 2,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        };
        let sw = if hard {
            let soft = Switch::periodic(mol, lat, shell);
            Switch {
                on: soft.cut,
                cut: soft.cut,
            }
        } else {
            Switch::periodic(mol, lat, shell)
        };
        let grid = PeriodicGrid::build_with_images(
            mol,
            &opts,
            sw,
            true,
            None,
            |point| Ok(local_images(mol, lat, point, sw.cut)),
            |_, z, count| radial_treutler_ahlrichs(count, treutler_xi(z)),
        )
        .unwrap();
        let norm = (ALPHA / std::f64::consts::PI).powf(1.5);
        grid.points
            .iter()
            .zip(&grid.weights)
            .map(|(point, weight)| {
                let value: f64 = local_images(mol, lat, *point, 9.0)
                    .iter()
                    .map(|(centre, _, _)| norm * (-ALPHA * (*point - *centre).norm2()).exp())
                    .sum();
                weight * value
            })
            .sum()
    }

    fn skewed_cell() -> (Molecule, Lattice) {
        let lattice = Lattice::new(
            Vec3::new(4.2, 0.0, 0.0),
            Vec3::new(0.3, 4.6, 0.0),
            Vec3::new(0.2, -0.25, 4.9),
        );
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.2, -0.3),
            },
            Atom {
                z: 8,
                position: Vec3::new(1.2, 0.9, 0.8),
            },
        ]);
        (mol, lattice)
    }

    /// Strain `eps = h e_a e_b^T`: atoms and cell vectors move as
    /// `x -> (1 + eps) x`.
    fn strained(
        mol: &Molecule,
        lat: &Lattice,
        (a, b): (usize, usize),
        h: f64,
    ) -> (Molecule, Lattice) {
        let apply = |v: Vec3| {
            let mut c = v.to_array();
            c[a] += h * v.to_array()[b];
            Vec3::new(c[0], c[1], c[2])
        };
        let atoms = mol
            .atoms
            .iter()
            .map(|a| Atom {
                z: a.z,
                position: apply(a.position),
            })
            .collect();
        (
            Molecule::new(atoms),
            Lattice::new(apply(lat.a[0]), apply(lat.a[1]), apply(lat.a[2])),
        )
    }

    fn displaced(mol: &Molecule, h: f64) -> Molecule {
        let mut out = mol.clone();
        out.atoms[1].position.z += h;
        out
    }

    /// Successive central-difference derivatives `D(h) = (I(h) - I(-h))/2h` of
    /// a smooth function differ by `3/4 c h^2 + O(h^4)`; a noise floor of
    /// jumps in `I` makes the differences stop falling. Returns the exponent
    /// and R^2 of `log|D(h_k) - D(h_k+1)|` against `log h_k`.
    fn successive_difference_fit(
        label: &str,
        steps: &[f64],
        integral: impl Fn(f64) -> f64,
    ) -> (f64, f64) {
        let derivative: Vec<f64> = steps
            .iter()
            .map(|&h| (integral(h) - integral(-h)) / (2.0 * h))
            .collect();
        let errors: Vec<f64> = derivative.windows(2).map(|w| (w[0] - w[1]).abs()).collect();
        let h: Vec<f64> = steps[..errors.len()].to_vec();
        let (x, y): (Vec<f64>, Vec<f64>) =
            h.iter().zip(&errors).map(|(h, e)| (h.ln(), e.ln())).unzip();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        let (slope, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
        println!(
            "{label}: I(0) {:.12}, D(h) {derivative:?}, |D(h_k) - D(h_k+1)| {errors:?} at h {h:?}, p = {slope:.4}, R2 = {r2:.6}",
            integral(0.0)
        );
        (slope, r2)
    }

    const SMOOTH_STEPS: [f64; 5] = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4];

    /// The switched quadrature is a smooth function of the geometry, under a
    /// shear, a volume-changing strain (the switching radii scale with the
    /// length per atom) and a displacement.
    #[test]
    fn switched_quadrature_is_smooth_under_strain_and_displacement() {
        let (mol, lat) = skewed_cell();
        for shell in [1, 2] {
            for component in [(2usize, 1usize), (0, 0), (1, 1)] {
                let strain = |h: f64| {
                    let (m, l) = strained(&mol, &lat, component, h);
                    gaussian_quadrature(&m, &l, shell, false)
                };
                let label = format!("strain {component:?}, shell {shell}");
                let (p, r2) = successive_difference_fit(&label, &SMOOTH_STEPS, strain);
                assert!(
                    (p - 2.0).abs() < 0.1 && r2 > 0.9999,
                    "{label}: p {p}, R2 {r2}"
                );
            }
            let shift = |h: f64| gaussian_quadrature(&displaced(&mol, h), &lat, shell, false);
            let (p, r2) = successive_difference_fit(
                &format!("atom 1 along z, shell {shell}"),
                &SMOOTH_STEPS,
                shift,
            );
            assert!(
                (p - 2.0).abs() < 0.1 && r2 > 0.9999,
                "displacement, shell {shell}: p {p}, R2 {r2}"
            );
        }
    }

    /// The volume dependence of the switching radii: the derivative of the
    /// weight with respect to `ln L` at fixed distances (the radii scale with
    /// the length per atom) against four-step central differences, for two
    /// centres whose distances to the point both lie in the switching zone.
    #[test]
    fn weight_derivative_with_respect_to_the_switching_scale_passes_a_four_step_fit() {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 8,
                position: Vec3::new(3.0, 0.0, 0.0),
            },
        ]);
        let centers: Vec<(Vec3, f64, usize)> = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.position, bragg_slater_bohr(a.z), i))
            .collect();
        let point = Vec3::new(1.2, 0.3, 0.2);
        let switch = |scale: f64| Switch {
            on: scale,
            cut: 2.5 * scale,
        };
        let steps = [4e-3, 2e-3, 1e-3, 5e-4];
        for owner in 0..2 {
            let (weight, _, analytic) =
                image_weight_gradient(&mol, &centers, point, owner, &switch(1.0)).unwrap();
            let value = |scale: f64| local_becke_weight(&centers, owner, point, &switch(scale));
            assert!((value(1.0) - weight).abs() < 1e-14);
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| ((value(1.0 + h) - value(1.0 - h)) / (2.0 * h) - analytic).abs())
                .collect();
            let (x, y): (Vec<f64>, Vec<f64>) = steps
                .iter()
                .zip(&errors)
                .map(|(h, e)| (h.ln(), e.ln()))
                .unzip();
            let (mx, my) = (x.iter().sum::<f64>() / 4.0, y.iter().sum::<f64>() / 4.0);
            let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
            let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
            let (slope, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
            println!("dw/dlnL owner {owner}: weight {weight:.6}, analytic {analytic:.6e}, errors {errors:?}, p = {slope:.4}, R2 = {r2:.7}");
            assert!(
                analytic.abs() > 1e-4,
                "the scale derivative must be resolved"
            );
            assert!((slope - 2.0).abs() < 0.1 && r2 > 0.9999, "owner {owner}");
        }
    }

    /// The analytic strain virial of the weight (centre virial and the volume
    /// dependence of the switching radii) against four-step central
    /// differences of the weight of a point that rides its owner atom.
    #[test]
    fn weight_virial_passes_a_four_step_fit_under_volume_changing_strains() {
        let (mol, lat) = skewed_cell();
        let shell = 1;
        let steps = [4e-3, 2e-3, 1e-3, 5e-4];
        // Points between the two atoms, with intermediate weights.
        let offsets = [
            Vec3::new(0.9, -0.4, 0.5),
            Vec3::new(-1.3, 0.7, 0.2),
            Vec3::new(0.3, 1.1, -0.9),
        ];
        let mut fitted = 0;
        for owner in 0..2 {
            for offset in offsets {
                let virial = periodic_weight_virial(
                    &mol,
                    &lat,
                    mol.atoms[owner].position + offset,
                    owner,
                    shell,
                );
                for component in [(0usize, 0usize), (1, 1), (2, 1), (0, 2)] {
                    let weight = |h: f64| -> f64 {
                        let (m, l) = strained(&mol, &lat, component, h);
                        let point = m.atoms[owner].position + offset;
                        periodic_weight_gradient(&m, &l, point, owner, shell).0
                    };
                    let analytic = virial[component.0][component.1];
                    let errors: Vec<f64> = steps
                        .iter()
                        .map(|&h| ((weight(h) - weight(-h)) / (2.0 * h) - analytic).abs())
                        .collect();
                    if errors.iter().all(|e| *e < 1e-12) {
                        continue;
                    }
                    let (x, y): (Vec<f64>, Vec<f64>) = steps
                        .iter()
                        .zip(&errors)
                        .map(|(h, e)| (h.ln(), e.ln()))
                        .unzip();
                    let n = 4.0;
                    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
                    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
                    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
                    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
                    let (slope, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
                    fitted += 1;
                    println!(
                        "weight virial owner {owner} offset {offset:?} {component:?}: analytic \
                        {analytic:.6e}, errors {errors:?}, p = {slope:.4}, R2 = {r2:.7}"
                    );
                    assert!(
                        (slope - 2.0).abs() < 0.1 && r2 > 0.9999,
                        "owner {owner} offset {offset:?} {component:?}"
                    );
                }
            }
        }
        assert!(
            fitted >= 6,
            "only {fitted} virial components had a resolvable derivative"
        );
    }

    /// The old hard-sphere partition is not: the same fit fails. (This pins
    /// the diagnosis; it asserts the failure of the comparison scheme.)
    #[test]
    fn hard_sphere_quadrature_is_not_smooth() {
        let (mol, lat) = skewed_cell();
        let strain = |h: f64| {
            let (m, l) = strained(&mol, &lat, (2, 1), h);
            gaussian_quadrature(&m, &l, 1, true)
        };
        let (p, r2) =
            successive_difference_fit("hard sphere, shear [2][1], shell 1", &SMOOTH_STEPS, strain);
        let shift = |h: f64| gaussian_quadrature(&displaced(&mol, h), &lat, 1, true);
        let (q, s2) =
            successive_difference_fit("hard sphere, atom 1 along z, shell 1", &SMOOTH_STEPS, shift);
        let smooth = (p - 2.0).abs() < 0.1 && r2 > 0.9999 && (q - 2.0).abs() < 0.1 && s2 > 0.9999;
        assert!(
            !smooth,
            "the hard sphere unexpectedly passes the smoothness fit"
        );
    }

    /// Every weight, gradient and Hessian of the whole grid is finite, also
    /// for points in the switching zone and cells in the shadow of another
    /// centre (a factor below roundoff).
    #[test]
    fn weight_derivatives_are_finite_on_the_whole_grid() {
        let (mol, lat) = skewed_cell();
        let opts = GridOptions {
            level: 2,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        };
        for shell in [1, 2] {
            let grid = PeriodicGrid::build(&mol, &lat, &opts, shell).unwrap();
            for (i, point) in grid.points.iter().enumerate() {
                let owner = grid.atom_of[i];
                assert!(grid.weights[i].is_finite(), "weight {i}");
                let (w, g) = periodic_weight_gradient(&mol, &lat, *point, owner, shell);
                assert!(
                    w.is_finite() && g.iter().all(|v| v.to_array().iter().all(|x| x.is_finite())),
                    "gradient {i}"
                );
                if i % 7 == 0 {
                    let h = periodic_weight_hessian(&mol, &lat, *point, owner, shell);
                    assert!(
                        h.first.iter().chain(&h.second).all(|v| v.is_finite()),
                        "Hessian {i}"
                    );
                }
            }
        }
    }
}
