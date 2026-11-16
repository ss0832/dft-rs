// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic Grimme D3 dispersion: the lattice sum of the two-body D3 energy
//! of [`crate::dispersion`] with zero or Becke-Johnson damping, with analytic
//! forces and strain derivative.
//!
//! - Model: S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 132,
//!   154104 (2010); BJ damping: S. Grimme, S. Ehrlich, L. Goerigk,
//!   J. Comput. Chem. 32, 1456 (2011).
//! - Reference data, coordination-number function and C6 interpolation are
//!   those of the molecular module (`src/dispersion/data/*.csv`); nothing is
//!   duplicated here.
//!
//! With the atoms `i` at `r_i`, the lattice translations `T` and the pair
//! vector `v = r_j + T - r_i` of length `r` (the prime excludes `i = j`,
//! `T = 0`):
//!
//! ```text
//! CN_i  = sum_j sum_T' cn(r),   cn(r) = 1 / (1 + exp(-16 (R_ij / r - 1)))   (r < cn_cutoff)
//! E     = -1/2 sum_ij C6_ij(CN_i, CN_j) A_ij,
//! A_ij  = sum_T' [s6 g6(r) + s8 c8_ij g8(r)]                                (r < disp_cutoff)
//! ```
//!
//! where `C6_ij` is the Gaussian-weighted reference interpolation of the
//! molecular code, `c8_ij = 3 r2r4_i r2r4_j` and the damped radial functions
//! are `g_n = 1/(r^n + f^n)` with `f = a1 sqrt(c8_ij) + a2` (BJ) or
//! `g_n = 1/(r^n + 6 (rs_n r0_ij)^alpha_n r^(n - alpha_n))` (zero damping).
//! Since `C6_ij` does not depend on the image, the lattice sums `A_ij` and the
//! coordination sums are accumulated once per atom pair, and every
//! derivative follows in closed form:
//!
//! ```text
//! dE/dx = -1/2 sum_ij C6_ij dA_ij/dx + sum_i Ebar_i dCN_i/dx,
//! Ebar_i = dE/dCN_i = -sum_j A_ij dC6_ij/dCN_i
//! ```
//!
//! (`dC6/dCN` from forward-mode duals of the interpolation). The strain
//! derivative uses `x -> (1 + epsilon) x` for positions and cell vectors, so
//! `dr/d epsilon_ab = v_a v_b / r` and is symmetric. The sums are truncated
//! at fixed pair lists (spheres of radius `disp_cutoff` and `cn_cutoff` around
//! each atom); the derivatives are those of the truncated sums.
//!
//! D2 and the Chai-Head-Gordon term use the same lattice sum with
//! `C6_ij = sqrt(C6_i C6_j)` from the D2 tables and no coordination-number
//! term. D2 uses the original logistic damping and functional scale. The
//! Chai-Head-Gordon radial function is `g(r) = r^6 / (r^12 + a R_r^12)`.
//!
//! The optional three-body term ([`ThreeBody`]) adds, for every triple of
//! atoms with all three distances inside its cutoff,
//! `s9 sqrt(C6_ab C6_ac C6_bc) [3 cos cos cos + 1] / (r_ab r_ac r_bc)^3`
//! with the zero damping `1 / (1 + 6 (R0 / r_ab r_ac r_bc)^(alpha/3))`,
//! `R0` the product of the three pair radii scaled by `rs9`. Its energy,
//! forces, strain derivative and force constants are duals of the generic
//! energy function.

mod high_order;
pub use high_order::{
    dimensional_dispersion_force_constant_tensor, periodic_dispersion_force_constant_tensor,
    periodic_dispersion_force_constants_along, DispersionForceConstantTensor,
    DispersionForceConstants,
};

use crate::dispersion::d1;
use crate::dispersion::{
    atm_term_g, d2_pair_parameters, d2_radial_g, d2_s6, d3_bj_params, d3_zero_params, data,
    getc6_g, D3BjParams, D3ZeroParams, Dispersion, CHG_A_WB97XD,
};
use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::path_series::{geometry_energy_series, LatticePath};
use crate::periodic::stress::PeriodicStress;
use crate::periodic::{Cell, Lattice, PeriodicGeometry};
use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

/// Default radius of the pair sum (Bohr).
pub const DEFAULT_DISP_CUTOFF: f64 = 95.0;
/// Default radius of the coordination-number sum (Bohr).
pub const DEFAULT_CN_CUTOFF: f64 = 40.0;

type Tensor = [[f64; 3]; 3];

/// Damping function and its published parameters.
#[derive(Clone, Debug)]
pub enum D3Damping {
    Zero(D3ZeroParams),
    Bj(D3BjParams),
    /// Original D1 with explicitly supplied free-atom references.
    D1 {
        s6: f64,
        atoms: Arc<BTreeMap<u8, d1::AtomParameters>>,
    },
    /// D2 logistic damping on the D2 radii and constant pair coefficients.
    D2 {
        s6: f64,
    },
    /// Chai-Head-Gordon damping of the pair term `-C6/r^6` on the D2 tables,
    /// `g(r) = r^6 / (r^12 + a R_r^12)`: no coordination-number dependence
    /// (J.-D. Chai, M. Head-Gordon, Phys. Chem. Chem. Phys. 10, 6615 (2008)).
    Chg {
        a: f64,
    },
}

impl D3Damping {
    fn has_coordination_dependence(&self) -> bool {
        matches!(self, Self::Zero(_) | Self::Bj(_))
    }

    fn validate_three_body(&self) -> Result<()> {
        let model = match self {
            Self::D1 { .. } => "D1",
            Self::D2 { .. } => "D2",
            Self::Chg { .. } => "CHG",
            Self::Zero(_) | Self::Bj(_) => return Ok(()),
        };
        Err(DftError::Unsupported {
            feature: format!("a three-body term of the {model} pair dispersion"),
            missing: "the coordination-number dependent C6 coefficients of D3".into(),
        })
    }

    fn constant_pair(&self, zi: u8, zj: u8) -> Option<(f64, f64)> {
        match self {
            Self::D1 { atoms, .. } => Some(d1::pair_parameters(*atoms.get(&zi)?, *atoms.get(&zj)?)),
            _ => d2_pair_parameters(zi, zj),
        }
    }
}

/// Three-body (Axilrod-Teller-Muto) term of the D3 model.
#[derive(Clone, Copy, Debug)]
pub struct ThreeBody {
    /// Scale of the term (1 in the published model).
    pub s9: f64,
    /// Scale of the pair radii in the damping (4/3).
    pub rs9: f64,
    /// Exponent of the damping on the mean distance ratio (16).
    pub alpha: f64,
    /// Largest pair distance inside a triple (Bohr).
    pub cutoff: f64,
}

/// Default largest pair distance inside a triple (Bohr).
pub const DEFAULT_ABC_CUTOFF: f64 = 40.0;

impl Default for ThreeBody {
    fn default() -> Self {
        Self {
            s9: 1.0,
            rs9: 4.0 / 3.0,
            alpha: 16.0,
            cutoff: DEFAULT_ABC_CUTOFF,
        }
    }
}

/// A periodic empirical dispersion model with damping parameters and cutoffs.
#[derive(Clone, Debug)]
pub struct PeriodicDispersion {
    pub damping: D3Damping,
    /// Radius of the coordination-number sum (Bohr).
    pub cn_cutoff: f64,
    /// Radius of the pair sum (Bohr).
    pub disp_cutoff: f64,
    /// Optional three-body term (none by default).
    pub three_body: Option<ThreeBody>,
}

impl PeriodicDispersion {
    /// Check the geometry model required by a named functional.
    pub(crate) fn supplies(&self, required: crate::xc::RequiredNonlocalTerm) -> bool {
        use crate::xc::RequiredNonlocalTerm;
        match (required, &self.damping) {
            (RequiredNonlocalTerm::Chg, D3Damping::Chg { .. }) => true,
            (RequiredNonlocalTerm::D3Zero, D3Damping::Zero(parameters)) => {
                *parameters == crate::xc::wb97x_d3::PAIR_PARAMETERS && self.three_body.is_none()
            }
            _ => false,
        }
    }

    /// D1 takes free-atom references in primitive-cell atom order.
    /// Replicas and repeated elements use the same elemental reference.
    pub fn d1(
        numbers: &[u8],
        atoms: &[d1::AtomParameters],
        s6: f64,
        disp_cutoff: f64,
    ) -> Result<Self> {
        d1::validate_parameters(atoms, s6)?;
        if numbers.len() != atoms.len() || !disp_cutoff.is_finite() || disp_cutoff <= 0.0 {
            return Err(DftError::InvalidInput(
                "periodic D1 needs one reference per atom and a positive finite cutoff".into(),
            ));
        }
        let mut references = BTreeMap::new();
        for (&z, &atom) in numbers.iter().zip(atoms) {
            if z == 0
                || references
                    .insert(z, atom)
                    .is_some_and(|previous| previous != atom)
            {
                return Err(DftError::InvalidInput(format!(
                    "periodic D1 requires a consistent free-atom reference for atomic number {z}"
                )));
            }
        }
        Ok(Self {
            damping: D3Damping::D1 {
                s6,
                atoms: Arc::new(references),
            },
            cn_cutoff: DEFAULT_CN_CUTOFF,
            disp_cutoff,
            three_body: None,
        })
    }

    /// The periodic model for `kind` with the published parameters of
    /// `functional`; `None` for `Dispersion::None`. D2 and CHG use the
    /// shared constant-C6 pair sum.
    pub fn from_model(
        kind: &Dispersion,
        functional: &str,
        cn_cutoff: f64,
        disp_cutoff: f64,
    ) -> Result<Option<Self>> {
        let damping = match kind {
            Dispersion::None => return Ok(None),
            Dispersion::D1 { .. } => {
                return Err(DftError::InvalidInput(
                    "a periodic D1 model also needs the atomic numbers; use PeriodicDispersion::d1"
                        .into(),
                ));
            }
            Dispersion::D3Zero => D3Damping::Zero(d3_zero_params(functional)?),
            Dispersion::D3Bj => D3Damping::Bj(d3_bj_params(functional)?),
            Dispersion::D3ZeroAbc => D3Damping::Zero(d3_zero_params(functional)?),
            Dispersion::D3BjAbc => D3Damping::Bj(d3_bj_params(functional)?),
            Dispersion::Chg => D3Damping::Chg { a: CHG_A_WB97XD },
            Dispersion::D2 => D3Damping::D2 {
                s6: d2_s6(functional)?,
            },
        };
        if !(cn_cutoff.is_finite()
            && cn_cutoff > 0.0
            && disp_cutoff.is_finite()
            && disp_cutoff > 0.0)
        {
            return Err(DftError::InvalidInput(
                "`cn_cutoff` and `disp_cutoff` must be positive and finite (Bohr)".into(),
            ));
        }
        Ok(Some(Self {
            damping,
            cn_cutoff,
            disp_cutoff,
            three_body: matches!(kind, Dispersion::D3ZeroAbc | Dispersion::D3BjAbc)
                .then(ThreeBody::default),
        }))
    }

    /// The same D3 model with the three-body term.
    pub fn with_three_body(mut self, term: ThreeBody) -> Result<Self> {
        self.damping.validate_three_body()?;
        if !(term.s9.is_finite()
            && term.rs9 > 0.0
            && term.alpha > 0.0
            && term.cutoff.is_finite()
            && term.cutoff > 0.0)
        {
            return Err(DftError::InvalidInput(
                "the three-body term needs a finite scale and positive radius scale, exponent \
                 and cutoff"
                    .into(),
            ));
        }
        self.three_body = Some(term);
        Ok(self)
    }
}

/// Energy, forces and strain derivative of the periodic D3 sum.
#[derive(Clone, Debug)]
pub struct PeriodicDispersionResult {
    /// Dispersion energy per cell (Hartree).
    pub energy: f64,
    /// `-dE/dr_i` (Hartree/Bohr).
    pub forces: Vec<Vec3>,
    /// `dE/d(epsilon_ab)` per cell (Hartree), symmetric.
    pub strain_derivative: Tensor,
    /// Coordination numbers `CN_i` including all images.
    pub coordination_numbers: Vec<f64>,
}

impl PeriodicDispersionResult {
    pub(crate) fn add_to_gradient(&self, gradient: &mut [Vec3]) {
        for (value, force) in gradient.iter_mut().zip(&self.forces) {
            *value -= *force;
        }
    }
}

/// Evaluate a geometry-only correction on the same cell as the SCF.
pub(crate) fn cell_dispersion(
    cell: &Cell,
    model: &Option<PeriodicDispersion>,
) -> Result<Option<PeriodicDispersionResult>> {
    let Some(model) = model else { return Ok(None) };
    let atoms = &cell.molecule().atoms;
    let numbers: Vec<_> = atoms.iter().map(|atom| atom.z).collect();
    let positions: Vec<_> = atoms.iter().map(|atom| atom.position).collect();
    periodic_d3(&numbers, &positions, cell.lattice(), model).map(Some)
}

/// Add a dispersion strain derivative to a periodic stress as the term
/// `dispersion` and update the symmetric stress and rotation residual.
pub fn add_to_stress(stress: &mut PeriodicStress, strain_derivative: &Tensor) {
    stress.terms.push(("dispersion", *strain_derivative));
    for a in 0..3 {
        for b in 0..3 {
            stress.strain_derivative[a][b] += strain_derivative[a][b];
        }
    }
    let mut residual = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            stress.stress[a][b] = 0.5
                * (stress.strain_derivative[a][b] + stress.strain_derivative[b][a])
                / stress.volume;
            residual = residual
                .max((stress.strain_derivative[a][b] - stress.strain_derivative[b][a]).abs());
        }
    }
    stress.rotation_residual = residual;
}

/// Atoms and lattice of one geometry.
struct Geometry<'a> {
    positions: &'a [Vec3],
    lattice: &'a Lattice,
    reciprocal: [Vec3; 3],
}

impl<'a> Geometry<'a> {
    fn dimensional(positions: &'a [Vec3], geometry: &'a PeriodicGeometry) -> Self {
        Self {
            positions,
            lattice: geometry.lattice(),
            reciprocal: geometry.reciprocal(),
        }
    }

    fn bulk(positions: &'a [Vec3], lattice: &'a Lattice) -> Self {
        Self {
            positions,
            lattice,
            reciprocal: lattice.reciprocal(),
        }
    }
}

/// Calls `f(v, r)` for every translation `T` of the ordered pair `(i, j)`
/// with `|r_j - r_i + T| < cutoff` (`T = 0` is skipped for `i = j`). The
/// translation range is the exact box of the sphere in fractional
/// coordinates, so a skewed cell loses no image.
///
/// The image list is selected on a reference geometry while the pair vectors
/// come from the evaluated geometry, so that finite-difference tests see a
/// fixed (smooth) pair list; production passes the evaluated geometry twice.
/// Integer translations `n` of the ordered pair `(i, j)` with
/// `|r_j - r_i + n.A| < cutoff` at `geometry` (`n = 0` is skipped for
/// `i = j`). The range is the exact box of the sphere in fractional
/// coordinates, so a skewed cell loses no image.
fn image_indices(geometry: &Geometry, (i, j): (usize, usize), cutoff: f64) -> Vec<[i64; 3]> {
    let delta = geometry.positions[j] - geometry.positions[i];
    let b = geometry.reciprocal;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut lo = [0i64; 3];
    let mut hi = [0i64; 3];
    for k in 0..3 {
        let width = cutoff * b[k].norm() / two_pi;
        let s = delta.dot(b[k]) / two_pi;
        lo[k] = (-width - s).ceil() as i64;
        hi[k] = (width - s).floor() as i64;
    }
    let cutoff2 = cutoff * cutoff;
    let mut out = Vec::new();
    for n0 in lo[0]..=hi[0] {
        for n1 in lo[1]..=hi[1] {
            for n2 in lo[2]..=hi[2] {
                if i == j && n0 == 0 && n1 == 0 && n2 == 0 {
                    continue;
                }
                let n = [n0, n1, n2];
                if (delta + geometry.lattice.translation(n)).norm2() < cutoff2 {
                    out.push(n);
                }
            }
        }
    }
    out
}

fn for_each_image_pair(
    reference: &Geometry,
    current: &Geometry,
    (i, j): (usize, usize),
    cutoff: f64,
    mut f: impl FnMut(Vec3, f64),
) {
    let delta = current.positions[j] - current.positions[i];
    for n in image_indices(reference, (i, j), cutoff) {
        let v = delta + current.lattice.translation(n);
        f(v, v.norm());
    }
}

/// Damped radial function `g_n = 1/D` and its derivative for zero damping,
/// `D = r^n + 6 (rs r0)^alpha r^(n - alpha)`.
fn zero_term(r: f64, n: f64, scale: f64, alpha: f64, r0: f64) -> (f64, f64) {
    let weight = 6.0 * (scale * r0).powf(alpha);
    let d = r.powf(n) + weight * r.powf(n - alpha);
    let dd = n * r.powf(n - 1.0) + weight * (n - alpha) * r.powf(n - alpha - 1.0);
    (1.0 / d, -dd / (d * d))
}

/// `a(r) = s6 g6 + s8 c8 g8` and `da/dr`.
fn radial(damping: &D3Damping, r: f64, c8: f64, r0: f64) -> (f64, f64) {
    match damping {
        D3Damping::Zero(p) => {
            let (g6, d6) = zero_term(r, 6.0, p.rs6, p.alp6, r0);
            let (g8, d8) = zero_term(r, 8.0, p.rs8, p.alp8, r0);
            (p.s6 * g6 + p.s8 * c8 * g8, p.s6 * d6 + p.s8 * c8 * d8)
        }
        D3Damping::Bj(p) => {
            let f = c8.sqrt() * p.a1 + p.a2;
            let f6 = f.powi(6);
            let f8 = f.powi(8);
            let (r5, r6, r7, r8) = (r.powi(5), r.powi(6), r.powi(7), r.powi(8));
            let (d6, d8) = (r6 + f6, r8 + f8);
            (
                p.s6 / d6 + p.s8 * c8 / d8,
                -6.0 * p.s6 * r5 / (d6 * d6) - 8.0 * p.s8 * c8 * r7 / (d8 * d8),
            )
        }
        D3Damping::D2 { s6 } => {
            let damped = d2_radial_g(DualN::<f64, 1>::variable(r, 0), r0).scale(*s6);
            (damped.re, damped.im[0])
        }
        D3Damping::D1 { s6, .. } => {
            let damped = d1::radial(DualN::<f64, 1>::variable(r, 0), r0).scale(*s6);
            (damped.re, damped.im[0])
        }
        // g = r^6 / (r^12 + a R_r^12); `r0` carries R_r for this model.
        D3Damping::Chg { a } => {
            let b = a * r0.powi(12);
            let d = r.powi(12) + b;
            (r.powi(6) / d, 6.0 * r.powi(5) * (b - r.powi(12)) / (d * d))
        }
    }
}

/// Lattice sums of one ordered atom pair `(i, j)`: `S = sum f(r)`,
/// `G = sum f'(r) v/r` and `H = sum f'(r) v v^T / r`.
#[derive(Clone, Copy, Default)]
struct PairSum {
    value: f64,
    gradient: Vec3,
    strain: Tensor,
}

impl PairSum {
    fn add(&mut self, v: Vec3, r: f64, value: f64, slope: f64) {
        self.value += value;
        let unit_slope = slope / r;
        self.gradient += v * unit_slope;
        let components = v.to_array();
        for a in 0..3 {
            for b in 0..3 {
                self.strain[a][b] += unit_slope * components[a] * components[b];
            }
        }
    }
}

fn evaluate(
    numbers: &[u8],
    reference: &Geometry,
    current: &Geometry,
    settings: &PeriodicDispersion,
) -> Result<PeriodicDispersionResult> {
    let d = data();
    let n = numbers.len();
    if current.positions.len() != n || reference.positions.len() != n {
        return Err(DftError::InvalidInput(
            "dispersion needs one position per atom".into(),
        ));
    }
    if settings.three_body.is_some() {
        settings.damping.validate_three_body()?;
    }
    let pair_only = !settings.damping.has_coordination_dependence();
    validate_elements(numbers, &settings.damping)?;
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).collect();
    // Coordination sums: cn and its derivative.
    let coordination: Vec<PairSum> = pairs
        .par_iter()
        .map(|&(i, j)| {
            if pair_only {
                return PairSum::default();
            }
            let rco = d.rcov[numbers[i] as usize] + d.rcov[numbers[j] as usize];
            let mut sum = PairSum::default();
            for_each_image_pair(reference, current, (i, j), settings.cn_cutoff, |v, r| {
                let u = (-16.0 * (rco / r - 1.0)).exp();
                let cn = 1.0 / (1.0 + u);
                let slope = -16.0 * rco * u / (r * r * (1.0 + u) * (1.0 + u));
                sum.add(v, r, cn, slope);
            });
            sum
        })
        .collect();
    let mut cn = vec![0.0; n];
    for (&(i, _), s) in pairs.iter().zip(&coordination) {
        cn[i] += s.value;
    }
    // Damped lattice sums A_ij and their derivatives.
    let lattice_sums: Vec<PairSum> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (zi, zj) = (numbers[i] as usize, numbers[j] as usize);
            let (c8, r0) = if pair_only {
                (
                    0.0,
                    settings
                        .damping
                        .constant_pair(numbers[i], numbers[j])
                        .map_or(1.0, |p| p.1),
                )
            } else {
                (3.0 * d.r2r4[zi] * d.r2r4[zj], d.r0ab[zi - 1][zj - 1])
            };
            let mut sum = PairSum::default();
            for_each_image_pair(reference, current, (i, j), settings.disp_cutoff, |v, r| {
                let (value, slope) = radial(&settings.damping, r, c8, r0);
                sum.add(v, r, value, slope);
            });
            sum
        })
        .collect();
    // C6 and dC6/dCN_i (forward-mode duals of the molecular interpolation).
    let c6: Vec<(f64, f64)> = pairs
        .iter()
        .map(|&(i, j)| {
            if pair_only {
                return (
                    settings
                        .damping
                        .constant_pair(numbers[i], numbers[j])
                        .map_or(0.0, |p| p.0),
                    0.0,
                );
            }
            type Dual = DualN<f64, 2>;
            let c = getc6_g(
                d,
                numbers[i],
                numbers[j],
                Dual::variable(cn[i], 0),
                Dual::variable(cn[j], 1),
            );
            (c.re, c.im[0])
        })
        .collect();
    let mut energy = 0.0;
    let mut gradient = vec![Vec3::zero(); n];
    let mut strain = [[0.0; 3]; 3];
    let mut ebar = vec![0.0; n];
    for (k, &(i, _)) in pairs.iter().enumerate() {
        ebar[i] -= lattice_sums[k].value * c6[k].1;
    }
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let (a, s) = (&lattice_sums[k], &coordination[k]);
        energy -= 0.5 * c6[k].0 * a.value;
        // Explicit pair term and the coordination-number chain rule.
        let pair = a.gradient * (-0.5 * c6[k].0) + s.gradient * ebar[i];
        gradient[j] += pair;
        gradient[i] -= pair;
        for x in 0..3 {
            for y in 0..3 {
                strain[x][y] += -0.5 * c6[k].0 * a.strain[x][y] + ebar[i] * s.strain[x][y];
            }
        }
    }
    // Three-body term: its energy and first derivatives are forward duals of
    // the generic sum on the image lists of the reference geometry.
    if settings.three_body.is_some() {
        let images = image_lists(reference, n, settings);
        let plain: Vec<[f64; 3]> = current.positions.iter().map(|p| p.to_array()).collect();
        let rows: [[f64; 3]; 3] = std::array::from_fn(|k| current.lattice.a[k].to_array());
        energy += three_body_energy_g(numbers, &plain, &rows, &images, settings);
        type Move = DualN<f64, 3>;
        let rows_move: [[Move; 3]; 3] =
            std::array::from_fn(|k| std::array::from_fn(|c| Move::constant(rows[k][c])));
        let moved_gradient: Vec<[f64; 3]> = (0..n)
            .into_par_iter()
            .map(|atom| {
                let moved: Vec<[Move; 3]> = plain
                    .iter()
                    .enumerate()
                    .map(|(k, p)| {
                        std::array::from_fn(|c| {
                            if k == atom {
                                Move::variable(p[c], c)
                            } else {
                                Move::constant(p[c])
                            }
                        })
                    })
                    .collect();
                three_body_energy_g(numbers, &moved, &rows_move, &images, settings).im
            })
            .collect();
        for (total, extra) in gradient.iter_mut().zip(&moved_gradient) {
            *total += Vec3::new(extra[0], extra[1], extra[2]);
        }
        // Nine strain seeds: x -> (1 + eps) x for positions and lattice rows.
        type Strain = DualN<f64, 9>;
        let strained = |v: [f64; 3]| -> [Strain; 3] {
            std::array::from_fn(|c| {
                let mut x = Strain::constant(v[c]);
                for b in 0..3 {
                    x.im[3 * c + b] = v[b];
                }
                x
            })
        };
        let seeded: Vec<[Strain; 3]> = plain.iter().map(|p| strained(*p)).collect();
        let seeded_rows: [[Strain; 3]; 3] = std::array::from_fn(|k| strained(rows[k]));
        let jet = three_body_energy_g(numbers, &seeded, &seeded_rows, &images, settings);
        for a in 0..3 {
            for b in 0..3 {
                strain[a][b] += jet.im[3 * a + b];
            }
        }
    }
    Ok(PeriodicDispersionResult {
        energy,
        forces: gradient.into_iter().map(|g| -g).collect(),
        strain_derivative: strain,
        coordination_numbers: cn,
    })
}

/// One lattice image of an ordered atom pair: `v = r_j + n.A - r_i`.
type Image = (usize, usize, [i64; 3]);

/// Fixed lists of one geometry: the images of every ordered pair inside the
/// pair cutoff and inside the coordination-number cutoff and, for the
/// three-body term, the partners of every atom inside the triple cutoff with
/// their vectors at this geometry.
struct Images {
    disp: Vec<Image>,
    cn: Vec<Image>,
    near: Vec<Vec<(usize, [i64; 3], Vec3)>>,
}

fn image_lists(geometry: &Geometry, n: usize, settings: &PeriodicDispersion) -> Images {
    let list = |cutoff: f64| -> Vec<Image> {
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for t in image_indices(geometry, (i, j), cutoff) {
                    out.push((i, j, t));
                }
            }
        }
        out
    };
    let near = match settings.three_body {
        None => Vec::new(),
        Some(term) => (0..n)
            .map(|i| {
                let mut partners = Vec::new();
                for j in 0..n {
                    for t in image_indices(geometry, (i, j), term.cutoff) {
                        let v = geometry.positions[j] - geometry.positions[i]
                            + geometry.lattice.translation(t);
                        partners.push((j, t, v));
                    }
                }
                partners
            })
            .collect(),
    };
    Images {
        disp: list(settings.disp_cutoff),
        cn: if settings.damping.has_coordination_dependence() {
            list(settings.cn_cutoff)
        } else {
            Vec::new()
        },
        near,
    }
}

/// `r_j + t.A - r_i` for any scalar type.
fn image_vector<S: Scalar>(
    positions: &[[S; 3]],
    rows: &[[S; 3]; 3],
    i: usize,
    j: usize,
    t: [i64; 3],
) -> [S; 3] {
    std::array::from_fn(|c| {
        let mut v = positions[j][c] - positions[i][c];
        for k in 0..3 {
            v = v + rows[k][c].scale(t[k] as f64);
        }
        v
    })
}

fn norm2_g<S: Scalar>(v: &[S; 3]) -> S {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// `C6_ij` of every atom pair (row-major `n x n`) at the coordination
/// numbers of the image list; a constant for the pair-only model.
fn c6_matrix_g<S: Scalar>(
    numbers: &[u8],
    positions: &[[S; 3]],
    rows: &[[S; 3]; 3],
    cn: &[Image],
    settings: &PeriodicDispersion,
) -> Vec<S> {
    let d = data();
    let n = numbers.len();
    if !settings.damping.has_coordination_dependence() {
        return (0..n * n)
            .map(|k| {
                S::from_f64(
                    settings
                        .damping
                        .constant_pair(numbers[k / n], numbers[k % n])
                        .map_or(0.0, |q| q.0),
                )
            })
            .collect();
    }
    let one = S::from_f64(1.0);
    let mut coordination = vec![S::from_f64(0.0); n];
    for &(i, j, t) in cn {
        let rco = d.rcov[numbers[i] as usize] + d.rcov[numbers[j] as usize];
        let r = norm2_g(&image_vector(positions, rows, i, j, t)).sqrt();
        let u = (r.recip().scale(rco) - one).scale(-16.0).exp();
        coordination[i] = coordination[i] + (one + u).recip();
    }
    (0..n * n)
        .map(|k| {
            getc6_g(
                d,
                numbers[k / n],
                numbers[k % n],
                coordination[k / n],
                coordination[k % n],
            )
        })
        .collect()
}

/// Two-body energy `-1/2 sum_ij C6_ij A_ij` on the pair image list.
fn two_body_g<S: Scalar>(
    numbers: &[u8],
    positions: &[[S; 3]],
    rows: &[[S; 3]; 3],
    disp: &[Image],
    c6: &[S],
    settings: &PeriodicDispersion,
) -> S {
    let d = data();
    let n = numbers.len();
    let mut sums = vec![S::from_f64(0.0); n * n];
    for &(i, j, t) in disp {
        let (zi, zj) = (numbers[i] as usize, numbers[j] as usize);
        let r = norm2_g(&image_vector(positions, rows, i, j, t)).sqrt();
        let value = match &settings.damping {
            D3Damping::Zero(p) => {
                let r0 = d.r0ab[zi - 1][zj - 1];
                let c8 = 3.0 * d.r2r4[zi] * d.r2r4[zj];
                let term = |order: i32, scale: f64, alpha: f64| -> S {
                    (r.powi(order)
                        + r.powf(order as f64 - alpha)
                            .scale(6.0 * (scale * r0).powf(alpha)))
                    .recip()
                };
                term(6, p.rs6, p.alp6).scale(p.s6) + term(8, p.rs8, p.alp8).scale(p.s8 * c8)
            }
            D3Damping::Bj(p) => {
                let c8 = 3.0 * d.r2r4[zi] * d.r2r4[zj];
                let f = c8.sqrt() * p.a1 + p.a2;
                (r.powi(6) + S::from_f64(f.powi(6))).recip().scale(p.s6)
                    + (r.powi(8) + S::from_f64(f.powi(8)))
                        .recip()
                        .scale(p.s8 * c8)
            }
            D3Damping::D2 { s6 } => {
                let rr = d2_pair_parameters(numbers[i], numbers[j]).map_or(1.0, |q| q.1);
                d2_radial_g(r, rr).scale(*s6)
            }
            D3Damping::D1 { s6, .. } => {
                let r0 = settings
                    .damping
                    .constant_pair(numbers[i], numbers[j])
                    .map_or(1.0, |q| q.1);
                d1::radial(r, r0).scale(*s6)
            }
            D3Damping::Chg { a } => {
                let rr = d2_pair_parameters(numbers[i], numbers[j]).map_or(1.0, |q| q.1);
                r.powi(6) / (r.powi(12) + S::from_f64(a * rr.powi(12)))
            }
        };
        sums[i * n + j] = sums[i * n + j] + value;
    }
    let mut energy = S::from_f64(0.0);
    for k in 0..n * n {
        energy = energy - (c6[k] * sums[k]).scale(0.5);
    }
    energy
}

/// Three-body energy per cell: one third of the sum over every atom `i` of
/// the home cell and every unordered pair of its partners, with all three
/// distances of the triple inside the cutoff at the geometry of the lists.
fn three_body_g<S: Scalar>(
    numbers: &[u8],
    positions: &[[S; 3]],
    rows: &[[S; 3]; 3],
    near: &[Vec<(usize, [i64; 3], Vec3)>],
    c6: &[S],
    term: &ThreeBody,
) -> S {
    let d = data();
    let n = numbers.len();
    let cutoff2 = term.cutoff * term.cutoff;
    let radius =
        |a: usize, b: usize| term.rs9 * d.r0ab[numbers[a] as usize - 1][numbers[b] as usize - 1];
    let mut energy = S::from_f64(0.0);
    for (i, partners) in near.iter().enumerate() {
        for (p, &(j, tj, vj)) in partners.iter().enumerate() {
            let rij = image_vector(positions, rows, i, j, tj);
            let r2ij = norm2_g(&rij);
            for &(k, tk, vk) in &partners[p + 1..] {
                if (vj - vk).norm2() >= cutoff2 {
                    continue;
                }
                let product = c6[i * n + j] * c6[i * n + k] * c6[j * n + k];
                if !(product.value() > 0.0) {
                    continue;
                }
                let rik = image_vector(positions, rows, i, k, tk);
                let rjk: [S; 3] = std::array::from_fn(|c| rik[c] - rij[c]);
                let r0 = radius(i, j) * radius(i, k) * radius(j, k);
                energy = energy
                    + atm_term_g(
                        r2ij,
                        norm2_g(&rik),
                        norm2_g(&rjk),
                        product.sqrt().scale(term.s9),
                        r0,
                        term.alpha,
                    );
            }
        }
    }
    energy.scale(1.0 / 3.0)
}

/// Three-body energy alone (zero without the term).
fn three_body_energy_g<S: Scalar>(
    numbers: &[u8],
    positions: &[[S; 3]],
    rows: &[[S; 3]; 3],
    images: &Images,
    settings: &PeriodicDispersion,
) -> S {
    match &settings.three_body {
        None => S::from_f64(0.0),
        Some(term) => {
            let c6 = c6_matrix_g(numbers, positions, rows, &images.cn, settings);
            three_body_g(numbers, positions, rows, &images.near, &c6, term)
        }
    }
}

fn validate_elements(numbers: &[u8], damping: &D3Damping) -> Result<()> {
    for &z in numbers {
        let missing = if !damping.has_coordination_dependence() {
            damping.constant_pair(z, z).is_none()
        } else {
            z == 0 || z as usize > data().max_elem
        };
        if missing {
            return Err(DftError::Unsupported {
                feature: format!("periodic dispersion for atomic number {z}"),
                missing: "reference data of this dispersion model for the element".into(),
            });
        }
    }
    Ok(())
}

/// Dispersion energy per cell on fixed image lists for any scalar type: the
/// same sums as [`evaluate`], written so that the positions and the lattice
/// rows can carry derivative seeds. Every term of the model belongs here, so
/// that all derivatives follow from one definition of the energy.
fn energy_g<S: Scalar>(
    numbers: &[u8],
    positions: &[[S; 3]],
    rows: &[[S; 3]; 3],
    images: &Images,
    settings: &PeriodicDispersion,
) -> S {
    let c6 = c6_matrix_g(numbers, positions, rows, &images.cn, settings);
    let mut energy = two_body_g(numbers, positions, rows, &images.disp, &c6, settings);
    if let Some(term) = &settings.three_body {
        energy = energy + three_body_g(numbers, positions, rows, &images.near, &c6, term);
    }
    energy
}

/// Geometry-only energy on lattice paths with a shared reference image list.
pub(crate) struct PeriodicDispersionPath<'a> {
    numbers: Vec<u8>,
    positions: Vec<Vec3>,
    lattice: &'a Lattice,
    images: Images,
    settings: &'a PeriodicDispersion,
}

impl<'a> PeriodicDispersionPath<'a> {
    pub(crate) fn new(cell: &'a Cell, settings: &'a PeriodicDispersion) -> Result<Self> {
        let atoms = &cell.molecule().atoms;
        let numbers: Vec<_> = atoms.iter().map(|atom| atom.z).collect();
        if settings.three_body.is_some() {
            settings.damping.validate_three_body()?;
        }
        validate_elements(&numbers, &settings.damping)?;
        let positions: Vec<_> = atoms.iter().map(|atom| atom.position).collect();
        let lattice = cell.lattice();
        let geometry = Geometry::bulk(&positions, lattice);
        let images = image_lists(&geometry, numbers.len(), settings);
        Ok(Self {
            numbers,
            positions,
            lattice,
            images,
            settings,
        })
    }

    pub(crate) fn energy(&self, path: &LatticePath, order: usize) -> Result<Vec<f64>> {
        geometry_energy_series(
            &self.positions,
            self.lattice,
            path,
            order,
            |positions, rows| energy_g(&self.numbers, positions, rows, &self.images, self.settings),
        )
    }
}

/// Differentiate the same pair and three-body energy under
/// `r = (I + eps)(r0 + u)` and `A = (I + eps) A0`. The mixed derivative
/// includes the first-order term from the changing strain lever arm.
pub(crate) fn dispersion_strain_second(
    numbers: &[u8],
    positions: &[Vec3],
    lattice: &Lattice,
    settings: &PeriodicDispersion,
) -> crate::periodic::strain::StrainSecond {
    type Inner = DualN<f64, 9>;
    let images = image_lists(&Geometry::bulk(positions, lattice), numbers.len(), settings);
    let affine = |row: [f64; 3]| -> [Inner; 3] {
        std::array::from_fn(|a| {
            (0..3).fold(Inner::constant(row[a]), |value, b| {
                value + Inner::variable(0.0, 3 * a + b).scale(row[b])
            })
        })
    };
    let rows = lattice.a.map(|row| affine(row.to_array()));
    let positions: Vec<_> = positions.iter().map(|row| affine(row.to_array())).collect();
    // Two affine seeds differentiate the clamped-ion strain energy.
    type Affine = DualN<Inner, 9>;
    let seed_strain = |row: [Inner; 3]| -> [Affine; 3] {
        std::array::from_fn(|a| {
            (0..3).fold(Affine::constant(row[a]), |value, b| {
                value + Affine::variable(Inner::constant(0.0), 3 * a + b).scale(row[b].re)
            })
        })
    };
    let strained: Vec<_> = positions.iter().copied().map(seed_strain).collect();
    let clamped = energy_g(
        numbers,
        &strained,
        &rows.map(seed_strain),
        &images,
        settings,
    );
    // Seed one atom at a time to bound the forward-mode working set.
    type Mixed = DualN<Inner, 3>;
    let rows = rows.map(|row| row.map(Mixed::constant));
    let mixed: Vec<_> = (0..positions.len())
        .into_par_iter()
        .map(|owner| {
            let moved: Vec<[Mixed; 3]> = positions
                .iter()
                .enumerate()
                .map(|(atom, row)| {
                    std::array::from_fn(|a| {
                        let mut value = Mixed::constant(row[a]);
                        if atom == owner {
                            for b in 0..3 {
                                value.im[b] = Inner::constant(f64::from(a == b))
                                    + Inner::variable(0.0, 3 * a + b);
                            }
                        }
                        value
                    })
                })
                .collect();
            energy_g(numbers, &moved, &rows, &images, settings).im
        })
        .collect();
    let mut result = crate::periodic::strain::StrainSecond::zeros(positions.len());
    for (strain, block) in clamped.im.iter().enumerate() {
        result.strain[9 * strain..9 * (strain + 1)].copy_from_slice(&block.im);
        for (atom, derivatives) in mixed.iter().enumerate() {
            for (axis, derivative) in derivatives.iter().enumerate() {
                result.mixed[3 * (strain * positions.len() + atom) + axis] = derivative.im[strain];
            }
        }
    }
    result
}

/// Force constants `d2E/du_a du_b` (Hartree/Bohr^2, `3N x 3N`) of the
/// dispersion energy per cell for rigid displacements of every atom together
/// with its images, by nested forward duals of [`energy_g`] on the image
/// lists of the given geometry (the construction of the molecular
/// dispersion Hessian). It covers every model of the lattice sum, including
/// the coordination-number dependence of the D3 coefficients.
pub fn periodic_dispersion_hessian(
    numbers: &[u8],
    positions: &[Vec3],
    lattice: &Lattice,
    settings: &PeriodicDispersion,
) -> Result<Matrix> {
    let geometry = Geometry::bulk(positions, lattice);
    // Validates the sizes and the elements.
    evaluate(numbers, &geometry, &geometry, settings)?;
    let n = numbers.len();
    let images = image_lists(&geometry, n, settings);
    type Inner = DualN<f64, 3>;
    type Outer = DualN<Inner, 3>;
    let rows: [[Outer; 3]; 3] = std::array::from_fn(|k| {
        let a = lattice.a[k].to_array();
        std::array::from_fn(|c| Outer::constant(Inner::constant(a[c])))
    });
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (a..n).map(move |b| (a, b))).collect();
    // Atom a is seeded in the outer dual and atom b in the inner one, so
    // that e.im[x].im[y] = d2E / d(a, x) d(b, y).
    let blocks: Vec<[[f64; 3]; 3]> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let seeded: Vec<[Outer; 3]> = positions
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    let p = p.to_array();
                    std::array::from_fn(|c| {
                        let inner = if k == b {
                            Inner::variable(p[c], c)
                        } else {
                            Inner::constant(p[c])
                        };
                        if k == a {
                            Outer::variable(inner, c)
                        } else {
                            Outer::constant(inner)
                        }
                    })
                })
                .collect();
            let e = energy_g(numbers, &seeded, &rows, &images, settings);
            std::array::from_fn(|x| std::array::from_fn(|y| e.im[x].im[y]))
        })
        .collect();
    let mut hessian = Matrix::zeros(3 * n, 3 * n);
    for (&(a, b), block) in pairs.iter().zip(&blocks) {
        for x in 0..3 {
            for y in 0..3 {
                hessian[(3 * a + x, 3 * b + y)] = block[x][y];
                hessian[(3 * b + y, 3 * a + x)] = block[x][y];
            }
        }
    }
    Ok(hessian)
}

/// Periodic D3 energy, forces and strain derivative of atoms `positions`
/// (Bohr) with atomic numbers `numbers` in `lattice`.
pub fn periodic_d3(
    numbers: &[u8],
    positions: &[Vec3],
    lattice: &Lattice,
    settings: &PeriodicDispersion,
) -> Result<PeriodicDispersionResult> {
    let geometry = Geometry::bulk(positions, lattice);
    evaluate(numbers, &geometry, &geometry, settings)
}

/// Geometry-only dispersion over the active lattice dimensions.
/// Energies are per primitive cell; strain derivatives use all ordered
/// Cartesian affine components before division by the active measure.
pub fn dimensional_dispersion(
    numbers: &[u8],
    positions: &[Vec3],
    geometry: &PeriodicGeometry,
    settings: &PeriodicDispersion,
) -> Result<PeriodicDispersionResult> {
    let geometry = Geometry::dimensional(positions, geometry);
    // Open-axis duals are exactly zero, so the shared image bounds admit
    // only the zero translation along an open axis.
    evaluate(numbers, &geometry, &geometry, settings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispersion::{chg_energy_g, d2_energy_g, d3_energy_g};

    const CELL: [[f64; 3]; 3] = [[9.1, 0.0, 0.0], [1.4, 8.6, 0.0], [0.9, 1.3, 9.8]];
    const NUMBERS: [u8; 3] = [6, 7, 8];
    const POSITIONS: [[f64; 3]; 3] = [[0.2, 0.1, 0.3], [2.5, 0.6, 0.2], [0.9, 2.7, 1.6]];

    fn lattice(rows: [[f64; 3]; 3]) -> Lattice {
        Lattice::new(
            Vec3::new(rows[0][0], rows[0][1], rows[0][2]),
            Vec3::new(rows[1][0], rows[1][1], rows[1][2]),
            Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
        )
    }

    fn atoms(p: &[[f64; 3]]) -> Vec<Vec3> {
        p.iter().map(|x| Vec3::new(x[0], x[1], x[2])).collect()
    }

    fn models(cn_cutoff: f64, disp_cutoff: f64) -> Vec<(&'static str, PeriodicDispersion)> {
        vec![
            (
                "D3(BJ)",
                PeriodicDispersion::from_model(&Dispersion::D3Bj, "pbe", cn_cutoff, disp_cutoff)
                    .unwrap()
                    .unwrap(),
            ),
            (
                "D3(zero)",
                PeriodicDispersion::from_model(&Dispersion::D3Zero, "pbe", cn_cutoff, disp_cutoff)
                    .unwrap()
                    .unwrap(),
            ),
            (
                "CHG",
                PeriodicDispersion::from_model(&Dispersion::Chg, "wb97xd", cn_cutoff, disp_cutoff)
                    .unwrap()
                    .unwrap(),
            ),
            (
                "D3(BJ)+ATM",
                PeriodicDispersion::from_model(&Dispersion::D3Bj, "pbe", cn_cutoff, disp_cutoff)
                    .unwrap()
                    .unwrap()
                    .with_three_body(ThreeBody {
                        cutoff: 16.0,
                        ..ThreeBody::default()
                    })
                    .unwrap(),
            ),
        ]
    }

    /// Log-log least squares `log|e| = log C + p log h`: exponent, R^2.
    fn power_fit(h: &[f64], errors: &[f64]) -> (f64, f64) {
        let x: Vec<f64> = h.iter().map(|v| v.ln()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.abs().ln()).collect();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// Energy of the same pair list (selected at the reference geometry) with
    /// atoms and cell strained by `eps` (`x -> (1 + eps) x`) and atom
    /// `moved` displaced along `axis` by `shift`.
    fn fixed_list_energy(
        settings: &PeriodicDispersion,
        eps: &Tensor,
        moved: Option<(usize, usize, f64)>,
    ) -> f64 {
        let l0 = lattice(CELL);
        let p0 = atoms(&POSITIONS);
        let strain = |v: Vec3| -> Vec3 {
            let c = v.to_array();
            let mut out = c;
            for a in 0..3 {
                for b in 0..3 {
                    out[a] += eps[a][b] * c[b];
                }
            }
            Vec3::new(out[0], out[1], out[2])
        };
        let l1 = Lattice::new(strain(l0.a[0]), strain(l0.a[1]), strain(l0.a[2]));
        let mut p1: Vec<Vec3> = p0.iter().map(|p| strain(*p)).collect();
        if let Some((atom, axis, shift)) = moved {
            let mut c = p1[atom].to_array();
            c[axis] += shift;
            p1[atom] = Vec3::new(c[0], c[1], c[2]);
        }
        let reference = Geometry::bulk(&p0, &l0);
        let current = Geometry::bulk(&p1, &l1);
        evaluate(&NUMBERS, &reference, &current, settings)
            .unwrap()
            .energy
    }

    #[test]
    fn isolated_molecule_in_a_large_box_matches_the_molecular_energy() {
        let positions = atoms(&[[0.0, 0.0, 0.0], [2.2, 0.3, 0.0], [0.5, 2.4, 0.4]]);
        let flat: Vec<[f64; 3]> = positions.iter().map(|p| p.to_array()).collect();
        let cell = lattice([[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]]);
        for (name, model) in models(25.0, 25.0) {
            let periodic = periodic_d3(&NUMBERS, &positions, &cell, &model)
                .unwrap()
                .energy;
            let pair = match model.damping {
                D3Damping::Zero(p) => d3_energy_g(&NUMBERS, &flat, Some(&p), None),
                D3Damping::Bj(p) => d3_energy_g(&NUMBERS, &flat, None, Some(&p)),
                D3Damping::D2 { s6 } => d2_energy_g(&NUMBERS, &flat, s6),
                D3Damping::Chg { a } => chg_energy_g(&NUMBERS, &flat, a),
                D3Damping::D1 { .. } => unreachable!("D1 needs explicit molecular references"),
            };
            let molecular = if model.three_body.is_some() {
                pair + crate::dispersion::d3_three_body_g(&NUMBERS, &flat)
            } else {
                pair
            };
            println!(
                "{name}: periodic {periodic:.15e}, molecular {molecular:.15e}, difference {:e}",
                periodic - molecular
            );
            assert!((periodic - molecular).abs() < 1e-12, "{name}");
        }
    }

    #[test]
    fn forces_pass_a_four_step_fit_and_sum_to_zero() {
        let steps = [0.02, 0.01, 0.005, 0.0025];
        let positions = atoms(&POSITIONS);
        let zero = [[0.0; 3]; 3];
        for (name, model) in models(DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF) {
            let result = periodic_d3(&NUMBERS, &positions, &lattice(CELL), &model).unwrap();
            let mut total = Vec3::zero();
            for f in &result.forces {
                total += *f;
            }
            println!(
                "{name}: E {:.12e}, CN {:?}, force sum {:e}",
                result.energy,
                result.coordination_numbers,
                total.norm()
            );
            assert!(total.norm() < 1e-12, "{name}");
            for (atom, axis) in [(1usize, 0usize), (2, 2)] {
                let analytic = result.forces[atom].to_array()[axis];
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        let plus = fixed_list_energy(&model, &zero, Some((atom, axis, h)));
                        let minus = fixed_list_energy(&model, &zero, Some((atom, axis, -h)));
                        -(plus - minus) / (2.0 * h) - analytic
                    })
                    .collect();
                let (p, r2) = power_fit(&steps, &errors);
                println!(
                    "{name} force atom {atom} axis {axis}: analytic {analytic:.10e}, errors {errors:?}, h {}..{}, p {p:.6}, R^2 {r2:.8}",
                    steps[3], steps[0]
                );
                assert!(
                    (p - 2.0).abs() < 0.1 && r2 > 0.9999,
                    "{name} atom {atom} axis {axis}"
                );
            }
        }
    }

    #[test]
    fn strain_derivative_passes_a_four_step_fit_and_is_symmetric() {
        let steps = [0.004, 0.002, 0.001, 0.0005];
        let positions = atoms(&POSITIONS);
        for (name, model) in models(DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF) {
            let result = periodic_d3(&NUMBERS, &positions, &lattice(CELL), &model).unwrap();
            let d = result.strain_derivative;
            for (a, b) in [(0usize, 0usize), (0, 1), (2, 1)] {
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        let mut plus = [[0.0; 3]; 3];
                        plus[a][b] = h;
                        let mut minus = [[0.0; 3]; 3];
                        minus[a][b] = -h;
                        (fixed_list_energy(&model, &plus, None)
                            - fixed_list_energy(&model, &minus, None))
                            / (2.0 * h)
                            - d[a][b]
                    })
                    .collect();
                let (p, r2) = power_fit(&steps, &errors);
                println!(
                    "{name} dE/d eps[{a}][{b}]: analytic {:.10e}, errors {errors:?}, h {}..{}, p {p:.6}, R^2 {r2:.8}",
                    d[a][b], steps[3], steps[0]
                );
                assert!((p - 2.0).abs() < 0.1 && r2 > 0.9999, "{name} [{a}][{b}]");
            }
            let scale = d.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        (d[a][b] - d[b][a]).abs() <= 1e-10 * scale,
                        "{name} [{a}][{b}]"
                    );
                }
            }
            // One atom in a cubic cell: isotropic strain derivative.
            let cubic = lattice([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]);
            let one = periodic_d3(&[6], &[Vec3::new(0.3, 0.2, 0.1)], &cubic, &model)
                .unwrap()
                .strain_derivative;
            println!("{name} cubic one-atom strain derivative {one:?}");
            for a in 0..3 {
                for b in 0..3 {
                    if a == b {
                        assert!(
                            (one[a][a] - one[0][0]).abs() <= 1e-10 * one[0][0].abs(),
                            "{name}"
                        );
                    } else {
                        assert!(one[a][b].abs() <= 1e-12 * one[0][0].abs(), "{name}");
                    }
                }
            }
        }
    }

    #[test]
    fn affine_and_internal_dispersion_curvatures_fit_every_ordered_component() {
        use crate::acceptance::{components, stencils};
        use stencils::Stencil;

        let cell = lattice(CELL);
        let positions = atoms(&POSITIONS);
        let natoms = positions.len();
        let dim = 3 * natoms;
        let reference = Geometry::bulk(&positions, &cell);
        // The analytic stress oracle rebuilds the contractions at each geometry
        // with the same reference cutoff lists as the differentiated energy.
        for (name, model) in models(23.17, 27.31) {
            let exact = dispersion_strain_second(&NUMBERS, &positions, &cell, &model);
            let first = periodic_d3(&NUMBERS, &positions, &cell, &model)
                .unwrap()
                .strain_derivative;
            let columns = 9 + dim;
            let labels: Vec<_> = (0..columns)
                .flat_map(|column| {
                    (0..9)
                        .map(move |row| format!("dispersion {name} strain/mixed [{row},{column}]"))
                })
                .collect();
            let sample = |column: usize, h: f64| {
                let mut moved = positions.clone();
                let mut lattice = cell.clone();
                if column < 9 {
                    let (a, b) = (column / 3, column % 3);
                    let deform = |v: Vec3| {
                        let mut xyz = v.to_array();
                        xyz[a] += h * xyz[b];
                        Vec3::new(xyz[0], xyz[1], xyz[2])
                    };
                    moved.iter_mut().for_each(|r| *r = deform(*r));
                    let [a, b, c] = lattice.a.map(deform);
                    lattice = Lattice::new(a, b, c);
                } else {
                    let coordinate = column - 9;
                    let mut xyz = moved[coordinate / 3].to_array();
                    xyz[coordinate % 3] += h;
                    moved[coordinate / 3] = Vec3::new(xyz[0], xyz[1], xyz[2]);
                }
                let current = Geometry::bulk(&moved, &lattice);
                evaluate(&NUMBERS, &reference, &current, &model)
                    .unwrap()
                    .strain_derivative
            };
            let expected = |row: usize, column: usize| {
                if column < 9 {
                    // The fresh stress acts on the already deformed geometry.
                    exact.strain[9 * row + column]
                        + if row % 3 == column / 3 {
                            first[row / 3][column % 3]
                        } else {
                            0.0
                        }
                } else {
                    exact.mixed[dim * row + column - 9]
                }
            };
            let mut samples = std::collections::BTreeMap::new();
            let mut failures = Vec::new();
            let mut richardson = 0.0_f64;
            stencils::fit_components_staged(
                &labels,
                &[
                    ([0.008, 0.006, 0.004, 0.002], Stencil::Second),
                    ([0.004, 0.003, 0.002, 0.001], Stencil::Second),
                    ([0.006, 0.004, 0.003, 0.002], Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; labels.len()];
                    for column in 0..columns {
                        let mut at = |h: f64| {
                            *samples
                                .entry((column, h.to_bits()))
                                .or_insert_with(|| sample(column, h))
                        };
                        for (i, &h) in steps.iter().enumerate() {
                            let (plus, minus) = (at(h), at(-h));
                            let (plus2, minus2) = if stencil == Stencil::Fourth {
                                (at(2.0 * h), at(-2.0 * h))
                            } else {
                                ([[0.0; 3]; 3], [[0.0; 3]; 3])
                            };
                            for row in 0..9 {
                                let (a, b) = (row / 3, row % 3);
                                errors[9 * column + row][i] = (stencil.first(
                                    plus[a][b],
                                    minus[a][b],
                                    plus2[a][b],
                                    minus2[a][b],
                                    h,
                                ) - expected(row, column))
                                .abs();
                            }
                        }
                        for row in 0..9 {
                            let mut derivative = |h| {
                                let (p, m) = (at(h), at(-h));
                                (p[row / 3][row % 3] - m[row / 3][row % 3]) / (2.0 * h)
                            };
                            richardson = richardson.max(
                                ((4.0 * derivative(0.002) - derivative(0.004)) / 3.0
                                    - expected(row, column))
                                .abs(),
                            );
                        }
                    }
                    errors
                },
                &mut failures,
            );
            println!("dispersion {name} max Richardson defect={richardson:.15e}");
            components::assert_no_failures(name, &failures);
            assert!(richardson < 2e-8, "{name}: {richardson:e}");
        }
    }

    #[test]
    fn generic_energy_reproduces_the_analytic_sums_and_first_derivatives() {
        let positions = atoms(&POSITIONS);
        let cell = lattice(CELL);
        let geometry = Geometry::bulk(&positions, &cell);
        let n = NUMBERS.len();
        for (name, model) in models(DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF) {
            let result = periodic_d3(&NUMBERS, &positions, &cell, &model).unwrap();
            let images = image_lists(&geometry, n, &model);
            // Plain numbers: the same energy.
            let plain: Vec<[f64; 3]> = positions.iter().map(|p| p.to_array()).collect();
            let rows: [[f64; 3]; 3] = std::array::from_fn(|k| cell.a[k].to_array());
            let energy = energy_g(&NUMBERS, &plain, &rows, &images, &model);
            // Nine strain seeds: x -> (1 + eps) x for positions and rows.
            type Strain = DualN<f64, 9>;
            let strained = |v: [f64; 3]| -> [Strain; 3] {
                std::array::from_fn(|c| {
                    let mut x = Strain::constant(v[c]);
                    for b in 0..3 {
                        x.im[3 * c + b] = v[b];
                    }
                    x
                })
            };
            let seeded: Vec<[Strain; 3]> = plain.iter().map(|p| strained(*p)).collect();
            let seeded_rows: [[Strain; 3]; 3] = std::array::from_fn(|k| strained(rows[k]));
            let jet = energy_g(&NUMBERS, &seeded, &seeded_rows, &images, &model);
            let mut strain_gap = 0.0_f64;
            for a in 0..3 {
                for b in 0..3 {
                    strain_gap =
                        strain_gap.max((jet.im[3 * a + b] - result.strain_derivative[a][b]).abs());
                }
            }
            // Three position seeds per atom: minus the forces.
            type Move = DualN<f64, 3>;
            let rows_move: [[Move; 3]; 3] =
                std::array::from_fn(|k| std::array::from_fn(|c| Move::constant(rows[k][c])));
            let mut force_gap = 0.0_f64;
            for atom in 0..n {
                let moved: Vec<[Move; 3]> = plain
                    .iter()
                    .enumerate()
                    .map(|(k, p)| {
                        std::array::from_fn(|c| {
                            if k == atom {
                                Move::variable(p[c], c)
                            } else {
                                Move::constant(p[c])
                            }
                        })
                    })
                    .collect();
                let e = energy_g(&NUMBERS, &moved, &rows_move, &images, &model);
                let f = result.forces[atom].to_array();
                for c in 0..3 {
                    force_gap = force_gap.max((e.im[c] + f[c]).abs());
                }
            }
            println!(
                "{name}: generic energy gap {:e}, force gap {force_gap:e}, strain gap {strain_gap:e}",
                energy - result.energy
            );
            assert!((energy - result.energy).abs() < 1e-14, "{name}");
            assert!(force_gap < 1e-13 && strain_gap < 1e-12, "{name}");
        }
    }

    #[test]
    fn force_constants_pass_a_four_step_fit_and_obey_the_sum_rules() {
        let steps = [0.02, 0.01, 0.005, 0.0025];
        let positions = atoms(&POSITIONS);
        let cell = lattice(CELL);
        let n = NUMBERS.len();
        for (name, model) in models(DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF) {
            let h = periodic_dispersion_hessian(&NUMBERS, &positions, &cell, &model).unwrap();
            let scale = h.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            // A rigid translation of the crystal costs nothing.
            let mut acoustic = 0.0_f64;
            for row in 0..3 * n {
                for y in 0..3 {
                    let sum: f64 = (0..n).map(|b| h[(row, 3 * b + y)]).sum();
                    acoustic = acoustic.max(sum.abs());
                }
            }
            println!("{name}: largest force constant {scale:e}, acoustic sum {acoustic:e}");
            assert!(acoustic < 1e-10 * scale, "{name}");
            // Columns against central differences of the analytic forces on
            // the image lists of the reference geometry.
            let forces_at = |atom: usize, axis: usize, shift: f64| -> Vec<f64> {
                let mut moved = positions.clone();
                let mut c = moved[atom].to_array();
                c[axis] += shift;
                moved[atom] = Vec3::new(c[0], c[1], c[2]);
                let reference = Geometry::bulk(&positions, &cell);
                let current = Geometry::bulk(&moved, &cell);
                evaluate(&NUMBERS, &reference, &current, &model)
                    .unwrap()
                    .forces
                    .iter()
                    .flat_map(|f| f.to_array())
                    .collect()
            };
            for (atom, axis) in [(1usize, 0usize), (2, 2)] {
                let column = 3 * atom + axis;
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&s| {
                        let (plus, minus) = (forces_at(atom, axis, s), forces_at(atom, axis, -s));
                        (0..3 * n)
                            .map(|row| {
                                (-(plus[row] - minus[row]) / (2.0 * s) - h[(row, column)]).abs()
                            })
                            .fold(0.0_f64, f64::max)
                    })
                    .collect();
                let (p, r2) = power_fit(&steps, &errors);
                println!(
                    "{name} force-constant column atom {atom} axis {axis}: errors {errors:?}, h {}..{}, p {p:.6}, R^2 {r2:.8}",
                    steps[3], steps[0]
                );
                assert!(
                    (p - 2.0).abs() < 0.1 && r2 > 0.9999,
                    "{name} atom {atom} axis {axis}"
                );
            }
        }
    }

    #[test]
    fn force_constants_of_a_boxed_molecule_match_the_molecular_hessian() {
        use crate::system::{Atom, Molecule};
        let raw = [[0.0, 0.0, 0.0], [2.2, 0.3, 0.0], [0.5, 2.4, 0.4]];
        let positions = atoms(&raw);
        let cell = lattice([[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]]);
        let molecule = Molecule::new(
            NUMBERS
                .iter()
                .zip(&positions)
                .map(|(z, p)| Atom {
                    z: *z,
                    position: *p,
                })
                .collect(),
        );
        for (kind, functional) in [
            (Dispersion::D3Bj, "pbe"),
            (Dispersion::D3Zero, "pbe"),
            (Dispersion::Chg, "wb97xd"),
        ] {
            let model = PeriodicDispersion::from_model(&kind, functional, 25.0, 25.0)
                .unwrap()
                .unwrap();
            let periodic =
                periodic_dispersion_hessian(&NUMBERS, &positions, &cell, &model).unwrap();
            let molecular =
                crate::dispersion::dispersion_hessian(&molecule, &kind, functional).unwrap();
            let gap = periodic.max_abs_diff(&molecular);
            let scale = molecular
                .as_slice()
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            println!("{kind:?}: largest element {scale:e}, periodic minus molecular {gap:e}");
            assert!(gap < 1e-12 * scale.max(1e-6), "{kind:?}");
        }
    }

    #[test]
    fn three_body_term_matches_the_closed_form_of_single_triangles() {
        let d = data();
        let cell = lattice([[90.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 0.0, 90.0]]);
        let two = PeriodicDispersion::from_model(&Dispersion::D3Bj, "pbe", 25.0, 30.0)
            .unwrap()
            .unwrap();
        let term = ThreeBody {
            cutoff: 30.0,
            ..ThreeBody::default()
        };
        let three = two.clone().with_three_body(term).unwrap();
        let numbers = [6u8, 7, 8];
        // An acute, an obtuse and a nearly linear triangle (Bohr).
        for (label, raw) in [
            ("acute", [[0.0, 0.0, 0.0], [7.0, 0.5, 0.0], [3.0, 6.5, 1.0]]),
            (
                "obtuse",
                [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [13.0, 2.0, 0.5]],
            ),
            (
                "linear",
                [[0.0, 0.0, 0.0], [6.5, 0.1, 0.0], [13.2, 0.0, 0.2]],
            ),
        ] {
            let positions = atoms(&raw);
            let with = periodic_d3(&numbers, &positions, &cell, &three).unwrap();
            let without = periodic_d3(&numbers, &positions, &cell, &two).unwrap();
            // Independent evaluation with explicit angles.
            let cn = &without.coordination_numbers;
            let c6 = |a: usize, b: usize| getc6_g(d, numbers[a], numbers[b], cn[a], cn[b]);
            let c9 = (c6(0, 1) * c6(0, 2) * c6(1, 2)).sqrt();
            let side = |a: usize, b: usize| (positions[b] - positions[a]).norm();
            let (rab, rac, rbc) = (side(0, 1), side(0, 2), side(1, 2));
            let cosine = |p: usize, q: usize, r: usize| {
                (positions[q] - positions[p]).dot(positions[r] - positions[p])
                    / (side(p, q) * side(p, r))
            };
            let angular = 3.0 * cosine(0, 1, 2) * cosine(1, 0, 2) * cosine(2, 0, 1) + 1.0;
            let radius = |a: usize, b: usize| {
                term.rs9 * d.r0ab[numbers[a] as usize - 1][numbers[b] as usize - 1]
            };
            let ratio = radius(0, 1) * radius(0, 2) * radius(1, 2) / (rab * rac * rbc);
            let expected = term.s9 * c9 * angular
                / (rab * rac * rbc).powi(3)
                / (1.0 + 6.0 * ratio.powf(term.alpha / 3.0));
            let found = with.energy - without.energy;
            println!(
                "{label}: three-body energy {found:.12e}, closed form {expected:.12e}, angular factor {angular:.6}"
            );
            assert!((found - expected).abs() < 1e-12 * expected.abs(), "{label}");
        }
    }

    #[test]
    fn energy_converges_monotonically_with_the_pair_cutoff() {
        let positions = atoms(&POSITIONS);
        for (name, _) in models(DEFAULT_CN_CUTOFF, DEFAULT_DISP_CUTOFF) {
            let mut previous = f64::INFINITY;
            let mut energies = Vec::new();
            for cutoff in [40.0, 60.0, 80.0, 95.0] {
                let model = models(DEFAULT_CN_CUTOFF, cutoff)
                    .into_iter()
                    .find(|(n, _)| *n == name)
                    .unwrap()
                    .1;
                let e = periodic_d3(&NUMBERS, &positions, &lattice(CELL), &model)
                    .unwrap()
                    .energy;
                assert!(e < previous, "{name} cutoff {cutoff}");
                previous = e;
                energies.push(e);
            }
            println!(
                "{name}: E(40, 60, 80, 95 bohr) = {energies:?}, E(95) - E(80) = {:e}",
                energies[3] - energies[2]
            );
            assert!((energies[3] - energies[2]).abs() < 1e-6, "{name}");
        }
    }

    #[test]
    fn unsupported_models_and_elements_are_refused() {
        assert!(PeriodicDispersion::from_model(&Dispersion::D2, "pbe", 40.0, 95.0).is_err());
        assert!(
            PeriodicDispersion::from_model(&Dispersion::None, "pbe", 40.0, 95.0)
                .unwrap()
                .is_none()
        );
        assert!(PeriodicDispersion::from_model(&Dispersion::D3Bj, "pbe", -1.0, 95.0).is_err());
        let model = models(40.0, 40.0).remove(0).1;
        assert!(
            PeriodicDispersion::from_model(&Dispersion::Chg, "wb97xd", 40.0, 95.0)
                .unwrap()
                .unwrap()
                .with_three_body(ThreeBody::default())
                .is_err()
        );
        assert!(periodic_d3(&[0], &[Vec3::zero()], &lattice(CELL), &model).is_err());
    }
}
