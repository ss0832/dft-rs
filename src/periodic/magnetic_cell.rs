// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Uniform-flux London references in physical slabs and bulk cells.
//! Finite Gamma and magnetic Brillouin-zone models are explicit choices;
//! their Green operators retain the physical electrostatic boundary.
//! No hopping phase replaces the continuum kinetic or Coulomb operators.

use super::electrostatics::{
    periodic_electrostatics, ChargeCompensation, DimensionalCoulomb, ElectrostaticOptions,
    ElectrostaticSource,
};
use super::spinor_scf::{
    from_pauli, pauli_components, spin_diagonal, SpinorEnergyBackend, SpinorFields,
};
use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::MolBasis;
use crate::dfpt::orbitals::{adjoint, matmul};
use crate::grid::gauss_legendre;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::system::Molecule;
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Mutex;

mod bloch;
mod electric;
mod nuclear;
mod nuclear_response;
mod orbitals;
mod slab;
mod xc;
pub use bloch::{
    gaussian_singularity_correction, slab_gaussian_singularity_correction, MagneticBlochBackend,
};
pub use nuclear::MagneticPairOperator;
pub use nuclear_response::MagneticNuclearTaylor;

const I: Complex = Complex::new(0.0, 1.0);
fn phase(x: f64) -> Complex {
    Complex::new(x.cos(), x.sin())
}
fn z(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn put(m: &mut BlochMatrix, i: usize, j: usize, v: Complex) {
    m.re[(i, j)] = v.re;
    m.im[(i, j)] = v.im;
}
fn gap(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
fn add(m: &mut BlochMatrix, a: &BlochMatrix, c: Complex) {
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            put(m, i, j, z(m, i, j) + z(a, i, j) * c);
        }
    }
}
fn valid(m: &BlochMatrix, n: usize) -> bool {
    m.rows() == n
        && m.cols() == n
        && m.im.rows == n
        && m.im.cols == n
        && m.re
            .as_slice()
            .iter()
            .chain(m.im.as_slice())
            .all(|x| x.is_finite())
}

/// Commuting ordered magnetic translations. Fluxes are in quantum units
/// B.(a_i cross a_j)/(2 pi); the field is not rounded to an allowed value.
#[derive(Clone, Debug)]
pub struct MagneticCell {
    pub geometry: PeriodicGeometry,
    pub field: Vec3,
    pub gauge_origin: Vec3,
    /// Cartesian magnetic Bloch wave; the ordered translation eigenvalue
    /// includes exp(i k.T), independently of the London gauge phase.
    pub bloch_wave: Vec3,
    /// Ordered faces (a1,a2), (a2,a3), (a3,a1).
    pub flux_quanta: [i64; 3],
    pub flux_defect: f64,
}
impl MagneticCell {
    pub fn new(geometry: PeriodicGeometry, field: Vec3, gauge_origin: Vec3) -> Result<Self> {
        if !(1..=3).contains(&geometry.dimension())
            || !field.norm().is_finite()
            || !gauge_origin.norm().is_finite()
        {
            return Err(DftError::InvalidInput("a magnetic cell requires independent physical periodic vectors and finite field/origin".into()));
        }
        let a = geometry.lattice().a;
        let mut flux = [
            field.dot(a[0].cross(a[1])),
            field.dot(a[1].cross(a[2])),
            field.dot(a[2].cross(a[0])),
        ]
        .map(|f| f / (2.0 * PI));
        for (i, (a, b)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
            if !geometry.boundary().axes[a] || !geometry.boundary().axes[b] {
                flux[i] = 0.0;
            }
        }
        if flux.iter().any(|f| !f.is_finite() || f.abs() > 1e12) {
            return Err(DftError::InvalidInput(
                "magnetic flux exceeds the exactly representable translation range".into(),
            ));
        }
        let defect = flux
            .iter()
            .map(|f| (f - f.round()).abs())
            .fold(0.0_f64, f64::max);
        if flux
            .iter()
            .any(|f| (f - f.round()).abs() > 1e-11 * (1.0 + f.abs()))
        {
            return Err(DftError::InvalidInput(format!(
                "uniform magnetic flux must obey B.(a_i cross a_j)/(2 pi) in Z on every \
                magnetic-cell face; measured fluxes {flux:?}; supply a flux-compatible \
                magnetic supercell"
            )));
        }
        Ok(Self {
            geometry,
            field,
            gauge_origin,
            bloch_wave: Vec3::zero(),
            flux_quanta: flux.map(|f| f.round() as i64),
            flux_defect: defect,
        })
    }
    /// Ordered generators supply the cocycle, including odd integer fluxes.
    /// T_n = exp(-i pi sum_{i<j} m_ij n_i n_j) U_(A n).
    pub fn image_phase(&self, center: Vec3, index: [i64; 3]) -> Complex {
        let m = self.flux_quanta.map(i128::from);
        let n = index.map(i128::from);
        let parity = (m[0] * n[0] * n[1] + m[1] * n[1] * n[2] - m[2] * n[0] * n[2]).rem_euclid(2);
        let translation = self.geometry.lattice().translation(index);
        phase(self.bloch_wave.dot(translation) + 0.5 * self.field.dot(center.cross(translation)))
            * if parity == 0 { 1.0 } else { -1.0 }
    }
    pub fn with_bloch_wave(&self, wave: Vec3) -> Result<Self> {
        if !wave.norm().is_finite()
            || self.geometry.perpendicular(wave).norm() > 1e-12 * (1.0 + wave.norm())
        {
            return Err(DftError::InvalidInput(
                "magnetic Bloch wave must be finite and lie in the physical periodic subspace"
                    .into(),
            ));
        }
        let mut cell = self.clone();
        cell.bloch_wave = wave;
        Ok(cell)
    }
    /// Rectangular-space vertex <phi(left)|exp(i q.r)|phi(right)>.
    /// q+right-left must be reciprocal; left and right are independent
    /// magnetic Bloch spaces. No same-q Hermitian projection is applied.
    pub fn fourier_transfer(
        &self,
        basis: &MolBasis,
        cutoff: f64,
        left: Vec3,
        right: Vec3,
        wave: Vec3,
    ) -> Result<BlochMatrix> {
        if !left.norm().is_finite() || !right.norm().is_finite() {
            return Err(DftError::InvalidInput(
                "magnetic transfer requires finite left/right Bloch waves".into(),
            ));
        }
        self.validate_wave(wave + right - left)?;
        pair_operator(&self.with_bloch_wave(right)?, basis, cutoff, wave, None)
    }
    /// London orbital wave vector. Shifting the gauge origin multiplies all
    /// orbitals by the same phase, leaving every AO operator unchanged.
    pub fn london_wave(&self, center: Vec3) -> Vec3 {
        self.field.cross(center - self.gauge_origin) * -0.5
    }
    pub fn fourier(&self, basis: &MolBasis, cutoff: f64, wave: Vec3) -> Result<BlochMatrix> {
        self.validate_wave(wave)?;
        pair_operator(self, basis, cutoff, wave, None)
    }
    pub fn kinetic(&self, basis: &MolBasis, cutoff: f64) -> Result<BlochMatrix> {
        pair_operator(self, basis, cutoff, Vec3::zero(), Some(3))
    }
    pub fn momentum(
        &self,
        basis: &MolBasis,
        cutoff: f64,
        wave: Vec3,
        axis: usize,
    ) -> Result<BlochMatrix> {
        if axis >= 3 {
            return Err(DftError::InvalidInput(
                "kinetic momentum axis must be 0..2".into(),
            ));
        }
        self.validate_wave(wave)?;
        pair_operator(self, basis, cutoff, wave, Some(axis))
    }
    fn validate_wave(&self, wave: Vec3) -> Result<()> {
        if !wave.norm().is_finite()
            || self
                .geometry
                .lattice()
                .a
                .iter()
                .enumerate()
                .any(|(axis, a)| {
                    let n = wave.dot(*a) / (2.0 * PI);
                    self.geometry.boundary().axes[axis]
                        && (n - n.round()).abs() > 1e-11 * (1.0 + n.abs())
                })
        {
            return Err(DftError::InvalidInput(
                "a single magnetic Gamma-sector Fourier operator requires reciprocal \
                integer wave indices; a general transfer belongs to separate magnetic \
                Bloch spaces"
                    .into(),
            ));
        }
        Ok(())
    }
}

fn momentum_terms(
    angular: [usize; 3],
    exponent: f64,
    field: Vec3,
    axis: usize,
) -> Vec<([usize; 3], Complex)> {
    let mut terms = Vec::new();
    let mut raised = angular;
    raised[axis] += 1;
    terms.push((raised, I * (2.0 * exponent)));
    if angular[axis] > 0 {
        let mut lowered = angular;
        lowered[axis] -= 1;
        terms.push((lowered, I * -(angular[axis] as f64)));
    }
    let x = (axis + 1) % 3;
    let y = (axis + 2) % 3;
    let mut upper = angular;
    upper[y] += 1;
    terms.push((upper, Complex::from(0.5 * field.get(x))));
    let mut upper = angular;
    upper[x] += 1;
    terms.push((upper, Complex::from(-0.5 * field.get(y))));
    terms
}
fn pair_operator(
    cell: &MagneticCell,
    basis: &MolBasis,
    cutoff: f64,
    wave: Vec3,
    kind: Option<usize>,
) -> Result<BlochMatrix> {
    let centres: Vec<_> = basis.shells.iter().map(|s| s.center.to_array()).collect();
    let [real, imaginary] =
        nuclear::scalar_pair_operator(cell, basis, cutoff, wave, kind, &centres, |x| {
            (x.cos(), x.sin())
        })?;
    Ok(BlochMatrix {
        re: crate::linalg::Matrix::from_row_major(basis.nao_sph, basis.nao_sph, real),
        im: crate::linalg::Matrix::from_row_major(basis.nao_sph, basis.nao_sph, imaginary),
    })
}

#[derive(Clone, Debug)]
pub struct MagneticScfOptions {
    /// Uniform Cartesian electric bias, Hartree/(e Bohr), on the finite
    /// 3D insulating single-point Berry branch at fixed quantized flux.
    pub electric_field: [f64; 3],
    pub basis_cutoff: f64,
    /// Production reciprocal cutoff. A separate 0.8-cutoff operator is
    /// evaluated to report the raw Coulomb/SOC convergence defect.
    pub reciprocal_cutoff: f64,
    pub nuclear_widths: Vec<f64>,
    pub nuclear_soc_scale: f64,
    pub operator_tolerance: f64,
    pub electrostatics: ElectrostaticOptions,
    /// Initial tensor Gauss-Legendre order per magnetic-cell direction.
    /// Separate adjacent orders measure raw XC energy/Fock convergence;
    /// the provider refines by eight up to 128 without renormalization.
    pub grid_order: usize,
    pub grid_cutoff: f64,
    /// Inverse-Bohr Gaussian exchange auxiliary width for crystal BZ
    /// integration. Zero selects raw punctured quadrature. The finite
    /// one-sector torus does not use a singularity correction.
    pub exchange_auxiliary_width: f64,
    /// Continuous open-normal Fourier quadrature for slab Coulomb operators.
    /// An independent half-order rule and a smaller reciprocal cutoff are kept.
    pub open_quadrature_order: usize,
}
impl Default for MagneticScfOptions {
    fn default() -> Self {
        Self {
            electric_field: [0.0; 3],
            basis_cutoff: 12.0,
            reciprocal_cutoff: 32.0,
            nuclear_widths: Vec::new(),
            nuclear_soc_scale: 1.0,
            operator_tolerance: 1e-10,
            electrostatics: ElectrostaticOptions::default(),
            grid_order: 40,
            grid_cutoff: 12.0,
            exchange_auxiliary_width: 0.5,
            open_quadrature_order: 64,
        }
    }
}
struct FourierMode {
    wave: Vec3,
    density: BlochMatrix,
    coulomb: f64,
    long_range: f64,
    nuclear: Complex,
}
/// Local density and the Pauli current decomposition. Additional supplied
/// nonlocal AO operators need their own vector-potential vertex. The electric
/// current has the opposite sign because the electron charge is -1.
pub struct MagneticCurrentSample {
    pub number_density: Vec<f64>,
    pub spin_density: Vec<[f64; 3]>,
    pub source_potential_gradient: Vec<[f64; 3]>,
    pub kinetic_current: Vec<[f64; 3]>,
    pub spin_magnetization_current: Vec<[f64; 3]>,
    pub nuclear_soc_current: Vec<[f64; 3]>,
    pub particle_current: Vec<[f64; 3]>,
}
/// Variational magnetic-torus SCF in the full London spin/AO metric.
/// The Coulomb and semilocal providers share the stationary spinor solver.
pub struct MagneticDftBackend<'a> {
    pub cell: MagneticCell,
    overlap: Vec<BlochMatrix>,
    initial: Vec<BlochMatrix>,
    kinetic: BlochMatrix,
    nuclear: BlochMatrix,
    nuclear_coarse: BlochMatrix,
    soc: BlochMatrix,
    soc_coarse: BlochMatrix,
    zeeman: BlochMatrix,
    spin: BlochMatrix,
    modes: Vec<FourierMode>,
    coarse_modes: Option<Vec<FourierMode>>,
    open_refinement_history: Vec<[f64; 3]>,
    source_energy: f64,
    sources: Vec<ElectrostaticSource>,
    source_green: DimensionalCoulomb,
    electric: Option<crate::response::berry_field::BerryField>,
    electric_coarse: Option<crate::response::berry_field::BerryField>,
    electric_background: Vec3,
    source_defect: f64,
    pub image_operator_defect: f64,
    functional: &'a ResolvedFunctional,
    basis: MolBasis,
    grid: Option<Mutex<xc::Cache>>,
    electrons: f64,
    options: MagneticScfOptions,
}
impl<'a> MagneticDftBackend<'a> {
    pub fn new(
        mol: &Molecule,
        basis: &MolBasis,
        cell: MagneticCell,
        functional: &'a ResolvedFunctional,
        options: MagneticScfOptions,
        spin_operator: Option<BlochMatrix>,
    ) -> Result<Self> {
        if options.electric_field.iter().any(|x| !x.is_finite())
            || (options.electric_field.iter().any(|x| *x != 0.0) && cell.geometry.dimension() != 3)
        {
            return Err(DftError::Unsupported {
                feature: "Berry electric bias on this magnetic boundary".into(),
                missing: "the three-dimensional finite-cell insulating electric-enthalpy provider"
                    .into(),
            });
        }
        if !basis.ecps.is_empty() || functional.required_nonlocal_term().is_some() {
            return Err(DftError::Unsupported {
                feature: "magnetic torus ECP or nonlocal XC".into(),
                missing: "their variational gauge-covariant London operators".into(),
            });
        }
        if ![
            options.basis_cutoff,
            options.reciprocal_cutoff,
            options.operator_tolerance,
            options.grid_cutoff,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
            || !options.nuclear_soc_scale.is_finite()
            || ![
                functional.hybrid.cx,
                functional.hybrid.clr,
                functional.hybrid.omega,
            ]
            .iter()
            .all(|v| v.is_finite())
            || functional.hybrid.omega < 0.0
            || !(16..=128).contains(&options.grid_order)
            || mol.is_empty()
            || basis.nao_sph == 0
            || (!options.nuclear_widths.is_empty() && options.nuclear_widths.len() != mol.len())
            || options
                .nuclear_widths
                .iter()
                .any(|w| !w.is_finite() || *w < 0.0)
        {
            return Err(DftError::InvalidInput(
                "magnetic SCF requires finite positive cutoffs/tolerances, finite SOC \
                scale, AOs and nonnegative nuclear widths matching atoms"
                    .into(),
            ));
        }
        let n = basis.nao_sph;
        if spin_operator
            .as_ref()
            .is_some_and(|m| !valid(m, 2 * n) || m.hermiticity_error() > options.operator_tolerance)
        {
            return Err(DftError::InvalidInput(
                "magnetic spin/SOC operator must be a finite Hermitian 2*AO matrix".into(),
            ));
        }
        if mol.charge != 0
            && matches!(
                options.electrostatics.compensation,
                ChargeCompensation::RequireNeutral
            )
        {
            return Err(DftError::InvalidInput(
                "a formally charged magnetic cell requires explicit compensation".into(),
            ));
        }
        let mut sources: Vec<_> = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| ElectrostaticSource {
                position: a.position,
                charge: a.z as f64,
                width: options.nuclear_widths.get(i).copied().unwrap_or(0.0),
            })
            .collect();
        if let ChargeCompensation::GaussianCountercharge { center, width } =
            options.electrostatics.compensation
        {
            if !center.norm().is_finite() || !width.is_finite() || width <= 0.0 {
                return Err(DftError::InvalidInput(
                    "Gaussian compensation requires finite center and positive width".into(),
                ));
            }
            sources.push(ElectrostaticSource {
                position: center,
                charge: -(mol.charge as f64),
                width,
            });
        }
        if cell.geometry.dimension() == 2 {
            return slab::new_backend(
                mol,
                basis,
                cell,
                functional,
                options,
                spin_operator,
                sources,
            );
        }
        if cell.geometry.dimension() != 3 {
            return Err(DftError::InvalidInput("wire uniform fields use the continuum periodic-linear-gauge dimensional SCF driver".into()));
        }
        // Nuclear/source and electronic terms all use the same 3D G=0 gauge.
        let source_options = ElectrostaticOptions {
            compensation: ChargeCompensation::UniformBackground,
            ..options.electrostatics.clone()
        };
        let source = periodic_electrostatics(&cell.geometry, &sources, &source_options)?;
        let source_green = DimensionalCoulomb::new(
            cell.geometry.clone(),
            source.params,
            source.quadrature_order,
        )?;
        let s = cell.fourier(basis, options.basis_cutoff, Vec3::zero())?;
        let kinetic = cell.kinetic(basis, options.basis_cutoff)?;
        let small_s = cell.fourier(basis, options.basis_cutoff * 0.8, Vec3::zero())?;
        let small_t = cell.kinetic(basis, options.basis_cutoff * 0.8)?;
        let image_operator_defect = gap(&s, &small_s).max(gap(&kinetic, &small_t));
        let waves = cell.geometry.reciprocal_within(options.reciprocal_cutoff)?;
        let modes = waves
            .par_iter()
            .map(|&g| {
                let density = cell.fourier(basis, options.basis_cutoff, g)?;
                let coulomb = 4.0 * PI / (cell.geometry.measure() * g.norm2());
                let nuclear = sources
                    .iter()
                    .map(|s| {
                        phase(-g.dot(s.position))
                            * (-coulomb * s.charge * (-0.25 * g.norm2() * s.width * s.width).exp())
                    })
                    .fold(Complex::ZERO, |a, b| a + b);
                let long_range = if functional.hybrid.clr == 0.0 || functional.hybrid.omega == 0.0 {
                    0.0
                } else {
                    coulomb * (-g.norm2() / (4.0 * functional.hybrid.omega.powi(2))).exp()
                };
                let momenta = if options.nuclear_soc_scale != 0.0 {
                    Some([
                        cell.momentum(basis, options.basis_cutoff, g, 0)?,
                        cell.momentum(basis, options.basis_cutoff, g, 1)?,
                        cell.momentum(basis, options.basis_cutoff, g, 2)?,
                    ])
                } else {
                    None
                };
                Ok((
                    FourierMode {
                        wave: g,
                        density,
                        coulomb,
                        long_range,
                        nuclear,
                    },
                    momenta,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut nuclear = BlochMatrix::zeros(n, n);
        let mut nuclear_coarse = nuclear.clone();
        let mut pauli = std::array::from_fn::<_, 4, _>(|_| BlochMatrix::zeros(n, n));
        let mut coarse_pauli = pauli.clone();
        let c = 137.035_999_084_f64;
        for (mode, momenta) in &modes {
            let coarse = mode.wave.norm() <= 0.8 * options.reciprocal_cutoff;
            add(&mut nuclear, &mode.density, mode.nuclear);
            if coarse {
                add(&mut nuclear_coarse, &mode.density, mode.nuclear);
            }
            if let Some(p) = momenta {
                for a in 0..3 {
                    let x = (a + 1) % 3;
                    let y = (a + 2) % 3;
                    let coefficient =
                        I * mode.nuclear * (options.nuclear_soc_scale / (2.0 * c * c));
                    add(&mut pauli[a + 1], &p[y], coefficient * mode.wave.get(x));
                    add(&mut pauli[a + 1], &p[x], coefficient * -mode.wave.get(y));
                    if coarse {
                        add(
                            &mut coarse_pauli[a + 1],
                            &p[y],
                            coefficient * mode.wave.get(x),
                        );
                        add(
                            &mut coarse_pauli[a + 1],
                            &p[x],
                            coefficient * -mode.wave.get(y),
                        );
                    }
                }
            }
        }
        let soc = from_pauli(&pauli)?;
        let soc_coarse = from_pauli(&coarse_pauli)?;
        let zeeman = from_pauli(&std::array::from_fn(|a| {
            let mut out = s.clone();
            out.scale(if a == 0 { 0.0 } else { cell.field.get(a - 1) });
            out
        }))?;
        let mut spin = spin_operator.unwrap_or_else(|| BlochMatrix::zeros(2 * n, 2 * n));
        spin.add_scaled(&zeeman, 1.0);
        let mut h = kinetic.clone();
        h.add_scaled(&nuclear, 1.0);
        let mut initial = spin_diagonal(&h);
        initial.add_scaled(&spin, 1.0);
        initial.add_scaled(&soc, 1.0);
        for matrix in [&s, &kinetic, &nuclear, &initial] {
            if !valid(matrix, matrix.rows())
                || matrix.hermiticity_error() > options.operator_tolerance
            {
                return Err(DftError::InvalidInput(format!(
                    "magnetic operator violates finite Hermiticity before projection: {:.6e}",
                    matrix.hermiticity_error()
                )));
            }
        }
        let grid = xc::Cache::new(functional, options.grid_order, mol);
        let electrons = mol.atoms.iter().map(|a| a.z as f64).sum::<f64>() - mol.charge as f64;
        let electric_background = if matches!(
            options.electrostatics.compensation,
            ChargeCompensation::UniformBackground
        ) {
            cell.geometry
                .lattice()
                .a
                .iter()
                .copied()
                .fold(Vec3::zero(), |r, a| r + a)
                * (-0.5 * mol.charge as f64)
        } else {
            Vec3::zero()
        };
        let electric = electric::geometry(
            &cell,
            basis,
            options.basis_cutoff,
            &sources,
            electric_background,
            options.electric_field,
            electrons,
            &vec![[0.0; 3]; mol.len()],
            0,
        )?;
        let electric_coarse = electric::geometry(
            &cell,
            basis,
            options.basis_cutoff * 0.8,
            &sources,
            electric_background,
            options.electric_field,
            electrons,
            &vec![[0.0; 3]; mol.len()],
            0,
        )?;
        Ok(Self {
            cell,
            overlap: vec![spin_diagonal(&s)],
            initial: vec![initial],
            kinetic,
            nuclear,
            nuclear_coarse,
            soc,
            soc_coarse,
            zeeman,
            spin,
            modes: modes.into_iter().map(|(m, _)| m).collect(),
            coarse_modes: None,
            open_refinement_history: Vec::new(),
            source_energy: source.energy,
            sources,
            source_green,
            electric,
            electric_coarse,
            electric_background,
            source_defect: source
                .quadrature_energy_defect
                .max(source.quadrature_force_defect),
            image_operator_defect,
            functional,
            basis: basis.clone(),
            grid,
            electrons,
            options,
        })
    }
    pub fn nuclear_operator(&self) -> &BlochMatrix {
        &self.nuclear
    }
    pub fn kinetic_operator(&self) -> &BlochMatrix {
        &self.kinetic
    }
    pub fn soc_operator(&self) -> &BlochMatrix {
        &self.soc
    }
    /// Replace the additional Hermitian London AO operator at fixed B.
    /// Rebuild the one-body operator from its components to avoid cumulative
    /// subtraction error when traversing a perturbation interval.
    pub fn set_external_spin_operator(&mut self, operator: BlochMatrix) -> Result<()> {
        let n = self.kinetic.rows();
        if !valid(&operator, 2 * n)
            || operator.hermiticity_error() > self.options.operator_tolerance
        {
            return Err(DftError::InvalidInput(
                "the additional magnetic spin/SOC operator must be a finite Hermitian 2*AO matrix"
                    .into(),
            ));
        }
        self.spin = self.zeeman.clone();
        self.spin.add_scaled(&operator, 1.0);
        let mut scalar = self.kinetic.clone();
        scalar.add_scaled(&self.nuclear, 1.0);
        self.initial[0] = spin_diagonal(&scalar);
        self.initial[0].add_scaled(&self.spin, 1.0);
        self.initial[0].add_scaled(&self.soc, 1.0);
        Ok(())
    }
    /// Local Pauli current: kinetic Re sum_ab P_ab phi_b* (p+A) phi_a,
    /// one-half curl of the spin density, and nuclear SOC anomalous velocity
    /// m cross grad(V)/(4 c^2). London phase derivatives are analytic. The
    /// source gradient uses the same Ewald Green function, avoiding a hard
    /// reciprocal truncation of a pointwise singular nuclear gradient.
    /// Samples on point nuclei are undefined and rejected by that Green function.
    pub fn sample_particle_current(
        &self,
        density: &BlochMatrix,
        points: &[Vec3],
    ) -> Result<MagneticCurrentSample> {
        let n = self.kinetic.rows();
        if !valid(density, 2 * n) || density.hermiticity_error() > self.options.operator_tolerance {
            return Err(DftError::InvalidInput(
                "magnetic current requires a finite Hermitian full spinor density".into(),
            ));
        }
        let cutoff = self.options.basis_cutoff.max(self.options.grid_cutoff);
        let (phi, momenta) = self
            .cell
            .orbital_values_with_momentum(&self.basis, points, cutoff)?;
        let pauli = pauli_components(density)?;
        let mut number_density = Vec::with_capacity(points.len());
        let mut spin_density = Vec::with_capacity(points.len());
        let mut source_potential_gradient = Vec::with_capacity(points.len());
        let mut kinetic_current = Vec::with_capacity(points.len());
        let mut spin_magnetization_current = Vec::with_capacity(points.len());
        let mut nuclear_soc_current = Vec::with_capacity(points.len());
        let mut particle_current = Vec::with_capacity(points.len());
        for g in 0..points.len() {
            let mut local = [0.0; 4];
            let mut current = [0.0; 3];
            let mut spin_gradient = [[0.0; 3]; 3];
            for a in 0..n {
                for b in 0..n {
                    for component in 0..4 {
                        let coefficient = z(&pauli[component], a, b) * z(&phi, g, b).conj();
                        local[component] += (coefficient * z(&phi, g, a)).re;
                        for axis in 0..3 {
                            let value = coefficient * z(&momenta[axis], g, a);
                            if component == 0 {
                                current[axis] += value.re;
                            } else {
                                spin_gradient[component - 1][axis] -= 2.0 * value.im;
                            }
                        }
                    }
                }
            }
            let spin_curl = std::array::from_fn::<_, 3, _>(|axis| {
                let x = (axis + 1) % 3;
                let y = (axis + 2) % 3;
                0.5 * (spin_gradient[y][x] - spin_gradient[x][y])
            });
            let mut gradient = Vec3::zero();
            for source in &self.sources {
                gradient -= self
                    .source_green
                    .pair(points[g] - source.position, source.width, false)?
                    .1
                    * source.charge;
            }
            let spin = Vec3::new(local[1], local[2], local[3]);
            let anomalous = spin.cross(gradient)
                * (self.options.nuclear_soc_scale / (4.0 * 137.035_999_084_f64.powi(2)));
            number_density.push(local[0]);
            spin_density.push([spin.x, spin.y, spin.z]);
            source_potential_gradient.push([gradient.x, gradient.y, gradient.z]);
            kinetic_current.push(current);
            spin_magnetization_current.push(spin_curl);
            nuclear_soc_current.push([anomalous.x, anomalous.y, anomalous.z]);
            particle_current.push(std::array::from_fn(|axis| {
                current[axis] + spin_curl[axis] + anomalous.get(axis)
            }));
        }
        Ok(MagneticCurrentSample {
            number_density,
            spin_density,
            source_potential_gradient,
            kinetic_current,
            spin_magnetization_current,
            nuclear_soc_current,
            particle_current,
        })
    }
    pub fn reciprocal_modes(&self) -> usize {
        self.modes.len()
    }
    pub fn open_quadrature_order(&self) -> usize {
        if self.cell.geometry.dimension() == 2 {
            self.options.open_quadrature_order
        } else {
            0
        }
    }
    /// All measured [fine order, nuclear operator, nuclear SOC] defects,
    /// including failed open-normal quadrature refinements.
    pub fn open_refinement_history(&self) -> &[[f64; 3]] {
        &self.open_refinement_history
    }
    pub fn grid_order(&self) -> usize {
        self.grid.as_ref().map_or(0, |grid| {
            grid.lock().expect("magnetic grid cache poisoned").order
        })
    }
    /// Raw quadrature measurements, in evaluation order. No failed interval
    /// is removed; repeated orders refer to different evaluated densities.
    pub fn grid_refinement_history(&self) -> Vec<[f64; 4]> {
        self.grid.as_ref().map_or_else(Vec::new, |grid| {
            grid.lock()
                .expect("magnetic grid cache poisoned")
                .measurements
                .clone()
        })
    }
    /// Independent quadrature defects [energy, potential, number]. Number
    /// is compared to the analytic London overlap, without renormalization.
    pub fn grid_diagnostics(&self, density: &BlochMatrix) -> Result<[f64; 3]> {
        if let Some(grid) = &self.grid {
            if let Some(spatial) = grid
                .lock()
                .map_err(|_| DftError::InvalidInput("magnetic grid cache poisoned".into()))?
                .retained_spatial(std::slice::from_ref(density))
            {
                return Ok(spatial);
            }
        }
        Ok(self.local_density(density)?.spatial)
    }
    fn local_density(&self, density: &BlochMatrix) -> Result<xc::Evaluation> {
        let Some(grid) = &self.grid else {
            return Ok(xc::Evaluation::default());
        };
        grid.lock()
            .map_err(|_| DftError::InvalidInput("magnetic grid cache poisoned".into()))?
            .evaluate(
                &[&self.cell],
                &self.basis,
                self.functional,
                &self.options,
                std::slice::from_ref(density),
                &self.overlap,
            )
    }
    fn interactions(
        &self,
        p: &BlochMatrix,
        coarse: bool,
    ) -> Result<(BlochMatrix, BlochMatrix, BlochMatrix)> {
        let n = p.rows() / 2;
        let charge = pauli_components(p)?[0].clone();
        let mut j = BlochMatrix::zeros(n, n);
        let mut k = BlochMatrix::zeros(2 * n, 2 * n);
        let mut lr = k.clone();
        let modes = if coarse {
            self.coarse_modes.as_ref().unwrap_or(&self.modes)
        } else {
            &self.modes
        };
        for mode in modes {
            if coarse
                && self.coarse_modes.is_none()
                && mode.wave.norm() > 0.8 * self.options.reciprocal_cutoff
            {
                continue;
            }
            let rho = (0..n).fold(Complex::ZERO, |s, a| {
                s + (0..n).fold(Complex::ZERO, |s, b| {
                    s + z(&charge, a, b) * z(&mode.density, a, b).conj()
                })
            });
            add(&mut j, &mode.density, rho * mode.coulomb);
            let r = spin_diagonal(&mode.density);
            let v = matmul(&matmul(&r, p)?, &adjoint(&r)?)?;
            k.add_scaled(&v, mode.coulomb);
            lr.add_scaled(&v, mode.long_range);
        }
        Ok((spin_diagonal(&j), k, lr))
    }
}
impl SpinorEnergyBackend for MagneticDftBackend<'_> {
    fn validate_occupation(&self, occupation: super::occupations::OccupationScheme) -> Result<()> {
        if self.electric.is_some() && occupation != super::occupations::OccupationScheme::Insulator
        {
            return Err(DftError::Unsupported {
                feature: "Berry electric enthalpy with fractional thermal occupations".into(),
                missing: "an integer occupied insulating magnetic subspace".into(),
            });
        }
        Ok(())
    }
    fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        self.electrons
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let n = self.kinetic.rows();
        if density.len() != 1
            || !valid(&density[0], 2 * n)
            || density[0].hermiticity_error() > self.options.operator_tolerance
        {
            return Err(DftError::InvalidInput(
                "provide one finite Hermitian full spinor density for the magnetic Gamma sector"
                    .into(),
            ));
        }
        let p = &density[0];
        let (j, k, lr) = self.interactions(p, false)?;
        let (jc, kc, lrc) = self.interactions(p, true)?;
        let mut f = self.initial[0].clone();
        f.add_scaled(&j, 1.0);
        f.add_scaled(&k, -self.functional.hybrid.cx);
        let clr = self.functional.hybrid.clr;
        f.add_scaled(&lr, -clr);
        let mut old = self.kinetic.clone();
        old.add_scaled(&self.nuclear_coarse, 1.0);
        let mut old = spin_diagonal(&old);
        old.add_scaled(&self.spin, 1.0);
        old.add_scaled(&self.soc_coarse, 1.0);
        old.add_scaled(&jc, 1.0);
        old.add_scaled(&kc, -self.functional.hybrid.cx);
        old.add_scaled(&lrc, -clr);
        // Check terms separately so one-electron J/K cancellation cannot
        // hide reciprocal truncation of the physical operators.
        let mut defect = gap(&f, &old)
            .max(self.source_defect)
            .max(gap(&self.nuclear, &self.nuclear_coarse))
            .max(gap(&self.soc, &self.soc_coarse))
            .max(gap(&j, &jc))
            .max(gap(&k, &kc) * self.functional.hybrid.cx.abs())
            .max(gap(&lr, &lrc) * clr.abs());
        let charge = pauli_components(p)?[0].clone();
        let mut components = [0.0; 10];
        components[0] = self.source_energy;
        components[1] = charge.trace_product(&self.kinetic);
        components[2] = charge.trace_product(&self.nuclear);
        components[4] = 0.5 * p.trace_product(&j);
        components[5] = -0.5 * self.functional.hybrid.cx * p.trace_product(&k);
        components[6] = -0.5 * clr * p.trace_product(&lr);
        let nuclear_soc_energy = p.trace_product(&self.soc);
        components[9] = p.trace_product(&self.spin) + nuclear_soc_energy;
        let xc = self.local_density(p)?;
        components[7] = xc.energy;
        if let Some(electric) = &self.electric {
            let (energy, potential) =
                electric.evaluate(&[vec![(p.re.clone(), p.im.clone())]], 0)?;
            components[8] = energy[0];
            if let Some(coarse) = &self.electric_coarse {
                let (coarse_energy, coarse_potential) =
                    coarse.evaluate(&[vec![(p.re.clone(), p.im.clone())]], 0)?;
                defect = defect
                    .max((energy[0] - coarse_energy[0]).abs())
                    .max(potential[0][0].0.max_abs_diff(&coarse_potential[0][0].0))
                    .max(potential[0][0].1.max_abs_diff(&coarse_potential[0][0].1));
            }
            f.add_scaled(
                &BlochMatrix {
                    re: potential[0][0].0.clone(),
                    im: potential[0][0].1.clone(),
                },
                1.0,
            );
        }
        if let Some(v) = xc.potential.first() {
            f.add_scaled(v, 1.0);
        }
        if f.hermiticity_error() > self.options.operator_tolerance || !valid(&f, 2 * n) {
            return Err(DftError::InvalidInput(
                "magnetic variational Fock violates finite Hermiticity before projection".into(),
            ));
        }
        Ok(SpinorFields {
            fock: vec![f],
            energy: components.iter().sum(),
            energy_components: components,
            xc_component_energies: xc.parts,
            nuclear_soc_energy,
            operator_converged: defect <= self.options.operator_tolerance
                && self.image_operator_defect <= self.options.operator_tolerance
                && xc.angular.as_ref().is_none_or(|d| d.converged)
                && xc
                    .spatial
                    .iter()
                    .all(|d| *d <= self.options.operator_tolerance),
            coulomb_operator_defect: defect,
            xc_grid_number_defect: xc.spatial[2],
            xc_diagnostics: xc.angular,
        })
    }
}
