// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Variational complex two-component Bloch references. Spin populations are
//! not constrained separately: one chemical potential fixes the total number.
//! SOC is a general Hermitian spinor operator; no collinear projection is used.

use super::magnetic_scf::StaticPeriodicField;
use super::occupations::{solve_occupations, solve_occupations_during_scf, OccupationScheme};
use super::scf_boundary::{DimensionalScfBackend, DimensionalScfOptions};
use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::MolBasis;
use crate::dfpt::orbitals::{generalized_orbitals, matmul, BandOrbitals};
use crate::integrals::CoulombKernel;
use crate::linalg::Orthogonalizer;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::scf::diis::Diis;
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use crate::{DftError, Result};
const I: Complex = Complex::new(0.0, 1.0);

pub mod acceptance;
mod kernel;
pub(crate) mod minimize;
mod orbital_current;
pub use orbital_current::SpinorOrbitalSusceptibilityPoint;

fn z(a: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(a.re[(i, j)], a.im[(i, j)])
}
fn put(a: &mut BlochMatrix, i: usize, j: usize, v: Complex) {
    a.re[(i, j)] = v.re;
    a.im[(i, j)] = v.im;
}
fn valid(a: &BlochMatrix, n: usize) -> bool {
    a.rows() == n
        && a.cols() == n
        && a.im.rows == n
        && a.im.cols == n
        && a.re
            .as_slice()
            .iter()
            .chain(a.im.as_slice())
            .all(|x| x.is_finite())
        && a.hermiticity_error() < 1e-10
}
fn defect(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
/// Pauli decomposition P=(P0 I+Px sigma_x+Py sigma_y+Pz sigma_z)/2.
/// All four orbital matrices are Hermitian, including for general P_ab.
pub fn pauli_components(p: &BlochMatrix) -> Result<[BlochMatrix; 4]> {
    let n = p.rows() / 2;
    if n == 0 || p.rows() != 2 * n || !valid(p, 2 * n) {
        return Err(DftError::InvalidInput(
            "spinor density must be a finite even-dimensional Hermitian matrix".into(),
        ));
    }
    let mut out = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
    for i in 0..n {
        for j in 0..n {
            let (a, b, c, d) = (
                z(p, i, j),
                z(p, i, n + j),
                z(p, n + i, j),
                z(p, n + i, n + j),
            );
            for (v, m) in [a + d, b + c, I * (b - c), a - d].into_iter().zip(&mut out) {
                put(m, i, j, v);
            }
        }
    }
    Ok(out)
}
/// Inverse Pauli decomposition, with spin-major [up AOs, down AOs] ordering.
pub fn from_pauli(p: &[BlochMatrix; 4]) -> Result<BlochMatrix> {
    let n = p[0].rows();
    if n == 0 || p.iter().any(|a| !valid(a, n)) {
        return Err(DftError::InvalidInput(
            "Pauli components must be finite Hermitian orbital matrices of equal shape".into(),
        ));
    }
    let mut out = BlochMatrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let (a, x, y, v) = (
                z(&p[0], i, j),
                z(&p[1], i, j),
                z(&p[2], i, j),
                z(&p[3], i, j),
            );
            put(&mut out, i, j, (a + v) * 0.5);
            put(&mut out, n + i, n + j, (a - v) * 0.5);
            put(&mut out, i, n + j, (x - I * y) * 0.5);
            put(&mut out, n + i, j, (x + I * y) * 0.5);
        }
    }
    Ok(out)
}
pub fn spin_diagonal(a: &BlochMatrix) -> BlochMatrix {
    let n = a.rows();
    let mut out = BlochMatrix::zeros(2 * n, 2 * n);
    for s in 0..2 {
        for i in 0..n {
            for j in 0..n {
                put(&mut out, s * n + i, s * n + j, z(a, i, j));
            }
        }
    }
    out
}

pub struct SpinorFields {
    pub fock: Vec<BlochMatrix>,
    pub energy: f64,
    /// Source, kinetic, electron-nuclear, ECP, Hartree, full exchange,
    /// long-range exchange, semilocal XC, orbital field, spin/SOC operator.
    /// Slot 7 is the integrated XC energy, including density-only nonlocal XC.
    pub energy_components: [f64; 10],
    pub xc_component_energies: Vec<f64>,
    pub nuclear_soc_energy: f64,
    pub operator_converged: bool,
    pub coulomb_operator_defect: f64,
    pub xc_grid_number_defect: f64,
    pub xc_diagnostics: Option<super::spinor_xc::SpinorXcDiagnostics>,
}

impl SpinorFields {
    pub(crate) fn periodic_vv10_energy(&self, enabled: bool) -> Option<f64> {
        enabled.then(|| {
            *self
                .xc_component_energies
                .last()
                .expect("periodic VV10 is the last XC component")
        })
    }
}

/// An energy provider supplies the same variational Fock used by SCF.
/// The solver is independent of the registered functional name and family.
/// Analytic Fock differential at one immutable spinor density.
pub trait SpinorDensityKernel {
    fn apply(&self, direction: &[BlochMatrix]) -> Result<Vec<BlochMatrix>>;
}

pub trait SpinorEnergyBackend {
    fn overlap(&self) -> &[BlochMatrix];
    fn initial_fock(&self) -> &[BlochMatrix];
    fn electrons(&self) -> f64;
    /// Validate statistical occupations against the provider's energy model.
    fn validate_occupation(&self, _occupation: OccupationScheme) -> Result<()> {
        Ok(())
    }
    /// Whether the energy is exactly quadratic in the complete density.
    /// This enables cancellation-resistant energy work in SCF line searches.
    /// Providers with nonlinear terms retain the actual energy difference.
    fn quadratic_density_energy(&self) -> bool {
        false
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields>;
    /// Providers opt in only with the complete analytic energy Hessian.
    fn orbital_kernel<'a>(
        &'a self,
        _density: &'a [BlochMatrix],
    ) -> Result<Box<dyn SpinorDensityKernel + 'a>> {
        Err(DftError::Unsupported {
            feature: "spinor trust-region augmented-Hessian SCF".into(),
            missing: "the complete analytic density kernel of this backend".into(),
        })
    }
}

/// Physical dimensional Coulomb and full/range exchange, plus rotationally
/// invariant local spin-density XC and generic globally projected GGA/meta-GGA.
/// The latter retains all gradients and gauge-corrected kinetic/current terms.
pub struct SpinorDftBackend<'a> {
    scalar: DimensionalScfBackend<'a>,
    functional: &'a ResolvedFunctional,
    overlap: Vec<BlochMatrix>,
    initial: Vec<BlochMatrix>,
    spin_operator: Vec<BlochMatrix>,
    nuclear_soc: Vec<BlochMatrix>,
    nuclear_soc_diagnostics: [f64; 2],
    uniform_zeeman_derivative: Vec<BlochMatrix>,
    uniform_soc_derivative: Vec<BlochMatrix>,
    electrons: f64,
    xc_momentum: Vec<[BlochMatrix; 3]>,
}
impl<'a> SpinorDftBackend<'a> {
    pub fn new(
        mol: &'a Molecule,
        basis: &'a MolBasis,
        geometry: &'a PeriodicGeometry,
        functional: &'a ResolvedFunctional,
        options: &'a DimensionalScfOptions,
        field: Option<&StaticPeriodicField>,
        spin_operator: Vec<BlochMatrix>,
    ) -> Result<Self> {
        let mut scalar =
            DimensionalScfBackend::new_spinor(mol, basis, geometry, functional, options)?
                .with_complex_density();
        if let Some(field) = field {
            scalar = scalar.with_static_field(field)?;
        }
        let n = basis.nao_sph;
        if spin_operator.len() != scalar.k_points().len()
            || spin_operator.iter().any(|h| !valid(h, 2 * n))
        {
            return Err(DftError::InvalidInput("spin/SOC operators must be finite Hermitian [k][2*AO][2*AO] matrices on the SCF mesh".into()));
        }
        let overlap = scalar.overlap().iter().map(spin_diagonal).collect();
        let initial = scalar
            .h_core()
            .iter()
            .zip(&spin_operator)
            .map(|(h, v)| {
                let mut a = spin_diagonal(h);
                a.add_scaled(v, 1.0);
                a
            })
            .collect();
        let electrons = basis.effective_charges(mol).iter().sum::<f64>() - mol.charge as f64;
        let nuclear_soc = vec![BlochMatrix::zeros(2 * n, 2 * n); spin_operator.len()];
        let uniform_zeeman_derivative = nuclear_soc.clone();
        let uniform_soc_derivative = nuclear_soc.clone();
        let xc_momentum = if !functional.parts.is_empty() && functional.family != Family::Lda {
            scalar.xc_momentum()?
        } else {
            Vec::new()
        };
        Ok(Self {
            scalar,
            functional,
            overlap,
            initial,
            spin_operator,
            nuclear_soc,
            nuclear_soc_diagnostics: [0.0; 2],
            uniform_zeeman_derivative,
            uniform_soc_derivative,
            electrons,
            xc_momentum,
        })
    }
    /// Attach a continuum uniform open-boundary field and sigma.B/2.
    /// Origin changes use a common AO unitary phase, preserving all scalar
    /// Coulomb/exchange matrix elements and the physical kinetic momentum.
    pub fn with_uniform_open_field(
        mut self,
        field: &super::uniform_open::UniformOpenField,
    ) -> Result<Self> {
        if self.nuclear_soc.iter().any(|h| {
            h.re.as_slice()
                .iter()
                .chain(h.im.as_slice())
                .any(|v| *v != 0.0)
        }) {
            return Err(DftError::InvalidInput("attach the uniform field before nuclear SOC so the kinetic-momentum contact is included".into()));
        }
        let old = self.scalar.h_core().to_vec();
        self.scalar = self.scalar.with_uniform_open_field(field)?;
        if !self.xc_momentum.is_empty() {
            self.xc_momentum = self.scalar.xc_momentum()?;
        }
        for ik in 0..self.overlap.len() {
            let mut difference = self.scalar.h_core()[ik].clone();
            difference.add_scaled(&old[ik], -1.0);
            self.initial[ik].add_scaled(&spin_diagonal(&difference), 1.0);
            let s = &self.scalar.overlap()[ik];
            let n = s.rows();
            let mut pauli = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
            for a in 0..3 {
                pauli[a + 1] = s.clone();
                pauli[a + 1].scale(field.magnetic_field.get(a));
            }
            self.uniform_zeeman_derivative[ik] = from_pauli(&pauli)?;
            self.spin_operator[ik].add_scaled(&self.uniform_zeeman_derivative[ik], field.amplitude);
            self.initial[ik].add_scaled(&self.uniform_zeeman_derivative[ik], field.amplitude);
        }
        Ok(self)
    }
    /// Add the full one-electron Pauli nuclear SOC using the dimensional
    /// point/finite-nucleus Green operator. This term does not scale with
    /// the prescribed external field amplitude. Two-electron SOC and scalar
    /// relativistic/Dirac terms are separate models.
    pub fn with_nuclear_soc(mut self, scale: f64) -> Result<Self> {
        if !scale.is_finite() {
            return Err(DftError::InvalidInput(
                "nuclear SOC scale must be finite".into(),
            ));
        }
        self.nuclear_soc_diagnostics = [0.0; 2];
        for h in &mut self.uniform_soc_derivative {
            *h = BlochMatrix::zeros(h.rows(), h.cols());
        }
        for (initial, old) in self.initial.iter_mut().zip(&mut self.nuclear_soc) {
            initial.add_scaled(old, -1.0);
            *old = BlochMatrix::zeros(initial.rows(), initial.cols());
        }
        if scale != 0.0 {
            if self.scalar.has_vector_field() && self.scalar.uniform_field().is_none() {
                return Err(DftError::Unsupported {feature:"Pauli nuclear SOC with orbital vector potential".into(),
                    missing:"the nuclear grad(V) cross A contact and its variational amplitude derivative".into()});
            }
            let report = self.scalar.coulomb_backend().nuclear_spin_orbit_report()?;
            self.nuclear_soc_diagnostics = [
                report.quadrature_defect * scale.abs(),
                report.hermiticity_defect * scale.abs(),
            ];
            let orbital = report.operators;
            let contact = self
                .scalar
                .uniform_gauge_matrix()?
                .map(|matrix| {
                    self.scalar
                        .coulomb_backend()
                        .nuclear_uniform_contact_report(matrix)
                })
                .transpose()?;
            if let Some(c) = &contact {
                let amplitude = self.scalar.uniform_field().unwrap().amplitude;
                self.nuclear_soc_diagnostics[0] +=
                    c.quadrature_defect * scale.abs() * amplitude.abs();
                self.nuclear_soc_diagnostics[1] +=
                    c.hermiticity_defect * scale.abs() * amplitude.abs();
            }
            for ik in 0..self.overlap.len() {
                let n = orbital[0][ik].rows();
                let mut p = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
                for a in 0..3 {
                    p[a + 1] = orbital[a][ik].clone();
                    p[a + 1].scale(2.0 * scale);
                }
                self.nuclear_soc[ik] = from_pauli(&p)?;
                if let Some(c) = &contact {
                    let mut pc = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
                    for a in 0..3 {
                        pc[a + 1] = c.operators[a][ik].clone();
                        pc[a + 1].scale(2.0 * scale);
                    }
                    self.uniform_soc_derivative[ik] = from_pauli(&pc)?;
                    self.nuclear_soc[ik].add_scaled(
                        &self.uniform_soc_derivative[ik],
                        self.scalar.uniform_field().unwrap().amplitude,
                    );
                }
                self.initial[ik].add_scaled(&self.nuclear_soc[ik], 1.0);
            }
        }
        Ok(self)
    }
    pub fn k_points(&self) -> &[Vec3] {
        self.scalar.k_points()
    }
    pub fn nuclear_soc_diagnostics(&self) -> [f64; 2] {
        self.nuclear_soc_diagnostics
    }
    pub fn amplitude_derivative(
        &self,
        density: &[BlochMatrix],
        orbital: &[BlochMatrix],
        spin: &[BlochMatrix],
    ) -> Result<f64> {
        if density.len() != self.overlap.len()
            || orbital.len() != density.len()
            || spin.len() != density.len()
        {
            return Err(DftError::InvalidInput(
                "spinor amplitude derivative requires matching k meshes".into(),
            ));
        }
        let mut out = 0.0;
        for (ik, ((p, h), v)) in density.iter().zip(orbital).zip(spin).enumerate() {
            if !valid(p, self.overlap[0].rows()) || !valid(h, p.rows() / 2) || !valid(v, p.rows()) {
                return Err(DftError::InvalidInput("spinor amplitude derivative requires finite Hermitian operators with matching AO shapes".into()));
            }
            out += p.trace_product(&spin_diagonal(h)) + p.trace_product(v);
            out += p.trace_product(&self.uniform_zeeman_derivative[ik])
                + p.trace_product(&self.uniform_soc_derivative[ik]);
        }
        Ok(out / density.len() as f64)
    }
    /// Analytic nuclear gradient of this same field-free all-electron model.
    /// Complex spinors are retained; no constrained collinear parent is used.
    pub fn nuclear_gradient(
        &self,
        reference: &SpinorScfReport,
        config: &SpinorScfConfig,
    ) -> Result<super::spinor_gradient::SpinorNuclearGradient> {
        if self.spin_operator.iter().chain(&self.nuclear_soc).any(|m| {
            m.re.as_slice()
                .iter()
                .chain(m.im.as_slice())
                .any(|v| *v != 0.)
        }) {
            return Err(DftError::Unsupported {
                feature: "spinor nuclear gradient with additional spin/SOC operators".into(),
                missing: "geometry derivatives of the supplied/generated spin operators".into(),
            });
        }
        super::spinor_gradient::stationary::evaluate(
            self.scalar.field_free_nuclear_frame()?,
            self,
            reference,
            config,
        )
    }
    pub fn stress(
        &self,
        reference: &SpinorScfReport,
        config: &SpinorScfConfig,
    ) -> Result<super::spinor_gradient::SpinorStress> {
        if self.spin_operator.iter().chain(&self.nuclear_soc).any(|m| {
            m.re.as_slice()
                .iter()
                .chain(m.im.as_slice())
                .any(|v| *v != 0.)
        }) {
            return Err(DftError::Unsupported {
                feature: "spinor stress with additional spin/SOC operators".into(),
                missing: "geometry derivatives of the supplied/generated spin operators".into(),
            });
        }
        super::spinor_gradient::stress::evaluate(
            self.scalar.field_free_nuclear_frame()?,
            self,
            reference,
            config,
        )
    }
}
impl SpinorEnergyBackend for SpinorDftBackend<'_> {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        self.electrons
    }
    fn orbital_kernel<'a>(
        &'a self,
        density: &'a [BlochMatrix],
    ) -> Result<Box<dyn SpinorDensityKernel + 'a>> {
        if self.functional.family != Family::Lda && !self.functional.parts.is_empty() {
            return Err(DftError::Unsupported {
                feature: "periodic spinor trust-region augmented-Hessian SCF".into(),
                missing: "the projected XC kernel on nonstationary SCF trial densities".into(),
            });
        }
        Ok(Box::new(kernel::DensityKernel::new(self, density)?))
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        if density.len() != self.overlap.len()
            || density.iter().any(|p| !valid(p, self.overlap[0].rows()))
        {
            return Err(DftError::InvalidInput(
                "spinor density must match the cached Hermitian spin/AO mesh".into(),
            ));
        }
        let pc: Vec<_> = density
            .iter()
            .map(pauli_components)
            .collect::<Result<_>>()?;
        let diag: [Vec<_>; 2] = std::array::from_fn(|s| {
            pc.iter()
                .map(|p| {
                    let mut v = p[0].clone();
                    v.add_scaled(&p[3], if s == 0 { 1.0 } else { -1.0 });
                    v.scale(0.5);
                    v
                })
                .collect()
        });
        let fields = self.scalar.evaluate_without_semilocal(&diag)?;
        let mut fock: Vec<_> = fields.fock[0]
            .iter()
            .zip(&fields.fock[1])
            .map(|(a, b)| {
                let mut p = std::array::from_fn(|_| BlochMatrix::zeros(a.rows(), a.cols()));
                p[0] = a.clone();
                p[0].add_scaled(b, 1.0);
                p[3] = a.clone();
                p[3].add_scaled(b, -1.0);
                from_pauli(&p)
            })
            .collect::<Result<_>>()?;
        let nk = density.len() as f64;
        let mut ex = [
            fields.full_exchange_energy,
            fields.long_range_exchange_energy,
        ];
        for (component, coefficient, kernel) in [
            (0, self.functional.hybrid.cx, CoulombKernel::Coulomb),
            (
                1,
                self.functional.hybrid.clr,
                CoulombKernel::Erf {
                    omega: self.functional.hybrid.omega,
                },
            ),
        ] {
            if coefficient == 0.0 {
                continue;
            }
            let mut transverse: Vec<[BlochMatrix; 4]> = pc
                .iter()
                .map(|p| std::array::from_fn(|_| BlochMatrix::zeros(p[0].rows(), p[0].cols())))
                .collect();
            for axis in 1..=2 {
                let d: Vec<_> = pc.iter().map(|p| p[axis].clone()).collect();
                let k = self.scalar.exchange(&d, kernel)?;
                for ik in 0..density.len() {
                    ex[component] -= 0.25 * coefficient / nk * pc[ik][axis].trace_product(&k[ik]);
                    transverse[ik][axis] = k[ik].clone();
                    transverse[ik][axis].scale(-coefficient);
                }
            }
            for (f, p) in fock.iter_mut().zip(&transverse) {
                f.add_scaled(&from_pauli(p)?, 1.0);
            }
        }
        let mut xc_diagnostics = None;
        let mut grid_number_defect = 0.0;
        if let Some(xc) = &fields.xc {
            grid_number_defect = xc.grid_number_defect;
        } else if let Some(grid) = self.scalar.xc_grid() {
            let mut grid_number = 0.0;
            for (p, ao) in pc.iter().zip(&grid.aos) {
                for g in 0..grid.weights.len() {
                    for a in 0..p[0].rows() {
                        for b in 0..p[0].cols() {
                            grid_number += grid.weights[g]
                                * (z(&p[0], a, b) * z(&ao.phi, g, a) * z(&ao.phi, g, b).conj()).re
                                / nk;
                        }
                    }
                }
            }
            let analytic_number = density
                .iter()
                .zip(&self.overlap)
                .map(|(p, s)| p.trace_product(s))
                .sum::<f64>()
                / nk;
            grid_number_defect = (grid_number - analytic_number).abs();
        }
        let (semilocal, mut xc_parts) = if let Some(grid) = self.scalar.xc_grid() {
            let (energy, potential, parts) = if self.functional.family == Family::Lda {
                noncollinear_lda(self.functional, grid, &pc)?
            } else {
                let result = super::spinor_xc::spinor_semilocal_grid_converged(
                    self.functional,
                    &grid.weights,
                    &grid.aos.iter().map(|a| &a.phi).collect::<Vec<_>>(),
                    &self.xc_momentum.iter().collect::<Vec<_>>(),
                    &pc,
                    self.scalar.xc_tolerances()[0],
                )?;
                xc_diagnostics = Some(result.3);
                (result.0, result.1, result.2)
            };
            for (f, v) in fock.iter_mut().zip(&potential) {
                f.add_scaled(v, 1.0);
            }
            (energy, parts)
        } else {
            (0.0, Vec::new())
        };
        // The scalar VV10 potential acts on both diagonal spin blocks.
        let nonlocal = fields.xc.as_ref().map_or(0.0, |xc| {
            xc_parts.push(xc.nonlocal_energy);
            xc.nonlocal_energy
        });
        let nuclear_soc_energy = density
            .iter()
            .zip(&self.nuclear_soc)
            .map(|(p, h)| p.trace_product(h))
            .sum::<f64>()
            / nk;
        let spin_energy = density
            .iter()
            .zip(&self.spin_operator)
            .map(|(p, h)| p.trace_product(h))
            .sum::<f64>()
            / nk
            + nuclear_soc_energy;
        for (f, h) in fock.iter_mut().zip(&self.spin_operator) {
            f.add_scaled(h, 1.0);
        }
        for (f, h) in fock.iter_mut().zip(&self.nuclear_soc) {
            f.add_scaled(h, 1.0);
        }
        let components = [
            self.scalar.source_energy(),
            fields.kinetic_energy,
            fields.electron_nuclear_energy,
            fields.ecp_energy,
            fields.hartree_energy,
            ex[0],
            ex[1],
            semilocal + nonlocal,
            fields.external_field_energy,
            spin_energy,
        ];
        let energy = components.iter().sum::<f64>();
        if !energy.is_finite() || fock.iter().any(|f| !valid(f, self.overlap[0].rows())) {
            return Err(DftError::InvalidInput(
                "non-finite or non-Hermitian spinor energy/Fock".into(),
            ));
        }
        Ok(SpinorFields {
            fock,
            energy,
            energy_components: components,
            xc_component_energies: xc_parts,
            nuclear_soc_energy,
            operator_converged: fields.coulomb.converged
                && grid_number_defect <= self.scalar.xc_tolerances()[1]
                && fields.xc.as_ref().is_none_or(|xc| xc.converged)
                && xc_diagnostics.as_ref().is_none_or(|x| x.converged),
            coulomb_operator_defect: fields.coulomb.quadrature_operator_defect,
            xc_grid_number_defect: grid_number_defect,
            xc_diagnostics,
        })
    }
}

/// A backend with an additional fixed one-electron spinor operator and a
/// prescribed starting Fock. Seeding a broken-symmetry state, releasing the
/// seed and restarting from a converged state reuse the wrapped integrals.
pub struct BiasedSpinorBackend<'a, B: SpinorEnergyBackend> {
    inner: &'a B,
    bias: Vec<BlochMatrix>,
    initial: Vec<BlochMatrix>,
}
impl<'a, B: SpinorEnergyBackend> BiasedSpinorBackend<'a, B> {
    /// `bias` enters the Fock and, as its expectation value per k point, the
    /// energy. `initial` replaces the starting Fock; without it the SCF
    /// starts from the starting Fock of the wrapped backend plus the bias.
    pub fn new(
        inner: &'a B,
        bias: Option<Vec<BlochMatrix>>,
        initial: Option<Vec<BlochMatrix>>,
    ) -> Result<Self> {
        let count = inner.overlap().len();
        let size = inner.overlap().first().map_or(0, BlochMatrix::rows);
        let bias = bias.unwrap_or_else(|| vec![BlochMatrix::zeros(size, size); count]);
        if bias.len() != count || bias.iter().any(|h| !valid(h, size)) {
            return Err(DftError::InvalidInput(
                "a spinor bias must be finite Hermitian [k][2*AO][2*AO] matrices on the SCF mesh"
                    .into(),
            ));
        }
        let initial = initial.unwrap_or_else(|| {
            inner
                .initial_fock()
                .iter()
                .zip(&bias)
                .map(|(f, v)| {
                    let mut a = f.clone();
                    a.add_scaled(v, 1.0);
                    a
                })
                .collect()
        });
        if initial.len() != count || initial.iter().any(|h| !valid(h, size)) {
            return Err(DftError::InvalidInput(
                "a starting spinor Fock must be finite Hermitian [k][2*AO][2*AO] matrices on \
                 the SCF mesh"
                    .into(),
            ));
        }
        Ok(Self {
            inner,
            bias,
            initial,
        })
    }
    /// Expectation value of the bias per k point.
    pub fn bias_energy(&self, density: &[BlochMatrix]) -> f64 {
        density
            .iter()
            .zip(&self.bias)
            .map(|(p, v)| p.trace_product(v))
            .sum::<f64>()
            / self.bias.len() as f64
    }
}
impl<B: SpinorEnergyBackend> SpinorEnergyBackend for BiasedSpinorBackend<'_, B> {
    fn overlap(&self) -> &[BlochMatrix] {
        self.inner.overlap()
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        self.inner.electrons()
    }
    fn validate_occupation(&self, occupation: OccupationScheme) -> Result<()> {
        self.inner.validate_occupation(occupation)
    }
    fn quadratic_density_energy(&self) -> bool {
        self.inner.quadratic_density_energy()
    }
    fn orbital_kernel<'a>(
        &'a self,
        density: &'a [BlochMatrix],
    ) -> Result<Box<dyn SpinorDensityKernel + 'a>> {
        // A fixed one-electron bias has zero density derivative.
        self.inner.orbital_kernel(density)
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let mut fields = self.inner.evaluate(density)?;
        let energy = self.bias_energy(density);
        for (f, v) in fields.fock.iter_mut().zip(&self.bias) {
            f.add_scaled(v, 1.0);
        }
        fields.energy += energy;
        fields.energy_components[9] += energy;
        Ok(fields)
    }
}

/// Local eigenvalues (n +/- |m|)/2 define the rotationally invariant LDA.
/// Differentiating these eigenvalues gives the full noncollinear XC potential.
pub(crate) fn noncollinear_lda(
    functional: &ResolvedFunctional,
    grid: &super::xc::PeriodicXcGrid,
    pauli: &[[BlochMatrix; 4]],
) -> Result<(f64, Vec<BlochMatrix>, Vec<f64>)> {
    noncollinear_lda_values(
        functional,
        &grid.weights,
        &grid.aos.iter().map(|a| &a.phi).collect::<Vec<_>>(),
        pauli,
    )
}

/// Local spin-density energy and its variational potential on supplied
/// complex orbitals. Providers own the quadrature and its convergence tests.
pub(crate) fn noncollinear_lda_values(
    functional: &ResolvedFunctional,
    weights: &[f64],
    aos: &[&BlochMatrix],
    pauli: &[[BlochMatrix; 4]],
) -> Result<(f64, Vec<BlochMatrix>, Vec<f64>)> {
    let ng = weights.len();
    let n = pauli[0][0].rows();
    let local = lda_collocate(ng, aos, pauli);
    let (rho, magnetization) = lda_spin_frame(&local)?;
    let vars = GridVars {
        rho: [&rho[0], &rho[1]],
        sigma: None,
        tau: None,
    };
    let mut out = XcOut::default();
    functional.eval(&vars, 1, &mut out)?;
    let energy = out.e.iter().zip(weights).map(|(e, w)| e * w).sum();
    let parts = functional
        .parts
        .iter()
        .map(|(weight, part)| {
            let mut o = XcOut::default();
            o.prepare(ng, 0, functional.family);
            part.eval_add(&vars, 0, *weight, &mut o)?;
            Ok(o.e.iter().zip(weights).map(|(e, w)| e * w).sum())
        })
        .collect::<Result<Vec<f64>>>()?;
    let potential: Vec<_> = magnetization
        .iter()
        .enumerate()
        .map(|(g, &(_, axis))| {
            let plus = out.vrho[0][g] + out.vrho[1][g];
            let minus = out.vrho[0][g] - out.vrho[1][g];
            [plus, minus * axis[0], minus * axis[1], minus * axis[2]]
        })
        .collect();
    Ok((energy, lda_potential(n, weights, aos, &potential)?, parts))
}

fn lda_collocate(ng: usize, aos: &[&BlochMatrix], pauli: &[[BlochMatrix; 4]]) -> Vec<[f64; 4]> {
    let n = pauli[0][0].rows();
    let nk = pauli.len() as f64;
    let mut local = vec![[0.0; 4]; ng];
    for (p, phi) in pauli.iter().zip(aos) {
        for (g, row) in local.iter_mut().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let q = z(phi, g, i) * z(phi, g, j).conj();
                    for axis in 0..4 {
                        row[axis] += (z(&p[axis], i, j) * q).re / nk;
                    }
                }
            }
        }
    }
    local
}

type LdaSpinFrame = ([Vec<f64>; 2], Vec<(f64, [f64; 3])>);

fn lda_spin_frame(local: &[[f64; 4]]) -> Result<LdaSpinFrame> {
    let mut rho = [vec![0.0; local.len()], vec![0.0; local.len()]];
    let mut frame = Vec::with_capacity(local.len());
    for (g, row) in local.iter().enumerate() {
        let mag = row[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if row[0] < -1e-12 || mag > row[0] + 1e-12 {
            return Err(DftError::InvalidInput(
                "spinor grid density has a negative local eigenvalue".into(),
            ));
        }
        rho[0][g] = 0.5 * (row[0] + mag);
        rho[1][g] = 0.5 * (row[0] - mag);
        let axis = if mag > 0.0 {
            std::array::from_fn(|a| row[a + 1] / mag)
        } else {
            [0.0; 3]
        };
        frame.push((mag, axis));
    }
    Ok((rho, frame))
}

fn lda_potential(
    n: usize,
    weights: &[f64],
    aos: &[&BlochMatrix],
    values: &[[f64; 4]],
) -> Result<Vec<BlochMatrix>> {
    let mut potential = Vec::with_capacity(aos.len());
    for phi in aos {
        let mut p = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
        for (g, (value, &weight)) in values.iter().zip(weights).enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let q = z(phi, g, i).conj() * z(phi, g, j) * weight;
                    for (block, &v) in p.iter_mut().zip(value) {
                        let old = z(block, i, j);
                        put(block, i, j, old + q * v);
                    }
                }
            }
        }
        potential.push(from_pauli(&p)?);
    }
    Ok(potential)
}

#[derive(Clone, Debug)]
pub struct SpinorScfConfig {
    pub max_iter: usize,
    pub energy_tolerance: f64,
    pub density_tolerance: f64,
    pub orbital_tolerance: f64,
    pub overlap_cutoff: f64,
    pub damping: f64,
    pub diis_size: usize,
    pub occupation: OccupationScheme,
    pub orbital_optimizer: crate::scf::OrbitalOptimizer,
}
impl Default for SpinorScfConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            energy_tolerance: 1e-10,
            density_tolerance: 1e-10,
            orbital_tolerance: 1e-10,
            overlap_cutoff: 1e-7,
            damping: 0.25,
            diis_size: 12,
            occupation: OccupationScheme::Insulator,
            orbital_optimizer: crate::scf::OrbitalOptimizer::Auto,
        }
    }
}
pub struct SpinorScfReport {
    pub density: Vec<BlochMatrix>,
    pub fields: SpinorFields,
    pub bands: Vec<BandOrbitals>,
    pub occupations: Vec<Vec<f64>>,
    pub chemical_potential: Option<f64>,
    pub entropy_term: f64,
    pub free_energy: f64,
    pub iterations: usize,
    pub converged: bool,
    pub density_change: f64,
    pub orbital_residual: f64,
    pub occupation_consistency: f64,
    pub particle_number_error: f64,
    pub metric_defect: f64,
    pub energy_change: f64,
}
fn occupied_density(bands: &[BandOrbitals], occupations: &[Vec<f64>]) -> Vec<BlochMatrix> {
    bands
        .iter()
        .zip(occupations)
        .map(|(b, f)| {
            let n = b.coefficients.rows();
            let mut p = BlochMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let value = f.iter().enumerate().fold(Complex::ZERO, |sum, (a, fa)| {
                        sum + z(&b.coefficients, i, a) * z(&b.coefficients, j, a).conj() * *fa
                    });
                    put(&mut p, i, j, value);
                }
            }
            p
        })
        .collect()
}
/// Exact endpoint work for a quadratic density functional. The supplied
/// change avoids subtracting nearly equal absolute energies; all Hermitian
/// real/imaginary and k-point terms use compensated products.
#[cfg(test)]
fn quadratic_density_work(change: &[BlochMatrix], old: &[BlochMatrix], new: &[BlochMatrix]) -> f64 {
    minimize::quadratic_density_work(change, old, new) / change.len() as f64
}

/// Matrix-independent variational SCF. Residuals are reevaluated with fresh
/// physical Focks; CDIIS extrapolates all k points in one common subspace.
pub fn run_spinor_scf(
    backend: &impl SpinorEnergyBackend,
    config: &SpinorScfConfig,
) -> Result<SpinorScfReport> {
    backend.validate_occupation(config.occupation)?;
    if config.orbital_optimizer == crate::scf::OrbitalOptimizer::Trah
        && !matches!(config.occupation, OccupationScheme::Insulator)
    {
        return Err(DftError::Unsupported {
            feature: "forced spinor TRAH".into(),
            missing: "the finite-temperature occupation Hessian".into(),
        });
    }
    config.occupation.validate()?;
    if config.max_iter == 0
        || config.diis_size == 0
        || ![
            config.energy_tolerance,
            config.density_tolerance,
            config.orbital_tolerance,
            config.overlap_cutoff,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
        || !config.damping.is_finite()
        || !(0.0..1.0).contains(&config.damping)
    {
        return Err(DftError::InvalidInput("spinor SCF requires positive finite tolerances, iteration/history sizes and damping in [0,1)".into()));
    }
    let overlaps = backend.overlap();
    if overlaps.is_empty() || overlaps.len() != backend.initial_fock().len() {
        return Err(DftError::InvalidInput(
            "spinor SCF requires matching nonempty overlap/initial Fock meshes".into(),
        ));
    }
    let diagonalize = |f: &[BlochMatrix]| -> Result<Vec<BandOrbitals>> {
        f.iter()
            .zip(overlaps)
            .map(|(h, s)| generalized_orbitals(h, s, config.overlap_cutoff))
            .collect()
    };
    let initial = diagonalize(backend.initial_fock())?;
    let energies: Vec<_> = initial.iter().map(|b| b.energies.clone()).collect();
    let occ = solve_occupations_during_scf(&energies, backend.electrons(), config.occupation)?;
    let mut density = occupied_density(&initial, &occ.occupations);
    let orth = overlaps
        .iter()
        .map(|s| Orthogonalizer::canonical(&s.real_embedding(), config.overlap_cutoff))
        .collect::<Result<Vec<_>>>()?;
    let mut diis = Diis::new(config.diis_size);
    let mut last_energy = f64::INFINITY;
    let mut dp = f64::INFINITY;
    let mut best_orbital = f64::INFINITY;
    let mut stagnant = 0usize;
    let mut minimizer: Option<minimize::OrbitalMinimizer> = None;
    let mut augmented: Option<minimize::AugmentedHessianMinimizer> = None;
    let mut trah_unavailable = false;
    let mut globalization_entry_residual = f64::INFINITY;
    for iteration in 1..=config.max_iter {
        let fields = backend.evaluate(&density)?;
        let bands = diagonalize(&fields.fock)?;
        let energies: Vec<_> = bands.iter().map(|b| b.energies.clone()).collect();
        let occ = solve_occupations_during_scf(&energies, backend.electrons(), config.occupation)?;
        let canonical = occupied_density(&bands, &occ.occupations);
        let consistency = density
            .iter()
            .zip(&canonical)
            .map(|(a, b)| defect(a, b))
            .fold(0.0_f64, f64::max);
        let number = density
            .iter()
            .zip(overlaps)
            .map(|(p, s)| p.trace_product(s))
            .sum::<f64>()
            / density.len() as f64;
        let free = fields.energy - occ.entropy_term;
        let de = (free - last_energy).abs();
        let mut errors = Vec::new();
        let mut orbital = 0.0_f64;
        let thermal_energy =
            config.occupation.temperature_kelvin() * super::occupations::BOLTZMANN_HARTREE_PER_K;
        for ((((f, p), s), o), target) in fields
            .fock
            .iter()
            .zip(&density)
            .zip(overlaps)
            .zip(&orth)
            .zip(&canonical)
        {
            let mut comm = matmul(&matmul(f, p)?, s)?;
            comm.add_scaled(&matmul(&matmul(s, p)?, f)?, -1.0);
            let raw = o.x.transpose_matmul(&comm.real_embedding().matmul(&o.x));
            orbital = orbital.max(
                raw.as_slice()
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max),
            );
            errors.extend_from_slice(raw.as_slice());
            if thermal_energy > 0.0 {
                // Commuting F and P do not establish thermal stationarity.
                // Include the fresh total-number occupation fixed-point
                // residual in the same immutable overlap frame as CDIIS.
                // Multiplication by kBT gives energy units without changing
                // the physical Fock, occupations, number or acceptance gates.
                let mut difference = target.clone();
                difference.add_scaled(p, -1.0);
                let metric = matmul(s, &matmul(&difference, s)?)?.real_embedding();
                let occupation = o.x.transpose_matmul(&metric.matmul(&o.x));
                errors.extend(occupation.as_slice().iter().map(|v| thermal_energy * v));
            }
        }
        crate::logger::log(&format!(
            "[spinor-scf] iter {iteration:>3} E={:.12} F={free:.12} dF={de:.3e} residual={orbital:.3e} dP={dp:.3e} occupation={consistency:.3e}",
            fields.energy
        ));
        let converged = de < config.energy_tolerance
            && dp < config.density_tolerance
            && consistency < config.density_tolerance
            && orbital < config.orbital_tolerance
            && (number - backend.electrons()).abs() < config.density_tolerance;
        if converged || iteration == config.max_iter {
            if converged {
                solve_occupations(&energies, backend.electrons(), config.occupation)?;
            }
            let metric = bands
                .iter()
                .map(|b| b.metric_defect)
                .fold(0.0_f64, f64::max);
            return Ok(SpinorScfReport {
                density,
                fields,
                bands,
                occupations: occ.occupations,
                chemical_potential: occ.chemical_potential,
                entropy_term: occ.entropy_term,
                free_energy: free,
                iterations: iteration,
                converged,
                density_change: dp,
                orbital_residual: orbital,
                occupation_consistency: consistency,
                particle_number_error: (number - backend.electrons()).abs(),
                metric_defect: metric,
                energy_change: de,
            });
        }
        if orbital < 0.7 * best_orbital {
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        best_orbital = best_orbital.min(orbital);
        let trah_requested = config.orbital_optimizer == crate::scf::OrbitalOptimizer::Trah
            || (config.orbital_optimizer == crate::scf::OrbitalOptimizer::Auto && stagnant >= 6);
        if trah_requested
            && orbital > config.orbital_tolerance
            && !trah_unavailable
            && augmented.is_none()
            && matches!(config.occupation, OccupationScheme::Insulator)
        {
            match backend.orbital_kernel(&canonical).map(|_| ()) {
                Ok(_) => {
                    augmented = Some(minimize::AugmentedHessianMinimizer::new(
                        bands.iter().map(|b| b.coefficients.clone()).collect(),
                        &occ.occupations,
                    )?);
                    dp = canonical
                        .iter()
                        .zip(&density)
                        .map(|(a, b)| defect(a, b))
                        .fold(0.0, f64::max);
                    density = canonical;
                    last_energy = free;
                    minimizer = None;
                    diis = Diis::new(config.diis_size);
                    continue;
                }
                Err(DftError::Unsupported { .. })
                    if config.orbital_optimizer == crate::scf::OrbitalOptimizer::Auto =>
                {
                    trah_unavailable = true;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(optimizer) = &mut augmented {
            if let Some(step) = optimizer.step(backend, &density, &fields.fock, fields.energy)? {
                dp = step
                    .density
                    .iter()
                    .zip(&density)
                    .map(|(a, b)| defect(a, b))
                    .fold(0.0, f64::max);
                density = step.density;
                last_energy = free;
                continue;
            }
            // No rejected or roundoff-unresolved step counts as convergence.
            augmented = None;
            stagnant = 0;
            diis = Diis::new(config.diis_size);
        }
        if minimizer.is_some() && orbital < 0.01 * globalization_entry_residual {
            // Energy globalization finds a local basin. Once the physical
            // gradient has fallen by two orders, resume local CDIIS rather
            // than compare vanishing energy changes at floating precision.
            // This switch leaves the fresh physical convergence gates active.
            crate::logger::log(&format!(
                "[spinor-scf] resume local CDIIS residual={orbital:.6e} entry_residual={globalization_entry_residual:.6e}"
            ));
            minimizer = None;
            stagnant = 0;
            best_orbital = orbital;
            diis = Diis::new(config.diis_size);
        }
        // A stationary orbital residual can precede occupation and density
        // convergence. Let CDIIS finish those checks before taking a
        // downhill step that can leave the current stationary branch.
        if minimizer.is_none()
            && stagnant >= 6
            && orbital > config.orbital_tolerance
            && matches!(config.occupation, OccupationScheme::Insulator)
        {
            // Restart on the integer occupied-subspace manifold. Subsequent
            // gradient steps preserve full complex spin coherence and use
            // the same variational backend as the physical SCF checks.
            minimizer = Some(minimize::OrbitalMinimizer::new(
                bands.iter().map(|b| b.coefficients.clone()).collect(),
                &occ.occupations,
            ));
            globalization_entry_residual = orbital;
            dp = canonical
                .iter()
                .zip(&density)
                .map(|(a, b)| defect(a, b))
                .fold(0.0, f64::max);
            density = canonical;
            last_energy = free;
            diis = Diis::new(config.diis_size);
            crate::logger::log(
                "[spinor-scf] start complex orbital limited-memory quasi-Newton globalization",
            );
            continue;
        }
        if let Some(optimizer) = &mut minimizer {
            if let Some(step) = optimizer.step(
                &density,
                &fields.fock,
                fields.energy,
                1.0 / density.len() as f64,
                backend.quadratic_density_energy(),
                |p| {
                    let result = backend.evaluate(p)?;
                    Ok((result.energy, result.fock))
                },
            )? {
                dp = step
                    .density
                    .iter()
                    .zip(&density)
                    .map(|(a, b)| defect(a, b))
                    .fold(0.0, f64::max);
                crate::logger::log(&format!(
                    "[spinor-scf] orbital quasi-Newton fraction={:.6e} secants={} energy_work={:.6e}",
                    step.fraction, step.history_size, step.work
                ));
                density = step.density;
                last_energy = free;
                continue;
            }
            // A line-search failure returns to a fresh local subspace;
            // neither it nor the gradient-reduction switch accepts SCF.
            minimizer = None;
            stagnant = 0;
            best_orbital = orbital;
            diis = Diis::new(config.diis_size);
        }
        let embedded: Vec<_> = fields
            .fock
            .iter()
            .map(BlochMatrix::real_embedding)
            .collect();
        diis.push_with_errors(&embedded, errors);
        let trial = if let Some(f) = diis.extrapolate() {
            f.iter()
                .map(|a| {
                    let n = a.rows / 2;
                    let mut h = BlochMatrix::zeros(n, n);
                    for i in 0..n {
                        for j in 0..n {
                            put(&mut h, i, j, Complex::new(a[(i, j)], a[(n + i, j)]));
                        }
                    }
                    h
                })
                .collect()
        } else {
            fields.fock.clone()
        };
        let next_bands = diagonalize(&trial)?;
        let next_energies: Vec<_> = next_bands.iter().map(|b| b.energies.clone()).collect();
        let next_occ =
            solve_occupations_during_scf(&next_energies, backend.electrons(), config.occupation)?;
        let mut next = occupied_density(&next_bands, &next_occ.occupations);
        // Remove damping in the convergence region; every accepted density is
        // subsequently checked against a fresh canonical physical Fock.
        if orbital > 0.01 {
            for (p, old) in next.iter_mut().zip(&density) {
                p.scale(1.0 - config.damping);
                p.add_scaled(old, config.damping);
            }
        }
        dp = next
            .iter()
            .zip(&density)
            .map(|(a, b)| defect(a, b))
            .fold(0.0_f64, f64::max);
        density = next;
        last_energy = free;
    }
    unreachable!("positive spinor iteration limit")
}

/// Pauli SOC of a prescribed periodic scalar electron potential and a general
/// periodic Zeeman field, in atomic units: sigma.(grad V cross p)/(4 c^2)
/// and sigma.B/2. This model does not substitute for nuclear/core SOC.
pub fn periodic_pauli_operators(
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    points: &[Vec3],
    cutoff: f64,
    wave: Vec3,
    scalar: Complex,
    spin_field: [Complex; 3],
    soc_scale: f64,
) -> Result<Vec<BlochMatrix>> {
    if !soc_scale.is_finite() || !scalar.is_finite() || spin_field.iter().any(|z| !z.is_finite()) {
        return Err(DftError::InvalidInput(
            "Pauli SOC and spin-field coefficients must be finite".into(),
        ));
    }
    // Reuse the defining Bloch-periodicity and field validation.
    StaticPeriodicField {
        wave,
        scalar_potential: scalar,
        vector_potential: [Complex::ZERO; 3],
        amplitude: 0.0,
    }
    .operators(basis, geometry, points, cutoff)?;
    let n = basis.nao_sph;
    let mut pauli: Vec<_> = points
        .iter()
        .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(n, n)))
        .collect();
    let c = 137.035_999_084_f64;
    for sign in [1.0, -1.0] {
        let q = wave * sign;
        let v = if sign > 0.0 { scalar } else { scalar.conj() };
        let b = spin_field.map(|v| if sign > 0.0 { v } else { v.conj() });
        let (d, current) = super::field::vertices(basis, geometry, points, q, cutoff, true)?;
        for ik in 0..points.len() {
            for a in 0..3 {
                let x = (a + 1) % 3;
                let y = (a + 2) % 3;
                for i in 0..n {
                    for j in 0..n {
                        let soc = I
                            * v
                            * (z(&current[y][ik], i, j) * q.get(x)
                                - z(&current[x][ik], i, j) * q.get(y))
                            * (soc_scale / (4.0 * c * c));
                        // from_pauli has a factor 1/2: store twice each Pauli coefficient.
                        let value = z(&pauli[ik][a + 1], i, j) + z(&d[ik], i, j) * b[a] + soc * 2.0;
                        put(&mut pauli[ik][a + 1], i, j, value);
                    }
                }
            }
        }
    }
    pauli.iter().map(from_pauli).collect()
}

/// Raw spinful time-reversal defect P(k)-i sigma_y P(-k)* (-i sigma_y).
/// This diagnostic never projects either the spin or orbital density.
pub fn spinful_time_reversal_defect(matrices: &[BlochMatrix], mesh: [usize; 3]) -> Result<f64> {
    let count = mesh.iter().try_fold(
        1usize,
        |n, v| if *v == 0 { None } else { n.checked_mul(*v) },
    );
    if matrices.is_empty()
        || count != Some(matrices.len())
        || matrices[0].rows() % 2 != 0
        || matrices.iter().any(|a| !valid(a, matrices[0].rows()))
    {
        return Err(DftError::InvalidInput("spinful time reversal requires a complete uniform mesh of finite Hermitian even-dimensional matrices".into()));
    }
    let n = matrices[0].rows() / 2;
    let mut worst = 0.0_f64;
    for x in 0..mesh[0] {
        for y in 0..mesh[1] {
            for k in 0..mesh[2] {
                let i = (x * mesh[1] + y) * mesh[2] + k;
                let j = (((mesh[0] - x) % mesh[0]) * mesh[1] + (mesh[1] - y) % mesh[1]) * mesh[2]
                    + (mesh[2] - k) % mesh[2];
                for s in 0..2 {
                    for t in 0..2 {
                        for a in 0..n {
                            for b in 0..n {
                                let sign = if s == t { 1.0 } else { -1.0 };
                                worst = worst.max(
                                    (z(&matrices[i], s * n + a, t * n + b)
                                        - z(&matrices[j], (1 - s) * n + a, (1 - t) * n + b).conj()
                                            * sign)
                                        .abs(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(worst)
}

/// The exact second amplitude derivative of sigma.(grad V cross A)/(4 c^2).
/// V and A scale together. Both zero and +/-2q Fourier components are retained.
pub fn periodic_pauli_contact(
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    points: &[Vec3],
    cutoff: f64,
    field: &StaticPeriodicField,
    soc_scale: f64,
) -> Result<Vec<BlochMatrix>> {
    if !soc_scale.is_finite() {
        return Err(DftError::InvalidInput("SOC scale must be finite".into()));
    }
    field.operators(basis, geometry, points, cutoff)?;
    let n = basis.nao_sph;
    let mut pauli: Vec<_> = points
        .iter()
        .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(n, n)))
        .collect();
    let c = 137.035_999_084_f64;
    for sv in [1.0, -1.0] {
        for sa in [1.0, -1.0] {
            let v = if sv > 0.0 {
                field.scalar_potential
            } else {
                field.scalar_potential.conj()
            };
            let a = field
                .vector_potential
                .map(|a| if sa > 0.0 { a } else { a.conj() });
            let q = field.wave * sv;
            let (density, _) = super::field::vertices(
                basis,
                geometry,
                points,
                field.wave * (sv + sa),
                cutoff,
                false,
            )?;
            for axis in 0..3 {
                let x = (axis + 1) % 3;
                let y = (axis + 2) % 3;
                let coefficient =
                    I * v * (a[y] * q.get(x) - a[x] * q.get(y)) * (soc_scale / (c * c));
                // from_pauli contributes 1/2 and H(lambda) contributes lambda^2/2.
                for ik in 0..points.len() {
                    for i in 0..n {
                        for j in 0..n {
                            let old = z(&pauli[ik][axis + 1], i, j);
                            put(
                                &mut pauli[ik][axis + 1],
                                i,
                                j,
                                old + z(&density[ik], i, j) * coefficient,
                            );
                        }
                    }
                }
            }
        }
    }
    pauli.iter().map(from_pauli).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_work_matches_closed_complex_multik_polynomials() {
        let matrix = |v: [f64; 4]| {
            let mut m = BlochMatrix::zeros(2, 2);
            put(&mut m, 0, 0, Complex::new(v[0], 0.0));
            put(&mut m, 1, 1, Complex::new(v[1], 0.0));
            put(&mut m, 0, 1, Complex::new(v[2], v[3]));
            put(&mut m, 1, 0, Complex::new(v[2], -v[3]));
            m
        };
        // Closed Hermitian two-by-two trace, independent of matrix products
        // and of the production compensated endpoint-work contraction.
        let trace = |a: [f64; 4], b: [f64; 4]| {
            a[0] * b[0] + a[1] * b[1] + 2.0 * (a[2] * b[2] + a[3] * b[3])
        };
        let core = [[-1.1, 0.7, 0.17, -0.21], [-0.8, 0.4, -0.09, 0.13]];
        let source = [[0.3, -0.2, 0.11, 0.07], [-0.4, 0.5, 0.08, -0.06]];
        let density = [[0.8, 0.2, 0.13, -0.1], [0.6, 0.4, -0.17, 0.04]];
        let direction = [[-0.2, 0.2, 0.07, 0.03], [0.15, -0.15, -0.04, 0.09]];
        let coupling = [0.31, -0.19];
        for h in [0.1, 0.05, 0.025, 0.0125, 1e-10, 1e-12, 1e-14] {
            let mut old = Vec::new();
            let mut new = Vec::new();
            let mut change = Vec::new();
            let mut expected = 0.0;
            for k in 0..2 {
                let q = trace(density[k], source[k]);
                let r = trace(direction[k], source[k]);
                old.push(matrix(std::array::from_fn(|a| {
                    core[k][a] + coupling[k] * q * source[k][a]
                })));
                new.push(matrix(std::array::from_fn(|a| {
                    core[k][a] + coupling[k] * (q + h * r) * source[k][a]
                })));
                change.push(matrix(direction[k].map(|v| h * v)));
                expected += (h * (trace(direction[k], core[k]) + coupling[k] * q * r)
                    + 0.5 * coupling[k] * h * h * r * r)
                    / 2.0;
            }
            let measured = quadratic_density_work(&change, &old, &new);
            let error = (measured - expected).abs();
            println!("complex multi-k quadratic work h={h:.1e} measured={measured:.16e} defect={error:.12e}");
            assert!(error <= 16.0 * f64::EPSILON * expected.abs());
        }
    }
}
