// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent dimensional Bloch DFT with one compensated Coulomb gauge.
//! Localized full/range-separated exchange and semilocal providers use the
//! same spin density, physical periodic subspace and mesh-global occupations.

use super::coulomb_boundary::{
    DimensionalCoulombBackend, PeriodicCoulombOptions, PeriodicCoulombReport,
};
use super::electrostatics::ElectrostaticOptions;
use super::exchange::{dimensional_exchange_component_images, dimensional_exchange_image_blocks};
use super::integrals::{bloch, dimensional_image_blocks, ImageBlock, Operator};
use super::scf::{
    gwh_guess_focks, run_scf_iterations, validate_solver_options, Fields, PeriodicScfOptions,
    PeriodicScfResult, ScfIterationInput,
};
use super::xc_boundary::{DimensionalXcBackend, PeriodicXcOptions, PeriodicXcReport};
use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::MolBasis;
use crate::dfpt::orbitals::generalized_orbitals;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::response::diagnostics::ReferenceDiagnostics;
use crate::scf::{Guess, Reference};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod geometry_response;
pub(crate) use geometry_response::{
    dimensional_geometry, run_dimensional_scf_with_geometry, GeometryDerivatives,
};

#[derive(Clone, Debug)]
pub struct DimensionalScfOptions {
    /// The boundary must match the immutable geometry. The legacy grid_shell
    /// control must retain its default; grid_cutoff selects a physical radius.
    pub scf: PeriodicScfOptions,
    pub basis_cutoff: f64,
    pub grid_cutoff: f64,
    pub nuclear_widths: Vec<f64>,
    pub electrostatics: ElectrostaticOptions,
    pub operator_tolerance: f64,
    pub grid_tolerance: f64,
}
impl Default for DimensionalScfOptions {
    fn default() -> Self {
        Self {
            scf: PeriodicScfOptions::default(),
            basis_cutoff: 12.0,
            grid_cutoff: 12.0,
            nuclear_widths: Vec::new(),
            electrostatics: ElectrostaticOptions::default(),
            operator_tolerance: 1e-11,
            grid_tolerance: 1e-5,
        }
    }
}
impl DimensionalScfOptions {
    pub fn for_boundary(boundary: super::features::PeriodicBoundary) -> Self {
        let mut options = Self::default();
        options.scf.boundary = boundary;
        options
    }
}

/// Every energy component is in Hartree per physical primitive cell.
pub struct DimensionalElectronicFields {
    pub fock: [Vec<BlochMatrix>; 2],
    pub kinetic_energy: f64,
    pub electron_nuclear_energy: f64,
    pub ecp_energy: f64,
    pub external_field_energy: f64,
    pub hartree_energy: f64,
    pub full_exchange_energy: f64,
    pub long_range_exchange_energy: f64,
    pub semilocal_energy: f64,
    pub electronic_energy: f64,
    pub coulomb: PeriodicCoulombReport,
    pub xc: Option<PeriodicXcReport>,
}

/// Immutable operator cache, also usable for independent energy/Fock checks.
pub struct DimensionalScfBackend<'a> {
    mol: &'a Molecule,
    basis: &'a MolBasis,
    geometry: &'a PeriodicGeometry,
    functional: &'a ResolvedFunctional,
    options: &'a DimensionalScfOptions,
    coulomb: DimensionalCoulombBackend,
    xc: Option<DimensionalXcBackend>,
    overlap: Vec<BlochMatrix>,
    kinetic: Vec<BlochMatrix>,
    h_core: Vec<BlochMatrix>,
    ecp_images: Vec<ImageBlock>,
    static_field: Option<super::magnetic_scf::StaticPeriodicField>,
    uniform_field: Option<super::uniform_open::UniformOpenField>,
    external_operators: Vec<BlochMatrix>,
    complex_density: bool,
    spinor_mode: bool,
    n_alpha: usize,
    n_beta: usize,
    restricted: bool,
    source_energy: f64,
}
impl<'a> DimensionalScfBackend<'a> {
    pub fn new(
        mol: &'a Molecule,
        basis: &'a MolBasis,
        geometry: &'a PeriodicGeometry,
        functional: &'a ResolvedFunctional,
        options: &'a DimensionalScfOptions,
    ) -> Result<Self> {
        Self::new_internal(mol, basis, geometry, functional, options, false)
    }
    pub(crate) fn new_spinor(
        mol: &'a Molecule,
        basis: &'a MolBasis,
        geometry: &'a PeriodicGeometry,
        functional: &'a ResolvedFunctional,
        options: &'a DimensionalScfOptions,
    ) -> Result<Self> {
        Self::new_internal(mol, basis, geometry, functional, options, true)
    }
    fn new_internal(
        mol: &'a Molecule,
        basis: &'a MolBasis,
        geometry: &'a PeriodicGeometry,
        functional: &'a ResolvedFunctional,
        options: &'a DimensionalScfOptions,
        spinor: bool,
    ) -> Result<Self> {
        validate_solver_options(&options.scf)?;
        if geometry.dimension() == 0
            || options.scf.boundary != geometry.boundary()
            || options.scf.grid_shell != PeriodicScfOptions::default().grid_shell
            || [
                options.basis_cutoff,
                options.grid_cutoff,
                options.operator_tolerance,
                options.grid_tolerance,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(DftError::InvalidInput(
                "dimensional SCF requires matching 1D/2D/3D boundaries, finite positive \
                controls and the default legacy grid_shell; use grid_cutoff for the \
                physical grid radius"
                    .into(),
            ));
        }
        if let Some(term) = functional.required_nonlocal_kind().filter(|term| {
            !term.is_supplied(
                options.scf.vv10.is_some(),
                options.scf.pair_dispersion_from_caller,
            )
        }) {
            return Err(DftError::Unsupported {
                feature: "dimensional nonlocal DFT SCF".into(),
                missing: format!(
                    "the defining lattice-summed {} energy and potential",
                    term.name()
                ),
            });
        }
        if [
            functional.hybrid.cx,
            functional.hybrid.clr,
            functional.hybrid.omega,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || functional.hybrid.omega < 0.0
        {
            return Err(DftError::InvalidInput(
                "exchange coefficients and nonnegative range parameter must be finite".into(),
            ));
        }
        let electrons = basis.effective_charges(mol).iter().sum::<f64>() - mol.charge as f64;
        if !electrons.is_finite()
            || electrons < 0.0
            || (electrons - electrons.round()).abs() > 1e-10
        {
            return Err(DftError::InvalidInput(
                "SCF formal charge must define a nonnegative integer electron count".into(),
            ));
        }
        let (n_alpha, n_beta) = if spinor {
            (electrons.round() as usize, 0)
        } else {
            mol.n_alpha_beta_for(electrons.round() as usize)?
        };
        let restricted = if spinor {
            false
        } else {
            match options.scf.reference {
                Reference::Auto => n_alpha == n_beta,
                Reference::Uhf => false,
                Reference::Rhf if n_alpha == n_beta => true,
                Reference::Rhf => {
                    return Err(DftError::InvalidInput(
                        "restricted dimensional SCF requires equal spin populations".into(),
                    ))
                }
            }
        };
        let coulomb = DimensionalCoulombBackend::new(
            mol,
            basis,
            geometry,
            &PeriodicCoulombOptions {
                k_mesh: options.scf.k_mesh,
                basis_cutoff: options.basis_cutoff,
                nuclear_widths: options.nuclear_widths.clone(),
                electrostatics: options.electrostatics.clone(),
            },
        )?;
        let nuclear = coulomb.evaluate(None)?;
        let k_points = coulomb.k_points();
        let build = |operator| -> Result<Vec<BlochMatrix>> {
            let images = dimensional_image_blocks(basis, geometry, options.basis_cutoff, operator)?;
            Ok(k_points
                .iter()
                .map(|&k| BlochMatrix::from_pair(bloch(&images, k)))
                .collect())
        };
        let overlap = build(Operator::Overlap)?;
        let kinetic = build(Operator::Kinetic)?;
        let ecp_images = super::integrals::dimensional_ecp_image_blocks(
            basis,
            geometry,
            options.basis_cutoff,
            options.basis_cutoff,
        )?;
        let h_core = kinetic
            .iter()
            .zip(&nuclear.nuclear)
            .zip(k_points)
            .map(|((t, v), k)| {
                let mut h = t.clone();
                h.add_scaled(v, 1.0);
                if !ecp_images.is_empty() {
                    h.add_scaled(&BlochMatrix::from_pair(bloch(&ecp_images, *k)), 1.0);
                }
                h
            })
            .collect();
        let xc = (!functional.parts.is_empty() || options.scf.vv10.is_some())
            .then(|| {
                DimensionalXcBackend::with_reference_overlap(
                    mol,
                    basis,
                    geometry,
                    functional,
                    &PeriodicXcOptions {
                        vv10: options.scf.vv10,
                        k_mesh: options.scf.k_mesh,
                        basis_cutoff: options.scf.xc_ao_cutoff.unwrap_or(options.basis_cutoff),
                        grid_cutoff: options.grid_cutoff,
                        grid: options.scf.grid.clone(),
                        operator_tolerance: options.operator_tolerance,
                        grid_tolerance: options.grid_tolerance,
                    },
                    Some(&overlap),
                    spinor,
                )
            })
            .transpose()?;
        Ok(Self {
            mol,
            basis,
            geometry,
            functional,
            options,
            coulomb,
            xc,
            overlap,
            kinetic,
            h_core,
            ecp_images,
            static_field: None,
            uniform_field: None,
            external_operators: Vec::new(),
            complex_density: false,
            spinor_mode: spinor,
            n_alpha,
            n_beta,
            restricted,
            source_energy: nuclear.source_energy,
        })
    }
    pub fn k_points(&self) -> &[crate::math::Vec3] {
        self.coulomb.k_points()
    }
    pub fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    pub fn h_core(&self) -> &[BlochMatrix] {
        &self.h_core
    }
    /// Common real orbitals at Gamma with the cached dimensional operators.
    pub fn gamma_roks_problem(&self) -> Result<super::roks::GammaRoksProblem<'_>> {
        self.validate_roks_operators()?;
        super::roks::GammaRoksProblem::new_dimensional(
            self,
            self.functional,
            &self.options.scf,
            [self.n_alpha, self.n_beta],
        )
    }
    /// Fixed integer common complex orbitals on the cached dimensional mesh.
    pub fn mesh_roks_problem(&self) -> Result<super::roks::MeshRoksProblem<'_>> {
        self.validate_roks_operators()?;
        super::roks::MeshRoksProblem::new_dimensional(
            self,
            self.functional,
            &self.options.scf,
            [self.n_alpha, self.n_beta],
        )
    }
    fn validate_roks_operators(&self) -> Result<()> {
        if self.spinor_mode || self.static_field.is_some() || self.uniform_field.is_some() {
            return Err(DftError::Unsupported {
                feature: "dimensional ROKS with orbital fields or spinor operators".into(),
                missing: "their constrained common-orbital reference".into(),
            });
        }
        Ok(())
    }
    /// Enable unrestricted complex orbital densities without imposing a
    /// zero-field conjugation constraint. Spinor exchange needs this map.
    pub(crate) fn with_complex_density(mut self) -> Self {
        self.complex_density = true;
        self
    }
    pub(crate) fn source_energy(&self) -> f64 {
        self.source_energy
    }
    pub(crate) fn has_vector_field(&self) -> bool {
        self.uniform_field
            .as_ref()
            .is_some_and(|f| f.magnetic_field.norm() != 0.0)
            || self.static_field.as_ref().is_some_and(|f| {
                f.vector_potential
                    .iter()
                    .any(|z| *z != crate::response::complex::Complex::ZERO)
            })
    }
    pub(crate) fn uniform_field(&self) -> Option<&super::uniform_open::UniformOpenField> {
        self.uniform_field.as_ref()
    }
    pub(crate) fn uniform_gauge_matrix(&self) -> Result<Option<[[f64; 3]; 3]>> {
        self.uniform_field
            .as_ref()
            .map(|f| f.gauge_matrix(self.geometry))
            .transpose()
    }
    pub fn with_uniform_open_field(
        mut self,
        field: &super::uniform_open::UniformOpenField,
    ) -> Result<Self> {
        if self.static_field.is_some() || self.uniform_field.is_some() {
            return Err(DftError::InvalidInput(
                "attach a single uniform field before adding any independent periodic field model"
                    .into(),
            ));
        }
        if !self.basis.ecps.is_empty() {
            return Err(DftError::Unsupported {
                feature: "uniform orbital field with nonlocal ECP".into(),
                missing: "the gauge-covariant nonlocal core current and contact vertices".into(),
            });
        }
        if self.functional.family == crate::xc::Family::MetaGga && !self.spinor_mode {
            return Err(DftError::Unsupported {
                feature: "uniform orbital field meta-GGA".into(),
                missing: "the gauge-covariant current and kinetic-density functional".into(),
            });
        }
        let operators = field.operators(
            self.basis,
            self.geometry,
            self.k_points(),
            self.options.basis_cutoff,
        )?;
        self.external_operators = operators.hamiltonian;
        for (h, v) in self.h_core.iter_mut().zip(&self.external_operators) {
            h.add_scaled(v, 1.0);
        }
        self.uniform_field = Some(field.clone());
        self.complex_density = true;
        Ok(self)
    }
    /// Real finite periodic minimal coupling, retaining complex orbitals and
    /// the complete complex exchange map. The same operator enters E and F.
    pub fn with_static_field(
        mut self,
        field: &super::magnetic_scf::StaticPeriodicField,
    ) -> Result<Self> {
        if !self.basis.ecps.is_empty()
            && field
                .vector_potential
                .iter()
                .any(|z| *z != crate::response::complex::Complex::ZERO)
        {
            return Err(DftError::Unsupported {
                feature: "finite orbital field with nonlocal ECP".into(),
                missing: "gauge-covariant nonlocal-core field operators".into(),
            });
        }
        if self.functional.family == crate::xc::Family::MetaGga
            && !self.spinor_mode
            && field
                .vector_potential
                .iter()
                .any(|z| *z != crate::response::complex::Complex::ZERO)
        {
            return Err(DftError::Unsupported {
                feature: "finite orbital field meta-GGA".into(),
                missing: "gauge-covariant current and kinetic-density functional".into(),
            });
        }
        let operator = field.operators(
            self.basis,
            self.geometry,
            self.k_points(),
            self.options.basis_cutoff,
        )?;
        self.external_operators = operator.hamiltonian;
        for (h, v) in self.h_core.iter_mut().zip(&self.external_operators) {
            h.add_scaled(v, 1.0);
        }
        self.static_field = Some(field.clone());
        self.complex_density = true;
        Ok(self)
    }
    pub(crate) fn field_free_nuclear_frame(
        &self,
    ) -> Result<super::spinor_gradient::stationary::FieldFreeFrame<'_>> {
        if self.static_field.is_some()
            || self.uniform_field.is_some()
            || self.external_operators.iter().any(|m| {
                m.re.as_slice()
                    .iter()
                    .chain(m.im.as_slice())
                    .any(|v| *v != 0.)
            })
        {
            return Err(DftError::Unsupported {
                feature: "dimensional geometry derivatives with orbital fields".into(),
                missing: "matching gauge operator geometry derivatives".into(),
            });
        }
        Ok(self.stationary_geometry_frame())
    }

    pub(crate) fn stationary_geometry_frame(
        &self,
    ) -> super::spinor_gradient::stationary::FieldFreeFrame<'_> {
        super::spinor_gradient::stationary::FieldFreeFrame {
            molecule: self.mol,
            basis: self.basis,
            geometry: self.geometry,
            functional: self.functional,
            options: self.options,
            coulomb: &self.coulomb,
            grid: self.xc.as_ref().map(|x| x.grid()),
            ecp_images: &self.ecp_images,
        }
    }
    pub(crate) fn coulomb_backend(&self) -> &DimensionalCoulombBackend {
        &self.coulomb
    }
    pub(crate) fn xc_grid(&self) -> Option<&super::xc::PeriodicXcGrid> {
        self.xc.as_ref().map(|x| x.grid())
    }
    pub(crate) fn xc_tolerances(&self) -> [f64; 2] {
        [self.options.operator_tolerance, self.options.grid_tolerance]
    }
    /// Covariant momentum in the stored origin-zero AO representation.
    /// A common origin phase cancels in all gauge-corrected descriptors.
    pub(crate) fn xc_momentum(&self) -> Result<Vec<[BlochMatrix; 3]>> {
        let grid = self
            .xc_grid()
            .ok_or_else(|| DftError::InvalidInput("XC momentum requires an XC grid".into()))?;
        let matrix = self.uniform_gauge_matrix()?;
        grid.aos
            .iter()
            .map(|ao| {
                let gradients = ao.grad.as_ref().ok_or_else(|| {
                    DftError::InvalidInput("semilocal spinor momentum requires AO gradients".into())
                })?;
                let mut momentum =
                    std::array::from_fn(|_| BlochMatrix::zeros(ao.phi.rows(), ao.phi.cols()));
                for (g, r) in grid.points.iter().enumerate() {
                    let mut vector = [0.0; 3];
                    if let (Some(m), Some(field)) = (&matrix, &self.uniform_field) {
                        for d in 0..3 {
                            vector[d] =
                                field.amplitude * (0..3).map(|b| m[d][b] * r.get(b)).sum::<f64>();
                        }
                    }
                    if let Some(field) = &self.static_field {
                        let phase = field.wave.dot(*r);
                        for d in 0..3 {
                            vector[d] += 2.0
                                * field.amplitude
                                * (field.vector_potential[d].re * phase.cos()
                                    - field.vector_potential[d].im * phase.sin());
                        }
                    }
                    for d in 0..3 {
                        for a in 0..ao.phi.cols() {
                            momentum[d].re[(g, a)] =
                                gradients[d].im[(g, a)] + vector[d] * ao.phi.re[(g, a)];
                            momentum[d].im[(g, a)] =
                                -gradients[d].re[(g, a)] + vector[d] * ao.phi.im[(g, a)];
                        }
                    }
                }
                Ok(momentum)
            })
            .collect()
    }
    pub fn reference_fock_at(
        &self,
        density: &[Vec<BlochMatrix>; 2],
        points: &[crate::math::Vec3],
    ) -> Result<(Vec<BlochMatrix>, [Vec<BlochMatrix>; 2])> {
        let n = self.basis.nao_sph;
        if points.is_empty()
            || points.iter().any(|k| !(0..3).all(|a| k.get(a).is_finite()))
            || density.iter().any(|spin| {
                spin.len() != self.k_points().len()
                    || spin.iter().any(|p| {
                        p.re.rows != n
                            || p.re.cols != n
                            || p.im.rows != n
                            || p.im.cols != n
                            || p.re
                                .as_slice()
                                .iter()
                                .chain(p.im.as_slice())
                                .any(|v| !v.is_finite())
                            || p.hermiticity_error() > self.options.operator_tolerance
                    })
            })
        {
            return Err(DftError::InvalidInput(
                "shifted dimensional reference requires non-empty finite k points and \
                finite Hermitian spin densities on the original mesh with the cached AO \
                shape"
                    .into(),
            ));
        }
        if points
            .iter()
            .any(|k| self.geometry.perpendicular(*k).norm() > 1e-12 * (1.0 + k.norm()))
        {
            return Err(DftError::InvalidInput(
                "dimensional Bloch k points must lie in the physical periodic reciprocal \
                subspace; open directions have no Bloch wave number"
                    .into(),
            ));
        }
        let mut total = density[0].clone();
        for (p, b) in total.iter_mut().zip(&density[1]) {
            p.add_scaled(b, 1.0);
        }
        let (nuclear, hartree) = self.coulomb.reference_operators_at(&total, points)?;
        let overlaps = dimensional_image_blocks(
            self.basis,
            self.geometry,
            self.options.basis_cutoff,
            Operator::Overlap,
        )?;
        let kinetics = dimensional_image_blocks(
            self.basis,
            self.geometry,
            self.options.basis_cutoff,
            Operator::Kinetic,
        )?;
        let overlap = points
            .iter()
            .map(|&k| BlochMatrix::from_pair(bloch(&overlaps, k)))
            .collect();
        let mut core: Vec<_> = points
            .iter()
            .map(|&k| BlochMatrix::from_pair(bloch(&kinetics, k)))
            .collect();
        for ((f, v), j) in core.iter_mut().zip(&nuclear).zip(&hartree) {
            f.add_scaled(v, 1.0);
            f.add_scaled(j, 1.0);
        }
        for (f, k) in core.iter_mut().zip(points) {
            if !self.ecp_images.is_empty() {
                f.add_scaled(&BlochMatrix::from_pair(bloch(&self.ecp_images, *k)), 1.0);
            }
        }
        if let Some(field) = &self.static_field {
            let operator =
                field.operators(self.basis, self.geometry, points, self.options.basis_cutoff)?;
            for (h, v) in core.iter_mut().zip(&operator.hamiltonian) {
                h.add_scaled(v, 1.0);
            }
        }
        if let Some(field) = &self.uniform_field {
            let operator =
                field.operators(self.basis, self.geometry, points, self.options.basis_cutoff)?;
            for (h, v) in core.iter_mut().zip(&operator.hamiltonian) {
                h.add_scaled(v, 1.0);
            }
        }
        let mut fock = [core.clone(), core];
        for (coefficient, kernel) in [
            (self.functional.hybrid.cx, CoulombKernel::Coulomb),
            (
                self.functional.hybrid.clr,
                CoulombKernel::Erf {
                    omega: self.functional.hybrid.omega,
                },
            ),
        ] {
            if coefficient != 0.0 {
                for spin in 0..2 {
                    let (blocks, imaginary) = self.exchange_components(&density[spin], kernel)?;
                    for (f, k) in fock[spin].iter_mut().zip(points) {
                        f.add_scaled(
                            &complex_bloch(&blocks, imaginary.as_deref(), *k),
                            -coefficient,
                        );
                    }
                }
            }
        }
        if let Some(grid) = self.xc_grid() {
            let target = grid.at_dimensional_k_points(
                self.basis,
                self.geometry,
                points,
                self.options
                    .scf
                    .xc_ao_cutoff
                    .unwrap_or(self.options.basis_cutoff),
            )?;
            let xc = super::xc::xc_contribution_at_grid(self.functional, grid, density, &target)?;
            for spin in 0..2 {
                for (f, v) in fock[spin].iter_mut().zip(&xc.vxc[spin]) {
                    f.add_scaled(v, 1.0);
                }
            }
        }
        Ok((overlap, fock))
    }

    fn exchange_components(
        &self,
        density: &[BlochMatrix],
        kernel: CoulombKernel,
    ) -> Result<(Vec<ImageBlock>, Option<Vec<ImageBlock>>)> {
        let cutoff = self
            .options
            .scf
            .exchange_cutoff
            .unwrap_or(self.options.basis_cutoff);
        let mut images = Vec::new();
        let mut imaginary_images = Vec::new();
        for translation in self.geometry.translations_within(2.0 * cutoff)? {
            let mut real = Matrix::zeros(self.basis.nao_sph, self.basis.nao_sph);
            let mut imaginary_block = real.clone();
            let mut imaginary_max = 0.0_f64;
            for i in 0..real.rows {
                for j in 0..real.cols {
                    let mut imaginary = 0.0;
                    for (k, p) in self.k_points().iter().zip(density) {
                        let phase = k.dot(translation);
                        real[(i, j)] += (phase.cos() * p.re[(i, j)] + phase.sin() * p.im[(i, j)])
                            / density.len() as f64;
                        imaginary += (phase.cos() * p.im[(i, j)] - phase.sin() * p.re[(i, j)])
                            / density.len() as f64;
                    }
                    imaginary_max = imaginary_max.max(imaginary.abs());
                    imaginary_block[(i, j)] = imaginary;
                }
            }
            if !self.complex_density && imaginary_max > self.options.operator_tolerance {
                return Err(DftError::InvalidInput(format!(
                    "zero-field localized exchange requires time-reversal paired densities; \
                    inverse Bloch imaginary defect {imaginary_max:e}"
                )));
            }
            images.push(ImageBlock {
                translation,
                matrix: real,
            });
            imaginary_images.push(ImageBlock {
                translation,
                matrix: imaginary_block,
            });
        }
        let real =
            dimensional_exchange_image_blocks(self.basis, self.geometry, &images, cutoff, kernel)?;
        let imaginary = if self.complex_density {
            Some(dimensional_exchange_component_images(
                self.basis,
                self.geometry,
                &imaginary_images,
                cutoff,
                kernel,
                true,
            )?)
        } else {
            None
        };
        Ok((real, imaginary))
    }

    pub(crate) fn exchange(
        &self,
        density: &[BlochMatrix],
        kernel: CoulombKernel,
    ) -> Result<Vec<BlochMatrix>> {
        let (blocks, imaginary) = self.exchange_components(density, kernel)?;
        Ok(self
            .k_points()
            .iter()
            .map(|&k| complex_bloch(&blocks, imaginary.as_deref(), k))
            .collect())
    }

    /// Complete energy and Fock of the declared local/hybrid model at a supplied
    /// density. Analytic contractions are used throughout; no finite differences.
    pub fn evaluate(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<DimensionalElectronicFields> {
        self.evaluate_internal(density, true)
    }
    pub(crate) fn evaluate_without_semilocal(
        &self,
        density: &[Vec<BlochMatrix>; 2],
    ) -> Result<DimensionalElectronicFields> {
        self.evaluate_internal(density, false)
    }
    fn evaluate_internal(
        &self,
        density: &[Vec<BlochMatrix>; 2],
        include_semilocal: bool,
    ) -> Result<DimensionalElectronicFields> {
        let n = self.basis.nao_sph;
        if density.iter().any(|spin| {
            spin.len() != self.k_points().len()
                || spin.iter().any(|p| {
                    p.re.rows != n
                        || p.re.cols != n
                        || p.im.rows != n
                        || p.im.cols != n
                        || p.hermiticity_error() > self.options.operator_tolerance
                        || p.re
                            .as_slice()
                            .iter()
                            .chain(p.im.as_slice())
                            .any(|x| !x.is_finite())
                })
        }) {
            return Err(DftError::InvalidInput("dimensional SCF density must contain finite Hermitian spin/k matrices with the cached AO shape".into()));
        }
        let total: Vec<_> = density[0]
            .iter()
            .zip(&density[1])
            .map(|(a, b)| {
                let mut p = a.clone();
                p.add_scaled(b, 1.0);
                p
            })
            .collect();
        let coulomb = self.coulomb.evaluate(Some(&total))?;
        let hartree = coulomb.hartree.as_ref().expect("density supplied");
        let mut fock: [Vec<_>; 2] = std::array::from_fn(|_| {
            self.h_core
                .iter()
                .zip(hartree)
                .map(|(h, j)| {
                    let mut f = h.clone();
                    f.add_scaled(j, 1.0);
                    f
                })
                .collect()
        });
        let inv_nk = 1.0 / self.k_points().len() as f64;
        let mut exchange_energies = [0.0; 2];
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
            for spin in 0..2 {
                let exchange = self.exchange(&density[spin], kernel)?;
                for ik in 0..self.k_points().len() {
                    fock[spin][ik].add_scaled(&exchange[ik], -coefficient);
                    exchange_energies[component] -=
                        0.5 * coefficient * inv_nk * density[spin][ik].trace_product(&exchange[ik]);
                }
            }
        }
        let xc = match &self.xc {
            Some(backend) if include_semilocal => Some(backend.evaluate(self.functional, density)?),
            Some(backend) if backend.grid().nonlocal_options().is_some() => {
                // Use only the attached density-only VV10 term. The empty
                // local provider leaves the separate exchange operators intact.
                Some(backend.evaluate(&ResolvedFunctional::hartree_fock(), density)?)
            }
            _ => None,
        };
        if let Some(xc) = &xc {
            for spin in 0..2 {
                for ik in 0..self.k_points().len() {
                    fock[spin][ik].add_scaled(&xc.potential[spin][ik], 1.0);
                }
            }
        }
        let kinetic_energy = total
            .iter()
            .zip(&self.kinetic)
            .map(|(p, t)| p.trace_product(t))
            .sum::<f64>()
            * inv_nk;
        let electron_nuclear_energy = coulomb
            .electron_nuclear_components
            .expect("density supplied")
            .iter()
            .sum::<f64>();
        let ecp_energy = if self.ecp_images.is_empty() {
            0.0
        } else {
            total
                .iter()
                .zip(self.k_points())
                .map(|(p, k)| p.trace_product(&BlochMatrix::from_pair(bloch(&self.ecp_images, *k))))
                .sum::<f64>()
                * inv_nk
        };
        let hartree_energy = coulomb
            .hartree_energy_components
            .expect("density supplied")
            .iter()
            .sum::<f64>();
        let semilocal_energy = xc.as_ref().map_or(0.0, |xc| xc.energy);
        let external_field_energy = total
            .iter()
            .zip(&self.external_operators)
            .map(|(p, v)| p.trace_product(v))
            .sum::<f64>()
            * inv_nk;
        let electronic_energy = kinetic_energy
            + electron_nuclear_energy
            + ecp_energy
            + external_field_energy
            + hartree_energy
            + exchange_energies.iter().sum::<f64>()
            + semilocal_energy;
        if !electronic_energy.is_finite()
            || fock
                .iter()
                .flatten()
                .any(|m| m.hermiticity_error() > self.options.operator_tolerance)
        {
            return Err(DftError::InvalidInput(
                "dimensional energy/Fock evaluation is non-finite or non-Hermitian".into(),
            ));
        }
        Ok(DimensionalElectronicFields {
            external_field_energy,
            fock,
            kinetic_energy,
            electron_nuclear_energy,
            ecp_energy,
            hartree_energy,
            full_exchange_energy: exchange_energies[0],
            long_range_exchange_energy: exchange_energies[1],
            semilocal_energy,
            electronic_energy,
            coulomb,
            xc,
        })
    }
}

pub struct DimensionalScfReport {
    pub reference: PeriodicScfResult,
    pub reference_diagnostics: ReferenceDiagnostics,
    /// |P - C f C^dagger| in the independently reconstructed band metric.
    pub occupation_consistency: f64,
    pub kinetic_energy: f64,
    pub electron_nuclear_energy: f64,
    pub ecp_energy: f64,
    pub external_field_energy: f64,
    pub hartree_energy: f64,
    pub full_exchange_energy: f64,
    pub long_range_exchange_energy: f64,
    pub semilocal_component_energies: Vec<f64>,
    pub semilocal_component_labels: Vec<String>,
    pub component_energy_defect: f64,
    pub final_fock_change: f64,
    pub coulomb_operator_defect: f64,
    pub coulomb_derivative_defect: f64,
    pub coulomb_source_defect: f64,
    pub grid_number_defect: f64,
    pub formal_charge: i32,
    pub compensating_charge: f64,
    pub boundary: super::features::PeriodicBoundary,
    pub converged: bool,
    pub model: &'static str,
    pub units: &'static str,
    pub warnings: Vec<String>,
}

/// Common Rust/CLI/native Python/ASE self-consistent dimensional driver.
pub fn run_dimensional_scf(
    mol: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    functional: &ResolvedFunctional,
    options: &DimensionalScfOptions,
) -> Result<DimensionalScfReport> {
    crate::linalg::ensure_parallelism();
    let backend = DimensionalScfBackend::new(mol, basis, geometry, functional, options)?;
    run_cached_dimensional_scf(&backend, false)
}

/// Self-consistent complex reference in a real commensurate periodic field.
pub fn run_dimensional_field_scf(
    mol: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    functional: &ResolvedFunctional,
    options: &DimensionalScfOptions,
    field: &super::magnetic_scf::StaticPeriodicField,
) -> Result<DimensionalScfReport> {
    crate::linalg::ensure_parallelism();
    let backend = DimensionalScfBackend::new(mol, basis, geometry, functional, options)?
        .with_static_field(field)?;
    run_cached_dimensional_scf(&backend, false)
}

/// Orbital-only uniform-field SCF for wires and slabs with a periodic gauge.
pub fn run_dimensional_uniform_scf(
    mol: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    functional: &ResolvedFunctional,
    options: &DimensionalScfOptions,
    field: &super::uniform_open::UniformOpenField,
) -> Result<DimensionalScfReport> {
    crate::linalg::ensure_parallelism();
    let backend = DimensionalScfBackend::new(mol, basis, geometry, functional, options)?
        .with_uniform_open_field(field)?;
    run_cached_dimensional_scf(&backend, false)
}

fn run_cached_dimensional_scf(
    backend: &DimensionalScfBackend<'_>,
    refresh_energy_weighted_density: bool,
) -> Result<DimensionalScfReport> {
    let options = backend.options;
    let mol = backend.mol;
    let basis = backend.basis;
    let geometry = backend.geometry;
    let guess_focks = match options.scf.guess {
        Guess::Core => backend.h_core.clone(),
        Guess::Gwh => gwh_guess_focks(&backend.h_core, &backend.overlap),
        Guess::Sad => {
            let total = crate::scf::sad::sad_density(mol, basis)?;
            let mut spin = BlochMatrix {
                re: total,
                im: Matrix::zeros(basis.nao_sph, basis.nao_sph),
            };
            spin.scale(0.5);
            let seed = vec![spin; backend.k_points().len()];
            backend.evaluate(&[seed.clone(), seed])?.fock[0].clone()
        }
    };
    let mut reference = run_scf_iterations(
        ScfIterationInput {
            k_points: backend.k_points().to_vec(),
            overlap: backend.overlap.clone(),
            h_core: backend.h_core.clone(),
            guess_focks,
            n_alpha: backend.n_alpha,
            n_beta: backend.n_beta,
            restricted: backend.restricted,
            enforce_orbital_conjugation: !backend.complex_density,
            e_nuc: backend.source_energy,
            derivative_plan: None,
        },
        &options.scf,
        |density| {
            let field = backend.evaluate(density)?;
            Ok(Fields {
                fock: field.fock,
                e_elec: field.electronic_energy,
                e_xc: field.semilocal_energy,
                e_nlc: field.xc.as_ref().map_or(0.0, |v| v.nonlocal_energy),
                grid_electron_count: field.xc.as_ref().map(|xc| xc.grid_electron_count),
            })
        },
    )?;
    // Rebuild the actual F[P] at the returned canonical density. Report the
    // original-to-fresh Fock change instead of reusing the SCF stopping flag.
    let field = backend.evaluate(&reference.density)?;
    let final_fock_change = reference
        .fock
        .iter()
        .flatten()
        .zip(field.fock.iter().flatten())
        .map(|(a, b)| a.rms_diff(b))
        .fold(0.0, f64::max);
    reference.fock = field.fock;
    reference.e_elec = field.electronic_energy;
    reference.e_xc = field.semilocal_energy;
    reference.e_nlc = field.xc.as_ref().map_or(0.0, |v| v.nonlocal_energy);
    reference.energy = reference.e_nuc + reference.e_elec;
    reference.free_energy = reference.energy - reference.entropy_term;
    reference.grid_electron_count = field.xc.as_ref().map(|xc| xc.grid_electron_count);
    let orbitals = std::array::from_fn::<_, 2, _>(|spin| {
        reference.fock[spin]
            .iter()
            .zip(&reference.overlap)
            .map(|(f, s)| generalized_orbitals(f, s, options.scf.orth_threshold))
            .collect::<Result<Vec<_>>>()
    });
    let [alpha, beta] = orbitals;
    let orbitals = [alpha?, beta?];
    let (diagnostics, occupation_consistency) =
        crate::dfpt::density::reference_diagnostics(&reference, &orbitals)?;
    if refresh_energy_weighted_density {
        for (spin, bands) in orbitals.iter().enumerate() {
            if bands.iter().any(|b| b.coefficients.cols() != basis.nao_sph) {
                return Err(DftError::Unsupported {
                    feature: "dimensional affine or nuclear derivative with truncated overlap"
                        .into(),
                    missing: "the geometry derivative of the retained overlap projector".into(),
                });
            }
            reference.energy_weighted_density[spin] = bands
                .iter()
                .zip(&reference.occupations[spin])
                .map(|(band, f)| crate::dfpt::orbitals::energy_weighted_band_density(band, f))
                .collect::<Result<_>>()?;
        }
    }
    let component_energy_defect = (reference.energy
        - (reference.e_nuc
            + field.kinetic_energy
            + field.electron_nuclear_energy
            + field.ecp_energy
            + field.external_field_energy
            + field.hartree_energy
            + field.full_exchange_energy
            + field.long_range_exchange_energy
            + field.semilocal_energy))
        .abs();
    let grid_number_defect = field.xc.as_ref().map_or(0.0, |xc| xc.grid_number_defect);
    let converged = reference.converged
        && field.coulomb.converged
        && field.xc.as_ref().is_none_or(|xc| xc.converged)
        && diagnostics.orbital_residual <= options.scf.conv_tol_grad
        && occupation_consistency <= options.scf.p_tol
        && component_energy_defect <= options.operator_tolerance;
    let mut warnings = vec![
        "The collinear orbital reference omits spin Zeeman and SOC. Use the \
        spinor driver for those terms. Uniform normal slab flux and 3D flux \
        require magnetic translations."
            .into(),
        "Converge AO, localized exchange, electrostatic real/reciprocal and \
            open-direction grid cutoffs independently; an SCF stopping flag does \
            not measure truncation error."
            .into(),
        "No isolated-ion monopole correction is added to the compensated total Coulomb energy."
            .into(),
    ];
    if !converged {
        warnings.push("A fresh stationarity, occupation, compensated Coulomb or grid diagnostic exceeds its declared tolerance.".into());
    }
    Ok(DimensionalScfReport { reference, reference_diagnostics: diagnostics, occupation_consistency,
        kinetic_energy: field.kinetic_energy, electron_nuclear_energy: field.electron_nuclear_energy,
        ecp_energy: field.ecp_energy,
        external_field_energy: field.external_field_energy,
        hartree_energy: field.hartree_energy, full_exchange_energy: field.full_exchange_energy,
        long_range_exchange_energy: field.long_range_exchange_energy,
        semilocal_component_energies: field.xc.as_ref().map_or_else(Vec::new, |xc| xc.component_energies.clone()),
        semilocal_component_labels: field.xc.as_ref().map_or_else(Vec::new, |xc| xc.component_labels.clone()),
        component_energy_defect, final_fock_change,
        coulomb_operator_defect: field.coulomb.quadrature_operator_defect,
        coulomb_derivative_defect: field.coulomb.quadrature_derivative_defect,
        coulomb_source_defect: field.coulomb.quadrature_source_defect, grid_number_defect,
        formal_charge: backend.mol.charge, compensating_charge: field.coulomb.compensating_charge,
        boundary: geometry.boundary(), converged,
        model: if backend.uniform_field.is_some() {"Continuum uniform-field dimensional collinear SCF; periodic linear \
            gauge, analytic (p+A)^2/2, complex variational exchange and compensated \
            open-boundary Coulomb; common gauge-origin AO phase; no spin Zeeman or \
            SOC"} else if backend.static_field.is_some() {"Complex dimensional collinear static periodic-field SCF with analytic \
            (p+A)^2/2, scalar potential, complex variational exchange and \
            self-consistent compensated Hartree/XC; no imposed orbital conjugation \
            symmetry"} else {"Zero-field dimensional collinear HF/semilocal/global/range-separated \
            hybrid and local/nonlocal ECP SCF; compensated dimensional Coulomb; \
            mesh-global insulating or Fermi-Dirac occupations"},
        units: "energies/Fock: Hartree per cell; geometry/cutoffs: Bohr; k: inverse Bohr; temperature: kelvin; counts/metric defects: dimensionless",
        warnings })
}

fn complex_bloch(
    real: &[ImageBlock],
    imaginary: Option<&[ImageBlock]>,
    k: crate::math::Vec3,
) -> BlochMatrix {
    let mut out = BlochMatrix::from_pair(bloch(real, k));
    if let Some(images) = imaginary {
        let (re, im) = bloch(images, k);
        for i in 0..out.rows() {
            for j in 0..out.cols() {
                out.re[(i, j)] -= im[(i, j)];
                out.im[(i, j)] += re[(i, j)];
            }
        }
    }
    out
}
