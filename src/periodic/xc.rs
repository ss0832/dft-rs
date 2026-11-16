// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic LDA/GGA/meta-GGA quadrature in a Bloch AO basis.
//!
//! The density is formed from every k point,
//! `rho(r) = Nk^-1 sum_k,mu,nu P_mu,nu(k) Phi_mu,k(r) Phi_nu,k(r)*`.
//! The same Bloch functions are then used to assemble `Vxc(k)`.  Keeping
//! both operations on this path is essential: evaluating a Gamma-point
//! image density and copying its potential to every k point is not the
//! functional derivative of a finite-k total energy.

use crate::basis::{n_sph, MolBasis};
use crate::error::{DftError, Result};
use crate::grid::eval::{eval_aos_full, AoBatch};
use crate::grid::GridOptions;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::grid::{AngularGridFrame, PeriodicGrid};
use crate::periodic::integrals::{nearby_translation_indices, shifted};
use crate::periodic::Cell;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Highest spatial derivative of a Bloch AO retained on the periodic XC grid.
///
/// Keeping this typed avoids the former `want_grad` ambiguity, where asking
/// for first derivatives also allocated and evaluated all six Hessian
/// components.  Lower-order data are always present: `Hessian` includes
/// gradients and values, and `Gradient` includes values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AoDerivativeOrder {
    Value,
    Gradient,
    Hessian,
}

impl AoDerivativeOrder {
    /// AO data needed to evaluate an SCF XC energy and potential.
    pub fn for_scf(family: Family) -> Self {
        match family {
            Family::Lda => Self::Value,
            Family::Gga | Family::MetaGga => Self::Gradient,
        }
    }

    /// AO data needed by the analytic periodic XC nuclear gradient.
    pub fn for_forces(family: Family) -> Self {
        match family {
            Family::Lda => Self::Gradient,
            Family::Gga | Family::MetaGga => Self::Hessian,
        }
    }

    fn wants_gradient(self) -> bool {
        self >= Self::Gradient
    }

    fn wants_hessian(self) -> bool {
        self >= Self::Hessian
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BlochAos {
    pub(crate) phi: BlochMatrix,
    pub(crate) grad: Option<[BlochMatrix; 3]>,
    pub(crate) hess: Option<[BlochMatrix; 6]>,
}

impl BlochAos {
    fn zeros(npts: usize, nao: usize, derivative_order: AoDerivativeOrder) -> Self {
        Self {
            phi: BlochMatrix::zeros(npts, nao),
            grad: derivative_order
                .wants_gradient()
                .then(|| std::array::from_fn(|_| BlochMatrix::zeros(npts, nao))),
            hess: derivative_order
                .wants_hessian()
                .then(|| std::array::from_fn(|_| BlochMatrix::zeros(npts, nao))),
        }
    }

    fn add_screened_image(
        &mut self,
        values: &AoBatch,
        basis: &MolBasis,
        point_shells: &[(usize, usize)],
        cosine: f64,
        sine: f64,
    ) {
        for &(point, shell_index) in point_shells {
            let shell = &basis.shells[shell_index];
            for mu in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                let value = values.phi[(point, mu)];
                self.phi.re[(point, mu)] += cosine * value;
                self.phi.im[(point, mu)] += sine * value;
            }
        }
        if let (Some(target), Some(source)) = (self.grad.as_mut(), values.grad.as_ref()) {
            for &(point, shell_index) in point_shells {
                let shell = &basis.shells[shell_index];
                for mu in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                    for axis in 0..3 {
                        let value = source[axis][(point, mu)];
                        target[axis].re[(point, mu)] += cosine * value;
                        target[axis].im[(point, mu)] += sine * value;
                    }
                }
            }
        }
        if let (Some(target), Some(source)) = (self.hess.as_mut(), values.hess.as_ref()) {
            for &(point, shell_index) in point_shells {
                let shell = &basis.shells[shell_index];
                for mu in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                    for component in 0..6 {
                        let value = source[component][(point, mu)];
                        target[component].re[(point, mu)] += cosine * value;
                        target[component].im[(point, mu)] += sine * value;
                    }
                }
            }
        }
    }
}

/// One lattice translation together with the point/shell pairs whose actual
/// distance is inside the periodic AO cutoff.
///
/// A translation cannot be screened by `|T|` alone: a shell near one cell
/// face can be close to a point near the opposite face even when the needed
/// lattice vector is much longer than the cutoff. Retaining the pair list
/// also prevents a translation selected for one shell from leaking every
/// other shell into the Bloch sum.
#[derive(Clone, Debug)]
pub(crate) struct XcAoImage {
    pub(crate) translation: Vec3,
    pub(crate) point_shells: Vec<(usize, usize)>,
}

pub(crate) fn xc_ao_images(
    basis: &MolBasis,
    lattice: &crate::periodic::Lattice,
    points: &[Vec3],
    cutoff: f64,
) -> Vec<XcAoImage> {
    let mut by_translation: BTreeMap<[i64; 3], Vec<(usize, usize)>> = BTreeMap::new();
    for (point_index, point) in points.iter().enumerate() {
        for (shell_index, shell) in basis.shells.iter().enumerate() {
            let displacement = shell.center - *point;
            for index in nearby_translation_indices(lattice, displacement, cutoff) {
                by_translation
                    .entry(index)
                    .or_default()
                    .push((point_index, shell_index));
            }
        }
    }

    let mut images: Vec<XcAoImage> = by_translation
        .into_iter()
        .map(|(index, point_shells)| XcAoImage {
            translation: lattice.translation(index),
            point_shells,
        })
        .collect();
    images.sort_by(|a, b| {
        a.translation
            .norm2()
            .total_cmp(&b.translation.norm2())
            .then_with(|| a.translation.x.total_cmp(&b.translation.x))
            .then_with(|| a.translation.y.total_cmp(&b.translation.y))
            .then_with(|| a.translation.z.total_cmp(&b.translation.z))
    });
    images
}

fn shifted_active_basis(basis: &MolBasis, image: &XcAoImage) -> MolBasis {
    let mut active = vec![false; basis.shells.len()];
    for &(_, shell_index) in &image.point_shells {
        active[shell_index] = true;
    }
    let mut image_basis = shifted(basis, image.translation);
    image_basis.shells = image_basis
        .shells
        .into_iter()
        .enumerate()
        .filter_map(|(shell_index, shell)| active[shell_index].then_some(shell))
        .collect();
    image_basis
}

pub(super) fn build_bloch_aos_on_points(
    cell: &Cell,
    k_points: &[Vec3],
    points: &[Vec3],
    ao_cutoff: f64,
    derivative_order: AoDerivativeOrder,
) -> Vec<BlochAos> {
    const POINT_BATCH: usize = 64;
    if points.len() <= POINT_BATCH {
        let images = xc_ao_images(&cell.basis, &cell.lattice, points, ao_cutoff);
        return build_bloch_aos_from_images(
            &cell.basis,
            k_points,
            points,
            &images,
            derivative_order,
        );
    }
    // Keep the returned AO arrays, but bound the temporary point/image
    // selection lists and per-image evaluation batches. Every point sees
    // the same translated shells in the same order as the complete build.
    let nao = cell.nao();
    let mut output: Vec<BlochAos> = k_points
        .iter()
        .map(|_| BlochAos::zeros(points.len(), nao, derivative_order))
        .collect();
    for (batch, local_points) in points.chunks(POINT_BATCH).enumerate() {
        let start = batch * POINT_BATCH * nao;
        let end = start + local_points.len() * nao;
        let images = xc_ao_images(&cell.basis, &cell.lattice, local_points, ao_cutoff);
        let local = build_bloch_aos_from_images(
            &cell.basis,
            k_points,
            local_points,
            &images,
            derivative_order,
        );
        let copy = |target: &mut BlochMatrix, source: &BlochMatrix| {
            target.re.as_mut_slice()[start..end].copy_from_slice(source.re.as_slice());
            target.im.as_mut_slice()[start..end].copy_from_slice(source.im.as_slice());
        };
        for (target, source) in output.iter_mut().zip(local) {
            copy(&mut target.phi, &source.phi);
            if let (Some(left), Some(right)) = (target.grad.as_mut(), source.grad.as_ref()) {
                for axis in 0..3 {
                    copy(&mut left[axis], &right[axis]);
                }
            }
            if let (Some(left), Some(right)) = (target.hess.as_mut(), source.hess.as_ref()) {
                for axis in 0..6 {
                    copy(&mut left[axis], &right[axis]);
                }
            }
        }
    }
    output
}

fn build_bloch_aos_from_images(
    basis: &MolBasis,
    k_points: &[Vec3],
    points: &[Vec3],
    images: &[XcAoImage],
    derivative_order: AoDerivativeOrder,
) -> Vec<BlochAos> {
    let want_grad = derivative_order.wants_gradient();
    let want_hess = derivative_order.wants_hessian();
    let mut aos: Vec<BlochAos> = k_points
        .iter()
        .map(|_| BlochAos::zeros(points.len(), basis.nao_sph, derivative_order))
        .collect();

    // Evaluate every translated shell image once, then reuse it for all k
    // points. Only shells selected somewhere for this translation are
    // evaluated, and only exact point/shell blocks inside the cutoff are
    // accumulated.
    for image in images {
        let image_basis = shifted_active_basis(basis, image);
        let values = eval_aos_full(&image_basis, points, want_grad, want_hess);
        aos.par_iter_mut()
            .zip(k_points.par_iter())
            .for_each(|(target, k)| {
                let phase = k.dot(image.translation);
                target.add_screened_image(
                    &values,
                    basis,
                    &image.point_shells,
                    phase.cos(),
                    phase.sin(),
                );
            });
    }
    aos
}

/// Bloch sums of AO derivatives weighted by the point-to-centre vector
/// `u = r - A - T`: `grad_u[a][b] = sum_T exp(ik.T) d_a phi(u) u_b` and, when
/// requested, `hess_u[ac][b] = sum_T exp(ik.T) d_a d_c phi(u) u_b` with the
/// Hessian components ordered (xx, xy, xz, yy, yz, zz). These are the
/// strain derivatives of the rigidly moving Bloch AOs at a strained point.
#[derive(Clone, Debug)]
pub(crate) struct BlochAoMoments {
    pub(crate) grad_u: [[BlochMatrix; 3]; 3],
    pub(crate) hess_u: Option<[[BlochMatrix; 3]; 6]>,
}

/// [`BlochAoMoments`] on `points` for every k point, with the same
/// point/shell screening as the XC Bloch AOs.
fn bloch_ao_moments_from_images(
    basis: &MolBasis,
    images: &[XcAoImage],
    k_points: &[Vec3],
    points: &[Vec3],
    want_hessian: bool,
) -> Vec<BlochAoMoments> {
    let (npts, nao) = (points.len(), basis.nao_sph);
    let mut out: Vec<BlochAoMoments> = k_points
        .iter()
        .map(|_| BlochAoMoments {
            grad_u: std::array::from_fn(|_| std::array::from_fn(|_| BlochMatrix::zeros(npts, nao))),
            hess_u: want_hessian.then(|| {
                std::array::from_fn(|_| std::array::from_fn(|_| BlochMatrix::zeros(npts, nao)))
            }),
        })
        .collect();
    for image in images {
        let image_basis = shifted_active_basis(basis, image);
        let values = eval_aos_full(&image_basis, points, true, want_hessian);
        let grad = values.grad.as_ref().expect("AO gradients were requested");
        out.par_iter_mut()
            .zip(k_points.par_iter())
            .for_each(|(target, k)| {
                let phase = k.dot(image.translation);
                let (cosine, sine) = (phase.cos(), phase.sin());
                for &(point, shell_index) in &image.point_shells {
                    let shell = &basis.shells[shell_index];
                    let u = points[point] - (shell.center + image.translation);
                    for mu in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                        for a in 0..3 {
                            let value = grad[a][(point, mu)];
                            for b in 0..3 {
                                let weighted = value * u.get(b);
                                target.grad_u[a][b].re[(point, mu)] += cosine * weighted;
                                target.grad_u[a][b].im[(point, mu)] += sine * weighted;
                            }
                        }
                        if let (Some(hess_u), Some(hess)) =
                            (target.hess_u.as_mut(), values.hess.as_ref())
                        {
                            for component in 0..6 {
                                let value = hess[component][(point, mu)];
                                for b in 0..3 {
                                    let weighted = value * u.get(b);
                                    hess_u[component][b].re[(point, mu)] += cosine * weighted;
                                    hess_u[component][b].im[(point, mu)] += sine * weighted;
                                }
                            }
                        }
                    }
                }
            });
    }
    out
}

pub(crate) fn bloch_ao_moments(
    cell: &Cell,
    k_points: &[Vec3],
    points: &[Vec3],
    ao_cutoff: f64,
    want_hessian: bool,
) -> Vec<BlochAoMoments> {
    let images = xc_ao_images(&cell.basis, &cell.lattice, points, ao_cutoff);
    bloch_ao_moments_from_images(&cell.basis, &images, k_points, points, want_hessian)
}
pub(crate) fn dimensional_bloch_ao_moments(
    basis: &MolBasis,
    geometry: &super::PeriodicGeometry,
    k_points: &[Vec3],
    points: &[Vec3],
    cutoff: f64,
    want_hessian: bool,
) -> Result<Vec<BlochAoMoments>> {
    let images = dimensional_xc_ao_images(basis, geometry, points, cutoff)?;
    Ok(bloch_ao_moments_from_images(
        basis,
        &images,
        k_points,
        points,
        want_hessian,
    ))
}
/// Geometry- and k-mesh-dependent part of periodic XC quadrature.
///
/// Construct once and reuse through all SCF iterations.  This is both much
/// faster and safer than rebuilding translated AO values in the nonlinear
/// SCF loop.
#[derive(Clone, Debug)]
pub struct PeriodicXcGrid {
    pub(crate) vv10: Option<(
        super::PeriodicGeometry,
        super::nonlocal::PeriodicVv10Options,
    )>,
    pub points: Vec<Vec3>,
    pub weights: Vec<f64>,
    pub(crate) raw_weights: Vec<f64>,
    pub(crate) atom_of: Vec<usize>,
    pub k_points: Vec<Vec3>,
    pub(crate) aos: Vec<BlochAos>,
    pub(crate) becke_shell: i64,
    pub(crate) angular_frame: Option<AngularGridFrame>,
    derivative_order: AoDerivativeOrder,
    /// Learned functional evaluated on this grid in addition to the
    /// semilocal parts.
    pub(crate) learned: Option<LearnedGrid>,
}

/// A learned functional attached to a periodic grid: the model and the
/// positions of the atoms that own the grid points. The non-local layers of
/// the model sum over the atomic grid of each atom inside the radius of the
/// periodic partition.
#[derive(Clone, Debug)]
pub(crate) struct LearnedGrid {
    pub(crate) model: std::sync::Arc<crate::skala::SkalaModel>,
    pub(crate) centers: Vec<[f64; 3]>,
}

impl LearnedGrid {
    /// Model output for the densities at the points of `grid`; the density
    /// gradients are stored `[spin][axis][point]`.
    pub(crate) fn evaluate(
        &self,
        grid: &PeriodicXcGrid,
        rho: &[Vec<f64>; 2],
        gradient: &[[Vec<f64>; 3]; 2],
        tau: &[Vec<f64>; 2],
    ) -> Result<crate::skala::SkalaOutput> {
        let npts = grid.len();
        let points: Vec<[f64; 3]> = grid.points.iter().map(|p| p.to_array()).collect();
        let vectors: [Vec<[f64; 3]>; 2] = std::array::from_fn(|spin| {
            (0..npts)
                .map(|point| {
                    [
                        gradient[spin][0][point],
                        gradient[spin][1][point],
                        gradient[spin][2][point],
                    ]
                })
                .collect()
        });
        self.model.evaluate(
            &crate::skala::SkalaInput {
                atom_of: &grid.atom_of,
                points: &points,
                centers: &self.centers,
                weights: &grid.weights,
                atomic_weights: &grid.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&vectors[0], &vectors[1]],
                tau: [&tau[0], &tau[1]],
            },
            true,
        )
    }
}

/// Periodic XC energy, one potential matrix per spin and k point, and the
/// quadrature electron-count diagnostic.
pub struct PeriodicXcContribution {
    pub exc: f64,
    /// Included in `exc`, retained separately for term-by-term validation.
    pub nonlocal_energy: f64,
    pub vxc: [Vec<BlochMatrix>; 2],
    pub n_elec: f64,
}

pub(crate) struct PeriodicXcEvaluation {
    pub contribution: PeriodicXcContribution,
    /// Raw assembly measurement before any Hermitian roundoff projection.
    pub hermiticity_defect: f64,
    /// Weighted semilocal part energies, when explicitly requested.
    pub part_energies: Option<Vec<f64>>,
}

impl PeriodicXcGrid {
    /// One periodic cell in the active subspace and all open directions.
    /// Physical cutoffs are independent of unused open cell vectors.
    pub fn build_dimensional(
        molecule: &crate::system::Molecule,
        basis: &MolBasis,
        geometry: &super::PeriodicGeometry,
        k_points: &[Vec3],
        opts: &GridOptions,
        grid_cutoff: f64,
        ao_cutoff: f64,
        derivative_order: AoDerivativeOrder,
    ) -> Result<Self> {
        Self::build_dimensional_oriented(
            molecule,
            basis,
            geometry,
            k_points,
            opts,
            grid_cutoff,
            ao_cutoff,
            derivative_order,
            None,
        )
    }

    pub(crate) fn build_dimensional_oriented(
        molecule: &crate::system::Molecule,
        basis: &MolBasis,
        geometry: &super::PeriodicGeometry,
        k_points: &[Vec3],
        opts: &GridOptions,
        grid_cutoff: f64,
        ao_cutoff: f64,
        derivative_order: AoDerivativeOrder,
        angular_frame: Option<AngularGridFrame>,
    ) -> Result<Self> {
        let grid = PeriodicGrid::build_dimensional_oriented(
            molecule,
            geometry,
            opts,
            grid_cutoff,
            angular_frame.as_ref(),
        )?;
        let aos = build_dimensional_bloch_aos(
            basis,
            geometry,
            k_points,
            &grid.points,
            ao_cutoff,
            derivative_order,
        )?;
        Ok(Self {
            points: grid.points,
            weights: grid.weights,
            raw_weights: grid.raw_weights,
            atom_of: grid.atom_of,
            k_points: k_points.to_vec(),
            aos,
            becke_shell: 0,
            angular_frame,
            derivative_order,
            learned: None,
            vv10: None,
        })
    }

    /// Reuse the reference open-direction quadrature on a shifted Bloch mesh.
    pub fn at_dimensional_k_points(
        &self,
        basis: &MolBasis,
        geometry: &super::PeriodicGeometry,
        k_points: &[Vec3],
        ao_cutoff: f64,
    ) -> Result<Self> {
        let mut target = self.clone();
        target.aos = build_dimensional_bloch_aos(
            basis,
            geometry,
            k_points,
            &self.points,
            ao_cutoff,
            self.derivative_order,
        )?;
        target.k_points = k_points.to_vec();
        Ok(target)
    }

    /// Build Bloch AOs on a periodic atom-centred grid.
    ///
    /// `ao_cutoff` controls the localized lattice sum defining each Bloch
    /// AO; `becke_shell` controls the physical neighbourhood used by the
    /// periodic Becke partition.  Both are convergence parameters in Bohr
    /// (the latter through the cell-volume-scaled shell definition).
    pub fn build(
        cell: &Cell,
        k_points: &[Vec3],
        opts: &GridOptions,
        becke_shell: i64,
        ao_cutoff: f64,
        derivative_order: AoDerivativeOrder,
    ) -> Result<Self> {
        if k_points.is_empty() {
            return Err(DftError::InvalidInput(
                "periodic XC needs at least one k point".to_string(),
            ));
        }
        if !(ao_cutoff.is_finite() && ao_cutoff > 0.0) {
            return Err(DftError::InvalidInput(
                "periodic XC AO cutoff must be finite and positive".to_string(),
            ));
        }
        let grid = PeriodicGrid::build(&cell.molecule, &cell.lattice, opts, becke_shell)?;
        let aos =
            build_bloch_aos_on_points(cell, k_points, &grid.points, ao_cutoff, derivative_order);

        Ok(Self {
            points: grid.points,
            weights: grid.weights,
            raw_weights: grid.raw_weights,
            atom_of: grid.atom_of,
            k_points: k_points.to_vec(),
            aos,
            becke_shell,
            angular_frame: None,
            derivative_order,
            learned: None,
            vv10: None,
        })
    }

    /// Attach the explicit periodic pair operator to the same Bloch grid.
    pub fn with_vv10(
        self,
        cell: &Cell,
        settings: super::nonlocal::PeriodicVv10Options,
    ) -> Result<Self> {
        self.with_dimensional_vv10(
            super::PeriodicGeometry::new(
                cell.lattice().clone(),
                super::features::PeriodicBoundary::FULL_3D,
            )?,
            settings,
        )
    }

    pub fn with_dimensional_vv10(
        mut self,
        geometry: super::PeriodicGeometry,
        settings: super::nonlocal::PeriodicVv10Options,
    ) -> Result<Self> {
        settings.validate()?;
        if self.derivative_order < AoDerivativeOrder::Gradient {
            return Err(DftError::InvalidInput(
                "periodic VV10 requires Bloch AO gradients".into(),
            ));
        }
        self.vv10 = Some((geometry, settings));
        Ok(self)
    }

    pub fn nonlocal_options(&self) -> Option<super::nonlocal::PeriodicVv10Options> {
        self.vv10.as_ref().map(|v| v.1)
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The same grid with a learned functional evaluated on it.
    pub fn with_learned(
        mut self,
        model: std::sync::Arc<crate::skala::SkalaModel>,
        cell: &Cell,
    ) -> Self {
        let centers = cell
            .molecule
            .atoms
            .iter()
            .map(|atom| atom.position.to_array())
            .collect();
        self.learned = Some(LearnedGrid { model, centers });
        self
    }
    pub fn derivative_order(&self) -> AoDerivativeOrder {
        self.derivative_order
    }

    /// Evaluate a different Bloch mesh on exactly the same quadrature.
    /// The reference density and partition weights are not recomputed.
    pub fn at_k_points(&self, cell: &Cell, k_points: &[Vec3], ao_cutoff: f64) -> Result<Self> {
        if k_points.is_empty()
            || k_points
                .iter()
                .any(|k| !k.x.is_finite() || !k.y.is_finite() || !k.z.is_finite())
            || !ao_cutoff.is_finite()
            || ao_cutoff <= 0.0
        {
            return Err(DftError::InvalidInput(
                "shifted Bloch grid requires finite k points and a positive AO cutoff".into(),
            ));
        }
        let mut shifted = self.clone();
        shifted.k_points = k_points.to_vec();
        shifted.aos = build_bloch_aos_on_points(
            cell,
            k_points,
            &self.points,
            ao_cutoff,
            self.derivative_order,
        );
        Ok(shifted)
    }
}

#[inline]
fn product(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
    (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
}

#[inline]
fn conj_product(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
    (a_re * b_re + a_im * b_im, a_re * b_im - a_im * b_re)
}

fn validate_density(grid: &PeriodicXcGrid, density: &[Vec<BlochMatrix>; 2]) -> Result<usize> {
    let nk = grid.k_points.len();
    if density[0].len() != nk || density[1].len() != nk {
        return Err(DftError::InvalidInput(
            "periodic XC needs one alpha and beta density per k point".to_string(),
        ));
    }
    let nao = grid.aos.first().map(|a| a.phi.cols()).unwrap_or(0);
    for (spin, matrices) in density.iter().enumerate() {
        for (ik, matrix) in matrices.iter().enumerate() {
            if matrix.rows() != nao
                || matrix.cols() != nao
                || matrix.im.rows != nao
                || matrix.im.cols != nao
                || matrix
                    .re
                    .as_slice()
                    .iter()
                    .chain(matrix.im.as_slice())
                    .any(|v| !v.is_finite())
            {
                return Err(DftError::InvalidInput(format!(
                    "periodic XC density spin {spin}, k point {ik} is not {nao}x{nao}"
                )));
            }
            let error = matrix.hermiticity_error();
            if error > 1e-9 {
                return Err(DftError::InvalidInput(format!(
                    "periodic XC density spin {spin}, k point {ik} is not Hermitian (error {error:e})"
                )));
            }
        }
    }
    Ok(nao)
}

/// Evaluate periodic `Exc` and its k-resolved spin potentials.
pub fn xc_contribution(
    functional: &ResolvedFunctional,
    grid: &PeriodicXcGrid,
    density: &[Vec<BlochMatrix>; 2],
) -> Result<PeriodicXcContribution> {
    xc_contribution_at_grid(functional, grid, density, grid)
}

/// Assemble a ground-state XC potential on another mesh from the original
/// density. Both grids must contain identical quadrature points and weights.
/// This is interpolation of the same self-consistent reference, not an SCF
/// calculation on a shifted integration mesh.
pub fn xc_contribution_at_grid(
    functional: &ResolvedFunctional,
    grid: &PeriodicXcGrid,
    density: &[Vec<BlochMatrix>; 2],
    target: &PeriodicXcGrid,
) -> Result<PeriodicXcContribution> {
    Ok(xc_contribution_at_grid_impl(functional, grid, density, target, false)?.contribution)
}

pub(crate) fn xc_contribution_with_components(
    functional: &ResolvedFunctional,
    grid: &PeriodicXcGrid,
    density: &[Vec<BlochMatrix>; 2],
) -> Result<PeriodicXcEvaluation> {
    xc_contribution_at_grid_impl(functional, grid, density, grid, true)
}

fn xc_contribution_at_grid_impl(
    functional: &ResolvedFunctional,
    grid: &PeriodicXcGrid,
    density: &[Vec<BlochMatrix>; 2],
    target: &PeriodicXcGrid,
    components: bool,
) -> Result<PeriodicXcEvaluation> {
    if grid.points != target.points
        || grid.weights != target.weights
        || target.derivative_order != grid.derivative_order
        || target.aos.is_empty()
        || target
            .aos
            .iter()
            .any(|a| a.phi.rows() != grid.len() || a.phi.cols() != grid.aos[0].phi.cols())
    {
        return Err(DftError::InvalidInput(
            "target XC mesh must use the same reference quadrature and AO space".into(),
        ));
    }
    let required_order =
        AoDerivativeOrder::for_scf(functional.family).max(if grid.vv10.is_some() {
            AoDerivativeOrder::Gradient
        } else {
            AoDerivativeOrder::Value
        });
    if grid.derivative_order < required_order {
        return Err(DftError::InvalidInput(format!(
            "periodic {:?} XC requires {:?} Bloch-AO data, but the grid contains only {:?}",
            functional.family, required_order, grid.derivative_order
        )));
    }
    let nao = validate_density(grid, density)?;
    let nk = grid.k_points.len();
    let inv_nk = 1.0 / nk as f64;
    let npts = grid.len();
    let gga = functional.family != Family::Lda;
    let gradient_needed = gga || grid.vv10.is_some();
    let meta = functional.family == Family::MetaGga;
    if gradient_needed && grid.aos.iter().any(|a| a.grad.is_none()) {
        return Err(DftError::InvalidInput(
            "GGA/meta-GGA periodic XC grid was built without AO gradients".to_string(),
        ));
    }

    let mut rho = [vec![0.0; npts], vec![0.0; npts]];
    let mut grad: [[Vec<f64>; 3]; 2] = std::array::from_fn(|_| {
        std::array::from_fn(|_| {
            if gradient_needed {
                vec![0.0; npts]
            } else {
                Vec::new()
            }
        })
    });
    let mut tau = [
        if meta { vec![0.0; npts] } else { Vec::new() },
        if meta { vec![0.0; npts] } else { Vec::new() },
    ];

    for spin in 0..2 {
        for (ik, aos) in grid.aos.iter().enumerate() {
            let p = &density[spin][ik];
            for point in 0..npts {
                let mut value = 0.0;
                let mut gradient = [0.0; 3];
                let mut kinetic = 0.0;
                for mu in 0..nao {
                    let ar = aos.phi.re[(point, mu)];
                    let ai = aos.phi.im[(point, mu)];
                    for nu in 0..nao {
                        // P_mu,nu Phi_mu Phi_nu*.
                        let br = aos.phi.re[(point, nu)];
                        let bi = -aos.phi.im[(point, nu)];
                        let (qr, qi) = product(ar, ai, br, bi);
                        value += p.re[(mu, nu)] * qr - p.im[(mu, nu)] * qi;
                        if let Some(g) = &aos.grad {
                            for axis in 0..3 {
                                let (left_r, left_i) = product(
                                    g[axis].re[(point, mu)],
                                    g[axis].im[(point, mu)],
                                    br,
                                    bi,
                                );
                                let (right_r, right_i) = product(
                                    ar,
                                    ai,
                                    g[axis].re[(point, nu)],
                                    -g[axis].im[(point, nu)],
                                );
                                gradient[axis] += p.re[(mu, nu)] * (left_r + right_r)
                                    - p.im[(mu, nu)] * (left_i + right_i);
                                if meta {
                                    let (tr, ti) = product(
                                        g[axis].re[(point, mu)],
                                        g[axis].im[(point, mu)],
                                        g[axis].re[(point, nu)],
                                        -g[axis].im[(point, nu)],
                                    );
                                    kinetic += 0.5 * (p.re[(mu, nu)] * tr - p.im[(mu, nu)] * ti);
                                }
                            }
                        }
                    }
                }
                rho[spin][point] += inv_nk * value;
                if gradient_needed {
                    for axis in 0..3 {
                        grad[spin][axis][point] += inv_nk * gradient[axis];
                    }
                }
                if meta {
                    tau[spin][point] += inv_nk * kinetic;
                }
            }
        }
        // Positive-semidefinite P gives rho >= 0.  Remove only quadrature
        // roundoff before calling functionals with fractional powers.
        for value in &mut rho[spin] {
            if *value < 0.0 && *value > -1e-12 {
                *value = 0.0;
            }
        }
    }

    let sigma_store;
    let vars = if gga {
        let mut saa = vec![0.0; npts];
        let mut sab = vec![0.0; npts];
        let mut sbb = vec![0.0; npts];
        for point in 0..npts {
            for axis in 0..3 {
                saa[point] += grad[0][axis][point] * grad[0][axis][point];
                sab[point] += grad[0][axis][point] * grad[1][axis][point];
                sbb[point] += grad[1][axis][point] * grad[1][axis][point];
            }
        }
        sigma_store = [saa, sab, sbb];
        GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
            tau: meta.then_some([&tau[0], &tau[1]]),
        }
    } else {
        GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: None,
            tau: None,
        }
    };
    let semilocal = !functional.parts.is_empty();
    let mut out = XcOut::default();
    if semilocal {
        functional.eval(&vars, 1, &mut out)?;
    }
    let part_energies = if components {
        let mut energies = Vec::with_capacity(functional.parts.len());
        for (weight, part) in &functional.parts {
            let mut output = XcOut::default();
            output.prepare(npts, 0, functional.family);
            part.eval_add(&vars, 0, *weight, &mut output)?;
            energies.push(output.e.iter().zip(&grid.weights).map(|(e, w)| e * w).sum());
        }
        Some(energies)
    } else {
        None
    };
    let mut exc = 0.0;
    let mut n_elec = 0.0;
    for point in 0..npts {
        if semilocal {
            exc += grid.weights[point] * out.e[point];
        }
        n_elec += grid.weights[point] * (rho[0][point] + rho[1][point]);
    }
    // Derivatives of the energy with respect to the densities at the grid
    // points, quadrature weights included.
    let mut d_rho = [vec![0.0; npts], vec![0.0; npts]];
    let mut d_gradient = [vec![[0.0_f64; 3]; npts], vec![[0.0_f64; 3]; npts]];
    let mut d_tau = [vec![0.0; npts], vec![0.0; npts]];
    if semilocal {
        for spin in 0..2 {
            let other = 1 - spin;
            for point in 0..npts {
                let weight = grid.weights[point];
                d_rho[spin][point] = weight * out.vrho[spin][point];
                if gga {
                    let (own, cross) = if spin == 0 {
                        (&out.vsigma[0], &out.vsigma[1])
                    } else {
                        (&out.vsigma[2], &out.vsigma[1])
                    };
                    for axis in 0..3 {
                        d_gradient[spin][point][axis] = weight
                            * (2.0 * own[point] * grad[spin][axis][point]
                                + cross[point] * grad[other][axis][point]);
                    }
                }
                if meta {
                    d_tau[spin][point] = weight * out.vtau[spin][point];
                }
            }
        }
    }
    let mut nonlocal_energy = 0.0;
    if let Some((geometry, settings)) = &grid.vv10 {
        let total: Vec<_> = (0..npts).map(|i| rho[0][i] + rho[1][i]).collect();
        let gradient: Vec<[f64; 3]> = (0..npts)
            .map(|i| std::array::from_fn(|a| grad[0][a][i] + grad[1][a][i]))
            .collect();
        let sigma: Vec<_> = gradient
            .iter()
            .map(|g| g.iter().map(|v| v * v).sum())
            .collect();
        let points: Vec<_> = grid.points.iter().map(|p| p.to_array()).collect();
        let fields = super::nonlocal::vv10_fields(
            geometry,
            &grid.points,
            &points,
            geometry.lattice().a.map(|a| a.to_array()),
            &grid.weights,
            &total,
            &sigma,
            *settings,
        )?;
        exc += fields.energy;
        nonlocal_energy = fields.energy;
        for spin in 0..2 {
            for point in 0..npts {
                d_rho[spin][point] += fields.d_rho[point];
                for a in 0..3 {
                    d_gradient[spin][point][a] += 2.0 * fields.d_sigma[point] * gradient[point][a];
                }
            }
        }
    }
    if let Some(learned) = &grid.learned {
        if !(gga && meta) {
            return Err(DftError::InvalidInput(
                "the learned functional needs density gradients and kinetic energy densities on the grid"
                    .into(),
            ));
        }
        let result = learned.evaluate(grid, &rho, &grad, &tau)?;
        exc += result.energy;
        for spin in 0..2 {
            for point in 0..npts {
                d_rho[spin][point] += result.d_rho[spin][point];
                d_tau[spin][point] += result.d_tau[spin][point];
                for axis in 0..3 {
                    d_gradient[spin][point][axis] += result.d_gradient[spin][point][axis];
                }
            }
        }
    }

    let target_nk = target.k_points.len();
    let mut vxc: [Vec<BlochMatrix>; 2] = [
        (0..target_nk)
            .map(|_| BlochMatrix::zeros(nao, nao))
            .collect(),
        (0..target_nk)
            .map(|_| BlochMatrix::zeros(nao, nao))
            .collect(),
    ];
    let mut hermiticity_defect = 0.0_f64;
    for spin in 0..2 {
        for (ik, aos) in target.aos.iter().enumerate() {
            let matrix = &mut vxc[spin][ik];
            for point in 0..npts {
                let b = d_gradient[spin][point];
                for mu in 0..nao {
                    for nu in 0..nao {
                        // V_mu,nu = integral Phi_mu* v Phi_nu.
                        let mut term = conj_product(
                            aos.phi.re[(point, mu)],
                            aos.phi.im[(point, mu)],
                            aos.phi.re[(point, nu)],
                            aos.phi.im[(point, nu)],
                        );
                        let local = d_rho[spin][point];
                        term.0 *= local;
                        term.1 *= local;
                        if let Some(g) = &aos.grad {
                            for axis in 0..3 {
                                let left = conj_product(
                                    g[axis].re[(point, mu)],
                                    g[axis].im[(point, mu)],
                                    aos.phi.re[(point, nu)],
                                    aos.phi.im[(point, nu)],
                                );
                                let right = conj_product(
                                    aos.phi.re[(point, mu)],
                                    aos.phi.im[(point, mu)],
                                    g[axis].re[(point, nu)],
                                    g[axis].im[(point, nu)],
                                );
                                term.0 += b[axis] * (left.0 + right.0);
                                term.1 += b[axis] * (left.1 + right.1);
                                if meta {
                                    let kinetic = conj_product(
                                        g[axis].re[(point, mu)],
                                        g[axis].im[(point, mu)],
                                        g[axis].re[(point, nu)],
                                        g[axis].im[(point, nu)],
                                    );
                                    let factor = 0.5 * d_tau[spin][point];
                                    term.0 += factor * kinetic.0;
                                    term.1 += factor * kinetic.1;
                                }
                            }
                        }
                        matrix.re[(mu, nu)] += term.0;
                        matrix.im[(mu, nu)] += term.1;
                    }
                }
            }
            hermiticity_defect = hermiticity_defect.max(matrix.hermiticity_error());
            matrix.make_hermitian();
        }
    }

    Ok(PeriodicXcEvaluation {
        contribution: PeriodicXcContribution {
            exc,
            nonlocal_energy,
            vxc,
            n_elec,
        },
        hermiticity_defect,
        part_energies,
    })
}

pub(crate) fn build_dimensional_bloch_aos(
    basis: &MolBasis,
    geometry: &super::PeriodicGeometry,
    k_points: &[Vec3],
    points: &[Vec3],
    cutoff: f64,
    derivative_order: AoDerivativeOrder,
) -> Result<Vec<BlochAos>> {
    if k_points.is_empty()
        || k_points.iter().any(|k| !k.norm().is_finite())
        || !cutoff.is_finite()
        || cutoff <= 0.0
        || points.iter().any(|p| !p.norm().is_finite())
    {
        return Err(DftError::InvalidInput(
            "dimensional Bloch grid requires finite points, wave vectors and a positive AO cutoff"
                .into(),
        ));
    }
    let images = dimensional_xc_ao_images(basis, geometry, points, cutoff)?;
    Ok(build_bloch_aos_from_images(
        basis,
        k_points,
        points,
        &images,
        derivative_order,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::linalg::Matrix;
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    #[test]
    fn dimensional_bloch_spatial_gradients_and_hessians_have_four_step_fits() {
        let cell = test_cell();
        for dimension in 1..=3 {
            let geometry = super::super::PeriodicGeometry::new(
                super::super::Lattice::new(
                    Vec3::new(5.2, 0.5, 0.3),
                    Vec3::new(0.8, 5.6, 0.4),
                    Vec3::new(0.3, 0.2, 6.0),
                ),
                super::super::features::PeriodicBoundary::new(std::array::from_fn(|i| {
                    i < dimension
                })),
            )
            .unwrap();
            let k = geometry.k_points([3, 1, 1]).unwrap();
            let points = [Vec3::new(0.3, 0.8, 0.4), Vec3::new(1.4, 0.2, -0.3)];
            let exact = build_dimensional_bloch_aos(
                &cell.basis,
                &geometry,
                &k,
                &points,
                15.0,
                AoDerivativeOrder::Hessian,
            )
            .unwrap();
            let mut errors = [[0.0_f64; 4]; 2];
            let mut estimates = [Vec::new(), Vec::new()];
            for (index, h) in [0.04, 0.02, 0.01, 0.005].into_iter().enumerate() {
                let mut plus = points;
                let mut minus = points;
                for p in &mut plus {
                    p.x += h;
                }
                for p in &mut minus {
                    p.x -= h;
                }
                let a = build_dimensional_bloch_aos(
                    &cell.basis,
                    &geometry,
                    &k,
                    &plus,
                    15.0,
                    AoDerivativeOrder::Gradient,
                )
                .unwrap();
                let b = build_dimensional_bloch_aos(
                    &cell.basis,
                    &geometry,
                    &k,
                    &minus,
                    15.0,
                    AoDerivativeOrder::Gradient,
                )
                .unwrap();
                for order in 0..2 {
                    let mut values = Vec::new();
                    for ik in 0..k.len() {
                        let av = if order == 0 {
                            &a[ik].phi
                        } else {
                            &a[ik].grad.as_ref().unwrap()[0]
                        };
                        let bv = if order == 0 {
                            &b[ik].phi
                        } else {
                            &b[ik].grad.as_ref().unwrap()[0]
                        };
                        let actual = if order == 0 {
                            &exact[ik].grad.as_ref().unwrap()[0]
                        } else {
                            &exact[ik].hess.as_ref().unwrap()[0]
                        };
                        let mut value = av.clone();
                        value.add_scaled(bv, -1.0);
                        value.scale(0.5 / h);
                        errors[order][index] = errors[order][index].max(
                            value
                                .re
                                .max_abs_diff(&actual.re)
                                .max(value.im.max_abs_diff(&actual.im)),
                        );
                        values.push(value);
                    }
                    estimates[order].push(values);
                }
            }
            for order in 0..2 {
                assert!(errors[order].iter().all(|e| e.is_finite() && *e > 0.0));
                let x = [0.04_f64, 0.02, 0.01, 0.005].map(f64::ln);
                let y = errors[order].map(f64::ln);
                let xm = x.iter().sum::<f64>() / 4.0;
                let ym = y.iter().sum::<f64>() / 4.0;
                let p = x
                    .iter()
                    .zip(y)
                    .map(|(a, b)| (a - xm) * (b - ym))
                    .sum::<f64>()
                    / x.iter().map(|a| (a - xm).powi(2)).sum::<f64>();
                let r2 = 1.0
                    - x.iter()
                        .zip(y)
                        .map(|(a, b)| (b - ym - p * (a - xm)).powi(2))
                        .sum::<f64>()
                        / y.iter().map(|b| (b - ym).powi(2)).sum::<f64>();
                println!("{dimension}D Bloch spatial order {} errors={:?}; Bohr steps=0.04..0.005 p={p:.12} R2={r2:.12}",order+1,errors[order]);
                assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
                for pair in errors[order].windows(2) {
                    assert!((pair[0] / pair[1] / 4.0 - 1.0).abs() < 0.03);
                }
                for ik in 0..k.len() {
                    let mut richardson = estimates[order][3][ik].clone();
                    richardson.scale(4.0 / 3.0);
                    richardson.add_scaled(&estimates[order][2][ik], -1.0 / 3.0);
                    let actual = if order == 0 {
                        &exact[ik].grad.as_ref().unwrap()[0]
                    } else {
                        &exact[ik].hess.as_ref().unwrap()[0]
                    };
                    assert!(
                        richardson.re.max_abs_diff(&actual.re) < 1e-8
                            && richardson.im.max_abs_diff(&actual.im) < 1e-8
                    );
                }
            }
        }
    }

    fn test_cell() -> Cell {
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]);
        Cell::new(
            molecule,
            Lattice::new(
                Vec3::new(7.0, 0.0, 0.0),
                Vec3::new(0.0, 12.0, 0.0),
                Vec3::new(0.0, 0.0, 12.0),
            ),
            &data::load("sto-3g").unwrap(),
            8.0,
        )
        .unwrap()
    }

    fn coarse_grid() -> GridOptions {
        GridOptions {
            radial_override: Some(12),
            angular_override: Some(9),
            prune: false,
            ..GridOptions::default()
        }
    }

    fn build_bloch_aos_k_first(
        cell: &Cell,
        k_points: &[Vec3],
        points: &[Vec3],
        ao_cutoff: f64,
        derivative_order: AoDerivativeOrder,
    ) -> Vec<BlochAos> {
        let images = xc_ao_images(&cell.basis, &cell.lattice, points, ao_cutoff);
        let want_grad = derivative_order.wants_gradient();
        let want_hess = derivative_order.wants_hessian();
        k_points
            .iter()
            .map(|k| {
                let mut target = BlochAos::zeros(points.len(), cell.nao(), derivative_order);
                for image in &images {
                    let image_basis = shifted_active_basis(&cell.basis, image);
                    let values = eval_aos_full(&image_basis, points, want_grad, want_hess);
                    let phase = k.dot(image.translation);
                    target.add_screened_image(
                        &values,
                        &cell.basis,
                        &image.point_shells,
                        phase.cos(),
                        phase.sin(),
                    );
                }
                target
            })
            .collect()
    }

    fn assert_bloch_aos_close(actual: &BlochAos, reference: &BlochAos, tolerance: f64) {
        assert!(actual.phi.re.max_abs_diff(&reference.phi.re) < tolerance);
        assert!(actual.phi.im.max_abs_diff(&reference.phi.im) < tolerance);
        match (&actual.grad, &reference.grad) {
            (Some(actual), Some(reference)) => {
                for axis in 0..3 {
                    assert!(actual[axis].re.max_abs_diff(&reference[axis].re) < tolerance);
                    assert!(actual[axis].im.max_abs_diff(&reference[axis].im) < tolerance);
                }
            }
            (None, None) => {}
            _ => panic!("Bloch AO gradient availability differs"),
        }
        match (&actual.hess, &reference.hess) {
            (Some(actual), Some(reference)) => {
                for component in 0..6 {
                    assert!(
                        actual[component].re.max_abs_diff(&reference[component].re) < tolerance
                    );
                    assert!(
                        actual[component].im.max_abs_diff(&reference[component].im) < tolerance
                    );
                }
            }
            (None, None) => {}
            _ => panic!("Bloch AO Hessian availability differs"),
        }
    }

    #[test]
    fn translation_first_bloch_aos_match_k_first_reference() {
        let cell = test_cell();
        let k_points = cell.k_points([3, 1, 1]);
        let grid = PeriodicXcGrid::build(
            &cell,
            &k_points,
            &coarse_grid(),
            1,
            cell.overlap_cutoff,
            AoDerivativeOrder::Hessian,
        )
        .unwrap();
        let reference = build_bloch_aos_k_first(
            &cell,
            &k_points,
            &grid.points,
            cell.overlap_cutoff,
            AoDerivativeOrder::Hessian,
        );

        assert_eq!(grid.aos.len(), reference.len());
        for (actual, reference) in grid.aos.iter().zip(&reference) {
            assert_bloch_aos_close(actual, reference, 1e-14);
        }
    }

    fn assert_entry_has_phase(
        actual: &BlochMatrix,
        reference: &BlochMatrix,
        point: usize,
        ao: usize,
        phase: f64,
        tolerance: f64,
        label: &str,
    ) {
        let (cosine, sine) = (phase.cos(), phase.sin());
        let reference_re = reference.re[(point, ao)];
        let reference_im = reference.im[(point, ao)];
        let expected_re = cosine * reference_re - sine * reference_im;
        let expected_im = sine * reference_re + cosine * reference_im;
        let error = (actual.re[(point, ao)] - expected_re)
            .abs()
            .max((actual.im[(point, ao)] - expected_im).abs());
        assert!(
            error < tolerance,
            "{label} Bloch gauge mismatch at point {point}, AO {ao}: error={error:e}"
        );
    }

    #[test]
    fn shell_grid_screening_is_bloch_covariant_under_cell_relabelling() {
        let a = Vec3::new(7.0, 0.0, 0.0);
        let lattice = Lattice::new(a, Vec3::new(0.0, 9.0, 0.0), Vec3::new(0.0, 0.0, 10.0));
        let original_molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.4, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.8, 0.0, 0.0),
            },
        ]);
        let relabelled_molecule = Molecule::new(vec![
            original_molecule.atoms[0].clone(),
            Atom {
                z: 1,
                position: original_molecule.atoms[1].position + a,
            },
        ]);
        let basis = data::load("sto-3g").unwrap();
        let ao_cutoff = 2.0;
        let original = Cell::new(original_molecule, lattice.clone(), &basis, ao_cutoff).unwrap();
        let relabelled =
            Cell::new(relabelled_molecule, lattice.clone(), &basis, ao_cutoff).unwrap();
        let reciprocal = lattice.reciprocal();
        let k_points = vec![reciprocal[0] * 0.37 + reciprocal[1] * 0.11];
        let opts = GridOptions {
            radial_override: Some(6),
            angular_override: Some(5),
            prune: false,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        };
        let original_grid = PeriodicXcGrid::build(
            &original,
            &k_points,
            &opts,
            1,
            ao_cutoff,
            AoDerivativeOrder::Hessian,
        )
        .unwrap();
        let relabelled_grid = PeriodicXcGrid::build(
            &relabelled,
            &k_points,
            &opts,
            1,
            ao_cutoff,
            AoDerivativeOrder::Hessian,
        )
        .unwrap();

        assert_eq!(original_grid.atom_of, relabelled_grid.atom_of);
        assert_eq!(original_grid.points.len(), relabelled_grid.points.len());
        let atom_shifts = [Vec3::new(0.0, 0.0, 0.0), a];
        for point in 0..original_grid.points.len() {
            let owner = original_grid.atom_of[point];
            assert!(
                (relabelled_grid.points[point] - original_grid.points[point] - atom_shifts[owner])
                    .norm()
                    < 2e-12
            );
            assert!((relabelled_grid.weights[point] - original_grid.weights[point]).abs() < 2e-11);
            assert!(
                (relabelled_grid.raw_weights[point] - original_grid.raw_weights[point]).abs()
                    < 2e-14
            );
        }

        // The relabelled second atom is physically close to points owned by
        // atom 0 only through T=-a. Since |a| > ao_cutoff, screening by |T|
        // would omit this valid shell/point block.
        let moved_shell = relabelled
            .basis
            .shells
            .iter()
            .position(|shell| shell.atom == 1)
            .unwrap();
        let images = xc_ao_images(
            &relabelled.basis,
            &relabelled.lattice,
            &relabelled_grid.points,
            ao_cutoff,
        );
        assert!(images.iter().any(|image| {
            (image.translation + a).norm() < 2e-12
                && image.point_shells.iter().any(|&(point, shell)| {
                    relabelled_grid.atom_of[point] == 0 && shell == moved_shell
                })
        }));

        let ao_owners = original.basis.sph_ao_atoms();
        let reference = &original_grid.aos[0];
        let actual = &relabelled_grid.aos[0];
        let reference_grad = reference.grad.as_ref().unwrap();
        let actual_grad = actual.grad.as_ref().unwrap();
        let reference_hess = reference.hess.as_ref().unwrap();
        let actual_hess = actual.hess.as_ref().unwrap();
        for point in 0..original_grid.points.len() {
            let point_shift = atom_shifts[original_grid.atom_of[point]];
            for ao in 0..original.nao() {
                let ao_shift = atom_shifts[ao_owners[ao]];
                let phase = k_points[0].dot(point_shift - ao_shift);
                assert_entry_has_phase(
                    &actual.phi,
                    &reference.phi,
                    point,
                    ao,
                    phase,
                    3e-12,
                    "AO value",
                );
                for axis in 0..3 {
                    assert_entry_has_phase(
                        &actual_grad[axis],
                        &reference_grad[axis],
                        point,
                        ao,
                        phase,
                        3e-12,
                        "AO gradient",
                    );
                }
                for component in 0..6 {
                    assert_entry_has_phase(
                        &actual_hess[component],
                        &reference_hess[component],
                        point,
                        ao,
                        phase,
                        4e-12,
                        "AO Hessian",
                    );
                }
            }
        }
    }

    fn assert_periodic_potential_derivative(functional_name: &str, tolerance: f64) {
        let cell = test_cell();
        let k_points = cell.k_points([3, 1, 1]);
        let functional = resolve(&FunctionalSpec::Combined(functional_name.to_string())).unwrap();
        let grid = PeriodicXcGrid::build(
            &cell,
            &k_points,
            &coarse_grid(),
            1,
            cell.overlap_cutoff,
            AoDerivativeOrder::for_scf(functional.family),
        )
        .unwrap();
        let n = cell.nao();
        let base = Matrix::from_row_major(n, n, vec![0.45, 0.08, 0.08, 0.45]);
        let zero = Matrix::zeros(n, n);
        let mut density: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| {
            k_points
                .iter()
                .map(|_| BlochMatrix {
                    re: base.clone(),
                    im: zero.clone(),
                })
                .collect()
        });
        let reference = xc_contribution(&functional, &grid, &density).unwrap();
        let mut direction = BlochMatrix::zeros(n, n);
        direction.re[(0, 0)] = 0.2;
        direction.re[(1, 1)] = -0.1;
        direction.re[(0, 1)] = 0.04;
        direction.re[(1, 0)] = 0.04;
        let predicted = reference.vxc[0][1].trace_product(&direction) / k_points.len() as f64
            + reference.vxc[0][2].trace_product(&direction) / k_points.len() as f64;
        let step = 2e-5;
        for ik in [1usize, 2usize] {
            density[0][ik].add_scaled(&direction, step);
        }
        let plus = xc_contribution(&functional, &grid, &density).unwrap().exc;
        for ik in [1usize, 2usize] {
            density[0][ik].add_scaled(&direction, -2.0 * step);
        }
        let minus = xc_contribution(&functional, &grid, &density).unwrap().exc;
        let finite_difference = (plus - minus) / (2.0 * step);
        assert!(
            (finite_difference - predicted).abs() < tolerance,
            "{functional_name} periodic XC derivative: FD={finite_difference:e}, Vxc={predicted:e}"
        );

        for ik in 0..k_points.len() {
            assert!(reference.vxc[0][ik].hermiticity_error() < 1e-12);
        }
        assert!(reference.vxc[0][1].re.max_abs_diff(&reference.vxc[0][2].re) < 2e-10);
        let mut conjugacy = 0.0_f64;
        for (a, b) in reference.vxc[0][1]
            .im
            .as_slice()
            .iter()
            .zip(reference.vxc[0][2].im.as_slice())
        {
            conjugacy = conjugacy.max((a + b).abs());
        }
        assert!(conjugacy < 2e-10, "Vxc(-k) != Vxc(k)*");
    }

    #[test]
    fn periodic_gga_potential_is_the_density_derivative() {
        assert_periodic_potential_derivative("pbe", 3e-6);
    }

    #[test]
    fn periodic_meta_gga_potentials_are_density_derivatives() {
        // Both functionals have a non-zero v_tau channel.  This is the
        // variational condition needed by finite-k meta-GGA SCF, including
        // the 1/2 convention in tau and in its generalized-KS matrix.
        for functional in ["r2scan", "tpss"] {
            assert_periodic_potential_derivative(functional, 5e-6);
        }
    }

    #[test]
    fn derivative_orders_are_minimal_and_preserve_lower_order_values() {
        assert_eq!(
            AoDerivativeOrder::for_scf(Family::Lda),
            AoDerivativeOrder::Value
        );
        assert_eq!(
            AoDerivativeOrder::for_scf(Family::Gga),
            AoDerivativeOrder::Gradient
        );
        assert_eq!(
            AoDerivativeOrder::for_scf(Family::MetaGga),
            AoDerivativeOrder::Gradient
        );
        assert_eq!(
            AoDerivativeOrder::for_forces(Family::Lda),
            AoDerivativeOrder::Gradient
        );
        assert_eq!(
            AoDerivativeOrder::for_forces(Family::Gga),
            AoDerivativeOrder::Hessian
        );
        assert_eq!(
            AoDerivativeOrder::for_forces(Family::MetaGga),
            AoDerivativeOrder::Hessian
        );

        let cell = test_cell();
        let k_points = vec![cell.lattice.reciprocal()[0] * 0.37];
        let build = |order| {
            PeriodicXcGrid::build(
                &cell,
                &k_points,
                &coarse_grid(),
                1,
                cell.overlap_cutoff,
                order,
            )
            .unwrap()
        };
        let values = build(AoDerivativeOrder::Value);
        let gradients = build(AoDerivativeOrder::Gradient);
        let hessians = build(AoDerivativeOrder::Hessian);

        assert_eq!(values.derivative_order(), AoDerivativeOrder::Value);
        assert_eq!(gradients.derivative_order(), AoDerivativeOrder::Gradient);
        assert_eq!(hessians.derivative_order(), AoDerivativeOrder::Hessian);
        for aos in &values.aos {
            assert!(aos.grad.is_none());
            assert!(aos.hess.is_none());
        }
        for aos in &gradients.aos {
            assert!(aos.grad.is_some());
            assert!(aos.hess.is_none());
        }
        for aos in &hessians.aos {
            assert!(aos.grad.is_some());
            assert!(aos.hess.is_some());
        }

        for ((value, gradient), hessian) in values.aos.iter().zip(&gradients.aos).zip(&hessians.aos)
        {
            assert!(value.phi.re.max_abs_diff(&hessian.phi.re) < 1e-14);
            assert!(value.phi.im.max_abs_diff(&hessian.phi.im) < 1e-14);
            for axis in 0..3 {
                let low = &gradient.grad.as_ref().unwrap()[axis];
                let full = &hessian.grad.as_ref().unwrap()[axis];
                assert!(low.re.max_abs_diff(&full.re) < 1e-14);
                assert!(low.im.max_abs_diff(&full.im) < 1e-14);
            }
        }

        let n = cell.nao();
        let mut re = Matrix::zeros(n, n);
        let mut im = Matrix::zeros(n, n);
        for i in 0..n {
            re[(i, i)] = 0.4 - 0.05 * i as f64;
        }
        if n > 1 {
            re[(0, 1)] = 0.05;
            re[(1, 0)] = 0.05;
            im[(0, 1)] = 0.02;
            im[(1, 0)] = -0.02;
        }
        let matrix = BlochMatrix { re, im };
        let density = [vec![matrix.clone()], vec![matrix]];
        for (name, minimal) in [
            ("svwn5", &values),
            ("pbe", &gradients),
            ("r2scan", &gradients),
        ] {
            let functional = resolve(&FunctionalSpec::Combined(name.to_string())).unwrap();
            let lean = xc_contribution(&functional, minimal, &density).unwrap();
            let legacy_full = xc_contribution(&functional, &hessians, &density).unwrap();
            assert!((lean.exc - legacy_full.exc).abs() < 1e-14);
            assert!((lean.n_elec - legacy_full.n_elec).abs() < 1e-14);
            for spin in 0..2 {
                assert!(
                    lean.vxc[spin][0]
                        .re
                        .max_abs_diff(&legacy_full.vxc[spin][0].re)
                        < 1e-14
                );
                assert!(
                    lean.vxc[spin][0]
                        .im
                        .max_abs_diff(&legacy_full.vxc[spin][0].im)
                        < 1e-14
                );
            }
        }

        let pbe = resolve(&FunctionalSpec::Combined("pbe".to_string())).unwrap();
        assert!(xc_contribution(&pbe, &values, &density).is_err());
    }
}

#[cfg(test)]
#[path = "xc_batch_tests.rs"]
mod batch_tests;
fn dimensional_xc_ao_images(
    basis: &MolBasis,
    geometry: &super::PeriodicGeometry,
    points: &[Vec3],
    cutoff: f64,
) -> Result<Vec<XcAoImage>> {
    let mut by_translation: BTreeMap<[i64; 3], Vec<(usize, usize)>> = BTreeMap::new();
    for (point_index, point) in points.iter().enumerate() {
        for (shell_index, shell) in basis.shells.iter().enumerate() {
            for index in geometry.nearby_translation_indices(shell.center - *point, cutoff)? {
                by_translation
                    .entry(index)
                    .or_default()
                    .push((point_index, shell_index));
            }
        }
    }
    let images: Vec<_> = by_translation
        .into_iter()
        .map(|(index, point_shells)| XcAoImage {
            translation: geometry.lattice().translation(index),
            point_shells,
        })
        .collect();
    Ok(images)
}
