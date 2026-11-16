// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! DFT quadrature energy and analytic Bloch potentials in a periodic subspace.

use super::basis_boundary::weighted_bloch_overlap;
use super::integrals::{bloch, dimensional_image_blocks, Operator};
use super::xc::{xc_contribution_with_components, AoDerivativeOrder, PeriodicXcGrid};
use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::{aos::MAX_L, n_sph, MolBasis};
use crate::grid::GridOptions;
use crate::linalg::Orthogonalizer;
use crate::math::Vec3;
use crate::system::Molecule;
use crate::xc::{HybridParams, ResolvedFunctional};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct PeriodicXcOptions {
    pub vv10: Option<super::nonlocal::PeriodicVv10Options>,
    pub k_mesh: [usize; 3],
    /// Independent AO-pair and point/shell image cutoff in Bohr.
    pub basis_cutoff: f64,
    /// Radial and local Becke image cutoff in Bohr.
    pub grid_cutoff: f64,
    pub grid: GridOptions,
    pub operator_tolerance: f64,
    /// Absolute grid electron-count error against the analytic overlap metric.
    pub grid_tolerance: f64,
}
impl Default for PeriodicXcOptions {
    fn default() -> Self {
        Self {
            vv10: None,
            k_mesh: [1; 3],
            basis_cutoff: 12.0,
            grid_cutoff: 12.0,
            grid: GridOptions::default(),
            operator_tolerance: 1e-11,
            grid_tolerance: 1e-5,
        }
    }
}

pub struct PeriodicXcReport {
    pub energy: f64,
    pub nonlocal_energy: f64,
    /// Weighted parts in provider order, followed by VV10 when enabled.
    pub component_energies: Vec<f64>,
    pub component_labels: Vec<String>,
    /// [spin][k][AO,AO], the derivative of the reported XC energy.
    pub potential: [Vec<BlochMatrix>; 2],
    pub k_points: Vec<Vec3>,
    pub electron_count: f64,
    pub grid_electron_count: f64,
    pub expected_electrons: f64,
    pub charge_residual: f64,
    pub grid_number_defect: f64,
    pub hermiticity_defect: f64,
    pub component_energy_defect: f64,
    pub grid_points: usize,
    pub hybrid: HybridParams,
    pub required_nonlocal: Option<&'static str>,
    pub converged: bool,
    pub model: &'static str,
    pub units: &'static str,
    pub warnings: Vec<String>,
}

/// Immutable geometry/grid cache. Functional providers are evaluated through
/// the same local variables and variational contraction as the 3D SCF path.
pub struct DimensionalXcBackend {
    grid: PeriodicXcGrid,
    overlap: Vec<BlochMatrix>,
    expected_electrons: f64,
    operator_tolerance: f64,
    grid_tolerance: f64,
}
impl DimensionalXcBackend {
    pub fn new(
        mol: &Molecule,
        basis: &MolBasis,
        geometry: &PeriodicGeometry,
        functional: &ResolvedFunctional,
        options: &PeriodicXcOptions,
    ) -> Result<Self> {
        Self::with_reference_overlap(mol, basis, geometry, functional, options, None, false)
    }

    pub(super) fn with_reference_overlap(
        mol: &Molecule,
        basis: &MolBasis,
        geometry: &PeriodicGeometry,
        functional: &ResolvedFunctional,
        options: &PeriodicXcOptions,
        overlap: Option<&[BlochMatrix]>,
        spinor: bool,
    ) -> Result<Self> {
        let invalid = |message: &str| DftError::InvalidInput(message.into());
        if mol.is_empty()
            || basis.nao_sph == 0
            || geometry.dimension() == 0
            || [
                options.basis_cutoff,
                options.grid_cutoff,
                options.operator_tolerance,
                options.grid_tolerance,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || options.grid.level > 9
            || !options.grid.weight_cutoff.is_finite()
            || options.grid.weight_cutoff < 0.0
        {
            return Err(invalid("dimensional XC requires atoms, AOs, 1D/2D/3D periodicity and finite positive controls"));
        }
        if mol
            .atoms
            .iter()
            .any(|a| !(1..=118).contains(&a.z) || !a.position.norm().is_finite())
        {
            return Err(invalid(
                "XC atom numbers and positions must be valid and finite",
            ));
        }
        let mut assigned = vec![false; basis.nao_sph];
        for shell in &basis.shells {
            if shell.l > MAX_L
                || shell.atom >= mol.len()
                || !shell.center.norm().is_finite()
                || (shell.center - mol.atoms[shell.atom].position).norm() > 1e-10
                || shell.exps.is_empty()
                || shell.exps.len() != shell.coefs.len()
                || shell.exps.iter().any(|v| !v.is_finite() || *v <= 0.0)
                || shell.coefs.iter().any(|v| !v.is_finite())
                || shell
                    .sph_off
                    .checked_add(n_sph(shell.l))
                    .is_none_or(|end| end > assigned.len())
            {
                return Err(invalid("XC shells must have valid angular orders, exponents, coefficients, AO offsets and atom centres"));
            }
            for ao in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                if assigned[ao] {
                    return Err(invalid("XC shell AO ranges overlap"));
                }
                assigned[ao] = true;
            }
        }
        if assigned.contains(&false)
            || basis
                .ecps
                .iter()
                .any(|e| e.atom >= mol.len() || e.potential.n_core > mol.atoms[e.atom].z as u32)
        {
            return Err(invalid(
                "XC basis AO coverage and effective nuclear charges must be valid",
            ));
        }
        let k_points = geometry.k_points(options.k_mesh)?;
        let angular_frame = if spinor {
            super::grid::AngularGridFrame::bulk(geometry)
        } else {
            None
        };
        // An SCF density is normalized in the reference overlap metric.
        // The grid AO image radius independently controls XC quadrature.
        let overlap = if let Some(overlap) = overlap {
            overlap.to_vec()
        } else {
            let images =
                dimensional_image_blocks(basis, geometry, options.basis_cutoff, Operator::Overlap)?;
            k_points
                .iter()
                .map(|&k| BlochMatrix::from_pair(bloch(&images, k)))
                .collect()
        };
        let expected_electrons =
            basis.effective_charges(mol).into_iter().sum::<f64>() - mol.charge as f64;
        if expected_electrons < 0.0 {
            return Err(invalid("formal charge exceeds effective nuclear charge"));
        }
        let derivative_order =
            AoDerivativeOrder::for_scf(functional.family).max(if options.vv10.is_some() {
                AoDerivativeOrder::Gradient
            } else {
                AoDerivativeOrder::Value
            });
        let build_grid = |grid_options| {
            PeriodicXcGrid::build_dimensional_oriented(
                mol,
                basis,
                geometry,
                &k_points,
                grid_options,
                options.grid_cutoff,
                options.basis_cutoff,
                derivative_order,
                angular_frame.clone(),
            )
        };
        let mut grid = build_grid(&options.grid)?;
        if angular_frame.is_some() && options.grid.prune && expected_electrons > 0.0 {
            // Choose a single grid before SCF. Occupation changes in SCF
            // cannot alter the discretized energy, potential or derivatives.
            let bound = grid_number_error_bound(&grid, &overlap, expected_electrons)?;
            let quadrature = if bound <= options.grid_tolerance {
                "pruned"
            } else {
                let unpruned = GridOptions {
                    prune: false,
                    ..options.grid.clone()
                };
                drop(grid);
                grid = build_grid(&unpruned)?;
                "unpruned"
            };
            crate::logger::log(&format!(
                "[periodic-xc] angular_quadrature={quadrature} level={} pruned_number_error_bound={bound:.6e} grid_tolerance={:.6e}",
                options.grid.level, options.grid_tolerance
            ));
        }
        let grid = match options.vv10 {
            Some(settings) => grid.with_dimensional_vv10(geometry.clone(), settings)?,
            None => grid,
        };
        Ok(Self {
            grid,
            overlap,
            expected_electrons,
            operator_tolerance: options.operator_tolerance,
            grid_tolerance: options.grid_tolerance,
        })
    }

    pub fn grid(&self) -> &PeriodicXcGrid {
        &self.grid
    }

    /// Evaluate the quadrature functional including any attached VV10 term.
    /// Exact exchange and Coulomb use separate operators.
    pub fn evaluate(
        &self,
        functional: &ResolvedFunctional,
        density: &[Vec<BlochMatrix>; 2],
    ) -> Result<PeriodicXcReport> {
        if functional
            .parts
            .iter()
            .any(|(w, p)| !w.is_finite() || p.family().rung() > functional.family.rung())
        {
            return Err(DftError::InvalidInput(
                "XC weights and declared maximum functional family must be valid".into(),
            ));
        }
        let evaluation = xc_contribution_with_components(functional, &self.grid, density)?;
        let contribution = evaluation.contribution;
        let mut energies = evaluation
            .part_energies
            .expect("component evaluation was requested");
        let mut labels: Vec<_> = functional
            .parts
            .iter()
            .map(|(_, p)| p.name().to_string())
            .collect();
        if self.grid.nonlocal_options().is_some() {
            energies.push(contribution.nonlocal_energy);
            labels.push("VV10".into());
        }
        if !contribution.exc.is_finite()
            || !contribution.n_elec.is_finite()
            || energies.iter().any(|v| !v.is_finite())
            || contribution.vxc.iter().flatten().any(|m| {
                m.re.as_slice()
                    .iter()
                    .chain(m.im.as_slice())
                    .any(|v| !v.is_finite())
            })
        {
            return Err(DftError::InvalidInput(
                "XC provider returned non-finite energies or potentials".into(),
            ));
        }
        let nk = self.grid.k_points.len() as f64;
        let electron_count = density
            .iter()
            .map(|spin| {
                spin.iter()
                    .zip(&self.overlap)
                    .map(|(p, s)| p.trace_product(s))
                    .sum::<f64>()
                    / nk
            })
            .sum::<f64>();
        let charge_residual = electron_count - self.expected_electrons;
        let grid_number_defect = (contribution.n_elec - electron_count).abs();
        let component_energy_defect = (energies.iter().sum::<f64>() - contribution.exc).abs();
        let converged = evaluation.hermiticity_defect <= self.operator_tolerance
            && component_energy_defect <= self.operator_tolerance
            && charge_residual.abs() <= self.operator_tolerance
            && grid_number_defect <= self.grid_tolerance;
        let mut warnings = vec![
            "This report contains XC component energies and potentials; SCF \
            stationarity and complete functional energy are not inferred."
                .into(),
        ];
        if grid_number_defect > self.grid_tolerance {
            warnings.push("Grid electron count differs from the analytic overlap metric; refine the physical cutoffs and quadrature independently.".into());
        }
        if charge_residual.abs() > self.operator_tolerance {
            warnings.push(
                "Supplied density electron count does not match the requested formal charge."
                    .into(),
            );
        }
        Ok(PeriodicXcReport { energy: contribution.exc, nonlocal_energy:contribution.nonlocal_energy, component_energies: energies,
            component_labels: labels,
            potential: contribution.vxc, k_points: self.grid.k_points.clone(), electron_count,
            grid_electron_count: contribution.n_elec, expected_electrons: self.expected_electrons,
            charge_residual, grid_number_defect, hermiticity_defect: evaluation.hermiticity_defect,
            component_energy_defect, grid_points: self.grid.len(), hybrid: functional.hybrid,
            required_nonlocal: functional.required_nonlocal_term(), converged,
            model: "Zero-field LDA/GGA/meta-GGA and explicitly requested VV10 components on dimensional Bloch-AO quadrature; exact exchange and self-consistency are separate.",
            units: "energy/potential: Hartree; electron counts: dimensionless; geometry/cutoffs: Bohr; k: inverse Bohr", warnings })
    }
}

/// Each positive density is normalized against the analytic overlap. In an
/// orthonormal basis, the error in the number is bounded by that number
/// times the spectral norm of the quadrature overlap defect. The largest
/// absolute row sum of the symmetric real embedding bounds this norm.
fn grid_number_error_bound(
    grid: &PeriodicXcGrid,
    overlap: &[BlochMatrix],
    electrons: f64,
) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for (ao, metric) in grid.aos.iter().zip(overlap) {
        let mut defect = weighted_bloch_overlap(&ao.phi, &grid.weights);
        defect.add_scaled(metric, -1.0);
        let x = Orthogonalizer::canonical(&metric.real_embedding(), 0.0)?.x;
        let orthogonal_defect = x.transpose_matmul(&defect.real_embedding().matmul(&x));
        let row_sum = orthogonal_defect
            .as_slice()
            .chunks_exact(orthogonal_defect.cols)
            .map(|row| row.iter().map(|v| v.abs()).sum())
            .fold(0.0_f64, f64::max);
        maximum = maximum.max(row_sum);
    }
    Ok(electrons * maximum)
}

/// Common Rust/CLI/native Python/ASE semilocal dimensional XC driver.
pub fn periodic_xc_operators(
    mol: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    functional: &ResolvedFunctional,
    density: &[Vec<BlochMatrix>; 2],
    options: &PeriodicXcOptions,
) -> Result<PeriodicXcReport> {
    DimensionalXcBackend::new(mol, basis, geometry, functional, options)?
        .evaluate(functional, density)
}
