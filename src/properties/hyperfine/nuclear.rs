// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear derivatives of the nonrelativistic collinear hyperfine tensor.
//! Both the target nucleus and every Gaussian centre move. The shared
//! stationary nuclear response supplies the metric, orbital, integral and
//! moving-grid response; no displaced SCF is used by this implementation.

use crate::basis::MolBasis;
use crate::constants::{
    ELECTRON_G_FACTOR, FINE_STRUCTURE, HARTREE_TO_JOULE, HBAR_J_S, PROTON_ELECTRON_MASS_RATIO,
};
use crate::grid::MolecularGrid;
use crate::response::nuclear::magnetic_operators::spin_dipole_operator_path;
use crate::response::nuclear::one_body::nuclear_one_body_coefficients;
use crate::response::nuclear::{nuclear_taylor, NuclearTaylor};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Tensor coefficient or derivative, in MHz per dimensionless nuclear g.
/// Nuclear derivatives have an additional inverse Bohr per derivative.
#[derive(Clone, Debug, Default)]
pub struct HyperfineCoefficient {
    pub contact: f64,
    pub dipolar: [[f64; 3]; 3],
    pub total: [[f64; 3]; 3],
}

impl HyperfineCoefficient {
    fn add_scaled(&mut self, other: &Self, scale: f64) {
        self.contact += scale * other.contact;
        for a in 0..3 {
            for b in 0..3 {
                self.dipolar[a][b] += scale * other.dipolar[a][b];
                self.total[a][b] += scale * other.total[a][b];
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct HyperfineNuclearPath {
    pub nuclei: Vec<usize>,
    /// `[requested nucleus][Taylor order]`; coefficient k is derivative/k!.
    pub coefficients: Vec<Vec<HyperfineCoefficient>>,
    /// Raw contraction accumulation bounds before MHz/spin prefactors,
    /// `[nucleus][order][contact, nine dipolar entries]`.
    pub accumulation_bounds: Vec<Vec<[f64; 10]>>,
    pub operator_poisson_residuals: Vec<Vec<f64>>,
    pub operator_hermiticity_residuals: Vec<Vec<f64>>,
    /// Complete raw metric/projector/stationarity and coefficient diagnostics.
    pub response: ResponseReport<NuclearTaylor>,
}

fn scale(twice_spin: f64) -> f64 {
    FINE_STRUCTURE * FINE_STRUCTURE * ELECTRON_G_FACTOR
        / (4.0 * PROTON_ELECTRON_MASS_RATIO * twice_spin)
        * HARTREE_TO_JOULE
        / (2.0 * PI * HBAR_J_S)
        * 1e-6
}

/// Full collinear SCF response of the hyperfine tensor, through order two.
/// Fractional occupations, ECP sites, spinor/complex and constrained ROKS
/// references are outside this initial all-electron integer UKS contract.
/// This accepts a stationary SCF functional, not a double-hybrid recipe.
/// A caller handling a correlated recipe must reject it before extracting
/// its SCF parent; the parent alone carries no correlated-property metadata.
#[allow(clippy::too_many_arguments)]
pub fn hyperfine_nuclear_path(
    mol: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    nuclei: &[usize],
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<HyperfineNuclearPath> {
    validate_order(order)?;
    if !basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "ECP hyperfine nuclear response".into(),
            missing: "effective-core hyperfine operators and their nuclear response".into(),
        });
    }
    if nuclei.is_empty()
        || nuclei.iter().any(|&i| i >= mol.len())
        || basis.shells.iter().any(|shell| shell.atom >= mol.len())
        || direction.len() != mol.len()
        || direction.iter().flatten().any(|x| !x.is_finite())
        || scf.n_alpha == scf.n_beta
    {
        return Err(DftError::InvalidInput("hyperfine nuclear response requires all-electron collinear open-shell data, valid nuclei/directions and order 0..2".into()));
    }
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    crate::response::diagnostics::reference_diagnostics(scf)?;
    let response = nuclear_taylor(
        mol, basis, functional, grid, scf, opts, direction, order, config,
    )?;
    let factor = scale(scf.n_alpha as f64 - scf.n_beta as f64);
    let mut coefficients = Vec::with_capacity(nuclei.len());
    let mut accumulation_bounds = Vec::with_capacity(nuclei.len());
    let mut operator_poisson_residuals = Vec::with_capacity(nuclei.len());
    let mut operator_hermiticity_residuals = Vec::with_capacity(nuclei.len());
    for &nucleus in nuclei {
        let operator = spin_dipole_operator_path(mol, basis, nucleus, direction, order)?;
        let contractions = std::iter::once(&operator.contact)
            .chain(operator.dipolar.iter())
            .map(|op| {
                nuclear_one_body_coefficients(
                    &response.value.density_coefficients,
                    &response.value.density_corrections,
                    op,
                    [1.0, -1.0],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut path = Vec::with_capacity(order + 1);
        for degree in 0..=order {
            let contact = factor * contractions[0].coefficients[degree] * 8.0 * PI / 3.0;
            let dipolar = std::array::from_fn(|a| {
                std::array::from_fn(|b| factor * contractions[1 + 3 * a + b].coefficients[degree])
            });
            let total = std::array::from_fn(|a| {
                std::array::from_fn(|b| dipolar[a][b] + if a == b { contact } else { 0.0 })
            });
            if !contact.is_finite() || total.iter().flatten().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput(
                    "nonfinite hyperfine nuclear coefficient".into(),
                ));
            }
            path.push(HyperfineCoefficient {
                contact,
                dipolar,
                total,
            });
        }
        accumulation_bounds.push(
            (0..=order)
                .map(|degree| std::array::from_fn(|c| contractions[c].accumulation_bounds[degree]))
                .collect(),
        );
        operator_poisson_residuals.push(operator.poisson_residuals);
        operator_hermiticity_residuals.push(operator.hermiticity_residuals);
        coefficients.push(path);
    }
    Ok(HyperfineNuclearPath {
        nuclei: nuclei.to_vec(),
        coefficients,
        accumulation_bounds,
        operator_poisson_residuals,
        operator_hermiticity_residuals,
        response,
    })
}

fn validate_order(order: usize) -> Result<()> {
    if order > 2 {
        return Err(DftError::Unsupported {
            feature: format!("hyperfine nuclear derivative order {order}"),
            missing: "hyperfine nuclear response beyond second order".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{output_layout, validate_order};
    use crate::DftError;

    #[test]
    fn unimplemented_positive_order_is_unsupported() {
        for order in [0, 1, 2] {
            assert!(validate_order(order).is_ok());
        }
        for order in [3, usize::MAX] {
            assert!(matches!(
                validate_order(order),
                Err(DftError::Unsupported { .. })
            ));
        }
    }

    #[test]
    fn output_shape_counts_and_overflow_are_checked() {
        let first = output_layout(4, 4, 1).unwrap();
        let second = output_layout(4, 4, 2).unwrap();
        assert_eq!((first.coordinates, first.paths), (12, 12));
        assert_eq!((second.coordinates, second.paths), (12, 78));
        assert!(second.bytes > first.bytes);
        assert!(output_layout(usize::MAX / 3 + 1, 1, 1).is_err());
        assert!(output_layout(1, usize::MAX, 2).is_err());
    }

    #[test]
    fn output_reservations_refuse_over_budget_and_release_on_drop() {
        crate::resources::ResourceLimits::new(Some(1), Some(1))
            .unwrap()
            .install(|| {
                let baseline = crate::resources::current_usage().unwrap().reserved_bytes;
                let small = output_layout(4, 4, 2).unwrap();
                let lease =
                    crate::resources::reserve(small.bytes, "hyperfine test output").unwrap();
                assert_eq!(
                    crate::resources::current_usage().unwrap().reserved_bytes,
                    baseline + small.bytes
                );
                let large = output_layout(100, 100, 2).unwrap();
                assert!(matches!(
                    crate::resources::reserve(large.bytes, "hyperfine test output"),
                    Err(DftError::ResourceLimit { .. })
                ));
                drop(lease);
                assert_eq!(
                    crate::resources::current_usage().unwrap().reserved_bytes,
                    baseline
                );
            })
            .unwrap();
    }
}

#[derive(Clone, Debug)]
pub struct HyperfinePathDiagnostics {
    pub direction: Vec<[f64; 3]>,
    pub converged: bool,
    /// Inner coefficient-response iterations, indexed by nonconstant order.
    pub iterations: Vec<usize>,
    pub residuals: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub metric_defects: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub particle_number_coefficients: Vec<[f64; 2]>,
    pub reference_energy_error: f64,
    pub reference_fock_error: f64,
    pub coefficient_equations:
        Option<Vec<crate::response::nuclear::CoefficientEquationDiagnostics>>,
    pub accumulation_bounds: Vec<Vec<[f64; 10]>>,
    pub operator_poisson_residuals: Vec<Vec<f64>>,
    pub operator_hermiticity_residuals: Vec<Vec<f64>>,
    pub warnings: Vec<String>,
}

impl From<HyperfineNuclearPath> for HyperfinePathDiagnostics {
    fn from(path: HyperfineNuclearPath) -> Self {
        Self {
            direction: path.response.value.direction,
            converged: path.response.converged,
            iterations: path.response.iterations,
            residuals: path.response.residuals,
            stationarity_defects: path.response.value.stationarity_defects,
            metric_defects: path.response.value.metric_defects,
            projector_defects: path.response.value.projector_defects,
            particle_number_coefficients: path.response.value.particle_number_coefficients,
            reference_energy_error: path.response.value.reference_energy_error,
            reference_fock_error: path.response.value.reference_fock_error,
            coefficient_equations: path.response.value.coefficient_equations,
            accumulation_bounds: path.accumulation_bounds,
            operator_poisson_residuals: path.operator_poisson_residuals,
            operator_hermiticity_residuals: path.operator_hermiticity_residuals,
            warnings: path.response.warnings,
        }
    }
}

#[derive(Debug)]
pub struct HyperfineNuclearDerivatives {
    pub nuclei: Vec<usize>,
    pub order: usize,
    pub value: Vec<HyperfineCoefficient>,
    /// `[nucleus][3*atom+Cartesian axis]`, MHz/g/Bohr.
    pub gradient: Vec<Vec<HyperfineCoefficient>>,
    /// `[nucleus][coordinate][coordinate]`, MHz/g/Bohr^2.
    pub hessian: Option<Vec<Vec<Vec<HyperfineCoefficient>>>>,
    pub paths: Vec<HyperfinePathDiagnostics>,
    pub converged: bool,
    // Keep output reservations alive with their owning buffers. This type
    // deliberately has no deep Clone sharing a single accounting lease.
    _output_memory: crate::resources::MemoryReservation,
    _diagnostic_memory: Vec<crate::resources::MemoryReservation>,
}

struct OutputLayout {
    coordinates: usize,
    paths: usize,
    bytes: usize,
}

fn output_layout(atoms: usize, targets: usize, order: usize) -> Result<OutputLayout> {
    use crate::resources::checked_bytes;
    use std::mem::size_of;
    let coordinates = checked_bytes(&[3, atoms], "hyperfine coordinate count")?;
    let paths = if order == 2 {
        let next = coordinates
            .checked_add(1)
            .ok_or_else(|| DftError::InvalidInput("hyperfine path count overflows".into()))?;
        checked_bytes(&[coordinates, next], "hyperfine path count")? / 2
    } else {
        coordinates
    };
    let mut bytes = 0usize;
    let mut add = |factors: &[usize]| -> Result<()> {
        bytes = bytes
            .checked_add(checked_bytes(factors, "hyperfine output")?)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| DftError::InvalidInput("hyperfine output size overflows".into()))?;
        Ok(())
    };
    add(&[targets, size_of::<usize>()])?;
    add(&[targets, size_of::<HyperfineCoefficient>()])?;
    add(&[targets, size_of::<Vec<HyperfineCoefficient>>()])?;
    add(&[targets, coordinates, size_of::<HyperfineCoefficient>()])?;
    add(&[paths, size_of::<HyperfinePathDiagnostics>()])?;
    add(&[paths, size_of::<crate::resources::MemoryReservation>()])?;
    add(&[atoms, size_of::<[f64; 3]>()])?;
    if order == 2 {
        add(&[targets, size_of::<Vec<Vec<HyperfineCoefficient>>>()])?;
        add(&[targets, coordinates, size_of::<Vec<HyperfineCoefficient>>()])?;
        add(&[
            targets,
            coordinates,
            coordinates,
            size_of::<HyperfineCoefficient>(),
        ])?;
    }
    Ok(OutputLayout {
        coordinates,
        paths,
        bytes,
    })
}

fn diagnostic_bytes(path: &HyperfinePathDiagnostics) -> Result<usize> {
    use crate::resources::checked_bytes;
    use std::mem::size_of;
    fn vector_bytes<T>(values: &Vec<T>) -> Result<usize> {
        checked_bytes(
            &[values.capacity(), size_of::<T>()],
            "hyperfine diagnostic output",
        )
    }
    fn nested_bytes<T>(values: &Vec<Vec<T>>) -> Result<usize> {
        values.iter().try_fold(vector_bytes(values)?, |sum, value| {
            sum.checked_add(vector_bytes(value)?)
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(|| DftError::InvalidInput("hyperfine diagnostic size overflows".into()))
        })
    }
    let mut bytes = 0usize;
    for part in [
        vector_bytes(&path.direction)?,
        vector_bytes(&path.iterations)?,
        vector_bytes(&path.residuals)?,
        vector_bytes(&path.stationarity_defects)?,
        vector_bytes(&path.metric_defects)?,
        vector_bytes(&path.projector_defects)?,
        vector_bytes(&path.particle_number_coefficients)?,
        path.coefficient_equations
            .as_ref()
            .map_or(Ok(0), vector_bytes)?,
        nested_bytes(&path.accumulation_bounds)?,
        nested_bytes(&path.operator_poisson_residuals)?,
        nested_bytes(&path.operator_hermiticity_residuals)?,
        vector_bytes(&path.warnings)?,
    ]
    .into_iter()
    .chain(path.warnings.iter().map(|value| value.capacity()))
    {
        bytes = bytes
            .checked_add(part)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| DftError::InvalidInput("hyperfine diagnostic size overflows".into()))?;
    }
    Ok(bytes)
}

fn retain_path(
    path: HyperfineNuclearPath,
    paths: &mut Vec<HyperfinePathDiagnostics>,
    memory: &mut Vec<crate::resources::MemoryReservation>,
) -> Result<()> {
    // Temporary shared AO/Taylor workspace is outside this output-specific
    // reservation; this is not a total-workspace or process-RSS cap. Charge every
    // retained diagnostic allocation before adopting it into the result.
    let path = HyperfinePathDiagnostics::from(path);
    let lease =
        crate::resources::reserve(diagnostic_bytes(&path)?, "hyperfine retained diagnostics")?;
    paths.push(path);
    memory.push(lease);
    Ok(())
}

/// Assemble every Cartesian gradient and Hessian component from analytic
/// quadratic paths. Mixed entries use polarization, with no displaced SCF.
#[allow(clippy::too_many_arguments)]
pub fn hyperfine_nuclear_derivatives(
    mol: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    nuclei: &[usize],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<HyperfineNuclearDerivatives> {
    validate_order(order)?;
    if !(1..=2).contains(&order)
        || mol.is_empty()
        || nuclei.is_empty()
        || nuclei.iter().any(|&n| n >= mol.len())
        || nuclei
            .iter()
            .enumerate()
            .any(|(i, n)| nuclei[..i].contains(n))
    {
        return Err(DftError::InvalidInput(
            "assembled hyperfine nuclear derivatives require valid nuclei and order1 or2".into(),
        ));
    }
    let layout = output_layout(mol.len(), nuclei.len(), order)?;
    let output_memory = crate::resources::reserve(layout.bytes, "hyperfine Cartesian output")?;
    let dimension = layout.coordinates;
    let mut value = Vec::new();
    let mut gradient = vec![vec![HyperfineCoefficient::default(); dimension]; nuclei.len()];
    let mut hessian = (order == 2).then(|| {
        vec![vec![vec![HyperfineCoefficient::default(); dimension]; dimension]; nuclei.len()]
    });
    let mut paths = Vec::with_capacity(layout.paths);
    let mut diagnostic_memory = Vec::with_capacity(layout.paths);
    for i in 0..dimension {
        let mut direction = vec![[0.0; 3]; mol.len()];
        direction[i / 3][i % 3] = 1.0;
        let path = hyperfine_nuclear_path(
            mol, basis, functional, grid, scf, opts, nuclei, &direction, order, config,
        )?;
        if i == 0 {
            value = path.coefficients.iter().map(|p| p[0].clone()).collect();
        }
        for nucleus in 0..nuclei.len() {
            gradient[nucleus][i] = path.coefficients[nucleus][1].clone();
            if let Some(hessian) = &mut hessian {
                hessian[nucleus][i][i].add_scaled(&path.coefficients[nucleus][2], 2.0);
            }
        }
        retain_path(path, &mut paths, &mut diagnostic_memory)?;
    }
    if let Some(hessian) = &mut hessian {
        for i in 0..dimension {
            for j in 0..i {
                let mut direction = vec![[0.0; 3]; mol.len()];
                direction[i / 3][i % 3] = 1.0;
                direction[j / 3][j % 3] = 1.0;
                let path = hyperfine_nuclear_path(
                    mol, basis, functional, grid, scf, opts, nuclei, &direction, 2, config,
                )?;
                for nucleus in 0..nuclei.len() {
                    let mut mixed = path.coefficients[nucleus][2].clone();
                    mixed.add_scaled(&hessian[nucleus][i][i], -0.5);
                    mixed.add_scaled(&hessian[nucleus][j][j], -0.5);
                    hessian[nucleus][i][j] = mixed.clone();
                    hessian[nucleus][j][i] = mixed;
                }
                retain_path(path, &mut paths, &mut diagnostic_memory)?;
            }
        }
    }
    let converged = paths.iter().all(|path| path.converged);
    Ok(HyperfineNuclearDerivatives {
        nuclei: nuclei.to_vec(),
        order,
        value,
        gradient,
        hessian,
        paths,
        converged,
        _output_memory: output_memory,
        _diagnostic_memory: diagnostic_memory,
    })
}
