// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Direct electron spin-spin part of the collinear zero-field splitting.
//! This is a contraction of an already computed stationary nuclear response,
//! not a complete spin-orbit ZFS model or a public reference selector.

use crate::basis::MolBasis;
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::dual::Taylor;
use crate::integrals::analytic_basis::AnalyticShell;
use crate::linalg::AccurateSum;
use crate::response::nuclear::spin_spin_operators::spin_spin_integrals;
use crate::response::nuclear::NuclearTaylor;
use crate::{DftError, Result};

pub type SpinSpinTensor = [[f64; 3]; 3];

#[derive(Clone, Debug)]
pub struct SpinSpinNuclearPath {
    pub tensors: SpinSpinNuclearCoefficients,
    pub response: crate::response::ResponseReport<NuclearTaylor>,
}

/// Compute an actual stationary all-electron integer-UHF/UKS path before
/// contracting the direct electron spin-spin contribution. This is an SCF
/// property, not a correlated-parent or constrained/spinor substitution.
#[allow(clippy::too_many_arguments)]
pub fn spin_spin_nuclear_path(
    molecule: &crate::system::Molecule,
    basis: &MolBasis,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    reference: &crate::scf::ScfResult,
    options: &crate::scf::ScfOptions,
    direction: &[[f64; 3]],
    order: usize,
    config: &crate::response::ResponseSolverConfig,
) -> Result<SpinSpinNuclearPath> {
    if order > 2 {
        return Err(DftError::Unsupported {
            feature: "spin-spin nuclear order above two".into(),
            missing: "higher-order spin-spin response acceptance".into(),
        });
    }
    if !reference.converged {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: f64::NAN,
        });
    }
    if !basis.ecps.is_empty()
        || options.reference != crate::scf::Reference::Uhf
        || !reference.unrestricted
        || options.electronic_temperature > 0.0
        || reference
            .occupations
            .iter()
            .flatten()
            .any(|&value| value != 0.0 && value != 1.0)
    {
        return Err(DftError::Unsupported {
            feature: "spin-spin nuclear response outside integer all-electron UHF/UKS".into(),
            missing: "dedicated ECP, constrained, spinor or ensemble spin-spin response".into(),
        });
    }
    let twice_spin = reference.n_alpha.saturating_sub(reference.n_beta);
    if twice_spin < 2 || direction.len() != molecule.len() {
        return Err(DftError::InvalidInput(
            "spin-spin nuclear response requires M_S=S>=1 and one direction per atom".into(),
        ));
    }
    crate::response::diagnostics::reference_diagnostics(reference)?;
    let response = crate::response::nuclear::nuclear_taylor(
        molecule, basis, functional, grid, reference, options, direction, order, config,
    )?;
    let tensors =
        spin_spin_nuclear_coefficients(basis, direction, &response.value, twice_spin, order)?;
    Ok(SpinSpinNuclearPath { tensors, response })
}

#[derive(Clone, Debug)]
pub struct SpinSpinPathDiagnostics {
    pub direction: Vec<[f64; 3]>,
    pub converged: bool,
    /// Nonconstant-order diagnostics are indexed from degree one, not zero.
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
    pub accumulation_bounds: Vec<SpinSpinTensor>,
    pub trace_residuals: Vec<f64>,
    pub antisymmetry_residuals: Vec<f64>,
    pub warnings: Vec<String>,
}

impl From<SpinSpinNuclearPath> for SpinSpinPathDiagnostics {
    fn from(path: SpinSpinNuclearPath) -> Self {
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
            accumulation_bounds: path.tensors.accumulation_bounds,
            trace_residuals: path.tensors.trace_residuals,
            antisymmetry_residuals: path.tensors.antisymmetry_residuals,
            warnings: path.response.warnings,
        }
    }
}

#[derive(Debug)]
pub struct SpinSpinNuclearDerivatives {
    pub order: usize,
    /// Direct spin-spin contribution in Hartree, H_D=S.D.S.
    pub value: SpinSpinTensor,
    /// [3*atom+axis][spin axis][spin axis], Hartree/bohr.
    pub gradient: Vec<SpinSpinTensor>,
    /// [coordinate][coordinate][spin axis][spin axis], Hartree/bohr^2.
    pub hessian: Option<Vec<Vec<SpinSpinTensor>>>,
    pub paths: Vec<SpinSpinPathDiagnostics>,
    pub converged: bool,
    _output_memory: crate::resources::MemoryReservation,
    _diagnostic_memory: Vec<crate::resources::MemoryReservation>,
}

fn retain_diagnostic(
    path: SpinSpinNuclearPath,
    paths: &mut Vec<SpinSpinPathDiagnostics>,
    memory: &mut Vec<crate::resources::MemoryReservation>,
) -> Result<()> {
    fn vector_bytes<T>(values: &Vec<T>) -> Result<usize> {
        crate::resources::checked_bytes(
            &[values.capacity(), std::mem::size_of::<T>()],
            "spin-spin retained diagnostics",
        )
    }
    let path = SpinSpinPathDiagnostics::from(path);
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
        vector_bytes(&path.accumulation_bounds)?,
        vector_bytes(&path.trace_residuals)?,
        vector_bytes(&path.antisymmetry_residuals)?,
        vector_bytes(&path.warnings)?,
    ]
    .into_iter()
    .chain(path.warnings.iter().map(|warning| warning.capacity()))
    {
        bytes = bytes
            .checked_add(part)
            .filter(|&value| value <= isize::MAX as usize)
            .ok_or_else(|| DftError::InvalidInput("spin-spin diagnostic size overflow".into()))?;
    }
    let lease = crate::resources::reserve(bytes, "spin-spin retained diagnostics")?;
    paths.push(path);
    memory.push(lease);
    Ok(())
}

/// Assemble full Cartesian derivatives by analytic path polarization. No
/// displaced SCF is used. Each path keeps its own raw convergence diagnostics;
/// the caller must refuse a nonconverged result before exposing final values.
#[allow(clippy::too_many_arguments)]
pub fn spin_spin_nuclear_derivatives(
    molecule: &crate::system::Molecule,
    basis: &MolBasis,
    functional: &crate::xc::ResolvedFunctional,
    grid: Option<&crate::grid::MolecularGrid>,
    reference: &crate::scf::ScfResult,
    options: &crate::scf::ScfOptions,
    order: usize,
    config: &crate::response::ResponseSolverConfig,
) -> Result<SpinSpinNuclearDerivatives> {
    if order > 2 {
        return Err(DftError::Unsupported {
            feature: "spin-spin nuclear order above two".into(),
            missing: "higher-order Cartesian spin-spin response".into(),
        });
    }
    let dimension = molecule
        .len()
        .checked_mul(3)
        .ok_or_else(|| DftError::InvalidInput("spin-spin Cartesian dimension overflow".into()))?;
    let square = dimension
        .checked_mul(dimension)
        .ok_or_else(|| DftError::InvalidInput("spin-spin Hessian dimension overflow".into()))?;
    let tensors = 1usize
        .checked_add(if order > 0 { dimension } else { 0 })
        .and_then(|value| value.checked_add(if order == 2 { square } else { 0 }))
        .ok_or_else(|| DftError::InvalidInput("spin-spin output count overflow".into()))?;
    let mut bytes = crate::resources::checked_bytes(
        &[tensors, std::mem::size_of::<SpinSpinTensor>()],
        "Cartesian spin-spin output",
    )?;
    let path_count = 1usize
        .checked_add(if order == 2 {
            dimension
                .checked_mul(dimension.checked_add(1).ok_or_else(|| {
                    DftError::InvalidInput("spin-spin path count overflow".into())
                })?)
                .ok_or_else(|| DftError::InvalidInput("spin-spin path count overflow".into()))?
                / 2
        } else if order == 1 {
            dimension
        } else {
            0
        })
        .ok_or_else(|| DftError::InvalidInput("spin-spin path count overflow".into()))?;
    for factors in [
        vec![path_count, std::mem::size_of::<SpinSpinPathDiagnostics>()],
        vec![
            path_count,
            std::mem::size_of::<crate::resources::MemoryReservation>(),
        ],
        vec![molecule.len(), std::mem::size_of::<[f64; 3]>()],
        vec![
            if order == 2 { dimension } else { 0 },
            std::mem::size_of::<Vec<SpinSpinTensor>>(),
        ],
    ] {
        bytes = bytes
            .checked_add(crate::resources::checked_bytes(
                &factors,
                "spin-spin output",
            )?)
            .filter(|&value| value <= isize::MAX as usize)
            .ok_or_else(|| DftError::InvalidInput("spin-spin output size overflow".into()))?;
    }
    let output_memory = crate::resources::reserve(bytes, "Cartesian spin-spin output")?;
    let zero = vec![[0.0; 3]; molecule.len()];
    let center = spin_spin_nuclear_path(
        molecule, basis, functional, grid, reference, options, &zero, 0, config,
    )?;
    let value = center.tensors.coefficients[0];
    let mut paths = Vec::with_capacity(path_count);
    let mut diagnostic_memory = Vec::with_capacity(path_count);
    retain_diagnostic(center, &mut paths, &mut diagnostic_memory)?;
    let mut gradient = if order > 0 {
        vec![[[0.0; 3]; 3]; dimension]
    } else {
        Vec::new()
    };
    let mut hessian = if order == 2 {
        Some(vec![vec![[[0.0; 3]; 3]; dimension]; dimension])
    } else {
        None
    };
    if order > 0 {
        for coordinate in 0..dimension {
            let mut direction = zero.clone();
            direction[coordinate / 3][coordinate % 3] = 1.0;
            let path = spin_spin_nuclear_path(
                molecule, basis, functional, grid, reference, options, &direction, order, config,
            )?;
            gradient[coordinate] = path.tensors.coefficients[1];
            if let Some(matrix) = &mut hessian {
                matrix[coordinate][coordinate] =
                    path.tensors.coefficients[2].map(|row| row.map(|v| 2.0 * v));
            }
            retain_diagnostic(path, &mut paths, &mut diagnostic_memory)?;
        }
    }
    if let Some(matrix) = &mut hessian {
        for i in 0..dimension {
            for j in 0..i {
                let mut direction = zero.clone();
                direction[i / 3][i % 3] = 1.0 / 2.0_f64.sqrt();
                direction[j / 3][j % 3] = 1.0 / 2.0_f64.sqrt();
                let path = spin_spin_nuclear_path(
                    molecule, basis, functional, grid, reference, options, &direction, 2, config,
                )?;
                let mixed = std::array::from_fn(|a| {
                    std::array::from_fn(|b| {
                        2.0 * path.tensors.coefficients[2][a][b]
                            - 0.5 * (matrix[i][i][a][b] + matrix[j][j][a][b])
                    })
                });
                matrix[i][j] = mixed;
                matrix[j][i] = mixed;
                retain_diagnostic(path, &mut paths, &mut diagnostic_memory)?;
            }
        }
    }
    let converged = paths.iter().all(|path| path.converged);
    Ok(SpinSpinNuclearDerivatives {
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

#[derive(Clone, Debug)]
pub struct SpinSpinNuclearCoefficients {
    /// Lab Cartesian D tensors, in Hartree/bohr^order divided by order!.
    pub coefficients: Vec<[[f64; 3]; 3]>,
    /// Product formation and accumulation bounds. Input/SCF and integral
    /// truncation errors remain separate.
    pub accumulation_bounds: Vec<[[f64; 3]; 3]>,
    /// Raw defects; no trace subtraction or tensor symmetrization is applied.
    pub trace_residuals: Vec<f64>,
    pub antisymmetry_residuals: Vec<f64>,
}

fn upper_add(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return a;
    }
    let value = a + b;
    if value.is_finite() {
        f64::from_bits(value.to_bits() + 1)
    } else {
        value
    }
}

fn scale_bound(factor: f64, bound: f64) -> f64 {
    if factor == 0.0 || bound == 0.0 {
        return 0.0;
    }
    let value = factor * bound;
    if value.is_finite() {
        f64::from_bits(value.to_bits() + 1)
    } else {
        value
    }
}

/// Return a bound on multiplication loss preceding the compensated sum.
/// Balance only extreme intermediate products; ordinary arithmetic retains
/// its existing operation order and represented high/low terms.
fn triple_product(sum: &mut AccurateSum, a: f64, b: f64, c: f64) -> f64 {
    if a == 0.0 || b == 0.0 || c == 0.0 {
        return 0.0;
    }
    let threshold = 2.0 * f64::MIN_POSITIVE / f64::EPSILON;
    let original = a * b;
    let (left, right, last) = if original.is_finite() && original.abs() >= threshold {
        (a, b, c)
    } else {
        let mut values = [a, b, c];
        values.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
        (values[0], values[2], values[1])
    };
    let product = left * right;
    sum.add_product(product, last);
    sum.add_product(left.mul_add(right, -product), last);
    if product.abs() < threshold {
        // Only a genuinely tiny nonzero-factor product reaches this case.
        // The missing FMA remainder is below one subnormal quantum before
        // multiplication by `last`. Round its propagated bound upward.
        let bound = f64::from_bits(1) * last.abs();
        f64::from_bits(bound.to_bits() + 1)
    } else {
        0.0
    }
}

/// Contract all direct and exchange terms in the same moving AO basis.
/// The supplied response must describe the requested integer collinear state;
/// the caller must check its stationarity/convergence diagnostics separately.
/// `twice_spin` is 2S in H_D = S.D.S and must exceed one. SOC contributions,
/// spin projection of contaminated determinants and generalized spinors are
/// outside this explicitly spin-spin-only contraction.
pub fn spin_spin_nuclear_coefficients(
    basis: &MolBasis,
    direction: &[[f64; 3]],
    response: &NuclearTaylor,
    twice_spin: usize,
    order: usize,
) -> Result<SpinSpinNuclearCoefficients> {
    match order {
        0 => spin_spin_contraction::<1>(basis, direction, response, twice_spin, order),
        1 => spin_spin_contraction::<2>(basis, direction, response, twice_spin, order),
        2 => spin_spin_contraction::<3>(basis, direction, response, twice_spin, order),
        _ => Err(DftError::Unsupported {
            feature: "spin-spin nuclear order above two".into(),
            missing: "higher-order spin-spin contraction acceptance".into(),
        }),
    }
}

fn spin_spin_contraction<const W: usize>(
    basis: &MolBasis,
    direction: &[[f64; 3]],
    response: &NuclearTaylor,
    twice_spin: usize,
    order: usize,
) -> Result<SpinSpinNuclearCoefficients> {
    if order > 2 {
        return Err(DftError::Unsupported {
            feature: "spin-spin nuclear order above two".into(),
            missing: "higher-order spin-spin contraction acceptance".into(),
        });
    }
    let n = basis.nao_sph;
    if twice_spin < 2
        || n == 0
        || !basis.ecps.is_empty()
        || direction != response.direction.as_slice()
        || basis.shells.iter().any(|shell| {
            shell.atom >= direction.len()
                || shell
                    .sph_off
                    .checked_add(crate::basis::n_sph(shell.l))
                    .is_none_or(|end| end > n)
        })
        || direction.iter().flatten().any(|value| !value.is_finite())
        || response.density_coefficients.len() <= order
        || response.density_corrections.len() <= order
        || response.density_coefficients[..=order]
            .iter()
            .chain(&response.density_corrections[..=order])
            .flatten()
            .any(|matrix| {
                matrix.rows != n
                    || matrix.cols != n
                    || matrix.as_slice().iter().any(|value| !value.is_finite())
            })
    {
        return Err(DftError::InvalidInput(
            "spin-spin contraction requires complete finite all-electron collinear data and S >= 1"
                .into(),
        ));
    }
    if response.field_direction.iter().any(|value| *value != 0.0) {
        return Err(DftError::Unsupported {
            feature: "mixed electric-field spin-spin nuclear coefficients".into(),
            missing: "dedicated mixed-field spin-spin acceptance".into(),
        });
    }
    // The streamed algorithm retains one shell quartet, not nine full AO^4 arrays.
    let max_shell = basis
        .shells
        .iter()
        .map(|shell| crate::basis::n_sph(shell.l))
        .max()
        .ok_or_else(|| DftError::InvalidInput("empty spin-spin shell space".into()))?;
    let bytes = crate::resources::checked_bytes(
        &[
            max_shell,
            max_shell,
            max_shell,
            max_shell,
            9,
            std::mem::size_of::<Taylor<W>>(),
        ],
        "spin-spin output block",
    )?;
    let _workspace = crate::resources::reserve(bytes, "streamed spin-spin output block")?;
    // Internal Gaussian/AD scratch is separate; this is not an RSS guarantee.
    let shells: Vec<_> = basis
        .shells
        .iter()
        .map(|shell| {
            let center = std::array::from_fn(|axis| {
                Taylor::<W>::from_coefficients(&[
                    shell.center.get(axis),
                    direction[shell.atom][axis],
                ])
            });
            AnalyticShell::from_placed(shell, center)
        })
        .collect::<Result<_>>()?;
    let mut sums = vec![[AccurateSum::default(); 9]; order + 1];
    let mut formation_bounds = vec![[0.0; 9]; order + 1];
    let weights = [1.0, -1.0];
    for (ia, a) in shells.iter().enumerate() {
        for (ib, b) in shells.iter().enumerate() {
            for (ic, c) in shells.iter().enumerate() {
                for (id, d) in shells.iter().enumerate() {
                    let block = spin_spin_integrals(a, b, c, d, order)?;
                    for (flat, tensor) in block.values.iter().enumerate() {
                        if tensor
                            .iter()
                            .any(|value| value.c[..=order].iter().any(|v| !v.is_finite()))
                        {
                            return Err(DftError::InvalidInput(
                                "nonfinite spin-spin operator coefficient".into(),
                            ));
                        }
                        let mut remaining = flat;
                        let mut index = [0; 4];
                        for slot in (0..4).rev() {
                            index[slot] = remaining % block.shape[slot];
                            remaining /= block.shape[slot];
                        }
                        let mu = basis.shells[ia].sph_off + index[0];
                        let nu = basis.shells[ib].sph_off + index[1];
                        let kappa = basis.shells[ic].sph_off + index[2];
                        let lambda = basis.shells[id].sph_off + index[3];
                        for degree in 0..=order {
                            for left_order in 0..=degree {
                                for right_order in 0..=degree - left_order {
                                    let operator_order = degree - left_order - right_order;
                                    for (left_spin, left_weight) in weights.iter().enumerate() {
                                        for (right_spin, right_weight) in weights.iter().enumerate()
                                        {
                                            for left in [
                                                &response.density_coefficients[left_order]
                                                    [left_spin],
                                                &response.density_corrections[left_order]
                                                    [left_spin],
                                            ] {
                                                for right in [
                                                    &response.density_coefficients[right_order]
                                                        [right_spin],
                                                    &response.density_corrections[right_order]
                                                        [right_spin],
                                                ] {
                                                    for component in 0..9 {
                                                        let value = left_weight
                                                            * right_weight
                                                            * tensor[component].c[operator_order];
                                                        let direct_loss = triple_product(
                                                            &mut sums[degree][component],
                                                            left[(mu, nu)],
                                                            right[(kappa, lambda)],
                                                            value,
                                                        );
                                                        let exchange_loss = triple_product(
                                                            &mut sums[degree][component],
                                                            left[(mu, kappa)],
                                                            right[(nu, lambda)],
                                                            -value,
                                                        );
                                                        formation_bounds[degree][component] =
                                                            upper_add(
                                                                upper_add(
                                                                    formation_bounds[degree]
                                                                        [component],
                                                                    direct_loss,
                                                                ),
                                                                exchange_loss,
                                                            );
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
            }
        }
    }
    // Ordered AO pairs count each occupied pair twice. In atomic units,
    // mu0/(4pi)*(gamma_e*hbar)^2 = alpha^2*g_e^2/4.
    let factor = FINE_STRUCTURE.powi(2) * ELECTRON_G_FACTOR.powi(2)
        / (8.0 * twice_spin as f64 * (twice_spin - 1) as f64);
    let coefficients: Vec<[[f64; 3]; 3]> = sums
        .iter()
        .map(|tensor| {
            std::array::from_fn(|i| std::array::from_fn(|j| factor * tensor[3 * i + j].value()))
        })
        .collect();
    let accumulation_bounds = sums
        .iter()
        .zip(&formation_bounds)
        .map(|(tensor, formation)| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    scale_bound(
                        factor.abs(),
                        upper_add(tensor[3 * i + j].roundoff_bound(), formation[3 * i + j]),
                    )
                })
            })
        })
        .collect();
    if coefficients
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(DftError::InvalidInput(
            "nonfinite spin-spin contraction".into(),
        ));
    }
    let trace_residuals = coefficients
        .iter()
        .map(|tensor| (tensor[0][0] + tensor[1][1] + tensor[2][2]).abs())
        .collect();
    let antisymmetry_residuals = coefficients
        .iter()
        .map(|tensor| {
            (0..3)
                .flat_map(|i| (0..3).map(move |j| (tensor[i][j] - tensor[j][i]).abs()))
                .fold(0.0_f64, f64::max)
        })
        .collect();
    Ok(SpinSpinNuclearCoefficients {
        coefficients,
        accumulation_bounds,
        trace_residuals,
        antisymmetry_residuals,
    })
}

#[cfg(test)]
mod product_precision_tests {
    use super::*;

    #[test]
    fn finite_triples_survive_intermediate_underflow_and_overflow() {
        for (a, b, c, expected) in [
            (
                2.0_f64.powi(-600),
                2.0_f64.powi(-600),
                2.0_f64.powi(1000),
                2.0_f64.powi(-200),
            ),
            (
                2.0_f64.powi(700),
                2.0_f64.powi(700),
                2.0_f64.powi(-600),
                2.0_f64.powi(800),
            ),
            (
                -2.0_f64.powi(700),
                2.0_f64.powi(700),
                2.0_f64.powi(-600),
                -2.0_f64.powi(800),
            ),
        ] {
            let mut sum = AccurateSum::default();
            let bound = triple_product(&mut sum, a, b, c);
            assert_eq!(sum.value(), expected);
            assert_eq!(bound, 0.0);
        }
    }

    #[test]
    fn zero_and_unrepresentable_products_are_distinguished() {
        let mut zero = AccurateSum::default();
        assert_eq!(
            triple_product(&mut zero, 2.0_f64.powi(700), 2.0_f64.powi(700), 0.0),
            0.0
        );
        assert_eq!(zero.value(), 0.0);
        assert_eq!(zero.roundoff_bound(), 0.0);
        let mut tiny = AccurateSum::default();
        let bound = triple_product(
            &mut tiny,
            2.0_f64.powi(-600),
            2.0_f64.powi(-600),
            2.0_f64.powi(-600),
        );
        assert_eq!(tiny.value(), 0.0);
        assert!(bound >= f64::from_bits(1));
        assert_eq!(scale_bound(0.25, bound), f64::from_bits(1));
        assert_eq!(scale_bound(0.25, 0.0), 0.0);
    }
}
