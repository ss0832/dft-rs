// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving one-electron Pauli g tensor on an integer collinear HF reference.
//! London phases, all nuclear operators and the stationary magnetic density
//! share the physical AO representation. Two-electron SOC, picture change,
//! spin-flip and constrained/spinor references are not supplied by this model.

use crate::basis::MolBasis;
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::grid::MolecularGrid;
use crate::linalg::{AccurateSum, Matrix};
use crate::math::Vec3;
use crate::response::field_path::FieldPathTaylor;
use crate::response::nuclear::one_body::nuclear_one_body_coefficients;
use crate::response::nuclear::spin_orbit_operators::spin_orbit_operator_path;
use crate::response::nuclear::{magnetic_nuclear_field_taylor, MixedMagneticNuclearTaylor};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::magnetic::{magnetic_field_scf, MagneticFieldOptions, MagneticFieldResult};
use crate::scf::{Reference, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub type GTensor = [[f64; 3]; 3];

/// Immutable borrowed inputs prevent reusing a magnetic response at a
/// different geometry, functional, grid, reference or resource configuration.
pub struct PauliGNuclearContext<'a> {
    molecule: &'a Molecule,
    basis: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    reference: &'a ScfResult,
    options: &'a ScfOptions,
    config: &'a ResponseSolverConfig,
    origin: Vec3,
    charges: Vec<f64>,
    magnetic_reference: MagneticFieldResult,
    pure_field: [MixedMagneticNuclearTaylor; 3],
    orbital_time_reversal: bool,
    _retained_memory: crate::resources::MemoryReservation,
}

pub struct PauliGNuclearPath {
    /// Coefficient k is the derivative/k!, in inverse bohr^k.
    pub tensor: Vec<GTensor>,
    pub mass_correction: Vec<f64>,
    pub gauge_correction: Vec<GTensor>,
    pub spin_orbit: Vec<GTensor>,
    /// Accumulation-only bounds; response/integral input errors are separate.
    pub accumulation_bounds: Vec<GTensor>,
    pub real_response: ResponseReport<crate::response::nuclear::NuclearTaylor>,
    /// Raw diagnostics for actual solves, ordered by mixed_report_directions.
    pub mixed_reports: Vec<FieldPathTaylor>,
    /// Actual directions for retained reports; never inferred diagnostics.
    pub mixed_report_directions: Vec<(usize, i32)>,
    /// Present only when both paths were actually solved and compared.
    pub explicit_pair_parity_defect: Option<f64>,
    _retained_memory: crate::resources::MemoryReservation,
}

impl<'a> PauliGNuclearContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        reference: &'a ScfResult,
        options: &'a ScfOptions,
        config: &'a ResponseSolverConfig,
        origin: Vec3,
        charges: &[f64],
    ) -> Result<Self> {
        config.validate()?;
        if molecule.is_empty()
            || charges.len() != molecule.len()
            || charges.iter().any(|v| !v.is_finite())
            || origin.to_array().iter().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "Pauli g requires finite origin and one SOC charge per atom".into(),
            ));
        }
        if !basis.ecps.is_empty()
            || options.reference != Reference::Uhf
            || !reference.unrestricted
            || !reference.has_integer_occupations()
            || reference.electronic_temperature != 0.0
            || reference.n_alpha <= reference.n_beta
            || !functional.parts.is_empty()
            || functional.hybrid.cx != 1.0
            || functional.hybrid.clr != 0.0
        {
            return Err(DftError::Unsupported {
                feature:"nuclear Pauli g outside integer all-electron collinear HF".into(),
                missing:"independently accepted semilocal, constrained, spinor and additional SOC response".into(),
            });
        }
        if !reference.converged {
            return Err(DftError::ScfNotConverged {
                iterations: reference.iterations,
                error: f64::NAN,
            });
        }
        crate::response::diagnostics::reference_diagnostics(reference)?;
        let field = MagneticFieldOptions {
            field: [0.0; 3],
            spin_zeeman: false,
            series_order: Some(0),
            series_tolerance: config.tolerance,
        };
        let mut magnetic_reference = magnetic_field_scf(
            molecule, basis, functional, grid, reference, options, &field,
        )?;
        if !magnetic_reference.converged {
            return Err(DftError::ScfNotConverged {
                iterations: magnetic_reference.iterations,
                error: magnetic_reference.residual,
            });
        }
        let tolerance = config
            .acceptance()
            .max(options.conv_tol_grad)
            .max(options.e_tol);
        let mut defect =
            (magnetic_reference.energy - reference.energy).abs() / reference.energy.abs().max(1.0);
        for spin in 0..2 {
            for (a, b) in magnetic_reference.density[spin]
                .re
                .as_slice()
                .iter()
                .zip(reference.density[spin].as_slice())
            {
                defect = defect.max((a - b).abs());
            }
        }
        if defect > tolerance || !defect.is_finite() {
            return Err(DftError::InvalidInput(format!(
                "zero-field magnetic state differs from nuclear reference: {defect:.16e}"
            )));
        }
        // At zero field the physical AO density is real and independent of
        // a translation of the integral origin. Subsequent London response
        // uses this explicit fixed laboratory origin consistently.
        magnetic_reference.shift = origin.to_array().map(|v| -v);
        let bytes = crate::resources::checked_bytes(
            &[3, 2, 4, basis.nao_sph, basis.nao_sph, 2, 8],
            "retained pure magnetic AO density coefficients",
        )?;
        let retained_memory =
            crate::resources::reserve(bytes, "retained pure magnetic AO density coefficients")?;
        let zero = vec![[0.0; 3]; molecule.len()];
        let mut pure = Vec::with_capacity(3);
        for axis in 0..3 {
            let mut magnetic = [0.0; 3];
            magnetic[axis] = 1.0;
            pure.push(magnetic_nuclear_field_taylor(
                molecule,
                basis,
                functional,
                grid,
                options,
                &field,
                &magnetic_reference,
                &zero,
                magnetic,
                3,
                config,
            )?);
        }
        let pure_field = pure
            .try_into()
            .map_err(|_| DftError::InvalidInput("incomplete magnetic axis layout".into()))?;
        Ok(Self {
            molecule,
            basis,
            functional,
            grid,
            reference,
            options,
            config,
            origin,
            charges: charges.to_vec(),
            magnetic_reference,
            pure_field,
            orbital_time_reversal: false,
            _retained_memory: retained_memory,
        })
    }

    pub fn pure_field(&self) -> &[MixedMagneticNuclearTaylor; 3] {
        &self.pure_field
    }

    /// Reconstruct the reversed nuclear path using exact orbital complex
    /// conjugation at B=0, without spin Zeeman coupling. The constructor
    /// already restricts the reference to real integer collinear HF.
    /// Im P_minus[k]=(-1)^(k+1) Im P_plus[k]. Only actually solved paths
    /// receive diagnostics; the default method retains explicit pairs.
    pub fn with_orbital_time_reversal(mut self) -> Self {
        self.orbital_time_reversal = true;
        self
    }

    pub fn nuclear_path(&self, direction: &[[f64; 3]], order: usize) -> Result<PauliGNuclearPath> {
        if order > 2 {
            return Err(DftError::Unsupported {
                feature: "Pauli g nuclear order above two".into(),
                missing: "higher nuclear g derivative acceptance".into(),
            });
        }
        if direction.len() != self.molecule.len()
            || direction.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "Pauli g requires one finite direction per atom".into(),
            ));
        }
        let width = order + 1;
        let n = self.basis.nao_sph;
        // Explicit dense returned arrays and temporary coefficient matrices;
        // integral/solver scratch has separate backend accounting. This is
        // a managed workspace estimate, not a process RSS guarantee.
        let retained_bytes =
            crate::resources::checked_bytes(&[width, n, n, 9, 8], "retained nuclear g matrices")?;
        let retained_memory =
            crate::resources::reserve(retained_bytes, "retained nuclear g matrices")?;
        let scratch_bytes = crate::resources::checked_bytes(
            &[width, n, n, 24, 8],
            "mixed nuclear g density workspace",
        )?;
        let _scratch =
            crate::resources::reserve(scratch_bytes, "mixed nuclear g density workspace")?;
        let real = crate::response::nuclear::nuclear_taylor(
            self.molecule,
            self.basis,
            self.functional,
            self.grid,
            self.reference,
            self.options,
            direction,
            order,
            self.config,
        )?;
        if !real.converged {
            return Err(response_failure(&real.iterations, &real.residuals));
        }
        let twice_spin = self.reference.n_alpha - self.reference.n_beta;
        let factor = FINE_STRUCTURE.powi(2) / twice_spin as f64;
        let mass = crate::properties::g_mass_correction::mass_correction_nuclear_coefficients(
            self.basis,
            direction,
            &real.value,
            twice_spin,
            order,
        )?;
        let mut magnetic_density = Vec::with_capacity(3);
        let mut mixed_reports = Vec::new();
        let mut mixed_report_directions = Vec::new();
        let mut explicit_pair_parity_defect: Option<f64> = None;
        let field = MagneticFieldOptions {
            field: [0.0; 3],
            spin_zeeman: false,
            series_order: Some(0),
            series_tolerance: self.config.tolerance,
        };
        for axis in 0..3 {
            let mut magnetic = [0.0; 3];
            magnetic[axis] = 1.0;
            let pair = if order > 0 {
                let plus = magnetic_nuclear_field_taylor(
                    self.molecule,
                    self.basis,
                    self.functional,
                    self.grid,
                    self.options,
                    &field,
                    &self.magnetic_reference,
                    direction,
                    magnetic,
                    order + 1,
                    self.config,
                )?;
                let minus = if self.orbital_time_reversal {
                    None
                } else {
                    let negative: Vec<_> = direction.iter().map(|v| v.map(|x| -x)).collect();
                    let actual_minus = magnetic_nuclear_field_taylor(
                        self.molecule,
                        self.basis,
                        self.functional,
                        self.grid,
                        self.options,
                        &field,
                        &self.magnetic_reference,
                        &negative,
                        magnetic,
                        order + 1,
                        self.config,
                    )?;
                    Some(actual_minus)
                };
                Some((plus, minus))
            } else {
                None
            };
            let mut density = Vec::with_capacity(width);
            for degree in 0..width {
                let mut matrices = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
                for (spin, matrix) in matrices.iter_mut().enumerate() {
                    for i in 0..n {
                        for j in 0..n {
                            let mut sum = AccurateSum::default();
                            if degree == 0 {
                                sum.add_product(
                                    1.0,
                                    self.pure_field[axis].density[spin][1].im[(i, j)],
                                );
                            } else if let Some((plus, minus)) = &pair {
                                if let Some(minus) = minus {
                                    sum.add_product(0.5, plus.density[spin][degree + 1].im[(i, j)]);
                                    sum.add_product(
                                        if degree == 1 { -0.5 } else { 0.5 },
                                        minus.density[spin][degree + 1].im[(i, j)],
                                    );
                                } else {
                                    sum.add_product(1.0, plus.density[spin][degree + 1].im[(i, j)]);
                                }
                                if degree == 2 {
                                    sum.add_product(
                                        -1.0,
                                        self.pure_field[axis].density[spin][3].im[(i, j)],
                                    );
                                }
                            }
                            matrix[(i, j)] = sum.value();
                        }
                    }
                }
                density.push(matrices);
            }
            magnetic_density.push(density);
            if let Some((plus, minus)) = pair {
                if let Some(minus) = minus {
                    let mut defect = 0.0_f64;
                    for spin in 0..2 {
                        for degree in 0..=order + 1 {
                            for i in 0..n {
                                for j in 0..n {
                                    let real_sign = if degree % 2 == 0 { 1.0 } else { -1.0 };
                                    defect = defect.max(
                                        (minus.density[spin][degree].re[(i, j)]
                                            - real_sign * plus.density[spin][degree].re[(i, j)])
                                            .abs(),
                                    );
                                    defect = defect.max(
                                        (minus.density[spin][degree].im[(i, j)]
                                            + real_sign * plus.density[spin][degree].im[(i, j)])
                                            .abs(),
                                    );
                                }
                            }
                        }
                    }
                    explicit_pair_parity_defect =
                        Some(explicit_pair_parity_defect.unwrap_or(0.0).max(defect));
                    mixed_reports.push(plus.report);
                    mixed_report_directions.push((axis, 1));
                    mixed_reports.push(minus.report);
                    mixed_report_directions.push((axis, -1));
                } else {
                    mixed_reports.push(plus.report);
                    mixed_report_directions.push((axis, 1));
                }
            }
        }
        let mut gc: Vec<[[AccurateSum; 3]; 3]> = (0..width)
            .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| AccurateSum::default())))
            .collect();
        let mut so = gc.clone();
        let mut bounds = vec![[[0.0; 3]; 3]; width];
        let zero: Vec<_> = (0..width)
            .map(|_| [Matrix::zeros(n, n), Matrix::zeros(n, n)])
            .collect();
        for (nucleus, charge) in self.charges.iter().enumerate() {
            if *charge == 0.0 {
                continue;
            }
            let operator = spin_orbit_operator_path(
                self.molecule,
                self.basis,
                nucleus,
                direction,
                self.origin,
                [0.0; 3],
                true,
                order,
            )?;
            for (a, magnetic_axis) in magnetic_density.iter().enumerate() {
                for b in 0..3 {
                    let gauge = nuclear_one_body_coefficients(
                        &real.value.density_coefficients,
                        &real.value.density_corrections,
                        &operator.mixed[a][b],
                        [1.0, -1.0],
                    )?;
                    let spin_orbit = nuclear_one_body_coefficients(
                        magnetic_axis,
                        &zero,
                        &operator.paramagnetic[b],
                        [1.0, -1.0],
                    )?;
                    for degree in 0..width {
                        gc[degree][a][b].add_product(factor * charge, gauge.coefficients[degree]);
                        so[degree][a][b]
                            .add_product(factor * charge, spin_orbit.coefficients[degree]);
                        bounds[degree][a][b] += factor.abs()
                            * charge.abs()
                            * (gauge.accumulation_bounds[degree]
                                + spin_orbit.accumulation_bounds[degree]);
                    }
                }
            }
        }
        let gauge_correction: Vec<GTensor> = gc
            .iter()
            .map(|row| std::array::from_fn(|a| std::array::from_fn(|b| row[a][b].value())))
            .collect();
        let spin_orbit: Vec<GTensor> = so
            .iter()
            .map(|row| std::array::from_fn(|a| std::array::from_fn(|b| row[a][b].value())))
            .collect();
        let tensor = (0..width)
            .map(|degree| {
                std::array::from_fn(|a| {
                    std::array::from_fn(|b| {
                        let mut sum = AccurateSum::default();
                        sum.add_product(1.0, gauge_correction[degree][a][b]);
                        sum.add_product(1.0, spin_orbit[degree][a][b]);
                        if a == b {
                            sum.add_product(1.0, mass.coefficients[degree]);
                            if degree == 0 {
                                sum.add_product(1.0, ELECTRON_G_FACTOR);
                            }
                        }
                        bounds[degree][a][b] += gc[degree][a][b].roundoff_bound()
                            + so[degree][a][b].roundoff_bound()
                            + sum.roundoff_bound();
                        if a == b {
                            bounds[degree][a][b] += mass.accumulation_bounds[degree];
                        }
                        sum.value()
                    })
                })
            })
            .collect();
        Ok(PauliGNuclearPath {
            tensor,
            mass_correction: mass.coefficients,
            gauge_correction,
            spin_orbit,
            accumulation_bounds: bounds,
            real_response: real,
            mixed_reports,
            mixed_report_directions,
            explicit_pair_parity_defect,
            _retained_memory: retained_memory,
        })
    }
}

fn response_failure(iterations: &[usize], residuals: &[f64]) -> DftError {
    let residual = residuals
        .iter()
        .copied()
        .find(|r| !r.is_finite())
        .unwrap_or_else(|| {
            if residuals.is_empty() {
                f64::NAN
            } else {
                residuals.iter().copied().fold(0.0_f64, f64::max)
            }
        });
    DftError::ResponseNotConverged {
        iterations: iterations.iter().copied().max().unwrap_or(0),
        residual,
    }
}

#[cfg(test)]
mod error_contract_tests {
    use super::{response_failure, DftError};
    #[test]
    fn failed_response_is_solver_failure_not_missing_capability() {
        assert!(
            matches!(response_failure(&[3,7],&[0.1,0.01]),DftError::ResponseNotConverged{iterations:7,residual} if residual==0.1)
        );
        assert!(
            matches!(response_failure(&[],&[]),DftError::ResponseNotConverged{iterations:0,residual} if residual.is_nan())
        );
    }
}

/// One physical value or unnormalized nuclear derivative. The value is
/// dimensionless; order-k derivatives have units bohr^(-k).
#[derive(Clone, Copy, Debug)]
pub struct PauliGDerivative {
    pub tensor: GTensor,
    pub mass_correction: f64,
    pub gauge_correction: GTensor,
    pub spin_orbit: GTensor,
}

impl PauliGDerivative {
    fn coefficient(path: &PauliGNuclearPath, degree: usize) -> Self {
        Self {
            tensor: path.tensor[degree],
            mass_correction: path.mass_correction[degree],
            gauge_correction: path.gauge_correction[degree],
            spin_orbit: path.spin_orbit[degree],
        }
    }

    pub fn components(&self) -> [f64; 28] {
        std::array::from_fn(|i| match i {
            0 => self.mass_correction,
            1..=9 => self.gauge_correction[(i - 1) / 3][(i - 1) % 3],
            10..=18 => self.spin_orbit[(i - 10) / 3][(i - 10) % 3],
            _ => self.tensor[(i - 19) / 3][(i - 19) % 3],
        })
    }

    fn combination(terms: &[(f64, Self)]) -> Self {
        let result: [f64; 28] = std::array::from_fn(|i| {
            let mut sum = AccurateSum::default();
            for (scale, value) in terms {
                sum.add_product(*scale, value.components()[i]);
            }
            sum.value()
        });
        Self {
            tensor: std::array::from_fn(|a| std::array::from_fn(|b| result[19 + 3 * a + b])),
            mass_correction: result[0],
            gauge_correction: std::array::from_fn(|a| {
                std::array::from_fn(|b| result[1 + 3 * a + b])
            }),
            spin_orbit: std::array::from_fn(|a| std::array::from_fn(|b| result[10 + 3 * a + b])),
        }
    }
}

pub struct PauliGCartesianDerivatives {
    pub order: usize,
    pub value: PauliGDerivative,
    /// [3*atom+axis], bohr^(-1); None when unrequested.
    pub gradient: Option<Vec<PauliGDerivative>>,
    /// [coordinate][coordinate], bohr^(-2); None when unrequested.
    pub hessian: Option<Vec<Vec<PauliGDerivative>>>,
    /// Actual directional solves and diagnostics, retained without projection.
    pub paths: Vec<PauliGNuclearPath>,
    _output_memory: crate::resources::MemoryReservation,
}

impl PauliGNuclearContext<'_> {
    /// Assemble all Cartesian derivatives analytically by polarization.
    /// This inherits the context's restricted physical model and optional
    /// orbital time-reversal strategy; it performs no displaced SCFs.
    pub fn cartesian_derivatives(&self, order: usize) -> Result<PauliGCartesianDerivatives> {
        if order > 2 {
            return Err(DftError::Unsupported {
                feature: "Cartesian Pauli g nuclear order above two".into(),
                missing: "accepted higher mixed magnetic/nuclear response".into(),
            });
        }
        if self.molecule.is_empty() {
            return Err(DftError::InvalidInput(
                "Cartesian Pauli g requires a nonempty molecule".into(),
            ));
        }
        if self.molecule.len() > 12 {
            return Err(DftError::Unsupported {
                feature: "Cartesian Pauli g for more than twelve nuclei".into(),
                missing: "accepted larger-system resource and derivative validation".into(),
            });
        }
        let n = 3 * self.molecule.len();
        let count = match order {
            0 => 1,
            1 => n,
            _ => n * (n + 1) / 2,
        };
        let elements = 1 + if order > 0 { n } else { 0 } + if order > 1 { n * n } else { 0 };
        let values = crate::resources::checked_bytes(
            &[elements + n, std::mem::size_of::<PauliGDerivative>()],
            "Cartesian Pauli g tensor outputs and diagonal scratch",
        )?;
        let paths_size = crate::resources::checked_bytes(
            &[count, std::mem::size_of::<PauliGNuclearPath>()],
            "Cartesian Pauli g path containers",
        )?;
        let rows_size = if order > 1 {
            crate::resources::checked_bytes(
                &[n, std::mem::size_of::<Vec<PauliGDerivative>>()],
                "Cartesian Pauli g Hessian rows",
            )?
        } else {
            0
        };
        let output_memory = crate::resources::reserve(
            values.saturating_add(paths_size).saturating_add(rows_size),
            "Cartesian Pauli g output containers",
        )?;
        let mut paths = Vec::with_capacity(count);
        if order == 0 {
            paths.push(self.nuclear_path(&vec![[0.0; 3]; self.molecule.len()], 0)?);
            let value = PauliGDerivative::coefficient(&paths[0], 0);
            return Ok(PauliGCartesianDerivatives {
                order,
                value,
                gradient: None,
                hessian: None,
                paths,
                _output_memory: output_memory,
            });
        }
        let mut gradient = Vec::with_capacity(n);
        let mut diagonal = Vec::with_capacity(if order == 2 { n } else { 0 });
        for coordinate in 0..n {
            let mut direction = vec![[0.0; 3]; self.molecule.len()];
            direction[coordinate / 3][coordinate % 3] = 1.0;
            let path = self.nuclear_path(&direction, order)?;
            gradient.push(PauliGDerivative::coefficient(&path, 1));
            if order == 2 {
                diagonal.push(PauliGDerivative::combination(&[(
                    2.0,
                    PauliGDerivative::coefficient(&path, 2),
                )]));
            }
            paths.push(path);
        }
        let mut hessian = if order == 2 {
            Some(vec![vec![diagonal[0]; n]; n])
        } else {
            None
        };
        if let Some(hessian) = &mut hessian {
            for i in 0..n {
                hessian[i][i] = diagonal[i];
            }
            for i in 0..n {
                for j in i + 1..n {
                    let mut direction = vec![[0.0; 3]; self.molecule.len()];
                    direction[i / 3][i % 3] = std::f64::consts::FRAC_1_SQRT_2;
                    direction[j / 3][j % 3] = std::f64::consts::FRAC_1_SQRT_2;
                    let path = self.nuclear_path(&direction, 2)?;
                    let mixed = PauliGDerivative::combination(&[
                        (2.0, PauliGDerivative::coefficient(&path, 2)),
                        (-0.5, diagonal[i]),
                        (-0.5, diagonal[j]),
                    ]);
                    hessian[i][j] = mixed;
                    hessian[j][i] = mixed;
                    paths.push(path);
                }
            }
        }
        debug_assert_eq!(paths.len(), count);
        Ok(PauliGCartesianDerivatives {
            order,
            value: PauliGDerivative::coefficient(&paths[0], 0),
            gradient: Some(gradient),
            hessian,
            paths,
            _output_memory: output_memory,
        })
    }
}
