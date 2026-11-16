// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic frequency-resolved molecular response using the production
//! adiabatic KS/HF Casida operator, including its XC, nonlocal and reaction
//! field kernels. There is no sum-over-states truncation or finite field.

use super::complex::Complex;
use super::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use super::tddft::{TdaSolver, TdaSpin};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::dipole_matrices;
use crate::math::Vec3;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Frequency {
    Static,
    Retarded { omega: f64, damping: f64 },
    Imaginary { frequency: f64 },
}

impl Frequency {
    pub fn validate(self) -> Result<()> {
        let valid = match self {
            Self::Static => true,
            Self::Retarded { omega, damping } => {
                omega.is_finite() && damping.is_finite() && damping >= 0.0
            }
            Self::Imaginary { frequency } => frequency.is_finite() && frequency >= 0.0,
        };
        if valid {
            Ok(())
        } else {
            Err(DftError::InvalidInput("response frequency must be finite, with non-negative damping or imaginary frequency".into()))
        }
    }
    pub fn energy(self) -> Result<Complex> {
        self.validate()?;
        Ok(match self {
            Self::Static => Complex::ZERO,
            Self::Retarded { omega, damping } => Complex::new(omega, damping),
            Self::Imaginary { frequency } => Complex::new(0.0, frequency),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ResponseReport<T> {
    pub value: T,
    pub shape: Vec<usize>,
    pub functional: String,
    pub reference: super::diagnostics::ReferenceDiagnostics,
    pub units: &'static str,
    pub model: &'static str,
    pub axis_convention: &'static str,
    pub converged: bool,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct DynamicPolarizability {
    pub frequency: Frequency,
    pub alpha: [[Complex; 3]; 3],
    pub isotropic: Complex,
    pub reciprocity_defect: f64,
}

pub(super) struct CasidaOperator<'a> {
    pub(super) solver: TdaSolver<'a>,
    pub(super) frequency: Complex,
}

impl LinearOperator for CasidaOperator<'_> {
    fn dimension(&self) -> usize {
        2 * self.solver.dim()
    }
    fn structure(&self) -> OperatorStructure {
        if self.frequency.im == 0.0 {
            OperatorStructure::Hermitian
        } else {
            OperatorStructure::General
        }
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        let n = self.solver.dim();
        if x.len() != 2 * n {
            return Err(DftError::InvalidInput(
                "invalid Casida amplitude dimension".into(),
            ));
        }
        let mut out = krylov::apply_real_linear(x, |v| {
            // Exact zero transition densities have zero Coulomb, exchange and
            // XC images. Static electric sources keep the difference block zero;
            // skip its AO contractions without screening small nonzero amplitudes.
            let sum = if v[..n].iter().all(|value| *value == 0.0) {
                vec![0.0; n]
            } else {
                self.solver.apb_vec(&v[..n])?
            };
            let difference = if v[n..].iter().all(|value| *value == 0.0) {
                vec![0.0; n]
            } else {
                self.solver.amb_vec(&v[n..])?
            };
            let mut result = vec![0.0; 2 * n];
            for k in 0..n {
                result[k] = sum[k] - self.frequency.re * v[n + k];
                result[n + k] = difference[k] - self.frequency.re * v[k];
            }
            Ok(result)
        })?;
        if self.frequency.im != 0.0 {
            let shift = Complex::new(0.0, self.frequency.im);
            for k in 0..n {
                out[k] -= shift * x[n + k];
                out[n + k] -= shift * x[k];
            }
        }
        Ok(out)
    }
}

/// `alpha_ab(z) = d mu_a(z)/d E_b(z) = -Tr(delta P_b r_a)`
/// using `H' = +E.r` and electronic dipole `mu = -Tr(P r)`.
/// Restricted references count both spin channels; unrestricted responses
/// retain separate coupled alpha and beta amplitudes.
pub fn dynamic_polarizability<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<DynamicPolarizability>> {
    let (amplitudes, iterations) =
        solve_dynamic_amplitudes(mol, mb, functional, grid, scf, opts, frequency, config)?;
    let alpha = amplitudes.alpha;
    let mut reciprocity_defect: f64 = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            reciprocity_defect = reciprocity_defect.max((alpha[a][b] - alpha[b][a]).abs());
        }
    }
    let isotropic = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
    Ok(ResponseReport {
        shape: vec![3, 3],
        functional: functional.description.clone(),
        reference: super::diagnostics::reference_diagnostics(scf)?,
        value: DynamicPolarizability {
            frequency,
            alpha,
            isotropic,
            reciprocity_defect,
        },
        units: "bohr^3",
        model: "adiabatic KS/HF paired Casida response",
        axis_convention: "alpha[observable Cartesian axis][electric-field Cartesian axis]",
        converged: true,
        residuals: amplitudes.residuals.to_vec(),
        iterations: iterations.to_vec(),
        warnings: Vec::new(),
    })
}

/// Amplitudes of the response to a uniform field along each axis at one
/// frequency, for reuse.
#[derive(Clone, Debug)]
pub struct DynamicAmplitudes {
    pub frequency: Frequency,
    /// `X + Y` per field axis in the order of the response solver: spin
    /// blocks (one for a restricted reference), `a * n_occ + i` inside.
    pub sum: [Vec<Complex>; 3],
    /// `X - Y` in the same order.
    pub difference: [Vec<Complex>; 3],
    /// `alpha[observable axis][field axis]`.
    pub alpha: [[Complex; 3]; 3],
    pub residuals: [f64; 3],
}

/// The paired response equations `(A + B)(X + Y) - z (X - Y) = -2 r`,
/// `(A - B)(X - Y) - z (X + Y) = 0` of [`dynamic_polarizability`], with the
/// amplitudes of every field axis kept.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_amplitudes<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<DynamicAmplitudes> {
    solve_dynamic_amplitudes(mol, mb, functional, grid, scf, opts, frequency, config)
        .map(|(amplitudes, _)| amplitudes)
}

/// One solve supplies both public views, with identical raw residuals and
/// iteration counts. Public report and amplitude structures stay compatible.
#[allow(clippy::too_many_arguments)]
fn solve_dynamic_amplitudes<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<(DynamicAmplitudes, [usize; 3])> {
    let sources = dipole_matrices(mb, Vec3::zero());
    let result = solve_operator_amplitudes(
        mol, mb, functional, grid, scf, opts, &sources, frequency, config,
    )?;
    let tensor = result.value;
    Ok((
        DynamicAmplitudes {
            frequency,
            sum: result.sum.try_into().expect("three electric sources"),
            difference: result
                .difference
                .try_into()
                .expect("three electric sources"),
            alpha: tensor
                .tensor
                .into_iter()
                .map(|row| row.try_into().expect("three electric observables"))
                .collect::<Vec<_>>()
                .try_into()
                .expect("three electric observables"),
            residuals: tensor.residuals.try_into().expect("three electric sources"),
        },
        tensor
            .iterations
            .try_into()
            .expect("three electric sources"),
    ))
}

pub(crate) struct OperatorPolarizability {
    pub tensor: Vec<Vec<Complex>>,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
}

struct OperatorAmplitudes {
    value: OperatorPolarizability,
    sum: Vec<Vec<Complex>>,
    difference: Vec<Vec<Complex>>,
}

/// Negative expectation response to real, spin-independent AO sources
/// in the same paired Casida space used for the electric dipole.
#[allow(clippy::too_many_arguments)]
pub(crate) fn real_operator_polarizability<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    sources: &[crate::linalg::Matrix],
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<OperatorPolarizability> {
    solve_operator_amplitudes(
        mol, mb, functional, grid, scf, opts, sources, frequency, config,
    )
    .map(|result| result.value)
}

#[allow(clippy::too_many_arguments)]
fn solve_operator_amplitudes<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    sources: &[crate::linalg::Matrix],
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<OperatorAmplitudes> {
    let z = frequency.energy()?;
    config.validate()?;
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "dynamic response requires a converged SCF reference".into(),
        ));
    }
    let spin = if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    };
    let solver = TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?;
    let n = solver.dim();
    let operator = CasidaOperator {
        solver,
        frequency: z,
    };
    let count = sources.len();
    let mut d = vec![vec![0.0; n]; count];
    for (projected, source) in d.iter_mut().zip(sources) {
        let mut off = 0;
        for s in 0..if scf.unrestricted { 2 } else { 1 } {
            let no = if s == 0 { scf.n_alpha } else { scf.n_beta };
            let c = &scf.mo_coeff[s];
            let mo_source = c.transpose_matmul(source).matmul(c);
            for a in 0..c.cols - no {
                for i in 0..no {
                    projected[off + a * no + i] = mo_source[(no + a, i)];
                }
            }
            off += no * (c.cols - no);
        }
    }
    let factor = if scf.unrestricted { 2.0 } else { 4.0 };
    let mut alpha = vec![vec![Complex::ZERO; count]; count];
    let mut residuals = vec![0.0; count];
    let mut iterations = vec![0; count];
    let mut sum = vec![Vec::new(); count];
    let mut difference = vec![Vec::new(); count];
    for b in 0..count {
        let mut rhs = vec![Complex::ZERO; 2 * n];
        for k in 0..n {
            rhs[k] = (-d[b][k]).into();
        }
        let solution = krylov::solve(&operator, &rhs, None, config)?;
        for a in 0..count {
            for k in 0..n {
                alpha[a][b] -= solution.amplitudes[k] * (factor * d[a][k]);
            }
        }
        // The solver holds (X + Y) / 2 and (X - Y) / 2.
        sum[b] = solution.amplitudes[..n].iter().map(|v| *v * 2.0).collect();
        difference[b] = solution.amplitudes[n..].iter().map(|v| *v * 2.0).collect();
        residuals[b] = solution.relative_residual;
        iterations[b] = solution.iterations;
    }
    Ok(OperatorAmplitudes {
        value: OperatorPolarizability {
            tensor: alpha,
            residuals,
            iterations,
        },
        sum,
        difference,
    })
}

/// Optical rotation tensor `G'(omega)` in the length gauge about `origin`.
#[derive(Clone, Debug)]
pub struct OpticalRotation {
    pub frequency: Frequency,
    /// `G'[a][b]`: electric-field axis `a`, magnetic-dipole axis `b`.
    pub tensor: [[Complex; 3]; 3],
    /// `beta = -Tr G' / (3 z)` with the complex response energy `z`.
    pub beta: Complex,
    pub origin: Vec3,
}

/// Electric dipole - magnetic dipole polarizability of the paired Casida
/// response. With `H' = +E.r` the amplitudes `(X - Y) / 2` of the response
/// carry the expectation value of an imaginary antisymmetric operator; for
/// the magnetic dipole `m = (i/2) (r - O) x grad` the induced moment is
/// `m_b = -i (factor / 2) sum_k l_b,k Q_k`, which defines
/// `G'[a][b] = -(factor / 2) sum_k l_b,k Q_k^(a)` (`factor` is 4 for a
/// restricted and 2 for an unrestricted reference). The tensor depends on
/// the origin in a finite basis.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_optical_rotation<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    frequency: Frequency,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<OpticalRotation>> {
    let z = frequency.energy()?;
    config.validate()?;
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "dynamic response requires a converged SCF reference".into(),
        ));
    }
    if z.re == 0.0 && z.im == 0.0 {
        return Err(DftError::InvalidInput(
            "the optical rotation needs a nonzero response frequency".into(),
        ));
    }
    let spin = if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    };
    let solver = TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?;
    let n = solver.dim();
    let operator = CasidaOperator {
        solver,
        frequency: z,
    };
    let dipoles = dipole_matrices(mb, origin);
    let angular = crate::properties::nmr::angular_momentum_operator(mb, origin);
    let mut d = vec![vec![0.0; n]; 3];
    let mut l = vec![vec![0.0; n]; 3];
    for axis in 0..3 {
        let mut off = 0;
        for s in 0..if scf.unrestricted { 2 } else { 1 } {
            let no = if s == 0 { scf.n_alpha } else { scf.n_beta };
            let c = &scf.mo_coeff[s];
            let mo_dipole = c.transpose_matmul(&dipoles[axis]).matmul(c);
            let mo_angular = c.transpose_matmul(&angular[axis]).matmul(c);
            for a in 0..c.cols - no {
                for i in 0..no {
                    d[axis][off + a * no + i] = mo_dipole[(no + a, i)];
                    l[axis][off + a * no + i] = mo_angular[(no + a, i)];
                }
            }
            off += no * (c.cols - no);
        }
    }
    let mut tensor = [[Complex::ZERO; 3]; 3];
    let mut residuals = Vec::with_capacity(3);
    let mut iterations = Vec::with_capacity(3);
    let factor = if scf.unrestricted { 2.0 } else { 4.0 };
    for a in 0..3 {
        let mut rhs = vec![Complex::ZERO; 2 * n];
        for k in 0..n {
            rhs[k] = (-d[a][k]).into();
        }
        let solution = krylov::solve(&operator, &rhs, None, config)?;
        for b in 0..3 {
            for k in 0..n {
                tensor[a][b] -= solution.amplitudes[n + k] * (0.5 * factor * l[b][k]);
            }
        }
        residuals.push(solution.relative_residual);
        iterations.push(solution.iterations);
    }
    let trace = tensor[0][0] + tensor[1][1] + tensor[2][2];
    let beta = Complex::ZERO - trace / (z * 3.0);
    Ok(ResponseReport {
        shape: vec![3, 3],
        functional: functional.description.clone(),
        reference: super::diagnostics::reference_diagnostics(scf)?,
        value: OpticalRotation {
            frequency,
            tensor,
            beta,
            origin,
        },
        units: "atomic units",
        model: "adiabatic KS/HF paired Casida response, length gauge",
        axis_convention: "G'[electric-field Cartesian axis][magnetic-dipole Cartesian axis]",
        converged: true,
        residuals,
        iterations,
        warnings: Vec::new(),
    })
}

/// Optical rotation tensor in the modified velocity gauge. With the
/// momentum `p = -i grad` in place of the electric dipole,
/// `G'[a][b] = -(1 / z) [R(z) - R(0)]`, where `R[a][b]` is the magnetic
/// dipole `b` induced by a perturbation along `p_a`; subtracting the static
/// limit removes the unphysical static rotation of a finite basis. The trace
/// and therefore `beta` do not depend on the origin of the magnetic dipole.
/// For an imaginary antisymmetric perturbation the right-hand side enters the
/// second block of the paired equations and the amplitudes `(X - Y) / 2i`
/// carry the expectation value of an imaginary antisymmetric operator.
#[allow(clippy::too_many_arguments)]
pub fn dynamic_optical_rotation_velocity<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    opts: &'a ScfOptions,
    frequency: Frequency,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<OpticalRotation>> {
    let z = frequency.energy()?;
    config.validate()?;
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "dynamic response requires a converged SCF reference".into(),
        ));
    }
    if z.re == 0.0 && z.im == 0.0 {
        return Err(DftError::InvalidInput(
            "the optical rotation needs a nonzero response frequency".into(),
        ));
    }
    let spin = if scf.unrestricted {
        TdaSpin::SpinResolved
    } else {
        TdaSpin::Singlet
    };
    let dynamic = CasidaOperator {
        solver: TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?,
        frequency: z,
    };
    let fixed = CasidaOperator {
        solver: TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?,
        frequency: Complex::ZERO,
    };
    let n = dynamic.solver.dim();
    let gradient = crate::properties::nmr::gradient_operator(mb);
    let angular = crate::properties::nmr::angular_momentum_operator(mb, origin);
    let mut g = vec![vec![0.0; n]; 3];
    let mut l = vec![vec![0.0; n]; 3];
    for axis in 0..3 {
        let mut off = 0;
        for s in 0..if scf.unrestricted { 2 } else { 1 } {
            let no = if s == 0 { scf.n_alpha } else { scf.n_beta };
            let c = &scf.mo_coeff[s];
            let mo_gradient = c.transpose_matmul(&gradient[axis]).matmul(c);
            let mo_angular = c.transpose_matmul(&angular[axis]).matmul(c);
            for a in 0..c.cols - no {
                for i in 0..no {
                    g[axis][off + a * no + i] = mo_gradient[(no + a, i)];
                    l[axis][off + a * no + i] = mo_angular[(no + a, i)];
                }
            }
            off += no * (c.cols - no);
        }
    }
    let mut tensor = [[Complex::ZERO; 3]; 3];
    let mut residuals = Vec::with_capacity(6);
    let mut iterations = Vec::with_capacity(6);
    let factor = if scf.unrestricted { 2.0 } else { 4.0 };
    for a in 0..3 {
        // p_a = i (-grad_a): the source of the second block is +grad_a.
        let mut rhs = vec![Complex::ZERO; 2 * n];
        for k in 0..n {
            rhs[n + k] = g[a][k].into();
        }
        let at_frequency = krylov::solve(&dynamic, &rhs, None, config)?;
        let at_zero = krylov::solve(&fixed, &rhs, None, config)?;
        for b in 0..3 {
            let mut induced = Complex::ZERO;
            for k in 0..n {
                induced += (at_frequency.amplitudes[n + k] - at_zero.amplitudes[n + k])
                    * (0.5 * factor * l[b][k]);
            }
            tensor[a][b] = Complex::ZERO - induced / z;
        }
        residuals.push(at_frequency.relative_residual);
        residuals.push(at_zero.relative_residual);
        iterations.push(at_frequency.iterations);
        iterations.push(at_zero.iterations);
    }
    let trace = tensor[0][0] + tensor[1][1] + tensor[2][2];
    let beta = Complex::ZERO - trace / (z * 3.0);
    Ok(ResponseReport {
        shape: vec![3, 3],
        functional: functional.description.clone(),
        reference: super::diagnostics::reference_diagnostics(scf)?,
        value: OpticalRotation {
            frequency,
            tensor,
            beta,
            origin,
        },
        units: "atomic units",
        model: "adiabatic KS/HF paired Casida response, modified velocity gauge",
        axis_convention: "G'[momentum Cartesian axis][magnetic-dipole Cartesian axis]",
        converged: true,
        residuals,
        iterations,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::scf::run_scf;
    use crate::system::Atom;
    use crate::xc::{resolve, FunctionalSpec};

    #[test]
    fn zero_block_action_matches_unscreened_casida_columns() {
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.13, -0.21, 0.08),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.47, 0.63, -0.19),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.54, 1.36, 0.37),
            },
        ]);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
        let opts = ScfOptions {
            e_tol: 1e-12,
            p_tol: 1e-10,
            ..Default::default()
        };
        let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
        let mut operator = CasidaOperator {
            solver: TdaSolver::new(&mol, &mb, &functional, None, &scf, &opts, TdaSpin::Singlet)
                .unwrap(),
            frequency: Complex::ZERO,
        };
        let n = operator.solver.dim();
        // All columns of both paired blocks, including their imaginary images,
        // are compared to the original unscreened AO contractions.
        for frequency in [Complex::ZERO, Complex::new(0.04, 0.004)] {
            operator.frequency = frequency;
            for column in 0..2 * n {
                let mut real = vec![0.0; 2 * n];
                real[column] = 1.0;
                let apb = operator.solver.apb_vec(&real[..n]).unwrap();
                let amb = operator.solver.amb_vec(&real[n..]).unwrap();
                for scale in [
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 1.0),
                    Complex::new(1e-100, -2e-100),
                ] {
                    let x: Vec<_> = real.iter().map(|v| scale * *v).collect();
                    let actual = operator.apply(&x).unwrap();
                    for row in 0..n {
                        let top = scale * apb[row] - frequency * x[n + row];
                        let bottom = scale * amb[row] - frequency * x[row];
                        let tolerance = 1e-11 * scale.abs();
                        assert!((actual[row] - top).abs() < tolerance);
                        assert!((actual[n + row] - bottom).abs() < tolerance);
                    }
                }
            }
            assert!(operator
                .apply(&vec![Complex::ZERO; 2 * n])
                .unwrap()
                .iter()
                .all(|v| v.abs() == 0.0));
        }
    }
}
