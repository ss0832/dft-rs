// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Collisionless electric response of a real collinear Fermi reference.
//!
//! Populations are held at their reference values. This is not the static
//! isothermal Mermin response: diagonal population relaxation and chemical
//! potential derivatives are absent, including at zero response frequency.
//! The supplied resolved functional describes only its semilocal/exchange
//! parent. Additional correlated-method metadata is not encoded in that type;
//! callers requesting a composite method must use the common driver, which
//! checks that context. No causal PT2 contribution is implied by this kernel.
//! For every ordered pair with unequal populations, solve
//! `(z - epsilon_p + epsilon_q) X_pq - (f_q-f_p) K[X]_pq
//! = (f_q-f_p) V_pq`, with `exp(-i z t)` and `V = +E.r`.

use super::complex::Complex;
use super::diagnostics::{density_diagnostics, ReferenceDiagnostics};
use super::dynamic::{DynamicPolarizability, Frequency, ResponseReport};
use super::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use super::ResponseSolver;
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};

/// Complex density response; neither part is assumed symmetric.
#[derive(Clone, Debug)]
pub struct ThermalDensityResponse {
    pub real: [Matrix; 2],
    pub imaginary: [Matrix; 2],
    pub relative_residual: f64,
    pub iterations: usize,
    /// Absolute complex trace per spin, measured in the AO overlap metric.
    /// Units are electron count per unit of the supplied perturbation;
    /// an electric-field response has units electrons * e Bohr / Hartree.
    pub number_defects: [f64; 2],
}

/// All nine contractions and the actual canonical reference constraints.
#[derive(Clone, Debug)]
pub struct ThermalPolarizability {
    pub response: ResponseReport<DynamicPolarizability>,
    /// Electronic k_B T in Hartree, not a lattice or rotational temperature.
    pub temperature: f64,
    /// Canonical spin Fermi levels in Hartree.
    pub chemical_potentials: [f64; 2],
    pub population_defect: f64,
    pub number_defects: [[f64; 2]; 3],
}

#[derive(Clone, Copy)]
struct Pair {
    spin: usize,
    p: usize,
    q: usize,
    difference: f64,
    gap: f64,
}

/// Reuses the physical Hartree/exchange/XC kernel, without an OV partition.
pub struct ThermalSolver<'a> {
    kernel: ResponseSolver<'a>,
    reference: &'a ScfResult,
    pairs: Vec<Pair>,
    pub diagnostics: ReferenceDiagnostics,
    pub population_defect: f64,
    pub chemical_potentials: [f64; 2],
}

impl<'a> ThermalSolver<'a> {
    pub fn new(
        mol: &Molecule,
        basis: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        options: &ScfOptions,
    ) -> Result<Self> {
        if functional.family == Family::MetaGga
            || functional
                .parts
                .iter()
                .any(|(_, part)| part.family() == Family::MetaGga)
            || functional.required_nonlocal_term().is_some()
            || options.skala.is_some()
            || options.vv10.is_some()
            || options.pcm.is_some()
        {
            return Err(DftError::Unsupported {
                feature: "collisionless thermal electric response of this kernel".into(),
                missing: "a validated symmetric/antisymmetric action for kinetic, learned, nonlocal or continuum kernels; the present route supports local/GGA and exact exchange".into(),
            });
        }
        if !scf.converged
            || !scf.electronic_temperature.is_finite()
            || scf.electronic_temperature <= 0.0
            || scf.fixed_occupation_residual.is_some()
            || options.electron_offset != [0.0; 2]
            || options.electronic_temperature != scf.electronic_temperature
            || !options.conv_tol_grad.is_finite()
            || options.conv_tol_grad <= 0.0
            || !options.p_tol.is_finite()
            || options.p_tol <= 0.0
        {
            return Err(DftError::InvalidInput(
                "collisionless thermal response requires a converged positive-temperature canonical Fermi reference without prescribed populations or electron offsets".into(),
            ));
        }
        let diagnostics = density_diagnostics(scf)?;
        // Reuse the caller's actual thresholds without floors or multipliers.
        // The orbital residual is in Hartree; the relative eigen residual is
        // dimensionless. Density, metric and population defects use p_tol.
        if diagnostics.eigen_residual > options.conv_tol_grad
            || diagnostics.orbital_residual > options.conv_tol_grad
            || diagnostics.orthonormality_defect > options.p_tol
            || diagnostics.particle_number_error > options.p_tol
        {
            return Err(DftError::InvalidInput(
                "collisionless thermal reference fails its measured stationarity, metric or canonical number constraint".into(),
            ));
        }
        let mut chemical_potentials = [0.0; 2];
        let mut population_defect = 0.0_f64;
        let mut pairs = Vec::new();
        for spin in 0..2 {
            let mu = scf.chemical_potentials[spin]
                .filter(|x| x.is_finite())
                .ok_or_else(|| DftError::InvalidInput("missing finite Fermi level".into()))?;
            chemical_potentials[spin] = mu;
            let c = &scf.mo_coeff[spin];
            let f = &scf.occupations[spin];
            let e = &scf.mo_energy[spin];
            if f.len() != c.cols || e.len() != c.cols {
                return Err(DftError::InvalidInput(
                    "inconsistent thermal orbital arrays".into(),
                ));
            }
            let sc = scf.overlap.matmul(c);
            let density_mo = sc.transpose_matmul(&scf.density[spin].matmul(&sc));
            for p in 0..c.cols {
                if !f[p].is_finite() || !(0.0..=1.0).contains(&f[p]) {
                    return Err(DftError::InvalidInput("invalid Fermi population".into()));
                }
                let x = (e[p] - mu) / scf.electronic_temperature;
                let expected = if x >= 0.0 {
                    let y = (-x).exp();
                    y / (1.0 + y)
                } else {
                    1.0 / (1.0 + x.exp())
                };
                population_defect = population_defect.max((f[p] - expected).abs());
                for q in 0..c.cols {
                    population_defect = population_defect
                        .max((density_mo[(p, q)] - if p == q { f[p] } else { 0.0 }).abs());
                    // Exact equal populations have zero commutator weight,
                    // including degenerate orbitals. No small weight is cut off.
                    if p != q && f[p] != f[q] {
                        pairs.push(Pair {
                            spin,
                            p,
                            q,
                            difference: f[q] - f[p],
                            gap: e[p] - e[q],
                        });
                    }
                }
            }
        }
        if !population_defect.is_finite() || population_defect > options.p_tol {
            return Err(DftError::InvalidInput(
                "reference density, spectrum and canonical Fermi populations disagree".into(),
            ));
        }
        Ok(Self {
            kernel: ResponseSolver::kernel_only(mol, basis, functional, grid, scf, options)?,
            reference: scf,
            pairs,
            diagnostics,
            population_defect,
            chemical_potentials,
        })
    }

    fn to_ao(&self, amplitudes: &[f64]) -> [Matrix; 2] {
        let mut mo = self
            .reference
            .mo_coeff
            .each_ref()
            .map(|c| Matrix::zeros(c.cols, c.cols));
        for (pair, &value) in self.pairs.iter().zip(amplitudes) {
            mo[pair.spin][(pair.p, pair.q)] = value;
        }
        std::array::from_fn(|s| {
            let c = &self.reference.mo_coeff[s];
            c.matmul(&mo[s]).matmul(&c.transpose())
        })
    }

    fn to_pairs(&self, matrices: &[Matrix; 2]) -> Vec<f64> {
        let mo: [Matrix; 2] = std::array::from_fn(|s| {
            let c = &self.reference.mo_coeff[s];
            c.transpose_matmul(&matrices[s]).matmul(c)
        });
        self.pairs
            .iter()
            .map(|pair| mo[pair.spin][(pair.p, pair.q)])
            .collect()
    }

    fn kernel_action(&self, x: &[f64]) -> Result<Vec<f64>> {
        let density = self.to_ao(x);
        let mut symmetric = density.clone();
        let mut antisymmetric = density.clone();
        for s in 0..2 {
            for i in 0..density[s].rows {
                for j in 0..density[s].cols {
                    symmetric[s][(i, j)] = 0.5 * (density[s][(i, j)] + density[s][(j, i)]);
                    antisymmetric[s][(i, j)] = 0.5 * (density[s][(i, j)] - density[s][(j, i)]);
                }
            }
        }
        // GGA density-gradient contractions assume a symmetric density.
        // Passing a general transition density directly would add a false
        // antisymmetric gradient. Exchange retains both ordered endpoints.
        let mut fock = self.kernel.fock_response(&symmetric)?;
        let odd = self.kernel.imaginary_fock_response(&antisymmetric)?;
        for s in 0..2 {
            for (a, b) in fock[s].as_mut_slice().iter_mut().zip(odd[s].as_slice()) {
                *a += b;
            }
        }
        Ok(self.to_pairs(&fock))
    }

    fn liouvillian_action(&self, x: &[Complex], z: Complex) -> Result<Vec<Complex>> {
        let kernel = krylov::apply_real_linear(x, |v| self.kernel_action(v))?;
        Ok(self
            .pairs
            .iter()
            .zip(x)
            .zip(kernel)
            .map(|((p, x), k)| (z - Complex::from(p.gap)) * *x - k * p.difference)
            .collect())
    }

    /// Linear response to a real AO one-electron perturbation per spin.
    pub fn solve(
        &self,
        potential: &[Matrix; 2],
        frequency: Frequency,
        config: &ResponseSolverConfig,
    ) -> Result<ThermalDensityResponse> {
        let z = frequency.energy()?;
        config.validate()?;
        let n = self.reference.overlap.rows;
        if potential
            .iter()
            .any(|v| v.rows != n || v.cols != n || v.as_slice().iter().any(|x| !x.is_finite()))
        {
            return Err(DftError::InvalidInput(
                "invalid thermal AO perturbation".into(),
            ));
        }
        struct Operator<'s, 'a> {
            solver: &'s ThermalSolver<'a>,
            z: Complex,
            diagonal: Vec<Complex>,
        }
        impl LinearOperator for Operator<'_, '_> {
            fn dimension(&self) -> usize {
                self.solver.pairs.len()
            }
            fn structure(&self) -> OperatorStructure {
                OperatorStructure::General
            }
            fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
                let physical: Vec<_> = x.iter().zip(&self.diagonal).map(|(x, d)| *x / *d).collect();
                self.solver.liouvillian_action(&physical, self.z)
            }
        }
        let rhs: Vec<Complex> = self
            .to_pairs(potential)
            .iter()
            .zip(&self.pairs)
            .map(|(v, p)| Complex::from(v * p.difference))
            .collect();
        let (amplitudes, residual, iterations) = if self.pairs.is_empty() {
            (Vec::new(), 0.0, 0)
        } else {
            // Exact right preconditioning A D^-1 y = b, x = D^-1 y.
            // Deep-core gaps otherwise stall restarted GMRES even for a
            // well-resolved damped response. This changes neither A nor b.
            // An exactly zero free denominator uses the identity ONLY in
            // that preconditioner entry; the physical operator retains zero.
            let diagonal: Vec<_> = self
                .pairs
                .iter()
                .map(|p| {
                    let d = z - Complex::from(p.gap);
                    if d == Complex::ZERO {
                        Complex::ONE
                    } else {
                        d
                    }
                })
                .collect();
            let operator = Operator {
                solver: self,
                z,
                diagonal,
            };
            let solved = krylov::solve(&operator, &rhs, None, config)?;
            let amplitudes: Vec<_> = solved
                .amplitudes
                .iter()
                .zip(&operator.diagonal)
                .map(|(y, d)| *y / *d)
                .collect();
            // Recheck the original physical equation on the returned density,
            // independently of the preconditioned solver's stopping report.
            let image = self.liouvillian_action(&amplitudes, z)?;
            let norm_rhs = rhs.iter().fold(0.0_f64, |n, b| n.hypot(b.abs()));
            let norm_error = image
                .iter()
                .zip(&rhs)
                .fold(0.0_f64, |n, (a, b)| n.hypot((*a - *b).abs()));
            let residual = if norm_rhs == 0.0 {
                norm_error
            } else {
                norm_error / norm_rhs
            };
            if !residual.is_finite() || residual > config.acceptance() {
                return Err(DftError::ResponseNotConverged {
                    iterations: solved.iterations,
                    residual,
                });
            }
            (amplitudes, residual, solved.iterations)
        };
        let real = self.to_ao(&amplitudes.iter().map(|v| v.re).collect::<Vec<_>>());
        let imaginary = self.to_ao(&amplitudes.iter().map(|v| v.im).collect::<Vec<_>>());
        let number_defects = std::array::from_fn(|s| {
            real[s]
                .frobenius_dot(&self.reference.overlap)
                .hypot(imaginary[s].frobenius_dot(&self.reference.overlap))
        });
        if number_defects
            .iter()
            .any(|d| !d.is_finite() || *d > config.acceptance())
        {
            return Err(DftError::ResponseNotConverged {
                iterations,
                residual: number_defects[0].max(number_defects[1]),
            });
        }
        Ok(ThermalDensityResponse {
            real,
            imaginary,
            relative_residual: residual,
            iterations,
            number_defects,
        })
    }
}

/// Canonical collisionless dipole susceptibility, with no tensor averaging.
#[allow(clippy::too_many_arguments)]
pub fn polarizability<'a>(
    mol: &Molecule,
    basis: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    scf: &'a ScfResult,
    options: &ScfOptions,
    frequency: Frequency,
    config: &ResponseSolverConfig,
) -> Result<ThermalPolarizability> {
    let solver = ThermalSolver::new(mol, basis, functional, grid, scf, options)?;
    let dipoles = dipole_matrices(basis, Vec3::zero());
    let mut alpha = [[Complex::ZERO; 3]; 3];
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut number_defects = [[0.0; 2]; 3];
    for b in 0..3 {
        let value = solver.solve(&[dipoles[b].clone(), dipoles[b].clone()], frequency, config)?;
        number_defects[b] = value.number_defects;
        residuals.push(value.relative_residual);
        iterations.push(value.iterations);
        for a in 0..3 {
            for s in 0..2 {
                alpha[a][b] -= Complex::new(
                    dipoles[a].frobenius_dot(&value.real[s]),
                    dipoles[a].frobenius_dot(&value.imaginary[s]),
                );
            }
        }
    }
    let isotropic = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
    let reciprocity_defect = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| (alpha[a][b] - alpha[b][a]).abs())
        .fold(0.0_f64, f64::max);
    Ok(ThermalPolarizability {
        temperature: scf.electronic_temperature,
        chemical_potentials: solver.chemical_potentials,
        population_defect: solver.population_defect,
        number_defects,
        response: ResponseReport {
            value: DynamicPolarizability { frequency, alpha, isotropic, reciprocity_defect },
            shape: vec![3,3], functional: functional.description.clone(), reference: solver.diagnostics,
            units: "bohr^3", model: "canonical collisionless Fermi-population Liouvillian; adiabatic collinear local/GGA and exact-exchange kernel",
            axis_convention: "alpha[observable Cartesian axis][electric-field Cartesian axis]; H'=+E.r; exp(-i z t)",
            converged: true, residuals, iterations, warnings: vec![
                "Collisionless response holds the reference Fermi populations fixed; the zero-frequency limit is not static isothermal response. Thermal projector_defect is physical non-idempotency, not a convergence defect.".into(),
                "The supplied resolved semilocal/exchange parent contains no opaque external correlation metadata; no causal PT2 contribution is included. Composite-method callers must use the common driver.".into(),
            ],
        },
    })
}
