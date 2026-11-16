// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Static linear response of a molecular Fermi-Dirac or fixed-population reference.
//!
//! In the canonical ensemble with a fixed electron count per spin channel,
//! the density matrix in the reference MO basis responds to a change `dF` of
//! the Fock matrix and `dN` of the electron counts as
//!
//! ```text
//! dP_ij = W_ij (dF_ij - dmu delta_ij),
//! W_ij  = (f_i - f_j) / (e_i - e_j)      (i, j non-degenerate)
//!       = df/de = -f_i (1 - f_i) / kT    (i = j or degenerate),
//! sum_i dP_ii = dN,
//! ```
//!
//! For prescribed nonthermal populations, `W_ii = 0` and the divided
//! difference is used for every pair of different populations, including
//! pairs inside the nominal occupied or virtual spaces. Equal populations
//! have zero weight even at degeneracy. Unequal populations at an exact
//! degeneracy do not define a differentiable energy-ordered ensemble and
//! are refused. No denominator is replaced or clipped. The fixed-population
//! response conserves each spin count and has no chemical-potential response.
//!
//! The self-consistent field is `dF = V + K[dP]` with the full Hartree, exchange, XC, VV10 and
//! continuum kernel of the reference. Eliminating `dmu` with the number
//! constraint leaves a linear equation for `dF` that is solved matrix-free
//! with GMRES on the original operator. The occupation response (diagonal
//! `W_ii`) and the orbital rotations among partially occupied orbitals are
//! both included, which is what an aufbau CPKS solve omits.
//!
//! `dmu_sigma / dN_sigma'` gives the hardness matrix of the ensemble and
//! `dP / dN` the Fukui matrices used by conceptual DFT.

use super::complex::Complex;
use super::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use super::ResponseSolver;
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod fermi_path;
mod path;
pub(crate) use fermi_path::canonical_fermi_path;
pub use fermi_path::FermiPathDiagnostics;
pub(crate) use path::fixed_population_path;

/// Measured constraints of a prescribed-population orbital Taylor path.
#[derive(Clone, Debug)]
pub struct FixedPopulationDiagnostics {
    /// Reference MO density minus diag(f), in the physical overlap metric.
    pub reference_population_defect: f64,
    /// Reference AO density minus C diag(f) C^T.
    pub reference_density_defect: f64,
    /// Coefficients of C(t)^T S(t) P(t) S(t) C(t) minus diag(f), orders 1..n.
    pub population_defects: Vec<f64>,
    /// Coefficients of C(t)^T S(t) C(t) minus I, orders 1..n.
    pub orthogonality_defects: Vec<f64>,
    /// Full commutator residual divided by its actual sum-of-products bound.
    pub relative_stationarity_defects: Vec<f64>,
}

impl FixedPopulationDiagnostics {
    pub(crate) fn maximum_defect(&self) -> f64 {
        std::iter::once(&self.reference_population_defect)
            .chain(std::iter::once(&self.reference_density_defect))
            .chain(&self.population_defects)
            .chain(&self.orthogonality_defects)
            .chain(&self.relative_stationarity_defects)
            .fold(0.0_f64, |m, v| {
                if v.is_finite() {
                    m.max(v.abs())
                } else {
                    f64::INFINITY
                }
            })
    }
}

/// Solution of one ensemble response.
#[derive(Clone, Debug)]
pub struct EnsembleResponse {
    /// AO density change per spin channel.
    pub delta_density: [Matrix; 2],
    /// Thermal chemical-potential derivative in the supplied perturbation's
    /// units: Hartree for a dimensionless potential direction, e Bohr for
    /// an electric-field derivative. Zero is the unused fixed-population gauge.
    pub delta_chemical_potentials: [f64; 2],
    /// Fresh relative residual of the self-consistency equation.
    pub relative_residual: f64,
    pub iterations: usize,
    /// `|Tr(S dP) - dN|` per spin, measured on the returned AO solution.
    pub number_defects: [f64; 2],
}

/// Canonical response in a moving nonorthogonal AO basis.
#[derive(Clone, Debug)]
pub struct MetricEnsembleResponse {
    pub response: EnsembleResponse,
    /// Sum over spins of d(C f epsilon C^T), including metric and occupation terms.
    pub delta_energy_weighted_density: Matrix,
}

/// Static electric dipole response, without tensor symmetrization.
#[derive(Clone, Debug)]
pub struct EnsemblePolarizability {
    /// `alpha_ab = d mu_a / d F_b`, in e^2 Bohr^2/Hartree.
    pub tensor: [[f64; 3]; 3],
    /// Independent electric-field solutions in Cartesian x, y, z order.
    pub responses: [EnsembleResponse; 3],
}

#[derive(Clone, Copy)]
enum PopulationLaw {
    Fixed,
    FermiDirac(f64),
}

impl PopulationLaw {
    fn weight(self, ei: f64, ej: f64, fi: f64, fj: f64) -> Result<f64> {
        let de = ei - ej;
        match self {
            Self::Fixed => {
                if fi == fj {
                    Ok(0.0)
                } else if de == 0.0 {
                    Err(DftError::InvalidInput(
                        "fixed-population response is undefined for exactly degenerate orbitals with unequal populations; specify equal populations in the degenerate subspace".into(),
                    ))
                } else {
                    Ok((fi - fj) / de)
                }
            }
            Self::FermiDirac(kt) => {
                // Below this scale the thermal divided difference loses digits.
                // The analytic Fermi derivative is its continuous limit.
                Ok(if de.abs() > 1e-8 * kt {
                    (fi - fj) / de
                } else {
                    let fm = 0.5 * (fi + fj);
                    -fm * (1.0 - fm) / kt
                })
            }
        }
    }
}

/// Longitudinal response to a uniform spin Zeeman field at fixed total
/// electron number. The bare result retains the actual reference spectrum
/// and occupations while omitting its induced Hartree/XC/exchange field.
#[derive(Clone, Debug)]
pub struct SpinSusceptibility {
    pub susceptibility: f64,
    pub bare_susceptibility: f64,
    pub enhancement: f64,
    pub stiffness: f64,
    pub bare_stiffness: f64,
    /// dP_sigma/dB in the AO basis; M=-g(N_alpha-N_beta)/4.
    pub delta_density: [Matrix; 2],
    pub response_residual: f64,
    pub iterations: usize,
    pub number_defects: [f64; 2],
}

struct Channel {
    c: Matrix,
    w: Matrix,
    w_diag_sum: f64,
}

/// Matrix-free response for a converged thermal or prescribed-population reference.
pub struct EnsembleSolver<'a> {
    pub(super) kernel: ResponseSolver<'a>,
    channels: Vec<Channel>,
    restricted: bool,
    nmo: [usize; 2],
    population_law: PopulationLaw,
    minimum_fixed_gap: Option<f64>,
}

impl<'a> EnsembleSolver<'a> {
    /// The complete orbital/occupation response, including spin transfer
    /// between channels. Constraining each channel's number separately
    /// would remove the Pauli response to a uniform spin-only field.
    pub fn spin_susceptibility(
        &self,
        g: f64,
        config: &ResponseSolverConfig,
    ) -> Result<SpinSusceptibility> {
        if !g.is_finite() || g < 0.0 {
            return Err(DftError::InvalidInput(
                "spin_g_factor must be finite and non-negative".into(),
            ));
        }
        let response = self.solve(None, [1.0, -1.0], config)?;
        let stiffness =
            response.delta_chemical_potentials[0] - response.delta_chemical_potentials[1];
        if !stiffness.is_finite() || stiffness <= 0.0 {
            return Err(DftError::InvalidInput(format!(
                "the spin stiffness is {stiffness:e} Hartree: the reference is unstable against spin transfer")));
        }
        let bare_stiffness = -1.0 / self.channels[0].w_diag_sum - 1.0 / self.channels[1].w_diag_sum;
        let factor = -g / (2.0 * stiffness);
        let delta_density = response.delta_density.map(|mut p| {
            for value in p.as_mut_slice() {
                *value *= factor;
            }
            p
        });
        Ok(SpinSusceptibility {
            susceptibility: 0.25 * g * g / stiffness,
            bare_susceptibility: 0.25 * g * g / bare_stiffness,
            enhancement: bare_stiffness / stiffness,
            stiffness,
            bare_stiffness,
            delta_density,
            response_residual: response.relative_residual,
            iterations: response.iterations,
            number_defects: response.number_defects.map(|v| v * factor.abs()),
        })
    }

    pub fn new(
        mol: &Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        scf: &'a ScfResult,
        opts: &ScfOptions,
    ) -> Result<Self> {
        let kt = scf.electronic_temperature;
        let population_law = if scf.fixed_occupation_residual.is_some() && kt == 0.0 {
            PopulationLaw::Fixed
        } else if kt.is_finite() && kt > 0.0 && scf.fixed_occupation_residual.is_none() {
            PopulationLaw::FermiDirac(kt)
        } else {
            return Err(DftError::InvalidInput(
                "ensemble response needs either positive electronic temperature or a fixed-population reference at zero electronic temperature".into(),
            ));
        };
        if !scf.converged {
            return Err(DftError::InvalidInput(
                "the ensemble reference is not converged".into(),
            ));
        }
        let kernel = ResponseSolver::kernel_only(mol, mb, functional, grid, scf, opts)?;
        let restricted = !scf.unrestricted;
        let mut channels = Vec::new();
        let mut nmo = [0; 2];
        let mut minimum_fixed_gap: Option<f64> = None;
        for spin in 0..2 {
            let c = scf.mo_coeff[spin].clone();
            let e = &scf.mo_energy[spin];
            let f = &scf.occupations[spin];
            let n = c.cols;
            nmo[spin] = n;
            if e.len() != n
                || f.len() != n
                || e.iter().any(|x| !x.is_finite())
                || f.iter().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            {
                return Err(DftError::InvalidInput(
                    "inconsistent ensemble reference".into(),
                ));
            }
            let mut w = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    w[(i, j)] = population_law.weight(e[i], e[j], f[i], f[j])?;
                    if !w[(i, j)].is_finite() {
                        return Err(DftError::LinearAlgebra(
                            "non-finite ensemble spectral weight; the orbital gap must be resolved without regularization".into(),
                        ));
                    }
                    if matches!(population_law, PopulationLaw::Fixed) && f[i] != f[j] {
                        let gap = (e[i] - e[j]).abs();
                        minimum_fixed_gap = Some(minimum_fixed_gap.map_or(gap, |g| g.min(gap)));
                    }
                }
            }
            let w_diag_sum: f64 = (0..n).map(|i| w[(i, i)]).sum();
            channels.push(Channel { c, w, w_diag_sum });
        }
        Ok(Self {
            kernel,
            channels,
            restricted,
            nmo,
            population_law,
            minimum_fixed_gap,
        })
    }

    /// Population derivative held in the response; independent of functional names.
    pub fn population_model(&self) -> &'static str {
        match self.population_law {
            PopulationLaw::Fixed => "fixed nonthermal per-spin populations; df = 0",
            PopulationLaw::FermiDirac(_) => {
                "canonical Fermi-Dirac populations at fixed temperature"
            }
        }
    }

    /// Solve all three static electric fields using the same interaction kernel.
    /// A pre-existing electric bias is retained in the reference and its kernel.
    pub fn polarizability(&self, config: &ResponseSolverConfig) -> Result<EnsemblePolarizability> {
        let dipoles = crate::integrals::dipole_matrices(self.kernel.mb, crate::math::Vec3::zero());
        let mut responses = Vec::with_capacity(3);
        let mut tensor = [[0.0; 3]; 3];
        for (b, potential) in dipoles.iter().enumerate() {
            let response = self.solve(
                Some(&[potential.clone(), potential.clone()]),
                [0.0; 2],
                config,
            )?;
            for (a, moment) in dipoles.iter().enumerate() {
                tensor[a][b] = -response
                    .delta_density
                    .iter()
                    .map(|p| p.frobenius_dot(moment))
                    .sum::<f64>();
            }
            responses.push(response);
        }
        let [x, y, z] = <[EnsembleResponse; 3]>::try_from(responses).map_err(|_| {
            DftError::LinearAlgebra("incomplete Cartesian ensemble response".into())
        })?;
        Ok(EnsemblePolarizability {
            tensor,
            responses: [x, y, z],
        })
    }

    fn offsets(&self) -> [usize; 3] {
        let a = self.nmo[0] * self.nmo[0];
        [0, a, a + self.nmo[1] * self.nmo[1]]
    }

    /// Number-conserving part of the response density: `W o x` with the
    /// chemical-potential shift that keeps each trace fixed.
    fn conserving_density(&self, x: &[f64]) -> Vec<f64> {
        let offsets = self.offsets();
        let mut out = vec![0.0; x.len()];
        for spin in 0..2 {
            let ch = &self.channels[spin];
            let n = self.nmo[spin];
            let block = &x[offsets[spin]..offsets[spin + 1]];
            let trace_weighted: f64 = (0..n).map(|i| ch.w[(i, i)] * block[i * n + i]).sum();
            // At the integer-occupation limit the number-preserving density
            // remains defined; the chemical potential inside the gap is a gauge.
            let dmu = if ch.w_diag_sum == 0.0 {
                0.0
            } else {
                trace_weighted / ch.w_diag_sum
            };
            for i in 0..n {
                for j in 0..n {
                    let mut v = ch.w[(i, j)] * block[i * n + j];
                    if i == j {
                        v -= ch.w[(i, i)] * dmu;
                    }
                    out[offsets[spin] + i * n + j] = v;
                }
            }
        }
        out
    }

    fn to_ao(&self, mo: &[f64]) -> [Matrix; 2] {
        let offsets = self.offsets();
        let mut out = [Matrix::zeros(0, 0), Matrix::zeros(0, 0)];
        for spin in 0..2 {
            let ch = &self.channels[spin];
            let n = self.nmo[spin];
            let m = Matrix::from_row_major(n, n, mo[offsets[spin]..offsets[spin + 1]].to_vec());
            out[spin] = ch.c.matmul(&m.matmul(&ch.c.transpose()));
        }
        out
    }

    fn to_mo(&self, ao: &[Matrix; 2]) -> Vec<f64> {
        let offsets = self.offsets();
        let mut out = vec![0.0; offsets[2]];
        for spin in 0..2 {
            let ch = &self.channels[spin];
            let m = ch.c.transpose_matmul(&ao[spin].matmul(&ch.c));
            out[offsets[spin]..offsets[spin + 1]].copy_from_slice(m.as_slice());
        }
        out
    }

    fn kernel_mo(&self, density_mo: &[f64]) -> Result<Vec<f64>> {
        let dp = self.to_ao(density_mo);
        let df = self.kernel.fock_response(&dp)?;
        Ok(self.to_mo(&df))
    }

    /// Solve for an AO perturbation `potential` (per spin, may be zero) and
    /// electron-count changes `delta_numbers`.
    pub fn solve(
        &self,
        potential: Option<&[Matrix; 2]>,
        delta_numbers: [f64; 2],
        config: &ResponseSolverConfig,
    ) -> Result<EnsembleResponse> {
        let nao = self.kernel.scf.overlap.rows;
        if delta_numbers.iter().any(|v| !v.is_finite())
            || potential.is_some_and(|v| {
                v.iter().any(|m| {
                    m.rows != nao || m.cols != nao || m.as_slice().iter().any(|v| !v.is_finite())
                })
            })
        {
            return Err(DftError::InvalidInput(
                "ensemble response needs finite electron-count changes and matching finite AO potentials".into(),
            ));
        }
        let potential = potential.map_or_else(|| vec![0.0; self.offsets()[2]], |v| self.to_mo(v));
        Ok(self.solve_mo(&potential, delta_numbers, config)?.0)
    }

    fn solve_mo(
        &self,
        potential: &[f64],
        delta_numbers: [f64; 2],
        config: &ResponseSolverConfig,
    ) -> Result<(EnsembleResponse, Vec<f64>, Vec<f64>)> {
        config.validate()?;
        if self
            .minimum_fixed_gap
            .is_some_and(|g| g < config.min_denominator)
        {
            return Err(DftError::InvalidInput(format!(
                "fixed-population orbital gap {:.16e} Hartree is below response_min_denominator = {:.16e}; no denominator regularization is applied",
                self.minimum_fixed_gap.unwrap(), config.min_denominator,
            )));
        }
        if self.restricted && delta_numbers[0] != delta_numbers[1] {
            return Err(DftError::InvalidInput(
                "a restricted ensemble reference needs equal alpha and beta number changes".into(),
            ));
        }
        let offsets = self.offsets();
        // The density added by dN with no self-consistent field:
        // d_ii = W_ii / sum_k W_kk.
        let mut seed = vec![0.0; offsets[2]];
        for spin in 0..2 {
            let ch = &self.channels[spin];
            let n = self.nmo[spin];
            if ch.w_diag_sum == 0.0 {
                if delta_numbers[spin] != 0.0 {
                    return Err(DftError::Unsupported {
                        feature: "electron-number response without a resolved Fermi-surface weight"
                            .into(),
                        missing: "a population law with a nonzero number derivative; prescribed populations allow only fixed-number response".into(),
                    });
                }
                continue;
            }
            for i in 0..n {
                seed[offsets[spin] + i * n + i] =
                    delta_numbers[spin] * ch.w[(i, i)] / ch.w_diag_sum;
            }
        }
        let mut rhs = self.kernel_mo(&seed)?;
        for (r, x) in rhs.iter_mut().zip(potential) {
            *r += x;
        }
        struct Operator<'s, 'a>(&'s EnsembleSolver<'a>);
        impl LinearOperator for Operator<'_, '_> {
            fn dimension(&self) -> usize {
                self.0.offsets()[2]
            }
            fn structure(&self) -> OperatorStructure {
                OperatorStructure::General
            }
            fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
                krylov::apply_real_linear(x, |v| {
                    let induced = self.0.kernel_mo(&self.0.conserving_density(v))?;
                    Ok(v.iter().zip(induced).map(|(a, b)| a - b).collect())
                })
            }
        }
        let rhs_complex: Vec<Complex> = rhs.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let solution = krylov::solve(&Operator(self), &rhs_complex, None, config)?;
        let x: Vec<f64> = solution.amplitudes.iter().map(|z| z.re).collect();
        let mut density_mo = self.conserving_density(&x);
        for (d, s) in density_mo.iter_mut().zip(&seed) {
            *d += s;
        }
        let mut delta_mu = [0.0; 2];
        let mut number_defects = [0.0; 2];
        for spin in 0..2 {
            let ch = &self.channels[spin];
            let n = self.nmo[spin];
            let block = &x[offsets[spin]..offsets[spin + 1]];
            let trace_weighted: f64 = (0..n).map(|i| ch.w[(i, i)] * block[i * n + i]).sum();
            delta_mu[spin] = if ch.w_diag_sum == 0.0 {
                0.0
            } else {
                (trace_weighted - delta_numbers[spin]) / ch.w_diag_sum
            };
        }
        let delta_density = self.to_ao(&density_mo);
        for spin in 0..2 {
            number_defects[spin] = (self.kernel.scf.overlap.frobenius_dot(&delta_density[spin])
                - delta_numbers[spin])
                .abs();
            if !number_defects[spin].is_finite() || number_defects[spin] > config.acceptance() {
                return Err(DftError::ResponseNotConverged {
                    iterations: solution.iterations,
                    residual: number_defects[spin],
                });
            }
        }
        Ok((
            EnsembleResponse {
                delta_density,
                delta_chemical_potentials: delta_mu,
                relative_residual: solution.relative_residual,
                iterations: solution.iterations,
                number_defects,
            },
            density_mo,
            x,
        ))
    }

    /// Solve a fixed-temperature or fixed-population nuclear perturbation,
    /// conserving the number in each spin channel.
    /// `fock_skeleton` is the fixed-density AO derivative, `overlap` is dS.
    /// The reference MO frame is transported by -{dS,f}/2; the source includes
    /// its induced Hartree/XC/exchange potential and -{dS,epsilon}/2.
    /// No occupied/virtual partition or denominator modification is used.
    pub fn solve_metric(
        &self,
        fock_skeleton: &[Matrix; 2],
        overlap: &Matrix,
        config: &ResponseSolverConfig,
    ) -> Result<MetricEnsembleResponse> {
        let nao = self.kernel.scf.overlap.rows;
        if std::iter::once(overlap)
            .chain(fock_skeleton.iter())
            .any(|m| m.rows != nao || m.cols != nao || m.as_slice().iter().any(|v| !v.is_finite()))
        {
            return Err(DftError::InvalidInput(
                "metric ensemble response needs finite matching AO source/overlap matrices".into(),
            ));
        }
        let offsets = self.offsets();
        let mut metric = vec![0.0; offsets[2]];
        let sm = self.to_mo(&[overlap.clone(), overlap.clone()]);
        for spin in 0..2 {
            let n = self.nmo[spin];
            let occupations = &self.kernel.scf.occupations[spin];
            for i in 0..n {
                for j in 0..n {
                    let index = offsets[spin] + i * n + j;
                    metric[index] = -0.5 * sm[index] * (occupations[i] + occupations[j]);
                }
            }
        }
        let metric_ao = self.to_ao(&metric);
        let mut potential = self.to_mo(fock_skeleton);
        let induced = self.kernel_mo(&metric)?;
        for spin in 0..2 {
            let n = self.nmo[spin];
            let energies = &self.kernel.scf.mo_energy[spin];
            for i in 0..n {
                for j in 0..n {
                    let index = offsets[spin] + i * n + j;
                    potential[index] +=
                        induced[index] - 0.5 * sm[index] * (energies[i] + energies[j]);
                }
            }
        }
        let (mut response, density, hamiltonian) = self.solve_mo(&potential, [0.0; 2], config)?;
        let mut weighted = vec![0.0; offsets[2]];
        for spin in 0..2 {
            let n = self.nmo[spin];
            let energies = &self.kernel.scf.mo_energy[spin];
            let occupations = &self.kernel.scf.occupations[spin];
            for i in 0..n {
                for j in 0..n {
                    let index = offsets[spin] + i * n + j;
                    weighted[index] = 0.5 * (energies[i] + energies[j]) * density[index]
                        + 0.5 * (occupations[i] + occupations[j]) * hamiltonian[index]
                        - 0.5
                            * sm[index]
                            * (energies[i] * occupations[i] + energies[j] * occupations[j]);
                }
            }
            for (p, m) in response.delta_density[spin]
                .as_mut_slice()
                .iter_mut()
                .zip(metric_ao[spin].as_slice())
            {
                *p += m;
            }
            response.number_defects[spin] = (self
                .kernel
                .scf
                .overlap
                .frobenius_dot(&response.delta_density[spin])
                + overlap.frobenius_dot(&self.kernel.scf.density[spin]))
            .abs();
        }
        let mut weighted_ao = self.to_ao(&weighted);
        let beta = weighted_ao[1].clone();
        for (a, b) in weighted_ao[0]
            .as_mut_slice()
            .iter_mut()
            .zip(beta.as_slice())
        {
            *a += b;
        }
        let number_residual = response.number_defects.iter().fold(0.0_f64, |m, &x| {
            if x.is_finite() {
                m.max(x)
            } else {
                f64::INFINITY
            }
        });
        if number_residual > config.acceptance() {
            return Err(DftError::ResponseNotConverged {
                iterations: response.iterations,
                residual: number_residual,
            });
        }
        if response
            .delta_density
            .iter()
            .chain(std::iter::once(&weighted_ao[0]))
            .any(|m| m.as_slice().iter().any(|v| !v.is_finite()))
        {
            return Err(DftError::LinearAlgebra(
                "non-finite moving-metric ensemble density response".into(),
            ));
        }
        Ok(MetricEnsembleResponse {
            response,
            delta_energy_weighted_density: weighted_ao[0].clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PopulationLaw;

    #[test]
    fn fixed_degenerate_subspaces_and_exact_population_weights() {
        let law = PopulationLaw::Fixed;
        assert_eq!(law.weight(0.2, 0.2, 0.3, 0.3).unwrap(), 0.0);
        assert_eq!(law.weight(0.2, 0.4, 0.3, 0.3).unwrap(), 0.0);
        assert!(law.weight(0.2, 0.2, 0.3, 0.4).is_err());
        let gap = 1e-14;
        assert_eq!(law.weight(0.0, gap, 0.3, 0.4).unwrap(), (0.3 - 0.4) / -gap);
        assert_eq!(law.weight(0.0, 1.0, 0.3, 0.8).unwrap(), 0.5);
        assert_eq!(law.weight(1.0, 0.0, 0.8, 0.3).unwrap(), 0.5);
        assert_eq!(law.weight(0.1, 0.1, 1.0, 1.0).unwrap(), 0.0);
    }
}
