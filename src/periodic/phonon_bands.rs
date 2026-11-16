// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phonons at arbitrary wave vectors from analytic supercell force
//! constants, phonon densities of states and harmonic lattice
//! thermodynamics.
//!
//! The Gamma-point force constants of an `N1 x N2 x N3` supercell
//! ([`super::phonon::gamma_force_constants`]) are the interatomic force
//! constants `Phi(k0; k'n)` of the primitive cell folded over the supercell
//! lattice. Every pair is assigned to the shortest images
//! `n.A + L.A_s` of the supercell lattice, with equal weights on images tied
//! within a tolerance, and
//!
//! ```text
//! D_(ka),(k'b)(q) = sum_(n,L) w Phi_(ka),(k'b)(n) exp(i q . (n.A + L.A_s)) / sqrt(m_k m_k')
//! ```
//!
//! is exact at the wave vectors commensurate with the supercell (where every
//! image phase is one) and an interpolation elsewhere. No acoustic sum rule
//! is imposed; the raw residual is reported.

use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::Lattice;
use crate::response::complex::Complex;
use crate::response::constraints::{solve_constrained, ConstraintSpace};
use crate::response::krylov::{LinearOperator, OperatorStructure, ResponseSolverConfig};

/// One supercell image of a primitive atom pair.
#[derive(Clone, Debug)]
struct Image {
    /// Lattice vector from the cell of atom `k` to the image cell of `k'`.
    translation: Vec3,
    weight: f64,
    block: [[f64; 3]; 3],
}

/// Interatomic force constants of the primitive cell from a supercell.
#[derive(Clone, Debug)]
pub struct ForceConstantModel {
    pub supercell: [usize; 3],
    pub lattice: Lattice,
    pub positions: Vec<Vec3>,
    /// Atomic masses (electron masses).
    pub masses: Vec<f64>,
    /// `pairs[k * natoms + k']`.
    pairs: Vec<Vec<Image>>,
    /// Largest `|sum_(k',n) Phi(k0; k'n)|` (Hartree/Bohr^2).
    pub acoustic_residual: f64,
}

/// Cell-periodic displacement amplitudes use R; atomic amplitudes use
/// R+tau_final-tau_initial. The latter convention is needed by acoustic
/// long-wave/strain-gradient moment expansions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhononPhase {
    Cell,
    Atomic,
}

/// Factorial-normalized Fourier coefficients at q(t)=q+t*direction. Both
/// triangles are accumulated from the force constants, without projection.
#[derive(Clone, Debug)]
pub struct ForceConstantFourier {
    pub real: Vec<Matrix>,
    pub imaginary: Vec<Matrix>,
    /// Maximum raw Hermiticity defect per coefficient (same units as it).
    pub hermiticity_defects: Vec<f64>,
}

/// Acoustic internal relaxation at fixed mass center, q=kappa*direction.
/// Displacement coefficient n has units Bohr^n per macroscopic displacement;
/// direction is dimensionless and kappa is 1/Bohr. Three columns are the
/// imposed Cartesian acoustic polarizations. No force-constant sum rule is
/// restored: the raw uniform-displacement force is reported separately.
#[derive(Clone, Debug)]
pub struct LongWaveRelaxation {
    pub real: Vec<Matrix>,
    pub imaginary: Vec<Matrix>,
    pub relative_residuals: Vec<Vec<f64>>,
    pub center_of_mass_defects: Vec<Vec<f64>>,
    pub acoustic_reactions_real: Vec<Matrix>,
    pub acoustic_reactions_imaginary: Vec<Matrix>,
    pub uniform_force_defect: f64,
}

struct DynamicalAction<'a> {
    real: &'a Matrix,
    imaginary: &'a Matrix,
}
impl LinearOperator for DynamicalAction<'_> {
    fn dimension(&self) -> usize {
        self.real.rows
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        if x.len() != self.dimension() {
            return Err(DftError::InvalidInput(
                "long-wave source shape mismatch".into(),
            ));
        }
        Ok((0..x.len())
            .map(|i| {
                (0..x.len()).fold(Complex::ZERO, |sum, j| {
                    sum + Complex::new(self.real[(i, j)], self.imaginary[(i, j)]) * x[j]
                })
            })
            .collect())
    }
}

impl LongWaveRelaxation {
    /// Born-charge part of the lattice-mediated polarization. The n-th
    /// coefficient has units e*Bohr^(n-3). In particular -P_2 is the type-I
    /// strain-gradient coefficient contracted with direction twice, e/Bohr.
    /// Clamped-ion polarization and dynamic multipole/metric terms are not
    /// supplied by Born charges and are not silently set to zero in a total.
    pub fn born_polarization(
        &self,
        born: &[[[f64; 3]; 3]],
        volume: f64,
    ) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
        if self.real.is_empty()
            || 3 * born.len() != self.real[0].rows
            || !volume.is_finite()
            || volume <= 0.0
            || born.iter().flatten().flatten().any(|x| !x.is_finite())
        {
            return Err(DftError::InvalidInput(
                "long-wave polarization needs matching finite Born charges and positive volume"
                    .into(),
            ));
        }
        let contract = |series: &[Matrix]| {
            series
                .iter()
                .map(|u| {
                    let mut p = Matrix::zeros(3, 3);
                    for (atom, z) in born.iter().enumerate() {
                        for a in 0..3 {
                            for b in 0..3 {
                                for x in 0..3 {
                                    p[(a, b)] += z[a][x] * u[(3 * atom + x, b)] / volume;
                                }
                            }
                        }
                    }
                    p
                })
                .collect::<Vec<_>>()
        };
        Ok((contract(&self.real), contract(&self.imaginary)))
    }
}

/// Supercell atom index of primitive atom `atom` in cell `n`.
pub fn supercell_index(atom: usize, natoms: usize, n: [usize; 3], supercell: [usize; 3]) -> usize {
    atom + natoms * (n[0] + supercell[0] * (n[1] + supercell[1] * n[2]))
}

/// The supercell lattice and atom positions (in [`supercell_index`] order).
pub fn build_supercell(
    lattice: &Lattice,
    positions: &[Vec3],
    supercell: [usize; 3],
) -> (Lattice, Vec<Vec3>) {
    let a = lattice.a;
    let big = Lattice::new(
        a[0] * supercell[0] as f64,
        a[1] * supercell[1] as f64,
        a[2] * supercell[2] as f64,
    );
    let natoms = positions.len();
    let mut out = vec![Vec3::zero(); natoms * supercell.iter().product::<usize>()];
    for n2 in 0..supercell[2] {
        for n1 in 0..supercell[1] {
            for n0 in 0..supercell[0] {
                let shift = a[0] * n0 as f64 + a[1] * n1 as f64 + a[2] * n2 as f64;
                for (atom, p) in positions.iter().enumerate() {
                    out[supercell_index(atom, natoms, [n0, n1, n2], supercell)] = *p + shift;
                }
            }
        }
    }
    (big, out)
}

impl ForceConstantModel {
    /// Fold the supercell Gamma Hessian (`3 Ns x 3 Ns`, atoms in
    /// [`supercell_index`] order) onto the primitive cell.
    pub fn from_supercell_hessian(
        lattice: &Lattice,
        positions: &[Vec3],
        masses: &[f64],
        supercell: [usize; 3],
        hessian: &Matrix,
    ) -> Result<Self> {
        let natoms = positions.len();
        let cells: usize = supercell.iter().product();
        if natoms == 0
            || masses.len() != natoms
            || masses.iter().any(|m| !m.is_finite() || *m <= 0.0)
            || positions.iter().any(|r| !r.norm().is_finite())
            || hessian.as_slice().iter().any(|v| !v.is_finite())
            || supercell.contains(&0)
            || hessian.rows != 3 * natoms * cells
            || hessian.cols != hessian.rows
        {
            return Err(DftError::InvalidInput(format!(
                "a {supercell:?} supercell Hessian of {natoms} atoms must be {0} x {0}",
                3 * natoms * cells
            )));
        }
        let a = lattice.a;
        let big = [
            a[0] * supercell[0] as f64,
            a[1] * supercell[1] as f64,
            a[2] * supercell[2] as f64,
        ];
        let mut pairs: Vec<Vec<Image>> = vec![Vec::new(); natoms * natoms];
        let mut acoustic_residual = 0.0_f64;
        for k in 0..natoms {
            let row = supercell_index(k, natoms, [0, 0, 0], supercell);
            let mut sums = [[0.0; 3]; 3];
            for n2 in 0..supercell[2] {
                for n1 in 0..supercell[1] {
                    for n0 in 0..supercell[0] {
                        let cell = a[0] * n0 as f64 + a[1] * n1 as f64 + a[2] * n2 as f64;
                        for kp in 0..natoms {
                            let col = supercell_index(kp, natoms, [n0, n1, n2], supercell);
                            let block: [[f64; 3]; 3] = std::array::from_fn(|x| {
                                std::array::from_fn(|y| hessian[(3 * row + x, 3 * col + y)])
                            });
                            for x in 0..3 {
                                for y in 0..3 {
                                    sums[x][y] += block[x][y];
                                }
                            }
                            // Shortest supercell images of the pair vector.
                            let base = positions[kp] + cell - positions[k];
                            let mut candidates: Vec<(Vec3, f64)> = Vec::new();
                            for l0 in -2..=2i32 {
                                for l1 in -2..=2i32 {
                                    for l2 in -2..=2i32 {
                                        let shift = big[0] * l0 as f64
                                            + big[1] * l1 as f64
                                            + big[2] * l2 as f64;
                                        candidates.push((cell + shift, (base + shift).norm()));
                                    }
                                }
                            }
                            let shortest =
                                candidates.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
                            let tied: Vec<Vec3> = candidates
                                .iter()
                                .filter(|c| c.1 <= shortest * (1.0 + 1e-8) + 1e-8)
                                .map(|c| c.0)
                                .collect();
                            let weight = 1.0 / tied.len() as f64;
                            for translation in tied {
                                pairs[k * natoms + kp].push(Image {
                                    translation,
                                    weight,
                                    block,
                                });
                            }
                        }
                    }
                }
            }
            for row in sums {
                for v in row {
                    acoustic_residual = acoustic_residual.max(v.abs());
                }
            }
        }
        Ok(Self {
            supercell,
            lattice: lattice.clone(),
            positions: positions.to_vec(),
            masses: masses.to_vec(),
            pairs,
            acoustic_residual,
        })
    }

    pub fn natoms(&self) -> usize {
        self.positions.len()
    }

    /// Analytic constrained solution of Q D(q) Q v(q)=-Q D(q) u0.
    /// Excluded translations are sqrt(m_atom)*e_axis; internal Cartesian
    /// displacements therefore obey sum_atom m_atom*u_atom,n=0 for n>0.
    /// The raw original equations and force-pattern reactions are reevaluated
    /// by the common constrained solver at every order. Unstable or additional
    /// zero optical modes are not removed or shifted by a denominator floor.
    pub fn long_wave_relaxation(
        &self,
        direction: Vec3,
        order: usize,
        config: &ResponseSolverConfig,
    ) -> Result<LongWaveRelaxation> {
        if order > 4 || !direction.norm().is_finite() || direction.norm() == 0.0 {
            return Err(DftError::InvalidInput(
                "long-wave relaxation needs finite nonzero direction and order zero through four"
                    .into(),
            ));
        }
        config.validate()?;
        let series =
            self.fourier_taylor(Vec3::zero(), direction, order, PhononPhase::Atomic, true)?;
        let dim = 3 * self.natoms();
        let total_mass_root = self.masses.iter().sum::<f64>().sqrt();
        let translations: Vec<Vec<Complex>> = (0..3)
            .map(|a| {
                (0..dim)
                    .map(|i| {
                        Complex::from(if i % 3 == a {
                            self.masses[i / 3].sqrt()
                        } else {
                            0.0
                        })
                    })
                    .collect()
            })
            .collect();
        let space = ConstraintSpace::new(dim, &translations)?;
        let op = DynamicalAction {
            real: &series.real[0],
            imaginary: &series.imaginary[0],
        };
        let mut real = vec![Matrix::zeros(dim, 3); order + 1];
        let mut imaginary = real.clone();
        let mut reactions_real = vec![Matrix::zeros(3, 3); order + 1];
        let mut reactions_imag = reactions_real.clone();
        let mut residuals = vec![vec![0.0; 3]; order + 1];
        let mut centers = residuals.clone();
        let mut uniform = 0.0_f64;
        for a in 0..3 {
            let mut coefficients = vec![translations[a].clone()];
            for (i, value) in op.apply(&coefficients[0])?.iter().enumerate() {
                uniform = uniform.max(value.abs() * self.masses[i / 3].sqrt());
            }
            for n in 1..=order {
                let mut rhs = vec![Complex::ZERO; dim];
                for k in 1..=n {
                    let image = DynamicalAction {
                        real: &series.real[k],
                        imaginary: &series.imaginary[k],
                    }
                    .apply(&coefficients[n - k])?;
                    for (b, x) in rhs.iter_mut().zip(image) {
                        *b -= x;
                    }
                }
                let solved = solve_constrained(&op, &rhs, &space, config)?;
                residuals[n][a] = solved.solution.relative_residual;
                centers[n][a] = solved.constraint_defect / total_mass_root;
                for b in 0..3 {
                    reactions_real[n][(b, a)] = solved.multipliers[b].re * total_mass_root;
                    reactions_imag[n][(b, a)] = solved.multipliers[b].im * total_mass_root;
                }
                coefficients.push(solved.solution.amplitudes);
            }
            for n in 0..=order {
                for i in 0..dim {
                    real[n][(i, a)] = coefficients[n][i].re / self.masses[i / 3].sqrt();
                    imaginary[n][(i, a)] = coefficients[n][i].im / self.masses[i / 3].sqrt();
                }
            }
        }
        Ok(LongWaveRelaxation {
            real,
            imaginary,
            relative_residuals: residuals,
            center_of_mass_defects: centers,
            acoustic_reactions_real: reactions_real,
            acoustic_reactions_imaginary: reactions_imag,
            uniform_force_defect: uniform,
        })
    }

    /// Mass-weighted dynamical matrix at `q` (1/Bohr), real and imaginary
    /// parts (Hartree/(Bohr^2 electron mass)).
    pub fn dynamical_matrix(&self, q: Vec3) -> (Matrix, Matrix) {
        let mut series = self.fourier_series(q, Vec3::zero(), 0, PhononPhase::Cell, true);
        (series.real.remove(0), series.imaginary.remove(0))
    }

    /// Analytic wavevector response of the same real-space model used by
    /// `dynamical_matrix`. Direction is Cartesian 1/Bohr per dimensionless t.
    /// Mass weighting divides by sqrt(m_initial*m_final). Without it the
    /// returned coefficients are Fourier force constants (Hartree/Bohr^2).
    /// Converge supercell size before interpreting a long-wave moment as a
    /// bulk material coefficient; the interpolation does not restore missing
    /// long-range dipole, quadrupole, or electronic flexoelectric terms.
    pub fn fourier_taylor(
        &self,
        q: Vec3,
        direction: Vec3,
        order: usize,
        phase: PhononPhase,
        mass_weighted: bool,
    ) -> Result<ForceConstantFourier> {
        if !q.norm().is_finite() || !direction.norm().is_finite() || order > 32 {
            return Err(DftError::InvalidInput("force-constant Fourier response needs finite q/direction and order zero through 32".into()));
        }
        let result = self.fourier_series(q, direction, order, phase, mass_weighted);
        if result
            .real
            .iter()
            .chain(&result.imaginary)
            .flat_map(|a| a.as_slice())
            .any(|x| !x.is_finite())
        {
            return Err(DftError::LinearAlgebra(
                "force-constant Fourier coefficients overflow".into(),
            ));
        }
        Ok(result)
    }

    fn fourier_series(
        &self,
        q: Vec3,
        direction: Vec3,
        order: usize,
        phase: PhononPhase,
        mass_weighted: bool,
    ) -> ForceConstantFourier {
        let natoms = self.natoms();
        let dim = 3 * natoms;
        let mut re = vec![Matrix::zeros(dim, dim); order + 1];
        let mut im = re.clone();
        for k in 0..natoms {
            for kp in 0..natoms {
                let scale = if mass_weighted {
                    1.0 / (self.masses[k] * self.masses[kp]).sqrt()
                } else {
                    1.0
                };
                for image in &self.pairs[k * natoms + kp] {
                    let vector = image.translation
                        + if phase == PhononPhase::Atomic {
                            self.positions[kp] - self.positions[k]
                        } else {
                            Vec3::zero()
                        };
                    let angle = q.dot(vector);
                    let slope = direction.dot(vector);
                    let (mut c, mut s) = (
                        angle.cos() * image.weight * scale,
                        angle.sin() * image.weight * scale,
                    );
                    for n in 0..=order {
                        for x in 0..3 {
                            for y in 0..3 {
                                re[n][(3 * k + x, 3 * kp + y)] += c * image.block[x][y];
                                im[n][(3 * k + x, 3 * kp + y)] += s * image.block[x][y];
                            }
                        }
                        (c, s) = (-s * slope / (n + 1) as f64, c * slope / (n + 1) as f64);
                    }
                }
            }
        }
        let hermiticity_defects = (0..=order)
            .map(|n| {
                let mut defect = 0.0_f64;
                for i in 0..dim {
                    for j in 0..dim {
                        defect = defect
                            .max((re[n][(i, j)] - re[n][(j, i)]).abs())
                            .max((im[n][(i, j)] + im[n][(j, i)]).abs());
                    }
                }
                defect
            })
            .collect();
        ForceConstantFourier {
            real: re,
            imaginary: im,
            hermiticity_defects,
        }
    }

    /// Frequencies at `q` in cm^-1 (negative = imaginary), ascending, and
    /// the largest Hermiticity defect of the raw dynamical matrix.
    pub fn frequencies(&self, q: Vec3) -> Result<(Vec<f64>, f64)> {
        let (re, im) = self.dynamical_matrix(q);
        let dim = re.rows;
        let mut defect = 0.0_f64;
        // Hermitian part as the real symmetric 2n x 2n matrix [[A, -B], [B, A]].
        let mut big = Matrix::zeros(2 * dim, 2 * dim);
        for x in 0..dim {
            for y in 0..dim {
                defect = defect
                    .max((re[(x, y)] - re[(y, x)]).abs())
                    .max((im[(x, y)] + im[(y, x)]).abs());
                let a = 0.5 * (re[(x, y)] + re[(y, x)]);
                let b = 0.5 * (im[(x, y)] - im[(y, x)]);
                big[(x, y)] = a;
                big[(dim + x, dim + y)] = a;
                big[(x, dim + y)] = -b;
                big[(dim + x, y)] = b;
            }
        }
        let (values, _) = crate::linalg::symmetric_eigen(&big)?;
        // Every eigenvalue of the Hermitian matrix appears twice.
        let mut sorted = values;
        sorted.sort_by(|a, b| a.total_cmp(b));
        let frequencies = sorted
            .chunks(2)
            .map(|pair| {
                let l = 0.5 * (pair[0] + pair[1]);
                l.signum() * l.abs().sqrt() * crate::constants::HARTREE_TO_CM1
            })
            .collect();
        Ok((frequencies, defect))
    }
}

/// Harmonic lattice thermodynamics per primitive cell from frequencies on a
/// uniform q mesh (cm^-1); modes below `cutoff_cm1` (the acoustic modes at
/// Gamma and any imaginary mode) are left out and counted.
#[derive(Clone, Debug)]
pub struct HarmonicThermo {
    pub temperature: f64,
    /// Zero-point energy (Hartree per cell).
    pub zero_point_energy: f64,
    /// Vibrational internal energy including the zero point (Hartree per cell).
    pub internal_energy: f64,
    /// Helmholtz free energy including the zero point (Hartree per cell).
    pub free_energy: f64,
    /// Entropy (Hartree/K per cell).
    pub entropy: f64,
    /// Heat capacity at constant volume (Hartree/K per cell).
    pub heat_capacity: f64,
}

pub fn harmonic_thermo(
    frequencies: &[Vec<f64>],
    temperatures: &[f64],
    cutoff_cm1: f64,
) -> (Vec<HarmonicThermo>, usize) {
    let kb = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let nq = frequencies.len().max(1) as f64;
    let modes: Vec<f64> = frequencies
        .iter()
        .flatten()
        .filter(|f| **f > cutoff_cm1)
        .map(|f| f / crate::constants::HARTREE_TO_CM1)
        .collect();
    let excluded = frequencies.iter().flatten().count() - modes.len();
    let out = temperatures
        .iter()
        .map(|&t| {
            let mut thermo = HarmonicThermo {
                temperature: t,
                zero_point_energy: 0.0,
                internal_energy: 0.0,
                free_energy: 0.0,
                entropy: 0.0,
                heat_capacity: 0.0,
            };
            for &w in &modes {
                thermo.zero_point_energy += 0.5 * w;
                thermo.internal_energy += 0.5 * w;
                thermo.free_energy += 0.5 * w;
                if t > 0.0 {
                    let x = w / (kb * t);
                    let em = (-x).exp();
                    let n = em / (1.0 - em);
                    thermo.internal_energy += w * n;
                    thermo.free_energy += kb * t * (-em).ln_1p();
                    thermo.entropy += kb * (x * n - (-em).ln_1p());
                    thermo.heat_capacity += kb * x * x * em / ((1.0 - em) * (1.0 - em));
                }
            }
            thermo.zero_point_energy /= nq;
            thermo.internal_energy /= nq;
            thermo.free_energy /= nq;
            thermo.entropy /= nq;
            thermo.heat_capacity /= nq;
            thermo
        })
        .collect();
    (out, excluded)
}

/// Gaussian-broadened phonon density of states per primitive cell
/// (states per cm^-1), normalized to `3N` modes.
pub fn phonon_dos(frequencies: &[Vec<f64>], grid: &[f64], sigma_cm1: f64) -> Vec<f64> {
    let nq = frequencies.len().max(1) as f64;
    let norm = 1.0 / (sigma_cm1 * (2.0 * std::f64::consts::PI).sqrt() * nq);
    grid.iter()
        .map(|&e| {
            frequencies
                .iter()
                .flatten()
                .map(|f| {
                    let x = (e - f) / sigma_cm1;
                    norm * (-0.5 * x * x).exp()
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A monatomic chain with nearest- and next-nearest-neighbour springs:
    /// the folded supercell constants reproduce the closed-form dispersion
    /// at every commensurate wave vector.
    #[test]
    fn folded_chain_reproduces_the_closed_form_dispersion() {
        let spacing = 3.0;
        let lattice = Lattice::new(
            Vec3::new(spacing, 0.0, 0.0),
            Vec3::new(0.0, 20.0, 0.0),
            Vec3::new(0.0, 0.0, 20.0),
        );
        let (k1, k2, mass) = (0.08, 0.015, 1837.0);
        for cells in [4usize, 6] {
            let n = cells;
            let mut h = Matrix::zeros(3 * n, 3 * n);
            for i in 0..n {
                for (j, k) in [
                    ((i + 1) % n, k1),
                    ((i + n - 1) % n, k1),
                    ((i + 2) % n, k2),
                    ((i + n - 2) % n, k2),
                ] {
                    h[(3 * i, 3 * j)] -= k;
                    h[(3 * i, 3 * i)] += k;
                }
            }
            let model = ForceConstantModel::from_supercell_hessian(
                &lattice,
                &[Vec3::zero()],
                &[mass],
                [n, 1, 1],
                &h,
            )
            .unwrap();
            assert!(model.acoustic_residual < 1e-14);
            for m in 0..n {
                let q = 2.0 * std::f64::consts::PI * m as f64 / (n as f64 * spacing);
                let (f, defect) = model.frequencies(Vec3::new(q, 0.0, 0.0)).unwrap();
                assert!(defect < 1e-14);
                let w2 = (2.0 * k1 * (1.0 - (q * spacing).cos())
                    + 2.0 * k2 * (1.0 - (2.0 * q * spacing).cos()))
                    / mass;
                let expected = w2.sqrt() * crate::constants::HARTREE_TO_CM1;
                let longitudinal = f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    (longitudinal - expected).abs() < 1e-8 * (1.0 + expected),
                    "{cells} {m}: {longitudinal} {expected}"
                );
            }
        }
    }

    #[test]
    fn harmonic_thermodynamics_matches_one_einstein_oscillator() {
        let w_cm = 500.0;
        let (thermo, excluded) = harmonic_thermo(&[vec![0.0, w_cm]], &[0.0, 300.0], 1.0);
        assert_eq!(excluded, 1);
        let kb = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
        let w = w_cm / crate::constants::HARTREE_TO_CM1;
        assert!((thermo[0].free_energy - 0.5 * w).abs() < 1e-16);
        let x = w / (kb * 300.0);
        let f = 0.5 * w + kb * 300.0 * (1.0 - (-x).exp()).ln();
        let s = kb * (x / (x.exp() - 1.0) - (1.0 - (-x).exp()).ln());
        assert!((thermo[1].free_energy - f).abs() < 1e-15);
        assert!((thermo[1].entropy - s).abs() < 1e-18);
        // F = U - T S.
        assert!(
            (thermo[1].free_energy - (thermo[1].internal_energy - 300.0 * thermo[1].entropy)).abs()
                < 1e-15
        );
    }
}
