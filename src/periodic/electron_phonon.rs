// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electron-phonon coupling from the self-consistent displacement responses
//! of a (super)cell at Gamma.
//!
//! The Gamma point of an `N1 x N2 x N3` supercell contains every primitive
//! wave vector commensurate with it, and the supercell k mesh folds the
//! primitive one, so the Gamma vertices of the supercell are the
//! electron-phonon matrix elements of the primitive cell at those wave
//! vectors. With the total first-order Fock `F^x = F^x_skel + G[P^x]`, the
//! overlap derivative `S^x`, the mass-weighted mode `e_nu` of frequency
//! `w_nu` and the orbitals `C`,
//!
//! ```text
//! g^nu_mn(k) = [C^+ (dF^nu - (e_m + e_n)/2 dS^nu) C]_mn / sqrt(2 w_nu)
//! dF^nu = sum_x e_nu(x) F^x / sqrt(M_x),   dS^nu likewise
//! lambda_nu = 2 / (N_F w_nu Nk) sum_(k,m,n) |g^nu_mn|^2 d(e_m - mu) d(e_n - mu)
//! a2F(w) = 1/2 sum_nu w_nu lambda_nu delta(w - w_nu)
//! ```
//!
//! per spin, with `N_F = Nk^-1 sum_(k,n) d(e_n - mu)` and Gaussian
//! broadenings of the electronic and phonon delta functions. The diagonal
//! of the vertex is the eigenvalue derivative
//! `d e_n / d x = [C^+ (F^x - e_n S^x) C]_nn` (Hellmann-Feynman), which the
//! acceptance tests check against band energies of displaced cells.

use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::eliashberg::{
    kernel, matsubara_count, matsubara_frequency, renormalization, Spectrum,
};
use crate::periodic::phonon::{
    adjoint, assemble_force_constants, check_response_request, displacement_skeleton, product,
    DisplacementSkeleton, GammaForceConstants, Reference, Response, ResponseSolver,
};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::response::krylov::ResponseSolverConfig;
use crate::xc::ResolvedFunctional;

pub mod screening;

/// Controls of the coupling analysis.
#[derive(Clone, Copy, Debug)]
pub struct ElectronPhononOptions {
    /// Gaussian width of the electronic delta functions (Hartree).
    pub electron_smearing: f64,
    /// Gaussian width of the phonon delta functions in `a2F` (Hartree).
    pub phonon_smearing: f64,
    /// Modes at or below this frequency (Hartree) are left out.
    pub mode_cutoff: f64,
}

/// Electron-phonon coupling of one reference.
#[derive(Clone, Debug)]
pub struct ElectronPhonon {
    pub force_constants: GammaForceConstants,
    /// Mode frequencies (Hartree; negative = imaginary).
    pub frequencies: Vec<f64>,
    /// Coupling constant of every mode (zero for excluded modes).
    pub mode_lambda: Vec<f64>,
    /// Density of states at the Fermi level per spin and cell (1/Hartree).
    pub fermi_dos: [f64; 2],
    pub lambda: f64,
    /// Logarithmic and second-moment frequencies (Hartree).
    pub omega_log: f64,
    pub omega_2: f64,
    /// Eigenvalue derivatives `d e_n(k)/d x` for every spin, k, band and
    /// displacement, `[spin][k][band][x]` (Hartree/Bohr).
    pub eigenvalue_derivatives: Vec<Vec<Vec<Vec<f64>>>>,
    pub excluded_modes: usize,
    /// Actual Hartree vertices, [spin][k][mode][row,column]. Excluded
    /// nonpositive/cutoff modes have zero matrices, not shifted frequencies.
    pub vertices: [Vec<Vec<BlochMatrix>>; 2],
    pub normal_energies: [Vec<Vec<f64>>; 2],
    pub chemical_potentials: [f64; 2],
}

/// Scalar-generic mode vertex including both endpoint Pulay energies.
/// Fock/overlap rows are Cartesian displacement derivatives. Mode vectors
/// are normalized in mass-weighted coordinates; masses are electron masses.
pub fn mode_vertex<S: crate::dual::Scalar>(
    energies: [S; 2],
    fock: &[(S, S)],
    overlap: &[(S, S)],
    mode: &[S],
    masses: &[S],
    omega: S,
) -> Result<(S, S)> {
    if fock.len() != 3 * masses.len()
        || overlap.len() != fock.len()
        || mode.len() != fock.len()
        || !omega.value().is_finite()
        || omega.value() <= 0.0
        || masses
            .iter()
            .any(|x| !x.value().is_finite() || x.value() <= 0.0)
        || energies
            .iter()
            .chain(masses)
            .chain(mode)
            .any(|x| !x.value().is_finite())
        || fock
            .iter()
            .chain(overlap)
            .any(|(r, i)| !r.value().is_finite() || !i.value().is_finite())
    {
        return Err(DftError::InvalidInput(
            "mode vertex needs finite matching 3N directions, positive masses and frequency".into(),
        ));
    }
    let average = (energies[0] + energies[1]).scale(0.5);
    let (mut re, mut im) = (S::from_f64(0.0), S::from_f64(0.0));
    for x in 0..fock.len() {
        let amplitude = mode[x] / (masses[x / 3] * omega.scale(2.0)).sqrt();
        re = re + amplitude * (fock[x].0 - average * overlap[x].0);
        im = im + amplitude * (fock[x].1 - average * overlap[x].1);
    }
    Ok((re, im))
}

impl ElectronPhonon {
    /// Assemble a static band-decoupled singlet input from the computed
    /// supercell-Gamma vertices. Spin degeneracy is checked, never imposed.
    /// Each k block is retained; unavailable inter-supercell-k transfers are
    /// zero in this stated Gamma-vertex model, not a full fine-q interpolation.
    /// The screened Coulomb matrix is in flattened k/band order. Use the
    /// `screening` module to generate it from the same normal reference.
    pub fn pairing_input(
        &self,
        screened_coulomb: Vec<f64>,
        kt: f64,
        cutoff: f64,
        tolerance: f64,
    ) -> Result<(
        crate::properties::pairing::PairingInput<f64>,
        Vec<f64>,
        Vec<usize>,
        f64,
    )> {
        let nk = self.normal_energies[0].len();
        if nk == 0
            || self.normal_energies[1].len() != nk
            || self.vertices.iter().any(|spin| spin.len() != nk)
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || !cutoff.is_finite()
            || cutoff < 0.0
        {
            return Err(DftError::InvalidInput("SCDFT Gamma adapter needs finite mode cutoff, positive spin tolerance and nonempty matching k meshes".into()));
        }
        let mut spin_defect = (self.chemical_potentials[0] - self.chemical_potentials[1]).abs();
        for k in 0..nk {
            let a = &self.normal_energies[0][k];
            let b = &self.normal_energies[1][k];
            if a.len() != b.len() {
                return Err(DftError::InvalidInput(
                    "singlet pairing needs equal up/down band dimensions".into(),
                ));
            }
            for spin in 0..2 {
                if self.vertices[spin][k].len() != self.frequencies.len()
                    || self.vertices[spin][k].iter().any(|v| {
                        v.re.rows != a.len()
                            || v.re.cols != a.len()
                            || v.im.rows != a.len()
                            || v.im.cols != a.len()
                    })
                {
                    return Err(DftError::InvalidInput(
                        "Gamma pairing vertex dimensions do not match normal bands/modes".into(),
                    ));
                }
            }
            for (&x, &y) in a.iter().zip(b) {
                spin_defect = spin_defect.max((x - y).abs());
            }
            for (a, b) in self.vertices[0][k].iter().zip(&self.vertices[1][k]) {
                for (&x, &y) in
                    a.re.as_slice()
                        .iter()
                        .zip(b.re.as_slice())
                        .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
                {
                    spin_defect = spin_defect.max((x - y).abs());
                }
            }
        }
        if !spin_defect.is_finite() || spin_defect > tolerance {
            return Err(DftError::InvalidInput(format!("band-decoupled singlet Gamma adapter requires a spin-degenerate normal reference; raw spin defect {spin_defect:.6e}")));
        }
        let energies: Vec<_> = self.normal_energies[0].iter().flatten().copied().collect();
        let n = energies.len();
        let selected: Vec<_> = self
            .frequencies
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| (w > cutoff).then_some(i))
            .collect();
        if selected.is_empty() {
            return Err(DftError::InvalidInput(
                "SCDFT adapter has no positive mode above the explicit cutoff".into(),
            ));
        }
        let size = n
            .checked_mul(n)
            .ok_or_else(|| DftError::InvalidInput("Gamma pairing band count overflow".into()))?;
        let mut squared = vec![vec![0.0; size]; selected.len()];
        let mut offset = 0;
        for k in 0..nk {
            let bands = self.normal_energies[0][k].len();
            for (m, &nu) in selected.iter().enumerate() {
                let v = &self.vertices[0][k][nu];
                for i in 0..bands {
                    for j in 0..bands {
                        squared[m][(offset + i) * n + offset + j] =
                            v.re[(i, j)].powi(2) + v.im[(i, j)].powi(2);
                    }
                }
            }
            offset += bands;
        }
        let input = crate::properties::pairing::PairingInput {
            energies,
            chemical_potential: self.chemical_potentials[0],
            temperature_energy: kt,
            phonon_energies: selected.iter().map(|&i| self.frequencies[i]).collect(),
            coupling_squared: squared,
            screened_coulomb,
        };
        input.validate()?;
        Ok((input, vec![1.0 / nk as f64; n], selected, spin_defect))
    }
}

fn gaussian(x: f64, width: f64) -> f64 {
    (-0.5 * (x / width).powi(2)).exp() / (width * (2.0 * std::f64::consts::PI).sqrt())
}

/// Electron-phonon coupling from the Gamma displacement responses of `cell`.
pub fn electron_phonon(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    solver: &ResponseSolverConfig,
    controls: &ElectronPhononOptions,
) -> Result<ElectronPhonon> {
    check_response_request(
        cell,
        functional,
        result,
        options,
        "electron-phonon coupling",
        super::phonon::ResponseCapabilities {
            metal: true,
            exchange: true,
            scalar_ecp: false,
        },
    )?;
    if !(controls.electron_smearing > 0.0 && controls.phonon_smearing > 0.0) {
        return Err(DftError::InvalidInput(
            "electron and phonon smearing widths must be positive".into(),
        ));
    }
    let natoms = cell.molecule.len();
    let dim = 3 * natoms;
    let reference = Reference::new(cell, result, options)?;
    let DisplacementSkeleton { skeleton, first } =
        displacement_skeleton(cell, functional, options, &reference)?;
    let response = ResponseSolver::new(cell, functional, result, options, &reference)?;
    let responses: Vec<Response> = first
        .iter()
        .map(|f| response.solve(f, solver))
        .collect::<Result<_>>()?;
    let force_constants = assemble_force_constants(natoms, skeleton, &first, &responses);
    let masses: Vec<f64> = cell
        .molecule
        .atoms
        .iter()
        .map(|a| crate::constants::atomic_mass_au(a.z))
        .collect();
    let mut weighted = Matrix::zeros(dim, dim);
    for x in 0..dim {
        for y in 0..dim {
            weighted[(x, y)] =
                force_constants.hessian[(x, y)] / (masses[x / 3] * masses[y / 3]).sqrt();
        }
    }
    let (values, vectors) = crate::linalg::symmetric_eigen(&weighted)?;
    let frequencies: Vec<f64> = values.iter().map(|l| l.signum() * l.abs().sqrt()).collect();
    let nk = result.k_points.len();
    let mut fermi_dos = [0.0; 2];
    let mut mode_sums = vec![[0.0; 2]; dim];
    let mut eigenvalue_derivatives = vec![vec![Vec::new(); nk]; 2];
    let mut vertices: [Vec<Vec<BlochMatrix>>; 2] = std::array::from_fn(|_| vec![Vec::new(); nk]);
    let mut normal_energies: [Vec<Vec<f64>>; 2] = std::array::from_fn(|_| vec![Vec::new(); nk]);
    let mut chemical_potentials = [0.0; 2];
    for spin in 0..2 {
        let mu = match result.chemical_potential[spin] {
            Some(mu) => mu,
            None => {
                // Integer occupations: the midgap of this channel.
                let n_occ = if spin == 0 {
                    result.n_alpha
                } else {
                    result.n_beta
                };
                let homo = result.mo_energy[spin]
                    .iter()
                    .map(|e| e[n_occ.saturating_sub(1)])
                    .fold(f64::NEG_INFINITY, f64::max);
                let lumo = result.mo_energy[spin]
                    .iter()
                    .map(|e| e.get(n_occ).copied().unwrap_or(f64::INFINITY))
                    .fold(f64::INFINITY, f64::min);
                0.5 * (homo + lumo)
            }
        };
        chemical_potentials[spin] = mu;
        for ik in 0..nk {
            let orbitals = crate::dfpt::orbitals::generalized_orbitals(
                &result.fock[spin][ik],
                &result.overlap[ik],
                options.orth_threshold,
            )?;
            let c = &orbitals.coefficients;
            let ch = adjoint(c);
            let e = &orbitals.energies;
            let nmo = e.len();
            normal_energies[spin][ik] = e.clone();
            vertices[spin][ik] = vec![
                BlochMatrix {
                    re: Matrix::zeros(nmo, nmo),
                    im: Matrix::zeros(nmo, nmo)
                };
                dim
            ];
            let weights: Vec<f64> = e
                .iter()
                .map(|en| gaussian(en - mu, controls.electron_smearing))
                .collect();
            fermi_dos[spin] += weights.iter().sum::<f64>() / nk as f64;
            // Orbital-basis first-order matrices of every displacement.
            let f_mo: Vec<BlochMatrix> = responses
                .iter()
                .map(|r| product(&product(&ch, &r.fock[spin][ik]), c))
                .collect();
            let s_mo: Vec<BlochMatrix> = first
                .iter()
                .map(|f| product(&product(&ch, &f.overlap[ik]), c))
                .collect();
            eigenvalue_derivatives[spin][ik] = (0..nmo)
                .map(|n| {
                    (0..dim)
                        .map(|x| f_mo[x].re[(n, n)] - e[n] * s_mo[x].re[(n, n)])
                        .collect()
                })
                .collect();
            for (nu, w) in frequencies.iter().enumerate() {
                if *w <= controls.mode_cutoff {
                    continue;
                }
                let mut sum = 0.0;
                let mode: Vec<_> = (0..dim).map(|x| vectors[(x, nu)]).collect();
                for m in 0..nmo {
                    for n in 0..nmo {
                        let f: Vec<_> = (0..dim)
                            .map(|x| (f_mo[x].re[(m, n)], f_mo[x].im[(m, n)]))
                            .collect();
                        let s: Vec<_> = (0..dim)
                            .map(|x| (s_mo[x].re[(m, n)], s_mo[x].im[(m, n)]))
                            .collect();
                        let (re, im) = mode_vertex([e[m], e[n]], &f, &s, &mode, &masses, *w)?;
                        vertices[spin][ik][nu].re[(m, n)] = re;
                        vertices[spin][ik][nu].im[(m, n)] = im;
                        sum += weights[m] * weights[n] * (re * re + im * im);
                    }
                }
                mode_sums[nu][spin] += sum / nk as f64;
            }
        }
    }
    let mut mode_lambda = vec![0.0; dim];
    let mut excluded_modes = 0;
    for (nu, w) in frequencies.iter().enumerate() {
        if *w <= controls.mode_cutoff {
            excluded_modes += 1;
            continue;
        }
        // lambda_nu = 2 sum / (N_F w) per spin channel, averaged over spins.
        let mut lambda = 0.0;
        for spin in 0..2 {
            if fermi_dos[spin] > 0.0 {
                lambda += 0.5 * 2.0 * mode_sums[nu][spin] / (fermi_dos[spin] * w);
            }
        }
        mode_lambda[nu] = lambda;
    }
    let lambda: f64 = mode_lambda.iter().sum();
    let (omega_log, omega_2) = if lambda > 0.0 {
        let log: f64 = mode_lambda
            .iter()
            .zip(&frequencies)
            .filter(|(l, _)| **l > 0.0)
            .map(|(l, w)| l * w.ln())
            .sum();
        let second: f64 = mode_lambda
            .iter()
            .zip(&frequencies)
            .map(|(l, w)| l * w * w)
            .sum();
        ((log / lambda).exp(), (second / lambda).sqrt())
    } else {
        (0.0, 0.0)
    };
    Ok(ElectronPhonon {
        force_constants,
        frequencies,
        mode_lambda,
        fermi_dos,
        lambda,
        omega_log,
        omega_2,
        eigenvalue_derivatives,
        excluded_modes,
        vertices,
        normal_energies,
        chemical_potentials,
    })
}

/// `a2F(w)` on `grid` (Hartree) with Gaussian phonon broadening:
/// `1/2 sum_nu w_nu lambda_nu delta(w - w_nu)`.
pub fn eliashberg_function(
    frequencies: &[f64],
    mode_lambda: &[f64],
    grid: &[f64],
    width: f64,
) -> Vec<f64> {
    grid.iter()
        .map(|&w| {
            frequencies
                .iter()
                .zip(mode_lambda)
                .map(|(wn, l)| 0.5 * wn * l * gaussian(w - wn, width))
                .sum()
        })
        .collect()
}

/// Allen-Dynes critical temperature (same units as `omega_log`), with the
/// strong-coupling and shape factors `f1`, `f2`; `None` when the exponent
/// denominator is not positive. With `f1 = f2 = 1` it is the McMillan form
/// with `omega_log / 1.2` as prefactor.
pub fn allen_dynes(
    lambda: f64,
    mu_star: f64,
    omega_log: f64,
    omega_2: f64,
    corrections: bool,
) -> Option<f64> {
    let denominator = lambda - mu_star * (1.0 + 0.62 * lambda);
    if denominator <= 0.0 || omega_log <= 0.0 {
        return None;
    }
    let (mut f1, mut f2) = (1.0, 1.0);
    if corrections {
        let lambda1 = 2.46 * (1.0 + 3.8 * mu_star);
        let ratio = omega_2 / omega_log;
        let lambda2 = 1.82 * (1.0 + 6.3 * mu_star) * ratio;
        f1 = (1.0 + (lambda / lambda1).powf(1.5)).cbrt();
        f2 = 1.0 + (ratio - 1.0) * lambda * lambda / (lambda * lambda + lambda2 * lambda2);
    }
    Some(f1 * f2 * omega_log / 1.2 * (-1.04 * (1.0 + lambda) / denominator).exp())
}

/// Critical temperature of the isotropic Migdal-Eliashberg equations on the
/// imaginary axis for a discrete spectrum `a2F = 1/2 sum w_nu lambda_nu
/// delta(w - w_nu)`, so that `lambda(n) = sum lambda_nu w_nu^2 / (w_nu^2 + (2 pi n T)^2)`.
/// With positive Matsubara frequencies `w_n = pi T (2n + 1)` up to
/// `cutoff`, the linearized gap equation
///
/// ```text
/// rho Delta_n = sum_m pi T [lambda(n - m) + lambda(n + m + 1) - 2 mu*] Delta_m / (w_m Z_n)
/// Z_n = 1 + (pi T / w_n) sum_m [lambda(n - m) - lambda(n + m + 1)]
/// ```
///
/// has its largest eigenvalue `rho = 1` at `T_c`, found by bisection. The
/// spectrum, Matsubara grid and kernel are those of the nonlinear gap solver
/// ([`crate::periodic::eliashberg::eliashberg_gap`]).
/// Returns `None` when no superconducting solution exists above `t_min`.
pub fn eliashberg_tc(
    frequencies: &[f64],
    mode_lambda: &[f64],
    mu_star: f64,
    cutoff: f64,
    t_min: f64,
) -> Result<Option<f64>> {
    let spectrum = Spectrum::new(frequencies, mode_lambda);
    if spectrum.is_empty() || cutoff <= 0.0 {
        return Ok(None);
    }
    let largest = |t: f64| -> Result<f64> {
        let pi_t = std::f64::consts::PI * t;
        let n = matsubara_count(t, cutoff);
        let table = spectrum.table(t, n);
        let w = |m: usize| matsubara_frequency(t, m);
        let tail = spectrum.tail(t, n, &table);
        let z = renormalization(&table, &tail, pi_t, &vec![1.0; n]);
        // Symmetrize with D = diag(sqrt(Z_n w_n)): the kernel
        // pi T [lambda(n-m) + lambda(n+m+1) - 2 mu*] is symmetric.
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            for m in 0..n {
                let k = pi_t * kernel(&table, i, m, mu_star);
                a[(i, m)] = k / ((z[i] * w(i)).sqrt() * (z[m] * w(m)).sqrt());
            }
        }
        let (values, _) = crate::linalg::symmetric_eigen(&a)?;
        Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    };
    let mut high = spectrum.highest();
    while largest(high)? > 1.0 {
        high *= 2.0;
    }
    let mut low = t_min;
    if largest(low)? < 1.0 {
        return Ok(None);
    }
    for _ in 0..60 {
        let mid = (low * high).sqrt();
        if largest(mid)? > 1.0 {
            low = mid;
        } else {
            high = mid;
        }
        if high / low - 1.0 < 1e-6 {
            break;
        }
    }
    Ok(Some((low * high).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eliashberg_function_integrates_to_lambda_and_allen_dynes_limits() {
        let w = [0.001, 0.002, 0.0035];
        let l = [0.3, 0.5, 0.2];
        let grid: Vec<f64> = (0..20000).map(|i| i as f64 * 5e-7).collect();
        let a2f = eliashberg_function(&w, &l, &grid, 2e-5);
        let dw = 5e-7;
        let lambda: f64 = a2f
            .iter()
            .zip(&grid)
            .skip(1)
            .map(|(a, x)| 2.0 * a / x * dw)
            .sum();
        assert!((lambda - 1.0).abs() < 1e-3, "{lambda}");
        // A single Einstein mode: omega_log = omega_2 = w, so f2 = 1.
        let tc = allen_dynes(1.0, 0.1, 0.002, 0.002, true).unwrap();
        let mcmillan = allen_dynes(1.0, 0.1, 0.002, 0.002, false).unwrap();
        let f1 = (1.0 + (1.0_f64 / (2.46 * 1.38)).powf(1.5)).cbrt();
        assert!((tc / mcmillan - f1).abs() < 1e-12);
        assert!(
            (mcmillan - 0.002 / 1.2 * (-1.04_f64 * 2.0 / (1.0 - 0.1 * 1.62)).exp()).abs() < 1e-15
        );
        assert!(allen_dynes(0.1, 0.2, 0.002, 0.002, true).is_none());
    }
}
