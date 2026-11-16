// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical complex singlet gap/number response in the band-decoupling model.
//! The provider supplies the same scalar-generic normal bands, diagonal mass
//! term Z and off-diagonal kernel K at every trial chemical potential/gap.
//! The equations are (1+Z)Delta+K*w*chi=0 and N=sum w*(1-2*xi*q),
//! chi=q*Delta, q=tanh(E/2kT)/(2E). A bordered global phase constraint removes
//! the Goldstone ambiguity. Its multiplier and all original physical
//! residuals are checked, rather than accepting only a projected equation.

use crate::dual::{DualN, Scalar, Taylor};
use crate::properties::pairing::{Lm2005, Lm2005OffDiagonal, PairingInput};
use crate::response::complex::{norm, Complex};
use crate::response::constraints::ConstraintSpace;
use crate::response::krylov::{
    apply_real_linear, solve, LinearOperator, OperatorStructure, ResponseSolverConfig,
};
use crate::{DftError, Result};

mod critical_temperature;
mod phase;
pub use critical_temperature::{critical_temperature, CriticalTemperature, NormalBandSpectrum};
pub use phase::{select_band_phase, BandPhase, BandPhaseResult};

#[derive(Clone, Debug)]
pub struct GapFunctionalData<S: Scalar> {
    pub energies: Vec<S>,
    pub weights: Vec<S>,
    pub temperature_energy: S,
    pub mass_renormalization: Vec<S>,
    /// Real reciprocal kernel, row-major, Hartree; band weights excluded.
    pub kernel: Vec<S>,
}

/// A static U(1)-covariant singlet functional, independent of registry names.
/// All dependence on parameters, chemical potential and gap must retain S.
/// This includes mass/kernel response when their providers depend on mu.
pub trait GapFunctional {
    fn bands(&self) -> usize;
    fn evaluate<S: Scalar + Send + Sync>(
        &self,
        parameters: &[S],
        chemical_potential: S,
        gap_real: &[S],
        gap_imaginary: &[S],
    ) -> Result<GapFunctionalData<S>>;
}

/// Supplied band model. Parameters are energies[n], weights[n], Z[n],
/// kT[1], then the reciprocal K[n*n]. Z must be explicitly supplied.
#[derive(Clone, Debug)]
pub struct SuppliedBandFunctional {
    pub bands: usize,
}

/// LM2005 off-diagonal phonon/Coulomb provider with an explicitly supplied
/// diagonal mass term. Parameters: energies[n], weights[n], Z[n], kT[1],
/// phonon energies[m], squared vertices[m*n*n], screened Coulomb[n*n].
/// This provider recomputes K at the responding chemical potential.
#[derive(Clone, Debug)]
pub struct PhononBandFunctional {
    pub bands: usize,
    pub modes: usize,
    pub quadrature: Lm2005OffDiagonal,
}

/// LM2005 phonon K and Z on the same normal bands, recomputed at each mu.
/// Parameters: energies[n], weights[n], kT[1], phonon energies[m], squared
/// vertices[m*n*n], screened Coulomb[n*n]. No mass input is substituted.
#[derive(Clone, Debug)]
pub struct Lm2005BandFunctional {
    pub bands: usize,
    pub modes: usize,
    pub quadrature: Lm2005,
}

fn header<S: Scalar>(
    parameters: &[S],
    n: usize,
    tail: usize,
    supplied_mass: bool,
) -> Result<GapFunctionalData<S>> {
    let components = if supplied_mass { 3 } else { 2 };
    if n == 0
        || n.checked_mul(components)
            .and_then(|x| x.checked_add(1))
            .and_then(|x| x.checked_add(tail))
            != Some(parameters.len())
    {
        return Err(DftError::InvalidInput(
            "invalid superconducting functional parameter layout".into(),
        ));
    }
    Ok(GapFunctionalData {
        energies: parameters[..n].to_vec(),
        weights: parameters[n..2 * n].to_vec(),
        mass_renormalization: if supplied_mass {
            parameters[2 * n..3 * n].to_vec()
        } else {
            Vec::new()
        },
        temperature_energy: parameters[components * n],
        kernel: Vec::new(),
    })
}

impl GapFunctional for SuppliedBandFunctional {
    fn bands(&self) -> usize {
        self.bands
    }
    fn evaluate<S: Scalar + Send + Sync>(
        &self,
        p: &[S],
        _: S,
        _: &[S],
        _: &[S],
    ) -> Result<GapFunctionalData<S>> {
        let size = self
            .bands
            .checked_mul(self.bands)
            .ok_or_else(|| DftError::InvalidInput("gap band count overflow".into()))?;
        let mut data = header(p, self.bands, size, true)?;
        data.kernel = p[3 * self.bands + 1..].to_vec();
        Ok(data)
    }
}
impl GapFunctional for PhononBandFunctional {
    fn bands(&self) -> usize {
        self.bands
    }
    fn evaluate<S: Scalar + Send + Sync>(
        &self,
        p: &[S],
        mu: S,
        _: &[S],
        _: &[S],
    ) -> Result<GapFunctionalData<S>> {
        phonon_data(self.bands, self.modes, &self.quadrature, p, mu, true)
    }
}

impl GapFunctional for Lm2005BandFunctional {
    fn bands(&self) -> usize {
        self.bands
    }
    fn evaluate<S: Scalar + Send + Sync>(
        &self,
        p: &[S],
        mu: S,
        _: &[S],
        _: &[S],
    ) -> Result<GapFunctionalData<S>> {
        phonon_data(self.bands, self.modes, &self.quadrature, p, mu, false)
    }
}

fn phonon_data<S: Scalar + Send + Sync>(
    bands: usize,
    modes: usize,
    quadrature: &Lm2005,
    p: &[S],
    mu: S,
    supplied_mass: bool,
) -> Result<GapFunctionalData<S>> {
    let size = bands
        .checked_mul(bands)
        .ok_or_else(|| DftError::InvalidInput("gap band count overflow".into()))?;
    let tail = size
        .checked_mul(modes)
        .and_then(|x| x.checked_add(modes))
        .and_then(|x| x.checked_add(size))
        .ok_or_else(|| DftError::InvalidInput("gap phonon layout overflow".into()))?;
    let mut data = header(p, bands, tail, supplied_mass)?;
    let start = (if supplied_mass { 3 } else { 2 }) * bands + 1;
    let vertices = start + modes;
    let coulomb = vertices + modes * size;
    let input = PairingInput {
        energies: data.energies.clone(),
        chemical_potential: mu,
        temperature_energy: data.temperature_energy,
        phonon_energies: p[start..vertices].to_vec(),
        coupling_squared: p[vertices..coulomb]
            .chunks(size)
            .map(|row| row.to_vec())
            .collect(),
        screened_coulomb: p[coulomb..].to_vec(),
    };
    data.kernel = quadrature.evaluate(&input)?.total;
    if !supplied_mass {
        data.mass_renormalization = quadrature.mass(&input, &data.weights)?;
    }
    Ok(data)
}

/// Initial branch/phase selection. Gap and chemical potential are Hartree.
#[derive(Clone, Debug)]
pub struct GapState {
    pub gap: Vec<Complex>,
    pub chemical_potential: f64,
}

#[derive(Clone, Debug)]
pub struct GapTaylor {
    pub gap_real: Vec<Vec<f64>>,
    pub gap_imaginary: Vec<Vec<f64>>,
    pub chemical_potential: Vec<f64>,
    pub anomalous_real: Vec<Vec<f64>>,
    pub anomalous_imaginary: Vec<Vec<f64>>,
    /// Spin-summed occupations per band, before quadrature weights.
    pub occupations: Vec<Vec<f64>>,
    pub quasiparticle_energies: Vec<f64>,
    /// Squared quasiparticle energies remain analytic at normal-state zeros.
    pub quasiparticle_energy_squared: Vec<Vec<f64>>,
    pub kernel_coefficients: Vec<Vec<Vec<f64>>>,
    pub kernel_asymmetry: Vec<f64>,
    pub mass_renormalization_coefficients: Vec<Vec<f64>>,
    pub gap_residuals: Vec<f64>,
    pub number_residuals: Vec<f64>,
    pub phase_residuals: Vec<f64>,
    pub phase_multipliers: Vec<f64>,
    pub goldstone_defects: Vec<f64>,
    pub linear_residuals: Vec<f64>,
    pub linear_iterations: Vec<usize>,
    pub reference_iterations: usize,
    pub reference_scaled_residual: f64,
    pub energy_scale: f64,
    pub phase_reference: Vec<Complex>,
}

fn validate_data<S: Scalar>(d: &GapFunctionalData<S>, n: usize) -> Result<()> {
    if n == 0
        || d.energies.len() != n
        || d.weights.len() != n
        || d.mass_renormalization.len() != n
        || n.checked_mul(n) != Some(d.kernel.len())
        || !d.temperature_energy.value().is_finite()
        || d.temperature_energy.value() <= 0.0
        || d.energies
            .iter()
            .chain(&d.kernel)
            .any(|x| !x.value().is_finite())
        || d.weights
            .iter()
            .any(|x| !x.value().is_finite() || x.value() <= 0.0)
        || d.mass_renormalization
            .iter()
            .any(|x| !x.value().is_finite() || x.value() <= -1.0)
    {
        return Err(DftError::InvalidInput("singlet gap data require matching finite bands/kernel/Z, positive weights/kT and 1+Z>0".into()));
    }
    let scale = d.kernel.iter().fold(0.0_f64, |m, x| m.max(x.value().abs()));
    for i in 0..n {
        for j in 0..n {
            if (d.kernel[i * n + j].value() - d.kernel[j * n + i].value()).abs() > 1e-10 * scale {
                return Err(DftError::InvalidInput("static singlet pairing kernel must be reciprocal before band weights; no symmetrization is applied".into()));
            }
        }
    }
    Ok(())
}

/// Analytic thermal factor as a function of E^2, including E=0. The local
/// tanh series is convergent; no energy or gap denominator is floored.
pub fn anomalous_factor<S: Scalar>(energy_squared: S, kt: S) -> Result<S> {
    if !energy_squared.value().is_finite()
        || energy_squared.value() < 0.0
        || !kt.value().is_finite()
        || kt.value() <= 0.0
    {
        return Err(DftError::InvalidInput(
            "anomalous thermal factor needs nonnegative finite E squared and positive kT".into(),
        ));
    }
    let u = energy_squared / (kt * kt).scale(4.0);
    let value = if u.value() < 0.25 {
        let mut coefficients = [0.0; 32];
        coefficients[0] = 1.0;
        for n in 1..32 {
            coefficients[n] = -(0..n)
                .map(|k| coefficients[k] * coefficients[n - 1 - k])
                .sum::<f64>()
                / (2 * n + 1) as f64;
        }
        coefficients
            .iter()
            .rev()
            .fold(S::from_f64(0.0), |s, &c| s * u + S::from_f64(c))
            / kt.scale(4.0)
    } else {
        let e = energy_squared.sqrt();
        let exponential = (-e / kt).exp();
        (S::from_f64(1.0) - exponential) / (S::from_f64(1.0) + exponential) / e.scale(2.0)
    };
    if !value.value().is_finite() {
        return Err(DftError::LinearAlgebra(
            "non-finite anomalous thermal factor".into(),
        ));
    }
    Ok(value)
}

struct Evaluation<S: Scalar> {
    scaled: Vec<S>,
    physical: Vec<S>,
    number: S,
    phase: S,
    anomalous_real: Vec<S>,
    anomalous_imaginary: Vec<S>,
    occupations: Vec<S>,
    energy_squared: Vec<S>,
    kernel: Vec<S>,
    mass: Vec<S>,
}

fn evaluate<S: Scalar + Send + Sync>(
    functional: &impl GapFunctional,
    p: &[S],
    target: S,
    x: &[S],
    phase: &[f64],
    scale: f64,
) -> Result<Evaluation<S>> {
    let n = functional.bands();
    if x.len() != 2 * n + 2 || phase.len() != 2 * n {
        return Err(DftError::InvalidInput(
            "invalid gap state/phase dimensions".into(),
        ));
    }
    let data = functional.evaluate(p, x[2 * n], &x[..n], &x[n..2 * n])?;
    validate_data(&data, n)?;
    let total_weight = data
        .weights
        .iter()
        .copied()
        .fold(S::from_f64(0.0), |a, b| a + b);
    if !target.value().is_finite()
        || target.value() <= 0.0
        || target.value() >= 2.0 * total_weight.value()
    {
        return Err(DftError::InvalidInput("canonical gap electron number must lie strictly between zero and twice the total band weight".into()));
    }
    let mut anomalous_real = Vec::with_capacity(n);
    let mut anomalous_imaginary = Vec::with_capacity(n);
    let mut occupations = Vec::with_capacity(n);
    let mut energy_squared = Vec::with_capacity(n);
    let mut number = -target;
    for i in 0..n {
        let xi = data.energies[i] - x[2 * n];
        let e2 = xi * xi + x[i] * x[i] + x[n + i] * x[n + i];
        let q = anomalous_factor(e2, data.temperature_energy)?;
        anomalous_real.push(q * x[i]);
        anomalous_imaginary.push(q * x[n + i]);
        let occupation = S::from_f64(1.0) - (xi * q).scale(2.0);
        occupations.push(occupation);
        energy_squared.push(e2);
        number = number + data.weights[i] * occupation;
    }
    let mut physical = vec![S::from_f64(0.0); 2 * n];
    for i in 0..n {
        for (offset, chi) in [(0, &anomalous_real), (n, &anomalous_imaginary)] {
            let mut value = (S::from_f64(1.0) + data.mass_renormalization[i]) * x[offset + i];
            for j in 0..n {
                value = value + data.kernel[i * n + j] * data.weights[j] * chi[j];
            }
            physical[offset + i] = value;
        }
    }
    let phase_value = (0..2 * n).fold(S::from_f64(0.0), |s, i| s + x[i].scale(phase[i]));
    let mut scaled: Vec<_> = physical
        .iter()
        .enumerate()
        .map(|(i, &f)| (f + x[2 * n + 1].scale(phase[i])).scale(1.0 / scale))
        .collect();
    scaled.push(number / total_weight.scale(2.0));
    scaled.push(phase_value.scale(1.0 / scale));
    if scaled.iter().any(|v| !v.value().is_finite()) {
        return Err(DftError::LinearAlgebra(
            "non-finite coupled gap/number/phase residual".into(),
        ));
    }
    Ok(Evaluation {
        scaled,
        physical,
        number,
        phase: phase_value,
        anomalous_real,
        anomalous_imaginary,
        occupations,
        energy_squared,
        kernel: data.kernel,
        mass: data.mass_renormalization,
    })
}

struct Jacobian<'a, F> {
    functional: &'a F,
    parameters: &'a [f64],
    target: f64,
    state: &'a [f64],
    phase: &'a [f64],
    scale: f64,
}
impl<F: GapFunctional> LinearOperator for Jacobian<'_, F> {
    fn dimension(&self) -> usize {
        self.state.len()
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, direction: &[Complex]) -> Result<Vec<Complex>> {
        apply_real_linear(direction, |direction| {
            let state: Vec<_> = self
                .state
                .iter()
                .zip(direction)
                .map(|(&re, &im)| DualN { re, im: [im] })
                .collect();
            let parameters: Vec<_> = self
                .parameters
                .iter()
                .copied()
                .map(DualN::constant)
                .collect();
            Ok(evaluate(
                self.functional,
                &parameters,
                DualN::constant(self.target),
                &state,
                self.phase,
                self.scale,
            )?
            .scaled
            .into_iter()
            .map(|v| v.im[0])
            .collect())
        })
    }
}

fn largest(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, |m, x| {
        if x.is_finite() {
            m.max(x.abs())
        } else {
            f64::INFINITY
        }
    })
}

/// Solve the complex canonical gap and chemical potential, then every
/// requested Taylor coefficient with the same original AD Jacobian.
/// The explicit seed chooses the nonlinear branch and global phase; a zero
/// seed chooses the normal solution, which is not claimed to be stable.
pub fn solve_gap_taylor<const W: usize>(
    functional: &impl GapFunctional,
    parameters: &[Taylor<W>],
    electron_number: Taylor<W>,
    initial: &GapState,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<GapTaylor> {
    config.validate()?;
    let n = functional.bands();
    if n == 0
        || n.checked_mul(2).and_then(|x| x.checked_add(2)).is_none()
        || initial.gap.len() != n
        || order >= W
        || order > 4
        || !initial.chemical_potential.is_finite()
        || initial.gap.iter().any(|z| !z.is_finite())
        || parameters
            .iter()
            .chain(std::iter::once(&electron_number))
            .any(|x| x.c.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput("gap Taylor response requires finite matching seed/parameters and order zero through four within its Taylor width".into()));
    }
    let p: Vec<_> = parameters.iter().map(|x| x.value()).collect();
    let target = electron_number.value();
    let mut state: Vec<_> = initial
        .gap
        .iter()
        .map(|z| z.re)
        .chain(initial.gap.iter().map(|z| z.im))
        .chain([initial.chemical_potential, 0.0])
        .collect();
    let mut phase: Vec<_> = initial
        .gap
        .iter()
        .map(|z| -z.im)
        .chain(initial.gap.iter().map(|z| z.re))
        .collect();
    let length = norm(&phase.iter().copied().map(Complex::from).collect::<Vec<_>>());
    if length == 0.0 {
        phase[n] = 1.0;
    } else {
        for x in &mut phase {
            *x /= length;
        }
    }
    let mut mode: Vec<_> = phase.iter().copied().map(Complex::from).collect();
    mode.extend([Complex::ZERO; 2]);
    let space = ConstraintSpace::new(2 * n + 2, &[mode])?;
    state = space
        .project(&state.iter().copied().map(Complex::from).collect::<Vec<_>>())?
        .iter()
        .map(|x| x.re)
        .collect();
    let data = functional.evaluate(&p, state[2 * n], &state[..n], &state[n..2 * n])?;
    validate_data(&data, n)?;
    let scale = largest(
        data.energies
            .iter()
            .map(|x| *x - state[2 * n])
            .chain(initial.gap.iter().map(|z| z.abs()))
            .chain([data.temperature_energy]),
    );
    let mut reference_iterations = 0;
    let mut linear_residuals = Vec::new();
    let mut linear_iterations = Vec::new();
    let mut reference_residual = f64::INFINITY;
    for iteration in 0..=config.max_iterations {
        let current = evaluate(functional, &p, target, &state, &phase, scale)?;
        reference_residual = largest(current.scaled.iter().copied());
        reference_residual = reference_residual
            .max(largest(current.physical.iter().map(|x| *x / scale)))
            .max(current.number.abs())
            .max((state[2 * n + 1] / scale).abs());
        crate::logger::log(&format!("[scdft] iter {iteration:>3} mu={:.12} max|Delta|={:.6e} |gap|={:.3e} |number|={:.3e} |phase|={:.3e}",
            state[2*n],largest(state[..2*n].iter().copied()),largest(current.physical.iter().copied()),current.number.abs(),current.phase.abs()));
        reference_iterations = iteration;
        if reference_residual <= config.tolerance {
            break;
        }
        if iteration == config.max_iterations {
            return Err(DftError::ResponseNotConverged {
                iterations: iteration,
                residual: reference_residual,
            });
        }
        let op = Jacobian {
            functional,
            parameters: &p,
            target,
            state: &state,
            phase: &phase,
            scale,
        };
        let rhs: Vec<_> = current.scaled.iter().map(|x| Complex::from(-*x)).collect();
        let step = solve(&op, &rhs, None, config)?;
        linear_residuals.push(step.relative_residual);
        linear_iterations.push(step.iterations);
        let mut accepted = None;
        for trial in 0..32 {
            let alpha = 0.5_f64.powi(trial);
            let candidate: Vec<_> = state
                .iter()
                .zip(&step.amplitudes)
                .map(|(&x, d)| x + alpha * d.re)
                .collect();
            let r = evaluate(functional, &p, target, &candidate, &phase, scale)?;
            let merit = largest(r.scaled.iter().copied())
                .max(largest(r.physical.iter().map(|x| *x / scale)))
                .max(r.number.abs())
                .max((candidate[2 * n + 1] / scale).abs());
            if merit < (1.0 - 1e-4 * alpha) * reference_residual || merit <= config.tolerance {
                accepted = Some(candidate);
                break;
            }
        }
        state = accepted.ok_or(DftError::ResponseNotConverged {
            iterations: iteration + 1,
            residual: reference_residual,
        })?;
    }
    let mut jets: Vec<_> = state.iter().copied().map(Taylor::<W>::from_f64).collect();
    let op = Jacobian {
        functional,
        parameters: &p,
        target,
        state: &state,
        phase: &phase,
        scale,
    };
    for k in 1..=order {
        let r = evaluate(
            functional,
            parameters,
            electron_number,
            &jets,
            &phase,
            scale,
        )?;
        let rhs: Vec<_> = r.scaled.iter().map(|x| Complex::from(-x.c[k])).collect();
        let solved = solve(&op, &rhs, None, config)?;
        for (x, dx) in jets.iter_mut().zip(solved.amplitudes) {
            x.c[k] = dx.re;
        }
        linear_residuals.push(solved.relative_residual);
        linear_iterations.push(solved.iterations);
    }
    let raw = evaluate(
        functional,
        parameters,
        electron_number,
        &jets,
        &phase,
        scale,
    )?;
    let lift = |x: Taylor<W>| DualN::<Taylor<W>, 1>::constant(x);
    let mut rotated: Vec<_> = jets.iter().copied().map(lift).collect();
    for i in 0..n {
        rotated[i].im[0] = -jets[n + i];
        rotated[n + i].im[0] = jets[i];
    }
    let goldstone = evaluate(
        functional,
        &parameters.iter().copied().map(lift).collect::<Vec<_>>(),
        lift(electron_number),
        &rotated,
        &phase,
        scale,
    )?;
    let coefficients = |values: &[Taylor<W>]| -> Vec<Vec<f64>> {
        (0..=order)
            .map(|k| values.iter().map(|x| x.c[k]).collect())
            .collect()
    };
    let gap_residuals: Vec<_> = (0..=order)
        .map(|k| largest(raw.physical.iter().map(|x| x.c[k])))
        .collect();
    let number_residuals: Vec<_> = (0..=order).map(|k| raw.number.c[k].abs()).collect();
    let phase_residuals: Vec<_> = (0..=order).map(|k| raw.phase.c[k].abs()).collect();
    let multipliers = jets[2 * n + 1].c[..=order].to_vec();
    let goldstone_defects: Vec<_> = (0..=order)
        .map(|k| largest(goldstone.physical.iter().map(|x| x.im[0].c[k])))
        .collect();
    let mut kernel_scale = 0.0_f64;
    let mut kernel_asymmetry = Vec::with_capacity(order + 1);
    for k in 0..=order {
        kernel_scale = kernel_scale.max(largest(raw.kernel.iter().map(|x| x.c[k])));
        let asymmetry = largest(
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| raw.kernel[i * n + j].c[k] - raw.kernel[j * n + i].c[k]),
        );
        if !kernel_scale.is_finite() || !asymmetry.is_finite() || asymmetry > 1e-10 * kernel_scale {
            return Err(DftError::InvalidInput("non-finite or nonreciprocal superconducting kernel Taylor path; no coefficients are symmetrized".into()));
        }
        kernel_asymmetry.push(asymmetry);
    }
    let residual = largest(
        gap_residuals
            .iter()
            .chain(&phase_residuals)
            .chain(&multipliers)
            .chain(&goldstone_defects)
            .map(|x| *x / scale),
    )
    .max(largest(number_residuals.iter().copied()));
    if !residual.is_finite() || residual > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: reference_iterations,
            residual,
        });
    }
    if jets
        .iter()
        .chain(&raw.anomalous_real)
        .chain(&raw.anomalous_imaginary)
        .chain(&raw.occupations)
        .chain(&raw.energy_squared)
        .chain(&raw.physical)
        .chain(&raw.scaled)
        .chain(&raw.kernel)
        .chain(&raw.mass)
        .any(|x| x.c[..=order].iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::LinearAlgebra(
            "gap response coefficient overflow".into(),
        ));
    }
    Ok(GapTaylor {
        gap_real: coefficients(&jets[..n]),
        gap_imaginary: coefficients(&jets[n..2 * n]),
        chemical_potential: jets[2 * n].c[..=order].to_vec(),
        anomalous_real: coefficients(&raw.anomalous_real),
        anomalous_imaginary: coefficients(&raw.anomalous_imaginary),
        occupations: coefficients(&raw.occupations),
        quasiparticle_energies: raw
            .energy_squared
            .iter()
            .map(|x| x.value().sqrt())
            .collect(),
        quasiparticle_energy_squared: coefficients(&raw.energy_squared),
        kernel_coefficients: (0..=order)
            .map(|k| {
                raw.kernel
                    .chunks(n)
                    .map(|r| r.iter().map(|x| x.c[k]).collect())
                    .collect()
            })
            .collect(),
        kernel_asymmetry,
        mass_renormalization_coefficients: coefficients(&raw.mass),
        gap_residuals,
        number_residuals,
        phase_residuals,
        phase_multipliers: multipliers,
        goldstone_defects,
        linear_residuals,
        linear_iterations,
        reference_iterations,
        reference_scaled_residual: reference_residual,
        energy_scale: scale,
        phase_reference: (0..n)
            .map(|i| Complex::new(phase[n + i], -phase[i]))
            .collect(),
    })
}
