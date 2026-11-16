// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Rectilinear vibrational perturbation theory and transformed observables.
//! Cubic interactions enter through second order, quartic interactions through
//! first order. The historical method retains a variational resonance polyad;
//! strict VPT2 uses normalized nondegenerate wavefunctions and rejects coupled
//! resonances. Neither method floors denominators.

use crate::constants::{AMU_TO_AU_MASS, BOHR_TO_ANGSTROM, HARTREE_TO_CM1};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::{DftError, Result};
use std::collections::{BTreeMap, BTreeSet};

pub mod molecular;
pub mod rotation;
mod strict;
use rotation::{kinetic_action, WatsonRotation};

/// Treatment of anharmonic mixing beyond the harmonic reference.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AnharmonicMethod {
    /// Historical contact-transformed retained-state diagonalization.
    #[default]
    RetainedPolyad,
    /// Nonresonant Rayleigh--Schrodinger expansion through second order.
    StrictVpt2,
}
impl AnharmonicMethod {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "retained-polyad" => Ok(Self::RetainedPolyad),
            "strict-vpt2" => Ok(Self::StrictVpt2),
            _ => Err(DftError::InvalidInput(format!(
                "unknown anharmonic method `{value}` (retained-polyad or strict-vpt2)"
            ))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RetainedPolyad => "retained-polyad",
            Self::StrictVpt2 => "strict-vpt2",
        }
    }
    pub fn energy_treatment(self) -> &'static str {
        match self {
            Self::RetainedPolyad => "retained-state diagonalization of a second-order contact Hamiltonian; includes higher-order internal mixing",
            Self::StrictVpt2 => "nondegenerate Rayleigh-Schrodinger energies through second perturbative order; no retained-state resummation",
        }
    }
    pub fn intensity_treatment(self) -> &'static str {
        match self {
            Self::RetainedPolyad => {
                "squared contact-transformed, retained-state-mixed transition moments"
            }
            Self::StrictVpt2 => {
                "squared second-order transition moments; not order-truncated intensity series"
            }
        }
    }
    pub fn resonance_treatment(self) -> &'static str {
        match self {
            Self::RetainedPolyad => "coupled resonances retained and diagonalized; no denominator regularization",
            Self::StrictVpt2 => "coupled cubic and effective second-order near-resonances rejected; no denominator regularization or fallback",
        }
    }
    pub fn eigensystem_kind(self) -> &'static str {
        match self {
            Self::RetainedPolyad => "retained-polyad-eigenvectors",
            Self::StrictVpt2 => "diagonal-perturbative-energies",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnharmonicConfig {
    pub method: AnharmonicMethod,
    pub resonance_tolerance: f64,
    pub resonance_coupling_factor: f64,
    pub max_basis_states: usize,
    pub max_polyad_states: usize,
    pub tolerance: f64,
    pub memory_mb: usize,
    pub equilibrium_tolerance: f64,
}
impl Default for AnharmonicConfig {
    fn default() -> Self {
        Self {
            method: AnharmonicMethod::RetainedPolyad,
            resonance_tolerance: 1e-8,
            resonance_coupling_factor: 1.0,
            max_basis_states: 1024,
            max_polyad_states: 256,
            tolerance: 1e-10,
            memory_mb: 512,
            equilibrium_tolerance: 1e-4,
        }
    }
}
impl AnharmonicConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.resonance_tolerance.is_finite()
            || self.resonance_tolerance < 0.0
            || !self.resonance_coupling_factor.is_finite()
            || self.resonance_coupling_factor < 0.0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || !self.equilibrium_tolerance.is_finite()
            || self.equilibrium_tolerance <= 0.0
            || self.max_polyad_states == 0
            || self.max_basis_states < self.max_polyad_states
            || self.memory_mb == 0
        {
            return Err(DftError::InvalidInput(
                "invalid anharmonic resonance, convergence, equilibrium or storage controls".into(),
            ));
        }
        Ok(())
    }
}

/// Actual first/second/third derivatives; component index varies fastest.
#[derive(Clone, Debug)]
pub struct ObservablePolynomial {
    pub components: usize,
    pub first: Vec<f64>,
    pub second: Vec<f64>,
    pub third: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct VibrationalPolynomial {
    /// Positive harmonic angular frequencies in Hartree, masses in electron units.
    pub frequencies: Vec<f64>,
    pub cubic: Vec<f64>,
    pub quartic: Vec<f64>,
    pub dipole: ObservablePolynomial,
    pub polarizability: Option<ObservablePolynomial>,
    pub rotation: Option<WatsonRotation>,
}
#[derive(Clone, Debug)]
pub struct AnharmonicBand {
    pub dominant_occupation: Vec<usize>,
    pub character: &'static str,
    pub dominant_weight: f64,
    pub frequency_cm: f64,
    pub harmonic_frequency_cm: f64,
    pub transition_dipole: [f64; 3],
    pub double_harmonic_dipole: [f64; 3],
    pub mechanical_dipole: [f64; 3],
    pub electrical_dipole: [f64; 3],
    pub mixed_dipole: [f64; 3],
    pub ir_intensity: f64,
    pub double_harmonic_ir_intensity: f64,
    pub transition_polarizability: Option<[f64; 9]>,
    pub raman_activity: Option<f64>,
    pub double_harmonic_raman_activity: Option<f64>,
}
#[derive(Clone, Debug)]
pub struct AnharmonicSpectrum {
    pub method: AnharmonicMethod,
    pub bands: Vec<AnharmonicBand>,
    pub zero_point_energy: f64,
    pub harmonic_zero_point_energy: f64,
    pub polyad_occupations: Vec<Vec<usize>>,
    pub polyad_energies: Vec<f64>,
    pub polyad_vectors: Matrix,
    pub effective_hamiltonian: Matrix,
    pub basis_states: usize,
    pub added_resonant_states: usize,
    pub minimum_external_denominator: Option<f64>,
    pub exterior_mixing_norm: f64,
    pub hermiticity_defect: f64,
    pub eigen_residual: f64,
    pub converged: bool,
    pub warnings: Vec<String>,
}

#[derive(Clone)]
struct Term {
    indices: Vec<usize>,
    coefficient: f64,
}
fn decode(mut flat: usize, m: usize, rank: usize) -> Vec<usize> {
    let mut out = vec![0; rank];
    for i in (0..rank).rev() {
        out[i] = flat % m;
        flat /= m;
    }
    out
}
fn terms(values: &[f64], m: usize, rank: usize, config: &AnharmonicConfig) -> Result<Vec<Term>> {
    if values.len()
        != m.checked_pow(rank as u32)
            .ok_or_else(|| DftError::InvalidInput("vibrational tensor size overflow".into()))?
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "vibrational derivative tensor has an invalid shape or nonfinite coefficient".into(),
        ));
    }
    let mut grouped = BTreeMap::<Vec<usize>, (f64, usize, f64, f64)>::new();
    for (flat, &value) in values.iter().enumerate() {
        let mut tuple = decode(flat, m, rank);
        tuple.sort_unstable();
        let entry = grouped.entry(tuple).or_insert((0.0, 0, value, value));
        entry.0 += value;
        entry.1 += 1;
        entry.2 = entry.2.min(value);
        entry.3 = entry.3.max(value);
    }
    let factorial = (1..=rank).product::<usize>() as f64;
    let scale = values.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let mut result = Vec::new();
    for (indices, (sum, _, min, max)) in grouped {
        if max - min > config.tolerance * scale {
            return Err(DftError::InvalidInput(
                "raw vibrational derivative tensor violates index permutation symmetry".into(),
            ));
        }
        if sum != 0.0 {
            result.push(Term {
                indices,
                coefficient: sum / factorial,
            });
        }
    }
    Ok(result)
}
fn action(terms: &[Term], state: &[usize], omega: &[f64]) -> BTreeMap<Vec<usize>, f64> {
    let mut result = BTreeMap::new();
    for term in terms {
        let mut paths = BTreeMap::from([(state.to_vec(), term.coefficient)]);
        for &mode in &term.indices {
            let mut next = BTreeMap::new();
            for (occupation, value) in paths {
                let n = occupation[mode];
                let mut up = occupation.clone();
                up[mode] += 1;
                *next.entry(up).or_insert(0.0) +=
                    value * ((n + 1) as f64 / (2.0 * omega[mode])).sqrt();
                if n > 0 {
                    let mut down = occupation;
                    down[mode] -= 1;
                    *next.entry(down).or_insert(0.0) +=
                        value * (n as f64 / (2.0 * omega[mode])).sqrt();
                }
            }
            paths = next;
        }
        for (state, value) in paths {
            *result.entry(state).or_insert(0.0) += value;
        }
    }
    result.retain(|_, v| *v != 0.0);
    result
}
fn energy(state: &[usize], omega: &[f64]) -> f64 {
    state
        .iter()
        .zip(omega)
        .map(|(&n, &w)| (n as f64 + 0.5) * w)
        .sum()
}
fn matrix(terms: &[Term], states: &[Vec<usize>], omega: &[f64]) -> Matrix {
    let index: BTreeMap<_, _> = states
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    let mut result = Matrix::zeros(states.len(), states.len());
    for (j, state) in states.iter().enumerate() {
        for (to, v) in action(terms, state, omega) {
            if let Some(&i) = index.get(&to) {
                result[(i, j)] = v;
            }
        }
    }
    result
}
fn second_order_action(
    quartic: &[Term],
    rotation: Option<&WatsonRotation>,
    state: &[usize],
    omega: &[f64],
) -> BTreeMap<Vec<usize>, f64> {
    let mut result = action(quartic, state, omega);
    if let Some(rotation) = rotation {
        for (to, value) in kinetic_action(rotation, state, omega) {
            *result.entry(to).or_insert(0.0) += value;
        }
    }
    result.retain(|_, v| *v != 0.0);
    result
}
fn second_order_matrix(
    quartic: &[Term],
    rotation: Option<&WatsonRotation>,
    states: &[Vec<usize>],
    omega: &[f64],
) -> Matrix {
    let index: BTreeMap<_, _> = states
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    let mut result = Matrix::zeros(states.len(), states.len());
    for (j, state) in states.iter().enumerate() {
        for (to, value) in second_order_action(quartic, rotation, state, omega) {
            if let Some(&i) = index.get(&to) {
                result[(i, j)] = value;
            }
        }
    }
    result
}
fn commutator(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = a.matmul_seq(b);
    let reverse = b.matmul_seq(a);
    for (v, &r) in c.as_mut_slice().iter_mut().zip(reverse.as_slice()) {
        *v -= r;
    }
    c
}
fn add(a: &mut Matrix, b: &Matrix, factor: f64) {
    for (v, &w) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *v += factor * w;
    }
}
fn defect(a: &Matrix) -> f64 {
    (0..a.rows)
        .flat_map(|i| (0..a.cols).map(move |j| (a[(i, j)] - a[(j, i)]).abs()))
        .fold(0.0, f64::max)
}
fn block(a: &Matrix, p: usize) -> Matrix {
    let mut out = Matrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            out[(i, j)] = a[(i, j)];
        }
    }
    out
}

fn observable_terms(
    observable: &ObservablePolynomial,
    m: usize,
    config: &AnharmonicConfig,
) -> Result<Vec<[Vec<Term>; 3]>> {
    let mut result = Vec::new();
    for (rank, values) in [
        (1, &observable.first),
        (2, &observable.second),
        (3, &observable.third),
    ] {
        if values.len() != m.pow(rank as u32) * observable.components {
            return Err(DftError::InvalidInput(
                "observable derivative shape does not match vibrational modes and components"
                    .into(),
            ));
        }
    }
    for c in 0..observable.components {
        let mut ranks = [Vec::new(), Vec::new(), Vec::new()];
        for (i, values) in [&observable.first, &observable.second, &observable.third]
            .iter()
            .enumerate()
        {
            let data: Vec<_> = values
                .iter()
                .skip(c)
                .step_by(observable.components)
                .copied()
                .collect();
            ranks[i] = terms(&data, m, i + 1, config)?;
        }
        result.push(ranks);
    }
    Ok(result)
}

/// Canonical contact transformation of a supplied rectilinear vibrational model.
/// Fundamental, overtone and combination states initiate a resonance-complete
/// polyad. Oscillator intermediates are generated by analytic ladder actions.
pub fn vibrational_spectrum(
    model: &VibrationalPolynomial,
    config: &AnharmonicConfig,
) -> Result<AnharmonicSpectrum> {
    config.validate()?;
    let m = model.frequencies.len();
    if m == 0
        || model
            .frequencies
            .iter()
            .any(|w| !w.is_finite() || *w <= 0.0)
        || model.dipole.components != 3
        || model
            .polarizability
            .as_ref()
            .is_some_and(|o| o.components != 9)
    {
        return Err(DftError::InvalidInput(
            "anharmonic analysis requires positive finite frequencies, three dipole \
            and nine optional polarizability components"
                .into(),
        ));
    }
    let cubic = terms(&model.cubic, m, 3, config)?;
    let quartic = terms(&model.quartic, m, 4, config)?;
    if let Some(rotation) = &model.rotation {
        rotation.validate(m, config.tolerance)?;
    }
    let dipoles = observable_terms(&model.dipole, m, config)?;
    let polarizabilities = model
        .polarizability
        .as_ref()
        .map(|o| observable_terms(o, m, config))
        .transpose()?;
    if config.method == AnharmonicMethod::StrictVpt2 {
        return strict::spectrum(
            model,
            config,
            &cubic,
            &quartic,
            &dipoles,
            polarizabilities.as_deref(),
        );
    }
    let mut retained = BTreeSet::from([vec![0; m]]);
    for i in 0..m {
        let mut n = vec![0; m];
        n[i] = 1;
        retained.insert(n.clone());
        n[i] = 2;
        retained.insert(n);
        for j in 0..i {
            let mut n = vec![0; m];
            n[i] = 1;
            n[j] = 1;
            retained.insert(n);
        }
    }
    let initial = retained.len();
    loop {
        if retained.len() > config.max_polyad_states {
            return Err(DftError::InvalidInput(
                "resonance polyad exceeds max_polyad_states; enlarge the explicit limit".into(),
            ));
        }
        let mut additions = BTreeSet::new();
        for state in &retained {
            let base = energy(state, &model.frequencies);
            let a3 = action(&cubic, state, &model.frequencies);
            let a4 =
                second_order_action(&quartic, model.rotation.as_ref(), state, &model.frequencies);
            for (to, v) in a3.iter().chain(a4.iter()) {
                if !retained.contains(to)
                    && (base - energy(to, &model.frequencies)).abs()
                        <= config.resonance_tolerance + config.resonance_coupling_factor * v.abs()
                {
                    additions.insert(to.clone());
                }
            }
        }
        if !additions.is_empty() {
            retained.extend(additions);
            continue;
        }
        // A second-order cubic path can connect a degenerate state even
        // when its direct cubic/quartic matrix elements vanish.
        for state in &retained {
            let base = energy(state, &model.frequencies);
            let mut indirect = BTreeMap::<Vec<usize>, f64>::new();
            for (middle, first) in action(&cubic, state, &model.frequencies) {
                let mid_energy = energy(&middle, &model.frequencies);
                for (to, second) in action(&cubic, &middle, &model.frequencies) {
                    if retained.contains(&to) {
                        continue;
                    }
                    let denominator = if retained.contains(&middle) {
                        mid_energy - energy(&to, &model.frequencies)
                    } else {
                        base - mid_energy
                    };
                    if denominator == 0.0 {
                        return Err(DftError::InvalidInput("direct resonance closure failed before the second-order polyad construction".into()));
                    }
                    let sign = if retained.contains(&middle) {
                        -1.0
                    } else {
                        1.0
                    };
                    *indirect.entry(to).or_insert(0.0) += sign * first * second / denominator;
                }
            }
            for (to, value) in indirect {
                if value != 0.0
                    && (base - energy(&to, &model.frequencies)).abs()
                        <= config.resonance_tolerance
                            + config.resonance_coupling_factor * value.abs()
                {
                    additions.insert(to);
                }
            }
        }
        if additions.is_empty() {
            break;
        }
        retained.extend(additions);
    }
    let mut outside = BTreeSet::new();
    for state in &retained {
        if let Some(rotation) = &model.rotation {
            outside.extend(kinetic_action(rotation, state, &model.frequencies).into_keys());
        }
        for polynomial in std::iter::once(&cubic)
            .chain(std::iter::once(&quartic))
            .chain(dipoles.iter().flat_map(|r| r.iter()))
            .chain(
                polarizabilities
                    .iter()
                    .flat_map(|v| v.iter().flat_map(|r| r.iter())),
            )
        {
            outside.extend(action(polynomial, state, &model.frequencies).into_keys());
        }
        if outside.len() > config.max_basis_states {
            return Err(DftError::InvalidInput(
                "oscillator intermediates exceed max_basis_states".into(),
            ));
        }
    }
    for state in &retained {
        outside.remove(state);
    }
    let p = retained.len();
    let states: Vec<_> = retained.into_iter().chain(outside).collect();
    let n = states.len();
    // Account for full intermediate matrices and cached component polyad blocks.
    let matrix_storage = n
        .checked_mul(n)
        .and_then(|s| s.checked_mul(20))
        .and_then(|intermediates| {
            p.checked_mul(p)
                .and_then(|s| s.checked_mul(if polarizabilities.is_some() { 56 } else { 20 }))
                .and_then(|blocks| intermediates.checked_add(blocks))
        })
        .and_then(|elements| elements.checked_mul(8));
    if n > config.max_basis_states
        || matrix_storage.is_none_or(|bytes| bytes > config.memory_mb.saturating_mul(1024 * 1024))
    {
        return Err(DftError::InvalidInput(
            "anharmonic contact-transformation storage exceeds the explicit resource limits".into(),
        ));
    }
    let energies: Vec<_> = states
        .iter()
        .map(|s| energy(s, &model.frequencies))
        .collect();
    let v3 = matrix(&cubic, &states, &model.frequencies);
    let v4 = second_order_matrix(
        &quartic,
        model.rotation.as_ref(),
        &states,
        &model.frequencies,
    );
    let mut raw_defect = defect(&v3).max(defect(&v4));
    let mut s1 = Matrix::zeros(n, n);
    let mut minimum: Option<f64> = None;
    for a in 0..p {
        for k in p..n {
            let delta = energies[a] - energies[k];
            if v3[(a, k)] != 0.0 || v4[(a, k)] != 0.0 {
                minimum = Some(minimum.map_or(delta.abs(), |v| v.min(delta.abs())));
            }
            if v3[(a, k)] != 0.0 {
                if delta == 0.0 {
                    return Err(DftError::InvalidInput(
                        "an exactly resonant cubic intermediate was not retained".into(),
                    ));
                }
                s1[(a, k)] = -v3[(a, k)] / delta;
                s1[(k, a)] = -s1[(a, k)];
            }
        }
    }
    let comm = commutator(&v3, &s1);
    let mut heff = block(&v3, p);
    add(&mut heff, &block(&v4, p), 1.0);
    add(&mut heff, &block(&comm, p), 0.5);
    for a in 0..p {
        heff[(a, a)] += energies[a];
    }
    raw_defect = raw_defect.max(defect(&heff));
    let mut s2 = Matrix::zeros(n, n);
    for a in 0..p {
        for k in p..n {
            let value = v4[(a, k)] + comm[(a, k)];
            if value != 0.0 {
                let delta = energies[a] - energies[k];
                minimum = Some(minimum.map_or(delta.abs(), |v| v.min(delta.abs())));
                if delta == 0.0 {
                    return Err(DftError::InvalidInput(
                        "an exactly resonant second-order intermediate was not retained".into(),
                    ));
                }
                s2[(a, k)] = -value / delta;
                s2[(k, a)] = -s2[(a, k)];
            }
        }
    }
    let (evals, mut vectors) = symmetric_eigen(&heff)?;
    for col in 0..p {
        let dominant = (0..p)
            .max_by(|&a, &b| vectors[(a, col)].abs().total_cmp(&vectors[(b, col)].abs()))
            .unwrap();
        if vectors[(dominant, col)] < 0.0 {
            for row in 0..p {
                vectors[(row, col)] = -vectors[(row, col)];
            }
        }
    }
    let mut eigen_residual = 0.0_f64;
    let hvec = heff.matmul_seq(&vectors);
    for i in 0..p {
        for j in 0..p {
            eigen_residual = eigen_residual.max((hvec[(i, j)] - evals[j] * vectors[(i, j)]).abs());
        }
    }
    let mut transform = |ranks: &[Vec<Term>; 3]| -> [Matrix; 4] {
        let linear = matrix(&ranks[0], &states, &model.frequencies);
        let quadratic = matrix(&ranks[1], &states, &model.frequencies);
        let third = matrix(&ranks[2], &states, &model.frequencies);
        let first = commutator(&linear, &s1);
        let mut mechanical = first.clone();
        add(&mut mechanical, &commutator(&linear, &s2), 1.0);
        add(&mut mechanical, &commutator(&first, &s1), 0.5);
        let mut electrical = quadratic.clone();
        add(&mut electrical, &third, 1.0);
        let mixed = commutator(&quadratic, &s1);
        let mut blocks = [
            block(&linear, p),
            block(&mechanical, p),
            block(&electrical, p),
            block(&mixed, p),
        ];
        for b in &mut blocks {
            raw_defect = raw_defect.max(defect(b));
            *b = vectors.transpose().matmul_seq(b).matmul_seq(&vectors);
        }
        blocks
    };
    let mu: Vec<_> = dipoles.iter().map(&mut transform).collect();
    let alpha: Option<Vec<_>> = polarizabilities
        .as_ref()
        .map(|v| v.iter().map(&mut transform).collect());
    let vacuum = vec![0; m];
    let ground_basis = states.iter().position(|s| s == &vacuum).unwrap();
    let mut bands = Vec::new();
    for root in 1..p {
        let dominant = (0..p)
            .max_by(|&a, &b| {
                vectors[(a, root)]
                    .abs()
                    .total_cmp(&vectors[(b, root)].abs())
            })
            .unwrap();
        let occupation = states[dominant].clone();
        let quanta: usize = occupation.iter().sum();
        let bare_dipole = std::array::from_fn(|c| {
            matrix(&dipoles[c][0], &states, &model.frequencies)[(ground_basis, dominant)]
        });
        let mechanical =
            std::array::from_fn(|c| mu[c][0][(0, root)] - bare_dipole[c] + mu[c][1][(0, root)]);
        let electrical = std::array::from_fn(|c| mu[c][2][(0, root)]);
        let mixed = std::array::from_fn(|c| mu[c][3][(0, root)]);
        let transition =
            std::array::from_fn(|c| bare_dipole[c] + mechanical[c] + electrical[c] + mixed[c]);
        let frequency = evals[root] - evals[0];
        let harmonic = energies[dominant] - energies[ground_basis];
        let ir =
            1949.7602 * AMU_TO_AU_MASS * frequency * transition.iter().map(|v| v * v).sum::<f64>();
        let harmonic_ir =
            1949.7602 * AMU_TO_AU_MASS * harmonic * bare_dipole.iter().map(|v| v * v).sum::<f64>();
        let transition_alpha: Option<[f64; 9]> = alpha
            .as_ref()
            .map(|v| std::array::from_fn(|c| v[c].iter().map(|b| b[(0, root)]).sum()));
        let bare_alpha: Option<[f64; 9]> = polarizabilities.as_ref().map(|v| {
            std::array::from_fn(|c| {
                matrix(&v[c][0], &states, &model.frequencies)[(ground_basis, dominant)]
            })
        });
        bands.push(AnharmonicBand {
            dominant_occupation: occupation,
            character: match quanta {
                1 => "fundamental",
                2 => {
                    if states[dominant].contains(&2) {
                        "overtone"
                    } else {
                        "combination"
                    }
                }
                _ => "resonant_polyad",
            },
            dominant_weight: vectors[(dominant, root)].powi(2),
            frequency_cm: frequency * HARTREE_TO_CM1,
            harmonic_frequency_cm: harmonic * HARTREE_TO_CM1,
            transition_dipole: transition,
            double_harmonic_dipole: bare_dipole,
            mechanical_dipole: mechanical,
            electrical_dipole: electrical,
            mixed_dipole: mixed,
            ir_intensity: ir,
            double_harmonic_ir_intensity: harmonic_ir,
            transition_polarizability: transition_alpha,
            raman_activity: transition_alpha.map(|a| raman_activity(a, frequency)),
            double_harmonic_raman_activity: bare_alpha.map(|a| raman_activity(a, harmonic)),
        });
    }
    let exterior_mixing_norm = (0..p)
        .map(|a| {
            (p..n)
                .map(|k| (s1[(a, k)] + s2[(a, k)]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    let scale = heff.as_slice().iter().fold(1.0_f64, |v, h| v.max(h.abs()));
    let mut warnings = Vec::new();
    if model.rotation.is_none() {
        warnings.push(
            "supplied rectilinear vibrational Hamiltonian has no rotational kinetic coefficients"
                .into(),
        );
    }
    if model
        .rotation
        .as_ref()
        .is_some_and(|r| r.class == rotation::RotorClass::Linear)
        && m > 1
    {
        warnings.push("linear-rotor vibrational band origins: physical bending rovibrational states require Sayvetz K=abs(l) and J>=abs(l)".into());
    }
    if exterior_mixing_norm > 0.3 {
        warnings.push(
            "large exterior perturbative mixing; truncation needs an independent convergence check"
                .into(),
        );
    }
    Ok(AnharmonicSpectrum {
        method: AnharmonicMethod::RetainedPolyad,
        bands,
        zero_point_energy: evals[0],
        harmonic_zero_point_energy: energies[ground_basis],
        polyad_occupations: states[..p].to_vec(),
        polyad_energies: evals,
        polyad_vectors: vectors,
        effective_hamiltonian: heff,
        basis_states: n,
        added_resonant_states: p - initial,
        minimum_external_denominator: minimum,
        exterior_mixing_norm,
        hermiticity_defect: raw_defect,
        eigen_residual,
        converged: raw_defect <= config.tolerance * scale
            && eigen_residual <= config.tolerance * scale,
        warnings,
    })
}
fn raman_activity(a: [f64; 9], frequency: f64) -> f64 {
    let iso = (a[0] + a[4] + a[8]) / 3.0;
    let anis = 0.5 * ((a[0] - a[4]).powi(2) + (a[4] - a[8]).powi(2) + (a[8] - a[0]).powi(2))
        + 0.75 * ((a[1] + a[3]).powi(2) + (a[2] + a[6]).powi(2) + (a[5] + a[7]).powi(2));
    2.0 * frequency * AMU_TO_AU_MASS * BOHR_TO_ANGSTROM.powi(4) * (45.0 * iso * iso + 7.0 * anis)
}
