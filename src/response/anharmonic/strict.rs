// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nonresonant, normalized Rayleigh--Schrodinger VPT2. No retained-state
//! diagonalization occurs: V3 has order one, V4/rotation order two, and the
//! linear/quadratic/cubic observable ranks have orders zero/one/two.

use super::*;

type Amplitudes = BTreeMap<Vec<usize>, f64>;

struct Wave {
    state: Vec<usize>,
    first: Amplitudes,
    second: Amplitudes,
    energy: f64,
    first_norm: f64,
}

/// Conservative sparse-storage accounting, including retained output matrices,
/// all cached wavefunctions and temporary action maps. Check before caching.
struct Storage<'a> {
    states: BTreeSet<Vec<usize>>,
    retained: usize,
    modes: usize,
    config: &'a AnharmonicConfig,
}
impl Storage<'_> {
    fn observe<'a>(&mut self, states: impl IntoIterator<Item = &'a Vec<usize>>) -> Result<()> {
        for state in states {
            self.states.insert(state.clone());
            let bytes = self
                .modes
                .checked_add(16)
                .and_then(|v| v.checked_mul(std::mem::size_of::<usize>()))
                .and_then(|v| v.checked_mul(self.states.len()))
                .and_then(|v| v.checked_mul(self.retained.checked_add(4)?))
                .and_then(|v| v.checked_mul(12))
                .and_then(|v| {
                    self.retained
                        .checked_mul(self.retained)
                        .and_then(|p| p.checked_mul(16))
                        .and_then(|p| p.checked_add(v))
                });
            if self.states.len() > self.config.max_basis_states
                || bytes.is_none_or(|v| v > self.config.memory_mb.saturating_mul(1024 * 1024))
            {
                return Err(DftError::InvalidInput(
                    "strict-vpt2 oscillator intermediates exceed explicit state or memory limits"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

fn resonance_error(state: &[usize], to: &[usize], order: &str) -> DftError {
    DftError::InvalidInput(format!(
        "strict-vpt2 coupled {order} resonance between {state:?} and {to:?}; \
         nondegenerate perturbation theory is undefined or unsafe here; \
         use retained-polyad explicitly for a resonant model"
    ))
}

fn denominator(
    state: &[usize],
    to: &[usize],
    omega: &[f64],
    coupling: f64,
    order: &str,
    config: &AnharmonicConfig,
    minimum: &mut Option<f64>,
) -> Result<f64> {
    let delta = energy(state, omega) - energy(to, omega);
    *minimum = Some(minimum.map_or(delta.abs(), |old| old.min(delta.abs())));
    if delta.abs() <= config.resonance_tolerance + config.resonance_coupling_factor * coupling.abs()
    {
        return Err(resonance_error(state, to, order));
    }
    Ok(delta)
}

fn wave(
    state: &[usize],
    model: &VibrationalPolynomial,
    config: &AnharmonicConfig,
    cubic: &[Term],
    quartic: &[Term],
    storage: &mut Storage<'_>,
    minimum: &mut Option<f64>,
) -> Result<Wave> {
    let omega = &model.frequencies;
    let a3 = action(cubic, state, omega);
    let mut numerator = second_order_action(quartic, model.rotation.as_ref(), state, omega);
    storage.observe(a3.keys().chain(numerator.keys()))?;
    let diagonal_first = a3.get(state).copied().unwrap_or(0.0);
    let mut corrected_energy =
        energy(state, omega) + diagonal_first + numerator.get(state).copied().unwrap_or(0.0);
    let mut first = Amplitudes::new();
    for (middle, &value) in &a3 {
        if middle == state || value == 0.0 {
            continue;
        }
        let delta = denominator(state, middle, omega, value, "cubic", config, minimum)?;
        first.insert(middle.clone(), value / delta);
        corrected_energy += value * value / delta;
    }
    // c2_k = [W_kn + sum_m V_km V_mn/Delta_nm
    //                   - V_nn V_kn/Delta_nk] / Delta_nk.
    // The diagonal c2_n fixes normalization, not the energy.
    for (middle, &amplitude) in &first {
        let next = action(cubic, middle, omega);
        storage.observe(next.keys())?;
        for (to, value) in next {
            *numerator.entry(to).or_insert(0.0) += value * amplitude;
        }
        *numerator.entry(middle.clone()).or_insert(0.0) -= diagonal_first * amplitude;
    }
    let first_norm = first.values().map(|v| v * v).sum::<f64>();
    let mut second = Amplitudes::from([(state.to_vec(), -0.5 * first_norm)]);
    for (to, value) in numerator {
        if to == state || value == 0.0 {
            continue;
        }
        let delta = denominator(state, &to, omega, value, "second-order", config, minimum)?;
        second.insert(to, value / delta);
    }
    storage.observe(first.keys().chain(second.keys()))?;
    if !corrected_energy.is_finite()
        || first
            .values()
            .chain(second.values())
            .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "strict-vpt2 perturbative coefficients are nonfinite".into(),
        ));
    }
    Ok(Wave {
        state: state.to_vec(),
        first,
        second,
        energy: corrected_energy,
        first_norm,
    })
}

fn element(
    left: &Amplitudes,
    operator: &[Term],
    right: &Amplitudes,
    omega: &[f64],
    storage: &mut Storage<'_>,
) -> Result<f64> {
    let mut value = 0.0;
    for (state, coefficient) in right {
        let applied = action(operator, state, omega);
        storage.observe(applied.keys())?;
        for (to, amplitude) in applied {
            value += coefficient * amplitude * left.get(&to).copied().unwrap_or(0.0);
        }
    }
    Ok(value)
}

/// [double harmonic, mechanical, electrical, mixed]. Each amplitude is truncated
/// consistently; squaring their sum deliberately is not an order-truncated
/// intensity expansion and is identified as such in the public metadata.
fn moment(
    left: &Wave,
    right: &Wave,
    ranks: &[Vec<Term>; 3],
    omega: &[f64],
    storage: &mut Storage<'_>,
) -> Result<[f64; 4]> {
    let l0 = Amplitudes::from([(left.state.clone(), 1.0)]);
    let r0 = Amplitudes::from([(right.state.clone(), 1.0)]);
    let bare = element(&l0, &ranks[0], &r0, omega, storage)?;
    let mechanical = element(&left.first, &ranks[0], &r0, omega, storage)?
        + element(&l0, &ranks[0], &right.first, omega, storage)?
        + element(&left.second, &ranks[0], &r0, omega, storage)?
        + element(&l0, &ranks[0], &right.second, omega, storage)?
        + element(&left.first, &ranks[0], &right.first, omega, storage)?;
    let electrical = element(&l0, &ranks[1], &r0, omega, storage)?
        + element(&l0, &ranks[2], &r0, omega, storage)?;
    let mixed = element(&left.first, &ranks[1], &r0, omega, storage)?
        + element(&l0, &ranks[1], &right.first, omega, storage)?;
    let parts = [bare, mechanical, electrical, mixed];
    if parts.iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput(
            "strict-vpt2 transition-moment coefficients are nonfinite".into(),
        ));
    }
    Ok(parts)
}

pub(super) fn spectrum(
    model: &VibrationalPolynomial,
    config: &AnharmonicConfig,
    cubic: &[Term],
    quartic: &[Term],
    dipoles: &[[Vec<Term>; 3]],
    polarizabilities: Option<&[[Vec<Term>; 3]]>,
) -> Result<AnharmonicSpectrum> {
    let m = model.frequencies.len();
    let mut retained = vec![vec![0; m]];
    for i in 0..m {
        let mut state = vec![0; m];
        state[i] = 1;
        retained.push(state.clone());
        state[i] = 2;
        retained.push(state);
        for j in 0..i {
            let mut state = vec![0; m];
            state[i] = 1;
            state[j] = 1;
            retained.push(state);
        }
        if retained.len() > config.max_polyad_states {
            return Err(DftError::InvalidInput(
                "strict-vpt2 labelled states exceed max_polyad_states".into(),
            ));
        }
    }
    retained.sort_by(|a, b| {
        energy(a, &model.frequencies)
            .total_cmp(&energy(b, &model.frequencies))
            .then(a.cmp(b))
    });
    let p = retained.len();
    let mut storage = Storage {
        states: BTreeSet::new(),
        retained: p,
        modes: m,
        config,
    };
    storage.observe(&retained)?;
    let mut minimum = None;
    let waves = retained
        .iter()
        .map(|state| {
            wave(
                state,
                model,
                config,
                cubic,
                quartic,
                &mut storage,
                &mut minimum,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ground = &waves[0];
    let mut bands = Vec::with_capacity(p - 1);
    for excited in &waves[1..] {
        let frequency = excited.energy - ground.energy;
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(DftError::InvalidInput(
                "strict-vpt2 gives a nonpositive band origin; perturbation theory is not valid for this model".into(),
            ));
        }
        let harmonic =
            energy(&excited.state, &model.frequencies) - energy(&ground.state, &model.frequencies);
        let mut mu = [[0.0; 4]; 3];
        for c in 0..3 {
            mu[c] = moment(
                ground,
                excited,
                &dipoles[c],
                &model.frequencies,
                &mut storage,
            )?;
        }
        let bare = std::array::from_fn(|c| mu[c][0]);
        let mechanical = std::array::from_fn(|c| mu[c][1]);
        let electrical = std::array::from_fn(|c| mu[c][2]);
        let mixed = std::array::from_fn(|c| mu[c][3]);
        let transition: [f64; 3] = std::array::from_fn(|c| mu[c].iter().sum());
        let alpha = polarizabilities
            .map(|operators| {
                operators
                    .iter()
                    .map(|ranks| moment(ground, excited, ranks, &model.frequencies, &mut storage))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let transition_alpha = alpha
            .as_ref()
            .map(|a| std::array::from_fn(|c| a[c].iter().sum()));
        let bare_alpha = alpha.as_ref().map(|a| std::array::from_fn(|c| a[c][0]));
        let quanta: usize = excited.state.iter().sum();
        bands.push(AnharmonicBand {
            dominant_occupation: excited.state.clone(),
            character: if quanta == 1 {
                "fundamental"
            } else if excited.state.contains(&2) {
                "overtone"
            } else {
                "combination"
            },
            dominant_weight: 1.0,
            frequency_cm: frequency * HARTREE_TO_CM1,
            harmonic_frequency_cm: harmonic * HARTREE_TO_CM1,
            transition_dipole: transition,
            double_harmonic_dipole: bare,
            mechanical_dipole: mechanical,
            electrical_dipole: electrical,
            mixed_dipole: mixed,
            ir_intensity: 1949.7602
                * AMU_TO_AU_MASS
                * frequency
                * transition.iter().map(|v| v * v).sum::<f64>(),
            double_harmonic_ir_intensity: 1949.7602
                * AMU_TO_AU_MASS
                * harmonic
                * bare.iter().map(|v| v * v).sum::<f64>(),
            transition_polarizability: transition_alpha,
            raman_activity: transition_alpha.map(|a| raman_activity(a, frequency)),
            double_harmonic_raman_activity: bare_alpha.map(|a| raman_activity(a, harmonic)),
        });
    }
    let mixing = waves
        .iter()
        .map(|w| w.first_norm.sqrt())
        .fold(0.0, f64::max);
    let mut warnings = vec![
        "strict-vpt2 transition moments are truncated through second perturbative order; intensities square the corrected moments and are not order-truncated intensity series".into(),
        "strict-vpt2 occupation labels and unit dominant weights denote unmixed harmonic reference states, not variational probabilities; diagonal output matrices contain perturbative energies without a polyad eigensolve".into(),
    ];
    if mixing > 0.3 {
        warnings.push("large strict-vpt2 first-order wavefunction norm; physical perturbative validity requires an independent convergence check".into());
    }
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
    let mut diagonal = Matrix::zeros(p, p);
    let mut identity = Matrix::zeros(p, p);
    for (i, wave) in waves.iter().enumerate() {
        diagonal[(i, i)] = wave.energy;
        identity[(i, i)] = 1.0;
    }
    Ok(AnharmonicSpectrum {
        method: AnharmonicMethod::StrictVpt2,
        bands,
        zero_point_energy: ground.energy,
        harmonic_zero_point_energy: energy(&ground.state, &model.frequencies),
        polyad_occupations: retained,
        polyad_energies: waves.iter().map(|w| w.energy).collect(),
        polyad_vectors: identity,
        effective_hamiltonian: diagonal,
        basis_states: storage.states.len(),
        added_resonant_states: 0,
        minimum_external_denominator: minimum,
        exterior_mixing_norm: mixing,
        hermiticity_defect: 0.0,
        eigen_residual: 0.0,
        converged: true,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(scale: f64) -> VibrationalPolynomial {
        VibrationalPolynomial {
            frequencies: vec![0.05],
            cubic: vec![scale * 0.00004],
            quartic: vec![scale * scale * 0.000005],
            dipole: ObservablePolynomial {
                components: 3,
                first: vec![0.7, -0.2, 0.3],
                second: vec![scale * 0.04, scale * -0.03, scale * 0.01],
                third: vec![
                    scale * scale * 0.006,
                    scale * scale * 0.002,
                    scale * scale * -0.004,
                ],
            },
            polarizability: Some(ObservablePolynomial {
                components: 9,
                first: (0..9)
                    .map(|c| [0.7, -0.2, 0.3][c % 3] * (c + 1) as f64 / 10.0)
                    .collect(),
                second: (0..9)
                    .map(|c| scale * [0.04, -0.03, 0.01][c % 3] * (c + 1) as f64 / 10.0)
                    .collect(),
                third: (0..9)
                    .map(|c| scale * scale * [0.006, 0.002, -0.004][c % 3] * (c + 1) as f64 / 10.0)
                    .collect(),
            }),
            rotation: None,
        }
    }
    fn config() -> AnharmonicConfig {
        AnharmonicConfig {
            method: AnharmonicMethod::StrictVpt2,
            ..Default::default()
        }
    }

    /// Independent finite dense ladder matrices. No sparse production actions,
    /// term grouping, wavefunction routines or observable transforms are used.
    fn dense(input: &VibrationalPolynomial, size: usize) -> (Matrix, Matrix, Vec<Matrix>) {
        let w = input.frequencies[0];
        let mut q = Matrix::zeros(size, size);
        for n in 1..size {
            q[(n, n - 1)] = (n as f64 / (2.0 * w)).sqrt();
            q[(n - 1, n)] = q[(n, n - 1)];
        }
        let q2 = q.matmul_seq(&q);
        let q3 = q2.matmul_seq(&q);
        let q4 = q2.matmul_seq(&q2);
        let mut v = q3.clone();
        let mut u = q4;
        for value in v.as_mut_slice() {
            *value *= input.cubic[0] / 6.0;
        }
        for value in u.as_mut_slice() {
            *value *= input.quartic[0] / 24.0;
        }
        let observable = (0..3)
            .map(|c| {
                let mut o = Matrix::zeros(size, size);
                for i in 0..size {
                    for j in 0..size {
                        o[(i, j)] = input.dipole.first[c] * q[(i, j)]
                            + input.dipole.second[c] * q2[(i, j)] / 2.0
                            + input.dipole.third[c] * q3[(i, j)] / 6.0;
                    }
                }
                o
            })
            .collect();
        (v, u, observable)
    }

    #[test]
    fn strict_energies_match_closed_form_and_default_stays_retained() {
        let model = input(1.0);
        let output = vibrational_spectrum(&model, &config()).unwrap();
        assert_eq!(output.method, AnharmonicMethod::StrictVpt2);
        assert_eq!(
            AnharmonicMethod::default(),
            AnharmonicMethod::RetainedPolyad
        );
        for method in [
            AnharmonicMethod::StrictVpt2,
            AnharmonicMethod::RetainedPolyad,
        ] {
            assert_eq!(AnharmonicMethod::parse(method.as_str()).unwrap(), method);
        }
        assert!(AnharmonicMethod::parse("vpt2-typo").is_err());
        let w = model.frequencies[0];
        for (n, &e) in output.polyad_energies.iter().enumerate() {
            let n = n as f64;
            let expected = w * (n + 0.5)
                + model.quartic[0] * (2.0 * n * n + 2.0 * n + 1.0) / (32.0 * w * w)
                - model.cubic[0].powi(2) * (30.0 * n * n + 30.0 * n + 11.0) / (288.0 * w.powi(4));
            assert!((e - expected).abs() < 2e-17, "{e} {expected}");
        }
        let old_default = vibrational_spectrum(&model, &AnharmonicConfig::default()).unwrap();
        let old_named = vibrational_spectrum(
            &model,
            &AnharmonicConfig {
                method: AnharmonicMethod::RetainedPolyad,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(old_default.polyad_energies, old_named.polyad_energies);
        assert_eq!(
            old_default.effective_hamiltonian.as_slice(),
            old_named.effective_hamiltonian.as_slice()
        );
        for (a, b) in old_default.bands.iter().zip(&old_named.bands) {
            assert_eq!(a.transition_dipole, b.transition_dipole);
            assert_eq!(a.ir_intensity, b.ir_intensity);
        }
    }

    #[test]
    fn normalized_moments_match_independent_dense_order_coefficients() {
        // Dense expansion at lambda=0 gives the exact Taylor coefficients;
        // coefficients are evaluated separately, never by squaring full waves.
        let model = input(1.0);
        let output = vibrational_spectrum(&model, &config()).unwrap();
        let (v, u, _) = dense(&model, 14);
        let w = model.frequencies[0];
        let mut c = [
            Matrix::zeros(14, 3),
            Matrix::zeros(14, 3),
            Matrix::zeros(14, 3),
        ];
        for n in 0..3 {
            c[0][(n, n)] = 1.0;
            let mut norm = 0.0;
            for k in 0..14 {
                if k != n {
                    let delta = (n as f64 - k as f64) * w;
                    c[1][(k, n)] = v[(k, n)] / delta;
                    norm += c[1][(k, n)].powi(2);
                    let mut value = u[(k, n)];
                    for m in 0..14 {
                        if m != n {
                            value += v[(k, m)] * v[(m, n)] / ((n as f64 - m as f64) * w);
                        }
                    }
                    c[2][(k, n)] = value / delta;
                }
            }
            c[2][(n, n)] = -0.5 * norm;
        }
        for root in 1..3 {
            for component in 0..3 {
                let mut expected = 0.0;
                for rank in 0..3 {
                    let mut only = model.clone();
                    if rank != 0 {
                        only.dipole.first.fill(0.0);
                    }
                    if rank != 1 {
                        only.dipole.second.fill(0.0);
                    }
                    if rank != 2 {
                        only.dipole.third.fill(0.0);
                    }
                    let (_, _, operator) = dense(&only, 14);
                    for a in 0..=2 - rank {
                        for b in 0..=2 - rank - a {
                            for i in 0..14 {
                                for j in 0..14 {
                                    expected += c[a][(i, 0)]
                                        * operator[component][(i, j)]
                                        * c[b][(j, root)];
                                }
                            }
                        }
                    }
                }
                let actual = output.bands[root - 1].transition_dipole[component];
                assert!(
                    (actual - expected).abs() < 2e-14,
                    "root{root} component{component}: {actual} {expected}"
                );
            }
        }
        assert!(output
            .warnings
            .iter()
            .any(|w| w.contains("not order-truncated intensity")));
        for band in &output.bands {
            let alpha = band.transition_polarizability.unwrap();
            for c in 0..9 {
                let expected = band.transition_dipole[c % 3] * (c + 1) as f64 / 10.0;
                assert!((alpha[c] - expected).abs() < 2e-14);
            }
        }
    }

    fn fit(steps: &[f64], errors: &[f64]) -> (f64, f64) {
        let x: Vec<_> = steps.iter().map(|v| v.ln()).collect();
        let y: Vec<_> = errors.iter().map(|v| v.ln()).collect();
        let mx = x.iter().sum::<f64>() / x.len() as f64;
        let my = y.iter().sum::<f64>() / y.len() as f64;
        let xx = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>();
        let xy = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>();
        let yy = y.iter().map(|v| (v - my).powi(2)).sum::<f64>();
        (xy / xx, xy * xy / (xx * yy))
    }

    #[test]
    fn four_strengths_resolve_energy_and_all_moment_component_orders() {
        let steps = [0.4, 0.3, 0.2, 0.1];
        let mut energy_errors = Vec::new();
        let mut moment_errors = [Vec::new(), Vec::new(), Vec::new()];
        for scale in steps {
            let model = input(scale);
            let output = vibrational_spectrum(&model, &config()).unwrap();
            let (v, mut h, operators) = dense(&model, 32);
            add(&mut h, &v, 1.0);
            for i in 0..32 {
                h[(i, i)] += (i as f64 + 0.5) * model.frequencies[0];
            }
            let (energies, mut vectors) = symmetric_eigen(&h).unwrap();
            for n in 0..3 {
                if vectors[(n, n)] < 0.0 {
                    for i in 0..32 {
                        vectors[(i, n)] = -vectors[(i, n)];
                    }
                }
            }
            energy_errors.push((output.zero_point_energy - energies[0]).abs());
            for c in 0..3 {
                let transformed = vectors
                    .transpose()
                    .matmul_seq(&operators[c])
                    .matmul_seq(&vectors);
                moment_errors[c]
                    .push((output.bands[0].transition_dipole[c] - transformed[(0, 1)]).abs());
            }
        }
        for (label, errors) in std::iter::once(("energy", &energy_errors))
            .chain(moment_errors.iter().map(|e| ("moment", e)))
        {
            let (p, r2) = fit(&steps, errors);
            eprintln!("strict {label}: errors={errors:?}, p={p}, R2={r2}");
            assert!((3.8..4.2).contains(&p) && r2 > 0.999, "{label}: {p} {r2}");
        }
    }

    #[test]
    fn mode_phase_changes_covariantly_and_resources_are_explicit() {
        let model = input(0.5);
        let output = vibrational_spectrum(&model, &config()).unwrap();
        let mut reversed = model.clone();
        reversed.cubic[0] *= -1.0;
        for v in reversed
            .dipole
            .first
            .iter_mut()
            .chain(reversed.dipole.third.iter_mut())
        {
            *v *= -1.0;
        }
        let alpha = reversed.polarizability.as_mut().unwrap();
        for v in alpha.first.iter_mut().chain(alpha.third.iter_mut()) {
            *v *= -1.0;
        }
        let other = vibrational_spectrum(&reversed, &config()).unwrap();
        assert_eq!(output.polyad_energies, other.polyad_energies);
        for (i, (a, b)) in output.bands.iter().zip(&other.bands).enumerate() {
            let phase = if i == 0 { -1.0 } else { 1.0 };
            for c in 0..3 {
                assert!((a.transition_dipole[c] - phase * b.transition_dipole[c]).abs() < 1e-14);
            }
            assert_eq!(a.ir_intensity, b.ir_intensity);
            for (a, b) in a
                .transition_polarizability
                .unwrap()
                .iter()
                .zip(b.transition_polarizability.unwrap())
            {
                assert!((a - phase * b).abs() < 1e-14);
            }
            assert_eq!(a.raman_activity, b.raman_activity);
        }
        let error = vibrational_spectrum(
            &model,
            &AnharmonicConfig {
                max_basis_states: 4,
                max_polyad_states: 3,
                ..config()
            },
        )
        .unwrap_err();
        assert!(matches!(error, DftError::InvalidInput(_)));
        assert!(error
            .to_string()
            .contains("explicit state or memory limits"));
    }

    #[test]
    fn coupled_first_and_second_order_resonances_reject_but_uncoupled_degeneracy_does_not() {
        let mut model = VibrationalPolynomial {
            frequencies: vec![0.05, 0.1],
            cubic: vec![0.0; 8],
            quartic: vec![0.0; 16],
            dipole: ObservablePolynomial {
                components: 3,
                first: vec![0.2; 6],
                second: vec![0.0; 12],
                third: vec![0.0; 24],
            },
            polarizability: None,
            rotation: None,
        };
        for i in [1, 2, 4] {
            model.cubic[i] = 1e-5;
        }
        for shift in [0.0, 1e-7] {
            model.frequencies[1] = 0.1 + shift;
            let error = vibrational_spectrum(&model, &config()).unwrap_err();
            assert!(matches!(error, DftError::InvalidInput(_)));
            assert!(error
                .to_string()
                .contains("strict-vpt2 coupled cubic resonance"));
            assert!(error.to_string().contains("retained-polyad"));
        }
        model.cubic.fill(0.0);
        model.frequencies[1] = 0.05;
        assert!(vibrational_spectrum(&model, &config()).unwrap().converged);
        // No direct cubic element connects the equal-frequency fundamentals,
        // but two cubic interactions do. This must not divide by their zero
        // second-order energy separation or silently ignore the coupling.
        model.cubic[0] = 1e-5;
        for i in [1, 2, 4] {
            model.cubic[i] = 1e-5;
        }
        let error = vibrational_spectrum(&model, &config()).unwrap_err();
        assert!(error
            .to_string()
            .contains("strict-vpt2 coupled second-order resonance"));
        model.cubic.fill(0.0);
        // Q0^3 Q1 couples degenerate fundamental reference states at order2.
        for i in [1, 2, 4, 8] {
            model.quartic[i] = 1e-6;
        }
        let error = vibrational_spectrum(&model, &config()).unwrap_err();
        assert!(error
            .to_string()
            .contains("strict-vpt2 coupled second-order resonance"));
    }
}

#[cfg(test)]
mod native_water_fixture {
    use super::*;
    #[test]
    fn identical_native_water_force_field_matches_strict_order_bands() {
        // Native HF/6-31G stationary all-mode force/rotation output, immutable
        // extension 8d59be3c...; H1.008/O15.999 u, no external implementation.
        let model = VibrationalPolynomial {
            frequencies: vec![
                0.007913681989140145,
                0.018171353375617445,
                0.01888628286771247,
            ],
            cubic: vec![
                1.221539883987645e-6,
                -1.6456952052503563e-6,
                -1.6034466040421914e-17,
                -1.6456952052503563e-6,
                -8.937459199422973e-7,
                1.3029907827612902e-17,
                -1.6034466040421914e-17,
                1.3029907827612902e-17,
                -2.494873702877084e-6,
                -1.6456952052503563e-6,
                -8.937459199422973e-7,
                1.3029907827612902e-17,
                -8.937459199422973e-7,
                2.073486488425812e-5,
                -1.643921544031146e-16,
                1.3029907827612902e-17,
                -1.643921544031146e-16,
                2.1827555013570245e-5,
                -1.6034466040421914e-17,
                1.3029907827612902e-17,
                -2.494873702877084e-6,
                1.3029907827612902e-17,
                -1.643921544031146e-16,
                2.1827555013570245e-5,
                -2.494873702877084e-6,
                2.1827555013570245e-5,
                5.161602409805455e-16,
            ],
            quartic: vec![
                -3.854113983503675e-8,
                7.417120910093965e-8,
                5.7836468429707695e-19,
                7.417120910093965e-8,
                -2.1326763399374614e-7,
                5.828810904416165e-20,
                5.7836468429707695e-19,
                5.828810904416165e-20,
                -2.536124508257263e-7,
                7.417120910093965e-8,
                -2.1326763399374614e-7,
                5.828810904416165e-20,
                -2.1326763399374614e-7,
                -1.0581768705006579e-7,
                1.5430293321013182e-18,
                5.828810904416165e-20,
                1.5430293321013182e-18,
                -1.5027667077967518e-7,
                5.7836468429707695e-19,
                5.828810904416165e-20,
                -2.536124508257263e-7,
                5.828810904416165e-20,
                1.5430293321013182e-18,
                -1.5027667077967518e-7,
                -2.536124508257263e-7,
                -1.5027667077967518e-7,
                -3.370449976243268e-18,
                7.417120910093965e-8,
                -2.1326763399374614e-7,
                5.828810904416165e-20,
                -2.1326763399374614e-7,
                -1.0581768705006579e-7,
                1.5430293321013182e-18,
                5.828810904416165e-20,
                1.5430293321013182e-18,
                -1.5027667077967518e-7,
                -2.1326763399374614e-7,
                -1.0581768705006579e-7,
                1.5430293321013182e-18,
                -1.0581768705006579e-7,
                1.1743090522193544e-6,
                -1.915903823394502e-17,
                1.5430293321013182e-18,
                -1.915903823394502e-17,
                1.2520397911367876e-6,
                5.828810904416165e-20,
                1.5430293321013182e-18,
                -1.5027667077967518e-7,
                1.5430293321013182e-18,
                -1.915903823394502e-17,
                1.2520397911367876e-6,
                -1.5027667077967518e-7,
                1.2520397911367876e-6,
                1.9388107706618104e-17,
                5.7836468429707695e-19,
                5.828810904416165e-20,
                -2.536124508257263e-7,
                5.828810904416165e-20,
                1.5430293321013182e-18,
                -1.5027667077967518e-7,
                -2.536124508257263e-7,
                -1.5027667077967518e-7,
                -3.370449976243268e-18,
                5.828810904416165e-20,
                1.5430293321013182e-18,
                -1.5027667077967518e-7,
                1.5430293321013182e-18,
                -1.915903823394502e-17,
                1.2520397911367876e-6,
                -1.5027667077967518e-7,
                1.2520397911367876e-6,
                1.9388107706618104e-17,
                -2.536124508257263e-7,
                -1.5027667077967518e-7,
                -3.370449976243268e-18,
                -1.5027667077967518e-7,
                1.2520397911367876e-6,
                1.9388107706618104e-17,
                -3.370449976243268e-18,
                1.9388107706618104e-17,
                1.3307684092069222e-6,
            ],
            dipole: ObservablePolynomial {
                components: 3,
                first: vec![
                    7.693910447784213e-18,
                    -5.512083844916305e-16,
                    0.008320609293982309,
                    -1.3571170175623589e-17,
                    -3.724960877943273e-14,
                    -0.0012893889154054344,
                    1.9411083113361177e-18,
                    0.005527524339319179,
                    -3.875182509952002e-14,
                ],
                second: vec![
                    -8.425433581773496e-20,
                    -3.6731609995649914e-17,
                    -0.0001405268441725364,
                    7.023933221394828e-20,
                    1.4717297835154147e-15,
                    9.585744128563328e-5,
                    -7.972516068437375e-18,
                    -0.0001878642099562458,
                    6.884954845826075e-16,
                    7.023933221394828e-20,
                    1.4717297835154147e-15,
                    9.585744128563328e-5,
                    -1.6433029999149723e-19,
                    -2.096045092615425e-15,
                    -7.68847117011121e-05,
                    6.97479491887605e-19,
                    0.00011998194583673271,
                    -6.70680687635955e-16,
                    -7.972516068437375e-18,
                    -0.0001878642099562458,
                    6.884954845826075e-16,
                    6.97479491887605e-19,
                    0.00011998194583673271,
                    -6.70680687635955e-16,
                    8.960987642471862e-21,
                    3.1278808682711665e-15,
                    -1.2125616891662248e-5,
                ],
                third: vec![
                    -1.0606226575773503e-23,
                    -4.044151582122333e-18,
                    2.6239899426166713e-6,
                    5.905889614462381e-21,
                    -8.465275849735878e-18,
                    -3.0728366768222794e-6,
                    -2.4188395464748787e-19,
                    1.1375783656339945e-6,
                    -4.6854262752082064e-17,
                    5.905889614462381e-21,
                    -8.465275849735878e-18,
                    -3.0728366768222794e-6,
                    -4.098842291480838e-21,
                    1.5568195137498577e-18,
                    1.3078330184538764e-6,
                    1.3643836058582436e-19,
                    -2.6439679236221783e-8,
                    9.4084078737153e-17,
                    -2.4188395464748787e-19,
                    1.1375783656339945e-6,
                    -4.6854262752082064e-17,
                    2.1559234064139698e-19,
                    -2.6439679162360987e-8,
                    -5.287211420496865e-17,
                    -2.877204698438554e-20,
                    -3.2813579920598765e-17,
                    -2.2016929956355052e-6,
                    5.905889614462381e-21,
                    -8.465275849735878e-18,
                    -3.0728366768222794e-6,
                    -4.0988422914808386e-21,
                    1.5568195137498546e-18,
                    1.3078330184538764e-6,
                    1.3643836058582436e-19,
                    -2.6439679236221783e-8,
                    9.4084078737153e-17,
                    -4.0988422914808386e-21,
                    1.5568195137498546e-18,
                    1.3078330184538764e-6,
                    4.584165711869156e-20,
                    1.3657817725799325e-16,
                    3.1627224425565456e-6,
                    2.2792939480340986e-19,
                    -3.305802573252723e-6,
                    -8.816854886429474e-16,
                    3.115566500244355e-18,
                    -2.6439679268576324e-8,
                    4.0612159326409904e-17,
                    2.2792939480340986e-19,
                    -3.305802573252723e-6,
                    -8.816854886429474e-16,
                    -1.9836533772563087e-17,
                    -1.5653264156466036e-16,
                    1.9741099012131763e-6,
                    -2.4188395464748787e-19,
                    1.1375783656339945e-6,
                    -4.6854262752082064e-17,
                    2.1559234064139698e-19,
                    -2.6439679162360987e-8,
                    -5.287211420496865e-17,
                    -2.8772046984385636e-20,
                    -3.2813579920598765e-17,
                    -2.2016929956355052e-6,
                    3.115566500244355e-18,
                    -2.6439679268576324e-8,
                    4.0612159326409904e-17,
                    2.2792939480340986e-19,
                    -3.305802573252723e-6,
                    -8.816854886429474e-16,
                    -1.9836533772563087e-17,
                    -1.5653264156466036e-16,
                    1.9741099012131763e-6,
                    -2.8772046984385636e-20,
                    -3.2813579920598765e-17,
                    -2.2016929956355052e-6,
                    -1.9836533772563087e-17,
                    -1.5653264156466036e-16,
                    1.9741099012131763e-6,
                    -5.504212234338557e-21,
                    -1.120867304549453e-7,
                    9.540684679858845e-17,
                ],
            },
            polarizability: None,
            rotation: Some(WatsonRotation {
                class: rotation::RotorClass::Nonlinear,
                inverse_inertia: [
                    [8.75986202345926e-05, 0.0, 0.0],
                    [0.0, 0.00030072750904295176, 2.912271496511192e-17],
                    [0.0, 2.912271496511192e-17, 0.00012360274107390423],
                ],
                principal_moments: [3325.269454671584, 8090.435465359806, 11415.704920031389],
                coriolis: vec![
                    [0.0, 0.0, 0.0],
                    [
                        7.595979401031627e-12,
                        2.8862070473150854e-16,
                        -3.0189206164380596e-16,
                    ],
                    [
                        -0.999683220246922,
                        -3.327009353060112e-16,
                        2.8453428215566037e-16,
                    ],
                    [
                        -7.595979401031627e-12,
                        -2.8862070473150854e-16,
                        3.0189206164380596e-16,
                    ],
                    [0.0, 0.0, 0.0],
                    [
                        -0.02516861451777133,
                        -6.989641422267488e-17,
                        8.627744834632581e-17,
                    ],
                    [
                        0.999683220246922,
                        3.327009353060112e-16,
                        -2.8453428215566037e-16,
                    ],
                    [
                        0.02516861451777133,
                        6.989641422267488e-17,
                        -8.627744834632581e-17,
                    ],
                    [0.0, 0.0, 0.0],
                ],
                watson_pseudopotential: -6.399110879393108e-5,
                eckart_defect: Some(5.097255133808767e-16),
                mode_metric_defect: Some(2.220446049250313e-16),
            }),
        };
        let strict = vibrational_spectrum(
            &model,
            &AnharmonicConfig {
                method: AnharmonicMethod::StrictVpt2,
                ..Default::default()
            },
        )
        .unwrap();
        let expected = [1669.7287847916923, 3816.642161841686, 3956.299629927213];
        let fundamentals: Vec<_> = strict
            .bands
            .iter()
            .filter(|b| b.character == "fundamental")
            .collect();
        assert_eq!(fundamentals.len(), 3);
        for (band, expected) in fundamentals.iter().zip(expected) {
            assert!(
                (band.frequency_cm - expected).abs() < 1e-8,
                "{} {}",
                band.frequency_cm,
                expected
            );
        }
        let retained = vibrational_spectrum(&model, &Default::default()).unwrap();
        let legacy: Vec<_> = retained
            .bands
            .iter()
            .filter(|b| b.character == "fundamental")
            .collect();
        for (band, expected) in
            legacy
                .iter()
                .zip([1671.3055131110639, 3726.6161329141723, 3877.608691119953])
        {
            assert!((band.frequency_cm - expected).abs() < 1e-8);
        }
    }
}
