// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent population and finite-field Drude probes of energy-dependent
//! relaxation-time response. Finite differences occur only in this suite.

#[path = "support/convergence.rs"]
mod convergence;

use bands::{HallState, TransportState};
use dft_rs::periodic::{band_transport as bands, scattering as model};
use dft_rs::response::complex::Complex;

const VOLUME: f64 = 7.0;
const NK: usize = 3;
const MU: f64 = 0.03;
const KT: f64 = 0.11;
const MASS: f64 = 1.3;
// The wider [0.08, 0.04, 0.02, 0.01] interval is pre-asymptotic for
// this multienergy population probe (documented in response.md).
const STEPS: [f64; 4] = [0.002, 0.001, 0.0005, 0.00025];

fn states() -> (Vec<TransportState>, Vec<HallState>) {
    let mut t = Vec::new();
    let mut h = Vec::new();
    for (energy, speed) in [(-0.13, 0.4), (0.02, 0.7), (0.19, 1.1)] {
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut velocity = [0.0; 3];
                velocity[axis] = sign * speed;
                t.push(TransportState {
                    energy,
                    count: 1.0,
                    velocity2: std::array::from_fn(|a| {
                        std::array::from_fn(|b| velocity[a] * velocity[b])
                    }),
                });
                h.push(HallState {
                    energy,
                    velocity,
                    inverse_mass: std::array::from_fn(|a| {
                        std::array::from_fn(|b| if a == b { 1.0 / MASS } else { 0.0 })
                    }),
                });
            }
        }
    }
    (t, h)
}

fn table(shift: f64) -> model::RelaxationTimeTable {
    model::RelaxationTimeTable::new(vec![[-0.3 + shift, 1.7], [0.3 + shift, 3.2]]).unwrap()
}

fn occupation(energy: f64) -> f64 {
    0.5 * (1.0 - ((energy - MU) / (2.0 * KT)).tanh())
}
fn minus_derivative(energy: f64) -> f64 {
    1.0 / (4.0 * KT * ((energy - MU) / (2.0 * KT)).cosh().powi(2))
}

#[test]
fn moments_and_hall_match_independent_sums_and_preserve_particle_number() {
    let (states, hall) = states();
    let table = table(0.0);
    let tau = |e| table.lifetime(e);
    let moments = model::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0, &tau).unwrap();
    let h = model::hall_integrals(&hall, VOLUME, NK, 2.0, MU, KT, 40.0, &tau).unwrap();
    let coefficients = model::transport_coefficients(&moments, KT);
    let mut worst = 0.0_f64;
    for (order, actual) in [moments.l0, moments.l1, moments.l2].iter().enumerate() {
        for a in 0..3 {
            for b in 0..3 {
                let expected: f64 = hall
                    .iter()
                    .map(|s| {
                        tau(s.energy).unwrap()
                            * minus_derivative(s.energy)
                            * (s.energy - MU).powi(order as i32)
                            * s.velocity[a]
                            * s.velocity[b]
                    })
                    .sum::<f64>()
                    * 2.0
                    / (VOLUME * NK as f64);
                worst = worst.max((expected - actual[a][b]).abs());
            }
        }
    }
    let expected_h: f64 = -hall
        .iter()
        .map(|s| {
            tau(s.energy).unwrap().powi(2) * minus_derivative(s.energy) * s.velocity[0].powi(2)
                / MASS
        })
        .sum::<f64>()
        * 2.0
        / (VOLUME * NK as f64);
    worst = worst.max((h.conductivity_derivative[0][1][2] - expected_h).abs());
    let expected_n =
        states.iter().map(|s| occupation(s.energy)).sum::<f64>() * 2.0 / (VOLUME * NK as f64);
    let unweighted = bands::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0).unwrap();
    assert_eq!(moments.electron_density, unweighted.electron_density);
    assert!((moments.electron_density - expected_n).abs() < 1e-15);
    assert_eq!(coefficients.conductivity, moments.l0);
    assert!(coefficients.thermal_conductivity.unwrap()[0][0] > 0.0);
    assert!(model::hall_coefficient(&moments, &h).unwrap()[1][0][2] < 0.0);
    assert!(h.raw_onsager_residual < 1e-15);
    println!(
        "energy-scattering dense moments and Hall max defect {worst:e}; raw Onsager {:e}",
        h.raw_onsager_residual
    );
    assert!(worst < 1e-14);
}

#[test]
fn electric_and_hall_response_pass_independent_four_step_error_fits() {
    let (states, hall) = states();
    let table = table(0.0);
    let tau = |e| table.lifetime(e);
    let dc = model::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0, &tau).unwrap();
    let h = model::hall_integrals(&hall, VOLUME, NK, 2.0, MU, KT, 40.0, &tau).unwrap();
    // Independent central response of Fermi populations to a signed energy
    // probe tau(E) v_x h. This tests linear response, not nonlinear DC flow.
    let electric: Vec<_> = STEPS
        .iter()
        .map(|step| {
            -hall
                .iter()
                .map(|s| {
                    let shift = step * tau(s.energy).unwrap() * s.velocity[0];
                    s.velocity[0] * (occupation(s.energy + shift) - occupation(s.energy - shift))
                        / (2.0 * step)
                })
                .sum::<f64>()
                * 2.0
                / (VOLUME * NK as f64)
        })
        .collect();
    // Independent exact isotropic Drude shells at finite signed B. The
    // shell Hall current is -B tau^2 v_x^2 f' / [m (1+(B tau/m)^2)].
    let magnetic: Vec<_> = STEPS
        .iter()
        .map(|step| {
            -hall
                .iter()
                .map(|s| {
                    let lifetime = tau(s.energy).unwrap();
                    lifetime.powi(2) * s.velocity[0].powi(2) * minus_derivative(s.energy)
                        / (MASS * (1.0 + (step * lifetime / MASS).powi(2)))
                })
                .sum::<f64>()
                * 2.0
                / (VOLUME * NK as f64)
        })
        .collect();
    for (name, measured, analytic) in [
        ("electric", electric, dc.l0[0][0]),
        ("Hall", magnetic, h.conductivity_derivative[0][1][2]),
    ] {
        let errors: Vec<_> = measured.iter().map(|x| (x - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&STEPS, &errors);
        let richardson = ((4.0 * measured[3] - measured[2]) / 3.0 - analytic).abs();
        println!("energy-scattering {name}: steps {STEPS:?} errors {errors:?} p={:.9} R2={:.12}; Richardson {richardson:e}", fit.exponent, fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(richardson < 1e-5);
    }
}

#[test]
fn constant_lifetime_static_and_frequency_limits_and_energy_origin_agree() {
    let (states, hall) = states();
    let l = bands::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0).unwrap();
    let h = bands::hall_integrals(&hall, VOLUME, NK, 2.0, MU, KT, 40.0).unwrap();
    let tau = 2.7;
    let mut worst = 0.0_f64;
    for omega in [0.0, 0.23, -0.23] {
        let old = bands::frequency_transport(&l, Some(&h), omega, tau).unwrap();
        let actual = model::frequency_transport(
            &states,
            Some(&hall),
            VOLUME,
            NK,
            2.0,
            MU,
            KT,
            40.0,
            omega,
            &|_| Ok(tau),
        )
        .unwrap();
        for (a, b) in actual
            .conductivity
            .iter()
            .flatten()
            .zip(old.conductivity.iter().flatten())
        {
            worst = worst.max((*a - *b).abs());
        }
        for (a, b) in actual
            .hall_conductivity_derivative
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .zip(
                old.hall_conductivity_derivative
                    .unwrap()
                    .iter()
                    .flatten()
                    .flatten(),
            )
        {
            worst = worst.max((*a - *b).abs());
        }
    }
    let t = table(0.0);
    let shifted = table(0.8);
    let moved: Vec<_> = states
        .iter()
        .cloned()
        .map(|mut s| {
            s.energy += 0.8;
            s
        })
        .collect();
    let initial =
        model::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0, &|e| t.lifetime(e))
            .unwrap();
    let translated =
        model::transport_integrals(&moved, VOLUME, NK, 2.0, MU + 0.8, KT, 40.0, &|e| {
            shifted.lifetime(e)
        })
        .unwrap();
    for (a, b) in initial
        .l0
        .iter()
        .flatten()
        .zip(translated.l0.iter().flatten())
    {
        worst = worst.max((a - b).abs());
    }
    let plus = model::frequency_transport(
        &states,
        Some(&hall),
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        0.23,
        &|e| t.lifetime(e),
    )
    .unwrap();
    let minus = model::frequency_transport(
        &states,
        Some(&hall),
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        -0.23,
        &|e| t.lifetime(e),
    )
    .unwrap();
    for (a, b) in plus
        .conductivity
        .iter()
        .flatten()
        .zip(minus.conductivity.iter().flatten())
    {
        assert!((*a - b.conj()).abs() < 1e-15);
    }
    assert!(plus.conductivity[0][0].re > 0.0 && plus.conductivity[0][0].im > 0.0);
    let zero =
        model::frequency_transport(&states, None, VOLUME, NK, 2.0, MU, KT, 40.0, 0.0, &|e| {
            t.lifetime(e)
        })
        .unwrap();
    for (a, b) in zero
        .conductivity
        .iter()
        .flatten()
        .zip(initial.l0.iter().flatten())
    {
        worst = worst.max((a.re - b).abs());
        assert_eq!(a.im, 0.0);
    }
    println!("constant-time, static and energy-origin max defect {worst:e}");
    assert!(worst < 1e-14);
}

#[test]
fn invalid_lifetimes_tables_and_nonfinite_states_are_rejected_without_floors() {
    for rows in [
        vec![],
        vec![[0.0, 0.0]],
        vec![[0.0, 1.0], [0.0, 2.0]],
        vec![[1.0, 1.0], [0.0, 2.0]],
        vec![[f64::NAN, 1.0]],
    ] {
        assert!(model::RelaxationTimeTable::new(rows).is_err());
    }
    let t = model::RelaxationTimeTable::new(vec![[0.0, 1.0], [2.0, 9.0]]).unwrap();
    assert!((t.lifetime(1.0).unwrap() - 3.0).abs() < 1e-15);
    assert_eq!(t.lifetime(-1.0).unwrap(), 1.0);
    assert_eq!(t.lifetime(3.0).unwrap(), 9.0);
    assert!(t.lifetime(f64::NAN).is_err());
    let (mut states, hall) = states();
    for tau in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            model::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0, &|_| Ok(tau))
                .is_err()
        );
        assert!(model::hall_integrals(&hall, VOLUME, NK, 2.0, MU, KT, 40.0, &|_| Ok(tau)).is_err());
    }
    states[0].energy = f64::NAN;
    assert!(
        model::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0, &|_| Ok(1.0)).is_err()
    );
}

#[test]
fn retarded_heat_current_passes_independent_four_step_population_fits() {
    let (states, hall) = states();
    let table = table(0.0);
    let omega = 0.23;
    let response = model::thermoelectric_frequency_transport(
        &states,
        None,
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        omega,
        &|e| table.lifetime(e),
    )
    .unwrap();
    for imaginary in [false, true] {
        let analytic = if imaginary {
            response.moments[2][0][0].im / KT
        } else {
            response.moments[2][0][0].re / KT
        };
        let estimates: Vec<_> = STEPS
            .iter()
            .map(|step| {
                -hall
                    .iter()
                    .map(|s| {
                        let tau = table.lifetime(s.energy).unwrap();
                        let propagator = if imaginary { omega * tau * tau } else { tau }
                            / (1.0 + (omega * tau).powi(2));
                        let shift = step * propagator * s.velocity[0] * (s.energy - MU) / KT;
                        (s.energy - MU)
                            * s.velocity[0]
                            * (occupation(s.energy + shift) - occupation(s.energy - shift))
                            / (2.0 * step)
                    })
                    .sum::<f64>()
                    * 2.0
                    / (VOLUME * NK as f64)
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|v| (v - analytic).abs()).collect();
        let fit = convergence::error_power_fit(&STEPS, &errors);
        let richardson = ((4.0 * estimates[3] - estimates[2]) / 3.0 - analytic).abs();
        println!("retarded heat current imaginary={imaginary}: steps {STEPS:?} errors {errors:?} p={:.9} R2={:.12}; Richardson {richardson:e}", fit.exponent, fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999);
        assert!(richardson < 1e-8);
    }
}

#[test]
fn complex_thermoelectric_tensors_match_cofactor_oracles_and_physical_limits() {
    let (mut states, mut hall) = states();
    // Energy-dependent anisotropy makes K0 and K1 noncommuting. The oracle
    // uses cofactors, independently of the production real-block faer LU.
    for (s, h) in states.iter_mut().zip(&mut hall) {
        let v = h.velocity;
        h.velocity = [
            v[0] + (0.3 + h.energy) * v[1],
            0.2 * v[0] + 1.4 * v[1] + (0.1 - h.energy) * v[2],
            0.4 * v[1] + 0.8 * v[2],
        ];
        s.velocity2 =
            std::array::from_fn(|a| std::array::from_fn(|b| h.velocity[a] * h.velocity[b]));
    }
    let table = table(0.0);
    let omega = 0.23;
    let actual = model::thermoelectric_frequency_transport(
        &states,
        None,
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        omega,
        &|e| table.lifetime(e),
    )
    .unwrap();
    let mut k = [[[Complex::ZERO; 3]; 3]; 3];
    for s in &hall {
        let tau = table.lifetime(s.energy).unwrap();
        let denominator = 1.0 + (omega * tau).powi(2);
        let p = Complex::new(tau / denominator, omega * tau * tau / denominator);
        for n in 0..3 {
            for a in 0..3 {
                for b in 0..3 {
                    k[n][a][b] += p
                        * (2.0 / (VOLUME * NK as f64)
                            * minus_derivative(s.energy)
                            * (s.energy - MU).powi(n as i32)
                            * s.velocity[a]
                            * s.velocity[b]);
                }
            }
        }
    }
    let m = k[0];
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inverse: [[Complex; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
            (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / determinant
        })
    });
    let product = |a: [[Complex; 3]; 3], b: [[Complex; 3]; 3]| -> [[Complex; 3]; 3] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).fold(Complex::ZERO, |v, l| v + a[i][l] * b[l][j]))
        })
    };
    let left = product(inverse, k[1]);
    let right = product(k[1], inverse);
    let correction = product(k[1], left);
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            for (x, y) in [
                (actual.seebeck.unwrap()[a][b], -left[a][b] / KT),
                (actual.peltier.unwrap()[a][b], -right[a][b]),
                (
                    actual.thermal_conductivity.unwrap()[a][b],
                    (k[2][a][b] - correction[a][b]) / KT,
                ),
            ] {
                worst = worst.max((x - y).abs());
            }
        }
    }
    assert!(actual.inverse_residual.unwrap() < 1e-14 && actual.kelvin_residual.unwrap() < 1e-14);
    let minus = model::thermoelectric_frequency_transport(
        &states,
        None,
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        -omega,
        &|e| table.lifetime(e),
    )
    .unwrap();
    for (a, b) in [actual.seebeck, actual.peltier, actual.thermal_conductivity]
        .into_iter()
        .zip([minus.seebeck, minus.peltier, minus.thermal_conductivity])
    {
        for (x, y) in a.unwrap().iter().flatten().zip(b.unwrap().iter().flatten()) {
            worst = worst.max((*x - y.conj()).abs());
        }
    }
    let zero = model::thermoelectric_frequency_transport(
        &states,
        None,
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        0.0,
        &|e| table.lifetime(e),
    )
    .unwrap();
    let dc = model::transport_coefficients(
        &model::transport_integrals(&states, VOLUME, NK, 2.0, MU, KT, 40.0, &|e| {
            table.lifetime(e)
        })
        .unwrap(),
        KT,
    );
    for (complex, real) in [
        (zero.seebeck, dc.seebeck),
        (zero.thermal_conductivity, dc.thermal_conductivity),
    ] {
        for (x, y) in complex
            .unwrap()
            .iter()
            .flatten()
            .zip(real.unwrap().iter().flatten())
        {
            worst = worst.max((*x - Complex::from(*y)).abs());
        }
    }
    // Tiny carrier weights change conductivity and heat conduction, while
    // the dimensionless thermopower remains finite without an inverse floor.
    let small: Vec<_> = states
        .iter()
        .cloned()
        .map(|mut s| {
            s.velocity2 = s.velocity2.map(|r| r.map(|v| v * 1e-250));
            s
        })
        .collect();
    let tiny = model::thermoelectric_frequency_transport(
        &small,
        None,
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        omega,
        &|e| table.lifetime(e),
    )
    .unwrap();
    for (x, y) in tiny
        .seebeck
        .unwrap()
        .iter()
        .flatten()
        .zip(actual.seebeck.unwrap().iter().flatten())
    {
        worst = worst.max((*x - *y).abs());
    }
    println!("complex thermoelectric cofactor, static, reversal and tiny-weight max defect {worst:e}; inverse {:e}; Kelvin {:e}", actual.inverse_residual.unwrap(), actual.kelvin_residual.unwrap());
    assert!(worst < 2e-14);
    let mut singular = states;
    for s in &mut singular {
        for a in 0..3 {
            for b in 0..3 {
                if a != 0 || b != 0 {
                    s.velocity2[a][b] = 0.0;
                }
            }
        }
    }
    let singular = model::thermoelectric_frequency_transport(
        &singular,
        None,
        VOLUME,
        NK,
        2.0,
        MU,
        KT,
        40.0,
        omega,
        &|e| table.lifetime(e),
    )
    .unwrap();
    assert!(
        singular.seebeck.is_none()
            && singular.peltier.is_none()
            && singular.thermal_conductivity.is_none()
    );
    assert!(singular.electric.conductivity[0][0].abs() > 0.0);
}

#[test]
fn parallel_weight_evaluation_preserves_deterministic_transport_sums() {
    let (states, _) = states();
    let large: Vec<_> = (0..80).flat_map(|_| states.iter().cloned()).collect();
    let table = table(0.0);
    let calculate = || {
        model::transport_integrals(&large, VOLUME, NK * 80, 2.0, MU, KT, 40.0, &|e| {
            table.lifetime(e)
        })
        .unwrap()
    };
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(calculate);
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(calculate);
    assert_eq!(serial.l0, parallel.l0);
    assert_eq!(serial.l1, parallel.l1);
    assert_eq!(serial.l2, parallel.l2);
    assert_eq!(serial.electron_density, parallel.electron_density);
}
