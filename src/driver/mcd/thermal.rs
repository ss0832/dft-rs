// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical thermal MCD of a supplied real symmetric electronic manifold.
//!
//! H(B)=H0-sum_c B_c M_c. The supplied Q_a are normalized optical-helicity
//! observables in the SAME manifold, including any intended SOC/transition
//! information. No molecular orbitals, excitation moments, host parameters,
//! absolute absorption or orientation average are inferred from the geometry.
//! The exact finite-field ensemble is separate from its zero-field population
//! C response and interlevel (Van Vleck) mixing response. Fermi occupations of
//! a closed-shell SCF are not used as a paramagnetic thermal ensemble.

use crate::driver::{Controls, Value};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
use crate::{DftError, Result};

fn invalid(message: &str) -> DftError {
    DftError::InvalidInput(format!("thermal-manifold MCD: {message}"))
}

fn matrix(controls: &Controls, key: &str, n: Option<usize>) -> Result<Matrix> {
    let rows = controls
        .opt_rows(key)
        .map_err(|error| invalid(&format!("{key} requires real finite numeric rows; complex operators are unsupported ({error})")))?
        .ok_or_else(|| invalid(&format!("{key} is required")))?;
    let size = rows.len();
    if size == 0
        || size > 128
        || n.is_some_and(|n| n != size)
        || rows
            .iter()
            .any(|r| r.len() != size || r.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid(
            "all operators must have the same square size, between 1 and 128",
        ));
    }
    // Never silently copy a triangle or symmetrize supplied physical inputs.
    for i in 0..size {
        for j in 0..size {
            if rows[i][j] != rows[j][i] {
                return Err(invalid(&format!(
                    "{key} must be exactly real symmetric; supply both triangles"
                )));
            }
        }
    }
    Ok(Matrix::from_row_major(
        size,
        size,
        rows.into_iter().flatten().collect(),
    ))
}

fn populations(energies: &[f64], kt: f64) -> Result<Vec<f64>> {
    if energies.iter().any(|e| !e.is_finite()) {
        return Err(invalid("non-finite manifold eigenenergy"));
    }
    let ground = energies[0];
    let mut p: Vec<f64> = energies
        .iter()
        .map(|e| (-(e - ground) / kt).exp())
        .collect();
    let z: f64 = p.iter().sum();
    if !(z.is_finite() && z > 0.0) {
        return Err(invalid("canonical partition sum is not representable"));
    }
    for v in &mut p {
        *v /= z;
    }
    Ok(p)
}

fn mean(operator: &Matrix, p: &[f64]) -> f64 {
    p.iter()
        .enumerate()
        .map(|(i, p)| p * operator[(i, i)])
        .sum()
}

/// Stable spectral divided difference of the normalized exponential.
fn divided(p: &[f64], e: &[f64], i: usize, j: usize, kt: f64) -> f64 {
    let (low, high) = if e[i] <= e[j] { (i, j) } else { (j, i) };
    let gap = e[high] - e[low];
    if gap == 0.0 {
        p[low] / kt
    } else {
        p[low] * (-(-gap / kt).exp_m1()) / gap
    }
}

struct Thermal {
    populations: Vec<f64>,
    energies: Vec<f64>,
    relative_energies: Vec<f64>,
    energy_origin: f64,
    signal: [f64; 3],
    moment: [f64; 3],
    residual: f64,
}

fn centered(h: &Matrix) -> Result<(Matrix, f64)> {
    let origin = h[(0, 0)];
    let mut out = h.clone();
    for i in 0..h.rows {
        out[(i, i)] -= origin;
    }
    if out.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(invalid("energy-centered Hamiltonian overflow"));
    }
    Ok((out, origin))
}

fn ensemble(
    h0: &Matrix,
    magnetic: &[Matrix; 3],
    optical: &[Matrix; 3],
    b: [f64; 3],
    kt: f64,
) -> Result<Thermal> {
    // Center BEFORE adding the small field. Otherwise its diagonal shifts
    // lose precision against an arbitrary large absolute electronic energy.
    let (mut h, energy_origin) = centered(h0)?;
    for c in 0..3 {
        for (v, m) in h.as_mut_slice().iter_mut().zip(magnetic[c].as_slice()) {
            *v -= b[c] * m;
        }
    }
    if h.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(invalid("field Hamiltonian overflow"));
    }
    let (e, u) = symmetric_eigen(&h)?;
    let p = populations(&e, kt)?;
    let transformed = |o: &Matrix| u.transpose_matmul(&o.matmul(&u));
    let signal = std::array::from_fn(|a| mean(&transformed(&optical[a]), &p));
    let moment = std::array::from_fn(|a| mean(&transformed(&magnetic[a]), &p));
    if signal.iter().chain(moment.iter()).any(|v| !v.is_finite()) {
        return Err(invalid("thermal expectation overflow"));
    }
    let hu = h.matmul(&u);
    let mut residual = 0.0_f64;
    for i in 0..h.rows {
        for j in 0..h.cols {
            residual = residual.max((hu[(i, j)] - u[(i, j)] * e[j]).abs());
        }
    }
    let scale = h.as_slice().iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    if !residual.is_finite() || residual > 1e-10 * scale {
        return Err(DftError::LinearAlgebra(format!(
            "thermal MCD fresh eigen residual {residual:e} exceeds the scaled tolerance"
        )));
    }
    let absolute_energies: Vec<f64> = e.iter().map(|e| e + energy_origin).collect();
    if absolute_energies.iter().any(|e| !e.is_finite()) {
        return Err(invalid("absolute manifold eigenenergy overflow"));
    }
    Ok(Thermal {
        populations: p,
        energies: absolute_energies,
        relative_energies: e,
        energy_origin,
        signal,
        moment,
        residual,
    })
}

fn linear(
    h: &Matrix,
    magnetic: &[Matrix; 3],
    optical: &[Matrix; 3],
    kt: f64,
    tolerance: f64,
) -> Result<([[f64; 3]; 3], [[f64; 3]; 3])> {
    let (shifted, _) = centered(h)?;
    let (e, u) = symmetric_eigen(&shifted)?;
    let p = populations(&e, kt)?;
    let m: [Matrix; 3] = std::array::from_fn(|c| u.transpose_matmul(&magnetic[c].matmul(&u)));
    let q: [Matrix; 3] = std::array::from_fn(|a| u.transpose_matmul(&optical[a].matmul(&u)));
    let mut population = [[0.0; 3]; 3];
    let mut mixing = [[0.0; 3]; 3];
    for a in 0..3 {
        for c in 0..3 {
            let average_m = mean(&m[c], &p);
            for i in 0..h.rows {
                population[a][c] += q[a][(i, i)] * p[i] * (m[c][(i, i)] - average_m) / kt;
                for j in 0..i {
                    let value = 2.0 * q[a][(i, j)] * m[c][(j, i)] * divided(&p, &e, i, j, kt);
                    if (e[i] - e[j]).abs() <= tolerance {
                        population[a][c] += value;
                    } else {
                        mixing[a][c] += value;
                    }
                }
            }
        }
    }
    if population
        .iter()
        .flatten()
        .chain(mixing.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(invalid("linear thermal response overflow"));
    }
    Ok((population, mixing))
}

pub(super) fn run(controls: &Controls) -> Result<Value> {
    if controls.contains("reference") {
        return Err(invalid(
            "a supplied thermal manifold does not accept an SCF reference",
        ));
    }
    let temperature = controls.f64("temperature", 0.0)?;
    let kt = temperature * BOLTZMANN_HARTREE_PER_K;
    if !(temperature > 0.0 && kt.is_finite() && kt > 0.0 && (1.0 / kt).is_finite()) {
        return Err(invalid(
            "temperature must be positive and have representable kT and inverse kT",
        ));
    }
    let tolerance = controls.f64("degeneracy_tolerance", 1e-12)?;
    if tolerance < 0.0 {
        return Err(invalid("degeneracy_tolerance must be nonnegative"));
    }
    let field = controls
        .opt_vec3("bfield")?
        .ok_or_else(|| invalid("bfield is required, including for zero field"))?
        .to_array();
    let h = matrix(controls, "mcd_thermal_hamiltonian", None)?;
    let n = Some(h.rows);
    let magnetic = [
        matrix(controls, "mcd_thermal_magnetic_x", n)?,
        matrix(controls, "mcd_thermal_magnetic_y", n)?,
        matrix(controls, "mcd_thermal_magnetic_z", n)?,
    ];
    let optical = [
        matrix(controls, "mcd_thermal_helicity_x", n)?,
        matrix(controls, "mcd_thermal_helicity_y", n)?,
        matrix(controls, "mcd_thermal_helicity_z", n)?,
    ];
    controls.finish("mcd thermal-manifold")?;
    let zero = ensemble(&h, &magnetic, &optical, [0.0; 3], kt)?;
    let finite = ensemble(&h, &magnetic, &optical, field, kt)?;
    let (population, mixing) = linear(&h, &magnetic, &optical, kt, tolerance)?;
    let total: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|c| population[a][c] + mixing[a][c]));
    let numerator: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|c| kt * population[a][c]));
    let difference: [f64; 3] = std::array::from_fn(|a| finite.signal[a] - zero.signal[a]);
    let labels = Value::object()
        .with("temperature", "K")
        .with("thermal_energy", "hartree")
        .with("bfield", "atomic units of magnetic flux density")
        .with("mcd_thermal_energies", "hartree")
        .with("mcd_zero_field_energies", "hartree")
        .with(
            "mcd_thermal_relative_energies",
            "hartree, relative to mcd_energy_origin",
        )
        .with(
            "mcd_zero_field_relative_energies",
            "hartree, relative to mcd_energy_origin",
        )
        .with("mcd_energy_origin", "hartree")
        .with("mcd_thermal_populations", "dimensionless")
        .with("mcd_zero_field_populations", "dimensionless")
        .with(
            "mcd_thermal_signal",
            "dimensionless, supplied normalized optical-helicity observable",
        )
        .with(
            "mcd_zero_field_signal",
            "dimensionless, supplied normalized optical-helicity observable",
        )
        .with(
            "mcd_thermal_difference",
            "dimensionless, finite minus zero field normalized helicity",
        )
        .with(
            "mcd_thermal_moment",
            "atomic units of magnetic dipole moment",
        )
        .with(
            "mcd_population_c_tensor",
            "atomic units of normalized helicity per magnetic flux density",
        )
        .with(
            "mcd_state_mixing_tensor",
            "atomic units of normalized helicity per magnetic flux density",
        )
        .with(
            "mcd_total_thermal_tensor",
            "atomic units of normalized helicity per magnetic flux density",
        )
        .with(
            "mcd_population_c_numerator",
            "atomic units of normalized helicity times magnetic dipole moment",
        )
        .with("degeneracy_tolerance", "hartree")
        .with("thermal_eigen_residual", "hartree");
    Ok(Value::object()
        .with("model","canonical Boltzmann ensemble of supplied real symmetric electronic sublevel Hamiltonian, magnetic moments and normalized optical-helicity operators; no generated molecular DFT/SOC transition manifold")
        .with("geometry_used",false)
        .with("mcd_model","thermal-manifold").with("temperature",temperature).with("thermal_energy",kt).with("bfield",field)
        .with("mcd_thermal_energies",finite.energies).with("mcd_zero_field_energies",zero.energies)
        .with("mcd_thermal_relative_energies",finite.relative_energies).with("mcd_zero_field_relative_energies",zero.relative_energies).with("mcd_energy_origin",finite.energy_origin)
        .with("mcd_thermal_populations",finite.populations).with("mcd_zero_field_populations",zero.populations)
        .with("mcd_thermal_signal",finite.signal).with("mcd_zero_field_signal",zero.signal).with("mcd_thermal_difference",difference)
        .with("mcd_thermal_moment",finite.moment).with("mcd_population_c_tensor",population)
        .with("mcd_state_mixing_tensor",mixing).with("mcd_total_thermal_tensor",total).with("mcd_population_c_numerator",numerator)
        .with("degeneracy_tolerance",tolerance).with("thermal_eigen_residual",finite.residual.max(zero.residual))
        .with("mcd_thermal_convention","H=H0-B.M; positive M lowers energy along positive B. Center H0 by mcd_energy_origin before applying the field; populations, relative energies and the eigen residual use this centered operator, while absolute eigenenergies are retained separately. Q is supplied normalized left-minus-right optical helicity and is field independent. Population C includes the diagonal canonical covariance and within-degenerate-level coherences; interlevel mixing is separate. Their sum is the exact zero-field derivative within this supplied manifold. Mixing here is only within that manifold, not the complete excited-state Faraday A/B terms or field derivatives of optical transition operators. Finite-field signal includes the two supplied-manifold mechanisms and saturation; it is not a pure C term at arbitrary B. Tolerance changes only the decomposition, never the total response.")
        .with("converged",true).with("provenance","analytic finite-state thermal ensemble")
        .with("warnings",vec!["Supplied normalized effective manifold only: no automatic molecular SOC/optical-state generation, absolute absorption, vibronic spectrum, host model or orientation average; complex operators require a separate implementation."])
        .with("units",labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn m(rows: &[[f64; 2]; 2]) -> Matrix {
        Matrix::from_row_major(2, 2, rows.iter().flatten().copied().collect())
    }
    #[test]
    fn canonical_doublet_c_is_nonzero_has_curie_limit_and_saturates() {
        let h = m(&[[0., 0.], [0., 0.]]);
        let z = m(&[[1., 0.], [0., -1.]]);
        let zero = Matrix::zeros(2, 2);
        let mag = [zero.clone(), zero.clone(), z.clone()];
        let opt = [zero.clone(), zero, z];
        let kt = 0.03;
        let (c, mix) = linear(&h, &mag, &opt, kt, 1e-12).unwrap();
        assert!((c[2][2] - 1. / kt).abs() < 1e-13);
        assert_eq!(mix, [[0.; 3]; 3]);
        for b in [0.001, 0.02, 0.2] {
            let r = ensemble(&h, &mag, &opt, [0., 0., b], kt).unwrap();
            assert!((r.signal[2] - (b / kt).tanh()).abs() < 1e-14);
        }
        assert!(ensemble(&h, &mag, &opt, [0., 0., 1.], kt).unwrap().signal[2] > 0.999999999);
    }
    #[test]
    fn all_nine_population_and_mixing_response_components_fit_four_fields() {
        for gap in [0., 0.017] {
            let h = m(&[[0., 0.], [0., gap]]);
            let mag = [
                m(&[[0.17, 0.23], [0.23, -0.31]]),
                m(&[[-0.27, 0.11], [0.11, 0.39]]),
                m(&[[0.42, -0.19], [-0.19, -0.13]]),
            ];
            let opt = [
                m(&[[0.31, -0.16], [-0.16, 0.12]]),
                m(&[[-0.23, 0.27], [0.27, 0.41]]),
                m(&[[0.14, 0.39], [0.39, -0.37]]),
            ];
            let kt = 0.08;
            let (c, mix) = linear(&h, &mag, &opt, kt, 1e-12).unwrap();
            let steps: [f64; 4] = [0.008, 0.004, 0.002, 0.001];
            for a in 0..3 {
                for axis in 0..3 {
                    let target = c[a][axis] + mix[a][axis];
                    let mut ys = [0.; 4];
                    for (i, step) in steps.iter().enumerate() {
                        let mut b = [0.; 3];
                        b[axis] = *step;
                        let plus = ensemble(&h, &mag, &opt, b, kt).unwrap().signal[a];
                        b[axis] = -step;
                        let minus = ensemble(&h, &mag, &opt, b, kt).unwrap().signal[a];
                        let error = ((plus - minus) / (2. * step) - target).abs();
                        assert!(error > 0.);
                        ys[i] = error.ln();
                    }
                    let xs = steps.map(f64::ln);
                    let xm = xs.iter().sum::<f64>() / 4.;
                    let ym = ys.iter().sum::<f64>() / 4.;
                    let slope = xs
                        .iter()
                        .zip(ys)
                        .map(|(x, y)| (x - xm) * (y - ym))
                        .sum::<f64>()
                        / xs.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
                    let ss = ys.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
                    let residual = xs
                        .iter()
                        .zip(ys)
                        .map(|(x, y)| (y - ym - slope * (x - xm)).powi(2))
                        .sum::<f64>();
                    let r2 = 1. - residual / ss;
                    eprintln!("thermal MCD a={a} c={axis} p={slope:0.10} R2={r2:0.12}");
                    assert!((slope - 2.).abs() < 0.15 && r2 > 0.999);
                }
            }
        }
    }
    #[test]
    fn driver_report_and_strict_input_contract_are_executed() {
        let zero = [[0.0, 0.0], [0.0, 0.0]];
        let z = [[1.0, 0.0], [0.0, -1.0]];
        let request = Value::object()
            .with("temperature", 300.0)
            .with("bfield", [0.0, 0.0, 1e-4])
            .with("mcd_thermal_hamiltonian", zero)
            .with("mcd_thermal_magnetic_x", zero)
            .with("mcd_thermal_magnetic_y", zero)
            .with("mcd_thermal_magnetic_z", z)
            .with("mcd_thermal_helicity_x", zero)
            .with("mcd_thermal_helicity_y", zero)
            .with("mcd_thermal_helicity_z", z);
        let r = run(&Controls::from_value(&request).unwrap()).unwrap();
        assert_eq!(r.get("converged").unwrap().as_bool(), Some(true));
        assert!(r.get("thermal_eigen_residual").unwrap().as_f64().unwrap() < 1e-14);
        let p = r
            .get("mcd_thermal_populations")
            .unwrap()
            .as_array()
            .unwrap();
        assert!((p.iter().map(|v| v.as_f64().unwrap()).sum::<f64>() - 1.0).abs() < 1e-15);
        for key in [
            "mcd_thermal_moment",
            "mcd_population_c_tensor",
            "mcd_state_mixing_tensor",
        ] {
            assert!(r.get("units").unwrap().get(key).is_some());
        }
        let mut bad = request.clone();
        bad.set("temperature", 0.0);
        assert!(run(&Controls::from_value(&bad).unwrap()).is_err());
        let mut bad = request.clone();
        bad.set("reference", "gks");
        assert!(run(&Controls::from_value(&bad).unwrap()).is_err());
        let mut bad = request.clone();
        bad.set("mcd_thermal_hamiltonian", [[0.0, 0.2], [0.1, 0.0]]);
        assert!(run(&Controls::from_value(&bad).unwrap()).is_err());
        let mut bad = request.clone();
        bad.set(
            "mcd_thermal_hamiltonian",
            [[[0.0, 0.0], [0.0, 1.0]], [[0.0, -1.0], [0.0, 0.0]]],
        );
        let failure = run(&Controls::from_value(&bad).unwrap())
            .unwrap_err()
            .to_string();
        assert!(failure.contains("complex operators are unsupported"));
        let mut bad = request;
        bad.set("unknown_thermal_parameter", 1.0);
        assert!(run(&Controls::from_value(&bad).unwrap()).is_err());
    }
    #[test]
    fn near_degenerate_divided_difference_is_continuous_without_denominator_floor() {
        let kt = 0.1;
        for gap in [0., 1e-18, 1e-14, 1e-10] {
            let e = [0., gap];
            let p = populations(&e, kt).unwrap();
            assert!((divided(&p, &e, 0, 1, kt) - 5.).abs() < 1e-8);
        }
    }
    #[test]
    fn population_and_interlevel_outputs_each_fit_all_nine_components_independently() {
        let h = m(&[[0.0, 0.0], [0.0, 0.017]]);
        let magnetic = [
            m(&[[0.17, 0.23], [0.23, -0.31]]),
            m(&[[-0.27, 0.11], [0.11, 0.39]]),
            m(&[[0.42, -0.19], [-0.19, -0.13]]),
        ];
        let optical = [
            m(&[[0.31, -0.16], [-0.16, 0.12]]),
            m(&[[-0.23, 0.27], [0.27, 0.41]]),
            m(&[[0.14, 0.39], [0.39, -0.37]]),
        ];
        let frozen = magnetic
            .each_ref()
            .map(|v| m(&[[v[(0, 0)], 0.0], [0.0, v[(1, 1)]]]));
        let kt = 0.08;
        let (c, mixing) = linear(&h, &magnetic, &optical, kt, 1e-12).unwrap();
        let hs: [f64; 4] = [0.008, 0.004, 0.002, 0.001];
        for axis in 0..3 {
            let mut errors = [[[0.0_f64; 4]; 3]; 2];
            for (i, step) in hs.iter().enumerate() {
                let mut b = [0.0; 3];
                b[axis] = *step;
                let plus = ensemble(&h, &magnetic, &optical, b, kt).unwrap().signal;
                let frozen_plus = ensemble(&h, &frozen, &optical, b, kt).unwrap().signal;
                b[axis] = -step;
                let minus = ensemble(&h, &magnetic, &optical, b, kt).unwrap().signal;
                let frozen_minus = ensemble(&h, &frozen, &optical, b, kt).unwrap().signal;
                for a in 0..3 {
                    let cp = (frozen_plus[a] - frozen_minus[a]) / (2.0 * step);
                    let total = (plus[a] - minus[a]) / (2.0 * step);
                    errors[0][a][i] = (cp - c[a][axis]).abs();
                    errors[1][a][i] = (total - cp - mixing[a][axis]).abs();
                }
            }
            for (channel, rows) in errors.iter().enumerate() {
                for (a, error) in rows.iter().enumerate() {
                    assert!(error.iter().all(|e| *e > 0.0));
                    let x = hs.map(f64::ln);
                    let y = error.map(f64::ln);
                    let xm = x.iter().sum::<f64>() / 4.0;
                    let ym = y.iter().sum::<f64>() / 4.0;
                    let p = x
                        .iter()
                        .zip(y)
                        .map(|(x, y)| (x - xm) * (y - ym))
                        .sum::<f64>()
                        / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
                    let r2 = 1.0
                        - x.iter()
                            .zip(y)
                            .map(|(x, y)| (y - ym - p * (x - xm)).powi(2))
                            .sum::<f64>()
                            / y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
                    eprintln!("separate channel={channel} a={a} c={axis} p={p:.10} R2={r2:.12}");
                    assert!((p - 2.0).abs() < 0.15 && r2 > 0.999);
                }
            }
        }
    }
    #[test]
    fn near_degenerate_grouping_changes_only_the_decomposition() {
        let h = m(&[[0.0, 0.0], [0.0, 1e-14]]);
        let magnetic = [
            m(&[[0.1, 0.2], [0.2, -0.3]]),
            m(&[[0.2, -0.1], [-0.1, 0.4]]),
            m(&[[0.3, 0.4], [0.4, -0.2]]),
        ];
        let optical = magnetic.clone();
        let (a, b) = linear(&h, &magnetic, &optical, 0.03, 0.0).unwrap();
        let (c, d) = linear(&h, &magnetic, &optical, 0.03, 1e-12).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((a[i][j] + b[i][j] - c[i][j] - d[i][j]).abs() < 2e-13);
            }
        }
    }
    #[test]
    fn finite_field_signal_and_populations_ignore_large_absolute_energy_origins() {
        let h = m(&[[0.0, 0.0], [0.0, 0.0]]);
        let magnetic = [
            m(&[[0.1, 0.2], [0.2, -0.3]]),
            m(&[[0.2, -0.1], [-0.1, 0.4]]),
            m(&[[0.3, 0.4], [0.4, -0.2]]),
        ];
        let optical = magnetic.clone();
        let b = [0.017, 0.023, -0.011];
        let base = ensemble(&h, &magnetic, &optical, b, 0.03).unwrap();
        let response = linear(&h, &magnetic, &optical, 0.03, 1e-12).unwrap();
        for origin in [-1000.0, -100.0, 1000.0] {
            let shifted = m(&[[origin, 0.0], [0.0, origin]]);
            let r = ensemble(&shifted, &magnetic, &optical, b, 0.03).unwrap();
            assert_eq!(r.signal, base.signal);
            assert_eq!(r.populations, base.populations);
            assert_eq!(r.relative_energies, base.relative_energies);
            assert_eq!(
                linear(&shifted, &magnetic, &optical, 0.03, 1e-12).unwrap(),
                response
            );
        }
    }
    #[test]
    fn energy_offset_and_degenerate_basis_rotation_preserve_population_c() {
        let h = m(&[[0., 0.], [0., 0.]]);
        let mag = [
            m(&[[0.1, 0.2], [0.2, -0.3]]),
            m(&[[0.2, -0.1], [-0.1, 0.4]]),
            m(&[[0.3, 0.4], [0.4, -0.2]]),
        ];
        let opt = mag.clone();
        let first = linear(&h, &mag, &opt, 0.03, 1e-12).unwrap();
        let angle: f64 = 0.37;
        let u = m(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
        let rotate = |a: &Matrix| u.transpose_matmul(&a.matmul(&u));
        let second = linear(
            &m(&[[3., 0.], [0., 3.]]),
            &mag.each_ref().map(rotate),
            &opt.each_ref().map(rotate),
            0.03,
            1e-12,
        )
        .unwrap();
        for a in 0..3 {
            for c in 0..3 {
                assert!((first.0[a][c] - second.0[a][c]).abs() < 1e-13);
                assert_eq!(second.1[a][c], 0.);
            }
        }
    }
}
