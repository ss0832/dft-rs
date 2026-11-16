// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex Hermitian thermal operators, retaining both quadratures exactly.
use crate::driver::{Controls, Value};
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

pub(super) fn invalid(message: &str) -> DftError {
    DftError::InvalidInput(format!("complex thermal MCD: {message}"))
}

pub(super) fn operator(controls: &Controls, key: &str, size: Option<usize>) -> Result<BlochMatrix> {
    let re = controls
        .opt_rows(key)?
        .ok_or_else(|| invalid(&format!("{key} is required")))?;
    let n = re.len();
    if n == 0
        || n > 64
        || size.is_some_and(|s| s != n)
        || re
            .iter()
            .any(|r| r.len() != n || r.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "operators require matching square real parts with size 1..64",
        ));
    }
    let im = controls
        .opt_rows(&format!("{key}_imag"))?
        .unwrap_or_else(|| vec![vec![0.0; n]; n]);
    if im.len() != n
        || im
            .iter()
            .any(|r| r.len() != n || r.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "imaginary parts must match their real parts and be finite",
        ));
    }
    let out = BlochMatrix {
        re: Matrix::from_row_major(n, n, re.into_iter().flatten().collect()),
        im: Matrix::from_row_major(n, n, im.into_iter().flatten().collect()),
    };
    if out.hermiticity_error() != 0.0 {
        return Err(invalid(
            "operators must be exactly Hermitian; no triangle or imaginary part is discarded",
        ));
    }
    Ok(out)
}

pub(super) fn rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect())
        .collect()
}

pub(super) fn run(controls: &Controls) -> Result<Value> {
    if controls.contains("reference") {
        return Err(invalid(
            "a supplied manifold does not accept an SCF reference",
        ));
    }
    let h = operator(controls, "mcd_thermal_hamiltonian", None)?;
    let n = h.rows();
    let mut value = Value::object()
        .with("temperature", controls.f64("temperature", 0.0)?)
        .with(
            "degeneracy_tolerance",
            controls.f64("degeneracy_tolerance", 1e-12)?,
        )
        .with(
            "bfield",
            controls
                .opt_vec3("bfield")?
                .ok_or_else(|| invalid("bfield is required"))?
                .to_array(),
        )
        .with("mcd_thermal_hamiltonian", rows(&h.real_embedding()));
    for key in [
        "mcd_thermal_magnetic_x",
        "mcd_thermal_magnetic_y",
        "mcd_thermal_magnetic_z",
        "mcd_thermal_helicity_x",
        "mcd_thermal_helicity_y",
        "mcd_thermal_helicity_z",
    ] {
        value.set(
            key,
            rows(&operator(controls, key, Some(n))?.real_embedding()),
        );
    }
    controls.finish("mcd thermal-complex-manifold")?;
    // The real representation duplicates every physical eigenvalue. Both
    // numerator traces and partition sum duplicate, so expectations and all
    // response tensors are unchanged, including degenerate coherences.
    let mut report = super::thermal::run(&Controls::from_value(&value)?)?;
    for key in [
        "mcd_thermal_energies",
        "mcd_zero_field_energies",
        "mcd_thermal_relative_energies",
        "mcd_zero_field_relative_energies",
        "mcd_thermal_populations",
        "mcd_zero_field_populations",
    ] {
        let data = report
            .get(key)
            .and_then(Value::numeric_array)
            .ok_or_else(|| invalid("missing embedded spectrum"))?
            .1;
        let population = key.ends_with("populations");
        report.set(
            key,
            data.chunks_exact(2)
                .map(|v| {
                    if population {
                        v[0] + v[1]
                    } else {
                        0.5 * v[0] + 0.5 * v[1]
                    }
                })
                .collect::<Vec<_>>(),
        );
    }
    report.set("mcd_model", "thermal-complex-manifold");
    report.set("electronic_states", n);
    report.set("model","canonical ensemble of explicitly supplied complex Hermitian electronic sublevel Hamiltonian, magnetic and normalized helicity operators; no generated molecular states");
    report.set("complex_operator_convention","Each base matrix contains its real part; the optional matching _imag matrix contains its imaginary part. Both are retained. Exact Hermiticity is required. Unitary basis transformations must be applied to every operator together. The doubled real representation has no extra physical states.");
    report.set("warnings",vec!["Supplied effective manifold only; no generated molecular states or absolute optical spectrum. General complex Hermitian operators are retained without a real projection."]);
    let mut units = report
        .get("units")
        .cloned()
        .ok_or_else(|| invalid("missing units"))?;
    units.set("electronic_states", "dimensionless");
    report.set("units", units);
    Ok(report)
}
