// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};

fn number(value: &Value, key: &str) -> f64 {
    value.get(key).unwrap().as_f64().unwrap()
}

fn rows(value: &Value) -> Vec<f64> {
    match value {
        Value::Array(entries) => entries.iter().flat_map(rows).collect(),
        value => vec![value.as_f64().unwrap()],
    }
}

#[test]
fn source_free_ammonia_all_magnetic_quadrupole_components() {
    // Deliberately distorted C1 ammonia with unequal N-H bond lengths.
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.13, -0.27, 0.19],
            [1.78, 0.34, -0.08],
            [-0.49, 1.65, 0.41],
            [0.31, -0.78, 1.91],
        ],
    );
    let origin = [0.23, -0.31, 0.39];
    let field = [0.014, -0.019, 0.023];
    let gradient = [
        [0.002, 0.003, -0.002],
        [0.003, -0.001, 0.0015],
        [-0.002, 0.0015, -0.001],
    ];
    let evaluate = |gradient| {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "wb97x-d")
                .with("grid_level", 0usize)
                .with("reference", "gks")
                .with("origin", origin)
                .with("magnetic_field", field)
                .with("magnetic_field_gradient", gradient)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-12)
                .with("conv_tol_grad", 1e-10)
                .with("conv_tol_dp", 1e-10),
        )
        .unwrap();
        run("magnetic-quadrupole", &system, &controls).unwrap()
    };
    let center = evaluate(gradient);
    let analytic = rows(center.get("magnetic_quadrupole_components").unwrap());
    let modes = rows(center.get("magnetic_quadrupole_modes").unwrap());
    println!(
        "magnetic ammonia orbital {:?} spin {:?} total {analytic:?} orbital residual {:.12e}",
        rows(center.get("orbital_quadrupole").unwrap()),
        rows(center.get("spin_quadrupole").unwrap()),
        number(&center, "scf_orbital_residual")
    );
    let names: Vec<_> = (0..5)
        .map(|i| format!("ammonia magnetic quadrupole mode {i}"))
        .collect();
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.009, 0.006, 0.004, 0.0027], stencils::Stencil::Second),
            ([0.004, 0.0028, 0.0019, 0.0013], stencils::Stencil::Second),
            ([0.012, 0.009, 0.007, 0.005], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 5];
            for (k, &h) in steps.iter().enumerate() {
                for a in 0..5 {
                    let shift = |scale: f64| {
                        std::array::from_fn(|i| {
                            std::array::from_fn(|j| {
                                gradient[i][j] + scale * modes[9 * a + 3 * i + j]
                            })
                        })
                    };
                    let plus = number(&evaluate(shift(h)), "energy");
                    let minus = number(&evaluate(shift(-h)), "energy");
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (
                            number(&evaluate(shift(2.0 * h)), "energy"),
                            number(&evaluate(shift(-2.0 * h)), "energy"),
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    errors[a][k] =
                        (stencil.first(plus, minus, plus2, minus2, h) + analytic[a]).abs();
                }
            }
            errors
        },
        &mut failed,
    );
    components::assert_no_failures("ammonia magnetic quadrupole", &failed);
}

#[test]
fn open_shell_no2_spin_quadrupole_origin_translation() {
    // Bent doublet with a representative gas-phase bond length and angle.
    let system = System::molecule(
        vec![7, 8, 8],
        vec![[0.0, 0.0, 0.0], [2.26, 0.0, 0.0], [-1.51, 1.68, 0.0]],
    );
    let evaluate = |origin| {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "wb97x-v")
                .with("grid_level", 0usize)
                .with("vv10_grid_level", 0usize)
                .with("reference", "uks")
                .with("multiplicity", 2)
                .with("origin", origin)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol_grad", 1e-9),
        )
        .unwrap();
        run("magnetic-quadrupole", &system, &controls).unwrap()
    };
    let delta = [0.34, -0.27, 0.19];
    let original = evaluate([0.0; 3]);
    let displaced = evaluate(delta);
    let a = rows(original.get("spin_quadrupole").unwrap());
    let b = rows(displaced.get("spin_quadrupole").unwrap());
    let m = rows(original.get("magnetic_dipole").unwrap());
    assert!(a.iter().map(|x| x.abs()).fold(0.0, f64::max) > 0.05);
    let trace: f64 = m.iter().zip(delta).map(|(x, y)| x * y).sum();
    let mut defect = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let expected = a[3 * i + j] - 0.5 * (m[i] * delta[j] + m[j] * delta[i])
                + if i == j { trace / 3.0 } else { 0.0 };
            defect = defect.max((b[3 * i + j] - expected).abs());
        }
    }
    println!("NO2 spin quadrupole {a:?}, magnetic dipole {m:?}, origin covariance {defect:.12e}");
    assert!(defect < 2e-10);
}

#[test]
fn open_shell_no2_current_invariant_meta_gga_zero_field_limit() {
    let system = System::molecule(
        vec![7, 8, 8],
        vec![[0.0, 0.0, 0.0], [2.26, 0.0, 0.0], [-1.51, 1.68, 0.0]],
    );
    for basis in ["sto-3g", "6-31g"] {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "wb97m-v")
                .with("basis", basis)
                .with("reference", "uks")
                .with("multiplicity", 2)
                .with("grid_level", 0usize)
                .with("vv10_grid_level", 0usize)
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol_grad", 1e-8),
        )
        .unwrap();
        let report = run("magnetic-quadrupole", &system, &controls).unwrap();
        let orbital = rows(report.get("orbital_quadrupole").unwrap());
        let spin = rows(report.get("spin_quadrupole").unwrap());
        let dipole = rows(report.get("magnetic_dipole").unwrap());
        println!(
            "NO2 wb97m-v {basis} spin Q={spin:?} m={dipole:?} orbital residual={:.12e}",
            number(&report, "scf_orbital_residual")
        );
        assert!(orbital.iter().all(|x| *x == 0.0));
        assert!((dipole[2] + 0.5).abs() < 1e-8);
        assert!(spin.iter().map(|x| x.abs()).fold(0.0, f64::max) > 0.01);
    }
}
