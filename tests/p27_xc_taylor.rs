// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dual::Scalar;
use dft_rs::response::series::evaluate_series;
use dft_rs::xc::{
    resolve, FunctionalSpec, GridVars, XcOut, XcTaylorInput, COMBINED_NAMES, CORRELATION_NAMES,
    EXCHANGE_NAMES,
};

fn close(a: f64, b: f64, label: &str) {
    assert!(
        (a - b).abs() < 2e-10 * (1.0 + a.abs() + b.abs()),
        "{label}: {a} != {b}"
    );
}

#[test]
fn every_registered_semilocal_body_matches_dense_derivatives() {
    let values = [0.3, 0.2, 0.02, 0.003, 0.01, 0.4, 0.3];
    let inputs = XcTaylorInput::linear(values, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6).unwrap();
    let specs: Vec<_> = COMBINED_NAMES
        .iter()
        .map(|name| FunctionalSpec::Combined((*name).into()))
        .chain(EXCHANGE_NAMES.iter().map(|name| FunctionalSpec::Split {
            x: (*name).into(),
            c: "vwn5".into(),
        }))
        .chain(CORRELATION_NAMES.iter().map(|name| FunctionalSpec::Split {
            x: "slater".into(),
            c: (*name).into(),
        }))
        .collect();
    let arrays = values.map(|value| [value]);
    let vars = GridVars {
        rho: [&arrays[0], &arrays[1]],
        sigma: Some([&arrays[2], &arrays[3], &arrays[4]]),
        tau: Some([&arrays[5], &arrays[6]]),
    };
    for spec in specs {
        let functional = resolve(&spec).unwrap();
        let series = functional
            .semilocal_taylor_energy(&inputs)
            .unwrap_or_else(|e| panic!("{}: {e}", functional.description));
        let mut dense = XcOut::default();
        functional.eval(&vars, 3, &mut dense).unwrap();
        let potential_series = functional.semilocal_taylor_potentials(&inputs).unwrap();
        for (a, b) in potential_series.energy.iter().zip(&series) {
            close(*a, *b, &functional.description);
        }
        for (a, b) in [
            potential_series.potentials[0][0],
            potential_series.potentials[0][1],
            2.0 * potential_series.potentials[0][2],
        ]
        .into_iter()
        .zip([dense.vrho[0][0], dense.v2rho2[0][0], dense.v3rho3[0][0]])
        {
            close(a, b, &functional.description);
        }
        for (a, b) in [series[0], series[1], 2.0 * series[2], 6.0 * series[3]]
            .into_iter()
            .zip([
                dense.e[0],
                dense.vrho[0][0],
                dense.v2rho2[0][0],
                dense.v3rho3[0][0],
            ])
        {
            close(a, b, &functional.description);
        }
        assert!(series.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn sixth_order_exchange_and_mixed_parameter_response_have_closed_forms() {
    let f = dft_rs::xc::lda::slater();
    let input = XcTaylorInput::linear(
        [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        8,
    )
    .unwrap();
    let c = f.taylor_energy(&input).unwrap();
    let mut falling = 1.0;
    let mut factorial = 1.0;
    let ra = input.variables[0][0];
    let prefactor = -0.75 * (6.0 / std::f64::consts::PI).cbrt();
    for n in 1..=8 {
        falling *= 4.0 / 3.0 - (n - 1) as f64;
        factorial *= n as f64;
        close(
            c[n],
            prefactor * falling * ra.powf(4.0 / 3.0 - n as f64) / factorial,
            "Slater closed form",
        );
    }
    let f = dft_rs::xc::gga::pbe_x();
    let mut input = XcTaylorInput::linear(
        [0.3, 0.2, 0.02, 0.003, 0.01, 0.4, 0.3],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        5,
    )
    .unwrap();
    let defaults = f.params();
    input.parameters = Some(
        defaults
            .iter()
            .map(|&p| vec![p, 0.02, 0.0, 0.0, 0.0, 0.0])
            .collect(),
    );
    let coefficients = f.taylor_energy(&input).unwrap();
    // Independent finite difference changes density and all parameters together.
    let energy = |t: f64| {
        let params: Vec<_> = defaults.iter().map(|p| p + 0.02 * t).collect();
        let shifted = f.with_params(&params).unwrap();
        let point = XcTaylorInput::linear(
            [0.3 + 0.1 * t, 0.2, 0.02, 0.003, 0.01, 0.4, 0.3],
            [0.0; 7],
            0,
        )
        .unwrap();
        shifted.taylor_energy(&point).unwrap()[0]
    };
    let h = 1e-3;
    let first =
        (energy(-2.0 * h) - 8.0 * energy(-h) + 8.0 * energy(h) - energy(2.0 * h)) / (12.0 * h);
    close(
        coefficients[1],
        first,
        "mixed density and parameter direction",
    );
}

#[test]
fn runtime_series_handles_nonlinear_inputs_threads_and_unwinding() {
    let input = vec![vec![0.2, 0.3, -0.1, 0.04, 0.0, 0.0, 0.0]];
    let series = evaluate_series(&input, |x| x[0].exp().ln()).unwrap();
    for (a, b) in series.iter().zip(&input[0]) {
        close(*a, *b, "exp/ln composition");
    }
    evaluate_series(&input, |x| {
        assert!(evaluate_series(&input, |x| x[0]).is_err());
        x[0]
    })
    .unwrap();
    assert!(
        std::panic::catch_unwind(|| evaluate_series(&input, |_| panic!("scope reset test")))
            .is_err()
    );
    assert!(evaluate_series(&input, |x| x[0]).is_ok());
    let workers: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                evaluate_series(&[vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], |x| x[0].exp()).unwrap()
            })
        })
        .collect();
    for worker in workers {
        close(
            worker.join().unwrap()[5],
            1.0 / 120.0,
            "thread isolated arena",
        );
    }
}
