// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent relaxed-energy checks of the ordered correlated stress.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{run, Controls, System, Value};
use std::collections::BTreeMap;
use stencils::Stencil;

fn calculate(feature: &str, system: &System, options: &Value) -> Value {
    let report = run(feature, system, &Controls::from_value(options).unwrap()).unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    report
}

fn scalar(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn tensor(report: &Value, key: &str) -> Vec<f64> {
    let (shape, data) = report.get(key).unwrap().numeric_array().unwrap();
    assert_eq!(shape, [3, 3]);
    data
}

#[test]
fn charged_double_hybrid_fits_all_ordered_strain_components() {
    let system = System {
        numbers: vec![2, 1],
        positions: vec![[0.17, 0.41, -0.19], [0.49, -0.12, 1.19]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let options = Value::object()
        .with("xc", "b2plyp")
        .with("reference", "rks")
        .with("charge", 1)
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("forces", false)
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 5.5)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("response_tolerance", 1e-11);
    fit_stress(&system, &options, "charged B2PLYP RKS", false);
}

#[test]
fn charged_unrestricted_double_hybrid_fits_all_spin_resolved_strain_components() {
    // Three electrons provide a nonzero alpha-alpha pair channel and one
    // beta electron whose same-spin contribution must vanish identically.
    let system = System {
        numbers: vec![1, 1, 1, 1],
        positions: vec![
            [0.17, 0.41, -0.19],
            [0.49, -0.12, 1.19],
            [1.71, 0.28, 0.33],
            [1.39, 1.62, 1.47],
        ],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let options = Value::object()
        .with("xc", "b2plyp")
        .with("reference", "uks")
        .with("charge", 1)
        .with("multiplicity", 2)
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("forces", false)
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 5.5)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("response_tolerance", 1e-11)
        .with("max_iter", 500);
    fit_stress(&system, &options, "charged B2PLYP UKS", true);
}

#[test]
fn finite_cartesian_bias_double_hybrid_fits_all_ordered_strain_components() {
    let system = System {
        numbers: vec![3, 1],
        positions: vec![[0.17, 0.41, -0.19], [1.09, -0.82, 2.45]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let options = Value::object()
        .with("xc", "b2plyp")
        .with("reference", "rks")
        .with(
            "basis",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/periodic_lih.g94"),
        )
        .with("dispersion", "none")
        .with("forces", false)
        .with("field", [0.0013, -0.0021, 0.0017])
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 5.5)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("response_tolerance", 1e-11);
    // Each independent energy is rebuilt on its deformed lattice with
    // exactly the same Cartesian field and the same local Berry branch.
    fit_stress(&system, &options, "finite Cartesian bias B2PLYP RKS", false);
}

#[test]
fn scalar_ecp_double_hybrid_fits_every_nuclear_and_ordered_strain_component() {
    let system = System {
        numbers: vec![1, 11, 1],
        positions: vec![[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]],
        cell: Some([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]),
        pbc: [true; 3],
    };
    let basis = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/periodic_scalar_ecp.g94"
    );
    let options = Value::object()
        .with("xc", "b2plyp")
        .with("reference", "rks")
        .with("charge", 1)
        .with("basis", basis)
        .with("dispersion", "none")
        .with("forces", true)
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 5.5)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("response_tolerance", 1e-11);
    let report = calculate("stress", &system, &options);
    assert_eq!(scalar(&report, "n_alpha"), 1.0);
    assert_eq!(scalar(&report, "n_beta"), 1.0);
    let forces = tensor(&report, "forces");
    let strain = tensor(&report, "strain_derivative");
    let stress = tensor(&report, "stress");
    let volume = scalar(&report, "volume");
    for i in 0..9 {
        assert!(
            (stress[i] - (strain[i] + strain[(i % 3) * 3 + i / 3]) / (2.0 * volume)).abs() < 1e-15
        );
    }
    let analytic: Vec<_> = forces
        .iter()
        .map(|f| -f)
        .chain(strain.iter().copied())
        .collect();
    let labels: Vec<_> = (0..18)
        .map(|i| format!("charged scalar ECP B2PLYP derivative {i}"))
        .collect();
    let energy_options = options.clone().with("forces", false);
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    let mut richardson = [f64::INFINITY; 18];
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.01, 0.007, 0.005, 0.0035], Stencil::Fourth),
        ],
        |steps, stencil| {
            (0..18)
                .map(|i| {
                    let mut sample = |h: f64| {
                        *samples.entry((i, h.to_bits())).or_insert_with(|| {
                            let mut moved = system.clone();
                            if i < 9 {
                                moved.positions[i / 3][i % 3] += h;
                            } else {
                                let j = i - 9;
                                for row in moved
                                    .positions
                                    .iter_mut()
                                    .chain(moved.cell.as_mut().unwrap())
                                {
                                    row[j / 3] += h * row[j % 3];
                                }
                            }
                            scalar(
                                &calculate("periodic-energy", &moved, &energy_options),
                                "energy",
                            )
                        })
                    };
                    let values = steps.map(|h| {
                        let (plus, minus) = (sample(h), sample(-h));
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (sample(2.0 * h), sample(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        stencil.first(plus, minus, plus2, minus2, h)
                    });
                    let coarse = steps[2].powf(stencil.order());
                    let fine = steps[3].powf(stencil.order());
                    let defect = ((coarse * values[3] - fine * values[2]) / (coarse - fine)
                        - analytic[i])
                        .abs();
                    richardson[i] = richardson[i].min(defect);
                    println!(
                        "{}: analytic={:.16e} Richardson defect={defect:.9e}",
                        labels[i], analytic[i]
                    );
                    values.map(|v| (v - analytic[i]).abs())
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("charged scalar ECP double-hybrid derivatives", &failures);
    assert!(richardson.iter().all(|&error| error < 2e-6));
    for axis in 0..3 {
        assert!(
            (0..3)
                .map(|atom| forces[3 * atom + axis])
                .sum::<f64>()
                .abs()
                < 2e-8
        );
    }
    println!("Scalar ECP B2PLYP energy={:.16e} forces={forces:?} strain={strain:?} stress={stress:?} series residual={:.9e}",
        scalar(&report, "energy"), scalar(&report, "series_reference_residual"));
}

fn fit_stress(system: &System, options: &Value, label: &str, alpha_pairs: bool) {
    let report = calculate("stress", system, options);
    if options.get("field").is_some() {
        assert_eq!(
            report
                .get("stress_field_convention")
                .and_then(Value::as_str),
            Some("fixed Cartesian Berry electric field")
        );
    }
    let strain = tensor(&report, "strain_derivative");
    let stress = tensor(&report, "stress");
    let terms = report.get("strain_derivative_terms").unwrap();
    let target = tensor(terms, "target_density_functional");
    let os = tensor(terms, "pt2_opposite_spin");
    let ss_alpha = tensor(terms, "pt2_same_spin_alpha");
    let ss_beta = tensor(terms, "pt2_same_spin_beta");
    let volume = scalar(&report, "volume");
    if options.get("xc_ao_cutoff").is_some() {
        // Establish AO image-range convergence separately from the step fit.
        // The original 5.5-Bohr range can move appreciable AO tails across
        // the hard cutoff as the lattice changes.
        let wider = calculate(
            "stress",
            system,
            &options.clone().with("xc_ao_cutoff", 20.0),
        );
        let energy_defect = (scalar(&wider, "energy") - scalar(&report, "energy")).abs();
        let strain_defect = tensor(&wider, "strain_derivative")
            .iter()
            .zip(&strain)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("{label} XC AO cutoff 16 -> 20 Bohr: energy defect={energy_defect:.9e} strain defect={strain_defect:.9e}");
        assert!(energy_defect < 1e-10 && strain_defect < 1e-9);
    }
    for a in 0..3 {
        for b in 0..3 {
            let i = 3 * a + b;
            let expected = (strain[i] + strain[3 * b + a]) / (2.0 * volume);
            assert!((stress[i] - expected).abs() < 1e-15);
            assert!((strain[i] - target[i] - os[i] - ss_alpha[i] - ss_beta[i]).abs() < 1e-14);
        }
    }
    assert!(os.iter().any(|v| v.abs() > 1e-6));
    println!(
        "{label} stress energy={:.16e} volume={volume:.16e} strain={strain:?} stress={stress:?} target={target:?} os={os:?} rotation={:.6e} reference={:.6e}",
        scalar(&report, "energy"),
        scalar(&report, "rotation_residual"),
        scalar(&report, "series_reference_residual")
    );

    let mut channels = vec!["total", "target", "weighted opposite spin"];
    let mut derivatives = vec![&strain, &target, &os];
    if alpha_pairs {
        assert!(ss_alpha.iter().any(|v| v.abs() > 1e-12));
        channels.push("weighted alpha same spin");
        derivatives.push(&ss_alpha);
        assert!(ss_beta.iter().all(|v| v.abs() < 1e-15));
    } else {
        // The RKS fixtures have at most one virtual orbital, or at most
        // one occupied orbital. Neither spin admits a same-spin pair.
        assert!(ss_alpha.iter().chain(&ss_beta).all(|v| v.abs() < 1e-15));
    }
    let labels: Vec<_> = channels
        .iter()
        .flat_map(|term| {
            (0..9).map(move |i| format!("{label} {term} strain [{},{}]", i / 3, i % 3))
        })
        .collect();
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    let mut richardson_defects = vec![f64::INFINITY; labels.len()];
    let mut intervals = vec![
        ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
        ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
        ([0.01, 0.007, 0.005, 0.0035], Stencil::Fourth),
    ];
    if options.get("field").is_some() {
        // Keep the coarser attempts. The signed PT2 strain error on this
        // fixture resolves again on the four consecutive finer steps.
        intervals.push(([0.003, 0.002, 0.0015, 0.001], Stencil::Second));
    }
    stencils::fit_components_staged(
        &labels,
        &intervals,
        |steps, stencil| {
            (0..labels.len())
                .map(|component| {
                    let (term, i) = (component / 9, component % 9);
                    let mut sample = |step: f64| {
                        samples.entry((i, step.to_bits())).or_insert_with(|| {
                            let mut displaced = system.clone();
                            for v in displaced
                                .positions
                                .iter_mut()
                                .chain(displaced.cell.as_mut().unwrap())
                            {
                                v[i / 3] += step * v[i % 3];
                            }
                            let relaxed = calculate("periodic-energy", &displaced, options);
                            let terms = relaxed.get("energy_components").unwrap();
                            [
                                scalar(&relaxed, "energy"),
                                scalar(terms, "target_density_functional"),
                                scalar(terms, "pt2_opposite_spin"),
                                scalar(terms, "pt2_same_spin_alpha"),
                            ]
                        })[term]
                    };
                    let values = steps.map(|h| {
                        let (plus, minus) = (sample(h), sample(-h));
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (sample(2.0 * h), sample(-2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        stencil.first(plus, minus, plus2, minus2, h)
                    });
                    let coarse = steps[2].powf(stencil.order());
                    let fine = steps[3].powf(stencil.order());
                    let analytic = derivatives[term][i];
                    let defect = ((coarse * values[3] - fine * values[2]) / (coarse - fine)
                        - analytic)
                        .abs();
                    richardson_defects[component] = richardson_defects[component].min(defect);
                    println!("{}: Richardson defect={defect:.6e}", labels[component]);
                    values.map(|v| (v - analytic).abs())
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures(label, &failures);
    assert!(richardson_defects.iter().all(|&error| error < 2e-6));
}
