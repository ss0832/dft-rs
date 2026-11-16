// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full force-constant, commensurate folding and oscillator checks for the
//! periodic double-hybrid phonon spectrum.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::constants::{atomic_mass_au, HARTREE_TO_CM1};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
use std::collections::BTreeMap;
use stencils::Stencil;

const POSITIONS: [[f64; 3]; 2] = [[0.17, 0.41, -0.19], [0.49, -0.12, 1.19]];
const CELL: [[f64; 3]; 3] = [[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]];

fn crystal() -> System {
    System {
        numbers: vec![1, 1],
        positions: POSITIONS.to_vec(),
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

fn electronic() -> Value {
    Value::object()
        .with("xc", "b2plyp")
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("overlap_cutoff", 5.5)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("response_tolerance", 1e-11)
}

fn value(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn array(report: &Value, key: &str) -> (Vec<usize>, Vec<f64>) {
    report.get(key).unwrap().numeric_array().unwrap()
}

fn calculate(feature: &str, system: &System, options: &Value) -> Value {
    let report = run(feature, system, &Controls::from_value(options).unwrap())
        .unwrap_or_else(|error| panic!("{feature} failed: {error}"));
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    assert!(value(&report, "series_reference_residual").abs() < 1e-9);
    report
}

#[test]
fn correlated_phonon_dos_and_thermo_recover_the_supercell_modes() {
    let system = crystal();
    let options = electronic()
        .with("supercell", [2, 1, 1])
        .with("k_mesh", [2, 1, 1]);
    let bands = calculate(
        "phonon-bands",
        &system,
        &options
            .clone()
            .with("path", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
            .with("points", 2)
            .with("dos_mesh", [2, 1, 1])
            .with("dos_sigma", 5.0)
            .with("dos_points", 2048),
    );
    let temperatures = [0.0, 300.0, 600.0];
    let thermo = calculate(
        "phonon-thermo",
        &system,
        &options
            .with("q_mesh", [2, 1, 1])
            .with("temperatures", temperatures),
    );
    let (shape, hessian) = array(&bands, "supercell_force_constants");
    assert_eq!(shape, [12, 12]);
    assert_eq!(
        array(&bands, "supercell_force_constants"),
        array(&thermo, "supercell_force_constants")
    );
    assert_eq!(
        array(&bands, "born_von_karman_supercell").1,
        [2.0, 1.0, 1.0]
    );
    assert_eq!(array(&bands, "supercell_k_mesh").1, [1.0, 1.0, 1.0]);
    assert!(value(&bands, "acoustic_residual") < 1e-8);
    assert!(value(&bands, "dynamical_matrix_hermiticity") < 1e-8);
    assert!(value(&bands, "pt2_opposite_spin").abs() > 1e-6);
    assert_eq!(bands.get("force_constant_asymmetry"), Some(&Value::Null));
    let (shape, frequencies) = array(&thermo, "mesh_frequencies_cm1");
    assert_eq!(shape, [2, 6]);
    let (_, path) = array(&bands, "frequencies_cm1");
    assert_eq!(&path[..6], &frequencies[..6]);
    assert_eq!(&path[path.len() - 6..], &frequencies[6..]);

    // Diagonalize the whole supercell independently of the Fourier fold.
    let mut weighted = Matrix::zeros(12, 12);
    for i in 0..12 {
        for j in 0..12 {
            let mass_i = atomic_mass_au(system.numbers[(i / 3) % 2]);
            let mass_j = atomic_mass_au(system.numbers[(j / 3) % 2]);
            weighted[(i, j)] = hessian[12 * i + j] / (mass_i * mass_j).sqrt();
        }
    }
    let (mut direct, _) = symmetric_eigen(&weighted).unwrap();
    direct.sort_by(f64::total_cmp);
    let mut folded: Vec<_> = frequencies
        .iter()
        .map(|f| f.signum() * (f / HARTREE_TO_CM1).powi(2))
        .collect();
    folded.sort_by(f64::total_cmp);
    let eigenvalue_defect = direct
        .iter()
        .zip(&folded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(eigenvalue_defect < 1e-10);

    // Integrate the reported DOS on its finite grid. Its two tails extend to
    // at least five Gaussian widths beyond all modes.
    let (_, grid) = array(&bands, "dos_frequencies_cm1");
    let (_, dos) = array(&bands, "dos");
    let state_count: f64 = grid
        .windows(2)
        .zip(dos.windows(2))
        .map(|(x, y)| (x[1] - x[0]) * 0.5 * (y[0] + y[1]))
        .sum();
    assert!((state_count - 6.0).abs() < 5e-6);

    // Compute the partition function from the dense spectrum with the same
    // positive-mode cutoff and normalize by the two primitive cells.
    let modes: Vec<_> = direct
        .iter()
        .filter(|&&x| x > 1.0 / HARTREE_TO_CM1.powi(2))
        .map(|x| x.sqrt())
        .collect();
    assert!(!modes.is_empty());
    let zpe = 0.25 * modes.iter().sum::<f64>();
    let (_, free) = array(&thermo, "free_energy");
    let (_, internal) = array(&thermo, "internal_energy");
    let (_, entropy) = array(&thermo, "entropy");
    let (_, capacity) = array(&thermo, "heat_capacity");
    assert!((value(&thermo, "zero_point_energy") - zpe).abs() < 1e-7);
    for (i, temperature) in temperatures.into_iter().enumerate() {
        let thermal_energy = BOLTZMANN_HARTREE_PER_K * temperature;
        let mut expected = [zpe, zpe, 0.0];
        if temperature > 0.0 {
            for &mode in &modes {
                let x = mode / thermal_energy;
                let n = x.exp_m1().recip();
                expected[0] -= 0.5 * thermal_energy * n.ln_1p();
                expected[1] += 0.5 * mode * n;
                expected[2] += 0.5 * BOLTZMANN_HARTREE_PER_K * x.powi(2) * n * (1.0 + n);
            }
        }
        assert!((expected[0] - free[i]).abs() < 1e-7);
        assert!((expected[1] - internal[i]).abs() < 1e-7);
        assert!((expected[2] - capacity[i]).abs() < 1e-8);
        assert!((free[i] + temperature * entropy[i] - internal[i]).abs() < 1e-13);
    }
    println!("correlated phonon spectrum: energy={:.15e} modes={frequencies:?} count={state_count:.15e} dense_eigenvalue_defect={eigenvalue_defect:.3e} acoustic={:.3e} zpe={zpe:.15e} F={free:?} U={internal:?} S={entropy:?} Cv={capacity:?} imaginary={}",
        value(&bands, "energy"), value(&bands, "acoustic_residual"), value(&thermo, "imaginary_modes"));
}

#[test]
fn correlated_gamma_phonon_spectrum_fits_every_force_constant() {
    // Unequal nuclear charges at generic centres remove the inversion of
    // the neutral H2 spectrum fixture.
    let mut system = crystal();
    system.numbers = vec![3, 1];
    system.positions[1] = [1.09, -0.82, 2.45];
    let basis = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/periodic_lih.g94");
    let controls = electronic().with("basis", basis);
    fit_gamma_force_constants(&system, controls, false);
}

#[test]
fn charged_correlated_gamma_phonon_spectrum_fits_every_force_constant() {
    let mut system = crystal();
    system.numbers = vec![2, 1];
    fit_gamma_force_constants(&system, electronic().with("charge", 1), true);
}

#[test]
fn scalar_ecp_correlated_gamma_phonon_fits_every_force_constant() {
    let system = System {
        numbers: vec![1, 11, 1],
        positions: vec![[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]],
        cell: Some(CELL),
        pbc: [true; 3],
    };
    let basis = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/periodic_scalar_ecp.g94"
    );
    let controls = electronic()
        .with("basis", basis)
        .with("charge", 1)
        .with("xc_ao_cutoff", 16.0);
    fit_gamma_force_constants(&system, controls, true);
}

fn fit_gamma_force_constants(system: &System, controls: Value, charged: bool) {
    let n = 3 * system.numbers.len();
    let bands = calculate(
        "phonon-bands",
        system,
        &controls
            .clone()
            .with("supercell", [1, 1, 1])
            .with("path", [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
            .with("points", 2),
    );
    if charged {
        assert_eq!(
            bands.get("charge_compensation").and_then(Value::as_str),
            Some("uniform 3D Ewald background")
        );
        for key in [
            "primitive_cell_charge",
            "force_constant_supercell_charge",
            "born_von_karman_charge",
        ] {
            assert_eq!(value(&bands, key), 1.);
        }
    }
    println!(
        "correlated Gamma charged={charged} energy={:.17e} acoustic={:.17e} pt2_ss_alpha={:.17e} pt2_os={:.17e} pt2_ss_beta={:.17e} series_residual={:.17e}",
        value(&bands, "energy"),
        value(&bands, "acoustic_residual"),
        value(&bands, "pt2_same_spin_alpha"),
        value(&bands, "pt2_opposite_spin"),
        value(&bands, "pt2_same_spin_beta"),
        value(&bands, "series_reference_residual")
    );
    assert!(value(&bands, "acoustic_residual") < 1e-8);
    let (shape, hessian) = array(&bands, "supercell_force_constants");
    assert_eq!(shape, [n, n]);
    // One signed homogeneous path for each diagonal and two for each
    // independent off-diagonal. Opposite paths must not solve twice.
    assert_eq!(value(&bands, "path_evaluations"), (n * n) as f64);
    println!("correlated Gamma charged={charged} Hessian={hessian:?}");
    let force_controls = controls.with("forces", true);
    let components: Vec<_> = (0..n).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    let labels: Vec<_> = components
        .iter()
        .map(|(a, b)| format!("B2PLYP phonon Hessian [{a},{b}]"))
        .collect();
    let mut failures = Vec::new();
    // Overlapping stencil intervals can share the same independent force
    // evaluation. Keep each distinct displacement and step sign separate.
    let mut samples = BTreeMap::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; components.len()];
            for (i, &h) in steps.iter().enumerate() {
                for b in 0..n {
                    let mut sample = |step: f64| {
                        samples
                            .entry((b, step.to_bits()))
                            .or_insert_with(|| {
                                let mut displaced = system.clone();
                                displaced.positions[b / 3][b % 3] += step;
                                array(
                                    &calculate("periodic-energy", &displaced, &force_controls),
                                    "forces",
                                )
                                .1
                            })
                            .clone()
                    };
                    let (plus, minus) = (sample(h), sample(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (sample(2.0 * h), sample(-2.0 * h))
                    } else {
                        (vec![0.0; n], vec![0.0; n])
                    };
                    for (k, &(a, column)) in components
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, column))| *column == b)
                    {
                        debug_assert_eq!(column, b);
                        let estimate = -stencil.first(plus[a], minus[a], plus2[a], minus2[a], h);
                        errors[k][i] = (estimate - hessian[n * a + b]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("correlated phonon spectrum", &failures);
}
