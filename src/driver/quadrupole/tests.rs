// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::{components, stencils};
use crate::driver::double_hybrid::moment_tests::{
    correlated_controls, perturbation, system, COMPONENTS, EXPONENT, ORIGIN,
};
use crate::driver::run;
use crate::integrals::london::ComplexMatrix;
use stencils::Stencil;

fn values(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn solve(id: &str, system: &System, controls: &Value) -> Value {
    let report = run(id, system, &Controls::from_value(controls).unwrap())
        .unwrap_or_else(|error| panic!("{id} {controls:?}: {error}"));
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    report
}

fn gaussian_sources(system: &System) -> [(Matrix, f64); 5] {
    let terms = COMPONENTS.map(|(a, b)| perturbation(system, a, b));
    modes().map(|mode| {
        let mut operator = Matrix::zeros(system.numbers.len(), system.numbers.len());
        let mut ionic = 0.0;
        for (index, (a, b)) in COMPONENTS.into_iter().enumerate() {
            let weight = (mode[a][b] + if a == b { 0.0 } else { mode[b][a] }) / 2.0;
            let (source, charge) = &terms[index];
            for (out, value) in operator.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *out += weight * value;
            }
            ionic += weight * charge;
        }
        (operator, ionic)
    })
}

fn gaussian_racah_sources(system: &System, rank: usize) -> Vec<(Matrix, f64)> {
    let powers = crate::basis::cart_components(rank as u8);
    spherical_weights(rank)
        .into_iter()
        .map(|weights| {
            let harmonic = |r: [f64; 3]| {
                weights
                    .iter()
                    .zip(&powers)
                    .map(|(w, &(x, y, z))| w * r[0].powi(x) * r[1].powi(y) * r[2].powi(z))
                    .sum::<f64>()
            };
            let n = system.positions.len();
            let mut operator = Matrix::zeros(n, n);
            for (i, a) in system.positions.iter().enumerate() {
                for (j, b) in system.positions.iter().enumerate() {
                    let distance2 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
                    let centre = std::array::from_fn(|k| 0.5 * (a[k] + b[k]) - ORIGIN[k]);
                    // An isotropic Gaussian averages a harmonic polynomial
                    // to its value at the Gaussian mean, times the overlap.
                    operator[(i, j)] = (-0.5 * EXPONENT * distance2).exp() * harmonic(centre);
                }
            }
            let ionic = system
                .positions
                .iter()
                .map(|r| harmonic(std::array::from_fn(|k| r[k] - ORIGIN[k])))
                .sum();
            (operator, ionic)
        })
        .collect()
}

/// Invert full-rank cyclic sources e_b + mixing*e_(b+1).
fn undo_cyclic_mixing(directional: &[f64], column: usize, mixing: f64) -> f64 {
    let count = directional.len();
    (0..count)
        .map(|k| (-mixing).powi(k as i32) * directional[(column + k) % count])
        .sum::<f64>()
        / (1.0 - (-mixing).powi(count as i32))
}

fn fixed_spinor(n: usize) -> (Matrix, Matrix) {
    let mut real = Matrix::zeros(2 * n, 2 * n);
    let mut imaginary = real.clone();
    for i in 0..n {
        let weight = (i + 1) as f64;
        real[(i, i)] = -0.013 * weight;
        real[(n + i, n + i)] = 0.013 * weight;
        real[(i, n + i)] = 0.007 * weight;
        real[(n + i, i)] = real[(i, n + i)];
        imaginary[(i, n + i)] = 0.009 * (n - i) as f64;
        imaginary[(n + i, i)] = -imaginary[(i, n + i)];
    }
    (real, imaginary)
}

fn fit_case(
    reference: &str,
    basis_path: &str,
    xc: Option<&str>,
    grid_level: usize,
    racah_rank: Option<usize>,
) {
    let system = system(reference != "rks");
    let mut options = Value::object()
        .with("basis", basis_path)
        .with("reference", reference)
        .with("guess", "core")
        .with("multiplicity", if reference == "rks" { 1 } else { 2 })
        .with("grid_level", grid_level)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-12)
        .with("conv_tol_grad", 1e-11)
        .with("max_iter", 300)
        .with("field", [0.0012, -0.0017, 0.0008]);
    if let Some(xc) = xc {
        options.set("xc", xc);
    } else {
        options.set(
            "xc_terms",
            vec![
                Value::object().with("xc", "hf").with("weight", 0.65),
                Value::object().with("xc", "svwn5").with("weight", 0.35),
            ],
        );
    }
    let n = system.numbers.len();
    let (fixed_real, fixed_imaginary) = fixed_spinor(n);
    if reference == "gks" {
        options.set("gks_operator_real", &fixed_real);
        options.set("gks_operator_imaginary", &fixed_imaginary);
    }
    let mut controls = options.clone().with("origin", ORIGIN);
    let (id, convention, sources) = if let Some(rank) = racah_rank {
        controls.set("multipole_rank", rank);
        (
            MULTIPOLE_ID,
            Convention::Racah(rank),
            gaussian_racah_sources(&system, rank),
        )
    } else {
        (
            ID,
            Convention::Buckingham,
            gaussian_sources(&system).to_vec(),
        )
    };
    let property = convention.property();
    let count = convention.count();
    let analytic = solve(id, &system, &controls);
    let tensor = values(&analytic, property);
    assert_eq!(tensor.len(), count * count);
    let label = format!(
        "{reference} {} {} rank {}",
        xc.unwrap_or("weighted"),
        id,
        convention.rank()
    );
    println!(
        "{label} electric quadrupole polarizability {}",
        analytic.to_json()
    );

    let mut fixed_setup = (reference != "gks").then(|| {
        let scf_controls = Controls::from_value(&options).unwrap();
        let _ = scf_controls.take("reference");
        setup(&system, &scf_controls).unwrap()
    });
    let kind = match reference {
        "rks" => CollinearPropertyReference::Rks,
        "uks" => CollinearPropertyReference::Uks,
        "roks" => CollinearPropertyReference::Roks,
        "gks" => CollinearPropertyReference::Auto,
        _ => unreachable!(),
    };
    let mixing: f64 = if racah_rank.is_some() { 0.23 } else { 0.0 };
    let mut sample = |column: usize, strength: f64| -> Vec<f64> {
        let mut operator = sources[column].0.clone();
        if mixing != 0.0 {
            let other = &sources[(column + 1) % count].0;
            for (v, q) in operator.as_mut_slice().iter_mut().zip(other.as_slice()) {
                *v += mixing * q;
            }
        }
        let density = if let Some(fixed_setup) = &mut fixed_setup {
            let mut core = operator;
            for value in core.as_mut_slice() {
                *value *= strength;
            }
            fixed_setup.options.extra_hcore = Some(core);
            let state = run_collinear_property_reference(fixed_setup, kind).unwrap();
            assert!(state.scf.converged);
            state.scf.p_total()
        } else {
            let mut full = fixed_real.clone();
            for a in 0..n {
                for b in 0..n {
                    let value = strength * operator[(a, b)];
                    full[(a, b)] += value;
                    full[(n + a, n + b)] += value;
                }
            }
            let report = solve(
                "orbitals",
                &system,
                &options
                    .clone()
                    .with("density", true)
                    .with("gks_operator_real", &full),
            );
            let channel = &report.get("channels").unwrap().as_array().unwrap()[0];
            let spinor = Matrix::from_row_major(2 * n, 2 * n, values(channel, "density_real"));
            let mut charge = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    charge[(i, j)] = spinor[(i, j)] + spinor[(i + n, j + n)];
                }
            }
            charge
        };
        sources
            .iter()
            .map(|(operator, ionic)| convention.scale() * (ionic - density.frobenius_dot(operator)))
            .collect()
    };

    let names: Vec<_> = (0..count)
        .flat_map(|a| {
            let label = &label;
            (0..count).map(move |b| format!("{label} C[{a},{b}]"))
        })
        .collect();
    let mut stages = [
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
        ([0.016, 0.012, 0.008, 0.006], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Fourth),
    ]
    .to_vec();
    if let Some(rank) = racah_rank {
        // High-rank potentials can remain outside the asymptotic weak-field
        // interval after the original four stages. Retain those failed
        // intervals and resolve higher-order terms with weaker fields,
        // keeping the same component-fit gates and no error floor.
        stages.extend([
            ([0.002, 0.0015, 0.001, 0.00075], Stencil::Second),
            ([0.001, 0.00075, 0.0005, 0.000375], Stencil::Second),
        ]);
        // Higher-rank polynomials grow rapidly over the same basis.
        let field_scale = 0.65_f64.powi((rank - 1) as i32);
        for (steps, _) in &mut stages {
            for step in steps {
                *step *= field_scale;
            }
        }
    }
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; count * count];
            for (i, &h) in steps.iter().enumerate() {
                let mut directional = vec![vec![0.0; count]; count];
                for column in 0..count {
                    let (plus, minus) = (sample(column, h), sample(column, -h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (sample(column, 2.0 * h), sample(column, -2.0 * h))
                    } else {
                        (vec![0.0; count], vec![0.0; count])
                    };
                    for row in 0..count {
                        directional[row][column] =
                            stencil.first(plus[row], minus[row], plus2[row], minus2[row], h);
                    }
                }
                // Sources are e_b + mixing*e_(b+1). Invert this full-rank
                // cyclic transform before fitting each response component.
                for row in 0..count {
                    for column in 0..count {
                        let measured = undo_cyclic_mixing(&directional[row], column, mixing);
                        let index = count * row + column;
                        errors[index][i] = (measured - tensor[index]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(&label, &failures);

    let damped = |omega| {
        solve(
            id,
            &system,
            &controls
                .clone()
                .with("frequency", omega)
                .with("frequency_damping", 0.005),
        )
    };
    let plus = damped(0.02);
    let minus = damped(-0.02);
    println!(
        "{label} retarded electric quadrupole polarizability {}",
        plus.to_json()
    );
    for (key, sign) in [
        (format!("{property}_real"), 1.0),
        (format!("{property}_imag"), -1.0),
    ] {
        for (a, b) in values(&plus, &key).iter().zip(values(&minus, &key)) {
            assert!((a - sign * b).abs() < 1e-8);
        }
    }
    // A finite imaginary frequency includes a physical frequency shift,
    // which can exceed the static-limit tolerance for high-rank moments.
    // Eliminate the leading powers without assuming an even response.
    // Complex references can have an odd transverse response.
    let frequencies = [1e-6, 2e-6, 4e-6, 8e-6];
    let near_static = frequencies.map(|frequency| {
        let response = solve(
            id,
            &system,
            &controls
                .clone()
                .with("frequency", frequency)
                .with("imaginary_frequency", true),
        );
        assert_eq!(
            response.get("frequency_imag").and_then(Value::as_f64),
            Some(frequency)
        );
        let residuals = values(&response, "response_residuals");
        assert_eq!(residuals.len(), count);
        assert!(residuals.iter().all(|v| v.is_finite()));
        println!("{label} imaginary frequency {frequency:.1e}: response residuals {residuals:?}");
        values(&response, &format!("{property}_real"))
    });
    let finite_defects = near_static.each_ref().map(|sample| {
        sample
            .iter()
            .zip(&tensor)
            .map(|(dynamic, static_value)| dynamic - static_value)
            .collect::<Vec<_>>()
    });
    println!("{label} signed finite-frequency defects at {frequencies:?}: {finite_defects:?}");
    let remove_first_two_orders =
        |h: f64, twice: f64, four_times: f64| h + ((four_times - h) - 6.0 * (twice - h)) / 3.0;
    let mut extrapolated_defect = 0.0_f64;
    for (i, &static_value) in tensor.iter().enumerate() {
        let [at_h, at_2h, at_4h, at_8h] = near_static.each_ref().map(|sample| sample[i]);
        let fine = remove_first_two_orders(at_h, at_2h, at_4h);
        let coarse = remove_first_two_orders(at_2h, at_4h, at_8h);
        let extrapolated = fine + (fine - coarse) / 7.0;
        let error = (extrapolated - static_value).abs();
        extrapolated_defect = extrapolated_defect.max(error);
        assert!(error < 1e-7, "{label} static limit entry {i}: {error:.9e}");
    }
    println!("{label} static limit: largest extrapolated defect {extrapolated_defect:.9e}");
}

fn fit_cases(fixture: &str, cases: &[(&str, Option<&str>)], grid_level: usize) {
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-quadrupole-alpha-{fixture}-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for &(reference, xc) in cases {
        fit_case(reference, basis.to_str().unwrap(), xc, grid_level, None);
    }
    std::fs::remove_file(basis).unwrap();
}

#[test]
fn racah_higher_multipole_response_fits_all_components() {
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-racah-multipole-alpha-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for rank in 1..=5 {
        for reference in ["rks", "uks", "roks", "gks"] {
            fit_case(
                reference,
                basis.to_str().unwrap(),
                Some("hf"),
                1,
                Some(rank),
            );
        }
    }
    std::fs::remove_file(basis).unwrap();
}

#[test]
fn electric_quadrupole_polarizability_matches_independent_gradient_fields() {
    fit_cases(
        "hf",
        &[
            ("rks", Some("hf")),
            ("uks", Some("hf")),
            ("roks", Some("hf")),
            ("gks", Some("hf")),
            ("rks", None),
        ],
        1,
    );
}

#[test]
fn racah_multipole_functional_matrix() {
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-racah-functional-alpha-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for (reference, rank) in [("rks", 3), ("uks", 4), ("roks", 5), ("gks", 3)] {
        for xc in [Some("svwn5"), Some("pbe"), Some("tpss"), Some("pbe0"), None] {
            fit_case(reference, basis.to_str().unwrap(), xc, 0, Some(rank));
        }
    }
    std::fs::remove_file(basis).unwrap();
}

#[test]
fn racah_dynamic_response_matches_dense_one_electron_sum() {
    let system = system(true);
    let n = system.positions.len();
    let mut overlap = Matrix::zeros(n, n);
    let mut core = Matrix::zeros(n, n);
    for (i, a) in system.positions.iter().enumerate() {
        for (j, b) in system.positions.iter().enumerate() {
            let distance2 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
            let gaussian_overlap = (-0.5 * EXPONENT * distance2).exp();
            let kinetic = 0.5 * EXPONENT * (3.0 - EXPONENT * distance2);
            let attraction = system
                .positions
                .iter()
                .map(|c| {
                    let distance = (0..3)
                        .map(|k| (0.5 * (a[k] + b[k]) - c[k]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if distance == 0.0 {
                        2.0 * (2.0 * EXPONENT / std::f64::consts::PI).sqrt()
                    } else {
                        libm::erf((2.0 * EXPONENT).sqrt() * distance) / distance
                    }
                })
                .sum::<f64>();
            overlap[(i, j)] = gaussian_overlap;
            core[(i, j)] = gaussian_overlap * (kinetic - attraction);
        }
    }
    // One electron has no net interaction in the variational HF energy.
    // Solve the dense one-body spectrum from independent s-Gaussian formulas.
    let orth = crate::linalg::Orthogonalizer::canonical(&overlap, 1e-12).unwrap();
    let (energies, orbitals) = orth.solve_roothaan(&core).unwrap();
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-racah-dense-response-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    let options = Value::object()
        .with("basis", basis.to_str().unwrap())
        .with("reference", "uks")
        .with("xc", "hf")
        .with("charge", (n - 1) as i64)
        .with("multiplicity", 2usize)
        .with("grid_weight_cutoff", 0.0)
        .with("guess", "core")
        .with("conv_tol_grad", 1e-12)
        .with("conv_tol_dp", 1e-12)
        .with("origin", ORIGIN);
    for rank in 3..=5 {
        let count = 2 * rank + 1;
        let source_mo: Vec<_> = gaussian_racah_sources(&system, rank)
            .into_iter()
            .map(|(op, _)| orbitals.transpose_matmul_seq(&op.matmul_seq(&orbitals)))
            .collect();
        let controls = options.clone().with("multipole_rank", rank);
        let retarded = Complex::new(0.08, 0.012);
        let imaginary = Complex::new(0.0, 1e-6);
        for (z, input) in [
            (
                retarded,
                controls
                    .clone()
                    .with("frequency", retarded.re)
                    .with("frequency_damping", retarded.im),
            ),
            (
                imaginary,
                controls
                    .with("frequency", imaginary.im)
                    .with("imaginary_frequency", true),
            ),
        ] {
            let report = solve(MULTIPOLE_ID, &system, &input);
            let real = values(&report, "multipole_polarizability_real");
            let imag = values(&report, "multipole_polarizability_imag");
            let mut defect = 0.0_f64;
            for a in 0..count {
                for b in 0..count {
                    let expected = (1..energies.len()).fold(Complex::ZERO, |sum, v| {
                        let gap = Complex::from(energies[v] - energies[0]);
                        let pair = source_mo[a][(0, v)] * source_mo[b][(v, 0)];
                        sum + (Complex::from(1.0) / (gap - z) + Complex::from(1.0) / (gap + z))
                            * pair
                    });
                    let actual = Complex::new(real[count * a + b], imag[count * a + b]);
                    defect = defect.max((actual - expected).abs());
                }
            }
            println!("Racah rank {rank} dense z={z:?} max defect={defect:.17e}");
            assert!(defect < 5e-8);
        }
    }
    std::fs::remove_file(basis).unwrap();
}

fn fit_functional_classes(reference: &str) {
    let cases =
        [Some("svwn5"), Some("pbe"), Some("tpss"), Some("pbe0"), None].map(|xc| (reference, xc));
    fit_cases(reference, &cases, 0);
}

#[test]
fn quadrupole_functional_matrix_rks() {
    fit_functional_classes("rks");
}

#[test]
fn quadrupole_functional_matrix_uks() {
    fit_functional_classes("uks");
}

#[test]
fn quadrupole_functional_matrix_roks() {
    fit_functional_classes("roks");
}

#[test]
fn quadrupole_functional_matrix_gks() {
    fit_functional_classes("gks");
}

fn fit_correlated_case(model: &str, unrestricted: bool, basis: &str, racah_rank: Option<usize>) {
    use crate::driver::double_hybrid::{calculate_prepared, prepare_with_core};

    let system = system(unrestricted);
    let mut controls = correlated_controls(model, unrestricted, basis).with("origin", ORIGIN);
    let (id, convention, sources) = if let Some(rank) = racah_rank {
        controls.set("multipole_rank", rank);
        (
            MULTIPOLE_ID,
            Convention::Racah(rank),
            gaussian_racah_sources(&system, rank),
        )
    } else {
        (
            ID,
            Convention::Buckingham,
            gaussian_sources(&system).to_vec(),
        )
    };
    let rank = convention.rank();
    let count = convention.count();
    let property = convention.property();
    let analytic = solve(id, &system, &controls);
    let label = format!(
        "{model} {} {id} rank {rank}",
        if unrestricted { "uks" } else { "rks" }
    );
    println!(
        "{label} electric quadrupole polarizability {}",
        analytic.to_json()
    );

    let keys = [
        (property.to_owned(), "cartesian"),
        (format!("parent_{property}"), "parent_cartesian"),
        (
            format!("pt2_{property}_correction"),
            "pt2_cartesian_correction",
        ),
        (
            format!("pt2_same_spin_alpha_{property}"),
            "pt2_same_spin_alpha",
        ),
        (
            format!("pt2_same_spin_beta_{property}"),
            "pt2_same_spin_beta",
        ),
        (format!("pt2_opposite_spin_{property}"), "pt2_opposite_spin"),
        (
            format!("target_{property}_correction"),
            "target_cartesian_correction",
        ),
    ];
    let tensors = keys
        .each_ref()
        .map(|(response, _)| values(&analytic, response));
    let ss = analytic
        .get("pt2_same_spin_coefficient")
        .unwrap()
        .as_f64()
        .unwrap();
    let os = analytic
        .get("pt2_opposite_spin_coefficient")
        .unwrap()
        .as_f64()
        .unwrap();
    for i in 0..count * count {
        assert!((tensors[0][i] - tensors[1][i] - tensors[2][i] - tensors[6][i]).abs() < 1e-11);
        assert!(
            (tensors[2][i] - ss * (tensors[3][i] + tensors[4][i]) - os * tensors[5][i]).abs()
                < 1e-11
        );
    }
    let channels = if model == "custom" {
        keys.len()
    } else {
        assert_eq!(tensors[6], vec![0.0; count * count]);
        keys.len() - 1
    };
    let pairs: Vec<_> = (0..count)
        .flat_map(|b| (0..=b).map(move |a| (a, b)))
        .collect();
    let names: Vec<_> = pairs
        .iter()
        .flat_map(|&(a, b)| {
            let label = &label;
            keys[..channels]
                .iter()
                .map(move |(key, _)| format!("{label} {key}[{a},{b}]"))
        })
        .collect();
    let moment_input = correlated_controls(model, unrestricted, basis)
        .with("max_rank", rank)
        .with("origin", ORIGIN);
    let projections = convention.weights();
    let mixing: f64 = if racah_rank.is_some() { 0.23 } else { 0.0 };
    let mut cache = std::collections::BTreeMap::new();
    let mut sample = |column: usize, step: f64| -> [Vec<f64>; 7] {
        cache
            .entry((column, step.to_bits()))
            .or_insert_with(|| {
                let request = Controls::from_value(&moment_input).unwrap();
                let state = prepare_with_core(&system, &request, "molecular-multipole", |_| {
                    let mut core = sources[column].0.clone();
                    let mixed = &sources[(column + 1) % count].0;
                    for (value, other) in core.as_mut_slice().iter_mut().zip(mixed.as_slice()) {
                        *value = step * (*value + mixing * other);
                    }
                    Some(core)
                })
                .unwrap();
                let report = calculate_prepared(state, "molecular-multipole").unwrap();
                assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
                let rank = &report.get("ranks").unwrap().as_array().unwrap()[rank];
                keys.each_ref().map(|(_, key)| {
                    let raw = values(rank, key);
                    projections
                        .iter()
                        .map(|row| {
                            convention.scale()
                                * row
                                    .iter()
                                    .zip(&raw)
                                    .map(|(weight, value)| weight * value)
                                    .sum::<f64>()
                        })
                        .collect()
                })
            })
            .clone()
    };
    let mut stages = [
        ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
        ([0.002, 0.0015, 0.001, 0.00075], Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Second),
        ([0.004, 0.003, 0.002, 0.0015], Stencil::Fourth),
        ([0.008, 0.006, 0.004, 0.003], Stencil::Fourth),
    ];
    if let Some(rank) = racah_rank {
        for (steps, _) in &mut stages {
            for step in steps {
                *step *= 0.65_f64.powi((rank - 1) as i32);
            }
        }
    }
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &stages,
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (i, &h) in steps.iter().enumerate() {
                let mut derivatives = vec![vec![vec![0.0; count]; count]; channels];
                for column in 0..count {
                    let (plus, minus) = (sample(column, h), sample(column, -h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (sample(column, 2.0 * h), sample(column, -2.0 * h))
                    } else {
                        (
                            std::array::from_fn(|_| vec![0.0; count]),
                            std::array::from_fn(|_| vec![0.0; count]),
                        )
                    };
                    for c in 0..channels {
                        for row in 0..count {
                            derivatives[c][row][column] = stencil.first(
                                plus[c][row],
                                minus[c][row],
                                plus2[c][row],
                                minus2[c][row],
                                h,
                            );
                        }
                    }
                }
                for (index, &(row, column)) in pairs.iter().enumerate() {
                    for c in 0..channels {
                        let measured = undo_cyclic_mixing(&derivatives[c][row], column, mixing);
                        errors[index * channels + c][i] =
                            (measured - tensors[c][count * row + column]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(&label, &failures);

    let near_static = solve(
        id,
        &system,
        &controls
            .with("frequency", 1e-6)
            .with("imaginary_frequency", true),
    );
    for (index, (response, _)) in keys.iter().enumerate() {
        let real = values(&near_static, &format!("{response}_real"));
        let imag = values(&near_static, &format!("{response}_imag"));
        assert!(real
            .iter()
            .zip(&tensors[index])
            .all(|(a, b)| (a - b).abs() < 1e-7));
        assert!(imag.iter().all(|x| x.abs() < 1e-9));
    }
}

#[test]
fn correlated_electric_quadrupole_polarizability_fits_all_components_and_channels() {
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-pt2-quadrupole-alpha-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for model in ["b2plyp", "custom"] {
        for unrestricted in [false, true] {
            fit_correlated_case(model, unrestricted, basis.to_str().unwrap(), None);
        }
    }
    std::fs::remove_file(basis).unwrap();
}

#[test]
fn correlated_racah_multipole_response_fits_all_components_and_channels() {
    let basis =
        std::env::temp_dir().join(format!("dft-rs-pt2-racah-alpha-{}.g94", std::process::id()));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    for rank in 3..=5 {
        for unrestricted in [false, true] {
            fit_correlated_case("custom", unrestricted, basis.to_str().unwrap(), Some(rank));
        }
    }
    std::fs::remove_file(basis).unwrap();
}

#[path = "../../../tests/support/spinor_pt2_oracle.rs"]
mod spinor_pt2_oracle;
use crate as dft_rs;

#[test]
fn generalized_correlated_quadrupole_polarizability_fits_stationary_spinor_energies() {
    fit_generalized_quadrupole(None, false);
}

#[test]
fn generalized_distinct_target_quadrupole_polarizability_fits_stationary_spinor_energies() {
    fit_generalized_quadrupole(Some("svwn5"), false);
}

#[test]
fn generalized_weighted_parent_quadrupole_polarizability_fits_stationary_spinor_energies() {
    fit_generalized_quadrupole(None, true);
}

#[test]
fn generalized_racah_multipole_response_fits_stationary_spinor_energies() {
    fn fit<const N: usize>() {
        fit_generalized_electric::<N>(None, false, MULTIPOLE_ID, 1.0, |system| {
            let rank = (N - 1) / 2;
            gaussian_racah_sources(system, rank)
                .into_iter()
                .map(|(operator, _)| operator)
                .collect::<Vec<_>>()
                .try_into()
                .expect("Racah mode count")
        });
    }
    fit::<7>();
    fit::<9>();
    fit::<11>();
}

fn fit_generalized_quadrupole(target_name: Option<&str>, weighted_parent: bool) {
    fit_generalized_electric(target_name, weighted_parent, ID, 3.0, |system| {
        gaussian_sources(system).map(|(operator, _)| operator)
    });
}

#[test]
fn common_orbital_correlated_dipole_response_requires_a_constrained_pair_model() {
    let controls = Controls::from_value(
        &Value::object()
            .with("reference", "roks")
            .with("xc", "hf")
            .with("pt2_same_spin", 0.27)
            .with("pt2_opposite_spin", 0.27),
    )
    .unwrap();
    for feature in ["polarizability", "dynamic-polarizability"] {
        let error = run(feature, &system(true), &controls).unwrap_err();
        assert!(matches!(
            error,
            DftError::Unsupported { feature, .. }
                if feature == "common-orbital double-hybrid dipole polarizability"
        ));
    }
}

#[test]
fn generalized_distinct_target_dipole_polarizability_fits_stationary_spinor_energies() {
    fit_generalized_electric(Some("svwn5"), false, "polarizability", 1.0, |system| {
        std::array::from_fn::<_, 3, _>(|axis| {
            let n = system.positions.len();
            let mut operator = Matrix::zeros(n, n);
            for (i, ri) in system.positions.iter().enumerate() {
                for (j, rj) in system.positions.iter().enumerate() {
                    let separation_squared = (0..3).map(|a| (ri[a] - rj[a]).powi(2)).sum::<f64>();
                    let overlap = (-0.5 * EXPONENT * separation_squared).exp();
                    operator[(i, j)] = 0.5 * overlap * (ri[axis] + rj[axis]);
                }
            }
            operator
        })
    });
}

fn fit_generalized_electric<const N: usize>(
    target_name: Option<&str>,
    weighted_parent: bool,
    id: &str,
    scale: f64,
    operators: impl FnOnce(&System) -> [Matrix; N],
) {
    use crate::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    use crate::grid::MolecularGrid;
    use crate::integrals::{eri_incore, CoulombKernel};
    use crate::periodic::spinor_scf::SpinorEnergyBackend;
    use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
    use crate::xc::{resolve, FunctionalSpec};
    use std::collections::BTreeMap;

    let system = system(true);
    let parent = if weighted_parent {
        "weighted-parent"
    } else {
        "HF-parent"
    };
    let (moment, property) = match id {
        ID => ("quadrupole", "quadrupole_polarizability"),
        MULTIPOLE_ID => ("Racah multipole", "multipole_polarizability"),
        _ => ("dipole", "polarizability"),
    };
    let fixture = target_name.unwrap_or(parent);
    let label = target_name.map_or_else(
        || parent.to_string(),
        |name| format!("{parent}/{name}-target"),
    );
    let basis = std::env::temp_dir().join(format!(
        "dft-rs-gks-{moment}-{fixture}-{}.g94",
        std::process::id()
    ));
    std::fs::write(&basis, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    let n = system.numbers.len();
    let coefficient = 0.27;
    let (real, imaginary) = fixed_spinor(n);
    let mut common = Value::object()
        .with("reference", "gks")
        .with("multiplicity", 2)
        .with("basis", basis.to_str().unwrap())
        .with("grid_level", 0)
        .with("grid_weight_cutoff", 0.0)
        .with("guess", "core")
        .with("conv_tol", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("max_iter", 500)
        .with("field", [0.0012, -0.0017, 0.0008])
        .with("gks_operator_imaginary", &imaginary);
    if weighted_parent {
        common.set(
            "xc_terms",
            vec![
                Value::object().with("xc", "hf").with("weight", 0.80),
                Value::object().with("xc", "svwn5").with("weight", 0.20),
            ],
        );
    } else {
        common.set("xc", "hf");
    }
    let mut controls = common
        .clone()
        .with("response_tol", 1e-12)
        .with("gks_operator_real", &real)
        .with("pt2_same_spin", coefficient)
        .with("pt2_opposite_spin", coefficient);
    if let Some(target) = target_name {
        controls.set("double_hybrid_target_xc", target);
    }
    if is_electric_multipole(id) {
        controls.set("origin", ORIGIN);
    }
    if id == MULTIPOLE_ID {
        controls.set("multipole_rank", (N - 1) / 2);
    }
    let report = solve(id, &system, &controls);
    println!(
        "generalized {label} electric {moment} polarizability {}",
        report.to_json()
    );
    let mut keys = vec![
        format!("parent_{property}"),
        format!("generalized_pt2_{property}"),
        property.to_string(),
    ];
    if target_name.is_some() {
        keys.push(format!("target_{property}_correction"));
    }
    let exact: Vec<_> = keys.iter().map(|key| values(&report, key)).collect();
    let correction = values(&report, &format!("pt2_{property}_correction"));
    assert!(exact.iter().all(|tensor| tensor.len() == N * N));
    for p in 0..N * N {
        assert!((correction[p] - coefficient * exact[1][p]).abs() < 1e-13);
        let target = exact.get(3).map_or(0.0, |v| v[p]);
        assert!((exact[2][p] - exact[0][p] - correction[p] - target).abs() < 1e-13);
    }
    let method = MolecularMethod::from_controls(&Controls::from_value(&common).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 128).unwrap();
    let target = target_name.map(|name| resolve(&FunctionalSpec::Combined(name.into())).unwrap());
    let grid = target
        .as_ref()
        .map(|_| MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap());
    let seed = target
        .as_ref()
        .map(|_| run_molecular_reference(&setup).unwrap());
    let backends = target.as_ref().map(|target| {
        [&setup.functional, target].map(|functional| {
            MolecularSpinorBackend::new(
                &setup.molecule,
                &setup.basis,
                functional,
                grid.as_ref(),
                &seed.as_ref().unwrap().scf,
                &setup.options,
                BlochMatrix::zeros(2 * n, 2 * n),
                [0.0, 0.0, 1.0],
                MolecularSpinorFunctionalModel::SpinCovariant,
                1e-12,
            )
            .unwrap()
        })
    });
    let sources = operators(&system);
    let mut cache = BTreeMap::new();
    let mut sample = |a: usize, x: f64, b: usize, y: f64| -> [f64; 4] {
        let strengths = std::array::from_fn::<_, N, _>(|k| {
            (if k == a { x } else { 0.0 }) + (if k == b { y } else { 0.0 })
        });
        *cache.entry(strengths.map(f64::to_bits)).or_insert_with(|| {
            let mut biased = real.clone();
            for (mode, source) in sources.iter().enumerate() {
                for i in 0..n {
                    for j in 0..n {
                        let element = strengths[mode] * source[(i, j)];
                        biased[(i, j)] += element;
                        biased[(n + i, n + j)] += element;
                    }
                }
            }
            let result = solve(
                "orbitals",
                &system,
                &common
                    .clone()
                    .with("gks_operator_real", &biased)
                    .with("coefficients", true)
                    .with("density", backends.is_some()),
            );
            let channel = &result.get("channels").unwrap().as_array().unwrap()[0];
            let matrix = |key| Matrix::from_row_major(2 * n, 2 * n, values(channel, key));
            let spinor = ComplexMatrix {
                re: matrix("coefficients_real"),
                im: matrix("coefficients_imaginary"),
            };
            let spinor_energy = values(channel, "energies");
            let occupied = values(channel, "occupations")
                .iter()
                .filter(|f| **f == 1.0)
                .count();
            let pair = spinor_pt2_oracle::stationary_pt2(&spinor, &spinor_energy, occupied, &eri);
            let energy = result.get("energy").unwrap().as_f64().unwrap();
            let target = backends.as_ref().map_or(0.0, |[parent, target]| {
                let density = [BlochMatrix {
                    re: matrix("density_real"),
                    im: matrix("density_imaginary"),
                }];
                let reference = parent.evaluate(&density).unwrap();
                let evaluation = target.evaluate(&density).unwrap();
                assert!(reference.operator_converged && evaluation.operator_converged);
                evaluation.energy - reference.energy
            });
            [energy, pair, energy + coefficient * pair + target, target]
        })
    };
    let centre = sample(0, 0.0, 0, 0.0);
    assert!(
        (report
            .get("generalized_pt2_correlation")
            .unwrap()
            .as_f64()
            .unwrap()
            - centre[1])
            .abs()
            < 2e-12
    );
    assert!((report.get("energy").unwrap().as_f64().unwrap() - centre[2]).abs() < 2e-12);
    if target_name.is_some() {
        let reported = report
            .get("energy_components")
            .unwrap()
            .get("target_correction")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((reported - centre[3]).abs() < 2e-12);
    }
    let mut energy_controls = common
        .clone()
        .with("gks_operator_real", &real)
        .with("pt2_same_spin", coefficient)
        .with("pt2_opposite_spin", coefficient);
    if let Some(target) = target_name {
        energy_controls.set("double_hybrid_target_xc", target);
    }
    let stationary = solve("energy", &system, &energy_controls);
    let stationary_energy = stationary.get("energy").unwrap().as_f64().unwrap();
    let stationary_pt2 = stationary
        .get("generalized_pt2_correlation")
        .unwrap()
        .as_f64()
        .unwrap();
    let pair_residual = stationary
        .get("generalized_pt2_amplitude_residual")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((stationary_energy - centre[2]).abs() < 2e-12);
    assert!((stationary_pt2 - centre[1]).abs() < 2e-12);
    assert!(pair_residual < 1e-12);
    println!("GKS {label} stationary energy={stationary_energy:.16e} pair={stationary_pt2:.16e} amplitude_residual={pair_residual:.12e}");
    let pairs = (0..N)
        .flat_map(|a| (a..N).map(move |b| (a, b)))
        .collect::<Vec<_>>();
    let names = pairs
        .iter()
        .flat_map(|(a, b)| {
            let label = &label;
            keys.iter()
                .map(move |key| format!("GKS {label} {key}[{a},{b}]"))
        })
        .collect::<Vec<_>>();
    let mut intervals = [
        [0.008, 0.006, 0.004, 0.003],
        [0.004, 0.003, 0.002, 0.0015],
        [0.016, 0.012, 0.008, 0.006],
    ];
    if id == MULTIPOLE_ID {
        for steps in &mut intervals {
            for step in steps {
                *step *= 0.65_f64.powi(((N - 1) / 2 - 1) as i32);
            }
        }
    }
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        &intervals,
        |steps| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for (index, &(a, b)) in pairs.iter().enumerate() {
                for (step, &h) in steps.iter().enumerate() {
                    let pp = sample(a, h, b, if a == b { 0.0 } else { h });
                    let mm = sample(a, -h, b, if a == b { 0.0 } else { -h });
                    let measured: [f64; 4] = if a == b {
                        std::array::from_fn(|k| (pp[k] - 2.0 * centre[k] + mm[k]) / (h * h))
                    } else {
                        let pm = sample(a, h, b, -h);
                        let mp = sample(a, -h, b, h);
                        std::array::from_fn(|k| {
                            let plus = Stencil::Second.first(pp[k], pm[k], 0.0, 0.0, h);
                            let minus = Stencil::Second.first(mp[k], mm[k], 0.0, 0.0, h);
                            Stencil::Second.first(plus, minus, 0.0, 0.0, h)
                        })
                    };
                    for k in 0..keys.len() {
                        errors[keys.len() * index + k][step] =
                            (-scale * measured[k] - exact[k][N * a + b]).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        &format!("generalized {label} {moment} component fits"),
        &failures,
    );

    for invalid in [
        controls.clone().with("pt2_opposite_spin", 0.52),
        controls.clone().with("double_hybrid_target_xc", "wb97x-v"),
        controls.clone().with("frequency", 0.02),
    ] {
        let error = run(id, &system, &Controls::from_value(&invalid).unwrap()).unwrap_err();
        assert!(matches!(error, DftError::Unsupported { .. }), "{error}");
    }
    std::fs::remove_file(basis).unwrap();
}
