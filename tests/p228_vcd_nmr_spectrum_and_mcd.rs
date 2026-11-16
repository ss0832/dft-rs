// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of vibrational circular dichroism, the first-order NMR
//! spectrum and magnetic circular dichroism on the common driver.
//!
//! `vcd`: the electronic atomic axial tensor `<d psi/dR_x | d psi/dB_b>` is
//! checked against an independent four-step central difference of the
//! projector of the displaced determinant: with `C` the orbitals of the
//! reference, `S_x(h)` the overlap between the basis of the reference and
//! the basis of the displaced geometry and `P(h)` the occupied density of
//! the displaced self-consistent field, `T(h) = C^T S_x P S_x^T C` has
//! `dT_ai/dh = <phi_a | d phi_i/dR_x>`, independent of how the displaced
//! occupied orbitals are rotated among themselves. The magnetic amplitudes
//! come from the magnetic response of the reference (validated by the
//! shielding and magnetizability acceptance). Physical checks: a
//! triatomic molecule is planar, so all its rotational strengths vanish;
//! the mirror image changes the sign of every rotational strength and keeps
//! frequencies and dipole strengths; distributed-origin strengths do not
//! depend on a rigid translation.
//!
//! `nmr-spectrum`: the chemical shifts are the reference shieldings minus
//! the isotropic shieldings of `nmr-shielding`; every element integrates to
//! the closed-form area of its Lorentzians on the reported axis; with
//! couplings the hydrogen fluoride spectrum is the closed-form AX doublet of
//! the reported coupling constant. No derivative beyond the validated
//! shieldings and couplings is involved.
//!
//! `mcd`: Faraday B terms are equal for a molecule and its mirror image and
//! do not change under a rigid rotation or translation; a molecule without
//! degenerate states has no A terms, while ammonia, with degenerate E
//! states, has levels of two states and A terms that do not change under a
//! rotation. Full projected ground-plus-singles Hamiltonians additionally
//! check the B coefficients by independent finite magnetic fields, retaining
//! all states and Cartesian axes. Atomic P levels fix the canonical A sign.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::basis::MolBasis;
use dft_rs::driver::method::{
    molecule_from, prepare_molecular, run_molecular_reference, MolecularMethod,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::one_electron::overlap_matrix;
use dft_rs::properties::magnetic::{magnetic_density_response, Gauge};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::{Matrix, Vec3};

fn controls(pairs: &[(&str, Value)]) -> Controls {
    Controls::from_value(&Value::Object(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    ))
    .unwrap()
}

fn tight() -> Vec<(&'static str, Value)> {
    vec![
        ("basis", Value::Str("sto-3g".into())),
        ("xc", Value::Str("hf".into())),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-12)),
    ]
}

fn list(items: &[f64]) -> Value {
    Value::Array(items.iter().map(|v| Value::Float(*v)).collect())
}

fn number(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap()
}

fn vector(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

/// Twisted hydrogen peroxide (Bohr), chiral; `mirror` reflects it through
/// the xy plane.
fn peroxide(mirror: bool) -> System {
    let s = if mirror { -1.0 } else { 1.0 };
    System::molecule(
        vec![8, 8, 1, 1],
        vec![
            [-1.37, 0.0, 0.0],
            [1.37, 0.0, 0.0],
            [-1.6883, 1.03539, 1.4787 * s],
            [1.6883, 1.03539, -1.4787 * s],
        ],
    )
}

/// Water distorted away from C2v (Bohr).
fn water() -> System {
    System::molecule(
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.23],
            [0.0, 1.4309, -0.8867],
            [0.12, -1.4809, -0.9067],
        ],
    )
}

fn moved(system: &System, rotation: Option<[[f64; 3]; 3]>, shift: [f64; 3]) -> System {
    let mut out = system.clone();
    for r in out.positions.iter_mut() {
        let p = match rotation {
            Some(m) => std::array::from_fn(|i| (0..3).map(|j| m[i][j] * r[j]).sum::<f64>()),
            None => *r,
        };
        *r = std::array::from_fn(|i| p[i] + shift[i]);
    }
    out
}

/// Rotation by 0.7 rad about (1, 2, 3)/sqrt(14).
fn rotation() -> [[f64; 3]; 3] {
    let n = 14.0_f64.sqrt();
    let k = [1.0 / n, 2.0 / n, 3.0 / n];
    let (s, c) = 0.7_f64.sin_cos();
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let delta = if i == j { 1.0 } else { 0.0 };
            let cross = match (i, j) {
                (0, 1) => -k[2],
                (0, 2) => k[1],
                (1, 0) => k[2],
                (1, 2) => -k[0],
                (2, 0) => -k[1],
                (2, 1) => k[0],
                _ => 0.0,
            };
            c * delta + s * cross + (1.0 - c) * k[i] * k[j]
        })
    })
}

/// Entries of a list that may hold nulls (modes without a positive frequency).
fn optional(report: &Value, key: &str) -> Vec<Option<f64>> {
    report
        .get(key)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .collect()
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn nearly_axial_magnetic_sources_retain_the_original_equation() {
    // A nearly axial field gives a small but nonzero occupied-virtual
    // source. Its magnitude must not change the physical integral mask or
    // cause it to be replaced by a symmetry zero.
    for tilt in [0.0, 1e-13] {
        let system = System::molecule(vec![1, 9], vec![[0.0; 3], [tilt, 0.0, 1.73]]);
        let method = MolecularMethod::from_controls(&controls(&tight())).unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        let reference = run_molecular_reference(&setup).unwrap();
        let solve = |mode| {
            let mut options = setup.options.clone();
            options.eri_mode = mode;
            magnetic_density_response(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &options,
                Gauge::Giao,
                Vec3::new(0.0, 0.0, 0.0),
                &ResponseSolverConfig::default(),
            )
            .unwrap()
        };
        let direct = solve(dft_rs::scf::EriMode::Direct {
            screen_threshold: 1e-14,
        });
        let dense = solve(dft_rs::scf::EriMode::InCore { limit_mb: 64 });
        for spin in 0..2 {
            for axis in 0..3 {
                let source = &direct.value.gradient_hat[spin][axis];
                let largest = source.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let index = 3 * spin + axis;
                println!(
                    "axial tilt {tilt:.3e} spin {spin} axis {axis}: source {largest:.12e} residual {:.12e} iterations {} method {}",
                    direct.residuals[index], direct.iterations[index], direct.value.solver_methods[index],
                );
                if largest > 0.0 {
                    assert!(direct.iterations[index] > 0);
                }
                assert!(direct.residuals[index] <= 1e-10);
                let defect = direct.value.density_hat[spin][axis]
                    .max_abs_diff(&dense.value.density_hat[spin][axis]);
                assert!(
                    defect < 1e-11,
                    "direct/in-core density defect {defect:.12e}"
                );
            }
        }
    }
}

#[test]
fn electronic_axial_tensor_passes_a_four_step_fit_of_displaced_projectors() {
    let system = peroxide(false);
    let origin = [0.3, -0.2, 0.1];
    let mut request = tight();
    request.push(("gauge", Value::Str("common-origin".into())));
    request.push(("gauge_origin", list(&origin)));
    let report = run("vcd", &system, &controls(&request)).unwrap();
    let electronic = vector(&report, "atomic_axial_tensors_electronic");

    let method = MolecularMethod::from_controls(&controls(&tight())).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    let scf = &reference.scf;
    let n = scf.overlap.rows;
    let response = magnetic_density_response(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        Gauge::CommonOrigin,
        Vec3::new(origin[0], origin[1], origin[2]),
        &ResponseSolverConfig::default(),
    )
    .unwrap();
    let occupied = [scf.n_alpha, scf.n_beta];
    // Magnetic amplitudes in the orbital basis of the reference.
    let amplitudes: Vec<Vec<Matrix>> = (0..2)
        .map(|spin| {
            let sc = scf.overlap.matmul(&scf.mo_coeff[spin]);
            (0..3)
                .map(|b| {
                    sc.transpose_matmul(&response.value.density_hat[spin][b])
                        .matmul(&sc)
                })
                .collect()
        })
        .collect();
    // T(h) of one spin for the displaced geometry.
    let projector = |displaced: &System| -> [Matrix; 2] {
        let setup_h = prepare_molecular(&displaced.numbers, &displaced.positions, &method).unwrap();
        let scf_h = run_molecular_reference(&setup_h).unwrap().scf;
        let mut numbers = system.numbers.clone();
        numbers.extend(displaced.numbers.iter().copied());
        let mut positions = system.positions.clone();
        positions.extend(displaced.positions.iter().copied());
        let both = molecule_from(&numbers, &positions, 0, 1).unwrap();
        let basis = MolBasis::build(&both, &setup.basis_set).unwrap();
        let s = overlap_matrix(&basis);
        assert_eq!(s.rows, 2 * n);
        let mut cross = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                cross[(i, j)] = s[(i, n + j)];
                assert!((s[(i, j)] - scf.overlap[(i, j)]).abs() < 1e-12);
            }
        }
        std::array::from_fn(|spin| {
            let c = &scf.mo_coeff[spin];
            let left = c.transpose_matmul(&cross);
            left.matmul(&scf_h.density[spin])
                .matmul_transpose_seq(&left)
        })
    };
    // Convention check at h = 0: T is the occupied projector.
    let unmoved = projector(&system);
    for spin in 0..2 {
        for p in 0..n {
            let expected = if p < occupied[spin] { 1.0 } else { 0.0 };
            assert!((unmoved[spin][(p, p)] - expected).abs() < 1e-10);
        }
    }
    let steps = [0.02, 0.01, 0.005, 0.0025];
    // Hydrogen z and oxygen y.
    for component in [8usize, 1] {
        let (atom, axis) = (component / 3, component % 3);
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| {
                let shifted = |sign: f64| {
                    let mut displaced = system.clone();
                    displaced.positions[atom][axis] += sign * h;
                    projector(&displaced)
                };
                let (plus, minus) = (shifted(1.0), shifted(-1.0));
                (0..3).fold(0.0_f64, |m, b| {
                    let mut value = 0.0;
                    for spin in 0..2 {
                        let no = occupied[spin];
                        for a in no..n {
                            for i in 0..no {
                                let derivative =
                                    (plus[spin][(a, i)] - minus[spin][(a, i)]) / (2.0 * h);
                                value += amplitudes[spin][b][(a, i)] * derivative;
                            }
                        }
                    }
                    m.max((value - electronic[3 * component + b]).abs())
                })
            })
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "electronic axial tensor row {component}: {:?} steps {steps:?} errors {errors:?} \
             p={:.6} R2={:.8}",
            &electronic[3 * component..3 * component + 3],
            fit.exponent,
            fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.05, "p = {}", fit.exponent);
        assert!(fit.r_squared > 0.9999, "R2 = {}", fit.r_squared);
    }
}

#[test]
fn rotational_strengths_follow_the_symmetry_of_the_molecule() {
    let planar = run("vcd", &water(), &controls(&tight())).unwrap();
    let flat = vector(&planar, "rotational_strengths");
    let flat_largest = flat.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
    println!("largest rotational strength of a triatomic molecule {flat_largest:.3e}");
    assert!(flat_largest < 1e-10);

    let left = run("vcd", &peroxide(false), &controls(&tight())).unwrap();
    let right = run("vcd", &peroxide(true), &controls(&tight())).unwrap();
    let translated = run(
        "vcd",
        &moved(&peroxide(false), None, [1.5, -2.0, 0.7]),
        &controls(&tight()),
    )
    .unwrap();
    let (r_left, r_right) = (
        vector(&left, "rotational_strengths"),
        vector(&right, "rotational_strengths"),
    );
    let scale = r_left.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
    let mirror: Vec<f64> = r_right.iter().map(|r| -r).collect();
    let mirror_defect = largest_difference(&r_left, &mirror);
    let frequency_defect = largest_difference(
        &vector(&left, "frequencies_cm1"),
        &vector(&right, "frequencies_cm1"),
    );
    let (d_left, d_right) = (
        optional(&left, "dipole_strengths"),
        optional(&right, "dipole_strengths"),
    );
    assert_eq!(d_left.len(), d_right.len());
    let dipole_defect = d_left
        .iter()
        .zip(&d_right)
        .fold(0.0_f64, |m, pair| match pair {
            (Some(x), Some(y)) => m.max((x - y).abs()),
            (None, None) => m,
            _ => f64::INFINITY,
        });
    let translation_defect =
        largest_difference(&r_left, &vector(&translated, "rotational_strengths"));
    println!(
        "peroxide rotational strengths {r_left:?}: mirror {mirror_defect:.3e}, frequencies \
         {frequency_defect:.3e} cm-1, dipole strengths {dipole_defect:.3e}, translation \
         {translation_defect:.3e}"
    );
    assert!(scale > 1e-8, "the chiral molecule has rotational strengths");
    assert!(mirror_defect < 1e-8 * scale.max(1.0));
    assert!(frequency_defect < 1e-6);
    assert!(dipole_defect < 1e-8);
    assert!(translation_defect < 1e-7 * scale.max(1.0));
}

const PROTON_G: f64 = 5.5856946893;
const FLUORINE_G: f64 = 5.257736;

fn hydrogen_fluoride() -> System {
    System::molecule(vec![1, 9], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.7325]])
}

fn spectrum_request(couplings: bool) -> Vec<(&'static str, Value)> {
    let mut request = tight();
    request.push((
        "nmr_reference_elements",
        Value::Array(vec![Value::Int(1), Value::Int(9)]),
    ));
    request.push(("nmr_reference_shieldings", list(&[31.0, 410.0])));
    request.push(("nmr_broadening", Value::Float(0.05)));
    request.push(("nmr_points", Value::Int(20001)));
    request.push(("nmr_margin", Value::Float(2.0)));
    if couplings {
        request.push(("nmr_couplings", Value::Bool(true)));
        request.push(("nuclear_g", list(&[PROTON_G, FLUORINE_G])));
        request.push(("nmr_spectrometer_mhz", Value::Float(400.0)));
    }
    request
}

fn rows(report: &Value, key: &str) -> Vec<Vec<f64>> {
    report
        .get(key)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.numeric_array().unwrap().1)
        .collect()
}

/// Trapezoidal area of a sampled curve.
fn area(axis: &[f64], values: &[f64]) -> f64 {
    axis.windows(2)
        .zip(values.windows(2))
        .map(|(x, y)| 0.5 * (x[1] - x[0]) * (y[0] + y[1]))
        .sum()
}

#[test]
fn nmr_spectrum_is_the_closed_form_of_its_shifts_and_couplings() {
    let shielding = run("nmr-shielding", &hydrogen_fluoride(), &controls(&tight())).unwrap();
    let isotropic = vector(&shielding, "isotropic_shieldings");
    let half = 0.025;
    for couplings in [false, true] {
        let report = run(
            "nmr-spectrum",
            &hydrogen_fluoride(),
            &controls(&spectrum_request(couplings)),
        )
        .unwrap();
        let shifts = vector(&report, "chemical_shifts");
        assert!((shifts[0] - (31.0 - isotropic[0])).abs() < 1e-9);
        assert!((shifts[1] - (410.0 - isotropic[1])).abs() < 1e-9);
        let axes = rows(&report, "spectrum_axis");
        let intensities = rows(&report, "spectrum_intensity");
        // Lines of each element, from the reported numbers only.
        let lines: Vec<Vec<(f64, f64)>> = if couplings {
            let j = vector(&report, "coupling_constants_hz")[0];
            let spectrometer = number(&report, "spectrometer_mhz");
            let larmor = [spectrometer, spectrometer * FLUORINE_G / PROTON_G];
            (0..2)
                .map(|n| {
                    let offset = 0.5 * j / larmor[n];
                    vec![(shifts[n] - offset, 0.5), (shifts[n] + offset, 0.5)]
                })
                .collect()
        } else {
            (0..2).map(|n| vec![(shifts[n], 1.0)]).collect()
        };
        for element in 0..2 {
            let (axis, values) = (&axes[element], &intensities[element]);
            let (low, high) = (axis[0], axis[axis.len() - 1]);
            let mut model_defect = 0.0_f64;
            let mut peak = 0.0_f64;
            for (x, y) in axis.iter().zip(values) {
                let model: f64 = lines[element]
                    .iter()
                    .map(|(line, weight)| {
                        weight * half / (std::f64::consts::PI * ((x - line).powi(2) + half * half))
                    })
                    .sum();
                model_defect = model_defect.max((model - y).abs());
                peak = peak.max(model);
            }
            let closed: f64 = lines[element]
                .iter()
                .map(|(line, weight)| {
                    weight / std::f64::consts::PI
                        * (((high - line) / half).atan() - ((low - line) / half).atan())
                })
                .sum();
            let sampled = area(axis, values);
            println!(
                "couplings {couplings}, element {element}: lines {:?}, model defect {model_defect:.3e}, \
                 area {sampled:.8} against {closed:.8}",
                lines[element]
            );
            assert!(model_defect < 1e-9 * peak);
            assert!((sampled - closed).abs() < 1e-4);
        }
    }
}

fn ammonia() -> System {
    // C3v ammonia (Bohr): N on the axis, three hydrogens at 120 degrees.
    let (r, z) = (1.7720, -0.6800);
    let hydrogens = (0..3).map(|k| {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
        [r * angle.cos(), r * angle.sin(), z]
    });
    let mut positions = vec![[0.0, 0.0, 0.0]];
    positions.extend(hydrogens);
    System::molecule(vec![7, 1, 1, 1], positions)
}

fn mcd_request() -> Vec<(&'static str, Value)> {
    let mut request = tight();
    request.push(("nstates", Value::Int(6)));
    request
}

#[test]
fn faraday_terms_follow_the_symmetry_of_the_molecule() {
    let left = run("mcd", &peroxide(false), &controls(&mcd_request())).unwrap();
    let right = run("mcd", &peroxide(true), &controls(&mcd_request())).unwrap();
    let turned = run(
        "mcd",
        &moved(&peroxide(false), Some(rotation()), [1.0, -0.5, 2.0]),
        &controls(&mcd_request()),
    )
    .unwrap();
    let b = vector(&left, "b_terms");
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mirror = largest_difference(&b, &vector(&right, "b_terms"));
    let motion = largest_difference(&b, &vector(&turned, "b_terms"));
    let sizes = vector(&left, "level_sizes");
    let a = vector(&left, "a_terms");
    println!(
        "peroxide B terms {b:?}: mirror {mirror:.3e}, rotation and translation {motion:.3e}; \
         level sizes {sizes:?}, A terms {a:?}"
    );
    assert!(scale > 0.0);
    assert!(mirror < 1e-7 * scale);
    assert!(motion < 1e-6 * scale);
    assert!(sizes.iter().all(|s| *s == 1.0));
    assert!(a.iter().all(|v| *v == 0.0));

    let upright = run("mcd", &ammonia(), &controls(&mcd_request())).unwrap();
    let tilted = run(
        "mcd",
        &moved(&ammonia(), Some(rotation()), [0.0; 3]),
        &controls(&mcd_request()),
    )
    .unwrap();
    // Within a degenerate level the terms of one state depend on the basis
    // chosen in that level; the sums over complete levels do not.
    let sizes = vector(&upright, "level_sizes");
    let level_sums = |report: &Value, key: &str| -> Vec<f64> {
        let values = vector(report, key);
        let sizes = vector(report, "level_sizes");
        let mut sums = Vec::new();
        let mut j = 0;
        while j < values.len() {
            let size = sizes[j] as usize;
            if j + size > values.len() {
                break;
            }
            sums.push(values[j..j + size].iter().sum());
            j += size;
        }
        sums
    };
    let a_upright = level_sums(&upright, "a_terms");
    let b_upright = level_sums(&upright, "b_terms");
    let a_scale = a_upright.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let b_scale = b_upright.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let a_rotated = largest_difference(&a_upright, &level_sums(&tilted, "a_terms"));
    let b_rotated = largest_difference(&b_upright, &level_sums(&tilted, "b_terms"));
    println!(
        "ammonia level sizes {sizes:?}, level A terms {a_upright:?} (rotation {a_rotated:.3e}),          level B terms {b_upright:?} (rotation {b_rotated:.3e})"
    );
    assert!(sizes.contains(&2.0), "C3v ammonia has degenerate E states");
    assert!(a_scale > 0.0);
    assert!(a_rotated < 1e-6 * a_scale);
    assert!(b_rotated < 1e-6 * b_scale);
}

struct McdProjectedCis {
    h0: Matrix,
    dipole: Vec<Matrix>,
    field_imaginary: Vec<Matrix>,
    energies: Vec<f64>,
    vectors: Matrix,
}

/// Slater--Condon one-body matrix in the reference plus normalized singlet
/// singles. This includes the complete ground diagonal on EVERY state;
/// no MCD sum-over-states expression is used in this oracle.
fn mcd_cis_operator(o: &Matrix, occupied: usize, constant: f64) -> Matrix {
    let virtuals = o.rows - occupied;
    let size = 1 + occupied * virtuals;
    let ground = constant + 2.0 * (0..occupied).map(|i| o[(i, i)]).sum::<f64>();
    let mut result = Matrix::zeros(size, size);
    for u in 0..size {
        result[(u, u)] = ground;
    }
    for a in 0..virtuals {
        for i in 0..occupied {
            let u = 1 + a * occupied + i;
            result[(0, u)] = std::f64::consts::SQRT_2 * o[(i, occupied + a)];
            result[(u, 0)] = std::f64::consts::SQRT_2 * o[(occupied + a, i)];
            for b in 0..virtuals {
                for j in 0..occupied {
                    let v = 1 + b * occupied + j;
                    if i == j {
                        result[(u, v)] += o[(occupied + a, occupied + b)];
                    }
                    if a == b {
                        result[(u, v)] -= o[(j, i)];
                    }
                }
            }
        }
    }
    result
}

fn mcd_projected_cis(system: &System, request: &[(&str, Value)]) -> McdProjectedCis {
    use dft_rs::integrals::dipole_matrices;
    use dft_rs::properties::nmr::angular_momentum_operator;
    use dft_rs::response::tddft::{TdaSolver, TdaSpin};
    let method = MolecularMethod::from_controls(&controls(request)).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    assert!(reference.scf.converged);
    let scf = &reference.scf;
    let solver = TdaSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        TdaSpin::Singlet,
    )
    .unwrap();
    let size = solver.dim() + 1;
    let mut h0 = Matrix::zeros(size, size);
    for column in 0..size - 1 {
        let mut unit = vec![0.0; size - 1];
        unit[column] = 1.0;
        let image = solver.a_vec(&unit).unwrap();
        for row in 0..size - 1 {
            h0[(row + 1, column + 1)] = image[row];
        }
    }
    assert!(h0.max_abs_diff(&h0.transpose()) < 1e-11);
    let (energies, vectors) = dft_rs::linalg::symmetric_eigen(&h0).unwrap();
    let c = &scf.mo_coeff[0];
    let position = dipole_matrices(&setup.basis, Vec3::zero());
    let angular = angular_momentum_operator(&setup.basis, Vec3::zero());
    let mut dipole = Vec::new();
    let mut field_imaginary = Vec::new();
    let charges = setup.basis.effective_charges(&setup.molecule);
    for axis in 0..3 {
        let mut electric = c.transpose_matmul(&position[axis]).matmul_seq(c);
        for value in electric.as_mut_slice() {
            *value = -*value;
        }
        let nucleus = setup
            .molecule
            .atoms
            .iter()
            .zip(&charges)
            .map(|(a, z)| z * a.position.get(axis))
            .sum();
        dipole.push(mcd_cis_operator(&electric, scf.n_alpha, nucleus));
        let mut magnetic = c.transpose_matmul(&angular[axis]).matmul_seq(c);
        // H_B=+L/2=-i(r cross grad)/2, not the magnetic moment itself.
        for value in magnetic.as_mut_slice() {
            *value *= -0.5;
        }
        field_imaginary.push(mcd_cis_operator(&magnetic, scf.n_alpha, 0.0));
    }
    McdProjectedCis {
        h0,
        dipole,
        field_imaginary,
        energies,
        vectors,
    }
}

fn mcd_cis_circular(
    model: &McdProjectedCis,
    axis: usize,
    field: f64,
    shift: [f64; 3],
) -> (Vec<[f64; 3]>, Vec<f64>) {
    use dft_rs::periodic::BlochMatrix;
    use dft_rs::response::complex::Complex;
    let size = model.h0.rows;
    let mut imaginary = model.field_imaginary[axis].clone();
    for value in imaginary.as_mut_slice() {
        *value *= field;
    }
    let orbitals = dft_rs::dfpt::orbitals::generalized_orbitals(
        &BlochMatrix {
            re: model.h0.clone(),
            im: imaginary,
        },
        &BlochMatrix {
            re: Matrix::identity(size),
            im: Matrix::zeros(size, size),
        },
        1e-12,
    )
    .unwrap();
    let coefficients = &orbitals.coefficients;
    let mut values = Vec::new();
    let mut energies = Vec::new();
    for root in 1..size {
        let mut d = [Complex::ZERO; 3];
        for c in 0..3 {
            for r in 0..size {
                for s in 0..size {
                    let ground =
                        Complex::new(coefficients.re[(r, 0)], coefficients.im[(r, 0)]).conj();
                    let excited =
                        Complex::new(coefficients.re[(s, root)], coefficients.im[(s, root)]);
                    let mu = model.dipole[c][(r, s)] + if r == s { shift[c] } else { 0.0 };
                    d[c] += ground * excited * mu;
                }
            }
        }
        values.push([
            (d[1] * d[2].conj()).im,
            (d[2] * d[0].conj()).im,
            (d[0] * d[1].conj()).im,
        ]);
        energies.push(orbitals.energies[root] - orbitals.energies[0]);
        // The finite fields must not change which nondegenerate level is followed.
        assert!((energies[root - 1] - (model.energies[root] - model.energies[0])).abs() < 0.02);
    }
    (values, energies)
}

fn mcd_cis_analytic_axes(model: &McdProjectedCis) -> Vec<[f64; 3]> {
    let mu: Vec<_> = model
        .dipole
        .iter()
        .map(|o| model.vectors.transpose_matmul(o).matmul_seq(&model.vectors))
        .collect();
    let v: Vec<_> = model
        .field_imaginary
        .iter()
        .map(|o| model.vectors.transpose_matmul(o).matmul_seq(&model.vectors))
        .collect();
    let size = model.h0.rows;
    let mut result = Vec::new();
    for root in 1..size {
        let mut axes = [0.0; 3];
        for c in 0..3 {
            let mut imaginary = [0.0; 3];
            for a in 0..3 {
                for k in 1..size {
                    imaginary[a] +=
                        v[c][(0, k)] * mu[a][(k, root)] / (model.energies[0] - model.energies[k]);
                }
                // Includes k=0, and hence the ground permanent-dipole counterterm.
                for k in 0..size {
                    if k != root {
                        imaginary[a] += mu[a][(0, k)] * v[c][(k, root)]
                            / (model.energies[root] - model.energies[k]);
                    }
                }
            }
            let (a, b) = [(1, 2), (2, 0), (0, 1)][c];
            axes[c] = imaginary[a] * mu[b][(0, root)] - mu[a][(0, root)] * imaginary[b];
        }
        result.push(axes);
    }
    result
}

#[test]
fn faraday_b_matches_all_projected_cis_field_derivatives_and_constant_shifts() {
    let system = water();
    let mut request = tight();
    request.push(("conv_tol_grad", Value::Float(1e-11)));
    request.push(("nstates", Value::Int(10)));
    request.push(("gauge_origin", list(&[0.0; 3])));
    let mut reference_request = request.clone();
    reference_request.retain(|(k, _)| *k != "nstates" && *k != "gauge_origin");
    let model = mcd_projected_cis(&system, &reference_request);
    let native = run("mcd", &system, &controls(&request)).unwrap();
    let target = vector(&native, "b_terms");
    let excitations = vector(&native, "excitation_energies");
    assert_eq!(target.len(), 10);
    for (j, &energy) in excitations.iter().enumerate() {
        assert!((energy - model.energies[j + 1] + model.energies[0]).abs() < 1e-10);
    }
    let analytic = mcd_cis_analytic_axes(&model);
    let steps = [0.008, 0.004, 0.002, 0.001];
    let mut samples = Vec::new();
    let mut shift_defect = 0.0_f64;
    for step in steps {
        let mut derivative = vec![[0.0; 3]; 10];
        for c in 0..3 {
            let (plus, _) = mcd_cis_circular(&model, c, step, [0.0; 3]);
            let (minus, _) = mcd_cis_circular(&model, c, -step, [0.0; 3]);
            let (shifted, _) = mcd_cis_circular(&model, c, step, [0.31, -0.27, 0.19]);
            for j in 0..10 {
                derivative[j][c] = (plus[j][c] - minus[j][c]) / (2.0 * step);
                shift_defect = shift_defect.max((plus[j][c] - shifted[j][c]).abs());
            }
        }
        samples.push(derivative);
    }
    assert!(
        shift_defect < 1e-12,
        "constant-I transition defect {shift_defect}"
    );
    let mut failures = Vec::new();
    for j in 0..10 {
        for c in 0..3 {
            let errors: Vec<_> = samples
                .iter()
                .map(|s| (s[j][c] - analytic[j][c]).abs())
                .collect();
            let fit = convergence::error_power_fit(&steps, &errors);
            eprintln!(
                "MCD axis({c}) root({j}): errors={errors:?}, p={}, R2={}",
                fit.exponent, fit.r_squared
            );
            if (fit.exponent - 2.0).abs() >= 0.02 || fit.r_squared <= 0.99999 {
                failures.push(format!("axis{c}/root{j}"));
            }
        }
        let errors: Vec<_> = samples
            .iter()
            .map(|s| (s[j].iter().sum::<f64>() - target[j]).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        eprintln!(
            "MCD B root({j}): native={}, analytic={}, errors={errors:?}, p={}, R2={}",
            target[j],
            analytic[j].iter().sum::<f64>(),
            fit.exponent,
            fit.r_squared
        );
        if (fit.exponent - 2.0).abs() >= 0.02 || fit.r_squared <= 0.99999 {
            failures.push(format!("B/root{j}"));
        }
        if (target[j] - analytic[j].iter().sum::<f64>()).abs() >= 1e-9 {
            failures.push(format!("point/root{j}"));
        }
    }
    assert!(
        failures.is_empty(),
        "failed complete MCD fits: {failures:?}"
    );
    assert!(native
        .get("mcd_convention")
        .and_then(Value::as_str)
        .unwrap()
        .contains("faraday_a_terms=-a_terms"));
}

#[test]
fn faraday_atomic_p_levels_have_quantized_zeeman_slopes_and_canonical_a() {
    let system = System::molecule(vec![2], vec![[0.0; 3]]);
    let mut reference_request = tight();
    reference_request.retain(|(k, _)| *k != "basis");
    reference_request.push(("basis", Value::Str("aug-cc-pvdz".into())));
    reference_request.push(("conv_tol_grad", Value::Float(1e-11)));
    let model = mcd_projected_cis(&system, &reference_request);
    let mut request = reference_request;
    request.push(("nstates", Value::Int(8)));
    request.push(("gauge_origin", list(&[0.0; 3])));
    let native = run("mcd", &system, &controls(&request)).unwrap();
    let energy = vector(&native, "excitation_energies");
    let strength = vector(&native, "dipole_strengths");
    let legacy = vector(&native, "a_terms");
    let canonical = vector(&native, "faraday_a_terms");
    assert_eq!(canonical.len(), 8);
    for j in 0..8 {
        assert_eq!(canonical[j], -legacy[j]);
    }
    let units = native.get("units").unwrap();
    assert!(units
        .get("faraday_a_terms")
        .and_then(Value::as_str)
        .unwrap()
        .contains("canonical Faraday A"));
    let mut checked = Vec::new();
    let mut groups = 0;
    for j in 0..8 {
        if checked.contains(&j) {
            continue;
        }
        let group: Vec<_> = (0..8)
            .filter(|&k| (energy[k] - energy[j]).abs() < 1e-7)
            .collect();
        checked.extend(&group);
        if group.len() != 3 {
            continue;
        }
        groups += 1;
        let expected = 0.5 * group.iter().map(|&k| strength[k]).sum::<f64>();
        let actual = group.iter().map(|&k| canonical[k]).sum::<f64>();
        assert!((actual - expected).abs() < 1e-10);
        for step in [0.008, 0.004, 0.002, 0.001] {
            let mut line_first_moment = 0.0;
            for c in 0..3 {
                let (plus, ep) = mcd_cis_circular(&model, c, step, [0.0; 3]);
                let (minus, em) = mcd_cis_circular(&model, c, -step, [0.0; 3]);
                let mut slopes: Vec<_> =
                    group.iter().map(|&k| (ep[k] - energy[k]) / step).collect();
                slopes.sort_by(f64::total_cmp);
                for (value, exact) in slopes.iter().zip([-0.5, 0.0, 0.5]) {
                    assert!((value - exact).abs() < 1e-8);
                }
                let qp = group
                    .iter()
                    .map(|&k| plus[k][c] * (ep[k] - energy[k]))
                    .sum::<f64>();
                let qm = group
                    .iter()
                    .map(|&k| minus[k][c] * (em[k] - energy[k]))
                    .sum::<f64>();
                line_first_moment += (qp - qm) / (2.0 * step);
            }
            assert!((actual + line_first_moment).abs() < 1e-9);
        }
    }
    assert_eq!(groups, 2);
}
