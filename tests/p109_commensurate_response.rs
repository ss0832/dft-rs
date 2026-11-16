// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Response of a periodic cell at wave vectors commensurate with the k mesh
//! on the common driver: sign, bounds and the cosine/sine balance of the
//! density response, and the sign and polarization dependence of the
//! orbital magnetic susceptibility. The restricted LDA orbital response
//! also uses an independent complex spectrum and finite perturbations of
//! a separately converged Fock matrix on a skew cell.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::basis::{g94, MolBasis};
use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul, BandOrbitals};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_scf::StaticPeriodicField;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::system::{Atom, Molecule};

const CELL: [[f64; 3]; 3] = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]];
const POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]];

fn chain() -> System {
    System {
        numbers: vec![1, 1],
        positions: POSITIONS.to_vec(),
        cell: Some(CELL),
        pbc: [true; 3],
    }
}

fn controls(extra: &[(&str, Value)]) -> Controls {
    let mut pairs = vec![
        ("xc".to_string(), Value::Str("svwn5".into())),
        ("basis".to_string(), Value::Str("sto-3g".into())),
        (
            "k_mesh".to_string(),
            Value::Array(vec![Value::Int(4), Value::Int(1), Value::Int(1)]),
        ),
        ("overlap_cutoff".to_string(), Value::Float(20.0)),
        ("grid_level".to_string(), Value::Int(2)),
        ("grid_shell".to_string(), Value::Int(2)),
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_dp".to_string(), Value::Float(1e-10)),
        ("max_iter".to_string(), Value::Int(200)),
    ];
    for (key, value) in extra {
        pairs.retain(|(k, _)| k != key);
        pairs.push((key.to_string(), value.clone()));
    }
    Controls::from_value(&Value::Object(pairs)).unwrap()
}

fn q_points() -> Value {
    Value::Array(
        [0.25, 0.0, 0.0, 0.5, 0.0, 0.0]
            .iter()
            .map(|x| Value::Float(*x))
            .collect(),
    )
}

fn points(report: &Value) -> Vec<Value> {
    match report.get("points") {
        Some(Value::Array(list)) => list.clone(),
        other => panic!("points missing: {other:?}"),
    }
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

#[test]
fn density_response_is_negative_and_screens() {
    let report = run(
        "dielectric-q",
        &chain(),
        &controls(&[("q_points", q_points())]),
    )
    .unwrap();
    let list = points(&report);
    assert_eq!(list.len(), 2);
    for point in &list {
        let chi = number(point, "chi");
        let inverse = number(point, "inverse_dielectric");
        println!(
            "density response: chi {chi:.8e}, cosine {:.8e}, sine {:.8e}, 1/epsilon {inverse:.8}",
            number(point, "chi_cosine"),
            number(point, "chi_sine")
        );
        assert!(chi < 0.0);
        assert!(inverse > 0.0 && inverse < 1.0);
    }
    // Away from the zone boundary the two waves respond alike.
    let interior = &list[0];
    let (cosine, sine) = (number(interior, "chi_cosine"), number(interior, "chi_sine"));
    assert!((cosine - sine).abs() < 1e-6 * cosine.abs().max(sine.abs()));
}

#[test]
fn orbital_susceptibility_is_finite_for_both_polarizations() {
    let report = run(
        "orbital-susceptibility-q",
        &chain(),
        &controls(&[("q_points", q_points())]),
    )
    .unwrap();
    for point in points(&report) {
        let chi = point.get("chi").unwrap().numeric_array().unwrap().1;
        let paramagnetic = point
            .get("paramagnetic")
            .unwrap()
            .numeric_array()
            .unwrap()
            .1;
        let uniform = point.get("uniform").unwrap().numeric_array().unwrap().1;
        println!("orbital susceptibility: chi {chi:?}, paramagnetic {paramagnetic:?}, uniform {uniform:?}");
        assert_eq!(chi.len(), 2);
        // Second-order energies of a Hermitian perturbation are negative.
        assert!(paramagnetic.iter().chain(&uniform).all(|e| *e < 0.0));
        assert!(chi.iter().all(|x| x.is_finite()));
    }
}

fn in_orbital_basis(operator: &BlochMatrix, orbitals: &BandOrbitals) -> BlochMatrix {
    let c = &orbitals.coefficients;
    matmul(&adjoint(c).unwrap(), &matmul(operator, c).unwrap()).unwrap()
}

#[test]
fn lda_orbital_susceptibility_matches_independent_transverse_spectra() {
    const REPLICAS: f64 = 2.0;
    const OCCUPIED: usize = 2;
    const CUTOFF: f64 = 12.0;
    const BASIS: &str = "H 0\nS 1 1.0\n0.7 1.0\n****\n";
    let rows = [[3.6, 0.19, -0.23], [0.31, 7.0, 0.27], [-0.22, 0.41, 7.0]];
    let positions = [[0.11, -0.13, 0.17], [1.54, 0.17, -0.12]];
    let primitive = System {
        numbers: vec![1, 1],
        positions: positions.to_vec(),
        cell: Some(rows),
        pbc: [true; 3],
    };
    let doubled = [rows[0].map(|v| 2.0 * v), rows[1], rows[2]];
    let supercell = System {
        numbers: vec![1; 4],
        positions: positions
            .into_iter()
            .chain(positions.map(|r| std::array::from_fn(|i| r[i] + rows[0][i])))
            .collect(),
        cell: Some(doubled),
        pbc: [true; 3],
    };
    let path = std::env::temp_dir().join(format!("dft-rs-orbital-q-{}.g94", std::process::id()));
    std::fs::write(&path, BASIS).unwrap();
    let base = [
        ("basis", Value::Str(path.to_str().unwrap().into())),
        ("grid_weight_cutoff", Value::Float(0.0)),
        ("overlap_cutoff", Value::Float(CUTOFF)),
        ("conv_tol", Value::Float(1e-13)),
        ("conv_tol_dp", Value::Float(1e-12)),
        ("conv_tol_grad", Value::Float(1e-11)),
    ];
    let analytical = run(
        "orbital-susceptibility-q",
        &primitive,
        &controls(
            &[
                base.as_slice(),
                &[
                    ("k_mesh", Value::Array([2, 1, 1].map(Value::Int).to_vec())),
                    (
                        "q_points",
                        Value::Array([0.5, 0.0, 0.0].map(Value::Float).to_vec()),
                    ),
                ],
            ]
            .concat(),
        ),
    );
    let stationary = run(
        "periodic-field-scf",
        &supercell,
        &controls(
            &[
                base.as_slice(),
                &[
                    ("k_mesh", Value::Array([1, 1, 1].map(Value::Int).to_vec())),
                    (
                        "field_wave_indices",
                        Value::Array([0, 0, 0].map(Value::Int).to_vec()),
                    ),
                ],
            ]
            .concat(),
        ),
    );
    std::fs::remove_file(path).unwrap();
    let analytical = analytical.unwrap();
    let stationary = stationary.unwrap();
    assert_eq!(stationary.get("converged"), Some(&Value::Bool(true)));
    let reference_energy_defect =
        (number(&stationary, "energy") - number(&analytical, "energy")).abs();
    let orbital_residual = number(&stationary, "orbital_residual");
    assert!(reference_energy_defect < 1e-8 && orbital_residual < 1e-11);
    println!("transverse spectral reference difference {reference_energy_defect:.6e} Hartree per supercell, orbital residual {orbital_residual:.6e} Hartree");
    let reported = points(&analytical);
    assert_eq!(reported.len(), 1);
    let report = &reported[0];
    let array = |v: &Value, key: &str| v.get(key).unwrap().numeric_array().unwrap().1;
    let vec3 = |r: &[f64]| Vec3::new(r[0], r[1], r[2]);
    let wave = vec3(&array(report, "q"));
    let volume = number(&analytical, "volume");
    let vectors = doubled.map(|r| vec3(&r));
    let geometry = PeriodicGeometry::new(
        Lattice::new(vectors[0], vectors[1], vectors[2]),
        PeriodicBoundary::FULL_3D,
    )
    .unwrap();
    let molecule = Molecule::new(
        supercell
            .positions
            .iter()
            .map(|r| Atom {
                z: 1,
                position: vec3(r),
            })
            .collect(),
    );
    let basis =
        MolBasis::build(&molecule, &g94::parse("orbital-q-oracle", BASIS).unwrap()).unwrap();
    let fourier = |q, vector_potential| {
        StaticPeriodicField {
            wave: q,
            vector_potential,
            scalar_potential: Complex::ZERO,
            amplitude: 0.0,
        }
        .operators(&basis, &geometry, &[Vec3::zero()], CUTOFF)
        .unwrap()
    };
    let overlap = dft_rs::periodic::field::periodic_field_operators(
        &molecule,
        &basis,
        &geometry,
        &dft_rs::periodic::field::PeriodicFieldOptions {
            basis_cutoff: CUTOFF,
            ..Default::default()
        },
    )
    .unwrap()
    .density_vertex
    .remove(0);
    let n = basis.nao_sph;
    let spatial = |key| {
        let full = array(&stationary, key);
        assert!(full.len() >= n * n);
        Matrix::from_row_major(n, n, full[..n * n].to_vec())
    };
    let fock = BlochMatrix {
        re: spatial("fock_real"),
        im: spatial("fock_imag"),
    };
    let canonical = |matrix: &BlochMatrix| generalized_orbitals(matrix, &overlap, 1e-10).unwrap();
    let reference = canonical(&fock);
    let gap = |i: usize, a: usize| reference.energies[a] - reference.energies[i];
    let closed = |operator: &BlochMatrix| {
        let v = in_orbital_basis(operator, &reference);
        -(2.0 / REPLICAS)
            * (0..OCCUPIED)
                .flat_map(|i| (OCCUPIED..n).map(move |a| (i, a)))
                .map(|(i, a)| (v.re[(a, i)].powi(2) + v.im[(a, i)].powi(2)) / gap(i, a))
                .sum::<f64>()
    };
    // At a real LDA reference, the first-order density of an orbital current
    // perturbation is imaginary. Hartree and LDA respond to real density only.
    let slope = |operator: &BlochMatrix, h: f64| {
        let mut biased = fock.clone();
        biased.add_scaled(operator, h);
        let v = in_orbital_basis(operator, &canonical(&biased));
        2.0 * (0..OCCUPIED).map(|i| v.re[(i, i)]).sum::<f64>()
    };
    let extract = |[cosine, sine, flat]: [f64; 3]| {
        let paramagnetic = (cosine + sine) * 0.5;
        let chi = -4.0 * (paramagnetic - 0.5 * flat) / (volume * wave.norm2());
        [paramagnetic, flat, chi]
    };
    let polarizations = array(report, "polarization");
    let labels = ["paramagnetic", "uniform", "chi"];
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut failures = Vec::new();
    for (axis, polarization) in polarizations.chunks_exact(3).enumerate() {
        let real = std::array::from_fn(|i| Complex::from(polarization[i] * 0.5));
        let imag = std::array::from_fn(|i| Complex::new(0.0, -polarization[i] * 0.5));
        let operators = [
            fourier(wave, real).first_derivative.remove(0),
            fourier(wave, imag).first_derivative.remove(0),
            fourier(Vec3::zero(), real).first_derivative.remove(0),
        ];
        let expected = labels.map(|label| array(report, label)[axis]);
        let exact = extract(std::array::from_fn(|i| closed(&operators[i])));
        let estimate = |h, stencil: stencils::Stencil| {
            extract(std::array::from_fn(|i| {
                let operator = &operators[i];
                stencil.first(
                    slope(operator, h),
                    slope(operator, -h),
                    slope(operator, 2.0 * h),
                    slope(operator, -2.0 * h),
                    h,
                ) / (2.0 * REPLICAS)
            }))
        };
        let richardson = estimate(steps[3], stencils::Stencil::Fourth);
        for i in 0..3 {
            let oracle = (exact[i] - expected[i]).abs();
            let richardson_defect = (richardson[i] - expected[i]).abs();
            println!("transverse axis {axis} {} value {:.14e} dense defect {oracle:.6e} Richardson {:.6e}",
                labels[i], expected[i], richardson_defect);
            assert!(oracle < 1e-9 && richardson_defect < 3e-9);
        }
        stencils::fit_components_staged(
            &labels.map(|name| format!("transverse axis {axis} {name}")),
            &[(steps, stencils::Stencil::Second)],
            |steps, stencil| {
                let samples = steps.map(|h| estimate(h, stencil));
                (0..3)
                    .map(|i| samples.map(|value| (value[i] - expected[i]).abs()))
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("orbital susceptibility", &failures);
}
