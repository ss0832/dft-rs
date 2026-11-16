// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use dft_rs::driver::method::{
    prepare_molecular, run_molecular_reference, MolecularMethod, MolecularSetup,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::london::{one_electron_series, ComplexMatrix, FieldPath, OneElectronSeries};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::BlochMatrix;
use dft_rs::response::complex::Complex;
use dft_rs::response::field_path::{
    causal_faraday_tensor, causal_spin_orbit_polarizability, faraday_tensor,
};
use dft_rs::response::{Frequency, ResponseSolverConfig};

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

const POSITIONS: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.19],
    [1.57, 0.16, -0.08],
    [-0.19, 1.93, 0.37],
    [0.42, 0.29, 2.28],
];
const ORIGIN: [f64; 3] = [0.23, -0.50, 0.42];
const OMEGA: f64 = 0.18;
const DAMPING: f64 = 0.028;

fn model() -> Value {
    Value::object()
        .with("reference", "uks")
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("charge", 3)
        .with("multiplicity", 2)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-13)
        .with("conv_tol_grad", 1e-13)
}

fn sum_integrals(series: Vec<ComplexMatrix>) -> BlochMatrix {
    let mut matrices = series.into_iter();
    let first = matrices.next().unwrap();
    let mut value = BlochMatrix::from_pair((first.re, first.im));
    for next in matrices {
        value.add_scaled(&BlochMatrix::from_pair((next.re, next.im)), 1.0);
    }
    value
}

/// A one-electron HF state has no electron interaction in its excitation
/// spectrum. Diagonalize the finite-field core pencil and sum both spectral
/// poles over the entire virtual space.
fn spectral(
    setup: &MolecularSetup,
    field: [f64; 3],
    z: Complex,
    order: usize,
) -> [[Complex; 3]; 3] {
    let integrals = one_electron_series(
        &setup.molecule,
        &setup.basis,
        &FieldPath {
            electric: [0.0; 3],
            magnetic: field,
            origin: ORIGIN,
            london: true,
            order,
        },
    )
    .unwrap();
    spectral_core(integrals, z, None)
}

fn spectral_core(
    integrals: OneElectronSeries,
    z: Complex,
    spin_orbit: Option<(&Matrix, f64)>,
) -> [[Complex; 3]; 3] {
    let mut core = sum_integrals(integrals.core);
    if let Some((operator, strength)) = spin_orbit {
        for (output, source) in core.im.as_mut_slice().iter_mut().zip(operator.as_slice()) {
            *output -= 0.5 * strength * source;
        }
    }
    let overlap = sum_integrals(integrals.overlap);
    let states = generalized_orbitals(&core, &overlap, 1e-12).unwrap();
    assert!(states.eigen_residual < 1e-12 && states.metric_defect < 1e-12);
    let c = &states.coefficients;
    let ct = adjoint(c).unwrap();
    let dipoles = integrals
        .dipole
        .map(|series| matmul(&ct, &matmul(&sum_integrals(series), c).unwrap()).unwrap());
    let dipole = |a: usize, v| Complex::new(dipoles[a].re[(0, v)], dipoles[a].im[(0, v)]);
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            (1..states.energies.len()).fold(Complex::ZERO, |sum, v| {
                let gap = Complex::from(states.energies[v] - states.energies[0]);
                let numerator = dipole(a, v) * dipole(b, v).conj();
                sum + numerator / (gap - z) + numerator.conj() / (gap + z)
            })
        })
    })
}

#[test]
fn causal_faraday_all_independent_components_match_finite_london_spectra() {
    let method = MolecularMethod::from_controls(&Controls::from_value(&model()).unwrap()).unwrap();
    let setup = prepare_molecular(&[1; 4], &POSITIONS, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    assert!(reference.scf.converged);
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..ResponseSolverConfig::default()
    };
    let solve = |omega, damping, origin| {
        causal_faraday_tensor(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            origin,
            true,
            Frequency::Retarded { omega, damping },
            &config,
        )
        .unwrap()
    };
    let value = solve(OMEGA, DAMPING, Vec3::new(ORIGIN[0], ORIGIN[1], ORIGIN[2]));
    let negative = solve(-OMEGA, DAMPING, Vec3::new(ORIGIN[0], ORIGIN[1], ORIGIN[2]));
    let translated = solve(OMEGA, DAMPING, Vec3::new(-0.17, 0.38, 0.05));
    let undamped = solve(OMEGA, 0.0, Vec3::new(ORIGIN[0], ORIGIN[1], ORIGIN[2]));
    let legacy = faraday_tensor(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        Vec3::new(ORIGIN[0], ORIGIN[1], ORIGIN[2]),
        true,
        OMEGA,
        &config,
    )
    .unwrap();
    let z = Complex::new(OMEGA, DAMPING);
    let centre = spectral(&setup, [0.0; 3], z, 0);
    let mut worst = [0.0_f64; 4];
    for (a, row) in centre.iter().enumerate() {
        for (b, alpha) in row.iter().enumerate() {
            worst[0] = worst[0].max((value.polarizability[a][b] - *alpha).abs());
            for c in 0..3 {
                let entry = value.derivative[a][b][c];
                worst[1] = worst[1].max((entry + value.derivative[b][a][c]).abs());
                worst[2] = worst[2].max((entry.conj() - negative.derivative[a][b][c]).abs());
                worst[3] = worst[3].max((entry - translated.derivative[a][b][c]).abs());
                assert!((undamped.derivative[a][b][c].im - legacy.1[a][b][c]).abs() < 1e-13);
            }
        }
    }
    assert!(worst.iter().all(|x| *x < 2e-10), "{worst:?}");
    assert!(value.residual < 1e-12);
    println!("Causal London Faraday tensor {:?} residual {:.9e} reference defect {:.9e} checks {worst:?}",
             value.derivative, value.residual, value.reference_defect);
    for c in 0..3 {
        let mut field = [0.0; 3];
        field[c] = 0.032;
        let low = spectral(&setup, field, z, 10);
        let high = spectral(&setup, field, z, 14);
        assert!(low
            .iter()
            .flatten()
            .zip(high.iter().flatten())
            .all(|(x, y)| (*x - *y).abs() < 2e-13));
    }
    fit_derivative("Causal Faraday", &value.derivative, |c, b| {
        let mut field = [0.0; 3];
        field[c] = b;
        spectral(&setup, field, z, 10)
    });
}

#[test]
fn causal_spin_orbit_matches_finite_one_electron_spectra() {
    let method = MolecularMethod::from_controls(&Controls::from_value(&model()).unwrap()).unwrap();
    let setup = prepare_molecular(&[1; 4], &POSITIONS, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    assert!(reference.scf.converged);
    let orbital =
        dft_rs::properties::nmr::paramagnetic_operator(&setup.basis, Vec3::new(0.21, 0.31, -0.14));
    let frequency = Frequency::Retarded {
        omega: OMEGA,
        damping: DAMPING,
    };
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..ResponseSolverConfig::default()
    };
    let value = causal_spin_orbit_polarizability(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &reference.scf,
        &setup.options,
        &orbital,
        frequency,
        &config,
    )
    .unwrap();
    let integrals = one_electron_series(
        &setup.molecule,
        &setup.basis,
        &FieldPath {
            electric: [0.0; 3],
            magnetic: [0.0; 3],
            origin: ORIGIN,
            london: false,
            order: 0,
        },
    )
    .unwrap();
    assert!(value.residual < 1e-12);
    for (a, row) in value.derivative.iter().enumerate() {
        for (b, entries) in row.iter().enumerate() {
            assert!(entries
                .iter()
                .zip(&value.derivative[b][a])
                .all(|(x, y)| (*x + *y).abs() < 2e-10));
        }
    }
    println!(
        "Causal spin-orbit tensor {:?} residual {:.9e}",
        value.derivative, value.residual
    );
    let z = frequency.energy().unwrap();
    fit_derivative("Causal spin-orbit", &value.derivative, |axis, strength| {
        spectral_core(integrals.clone(), z, Some((&orbital[axis], strength)))
    });
}

fn fit_derivative(
    label: &str,
    derivative: &[[[Complex; 3]; 3]; 3],
    calculate: impl Fn(usize, f64) -> [[Complex; 3]; 3],
) {
    let pairs = [(0, 1), (0, 2), (1, 2)];
    let names: Vec<_> = (0..3)
        .flat_map(|c| {
            pairs.into_iter().flat_map(move |(a, b)| {
                ["real", "imag"].map(|part| format!("{label}[{a},{b},{c}] {part}"))
            })
        })
        .collect();
    let mut sample = std::collections::BTreeMap::new();
    let mut failures = Vec::new();
    let intervals = [
        ([0.016, 0.012, 0.008, 0.006], stencils::Stencil::Second),
        ([0.008, 0.006, 0.004, 0.003], stencils::Stencil::Second),
        ([0.024, 0.018, 0.012, 0.009], stencils::Stencil::Second),
        ([0.016, 0.012, 0.008, 0.006], stencils::Stencil::Fourth),
    ];
    let mut at = |c, b: f64| {
        *sample
            .entry((c, b.to_bits()))
            .or_insert_with(|| calculate(c, b))
    };
    stencils::fit_components_staged(
        &names,
        &intervals,
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; names.len()];
            for c in 0..3 {
                for (i, &h) in steps.iter().enumerate() {
                    let plus = at(c, h);
                    let minus = at(c, -h);
                    let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                        (at(c, 2.0 * h), at(c, -2.0 * h))
                    } else {
                        ([[Complex::ZERO; 3]; 3], [[Complex::ZERO; 3]; 3])
                    };
                    for (k, &(a, b)) in pairs.iter().enumerate() {
                        let parts = |part: fn(Complex) -> f64| {
                            stencil.first(
                                part(plus[a][b]),
                                part(minus[a][b]),
                                part(plus2[a][b]),
                                part(minus2[a][b]),
                                h,
                            )
                        };
                        let exact = derivative[a][b][c];
                        errors[6 * c + 2 * k][i] = (parts(|z| z.re) - exact.re).abs();
                        errors[6 * c + 2 * k + 1][i] = (parts(|z| z.im) - exact.im).abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(label, &failures);
}

#[test]
fn magnetic_medium_common_driver_is_odd_under_field_reversal() {
    let molecule = System::molecule(vec![1; 4], POSITIONS.to_vec());
    let common = model()
        .with("frequency", OMEGA)
        .with("frequency_damping", DAMPING)
        .with("gauge_origin", ORIGIN)
        .with("medium_number_density", 0.002)
        .with("slab_thickness", 1300.0)
        .with("incidence_angle", 0.31)
        .with("exit_refractive_index", 1.3);
    let calculate = |b| {
        run(
            "molecular-medium-optics",
            &molecule,
            &Controls::from_value(&common.clone().with("medium_magnetic_field", [0.0, 0.0, b]))
                .unwrap(),
        )
        .unwrap()
    };
    let plus = calculate(0.003);
    let minus = calculate(-0.003);
    let values = |r: &Value, key| r.get(key).unwrap().numeric_array().unwrap().1;
    for key in ["magnetic_kerr_rotation", "magnetic_kerr_ellipticity"] {
        let pos = values(&plus, key);
        let neg = values(&minus, key);
        assert!(pos.iter().any(|x| x.abs() > 1e-9), "{key} {pos:?}");
        assert!(pos.iter().zip(neg).all(|(p, n)| (p + n).abs() < 1e-12));
    }
    assert_eq!(
        values(&plus, "reflectance"),
        values(&minus, "reversed_field_reflectance")
    );
    println!("Magnetic molecular medium optics {}", plus.to_json());
}
