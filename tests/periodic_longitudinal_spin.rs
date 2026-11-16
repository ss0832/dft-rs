// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent Gaussian source, dense screening and complete component fits.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
use dft_rs::dfpt::orbitals::{band_density, generalized_orbitals};
use dft_rs::dfpt::spin::periodic_longitudinal_spin_response;
use dft_rs::grid::GridOptions;
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::periodic::scf::{
    fock_at_density, run_periodic_scf, PeriodicScfOptions, PeriodicScfResult,
};
use dft_rs::periodic::{BlochMatrix, Cell, Lattice};
use dft_rs::response::complex::Complex;
use dft_rs::response::krylov::LinearOperator;
use dft_rs::response::Frequency;
use dft_rs::xc::ResolvedFunctional;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, Vec3};

const POSITIONS: [[f64; 3]; 4] = [
    [0.1, -0.2, 0.3],
    [1.35, 0.1, 0.2],
    [-0.3, 3.2, 0.9],
    [0.2, 3.4, 2.2],
];
fn setup() -> (Cell, PeriodicScfOptions) {
    c1_geometry::assert_c1(&[1; 4], &POSITIONS);
    let mut basis = BasisSet::new("compact C1 longitudinal spin fixture");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.91],
            coefs: vec![1.],
        }],
    });
    let atoms = POSITIONS
        .iter()
        .map(|p| Atom {
            z: 1,
            position: Vec3::new(p[0], p[1], p[2]),
        })
        .collect();
    let cell = Cell::new(
        Molecule::new(atoms),
        Lattice::new(
            Vec3::new(8.7, 0.31, 0.13),
            Vec3::new(0.2, 9.2, 0.43),
            Vec3::new(0.11, 0.27, 9.7),
        ),
        &basis,
        4.5,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        exchange_cutoff: Some(4.5),
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            weight_cutoff: 0.,
            ..Default::default()
        },
        ..Default::default()
    };
    (cell, options)
}

// Every pair is inside the AO reach and every nonzero lattice image is
// outside it. Normalized equal-exponent s Gaussians give this closed Fourier
// integral; no production Fourier-integral routine supplies this oracle.
fn gaussian_vertex(q: Vec3) -> BlochMatrix {
    let mut n = BlochMatrix::zeros(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            let a = Vec3::new(POSITIONS[i][0], POSITIONS[i][1], POSITIONS[i][2]);
            let b = Vec3::new(POSITIONS[j][0], POSITIONS[j][1], POSITIONS[j][2]);
            assert!((a - b).norm() < 4.5);
            let weight = (-0.455 * (a - b).dot(a - b) - q.dot(q) / (8. * 0.91)).exp();
            let phase = q.dot((a + b) * 0.5);
            n.re[(i, j)] = weight * phase.cos();
            n.im[(i, j)] = weight * phase.sin();
        }
    }
    n
}

fn spin_source(vertex: &BlochMatrix, field: Complex, g: f64) -> [Vec<BlochMatrix>; 2] {
    std::array::from_fn(|s| {
        let mut v = BlochMatrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let z = Complex::new(vertex.re[(i, j)], vertex.im[(i, j)])
                    * field
                    * (if s == 0 { g / 4. } else { -g / 4. });
                v.re[(i, j)] = z.re;
                v.im[(i, j)] = z.im;
            }
        }
        vec![v]
    })
}

#[test]
fn independent_gaussian_zeeman_source_matches_every_dense_complex_density_entry() {
    let (cell, options) = setup();
    for name in ["svwn5", "pbe", "tpss", "pbe0", "cam-b3lyp"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged, "{name}");
        for frequency in [
            Frequency::Static,
            Frequency::Retarded {
                omega: 0.13,
                damping: 0.04,
            },
            Frequency::Imaginary { frequency: 0.17 },
        ] {
            let request = PeriodicResponseOptions {
                q: Vec3::new(0.043, 0.017, -0.021),
                frequency,
                reciprocal_cutoff: 1.,
                fixed_number: false,
                ..Default::default()
            };
            let amplitude = Complex::new(0.31, -0.23);
            let g = 2.1;
            let actual = periodic_longitudinal_spin_response(
                &cell,
                &functional,
                &reference,
                &options,
                &request,
                amplitude,
                g,
            )
            .unwrap();
            let solver =
                PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &request)
                    .unwrap();
            let n = gaussian_vertex(request.q);
            let source = spin_source(&n, amplitude, g);
            let rhs = solver
                .independent_response(&solver.potential_to_bands(&source).unwrap())
                .unwrap();
            let dimension = solver.dimension();
            let mut dense = Matrix::zeros(2 * dimension, 2 * dimension);
            for j in 0..dimension {
                let mut unit = vec![Complex::ZERO; dimension];
                unit[j] = Complex::ONE;
                let image = solver.apply(&unit).unwrap();
                for i in 0..dimension {
                    dense[(i, j)] = image[i].re;
                    dense[(dimension + i, j)] = image[i].im;
                    dense[(i, dimension + j)] = -image[i].im;
                    dense[(dimension + i, dimension + j)] = image[i].re;
                }
            }
            let b = rhs
                .iter()
                .map(|v| v.re)
                .chain(rhs.iter().map(|v| v.im))
                .collect::<Vec<_>>();
            let x = solve_linear(&dense, &b).unwrap();
            let exact = solver
                .density_to_aos(
                    &(0..dimension)
                        .map(|i| Complex::new(x[i], x[dimension + i]))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut defect = 0_f64;
            for s in 0..2 {
                for i in 0..4 {
                    for j in 0..4 {
                        let a = &actual.value.density_response.density[s][0];
                        let e = &exact[s][0];
                        defect = defect
                            .max((a.re[(i, j)] - e.re[(i, j)]).abs())
                            .max((a.im[(i, j)] - e.im[(i, j)]).abs());
                    }
                }
            }
            let spin_number: [Complex; 2] = std::array::from_fn(|s| {
                let mut total = Complex::ZERO;
                for i in 0..4 {
                    for j in 0..4 {
                        total += Complex::new(n.re[(i, j)], -n.im[(i, j)])
                            * Complex::new(exact[s][0].re[(i, j)], exact[s][0].im[(i, j)]);
                    }
                }
                total
            });
            let moment = (spin_number[0] - spin_number[1]) * (-g / 4.);
            assert!(defect < 1e-10, "{name} {frequency:?}: {defect:e}");
            assert!((moment - actual.value.magnetic_moment).abs() < 1e-10);
            assert!((moment / amplitude - actual.value.susceptibility.unwrap()).abs() < 1e-10);
            assert!(actual.converged && actual.residuals[0] <= 1e-10);
            println!(
                "longitudinal spin {name} {frequency:?}: all-entry dense defect={defect:.12e}"
            );
        }
    }
}

#[test]
fn collisionless_uniform_response_and_equilibrium_static_limit_are_distinct() {
    let (cell, options) = setup();
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    for frequency in [
        Frequency::Retarded {
            omega: 0.13,
            damping: 0.04,
        },
        Frequency::Imaginary { frequency: 0.17 },
    ] {
        let request = PeriodicResponseOptions {
            frequency,
            reciprocal_cutoff: 1.,
            ..Default::default()
        };
        let response = periodic_longitudinal_spin_response(
            &cell,
            &functional,
            &reference,
            &options,
            &request,
            Complex::ONE,
            2.,
        )
        .unwrap();
        assert!(response.value.magnetic_moment.abs() < 1e-11);
        for p in response.value.density_response.density.iter().flatten() {
            assert!(p
                .re
                .as_slice()
                .iter()
                .chain(p.im.as_slice())
                .all(|x| x.abs() < 1e-11));
        }
    }
    assert!(periodic_longitudinal_spin_response(
        &cell,
        &functional,
        &reference,
        &options,
        &PeriodicResponseOptions::default(),
        Complex::ONE,
        2.
    )
    .unwrap_err()
    .to_string()
    .contains("common-total-number"));
}

#[test]
fn thermal_spin_probe_retains_fermi_weights_and_refuses_the_wrong_static_ensemble() {
    let (cell, mut options) = setup();
    options.occupation = dft_rs::periodic::occupations::OccupationScheme::FermiDirac {
        temperature_kelvin: 40000.,
    };
    let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    assert!(reference
        .occupations
        .iter()
        .flatten()
        .flatten()
        .any(|f| *f > 1e-5 && *f < 1. - 1e-5));
    let request = PeriodicResponseOptions {
        frequency: Frequency::Retarded {
            omega: 0.13,
            damping: 0.04,
        },
        reciprocal_cutoff: 12.,
        ..Default::default()
    };
    let collisionless = periodic_longitudinal_spin_response(
        &cell,
        &functional,
        &reference,
        &options,
        &request,
        Complex::ONE,
        2.,
    )
    .unwrap();
    assert!(collisionless.value.magnetic_moment.abs() < 1e-11);
    let equilibrium = dft_rs::periodic::spin_susceptibility::longitudinal_spin_susceptibility(
        &cell,
        &functional,
        &reference,
        &options,
        &request.solver,
        12.,
        2.,
    )
    .unwrap();
    assert!(equilibrium.value.susceptibility.abs() > 1e-8);
    assert!(equilibrium.value.charge_constraint_residual < 1e-10);
    let mut finite_q = request.clone();
    finite_q.q = Vec3::new(0.043, 0.017, -0.021);
    let warm = periodic_longitudinal_spin_response(
        &cell,
        &functional,
        &reference,
        &options,
        &finite_q,
        Complex::ONE,
        2.,
    )
    .unwrap();
    assert!(warm.converged && warm.residuals[0] <= 1e-10);
    assert!(warm.value.density_response.temperature_hartree > 0.);
    assert!(warm.value.density_response.occupation_consistency_defect < 1e-10);
    assert!(warm.value.magnetic_moment.abs() > 1e-12);
    let mut static_reciprocal = finite_q;
    static_reciprocal.q = cell.lattice().reciprocal()[0];
    static_reciprocal.frequency = Frequency::Static;
    assert!(periodic_longitudinal_spin_response(
        &cell,
        &functional,
        &reference,
        &options,
        &static_reciprocal,
        Complex::ONE,
        2.
    )
    .unwrap_err()
    .to_string()
    .contains("common-total-number"));
    println!("thermal spin: equilibrium chi={:.12e}, uniform collisionless={:.12e}, finite-q moment={:.12e}, occupation defect={:.12e}",
        equilibrium.value.susceptibility,collisionless.value.magnetic_moment.abs(),
        warm.value.magnetic_moment.abs(),warm.value.density_response.occupation_consistency_defect);
}

fn finite_field(
    cell: &Cell,
    f: &ResolvedFunctional,
    r: &PeriodicScfResult,
    o: &PeriodicScfOptions,
    source: &[Vec<BlochMatrix>; 2],
    field: f64,
) -> [Vec<BlochMatrix>; 2] {
    let mut density = r.density.clone();
    for iteration in 0..500 {
        let mut fock =
            fock_at_density(cell, f, &r.k_points, &density, o, &r.k_points, false).unwrap();
        let mut next: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::new());
        let mut defect = 0_f64;
        for s in 0..2 {
            fock[s][0].add_scaled(&source[s][0], field);
            let orbitals =
                generalized_orbitals(&fock[s][0], &r.overlap[0], o.orth_threshold).unwrap();
            let occupied = if s == 0 { r.n_alpha } else { r.n_beta };
            let populations = (0..4)
                .map(|i| if i < occupied { 1. } else { 0. })
                .collect::<Vec<_>>();
            let p = band_density(&orbitals, &populations).unwrap();
            defect = defect.max(p.rms_diff(&density[s][0]));
            next[s].push(p);
        }
        if defect < 2e-14 {
            return next;
        }
        for s in 0..2 {
            density[s][0].scale(0.5);
            density[s][0].add_scaled(&next[s][0], 0.5);
        }
        assert!(
            iteration < 499,
            "independent longitudinal field SCF failed: {field:e} {defect:e}"
        );
    }
    unreachable!()
}
fn components_of(density: &[Vec<BlochMatrix>; 2]) -> Vec<f64> {
    density
        .iter()
        .flat_map(|s| (0..4).flat_map(move |i| (i..4).map(move |j| s[0].re[(i, j)])))
        .collect()
}

#[test]
fn all_independent_c1_spin_density_components_have_four_step_field_fits() {
    let (cell, options) = setup();
    let mut failures = Vec::new();
    let q = cell.lattice().reciprocal()[0];
    for name in ["pbe", "tpss", "pbe0"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged, "{name}");
        for (phase, label) in [(Complex::ONE, "cosine"), (Complex::new(0., 1.), "sine")] {
            let request = PeriodicResponseOptions {
                q,
                reciprocal_cutoff: 12.,
                solver: dft_rs::response::ResponseSolverConfig {
                    tolerance: 1e-12,
                    ..Default::default()
                },
                ..Default::default()
            };
            let plus = periodic_longitudinal_spin_response(
                &cell,
                &functional,
                &reference,
                &options,
                &request,
                phase,
                2.,
            )
            .unwrap();
            let mut minus_request = request.clone();
            minus_request.q = -q;
            let minus = periodic_longitudinal_spin_response(
                &cell,
                &functional,
                &reference,
                &options,
                &minus_request,
                phase.conj(),
                2.,
            )
            .unwrap();
            let mut predicted = plus.value.density_response.density.clone();
            for s in 0..2 {
                predicted[s][0].add_scaled(&minus.value.density_response.density[s][0], 1.);
            }
            assert!(predicted.iter().flatten().all(|p| p
                .im
                .as_slice()
                .iter()
                .all(|v| v.abs() < 1e-11)));
            let exact = components_of(&predicted);
            let mut source = spin_source(&gaussian_vertex(q), phase, 2.);
            let other = spin_source(&gaussian_vertex(-q), phase.conj(), 2.);
            for s in 0..2 {
                source[s][0].add_scaled(&other[s][0], 1.);
            }
            let names = (0..exact.len())
                .map(|i| format!("{name} longitudinal {label} spin-density component {i}"))
                .collect::<Vec<_>>();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.; 4]; exact.len()];
                    for (index, &h) in steps.iter().enumerate() {
                        let at = |v| {
                            components_of(&finite_field(
                                &cell,
                                &functional,
                                &reference,
                                &options,
                                &source,
                                v,
                            ))
                        };
                        let p = at(h);
                        let m = at(-h);
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (at(2. * h), at(-2. * h))
                        } else {
                            (vec![0.; exact.len()], vec![0.; exact.len()])
                        };
                        for i in 0..exact.len() {
                            errors[i][index] =
                                (stencil.first(p[i], m[i], p2[i], m2[i], h) - exact[i]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("longitudinal periodic spin response", &failures);
}

#[test]
fn tpss_spin_density_has_resolved_asymptotic_four_step_fits() {
    let (cell, options) = setup();
    let functional = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    assert!(reference.converged);
    let q = cell.lattice().reciprocal()[0];
    let steps = [0.01, 0.005, 0.0025, 0.00125];
    for (phase, label) in [(Complex::ONE, "cosine"), (Complex::new(0., 1.), "sine")] {
        let request = PeriodicResponseOptions {
            q,
            reciprocal_cutoff: 12.,
            solver: dft_rs::response::ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            },
            ..Default::default()
        };
        let plus = periodic_longitudinal_spin_response(
            &cell,
            &functional,
            &reference,
            &options,
            &request,
            phase,
            2.,
        )
        .unwrap();
        let mut minus_request = request.clone();
        minus_request.q = -q;
        let minus = periodic_longitudinal_spin_response(
            &cell,
            &functional,
            &reference,
            &options,
            &minus_request,
            phase.conj(),
            2.,
        )
        .unwrap();
        let mut predicted = plus.value.density_response.density.clone();
        for s in 0..2 {
            predicted[s][0].add_scaled(&minus.value.density_response.density[s][0], 1.);
        }
        let exact = components_of(&predicted);
        let mut source = spin_source(&gaussian_vertex(q), phase, 2.);
        let other = spin_source(&gaussian_vertex(-q), phase.conj(), 2.);
        for s in 0..2 {
            source[s][0].add_scaled(&other[s][0], 1.);
        }
        let mut errors = vec![[0.; 4]; exact.len()];
        for (index, &h) in steps.iter().enumerate() {
            let p = components_of(&finite_field(
                &cell,
                &functional,
                &reference,
                &options,
                &source,
                h,
            ));
            let m = components_of(&finite_field(
                &cell,
                &functional,
                &reference,
                &options,
                &source,
                -h,
            ));
            for i in 0..exact.len() {
                errors[i][index] = ((p[i] - m[i]) / (2. * h) - exact[i]).abs();
            }
        }
        for i in 0..exact.len() {
            let name = format!("TPSS strict fine {label} independent spin-density component {i}");
            assert!(components::component_fit_of_order(
                &name, &steps, &errors[i], 2.
            ));
            let fit = convergence::error_power_fit(&steps, &errors[i]);
            assert!(
                (fit.exponent - 2.).abs() < 0.02 && fit.r_squared > 0.99999,
                "{name}: p={} R2={} errors={:?}",
                fit.exponent,
                fit.r_squared,
                errors[i]
            );
        }
    }
}
