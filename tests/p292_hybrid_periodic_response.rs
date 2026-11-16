// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Hybrid finite-q screening and independent nonlinear Gamma-field fits.
//! Every real symmetric AO density entry in both spin channels is fitted
//! for six independent charge/spin sources on a C1 triclinic four-centre cell.
//! Finite-field SCF uses only the nonlinear reference Fock, not response jets.
//! Bounded post-repair acceptance: 360 four-step density-output fits,
//! p=1.995728514..2.006077233, minimum R2=0.999997493811 on
//! h=[0.04,0.02,0.01,0.005] Hartree source coordinates; no failed interval.
//! Eighteen full-complex mesh/frequency dense cases and 32 restricted/
//! unrestricted mesh/temperature physical-reference identities also pass.
//! This does not certify the remaining electronic spaces or model terms.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::dfpt::coulomb::FiniteQHartree;
use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
use dft_rs::dfpt::orbitals::{adjoint, band_density, generalized_orbitals, matmul};
use dft_rs::grid::GridOptions;
use dft_rs::linalg::{solve_linear, symmetric_eigen, Matrix};
use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use dft_rs::periodic::scf::{
    fock_at_density, reference_fock_at, run_periodic_scf, PeriodicScfOptions, PeriodicScfResult,
};
use dft_rs::periodic::{BlochMatrix, Cell, Lattice};
use dft_rs::response::complex::Complex;
use dft_rs::response::krylov::LinearOperator;
use dft_rs::response::Frequency;
use dft_rs::xc::ResolvedFunctional;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, Vec3};

fn setup() -> (Cell, PeriodicScfOptions) {
    setup_for_multiplicity(1)
}

fn setup_for_multiplicity(multiplicity: usize) -> (Cell, PeriodicScfOptions) {
    setup_positions_with_multiplicity(
        [
            [0.1, -0.2, 0.3],
            [1.4, 0.31, -0.17],
            [-0.23, 1.7, 0.21],
            [0.38, -0.13, 1.89],
        ],
        multiplicity,
    )
}

fn stable_setup() -> (Cell, PeriodicScfOptions) {
    setup_positions([
        [0.1, -0.2, 0.3],
        [1.35, 0.1, 0.2],
        [-0.3, 3.2, 0.9],
        [0.2, 3.4, 2.2],
    ])
}

fn setup_positions(positions: [[f64; 3]; 4]) -> (Cell, PeriodicScfOptions) {
    setup_positions_with_multiplicity(positions, 1)
}

fn setup_positions_with_multiplicity(
    positions: [[f64; 3]; 4],
    multiplicity: usize,
) -> (Cell, PeriodicScfOptions) {
    c1_geometry::assert_c1(&[1; 4], &positions);
    let atoms = positions
        .into_iter()
        .map(|p| Atom {
            z: 1,
            position: Vec3::new(p[0], p[1], p[2]),
        })
        .collect();
    let mut basis = BasisSet::new("compact periodic response validation model");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.91],
            coefs: vec![1.0],
        }],
    });
    let cell = Cell::new(
        Molecule::new(atoms).with_multiplicity(multiplicity),
        Lattice::new(
            Vec3::new(8.7, 0.31, 0.13),
            Vec3::new(0.2, 9.2, 0.43),
            Vec3::new(0.11, 0.27, 9.7),
        ),
        &basis,
        4.0,
    )
    .unwrap();
    let options = PeriodicScfOptions {
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        exchange_cutoff: Some(4.0),
        grid_shell: 1,
        grid: GridOptions {
            radial_override: Some(10),
            angular_override: Some(7),
            prune: false,
            weight_cutoff: 0.0,
            ..Default::default()
        },
        ..Default::default()
    };
    (cell, options)
}

fn lowest_triplet_curvature(solver: &PeriodicDensitySolver, reference: &PeriodicScfResult) -> f64 {
    assert_eq!(reference.k_points.len(), 1);
    assert_eq!(reference.n_alpha, reference.n_beta);
    let energies = &reference.mo_energy[0][0];
    let n = energies.len();
    let occupied = reference.n_alpha;
    let pairs = (0..occupied)
        .flat_map(|i| (occupied..n).map(move |a| (i, a)))
        .collect::<Vec<_>>();
    let mut hessian = Matrix::zeros(pairs.len(), pairs.len());
    for (column, &(j, b)) in pairs.iter().enumerate() {
        let mut direction = vec![Complex::ZERO; solver.dimension()];
        for (spin, sign) in [(0, 1.0), (1, -1.0)] {
            direction[spin * n * n + b * n + j] = Complex::from(sign);
            direction[spin * n * n + j * n + b] = Complex::from(sign);
        }
        let potential = solver.induced_potential(&direction).unwrap();
        for (row, &(i, a)) in pairs.iter().enumerate() {
            hessian[(row, column)] = potential[a * n + i].re
                + if row == column {
                    energies[a] - energies[i]
                } else {
                    0.0
                };
        }
    }
    assert!(hessian.max_abs_diff(&hessian.transpose()) < 2e-11);
    symmetric_eigen(&hessian).unwrap().0[0]
}

#[test]
fn original_compact_reference_retains_its_negative_triplet_curvature_regression() {
    let (cell, options) = setup();
    let functional = ResolvedFunctional::hartree_fock();
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    let request = PeriodicResponseOptions {
        reciprocal_cutoff: 12.0,
        ..Default::default()
    };
    let solver =
        PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &request).unwrap();
    let lowest = lowest_triplet_curvature(&solver, &reference);
    println!("Original compact C1 RHF reference: lowest triplet orbital curvature={lowest:.12e} Hartree; unconstrained finite fields follow a different spin-broken branch");
    assert!(lowest < -1e-3);
}

#[test]
fn returned_reference_is_the_physical_field_and_free_energy_of_its_density() {
    for multiplicity in [1, 3] {
        let (cell, base_options) = setup_for_multiplicity(multiplicity);
        for mesh in [[1, 1, 1], [2, 1, 1]] {
            for temperature in [0.0, 40000.0] {
                let options = PeriodicScfOptions {
                    k_mesh: mesh,
                    e_tol: 1e-7,
                    p_tol: 1e-5,
                    conv_tol_grad: 1e-5,
                    occupation: if temperature == 0.0 {
                        OccupationScheme::Insulator
                    } else {
                        OccupationScheme::FermiDirac {
                            temperature_kelvin: temperature,
                        }
                    },
                    ..base_options.clone()
                };
                for name in ["hf", "pbe", "pbe0", "cam-b3lyp"] {
                    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
                    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
                    assert_eq!(reference.unrestricted, multiplicity != 1);
                    assert_eq!(reference.n_alpha - reference.n_beta, multiplicity - 1);
                    let rebuilt = reference_fock_at(
                        &cell,
                        &functional,
                        &reference,
                        &options,
                        &reference.k_points,
                    )
                    .unwrap();
                    let mut defect = 0.0_f64;
                    for spin in 0..2 {
                        for k in 0..reference.k_points.len() {
                            defect = defect.max(
                                rebuilt[spin][k]
                                    .re
                                    .max_abs_diff(&reference.fock[spin][k].re),
                            );
                            defect = defect.max(
                                rebuilt[spin][k]
                                    .im
                                    .max_abs_diff(&reference.fock[spin][k].im),
                            );
                        }
                    }
                    println!("{name} multiplicity={multiplicity} mesh={mesh:?} T={temperature}: all-entry returned F[P] defect={defect:.12e}");
                    assert!(defect < 3e-14,"the returned Fock must belong to the returned density, even at loose convergence gates");
                    if name == "hf" {
                        let mut energy = reference.e_nuc;
                        for spin in 0..2 {
                            for k in 0..reference.k_points.len() {
                                let mut h_plus_f = reference.h_core[k].clone();
                                h_plus_f.add_scaled(&rebuilt[spin][k], 1.0);
                                energy += 0.5 * reference.density[spin][k].trace_product(&h_plus_f)
                                    / reference.k_points.len() as f64;
                            }
                        }
                        let error = (energy - reference.energy).abs();
                        println!("HF physical density energy identity defect={error:.12e} Hartree");
                        assert!(error < 3e-13);
                    }
                    if temperature > 0.0 {
                        let mut entropy_term = 0.0;
                        for spin in 0..2 {
                            for k in 0..reference.k_points.len() {
                                let orbitals = generalized_orbitals(
                                    &reference.fock[spin][k],
                                    &reference.overlap[k],
                                    options.orth_threshold,
                                )
                                .unwrap();
                                let sc =
                                    matmul(&reference.overlap[k], &orbitals.coefficients).unwrap();
                                let population = matmul(
                                    &adjoint(&sc).unwrap(),
                                    &matmul(&reference.density[spin][k], &sc).unwrap(),
                                )
                                .unwrap();
                                let mut identity =
                                    BlochMatrix::zeros(population.rows(), population.rows());
                                for i in 0..population.rows() {
                                    identity.re[(i, i)] = 1.0;
                                }
                                let eigenvalues =
                                    generalized_orbitals(&population, &identity, 1e-12)
                                        .unwrap()
                                        .energies;
                                for f in eigenvalues {
                                    assert!(f > 0.0 && f < 1.0);
                                    entropy_term -= temperature
                                        * BOLTZMANN_HARTREE_PER_K
                                        * (f * f.ln() + (1.0 - f) * (-f).ln_1p())
                                        / reference.k_points.len() as f64;
                                }
                            }
                        }
                        let defect = (entropy_term - reference.entropy_term).abs();
                        let free_defect =
                            (reference.energy - entropy_term - reference.free_energy).abs();
                        println!("{name}: density-spectrum entropy defect={defect:.12e}, Mermin free energy defect={free_defect:.12e} Hartree");
                        assert!(defect < 3e-13 && free_defect < 3e-13);
                    }
                }
            }
        }
    }
}

fn finite_field(
    cell: &Cell,
    functional: &ResolvedFunctional,
    reference: &PeriodicScfResult,
    options: &PeriodicScfOptions,
    source: &[Vec<BlochMatrix>; 2],
    field: f64,
) -> [Vec<BlochMatrix>; 2] {
    let mut density = reference.density.clone();
    for iteration in 0..500 {
        let mut fock = fock_at_density(
            cell,
            functional,
            &reference.k_points,
            &density,
            options,
            &reference.k_points,
            false,
        )
        .unwrap();
        let mut next: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| Vec::new());
        let mut defect = 0.0_f64;
        for spin in 0..2 {
            for k in 0..reference.k_points.len() {
                fock[spin][k].add_scaled(&source[spin][k], field);
                let orbitals = generalized_orbitals(
                    &fock[spin][k],
                    &reference.overlap[k],
                    options.orth_threshold,
                )
                .unwrap();
                let occupied = if spin == 0 {
                    reference.n_alpha
                } else {
                    reference.n_beta
                };
                let population = (0..orbitals.energies.len())
                    .map(|i| if i < occupied { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>();
                let p = band_density(&orbitals, &population).unwrap();
                defect = defect.max(p.rms_diff(&density[spin][k]));
                next[spin].push(p);
            }
        }
        if defect < 2e-14 {
            return next;
        }
        for spin in 0..2 {
            for k in 0..reference.k_points.len() {
                density[spin][k].scale(0.5);
                density[spin][k].add_scaled(&next[spin][k], 0.5);
            }
        }
        assert!(
            iteration < 499,
            "independent field SCF failed: field={field:e} residual={defect:e}"
        );
    }
    unreachable!()
}

fn descriptors(density: &[Vec<BlochMatrix>; 2]) -> Vec<f64> {
    density
        .iter()
        .flat_map(|s| {
            s.iter().flat_map(|p| {
                (0..p.rows()).flat_map(move |i| (i..p.cols()).map(move |j| p.re[(i, j)]))
            })
        })
        .collect()
}

#[test]
fn all_gamma_hartree_density_kernel_entries_converge_to_reference_fock() {
    let (cell, options) = setup();
    let n = cell.nao();
    let points = [Vec3::zero()];
    let mut hartree = ResolvedFunctional::hartree_fock();
    hartree.hybrid.cx = 0.0;
    let zero: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| vec![BlochMatrix::zeros(n, n)]);
    let base = fock_at_density(&cell, &hartree, &points, &zero, &options, &points, false).unwrap();
    for cutoff in [6.0, 8.0, 10.0, 12.0] {
        let map = FiniteQHartree::new(
            &cell,
            &points,
            Vec3::zero(),
            cutoff,
            cell.hartree_real_space_cutoff(),
        )
        .unwrap();
        let mut defect = 0.0_f64;
        for a in 0..n {
            for b in a..n {
                let mut direction = zero.clone();
                direction[0][0].re[(a, b)] = 1.0;
                direction[0][0].re[(b, a)] = 1.0;
                let exact = fock_at_density(
                    &cell, &hartree, &points, &direction, &options, &points, false,
                )
                .unwrap();
                let image = map.apply(&direction).unwrap();
                for i in 0..n {
                    for j in 0..n {
                        defect = defect.max(
                            (exact[0][0].re[(i, j)]
                                - base[0][0].re[(i, j)]
                                - image[0][0].re[(i, j)])
                                .abs(),
                        );
                        defect = defect.max(image[0][0].im[(i, j)].abs());
                    }
                }
            }
        }
        println!("Gamma Hartree every independent density source/all AO outputs cutoff={cutoff} max_reference_defect={defect:.12e}");
        if cutoff == 12.0 {
            assert!(
                defect < 2e-11,
                "Hartree reciprocal convergence defect={defect:e}"
            );
        }
    }
}

#[test]
fn hybrid_gamma_density_has_complete_independent_nonlinear_field_fits() {
    let (cell, options) = stable_setup();
    let n = cell.nao();
    let mut failures = Vec::new();
    for name in ["hf", "pbe0", "cam-b3lyp"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(reference.converged, "{name} reference did not converge");
        let request = PeriodicResponseOptions {
            reciprocal_cutoff: 12.0,
            solver: dft_rs::response::ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            },
            ..Default::default()
        };
        let solver =
            PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &request).unwrap();
        let curvature = lowest_triplet_curvature(&solver, &reference);
        println!("{name}: finite-field fixture lowest triplet curvature={curvature:.12e} Hartree");
        assert!(
            curvature > 1e-3,
            "the independent field oracle requires a spin-stable reference"
        );
        for channel in 0..6 {
            let mut source: [Vec<BlochMatrix>; 2] =
                std::array::from_fn(|_| vec![BlochMatrix::zeros(n, n)]);
            for spin in 0..2 {
                for i in 0..n {
                    for j in i..n {
                        let sign = if channel >= 3 && spin == 1 { -1.0 } else { 1.0 };
                        let value = sign
                            * 0.17
                            * (((channel % 3 + 1) * (i + j + 2) + (i + 1) * (j + 1)) as f64 + 0.31)
                                .sin();
                        source[spin][0].re[(i, j)] = value;
                        source[spin][0].re[(j, i)] = value;
                    }
                }
            }
            let report = solver.solve(&source).unwrap();
            let exact = descriptors(&report.value.density);
            let names = (0..exact.len())
                .map(|i| format!("{name} periodic Gamma source={channel} independent density={i}"))
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
                    let mut errors = vec![[0.0; 4]; exact.len()];
                    for (step, &h) in steps.iter().enumerate() {
                        let at = |t| {
                            descriptors(&finite_field(
                                &cell,
                                &functional,
                                &reference,
                                &options,
                                &source,
                                t,
                            ))
                        };
                        let plus = at(h);
                        let minus = at(-h);
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            (vec![0.0; exact.len()], vec![0.0; exact.len()])
                        };
                        for i in 0..exact.len() {
                            errors[i][step] = (stencil
                                .first(plus[i], minus[i], plus2[i], minus2[i], h)
                                - exact[i])
                                .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
            println!(
                "{name} Gamma source={channel} raw_response={:?} reference={:?}",
                report.residuals, report.reference
            );
        }
    }
    components::assert_no_failures("hybrid periodic physical density", &failures);
}

#[test]
fn finite_q_hybrid_complex_screening_matches_all_dense_response_entries() {
    let (cell, options) = setup();
    for mesh in [[1, 1, 1], [2, 1, 1]] {
        let mut options = options.clone();
        options.k_mesh = mesh;
        for name in ["hf", "pbe0", "cam-b3lyp"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
            assert!(reference.converged);
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
                    reciprocal_cutoff: 1.0,
                    ..Default::default()
                };
                let solver =
                    PeriodicDensitySolver::new(&cell, &functional, &reference, &options, &request)
                        .unwrap();
                let source = solver.plane_wave_source(0).unwrap();
                let report = solver.solve(&source).unwrap();
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
                let vector = rhs
                    .iter()
                    .map(|v| v.re)
                    .chain(rhs.iter().map(|v| v.im))
                    .collect::<Vec<_>>();
                let solved = solve_linear(&dense, &vector).unwrap();
                let exact = solver
                    .density_to_aos(
                        &(0..dimension)
                            .map(|i| Complex::new(solved[i], solved[dimension + i]))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let defect = exact
                    .iter()
                    .flatten()
                    .zip(report.value.density.iter().flatten())
                    .map(|(a, b)| a.rms_diff(b))
                    .fold(0.0_f64, f64::max);
                println!("{name} mesh={mesh:?} {frequency:?}: full complex density dense defect={defect:.12e}, raw_response={:?}",report.residuals);
                assert!(defect < 1e-10);
                assert!(report.residuals[0] <= 1e-10);
            }
        }
    }
}
