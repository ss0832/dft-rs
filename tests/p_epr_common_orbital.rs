// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent mixed-energy fits for the constrained common-orbital EPR
//! provider. Finite fields are an oracle, never the production calculation.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::epr::common_orbital_g_tensor;

fn system() -> System {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    );
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    system
}

fn setup(xc: &str) -> dft_rs::driver::method::MolecularSetup {
    let method = MolecularMethod::from_controls(
        &Controls::from_value(
            &Value::object()
                .with("xc", xc)
                .with("basis", "sto-3g")
                .with("charge", 3i32)
                .with("multiplicity", 2usize)
                .with("guess", "core")
                .with("dispersion", "none")
                .with("grid_weight_cutoff", 0.0)
                .with("grid_level", 2usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-13)
                .with("max_iter", 250usize),
        )
        .unwrap(),
    )
    .unwrap();
    let system = system();
    prepare_molecular(&system.numbers, &system.positions, &method).unwrap()
}

#[test]
fn rohf_all_nine_soc_entries_fit_independent_complex_energy_eigenvalues() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let setup = setup("hf");
            let (reference, orbitals) = dft_rs::scf::roks::run_roks_with_orbitals(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
            )
            .unwrap();
            assert!(reference.converged);
            let origin = Vec3::new(0.19, -0.11, 0.23);
            let config = dft_rs::response::ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            };
            let result = common_orbital_g_tensor(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference,
                &orbitals,
                &setup.options,
                origin,
                &[1.; 4],
                &config,
            )
            .unwrap();
            assert!(result.reference_stationarity < config.acceptance());
            assert!(result.metric_defect < config.acceptance());
            assert!(result
                .response_residuals
                .iter()
                .all(|v| *v < config.acceptance()));
            assert!(result.tensor.iter().flatten().all(|v| v.is_finite()));
            println!(
                "ROHF common-origin tensor={:?}, RMC={}, GC={:?}",
                result.tensor, result.mass_correction, result.gauge_correction
            );
            let angular = dft_rs::properties::nmr::angular_momentum_operator(&setup.basis, origin);
            let mut soc = std::array::from_fn::<_, 3, _>(|_| {
                Matrix::zeros(setup.basis.nao_sph, setup.basis.nao_sph)
            });
            for atom in &setup.molecule.atoms {
                let local =
                    dft_rs::properties::nmr::paramagnetic_operator(&setup.basis, atom.position);
                for a in 0..3 {
                    for (out, value) in soc[a].as_mut_slice().iter_mut().zip(local[a].as_slice()) {
                        *out += value;
                    }
                }
            }
            // One electron: the HF total energy is exactly the one-electron
            // eigenvalue. This oracle uses neither the response kernel nor
            // spin semicanonical virtual energies.
            let overlap = BlochMatrix {
                re: reference.overlap.clone(),
                im: Matrix::zeros(reference.overlap.rows, reference.overlap.cols),
            };
            let mut hcore = dft_rs::integrals::kinetic_matrix(&setup.basis);
            let attraction = dft_rs::integrals::nuclear_matrix(&setup.basis, &setup.molecule, None);
            for (value, nuclear) in hcore.as_mut_slice().iter_mut().zip(attraction.as_slice()) {
                *value += nuclear;
            }
            let energy = |field: usize, spin: usize, b: f64, lambda: f64| {
                let mut imaginary = Matrix::zeros(hcore.rows, hcore.cols);
                for i in 0..hcore.rows {
                    for j in 0..hcore.cols {
                        imaginary[(i, j)] =
                            -0.5 * b * angular[field][(i, j)] + lambda * soc[spin][(i, j)];
                    }
                }
                let eigen = dft_rs::dfpt::orbitals::generalized_orbitals(
                    &BlochMatrix {
                        re: hcore.clone(),
                        im: imaginary,
                    },
                    &overlap,
                    setup.options.orth_threshold,
                )
                .unwrap();
                eigen.energies[0]
            };
            let names = (0..9)
                .map(|p| format!("ROHF C1 g-SOC mixed energy ({},{})", p / 3, p % 3))
                .collect::<Vec<_>>();
            let mut failures = Vec::new();
            components::fit_components(
                &names,
                &[[0.08, 0.04, 0.02, 0.01], [0.16, 0.08, 0.04, 0.02]],
                |steps| {
                    let mut errors = vec![[0.; 4]; 9];
                    for field in 0..3 {
                        for spin in 0..3 {
                            for (index, &h) in steps.iter().enumerate() {
                                let mixed = (energy(field, spin, h, h)
                                    - energy(field, spin, h, -h)
                                    - energy(field, spin, -h, h)
                                    + energy(field, spin, -h, -h))
                                    / (4. * h * h);
                                let expected = -dft_rs::constants::FINE_STRUCTURE.powi(2) * mixed;
                                errors[3 * field + spin][index] =
                                    (expected - result.spin_orbit[field][spin]).abs();
                            }
                        }
                    }
                    errors
                },
                &mut failures,
            );
            components::assert_no_failures(
                "all nine common-orbital g SOC energy entries",
                &failures,
            );
        })
        .unwrap();
}

#[test]
fn one_electron_roks_matches_unrestricted_common_origin_all_entries() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
            let setup = setup("pbe");
            let (reference, orbitals) = dft_rs::scf::roks::run_roks_with_orbitals(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
            )
            .unwrap();
            let origin = Vec3::new(0.19, -0.11, 0.23);
            let result = common_orbital_g_tensor(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference,
                &orbitals,
                &setup.options,
                origin,
                &[1.; 4],
                &Default::default(),
            )
            .unwrap();
            let report = dft_rs::driver::run(
                "g-tensor",
                &system(),
                &Controls::from_value(
                    &Value::object()
                        .with("xc", "pbe")
                        .with("basis", "sto-3g")
                        .with("charge", 3i32)
                        .with("multiplicity", 2usize)
                        .with("unrestricted", true)
                        .with("guess", "core")
                        .with("gauge", "common-origin")
                        .with("gauge_origin", origin.to_array())
                        .with("grid_level", 2usize)
                        .with("grid_weight_cutoff", 0.)
                        .with("dispersion", "none")
                        .with("conv_tol", 1e-13)
                        .with("conv_tol_grad", 1e-12)
                        .with("conv_tol_dp", 1e-13),
                )
                .unwrap(),
            )
            .unwrap();
            let tensor = report.get("g_tensor").unwrap().numeric_array().unwrap().1;
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        (result.tensor[a][b] - tensor[3 * a + b]).abs() < 1e-9,
                        "one-electron constrained/unrestricted mismatch ({a},{b})"
                    );
                }
            }
            println!("ROKS/PBE common-origin tensor={:?}", result.tensor);
        })
        .unwrap();
}
