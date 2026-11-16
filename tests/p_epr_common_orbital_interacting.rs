// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All nine interacting common-orbital SOC entries. Finite fields are
//! independent validation of the analytic production response, never a fallback.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
pub mod components;
#[path = "support/convergence.rs"]
pub mod convergence;
#[path = "support/complex_common_hf.rs"]
mod oracle;

use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;

#[test]
fn all_nine_triplet_rohf_entries_fit_independent_interacting_complex_scf() {
    dft_rs::resources::ResourceLimits::new(Some(1), Some(1280))
        .unwrap()
        .install(|| {
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
            let parameters = Value::object()
                .with("xc", "hf")
                .with("basis", "sto-3g")
                .with("multiplicity", 3usize)
                .with("guess", "core")
                .with("dispersion", "none")
                .with("grid_weight_cutoff", 0.0)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-12)
                .with("conv_tol_dp", 1e-13)
                .with("max_iter", 500usize);
            let method =
                MolecularMethod::from_controls(&Controls::from_value(&parameters).unwrap())
                    .unwrap();
            let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
            let (reference, common) = dft_rs::scf::roks::run_roks_with_orbitals(
                &setup.molecule,
                &setup.basis,
                &setup.options,
                &setup.functional,
                setup.grid.as_ref(),
            )
            .unwrap();
            assert!(reference.converged);
            assert_eq!([reference.n_alpha, reference.n_beta], [3, 1]);
            let origin = Vec3::new(0.19, -0.11, 0.23);
            let mut request = parameters.clone();
            request.set("reference", "rohf");
            request.set("gauge", "common-origin");
            request.set("gauge_origin", origin.to_array());
            request.set("response_tolerance", 1e-12);
            let report = dft_rs::driver::run(
                "g-tensor",
                &system,
                &Controls::from_value(&request).unwrap(),
            )
            .unwrap();
            let target = report
                .get("spin_orbit_ppm")
                .unwrap()
                .numeric_array()
                .unwrap()
                .1;
            let angular = dft_rs::properties::nmr::angular_momentum_operator(&setup.basis, origin);
            let n = setup.basis.nao_sph;
            let fields = angular.map(|mut a| {
                for v in a.as_mut_slice() {
                    *v *= -0.5;
                }
                a
            });
            let mut soc = std::array::from_fn::<_, 3, _>(|_| Matrix::zeros(n, n));
            for atom in &setup.molecule.atoms {
                let local =
                    dft_rs::properties::nmr::paramagnetic_operator(&setup.basis, atom.position);
                for a in 0..3 {
                    for (v, l) in soc[a].as_mut_slice().iter_mut().zip(local[a].as_slice()) {
                        *v += atom.z as f64 * l;
                    }
                }
            }
            let mut core = dft_rs::integrals::kinetic_matrix(&setup.basis);
            let nuclear = dft_rs::integrals::nuclear_matrix(&setup.basis, &setup.molecule, None);
            for (v, n) in core.as_mut_slice().iter_mut().zip(nuclear.as_slice()) {
                *v += n;
            }
            let eri = dft_rs::integrals::eri_incore(
                &setup.basis,
                dft_rs::integrals::CoulombKernel::Coulomb,
                1280,
            )
            .unwrap();
            let model = oracle::Model::new(
                reference.overlap.clone(),
                core,
                eri,
                reference.e_nuc,
                [3, 1],
            );
            let zero = model.stationary(&common, &fields[0], 0.0);
            assert!((zero.energy - reference.energy).abs() < 1e-10);
            println!(
                "Interacting common-HF oracle E0={} productionE={} residual={}",
                zero.energy, reference.energy, zero.residual
            );
            let names = (0..9)
                .map(|p| format!("C1 triplet ROHF interacting g-SOC ({},{})", p / 3, p % 3))
                .collect::<Vec<_>>();
            let mut failures = Vec::new();
            let factor = dft_rs::constants::FINE_STRUCTURE.powi(2) / 2.0;
            components::fit_components(
                &names,
                &[
                    [0.08, 0.04, 0.02, 0.01],
                    [0.04, 0.02, 0.01, 0.005],
                    [0.016, 0.008, 0.004, 0.002],
                ],
                |steps| {
                    let mut errors = vec![[0.0; 4]; 9];
                    for field in 0..3 {
                        for (s, &h) in steps.iter().enumerate() {
                            let plus = model.stationary(&common, &fields[field], h);
                            let minus = model.stationary(&common, &fields[field], -h);
                            for spin in 0..3 {
                                let expectation = |state: &oracle::State| {
                                    state.density[0].im.frobenius_dot(&soc[spin])
                                        - state.density[1].im.frobenius_dot(&soc[spin])
                                };
                                let numerical = -factor
                                    * (expectation(&plus) - expectation(&minus))
                                    / (2.0 * h);
                                errors[3 * field + spin][s] =
                                    (numerical - 1e-6 * target[3 * field + spin]).abs();
                            }
                        }
                    }
                    errors
                },
                &mut failures,
            );
            components::assert_no_failures("all9 interacting common-orbital g entries", &failures);
        })
        .unwrap();
}
