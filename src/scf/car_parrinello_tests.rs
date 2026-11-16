// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use super::{initial_coordinates, Model};
use crate::acceptance::{components, stencils};
use crate::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::BlochMatrix;

#[test]
fn moving_lowdin_ammonia_off_shell_derivatives_fit_every_component() {
    fit_ammonia_off_shell_derivatives("wb97x-d");
}

#[test]
fn moving_lowdin_ammonia_vv10_off_shell_derivatives_fit_every_component() {
    fit_ammonia_off_shell_derivatives("wb97x-v");
}

fn fit_ammonia_off_shell_derivatives(xc: &str) {
    let setup = prepare_molecular(
        &[7, 1, 1, 1],
        &[
            [0.13, -0.27, 0.19],
            [1.78, 0.34, -0.08],
            [-0.49, 1.65, 0.41],
            [0.31, -0.78, 1.91],
        ],
        &MolecularMethod {
            xc: Some(xc.into()),
            grid_level: 0,
            vv10_grid_level: 0,
            grid_weight_cutoff: 0.0,
            ..MolecularMethod::default()
        },
    )
    .unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    let scf = &reference.scf;
    let spins = [scf.n_alpha, scf.n_beta];
    let functional = setup.method.dispersion_functional_name();
    let model = Model {
        basis_set: &setup.basis_set,
        functional: &setup.functional,
        grid_options: &setup.grid_options,
        options: &setup.options,
        dispersion: &setup.dispersion,
        dispersion_functional: &functional,
        spins,
        common: true,
    };
    let initial = initial_coordinates(
        &setup.basis,
        setup.options.orth_threshold,
        [&scf.mo_coeff[0], &scf.mo_coeff[1]],
        spins,
        true,
    )
    .unwrap();
    let initial_energy = model.evaluate(&setup.molecule, &initial).unwrap().energy;
    assert!((initial_energy - reference.total_energy()).abs() < 2e-10);
    // Perturb every coefficient deterministically so the chart derivatives
    // are sampled away from the initial stationary determinant.
    let mut orbitals = initial;
    for (i, x) in orbitals[0].as_mut_slice().iter_mut().enumerate() {
        *x += 0.013 * (0.53 * (i + 1) as f64).sin();
    }
    let point = model.evaluate(&setup.molecule, &orbitals).unwrap();
    println!(
        "CP ammonia {xc} off-shell potential={:.12e} minimum overlap={:.12e} initial SCF parity={:.12e}",
        point.energy,
        point.minimum_overlap,
        (initial_energy - reference.total_energy()).abs(),
    );
    let cartesian = 3 * setup.molecule.len();
    let coefficients = orbitals[0].rows * orbitals[0].cols;
    let expected: Vec<_> = point
        .nuclear_gradient
        .iter()
        .flat_map(|p| p.to_array())
        .chain(point.orbital_gradient[0].as_slice().iter().copied())
        .collect();
    let names: Vec<_> = (0..cartesian)
        .map(|i| format!("CP ammonia {xc} nuclear {i}"))
        .chain((0..coefficients).map(|i| format!("CP ammonia {xc} electronic {i}")))
        .collect();
    let energy = |component: usize, displacement: f64| {
        let mut geometry = setup.molecule.clone();
        let mut q: Vec<Matrix> = orbitals.clone();
        if component < cartesian {
            let mut xyz = [0.0; 3];
            xyz[component % 3] = displacement;
            geometry.atoms[component / 3].position += Vec3::new(xyz[0], xyz[1], xyz[2]);
        } else {
            q[0].as_mut_slice()[component - cartesian] += displacement;
        }
        model.evaluate(&geometry, &q).unwrap().energy
    };
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.009, 0.006, 0.004, 0.0027], stencils::Stencil::Second),
            ([0.004, 0.0028, 0.0019, 0.0013], stencils::Stencil::Second),
            ([0.015, 0.011, 0.008, 0.006], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            expected
                .iter()
                .enumerate()
                .map(|(component, target)| {
                    steps.map(|h| {
                        let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                            (energy(component, 2.0 * h), energy(component, -2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(
                            energy(component, h),
                            energy(component, -h),
                            plus2,
                            minus2,
                            h,
                        ) - target)
                            .abs()
                    })
                })
                .collect()
        },
        &mut failed,
    );
    components::assert_no_failures(&format!("Car-Parrinello off-shell ammonia {xc}"), &failed);
}

#[test]
fn complex_spinor_off_shell_derivatives_fit_every_component() {
    let setup = prepare_molecular(
        &[8, 1, 1],
        &[
            [0.12, -0.17, 0.08],
            [1.68, 0.61, -0.29],
            [-0.66, 1.37, 0.67],
        ],
        &MolecularMethod {
            xc: Some("wb97x-v".into()),
            grid_level: 0,
            vv10_grid_level: 0,
            grid_weight_cutoff: 0.0,
            ..MolecularMethod::default()
        },
    )
    .unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    let scf = &reference.scf;
    let spins = [scf.n_alpha, scf.n_beta];
    let dispersion = setup.method.dispersion_functional_name();
    let model = Model {
        basis_set: &setup.basis_set,
        functional: &setup.functional,
        grid_options: &setup.grid_options,
        options: &setup.options,
        dispersion: &setup.dispersion,
        dispersion_functional: &dispersion,
        spins,
        common: false,
    };
    let q = initial_coordinates(
        &setup.basis,
        setup.options.orth_threshold,
        [&scf.mo_coeff[0], &scf.mo_coeff[1]],
        spins,
        false,
    )
    .unwrap();
    let n = setup.basis.nao_sph;
    let mut orbitals = BlochMatrix::zeros(2 * n, spins.iter().sum());
    for i in 0..n {
        for j in 0..spins[0] {
            orbitals.re[(i, j)] = q[0][(i, j)];
        }
        for j in 0..spins[1] {
            orbitals.re[(n + i, spins[0] + j)] = q[1][(i, j)];
        }
    }
    // Perturb both components away from the stationary projector. Keep the
    // perturbation small enough for a smooth noncollinear angular quadrature.
    for (i, (r, v)) in orbitals
        .re
        .as_mut_slice()
        .iter_mut()
        .zip(orbitals.im.as_mut_slice())
        .enumerate()
    {
        *r += 0.002 * (0.67 * (i + 1) as f64).sin();
        *v += 0.003 * (0.41 * (i + 1) as f64).cos();
    }
    let (energy, nuclear, orbital, _) = model
        .evaluate_spinor(&setup.molecule, &orbitals, 1e-9)
        .unwrap();
    let expected: Vec<_> = nuclear
        .iter()
        .flat_map(|v| v.to_array())
        .chain(orbital.re.as_slice().iter().copied())
        .chain(orbital.im.as_slice().iter().copied())
        .collect();
    println!(
        "CPMD complex spinor wb97x-v off-shell energy={energy:.15e} components={}",
        expected.len()
    );
    let nuclear_count = 3 * setup.molecule.len();
    let size = orbitals.re.as_slice().len();
    let energy = |i: usize, step: f64| {
        let mut mol = setup.molecule.clone();
        let mut q = orbitals.clone();
        if i < nuclear_count {
            let mut shift = [0.0; 3];
            shift[i % 3] = step;
            mol.atoms[i / 3].position += Vec3::new(shift[0], shift[1], shift[2]);
        } else if i < nuclear_count + size {
            q.re.as_mut_slice()[i - nuclear_count] += step;
        } else {
            q.im.as_mut_slice()[i - nuclear_count - size] += step;
        }
        model.spinor_energy(&mol, &q, 1e-9).unwrap()
    };
    let names: Vec<_> = (0..expected.len())
        .map(|i| format!("CPMD complex spinor wb97x-v {i}"))
        .collect();
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.010, 0.007, 0.005, 0.003], stencils::Stencil::Second),
            ([0.006, 0.004, 0.0028, 0.0018], stencils::Stencil::Second),
            ([0.016, 0.012, 0.009, 0.006], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            expected
                .iter()
                .enumerate()
                .map(|(i, target)| {
                    steps.map(|h| {
                        let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                            (energy(i, 2.0 * h), energy(i, -2.0 * h))
                        } else {
                            (0.0, 0.0)
                        };
                        (stencil.first(energy(i, h), energy(i, -h), p2, m2, h) - target).abs()
                    })
                })
                .collect()
        },
        &mut failed,
    );
    components::assert_no_failures("Car-Parrinello full complex off-shell water", &failed);
}
