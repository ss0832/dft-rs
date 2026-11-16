// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::BlochMatrix;
use crate::acceptance::{components, stencils};
use crate::driver::periodic::{prepare_periodic, PeriodicMethod};
use crate::driver::{run, Controls, System, Value};
use crate::math::Vec3;
use crate::periodic::integrals::{image_blocks, Operator};
use crate::periodic::scf::{build_fields, run_scf_iterations, ScfIterationInput};
use crate::periodic::xc::{AoDerivativeOrder, PeriodicXcGrid};
use std::collections::BTreeMap;
use stencils::Stencil;

fn flat(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

/// Converge the direct coupling +F.r in the localized AO basis and
/// evaluate the dipole as an expectation of that operator.
fn local_dipole(system: &System, settings: &Value, field: [f64; 3]) -> ([f64; 3], f64) {
    let method = PeriodicMethod::from_controls(&Controls::from_value(settings).unwrap()).unwrap();
    let setup = prepare_periodic(system, &method).unwrap();
    let (cell, functional, options) = (&setup.cell, &setup.functional, &setup.options);
    let gamma = Vec3::zero();
    let k_points = [gamma];
    let basis = cell.basis();
    let images = image_blocks(
        basis,
        cell.lattice(),
        cell.overlap_cutoff(),
        Operator::Overlap,
    );
    assert_eq!(
        images.len(),
        1,
        "Cartesian field oracle needs on-site AO pairs"
    );
    assert_eq!(images[0].translation.norm(), 0.0);
    let moment = crate::integrals::one_electron::dipole_matrices(basis, gamma);
    let mut core = BlochMatrix::from_pair(cell.core_hamiltonian(gamma));
    for (operator, strength) in moment.iter().zip(field) {
        for (element, r) in core.re.as_mut_slice().iter_mut().zip(operator.as_slice()) {
            *element += strength * r;
        }
    }
    let grid = PeriodicXcGrid::build(
        cell,
        &k_points,
        &options.grid,
        options.grid_shell,
        options.xc_ao_cutoff.unwrap(),
        AoDerivativeOrder::for_scf(functional.family),
    )
    .unwrap();
    let electrons = cell.n_electrons().unwrap() / 2;
    let result = run_scf_iterations(
        ScfIterationInput {
            k_points: k_points.to_vec(),
            overlap: vec![BlochMatrix::from_pair(cell.overlap(gamma))],
            h_core: vec![core.clone()],
            guess_focks: vec![core.clone()],
            n_alpha: electrons,
            n_beta: electrons,
            restricted: true,
            enforce_orbital_conjugation: true,
            e_nuc: cell.nuclear_energy(),
            derivative_plan: None,
        },
        options,
        |density| {
            build_fields(
                cell,
                functional,
                &k_points,
                density,
                std::slice::from_ref(&core),
                Some(&grid),
                cell.overlap_cutoff(),
                true,
            )
        },
    )
    .unwrap();
    assert!(result.converged);
    let charges = basis.effective_charges(cell.molecule());
    let dipole = std::array::from_fn(|a| {
        let ionic = cell
            .molecule()
            .atoms
            .iter()
            .zip(&charges)
            .map(|(atom, charge)| charge * atom.position.get(a))
            .sum::<f64>();
        ionic - 2.0 * result.density[0][0].re.frobenius_dot(&moment[a])
    });
    (dipole, result.orbital_residual.unwrap())
}

/// Compare the conventional covariant-k response with finite perturbations
/// of independently converged SCF dipoles. In the localized Gamma cell
/// the retained AO pairs are all on-site, so the position operator reduces
/// to the ordinary Cartesian dipole matrix.
#[test]
fn scalar_ecp_electric_and_proper_strain_response_fit_all_components() {
    let system = System {
        numbers: vec![1, 11, 1, 1],
        positions: vec![
            [0.13, -0.21, 0.07],
            [2.0, 0.45, -0.31],
            [-0.46, 1.29, 1.62],
            [-1.34, 1.72, 2.63],
        ],
        cell: Some([[9.4, 0.7, 0.3], [0.4, 9.9, 0.6], [0.2, -0.5, 10.2]]),
        pbc: [true; 3],
    };
    let settings = Value::object()
        .with("xc", "pbe")
        .with("reference", "rks")
        .with("dispersion", "none")
        .with(
            "basis",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/periodic_scalar_ecp.g94"
            ),
        )
        .with("k_mesh", [1, 1, 1])
        .with("overlap_cutoff", 6.0)
        .with("xc_ao_cutoff", 16.0)
        .with("grid_level", 1)
        .with("grid_shell", 1)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-11)
        .with("conv_tol_grad", 1e-11)
        .with("max_iter", 400);
    let calculate = |feature, geometry: &System, input: &Value| {
        let result = run(feature, geometry, &Controls::from_value(input).unwrap())
            .unwrap_or_else(|error| panic!("{feature} scalar ECP calculation: {error}"));
        assert_eq!(
            result.get("converged").and_then(Value::as_bool),
            Some(true),
            "{feature} scalar ECP SCF"
        );
        result
    };
    let scalar = |report: &Value, key| report.get(key).and_then(Value::as_f64).unwrap();
    let born = calculate(
        "born-charges",
        &system,
        &settings.clone().with("response_tolerance", 1e-11),
    );
    let piezo = calculate(
        "piezoelectric-tensor",
        &system,
        &settings
            .clone()
            .with("response_tolerance", 1e-11)
            .with("relaxed_ion", false),
    );
    let volume = scalar(&born, "volume");
    let z = flat(&born, "born_charges");
    let chi = flat(&born, "susceptibility");
    let e = flat(&piezo, "clamped_ion_full");
    let neutrality = flat(&born, "born_sum_rule");
    for report in [&born, &piezo] {
        assert!(scalar(report, "fock_image_residual") < 1e-8);
    }
    assert!(neutrality.iter().all(|z| z.abs() < 1e-8));
    println!(
        "scalar ECP PBE RKS E={:.17e} epsilon={:?} Born={z:?} neutrality={neutrality:?} proper piezo={e:?} Fock defect={:.17e} field solves={:?} displacement solves={:?} strain solves={:?}",
        scalar(&born, "energy"),
        flat(&born, "epsilon_infinity"),
        scalar(&born, "fock_image_residual"),
        born.get("field_solves"),
        born.get("displacement_solves"),
        piezo.get("strain_solves"),
    );
    let dof = 3 * system.numbers.len();
    let mut exact = Vec::new();
    let mut labels = Vec::new();
    for col in 0..(dof + 12) {
        for a in 0..3 {
            let (label, value) = if col < 3 {
                (
                    format!("susceptibility [{a},{col}]"),
                    volume * chi[3 * a + col],
                )
            } else if col < 3 + dof {
                let (atom, b) = ((col - 3) / 3, (col - 3) % 3);
                (format!("Born [{atom},{a},{b}]"), z[9 * atom + 3 * a + b])
            } else {
                let c = col - 3 - dof;
                (format!("proper piezo [{a},{c}]"), volume * e[9 * a + c])
            };
            labels.push(format!("scalar ECP PBE RKS {label}"));
            exact.push(value);
        }
    }
    let sample = |col: usize, h: f64| {
        let mut moved = system.clone();
        let mut field = [0.0; 3];
        if col < 3 {
            field[col] = h;
        } else if col < 3 + dof {
            let u = col - 3;
            moved.positions[u / 3][u % 3] += h;
        } else {
            let c = col - 3 - dof;
            // The same affine map acts on the ions and the cell columns.
            for point in moved
                .positions
                .iter_mut()
                .chain(moved.cell.as_mut().unwrap())
            {
                point[c / 3] += h * point[c % 3];
            }
        }
        let (mut dipole, residual) = local_dipole(&moved, &settings, field);
        if col >= 3 + dof {
            let c = col - 3 - dof;
            let (a, b) = (c / 3, c % 3);
            // Proper polarization is the dipole pulled back by F^-1.
            dipole[a] -= h * dipole[b] / (1.0 + h * f64::from(a == b));
        }
        println!(
            "scalar ECP PBE RKS dipole sample [{col}] h={h:.17e} dipole={dipole:?} SCF={:.17e}",
            residual
        );
        dipole
    };
    let mut samples = BTreeMap::new();
    let mut failures = Vec::new();
    let field_scale = |col| if col < 3 { 0.1 } else { 1.0 };
    // Common step coordinates have a fixed scale per perturbation family.
    // The logged signed samples give the physical steps for each fit.
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; exact.len()];
            for col in 0..(dof + 12) {
                let mut at = |h: f64| {
                    *samples
                        .entry((col, h.to_bits()))
                        .or_insert_with(|| sample(col, h))
                };
                for (s, &step) in steps.iter().enumerate() {
                    let h = step * field_scale(col);
                    let (plus, minus) = (at(h), at(-h));
                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        ([0.0; 3], [0.0; 3])
                    };
                    for a in 0..3 {
                        let i = 3 * col + a;
                        errors[i][s] = (stencil.first(plus[a], minus[a], plus2[a], minus2[a], h)
                            - exact[i])
                            .abs();
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures("scalar ECP polarization response", &failures);
    let mut richardson = [0.0_f64; 3];
    for col in 0..(dof + 12) {
        let class = if col < 3 {
            0
        } else if col < 3 + dof {
            1
        } else {
            2
        };
        for a in 0..3 {
            let derivative = |step: f64| {
                let h = step * field_scale(col);
                Stencil::Second.first(
                    samples[&(col, h.to_bits())][a],
                    samples[&(col, (-h).to_bits())][a],
                    0.0,
                    0.0,
                    h,
                )
            };
            let error =
                ((4.0 * derivative(0.005) - derivative(0.01)) / 3.0 - exact[3 * col + a]).abs();
            richardson[class] = richardson[class].max(error);
        }
    }
    println!("scalar ECP dipole response Richardson defects {richardson:?}");
    assert!(richardson.iter().all(|&error| error < 2e-7));
}
