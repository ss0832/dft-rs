// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! C1 full-spin Ramsey response. All 81 independent FC/SD/PSO source-pair
//! components are fitted against separately converged complex finite-field
//! SCFs, with no spin/Cartesian symmetry substitution. This initial fixture
//! is all-electron GHF with a generic spin-pinning source, free spin response.
//! It does not certify common-orbital ROKS, PT2 or constrained UKS response.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, Value};
use dft_rs::periodic::spinor_scf::{run_spinor_scf, SpinorScfConfig};
use dft_rs::response::spin_spin::{spinor_nuclear_magnetic_sources, spinor_spin_spin_couplings};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::ResponseSolverConfig;

#[test]
fn every_closed_shell_ramsey_component_retains_the_existing_restricted_limit() {
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("unrestricted", false)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-13)
            .with("conv_tol_grad", 1e-13)
            .with("max_iter", 500usize)
            .with("dispersion", "none"),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let positions = [
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    c1_geometry::assert_c1(&[1; 4], &positions);
    let setup = prepare_molecular(&[1; 4], &positions, &method).unwrap();
    let reference = run_molecular_reference(&setup).unwrap();
    // HF has no XC grid. Both DSO expectations must use the identical
    // explicit zero-cutoff quadrature for a meaningful restricted limit.
    let grid = dft_rs::grid::MolecularGrid::build(
        &setup.molecule,
        &dft_rs::grid::GridOptions {
            weight_cutoff: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let embedded = dft_rs::response::spinor::embedded_collinear_spinor_reference(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        Some(&grid),
        &reference.scf,
        &setup.options,
        1e-11,
    )
    .unwrap();
    let full = spinor_spin_spin_couplings(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        Some(&grid),
        &embedded,
        &setup.options,
        &dft_rs::periodic::BlochMatrix::zeros(8, 8),
        &[],
        &[0, 1],
        false,
        &config,
    )
    .unwrap();
    let old = dft_rs::response::spin_spin::spin_spin_couplings(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        Some(&grid),
        &reference.scf,
        &setup.options,
        &[0, 1],
        false,
        &config,
    )
    .unwrap();
    for (name, a, b) in [
        ("FC", &full.value.fermi_contact, &old.fermi_contact),
        ("SD", &full.value.spin_dipole, &old.spin_dipole),
        ("PSO", &full.value.paramagnetic, &old.paramagnetic),
        ("DSO", &full.value.diamagnetic, &old.diamagnetic),
        ("total", &full.value.total, &old.total),
    ] {
        let mut scale = 0.0_f64;
        let mut defect = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                scale = scale.max(b[0][i][j].abs());
                defect = defect.max((a[0][i][j] - b[0][i][j]).abs());
            }
        }
        println!("full-spin closed-shell {name}: defect={defect:.12e} scale={scale:.12e}");
        assert!(defect < 1e-10 * scale + 1e-21, "closed-shell {name} limit");
    }
    assert!(full.spin_orbit_mixed[0]
        .iter()
        .flatten()
        .all(|v| v.abs() < 1e-21));
}

#[test]
fn every_c1_ghf_electronic_ramsey_component_has_an_independent_scf_fit() {
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("basis", "sto-3g")
            .with("charge", 1)
            .with("multiplicity", 2usize)
            .with("unrestricted", true)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-14)
            .with("conv_tol_dp", 1e-14)
            .with("conv_tol_grad", 1e-14)
            .with("max_iter", 500usize)
            .with("dispersion", "none"),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&controls).unwrap();
    let positions = [
        [0.13, -0.21, 0.19],
        [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37],
        [0.42, 0.29, 2.28],
    ];
    c1_geometry::assert_c1(&[1; 4], &positions);
    let setup = prepare_molecular(&[1; 4], &positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let mut base = spinor_operator::source(4);
    // A C1 nuclear geometry alone does not remove an antiunitary spin-plane
    // symmetry. The earlier on-site source had coplanar spin vectors and
    // identically zero FC-y/PSO components; their retained noise-only fits
    // are not acceptance. Add generic orbital current and noncoplanar spin
    // pinning so that every source-pair component is independently visible.
    for i in 0..4 {
        let spin_z = 0.003 * ((i * i + 2) as f64).sin();
        base.re[(i, i)] += spin_z;
        base.re[(4 + i, 4 + i)] -= spin_z;
        for j in i + 1..4 {
            let current = 0.004 * ((2 * i + 3 * j + 1) as f64).cos();
            for s in 0..2 {
                base.im[(4 * s + i, 4 * s + j)] += current;
                base.im[(4 * s + j, 4 * s + i)] -= current;
            }
        }
    }
    let config = SpinorScfConfig {
        max_iter: 500,
        energy_tolerance: 1e-14,
        density_tolerance: 1e-14,
        orbital_tolerance: 1e-14,
        ..Default::default()
    };
    let solve = |operator| {
        let backend = MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &seed.scf,
            &setup.options,
            operator,
            [0.0, 0.0, 1.0],
            MolecularSpinorFunctionalModel::SpinCovariant,
            1e-12,
        )
        .unwrap();
        let state = run_spinor_scf(&backend, &config).unwrap();
        assert!(
            state.converged,
            "independent finite-source SCF residual {}",
            state.orbital_residual
        );
        state
    };
    let state = solve(base.clone());
    let config = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let coupled = spinor_spin_spin_couplings(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &state,
        &setup.options,
        &base,
        &[],
        &[0, 1],
        false,
        &config,
    )
    .unwrap();
    let sources =
        spinor_nuclear_magnetic_sources(&setup.molecule, &setup.basis, &[0, 1], false).unwrap();
    let norms = sources
        .iter()
        .map(|m| {
            m.re.as_slice()
                .iter()
                .chain(m.im.as_slice())
                .fold(0.0_f64, |a, &b| a.hypot(b))
        })
        .collect::<Vec<_>>();
    assert!(norms.iter().all(|v| *v > 0.0));
    let raw = &coupled.response.values[0];
    let mut failures = Vec::new();
    for source in 9..18 {
        let names = (0..9)
            .map(|observable| format!("C1 GHF Ramsey observable={observable} source={source}"))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &[
                ([0.002, 0.001, 0.0005, 0.00025], stencils::Stencil::Second),
                (
                    [0.0005, 0.00025, 0.000125, 0.0000625],
                    stencils::Stencil::Second,
                ),
                ([0.008, 0.004, 0.002, 0.001], stencils::Stencil::Second),
                ([0.002, 0.001, 0.0005, 0.00025], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                let at = |t: f64| {
                    let mut field = base.clone();
                    field.add_scaled(&sources[source], t / norms[source]);
                    let density = solve(field).density.remove(0);
                    (0..9)
                        .map(|i| sources[i].trace_product(&density) / norms[i])
                        .collect::<Vec<_>>()
                };
                let mut errors = vec![[0.0; 4]; 9];
                for (step, &h) in steps.iter().enumerate() {
                    let (p, m) = (at(h), at(-h));
                    let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                        (at(2.0 * h), at(-2.0 * h))
                    } else {
                        (vec![0.0; 9], vec![0.0; 9])
                    };
                    for i in 0..9 {
                        errors[i][step] = (stencil.first(p[i], m[i], p2[i], m2[i], h)
                            - raw.susceptibility[i][source].re / (norms[i] * norms[source]))
                            .abs();
                    }
                }
                errors
            },
            &mut failures,
        );
    }
    println!(
        "C1 full-spin Ramsey fits=81 response={:.12e} reference={:.12e}",
        coupled.value.residual, coupled.response.reference_fock_defect
    );
    components::assert_no_failures("full-spin Ramsey tensor", &failures);
}
