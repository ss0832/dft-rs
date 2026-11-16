// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All gradient/Hessian entries on C1 complex spinor references with
//! independently moved atom-owned quadrature, for GGA/meta-GGA/hybrid XC.
//! The small explicit quadrature isolates derivative algebra; it does not
//! claim spatial quadrature convergence of a chemical prediction.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_reference.rs"]
mod nuclear_reference;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::grid::{becke_weight_gradient, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::periodic::spinor_scf::{run_spinor_scf, SpinorScfConfig};
use dft_rs::response::nuclear::SpinorNuclearContext;
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::xc::{resolve, FunctionalSpec};

fn evaluate(feature: &str, system: &System, controls: &Controls) -> Value {
    let started = std::time::Instant::now();
    let xc = controls.str("xc", "pbe").unwrap();
    let method = MolecularMethod {
        xc: Some("hf".into()),
        basis: "sto-3g".into(),
        conv_tol: 1e-13,
        conv_tol_grad: Some(1e-12),
        conv_tol_dp: 1e-12,
        max_iter: 300,
        grid_weight_cutoff: 0.0,
        ..Default::default()
    };
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(xc.clone())).unwrap();
    let centers: Vec<_> = setup.molecule.atoms.iter().map(|a| a.position).collect();
    let owner = [0, 1, 2, 1, 3];
    let offsets = [
        [0.31, 0.42, 0.53],
        [-0.21, 0.33, 0.44],
        [0.27, -0.38, 0.49],
        [-0.36, -0.29, 0.41],
        [0.19, 0.37, -0.28],
    ];
    let points: Vec<_> = owner
        .iter()
        .zip(offsets)
        .map(|(&a, r)| centers[a] + Vec3::new(r[0], r[1], r[2]))
        .collect();
    let raw_weights = vec![0.43, 0.67, 0.39, 0.51, 0.62];
    let aij = MolecularGrid::becke_aij(&setup.molecule);
    let weights = points
        .iter()
        .enumerate()
        .map(|(i, &p)| raw_weights[i] * becke_weight_gradient(&centers, &aij, 4, owner[i], p).0)
        .collect();
    let grid = MolecularGrid {
        points,
        weights,
        raw_weights,
        atom_of: owner.to_vec(),
    };
    let operator = spinor_operator::source(4);
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &functional,
        Some(&grid),
        &seed.scf,
        &setup.options,
        operator.clone(),
        [0.3, 0.4, 0.5],
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-12,
    )
    .unwrap();
    let config = SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-11,
        max_iter: 300,
        ..Default::default()
    };
    let scf = run_spinor_scf(&backend, &config).unwrap();
    assert!(scf.converged);
    println!(
        "complex {xc} {feature} reference: {} iterations, orbital residual {:.3e}, elapsed {:.3}s",
        scf.iterations,
        scf.orbital_residual,
        started.elapsed().as_secs_f64()
    );
    let response = ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let order = if feature == "hessian" { 2 } else { 1 };
    let nuclear = SpinorNuclearContext::new(
        &setup.molecule,
        &setup.basis,
        Some(&grid),
        &functional,
        &scf,
        &setup.options,
        &operator,
        order,
        &response,
    )
    .unwrap();
    println!(
        "complex {xc} {feature} response context elapsed {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let tensors = dft_rs::driver::nuclear_tensor::assemble(4, order, 1, |direction| {
        let path = nuclear.energy_taylor(&direction, order).unwrap();
        assert_eq!(path.solved_order, order / 2);
        assert!(
            path.reference_defect < 1e-10,
            "reference Fock defect {}",
            path.reference_defect
        );
        assert!(path.residuals.iter().all(|v| *v < 1e-10));
        Ok(vec![path.energy])
    })
    .unwrap();
    Value::object()
        .with("energy", scf.fields.energy)
        .with(
            "gradient",
            tensors.gradients[0]
                .chunks_exact(3)
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "forces",
            tensors.gradients[0]
                .chunks_exact(3)
                .map(|v| v.iter().map(|v| -*v).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
        .with("hessian", &tensors.hessians[0])
}

fn fit(xc: &str) {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    let controls = Controls::from_value(&Value::object().with("xc", xc)).unwrap();
    nuclear_reference::fit(
        &format!("complex {xc} nuclear"),
        &system,
        &controls,
        &[
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ],
        &[""],
        evaluate,
    );
}

#[test]
fn gga_complex_full_nuclear_tensors() {
    fit("pbe");
}
#[test]
fn meta_gga_complex_full_nuclear_tensors() {
    fit("tpss");
}
#[test]
fn hybrid_complex_full_nuclear_tensors() {
    fit("pbe0");
}
