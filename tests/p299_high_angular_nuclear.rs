// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! C1 high-angular nuclear acceptance through the supported i-shell limit.
//! Every model has occupied and virtual orbitals. Full Hessians and complete
//! unisolvent cubic/quartic sets use exactly the shared p290 fit gates.
//! These compact ionic models test derivatives, not chemical basis accuracy.
//! Measured complete subsets and retained failures are recorded in response.md.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nuclear_components.rs"]
mod nuclear_components;
#[path = "support/quartic_oracle.rs"]
mod quartic_oracle;
#[path = "support/spdf_fixture.rs"]
mod spdf_fixture;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_cartesian_directions.rs"]
mod symmetric_directions;

fn fixture(
    l: u8,
    ecp: bool,
    unrestricted: bool,
) -> (dft_rs::driver::System, dft_rs::driver::Value, &'static str) {
    assert!((4..=dft_rs::basis::aos::MAX_L).contains(&l));
    let name = match l {
        4 => "g",
        5 => "h",
        6 => "i",
        _ => unreachable!(),
    };
    let (mut system, controls) = spdf_fixture::fixture(ecp, unrestricted, "hf");
    system.numbers[1] = 2; // Off-core high-l centre; two remaining H centres.
    let electrons = if unrestricted { 3i32 } else { 2i32 };
    let charge = if ecp { 9i32 } else { 11i32 } - electrons;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/data/c1_{name}_model.g94"));
    let controls = controls
        .with("basis", path.to_str().unwrap())
        .with("charge", charge);
    (system, controls, name)
}

fn fit_case(l: u8, ecp: bool, unrestricted: bool, order: usize) {
    let (system, controls, name) = fixture(l, ecp, unrestricted);
    let electrons = if unrestricted { 3i32 } else { 2i32 };
    nuclear_components::fit_case(
        system,
        controls,
        nuclear_components::BasisContract {
            scalar_ecp: ecp,
            highest_l: l,
            ao_count: 2 * l as usize + 6,
            required_l: &[0, l],
        },
        format!(
            "C1 s/{name} ecp={ecp} unrestricted={unrestricted} electrons={electrons} order={order}"
        ),
        order,
    );
}

#[test]
fn i_shell_uhf_newton_stall_restarts_without_relaxing_scf_gates() {
    use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    use dft_rs::driver::Controls;

    let (mut system, controls, _) = fixture(6, false, true);
    let direction = &symmetric_directions::directions(12, 3)[285];
    for (position, &d) in system.positions.iter_mut().flatten().zip(direction) {
        *position += 0.04 * d;
    }
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let mut energies = Vec::new();
    for disable_newton in [false, true] {
        let method = MolecularMethod::from_controls(
            &Controls::from_value(&controls.clone().with("no_soscf", disable_newton)).unwrap(),
        )
        .unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        assert_eq!(setup.options.e_tol, 1e-14);
        assert_eq!(setup.options.p_tol, 1e-14);
        assert_eq!(setup.options.conv_tol_grad, 1e-14);
        let reference = run_molecular_reference(&setup).unwrap();
        assert!(reference.scf.converged);
        assert!(reference.scf.iterations < 64);
        let orth =
            dft_rs::Orthogonalizer::canonical(&reference.scf.overlap, setup.options.orth_threshold)
                .unwrap();
        let residual = dft_rs::scf::diis::Diis::new(2).push(
            &reference.scf.fock,
            &reference.scf.density,
            &reference.scf.overlap,
            &orth.x,
        );
        let diagnostics =
            dft_rs::response::diagnostics::reference_diagnostics(&reference.scf).unwrap();
        println!("i-UHF Newton stall recovery no_soscf={disable_newton} iterations={} physical_commutator={residual:.12e} energy={:.16e} diagnostics={diagnostics:?}",reference.scf.iterations,reference.scf.energy);
        assert!(residual < setup.options.conv_tol_grad);
        energies.push(reference.scf.energy);
    }
    assert!((energies[0] - energies[1]).abs() < 2e-12);
}

macro_rules! cases {
    ($(($name:ident,$l:expr,$ecp:expr,$unrestricted:expr,$order:expr)),* $(,)?)=>{$(
        #[test] fn $name(){fit_case($l,$ecp,$unrestricted,$order);}
    )*};
}
cases! {
    (g_ae_rhf_hessian,4,false,false,2),(g_ae_rhf_cubic,4,false,false,3),(g_ae_rhf_quartic,4,false,false,4),
    (g_ecp_rhf_hessian,4,true,false,2),(g_ecp_rhf_cubic,4,true,false,3),(g_ecp_rhf_quartic,4,true,false,4),
    (g_ae_uhf_hessian,4,false,true,2),(g_ae_uhf_cubic,4,false,true,3),(g_ae_uhf_quartic,4,false,true,4),
    (g_ecp_uhf_hessian,4,true,true,2),(g_ecp_uhf_cubic,4,true,true,3),(g_ecp_uhf_quartic,4,true,true,4),
    (h_ae_rhf_hessian,5,false,false,2),(h_ae_rhf_cubic,5,false,false,3),(h_ae_rhf_quartic,5,false,false,4),
    (h_ecp_rhf_hessian,5,true,false,2),(h_ecp_rhf_cubic,5,true,false,3),(h_ecp_rhf_quartic,5,true,false,4),
    (h_ae_uhf_hessian,5,false,true,2),(h_ae_uhf_cubic,5,false,true,3),(h_ae_uhf_quartic,5,false,true,4),
    (h_ecp_uhf_hessian,5,true,true,2),(h_ecp_uhf_cubic,5,true,true,3),(h_ecp_uhf_quartic,5,true,true,4),
    (i_ae_rhf_hessian,6,false,false,2),(i_ae_rhf_cubic,6,false,false,3),(i_ae_rhf_quartic,6,false,false,4),
    (i_ecp_rhf_hessian,6,true,false,2),(i_ecp_rhf_cubic,6,true,false,3),(i_ecp_rhf_quartic,6,true,false,4),
    (i_ae_uhf_hessian,6,false,true,2),(i_ae_uhf_cubic,6,false,true,3),(i_ae_uhf_quartic,6,false,true,4),
    (i_ecp_uhf_hessian,6,true,true,2),(i_ecp_uhf_cubic,6,true,true,3),(i_ecp_uhf_quartic,6,true,true,4),
}

#[test]
fn every_high_angular_ecp_matrix_entry_matches_direct_quadrature() {
    use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
    use dft_rs::driver::{Controls, Value};

    let oracle = Value::parse_json(include_str!("oracle_data/high_angular_ecp.json")).unwrap();
    for row in oracle.get("cases").unwrap().as_array().unwrap() {
        let shell = row.get("shell").unwrap().as_str().unwrap();
        let (mut system, controls) = spdf_fixture::fixture(true, false, "hf");
        system.numbers[1] = 2;
        let basis = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/data/c1_{shell}_model.g94"));
        let controls = controls
            .with("basis", basis.to_str().unwrap())
            .with("charge", 7i32);
        let method =
            MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        let actual = dft_rs::integrals::ecp_matrix(&setup.basis).unwrap();
        let (shape, expected) = row.get("matrix").unwrap().numeric_array().unwrap();
        assert_eq!(shape, vec![actual.rows, actual.cols]);
        let defect = actual
            .as_slice()
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "C1 s/{shell} ECP independent radial/angular oracle entries={} max_defect={defect:.12e}",
            expected.len()
        );
        assert!(defect < 2e-11);
    }
}

#[test]
fn high_angular_hf_energies_and_every_energy_term_match_independent_oracles() {
    use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    use dft_rs::driver::{Controls, Value};
    use dft_rs::integrals::{eri_incore, CoulombKernel};
    use dft_rs::scf::fock::jk_incore;
    use dft_rs::Matrix;

    let oracle = Value::parse_json(include_str!("oracle_data/high_angular_hf_terms.json")).unwrap();
    let dot = |a: &Matrix, b: &Matrix| {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
    for row in oracle.get("cases").unwrap().as_array().unwrap() {
        let tag = row.get("case").unwrap().as_str().unwrap();
        let shell = row.get("shell").unwrap().as_str().unwrap();
        let ecp = row.get("scalar_ecp").unwrap().as_bool().unwrap();
        let unrestricted = row.get("unrestricted").unwrap().as_bool().unwrap();
        let (mut system, controls) = spdf_fixture::fixture(ecp, unrestricted, "hf");
        system.numbers[1] = 2;
        c1_geometry::assert_c1(&system.numbers, &system.positions);
        let electrons = if unrestricted { 3i32 } else { 2i32 };
        let basis = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/data/c1_{shell}_model.g94"));
        let controls = controls
            .with("basis", basis.to_str().unwrap())
            .with("charge", if ecp { 9i32 } else { 11i32 } - electrons);
        let method =
            MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
        let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
        let result = run_molecular_reference(&setup).unwrap();
        assert!(result.scf.converged);
        let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 64).unwrap();
        let spin = &result.scf.density;
        let mut total = spin[0].clone();
        for (a, b) in total.as_mut_slice().iter_mut().zip(spin[1].as_slice()) {
            *a += b;
        }
        let (j, k) = jk_incore(&eri, &total, [&spin[0], &spin[1]], true);
        let terms = [
            ("nuclear_repulsion", result.scf.e_nuc),
            ("one_electron", dot(&total, &result.scf.h_core)),
            ("hartree", 0.5 * dot(&total, &j)),
            (
                "exchange",
                -0.5 * (dot(&spin[0], &k[0]) + dot(&spin[1], &k[1])),
            ),
        ];
        for (name, actual) in terms {
            let expected = row.get(name).unwrap().as_f64().unwrap();
            let defect = (actual - expected).abs();
            println!("C1 {tag} independent HF term={name} defect={defect:.12e} Hartree");
            assert!(defect < 2e-11, "{tag} {name} defect={defect:.12e}");
        }
        let expected = row.get("energy").unwrap().as_f64().unwrap();
        let defect = (result.scf.energy - expected).abs();
        println!("C1 {tag} independent HF total defect={defect:.12e} Hartree");
        assert!(defect < 2e-10);
        assert!(
            (terms.iter().map(|(_, value)| value).sum::<f64>() - result.scf.energy).abs() < 2e-11
        );
    }
}
