// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical SCF root acceptance, separate from the older Fe energy-reference branch.

use dft_rs::basis::{data, MolBasis};
use dft_rs::driver::Value;
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix, CoulombKernel};
use dft_rs::linalg::Orthogonalizer;
use dft_rs::scf::fock::JkEngine;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::xc::{resolve, FunctionalSpec};
use dft_rs::{Atom, Molecule, Vec3};

#[test]
fn default_wc04_ferrocene_satisfies_fresh_physical_fock_stationarity() {
    let data = Value::parse_json(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle_data/catalogue_hybrid_molecules.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let geometry = data.get("molecules").unwrap().get("ferrocene").unwrap();
    let numbers = geometry.get("numbers").unwrap().numeric_array().unwrap().1;
    let Value::Array(positions) = geometry.get("positions").unwrap() else {
        panic!("geometry positions required")
    };
    let mol = Molecule::new(
        numbers
            .iter()
            .zip(positions)
            .map(|(z, p)| {
                let r = p.numeric_array().unwrap().1;
                Atom {
                    z: *z as u8,
                    position: Vec3::new(r[0], r[1], r[2]),
                }
            })
            .collect(),
    );
    let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &mol,
        &GridOptions {
            level: 9,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined("wc04".into())).unwrap();
    let options = ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-10,
        conv_tol_grad: 1e-9,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..ScfOptions::default()
    };
    let scf = run_scf(&mol, &basis, &options, &functional, Some(&grid)).unwrap();
    println!(
        "WC04 Fe default converged={} iterations={} energy={:.15e}",
        scf.converged, scf.iterations, scf.energy
    );
    assert!(
        scf.converged,
        "the original physical convergence gates remain mandatory"
    );
    // Rebuild the operator independently of the saved iteration Fock.
    // An accepted minimizer step or a small energy change alone is insufficient.
    let overlap = overlap_matrix(&basis);
    let orth = Orthogonalizer::canonical(&overlap, options.orth_threshold).unwrap();
    let mut total = scf.density[0].clone();
    for (a, b) in total
        .as_mut_slice()
        .iter_mut()
        .zip(scf.density[1].as_slice())
    {
        *a += b;
    }
    let mut core = kinetic_matrix(&basis);
    let nuclear = nuclear_matrix(&basis, &mol, None);
    for (a, b) in core.as_mut_slice().iter_mut().zip(nuclear.as_slice()) {
        *a += b;
    }
    let engine = JkEngine::new(&basis, options.eri_mode, CoulombKernel::Coulomb).unwrap();
    let (j, k) = engine
        .build(&basis, &total, [&scf.density[0], &scf.density[1]], true)
        .unwrap();
    let xc = dft_rs::ks::xc_contribution(
        &functional,
        &basis,
        &grid,
        [&scf.density[0], &scf.density[1]],
    )
    .unwrap();
    assert_eq!(functional.hybrid.clr, 0.0);
    let mut energy = mol.nuclear_repulsion()
        + core.frobenius_dot(&total)
        + 0.5 * j.frobenius_dot(&total)
        + xc.exc;
    for (spin, exchange) in k.iter().enumerate() {
        energy -= 0.5 * functional.hybrid.cx * exchange.frobenius_dot(&scf.density[spin]);
        let mut fock = core.clone();
        for (i, v) in fock.as_mut_slice().iter_mut().enumerate() {
            *v += j.as_slice()[i] - functional.hybrid.cx * exchange.as_slice()[i]
                + xc.vxc[spin].as_slice()[i];
        }
        assert!(fock.max_abs_diff(&scf.fock[spin]) < 1e-10);
        let density = &scf.density[spin];
        let mut commutator = fock.matmul_seq(density).matmul_seq(&overlap);
        let reverse = overlap.matmul_seq(density).matmul_seq(&fock);
        for (a, b) in commutator.as_mut_slice().iter_mut().zip(reverse.as_slice()) {
            *a -= b;
        }
        let metric = orth.x.transpose_matmul_seq(&commutator).matmul_seq(&orth.x);
        let residual = metric
            .as_slice()
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        println!("WC04 Fe spin{spin} fresh metric commutator={residual:.12e}");
        assert!(residual < options.conv_tol_grad);
        assert!(
            density
                .matmul_seq(&overlap)
                .matmul_seq(density)
                .max_abs_diff(density)
                < 1e-10
        );
        let count = [scf.n_alpha, scf.n_beta][spin] as f64;
        assert!((density.frobenius_dot(&overlap) - count).abs() < 1e-10);
    }
    assert!((energy - scf.energy).abs() < 1e-10);
}
