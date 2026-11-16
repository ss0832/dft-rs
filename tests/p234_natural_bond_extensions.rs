// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance of the extensions of the natural bond orbital analysis:
//! functional classes, correlated densities, periodic cells, natural
//! resonance theory, the natural chemical shielding and J-coupling
//! decompositions and the natural energy decomposition.
//!
//! Oracles (finite differences are validation only): the dipole of the
//! relaxed PT2 (double-hybrid) and CASSCF densities against four-step fits
//! of central differences of the energies in a uniform field, and the
//! relaxed-density dipole against the analytic double-hybrid dipole; the
//! Born-von Karman unfolding against the explicit supercell at the Gamma
//! point; resonance weights of symmetry-equivalent Kekule structures; the
//! shielding and coupling decompositions against their totals and the
//! symmetry of equivalent bonds; the energy decomposition identity, its
//! separated-fragment limit and the counterpoise interaction energy of the
//! independent SAPT supermolecular route; per functional class the NBO
//! metric, electron count, NLMO density, independent second-order energies
//! and invariance under a rotation of the molecule.

#[path = "support/convergence.rs"]
mod convergence;

use dft_rs::driver::method::{
    prepare_molecular, run_collinear_property_reference, run_molecular_reference,
    CollinearPropertyReference, MolecularMethod,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::eval::eval_aos;
use dft_rs::math::Vec3;
use dft_rs::properties::natural_bond::spectral::{
    effectively_unpaired_weights, NaturalDensityOrbitals,
};
use dft_rs::properties::natural_bond::{natural_bond_analysis, NboChannel, NboOptions, NboSpin};
use dft_rs::Matrix;

const WATER: [(u8, [f64; 3]); 3] = [
    (8, [0.0, 0.0, 0.23]),
    (1, [0.0, 1.4309, -0.8867]),
    (1, [0.12, -1.4809, -0.9067]),
];

fn system(atoms: &[(u8, [f64; 3])]) -> System {
    System::molecule(
        atoms.iter().map(|(z, _)| *z).collect(),
        atoms.iter().map(|(_, r)| *r).collect(),
    )
}

fn controls(pairs: &[(&str, Value)]) -> Controls {
    let mut all = vec![
        ("conv_tol".to_string(), Value::Float(1e-12)),
        ("conv_tol_grad".to_string(), Value::Float(1e-9)),
    ];
    for (k, v) in pairs {
        all.retain(|(key, _)| key != k);
        all.push((k.to_string(), v.clone()));
    }
    Controls::from_value(&Value::Object(all)).unwrap()
}

fn text(v: &str) -> Value {
    Value::Str(v.into())
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}

fn number(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap()
}

fn field(h: f64) -> Value {
    Value::Array(vec![Value::Float(0.0), Value::Float(0.0), Value::Float(h)])
}

fn fit(label: &str, steps: &[f64], errors: &[f64]) {
    let fit = convergence::error_power_fit(steps, errors);
    println!(
        "{label}: steps {steps:?} errors {errors:?} p={:.6} R2={:.8} C={:.3e}",
        fit.exponent, fit.r_squared, fit.prefactor
    );
    assert!(
        (fit.exponent - 2.0).abs() < 0.2,
        "{label}: p = {}",
        fit.exponent
    );
    assert!(fit.r_squared > 0.999, "{label}: R2 = {}", fit.r_squared);
}

// Reconstruct the density and the donor-acceptor energies from the reported
// orbitals and the original SCF overlap and Fock operators.
fn channel_defects(
    channel: &NboChannel,
    nao: &Matrix,
    density: &Matrix,
    overlap: &Matrix,
    fock: &Matrix,
    electrons: usize,
) -> [f64; 4] {
    let b = nao.matmul(&channel.coefficients);
    let metric = b.transpose_matmul(&overlap.matmul(&b));
    let total: f64 = channel.nbos.iter().map(|n| n.occupancy).sum();
    let mut rebuilt = Matrix::zeros(density.rows, density.cols);
    for (k, nlmo) in channel.nlmos.iter().enumerate() {
        for i in 0..density.rows {
            for j in 0..density.cols {
                rebuilt[(i, j)] += nlmo.occupancy
                    * channel.nlmo_coefficients[(i, k)]
                    * channel.nlmo_coefficients[(j, k)];
            }
        }
    }
    let fock = b.transpose_matmul(&fock.matmul(&b));
    let mut independent = 0.0;
    for (i, donor) in channel.nbos.iter().enumerate() {
        for (j, acceptor) in channel.nbos.iter().enumerate() {
            if donor.kind.is_lewis() && !acceptor.kind.is_lewis() {
                let gap = fock[(j, j)] - fock[(i, i)];
                if gap > 0.0 {
                    independent += donor.occupancy * fock[(i, j)].powi(2) / gap;
                }
            }
        }
    }
    [
        metric.max_abs_diff(&Matrix::identity(metric.rows)),
        (total - electrons as f64).abs(),
        rebuilt.max_abs_diff(density),
        (independent - channel.perturbation_total.unwrap()).abs(),
    ]
}

#[test]
fn every_functional_class_satisfies_the_orbital_identities() {
    for xc in ["svwn5", "pbe", "tpss", "pbe0", "wb97x"] {
        let mut defects = [0.0_f64; 5];
        let mut charges = Vec::new();
        for rotated in [false, true] {
            let atoms: Vec<(u8, [f64; 3])> = WATER
                .iter()
                .map(|(z, r)| {
                    let r = if rotated { [r[2], r[0], r[1]] } else { *r };
                    (*z, r)
                })
                .collect();
            let method = MolecularMethod::from_controls(&controls(&[
                ("xc", text(xc)),
                ("basis", text("6-31g*")),
                ("grid_level", Value::Int(7)),
            ]))
            .unwrap();
            let s = system(&atoms);
            let setup = prepare_molecular(&s.numbers, &s.positions, &method).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            let scf = &reference.scf;
            let analysis = natural_bond_analysis(
                &setup.molecule,
                &setup.basis,
                scf,
                NboSpin::Auto,
                &NboOptions::default(),
            )
            .unwrap();
            for (max, defect) in defects.iter_mut().zip(channel_defects(
                &analysis.channels[0],
                &analysis.nao.nao_coefficients,
                &analysis.nao.density_nao,
                &scf.overlap,
                &scf.fock[0],
                10,
            )) {
                *max = max.max(defect);
            }
            charges.push(analysis.nao.atomic_charges.clone());
        }
        defects[4] = charges[0]
            .iter()
            .zip(&charges[1])
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "{xc}: NBO metric {:.3e}, electron count {:.3e}, NLMO density {:.3e}, second-order \
             total {:.3e}, rotated natural charges {:.3e}; charges {:?}",
            defects[0], defects[1], defects[2], defects[3], defects[4], charges[0]
        );
        assert!(defects[0] < 1e-10 && defects[1] < 1e-8 && defects[2] < 1e-8);
        assert!(defects[3] < 1e-10);
        // The molecular grid turns with the molecule only approximately (the
        // first run at grid level 5 changed the wB97X charges by 1.4e-5).
        assert!(defects[4] < 1e-5);
    }
}

#[test]
fn open_shell_range_separated_nbo_follows_both_spin_channels() {
    let system = system(&WATER);
    for xc in ["wb97x-d", "wb97x-v"] {
        for kind in [
            CollinearPropertyReference::Uks,
            CollinearPropertyReference::Roks,
        ] {
            let spin = kind.name();
            let method = MolecularMethod::from_controls(&controls(&[
                ("xc", text(xc)),
                ("basis", text("sto-3g")),
                ("charge", Value::Int(1)),
                ("multiplicity", Value::Int(2)),
                ("grid_level", Value::Int(1)),
                ("vv10_grid_level", Value::Int(0)),
                ("grid_weight_cutoff", Value::Float(0.0)),
            ]))
            .unwrap();
            let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
            let scf = run_collinear_property_reference(&setup, kind).unwrap().scf;
            let analysis = natural_bond_analysis(
                &setup.molecule,
                &setup.basis,
                &scf,
                NboSpin::Auto,
                &NboOptions::default(),
            )
            .unwrap();
            assert!(scf.converged && analysis.open_shell);
            assert_eq!(analysis.channels.len(), 2);
            assert!((analysis.nao.atomic_charges.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            let spins = &analysis.nao.spin_populations;
            assert!((spins.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            let uco = dft_rs::properties::natural_bond::corresponding::corresponding_orbitals(&scf)
                .unwrap();
            assert!(uco.cross_overlap_residual < 1e-12);
            assert!((uco.spin_squared - scf.s_squared).abs() < 1e-11);
            assert_eq!(
                (
                    uco.unpaired_coefficients[0].cols,
                    uco.unpaired_coefficients[1].cols
                ),
                (1, 0)
            );
            for (i, &sigma) in uco.paired_overlap.iter().enumerate() {
                let a = &uco.paired_coefficients[0];
                let b = &uco.paired_coefficients[1];
                let direct: f64 = (0..a.rows)
                    .flat_map(|p| (0..a.rows).map(move |q| (p, q)))
                    .map(|(p, q)| a[(p, i)] * scf.overlap[(p, q)] * b[(q, i)])
                    .sum();
                assert!((direct - sigma).abs() < 1e-12);
            }
            println!("{xc}/{spin} water cation corresponding overlaps {:?}; cross defect {:.3e}; spin squared {:.12e}",
                     uco.paired_overlap, uco.cross_overlap_residual, uco.spin_squared);
            let natural = NaturalDensityOrbitals::from_spin_density(&analysis.nao).unwrap();
            let charge = NaturalDensityOrbitals::from_charge_density(&analysis.nao).unwrap();
            let unpaired: f64 = effectively_unpaired_weights(&charge.weights).iter().sum();
            let density = &analysis.nao.density_nao;
            let squared = density.matmul(density);
            let unpaired_matrix_trace: f64 = density
                .as_slice()
                .iter()
                .zip(squared.as_slice())
                .map(|(&p, &p2)| (2.0 * p - p2).powi(2))
                .sum();
            assert!((unpaired - unpaired_matrix_trace).abs() < 1e-12);
            assert!(charge.reconstruction_residual < 1e-12 && charge.trace_residual.abs() < 1e-12);
            assert!((charge.weights.iter().sum::<f64>() - 9.0).abs() < 1e-9);
            println!("{xc}/{spin} water cation spatial occupations {:?}; effectively unpaired electrons {unpaired:.15e}", charge.weights);
            let points = [
                Vec3::new(0.34, -0.19, 0.32),
                Vec3::new(-0.81, 0.16, -0.17),
                Vec3::new(1.17, 0.62, -0.29),
            ];
            let ao = eval_aos(&setup.basis, &points, false).phi;
            let mut spin_ao = scf.density[0].clone();
            for (a, b) in spin_ao
                .as_mut_slice()
                .iter_mut()
                .zip(scf.density[1].as_slice())
            {
                *a -= b;
            }
            let direct = ao.matmul(&spin_ao).matmul_transpose_seq(&ao);
            let values = natural.values_at(&setup.basis, &points).unwrap();
            let max_spatial_error = values
                .iter()
                .enumerate()
                .map(|(i, &value)| (value - direct[(i, i)]).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "{xc}/{spin} water cation spin-density weights {:?}; point values {values:?}; AO density error {max_spatial_error:.3e}; NAO reconstruction {:.3e}; trace {:.3e}",
                natural.weights, natural.reconstruction_residual, natural.trace_residual,
            );
            assert!(max_spatial_error < 1e-12 && natural.reconstruction_residual < 1e-12);
            assert!(natural.trace_residual.abs() < 1e-12);
            assert!((natural.weights.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            if kind == CollinearPropertyReference::Roks {
                let (unpaired, paired) = natural.weights.split_last().unwrap();
                assert!((unpaired - 1.0).abs() < 1e-10);
                assert!(paired.iter().all(|v| v.abs() < 1e-10));
                assert!((unpaired - 1.0).abs() < 1e-10);
            }
            for (index, electrons) in [5, 4].into_iter().enumerate() {
                let channel = &analysis.channels[index];
                let defects = channel_defects(
                    channel,
                    &analysis.nao.nao_coefficients,
                    &analysis.nao.spin_density_nao[index],
                    &scf.overlap,
                    &scf.fock[index],
                    electrons,
                );
                println!(
                    "{xc}/{spin} water cation spin {index}: charges {:?}, spin populations {spins:?}, +                     donor-acceptor total {:?}, metric/count/density/energy defects {defects:?}",
                    analysis.nao.atomic_charges, channel.perturbation_total,
                );
                assert!(defects[0] < 1e-10 && defects[1] < 1e-8);
                assert!(defects[2] < 1e-8 && defects[3] < 1e-10);
            }
        }
    }
}

#[test]
fn correlated_hydrogen_dissociation_distinguishes_spin_and_unpaired_density() {
    let request = controls(&[
        ("xc", text("pbe")),
        ("basis", text("sto-3g")),
        ("cas", Value::Array(vec![Value::Int(2), Value::Int(2)])),
        ("nbo_density", text("casscf")),
        ("nbo_natural_orbitals", Value::Bool(true)),
        ("nbo_spin_density", Value::Bool(true)),
        ("casscf_gradient_tol", Value::Float(1e-10)),
        ("ci_tol", Value::Float(1e-12)),
    ]);
    let mut effective = Vec::new();
    for length in [1.4, 5.5] {
        let molecule = System::molecule(
            vec![1, 1],
            vec![[0.0, 0.0, -0.5 * length], [0.0, 0.0, 0.5 * length]],
        );
        let report = run("natural-bond", &molecule, &request).unwrap();
        assert_eq!(
            report.get("orbitals_converged").and_then(Value::as_bool),
            Some(true)
        );
        let natural = report.get("natural_orbitals").unwrap();
        let spin = report.get("natural_spin_density").unwrap();
        let weights = floats(spin, "weights");
        let occupations = floats(natural, "occupations");
        let unpaired = number(natural, "effectively_unpaired_electrons");
        assert!((occupations.iter().sum::<f64>() - 2.0).abs() < 1e-12);
        assert!(weights.iter().all(|w| w.abs() < 1e-12));
        assert!(number(natural, "reconstruction_residual") < 1e-12);
        println!("H2 CAS(2,2) at {length} bohr: occupations {occupations:?}, signed spin weights {weights:?}, effectively unpaired electrons {unpaired:.12e}");
        effective.push(unpaired);
    }
    assert!(effective[0] < 0.2 && effective[1] > 1.95);
}

#[test]
fn relaxed_double_hybrid_density_gives_the_field_derivative_of_the_energy() {
    let steps = [0.004, 0.002, 0.001, 0.0005];
    for (label, xc) in [
        (
            "mp2",
            vec![
                ("xc", text("hf")),
                ("double_hybrid_model", text("custom")),
                ("pt2_same_spin", Value::Float(1.0)),
                ("pt2_opposite_spin", Value::Float(1.0)),
            ],
        ),
        (
            "b2plyp",
            vec![("xc", text("b2plyp")), ("grid_level", Value::Int(5))],
        ),
    ] {
        let mut base = xc.clone();
        base.push(("basis", text("sto-3g")));
        let mut request = base.clone();
        request.push(("nbo_density", text("pt2_relaxed")));
        let report = run("natural-bond", &system(&WATER), &controls(&request)).unwrap();
        let dipole = floats(&report, "density_dipole");
        let electrons = number(&report, "density_electron_count");
        let analytic = run("dipole", &system(&WATER), &controls(&base)).unwrap();
        let reference = floats(&analytic, "dipole");
        // The analytic dipole is about the centre of mass; the density
        // dipole about the origin. The molecule is neutral.
        let route = dipole
            .iter()
            .zip(&reference)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let energy = |h: f64| {
            let mut c = base.clone();
            c.push(("field", field(h)));
            number(
                &run("energy", &system(&WATER), &controls(&c)).unwrap(),
                "energy",
            )
        };
        let errors: Vec<f64> = steps
            .iter()
            .map(|&h| (-(energy(h) - energy(-h)) / (2.0 * h) - dipole[2]).abs())
            .collect();
        println!(
            "{label}: relaxed-density dipole {dipole:?}, analytic dipole {reference:?}, \
             difference {route:.3e}, electrons {electrons:.12}"
        );
        assert!(route < 1e-6);
        assert!((electrons - 10.0).abs() < 1e-9);
        fit(&format!("{label} relaxed-density dipole"), &steps, &errors);
    }
}

#[test]
fn casscf_density_gives_the_field_derivative_of_the_casscf_energy() {
    let lih = System::molecule(vec![3, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]]);
    let base = vec![
        ("xc", text("hf")),
        ("basis", text("sto-3g")),
        ("cas", Value::Array(vec![Value::Int(2), Value::Int(2)])),
        ("casscf_gradient_tol", Value::Float(1e-10)),
        ("ci_tol", Value::Float(1e-12)),
    ];
    let mut request = base.clone();
    request.push(("nbo_density", text("casscf")));
    let report = run("natural-bond", &lih, &controls(&request)).unwrap();
    let dipole = floats(&report, "density_dipole");
    let energy = |h: f64| {
        let mut c = base.clone();
        c.push(("field", field(h)));
        floats(&run("casscf", &lih, &controls(&c)).unwrap(), "cas_energies")[0]
    };
    let steps = [0.004, 0.002, 0.001, 0.0005];
    let errors: Vec<f64> = steps
        .iter()
        .map(|&h| (-(energy(h) - energy(-h)) / (2.0 * h) - dipole[2]).abs())
        .collect();
    let channel = &report.get("channels").and_then(Value::as_array).unwrap()[0];
    let bonds: Vec<String> = channel
        .get("nbos")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .filter(|b| b.get("lewis").and_then(Value::as_bool) == Some(true))
        .map(|b| b.get("label").and_then(Value::as_str).unwrap().to_string())
        .collect();
    println!(
        "LiH CASSCF density: dipole {dipole:?}, electrons {:.12}, Lewis {bonds:?}, non-Lewis {:.6}",
        number(&report, "density_electron_count"),
        number(channel, "non_lewis_occupancy")
    );
    assert!((number(&report, "density_electron_count") - 4.0).abs() < 1e-9);
    fit("CASSCF density dipole", &steps, &errors);
}

#[test]
fn born_von_karman_unfolding_is_the_gamma_supercell() {
    let chain = |cells: usize, mesh: i64| {
        let a = 4.0;
        let mut numbers = Vec::new();
        let mut positions = Vec::new();
        for c in 0..cells {
            numbers.extend([1, 1]);
            positions.push([a * c as f64, 0.0, 0.0]);
            positions.push([a * c as f64, 1.4, 0.0]);
        }
        let system = System {
            numbers,
            positions,
            cell: Some([[a * cells as f64, 0.0, 0.0], [0.0; 3], [0.0; 3]]),
            pbc: [true, false, false],
        };
        run(
            "periodic-natural-bond",
            &system,
            &controls(&[
                ("xc", text("svwn5")),
                ("basis", text("sto-3g")),
                (
                    "k_mesh",
                    Value::Array(vec![Value::Int(mesh), Value::Int(1), Value::Int(1)]),
                ),
            ]),
        )
        .unwrap()
    };
    let unfolded = chain(1, 3);
    let explicit = chain(3, 1);
    let q1 = floats(&unfolded, "natural_charges");
    let q3 = floats(&explicit, "natural_charges");
    let difference = q1
        .iter()
        .zip(&q3)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let channel = &unfolded.get("channels").and_then(Value::as_array).unwrap()[0];
    let labels: Vec<String> = channel
        .get("nbos")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .filter(|b| b.get("lewis").and_then(Value::as_bool) == Some(true))
        .map(|b| b.get("label").and_then(Value::as_str).unwrap().to_string())
        .collect();
    println!(
        "H2 chain: unfolded charges {q1:?}, explicit supercell {q3:?}, difference {difference:.3e}, \
         translation defect {:.3e}, imaginary {:.3e}, Lewis per cell {labels:?}",
        number(&unfolded, "translation_defect"),
        number(&unfolded, "unfolding_imaginary")
    );
    assert!(number(&unfolded, "translation_defect") < 1e-8);
    assert!(number(&unfolded, "unfolding_imaginary") < 1e-10);
    assert!(difference < 1e-6);
    assert_eq!(labels, vec!["BD(1) H1-H2".to_string()]);
    assert!((number(channel, "electrons_per_cell") - 2.0).abs() < 1e-8);
}

#[test]
fn kekule_structures_of_benzene_share_the_resonance_weight() {
    let mut atoms = Vec::new();
    for k in 0..6 {
        let angle = std::f64::consts::PI / 3.0 * k as f64;
        atoms.push((6u8, [2.64 * angle.cos(), 2.64 * angle.sin(), 0.0]));
    }
    for k in 0..6 {
        let angle = std::f64::consts::PI / 3.0 * k as f64;
        atoms.push((1u8, [4.68 * angle.cos(), 4.68 * angle.sin(), 0.0]));
    }
    let report = run(
        "natural-bond",
        &system(&atoms),
        &controls(&[
            ("xc", text("hf")),
            ("basis", text("sto-3g")),
            ("nbo_resonance", Value::Bool(true)),
        ]),
    )
    .unwrap();
    let channel = &report.get("channels").and_then(Value::as_array).unwrap()[0];
    let resonance = channel.get("resonance").unwrap();
    let structures = resonance
        .get("structures")
        .and_then(Value::as_array)
        .unwrap();
    let weights: Vec<f64> = structures.iter().map(|s| number(s, "weight")).collect();
    let orders = floats(resonance, "bond_orders");
    let cc = orders[1];
    println!(
        "benzene: {} structures enumerated, {} evaluated, weights {weights:?}, C1-C2 NRT bond \
         order {cc:.6}, fractional deviation {:.6} (reference {:.6}), optimality {:.3e}",
        number(resonance, "enumerated"),
        number(resonance, "evaluated"),
        number(resonance, "fractional_deviation"),
        number(resonance, "reference_deviation"),
        number(resonance, "optimality_residual")
    );
    // Structures whose prescribed orbitals are dependent are left out; the
    // first run built lone pairs before bonds and lost 110 of 117.
    println!(
        "benzene: skipped structures {}, degenerate weight directions {}",
        number(resonance, "skipped"),
        number(resonance, "degenerate_directions")
    );
    assert!(number(resonance, "skipped") <= 0.1 * number(resonance, "enumerated"));
    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    assert!(number(resonance, "optimality_residual") < 1e-8);
    assert!(
        number(resonance, "fractional_deviation")
            <= number(resonance, "reference_deviation") + 1e-12
    );
    // The two leading structures are the Kekule structures, equally weighted.
    assert!(weights.len() >= 2 && (weights[0] - weights[1]).abs() < 1e-6);
    assert!(weights[0] + weights[1] > 0.5);
    assert!((1.2..1.8).contains(&cc));
    for k in 0..6 {
        let next = (k + 1) % 6;
        assert!((orders[k * 12 + next] - cc).abs() < 1e-6);
    }
}

#[test]
fn shielding_and_coupling_decompositions_sum_to_their_totals() {
    let s = 1.186;
    let methane = [
        (6u8, [0.0, 0.0, 0.0]),
        (1, [s, s, s]),
        (1, [-s, -s, s]),
        (1, [-s, s, -s]),
        (1, [s, -s, -s]),
    ];
    let base = [("xc", text("hf")), ("basis", text("6-31g*"))];
    // Equivalent bonds have equivalent NBOs (the first run searched every
    // bond in the density depleted of the previous ones and found s
    // characters of 23.5 to 26.6 % on carbon).
    let nbo = run("natural-bond", &system(&methane), &controls(&base)).unwrap();
    let channel = &nbo.get("channels").and_then(Value::as_array).unwrap()[0];
    let bonds: Vec<(f64, f64)> = channel
        .get("nbos")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("BD"))
        .map(|b| {
            let carbon = &b.get("hybrids").and_then(Value::as_array).unwrap()[0];
            (number(b, "occupancy"), floats(carbon, "composition")[0])
        })
        .collect();
    println!("methane C-H bonds (occupancy, carbon s character %): {bonds:?}");
    assert_eq!(bonds.len(), 4);
    for (occupancy, s_character) in &bonds {
        assert!((occupancy - bonds[0].0).abs() < 1e-8);
        assert!((s_character - bonds[0].1).abs() < 1e-6);
    }
    let ncs = run("natural-shielding", &system(&methane), &controls(&base)).unwrap();
    let total = run("nmr-shielding", &system(&methane), &controls(&base)).unwrap();
    let isotropic = floats(&total, "isotropic_shieldings");
    let shieldings = ncs.get("shieldings").and_then(Value::as_array).unwrap();
    let carbon = &shieldings[0];
    let parts = carbon
        .get("contributions")
        .and_then(Value::as_array)
        .unwrap();
    let bonds: Vec<f64> = parts
        .iter()
        .filter(|p| {
            p.get("orbital")
                .and_then(Value::as_str)
                .unwrap()
                .starts_with("BD")
        })
        .map(|p| number(p, "isotropic"))
        .collect();
    let spread = bonds
        .iter()
        .fold(0.0_f64, |m, b| m.max((b - bonds[0]).abs()));
    println!(
        "methane carbon shielding {:.6} ppm (nmr-shielding {:.6}); C-H bond contributions \
         {bonds:?} (spread {spread:.3e}); decomposition residual {:.3e}",
        number(carbon, "isotropic"),
        isotropic[0],
        number(&ncs, "decomposition_residual")
    );
    assert!((number(carbon, "isotropic") - isotropic[0]).abs() < 1e-8);
    assert!(number(&ncs, "decomposition_residual") < 1e-8);
    assert_eq!(bonds.len(), 4);
    assert!(spread < 1e-5);

    let mut request = base.to_vec();
    request.push(("nuclei", Value::Array(vec![Value::Int(0), Value::Int(1)])));
    let njc = run("natural-coupling", &system(&methane), &controls(&request)).unwrap();
    let reference = run(
        "spin-spin-couplings",
        &system(&methane),
        &controls(&request),
    )
    .unwrap();
    let pair = &njc.get("couplings").and_then(Value::as_array).unwrap()[0];
    let k = number(pair, "reduced_isotropic");
    let k_reference = floats(&reference, "reduced_isotropic")[0];
    let contributions = pair.get("contributions").and_then(Value::as_array).unwrap();
    let top = &contributions[0];
    println!(
        "methane C-H reduced coupling {k:.10} (spin-spin-couplings {k_reference:.10}), largest \
         contribution {} {:.10}, decomposition residual {:.3e}",
        top.get("orbital").and_then(Value::as_str).unwrap(),
        number(top, "reduced_isotropic"),
        number(&njc, "decomposition_residual")
    );
    assert!((k - k_reference).abs() < 1e-10 * k.abs().max(1.0));
    assert!(number(&njc, "decomposition_residual") < 1e-8 * k.abs().max(1e-3));
    assert_eq!(
        top.get("orbital").and_then(Value::as_str),
        Some("BD(1) C1-H2")
    );
}

#[test]
fn energy_decomposition_identity_limits_and_counterpoise_interaction() {
    let dimer = |distance: f64| -> Vec<(u8, [f64; 3])> {
        vec![
            (8, [0.0, 0.0, 0.0]),
            (1, [1.81, 0.0, 0.0]),
            (1, [-0.45, 1.75, 0.0]),
            (8, [distance, 0.0, 0.0]),
            (1, [distance + 0.6, 0.0, 1.71]),
            (1, [distance + 0.6, 0.0, -1.71]),
        ]
    };
    let request = [
        ("xc", text("hf")),
        ("basis", text("6-31g*")),
        (
            "fragments",
            Value::Array([0, 0, 0, 1, 1, 1].iter().map(|&f| Value::Int(f)).collect()),
        ),
    ];
    let near = run(
        "natural-energy-decomposition",
        &system(&dimer(5.5)),
        &controls(&request),
    )
    .unwrap();
    let far = run(
        "natural-energy-decomposition",
        &system(&dimer(40.0)),
        &controls(&request),
    )
    .unwrap();
    let farther = run(
        "natural-energy-decomposition",
        &system(&dimer(80.0)),
        &controls(&request),
    )
    .unwrap();
    let farthest = run(
        "natural-energy-decomposition",
        &system(&dimer(160.0)),
        &controls(&request),
    )
    .unwrap();
    let terms = [
        "charge_transfer",
        "electrostatic",
        "exchange_correlation",
        "deformation",
    ];
    let print = |name: &str, r: &Value| {
        println!(
            "{name}: interaction {:.8}, terms {:?}, identity {:.3e}, functional {:.3e}, \
             idempotency {:.3e}",
            number(r, "interaction_energy"),
            terms.iter().map(|t| number(r, t)).collect::<Vec<_>>(),
            number(r, "decomposition_residual"),
            number(r, "energy_functional_residual"),
            number(r, "localized_idempotency_residual")
        )
    };
    print("water dimer at 5.5 Bohr", &near);
    print("water dimer at 40 Bohr", &far);
    for r in [&near, &far] {
        assert!(number(r, "decomposition_residual") < 1e-9);
        assert!(number(r, "energy_functional_residual") < 1e-8);
        assert!(number(r, "localized_idempotency_residual") < 1e-8);
    }
    assert!(number(&near, "charge_transfer") < 0.0);
    assert!(number(&near, "electrostatic") < 0.0);
    // Separated fragments: no charge transfer, exchange or deformation; the
    // electrostatic energy tends to the dipole-dipole energy, R^-3 (the first
    // run asserted it to vanish at 40 Bohr and found the -1.1e-5 Hartree of
    // two water dipoles; the second asserted the ratio of 40 to 80 Bohr to
    // be 8 within 0.1 and found 8.713, the dipole-quadrupole terms of order
    // 1/R): the local exponents of successive doublings approach 3.
    for t in ["charge_transfer", "exchange_correlation", "deformation"] {
        assert!(
            number(&far, t).abs() < 1e-7,
            "{t} at 40 Bohr: {}",
            number(&far, t)
        );
    }
    let first = (number(&far, "electrostatic") / number(&farther, "electrostatic")).log2();
    let second = (number(&farther, "electrostatic") / number(&farthest, "electrostatic")).log2();
    println!(
        "electrostatic energy: local exponents {first:.6} (40-80 Bohr), {second:.6} (80-160 Bohr)"
    );
    assert!(second < first && (second - 3.0).abs() < 0.08);
    // Counterpoise interaction of the independent SAPT supermolecular route.
    use dft_rs::basis::data;
    use dft_rs::sapt::sapt_interaction;
    use dft_rs::{resolve, Atom, FunctionalSpec, Molecule, ScfOptions, Vec3};
    let atoms = dimer(5.5);
    let monomer = |range: std::ops::Range<usize>| {
        Molecule::new(
            atoms[range]
                .iter()
                .map(|(z, r)| Atom {
                    z: *z,
                    position: Vec3::new(r[0], r[1], r[2]),
                })
                .collect(),
        )
    };
    let set = data::load("6-31g*").unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let options = ScfOptions {
        e_tol: 1e-12,
        ..ScfOptions::default()
    };
    let sapt = sapt_interaction(
        &monomer(0..3),
        &monomer(3..6),
        &set,
        &functional,
        &options,
        None,
    )
    .unwrap();
    let difference = (sapt.e_int_supermolecular - number(&near, "interaction_energy")).abs();
    println!(
        "counterpoise interaction: SAPT supermolecular {:.10}, decomposition {:.10}, difference {difference:.3e}",
        sapt.e_int_supermolecular,
        number(&near, "interaction_energy")
    );
    assert!(difference < 1e-7);
}
