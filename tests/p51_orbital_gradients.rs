// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::response::{frontier_gap_gradient, orbital_energy_gradients, OrbitalGradientOptions};
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

fn molecule(r: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::new(0.1, -0.2, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1, -0.2, r),
        },
    ])
    .with_charge(1)
}
fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-12,
        max_iter: 500,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..Default::default()
    }
}
fn grid(mol: &Molecule, f: &ResolvedFunctional) -> Option<MolecularGrid> {
    (!f.parts.is_empty()).then(|| {
        MolecularGrid::build(
            mol,
            &GridOptions {
                level: 2,
                prune: false,
                weight_cutoff: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
    })
}
fn energies(r: f64, f: &ResolvedFunctional) -> Vec<f64> {
    let mol = molecule(r);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let g = grid(&mol, f);
    let scf = run_scf(&mol, &mb, &options(), f, g.as_ref()).unwrap();
    assert!(scf.converged);
    scf.mo_energy[0].clone()
}
fn verify_fit(name: &str, kind: &str, steps: &[f64], estimates: &[f64], exact: f64) {
    let errors: Vec<_> = estimates.iter().map(|v| (v - exact).abs()).collect();
    assert!(errors.iter().all(|v| v.is_finite() && *v > 0.0));
    let x: Vec<_> = steps.iter().map(|v| v.ln()).collect();
    let y: Vec<_> = errors.iter().map(|v| v.ln()).collect();
    let xm = x.iter().sum::<f64>() / 4.0;
    let ym = y.iter().sum::<f64>() / 4.0;
    let p = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - xm) * (b - ym))
        .sum::<f64>()
        / x.iter().map(|v| (v - xm).powi(2)).sum::<f64>();
    let r2 = 1.0
        - x.iter()
            .zip(&y)
            .map(|(a, b)| (b - ym - p * (a - xm)).powi(2))
            .sum::<f64>()
            / y.iter().map(|v| (v - ym).powi(2)).sum::<f64>();
    eprintln!("{name} {kind} errors {errors:?}; h=0.02..0.0025 Bohr p={p:.12} R2={r2:.12}");
    assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
    for pair in errors.windows(2) {
        assert!((pair[0] / pair[1] / 4.0 - 1.0).abs() < 0.03);
    }
    assert!(((4.0 * estimates[3] - estimates[2]) / 3.0 - exact).abs() < 1e-8);
}

#[test]
fn relaxed_orbitals_and_frontier_gaps_have_generic_four_step_scf_oracles() {
    let steps = [0.02, 0.01, 0.005, 0.0025];
    for name in [
        "hf",
        "svwn5",
        "pbe",
        "b3lyp",
        "cam-b3lyp",
        "r2scan",
        "custom-provider",
    ] {
        let mut f = resolve(&FunctionalSpec::Combined(
            if name == "custom-provider" {
                "pbe"
            } else {
                name
            }
            .into(),
        ))
        .unwrap();
        if name == "custom-provider" {
            f.description = "Unregistered weighted exchange/correlation provider".into();
            for part in &mut f.parts {
                part.0 *= 0.93;
            }
            f.hybrid.cx = 0.11;
        }
        let mol = molecule(1.5);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let g = grid(&mol, &f);
        let opts = options();
        let scf = run_scf(&mol, &mb, &opts, &f, g.as_ref()).unwrap();
        let request = OrbitalGradientOptions {
            coordinates: Some(vec![5]),
            ..Default::default()
        };
        let report = frontier_gap_gradient(
            &mol,
            &mb,
            &f,
            g.as_ref(),
            &scf,
            &opts,
            &request,
            &Default::default(),
        )
        .unwrap();
        assert!(report.converged, "{:?}", report.warnings);
        let estimates: Vec<Vec<_>> = steps
            .iter()
            .map(|&h| {
                energies(1.5 + h, &f)
                    .iter()
                    .zip(energies(1.5 - h, &f))
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect()
            })
            .collect();
        for orbital in 0..2 {
            let oracle: Vec<_> = estimates.iter().map(|e| e[orbital]).collect();
            verify_fit(
                name,
                &format!("orbital-{orbital}"),
                &steps,
                &oracle,
                report.value.orbitals.gradients[0][orbital]
                    .as_ref()
                    .unwrap()[0],
            );
        }
        let gap: Vec<_> = estimates.iter().map(|e| e[1] - e[0]).collect();
        verify_fit(
            name,
            "frontier-gap",
            &steps,
            &gap,
            report.value.gaps[0]
                .as_ref()
                .unwrap()
                .gradient
                .as_ref()
                .unwrap()[0],
        );
        for error in &report.value.orbitals.reference_fock_errors {
            assert!(*error < 1e-10);
        }
    }
}

#[test]
fn orbital_gradients_preserve_translation_order_and_cluster_contracts() {
    let mol = molecule(1.5);
    let f = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &f, None).unwrap();
    let report = orbital_energy_gradients(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert!(report.converged);
    assert!(report.value.translation_residual.unwrap() < 1e-10);
    let reordered = OrbitalGradientOptions {
        coordinates: Some(vec![5, 2]),
        ..Default::default()
    };
    let other = orbital_energy_gradients(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        &reordered,
        &Default::default(),
    )
    .unwrap();
    for s in 0..2 {
        for i in 0..2 {
            let a = report.value.gradients[s][i].as_ref().unwrap();
            let b = other.value.gradients[s][i].as_ref().unwrap();
            assert!((a[5] - b[0]).abs() < 1e-12 && (a[2] - b[1]).abs() < 1e-12);
        }
    }
    assert!(other.value.translation_residual.is_none());
    for coordinates in [vec![], vec![6], vec![5, 5]] {
        assert!(orbital_energy_gradients(
            &mol,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            &OrbitalGradientOptions {
                coordinates: Some(coordinates),
                ..Default::default()
            },
            &Default::default()
        )
        .is_err());
    }
    // Grouping close levels returns the full block and preserves their
    // actual reference splitting rather than inventing individual slopes.
    let grouped = OrbitalGradientOptions {
        coordinates: Some(vec![5]),
        degeneracy_tolerance: 2.0,
    };
    let block = orbital_energy_gradients(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &opts,
        &grouped,
        &Default::default(),
    )
    .unwrap();
    assert!(block.value.gradients.iter().flatten().all(|v| v.is_none()));
    assert_eq!(block.value.clusters.len(), 2);
    let cluster = &block.value.clusters[0];
    let at_zero = cluster.linearized_energies(&[1.0], 0.0).unwrap();
    for (a, b) in at_zero.iter().zip(&scf.mo_energy[0]) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn exact_degenerate_directional_slopes_have_branch_tracked_dense_oracles() {
    let mut derivative = Matrix::zeros(2, 2);
    derivative[(0, 0)] = 0.8;
    derivative[(1, 1)] = -0.8;
    derivative[(0, 1)] = 0.6;
    derivative[(1, 0)] = 0.6;
    let cluster = dft_rs::response::OrbitalCluster {
        spin: 0,
        orbitals: vec![0, 1],
        energies: vec![0.4; 2],
        energy_spread: 0.0,
        derivative_matrices: vec![derivative],
        commutator_defect: 0.0,
    };
    let slopes = cluster.directional_slopes(&[1.0]).unwrap();
    assert!((slopes[0] + 1.0).abs() < 1e-14 && (slopes[1] - 1.0).abs() < 1e-14);
    let dense = |t: f64| {
        let mut h = Matrix::zeros(3, 3);
        h[(0, 0)] = 0.4 + 0.8 * t;
        h[(1, 1)] = 0.4 - 0.8 * t;
        h[(2, 2)] = 4.0 + 0.2 * t;
        h[(0, 1)] = 0.6 * t;
        h[(1, 0)] = 0.6 * t;
        h[(0, 2)] = 0.3 * t;
        h[(2, 0)] = 0.3 * t;
        h[(1, 2)] = 0.7 * t;
        h[(2, 1)] = 0.7 * t;
        symmetric_eigen(&h).unwrap().0
    };
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let estimates: Vec<_> = steps
        .iter()
        .map(|&h| {
            let plus = dense(h);
            let minus = dense(-h);
            // For this single direction, the two degenerate branches exchange
            // their sorted energy indices across zero. Matching equal sorted
            // indices would incorrectly produce zero slopes.
            [
                (plus[0] - minus[1]) / (2.0 * h),
                (plus[1] - minus[0]) / (2.0 * h),
            ]
        })
        .collect();
    for branch in 0..2 {
        verify_fit(
            "dense-degenerate",
            &format!("directional-branch-{branch}"),
            &steps,
            &estimates.iter().map(|v| v[branch]).collect::<Vec<_>>(),
            slopes[branch],
        );
    }
}
