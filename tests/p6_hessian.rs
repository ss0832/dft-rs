// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 6 gates: numerical Hessian → harmonic frequencies vs the frozen
//! PySCF analytic-Hessian oracle at PySCF-optimized geometries.

use dft_rs::basis::data;
use dft_rs::dispersion::Dispersion;
use dft_rs::grid::GridOptions;
use dft_rs::hessian::numerical_hessian;
use dft_rs::math::Vec3;
use dft_rs::properties::vibrational::vibrational_analysis;
use dft_rs::scf::{EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[test]
fn frequencies_match_pyscf_oracle() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_data");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with("_freq.csv") {
            continue;
        }
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let mut basis = String::new();
        let mut functional = String::new();
        let mut atoms = Vec::new();
        let mut freqs_ref: Vec<f64> = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p: Vec<&str> = line.split(',').collect();
            match p[0] {
                "basis" => basis = p[1].into(),
                "functional" => functional = p[1].into(),
                "charge" | "mult" => {}
                "atom" => atoms.push(Atom {
                    z: p[1].parse::<f64>().unwrap() as u8,
                    position: Vec3::new(
                        p[2].parse().unwrap(),
                        p[3].parse().unwrap(),
                        p[4].parse().unwrap(),
                    ),
                }),
                "freq" => freqs_ref.push(p[1].parse().unwrap()),
                other => panic!("unknown key {other}"),
            }
        }
        let mol = Molecule::new(atoms);
        let set = data::load(&basis).unwrap();
        let func = resolve(&FunctionalSpec::Combined(functional.clone())).unwrap();
        let opts = ScfOptions {
            e_tol: 1e-11,
            p_tol: 1e-9,
            max_iter: 200,
            eri_mode: EriMode::InCore { limit_mb: 512 },
            ..ScfOptions::default()
        };
        let (grid_opts, tol) = if func.parts.is_empty() {
            (None, 3.0)
        } else {
            (
                Some(GridOptions {
                    level: 7,
                    ..GridOptions::default()
                }),
                12.0,
            )
        };
        let res = numerical_hessian(
            &mol,
            &set,
            &opts,
            &func,
            &functional,
            grid_opts.as_ref(),
            &Dispersion::None,
            5e-3,
        )
        .unwrap();
        // Symmetry is enforced; verify the mass-weighted analysis output.
        let vib = vibrational_analysis(&mol, &res.hessian, Some(&res.dipole_derivatives)).unwrap();
        assert_eq!(vib.n_projected, 6, "{fname}: expected 6 TR modes");
        assert_eq!(
            vib.frequencies_cm.len(),
            freqs_ref.len(),
            "{fname}: mode count"
        );
        assert_eq!(vib.n_imaginary, 0, "{fname}: spurious imaginary modes");
        for (ours, want) in vib.frequencies_cm.iter().zip(&freqs_ref) {
            assert!(
                (ours - want).abs() < tol,
                "{fname}: {ours:.2} vs oracle {want:.2} cm⁻¹"
            );
        }
        // IR intensities exist and are non-negative; the bend of water is
        // IR active.
        assert!(vib.ir_intensities.iter().all(|&i| i >= 0.0));
        assert!(
            vib.ir_intensities.iter().any(|&i| i > 1.0),
            "{fname}: all IR-dark?"
        );
        checked += 1;
    }
    assert!(checked >= 2, "expected 2+ frequency oracle cases");
}
