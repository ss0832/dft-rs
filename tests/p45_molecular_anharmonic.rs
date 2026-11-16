// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::hessian::analytic_hessian_full;
use dft_rs::math::Vec3;
use dft_rs::response::anharmonic::molecular::molecular_anharmonic;
use dft_rs::response::polarizability::polarizability;
use dft_rs::scf::{run_scf, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::process::Command;

fn molecule(r: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, r),
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
fn setup(
    r: f64,
    name: &str,
) -> (
    Molecule,
    MolBasis,
    dft_rs::xc::ResolvedFunctional,
    Option<MolecularGrid>,
    dft_rs::scf::ScfResult,
) {
    let mol = molecule(r);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let grid = (!functional.parts.is_empty()).then(|| {
        MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 1,
                prune: false,
                weight_cutoff: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
    });
    let scf = run_scf(&mol, &mb, &options(), &functional, grid.as_ref()).unwrap();
    (mol, mb, functional, grid, scf)
}
fn equilibrium(name: &str) -> f64 {
    let mut r = 1.5;
    for _ in 0..15 {
        let (mol, mb, f, grid, scf) = setup(r, name);
        let g = dft_rs::gradient::assemble_gradient(&mol, &mb, &f, grid.as_ref(), &scf, &options())
            .unwrap()[1]
            .z;
        if g.abs() < 1e-10 {
            return r;
        }
        let h = analytic_hessian_full(&mol, &mb, &options(), &f, &scf, grid.as_ref()).unwrap();
        r -= (g / h.hessian[(5, 5)]).clamp(-0.15, 0.15);
    }
    panic!("stationary fixture failed to converge for {name}");
}
fn properties(r: f64, name: &str) -> [f64; 2] {
    let (mol, mb, f, grid, scf) = setup(r, name);
    [
        dft_rs::properties::dipole_moment(&mol, &mb, &scf).z,
        polarizability(&mol, &mb, &f, grid.as_ref(), &scf, &options()).unwrap()[2][2],
    ]
}
#[test]
fn molecular_property_jets_through_fifth_energy_order_match_displaced_independent_cpks() {
    #[path = "support/convergence.rs"]
    mod convergence;
    let name = "hf";
    let r = equilibrium(name);
    eprintln!("stationary {name} bond {r:.16} Bohr");
    let (mol, mb, f, grid, scf) = setup(r, name);
    let result = molecular_anharmonic(
        &mol,
        &mb,
        &f,
        grid.as_ref(),
        &scf,
        &options(),
        None,
        true,
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert!(result.converged, "{:?}", result.warnings);
    assert!(result.value.maximum_force < 1e-9);
    assert_eq!(result.value.n_projected, 5);
    assert_eq!(result.value.selected_mode_ids, vec![0]);
    assert_eq!(result.value.force_permutation_comparisons, [0, 0]);
    assert_eq!(result.value.polarizability_symmetry_comparisons, 9);
    assert!(result.value.polarizability_symmetry_defect.unwrap() < 1e-10);
    assert_eq!(result.value.spectrum.bands.len(), 2);
    assert!(
        result.value.spectrum.bands[0].frequency_cm
            < result.value.spectrum.bands[0].harmonic_frequency_cm
    );
    assert!(result
        .value
        .spectrum
        .bands
        .iter()
        .all(|b| b.ir_intensity > 0.0 && b.raman_activity.unwrap() > 0.0));
    let path = &result.value.patterns[0];
    let stretch = path[1][2] - path[0][2];
    let mut previous: Option<[f64; 2]> = None;
    let analytic = [
        result.value.polynomial.dipole.third[2],
        result
            .value
            .polynomial
            .polarizability
            .as_ref()
            .unwrap()
            .third[8],
    ];
    let mut extrapolated = [0.0; 2];
    // At 0.0025 Bohr the third-dipole stencil has cancellation noise (the
    // measured h-halving ratio was 4.235679). Extend the coarse end instead,
    // retaining all three original steps and their unchanged strict bounds.
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut errors = [Vec::new(), Vec::new()];
    for h in steps {
        let far = properties(r + 2.0 * h, name);
        let plus = properties(r + h, name);
        let minus = properties(r - h, name);
        let back = properties(r - 2.0 * h, name);
        let values = std::array::from_fn::<_, 2, _>(|a| {
            (far[a] - 2.0 * plus[a] + 2.0 * minus[a] - back[a]) / (2.0 * h * h * h)
                * stretch.powi(3)
        });
        for a in 0..2 {
            errors[a].push((values[a] - analytic[a]).abs());
        }
        if let Some(old) = previous {
            for a in 0..2 {
                extrapolated[a] = (4.0 * values[a] - old[a]) / 3.0;
                let ratio = (old[a] - analytic[a]).abs() / (values[a] - analytic[a]).abs();
                eprintln!("property {a} h-halving error ratio {ratio:.10}");
                assert!((ratio - 4.0).abs() < 0.03);
            }
        }
        previous = Some(values);
        eprintln!("property third derivatives h {h}: {values:?}");
    }
    for (a, errors) in errors.iter().enumerate() {
        let fit = convergence::error_power_fit(&steps, errors);
        eprintln!("property {a} four-step error fit: p={:.12}, C={:.12e}, R_squared={:.12}, log_rmse={:.12e}",
            fit.exponent, fit.prefactor, fit.r_squared, fit.log_rmse);
        assert!((fit.exponent - 2.0).abs() < 0.02);
        assert!(fit.r_squared > 0.99999);
    }
    let mu = result.value.polynomial.dipole.third[2];
    let alpha = result
        .value
        .polynomial
        .polarizability
        .as_ref()
        .unwrap()
        .third[8];
    eprintln!("analytic mu3 {mu:.12e}, alpha3 {alpha:.12e}; extrapolated {extrapolated:?}");
    assert!((mu - extrapolated[0]).abs() < 1e-10 + mu.abs() * 2e-5);
    assert!((alpha - extrapolated[1]).abs() < 1e-9 + alpha.abs() * 2e-5);
    assert!(result.model.contains("third nuclear derivatives"));
    assert!(result.model.contains("Coriolis"));
    assert_eq!(
        result
            .value
            .polynomial
            .rotation
            .as_ref()
            .unwrap()
            .watson_pseudopotential,
        0.0
    );
    let path =
        std::env::temp_dir().join(format!("dft_anharmonic_routes_{}.xyz", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    std::fs::write(
        &path,
        format!(
            "2\nstationary anharmonic route validation\nHe 0 0 0\nH 0 0 {:.16}\n",
            r * dft_rs::constants::BOHR_TO_ANGSTROM
        ),
    )
    .unwrap();
    for executable in [
        env!("CARGO_BIN_EXE_dft-rs"),
        env!("CARGO_BIN_EXE_dft_rs_cli"),
    ] {
        let run = Command::new(executable)
            .arg("anharmonic")
            .arg(&path)
            .args([
                "--charge",
                "1",
                "--xc",
                "hf",
                "--anharmonic-raman",
                "--vibrational-modes",
                "0",
                "--conv-tol",
                "1e-13",
                "--conv-tol-dp",
                "1e-12",
                "--conv-tol-grad",
                "1e-12",
                "--log-mode",
                "none",
            ])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let stdout = String::from_utf8(run.stdout).unwrap();
        let bands: Vec<_> = stdout
            .lines()
            .filter(|l| l.starts_with("anharmonic_band "))
            .collect();
        assert_eq!(bands.len(), 2);
        for (line, expected) in bands.iter().zip(&result.value.spectrum.bands) {
            let columns: Vec<_> = line.split_whitespace().collect();
            assert!((columns[3].parse::<f64>().unwrap() - expected.frequency_cm).abs() < 1e-6);
            assert!((columns[5].parse::<f64>().unwrap() - expected.ir_intensity).abs() < 1e-7);
        }
        assert!(stdout.contains("Coriolis") && stdout.contains("IR km/mol"));
    }
}

#[test]
fn semilocal_dft_uses_the_same_stationary_anharmonic_driver_and_invalid_references_fail() {
    let name = "pbe";
    let r = equilibrium(name);
    eprintln!("stationary {name} bond {r:.16} Bohr");
    let (mol, mb, f, grid, scf) = setup(r, name);
    let result = molecular_anharmonic(
        &mol,
        &mb,
        &f,
        grid.as_ref(),
        &scf,
        &options(),
        None,
        false,
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert!(result.converged, "{:?}", result.warnings);
    assert!(result.value.spectrum.bands[0].ir_intensity > 0.0);
    assert!(result.value.polynomial.polarizability.is_none());
    assert!(molecular_anharmonic(
        &mol,
        &mb,
        &f,
        grid.as_ref(),
        &scf,
        &options(),
        Some(&[1]),
        false,
        &Default::default(),
        &Default::default()
    )
    .is_err());
    let (mol, mb, f, grid, scf) = setup(1.5, name);
    assert!(molecular_anharmonic(
        &mol,
        &mb,
        &f,
        grid.as_ref(),
        &scf,
        &options(),
        None,
        false,
        &Default::default(),
        &Default::default()
    )
    .is_err());
}

#[test]
fn nonlinear_molecular_driver_includes_nonzero_rotational_energy_and_transition_corrections() {
    let triangle = |r: f64| {
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(r, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.5 * r, 0.5 * 3.0_f64.sqrt() * r, 0.0),
            },
        ])
        .with_charge(1)
    };
    let direction = [
        Vec3::zero(),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.5, 0.5 * 3.0_f64.sqrt(), 0.0),
    ];
    let f = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mut r = 1.8;
    for _ in 0..15 {
        let mol = triangle(r);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let scf = run_scf(&mol, &mb, &options(), &f, None).unwrap();
        let gradient =
            dft_rs::gradient::assemble_gradient(&mol, &mb, &f, None, &scf, &options()).unwrap();
        let g = gradient
            .iter()
            .zip(direction)
            .map(|(g, d)| g.x * d.x + g.y * d.y + g.z * d.z)
            .sum::<f64>();
        if g.abs() < 1e-10 {
            break;
        }
        let h = analytic_hessian_full(&mol, &mb, &options(), &f, &scf, None).unwrap();
        let mut curvature = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                for axis in 0..3 {
                    for other in 0..3 {
                        let p = [direction[a].x, direction[a].y, direction[a].z];
                        let q = [direction[b].x, direction[b].y, direction[b].z];
                        curvature += p[axis] * h.hessian[(3 * a + axis, 3 * b + other)] * q[other];
                    }
                }
            }
        }
        r -= (g / curvature).clamp(-0.15, 0.15);
    }
    eprintln!("stationary nonlinear bond {r:.16} Bohr");
    let mol = triangle(r);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let scf = run_scf(&mol, &mb, &options(), &f, None).unwrap();
    let report = molecular_anharmonic(
        &mol,
        &mb,
        &f,
        None,
        &scf,
        &options(),
        Some(&[0, 1]),
        false,
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    assert!(report.converged, "{:?}", report.warnings);
    let rotation = report.value.polynomial.rotation.as_ref().unwrap();
    assert_eq!(
        rotation.class,
        dft_rs::response::anharmonic::rotation::RotorClass::Nonlinear
    );
    assert!(rotation.watson_pseudopotential < 0.0);
    assert!(rotation.coriolis.iter().flatten().any(|v| v.abs() > 0.1));
    let mut rectilinear = report.value.polynomial.clone();
    rectilinear.rotation = None;
    let bare =
        dft_rs::response::anharmonic::vibrational_spectrum(&rectilinear, &Default::default())
            .unwrap();
    assert!((bare.zero_point_energy - report.value.spectrum.zero_point_energy).abs() > 1e-7);
    assert!(bare
        .bands
        .iter()
        .zip(&report.value.spectrum.bands)
        .any(|(a, b)| (a.frequency_cm - b.frequency_cm).abs() > 0.01));
    assert!(bare
        .bands
        .iter()
        .zip(&report.value.spectrum.bands)
        .any(|(a, b)| a
            .transition_dipole
            .iter()
            .zip(b.transition_dipole)
            .any(|(x, y)| (x - y).abs() > 1e-8)));
}
