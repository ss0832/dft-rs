// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin magnetic response: independent generalized spectral projectors,
//! unrestricted gauge cancellation, dense eigenproblems and four-step fits.
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::properties::magnetic::{magnetic_density_response, molecular_nmr_response, Gauge};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::{run_scf, EriMode, Reference, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

fn molecule(open: bool) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.16, -0.24, 0.31),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.41, 0.38, -0.62),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.72, 1.22, 0.47),
        },
    ])
    .with_charge(if open { 1 } else { 0 })
    .with_multiplicity(if open { 2 } else { 1 })
}
fn options() -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        conv_tol_grad: 1e-10,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        ..Default::default()
    }
}
fn grid(m: &Molecule) -> MolecularGrid {
    MolecularGrid::build(
        m,
        &GridOptions {
            level: 2,
            ..Default::default()
        },
    )
    .unwrap()
}
fn add(a: &mut Matrix, b: &Matrix, w: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += w * y;
    }
}
// Real representation of a complex Hermitian matrix, preserving both
// independent real components. No production response construction is used.
fn real_representation(real: &Matrix, imag: &Matrix) -> Matrix {
    let n = real.rows;
    let mut block = Matrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            block[(i, j)] = real[(i, j)];
            block[(n + i, n + j)] = real[(i, j)];
            block[(n + i, j)] = imag[(i, j)];
            block[(i, n + j)] = -imag[(i, j)];
        }
    }
    block
}
fn spectral_projector(
    f: &Matrix,
    s: &Matrix,
    fh: &Matrix,
    sh: &Matrix,
    h: f64,
    no: usize,
) -> Matrix {
    let mut imf = fh.clone();
    let mut ims = sh.clone();
    for v in imf.as_mut_slice() {
        *v *= h;
    }
    for v in ims.as_mut_slice() {
        *v *= h;
    }
    let fr = real_representation(f, &imf);
    let sr = real_representation(s, &ims);
    let (se, sv) = symmetric_eigen(&sr).unwrap();
    assert!(se.iter().all(|e| *e > 1e-8));
    let mut x = sv.clone();
    for j in 0..x.cols {
        for i in 0..x.rows {
            x[(i, j)] /= se[j].sqrt();
        }
    }
    let x = x.matmul_seq(&sv.transpose());
    let orth = x.transpose_matmul_seq(&fr.matmul_seq(&x));
    let (_, v) = symmetric_eigen(&orth).unwrap();
    let c = x.matmul_seq(&v);
    let n = f.rows;
    let mut p = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] = (0..2 * no).map(|a| c[(n + i, a)] * c[(j, a)]).sum();
        }
    }
    p
}
fn fit(label: &str, errors: &[f64; 4]) {
    let hs = [0.01_f64, 0.005, 0.0025, 0.00125];
    assert!(errors.iter().all(|v| v.is_finite() && *v > 0.0));
    let x = hs.map(f64::ln);
    let y = errors.map(f64::ln);
    let xm = x.iter().sum::<f64>() / 4.;
    let ym = y.iter().sum::<f64>() / 4.;
    let p = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - xm) * (y - ym))
        .sum::<f64>()
        / x.iter().map(|x| (x - xm).powi(2)).sum::<f64>();
    let b = ym - p * xm;
    let r2 = 1.
        - x.iter()
            .zip(y)
            .map(|(x, y)| (y - b - p * x).powi(2))
            .sum::<f64>()
            / y.iter().map(|y| (y - ym).powi(2)).sum::<f64>();
    println!(
        "{label}: errors={errors:?}, reduced field steps 0.01..0.00125, p={p:0.12}, R2={r2:0.12}"
    );
    assert!(
        (p - 2.).abs() < 0.02 && r2 > 0.99999,
        "{label}: p={p} R2={r2}"
    );
    for pair in errors.windows(2) {
        assert!((pair[0] / pair[1] / 4. - 1.).abs() < 0.03);
    }
}

#[test]
fn spin_density_derivatives_match_independent_spectral_projector_four_step_fits() {
    for open in [false, true] {
        let mol = molecule(open);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let grid = grid(&mol);
        let opts = options();
        for name in ["slater", "svwn5", "pbe"] {
            let spec = if name == "slater" {
                FunctionalSpec::Split {
                    x: name.into(),
                    c: "none".into(),
                }
            } else {
                FunctionalSpec::Combined(name.into())
            };
            let func = resolve(&spec).unwrap();
            let scf = run_scf(&mol, &mb, &opts, &func, Some(&grid)).unwrap();
            assert!(scf.converged);
            for gauge in [Gauge::Giao, Gauge::CommonOrigin] {
                let report = magnetic_density_response(
                    &mol,
                    &mb,
                    &func,
                    Some(&grid),
                    &scf,
                    &opts,
                    gauge,
                    Vec3::new(0.43, -0.36, 0.29),
                    &ResponseSolverConfig::default(),
                )
                .unwrap();
                assert!(report.residuals.iter().all(|r| *r < 1e-10));
                assert!(
                    report.value.projector_defect < 1e-9
                        && report.value.antisymmetry_defect < 1e-12
                );
                let w = [0.4, -0.3, 0.2];
                let n = mb.nao_sph;
                let mut fh = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
                let mut exact = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
                let mut sh = Matrix::zeros(n, n);
                for a in 0..3 {
                    add(&mut sh, &report.value.overlap_hat[a], w[a]);
                    for spin in 0..2 {
                        add(
                            &mut fh[spin],
                            &report.value.explicit_fock_hat[spin][a],
                            w[a],
                        );
                        add(&mut exact[spin], &report.value.density_hat[spin][a], w[a]);
                    }
                }
                let mut errors = [0.0_f64; 4];
                let mut differences = Vec::new();
                for (k, h) in [0.01, 0.005, 0.0025, 0.00125].into_iter().enumerate() {
                    let mut values = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
                    for spin in 0..2 {
                        let p = spectral_projector(
                            &scf.fock[spin],
                            &scf.overlap,
                            &fh[spin],
                            &sh,
                            h,
                            report.value.occupied[spin],
                        );
                        let m = spectral_projector(
                            &scf.fock[spin],
                            &scf.overlap,
                            &fh[spin],
                            &sh,
                            -h,
                            report.value.occupied[spin],
                        );
                        add(&mut values[spin], &p, 1. / (2. * h));
                        add(&mut values[spin], &m, -1. / (2. * h));
                        errors[k] = errors[k].max(values[spin].max_abs_diff(&exact[spin]));
                    }
                    differences.push(values);
                }
                fit(
                    &format!("{name} open={open} {gauge:?} spin spectral projector"),
                    &errors,
                );
                for spin in 0..2 {
                    let mut rich = differences[3][spin].clone();
                    for v in rich.as_mut_slice() {
                        *v *= 4. / 3.;
                    }
                    add(&mut rich, &differences[2][spin], -1. / 3.);
                    assert!(rich.max_abs_diff(&exact[spin]) < 1e-8);
                }
            }
        }
    }
}

#[test]
fn unrestricted_orbital_nmr_has_gauge_cancellation_all_existing_magnetic_classes() {
    let mol = molecule(true);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let grid = grid(&mol);
    let opts = options();
    let config = ResponseSolverConfig::default();
    for name in ["hf", "svwn5", "pbe", "pbe0", "cam-b3lyp", "r2scan"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let scf = run_scf(&mol, &mb, &opts, &functional, Some(&grid)).unwrap();
        let a = molecular_nmr_response(
            &mol,
            &mb,
            &functional,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            Vec3::zero(),
            &config,
        )
        .unwrap();
        let b = molecular_nmr_response(
            &mol,
            &mb,
            &functional,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            Vec3::new(0.43, -0.36, 0.29),
            &config,
        )
        .unwrap();
        let mut defect = 0.0_f64;
        for (a, b) in a.value.iter().zip(&b.value) {
            for i in 0..3 {
                for j in 0..3 {
                    defect = defect.max((a.tensor[i][j] - b.tensor[i][j]).abs());
                }
            }
        }
        println!("{name} unrestricted GIAO total shielding origin defect {defect:0.12e} ppm");
        assert!(defect < 1e-7);
        let d = magnetic_density_response(
            &mol,
            &mb,
            &functional,
            Some(&grid),
            &scf,
            &opts,
            Gauge::Giao,
            Vec3::zero(),
            &config,
        )
        .unwrap();
        assert_eq!(d.value.occupied, [5, 4]);
        assert!(d.value.projector_defect < 1e-9 && d.value.particle_number_defect < 1e-12);
        assert!(d.value.density_hat[0][0].max_abs_diff(&d.value.density_hat[1][0]) > 1e-6);
        assert!(d.residuals.iter().all(|r| *r <= config.tolerance));
        assert!(d.value.solver_methods.iter().all(|m| *m == "MINRES"));
    }
}

#[test]
fn forced_unrestricted_closed_shell_limit_and_raw_reference_contracts() {
    let mol = molecule(false);
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let functional = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let opts = options();
    let scf = run_scf(&mol, &mb, &opts, &functional, None).unwrap();
    let mut uopts = opts.clone();
    uopts.reference = Reference::Uhf;
    let uscf = run_scf(&mol, &mb, &uopts, &functional, None).unwrap();
    let config = ResponseSolverConfig::default();
    let a = magnetic_density_response(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        Gauge::Giao,
        Vec3::zero(),
        &config,
    )
    .unwrap();
    let b = magnetic_density_response(
        &mol,
        &mb,
        &functional,
        None,
        &uscf,
        &uopts,
        Gauge::Giao,
        Vec3::zero(),
        &config,
    )
    .unwrap();
    for spin in 0..2 {
        for axis in 0..3 {
            assert!(
                a.value.density_hat[spin][axis].max_abs_diff(&b.value.density_hat[spin][axis])
                    < 1e-8
            );
        }
    }
    let mut invalid = scf.clone();
    invalid.converged = false;
    assert!(magnetic_density_response(
        &mol,
        &mb,
        &functional,
        None,
        &invalid,
        &opts,
        Gauge::Giao,
        Vec3::zero(),
        &config
    )
    .is_err());
    assert!(magnetic_density_response(
        &mol,
        &mb,
        &functional,
        None,
        &scf,
        &opts,
        Gauge::Giao,
        Vec3::new(f64::NAN, 0., 0.),
        &config
    )
    .is_err());
}
