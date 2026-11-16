// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use dft_rs::basis::{data, MolBasis};
use dft_rs::linalg::{symmetric_eigen, Matrix};
use dft_rs::math::Vec3;
use dft_rs::response::complex::Complex;
use dft_rs::response::{dynamic_electric_tensor, dynamic_polarizability, Frequency};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};

#[path = "support/convergence.rs"]
mod convergence;

fn model() -> (
    Molecule,
    MolBasis,
    ScfOptions,
    dft_rs::xc::ResolvedFunctional,
    dft_rs::scf::ScfResult,
) {
    let mut mol = Molecule::new(vec![
        Atom {
            z: 2,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 1.75),
        },
    ]);
    mol.charge = 2;
    mol.multiplicity = 2;
    let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
    let opts = ScfOptions {
        max_iter: 250,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-12,
        ..Default::default()
    };
    let f = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let scf = run_scf(&mol, &mb, &opts, &f, None).unwrap();
    (mol, mb, opts, f, scf)
}

// Independent bare one-electron Liouville resolvents in the exact H eigenbasis.
// No projector blocks, nonlinear functional kernels or Casida operator enter.
fn spectral_dipole(energies: &[f64], r: &Matrix, zs: &[Complex]) -> Complex {
    let count = 1usize << zs.len();
    let mut d = vec![[Complex::ZERO; 4]; count];
    d[0][0] = Complex::ONE;
    for mask in 1..count {
        let z = zs
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .fold(Complex::ZERO, |a, (_, b)| a + *b);
        // Static proper subsets use closed derivatives of the two-level
        // ground eigenprojector (not a density idempotency recurrence).
        if zs
            .iter()
            .enumerate()
            .all(|(i, z)| mask & (1 << i) == 0 || *z == Complex::ZERO)
        {
            let gap = energies[1] - energies[0];
            let off = r[(0, 1)];
            if mask.count_ones() == 1 {
                d[mask][1] = (-off / gap).into();
                d[mask][2] = d[mask][1];
                continue;
            }
            if mask.count_ones() == 2 {
                d[mask][0] = (-2.0 * off * off / (gap * gap)).into();
                d[mask][3] = -d[mask][0];
                d[mask][1] = (2.0 * off * (r[(1, 1)] - r[(0, 0)]) / (gap * gap)).into();
                d[mask][2] = d[mask][1];
                continue;
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                let mut source = Complex::ZERO;
                for leg in 0..zs.len() {
                    if mask & (1 << leg) != 0 {
                        let b = mask ^ (1 << leg);
                        for k in 0..2 {
                            source += d[b][k * 2 + j] * r[(i, k)] - d[b][i * 2 + k] * r[(k, j)];
                        }
                    }
                }
                d[mask][i * 2 + j] = source / (z - Complex::from(energies[i] - energies[j]));
            }
        }
    }
    -(0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .fold(Complex::ZERO, |v, (i, j)| {
            v + d[count - 1][i * 2 + j] * r[(j, i)]
        })
}

#[test]
fn complex_mixing_matches_independent_spectral_resolvents_and_static_limits() {
    let (mol, mb, opts, f, scf) = model();
    let (eps, c) = symmetric_eigen(
        &scf.mo_coeff[0]
            .transpose()
            .matmul_seq(&scf.h_core)
            .matmul_seq(&scf.mo_coeff[0]),
    )
    .unwrap();
    let rao = dft_rs::integrals::dipole_matrices(&mb, Vec3::zero());
    let r = c
        .transpose()
        .matmul_seq(
            &scf.mo_coeff[0]
                .transpose()
                .matmul_seq(&rao[2])
                .matmul_seq(&scf.mo_coeff[0]),
        )
        .matmul_seq(&c);
    for order in 1..=3 {
        let fs: Vec<_> = (0..order)
            .map(|i| Frequency::Retarded {
                omega: 0.05 + 0.02 * i as f64,
                damping: 0.006 + 0.001 * i as f64,
            })
            .collect();
        let report =
            dynamic_electric_tensor(&mol, &mb, &f, None, &scf, &opts, &fs, &Default::default())
                .unwrap();
        assert!(
            report.converged,
            "{:?}; equation={}",
            report.warnings, report.value.equation_defect
        );
        let zs: Vec<_> = fs.iter().map(|f| f.energy().unwrap()).collect();
        let exact = spectral_dipole(&eps, &r, &zs);
        let actual = *report.value.tensor.last().unwrap();
        eprintln!(
            "order={order}, actual={actual:?}, spectral={exact:?}, equation={:.3e}",
            report.value.equation_defect
        );
        assert!((actual - exact).abs() < 2e-8 * exact.abs().max(1.0));
        assert_eq!(
            report.value.subset_solves,
            3usize.pow(order as u32) * ((1 << order) - 1)
        );
        assert!(report.value.projector_defect < 1e-9 && report.value.particle_number_defect < 1e-9);
        for z in &report.value.tensor[..report.value.tensor.len() - 1] {
            assert!(z.abs() < 1e-9);
        }
        let negative: Vec<_> = fs
            .iter()
            .map(|f| match f {
                Frequency::Retarded { omega, damping } => Frequency::Retarded {
                    omega: -omega,
                    damping: *damping,
                },
                _ => unreachable!(),
            })
            .collect();
        let reverse = dynamic_electric_tensor(
            &mol,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            &negative,
            &Default::default(),
        )
        .unwrap();
        assert!((*reverse.value.tensor.last().unwrap() - actual.conj()).abs() < 2e-9);
        let zero = dynamic_electric_tensor(
            &mol,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            &vec![Frequency::Static; order],
            &Default::default(),
        )
        .unwrap();
        let taylor = dft_rs::response::electric_field_taylor(
            &mol,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            [0.0, 0.0, 1.0],
            order,
            &Default::default(),
        )
        .unwrap();
        let factorial = if order == 3 { 6.0 } else { order as f64 };
        assert!(
            (zero.value.tensor.last().unwrap().re
                - factorial * taylor.value.dipole_coefficients[order][2])
                .abs()
                < 1e-8
        );
        if order == 1 {
            let alpha = dynamic_polarizability(
                &mol,
                &mb,
                &f,
                None,
                &scf,
                &opts,
                fs[0],
                &Default::default(),
            )
            .unwrap();
            assert!((actual - alpha.value.alpha[2][2]).abs() < 1e-9);
        }
    }
}

// Independently diagonalize a finite-amplitude monochromatic Floquet
// Hamiltonian, then Fourier analyze its occupied orbital density. The
// frequency-domain response code and its density recursion are not called.
fn floquet_harmonics(eps: &[f64], r: &Matrix, omega: f64, amplitude: f64) -> [f64; 3] {
    let bands = 17;
    let center = bands / 2;
    let mut h = Matrix::zeros(2 * bands, 2 * bands);
    for m in 0..bands {
        for i in 0..2 {
            h[(2 * m + i, 2 * m + i)] = eps[i] + (m as f64 - center as f64) * omega;
            if m + 1 < bands {
                for j in 0..2 {
                    h[(2 * m + i, 2 * (m + 1) + j)] = amplitude * r[(i, j)] / 2.0;
                    h[(2 * (m + 1) + j, 2 * m + i)] = amplitude * r[(i, j)] / 2.0;
                }
            }
        }
    }
    let (_, vectors) = symmetric_eigen(&h).unwrap();
    let root = (0..2 * bands)
        .max_by(|&a, &b| {
            vectors[(2 * center, a)]
                .abs()
                .total_cmp(&vectors[(2 * center, b)].abs())
        })
        .unwrap();
    std::array::from_fn(|axis| {
        let harmonic = axis + 1;
        let mut mu = 0.0;
        for m in 0..bands - harmonic {
            for i in 0..2 {
                for j in 0..2 {
                    mu -= vectors[(2 * m + i, root)]
                        * r[(i, j)]
                        * vectors[(2 * (m + harmonic) + j, root)];
                }
            }
        }
        mu * [2.0, 8.0, 48.0][axis] / amplitude.powi(harmonic as i32)
    })
}

#[test]
fn finite_amplitude_floquet_oracle_has_four_step_second_order_error_for_alpha_beta_gamma() {
    let (mol, mb, opts, f, scf) = model();
    let (eps, c) = symmetric_eigen(
        &scf.mo_coeff[0]
            .transpose()
            .matmul_seq(&scf.h_core)
            .matmul_seq(&scf.mo_coeff[0]),
    )
    .unwrap();
    let dip = dft_rs::integrals::dipole_matrices(&mb, Vec3::zero());
    let r = c
        .transpose()
        .matmul_seq(
            &scf.mo_coeff[0]
                .transpose()
                .matmul_seq(&dip[2])
                .matmul_seq(&scf.mo_coeff[0]),
        )
        .matmul_seq(&c);
    let omega = 0.05;
    let steps = [0.04, 0.02, 0.01, 0.005];
    let sampled: Vec<_> = steps
        .iter()
        .map(|h| floquet_harmonics(&eps, &r, omega, *h))
        .collect();
    for order in 1..=3 {
        let report = dynamic_electric_tensor(
            &mol,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            &vec![
                Frequency::Retarded {
                    omega,
                    damping: 0.0
                };
                order
            ],
            &Default::default(),
        )
        .unwrap();
        assert!(report.converged, "{:?}", report.warnings);
        let exact = report.value.tensor.last().unwrap().re;
        let errors: Vec<_> = sampled
            .iter()
            .map(|x| (x[order - 1] - exact).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        eprintln!(
            "dynamic order {order} Floquet errors={errors:?}; p={:.12}, C={:.12e}, \
            R_squared={:.12}, log_rmse={:.12e}",
            fit.exponent, fit.prefactor, fit.r_squared, fit.log_rmse
        );
        assert!((fit.exponent - 2.0).abs() < 0.02);
        assert!(fit.r_squared > 0.99999);
        for w in errors.windows(2) {
            assert!((w[0] / w[1] / 4.0 - 1.0).abs() < 0.03);
        }
        let richardson = (4.0 * sampled[3][order - 1] - sampled[2][order - 1]) / 3.0;
        assert!((richardson - exact).abs() < 1e-6 * exact.abs().max(1.0));
    }
}

#[test]
fn functional_classes_match_static_limits_and_four_step_electro_optic_oracles() {
    let (mut mol, mb, opts, _, _) = model();
    mol.charge = 1;
    mol.multiplicity = 1;
    let grid = dft_rs::grid::MolecularGrid::build(
        &mol,
        &dft_rs::grid::GridOptions {
            level: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let frequency = Frequency::Retarded {
        omega: 0.08,
        damping: 0.005,
    };
    let mut models: Vec<_> = ["svwn5", "pbe", "b3lyp", "cam-b3lyp", "r2scan"]
        .iter()
        .map(|s| resolve(&FunctionalSpec::Combined((*s).into())).unwrap())
        .collect();
    let mut weighted = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    for (weight, _) in &mut weighted.parts {
        *weight *= 0.63;
    }
    weighted.hybrid.cx = 0.37;
    weighted.description = "unregistered weighted local hybrid".into();
    models.push(weighted);
    for f in models {
        let scf = run_scf(&mol, &mb, &opts, &f, Some(&grid)).unwrap();
        let alpha = dynamic_polarizability(
            &mol,
            &mb,
            &f,
            Some(&grid),
            &scf,
            &opts,
            frequency,
            &Default::default(),
        )
        .unwrap();
        let direct = dynamic_electric_tensor(
            &mol,
            &mb,
            &f,
            Some(&grid),
            &scf,
            &opts,
            &[frequency],
            &Default::default(),
        )
        .unwrap();
        assert!((direct.value.tensor[8] - alpha.value.alpha[2][2]).abs() < 1e-8);
        let mut exact = [Complex::ZERO; 2];
        for order in 2..=3 {
            let mut frequencies = vec![Frequency::Static; order];
            frequencies[0] = frequency;
            let report = dynamic_electric_tensor(
                &mol,
                &mb,
                &f,
                Some(&grid),
                &scf,
                &opts,
                &frequencies,
                &Default::default(),
            )
            .unwrap();
            assert!(
                report.converged,
                "{} {:?}; equation={}",
                f.description, report.warnings, report.value.equation_defect
            );
            exact[order - 2] = *report.value.tensor.last().unwrap();
            let zero = dynamic_electric_tensor(
                &mol,
                &mb,
                &f,
                Some(&grid),
                &scf,
                &opts,
                &vec![Frequency::Static; order],
                &Default::default(),
            )
            .unwrap();
            let t = dft_rs::response::electric_field_taylor(
                &mol,
                &mb,
                &f,
                Some(&grid),
                &scf,
                &opts,
                [0.0, 0.0, 1.0],
                order,
                &Default::default(),
            )
            .unwrap();
            assert!(
                (zero.value.tensor.last().unwrap().re
                    - t.value.dipole_coefficients[order][2] * if order == 2 { 2.0 } else { 6.0 })
                .abs()
                    < 2e-8
            );
        }
        let mut errors: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        for &h in &steps {
            let mut samples = [Complex::ZERO; 2];
            for (i, sign) in [-1.0, 1.0].iter().enumerate() {
                let mut biased = opts.clone();
                biased.external_field = Some(Vec3::new(0.0, 0.0, sign * h));
                let field_scf = run_scf(&mol, &mb, &biased, &f, Some(&grid)).unwrap();
                let response = dynamic_polarizability(
                    &mol,
                    &mb,
                    &f,
                    Some(&grid),
                    &field_scf,
                    &biased,
                    frequency,
                    &Default::default(),
                )
                .unwrap();
                samples[i] = response.value.alpha[2][2];
            }
            errors[0].push(((samples[1] - samples[0]) / (2.0 * h) - exact[0]).abs());
            errors[1].push(
                ((samples[1] + samples[0] - alpha.value.alpha[2][2] * 2.0) / (h * h) - exact[1])
                    .abs(),
            );
        }
        for (i, errors) in errors.iter().enumerate() {
            let fit = convergence::error_power_fit(&steps, errors);
            eprintln!(
                "{} dynamic electro-optic order {}: errors={errors:?}, p={:.12}, \
                C={:.12e}, R_squared={:.12}, log_rmse={:.12e}",
                f.description,
                i + 2,
                fit.exponent,
                fit.prefactor,
                fit.r_squared,
                fit.log_rmse
            );
            assert!((fit.exponent - 2.0).abs() < 0.02);
            assert!(fit.r_squared > 0.99999);
        }
    }
}

#[test]
fn all_raw_complex_entries_transform_covariantly_and_frequency_slots_interchange() {
    let (mol, _, opts, f, _) = model();
    let axis = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let shift = Vec3::new(0.4, -0.3, 0.2);
    let mut rotated = mol.clone();
    rotated.atoms[0].position = shift;
    rotated.atoms[1].position = shift + Vec3::new(axis[0], axis[1], axis[2]) * 1.75;
    let mb = MolBasis::build(&rotated, &data::load("sto-3g").unwrap()).unwrap();
    let scf = run_scf(&rotated, &mb, &opts, &f, None).unwrap();
    let (eps, c) = symmetric_eigen(
        &scf.mo_coeff[0]
            .transpose()
            .matmul_seq(&scf.h_core)
            .matmul_seq(&scf.mo_coeff[0]),
    )
    .unwrap();
    let dips = dft_rs::integrals::dipole_matrices(&mb, Vec3::zero());
    let mut longitudinal = Matrix::zeros(mb.nao_sph, mb.nao_sph);
    for a in 0..3 {
        for (v, w) in longitudinal
            .as_mut_slice()
            .iter_mut()
            .zip(dips[a].as_slice())
        {
            *v += axis[a] * w;
        }
    }
    let r = c
        .transpose()
        .matmul_seq(
            &scf.mo_coeff[0]
                .transpose()
                .matmul_seq(&longitudinal)
                .matmul_seq(&scf.mo_coeff[0]),
        )
        .matmul_seq(&c);
    for order in 2..=3 {
        let fs: Vec<_> = (0..order)
            .map(|i| Frequency::Retarded {
                omega: 0.03 + 0.02 * i as f64,
                damping: 0.004 + 0.001 * i as f64,
            })
            .collect();
        let zs: Vec<_> = fs.iter().map(|f| f.energy().unwrap()).collect();
        let exact = spectral_dipole(&eps, &r, &zs);
        let report = dynamic_electric_tensor(
            &rotated,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            &fs,
            &Default::default(),
        )
        .unwrap();
        assert!(report.converged, "{:?}", report.warnings);
        for (flat, value) in report.value.tensor.iter().enumerate() {
            let mut code = flat;
            let mut product = 1.0;
            for _ in 0..=order {
                product *= axis[code % 3];
                code /= 3;
            }
            assert!((*value - exact * product).abs() < 1e-8);
        }
        let mut swapped_fs = fs.clone();
        swapped_fs.swap(0, 1);
        let swapped = dynamic_electric_tensor(
            &rotated,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            &swapped_fs,
            &Default::default(),
        )
        .unwrap();
        for flat in 0..report.value.tensor.len() {
            let mut code = flat;
            let mut axes = vec![0; order + 1];
            for a in axes.iter_mut().rev() {
                *a = code % 3;
                code /= 3;
            }
            axes.swap(1, 2);
            let partner = axes.iter().fold(0, |v, a| 3 * v + a);
            assert!((report.value.tensor[flat] - swapped.value.tensor[partner]).abs() < 1e-9);
        }
        assert_eq!(report.value.intrinsic_permutation_comparisons, 0);
    }
}

#[test]
fn physical_process_presets_match_independent_frequency_and_spectral_oracles() {
    use dft_rs::response::{dynamic_electric_process, ElectricProcess};
    let (mol, mb, opts, f, scf) = model();
    let (eps, c) = symmetric_eigen(
        &scf.mo_coeff[0]
            .transpose()
            .matmul_seq(&scf.h_core)
            .matmul_seq(&scf.mo_coeff[0]),
    )
    .unwrap();
    let dip = dft_rs::integrals::dipole_matrices(&mb, Vec3::zero());
    let r = c
        .transpose()
        .matmul_seq(
            &scf.mo_coeff[0]
                .transpose()
                .matmul_seq(&dip[2])
                .matmul_seq(&scf.mo_coeff[0]),
        )
        .matmul_seq(&c);
    let cases = [
        ("shg", vec![0.05], vec![0.05, 0.05]),
        ("sfg", vec![0.05, 0.07], vec![0.05, 0.07]),
        ("dfg", vec![0.07, 0.05], vec![0.07, -0.05]),
        ("optical-rectification", vec![0.05], vec![0.05, -0.05]),
        ("electro-optic", vec![0.05], vec![0.05, 0.0]),
        ("thg", vec![0.05], vec![0.05, 0.05, 0.05]),
        ("efish", vec![0.05], vec![0.05, 0.05, 0.0]),
        (
            "field-induced-rectification",
            vec![0.05],
            vec![0.05, -0.05, 0.0],
        ),
        ("two-photon-absorption", vec![0.05], vec![0.05, 0.05, -0.05]),
        ("optical-kerr", vec![0.05], vec![0.05, -0.05, 0.05]),
        ("dc-kerr", vec![0.05], vec![0.05, 0.0, 0.0]),
        (
            "four-wave-mixing",
            vec![0.05, 0.07, 0.09],
            vec![0.05, 0.07, -0.09],
        ),
    ];
    for (name, magnitudes, expected) in cases {
        let process: ElectricProcess = name.parse().unwrap();
        let report = dynamic_electric_process(
            &mol,
            &mb,
            &f,
            None,
            &scf,
            &opts,
            process,
            &magnitudes,
            0.006,
            &Default::default(),
        )
        .unwrap();
        assert_eq!(report.value.process, Some(process));
        assert!(report.converged, "{name}: {:?}", report.warnings);
        let zs: Vec<_> = expected
            .iter()
            .map(|&w| Complex::new(w, if w == 0.0 { 0.0 } else { 0.006 }))
            .collect();
        for (actual, expected) in report.value.input_frequencies.iter().zip(&zs) {
            assert_eq!(actual.energy().unwrap(), *expected);
        }
        let exact = spectral_dipole(&eps, &r, &zs);
        assert!(
            (*report.value.tensor.last().unwrap() - exact).abs() < 2e-8 * exact.abs().max(1.0),
            "{name}"
        );
        assert!(
            (report.value.output_frequency - zs.iter().fold(Complex::ZERO, |a, b| a + *b)).abs()
                < 1e-15
        );
        assert!(process.inputs(&[], 0.006).is_err());
        assert!(process.inputs(&magnitudes, -0.006).is_err());
    }
    assert!("unknown".parse::<ElectricProcess>().is_err());
}
