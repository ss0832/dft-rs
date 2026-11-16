// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Actual dimensional screened field response and independent nonlinear oracles.
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::dfpt::field_dimensional::{
    dimensional_field_response, DimensionalDensitySolver, DimensionalFieldResponseOptions,
};
use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
use dft_rs::periodic::occupations::OccupationScheme;
use dft_rs::periodic::quasiperiodic::PeriodicTransfer;
use dft_rs::periodic::scf_boundary::{
    run_dimensional_scf, DimensionalScfBackend, DimensionalScfOptions, DimensionalScfReport,
};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::response::krylov::LinearOperator;
use dft_rs::response::Frequency;
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};
use dft_rs::{Atom, Molecule};

fn setup(
    mask: usize,
    name: &str,
) -> (
    Molecule,
    MolBasis,
    PeriodicGeometry,
    ResolvedFunctional,
    DimensionalScfOptions,
    DimensionalScfReport,
) {
    let molecule = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, -0.7),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.4, 0.2, 0.7),
        },
    ]);
    let mut basis = BasisSet::new("primitive-dimensional-field-oracle");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
    let basis = MolBasis::build(&molecule, &basis).unwrap();
    let boundary = PeriodicBoundary::new(std::array::from_fn(|a| mask & (1 << a) != 0));
    let geometry = PeriodicGeometry::new(Lattice::cubic(9.0), boundary).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let mut options = DimensionalScfOptions::for_boundary(boundary);
    options.basis_cutoff = 4.3;
    options.grid_cutoff = 12.0;
    options.scf.max_iter = 250;
    options.scf.e_tol = 1e-14;
    options.scf.p_tol = 1e-13;
    options.scf.conv_tol_grad = 1e-13;
    options.scf.grid.radial_override = Some(10);
    options.scf.grid.angular_override = Some(7);
    options.scf.grid.prune = false;
    options.electrostatics.alpha = Some(0.65);
    options.electrostatics.real_cutoff = Some(18.0);
    options.electrostatics.reciprocal_cutoff = Some(3.7);
    let reference =
        run_dimensional_scf(&molecule, &basis, &geometry, &functional, &options).unwrap();
    assert!(reference.reference.converged);
    (molecule, basis, geometry, functional, options, reference)
}
struct NonlinearOracle<'a> {
    backend: &'a DimensionalScfBackend<'a>,
    coefficients: BlochMatrix,
}
impl NonlinearOracle<'_> {
    fn density(&self, linear: &BlochMatrix, quadratic: &BlochMatrix, h: f64) -> BlochMatrix {
        let mut mo = BlochMatrix::zeros(2, 2);
        mo.re[(0, 0)] = 1.0;
        let cadj = adjoint(&self.coefficients).unwrap();
        for iteration in 0..5000 {
            let ao = matmul(&matmul(&self.coefficients, &mo).unwrap(), &cadj).unwrap();
            let spin = [vec![ao.clone()], vec![ao]];
            let mut fock = self.backend.evaluate(&spin).unwrap().fock[0][0].clone();
            fock.add_scaled(linear, h);
            fock.add_scaled(quadratic, h * h);
            let f = matmul(&matmul(&cadj, &fock).unwrap(), &self.coefficients).unwrap();
            assert!(f.hermiticity_error() < 1e-12);
            let d = 0.5 * (f.re[(0, 0)] - f.re[(1, 1)]);
            let off = Complex::new(f.re[(0, 1)], f.im[(0, 1)]);
            let radius = (d * d + off.abs().powi(2)).sqrt();
            assert!(radius > 0.0);
            let mut next = BlochMatrix::zeros(2, 2);
            next.re[(0, 0)] = 0.5 * (1.0 - d / radius);
            next.re[(1, 1)] = 0.5 * (1.0 + d / radius);
            next.re[(0, 1)] = -off.re / (2.0 * radius);
            next.re[(1, 0)] = next.re[(0, 1)];
            next.im[(0, 1)] = -off.im / (2.0 * radius);
            next.im[(1, 0)] = -next.im[(0, 1)];
            if next.rms_diff(&mo) < 1e-14 {
                return matmul(&matmul(&self.coefficients, &next).unwrap(), &cadj).unwrap();
            }
            mo.scale(0.65);
            mo.add_scaled(&next, 0.35);
            assert!(
                iteration < 4999,
                "independent nonlinear SCF did not converge"
            );
        }
        unreachable!()
    }
}
fn fit(label: &str, h: &[f64], errors: &[f64]) {
    assert!(errors.iter().all(|e| e.is_finite() && *e > 0.0));
    let x: Vec<_> = h.iter().map(|v| v.ln()).collect();
    let y: Vec<_> = errors.iter().map(|v| v.ln()).collect();
    let xm = x.iter().sum::<f64>() / 4.0;
    let ym = y.iter().sum::<f64>() / 4.0;
    let xx = x.iter().map(|v| (v - xm).powi(2)).sum::<f64>();
    let p = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - xm) * (b - ym))
        .sum::<f64>()
        / xx;
    let r2 = 1.0
        - x.iter()
            .zip(&y)
            .map(|(a, b)| (b - ym - p * (a - xm)).powi(2))
            .sum::<f64>()
            / y.iter().map(|v| (v - ym).powi(2)).sum::<f64>();
    println!(
        "dimensional screened field fit {label}: h={h:?} errors={errors:?} p={p:.12} R2={r2:.12}"
    );
    assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
    for e in errors.windows(2) {
        assert!((e[0] / e[1] / 4.0 - 1.0).abs() < 0.03);
    }
}

#[test]
fn all_periodic_masks_match_dense_complex_screening_and_raw_observables() {
    for mask in 1..8 {
        for name in ["svwn5", "pbe"] {
            let (m, b, g, f, o, p) = setup(mask, name);
            let mut fractional = [0.0; 3];
            fractional[g.boundary().axes.iter().position(|&a| a).unwrap()] = 0.231;
            let r = DimensionalFieldResponseOptions {
                transfer_fractional: fractional,
                frequency: Frequency::Retarded {
                    omega: 0.13,
                    damping: 0.04,
                },
                scalar_potential: Complex::new(0.37, 0.22),
                vector_potential: [
                    Complex::new(0.1, -0.2),
                    Complex::new(-0.3, 0.11),
                    Complex::new(0.27, 0.19),
                ],
                ..Default::default()
            };
            let actual = dimensional_field_response(&m, &b, &g, &f, &p, &o, &r).unwrap();
            let backend = DimensionalScfBackend::new(&m, &b, &g, &f, &o).unwrap();
            let solver = DimensionalDensitySolver::new(&backend, &b, &g, &f, &p, &o, &r).unwrap();
            let wave = PeriodicTransfer::new(&g, fractional).unwrap().cartesian();
            let v = periodic_field_operators(
                &m,
                &b,
                &g,
                &PeriodicFieldOptions {
                    k_mesh: o.scf.k_mesh,
                    basis_cutoff: o.basis_cutoff,
                    wave,
                    frequency: r.frequency,
                    scalar_potential: r.scalar_potential,
                    vector_potential: r.vector_potential,
                    ..Default::default()
                },
            )
            .unwrap();
            let rhs = solver
                .independent_response(
                    &solver
                        .potential_to_bands(&[v.first_order.clone(), v.first_order.clone()])
                        .unwrap(),
                )
                .unwrap();
            let n = solver.dimension();
            let mut a = Matrix::zeros(2 * n, 2 * n);
            for j in 0..n {
                let mut unit = vec![Complex::ZERO; n];
                unit[j] = Complex::from(1.0);
                let col = solver.apply(&unit).unwrap();
                for i in 0..n {
                    a[(i, j)] = col[i].re;
                    a[(i, n + j)] = -col[i].im;
                    a[(n + i, j)] = col[i].im;
                    a[(n + i, n + j)] = col[i].re;
                }
            }
            let rhs: Vec<_> = rhs
                .iter()
                .map(|z| z.re)
                .chain(rhs.iter().map(|z| z.im))
                .collect();
            let x = solve_linear(&a, &rhs).unwrap();
            let x: Vec<_> = (0..n).map(|i| Complex::new(x[i], x[n + i])).collect();
            let expected = solver.density_to_aos(&x).unwrap();
            for spin in 0..2 {
                assert!(expected[spin][0].rms_diff(&actual.value.density[spin][0]) < 1e-11);
            }
            assert!(actual.residuals[0] < 1e-10);
            assert!(actual.value.hartree_quadrature_defect < 1e-11);
            assert_eq!(
                actual.value.target_k_points[0],
                p.reference.k_points[0] + wave
            );
            assert!(actual.value.density[0][0].hermiticity_error() > 1e-5);
            for axis in 0..3 {
                assert!(
                    (actual.value.diamagnetic_number_current[axis]
                        - r.vector_potential[axis] * 2.0)
                        .abs()
                        < 1e-12
                );
                assert!(
                    (actual.value.number_current[axis] + actual.value.charge_current[axis]).abs()
                        < 1e-12
                );
            }
            let raw = (r.frequency.energy().unwrap() * actual.value.number_density
                - (0..3).fold(Complex::ZERO, |s, a| {
                    s + actual.value.number_current[a] * wave.get(a)
                }))
            .abs();
            assert!((raw - actual.value.continuity_defect).abs() < 1e-14);
            println!(
                "dimensional screened field mask={mask} {name} residual={:.12e} \
                quadrature={:.12e} continuity={:.12e}",
                actual.residuals[0], actual.value.hartree_quadrature_defect, raw
            );
        }
    }
}
#[test]
fn electric_and_magnetic_derivatives_match_four_step_nonlinear_dimensional_scf() {
    for mask in [1, 3, 7] {
        for name in ["svwn5", "pbe"] {
            let (m, b, g, f, o, p) = setup(mask, name);
            let backend = DimensionalScfBackend::new(&m, &b, &g, &f, &o).unwrap();
            let oracle = NonlinearOracle {
                backend: &backend,
                coefficients: generalized_orbitals(
                    &p.reference.fock[0][0],
                    &p.reference.overlap[0],
                    o.scf.orth_threshold,
                )
                .unwrap()
                .coefficients,
            };
            for magnetic in [false, true] {
                let r = DimensionalFieldResponseOptions {
                    transfer_fractional: [1.0, 0.0, 0.0],
                    scalar_potential: if magnetic {
                        Complex::ZERO
                    } else {
                        Complex::new(0.37, 0.22)
                    },
                    vector_potential: if magnetic {
                        [Complex::ZERO, Complex::ZERO, Complex::new(0.27, 0.19)]
                    } else {
                        [Complex::ZERO; 3]
                    },
                    ..Default::default()
                };
                let plus = dimensional_field_response(&m, &b, &g, &f, &p, &o, &r).unwrap();
                let negative = DimensionalFieldResponseOptions {
                    transfer_fractional: [-1.0, 0.0, 0.0],
                    scalar_potential: r.scalar_potential.conj(),
                    vector_potential: r.vector_potential.map(|a| a.conj()),
                    ..r.clone()
                };
                let minus = dimensional_field_response(&m, &b, &g, &f, &p, &o, &negative).unwrap();
                let mut predicted = plus.value.density[0][0].clone();
                predicted.add_scaled(&minus.value.density[0][0], 1.0);
                assert!(predicted.hermiticity_error() < 1e-12);
                let wave = PeriodicTransfer::new(&g, r.transfer_fractional)
                    .unwrap()
                    .cartesian();
                let v = periodic_field_operators(
                    &m,
                    &b,
                    &g,
                    &PeriodicFieldOptions {
                        basis_cutoff: o.basis_cutoff,
                        wave,
                        scalar_potential: r.scalar_potential,
                        vector_potential: r.vector_potential,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut linear = v.first_order[0].clone();
                linear.add_scaled(&adjoint(&linear).unwrap(), 1.0);
                let mut quadratic = v.second_order[0].clone();
                quadratic.add_scaled(&adjoint(&quadratic).unwrap(), 1.0);
                quadratic.scale(0.5);
                quadratic.add_scaled(
                    &p.reference.overlap[0],
                    r.vector_potential.iter().map(|a| a.abs().powi(2)).sum(),
                );
                let hs = [0.02, 0.01, 0.005, 0.0025];
                let mut approximations = Vec::new();
                let mut errors = Vec::new();
                for &h in &hs {
                    let mut fd = oracle.density(&linear, &quadratic, h);
                    fd.add_scaled(&oracle.density(&linear, &quadratic, -h), -1.0);
                    fd.scale(0.5 / h);
                    errors.push(fd.rms_diff(&predicted));
                    approximations.push(fd);
                }
                fit(
                    &format!(
                        "mask={mask} {name} {}",
                        if magnetic { "magnetic" } else { "electric" }
                    ),
                    &hs,
                    &errors,
                );
                let mut rich = approximations[3].clone();
                rich.scale(4.0 / 3.0);
                rich.add_scaled(&approximations[2], -1.0 / 3.0);
                assert!(rich.rms_diff(&predicted) < 1e-8);
            }
        }
    }
}
#[test]
fn thermal_folded_uniform_channel_conserves_number_and_validates_reference() {
    let (m, b, g, f, mut o, _) = setup(1, "svwn5");
    o.scf.occupation = OccupationScheme::FermiDirac {
        temperature_kelvin: 200000.0,
    };
    let mut p = run_dimensional_scf(&m, &b, &g, &f, &o).unwrap();
    for fractional in [[0.0; 3], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] {
        let r = DimensionalFieldResponseOptions {
            transfer_fractional: fractional,
            scalar_potential: Complex::new(0.37, 0.22),
            ..Default::default()
        };
        let value = dimensional_field_response(&m, &b, &g, &f, &p, &o, &r).unwrap();
        let number =
            dft_rs::dfpt::field::fourier_observable(&p.reference.overlap, &value.value.density)
                .unwrap();
        assert!(number.abs() < 1e-11);
        assert_eq!(value.value.reduced_transfer_fractional, [0.0; 3]);
        assert_eq!(value.value.target_k_points, p.reference.k_points);
        if fractional == [0.0; 3] {
            assert!(value.value.number_density.abs() < 1e-11);
        }
    }
    p.reference.fock[0][0].re[(0, 0)] += 0.001;
    assert!(dimensional_field_response(&m, &b, &g, &f, &p, &o, &Default::default()).is_err());
}
