// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Screened field/current dense oracles and nonlinear finite-field SCF fits.
use dft_rs::basis::{BasisSet, ElementBasis, Shell};
use dft_rs::dfpt::coulomb::FiniteQHartree;
use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
use dft_rs::dfpt::field::{periodic_field_response, PeriodicFieldResponseOptions};
use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use dft_rs::linalg::{solve_linear, Matrix};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions, PeriodicScfResult};
use dft_rs::periodic::xc::{xc_contribution, AoDerivativeOrder, PeriodicXcGrid};
use dft_rs::periodic::{BlochMatrix, Cell, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::response::krylov::LinearOperator;
use dft_rs::response::Frequency;
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};
use dft_rs::{Atom, Molecule};

fn setup(
    name: &str,
) -> (
    Cell,
    ResolvedFunctional,
    PeriodicScfOptions,
    PeriodicScfResult,
) {
    let mut basis = BasisSet::new("primitive-field-oracle");
    basis.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![0.7],
            coefs: vec![1.0],
        }],
    });
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
    let cell = Cell::new(molecule, Lattice::cubic(9.0), &basis, 4.3).unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let mut options = PeriodicScfOptions {
        max_iter: 250,
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        grid_shell: 1,
        ..Default::default()
    };
    options.grid.radial_override = Some(10);
    options.grid.angular_override = Some(7);
    options.grid.prune = false;
    let reference = run_periodic_scf(&cell, &functional, &options).unwrap();
    (cell, functional, options, reference)
}
fn vertices(
    cell: &Cell,
    o: &PeriodicScfOptions,
    r: &PeriodicFieldResponseOptions,
) -> dft_rs::periodic::field::PeriodicFieldReport {
    periodic_field_operators(
        cell.molecule(),
        cell.basis(),
        &PeriodicGeometry::new(cell.lattice().clone(), PeriodicBoundary::FULL_3D).unwrap(),
        &PeriodicFieldOptions {
            k_mesh: o.k_mesh,
            basis_cutoff: cell.overlap_cutoff(),
            wave: r.response.q,
            frequency: r.response.frequency,
            scalar_potential: r.scalar_potential,
            vector_potential: r.vector_potential,
            ..Default::default()
        },
    )
    .unwrap()
}
fn contraction(v: &BlochMatrix, p: &BlochMatrix) -> Complex {
    let mut z = Complex::ZERO;
    for i in 0..v.rows() {
        for j in 0..v.cols() {
            z += Complex::new(v.re[(i, j)], -v.im[(i, j)])
                * Complex::new(p.re[(i, j)], p.im[(i, j)]);
        }
    }
    z
}
#[test]
fn nonmesh_electric_magnetic_response_matches_independent_dense_screening() {
    for name in ["svwn5", "pbe"] {
        let (cell, f, o, p) = setup(name);
        let r = PeriodicFieldResponseOptions {
            response: PeriodicResponseOptions {
                q: Vec3::new(0.043, 0.017, -0.021),
                frequency: Frequency::Retarded {
                    omega: 0.13,
                    damping: 0.04,
                },
                reciprocal_cutoff: 0.85,
                ..Default::default()
            },
            scalar_potential: Complex::new(0.37, 0.22),
            vector_potential: [
                Complex::new(0.1, -0.2),
                Complex::new(-0.3, 0.11),
                Complex::new(0.27, 0.19),
            ],
        };
        let actual = periodic_field_response(&cell, &f, &p, &o, &r).unwrap();
        let v = vertices(&cell, &o, &r);
        let solver = PeriodicDensitySolver::new(&cell, &f, &p, &o, &r.response).unwrap();
        let source = [v.first_order.clone(), v.first_order.clone()];
        let rhs = solver
            .independent_response(&solver.potential_to_bands(&source).unwrap())
            .unwrap();
        let n = solver.dimension();
        let mut dense = Matrix::zeros(2 * n, 2 * n);
        for j in 0..n {
            let mut unit = vec![Complex::ZERO; n];
            unit[j] = Complex::from(1.0);
            let a = solver.apply(&unit).unwrap();
            for i in 0..n {
                dense[(i, j)] = a[i].re;
                dense[(i, n + j)] = -a[i].im;
                dense[(n + i, j)] = a[i].im;
                dense[(n + i, n + j)] = a[i].re;
            }
        }
        let b: Vec<_> = rhs
            .iter()
            .map(|v| v.re)
            .chain(rhs.iter().map(|v| v.im))
            .collect();
        let answer = solve_linear(&dense, &b).unwrap();
        let band: Vec<_> = (0..n)
            .map(|i| Complex::new(answer[i], answer[n + i]))
            .collect();
        let density = solver.density_to_aos(&band).unwrap();
        let mut number = Complex::ZERO;
        let mut current = [Complex::ZERO; 3];
        for spin in 0..2 {
            assert!(
                density[spin][0].rms_diff(&actual.value.density_response.density[spin][0]) < 1e-11
            );
            number += contraction(&v.density_vertex[0], &density[spin][0]);
            for a in 0..3 {
                current[a] += contraction(&v.current_vertex[a][0], &density[spin][0]);
            }
        }
        assert!((number - actual.value.number_density).abs() < 1e-12);
        assert!((actual.value.charge_density + number).abs() < 1e-12);
        for a in 0..3 {
            assert!((current[a] - actual.value.paramagnetic_number_current[a]).abs() < 1e-12);
            let contact = r.vector_potential[a] * 2.0;
            assert!((contact - actual.value.diamagnetic_number_current[a]).abs() < 1e-12);
            assert!((actual.value.number_current[a] - current[a] - contact).abs() < 1e-12);
            assert!(
                (actual.value.number_current[a] + actual.value.charge_current[a]).abs() < 1e-12
            );
        }
        let q = r.response.q;
        let z = r.response.frequency.energy().unwrap();
        let ward = (z * number
            - (0..3).fold(Complex::ZERO, |s, a| {
                s + actual.value.number_current[a] * q.get(a)
            }))
        .abs();
        assert!((ward - actual.value.continuity_defect).abs() < 1e-14);
        assert!(
            actual
                .value
                .magnetic_field
                .iter()
                .map(|b| b.abs())
                .sum::<f64>()
                > 0.0
        );
        assert!(
            actual
                .value
                .electric_field
                .iter()
                .map(|b| b.abs())
                .sum::<f64>()
                > 0.0
        );
        assert!(actual.residuals[0] < 1e-10);
        assert_eq!(actual.value.target_k_points[0], p.k_points[0] + q);
        assert!(actual.value.density_response.density[0][0].hermiticity_error() > 1e-5);
        println!(
            "screened field {name}: residual={:.12e} continuity={:.12e}",
            actual.residuals[0], ward
        );
    }
}

/// Independent nonlinear two-band SCF oracle. Diagonalize the Hermitian
/// 2x2 Fock analytically with Pauli components, including arbitrary imaginary
/// mixing. Evaluate full semilocal potentials and reciprocal Hartree at each
/// iterate, rather than using a finite difference of the response solver.
struct NonlinearOracle<'a> {
    functional: &'a ResolvedFunctional,
    grid: PeriodicXcGrid,
    hartree: FiniteQHartree,
    coefficients: BlochMatrix,
    base: BlochMatrix,
}
impl<'a> NonlinearOracle<'a> {
    fn new(
        cell: &Cell,
        f: &'a ResolvedFunctional,
        o: &PeriodicScfOptions,
        p: &PeriodicScfResult,
    ) -> Self {
        let grid = PeriodicXcGrid::build(
            cell,
            &p.k_points,
            &o.grid,
            o.grid_shell,
            cell.overlap_cutoff(),
            AoDerivativeOrder::for_scf(f.family),
        )
        .unwrap();
        let hartree = FiniteQHartree::new(
            cell,
            &p.k_points,
            Vec3::zero(),
            0.85,
            cell.hartree_real_space_cutoff(),
        )
        .unwrap();
        let j = hartree.apply(&p.density).unwrap();
        let x = xc_contribution(f, &grid, &p.density).unwrap();
        // Recenter the independent truncated-Coulomb model on the actual
        // reference. Its derivative has precisely the declared response
        // reciprocal cutoff, without importing an Ewald ground-state tail.
        let mut base = p.fock[0][0].clone();
        base.add_scaled(&j[0][0], -1.0);
        base.add_scaled(&x.vxc[0][0], -1.0);
        let coefficients = generalized_orbitals(&p.fock[0][0], &p.overlap[0], o.orth_threshold)
            .unwrap()
            .coefficients;
        Self {
            functional: f,
            grid,
            hartree,
            coefficients,
            base,
        }
    }
    fn density(&self, linear: &BlochMatrix, quadratic: &BlochMatrix, h: f64) -> BlochMatrix {
        let mut mo = BlochMatrix::zeros(2, 2);
        mo.re[(0, 0)] = 1.0;
        let cadj = adjoint(&self.coefficients).unwrap();
        for iteration in 0..5000 {
            let ao = matmul(&matmul(&self.coefficients, &mo).unwrap(), &cadj).unwrap();
            let spin = [vec![ao.clone()], vec![ao]];
            let j = self.hartree.apply(&spin).unwrap();
            let x = xc_contribution(self.functional, &self.grid, &spin).unwrap();
            let mut fock = self.base.clone();
            fock.add_scaled(&j[0][0], 1.0);
            fock.add_scaled(&x.vxc[0][0], 1.0);
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
    println!("field response fit {label}: h={h:?} errors={errors:?} p={p:.12} R2={r2:.12}");
    assert!((p - 2.0).abs() < 0.02 && r2 > 0.99999);
    for e in errors.windows(2) {
        assert!((e[0] / e[1] / 4.0 - 1.0).abs() < 0.03);
    }
}
#[test]
fn electric_and_magnetic_derivatives_match_four_step_nonlinear_scf_fits() {
    for name in ["svwn5", "pbe"] {
        let (cell, f, o, p) = setup(name);
        let oracle = NonlinearOracle::new(&cell, &f, &o, &p);
        for magnetic in [false, true] {
            let q = Vec3::new(2.0 * std::f64::consts::PI / 9.0, 0.0, 0.0);
            let r = PeriodicFieldResponseOptions {
                response: PeriodicResponseOptions {
                    q,
                    reciprocal_cutoff: 0.85,
                    ..Default::default()
                },
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
            };
            let plus = periodic_field_response(&cell, &f, &p, &o, &r).unwrap();
            let minus_request = PeriodicFieldResponseOptions {
                response: PeriodicResponseOptions {
                    q: -q,
                    ..r.response.clone()
                },
                scalar_potential: r.scalar_potential.conj(),
                vector_potential: r.vector_potential.map(|a| a.conj()),
            };
            let minus = periodic_field_response(&cell, &f, &p, &o, &minus_request).unwrap();
            let mut predicted = plus.value.density_response.density[0][0].clone();
            predicted.add_scaled(&minus.value.density_response.density[0][0], 1.0);
            assert!(predicted.hermiticity_error() < 1e-12);
            let v = vertices(&cell, &o, &r);
            let mut linear = v.first_order[0].clone();
            linear.add_scaled(&adjoint(&v.first_order[0]).unwrap(), 1.0);
            let mut quadratic = v.second_order[0].clone();
            quadratic.add_scaled(&adjoint(&v.second_order[0]).unwrap(), 1.0);
            quadratic.scale(0.5);
            quadratic.add_scaled(
                &p.overlap[0],
                r.vector_potential.iter().map(|a| a.abs().powi(2)).sum(),
            );
            let h = [0.02, 0.01, 0.005, 0.0025];
            let mut errors = Vec::new();
            let mut derivatives = Vec::new();
            for &step in &h {
                let mut fd = oracle.density(&linear, &quadratic, step);
                fd.add_scaled(&oracle.density(&linear, &quadratic, -step), -1.0);
                fd.scale(0.5 / step);
                errors.push(fd.rms_diff(&predicted));
                derivatives.push(fd);
            }
            fit(
                &format!("{name} {}", if magnetic { "magnetic" } else { "electric" }),
                &h,
                &errors,
            );
            let mut rich = derivatives[3].clone();
            rich.scale(4.0 / 3.0);
            rich.add_scaled(&derivatives[2], -1.0 / 3.0);
            assert!(rich.rms_diff(&predicted) < 1e-8);
        }
    }
}

#[test]
fn constant_scalar_gauge_and_invalid_models_do_not_fake_a_field() {
    let (cell, f, o, p) = setup("svwn5");
    let r = PeriodicFieldResponseOptions {
        scalar_potential: Complex::from(1.0),
        response: PeriodicResponseOptions {
            reciprocal_cutoff: 0.85,
            ..Default::default()
        },
        ..Default::default()
    };
    let report = periodic_field_response(&cell, &f, &p, &o, &r).unwrap();
    assert!(report.value.number_density.abs() < 1e-12);
    for d in report.value.density_response.density.iter().flatten() {
        assert!(d
            .re
            .as_slice()
            .iter()
            .chain(d.im.as_slice())
            .all(|v| v.abs() < 1e-12));
    }
    assert_eq!(report.value.electric_field, [Complex::ZERO; 3]);
    assert_eq!(report.value.magnetic_field, [Complex::ZERO; 3]);
    let mut bad = r.clone();
    bad.vector_potential[0] = Complex::new(f64::NAN, 0.0);
    assert!(periodic_field_response(&cell, &f, &p, &o, &bad).is_err());
    let meta = resolve(&FunctionalSpec::Combined("r2scan".into())).unwrap();
    assert!(periodic_field_response(&cell, &meta, &p, &o, &r).is_err());
    let hybrid = resolve(&FunctionalSpec::Combined("pbe0".into())).unwrap();
    assert!(periodic_field_response(&cell, &hybrid, &p, &o, &r).is_err());
}

#[test]
fn matching_hybrid_references_preserve_the_uniform_scalar_gauge() {
    // A different functional paired with a semilocal reference already fails
    // the reference fingerprint check. Use the actual hybrid parent here so
    // that this test exercises the connected exchange kernel.
    for name in ["pbe0", "cam-b3lyp"] {
        let (cell, functional, options, reference) = setup(name);
        let request = PeriodicFieldResponseOptions {
            response: PeriodicResponseOptions {
                frequency: Frequency::Retarded {
                    omega: 0.13,
                    damping: 0.04,
                },
                reciprocal_cutoff: 0.85,
                ..Default::default()
            },
            scalar_potential: Complex::from(1.0),
            ..Default::default()
        };
        let report =
            periodic_field_response(&cell, &functional, &reference, &options, &request).unwrap();
        assert!(report.converged);
        assert!(report.model.contains("exact-exchange kernels"));
        assert!(report.value.number_density.abs() < 1e-12);
        assert!(report.value.number_current.iter().all(|v| v.abs() < 1e-12));
        assert_eq!(report.value.electric_field, [Complex::ZERO; 3]);
        assert_eq!(report.value.magnetic_field, [Complex::ZERO; 3]);
        for density in report.value.density_response.density.iter().flatten() {
            assert!(density
                .re
                .as_slice()
                .iter()
                .chain(density.im.as_slice())
                .all(|v| v.abs() < 1e-12));
        }
    }
}

#[test]
fn closed_free_electron_modes_obey_continuity_and_static_vector_gauge() {
    use dft_rs::dfpt::field::fourier_observable;
    use dft_rs::dfpt::spectral::{BandPair, BandSpace, SpectralPropagator};
    // Reciprocal plane waves 0,+q,-q. Occupied-to-empty transitions close
    // the linear number/current commutator exactly for the occupied state.
    let q = 0.7;
    let epsilon = 0.5 * q * q;
    let hartree = 0.31;
    let mut n = BlochMatrix::zeros(3, 3);
    n.re[(1, 0)] = 1.0;
    n.re[(0, 2)] = 1.0;
    let mut j = n.clone();
    j.re[(1, 0)] = q / 2.0;
    j.re[(0, 2)] = -q / 2.0;
    for frequency in [
        Frequency::Static,
        Frequency::Retarded {
            omega: 0.13,
            damping: 0.04,
        },
        Frequency::Imaginary { frequency: 0.2 },
        Frequency::Retarded {
            omega: 5.0,
            damping: 0.02,
        },
    ] {
        let bands = BandSpace::integer(vec![0.0, epsilon, epsilon], 1).unwrap();
        let response = SpectralPropagator::new(
            vec![BandPair {
                bra: bands.clone(),
                ket: bands,
                weight: 1.0,
            }],
            [q, 0.0, 0.0],
            frequency,
            true,
        )
        .unwrap();
        let apply = |v: &BlochMatrix| {
            let source: Vec<_> = (0..3)
                .flat_map(|i| (0..3).map(move |k| Complex::new(v.re[(i, k)], v.im[(i, k)])))
                .collect();
            let result = response.apply(&source).unwrap();
            let mut p = BlochMatrix::zeros(3, 3);
            for i in 0..3 {
                for k in 0..3 {
                    p.re[(i, k)] = result[3 * i + k].re;
                    p.im[(i, k)] = result[3 * i + k].im;
                }
            }
            p
        };
        let rn = apply(&n);
        let rj = apply(&j);
        let chi = fourier_observable(&[n.clone()], &[vec![rn.clone()], vec![rn]]).unwrap();
        let mixed = fourier_observable(&[n.clone()], &[vec![rj.clone()], vec![rj]]).unwrap();
        let z = frequency.energy().unwrap();
        let closed = Complex::from(4.0 * epsilon) / (z * z - Complex::from(epsilon * epsilon));
        assert!((chi - closed).abs() < 1e-13);
        for scalar in [0.0, 0.37] {
            for vector in [0.0, 0.27] {
                let rho = (chi * scalar + mixed * vector) / (Complex::from(1.0) - chi * hartree);
                let expected = closed * (Complex::from(scalar) + z * (vector / q))
                    / (Complex::from(1.0) - closed * hartree);
                assert!((rho - expected).abs() < 1e-13);
                let mut v = n.clone();
                let scalar_total = Complex::from(scalar) + rho * hartree;
                for i in 0..3 {
                    for k in 0..3 {
                        v.re[(i, k)] = scalar_total.re * n.re[(i, k)] + vector * j.re[(i, k)];
                        v.im[(i, k)] = scalar_total.im * n.re[(i, k)];
                    }
                }
                let p = apply(&v);
                let current = fourier_observable(&[j.clone()], &[vec![p.clone()], vec![p]])
                    .unwrap()
                    + Complex::from(2.0 * vector);
                assert!((z * rho - current * q).abs() < 1e-13);
                if z == Complex::ZERO && scalar == 0.0 {
                    assert!(current.abs() < 1e-13 && rho.abs() < 1e-13);
                }
            }
        }
    }
}
