// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full spin coherence, variational energy derivatives and Pauli SOC checks.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_scf::StaticPeriodicField;
use dft_rs::periodic::scf_boundary::DimensionalScfOptions;
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
fn z(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn put(m: &mut BlochMatrix, i: usize, j: usize, z: Complex) {
    m.re[(i, j)] = z.re;
    m.im[(i, j)] = z.im;
}
fn basis(mol: &Molecule) -> MolBasis {
    let mut set = BasisSet::new("compact-spinor-oracle");
    set.insert(ElementBasis {
        z: mol.atoms[0].z,
        shells: vec![
            Shell {
                l: 0,
                exps: vec![0.7],
                coefs: vec![1.0],
            },
            Shell {
                l: 1,
                exps: vec![0.5],
                coefs: vec![1.0],
            },
        ],
    });
    MolBasis::build(mol, &set).unwrap()
}
fn options(boundary: PeriodicBoundary) -> DimensionalScfOptions {
    let mut o = DimensionalScfOptions::for_boundary(boundary);
    o.basis_cutoff = 4.3;
    o.grid_cutoff = 8.0;
    o.scf.exchange_cutoff = Some(4.3);
    o.scf.grid.level = 2;
    o.electrostatics.alpha = Some(0.55);
    o.electrostatics.real_cutoff = Some(16.0);
    o.electrostatics.reciprocal_cutoff = Some(8.0);
    o
}
fn projector(theta: f64, n: usize) -> (BlochMatrix, BlochMatrix) {
    // Orthogonal s-up and p-down components: all three spin directions can
    // be recovered after a general complex global spin rotation.
    let mut c = BlochMatrix::zeros(2 * n, 1);
    let mut dc = c.clone();
    put(&mut c, 0, 0, Complex::new(theta.cos(), 0.0));
    put(&mut c, n + 1, 0, Complex::new(0.0, theta.sin()));
    put(&mut dc, 0, 0, Complex::new(-theta.sin(), 0.0));
    put(&mut dc, n + 1, 0, Complex::new(0.0, theta.cos()));
    let p = matmul(&c, &adjoint(&c).unwrap()).unwrap();
    let mut dp = matmul(&dc, &adjoint(&c).unwrap()).unwrap();
    dp.add_scaled(&matmul(&c, &adjoint(&dc).unwrap()).unwrap(), 1.0);
    (p, dp)
}
fn rotation(n: usize) -> BlochMatrix {
    let mut u = BlochMatrix::zeros(2 * n, 2 * n);
    let a = 0.37_f64;
    let phase = Complex::new(0.43_f64.cos(), 0.43_f64.sin());
    for i in 0..n {
        put(&mut u, i, i, Complex::new(a.cos(), 0.0));
        put(&mut u, n + i, n + i, Complex::new(a.cos(), 0.0));
        put(&mut u, i, n + i, -phase.conj() * a.sin());
        put(&mut u, n + i, i, phase * a.sin());
    }
    u
}
fn rotated(a: &BlochMatrix, u: &BlochMatrix) -> BlochMatrix {
    matmul(&matmul(u, a).unwrap(), &adjoint(u).unwrap()).unwrap()
}
fn delta(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}

#[test]
fn noncollinear_hf_and_lda_are_variational_and_spin_rotation_covariant() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.3, -0.1, 0.2),
    }]);
    let mb = basis(&mol);
    let n = mb.nao_sph;
    let geometry = PeriodicGeometry::new(
        Lattice::cubic(40.0),
        PeriodicBoundary::new([true, false, false]),
    )
    .unwrap();
    let o = options(geometry.boundary());
    let u = rotation(n);
    let theta = 0.41_f64;
    let grid = dft_rs::periodic::xc::PeriodicXcGrid::build_dimensional(
        &mol,
        &mb,
        &geometry,
        &geometry.k_points(o.scf.k_mesh).unwrap(),
        &o.scf.grid,
        o.grid_cutoff,
        o.basis_cutoff,
        dft_rs::periodic::xc::AoDerivativeOrder::Value,
    )
    .unwrap();
    let a = 0.7_f64;
    let b = 0.5_f64;
    let ns = (2.0 * a / std::f64::consts::PI).powf(0.75);
    let np = (2.0 * b / std::f64::consts::PI).powf(0.75) * (4.0 * b).sqrt();
    // Closed Gaussian density and saturated-spin Slater/VWN energies. The
    // explicit radial cutoff is the declared point/shell AO quadrature model.
    let mut local_exchange = 0.0;
    let mut local_correlation = 0.0;
    for (point, weight) in grid.points.iter().zip(&grid.weights) {
        let r = *point - mol.atoms[0].position;
        if r.norm() > o.basis_cutoff {
            continue;
        }
        let density = (theta.cos() * ns * (-a * r.norm2()).exp()).powi(2)
            + (theta.sin() * np * r.z * (-b * r.norm2()).exp()).powi(2);
        local_exchange -=
            weight * 0.75 * (6.0 / std::f64::consts::PI).cbrt() * density.powf(4.0 / 3.0);
        if density > 0.0 {
            let rs = (3.0 / (4.0 * std::f64::consts::PI * density)).cbrt();
            let x = rs.sqrt();
            let bp = 7.06042_f64;
            let cp = 18.0578_f64;
            let x0 = -0.32500_f64;
            let q = (4.0 * cp - bp * bp).sqrt();
            let xx = x * x + bp * x + cp;
            let xx0 = x0 * x0 + bp * x0 + cp;
            let angle = (q / (2.0 * x + bp)).atan();
            let correlation = 0.01554535
                * ((x * x / xx).ln() + 2.0 * bp / q * angle
                    - bp * x0 / xx0
                        * (((x - x0).powi(2) / xx).ln() + 2.0 * (bp + 2.0 * x0) / q * angle));
            local_correlation += weight * density * correlation;
        }
    }
    for name in ["hf", "slater", "svwn5"] {
        let spec = if name == "slater" {
            FunctionalSpec::Split {
                x: "slater".into(),
                c: "none".into(),
            }
        } else {
            FunctionalSpec::Combined(name.into())
        };
        let xc = resolve(&spec).unwrap();
        let backend = SpinorDftBackend::new(
            &mol,
            &mb,
            &geometry,
            &xc,
            &o,
            None,
            vec![BlochMatrix::zeros(2 * n, 2 * n)],
        )
        .unwrap();
        let (p, dp) = projector(theta, n);
        let fields = backend.evaluate(std::slice::from_ref(&p)).unwrap();
        let analytic = p.trace_product(&fields.fock[0]);
        assert!(analytic.is_finite());
        let derivative = dp.trace_product(&fields.fock[0]);
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut estimates = Vec::new();
        for h in steps {
            let ep = backend
                .evaluate(&[projector(theta + h, n).0])
                .unwrap()
                .energy;
            let em = backend
                .evaluate(&[projector(theta - h, n).0])
                .unwrap()
                .energy;
            let d = (ep - em) / (2.0 * h);
            estimates.push(d);
            errors.push((d - derivative).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = (4.0 * estimates[3] - estimates[2]) / 3.0;
        println!(
            "spinor {name}: derivative={derivative:.12e} steps={steps:?} \
            errors={errors:?} p={:.9} R2={:.10} Richardson={:.3e}",
            fit.exponent,
            fit.r_squared,
            (richardson - derivative).abs()
        );
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!((richardson - derivative).abs() < 1e-8);
        let pr = rotated(&p, &u);
        let fr = backend.evaluate(&[pr]).unwrap();
        println!(
            "spinor {name}: rotation energy={:.3e} Fock={:.3e} XC parts={:?}",
            (fields.energy - fr.energy).abs(),
            delta(&rotated(&fields.fock[0], &u), &fr.fock[0]),
            fields.xc_component_energies
        );
        assert!((fields.energy - fr.energy).abs() < 1e-12);
        assert!(delta(&rotated(&fields.fock[0], &u), &fr.fock[0]) < 1e-11);
        assert!((fields.energy - components_sum(&fields.energy_components)).abs() < 1e-14);
        assert!(
            (fields.energy_components[7] - fields.xc_component_energies.iter().sum::<f64>()).abs()
                < 1e-12
        );
        if name != "hf" {
            println!(
                "spinor {name}: closed Slater defect={:.3e}",
                (fields.xc_component_energies[0] - local_exchange).abs()
            );
            assert!((fields.xc_component_energies[0] - local_exchange).abs() < 1e-12);
        }
        if name == "svwn5" {
            println!(
                "spinor VWN closed term defect={:.3e}",
                (fields.xc_component_energies[1] - local_correlation).abs()
            );
            assert!((fields.xc_component_energies[1] - local_correlation).abs() < 1e-12);
        }
        let pc = pauli_components(&p).unwrap();
        assert!(delta(&p, &from_pauli(&pc).unwrap()) < 1e-14);
        // An independent full spin-index dense ERI oracle, with no Pauli map.
        if name == "hf" {
            let eri =
                dft_rs::integrals::eri_incore(&mb, dft_rs::integrals::CoulombKernel::Coulomb, 16)
                    .unwrap();
            let mut k = BlochMatrix::zeros(2 * n, 2 * n);
            for s in 0..2 {
                for t in 0..2 {
                    for i in 0..n {
                        for j in 0..n {
                            let mut value = Complex::ZERO;
                            for a in 0..n {
                                for b in 0..n {
                                    value += z(&p, s * n + a, t * n + b) * eri.get(i, a, j, b);
                                }
                            }
                            put(&mut k, s * n + i, t * n + j, value);
                        }
                    }
                }
            }
            let energy = -0.5 * p.trace_product(&k);
            println!(
                "spinor dense exchange defect={:.3e}",
                (energy - fields.energy_components[5]).abs()
            );
            assert!((energy - fields.energy_components[5]).abs() < 1e-12);
        }
    }
}
fn components_sum(v: &[f64; 10]) -> f64 {
    v.iter().sum()
}

#[test]
fn zero_and_partial_magnetization_match_collinear_provider_and_spin_rotation() {
    use dft_rs::periodic::scf_boundary::DimensionalScfBackend;
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol);
    let n = mb.nao_sph;
    let g = PeriodicGeometry::new(
        Lattice::cubic(40.0),
        PeriodicBoundary::new([true, false, false]),
    )
    .unwrap();
    let o = options(g.boundary());
    let xc = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let scalar = DimensionalScfBackend::new(&mol, &mb, &g, &xc, &o).unwrap();
    let spinor = SpinorDftBackend::new(
        &mol,
        &mb,
        &g,
        &xc,
        &o,
        None,
        vec![BlochMatrix::zeros(2 * n, 2 * n)],
    )
    .unwrap();
    for fraction in [0.0, 0.4] {
        let mut a = BlochMatrix::zeros(n, n);
        a.re[(0, 0)] = 1.0;
        let mut b = BlochMatrix::zeros(n, n);
        b.re[(0, 0)] = 1.0 - fraction;
        b.re[(1, 1)] = fraction;
        let mut pauli = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
        pauli[0] = a.clone();
        pauli[0].add_scaled(&b, 1.0);
        pauli[3] = a.clone();
        pauli[3].add_scaled(&b, -1.0);
        let p = from_pauli(&pauli).unwrap();
        let physical = spinor.evaluate(std::slice::from_ref(&p)).unwrap();
        let reference = scalar.evaluate(&[vec![a], vec![b]]).unwrap();
        let mut expected = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
        expected[0] = reference.fock[0][0].clone();
        expected[0].add_scaled(&reference.fock[1][0], 1.0);
        expected[3] = reference.fock[0][0].clone();
        expected[3].add_scaled(&reference.fock[1][0], -1.0);
        let f = from_pauli(&expected).unwrap();
        let energy = reference.electronic_energy + physical.energy_components[0];
        assert!((energy - physical.energy).abs() < 1e-12);
        assert!(delta(&f, &physical.fock[0]) < 1e-11);
        let u = rotation(n);
        let rotated_p = rotated(&p, &u);
        let r = spinor.evaluate(&[rotated_p]).unwrap();
        println!(
            "spinor beta p fraction={fraction}: collinear energy={:.3e} Fock={:.3e} \
            rotation={:.3e}",
            (energy - physical.energy).abs(),
            delta(&f, &physical.fock[0]),
            delta(&rotated(&f, &u), &r.fock[0])
        );
        assert!((r.energy - physical.energy).abs() < 1e-12);
        assert!(delta(&rotated(&f, &u), &r.fock[0]) < 1e-11);
    }
}

struct TwoLevel {
    s: Vec<BlochMatrix>,
    h: Vec<BlochMatrix>,
}
impl TwoLevel {
    fn new(lambda: f64) -> Self {
        let mut s = BlochMatrix::zeros(2, 2);
        s.re[(0, 0)] = 1.0;
        s.re[(1, 1)] = 1.0;
        let mut h = BlochMatrix::zeros(2, 2);
        h.re[(0, 0)] = 0.7;
        h.re[(1, 1)] = -0.7;
        h.re[(0, 1)] = lambda;
        h.re[(1, 0)] = lambda;
        h.im[(0, 1)] = -0.31 * lambda;
        h.im[(1, 0)] = 0.31 * lambda;
        Self {
            s: vec![s],
            h: vec![h],
        }
    }
}
impl SpinorEnergyBackend for TwoLevel {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.s
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.h
    }
    fn electrons(&self) -> f64 {
        1.0
    }
    fn evaluate(&self, p: &[BlochMatrix]) -> dft_rs::Result<SpinorFields> {
        let e = p[0].trace_product(&self.h[0]);
        let mut c = [0.0; 10];
        c[9] = e;
        Ok(SpinorFields {
            fock: self.h.clone(),
            energy: e,
            energy_components: c,
            xc_component_energies: Vec::new(),
            nuclear_soc_energy: 0.0,
            operator_converged: true,
            coulomb_operator_defect: 0.0,
            xc_grid_number_defect: 0.0,
            xc_diagnostics: None,
        })
    }
}
#[test]
fn unconstrained_spinor_solver_matches_dense_levels_and_four_step_energy_fit() {
    let config = SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-12,
        ..SpinorScfConfig::default()
    };
    let lambda = 0.38;
    let backend = TwoLevel::new(lambda);
    let r = run_spinor_scf(&backend, &config).unwrap();
    assert!(r.converged);
    let exact = -(0.7_f64.powi(2) + (1.0 + 0.31_f64.powi(2)) * lambda * lambda).sqrt();
    assert!((exact - r.fields.energy).abs() < 1e-14);
    assert!(
        r.particle_number_error < 1e-14
            && r.occupation_consistency < 1e-14
            && r.orbital_residual < 1e-14
    );
    assert!(r.density[0].re[(0, 1)].abs() > 0.1 && r.density[0].im[(0, 1)].abs() > 0.01);
    let mut dh = backend.h[0].clone();
    dh.re[(0, 0)] = 0.0;
    dh.re[(1, 1)] = 0.0;
    dh.scale(1.0 / lambda);
    let derivative = r.density[0].trace_product(&dh);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut errors = Vec::new();
    for h in steps {
        let plus = run_spinor_scf(&TwoLevel::new(lambda + h), &config).unwrap();
        let minus = run_spinor_scf(&TwoLevel::new(lambda - h), &config).unwrap();
        errors.push(((plus.fields.energy - minus.fields.energy) / (2.0 * h) - derivative).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!(
        "spinor two-level: steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
}

#[test]
fn spinor_single_chemical_potential_matches_mermin_oracle_and_four_step_fit() {
    use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
    let temperature = 100_000.0;
    let kt = temperature * BOLTZMANN_HARTREE_PER_K;
    let config = SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-12,
        occupation: OccupationScheme::FermiDirac {
            temperature_kelvin: temperature,
        },
        ..SpinorScfConfig::default()
    };
    let lambda = 0.38;
    let backend = TwoLevel::new(lambda);
    let r = run_spinor_scf(&backend, &config).unwrap();
    assert!(r.converged);
    assert!(r.particle_number_error < 1e-12);
    let alpha = 1.0 + 0.31_f64.powi(2);
    let delta = (0.7_f64.powi(2) + alpha * lambda * lambda).sqrt();
    let free = -2.0 * kt * (2.0 * (delta / (2.0 * kt)).cosh()).ln();
    assert!((r.free_energy - free).abs() < 1e-13);
    assert!(r.chemical_potential.unwrap().abs() < 1e-12);
    assert!(r.occupations[0].iter().all(|f| *f > 0.0 && *f < 1.0));
    let mut dh = backend.h[0].clone();
    dh.re[(0, 0)] = 0.0;
    dh.re[(1, 1)] = 0.0;
    dh.scale(1.0 / lambda);
    let derivative = r.density[0].trace_product(&dh);
    let exact = -alpha * lambda / delta * (delta / (2.0 * kt)).tanh();
    assert!((derivative - exact).abs() < 1e-13);
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut errors = Vec::new();
    for h in steps {
        let p = run_spinor_scf(&TwoLevel::new(lambda + h), &config).unwrap();
        let m = run_spinor_scf(&TwoLevel::new(lambda - h), &config).unwrap();
        errors.push(((p.free_energy - m.free_energy) / (2.0 * h) - derivative).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    println!("spinor Mermin: temperature={temperature} steps={steps:?} errors={errors:?} p={:.9} R2={:.10}",fit.exponent,fit.r_squared);
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
}

#[test]
fn pauli_soc_matches_closed_gaussian_and_kinetic_momentum_contact() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol);
    let n = mb.nao_sph;
    let g = PeriodicGeometry::new(Lattice::cubic(40.0), PeriodicBoundary::FULL_3D).unwrap();
    let points = [Vec3::zero()];
    let wave = g.reciprocal()[0];
    let scalar = Complex::new(0.23, 0.17);
    let spin = [
        Complex::new(0.01, 0.02),
        Complex::new(-0.03, 0.04),
        Complex::new(0.05, -0.01),
    ];
    let h = periodic_pauli_operators(&mb, &g, &points, 4.3, wave, scalar, spin, 1.0).unwrap();
    assert!(h[0].hermiticity_error() < 1e-14);
    let field = StaticPeriodicField {
        wave,
        scalar_potential: scalar,
        vector_potential: [
            Complex::ZERO,
            Complex::new(0.31, 0.11),
            Complex::new(-0.21, 0.07),
        ],
        amplitude: 0.4,
    };
    let contact = periodic_pauli_contact(&mb, &g, &points, 4.3, &field, 1.0).unwrap();
    assert!(contact[0].hermiticity_error() < 1e-14);
    assert!(contact[0]
        .re
        .as_slice()
        .iter()
        .chain(contact[0].im.as_slice())
        .any(|v| v.abs() > 1e-10));
    // SOC scalar-potential reversal reverses all spin-orbit matrix elements.
    let soc =
        periodic_pauli_operators(&mb, &g, &points, 4.3, wave, scalar, [Complex::ZERO; 3], 1.0)
            .unwrap();
    // Independent normalized one-center Gaussian integral. S=(2a/pi)^3/4
    // exp(-a r^2), Py=(2b/pi)^3/4 sqrt(4b) y exp(-b r^2).
    // <S|d_x V p_y|Py> = i 2q Im(Vq) Ns Np (pi/(a+b))^3/2
    // exp(-q^2/(4(a+b))) a/(a+b). Canonical p order is z,x,y.
    let a = 0.7_f64;
    let b = 0.5_f64;
    let sum = a + b;
    let q = wave.x;
    let c = 137.035_999_084_f64;
    let ns = (2.0 * a / std::f64::consts::PI).powf(0.75);
    let np = (2.0 * b / std::f64::consts::PI).powf(0.75) * (4.0 * b).sqrt();
    let sp = 2.0
        * q
        * scalar.im
        * ns
        * np
        * (std::f64::consts::PI / sum).powf(1.5)
        * (-q * q / (4.0 * sum)).exp()
        * a
        / sum
        / (4.0 * c * c);
    println!(
        "Pauli SOC closed Gaussian up-S/up-Py={:?} exact imaginary={sp:.12e}",
        z(&soc[0], 0, 3)
    );
    assert!((z(&soc[0], 0, 3) - Complex::new(0.0, sp)).abs() < 1e-14);
    assert!((z(&soc[0], 0, n + 1) - Complex::new(-sp, 0.0)).abs() < 1e-14);
    // Independent closed Gaussian expectation of the mixed grad(V) x A
    // contact, including q=0 and both +/-2q components.
    let mut contact_z = Complex::ZERO;
    for sv in [1.0, -1.0] {
        for sa in [1.0, -1.0] {
            let v = if sv > 0.0 { scalar } else { scalar.conj() };
            let ay = if sa > 0.0 {
                field.vector_potential[1]
            } else {
                field.vector_potential[1].conj()
            };
            let density = (-((sv + sa) * q).powi(2) / (8.0 * a)).exp();
            contact_z += Complex::new(0.0, sv * q) * v * ay * (density / (2.0 * c * c));
        }
    }
    println!(
        "Pauli contact closed Gaussian={contact_z:?} production={:?}",
        z(&contact[0], 0, 0)
    );
    assert!((contact_z - z(&contact[0], 0, 0)).abs() < 1e-14);
    let reversed = periodic_pauli_operators(
        &mb,
        &g,
        &points,
        4.3,
        wave,
        -scalar,
        [Complex::ZERO; 3],
        1.0,
    )
    .unwrap();
    let mut sum = soc[0].clone();
    sum.add_scaled(&reversed[0], 1.0);
    assert!(sum
        .re
        .as_slice()
        .iter()
        .chain(sum.im.as_slice())
        .all(|v| v.abs() < 1e-15));
    // At q=0 the scalar gradient vanishes: general Zeeman directions survive.
    let zero =
        periodic_pauli_operators(&mb, &g, &points, 4.3, Vec3::zero(), scalar, spin, 1.0).unwrap();
    let overlap = dft_rs::integrals::overlap_matrix(&mb);
    let mut expected = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
    for axis in 0..3 {
        for i in 0..n {
            for j in 0..n {
                expected[axis + 1].re[(i, j)] = 2.0 * spin[axis].re * overlap[(i, j)];
            }
        }
    }
    assert!(delta(&zero[0], &from_pauli(&expected).unwrap()) < 1e-13);
    // A real scalar SOC is spinful time-reversal even at Gamma.
    for s in 0..2 {
        for t in 0..2 {
            for i in 0..n {
                for j in 0..n {
                    let sign = if s == t { 1.0 } else { -1.0 };
                    assert!(
                        (z(&soc[0], s * n + i, t * n + j)
                            - z(&soc[0], (1 - s) * n + i, (1 - t) * n + j).conj() * sign)
                            .abs()
                            < 1e-13
                    );
                }
            }
        }
    }
    let orbitals = generalized_orbitals(
        &soc[0],
        &spin_diagonal(&BlochMatrix::from_pair((
            overlap,
            dft_rs::linalg::Matrix::zeros(n, n),
        ))),
        1e-7,
    )
    .unwrap();
    for pair in orbitals.energies.chunks_exact(2) {
        assert!((pair[0] - pair[1]).abs() < 1e-12);
    }
}

#[test]
fn physical_spinor_hf_scf_retains_soc_coherence_and_variational_field_fit() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.3, -0.1, 0.2),
    }]);
    let mb = basis(&mol);
    let g = PeriodicGeometry::new(
        Lattice::cubic(40.0),
        PeriodicBoundary::new([true, false, false]),
    )
    .unwrap();
    let o = options(g.boundary());
    let q = g.reciprocal()[0];
    let points = g.k_points(o.scf.k_mesh).unwrap();
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let scalar = Complex::new(0.23, 0.17);
    let spin = [
        Complex::new(0.03, 0.01),
        Complex::new(0.04, -0.03),
        Complex::new(0.05, 0.02),
    ];
    let first = periodic_pauli_operators(&mb, &g, &points, 4.3, q, scalar, spin, 1.0).unwrap();
    let config = SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-11,
        ..SpinorScfConfig::default()
    };
    let calculate = |amplitude| {
        let field = StaticPeriodicField {
            wave: q,
            scalar_potential: scalar,
            vector_potential: [Complex::ZERO; 3],
            amplitude,
        };
        let mut operator = first.clone();
        for h in &mut operator {
            h.scale(amplitude);
        }
        let backend = SpinorDftBackend::new(&mol, &mb, &g, &xc, &o, Some(&field), operator)
            .unwrap()
            .with_nuclear_soc(1.0)
            .unwrap();
        let r = run_spinor_scf(&backend, &config).unwrap();
        assert!(
            r.converged,
            "spinor residual={} density={} occupation={}",
            r.orbital_residual, r.density_change, r.occupation_consistency
        );
        let orbital = field.operators(&mb, &g, &points, 4.3).unwrap();
        let derivative = backend
            .amplitude_derivative(&r.density, &orbital.first_derivative, &first)
            .unwrap();
        (r, derivative)
    };
    let (reference, derivative) = calculate(0.4);
    println!(
        "physical SOC spinor E={:.12e} orbital={:.3e} occupation={:.3e} number={:.3e}",
        reference.fields.energy,
        reference.orbital_residual,
        reference.occupation_consistency,
        reference.particle_number_error
    );
    let n = mb.nao_sph;
    assert!((0..n).any(|i| (0..n).any(|j| z(&reference.density[0], i, n + j).abs() > 1e-3)));
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut errors = Vec::new();
    let mut estimates = Vec::new();
    for h in steps {
        let (p, _) = calculate(0.4 + h);
        let (m, _) = calculate(0.4 - h);
        let estimate = (p.fields.energy - m.fields.energy) / (2.0 * h);
        estimates.push(estimate);
        errors.push((estimate - derivative).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    let richardson = (4.0 * estimates[3] - estimates[2]) / 3.0;
    println!(
        "physical SOC spinor: steps={steps:?} errors={errors:?} p={:.9} \
        R2={:.10} Richardson={:.3e}",
        fit.exponent,
        fit.r_squared,
        (richardson - derivative).abs()
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    assert!((richardson - derivative).abs() < 1e-8);
}

#[test]
fn nuclear_soc_atomic_p_splitting_matches_independent_radial_oracle() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol);
    let n = mb.nao_sph;
    let g = PeriodicGeometry::new(Lattice::cubic(40.0), PeriodicBoundary::FULL_3D).unwrap();
    for width in [0.0, 0.3, 0.8] {
        let mut o = options(g.boundary());
        o.nuclear_widths = vec![width];
        let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
        let base = SpinorDftBackend::new(
            &mol,
            &mb,
            &g,
            &xc,
            &o,
            None,
            vec![BlochMatrix::zeros(2 * n, 2 * n)],
        )
        .unwrap();
        let core = base.initial_fock()[0].clone();
        let backend = base.with_nuclear_soc(1.0).unwrap();
        let mut soc = backend.initial_fock()[0].clone();
        soc.add_scaled(&core, -1.0);
        let bands = generalized_orbitals(&soc, &backend.overlap()[0], 1e-7).unwrap();
        // Normalized p radial expectation <r^-3>=4(2b)^3/2/(3 sqrt(pi)).
        // Cubic periodic nuclear/background Green: grad(V)/r includes
        // -4pi/(3 volume). Higher cubic harmonics cannot couple two l=1 AOs.
        let c = 137.035_999_084_f64;
        let b = 0.5_f64;
        let lambda = ((4.0 / (3.0 * std::f64::consts::PI.sqrt()))
            * (2.0 * b).powf(1.5)
            * (1.0 + 2.0 * b * width * width).powf(-1.5)
            - 4.0 * std::f64::consts::PI / (3.0 * 40.0_f64.powi(3)))
            / (4.0 * c * c);
        let expected = [
            -2.0 * lambda,
            -2.0 * lambda,
            0.0,
            0.0,
            lambda,
            lambda,
            lambda,
            lambda,
        ];
        let defect = bands
            .energies
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!(
        "nuclear SOC width={width} p splitting={:?} lambda={lambda:.12e} closed radial defect={defect:.3e}",
        bands.energies
    );
        assert!(defect < 1e-12);
        let zero = backend.with_nuclear_soc(0.0).unwrap();
        // Zero scale is an explicit model selection, not a numerical denominator floor.
        assert!(zero.initial_fock()[0].hermiticity_error() < 1e-14);
        assert!(delta(&zero.initial_fock()[0], &core) < 1e-14);
    }
}

#[test]
fn common_field_geometry_uses_only_the_physical_periodic_rank() {
    use dft_rs::driver::{periodic::dimensional_geometry_of, System};
    for (pbc, rows, measure) in [
        (
            [true, false, false],
            [[8.0, 0.0, 0.0], [0.0; 3], [0.0; 3]],
            8.0,
        ),
        (
            [false, true, false],
            [[0.0; 3], [0.0, 9.0, 0.0], [0.0; 3]],
            9.0,
        ),
        (
            [true, true, false],
            [[8.0, 0.0, 0.0], [0.3, 9.0, 0.0], [0.0; 3]],
            72.0,
        ),
        (
            [true, false, true],
            [[8.0, 0.0, 0.0], [0.0; 3], [0.3, 0.0, 9.0]],
            72.0,
        ),
    ] {
        let system = System {
            cell: Some(rows),
            pbc,
            ..System::default()
        };
        let geometry = dimensional_geometry_of(&system).unwrap();
        assert_eq!(geometry.dimension(), pbc.iter().filter(|&&v| v).count());
        assert!((geometry.measure() - measure).abs() < 1e-12);
        for a in 0..3 {
            for b in 0..3 {
                let expected = if pbc[a] && a == b {
                    2.0 * std::f64::consts::PI
                } else {
                    0.0
                };
                assert!(
                    (geometry.lattice().a[a].dot(geometry.reciprocal()[b]) - expected).abs()
                        < 1e-12
                );
            }
        }
        let invalid = System {
            pbc: [true; 3],
            ..system
        };
        assert!(dimensional_geometry_of(&invalid).is_err());
    }
}
