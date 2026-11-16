// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Continuum London operators and variational magnetic-torus references.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/crystals.rs"]
mod crystals;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::basis::{aos::cart_to_sph, cart_components, BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::dfpt::orbitals::{adjoint, matmul};
use dft_rs::math::Vec3;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_cell::*;
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
fn z(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn put(m: &mut BlochMatrix, i: usize, j: usize, v: Complex) {
    m.re[(i, j)] = v.re;
    m.im[(i, j)] = v.im;
}
fn gap(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
fn basis(mol: &Molecule, p: bool) -> MolBasis {
    let mut set = BasisSet::new("uniform-flux-oracle");
    let mut shells = vec![Shell {
        l: 0,
        exps: vec![0.7],
        coefs: vec![1.0],
    }];
    if p {
        shells.push(Shell {
            l: 1,
            exps: vec![0.5],
            coefs: vec![1.0],
        });
    }
    set.insert(ElementBasis {
        z: mol.atoms[0].z,
        shells,
    });
    MolBasis::build(mol, &set).unwrap()
}
fn cell(lattice: Lattice, flux: [i32; 3], origin: Vec3) -> MagneticCell {
    let v = lattice.volume();
    let a = lattice.a;
    let b = (a[2] * (flux[0] as f64) + a[0] * (flux[1] as f64) + a[1] * (flux[2] as f64))
        * (2.0 * std::f64::consts::PI / v);
    MagneticCell::new(
        PeriodicGeometry::new(lattice, PeriodicBoundary::FULL_3D).unwrap(),
        b,
        origin,
    )
    .unwrap()
}
fn options() -> MagneticScfOptions {
    let mut o = MagneticScfOptions {
        basis_cutoff: 12.0,
        reciprocal_cutoff: 16.0,
        grid_order: 40,
        grid_cutoff: 8.0,
        ..Default::default()
    };
    o.electrostatics.alpha = Some(0.55);
    o.electrostatics.real_cutoff = Some(16.0);
    o.electrostatics.reciprocal_cutoff = Some(8.0);
    o
}

#[test]
fn magnetic_gradient_and_meta_functionals_have_variational_spinor_focks() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.18, -0.23, 0.33),
    }]);
    let mb = basis(&mol, false);
    let lattice = Lattice::new(
        Vec3::new(8.0, 0.13, -0.21),
        Vec3::new(0.31, 7.8, 0.26),
        Vec3::new(-0.36, 0.27, 8.3),
    );
    let origin = Vec3::new(0.17, -0.38, 0.23);
    let mut p = BlochMatrix::zeros(2, 2);
    put(&mut p, 0, 0, Complex::new(0.63, 0.0));
    put(&mut p, 1, 1, Complex::new(0.37, 0.0));
    put(&mut p, 0, 1, Complex::new(0.11, 0.13));
    put(&mut p, 1, 0, Complex::new(0.11, -0.13));
    let directions = [
        (0, 0, Complex::ONE),
        (1, 1, Complex::ONE),
        (0, 1, Complex::ONE),
        (0, 1, Complex::new(0.0, 1.0)),
    ]
    .map(|(a, b, value)| {
        let mut d = BlochMatrix::zeros(2, 2);
        put(&mut d, a, b, value);
        put(&mut d, b, a, value.conj());
        d
    });
    let mut options = options();
    options.basis_cutoff = 5.0;
    options.grid_cutoff = 5.0;
    options.nuclear_widths = vec![0.4];
    // This derivative check differentiates a fixed spatial quadrature.
    // Separate SCF fixtures verify the tighter production quadrature gate.
    options.operator_tolerance = 1e-7;
    let mut failures = Vec::new();
    for name in ["blyp", "m06l"] {
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let backend = |gauge| {
            MagneticDftBackend::new(
                &mol,
                &mb,
                cell(lattice.clone(), [1, 0, 0], gauge),
                &functional,
                options.clone(),
                None,
            )
            .unwrap()
        };
        let first = backend(origin);
        let fields = first.evaluate(std::slice::from_ref(&p)).unwrap();
        let grid_order = first.grid_order();
        let shifted = backend(origin + Vec3::new(0.27, 0.16, -0.22))
            .evaluate(std::slice::from_ref(&p))
            .unwrap();
        let energy_defect = (fields.energy - shifted.energy).abs();
        let fock_defect = gap(&fields.fock[0], &shifted.fock[0]);
        println!("{name} magnetic gauge defects energy={energy_defect:.4e} Fock={fock_defect:.4e}");
        assert!(energy_defect < 1e-11 && fock_defect < 1e-11);
        let names = [
            "diagonal alpha",
            "diagonal beta",
            "real coherence",
            "imaginary coherence",
        ]
        .map(|component| format!("{name} magnetic Fock {component}"));
        stencils::fit_components_staged(
            &names,
            &[
                ([0.04, 0.03, 0.02, 0.014], stencils::Stencil::Second),
                ([0.012, 0.009, 0.006, 0.004], stencils::Stencil::Second),
                ([0.03, 0.02, 0.014, 0.01], stencils::Stencil::Fourth),
            ],
            |steps, stencil| {
                directions
                    .iter()
                    .map(|d| {
                        let derivative = fields.fock[0].trace_product(d);
                        steps.map(|h| {
                            let energy = |step| {
                                let mut trial = p.clone();
                                trial.add_scaled(d, step);
                                let energy = first.evaluate(&[trial]).unwrap().energy;
                                assert_eq!(first.grid_order(), grid_order);
                                energy
                            };
                            let (outer_plus, outer_minus) = if stencil == stencils::Stencil::Fourth
                            {
                                (energy(2.0 * h), energy(-2.0 * h))
                            } else {
                                (0.0, 0.0)
                            };
                            (stencil.first(energy(h), energy(-h), outer_plus, outer_minus, h)
                                - derivative)
                                .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("complete magnetic spin-density Fock", &failures);
}
#[test]
fn london_single_gaussian_matches_closed_kinetic_and_momentum() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.2, -0.3, 0.4),
    }]);
    let mb = basis(&mol, false);
    let c = cell(Lattice::cubic(8.0), [1, 1, -1], Vec3::new(1.2, -0.7, 0.5));
    let overlap = c.fourier(&mb, 4.3, Vec3::zero()).unwrap();
    let kinetic = c.kinetic(&mb, 4.3).unwrap();
    assert!((overlap.re[(0, 0)] - 1.0).abs() < 1e-14);
    let exact = 1.5 * 0.7 + c.field.norm2() / (16.0 * 0.7);
    println!(
        "closed London kinetic {:.12e} defect {:.3e}",
        kinetic.re[(0, 0)],
        (kinetic.re[(0, 0)] - exact).abs()
    );
    assert!((kinetic.re[(0, 0)] - exact).abs() < 1e-13);
    let b = c.geometry.reciprocal();
    let wave = b[0] - b[1] + b[2];
    let norm = (-wave.norm2() / (8.0 * 0.7)).exp();
    let ft = Complex::new(
        wave.dot(mol.atoms[0].position).cos(),
        wave.dot(mol.atoms[0].position).sin(),
    ) * norm;
    assert!((z(&c.fourier(&mb, 4.3, wave).unwrap(), 0, 0) - ft).abs() < 1e-14);
    let cross = c.field.cross(wave);
    for a in 0..3 {
        let expected = ft * Complex::new(-wave.get(a) * 0.5, cross.get(a) / (8.0 * 0.7));
        assert!((z(&c.momentum(&mb, 4.3, wave, a).unwrap(), 0, 0) - expected).abs() < 1e-13);
    }
    assert!(c.fourier(&mb, 4.3, Vec3::new(0.1, 0.2, 0.3)).is_err());
}
#[test]
fn seven_crystal_cells_have_magnetic_boundaries_and_gauge_covariance() {
    let rows = crystals::rows();
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.2, 0.3, -0.4),
    }]);
    let mb = basis(&mol, true);
    let r = Vec3::new(0.13, -0.24, 0.31);
    let origin = Vec3::new(0.4, -0.2, 0.6);
    for (family, rows) in rows.into_iter().enumerate() {
        let vectors = rows.map(|r| Vec3::new(r[0], r[1], r[2]));
        let lattice = Lattice::new(vectors[0], vectors[1], vectors[2]);
        let c = cell(lattice.clone(), [1, 1, -1], origin);
        let shifted = cell(lattice, [1, 1, -1], origin + Vec3::new(0.2, 0.4, -0.3));
        let points = [
            r,
            r + c.geometry.lattice().a[0],
            r + c.geometry.lattice().a[1],
            r + c.geometry.lattice().a[2],
        ];
        let phi = c.orbital_values(&mb, &points, 10.0).unwrap();
        let other = shifted.orbital_values(&mb, &points, 10.0).unwrap();
        let mut boundary = 0.0_f64;
        let mut gauge = 0.0_f64;
        for a in 0..3 {
            let theta = -0.5 * c.field.cross(c.geometry.lattice().a[a]).dot(r + origin);
            let phase = Complex::new(theta.cos(), theta.sin());
            for i in 0..mb.nao_sph {
                boundary = boundary.max((z(&phi, a + 1, i) - phase * z(&phi, 0, i)).abs());
            }
        }
        for (g, point) in points.iter().enumerate() {
            let theta = 0.5 * c.field.cross(shifted.gauge_origin - origin).dot(*point);
            let phase = Complex::new(theta.cos(), theta.sin());
            for i in 0..mb.nao_sph {
                gauge = gauge.max((z(&other, g, i) - phase * z(&phi, g, i)).abs());
            }
        }
        let wave = c.geometry.reciprocal()[0] + c.geometry.reciprocal()[1];
        let f = c.fourier(&mb, 10.0, wave).unwrap();
        let m = c.fourier(&mb, 10.0, wave * -1.0).unwrap();
        let hermitian = gap(&adjoint(&f).unwrap(), &m);
        println!("magnetic crystal {family}: boundary={boundary:.3e} gauge={gauge:.3e} Fourier-adjoint={hermitian:.3e}");
        assert!(boundary < 1e-12 && gauge < 1e-12 && hermitian < 1e-12);
        assert!(
            gap(
                &c.kinetic(&mb, 10.0).unwrap(),
                &shifted.kinetic(&mb, 10.0).unwrap()
            ) < 1e-14
        );
    }
    let c = cell(Lattice::cubic(5.0), [1, 0, 0], Vec3::zero());
    assert!(MagneticCell::new(c.geometry, c.field * 0.5, Vec3::zero()).is_err());
}

#[test]
fn magnetic_collocation_matches_primitive_operator_through_i_shells() {
    let center = Vec3::new(0.29, -0.41, 0.33);
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: center,
    }]);
    let mut set = BasisSet::new("london-collocation-oracle");
    set.insert(ElementBasis {
        z: 1,
        shells: (0..=6)
            .map(|l| Shell {
                l,
                exps: vec![0.42, 1.13],
                coefs: vec![0.73, -0.21],
            })
            .collect(),
    });
    let basis = MolBasis::build(&mol, &set).unwrap();
    let lattice = Lattice::new(
        Vec3::new(6.8, 0.24, -0.31),
        Vec3::new(0.71, 5.9, 0.42),
        Vec3::new(-0.18, 0.66, 6.3),
    );
    let magnetic = cell(lattice, [1, -1, 1], Vec3::new(0.31, -0.45, 0.28))
        .with_bloch_wave(Vec3::new(0.08, 0.13, -0.17))
        .unwrap();
    let mut points: Vec<_> = (0..28)
        .map(|g| {
            let t = g as f64 * 0.23 + 0.17;
            Vec3::new(t + 0.31, (1.37 * t).sin() * 1.81, (0.83 * t).cos())
        })
        .collect();
    points.extend([
        center,
        center + Vec3::new(0.0, 0.37, -0.28),
        center + magnetic.geometry.lattice().a[0],
    ]);
    let cutoff = 7.4;
    let (value, [px, py, pz]) = magnetic
        .orbital_values_with_momentum(&basis, &points, cutoff)
        .unwrap();
    let matrices = [&value, &px, &py, &pz];
    let mut shell_errors = [[0.0_f64; 4]; 7];
    for (g, &point) in points.iter().enumerate() {
        let potential = magnetic.field.cross(point - magnetic.gauge_origin) * 0.5;
        for shell in &basis.shells {
            let transform = cart_to_sph(shell.l);
            let components = cart_components(shell.l);
            let mut reference = vec![[Complex::ZERO; 4]; transform.cols];
            for image in magnetic
                .geometry
                .nearby_translation_indices(shell.center - point, cutoff)
                .unwrap()
            {
                let moved = shell.center + magnetic.geometry.lattice().translation(image);
                let delta = point - moved;
                let wave = magnetic.london_wave(moved);
                let theta = wave.dot(point);
                let phase = magnetic.image_phase(shell.center, image)
                    * Complex::new(theta.cos(), theta.sin());
                let primitive = |[x, y, z]: [i32; 3], exponent: f64, coefficient: f64| {
                    coefficient
                        * (-exponent * delta.norm2()).exp()
                        * delta.x.powi(x)
                        * delta.y.powi(y)
                        * delta.z.powi(z)
                };
                for (cart, &(x, y, z)) in components.iter().enumerate() {
                    let angular = [x, y, z];
                    for (&exponent, &coefficient) in shell.exps.iter().zip(&shell.coefs) {
                        let base = primitive(angular, exponent, coefficient);
                        let mut operator = [Complex::from(base); 4];
                        // Apply p+A to each phase-modulated primitive using
                        // separately raised and lowered Cartesian monomials.
                        for (axis, value) in operator[1..].iter_mut().enumerate() {
                            let mut raised = angular;
                            raised[axis] += 1;
                            let mut momentum =
                                2.0 * exponent * primitive(raised, exponent, coefficient);
                            if angular[axis] > 0 {
                                let mut lowered = angular;
                                lowered[axis] -= 1;
                                momentum -= angular[axis] as f64
                                    * primitive(lowered, exponent, coefficient);
                            }
                            *value = Complex::new((potential + wave).get(axis) * base, momentum);
                        }
                        for (spherical, channels) in reference.iter_mut().enumerate() {
                            let weight = transform[(cart, spherical)];
                            for (sum, value) in channels.iter_mut().zip(operator) {
                                *sum += phase * value * weight;
                            }
                        }
                    }
                }
            }
            for (orbital, channels) in reference.into_iter().enumerate() {
                for (channel, (matrix, expected)) in matrices.iter().zip(channels).enumerate() {
                    let actual = z(matrix, g, shell.sph_off + orbital);
                    let error = (actual - expected).abs();
                    assert!(
                        error < 3e-14,
                        "point {g}, l={}, AO {orbital}, channel {channel}: error {error:e}",
                        shell.l
                    );
                    let max_error = &mut shell_errors[shell.l as usize][channel];
                    *max_error = max_error.max(error);
                }
            }
        }
    }
    println!(
        "London collocation: {} complex values, maximum differences [value, px, py, pz] by shell: {shell_errors:?}",
        points.len() * basis.nao_sph * matrices.len(),
    );
}

#[test]
fn magnetic_kinetic_momentum_has_independent_four_step_spatial_fits() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::new(0.2, 0.3, -0.4),
    }]);
    let mb = basis(&mol, true);
    let c = cell(Lattice::cubic(5.0), [1, 1, -1], Vec3::new(0.4, -0.2, 0.6));
    let points = [Vec3::new(0.13, -0.24, 0.31), Vec3::new(4.8, 0.1, -0.3)];
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let base = c.orbital_values(&mb, &points, 10.0).unwrap();
    for axis in 0..3 {
        let analytic = c.orbital_momentum_values(&mb, &points, 10.0, axis).unwrap();
        let mut errors = Vec::new();
        let mut values = Vec::new();
        for h in steps {
            let displacement = match axis {
                0 => Vec3::new(h, 0.0, 0.0),
                1 => Vec3::new(0.0, h, 0.0),
                _ => Vec3::new(0.0, 0.0, h),
            };
            let plus = c
                .orbital_values(&mb, &points.map(|p| p + displacement), 10.0)
                .unwrap();
            let minus = c
                .orbital_values(&mb, &points.map(|p| p - displacement), 10.0)
                .unwrap();
            let mut estimate = analytic.clone();
            for (g, &r) in points.iter().enumerate() {
                let a = c.field.cross(r - c.gauge_origin) * 0.5;
                for i in 0..mb.nao_sph {
                    let derivative = (z(&plus, g, i) - z(&minus, g, i)) / (2.0 * h);
                    put(
                        &mut estimate,
                        g,
                        i,
                        Complex::new(derivative.im, -derivative.re) + z(&base, g, i) * a.get(axis),
                    );
                }
            }
            errors.push(gap(&estimate, &analytic));
            values.push(estimate);
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let mut rich = values[3].clone();
        rich.scale(4.0 / 3.0);
        rich.add_scaled(&values[2], -1.0 / 3.0);
        println!(
            "magnetic momentum axis={axis}: steps={steps:?} errors={errors:?} \
            p={:.9} R2={:.10} Richardson={:.3e}",
            fit.exponent,
            fit.r_squared,
            gap(&rich, &analytic)
        );
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!(gap(&rich, &analytic) < 1e-8);
    }
}
fn density(theta: f64, n: usize) -> (BlochMatrix, BlochMatrix) {
    let mut c = BlochMatrix::zeros(2 * n, 1);
    let mut d = c.clone();
    put(&mut c, 0, 0, Complex::from(theta.cos()));
    put(&mut c, n + 1, 0, Complex::new(0.0, theta.sin()));
    put(&mut d, 0, 0, Complex::from(-theta.sin()));
    put(&mut d, n + 1, 0, Complex::new(0.0, theta.cos()));
    let p = matmul(&c, &adjoint(&c).unwrap()).unwrap();
    let mut dp = matmul(&d, &adjoint(&c).unwrap()).unwrap();
    dp.add_scaled(&matmul(&c, &adjoint(&d).unwrap()).unwrap(), 1.0);
    (p, dp)
}
#[test]
fn magnetic_energy_has_four_step_variational_spinor_fits() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol, true);
    for (name, exchange) in [("hf", None), ("svwn5", None), ("hf", Some([0.4, 0.3, 0.5]))] {
        let mut xc = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        if let Some(p) = exchange {
            xc.hybrid = dft_rs::xc::HybridParams {
                cx: p[0],
                clr: p[1],
                omega: p[2],
            };
        }
        let name = if exchange.is_some() {
            "weighted_range_exchange"
        } else {
            name
        };
        let c = cell(Lattice::cubic(8.0), [1, 1, -1], Vec3::zero());
        let backend = MagneticDftBackend::new(&mol, &mb, c, &xc, options(), None).unwrap();
        let (p, dp) = density(0.37, mb.nao_sph);
        let reference = backend.evaluate(std::slice::from_ref(&p)).unwrap();
        let exact = dp.trace_product(&reference.fock[0]);
        let history_before_diagnostics = backend.grid_refinement_history();
        let grid_defects = backend.grid_diagnostics(&p).unwrap();
        assert_eq!(
            backend.grid_refinement_history(),
            history_before_diagnostics
        );
        println!("magnetic {name} independent grid energy/Fock/number defects {grid_defects:?}");
        println!(
            "magnetic {name} retained grid history {:?}",
            backend.grid_refinement_history()
        );
        assert!(reference.operator_converged);
        if !xc.parts.is_empty() {
            let history = backend.grid_refinement_history();
            assert!(history[0][2] > options().operator_tolerance);
            assert!(backend.grid_order() > options().grid_order);
            assert!(grid_defects
                .iter()
                .all(|d| *d <= options().operator_tolerance));
        }
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut values = Vec::new();
        for h in steps {
            let plus = backend
                .evaluate(&[density(0.37 + h, mb.nao_sph).0])
                .unwrap();
            let minus = backend
                .evaluate(&[density(0.37 - h, mb.nao_sph).0])
                .unwrap();
            assert!(plus.operator_converged && minus.operator_converged);
            let value = (plus.energy - minus.energy) / (2.0 * h);
            values.push(value);
            errors.push((value - exact).abs());
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let rich = (4.0 * values[3] - values[2]) / 3.0;
        println!(
            "magnetic {name} density fit: steps={steps:?} errors={errors:?} p={:.9} \
            R2={:.10} Richardson={:.3e} Coulomb={:.3e} grid={:.3e}",
            fit.exponent,
            fit.r_squared,
            (rich - exact).abs(),
            reference.coulomb_operator_defect,
            reference.xc_grid_number_defect
        );
        assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
        assert!((rich - exact).abs() < 1e-8);
        assert!((reference.energy - reference.energy_components.iter().sum::<f64>()).abs() < 1e-14);
    }
}
#[test]
fn genuine_uniform_flux_scf_includes_nuclear_contact_and_self_interaction_cancellation() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol, false);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    for width in [0.0, 0.3, 0.8] {
        let c = cell(Lattice::cubic(8.0), [1, 1, -1], Vec3::new(1.0, -2.0, 0.5));
        let mut o = options();
        o.nuclear_widths = vec![width];
        let backend = MagneticDftBackend::new(&mol, &mb, c, &xc, o, None).unwrap();
        let cfg = SpinorScfConfig {
            energy_tolerance: 1e-13,
            density_tolerance: 1e-12,
            orbital_tolerance: 1e-11,
            ..Default::default()
        };
        let report = run_spinor_scf(&backend, &cfg).unwrap();
        println!(
            "uniform SCF convergence scf={} backend={} Coulomb={:.3e} image={:.3e}",
            report.converged,
            report.fields.operator_converged,
            report.fields.coulomb_operator_defect,
            backend.image_operator_defect
        );
        assert!(report.converged && report.fields.operator_converged);
        let c = 137.035_999_084_f64;
        let a = 0.7_f64;
        let radial = (2.0 * (2.0 * a / std::f64::consts::PI).sqrt()
            / (1.0 + 2.0 * a * width * width).powf(1.5)
            - std::f64::consts::PI / (a * 8.0_f64.powi(3)))
            / (12.0 * c * c);
        let expected = -backend.cell.field.norm() * radial;
        let error = (report.fields.nuclear_soc_energy - expected).abs();
        println!(
            "uniform flux width={width}: E={:.12e} SOC={:.12e} \
            closed-contact-defect={error:.3e} residual={:.3e} number={:.3e} \
            cutoff={:.3e} J+K={:.3e}",
            report.fields.energy,
            report.fields.nuclear_soc_energy,
            report.orbital_residual,
            report.particle_number_error,
            report.fields.coulomb_operator_defect,
            report.fields.energy_components[4] + report.fields.energy_components[5]
        );
        assert!(error < 1e-11);
        assert!(
            (report.fields.energy_components[4] + report.fields.energy_components[5]).abs() < 1e-12
        );
        assert!(report.particle_number_error < 1e-12 && report.occupation_consistency < 1e-12);
        assert!(
            z(&report.density[0], 0, 1).abs() > 0.1 && report.density[0].im[(0, 1)].abs() > 0.1
        );
        assert!(
            (report.fields.energy - report.fields.energy_components.iter().sum::<f64>()).abs()
                < 1e-14
        );
        let points = [Vec3::new(0.2, 0.3, -0.1), Vec3::new(-0.25, 0.1, 0.2)];
        let sampled = backend
            .sample_particle_current(&report.density[0], &points)
            .unwrap();
        let s = backend.overlap()[0].re[(0, 0)];
        for (g, &r) in points.iter().enumerate() {
            let rho = (2.0 * a / std::f64::consts::PI).powf(1.5) * (-2.0 * a * r.norm2()).exp() / s;
            let current = backend.cell.field.cross(r) * (0.5 * rho);
            assert!((sampled.number_density[g] - rho).abs() < 1e-12);
            for axis in 0..3 {
                assert!((sampled.kinetic_current[g][axis] - current.get(axis)).abs() < 1e-12);
            }
            let spin = backend.cell.field * (-rho / backend.cell.field.norm());
            let spin_curl = (r * (-4.0 * a * rho))
                .cross(backend.cell.field * (-1.0 / backend.cell.field.norm()))
                * 0.5;
            let gradient = Vec3::new(
                sampled.source_potential_gradient[g][0],
                sampled.source_potential_gradient[g][1],
                sampled.source_potential_gradient[g][2],
            );
            let anomalous = spin.cross(gradient) / (4.0 * 137.035_999_084_f64.powi(2));
            for axis in 0..3 {
                assert!((sampled.spin_density[g][axis] - spin.get(axis)).abs() < 1e-12);
                assert!(
                    (sampled.spin_magnetization_current[g][axis] - spin_curl.get(axis)).abs()
                        < 1e-12
                );
                assert!((sampled.nuclear_soc_current[g][axis] - anomalous.get(axis)).abs() < 1e-12);
            }
        }
        if width == 0.3 {
            spin_curl_acceptance(&backend, &report.density[0]);
        }
    }
}

fn spin_curl_acceptance(backend: &MagneticDftBackend<'_>, density: &BlochMatrix) {
    let r = Vec3::new(0.2, 0.3, -0.1);
    let exact = backend
        .sample_particle_current(density, &[r])
        .unwrap()
        .spin_magnetization_current[0];
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mut errors = Vec::new();
    let mut estimates = Vec::new();
    for h in steps {
        let mut gradient = [[0.0; 3]; 3];
        for axis in 0..3 {
            let step = match axis {
                0 => Vec3::new(h, 0.0, 0.0),
                1 => Vec3::new(0.0, h, 0.0),
                _ => Vec3::new(0.0, 0.0, h),
            };
            let displaced = backend
                .sample_particle_current(density, &[r + step, r - step])
                .unwrap();
            for component in 0..3 {
                gradient[component][axis] = (displaced.spin_density[0][component]
                    - displaced.spin_density[1][component])
                    / (2.0 * h);
            }
        }
        let estimate = std::array::from_fn::<_, 3, _>(|axis| {
            let x = (axis + 1) % 3;
            let y = (axis + 2) % 3;
            0.5 * (gradient[y][x] - gradient[x][y])
        });
        errors.push(
            (0..3)
                .map(|a| (estimate[a] - exact[a]).abs())
                .fold(0.0_f64, f64::max),
        );
        estimates.push(estimate);
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = (0..3)
        .map(|a| ((4.0 * estimates[3][a] - estimates[2][a]) / 3.0 - exact[a]).abs())
        .fold(0.0_f64, f64::max);
    println!("magnetic spin curl: steps={steps:?} errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e}",fit.exponent,fit.r_squared);
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999 && rich < 1e-8);
}

fn amplitude_acceptance(
    backend: &mut MagneticDftBackend<'_>,
    generator: &BlochMatrix,
    amplitude: f64,
    closed: Option<f64>,
) {
    let cfg = SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-11,
        ..SpinorScfConfig::default()
    };
    let solve = |backend: &mut MagneticDftBackend<'_>, a: f64| {
        let mut applied = generator.clone();
        applied.scale(a);
        backend.set_external_spin_operator(applied).unwrap();
        let report = run_spinor_scf(backend, &cfg).unwrap();
        assert!(
            report.converged && report.fields.operator_converged,
            "amplitude {a}: iterations={} stationarity={} number={} Coulomb={}",
            report.iterations,
            report.orbital_residual,
            report.particle_number_error,
            report.fields.coulomb_operator_defect
        );
        assert!(report.orbital_residual < 1e-11 && report.particle_number_error < 1e-12);
        report
    };
    let reference = solve(backend, amplitude);
    let exact = reference.density[0].trace_product(generator);
    if let Some(closed) = closed {
        assert!(
            (exact - closed).abs() < 1e-12,
            "closed spin derivative {exact} vs {closed}"
        );
    }
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let mut errors = Vec::new();
    let mut estimates = Vec::new();
    for h in steps {
        let plus = solve(backend, amplitude + h);
        let minus = solve(backend, amplitude - h);
        let estimate = (plus.fields.energy - minus.fields.energy) / (2.0 * h);
        estimates.push(estimate);
        errors.push((estimate - exact).abs());
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = (4.0 * estimates[3] - estimates[2]) / 3.0;
    println!(
        "uniform relaxed amplitude N={}: analytic={exact:.12e} steps={steps:?} \
        errors={errors:?} p={:.9} R2={:.10} Richardson={:.3e} residual={:.3e}",
        backend.electrons(),
        fit.exponent,
        fit.r_squared,
        (rich - exact).abs(),
        reference.orbital_residual
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    assert!((rich - exact).abs() < 1e-8);
    if backend.electrons() == 2.0 {
        let n = reference.density[0].rows() / 2;
        let mixing = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| z(&reference.density[0], i, n + j).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            mixing > 1e-3,
            "the interacting reference must retain orbital/spin coherence"
        );
    }
}

#[test]
fn uniform_flux_scf_derivative_has_a_closed_spin_oracle_and_four_steps() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol, false);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let c = cell(Lattice::cubic(8.0), [1, 1, -1], Vec3::new(0.4, -0.3, 0.2));
    let mut o = options();
    o.nuclear_widths = vec![0.3];
    let mut backend = MagneticDftBackend::new(&mol, &mb, c, &xc, o, None).unwrap();
    let v = Vec3::new(0.3, 0.1, -0.2);
    let scalar_s = backend.overlap()[0].re[(0, 0)];
    let generator = from_pauli(&std::array::from_fn(|axis| {
        let mut m = BlochMatrix::zeros(1, 1);
        m.re[(0, 0)] = if axis == 0 {
            0.0
        } else {
            v.get(axis - 1) * scalar_s
        };
        m
    }))
    .unwrap();
    let radial = (2.0 * (1.4 / std::f64::consts::PI).sqrt() / (1.0 + 1.4 * 0.09_f64).powf(1.5)
        - std::f64::consts::PI / (0.7 * 512.0))
        / (12.0 * 137.035_999_084_f64.powi(2));
    let amplitude = 0.6;
    let total = backend.cell.field * (1.0 + 2.0 * radial) + v * amplitude;
    let closed = -0.5 * total.dot(v) / total.norm();
    amplitude_acceptance(&mut backend, &generator, amplitude, Some(closed));
}

#[test]
fn interacting_two_electron_uniform_flux_scf_has_four_step_relaxed_derivatives() {
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol, true);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let c = cell(Lattice::cubic(8.0), [1, 1, -1], Vec3::zero());
    let mut o = options();
    o.nuclear_widths = vec![0.3];
    let mut backend = MagneticDftBackend::new(&mol, &mb, c, &xc, o, None).unwrap();
    let n = mb.nao_sph;
    let mut generator = BlochMatrix::zeros(2 * n, 2 * n);
    // sigma_x times an imaginary antisymmetric s/p spatial operator is
    // Hermitian and spinful time-reversal even. The fixed B breaks time reversal.
    for (a, b) in [(0, n + 1), (n, 1)] {
        put(&mut generator, a, b, Complex::new(0.0, 0.4));
        put(&mut generator, b, a, Complex::new(0.0, -0.4));
    }
    amplitude_acceptance(&mut backend, &generator, 0.7, None);
}

#[test]
fn interacting_london_references_obey_physical_field_reversal() {
    let mol = Molecule::new(vec![Atom {
        z: 2,
        position: Vec3::zero(),
    }]);
    let mb = basis(&mol, true);
    let xc = resolve(&FunctionalSpec::Combined("hf".into())).unwrap();
    let mut o = options();
    o.nuclear_widths = vec![0.3];
    let n = mb.nao_sph;
    let mut generator = BlochMatrix::zeros(2 * n, 2 * n);
    // This additional orbital/spin coupling is time-reversal even. Physical
    // reversal conjugates the London orbitals as well as the spin coefficients.
    for (a, b) in [(0, n + 1), (n, 1)] {
        put(&mut generator, a, b, Complex::new(0.0, 0.28));
        put(&mut generator, b, a, Complex::new(0.0, -0.28));
    }
    let cfg = SpinorScfConfig {
        energy_tolerance: 1e-13,
        density_tolerance: 1e-12,
        orbital_tolerance: 1e-11,
        ..SpinorScfConfig::default()
    };
    let points = [Vec3::new(0.2, 0.3, -0.1), Vec3::new(-0.25, 0.1, 0.2)];
    let mut states = Vec::new();
    let mut currents = Vec::new();
    for sign in [1, -1] {
        let c = cell(
            Lattice::cubic(8.0),
            [sign, sign, -sign],
            Vec3::new(0.4, -0.3, 0.2),
        );
        let backend =
            MagneticDftBackend::new(&mol, &mb, c, &xc, o.clone(), Some(generator.clone())).unwrap();
        let state = run_spinor_scf(&backend, &cfg).unwrap();
        assert!(state.converged && state.fields.operator_converged);
        currents.push(
            backend
                .sample_particle_current(&state.density[0], &points)
                .unwrap(),
        );
        states.push(state);
    }
    let reverse = |m: &BlochMatrix| {
        let mut out = BlochMatrix::zeros(2 * n, 2 * n);
        for a in 0..2 * n {
            for b in 0..2 * n {
                let flip = |i: usize| if i < n { i + n } else { i - n };
                let sign = |i: usize| if i < n { 1.0 } else { -1.0 };
                put(
                    &mut out,
                    a,
                    b,
                    z(m, flip(a), flip(b)).conj() * (sign(a) * sign(b)),
                );
            }
        }
        out
    };
    let density_defect = gap(&reverse(&states[0].density[0]), &states[1].density[0]);
    let fock_defect = gap(
        &reverse(&states[0].fields.fock[0]),
        &states[1].fields.fock[0],
    );
    let energy_defect = (states[0].fields.energy - states[1].fields.energy).abs();
    let mut number_defect = 0.0_f64;
    let mut spin_defect = 0.0_f64;
    let mut current_defect = 0.0_f64;
    let mut nonzero_current = 0.0_f64;
    for g in 0..points.len() {
        number_defect = number_defect
            .max((currents[0].number_density[g] - currents[1].number_density[g]).abs());
        for axis in 0..3 {
            spin_defect = spin_defect
                .max((currents[0].spin_density[g][axis] + currents[1].spin_density[g][axis]).abs());
            for values in [
                (&currents[0].kinetic_current, &currents[1].kinetic_current),
                (
                    &currents[0].spin_magnetization_current,
                    &currents[1].spin_magnetization_current,
                ),
                (
                    &currents[0].nuclear_soc_current,
                    &currents[1].nuclear_soc_current,
                ),
                (&currents[0].particle_current, &currents[1].particle_current),
            ] {
                current_defect = current_defect.max((values.0[g][axis] + values.1[g][axis]).abs());
            }
            nonzero_current = nonzero_current.max(currents[0].particle_current[g][axis].abs());
        }
    }
    println!(
        "physical B reversal: energy={energy_defect:.3e} \
        density={density_defect:.3e} Fock={fock_defect:.3e} \
        number={number_defect:.3e} spin={spin_defect:.3e} \
        currents={current_defect:.3e} nonzero-current={nonzero_current:.3e}"
    );
    assert!(energy_defect < 1e-10 && density_defect < 1e-10 && fock_defect < 1e-10);
    assert!(number_defect < 1e-10 && spin_defect < 1e-10 && current_defect < 1e-10);
    assert!(nonzero_current > 1e-3);
}
