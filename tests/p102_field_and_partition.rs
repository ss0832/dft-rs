// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Acceptance tests of the combined electric and magnetic field response
//! with a complex Hermitian density and of the divide-and-conquer iteration.
//!
//! Oracles. Without a magnetic field the complex recursion must reproduce
//! the real electric-field recursion order by order. Its first-order magnetic
//! moment is compared with the magnetizability of the coupled-perturbed
//! magnetic response, an independent route. The hypermagnetizability is
//! fitted against second differences of that independent magnetizability in
//! a finite electric field with four field strengths; the differences are
//! validation only. The divide-and-conquer energy and gradient are compared
//! with the conventional Fermi-Dirac calculation in the limit of a complete
//! buffer.

use dft_rs::basis::{data, MolBasis};
use dft_rs::gradient::{assemble_gradient, assemble_gradient_weighted};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::integrals::one_electron::moment_matrices;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::properties::magnetic::{magnetic_density_response, Gauge};
use dft_rs::properties::nmr::angular_momentum_operator;
use dft_rs::response::field_path::{
    dynamic_hypermagnetizability, field_hypermagnetizability, field_magnetizability,
    field_path_taylor, field_polarizability_path, spin_orbit_polarizability,
};
use dft_rs::response::krylov::ResponseSolverConfig;
use dft_rs::response::magnetic_taylor::{hypermagnetizability, magnetic_field_taylor};
use dft_rs::response::nonlinear::electric_field_taylor;
use dft_rs::scf::divide_conquer::{
    divide_conquer_gradient, run_divide_conquer, DivideConquerOptions,
};
use dft_rs::scf::{run_scf, EriMode, ScfOptions, ScfResult};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec, ResolvedFunctional};

#[path = "support/convergence.rs"]
mod convergence;

/// Water distorted away from every symmetry element, Bohr.
fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.05, 0.2300),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.15, 1.4309, -0.8867),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.1200, -1.4809, -0.9067),
        },
    ])
}

fn options(field: Option<Vec3>) -> ScfOptions {
    ScfOptions {
        e_tol: 1e-12,
        p_tol: 1e-11,
        max_iter: 300,
        eri_mode: EriMode::InCore { limit_mb: 512 },
        external_field: field,
        ..ScfOptions::default()
    }
}

fn config() -> ResponseSolverConfig {
    ResponseSolverConfig {
        tolerance: 1e-11,
        ..ResponseSolverConfig::default()
    }
}

struct Point {
    mol: Molecule,
    mb: MolBasis,
    grid: Option<MolecularGrid>,
    scf: ScfResult,
    opts: ScfOptions,
}

fn reference(mol: &Molecule, func: &ResolvedFunctional, opts: ScfOptions) -> Point {
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(mol, &set).unwrap();
    let grid = (!func.parts.is_empty()).then(|| {
        MolecularGrid::build(
            mol,
            &GridOptions {
                level: 3,
                ..GridOptions::default()
            },
        )
        .unwrap()
    });
    let scf = run_scf(mol, &mb, &opts, func, grid.as_ref()).unwrap();
    assert!(scf.converged);
    Point {
        mol: mol.clone(),
        mb,
        grid,
        scf,
        opts,
    }
}

fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            sum += a[(i, j)] * b[(j, i)];
        }
    }
    sum
}

/// Magnetizability from the coupled-perturbed magnetic response with a
/// common gauge origin: the independent route.
fn magnetizability(point: &Point, func: &ResolvedFunctional, origin: Vec3) -> [[f64; 3]; 3] {
    let response = magnetic_density_response(
        &point.mol,
        &point.mb,
        func,
        point.grid.as_ref(),
        &point.scf,
        &point.opts,
        Gauge::CommonOrigin,
        origin,
        &config(),
    )
    .unwrap();
    let hat = &response.value.density_hat;
    let first: Vec<Matrix> = (0..3)
        .map(|a| {
            let mut m = hat[0][a].clone();
            for (t, s) in m.as_mut_slice().iter_mut().zip(hat[1][a].as_slice()) {
                *t += *s;
            }
            m
        })
        .collect();
    let p_total = point.scf.p_total();
    let angular = angular_momentum_operator(&point.mb, origin);
    let mut moments = [[0.0_f64; 3]; 3];
    for (power, matrix) in moment_matrices(&point.mb, origin, 2) {
        let value = trace_product(&p_total, &matrix);
        let powers = [power.0, power.1, power.2];
        let mut axes = Vec::new();
        for axis in 0..3 {
            for _ in 0..powers[axis] {
                axes.push(axis);
            }
        }
        moments[axes[0]][axes[1]] = value;
        moments[axes[1]][axes[0]] = value;
    }
    let r2 = moments[0][0] + moments[1][1] + moments[2][2];
    let mut tensor = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { r2 } else { 0.0 };
            tensor[a][b] =
                -0.25 * (delta - moments[a][b]) - 0.5 * trace_product(&first[a], &angular[b]);
        }
    }
    tensor
}

#[test]
fn complex_field_recursion_reduces_to_the_real_one_and_to_the_magnetizability() {
    let mol = water();
    let origin = Vec3::new(0.1, -0.2, 0.3);
    for name in ["hf", "pbe", "pbe0"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func, options(None));
        let direction = [0.3, -0.5, 0.8];
        let real = electric_field_taylor(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            direction,
            3,
            &config(),
        )
        .unwrap();
        let complex = magnetic_field_taylor(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            direction,
            [0.0; 3],
            origin,
            3,
            &config(),
        )
        .unwrap();
        let mut worst = 0.0_f64;
        for n in 0..=3 {
            for a in 0..3 {
                worst = worst.max(
                    (real.value.dipole_coefficients[n][a]
                        - complex.value.dipole_coefficients[n][a])
                        .abs(),
                );
            }
        }
        println!("{name}: electric coefficients through order 3 differ by {worst:.3e}");
        assert!(
            worst < 1e-8,
            "{name}: real and complex recursions differ by {worst:e}"
        );

        // First-order magnetic moment against the independent magnetizability.
        let oracle = magnetizability(&point, &func, origin);
        let mut defect = 0.0_f64;
        let mut tensor = [[0.0_f64; 3]; 3];
        for d in 0..3 {
            let mut unit = [0.0; 3];
            unit[d] = 1.0;
            let path = magnetic_field_taylor(
                &point.mol,
                &point.mb,
                &func,
                point.grid.as_ref(),
                &point.scf,
                &point.opts,
                [0.0; 3],
                unit,
                origin,
                1,
                &config(),
            )
            .unwrap();
            for c in 0..3 {
                tensor[c][d] = path.value.magnetic_coefficients[1][c];
                defect = defect.max((tensor[c][d] - oracle[c][d]).abs());
            }
            println!(
                "{name}: B along {d}: projector defects {:?} stationarity defects {:?}",
                path.value.projector_defects, path.value.stationarity_defects
            );
        }
        println!("{name}: magnetizability {tensor:?} oracle {oracle:?} defect {defect:.3e}");
        assert!(defect < 1e-7, "{name}: magnetizability defect {defect:e}");
    }
}

#[test]
fn hypermagnetizability_passes_a_four_step_fit_of_the_magnetizability_in_a_field() {
    let mol = water();
    let origin = Vec3::new(0.1, -0.2, 0.3);
    for name in ["hf", "pbe", "pbe0"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func, options(None));
        let result = hypermagnetizability(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            origin,
            &config(),
        )
        .unwrap();
        let eta = &result.tensor;
        // Independently evaluated index orders agree.
        let mut permutation = 0.0_f64;
        let mut scale = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        permutation = permutation.max((eta[a][b][c][d] - eta[b][a][c][d]).abs());
                        scale = scale.max(eta[a][b][c][d].abs());
                    }
                }
            }
        }
        println!(
            "{name}: largest eta {scale:.6} permutation defect {permutation:.3e} evaluations {} \
             residual {:.3e} roundoff {:.3e}",
            result.directional_evaluations, result.residual, result.accumulation_roundoff
        );
        assert!(permutation < 1e-6 * scale.max(1.0));
        let center = magnetizability(&point, &func, origin);
        let steps = [0.04, 0.02, 0.01, 0.005];
        for axis in 0..3 {
            let mut errors = Vec::new();
            let mut values = Vec::new();
            for &h in &steps {
                let mut field = [0.0; 3];
                field[axis] = h;
                let plus = reference(
                    &mol,
                    &func,
                    options(Some(Vec3::new(field[0], field[1], field[2]))),
                );
                let minus = reference(
                    &mol,
                    &func,
                    options(Some(Vec3::new(-field[0], -field[1], -field[2]))),
                );
                let xi_plus = magnetizability(&plus, &func, origin);
                let xi_minus = magnetizability(&minus, &func, origin);
                let mut worst = 0.0_f64;
                let mut second = [[0.0_f64; 3]; 3];
                for c in 0..3 {
                    for d in 0..3 {
                        second[c][d] =
                            (xi_plus[c][d] - 2.0 * center[c][d] + xi_minus[c][d]) / (h * h);
                        worst = worst.max((second[c][d] - eta[axis][axis][c][d]).abs());
                    }
                }
                errors.push(worst);
                values.push(second);
            }
            let fit = convergence::error_power_fit(&steps, &errors);
            let mut richardson = 0.0_f64;
            for c in 0..3 {
                for d in 0..3 {
                    let extrapolated = (4.0 * values[3][c][d] - values[2][c][d]) / 3.0;
                    richardson = richardson.max((extrapolated - eta[axis][axis][c][d]).abs());
                }
            }
            println!(
                "{name} field axis {axis}: p={:.6} R2={:.6} errors {errors:?} Richardson defect \
                 {richardson:.3e} steps {steps:?}",
                fit.exponent, fit.r_squared
            );
            assert!(
                (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
                "{name} axis {axis}: p={} R2={}",
                fit.exponent,
                fit.r_squared
            );
        }
    }
}

/// A bent chain of hydrogen atoms, Bohr.
fn chain(count: usize) -> Molecule {
    Molecule::new(
        (0..count)
            .map(|i| Atom {
                z: 1,
                position: Vec3::new(1.7 * i as f64, 0.25 * (i % 2) as f64, 0.1 * (i % 3) as f64),
            })
            .collect(),
    )
}

#[test]
fn divide_conquer_with_a_complete_buffer_is_the_conventional_fermi_dirac_result() {
    let kt = 0.01;
    for (name, mol) in [("hf", water()), ("pbe", chain(6)), ("pbe0", chain(4))] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let mut opts = options(None);
        opts.electronic_temperature = kt;
        let point = reference(&mol, &func, opts);
        let conventional = assemble_gradient(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
        )
        .unwrap();
        let natoms = mol.len();
        let mut plain = options(None);
        plain.conv_tol_grad = 1e-9;
        let partition = DivideConquerOptions {
            fragment_of: (0..natoms).collect(),
            buffer: 1.0e3,
            kt,
        };
        let result = run_divide_conquer(
            &point.mol,
            &point.mb,
            &plain,
            &func,
            point.grid.as_ref(),
            &partition,
        )
        .unwrap();
        assert!(result.scf.converged, "{name}: iteration did not converge");
        let gradient = assemble_gradient_weighted(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &result.scf,
            &plain,
            &result.energy_weighted,
        )
        .unwrap();
        let mut worst = 0.0_f64;
        for (a, b) in gradient.iter().zip(&conventional) {
            worst = worst.max((*a - *b).norm());
        }
        println!(
            "{name}: conventional free energy {:.10} divide-and-conquer {:.10} difference {:.3e} \
             iterations {} density residual {:.3e} gradient difference {worst:.3e}",
            point.scf.energy,
            result.scf.energy,
            result.scf.energy - point.scf.energy,
            result.scf.iterations,
            result.density_residual
        );
        assert!((result.scf.energy - point.scf.energy).abs() < 1e-7);
        assert!(worst < 1e-5, "{name}: gradient difference {worst:e}");
    }
}

#[test]
fn divide_conquer_energy_approaches_the_conventional_one_as_the_buffer_grows() {
    let kt = 0.01;
    let mol = chain(8);
    let func = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let mut opts = options(None);
    opts.electronic_temperature = kt;
    let point = reference(&mol, &func, opts);
    let mut plain = options(None);
    plain.conv_tol_grad = 1e-9;
    let mut errors = Vec::new();
    for buffer in [2.0, 4.0, 6.0, 9.0, 1.0e3] {
        let partition = DivideConquerOptions {
            fragment_of: (0..mol.len()).collect(),
            buffer,
            kt,
        };
        let result = run_divide_conquer(
            &point.mol,
            &point.mb,
            &plain,
            &func,
            point.grid.as_ref(),
            &partition,
        )
        .unwrap();
        let sizes: Vec<usize> = result
            .subsystems
            .iter()
            .map(|part| part.functions.len())
            .collect();
        let error = (result.scf.energy - point.scf.energy).abs();
        println!(
            "buffer {buffer}: subsystem sizes {sizes:?} energy {:.10} error {error:.3e} \
             iterations {} converged {}",
            result.scf.energy, result.scf.iterations, result.scf.converged
        );
        assert!(result.scf.converged);
        errors.push(error);
    }
    assert!(errors[4] < 1e-7, "complete buffer error {:e}", errors[4]);
    assert!(
        errors[3] < errors[0],
        "the error does not decrease with the buffer"
    );
}

/// Divide-and-conquer free energy of a displaced chain with an incomplete
/// buffer.
fn partitioned_energy(mol: &Molecule, func: &ResolvedFunctional, buffer: f64, kt: f64) -> f64 {
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(mol, &set).unwrap();
    let grid = (!func.parts.is_empty()).then(|| {
        MolecularGrid::build(
            mol,
            &GridOptions {
                level: 3,
                ..GridOptions::default()
            },
        )
        .unwrap()
    });
    let mut opts = options(None);
    opts.conv_tol_grad = 1e-10;
    let partition = DivideConquerOptions {
        fragment_of: (0..mol.len()).collect(),
        buffer,
        kt,
    };
    let result = run_divide_conquer(mol, &mb, &opts, func, grid.as_ref(), &partition).unwrap();
    assert!(result.scf.converged);
    result.scf.energy
}

/// The gradient with the partition response is the derivative of the free
/// energy for an incomplete buffer: four-step fit against central
/// differences along a displacement pattern.
#[test]
fn divide_conquer_gradient_passes_a_four_step_fit_with_an_incomplete_buffer() {
    let kt = 0.01;
    let buffer = 4.0;
    let mol = chain(6);
    let direction: Vec<[f64; 3]> = (0..mol.len())
        .map(|i| {
            [
                0.2 * ((i % 3) as f64 - 1.0),
                0.15 * (i % 2) as f64 - 0.05,
                0.1 * (i as f64 - 2.5),
            ]
        })
        .collect();
    for name in ["hf", "pbe", "pbe0"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let set = data::load("sto-3g").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let grid = (!func.parts.is_empty()).then(|| {
            MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: 3,
                    ..GridOptions::default()
                },
            )
            .unwrap()
        });
        let mut opts = options(None);
        opts.conv_tol_grad = 1e-10;
        let partition = DivideConquerOptions {
            fragment_of: (0..mol.len()).collect(),
            buffer,
            kt,
        };
        let result =
            run_divide_conquer(&mol, &mb, &opts, &func, grid.as_ref(), &partition).unwrap();
        assert!(result.scf.converged);
        let (gradient, residual) = divide_conquer_gradient(
            &mol,
            &mb,
            &opts,
            &func,
            grid.as_ref(),
            &partition,
            &result,
            &config(),
        )
        .unwrap();
        let analytic: f64 = gradient
            .iter()
            .zip(&direction)
            .map(|(g, d)| g.x * d[0] + g.y * d[1] + g.z * d[2])
            .sum();
        let plain = assemble_gradient_weighted(
            &mol,
            &mb,
            &func,
            grid.as_ref(),
            &result.scf,
            &opts,
            &result.energy_weighted,
        )
        .unwrap();
        let unrelaxed: f64 = plain
            .iter()
            .zip(&direction)
            .map(|(g, d)| g.x * d[0] + g.y * d[1] + g.z * d[2])
            .sum();
        let steps = [0.04, 0.02, 0.01, 0.005];
        let mut errors = Vec::new();
        let mut values = Vec::new();
        for &h in &steps {
            let mut plus = mol.clone();
            let mut minus = mol.clone();
            for (i, d) in direction.iter().enumerate() {
                plus.atoms[i].position.x += h * d[0];
                plus.atoms[i].position.y += h * d[1];
                plus.atoms[i].position.z += h * d[2];
                minus.atoms[i].position.x -= h * d[0];
                minus.atoms[i].position.y -= h * d[1];
                minus.atoms[i].position.z -= h * d[2];
            }
            let difference = (partitioned_energy(&plus, &func, buffer, kt)
                - partitioned_energy(&minus, &func, buffer, kt))
                / (2.0 * h);
            errors.push((difference - analytic).abs());
            values.push(difference);
        }
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = (4.0 * values[3] - values[2]) / 3.0;
        println!(
            "{name}: analytic {analytic:.10} energy-weighted expression {unrelaxed:.10} \
             Richardson {richardson:.10} difference {:.3e}; p={:.6} R2={:.6} errors {errors:?} \
             steps {steps:?}; adjoint residual {residual:.3e}",
            analytic - richardson,
            fit.exponent,
            fit.r_squared
        );
        assert!(
            (fit.exponent - 2.0).abs() < 0.1 && fit.r_squared > 0.999,
            "{name}: p={} R2={}",
            fit.exponent,
            fit.r_squared
        );
        assert!((analytic - richardson).abs() < 1e-6);
    }
}

/// With a common gauge origin the field path over dense integral series
/// reproduces the recursion over the production engines; with London
/// orbitals the magnetizability does not change when the molecule is
/// translated, while the common-origin tensor about a fixed origin does.
#[test]
fn london_field_path_is_translation_invariant_and_reduces_to_the_common_origin_route() {
    let mol = water();
    let origin = Vec3::new(0.1, -0.2, 0.3);
    let translation = [1.3, -0.7, 2.1];
    let mut shifted = mol.clone();
    for atom in &mut shifted.atoms {
        atom.position.x += translation[0];
        atom.position.y += translation[1];
        atom.position.z += translation[2];
    }
    for name in ["hf", "pbe", "pbe0"] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func, options(None));
        let electric = [0.3, -0.5, 0.8];
        let magnetic = [0.6, 0.2, -0.4];
        let dense = field_path_taylor(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            electric,
            magnetic,
            origin,
            false,
            3,
            &config(),
        )
        .unwrap();
        let production = magnetic_field_taylor(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            electric,
            magnetic,
            origin,
            3,
            &config(),
        )
        .unwrap();
        let mut worst = 0.0_f64;
        for n in 0..=3 {
            for a in 0..3 {
                worst = worst
                    .max((dense.dipole[n][a] - production.value.dipole_coefficients[n][a]).abs());
            }
        }
        println!(
            "{name}: common-origin dipole coefficients of the two routes differ by {worst:.3e}; \
             reference defect {:.3e}; energy {:?}",
            dense.reference_defect, dense.energy
        );
        assert!(dense.reference_defect < 1e-6);
        // A meta-GGA would differ here by the gauge-invariant kinetic energy
        // density; these functionals have none.
        assert!(worst < 1e-6, "{name}: routes differ by {worst:e}");

        let moved = reference(&shifted, &func, options(None));
        let (here, _) = field_magnetizability(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            origin,
            true,
            &config(),
        )
        .unwrap();
        let (there, _) = field_magnetizability(
            &moved.mol,
            &moved.mb,
            &func,
            moved.grid.as_ref(),
            &moved.scf,
            &moved.opts,
            origin,
            true,
            &config(),
        )
        .unwrap();
        let (fixed_here, _) = field_magnetizability(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            origin,
            false,
            &config(),
        )
        .unwrap();
        let (fixed_there, _) = field_magnetizability(
            &moved.mol,
            &moved.mb,
            &func,
            moved.grid.as_ref(),
            &moved.scf,
            &moved.opts,
            origin,
            false,
            &config(),
        )
        .unwrap();
        let oracle = magnetizability(&point, &func, origin);
        let mut london_change = 0.0_f64;
        let mut common_change = 0.0_f64;
        let mut common_defect = 0.0_f64;
        for c in 0..3 {
            for d in 0..3 {
                london_change = london_change.max((here[c][d] - there[c][d]).abs());
                common_change = common_change.max((fixed_here[c][d] - fixed_there[c][d]).abs());
                common_defect = common_defect.max((fixed_here[c][d] - oracle[c][d]).abs());
            }
        }
        println!(
            "{name}: London magnetizability {here:?}; change under translation {london_change:.3e} \
             (common origin: {common_change:.3e}); common-origin energy route against the \
             coupled-perturbed route {common_defect:.3e}"
        );
        assert!(
            london_change < 1e-6,
            "{name}: London tensor changes by {london_change:e}"
        );
        assert!(common_defect < 1e-6);
    }
}

/// At zero frequency the second magnetic-field coefficient of the
/// polarizability of the paired response is the static hypermagnetizability
/// of the dipole route, and its zeroth coefficient the static
/// polarizability; at a photon energy the first coefficient is imaginary and
/// antisymmetric (the Faraday tensor) and the zeroth one is the fixed-field
/// polarizability.
#[test]
fn frequency_dependent_field_path_reduces_to_the_static_tensors() {
    let mol = water();
    let origin = Vec3::new(0.1, -0.2, 0.3);
    for (name, london) in [("hf", false), ("pbe", true)] {
        let func = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let point = reference(&mol, &func, options(None));
        let fixed = field_hypermagnetizability(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            origin,
            london,
            &config(),
        )
        .unwrap();
        let limit = dynamic_hypermagnetizability(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            origin,
            london,
            0.0,
            &config(),
        )
        .unwrap();
        let mut worst = [0.0_f64; 2];
        let mut scale = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                worst[0] =
                    worst[0].max((limit.polarizability[a][b] - fixed.polarizability[a][b]).abs());
                for c in 0..3 {
                    for d in 0..3 {
                        worst[1] = worst[1]
                            .max((limit.tensor[a][b][c][d] - fixed.tensor[a][b][c][d]).abs());
                        scale = scale.max(fixed.tensor[a][b][c][d].abs());
                    }
                }
            }
        }
        println!(
            "{name} london={london}: static limit defects alpha {:.3e} eta {:.3e} (largest eta \
             {scale:.6})",
            worst[0], worst[1]
        );
        assert!(worst[0] < 1e-7);
        assert!(worst[1] < 1e-6 * scale.max(1.0));

        let omega = 0.1;
        let path = field_polarizability_path(
            &point.mol,
            &point.mb,
            &func,
            point.grid.as_ref(),
            &point.scf,
            &point.opts,
            [0.3, -0.5, 0.8],
            origin,
            london,
            1,
            omega,
            &config(),
        )
        .unwrap();
        let mut first_real = 0.0_f64;
        let mut asymmetry = 0.0_f64;
        let mut zeroth_imaginary = 0.0_f64;
        let mut faraday = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                first_real = first_real.max(path.real[1][a][b].abs());
                asymmetry =
                    asymmetry.max((path.imaginary[1][a][b] + path.imaginary[1][b][a]).abs());
                zeroth_imaginary = zeroth_imaginary.max(path.imaginary[0][a][b].abs());
                faraday = faraday.max(path.imaginary[1][a][b].abs());
            }
        }
        println!(
            "{name} london={london}: first-order real part {first_real:.3e}, Faraday tensor \
             magnitude {faraday:.6} with symmetric part {asymmetry:.3e}, zeroth-order imaginary \
             part {zeroth_imaginary:.3e}, residuals {:?}",
            path.residuals
        );
        assert!(first_real < 1e-7 && asymmetry < 1e-7 && zeroth_imaginary < 1e-9);
        assert!(faraday > 1e-6);
    }
}

/// The spin-orbit magneto-optical tensor of an open-shell molecule is
/// imaginary and antisymmetric in its two polarizability indices, and its
/// zeroth order is the polarizability of the fixed-geometry route.
#[test]
fn spin_orbit_magneto_optical_tensor_is_imaginary_and_antisymmetric() {
    use dft_rs::properties::nmr::paramagnetic_operator;
    use dft_rs::scf::Reference;
    // Hydroxyl radical, doublet, away from the coordinate axes (Bohr).
    let mut mol = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.1, -0.05, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9, 0.7, 1.3),
        },
    ]);
    mol.multiplicity = 2;
    let func = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let mut opts = options(None);
    opts.reference = Reference::Uhf;
    let point = reference(&mol, &func, opts);
    assert!(point.scf.unrestricted);
    let n = point.mb.nao_sph;
    let scale = 0.5 * 7.2973525693e-3_f64.powi(2);
    let mut orbital: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(n, n));
    for atom in &mol.atoms {
        let operator = paramagnetic_operator(&point.mb, atom.position);
        for (target, source) in orbital.iter_mut().zip(&operator) {
            for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *t += scale * atom.z as f64 * s;
            }
        }
    }
    let result = spin_orbit_polarizability(
        &point.mol,
        &point.mb,
        &func,
        point.grid.as_ref(),
        &point.scf,
        &point.opts,
        &orbital,
        0.05,
        &config(),
    )
    .unwrap();
    let mut largest = 0.0_f64;
    let mut symmetric = 0.0_f64;
    let mut real = 0.0_f64;
    for k in 0..3 {
        for a in 0..3 {
            for b in 0..3 {
                largest = largest.max(result.imaginary[a][b][k].abs());
                symmetric =
                    symmetric.max((result.imaginary[a][b][k] + result.imaginary[b][a][k]).abs());
                real = real.max(result.real[a][b][k].abs());
            }
        }
    }
    println!(
        "spin-orbit magneto-optical tensor: largest element {largest:.6e}, symmetric part \
         {symmetric:.3e}, real part {real:.3e}, residual {:.3e}",
        result.residual
    );
    assert!(largest > 1e-10);
    assert!(symmetric < 1e-6 * largest.max(1e-6));
    assert!(real < 1e-6 * largest.max(1e-6));
}
