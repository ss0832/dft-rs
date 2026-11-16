// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent finite-field energies for every symmetric component of the
//! complete generalized-spinor PT2 hyperpolarizabilities.

// Bind the same library path for the shared oracle when compiled as an
// integration test. The in-crate test binds this name to its crate root.
extern crate dft_rs;

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/spinor_pt2_oracle.rs"]
mod spinor_pt2_oracle;
#[path = "support/stencils.rs"]
mod stencils;
#[path = "support/symmetric_directions.rs"]
mod symmetric_directions;

use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::grid::MolecularGrid;
use dft_rs::integrals::{eri_incore, london::ComplexMatrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::{
    spinor_scf::{run_spinor_scf, SpinorEnergyBackend},
    BlochMatrix,
};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::collections::BTreeMap;
use stencils::Stencil;

const COEFFICIENT: f64 = 0.27;
const EXPONENT: f64 = 0.82;

fn scalar(report: &Value, key: &str) -> f64 {
    report.get(key).unwrap().as_f64().unwrap()
}

fn array(report: &Value, key: &str) -> Vec<f64> {
    report.get(key).unwrap().numeric_array().unwrap().1
}

fn contract(tensor: &[f64], direction: [f64; 3], rank: usize) -> f64 {
    tensor
        .iter()
        .enumerate()
        .map(|(mut index, &value)| {
            let mut product = value;
            for _ in 0..rank {
                product *= direction[index % 3];
                index /= 3;
            }
            product
        })
        .sum()
}

#[test]
fn generalized_double_hybrid_beta_gamma_fit_every_symmetric_component() {
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.20, -1.31, -0.61],
        ],
    );
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let path =
        std::env::temp_dir().join(format!("dft-rs-gks-high-order-{}.g94", std::process::id()));
    std::fs::write(&path, format!("H 0\nS 1 1.0\n{EXPONENT} 1.0\n****\n")).unwrap();
    let base = Value::object()
        .with("basis", path.to_str().unwrap())
        .with("xc", "hf")
        .with("charge", 1i32)
        .with("multiplicity", 2usize)
        .with("field", [0.003, -0.002, 0.004])
        .with("guess", "core")
        .with("grid_level", 0)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("dispersion", "none")
        .with("max_iter", 500usize);
    let method = MolecularMethod::from_controls(&Controls::from_value(&base).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let config = setup.spinor_scf_config();
    let target = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
    let grid = MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap();
    let n = setup.basis.nao_sph;
    let operator = spinor_operator::source(n);
    let make_backend = |functional, op: BlochMatrix| {
        MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            functional,
            Some(&grid),
            &seed.scf,
            &setup.options,
            op,
            [0., 0., 1.],
            MolecularSpinorFunctionalModel::SpinCovariant,
            1e-12,
        )
        .unwrap()
    };
    let parent_energy = make_backend(&setup.functional, operator.clone());
    let target_energy = make_backend(&target, operator.clone());
    let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 128).unwrap();
    let controls = base
        .with("reference", "gks")
        .with("pt2_same_spin", COEFFICIENT)
        .with("pt2_opposite_spin", COEFFICIENT)
        .with("double_hybrid_target_xc", "svwn5")
        .with("response_tol", 1e-12)
        .with("gks_operator_real", &operator.re)
        .with("gks_operator_imaginary", &operator.im)
        .with("spinor_xc_tolerance", 1e-12);
    let calculate = |id| {
        let report = run(id, &system, &Controls::from_value(&controls).unwrap())
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(
            report.get("converged").and_then(Value::as_bool),
            Some(true),
            "{id}: {}",
            report.to_json()
        );
        report
    };

    // Closed-form normalized s-Gaussian matrix elements provide a source
    // independent of the derivative route's AO dipole-integral assembly.
    let source = |direction: [f64; 3]| {
        let mut h = Matrix::zeros(2 * n, 2 * n);
        for (i, left) in system.positions.iter().enumerate() {
            for (j, right) in system.positions.iter().enumerate() {
                let distance = (0..3).map(|a| (left[a] - right[a]).powi(2)).sum::<f64>();
                let projection = (0..3)
                    .map(|a| direction[a] * (left[a] + right[a]) * 0.5)
                    .sum::<f64>();
                let entry = (-0.5 * EXPONENT * distance).exp() * projection;
                h[(i, j)] = entry;
                h[(n + i, n + j)] = entry;
            }
        }
        BlochMatrix {
            re: h,
            im: Matrix::zeros(2 * n, 2 * n),
        }
    };
    let sample = |source: &BlochMatrix, strength: f64| {
        let mut biased = operator.clone();
        biased.add_scaled(source, strength);
        let reference = run_spinor_scf(&make_backend(&setup.functional, biased), &config).unwrap();
        assert!(reference.converged && reference.orbital_residual <= config.orbital_tolerance);
        let band = &reference.bands[0];
        let orbitals = ComplexMatrix {
            re: band.coefficients.re.clone(),
            im: band.coefficients.im.clone(),
        };
        let pair = spinor_pt2_oracle::stationary_pt2(&orbitals, &band.energies, 3, &eri);
        let p = parent_energy.evaluate(&reference.density).unwrap();
        let t = target_energy.evaluate(&reference.density).unwrap();
        assert!(p.operator_converged && t.operator_converged);
        let parent = reference.fields.energy;
        let correction = t.energy - p.energy;
        [
            parent + correction + COEFFICIENT * pair,
            parent,
            correction,
            pair,
        ]
    };
    let centre = sample(&source([0.; 3]), 0.);
    let channels = [
        "tensor",
        "parent_tensor",
        "target_tensor_correction",
        "generalized_pt2_tensor",
    ];
    let mut failures = Vec::new();
    for (rank, id) in [(3, "beta"), (4, "gamma")] {
        let report = calculate(id);
        let tensors = channels.map(|key| array(&report, key));
        let weighted = array(&report, "pt2_tensor");
        let target = array(&report, "target_tensor");
        assert_eq!(array(&report, "shape"), vec![3.; rank]);
        assert_eq!(report.get("energy_permutation_defect"), Some(&Value::Null));
        assert_eq!(report.get("response_converged"), Some(&Value::Bool(true)));
        assert!((scalar(&report, "energy") - centre[0]).abs() < 2e-12);
        assert!((scalar(&report, "generalized_pt2_correlation") - centre[3]).abs() < 2e-12);
        assert!(
            (scalar(
                report.get("energy_components").unwrap(),
                "target_correction"
            ) - centre[2])
                .abs()
                < 2e-12
        );
        for i in 0..3usize.pow(rank as u32) {
            assert!((target[i] - tensors[1][i] - tensors[2][i]).abs() < 2e-10);
            assert!((weighted[i] - COEFFICIENT * tensors[3][i]).abs() < 2e-10);
            assert!((tensors[0][i] - target[i] - weighted[i]).abs() < 2e-10);
        }
        let maxima = [
            "response_residuals",
            "projector_defects",
            "nuclear_stationarity_defects",
            "amplitude_response_residuals",
            "amplitude_relative_residuals",
            "response_number_defects",
            "orbital_frame_defects",
            "imaginary_energy_defects",
            "reference_defects",
        ]
        .map(|key| {
            let maximum = array(&report, key)
                .into_iter()
                .map(f64::abs)
                .fold(0.0, f64::max);
            (key, maximum)
        });
        println!("GKS correlated {id} energy={:.16e} pair={:.16e} tensor[0]={:.16e} residuals={maxima:?}", centre[0], centre[3], tensors[0][0]);
        println!("GKS correlated {id} Cartesian tensor={:?}", tensors[0]);
        for (index, direction) in symmetric_directions::directions(rank)
            .into_iter()
            .enumerate()
        {
            let exact = tensors
                .each_ref()
                .map(|tensor| contract(tensor, direction, rank));
            let perturbation = source(direction);
            let mut samples = BTreeMap::new();
            let mut finite = |t: f64| {
                *samples
                    .entry(t.to_bits())
                    .or_insert_with(|| sample(&perturbation, t))
            };
            let mut central = |h: f64| -> [f64; 4] {
                let (pp, p, m, mm) = (finite(2. * h), finite(h), finite(-h), finite(-2. * h));
                std::array::from_fn(|c| {
                    if rank == 3 {
                        -Stencil::Second.first(
                            pp[c] - 2. * p[c] + centre[c],
                            mm[c] - 2. * m[c] + centre[c],
                            0.,
                            0.,
                            h,
                        ) / (h * h)
                    } else {
                        -((pp[c] + mm[c] - 2. * centre[c]) - 4. * (p[c] + m[c] - 2. * centre[c]))
                            / h.powi(4)
                    }
                })
            };
            let names = channels.map(|key| format!("GKS {id} generic direction {index} {key}"));
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.025, 0.020, 0.015, 0.010], Stencil::Second),
                    ([0.015, 0.012, 0.009, 0.006], Stencil::Second),
                    ([0.04, 0.03, 0.022, 0.015], Stencil::Second),
                    // The small-gap reference can leave its asymptotic
                    // region at the larger fields, even when SCF converges.
                    ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
                    ([0.002, 0.0015, 0.001, 0.0007], Stencil::Second),
                    ([0.004, 0.003, 0.002, 0.0015], Stencil::Fourth),
                ],
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; channels.len()];
                    for (i, &h) in steps.iter().enumerate() {
                        let estimate = central(h);
                        let coarser = if stencil == Stencil::Fourth {
                            central(2.0 * h)
                        } else {
                            [0.0; 4]
                        };
                        for c in 0..channels.len() {
                            let derivative = if stencil == Stencil::Fourth {
                                (4. * estimate[c] - coarser[c]) / 3.
                            } else {
                                estimate[c]
                            };
                            errors[c][i] = (derivative - exact[c]).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
    }
    std::fs::remove_file(path).unwrap();
    components::assert_no_failures(
        "all generalized double-hybrid beta/gamma directions and energy terms",
        &failures,
    );
}
