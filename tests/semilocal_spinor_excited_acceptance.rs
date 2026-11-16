// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete C1 spinor pole/absolute-energy derivatives on a bounded fixed grid.
//! This validates moving-grid derivatives, not spatial-grid accuracy.
#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/spinor_operator.rs"]
mod spinor_operator;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::basis::{data, MolBasis};
use dft_rs::dfpt::orbitals::{adjoint, matmul};
use dft_rs::driver::method::{
    prepare_molecular, run_molecular_reference, MolecularMethod, MolecularSetup,
};
use dft_rs::driver::{Controls, System, Value};
use dft_rs::grid::MolecularGrid;
use dft_rs::integrals::overlap_matrix;
use dft_rs::linalg::Matrix;
use dft_rs::periodic::{
    spinor_scf::{run_spinor_scf, SpinorScfConfig, SpinorScfReport},
    BlochMatrix,
};
use dft_rs::response::complex::Complex;
use dft_rs::response::nuclear::{
    spinor_excitation_path, spinor_excitations, spinor_excited_state_hessian,
};
use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use dft_rs::response::spinor_tda::{SpinorTdaConfig, SpinorTdaState};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::system::Molecule;
use std::collections::HashMap;

fn system() -> System {
    System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    )
}
fn tda() -> SpinorTdaConfig {
    SpinorTdaConfig {
        states: 12,
        memory_mb: 128,
        tolerance: 1e-9,
        minimum_root_gap: 1e-6,
    }
}
fn config() -> ResponseSolverConfig {
    ResponseSolverConfig {
        tolerance: 1e-10,
        ..Default::default()
    }
}
struct Point {
    setup: MolecularSetup,
    reference: SpinorScfReport,
    operator: BlochMatrix,
    overlap: Matrix,
}
fn prepare(system: &System, name: &str) -> Point {
    let options = Controls::from_value(
        &Value::object()
            .with("xc", name)
            .with("basis", "sto-3g")
            .with("guess", "core")
            .with("charge", 2i32)
            .with("dispersion", "none")
            .with("conv_tol", 1e-13)
            .with("conv_tol_dp", 1e-13)
            .with("conv_tol_grad", 1e-12)
            .with("max_iter", 400usize)
            .with("grid_weight_cutoff", 0.0),
    )
    .unwrap();
    let method = MolecularMethod::from_controls(&options).unwrap();
    let mut setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    setup.grid_options.radial_override = Some(2);
    setup.grid_options.angular_override = Some(3);
    setup.grid_options.prune = false;
    setup.grid = Some(MolecularGrid::build(&setup.molecule, &setup.grid_options).unwrap());
    let seed = run_molecular_reference(&setup).unwrap();
    let mut operator = spinor_operator::source(4);
    for v in operator
        .re
        .as_mut_slice()
        .iter_mut()
        .chain(operator.im.as_mut_slice())
    {
        *v *= 0.01;
    }
    let backend = MolecularSpinorBackend::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        &seed.scf,
        &setup.options,
        operator.clone(),
        [0.0, 0.0, 1.0],
        MolecularSpinorFunctionalModel::SpinCovariant,
        1e-11,
    )
    .unwrap();
    let reference = run_spinor_scf(
        &backend,
        &SpinorScfConfig {
            energy_tolerance: 1e-13,
            density_tolerance: 1e-13,
            orbital_tolerance: 1e-12,
            max_iter: 400,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reference.converged);
    drop(backend);
    Point {
        setup,
        reference,
        operator,
        overlap: seed.scf.overlap.clone(),
    }
}
fn spectrum(point: &Point) -> Vec<SpinorTdaState> {
    let s = &point.setup;
    spinor_excitations(
        &s.molecule,
        &s.basis,
        s.grid.as_ref(),
        &s.functional,
        &point.reference,
        &s.options,
        &point.operator,
        &tda(),
        &config(),
    )
    .unwrap()
    .states
}
fn inner(a: &BlochMatrix, b: &BlochMatrix, cross: &BlochMatrix) -> Complex {
    let product = matmul(
        &matmul(&matmul(&adjoint(a).unwrap(), cross).unwrap(), b).unwrap(),
        &adjoint(cross).unwrap(),
    )
    .unwrap();
    (0..product.rows())
        .map(|i| Complex::new(product.re[(i, i)], product.im[(i, i)]))
        .fold(Complex::ZERO, |a, b| a + b)
}
fn spin_metric(spatial: &Matrix) -> BlochMatrix {
    let n = spatial.rows;
    let mut out = BlochMatrix::zeros(2 * n, 2 * n);
    for s in 0..2 {
        for i in 0..n {
            for j in 0..n {
                out.re[(s * n + i, s * n + j)] = spatial[(i, j)];
            }
        }
    }
    out
}
fn overlaps(
    base: &Point,
    base_state: &SpinorTdaState,
    point: &Point,
    states: &[SpinorTdaState],
) -> Vec<f64> {
    let mut atoms = base.setup.molecule.atoms.clone();
    atoms.extend(point.setup.molecule.atoms.iter().cloned());
    let basis = MolBasis::build(&Molecule::new(atoms), &data::load("sto-3g").unwrap()).unwrap();
    let n = base.setup.basis.nao_sph;
    let metric = overlap_matrix(&basis);
    let mut cross = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cross[(i, j)] = metric[(i, n + j)];
        }
    }
    let left = inner(
        &base_state.transition_density,
        &base_state.transition_density,
        &spin_metric(&base.overlap),
    )
    .re;
    let right_metric = spin_metric(&point.overlap);
    let cross = spin_metric(&cross);
    states
        .iter()
        .map(|s| {
            inner(
                &base_state.transition_density,
                &s.transition_density,
                &cross,
            )
            .abs()
                / (left * inner(&s.transition_density, &s.transition_density, &right_metric).re)
                    .sqrt()
        })
        .collect()
}

fn complete(name: &str) {
    let system = system();
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let base = prepare(&system, name);
    println!(
        "{name} stage=reference complete residual={:.12e}",
        base.reference.orbital_residual
    );
    let states = spectrum(&base);
    println!(
        "{name} stage=static complete pole={:.12e} gap={:.12e}",
        states[0].energy, states[0].nearest_gap
    );
    let s = &base.setup;
    let hessian = spinor_excited_state_hessian(
        &s.molecule,
        &s.basis,
        s.grid.as_ref(),
        &s.functional,
        &base.reference,
        &s.options,
        &base.operator,
        0,
        &tda(),
        &config(),
    )
    .unwrap();
    println!(
        "{name} stage=analytic Hessian complete paths={} residual={:.12e}",
        hessian.paths, hessian.residual
    );
    let mut pole_gradient = Vec::new();
    for axis in 0..12 {
        let mut direction = vec![[0.0; 3]; 4];
        direction[axis / 3][axis % 3] = 1.0;
        let path = spinor_excitation_path(
            &s.molecule,
            &s.basis,
            s.grid.as_ref(),
            &s.functional,
            &base.reference,
            &s.options,
            &base.operator,
            &direction,
            0,
            1,
            &tda(),
            &config(),
        )
        .unwrap();
        assert!((path.excitation[0] - states[0].energy).abs() < 1e-9);
        pole_gradient.push(path.excitation[1]);
    }
    let gradients = [pole_gradient, hessian.gradient.clone()];
    let matrices = [hessian.excitation_hessian, hessian.hessian];
    let mut names = Vec::new();
    let mut analytic = Vec::new();
    for c in 0..2 {
        for i in 0..12 {
            names.push(format!("{name} full spinor channel{c} gradient{i}"));
            analytic.push(gradients[c][i]);
        }
        for i in 0..12 {
            for j in 0..12 {
                names.push(format!("{name} full spinor channel{c} Hessian{i},{j}"));
                analytic.push(matrices[c][(i, j)]);
            }
        }
    }
    let zero = [
        states[0].energy,
        base.reference.free_energy + states[0].energy,
    ];
    let base_state = &states[0];
    let mut cache: HashMap<[u64; 12], [f64; 2]> = HashMap::new();
    let mut min_overlap = 1.0_f64;
    let mut min_gap = states[0].nearest_gap;
    let mut at = |displacement: [f64; 12]| {
        *cache
            .entry(displacement.map(f64::to_bits))
            .or_insert_with(|| {
                if displacement.iter().all(|v| *v == 0.0) {
                    return zero;
                }
                let mut shifted = system.clone();
                for (i, v) in shifted.positions.iter_mut().flatten().enumerate() {
                    *v += displacement[i];
                }
                let point = prepare(&shifted, name);
                let states = spectrum(&point);
                let scores = overlaps(&base, base_state, &point, &states);
                let selected = (0..scores.len())
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                    .unwrap();
                assert_eq!(selected, 0, "spinor state changed {scores:?}");
                assert!(scores[0] > 0.9, "spinor state overlap {scores:?}");
                min_overlap = min_overlap.min(scores[0]);
                min_gap = min_gap.min(states[0].nearest_gap);
                [
                    states[0].energy,
                    point.reference.free_energy + states[0].energy,
                ]
            })
    };
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.01, 0.005, 0.0025, 0.00125], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let mut errors = vec![[0.0; 4]; 312];
            for (k, &h) in steps.iter().enumerate() {
                for i in 0..12 {
                    let mut delta = [0.0; 12];
                    delta[i] = h;
                    let p = at(delta);
                    delta[i] = -h;
                    let m = at(delta);
                    let extra = if stencil == stencils::Stencil::Fourth {
                        delta[i] = 2.0 * h;
                        let p = at(delta);
                        delta[i] = -2.0 * h;
                        Some((p, at(delta)))
                    } else {
                        None
                    };
                    for c in 0..2 {
                        let (pp, mm) = extra.map_or((0.0, 0.0), |(p, m)| (p[c], m[c]));
                        let index = 156 * c + i;
                        errors[index][k] =
                            (stencil.first(p[c], m[c], pp, mm, h) - analytic[index]).abs();
                    }
                    for j in 0..12 {
                        let mut fd = |h: f64| {
                            if i == j {
                                let mut x = [0.0; 12];
                                x[i] = h;
                                let p = at(x);
                                x[i] = -h;
                                let m = at(x);
                                std::array::from_fn(|c| (p[c] + m[c] - 2.0 * zero[c]) / (h * h))
                            } else {
                                let mut x = [0.0; 12];
                                x[i] = h;
                                x[j] = h;
                                let pp = at(x);
                                x[j] = -h;
                                let pm = at(x);
                                x[i] = -h;
                                let mm = at(x);
                                x[j] = h;
                                let mp = at(x);
                                std::array::from_fn(|c| {
                                    (pp[c] - pm[c] - mp[c] + mm[c]) / (4.0 * h * h)
                                })
                            }
                        };
                        let first: [f64; 2] = fd(h);
                        let value = if stencil == stencils::Stencil::Fourth {
                            let second = fd(2.0 * h);
                            std::array::from_fn(|c| (4.0 * first[c] - second[c]) / 3.0)
                        } else {
                            first
                        };
                        for c in 0..2 {
                            let index = 156 * c + 12 + 12 * i + j;
                            errors[index][k] = (value[c] - analytic[index]).abs();
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "complete moving semilocal spinor excited tensors",
        &failures,
    );
    println!("{name} accepted=312 min_AO_overlap={min_overlap:.12e} min_gap={min_gap:.12e} static_points={}",cache.len());
}
#[test]
fn lda_all_molecular_spinor_excited_components() {
    complete("svwn5");
}

#[test]
fn gga_all_molecular_spinor_excited_components() {
    complete("pbe");
}
#[test]
fn meta_gga_all_molecular_spinor_excited_components() {
    complete("tpss");
}
#[test]
fn global_hybrid_all_molecular_spinor_excited_components() {
    complete("pbe0");
}
#[test]
fn range_separated_hybrid_all_molecular_spinor_excited_components() {
    complete("cam-b3lyp");
}

#[test]
fn gga_reference_kernel_preflight() {
    let point = prepare(&system(), "pbe");
    let states = spectrum(&point);
    assert_eq!(states.len(), 12);
    assert!(states.iter().all(|s| s.energy.is_finite()));
    assert!(states[0].energy > 0.0 && states[0].nearest_gap > tda().minimum_root_gap);
    println!("PBE reference/kernel preflight: poles={} first={:.12e} gap={:.12e} stationarity={:.12e}; no nuclear acceptance inferred",states.len(),states[0].energy,states[0].nearest_gap,point.reference.orbital_residual);
}
