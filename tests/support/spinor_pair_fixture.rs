// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared independent full-spin field-Hessian oracle for a C1 odd-electron molecule.
use super::{c1_geometry, components, spinor_operator};
use ::dft_rs;
#[path = "spinor_pt2_oracle.rs"]
mod spinor_pt2_oracle;
use dft_rs::integrals::london::ComplexMatrix as Cm;
use dft_rs::periodic::BlochMatrix;
use dft_rs::{basis::MolBasis, system::Molecule};
pub fn fit(
    label: &str,
    make_sources: impl Fn(&Molecule, &MolBasis) -> Vec<BlochMatrix>,
    intervals: &[[f64; 4]],
) {
    use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
    use dft_rs::driver::{Controls, System, Value};
    use dft_rs::periodic::{
        spinor_scf::{run_spinor_scf, SpinorScfConfig},
        BlochMatrix,
    };
    use dft_rs::response::nuclear::spinor_correlated_one_body_taylor;
    use dft_rs::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
    let system = System::molecule(
        vec![1; 4],
        vec![
            [0.17, -0.11, 0.23],
            [1.77, 0.18, -0.53],
            [-0.31, 1.74, -0.45],
            [-1.2, -1.31, -0.61],
        ],
    );
    c1_geometry::assert_c1(&system.numbers, &system.positions);
    let controls = Value::object()
        .with("xc", "hf")
        .with("basis", "sto-3g")
        .with("reference", "uks")
        .with("charge", 1i32)
        .with("multiplicity", 2usize)
        .with("dispersion", "none")
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-14)
        .with("conv_tol_grad", 1e-14)
        .with("conv_tol_dp", 1e-14)
        .with("max_iter", 500usize)
        .with("guess", "core");
    let method = MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let seed = run_molecular_reference(&setup).unwrap();
    let n = setup.basis.nao_sph;
    assert_eq!(n, 4);
    let operator = spinor_operator::source(n);
    let sources = make_sources(&setup.molecule, &setup.basis);
    let variables = sources.len();
    assert!(variables >= 2);
    let scf_config = SpinorScfConfig {
        energy_tolerance: 1e-14,
        density_tolerance: 1e-14,
        orbital_tolerance: 1e-14,
        max_iter: 500,
        ..Default::default()
    };
    let at = |op: &BlochMatrix| {
        let backend = MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &seed.scf,
            &setup.options,
            op.clone(),
            [0.0, 0.0, 1.0],
            MolecularSpinorFunctionalModel::SpinCovariant,
            1e-12,
        )
        .unwrap();
        let r = run_spinor_scf(&backend, &scf_config).unwrap();
        assert!(r.converged);
        r
    };
    let reference = at(&operator);
    let eri = dft_rs::integrals::eri::eri_incore(
        &setup.basis,
        dft_rs::integrals::CoulombKernel::Coulomb,
        1024,
    )
    .unwrap();
    let independent = |op: &BlochMatrix| {
        let r = at(op);
        let c = Cm {
            re: r.bands[0].coefficients.re.clone(),
            im: r.bands[0].coefficients.im.clone(),
        };
        let e = &r.bands[0].energies;
        let correlation = spinor_pt2_oracle::stationary_pt2(&c, e, 3, &eri);
        [
            r.fields.energy,
            correlation,
            r.fields.energy + 0.27 * correlation,
        ]
    };
    let config = dft_rs::response::ResponseSolverConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let path = |direction: Vec<f64>| {
        let mut source = BlochMatrix::zeros(2 * n, 2 * n);
        for a in 0..variables {
            source.add_scaled(&sources[a], direction[a]);
        }
        let value = spinor_correlated_one_body_taylor(
            &setup.molecule,
            &setup.basis,
            setup.grid.as_ref(),
            &setup.functional,
            &reference,
            &setup.options,
            &operator,
            &[
                Cm::zeros(2 * n, 2 * n),
                Cm {
                    re: source.re,
                    im: source.im,
                },
                Cm::zeros(2 * n, 2 * n),
            ],
            0.27,
            &config,
        )
        .unwrap();
        let physical = value
            .parent
            .parent
            .stationarity_defects
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let pair = value
            .pairs
            .residuals
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let number = value
            .parent
            .particle_number_defects
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        println!("C1 open-shell PT2 path direction={direction:?} stationarity={physical:.12e} pair={pair:.12e} frame={:.12e} particle_number={number:.12e}",value.parent.frame_defect);
        assert!(
            physical <= config.acceptance()
                && pair <= config.acceptance()
                && number <= config.acceptance()
                && value.parent.frame_defect <= config.acceptance()
        );
        value
    };
    let axes = (0..variables)
        .map(|a| {
            path(
                (0..variables)
                    .map(|b| if a == b { 1.0 } else { 0.0 })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let channel = |v: &dft_rs::response::nuclear::SpinorCorrelatedPath, n: usize| {
        [v.parent.parent.energy[n], v.pairs.energy[n].re, v.energy[n]]
    };
    let mut exact = vec![0.0; 3 * variables * variables];
    for a in 0..variables {
        for b in 0..variables {
            let value = if a == b {
                channel(&axes[a], 2).map(|v| 2.0 * v)
            } else {
                let d = path(
                    (0..variables)
                        .map(|k| if k == a || k == b { 1.0 } else { 0.0 })
                        .collect(),
                );
                let d = channel(&d, 2);
                let x = channel(&axes[a], 2);
                let y = channel(&axes[b], 2);
                std::array::from_fn(|s| d[s] - x[s] - y[s])
            };
            for s in 0..3 {
                exact[variables * variables * s + variables * a + b] = value[s];
            }
        }
    }
    let centre = independent(&operator);
    let base = channel(&axes[0], 0);
    for s in 0..3 {
        assert!((base[s] - centre[s]).abs() < 2e-12);
    }
    let names = (0..3 * variables * variables)
        .map(|p| {
            format!(
                "{label} channel={} ({},{})",
                p / (variables * variables),
                p / variables % variables,
                p % variables
            )
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    components::fit_components(
        &names,
        intervals,
        |steps| {
            let mut errors = vec![[0.0; 4]; 3 * variables * variables];
            for (k, &h) in steps.iter().enumerate() {
                for a in 0..variables {
                    for b in 0..variables {
                        let displaced = |x: f64, y: f64| {
                            let mut op = operator.clone();
                            op.add_scaled(&sources[a], x);
                            op.add_scaled(&sources[b], y);
                            independent(&op)
                        };
                        let measured: [f64; 3] = if a == b {
                            let plus = displaced(h, 0.0);
                            let minus = displaced(-h, 0.0);
                            std::array::from_fn(|s| {
                                (plus[s] - 2.0 * centre[s] + minus[s]) / (h * h)
                            })
                        } else {
                            let pp = displaced(h, h);
                            let pm = displaced(h, -h);
                            let mp = displaced(-h, h);
                            let mm = displaced(-h, -h);
                            std::array::from_fn(|s| (pp[s] - pm[s] - mp[s] + mm[s]) / (4.0 * h * h))
                        };
                        for s in 0..3 {
                            let p = variables * variables * s + variables * a + b;
                            errors[p][k] = (measured[s] - exact[p]).abs();
                        }
                    }
                }
            }
            errors
        },
        &mut failures,
    );
    components::assert_no_failures(
        "complete transverse parent/PT2/total field Hessians",
        &failures,
    );
}
