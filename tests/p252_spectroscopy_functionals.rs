// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Shared RKS/UKS spectroscopy providers. The nuclear projector oracle fits
//! every electronic VCD axial-tensor element, without symmetry copying.
//! Broadened ECD/NMR line models are compared with their unbroadened inputs;
//! these checks do not certify correlated PT2 transition or mixed kernels.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::basis::MolBasis;
use dft_rs::driver::method::{
    molecule_from, prepare_molecular, run_molecular_reference, MolecularMethod,
};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::one_electron::overlap_matrix;
use dft_rs::properties::magnetic::{magnetic_density_response, Gauge};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::skala::{random_tensors, write_tensors, SkalaSizes};
use dft_rs::{Matrix, Vec3};

fn array(x: &[f64]) -> Value {
    Value::Array(x.iter().map(|&v| Value::Float(v)).collect())
}
fn flat(v: &Value, key: &str) -> Vec<f64> {
    v.get(key).unwrap().numeric_array().unwrap().1
}
fn request(xc: &str, u: bool, model: &std::path::Path) -> Value {
    let mut v = Value::object()
        .with("xc", xc)
        .with("basis", "sto-3g")
        .with("reference", if u { "uks" } else { "rks" })
        .with("charge", if u { 1i32 } else { 0i32 })
        .with("multiplicity", if u { 2usize } else { 1usize })
        .with("grid_level", 0usize)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-13)
        .with("conv_tol_dp", 1e-13)
        .with("conv_tol_grad", 1e-13)
        .with("response_tolerance", 1e-11);
    if xc == "skala" {
        v.set("skala_model", model.to_str().unwrap());
    }
    v
}
fn calculate(id: &str, system: &System, v: &Value) -> Value {
    run(id, system, &Controls::from_value(v).unwrap()).unwrap_or_else(|e| panic!("{id}: {e}"))
}
#[test]
fn collinear_functional_spectra_and_every_vcd_axial_element() {
    let system = System::molecule(
        vec![7, 1, 1, 1],
        vec![
            [0.13, -0.21, 0.19],
            [1.57, 0.16, -0.08],
            [-0.19, 1.93, 0.37],
            [0.42, 0.29, 2.28],
        ],
    );
    let path =
        std::env::temp_dir().join(format!("dft_rs_spectroscopy_{}.model", std::process::id()));
    write_tensors(
        &path,
        &random_tensors(
            7,
            SkalaSizes {
                width: 6,
                channels: 2,
                shells: 1,
                correlation: 2,
                layers: 1,
                post: 2,
            },
        ),
    )
    .unwrap();
    // Optional fixture export is explicit; normal test runs keep no files.
    if let Some(export) = std::env::var_os("DFT_RS_SPECTROSCOPY_MODEL_EXPORT") {
        std::fs::copy(&path, std::path::PathBuf::from(export)).unwrap();
    }
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    for xc in ["hf", "svwn5", "pbe", "tpss", "pbe0", "skala"] {
        for u in [false, true] {
            if std::env::var("DFT_RS_TEST_MODEL")
                .ok()
                .is_some_and(|v| v != xc)
            {
                continue;
            }
            let controls = request(xc, u, &path);
            let mut ecd_controls = controls.clone();
            if let Value::Object(fields) = &mut ecd_controls {
                fields.retain(|(key, _)| key != "response_tolerance");
            }
            ecd_controls.set("nstates", 3usize);
            ecd_controls.set("broadening_ev", 0.2);
            ecd_controls.set("points", 301usize);
            let ecd = calculate("ecd-spectrum", &system, &ecd_controls);
            for key in ["ecd_length", "ecd_velocity", "molar_absorptivity"] {
                let values = flat(&ecd, key);
                assert_eq!(values.len(), 301);
                assert!(values.iter().all(|v| v.is_finite()));
            }
            let mut nmr_controls = controls.clone();
            nmr_controls.set("nmr_reference_elements", vec![1usize, 7usize]);
            nmr_controls.set("nmr_reference_shieldings", array(&[30.0, 350.0]));
            nmr_controls.set("nmr_points", 201usize);
            let nmr = calculate("nmr-spectrum", &system, &nmr_controls);
            let shifts = flat(&nmr, "chemical_shifts");
            let isotropic = flat(&nmr, "isotropic_shieldings");
            for i in 0..4 {
                let standard = if system.numbers[i] == 1 { 30.0 } else { 350.0 };
                assert!((shifts[i] - (standard - isotropic[i])).abs() < 1e-10);
            }
            let axes = nmr.get("spectrum_axis").unwrap().as_array().unwrap();
            let lines = nmr.get("spectrum_intensity").unwrap().as_array().unwrap();
            let mut line_defect = 0.0_f64;
            for (element, &z) in [1u8, 7].iter().enumerate() {
                let axis = axes[element].numeric_array().unwrap().1;
                let intensity = lines[element].numeric_array().unwrap().1;
                for (&x, &actual) in axis.iter().zip(&intensity) {
                    let expected: f64 = (0..4)
                        .filter(|&i| system.numbers[i] == z)
                        .map(|i| 0.01 / (std::f64::consts::PI * ((x - shifts[i]).powi(2) + 0.0001)))
                        .sum();
                    line_defect = line_defect.max((actual - expected).abs());
                }
            }
            println!("{xc} U={u} NMR line-model defect {line_defect:.12e}");
            assert!(line_defect < 1e-10);
            let tensors = flat(&nmr, "shielding_tensors");
            assert_eq!(tensors.len(), 36);
            nmr_controls.set("gauge_origin", array(&[0.47, -0.31, 0.82]));
            let moved = calculate("nmr-spectrum", &system, &nmr_controls);
            let gauge_defect = tensors
                .iter()
                .zip(flat(&moved, "shielding_tensors"))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            println!("{xc} U={u} complete shielding gauge defect {gauge_defect:.12e} ppm");
            assert!(gauge_defect < 1e-7);
            let origin = [0.3, -0.2, 0.1];
            let mut vcd_controls = controls.clone();
            vcd_controls.set("gauge", "common-origin");
            vcd_controls.set("gauge_origin", array(&origin));
            let vcd = calculate("vcd", &system, &vcd_controls);
            let electronic = flat(&vcd, "atomic_axial_tensors_electronic");
            assert_eq!(electronic.len(), 36);
            let method =
                MolecularMethod::from_controls(&Controls::from_value(&controls).unwrap()).unwrap();
            let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
            let reference = run_molecular_reference(&setup).unwrap();
            let scf = &reference.scf;
            let n = scf.overlap.rows;
            let response = magnetic_density_response(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
                Gauge::CommonOrigin,
                Vec3::new(origin[0], origin[1], origin[2]),
                &ResponseSolverConfig {
                    tolerance: 1e-11,
                    ..Default::default()
                },
            )
            .unwrap();
            println!("{xc} U={u} magnetic residuals {:?}", response.residuals);
            let amplitudes: Vec<Vec<Matrix>> = (0..2)
                .map(|s| {
                    let sc = scf.overlap.matmul(&scf.mo_coeff[s]);
                    (0..3)
                        .map(|b| {
                            sc.transpose_matmul(&response.value.density_hat[s][b])
                                .matmul(&sc)
                        })
                        .collect()
                })
                .collect();
            let projector = |displaced: &System| -> [Matrix; 2] {
                let shifted =
                    prepare_molecular(&displaced.numbers, &displaced.positions, &method).unwrap();
                let shifted = run_molecular_reference(&shifted).unwrap().scf;
                let mut numbers = system.numbers.clone();
                numbers.extend(&displaced.numbers);
                let mut positions = system.positions.clone();
                positions.extend(&displaced.positions);
                let both = molecule_from(&numbers, &positions, 0, 1).unwrap();
                let basis = MolBasis::build(&both, &setup.basis_set).unwrap();
                let overlap = overlap_matrix(&basis);
                let mut cross = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        cross[(i, j)] = overlap[(i, n + j)];
                    }
                }
                std::array::from_fn(|s| {
                    let left = scf.mo_coeff[s].transpose_matmul(&cross);
                    left.matmul(&shifted.density[s]).matmul_transpose_seq(&left)
                })
            };
            let stages = [
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.004, 0.003, 0.002, 0.0015], stencils::Stencil::Second),
                ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
                ([0.024, 0.018, 0.012, 0.009], stencils::Stencil::Second),
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
            ];
            let mut failures = Vec::new();
            for x in 0..12 {
                let names: Vec<_> = (0..3)
                    .map(|b| format!("{xc} U={u} electronic AAT({x},{b})"))
                    .collect();
                stencils::fit_components_staged(
                    &names,
                    &stages,
                    |steps, stencil| {
                        let mut errors = vec![[0.0; 4]; 3];
                        for (k, &h) in steps.iter().enumerate() {
                            let sample = |sign: f64| {
                                let mut moved = system.clone();
                                moved.positions[x / 3][x % 3] += sign * h;
                                projector(&moved)
                            };
                            let (plus, minus) = (sample(1.0), sample(-1.0));
                            let doubled = if stencil == stencils::Stencil::Fourth {
                                Some((sample(2.0), sample(-2.0)))
                            } else {
                                None
                            };
                            for b in 0..3 {
                                let mut value = 0.0;
                                for s in 0..2 {
                                    let no = if s == 0 { scf.n_alpha } else { scf.n_beta };
                                    for a in no..n {
                                        for i in 0..no {
                                            let (plus2, minus2) =
                                                doubled.as_ref().map_or((0.0, 0.0), |(p, m)| {
                                                    (p[s][(a, i)], m[s][(a, i)])
                                                });
                                            value += amplitudes[s][b][(a, i)]
                                                * stencil.first(
                                                    plus[s][(a, i)],
                                                    minus[s][(a, i)],
                                                    plus2,
                                                    minus2,
                                                    h,
                                                );
                                        }
                                    }
                                }
                                errors[b][k] = (value - electronic[3 * x + b]).abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
            components::assert_no_failures(
                &format!("{xc} U={u} complete electronic axial tensor"),
                &failures,
            );
        }
    }
}
